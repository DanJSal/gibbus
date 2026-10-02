"""Independent parity/oracle tests for compiled numerical hot paths."""

from concurrent.futures import ThreadPoolExecutor
from math import factorial

import numpy as np
import pytest
from gibbus._model._moment_kernels import product_moment
from gibbus._postfit._mix_kernels import neg_log_mix_derivs_batch, neg_logsumexp_batch
from gibbus._spectral._tail_integrals import TailIntegrator
from numpy.polynomial.polynomial import polymul
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import beta, gamma, norm

from gibbus import Distribution


def _python_neg_log_mix_derivs(ell_jets, max_order):
    """Reference Taylor recurrence written only with NumPy/SciPy."""
    jets = np.asarray(ell_jets, dtype=np.float64)
    order = int(max_order)
    fact = np.array([factorial(m) for m in range(order + 1)], dtype=np.float64)
    ell_tc = jets / fact[None, :, None]
    ell0 = ell_tc[:, 0, :]
    offset = logsumexp(ell0, axis=0)

    a = np.zeros_like(ell_tc)
    a[:, 0, :] = np.exp(ell0 - offset[None, :])
    for n in range(1, order + 1):
        for m in range(1, n + 1):
            a[:, n, :] += m * ell_tc[:, m, :] * a[:, n - m, :]
        a[:, n, :] /= n

    summed = a.sum(axis=0)
    h = np.zeros((order + 1, jets.shape[2]), dtype=np.float64)
    h[0] = np.log(summed[0]) + offset
    for n in range(1, order + 1):
        acc = np.zeros(jets.shape[2], dtype=np.float64)
        for m in range(1, n):
            acc += m * h[m] * summed[n - m]
        h[n] = (summed[n] - acc / n) / summed[0]
    return -(h * fact[:, None])


def test_fused_product_moment_matches_polynomial_convolution():
    rng = np.random.default_rng(1200)
    for na, nb in ((1, 1), (3, 4), (7, 6)):
        a = np.ascontiguousarray(rng.normal(size=na))
        b = np.ascontiguousarray(rng.normal(size=nb))
        moments = np.ascontiguousarray(rng.normal(size=na + nb + 3))
        expected = float(np.dot(polymul(a, b), moments[: na + nb - 1]))
        assert product_moment(a, b, moments) == pytest.approx(
            expected, rel=3e-15, abs=3e-15
        )


def test_compiled_mixture_order_zero_matches_scipy_logsumexp():
    rng = np.random.default_rng(1201)
    ell0 = np.ascontiguousarray(rng.normal(size=(5, 19)) * 30.0)
    expected = -logsumexp(ell0, axis=0)
    got = neg_logsumexp_batch(ell0)
    np.testing.assert_allclose(got, expected, rtol=3e-15, atol=3e-15)


def test_compiled_mixture_derivative_recurrence_matches_python_reference():
    rng = np.random.default_rng(1202)
    jets = np.ascontiguousarray(rng.normal(size=(4, 6, 13)))
    jets[:, 0, :] *= 20.0
    expected = _python_neg_log_mix_derivs(jets, 5)
    got = neg_log_mix_derivs_batch(jets, 5)
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("upper", [False, True])
def test_compiled_infinite_tail_matches_normal_oracle(upper):
    q_poly = np.array([0.5 * np.log(2.0 * np.pi), 0.0, 0.5])
    integrator = TailIntegrator(q_poly)
    p = 1e-30
    x = float(norm.isf(p) if upper else norm.ppf(p))
    endpoint = np.inf if upper else -np.inf
    got, message = integrator.log_mass(
        x,
        endpoint,
        upper,
        np.array([-np.inf, np.inf]),
        np.array([np.nan, np.nan]),
        0.0,
        1.0,
    )
    assert message is None
    assert got == pytest.approx(np.log(p), abs=3e-12)


@pytest.mark.parametrize("upper", [False, True])
def test_batched_tail_masses_match_normal_oracle(upper):
    q_poly = np.array([0.5 * np.log(2.0 * np.pi), 0.0, 0.5])
    integrator = TailIntegrator(q_poly)
    p = np.logspace(-3.0, -60.0, 40)
    x = norm.isf(p) if upper else norm.ppf(p)
    values, failed = integrator.log_masses(
        x,
        np.inf if upper else -np.inf,
        upper,
        np.array([-np.inf, np.inf]),
        np.array([np.nan, np.nan]),
        0.0,
        1.0,
    )
    assert failed == 0
    np.testing.assert_allclose(values, np.log(p), rtol=0.0, atol=5e-11)


def test_batched_tail_pieces_are_not_held_below_the_rounding_noise_of_q():
    """Far out on a steep tail the integrand's accuracy is set by ``q``'s rounding.

    Regression: with ``q`` of order 1e3 to 1e5 (a fitted degree-6 component of
    a lognormal mixture) a fixed ``1e-13`` piece tolerance sat below that noise
    and every piece exhausted its subdivision limit.
    """
    q_poly = np.array(
        [
            4.0280767819891716,
            2.2434561878046151,
            -0.18568696007269678,
            -0.033247451223551647,
            0.033784237598089836,
            -0.0068636016207788021,
            0.00044544952662950470,
        ]
    )
    support = np.array([0.0, np.inf])
    amplitudes = np.array([3.9845637464359442, 0.0])
    mu_eff, sigma_eff = -2.1047013926129257, 2.4638996655463976
    integrator = TailIntegrator(q_poly)
    grid = np.linspace(7.0, 11.8, 105)
    values, failed = integrator.log_masses(
        grid, np.inf, True, support, amplitudes, mu_eff, sigma_eff
    )
    assert failed == 0
    reference = []
    for x in grid:
        value, message = integrator.log_mass(
            x, np.inf, True, support, amplitudes, mu_eff, sigma_eff
        )
        assert message is None
        reference.append(value)
    np.testing.assert_allclose(values, reference, rtol=1e-13, atol=0.0)


def test_reusable_tail_context_is_thread_safe():
    q_poly = np.array([0.5 * np.log(2.0 * np.pi), 0.0, 0.5])
    integrator = TailIntegrator(q_poly)
    support = np.array([-np.inf, np.inf])
    amplitudes = np.array([np.nan, np.nan])
    probabilities = [1e-8, 1e-16, 1e-30, 1e-50] * 6

    def evaluate(item):
        idx, p = item
        upper = bool(idx % 2)
        x = float(norm.isf(p) if upper else norm.ppf(p))
        endpoint = np.inf if upper else -np.inf
        value, message = integrator.log_mass(
            x, endpoint, upper, support, amplitudes, 0.0, 1.0
        )
        return p, value, message

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, enumerate(probabilities)))
    for p, value, message in results:
        assert message is None
        assert value == pytest.approx(np.log(p), abs=5e-11)


def test_compiled_tail_respects_public_affine_transform():
    q_poly = np.array([0.5 * np.log(2.0 * np.pi), 0.0, 0.5])
    integrator = TailIntegrator(q_poly)
    mu, sigma = 1.0e6, 2.75
    p = 1e-20
    x = float(norm(loc=mu, scale=sigma).isf(p))
    got, message = integrator.log_mass(
        x,
        np.inf,
        True,
        np.array([-np.inf, np.inf]),
        np.array([np.nan, np.nan]),
        -mu / sigma,
        1.0 / sigma,
    )
    assert message is None
    assert got == pytest.approx(np.log(p), abs=2e-10)


@pytest.mark.parametrize("upper", [False, True])
def test_compiled_finite_boundary_tail_matches_beta_oracle(upper):
    a, b = 2.5, 4.25
    q_poly = np.array([betaln(a, b)])
    integrator = TailIntegrator(q_poly)
    dist = beta(a, b)
    p = 1e-14
    x = float(dist.isf(p) if upper else dist.ppf(p))
    endpoint = 1.0 if upper else 0.0
    got, message = integrator.log_mass(
        x,
        endpoint,
        upper,
        np.array([0.0, 1.0]),
        np.array([a - 1.0, b - 1.0]),
        0.0,
        1.0,
    )
    assert message is None
    assert got == pytest.approx(np.log(p), abs=3e-10)


def test_compiled_tail_handles_reflected_upper_boundary_geometry():
    shape = 2.75
    boundary_power = shape - 1.0
    q_poly = np.array([gammaln(shape), 1.0])
    integrator = TailIntegrator(q_poly)
    p = 1e-14
    z = float(gamma(shape).ppf(p))
    x = -z
    got, message = integrator.log_mass(
        x,
        0.0,
        True,
        np.array([-np.inf, 0.0]),
        np.array([np.nan, boundary_power]),
        0.0,
        -1.0,
    )
    assert message is None
    assert got == pytest.approx(np.log(p), abs=5e-10)


def test_public_mixture_tail_is_weighted_component_tail_identity():
    rng = np.random.default_rng(1203)
    data = np.concatenate(
        [
            rng.normal(-2.0, 0.55, 180),
            rng.normal(2.0, 0.65, 180),
        ]
    )
    model = Distribution().fit(
        data, n_components=2, poly_degree=2, support=(-np.inf, np.inf), rng=0
    )
    x = float(max(component.base.ppf(1.0 - 1e-7) for component in model.components))
    component_logs = np.array(
        [component.base.logsf(x) for component in model.components]
    )
    expected = float(logsumexp(np.log(model.weights) + component_logs))
    assert model.logsf(x) == pytest.approx(expected, rel=0.0, abs=2e-10)


def _natural_state(support, x, degree, lower, upper, params, /):
    from gibbus._model.coords import _build_fit_coordinate
    from gibbus._model.natural_state import _NaturalCoreState
    from gibbus._model.spec import _build_model_spec

    coord = _build_fit_coordinate(support, x, None, None)
    z = coord.to_canonical(x)
    spec = _build_model_spec(coord, degree, lower, upper)
    state = _NaturalCoreState(
        coord,
        spec.layout,
        np.asarray(params, dtype=np.float64),
        (float(z.min()), float(z.max())),
    )
    return coord, spec, state, z


def test_compiled_adaptive_interval_reduction_matches_scipy_quad():
    from scipy.integrate import quad

    from gibbus._observations.intervals import _prepare_partial_interval_reducer

    _, spec, state, z = _natural_state(
        (0.0, np.inf),
        np.array([0.25, 0.7, 1.3, 2.4, 4.8]),
        3,
        True,
        False,
        [0.3, 0.9, 0.25, 0.35],
    )
    prepared = _prepare_partial_interval_reducer(state)

    def statistic(value, i):
        return float(state.partials[i].evaluate(value, spec.support))

    def integral(f, lo, hi):
        return quad(
            lambda v: f(v) * state.pdf(v), lo, hi, epsabs=0.0, epsrel=1e-12, limit=400
        )[0]

    n = len(state.partials)
    cut = float(np.median(z))
    for lo, hi in ((float(spec.support[0]), cut), (cut, np.inf)):
        mass = integral(lambda v: 1.0, lo, hi)
        mean = np.array(
            [integral(lambda v, i=i: statistic(v, i), lo, hi) for i in range(n)]
        )
        mean /= mass
        second = (
            np.array(
                [
                    [
                        integral(
                            lambda v, i=i, j=j: statistic(v, i) * statistic(v, j),
                            lo,
                            hi,
                        )
                        for j in range(n)
                    ]
                    for i in range(n)
                ]
            )
            / mass
        )
        got = prepared.reduce((lo, hi))
        assert got.log_probability == pytest.approx(np.log(mass), rel=2e-10, abs=2e-11)
        np.testing.assert_allclose(got.mean, mean, rtol=2e-9, atol=2e-10)
        np.testing.assert_allclose(
            got.covariance, second - np.outer(mean, mean), rtol=3e-8, atol=3e-10
        )


@pytest.mark.parametrize(
    "interval",
    [
        (3.0, 4.0),
        (30.0, 40.0),
        (1000.0, 2000.0),
        (-2000.0, -1000.0),
        (1000.0, np.inf),
        (-np.inf, -1000.0),
        (10.0, 1e4),
    ],
)
def test_adaptive_interval_reduction_resolves_steep_far_tails(interval):
    """A standard normal far in its tail: the peak sits at an endpoint and the
    potential rises past the exponent range before the first quadrature node."""
    from gibbus._observations.intervals import _prepare_partial_interval_reducer

    x = np.random.default_rng(0).normal(size=200)
    _, _, state, _ = _natural_state((-np.inf, np.inf), x, 2, False, False, [0.0, 1.0])
    np.testing.assert_allclose(state.q_poly, [0.0, 0.0, 0.5], atol=1e-15)
    got = _prepare_partial_interval_reducer(state).reduce(interval)

    lo, hi = interval
    if lo > 0.0:
        log_mass = float(
            norm.logsf(lo) + np.log(-np.expm1(norm.logsf(hi) - norm.logsf(lo)))
        )
    else:
        log_mass = float(
            norm.logcdf(hi) + np.log(-np.expm1(norm.logcdf(lo) - norm.logcdf(hi)))
        )
    mean = np.exp(norm.logpdf(lo) - log_mass) - np.exp(norm.logpdf(hi) - log_mass)
    assert got.log_probability == pytest.approx(log_mass, rel=1e-13, abs=1e-9)
    assert got.mean[0] == pytest.approx(mean, rel=1e-9)


def test_batched_adaptive_interval_reduction_matches_scalar_and_weighted_sums():
    from gibbus._observations.intervals import _prepare_partial_interval_reducer

    _, spec, state, z = _natural_state(
        (0.0, np.inf),
        np.array([0.2, 0.55, 1.1, 2.0, 3.7, 5.2]),
        3,
        True,
        False,
        [0.2, 0.82, 0.18, 0.31],
    )
    prepared = _prepare_partial_interval_reducer(state)
    cuts = np.quantile(z, [0.25, 0.5, 0.75])
    intervals = np.ascontiguousarray(
        [
            [float(spec.support[0]), float(cuts[0])],
            [float(cuts[0]), np.inf],
            [float(cuts[1]), np.inf],
            [float(cuts[2]), np.inf],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    scalar = [prepared.reduce(row) for row in intervals]
    logp, mean, cov, _ = prepared.reduce_many(intervals)
    np.testing.assert_allclose(
        logp, [row.log_probability for row in scalar], rtol=0.0, atol=2e-13
    )
    np.testing.assert_allclose(
        mean, np.stack([row.mean for row in scalar]), rtol=2e-12, atol=2e-13
    )
    np.testing.assert_allclose(
        cov, np.stack([row.covariance for row in scalar]), rtol=2e-11, atol=2e-13
    )

    w_logp, w_mean, w_cov, _ = prepared.reduce_weighted(intervals, weights)
    np.testing.assert_allclose(w_logp, logp, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(w_mean, weights @ mean, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(
        w_cov, np.einsum("r,rij->ij", weights, cov), rtol=2e-11, atol=2e-13
    )


def test_fused_finite_interval_objective_matches_the_quadrature_plan():
    """The one-pass finite-row kernel reproduces the explicit Gauss--Legendre
    reduction of the quadrature plan."""
    from gibbus._observations._finite_reductions import evaluate_finite_objective

    from gibbus._defaults import INTERVAL_W_EPS_MULT
    from gibbus._fit.objective import _finite_interval_log_kernel
    from gibbus._model.natural_state import _layout_numerics
    from gibbus._model.vec import _q_eval
    from gibbus._observations.intervals import (
        _GL_LOG_W,
        _GL_X,
        _build_finite_interval_quadrature,
    )

    coord, spec, state, _ = _natural_state(
        (0.0, 1.0),
        np.array([0.08, 0.2, 0.42, 0.63, 0.82, 0.95]),
        4,
        True,
        True,
        [0.08, 0.85, 0.18, -0.12, 0.35, 0.28],
    )
    user_rows = np.array(
        [
            [0.11, 0.27],
            [0.31, 0.74],  # contains the mode for this fixture in canonical space
            [0.79, 0.91],
            [0.57, 0.57],  # exact-point convention
        ],
        dtype=np.float64,
    )
    intervals = np.ascontiguousarray(
        np.column_stack(
            [
                coord.to_canonical(user_rows[:, 0]),
                coord.to_canonical(user_rows[:, 1]),
            ]
        ),
        dtype=np.float64,
    )
    row_weights = np.array([0.15, 0.35, 0.3, 0.2], dtype=np.float64)
    point_lower = np.full(intervals.shape[0], np.nan, dtype=np.float64)
    point_upper = np.full(intervals.shape[0], np.nan, dtype=np.float64)
    numerics = _layout_numerics(state.layout)

    plan = _build_finite_interval_quadrature(intervals, state.mode)
    nodes, log_kernel, point_mid, log_integrals = _finite_interval_log_kernel(
        state,
        spec.support,
        plan,
        lambda zz: _q_eval(
            zz, spec.support, state.q_poly, state.boundary_amplitudes, 0
        ),
    )

    def statistics(values):
        return np.stack(
            [p.evaluate(values, spec.support) for p in state.partials], axis=-1
        )

    n = len(state.partials)
    expected_h = np.zeros(n)
    expected_cov = np.zeros((n, n))
    point_index = 0
    for r in range(intervals.shape[0]):
        if plan.point_limit[r]:
            expected_h += (
                row_weights[r] * statistics(np.array([point_mid[point_index]]))[0]
            )
            point_index += 1
            continue
        alpha = np.exp(log_kernel[r] + plan.log_weights[r] - log_integrals[r])
        t = statistics(nodes[r])
        mean = alpha @ t
        expected_h += row_weights[r] * mean
        expected_cov += row_weights[r] * (
            (t * alpha[:, None]).T @ t - np.outer(mean, mean)
        )
    expected_logp = np.asarray(log_integrals - np.log(state.Z), dtype=np.float64)
    expected_logp[plan.widths == 0.0] -= np.log(spec.coordinate.scale)

    got_logp, got_h, got_cov = evaluate_finite_objective(
        intervals,
        row_weights,
        point_lower,
        point_upper,
        state.q_poly,
        state.boundary_amplitudes,
        float(state.q_shift),
        float(np.log(state.Z)),
        float(state.mode),
        float(spec.coordinate.scale),
        numerics.kinds,
        numerics.lengths,
        numerics.coefficients,
        float(spec.support[0]),
        float(spec.support[1]),
        _GL_X,
        _GL_LOG_W,
        float(INTERVAL_W_EPS_MULT),
    )

    np.testing.assert_allclose(got_logp, expected_logp, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(got_h, expected_h, rtol=4e-14, atol=4e-14)
    np.testing.assert_allclose(got_cov, expected_cov, rtol=2e-12, atol=2e-13)
