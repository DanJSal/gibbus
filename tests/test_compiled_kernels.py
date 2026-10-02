"""Compiled kernels against independent checks and test-only construction harnesses."""

import numpy as np
import pytest
from conic_reference import _preconditioned_subproblem_reference, _solve_conic_newton_qp
from spectral_builder_harness import PythonSpectralCDFBuilder, PythonSpectralPPFBuilder

from gibbus._fit import _conic_kernels, _curvature_certificate
from gibbus._fit.conic_newton import _preconditioned_subproblem
from gibbus._fit.conic_qp import _support_representation
from gibbus._fit.separation import _separate_full_curvature
from gibbus._model.natural import _natural_layout

_GEOMETRIES = [
    ((-np.inf, np.inf), False, False),
    ((-1.3, np.inf), True, False),
    ((-1.3, np.inf), False, False),
    ((-np.inf, 0.7), False, True),
    ((-1.3, 1.7), True, True),
    ((-1.3, 1.7), False, False),
    ((-1.3, 1.7), True, False),
]


def _compiled_qp(hessian, gradient, theta0, representation, blocks, /):
    a_packed, sizes, a_offsets, q_offsets = representation.packed
    return _conic_kernels.solve_conic_qp(
        hessian,
        gradient,
        theta0,
        representation.b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        representation.reference_dual,
        representation.pack_blocks(blocks),
        1e-12,
        100,
        0.99,
    )


def _random_model(rng, layout, representation, /):
    n = layout.n_params
    a = rng.normal(size=(n, n))
    hessian = a @ a.T / n + 10.0 ** rng.uniform(-6, 0) * np.eye(n)
    blocks = [
        np.eye(m.shape[1]) * 10.0 ** rng.uniform(-2, 1)
        for m in representation.row_matrices
    ]
    theta0 = representation.reconstruct(np.zeros(n), blocks)
    target = rng.normal(size=n) * 10.0 ** rng.uniform(-1, 1)
    gradient = hessian @ (theta0 - target)
    return hessian, gradient, theta0, blocks


def _assert_psd_blocks(blocks, /):
    for block in blocks:
        scale = max(1.0, np.abs(block).max())
        assert np.linalg.eigvalsh(block).min() >= -1e-12 * scale


@pytest.mark.parametrize(("support", "lower", "upper"), _GEOMETRIES)
@pytest.mark.parametrize("degree", [4, 6, 8])
def test_compiled_qp_is_consistent_with_the_python_reference(support, lower, upper, degree):
    layout = _natural_layout(support, degree, lower, upper)
    representation = _support_representation(layout)
    rng = np.random.default_rng(degree + 17 * int(lower) + 31 * int(upper))
    for _ in range(4):
        hessian, gradient, theta0, blocks = _random_model(rng, layout, representation)
        reference = _solve_conic_newton_qp(
            hessian, gradient, theta0, representation, start_blocks=blocks
        )
        params, packed, _, model, gap, _ = _compiled_qp(
            hessian, gradient, theta0, representation, blocks
        )
        scale = max(1.0, abs(reference.model_value))
        # Each model value is a primal value within its certified gap of the
        # common optimum, so they differ by at most the larger gap.
        assert abs(model - reference.model_value) <= max(gap, reference.gap) + 1e-12 * scale
        assert gap <= 1e-6 * scale
        _assert_psd_blocks(representation.unpack_blocks(packed))
        assert np.all(np.isfinite(params))


def test_compiled_preconditioned_solve_matches_the_reference():
    # Random models with random-scale starts are much harder than the
    # subproblems of a fit; neither solver certifies all of them to 1e-12, so
    # the check is consistency of the certified values plus convergence at
    # least as often as the reference.
    converged = [0, 0]
    for support, lower, upper in _GEOMETRIES:
        for degree in (4, 6, 8, 10):
            layout = _natural_layout(support, degree, lower, upper)
            representation = _support_representation(layout)
            rng = np.random.default_rng(3 * degree + 5 * int(lower) + 7 * int(upper))
            for _ in range(4):
                hessian, gradient, theta0, blocks = _random_model(rng, layout, representation)
                compiled, _, blocks_c, model_c = _preconditioned_subproblem(
                    hessian, gradient, theta0, representation, blocks
                )
                reference, _, _, model_r = _preconditioned_subproblem_reference(
                    hessian, gradient, theta0, representation, blocks
                )
                converged[0] += compiled.converged
                converged[1] += reference.converged
                scale = max(1.0, abs(model_r))
                assert compiled.gap <= 1e-5 * scale
                tolerance = max(compiled.gap, reference.gap) + 1e-11 * scale
                assert abs(model_c - model_r) <= tolerance
                _assert_psd_blocks(blocks_c)
    assert converged[0] >= converged[1]


def _exact_status(q_d2, support, amplitudes, /):
    result = _separate_full_curvature(
        q_d2, np.asarray(support), amplitudes, 1e-12, 1e-10, 80, 2
    )
    return result.status


_CODES = {1: "feasible", 0: "violated"}


def test_compiled_certificate_never_contradicts_the_exact_separator():
    rng = np.random.default_rng(0)
    checked = 0
    for trial in range(240):
        degree = int(rng.integers(0, 9))
        q_d2 = rng.normal(size=degree + 1)
        kind = trial % 4
        if kind == 0:
            support = (-np.inf, np.inf)
            amplitudes = np.array([np.nan, np.nan])
        elif kind == 1:
            support = (-1.3, np.inf)
            amplitudes = np.array([abs(rng.normal()), np.nan])
        elif kind == 2:
            support = (-np.inf, 0.7)
            amplitudes = np.array([np.nan, abs(rng.normal())])
        else:
            support = (-1.3, 1.7)
            amplitudes = np.array([abs(rng.normal()), abs(rng.normal())])
        code = _curvature_certificate.certify_full_curvature(
            q_d2, support[0], support[1], amplitudes[0], amplitudes[1], 1e-12
        )
        if code not in _CODES:
            continue
        exact = _exact_status(q_d2, support, amplitudes)
        if exact == "uncertain":
            continue
        assert _CODES[code] == exact, (trial, q_d2, support, amplitudes)
        checked += 1
    assert checked >= 200


@pytest.mark.parametrize("shift", [1e-3, 1e-9, 5e-13])
def test_compiled_certificate_detects_small_violations(shift):
    # (z^2 - 1)^2 touches zero at z = ±1; lowering it by any positive amount
    # beyond the tolerance must be reported as a violation.
    q_d2 = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
    touching = _curvature_certificate.certify_full_curvature(
        q_d2, -np.inf, np.inf, np.nan, np.nan, 1e-12
    )
    assert touching in (1, -1)
    lowered = q_d2.copy()
    lowered[0] -= shift
    code = _curvature_certificate.certify_full_curvature(
        lowered, -np.inf, np.inf, np.nan, np.nan, 1e-12
    )
    if shift > 1e-12:
        assert code in (0, -1)
        if shift >= 1e-9:
            assert code == 0


# ---------------------------------------------------------------------------
# Compiled state numerics, point statistics and mixture reductions
# ---------------------------------------------------------------------------


def _fitted_states():
    """Natural states at fitted parameters across support geometries."""
    from gibbus._fit.natural_objective import _fit_natural_conic_points

    rng = np.random.default_rng(11)
    cases = [
        ((-np.inf, np.inf), rng.gumbel(size=300), False, False),
        ((0.0, np.inf), rng.gamma(2.5, size=300), True, False),
        ((0.0, np.inf), rng.gamma(2.5, size=300), False, False),
        ((-np.inf, 0.0), -rng.gamma(1.5, size=300), False, True),
        ((0.0, 1.0), rng.beta(2.0, 3.0, size=300), True, True),
        ((0.0, 1.0), rng.beta(2.0, 3.0, size=300), False, True),
    ]
    for support, x, lower, upper in cases:
        for degree in (4, 6):
            objective, result = _fit_natural_conic_points(support, x, degree, lower, upper)
            yield objective, result.params


class _ReferenceNumericsState:
    """Factory for natural states normalized by the scalar reference path."""

    @staticmethod
    def build(objective, params):
        from gibbus._model.natural_state import _NaturalCoreState

        class _Reference(_NaturalCoreState):
            def _compiled_numerics(self, *args):
                return False

        return _Reference(
            objective.spec.coordinate, objective.layout, params, objective.z_data_bounds
        )


def test_compiled_state_numerics_match_the_reference_quadrature():
    from gibbus._fit.objective import _model_first_means_and_fisher

    for objective, params in _fitted_states():
        compiled = objective.build_state(params)
        assert compiled._first_means_fisher is not None
        reference = _ReferenceNumericsState.build(objective, params)
        assert reference._first_means_fisher is None
        assert compiled.log_Z == pytest.approx(reference.log_Z, abs=1e-10)
        np.testing.assert_array_equal(compiled.window, reference.window)
        assert compiled.mode == reference.mode
        assert compiled.quad_points == reference.quad_points
        mu_c, fisher_c = _model_first_means_and_fisher(compiled)
        mu_r, fisher_r = _model_first_means_and_fisher(reference)
        scale = np.sqrt(np.diag(fisher_r))
        np.testing.assert_allclose(mu_c / scale, mu_r / scale, rtol=0, atol=1e-8)
        np.testing.assert_allclose(
            fisher_c / np.outer(scale, scale), fisher_r / np.outer(scale, scale),
            rtol=0, atol=1e-7,
        )
        order = 2 * objective.layout.effective_poly_degree
        np.testing.assert_allclose(
            compiled.moments.power(order), reference.moments.power(order), rtol=1e-8, atol=1e-10
        )


def test_compiled_point_newton_is_certified_and_self_consistent():
    """Point Newton endpoints are feasible, stationary, and exactly re-evaluable."""
    from gibbus._fit.conic_newton import (
        _certify,
        _default_blocks,
        _interior_start,
        _newton_on_representation,
        _NewtonOptions,
    )
    from gibbus._fit.natural_objective import _prepare_natural_point_objective

    options = _NewtonOptions(1e-12, 1e-10, 1e-7, 60, 1e-4, 0.5, 40)
    rng = np.random.default_rng(123)
    cases = [
        ((-np.inf, np.inf), rng.gumbel(size=180), 6, False, False),
        ((0.0, np.inf), rng.gamma(2.3, size=180), 6, True, False),
        ((-np.inf, 0.0), -rng.gamma(1.8, size=180), 5, False, True),
        ((0.0, 1.0), rng.beta(2.0, 3.0, size=180), 6, True, True),
    ]

    for support, samples, degree, lower, upper in cases:
        objective = _prepare_natural_point_objective(
            support, samples, degree, lower, upper, None
        )
        params = _interior_start(objective)
        representation = _support_representation(objective.layout)
        blocks = _default_blocks(representation)
        run = _newton_on_representation(
            objective, representation, params, blocks, objective(params), options, 0
        )
        assert run.status in ("converged", "converged_approximately")
        assert _certify(objective.layout, run.params).feasible
        scale = max(1.0, abs(float(run.evaluation.nll)))
        assert run.decrease_bound <= options.accuracy_floor * scale
        exact = objective(run.params)
        assert run.evaluation.nll == pytest.approx(exact.nll, abs=5e-13)
        np.testing.assert_allclose(run.evaluation.gradient, exact.gradient, rtol=0, atol=5e-13)
        np.testing.assert_allclose(run.evaluation.hessian, exact.hessian, rtol=0, atol=1e-11)
        np.testing.assert_allclose(
            run.evaluation.model_partial_means,
            exact.model_partial_means,
            rtol=0,
            atol=5e-13,
        )

def test_compiled_interval_newton_is_certified_and_self_consistent():
    """Real-line interval Newton endpoints carry exact accepted-trial geometry."""
    from gibbus._fit.conic_newton import (
        _certify,
        _default_blocks,
        _interior_start,
        _newton_on_representation,
        _NewtonOptions,
    )
    from gibbus._fit.natural_objective import (
        _NaturalIntervalObjectiveFunction,
        _prepare_natural_interval_objective,
    )

    options = _NewtonOptions(1e-12, 1e-10, 1e-7, 60, 1e-4, 0.5, 40)
    rng = np.random.default_rng(321)

    for degree in (2, 4, 6):
        centers = rng.normal(size=160)
        widths = rng.uniform(0.01, 0.2, size=centers.size)
        rows = np.column_stack((centers - 0.5 * widths, centers + 0.5 * widths))
        prepared = _prepare_natural_interval_objective(
            (-np.inf, np.inf), rows, degree, False, False, None
        )
        objective = _NaturalIntervalObjectiveFunction(
            prepared.spec,
            prepared.observations,
            prepared.z_data_bounds,
            nonparametric_bound=False,
        )
        params = _interior_start(objective)
        representation = _support_representation(objective.layout)
        blocks = _default_blocks(representation)
        run = _newton_on_representation(
            objective, representation, params, blocks, objective(params), options, 0
        )
        assert run.status in ("converged", "converged_approximately")
        assert _certify(objective.layout, run.params).feasible
        scale = max(1.0, abs(float(run.evaluation.nll)))
        assert run.decrease_bound <= options.accuracy_floor * scale
        exact = objective(run.params)
        np.testing.assert_allclose(run.evaluation.gradient, exact.gradient, rtol=0, atol=5e-14)
        np.testing.assert_allclose(run.evaluation.hessian, exact.hessian, rtol=0, atol=5e-14)
        np.testing.assert_allclose(run.evaluation.fisher, exact.fisher, rtol=0, atol=5e-14)
        np.testing.assert_allclose(
            run.evaluation.missing_information, exact.missing_information, rtol=0, atol=5e-14
        )
        assert run.evaluation.smallest_curvature == pytest.approx(
            exact.smallest_curvature, rel=0, abs=1e-13
        )

def test_compiled_standalone_interval_newton_ignores_global_bound_locally():
    """Standalone global bounds do not contaminate the local fixed-face solve."""
    from gibbus._fit.conic_newton import (
        _certify,
        _default_blocks,
        _newton_on_representation,
        _NewtonOptions,
    )
    from gibbus._fit.natural_objective import (
        _natural_interval_start,
        _prepare_natural_interval_objective,
    )

    options = _NewtonOptions(1e-12, 1e-10, 1e-7, 60, 1e-4, 0.5, 40)
    rng = np.random.default_rng(913)
    normal = rng.normal(size=180)
    lo = np.floor(normal / 0.3) * 0.3
    real_rows = np.column_stack([lo, lo + 0.3])
    beta = rng.beta(2.0, 3.0, size=160)
    lo = np.floor(beta / 0.1) * 0.1
    bounded_rows = np.column_stack([lo, np.minimum(lo + 0.1, 1.0)])
    edges = np.arange(0.0, 2.0001, 0.25)
    half_rows = np.vstack([np.column_stack([edges[:-1], edges[1:]]), [2.0, np.inf]])
    half_weights = np.r_[np.exp(-edges[:-1]) - np.exp(-edges[1:]), np.exp(-2.0)]
    cases = [
        ((-np.inf, np.inf), real_rows, 4, False, False, None),
        ((0.0, 1.0), bounded_rows, 4, False, False, None),
        ((0.0, np.inf), half_rows, 4, False, False, half_weights),
    ]

    for support, rows, degree, lower, upper, weights in cases:
        objective = _prepare_natural_interval_objective(
            support, rows, degree, lower, upper, weights
        )
        assert objective.nll_lower_bound is not None
        params, warm_blocks = _natural_interval_start(objective)
        representation = _support_representation(objective.layout)
        shapes = [matrices.shape[1:] for matrices in representation.row_matrices]
        blocks = (
            warm_blocks
            if warm_blocks is not None and [np.shape(q) for q in warm_blocks] == shapes
            else _default_blocks(representation)
        )
        run = _newton_on_representation(
            objective, representation, params, blocks, objective(params), options, 0
        )
        assert run.status in ("converged", "converged_approximately")
        assert _certify(objective.layout, run.params).feasible
        exact = objective(run.params)
        assert run.evaluation.nll == pytest.approx(exact.nll, abs=2e-12)
        scale = max(1.0, abs(float(exact.nll)))
        assert run.decrease_bound <= options.accuracy_floor * scale

def test_state_numerics_decline_a_nonconvex_potential():
    from gibbus._model._state_kernels import state_numerics
    from gibbus._model.coords import _build_fit_coordinate
    from gibbus._model.natural import _natural_layout
    from gibbus._model.natural_state import _MODE_CONTROLS, _layout_numerics, _NaturalCoreState

    layout = _natural_layout((-np.inf, np.inf), 4)
    numerics = _layout_numerics(layout)
    # Double well q = z^4/12 - 2 z^2: curvature z^2 - 4 is negative near zero.
    q_poly = np.array([0.0, 0.3, -2.0, 0.0, 1.0 / 12.0])
    status = state_numerics(
        numerics.support, q_poly, np.full(2, np.nan), np.array([-3.0, 3.0]),
        False, False, numerics.kinds, numerics.lengths, numerics.coefficients,
        _MODE_CONTROLS, 1.49e-8, 1.49e-8, 100,
    )[0]
    assert status in (0, 2)
    # Whatever the compiled traversal decides, the state is normalized.
    coordinate = _build_fit_coordinate((-np.inf, np.inf), np.linspace(-3, 3, 20), None, None)
    params = np.array([0.3, -4.0, 0.0, 2.0])
    state = _NaturalCoreState(coordinate, layout, params, (-3.0, 3.0))
    assert np.isfinite(state.log_Z)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize(("support", "lower", "upper"), [
    ((-np.inf, np.inf), False, False),
    ((0.0, np.inf), True, False),
    ((0.0, 1.0), True, True),
])
def test_compiled_point_statistics_match_direct_sums(weighted, support, lower, upper):
    from gibbus._fit.natural_objective import _natural_point_stats

    rng = np.random.default_rng(5)
    z = rng.beta(2.0, 3.0, 400) if np.isfinite(support[1]) else rng.gamma(2.0, size=400)
    raw = rng.uniform(0.0, 3.0, z.size) if weighted else np.ones(z.size)
    if weighted:
        raw[::7] = 0.0
    compiled = _natural_point_stats(z, raw if weighted else None, 12, support, lower, upper)
    w = raw / raw.sum()
    powers = z[None, :] ** np.arange(13)[:, None]
    np.testing.assert_allclose(compiled.moments, powers @ w, rtol=1e-13, atol=0)
    contribution = w[None, :] * np.abs(powers)
    participation = np.minimum(
        contribution.sum(axis=1) ** 2 / np.sum(contribution**2, axis=1), z.size
    )
    np.testing.assert_allclose(
        compiled.moment_effective_n, participation, rtol=1e-12, atol=0
    )
    expected_log = [
        -np.dot(w, np.log(z - support[0])) if lower else np.nan,
        -np.dot(w, np.log(support[1] - z)) if upper else np.nan,
    ]
    np.testing.assert_allclose(
        compiled.boundary_log, expected_log, rtol=1e-13, equal_nan=True
    )
    assert compiled.total_weight == pytest.approx(raw.sum(), rel=1e-14)
    assert compiled.effective_n == pytest.approx(1.0 / np.dot(w, w), rel=1e-13)
    assert compiled.n_observations == z.size


def test_compiled_point_statistics_reject_invalid_inputs():
    from gibbus._fit.natural_objective import _natural_point_stats

    z = np.linspace(0.1, 2.0, 10)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _natural_point_stats(z, -np.ones(10), 4, (0.0, np.inf), False, False)
    with pytest.raises(ValueError, match="positive"):
        _natural_point_stats(z, np.zeros(10), 4, (0.0, np.inf), False, False)
    with pytest.raises(ValueError, match="above L"):
        _natural_point_stats(np.r_[0.0, z], None, 4, (0.0, np.inf), True, False)
    with pytest.raises(ValueError, match="within the support"):
        _natural_point_stats(np.r_[-1.0, z], None, 4, (0.0, np.inf), False, False)


def test_mixture_posterior_matches_logsumexp():
    from scipy.special import logsumexp

    from gibbus._fit._mixture_kernels import mixture_posterior

    rng = np.random.default_rng(3)
    log_values = rng.normal(size=(50, 3)) * 30.0
    log_weights = np.log(np.array([0.2, 0.5, 0.3]))
    w = rng.uniform(size=50)
    w /= w.sum()
    status, value, responsibilities, rows = mixture_posterior(log_values, log_weights, w)
    joint = log_values + log_weights
    expected_rows = logsumexp(joint, axis=1)
    assert status == 0
    np.testing.assert_allclose(rows, expected_rows, rtol=1e-15, atol=1e-13)
    assert value == pytest.approx(float(w @ expected_rows), rel=1e-14)
    np.testing.assert_allclose(
        responsibilities, np.exp(joint - expected_rows[:, None]), rtol=1e-13, atol=1e-300
    )
    log_values[7] = -np.inf
    assert mixture_posterior(log_values, log_weights, w)[0] == 1


def test_joint_information_matches_the_score_formula():
    from gibbus._fit._mixture_kernels import joint_information

    rng = np.random.default_rng(8)
    offsets = np.array([0, 3, 7], dtype=np.intp)
    rows, k_count = 40, 2
    n_total = offsets[-1] + k_count - 1
    responsibility = rng.dirichlet(np.ones(k_count), size=rows)
    centered = rng.normal(size=(rows, offsets[-1]))
    within = [None, np.stack([a @ a.T for a in rng.normal(size=(rows, 4, 4))])]
    pi = np.array([0.4, 0.6])
    w = rng.uniform(size=rows)
    w /= w.sum()
    gradient, missing = joint_information(responsibility, centered, within, offsets, pi, w)

    scores = np.zeros((rows, k_count, n_total))
    for k in range(k_count):
        scores[:, k, offsets[k]: offsets[k + 1]] = centered[:, offsets[k]: offsets[k + 1]]
        scores[:, k, offsets[-1]:] = pi[:-1]
        if k < k_count - 1:
            scores[:, k, offsets[-1] + k] -= 1.0
    mean_score = np.einsum("ik,ikn->in", responsibility, scores)
    expected_gradient = w @ mean_score
    expected = np.einsum("ik,ikn,ikm->nm", w[:, None] * responsibility, scores, scores)
    expected -= np.einsum("i,in,im->nm", w, mean_score, mean_score)
    block = slice(offsets[1], offsets[2])
    expected[block, block] += np.einsum("i,iab->ab", w * responsibility[:, 1], within[1])
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(missing, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(missing, missing.T)


# ---------------------------------------------------------------------------
# Compiled spectral CDF/PPF construction
# ---------------------------------------------------------------------------


def test_compiled_spectral_builders_reproduce_python_harness():
    from gibbus._spectral.cdf import (
        SpectralCDF,
        boundary_aware_breaks_from_amplitudes,
        density_spec,
    )
    from gibbus._spectral.ppf import SpectralPPF

    for objective, params in _fitted_states():
        state = objective.build_state(params)
        support = np.asarray(state.spec.support, dtype=np.float64)
        amps = np.asarray(state.boundary_amplitudes, dtype=np.float64)
        density = density_spec(
            [(state.q_poly, support[0], support[1], amps[0], amps[1], state.log_Z,
              0.0, 1.0, 1.0, 1.0, -np.inf, np.inf)],
            view=False,
        )
        grid = np.linspace(-4.0, 4.0, 101)
        np.testing.assert_array_equal(density.pdf(grid), state.pdf(grid))
        breaks = boundary_aware_breaks_from_amplitudes(support, amps)
        moments = state.moments.power(2)
        std = float(np.sqrt(max(moments[2] - moments[1] ** 2, 0.0)))
        reference = PythonSpectralCDFBuilder(state.pdf, support, mode=state.mode, std=std,
                                         initial_breaks=breaks)
        compiled = SpectralCDF(support, density=density, mode=state.mode, std=std,
                               initial_breaks=breaks)
        np.testing.assert_array_equal(compiled.breaks, reference.breaks)
        assert compiled.map == reference.map
        z = np.linspace(-1.0, 1.0, 1001)
        np.testing.assert_array_equal(compiled.cdf_z(z), reference.cdf_z(z))
        p = np.linspace(1e-9, 1.0 - 1e-9, 1001)
        np.testing.assert_array_equal(
            SpectralPPF(compiled).ppf_z(p), PythonSpectralPPFBuilder(reference).ppf_z(p)
        )


def test_compiled_mixture_spectral_cache_reproduces_python_harness(monkeypatch):
    import gibbus._api.mixture_stats as mixture_stats
    from gibbus import Distribution

    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(-3, 0.6, 200), rng.gumbel(1.5, 0.8, 300)])
    fitted = Distribution().fit(x, n_components=2, poly_degree=4, rng=0)
    grid = np.linspace(-6.0, 8.0, 801)
    p = np.linspace(1e-8, 1.0 - 1e-8, 801)
    compiled = (fitted.cdf(grid), fitted.ppf(p))
    monkeypatch.setattr(
        mixture_stats, "SpectralCDF",
        lambda support, **kwargs: PythonSpectralCDFBuilder(None, support, **kwargs),
    )
    monkeypatch.setattr(mixture_stats, "SpectralPPF", PythonSpectralPPFBuilder)
    fitted._spectral_cache_valid = False
    np.testing.assert_array_equal(fitted.cdf(grid), compiled[0])
    np.testing.assert_array_equal(fitted.ppf(p), compiled[1])


def test_compiled_interval_newton_is_consistent_across_finite_row_supports():
    """The fused interval loop is certified across ordinary finite-row supports."""
    from gibbus._fit.conic_newton import (
        _certify,
        _default_blocks,
        _interior_start,
        _newton_on_representation,
        _NewtonOptions,
        _support_representation,
    )
    from gibbus._fit.natural_objective import (
        _NaturalIntervalObjectiveFunction,
        _prepare_natural_interval_objective,
    )

    options = _NewtonOptions(1e-12, 1e-10, 1e-7, 60, 1e-4, 0.5, 40)
    rng = np.random.default_rng(270930)
    finite_centers = rng.uniform(0.15, 0.85, size=180)
    finite_widths = rng.uniform(0.01, 0.08, size=finite_centers.size)
    finite_rows = np.column_stack((finite_centers - 0.5 * finite_widths, finite_centers + 0.5 * finite_widths))
    lower_centers = rng.gamma(2.2, 0.8, size=180) + 0.2
    lower_widths = rng.uniform(0.01, 0.08, size=lower_centers.size)
    lower_rows = np.column_stack((lower_centers - 0.5 * lower_widths, lower_centers + 0.5 * lower_widths))
    upper_centers = -(rng.gamma(1.8, 0.9, size=180) + 0.2)
    upper_widths = rng.uniform(0.01, 0.08, size=upper_centers.size)
    upper_rows = np.column_stack((upper_centers - 0.5 * upper_widths, upper_centers + 0.5 * upper_widths))
    cases = [
        ((0.0, 1.0), finite_rows, 4, False, False),
        ((0.0, 1.0), finite_rows, 4, True, True),
        ((0.0, np.inf), lower_rows, 4, True, False),
        ((-np.inf, 0.0), upper_rows, 4, False, True),
    ]

    for support, rows, degree, lower, upper in cases:
        prepared = _prepare_natural_interval_objective(support, rows, degree, lower, upper, None)
        objective = _NaturalIntervalObjectiveFunction(
            prepared.spec, prepared.observations, prepared.z_data_bounds, nonparametric_bound=False
        )
        assert objective._compiled_interval_newton_eligible
        params = _interior_start(objective)
        representation = _support_representation(objective.layout)
        blocks = _default_blocks(representation)
        run = _newton_on_representation(
            objective, representation, params, blocks, objective(params), options, 0
        )
        assert run.status in ("converged", "converged_approximately")
        assert _certify(objective.layout, run.params).feasible
        exact = objective(run.params)
        scale = max(1.0, abs(float(exact.nll)))
        assert run.decrease_bound <= options.accuracy_floor * scale
        np.testing.assert_allclose(run.evaluation.gradient, exact.gradient, rtol=0, atol=2e-12)
        np.testing.assert_allclose(run.evaluation.hessian, exact.hessian, rtol=0, atol=2e-12)
        np.testing.assert_allclose(run.evaluation.fisher, exact.fisher, rtol=0, atol=2e-12)
        np.testing.assert_allclose(
            run.evaluation.missing_information, exact.missing_information, rtol=0, atol=2e-12
        )

def test_compiled_interval_newton_is_consistent_for_adaptive_row_geometries():
    """Boundary-touching and infinite censoring retain exact endpoint statistics."""
    from gibbus._fit.conic_newton import (
        _certify,
        _default_blocks,
        _interior_start,
        _newton_on_representation,
        _NewtonOptions,
        _support_representation,
    )
    from gibbus._fit.natural_objective import (
        _NaturalIntervalObjectiveFunction,
        _prepare_natural_interval_objective,
    )

    options = _NewtonOptions(1e-10, 1e-9, 1e-7, 50, 1e-4, 0.5, 30)
    rng = np.random.default_rng(271001)
    bounded_internal = np.column_stack((rng.uniform(0.15, 0.60, 80), rng.uniform(0.65, 0.85, 80)))
    bounded_rows = np.vstack([
        bounded_internal,
        np.column_stack((np.zeros(25), rng.uniform(0.03, 0.18, 25))),
        np.column_stack((rng.uniform(0.82, 0.95, 25), np.ones(25))),
        np.tile([0.0, 1.0], (5, 1)),
    ])
    lower = rng.gamma(2.0, 0.5, size=100) + 0.1
    halfline_rows = np.vstack([
        np.column_stack((lower, lower + rng.uniform(0.02, 0.20, 100))),
        np.column_stack((rng.uniform(0.8, 2.0, 30), np.full(30, np.inf))),
        np.column_stack((np.zeros(20), rng.uniform(0.03, 0.20, 20))),
        np.tile([0.0, np.inf], (5, 1)),
    ])
    upper = -(rng.gamma(1.8, 0.6, size=100) + 0.1)
    upper_halfline_rows = np.vstack([
        np.column_stack((upper - rng.uniform(0.02, 0.20, 100), upper)),
        np.column_stack((np.full(30, -np.inf), -rng.uniform(0.8, 2.0, 30))),
        np.column_stack((-rng.uniform(0.03, 0.20, 20), np.zeros(20))),
        np.tile([-np.inf, 0.0], (5, 1)),
    ])
    lower = rng.normal(0.0, 1.0, size=100)
    real_rows = np.vstack([
        np.column_stack((lower, lower + rng.uniform(0.05, 0.25, 100))),
        np.column_stack((np.full(20, -np.inf), rng.normal(-1.0, 0.3, 20))),
        np.column_stack((rng.normal(1.0, 0.3, 20), np.full(20, np.inf))),
        np.tile([-np.inf, np.inf], (5, 1)),
    ])
    cases = [
        ((0.0, 1.0), bounded_rows, True, True),
        ((0.0, np.inf), halfline_rows, True, False),
        ((-np.inf, 0.0), upper_halfline_rows, False, True),
        ((-np.inf, np.inf), real_rows, False, False),
    ]
    for support, rows, lower_boundary, upper_boundary in cases:
        prepared = _prepare_natural_interval_objective(
            support, rows, 2, lower_boundary, upper_boundary, None
        )
        objective = _NaturalIntervalObjectiveFunction(
            prepared.spec, prepared.observations, prepared.z_data_bounds, nonparametric_bound=False
        )
        assert objective._compiled_interval_newton_eligible
        packed = objective._compiled_interval_newton_inputs()
        assert packed is not None
        assert packed[10].shape[0] > 0
        assert packed[12] > 0.0
        params = _interior_start(objective)
        representation = _support_representation(objective.layout)
        blocks = _default_blocks(representation)
        run = _newton_on_representation(
            objective, representation, params, blocks, objective(params), options, 0
        )
        # A raw fixed-face solve can legitimately stall on a face that the
        # outer active-set controller will subsequently change.  Regardless of
        # status, its accepted endpoint must be feasible and exactly
        # re-evaluable; converged statuses additionally carry the local gap.
        assert run.status in {
            "converged", "converged_approximately", "non_descent",
            "line_search_failed", "iteration_limit",
        }
        assert _certify(objective.layout, run.params).feasible
        exact = objective(run.params)
        scale = max(1.0, abs(float(exact.nll)))
        if run.status in ("converged", "converged_approximately"):
            assert run.decrease_bound <= options.accuracy_floor * scale
        np.testing.assert_allclose(run.evaluation.gradient, exact.gradient, rtol=0, atol=3e-12)
        np.testing.assert_allclose(run.evaluation.hessian, exact.hessian, rtol=0, atol=3e-12)
        np.testing.assert_allclose(run.evaluation.fisher, exact.fisher, rtol=0, atol=3e-12)
        np.testing.assert_allclose(
            run.evaluation.missing_information, exact.missing_information, rtol=0, atol=3e-12
        )

