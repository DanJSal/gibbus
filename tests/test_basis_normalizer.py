"""G3 Stage 3b: normalizer and moments.

Analytic families (Gaussian at several widths, gamma, beta, a non-convex
two-well potential) and an independent graded composite Gauss-Legendre
reference (test-only, self-checked by panel doubling) check ``log Z``,
``E[Phi]`` and ``Cov[Phi]``.  Improper vectors must be reported as such, and
an exhausted budget must fail loudly.
"""

import math
import time
import warnings

import numpy as np
import pytest
from scipy.special import betaln, gammaln
from test_basis_admissibility import _grid_lp
from test_basis_evaluation import _broad_battery

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_normalizer import (
    _basis_state,
    _NormalizerContext,
    _NormalizerOptions,
    _QuadratureFailure,
)

RNG = np.random.default_rng(1016)
NORMAL = RNG.normal(1.5, 2.0, size=400)
HALF = RNG.gamma(2.0, 1.5, size=400) + 0.25
BOUNDED = RNG.beta(2.0, 3.0, size=400) * 4.0 - 1.0


def _data(support):
    lo, up = support
    if math.isinf(lo) and math.isinf(up):
        return NORMAL
    if math.isinf(up):
        return HALF + lo
    if math.isinf(lo):
        return up - HALF
    return BOUNDED


def _model(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, _data(spec.support))


def _theta(model, weights):
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    for label, value in weights.items():
        theta[labels.index(label)] = value
    return theta, labels


# ---------------------------------------------------------------------------
# Analytic families
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1e-6, 1e-2, 0.7, 50.0, 1e6])
def test_gaussian_normalizer_and_moments(a):
    model = _model(BasisSpec(bulk_degree=2))
    b = 0.3 * math.sqrt(a)
    theta, labels = _theta(model, {"x^2": a, "x": b})
    res = _basis_state(_NormalizerContext(model), theta)
    h = model.geometry.scale
    exact = math.log(h * math.sqrt(math.pi / a)) + b * b / (4.0 * a)
    assert res.status == "proper"
    assert res.log_z == pytest.approx(exact, abs=1e-12 * max(1.0, abs(exact)))
    ix, i2 = labels.index("x"), labels.index("x^2")
    mz, vz = -b / (2.0 * a), 1.0 / (2.0 * a)
    assert res.mean[ix] == pytest.approx(mz, rel=1e-11, abs=1e-12 * math.sqrt(vz))
    assert res.cov[ix, ix] == pytest.approx(vz, rel=1e-11)
    # E[z^2] and Var[z^2] = 2 v^2 + 4 m^2 v for a Gaussian.
    assert res.mean[i2] == pytest.approx(vz + mz**2, rel=1e-11)
    assert res.cov[i2, i2] == pytest.approx(2 * vz**2 + 4 * mz**2 * vz, rel=1e-10)
    assert res.rel_error <= 1e-12


@pytest.mark.parametrize(
    "gamma,beta", [(0.0, 1.0), (0.5, 2.0), (3.0, 0.7), (1e-3, 5.0)]
)
def test_gamma_half_line_with_log_endpoint(gamma, beta):
    model = _model(BasisSpec(support=(0.0, math.inf), bulk_degree=1, log_lower=True))
    theta, labels = _theta(model, {"logL": gamma, "x": beta})
    res = _basis_state(_NormalizerContext(model), theta)
    g = model.geometry
    # q = -gamma log x + beta (x - c) / h  ->  Z = e^{beta c / h} Gamma(gamma+1) (h/beta)^{gamma+1}
    exact = (
        beta * g.center / g.scale
        + gammaln(gamma + 1.0)
        + (gamma + 1.0) * math.log(g.scale / beta)
    )
    assert res.log_z == pytest.approx(exact, abs=1e-12 * max(1.0, abs(exact)))
    # E[x] = (gamma + 1) h / beta  ->  E[z] = (E[x] - c) / h
    ez = ((gamma + 1.0) * g.scale / beta - g.center) / g.scale
    assert res.mean[labels.index("x")] == pytest.approx(ez, rel=1e-11)
    # E[-log x] = -(digamma(gamma + 1) + log(h / beta))
    from scipy.special import digamma

    elog = -(digamma(gamma + 1.0) + math.log(g.scale / beta))
    assert res.mean[labels.index("logL")] == pytest.approx(elog, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("g1,g2", [(0.0, 0.0), (0.5, 2.0), (3.0, 1e-3), (10.0, 10.0)])
def test_beta_bounded_with_two_log_endpoints(g1, g2):
    model = _model(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=1, log_lower=True, log_upper=True)
    )
    theta, labels = _theta(model, {"logL": g1, "logU": g2})
    res = _basis_state(_NormalizerContext(model), theta)
    # (x + 1)^g1 (3 - x)^g2 on (-1, 3): Z = 4^{g1+g2+1} B(g1 + 1, g2 + 1)
    exact = (g1 + g2 + 1.0) * math.log(4.0) + betaln(g1 + 1.0, g2 + 1.0)
    assert res.log_z == pytest.approx(exact, abs=1e-12 * max(1.0, abs(exact)))
    # E[x] = -1 + 4 (g1 + 1) / (g1 + g2 + 2)
    ex = -1.0 + 4.0 * (g1 + 1.0) / (g1 + g2 + 2.0)
    ez = (ex - model.geometry.center) / model.geometry.scale
    assert res.mean[labels.index("x")] == pytest.approx(ez, rel=1e-11, abs=1e-12)


def test_non_convex_two_well_potential():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1.0, "x^2": -4.0, "x": 0.5})
    res = _basis_state(_NormalizerContext(model), theta)
    ref = _reference(model, theta)
    assert res.log_z == pytest.approx(ref[0], abs=1e-12)
    np.testing.assert_allclose(res.mean, ref[1], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(res.cov, ref[2], rtol=1e-10, atol=1e-11)


def test_improper_vectors_are_reported():
    model = _model(BasisSpec(bulk_degree=2))
    theta, _ = _theta(model, {"x^2": -0.1})
    res = _basis_state(_NormalizerContext(model), theta)
    assert res.status == "improper" and res.log_z == math.inf and res.mean is None


def test_negative_log_amplitude_is_rejected():
    model = _model(BasisSpec(support=(0.0, math.inf), bulk_degree=1, log_lower=True))
    theta, _ = _theta(model, {"logL": -0.5, "x": 1.0})
    with pytest.raises(ValueError):
        _basis_state(_NormalizerContext(model), theta)


def test_exhausted_budget_fails_loudly():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1e4, "x^2": -4e4, "x": 50.0})
    opts = _NormalizerOptions(max_cells=8, initial_cells=4)
    with pytest.raises(_QuadratureFailure) as info:
        _basis_state(_NormalizerContext(model), theta, opts)
    assert "cells" in info.value.diagnostics


# ---------------------------------------------------------------------------
# Independent reference across the battery
# ---------------------------------------------------------------------------


def _gl_panels(model, theta, panels):
    """Composite 40-point Gauss-Legendre over panels.

    Returns ``(log Z, mean, cov)`` with a centred (two-pass) covariance.
    """
    nodes, wts = np.polynomial.legendre.leggauss(40)
    a, b = panels[:-1], panels[1:]
    x = (0.5 * (a + b))[:, None] + (0.5 * (b - a))[:, None] * nodes
    w = (0.5 * (b - a))[:, None] * wts
    x, w = x.ravel(), w.ravel()
    phi = _evaluate_basis(model, x, max_order=0).phi
    used = theta != 0.0
    q = phi[:, used] @ theta[used]
    q_ref = float(np.min(q))
    e = w * np.exp(-(q - q_ref))
    phi = np.where(np.isfinite(phi), phi, 0.0)
    z = e.sum()
    mean = (e @ phi) / z
    centred = phi - mean
    cov = (centred * (e / z)[:, None]).T @ centred
    return math.log(z) - q_ref, mean, cov


def _reference(model, theta, n=400):
    """Independent ``(log Z, mean, cov)``: graded composite Gauss-Legendre.

    Truncated where ``q - min q > 80``; panels graded geometrically toward
    finite endpoints; the result is accepted only if doubling the panels
    changes ``log Z`` by less than 1e-13.
    """
    req, g = model.request, model.geometry
    used = theta != 0.0

    def q_at(x):
        return _evaluate_basis(model, x, max_order=0).phi[:, used] @ theta[used]

    # Grow the window until q - min q > 80 at both truncated ends (linear tails
    # can need thousands of scales).
    span = 20.0 * g.scale
    while True:
        lo = req.lower if math.isfinite(req.lower) else g.center - span
        hi = req.upper if math.isfinite(req.upper) else g.center + span
        probe = np.linspace(lo, hi, 40001)[1:-1]
        q = q_at(probe)
        qmin = q.min()
        ends_ok = (math.isfinite(req.lower) or q[0] - qmin > 80.0) and (
            math.isfinite(req.upper) or q[-1] - qmin > 80.0
        )
        if ends_ok:
            break
        span *= 2.0
        assert span < 1e7 * g.scale, "reference window did not converge"
    keep = q - qmin < 80.0
    lo = lo if keep[0] else max(probe[keep][0] - 0.01 * g.scale, req.lower)
    hi = hi if keep[-1] else min(probe[keep][-1] + 0.01 * g.scale, req.upper)

    def panels(k):
        core = np.linspace(lo, hi, k + 1)
        extra = []
        for end, sign in ((req.lower, 1.0), (req.upper, -1.0)):
            if math.isfinite(end) and (end == lo or end == hi):
                extra.extend(end + sign * (hi - lo) / k * 2.0 ** -np.arange(1, 50))
        return np.unique(np.concatenate([core, extra]))

    out = [_gl_panels(model, theta, panels(k)) for k in (n, 2 * n)]
    assert abs(out[0][0] - out[1][0]) < 1e-13 * max(
        1.0, abs(out[1][0])
    ), "reference not converged"
    return out[1]


BATTERY = list(_broad_battery())[::3]


@pytest.mark.parametrize("spec", BATTERY, ids=lambda s: "")
def test_normalizer_matches_independent_quadrature(spec):
    model = _model(spec)
    ctx = _NormalizerContext(model)
    feasible = _grid_lp(model, 8.0, 801)
    thetas = (
        []
        if feasible is None
        else [
            feasible,
            feasible + 0.1 * np.max(np.abs(feasible)) * RNG.normal(size=feasible.size),
        ]
    )
    nonneg = [i for i, c in enumerate(model.columns) if c.nonnegative]
    for theta in thetas:
        theta[nonneg] = np.abs(theta[nonneg])
        res = _basis_state(ctx, theta)
        if res.status == "improper":
            continue
        ref = _reference(model, theta)
        assert res.log_z == pytest.approx(ref[0], abs=1e-11 * max(1.0, abs(ref[0])))
        scale = np.sqrt(np.abs(np.diag(ref[2]))) + np.abs(ref[1]) + 1e-300
        assert np.all(np.abs(res.mean - ref[1]) <= 1e-10 * scale)
        assert np.all(np.abs(res.cov - ref[2]) <= 1e-9 * np.outer(scale, scale))
        assert res.rel_error <= 1e-12
        assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_typical_normalizer_time():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 0.05, "x^2": 0.5, "x^3": 0.1})
    ctx = _NormalizerContext(model)
    start = time.perf_counter()
    for _ in range(5):
        _basis_state(ctx, theta)
    assert (time.perf_counter() - start) / 5 < 1.0


HARD_SPECS = [
    BasisSpec(bulk_degree=3, upper_tail=Polynomial(1), expansion="full"),
    BasisSpec(
        bulk_degree=3,
        lower_tail=SoftPower(1.5),
        upper_tail=SoftPower(1.5),
        expansion="full",
    ),
    BasisSpec(
        bulk_degree=2,
        lower_tail=Exponential(0.7),
        upper_tail=SoftPower(1.5),
        expansion="full",
    ),
    BasisSpec(bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Exponential(0.7)),
    BasisSpec(support=(0.0, math.inf), bulk_degree=4),
]


@pytest.mark.parametrize("spec", HARD_SPECS, ids=lambda s: "")
def test_steep_and_cancelling_vectors_resolve(spec):
    # Regression for the first sweep's loud failures: steep (x30) and strongly
    # perturbed vectors, far linear tails and large cancelling coefficients.
    model = _model(spec)
    ctx = _NormalizerContext(model)
    base = _grid_lp(model, 8.0, 801)
    assert base is not None
    scale = np.max(np.abs(base))
    rng = np.random.default_rng(31)
    for theta in (
        base * 30.0,
        base + scale * rng.normal(size=base.size),
        base * 30.0 + 30.0 * scale * rng.normal(size=base.size),
    ):
        res = _basis_state(ctx, theta)
        if res.status == "improper":
            continue
        ref = _reference(model, theta)
        tol = max(1e-11, 10.0 * res.cond_error)
        assert res.log_z == pytest.approx(ref[0], abs=tol * max(1.0, abs(ref[0])))
        assert res.cells < 512


def test_conditioning_floor_is_reported_not_failed():
    # Steep, perturbed vectors with large cancelling coefficients: the binary64
    # rounding of q itself limits Z.  The floor is reported, the quadrature
    # still meets its target, and the answer is within the floor of the truth.
    model = _model(HARD_SPECS[2])
    ctx = _NormalizerContext(model)
    base = _grid_lp(model, 8.0, 801)
    scale = np.max(np.abs(base))
    rng = np.random.default_rng(5)
    best = None
    for _ in range(12):
        theta = base * 30.0 + 30.0 * scale * rng.normal(size=base.size)
        res = _basis_state(ctx, theta)
        if res.status == "proper" and (
            best is None or res.cond_error > best[0].cond_error
        ):
            best = (res, theta)
    res, theta = best
    assert res.cond_error > 1e-13
    assert res.rel_error <= 1e-12
    ref = _reference(model, theta)
    assert res.log_z == pytest.approx(
        ref[0], abs=max(1e-11, 10.0 * res.cond_error) * max(1.0, abs(ref[0]))
    )


def test_nearly_improper_growth_fails_loudly_not_with_overflow():
    # Slope 1e-14 per scale: proper, but truncation would need ~1e16 scales.
    model = _model(BasisSpec(support=(0.0, math.inf), bulk_degree=1))
    theta, _ = _theta(model, {"x": 1e-14})
    with pytest.raises(_QuadratureFailure, match="cutoff cap"):
        _basis_state(_NormalizerContext(model), theta)
    # A modest slope is fine.
    theta, _ = _theta(model, {"x": 1e-3})
    assert _basis_state(_NormalizerContext(model), theta).status == "proper"


def test_gauss_kronrod_rule_is_exact_to_its_degree():
    from gibbus._model.gauss_kronrod import WG10, WK21, XK21

    xk = np.concatenate([-XK21[:-1], XK21])
    wk = np.concatenate([WK21[:-1], WK21])
    xg = np.concatenate([-XK21[1::2], XK21[1::2]])
    wg = np.concatenate([WG10, WG10])
    for k in range(0, 34, 2):
        exact = 2.0 / (k + 1)
        kron, gauss = wk @ xk**k, wg @ xg**k
        assert (abs(kron - exact) <= 1e-15) == (k <= 31)
        assert (abs(gauss - exact) <= 1e-15) == (k <= 19)
    nodes, _ = np.polynomial.legendre.leggauss(10)
    assert np.array_equal(np.sort(np.abs(nodes))[::2][::-1], XK21[1::2])


# ---------------------------------------------------------------------------
# Stage 6: fitting-coordinate normalization (OI-E2-1, G3-052)
# ---------------------------------------------------------------------------


def test_large_offset_gaussian_normalizer_uses_fitting_units():
    """A Gaussian at 1e6 with nanometric spread must retain its true Z."""
    x = 1e6 + 1e-9 * np.random.default_rng(48).standard_normal(600)
    model = _build_basis(BasisSpec(bulk_degree=2), x)
    theta, labels = _theta(model, {"x^2": 0.5})
    state = _basis_state(_NormalizerContext(model), theta)
    expected = math.log(model.geometry.scale) + 0.5 * math.log(2.0 * math.pi)
    assert state.log_z == pytest.approx(expected, abs=2e-12)
    assert state.mean[labels.index("x")] == pytest.approx(0.0, abs=2e-12)
    assert state.cov[labels.index("x"), labels.index("x")] == pytest.approx(
        1.0, rel=2e-11
    )


def test_tiny_bounded_support_normalizer_and_interval_rows():
    """256 adjacent user ulps must not collapse the normalizer's cells."""
    from gibbus._model.basis_normalizer import _interval_state

    lower = 1.0
    upper = lower + 256.0 * np.spacing(lower)
    x = lower + np.arange(1, 256, 4) * np.spacing(lower)
    model = _build_basis(BasisSpec(support=(lower, upper), bulk_degree=2), x)
    theta = np.zeros(model.n_coefficients)
    state = _basis_state(_NormalizerContext(model), theta)
    width = upper - lower
    assert state.log_z == pytest.approx(math.log(width), abs=2e-12)
    rows = _interval_state(
        _NormalizerContext(model),
        theta,
        np.array([lower, lower + 128.0 * np.spacing(lower), x[20]]),
        np.array([lower + 128.0 * np.spacing(lower), upper, x[20]]),
        np.ones(3) / 3,
        per_row=True,
    )
    np.testing.assert_allclose(
        rows.row_log_p,
        [math.log(0.5), math.log(0.5), -math.log(width)],
        rtol=0.0,
        atol=2e-12,
    )


def test_tiny_bounded_endpoint_log_restores_physical_constant():
    """The endpoint-log basis contributes its constant to log Z and means."""
    lower = 1.0
    upper = lower + 256.0 * np.spacing(lower)
    x = lower + np.arange(1, 256, 4) * np.spacing(lower)
    model = _build_basis(
        BasisSpec(support=(lower, upper), bulk_degree=2, log_lower=True), x
    )
    gamma = 0.5
    theta, labels = _theta(model, {"logL": gamma})
    state = _basis_state(_NormalizerContext(model), theta)
    width = upper - lower
    expected = (gamma + 1.0) * math.log(width) - math.log(gamma + 1.0)
    assert state.log_z == pytest.approx(expected, abs=3e-12)

    elog = -math.log(width) + 1.0 / (gamma + 1.0)
    # E[-log(x-L)] under Beta(gamma+1, 1) scaled to [L,U].
    assert state.mean[labels.index("logL")] == pytest.approx(elog, abs=2e-11)
