"""G3 Stage 5 Phase A: interval-row masses and conditional moments (G3-043).

Defining properties: ``log P(B)``, ``E[Phi | B]`` and ``Cov(Phi | B)`` of a
row equal closed forms (truncated Gaussian, incomplete gamma at a log
endpoint) and an independent graded Gauss-Legendre reference to the
normalizer's tolerance relative to the row's own mass, including small masses
far in a tail and narrow rows; rows partitioning the support have total mass
one and recover ``E[Phi]``; a whole-support row is the normalizer; a point
row is a log density; ``d log P / d theta = E[Phi] - E[Phi | B]``; an
unresolvable row fails loudly and is named.
"""

import math
import time
import warnings

import mpmath as mp
import numpy as np
import pytest
from test_basis_admissibility import _grid_lp
from test_basis_evaluation import _broad_battery
from test_basis_normalizer import _gl_panels

from gibbus import BasisSpec, Exponential, GibbusBasisWarning
from gibbus._fit.basis_newton import _solve_basis
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_normalizer import (
    _basis_state,
    _interval_state,
    _NormalizerContext,
    _QuadratureFailure,
)

INF = math.inf
RNG = np.random.default_rng(1017)
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
    return lo + (up - lo) * (BOUNDED + 1.0) / 4.0


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


def _rows(model, theta, lower, upper):
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    return _interval_state(
        _NormalizerContext(model),
        theta,
        lower,
        upper,
        np.ones(lower.size),
        per_row=True,
    )


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

GAUSS_ROWS = [
    (-INF, 0.0),
    (-1.0, 2.0),
    (0.99, 1.01),
    (5.0, INF),
    (12.0, 12.5),
    (-INF, -20.0),
    (30.0, INF),
    (60.0, INF),
    (-50.0, -50.0 + 1e-9),
    (-INF, INF),
]


def _truncated_normal(a, b, mu, s):
    """``log P``, mean and variance of N(mu, s^2) on [a, b] (mpmath, 80 digits)."""
    with mp.workdps(80):
        return _truncated_normal_mp(a, b, mu, s)


def _truncated_normal_mp(a, b, mu, s):
    mu, s = mp.mpf(mu), mp.mpf(s)
    al = (mp.mpf(a) - mu) / s if math.isfinite(a) else -mp.inf
    be = (mp.mpf(b) - mu) / s if math.isfinite(b) else mp.inf

    def sf(t):
        return mp.erfc(t / mp.sqrt(2)) / 2

    def pdf(t):
        return mp.npdf(t) if mp.isfinite(t) else mp.mpf(0)

    if al >= 0:
        z = sf(al) - sf(be)
    elif be <= 0:
        z = sf(-be) - sf(-al)
    else:
        z = 1 - sf(-al) - sf(be)
    da = pdf(al) - pdf(be)
    ta = al * pdf(al) if mp.isfinite(al) else 0
    tb = be * pdf(be) if mp.isfinite(be) else 0
    mean = mu + s * da / z
    var = s * s * (1 + (ta - tb) / z - (da / z) ** 2)
    return float(mp.log(z)), float(mean), float(var)


def test_gaussian_rows_match_the_truncated_normal():
    model = _model(BasisSpec(bulk_degree=2))
    g = model.geometry
    mu, s = 1.0, 1.7
    zm, zs = (mu - g.center) / g.scale, s / g.scale
    theta, labels = _theta(model, {"x": -zm / zs**2, "x^2": 0.5 / zs**2})
    lower, upper = zip(*GAUSS_ROWS, strict=True)
    res = _rows(model, theta, lower, upper)
    ix = labels.index("x")
    vz = zs * zs
    for i, (a, b) in enumerate(GAUSS_ROWS):
        lp, mean, var = _truncated_normal(a, b, mu, s)
        assert res.row_log_p[i] == pytest.approx(lp, abs=1e-12 * max(1.0, abs(lp)))
        ez = (mean - g.center) / g.scale
        assert res.row_mean[i, ix] == pytest.approx(ez, abs=1e-12 * max(1.0, abs(ez)))
        # The variance of a narrow row is ~ width^2: binary64 Phi values carry
        # an absolute error eps |z| there, so it is checked against Var(z).
        vrow = var / g.scale**2
        assert abs(res.row_cov[i, ix, ix] - vrow) <= 1e-10 * vrow + 1e-13 * vz


@pytest.mark.parametrize("gamma,beta", [(0.0, 1.0), (0.5, 2.0), (3.0, 0.7)])
def test_gamma_rows_at_a_log_endpoint_match_the_incomplete_gamma(gamma, beta):
    model = _model(BasisSpec(support=(0.0, INF), bulk_degree=1, log_lower=True))
    theta, labels = _theta(model, {"logL": gamma, "x": beta})
    g = model.geometry
    # q = -gamma log x + beta (x - c)/h: x ~ Gamma(gamma + 1, scale h / beta).
    rows = [(0.0, 0.1), (0.0, 2.0), (0.5, 3.0), (2.0, INF), (40.0, INF), (7.0, 7.25)]
    lower, upper = zip(*rows, strict=True)
    res = _rows(model, theta, lower, upper)
    for i, (a, b) in enumerate(rows):
        with mp.workdps(60):
            k, sc = mp.mpf(gamma) + 1, mp.mpf(g.scale) / mp.mpf(beta)
            ua = mp.mpf(a) / sc
            ub = mp.mpf(b) / sc if math.isfinite(b) else mp.inf
            p = mp.gammainc(k, ua, ub, regularized=True)
            # E[x | B] = k sc P_{k+1}(B) / P_k(B)
            ex = k * sc * mp.gammainc(k + 1, ua, ub, regularized=True) / p
            lp = float(mp.log(p))
        assert res.row_log_p[i] == pytest.approx(lp, abs=1e-12 * max(1.0, abs(lp)))
        ez = (float(ex) - g.center) / g.scale
        assert res.row_mean[i, labels.index("x")] == pytest.approx(
            ez, abs=1e-11 * max(1.0, abs(ez))
        )


# ---------------------------------------------------------------------------
# Structural identities
# ---------------------------------------------------------------------------

BATTERY = list(_broad_battery())[::4]


def _battery_theta(model):
    theta = _grid_lp(model, 8.0, 801)
    if theta is None:
        return None
    nonneg = [i for i, c in enumerate(model.columns) if c.nonnegative]
    theta[nonneg] = np.abs(theta[nonneg])
    return theta


def _fitted(model):
    """The point-data MLE (interior: perturbations stay proper)."""
    objective = _BasisPointObjective(
        _NormalizerContext(model),
        _point_statistics(model, _data((model.request.lower, model.request.upper))),
    )
    return _solve_basis(objective).params


def _partition(model, k):
    req, g = model.request, model.geometry
    lo = req.lower if math.isfinite(req.lower) else g.center - 6.0 * g.scale
    hi = req.upper if math.isfinite(req.upper) else g.center + 6.0 * g.scale
    cuts = np.linspace(lo, hi, k + 1)[1:-1]
    edges = np.concatenate([[req.lower], cuts, [req.upper]])
    return edges[:-1], edges[1:]


@pytest.mark.parametrize("spec", BATTERY, ids=lambda s: "")
def test_rows_partitioning_the_support_recover_the_model(spec):
    model = _model(spec)
    theta = _battery_theta(model)
    if theta is None:
        pytest.skip("no feasible vector")
    ctx = _NormalizerContext(model)
    state = _basis_state(ctx, theta)
    if state.status != "proper":
        pytest.skip("improper vector")
    lower, upper = _partition(model, 7)
    res = _interval_state(ctx, theta, lower, upper, np.ones(lower.size), per_row=True)
    p = np.exp(res.row_log_p)
    assert p.sum() == pytest.approx(1.0, abs=1e-11)
    scale = np.sqrt(np.abs(np.diag(state.cov))) + np.abs(state.mean) + 1e-300
    assert np.all(np.abs(p @ res.row_mean - state.mean) <= 1e-10 * scale)
    # Law of total covariance.
    between = (res.row_mean - state.mean).T @ (p[:, None] * (res.row_mean - state.mean))
    within = np.einsum("r,rij->ij", p, res.row_cov)
    assert np.all(np.abs(within + between - state.cov) <= 1e-9 * np.outer(scale, scale))


def test_whole_support_and_point_rows():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1.0, "x^2": -4.0, "x": 0.5})
    ctx = _NormalizerContext(model)
    state = _basis_state(ctx, theta)
    x = np.array([-1.3, 0.2, 2.9])
    res = _interval_state(
        ctx,
        theta,
        np.r_[-INF, x],
        np.r_[INF, x],
        np.array([0.5, 0.2, 0.2, 0.1]),
        per_row=True,
    )
    assert res.row_log_p[0] == 0.0
    np.testing.assert_array_equal(res.row_mean[0], state.mean)
    np.testing.assert_array_equal(res.row_cov[0], state.cov)
    phi = _evaluate_basis(model, x, max_order=0).phi
    np.testing.assert_allclose(
        res.row_log_p[1:], -(phi @ theta) - state.log_z, rtol=0, atol=1e-13
    )
    np.testing.assert_array_equal(res.row_mean[1:], phi)
    assert not np.any(res.row_cov[1:])
    w = np.array([0.5, 0.2, 0.2, 0.1])
    assert res.log_p == pytest.approx(w @ res.row_log_p, abs=1e-15)
    np.testing.assert_allclose(res.mean, w @ res.row_mean, rtol=1e-15, atol=1e-15)


def test_log_probability_gradient_is_the_mean_difference():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1.0, "x^2": -4.0, "x": 0.5})
    ctx = _NormalizerContext(model)
    rows = ([-INF, -1.0, 2.0], [-2.0, 1.5, INF])
    base = _interval_state(ctx, theta, *rows, np.ones(3), per_row=True)
    state = _basis_state(ctx, theta)
    eps = 1e-5
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = eps * max(1.0, abs(theta[k]))
        up = _interval_state(ctx, theta + step, *rows, np.ones(3), per_row=True)
        dn = _interval_state(ctx, theta - step, *rows, np.ones(3), per_row=True)
        fd = (up.row_log_p - dn.row_log_p) / (2.0 * step[k])
        exact = state.mean[k] - base.row_mean[:, k]
        assert np.all(np.abs(fd - exact) <= 1e-6 * (1.0 + np.abs(exact)))


# ---------------------------------------------------------------------------
# Independent reference
# ---------------------------------------------------------------------------


def _row_reference(model, theta, a, b, n=400):
    """Graded composite Gauss-Legendre over one row (doubling self-check)."""
    req, g = model.request, model.geometry
    used = theta != 0.0

    def q_at(x):
        return _evaluate_basis(model, x, max_order=0).phi[:, used] @ theta[used]

    # Window: the row itself, infinite ends grown until q - min q > 80; then
    # cut to where q - min q < 80 (as the normalizer reference does).
    span = 20.0 * g.scale
    while True:
        lo = a if math.isfinite(a) else min(g.center, b) - span
        hi = b if math.isfinite(b) else max(g.center, a) + span
        probe = np.linspace(lo, hi, 40001)
        inner = (
            probe[1:-1]
            if lo in (req.lower, req.upper)
            or hi
            in (
                req.lower,
                req.upper,
            )
            else probe
        )
        q = q_at(inner)
        if (math.isfinite(a) or q[0] - q.min() > 80.0) and (
            math.isfinite(b) or q[-1] - q.min() > 80.0
        ):
            break
        span *= 2.0
    keep = q - q.min() < 80.0
    step = (hi - lo) / 40000.0
    lo = lo if keep[0] else max(inner[keep][0] - step, lo)
    hi = hi if keep[-1] else min(inner[keep][-1] + step, hi)

    def panels(k):
        core = np.linspace(lo, hi, k + 1)
        extra = []
        for end, sign in ((req.lower, 1.0), (req.upper, -1.0)):
            if math.isfinite(end) and (end == lo or end == hi):
                extra.extend(end + sign * (hi - lo) / k * 2.0 ** -np.arange(1, 50))
        return np.unique(np.concatenate([core, extra]))

    out = [_gl_panels(model, theta, panels(k)) for k in (n, 2 * n)]
    assert abs(out[0][0] - out[1][0]) < 1e-13 * max(1.0, abs(out[1][0]))
    return out[1]


@pytest.mark.parametrize("spec", BATTERY, ids=lambda s: "")
def test_rows_match_independent_quadrature(spec):
    model = _model(spec)
    theta = _battery_theta(model)
    if theta is None:
        pytest.skip("no feasible vector")
    ctx = _NormalizerContext(model)
    state = _basis_state(ctx, theta)
    if state.status != "proper":
        pytest.skip("improper vector")
    req, g = model.request, model.geometry
    m = g.center
    rows = [(max(req.lower, m - 0.3 * g.scale), min(req.upper, m + 0.4 * g.scale))]
    rows.append(
        (max(req.lower, m + 0.01 * g.scale), min(req.upper, m + 0.011 * g.scale))
    )
    rows.append((max(req.lower, m + 2.0 * g.scale), req.upper))
    rows.append((req.lower, min(req.upper, m - 2.5 * g.scale)))
    rows = [(a, b) for a, b in rows if a < b]
    lower, upper = zip(*rows, strict=True)
    res = _interval_state(ctx, theta, lower, upper, np.ones(len(rows)), per_row=True)
    for i, (a, b) in enumerate(rows):
        lz, mean, cov = _row_reference(model, theta, a, b)
        lp = lz - state.log_z
        assert res.row_log_p[i] == pytest.approx(lp, abs=1e-11 * max(1.0, abs(lp)))
        scale = np.sqrt(np.abs(np.diag(state.cov))) + np.abs(mean) + 1e-300
        assert np.all(np.abs(res.row_mean[i] - mean) <= 1e-10 * scale)
        assert np.all(np.abs(res.row_cov[i] - cov) <= 1e-9 * np.outer(scale, scale))


# ---------------------------------------------------------------------------
# Tails, failures, cost
# ---------------------------------------------------------------------------


def test_flat_bulk_with_exponential_columns_resolves_its_far_moments():
    # q = b z^2 with zero exponential amplitudes: E[e^{l x}] lives near
    # z = l / 2b, far beyond the mass; the truncation now follows the moment
    # tails (G3-043), E[exp column] ~ 6e7 here.
    model = _model(
        BasisSpec(
            bulk_degree=2,
            lower_tail=Exponential(1.0),
            upper_tail=Exponential(1.0),
            expansion="minimal",
        )
    )
    theta, _ = _theta(model, {"x^2": 0.01})
    state = _basis_state(_NormalizerContext(model), theta)
    g = model.geometry
    panels = np.linspace(g.center - 900.0 * g.scale, g.center + 900.0 * g.scale, 9001)
    ref = _gl_panels(model, theta, panels)
    assert state.log_z == pytest.approx(ref[0], abs=1e-12)
    np.testing.assert_allclose(state.mean, ref[1], rtol=1e-10, atol=1e-12)


def test_an_unresolvable_row_fails_loudly_and_is_named():
    # A point row at a log endpoint has no finite density; the whole-support
    # normalization succeeds and the failure names the offending row.
    model = _model(BasisSpec(support=(0.0, INF), bulk_degree=1, log_lower=True))
    theta, _ = _theta(model, {"logL": 2.0, "x": 1.0})
    ctx = _NormalizerContext(model)
    with pytest.raises(_QuadratureFailure, match="point observation") as info:
        _interval_state(ctx, theta, [1.0, 0.0], [2.0, 0.0], [0.5, 0.5])
    assert info.value.diagnostics["row"] == 1
    assert info.value.diagnostics["interval"] == (0.0, 0.0)


def test_many_rows_cost():
    model = _model(BasisSpec(bulk_degree=4, lower_tail=Exponential(0.7)))
    theta = _fitted(model)
    ctx = _NormalizerContext(model)
    g = model.geometry
    rng = np.random.default_rng(0)
    centre = g.center + g.scale * rng.normal(size=1000)
    width = g.scale * rng.exponential(0.2, size=1000)
    lower = np.where(rng.random(1000) < 0.1, -INF, centre - width)
    upper = np.where(rng.random(1000) < 0.1, INF, centre + width)
    _interval_state(ctx, theta, lower, upper, np.full(1000, 1e-3))
    t0 = time.perf_counter()
    res = _interval_state(ctx, theta, lower, upper, np.full(1000, 1e-3))
    elapsed = time.perf_counter() - t0
    assert math.isfinite(res.log_p)
    assert elapsed < 1.0
