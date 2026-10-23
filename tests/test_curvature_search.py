"""G3 Stage 2c: curvature separation (compact search, refinement, cuts).

Every result is checked against an independent dense scan of ``q''`` on the
returned compact domain: the deepest recorded minimum is at least as deep as
the scan's, every strict scan minimum below the threshold is matched by a
recorded minimum, ``nonnegative`` never coexists with a negative scan value,
and cut rows reproduce ``q''`` at their points.  Planted shallow dips, a tiny
cell budget (fail closed), negative log amplitudes and timing are covered.
"""

import math
import time
import warnings

import numpy as np
import pytest
from test_basis_admissibility import _grid_lp
from test_basis_evaluation import _broad_battery

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._fit.curvature_search import (
    _CurvatureContext,
    _CurvatureOptions,
    _separate_curvature,
)
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis, _potential

RNG = np.random.default_rng(16102026)
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


def _thetas(model):
    out = [RNG.normal(size=model.n_coefficients)]
    feasible = _grid_lp(model, 8.0, 801)
    if feasible is not None:
        scale = np.max(np.abs(feasible))
        out.append(feasible)
        out.append(feasible + 0.05 * scale * RNG.normal(size=feasible.size))
    nonneg = [i for i, c in enumerate(model.columns) if c.nonnegative]
    for t in out:
        t[nonneg] = np.abs(t[nonneg])
    return out


def _check_against_scan(model, theta, res, n=60001):
    a, b = res.domain
    x = np.linspace(a, b, n)
    q2 = _potential(model, theta, x)[2]
    finite = np.isfinite(q2)
    x, q2 = x[finite], q2[finite]
    scale = np.max(np.abs(q2)) + 1.0 / model.geometry.scale**2
    tol = 1e-9 * scale
    dx = (b - a) / (n - 1)
    if res.status == "nonnegative":
        assert q2.min() >= -tol, (q2.min(), x[np.argmin(q2)])
    if q2.min() < -tol:
        assert res.status != "nonnegative"
    if res.unresolved:
        assert res.status != "nonnegative"
    best = min((m.value for m in res.minima), default=np.inf)
    # A positive value below tau at a proven tail cutoff is covered by the tail
    # proof and deliberately not cut (cutting there would chase the tail).
    proven = {t.side for t in res.tails if t.status == "nonnegative"}
    i_min = int(np.argmin(q2))
    at_proven_cutoff = (i_min == 0 and "lower" in proven) or (
        i_min == q2.size - 1 and "upper" in proven
    )
    if q2.min() < res.threshold - tol and not (at_proven_cutoff and q2.min() >= -tol):
        assert best <= q2.min() + tol, (best, q2.min(), x[i_min])
    # Every strict scan minimum below the threshold is matched.
    inner = (q2[1:-1] < q2[:-2] - tol) & (q2[1:-1] < q2[2:] - tol)
    idx = np.flatnonzero(inner) + 1
    for i in idx:
        if q2[i] >= res.threshold - tol:
            continue
        near = [
            m
            for m in res.minima
            if abs(m.x - x[i]) <= 3.0 * dx and m.value <= q2[i] + tol
        ]
        assert near, (x[i], q2[i], [(m.x, m.value) for m in res.minima])
    # Cut rows reproduce q'' at the cut points; recorded enclosures contain values.
    if res.cut_points.size:
        rows = res.cut_rows[: res.cut_points.size]
        np.testing.assert_allclose(
            rows @ theta,
            _potential(model, theta, res.cut_points)[2],
            rtol=1e-12,
            atol=tol,
        )
    for m in res.minima:
        assert m.lower <= m.value <= m.upper


BATTERY = list(_broad_battery())[::2]


@pytest.mark.parametrize("spec", BATTERY, ids=lambda s: "")
def test_search_agrees_with_dense_scan(spec):
    model = _model(spec)
    ctx = _CurvatureContext(model)
    for theta in _thetas(model):
        res = _separate_curvature(ctx, theta)
        _check_against_scan(model, theta, res)


def test_planted_shallow_dip_is_found_and_cut():
    # Ungated quartic bulk: q''(z) h^2 = 2 a2 + 6 a3 z + 12 a4 z^2, minimum at
    # z* = -a3 / (4 a4).  Choose a2 so the minimum sits at tau / 2 > 0.
    model = _model(BasisSpec(bulk_degree=4))
    labels = [c.label for c in model.columns]
    g = model.geometry
    a3, a4 = 0.3, 0.05
    zs = -a3 / (4.0 * a4)
    tau = 1e-3 / g.scale**2
    a2 = 0.5 * (0.5 * tau * g.scale**2 - 6.0 * a3 * zs - 12.0 * a4 * zs**2)
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("x^2")] = a2
    theta[labels.index("x^3")] = a3
    theta[labels.index("x^4")] = a4
    res = _separate_curvature(_CurvatureContext(model), theta)
    _check_against_scan(model, theta, res)
    assert res.status == "nonnegative"  # the dip is positive ...
    assert res.cut_points.size == 1  # ... but below tau, so it is cut
    (m,) = [m for m in res.minima if m.value < tau]
    assert m.value == pytest.approx(0.5 * tau, rel=1e-9)
    assert m.x == pytest.approx(g.center + zs * g.scale, abs=1e-9 * g.scale)
    # Deepen it below zero: now a violation at the same place.
    theta[labels.index("x^2")] -= tau * g.scale**2
    res = _separate_curvature(_CurvatureContext(model), theta)
    _check_against_scan(model, theta, res)
    assert res.status == "violated"
    assert res.cut_points[0] == pytest.approx(
        g.center + zs * g.scale, abs=1e-9 * g.scale
    )


def test_zero_coefficients_are_exactly_nonnegative():
    model = _model(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True, log_upper=True)
    )
    res = _separate_curvature(_CurvatureContext(model), np.zeros(model.n_coefficients))
    assert res.status == "nonnegative"
    assert np.all(np.isfinite(res.cut_rows))


def test_tiny_budget_fails_closed():
    model = _model(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(3), upper_tail=SoftPower(2.5))
    )
    ctx = _CurvatureContext(model)
    opts = _CurvatureOptions(max_cells=8, initial_cells=4)
    seen = 0
    for _ in range(10):
        theta = RNG.normal(size=model.n_coefficients)
        res = _separate_curvature(ctx, theta, opts)
        if res.unresolved:
            seen += 1
            assert res.status != "nonnegative"
    assert seen


def test_negative_log_amplitude_is_a_violation_with_bound_rows():
    model = _model(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True, log_upper=True)
    )
    theta = np.ones(model.n_coefficients)
    col = next(i for i, c in enumerate(model.columns) if c.label == "logL")
    theta[col] = -0.1
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "violated"
    np.testing.assert_array_equal(res.cut_rows, np.eye(model.n_coefficients)[[col]])


def test_tail_violation_adds_leading_functional_rows():
    model = _model(BasisSpec(bulk_degree=2, lower_tail=Exponential(0.7)))
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("expL")] = -1.0
    theta[labels.index("x^2")] = 1.0
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "violated"
    tail_rows = res.cut_rows[res.cut_points.size :]
    assert tail_rows.shape[0] >= 1 and np.all(tail_rows @ theta < 0.0)


def test_weak_positive_leader_extends_search_to_cap():
    # q'' = tiny x^3 leader + negative x^2 term: negative from ~20 to ~1e6 h,
    # beyond the 12 h fallback; the tail cannot be proven before the cap
    # (G3-037), so the finite search must cover the range and find the dip.
    model = _model(
        BasisSpec(
            bulk_degree=3,
            lower_tail=Exponential(0.7),
            upper_tail=Polynomial(3),
            expansion="minimal",
        )
    )
    assert [c.label for c in model.columns] == ["x", "x^2", "x^3", "expL"]
    theta = np.array([1.26, -1.66e-6, 5.5e-13, 2.91])
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert [t.status for t in res.tails] == ["nonnegative", "inconclusive"]
    assert res.status == "violated" and res.cut_points.size >= 1
    g = model.geometry
    assert res.cut_points.max() > g.center + 12.0 * g.scale
    assert np.all(res.cut_rows[: res.cut_points.size] @ theta < 0.0)


def test_cut_rows_are_curvature_rows():
    model = _model(BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)))
    theta = RNG.normal(size=model.n_coefficients)
    res = _separate_curvature(_CurvatureContext(model), theta)
    if res.cut_points.size:
        np.testing.assert_array_equal(
            res.cut_rows[: res.cut_points.size],
            _evaluate_basis(model, res.cut_points).d2phi,
        )


def test_typical_search_time():
    model = _model(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(2.5))
    )
    ctx = _CurvatureContext(model)
    theta = _grid_lp(model, 8.0, 801)
    start = time.perf_counter()
    for _ in range(5):
        _separate_curvature(ctx, theta)
    assert (time.perf_counter() - start) / 5 < 2.0


@pytest.mark.parametrize("support", [(0.0, math.inf), (-math.inf, 0.0)])
def test_exactly_linear_half_line_potential_is_nonnegative(support):
    # q = slope * x has q'' identically zero; exact zeros must stay exact.
    model = _model(BasisSpec(support=support, bulk_degree=2))
    theta = np.zeros(model.n_coefficients)
    theta[[c.label for c in model.columns].index("x")] = (
        1.0 if support[0] == 0.0 else -1.0
    )
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "nonnegative"
