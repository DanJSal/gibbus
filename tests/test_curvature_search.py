"""Curvature separation: panel search, verdicts and cut rows.

Every result is checked against an independent dense scan of the scaled
curvature ``H`` (``curvature``, itself checked against the high-precision
oracle) over the searched domain: every scan point below ``-2 tau`` lies in a
returned negative interval, every returned minimum is a point where ``H`` is
below ``-tau``, ``verified`` never coexists with a scan value below ``-2 tau``,
and the cut rows are the normalized column curvatures at their points.
Planted dips on both sides of the threshold, a starved resolution budget
(fail closed), negative log amplitudes, violated and weak tails and timing are
covered.
"""

import dataclasses
import math
import time
import warnings

import numpy as np
import pytest
from gibbus._model._basis_kernels import curvature, curvature_atom_rows
from test_basis_admissibility import _grid_lp
from test_basis_evaluation import _broad_battery

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._defaults import CURVATURE_SEARCH_OPTIONS
from gibbus._fit.curvature_search import _CurvatureContext, _separate_curvature
from gibbus._model.basis import _build_basis

RNG = np.random.default_rng(16102026)
NORMAL = RNG.normal(1.5, 2.0, size=400)
HALF = RNG.gamma(2.0, 1.5, size=400) + 0.25
BOUNDED = RNG.beta(2.0, 3.0, size=400) * 4.0 - 1.0
TAU = CURVATURE_SEARCH_OPTIONS.tau


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
    out = [RNG.normal(size=model.n_coefficients) for _ in range(2)]
    try:
        feasible = _grid_lp(model, 8.0, 801)
    except ValueError:  # the grid helper overflows for very steep exponentials
        feasible = None
    if feasible is not None:
        scale = np.max(np.abs(feasible))
        out.append(feasible)
        out.append(feasible + 0.05 * scale * RNG.normal(size=feasible.size))
    nonneg = [i for i, c in enumerate(model.columns) if c.nonnegative]
    for t in out:
        t[nonneg] = np.abs(t[nonneg])
    return out


def _canonical(model, x):
    g = model.geometry.coordinate
    return (np.asarray(x, dtype=float) - g.center) / g.scale


def _scan_points(model, res):
    """Dense canonical grid over the searched domain (both ends included)."""
    lo, hi = _canonical(model, res.domain)
    lo, hi = max(lo, -1e6), min(hi, 1e6)
    z = [np.linspace(max(lo, -16.0), min(hi, 16.0), 40001)]
    for end, sign in ((lo, 1.0), (hi, -1.0)):
        if abs(end) > 16.0:
            z.append(sign * -np.geomspace(16.0, abs(end), 4001))
        z.append(end + sign * np.geomspace(1e-9, 1.0, 2001))
    z = np.unique(np.concatenate(z))
    a, b = model.geometry.coordinate.canonical_support
    return z[(z > a) & (z < b) & (z >= lo) & (z <= hi)]


def _covered(z, intervals):
    inside = np.zeros(z.size, dtype=bool)
    for z_lo, z_hi, *_ in intervals:
        inside |= (z >= z_lo) & (z <= z_hi)
    return inside


def _check_against_scan(model, theta, res):
    if res.status == "violated" and not res.intervals and not res.tails:
        # A negative log amplitude: the bound rows alone.
        assert res.cut_rows.shape[0] >= 1
        return
    z = _scan_points(model, res)
    h, err, _ = curvature(model.kernel, theta, z)
    deep = h < -2.0 * TAU - 4.0 * err
    missed = deep & ~_covered(z, res.intervals)
    assert not missed.any(), (z[missed][:5], h[missed][:5], res.intervals)
    if res.status == "verified":
        assert not deep.any()
        assert res.unresolved == 0 and not res.intervals
    if res.unresolved:
        assert res.status != "verified"
    # Each interval's minimum is a genuine point below -tau.
    for z_lo, z_hi, z_min, h_min, to_inf, _ in res.intervals:
        assert z_lo <= z_min <= z_hi and h_min < -TAU
        if math.isfinite(z_min) and not to_inf:
            (hm,), (em,), _ = curvature(model.kernel, theta, np.array([z_min]))
            assert hm < -TAU + 4.0 * em + 1e-12, (z_min, hm, h_min)
            assert abs(hm - h_min) <= 1e-6 * max(1.0, abs(h_min)), (hm, h_min)
    for m in res.minima:
        assert m.value < -TAU
    _check_cut_rows(model, theta, res)


def _check_cut_rows(model, theta, res):
    k = res.cut_points.size
    if not k:
        return
    z = _canonical(model, res.cut_points)
    rows, _, _, _, _ = curvature_atom_rows(model.kernel, z)
    columns = rows @ model.kernel.atom_rows
    columns /= np.max(np.abs(columns), axis=1, keepdims=True)
    # The points round-trip through user units: agreement to rounding.
    np.testing.assert_allclose(res.cut_rows[:k], columns, rtol=1e-9, atol=1e-12)
    assert np.allclose(np.max(np.abs(res.cut_rows), axis=1), 1.0)
    h, err, _ = curvature(model.kernel, theta, z)
    sure = np.abs(h) > 4 * err
    assert np.all(np.sign(res.cut_rows[:k] @ theta)[sure] == np.sign(h[sure]))


BATTERY = list(_broad_battery())
EXTRA = [
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.01), upper_tail=SoftPower(40.5)),
    BasisSpec(
        bulk_degree=2, lower_tail=Exponential(1e-6), upper_tail=Exponential(50.0)
    ),
    BasisSpec(
        support=(0.0, math.inf),
        bulk_degree=2,
        log_lower=True,
        upper_tail=SoftPower(1.2),
    ),
    BasisSpec(
        support=(-math.inf, 0.0),
        bulk_degree=2,
        log_upper=True,
        lower_tail=Exponential(2.0),
    ),
]


@pytest.mark.parametrize("spec", BATTERY + EXTRA, ids=lambda s: "")
def test_search_agrees_with_dense_scan(spec):
    model = _model(spec)
    ctx = _CurvatureContext(model)
    for theta in _thetas(model):
        res = _separate_curvature(ctx, theta)
        _check_against_scan(model, theta, res)


def _quartic_with_minimum(model, value):
    """Ungated quartic bulk whose ``H`` has its minimum ``value`` near ``z*``."""
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("z^3")] = 0.3
    theta[labels.index("z^4")] = 0.05
    zs = -0.3 / (4.0 * 0.05)
    col = labels.index("z^2")

    def h_min(a2):
        theta[col] = a2
        z = np.linspace(zs - 1.0, zs + 1.0, 20001)
        return curvature(model.kernel, theta, z)[0].min()

    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if h_min(mid) < value:
            lo = mid
        else:
            hi = mid
    theta[col] = 0.5 * (lo + hi)
    return theta, zs


def test_planted_dips_on_both_sides_of_the_threshold():
    model = _model(BasisSpec(bulk_degree=4))
    ctx = _CurvatureContext(model)
    # Negative but within the threshold: verified, nothing to cut.
    theta, _ = _quartic_with_minimum(model, -0.5 * TAU)
    res = _separate_curvature(ctx, theta)
    _check_against_scan(model, theta, res)
    assert res.status == "verified" and res.cut_rows.shape[0] == 0
    # Beyond it: violated, cut at the minimum first.
    theta, zs = _quartic_with_minimum(model, -10.0 * TAU)
    res = _separate_curvature(ctx, theta)
    _check_against_scan(model, theta, res)
    assert res.status == "violated" and len(res.intervals) == 1
    (m,) = res.minima
    assert m.z == pytest.approx(zs, abs=1e-3)
    assert m.value == pytest.approx(-10.0 * TAU, rel=1e-3)
    assert _canonical(model, res.cut_points[0]) == pytest.approx(m.z, abs=1e-12)
    assert res.cut_points.size == 1 + CURVATURE_SEARCH_OPTIONS.n_extra


def test_zero_coefficients_are_verified():
    model = _model(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True, log_upper=True)
    )
    res = _separate_curvature(_CurvatureContext(model), np.zeros(model.n_coefficients))
    assert res.status == "verified"
    assert res.cut_rows.shape == (0, model.n_coefficients)


def test_starved_resolution_fails_closed():
    model = _model(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(3), upper_tail=SoftPower(2.5))
    )
    opts = dataclasses.replace(CURVATURE_SEARCH_OPTIONS, degree=2, max_depth=1)
    ctx = _CurvatureContext(model, opts)
    seen = 0
    for _ in range(10):
        theta = RNG.normal(size=model.n_coefficients)
        res = _separate_curvature(ctx, theta)
        if res.unresolved:
            seen += 1
            assert res.status != "verified"
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
    # The column expL*(-1+tL) is negative: a positive coefficient is concave.
    theta[next(i for i, s in enumerate(labels) if s.startswith("expL"))] = 1.0
    theta[labels.index("z^2")] = 1.0
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "violated"
    assert [(t.side, t.status) for t in res.tails][0] == ("lower", "violated")
    assert any(iv[4] for iv in res.intervals)  # an interval to infinity
    tail_rows = res.cut_rows[res.cut_points.size :]
    assert tail_rows.shape[0] == 1 and np.all(tail_rows @ theta < 0.0)
    np.testing.assert_allclose(np.max(np.abs(tail_rows), axis=1), 1.0)


def test_weak_positive_leader_extends_search_to_its_cutoff():
    # A tiny cubic leader against a negative quadratic: q'' < 0 from about 1
    # to about 1e6 canonical units, far beyond the fallback range.  The tail is
    # proven only beyond that, and the finite search covers everything inside.
    model = _model(
        BasisSpec(
            bulk_degree=3,
            lower_tail=Exponential(0.7),
            upper_tail=Polynomial(3),
            expansion="minimal",
        )
    )
    labels = [c.label for c in model.columns]
    assert labels[:3] == ["z", "z^2", "z^3"]
    theta = np.array([1.26, -1.66e-6, 5.5e-13, -2.91])
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert [t.status for t in res.tails] == ["verified", "verified"]
    assert res.tails[1].zeta > 1e6
    assert res.status == "violated" and res.cut_points.size >= 1
    (interval,) = res.intervals
    assert interval[1] == pytest.approx(-2 * theta[1] / (6 * theta[2]), rel=1e-3)
    assert (
        _canonical(model, res.cut_points).max() > CURVATURE_SEARCH_OPTIONS.fallback_zeta
    )
    assert np.all(res.cut_rows[: res.cut_points.size] @ theta < 0.0)
    _check_against_scan(model, theta, res)


def test_panel_rows_are_cached_across_coefficients():
    model = _model(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(2.5))
    )
    ctx = _CurvatureContext(model)
    theta = _grid_lp(model, 8.0, 801)
    _separate_curvature(ctx, theta)
    points, _, panels = ctx.search.statistics
    _separate_curvature(ctx, theta)
    assert ctx.search.statistics[0] == points and ctx.search.statistics[2] == panels
    leaves = [p for p in ctx.search.panels if p[5]]
    assert leaves and all(p[3] < p[4] for p in leaves)


def test_typical_search_time():
    model = _model(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(2.5))
    )
    ctx = _CurvatureContext(model)
    theta = _grid_lp(model, 8.0, 801)
    _separate_curvature(ctx, theta)
    start = time.perf_counter()
    for _ in range(20):
        _separate_curvature(ctx, theta)
    assert (time.perf_counter() - start) / 20 < 0.02


@pytest.mark.parametrize("support", [(0.0, math.inf), (-math.inf, 0.0)])
def test_exactly_linear_half_line_potential_is_verified(support):
    # q = slope * z has q'' identically zero; exact zeros must stay exact.
    model = _model(BasisSpec(support=support, bulk_degree=2))
    theta = np.zeros(model.n_coefficients)
    theta[[c.label for c in model.columns].index("z")] = (
        1.0 if support[0] == 0.0 else -1.0
    )
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "verified"


def test_cancelled_leader_resolves_without_runaway_splitting():
    # On a face whose upper leading functional cancels (theta orthogonal to
    # it up to rounding), the far curvature is set by atoms hundreds of
    # e-folds below the cancelled one: beyond the rows' dynamic range, where
    # panels are evaluated directly, and the rounding residue counts as zero
    # whatever the summation order.  This once split one shell 450,000 times.
    model = _model(
        BasisSpec(bulk_degree=2, lower_tail=Exponential(1.0), expansion="full")
    )
    upper = model.side_asymptotics("upper")
    lead = np.array([float(v) for v in upper.leading.row_exact])
    rng = np.random.default_rng(7)
    ctx = _CurvatureContext(model)
    for _ in range(5):
        theta = 0.1 * rng.normal(size=model.n_coefficients)
        theta -= (lead @ theta) / (lead @ lead) * lead
        res = _separate_curvature(ctx, theta)
        assert res.unresolved == 0 and res.panels < 5000
        assert res.status != "verified" or not res.intervals


def test_tiny_rows_do_not_underflow_the_norm():
    # q = 8e-19 z^12 near the origin: each row is ~1e-145 and its square in
    # the norm underflowed, turning H = 1 into noise and 65,000 splits.
    model = _model(BasisSpec(bulk_degree=12, expansion="minimal"))
    theta = np.zeros(model.n_coefficients)
    theta[[c.label for c in model.columns].index("z^12")] = 8.2e-19
    res = _separate_curvature(_CurvatureContext(model), theta)
    assert res.status == "verified" and res.panels < 1000


def test_cached_hierarchy_is_dropped_beyond_its_budget():
    model = _model(BasisSpec(bulk_degree=4))
    opts = dataclasses.replace(CURVATURE_SEARCH_OPTIONS, max_cached_panels=1)
    ctx = _CurvatureContext(model, opts)
    rng = np.random.default_rng(3)
    for _ in range(20):
        theta = rng.normal(size=model.n_coefficients)
        res = _separate_curvature(ctx, theta)
        _check_against_scan(model, theta, res)
    assert ctx.search.resets > 0
