"""Candidate generation is approximate; acceptance and rows remain independent."""

from dataclasses import replace

import numpy as np
import pytest
from gibbus._fit._basis_newton_kernels import ACCEPTED, SHAPE
from gibbus._fit._series_search import SeriesSearchKernel
from test_basis_curvature_rows import QUARTIC, _model, _objective, _theta_for
from test_basis_evaluation import ORACLE_SPECS, _compile

from gibbus._fit import series_search as shared_search
from gibbus._fit.basis_newton import _Blocks, _FaceObjective, _JointCurvature
from gibbus._fit.curvature_search import (
    _CurvatureContext,
    _CurvatureOptions,
    _separate_curvature,
)
from gibbus._fit.series_search import _SeriesCandidateCache
from gibbus._model.basis_eval import _evaluate_basis, _potential
from gibbus._model.basis_series import SeriesPlan


def clustered_tensor():
    """Independent derivative rows for a degree-six polynomial curvature."""
    n = 6
    tensor = np.zeros((1, 3, n + 1, n + 1))
    for j in range(n + 1):
        tensor[0, 0, j, j] = 1.0
        if j:
            tensor[0, 1, j - 1, j] = j
        if j > 1:
            tensor[0, 2, j - 2, j] = j * (j - 1)
    return tensor


def clustered_history(kernel):
    from numpy.polynomial import Polynomial

    result = []
    for scale in (0.6, 0.45, 0.3, 0.2, 0.14, 0.09, 0.06, 0.04):
        roots = 0.125 + scale * np.array([-1.0, 0.0, 1.0])
        q2 = Polynomial.fromroots(np.repeat(roots, 2))
        q2.coef[0] -= 0.02 * scale**6
        points = kernel.candidates(q2.coef)
        hits = sum(bool(np.any(abs(points - root) < 0.01 * scale)) for root in roots)
        result.append(
            dict(scale=scale, roots=roots.tolist(), points=points.tolist(), hits=hits)
        )
    return result


def test_multiple_warm_starts_track_three_minima_in_one_scan_interval():
    kernel = SeriesSearchKernel([[0.0, 1.0]], clustered_tensor(), tolerance=1e-7)
    result = clustered_history(kernel)
    assert all(row["hits"] == 3 for row in result)
    assert all(0.0 < root < 0.25 for root in result[-1]["roots"])
    assert kernel.statistics()["tracked_roots"] >= 3


@pytest.mark.parametrize("index", [14, 18])
def test_cached_scan_rows_are_actual_basis_derivatives(tensor_store, index):
    m = _compile(ORACLE_SPECS[index])
    g = m.geometry
    cache = _SeriesCandidateCache(m, _CurvatureOptions())
    data = cache.prepare((g.center - 4 * g.scale, g.center + 4 * g.scale)).data
    assert data.exact_scan
    values = _evaluate_basis(m, data.scan_points.ravel(), max_order=3)
    shape = data.scan_rows[:, 0].shape
    np.testing.assert_array_equal(data.scan_rows[:, 0], values.d2phi.reshape(shape))
    np.testing.assert_array_equal(data.scan_rows[:, 1], values.d3phi.reshape(shape))


def test_scanning_without_stationary_roots_builds_no_series():
    tensor = np.zeros((1, 3, 1, 1))
    tensor[0, 0, 0, 0] = -0.1
    tensor[0, 1, 0, 0] = 1.0
    kernel = SeriesSearchKernel([[0.0, 1.0]], tensor)
    assert kernel.candidates(np.ones(1)).size == 1
    stats = kernel.statistics()
    assert stats["scan_nodes"] == 9
    assert stats["q2_builds"] == stats["q3_builds"] == stats["q4_builds"] == 0


def test_positive_series_bound_avoids_higher_derivative_work():
    tensor = np.zeros((1, 3, 3, 1))
    tensor[0, 0, :, 0] = [2.0, 0.0, 1.0]
    tensor[0, 1, :, 0] = [0.0, 2.0, 0.0]
    tensor[0, 2, 0, 0] = 2.0
    kernel = SeriesSearchKernel([[0.0, 1.0]], tensor)
    assert kernel.candidates(np.ones(1)).size == 0
    stats = kernel.statistics()
    assert stats["q2_builds"] == 1 and stats["q3_builds"] == stats["q4_builds"] == 0


def test_exact_sample_survives_misleading_positive_series():
    from gibbus._fit._series_search import SeriesSearchData

    tensor = np.ones((1, 3, 1, 1))
    rows = np.zeros((1, 2, 9, 1))
    rows[:, 0] = -0.1
    data = SeriesSearchData([[0.0, 1.0]], tensor, scan_rows=rows)
    kernel = SeriesSearchKernel(data)
    assert kernel.candidates(np.ones(1)).size
    assert kernel.statistics()["q2_builds"] == 0


def test_refined_minimum_suppresses_redundant_sampled_cut():
    tensor = np.zeros((1, 3, 3, 1))
    tensor[0, 0, :, 0] = [0.13**2 - 0.2, -0.26, 1.0]
    tensor[0, 1, :, 0] = [-0.26, 2.0, 0.0]
    tensor[0, 2, 0, 0] = 2.0
    kernel = SeriesSearchKernel([[0.0, 1.0]], tensor)
    points = kernel.candidates(np.ones(1))
    assert points.size == 1 and abs(points[0] - 0.13) < 1e-5


def test_lower_exact_sample_is_kept_beside_a_refined_root():
    from gibbus._fit._series_search import SeriesSearchData

    tensor = np.zeros((1, 3, 3, 1))
    tensor[0, 0, :, 0] = [-0.1, 0.0, 1.0]
    tensor[0, 1, :, 0] = [0.0, 2.0, 0.0]
    tensor[0, 2, 0, 0] = 2.0
    rows = np.zeros((1, 2, 9, 1))
    rows[:, 0] = -0.5
    rows[0, 1, :, 0] = 2 * np.linspace(-1.0, 1.0, 9)
    data = SeriesSearchData([[0.0, 1.0]], tensor, scan_rows=rows)
    points = SeriesSearchKernel(data).candidates(np.ones(1))
    assert np.any(abs(points) < 1e-5) and np.any(points == -1.0)


@pytest.fixture
def tensor_store(monkeypatch):
    store = shared_search._SeriesTensorStore()
    monkeypatch.setattr(shared_search, "_TENSOR_STORE", store)
    return store


def test_retained_tensor_skips_construction_for_independent_model(
    tensor_store, monkeypatch
):
    m = _model(QUARTIC)
    g = m.geometry
    o = _CurvatureOptions()
    domain = (g.center - 4 * g.scale, g.center + 4 * g.scale)
    first = _SeriesCandidateCache(m, o)
    a = first.prepare(domain)
    duplicate = _model(QUARTIC)
    assert duplicate is not m

    def unexpected(*args, **kwargs):
        raise AssertionError("a retained tensor must not reconstruct primitive jets")

    monkeypatch.setattr(SeriesPlan, "cache", unexpected)
    second = _SeriesCandidateCache(duplicate, o)
    b = second.prepare(domain)
    assert b is not a and b.data is a.data
    assert first.tensor_builds == 1 and second.tensor_builds == 0
    assert second.tensor_reuses == 1 and tensor_store.info()["hits"] == 1
    assert "series_plan" not in duplicate.__dict__
    theta = _theta_for(m, -0.01)
    np.testing.assert_array_equal(a.candidates(theta), b.candidates(theta))
    assert a.statistics()["calls"] == b.statistics()["calls"] == 1
    b.candidates(theta)
    assert a.statistics()["calls"] == 1 and b.statistics()["calls"] == 2


def test_shared_table_is_an_immutable_snapshot():
    from gibbus._fit._series_search import SeriesSearchData

    geometry = np.array([[0.0, 1.0]])
    tensor = np.ones((1, 3, 3, 1))
    data = SeriesSearchData(geometry, tensor)
    geometry[:] = 99.0
    tensor[:] = 99.0
    np.testing.assert_array_equal(data.geometry, [[0.0, 1.0]])
    np.testing.assert_array_equal(data.tensor, np.ones((1, 3, 3, 1)))
    for array in (data.geometry, data.tensor, data.scan_points, data.scan_rows):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    # Read-only data still permits changing a returned view's shape metadata.
    # Such changes must never alter the geometry used by another kernel.
    exposed = data.tensor
    exposed.shape = (exposed.size,)
    assert data.tensor.shape == (1, 3, 3, 1)
    a = SeriesSearchKernel(data)
    b = SeriesSearchKernel(data)
    assert a.data is b.data is data


@pytest.mark.parametrize(
    "change",
    [
        "center",
        "gate",
        "soft",
        "exponential",
        "map",
        "domain",
        "order",
        "budget",
        "samples",
    ],
)
def test_changed_geometry_cannot_reuse_a_tensor(tensor_store, change):
    # Perturb a single binary64 parameter where possible: lookup never rounds
    # geometries together. Exact rational maps are also part of the key.
    from fractions import Fraction

    m = _compile(ORACLE_SPECS[18] if change == "exponential" else ORACLE_SPECS[14])
    g = m.geometry
    o = _CurvatureOptions(series_max_panels=256)
    domain = (g.center - 4 * g.scale, g.center + 4 * g.scale)
    first = _SeriesCandidateCache(m, o)
    a = first.prepare(domain)
    assert a is not None, first.reason
    other = m
    options = o
    other_domain = domain
    if change == "center":
        coord = replace(g.coordinate, center=np.nextafter(g.center, np.inf))
        other = replace(m, geometry=replace(g, coordinate=coord))
    elif change == "gate":
        other = replace(
            m,
            geometry=replace(
                g, lower_steepness=np.nextafter(g.lower_steepness, np.inf)
            ),
        )
    elif change in ("soft", "exponential"):
        functions = list(m.functions)
        j = next(
            i
            for i, f in enumerate(functions)
            if f.kind == ("soft_power" if change == "soft" else "exponential")
        )
        field = "delta" if change == "soft" else "rate"
        functions[j] = replace(
            functions[j], **{field: np.nextafter(getattr(functions[j], field), np.inf)}
        )
        other = replace(m, functions=tuple(functions))
    elif change == "map":
        rows = [list(row) for row in m.coefficient_map_exact]
        i, j = next(
            (i, j) for i, row in enumerate(rows) for j, v in enumerate(row) if v
        )
        rows[i][j] += Fraction(1, 2**50)
        exact = tuple(tuple(row) for row in rows)
        other = replace(
            m, coefficient_map_exact=exact, coefficient_map=np.array(exact, dtype=float)
        )
    elif change == "domain":
        other_domain = (domain[0] - 0.1 * g.scale, domain[1])
    elif change == "order":
        options = replace(o, series_order=o.series_order + 1)
    elif change == "budget":
        options = replace(o, series_max_panels=o.series_max_panels + 1)
    else:
        options = replace(o, series_samples=o.series_samples + 1)
    second = _SeriesCandidateCache(other, options)
    b = second.prepare(other_domain)
    assert b is not None, second.reason
    assert b.data is not a.data and second.tensor_reuses == 0


def test_runtime_options_use_private_workspaces(tensor_store):
    m = _model(QUARTIC)
    g = m.geometry
    o = _CurvatureOptions()
    domain = (g.center - 4 * g.scale, g.center + 4 * g.scale)
    first = _SeriesCandidateCache(m, o)
    a = first.prepare(domain)
    second = _SeriesCandidateCache(
        m, replace(o, series_steps=3, series_tolerance=2e-5, rho=0.5, initial_cells=16)
    )
    b = second.prepare(domain)
    assert b is not a and a.data is b.data and second.tensor_reuses == 1
    assert b.candidates(_theta_for(m, -0.01)).size
    assert a.statistics()["calls"] == 0


def test_eviction_and_clear_preserve_live_searches(tensor_store):
    tensor_store.max_entries = 1
    m = _model(QUARTIC)
    g = m.geometry
    o = _CurvatureOptions()
    first = _SeriesCandidateCache(m, o)
    a = first.prepare((g.center - 4 * g.scale, g.center + 4 * g.scale))
    other = _SeriesCandidateCache(m, o)
    b = other.prepare((g.center - 5 * g.scale, g.center + 5 * g.scale))
    assert tensor_store.info()["entries"] == 1 and tensor_store.info()["evictions"] == 1
    theta = _theta_for(m, -0.01)
    before = a.candidates(theta)
    shared_search._clear_series_tensor_cache()
    assert tensor_store.info()["entries"] == 0
    np.testing.assert_array_equal(a.candidates(theta), before)
    assert b.candidates(theta).size
    tensor_store.max_bytes = 1
    again = _SeriesCandidateCache(m, o)
    assert again.prepare((g.center - 4 * g.scale, g.center + 4 * g.scale)) is not None
    assert tensor_store.info()["entries"] == 0 and tensor_store.info()["bytes"] == 0


def test_shared_tensors_are_safe_for_concurrent_searches(tensor_store):
    from concurrent.futures import ThreadPoolExecutor

    m = _model(QUARTIC)
    g = m.geometry
    o = _CurvatureOptions()
    domain = (g.center - 4 * g.scale, g.center + 4 * g.scale)
    seed = _SeriesCandidateCache(m, o)
    data = seed.prepare(domain).data
    theta = _theta_for(m, -0.01)

    def run(_):
        cache = _SeriesCandidateCache(_model(QUARTIC), o)
        kernel = cache.prepare(domain)
        assert kernel.data is data
        return kernel.candidates(theta), kernel.statistics()

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, range(12)))
    for points, stats in results:
        np.testing.assert_array_equal(points, results[0][0])
        assert stats["calls"] == 1 and stats["replays"] == 0


def test_concurrent_cold_publication_selects_one_table(tensor_store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(4)

    def prepare(_):
        m = _model(QUARTIC)
        g = m.geometry
        cache = _SeriesCandidateCache(m, _CurvatureOptions())
        barrier.wait()
        return cache.prepare((g.center - 4 * g.scale, g.center + 4 * g.scale))

    with ThreadPoolExecutor(4) as pool:
        kernels = list(pool.map(prepare, range(4)))
    assert all(k.data is kernels[0].data for k in kernels)
    assert tensor_store.info()["entries"] == 1


def test_two_minima_and_cache_reuse():
    m = _model(QUARTIC)
    g = m.geometry
    o = _CurvatureOptions()
    cache = _SeriesCandidateCache(m, o)
    domain = (g.center - 4 * g.scale, g.center + 4 * g.scale)
    kernel = cache.prepare(domain)
    assert kernel is not None, cache.reason
    other = _SeriesCandidateCache(m, o)
    assert other.prepare(domain) is not kernel
    assert other.plan is cache.plan is m.series_plan
    assert other.kernel.statistics()["calls"] == 0
    for eps in (-0.01, -0.02, -0.001):
        theta = _theta_for(m, eps)
        points = kernel.candidates(theta)
        z = (points - g.center) / g.scale
        assert np.min(abs(z - 1)) < 2e-5
        assert np.min(abs(z + 1)) < 2e-5
        assert np.min(_potential(m, theta, points, max_order=2)[2]) < 0
    assert cache.prepare(domain) is kernel
    assert cache.prepare((domain[0] + 0.1, domain[1] - 0.1)) is kernel
    assert cache.builds == 1 and kernel.statistics()["calls"] == 3
    assert cache.prepare((domain[0] - 1, domain[1] + 1)) is not kernel
    assert cache.builds == 2


def make_face(enabled):
    m = _model(QUARTIC)
    d = m.n_coefficients
    obj = _FaceObjective(
        _objective(m),
        np.eye(d),
        _JointCurvature(_Blocks.single(m)),
        _CurvatureOptions(series_candidates=enabled),
    ).compiled()
    convex = _theta_for(m, 1.0)
    domain = (_separate_curvature(_CurvatureContext(m), convex).domain,)
    obj.begin(np.zeros((0, d)), domain)
    return m, obj, convex, domain


def test_real_newton_gate_uses_cache_and_actual_rows():
    m, obj, convex, domain = make_face(True)
    bad = _theta_for(m, -0.01)
    assert obj.trial(bad) == (SHAPE, None)
    stats = obj.series_statistics()
    assert stats["hits"] == 1 and stats["full_searches"] == 0
    assert stats["blocks"][0]["builds"] == 1
    pending = obj.pending()
    rows = pending[1][: pending[2]]
    # These rows are actual Phi'', not Taylor coefficients.
    # Polynomial minima are at c +/- h, up to the heuristic root tolerance.
    exact = _evaluate_basis(
        m, m.geometry.center + m.geometry.scale * np.array([-1.0, 1.0])
    ).d2phi
    for row in exact:
        assert np.min(np.linalg.norm(rows - row, axis=1)) < 2e-4
    assert np.all(rows @ bad < 0)
    assert obj.trial(convex)[0] == ACCEPTED
    assert obj.series_statistics()["full_searches"] == 1
    obj.end()
    obj.begin(np.zeros((0, m.n_coefficients)), domain)
    assert obj.series_statistics()["blocks"][0]["builds"] == 1


def test_disabled_path_and_budget_fallback():
    m, obj, convex, _ = make_face(False)
    assert obj.trial(_theta_for(m, -0.01))[0] == SHAPE
    assert obj.series_statistics()["blocks"] == [None]
    cache = _SeriesCandidateCache(m, _CurvatureOptions(series_max_panels=1))
    assert (
        cache.prepare(
            (
                m.geometry.center - 4 * m.geometry.scale,
                m.geometry.center + 4 * m.geometry.scale,
            )
        )
        is None
    )
    assert cache.reason == "panel budget"


@pytest.mark.parametrize("index", [0, 2, 8, 14, 18, 22])
def test_candidate_violations_against_dense_scan(index):
    m = _compile(ORACLE_SPECS[index])
    g = m.geometry
    lo = max(m.request.lower + 0.001 * g.scale, g.center - 4 * g.scale)
    hi = min(m.request.upper - 0.001 * g.scale, g.center + 4 * g.scale)
    cache = _SeriesCandidateCache(m, _CurvatureOptions(series_max_panels=256))
    k = cache.prepare((lo, hi))
    assert k is not None, cache.reason
    grid = np.linspace(lo, hi, 2001)
    rows = _evaluate_basis(m, grid, max_order=2).d2phi
    rng = np.random.default_rng(928 + index)
    checked = 0
    for _ in range(30):
        theta = rng.normal(size=m.n_coefficients)
        # Structural log amplitudes must be nonnegative in a proper trial.
        for j, col in enumerate(m.columns):
            if col.nonnegative:
                theta[j] = abs(theta[j])
        values = rows @ theta
        if np.min(values) > -0.01 / (g.scale * g.scale):
            continue
        pts = k.candidates(theta)
        assert pts.size
        actual = _potential(m, theta, pts, max_order=2)[2]
        assert np.min(actual) < 0
        checked += 1
    assert checked > 0


def test_numerical_failure_returns_to_direct_path():
    # The C kernel signals range failure instead of claiming no minima.
    tensor = np.full((1, 3, 3, 1), 1e100)
    k = SeriesSearchKernel(np.array([[0.0, 1.0]]), tensor)
    with pytest.raises(FloatingPointError):
        k.candidates(np.array([1e308]))
    assert k.statistics()["failures"] == 1


def test_short_scan_retains_warm_start_and_caller_order():
    # q''=(x-r)^2-.01, derivative rows independently specified.
    a = np.zeros((1, 3, 3, 2))
    a[0, 0, :, 0] = [0.24, -1.0, 1.0]
    a[0, 1, :, 0] = [-1.0, 2.0, 0.0]
    a[0, 2, 0, 0] = 2.0
    a[0, 0, 0, 1] = 1.0
    k = SeriesSearchKernel([[0.0, 1.0]], a, samples=2)
    p = k.candidates(np.array([1.0, 0.0]))
    assert np.min(abs(p - 0.5)) < 1e-5
    p = k.candidates(np.array([1.0, -0.01]))
    assert np.min(abs(p - 0.5)) < 1e-5


def test_missed_heuristic_violation_cannot_accept_trial(monkeypatch):
    m, obj, convex, domain = make_face(True)
    bad = _theta_for(m, -0.01)
    fake = np.zeros((1, 3, 1, m.n_coefficients))
    fake[0, 0, 0] = bad / np.dot(bad, bad)  # approximate q''=+1: no candidate
    kernel = SeriesSearchKernel([[m.geometry.center, m.geometry.scale]], fake)
    monkeypatch.setattr(_SeriesCandidateCache, "prepare", lambda self, domain: kernel)
    obj.end()
    obj.begin(np.zeros((0, m.n_coefficients)), domain)
    assert obj.trial(bad) == (SHAPE, None)
    assert obj.series_statistics()["hits"] == 0
    assert obj.series_statistics()["full_searches"] == 1
    assert obj.pending() is not None


def test_stacked_fit_preserves_certification_and_fit_quality():
    from test_basis_mixture import TWO, _data, _mixture_blocks, _model, _mstep_objective

    from gibbus import BasisSpec
    from gibbus._fit.basis_newton import _solve_basis

    model = _model(BasisSpec(bulk_degree=4), TWO)
    data = _data("points", TWO)
    resp = np.column_stack([TWO < 0.5, TWO >= 0.5]).astype(float) * 0.9 + 0.05
    objective = _mstep_objective((model,) * 2, data, resp)
    result = _solve_basis(
        objective,
        blocks=_mixture_blocks((model,) * 2),
        curvature_options=_CurvatureOptions(series_candidates=True),
    )
    baseline = _solve_basis(
        _mstep_objective((model,) * 2, data, resp),
        blocks=_mixture_blocks((model,) * 2),
        curvature_options=_CurvatureOptions(series_candidates=False),
    )
    assert result.status == baseline.status or result.status in (
        "converged",
        "converged_approximately",
    )
    assert result.final_separation.status == "nonnegative"
    assert result.objective_value <= baseline.objective_value + 1e-8


def test_near_contact_uses_refined_search():
    m, obj, convex, domain = make_face(True)
    near = _theta_for(m, -1e-11)
    assert obj.trial(near) == (SHAPE, None)
    stats = obj.series_statistics()
    assert stats["hits"] == 0 and stats["full_searches"] == 1


def test_gate_falls_back_on_series_overflow(monkeypatch):
    m, obj, _, domain = make_face(True)
    bad = 100 * _theta_for(m, -0.01)
    fake = np.empty((1, 3, 1, m.n_coefficients))
    fake[:] = np.sign(bad) * 1e308
    kernel = SeriesSearchKernel([[m.geometry.center, m.geometry.scale]], fake)
    monkeypatch.setattr(_SeriesCandidateCache, "prepare", lambda self, domain: kernel)
    obj.end()
    obj.begin(np.zeros((0, m.n_coefficients)), domain)
    assert obj.trial(bad) == (SHAPE, None)
    assert kernel.statistics()["failures"] == 1
    assert obj.series_statistics()["hits"] == 0
    assert obj.pending() is not None
