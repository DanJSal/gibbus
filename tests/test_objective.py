"""Definition-level tests for point and interval likelihood geometry."""
import numpy as np
import pytest
from scipy.integrate import quad

import gibbus._observations.intervals as interval_observations
from gibbus._fit.natural_objective import (
    _natural_interval_start,
    _natural_point_stats,
    _prepare_natural_interval_objective,
)
from gibbus._fit.objective import _evaluate_interval_objective, _evaluate_point_objective
from gibbus._model.coords import _build_fit_coordinate, _build_interval_fit_coordinate
from gibbus._model.natural_state import _NaturalCoreState
from gibbus._model.spec import _build_model_spec
from gibbus._model.vec import _q_eval
from gibbus._observations.empirical import _EmpiricalStats
from gibbus._observations.intervals import (
    _build_interval_observations,
    _prepare_partial_interval_reducer,
)
from gibbus._observations.points import _PointObservations


def _build_empirical_stats(z, weights, max_order, support, /, *,
                           has_lower_log=False, has_upper_log=False):
    return _natural_point_stats(
        z, weights, max_order, support, has_lower_log, has_upper_log
    )


def _state(spec, params, bounds, /):
    return _NaturalCoreState(spec.coordinate, spec.layout, params, bounds)


def _point_objective(spec, observations, bounds, /):
    return lambda params: _evaluate_point_objective(
        _state(spec, params, bounds), observations
    )


def _interval_objective(spec, observations, bounds, /):
    return lambda params: _evaluate_interval_objective(
        _state(spec, params, bounds), observations
    )


def _natural_params(spec, gamma, curvature, lower=None, upper=None, /):
    """Pack natural parameters, amplitudes given on the physical sides."""
    params = np.zeros(spec.n_params)
    params[spec.layout.gamma_index] = gamma
    c = np.zeros(spec.layout.curvature_degree + 1)
    c[: len(curvature)] = curvature
    params[spec.layout.curvature_slice] = c
    if lower is not None:
        params[spec.physical_lower_a_index] = lower
    if upper is not None:
        params[spec.physical_upper_a_index] = upper
    return params


def _problem(boundaries=False):
    """Build a bounded randomized point-objective fixture."""
    x = np.array([-1.7, -0.8, -0.1, 0.25, 0.9, 1.8, 2.5])
    weights = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 0.75])
    coord = _build_fit_coordinate((-2.0, 3.0), x, weights, None)
    z = coord.to_canonical(x)
    spec = _build_model_spec(coord, 4, boundaries, boundaries)
    params = np.array([0.12, 0.75, 0.23, 0.4] + ([0.45, 0.65] if boundaries else []))
    stats = _build_empirical_stats(
        z,
        weights,
        10,
        spec.support,
        has_lower_log=spec.canonical_lower_a_index is not None,
        has_upper_log=spec.canonical_upper_a_index is not None,
    )
    obs = _PointObservations(stats)
    state = _state(spec, params, (float(z.min()), float(z.max())))
    return x, z, weights / weights.sum(), spec, params, obs, state


def test_point_nll_and_gradient_match_direct_raw_evaluation_without_raw_hot_path():
    _, z, weights, spec, _, obs, state = _problem(boundaries=True)
    got = _evaluate_point_objective(state, obs)

    q_values = _q_eval(z, spec.support, state.q_poly, state.boundary_amplitudes, 0)
    expected_nll = (
        float(np.dot(weights, q_values))
        + state.log_Z
        + np.log(spec.coordinate.scale)
    )
    expected_empirical_h = np.array([
        np.dot(weights, partial.evaluate(z, spec.support))
        for partial in state.partials
    ])
    expected_gradient = expected_empirical_h - got.model_partial_means

    assert got.nll == pytest.approx(expected_nll, rel=2e-14, abs=2e-14)
    assert np.allclose(got.gradient, expected_gradient, rtol=2e-13, atol=2e-13)


def test_exact_hessian_matches_finite_difference_of_analytic_gradient():
    _, z, _, spec, params, obs, _ = _problem(boundaries=True)
    step = 2e-5
    finite = np.empty((spec.n_params, spec.n_params), dtype=float)
    for j in range(spec.n_params):
        up = params.copy()
        dn = params.copy()
        up[j] += step
        dn[j] -= step
        state_up = _state(spec, up, (float(z.min()), float(z.max())))
        state_dn = _state(spec, dn, (float(z.min()), float(z.max())))
        g_up = _evaluate_point_objective(state_up, obs).gradient
        g_dn = _evaluate_point_objective(state_dn, obs).gradient
        finite[:, j] = (g_up - g_dn) / (2.0 * step)

    state = _state(spec, params, (float(z.min()), float(z.max())))
    analytic = _evaluate_point_objective(state, obs).hessian
    assert np.allclose(analytic, finite, rtol=3e-5, atol=3e-6)
    assert np.allclose(analytic, analytic.T, rtol=0.0, atol=2e-13)


def test_point_hessian_is_exactly_fisher():
    _, _, _, _, _, obs, state = _problem(boundaries=False)
    got = _evaluate_point_objective(state, obs)
    np.testing.assert_array_equal(got.hessian, got.fisher)
    np.testing.assert_array_equal(got.missing_information, 0.0)


def test_fisher_is_positive_semidefinite_up_to_roundoff():
    _, _, _, _, _, obs, state = _problem(boundaries=True)
    got = _evaluate_point_objective(state, obs)
    eigenvalues = np.linalg.eigvalsh(got.fisher)
    assert eigenvalues.min() >= -2e-11 * max(1.0, np.max(np.abs(eigenvalues)))


def test_moment_matching_synthetic_summary_makes_gradient_zero_and_hessian_fisher():
    x = np.array([-2.0, 0.0, 1.0, 3.0])
    coord = _build_fit_coordinate((-np.inf, np.inf), x, None, None)
    spec = _build_model_spec(coord, 4)
    params = np.array([0.1, 0.9, 0.3, 0.4])
    state = _state(spec, params, (-4.0, 4.0))
    moments = state.moments.power(10)
    stats = _EmpiricalStats(
        moments=moments,
        boundary_log=np.array([np.nan, np.nan]),
        support=spec.support,
        total_weight=1000.0,
        effective_n=1000.0,
        n_observations=1000,
    )
    got = _evaluate_point_objective(state, _PointObservations(stats))
    assert np.allclose(got.gradient, 0.0, rtol=0.0, atol=2e-12)
    assert np.allclose(got.hessian, got.fisher, rtol=0.0, atol=3e-12)


def test_point_objective_rejects_mismatched_support_summary():
    _, _, _, spec, params, obs, _ = _problem(boundaries=False)
    wrong = _EmpiricalStats(
        moments=obs.stats.moments,
        boundary_log=obs.stats.boundary_log,
        support=(0.0, 1.0),
        total_weight=obs.stats.total_weight,
        effective_n=obs.stats.effective_n,
        n_observations=obs.stats.n_observations,
    )
    state = _state(spec, params, (-1.0, 1.0))
    with pytest.raises(ValueError, match="support"):
        _evaluate_point_objective(state, _PointObservations(wrong))


@pytest.mark.parametrize(
    "support,x,degree,boundaries,params",
    [
        ((-np.inf, np.inf), [-2.5, -0.7, 0.1, 1.4, 3.2], 4,
         (False, False), [-0.08, 0.85, 0.22, 0.15]),
        ((0.0, np.inf), [0.25, 0.7, 1.3, 2.4, 4.8], 3,
         (True, False), [0.3, 0.9, 0.25, 0.35]),
        ((-np.inf, 5.0), [-3.0, -0.4, 1.2, 3.3, 4.6], 3,
         (False, True), [0.28, 0.8, 0.2, 0.5]),
        ((-2.0, 4.0), [-1.6, -0.4, 0.8, 2.1, 3.5], 4,
         (True, True), [0.11, 0.7, 0.2, -0.12, 0.3, 0.55]),
    ],
)
def test_raw_vs_statistics_equivalence_across_support_geometries(
        support, x, degree, boundaries, params):
    x = np.asarray(x, dtype=float)
    weights = np.arange(1.0, x.size + 1.0)
    weights /= weights.sum()
    coord = _build_fit_coordinate(support, x, weights, None)
    z = coord.to_canonical(x)
    spec = _build_model_spec(coord, degree, *boundaries)
    params = np.asarray(params, dtype=float)
    stats = _build_empirical_stats(
        z,
        weights,
        12,
        spec.support,
        has_lower_log=spec.canonical_lower_a_index is not None,
        has_upper_log=spec.canonical_upper_a_index is not None,
    )
    obs = _PointObservations(stats)
    state = _state(spec, params, (float(z.min()), float(z.max())))
    got = _evaluate_point_objective(state, obs)

    q = _q_eval(z, spec.support, state.q_poly, state.boundary_amplitudes, 0)
    raw_nll = float(np.dot(weights, q) + state.log_Z + np.log(coord.scale))
    raw_empirical_h = np.array([
        np.dot(weights, partial.evaluate(z, spec.support))
        for partial in state.partials
    ])
    assert got.nll == pytest.approx(raw_nll, rel=3e-13, abs=3e-13)
    assert np.allclose(
        got.gradient,
        raw_empirical_h - got.model_partial_means,
        rtol=4e-12,
        atol=4e-12,
    )


def test_interval_objective_matches_direct_probability_integrals():
    support = (0.0, 1.0)
    user_intervals = np.array([[0.08, 0.22], [0.35, 0.55], [0.7, 0.92]])
    weights = np.array([0.2, 0.5, 0.3])
    mids = np.mean(user_intervals, axis=1)
    widths = user_intervals[:, 1] - user_intervals[:, 0]
    coord = _build_fit_coordinate(support, mids, weights, widths)
    obs = _build_interval_observations(user_intervals, weights, coordinate=coord)
    spec = _build_model_spec(coord, 4, True, True)
    params = _natural_params(spec, -0.1, [0.8, 0.15], 0.4, 0.25)
    bounds = (float(obs.intervals[:, 0].min()), float(obs.intervals[:, 1].max()))
    fun = _interval_objective(spec, obs, bounds)
    ev = fun(params)


    state = _state(spec, params, bounds)
    direct = []
    for lo, hi in obs.intervals:
        mass = quad(lambda z: state.pdf(z), lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
        direct.append(-np.log(mass))
    expected = float(np.dot(obs.weights, direct))
    assert ev.nll == pytest.approx(expected, rel=2e-8, abs=2e-8)


def test_interval_objective_gradient_and_hessian_match_finite_differences():
    support = (0.0, 1.0)
    user_intervals = np.array([[0.05, 0.18], [0.28, 0.47], [0.62, 0.83], [0.87, 0.97]])
    mids = np.mean(user_intervals, axis=1)
    widths = user_intervals[:, 1] - user_intervals[:, 0]
    coord = _build_fit_coordinate(support, mids, None, widths)
    obs = _build_interval_observations(user_intervals, None, coordinate=coord)
    spec = _build_model_spec(coord, 4, True, True)
    params = _natural_params(spec, -0.15, [0.9, 0.08, 0.12], 0.35, 0.3)
    bounds = (float(obs.intervals[:, 0].min()), float(obs.intervals[:, 1].max()))
    fun = _interval_objective(spec, obs, bounds)
    ev = fun(params)

    eps = 2e-6
    fd_grad = np.empty_like(params)
    for j in range(params.size):
        step = eps * max(1.0, abs(params[j]))
        p1 = params.copy(); p1[j] += step
        p0 = params.copy(); p0[j] -= step
        fd_grad[j] = (fun(p1).nll - fun(p0).nll) / (2.0 * step)
    assert np.allclose(ev.gradient, fd_grad, rtol=3e-5, atol=3e-6)

    fd_h = np.empty_like(ev.hessian)
    for j in range(params.size):
        step = eps * max(1.0, abs(params[j]))
        p1 = params.copy(); p1[j] += step
        p0 = params.copy(); p0[j] -= step
        fd_h[:, j] = (fun(p1).gradient - fun(p0).gradient) / (2.0 * step)
    assert np.allclose(ev.hessian, fd_h, rtol=2e-4, atol=2e-5)
    assert np.allclose(ev.hessian, ev.hessian.T, atol=2e-10)


def test_zero_width_interval_objective_equals_point_objective():
    support = (-np.inf, np.inf)
    x = np.array([-1.0, -0.2, 0.3, 1.4])
    weights = np.array([0.1, 0.2, 0.4, 0.3])
    coord = _build_fit_coordinate(support, x, weights, None)
    z = coord.to_canonical(x)
    spec = _build_model_spec(coord, 4, False, False)
    params = _natural_params(spec, 0.12, [0.75, 0.1, 0.3])
    stats = _build_empirical_stats(z, weights, 8, spec.support)
    pfun = _point_objective(spec, _PointObservations(stats), (z.min(), z.max()))
    intervals = np.column_stack([x, x])
    iobs = _build_interval_observations(intervals, weights, coordinate=coord)
    ifun = _interval_objective(spec, iobs, (z.min(), z.max()))
    pe = pfun(params)
    ie = ifun(params)
    assert ie.nll == pytest.approx(pe.nll, rel=2e-12, abs=2e-12)
    assert np.allclose(ie.gradient, pe.gradient, rtol=2e-11, atol=2e-11)
    assert np.allclose(ie.hessian, pe.hessian, rtol=2e-10, atol=2e-10)


def test_near_exponential_right_tail_reducer_preserves_high_order_moments():
    """A nearly affine half-line potential must not use its enormous
    curvature scale as the rational tail-map scale.  The probability can stay
    accurate while high-order conditional moments fail catastrophically, so
    compare the entire partial-mean vector with independent quadrature.
    """
    edges = np.arange(0.0, 2.0001, 0.25)
    intervals = np.vstack([
        np.column_stack([edges[:-1], edges[1:]]),
        [2.0, np.inf],
    ])
    weights = np.r_[
        np.exp(-edges[:-1]) - np.exp(-edges[1:]),
        np.exp(-2.0),
    ]
    support = (0.0, np.inf)
    coord = _build_interval_fit_coordinate(support, intervals, weights)
    obs = _build_interval_observations(intervals, weights, coordinate=coord)
    spec = _build_model_spec(coord, 4, True, False)

    # Exact exponential slope plus a tiny positive curvature.  The latter
    # makes 1/sqrt(q''(mode)) about 3e4 while the true tail decay length is O(1).
    params = np.zeros(spec.n_params)
    params[spec.layout.gamma_index] = coord.scale
    params[spec.layout.curvature_slice.start] = 1e-9
    finite = obs.intervals[np.isfinite(obs.intervals)]
    bounds = (float(finite.min()), float(finite.max()))
    state = _state(spec, params, bounds)
    assert state.local_scale > 1e4

    interval = obs.intervals[-1]
    reduced = _prepare_partial_interval_reducer(state).reduce(interval)
    lo = float(interval[0])
    mass = quad(state.pdf, lo, np.inf, epsabs=1e-12, epsrel=1e-12, limit=300)[0]
    expected = np.array([
        quad(
            lambda z, partial=partial: (
                partial.evaluate(np.array([z]), spec.support)[0] * state.pdf(z)
            ),
            lo,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-11,
            limit=300,
        )[0] / mass
        for partial in state.partials
    ])

    assert reduced.log_probability == pytest.approx(np.log(mass), abs=2e-12)
    np.testing.assert_allclose(reduced.mean, expected, rtol=2e-10, atol=2e-11)


def test_infinite_interval_objective_matches_direct_tail_probabilities_and_derivatives():
    support = (-np.inf, np.inf)
    intervals = np.array([
        [-np.inf, -0.8],
        [-0.25, 0.35],
        [0.9, np.inf],
        [-np.inf, np.inf],
    ])
    weights = np.array([0.2, 0.35, 0.3, 0.15])
    coord = _build_interval_fit_coordinate(support, intervals, weights)
    obs = _build_interval_observations(intervals, weights, coordinate=coord)
    spec = _build_model_spec(coord, 4)
    params = _natural_params(spec, -0.08, [0.85, 0.04, 0.3])
    finite = obs.intervals[np.isfinite(obs.intervals)]
    bounds = (float(finite.min()), float(finite.max()))
    fun = _interval_objective(spec, obs, bounds)
    ev = fun(params)
    state = _state(spec, params, bounds)

    direct_terms = []
    for lo, hi in obs.intervals:
        if lo == spec.support[0] and hi == spec.support[1]:
            mass = 1.0
        else:
            mass = quad(lambda z: state.pdf(z), lo, hi, epsabs=1e-11, epsrel=1e-11)[0]
        direct_terms.append(-np.log(mass))
    assert ev.nll == pytest.approx(float(np.dot(obs.weights, direct_terms)), rel=3e-8, abs=3e-8)

    eps = 2e-6
    fd_grad = np.empty_like(params)
    fd_h = np.empty_like(ev.hessian)
    for j in range(params.size):
        step = eps * max(1.0, abs(params[j]))
        p1 = params.copy(); p1[j] += step
        p0 = params.copy(); p0[j] -= step
        e1 = fun(p1); e0 = fun(p0)
        fd_grad[j] = (e1.nll - e0.nll) / (2.0 * step)
        fd_h[:, j] = (e1.gradient - e0.gradient) / (2.0 * step)
    assert np.allclose(ev.gradient, fd_grad, rtol=6e-5, atol=6e-6)
    assert np.allclose(ev.hessian, fd_h, rtol=5e-4, atol=5e-5)


def test_exact_boundary_rounded_point_uses_preserved_log_distance():
    points = np.array([1e-22, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
    intervals = np.column_stack([points, points])
    objective = _prepare_natural_interval_objective(
        (0.0, np.inf), intervals, 2, True, False, None
    )
    obs = objective.observations
    tiny = int(np.argmin(obs.point_lower_distance))
    assert obs.intervals[tiny, 0] == obs.support[0]
    assert 0.0 < obs.point_lower_distance[tiny] < 1e-18

    # Exercise the exact likelihood with a genuinely active boundary-log
    # amplitude.  Both the scalar objective and analytic derivative must use
    # the preserved physical distance rather than log(0) at the rounded z.
    params = _natural_interval_start(objective)[0].copy()
    params[objective.spec.canonical_lower_a_index] = 0.2
    fun = _interval_objective(objective.spec, obs, objective.z_data_bounds)
    ev = fun(params)
    assert np.isfinite(ev.nll)
    assert np.all(np.isfinite(ev.gradient))
    assert np.all(np.isfinite(ev.hessian))

    eps = 1e-6
    fd = np.empty_like(params)
    for j in range(params.size):
        step = eps * max(1.0, abs(params[j]))
        p1 = params.copy(); p1[j] += step
        p0 = params.copy(); p0[j] -= step
        fd[j] = (fun(p1).nll - fun(p0).nll) / (2.0 * step)
    assert np.allclose(ev.gradient, fd, rtol=2e-6, atol=2e-7)


def test_boundary_interval_adaptive_refinement_avoids_endpoint_log_collision(
        monkeypatch):

    intervals = np.array([
        [0.0, 1e-9],
        [0.01, 0.01],
        [0.1, 0.1],
        [0.4, 0.4],
        [1.0, 1.0],
    ])
    objective = _prepare_natural_interval_objective(
        (0.0, np.inf), intervals, 2, True, False, None
    )
    start = _natural_interval_start(objective)[0]
    fun = _interval_objective(
        objective.spec, objective.observations, objective.z_data_bounds
    )
    baseline = fun(start)

    monkeypatch.setattr(interval_observations, "QUAD_EPSABS", 1e-12)
    monkeypatch.setattr(interval_observations, "QUAD_EPSREL", 1e-12)
    monkeypatch.setattr(interval_observations, "QUAD_LIMIT", 400)
    refined = fun(start)

    assert np.isfinite(refined.nll)
    assert np.all(np.isfinite(refined.gradient))
    assert np.all(np.isfinite(refined.hessian))
    assert refined.nll == pytest.approx(baseline.nll, rel=2e-9, abs=2e-10)
    assert np.allclose(refined.gradient, baseline.gradient, rtol=2e-6, atol=2e-8)
