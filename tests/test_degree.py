"""Tests for information-based omitted-statistic degree diagnostics."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import gennorm

from gibbus._fit import mixture_degree
from gibbus._fit.degree import (
    _DegreeSelectionConfig,
    _interval_omitted_statistic_diagnostic,
    _omitted_statistic_diagnostic,
    _probe_orders_for_degree,
)
from gibbus._fit.mixture_degree import _joint_omitted_statistic_diagnostic
from gibbus._fit.natural_objective import (
    _degree_diagnostic_fit,
    _fit_natural_conic_intervals,
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points,
    _fit_natural_conic_points_auto,
    _natural_point_stats,
)


def _build_empirical_stats(z, weights, max_order, support, /):
    return _natural_point_stats(z, weights, max_order, support, False, False)


def _point_fit(support, x, degree, lower=False, upper=False, weights=None, /, **kw):
    """Fitted point model in the view the degree diagnostics consume."""
    objective, result = _fit_natural_conic_points(
        support, x, degree, lower, upper, weights, **kw
    )
    return _degree_diagnostic_fit(objective, result)


def _interval_fit(support, rows, degree, lower=False, upper=False, /):
    objective, result = _fit_natural_conic_intervals(
        support, rows, degree, lower, upper
    )
    return _degree_diagnostic_fit(objective, result)


def _point_auto(support, x, lower=False, upper=False, weights=None, /):
    """Automatic-degree point fit; returns its objective."""
    return _fit_natural_conic_points_auto(support, x, lower, upper, weights)[0]


def _interval_auto(support, rows, /):
    return _fit_natural_conic_intervals_auto(support, rows)[0]


def _joint_diagnostic(information, gradient, row_scores, weights=None, **kwargs):
    rows = len(row_scores)
    return _joint_omitted_statistic_diagnostic(
        gradient=np.asarray(gradient, dtype=float),
        observed_information=np.asarray(information, dtype=float),
        row_scores=np.asarray(row_scores, dtype=float),
        weights=np.full(rows, 1 / rows) if weights is None else weights,
        nuisance_indices=[0],
        probe_indices=[1],
        fitted_degree=2,
        probe_orders=(3,),
        component_index=1,
        **kwargs,
    )


def test_joint_degree_profiles_shared_amplitude_and_logits_not_component_only():
    scores = np.tile([[0.0, -0.3], [0.0, 0.7]], (100, 1))
    diagnostic = _joint_diagnostic([[2.0, 1.0], [1.0, 3.0]], [0.4, 0.2], scores)
    assert diagnostic.efficient_residual == pytest.approx([0.0])
    assert diagnostic.conditional_covariance == pytest.approx(np.array([[2.5]]))
    assert not diagnostic.should_expand


def test_joint_degree_never_reflects_indefinite_observed_information():
    scores = np.tile([[0.0, -1.0], [0.0, 1.0]], (100, 1))
    diagnostic = _joint_diagnostic([[1.0, 2.0], [2.0, 1.0]], [0.0, 10.0], scores)
    assert diagnostic.information_status == "indefinite_information"
    assert diagnostic.stopped_for_reliability
    assert diagnostic.score == 0.0
    assert not diagnostic.should_expand


def test_joint_degree_participation_stops_one_row_evidence():
    scores = np.zeros((200, 2))
    scores[-1, 1] = 1e8
    diagnostic = _joint_diagnostic(np.eye(2), [0.0, 1e6], scores)
    assert diagnostic.participation[0] == pytest.approx(1.0)
    assert diagnostic.information_status == "insufficient_participation"
    assert not diagnostic.should_expand


def test_joint_degree_duplicate_rows_retain_original_kish_size():
    scores = np.tile([[0.0, -0.5], [0.0, 0.5]], (100, 1))
    weights = np.tile([1.0, 2.0], 100)
    base = _joint_diagnostic(np.eye(2), [0.0, 0.2], scores, weights)
    duplicated = _joint_diagnostic(
        np.eye(2),
        [0.0, 0.2],
        np.repeat(scores, 2, axis=0),
        np.repeat(weights / 2, 2),
    )
    assert duplicated.score == pytest.approx(2 * base.score)
    assert duplicated.participation[0] == pytest.approx(2 * base.participation[0])


def test_joint_degree_whole_support_information_is_unresolved():
    diagnostic = _joint_diagnostic(np.zeros((2, 2)), [0.0, 0.0], np.zeros((100, 2)))
    assert diagnostic.stopped_for_reliability
    assert diagnostic.rank == 0
    assert not diagnostic.should_expand


def test_shared_degree_growth_changes_one_component_and_honors_fixed_policy(
    monkeypatch,
):
    @dataclass(frozen=True)
    class Fit:
        components: tuple
        log_likelihood: float
        degree_diagnostics: tuple = ()

    def make_fit(degrees):
        return Fit(
            tuple(
                SimpleNamespace(spec=SimpleNamespace(requested_poly_degree=d))
                for d in degrees
            ),
            float(sum(degrees)),
        )

    def diagnose(fit, policies, rows, weights, config):
        return tuple(
            (
                k,
                SimpleNamespace(
                    should_expand=policy == "auto"
                    and component.spec.requested_poly_degree < 4,
                    p_value=0.001 if k == 1 else 0.01,
                    score=100.0 if k == 1 else 20.0,
                ),
            )
            for k, (component, policy) in enumerate(zip(fit.components, policies))
        )

    monkeypatch.setattr(mixture_degree, "_shared_degree_diagnostics", diagnose)
    calls = []

    def refit(degrees, previous):
        old = tuple(c.spec.requested_poly_degree for c in previous.components)
        assert sum(a != b for a, b in zip(degrees, old)) == 1
        calls.append(degrees)
        return make_fit(degrees)

    result = mixture_degree._fit_shared_degree_growth(
        make_fit((2, 2, 6)),
        ("auto", "auto", 6),
        support=(-np.inf, np.inf),
        rows=np.zeros((100, 1)),
        observation_weights=np.full(100, 0.01),
        refit=refit,
    )
    assert calls == [(2, 4, 6), (4, 4, 6)]
    assert [item["expanded_component"] for item in result.degree_diagnostics] == [
        1,
        0,
        None,
    ]


@pytest.mark.parametrize("intervals", [False, True])
@pytest.mark.parametrize("support", [(0.0, np.inf), (-np.inf, 0.0), (0.0, 1.0)])
@pytest.mark.parametrize("contracted", [False, True])
def test_shared_degree_zero_lift_preserves_density_and_raw_joint_geometry(
    support, intervals, contracted
):
    from gibbus._fit.natural_mixture import (
        _CompiledJointMixture,
        _ComponentProblem,
        _degree_probe_problems,
    )

    bounded = np.isfinite(support).all()
    x = np.linspace(0.1, 0.9 if bounded else 3.0, 50)
    if support[1] == 0.0:
        x = -x
    rows = np.column_stack((x - 0.02, x + 0.02)) if intervals else x[:, None]
    weights = np.linspace(1.0, 2.0, len(x))
    weights /= weights.sum()
    resp = np.column_stack(
        (np.linspace(0.2, 0.8, len(x)), np.linspace(0.8, 0.2, len(x)))
    )
    components = []
    problems = []
    for k in range(2):
        problem = _ComponentProblem(
            support,
            rows,
            4 if contracted else 2,
            np.isfinite(support[0]),
            np.isfinite(support[1]),
            weights * resp[:, k],
        )
        layout = problem.spec.layout
        curvature = np.zeros(layout.curvature_degree + 1)
        curvature[0] = 1.0
        params = layout.pack(0.1, curvature, [1.5, 1.5])
        components.append(
            SimpleNamespace(
                coordinate=problem.coordinate,
                spec=problem.spec,
                layout=layout,
                params=params,
                z_data_bounds=problem.z_data_bounds,
                effective_curvature_degree=0,
            )
        )
        problems.append(problem)
    fit = SimpleNamespace(
        components=tuple(components),
        responsibilities=resp,
        weights=np.array([0.4, 0.6]),
    )
    original = _CompiledJointMixture(problems, [c.layout for c in components], weights)
    base = original(original.join([c.params for c in components], np.log(fit.weights)))
    degrees = (6, 6) if contracted else (4, 4)
    evaluation, row_scores, nuisance, probes = mixture_degree._mixture_probe_geometry(
        fit, rows, weights, degrees
    )
    assert evaluation.nll == pytest.approx(base.nll, abs=1e-10)
    np.testing.assert_allclose(weights @ row_scores, evaluation.gradient, atol=1e-10)
    np.testing.assert_allclose(
        evaluation.observed_hessian,
        evaluation.fisher - evaluation.missing_information,
        atol=1e-12,
    )
    assert np.linalg.norm(evaluation.missing_information) > 1e-5
    assert len(nuisance) == 5 + sum(np.isfinite(support))
    assert [len(probe) for probe in probes] == [2, 2]
    lifted, layouts, params = _degree_probe_problems(fit, rows, weights, degrees)
    assert [p.coordinate for p in lifted] == [c.coordinate for c in components]
    objective = _CompiledJointMixture(lifted, layouts, weights)
    point = objective.join(params, np.log(fit.weights))
    for index in nuisance:
        step = np.zeros_like(point)
        step[index] = 1e-5
        finite_difference = (
            objective(point + step).gradient - objective(point - step).gradient
        ) / 2e-5
        np.testing.assert_allclose(
            finite_difference,
            evaluation.observed_hessian[:, index],
            rtol=3e-4,
            atol=1e-7,
        )


def test_separated_shared_degree_growth_detects_only_quartic_component(monkeypatch):
    from gibbus._fit import natural_mixture, natural_objective

    def independent_proposal(*args, **kwargs):
        raise AssertionError(
            "mixture degree proposals must not fit independent amplitudes"
        )

    monkeypatch.setattr(
        natural_objective, "_fit_natural_conic_points_auto", independent_proposal
    )
    if hasattr(natural_mixture, "_fit_natural_conic_points_auto"):
        monkeypatch.setattr(
            natural_mixture, "_fit_natural_conic_points_auto", independent_proposal
        )
    rng = np.random.default_rng(56)
    quartic = gennorm.rvs(4.0, size=2000, random_state=rng)
    x = np.r_[quartic - 6, rng.normal(size=2000) + 6]
    resp = np.zeros((len(x), 2))
    resp[:2000, 0] = 1
    resp[2000:, 1] = 1
    fitted = natural_mixture._fit_natural_mixture(
        (-np.inf, np.inf),
        x,
        2,
        "auto",
        responsibilities=resp,
        paths=(("direct", "raw"),),
        max_em_steps=10,
    )
    assert tuple(c.spec.requested_poly_degree for c in fitted.components) == (4, 2)
    assert fitted.degree_diagnostics[0]["expanded_component"] == 0
    assert fitted.degree_diagnostics[-1]["expanded_component"] is None


def test_joint_degree_k1_reuses_existing_point_diagnostic():
    from gibbus._fit.natural_mixture import _fit_natural_mixture
    from gibbus._fit.natural_objective import _prepare_natural_point_objective

    x = np.random.default_rng(121).normal(size=600)
    rows = x[:, None]
    weights = np.full(len(x), 1.0 / len(x))
    fitted = _fit_natural_mixture(
        (-np.inf, np.inf),
        rows,
        1,
        2,
        responsibilities=np.ones((len(x), 1)),
        paths=(("direct", "raw"),),
    )
    ((index, actual),) = mixture_degree._shared_degree_diagnostics(
        fitted, ("auto",), rows, weights
    )
    objective = _prepare_natural_point_objective(
        (-np.inf, np.inf), x, 2, False, False, weights, moment_order=8
    )
    single = _degree_diagnostic_fit(objective, fitted.components[0].solver_result)
    expected = _omitted_statistic_diagnostic(single, (3, 4))
    assert index == 0
    assert actual.score == pytest.approx(expected.score, rel=1e-7, abs=1e-8)
    np.testing.assert_allclose(actual.participation, expected.participation)


def test_moment_participation_detects_tail_dominance():
    rng = np.random.default_rng(0)
    z = np.concatenate([rng.normal(size=800), [1e6]])
    stats = _build_empirical_stats(z, None, 12, (-np.inf, np.inf))
    assert stats.moment_participation(0) > 700
    assert stats.moment_participation(6) < 2.0


def test_power_covariance_matches_direct_empirical_covariance():
    rng = np.random.default_rng(1)
    z = rng.normal(size=500)
    stats = _build_empirical_stats(z, None, 10, (-np.inf, np.inf))
    got = stats.power_covariance(5)
    values = np.column_stack([z**k for k in range(6)])
    expected = np.cov(values, rowvar=False, bias=True)
    np.testing.assert_allclose(got, expected, rtol=3e-13, atol=3e-13)


def test_gaussian_degree_two_has_no_resolvable_omitted_block():
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    fit = _point_fit((-np.inf, np.inf), x, 2, moment_order=12)
    diagnostic = _omitted_statistic_diagnostic(fit, (3, 4))
    assert diagnostic.rank >= 1
    assert not diagnostic.should_expand
    assert diagnostic.p_value > 0.01


def test_quartic_log_concave_shape_requests_more_capacity():
    rng = np.random.default_rng(3)
    x = gennorm.rvs(beta=4.0, size=4000, random_state=rng)
    fit = _point_fit((-np.inf, np.inf), np.ascontiguousarray(x), 2, moment_order=12)
    diagnostic = _omitted_statistic_diagnostic(fit, (3, 4))
    assert diagnostic.should_expand
    assert diagnostic.p_value < 1e-4


def test_extreme_outlier_moments_are_stopped_by_participation():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(size=800), [1e9]])
    fit = _point_fit((-np.inf, np.inf), x, 2, moment_order=12)
    config = _DegreeSelectionConfig(min_participation=8.0)
    diagnostic = _omitted_statistic_diagnostic(fit, (3, 4), config=config)
    assert diagnostic.stopped_for_reliability
    assert not diagnostic.should_expand
    assert np.all(diagnostic.participation < 8.0)


def test_probe_order_block_is_consecutive_and_bounded():
    assert _probe_orders_for_degree(2, 12, block_size=2) == (3, 4)
    assert _probe_orders_for_degree(11, 12, block_size=3) == (12,)
    assert _probe_orders_for_degree(12, 12, block_size=2) == ()


def test_auto_selector_stops_at_two_for_gaussian():
    rng = np.random.default_rng(5)
    x = rng.normal(size=3000)
    fit = _point_auto((-np.inf, np.inf), x)
    assert fit.spec.requested_poly_degree == 2


def test_auto_selector_reaches_four_for_quartic_shape():
    rng = np.random.default_rng(6)
    x = gennorm.rvs(beta=4.0, size=4000, random_state=rng)
    fit = _point_auto((-np.inf, np.inf), np.ascontiguousarray(x))
    assert fit.spec.requested_poly_degree == 4


def test_auto_selector_does_not_chase_single_extreme_outlier():
    rng = np.random.default_rng(7)
    x = np.concatenate([rng.normal(size=800), [1e9]])
    fit = _point_auto((-np.inf, np.inf), x)
    assert fit.spec.requested_poly_degree == 2


def test_interval_diagnostic_matches_narrow_point_intuition():
    rng = np.random.default_rng(8)
    x = rng.normal(size=1500)
    intervals = np.column_stack([x - 0.01, x + 0.01])
    fit = _interval_fit((-np.inf, np.inf), intervals, 2)
    diagnostic = _interval_omitted_statistic_diagnostic(fit, (3, 4))
    assert diagnostic.rank == 2
    assert not diagnostic.should_expand


def test_interval_diagnostic_preserves_sub_ulp_boundary_point_distance():
    points = np.array([1e-22, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2])
    intervals = np.column_stack([points, points])
    fit = _interval_fit((0.0, np.inf), intervals, 2, True, False)
    tiny = int(np.argmin(fit.observations.point_lower_distance))
    assert fit.observations.intervals[tiny, 0] == fit.observations.support[0]
    assert 0.0 < fit.observations.point_lower_distance[tiny] < 1e-18

    diagnostic = _interval_omitted_statistic_diagnostic(fit, (3, 4))

    assert np.all(np.isfinite(diagnostic.efficient_residual))
    assert np.isfinite(diagnostic.score)
    assert np.isfinite(diagnostic.p_value)


def test_interval_diagnostic_detects_quartic_shape():
    rng = np.random.default_rng(9)
    x = gennorm.rvs(beta=4.0, size=2000, random_state=rng)
    intervals = np.column_stack([x - 0.01, x + 0.01])
    fit = _interval_fit((-np.inf, np.inf), intervals, 2)
    diagnostic = _interval_omitted_statistic_diagnostic(fit, (3, 4))
    assert diagnostic.should_expand
    assert diagnostic.p_value < 1e-5


def test_interval_auto_selector_reaches_four_for_quartic_shape():
    rng = np.random.default_rng(10)
    x = gennorm.rvs(beta=4.0, size=2000, random_state=rng)
    intervals = np.column_stack([x - 0.01, x + 0.01])
    fit = _interval_auto((-np.inf, np.inf), intervals)
    assert fit.spec.requested_poly_degree == 4


@pytest.mark.parametrize("seed", range(5))
def test_point_selector_calibration_gaussian_has_no_false_expansion(seed):
    x = np.random.default_rng(seed).normal(size=1000)
    fit = _point_auto((-np.inf, np.inf), x)
    assert fit.spec.requested_poly_degree == 2


@pytest.mark.parametrize("seed", range(5))
def test_point_selector_calibration_quartic_has_power(seed):
    rng = np.random.default_rng(seed)
    x = gennorm.rvs(beta=4.0, size=1000, random_state=rng)
    fit = _point_auto((-np.inf, np.inf), np.ascontiguousarray(x))
    assert fit.spec.requested_poly_degree == 4


@pytest.mark.parametrize("seed", range(5))
def test_interval_selector_calibration_gaussian_has_no_false_expansion(seed):
    x = np.random.default_rng(seed).normal(size=1000)
    intervals = np.column_stack([x - 0.015, x + 0.015])
    fit = _interval_auto((-np.inf, np.inf), intervals)
    assert fit.spec.requested_poly_degree == 2


@pytest.mark.parametrize("seed", range(5))
def test_interval_selector_calibration_quartic_has_power(seed):
    rng = np.random.default_rng(seed)
    x = gennorm.rvs(beta=4.0, size=1000, random_state=rng)
    intervals = np.column_stack([x - 0.015, x + 0.015])
    fit = _interval_auto((-np.inf, np.inf), intervals)
    assert fit.spec.requested_poly_degree == 4


def test_weighted_component_style_selector_calibration():
    rng = np.random.default_rng(11)
    left = rng.normal(-3.0, 0.7, 1200)
    right = gennorm.rvs(beta=4.0, size=1200, random_state=rng) + 3.0
    x = np.ascontiguousarray(np.concatenate([left, right]))

    w_left = np.concatenate([np.ones(left.size), np.zeros(right.size)])
    fit_left = _point_auto((-np.inf, np.inf), x, False, False, w_left)
    assert fit_left.spec.requested_poly_degree == 2

    w_right = np.concatenate([np.zeros(left.size), np.ones(right.size)])
    fit_right = _point_auto((-np.inf, np.inf), x, False, False, w_right)
    assert fit_right.spec.requested_poly_degree == 4
    # Zero weights drop out of the Kish effective size (up to summation
    # rounding); a regression would be far larger.
    assert fit_right.observations.stats.effective_n == pytest.approx(
        right.size, rel=1e-12
    )


def test_infinite_interval_diagnostic_uses_adaptive_missing_information():
    rng = np.random.default_rng(21)
    x = rng.normal(size=900)
    rows = []
    for value in x:
        if value < -1.0:
            rows.append((-np.inf, -1.0))
        elif value > 1.0:
            rows.append((1.0, np.inf))
        else:
            rows.append((value - 0.02, value + 0.02))
    intervals = np.asarray(rows, dtype=float)
    fit = _interval_fit((-np.inf, np.inf), intervals, 2)
    diagnostic = _interval_omitted_statistic_diagnostic(fit, (3, 4))
    assert diagnostic.rank >= 1
    assert not diagnostic.should_expand


def test_infinite_interval_auto_selector_stays_native():
    rng = np.random.default_rng(22)
    x = rng.normal(size=700)
    rows = []
    for value in x:
        if value < -0.9:
            rows.append((-np.inf, -0.9))
        elif value > 1.1:
            rows.append((1.1, np.inf))
        else:
            rows.append((value - 0.025, value + 0.025))
    fit = _interval_auto((-np.inf, np.inf), np.asarray(rows, dtype=float))
    assert fit.spec.requested_poly_degree == 2
    assert fit.observations.has_infinite_rows


def test_auto_selector_runs_on_full_boundary_cone():
    """Enabled amplitudes remain part of every degree candidate."""
    rng = np.random.default_rng(102)
    x = np.ascontiguousarray(rng.gamma(2.0, 1.0, 900))
    objective, result = _fit_natural_conic_points_auto((0.0, np.inf), x, True, False)
    assert 2 <= objective.spec.requested_poly_degree <= 12
    assert objective.layout.lower_a_index is not None
    assert result.status in ("converged", "converged_approximately")
