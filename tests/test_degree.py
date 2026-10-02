"""Tests for information-based omitted-statistic degree diagnostics."""

import numpy as np
import pytest
from scipy.stats import gennorm

from gibbus._fit.degree import (
    _DegreeSelectionConfig,
    _interval_omitted_statistic_diagnostic,
    _omitted_statistic_diagnostic,
    _probe_orders_for_degree,
)
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
