"""Tests for the new model-independent interval observation layer."""

import numpy as np
import pytest
from scipy.integrate import quad

from gibbus._fit.inputs import _canon_univariate_samples
from gibbus._model.coords import _build_fit_coordinate, _build_interval_fit_coordinate
from gibbus._observations import intervals as intervals_module
from gibbus._observations.empirical import _normalized_weights
from gibbus._observations.intervals import (
    _build_interval_observations,
)


def test_build_maps_upper_halfline_and_preserves_order():
    intervals = np.array([[7.0, 9.0], [2.0, 6.0]])
    mids = intervals.mean(axis=1)
    widths = intervals[:, 1] - intervals[:, 0]
    coord = _build_fit_coordinate((-np.inf, 10.0), mids, None, widths)
    obs = _build_interval_observations(intervals, coordinate=coord)

    assert obs.support == pytest.approx(coord.canonical_support)
    assert np.all(obs.intervals[:, 0] <= obs.intervals[:, 1])
    expected = coord.intervals_to_canonical(intervals)
    # The new representation deduplicates/sorts rows lexicographically.
    expected = np.unique(expected, axis=0)
    assert np.allclose(obs.intervals, expected)


def test_duplicate_compression_preserves_original_effective_n():
    intervals = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 2.0]])
    obs = _build_interval_observations(intervals)
    assert obs.n_observations == 3
    assert obs.n_unique == 2
    assert obs.effective_n == pytest.approx(3.0)
    assert obs.total_weight == pytest.approx(3.0)
    assert obs.weights.sum() == pytest.approx(1.0)
    assert np.allclose(obs.weights, [2.0 / 3.0, 1.0 / 3.0])


def test_weighted_duplicate_compression_is_exact():
    intervals = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 2.0]])
    weights = np.array([1.0, 3.0, 2.0])
    obs = _build_interval_observations(
        intervals, _normalized_weights(len(intervals), weights, "interval")
    )
    assert np.allclose(obs.weights, [4.0 / 6.0, 2.0 / 6.0])
    assert obs.total_weight == pytest.approx(6.0)
    assert obs.effective_n == pytest.approx(36.0 / 14.0)


def test_prepared_weight_summary_is_consumed_without_reprocessing(monkeypatch):
    rows = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 2.0]])
    summary = _normalized_weights(3, np.array([1.0, 3.0, 2.0]), "interval")

    def unexpected_weight_preparation(n, weights):
        raise AssertionError("prepared weights were processed again")

    monkeypatch.setattr(
        intervals_module, "_canonical_weights", unexpected_weight_preparation
    )
    obs = _build_interval_observations(rows, summary)
    assert obs.n_observations == 3
    assert obs.total_weight == summary.total_weight
    assert obs.effective_n == summary.effective_n
    np.testing.assert_allclose(obs.weights, [4.0 / 6.0, 2.0 / 6.0])


def test_weight_normalization_survives_overflowing_raw_sum():
    intervals = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    w = np.full(3, 1.0e308)
    with np.errstate(over="raise", invalid="raise"):
        obs = _build_interval_observations(
            intervals, _normalized_weights(len(intervals), w, "interval")
        )
    np.testing.assert_allclose(obs.weights, np.full(3, 1.0 / 3.0))
    assert np.isinf(obs.total_weight)
    assert obs.effective_n == pytest.approx(3.0)


def test_boundary_validates_rows_and_builder_checks_canonical_support():
    with pytest.raises(ValueError, match="must not contain NaN"):
        _canon_univariate_samples([[0.0, np.nan]], min_samples=1)
    rows, _ = _canon_univariate_samples([[2.0, 1.0]], min_samples=1)
    obs = _build_interval_observations(rows)
    np.testing.assert_array_equal(obs.intervals, [[1.0, 2.0]])
    with pytest.raises(ValueError, match="outside"):
        _build_interval_observations(np.array([[0.0, 2.0]]), support=(-1.0, 1.0))


def test_infinite_rows_are_classified_but_finite_plan_refuses_them():
    obs = _build_interval_observations(
        np.array([[-np.inf, -1.0], [0.0, np.inf], [-np.inf, np.inf]]),
        support=(-np.inf, np.inf),
    )
    assert obs.has_infinite_rows
    assert np.count_nonzero(obs.left_infinite_rows) == 1
    assert np.count_nonzero(obs.right_infinite_rows) == 1
    assert np.count_nonzero(obs.whole_support_rows) == 1
    with pytest.raises(NotImplementedError, match="infinite censoring"):
        obs.finite_quadrature(0.0)


def test_log_integrals_match_direct_quadrature_for_gaussian_kernel():
    intervals = np.array([[-2.0, -0.2], [-0.7, 0.8], [0.3, 2.2]])
    obs = _build_interval_observations(intervals, deduplicate=False)
    plan = obs.finite_quadrature(0.0)
    log_kernel = -0.5 * plan.nodes**2
    got = np.exp(plan.log_integrals(log_kernel))
    expected = np.array(
        [
            quad(lambda z: np.exp(-0.5 * z * z), lo, hi, epsabs=1e-13)[0]
            for lo, hi in intervals
        ]
    )
    assert np.allclose(got, expected, rtol=2e-13, atol=2e-14)


def test_point_limit_matches_density_and_zero_conditional_covariance():
    intervals = np.array([[0.25, 0.25], [1.0, 1.0 + 1e-15]])
    obs = _build_interval_observations(intervals, deduplicate=False)
    plan = obs.finite_quadrature(0.0)
    assert np.all(plan.point_limit)

    log_kernel = -(plan.nodes**2)
    point_log = -(plan.midpoints**2)
    log_i = plan.log_integrals(log_kernel, point_log_kernel=point_log)
    assert log_i[0] == pytest.approx(point_log[0])
    assert log_i[1] == pytest.approx(point_log[1] + np.log(plan.widths[1]))

    values = np.stack([plan.nodes, plan.nodes**2], axis=2)
    point_values = np.stack([plan.midpoints, plan.midpoints**2], axis=1)
    mean, cov = plan.conditional_mean_covariance(
        log_kernel,
        values,
        point_values=point_values,
        log_integrals=log_i,
    )
    assert np.allclose(mean, point_values)
    assert np.array_equal(cov, np.zeros_like(cov))


def test_conditional_mean_and_covariance_match_direct_integrals():
    intervals = np.array([[-1.2, 0.4], [0.2, 1.7]])
    obs = _build_interval_observations(intervals, deduplicate=False)
    plan = obs.finite_quadrature(0.1)
    log_kernel = -0.5 * (plan.nodes - 0.2) ** 2
    values = np.stack([plan.nodes, plan.nodes**2], axis=2)
    log_i = plan.log_integrals(log_kernel)
    mean, cov = plan.conditional_mean_covariance(
        log_kernel, values, log_integrals=log_i
    )

    for r, (lo, hi) in enumerate(intervals):
        z0 = quad(lambda z: np.exp(-0.5 * (z - 0.2) ** 2), lo, hi)[0]
        m1 = quad(lambda z: z * np.exp(-0.5 * (z - 0.2) ** 2), lo, hi)[0] / z0
        m2 = quad(lambda z: z**2 * np.exp(-0.5 * (z - 0.2) ** 2), lo, hi)[0] / z0
        m3 = quad(lambda z: z**3 * np.exp(-0.5 * (z - 0.2) ** 2), lo, hi)[0] / z0
        m4 = quad(lambda z: z**4 * np.exp(-0.5 * (z - 0.2) ** 2), lo, hi)[0] / z0
        expected_mean = np.array([m1, m2])
        expected_cov = np.array(
            [
                [m2 - m1 * m1, m3 - m1 * m2],
                [m3 - m1 * m2, m4 - m2 * m2],
            ]
        )
        assert np.allclose(mean[r], expected_mean, rtol=2e-12, atol=2e-13)
        assert np.allclose(cov[r], expected_cov, rtol=3e-11, atol=3e-12)


def test_exact_points_preserve_sub_ulp_boundary_distance_through_affine_map():

    points = np.array([1e-22, 2e-22, 0.1, 0.5, 1.0])
    intervals = np.column_stack([points, points])
    coord = _build_interval_fit_coordinate((0.0, np.inf), intervals)
    obs = _build_interval_observations(intervals, coordinate=coord)

    # The first two physical points are distinct, but subtracting the common
    # fitting-coordinate center rounds both onto the canonical endpoint.
    assert obs.intervals[0, 0] == obs.support[0]
    assert obs.intervals[1, 0] == obs.support[0]
    assert obs.point_lower_distance[0] > 0.0
    assert obs.point_lower_distance[1] > obs.point_lower_distance[0]
    assert obs.n_unique == points.size

    # Reflection of an upper half-line must preserve the same geometry on the
    # canonical lower side.
    reflected = np.column_stack([-points, -points])
    reflected_coord = _build_interval_fit_coordinate((-np.inf, 0.0), reflected)
    reflected_obs = _build_interval_observations(reflected, coordinate=reflected_coord)
    assert reflected_coord.direction == -1.0
    assert reflected_obs.intervals[0, 0] == reflected_obs.support[0]
    assert reflected_obs.point_lower_distance[0] == pytest.approx(
        obs.point_lower_distance[0], rel=0.0, abs=0.0
    )


def test_row_grouping_matches_numpy_unique_on_numeric_edge_cases():
    from gibbus._observations.intervals import _row_grouping

    rows = np.array(
        [
            [0.0, 1.0],
            [-0.0, 1.0],
            [np.inf, np.inf],
            [-np.inf, 0.0],
            [2.0, 3.0],
            [0.0, 1.0],
            [2.0, 3.0],
            [-np.inf, 0.0],
        ],
        dtype=np.float64,
    )
    unique, expected_first, expected_inverse = np.unique(
        rows, axis=0, return_index=True, return_inverse=True
    )
    first, inverse, n_unique = _row_grouping(rows)
    assert n_unique == unique.shape[0]
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(inverse, expected_inverse)
    np.testing.assert_array_equal(rows[first], unique)
