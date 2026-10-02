"""Tests for the fixed sufficient-statistics layer."""

import numpy as np
import pytest
from scipy.integrate import quad

from gibbus._fit.natural_objective import _natural_point_stats
from gibbus._observations.empirical import (
    _uniform_boundary_log_expectation,
    _uniform_interval_empirical_stats,
    _uniform_power_moments,
)
from gibbus._observations.intervals import _build_interval_observations


def _build_empirical_stats(z, weights, max_order, support, /, *,
                           has_lower_log=False, has_upper_log=False):
    return _natural_point_stats(
        np.asarray(z, dtype=float), weights, max_order, support,
        has_lower_log, has_upper_log,
    )


def test_unweighted_power_moments_match_raw_means():
    z = np.array([-1.5, -0.2, 0.4, 1.7, 2.1])
    stats = _build_empirical_stats(z, None, 10, (-np.inf, np.inf))
    expected = np.array([np.mean(z ** k) for k in range(11)])
    np.testing.assert_allclose(stats.moments, expected, rtol=1e-14, atol=1e-14)
    assert stats.total_weight == pytest.approx(z.size)
    assert stats.effective_n == pytest.approx(z.size)


def test_weighted_power_moments_accept_unnormalised_weights():
    z = np.array([-2.0, -0.1, 0.3, 1.2, 4.0])
    w = np.array([0.1, 2.0, 0.4, 3.0, 0.2])
    wn = w / w.sum()
    stats = _build_empirical_stats(z, w, 8, (-np.inf, np.inf))
    expected = np.array([np.dot(wn, z ** k) for k in range(9)])
    np.testing.assert_allclose(stats.moments, expected, rtol=2e-14, atol=2e-14)
    assert stats.total_weight == pytest.approx(w.sum())
    assert stats.effective_n == pytest.approx(w.sum() ** 2 / np.dot(w, w))


def test_weight_normalization_survives_overflowing_raw_sum():
    z = np.array([-1.0, 0.0, 1.0])
    w = np.full(3, 1.0e308)
    with np.errstate(over="raise", invalid="raise"):
        stats = _build_empirical_stats(z, w, 4, (-np.inf, np.inf))
    assert stats.moments[1] == pytest.approx(0.0)
    assert stats.moments[2] == pytest.approx(2.0 / 3.0)
    assert np.isinf(stats.total_weight)
    assert stats.effective_n == pytest.approx(3.0)


def test_polynomial_expectation_is_direct_raw_expectation():
    rng = np.random.default_rng(42)
    z = rng.normal(size=100)
    w = rng.random(z.size)
    c = np.array([0.7, -1.2, 0.0, 0.4, -0.08, 0.01])
    stats = _build_empirical_stats(z, w, 8, (-np.inf, np.inf))
    raw = np.polynomial.polynomial.polyval(z, c)
    expected = np.dot(w / w.sum(), raw)
    assert stats.poly_expectation(c) == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_boundary_log_stats_use_fixed_negative_log_basis():
    z = np.array([-0.8, -0.2, 0.1, 0.7])
    w = np.array([1.0, 2.0, 3.0, 4.0])
    wn = w / w.sum()
    stats = _build_empirical_stats(
        z, w, 5, (-1.0, 1.0), has_lower_log=True, has_upper_log=True
    )
    expected_lower = np.dot(wn, -np.log(z + 1.0))
    expected_upper = np.dot(wn, -np.log(1.0 - z))
    np.testing.assert_allclose(
        stats.boundary_log,
        [expected_lower, expected_upper],
        rtol=1e-14,
        atol=1e-14,
    )


def test_potential_expectation_combines_polynomial_and_log_statistics():
    z = np.array([-0.7, -0.1, 0.4])
    c = np.array([0.0, 0.3, 0.8])
    stats = _build_empirical_stats(
        z, None, 4, (-1.0, 1.0), has_lower_log=True, has_upper_log=True
    )
    aL, aU = 0.6, 0.2
    raw = (
        np.polynomial.polynomial.polyval(z, c)
        - aL * np.log(z + 1.0)
        - aU * np.log(1.0 - z)
    )
    assert stats.potential_expectation(
        c, lower_amplitude=aL, upper_amplitude=aU
    ) == pytest.approx(np.mean(raw), rel=1e-13, abs=1e-13)


def test_inactive_boundary_statistic_rejects_nonzero_amplitude():
    stats = _build_empirical_stats([0.0, 0.5], None, 2, (-1.0, 1.0))
    with pytest.raises(ValueError, match="lower boundary-log statistic"):
        stats.potential_expectation([0.0], lower_amplitude=1.0)


def test_endpoint_point_is_invalid_when_corresponding_log_basis_is_active():
    with pytest.raises(ValueError, match="positive-weight point above L"):
        _build_empirical_stats(
            [-1.0, 0.0], None, 2, (-1.0, 1.0), has_lower_log=True
        )
    with pytest.raises(ValueError, match="positive-weight point below U"):
        _build_empirical_stats(
            [0.0, 1.0], None, 2, (-1.0, 1.0), has_upper_log=True
        )


def test_active_log_basis_requires_finite_endpoint():
    with pytest.raises(ValueError, match="requires finite L"):
        _build_empirical_stats(
            [0.0, 1.0], None, 2, (-np.inf, np.inf), has_lower_log=True
        )
    with pytest.raises(ValueError, match="requires finite U"):
        _build_empirical_stats(
            [0.0, 1.0], None, 2, (-np.inf, np.inf), has_upper_log=True
        )


def test_approximate_mean_variance_matches_plugin_formula():
    z = np.array([-1.0, 0.0, 2.0, 3.0])
    stats = _build_empirical_stats(z, None, 8, (-np.inf, np.inf))
    k = 3
    expected = (np.mean(z ** (2 * k)) - np.mean(z ** k) ** 2) / z.size
    assert stats.approximate_mean_variance(k) == pytest.approx(expected)


def test_approximate_mean_variance_requires_double_order():
    stats = _build_empirical_stats([0.0, 1.0], None, 3, (-np.inf, np.inf))
    with pytest.raises(ValueError, match=r"order 2\*k"):
        stats.approximate_mean_variance(2)


def test_polynomial_expectation_requires_available_order():
    stats = _build_empirical_stats([0.0, 1.0], None, 2, (-np.inf, np.inf))
    with pytest.raises(ValueError, match="exceeds"):
        stats.poly_expectation([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "weights",
    [
        [1.0],
        [1.0, -1.0],
        [1.0, np.nan],
        [0.0, 0.0],
    ],
)
def test_invalid_weights_are_rejected(weights):
    z = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        _build_empirical_stats(z, weights, 2, (-np.inf, np.inf))


def test_skewed_scaled_data_high_order_summary_matches_direct_computation():
    rng = np.random.default_rng(1234)
    # Mimic already-canonical but skewed data rather than an enormous raw scale.
    z = rng.exponential(scale=0.8, size=2000) - np.log(2.0) * 0.8
    w = np.linspace(0.1, 2.0, z.size)
    stats = _build_empirical_stats(z, w, 16, (-np.inf, np.inf))
    wn = w / w.sum()
    expected = np.array([np.sum(wn * (z ** k)) for k in range(17)])
    np.testing.assert_allclose(stats.moments, expected, rtol=2e-13, atol=2e-13)


def test_out_of_support_points_are_rejected():
    with pytest.raises(ValueError, match="within the support"):
        _build_empirical_stats([-1.1, 0.0], None, 2, (-1.0, 1.0))
    with pytest.raises(ValueError, match="within the support"):
        _build_empirical_stats([0.0, 1.1], None, 2, (-1.0, 1.0))


def test_uniform_power_moments_match_direct_quadrature():
    intervals = np.array([[-1.5, 0.25], [0.4, 2.0]], dtype=float)
    weights = np.array([0.3, 0.7], dtype=float)
    got = _uniform_power_moments(intervals, weights, 8)
    for k in range(9):
        expected = 0.0
        for (lo, hi), w in zip(intervals, weights, strict=True):
            if hi == lo:
                row = lo ** k
            else:
                row = quad(lambda z, k=k: z ** k, lo, hi)[0] / (hi - lo)
            expected += w * row
        assert got[k] == pytest.approx(expected, rel=2e-12, abs=2e-12)


def test_uniform_power_moments_have_stable_point_limit():
    points = np.array([-2.0, 0.25, 1.7])
    eps = 1e-10
    intervals = np.column_stack([points - eps, points + eps])
    weights = np.array([0.2, 0.3, 0.5])
    got = _uniform_power_moments(intervals, weights, 6)
    expected = np.array([np.dot(weights, points ** k) for k in range(7)])
    assert np.allclose(got, expected, rtol=2e-9, atol=2e-9)


def test_uniform_boundary_log_average_allows_positive_width_touching_endpoint():
    intervals = np.array([[0.0, 0.2], [0.3, 0.8]], dtype=float)
    weights = np.array([0.4, 0.6], dtype=float)
    got = _uniform_boundary_log_expectation(intervals, weights, 0.0, "lower")
    expected = 0.0
    for (lo, hi), w in zip(intervals, weights, strict=True):
        row = quad(lambda z: -np.log(z), lo, hi, points=[lo])[0] / (hi - lo)
        expected += w * row
    assert got == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_zero_width_endpoint_log_statistic_is_explicitly_rejected():
    intervals = np.array([[0.0, 0.0], [0.2, 0.4]], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    with pytest.raises(ValueError, match="zero-width observation on an active boundary"):
        _uniform_boundary_log_expectation(intervals, weights, 0.0, "lower")


def test_uniform_interval_stats_preserve_weight_metadata_and_logs():
    intervals = np.array([[0.0, 0.2], [0.2, 0.5], [0.2, 0.5]], dtype=float)
    raw_weights = np.array([1.0, 2.0, 3.0])
    obs = _build_interval_observations(
        intervals, raw_weights, support=(0.0, 1.0), deduplicate=True
    )
    stats = _uniform_interval_empirical_stats(
        obs, 6, has_lower_log=True, has_upper_log=True
    )
    assert stats.n_observations == 3
    assert stats.total_weight == pytest.approx(6.0)
    assert stats.effective_n == pytest.approx(36.0 / 14.0)
    assert stats.moments[0] == pytest.approx(1.0)
    assert np.all(np.isfinite(stats.moments))
    assert np.all(np.isfinite(stats.boundary_log))


def test_infinite_intervals_are_not_silently_uniformized():
    obs = _build_interval_observations(
        np.array([[0.0, 1.0], [1.0, np.inf]]), None, support=(0.0, np.inf)
    )
    with pytest.raises(NotImplementedError, match="infinite censoring rows"):
        _uniform_interval_empirical_stats(obs, 4, has_lower_log=True)
