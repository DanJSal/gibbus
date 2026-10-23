"""Tests for the new model-independent interval observation layer."""

import numpy as np
import pytest

from gibbus._fit.inputs import _canon_univariate_samples
from gibbus._observations import intervals as intervals_module
from gibbus._observations.empirical import _normalized_weights
from gibbus._observations.intervals import (
    _build_interval_observations,
)


def test_build_orders_and_merges_rows_against_the_support():
    intervals = np.array([[7.0, 9.0], [2.0, 6.0], [7.0, 9.0]])
    obs = _build_interval_observations(intervals, support=(-np.inf, 10.0))
    assert obs.support == (-np.inf, 10.0)
    np.testing.assert_array_equal(obs.intervals, [[2.0, 6.0], [7.0, 9.0]])
    np.testing.assert_allclose(obs.weights, [1.0 / 3.0, 2.0 / 3.0])


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
