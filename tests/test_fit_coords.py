"""Definition-level tests for the canonical coordinate (the q10/q90 rule)."""

import numpy as np
import pytest

from gibbus._fit.inputs import _canon_univariate_samples, _validate_support
from gibbus._model.coords import (
    _BOUNDED,
    _LOWER_HALF_LINE,
    _REAL_LINE,
    _UPPER_HALF_LINE,
    _build_fit_coordinate,
    _build_interval_fit_coordinate,
    _interval_quantiles,
    _support_kind,
    _weighted_quantile,
)


@pytest.mark.parametrize(
    "support,expected",
    [
        ((-np.inf, np.inf), _REAL_LINE),
        ((0.0, np.inf), _LOWER_HALF_LINE),
        ((-np.inf, 3.0), _UPPER_HALF_LINE),
        ((-2.0, 5.0), _BOUNDED),
    ],
)
def test_support_kind_classifies_endpoint_geometry(support, expected):
    assert _support_kind(support) == expected


@pytest.mark.parametrize(
    "support",
    [
        ((0.0, 0.0)),
        ((1.0, -1.0)),
        ((np.nan, 1.0)),
        ((0.0, np.nan)),
        ((0.0, 1.0, 2.0)),
    ],
)
def test_support_boundary_rejects_invalid_support(support):
    with pytest.raises(ValueError):
        _validate_support(support)


def test_bounded_coordinate_maps_the_support_to_minus_one_one():
    # The data play no part on a bounded support.
    coord = _build_fit_coordinate((-3.0, 5.0), np.asarray([-2.0, 0.0, 4.0]))
    assert coord.support_kind == _BOUNDED
    assert (coord.center, coord.scale) == (1.0, 4.0)
    assert coord.canonical_support == (-1.0, 1.0)
    assert np.array_equal(coord.to_canonical([-3.0, 1.0, 5.0]), [-1.0, 0.0, 1.0])
    assert _build_fit_coordinate((-3.0, 5.0), None) == coord


def test_bounded_coordinate_does_not_overflow_near_the_largest_float():
    coord = _build_fit_coordinate((-1.7e308, 1.7e308), None)
    assert coord.center == 0.0 and coord.scale == 1.7e308


@pytest.mark.parametrize(
    "weights",
    [
        np.asarray([1.0, 2.0]),
        np.asarray([1.0, -1.0, 1.0]),
        np.asarray([1.0, np.nan, 1.0]),
        np.zeros(3),
    ],
)
def test_build_fit_coordinate_rejects_malformed_supplied_weights(weights):
    samples = np.asarray([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="weights"):
        _build_fit_coordinate((-np.inf, np.inf), samples, weights)


def test_weighted_quantile_uses_midpoint_cumulative_weights():
    values = np.asarray([3.0, 0.0, 1.0, 2.0])
    # Equal weights are the unweighted rule (numpy's "hazen" positions).
    for levels in ((0.1, 0.9), (0.3, 0.5, 0.77)):
        expected = np.quantile(values, levels, method="hazen")
        assert np.array_equal(_weighted_quantile(values, None, levels), expected)
        assert np.allclose(
            _weighted_quantile(values, np.full(4, 0.25), levels), expected, rtol=1e-15
        )
    # Midpoints (cumsum - w / 2) of weights (1, 1, 2, 4) / 8 on 0 < 1 < 2 < 3.
    w = np.asarray([4.0, 1.0, 1.0, 2.0])
    cdf = np.asarray([0.0625, 0.1875, 0.375, 0.75])
    assert _weighted_quantile(values, w, (0.75,))[0] == pytest.approx(3.0)
    assert _weighted_quantile(values, w, (0.5,))[0] == pytest.approx(
        np.interp(0.5, cdf, [0.0, 1.0, 2.0, 3.0])
    )
    # Zero-weight samples are ignored.
    w0 = np.asarray([0.0, 1.0, 1.0, 1.0])
    assert np.array_equal(
        _weighted_quantile(values, w0, (0.1, 0.9)),
        np.quantile([0.0, 1.0, 2.0], (0.1, 0.9), method="hazen"),
    )


def test_real_line_puts_q10_and_q90_at_minus_one_and_one():
    rng = np.random.default_rng(3)
    samples = rng.gamma(2.0, 1.5, size=501) - 4.0
    coord = _build_fit_coordinate((-np.inf, np.inf), samples)
    q10, q90 = np.quantile(samples, (0.1, 0.9), method="hazen")
    assert coord.support_kind == _REAL_LINE
    assert coord.center == pytest.approx(0.5 * (q10 + q90), rel=1e-15)
    assert coord.scale == pytest.approx(0.5 * (q90 - q10), rel=1e-15)
    assert coord.canonical_support == (-np.inf, np.inf)
    assert np.allclose(coord.to_canonical([q10, q90]), [-1.0, 1.0], rtol=1e-14)


def test_real_line_uses_responsibility_weighted_landmarks():
    # A component owns 0..49; four far samples carry negligible responsibility.
    samples = np.concatenate([np.arange(50.0), [1e3, 2e3, 3e3, 4e3]])
    weights = np.concatenate([np.full(50, 1.0), np.full(4, 1e-7)])
    coord = _build_fit_coordinate((-np.inf, np.inf), samples, weights)
    q10, q90 = _weighted_quantile(samples, weights, (0.1, 0.9))
    assert (coord.center, coord.scale) == (0.5 * q10 + 0.5 * q90, 0.5 * q90 - 0.5 * q10)
    own = np.quantile(np.arange(50.0), (0.1, 0.9), method="hazen")
    assert np.allclose((q10, q90), own, rtol=1e-6)


def test_lower_half_line_anchors_the_endpoint_at_zero():
    samples = np.asarray([2.5, 4.0, 8.0])
    coord = _build_fit_coordinate((2.0, np.inf), samples)
    q90 = np.quantile(samples, 0.9, method="hazen")
    assert coord.support_kind == _LOWER_HALF_LINE
    assert (coord.center, coord.scale) == (2.0, q90 - 2.0)
    assert coord.canonical_support == (0.0, np.inf)
    assert coord.to_canonical(q90) == pytest.approx(1.0, rel=1e-15)


def test_upper_half_line_anchors_the_endpoint_at_zero():
    samples = np.asarray([7.5, 6.0, 2.0])
    coord = _build_fit_coordinate((-np.inf, 8.0), samples)
    q10 = np.quantile(samples, 0.1, method="hazen")
    assert coord.support_kind == _UPPER_HALF_LINE
    assert (coord.center, coord.scale) == (8.0, 8.0 - q10)
    assert coord.canonical_support == (-np.inf, 0.0)
    assert coord.to_canonical(q10) == pytest.approx(-1.0, rel=1e-15)


@pytest.mark.parametrize(
    "support,scale",
    [((-np.inf, np.inf), 2.0), ((0.0, np.inf), 0.5), ((-np.inf, 0.0), 0.25)],
)
def test_coordinate_is_affine_equivariant(support, scale):
    rng = np.random.default_rng(11)
    samples = rng.gamma(2.0, 1.0, size=200) * (1.0 if support[1] == np.inf else -1.0)
    shift = 3.0 if support == (-np.inf, np.inf) else 0.0
    a = _build_fit_coordinate(support, samples)
    b = _build_fit_coordinate(support, scale * samples + shift)
    assert b.scale == pytest.approx(scale * a.scale, rel=1e-15)
    assert b.center == pytest.approx(scale * a.center + shift, rel=1e-15, abs=1e-15)
    assert np.allclose(
        a.to_canonical(samples), b.to_canonical(scale * samples + shift), rtol=1e-13
    )


def test_upper_half_line_interval_mapping_preserves_endpoint_order():
    coord = _build_fit_coordinate((-np.inf, 10.0), np.asarray([7.0, 8.0]))
    intervals = np.asarray([[6.0, 8.0], [8.0, 9.0]])
    got = coord.intervals_to_canonical(intervals)
    assert np.array_equal(got, coord.to_canonical(intervals))
    assert np.all(got[:, 0] <= got[:, 1])


@pytest.mark.parametrize(
    "support,samples",
    [
        ((-np.inf, np.inf), [-3.0, 0.0, 2.0, 7.0]),
        ((0.0, np.inf), [0.5, 1.0, 3.0, 8.0]),
        ((-np.inf, 5.0), [-3.0, 1.0, 4.0]),
        ((-2.0, 6.0), [-1.0, 0.0, 4.0]),
    ],
)
def test_coordinate_round_trip_is_exact_to_float_precision(support, samples):
    x = np.asarray(samples, dtype=float)
    coord = _build_fit_coordinate(support, x)
    reconstructed = coord.from_canonical(coord.to_canonical(x))
    assert np.allclose(
        reconstructed, x, rtol=0.0, atol=2e-15 * max(1.0, np.max(np.abs(x)))
    )


def test_coordinate_avoids_overflow_in_large_opposite_sign_differences():
    samples = np.asarray([-1.0e308, 1.0e308])
    coord = _build_fit_coordinate((-np.inf, np.inf), samples, np.asarray([1.0, 1.0]))
    z = coord.to_canonical(samples)
    assert np.all(np.isfinite(z))
    assert (coord.center, coord.scale) == (0.0, 1.0e308)
    assert np.array_equal(z, [-1.0, 1.0])


@pytest.mark.parametrize(
    "support,samples",
    [
        ((-np.inf, np.inf), [1.0, 1.0, 1.0]),
        ((0.0, np.inf), [0.0, 0.0, 0.0]),
        ((-np.inf, 2.0), [2.0, 2.0, 2.0]),
    ],
)
def test_degenerate_landmarks_are_rejected(support, samples):
    with pytest.raises(ValueError, match="not identified"):
        _build_fit_coordinate(support, np.asarray(samples))


def test_sample_boundary_rejects_empty_or_nonfinite_samples():
    with pytest.raises(ValueError, match="at least two|at least 2"):
        _canon_univariate_samples([])
    with pytest.raises(ValueError, match="only finite"):
        _canon_univariate_samples([0.0, np.nan])


def test_infinite_interval_coordinate_uses_censoring_cutpoints():

    intervals = np.array(
        [
            [-np.inf, -1.0],
            [-0.4, 0.2],
            [0.8, np.inf],
        ]
    )
    coord = _build_interval_fit_coordinate((-np.inf, np.inf), intervals)
    landmarks = np.array([-1.0, -0.1, 0.8])
    z = coord.to_canonical(landmarks)
    assert np.all(np.isfinite(z))
    assert np.std(z) > 0.1


def test_single_half_line_cutpoint_uses_the_endpoint_as_origin():
    intervals = np.array([[3.0, np.inf], [3.0, np.inf]])
    coord = _build_interval_fit_coordinate((0.0, np.inf), intervals)
    assert (coord.center, coord.scale) == (0.0, 3.0)
    assert coord.canonical_support == (0.0, np.inf)


def test_lone_full_line_cutpoint_does_not_identify_a_scale():
    intervals = np.array([[-np.inf, 2.0], [-np.inf, 2.0]])
    with pytest.raises(ValueError, match="not identified"):
        _build_interval_fit_coordinate((-np.inf, np.inf), intervals)


def test_whole_support_rows_carry_no_landmark():
    intervals = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
    with pytest.raises(ValueError, match="no finite landmark"):
        _build_interval_fit_coordinate((-np.inf, np.inf), intervals)
    # On a bounded support the coordinate needs no landmark at all.
    coord = _build_interval_fit_coordinate((0.0, 4.0), intervals.clip(0.0, 4.0))
    assert (coord.center, coord.scale) == (2.0, 2.0)


def test_interval_landmarks_spread_each_row_uniformly():
    # 25% of the weight on [0, 1], 50% on [1, 2] and 25% on [2, 3].
    rows = np.array([[0.0, 1.0], [1.0, 2.0], [1.0, 2.0], [2.0, 3.0]])
    np.testing.assert_allclose(
        _interval_quantiles(rows, np.ones(4), (0.1, 0.5, 0.9)), [0.4, 1.5, 2.6]
    )
    # Atoms (points and one-sided cutpoints) take the generalized inverse.
    rows = np.array([[1.0, 1.0], [2.0, 2.0], [-np.inf, 0.5], [3.0, np.inf]])
    np.testing.assert_allclose(
        _interval_quantiles(rows, np.ones(4), (0.1, 0.9)), [0.5, 3.0]
    )


def test_one_coarse_bin_identifies_the_coordinate():
    # Every row in one bin: midpoints would coincide, the spread rows do not.
    rows = np.tile([0.0, 1.0], (50, 1))
    coord = _build_interval_fit_coordinate((-np.inf, np.inf), rows)
    assert coord.center == pytest.approx(0.5) and coord.scale == pytest.approx(0.4)


def test_interval_landmarks_approach_the_sample_quantiles_under_fine_binning():
    y = np.random.default_rng(0).normal(size=20000)
    lo = np.floor(y * 8.0) / 8.0
    q = _interval_quantiles(np.c_[lo, lo + 0.125], np.ones(lo.size), (0.1, 0.9))
    np.testing.assert_allclose(q, np.quantile(y, [0.1, 0.9]), atol=0.02)


def test_point_rows_use_the_point_rule():
    x = np.random.default_rng(1).normal(size=101)
    rows = np.c_[x, x]
    a = _build_interval_fit_coordinate((-np.inf, np.inf), rows)
    b = _build_fit_coordinate((-np.inf, np.inf), x)
    assert (a.center, a.scale) == (b.center, b.scale)


@pytest.mark.parametrize("support", [(0.0, np.inf), (-np.inf, 2e3), (0.0, 2e3)])
def test_finite_ended_supports_reject_too_concentrated_data(support):
    from gibbus._defaults import COORD_MIN_SPREAD

    rng = np.random.default_rng(4)
    base = 1000.0 + rng.standard_normal(500)
    # Spread just above and just below the limit (relative to h ~ 1000).
    q10, q90 = np.quantile(base, [0.1, 0.9])
    unit = (q90 - q10) / 1000.0
    ok = 1000.0 + (base - 1000.0) * (2.0 * COORD_MIN_SPREAD / unit)
    bad = 1000.0 + (base - 1000.0) * (0.5 * COORD_MIN_SPREAD / unit)
    _build_fit_coordinate(support, ok)
    with pytest.raises(ValueError, match="too concentrated"):
        _build_fit_coordinate(support, bad)
    with pytest.raises(ValueError, match="too concentrated"):
        _build_interval_fit_coordinate(support, np.c_[bad, bad + 1e-6])
    # The full line has no finite end to be concentrated against.
    _build_fit_coordinate((-np.inf, np.inf), bad)


def _censored_component_weights(rows, /):
    """Responsibilities of a component holding the one-sided rows (a tail)."""
    one_sided = np.isfinite(rows[:, 0]) ^ np.isfinite(rows[:, 1])
    return np.where(one_sided, 1.0, 1e-3)


@pytest.mark.parametrize("support", [(-np.inf, np.inf), (0.0, np.inf)])
def test_unidentified_weighted_landmarks_use_the_fallback(support):
    # A tail component: almost all of its weight sits on one cutpoint, so
    # its weighted q10 and q90 coincide although the pooled rows identify
    # the coordinate.
    x = np.random.default_rng(2).gamma(3.0, size=200) + 0.5
    rows = np.c_[x, x]
    rows[x > 4.0] = [4.0, np.inf]
    pooled = _build_interval_fit_coordinate(support, rows)
    w = _censored_component_weights(rows)
    w = w / w.sum()
    with pytest.raises(ValueError, match="not identified|too concentrated"):
        _build_interval_fit_coordinate(support, rows, w)
    assert _build_interval_fit_coordinate(support, rows, w, fallback=pooled) is pooled


def test_identified_weighted_landmarks_ignore_the_fallback():
    x = np.random.default_rng(3).normal(size=300)
    w = np.where(x > 0.0, 1.0, 0.2)
    w = w / w.sum()
    pooled = _build_fit_coordinate((-np.inf, np.inf), x)
    own = _build_fit_coordinate((-np.inf, np.inf), x, w)
    got = _build_fit_coordinate((-np.inf, np.inf), x, w, fallback=pooled)
    assert (got.center, got.scale) == (own.center, own.scale)
    assert (own.center, own.scale) != (pooled.center, pooled.scale)


def test_whole_support_component_weight_uses_the_fallback():
    rows = np.array([[-np.inf, np.inf]] * 5 + [[0.0, 1.0], [1.0, 2.0]])
    pooled = _build_interval_fit_coordinate((-np.inf, np.inf), rows)
    w = np.r_[np.ones(5), 0.0, 0.0] / 5.0
    assert (
        _build_interval_fit_coordinate((-np.inf, np.inf), rows, w, fallback=pooled)
        is pooled
    )


@pytest.mark.parametrize("intervals", [False, True])
def test_mixture_component_of_tied_or_censored_rows_compiles(intervals):
    from gibbus import BasisSpec
    from gibbus._fit.basis_selection import _IntervalData, _PointData

    x = np.random.default_rng(5).normal(size=400)
    if intervals:
        rows = np.c_[x, x]
        rows[x > 0.8] = [0.8, np.inf]
        data = _IntervalData((-np.inf, np.inf), rows, None, float(rows.shape[0]))
        r = _censored_component_weights(rows)
    else:
        # 90% of the component's weight on one tied value.
        x = np.where(x > 0.6, 0.6, x)
        data = _PointData(x, None, float(x.size))
        r = np.where(x == 0.6, 1.0, 1e-3)
    model = data.build_weighted(BasisSpec(support=(-np.inf, np.inf)), r)
    assert np.isfinite(model.geometry.coordinate.scale)
    assert model.geometry.coordinate.scale > 0.0
