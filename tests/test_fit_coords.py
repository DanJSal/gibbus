"""Definition-level tests for fixed fitting-coordinate construction."""
import numpy as np
import pytest

from gibbus._defaults import MAD_TO_SIGMA, UNIFORM_WIDTH_TO_SIGMA
from gibbus._model.coords import (
    _BOUNDED,
    _LOWER_HALF_LINE,
    _REAL_LINE,
    _UPPER_HALF_LINE,
    _build_fit_coordinate,
    _build_interval_fit_coordinate,
    _support_kind,
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
def test_support_kind_rejects_invalid_support(support):
    with pytest.raises(ValueError):
        _support_kind(support)


def test_bounded_coordinate_is_data_centered_and_maps_support_affinely():
    coord = _build_fit_coordinate(
        (-3.0, 5.0), np.asarray([-2.0, 0.0, 4.0]), None, None)
    assert coord.support_kind == _BOUNDED
    assert coord.center == pytest.approx(0.0)
    assert coord.scale == pytest.approx(2.0 * MAD_TO_SIGMA)
    assert coord.direction == 1.0
    expected = (-3.0 / coord.scale, 5.0 / coord.scale)
    assert coord.canonical_support == pytest.approx(expected)
    assert np.allclose(
        coord.to_canonical([-3.0, 0.0, 5.0]),
        [expected[0], 0.0, expected[1]],
    )



def test_bounded_coordinate_caps_extreme_canonical_support_span():
    samples = np.asarray([1.0e-18, 2.0e-18, 3.0e-18])
    coord = _build_fit_coordinate((0.0, 1.0), samples, None, None)

    span = coord.canonical_support[1] - coord.canonical_support[0]
    assert span == pytest.approx(1.0e6, rel=2e-15)
    assert coord.scale == pytest.approx(1.0e-6, rel=2e-15)
    assert coord.center == pytest.approx(2.0e-18, rel=0.0, abs=0.0)

def test_real_line_uses_weighted_median_and_weighted_mad():
    samples = np.asarray([0.0, 1.0, 2.0, 100.0, 200.0, 300.0, 400.0])
    weights = np.asarray([0.2, 0.31, 0.39, 0.025, 0.025, 0.025, 0.025])
    coord = _build_fit_coordinate((-np.inf, np.inf), samples, weights, None)
    assert coord.support_kind == _REAL_LINE
    assert coord.center == pytest.approx(1.0)
    assert coord.scale == pytest.approx(1.0 * MAD_TO_SIGMA)
    assert coord.direction == 1.0
    assert coord.canonical_support == (-np.inf, np.inf)


def test_real_line_scale_is_not_controlled_by_negligible_responsibilities():
    samples = np.asarray([0.0, 1.0, 2.0, 100.0, 200.0, 300.0, 400.0])
    weights = np.asarray([0.2, 0.31, 0.39, 0.025, 0.025, 0.025, 0.025])
    coord = _build_fit_coordinate((-np.inf, np.inf), samples, weights, None)
    unweighted = MAD_TO_SIGMA * np.median(np.abs(samples - coord.center))
    assert coord.scale < unweighted / 20.0


def test_full_line_interval_width_supplies_scale_floor():
    samples = np.asarray([0.0, 0.0, 0.0])
    widths = np.asarray([4.0, 4.0, 4.0])
    coord = _build_fit_coordinate((-np.inf, np.inf), samples, None, widths)
    assert coord.scale == pytest.approx(4.0 * UNIFORM_WIDTH_TO_SIGMA)


def test_lower_half_line_is_data_centered_with_mapped_endpoint():
    samples = np.asarray([2.0, 4.0, 8.0])
    coord = _build_fit_coordinate((2.0, np.inf), samples, None, None)
    assert coord.support_kind == _LOWER_HALF_LINE
    assert coord.center == pytest.approx(4.0)
    assert coord.scale == pytest.approx(2.0 * MAD_TO_SIGMA)
    assert coord.direction == 1.0
    assert coord.canonical_support == pytest.approx((-2.0 / coord.scale, np.inf))
    assert np.allclose(
        coord.to_canonical(samples), (samples - coord.center) / coord.scale
    )

def test_half_line_scale_uses_weighted_mad_about_weighted_center():
    samples = np.asarray([1.0, 2.0, 3.0, 100.0, 200.0, 300.0, 400.0])
    weights = np.asarray([0.2, 0.31, 0.39, 0.025, 0.025, 0.025, 0.025])
    coord = _build_fit_coordinate((0.0, np.inf), samples, weights, None)
    assert coord.center == pytest.approx(2.0)
    assert coord.scale == pytest.approx(1.0 * MAD_TO_SIGMA)

def test_half_line_interval_width_supplies_scale_floor():
    samples = np.asarray([0.01, 0.02, 0.03])
    widths = np.asarray([2.0, 2.0, 2.0])
    coord = _build_fit_coordinate((0.0, np.inf), samples, None, widths)
    assert coord.scale == pytest.approx(2.0 * UNIFORM_WIDTH_TO_SIGMA)


def test_upper_half_line_reflects_about_data_center():
    samples = np.asarray([8.0, 6.0, 2.0])
    coord = _build_fit_coordinate((-np.inf, 8.0), samples, None, None)
    assert coord.support_kind == _UPPER_HALF_LINE
    assert coord.center == pytest.approx(6.0)
    assert coord.scale == pytest.approx(2.0 * MAD_TO_SIGMA)
    assert coord.direction == -1.0
    assert coord.canonical_support == pytest.approx((-2.0 / coord.scale, np.inf))
    expected = -(samples - coord.center) / coord.scale
    assert np.allclose(coord.to_canonical(samples), expected)

def test_upper_half_line_interval_mapping_preserves_endpoint_order():
    coord = _build_fit_coordinate(
        (-np.inf, 10.0), np.asarray([7.0, 8.0]), None, None)
    intervals = np.asarray([[6.0, 8.0], [8.0, 9.0]])
    got = coord.intervals_to_canonical(intervals)
    direct = coord.to_canonical(intervals)
    expected = direct[:, ::-1]
    assert np.array_equal(got, expected)
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
    coord = _build_fit_coordinate(support, x, None, None)
    reconstructed = coord.from_canonical(coord.to_canonical(x))
    assert np.allclose(reconstructed, x, rtol=0.0, atol=2e-15 * max(1.0, np.max(np.abs(x))))


def test_coordinate_avoids_overflow_in_large_opposite_sign_differences():
    samples = np.asarray([-1.0e308, 1.0e308])
    coord = _build_fit_coordinate(
        (-np.inf, np.inf), samples, np.asarray([1.0, 1.0]), None)
    z = coord.to_canonical(samples)
    assert np.all(np.isfinite(z))
    assert coord.scale == pytest.approx(np.sqrt(2.0) * 1.0e308, rel=2e-15)
    assert z == pytest.approx([0.0, np.sqrt(2.0)], rel=2e-15)


@pytest.mark.parametrize(
    "support,samples",
    [
        ((-np.inf, np.inf), [1.0, 1.0, 1.0]),
        ((0.0, np.inf), [0.0, 0.0, 0.0]),
        ((-np.inf, 2.0), [2.0, 2.0, 2.0]),
    ],
)
def test_data_derived_coordinate_rejects_zero_scale(support, samples):
    with pytest.raises(ValueError, match="robust scale is zero"):
        _build_fit_coordinate(support, np.asarray(samples), None, None)


def test_coordinate_rejects_empty_or_nonfinite_samples():
    with pytest.raises(ValueError, match="non-empty finite"):
        _build_fit_coordinate((-np.inf, np.inf), np.asarray([]), None, None)
    with pytest.raises(ValueError, match="non-empty finite"):
        _build_fit_coordinate((-np.inf, np.inf), np.asarray([0.0, np.nan]), None, None)


def test_infinite_interval_coordinate_uses_censoring_cutpoints():

    intervals = np.array([
        [-np.inf, -1.0],
        [-0.4, 0.2],
        [0.8, np.inf],
    ])
    coord = _build_interval_fit_coordinate((-np.inf, np.inf), intervals)
    landmarks = np.array([-1.0, -0.1, 0.8])
    z = coord.to_canonical(landmarks)
    assert np.all(np.isfinite(z))
    assert np.std(z) > 0.1


def test_single_half_line_cutpoint_uses_physical_endpoint_as_scale_anchor():

    intervals = np.array([[3.0, np.inf], [3.0, np.inf]])
    coord = _build_interval_fit_coordinate((0.0, np.inf), intervals)
    assert coord.center == pytest.approx(3.0)
    assert coord.scale == pytest.approx(3.0)
    assert coord.canonical_support[0] == pytest.approx(-1.0)


def test_lone_full_line_cutpoint_has_no_principled_affine_scale():

    intervals = np.array([[-np.inf, 2.0], [-np.inf, 2.0]])
    with pytest.raises(ValueError, match="finite numerical scale|affine scale"):
        _build_interval_fit_coordinate((-np.inf, np.inf), intervals)
