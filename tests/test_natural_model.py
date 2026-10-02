"""Definition-level tests for the affine natural-coordinate model layout."""

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyder, polyval

from gibbus._model.natural import (
    _BOUNDED,
    _LOWER_HALF_LINE,
    _REAL_LINE,
    _UPPER_HALF_LINE,
    _natural_layout,
)


@pytest.mark.parametrize(
    "support,degree,lower_enabled,upper_enabled,kind,effective,n_params",
    [
        ((-np.inf, np.inf), 2, False, False, _REAL_LINE, 2, 2),
        ((-np.inf, np.inf), 3, False, False, _REAL_LINE, 2, 2),
        ((-np.inf, np.inf), 8, False, False, _REAL_LINE, 8, 8),
        ((0.0, np.inf), 2, False, False, _LOWER_HALF_LINE, 2, 2),
        ((0.0, np.inf), 7, True, False, _LOWER_HALF_LINE, 7, 8),
        ((-np.inf, 3.0), 5, False, True, _UPPER_HALF_LINE, 5, 6),
        ((-2.0, 4.0), 6, False, False, _BOUNDED, 6, 6),
        ((-2.0, 4.0), 6, True, True, _BOUNDED, 6, 8),
    ],
)
def test_natural_layout_has_intrinsic_parameter_count(
    support, degree, lower_enabled, upper_enabled, kind, effective, n_params
):
    layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    assert layout.support_kind == kind
    assert layout.effective_poly_degree == effective
    assert layout.curvature_degree == effective - 2
    assert layout.n_params == n_params
    assert layout.gamma_index == 0
    assert layout.curvature_slice == slice(1, effective)


@pytest.mark.parametrize(
    "support,lower_enabled,upper_enabled,message",
    [
        ((-np.inf, np.inf), True, False, "finite lower"),
        ((-np.inf, np.inf), False, True, "finite upper"),
        ((0.0, np.inf), False, True, "finite upper"),
        ((-np.inf, 0.0), True, False, "finite lower"),
    ],
)
def test_natural_layout_rejects_boundary_basis_at_infinite_endpoint(
    support, lower_enabled, upper_enabled, message
):
    with pytest.raises(ValueError, match=message):
        _natural_layout(support, 4, lower_enabled, upper_enabled)


def test_natural_layout_rejects_invalid_degree_and_support():
    with pytest.raises(ValueError, match="poly_degree"):
        _natural_layout((-np.inf, np.inf), 1)
    with pytest.raises(ValueError, match="increasing"):
        _natural_layout((1.0, 1.0), 4)
    with pytest.raises(ValueError, match="outward"):
        _natural_layout((np.inf, np.inf), 4)


@pytest.mark.parametrize(
    "support,degree,lower_enabled,upper_enabled",
    [
        ((-np.inf, np.inf), 2, False, False),
        ((-np.inf, np.inf), 8, False, False),
        ((-1.5, np.inf), 7, True, False),
        ((-np.inf, 2.5), 7, False, True),
        ((-2.0, 3.0), 7, False, False),
        ((-2.0, 3.0), 7, True, False),
        ((-2.0, 3.0), 7, False, True),
        ((-2.0, 3.0), 7, True, True),
    ],
)
def test_pack_unpack_round_trip(support, degree, lower_enabled, upper_enabled):
    layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    rng = np.random.default_rng(21000 + degree + layout.n_params)
    curvature = rng.normal(size=layout.curvature_degree + 1)
    amplitudes = np.array([0.3, 0.7])
    raw = layout.pack(-0.4, curvature, amplitudes)
    gamma, recovered_curvature, recovered_amplitudes = layout.unpack(raw)
    assert gamma == -0.4
    np.testing.assert_array_equal(recovered_curvature, curvature)
    if lower_enabled:
        assert recovered_amplitudes[0] == amplitudes[0]
    else:
        assert np.isnan(recovered_amplitudes[0])
    if upper_enabled:
        assert recovered_amplitudes[1] == amplitudes[1]
    else:
        assert np.isnan(recovered_amplitudes[1])


@pytest.mark.parametrize(
    "support,degree,lower_enabled,upper_enabled",
    [
        ((-np.inf, np.inf), 8, False, False),
        ((-1.0, np.inf), 7, True, False),
        ((-np.inf, 2.0), 7, False, True),
        ((-1.0, 2.0), 7, True, True),
    ],
)
def test_candidate_polynomial_derivatives_recover_natural_curvature(
    support, degree, lower_enabled, upper_enabled
):
    layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    rng = np.random.default_rng(22000 + degree)
    curvature = rng.normal(size=layout.curvature_degree + 1)
    raw = layout.pack(0.37, curvature, np.array([0.2, 0.4]))
    candidate = layout.build_candidate(raw)
    np.testing.assert_allclose(candidate.q_d2, curvature, rtol=0.0, atol=5e-16)
    np.testing.assert_allclose(polyder(candidate.q_poly), candidate.q_d1, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(polyder(candidate.q_d1), candidate.q_d2, rtol=0.0, atol=5e-16)
    assert candidate.q_poly[0] == 0.0
    assert candidate.q_poly[1] == 0.37


@pytest.mark.parametrize(
    "support,lower_enabled,upper_enabled,points",
    [
        ((-np.inf, np.inf), False, False, np.array([-2.0, -0.2, 1.7])),
        ((-1.0, np.inf), True, False, np.array([-0.8, 0.3, 3.0])),
        ((-np.inf, 2.0), False, True, np.array([-3.0, 0.1, 1.8])),
        ((-1.0, 2.0), True, True, np.array([-0.8, 0.1, 1.7])),
    ],
)
def test_full_derivative_evaluators_match_finite_differences(
    support, lower_enabled, upper_enabled, points
):
    layout = _natural_layout(support, 6, lower_enabled, upper_enabled)
    curvature = np.array([0.8, -0.15, 0.3, 0.04, 0.02])
    raw = layout.pack(-0.2, curvature, np.array([0.35, 0.55]))
    candidate = layout.build_candidate(raw)

    h = 2e-5
    q_up = candidate.q(points + h)
    q_dn = candidate.q(points - h)
    fd_d1 = (q_up - q_dn) / (2.0 * h)
    np.testing.assert_allclose(candidate.q_d1_full(points), fd_d1, rtol=3e-8, atol=3e-8)

    d1_up = candidate.q_d1_full(points + h)
    d1_dn = candidate.q_d1_full(points - h)
    fd_d2 = (d1_up - d1_dn) / (2.0 * h)
    np.testing.assert_allclose(candidate.q_d2_full(points), fd_d2, rtol=3e-8, atol=3e-8)

    d2_up = candidate.q_d2_full(points + h)
    d2_dn = candidate.q_d2_full(points - h)
    fd_d3 = (d2_up - d2_dn) / (2.0 * h)
    np.testing.assert_allclose(candidate.q_d3_full(points), fd_d3, rtol=2e-7, atol=2e-7)


def test_boundary_amplitudes_are_direct_nonnegative_coordinates():
    layout = _natural_layout((-1.0, 2.0), 4, True, True)
    curvature = np.array([1.0, 0.0, 0.5])
    raw = layout.pack(0.0, curvature, np.array([0.0, 1e-300]))
    assert raw[layout.lower_a_index] == 0.0
    assert raw[layout.upper_a_index] == 1e-300

    bad = raw.copy()
    bad[layout.lower_a_index] = -np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="boundary amplitudes"):
        layout.validate_params(bad)


def test_exact_zero_boundary_amplitude_is_regular_at_endpoint():
    layout = _natural_layout((-1.0, 2.0), 4, True, True)
    curvature = np.array([0.8, -0.2, 0.1])
    candidate = layout.build_candidate(
        layout.pack(0.3, curvature, np.array([0.0, 0.0]))
    )
    lower, upper = layout.support
    assert np.isfinite(candidate.q(lower))
    assert np.isfinite(candidate.q_d1_full(lower))
    assert candidate.q_d2_full(lower) == pytest.approx(polyval(lower, curvature))
    assert np.isfinite(candidate.q(upper))
    assert np.isfinite(candidate.q_d1_full(upper))
    assert candidate.q_d2_full(upper) == pytest.approx(polyval(upper, curvature))


def test_tiny_positive_boundary_amplitude_remains_singular_at_endpoint():
    layout = _natural_layout((-1.0, 2.0), 4, True, False)
    curvature = np.array([0.8, -0.2, 0.1])
    candidate = layout.build_candidate(
        layout.pack(0.3, curvature, np.array([1e-300, np.nan]))
    )
    assert candidate.q(layout.support_lower) == np.inf
    assert candidate.q_d1_full(layout.support_lower) == -np.inf
    assert candidate.q_d2_full(layout.support_lower) == np.inf

@pytest.mark.parametrize(
    "support,lower_enabled,upper_enabled,points",
    [
        ((-np.inf, np.inf), False, False, np.array([-1.5, 0.2, 2.0])),
        ((-1.0, np.inf), True, False, np.array([-0.7, 0.2, 2.0])),
        ((-np.inf, 2.0), False, True, np.array([-2.0, 0.2, 1.7])),
        ((-1.0, 2.0), True, True, np.array([-0.7, 0.2, 1.7])),
    ],
)
def test_full_potential_is_affine_in_natural_coordinates(
    support, lower_enabled, upper_enabled, points
):
    layout = _natural_layout(support, 6, lower_enabled, upper_enabled)
    curvature_a = np.array([0.7, -0.2, 0.1, 0.03, 0.01])
    curvature_b = np.array([1.2, 0.1, 0.05, -0.02, 0.04])
    amplitudes_a = np.array([0.2, 0.4])
    amplitudes_b = np.array([0.6, 0.1])
    theta_a = layout.pack(-0.3, curvature_a, amplitudes_a)
    theta_b = layout.pack(0.8, curvature_b, amplitudes_b)
    weight = 0.37
    theta_mix = weight * theta_a + (1.0 - weight) * theta_b

    candidate_a = layout.build_candidate(theta_a)
    candidate_b = layout.build_candidate(theta_b)
    candidate_mix = layout.build_candidate(theta_mix)
    for evaluator in ("q", "q_d1_full", "q_d2_full"):
        lhs = getattr(candidate_mix, evaluator)(points)
        rhs = (
            weight * getattr(candidate_a, evaluator)(points)
            + (1.0 - weight) * getattr(candidate_b, evaluator)(points)
        )
        np.testing.assert_allclose(lhs, rhs, rtol=2e-14, atol=2e-14)
