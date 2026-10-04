"""Shared inference uses joint nuisance information, not component errors."""

from types import SimpleNamespace

import numpy as np
import pytest

from gibbus._postfit.mixture_inference import (
    _mixture_fit_metadata,
    _shared_standard_errors,
    _single_fit_metadata,
)


def test_shared_errors_include_private_and_logit_nuisance_parameters():
    information = np.array([[4.0, 0.5, 1.0], [0.5, 3.0, 0.75], [1.0, 0.75, 2.0]])
    errors = _shared_standard_errors(information, [0, 1, 2], (2, None), 120.0)
    expected = np.sqrt(np.linalg.inv(information)[2, 2] / 120.0)
    assert errors[0] == pytest.approx(expected)
    assert errors[0] > np.sqrt(1.0 / information[2, 2] / 120.0)
    assert np.isnan(errors[1])


def test_exact_face_excludes_fixed_coordinates_from_uncertainty():
    information = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 0.0]])
    errors = _shared_standard_errors(information, [0, 1], (1, 2), 50.0)
    assert errors[0] == pytest.approx(
        np.sqrt(np.linalg.inv(information[:2, :2])[1, 1] / 50.0)
    )
    assert np.isnan(errors[1])


@pytest.mark.parametrize("unresolved", [0.0, -1.0])
def test_nonpositive_shared_information_is_reported_as_unresolved(unresolved):
    errors = _shared_standard_errors(
        np.diag([1.0, unresolved]), [0, 1], (1, None), 100.0
    )
    assert np.isinf(errors[0])


def test_unrelated_unidentified_direction_does_not_destroy_shared_information():
    errors = _shared_standard_errors(np.diag([0.0, 4.0]), [0, 1], (1, None), 25.0)
    assert errors[0] == pytest.approx(0.1)


def test_nuisance_coordinate_units_do_not_change_shared_uncertainty():
    information = np.array([[4.0, 1.0], [1.0, 2.0]])
    original = _shared_standard_errors(information, [0, 1], (1, None), 100.0)
    units = np.diag([1e12, 1.0])
    scaled = _shared_standard_errors(
        units @ information @ units, [0, 1], (1, None), 100.0
    )
    np.testing.assert_allclose(original, scaled, equal_nan=True)


def test_shared_errors_scale_with_effective_sample_size():
    information = np.array([[4.0, 1.0], [1.0, 2.0]])
    first = _shared_standard_errors(information, [0, 1], (1, None), 100.0)
    second = _shared_standard_errors(information, [0, 1], (1, None), 400.0)
    assert second[0] == pytest.approx(first[0] / 2.0)


def test_two_shared_sides_use_the_same_full_nuisance_adjusted_covariance():
    factor = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.3, 1.0, 0.0, 0.0],
            [0.4, 0.2, 1.5, 0.0],
            [0.5, 0.1, 0.4, 0.8],
        ]
    )
    information = factor @ factor.T
    errors = _shared_standard_errors(information, [0, 1, 2, 3], (1, 2), 80.0)
    expected = np.sqrt(np.diag(np.linalg.inv(information))[[1, 2]] / 80.0)
    np.testing.assert_allclose(errors, expected)


@pytest.mark.parametrize("free", [[0, 0], [-1], [2]])
def test_shared_errors_reject_invalid_face_indices(free):
    with pytest.raises(ValueError, match="free parameter"):
        _shared_standard_errors(np.eye(2), free, (1, None), 100.0)


def _component(amplitude, *, reflected=False, active=True):
    return SimpleNamespace(
        spec=SimpleNamespace(
            physical_lower_a_index=None if reflected else 2,
            physical_upper_a_index=2 if reflected else None,
        ),
        layout=SimpleNamespace(lower_a_index=2, upper_a_index=None),
        params=np.array([0.0, 1.0, amplitude]),
        lower_amplitude_active=active,
        upper_amplitude_active=False,
    )


def _fit(components, *, reflected=False):
    return SimpleNamespace(
        components=components,
        n_parameters=6,
        n_face_parameters=6,
        shared_parameter_indices=(None, 4) if reflected else (4, None),
        free_parameter_indices=np.arange(6),
        observed_information=np.eye(6),
    )


def test_metadata_uses_physical_sides_for_reflected_components():
    fit = _fit([_component(0.5, reflected=True)] * 2, reflected=True)
    metadata = _mixture_fit_metadata(fit, 100.0, (np.nan, 0.01))
    shared = metadata["shared_boundary"]
    assert shared["allowed"] == (False, True)
    assert shared["active"] == (False, True)
    assert np.isnan(shared["amplitudes"][0])
    assert shared["amplitudes"][1] == 0.5
    assert shared["standard_errors"][1] == pytest.approx(0.1)
    assert metadata["n_face_parameters"] == 6


def test_metadata_rejects_unequal_amplitudes_in_an_alleged_shared_fit():
    fit = _fit([_component(0.5), _component(0.6)])
    with pytest.raises(ValueError, match="disagree"):
        _mixture_fit_metadata(fit, 100.0, (np.nan, np.nan))


def test_metadata_does_not_infer_zero_face_from_an_arbitrary_threshold():
    fit = _fit([_component(1e-14, active=False)] * 2)
    with pytest.raises(ValueError, match="exactly zero"):
        _mixture_fit_metadata(fit, 100.0, (np.nan, np.nan))


def test_single_metadata_separates_layout_dimension_and_reflected_exact_face():
    state = {
        "boundary_allowed": [False, True],
        "boundary_amplitudes": [np.nan, 0.2],
        "lower_amplitude_active": True,
        "upper_amplitude_active": False,
        "fit_direction": -1.0,
        "boundary_standard_errors": [np.nan, 0.3],
        "boundary_p_values": [np.nan, 0.02],
        "optimizer_params": np.zeros(7),
        "effective_curvature_degree": 2,
    }
    metadata = _single_fit_metadata(state)
    assert metadata["n_parameters"] == 7
    assert metadata["n_face_parameters"] == 5
    assert metadata["shared_boundary"]["active"] == (False, True)
    assert metadata["shared_boundary"]["weakly_identified"] == ("upper",)
