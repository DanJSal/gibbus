"""Tests for the model specification and the normalized natural state."""

import numpy as np
import pytest
from scipy.integrate import quad

from gibbus._model.coords import _build_fit_coordinate
from gibbus._model.natural_state import _NaturalCoreState
from gibbus._model.spec import _LOGDIST, _POLY, _build_model_spec
from gibbus._model.vec import _q_eval


def _direct_integrals(state, max_order=2):
    """Compute direct support integrals for one candidate state."""
    lo, hi = state.spec.support

    def kernel(z):
        return np.exp(
            -_q_eval(z, state.spec.support, state.q_poly, state.boundary_amplitudes, 0)
        )

    values = []
    for k in range(max_order + 1):
        val = quad(
            lambda z, k=k: z**k * kernel(z),
            lo,
            hi,
            epsabs=2e-11,
            epsrel=2e-11,
            limit=300,
        )[0]
        values.append(val)
    return np.asarray(values)


def _state(spec, params, z_bounds, /):
    return _NaturalCoreState(spec.coordinate, spec.layout, params, z_bounds)


def test_full_line_model_spec_uses_effective_even_degree():
    coord = _build_fit_coordinate(
        (-np.inf, np.inf), np.array([-2.0, -0.2, 0.4, 3.0]), None, None
    )
    spec = _build_model_spec(coord, 5)
    assert spec.requested_poly_degree == 5
    assert spec.effective_poly_degree == 4
    assert spec.n_params == 4  # gamma, c_0, c_1, c_2
    assert spec.canonical_lower_a_index is None
    assert spec.canonical_upper_a_index is None
    assert spec.support == (-np.inf, np.inf)


def test_bounded_boundary_layout_appends_nonnegative_amplitudes():
    coord = _build_fit_coordinate((-3.0, 5.0), np.array([-2.0, 0.0, 4.0]), None, None)
    spec = _build_model_spec(coord, 4, True, True)
    assert spec.canonical_lower_a_index == spec.n_params - 2
    assert spec.canonical_upper_a_index == spec.n_params - 1
    assert spec.physical_lower_a_index == spec.canonical_lower_a_index
    assert spec.physical_upper_a_index == spec.canonical_upper_a_index

    params = np.zeros(spec.n_params)
    params[1] = 1.0
    state = _state(spec, params, (-1.0, 1.0))
    np.testing.assert_array_equal(state.boundary_amplitudes, [0.0, 0.0])
    assert state.partials[-2].kind == _LOGDIST
    assert state.partials[-1].kind == _LOGDIST

    bad = params.copy()
    bad[-1] = -1e-12
    with pytest.raises(ValueError, match="boundary amplitudes"):
        _state(spec, bad, (-1.0, 1.0))


def test_reflected_upper_half_line_maps_physical_upper_to_canonical_lower():
    coord = _build_fit_coordinate(
        (-np.inf, 10.0), np.array([2.0, 6.0, 9.0]), None, None
    )
    spec = _build_model_spec(coord, 3, False, True)
    assert spec.canonical_lower_a_index is not None
    assert spec.canonical_upper_a_index is None
    assert spec.physical_upper_a_index == spec.canonical_lower_a_index
    assert spec.physical_lower_a_index is None

    params = np.zeros(spec.n_params)
    params[1] = 1.0
    params[2] = 0.5
    params[-1] = 0.75
    state = _state(spec, params, (spec.support[0] + 0.5, spec.support[0] + 3.0))
    assert np.allclose(state.boundary_amplitudes, [0.75, np.nan], equal_nan=True)


def test_boundary_request_on_infinite_physical_side_is_rejected():
    coord = _build_fit_coordinate((0.0, np.inf), np.array([0.5, 1.0, 2.0]), None, None)
    with pytest.raises(ValueError, match="finite upper"):
        _build_model_spec(coord, 3, False, True)


def test_first_partials_are_the_fixed_natural_basis():
    coord = _build_fit_coordinate((-1.0, 2.0), np.array([-0.5, 0.2, 1.2]), None, None)
    spec = _build_model_spec(coord, 4, True, True)
    params = np.zeros(spec.n_params)
    params[1] = 1.0
    state = _state(spec, params, (-0.5, 0.5))
    curvature = spec.layout.curvature_degree + 1
    assert len(state.partials) == spec.n_params
    assert all(p.kind == _POLY for p in state.partials[: 1 + curvature])
    assert all(p.kind == _LOGDIST for p in state.partials[1 + curvature :])
    # gamma multiplies z; c_k multiplies z^(k+2) / ((k+1)(k+2)).
    np.testing.assert_array_equal(state.partials[0].coefficients, [0, 1, 0, 0, 0])
    for k in range(curvature):
        expected = np.zeros(5)
        expected[k + 2] = 1.0 / ((k + 1.0) * (k + 2.0))
        np.testing.assert_array_equal(state.partials[1 + k].coefficients, expected)
    # The potential is exactly theta . t(z).
    theta = np.array([0.3, 0.9, -0.2, 0.4, 0.25, 0.1])
    state = _state(spec, theta, (-0.5, 0.5))
    z = np.linspace(spec.support[0] + 0.1, spec.support[1] - 0.1, 7)
    basis = np.array([p.evaluate(z, spec.support) for p in state.partials])
    np.testing.assert_allclose(
        theta @ basis,
        _q_eval(z, spec.support, state.q_poly, state.boundary_amplitudes, 0),
        rtol=1e-13,
        atol=1e-13,
    )


def test_boundary_partial_descriptors_evaluate_fixed_zero_offset_logs():
    coord = _build_fit_coordinate((-2.0, 6.0), np.array([-1.0, 0.0, 4.0]), None, None)
    spec = _build_model_spec(coord, 2, True, True)
    state = _state(spec, np.array([0.0, 1.0, 0.2, 0.3]), (-0.5, 0.5))
    z = np.array([-0.5, 0.0, 0.5])
    lower = state.partials[-2].evaluate(z, spec.support)
    upper = state.partials[-1].evaluate(z, spec.support)
    lo, hi = spec.support
    assert np.allclose(lower, -np.log(z - lo))
    assert np.allclose(upper, -np.log(hi - z))


@pytest.mark.parametrize(
    "support,samples,degree,params,boundary_flags,z_bounds",
    [
        (
            (-np.inf, np.inf),
            [-2.0, 0.0, 1.0, 3.0],
            2,
            [0.25, 1.0],
            (False, False),
            (-3.0, 4.0),
        ),
        (
            (0.0, np.inf),
            [0.5, 1.0, 2.0, 4.0],
            3,
            [0.35, 0.8, 0.3, 0.4],
            (True, False),
            (-1.0, 3.0),
        ),
        (
            (-2.0, 3.0),
            [-1.5, -0.2, 1.1, 2.5],
            4,
            [0.2, 0.6, 0.1, 0.3, 0.4, 0.7],
            (True, True),
            (-1.0, 1.0),
        ),
    ],
)
def test_normalization_and_moments_match_direct_integrals(
    support, samples, degree, params, boundary_flags, z_bounds
):
    coord = _build_fit_coordinate(support, np.asarray(samples, dtype=float), None, None)
    spec = _build_model_spec(coord, degree, *boundary_flags)
    params = np.asarray(params, dtype=float)
    assert params.size == spec.n_params
    state = _state(spec, params, z_bounds)
    direct = _direct_integrals(state, max_order=2)

    assert np.isfinite(state.log_Z)
    assert np.exp(state.log_Z) == pytest.approx(direct[0], rel=2e-8, abs=2e-10)
    assert state.moment_raw(1) == pytest.approx(
        direct[1] / direct[0], rel=2e-8, abs=2e-10
    )
    assert state.moment_raw(2) == pytest.approx(
        direct[2] / direct[0], rel=3e-8, abs=3e-10
    )


def test_state_pdf_normalizes_on_canonical_support():
    coord = _build_fit_coordinate(
        (-np.inf, np.inf), np.array([-2.0, 0.0, 1.0, 3.0]), None, None
    )
    spec = _build_model_spec(coord, 2)
    state = _state(spec, np.array([-0.3, 1.0]), (-3.0, 4.0))
    mass = quad(
        lambda z: state.pdf(z), -np.inf, np.inf, epsabs=2e-10, epsrel=2e-10, limit=300
    )[0]
    assert mass == pytest.approx(1.0, rel=2e-8, abs=2e-9)
