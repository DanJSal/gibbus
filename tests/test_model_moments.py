"""Tests for the batch ordinary/generalized model-moment service."""

import warnings

import numpy as np
import pytest
from gibbus._model._state_kernels import _q_window_and_mode
from scipy.integrate import IntegrationWarning, quad

from gibbus._defaults import (
    BACKTRACK_MAX_ITERS,
    BACKTRACK_REDUCE,
    BOUNDARY_EPS_MULT,
    BRACKET_INIT_STEP,
    BRACKET_MAX_EXPAND,
    BRACKET_STEP_GROWTH,
    GRAD_TOL,
    HESS_TOL,
    LOG_THRESH,
    NEWT_MAX,
    NEWT_TOL,
)
from gibbus._model.coords import _build_fit_coordinate
from gibbus._model.natural_state import _NaturalCoreState
from gibbus._model.spec import _build_model_spec
from gibbus._model.vec import _q_eval


def _state(support, samples, degree, params, lower=False, upper=False):
    """Build one test state from natural parameters in canonical coordinates."""
    coord = _build_fit_coordinate(support, np.asarray(samples), None, None)
    spec = _build_model_spec(coord, degree, lower, upper)
    if np.isfinite(spec.support[0]) and np.isfinite(spec.support[1]):
        bounds = spec.support
    elif np.isfinite(spec.support[0]):
        bounds = (spec.support[0], 5.0)
    elif np.isfinite(spec.support[1]):
        bounds = (-5.0, spec.support[1])
    else:
        bounds = (-4.0, 4.0)
    return _NaturalCoreState(
        coord, spec.layout, np.asarray(params, dtype=float), bounds
    )


def _direct_expectation(state, func):
    """Compute one reference expectation over the full canonical support."""
    lo, hi = state.spec.support

    def kernel(z):
        q = _q_eval(z, state.spec.support, state.q_poly, state.boundary_amplitudes, 0)
        return func(z) * np.exp(-q)

    numerator = quad(kernel, lo, hi, epsabs=2e-11, epsrel=2e-11, limit=300)[0]
    denominator = quad(
        lambda z: np.exp(
            -_q_eval(z, state.spec.support, state.q_poly, state.boundary_amplitudes, 0)
        ),
        lo,
        hi,
        epsabs=2e-11,
        epsrel=2e-11,
        limit=300,
    )[0]
    return numerator / denominator


def test_power_block_matches_scalar_state_moments_and_covariance_definition():
    state = _state((-np.inf, np.inf), [-2.0, 0.0, 1.0, 3.0], 4, [0.15, 0.9, 0.25, 0.4])
    moments = state.moments.power(6)
    assert moments[0] == 1.0
    for k in range(1, 7):
        assert moments[k] == pytest.approx(state.moment_raw(k), rel=0.0, abs=0.0)

    cov = state.moments.power_covariance(3)
    for i in range(4):
        for j in range(4):
            expected = moments[i + j] - moments[i] * moments[j]
            assert cov[i, j] == pytest.approx(expected, rel=2e-14, abs=2e-14)
    assert np.allclose(cov, cov.T, rtol=0.0, atol=2e-14)


def test_log_power_remains_available_at_exact_zero_boundary_amplitude():
    state = _state((0.0, np.inf), [0.4, 1.0, 2.0, 4.0], 2, [0.6, 1.0, 0.0], lower=True)
    got = state.moments.log_power("lower", 2)
    for k in range(3):
        lo = float(state.spec.support[0])
        expected = _direct_expectation(
            state, lambda z, k=k, lo=lo: z**k * np.log(z - lo)
        )
        assert got[k] == pytest.approx(expected, rel=3e-8, abs=3e-9)


def test_log_square_and_cross_match_direct_bounded_integrals():
    state = _state(
        (-2.0, 3.0),
        [-1.5, -0.2, 1.1, 2.5],
        4,
        [0.1, 0.7, 0.2, -0.15, 0.4, 0.8],
        lower=True,
        upper=True,
    )
    lower_sq = state.moments.log_square("lower")
    upper_sq = state.moments.log_square("upper")
    cross = state.moments.log_cross()
    lo, hi = state.spec.support
    assert lower_sq == pytest.approx(
        _direct_expectation(state, lambda z: np.log(z - lo) ** 2),
        rel=2e-8,
        abs=2e-9,
    )
    assert upper_sq == pytest.approx(
        _direct_expectation(state, lambda z: np.log(hi - z) ** 2),
        rel=2e-8,
        abs=2e-9,
    )
    assert cross == pytest.approx(
        _direct_expectation(state, lambda z: np.log(z - lo) * np.log(hi - z)),
        rel=2e-8,
        abs=2e-9,
    )


def test_generalized_moments_reject_unavailable_boundary_basis():
    state = _state((-np.inf, np.inf), [-2.0, 0.0, 1.0, 3.0], 2, [0.0, 1.0])
    with pytest.raises(ValueError, match="not enabled"):
        state.moments.log_power("lower", 0)
    with pytest.raises(ValueError, match="both boundary"):
        state.moments.log_cross()


def test_half_line_mode_solver_returns_exact_constrained_endpoint():
    """Monotone convex potentials have their mode at the finite endpoint."""

    common = (
        LOG_THRESH,
        GRAD_TOL,
        HESS_TOL,
        NEWT_TOL,
        NEWT_MAX,
        BOUNDARY_EPS_MULT,
        BRACKET_INIT_STEP,
        BRACKET_MAX_EXPAND,
        BRACKET_STEP_GROWTH,
        BACKTRACK_MAX_ITERS,
        BACKTRACK_REDUCE,
    )
    _, lower_mode, _, _ = _q_window_and_mode(
        np.array([0.0, np.inf]),
        np.array([0.0, 1.0]),
        np.array([np.nan, np.nan]),
        np.array([0.2, 2.0]),
        *common,
    )
    _, upper_mode, _, _ = _q_window_and_mode(
        np.array([-np.inf, 0.0]),
        np.array([0.0, -1.0]),
        np.array([np.nan, np.nan]),
        np.array([-2.0, -0.2]),
        *common,
    )
    assert lower_mode == 0.0
    assert upper_mode == 0.0


def test_half_line_log_moments_are_warning_free_at_boundary_mode():
    """Endpoint log moments should be resolved without QUADPACK warnings."""

    state = _state((0.0, np.inf), [0.2, 0.5, 1.0, 2.0], 2, [3.0, 1.0, 0.0], lower=True)
    assert state.mode == state.spec.support[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        got = state.moments.log_power("lower", 2)
    assert np.all(np.isfinite(got))


def test_log_square_with_enabled_zero_amplitude_matches_python_oracle():
    state = _state((0.0, np.inf), [0.25, 0.7, 1.5, 3.0], 2, [0.4, 0.8, 0.0], lower=True)
    lo = float(state.spec.support[0])
    got = state.moments.log_square("lower")
    expected = _direct_expectation(state, lambda z: np.log(z - lo) ** 2)
    assert got == pytest.approx(expected, rel=3e-8, abs=3e-9)


def test_fused_power_moments_match_scalar_quadrature_across_supports():
    """Shared adaptive power traversal agrees with the scalar oracle path."""
    from gibbus._model._moment_kernels import power_moments

    from gibbus._fit.natural_objective import _fit_natural_conic_points

    rng = np.random.default_rng(918311)
    cases = (
        ((-np.inf, np.inf), rng.normal(size=700), 6, False, False),
        ((0.0, np.inf), rng.gamma(2.2, 0.8, size=700), 6, True, False),
        ((0.0, 1.0), rng.beta(2.4, 3.1, size=700), 6, True, True),
    )
    for support, data, degree, lower_log, upper_log in cases:
        objective, result = _fit_natural_conic_points(
            support, data, degree, lower_log, upper_log
        )
        state = objective.build_state(result.params)
        raw = (
            power_moments(
                np.asarray(state.quad_poly, dtype=np.float64),
                np.asarray(state.spec.support, dtype=np.float64),
                np.asarray(state.boundary_amplitudes, dtype=np.float64),
                np.asarray(state.window, dtype=np.float64),
                np.asarray(state.quad_points, dtype=np.float64),
                12,
                epsabs=1.49e-8,
                epsrel=1.49e-8,
                limit=100,
            )
            / state.Z
        )
        scalar = np.array(
            [state.moments._integral(0, k, 0) / state.Z for k in range(13)],
            dtype=np.float64,
        )
        scalar[0] = 1.0
        raw[0] = 1.0
        np.testing.assert_allclose(raw, scalar, rtol=3e-9, atol=3e-10)
