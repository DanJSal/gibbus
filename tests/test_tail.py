"""Focused regression tests for extreme-tail quantile inversion."""

import warnings

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import IntegrationWarning

from gibbus._defaults import TAIL_ASYMPTOTIC_P
from gibbus._spectral.tail import (
    _bracket,
    exact_tail_log_cdf,
    invert_tail,
    needs_asymptotic_tail,
)

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _normal_potential(mu=0.0, sigma=1.0):
    log_sigma = np.log(sigma)

    def potential(x, n):
        z = (x - mu) / sigma
        if n == 0:
            return 0.5 * z * z + _LOG_SQRT_2PI + log_sigma
        if n == 1:
            return z / sigma
        if n == 2:
            return 1.0 / (sigma * sigma)
        return 0.0

    return potential


def _beta_potential(a, b):
    dist = stats.beta(a, b)

    def potential(x, n):
        if n == 0:
            return -dist.logpdf(x)
        if n == 1:
            return -(a - 1.0) / x + (b - 1.0) / (1.0 - x)
        raise AssertionError("test potential only supplies q and q'")

    return potential


def _gamma_potential(shape, scale=1.0, *, reflected=False):
    dist = stats.gamma(shape, scale=scale)

    def potential(x, n):
        y = -x if reflected else x
        if n == 0:
            return -dist.logpdf(y)
        if n == 1:
            dq_dy = 1.0 / scale - (shape - 1.0) / y
            return -dq_dy if reflected else dq_dy
        raise AssertionError("test potential only supplies q and q'")

    return potential


def test_tail_handover_is_inclusive_at_configured_probability():
    p = np.array([
        TAIL_ASYMPTOTIC_P,
        np.nextafter(TAIL_ASYMPTOTIC_P, 1.0),
        1.0 - TAIL_ASYMPTOTIC_P,
        np.nextafter(1.0 - TAIL_ASYMPTOTIC_P, 0.0),
    ])
    lower, upper = needs_asymptotic_tail(p)
    assert np.array_equal(lower, [True, False, False, False])
    assert np.array_equal(upper, [False, False, True, False])


def test_bidirectional_tail_bracket_can_move_inward_from_seed():
    # f(v) is a monotonically decreasing log-tail mass.  At v=0 the seed
    # has *less* mass than requested, so a correct bracket must search to v<0.
    lo, hi = _bracket(-1.0, lambda v: -2.0 - v)
    assert lo < 0.0 <= hi
    assert -2.0 - lo >= -1.0 >= -2.0 - hi


@pytest.mark.parametrize("p", [1e-10, 1e-12, 1e-20, 1e-50, 1e-300])
def test_exact_tail_correction_matches_standard_normal(p):
    potential = _normal_potential()
    start = float(stats.norm.ppf(TAIL_ASYMPTOTIC_P))
    got = invert_tail(potential, np.log(p), -np.inf, start)
    expected = float(stats.norm.ppf(p))

    # The exact correction should reduce inversion error to floating-point
    # resolution rather than retain the percent-level Mills approximation error.
    assert abs(got - expected) <= 32.0 * abs(np.spacing(expected))
    log_mass = exact_tail_log_cdf(potential, got, -np.inf)
    assert log_mass == pytest.approx(np.log(p), abs=2e-12)


def test_infinite_tail_root_tolerance_is_translation_independent():
    mu = 1e12
    potential = _normal_potential(mu=mu)
    p = 1e-50
    start = mu + float(stats.norm.ppf(TAIL_ASYMPTOTIC_P))

    with warnings.catch_warnings():
        warnings.simplefilter("error", IntegrationWarning)
        got = invert_tail(potential, np.log(p), -np.inf, start)

    expected = mu + float(stats.norm.ppf(p))
    assert abs(got - expected) <= abs(np.spacing(expected))


@pytest.mark.parametrize("upper", [False, True])
def test_finite_endpoint_algebraic_tail_uses_exact_scaled_quadrature(upper):
    # A gamma shape below one has an integrable algebraic density singularity
    # at zero.  Reflect it for the upper-tail case so even p=1e-50 remains
    # representable as a distinct float from the endpoint.
    shape = 0.2
    p = 1e-50
    dist = stats.gamma(shape)
    if upper:
        endpoint = 0.0
        expected = -float(dist.ppf(p))
        start = -float(dist.ppf(TAIL_ASYMPTOTIC_P))
        potential = _gamma_potential(shape, reflected=True)
    else:
        endpoint = 0.0
        expected = float(dist.ppf(p))
        start = float(dist.ppf(TAIL_ASYMPTOTIC_P))
        potential = _gamma_potential(shape)

    got = invert_tail(potential, np.log(p), endpoint, start, upper=upper)
    assert got == pytest.approx(expected, rel=3e-12, abs=0.0)
