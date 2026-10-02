"""Adversarial definition checks for post-fit numerical helpers."""

import math

import numpy as np
from gibbus._model._state_kernels import state_numerics
from scipy.integrate import quad

from gibbus._defaults import QUAD_EPSABS, QUAD_EPSREL, QUAD_LIMIT
from gibbus._model.natural_state import _MODE_CONTROLS
from gibbus._postfit.expectation import expect


def test_state_normalizer_resolves_density_window_inside_huge_data_padding():
    """Far-away observations must not make model quadrature miss a narrow tail edge."""
    # Convex degree-6 potential taken from a real stress fit.  Its density is
    # concentrated on O(1) canonical scales, while a mixture component's tiny
    # responsibilities on far-away observations can pad the working window to
    # O(1e6).  Without the density-defined tail breakpoints, the outer
    # Gauss--Kronrod panel samples only essentially-zero density and misses
    # about 8e-4 of the normalizer.
    q_poly = np.array(
        [
            0.0,
            7.17006415e-02,
            2.48820199e-01,
            -8.34627248e-02,
            1.45135296e-02,
            -1.21207371e-03,
            3.88709195e-05,
        ]
    )
    status, geometry, _points, shifted_z, *_ = state_numerics(
        np.array([-np.inf, np.inf]),
        q_poly,
        np.array([np.nan, np.nan]),
        np.array([-391449.15, 559804.41]),
        False,
        False,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty((0, 1), dtype=np.float64),
        _MODE_CONTROLS,
        QUAD_EPSABS,
        QUAD_EPSREL,
        QUAD_LIMIT,
    )
    assert status == 0
    got = -float(geometry[3]) + math.log(float(shifted_z))

    def density(z):
        return math.exp(-float(np.polynomial.polynomial.polyval(z, q_poly)))

    # q(+-30) is already thousands of nats above its minimum, so this finite
    # interval is an effectively exact independent normalizer reference.
    reference, error = quad(
        density,
        -30.0,
        30.0,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=2000,
        points=[-20.0, -10.0, float(geometry[2]), 10.0, 20.0],
    )
    assert error < 1e-11
    assert np.isclose(got, math.log(reference), rtol=0.0, atol=2e-12)


def test_scalar_expect_skips_user_function_where_density_is_zero():
    """Tail probes with zero fitted density must not overflow transformed functions."""
    log_norm = 0.5 * math.log(2.0 * math.pi)

    def standard_normal_potential(x, n):
        if n != 0:
            raise AssertionError("only the density potential is needed")
        return 0.5 * np.asarray(x) ** 2 + log_norm

    # Infinite-range QUADPACK probes |x| far beyond the range where exp(x) is
    # representable.  Those probes have exactly-zero float64 Gaussian density
    # and therefore contribute nothing.  The expectation is E[e^X] = e^(1/2).
    got = expect(
        standard_normal_potential,
        (-np.inf, np.inf),
        lambda x: np.exp(x),
        points=(0.0,),
    )
    assert np.isclose(got, math.exp(0.5), rtol=0.0, atol=2e-10)
