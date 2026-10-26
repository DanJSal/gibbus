"""Extreme-coordinate accuracy: large offsets and supports a few hundred floats wide.

An endpoint-log-only bounded potential has a closed-form beta-family
normalizer, log-basis expectation, and half-support probability.  These
independent quantities detect physical-coordinate rounding at offsets up to
``2**49`` widths and on supports only a few hundred representable floats wide.
"""

import math

import numpy as np
import pytest

from gibbus import BasisSpec
from gibbus._model.basis import _build_basis
from gibbus._model.basis_normalizer import (
    _basis_state,
    _interval_state,
    _NormalizerContext,
)


@pytest.mark.parametrize("ratio", [2**20, 2**24 - 1, 2**24 + 1, 2**32, 2**44, 2**49])
@pytest.mark.parametrize(
    "side,gamma", [("lower", 0.5), ("upper", 0.5), ("lower", 3.0), ("upper", 3.0)]
)
def test_endpoint_log_normalizer_and_interval_mass_across_coordinate_switch(
    ratio, side, gamma
):
    center = (1.0 if side == "lower" else -1.0) * 1e6
    scale = abs(center) / ratio
    lower = center - 8.0 * scale
    upper = center + 8.0 * scale
    width = upper - lower
    assert width >= 200 * np.spacing(abs(center))
    data = center + scale * np.linspace(-4.0, 4.0, 21)
    model = _build_basis(
        BasisSpec(
            support=(lower, upper),
            bulk_degree=2,
            log_lower=side == "lower",
            log_upper=side == "upper",
        ),
        data,
    )
    j = next(
        i
        for i, column in enumerate(model.columns)
        if column.label == ("logL" if side == "lower" else "logU")
    )
    theta = np.zeros(model.n_coefficients)
    theta[j] = gamma
    context = _NormalizerContext(model)
    state = _basis_state(context, theta)
    assert state.status == "proper"
    expected_log_z = (gamma + 1.0) * math.log(width) - math.log(gamma + 1.0)
    expected_log_mean = -math.log(width) + 1.0 / (gamma + 1.0)
    # Node positions are known to |x| eps, which the endpoint log turns into a
    # relative error |x| eps / (x - end): the floor the normalizer reports as
    # ``cond_error``.  Results must sit well inside it.
    atol = max(4e-12, 0.05 * state.cond_error)
    assert state.log_z == pytest.approx(expected_log_z, abs=atol)
    assert state.mean[j] == pytest.approx(expected_log_mean, abs=atol)
    row = _interval_state(
        context,
        theta,
        np.array([lower, center], dtype=np.float64),
        np.array([center, upper], dtype=np.float64),
        np.array([0.5, 0.5]),
        per_row=True,
    )
    p_left = ((center - lower) / width) ** (gamma + 1.0)
    if side == "upper":
        p_left = 1.0 - ((upper - center) / width) ** (gamma + 1.0)
    assert math.exp(row.row_log_p[0]) == pytest.approx(p_left, abs=atol)
    assert math.exp(row.row_log_p[1]) == pytest.approx(1.0 - p_left, abs=atol)
    assert row.row_log_p[0] <= 0.0 and row.row_log_p[1] <= 0.0


def test_large_offset_gaussian_postfit_moments_use_fitting_density():
    """The four canonical moments must survive a c/h ratio above 2**24."""
    from gibbus._postfit.basis_state import _BasisPotential, _canonical_moments

    x = 1e6 + 1e-9 * np.random.default_rng(48).standard_normal(600)
    model = _build_basis(BasisSpec(bulk_degree=2), x)
    j = next(i for i, col in enumerate(model.columns) if col.label == "z^2")
    theta = np.zeros(model.n_coefficients)
    theta[j] = 0.5
    state = _basis_state(_NormalizerContext(model), theta)
    pot = _BasisPotential(model, theta, state.log_z)
    moments = _canonical_moments(pot, 0.0)
    np.testing.assert_allclose(moments, [1.0, 0.0, 1.0, 0.0, 3.0], rtol=0, atol=1e-9)


@pytest.mark.parametrize("offset,spread", [(1e6, 1e-6), (-1e6, 1e-7), (1.0, 1e-10)])
def test_extreme_offset_two_component_fit_builds_valid_postfit_cdf(offset, spread):
    """A converged mixture must not fail while integrating its post-fit moments."""
    from gibbus import Distribution

    rng = np.random.default_rng(531)
    x = np.concatenate(
        [
            offset - 2.0 * spread + spread * rng.normal(size=90),
            offset + 2.0 * spread + spread * rng.normal(size=90),
        ]
    )
    fitted = Distribution().fit(
        x,
        n_components=2,
        poly_degree=2,
        support=(-np.inf, np.inf),
        log_boundary_lower=False,
        log_boundary_upper=False,
        rng=0,
    )
    grid = np.linspace(float(x.min()), float(x.max()), 31)
    cdf = np.asarray(fitted.cdf(grid))
    assert fitted.fit_diagnostics["converged"]
    assert np.all(np.isfinite(cdf))
    assert 0.0 < cdf[0] < cdf[-1] < 1.0
    assert np.all(np.diff(cdf) >= 0.0)
