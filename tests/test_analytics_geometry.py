"""Prepared scalar geometry and canonical separation regressions."""

import numpy as np
import pytest

from gibbus._fit.separation import (
    _curvature_polynomials,
    _exact_curvature_polynomials,
)
from gibbus._postfit.analytics import _ScalarPotential


def test_scalar_potential_prepares_derivatives_only_once(monkeypatch):
    coefficients = np.array([0.0, 0.0, 0.5])
    original = np.polynomial.polynomial.polyder
    calls = []

    def polyder(values, order=1):
        calls.append(order)
        return original(values, order)

    monkeypatch.setattr(np.polynomial.polynomial, "polyder", polyder)
    potential = _ScalarPotential(
        (-np.inf, np.inf), coefficients, np.array([np.nan, np.nan])
    )
    for point in np.linspace(-2.0, 2.0, 20):
        assert potential.value(point) == pytest.approx(0.5 * point**2)
        assert potential.gradient(point) == pytest.approx(point)
        assert potential.curvature(point) == pytest.approx(1.0)
    assert calls == [1, 2]


@pytest.mark.parametrize(("point", "gradient"), [(-1.0, -np.inf), (1.0, np.inf)])
def test_prepared_potential_retains_singular_endpoint_guards(point, gradient):
    potential = _ScalarPotential(
        (-1.0, 1.0), np.array([0.0, 0.0, 0.5]), np.array([2.0, 3.0])
    )
    assert potential.value(point) == np.inf
    assert potential.gradient(point) == gradient
    assert potential.curvature(point) == np.inf


def test_separation_reuses_canonical_geometry_without_recoercion(monkeypatch):
    support = np.array([0.0, 1.0])
    amplitudes = np.array([2.0, 3.0])
    original = np.asarray

    def asarray(value, *args, **kwargs):
        if value is support or value is amplitudes:
            raise AssertionError("canonical separation geometry must not be recoerced")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", asarray)
    ordinary = _curvature_polynomials(np.array([1.0]), support, amplitudes)
    exact = _exact_curvature_polynomials(np.array([1.0]), support, amplitudes)
    assert ordinary.lower_active and ordinary.upper_active
    assert exact[2:] == (True, True)
