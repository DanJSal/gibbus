"""Tests for generic compiled kernels used by the native fitting stack."""
import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval

from gibbus import Distribution
from gibbus._model.vec import _polyval


class TestPolyvalDispatch:
    """``_vec._polyval`` routes arrays through the compiled Horner kernel."""

    @pytest.mark.parametrize("degree", [0, 1, 2, 6, 9])
    def test_array_matches_numpy(self, degree):
        rng = np.random.default_rng(20 + degree)
        x = rng.normal(size=1000)
        coef = rng.normal(size=degree + 1)
        assert _polyval(x, coef) == pytest.approx(polyval(x, coef), rel=1e-12)

    def test_scalar_matches_array_path(self):
        coef = np.array([0.5, -1.0, 2.0, 0.25])
        scalar = _polyval(np.float64(0.7), coef)
        arr = _polyval(np.array([0.7]), coef)
        assert float(scalar) == pytest.approx(float(arr[0]), rel=1e-14)

    def test_preserves_shape(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(4, 5))
        assert _polyval(x, np.array([1.0, 2.0, 3.0])).shape == (4, 5)

    def test_empty_coefficients(self):
        assert _polyval(np.array([1.0, 2.0]), np.array([])).shape == (2,)


def test_degree_two_fit_uses_compiled_evaluation_kernels():
    """A degree-two fit remains healthy through the compiled PDF/window path."""

    rng = np.random.default_rng(3)
    data = np.ascontiguousarray(rng.normal(size=300))
    fit = Distribution().fit(data, n_components=1, poly_degree=2,
                      support=(-np.inf, np.inf))
    assert fit.is_fitted
    assert np.isfinite(fit.mean)
    assert fit.var > 0.0
