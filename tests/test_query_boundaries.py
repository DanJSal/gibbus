"""Canonical query boundaries and strict internal evaluator contracts."""

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._api import views
from gibbus._api.validation import (
    _validate_level,
    _validate_log_probabilities,
    _validate_probabilities,
)
from gibbus._postfit.evaluators import _potential_oriented_affine_eval
from gibbus._postfit.regions import hpd


@pytest.mark.parametrize(
    ("prepare", "values", "name"),
    [
        (_validate_probabilities, [0.0, 0.5, 1.0, np.nan], "isf"),
        (_validate_log_probabilities, [-np.inf, -1.0, 0.0, np.nan], "logppf"),
    ],
)
def test_query_probability_boundary_preserves_nan_and_endpoints(prepare, values, name):
    actual = prepare(values, name)
    assert actual.dtype == np.float64
    np.testing.assert_equal(actual, values)
    assert prepare(actual, name) is actual


@pytest.mark.parametrize("level", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_region_level_is_rejected_at_api_boundary(level):
    with pytest.raises(ValueError, match="level must"):
        _validate_level(level)


@pytest.fixture
def fitted_query_model():
    return Distribution().fit(np.linspace(-1.0, 1.0, 80), poly_degree=2)


@pytest.mark.parametrize(
    ("method", "helper", "valid", "invalid"),
    [
        ("isf", "_isf", [0.2, np.nan], [-0.1]),
        ("logppf", "_logppf", [-1.0, np.nan], [0.1]),
        ("logisf", "_logisf", [-1.0, np.nan], [0.1]),
    ],
)
def test_public_survival_prepares_inputs_before_computation(
    monkeypatch, fitted_query_model, method, helper, valid, invalid
):
    seen = []

    def compute(potential, ppf, support, probabilities, *, log_tail_mass):
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.dtype == np.float64
        seen.append(probabilities)
        return np.zeros_like(probabilities)

    monkeypatch.setattr(views, helper, compute)
    query = getattr(fitted_query_model.base, method)
    query(valid)
    assert len(seen) == 1
    with pytest.raises(ValueError):
        query(invalid)
    assert len(seen) == 1


def test_survival_body_does_not_reenter_public_ppf(monkeypatch, fitted_query_model):
    def unexpected(*args):
        raise AssertionError("derived probabilities must use the internal PPF")

    monkeypatch.setattr(views._BaseSpaceView, "ppf", unexpected)
    assert np.isfinite(fitted_query_model.base.isf(0.2))
    assert np.isfinite(fitted_query_model._base_ppf_for_extensions(0.2))


def test_flat_hpd_fallback_uses_prepared_level():
    result = hpd(
        lambda x: np.zeros_like(x),
        lambda x: np.log(x),
        lambda x: np.log1p(-x),
        lambda p: p,
        lambda p: 1.0 - p,
        (0.0, 1.0),
        _validate_level(0.8),
        modes=(),
    )
    np.testing.assert_allclose(result, [[0.1, 0.9]])


@pytest.mark.parametrize("order", [0, 1, 2])
def test_oriented_potential_consumes_canonical_geometry(order):
    x = np.array([0.0, 1.0, np.nan, np.inf])
    actual = _potential_oriented_affine_eval(
        x,
        np.array([-np.inf, np.inf]),
        0.0,
        1.0,
        np.array([0.0, 0.0, 0.5]),
        np.array([np.nan, np.nan]),
        order,
    )
    expected = {
        0: [0.0, 0.5, np.inf, np.inf],
        1: [0.0, 1.0, np.nan, np.nan],
        2: [1.0, 1.0, np.nan, np.nan],
    }[order]
    np.testing.assert_equal(actual, expected)
