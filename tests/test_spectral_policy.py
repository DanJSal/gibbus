"""Explicit construction policy and one compiled spectral evaluation path."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from gibbus._spectral.cdf import SpectralCDF, _builder_tables
from gibbus._spectral.ppf import SpectralPPF


@pytest.mark.parametrize(
    ("function", "names"),
    [
        (
            SpectralCDF,
            ("config", "mode", "std", "map_scale", "initial_breaks"),
        ),
        (SpectralPPF, ("config",)),
        (_builder_tables, ("ppf_degrees",)),
    ],
)
def test_spectral_policy_is_explicit(function, names):
    signature = inspect.signature(function)
    for name in names:
        assert signature.parameters[name].default is inspect.Parameter.empty


def test_cdf_representation_delegates_both_coordinate_spaces():
    calls = []

    class Evaluator:
        def __call__(self, value):
            calls.append(("physical", value))
            return value

        def eval_compact(self, value):
            calls.append(("compact", value))
            return value

    representation = object.__new__(SpectralCDF)
    representation._cython_evaluator = Evaluator()
    values = np.array([[0.0, np.nan], [-1.0, 1.0]])
    assert representation.cdf_z(values) is values
    assert representation.cdf(values) is values
    assert [kind for kind, _ in calls] == ["compact", "physical"]


def test_ppf_representation_delegates_both_coordinate_spaces():
    calls = []

    def compact(value):
        calls.append("compact")
        return value

    def physical(value):
        calls.append("physical")
        return value

    representation = object.__new__(SpectralPPF)
    representation._cython_evaluator = SimpleNamespace(
        eval_compact=compact, eval_x=physical
    )
    values = np.array([[0.0, 1.0], [0.5, np.nan]])
    assert representation.ppf_z(values) is values
    assert representation.ppf(values) is values
    assert calls == ["compact", "physical"]
