"""Canonical degree policies and request-owned selection configuration."""

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._api import fitting
from gibbus._fit import (
    basis_mixture,
    basis_selection,
)
from gibbus._fit.degree import _DegreeSelectionConfig


@pytest.mark.parametrize(
    ("global_degree", "options", "expected"),
    [
        (4, [{}, {}], (4, 4)),
        (None, [{}, {}], ("auto", "auto")),
        ("AUTO", [{}, {"poly_degree": 6}], ("auto", 6)),
        (6, [{"poly_degree": "auto"}, {"poly_degree": 4}], ("auto", 4)),
    ],
)
def test_natural_degree_policy_preserves_tuple_and_component_precedence(
    global_degree, options, expected
):
    assert fitting._natural_degree_policy(global_degree, options) == expected


def test_bimodal_default_fit_does_not_select_components(monkeypatch):
    rng = np.random.default_rng(102)
    data = np.r_[rng.normal(-4, 0.4, 100), rng.normal(4, 0.4, 100)]
    assert np.max(data[:100]) < np.min(data[100:])
    calls = []
    original = basis_mixture._select_components

    def select(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(basis_mixture, "_select_components", select)
    monkeypatch.setattr(fitting, "_select_components", select)
    default = Distribution().fit(data, support=(-np.inf, np.inf), poly_degree=2)
    assert default.n_components == 1
    assert default.selection_diagnostics is None
    assert not calls
    automatic = Distribution().fit(
        data,
        n_components="auto",
        support=(-np.inf, np.inf),
        poly_degree=2,
        k_max=2,
        auto_k_subsample=False,
        rng=0,
    )
    assert len(calls) == 1
    assert automatic.n_components == 2
    assert automatic.selection_diagnostics["selected_n_components"] == 2
    assert isinstance(calls[0]["config"], _DegreeSelectionConfig)


@pytest.mark.parametrize("subsample", [False, 120])
def test_selected_single_component_reuses_request_degree_config(monkeypatch, subsample):
    assert (
        fitting._FitRequest.__dataclass_fields__["degree_config"].default_factory
        is _DegreeSelectionConfig
    )
    requests = []
    selected_configs = []
    single_configs = []
    refit_configs = []
    original_context = fitting._prepare_mixture_context
    original_select = basis_mixture._select_components
    original_single = basis_selection._fit_basis
    original_refit = basis_mixture._fit_mixture_auto

    def prepare(request):
        requests.append(request)
        context = original_context(request)
        assert context.degree_config is request.degree_config
        return context

    def select(*args, **kwargs):
        selected_configs.append(kwargs["config"])
        return original_select(*args, **kwargs)

    def single(*args, **kwargs):
        single_configs.append(kwargs["config"])
        return original_single(*args, **kwargs)

    def refit(*args, **kwargs):
        refit_configs.append(kwargs["config"])
        return original_refit(*args, **kwargs)

    monkeypatch.setattr(fitting, "_prepare_mixture_context", prepare)
    monkeypatch.setattr(basis_mixture, "_select_components", select)
    monkeypatch.setattr(basis_selection, "_fit_basis", single)
    monkeypatch.setattr(basis_mixture, "_fit_basis", single)
    monkeypatch.setattr(basis_mixture, "_fit_mixture_auto", refit)
    monkeypatch.setattr(fitting, "_select_components", select)
    monkeypatch.setattr(fitting, "_fit_basis", single)
    monkeypatch.setattr(fitting, "_fit_mixture_auto", refit)
    x = np.random.default_rng(105).normal(size=400)
    fitted = Distribution().fit(
        x,
        n_components="auto",
        poly_degree=2,
        k_max=2,
        auto_k_subsample=subsample,
        support=(-np.inf, np.inf),
        rng=0,
    )
    assert fitted.n_components == 1
    assert fitted.selection_diagnostics["subsampled"] is (subsample is not False)
    assert len(requests) == 1
    config = requests[0].degree_config
    assert selected_configs and single_configs
    assert all(
        item is config for item in selected_configs + single_configs + refit_configs
    )
    # ``_fit_mixture_auto`` also refines each admitted count (G3), so only
    # the configuration identity is checked for it.
    assert refit_configs
