"""Canonical degree policies and request-owned selection configuration."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._api import component, fitting, selection
from gibbus._fit import degree, mixture_degree, natural_mixture, natural_objective
from gibbus._fit.degree import _DegreeSelectionConfig
from gibbus._fit.inputs import _normalize_degree_policies


@pytest.mark.parametrize(
    ("function", "keyword"),
    [
        (component._Component._fit, "degree_config"),
        (natural_mixture._fit_natural_mixture, "degree_config"),
        (natural_mixture._run_natural_em, "degree_config"),
        (natural_mixture._continue_natural_mixture, "degree_config"),
        (natural_mixture._policy_initial_components, "degree_config"),
        (natural_objective._fit_natural_conic_points_auto, "degree_config"),
        (natural_objective._fit_natural_conic_intervals_auto, "degree_config"),
        (degree._omitted_statistic_diagnostic, "config"),
        (degree._interval_omitted_statistic_diagnostic, "config"),
        (mixture_degree._joint_omitted_statistic_diagnostic, "config"),
        (mixture_degree._fit_shared_degree_growth, "degree_config"),
        (selection.select_n_components, "degree_config"),
    ],
)
def test_deep_fit_selection_config_is_required_keyword(function, keyword):
    parameter = inspect.signature(function).parameters[keyword]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_deep_mixture_does_not_own_raw_policy_normalization():
    assert not hasattr(natural_mixture, "_degree_policies")


@pytest.mark.parametrize(
    ("raw", "count", "expected"),
    [
        (4, 2, (4, 4)),
        (np.int64(6), 1, (6,)),
        (4.0, 2, (4, 4)),
        (None, 2, ("auto", "auto")),
        ("AuTo", 2, ("auto", "auto")),
        ([4, None, "AUTO"], 3, (4, "auto", "auto")),
        (np.array([2, 6]), 2, (2, 6)),
    ],
)
def test_degree_policy_normalization(raw, count, expected):
    actual = _normalize_degree_policies(raw, count)
    assert actual == expected
    assert isinstance(actual, tuple)
    assert all(type(value) in (int, str) for value in actual)


@pytest.mark.parametrize(
    "raw",
    [True, np.bool_(False), 1, 3.5, np.nan, np.inf, "4", "invalid", [4], [4, 6, 8]],
)
def test_degree_policy_rejects_coercion_and_component_count_mismatches(raw):
    with pytest.raises(ValueError):
        _normalize_degree_policies(raw, 2)


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
    original = selection.select_n_components

    def select(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(fitting, "select_n_components", select)
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
    assert isinstance(calls[0]["degree_config"], _DegreeSelectionConfig)


@pytest.mark.parametrize("intervals", [False, True])
def test_single_boundary_refits_share_explicit_degree_config(monkeypatch, intervals):
    config = _DegreeSelectionConfig(alpha=0.012, min_participation=13.0)
    calls = []
    rows = np.linspace(0.1, 0.9, 20)[:, None]
    if intervals:
        rows = np.column_stack((rows[:, 0], rows[:, 0] + 0.01))
    model = (
        SimpleNamespace(spec=SimpleNamespace(requested_poly_degree=4)),
        object(),
    )

    def auto(*args, degree_config):
        assert degree_config is config
        calls.append(args[2:4])
        return model

    def fixed(norm, degree, lower, upper):
        assert degree == 4
        return model

    def boundary(
        fit, value, amplitude, lower, upper, effective_n, *, fit_reduced, refit, alpha
    ):
        initial = fit(True, False)
        reduced = fit_reduced(initial, False, False)
        assert reduced is model
        return refit(False, False), None, (0.5, np.nan)

    monkeypatch.setattr(component, "_fit_natural_conic_points_auto", auto)
    monkeypatch.setattr(component, "_fit_natural_conic_intervals_auto", auto)
    monkeypatch.setattr(component, "_fit_natural_fixed_degree", fixed)
    monkeypatch.setattr(component, "_select_boundary_terms", boundary)
    norm = {
        "poly_degree": "auto",
        "support": (0.0, 1.0),
        "log_boundary_lower": "auto",
        "log_boundary_upper": False,
        "weights": None,
        "samples_rk": rows,
    }
    objective, result, _, _ = component._run_natural_fit(norm, config)
    assert (objective, result) == model
    assert calls == [(True, False), (False, False)]


def test_component_fit_reuses_explicit_degree_config(monkeypatch):
    config = _DegreeSelectionConfig()
    supplied = []

    def unexpected_config():
        raise AssertionError("component fit must not reconstruct request policy")

    def run(norm, degree_config):
        supplied.append(degree_config)
        return object(), object(), 20.0, (np.nan, np.nan)

    monkeypatch.setattr(component, "_DegreeSelectionConfig", unexpected_config)
    monkeypatch.setattr(component, "_run_natural_fit", run)
    monkeypatch.setattr(component, "_pack_natural_fit", lambda *args, **kwargs: None)
    fitted = component._Component._fit(
        np.linspace(-1.0, 1.0, 20),
        poly_degree=None,
        support=None,
        log_boundary_lower=None,
        log_boundary_upper=None,
        verbose=0,
        suppress_warnings=False,
        init_from=None,
        sample_weight=None,
        degree_config=config,
    )
    assert isinstance(fitted, component._Component)
    assert len(supplied) == 1
    assert supplied[0] is config


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
    original_select = fitting.select_n_components
    original_single = component._run_natural_fit
    original_refit = fitting._run_single_fit

    def prepare(request):
        requests.append(request)
        context = original_context(request)
        assert context.degree_config is request.degree_config
        return context

    def select(**kwargs):
        selected_configs.append(kwargs["degree_config"])
        return original_select(**kwargs)

    def single(norm, config):
        single_configs.append(config)
        return original_single(norm, config)

    def refit(request, **kwargs):
        refit_configs.append(request.degree_config)
        return original_refit(request, **kwargs)

    monkeypatch.setattr(fitting, "_prepare_mixture_context", prepare)
    monkeypatch.setattr(fitting, "select_n_components", select)
    monkeypatch.setattr(fitting, "_run_natural_fit", single)
    monkeypatch.setattr(component, "_run_natural_fit", single)
    monkeypatch.setattr(fitting, "_run_single_fit", refit)
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
    assert len(refit_configs) == int(subsample is not False)


@pytest.mark.parametrize("policy", [(2, 2), ("auto", "auto"), (2, "auto")])
def test_mixture_search_and_continuation_share_degree_config(monkeypatch, policy):
    config = _DegreeSelectionConfig(alpha=0.02, min_participation=12.0)
    seen = set()

    def wrap(module, name, keyword):
        original = getattr(module, name)

        def checked(*args, **kwargs):
            supplied = kwargs.get(keyword)
            if keyword == "config" and supplied is None:
                supplied = args[-1]
            assert supplied is config
            seen.add(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(module, name, checked)

    for name in (
        "_run_natural_em",
        "_continue_natural_mixture",
        "_policy_initial_components",
    ):
        wrap(natural_mixture, name, "degree_config")
    wrap(mixture_degree, "_fit_shared_degree_growth", "degree_config")
    wrap(mixture_degree, "_shared_degree_diagnostics", "config")
    rng = np.random.default_rng(103)
    x = np.r_[rng.normal(-3, 0.5, 60), rng.normal(3, 0.5, 60)]
    responsibilities = np.column_stack((x < 0, x >= 0)).astype(float)
    initial = natural_mixture._run_natural_em(
        (-np.inf, np.inf),
        x[:, None],
        policy,
        False,
        False,
        np.full(len(x), 1.0 / len(x)),
        responsibilities,
        degree_config=config,
        em_options=natural_mixture._EMOptions(max_steps=2, max_rounds=1),
    )
    fit = natural_mixture._fit_natural_mixture(
        (-np.inf, np.inf),
        x[:, None],
        2,
        policy,
        degree_config=config,
        initial_fit=initial,
        responsibilities=responsibilities,
        search_options=natural_mixture._MixtureSearchOptions(
            paths=(("direct", "raw"),), finalists=1
        ),
        em_options=natural_mixture._EMOptions(max_steps=2, max_rounds=1),
    )
    assert fit.separator_certified
    assert {
        "_run_natural_em",
        "_continue_natural_mixture",
    } <= seen
    if "auto" in policy:
        assert {
            "_policy_initial_components",
            "_fit_shared_degree_growth",
            "_shared_degree_diagnostics",
        } <= seen


@pytest.mark.parametrize("intervals", [False, True])
def test_single_auto_selector_passes_config_to_diagnostic(monkeypatch, intervals):
    config = _DegreeSelectionConfig(alpha=0.015)
    seen = []
    name = (
        "_interval_omitted_statistic_diagnostic"
        if intervals
        else "_omitted_statistic_diagnostic"
    )
    original = getattr(natural_objective, name)

    def diagnostic(*args, config):
        seen.append(config)
        return original(*args, config=config)

    monkeypatch.setattr(natural_objective, name, diagnostic)
    x = np.random.default_rng(104).normal(size=120)
    if intervals:
        natural_objective._fit_natural_conic_intervals_auto(
            (-np.inf, np.inf),
            np.column_stack((x - 0.01, x + 0.01)),
            degree_config=config,
        )
    else:
        natural_objective._fit_natural_conic_points_auto(
            (-np.inf, np.inf), x, degree_config=config
        )
    assert seen
    assert all(item is config for item in seen)


@pytest.mark.parametrize("schedule", ["direct", "ladder"])
def test_uniform_fixed_search_passes_canonical_tuples_through_degree_rungs(
    monkeypatch, schedule
):
    config = _DegreeSelectionConfig()
    degrees = []
    original = natural_mixture._run_natural_em

    def run(*args, **kwargs):
        policy = args[2]
        assert isinstance(policy, tuple)
        assert len(policy) == 2
        assert all(type(value) is int for value in policy)
        assert policy[0] == policy[1]
        assert kwargs["degree_config"] is config
        degrees.append(policy)
        return original(*args, **kwargs)

    monkeypatch.setattr(natural_mixture, "_run_natural_em", run)
    rng = np.random.default_rng(106)
    x = np.r_[rng.normal(-3, 0.5, 60), rng.normal(3, 0.5, 60)]
    responsibilities = np.column_stack((x < 0, x >= 0)).astype(float)
    fitted = natural_mixture._fit_natural_mixture(
        (-np.inf, np.inf),
        x[:, None],
        2,
        (6, 6),
        degree_config=config,
        responsibilities=responsibilities,
        search_options=natural_mixture._MixtureSearchOptions(
            paths=((schedule, "raw"),), finalists=1
        ),
        em_options=natural_mixture._EMOptions(max_steps=2, max_rounds=1),
    )
    assert fitted.separator_certified
    assert set(degrees) == (
        {(6, 6)} if schedule == "direct" else {(2, 2), (4, 4), (6, 6)}
    )
