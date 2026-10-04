"""Boundary selection compares nested shared fits with real warm starts."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from gibbus._api import fitting
from gibbus._fit.degree import _DegreeSelectionConfig


def _context():
    return SimpleNamespace(
        samples_rk=np.linspace(0.1, 3.0, 20)[:, None],
        support=(0.0, np.inf),
        observation_weights=None,
        fit_generator=np.random.default_rng(0),
        em_max_iter=10,
        em_tol=1e-5,
        log_boundary_lower="auto",
        log_boundary_upper=False,
        degree_config=_DegreeSelectionConfig(alpha=0.02),
    )


def _result(lower, *, log_likelihood=0.0):
    return SimpleNamespace(
        components=tuple(
            SimpleNamespace(
                spec=SimpleNamespace(
                    requested_poly_degree=degree,
                    physical_lower_a_index=0 if lower else None,
                    physical_upper_a_index=None,
                ),
                params=np.array([0.1]) if lower else np.empty(0),
            )
            for degree in (4, 6)
        ),
        log_likelihood=log_likelihood,
        responsibilities=np.full((20, 2), 0.5),
    )


def test_boundary_removal_locks_degrees_and_preserves_shared_numerical_seed(
    monkeypatch,
):
    calls = []

    def fit(*args, **kwargs):
        result = _result(args[4], log_likelihood=0.0 if args[4] else -1e-8)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(fitting, "_fit_natural_mixture", fit)
    context = _context()
    fitted, p_values = fitting._fit_mixture_with_boundary_policy(
        context, 2, (6, 6), None, (("direct", "raw"),), initial_fit=None
    )
    assert len(calls) == 2
    assert calls[1][0][3] == (4, 6)
    assert calls[1][1]["initial_fit"] is calls[0][2]
    assert calls[1][0][4:6] == (False, False)
    assert fitted is calls[1][2]
    assert p_values[0] > 0.05
    assert all(
        kwargs["degree_config"] is context.degree_config for _, kwargs, _ in calls
    )


@pytest.mark.parametrize("policy", [("auto", "auto"), (4, "auto")])
def test_automatic_degrees_are_reselected_only_after_nested_boundary_comparison(
    monkeypatch,
    policy,
):
    calls = []

    def fit(*args, **kwargs):
        result = _result(args[4], log_likelihood=0.0 if args[4] else -1e-8)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(fitting, "_fit_natural_mixture", fit)
    context = _context()
    fitted, _ = fitting._fit_mixture_with_boundary_policy(
        context, 2, policy, None, (("direct", "raw"),), initial_fit=None
    )
    assert len(calls) == 3
    assert calls[1][0][3] == (4, 6)
    assert calls[0][0][3] is policy
    assert calls[2][0][3] is policy
    assert calls[2][0][4:6] == (False, False)
    assert calls[2][1]["initial_fit"] is calls[1][2]
    assert fitted is calls[2][2]
    assert all(
        kwargs["degree_config"] is context.degree_config for _, kwargs, _ in calls
    )


def test_explicit_boundary_policy_performs_no_nested_selection(monkeypatch):
    context = _context()
    context.log_boundary_lower = True
    initial = _result(True)
    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return initial

    monkeypatch.setattr(fitting, "_fit_natural_mixture", fit)
    fitted, p_values = fitting._fit_mixture_with_boundary_policy(
        context, 2, ("auto", "auto"), None, (("direct", "raw"),), initial_fit=initial
    )
    assert len(calls) == 1
    assert calls[0][1]["initial_fit"] is initial
    assert fitted is initial
    assert np.all(np.isnan(p_values))


def test_selection_receives_full_data_policy_and_hides_completed_payload(monkeypatch):
    context = _context()
    context.n_components = "auto"
    context.generator = np.random.default_rng(2)
    context.samples_1d = context.samples_rk[:, 0]
    context.n_columns = 1
    context.verbose = 0
    context.suppress_warnings = False
    context.component_options = None
    request = SimpleNamespace(
        seed_components=None,
        k_max=3,
        auto_k_subsample=10,
        poly_degree=6,
        degree_config=context.degree_config,
    )
    payload = {"fit": object()}
    received = {}

    def select(**kwargs):
        received.update(kwargs)
        return (
            1,
            None,
            np.ones(1),
            {"_selected_fit": payload, "reuse_selected_fit": False},
        )

    monkeypatch.setattr(fitting, "_propose_n_components", lambda *a, **k: (1, 3))
    monkeypatch.setattr(fitting, "select_n_components", select)
    initialized = fitting._initialize_mixture(request, context)
    assert received["policy"].lower_boundary == "auto"
    assert received["policy"].upper_boundary is False
    assert received["policy"].degree_policy == 6
    assert received["policy"].subsample_size == 10
    assert received["degree_config"] is context.degree_config
    assert callable(received["refine_candidate"])
    assert initialized.completed_fit is payload
    assert "_selected_fit" not in initialized.selection_diagnostics


def test_full_data_single_winner_is_installed_without_another_fit(monkeypatch):
    completed = SimpleNamespace(objective=object(), result=object())
    metadata = {"n_face_parameters": 3}
    component = object()
    packed = {}
    initialization = fitting._MixtureInitialization(
        n_components=1,
        responsibilities=None,
        weights=np.ones(1),
        component_options=[{}],
        candidates=None,
        selection_diagnostics={"reuse_selected_fit": True},
        completed_fit={
            "fit": completed,
            "effective_n": 20.0,
            "boundary_p_values": (0.01, np.nan),
        },
    )
    monkeypatch.setattr(fitting, "_prepare_mixture_context", lambda _: _context())
    monkeypatch.setattr(fitting, "_initialize_mixture", lambda *args: initialization)

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("a complete full-data winner must not be refitted")

    def pack(objective, result, **kwargs):
        assert objective is completed.objective
        assert result is completed.result
        packed.update(kwargs)
        return {}

    monkeypatch.setattr(fitting, "_run_single_fit", unexpected_fit)
    monkeypatch.setattr(fitting, "_pack_natural_fit", pack)
    monkeypatch.setattr(fitting, "_Component", lambda state: component)
    monkeypatch.setattr(fitting, "_single_fit_metadata", lambda state: metadata)
    result = fitting._run_mixture_fit(
        SimpleNamespace(degree_config=_DegreeSelectionConfig())
    )
    assert result.components == [component]
    assert result.fit_metadata is metadata
    assert result.selection_diagnostics == {"reuse_selected_fit": True}
    assert packed["effective_n"] == 20.0
    assert packed["boundary_p_values"][0] == 0.01


def test_subsample_winner_does_not_reuse_observation_dependent_continuation(
    monkeypatch,
):
    context = _context()
    full_responsibilities = np.full((20, 2), 0.5)
    subsample_fit = SimpleNamespace(responsibilities=np.full((4, 2), 0.5))
    initialization = fitting._MixtureInitialization(
        n_components=2,
        responsibilities=full_responsibilities,
        weights=np.full(2, 0.5),
        component_options=[{}, {}],
        candidates=None,
        selection_diagnostics={"reuse_selected_fit": False},
        completed_fit={"fit": subsample_fit},
    )
    fitted = object()
    calls = []

    def fit(*args, **kwargs):
        calls.append((args, kwargs))
        return fitted, (np.nan, np.nan)

    monkeypatch.setattr(fitting, "_prepare_mixture_context", lambda _: context)
    monkeypatch.setattr(fitting, "_initialize_mixture", lambda *args: initialization)
    monkeypatch.setattr(fitting, "_fit_mixture_with_boundary_policy", fit)
    monkeypatch.setattr(
        fitting,
        "_components_from_natural_mixture",
        lambda *args: ([object(), object()], np.full(2, 0.5), {}),
    )
    monkeypatch.setattr(
        fitting,
        "_sort_components_by_mode",
        lambda components, weights: (list(reversed(components)), weights[::-1]),
    )
    monkeypatch.setattr(fitting, "_mixture_fit_metadata", lambda *args: {})
    context.samples_rk = np.arange(20.0)[:, None]
    request = SimpleNamespace(
        seed_components=None,
        poly_degree=2,
        progressive=False,
        degree_config=context.degree_config,
    )
    result = fitting._run_mixture_fit(request)
    assert calls[0][0][3] is full_responsibilities
    assert calls[0][1].get("initial_fit") is None
    assert result.em_diagnostics["solver_to_public"] == (1, 0)


def test_degree_trace_retains_unresolved_information_and_owns_nested_arrays(
    monkeypatch,
):
    @dataclass
    class Diagnostic:
        information_status: str
        residual: np.ndarray

    diagnostic = Diagnostic("indefinite_nuisance", np.array([0.1, 0.2]))
    fitted = SimpleNamespace(
        components=(object(), object()),
        responsibilities=np.full((20, 2), 0.5),
        weights=np.full(2, 0.5),
        status="converged",
        em_iterations=2,
        polish_iterations=1,
        rounds=1,
        log_likelihood=-0.5,
        decrease_bound=1e-12,
        separator_certified=True,
        initialization="test",
        degree_diagnostics=(
            {
                "degrees": (4, 2),
                "diagnostics": ((1, diagnostic),),
                "expanded_component": None,
            },
        ),
    )
    monkeypatch.setattr(fitting, "_pack_natural_component", lambda *args, **kwargs: {})
    monkeypatch.setattr(fitting, "_Component", lambda state: object())
    _, _, record = fitting._components_from_natural_mixture(
        fitted, None, (np.nan, np.nan)
    )
    probe = record["degree_diagnostics"][0]["diagnostics"][0]
    assert probe["component"] == 1
    assert probe["information_status"] == "indefinite_nuisance"
    assert record["degree_component_order"] == "solver"
    probe["residual"][0] = 99.0
    assert diagnostic.residual[0] == 0.1
