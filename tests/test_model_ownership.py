"""Immutable fitted payloads and a single model-owned affine presentation."""

import copy
import io
import pickle

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._fit.mixture import _pack_mixture_struct, _sort_components_by_mode


@pytest.fixture(scope="module", params=[1, 2])
def original(request):
    rng = np.random.default_rng(413)
    models = [
        Distribution().fit(rng.normal(loc, 0.6, 160), n_components=1, poly_degree=2)
        for loc in (-1.3, 1.1)[: request.param]
    ]
    if request.param == 1:
        return models[0]
    return Distribution(
        _pack_mixture_struct(
            np.array([0.35, 0.65]),
            "base",
            [model.components[0].data for model in models],
        )
    )


def test_component_interface_has_no_independent_mutators(original):
    assert isinstance(original.components, tuple)
    for name in ("mu", "sigma", "mean", "var", "support", "weights", "components"):
        with pytest.raises(AttributeError):
            setattr(original, name, 1.0)
    for component in original.components:
        for name in ("fit", "load", "transform", "set_default"):
            assert not hasattr(component, name)
        for name in ("mu", "sigma", "pullback", "mean", "var", "support"):
            with pytest.raises(AttributeError):
                setattr(component, name, 1.0)
        with pytest.raises(ValueError):
            component._data["q_poly"][0] = 99.0
        with pytest.raises(ValueError):
            component._data.setflags(write=True)


def test_public_arrays_cannot_alias_fitted_parameters(original):
    before = original.pdf(np.array([-1.0, 0.0, 1.0]))
    weights = original.weights
    weights.setflags(write=True)
    weights[:] = 0
    for component in original.components:
        for support in (
            component.support,
            component.base.support,
            component.exp.support,
        ):
            support.setflags(write=True)
            support[:] = -123.0
        exported = component.data
        exported["q_poly"][:] = 0.0
    exported = original.data
    exported["comp_q_poly"][:] = 0.0
    np.testing.assert_array_equal(original.pdf([-1.0, 0.0, 1.0]), before)


def test_uniform_envelope_and_portable_roundtrip(original):
    model = original.transform(mu=0.4, sigma=1.7, pullback=False, inplace=False)
    model.set_default("exp")
    state = model.data
    names = set(state.dtype.names)
    assert not state.dtype.hasobject
    assert {"mu", "sigma", "default_space", "n_components", "comp_fit_center"} <= names
    assert (
        not {"comp_mu", "comp_sigma", "comp_pullback", "comp_default_space", "pullback"}
        & names
    )
    assert state["mu"] == 0.4 and state["sigma"] == 1.7
    stream = io.BytesIO()
    np.save(stream, state, allow_pickle=False)
    stream.seek(0)
    loaded = Distribution(np.load(stream, allow_pickle=False))
    np.testing.assert_array_equal(
        loaded.pdf([0.2, 1.0, 4.0]), model.pdf([0.2, 1.0, 4.0])
    )
    assert loaded.default == "exp"
    with pytest.raises(ValueError, match="missing fields"):
        Distribution(model.components[0].data)


def test_composition_and_original_reset(original):
    model = original.transform(mu=2.0, sigma=3.0, pullback=False, inplace=False)
    model.transform(mu=4.0, pullback=False)
    assert (model.mu, model.sigma) == (6.0, 3.0)
    model.transform(sigma=2.0, pullback=False)
    assert (model.mu, model.sigma) == (12.0, 6.0)
    before = model.data.tobytes()
    model.transform(pullback=False)
    assert model.data.tobytes() == before
    model.transform(mu=2.0, sigma=2.0, pullback=True)
    assert (model.mu, model.sigma) == (5.0, 3.0)
    model.transform(mu=-1.0, sigma=0.5, pullback=False, relative_to="original")
    assert (model.mu, model.sigma) == (-1.0, 0.5)
    x = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(model.pdf(-1 + 0.5 * x), original.pdf(x) / 0.5)
    model.transform(pullback=True, relative_to="original")
    assert (model.mu, model.sigma) == (0.0, 1.0)
    np.testing.assert_array_equal(model.pdf(x), original.pdf(x))


@pytest.mark.parametrize("copier", [lambda c: c.copy(), copy.copy, copy.deepcopy])
def test_component_copies_snapshot_the_model_presentation(original, copier):
    model = original.transform(mu=0.4, sigma=1.2, pullback=False, inplace=False)
    model.set_default("exp")
    component = copier(model.components[0])
    expected_mean = component.mean
    x = np.array([0.2, 1.0, 4.0])
    expected_pdf = component.pdf(x)

    model.transform(mu=10.0, pullback=False)
    model.set_default("base")

    assert component.default == "exp"
    assert component.mean == expected_mean
    np.testing.assert_array_equal(component.pdf(x), expected_pdf)


@pytest.mark.parametrize(
    "operation",
    [
        {"mu": np.inf},
        {"sigma": 0},
        {"sigma": np.nan},
        {"relative_to": "other"},
        {"pullback": 1},
    ],
)
def test_transform_validation_is_atomic(original, operation):
    model = original.copy()
    _ = model.mean
    state = model.data.tobytes()
    kwargs = {"pullback": False, **operation}
    with pytest.raises((ValueError, TypeError)):
        model.transform(**kwargs)
    assert model.data.tobytes() == state


def test_composition_overflow_is_atomic(original):
    model = original.transform(sigma=1e200, pullback=False, inplace=False)
    before = model.data.tobytes()
    with pytest.raises(ValueError, match="composed transform"):
        model.transform(sigma=1e200, pullback=False)
    assert model.data.tobytes() == before


def test_transforms_cover_probabilities_derivatives_and_moments(original):
    shift, scale = 0.3, 1.4
    model = original.transform(mu=shift, sigma=scale, pullback=False, inplace=False)
    x = np.array([-2.0, -0.1, 1.0])
    y = shift + scale * x
    np.testing.assert_allclose(model.pdf(y), original.pdf(x) / scale)
    np.testing.assert_allclose(model.logpdf(y), original.logpdf(x) - np.log(scale))
    np.testing.assert_allclose(model.cdf(y), original.cdf(x), atol=5e-8)
    for order in (1, 2, 3):
        np.testing.assert_allclose(
            model.neg_log(y, order), original.neg_log(x, order) / scale**order
        )
    assert model.mean == pytest.approx(shift + scale * original.mean)
    assert model.var == pytest.approx(scale**2 * original.var)
    assert model.moment(3, central=True) == pytest.approx(
        scale**3 * original.moment(3, central=True)
    )
    for method in ("logppf", "logisf"):
        assert getattr(model, method)(-80) == pytest.approx(
            shift + scale * getattr(original, method)(-80), rel=2e-8
        )
    for comp, baseline in zip(model.components, original.components, strict=True):
        np.testing.assert_allclose(
            comp.exp.pdf(np.exp(y)),
            baseline.base.pdf(x) / (scale * np.exp(y)),
        )
    rows = np.column_stack([x - 0.05, x + 0.05])
    assert model.loglik(shift + scale * rows) == pytest.approx(
        original.loglik(rows), abs=1e-7
    )


def test_queries_only_mutate_derived_caches(original):
    model = original.copy()
    before = [component.data.tobytes() for component in model.components]
    model.moment(6)
    model.cumulant(4)
    model.transform(mu=0.2, sigma=0.8, pullback=False).set_default("exp")
    _ = model.mean
    _ = model.var
    model.cdf(1.2)
    model.ppf(0.2)
    assert [component.data.tobytes() for component in model.components] == before


def test_copy_pickle_load_have_independent_presentation(original):
    model = original.transform(mu=2, sigma=0.7, pullback=False, inplace=False)
    for clone in (
        copy.copy(model),
        copy.deepcopy(model),
        pickle.loads(pickle.dumps(model)),
        Distribution(model.data),
    ):
        clone.transform(mu=10, pullback=False)
        assert clone.mu == 12 and model.mu == 2
        assert not np.shares_memory(clone._weights, model._weights)
        for a, b in zip(clone.components, model.components, strict=True):
            assert not np.shares_memory(a._data, b._data)


def test_diagnostics_are_detached_from_model_metadata(original):
    model = original.copy()
    diagnostic = model.fit_diagnostics
    for component in diagnostic["components"]:
        assert "boundary_standard_errors" not in component
        assert "boundary_p_values" not in component
    diagnostic["shared_boundary"]["amplitudes"] = (999.0, 999.0)
    assert model.fit_diagnostics["shared_boundary"]["amplitudes"] == (0.0, 0.0)
    model._selection_diagnostics = {
        "scores": ({"details": {"degrees": [2]}},),
    }
    exported = model.selection_diagnostics
    exported["scores"][0]["details"]["degrees"][0] = 99
    assert model.selection_diagnostics["scores"][0]["details"]["degrees"] == [2]


def test_nested_degree_diagnostics_are_isolated_by_getter_copy_and_pickle(original):
    model = original.copy()
    information = np.array([[2.0, 0.1], [0.1, 1.0]])
    model._em_diagnostics = {
        "converged": True,
        "degree_component_order": "solver",
        "solver_to_public": tuple(reversed(range(model.n_components))),
        "degree_diagnostics": (
            {
                "information_status": "resolved",
                "probe": {
                    "observed_information": information.copy(),
                    "score": np.array([0.2, 0.4]),
                },
            },
        ),
    }
    exported = model.fit_diagnostics["em"]
    exported["degree_diagnostics"][0]["probe"]["observed_information"][0, 0] = -100
    np.testing.assert_array_equal(
        model._em_diagnostics["degree_diagnostics"][0]["probe"]["observed_information"],
        information,
    )
    for clone in (
        model.copy(),
        copy.deepcopy(model),
        pickle.loads(pickle.dumps(model)),
    ):
        diagnostics = clone.fit_diagnostics["em"]
        assert (
            diagnostics["solver_to_public"] == model._em_diagnostics["solver_to_public"]
        )
        assert diagnostics["degree_component_order"] == "solver"
        assert diagnostics["degree_diagnostics"][0]["information_status"] == "resolved"
        cloned = clone._em_diagnostics["degree_diagnostics"][0]["probe"][
            "observed_information"
        ]
        np.testing.assert_array_equal(cloned, information)
        assert not np.shares_memory(cloned, information)
        cloned[0, 0] = 99
        assert (
            model._em_diagnostics["degree_diagnostics"][0]["probe"][
                "observed_information"
            ][0, 0]
            == 2
        )
    assert Distribution(model.data).fit_diagnostics["em"] is None


def test_component_sorting_retains_immutable_object_identity(original):
    components = list(reversed(original.components))
    weights = original.weights[::-1]
    ordered, ordered_weights = _sort_components_by_mode(components, weights)
    assert [id(component) for component in ordered] == [
        id(component) for component in original.components
    ]
    np.testing.assert_array_equal(ordered_weights, original.weights)


def test_joint_component_packing_never_claims_independent_inference(monkeypatch):
    from types import SimpleNamespace

    from gibbus._postfit import fitted_state

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return "packed"

    monkeypatch.setattr(fitted_state, "_pack_natural_state", capture)
    component = SimpleNamespace(
        state=lambda: None, spec=None, coordinate=None, solver_result=None
    )
    assert (
        fitted_state._pack_natural_component(
            component, effective_n=100, boundary_p_values=(0.01, 0.02)
        )
        == "packed"
    )
    assert np.isnan(captured["effective_n"])
    assert np.isnan(captured["boundary_p_values"]).all()


@pytest.mark.parametrize("name", ["n_parameters", "n_face_parameters"])
def test_load_validates_authoritative_model_dimensions(original, name):
    state = original.data
    state[name] += 1
    with pytest.raises(ValueError, match=name):
        Distribution(state)


def test_failed_fit_and_load_preserve_entire_model(original, monkeypatch):
    import gibbus._api.distribution as api

    model = original.copy()
    before = model.data.tobytes()
    components = model.components

    def fail(request):
        raise RuntimeError("deliberate fitting failure")

    monkeypatch.setattr(api, "_run_fit_request", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        model.fit(np.linspace(-1, 1, 50), n_components=1)
    assert model.components is components
    assert model.data.tobytes() == before
    bad = model.data
    bad["shared_boundary_amplitudes"][0] = 5
    with pytest.raises(ValueError, match="amplitude"):
        model.load(bad)
    assert model.components is components
    assert model.data.tobytes() == before


def test_old_components_survive_whole_model_replacement(original):
    model = original.copy()
    previous = model.components
    values = [comp.pdf(0.0) for comp in previous]
    model.load(original.data)
    model.transform(mu=10, pullback=False)
    assert [comp.pdf(0.0) for comp in previous] == values


def test_transformed_truncation_preserves_original_potential_anchors():
    model = (
        Distribution()
        .fit(
            np.random.default_rng(19).beta(2, 3, 220),
            n_components=1,
            support=(0, 1),
            poly_degree=2,
            log_boundary_lower=True,
            log_boundary_upper=True,
        )
        .transform(mu=2, sigma=3, pullback=False)
    )
    truncated = model.truncate(2.6, 4.4)
    np.testing.assert_allclose(truncated.support, [2.6, 4.4])
    assert (truncated.mu, truncated.sigma) == (2, 3)
    a, b = model.components[0].data, truncated.components[0].data
    for name in (
        "canonical_support",
        "boundary_amplitudes",
        "fit_center",
        "fit_scale",
        "fit_direction",
    ):
        np.testing.assert_array_equal(a[name], b[name])
    x = np.array([2.8, 3.5, 4.2])
    mass = model.cdf(4.4) - model.cdf(2.6)
    np.testing.assert_allclose(truncated.pdf(x), model.pdf(x) / mass, rtol=2e-7)
    assert truncated.fit_diagnostics["provenance"] == "derived"
    assert np.isnan(
        truncated.fit_diagnostics["shared_boundary"]["standard_errors"]
    ).all()


@pytest.mark.parametrize("support", [(0.0, 1.0), (-np.inf, 1.0)])
def test_transformed_component_seed_refits_in_current_coordinates(support):
    rng = np.random.default_rng(527)
    samples = (
        rng.beta(2.0, 3.0, 160)
        if np.isfinite(support[0])
        else 1.0 - rng.gamma(2.0, 0.8, 160)
    )
    original = Distribution().fit(
        samples,
        n_components=1,
        poly_degree=2,
        support=support,
        log_boundary_lower=False,
        log_boundary_upper=False,
    )
    seed = original.transform(mu=4.0, sigma=2.5, pullback=False, inplace=False)
    before = seed.components[0].data.tobytes()
    fitted = Distribution().fit(4.0 + 2.5 * samples, init_from=seed)
    np.testing.assert_array_equal(fitted.support, seed.support)
    np.testing.assert_allclose(
        fitted.pdf(4.0 + 2.5 * samples[::10]),
        seed.pdf(4.0 + 2.5 * samples[::10]),
        rtol=2e-5,
        atol=1e-8,
    )
    assert (fitted.mu, fitted.sigma) == (0.0, 1.0)
    assert seed.components[0].data.tobytes() == before


def test_seed_polynomial_is_reexpressed_in_target_conditioning():
    from types import SimpleNamespace

    from gibbus._api.component import _natural_seed_params

    captured = {}

    def pack(gamma, curvature, boundary):
        captured.update(gamma=gamma, curvature=curvature, boundary=boundary)
        return np.r_[gamma, curvature, boundary]

    layout = SimpleNamespace(
        curvature_degree=0, lower_a_index=2, upper_a_index=3, pack=pack
    )
    seed = {
        "q_poly": np.array([0.0, 2.0, 3.0]),
        "fit_direction": -1.0,
        "fit_scale": 2.0,
        "fit_center": 5.0,
        "boundary_amplitudes": np.array([0.0, 4.0]),
    }
    target = SimpleNamespace(direction=-1.0, scale=4.0, center=7.0)
    _natural_seed_params(seed, layout, target)
    # z_old = -1 + 2*z_new: 2*z_old + 3*z_old**2 = 1 - 8*z_new + 12*z_new**2.
    assert captured["gamma"] == -8.0
    np.testing.assert_array_equal(captured["curvature"], [24.0])
    np.testing.assert_array_equal(captured["boundary"], [4.0, 0.0])
    seed["q_poly"] = np.array([1.0, 0.0, 0.0])
    _natural_seed_params(seed, layout, target)
    assert captured["gamma"] == 0.0
    np.testing.assert_array_equal(captured["curvature"], [0.0])
