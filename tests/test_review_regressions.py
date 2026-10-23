"""Public request semantics and persistence boundaries of the development build."""

import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gibbus import BasisSpec, Distribution, Exponential, Polynomial, SoftPower
from gibbus import _serialization as serialization
from gibbus._fit import _empirical_tail_likelihood as likelihood
from gibbus._fit.tail_estimation import estimate_tail
from gibbus._postfit.basis_state import _model_from_record, _model_record


@pytest.fixture
def sample():
    return np.random.default_rng(31).normal(size=100)


@pytest.fixture
def model(sample):
    return Distribution().fit(
        sample,
        basis=BasisSpec(bulk_degree=2, upper_tail=Polynomial(2), expansion="minimal"),
    )


@pytest.mark.parametrize("degrees", [(4,), (4, 4), (2, 4)])
def test_component_degrees_override_the_explicit_basis(degrees):
    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(6 * k, 0.5, 100) for k in range(len(degrees))])
    fitted = Distribution().fit(
        x,
        n_components=len(degrees),
        basis=BasisSpec(bulk_degree=2, expansion="minimal"),
        component_options=[{"poly_degree": degree} for degree in degrees],
        rng=0,
    )
    assert [int(c.data["bulk_degree"]) for c in fitted.components] == list(degrees)


def test_warm_start_accepts_automatic_component_degree(model, sample):
    fitted = Distribution().fit(
        sample, init_from=model, component_options=[{"poly_degree": "auto"}]
    )
    assert fitted.fit_diagnostics["converged"]


def test_general_basis_degree_overrides_retain_valid_odd_faces():
    rng = np.random.default_rng(3)
    x = np.r_[rng.laplace(-4.0, 0.5, 100), rng.laplace(4.0, 0.5, 100)]
    spec = BasisSpec(
        bulk_degree=2,
        lower_tail=Polynomial(1),
        upper_tail=Polynomial(1),
        expansion="minimal",
    )
    fitted = Distribution().fit(
        x,
        n_components=2,
        basis=spec,
        component_options=[{"poly_degree": 3}, {"poly_degree": 2}],
    )
    assert [int(c.data["bulk_degree"]) for c in fitted.components] == [3, 2]
    assert fitted.fit_diagnostics["converged"]


def test_bootstrap_preserves_family_and_accepts_basis_override(
    model, sample, monkeypatch
):
    actual = Distribution.fit
    records = []

    def fit(self, observations, **kwargs):
        result = actual(self, observations, **kwargs)
        records.append(self.components[0]._basis.parent.request)
        return result

    monkeypatch.setattr(Distribution, "fit", fit)
    model.bootstrap_bands(sample, [0.0], n_resamples=2, rng=1)
    assert all(
        r.bulk_degree == 2 and r.upper_tail.kind == "polynomial" for r in records
    )
    records.clear()
    spec = BasisSpec(bulk_degree=2, upper_tail=Exponential(0.5), expansion="minimal")
    model.bootstrap_bands(
        sample, [0.0], n_resamples=1, rng=1, fit_kwargs={"basis": spec}
    )
    assert records[0].upper_tail.kind == "exponential"
    assert records[0].upper_tail.value == 0.5


def test_bootstrap_resamples_weights_with_their_rows(model, sample, monkeypatch):
    actual = Distribution.fit
    weights = np.linspace(0.1, 2.0, sample.size)
    expected = dict(zip(sample, weights / weights.sum()))
    seen = []

    def fit(self, observations, **kwargs):
        x = np.asarray(observations).ravel()
        np.testing.assert_allclose(kwargs["sample_weight"], [expected[v] for v in x])
        seen.append(True)
        return actual(self, observations, **kwargs)

    monkeypatch.setattr(Distribution, "fit", fit)
    model.bootstrap_bands(
        sample, [0.0], n_resamples=2, rng=7, fit_kwargs={"sample_weight": weights}
    )
    assert len(seen) == 2


def test_monte_carlo_null_is_invariant_under_exp_view():
    x = np.random.default_rng(17).normal(size=100)
    x = x[abs(x) < 2]
    base = Distribution().fit(
        x,
        poly_degree=2,
        support=(-3.0, 3.0),
        log_boundary_lower=False,
        log_boundary_upper=False,
    )
    exp = base.copy().set_default("exp")
    options = dict(calibration="montecarlo", n_resamples=3, rng=5)
    a = base.goodness_of_fit(x, **options)
    b = exp.goodness_of_fit(np.exp(x), **options)
    assert a["value"] == pytest.approx(b["value"], abs=1e-12)
    assert a["pvalue"] == b["pvalue"]
    assert a["n_failed"] == b["n_failed"] == 0


@pytest.mark.parametrize("factor", [1e-200, 1e200])
def test_automatic_tail_rate_handles_extreme_units(factor):
    x = np.random.default_rng(47).gumbel(size=160)
    a = estimate_tail(x, family="exponential", side="lower")
    b = estimate_tail(factor * x, family="exponential", side="lower")
    assert b.primary * factor == pytest.approx(a.primary, rel=2e-5)


def test_unsuccessful_tail_optimization_is_not_an_estimate(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(fun=2.0, x=np.array([0.0, -1.0]), success=False)

    monkeypatch.setattr(likelihood, "minimize", fail)
    with pytest.raises(ArithmeticError, match="did not converge"):
        likelihood.potential_fixed(np.linspace(0.1, 2.0, 20), 1.0, 1.0, p=2)
    assert len(calls) == 2


def test_tail_search_checks_the_shared_deadline(monkeypatch):
    from gibbus._fit.budget import FitTimeoutError

    def expire(stage):
        raise FitTimeoutError(0.1, 0.2, stage)

    monkeypatch.setattr(likelihood, "_check_fit_time", expire)
    with pytest.raises(FitTimeoutError):
        likelihood.potential_fixed(np.linspace(0.1, 2.0, 20), 1.0, 1.0, p=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("columns", None),
        ("request", 5),
        ("geometry", 5),
        ("version", 2.5),
        ("version", True),
        ("slots", []),
    ],
)
def test_malformed_basis_records_raise_value_error(model, field, value):
    payload = json.loads(str(model.data["model"]["basis_models"][0]))
    payload[field] = value
    with pytest.raises(ValueError):
        _model_from_record(json.dumps(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("support_kind", "bounded"),
        (
            "canonical_support",
            {"tuple": [{"float": "-0x1.0p+0"}, {"float": "0x1.0p+0"}]},
        ),
    ],
)
def test_model_record_rejects_inconsistent_coordinates(model, field, value):
    payload = json.loads(str(model.data["model"]["basis_models"][0]))
    payload["geometry"]["fields"]["coordinate"]["fields"][field] = value
    with pytest.raises(ValueError, match="support.*disagrees"):
        _model_from_record(json.dumps(payload))


@pytest.mark.parametrize(
    "field", ["upper_location", "upper_steepness", "upper_rate", "upper_delta"]
)
def test_model_record_rejects_unresolved_active_geometry(sample, field):
    fitted = Distribution().fit(
        sample,
        basis=BasisSpec(
            upper_tail=SoftPower(1.5) if field == "upper_delta" else Exponential(1.0),
            expansion="minimal",
        ),
    )
    payload = json.loads(str(fitted.data["model"]["basis_models"][0]))
    payload["geometry"]["fields"][field] = None
    with pytest.raises(ValueError, match="unresolved upper parameters"):
        _model_from_record(json.dumps(payload))


@pytest.mark.parametrize("support", [(-np.inf, np.inf), (0.0, np.inf), (-np.inf, 0.0)])
@pytest.mark.parametrize(
    "tail", [Polynomial(1), Polynomial(2), SoftPower(1.5), SoftPower(2), Exponential(1)]
)
def test_resolved_basis_records_round_trip_across_tail_families(support, tail):
    from gibbus._model.basis import _build_basis

    x = np.random.default_rng(31).normal(size=100)
    if np.isfinite(support[0]):
        x = 1.0 + np.abs(x)
    elif np.isfinite(support[1]):
        x = -1.0 - np.abs(x)
    spec = BasisSpec(
        support=support,
        lower_tail=tail if np.isneginf(support[0]) else None,
        upper_tail=tail if np.isposinf(support[1]) else None,
        expansion="minimal",
    )
    record = _model_record(_build_basis(spec, x))
    rebuilt, suppressed = _model_from_record(record)
    assert suppressed == ()
    assert _model_record(rebuilt) == record


def test_changed_implementation_rebuilds_optional_cache(model, sample, monkeypatch):
    state = model.data
    actual = serialization._rebuild_basis_component
    rebuilt = []

    def rebuild(*args):
        rebuilt.append(True)
        return actual(*args)

    monkeypatch.setattr(serialization, "_implementation_digest", lambda: "f" * 64)
    monkeypatch.setattr(serialization, "_rebuild_basis_component", rebuild)
    loaded = Distribution(state)
    assert rebuilt == [True]
    np.testing.assert_allclose(loaded.logpdf(sample), model.logpdf(sample), atol=1e-10)


def test_uncertified_result_is_visible_at_public_boundary(model, sample, monkeypatch):
    import gibbus._api.fitting as fitting

    fitted = fitting._fit_basis(BasisSpec(bulk_degree=2), sample, None)
    result = dataclasses.replace(
        fitted, result=dataclasses.replace(fitted.result, status="uncertified")
    )
    monkeypatch.setattr(fitting, "_fit_basis", lambda *args, **kwargs: result)
    with pytest.warns(RuntimeWarning, match="uncertified shape"):
        loaded = Distribution().fit(sample, poly_degree=2)
    assert not loaded.fit_diagnostics["converged"]


def test_tail_accelerator_validates_node_dimensions():
    from gibbus._fit._tail_likelihood_kernels import objective_gradient

    with pytest.raises(ValueError, match="equal nonzero lengths"):
        objective_gradient(0.0, 0.0, 1.0, 1.0, 1.0, np.ones(2), np.ones(1), np.ones(2))
