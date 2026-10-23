"""Tests for the independent case generator; deliberately do not fit Gibbus."""

import json
import math

import numpy as np
import pytest
from scipy import integrate

from tests.g3_casegen import (
    Affine,
    CaseSpec,
    CaseUniverse,
    FitSpec,
    Mixture,
    Named,
    ObservationSpec,
    Sum,
    Truncated,
    cdf,
    default_distributions,
    from_dict,
    materialize,
    pdf,
    sample,
    smoothness,
    support,
    support_kind,
    to_dict,
    validate,
)
from tests.g3_casegen.cases import _canonical, case_from_dict, coverage_counts, manifest

CATALOG = default_distributions("standard")


@pytest.mark.parametrize("dist", CATALOG, ids=lambda x: str(to_dict(x))[:90])
def test_every_catalog_entry_generates_in_support(dist):
    validate(dist)
    rng = np.random.default_rng(123)
    x = sample(dist, 32, rng)
    lo, hi = support(dist)
    assert x.shape == (32,)
    assert np.all(np.isfinite(x))
    assert np.all((x >= lo) & (x <= hi))
    assert from_dict(to_dict(dist)) == dist


@pytest.mark.parametrize(
    "mode",
    ("points", "weighted", "left", "right", "interval", "rounded", "binned", "mixed"),
)
@pytest.mark.parametrize(
    "dist",
    (
        Named("normal", mu=0, sigma=1),
        Named("gamma", shape=2, scale=1),
        Affine(Named("gamma", shape=2, scale=1), scale=-1),
        Named("beta", a=2, b=3),
    ),
)
def test_observation_mechanism_contains_latent(mode, dist):
    s = CaseSpec(dist, ObservationSpec(mode, 0.5, 0.15), FitSpec(), 512)
    a = materialize(s, seed=7)
    b = materialize(s, seed=7)
    np.testing.assert_array_equal(a.latent, b.latent)
    np.testing.assert_array_equal(a.observations, b.observations)
    assert a.spec.case_id == b.spec.case_id
    if mode in ("points", "weighted"):
        assert a.observations.shape == (512,)
        if mode == "weighted":
            assert np.all(a.weights > 0)
            assert a.weights.mean() == pytest.approx(1.0)
        else:
            assert a.weights is None
    else:
        assert a.observations.shape == (512, 2)
        assert np.all(a.observations[:, 0] <= a.latent)
        assert np.all(a.observations[:, 1] >= a.latent)
        lo, hi = support(dist)
        assert np.all(a.observations[:, 0] >= lo)
        assert np.all(a.observations[:, 1] <= hi)
        assert np.any(a.censored)


def test_named_parameter_curvature_guards():
    invalid = (
        Named("gamma", shape=0.5, scale=1),
        Named("weibull", shape=0.5, scale=1),
        Named("beta", a=0.5, b=2),
        Named("kumaraswamy", a=2, b=0.7),
        Named("generalized_normal", p=1, mu=0, scale=1),
        Named("generalized_gamma", a=0.2, c=2, scale=1),
        Named("generalized_inverse_gaussian", p=0.1, b=1),
        Named("generalized_pareto_bounded", shape=0.3),
        Named("normal", mu=0, sigma=-1),
        Named("normal", mu=float("nan"), sigma=1),
        Named("cauchy_restricted"),
        Truncated(Named("cauchy_restricted"), -2, 2),
        Truncated(Named("lognormal_restricted", mu=0, sigma=1), 0.1, 2),
        Truncated(Named("student_t_restricted", df=3), -2, 2),
        Truncated(Named("beta", a=0.5, b=2), 0.1, 1),
        Truncated(Named("beta", a=2, b=0.5), 0, 0.9),
        Truncated(Named("inverse_gaussian_restricted", mu=1, scale=1), 0.1, 1),
    )
    for case in invalid:
        with pytest.raises(ValueError):
            validate(case)
    validate(Truncated(Named("cauchy_restricted"), -0.8, 0.8))
    validate(Truncated(Named("lognormal_restricted", mu=0, sigma=1), 0.01, 0.99))
    validate(Truncated(Named("student_t_restricted", df=4), -1.8, 1.8))
    validate(Truncated(Named("beta", a=0.5, b=2), 0.5, 1))
    validate(Truncated(Named("beta", a=2, b=0.5), 0, 0.5))
    validate(Truncated(Named("inverse_gaussian_restricted", mu=1, scale=1), 0.1, 0.5))


def test_mixture_rejects_incompatible_supports_and_weights():
    normal = Named("normal", mu=0, sigma=1)
    gamma = Named("gamma", shape=2, scale=1)
    with pytest.raises(ValueError, match="share exact support"):
        validate(Mixture(((0.5, normal), (0.5, gamma))))
    with pytest.raises(ValueError, match="sum to one"):
        validate(Mixture(((0.2, normal), (0.2, normal))))


def test_sum_support_and_conservative_smoothness():
    n = Named("normal", mu=0, sigma=1)
    b = Named("beta", a=2, b=3)
    s = Sum(b, b)
    validate(s)
    assert support(s) == (0, 2)
    assert smoothness(s) == "unknown"
    assert smoothness(Sum(n, b)) == "Cinf"
    x = sample(s, 200, np.random.default_rng(1))
    assert (x.min() > 0) and (x.max() < 2)


def test_oracles_basic_and_truncation_normalize():
    candidates = [
        Named("normal", mu=0, sigma=1),
        Named("beta_logistic", a=2, b=4),
        Named("kumaraswamy", a=2, b=3),
        Truncated(Named("cauchy_restricted"), -0.9, 0.9),
    ]
    for d in candidates:
        lo, hi = support(d)
        box = (-8, 8) if not math.isfinite(lo) else (lo, hi)
        integral = integrate.quad(lambda z: float(pdf(d, z)), *box, epsabs=2e-9)[0]
        if not math.isfinite(lo):
            assert integral > 0.999
        else:
            assert integral == pytest.approx(1, abs=2e-8)
        q = sample(d, 2000, np.random.default_rng(12))
        assert 0.35 < float(cdf(d, np.median(q))) < 0.65


def test_lazy_universe_and_all_nine_fit_categories():
    u = CaseUniverse(CATALOG, sample_sizes=(100, 500))
    selected = u.select(150, seed=42)
    assert len(selected) == 150
    assert len(set(c.case_id for c in selected)) == 150
    assert len(set(c.scenario_id for c in selected)) <= 150
    assert {c.fit.fit_category(support(c.distribution)) for c in selected} == {
        "S01",
        "S02",
        "S03",
        "S04",
        "S05",
        "S06",
        "S07",
        "S08",
        "S09",
    }
    assert {support_kind(support(c.distribution)) for c in selected} == {
        "full",
        "lower",
        "upper",
        "bounded",
    }
    assert selected == u.select(150, seed=42)
    assert selected != u.select(150, seed=43)
    assert (
        selected[0].case_id
        != CaseSpec(
            selected[0].distribution,
            selected[0].observation,
            selected[0].fit,
            selected[0].n,
            5,
        ).case_id
    )
    assert (
        len(
            list(
                CaseUniverse(
                    [Named("normal", mu=0, sigma=1)],
                    observations=[ObservationSpec("points")],
                    degrees=(2,),
                    tails=("none",),
                    sample_sizes=(10,),
                    component_counts=(1,),
                ).iter_cases()
            )
        )
        == 1
    )


def test_balancing_and_exclusion_and_fixed_case():
    u = CaseUniverse(CATALOG, sample_sizes=(100, 500))
    fixed = next(u.iter_cases())
    sel = u.select(90, seed=4, fixed=(fixed,), explore_fraction=0.25)
    assert fixed in sel
    selected_ids = {c.case_id for c in sel}
    new = u.select(30, seed=4, exclude_case_ids=selected_ids)
    assert not selected_ids.intersection(c.case_id for c in new)
    categories = {c.strata()["composition"] for c in sel}
    assert categories == {"named", "mixture", "sum"}


def test_serialized_spec_identifies_cases_independent_of_global_rng():
    s = CaseSpec(
        Named("beta", a=2, b=5),
        ObservationSpec("interval", 0.3),
        FitSpec(log_lower=True, log_upper=True),
        100,
    )
    json.dumps(s.as_dict(), allow_nan=False)
    assert s.scenario_id.startswith("G3S-") and s.case_id.startswith("G3C-")
    assert _canonical(s.as_dict()) == _canonical(s.as_dict())
    x = materialize(s, seed=3)
    y = materialize(s, seed=4)
    assert not np.array_equal(x.latent, y.latent)


def test_support_log_constraints():
    with pytest.raises(ValueError):
        CaseSpec(
            Named("normal", mu=0, sigma=1),
            ObservationSpec(),
            FitSpec(log_lower=True),
            100,
        ).validate()
    with pytest.raises(ValueError):
        CaseSpec(
            Named("beta", a=2, b=3), ObservationSpec(), FitSpec(tail="poly2"), 100
        ).validate()
    with pytest.raises(ValueError):
        FitSpec(degree="auto", tail="poly2").validate((-math.inf, math.inf))


def test_manifest_roundtrip_and_infinite_truncation_endpoints():
    dist = Truncated(Named("normal", mu=0, sigma=1), -math.inf, 1.0)
    case = CaseSpec(dist, ObservationSpec("left", 0.2), FitSpec(), 100)
    case.validate()
    packed = manifest([case], 42)
    serialized = json.dumps(packed, allow_nan=False)
    restored = case_from_dict(json.loads(serialized)["cases"][0]["spec"])
    assert restored == case
    assert restored.case_id == packed["cases"][0]["case_id"]
    assert coverage_counts([case])
    assert materialize(restored).observations.shape == (100, 2)


def test_fit_kwargs_adapter_works_when_gibbus_built():
    try:
        from gibbus import BasisSpec
    except ImportError:
        pytest.skip(
            "Gibbus compiled extensions not installed; generator needs only SciPy"
        )
    for d in (
        Named("normal", mu=0, sigma=1),
        Named("gamma", shape=2, scale=1),
        Affine(Named("gamma", shape=2, scale=1), scale=-1),
        Named("beta", a=2, b=3),
    ):
        bounds = support(d)
        for ll in ((False, True) if math.isfinite(bounds[0]) else (False,)):
            for lu in ((False, True) if math.isfinite(bounds[1]) else (False,)):
                kw = FitSpec(ll, lu).fit_kwargs(bounds)
                assert isinstance(kw["basis"], BasisSpec)
                assert (
                    "support" not in kw
                )  # Public API forbids support together with basis.
                assert kw["basis"].support == bounds


def test_pairwise_catalog_expands_all_valid_two_family_structures():
    expanded = default_distributions("core", pairwise=True)
    assert len(expanded) > 500
    kinds = {"named": 0, "sum": 0, "mixture": 0}
    for d in expanded:
        from tests.g3_casegen import category

        kinds[category(d)] += 1
        if isinstance(d, Mixture):
            assert support(d.components[0][1]) == support(d.components[1][1])
    assert kinds["sum"] > 100 and kinds["mixture"] > 100


def test_three_component_mixture_is_supported():
    d = Mixture(
        (
            (0.2, Named("normal", mu=-3, sigma=1)),
            (0.3, Named("logistic", mu=0, scale=1)),
            (0.5, Named("gumbel_r", mu=3, scale=1)),
        )
    )
    validate(d)
    case = CaseSpec(d, ObservationSpec("points"), FitSpec(n_components=3), 100)
    assert case.strata()["component_alignment"] == "matched_or_selected"
    assert materialize(case).observations.shape == (100,)
