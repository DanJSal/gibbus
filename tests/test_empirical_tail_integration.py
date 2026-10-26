"""Automatic within-family tail parameters at the public fit boundary."""

import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.special import logsumexp

from gibbus import BasisSpec, Distribution, Exponential, Polynomial, SoftPower
from gibbus._fit import _empirical_tail_likelihood as likelihood
from gibbus._fit.tail_estimation import TailEstimate, estimate_tail, resolve_auto_tails
from gibbus._model.basis_spec import _canonicalize_basis_spec


def test_omitted_parameter_and_explicit_api():
    assert Polynomial().power == "auto"
    assert SoftPower().power == "auto"
    assert Exponential().rate == "auto"
    assert Polynomial(4).power == 4
    assert SoftPower(2.5).power == 2.5
    assert Exponential(0.7).rate == 0.7
    assert BasisSpec(lower_tail=Polynomial(), upper_tail=Exponential())
    with pytest.raises(ValueError, match="fit observations"):
        _canonicalize_basis_spec(BasisSpec(upper_tail=Exponential()))
    with pytest.raises(ValueError):
        Polynomial(0)
    with pytest.raises(ValueError):
        Exponential(-1)


def test_deterministic_parameter_selection_and_affine_rate():
    x = np.random.default_rng(47).gumbel(size=300)
    a = estimate_tail(x, family="exponential", side="lower")
    b = estimate_tail(3 * x + 10, family="exponential", side="lower")
    assert a.primary > 0
    assert abs(a.primary / 3 - b.primary) < 1e-8
    assert a == estimate_tail(x, family="exponential", side="lower")
    for family in ("polynomial", "soft_power"):
        estimate = estimate_tail(x, family=family, side="upper")
        assert 1 <= estimate.primary <= 6
        if family == "polynomial":
            assert estimate.primary == int(estimate.primary)


def test_resolve_preserves_explicit_and_no_family():
    x = np.random.default_rng(17).normal(size=150)
    spec = BasisSpec(lower_tail=Polynomial(2), upper_tail=Exponential())
    resolved, candidates = resolve_auto_tails(spec, x.reshape(-1, 1))
    assert resolved.lower_tail == Polynomial(2)
    assert resolved.upper_tail.rate > 0
    assert tuple(candidates) == ("upper",)
    assert resolve_auto_tails(BasisSpec(), x.reshape(-1, 1))[0] == BasisSpec()
    assert (
        resolve_auto_tails(BasisSpec(upper_tail=Exponential(0.8)), x.reshape(-1, 1))[1]
        == {}
    )


def test_censored_is_not_silently_interpreted_as_points():
    spec = BasisSpec(upper_tail=Polynomial())
    with pytest.raises(ValueError, match="point observations"):
        resolve_auto_tails(spec, np.array([[0.0, 1.0], [1.0, 2.0]]))


def test_point_fit_and_result_diagnostics():
    x = np.random.default_rng(31).normal(size=160)
    spec = BasisSpec(bulk_degree=2, upper_tail=Polynomial(), expansion="minimal")
    fitted = Distribution().fit(x, n_components=1, basis=spec)
    diag = fitted.tail_parameter_diagnostics
    assert diag["upper"]["family"] == "polynomial"
    assert diag["upper"]["estimated_parameter"] in range(1, 7)
    assert set(diag["upper"]) == {"family", "estimated_parameter", "tail_n"}
    assert fitted.fit_diagnostics["converged"]
    assert fitted.copy().tail_parameter_diagnostics == diag
    assert pickle.loads(pickle.dumps(fitted)).tail_parameter_diagnostics == diag
    fixed = Distribution().fit(
        x,
        n_components=1,
        basis=BasisSpec(bulk_degree=2, upper_tail=Polynomial(2), expansion="minimal"),
    )
    assert fixed.tail_parameter_diagnostics is None


def test_a_failing_automatic_tail_fit_is_not_retried():
    import gibbus._api.fitting as fitting

    x = np.random.default_rng(90).normal(size=100)
    basis = BasisSpec(bulk_degree=2, upper_tail=Exponential(), expansion="minimal")
    proposal = TailEstimate(primary=0.9, n_tail=40, family="exponential")
    seen = []

    def primary_fails(spec, samples, weights, **kwargs):
        seen.append(spec.upper_tail.rate)
        raise RuntimeError("generated failure")

    with (
        patch("gibbus._fit.tail_estimation.estimate_tail", return_value=proposal),
        patch("gibbus._api.fitting._fit_basis", side_effect=primary_fails),
        pytest.raises(RuntimeError, match="generated failure"),
    ):
        Distribution().fit(x, n_components=1, basis=basis)
    assert seen == [0.9]
    assert not hasattr(fitting, "alternative_spec")


def test_soft_power_estimate_at_p_one_says_the_tail_is_too_heavy():
    x = np.random.default_rng(11)
    x.normal(size=500)
    x.laplace(size=500)
    t4 = x.standard_t(4, 500)
    with pytest.raises(ValueError, match="heavier than any SoftPower"):
        estimate_tail(t4, family="soft_power", side="upper")


def test_weighted_points_estimate_is_deterministic():
    x = np.random.default_rng(191).normal(size=150)
    w = np.linspace(0.1, 2, len(x))
    spec = BasisSpec(upper_tail=Exponential())
    a, _ = resolve_auto_tails(spec, x[:, None], w)
    b, _ = resolve_auto_tails(spec, x[:, None], w)
    assert a == b
    assert a.upper_tail.rate > 0


def _reference_objective_gradient(lb, la, ym, mh, rate, ys, h, logwr):
    """Direct NumPy form of the conditional tail likelihood and its gradient."""
    b, a = np.exp(lb), np.exp(la)
    score = logwr - b * ys - a * h
    logsum = logsumexp(score)
    nll = b * ym + a * mh + logsum - np.log(rate)
    if not np.isfinite(nll):
        return 1e30, (0.0, 0.0)
    prob = np.exp(score - logsum)
    # A quadrature point at infinite h has exactly zero posterior weight.
    mean_y = float(np.dot(prob, ys))
    mean_h = float(np.dot(prob, np.where(np.isfinite(h), h, 0.0)))
    return float(nll), (b * (ym - mean_y), a * (mh - mean_h))


def test_compiled_kernel_matches_the_reference_likelihood(monkeypatch):
    x = np.random.default_rng(333).gumbel(size=250)
    compiled = estimate_tail(x, family="exponential", side="lower")
    monkeypatch.setattr(
        likelihood,
        "_tail_kernels",
        SimpleNamespace(objective_gradient=_reference_objective_gradient),
    )
    reference = estimate_tail(x, family="exponential", side="lower")
    assert abs(compiled.primary - reference.primary) < 2e-5


def test_no_auto_inference_for_explicit_fit(monkeypatch):
    from gibbus._fit import tail_estimation

    x = np.random.default_rng(57).normal(size=120)
    with patch.object(
        tail_estimation,
        "estimate_tail",
        side_effect=AssertionError("unexpected inference"),
    ):
        d = Distribution().fit(
            x,
            n_components=1,
            basis=BasisSpec(
                bulk_degree=2, upper_tail=Polynomial(2), expansion="minimal"
            ),
        )
    assert d.fit_diagnostics["converged"]
    assert d.tail_parameter_diagnostics is None
