"""Tail classes crossed with point, interval and mixture fits (G3-049).

Every requested tail class at each end is fitted through the public API to
point data, binned interval data and a two-component mixture, then queried:
extreme quantiles, the exact format-2 round trip and outward growth far in
the tails.  Interval fits must reproduce their fitted likelihood through the
public CDF.  Targeted regressions cover the fixes of G3-049: leader faces
packed as exact face models, the collapsed-leader second start, valid tail
cuts at a zero leader, untestable larger spans and warm starts that keep
their seed's basis family.
"""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from gibbus import BasisSpec, Distribution, Exponential, Polynomial, SoftPower
from gibbus._fit.basis_selection import _fit_basis
from gibbus._model.basis import _build_basis, _face_model
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._postfit.basis_state import (
    _check_on_face,
    _model_from_record,
    _model_record,
)

N = 800
_RNG = np.random.default_rng(0)
DATA = {
    "laplace": _RNG.laplace(size=N),
    "gennorm1.5": stats.gennorm.rvs(1.5, size=N, random_state=_RNG),
    "normal": _RNG.normal(size=N),
    "gumbel": _RNG.gumbel(size=N),
}
TAILS = {
    "exp1": Exponential(1.0),
    "exp2": Exponential(2.0),
    "soft1.5": SoftPower(1.5),
    "soft2.5": SoftPower(2.5),
    "poly1": Polynomial(1),
    "poly3": Polynomial(3),
}
PAIRS = [
    ("laplace", "exp1"),
    ("laplace", "exp2"),
    ("gennorm1.5", "soft1.5"),
    ("normal", "soft2.5"),
    ("laplace", "poly1"),
    ("gumbel", "poly3"),
    ("normal", "exp1"),
]


def _cases():
    for data, tail in PAIRS:
        for end in ("both", "upper", "lower"):
            for path in ("points", "intervals", "mixture"):
                key = (data, tail, end, path)
                yield pytest.param(*key, id="-".join(key))


def _spec(tail, end):
    t = TAILS[tail]
    kw = {
        "lower_tail": t if end in ("both", "lower") else None,
        "upper_tail": t if end in ("both", "upper") else None,
    }
    if tail == "poly1" and end == "both":
        kw["bulk_degree"] = 2
    return BasisSpec(**kw)


def _binned(x):
    lo = np.floor(x * 4.0) / 4.0
    return np.c_[lo, lo + 0.25]


@pytest.mark.parametrize("data,tail,end,path", list(_cases()))
def test_tail_class_fits_answer_queries_and_round_trip(data, tail, end, path):
    x = DATA[data]
    if path == "intervals":
        samples, k = _binned(x), 1
    elif path == "mixture":
        samples, k = np.r_[x - 4.0, x + 4.0], 2
    else:
        samples, k = x, 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = Distribution().fit(samples, n_components=k, basis=_spec(tail, end), rng=0)
    assert d.n_components == k
    q = d.ppf([1e-12, 0.5, 1.0 - 1e-12])
    assert np.all(np.isfinite(q)) and np.all(np.diff(q) > 0.0)
    rebuilt = Distribution(d.data)
    np.testing.assert_allclose(rebuilt.logpdf(q), d.logpdf(q), rtol=1e-10)
    far = d.neg_log(np.array([-1e4, 1e4]))
    # Outward growth (``+inf`` once an exponential tail overflows), never NaN.
    assert np.all(far > 0.0)
    if path == "intervals":
        # The packed state is the fitted density: its interval masses give
        # the fitted likelihood.
        mass = d.cdf(samples[:, 1]) - d.cdf(samples[:, 0])
        nll = float(d.components[0].data["nll"])
        assert -float(np.mean(np.log(mass))) == pytest.approx(nll, rel=1e-9)


# --------------------------------------------------------------------------
# Leader faces as exact face models
# --------------------------------------------------------------------------


def _laplace_interval_fit(spec):
    return Distribution().fit(_binned(DATA["laplace"]), basis=spec)


@pytest.mark.parametrize(
    "spec",
    [
        BasisSpec(),
        BasisSpec(upper_tail=Exponential(1.0)),
        BasisSpec(lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)),
    ],
    ids=["auto-expansion", "upper-exp", "both-exp"],
)
def test_interval_fits_on_a_leader_face_pack_and_reproduce_their_likelihood(spec):
    """Regression: these fits ended on a leader face and failed to pack.

    The suppressed leader is a cancellation of gated columns in the parent
    basis; packing re-checked properness on the full basis and saw its
    rounding residue (``properness undecided``), or integrated a far tail
    the residue had made improper.
    """
    d = _laplace_interval_fit(spec)
    pot = d.components[0]._basis
    assert pot.suppressed
    assert pot.model.face == pot.suppressed
    samples = _binned(DATA["laplace"])
    mass = d.cdf(samples[:, 1]) - d.cdf(samples[:, 0])
    nll = float(d.components[0].data["nll"])
    assert -float(np.mean(np.log(mass))) == pytest.approx(nll, rel=1e-10)
    assert np.all(d.neg_log(np.array([-1e6, -1e3, 1e3, 1e6])) > 0.0)
    rebuilt = Distribution(d.data)
    assert rebuilt.components[0]._basis.suppressed == pot.suppressed
    x = np.linspace(-6.0, 6.0, 7)
    np.testing.assert_allclose(rebuilt.logpdf(x), d.logpdf(x), rtol=1e-12)


def test_face_model_cancels_the_suppressed_leader_exactly():
    model = _build_basis(
        BasisSpec(lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)),
        DATA["laplace"],
    )
    face, free = _face_model(model, ("lower", "upper"))
    assert face.face == ("lower", "upper")
    for side in ("lower", "upper"):
        parent = model.side_asymptotics(side).leading.order
        assert face.side_asymptotics(side).leading.order < parent
    # Any parent theta on the face: phi = theta[free] reproduces q in the
    # body to rounding and the face model has no exponential growth.
    rng = np.random.default_rng(1)
    phi = rng.normal(size=len(free))
    d = model.n_coefficients
    theta = np.zeros(d)
    vectors = np.array([[float(v) for v in col] for col in face.coefficient_map_exact])
    theta = np.linalg.lstsq(model.coefficient_map, vectors @ phi, rcond=None)[0]
    np.testing.assert_allclose(theta[list(free)], phi, atol=1e-12)
    x = np.linspace(-3.0, 3.0, 13)
    q_parent = _evaluate_basis(model, x, max_order=0).phi @ theta
    q_face = _evaluate_basis(face, x, max_order=0).phi @ phi
    np.testing.assert_allclose(q_face, q_parent, rtol=1e-12, atol=1e-12)


def test_record_carries_the_face_and_rejects_coefficients_off_it():
    model = _build_basis(BasisSpec(upper_tail=Exponential(1.0)), DATA["laplace"])
    text = _model_record(model, ("upper",))
    again, suppressed = _model_from_record(text)
    assert suppressed == ("upper",)
    assert [c.label for c in again.columns] == [c.label for c in model.columns]
    theta = np.ones(model.n_coefficients)
    with pytest.raises(ValueError, match="not on the recorded face"):
        _check_on_face(model, theta, ("upper",))
    with pytest.raises(ValueError, match="finite endpoint|not an infinite end"):
        _face_model(
            _build_basis(BasisSpec(support=(0.0, math.inf)), np.abs(DATA["normal"])),
            ("lower",),
        )


# --------------------------------------------------------------------------
# Fitter regressions
# --------------------------------------------------------------------------


def test_collapsed_exponential_leader_converges_on_its_face():
    """Regression: the leader crept to 1.6e-14 and hit the iteration limit."""
    fit = _fit_basis(
        BasisSpec(lower_tail=Exponential(2.0), expansion="minimal"), DATA["laplace"]
    )
    assert fit.result.status == "converged"
    assert (0, "lower") in fit.result.face.suppressed


def test_linear_tails_at_both_ends_find_a_strictly_convex_start():
    """Regression: a cut on the weaker tail atom emptied the cone.

    The first start direction had a zero leading tail-curvature coefficient;
    the next atom's row was cut, which is not valid where the leader is
    positive and excluded every member of the family.
    """
    d = Distribution().fit(
        DATA["laplace"],
        basis=BasisSpec(
            bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Polynomial(1)
        ),
    )
    assert d.fit_diagnostics["converged"]
    assert math.isfinite(d.tail_rate("lower")) and math.isfinite(d.tail_rate("upper"))


def test_untestable_larger_span_ends_the_automatic_expansion(monkeypatch):
    """An unresolved score test retains the fitted span and records why."""
    from gibbus._fit import basis_selection
    from gibbus._model.basis_normalizer import _QuadratureFailure

    baseline = _fit_basis(
        BasisSpec(
            lower_tail=Exponential(2.0),
            upper_tail=Exponential(2.0),
            expansion="minimal",
        ),
        DATA["laplace"],
    )

    def unresolved(*args, **kwargs):
        raise _QuadratureFailure("nonfinite larger-span moments", {"stage": "score"})

    monkeypatch.setattr(basis_selection, "_span_diagnostic", unresolved)
    fit = _fit_basis(
        BasisSpec(lower_tail=Exponential(2.0), upper_tail=Exponential(2.0)),
        DATA["laplace"],
    )
    assert fit.result.status == "converged"
    failures = [s.failure for s in fit.history if s.kind == "expansion"]
    assert failures and "untestable" in failures[-1]
    np.testing.assert_array_equal(fit.result.params, baseline.result.params)
    assert all(s.chosen is None for s in fit.history if s.kind == "expansion")


def test_finite_larger_span_stops_when_the_score_accepts():
    """Collapsed exponential leaders need not make larger-span moments infinite.

    A retained quadratic leader dominates every exponential statistic. The
    score may therefore be computed and accept the current span normally.
    """
    fit = _fit_basis(
        BasisSpec(lower_tail=Exponential(2.0), upper_tail=Exponential(2.0)),
        DATA["laplace"],
    )
    assert fit.result.status == "converged"
    last = [s for s in fit.history if s.kind == "expansion"][-1]
    assert not last.failure
    assert last.chosen is None
    assert last.diagnostic is not None
    assert not last.diagnostic.should_expand


# --------------------------------------------------------------------------
# Warm starts and the basis family
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tailed_seed():
    return Distribution().fit(
        DATA["laplace"],
        basis=BasisSpec(lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)),
    )


def _request(d):
    return d.components[0]._basis.parent.request


def test_warm_start_keeps_the_seed_basis_family(tailed_seed):
    new = np.random.default_rng(3).laplace(0.1, 1.0, N)
    d = Distribution().fit(new, init_from=tailed_seed)
    seed, warm = _request(tailed_seed), _request(d)
    assert warm.lower_tail == seed.lower_tail
    assert warm.upper_tail == seed.upper_tail
    assert warm.bulk_degree == seed.bulk_degree


def test_warm_start_maps_tail_rates_through_a_presentation(tailed_seed):
    moved = tailed_seed.transform(mu=3.0, sigma=2.0, pullback=False)
    new = 3.0 + 2.0 * np.random.default_rng(4).laplace(size=N)
    d = Distribution().fit(new, init_from=moved)
    # Rate 1 per model unit is rate 1/2 per public unit (sigma = 2).
    assert _request(d).upper_tail.value == pytest.approx(0.5)


def test_explicit_basis_replaces_the_seed_family(tailed_seed):
    new = np.random.default_rng(5).laplace(size=N)
    d = Distribution().fit(
        new, init_from=tailed_seed, basis=BasisSpec(upper_tail=Exponential(1.0))
    )
    assert _request(d).lower_tail is None
    assert _request(d).upper_tail.kind == "exponential"


def test_explicit_basis_must_keep_the_seed_support(tailed_seed):
    with pytest.raises(ValueError, match="init_from fixes the support"):
        Distribution().fit(
            DATA["laplace"],
            init_from=tailed_seed,
            basis=BasisSpec(support=(-10.0, math.inf)),
        )


def test_mixture_warm_start_keeps_each_component_family():
    x = np.r_[DATA["normal"] - 3.0, DATA["normal"] + 3.0]
    seed = Distribution().fit(
        x,
        n_components=2,
        basis=BasisSpec(lower_tail=Exponential(2.0), upper_tail=Exponential(2.0)),
        rng=0,
    )
    d = Distribution().fit(x + 0.1, init_from=seed)
    assert d.n_components == 2
    for comp in d.components:
        assert comp._basis.parent.request.upper_tail.value == pytest.approx(2.0)
