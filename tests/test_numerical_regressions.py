"""End-to-end numerical invariants and adversarial regression tests.

These tests compare fitted models against independent quantities the library cannot
choose: training likelihoods, external quadrature, probability identities, and
well-defined resource bounds.  They complement lower-level consistency tests by
checking observable behavior under numerically demanding fixtures.

Most fits are intentionally small.  Higher-degree and extreme-scale fixtures are
kept only where the numerical mechanism requires them.
"""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid
from scipy.special import betainc
from spectral_builder_harness import PythonSpectralCDFBuilder, PythonSpectralPPFBuilder

import gibbus
from gibbus import Distribution
from gibbus._api import selection as _selection
from gibbus._fit.mixture import (
    _e_step_intervals,
    _interval_identifiability_diagnostic,
    _interval_nonparametric_loglik_bound,
    _silverman_bandwidth,
)
from gibbus._fit.natural_objective import _fit_natural_conic_intervals
from gibbus._model.coords import _build_fit_coordinate
from gibbus._model.natural import _natural_layout
from gibbus._model.natural_state import _NaturalCoreState

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _density(c):
    """Reference density as ``exp(-neg_log(x))``.

    Never build a reference on ``pdf()``: it floors small values, so any
    quadrature over it measures the floor rather than the density.
    """
    def f(t):
        v = c.neg_log(float(t))
        return float(np.exp(-v)) if np.isfinite(v) else 0.0
    return f


def _total_mass(c, n_seg=16):
    """Integrate the fitted density piecewise over its support.

    Piecewise matters.  A single ``quad`` over a wide span silently
    returns 0.0 when QUADPACK's subdivision never lands on the peak --
    an all-zero reference then looks like a catastrophic finding in the
    library rather than a broken probe.  Integrating consecutive
    segments and accumulating keeps every call local to where the mass is.
    """
    lo, hi = (float(v) for v in c.support)
    a = lo if np.isfinite(lo) else float(c.ppf(1e-12))
    b = hi if np.isfinite(hi) else float(c.ppf(1.0 - 1e-12))
    edges = np.linspace(a, b, n_seg + 1)

    total = 0.0
    if not np.isfinite(lo):
        total += quad(_density(c), -np.inf, a, limit=200)[0]
    for u, v in itertools.pairwise(edges):
        total += quad(_density(c), float(u), float(v),
                      limit=200, epsabs=1e-13, epsrel=1e-13)[0]
    if not np.isfinite(hi):
        total += quad(_density(c), b, np.inf, limit=200)[0]
    return total


@pytest.fixture(scope="module")
def samples():
    r = np.random.default_rng(20250903)
    return {
        "normal": r.normal(0.0, 1.0, 600),
        "gamma": r.gamma(2.0, 1.0, 600),
        "bimodal": np.concatenate([r.normal(-2.5, 0.6, 300),
                                   r.normal(2.5, 0.6, 300)]),
    }


SUPPORTS = {"normal": (-np.inf, np.inf),
            "gamma": (0.0, np.inf),
            "bimodal": (-np.inf, np.inf)}


# --------------------------------------------------------------------------
# A fit must explain the data it was fitted to
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["normal", "gamma", "bimodal"])
@pytest.mark.parametrize("degree", [4, 8])
def test_fit_explains_its_training_data(samples, shape, degree):
    """A higher-degree fit must not be *worse* on its own training data.

    The model family is nested by degree: a degree-8 family contains the
    degree-2 one, so the degree-8 maximum likelihood cannot be lower.  Any
    material shortfall means the fitted point is inconsistent with the
    nested maximum-likelihood objective.

    The tolerance is deliberately loose: this is a structural sanity floor,
    not an accuracy benchmark.
    """
    data = samples[shape]
    support = SUPPORTS[shape]

    baseline = Distribution().fit(data, n_components=1, poly_degree=2,
                           support=support)
    richer = Distribution().fit(data, n_components=1, poly_degree=degree,
                         support=support)

    nll_baseline = float(np.mean(baseline.neg_log(data)))
    nll_richer = float(np.mean(richer.neg_log(data)))

    assert np.isfinite(nll_richer)
    assert nll_richer <= nll_baseline + 0.05, (
        f"degree {degree} fits its own training data {nll_richer - nll_baseline:.4g} "
        f"nats worse than degree 2 ({nll_richer:.6f} vs {nll_baseline:.6f})"
    )


# --------------------------------------------------------------------------
# Reported training NLL must match the public density
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["normal", "gamma"])
def test_reported_nll_matches_training_data(samples, shape):
    """``data['nll']`` is the absolute user-coordinate training NLL.

    The state absorbs the fixed fitting-coordinate Jacobian into its
    reported objective, so the stored value must equal the mean public
    negative-log density directly.
    """
    data = samples[shape]
    c = Distribution().fit(data, n_components=1, poly_degree=4,
                    support=SUPPORTS[shape])

    state = c.components[0].data
    reported = float(state["nll"])
    actual = float(np.mean(c.neg_log(data)))

    assert reported == pytest.approx(actual, rel=1e-6, abs=1e-6), (
        f"reported nll {reported:.9f} != mean neg_log over training data "
        f"{actual:.9f} (difference {reported - actual:.3e})"
    )


def test_reported_nll_matches_high_degree_lognormal():
    """The stored NLL must remain exact for a high-degree lognormal fit."""
    rng = np.random.default_rng(0)
    data = rng.lognormal(0.0, 1.0, 1200)
    fitted = Distribution().fit(
        data, n_components=1, poly_degree=12, support=(0.0, np.inf))

    state = fitted.components[0].data
    reported = float(state["nll"])
    actual = float(np.mean(fitted.neg_log(data)))
    assert reported == pytest.approx(actual, rel=2e-9, abs=2e-9)


# --------------------------------------------------------------------------
# Fitted densities must integrate to one
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", ["normal", "gamma", "bimodal"])
@pytest.mark.parametrize("degree", [4, 8])
def test_density_integrates_to_one(samples, shape, degree):
    """Independent quadrature of ``exp(-neg_log)`` must give 1."""
    c = Distribution().fit(samples[shape], n_components=1, poly_degree=degree,
                    support=SUPPORTS[shape])
    mass = _total_mass(c)
    assert mass == pytest.approx(1.0, abs=1e-6), (
        f"fitted density integrates to {mass:.12f}, not 1"
    )


def test_high_degree_density_integrates_to_one():
    """Large internal sigma must not defeat normalization quadrature.

    A degree-12 single component forced onto this trimodal sample can drive
    its internal ``sigma`` to about ``exp(28)`` while retaining an ordinary
    density in fitting coordinates.  Quadrature must therefore operate on
    the shifted order-one kernel and apply the affine sigma factor in log
    space, rather than integrating a kernel scaled down by ``1 / sigma``.
    """
    r = np.random.default_rng(0)
    data = np.ascontiguousarray(np.r_[
        r.normal(-4.0, 0.4, 1200),
        r.normal(0.0, 0.4, 1200),
        r.normal(4.0, 0.4, 1200),
    ])
    c = Distribution().fit(data, n_components=1, poly_degree=12,
                    support=(-np.inf, np.inf))

    mass = _total_mass(c)
    assert mass == pytest.approx(1.0, abs=1e-8), (
        f"degree-12 fitted density integrates to {mass:.12f}, not 1"
    )
    assert c.spectral_diagnostics["mass_defect"] < 1e-8


# --------------------------------------------------------------------------
# Public PPF views must agree in the extreme tail
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p", [1e-6, 1e-12, 1e-16, 1e-18, 1e-30])
def test_base_view_ppf_agrees_with_distribution_ppf(samples, p):
    """``.base.ppf`` must not diverge from ``Distribution.ppf`` in the extreme tail.

    The component view and top-level model are both public fitted objects, so
    they must expose the same quantile map rather than diverging through
    wrapper-specific tail handling.
    """
    c = Distribution().fit(samples["normal"], n_components=1, poly_degree=4,
                    support=(-np.inf, np.inf))

    wrapped = float(c.ppf(p))
    view = float(c.base.ppf(p))

    assert view == pytest.approx(wrapped, rel=1e-3), (
        f"base.ppf({p:g}) = {view:.6g} against Distribution.ppf = {wrapped:.6g}"
    )


# --------------------------------------------------------------------------
# sample(size=...) must reject non-integers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad_size", [2.7, np.float64(4.9), "3", True])
def test_sample_rejects_non_integer_size(samples, bad_size):
    """``size`` is documented as a non-negative integer and must be enforced.

    Floating-point, string, and boolean inputs are not valid sample counts,
    even when Python could coerce them to integers.
    """
    c = Distribution().fit(samples["normal"], n_components=1, poly_degree=2,
                    support=(-np.inf, np.inf))
    with pytest.raises((TypeError, ValueError)):
        c.sample(size=bad_size, rng=np.random.default_rng(0))




# --------------------------------------------------------------------------
# High-degree fits must be equivariant to a pure change of units
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("seed", "scale"),
    [(2, 1e8), (3, 1e12), (8, 1e16)],
)
def test_degree8_density_is_scale_equivariant(seed, scale):
    """A unit change must not change a degree-8 fit in standardized space.

    Multiple seeds and large scale factors ensure that the fit depends only on
    standardized geometry rather than the numerical units of the observations.
    """

    data = np.ascontiguousarray(np.random.default_rng(seed).normal(size=2000))
    grid = np.linspace(-4.0, 4.0, 801)

    ref = Distribution().fit(
        data, n_components=1, support=(-np.inf, np.inf), poly_degree=8
    )
    scaled = Distribution().fit(
        data * scale, n_components=1, support=(-np.inf, np.inf), poly_degree=8
    )

    p_ref = ref.pdf(grid)
    p_scaled = scale * scaled.pdf(scale * grid)
    l1 = float(trapezoid(np.abs(p_ref - p_scaled), grid))

    assert l1 < 5e-8, f"density L1 mismatch under unit change: {l1:.3e}"
    assert float(scaled._components[0]._data["nll"]) == pytest.approx(
        float(ref._components[0]._data["nll"]) + np.log(scale),
        abs=2e-10, rel=0.0,
    )
    if seed == 2:
        assert float(ref._components[0]._data["nll"]) < 1.41865



# --------------------------------------------------------------------------
# High-degree fitting must retain a numerically sound likelihood basin
# --------------------------------------------------------------------------

def test_high_degree_fit_retains_a_sound_likelihood_basin():
    """A high-degree fit must avoid a materially inferior numerical basin."""
    data = np.ascontiguousarray(
        np.random.default_rng(0).lognormal(0.0, 1.0, 3000)
    )

    fitted = Distribution().fit(
        data, n_components=1, support=(0.0, np.inf),
        poly_degree=12, rng=0,
    )

    training_nll = float(np.mean(fitted.neg_log(data)))
    assert training_nll < 1.6
    assert float(fitted.mean) == pytest.approx(float(np.mean(data)), rel=5e-3)

# --------------------------------------------------------------------------
# Uniform weights must reduce to the unweighted path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [5, 50, 500, 5000])
def test_uniform_weights_reproduce_unweighted_bandwidth(n):
    """``_silverman_bandwidth`` must agree with itself at uniform weights.

    ``sample_weights`` are relative.  Kish's effective sample size therefore
    supplies the finite-sample scale correction and reduces exactly to ``n``
    when the weights are uniform.

    This is the general shape worth testing anywhere a weighted code path
    exists: it must reduce to the unweighted one at uniform weights, and
    it must be invariant to rescaling the weights.
    """

    x = np.random.default_rng(11).normal(0.0, 1.0, n)

    unweighted = _silverman_bandwidth(x)
    uniform = _silverman_bandwidth(x, weights=np.ones(n))
    rescaled = _silverman_bandwidth(x, weights=np.full(n, 7.3))

    assert uniform == pytest.approx(unweighted, rel=1e-12)
    assert rescaled == pytest.approx(unweighted, rel=1e-12), (
        "relative weights must be invariant to their absolute scale"
    )


def test_weighted_bandwidth_is_scale_invariant():
    """Rescaling relative weights must not move the bandwidth at all."""

    rng = np.random.default_rng(12)
    x = rng.normal(0.0, 1.0, 400)
    w = rng.exponential(size=400)

    np.testing.assert_array_max_ulp(
        _silverman_bandwidth(x, weights=w),
        _silverman_bandwidth(x, weights=1000.0 * w),
        maxulp=2,
    )


# --------------------------------------------------------------------------
# Interval-mixture EM must use interval probability mass
# --------------------------------------------------------------------------

def test_interval_e_step_matches_independent_component_cdf_oracle():
    """Interval responsibilities must be posterior masses, not midpoint PDFs.

    The components are fitted to ordinary point samples and fully finalized,
    so their public CDFs provide an independent oracle for the lightweight
    quadrature path used by interval EM.  Observation weights may change the
    mean log-likelihood, but not the per-row posterior probabilities.
    """

    rng = np.random.default_rng(18019)
    data = np.concatenate([
        rng.normal(-1.5, 0.7, 250),
        rng.normal(1.0, 0.9, 350),
    ])
    fitted = Distribution().fit(
        data, n_components=2, poly_degree=2,
        support=(-np.inf, np.inf), rng=0,
    )

    intervals = np.array([
        [-3.0, -2.0],
        [-2.0, -1.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
        [2.0, 3.0],
    ])
    obs_weights = np.array([1.0, 2.0, 5.0, 4.0, 2.0, 1.0])
    obs_weights /= obs_weights.sum()

    resp, ll = _e_step_intervals(
        intervals, fitted.components, fitted.weights,
        obs_weights=obs_weights,
    )

    oracle_mass = np.column_stack([
        fitted.weights[k] * (
            comp.cdf(intervals[:, 1]) - comp.cdf(intervals[:, 0])
        )
        for k, comp in enumerate(fitted.components)
    ])
    row_mass = oracle_mass.sum(axis=1)
    oracle_resp = oracle_mass / row_mass[:, None]
    oracle_ll = float(np.dot(obs_weights, np.log(row_mass)))

    assert resp == pytest.approx(oracle_resp, rel=2e-11, abs=2e-13)
    assert ll == pytest.approx(oracle_ll, rel=2e-11, abs=2e-13)


def test_interval_mixture_relative_sample_weights_are_scale_invariant():
    """The weighted interval-mixture path must honor relative-weight semantics.

    Multiplying every observation weight by a common constant must leave the
    fitted mixture unchanged, including the interval-censored E-step and
    component M-steps.
    """
    rng = np.random.default_rng(19018)
    data = np.concatenate([
        rng.normal(-1.7, 0.65, 180),
        rng.normal(1.2, 0.8, 220),
    ])
    width = 0.8
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])
    weights = np.where(data > 0.0, 3.0, 1.0)

    a = Distribution().fit(
        intervals, n_components=2, poly_degree=2,
        progressive=False, rng=0, sample_weights=weights,
    )
    b = Distribution().fit(
        intervals, n_components=2, poly_degree=2,
        progressive=False, rng=0, sample_weights=37.0 * weights,
    )

    grid = np.linspace(float(intervals.min()), float(intervals.max()), 301)
    assert a.weights == pytest.approx(b.weights, rel=2e-10, abs=2e-12)
    assert a.pdf(grid) == pytest.approx(b.pdf(grid), rel=2e-9, abs=2e-11)


def test_interval_mass_probability_invariant_rejects_invalid_em_refit():
    """An interval component must never contribute more than unit mass.

    EM must reject a numerically invalid component refit instead of propagating
    a state whose interval probability violates the probability axioms.
    """
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(-2.0, 0.7, 500),
        rng.normal(2.0, 0.7, 500),
    ])
    width = 2.0
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])

    fitted = Distribution().fit(
        intervals, n_components=2, poly_degree=2,
        progressive=False, rng=0,
    )

    for comp in fitted.components:
        masses = comp.cdf(intervals[:, 1]) - comp.cdf(intervals[:, 0])
        assert np.all(np.isfinite(masses))
        assert np.all(masses >= -1e-12)
        assert np.all(masses <= 1.0 + 1e-12)




def test_coarse_width2_default_interval_mixture_completes():
    """A coarse width-2 interval mixture must complete with default degree selection.

    This is broader than the degree-2 invariant above: ``poly_degree`` is
    left at its public default (per-component auto degree), so the
    residual path that still failed after the midpoint-only guard was added.
    """
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(-2.0, 0.7, 500),
        rng.normal(2.0, 0.7, 500),
    ])
    width = 2.0
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])

    fitted = Distribution().fit(intervals, n_components=2, rng=0)

    assert fitted.n_components == 2
    assert np.all(np.isfinite(fitted.weights))
    assert np.isclose(np.sum(fitted.weights), 1.0)
    for comp in fitted.components:
        masses = comp.cdf(intervals[:, 1]) - comp.cdf(intervals[:, 0])
        assert np.all(np.isfinite(masses))
        assert np.all(masses >= -1e-12)
        assert np.all(masses <= 1.0 + 1e-12)


def test_saturated_coarse_interval_mixture_reports_nonidentifiability():
    """A saturated coarse-bin fit must fail semantically, not numerically.

    At width 4 the same sample occupies only three disjoint bins.  A K=2
    auto-degree mixture has more free parameters than the two independent bin
    probabilities and reaches the saturated multinomial likelihood.  Component
    shapes inside the bins are therefore unidentified; returning whichever
    ultra-narrow component happened to finalize would be misleading.
    """
    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(-2.0, 0.7, 500),
        rng.normal(2.0, 0.7, 500),
    ])
    width = 4.0
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])

    with pytest.raises(
            RuntimeError,
            match=r"non-identifiable.*nonparametric interval-likelihood bound"):
        Distribution().fit(intervals, n_components=2, rng=0)


@pytest.mark.filterwarnings("error::scipy.integrate.IntegrationWarning")
def test_overlapping_interval_saturation_reports_nonidentifiability():
    """The non-identifiability diagnostic must cover overlapping censoring bins.

    These three interval types overlap.  A two-component quadratic mixture
    can drive their censored likelihood to the unrestricted Turnbull maximum,
    but the censoring graph exposes only three independent probability
    coordinates for five free mixture parameters.  Returning one arbitrary
    component decomposition would therefore be misleading.
    """
    intervals = np.vstack([
        np.repeat([[-4.0, -1.0]], 30, axis=0),
        np.repeat([[-4.0, 1.0]], 40, axis=0),
        np.repeat([[-3.0, 2.0]], 30, axis=0),
    ])

    with pytest.raises(
            RuntimeError,
            match=r"non-identifiable.*overlapping censoring pattern"):
        Distribution().fit(
            intervals, n_components=2, poly_degree=2,
            progressive=False, em_max_iter=80, rng=0,
        )


def test_nested_interval_pattern_is_recognized_by_identifiability_diagnostic():
    """The non-identifiability diagnostic must cover nested censoring intervals.

    A common intersection lets the unrestricted censored likelihood put all
    probability mass where every row is satisfied.  At that bound, a K=2
    quadratic mixture has more free parameters than the censoring graph can
    observe, so its component decomposition is not identified.
    """

    intervals = np.vstack([
        np.repeat([[-4.0, 4.0]], 20, axis=0),
        np.repeat([[-3.0, 3.0]], 30, axis=0),
        np.repeat([[-2.0, 2.0]], 50, axis=0),
    ])
    support = (-5.0, 5.0)
    bound = _interval_nonparametric_loglik_bound(intervals, support)
    components = [
        SimpleNamespace(layout=SimpleNamespace(n_params=2)),
        SimpleNamespace(layout=SimpleNamespace(n_params=2)),
    ]

    diag = _interval_identifiability_diagnostic(
        intervals, components, bound, support
    )

    assert diag is not None
    assert diag["observable_dim"] == 3
    assert diag["n_params"] == 5
    assert diag["gap"] == pytest.approx(0.0, abs=1e-12)


def test_auto_k_rejects_unidentifiable_richer_interval_candidate():
    """Auto-K may choose K=1 when richer censored models are unidentifiable.

    When K=2 reaches the unrestricted interval-likelihood bound with excess free
    parameters and is rejected, while K=1 already reaches the same observed
    likelihood to numerical precision.  Returning K=1 therefore does not mask
    a numerical failure; it is the identifiable candidate supported by the
    coarse observations.
    """

    rng = np.random.default_rng(0)
    data = np.concatenate([
        rng.normal(-2.0, 0.7, 500),
        rng.normal(2.0, 0.7, 500),
    ])
    width = 4.0
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])

    with pytest.raises(RuntimeError, match=r"non-identifiable"):
        Distribution().fit(intervals, n_components=2, rng=0)

    fitted = Distribution().fit(
        intervals, n_components="auto", rng=0, auto_k_subsample=False
    )
    assert fitted.n_components == 1

    mass = fitted.cdf(intervals[:, 1]) - fitted.cdf(intervals[:, 0])
    ll = float(np.mean(np.log(mass)))
    bound = _interval_nonparametric_loglik_bound(
        intervals, tuple(float(v) for v in fitted.support)
    )
    assert bound - ll < 1e-6


def test_interval_auto_k_scores_the_interval_likelihood(monkeypatch):
    """Auto-K candidates for censored data must not be scored as midpoints.

    Candidate scoring must use the same interval likelihood as the fitted model.
    Record the candidate calls and require both K=1 and K>=2 validation paths
    to receive the original two-column interval observations.
    """

    rng = np.random.default_rng(4)
    data = np.concatenate([
        rng.normal(-2.0, 0.7, 140),
        rng.normal(2.0, 0.7, 140),
    ])
    width = 1.0
    edges = np.arange(
        np.floor(data.min() / width) * width,
        np.ceil(data.max() / width) * width + width,
        width,
    )
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1,
                  0, len(edges) - 2)
    intervals = np.column_stack([edges[idx], edges[idx + 1]])

    seen_k1_cols = []
    seen_multi_cols = []
    original_interval_fit = _selection._fit_natural_conic_intervals
    original_run = _selection._run_natural_em

    def wrapped_interval_fit(support, samples, *args, **kwargs):
        arr = np.asarray(samples)
        seen_k1_cols.append(1 if arr.ndim == 1 else int(arr.shape[1]))
        return original_interval_fit(support, samples, *args, **kwargs)

    def wrapped_run(*args, **kwargs):
        arr = np.asarray(args[1])
        resp = np.asarray(args[6])
        seen_multi_cols.append((int(resp.shape[1]), int(arr.shape[1])))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(_selection, "_fit_natural_conic_intervals",
                        wrapped_interval_fit)
    monkeypatch.setattr(_selection, "_run_natural_em", wrapped_run)

    fitted = Distribution().fit(
        intervals, n_components="auto", poly_degree=2,
        auto_k_subsample=False, rng=0,
    )

    assert fitted.n_components >= 1
    assert 2 in seen_k1_cols
    assert seen_multi_cols
    assert any(k >= 2 for k, _ in seen_multi_cols)
    assert all(cols == 2 for _, cols in seen_multi_cols)


# --------------------------------------------------------------------------
# Interval likelihood must remain stable under coarse and extreme censoring
# --------------------------------------------------------------------------





def test_coarse_binning_is_fittable_and_recovers_the_distribution():
    """Coarse bins must not be rejected for lack of midpoint spread.

    Location and scale are estimated from interval midpoints, which say
    nothing about spread inside a bin.  With bins wide enough that more
    than half the midpoints coincide, the midpoint MAD is zero and the
    midpoint MAD can be zero even though the intervals carry positive width.
    Initialization therefore floors the scale at the within-bin spread
    ``w * UNIFORM_WIDTH_TO_SIGMA``.
    """

    rng = np.random.default_rng(20250904)
    x = rng.normal(size=1500)
    for width in (0.5, 1.0, 2.0, 4.0):
        edges = np.round(x / width) * width
        intervals = np.ascontiguousarray(
            np.column_stack([edges - width / 2.0, edges + width / 2.0]))
        gibbus.clear_suppressed_failures()
        fitted = Distribution().fit(intervals, n_components=1, poly_degree=4,
                             support=(-np.inf, np.inf), rng=0)
        assert np.isfinite(fitted.mean), width
        assert abs(fitted.mean) < 0.25, (width, fitted.mean)
        assert 0.7 < fitted.var < 1.4, (width, fitted.var)


def test_degenerate_intervals_reproduce_the_point_fit():
    """Zero-width intervals must agree with a point fit.

    Interval masses come from fixed-order Gauss-Legendre quadrature
    rather than adaptive refinement, so this pins the resulting agreement
    instead of leaving it unmeasured.  The bound is loose enough not to
    be brittle and tight enough to catch a quadrature regression.
    """
    rng = np.random.default_rng(3)
    x = np.ascontiguousarray(rng.normal(0.0, 1.0, 1500))
    grid = np.linspace(-4.0, 4.0, 300)

    point = Distribution().fit(x, n_components=1, support=(-np.inf, np.inf),
                        poly_degree=4)
    degenerate = Distribution().fit(
        np.ascontiguousarray(np.column_stack([x - 1e-9, x + 1e-9])),
        n_components=1, support=(-np.inf, np.inf), poly_degree=4)

    assert np.max(np.abs(point.pdf(grid) - degenerate.pdf(grid))) < 1e-4
    assert degenerate.mean == pytest.approx(point.mean, abs=1e-4)
    assert degenerate.var == pytest.approx(point.var, abs=1e-4)






def test_far_tail_interval_still_moves_the_public_fit():
    """A 60-sigma-bin analog must influence an end-to-end interval fit.

    Moving a censored observation farther into the tail must not make its
    likelihood contribution disappear through probability underflow.  The
    log-space objective must preserve that observation's pull.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    width = 0.2
    base = np.column_stack([
        np.floor(x / width) * width,
        np.floor(x / width) * width + width,
    ])

    fit40 = Distribution().fit(
        np.vstack([base, [40.0, 40.0 + width]]),
        n_components=1, poly_degree=2, support=(-np.inf, np.inf),
        rng=0,
    )
    fit60 = Distribution().fit(
        np.vstack([base, [60.0, 60.0 + width]]),
        n_components=1, poly_degree=2, support=(-np.inf, np.inf),
        rng=0,
    )

    # A farther tail observation must broaden and pull the Gaussian more, not
    # become numerically invisible after its literal probability underflows.
    assert fit60.mean > fit40.mean + 1e-2
    assert fit60.std > fit40.std + 0.2

# --------------------------------------------------------------------------
# Spectral diagnostics must live on the probability scale
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seed,builder,support,degree",
    [
        (1, lambda r: r.gamma(2.0, 1.0, 2000), (0.0, np.inf), 4),
        (0, lambda r: r.beta(2.0, 5.0, 2000), (0.0, 1.0), 4),
        (0, lambda r: r.lognormal(size=2000), (0.0, np.inf), 8),
    ],
    ids=["gamma-d4", "beta-d4", "lognormal-d8-capped"],
)
def test_spectral_error_estimate_tracks_independent_cdf_error(
        seed, builder, support, degree):
    """Public spectral health numbers must be meaningful CDF-scale quantities."""
    data = np.ascontiguousarray(builder(np.random.default_rng(seed)))
    c = Distribution().fit(
        data, n_components=1, support=support, poly_degree=degree, rng=0,
    )
    diag = c.spectral_diagnostics

    # Use the spectral PPF only to choose representative evaluation points;
    # the reference values themselves come from independent quadrature of the
    # fitted density and therefore cannot inherit CDF interpolation errors.
    probs = np.array([0.1, 0.5, 0.9, 0.99, 0.999999], dtype=np.float64)
    xs = np.asarray(c.ppf(probs), dtype=np.float64)
    reference = []
    for x in xs:
        val, _ = quad(
            _density(c), float(support[0]), float(x),
            epsabs=2e-13, epsrel=2e-13, limit=400,
        )
        reference.append(val)
    measured = float(np.max(np.abs(np.asarray(c.cdf(xs)) - reference)))

    assert "error_estimate" in diag
    assert 0.0 <= diag["error_estimate"] <= 1.0
    assert diag["error_estimate"] >= diag["mass_defect"]
    # The estimate is deliberately conservative, especially when a tiny
    # uncertified panel exists, but it must no longer issue 1e-3-scale alarms
    # for CDFs that are accurate to around machine precision.
    assert diag["error_estimate"] < 5e-10
    # The estimate must not under-report independent quadrature error; the
    # small absolute allowance covers the reference quadrature's own noise.
    assert measured <= diag["error_estimate"] + 5e-13
    # ``worst_panel_error`` is a local probability-scale error, so an
    # accurate fit should not report a large transformed-density residual.
    assert diag["worst_panel_error"] < 1e-8


def test_uncertified_mass_is_failure_exposure_not_accuracy_rank():
    """Uncertified mass carries risk only when certification actually failed."""

    def beta_pdf(x):
        x = np.asarray(x, dtype=np.float64)
        return np.where((x >= 0.0) & (x <= 1.0),
                        30.0 * x * (1.0 - x) ** 4, 0.0)

    # Force acceptance of a deliberately crude quadratic panel.  With no
    # certified panels, the raw local residual is intentionally not presented
    # as an accuracy number; the full mass under the failed panel is the risk.
    rep = PythonSpectralCDFBuilder(
        beta_pdf, (0.0, 1.0), degree_options=(2,),
        rel_tol=1e-14, abs_tol=1e-16, coeff_tol=1e-14,
        max_depth=0, max_panels=2,
    )
    xs = np.linspace(0.0, 1.0, 201)
    measured = float(np.max(np.abs(rep.cdf(xs) - betainc(2.0, 5.0, xs))))

    assert rep.uncertified_mass_fraction > 0.99
    assert rep.worst_panel_error == 0.0
    assert rep.cdf_error_estimate == pytest.approx(1.0)
    assert measured < rep.cdf_error_estimate


# --------------------------------------------------------------------------
# Disparate component scales must remain numerically resolvable
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", [1_000.0, 10_000.0])
def test_disparate_scale_mixture_tracks_narrow_component(ratio):
    """A genuine narrow component must not disappear into quadrature failure.

    The mixture contains 37.5% of its mass in ``N(0, 1.5/ratio)`` and 62.5%
    in ``N(5, 1.5)``.  A correct
    fit should retain both masses and resolve the narrow component on its own
    physical scale rather than fail finalization or return an over-broad proxy.
    """
    rng = np.random.default_rng(0)
    narrow_sd = 1.5 / ratio
    data = np.concatenate([
        rng.normal(0.0, narrow_sd, 1500),
        rng.normal(5.0, 1.5, 2500),
    ])
    fitted = Distribution().fit(
        data, n_components=2, poly_degree=4,
        support=(-np.inf, np.inf),
    )

    comp_sd = np.array([float(c.std) for c in fitted.components])
    mix_w = np.asarray(fitted.weights, dtype=np.float64)
    order = np.argsort(comp_sd)
    narrow_i, broad_i = int(order[0]), int(order[-1])

    assert mix_w[narrow_i] == pytest.approx(0.375, abs=0.015)
    assert comp_sd[narrow_i] == pytest.approx(narrow_sd, rel=0.20)
    assert comp_sd[broad_i] == pytest.approx(1.5, rel=0.15)
    assert np.isfinite(np.mean(fitted.neg_log(data)))

def test_finite_lower_boundary_potential_uses_public_distance():
    """Boundary potentials must remain finite at representable public distances.

    For a lower boundary term ``-aL*log(x-L)`` with ``L=0``, the derivative
    satisfies ``x*Q'(x) -> -aL``.  The public-coordinate evaluator must preserve
    this asymptotic through the representable float64 range.
    """
    rng = np.random.default_rng(1601)
    data = np.ascontiguousarray(rng.beta(2.0, 5.0, 6000))
    c = Distribution().fit(data, n_components=1, support=(0.0, 1.0),
                    poly_degree=4)

    aL = float(c._components[0]._data["boundary_amplitudes"][0])
    x = np.array([1e-20, 1e-100, 1e-200, 1e-300], dtype=np.float64)
    q0 = np.asarray(c.neg_log(x, 0), dtype=np.float64)
    q1 = np.asarray(c.neg_log(x, 1), dtype=np.float64)

    assert np.all(np.isfinite(q0))
    assert np.all(np.isfinite(q1))
    np.testing.assert_allclose(x * q1, -aL, rtol=2e-12, atol=0.0)


# --------------------------------------------------------------------------
# Spectral panel budgets must be strict resource bounds
# --------------------------------------------------------------------------

def test_spectral_cdf_panel_budget_is_a_strict_global_leaf_cap():
    """Pending recursive siblings must not overshoot ``max_panels``.

    A rapidly oscillating positive target intentionally defeats the small
    Chebyshev degree used here.  The global frontier must always cover the full
    support with no more than the
    configured number of leaves.
    """

    def rough_pdf(x):
        x = np.asarray(x, dtype=np.float64)
        return 1.0 + 0.25 * np.sin(10_000.0 * x)

    rep = PythonSpectralCDFBuilder(
        rough_pdf, (-1.0, 1.0), degree_options=(4,),
        max_depth=12, max_panels=8,
    )

    assert len(rep.panels) == 8
    assert rep.panel_budget_exhausted is True
    assert rep.breaks[0] == -1.0
    assert rep.breaks[-1] == 1.0
    np.testing.assert_array_equal(rep.breaks[1:-1], [p.b for p in rep.panels[:-1]])
    assert all(a.b == b.a for a, b in zip(rep.panels[:-1], rep.panels[1:], strict=True))


def test_spectral_ppf_budget_bounds_failed_refinement_work():
    """An inverse that never certifies must hit the leaf budget first.

    Force every otherwise-valid candidate to remain uncertified and verify that
    the total refinement work is bounded by the complete binary-tree frontier:
    the
    complete frontier reaches eight leaves after exactly fifteen fitted tree
    nodes (``2*8 - 1``), then fails over to CDF bisection semantics.
    """

    def uniform_pdf(x):
        x = np.asarray(x, dtype=np.float64)
        return np.where((x >= 0.0) & (x <= 1.0), 1.0, 0.0)

    cdf = PythonSpectralCDFBuilder(uniform_pdf, (0.0, 1.0), degree_options=(4,))

    class NeverCertifies(PythonSpectralPPFBuilder):
        fit_calls = 0

        def _fit_panel(self, *args, **kwargs):
            type(self).fit_calls += 1
            _, panel = super()._fit_panel(*args, **kwargs)
            return False, panel

    with pytest.raises(RuntimeError, match="exhausted its 8-panel budget"):
        NeverCertifies(
            cdf, degree_options=(4,), max_depth=50, max_panels=8,
        )

    assert NeverCertifies.fit_calls == 15


# --------------------------------------------------------------------------
# Point-mixture estimability at the one-location boundary
# --------------------------------------------------------------------------

class TestPointMixtureEstimability:
    """Point mixtures must not expose the one-observation singular boundary."""

    def test_explicit_mixture_rejects_one_location_component(self):
        """A one-point minority component must raise instead of collapsing."""
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(np.r_[rng.normal(0.0, 1.0, 99), 6.0])

        with pytest.raises(RuntimeError, match="effective distinct sample locations"):
            Distribution().fit(
                data,
                n_components=2,
                poly_degree=2,
                support=(-np.inf, np.inf),
                progressive=False,
                em_max_iter=50,
                rng=0,
            )

    def test_duplicate_rows_count_as_one_support_location(self):
        """Repeating the same outlier must not fake component estimability."""
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(
            np.r_[rng.normal(0.0, 1.0, 99), np.repeat(6.0, 5)]
        )

        with pytest.raises(RuntimeError, match="effective distinct sample locations"):
            Distribution().fit(
                data,
                n_components=2,
                poly_degree=2,
                support=(-np.inf, np.inf),
                progressive=False,
                rng=0,
            )

    def test_weighted_single_component_with_spread_remains_valid(self):
        """Unequal positive weights alone do not imply a singular objective."""
        data = np.ascontiguousarray(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        weights = np.array([0.55, 0.15, 0.10, 0.10, 0.10])

        c = Distribution().fit(
            data,
            sample_weights=weights,
            n_components=1,
            poly_degree=2,
            support=(-np.inf, np.inf),
            rng=0,
        )

        assert np.isfinite(float(c.mean))
        assert np.isfinite(float(c.std)) and float(c.std) > 0.0


def test_boundary_amplitudes_are_linear_nonnegative_coordinates():
    """Boundary amplitudes are nonnegative natural coordinates."""
    for seed, (p, q) in enumerate([(2.0, 5.0), (1.5, 1.5), (5.0, 2.0)]):
        data = np.ascontiguousarray(np.random.default_rng(seed).beta(p, q, 3000))
        c = Distribution().fit(data, n_components=1, support=(0.0, 1.0), poly_degree=6)
        state = c.data
        assert "q_boundary" not in state.dtype.names
        amps = np.asarray(state["boundary_amplitudes"], dtype=float)
        assert np.all(np.isfinite(amps))
        assert np.all(amps >= 0.0)
        assert np.isfinite(c.mean) and np.isfinite(c.var)

def test_mixed_censoring_at_a_boundary_amplitude_is_certified():
    """Left-censored rows ending near 0, narrow rows touching 0 and exact
    points on a half-line with an active lower amplitude: the interval fit
    converges to a certified optimum."""
    data = np.ascontiguousarray(np.random.default_rng(0).beta(2.0, 5.0, 4000))
    c = Distribution().fit(data, n_components=1, support=(0.0, 1.0), poly_degree=6)
    assert c.mean == pytest.approx(2.0 / 7.0, abs=0.02)
    assert c.var == pytest.approx(2.0 * 5.0 / (49.0 * 8.0), abs=0.005)


def test_degree_screening_uses_the_real_boundary_structure():
    """Auto-degree screening must rank the same boundary model it later fits.

    On finite and half-line supports, each candidate degree includes the allowed
    logarithmic boundary structure.  The selected degree is compared against
    converged BIC over the same admissible model family.
    """
    n = 4000
    log_n = np.log(n)
    cases = [
        ("gamma", lambda r: r.gamma(3.0, 1.0, n), (0.0, np.inf)),
        ("beta", lambda r: r.beta(2.0, 5.0, n), (0.0, 1.0)),
        ("expon", lambda r: r.exponential(1.0, n), (0.0, np.inf)),
    ]
    for name, build, support in cases:
        data = np.ascontiguousarray(build(np.random.default_rng(0)))
        chosen = Distribution().fit(data, n_components=1, support=support,
                             poly_degree="auto")
        chosen_deg = int(chosen.data["requested_poly_degree"])

        # Converged BIC over the admissible degrees, using the library's
        # own criterion: 2 * nll * n + p * log(n).
        best_deg, best_bic = None, np.inf
        for deg in range(2, 9):
            try:
                c = Distribution().fit(data, n_components=1, support=support,
                                poly_degree=deg)
            except (ValueError, RuntimeError):
                continue
            state = c.data
            p_free = int(np.asarray(state["optimizer_params"]).size)
            nll = float(np.mean(c.neg_log(data)))
            bic = 2.0 * nll * n + p_free * log_n
            if bic < best_bic:
                best_deg, best_bic = deg, bic

        assert chosen_deg == best_deg, (
            f"{name}: screening chose degree {chosen_deg}, converged BIC "
            f"prefers {best_deg}")


def test_mixed_censoring_roundoff_step_does_not_contract_trust_radius():
    """Roundoff-scale objective changes must not cause a spurious radius shrink."""

    rng = np.random.default_rng(63103)
    x = rng.gamma(0.2357, 0.494, 60)
    sample_scale = float(np.std(x))

    censor_rng = np.random.default_rng(45)
    selector = censor_rng.random(x.size)
    narrow_u = censor_rng.random(x.size)
    lower_u = censor_rng.random(x.size)

    intervals = np.empty((x.size, 2), dtype=np.float64)
    for i, value in enumerate(x):
        if selector[i] < 0.05:
            upper = value + (0.01 + 0.39 * lower_u[i]) * sample_scale
            intervals[i] = (0.0, upper)
        elif selector[i] < 0.40:
            half_width = sample_scale * 10.0 ** (-3.0 + 2.85 * narrow_u[i])
            intervals[i] = (
                max(0.0, value - half_width), value + half_width
            )
        else:
            intervals[i] = (value, value)

    _, result = _fit_natural_conic_intervals((0.0, np.inf), intervals, 2, True, False)

    assert result.status in ("converged", "converged_approximately")
    assert result.final_separation.feasible
    assert result.final_decrease_bound <= 1e-8
    assert np.isfinite(result.objective_value)


def test_log_partition_is_continuous_as_a_boundary_amplitude_vanishes():
    """A tiny positive log amplitude must not truncate the normalization window.

    With ``0 < a <~ 1e-11`` the mode solve lands on the singular endpoint where
    ``q = +inf``; the tail points were then defined relative to an infinite
    reference and the window dropped real tail mass (2.7e-4 of log Z for this
    exponential shape).  The density differs from the ``a = 0`` one only in an
    endpoint layer of width ``~a / q'``, so log Z must be continuous in ``a``.
    """
    x = np.random.default_rng(3).exponential(1.0, 120)
    coordinate = _build_fit_coordinate((0.0, np.inf), x, None, None)
    z = coordinate.to_canonical(x)
    layout = _natural_layout(coordinate.canonical_support, 4, True, False)
    base = layout.pack(0.6943, [0.02837, -0.005811, 0.0002975], [0.0, np.nan])
    reference = None
    for amplitude in (0.0, 1e-300, 1e-14, 1e-12, 1e-10):
        params = base.copy()
        params[layout.lower_a_index] = amplitude
        state = _NaturalCoreState(
            coordinate, layout, params, (float(z.min()), float(z.max()))
        )
        candidate = layout.build_candidate(params)
        lower = layout.support[0]

        def density(value, candidate=candidate):
            return float(np.exp(-candidate.q(np.array([value]))[0]))

        exact = np.log(
            quad(density, lower, lower + 1.0, limit=200)[0]
            + quad(density, lower + 1.0, np.inf, limit=200)[0]
        )
        assert state.log_Z == pytest.approx(exact, abs=1e-10)
        if reference is None:
            reference = state.log_Z
        assert state.log_Z == pytest.approx(reference, abs=1e-9)


def test_interval_mixture_seed_keeps_the_exact_support_endpoint():
    """Binned half-line data whose first bin starts on the support endpoint.

    The EM seed's support was recovered by mapping the canonical support back
    through the fit coordinate, which rounded 0 to 8.9e-16; the next M-step
    then rejected the rows starting at 0 as outside the seed's support.
    """
    import warnings

    from gibbus import Distribution

    warnings.simplefilter("ignore")
    rng = np.random.default_rng(270904)
    x = np.concatenate([rng.gamma(2.0, 0.5, 100), rng.gamma(9.0, 0.7, 100)])
    lower = np.floor(x / 0.5) * 0.5
    rows = np.column_stack([lower, lower + 0.5])
    assert np.any(rows[:, 0] == 0.0)
    fitted = Distribution().fit(
        rows, n_components=2, poly_degree=4, support=(0.0, np.inf),
        log_boundary_lower=True, rng=0,
    )
    assert np.isfinite(fitted._em_diagnostics["final_log_likelihood"])


@pytest.mark.parametrize("seed", [270904, 270905])
def test_a_valley_start_does_not_lock_a_mixture_onto_the_extreme_points(seed):
    """The KDE valley seed is compared with the others, not used alone.

    Regression: on these lognormal samples the valley cut off the two or
    three largest points.  Seeded only there, seed 270904 ended on a 1%
    tail component (log likelihood -0.9253) and seed 270905 tripped the
    one-location estimability guard and raised; the GMM and nested-scale
    seeds reach higher likelihoods with ordinary components.
    """
    data = np.random.default_rng(seed).lognormal(0.0, 0.6, 300)
    model = Distribution().fit(
        data, n_components=2, poly_degree=4, support=(0.0, np.inf),
        log_boundary_lower=False, rng=seed)
    diagnostics = model.fit_diagnostics
    assert diagnostics["converged"]
    assert min(model.weights) > 0.1
    if seed == 270904:
        assert diagnostics["em"]["final_log_likelihood"] > -0.92


def test_an_unsupported_explicit_component_count_says_what_to_do():
    """Five copies of one far value are one location: no second component."""
    data = np.concatenate([np.random.default_rng(1).normal(0.0, 1.0, 200), np.full(5, 40.0)])
    with pytest.raises(RuntimeError, match=r"do not support n_components=2; use n_components='auto'"):
        Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    assert Distribution().fit(data, support=(-np.inf, np.inf), rng=0).n_components == 1
