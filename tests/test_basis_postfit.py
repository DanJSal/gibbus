"""Post-fit queries and persistence of basis fits.

Defining properties: a durable basis record recompiles to the same basis;
fitted components answer every public query as the closed form of the
family they represent (normal, exponential, truncated normal, two-normal
mixture); the spectral CDF's mass is the normalizer's ``Z``; serialization
round-trips bitwise through the cache and to quadrature accuracy through a
cacheless rebuild, and fails loudly on a record that does not reproduce its
normalization.
"""

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from gibbus import BasisSpec, Distribution, Exponential, GibbusBasisWarning
from gibbus._fit.basis_mixture import _fit_mixture_auto
from gibbus._fit.basis_selection import _fit_basis
from gibbus._model.basis import _build_basis, _grow_basis
from gibbus._postfit.basis_state import (
    _log_cumsum,
    _model_from_record,
    _model_record,
    _pack_basis_fit,
    _pack_basis_mixture,
)
from gibbus._serialization import pack_distribution_state, strip_cache

INF = math.inf


def _distribution(weights, states, metadata=None):
    return Distribution(
        pack_distribution_state(weights, "base", states, fit_metadata=metadata)
    )


@pytest.fixture(scope="module")
def normal():
    x = np.random.default_rng(0).normal(1.0, 2.0, 500)
    fit = _fit_basis(BasisSpec(bulk_degree=2, expansion="minimal"), x)
    weights, states, metadata = _pack_basis_fit(fit, 500.0)
    d = _distribution(weights, states, metadata)
    return d, stats.norm(d.mean, d.std), x


@pytest.fixture(scope="module")
def exponential():
    x = np.random.default_rng(0).exponential(2.0, 500)
    fit = _fit_basis(
        BasisSpec(support=(0.0, INF), bulk_degree=1, expansion="minimal"), x
    )
    weights, states, metadata = _pack_basis_fit(fit, 500.0)
    d = _distribution(weights, states, metadata)
    return d, float(fit.result.params[0] / fit.model.geometry.coordinate.scale)


def _models():
    rng = np.random.default_rng(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        tails = _build_basis(
            BasisSpec(
                bulk_degree=2, lower_tail=Exponential(0.5), upper_tail=Exponential(0.7)
            ),
            rng.gumbel(0.0, 1.0, 300),
        )
        bounded = _build_basis(
            BasisSpec(
                support=(0.0, 1.0), bulk_degree=3, log_lower=True, log_upper=True
            ),
            rng.beta(2.0, 5.0, 300),
        )
    grown = _grow_basis(tails)[0][1]
    return [tails, bounded, grown]


@pytest.mark.parametrize("index", range(3))
def test_model_record_recompiles_the_basis(index):
    model = _models()[index]
    again, suppressed = _model_from_record(_model_record(model))
    assert suppressed == ()
    assert [c.label for c in again.columns] == [c.label for c in model.columns]
    np.testing.assert_array_equal(again.coefficient_map, model.coefficient_map)
    assert again.request == model.request and again.geometry == model.geometry
    assert again.slot_choice == model.slot_choice


def test_normal_fit_answers_queries_as_the_normal(normal):
    d, ref, x = normal
    mu, sd = ref.mean(), ref.std()
    xs = mu + sd * np.array([-30.0, -10.0, -3.0, 0.0, 2.0, 12.0])
    np.testing.assert_allclose(d.pdf(xs), ref.pdf(xs), rtol=1e-12)
    np.testing.assert_allclose(d.logcdf(xs), ref.logcdf(xs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.logsf(xs), ref.logsf(xs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d.cdf(xs), ref.cdf(xs), rtol=1e-11, atol=1e-15)
    p = np.array([1e-300, 1e-30, 1e-12, 1e-3, 0.3, 0.5, 0.99, 1 - 1e-12])
    np.testing.assert_allclose(d.ppf(p), ref.ppf(p), rtol=0, atol=1e-11 * sd)
    q = np.array([1e-20, 1e-5])
    np.testing.assert_allclose(d.isf(q), ref.isf(q), rtol=0, atol=1e-9 * sd)
    assert d.median == pytest.approx(mu, abs=1e-13 * sd)
    assert d.mode == pytest.approx(mu, abs=1e-13 * sd)
    assert d.var == pytest.approx(x.var(), rel=1e-6)
    assert d.mean == pytest.approx(x.mean(), rel=1e-9)
    assert d.skew == pytest.approx(0.0, abs=1e-12)
    assert d.kurt == pytest.approx(3.0, rel=1e-12)
    assert d.entropy() == pytest.approx(ref.entropy(), rel=1e-12)
    np.testing.assert_allclose(d.interval(0.9), ref.interval(0.9), rtol=1e-12)
    assert d.loglik(x) == pytest.approx(ref.logpdf(x).sum(), rel=1e-13)
    comp = d._components[0]
    np.testing.assert_allclose(
        comp.neg_log(xs[2:5], 1), (xs[2:5] - mu) / sd**2, atol=1e-12
    )
    assert comp.neg_log(mu, 2) == pytest.approx(1.0 / sd**2, rel=1e-12)
    assert comp.tail_rate("upper") == math.inf
    # The numerical margin: the constant curvature of a normal.
    assert comp.log_concavity_margin() == pytest.approx(1.0 / sd**2, rel=1e-12)
    with pytest.raises(ValueError, match="up to order 4"):
        comp.neg_log(0.0, 5)


def test_log_concavity_margin_on_a_finite_support():
    """A finite end is approached in ``x``, not by ultratail inversion.

    The ``1e-300`` lower quantile of this bounded fit is not representable,
    so the margin uses the curvature minimum, which a dense independent
    grid locates.
    """
    x = stats.cosine.rvs(size=500, random_state=0)
    d = Distribution().fit(x, basis=BasisSpec(support=(-np.pi, np.pi), bulk_degree=2))
    comp = d._components[0]
    margin = comp.log_concavity_margin()
    grid = np.linspace(-np.pi, np.pi, 100_001)[1:-1]
    dense = float(np.min(comp.base.neg_log(grid, 2)))
    assert np.isfinite(margin) and margin >= 0.0
    assert margin == pytest.approx(dense, rel=1e-9)
    assert d.spectral_diagnostics["log_concavity_margin"] == margin


def test_normal_exp_space_is_the_lognormal(normal):
    d, ref, _ = normal
    mu, var = ref.mean(), ref.var()
    exp = d._components[0].exp
    assert exp.mean == pytest.approx(math.exp(mu + var / 2), rel=1e-12)
    assert exp.var == pytest.approx(
        (math.exp(var) - 1) * math.exp(2 * mu + var), rel=1e-11
    )
    assert exp.mode == pytest.approx(math.exp(mu - var), rel=1e-12)


def test_exponential_fit_has_its_rate_and_exp_moments(exponential):
    d, rate = exponential
    comp = d._components[0]
    assert comp.tail_rate("upper") == pytest.approx(rate, rel=1e-14)
    assert comp.tail_rate("lower") == math.inf
    xs = np.array([0.0, 0.5, 3.0, 40.0])
    np.testing.assert_allclose(d.sf(xs), np.exp(-rate * xs), rtol=1e-12)
    # E[e^{kX}] = rate / (rate - k) below the rate, infinite at or beyond it.
    assert math.exp(comp.exp._log_raw_moment(1)) == pytest.approx(
        rate / (rate - 1.0) if rate > 1.0 else math.inf
    )
    log_m = d._components[0].exp._log_raw_moment
    assert log_m(math.ceil(rate)) == math.inf


def test_truncation_is_the_truncated_normal(normal):
    d, ref, _ = normal
    mu, sd = ref.mean(), ref.std()
    t = d._components[0]._truncate_base(0.0, None)
    tn = stats.truncnorm(-mu / sd, np.inf, mu, sd)
    xs = np.array([0.0, 0.5, 2.0, 9.0])
    np.testing.assert_allclose(t.cdf(xs), tn.cdf(xs), rtol=1e-11, atol=1e-14)
    assert t.mean == pytest.approx(tn.mean(), rel=1e-12)
    assert t.var == pytest.approx(tn.var(), rel=1e-10)


def test_spectral_mass_is_the_normalizer_z(normal):
    d, _, _ = normal
    state = d._components[0].data
    assert float(state["cdf_mass_defect"]) <= 1e-12


@pytest.fixture(scope="module")
def mixture():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(-2, 1, 300), rng.normal(3, 1.5, 300)]
    fit, _, _ = _fit_mixture_auto(BasisSpec(bulk_degree=2, expansion="minimal"), x, 2)
    weights, states, metadata = _pack_basis_mixture(fit, 600.0)
    return _distribution(weights, states, metadata), fit, x


def test_mixture_queries_are_weighted_component_sums(mixture):
    d, fit, x = mixture
    w = d.weights
    refs = [stats.norm(c.mean, c.std) for c in d._components]
    xs = np.array([-12.0, -5.0, -2.0, 0.0, 1.0, 3.0, 8.0, 20.0])
    np.testing.assert_allclose(
        d.pdf(xs), sum(wk * r.pdf(xs) for wk, r in zip(w, refs)), rtol=1e-12
    )
    cdf = sum(wk * r.cdf(xs) for wk, r in zip(w, refs))
    np.testing.assert_allclose(d.cdf(xs), cdf, rtol=1e-10, atol=1e-15)
    p = np.array([1e-10, 0.01, 0.3, 0.5, 0.9, 1 - 1e-8])
    back = sum(wk * r.cdf(d.ppf(p)) for wk, r in zip(w, refs))
    np.testing.assert_allclose(back, p, rtol=1e-10)
    assert d.mean == pytest.approx(
        sum(wk * r.mean() for wk, r in zip(w, refs)), rel=1e-12
    )
    assert d.loglik(x) / x.size == pytest.approx(fit.log_likelihood, rel=1e-13)
    diag = d.fit_diagnostics
    assert diag["n_face_parameters"] == fit.n_face_parameters == diag["n_parameters"]
    assert diag["provenance"] == "fitted"


def test_round_trip_through_the_cache_is_bitwise(mixture, tmp_path):
    d, _, _ = mixture
    path = tmp_path / "model.npy"
    np.save(path, d.data)
    loaded = Distribution(np.load(path, allow_pickle=False))
    assert int(d.data["format_version"]) == 1
    xs = np.linspace(-8.0, 10.0, 37)
    np.testing.assert_array_equal(loaded.pdf(xs), d.pdf(xs))
    np.testing.assert_array_equal(loaded.cdf(xs), d.cdf(xs))
    assert loaded.fit_diagnostics["components"] == d.fit_diagnostics["components"]


@pytest.mark.parametrize("truncate", [False, True])
def test_cacheless_rebuild_reproduces_the_model(normal, truncate):
    d, _, _ = normal
    if truncate:
        comp = d._components[0]._truncate_base(-1.0, 4.0)
        d = _distribution(np.ones(1), [comp.data])
    rebuilt = Distribution(strip_cache(d.data))
    xs = np.linspace(-3.0, 5.0, 17)
    np.testing.assert_allclose(rebuilt.pdf(xs), d.pdf(xs), rtol=1e-14, atol=0)
    np.testing.assert_allclose(rebuilt.cdf(xs), d.cdf(xs), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(rebuilt.support, d.support)


def test_record_that_does_not_reproduce_log_z_fails_loudly(normal):
    d, _, _ = normal
    state = strip_cache(d.data)
    state["model"]["log_z"] = state["model"]["log_z"] + 1e-6
    with pytest.raises(ValueError, match="does not reproduce its normalization"):
        Distribution(state)


def test_cumulative_end_masses_match_independent_rows(normal):
    d, _, _ = normal
    basis = d._components[0]._basis
    end = 1.25
    x = np.array([-30.0, -9.0, -9.0, 0.0, end, 1.5, 4.0, 40.0, np.nan, 1.5, -np.inf])
    got = basis.log_masses_from(end, x)
    want = basis.log_mass(np.minimum(x, end), np.maximum(x, end))
    want[x == end] = -np.inf
    assert np.array_equal(np.isnan(got), np.isnan(x))
    ok = ~np.isnan(x)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-14, atol=1e-11)
    # From an infinite end: the CDF itself.
    lo = basis.log_masses_from(-np.inf, x[ok])
    np.testing.assert_allclose(
        lo, basis.log_mass(np.full(ok.sum(), -np.inf), x[ok]), rtol=1e-14, atol=1e-11
    )


def test_log_cumsum_keeps_relative_accuracy_over_many_terms():
    n = 200_000
    flat = _log_cumsum(np.full(n, -700.0))
    assert flat[-1] == pytest.approx(math.log(n) - 700.0, rel=0, abs=1e-12)
    rising = np.concatenate([[-np.inf], np.linspace(-1400.0, 50.0, n)])
    out = _log_cumsum(rising)
    assert out[0] == -np.inf
    exact = 50.0 + math.log(math.fsum(np.exp(rising[1:] - 50.0)))
    assert out[-1] == pytest.approx(exact, rel=0, abs=1e-13)


def test_tiny_bounded_spectral_density_preserves_endpoint_log_and_mass():
    """The spectral evaluator must not round canonical nodes to physical ulps."""
    from gibbus._model._basis_kernels import BasisKernel

    from gibbus._model.basis_normalizer import _basis_state, _NormalizerContext
    from gibbus._spectral.cdf import SpectralCDF, density_spec
    from gibbus._spectral.config import _SpectralCDFOptions

    lower = 1.0
    upper = lower + 256 * np.spacing(lower)
    values = lower + np.arange(1, 256, 4) * np.spacing(lower)
    model = _build_basis(
        BasisSpec(support=(lower, upper), bulk_degree=2, log_lower=True), values
    )
    theta = np.zeros(model.n_coefficients)
    theta[[c.label for c in model.columns].index("logL")] = 0.5
    state = _basis_state(_NormalizerContext(model), theta)
    c = model.geometry.center
    h = model.geometry.scale
    zs = ((lower - c) / h, (upper - c) / h)
    density = density_spec(
        [
            (
                BasisKernel(model),
                theta,
                zs[0],
                zs[1],
                state.log_z - math.log(h),
                0.0,
                1.0,
                1.0,
                1.0,
                -INF,
                INF,
            )
        ],
        view=False,
    )
    fraction = np.array([0.25, 0.5, 0.75])
    z = zs[0] + fraction * (zs[1] - zs[0])
    ref = 1.5 * np.sqrt(fraction) / (zs[1] - zs[0])
    np.testing.assert_allclose(density.pdf(z), ref, rtol=2e-12)
    cdf = SpectralCDF(
        zs,
        density=density,
        mode=zs[1],
        std=(zs[1] - zs[0]) / 4,
        initial_breaks=None,
        map_scale=None,
        config=_SpectralCDFOptions(),
    )
    assert abs(float(cdf.total_mass) - 1.0) < 1e-9


def test_tail_rates_follow_orientation_and_presentation():
    # An upper-bounded linear tail is the reflection of a lower-bounded one:
    # both report the exponential MLE rate on their infinite side, and a
    # presentation scale divides it.
    y = np.random.default_rng(1).exponential(0.5, 3000)
    mle = 1.0 / float(np.mean(y))
    upper = Distribution().fit(
        -y, basis=BasisSpec(support=(-math.inf, 0.0), bulk_degree=1, log_upper=False)
    )
    lower = Distribution().fit(
        y, basis=BasisSpec(support=(0.0, math.inf), bulk_degree=1, log_lower=False)
    )
    assert upper.tail_rate("lower") == pytest.approx(mle, rel=1e-8)
    assert upper.tail_rate("upper") == math.inf
    assert lower.tail_rate("upper") == pytest.approx(
        upper.tail_rate("lower"), rel=1e-10
    )
    assert lower.tail_rate("lower") == math.inf
    moved = upper.transform(mu=1.0, sigma=4.0, pullback=False, inplace=False)
    assert moved.tail_rate("lower") == pytest.approx(mle / 4.0, rel=1e-8)
