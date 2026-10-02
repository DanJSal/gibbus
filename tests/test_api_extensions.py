"""Defining-property tests for the post-0.1 public API extensions."""

import copy
import pickle
import warnings

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning
from scipy.stats import norm

from gibbus import (
    Distribution,
    _defaults,
    clear_suppressed_failures,
    suppressed_failures,
)
from gibbus._defaults import SF_HANDOVER_P
from gibbus._postfit import scoring as scoring_module
from gibbus._postfit.logspace import log1mexp, log_diff_exp, log_mass_between
from gibbus._postfit.survival import _quad_with_ledger
from gibbus._postfit.survival import mean_residual_life as _mean_residual_life


@pytest.fixture(scope="module")
def gaussian_fit():
    rng = np.random.default_rng(20260914)
    return Distribution().fit(
        rng.normal(loc=0.2, scale=1.1, size=1200),
        n_components=1,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )


@pytest.fixture(scope="module")
def mixture_fit():
    rng = np.random.default_rng(314159)
    sample = np.concatenate([
        rng.normal(-2.0, 0.55, 450),
        rng.normal(2.2, 0.75, 450),
    ])
    return Distribution().fit(
        sample,
        n_components=2,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )


def test_logspace_primitives_edge_cases():
    assert log1mexp(0.0) == -np.inf
    assert np.isclose(log1mexp(np.log(0.25)), np.log(0.75))
    with pytest.raises(ValueError):
        log1mexp(1e-12)

    assert log_diff_exp(-3.0, -3.0) == -np.inf
    assert log_diff_exp(-np.inf, -np.inf) == -np.inf
    with pytest.raises(ValueError):
        log_diff_exp(-4.0, -3.0)

    lo, hi = 0.2, 0.3
    lm = log_mass_between(np.log(lo), np.log(hi), np.log1p(-lo), np.log1p(-hi))
    assert np.isclose(lm, np.log(hi - lo), rtol=0.0, atol=1e-14)

    # The inaccurate CDF side may invert by roundoff in the deep upper tail.
    # Only the selected SF branch is allowed to be evaluated.
    lm = log_mass_between(-1e-17, -2e-17, np.log(3e-16), np.log(1e-16))
    assert np.isclose(lm, np.log(2e-16), rtol=0.0, atol=1e-14)

    lc_lo = np.array([np.log(0.2), -1e-17])
    lc_hi = np.array([np.log(0.3), -2e-17])
    ls_lo = np.array([np.log(0.8), np.log(3e-16)])
    ls_hi = np.array([np.log(0.7), np.log(1e-16)])
    got = log_mass_between(lc_lo, lc_hi, ls_lo, ls_hi)
    np.testing.assert_allclose(got, [np.log(0.1), np.log(2e-16)], atol=1e-14)


def test_log_probability_and_extreme_inverse_round_trip(gaussian_fit):
    c = gaussian_fit
    z = np.array([0.0, 5.0, 8.0, 12.0, 30.0])
    x = c.mean + c.std * z
    np.testing.assert_allclose(c.logsf(x), norm.logsf(z), rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(
        np.logaddexp(c.logcdf(x[:-1]), c.logsf(x[:-1])),
        0.0,
        rtol=0.0,
        atol=2e-14,
    )

    p = np.array([0.5, 1e-5, 1e-20, 1e-100, 1e-300])
    q = c.isf(p)
    np.testing.assert_allclose(c.logsf(q), np.log(p), rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(c.logisf(np.log(p)), q, rtol=1e-12, atol=1e-12)


def test_log_quantile_deep_tail_uses_scipy_ndtri_exp():
    logp = np.array([-10.0, -100.0, -745.0, -800.0, -2000.0])
    got = scoring_module._ndtri_from_log(logp)
    expected = np.array([
        -3.913946240531893,
        -13.888476033003888,
        -38.4819489643302,
        -39.88469483825668,
        -63.165418608783604,
    ])
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert np.all(np.isfinite(got))


def test_hazard_reliability_invariants(gaussian_fit):
    c = gaussian_fit
    grid = c.mean + c.std * np.linspace(-3.0, 4.0, 25)
    h = c.hazard(grid)
    hp = c.hazard(grid, n=1)
    mrl = c.mean_residual_life(grid)

    assert np.all(np.diff(h) >= -1e-10)
    assert np.all(hp >= -1e-10)
    assert np.all(mrl * h <= 1.0 + 1e-8)
    assert c.fit_diagnostics["hazard_is_monotone"] is True
    np.testing.assert_allclose(c.cumulative_hazard(grid), -c.logsf(grid), atol=0.0)


def test_hazard_monotonicity_numerical_failure_is_unknown(gaussian_fit, monkeypatch):
    clear_suppressed_failures()

    def fail_log_hazard(self, x):
        raise RuntimeError("synthetic diagnostic failure")

    monkeypatch.setattr(Distribution, "log_hazard", fail_log_hazard)
    if _defaults.DEBUG:
        # Debug mode makes an unexpected fallback fatal instead of recording it.
        with pytest.raises(RuntimeError, match="hazard monotonicity diagnostic"):
            _ = gaussian_fit.fit_diagnostics["hazard_is_monotone"]
        return
    assert gaussian_fit.fit_diagnostics["hazard_is_monotone"] is None
    records = suppressed_failures()
    assert records[-1]["context"] == "hazard monotonicity diagnostic"


def test_hazard_monotonicity_contract_errors_propagate(gaussian_fit, monkeypatch):
    def fail_log_hazard(self, x):
        raise ValueError("synthetic contract failure")

    monkeypatch.setattr(Distribution, "log_hazard", fail_log_hazard)
    with pytest.raises(ValueError, match="synthetic contract failure"):
        _ = gaussian_fit.fit_diagnostics


def test_hazard_monotonicity_is_unknown_when_body_check_cannot_run(
    gaussian_fit, monkeypatch
):
    monkeypatch.setattr(Distribution, "ppf", lambda self, p: -np.inf)
    monkeypatch.setattr(Distribution, "isf", lambda self, p: np.inf)
    assert gaussian_fit.fit_diagnostics["hazard_is_monotone"] is None


def test_logppf_body_and_tail_seams(gaussian_fit):
    c = gaussian_fit
    p = np.array([
        1e-10,
        SF_HANDOVER_P * 0.999,
        SF_HANDOVER_P,
        0.5,
        1.0 - SF_HANDOVER_P,
        1.0 - SF_HANDOVER_P * 0.1,
    ])
    got = c.logppf(np.log(p))
    expected = c.ppf(p)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=5e-10)
    assert np.isnan(c.logppf(np.nan))
    assert np.isnan(c.logisf(np.nan))
    assert np.isnan(c.isf(np.nan))



def test_mean_residual_life_lower_tail_mills_identity(gaussian_fit):
    c = gaussian_fit
    x = c.mean + c.std * np.array([-4.0, -6.0, -10.0])
    hazard = np.asarray(c.hazard(x), dtype=np.float64)
    got = np.asarray(c.mean_residual_life(x), dtype=np.float64)
    expected = c.mean + c.var * hazard - x

    np.testing.assert_allclose(got, expected, rtol=0.0, atol=5e-10)
    assert np.all(got >= c.mean - x - 1e-12)



def test_mixture_mean_residual_life_uses_rightmost_mode_handoff():
    rng = np.random.default_rng(44)
    sample = np.concatenate([
        rng.normal(-3.0, 0.6, 400),
        rng.normal(2.5, 0.8, 400),
    ])
    c = Distribution().fit(
        sample,
        n_components=2,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )
    grid = np.array([c.ppf(1e-8), -4.0, -2.0, 0.0, 2.0, c.ppf(0.99)])
    got = np.asarray(c.mean_residual_life(grid), dtype=np.float64)

    assert np.all(np.isfinite(got))
    assert np.all(got >= c.mean - grid - 1e-7)


def test_mean_residual_life_avoids_nested_survival_quadrature():
    calls = 0

    def potential(x, n):
        if n != 0:
            raise AssertionError("only the density potential should be requested")
        return 0.5 * float(x) ** 2 + 0.5 * np.log(2.0 * np.pi)

    def counted_logsf(x):
        nonlocal calls
        calls += 1
        return float(norm.logsf(x))

    grid = np.array([-6.0, -2.0, 0.0, 2.0, 6.0])
    got = _mean_residual_life(
        potential,
        counted_logsf,
        (-np.inf, np.inf),
        0.0,
        0.0,
        grid,
    )
    expected = norm.pdf(grid) / norm.sf(grid) - grid
    np.testing.assert_allclose(got, expected, rtol=2e-9, atol=2e-11)
    assert calls == grid.size


def test_postfit_tail_quadrature_does_not_leak_integration_warnings(gaussian_fit):
    c = gaussian_fit
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.mean_residual_life(c.mean - 6.0 * c.std)
        c.logsf(c.mean + 20.0 * c.std)
    assert not any(issubclass(item.category, IntegrationWarning) for item in caught)


def test_postfit_evaluation_never_swaps_global_warning_state(gaussian_fit, monkeypatch):
    # ``warnings.catch_warnings`` replaces process-global filters and
    # ``showwarning``.  When threads sharing a fitted model interleave its
    # enter/exit, the globals are left corrupted and every later warning in the
    # process is silently swallowed, so evaluation must not use it at all.
    entered = []

    class _Tripwire:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            entered.append(True)
            return []

        def __exit__(self, *exc_info):
            return False

    c = gaussian_fit
    monkeypatch.setattr(warnings, "catch_warnings", _Tripwire)
    c.mean_residual_life(c.mean - 6.0 * c.std)
    c.residual_entropy(c.mean)
    c.logsf(c.mean + 20.0 * c.std)
    c.logcdf(c.mean - 20.0 * c.std)
    assert not entered


def test_ledger_quadrature_reports_problems_without_emitting_warnings():
    clear_suppressed_failures()

    def run():
        return _quad_with_ledger(
            lambda t: np.sin(1.0 / t) / t, 1e-6, 1.0,
            epsabs=1e-14, epsrel=1e-14, limit=3, context="ledger quadrature test",
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if _defaults.DEBUG:
            with pytest.raises(RuntimeError, match="ledger quadrature test"):
                run()
        else:
            run()
    assert not caught
    if not _defaults.DEBUG:
        assert any(f["context"] == "ledger quadrature test" for f in suppressed_failures())


def test_tail_rate_reports_infinite_for_growing_polynomial_tail():
    rng = np.random.default_rng(731)
    x = rng.gamma(shape=3.0, scale=1.0, size=800)
    c = Distribution().fit(
        x,
        n_components=1,
        poly_degree="auto",
        support=(0.0, np.inf),
        rng=0,
    )
    assert np.isinf(c.tail_rate("upper"))


def test_deep_tail_interval_loglik_uses_represented_interval_width(gaussian_fit):
    c = gaussian_fit
    for z, relative_width in [
        (0.0, 1e-8),
        (5.0, 1e-8),
        (8.0, 1e-10),
        (12.0, 1e-12),
        (20.0, 1e-12),
    ]:
        mid = c.mean + z * c.std
        width = relative_width * c.std
        lo = mid - 0.5 * width
        hi = mid + 0.5 * width
        represented_width = hi - lo
        represented_mid = lo + 0.5 * represented_width
        got = c.loglik([[lo, hi]])
        expected = np.log(represented_width) + c.logpdf(represented_mid)
        assert np.isclose(got, expected, rtol=0.0, atol=5e-10)


def test_extreme_scale_fit_reports_actionable_error():
    rng = np.random.default_rng(883)
    # Debug mode surfaces the underlying moment-conversion fallback first.
    expected = "raw-moment conversion" if _defaults.DEBUG else "scale|rescale"
    with pytest.raises(RuntimeError, match=expected):
        Distribution().fit(
            rng.normal(size=200) * 1e100,
            n_components=1,
            poly_degree=2,
            support=(-np.inf, np.inf),
            rng=0,
        )

def test_set_default_invalidates_mixture_statistics(mixture_fit):
    c = mixture_fit.copy()
    base_mean = c.mean
    assert c.default == "base"
    c.set_default("exp")
    expected_exp_mean = sum(
        float(w) * float(comp.exp.mean)
        for w, comp in zip(c.weights, c.components, strict=True)
    )
    assert np.isclose(c.mean, expected_exp_mean, rtol=2e-12, atol=2e-12)
    assert not np.isclose(c.mean, base_mean)
    c.set_default("base")
    assert np.isclose(c.mean, base_mean, rtol=2e-12, atol=2e-12)


def test_mixture_state_roundtrip_preserves_more_modes_than_components(mixture_fit):
    c = mixture_fit.copy()
    stored_modes = tuple(np.linspace(-3.0, 3.0, 7))
    c._mode_cache = {"base": stored_modes}
    state = c.data
    assert int(state["n_modes"]) == len(stored_modes)
    assert np.asarray(state["base_modes"]).size == len(stored_modes)
    restored = Distribution(state)
    assert restored._mode_cache["base"] == stored_modes


def _rebuild_mixture_state(state, *, drop=(), replace=None):
    """Copy a structured mixture state, dropping or re-typing named fields."""
    replace = {} if replace is None else replace
    values = {
        name: np.asarray(state[name])
        for name in state.dtype.names
        if name not in drop
    }
    values.update({name: np.asarray(value) for name, value in replace.items()})
    dtype = [
        (name, value.dtype) if value.ndim == 0 else (name, value.dtype, value.shape)
        for name, value in values.items()
    ]
    rebuilt = np.zeros((), dtype=dtype)
    for name, value in values.items():
        rebuilt[name] = value
    return rebuilt


def test_mixture_state_without_cached_modes_uses_canonical_empty_layout(mixture_fit):
    c = mixture_fit.copy()
    c._mode_cache = None
    state = c.data
    assert int(state["n_modes"]) == 0
    assert np.asarray(state["base_modes"]).shape == (0,)
    restored = Distribution(state)
    assert restored._mode_cache is None
    assert restored.modes == mixture_fit.modes


@pytest.mark.parametrize("missing", [
    ("base_modes",),
    ("n_modes",),
    ("base_modes", "n_modes"),
])
def test_mixture_state_missing_mode_fields_is_rejected(mixture_fit, missing):
    state = _rebuild_mixture_state(mixture_fit.data, drop=missing)
    with pytest.raises(ValueError, match="not a gibbus mixture state; missing fields: "
                       + ", ".join(missing)):
        Distribution(state)


@pytest.mark.parametrize("base_modes,n_modes", [
    pytest.param([-1.0, 1.0], 1, id="n_modes-smaller"),
    pytest.param([-1.0, 1.0], 3, id="n_modes-larger"),
    pytest.param([-1.0, 1.0], -1, id="n_modes-negative"),
    pytest.param([-1.0, 1.0, np.nan], 2, id="nan-padded"),
    pytest.param([[-1.0, 1.0]], 2, id="two-dimensional"),
    pytest.param([-1.0, 1.0], [2], id="n_modes-not-scalar"),
    pytest.param([-1.0, 1.0], 2.0, id="n_modes-not-integer"),
])
def test_mixture_state_inconsistent_mode_layout_is_rejected(mixture_fit, base_modes,
                                                            n_modes):
    state = _rebuild_mixture_state(
        mixture_fit.data,
        replace={
            "base_modes": np.asarray(base_modes, dtype=np.float64),
            "n_modes": np.asarray(n_modes),
        },
    )
    with pytest.raises(ValueError, match="inconsistent n_modes/base_modes"):
        Distribution(state)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_mixture_state_non_finite_base_modes_are_rejected(mixture_fit, bad):
    state = _rebuild_mixture_state(
        mixture_fit.data,
        replace={
            "base_modes": np.array([-1.0, bad], dtype=np.float64),
            "n_modes": np.int64(2),
        },
    )
    with pytest.raises(ValueError, match="base_modes must be finite"):
        Distribution(state)


def test_malformed_mixture_mode_state_load_is_exception_safe(gaussian_fit, mixture_fit):
    bad = _rebuild_mixture_state(mixture_fit.data, drop=("n_modes",))
    target = gaussian_fit.copy()
    before = target.mean
    with pytest.raises(ValueError, match="missing fields: n_modes"):
        target.load(bad)
    assert target.n_components == 1
    assert target.mean == before


def test_invalid_mixture_state_load_is_exception_safe(gaussian_fit, mixture_fit):
    bad = np.array(mixture_fit.data, copy=True)
    bad["weights"] = [0.9, 0.9]
    target = gaussian_fit.copy()
    before = target.mean
    with pytest.raises(ValueError, match="component weights"):
        target.load(bad)
    assert target.is_fitted
    assert np.isclose(target.mean, before, rtol=0.0, atol=0.0)


def test_copy_and_pickle_preserve_diagnostic_records(mixture_fit):
    c = mixture_fit.copy()
    c._selection_diagnostics = {
        "method": "synthetic",
        "scores": ({"n_components": 2, "bic": 1.0},),
    }
    clone = copy.deepcopy(c)
    restored = pickle.loads(pickle.dumps(c))
    assert clone.selection_diagnostics == c.selection_diagnostics
    assert restored.selection_diagnostics == c.selection_diagnostics
    assert clone.fit_diagnostics["em"] == c.fit_diagnostics["em"]
    assert restored.fit_diagnostics["em"] == c.fit_diagnostics["em"]


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.sf(0.0),
        lambda c: c.log_hazard(0.0),
        lambda c: c.hazard(0.0),
        lambda c: c.cumulative_hazard(0.0),
        lambda c: c.mean_residual_life(0.0),
        lambda c: c.residual_entropy(0.0),
        lambda c: c.interval(0.9),
        lambda c: c.hpd(0.9),
        lambda c: c.loglik([]),
        lambda c: c.quantile_residuals([]),
        lambda c: c.goodness_of_fit([0.0]),
        lambda c: c.bootstrap_bands([0.0], [0.0]),
    ],
)
def test_extension_methods_guard_unfitted_state_first(call):
    with pytest.raises(RuntimeError, match="not fitted"):
        call(Distribution())


def test_bad_direct_interval_quadrature_falls_back_to_analytic(monkeypatch):
    def bad_quad(*args, **kwargs):
        return 1.0, 1.0

    monkeypatch.setattr(scoring_module, "_quad_with_ledger", bad_quad)
    lo, hi = 0.0, 1e-9
    rows = np.array([[lo, hi]])
    weights = np.ones(1)
    got = scoring_module.interval_loglik(
        rows, weights, norm.logpdf, norm.logcdf, norm.logsf
    )
    expected = log_mass_between(
        norm.logcdf(lo), norm.logcdf(hi), norm.logsf(lo), norm.logsf(hi)
    )
    assert np.isclose(got, expected, rtol=0.0, atol=1e-14)


def test_frozen_bounded_expectation_and_moment(gaussian_fit):
    c = gaussian_fit
    rv = c.frozen()
    got = rv.expect(lambda x: x * x, lb=-0.5, ub=0.75)
    expected = norm(loc=c.mean, scale=c.std).expect(
        lambda x: x * x, lb=-0.5, ub=0.75
    )
    assert np.isclose(got, expected, rtol=0.0, atol=2e-9)
    assert np.isclose(rv.moment(2), c.moment(2), rtol=0.0, atol=0.0)
    with pytest.raises(TypeError):
        rv.interval()


def test_equal_tailed_hpd_and_exp_view_algebra(gaussian_fit):
    c = gaussian_fit
    level = 0.9
    lo, hi = c.interval(level)
    assert np.isclose(c.cdf(hi) - c.cdf(lo), level, atol=2e-10)

    region = c.hpd(level)
    assert region.shape == (1, 2)
    mass = np.exp(c._log_mass(*region[0]))
    assert np.isclose(mass, level, atol=2e-7)
    assert np.isclose(c.logpdf(region[0, 0]), c.logpdf(region[0, 1]), atol=2e-7)

    y = np.exp(c.mean + 0.7 * c.std)
    x = np.log(y)
    assert np.isclose(c.exp.logsf(y), c.base.logsf(x), atol=1e-14)
    assert np.isclose(c.exp.cumulative_hazard(y), c.base.cumulative_hazard(x), atol=1e-14)
    assert np.isclose(c.exp.log_hazard(y), c.base.log_hazard(x) - np.log(y), atol=1e-14)
    assert np.isclose(c.exp.entropy(), c.base.entropy() + c.base.mean, atol=2e-10)
    np.testing.assert_allclose(c.exp.interval(level), np.exp(c.base.interval(level)), rtol=1e-13)


def test_expect_information_and_scoring(gaussian_fit):
    c = gaussian_fit
    assert np.isclose(c.expect(lambda x: x), c.mean, atol=2e-10)
    assert np.isclose(c.entropy(), 0.5 * np.log(2.0 * np.pi * np.e * c.var), atol=2e-10)
    assert np.isclose(c.kl_divergence(c), 0.0, atol=1e-12)
    assert np.isclose(c.cross_entropy(c), c.entropy(), atol=2e-10)

    exact = np.array([-0.5, 0.0, 1.0])
    residuals = c.quantile_residuals(np.column_stack([exact, exact]), rng=123)
    np.testing.assert_allclose(residuals, norm.ppf(c.cdf(exact)), atol=2e-12)

    mid = c.mean + 5.0 * c.std
    width = 1e-8 * c.std
    got = c.loglik([[mid - width / 2.0, mid + width / 2.0]])
    expected = np.log(width) + c.logpdf(mid)
    assert np.isclose(got, expected, rtol=0.0, atol=2e-6)

    y = np.exp(exact)
    base_ll = c.base.loglik(exact)
    exp_ll = c.exp.loglik(y)
    assert np.isclose(exp_ll, base_ll - np.sum(np.log(y)), atol=2e-12)


def test_truncate_composes_and_round_trips_state(gaussian_fit):
    c = gaussian_fit
    lo = c.mean - 0.8 * c.std
    hi = c.mean + 1.1 * c.std
    log_mass = c._log_mass(lo, hi)
    t = c.truncate(lo, hi)

    np.testing.assert_allclose(t.support, [lo, hi], rtol=0.0, atol=0.0)
    grid = np.linspace(lo, hi, 11)[1:-1]
    np.testing.assert_allclose(t.pdf(grid), c.pdf(grid) / np.exp(log_mass), rtol=1e-11)
    np.testing.assert_allclose(t.cdf([lo, hi]), [0.0, 1.0], atol=0.0)

    lo2 = c.mean - 0.3 * c.std
    hi2 = c.mean + 0.5 * c.std
    twice = t.truncate(lo2, hi2)
    direct = c.truncate(lo2, hi2)
    np.testing.assert_allclose(twice.pdf(grid[3:6]), direct.pdf(grid[3:6]), rtol=2e-11)

    loaded = Distribution(t.data)
    np.testing.assert_allclose(loaded.pdf(grid), t.pdf(grid), rtol=0.0, atol=0.0)


def test_frozen_adapter_is_live_and_uses_scipy_names(gaussian_fit):
    parent = gaussian_fit.copy()
    frozen = parent.frozen()
    before = frozen.mean()
    draws = frozen.rvs(size=4, random_state=123)
    assert np.asarray(draws).shape == (4,)
    assert frozen.stats("mv") == (frozen.mean(), frozen.var())
    assert np.isclose(frozen.expect(lambda x: x), frozen.mean(), atol=2e-10)

    parent.transform(mu=0.5, sigma=1.0, pullback=False, inplace=True)
    assert frozen.mean() != before
    assert frozen.mean() == parent.mean


def test_log_concavity_margin_is_nonnegative(gaussian_fit):
    margin = gaussian_fit.spectral_diagnostics["log_concavity_margin"]
    assert np.isfinite(margin)
    assert margin > 0.0



@pytest.fixture(scope="module")
def boundary_sweep_models():
    models = {}
    for n_components in (1, 2, 3):
        rng = np.random.default_rng(1700 + n_components)
        centers = np.linspace(-2.5, 2.5, n_components)
        sample = np.concatenate([rng.normal(c, 0.45, 30) for c in centers])
        models[n_components] = Distribution().fit(
            sample,
            n_components=n_components,
            poly_degree=2,
            support=(-np.inf, np.inf),
            rng=0,
        )
    return models


@pytest.mark.parametrize("support_kind", ["unbounded", "lower", "upper", "finite"])
@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_extension_boundary_sweep(boundary_sweep_models, support_kind, n_components):
    base = boundary_sweep_models[n_components]
    if support_kind == "unbounded":
        c = base
    elif support_kind == "lower":
        c = base.truncate(float(base.ppf(0.05)), np.inf)
    elif support_kind == "upper":
        c = base.truncate(-np.inf, float(base.ppf(0.95)))
    else:
        c = base.truncate(float(base.ppf(0.05)), float(base.ppf(0.95)))

    extreme_grid = np.asarray(c.ppf([1e-12, 1.0 - 1e-12]), dtype=np.float64)
    core_grid = np.asarray(c.ppf([0.01, 0.5, 0.99]), dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extreme_logcdf = np.asarray(c.logcdf(extreme_grid), dtype=np.float64)
        extreme_logsf = np.asarray(c.logsf(extreme_grid), dtype=np.float64)
        logsf = np.asarray(c.logsf(core_grid), dtype=np.float64)
        hazard = np.asarray(c.hazard(core_grid), dtype=np.float64)
        cumulative = np.asarray(c.cumulative_hazard(core_grid), dtype=np.float64)
        mrl = np.asarray(c.mean_residual_life(core_grid), dtype=np.float64)
        residual_entropy = np.asarray(c.residual_entropy(core_grid), dtype=np.float64)

    for values in (
        extreme_logcdf,
        extreme_logsf,
        logsf,
        hazard,
        cumulative,
        mrl,
        residual_entropy,
    ):
        assert not np.any(np.isnan(values))
    assert np.all(np.diff(extreme_logcdf) >= -1e-9)
    assert np.all(np.diff(extreme_logsf) <= 1e-9)
    assert np.all(np.diff(logsf) <= 1e-9)
    assert np.all(np.diff(cumulative) >= -1e-9)
    assert np.all(mrl >= c.mean - core_grid - 2e-6)
    if n_components == 1:
        assert np.all(mrl * hazard <= 1.0 + 2e-6)

    support = np.asarray(c.support, dtype=np.float64)
    outside = []
    if np.isfinite(support[0]):
        outside.append(float(support[0]) - 0.25)
    if np.isfinite(support[1]):
        outside.append(float(support[1]) + 0.25)
    if outside:
        outside = np.asarray(outside, dtype=np.float64)
        assert not np.any(np.isnan(c.logcdf(outside)))
        assert not np.any(np.isnan(c.logsf(outside)))
        assert not np.any(np.isnan(c.mean_residual_life(outside)))


def test_out_of_support_scoring_rows_warn_without_raising():
    """Scoring below/above a bounded support warns instead of failing.

    Regression guard: ``warn_out_of_support`` reaches ``warnings.warn`` on this
    path only, so a missing module import surfaces as ``NameError`` here rather
    than in any other test.
    """
    rng = np.random.default_rng(20260914)
    c = Distribution().fit(
        rng.beta(2.0, 3.0, size=800),
        n_components=1,
        poly_degree=2,
        support=(0.0, 1.0),
        rng=0,
    )
    rows = np.array([-0.5, 0.25, 0.75, 1.5], dtype=np.float64)

    with pytest.warns(RuntimeWarning, match="outside the fitted support"):
        value = c.loglik(rows)
    # Zero-probability rows contribute -inf; the point is that the warning path
    # completes rather than raising.
    assert value == -np.inf

    with pytest.warns(RuntimeWarning, match="outside the fitted support"):
        assert scoring_module.warn_out_of_support(rows.reshape(-1, 1), c.support) == 2

    intervals = np.array([[-0.9, -0.6], [0.2, 0.4]], dtype=np.float64)
    with pytest.warns(RuntimeWarning, match="outside the fitted support") as record:
        c.loglik(intervals)
    # Only the gibbus warning: a zero-mass row must not leak NumPy's
    # "invalid value" warning from -inf - (-inf).
    assert not [w for w in record if "invalid value" in str(w.message)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isfinite(c.loglik(np.array([0.25, 0.75])))


def test_goodness_of_fit_accepts_a_correct_model_and_rejects_a_wrong_one():
    rng = np.random.default_rng(555)
    truth = rng.normal(size=1500)
    c = Distribution().fit(truth, n_components=1, poly_degree=2, rng=0)
    holdout = rng.normal(size=1500)
    good = c.goodness_of_fit(holdout, statistic="cvm")
    wrong = c.goodness_of_fit(rng.normal(loc=3.0, size=1500), statistic="cvm")
    assert good["pvalue"] > 0.01
    assert wrong["pvalue"] < 1e-3
    assert wrong["value"] > good["value"]


@pytest.mark.parametrize("statistic", ["ks", "cvm", "ad"])
def test_goodness_of_fit_reports_its_calibration_contract(gaussian_fit, statistic):
    rng = np.random.default_rng(556)
    out = gaussian_fit.goodness_of_fit(rng.normal(0.2, 1.1, size=300),
                                       statistic=statistic)
    assert out["statistic"] == statistic
    assert out["calibration"] == "asymptotic"
    assert out["pvalue_valid_for"] == "held-out observations only"
    assert out["n"] == 300
    assert out["n_resamples"] == 0 and out["n_failed"] == 0
    assert np.isfinite(out["value"])
    # Anderson-Darling has no asymptotic null here; None records that.
    assert (out["pvalue"] is None) == (statistic == "ad")


def test_goodness_of_fit_monte_carlo_calibration_is_valid_in_sample():
    rng = np.random.default_rng(557)
    data = rng.normal(size=400)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    out = c.goodness_of_fit(data, statistic="ad", calibration="montecarlo",
                            n_resamples=25, rng=0)
    assert out["calibration"] == "montecarlo"
    assert out["pvalue_valid_for"] == "the sample the model was fitted to"
    assert 0.0 < out["pvalue"] <= 1.0
    assert out["n_resamples"] == 25
    assert out["n_failed"] <= 25


def test_goodness_of_fit_monte_carlo_is_deterministic_under_a_seed():
    rng = np.random.default_rng(558)
    data = rng.normal(size=250)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    kwargs = {"statistic": "ks", "calibration": "montecarlo", "n_resamples": 12, "rng": 7}
    assert c.goodness_of_fit(data, **kwargs) == c.goodness_of_fit(data, **kwargs)


def test_goodness_of_fit_rejects_intervals_and_bad_arguments(gaussian_fit):
    with pytest.raises(ValueError, match="no single probability-integral"):
        gaussian_fit.goodness_of_fit(np.array([[0.0, 1.0], [1.0, 2.0]]))
    with pytest.raises(ValueError, match="statistic must be one of"):
        gaussian_fit.goodness_of_fit([0.0, 1.0], statistic="chi2")
    with pytest.raises(ValueError, match="calibration must be"):
        gaussian_fit.goodness_of_fit([0.0, 1.0], calibration="bayes")


def test_goodness_of_fit_requires_a_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        Distribution().goodness_of_fit([0.0, 1.0])


def test_bootstrap_bands_bracket_the_point_estimate():
    rng = np.random.default_rng(559)
    data = rng.normal(size=400)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    grid = np.linspace(-2.0, 2.0, 9)
    out = c.bootstrap_bands(data, grid, quantity="pdf", n_resamples=30,
                            level=0.90, rng=0)
    assert out["quantity"] == "pdf"
    assert out["coverage_kind"] == "pointwise"
    assert out["level"] == 0.90
    assert out["n_resamples"] == 30
    np.testing.assert_allclose(out["x"], grid)
    np.testing.assert_allclose(out["estimate"], c.pdf(grid))
    assert np.all(out["lower"] <= out["upper"])
    assert np.all(out["lower"] >= 0.0)
    # The point estimate should sit inside a 90% band almost everywhere.
    inside = (out["lower"] <= out["estimate"]) & (out["estimate"] <= out["upper"])
    assert inside.sum() >= grid.size - 1


def test_bootstrap_bands_narrow_as_the_sample_grows():
    """Uncertainty must shrink with data, or the bands mean nothing."""
    rng = np.random.default_rng(560)
    grid = np.linspace(-1.0, 1.0, 5)
    widths = []
    for size in (200, 2000):
        data = rng.normal(size=size)
        c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
        out = c.bootstrap_bands(data, grid, n_resamples=30, level=0.90, rng=1)
        widths.append(float(np.mean(out["upper"] - out["lower"])))
    assert widths[1] < widths[0]


@pytest.mark.parametrize("quantity", ["pdf", "cdf", "sf"])
def test_bootstrap_bands_supports_each_quantity(quantity):
    rng = np.random.default_rng(561)
    data = rng.normal(size=300)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    out = c.bootstrap_bands(data, np.linspace(-1.5, 1.5, 6), quantity=quantity,
                            n_resamples=15, level=0.95, rng=0)
    assert out["quantity"] == quantity
    assert np.all(out["lower"] <= out["upper"])
    if quantity in ("cdf", "sf"):
        assert np.all(out["lower"] >= 0.0) and np.all(out["upper"] <= 1.0)


def test_bootstrap_bands_is_deterministic_under_a_seed():
    rng = np.random.default_rng(562)
    data = rng.normal(size=250)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    grid = np.linspace(-1.0, 1.0, 4)
    kwargs = {"n_resamples": 12, "level": 0.9, "rng": 5}
    first = c.bootstrap_bands(data, grid, **kwargs)
    second = c.bootstrap_bands(data, grid, **kwargs)
    np.testing.assert_array_equal(first["lower"], second["lower"])
    np.testing.assert_array_equal(first["upper"], second["upper"])


def test_bootstrap_bands_respects_the_active_space():
    """Replicates must be evaluated in the same space as the estimate."""
    rng = np.random.default_rng(563)
    data = rng.normal(size=300)
    c = Distribution().fit(data, n_components=1, poly_degree=2, rng=0)
    c.set_default("exp")
    grid = np.linspace(0.5, 2.0, 5)
    out = c.bootstrap_bands(data, grid, quantity="cdf", n_resamples=12,
                            level=0.9, rng=0)
    np.testing.assert_allclose(out["estimate"], c.cdf(grid))
    assert np.all(out["lower"] <= out["upper"])


def test_bootstrap_bands_rejects_bad_arguments(gaussian_fit):
    data = np.random.default_rng(564).normal(0.2, 1.1, size=100)
    with pytest.raises(ValueError, match="quantity must be"):
        gaussian_fit.bootstrap_bands(data, [0.0], quantity="hazard")
    with pytest.raises(ValueError, match="at least one abscissa"):
        gaussian_fit.bootstrap_bands(data, [])
    with pytest.raises(ValueError, match="abscissae must be finite"):
        gaussian_fit.bootstrap_bands(data, [0.0, np.inf])
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        gaussian_fit.bootstrap_bands(data, [0.0], level=1.0)
    with pytest.raises(TypeError, match="fit_kwargs must be a mapping"):
        gaussian_fit.bootstrap_bands(data, [0.0], fit_kwargs=["n_components", 1])


def test_bootstrap_bands_requires_a_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        Distribution().bootstrap_bands([0.0, 1.0], [0.0])


def test_log_concavity_margin_is_reported_through_spectral_diagnostics(gaussian_fit):
    diag = gaussian_fit.spectral_diagnostics
    assert "log_concavity_margin" in diag
    assert diag["log_concavity_margin"] >= -1e-9


def test_frozen_rvs_accepts_a_shape_tuple_like_scipy():
    rng = np.random.default_rng(413)
    c = Distribution().fit(
        rng.normal(size=400), n_components=1,
        support=(-np.inf, np.inf), rng=0,
    )
    draws = c.frozen().rvs(size=(2, 3), random_state=np.random.default_rng(0))
    assert draws.shape == (2, 3)
    assert np.all(np.isfinite(draws))


def test_frozen_interval_accepts_an_array_of_levels():
    rng = np.random.default_rng(414)
    c = Distribution().fit(
        rng.normal(size=400), n_components=1,
        support=(-np.inf, np.inf), rng=0,
    )
    rv = c.frozen()
    lower, upper = rv.interval(np.array([0.5, 0.9]))
    assert lower.shape == upper.shape == (2,)
    for index, level in enumerate((0.5, 0.9)):
        one_lo, one_hi = rv.interval(level)
        assert lower[index] == pytest.approx(one_lo)
        assert upper[index] == pytest.approx(one_hi)


def test_transform_requires_explicit_pullback_direction(gaussian_fit):
    with pytest.raises(TypeError, match="pullback"):
        gaussian_fit.transform(mu=1.0, sigma=2.0)


def test_cumulants_match_moments_and_affine_law(gaussian_fit):
    c = gaussian_fit
    assert c.cumulant(1) == pytest.approx(c.mean)
    assert c.cumulant(2) == pytest.approx(c.var, rel=2e-11, abs=2e-12)
    assert c.cumulant(3) == pytest.approx(c.moment(3, central=True), rel=2e-10, abs=2e-11)
    expected4 = c.moment(4, central=True) - 3.0 * c.var**2
    assert c.cumulant(4) == pytest.approx(expected4, rel=2e-10, abs=2e-11)

    moved = c.transform(mu=7.0, sigma=2.5, pullback=False, inplace=False)
    assert moved.cumulant(1) == pytest.approx(7.0 + 2.5 * c.cumulant(1), rel=2e-10)
    for order in (2, 3, 4):
        assert moved.cumulant(order) == pytest.approx(
            (2.5**order) * c.cumulant(order), rel=2e-8, abs=2e-9
        )


def test_cumulant_rejects_nonpositive_or_nonintegral_orders(gaussian_fit):
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            gaussian_fit.cumulant(bad)


def test_transform_rejects_non_boolean_pullback(gaussian_fit):
    with pytest.raises(TypeError, match="pullback must be a bool"):
        gaussian_fit.transform(mu=1.0, sigma=2.0, pullback=1)
