"""Targeted adversarial numerical regressions found outside the broad campaigns."""

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._postfit.analytics import (
    _exp_moment_from_stats,
    _exp_stats_from_log_moments,
)
from gibbus._serialization import pack_distribution_state


def test_extreme_lower_tail_cdf_roundtrips_single_component():
    rng = np.random.default_rng(811)
    model = Distribution().fit(
        rng.normal(size=1000),
        n_components=1,
        poly_degree=4,
        support=(-np.inf, np.inf),
        rng=0,
    )
    p = np.array([1e-300, 1e-200, 1e-100, 1e-30, 1e-16, 1e-12])
    q = np.asarray(model.ppf(p), dtype=np.float64)
    got = np.asarray(model.cdf(q), dtype=np.float64)

    assert np.all(got > 0.0)
    np.testing.assert_allclose(np.log(got), np.log(p), rtol=0.0, atol=3e-12)


def test_extreme_lower_tail_cdf_roundtrips_mixture():
    rng = np.random.default_rng(812)
    x = np.concatenate(
        [
            rng.normal(-2.0, 0.5, 700),
            rng.normal(2.0, 0.7, 700),
        ]
    )
    model = Distribution().fit(
        x,
        n_components=2,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )
    p = np.array([1e-200, 1e-100, 1e-30, 1e-16, 1e-12])
    q = np.asarray(model.ppf(p), dtype=np.float64)
    got = np.asarray(model.cdf(q), dtype=np.float64)

    assert np.all(got > 0.0)
    np.testing.assert_allclose(np.log(got), np.log(p), rtol=0.0, atol=3e-11)


def test_point_fit_is_invariant_to_extreme_global_weight_scale():
    rng = np.random.default_rng(813)
    x = rng.normal(size=350)
    weights = np.exp(rng.uniform(-20.0, 20.0, size=x.size))
    grid = np.linspace(-4.0, 4.0, 101)

    baseline = Distribution().fit(
        x,
        n_components=1,
        poly_degree=4,
        support=(-np.inf, np.inf),
        sample_weight=weights,
        rng=0,
    )
    expected = np.asarray(baseline.cdf(grid), dtype=np.float64)

    for factor in (1e-200, 1e200):
        fitted = Distribution().fit(
            x,
            n_components=1,
            poly_degree=4,
            support=(-np.inf, np.inf),
            sample_weight=weights * factor,
            rng=0,
        )
        np.testing.assert_allclose(fitted.cdf(grid), expected, rtol=0.0, atol=2e-13)


def test_bounded_fit_survives_support_only_256_ulps_wide():
    lower = 1.0
    ulp = np.spacing(lower)
    upper = lower + 256.0 * ulp
    levels = lower + np.arange(1, 256, dtype=np.float64) * ulp
    x = np.repeat(levels[::4], 4)

    fitted = Distribution().fit(
        x,
        n_components=1,
        poly_degree=2,
        support=(lower, upper),
        log_boundary_lower=False,
        log_boundary_upper=False,
        rng=0,
    )
    cdf = np.asarray(fitted.cdf(levels), dtype=np.float64)
    assert fitted.fit_diagnostics["converged"]
    assert fitted.isf(1e-6) == upper
    assert fitted.logppf(np.log(1e-6)) == lower
    assert fitted.logisf(np.log(1e-6)) == upper
    assert np.all(np.isfinite(cdf))
    assert np.all(np.diff(cdf) >= 0.0)
    assert 0.0 < cdf[0] < cdf[-1] < 1.0


def test_endpoint_concentrated_bounded_fit_keeps_explicit_lower_boundary_basis():
    """Endpoint-concentrated beta data with Fisher condition 2.5e37.

    The fit certifies since a shape-rejected trial step is shortened by an
    exact ratio test instead of halved.
    """
    rng = np.random.default_rng(91)
    x = np.clip(
        rng.beta(0.02, 2.0, 1200),
        np.nextafter(0.0, 1.0),
        np.nextafter(1.0, 0.0),
    )

    fitted = Distribution().fit(
        x,
        n_components=1,
        poly_degree=4,
        support=(0.0, 1.0),
        log_boundary_lower=True,
        rng=0,
    )

    assert fitted.fit_diagnostics["converged"]
    assert fitted.fit_diagnostics["shared_boundary"]["allowed"][0]


def test_nearly_coincident_forced_mixture_remains_certified():
    rng = np.random.default_rng(814)
    x = np.concatenate(
        [
            rng.normal(-0.02, 1.0, 600),
            rng.normal(0.02, 1.0, 600),
        ]
    )
    fitted = Distribution().fit(
        x,
        n_components=3,
        poly_degree=2,
        support=(-np.inf, np.inf),
        em_max_iter=200,
        rng=0,
    )
    assert fitted.fit_diagnostics["converged"]
    assert np.all(np.isfinite(fitted.weights))
    assert np.all(fitted.weights > 0.0)
    assert np.sum(fitted.weights) == pytest.approx(1.0, abs=2e-14)
    grid = np.linspace(-6.0, 6.0, 301)
    cdf = np.asarray(fitted.cdf(grid), dtype=np.float64)
    assert np.all(np.diff(cdf) >= -2e-13)


def test_exp_statistics_propagate_overflow_instead_of_reporting_degeneracy():
    """Huge valid transformed moments should produce infinities, not a spike error."""
    stats = _exp_stats_from_log_moments(
        [1e8, 3e8, 8e8, 15e8],
        lambda _log_mean, _k: np.nan,
        "adversarial exp-space",
    )
    assert np.isinf(stats["mean"])
    assert np.isinf(stats["var"])
    assert np.isinf(stats["std"])
    assert np.isnan(stats["skew"])
    assert np.isnan(stats["kurt"])

    divergent = _exp_stats_from_log_moments(
        [np.inf, np.inf, np.inf, np.inf],
        lambda _log_mean, _k: np.nan,
        "adversarial exp-space",
    )
    assert np.isinf(divergent["mean"])
    assert np.isinf(divergent["var"])


def test_exp_statistics_report_divergent_higher_moments_without_nan_cancellation():
    """Finite variance plus a divergent higher raw moment has infinite shape."""
    third = _exp_stats_from_log_moments(
        [0.0, 1.0, np.inf, np.inf],
        lambda _log_mean, _k: np.nan,
        "adversarial exp-space",
    )
    assert np.isinf(third["skew"])
    assert np.isinf(third["kurt"])
    assert np.isinf(_exp_moment_from_stats(3, third, False))
    assert np.isinf(_exp_moment_from_stats(4, third, False))

    infinite_variance = _exp_stats_from_log_moments(
        [np.log(3.0), np.inf, np.inf, np.inf],
        lambda _log_mean, _k: np.nan,
        "adversarial exp-space",
    )
    assert np.isinf(_exp_moment_from_stats(3, infinite_variance, False))
    assert np.isinf(_exp_moment_from_stats(4, infinite_variance, False))
    assert np.isnan(_exp_moment_from_stats(2, infinite_variance, True))


def test_zero_weight_mixture_component_cannot_contaminate_exp_moments():
    """Dormant serialized components must be ignored before moment evaluation."""
    rng = np.random.default_rng(815)
    active = Distribution().fit(
        rng.normal(size=700),
        n_components=1,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )
    dormant = Distribution().fit(
        rng.normal(1000.0, 1.0, size=700),
        n_components=1,
        poly_degree=2,
        support=(-np.inf, np.inf),
        rng=0,
    )
    assert np.isinf(dormant.exp.moment(1))

    state = pack_distribution_state(
        np.array([1.0, 0.0]),
        "exp",
        [active.components[0].data, dormant.components[0].data],
    )
    mixture = Distribution().load(state)

    assert mixture.mean == pytest.approx(active.exp.mean, rel=0.0, abs=2e-13)
    assert mixture.moment(1) == pytest.approx(active.exp.moment(1), rel=0.0, abs=2e-13)
    assert mixture.moment(2, central=True) == pytest.approx(
        active.exp.var, rel=2e-12, abs=2e-13
    )
    assert mixture.moment(5) == pytest.approx(active.exp.moment(5), rel=2e-12)
