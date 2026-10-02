"""Targeted adversarial numerical regressions found outside the broad campaigns."""

import numpy as np
import pytest
from scipy.special import erfcx, log_ndtr

from gibbus import Distribution
from gibbus._fit.conic_newton import _interior_start
from gibbus._fit.mixture import _pack_mixture_struct
from gibbus._fit.natural_objective import (
    _prepare_natural_interval_objective,
    _prepare_natural_point_objective,
)
from gibbus._model.numerics import _terms_for_quad
from gibbus._postfit.analytics import (
    _exp_moment_from_stats,
    _exp_stats_from_log_moments,
    _log_raw_moment_exp,
    _tail_rate_from_geometry,
)


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
    x = np.concatenate([
        rng.normal(-2.0, 0.5, 700),
        rng.normal(2.0, 0.7, 700),
    ])
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


def test_point_boundary_statistics_preserve_subulp_physical_distance():
    points = np.array([
        np.nextafter(0.0, 1.0),
        1e-300,
        1e-100,
        1e-20,
        1e-5,
        0.01,
        0.1,
        0.4,
        0.8,
    ])
    point = _prepare_natural_point_objective(
        (0.0, np.inf), points, 2, True, False, None
    )
    interval = _prepare_natural_interval_objective(
        (0.0, np.inf), np.column_stack([points, points]), 2, True, False, None
    )

    # The smallest physical points round onto the canonical endpoint, but the
    # preserved distance must keep their boundary-log statistic finite.
    z = point.spec.coordinate.to_canonical(points)
    assert z[0] == point.layout.support[0]
    assert np.isfinite(point.observations.stats.boundary_log[0])

    params = _interior_start(point)
    point_eval = point(params)
    interval_eval = interval(params)
    assert point_eval.nll == pytest.approx(interval_eval.nll, abs=2e-12)
    np.testing.assert_allclose(
        point_eval.gradient, interval_eval.gradient, rtol=2e-13, atol=2e-12
    )
    np.testing.assert_allclose(
        point_eval.hessian, interval_eval.hessian, rtol=2e-13, atol=2e-12
    )


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
        sample_weights=weights,
        rng=0,
    )
    expected = np.asarray(baseline.cdf(grid), dtype=np.float64)

    for factor in (1e-200, 1e200):
        fitted = Distribution().fit(
            x,
            n_components=1,
            poly_degree=4,
            support=(-np.inf, np.inf),
            sample_weights=weights * factor,
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
    assert bool(fitted.data["boundary_allowed"][0])


def test_nearly_coincident_forced_mixture_remains_certified():
    rng = np.random.default_rng(814)
    x = np.concatenate([
        rng.normal(-0.02, 1.0, 600),
        rng.normal(0.02, 1.0, 600),
    ])
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



def _log_halfline_quadratic_integral(a, c):
    """Return log integral_0^inf exp(-a z^2 - c z) dz stably."""
    x = float(c) / (2.0 * np.sqrt(float(a)))
    if x >= 0.0:
        return float(
            0.5 * np.log(np.pi)
            - np.log(2.0)
            - 0.5 * np.log(a)
            + np.log(erfcx(x))
        )
    return float(
        0.5 * np.log(np.pi)
        - 0.5 * np.log(a)
        + c * c / (4.0 * a)
        + log_ndtr(-x * np.sqrt(2.0))
    )


def test_exp_moment_expands_beyond_original_material_window():
    """An exponential tilt must follow its saddle beyond the base-density window."""
    a = 1e-12
    b = 0.9
    support = np.array([0.0, np.inf])
    amplitudes = np.zeros(2)
    log_normalizer = _log_halfline_quadratic_integral(a, b)
    q_poly = np.array([log_normalizer, b, a])
    window = np.array([0.0, 50.0])

    expected = (
        _log_halfline_quadratic_integral(a, b - 1.0)
        - log_normalizer
    )
    got = _log_raw_moment_exp(
        support,
        q_poly,
        amplitudes,
        window,
        0.0,
        1.0,
        1,
        _terms_for_quad(support, amplitudes),
    )

    assert expected > np.log(np.finfo(float).max)
    assert got == pytest.approx(expected, rel=0.0, abs=2e-6)


def test_exp_moment_detects_exact_exponential_threshold():
    """For an Exp(1) base tail, E[exp(X)] diverges exactly at rate one."""
    support = np.array([0.0, np.inf])
    amplitudes = np.zeros(2)
    q_poly = np.array([0.0, 1.0])  # normalized Exp(1) potential on [0, inf)

    got = _log_raw_moment_exp(
        support,
        q_poly,
        amplitudes,
        np.array([0.0, 40.0]),
        0.0,
        1.0,
        1,
        _terms_for_quad(support, amplitudes),
    )
    assert np.isposinf(got)

    # A faster Exp(2) tail has E[exp(X)] = 2 exactly.
    q_fast = np.array([-np.log(2.0), 2.0])
    finite = _log_raw_moment_exp(
        support,
        q_fast,
        amplitudes,
        np.array([0.0, 40.0]),
        0.0,
        1.0,
        1,
        _terms_for_quad(support, amplitudes),
    )
    assert finite == pytest.approx(np.log(2.0), abs=2e-13)


def test_tail_rate_uses_exact_asymptotic_polynomial_geometry():
    """Any positive quadratic curvature implies infinite limiting tail slope."""
    support = np.array([0.0, np.inf])
    almost_linear = np.array([0.0, 1.0, 1e-30])
    exactly_linear = np.array([0.0, 1.75])

    assert np.isinf(
        _tail_rate_from_geometry(support, almost_linear, 0.0, 1.0, "upper")
    )
    assert _tail_rate_from_geometry(
        support, exactly_linear, 0.0, 1.0, "upper"
    ) == pytest.approx(1.75)
    assert np.isinf(
        _tail_rate_from_geometry(support, exactly_linear, 0.0, 1.0, "lower")
    )



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
    dormant = active.transform(mu=1000.0, sigma=1.0, pullback=False, inplace=False)
    assert np.isinf(dormant.exp.moment(1))

    state = _pack_mixture_struct(
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
