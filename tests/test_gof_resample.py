"""Defining-property tests for the goodness-of-fit and resampling primitives.

These modules are deliberately independent of the fitting machinery -- the
statistics are pure functions of a PIT sample, and the resampling loops take
per-replicate work as a callable -- so everything here runs without building
a model.  The `Distribution`-level wiring is covered in `test_api_extensions.py`.
"""

import numpy as np
import pytest
from scipy import stats

from gibbus._defaults import BOOTSTRAP_MAX_FAILURE_FRACTION
from gibbus._postfit.gof import (
    GOF_STATISTICS,
    asymptotic_pvalue,
    canonical_pit,
    gof_statistic,
    monte_carlo_pvalue,
    validate_statistic,
)
from gibbus._postfit.resample import (
    bootstrap_curves,
    percentile_bands,
    simulated_statistics,
    validate_confidence,
    validate_resample_count,
)


@pytest.mark.parametrize("n", [10, 97, 1000])
def test_ks_and_cvm_statistics_match_scipy(n):
    u = np.random.default_rng(11).random(n)
    assert gof_statistic(u, "ks") == pytest.approx(
        stats.kstest(u, "uniform").statistic, rel=1e-12
    )
    assert gof_statistic(u, "cvm") == pytest.approx(
        stats.cramervonmises(u, "uniform").statistic, rel=1e-12
    )


@pytest.mark.parametrize("n", [10, 97, 1000])
def test_ks_and_cvm_pvalues_match_scipy(n):
    u = np.random.default_rng(12).random(n)
    assert asymptotic_pvalue(u, "ks") == pytest.approx(
        stats.kstest(u, "uniform").pvalue, rel=1e-12
    )
    assert asymptotic_pvalue(u, "cvm") == pytest.approx(
        stats.cramervonmises(u, "uniform").pvalue, rel=1e-12
    )


def test_anderson_darling_null_mean_is_one():
    """A-squared has asymptotic null mean 1; this pins the formula's scale.

    A transposed index or a missing reversal in the weighted sum changes the
    mean substantially, so this catches the errors the SciPy cross-checks
    cannot (SciPy exposes no uniform Anderson-Darling statistic).
    """
    rng = np.random.default_rng(13)
    draws = np.array([gof_statistic(rng.random(500), "ad") for _ in range(3000)])
    assert draws.mean() == pytest.approx(1.0, abs=0.06)
    assert np.all(np.isfinite(draws))


def test_cramer_von_mises_null_mean_is_one_sixth():
    rng = np.random.default_rng(14)
    draws = np.array([gof_statistic(rng.random(400), "cvm") for _ in range(3000)])
    assert draws.mean() == pytest.approx(1.0 / 6.0, abs=0.01)


@pytest.mark.parametrize("statistic", GOF_STATISTICS)
def test_statistics_detect_non_uniformity(statistic):
    rng = np.random.default_rng(15)
    uniform = gof_statistic(rng.random(400), statistic)
    skewed = gof_statistic(rng.random(400) ** 3, statistic)
    assert skewed > uniform


@pytest.mark.parametrize("statistic", GOF_STATISTICS)
def test_statistics_are_finite_at_support_endpoints(statistic):
    """PIT values of exactly 0 or 1 are legitimate on a bounded support.

    Unclipped they send the Anderson-Darling weight to infinity, which would
    make the statistic report a perfect fit as infinitely bad.
    """
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.isfinite(gof_statistic(u, statistic))


def test_anderson_darling_has_no_asymptotic_pvalue():
    """`None` records the absence of a calibrated reference, not a failure."""
    assert asymptotic_pvalue(np.linspace(0.05, 0.95, 20), "ad") is None


def test_gof_statistic_is_order_invariant():
    rng = np.random.default_rng(16)
    u = rng.random(50)
    shuffled = rng.permutation(u)
    for statistic in GOF_STATISTICS:
        assert gof_statistic(u, statistic) == pytest.approx(
            gof_statistic(shuffled, statistic), rel=1e-14
        )


@pytest.mark.parametrize("bad", [[], [0.5, np.nan], [0.5, 1.5], [-0.1, 0.5]])
def test_canonical_pit_rejects_invalid_samples(bad):
    with pytest.raises(ValueError):
        canonical_pit(np.asarray(bad, dtype=np.float64))


@pytest.mark.parametrize("bad", ["", "kolmogorov", "KS2", 3])
def test_validate_statistic_rejects_unknown_keys(bad):
    with pytest.raises(ValueError, match="statistic must be one of"):
        validate_statistic(bad)


def test_validate_statistic_normalizes_case_and_whitespace():
    assert validate_statistic("  AD ") == "ad"


def test_monte_carlo_pvalue_is_never_zero():
    """A plain proportion can report p = 0 from a finite simulation."""
    assert monte_carlo_pvalue(1e9, np.zeros(999)) == pytest.approx(1.0 / 1000.0)
    assert monte_carlo_pvalue(-1e9, np.zeros(9)) == pytest.approx(1.0)


def test_monte_carlo_pvalue_ignores_non_finite_replicates():
    reps = np.array([0.0, np.nan, 2.0, np.inf])
    assert monte_carlo_pvalue(1.0, reps) == pytest.approx(2.0 / 3.0)


def test_monte_carlo_pvalue_rejects_all_non_finite():
    with pytest.raises(ValueError, match="at least one finite replicate"):
        monte_carlo_pvalue(1.0, [np.nan, np.inf])


def test_percentile_bands_are_exact_on_a_known_sample():
    curves = np.arange(101, dtype=np.float64).reshape(-1, 1)
    lower, upper = percentile_bands(curves, 0.90)
    assert lower[0] == pytest.approx(5.0)
    assert upper[0] == pytest.approx(95.0)


def test_percentile_bands_widen_with_level():
    rng = np.random.default_rng(17)
    curves = rng.normal(size=(500, 4))
    narrow = percentile_bands(curves, 0.50)
    wide = percentile_bands(curves, 0.99)
    assert np.all(wide[0] <= narrow[0])
    assert np.all(wide[1] >= narrow[1])


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, np.nan])
def test_validate_resample_count_rejects_invalid(bad):
    with pytest.raises((TypeError, ValueError)):
        validate_resample_count(bad)


def test_validate_resample_count_accepts_integral_float():
    assert validate_resample_count(200.0) == 200


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.2, np.nan, np.inf])
def test_validate_confidence_rejects_closed_interval(bad):
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        validate_confidence(bad)


def test_bootstrap_curves_resamples_rows_with_replacement():
    seen = []

    def evaluate(indices):
        seen.append(np.asarray(indices))
        return np.array([float(indices.mean())])

    result = bootstrap_curves(
        evaluate,
        40,
        1,
        n_resamples=25,
        level=0.9,
        rng=np.random.default_rng(18),
    )
    assert result["n_resamples"] == 25
    assert result["n_failed"] == 0
    assert all(idx.size == 40 for idx in seen)
    assert all(idx.min() >= 0 and idx.max() < 40 for idx in seen)
    assert any(np.unique(idx).size < 40 for idx in seen), (
        "draws must be with replacement"
    )
    assert result["lower"] <= result["upper"]


def test_bootstrap_curves_is_deterministic_under_equal_seeds():
    def evaluate(indices):
        return np.array([float(indices.sum())])

    kwargs = {"n_resamples": 20, "level": 0.95}
    first = bootstrap_curves(evaluate, 30, 1, rng=np.random.default_rng(19), **kwargs)
    second = bootstrap_curves(evaluate, 30, 1, rng=np.random.default_rng(19), **kwargs)
    assert np.array_equal(first["lower"], second["lower"])
    assert np.array_equal(first["upper"], second["upper"])


def test_bootstrap_curves_counts_declined_and_non_finite_replicates():
    calls = {"n": 0}

    def evaluate(indices):
        calls["n"] += 1
        if calls["n"] == 3:
            return None
        if calls["n"] == 7:
            return np.array([np.nan])
        return np.array([1.0])

    result = bootstrap_curves(
        evaluate,
        10,
        1,
        n_resamples=20,
        level=0.9,
        rng=np.random.default_rng(20),
    )
    assert result["n_failed"] == 2
    assert result["n_resamples"] == 20


def test_bootstrap_curves_rejects_wrong_width_replicates():
    def evaluate(indices):
        return np.array([1.0, 2.0, 3.0])

    with pytest.raises(RuntimeError, match="every bootstrap replicate failed"):
        bootstrap_curves(
            evaluate,
            10,
            2,
            n_resamples=4,
            level=0.9,
            rng=np.random.default_rng(21),
        )


def test_bootstrap_curves_enforces_the_failure_budget():
    """Surviving replicates stop being a fair sample once most have failed."""
    calls = {"n": 0}
    allowed = 4

    def evaluate(indices):
        calls["n"] += 1
        return np.array([1.0]) if calls["n"] <= allowed else None

    n_resamples = 20
    assert allowed < (1.0 - BOOTSTRAP_MAX_FAILURE_FRACTION) * n_resamples
    with pytest.raises(RuntimeError, match="above the tolerated fraction"):
        bootstrap_curves(
            evaluate,
            10,
            1,
            n_resamples=n_resamples,
            level=0.9,
            rng=np.random.default_rng(22),
        )


def test_bootstrap_curves_raises_when_every_replicate_fails():
    with pytest.raises(RuntimeError, match="too fragile under resampling"):
        bootstrap_curves(
            lambda indices: None,
            10,
            1,
            n_resamples=6,
            level=0.9,
            rng=np.random.default_rng(23),
        )


@pytest.mark.parametrize("rows,points", [(0, 1), (5, 0)])
def test_bootstrap_curves_rejects_degenerate_shapes(rows, points):
    with pytest.raises(ValueError):
        bootstrap_curves(
            lambda indices: np.array([1.0]),
            rows,
            points,
            n_resamples=2,
            level=0.9,
            rng=np.random.default_rng(24),
        )


def test_simulated_statistics_returns_flat_finite_draws():
    values = iter([1.0, 2.0, None, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    draws, n_failed = simulated_statistics(lambda: next(values), n_resamples=12)
    assert draws.ndim == 1
    assert n_failed == 2
    assert n_failed <= BOOTSTRAP_MAX_FAILURE_FRACTION * 12
    np.testing.assert_allclose(
        draws, [1.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )


def test_simulated_statistics_propagates_the_failure_budget():
    with pytest.raises(RuntimeError, match="parametric bootstrap"):
        simulated_statistics(lambda: None, n_resamples=5)
