"""Tests for interval-censored fits with the natural conic solver."""

import numpy as np
import pytest
from scipy.integrate import quad

from gibbus._fit.conic_newton import _solve_natural_conic
from gibbus._fit.natural_objective import (
    _fit_natural_conic_intervals,
    _interval_nll_lower_bound,
    _natural_interval_start,
    _prepare_natural_interval_objective,
    _safeguarded_metric,
)

_REAL_LINE = (-np.inf, np.inf)

# Certified optimal NLLs; each was cross-checked, when recorded, against an
# independent local fitter that could only match or sit above it.
_BASELINE_NLL = {
    "binned_4": 3.0767408655457515,
    "binned_6": 3.07344762542732,
    "right_censored": 1.1875534035434059,
    "binned_beta": 2.091481127795866,
}


def _assert_baseline(result, key, /):
    assert result.status in ("converged", "converged_approximately")
    assert result.final_separation.feasible
    slack = 2.0 * max(1e-10, result.final_decrease_bound)
    assert result.objective_value == pytest.approx(_BASELINE_NLL[key], abs=slack)


def _binned(x, width, /):
    lower = np.floor(x / width) * width
    return np.column_stack([lower, lower + width])


def _right_censored(x, c, /):
    seen = x <= c
    return np.column_stack([np.where(seen, x, c), np.where(seen, x, np.inf)])


def test_interval_objective_is_the_bin_likelihood_with_exact_derivatives():
    """The NLL is ``-sum w log P(bin)`` of the model density; the gradient
    and observed Hessian ``F - C`` match finite differences."""
    x = np.random.default_rng(1).gamma(3.0, 1.0, 200)
    rows = _binned(x, 0.5)
    objective, fit = _fit_natural_conic_intervals(_REAL_LINE, rows, 4)
    layout = objective.layout
    theta = fit.params
    evaluation = objective(theta)
    state = objective.build_state(theta)
    observations = objective.observations
    masses = [
        quad(state.pdf, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200)[0]
        for lo, hi in observations.intervals
    ]
    expected = -float(np.dot(observations.weights, np.log(masses)))
    assert evaluation.nll == pytest.approx(expected, abs=1e-11)
    step = 1e-6
    for i in range(layout.n_params):
        e = np.zeros(layout.n_params)
        e[i] = step
        plus, minus = objective(theta + e), objective(theta - e)
        assert (plus.nll - minus.nll) / (2 * step) == pytest.approx(
            evaluation.gradient[i], rel=1e-5, abs=1e-7
        )
        np.testing.assert_allclose(
            (plus.gradient - minus.gradient) / (2 * step),
            evaluation.observed_hessian[:, i],
            rtol=1e-4,
            atol=1e-6,
        )


def test_safeguarded_metric_is_exact_newton_or_saddle_free():
    """Positive definite observed Hessian: returned unchanged.  Indefinite:
    the negative relative eigenvalues are reflected, so the metric keeps
    the observed Hessian's eigenvectors (relative to F) and magnitudes."""
    import scipy.linalg

    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 5))
    fisher = a @ a.T + 5 * np.eye(5)
    b = rng.normal(size=(5, 2))
    small = 0.1 * b @ b.T
    metric, smallest = _safeguarded_metric(fisher, small)
    assert smallest > 0.0
    np.testing.assert_allclose(metric, fisher - small)
    large = 50.0 * b @ b.T
    metric, smallest = _safeguarded_metric(fisher, large)
    assert smallest < 0.0
    observed = scipy.linalg.eigh(fisher - large, fisher, eigvals_only=True)
    reflected = scipy.linalg.eigh(metric, fisher, eigvals_only=True)
    np.testing.assert_allclose(
        np.sort(np.maximum(np.abs(observed), 1e-3)), reflected, rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize("degree", [4, 6])
def test_binned_fit_reaches_its_certified_optimum(degree):
    x = np.random.default_rng(3).normal(0.2, 1.3, 400)
    rows = _binned(x, 0.25)
    _, result = _fit_natural_conic_intervals(_REAL_LINE, rows, degree)
    _assert_baseline(result, f"binned_{degree}")


def test_right_censored_amplitude_fit_reaches_its_optimum():
    """Half-line with a boundary amplitude and right censoring."""
    rng = np.random.default_rng(4)
    rows = _right_censored(rng.gamma(3.0, 1.0, 300), rng.exponential(5.0, 300))
    support = (0.0, np.inf)
    objective = _prepare_natural_interval_objective(support, rows, 4, True, False, None)
    start, blocks = _natural_interval_start(objective)
    result = _solve_natural_conic(objective, initial=start, initial_blocks=blocks)
    _assert_baseline(result, "right_censored")


def test_right_censored_exponential_starts_on_zero_boundary_face():
    """Near-exponential censored data should reach the exact a=0 face
    directly instead of crawling toward it through hundreds of line-search
    evaluations on the degenerate positive-amplitude cone.
    """
    edges = np.arange(0.0, 2.0001, 0.25)
    rows = np.vstack([
        np.column_stack([edges[:-1], edges[1:]]),
        [2.0, np.inf],
    ])
    weights = np.r_[
        np.exp(-edges[:-1]) - np.exp(-edges[1:]),
        np.exp(-2.0),
    ]
    objective = _prepare_natural_interval_objective(
        (0.0, np.inf), rows, 4, True, False, weights
    )
    start, blocks = _natural_interval_start(objective)
    result = _solve_natural_conic(objective, initial=start, initial_blocks=blocks)

    assert result.status in ("converged", "converged_approximately")
    assert result.final_separation.feasible
    assert not result.lower_amplitude_active
    assert result.params[objective.layout.lower_a_index] == 0.0
    assert result.effective_curvature_degree == 0
    assert result.objective_evaluations <= 50
    assert result.objective_value == pytest.approx(2.0655927563831127, abs=3e-10)


def test_nonparametric_bound_is_the_saturated_multinomial_for_disjoint_bins():
    x = np.random.default_rng(5).normal(size=300)
    rows = _binned(x, 0.5)
    objective = _prepare_natural_interval_objective(_REAL_LINE, rows, 4, False, False, None)
    w = objective.observations.weights
    exact = -float(np.sum(w * np.log(w)))
    bound = _interval_nll_lower_bound(objective.observations)
    assert bound <= exact
    assert bound == pytest.approx(exact, abs=1e-12)


def test_nonparametric_bound_holds_for_overlapping_rows():
    rng = np.random.default_rng(6)
    x = rng.logistic(size=200)
    rows = np.column_stack([x - rng.exponential(0.5, 200), x + rng.exponential(0.5, 200)])
    _, result = _fit_natural_conic_intervals(_REAL_LINE, rows, 6)
    objective = _prepare_natural_interval_objective(_REAL_LINE, rows, 6, False, False, None)
    bound = _interval_nll_lower_bound(objective.observations)
    assert bound is not None and bound < result.objective_value
    exact_rows = np.column_stack([x, x])
    exact_objective = _prepare_natural_interval_objective(
        _REAL_LINE, exact_rows, 4, False, False, None
    )
    assert _interval_nll_lower_bound(exact_objective.observations) is None


def test_saturated_fit_is_certified_by_the_nonparametric_bound():
    """Four occupied bins, six parameters: the likelihood only approaches its
    supremum, so Newton cannot certify stationarity, but the fit comes within
    1e-7 of the nonparametric bound, which certifies it globally."""
    x = np.random.default_rng(270902).normal(0.2, 1.3, 30)
    objective, result = _fit_natural_conic_intervals(_REAL_LINE, _binned(x, 1.5), 6)
    assert objective.observations.n_unique == 4
    assert result.status == "converged_approximately"
    gap = result.objective_value - objective.nll_lower_bound
    assert 0.0 <= gap <= 1e-7
    assert result.final_decrease_bound == pytest.approx(gap)


def test_interval_start_is_log_concave_and_the_fit_reaches_its_optimum():
    """The pseudo-statistics start lies in the full-curvature cone."""
    rng = np.random.default_rng(270908)
    x = rng.beta(2.0, 3.0, 120)
    rows = _binned(x, 0.1)
    objective = _prepare_natural_interval_objective((0.0, 1.0), rows, 8, True, True, None)
    start, _ = _natural_interval_start(objective)
    layout = objective.layout
    candidate = layout.build_candidate(start)
    lower, upper = layout.support
    z = np.linspace(lower + 1e-6, upper - 1e-6, 4001)
    assert candidate.q_d2_full(z).min() >= -1e-9
    _, result = _fit_natural_conic_intervals((0.0, 1.0), rows, 8, True, True, None)
    _assert_baseline(result, "binned_beta")


@pytest.mark.parametrize("n_components", [1, 2])
def test_duplicate_rows_are_reduced_once_without_changing_the_fit(n_components):
    """Binned data repeat rows; merging them must be exact.

    Nudging each duplicate by a few ulps defeats the merge, so the two fits
    agree only if the merged objective equals the row-by-row one.
    """
    from gibbus import Distribution

    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(0.0, 1.0, 1500), rng.normal(5.0, 0.7, 1000)])
    lo = np.floor(x / 0.25) * 0.25
    rows = np.column_stack([lo, lo + 0.25])
    assert np.unique(rows, axis=0).shape[0] < rows.shape[0] // 10
    jitter = np.arange(rows.shape[0]) % 7 * 4.0 * np.finfo(float).eps
    nudged = rows + jitter[:, None] * np.maximum(1.0, np.abs(rows))

    merged = Distribution().fit(rows, n_components=n_components, poly_degree=4, rng=0)
    separate = Distribution().fit(nudged, n_components=n_components, poly_degree=4, rng=0)
    grid = np.linspace(-3.0, 8.0, 23)
    np.testing.assert_allclose(merged.logpdf(grid), separate.logpdf(grid), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(merged.weights, separate.weights, rtol=0.0, atol=1e-7)
