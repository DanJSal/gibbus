"""Shared fitting deadlines and estimator-state preservation."""

from contextvars import copy_context

import numpy as np
import pytest

from gibbus import Distribution, FitTimeoutError
from gibbus._fit import budget


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(budget.time, "monotonic", lambda: now[0])
    return now


def test_budget_is_shared_and_cannot_be_reset_by_nested_work(clock):
    with pytest.raises(FitTimeoutError) as caught:
        with budget._fit_time_budget(2.0):
            clock[0] += 1.0
            with budget._fit_time_budget(10.0):
                clock[0] += 1.1
                budget._check_fit_time("retry")
    assert caught.value.max_fit_seconds == 2.0
    assert caught.value.elapsed_seconds == pytest.approx(2.1)
    assert caught.value.stage == "retry"
    budget._check_fit_time()


def test_shorter_nested_budget_restores_outer_deadline(clock):
    with budget._fit_time_budget(10.0):
        with pytest.raises(FitTimeoutError):
            with budget._fit_time_budget(1.0):
                clock[0] += 2.0
        budget._check_fit_time()
    budget._check_fit_time()


def test_unlimited_fit_does_not_start_or_read_a_clock(monkeypatch):
    def unexpected():
        raise AssertionError("unlimited fit read the clock")

    monkeypatch.setattr(budget.time, "monotonic", unexpected)
    with budget._fit_time_budget(None):
        budget._check_fit_time()


def test_context_does_not_modify_an_independent_execution_context(clock):
    independent = copy_context()
    with budget._fit_time_budget(1.0):
        clock[0] += 2.0
        independent.run(budget._check_fit_time)
        with pytest.raises(FitTimeoutError):
            budget._check_fit_time()
        clock[0] = 100.0


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, 10**400])
def test_invalid_budget_rejected(value):
    with pytest.raises(ValueError, match="max_fit_seconds"):
        Distribution().fit([0.0, 1.0, 2.0], max_fit_seconds=value)


@pytest.mark.parametrize("value", [True, "3", object()])
def test_non_numeric_budget_rejected(value):
    with pytest.raises(TypeError, match="max_fit_seconds"):
        Distribution().fit([0.0, 1.0, 2.0], max_fit_seconds=value)


def test_timeout_does_not_replace_existing_fit_and_next_fit_is_unlimited(
    monkeypatch, clock
):
    import gibbus._api.distribution as api

    samples = np.random.default_rng(31).normal(size=100)
    fitted = Distribution().fit(samples, poly_degree=2)
    before = fitted.logpdf(samples).copy()
    original = api._run_fit_request

    def expire(request):
        clock[0] += 2.0
        budget._check_fit_time("component-count screening")
        raise AssertionError("timeout did not escape")

    monkeypatch.setattr(api, "_run_fit_request", expire)
    with pytest.raises(FitTimeoutError, match="component-count screening"):
        fitted.fit(samples, n_components="auto", max_fit_seconds=1.0)
    np.testing.assert_array_equal(fitted.logpdf(samples), before)
    monkeypatch.setattr(api, "_run_fit_request", original)
    assert fitted.fit(samples, poly_degree=2) is fitted


def test_expired_deadline_prevents_starting_a_component_solve(clock):
    from gibbus._fit.basis_newton import _solve_basis

    with pytest.raises(FitTimeoutError, match="component solve"):
        with budget._fit_time_budget(1.0):
            clock[0] += 2.0
            _solve_basis(None)


def test_timeout_is_not_a_recoverable_numeric_failure():
    from gibbus._defaults import NUMERIC_FAILURES

    assert not isinstance(FitTimeoutError(1, 2, "refinement"), NUMERIC_FAILURES)


def test_timeout_exception_round_trips_through_pickle():
    import pickle

    original = FitTimeoutError(3.0, 3.2, "Newton run")
    restored = pickle.loads(pickle.dumps(original))
    assert str(restored) == str(original)
    assert restored.max_fit_seconds == 3.0
    assert restored.elapsed_seconds == 3.2
    assert restored.stage == "Newton run"


def test_auto_search_does_not_swallow_a_timed_out_screen(monkeypatch, clock):
    import gibbus._fit.basis_mixture as mixture

    original = mixture._MixtureProblem.m_step
    calls = []

    def expire(self, *args):
        calls.append(self.k)
        clock[0] += 2.0
        return original(self, *args)

    monkeypatch.setattr(mixture._MixtureProblem, "m_step", expire)
    rng = np.random.default_rng(91)
    x = np.r_[rng.normal(-3, 0.5, 100), rng.normal(3, 0.5, 100)]
    fitted = Distribution()
    with pytest.raises(FitTimeoutError, match="mixture M-step"):
        fitted.fit(x, n_components="auto", max_fit_seconds=1)
    assert len(calls) == 1
    with pytest.raises(RuntimeError):
        fitted.logpdf(x)
