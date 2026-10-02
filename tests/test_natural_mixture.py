"""Tests for mixtures of natural components: EM, joint Newton polish, verification."""

import warnings

import numpy as np
import pytest

import gibbus._fit.natural_mixture as natural_mixture_module
from gibbus._fit.natural_mixture import (
    _ComponentProblem,
    _e_step,
    _fit_natural_mixture,
    _JointMixtureObjective,
    _run_natural_em,
)
from gibbus._model.coords import _build_fit_coordinate
from gibbus._observations.intervals import _prepare_partial_interval_reducer

_REAL_LINE = (-np.inf, np.inf)


def _binned(x, width, /):
    lower = np.floor(x / width) * width
    return np.column_stack([lower, lower + width])


def _mixture_sample(seed, n, /):
    rng = np.random.default_rng(seed)
    m = int(0.4 * n)
    return np.concatenate([rng.normal(-2.0, 0.7, m), rng.gumbel(1.5, 0.8, n - m)])


@pytest.mark.parametrize("censored", [False, True])
def test_joint_mixture_objective_has_exact_derivatives(censored):
    """The joint gradient and observed Hessian ``G - M`` match finite
    differences of the mixture NLL."""
    x = _mixture_sample(7, 150)
    rows = _binned(x, 0.5) if censored else x[:, None]
    w = np.full(rows.shape[0], 1.0 / rows.shape[0])
    responsibilities = np.column_stack([x < 0.0, x >= 0.0]).astype(float)
    fit = _run_natural_em(
        _REAL_LINE,
        rows,
        4,
        False,
        False,
        w,
        responsibilities,
        max_rounds=1,
        max_em_steps=2,
    )
    problems = [
        _ComponentProblem(_REAL_LINE, rows, 4, False, False, w * responsibilities[:, k])
        for k in range(2)
    ]
    # Reuse the fitted coordinates so the parameters are meaningful.
    for problem, component in zip(problems, fit.components, strict=True):
        assert problem.coordinate == component.coordinate
    layouts = [c.layout for c in fit.components]
    joint = _JointMixtureObjective(problems, layouts, w)
    x0 = joint.join([c.params for c in fit.components], np.log(fit.weights))
    evaluation = joint(x0)
    assert -evaluation.nll == pytest.approx(
        _e_step(
            problems,
            layouts,
            [c.params for c in fit.components],
            np.log(fit.weights),
            w,
        )[0],
        abs=1e-12,
    )
    step = 1e-6
    for i in range(joint.n_params):
        e = np.zeros(joint.n_params)
        e[i] = step
        plus, minus = joint(x0 + e), joint(x0 - e)
        assert (plus.nll - minus.nll) / (2 * step) == pytest.approx(
            evaluation.gradient[i], rel=1e-5, abs=1e-7
        )
        np.testing.assert_allclose(
            (plus.gradient - minus.gradient) / (2 * step),
            evaluation.observed_hessian[:, i],
            rtol=1e-4,
            atol=2e-6,
        )


@pytest.mark.parametrize("censored", [False, True])
def test_mixture_fit_is_certified_and_beats_production(censored):
    """EM + polish converges to a certified stationary point whose likelihood
    is at least the production mixture's (which stops EM at 1e-4)."""
    warnings.simplefilter("ignore")
    from gibbus import Distribution

    x = _mixture_sample(8, 400)
    rows = _binned(x, 0.5) if censored else x
    production = Distribution().fit(rows, n_components=2, poly_degree=4, rng=0)
    reference = production._em_diagnostics["final_log_likelihood"]
    fit = _fit_natural_mixture(_REAL_LINE, rows, 2, 4, rng=0)
    assert fit.status in ("converged", "converged_approximately")
    assert fit.separator_certified
    assert fit.log_likelihood >= reference - 1e-9 * (1.0 + abs(reference))
    np.testing.assert_allclose(np.sort(fit.weights), [0.4, 0.6], atol=0.05)
    # The accepted history never decreases (EM steps and polishes are monotone).
    assert np.all(np.diff(fit.history) >= -1e-12)


def test_mixture_auto_degree_selects_per_component_then_locks():
    """A Gaussian and quartic component may retain different selected degrees."""
    from scipy.stats import gennorm

    rng = np.random.default_rng(12)
    left = rng.normal(-3.0, 0.7, 600)
    right = gennorm.rvs(beta=4.0, size=600, random_state=rng) + 3.0
    x = np.ascontiguousarray(np.concatenate([left, right]))
    responsibilities = np.zeros((x.size, 2), dtype=np.float64)
    responsibilities[: left.size, 0] = 1.0
    responsibilities[left.size :, 1] = 1.0

    fit = _fit_natural_mixture(
        _REAL_LINE,
        x,
        2,
        "auto",
        responsibilities=responsibilities,
        paths=(("direct", "raw"),),
        finalists=1,
        max_em_steps=3,
        max_rounds=1,
    )

    assert [c.spec.requested_poly_degree for c in fit.components] == [2, 4]
    assert fit.status in ("converged", "converged_approximately")
    assert fit.separator_certified


def test_em_reuses_the_evaluated_input_posterior(monkeypatch):
    """One plain EM map evaluates only its output, not its cached input."""
    x = _mixture_sample(3, 120)
    rows = x[:, None]
    w = np.full(x.size, 1.0 / x.size)
    responsibilities = np.column_stack([x < 0.0, x >= 0.0]).astype(float)
    calls = 0
    original = natural_mixture_module._e_step

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(natural_mixture_module, "_e_step", counted)
    _run_natural_em(
        _REAL_LINE,
        rows,
        4,
        False,
        False,
        w,
        responsibilities,
        polish=False,
        accelerate=False,
        max_em_steps=1,
    )

    # Initial component fits are evaluated once, then the one EM map evaluates
    # its output once.  The input posterior is carried between those points.
    assert calls == 2


def test_default_finalist_continuation_matches_restart():
    """Reusing an explored finalist is equivalent to rerunning that EM phase."""
    x = _mixture_sample(5, 240)
    kwargs = {
        "responsibilities": np.column_stack([x < 0.0, x >= 0.0]).astype(float),
        "paths": (("direct", "raw"),),
        "finalists": 1,
    }
    for degree in (4, "auto"):
        continued = _fit_natural_mixture(_REAL_LINE, x, 2, degree, **kwargs)
        restarted = _fit_natural_mixture(
            _REAL_LINE, x, 2, degree, max_em_steps=20, **kwargs
        )

        assert continued.log_likelihood == pytest.approx(
            restarted.log_likelihood, abs=2e-12
        )
        np.testing.assert_allclose(
            continued.weights, restarted.weights, rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            continued.responsibilities,
            restarted.responsibilities,
            rtol=0.0,
            atol=2e-12,
        )
        assert [c.spec.requested_poly_degree for c in continued.components] == [
            c.spec.requested_poly_degree for c in restarted.components
        ]
        assert continued.status == restarted.status
        assert continued.separator_certified == restarted.separator_certified


def test_compact_duplicate_interval_mstep_preserves_reliability_weights():
    rows = np.array(
        [
            [-2.0, -1.5],
            [-2.0, -1.5],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [-0.5, 0.0],
            [1.0, 1.5],
        ]
    )
    observation_weights = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])
    observation_weights /= observation_weights.sum()
    problem = _ComponentProblem(_REAL_LINE, rows, 4, False, False, observation_weights)
    distinct, inverse = problem.distinct_rows
    responsibility = np.linspace(0.2, 0.8, distinct.shape[0])

    expanded = problem.objective(observation_weights * responsibility[inverse])
    compact = problem.compact_objective(responsibility, observation_weights)

    np.testing.assert_allclose(
        compact.observations.intervals,
        expanded.observations.intervals,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        compact.observations.weights,
        expanded.observations.weights,
        rtol=1e-15,
        atol=1e-15,
    )
    assert compact.observations.effective_n == pytest.approx(
        expanded.observations.effective_n, rel=5e-15
    )
    assert compact.observations.n_observations == rows.shape[0]


def test_duplicate_interval_coordinate_matches_expanded_weighted_geometry():
    """Compressing exact interval duplicates does not change the fit coordinate."""
    rows = np.array(
        [
            [-3.0, -2.5],
            [-3.0, -2.5],
            [-0.5, 0.25],
            [-0.5, 0.25],
            [-0.5, 0.25],
            [1.0, 2.0],
            [3.0, 3.5],
            [3.0, 3.5],
        ]
    )
    weights = np.array([1.0, 4.0, 2.0, 7.0, 3.0, 5.0, 6.0, 2.0])
    weights /= weights.sum()

    problem = _ComponentProblem(_REAL_LINE, rows, 4, False, False, weights)
    mid = 0.5 * (rows[:, 0] + rows[:, 1])
    width = rows[:, 1] - rows[:, 0]
    expanded = _build_fit_coordinate(_REAL_LINE, mid, weights, width)

    assert problem.coordinate.support_kind == expanded.support_kind
    assert problem.coordinate.center == pytest.approx(expanded.center, rel=0.0, abs=0.0)
    assert problem.coordinate.scale == pytest.approx(expanded.scale, rel=2e-15, abs=0.0)
    assert problem.coordinate.canonical_support == expanded.canonical_support


def test_real_line_mass_only_estep_matches_general_interval_reducer():
    """The finite real-line E-step specialization returns the generic masses."""
    rows = np.array(
        [
            [-2.0, -1.5],
            [-2.0, -1.5],
            [-0.75, -0.25],
            [0.0, 0.5],
            [0.0, 0.5],
            [1.0, 1.75],
            [2.25, 3.0],
        ]
    )
    weights = np.full(rows.shape[0], 1.0 / rows.shape[0])
    problem = _ComponentProblem(_REAL_LINE, rows, 4, False, False, weights)
    distinct, inverse = problem.distinct_rows
    responsibility = np.linspace(0.3, 0.9, distinct.shape[0])
    objective, result = problem.fit_compact(responsibility, weights, None)
    state = objective.build_state(result.params)

    fast, _, _ = problem.row_statistics(
        state, objective.layout, moments=False, distinct=True
    )
    generic, _, _, _ = _prepare_partial_interval_reducer(state).reduce_many(distinct)
    np.testing.assert_allclose(fast, generic, rtol=0.0, atol=3e-13)

    expanded, _, _ = problem.row_statistics(state, objective.layout, moments=False)
    np.testing.assert_allclose(expanded, fast[inverse], rtol=0.0, atol=0.0)


def test_public_mixture_responsibilities_expand_duplicate_rows():
    x = _mixture_sample(23, 300)
    rows = _binned(x, 0.5)
    fit = _fit_natural_mixture(_REAL_LINE, rows, 2, 4, rng=0)
    assert fit.responsibilities.shape == (rows.shape[0], 2)
    np.testing.assert_allclose(fit.responsibilities.sum(axis=1), 1.0, atol=1e-14)
    _, inverse = np.unique(rows, axis=0, return_inverse=True)
    for group in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == group)
        if members.size > 1:
            expected = np.repeat(
                fit.responsibilities[members[0]][None, :], members.size, axis=0
            )
            np.testing.assert_allclose(
                fit.responsibilities[members], expected, rtol=0.0, atol=0.0
            )


def test_joint_polish_uses_compiled_callback_newton(monkeypatch):
    """The coupled observed-likelihood polish has no Python Newton traversal."""
    from gibbus._fit import _conic_kernels

    x = _mixture_sample(29, 220)
    calls = 0
    original = _conic_kernels.solve_callback_newton

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_conic_kernels, "solve_callback_newton", counted)
    fit = _fit_natural_mixture(
        _REAL_LINE,
        x,
        2,
        4,
        rng=0,
        paths=(("direct", "raw"),),
        finalists=1,
        max_em_steps=4,
        max_rounds=1,
    )
    assert calls >= 1
    assert fit.status in ("converged", "converged_approximately")
