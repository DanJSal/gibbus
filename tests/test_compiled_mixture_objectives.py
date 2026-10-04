"""Fused compiled mixture objectives agree with independent checks."""

from dataclasses import replace

import numpy as np
import pytest
from gibbus._fit._shared_mixture_kernels import (
    JointMixtureObjective,
    SharedMStepObjective,
)
from scipy.special import xlogy

from gibbus._fit.natural_mixture import (
    _compiled_component,
    _CompiledJointMixture,
    _degree_probe_problems,
    _joint_component,
    _mixture_map,
    _mixture_newton_options,
    _run_natural_em,
    _solve_mixture_face,
)
from gibbus._fit.natural_objective import _safeguarded_metric

CASES = (
    ((0.0, 1.0), "points", 2),
    ((0.0, 1.0), "binned", 2),
    ((0.0, 1.0), "censored", 2),
    ((0.0, 1.0), "points", 3),
    ((0.0, np.inf), "points", 2),
    ((0.0, np.inf), "censored", 2),
    ((-np.inf, np.inf), "binned", 2),
)
_FITS = {}


def _sample(support, kind, count):
    rng = np.random.default_rng(4)
    x = np.r_[rng.beta(2, 8, 80), rng.beta(8, 2, 120)]
    if np.isinf(support[1]):
        x = 3.0 * x
    if kind == "points":
        rows = x[:, None]
    elif kind == "binned":
        rows = np.column_stack((np.floor(x * 30) / 30, np.ceil(x * 30) / 30))
    else:
        rows = np.column_stack((x, np.where(x > np.quantile(x, 0.85), support[1], x)))
        rows[:20, 0] = support[0]
    edges = np.quantile(x, np.linspace(0.0, 1.0, count + 1)[1:-1])
    label = np.searchsorted(edges, x)
    r = (label[:, None] == np.arange(count)).astype(float)
    w = np.linspace(0.8, 1.2, len(x))
    return rows, r, w / w.sum()


def _case(case):
    """Return a fitted mixture and its joint geometry, cached."""
    if case not in _FITS:
        support, kind, count = case
        rows, r, w = _sample(support, kind, count)
        lower, upper = np.isfinite(support[0]), np.isfinite(support[1])
        fit = _run_natural_em(
            support, rows, count, lower, upper, w, r, max_em_steps=3, max_rounds=1
        )
        degrees = tuple(c.spec.requested_poly_degree for c in fit.components)
        problems, layouts, params = _degree_probe_problems(fit, rows, w, degrees)
        _FITS[case] = (fit, rows, r, w, problems, layouts, params)
    return _FITS[case]


def _close(actual, expected, rtol=1e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    scale = max(1.0, float(np.max(np.abs(expected), initial=0.0)))
    assert float(np.max(np.abs(actual - expected), initial=0.0)) <= rtol * scale


def _full_face(mapping, layouts, n_logits=0):
    active = tuple(index is not None for index in mapping.shared_parameter_indices)
    return mapping.face(
        [layout.curvature_degree for layout in layouts], active, n_logits
    )


def _lift_amplitudes(compiled, x):
    """Move shared amplitudes off zero so central differences stay valid."""
    x = x.copy()
    for index in compiled.natural_map.shared_parameter_indices:
        if index is not None:
            x[index] = max(x[index], 0.25)
    return x


def _try_gradient(compiled, x):
    try:
        evaluation = compiled(x)
    except FloatingPointError:
        return None
    return evaluation.nll, evaluation.gradient


@pytest.mark.parametrize("case", CASES)
def test_joint_derivatives_match_finite_differences(case):
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    x = _lift_amplitudes(compiled, compiled.join(params, np.log(fit.weights)))
    evaluation = compiled(x)
    observed = evaluation.fisher - evaluation.missing_information
    scale = max(1.0, float(np.max(np.abs(observed))))
    checked = 0
    for i in range(x.size):
        step = 1e-5 * max(1.0, abs(x[i]))
        plus, minus = x.copy(), x.copy()
        plus[i] += step
        minus[i] -= step
        upper, lower = _try_gradient(compiled, plus), _try_gradient(compiled, minus)
        if upper is None or lower is None:
            # A zero leading tail coefficient is not normalizable on one side.
            continue
        checked += 1
        slope = (upper[0] - lower[0]) / (2.0 * step)
        assert slope == pytest.approx(evaluation.gradient[i], abs=1e-6 * scale)
        column = (upper[1] - lower[1]) / (2.0 * step)
        assert np.max(np.abs(column - observed[:, i])) <= 1e-5 * scale
    assert checked >= x.size // 2


@pytest.mark.parametrize("case", [case for case in CASES if case[1] == "points"])
def test_point_derivatives_match_the_score_formula(case):
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    x = compiled.join(params, np.log(fit.weights))
    evaluation, responsibilities, centered = compiled.evaluate(x, rows=True)
    count = len(layouts)
    scores = np.zeros((w.size, count, x.size))
    for k, indices in enumerate(compiled.natural_map.local_indices):
        scores[:, k, indices] = centered[k]
    mixture_weights = np.asarray(fit.weights)
    logits = slice(compiled.natural_map.n_params, x.size)
    scores[:, :, logits] = mixture_weights[:-1]
    for k in range(count - 1):
        scores[:, k, compiled.natural_map.n_params + k] -= 1.0
    means = np.einsum("nk,nkp->np", responsibilities, scores)
    second = np.einsum("nk,nkp,nkq->npq", responsibilities, scores, scores)
    missing = np.einsum("n,npq->pq", w, second - means[:, :, None] * means[:, None])
    _close(evaluation.gradient, w @ means, rtol=1e-10)
    _close(evaluation.missing_information, missing, rtol=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_posterior_matches_the_joint_rows(case):
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    log_weights = np.log(fit.weights)
    x = compiled.join(params, log_weights)
    evaluation, responsibilities, _ = compiled.evaluate(x, rows=True)
    log_likelihood, posterior = compiled.posterior(params, log_weights)
    # The polish guard compares E-step and polish log likelihoods directly.
    assert log_likelihood == -evaluation.nll
    np.testing.assert_array_equal(posterior, responsibilities)
    _close(posterior.sum(axis=1), np.ones(w.size), rtol=1e-13)


@pytest.mark.parametrize("case", CASES)
def test_joint_likelihood_satisfies_the_em_identity(case):
    """``log L = sum_k m_k (log pi_k - NLL_k) - sum_i w_i sum_k r_ik log r_ik``.

    ``NLL_k`` is the single-component objective under the normalized weights
    ``w r_k / m_k``.  The identity holds only at the exact posterior, so it
    checks the compiled likelihood and responsibilities together.
    """
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    log_weights = np.log(fit.weights)
    log_likelihood, posterior = compiled.posterior(params, log_weights)
    masses = w @ posterior
    expected = -float(np.sum(w[:, None] * xlogy(posterior, posterior)))
    for k, (problem, theta) in enumerate(zip(problems, params, strict=True)):
        nll = problem.objective(w * posterior[:, k])(theta).nll
        expected += masses[k] * (log_weights[k] - nll)
    assert log_likelihood == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_single_component_joint_matches_the_component_objective():
    _, _, _, w, problems, layouts, params = _case(((0.0, 1.0), "binned", 2))
    compiled = _CompiledJointMixture(problems[:1], layouts[:1], w)
    x = compiled.join(params[:1], np.zeros(1))
    actual = compiled(x)
    expected = problems[0].objective(w)(params[0])
    assert actual.nll == pytest.approx(expected.nll, rel=1e-12, abs=1e-12)
    _close(actual.gradient, expected.gradient, rtol=1e-10)
    _close(actual.fisher, expected.fisher, rtol=1e-10)
    _close(actual.missing_information, expected.missing_information, rtol=1e-10)


def _restricted(face, vector, matrices):
    free = face.free_indices
    return vector[free], *(matrix[np.ix_(free, free)] for matrix in matrices)


def _assert_face_evaluation(run, face, nll, gradient, fisher, missing):
    gradient, fisher, missing = _restricted(face, gradient, (fisher, missing))
    metric, smallest = _safeguarded_metric(fisher, missing)
    assert run.evaluation.nll == pytest.approx(nll, rel=1e-12, abs=1e-12)
    _close(run.evaluation.gradient, gradient)
    _close(run.evaluation.fisher, fisher)
    _close(run.evaluation.missing_information, missing)
    _close(run.evaluation.hessian, metric, rtol=1e-9)
    assert run.evaluation.smallest_curvature == pytest.approx(
        smallest, rel=1e-8, abs=1e-12
    )


@pytest.mark.parametrize("case", CASES)
def test_joint_face_restricts_the_full_evaluation(case):
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    face = _full_face(compiled.natural_map, layouts, len(layouts) - 1)
    x = compiled.join(params, np.log(fit.weights))
    options = replace(_mixture_newton_options({}), max_iterations=0)
    run, _ = _solve_mixture_face(compiled.solver(face), face, x, None, options)
    full = compiled(face.expand(run.params))
    _assert_face_evaluation(
        run,
        face,
        full.nll,
        full.gradient,
        full.fisher,
        full.missing_information,
    )


@pytest.mark.parametrize("case", [case for case in CASES if np.isfinite(case[0][0])])
def test_shared_m_step_matches_the_component_objectives(case):
    _, _, r, w, problems, layouts, params = _case(case)
    objectives = [p.objective(w * r[:, k]) for k, p in enumerate(problems)]
    masses = w @ r
    mapping = _mixture_map(problems, layouts)
    compiled = SharedMStepObjective(
        [
            (kind, mass, inputs)
            for (kind, inputs), mass in zip(
                map(_compiled_component, objectives), masses, strict=True
            )
        ]
    )
    face = _full_face(mapping, layouts)
    n_free = len(face.free_indices)

    def solve(*args):
        return compiled.solve(face.component_columns, n_free, *args)

    options = replace(_mixture_newton_options({}), max_iterations=0)
    run, _ = _solve_mixture_face(solve, face, mapping.pack(params), None, options)
    size = mapping.n_params
    total_nll, total_gradient = 0.0, np.zeros(size)
    total_fisher, total_missing = np.zeros((size, size)), np.zeros((size, size))
    for (nll, gradient, fisher, missing, _), objective, theta, mass, indices in zip(
        compiled.evaluate(face.component_columns, n_free, run.params),
        objectives,
        mapping.expand(face.expand(run.params)),
        masses,
        mapping.local_indices,
        strict=True,
    ):
        expected = objective(theta)
        assert nll == pytest.approx(expected.nll, rel=1e-12, abs=1e-12)
        _close(gradient, expected.gradient)
        _close(fisher, expected.fisher)
        _close(missing, expected.missing_information)
        block = np.ix_(indices, indices)
        total_nll += mass * expected.nll
        total_gradient[indices] += mass * expected.gradient
        total_fisher[block] += mass * expected.fisher
        total_missing[block] += mass * expected.missing_information
    _assert_face_evaluation(
        run, face, total_nll, total_gradient, total_fisher, total_missing
    )


def test_compiled_solves_keep_the_shared_face_optimum():
    case = ((0.0, 1.0), "binned", 2)
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    face = _full_face(compiled.natural_map, layouts, len(layouts) - 1)
    x = compiled.join(params, np.log(fit.weights))
    solved = _solve_mixture_face(
        compiled.solver(face), face, x, None, _mixture_newton_options({})
    )
    run, _ = solved
    assert run.status in ("converged", "converged_approximately")
    assert run.evaluation.nll <= compiled(x).nll + 1e-12
    full = face.expand(run.params)
    for index in compiled.natural_map.shared_parameter_indices:
        assert full[index] >= 0.0


def test_invalid_start_fails_the_face_without_fallback():
    case = ((0.0, 1.0), "points", 2)
    fit, _, _, w, problems, layouts, params = _case(case)
    compiled = _CompiledJointMixture(problems, layouts, w)
    x = compiled.join(params, np.log(fit.weights))
    bad = x.copy()
    bad[compiled.natural_map.local_indices[0][layouts[0].gamma_index]] = np.inf
    with pytest.raises(FloatingPointError):
        compiled(bad)
    face = _full_face(compiled.natural_map, layouts, len(layouts) - 1)

    def failing(*args):
        raise FloatingPointError("start is not a valid evaluation point")

    assert (
        _solve_mixture_face(failing, face, x, None, _mixture_newton_options({})) is None
    )


def test_joint_kernel_rejects_inconsistent_inputs():
    case = ((0.0, 1.0), "binned", 2)
    _, _, _, w, problems, layouts, _ = _case(case)
    components = [
        _joint_component(problem, layout)
        for problem, layout in zip(problems, layouts, strict=True)
    ]
    kind, inputs, (finite, adaptive, whole) = components[0]
    broken = [(kind, inputs, (finite[:-1], adaptive, whole)), components[1]]
    distinct = problems[0].distinct_rows
    weights = np.bincount(distinct[1], weights=w, minlength=distinct[0].shape[0])
    with pytest.raises(ValueError):
        JointMixtureObjective(broken, weights)
    with pytest.raises(ValueError):
        JointMixtureObjective([], weights)
