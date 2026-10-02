"""Point-likelihood geometry for the affine natural-coordinate path."""

import numpy as np
import pytest

from gibbus._fit.natural_objective import (
    _natural_point_stats,
    _NaturalPointObjectiveFunction,
)
from gibbus._fit.objective import _evaluate_point_objective
from gibbus._model.coords import _build_fit_coordinate
from gibbus._model.spec import _build_model_spec
from gibbus._model.vec import _q_eval
from gibbus._observations.points import _PointObservations


def _real_line_problem(degree=4):
    x = np.array([-2.4, -1.1, -0.25, 0.2, 0.8, 1.7, 2.6])
    weights = np.array([0.7, 1.3, 0.8, 1.5, 2.0, 0.9, 0.6])
    coordinate = _build_fit_coordinate((-np.inf, np.inf), x, weights, None)
    z = coordinate.to_canonical(x)
    spec = _build_model_spec(coordinate, degree)
    stats = _natural_point_stats(z, weights, 2 * degree + 2, spec.support, False, False)
    observations = _PointObservations(stats)
    objective = _NaturalPointObjectiveFunction(
        spec, observations, (float(z.min()), float(z.max()))
    )
    return z, spec, observations, objective


def test_natural_point_hessian_is_exactly_fisher_and_psd():
    _, _, _, objective = _real_line_problem(4)
    params = objective.layout.pack(0.15, [1.1, -0.2, 0.45])
    evaluation = objective(params)

    np.testing.assert_array_equal(
        evaluation.missing_information,
        np.zeros_like(evaluation.missing_information),
    )
    np.testing.assert_allclose(
        evaluation.hessian,
        evaluation.fisher,
        rtol=0.0,
        atol=2e-14,
    )
    eigenvalues = np.linalg.eigvalsh(evaluation.hessian)
    assert eigenvalues.min() >= -2e-11 * max(1.0, np.max(np.abs(eigenvalues)))


def test_natural_point_gradient_and_hessian_match_finite_differences():
    _, _, _, objective = _real_line_problem(4)
    params = objective.layout.pack(-0.2, [1.4, 0.15, 0.55])
    analytic = objective(params)

    gradient_step = 2e-6
    finite_gradient = np.empty(params.size)
    for j in range(params.size):
        up = params.copy()
        down = params.copy()
        up[j] += gradient_step
        down[j] -= gradient_step
        finite_gradient[j] = (objective(up).nll - objective(down).nll) / (
            2.0 * gradient_step
        )
    np.testing.assert_allclose(analytic.gradient, finite_gradient, rtol=2e-6, atol=2e-7)

    hessian_step = 1e-5
    finite_hessian = np.empty_like(analytic.hessian)
    for j in range(params.size):
        up = params.copy()
        down = params.copy()
        up[j] += hessian_step
        down[j] -= hessian_step
        finite_hessian[:, j] = (objective(up).gradient - objective(down).gradient) / (
            2.0 * hessian_step
        )
    np.testing.assert_allclose(analytic.hessian, finite_hessian, rtol=3e-5, atol=3e-6)


def test_compiled_point_objective_matches_the_generic_evaluator():
    _x, _spec, observations, objective = _real_line_problem(4)
    params = objective.layout.pack(0.3, [0.8, -0.15, 0.35])
    fast = objective(params)
    generic = _evaluate_point_objective(objective.build_state(params), observations)
    assert fast.nll == pytest.approx(generic.nll, rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(fast.gradient, generic.gradient, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(fast.fisher, generic.fisher, rtol=1e-12, atol=1e-13)


def test_point_nll_is_the_weighted_potential_mean_plus_log_normalizer():
    z, spec, _, objective = _real_line_problem(4)
    weights = np.array([0.7, 1.3, 0.8, 1.5, 2.0, 0.9, 0.6])
    params = objective.layout.pack(-0.1, [1.2, 0.1, 0.3])
    state = objective.build_state(params)
    q = _q_eval(z, spec.support, state.q_poly, state.boundary_amplitudes, 0)
    expected = (
        np.dot(weights / weights.sum(), q) + state.log_Z + np.log(spec.coordinate.scale)
    )
    assert objective(params).nll == pytest.approx(expected, rel=1e-13, abs=1e-13)
