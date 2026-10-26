"""Projection QP endpoints and bounds against enumerated active faces."""

import itertools

import numpy as np
import pytest

from gibbus._fit.linear_qp import _linear_subproblem


def _face_optimum(h, g, theta, b):
    best = np.inf
    for count in range(h.shape[0] + 1):
        for subset in itertools.combinations(range(len(b)), count):
            a = b[list(subset)]
            kkt = np.block([[h, -a.T], [a, np.zeros((count, count))]])
            try:
                solution = np.linalg.solve(kkt, np.r_[h @ theta - g, np.zeros(count)])
            except np.linalg.LinAlgError:
                continue
            point, dual = solution[: len(theta)], solution[len(theta) :]
            if np.min(b @ point) < -1e-10 or (count and np.min(dual) < -1e-10):
                continue
            delta = point - theta
            best = min(best, g @ delta + 0.5 * delta @ h @ delta)
    assert np.isfinite(best)
    return best


def _problem(seed):
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(3, 3))
    h = factor @ factor.T + np.eye(3)
    g, theta = rng.normal(size=(2, 3))
    # This cone has a strict interior and several redundant constraints.
    b = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    return h, g, theta, b


@pytest.mark.parametrize("seed", range(8))
def test_projection_gap_bounds_independent_active_face_solution(seed):
    h, g, theta, b = _problem(seed)
    result = _linear_subproblem(h, g, theta, b)
    assert np.min(b @ result.params) >= -1e-12
    assert np.all(result.dual <= 0.0)
    assert 0.0 <= result.gap < 1e-8
    optimum = _face_optimum(h, g, theta, b)
    assert result.model_value - optimum <= result.gap + 1e-12
    assert result.model_value == pytest.approx(optimum, abs=1e-9)


def test_projection_does_not_depend_on_an_unrelated_lp_scheduler():
    from scipy.optimize import linprog

    # Other callers may initialize HiGHS with its default worker settings.
    # Row restoration must not request a conflicting global scheduler.
    result = linprog([1.0], bounds=[(0.0, 1.0)], method="highs")
    assert result.success
    h, g, theta, b = _problem(0)
    projection = _linear_subproblem(h, g, theta, b)
    assert np.isfinite(projection.gap)
    assert np.min(b @ projection.params) >= -1e-12


def test_unresolved_metric_cannot_supply_a_convergence_bound():
    h = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]])
    result = _linear_subproblem(h, np.array([1.0, -1.0]), np.ones(2), np.eye(2))
    assert np.isinf(result.gap)


def test_projection_is_invariant_to_large_diagonal_coordinate_changes():
    h = np.array([[2.0, 0.2, 0.1], [0.2, 3.0, 0.4], [0.1, 0.4, 1.0]])
    g, theta = np.array([1.0, -2.0, 3.0]), np.ones(3)
    b = np.eye(3)
    expected = _linear_subproblem(h, g, theta, b)
    scale = np.array([1e-10, 1.0, 1e10])
    result = _linear_subproblem(
        h * scale[:, None] * scale[None, :], g * scale, theta / scale, b * scale
    )
    np.testing.assert_allclose(result.params * scale, expected.params, atol=1e-12)
    assert result.model_value == pytest.approx(expected.model_value, abs=1e-12)
    assert np.isfinite(result.gap)


def test_unresolved_projection_defers_to_the_common_conic_kernel():
    from gibbus._fit.linear_qp import _projection_candidate

    h = np.array([[1.0, 1.0 - 1e-15], [1.0 - 1e-15, 1.0]])
    assert (
        _projection_candidate(h, np.array([1.0, -1.0]), np.ones(2), np.eye(2)) is None
    )


@pytest.mark.parametrize("gradient", [np.array([np.nan, 1.0]), np.array([np.inf, 0.0])])
def test_non_finite_model_defers_without_a_contract_error(gradient):
    from gibbus._fit.linear_qp import _projection_candidate

    h = np.eye(2)
    assert _linear_subproblem(h, gradient, np.ones(2), np.eye(2)) is None
    assert _projection_candidate(h, gradient, np.ones(2), np.eye(2)) is None


def test_projection_contract_errors_propagate():
    from gibbus._fit.linear_qp import _projection_candidate

    # A row of the wrong width is a caller defect, not a deferral.
    with pytest.raises(ValueError):
        _projection_candidate(np.eye(2), np.ones(2), np.ones(2), np.ones((1, 3)))
