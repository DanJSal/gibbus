"""Span, derivative and likelihood invariants of runtime fitting coordinates."""

import numpy as np
import pytest

from gibbus import BasisSpec, Polynomial
from gibbus._fit.basis_objective import (
    _BasisIntervalObjective,
    _BasisPointObjective,
    _interval_rows,
    _point_statistics,
)
from gibbus._fit.coordinates import _polynomial_coordinates, _prepare_coordinates
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_normalizer import _NormalizerContext


@pytest.mark.parametrize(
    "support,family",
    [
        ((-2, 3), "chebyshev"),
        ((0, np.inf), "laguerre"),
        ((-np.inf, 3), "laguerre"),
        ((-np.inf, np.inf), "hermite"),
    ],
)
def test_coordinate_maps_and_all_derivatives_preserve_the_polynomial(support, family):
    lower, upper = support
    data = np.linspace(max(lower, -1), min(upper, 2), 100)
    canonical = _build_basis(BasisSpec(support=support, bulk_degree=6), data)
    numerical = _polynomial_coordinates(canonical, data)
    plan = numerical.polynomial_coordinates
    assert plan.family == family
    beta = np.random.default_rng(42).normal(size=canonical.n_coefficients)
    theta = plan.to_canonical(beta)
    np.testing.assert_allclose(plan.from_canonical(theta), beta, rtol=2e-11, atol=2e-11)
    xs = np.linspace(max(lower, -1), min(upper, 2), 17)
    original = _evaluate_basis(canonical, xs, max_order=4)
    changed = _evaluate_basis(numerical, xs, max_order=4)
    for order in range(5):
        difference = (
            changed.derivative(order) @ beta - original.derivative(order) @ theta
        )
        if order == 0:
            difference -= difference[0]
        np.testing.assert_allclose(difference, 0, atol=2e-9, rtol=0)


@pytest.mark.parametrize("tail", [Polynomial(1), Polynomial(2), None])
def test_gated_recurrence_preserves_the_complete_requested_span(tail):
    data = np.linspace(-2, 2, 101)
    canonical = _build_basis(
        BasisSpec(bulk_degree=4, lower_tail=tail, upper_tail=tail, expansion="full"),
        data,
    )
    numerical = _polynomial_coordinates(canonical, data)
    plan = numerical.polynomial_coordinates
    assert np.linalg.matrix_rank(plan.transform) == canonical.n_coefficients
    beta = np.random.default_rng(19).normal(size=canonical.n_coefficients)
    theta = plan.to_canonical(beta)
    points = np.array([-12, -4, -1, 0, 1, 4, 12.0])
    original = _evaluate_basis(canonical, points, max_order=4)
    changed = _evaluate_basis(numerical, points, max_order=4)
    for order in range(5):
        expected = original.derivative(order) @ theta
        actual = changed.derivative(order) @ beta
        if order == 0:
            actual = actual - actual[3]
            expected = expected - expected[3]
        np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=2e-10)


@pytest.mark.parametrize(
    "support", [(-2, 3), (0, np.inf), (-np.inf, 3), (-np.inf, np.inf)]
)
@pytest.mark.parametrize("interval", [False, True])
def test_point_and_interval_likelihoods_have_the_same_coordinate_contract(
    support, interval
):
    lo, hi = support
    data = np.linspace(max(lo, -1) + 0.1, min(hi, 2) - 0.1, 30)
    model = _build_basis(BasisSpec(support=support, bulk_degree=4), data)
    context = _NormalizerContext(model)
    if interval:
        rows = _interval_rows(support, np.column_stack([data - 0.05, data + 0.05]))
        objective = _BasisIntervalObjective(context, rows)
    else:
        objective = _BasisPointObjective(context, _point_statistics(model, data))
    theta = np.zeros(model.n_coefficients)
    theta[[c.label for c in model.columns].index("x^2")] = 0.5
    binding = _prepare_coordinates(objective, None, theta)
    first, second = objective(theta), binding.objective(binding.start)
    assert second.nll == pytest.approx(first.nll, abs=2e-11)
    np.testing.assert_allclose(
        second.gradient, binding.transform.T @ first.gradient, atol=2e-9, rtol=2e-9
    )
    np.testing.assert_allclose(
        second.fisher,
        binding.transform.T @ first.fisher @ binding.transform,
        atol=2e-8,
        rtol=2e-8,
    )


def test_coupled_point_statistics_are_recomputed_and_shared_logs_stay_shared():
    from gibbus._fit.basis_mixture import _MixtureRows, _mstep_objective

    data = np.linspace(0.2, 4.0, 60)
    weights = np.ones(len(data)) / len(data)
    models = [
        _build_basis(
            BasisSpec(support=(0, np.inf), bulk_degree=4, log_lower=True), data
        )
        for _ in range(2)
    ]
    objective = _mstep_objective(
        models, _MixtureRows(weights, x=data), np.full((len(data), 2), 0.5)
    )
    blocks = objective.blocks
    theta = np.zeros(blocks.n)
    for k, model in enumerate(models):
        labels = [c.label for c in model.columns]
        theta[blocks.index[k][labels.index("x^2")]] = 0.5
        theta[blocks.index[k][labels.index("logL")]] = 0.2
    binding = _prepare_coordinates(objective, blocks, theta)
    counts = np.bincount(np.concatenate(blocks.index), minlength=blocks.n)
    for index in np.flatnonzero(counts > 1):
        np.testing.assert_array_equal(binding.transform[index], np.eye(blocks.n)[index])
        np.testing.assert_array_equal(
            binding.transform[:, index], np.eye(blocks.n)[:, index]
        )
    first, second = objective(theta), binding.objective(binding.start)
    assert second.nll == pytest.approx(first.nll, abs=2e-11)
    np.testing.assert_allclose(
        second.gradient, binding.transform.T @ first.gradient, atol=2e-9, rtol=2e-9
    )


@pytest.mark.parametrize("projection", [False, True])
def test_common_controller_preserves_the_gaussian_mle_with_either_qp_kernel(
    monkeypatch, projection
):
    from gibbus._fit import linear_qp
    from gibbus._fit.basis_newton import _solve_basis

    if not projection:
        monkeypatch.setattr(linear_qp, "_projection_candidate", lambda *args: None)
    data = np.random.default_rng(136).normal(2.0, 1.7, 200)
    model = _build_basis(BasisSpec(bulk_degree=2), data)
    objective = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )
    result = _solve_basis(objective)
    assert result.status == "converged"
    assert result.final_separation.status == "nonnegative"
    expected = 0.5 * (np.log(2 * np.pi * np.var(data)) + 1)
    assert result.objective_value == pytest.approx(expected, abs=2e-10)


def test_failed_coordinate_accounting_retries_the_same_gaussian_family(monkeypatch):
    from gibbus._fit.basis_newton import _solve_basis
    from gibbus._fit.coordinates import _CoordinateBinding, _CoordinateMismatch

    restore = _CoordinateBinding.restore

    def reject_recurrence(binding, result, options):
        if binding.original is not binding.objective:
            raise _CoordinateMismatch("injected representation accounting failure")
        return restore(binding, result, options)

    monkeypatch.setattr(_CoordinateBinding, "restore", reject_recurrence)
    data = np.random.default_rng(136).normal(2.0, 1.7, 200)
    model = _build_basis(BasisSpec(bulk_degree=2), data)
    objective = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )
    result = _solve_basis(objective)
    assert result.status == "converged"
    assert result.final_separation.status == "nonnegative"
    expected = 0.5 * (np.log(2 * np.pi * np.var(data)) + 1)
    assert result.objective_value == pytest.approx(expected, abs=2e-10)


@pytest.mark.parametrize("coefficient,active", [(1e-34, False), (0.0, True)])
def test_face_activity_uses_likelihood_scale_for_tiny_coefficients(coefficient, active):
    from gibbus._fit.basis_constraints import _ConstraintRows
    from gibbus._fit.basis_newton import _multipliers

    rows = _ConstraintRows(np.eye(2), ("growth", "growth"))
    theta, gradient = np.array([1.0, coefficient]), np.array([0.0, 1e34])
    hessian = np.diag([1.0, 1e68])
    multipliers = _multipliers(rows, gradient, theta, hessian)
    assert bool(multipliers[1] > 0.0) == active
    scale = np.array([1.0, 1e-34])
    changed = _multipliers(
        rows, gradient * scale, theta / scale, hessian * scale[:, None] * scale[None, :]
    )
    assert bool(changed[1] > 0.0) == active
