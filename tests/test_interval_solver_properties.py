"""Randomized definition-based properties for interval conic fitting."""

import numpy as np
import pytest
from _interval_solver_cases import cdf_grid, make_case


@pytest.mark.parametrize("seed", [*range(16), 1022, 12511])
def test_randomized_interval_solver_satisfies_mathematical_contract(seed):
    """Random supports/censoring/weights end at a feasible stationary face."""
    from gibbus._api.component import _Component
    from gibbus._fit.conic_newton import (
        _certify,
        _preconditioned_subproblem,
        _solve_natural_conic,
    )
    from gibbus._fit.conic_qp import _support_representation
    from gibbus._fit.inputs import _normalize_sample_weight_1d
    from gibbus._fit.natural_objective import (
        _natural_interval_start,
        _prepare_natural_interval_objective,
    )
    from gibbus._postfit.fitted_state import _pack_natural_fit

    case = make_case(seed)
    objective = _prepare_natural_interval_objective(
        case.support,
        case.rows,
        case.degree,
        case.lower_boundary,
        case.upper_boundary,
        (
            None
            if case.weights is None
            else _normalize_sample_weight_1d(len(case.rows), case.weights)
        ),
    )
    start, blocks = _natural_interval_start(objective)
    initial = objective(start)
    result = _solve_natural_conic(objective, initial=start, initial_blocks=blocks)
    label = case.label

    assert result.status in ("converged", "converged_approximately"), label
    assert result.objective_value <= initial.nll + 1e-10 * max(
        1.0, abs(initial.nll)
    ), label
    assert _certify(objective.layout, result.params).feasible, label

    exact = objective(result.params)
    scale = max(1.0, abs(float(exact.nll)))
    assert result.objective_value == pytest.approx(exact.nll, abs=3e-11 * scale), label
    np.testing.assert_allclose(
        result.evaluation.gradient, exact.gradient, rtol=0, atol=3e-10 * scale
    )
    np.testing.assert_allclose(
        result.evaluation.fisher, exact.fisher, rtol=0, atol=3e-10 * scale
    )

    curvature = np.asarray(result.params[objective.layout.curvature_slice])
    assert np.all(curvature[result.effective_curvature_degree + 1 :] == 0.0), label
    for side in ("lower", "upper"):
        index = getattr(objective.layout, f"{side}_a_index")
        active = getattr(result, f"{side}_amplitude_active")
        if index is not None and not active:
            assert result.params[index] == 0.0, label

    representation = _support_representation(
        objective.layout,
        result.effective_curvature_degree,
        result.lower_amplitude_active,
        result.upper_amplitude_active,
    )
    qp, _, _, model_value = _preconditioned_subproblem(
        exact.hessian, exact.gradient, result.params, representation, result.blocks
    )
    local_bound = max(0.0, -model_value) + max(0.0, qp.gap)
    # Degenerate contact faces can floor above the requested 1e-10 tolerance;
    # the production accuracy floor is 1e-7 of the objective scale.
    assert local_bound <= 2e-7 * scale, f"local bound={local_bound:.3e}; {label}"

    if seed % 4 == 0 or seed in {1022, 12511}:
        component = _Component(
            _pack_natural_fit(
                objective, result, effective_n=objective.observations.effective_n
            )
        )
        values = np.asarray(component.cdf(cdf_grid(case)), dtype=np.float64)
        assert np.all(np.isfinite(values)), label
        assert np.all((0.0 <= values) & (values <= 1.0)), label
        assert np.all(np.diff(values) >= -5e-13), label


def test_randomized_case_matrix_covers_solver_geometry():
    """The fixed CI seed block spans the intended structural configurations."""
    cases = [make_case(seed) for seed in range(16)]
    assert {case.geometry for case in cases} == {
        "real",
        "lower_half",
        "upper_half",
        "bounded",
    }
    assert any(case.weights is not None for case in cases)
    assert any(case.lower_boundary for case in cases)
    assert any(case.upper_boundary for case in cases)
    assert all(
        np.unique(case.rows, axis=0).shape[0] < case.rows.shape[0] for case in cases
    )
