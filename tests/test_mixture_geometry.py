"""Physical sharing, exact-face injections and raw mixture information."""

import numpy as np
import pytest
from gibbus._fit._shared_mixture_kernels import SharedMStepObjective

from gibbus._fit.conic_newton import (
    _default_blocks,
    _interior_start,
    _preconditioned_subproblem,
    _project,
)
from gibbus._fit.mixture_geometry import _MixtureNaturalMap
from gibbus._fit.natural_mixture import (
    _compiled_component,
    _CompiledJointMixture,
    _ComponentProblem,
    _RawMixtureEvaluation,
)


def _problems(intervals=False, support=(0.0, 1.0)):
    x = np.linspace(0.02, 0.98, 35)
    if np.isneginf(support[0]):
        x = -4 * x
    elif np.isposinf(support[1]):
        x = 4 * x
    rows = np.column_stack((x - 0.005, x + 0.005)) if intervals else x[:, None]
    w = np.linspace(1.0, 2.0, x.size)
    w /= w.sum()
    r = np.column_stack(
        (np.linspace(0.85, 0.1, x.size), np.linspace(0.15, 0.9, x.size))
    )
    problems = [
        _ComponentProblem(
            support,
            rows,
            4,
            np.isfinite(support[0]),
            np.isfinite(support[1]),
            w * r[:, k],
        )
        for k in range(2)
    ]
    objectives = [p.objective(w * r[:, k]) for k, p in enumerate(problems)]
    return problems, objectives, w, r


@pytest.mark.parametrize("support", [(0.0, 1.0), (0.0, np.inf), (-np.inf, 0.0)])
def test_physical_map_strict_roundtrip_and_reflection(support):
    problems, objectives, _, _ = _problems(support=support)
    mapping = _MixtureNaturalMap([p.spec for p in problems])
    params = [_interior_start(objective) for objective in objectives]
    reduced = mapping.pack(params)
    for expected, actual in zip(params, mapping.expand(reduced), strict=True):
        np.testing.assert_array_equal(actual, expected)
    lower, upper = mapping.shared_parameter_indices
    assert (lower is not None) == np.isfinite(support[0])
    assert (upper is not None) == np.isfinite(support[1])
    if np.isneginf(support[0]):
        assert mapping.local_indices[0][problems[0].spec.layout.lower_a_index] == upper
    params[1][-1] = np.nextafter(params[1][-1], np.inf)
    with pytest.raises(ValueError, match="share physical"):
        mapping.pack(params)


@pytest.mark.parametrize(
    "active", [(False, False), (True, False), (False, True), (True, True)]
)
def test_face_injection_has_no_redundant_equalities(active):
    problems, objectives, _, _ = _problems()
    mapping = _MixtureNaturalMap([p.spec for p in problems])
    face = mapping.face((0, 2), active, 1)
    rep = face.representation
    matrix = np.column_stack(
        [rep.b_matrix] + [block.reshape(rep.n_rows, -1) for block in rep.row_matrices]
    )
    assert np.linalg.matrix_rank(matrix) == rep.n_rows
    assert all(
        np.linalg.eigvalsh(slack)[0] > 0
        for slack in rep.dual_slacks(rep.reference_dual)
    )
    constrained = np.any(rep.b_matrix != 0, axis=0)
    assert np.linalg.matrix_rank(rep.b_matrix[:, constrained]) == constrained.sum()
    full = np.r_[mapping.pack([_interior_start(item) for item in objectives]), 0.3]
    free = face.restrict(full)
    projected, blocks = _project(free, rep, _default_blocks(rep), np.eye(free.size))
    assert rep.residual(projected, blocks) < 2e-7
    expanded = face.expand(projected)
    excluded = np.setdiff1d(np.arange(expanded.size), face.free_indices)
    np.testing.assert_array_equal(expanded[excluded], 0)
    np.testing.assert_allclose(rep.reconstruct(projected, blocks), projected, atol=2e-7)
    gradient = np.zeros_like(projected)
    gradient[-1] = 0.15
    result, optimum, _, model_value = _preconditioned_subproblem(
        np.eye(free.size), gradient, projected, rep, blocks
    )
    assert result.gap < 1e-9
    assert model_value == pytest.approx(-0.5 * 0.15**2, abs=1e-9)
    np.testing.assert_allclose(optimum, projected - gradient, atol=2e-5)


def _composite(objectives, masses, mapping):
    """Return the compiled M-step objective summed over the full mixture face."""
    compiled = SharedMStepObjective(
        [
            (kind, mass, inputs)
            for (kind, inputs), mass in zip(
                map(_compiled_component, objectives), masses, strict=True
            )
        ]
    )
    face = mapping.face(
        [layout.curvature_degree for layout in mapping.layouts],
        tuple(index is not None for index in mapping.shared_parameter_indices),
    )
    assert len(face.free_indices) == mapping.n_params

    def evaluate(x):
        size = mapping.n_params
        nll, gradient = 0.0, np.zeros(size)
        fisher, missing = np.zeros((size, size)), np.zeros((size, size))
        for (value, g, f, c, _), mass, indices in zip(
            compiled.evaluate(face.component_columns, size, x),
            masses,
            mapping.local_indices,
            strict=True,
        ):
            block = np.ix_(indices, indices)
            nll += mass * value
            gradient[indices] += mass * g
            fisher[block] += mass * f
            missing[block] += mass * c
        return _RawMixtureEvaluation(nll, gradient, fisher, missing)

    return evaluate


@pytest.mark.parametrize("intervals", [False, True])
@pytest.mark.parametrize("observed", [False, True])
def test_shared_objective_raw_derivatives_and_mass_weights(intervals, observed):
    problems, objectives, w, r = _problems(intervals)
    mapping = _MixtureNaturalMap([p.spec for p in problems])
    params = [_interior_start(item) for item in objectives]
    if observed:
        objective = _CompiledJointMixture(problems, mapping.layouts, w)
        x = objective.join(params, np.log(w @ r))
    else:
        objective = _composite(objectives, w @ r, mapping)
        x = mapping.pack(params)
    evaluation = objective(x)
    if not observed:
        assert evaluation.nll == pytest.approx(
            sum(
                mass * item(theta).nll
                for mass, item, theta in zip(w @ r, objectives, params, strict=True)
            )
        )
    step = 1e-6
    for index in range(x.size):
        delta = np.zeros_like(x)
        delta[index] = step
        plus, minus = objective(x + delta), objective(x - delta)
        assert (plus.nll - minus.nll) / (2 * step) == pytest.approx(
            evaluation.gradient[index], abs=3e-7, rel=2e-5
        )
        np.testing.assert_allclose(
            (plus.gradient - minus.gradient) / (2 * step),
            evaluation.observed_hessian[:, index],
            rtol=2e-4,
            atol=3e-6,
        )
