"""Tests for exact conic Newton subproblems over the curvature cone.

The subproblem solver is the compiled interior point in ``_conic_kernels``;
``conic_reference`` supplies an independent Python dual-value computation.
"""

import numpy as np
import pytest
from conic_reference import _dual_value, _real_line_representation

from gibbus._fit import _conic_kernels
from gibbus._fit.conic_newton import _default_blocks, _preconditioned_subproblem
from gibbus._fit.conic_qp import (
    _ConicQPResult,
    _representation_ranks,
    _support_representation,
)
from gibbus._fit.separation import _separate_full_curvature
from gibbus._model.natural import _natural_layout


def _solve_conic_newton_qp(hessian, gradient, theta0, representation, *, start_blocks):
    """Run the compiled unscaled interior point and wrap its result."""
    a_packed, sizes, a_offsets, q_offsets = representation.packed
    params, packed, dual, model_value, gap, iterations = _conic_kernels.solve_conic_qp(
        hessian,
        gradient,
        theta0,
        representation.b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        representation.reference_dual,
        representation.pack_blocks(start_blocks),
        1e-12,
        100,
        0.99,
    )
    return _ConicQPResult(
        params=params,
        blocks=representation.unpack_blocks(packed),
        dual=dual,
        model_value=float(model_value),
        gap=float(gap),
        iterations=int(iterations),
        converged=bool(gap <= 1e-12 * max(1.0, abs(float(model_value)))),
    )


def _setup(degree, /):
    layout = _natural_layout((-np.inf, np.inf), degree)
    representation = _real_line_representation(layout)
    size = representation.row_matrices[0].shape[1]
    return layout, representation, size


def _random_model(rng, layout, representation, size, /):
    n = layout.n_params
    a = rng.normal(size=(n, n))
    hessian = a @ a.T / n + 10.0 ** rng.uniform(-6, 0) * np.eye(n)
    theta0 = representation.reconstruct(
        np.zeros(n), [np.eye(size) * 10.0 ** rng.uniform(-2, 1)]
    )
    target = rng.normal(size=n) * 10.0 ** rng.uniform(-1, 1)
    gradient = hessian @ (theta0 - target)
    return hessian, gradient, theta0, target


def test_representation_maps_gram_matrix_to_curvature_coefficients():
    layout, representation, size = _setup(8)
    rng = np.random.default_rng(1)
    factor = rng.normal(size=(size, size))
    gram = factor @ factor.T
    theta = representation.reconstruct(np.zeros(layout.n_params), [gram])
    z = np.linspace(-3.0, 3.0, 41)
    basis = np.vander(z, size, increasing=True)
    expected = np.einsum("ij,jk,ik->i", basis, gram, basis)
    curvature = np.polynomial.polynomial.polyval(z, theta[layout.curvature_slice])
    np.testing.assert_allclose(curvature, expected, rtol=1e-12, atol=1e-12)


def test_reference_dual_is_strictly_dual_feasible():
    for degree in (2, 4, 6, 8, 10):
        _, representation, _ = _setup(degree)
        for slack in representation.dual_slacks(representation.reference_dual):
            assert np.linalg.eigvalsh(slack)[0] > 0.0


def test_interior_model_minimum_is_the_unconstrained_newton_step():
    layout, representation, size = _setup(6)
    n = layout.n_params
    hessian = np.diag(np.linspace(1.0, 3.0, n))
    theta0 = representation.reconstruct(np.zeros(n), [np.eye(size)])
    # A strictly interior target: identity Gram matrix plus a linear term.
    target = representation.reconstruct(np.full(n, 0.3), [2.0 * np.eye(size)])
    gradient = hessian @ (theta0 - target)
    result = _solve_conic_newton_qp(
        hessian, gradient, theta0, representation, start_blocks=[np.eye(size)]
    )
    assert result.converged
    np.testing.assert_allclose(result.params, target, rtol=0.0, atol=1e-9)


def test_constant_curvature_reduces_to_a_nonnegative_bound():
    _layout, representation, size = _setup(2)
    assert size == 1
    hessian = np.diag([2.0, 5.0])
    theta0 = np.array([0.0, 1.0])
    target = np.array([0.7, -2.0])  # wants negative constant curvature
    gradient = hessian @ (theta0 - target)
    result = _solve_conic_newton_qp(
        hessian, gradient, theta0, representation, start_blocks=[np.eye(1)]
    )
    assert result.converged
    np.testing.assert_allclose(result.params, [0.7, 0.0], atol=1e-10)


@pytest.mark.parametrize("degree", [4, 6, 8, 10])
def test_random_models_are_solved_to_a_certified_duality_gap(degree):
    layout, representation, size = _setup(degree)
    rng = np.random.default_rng(1000 + degree)
    for _ in range(40):
        hessian, gradient, theta0, _ = _random_model(rng, layout, representation, size)
        result = _solve_conic_newton_qp(
            hessian, gradient, theta0, representation, start_blocks=[np.eye(size)]
        )
        assert result.gap <= 1e-10 * max(1.0, abs(result.model_value))
        assert result.iterations <= 40
        # Returned endpoints are reconstructed from their Gram certificate.
        curvature = representation.gram_map(result.blocks)
        np.testing.assert_allclose(
            result.params[layout.curvature_slice], curvature,
            rtol=0.0, atol=1e-13 * max(1.0, np.abs(curvature).max()),
        )
        block = result.blocks[0]  # PSD Gram certificate (to roundoff)
        assert np.linalg.eigvalsh(block).min() >= -1e-12 * max(1.0, np.abs(block).max())


def test_certificate_is_an_independent_weak_duality_bound():
    """The reported gap is recomputed from the dual alone and bounds any
    feasible point: a random feasible point never beats the dual value."""
    layout, representation, size = _setup(8)
    rng = np.random.default_rng(7)
    hessian, gradient, theta0, _ = _random_model(rng, layout, representation, size)
    result = _solve_conic_newton_qp(
        hessian, gradient, theta0, representation, start_blocks=[np.eye(size)]
    )
    lower = _dual_value(hessian, gradient, theta0, representation, result.dual)
    scale = max(1.0, abs(result.model_value))
    assert result.model_value - lower == pytest.approx(result.gap, abs=1e-12 * scale)
    for _ in range(200):
        factor = rng.normal(size=(size, size))
        feasible = representation.reconstruct(
            theta0 + rng.normal(size=layout.n_params), [factor @ factor.T]
        )
        step = feasible - theta0
        value = gradient @ step + 0.5 * step @ hessian @ step
        assert value >= lower - 1e-12 * max(1.0, abs(lower))


def test_face_representation_fixes_coefficients_to_exact_zero():
    layout = _natural_layout((-np.inf, np.inf), 8)
    face = _real_line_representation(layout, 4)
    size = face.row_matrices[0].shape[1]
    assert size == 3
    rng = np.random.default_rng(3)
    hessian, gradient, theta0, _ = _random_model(
        rng, layout, _real_line_representation(layout), 4
    )
    result = _solve_conic_newton_qp(
        hessian, gradient, theta0, face, start_blocks=[np.eye(size)]
    )
    curvature = result.params[layout.curvature_slice]
    assert curvature[5] == 0.0 and curvature[6] == 0.0
    assert result.gap <= 1e-10 * max(1.0, abs(result.model_value))


def test_representation_rejects_odd_or_out_of_range_effective_degree():
    layout = _natural_layout((-np.inf, np.inf), 8)
    with pytest.raises(ValueError):
        _real_line_representation(layout, 3)
    with pytest.raises(ValueError):
        _real_line_representation(layout, 8)
    with pytest.raises(ValueError):
        _real_line_representation(_natural_layout((0.0, np.inf), 6))


_SUPPORTS = {
    "real_line": (-np.inf, np.inf),
    "lower_half_line": (-1.3, np.inf),
    "upper_half_line": (-np.inf, 0.7),
    "bounded": (-1.3, 1.7),
}


def _layouts():
    for kind, support in _SUPPORTS.items():
        for degree in range(2, 11):
            flags = [(False, False)]
            if np.isfinite(support[0]):
                flags.append((True, False))
            if np.isfinite(support[1]):
                flags.append((False, True))
            if np.isfinite(support[0]) and np.isfinite(support[1]):
                flags.append((True, True))
            for lower, upper in flags:
                yield kind, _natural_layout(support, degree, lower, upper)


def test_gram_redundancy_matches_the_real_line_count():
    """``L`` maps ``k(k+1)/2`` Gram entries onto ``2k-1`` coefficients: its
    kernel has dimension ``(k-1)(k-2)/2`` and nothing else is redundant."""
    for degree in (2, 4, 6, 8, 10, 12):
        layout = _natural_layout((-np.inf, np.inf), degree)
        representation = _support_representation(layout)
        (matrices,) = representation.row_matrices
        size = matrices.shape[1]
        upper_a, upper_b = np.triu_indices(size)
        gram = matrices[:, upper_a, upper_b]
        kernel = gram.shape[1] - np.linalg.matrix_rank(gram)
        assert kernel == (size - 1) * (size - 2) // 2
        row_rank, cone_rank, cone_columns = _representation_ranks(layout, representation)
        assert row_rank == representation.n_rows
        assert cone_rank == cone_columns == layout.curvature_degree + 1


def test_every_description_passes_its_rank_checks():
    """No row is over-specified and no natural parameter is redundant, for
    every support, amplitude combination, exact face and degree."""
    checked = 0
    for kind, layout in _layouts():
        step = 2 if kind == "real_line" else 1
        faces = range(layout.curvature_degree, -1, -step)
        amplitude_choices = [(None, None)]
        if layout.lower_a_index is not None:
            amplitude_choices.append((False, None))
        if layout.upper_a_index is not None:
            amplitude_choices.append((None, False))
        for effective in faces:
            for lower, upper in amplitude_choices:
                representation = _support_representation(layout, effective, lower, upper)
                row_rank, cone_rank, cone_columns = _representation_ranks(
                    layout, representation
                )
                assert row_rank == representation.n_rows
                assert cone_rank == cone_columns
                checked += 1
    assert checked > 150


def test_representation_rank_check_is_invariant_to_extreme_column_scaling():
    layout = _natural_layout((-1.3, 1.7), 8, True, True)
    representation = _support_representation(layout)
    expected = _representation_ranks(layout, representation)

    b_matrix = representation.b_matrix.copy()
    cone = [i for i in range(layout.n_params) if i != layout.gamma_index]
    scales = np.geomspace(1e-120, 1e120, len(cone))
    b_matrix[:, cone] *= scales
    rescaled = type(representation)(
        b_matrix,
        representation.row_matrices,
        representation.reference_dual,
        representation.row_degrees,
    )

    assert _representation_ranks(layout, rescaled) == expected


def test_representation_rank_check_still_detects_exact_column_dependence():
    layout = _natural_layout((-1.3, 1.7), 8, True, True)
    representation = _support_representation(layout)
    cone = [i for i in range(layout.n_params) if i != layout.gamma_index]

    b_matrix = representation.b_matrix.copy()
    b_matrix[:, cone[-1]] = b_matrix[:, cone[-2]]
    dependent = type(representation)(
        b_matrix,
        representation.row_matrices,
        representation.reference_dual,
        representation.row_degrees,
    )
    _, cone_rank, cone_columns = _representation_ranks(layout, dependent)

    assert cone_rank == cone_columns - 1


def _full_curvature(layout, params, z, /):
    candidate = layout.build_candidate(params)
    value = np.polynomial.polynomial.polyval(z, candidate.q_d2)
    lower, upper = layout.support
    amplitudes = np.nan_to_num(candidate.boundary_amplitudes, nan=0.0)
    if np.isfinite(lower):
        value = value + amplitudes[0] / (z - lower) ** 2
    if np.isfinite(upper):
        value = value + amplitudes[1] / (upper - z) ** 2
    return value


def test_every_psd_certificate_gives_a_nonnegative_full_curvature():
    """Not over-specified: whatever PSD blocks the solver picks, parameters
    read off by least squares have nonnegative full curvature."""
    rng = np.random.default_rng(9)
    for _, layout in _layouts():
        if layout.curvature_degree > 6:
            continue
        representation = _support_representation(layout)
        lower, upper = layout.support
        grid_lower = lower + 1e-3 if np.isfinite(lower) else -8.0
        grid_upper = upper - 1e-3 if np.isfinite(upper) else 8.0
        z = np.linspace(grid_lower, grid_upper, 401)
        for _ in range(5):
            blocks = []
            for matrices in representation.row_matrices:
                factor = rng.normal(size=(matrices.shape[1], matrices.shape[1]))
                blocks.append(factor @ factor.T)
            params = representation.reconstruct(np.zeros(layout.n_params), blocks)
            residual = representation.residual(params, blocks)
            if residual > 1e-10:
                # Random Gram images need not lie in range(B) when amplitudes
                # couple rows; only images the description admits are tested.
                continue
            curvature = _full_curvature(layout, params, z)
            assert curvature.min() >= -1e-9 * max(1.0, np.abs(curvature).max())


def test_negative_polynomial_part_compensated_by_amplitude_is_representable():
    """Not under-specified: the full-curvature cone admits p < 0 near an
    endpoint when the amplitude compensates."""
    layout = _natural_layout((-1.3, 1.7), 5, True, False)
    # p(z) = (z - L - 1)^2 - 0.1 is negative near z = L + 1; a = 0.5 lifts it.
    lower = layout.support[0]
    p = np.polynomial.polynomial.polyfromroots([lower + 1.0, lower + 1.0])
    p[0] -= 0.1
    curvature = np.zeros(layout.curvature_degree + 1)
    curvature[: p.size] = p
    params = layout.pack(0.2, curvature, [0.5, np.nan])
    candidate = layout.build_candidate(params)
    assert _separate_full_curvature(
        candidate.q_d2, np.asarray(layout.support), candidate.boundary_amplitudes,
        1e-12, 1e-10, 80, 2,
    ).feasible
    n = layout.n_params
    representation = _support_representation(layout)
    _, projected, _, _ = _preconditioned_subproblem(
        np.eye(n), np.zeros(n), params, representation, _default_blocks(representation)
    )
    assert float(np.max(np.abs(projected - params))) < 1e-5
