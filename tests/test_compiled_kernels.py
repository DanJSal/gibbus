"""Compiled kernels against independent checks and test-only construction harnesses."""

import numpy as np
import pytest
from conic_reference import _preconditioned_subproblem_reference, _solve_conic_newton_qp
from gibbus._spectral._panel_kernels import chebval_many
from spectral_builder_harness import PythonSpectralCDFBuilder, PythonSpectralPPFBuilder

from gibbus._fit import _conic_kernels
from gibbus._fit.basis_constraints import (
    _ConstraintRows,
    _linear_representation,
)
from gibbus._fit.conic_newton import _preconditioned_subproblem

# G3 inequality cones use only scalar blocks for curvature and structural rows.
_DESCRIPTIONS = [(4, 3), (6, 4), (8, 5), (10, 6), (6, 1)]


def _representation(n, n_rows, rng, /):
    """Create a random cone of nonnegative, coefficient-linear rows."""
    rows = _ConstraintRows(np.zeros((0, n)), ()).merged(
        rng.normal(size=(n_rows, n)), "cut"
    )
    return _linear_representation(rows)


@pytest.mark.parametrize(
    ("points", "coefficients", "expected"),
    [
        ([], [1.0, 2.0], []),
        ([0.0, 0.5], [], [0.0, 0.0]),
    ],
)
def test_compiled_panel_evaluation_handles_empty_buffers(
    points, coefficients, expected
):
    actual = chebval_many(
        np.asarray(points, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
    )
    np.testing.assert_array_equal(actual, expected)


def _compiled_qp(hessian, gradient, theta0, representation, blocks, /):
    a_packed, sizes, a_offsets, q_offsets = representation.packed
    return _conic_kernels.solve_conic_qp(
        hessian,
        gradient,
        theta0,
        representation.b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        representation.reference_dual,
        representation.pack_blocks(blocks),
        1e-12,
        100,
        0.99,
    )


def _random_model(rng, representation, /):
    n = representation.b_matrix.shape[1]
    a = rng.normal(size=(n, n))
    hessian = a @ a.T / n + 10.0 ** rng.uniform(-6, 0) * np.eye(n)
    blocks = [
        np.eye(m.shape[1]) * 10.0 ** rng.uniform(-2, 1)
        for m in representation.row_matrices
    ]
    theta0 = representation.reconstruct(np.zeros(n), blocks)
    target = rng.normal(size=n) * 10.0 ** rng.uniform(-1, 1)
    gradient = hessian @ (theta0 - target)
    return hessian, gradient, theta0, blocks


def _assert_psd_blocks(blocks, /):
    for block in blocks:
        scale = max(1.0, np.abs(block).max())
        assert np.linalg.eigvalsh(block).min() >= -1e-12 * scale


@pytest.mark.parametrize(("n", "n_rows"), _DESCRIPTIONS)
def test_compiled_qp_is_consistent_with_the_python_reference(n, n_rows):
    rng = np.random.default_rng(n + 17 * n_rows)
    representation = _representation(n, n_rows, rng)
    for _ in range(4):
        hessian, gradient, theta0, blocks = _random_model(rng, representation)
        reference = _solve_conic_newton_qp(
            hessian, gradient, theta0, representation, start_blocks=blocks
        )
        params, packed, _, model, gap, _ = _compiled_qp(
            hessian, gradient, theta0, representation, blocks
        )
        scale = max(1.0, abs(reference.model_value))
        # Each model value is a primal value within its certified gap of the
        # common optimum, so they differ by at most the larger gap.
        assert (
            abs(model - reference.model_value)
            <= max(gap, reference.gap) + 1e-12 * scale
        )
        assert gap <= 1e-6 * scale
        _assert_psd_blocks(representation.unpack_blocks(packed))
        assert np.all(np.isfinite(params))


@pytest.mark.parametrize(("n", "n_rows"), _DESCRIPTIONS)
def test_compiled_preconditioned_solve_matches_the_reference(n, n_rows):
    # Consistency of the certified values (random models with random-scale
    # starts are harder than a fit's subproblems).
    rng = np.random.default_rng(3 * n + 5 * n_rows)
    representation = _representation(n, n_rows, rng)
    for _ in range(4):
        hessian, gradient, theta0, blocks = _random_model(rng, representation)
        compiled, _, blocks_c, model_c = _preconditioned_subproblem(
            hessian, gradient, theta0, representation, blocks
        )
        reference, _, _, model_r = _preconditioned_subproblem_reference(
            hessian, gradient, theta0, representation, blocks
        )
        scale = max(1.0, abs(model_r))
        assert compiled.gap <= 1e-5 * scale
        tolerance = max(compiled.gap, reference.gap) + 1e-11 * scale
        assert abs(model_c - model_r) <= tolerance
        _assert_psd_blocks(blocks_c)


# ---------------------------------------------------------------------------
# Compiled state numerics, point statistics and mixture reductions
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Compiled spectral CDF/PPF construction
# ---------------------------------------------------------------------------


def test_compiled_mixture_spectral_cache_agrees_with_python_harness(monkeypatch):
    from gibbus import Distribution
    from gibbus._api import mixture_stats

    rng = np.random.default_rng(9)
    x = np.concatenate([rng.normal(-3, 0.6, 200), rng.gumbel(1.5, 0.8, 300)])
    fitted = Distribution().fit(x, n_components=2, poly_degree=4, rng=0)
    grid = np.linspace(-6.0, 8.0, 801)
    p = np.linspace(1e-8, 1.0 - 1e-8, 801)
    compiled = (fitted.cdf(grid), fitted.ppf(p))
    monkeypatch.setattr(
        mixture_stats,
        "SpectralCDF",
        lambda support, **kwargs: PythonSpectralCDFBuilder(None, support, **kwargs),
    )
    monkeypatch.setattr(mixture_stats, "SpectralPPF", PythonSpectralPPFBuilder)
    fitted._spectral_cache_valid = False
    np.testing.assert_allclose(fitted.cdf(grid), compiled[0], rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(fitted.cdf(fitted.ppf(p)), p, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(fitted.cdf(compiled[1]), p, rtol=0.0, atol=1e-11)
