"""Cone descriptions and results of the Newton subproblems.

A description has the form

    B theta = sum_b L_b(Q_b),        Q_b PSD,

where ``B`` is a fixed linear map of the coefficients and
``L_b(Q)_i = <A_{b,i}, Q>``; Gram matrices are solver state.  A ``1 x 1``
block is a nonnegative scalar and a row whose matrices are zero in every
block is an exact linear equality.  The rows come from curvature cuts and
structural rows (``basis_constraints``).  The compiled solver
in ``_conic_kernels`` minimizes the Newton model over a description by an
infeasible-start primal-dual interior-point method (HKM direction, Mehrotra
predictor-corrector) and returns a weak-duality gap bound.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class _ConicRepresentation:
    """Exact finite description ``B theta = sum_b L_b(Q_b)`` with ``Q_b`` PSD.

    Parameters
    ----------
    b_matrix : numpy.ndarray, shape (r, n)
        Fixed linear map of the natural parameters.
    row_matrices : tuple of numpy.ndarray, each shape (r, k_b, k_b)
        Symmetric matrices ``A_{b,i}`` defining each block's linear map.
    reference_dual : numpy.ndarray, shape (r,)
        A strictly dual-feasible multiplier: every ``-sum_i y_i A_{b,i}`` is
        positive definite.  Used to build interior starting points.
    row_degrees : numpy.ndarray, shape (r,)
        Power of the polynomial coefficient each row represents, or ``-1``
        for amplitude and fixed-coordinate rows.  Used for preconditioning.
    """

    b_matrix: np.ndarray
    row_matrices: tuple
    reference_dual: np.ndarray
    row_degrees: np.ndarray

    # The cached properties below are fixed functions of the description; the
    # interior-point loop reads them every iteration.

    @cached_property
    def pseudo_inverse(self):
        """``B^+`` over ``range(B)`` as ``_range_complement`` resolves it.

        A direction that the complement treats as outside the range is not
        inverted either: inverting a near-null singular value (parallel rows
        differing by 1e-14) amplified rounding of ``L(Q)`` into a large move
        of the reconstructed endpoint (a certified subproblem endpoint 45%
        suboptimal).
        """
        left, values, right = np.linalg.svd(self.b_matrix, full_matrices=False)
        rank = _numerical_rank(values)
        return (right[:rank].T / values[:rank]) @ left[:, :rank].T

    @cached_property
    def exact_coordinates(self):
        """``(rows, columns)`` of coordinates read by one row and no other."""
        nonzero = self.b_matrix != 0.0
        rows, columns = [], []
        for row in np.flatnonzero(np.count_nonzero(nonzero, axis=1) == 1):
            column = int(np.flatnonzero(nonzero[row])[0])
            if np.count_nonzero(nonzero[:, column]) == 1:
                rows.append(int(row))
                columns.append(column)
        return np.asarray(rows, dtype=np.intp), np.asarray(columns, dtype=np.intp)

    @cached_property
    def range_complement(self):
        """Orthogonal projector onto the complement of ``range(B)``."""
        return _range_complement(self.b_matrix)

    @cached_property
    def gram_gram(self):
        """``G_ij = sum_b <A_{b,i}, A_{b,j}>``, the Gram matrix of the rows."""
        return sum(
            np.einsum("iab,jab->ij", matrices, matrices)
            for matrices in self.row_matrices
        )

    @cached_property
    def packed(self):
        """``(a_packed, sizes, a_offsets, q_offsets)`` for the compiled solver.

        Block ``b``'s ``(r, k_b, k_b)`` matrices are stored contiguously from
        ``a_offsets[b]``; Gram blocks pack as ``k_b * k_b`` from
        ``q_offsets[b]``.
        """
        sizes = np.array([m.shape[1] for m in self.row_matrices], dtype=np.intc)
        a_offsets = np.zeros(sizes.size, dtype=np.intp)
        q_offsets = np.zeros(sizes.size, dtype=np.intp)
        for b in range(1, sizes.size):
            a_offsets[b] = a_offsets[b - 1] + self.n_rows * int(sizes[b - 1]) ** 2
            q_offsets[b] = q_offsets[b - 1] + int(sizes[b - 1]) ** 2
        a_packed = np.concatenate(
            [
                np.ascontiguousarray(m, dtype=np.float64).reshape(-1)
                for m in self.row_matrices
            ]
        )
        return a_packed, sizes, a_offsets, q_offsets

    def pack_blocks(self, blocks, /):
        """Return Gram blocks as one packed vector.

        Parameters
        ----------
        blocks : sequence of numpy.ndarray
            One ``k_b x k_b`` matrix per block.
        """
        return np.concatenate(
            [np.asarray(q, dtype=np.float64).reshape(-1) for q in blocks]
        )

    def unpack_blocks(self, packed, /):
        """Return a packed vector as a tuple of Gram blocks.

        Parameters
        ----------
        packed : numpy.ndarray
            Packed blocks from ``pack_blocks`` or the compiled solver.
        """
        _, sizes, _, offsets = self.packed
        return tuple(
            np.array(packed[o : o + int(k) * int(k)]).reshape(int(k), int(k))
            for k, o in zip(sizes, offsets, strict=True)
        )

    @cached_property
    def reference_margin(self):
        """Smallest eigenvalue over the reference dual's slacks (positive)."""
        return min(
            float(np.linalg.eigvalsh(slack)[0])
            for slack in self.dual_slacks(self.reference_dual)
        )

    @property
    def n_rows(self):
        """Number of rows ``r`` of the description."""
        return int(self.b_matrix.shape[0])

    @property
    def barrier_parameter(self):
        """Sum of block sizes, the barrier parameter of the PSD cones."""
        return int(sum(matrices.shape[1] for matrices in self.row_matrices))

    def gram_map(self, blocks, /):
        """Return ``sum_b L_b(Q_b)``.

        Parameters
        ----------
        blocks : sequence of numpy.ndarray
            One symmetric matrix per block.
        """
        total = np.zeros(self.n_rows, dtype=np.float64)
        for matrices, block in zip(self.row_matrices, blocks, strict=True):
            total += np.einsum("rij,ij->r", matrices, block)
        return total

    def dual_slacks(self, dual, /):
        """Return ``S_b = -sum_i y_i A_{b,i}`` for every block.

        Parameters
        ----------
        dual : numpy.ndarray, shape (r,)
            Multipliers of the description's rows.
        """
        return tuple(
            -np.einsum("r,rij->ij", dual, matrices) for matrices in self.row_matrices
        )

    def residual(self, params, blocks, /):
        """Return ``max |B theta - sum_b L_b(Q_b)|``.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        blocks : sequence of numpy.ndarray
            Gram blocks.
        """
        return float(
            np.max(np.abs(self.b_matrix @ params - self.gram_map(blocks)), initial=0.0)
        )

    def reconstruct(self, params, blocks, /):
        """Return ``params`` moved onto ``B theta = sum_b L_b(Q_b)``.

        The minimum-norm correction along ``B^+`` is applied, then
        coordinates read by exactly one row (and no other) are imposed
        exactly.  When ``B`` selects
        coordinates (real line; half-line and bounded supports without
        boundary amplitudes) the result satisfies the description exactly, so
        the Gram blocks certify it.  With boundary amplitudes ``B`` maps into
        the product polynomial ``p w + ...``, whose coefficients satisfy
        linear relations the Gram image meets only up to the interior-point
        residual; the correction is then a least-squares fit and the exact
        separator certifies the final parameters.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters to correct.
        blocks : sequence of numpy.ndarray
            Gram certificate whose image the corrected parameters take.
        """
        theta = np.asarray(params, dtype=np.float64).copy()
        target = self.gram_map(blocks)
        theta += self.pseudo_inverse @ (target - self.b_matrix @ theta)
        # Impose exactly only coordinates read by a single row that no other
        # row involves; overwriting a coordinate shared with other rows would
        # break them.
        rows, columns = self.exact_coordinates
        theta[columns] = target[rows] / self.b_matrix[rows, columns]
        return theta


_REPRESENTATIONS: dict = {}


def _range_complement(matrix, /):
    """Return the orthogonal projector onto the complement of ``range(matrix)``.

    Exactly zero when ``matrix`` has full row rank.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (r, n)
        Linear map whose range is removed.
    """
    left, values, _ = np.linalg.svd(matrix, full_matrices=True)
    rest = left[:, _numerical_rank(values) :]
    return rest @ rest.T


def _numerical_rank(values, /):
    """Singular values above ``1e-12 max(1, sigma_max)``: the resolved range.

    Parameters
    ----------
    values : numpy.ndarray
        Singular values, descending.
    """
    cut = 1e-12 * max(1.0, float(values[0]) if values.size else 1.0)
    return int(np.sum(values > cut))


@dataclass(frozen=True)
class _ConicQPResult:
    """Solution of one exact conic Newton subproblem."""

    params: np.ndarray
    blocks: tuple
    dual: np.ndarray
    model_value: float
    gap: float
    iterations: int
    converged: bool
