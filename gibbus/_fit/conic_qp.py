"""Exact conic Newton subproblems over the curvature cone.

A univariate polynomial is nonnegative on the real line exactly when it is a
sum of squares, i.e. ``p(z) = v(z)^T Q v(z)`` for a positive-semidefinite Gram
matrix ``Q`` and monomial vector ``v(z) = (1, z, ..., z^k)``.  Half-lines and
bounded intervals have the same kind of finite description (Markov-Lukacs)
with more than one block.  Every such description has the form

    B theta = sum_b L_b(Q_b),        Q_b PSD,

where ``B`` is a fixed linear map of the natural parameters and
``L_b(Q)_i = <A_{b,i}, Q>``.  Gram matrices are solver state, never model
coordinates.  A ``1 x 1`` block is a nonnegative scalar (a boundary amplitude),
and a row whose matrices are zero in every block is an exact linear equality
``(B theta)_i = 0`` (an effective-degree face).

This module builds those descriptions (cached per support geometry, degree
and face) and packs them for the compiled interior-point solver in
``_conic_kernels``, which minimizes the Newton model

    minimize  g.(theta - theta0) + 1/2 (theta - theta0)^T H (theta - theta0)
    over      theta, Q_b   subject to the description above

by an infeasible-start primal-dual interior-point method (HKM direction,
Mehrotra predictor-corrector).  Its dual multipliers ``y`` give dual slacks
``S_b = -sum_i y_i A_{b,i}``; for the real line ``-y`` is a moment sequence and
``S`` its Hankel matrix.  Returned endpoints are reconstructed from their Gram
certificates, so they satisfy the cone description by construction, and every
result carries a weak-duality gap bound.
"""

from dataclasses import dataclass
from functools import cached_property
from math import comb

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
        """``B^+``, with the cutoff ``numpy.linalg.lstsq`` uses by default."""
        b = self.b_matrix
        return np.linalg.pinv(b, rcond=np.finfo(np.float64).eps * max(b.shape))

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
            np.einsum("iab,jab->ij", matrices, matrices) for matrices in self.row_matrices
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
            [np.ascontiguousarray(m, dtype=np.float64).reshape(-1) for m in self.row_matrices]
        )
        return a_packed, sizes, a_offsets, q_offsets

    def pack_blocks(self, blocks, /):
        """Return Gram blocks as one packed vector.

        Parameters
        ----------
        blocks : sequence of numpy.ndarray
            One ``k_b x k_b`` matrix per block.
        """
        return np.concatenate([np.asarray(q, dtype=np.float64).reshape(-1) for q in blocks])

    def unpack_blocks(self, packed, /):
        """Return a packed vector as a tuple of Gram blocks.

        Parameters
        ----------
        packed : numpy.ndarray
            Packed blocks from ``pack_blocks`` or the compiled solver.
        """
        _, sizes, _, offsets = self.packed
        return tuple(
            np.array(packed[o: o + int(k) * int(k)]).reshape(int(k), int(k))
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


def _binomial_shift(origin, scale, degree, /):
    """Return ``T`` with ``coefficients_t = T @ coefficients_z`` for ``z = origin + scale t``.

    Parameters
    ----------
    origin, scale : float
        Affine change of variable.
    degree : int
        Polynomial degree.
    """
    shift = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    for power in range(degree + 1):
        for k in range(power + 1):
            shift[k, power] = comb(power, k) * origin ** (power - k) * scale**k
    return shift


def _gram_rows(weight, size, rows, /):
    """Return ``A_i`` for the term ``weight(t) v(t)^T Q v(t)``, ``v = (1, t, ...)``.

    Parameters
    ----------
    weight : sequence of float
        Ascending coefficients of the fixed weight polynomial.
    size : int
        Gram block size.
    rows : int
        Number of polynomial coefficient rows available.
    """
    matrices = np.zeros((rows, size, size), dtype=np.float64)
    for a in range(size):
        for b in range(size):
            for j, value in enumerate(weight):
                if value != 0.0:
                    matrices[a + b + j, a, b] += value
    return matrices


def _reference_moments(kind, degree, /):
    """Moments ``int t^i d mu`` of a positive measure strictly inside the support.

    Their negatives form a strictly dual-feasible multiplier: moment and
    localizing matrices of a measure with enough interior atoms are PD.

    Parameters
    ----------
    kind : str
        Support kind in the shifted variable ``t``.
    degree : int
        Highest moment order.
    """
    if kind == _REAL_LINE_KIND:
        moments = np.zeros(degree + 1, dtype=np.float64)
        moments[0] = 1.0
        for index in range(2, degree + 1, 2):
            moments[index] = moments[index - 2] * (index - 1)
        return moments
    atoms = degree + 3
    if kind == _BOUNDED_KIND:
        points = np.cos(np.pi * (np.arange(atoms) + 0.5) / atoms)
    else:
        points = np.linspace(0.05, 2.0, atoms)
    return np.array([np.mean(points**power) for power in range(degree + 1)])


_REAL_LINE_KIND = "real_line"
_BOUNDED_KIND = "bounded"


_REPRESENTATIONS: dict = {}


def _support_representation(
    layout,
    effective_curvature_degree=None,
    lower_active=None,
    upper_active=None,
    /,
):
    """Return the (cached) exact cone description for any support geometry.

    Descriptions are immutable (their arrays are read-only) and depend only
    on the layout geometry and the face, so each is built, rank-checked and
    factorized once per process; the derived quantities cached on the
    description (pseudo-inverse, packed blocks, ...) are shared too.  See
    ``_build_support_representation`` for the construction.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    effective_curvature_degree : int or None
        Highest represented curvature degree; ``None`` means the layout's.
    lower_active, upper_active : bool or None
        Whether each enabled boundary amplitude is free; ``None`` means
        enabled.
    """
    key = (
        layout.support_kind,
        float(layout.support_lower),
        float(layout.support_upper),
        int(layout.curvature_degree),
        layout.gamma_index,
        layout.lower_a_index,
        layout.upper_a_index,
        None if effective_curvature_degree is None else int(effective_curvature_degree),
        None if lower_active is None else bool(lower_active),
        None if upper_active is None else bool(upper_active),
    )
    cached = _REPRESENTATIONS.get(key)
    if cached is None:
        cached = _build_support_representation(
            layout, effective_curvature_degree, lower_active, upper_active
        )
        for array in (cached.b_matrix, cached.reference_dual, cached.row_degrees,
                      *cached.row_matrices):
            array.flags.writeable = False
        _REPRESENTATIONS[key] = cached
    return cached


def _build_support_representation(
    layout,
    effective_curvature_degree=None,
    lower_active=None,
    upper_active=None,
    /,
):
    """Build the exact cone description for any support geometry.

    The full curvature ``p(z) + a_L/(z-L)^2 + a_U/(U-z)^2`` is nonnegative on
    the open support iff the product polynomial
    ``P = p w + a_L w/(z-L)^2 + a_U w/(U-z)^2`` with
    ``w = (z-L)^2 (U-z)^2`` (factors only for active amplitudes) is
    nonnegative on the closed support, and ``a >= 0``.  In a shifted variable
    ``t`` (``t = z - L`` on a lower half-line, ``t = U - z`` on an upper one,
    ``t in [-1, 1]`` on an interval) the Markov-Lukacs theorems give exact
    finite forms: ``sigma`` on the real line, ``sigma0 + t sigma1`` on a
    half-line, ``sigma0 + (1 - t^2) sigma1`` (even degree) or
    ``(1 + t) sigma0 + (1 - t) sigma1`` (odd degree) on an interval, each
    ``sigma`` a sum of squares, i.e. a PSD Gram block.  Every active amplitude
    is a ``1 x 1`` block.  Rows without Gram entries fix coordinates to exact
    zero: curvature coefficients above ``effective_curvature_degree`` and
    enabled amplitudes that are not active.

    The construction is verified: ``[B | -L]`` must have full row rank (no
    over-specified row) and ``theta -> B theta`` must be injective on the
    cone coordinates (no redundant parameter).

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    effective_curvature_degree : int or None
        Highest represented curvature degree; ``None`` means the layout's.
        Must be even on the real line.
    lower_active, upper_active : bool or None
        Whether each enabled boundary amplitude is free (``True``) or fixed
        to exact zero (``False``); ``None`` means enabled.

    Returns
    -------
    _ConicRepresentation
        Exact description in natural coordinates.

    Raises
    ------
    ValueError
        If the effective degree is out of range or odd on the real line, or
        an amplitude is activated that the layout does not enable.
    RuntimeError
        If the constructed description fails its rank checks.
    """
    kind = layout.support_kind
    lower, upper = layout.support
    degree = int(layout.curvature_degree)
    effective = degree if effective_curvature_degree is None else int(
        effective_curvature_degree
    )
    if effective < 0 or effective > degree:
        raise ValueError("effective curvature degree out of range")
    if kind == _REAL_LINE_KIND and effective % 2:
        raise ValueError("effective curvature degree must be even on the real line")
    enabled = (layout.lower_a_index is not None, layout.upper_a_index is not None)
    active = []
    for flag, allowed in zip((lower_active, upper_active), enabled, strict=True):
        value = allowed if flag is None else bool(flag)
        if value and not allowed:
            raise ValueError("cannot activate a boundary amplitude the layout lacks")
        active.append(value)
    lower_on, upper_on = active

    lower_square = np.array([lower * lower, -2.0 * lower, 1.0]) if lower_on else None
    upper_square = np.array([upper * upper, -2.0 * upper, 1.0]) if upper_on else None
    multiplier = np.array([1.0])
    for square in (lower_square, upper_square):
        if square is not None:
            multiplier = np.convolve(multiplier, square)
    top = effective + multiplier.size - 1

    n = layout.n_params
    start = layout.curvature_slice.start
    product = np.zeros((top + 1, n), dtype=np.float64)
    for power in range(effective + 1):
        product[power : power + multiplier.size, start + power] += multiplier
    if lower_on:
        other = upper_square if upper_on else np.array([1.0])
        product[: other.size, layout.lower_a_index] += other
    if upper_on:
        other = lower_square if lower_on else np.array([1.0])
        product[: other.size, layout.upper_a_index] += other

    if kind == _REAL_LINE_KIND:
        origin, scale, shifted_kind = 0.0, 1.0, _REAL_LINE_KIND
        forms = [(np.array([1.0]), top // 2 + 1)]
    elif kind == _BOUNDED_KIND:
        origin, scale, shifted_kind = 0.5 * (lower + upper), 0.5 * (upper - lower), _BOUNDED_KIND
        if top % 2 == 0:
            forms = [(np.array([1.0]), top // 2 + 1)]
            if top >= 2:
                forms.append((np.array([1.0, 0.0, -1.0]), top // 2))
        else:
            forms = [
                (np.array([1.0, 1.0]), (top - 1) // 2 + 1),
                (np.array([1.0, -1.0]), (top - 1) // 2 + 1),
            ]
    else:
        origin = lower if np.isfinite(lower) else upper
        scale = 1.0 if np.isfinite(lower) else -1.0
        shifted_kind = "half_line"
        forms = [(np.array([1.0]), top // 2 + 1)]
        if top >= 1:
            forms.append((np.array([0.0, 1.0]), (top - 1) // 2 + 1))
    polynomial_rows = _binomial_shift(origin, scale, top) @ product

    fixed_columns = [start + power for power in range(effective + 1, degree + 1)]
    amplitude_columns = []
    for on, allowed, index in zip(
        active, enabled, (layout.lower_a_index, layout.upper_a_index), strict=True
    ):
        if on:
            amplitude_columns.append(index)
        elif allowed:
            fixed_columns.append(index)
    rows = top + 1 + len(amplitude_columns) + len(fixed_columns)

    b_matrix = np.zeros((rows, n), dtype=np.float64)
    b_matrix[: top + 1] = polynomial_rows
    matrices = [_gram_rows(weight, size, rows) for weight, size in forms]
    reference = np.zeros(rows, dtype=np.float64)
    reference[: top + 1] = -_reference_moments(shifted_kind, top)
    row = top + 1
    for column in amplitude_columns:
        b_matrix[row, column] = 1.0
        block = np.zeros((rows, 1, 1), dtype=np.float64)
        block[row, 0, 0] = 1.0
        matrices.append(block)
        reference[row] = -1.0
        row += 1
    for column in fixed_columns:
        b_matrix[row, column] = 1.0
        row += 1
    row_degrees = np.full(rows, -1, dtype=np.int64)
    row_degrees[: top + 1] = np.arange(top + 1)

    # Row-equilibrate the exact equalities before rank checks and numerical
    # solution.  A finite support endpoint can lie many data scales from the
    # fitted coordinate origin (for example, a tiny-spread cluster near 1e6
    # on support [0, inf)).  The binomial shift is then exact but its raw
    # monomial rows can differ by hundreds of orders of magnitude, causing
    # otherwise independent equalities to look rank-deficient in float64.
    # Multiplying one equality row by a positive scalar does not change the
    # represented cone.  Scale B and every Gram map row together, and apply
    # the inverse scaling to the reference dual so its slack matrices are
    # unchanged exactly.
    row_norm = np.max(np.abs(b_matrix), axis=1)
    for block in matrices:
        row_norm = np.maximum(row_norm, np.max(np.abs(block), axis=(1, 2)))
    row_scale = np.ones(rows, dtype=np.float64)
    positive = row_norm > 0.0
    row_scale[positive] = 1.0 / row_norm[positive]
    b_matrix = row_scale[:, None] * b_matrix
    matrices = [row_scale[:, None, None] * block for block in matrices]
    reference = reference / row_scale

    representation = _ConicRepresentation(
        b_matrix, tuple(matrices), reference, row_degrees
    )
    row_rank, cone_rank, cone_columns = _representation_ranks(layout, representation)
    if row_rank != representation.n_rows or cone_rank != cone_columns:
        raise RuntimeError(
            "cone description failed its rank checks: "
            f"[B|-L] rank {row_rank} of {representation.n_rows} rows, "
            f"B rank {cone_rank} on {cone_columns} cone coordinates"
        )
    return representation


def _representation_ranks(layout, representation, /):
    """Return ``(rank [B | -L], rank of B on cone coordinates, #cone coordinates)``.

    ``[B | -L]`` acts on ``theta`` and the upper-triangular Gram entries.
    Full row rank means no row is implied by the others; full column rank of
    ``B`` on the curvature and amplitude coordinates means distinct natural
    parameters always give distinct constraint data.  The linear coefficient
    ``gamma`` does not enter curvature and is excluded.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    representation : _ConicRepresentation
        Description to check.
    """
    def equilibrated_rank(matrix):
        """Numerical rank after invertible row/column equilibration.

        Rank is invariant under nonzero diagonal row and column scalings.  The
        cone map can nevertheless contain columns separated by many orders of
        magnitude when a finite support endpoint is remote in the data-centered
        fitting coordinate.  Testing the raw SVD then measures units/scale
        rather than linear dependence.  Equilibrating both axes keeps the
        existing relative singular-value threshold while making the check
        insensitive to those arbitrary coordinate scales.  Exact zero rows or
        columns remain zero and therefore still fail when they should.
        """
        values = np.asarray(matrix, dtype=np.float64).copy()
        if values.size == 0:
            return 0
        column_norm = np.max(np.abs(values), axis=0)
        nonzero_columns = column_norm > 0.0
        values[:, nonzero_columns] /= column_norm[nonzero_columns]
        row_norm = np.max(np.abs(values), axis=1)
        nonzero_rows = row_norm > 0.0
        values[nonzero_rows] /= row_norm[nonzero_rows, None]
        singular = np.linalg.svd(values, compute_uv=False)
        if singular.size == 0 or not singular[0] > 0.0:
            return 0
        return int(np.sum(singular > 1e-10 * singular[0]))

    b = representation.b_matrix
    parts = [b]
    for matrices in representation.row_matrices:
        size = matrices.shape[1]
        upper_a, upper_b = np.triu_indices(size)
        factor = np.where(upper_a == upper_b, 1.0, 2.0)
        parts.append(-matrices[:, upper_a, upper_b] * factor)
    combined = np.hstack(parts)
    row_rank = equilibrated_rank(combined)
    cone = [column for column in range(layout.n_params) if column != layout.gamma_index]
    cone_rank = equilibrated_rank(b[:, cone])
    return row_rank, cone_rank, len(cone)


def _range_complement(matrix, /):
    """Return the orthogonal projector onto the complement of ``range(matrix)``.

    Exactly zero when ``matrix`` has full row rank.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (r, n)
        Linear map whose range is removed.
    """
    left, values, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.sum(values > 1e-12 * max(1.0, float(values[0]) if values.size else 1.0)))
    rest = left[:, rank:]
    return rest @ rest.T


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
