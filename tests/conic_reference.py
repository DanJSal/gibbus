"""Pure-Python reference implementations of the compiled conic kernels.

``_conic_kernels.solve_conic_qp`` and ``solve_preconditioned`` are C
translations of the solver below; the tests compare them against it.  Nothing
in the package imports this module.
"""

from dataclasses import dataclass

import numpy as np

from gibbus._fit.conic_qp import (
    _REAL_LINE_KIND,
    _ConicRepresentation,
    _support_representation,
)


@dataclass(frozen=True)
class _ReferenceConicQPResult:
    """Result of the independent NumPy conic-QP solve.

    The test oracle owns this record rather than reusing the production result
    type, so changes to runtime packaging/status metadata cannot make the
    reference agree accidentally.

    Parameters
    ----------
    params : numpy.ndarray
        Feasible natural-parameter endpoint reconstructed from ``blocks``.
    blocks : tuple of numpy.ndarray
        Primal PSD Gram blocks.
    dual : numpy.ndarray
        Dual multipliers used for the weak-duality certificate.
    model_value : float
        Quadratic model value at ``params``.
    gap : float
        Certified weak-duality gap.
    iterations : int
        Interior-point iterations performed.
    converged : bool
        Whether ``gap`` met the requested tolerance.
    """

    params: np.ndarray
    blocks: tuple[np.ndarray, ...]
    dual: np.ndarray
    model_value: float
    gap: float
    iterations: int
    converged: bool


def _real_line_representation(layout, effective_curvature_degree=None, /):
    """Return the real-line description ``c = L(Q)``, ``Q`` PSD.

    Coefficients above ``effective_curvature_degree`` are exact zeros: their
    rows carry no Gram entries.  The effective degree must be even, because a
    nonnegative polynomial on the real line has even degree.

    Parameters
    ----------
    layout : _NaturalLayout
        Real-line natural parameter layout.
    effective_curvature_degree : int or None
        Even degree of the represented curvature; ``None`` means the layout's
        full curvature degree.

    Returns
    -------
    _ConicRepresentation
        One Hankel block of size ``effective_curvature_degree / 2 + 1``.

    Raises
    ------
    ValueError
        If the layout is not real-line or the effective degree is odd or out
        of range.
    """
    if layout.support_kind != _REAL_LINE_KIND:
        raise ValueError("real-line representation requires a real-line layout")
    return _support_representation(layout, effective_curvature_degree)


def _max_step(matrix, direction, /):
    """Largest ``alpha`` keeping ``matrix + alpha * direction`` PSD.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (k, k)
        Strictly positive-definite matrix.
    direction : numpy.ndarray, shape (k, k)
        Symmetric step direction.
    """
    factor = np.linalg.cholesky(matrix)
    inverse = np.linalg.inv(factor)
    scaled = inverse @ direction @ inverse.T
    smallest = float(np.linalg.eigvalsh(0.5 * (scaled + scaled.T))[0])
    return np.inf if smallest >= 0.0 else -1.0 / smallest


def _sym(matrix, /):
    """Return the symmetric part of ``matrix``.

    Parameters
    ----------
    matrix : numpy.ndarray, shape (k, k)
        Square matrix.
    """
    return 0.5 * (matrix + matrix.T)


def _dual_value(hessian, gradient, params0, representation, dual, /):
    """Lagrangian dual value ``D(y) = min_theta m(theta) + y.B theta``.

    Valid as a lower bound on the subproblem optimum whenever every dual
    slack ``-L_b^*(y)`` is PSD.  Returns ``-inf`` when the stationary system is
    inconsistent (the dual function is unbounded below).

    Parameters
    ----------
    hessian, gradient : numpy.ndarray
        Newton model at ``params0``.
    params0 : numpy.ndarray, shape (n,)
        Model expansion point.
    representation : _ConicRepresentation
        Cone description.
    dual : numpy.ndarray, shape (r,)
        Multipliers ``y``.
    """
    b = representation.b_matrix
    w = -(gradient + b.T @ dual)
    step, *_ = np.linalg.lstsq(hessian, w, rcond=None)
    scale = max(1.0, float(np.max(np.abs(gradient))), float(np.max(np.abs(w))))
    if float(np.max(np.abs(hessian @ step - w))) > 1e-10 * scale:
        return -np.inf
    theta = params0 + step
    return float(
        gradient @ step + 0.5 * step @ hessian @ step + dual @ (b @ theta)
    )


def _certified_gap(hessian, gradient, params0, representation, params, blocks, dual, /):
    """Return ``(endpoint, model_value, gap)`` for one primal-dual pair.

    The endpoint is reconstructed from its Gram blocks.  ``gap = m(endpoint)
    - D(y)`` is a weak-duality bound on its suboptimality, returned only when
    the endpoint satisfies the description to roundoff and every dual slack
    is PSD; otherwise ``inf``.

    Parameters
    ----------
    hessian, gradient : numpy.ndarray
        Newton model at ``params0``.
    params0 : numpy.ndarray, shape (n,)
        Model expansion point.
    representation : _ConicRepresentation
        Cone description.
    params : numpy.ndarray, shape (n,)
        Primal iterate before reconstruction.
    blocks : sequence of numpy.ndarray
        Gram blocks of the primal iterate.
    dual : numpy.ndarray, shape (r,)
        Dual iterate.
    """
    endpoint = representation.reconstruct(params, blocks)
    step = endpoint - params0
    model_value = float(gradient @ step + 0.5 * step @ hessian @ step)
    # Weak duality bounds the suboptimality of a *feasible* point only.  When
    # B is not a coordinate selection (boundary amplitudes), reconstruction is
    # a least-squares fit and the endpoint is feasible only once the primal
    # residual has reached roundoff; an infeasible endpoint can sit below the
    # optimum and must not be certified.
    image = representation.gram_map(blocks)
    residual = float(np.max(np.abs(representation.b_matrix @ endpoint - image), initial=0.0))
    if residual > 1e-13 * max(1.0, float(np.max(np.abs(image), initial=0.0))):
        return endpoint, model_value, np.inf
    dual = np.asarray(dual, dtype=np.float64)
    # A dual on the boundary of its cone can show roundoff-negative slack
    # eigenvalues.  Move it along the strictly feasible reference dual just
    # far enough that every slack is PSD; weak duality then holds exactly at
    # the shifted multiplier, so the bound stays rigorous.
    reference = np.asarray(representation.reference_dual, dtype=np.float64)
    deficit = 0.0
    for slack in representation.dual_slacks(dual):
        deficit = max(deficit, -float(np.linalg.eigvalsh(slack)[0]))
    if deficit > 0.0:
        dual = dual + (2.0 * deficit / representation.reference_margin) * reference
        for slack in representation.dual_slacks(dual):
            if float(np.linalg.eigvalsh(slack)[0]) < 0.0:
                return endpoint, model_value, np.inf
    lower = _dual_value(hessian, gradient, params0, representation, dual)
    return endpoint, model_value, float(model_value - lower)


@dataclass(frozen=True)
class _KKTSystem:
    """Newton system of one interior-point iteration (HKM direction)."""

    representation: _ConicRepresentation
    matrix: np.ndarray
    dual_residual: np.ndarray
    primal_residual: np.ndarray
    blocks: tuple
    inverses: tuple
    gram_gram: np.ndarray
    complement: np.ndarray


def _kkt_system(hessian, gradient, params0, representation, params, blocks, dual, slacks, /):
    """Assemble the reduced HKM Newton system at one primal-dual iterate.

    Eliminating the Gram and slack directions leaves
    ``[[H, B^T], [B, -M]] [d_theta; d_y] = rhs`` with the Schur complement
    ``M_ij = sum_b tr(A_{b,i} Q_b A_{b,j} S_b^-1)``.

    Parameters
    ----------
    hessian, gradient : numpy.ndarray
        Newton model at ``params0``.
    params0 : numpy.ndarray, shape (n,)
        Model expansion point.
    representation : _ConicRepresentation
        Cone description.
    params : numpy.ndarray, shape (n,)
        Current primal parameters.
    blocks : sequence of numpy.ndarray
        Current PD Gram blocks.
    dual : numpy.ndarray, shape (r,)
        Current multipliers.
    slacks : sequence of numpy.ndarray
        Current PD dual slacks.

    Returns
    -------
    _KKTSystem
        Matrix, residuals and the factors needed to recover directions.
    """
    rep = representation
    b = rep.b_matrix
    r = rep.n_rows
    inverses = tuple(np.linalg.inv(slack) for slack in slacks)
    schur = np.zeros((r, r), dtype=np.float64)
    for matrices, q, s_inv in zip(rep.row_matrices, blocks, inverses, strict=True):
        left = np.einsum("iab,bc->iac", matrices, q)
        right = np.einsum("jab,bc->jac", matrices, s_inv)
        schur += np.einsum("iac,jca->ij", left, right)
    return _KKTSystem(
        representation=rep,
        matrix=np.block([[hessian, b.T], [b, -_sym(schur)]]),
        dual_residual=gradient + hessian @ (params - params0) + b.T @ dual,
        primal_residual=b @ params - rep.gram_map(blocks),
        blocks=tuple(blocks),
        inverses=inverses,
        gram_gram=rep.gram_gram,
        complement=rep.range_complement,
    )


def _kkt_direction(system, targets, /):
    """Solve for one search direction with complementarity targets.

    ``targets[b]`` is the desired change of ``Q_b S_b``; the Gram direction
    is ``sym((target - Q dS) S^-1)``.

    Parameters
    ----------
    system : _KKTSystem
        Newton system of the current iterate.
    targets : sequence of numpy.ndarray
        Complementarity targets, one per block.

    Returns
    -------
    d_theta, d_dual : numpy.ndarray
        Parameter and multiplier directions.
    d_blocks, d_slacks : list of numpy.ndarray
        Gram and dual-slack directions.
    """
    rep = system.representation
    n = system.matrix.shape[0] - rep.n_rows
    ell = np.zeros(rep.n_rows, dtype=np.float64)
    for matrices, target, s_inv in zip(
        rep.row_matrices, targets, system.inverses, strict=True
    ):
        ell += np.einsum("rij,ji->r", matrices, target @ s_inv)
    rhs = np.concatenate((-system.dual_residual, -system.primal_residual + ell))
    solution = np.linalg.solve(system.matrix, rhs)
    # One step of iterative refinement: late iterations are ill conditioned.
    solution += np.linalg.solve(system.matrix, rhs - system.matrix @ solution)
    d_theta = solution[:n]
    d_dual = solution[n:]
    d_slacks = list(rep.dual_slacks(d_dual))
    d_blocks = [
        _sym(target @ s_inv - q @ d_s @ s_inv)
        for target, s_inv, q, d_s in zip(
            targets, system.inverses, system.blocks, d_slacks, strict=True
        )
    ]
    # Recovering the Gram direction goes through S^-1, which is ill
    # conditioned late in the solve, so it misses the linearized primal
    # equation B d_theta - L(d_Q) = -r_p by accumulated roundoff.  Close that
    # gap with the minimum-norm Gram correction, computed from the fixed and
    # well-conditioned Gram matrix of the A_i; otherwise the primal residual
    # drifts upward exactly when the certificate needs it at roundoff.
    # Only the part outside range(B) needs it: reconstruction absorbs the rest
    # into theta (all of it when B selects coordinates, as on the real line).
    mismatch = rep.b_matrix @ d_theta - rep.gram_map(d_blocks) + system.primal_residual
    mismatch = system.complement @ mismatch
    if not np.any(mismatch):
        return d_theta, d_dual, d_blocks, d_slacks
    coefficients, *_ = np.linalg.lstsq(system.gram_gram, mismatch, rcond=None)
    d_blocks = [
        d_q + np.einsum("r,rij->ij", coefficients, matrices)
        for d_q, matrices in zip(d_blocks, rep.row_matrices, strict=True)
    ]
    return d_theta, d_dual, d_blocks, d_slacks


def _step_limit(blocks, d_blocks, slacks, d_slacks, /):
    """Largest common step keeping every block and slack PSD.

    Parameters
    ----------
    blocks, slacks : sequence of numpy.ndarray
        Current PD Gram blocks and dual slacks.
    d_blocks, d_slacks : sequence of numpy.ndarray
        Their search directions.
    """
    limit = np.inf
    for q, d_q, s, d_s in zip(blocks, d_blocks, slacks, d_slacks, strict=True):
        limit = min(limit, _max_step(q, d_q), _max_step(s, d_s))
    return limit


def _solve_conic_newton_qp(
    hessian,
    gradient,
    params,
    representation,
    /,
    *,
    start_blocks,
    gap_tolerance=1e-12,
    max_iterations=100,
    step_fraction=0.99,
):
    """Solve one Newton model exactly over the cone description.

    Parameters
    ----------
    hessian, gradient : numpy.ndarray
        Newton model at ``params``; ``hessian`` must be PSD.
    params : numpy.ndarray
        Current natural parameters ``theta0``; need not satisfy the cone.
    representation : _ConicRepresentation
        Exact cone description.
    start_blocks : sequence of numpy.ndarray
        Strictly positive-definite starting Gram blocks.
    gap_tolerance : float
        Target for the certified duality gap relative to
        ``max(1, |model value|)``.
    max_iterations : int
        Interior-point iteration limit.
    step_fraction : float
        Fraction of the maximal step to the cone boundary taken each
        iteration.

    Termination is decided by the certified weak-duality gap rather than by
    the primal and dual residuals, because late in the solve the KKT system becomes ill
    conditioned (dual slacks with eigenvalues near ``1e-15``) and residuals
    can grow while the certified gap is already at roundoff.  The iterate
    with the smallest certified gap is returned.  The solve also stops when
    that gap has not halved in five iterations or the step length collapses.

    Returns
    -------
    _ReferenceConicQPResult
        ``params`` is reconstructed from ``blocks`` and lies in the cone;
        ``gap`` bounds its model suboptimality.
    """
    h = _sym(np.asarray(hessian, dtype=np.float64))
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    theta0 = np.asarray(params, dtype=np.float64).reshape(-1)
    rep = representation
    nu = float(rep.barrier_parameter)

    theta = theta0.copy()
    blocks = [np.array(block, dtype=np.float64) for block in start_blocks]
    for block in blocks:
        np.linalg.cholesky(block)  # raises unless strictly positive definite

    # Interior dual start: a positive multiple of the reference dual, scaled
    # so that the initial complementarity is comparable to the model scale.
    reference = np.asarray(rep.reference_dual, dtype=np.float64)
    pairing = sum(
        float(np.sum(q * s))
        for q, s in zip(blocks, rep.dual_slacks(reference), strict=True)
    )
    target_mu = max(1e-8, float(np.max(np.abs(g), initial=0.0)))
    dual = (target_mu * nu / max(pairing, 1e-300)) * reference

    identities = [np.eye(block.shape[0]) for block in blocks]
    best = None
    best_rank = None
    history = []
    iterations = 0
    while iterations < int(max_iterations):
        iterations += 1
        endpoint, model_value, gap = _certified_gap(
            h, g, theta0, rep, theta, blocks, dual
        )
        # Rank by certified gap; among uncertified iterates prefer the one
        # closest to primal feasibility, never the starting guess.
        rank = (gap, rep.residual(endpoint, blocks))
        if best is None or rank < best_rank:
            best = (endpoint, model_value, gap, tuple(blocks), dual.copy())
            best_rank = rank
        if gap <= float(gap_tolerance) * max(1.0, abs(model_value)):
            break
        slacks = rep.dual_slacks(dual)
        mu = sum(
            float(np.sum(q * s)) for q, s in zip(blocks, slacks, strict=True)
        ) / nu
        # Stall detection applies only near the roundoff floor; earlier the
        # certified gap can legitimately plateau while infeasibility falls.
        if mu <= 1e-9 * max(1.0, abs(model_value)):
            history.append(best[2])
            if len(history) > 5 and history[-1] > 0.5 * history[-6]:
                break
        try:
            system = _kkt_system(h, g, theta0, rep, theta, blocks, dual, slacks)
            affine_targets = [-(q @ s) for q, s in zip(blocks, slacks, strict=True)]
            _, _, d_blocks_aff, d_slacks_aff = _kkt_direction(system, affine_targets)
            alpha_aff = min(
                1.0, _step_limit(blocks, d_blocks_aff, slacks, d_slacks_aff)
            )
            mu_aff = sum(
                float(np.sum((q + alpha_aff * d_q) * (s + alpha_aff * d_s)))
                for q, d_q, s, d_s in zip(
                    blocks, d_blocks_aff, slacks, d_slacks_aff, strict=True
                )
            ) / nu
            sigma = min(1.0, max(0.0, mu_aff / mu)) ** 3 if mu > 0.0 else 0.0
            corrector_targets = [
                sigma * mu * identity - q @ s - d_q @ d_s
                for identity, q, s, d_q, d_s in zip(
                    identities, blocks, slacks, d_blocks_aff, d_slacks_aff,
                    strict=True,
                )
            ]
            d_theta, d_dual, d_blocks, d_slacks = _kkt_direction(
                system, corrector_targets
            )
            alpha = min(
                1.0,
                float(step_fraction) * _step_limit(blocks, d_blocks, slacks, d_slacks),
            )
            if alpha < 1e-10:
                break
            new_blocks = [
                _sym(q + alpha * d_q) for q, d_q in zip(blocks, d_blocks, strict=True)
            ]
            new_dual = dual + alpha * d_dual
            for block in new_blocks:
                np.linalg.cholesky(block)
            for slack in rep.dual_slacks(new_dual):
                np.linalg.cholesky(slack)
        except np.linalg.LinAlgError:
            # A block or slack became numerically singular at the roundoff
            # floor; the best certified iterate so far is the answer.
            break
        theta = theta + alpha * d_theta
        dual = new_dual
        blocks = new_blocks

    endpoint, model_value, gap, best_blocks, best_dual = best
    return _ReferenceConicQPResult(
        params=endpoint,
        blocks=best_blocks,
        dual=best_dual,
        model_value=model_value,
        gap=gap,
        iterations=iterations,
        converged=gap <= float(gap_tolerance) * max(1.0, abs(model_value)),
    )


def _interior_shift(blocks, relative, /):
    """Return blocks shifted inward by ``relative`` times their mean eigenvalue.

    Parameters
    ----------
    blocks : sequence of numpy.ndarray
        PSD Gram blocks.
    relative : float
        Shift as a fraction of ``trace / size`` of each block.

    Returns
    -------
    list of numpy.ndarray
        Strictly positive-definite blocks.
    """
    shifted = []
    for block in blocks:
        size = block.shape[0]
        scale = max(float(np.trace(block)) / size, 1e-300)
        shifted.append(block + relative * scale * np.eye(size))
    return shifted


def _preconditioned_subproblem_reference(hessian, gradient, params, representation, blocks, /):
    """Solve one Newton model with a fixed diagonal rescaling (reference Python).

    In the monomial basis the Fisher matrix of a degree-10 fit can span
    thirteen orders of magnitude, which defeats the interior-point
    certificate.  Each subproblem is therefore solved in scaled variables:
    ``theta = D phi`` with Jacobi ``D = diag(H)^(-1/2)``; polynomial rows of
    power ``i`` scaled by ``sigma^i`` and every Gram basis by ``v(t / sigma)``
    (``Q = S^-1 Q~ S^-1``, ``S = diag(sigma^a)``), with ``sigma`` fitted so the
    scaled polynomial rows are as flat as possible; other rows are
    equilibrated.  Row ``i`` then reads
    ``R_i (B D phi)_i = sum_b <R_i S_b^-1 A_{b,i} S_b^-1, Q~_b>``.  This is a
    fixed invertible linear change of variables per subproblem: the model,
    the cone and the certificates are unchanged.

    Parameters
    ----------
    hessian : numpy.ndarray, shape (n, n)
        PSD Newton model Hessian at ``params``.
    gradient : numpy.ndarray, shape (n,)
        Objective gradient at ``params``.
    params : numpy.ndarray, shape (n,)
        Current natural parameters.
    representation : _ConicRepresentation
        Cone description in natural coordinates.
    blocks : sequence of numpy.ndarray
        Current Gram blocks (the starting point of the interior solve).

    Returns
    -------
    result : _ReferenceConicQPResult
        Subproblem result in scaled coordinates (for its gap and iterations).
    endpoint : numpy.ndarray, shape (n,)
        Subproblem endpoint in natural coordinates.
    endpoint_blocks : tuple of numpy.ndarray
        Gram certificate of ``endpoint`` in the original basis.
    model_value : float
        Newton model value of ``endpoint``.
    """
    h = np.asarray(hessian, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    theta = np.asarray(params, dtype=np.float64)
    diagonal = np.diag(h).copy()
    column = np.ones_like(diagonal)
    positive = diagonal > 0.0
    column[positive] = 1.0 / np.sqrt(diagonal[positive])

    b = representation.b_matrix
    degrees = np.asarray(representation.row_degrees)
    magnitude = np.max(np.abs(b * column[None, :]), axis=1)
    magnitude[magnitude == 0.0] = 1.0
    polynomial = np.flatnonzero(degrees >= 0)
    sigma = 1.0
    if polynomial.size >= 2 and np.ptp(degrees[polynomial]) > 0:
        slope = np.polyfit(degrees[polynomial], -np.log(magnitude[polynomial]), 1)[0]
        sigma = float(np.exp(slope))
    row_scale = 1.0 / magnitude
    row_scale[polynomial] = sigma ** degrees[polynomial].astype(np.float64)
    gram_scales = [
        sigma ** np.arange(matrices.shape[1], dtype=np.float64)
        for matrices in representation.row_matrices
    ]
    scaled = _ConicRepresentation(
        b_matrix=(row_scale[:, None] * b) * column[None, :],
        row_matrices=tuple(
            row_scale[:, None, None] * matrices / np.outer(scale, scale)[None, :, :]
            for matrices, scale in zip(representation.row_matrices, gram_scales, strict=True)
        ),
        reference_dual=representation.reference_dual / row_scale,
        row_degrees=degrees,
    )
    start = [
        np.asarray(block) * np.outer(scale, scale)
        for block, scale in zip(blocks, gram_scales, strict=True)
    ]
    result = _solve_conic_newton_qp(
        column[:, None] * h * column[None, :],
        column * g,
        theta / column,
        scaled,
        start_blocks=_interior_shift(start, 1e-3),
    )
    endpoint_blocks = tuple(
        np.asarray(block) / np.outer(scale, scale)
        for block, scale in zip(result.blocks, gram_scales, strict=True)
    )
    endpoint = representation.reconstruct(column * result.params, endpoint_blocks)
    step = endpoint - theta
    model_value = float(g @ step + 0.5 * step @ h @ step)
    return result, endpoint, endpoint_blocks, model_value
