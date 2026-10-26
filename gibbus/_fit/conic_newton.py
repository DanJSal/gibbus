"""Shared Newton policy, run record and subproblem of the face loop.

The contracts ``basis_newton`` relies on: the convergence status vocabulary,
the immutable Newton policy (``_NewtonOptions``), the run record
(``_NewtonRun``), the preconditioned compiled subproblem and the overlay of
an observation-level global certificate.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import (
    CONIC_QP_GAP_TOL,
    CONIC_QP_MAX_ITER,
    CONIC_QP_STEP_FRACTION,
)
from . import (  # type: ignore[attr-defined]  # compiled
    _conic_kernels,
)
from .conic_qp import (
    _ConicQPResult,
)

_CONVERGED = ("converged", "converged_approximately")
# A converged point whose shape is not verified within the threshold.
_UNVERIFIED = "unverified"


@dataclass(frozen=True)
class _NewtonOptions:
    """Immutable Newton policy shared by every solve in one fit.

    The defaults live here so single-component solves, mixture M-steps and
    joint polishing cannot silently reconstruct different solver policies.
    Warm starts and certification requests are per-solve state and therefore
    are deliberately not part of this object.

    NLL tolerances are absolute, in nats per observation: a change of data
    units shifts the NLL by ``log(scale)`` and leaves its differences
    unchanged, so tolerances relative to ``max(1, |NLL|)`` would stop a fit
    at a point that depended on the units.

    Parameters
    ----------
    tolerance : float, optional
        Predicted-decrease target for Newton convergence.
    certified_tolerance : float, optional
        Certified bound accepted as full convergence when the subproblem gap
        reaches its numerical floor.
    accuracy_floor : float, optional
        Larger certified bound reported as approximate convergence.
    face_trigger : float, optional
        Relative threshold that prompts an exact lower-dimensional face solve.
    max_iterations : int, optional
        Newton iteration limit per cone description.
    armijo : float, optional
        Armijo sufficient-decrease constant.
    backtrack : float, optional
        Backtracking reduction factor.
    max_line_search : int, optional
        Line-search trial limit per Newton step.
    lift_nll_slack : float, optional
        Interior lift of a lifted start (``interior_lift``): the NLL the
        move into the face interior may give up.
    lift_halvings : int, optional
        Step halvings allowed to meet ``lift_nll_slack``.
    cut_retention_exchanges : int, optional
        Retain point cuts through this many consecutive inactive exchange
        iterates. Zero disables retirement. Structural and tail rows persist.
    cut_activity_margin : float, optional
        Slack margin relative to the row's absolute coefficient contributions,
        with a unit floor, below which a point cut remains active.
    """

    tolerance: float = 1e-12
    certified_tolerance: float = 1e-10
    accuracy_floor: float = 1e-7
    face_trigger: float = 1e-6
    max_iterations: int = 60
    armijo: float = 1e-4
    backtrack: float = 0.5
    max_line_search: int = 40
    lift_nll_slack: float = 1e-3
    lift_halvings: int = 60
    cut_retention_exchanges: int = 4
    cut_activity_margin: float = 1e-5


@dataclass(frozen=True)
class _NewtonRun:
    """State returned by one Newton run on a fixed cone description."""

    status: str
    params: np.ndarray
    blocks: tuple
    dual: np.ndarray
    evaluation: object
    iterations: int
    evaluations: int
    subproblem_iterations: int
    decrease_bound: float


def _preconditioned_subproblem(
    hessian, gradient, params, representation, blocks, /, *, column_scale=None
):
    """Solve one Newton model with a fixed diagonal rescaling (compiled).

    In the monomial basis the Fisher matrix of a degree-10 fit can span
    thirteen orders of magnitude, which defeats the interior-point
    certificate.  Each subproblem is therefore solved in scaled variables:
    ``theta = D phi`` with Jacobi ``D = diag(H)^(-1/2)``; polynomial rows of
    power ``i`` scaled by ``sigma^i`` and every Gram basis by ``v(t / sigma)``
    (``Q = S^-1 Q~ S^-1``, ``S = diag(sigma^a)``), with ``sigma`` fitted so the
    scaled polynomial rows are as flat as possible; other rows are
    equilibrated.  This is a fixed invertible linear change of variables per
    subproblem: the model, the cone and the certificates are unchanged.  The
    whole solve runs in C without the GIL.

    Parameters
    ----------
    hessian : numpy.ndarray, shape (n, n)
        PSD Newton model Hessian at ``params``.
    gradient : numpy.ndarray, shape (n,)
        Objective gradient at ``params``.
    params : numpy.ndarray, shape (n,)
        Current parameters.
    representation : _ConicRepresentation
        Cone description in unscaled coordinates.
    blocks : sequence of numpy.ndarray
        Current Gram blocks (the starting point of the interior solve).
    column_scale : numpy.ndarray or None, optional
        Positive column scales replacing Jacobi ``D`` (row-flattening
        scales); ``None`` keeps Jacobi.

    Returns
    -------
    result : _ConicQPResult
        Subproblem result in scaled coordinates (for its gap and iterations).
    endpoint : numpy.ndarray, shape (n,)
        Subproblem endpoint in unscaled coordinates.
    endpoint_blocks : tuple of numpy.ndarray
        Gram certificate of ``endpoint`` in the original basis.
    model_value : float
        Newton model value of ``endpoint``.
    """
    a_packed, sizes, a_offsets, q_offsets = representation.packed
    endpoint, packed, model_value, gap, iterations, dual, scaled_model = (
        _conic_kernels.solve_preconditioned(
            hessian,
            gradient,
            params,
            representation.b_matrix,
            a_packed,
            sizes,
            a_offsets,
            q_offsets,
            representation.reference_dual,
            representation.row_degrees,
            representation.pack_blocks(blocks),
            gap_tolerance=CONIC_QP_GAP_TOL,
            max_iterations=CONIC_QP_MAX_ITER,
            step_fraction=CONIC_QP_STEP_FRACTION,
            column_scale=column_scale,
        )
    )
    endpoint_blocks = representation.unpack_blocks(packed)
    result = _ConicQPResult(
        params=endpoint,
        blocks=endpoint_blocks,
        dual=dual,
        model_value=float(scaled_model),
        gap=float(gap),
        iterations=int(iterations),
        converged=bool(gap <= CONIC_QP_GAP_TOL * max(1.0, abs(float(scaled_model)))),
    )
    return result, endpoint, endpoint_blocks, float(model_value)


def _apply_global_objective_certificate(objective, run, options, /):
    """Overlay an observation-level global NLL certificate on a local solve.

    Fixed-face Newton -- compiled or Python -- certifies only local conic
    optimality.  Some interval objectives additionally expose a rigorous
    lower bound on the NLL over *all* distributions.  Keeping that bound out
    of the Newton traversal makes local optimization identical across
    standalone and mixture fits; this Python controller then reports the
    stronger global certificate when the endpoint reaches the bound.

    The global certificate intentionally takes precedence inside the accuracy
    floor.  Thus a locally certified stationary point that is only known to
    be within ``accuracy_floor`` of the nonparametric optimum is reported as
    ``converged_approximately``, matching the historical standalone semantics.

    Parameters
    ----------
    objective : callable
        Objective whose optional ``nll_lower_bound`` supplies the global bound.
    run : _NewtonRun
        Completed local fixed-face solve.
    options : _NewtonOptions
        Certification tolerances used to interpret the global gap.

    Returns
    -------
    _NewtonRun
        ``run`` unchanged when no stronger global certificate applies;
        otherwise the same endpoint with the global status and gap.
    """
    lower_bound = getattr(objective, "nll_lower_bound", None)
    if lower_bound is None:
        return run
    gap = max(0.0, float(run.evaluation.nll) - float(lower_bound))
    if gap <= options.certified_tolerance:
        status = "converged"
    elif gap <= options.accuracy_floor:
        status = "converged_approximately"
    else:
        return run
    return _NewtonRun(
        status=status,
        params=run.params,
        blocks=run.blocks,
        dual=run.dual,
        evaluation=run.evaluation,
        iterations=run.iterations,
        evaluations=run.evaluations,
        subproblem_iterations=run.subproblem_iterations,
        decrease_bound=gap,
    )
