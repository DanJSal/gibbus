"""Newton solver for natural-coordinate fits over the exact curvature cone.

The optimizer variable is the natural parameter vector.  Each Newton model is
minimized exactly over the finite cone description of ``conic_qp`` (Gram
blocks are solver state), and the endpoint comes back with a Gram certificate
and a weak-duality gap.  Because the current iterate and the endpoint both lie
in the convex cone, every point of the segment between them does too, so the
Armijo line search needs no feasibility checks.  The Gram certificate of a
trial point is the matching convex combination of the two certificates.

Interior-point subproblems return strictly interior points, so model faces
that change the fitted model's qualitative behavior are made exact by a final
face step: on the real line, when the leading curvature coefficient has
collapsed toward zero, the embedded lower-degree model is solved on its exact
face and accepted only if it attains the same optimum to tolerance.  Nothing
is snapped: an exact face is chosen because it is optimal, never because a
coefficient is small.

The exact-arithmetic separator certifies the returned parameters.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from .._defaults import NUMERIC_FAILURES, _reraise_if_debug
from . import (  # type: ignore[attr-defined]  # compiled
    _conic_kernels,
    _curvature_certificate,
)
from .conic_qp import (
    _ConicQPResult,
    _support_representation,
)
from .separation import (
    _separate_full_curvature,
    _SeparationResult,
    _stationary_minimum_locations_exact,
)

_CONVERGED = ("converged", "converged_approximately")


@dataclass(frozen=True)
class _ConicNewtonResult:
    """Outcome of one natural-coordinate conic Newton fit.

    ``status`` is one of ``converged`` (certified within the certified
    tolerance), ``converged_approximately`` (certified bound reported in
    ``final_decrease_bound``), ``non_descent``, ``line_search_failed``,
    ``iteration_limit`` or ``uncertified`` (the exact separator rejected the
    returned parameters).  ``blocks`` is the Gram certificate of ``params``
    and can warm-start a later fit.  ``effective_curvature_degree`` and the
    ``*_amplitude_active`` flags describe the exact face the fit lies on: an
    inactive enabled amplitude is exactly zero.
    """

    status: str
    params: np.ndarray
    objective_value: float
    evaluation: object
    blocks: tuple
    dual: np.ndarray
    effective_curvature_degree: int
    lower_amplitude_active: bool
    upper_amplitude_active: bool
    newton_iterations: int
    objective_evaluations: int
    subproblem_iterations: int
    final_decrease_bound: float
    final_separation: object


@dataclass(frozen=True)
class _NewtonOptions:
    """Fixed settings shared by every Newton run of one fit."""

    tolerance: float
    certified_tolerance: float
    accuracy_floor: float
    max_iterations: int
    armijo: float
    backtrack: float
    max_line_search: int


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


_CERTIFICATE_STATUS = {1: "feasible", 0: "violated"}


def _certify(layout, params, /):
    """Certify full-curvature feasibility at tolerance ``1e-12``.

    The compiled Bernstein certificate decides first (rigorous floating-point
    error bounds, the same tolerance-shifted sign polynomial as the exact
    separator); only when it is inconclusive -- a contact within rounding
    error of the tolerance -- does the exact-dyadic separator decide.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    params : numpy.ndarray, shape (n,)
        Natural parameters to certify.

    Returns
    -------
    _SeparationResult
        ``feasible``, ``violated`` or ``uncertain``; compiled decisions carry
        no violation location.
    """
    candidate = layout.build_candidate(params)
    lower, upper = layout.support
    amplitudes = candidate.boundary_amplitudes
    code = _curvature_certificate.certify_full_curvature(
        candidate.q_d2, float(lower), float(upper), float(amplitudes[0]),
        float(amplitudes[1]), 1e-12,
    )
    if code in _CERTIFICATE_STATUS:
        return _SeparationResult(
            status=_CERTIFICATE_STATUS[code],
            violation_kind=None,
            violation_location=None,
            violation_value=None,
            isolated_roots=0,
            discarded_maxima=0,
            pruned_intervals=0,
            refined_intervals=0,
            ambiguous_intervals=0,
            root_subdivisions=0,
        )
    return _separate_full_curvature(
        candidate.q_d2,
        layout.support,
        candidate.boundary_amplitudes,
        1e-12,
        1e-10,
        80,
        2,
    )


def _interior_start(objective, /):
    """Return strictly interior natural parameters from the empirical moments.

    A Gaussian potential with the data mean and variance, plus small positive
    higher curvature terms that make every tail strictly admissible, plus
    small positive active boundary amplitudes.

    Parameters
    ----------
    objective : callable
        Natural objective; its empirical mean and variance are used when it
        exposes ``observations.stats``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Parameters strictly inside the full-curvature cone.
    """
    layout = objective.layout
    mean, variance = 0.0, 1.0
    stats = getattr(getattr(objective, "observations", None), "stats", None)
    if stats is not None and stats.moments.size > 2:
        mean = float(stats.moments[1])
        variance = float(stats.moments[2] - mean * mean)
        if not np.isfinite(variance) or variance <= 0.0:
            mean, variance = 0.0, 1.0
    degree = int(layout.curvature_degree)
    curvature = np.zeros(degree + 1, dtype=np.float64)
    # sum_j 10^(-2j) (z - mean)^(2j) / variance: positive everywhere with a
    # positive even leading term.
    for half in range(degree // 2 + 1):
        term = np.polynomial.polynomial.polypow([-mean, 1.0], 2 * half)
        curvature[: term.size] += 10.0 ** (-2 * half) * term / variance
    if degree % 2:
        # Odd degree is admissible only off the real line: add a term that is
        # nonnegative on the support and positive toward its infinite tail.
        lower, upper = layout.support
        root, sign = (lower, 1.0) if np.isfinite(lower) else (upper, -1.0)
        term = np.polynomial.polynomial.polypow([-root * sign, sign], degree)
        curvature += 10.0 ** (-2 * degree) * term / variance
    amplitudes = np.array(
        [
            0.1 if layout.lower_a_index is not None else np.nan,
            0.1 if layout.upper_a_index is not None else np.nan,
        ]
    )
    return layout.pack(-mean / variance, curvature, amplitudes)


def _infeasible_start_metric(objective, representation, params, blocks, /):
    """Return a projection metric when ``params`` is outside the cone, else ``None``.

    A start whose Gram blocks certify it (PSD blocks reproducing ``B theta``
    to roundoff) is feasible without further work; otherwise the exact
    separator decides.  The metric is the objective's Hessian at the start,
    regularized, or the identity when the start is not normalizable.

    Parameters
    ----------
    objective : callable
        Natural objective.
    representation : _ConicRepresentation
        Cone description of the starting face.
    params : numpy.ndarray, shape (n,)
        Starting parameters.
    blocks : sequence of numpy.ndarray or None
        Their claimed Gram certificate.
    """
    layout = objective.layout
    for index in (layout.lower_a_index, layout.upper_a_index):
        if index is not None and params[index] < 0.0:
            break
    else:
        shapes = [matrices.shape[1:] for matrices in representation.row_matrices]
        if blocks is not None and [np.shape(q) for q in blocks] == shapes:
            image = representation.b_matrix @ params
            residual = representation.residual(params, blocks)
            if residual <= 1e-10 * max(1.0, float(np.max(np.abs(image), initial=0.0))) and all(
                float(np.linalg.eigvalsh(0.5 * (q + q.T))[0]) >= 0.0 for q in blocks
            ):
                return None
        if _certify(layout, params).feasible:
            return None
    n = layout.n_params
    try:
        hessian = np.asarray(objective(params).hessian, dtype=np.float64)
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "conic Newton infeasible start", routine=True)
        return np.eye(n)
    return hessian + 1e-12 * max(1.0, float(np.trace(hessian)) / n) * np.eye(n)


def _default_blocks(representation, /):
    """Return identity Gram blocks shaped for ``representation``.

    Parameters
    ----------
    representation : _ConicRepresentation
        Cone description.
    """
    return tuple(np.eye(matrices.shape[1]) for matrices in representation.row_matrices)


def _project(params, representation, blocks, metric, /):
    """Project parameters onto the cone in a quadratic metric, with certificate.

    Minimizes ``1/2 (x - params)^T metric (x - params)`` over the
    description.  For a strictly interior point this returns the point
    itself with an interior Gram certificate; for a point outside a face it
    returns the nearest face point.

    Parameters
    ----------
    params : numpy.ndarray, shape (n,)
        Point to project.
    representation : _ConicRepresentation
        Target description.
    blocks : sequence of numpy.ndarray
        Positive-definite starting Gram blocks for ``representation``.
    metric : numpy.ndarray, shape (n, n)
        Positive-definite projection metric.

    Returns
    -------
    endpoint : numpy.ndarray, shape (n,)
        Projected parameters.
    blocks : tuple of numpy.ndarray
        Their Gram certificate.
    """
    _, endpoint, endpoint_blocks, _ = _preconditioned_subproblem(
        metric, np.zeros_like(params), params, representation, blocks
    )
    return endpoint, endpoint_blocks


def _preconditioned_subproblem(hessian, gradient, params, representation, blocks, /):
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
        Current natural parameters.
    representation : _ConicRepresentation
        Cone description in natural coordinates.
    blocks : sequence of numpy.ndarray
        Current Gram blocks (the starting point of the interior solve).

    Returns
    -------
    result : _ConicQPResult
        Subproblem result in scaled coordinates (for its gap and iterations).
    endpoint : numpy.ndarray, shape (n,)
        Subproblem endpoint in natural coordinates.
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
        converged=bool(gap <= 1e-12 * max(1.0, abs(float(scaled_model)))),
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
    scale = max(1.0, abs(float(run.evaluation.nll)))
    if gap <= options.certified_tolerance * scale:
        status = "converged"
    elif gap <= options.accuracy_floor * scale:
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


def _newton_on_representation(
    objective, representation, params, blocks, evaluation, options, min_steps, /
):
    """Run one fixed-face Newton solve, compiling eligible natural objectives.

    Point-data objectives expose immutable sufficient statistics to their
    fused C loop.  Interval objectives expose compressed censoring rows to a
    parallel fused loop across finite, half-infinite, and real-line supports.
    This local traversal is shared by standalone fits and mixture M-steps;
    standalone nonparametric likelihood bounds are applied only by the Python
    controller after face optimization.  Objectives whose likelihood evaluator
    remains Python-level, currently the joint mixture polish, still use a
    compiled Newton control loop and compiled conic subproblems; only their
    objective evaluations cross the Python boundary.  A declined compiled
    interval initialization is evaluated once through the
    ordinary objective and then retried in the fused traversal.

    Parameters
    ----------
    objective : callable
        Natural objective for the fixed face.
    representation : _ConicRepresentation
        Exact cone description of the face.
    params : numpy.ndarray
        Feasible starting natural parameters.
    blocks : sequence of numpy.ndarray
        Gram certificate of ``params``.
    evaluation : object
        Objective evaluation at ``params``.
    options : _NewtonOptions
        Newton and certification tolerances.
    min_steps : int
        Minimum Newton steps before an early convergence exit.

    Returns
    -------
    _NewtonRun
        Final fixed-face iterate and certification metadata.
    """
    interval_inputs = getattr(objective, "_compiled_interval_newton_inputs", None)
    if interval_inputs is not None:
        packed = interval_inputs()
        if packed is not None:
            (
                support,
                data_bounds,
                kinds,
                lengths,
                coefficients,
                controls,
                finite_intervals,
                finite_weights,
                point_lower_distance,
                point_upper_distance,
                adaptive_intervals,
                adaptive_weights,
                whole_weight,
                coordinate_scale,
                gl_nodes,
                gl_log_weights,
                width_eps_mult,
                curvature_degree,
                lower_index,
                upper_index,
                epsabs,
                epsrel,
                limit,
            ) = packed
            a_packed, sizes, a_offsets, q_offsets = representation.packed
            if evaluation is None:
                n_params = np.asarray(params, dtype=np.float64).size
                current_nll = 0.0
                current_gradient = np.zeros(n_params, dtype=np.float64)
                current_hessian = np.zeros((n_params, n_params), dtype=np.float64)
                current_fisher = np.zeros((n_params, n_params), dtype=np.float64)
                current_missing = np.zeros((n_params, n_params), dtype=np.float64)
                current_smallest = 0.0
                initialize = True
            else:
                current_nll = float(evaluation.nll)
                current_gradient = evaluation.gradient
                current_hessian = evaluation.hessian
                current_fisher = evaluation.fisher
                current_missing = evaluation.missing_information
                current_smallest = float(evaluation.smallest_curvature)
                initialize = False
            compiled = _conic_kernels.solve_interval_newton(
                params,
                representation.pack_blocks(blocks),
                representation.b_matrix,
                a_packed,
                sizes,
                a_offsets,
                q_offsets,
                representation.reference_dual,
                representation.row_degrees,
                support,
                data_bounds,
                kinds,
                lengths,
                coefficients,
                controls,
                finite_intervals,
                finite_weights,
                point_lower_distance,
                point_upper_distance,
                adaptive_intervals,
                adaptive_weights,
                whole_weight,
                coordinate_scale,
                gl_nodes,
                gl_log_weights,
                width_eps_mult,
                curvature_degree,
                lower_index,
                upper_index,
                current_nll,
                current_gradient,
                current_hessian,
                current_fisher,
                current_missing,
                current_smallest,
                options.tolerance,
                options.certified_tolerance,
                options.accuracy_floor,
                options.max_iterations,
                options.armijo,
                options.backtrack,
                options.max_line_search,
                min_steps,
                initialize,
                epsabs,
                epsrel,
                limit,
            )
            (
                status,
                theta,
                packed_blocks,
                dual,
                nll,
                gradient,
                hessian,
                fisher,
                missing,
                smallest,
                iterations,
                evaluations,
                sub_iterations,
                bound,
            ) = compiled
            if status == "fallback":
                if evaluation is not None:
                    raise RuntimeError(
                        "compiled interval Newton requested fallback after initialization"
                    )
                # The compiled initializer can decline a numerically awkward
                # starting state.  Evaluate that one point through the ordinary
                # objective, then resume the same fused traversal with explicit
                # initialized statistics; do not restart Newton in Python.
                initialized = objective(params)
                return _newton_on_representation(
                    objective, representation, params, blocks, initialized,
                    options, min_steps,
                )
            if evaluation is None:
                from .natural_objective import _NaturalIntervalEvaluation
                evaluation_type = _NaturalIntervalEvaluation
            else:
                evaluation_type = type(evaluation)
            final = evaluation_type(
                nll=float(nll),
                gradient=np.asarray(gradient, dtype=np.float64),
                hessian=np.asarray(hessian, dtype=np.float64),
                observed_hessian=np.asarray(fisher - missing, dtype=np.float64),
                fisher=np.asarray(fisher, dtype=np.float64),
                missing_information=np.asarray(missing, dtype=np.float64),
                smallest_curvature=float(smallest),
            )
            return _NewtonRun(
                status=status,
                params=theta,
                blocks=representation.unpack_blocks(packed_blocks),
                dual=dual,
                evaluation=final,
                iterations=iterations,
                evaluations=evaluations,
                subproblem_iterations=sub_iterations,
                decrease_bound=bound,
            )

    inputs = getattr(objective, "_compiled_point_newton_inputs", None)
    if inputs is not None:
        packed = inputs()
        if packed is not None:
            (
                support,
                data_bounds,
                lower_basis,
                upper_basis,
                kinds,
                lengths,
                coefficients,
                controls,
                empirical_means,
                coordinate_constant,
                curvature_degree,
                lower_index,
                upper_index,
                epsabs,
                epsrel,
                limit,
            ) = packed
            a_packed, sizes, a_offsets, q_offsets = representation.packed
            compiled = _conic_kernels.solve_point_newton(
                params,
                representation.pack_blocks(blocks),
                representation.b_matrix,
                a_packed,
                sizes,
                a_offsets,
                q_offsets,
                representation.reference_dual,
                representation.row_degrees,
                support,
                data_bounds,
                lower_basis,
                upper_basis,
                kinds,
                lengths,
                coefficients,
                controls,
                empirical_means,
                coordinate_constant,
                curvature_degree,
                lower_index,
                upper_index,
                float(evaluation.nll),
                evaluation.gradient,
                evaluation.hessian,
                evaluation.model_partial_means,
                options.tolerance,
                options.certified_tolerance,
                options.accuracy_floor,
                options.max_iterations,
                options.armijo,
                options.backtrack,
                options.max_line_search,
                min_steps,
                epsabs,
                epsrel,
                limit,
            )
            (
                status,
                theta,
                packed_blocks,
                dual,
                nll,
                gradient,
                hessian,
                means,
                iterations,
                evaluations,
                sub_iterations,
                bound,
            ) = compiled
            from .objective import _ObjectiveEvaluation

            final = _ObjectiveEvaluation(
                nll=nll,
                gradient=gradient,
                hessian=hessian,
                fisher=hessian,
                missing_information=np.zeros_like(hessian),
                model_partial_means=means,
            )
            return _NewtonRun(
                status=status,
                params=theta,
                blocks=representation.unpack_blocks(packed_blocks),
                dual=dual,
                evaluation=final,
                iterations=iterations,
                evaluations=evaluations,
                subproblem_iterations=sub_iterations,
                decrease_bound=bound,
            )
    a_packed, sizes, a_offsets, q_offsets = representation.packed
    compiled = _conic_kernels.solve_callback_newton(
        objective,
        params,
        representation.pack_blocks(blocks),
        representation.pack_blocks(_default_blocks(representation)),
        representation.b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        representation.reference_dual,
        representation.row_degrees,
        evaluation,
        options.tolerance,
        options.certified_tolerance,
        options.accuracy_floor,
        options.max_iterations,
        options.armijo,
        options.backtrack,
        options.max_line_search,
        min_steps,
    )
    (
        status, theta, packed_blocks, dual, final, iterations, evaluations,
        sub_iterations, bound,
    ) = compiled
    return _NewtonRun(
        status=status,
        params=np.asarray(theta, dtype=np.float64),
        blocks=representation.unpack_blocks(packed_blocks),
        dual=np.asarray(dual, dtype=np.float64),
        evaluation=final,
        iterations=int(iterations),
        evaluations=int(evaluations),
        subproblem_iterations=int(sub_iterations),
        decrease_bound=float(bound),
    )

def _amplitude_release_gain(layout, params, gradient, side, free_indices, /):
    """Reduced gradient of a boundary amplitude fixed at zero, from contacts.

    Lifting a zero amplitude through ``p (z-L)^2 + a`` forces a double root at
    the endpoint, so the conic subproblem is degenerate exactly there and
    cannot certify the face.  The face is instead tested on the original
    semi-infinite KKT conditions: at a face optimum
    ``grad f = sum_j lambda_j grad F(z_j)`` over the contacts ``z_j`` where the
    full curvature ``F`` touches zero, with ``lambda >= 0`` (recovered by
    NNLS), and keeping the amplitude at zero is optimal iff
    ``df/da >= sum_j lambda_j dF(z_j)/da``.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    params : numpy.ndarray, shape (n,)
        Face optimum.
    gradient : numpy.ndarray, shape (n,)
        NLL gradient at ``params``.
    side : str
        ``"lower"`` or ``"upper"``: the amplitude fixed at zero.
    free_indices : sequence of int
        Cone coordinates free on the face (represented curvature
        coefficients and free amplitudes).

    Returns
    -------
    gain : float
        ``df/da - sum_j lambda_j dF(z_j)/da``; negative means releasing the
        amplitude decreases the NLL to first order.
    residual : float
        Relative NNLS residual of the stationarity fit (small when the
        contact set is complete).
    """
    candidate = layout.build_candidate(params)
    q_d2 = candidate.q_d2
    amplitudes = np.nan_to_num(candidate.boundary_amplitudes, nan=0.0)
    lower, upper = layout.support
    polyval = np.polynomial.polynomial.polyval

    def full_curvature(z):
        value = float(polyval(z, q_d2))
        if np.isfinite(lower) and amplitudes[0] > 0.0 and z > lower:
            value += amplitudes[0] / (z - lower) ** 2
        if np.isfinite(upper) and amplitudes[1] > 0.0 and z < upper:
            value += amplitudes[1] / (upper - z) ** 2
        return value

    scale = max(1.0, float(np.max(np.abs(q_d2))))
    endpoint = lower if side == "lower" else upper
    if full_curvature(endpoint) <= 1e-7 * scale:
        # Contact at the fixed amplitude's own endpoint: a positive amplitude
        # relaxes the curvature there without bound.
        return -np.inf, 0.0
    minima, _ = _stationary_minimum_locations_exact(
        q_d2, np.asarray(layout.support), candidate.boundary_amplitudes, 1e-10, 80
    )
    contacts = [float(z) for z in minima if full_curvature(float(z)) <= 1e-7 * scale]

    def derivative(z, index):
        start = layout.curvature_slice.start
        if start <= index < layout.curvature_slice.stop:
            return z ** (index - start)
        if index == layout.lower_a_index:
            return 1.0 / (z - lower) ** 2
        return 1.0 / (upper - z) ** 2

    g = np.asarray(gradient, dtype=np.float64)
    free = list(free_indices)
    target = g[free]
    if contacts:
        design = np.array([[derivative(z, index) for z in contacts] for index in free])
        weights, residual = nnls(design, target)
    else:
        weights, residual = np.zeros(0), float(np.linalg.norm(target))
    index = layout.lower_a_index if side == "lower" else layout.upper_a_index
    gain = float(g[index]) - sum(
        weight * derivative(z, index) for weight, z in zip(weights, contacts, strict=True)
    )
    return gain, residual / max(1.0, float(np.linalg.norm(target)))


def _solve_natural_conic(
    objective,
    /,
    *,
    initial=None,
    initial_blocks=None,
    tolerance=1e-12,
    certified_tolerance=1e-10,
    accuracy_floor=1e-7,
    face_trigger=1e-6,
    max_iterations=60,
    armijo=1e-4,
    backtrack=0.5,
    max_line_search=40,
    certify=True,
):
    """Fit a natural objective over the exact full-curvature cone.

    Works on every support geometry (real line, half-lines, bounded
    intervals) with or without boundary amplitudes.

    Parameters
    ----------
    objective : callable
        ``objective(theta)`` returns an evaluation with ``nll``, ``gradient``
        and ``hessian``; ``objective.layout`` is the natural layout.  Any
        objective built from sufficient statistics works, which is what
        weighted fits, EM for censoring, and mixture M-steps supply.
    initial : numpy.ndarray or None, optional
        Starting parameters, for example an earlier result's ``params``.
        An infeasible start is projected onto the cone first.  Defaults to a
        near-Gaussian interior point from the empirical moments.
    initial_blocks : sequence of numpy.ndarray or None, optional
        Gram blocks to warm-start the first subproblem (an earlier result's
        ``blocks``); ignored when shaped for a different face.
    tolerance : float, optional
        Stopping target: iterate until predicted decrease plus certified
        subproblem gap is at most ``tolerance * max(1, |nll|)``.
    certified_tolerance : float, optional
        When the subproblem can no longer certify progress (its gap floor is
        about ``1e-11`` at unit scale), a certified bound within
        ``certified_tolerance * max(1, |nll|)`` still counts as converged, so
        ``converged`` always means certified within this bound.
    accuracy_floor : float, optional
        A larger certified bound, up to ``accuracy_floor * max(1, |nll|)``, is
        reported as ``converged_approximately`` with the bound in
        ``final_decrease_bound``.  This happens at degenerate optima, for
        example degree 10 on 30 points, where the optimal Gram matrix is rank
        one and contacts are weakly active.
    face_trigger : float, optional
        Size below which a boundary amplitude, or (relative to the largest
        curvature coefficient) a leading curvature coefficient on a support
        with an infinite tail, prompts an exact face solve: the amplitude
        fixed to zero, or the degree lowered.  A face is accepted only if its
        optimum matches the current optimum within ``tolerance``.
    max_iterations : int, optional
        Newton iteration limit per cone description.
    armijo, backtrack : float, optional
        Armijo sufficient-decrease constant and backtracking factor.
    max_line_search : int, optional
        Line-search trial limit per Newton step.
    certify : bool, optional
        Run the exact separator on the returned parameters.

    Returns
    -------
    _ConicNewtonResult
        Fitted parameters, their Gram certificate, status and counters.

    Raises
    ------
    RuntimeError
        If a cone description fails its rank checks.
    """
    layout = objective.layout
    options = _NewtonOptions(
        tolerance=float(tolerance),
        certified_tolerance=float(certified_tolerance),
        accuracy_floor=float(accuracy_floor),
        max_iterations=int(max_iterations),
        armijo=float(armijo),
        backtrack=float(backtrack),
        max_line_search=int(max_line_search),
    )
    n = layout.n_params
    enabled = {"lower": layout.lower_a_index, "upper": layout.upper_a_index}
    # Infinite censoring rows can make an enabled boundary amplitude converge
    # very slowly toward its exact zero face through the degenerate full-cone
    # representation.  Cold/uncertified censored starts therefore begin on
    # the zero-amplitude faces and release a term below only when the contact
    # KKT test predicts a strict likelihood improvement.  Certified warm
    # starts preserve genuinely positive amplitudes, and finite-row/point
    # problems retain the established full-face start.
    observations = getattr(objective, "observations", None)
    zero_first = bool(getattr(observations, "has_infinite_rows", False))
    active = {side: index is not None for side, index in enabled.items()}
    if zero_first:
        if initial is None or initial_blocks is None:
            active = dict.fromkeys(enabled, False)
        else:
            initial_array = np.asarray(initial, dtype=np.float64).reshape(-1)
            active = {
                side: index is not None and float(initial_array[index]) > face_trigger
                for side, index in enabled.items()
            }
    effective = int(layout.curvature_degree)
    counters = {"iterations": 0, "evaluations": 0, "sub": 0}

    def describe(face_effective, face_active):
        """Cone description of one face."""
        return _support_representation(
            layout, face_effective, face_active["lower"], face_active["upper"]
        )

    def solve_face(face_effective, face_active, start, start_blocks, metric, min_steps):
        """Run Newton on a face; ``None`` if its start is not normalizable.

        With ``metric`` the start is first projected onto the face (moving to
        a smaller face); without it the start must already be feasible there.
        """
        face = describe(face_effective, face_active)
        shapes = [matrices.shape[1:] for matrices in face.row_matrices]
        blocks = _default_blocks(face)
        if start_blocks is not None and [np.shape(q) for q in start_blocks] == shapes:
            blocks = tuple(np.asarray(q, dtype=np.float64) for q in start_blocks)
        theta = np.asarray(start, dtype=np.float64)
        if metric is not None:
            theta, blocks = _project(theta, face, blocks, metric)
        counters["evaluations"] += 1
        if getattr(objective, "_compiled_interval_newton_eligible", False):
            evaluation = None
        else:
            try:
                evaluation = objective(theta)
            except NUMERIC_FAILURES as exc:
                # The projected start is not normalizable; this face is skipped.
                _reraise_if_debug(exc, "conic Newton face start", routine=True)
                return None
        face_run = _newton_on_representation(
            objective, face, theta, blocks, evaluation, options, min_steps
        )
        counters["iterations"] += face_run.iterations
        counters["evaluations"] += face_run.evaluations
        counters["sub"] += face_run.subproblem_iterations
        return face_run

    def metric_at(current):
        """Projection metric: the Fisher matrix, regularized to be PD."""
        hessian = np.asarray(current.evaluation.hessian, dtype=np.float64)
        return hessian + 1e-12 * max(1.0, float(np.trace(hessian)) / n) * np.eye(n)

    def better(candidate, incumbent):
        """Whether a face run should replace the incumbent run.

        Every run ends at a feasible point, so its NLL bounds the optimum
        from above even when its certificate is incomplete: a face is
        accepted only if it converged and is no worse than the incumbent.
        """
        if candidate is None or candidate.status not in _CONVERGED:
            return False
        scale = max(1.0, abs(float(incumbent.evaluation.nll)))
        return float(candidate.evaluation.nll) <= (
            float(incumbent.evaluation.nll) + options.tolerance * scale
        )

    if initial is None:
        start, metric = _interior_start(objective), None
    else:
        start = np.asarray(initial, dtype=np.float64).copy()
        metric = None
    # Exact zero faces require exact zero amplitudes in the natural vector.
    # The generic interval start has small positive amplitudes by design; clear
    # only terms selected for the zero-first censored face.  Caller-provided
    # starts are projected if the remaining curvature is not feasible there.
    for side, index in enabled.items():
        if index is not None and not active[side]:
            start[index] = 0.0
    if initial is not None:
        metric = _infeasible_start_metric(
            objective, describe(effective, active), start, initial_blocks
        )
    # Every Armijo segment must lie in the cone, so an infeasible start is
    # first projected onto it (in the start's Fisher metric).
    run = solve_face(effective, dict(active), start, initial_blocks, metric, 0)
    if run is None:
        raise RuntimeError("starting point of the conic fit is not normalizable")

    # Active set over exact faces.  Boundary amplitudes in infinite-censoring
    # cold starts begin at exact zero and are released when the contact KKT
    # test says a positive amplitude would help; other starts retain the
    # established collapse-to-zero behavior.  A leading
    # curvature coefficient collapsing on a support with an infinite tail
    # lowers the effective degree.  Faces are accepted only if optimal.
    degree_step = {"real_line": 2, "bounded": 0}.get(layout.support_kind, 1)
    released = set()
    for _ in range(8):
        changed = False
        for side, index in enabled.items():
            if not active[side] or side in released:
                continue
            collapsed = float(run.params[index]) <= face_trigger
            # Near a = 0 the full description is degenerate (a double root of
            # the product polynomial at the endpoint), so the interior point
            # may stop at a small positive amplitude with only an approximate
            # certificate.  Any run not certified to ``converged`` therefore
            # also tries the zero face, which is accepted only if no worse.
            if run.status == "converged" and not collapsed:
                continue
            face_active = dict(active)
            face_active[side] = False
            face_run = solve_face(
                effective, face_active, run.params, None, metric_at(run), 1
            )
            if better(face_run, run):
                run, active, changed = face_run, face_active, True
        for side, index in enabled.items():
            if index is None or active[side] or run.status not in _CONVERGED:
                continue
            free = list(
                range(
                    layout.curvature_slice.start,
                    layout.curvature_slice.start + effective + 1,
                )
            ) + [enabled[other] for other in enabled if active[other]]
            gain, _ = _amplitude_release_gain(
                layout, run.params, run.evaluation.gradient, side, free
            )
            gradient_scale = max(1.0, float(np.max(np.abs(run.evaluation.gradient))))
            if gain >= -1e-9 * gradient_scale:
                continue
            release_active = dict(active)
            release_active[side] = True
            start = np.array(run.params, dtype=np.float64)
            start[index] = 1e-3
            # Raising a zero amplitude only relaxes the cone: already feasible.
            release_run = solve_face(effective, release_active, start, None, None, 1)
            scale = max(1.0, abs(float(run.evaluation.nll)))
            if release_run is not None and release_run.status in _CONVERGED and float(
                release_run.evaluation.nll
            ) < float(run.evaluation.nll) - options.tolerance * scale:
                run, active, changed = release_run, release_active, True
                released.add(side)
        curvature = run.params[layout.curvature_slice]
        if (
            run.status in _CONVERGED
            and degree_step
            and effective >= degree_step
            and abs(float(curvature[effective]))
            <= face_trigger * max(1.0, float(np.max(np.abs(curvature))))
        ):
            face_run = solve_face(
                effective - degree_step, active, run.params, None, metric_at(run), 1
            )
            if better(face_run, run):
                run, effective, changed = face_run, effective - degree_step, True
        if not changed:
            break
    # Face identification is based only on locally certified fixed-face
    # solves.  Once that local active-set search is complete, overlay any
    # observation-level global NLL certificate.  This keeps degree/amplitude
    # KKT decisions independent of the nonparametric selection certificate.
    run = _apply_global_objective_certificate(objective, run, options)
    iterations = counters["iterations"]
    evaluations = counters["evaluations"]
    sub_iterations = counters["sub"]
    lower_active, upper_active = active["lower"], active["upper"]

    status = run.status
    separation = _certify(layout, run.params) if certify else None
    if certify and not separation.feasible and status in _CONVERGED:
        status = "uncertified"
    return _ConicNewtonResult(
        status=status,
        params=run.params,
        objective_value=float(run.evaluation.nll),
        evaluation=run.evaluation,
        blocks=tuple(run.blocks),
        dual=run.dual,
        effective_curvature_degree=effective,
        lower_amplitude_active=lower_active,
        upper_amplitude_active=upper_active,
        newton_iterations=iterations,
        objective_evaluations=evaluations,
        subproblem_iterations=sub_iterations,
        final_decrease_bound=float(run.decrease_bound),
        final_separation=separation,
    )
