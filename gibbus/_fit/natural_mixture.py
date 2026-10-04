"""Mixtures of natural-coordinate log-concave components.

Component polynomial shapes are private; each enabled physical boundary
amplitude is one shared variable. A fit runs in three phases:

1. **EM**, accelerated by SQUAREM (Varadhan and Roland, 2008).  The E-step
   computes exact posteriors from point densities or interval masses; the
   M-step updates weights in closed form and jointly refits the shapes and
   shared amplitudes to responsibility-mass-weighted objectives. Point
   M-steps are convex; interval M-steps use the combined observed information.
   Independent fused component solves are used only when no amplitude is
   shared. Exact global amplitude masks and private degree faces are compared
   by actual coupled fits, including releases and degree re-expansion.
2. **Joint Newton polish** on the whole observed-data likelihood over all
   component parameters and the mixture logits at once, over the product of
   the components' cone descriptions.  The observed Hessian is the
   complete-data Fisher matrix ``G`` (block diagonal: ``m_k F_k`` per
   component, the multinomial Hessian for the logits) minus the missing
   information ``M`` (the covariance of the complete-data scores over the
   unknown labels, plus the within-row covariances of interval rows).  The
   raw complete and missing information are pulled back into the reduced
   physical coordinates and restricted to the exact face before the
   solver receives the same metric as a single interval fit: the observed
   Hessian where it is positive definite, which it is at a strict local
   maximum, so the polish converges quadratically; its saddle-free
   reflection elsewhere, so it moves away from saddle points (two components
   trading mass along a ridge) instead of stalling near them.
3. **Verification**: one more EM step from the polished point.  At a
   stationary point of the observed likelihood the EM surrogate is already
   minimized there, so the step cannot raise the likelihood; if it does (the
   polish sat on the wrong face), the fit returns to phase 1 from there.

Every component keeps the fixed numerical coordinate chosen from its initial
responsibilities, so all phases share one parameterization and warm starts
are exact.  The observed mixture likelihood is not concave: results are
certified stationary points, and the fit keeps the best of several
initializations.
"""

import functools
import weakref
from dataclasses import dataclass, replace
from itertools import pairwise, product

import numpy as np

from .._defaults import (
    AUTO_POLY_DEGREE_MIN,
    EM_MIN_EFFECTIVE_DISTINCT_N,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._model.coords import _build_fit_coordinate, _build_interval_fit_coordinate
from .._model.natural_state import _natural_partials, _NaturalCoreState
from .._model.spec import _LOGDIST, _LOWER, _build_model_spec
from .._observations.empirical import (
    _normalized_weights,
    _WeightSummary,
)
from .._observations.intervals import (
    _build_interval_observations,
    _row_grouping,
)
from .._observations.points import _PointObservations
from ._shared_mixture_kernels import JointMixtureObjective, SharedMStepObjective
from .conic_newton import (
    _CONVERGED,
    _certify,
    _ConicNewtonResult,
    _default_blocks,
    _interior_start,
    _NewtonOptions,
    _NewtonRun,
    _project,
    _solve_natural_conic,
)
from .inputs import _admissible_degrees
from .mixture import (
    _distinct_location_index,
    _effective_distinct_point_count,
    _initial_responsibility_candidates,
    _interval_initial_representatives,
    _point_order,
)
from .mixture_geometry import _MixtureNaturalMap
from .natural_objective import (
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points_auto,
    _interval_kernel_inputs,
    _interval_partitions,
    _natural_interval_start,
    _natural_point_stats,
    _NaturalIntervalEvaluation,
    _NaturalIntervalObjectiveFunction,
    _NaturalPointObjectiveFunction,
    _point_kernel_inputs,
    _preserved_point_boundary_distances,
)
from .objective import (
    _ObjectiveEvaluation,
)


@dataclass(frozen=True)
class _EMOptions:
    """Immutable EM scheduling policy for one mixture fit.

    Parameters
    ----------
    tolerance : float, optional
        Relative log-likelihood increase that ends an EM phase.
    max_steps : int, optional
        EM map evaluation limit per phase.
    max_rounds : int, optional
        EM-polish-verify round limit.
    accelerate : bool, optional
        Whether to use SQUAREM acceleration.
    """

    tolerance: float = 1e-4
    max_steps: int = 20
    max_rounds: int = 4
    accelerate: bool = True


@dataclass(frozen=True)
class _MixtureSearchOptions:
    """Immutable multistart search policy for one mixture fit.

    Parameters
    ----------
    paths : tuple of (str, str), optional
        ``(schedule, treatment)`` pairs explored from every initializer.
    finalists : int, optional
        Number of best explorations run to certified convergence.
    """

    paths: tuple = (
        ("ladder", "raw"),
        ("ladder", "sharpened"),
        ("direct", "raw"),
        ("direct", "sharpened"),
    )
    finalists: int = 2


def _resolve_search_options(search_options, /):
    """Return one validated immutable multistart search policy.

    Parameters
    ----------
    search_options : _MixtureSearchOptions or None
        Explicit policy, or ``None`` for the mixture defaults.

    Returns
    -------
    _MixtureSearchOptions
        Search policy used by the multistart controller.
    """
    if search_options is None:
        return _MixtureSearchOptions()
    if not isinstance(search_options, _MixtureSearchOptions):
        raise TypeError("search_options must be _MixtureSearchOptions or None")
    return search_options


def _resolve_newton_options(solver_options, /):
    """Return one validated immutable Newton policy.

    Parameters
    ----------
    solver_options : _NewtonOptions or None
        Explicit policy, or ``None`` for the shared defaults.

    Returns
    -------
    _NewtonOptions
        Policy used by every component, coupled M-step and polish solve.
    """
    if solver_options is None:
        return _NewtonOptions()
    if not isinstance(solver_options, _NewtonOptions):
        raise TypeError("solver_options must be _NewtonOptions or None")
    return solver_options


def _resolve_em_options(em_options, /):
    """Return one validated immutable EM policy.

    Parameters
    ----------
    em_options : _EMOptions or None
        Explicit policy, or ``None`` for the mixture defaults.

    Returns
    -------
    _EMOptions
        Scheduling policy used by the EM controller.
    """
    if em_options is None:
        return _EMOptions()
    if not isinstance(em_options, _EMOptions):
        raise TypeError("em_options must be _EMOptions or None")
    return em_options


@dataclass(frozen=True)
class _NaturalComponent:
    """One fitted mixture component in its fixed natural coordinate.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed coordinate chosen from the initial responsibilities.
    spec : _ModelSpec
        Model specification in that coordinate.
    layout : _NaturalLayout
        Natural parameter layout.
    z_data_bounds : tuple of (float, float)
        Canonical data landmarks used by quadrature windowing.
    params : numpy.ndarray
        Natural parameters.
    effective_curvature_degree : int
        Exact face: effective curvature degree.
    lower_amplitude_active, upper_amplitude_active : bool
        Exact face: whether each enabled amplitude is free (inactive means
        exactly zero).
    solver_result : _ConicNewtonResult
        Matching component geometry and feasibility certificate. For coupled
        solves its status is ``joint_feasible``; stationarity belongs to the
        mixture, not to independent component optimizations.
    """

    coordinate: object
    spec: object
    layout: object
    z_data_bounds: tuple
    params: np.ndarray
    effective_curvature_degree: int
    lower_amplitude_active: bool
    upper_amplitude_active: bool
    solver_result: object

    def state(self):
        """Return the normalized component state."""
        return _NaturalCoreState(
            self.coordinate, self.layout, self.params, self.z_data_bounds
        )


@dataclass(frozen=True)
class _NaturalMixtureFit:
    """Outcome of one natural mixture fit.

    Parameters
    ----------
    components : tuple of _NaturalComponent
        Fitted components.
    weights : numpy.ndarray, shape (K,)
        Mixture weights.
    log_likelihood : float
        Weighted mean observed-data log likelihood in user coordinates
        (log densities for points, log masses for intervals).
    status : str
        ``converged`` (polish certified to the certified tolerance and the
        verification EM step did not move the likelihood), or
        ``converged_approximately`` (same, certified bound in
        ``decrease_bound``); otherwise the status of the last polish, or
        ``em_only`` when no polish succeeded.
    decrease_bound : float
        Certified bound of the final polish: the largest decrease of the
        local model the solver could not exclude.
    em_iterations : int
        EM map evaluations over all phases.
    polish_iterations : int
        Joint Newton iterations over all polishes.
    rounds : int
        EM-polish-verify rounds.
    history : tuple of float
        Log likelihood after every accepted EM step and every polish.
    initialization : str
        Name of the responsibility initialization that won.
    separator_certified : bool
        Whether the exact separator certified every component's curvature.
    responsibilities : numpy.ndarray, shape (R, K)
        Posterior component probabilities at the fitted mixture.
    n_parameters, n_face_parameters : int
        Requested reduced dimension and effective exact-face dimension,
        including mixture logits but excluding normalization and Gram entries.
    observed_information : numpy.ndarray
        Raw final observed Hessian in the full requested reduced coordinates.
    shared_parameter_indices : tuple
        Physical lower and upper amplitude indices, or ``None`` if excluded.
    free_parameter_indices : tuple
        Exact face's free indices into the reduced observed objective.
    degree_diagnostics : tuple
        Per-component diagnostics retained from automatic/shared degree growth.
    """

    components: tuple
    weights: np.ndarray
    log_likelihood: float
    status: str
    decrease_bound: float
    em_iterations: int
    polish_iterations: int
    rounds: int
    history: tuple
    initialization: str
    separator_certified: bool
    responsibilities: np.ndarray
    n_parameters: int = 0
    n_face_parameters: int = 0
    observed_information: np.ndarray | None = None
    shared_parameter_indices: tuple = (None, None)
    free_parameter_indices: tuple = ()
    degree_diagnostics: tuple = ()


@dataclass(frozen=True)
class _EMState:
    """One evaluated EM iterate with its reusable posterior.

    Parameters
    ----------
    results : tuple of _ConicNewtonResult
        Component states at this iterate.
    log_weights : numpy.ndarray, shape (K,)
        Normalized log mixture weights.
    log_likelihood : float
        Observed log likelihood at this iterate.
    responsibilities : numpy.ndarray, shape (R, K) or (B, K)
        Posterior component probabilities at this iterate.  Duplicated interval
        fits retain one row per distinct interval internally.
    """

    results: tuple
    log_weights: np.ndarray
    log_likelihood: float
    responsibilities: np.ndarray


@dataclass(frozen=True)
class _EMContinuation:
    """Reusable state left by a finalist exploration.

    Parameters
    ----------
    problems : tuple of _ComponentProblem
        Fixed-coordinate component problems.
    layouts : tuple of _NaturalLayout
        Component natural layouts.
    state : _EMState
        Last evaluated EM iterate.
    history : tuple of float
        Accepted log-likelihood history through the exploration.
    em_iterations : int
        EM map evaluations already completed.
    rounds : int
        EM-polish rounds already entered.
    """

    problems: tuple
    layouts: tuple
    state: _EMState
    history: tuple
    em_iterations: int
    rounds: int


def _observation_weights(n, weights, /):
    """Use boundary-normalized observation weights, creating uniform ones if absent.

    Parameters
    ----------
    n : int
        Number of observations.
    weights : numpy.ndarray, shape (n,), dtype float64 or None
        Boundary-validated weights summing to one, or ``None`` for uniform.

    Returns
    -------
    numpy.ndarray, shape (n,), dtype float64
        The supplied canonical array, or newly created uniform weights.
    """
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return weights


_INTERVAL_COORDINATE_GEOMETRY = None
"""Cached midpoint/width geometry for the last frozen interval-row array.

The sample rows are immutable during a fit, while mixture starts and components
reuse them with different responsibility weights.  Midpoints, widths and their
sort orders therefore need to be formed only once.
"""


def _interval_coordinate_geometry(rows, /):
    """Return reusable finite-interval midpoint/width geometry.

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 2)
        Finite, read-only interval rows in user coordinates.

    Returns
    -------
    tuple
        ``(midpoints, widths, midpoint_order, width_order)``.
    """
    global _INTERVAL_COORDINATE_GEOMETRY
    entry = _INTERVAL_COORDINATE_GEOMETRY
    if entry is not None and entry[0]() is rows:
        return entry[1]

    lo = np.asarray(rows[:, 0], dtype=np.float64)
    hi = np.asarray(rows[:, 1], dtype=np.float64)
    mid = np.ascontiguousarray(0.5 * lo + 0.5 * hi, dtype=np.float64)
    width = np.ascontiguousarray(hi - lo, dtype=np.float64)
    mid_order = np.argsort(mid, kind="stable")
    width_order = np.argsort(width[np.isfinite(width)], kind="stable")
    for array in (mid, width, mid_order, width_order):
        array.setflags(write=False)
    geometry = (mid, width, mid_order, width_order)
    if isinstance(rows, np.ndarray) and not rows.flags.writeable:
        _INTERVAL_COORDINATE_GEOMETRY = (weakref.ref(rows), geometry)
    return geometry


def _basis(layout, support, z, /, *, lower_distance=None, upper_distance=None):
    """Return the natural potential basis ``h(z)`` with ``q = theta . h``.

    Preserved boundary distances are used for logarithmic columns when an
    affine coordinate rounds a physically interior point onto a finite
    canonical endpoint.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    support : tuple of (float, float)
        Canonical support.
    z : numpy.ndarray, shape (R,)
        Canonical points.
    lower_distance, upper_distance : array_like or None, optional
        Preserved canonical point-to-boundary distances.
    """
    values = np.asarray(z, dtype=np.float64)
    lower, upper = map(float, support)
    columns = []
    for partial in _natural_partials(layout):
        if partial.kind != _LOGDIST:
            columns.append(partial.evaluate(values, support))
            continue
        if partial.boundary_side == _LOWER:
            distance = (
                values - lower
                if lower_distance is None
                else np.asarray(lower_distance, dtype=np.float64)
            )
        else:
            distance = (
                upper - values
                if upper_distance is None
                else np.asarray(upper_distance, dtype=np.float64)
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            columns.append(-np.log(distance))
    return np.column_stack(columns)


class _ComponentProblem:
    """Fixed data, coordinate and specification of one mixture component.

    Builds the responsibility-weighted objective for each M-step and the
    per-row likelihood contributions and conditional statistics used by the
    E-step and the joint objective.
    """

    def __init__(
        self,
        support,
        rows,
        degree,
        lower,
        upper,
        initial_weights,
        /,
        *,
        _coordinate=None,
    ):
        """Fix the component coordinate from its initial responsibilities.

        Parameters
        ----------
        support : tuple of (float, float)
            User-coordinate support.
        rows : numpy.ndarray, shape (R, 1) or (R, 2)
            Point samples or interval rows in user coordinates.
        degree : int
            Requested polynomial degree.
        lower, upper : bool
            Boundary amplitude flags.
        initial_weights : numpy.ndarray, shape (R,)
            Initial responsibility-weighted observation weights.
        _coordinate : _FitCoordinate or None, optional
            Previously fixed coordinate for numerical continuation and probes.
        """
        self.rows = rows
        self.intervals = rows.shape[1] == 2
        self._distinct = None
        self._grouping_cache = None
        self._observation_group_cache = None
        self._compact_observation_template = None
        self.lower_distance = None
        self.upper_distance = None
        if not self.intervals:
            x = rows[:, 0]
            coordinate = (
                _build_fit_coordinate(
                    support, x, initial_weights, None, order=_point_order(rows)[0]
                )
                if _coordinate is None
                else _coordinate
            )
            self.z = coordinate.to_canonical(x)
            self.lower_distance, self.upper_distance = (
                _preserved_point_boundary_distances(coordinate, x)
            )
            bounds = (float(np.min(self.z)), float(np.max(self.z)))
        else:
            # Group exact duplicate intervals before choosing the component's
            # robust coordinate.  Weighted medians, MADs, interval-width
            # floors and RMS fallbacks depend on duplicate rows only through
            # their summed weights, so this is algebraically equivalent to
            # scanning the expanded sample while reducing binned data from
            # O(N) to O(N_unique) work for every mixture start/component.
            first, inverse, n_unique = _row_grouping(rows)
            coordinate_rows = rows
            coordinate_weights = initial_weights
            if n_unique < rows.shape[0]:
                coordinate_rows = np.ascontiguousarray(rows[first], dtype=np.float64)
                coordinate_weights = np.bincount(
                    inverse,
                    weights=np.asarray(initial_weights, dtype=np.float64),
                    minlength=n_unique,
                ).astype(np.float64)
            if _coordinate is not None:
                coordinate = _coordinate
            elif np.all(np.isfinite(coordinate_rows)):
                mid, width, mid_order, width_order = _interval_coordinate_geometry(
                    coordinate_rows
                )
                coordinate = _build_fit_coordinate(
                    support,
                    mid,
                    coordinate_weights,
                    width,
                    order=mid_order,
                    width_order=width_order,
                )
            else:
                coordinate = _build_interval_fit_coordinate(
                    support, coordinate_rows, coordinate_weights
                )
            if n_unique < rows.shape[0]:
                # For duplicated interval data no later likelihood calculation
                # needs expanded canonical rows: component statistics are
                # identical within each duplicate group.  Transform only the
                # distinct physical rows and retain the inverse map.
                z_distinct = coordinate.intervals_to_canonical(rows[first])
                self.z = np.ascontiguousarray(z_distinct, dtype=np.float64)
                self._distinct = (self.z, inverse)
            else:
                self.z = coordinate.intervals_to_canonical(rows)
                self._distinct = None
            finite = self.z[np.isfinite(self.z)]
            bounds = (float(np.min(finite)), float(np.max(finite)))
            # Per-row statistics are computed once per distinct interval.  Rows
            # are grouped in user coordinates, so every component shares the
            # same inverse mapping.
            self._grouping_cache = {"grouping": (first, inverse, n_unique)}
            if self._distinct is not None and coordinate.support_kind == "real_line":
                # Canonical distinct rows and their endpoint geometry never
                # change during EM.  Cache the immutable observation shell and
                # replace only its normalized weights/effective-n in M-steps.
                self._compact_observation_template = _build_interval_observations(
                    self._distinct[0],
                    support=coordinate.canonical_support,
                    deduplicate=False,
                )
        self.coordinate = coordinate
        self.spec = _build_model_spec(coordinate, int(degree), bool(lower), bool(upper))
        self.z_data_bounds = bounds
        self.order = max(4, 2 * int(self.spec.effective_poly_degree))
        self.log_scale = float(np.log(coordinate.scale))
        self._point_basis = None
        self._mixture_map = None
        self._compiled_mixture = None

    @property
    def distinct_rows(self):
        """``(distinct canonical rows, inverse index)`` of interval data, or ``None``."""
        return self._distinct

    def point_basis(self, layout, support, /):
        """Return the natural basis ``h(z)`` at this problem's fixed points.

        It depends only on the layout and the fixed canonical points, so it
        is built once per layout rather than at every joint evaluation.

        Parameters
        ----------
        layout : _NaturalLayout
            Natural layout.
        support : tuple of (float, float)
            Canonical support.
        """
        cached = self._point_basis
        if cached is not None and cached[0] is layout:
            return cached[1]
        basis = _basis(
            layout,
            support,
            self.z,
            lower_distance=self.lower_distance,
            upper_distance=self.upper_distance,
        )
        basis.setflags(write=False)
        self._point_basis = (layout, basis)
        return basis

    def objective(self, weights, /):
        """Return the M-step objective for responsibility-weighted rows.

        Parameters
        ----------
        weights : numpy.ndarray, shape (R,)
            Nonnegative M-step weights (observation weight times
            responsibility).
        """
        spec = self.spec
        if not self.intervals:
            stats = _natural_point_stats(
                self.z,
                weights,
                self.order,
                spec.support,
                spec.canonical_lower_a_index is not None,
                spec.canonical_upper_a_index is not None,
                lower_distance=self.lower_distance,
                upper_distance=self.upper_distance,
            )
            return _NaturalPointObjectiveFunction(
                spec, _PointObservations(stats), self.z_data_bounds
            )
        distinct = self._distinct
        if distinct is None:
            observations = _build_interval_observations(
                self.rows,
                _normalized_weights(self.rows.shape[0], weights, "interval"),
                coordinate=self.coordinate,
                grouping_cache=self._grouping_cache,
            )
        else:
            # The interval likelihood is additive over rows, so identical
            # censoring intervals need only one quadrature contribution.
            # Preserve the original-row Kish effective sample size: sample
            # weights are reliability weights, not frequency counts.
            normalized, total, effective_n = _normalized_weights(
                self.rows.shape[0], weights, "interval"
            )
            first, inverse, n_unique = self._grouping_cache["grouping"]
            grouped_weights = np.bincount(
                inverse, weights=normalized, minlength=n_unique
            ).astype(np.float64)
            template = self._compact_observation_template
            if template is not None:
                observations = replace(
                    template,
                    weights=np.ascontiguousarray(grouped_weights, dtype=np.float64),
                    total_weight=float(total),
                    effective_n=float(effective_n),
                    n_observations=int(self.rows.shape[0]),
                )
            else:
                observations = _build_interval_observations(
                    self.rows[first],
                    _WeightSummary(grouped_weights, total, effective_n),
                    coordinate=self.coordinate,
                )
                observations = replace(
                    observations,
                    n_observations=int(self.rows.shape[0]),
                )
        return _NaturalIntervalObjectiveFunction(
            spec, observations, self.z_data_bounds, nonparametric_bound=False
        )

    def _grouped_observation_moments(self, observation_weights, /):
        """Return per-distinct-row first and second observation-weight sums.

        These summaries are invariant across EM iterations.  They let an
        interval M-step preserve the original-row Kish effective sample size
        while doing its likelihood algebra on distinct censoring rows only.

        Parameters
        ----------
        observation_weights : numpy.ndarray, shape (R,)
            Original-row normalized reliability weights.

        Returns
        -------
        tuple of numpy.ndarray or None
            Distinct-row sums of ``w`` and ``w**2``; ``None`` when rows are
            not represented by a duplicate grouping.
        """
        distinct = self._distinct
        if distinct is None:
            return None
        cached = self._observation_group_cache
        if cached is not None and cached[0] is observation_weights:
            return cached[1], cached[2]
        inverse = distinct[1]
        n_unique = distinct[0].shape[0]
        w = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
        group_sum = np.bincount(inverse, weights=w, minlength=n_unique).astype(
            np.float64
        )
        group_sq = np.bincount(inverse, weights=w * w, minlength=n_unique).astype(
            np.float64
        )
        self._observation_group_cache = (observation_weights, group_sum, group_sq)
        return group_sum, group_sq

    def compact_objective(self, responsibilities, observation_weights, /):
        """Return an interval M-step objective from distinct-row posteriors.

        ``responsibilities`` has one value per distinct interval.  The
        resulting objective is algebraically the same as fitting the expanded
        original rows with ``observation_weights * responsibilities[inverse]``.
        Reliability-weight effective sample size is computed on the original
        rows, not on the compressed frequency representation.

        Parameters
        ----------
        responsibilities : numpy.ndarray, shape (R_unique,)
            Posterior component probabilities for distinct interval rows.
        observation_weights : numpy.ndarray, shape (R,)
            Original-row normalized reliability weights.

        Returns
        -------
        _NaturalIntervalObjectiveFunction
            Compressed M-step objective with original-row effective sample size.
        """
        if not self.intervals or self._distinct is None:
            raise ValueError("compact_objective requires duplicated interval rows")
        first, _inverse, n_unique = self._grouping_cache["grouping"]
        r = np.asarray(responsibilities, dtype=np.float64).reshape(-1)
        if r.size != n_unique or np.any(r < 0.0) or not np.all(np.isfinite(r)):
            raise ValueError("distinct responsibilities must be finite and nonnegative")
        group_sum, group_sq = self._grouped_observation_moments(observation_weights)
        raw = group_sum * r
        total = float(np.sum(raw, dtype=np.float64))
        if not (total > 0.0 and np.isfinite(total)):
            raise ValueError("component responsibility mass must be positive")
        square_sum = float(np.dot(group_sq, r * r))
        if not (square_sum > 0.0 and np.isfinite(square_sum)):
            raise ValueError("component squared responsibility mass must be positive")
        grouped_weights = np.ascontiguousarray(raw / total, dtype=np.float64)
        template = self._compact_observation_template
        if template is not None:
            observations = replace(
                template,
                weights=grouped_weights,
                total_weight=total,
                effective_n=float(total * total / square_sum),
                n_observations=int(self.rows.shape[0]),
            )
        else:
            observations = _build_interval_observations(
                self.rows[first],
                _WeightSummary(
                    grouped_weights, total, float(total * total / square_sum)
                ),
                coordinate=self.coordinate,
            )
            observations = replace(
                observations,
                n_observations=int(self.rows.shape[0]),
            )
        return _NaturalIntervalObjectiveFunction(
            self.spec, observations, self.z_data_bounds, nonparametric_bound=False
        )

    def fit_compact(
        self, responsibilities, observation_weights, previous, solver_options, /
    ):
        """Run one duplicated-interval M-step without expanding row weights.

        Parameters
        ----------
        responsibilities : numpy.ndarray, shape (R_unique,)
            Posterior component probabilities for distinct interval rows.
        observation_weights : numpy.ndarray, shape (R,)
            Original-row normalized reliability weights.
        previous : _ConicNewtonResult or None
            Previous component result for a warm start.
        solver_options : _NewtonOptions
            Shared immutable Newton policy.

        Returns
        -------
        objective : _NaturalIntervalObjectiveFunction
            Compressed M-step objective.
        result : _ConicNewtonResult
            Optimized component result.
        """
        objective = self.compact_objective(responsibilities, observation_weights)
        if previous is not None:
            initial, initial_blocks = previous.params, previous.blocks
        else:
            initial, initial_blocks = _natural_interval_start(objective)
        return objective, _solve_natural_conic(
            objective,
            initial=initial,
            initial_blocks=initial_blocks,
            solver_options=solver_options,
            certify=False,
        )

    def fit(self, weights, previous, solver_options, /):
        """Run one M-step for this component.

        Parameters
        ----------
        weights : numpy.ndarray, shape (R,)
            M-step weights.
        previous : _ConicNewtonResult or None
            Previous result for the warm start, or ``None`` for a cold start.
        solver_options : _NewtonOptions
            Shared immutable Newton policy.

        Returns
        -------
        objective, result
            The M-step objective and solver result.
        """
        objective = self.objective(weights)
        initial = initial_blocks = None
        if previous is not None:
            initial, initial_blocks = previous.params, previous.blocks
        elif self.intervals:
            initial, initial_blocks = _natural_interval_start(objective)
        return objective, _solve_natural_conic(
            objective,
            initial=initial,
            initial_blocks=initial_blocks,
            solver_options=solver_options,
            certify=False,
        )


def _logsumexp(values, /):
    """Return ``log(sum(exp(values)))`` of a short vector, stably.

    Parameters
    ----------
    values : numpy.ndarray, shape (K,)
        Log values.
    """
    top = float(np.max(values))
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.sum(np.exp(values - top))))


def _log_mixture_weights(mass, /):
    """Return normalized log mixture weights from component masses.

    Parameters
    ----------
    mass : numpy.ndarray, shape (K,)
        Nonnegative component masses.
    """
    log_weights = np.log(np.maximum(mass, np.finfo(float).tiny))
    return log_weights - _logsumexp(log_weights)


def _e_step(
    problems, layouts, params, log_weights, observation_weights, /, *, compact=False
):
    """Return ``(log_likelihood, responsibilities)`` at a mixture point.

    Parameters
    ----------
    problems : sequence of _ComponentProblem
        Component problems.
    layouts : sequence of _NaturalLayout
        Component layouts.
    params : sequence of numpy.ndarray
        Component parameters.
    log_weights : numpy.ndarray, shape (K,)
        Log mixture weights.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    compact : bool, optional
        For duplicated interval rows, return one posterior row per distinct
        interval instead of expanding back to the original observations.
    """
    # Duplicate interval rows share one compiled row; ``compact`` keeps the
    # posterior on those distinct rows.
    return _compiled_mixture(problems, layouts, observation_weights).posterior(
        params, log_weights, compact=compact
    )


@dataclass(frozen=True)
class _RawMixtureEvaluation:
    """Raw mixture likelihood geometry, never an optimizer metric."""

    nll: float
    gradient: np.ndarray
    fisher: np.ndarray
    missing_information: np.ndarray

    @property
    def observed_hessian(self):
        """Observed-information Hessian after subtracting missing information."""
        return self.fisher - self.missing_information


def _mixture_map(problems, layouts=None, /):
    """Return the natural map of a component problem set, built once.

    Parameters
    ----------
    problems : sequence of _ComponentProblem
        Component problems.
    layouts : sequence of _NaturalLayout or None, optional
        Component layouts; the problems' own layouts by default.
    """
    problems = tuple(problems)
    layouts = (
        tuple(problem.spec.layout for problem in problems)
        if layouts is None
        else tuple(layouts)
    )
    cached = problems[0]._mixture_map
    if (
        cached is not None
        and len(cached[0]) == len(problems)
        and all(a is b for a, b in zip(cached[0], problems, strict=True))
        and all(a is b for a, b in zip(cached[1], layouts, strict=True))
    ):
        return cached[2]
    mapping = _MixtureNaturalMap([problem.spec for problem in problems], layouts)
    problems[0]._mixture_map = (problems, layouts, mapping)
    return mapping


def _compiled_mixture(problems, layouts, observation_weights, /):
    """Return the compiled joint mixture of a problem set, built once.

    Parameters
    ----------
    problems : sequence of _ComponentProblem
        Component problems.
    layouts : sequence of _NaturalLayout
        Component layouts.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    """
    problems = tuple(problems)
    layouts = tuple(layouts)
    weights = np.asarray(observation_weights, dtype=np.float64)
    cached = problems[0]._compiled_mixture
    if (
        cached is not None
        and len(cached[0]) == len(problems)
        and all(a is b for a, b in zip(cached[0], problems, strict=True))
        and all(a is b for a, b in zip(cached[1], layouts, strict=True))
        and np.array_equal(cached[2], weights)
    ):
        return cached[3]
    objective = _CompiledJointMixture(problems, layouts, weights)
    problems[0]._compiled_mixture = (problems, layouts, weights.copy(), objective)
    return objective


def _compiled_component(objective, /):
    """Return ``(kind, inputs)`` of an M-step objective for the compiled kernels.

    Parameters
    ----------
    objective : _NaturalPointObjectiveFunction or _NaturalIntervalObjectiveFunction
        Responsibility-weighted component objective.
    """
    if isinstance(objective, _NaturalIntervalObjectiveFunction):
        return 1, objective._compiled_interval_newton_inputs()
    objective._point_empirical_means()
    return 0, objective._compiled_point_newton_inputs()


def _component_evaluation(nll, gradient, fisher, missing, means, /):
    """Wrap one compiled component evaluation as an objective evaluation.

    Parameters
    ----------
    nll : float
        Negative log likelihood at the evaluation point.
    gradient : numpy.ndarray
        Objective gradient.
    fisher : numpy.ndarray
        Complete-data/Fisher information matrix.
    missing : numpy.ndarray
        Missing-information contribution.
    means : numpy.ndarray
        Model partial means retained by the natural objective.
    """
    return _ObjectiveEvaluation(
        nll=float(nll),
        gradient=gradient,
        hessian=fisher - missing,
        fisher=fisher,
        missing_information=missing,
        model_partial_means=means,
    )


def _joint_component(problem, layout, /):
    """Return the compiled joint-objective inputs of one component.

    Parameters
    ----------
    problem : _ComponentProblem
        Component problem.
    layout : _NaturalLayout
        Its natural layout.

    Returns
    -------
    tuple
        ``(kind, inputs, rows)`` for ``JointMixtureObjective``: point data
        carry their natural basis, interval data the joint row of every row
        of each reducer partition.  Distinct interval rows are the joint rows
        when duplicates are grouped.
    """
    if not problem.intervals:
        inputs = _point_kernel_inputs(
            layout, problem.z_data_bounds, np.zeros(layout.n_params), problem.log_scale
        )
        basis = problem.point_basis(layout, tuple(map(float, layout.support)))
        return 0, inputs, np.ascontiguousarray(basis.T, dtype=np.float64)
    rows = problem.rows
    if problem.distinct_rows is not None:
        rows = rows[problem._grouping_cache["grouping"][0]]
    observations = _build_interval_observations(
        rows, coordinate=problem.coordinate, deduplicate=False
    )
    regular, adaptive, whole = _interval_partitions(
        observations.intervals, layout.support
    )
    inputs = _interval_kernel_inputs(
        layout,
        problem.z_data_bounds,
        observations,
        np.ones(observations.intervals.shape[0]),
        regular,
        adaptive,
        0.0,
        problem.coordinate.scale,
    )
    return 1, inputs, tuple(map(np.flatnonzero, (regular, adaptive, whole)))


class _CompiledJointMixture:
    """Observed-data mixture NLL and raw information from fused compiled code.

    The variable contains private polynomial shapes, shared physical
    amplitudes, and ``K-1`` logits with mixture weights ``softmax(eta, 0)``.
    Evaluations expose raw information; exact-face solves build their Newton
    metric in compiled code after restricting it to the free coordinates.
    """

    def __init__(self, problems, layouts, observation_weights, /):
        """Bind the component problems and observation weights.

        Parameters
        ----------
        problems : sequence of _ComponentProblem
            Component problems.
        layouts : sequence of _NaturalLayout
            Component layouts.
        observation_weights : numpy.ndarray, shape (R,)
            Normalized observation weights.
        """
        self.problems = tuple(problems)
        self.layouts = tuple(layouts)
        self.n_components = len(self.problems)
        self.natural_map = _mixture_map(self.problems, self.layouts)
        self.n_params = self.natural_map.n_params + self.n_components - 1
        # Identical interval rows contribute identically to every sum, so the
        # objective runs on distinct rows with their weights added.
        first = self.problems[0]
        distinct = first.distinct_rows if first.intervals else None
        self._inverse = None if distinct is None else distinct[1]
        row_weights = (
            observation_weights
            if distinct is None
            else np.bincount(
                distinct[1], weights=observation_weights, minlength=distinct[0].shape[0]
            )
        )
        self._kernel = JointMixtureObjective(
            [
                _joint_component(problem, layout)
                for problem, layout in zip(self.problems, self.layouts, strict=True)
            ],
            row_weights,
        )
        self._columns = tuple(
            np.asarray(indices, dtype=np.intc)
            for indices in self.natural_map.local_indices
        )
        self._logit_columns = np.arange(
            self.natural_map.n_params, self.n_params, dtype=np.intc
        )
        # The posterior reads each component's own parameter block, so it
        # needs no shared-amplitude consistency between the blocks.
        offsets = np.cumsum([0, *(layout.n_params for layout in self.layouts)])
        self._block_columns = tuple(
            np.arange(start, stop, dtype=np.intc) for start, stop in pairwise(offsets)
        )
        self._block_size = int(offsets[-1]) + self.n_components - 1
        self._block_logits = np.arange(offsets[-1], self._block_size, dtype=np.intc)

    def split(self, x, /):
        """Return ``(params, log_weights)`` from the joint variable.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,)
            Joint variable.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_params,):
            raise ValueError("invalid reduced joint mixture vector")
        params = self.natural_map.expand(x[: self.natural_map.n_params])
        eta = np.concatenate((x[self.natural_map.n_params :], [0.0]))
        return params, eta - _logsumexp(eta)

    def join(self, params, log_weights, /):
        """Return the joint variable for component params and log weights.

        Parameters
        ----------
        params : sequence of numpy.ndarray
            Component parameters.
        log_weights : numpy.ndarray, shape (K,)
            Log mixture weights.
        """
        eta = np.asarray(log_weights[:-1] - log_weights[-1], dtype=np.float64)
        return np.concatenate((self.natural_map.pack(params), eta))

    def __call__(self, x, /):
        """Return observed NLL, gradient and raw complete/missing information.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,)
            Joint variable.
        """
        return self.evaluate(x)

    def evaluate(self, x, /, rows=False):
        """Evaluate the raw geometry, optionally with per-row posterior terms.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,)
            Joint variable.
        rows : bool, optional
            Also return the responsibilities ``(R, K)`` and every component's
            centered conditional partial means ``(R, n_k)`` on the original
            observation rows.

        Raises
        ------
        FloatingPointError
            If ``x`` is not a valid evaluation point.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.n_params,):
            raise ValueError("invalid reduced joint mixture vector")
        output = self._kernel.evaluate(
            self._columns, self._logit_columns, self.n_params, x, rows
        )
        evaluation = _RawMixtureEvaluation(*output[:4])
        if not rows:
            return evaluation
        responsibilities, centered = output[4:]
        if self._inverse is not None:
            responsibilities = responsibilities[self._inverse]
            centered = tuple(values[self._inverse] for values in centered)
        return evaluation, responsibilities, centered

    def posterior(self, params, log_weights, /, compact=False):
        """Return ``(log_likelihood, responsibilities)`` at a mixture point.

        Parameters
        ----------
        params : sequence of numpy.ndarray
            Component parameters.
        log_weights : numpy.ndarray, shape (K,)
            Normalized log mixture weights.
        compact : bool, optional
            For grouped interval rows, return one posterior row per distinct
            interval instead of expanding back to the original observations.

        Raises
        ------
        FloatingPointError
            If a component is not normalizable or an observation has zero
            likelihood.
        """
        log_weights = np.asarray(log_weights, dtype=np.float64)
        x = np.concatenate(
            [
                *(
                    layout.validate_params(theta)
                    for layout, theta in zip(self.layouts, params, strict=True)
                ),
                log_weights[:-1] - log_weights[-1],
            ]
        )
        log_likelihood, responsibilities = self._kernel.posterior(
            self._block_columns, self._block_logits, self._block_size, x
        )
        if self._inverse is not None and not compact:
            responsibilities = responsibilities[self._inverse]
        return log_likelihood, responsibilities

    def solver(self, face, /):
        """Return the compiled exact-face solve of ``face`` (with its logits).

        Parameters
        ----------
        face : _MixtureFace
            Exact component/logit face whose column maps are bound.
        """
        return functools.partial(
            self._kernel.solve,
            face.component_columns,
            face.logit_columns,
            len(face.free_indices),
        )


def _result_face(natural_map, results, n_logits=0):
    """Return the shared exact face represented by component results.

    Parameters
    ----------
    natural_map : _MixtureNaturalMap
        Reduced shared-parameter map for the components.
    results : sequence of _ConicNewtonResult
        Component results whose effective degrees and physical boundary masks
        define the face.
    n_logits : int, optional
        Number of free mixture-logit coordinates appended to the face.

    Returns
    -------
    _MixtureFace
        Shared exact face in reduced mixture coordinates.
    """
    natural_map.pack([result.params for result in results])
    masks = [
        (
            (result.lower_amplitude_active, result.upper_amplitude_active)
            if spec.coordinate.direction > 0.0
            else (result.upper_amplitude_active, result.lower_amplitude_active)
        )
        for spec, result in zip(natural_map.specs, results, strict=True)
    ]
    if any(mask != masks[0] for mask in masks[1:]):
        raise ValueError("component exact faces do not share physical boundary masks")
    active = masks[0]
    return natural_map.face(
        [result.effective_curvature_degree for result in results], active, n_logits
    )


def _solve_mixture_face(solve, face, start, blocks, options):
    """Optimize an exact face with a fused compiled mixture objective.

    Parameters
    ----------
    solve : callable
        Compiled exact-face solve with the face's column maps bound
        (``SharedMStepObjective.solve`` or ``_CompiledJointMixture.solver``).
    face : _MixtureFace
        Exact face.
    start : numpy.ndarray
        Full reduced starting vector.
    blocks : sequence of numpy.ndarray or None
        Gram certificate of ``start``; reused only when it matches the face.
    options : _NewtonOptions
        Newton settings.

    Returns
    -------
    tuple or None
        ``(run, extra)`` with the ``_NewtonRun`` in free-face coordinates and
        any trailing outputs of ``solve``; ``None`` when the start is not a
        valid evaluation point or the face solve fails numerically.
    """
    params = face.restrict(start)
    representation = face.representation
    shapes = [matrix.shape[1:] for matrix in representation.row_matrices]
    matching = (
        blocks is not None
        and [np.shape(block) for block in blocks] == shapes
        and representation.residual(params, blocks)
        <= 1e-10 * max(1.0, np.max(np.abs(representation.b_matrix @ params)))
        and all(np.linalg.eigvalsh(block)[0] >= -1e-12 for block in blocks)
    )
    try:
        if not matching:
            blocks = _default_blocks(representation)
            metric = np.eye(params.size)
            params, blocks = _project(params, representation, blocks, metric)
        a_packed, sizes, a_offsets, q_offsets = representation.packed
        output = solve(
            params,
            representation.pack_blocks(blocks),
            representation.b_matrix,
            a_packed,
            sizes,
            a_offsets,
            q_offsets,
            representation.reference_dual,
            representation.row_degrees,
            options.tolerance,
            options.certified_tolerance,
            options.accuracy_floor,
            options.max_iterations,
            options.armijo,
            options.backtrack,
            options.max_line_search,
        )
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "shared mixture face solve", routine=True)
        return None
    (
        status,
        theta,
        packed_blocks,
        dual,
        nll,
        gradient,
        metric,
        fisher,
        missing,
        smallest,
        iterations,
        evaluations,
        subproblem_iterations,
        bound,
    ) = output[:14]
    run = _NewtonRun(
        status=status,
        params=theta,
        blocks=representation.unpack_blocks(packed_blocks),
        dual=dual,
        evaluation=_NaturalIntervalEvaluation(
            nll=nll,
            gradient=gradient,
            hessian=metric,
            observed_hessian=fisher - missing,
            fisher=fisher,
            missing_information=missing,
            smallest_curvature=smallest,
        ),
        iterations=iterations,
        evaluations=evaluations,
        subproblem_iterations=subproblem_iterations,
        decrease_bound=bound,
    )
    return run, output[14:]


def _coupled_component_results(run, face, evaluations):
    """Store matching component feasibility certificates, not local optimality.

    Parameters
    ----------
    run : _NewtonRun
        Face solve in free-face coordinates.
    face : _MixtureFace
        Its exact face.
    evaluations : sequence of tuple
        Compiled per-component ``(nll, gradient, fisher, missing, means)`` at
        the run's parameters.
    """
    params = face.natural_map.expand(
        face.expand(run.params)[: face.natural_map.n_params]
    )
    results = []
    for k, (theta, raw, layout) in enumerate(
        zip(params, evaluations, face.natural_map.layouts, strict=True)
    ):
        evaluation = _component_evaluation(*raw)
        lower_active, upper_active = (
            face.active
            if face.natural_map.specs[k].coordinate.direction > 0.0
            else face.active[::-1]
        )
        results.append(
            _ConicNewtonResult(
                status="joint_feasible",
                params=theta,
                objective_value=float(evaluation.nll),
                evaluation=evaluation,
                blocks=tuple(run.blocks[face.block_slices[k]]),
                dual=face.component_dual(run.dual, k),
                effective_curvature_degree=face.degrees[k],
                lower_amplitude_active=lower_active,
                upper_amplitude_active=upper_active,
                newton_iterations=0,
                objective_evaluations=0,
                subproblem_iterations=0,
                final_decrease_bound=np.nan,
                final_separation=_certify(layout, theta),
            )
        )
    return tuple(results)


def _coupled_m_step(problems, objectives, masses, previous, solver_options, /):
    """Compare globally shared amplitude masks by coupled optimization.

    Parameters
    ----------
    problems : sequence of _ComponentProblem
        Component geometry and observation contracts.
    objectives : sequence
        Current component M-step objectives.
    masses : numpy.ndarray, shape (K,)
        Responsibility mass of each component.
    previous : sequence of _ConicNewtonResult or None
        Previous component results used to warm-start the shared solve.
    solver_options : _NewtonOptions
        Shared immutable Newton policy.

    Returns
    -------
    tuple of _ConicNewtonResult
        Coupled component results on the selected shared boundary face.
    """
    mapping = _mixture_map(problems)
    objective = SharedMStepObjective(
        [
            (kind, float(mass), inputs)
            for (kind, inputs), mass in zip(
                map(_compiled_component, objectives), masses, strict=True
            )
        ]
    )
    controls = solver_options

    def solve_face(face, candidate_start, candidate_blocks):
        solve = functools.partial(
            objective.solve, face.component_columns, len(face.free_indices)
        )
        return _solve_mixture_face(
            solve, face, candidate_start, candidate_blocks, controls
        )

    degrees = tuple(layout.curvature_degree for layout in mapping.layouts)
    enabled = tuple(index is not None for index in mapping.shared_parameter_indices)
    blocks = None
    if previous is None:
        # These feasible starts use the same amplitude (0.1) by construction.
        start = mapping.pack([_interior_start(item) for item in objectives])
        zero_first = any(
            getattr(item.observations, "has_infinite_rows", False)
            for item in objectives
        )
        initial_active = (False, False) if zero_first else enabled
    else:
        start = mapping.pack([result.params for result in previous])
        initial_active = tuple(
            index is not None and start[index] > 0.0
            for index in mapping.shared_parameter_indices
        )
        blocks = tuple(block for result in previous for block in result.blocks)
    masks = list(product(*[(False, True) if flag else (False,) for flag in enabled]))
    masks.remove(initial_active)
    masks.insert(0, initial_active)
    best, best_face, best_components = None, None, None
    resolved = True
    for active in masks:
        face = mapping.face(degrees, active)
        candidate_start = (
            start.copy() if best is None else best_face.expand(best.params)
        )
        for index, on in zip(mapping.shared_parameter_indices, active, strict=True):
            if index is not None:
                candidate_start[index] = (
                    max(candidate_start[index], 1e-3) if on else 0.0
                )
        solved = solve_face(
            face, candidate_start, blocks if active == initial_active else None
        )
        if solved is None:
            resolved = False
            continue
        run, (components,) = solved
        resolved &= run.status in _CONVERGED
        if best is None:
            best, best_face, best_components = run, face, components
            continue
        scale = max(1.0, abs(best.evaluation.nll))
        difference = run.evaluation.nll - best.evaluation.nll
        if difference < -controls.tolerance * scale or (
            run.status in _CONVERGED
            and difference <= controls.tolerance * scale
            and sum(active) < sum(best_face.active)
        ):
            best, best_face, best_components = run, face, components
    if best is None:
        raise RuntimeError("no shared mixture face has a normalizable solution")
    # Re-expand to the requested degree at every M-step, then retain a smaller
    # private face only after optimizing it jointly with all free shared sides.
    for k, layout in enumerate(mapping.layouts):
        step = {"real_line": 2, "bounded": 0}.get(layout.support_kind, 1)
        if not step:
            continue
        while best_face.degrees[k] >= step:
            params = mapping.expand(best_face.expand(best.params))[k]
            curvature = params[layout.curvature_slice]
            if abs(curvature[best_face.degrees[k]]) > controls.face_trigger * max(
                1.0, np.max(np.abs(curvature))
            ):
                break
            trial_degrees = list(best_face.degrees)
            trial_degrees[k] -= step
            face = mapping.face(trial_degrees, best_face.active)
            solved = solve_face(face, best_face.expand(best.params), None)
            if solved is None:
                break
            run, (components,) = solved
            if (
                run.status not in _CONVERGED
                or run.evaluation.nll
                > best.evaluation.nll
                + controls.tolerance * max(1.0, abs(best.evaluation.nll))
            ):
                break
            best, best_face, best_components = run, face, components
    results = _coupled_component_results(best, best_face, best_components)
    if not resolved:
        results = tuple(
            replace(result, status="joint_face_unresolved") for result in results
        )
    return results


def _evaluated_em_state(
    problems, layouts, results, log_weights, observation_weights, /
):
    """Evaluate a mixture point once and retain its posterior for the next EM map.

    Parameters
    ----------
    problems : sequence of _ComponentProblem
        Component problems.
    layouts : sequence of _NaturalLayout
        Component layouts.
    results : sequence of _ConicNewtonResult
        Current component results (their params define the point).
    log_weights : numpy.ndarray, shape (K,)
        Current log mixture weights.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.

    Returns
    -------
    _EMState
        Evaluated iterate including the posterior probabilities.
    """
    compact = bool(
        problems and problems[0].intervals and problems[0].distinct_rows is not None
    )
    log_likelihood, responsibilities = _e_step(
        problems,
        layouts,
        [r.params for r in results],
        log_weights,
        observation_weights,
        compact=compact,
    )
    return _EMState(
        tuple(results),
        np.asarray(log_weights, dtype=np.float64),
        float(log_likelihood),
        responsibilities,
    )


def _em_map(state, problems, observation_weights, solver_options, /):
    """Apply one EM map using the posterior cached on ``state``.

    Parameters
    ----------
    state : _EMState
        Evaluated input iterate.
    problems : sequence of _ComponentProblem
        Component problems.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    solver_options : _NewtonOptions
        Shared immutable Newton policy.

    Returns
    -------
    results : tuple of _ConicNewtonResult
        Updated component results.
    log_weights : numpy.ndarray
        Updated log mixture weights.  The caller evaluates this output once
        to obtain the reusable posterior for the following map.
    """
    responsibilities = state.responsibilities
    compact = bool(
        problems
        and problems[0].intervals
        and problems[0].distinct_rows is not None
        and responsibilities.shape[0] == problems[0].distinct_rows[0].shape[0]
    )
    if problems and not problems[0].intervals:
        rows = problems[0].rows
        _check_point_estimability(rows, observation_weights, responsibilities)
    updated = []
    if len(problems) > 1 and any(
        index is not None
        for index in (
            problems[0].spec.physical_lower_a_index,
            problems[0].spec.physical_upper_a_index,
        )
    ):
        if compact:
            grouped, _ = problems[0]._grouped_observation_moments(observation_weights)
            mass = grouped @ responsibilities
            objectives = [
                problem.compact_objective(responsibilities[:, k], observation_weights)
                for k, problem in enumerate(problems)
            ]
        else:
            mass = observation_weights @ responsibilities
            objectives = [
                problem.objective(observation_weights * responsibilities[:, k])
                for k, problem in enumerate(problems)
            ]
        return _coupled_m_step(
            problems, objectives, mass, state.results, solver_options
        ), _log_mixture_weights(mass)
    if compact:
        grouped_observation_weights, _ = problems[0]._grouped_observation_moments(
            observation_weights
        )
        for k, problem in enumerate(problems):
            _, result = problem.fit_compact(
                responsibilities[:, k],
                observation_weights,
                state.results[k],
                solver_options,
            )
            updated.append(result)
        mass = grouped_observation_weights @ responsibilities
    else:
        for k, problem in enumerate(problems):
            _, result = problem.fit(
                observation_weights * responsibilities[:, k],
                state.results[k],
                solver_options,
            )
            updated.append(result)
        mass = observation_weights @ responsibilities
    return tuple(updated), _log_mixture_weights(mass)


def _stack(results, log_weights, natural_map, /):
    """Stack component parameters and log weights into one vector.

    Parameters
    ----------
    results : sequence of _ConicNewtonResult
        Component results.
    log_weights : numpy.ndarray
        Log mixture weights.
    natural_map : _MixtureNaturalMap
        Reduced map used to pack shared component parameters.
    """
    return np.concatenate((natural_map.pack([r.params for r in results]), log_weights))


def _unstack(vector, reference, layouts, natural_map, /):
    """Return admissible ``(results, log_weights)`` from a stacked vector, or ``None``.

    Component parameters are admissible when every enabled amplitude is
    nonnegative and the exact separator certifies the full curvature.

    Parameters
    ----------
    vector : numpy.ndarray
        Stacked component parameters and log weights.
    reference : sequence of _ConicNewtonResult
        Results supplying Gram blocks (warm starts only) and shapes.
    layouts : sequence of _NaturalLayout
        Component layouts.
    natural_map : _MixtureNaturalMap
        Reduced map used to unpack the stacked component parameters.

    Returns
    -------
    tuple or None
        Admissible ``(results, log_weights)`` pair, or ``None`` when the
        stacked point violates face feasibility.
    """
    if not np.all(np.isfinite(vector)):
        return None
    shapes = vector[: natural_map.n_params]
    params_all = natural_map.expand(shapes)
    for layout, params in zip(layouts, params_all, strict=True):
        for index in (layout.lower_a_index, layout.upper_a_index):
            if index is not None and params[index] < 0.0:
                return None
        if not _certify(layout, params).feasible:
            return None
    face = _result_face(natural_map, reference)
    representation = face.representation
    try:
        free, blocks = _project(
            face.restrict(shapes),
            representation,
            _default_blocks(representation),
            np.eye(len(face.free_indices)),
        )
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "reduced SQUAREM certificate", routine=True)
        return None
    params_all = natural_map.expand(face.expand(free))
    results = []
    for k, (result, params) in enumerate(zip(reference, params_all, strict=True)):
        results.append(
            replace(
                result,
                params=params,
                blocks=tuple(blocks[face.block_slices[k]]),
                dual=np.zeros(face.component_representations[k].n_rows),
                evaluation=None,
                objective_value=np.nan,
                status="extrapolated",
                final_decrease_bound=np.inf,
                final_separation=_certify(layouts[k], params),
            )
        )
    log_weights = np.asarray(vector[natural_map.n_params :], dtype=np.float64)
    return results, log_weights - _logsumexp(log_weights)


def _em_phase(
    state,
    problems,
    layouts,
    observation_weights,
    /,
    *,
    max_steps,
    tolerance,
    accelerate,
    history,
    solver_options,
):
    """Run SQUAREM-accelerated EM until the likelihood stalls.

    Parameters
    ----------
    state : _EMState
        Evaluated starting point.
    problems, layouts : sequences
        Component problems and layouts.
    observation_weights : numpy.ndarray
        Normalized observation weights.
    max_steps : int
        EM map evaluation limit for this phase.
    tolerance : float
        Stop when a cycle raises the log likelihood by at most
        ``tolerance * (1 + |log likelihood|)``.
    accelerate : bool
        Use SQUAREM.
    history : list
        Appended with the log likelihood after every accepted cycle.
    solver_options : _NewtonOptions
        Shared immutable Newton policy.

    Returns
    -------
    state : _EMState
        Evaluated point at the end of the phase.
    steps : int
        EM map evaluations used.
    """

    def em(point_state):
        point_results, point_log_weights = _em_map(
            point_state,
            problems,
            observation_weights,
            solver_options,
        )
        return _evaluated_em_state(
            problems, layouts, point_results, point_log_weights, observation_weights
        )

    natural_map = _mixture_map(problems, layouts)
    steps = 0
    while steps < int(max_steps):
        state1 = em(state)
        steps += 1
        if not accelerate:
            new = state1
        else:
            state2 = em(state1)
            steps += 1
            new = state2
            # SQUAREM (scheme S3).  alpha = -1 reproduces the second EM step;
            # steeper extrapolations are tried first, halved toward it, and
            # one is kept only if its stabilizing EM step beats two plain EM
            # steps, so the iteration stays monotone.
            x0 = _stack(state.results, state.log_weights, natural_map)
            step = _stack(state1.results, state1.log_weights, natural_map) - x0
            curvature = (
                _stack(state2.results, state2.log_weights, natural_map)
                - 2.0 * (x0 + step)
                + x0
            )
            norm = float(np.linalg.norm(curvature))
            alpha = -float(np.linalg.norm(step)) / norm if norm > 0.0 else -1.0
            for _ in range(4):
                if alpha > -1.5:
                    break
                point = _unstack(
                    x0 - 2.0 * alpha * step + alpha * alpha * curvature,
                    state2.results,
                    layouts,
                    natural_map,
                )
                if point is not None:
                    try:
                        point_state = _evaluated_em_state(
                            problems, layouts, point[0], point[1], observation_weights
                        )
                        trial = em(point_state)
                        steps += 1
                    except _PointMixtureEstimabilityFailure:
                        # Extrapolation may create a singular point-component;
                        # decline this acceleration step without ledger noise.
                        trial = None
                    except NUMERIC_FAILURES as exc:
                        _reraise_if_debug(
                            exc, "SQUAREM stabilization step", routine=True
                        )
                        trial = None
                    if trial is not None and trial.log_likelihood >= new.log_likelihood:
                        new = trial
                        break
                alpha = 0.5 * (alpha - 1.0)
        increase = new.log_likelihood - state.log_likelihood
        if increase < 0.0:
            break
        state = new
        history.append(state.log_likelihood)
        if increase <= tolerance * (1.0 + abs(state.log_likelihood)):
            break
    return state, steps


def _polish(state, problems, layouts, observation_weights, options, /):
    """Joint Newton polish of the observed likelihood on the current faces.

    Parameters
    ----------
    state : _EMState
        Evaluated mixture iterate.
    problems, layouts : sequences
        Component problems and layouts.
    observation_weights : numpy.ndarray
        Normalized observation weights.
    options : _NewtonOptions
        Newton settings.

    Returns
    -------
    _NewtonRun or None
        The polish run, or ``None`` when its start is not normalizable.
    objective : _CompiledJointMixture
        The joint objective (to split the run's parameters).
    """
    results, log_weights = state.results, state.log_weights
    objective = _compiled_mixture(problems, layouts, observation_weights)
    face = _result_face(objective.natural_map, results, len(results) - 1)
    objective.face = face
    x0 = objective.join([r.params for r in results], log_weights)
    blocks = tuple(block for r in results for block in r.blocks)
    solved = _solve_mixture_face(objective.solver(face), face, x0, blocks, options)
    if solved is None:
        return None, objective
    run = solved[0]
    objective.face_run = run
    return replace(run, params=face.expand(run.params)), objective


class _PointMixtureEstimabilityFailure(RuntimeError):
    """Expected rejection of a singular point-mixture candidate."""


def _check_point_estimability(rows, observation_weights, responsibilities, /):
    """Reject the singular one-location boundary of point-mixture likelihood.

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical observation rows.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Current component responsibilities.

    Returns
    -------
    None

    Raises
    ------
    _PointMixtureEstimabilityFailure
        If a point-data component has fewer than the required effective distinct
        sample locations.
    """
    if rows.shape[1] != 1 or responsibilities.shape[1] <= 1:
        return
    x = rows[:, 0]
    inverse = _distinct_location_index(rows)
    for k in range(responsibilities.shape[1]):
        n_eff = _effective_distinct_point_count(
            x, observation_weights * responsibilities[:, k], inverse=inverse
        )
        if n_eff < EM_MIN_EFFECTIVE_DISTINCT_N:
            raise _PointMixtureEstimabilityFailure(
                "point-mixture component is not estimable: component "
                f"{k + 1}/{responsibilities.shape[1]} has only {n_eff:.6g} "
                "effective distinct sample locations. The unconstrained "
                "point-mixture likelihood is singular when a component "
                "concentrates onto one observed location."
            )


def _needs_wider_search(fit, rows, observation_weights, /):
    """Whether a multi-start seeded only by a KDE valley should try other seeds.

    Parameters
    ----------
    fit : _NaturalMixtureFit or None
        Best fit from the valley seed, or ``None`` when every start failed.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Observation rows.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.

    Returns
    -------
    bool
        True when every start failed, when the best fit is not certified, or
        when a point-data component rests on no more effective distinct
        locations than it has natural parameters (too few observations to
        determine it).
    """
    if fit is None or fit.status not in _CONVERGED:
        return True
    if rows.shape[1] != 1:
        return False
    x = rows[:, 0]
    inverse = _distinct_location_index(rows)
    for k, component in enumerate(fit.components):
        n_eff = _effective_distinct_point_count(
            x, observation_weights * fit.responsibilities[:, k], inverse=inverse
        )
        if n_eff <= int(component.effective_curvature_degree) + 2:
            return True
    return False


def _policy_initial_components(
    support,
    rows,
    policies,
    lower,
    upper,
    observation_weights,
    responsibilities,
    solver_options,
    /,
    *,
    degree_config,
):
    """Run the first M-step for heterogeneous fixed/automatic degree policies.

    Parameters
    ----------
    support : tuple of (float, float)
        Shared user-coordinate support.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Point or interval observations.
    policies : tuple of int or str
        Canonical fixed degree or ``"auto"`` policy per component.
    lower, upper : bool
        Boundary-amplitude flags.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Initial component responsibilities.
    solver_options : _NewtonOptions
        Shared immutable Newton policy.
    degree_config : _DegreeSelectionConfig
        Omitted-information selection policy.

    Returns
    -------
    problems : list of _ComponentProblem
        Fixed-coordinate component problems.
    layouts : list of _NaturalLayout
        Locked natural layouts.
    results : list of _ConicNewtonResult
        First-M-step fitted component states.
    """
    problems = []
    layouts = []
    results = []
    if len(policies) > 1:
        degrees = [
            AUTO_POLY_DEGREE_MIN if policy == "auto" else int(policy)
            for policy in policies
        ]
        problems = [
            _ComponentProblem(
                support,
                rows,
                degree,
                lower,
                upper,
                observation_weights * responsibilities[:, k],
            )
            for k, degree in enumerate(degrees)
        ]
        layouts = [problem.spec.layout for problem in problems]
        objectives = [
            problem.objective(observation_weights * responsibilities[:, k])
            for k, problem in enumerate(problems)
        ]
        if lower or upper:
            results = _coupled_m_step(
                problems,
                objectives,
                observation_weights @ responsibilities,
                None,
                solver_options,
            )
        else:
            results = [
                problem.fit(
                    observation_weights * responsibilities[:, k], None, solver_options
                )[1]
                for k, problem in enumerate(problems)
            ]
        return problems, layouts, results
    for k, policy in enumerate(policies):
        component_weights = observation_weights * responsibilities[:, k]
        if policy == "auto":
            if rows.shape[1] == 1:
                objective, result = _fit_natural_conic_points_auto(
                    support,
                    rows[:, 0],
                    lower,
                    upper,
                    component_weights,
                    degree_config=degree_config,
                    solver_options=solver_options,
                )
            else:
                objective, result = _fit_natural_conic_intervals_auto(
                    support,
                    rows,
                    lower,
                    upper,
                    component_weights,
                    degree_config=degree_config,
                    solver_options=solver_options,
                )
            selected = int(objective.spec.requested_poly_degree)
            problem = _ComponentProblem(
                support, rows, selected, lower, upper, component_weights
            )
            if problem.coordinate != objective.spec.coordinate:
                raise RuntimeError(
                    "natural auto-degree component coordinate changed while locking degree"
                )
        else:
            problem = _ComponentProblem(
                support, rows, int(policy), lower, upper, component_weights
            )
            objective, result = problem.fit(component_weights, None, solver_options)
        problems.append(problem)
        layouts.append(objective.layout)
        results.append(result)
    return problems, layouts, results


def _run_natural_em(
    support,
    rows,
    degree,
    lower,
    upper,
    observation_weights,
    responsibilities,
    /,
    *,
    em_options=None,
    initialization="given",
    polish=True,
    solver_options=None,
    degree_config,
    _continuation=None,
    _return_continuation=False,
    _continue_with_em=False,
):
    """Fit a mixture from initial responsibilities: EM, polish, verify.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Point samples or interval rows.
    degree : tuple of int or str
        Canonical fixed degree or ``"auto"`` per component, prepared at the
        input boundary. Automatic entries initialize the shared growth policy.
    lower, upper : bool
        Boundary amplitude flags.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Initial responsibilities.
    em_options : _EMOptions or None, optional
        Immutable EM scheduling policy.
    initialization : str, optional
        Name recorded in the result.
    polish : bool, optional
        ``False`` runs one EM phase only (status ``em_only``), as the
        intermediate rungs of a degree ladder do.
    solver_options : _NewtonOptions or None, optional
        Immutable Newton policy shared by component M-steps and joint polish.
    degree_config : _DegreeSelectionConfig
        Explicit omitted-information policy shared by the owning fit.
    _continuation : _EMContinuation or None, optional
        Internal explored state to continue directly into polishing.
    _return_continuation : bool, optional
        Internally return the reusable EM continuation with the fit.
    _continue_with_em : bool, optional
        Whether a supplied continuation should resume with another EM phase
        before polishing.

    Returns
    -------
    _NaturalMixtureFit or tuple
        Fitted mixture, or ``(fit, continuation)`` when
        ``_return_continuation`` is true.
    """
    w = observation_weights
    em = _resolve_em_options(em_options)
    newton = _resolve_newton_options(solver_options)
    if _continuation is None:
        r = np.asarray(responsibilities, dtype=np.float64)
        k_count = r.shape[1]
        _check_point_estimability(rows, w, r)
        policies = degree
        if any(policy == "auto" for policy in policies):
            problems, layouts, results = _policy_initial_components(
                support,
                rows,
                policies,
                lower,
                upper,
                w,
                r,
                newton,
                degree_config=degree_config,
            )
        else:
            degrees = tuple(int(value) for value in policies)
            problems = [
                _ComponentProblem(support, rows, degrees[k], lower, upper, w * r[:, k])
                for k in range(k_count)
            ]
            results = []
            layouts = [problem.spec.layout for problem in problems]
            if k_count > 1 and (lower or upper):
                objectives = [
                    problem.objective(w * r[:, k]) for k, problem in enumerate(problems)
                ]
                results = _coupled_m_step(problems, objectives, w @ r, None, newton)
            else:
                for k, problem in enumerate(problems):
                    _, result = problem.fit(w * r[:, k], None, newton)
                    results.append(result)
        log_weights = _log_mixture_weights(w @ r)
        state = _evaluated_em_state(problems, layouts, results, log_weights, w)
        history = [state.log_likelihood]
        em_steps = 0
        rounds = 0
        need_em = True
    else:
        problems = list(_continuation.problems)
        layouts = list(_continuation.layouts)
        state = _continuation.state
        history = list(_continuation.history)
        em_steps = int(_continuation.em_iterations)
        rounds = int(_continuation.rounds)
        need_em = bool(_continue_with_em)
    polish_iterations = 0
    status = "em_only"
    bound = np.inf
    while True:
        if need_em:
            if rounds >= int(em.max_rounds):
                break
            rounds += 1
            state, steps = _em_phase(
                state,
                problems,
                layouts,
                w,
                max_steps=em.max_steps,
                tolerance=em.tolerance,
                accelerate=em.accelerate,
                history=history,
                solver_options=newton,
            )
            em_steps += steps
        if not polish:
            break
        run, objective = _polish(state, problems, layouts, w, newton)
        if run is None:
            status = "em_only"
            break
        polish_iterations += run.iterations
        params, log_weights = objective.split(run.params)
        polished_ll = -float(run.evaluation.nll)
        if polished_ll < state.log_likelihood:
            status = "em_only"
            break
        history.append(polished_ll)
        _, polished_posterior = _e_step(problems, layouts, params, log_weights, w)
        polished_objective = SharedMStepObjective(
            [
                (kind, 1.0, inputs)
                for kind, inputs in (
                    _compiled_component(problem.objective(w * polished_posterior[:, k]))
                    for k, problem in enumerate(problems)
                )
            ]
        )
        polished = _coupled_component_results(
            objective.face_run,
            objective.face,
            polished_objective.evaluate(
                objective.face.component_columns,
                len(objective.face.free_indices),
                objective.face_run.params,
            ),
        )
        state = _EMState(
            tuple(polished),
            np.asarray(log_weights, dtype=np.float64),
            polished_ll,
            polished_posterior,
        )
        status, bound = run.status, float(run.decrease_bound)
        if status not in _CONVERGED:
            need_em = True
            continue
        # Verification: one EM step from the polished point.
        verified, verified_log_weights = _em_map(state, problems, w, newton)
        em_steps += 1
        verified_state = _evaluated_em_state(
            problems, layouts, verified, verified_log_weights, w
        )
        scale = max(1.0, abs(polished_ll))
        if any(result.status == "joint_face_unresolved" for result in verified):
            status, bound = "face_search_unresolved", np.inf
            if verified_state.log_likelihood >= polished_ll:
                state = verified_state
                history.append(state.log_likelihood)
            need_em = True
            continue
        verified_face = _result_face(objective.natural_map, verified, len(verified) - 1)
        if (
            len(verified_face.free_indices) < len(objective.face.free_indices)
            and verified_state.log_likelihood >= polished_ll - newton.tolerance * scale
            and rounds < int(em.max_rounds)
        ):
            state = verified_state
            history.append(state.log_likelihood)
            rounds += 1
            need_em = False
            continue
        if verified_state.log_likelihood - polished_ll <= max(
            bound, newton.certified_tolerance * scale
        ):
            break
        # EM still climbs: the polish sat on the wrong face.  Continue from
        # the EM point with its (re-identified) faces.
        state = verified_state
        history.append(verified_state.log_likelihood)
        status = "em_only"
        need_em = True

    results = state.results
    log_weights = state.log_weights
    current = state.log_likelihood
    components = tuple(
        _NaturalComponent(
            coordinate=p.coordinate,
            spec=p.spec,
            layout=layout,
            z_data_bounds=p.z_data_bounds,
            params=np.asarray(res.params, dtype=np.float64),
            effective_curvature_degree=int(res.effective_curvature_degree),
            lower_amplitude_active=bool(res.lower_amplitude_active),
            upper_amplitude_active=bool(res.upper_amplitude_active),
            solver_result=res,
        )
        for p, layout, res in zip(problems, layouts, results, strict=True)
    )
    certified = all(_certify(c.layout, c.params).feasible for c in components)
    if not certified and status in _CONVERGED:
        status = "uncertified"
    final_objective = _compiled_mixture(problems, layouts, w)
    final_vector = final_objective.join([c.params for c in components], log_weights)
    final_face = _result_face(final_objective.natural_map, results, len(results) - 1)
    final_information = final_objective(final_vector).observed_hessian
    public_responsibilities = state.responsibilities
    if problems and problems[0].intervals and problems[0].distinct_rows is not None:
        inverse = problems[0].distinct_rows[1]
        if public_responsibilities.shape[0] != problems[0].rows.shape[0]:
            public_responsibilities = public_responsibilities[inverse]
    fit = _NaturalMixtureFit(
        components=components,
        weights=np.exp(log_weights),
        log_likelihood=float(current),
        status=status,
        decrease_bound=bound,
        em_iterations=em_steps,
        polish_iterations=polish_iterations,
        rounds=rounds,
        history=tuple(history),
        initialization=initialization,
        separator_certified=bool(certified),
        responsibilities=np.asarray(public_responsibilities, dtype=np.float64),
        n_parameters=final_objective.n_params,
        n_face_parameters=len(final_face.free_indices),
        observed_information=final_information,
        shared_parameter_indices=final_objective.natural_map.shared_parameter_indices,
        free_parameter_indices=tuple(map(int, final_face.free_indices)),
    )
    if not _return_continuation:
        return fit
    continuation = _EMContinuation(
        problems=tuple(problems),
        layouts=tuple(layouts),
        state=state,
        history=tuple(history),
        em_iterations=int(em_steps),
        rounds=int(rounds),
    )
    return fit, continuation


def _degree_probe_problems(fit, rows, observation_weights, degrees):
    """Lift a mixture's private degree blocks at exactly zero, without fitting.

    Parameters
    ----------
    fit : _NaturalMixtureFit
        Existing fit supplying coordinates, amplitudes and lower-order terms.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical observation rows.
    observation_weights : numpy.ndarray, shape (R,)
        Canonical normalized observation weights.
    degrees : sequence of int
        Target degree of each component probe.

    Returns
    -------
    tuple
        ``(problems, layouts, params)`` for the lifted probe geometry.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    problems, layouts, params = [], [], []
    for k, (component, degree) in enumerate(zip(fit.components, degrees, strict=True)):
        problem = _ComponentProblem(
            component.coordinate.physical_support,
            rows,
            int(degree),
            component.spec.physical_lower_a_index is not None,
            component.spec.physical_upper_a_index is not None,
            observation_weights * fit.responsibilities[:, k],
            _coordinate=component.coordinate,
        )
        layout = problem.spec.layout
        theta = np.zeros(layout.n_params)
        theta[layout.gamma_index] = component.params[component.layout.gamma_index]
        count = min(layout.curvature_degree, component.layout.curvature_degree) + 1
        theta[layout.curvature_slice.start : layout.curvature_slice.start + count] = (
            component.params[component.layout.curvature_slice][:count]
        )
        for side in ("lower", "upper"):
            new_index = getattr(problem.spec, f"physical_{side}_a_index")
            old_index = getattr(component.spec, f"physical_{side}_a_index")
            if new_index is not None:
                theta[new_index] = component.params[old_index]
        problems.append(problem)
        layouts.append(layout)
        params.append(theta)
    _MixtureNaturalMap([problem.spec for problem in problems], layouts).pack(params)
    return problems, layouts, params


def _continue_natural_mixture(
    fit,
    support,
    rows,
    degrees,
    lower,
    upper,
    observation_weights,
    /,
    *,
    em_options=None,
    polish=True,
    solver_options=None,
    degree_config,
    _return_continuation=False,
):
    """Continue a fitted mixture after degree/side changes, keeping coordinates.

    Lower-order polynomial coefficients and retained physical amplitudes are
    copied exactly into the new layouts before the first coupled optimization.

    Parameters
    ----------
    fit : _NaturalMixtureFit
        Existing fitted mixture supplying coordinates and warm-start state.
    support : tuple of (float, float)
        Physical support, which must match ``fit``.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical observation rows.
    degrees : tuple of int
        Canonical explicit target degrees, one per component.
    lower, upper : bool
        Boundary-amplitude flags.
    observation_weights : numpy.ndarray, shape (R,)
        Canonical normalized observation weights.
    em_options : _EMOptions or None, optional
        EM scheduling policy for the continued fit.
    polish : bool, optional
        Whether to run joint Newton polishing after EM.
    solver_options : _NewtonOptions or None, optional
        Shared Newton policy.
    degree_config : _DegreeSelectionConfig
        Omitted-information policy used by downstream degree growth.
    _return_continuation : bool, optional
        Internally return the reusable continuation alongside the fit.

    Returns
    -------
    _NaturalMixtureFit or tuple
        Continued fit, optionally paired with its reusable continuation.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    w = np.asarray(observation_weights, dtype=np.float64)
    em = _resolve_em_options(em_options)
    newton = _resolve_newton_options(solver_options)
    if tuple(map(float, support)) != tuple(
        map(float, fit.components[0].coordinate.physical_support)
    ):
        raise ValueError("continuation cannot change the fitted physical support")
    _MixtureNaturalMap([component.spec for component in fit.components]).pack(
        [component.params for component in fit.components]
    )
    previous_problems = [
        _ComponentProblem(
            support,
            rows,
            component.spec.requested_poly_degree,
            component.spec.physical_lower_a_index is not None,
            component.spec.physical_upper_a_index is not None,
            w,
            _coordinate=component.coordinate,
        )
        for component in fit.components
    ]
    _, responsibilities = _e_step(
        previous_problems,
        [component.layout for component in fit.components],
        [component.params for component in fit.components],
        np.log(fit.weights),
        w,
    )
    if any(degree == "auto" for degree in degrees):
        raise ValueError("continuation requires explicit per-component degrees")
    problems, layouts, results = [], [], []
    for k, (component, degree) in enumerate(zip(fit.components, degrees, strict=True)):
        problem = _ComponentProblem(
            support,
            rows,
            degree,
            lower,
            upper,
            w * responsibilities[:, k],
            _coordinate=component.coordinate,
        )
        layout = problem.spec.layout
        params = np.zeros(layout.n_params)
        params[layout.gamma_index] = component.params[component.layout.gamma_index]
        count = min(layout.curvature_degree, component.layout.curvature_degree) + 1
        params[layout.curvature_slice.start : layout.curvature_slice.start + count] = (
            component.params[
                component.layout.curvature_slice.start : component.layout.curvature_slice.start
                + count
            ]
        )
        for side in ("lower", "upper"):
            new_index = getattr(problem.spec, f"physical_{side}_a_index")
            old_index = getattr(component.spec, f"physical_{side}_a_index")
            if new_index is not None:
                params[new_index] = (
                    component.params[old_index] if old_index is not None else 0.0
                )
        results.append(
            replace(
                component.solver_result,
                params=params,
                blocks=(),
                dual=np.empty(0),
                evaluation=None,
                objective_value=np.nan,
                effective_curvature_degree=min(
                    layout.curvature_degree, component.effective_curvature_degree
                ),
                lower_amplitude_active=(
                    layout.lower_a_index is not None
                    and params[layout.lower_a_index] > 0
                ),
                upper_amplitude_active=(
                    layout.upper_a_index is not None
                    and params[layout.upper_a_index] > 0
                ),
                status="continued",
                final_decrease_bound=np.inf,
                final_separation=None,
            )
        )
        problems.append(problem)
        layouts.append(layout)
    state = _EMState(
        tuple(results),
        np.log(fit.weights),
        fit.log_likelihood,
        responsibilities,
    )
    results, log_weights = _em_map(state, problems, w, newton)
    state = _evaluated_em_state(problems, layouts, results, log_weights, w)
    continuation = _EMContinuation(
        tuple(problems), tuple(layouts), state, (state.log_likelihood,), 1, 0
    )
    return _run_natural_em(
        support,
        rows,
        degrees,
        lower,
        upper,
        w,
        responsibilities,
        em_options=em,
        initialization=fit.initialization,
        polish=polish,
        solver_options=newton,
        degree_config=degree_config,
        _continuation=continuation,
        _return_continuation=_return_continuation,
        _continue_with_em=True,
    )


def _fit_natural_mixture(
    support,
    samples,
    n_components,
    poly_degree,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    weights=None,
    /,
    *,
    rng=0,
    responsibilities=None,
    search_options=None,
    degree_config,
    initial_fit=None,
    em_options=None,
    solver_options=None,
):
    """Fit a mixture of natural log-concave components.

    The observed mixture likelihood has many stationary points and each fit
    is only certified stationary, so this is a multi-start.  Every
    initialization candidate (``mixture._initial_responsibility_candidates``)
    is explored along every path by EM alone; the ``finalists`` best
    explorations are then run to certified convergence, and the best is
    returned.  Explorations and fits are ranked by log likelihood, or by BIC
    when automatic degrees select different models. A resolved KDE
    valley is the only candidate; when its result is doubtful
    (``_needs_wider_search``) the GMM and nested-scale candidates are searched
    too.  Among finished fits a sound one (certified, every component
    determined by more effective observations than it has parameters) is
    preferred to an unsound one, then the higher score.

    A path is a pair ``(schedule, treatment)``.  The ``ladder`` schedule
    climbs the admissible degree ladder with the same shared boundary bases:
    each rung lifts the previous fit without changing its coordinates.
    ``direct`` starts at the target degree. The ``sharpened`` treatment squares
    and renormalizes the initial responsibilities; ``raw`` does not.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support shared by every component.
    samples : numpy.ndarray, shape (R, 1) or (R, 2), dtype float64
        Boundary-validated point samples or interval rows in user coordinates.
    n_components : int
        Number of components.
    poly_degree : tuple of int or str
        Canonical integer/``"auto"`` policy per component, prepared at the
        input boundary. Automatic components
        start at minimum degree; finalists grow jointly conditioned blocks.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Boundary amplitude flags.
    weights : numpy.ndarray, shape (R,), dtype float64 or None, optional
        Boundary-normalized observation weights, or ``None`` for equal weights.
    rng : int or numpy.random.Generator, optional
        Randomness for the initialization candidates.
    responsibilities : numpy.ndarray or None, optional
        Explicit initial responsibilities; replaces the candidates.
    search_options : _MixtureSearchOptions or None, optional
        Multistart path/finalist policy.
    degree_config : _DegreeSelectionConfig
        Explicit omitted-information policy shared by this fit and its refits.
    initial_fit : _NaturalMixtureFit or None, optional
        Shared fit to continue without restarting the initializer search or
        changing its numerical coordinates.
    em_options : _EMOptions or None, optional
        EM scheduling policy for finalists and warm continuations. Exploratory
        screening keeps the default policy so candidate ranking is comparable.
    solver_options : _NewtonOptions or None, optional
        Newton policy for finalists and warm continuations.

    Returns
    -------
    _NaturalMixtureFit
        Best fitted mixture.

    Raises
    ------
    RuntimeError
        If no initialization produced a fit.
    """
    rows = samples
    w = _observation_weights(rows.shape[0], weights)
    policies = poly_degree
    has_auto_degree = any(policy == "auto" for policy in policies)
    lower, upper = bool(allow_lower_boundary), bool(allow_upper_boundary)
    search_policy = _resolve_search_options(search_options)
    final_em = _resolve_em_options(em_options)
    final_newton = _resolve_newton_options(solver_options)
    exploration_em = _EMOptions()
    exploration_newton = _NewtonOptions()

    def grow(fit):
        # Cold K=1 initialization already runs the standalone selector.
        # Warm starts are refitted at minimum degree and must grow again.
        if not has_auto_degree or (int(n_components) == 1 and initial_fit is None):
            return fit
        # Degree growth calls back into mixture refitting; keep imports reciprocal
        # only at runtime, after both modules have initialized.
        from .mixture_degree import _fit_shared_degree_growth

        return _fit_shared_degree_growth(
            fit,
            policies,
            support=support,
            rows=rows,
            observation_weights=w,
            refit=lambda degrees, previous: _continue_natural_mixture(
                previous,
                support,
                rows,
                degrees,
                lower,
                upper,
                w,
                em_options=final_em,
                solver_options=final_newton,
                degree_config=degree_config,
            ),
            degree_config=degree_config,
        )

    if initial_fit is not None:
        if len(initial_fit.components) != int(n_components):
            raise ValueError(
                "initial fit component count does not match requested mixture"
            )
        degrees = tuple(
            AUTO_POLY_DEGREE_MIN if policy == "auto" else int(policy)
            for policy in policies
        )
        fit = _continue_natural_mixture(
            initial_fit,
            support,
            rows,
            degrees,
            lower,
            upper,
            w,
            em_options=final_em,
            solver_options=final_newton,
            degree_config=degree_config,
        )
        return grow(fit)
    if responsibilities is not None:
        candidates = [("given", np.asarray(responsibilities, dtype=np.float64), None)]
    else:
        representatives = (
            rows[:, 0]
            if rows.shape[1] == 1
            else _interval_initial_representatives(rows, support)
        )
        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        candidates = _initial_responsibility_candidates(
            representatives, int(n_components), generator, weights=weights
        )
    uniform_fixed = (
        not has_auto_degree and len({int(policy) for policy in policies}) == 1
    )
    degree = policies
    rungs = (
        []
        if not uniform_fixed
        else [
            int(d)
            for d in _admissible_degrees(support, policies[0])
            if int(d) < policies[0]
        ]
    )

    def prepare(posterior, treatment):
        if treatment == "raw":
            return posterior
        squared = np.square(posterior)
        return squared / np.sum(squared, axis=1, keepdims=True)

    # With automatic degrees, explorations from different initializations can
    # lock different degrees, so their likelihoods compare models of
    # different size; raw likelihood would always favor the larger one.
    # Rank them by BIC (per observation, as the likelihoods are), the
    # criterion the component-count selection uses.
    effective_n = 1.0 / float(np.dot(w, w))
    bic_penalty = 0.5 * np.log(effective_n) / effective_n if has_auto_degree else 0.0

    def score(fit):
        size = fit.n_face_parameters
        return float(fit.log_likelihood) - bic_penalty * size

    def rank(fit):
        # A sound fit (certified, every component determined by more
        # effective observations than it has parameters) outranks any
        # unsound one; the score decides within each class.  Otherwise an
        # uncertified component on a handful of extreme points can win on
        # likelihood over a certified ordinary fit.
        return (not _needs_wider_search(fit, rows, w), score(fit))

    # Exploration uses the default EM/M-step controls. Its terminal state is
    # reusable only when the finalist has the same EM schedule and the same
    # M-step mechanics. The three convergence tolerances remain polish-safe,
    # matching the previous continuation policy.
    default_newton = _NewtonOptions()
    reuse_exploration = final_em == exploration_em and all(
        getattr(final_newton, name) == getattr(default_newton, name)
        for name in (
            "face_trigger",
            "max_iterations",
            "armijo",
            "backtrack",
            "max_line_search",
        )
    )
    active_paths = list(search_policy.paths)
    if not uniform_fixed:
        # Heterogeneous or automatic degree policies do not have one shared
        # ladder. Preserve the distinct raw/sharpened responsibility
        # treatments without running duplicate schedule labels.
        treatments = []
        for _, treatment in active_paths:
            if treatment not in treatments:
                treatments.append(treatment)
        active_paths = [("direct", treatment) for treatment in treatments]

    def search(candidates):
        """Explore every candidate along every path; finish the best finalists."""
        explored = []
        failure = None
        for name, initial, _ in candidates:
            for schedule, treatment in active_paths:
                label = f"{name}/{schedule}/{treatment}"
                try:
                    posterior = initial
                    rung_fit = None
                    if schedule == "ladder":
                        for rung in rungs:
                            if rung_fit is None:
                                rung_fit = _run_natural_em(
                                    support,
                                    rows,
                                    (rung,) * n_components,
                                    lower,
                                    upper,
                                    w,
                                    prepare(posterior, treatment),
                                    em_options=exploration_em,
                                    polish=False,
                                    solver_options=exploration_newton,
                                    degree_config=degree_config,
                                )
                            else:
                                rung_fit = _continue_natural_mixture(
                                    rung_fit,
                                    support,
                                    rows,
                                    (rung,) * n_components,
                                    lower,
                                    upper,
                                    w,
                                    em_options=exploration_em,
                                    polish=False,
                                    solver_options=exploration_newton,
                                    degree_config=degree_config,
                                )
                            posterior = rung_fit.responsibilities
                    start = prepare(posterior, treatment)
                    if rung_fit is not None:
                        exploration, continuation = _continue_natural_mixture(
                            rung_fit,
                            support,
                            rows,
                            degree,
                            lower,
                            upper,
                            w,
                            em_options=exploration_em,
                            polish=False,
                            solver_options=exploration_newton,
                            degree_config=degree_config,
                            _return_continuation=True,
                        )
                    elif reuse_exploration:
                        exploration, continuation = _run_natural_em(
                            support,
                            rows,
                            degree,
                            lower,
                            upper,
                            w,
                            start,
                            em_options=exploration_em,
                            polish=False,
                            solver_options=exploration_newton,
                            degree_config=degree_config,
                            _return_continuation=True,
                        )
                    else:
                        exploration = _run_natural_em(
                            support,
                            rows,
                            degree,
                            lower,
                            upper,
                            w,
                            start,
                            em_options=exploration_em,
                            polish=False,
                            solver_options=exploration_newton,
                            degree_config=degree_config,
                        )
                        continuation = None
                except _PointMixtureEstimabilityFailure as exc:
                    # This seed reached the known singular point-mixture boundary.
                    # Other starts may remain estimable, so reject only this seed.
                    failure = exc
                    continue
                except NUMERIC_FAILURES as exc:
                    _reraise_if_debug(
                        exc, f"natural mixture EM from {label}", routine=True
                    )
                    failure = exc
                    continue
                locked_degrees = tuple(
                    int(component.spec.requested_poly_degree)
                    for component in exploration.components
                )
                explored.append(
                    (score(exploration), label, start, locked_degrees, continuation)
                )
        explored.sort(key=lambda item: -item[0])
        best = None
        for _, label, start, locked_degrees, continuation in explored[
            : max(1, int(search_policy.finalists))
        ]:
            final_degree = locked_degrees if has_auto_degree else degree
            try:
                fit = _run_natural_em(
                    support,
                    rows,
                    final_degree,
                    lower,
                    upper,
                    w,
                    start,
                    em_options=final_em,
                    initialization=label,
                    solver_options=final_newton,
                    degree_config=degree_config,
                    _continuation=continuation,
                )
                fit = grow(fit)
            except _PointMixtureEstimabilityFailure as exc:
                # This finalist reached the known singular point-mixture boundary.
                # Continue to another finalist without treating it as a fallback.
                failure = exc
                continue
            except NUMERIC_FAILURES as exc:
                _reraise_if_debug(
                    exc, f"natural mixture fit from {label}", routine=True
                )
                failure = exc
                continue
            if best is None or rank(fit) > rank(best):
                best = fit
        return best, failure

    best, failure = search(candidates)
    if (
        responsibilities is None
        and candidates[0][0] == "valley"
        and _needs_wider_search(best, rows, w)
    ):
        # A lone valley seed can cut off the two or three most extreme
        # points; seeded only there the multi-start ends on that split (or on
        # the one-location singularity) although other seeds reach a higher
        # likelihood.  Widen the search to the other families and keep the
        # better of the two by the same score.
        alternatives = _initial_responsibility_candidates(
            representatives,
            int(n_components),
            generator,
            weights=weights,
            include_valley=False,
        )
        other, other_failure = search(alternatives)
        if other is not None and (best is None or rank(other) > rank(best)):
            best = other
        failure = failure if failure is not None else other_failure
    if best is None:
        if isinstance(failure, _PointMixtureEstimabilityFailure):
            raise failure
        raise RuntimeError("no mixture initialization produced a fit") from failure
    return best
