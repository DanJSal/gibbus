"""Mixtures of natural-coordinate log-concave components.

Each component is fitted by the one natural conic solver.  A fit runs in
three phases:

1. **EM**, accelerated by SQUAREM (Varadhan and Roland, 2008).  The E-step
   computes exact posteriors from point densities or interval masses; the
   M-step updates the mixture weights in closed form and refits every
   component to responsibility-weighted observations, warm-started from its
   previous parameters and Gram certificate.  Point M-steps are convex fits to
   weighted sufficient statistics; interval M-steps are weighted interval
   fits.  EM is monotone and globally convergent but only linearly, and
   slowly when components overlap; it runs to a loose tolerance and fixes
   each component's exact face (effective degree, zero amplitudes).
2. **Joint Newton polish** on the whole observed-data likelihood over all
   component parameters and the mixture logits at once, over the product of
   the components' cone descriptions.  The observed Hessian is the
   complete-data Fisher matrix ``G`` (block diagonal: ``m_k F_k`` per
   component, the multinomial Hessian for the logits) minus the missing
   information ``M`` (the covariance of the complete-data scores over the
   unknown labels, plus the within-row covariances of interval rows).  The
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

import weakref
from dataclasses import dataclass, replace

import numpy as np

from .._defaults import (
    EM_MIN_EFFECTIVE_DISTINCT_N,
    INTERVAL_W_EPS_MULT,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._model.coords import _build_fit_coordinate, _build_interval_fit_coordinate
from .._model.natural_state import _NaturalCoreState
from .._model.spec import _build_model_spec
from .._model.vec import _q_eval
from .._observations._finite_reductions import (
    evaluate_finite_log_probabilities_real_line,
)
from .._observations.empirical import _normalized_weights
from .._observations.intervals import (
    _GL_LOG_W,
    _GL_X,
    _build_interval_observations,
    _prepare_partial_interval_reducer,
    _row_grouping,
)
from .._observations.points import _PointObservations
from ._mixture_kernels import joint_information, mixture_posterior
from .conic_newton import (
    _CONVERGED,
    _certify,
    _newton_on_representation,
    _NewtonOptions,
    _solve_natural_conic,
)
from .conic_qp import _ConicRepresentation, _support_representation
from .inputs import _admissible_degrees
from .mixture import (
    _distinct_location_index,
    _effective_distinct_point_count,
    _initial_responsibility_candidates,
    _interval_initial_representatives,
    _point_order,
)
from .natural_objective import (
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points_auto,
    _natural_interval_start,
    _natural_point_stats,
    _NaturalIntervalEvaluation,
    _NaturalIntervalObjectiveFunction,
    _NaturalPointObjectiveFunction,
    _preserved_point_boundary_distances,
    _safeguarded_metric,
)
from .objective import _model_first_means_and_fisher


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
        Final component M-step result retained for public fit diagnostics.
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
        return _NaturalCoreState(self.coordinate, self.layout, self.params, self.z_data_bounds)


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


def _normalized_observation_weights(n, weights, /):
    """Return observation weights normalized to sum to one.

    Parameters
    ----------
    n : int
        Number of observations.
    weights : array_like or None
        Nonnegative weights, or ``None`` for uniform.
    """
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n or np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite, nonnegative and match the samples")
    total = float(np.sum(w))
    if not total > 0.0:
        raise ValueError("weights must have a positive total")
    return w / total


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
    from .._model.natural_state import _natural_partials
    from .._model.spec import _LOGDIST, _LOWER

    values = np.asarray(z, dtype=np.float64)
    lower, upper = map(float, support)
    columns = []
    for partial in _natural_partials(layout):
        if partial.kind != _LOGDIST:
            columns.append(partial.evaluate(values, support))
            continue
        if partial.boundary_side == _LOWER:
            distance = values - lower if lower_distance is None else np.asarray(
                lower_distance, dtype=np.float64
            )
        else:
            distance = upper - values if upper_distance is None else np.asarray(
                upper_distance, dtype=np.float64
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

    def __init__(self, support, rows, degree, lower, upper, initial_weights, /):
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
            coordinate = _build_fit_coordinate(
                support, x, initial_weights, None, order=_point_order(rows)[0]
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
                    inverse, weights=np.asarray(initial_weights, dtype=np.float64),
                    minlength=n_unique,
                ).astype(np.float64)
            if np.all(np.isfinite(coordinate_rows)):
                mid, width, mid_order, width_order = _interval_coordinate_geometry(
                    coordinate_rows
                )
                coordinate = _build_fit_coordinate(
                    support, mid, coordinate_weights, width,
                    order=mid_order, width_order=width_order,
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
                    np.ones(n_unique, dtype=np.float64),
                    support=coordinate.canonical_support,
                    deduplicate=False,
                )
        self.coordinate = coordinate
        self.spec = _build_model_spec(coordinate, int(degree), bool(lower), bool(upper))
        self.z_data_bounds = bounds
        self.order = max(4, 2 * int(self.spec.effective_poly_degree))
        self.log_scale = float(np.log(coordinate.scale))
        self._point_basis = None

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
            layout, support, self.z,
            lower_distance=self.lower_distance, upper_distance=self.upper_distance,
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
                self.rows, weights, coordinate=self.coordinate,
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
                    self.rows[first], grouped_weights, coordinate=self.coordinate
                )
                observations = replace(
                    observations,
                    total_weight=float(total),
                    effective_n=float(effective_n),
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
        group_sum = np.bincount(
            inverse, weights=w, minlength=n_unique
        ).astype(np.float64)
        group_sq = np.bincount(
            inverse, weights=w * w, minlength=n_unique
        ).astype(np.float64)
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
                self.rows[first], grouped_weights, coordinate=self.coordinate
            )
            observations = replace(
                observations,
                total_weight=total,
                effective_n=float(total * total / square_sum),
                n_observations=int(self.rows.shape[0]),
            )
        return _NaturalIntervalObjectiveFunction(
            self.spec, observations, self.z_data_bounds, nonparametric_bound=False
        )

    def fit_compact(self, responsibilities, observation_weights, previous, /, **options):
        """Run one duplicated-interval M-step without expanding row weights.

        Parameters
        ----------
        responsibilities : numpy.ndarray, shape (R_unique,)
            Posterior component probabilities for distinct interval rows.
        observation_weights : numpy.ndarray, shape (R,)
            Original-row normalized reliability weights.
        previous : _ConicNewtonResult or None
            Previous component result for a warm start.
        **options : dict
            Passed to ``_solve_natural_conic``.

        Returns
        -------
        objective : _NaturalIntervalObjectiveFunction
            Compressed M-step objective.
        result : _ConicNewtonResult
            Optimized component result.
        """
        objective = self.compact_objective(responsibilities, observation_weights)
        if previous is not None:
            options.setdefault("initial", previous.params)
            options.setdefault("initial_blocks", previous.blocks)
        else:
            start, blocks = _natural_interval_start(objective)
            options.setdefault("initial", start)
            options.setdefault("initial_blocks", blocks)
        return objective, _solve_natural_conic(objective, certify=False, **options)

    def row_statistics(self, state, layout, /, moments=False, distinct=False):
        """Return per-row log contributions and conditional statistics.

        Log contributions are in user coordinates (log densities per user
        unit for points and zero-width rows, log masses otherwise), so they
        are comparable across components with different coordinates.

        Parameters
        ----------
        state : _NaturalCoreState
            Normalized component state in this problem's coordinate.
        layout : _NaturalLayout
            Its natural layout.
        moments : bool, optional
            Also return the conditional means ``E[h | row]`` and, for
            interval data, the weighted-sum-ready conditional covariances.
        distinct : bool, optional
            For interval data, return one entry per exact distinct row instead
            of expanding statistics back to the original observations.

        Returns
        -------
        log_values : numpy.ndarray, shape (R,)
        means : numpy.ndarray, shape (R, n) or None
        covariances : numpy.ndarray, shape (R, n, n) or None
            ``None`` for point data (zero within-row covariance).

        With ``distinct=True`` interval data return one row per distinct
        interval (``distinct_rows`` gives the mapping); callers that only
        sum over rows then add the weights of identical rows instead of
        expanding the arrays.
        """
        support = tuple(map(float, state.spec.support))
        if not self.intervals:
            q = np.polynomial.polynomial.polyval(self.z, state.q_poly)
            a_lower, a_upper = map(float, state.boundary_amplitudes)
            with np.errstate(divide="ignore", invalid="ignore"):
                if np.isfinite(a_lower) and a_lower > 0.0:
                    q = q - a_lower * np.log(self.lower_distance)
                if np.isfinite(a_upper) and a_upper > 0.0:
                    q = q - a_upper * np.log(self.upper_distance)
            log_values = -np.asarray(q, dtype=np.float64) - float(state.log_Z) - self.log_scale
            means = self.point_basis(layout, support) if moments else None
            return log_values, means, None

        z = self.z
        expand_distinct = not distinct
        distinct = self._distinct
        if distinct is not None:
            z = distinct[0]

        # The E-step asks only for row log masses.  On the real line every
        # finite interval can use the same deterministic local quadrature as
        # the M-step objective, without constructing conditional moments.
        if (
            not moments
            and np.isneginf(support[0])
            and np.isposinf(support[1])
            and np.all(np.isfinite(z))
        ):
            log_values = evaluate_finite_log_probabilities_real_line(
                np.ascontiguousarray(z, dtype=np.float64),
                np.ascontiguousarray(state.q_poly, dtype=np.float64),
                float(state.q_shift),
                float(np.log(state.Z)),
                float(state.mode),
                self.log_scale,
                _GL_X,
                _GL_LOG_W,
                float(INTERVAL_W_EPS_MULT),
            )
            if distinct is not None and expand_distinct:
                log_values = log_values[distinct[1]]
            return np.asarray(log_values, dtype=np.float64), None, None

        n = layout.n_params
        count = z.shape[0]
        log_values = np.empty(count, dtype=np.float64)
        means = np.empty((count, n), dtype=np.float64) if moments else None
        covariances = np.zeros((count, n, n), dtype=np.float64) if moments else None
        point = z[:, 0] == z[:, 1]
        whole = (z[:, 0] == support[0]) & (z[:, 1] == support[1])
        spread = ~point & ~whole
        if np.any(point):
            zp = z[point, 0]
            q = _q_eval(zp, support, state.q_poly, state.boundary_amplitudes, 0)
            log_values[point] = -np.asarray(q, dtype=np.float64) - float(state.log_Z) - self.log_scale
            if moments:
                means[point] = _basis(layout, support, zp)
        if np.any(whole):
            log_values[whole] = 0.0
            if moments:
                mu, fisher = _model_first_means_and_fisher(state)
                means[whole] = mu
                covariances[whole] = fisher
        if np.any(spread):
            reducer = _prepare_partial_interval_reducer(state)
            log_probability, mean, covariance, _ = reducer.reduce_many(
                np.ascontiguousarray(z[spread], dtype=np.float64)
            )
            log_values[spread] = np.asarray(log_probability, dtype=np.float64)
            if moments:
                means[spread] = np.asarray(mean, dtype=np.float64)
                covariances[spread] = np.asarray(covariance, dtype=np.float64)
        if distinct is not None and not expand_distinct:
            pass
        elif distinct is not None:
            inverse = distinct[1]
            log_values = log_values[inverse]
            if moments:
                means = means[inverse]
                covariances = covariances[inverse]
        return log_values, means, covariances

    def fit(self, weights, previous, /, **options):
        """Run one M-step for this component.

        Parameters
        ----------
        weights : numpy.ndarray, shape (R,)
            M-step weights.
        previous : _ConicNewtonResult or None
            Previous result for the warm start, or ``None`` for a cold start.
        **options : dict
            Passed to ``_solve_natural_conic``.

        Returns
        -------
        objective, result
            The M-step objective and solver result.
        """
        objective = self.objective(weights)
        if previous is not None:
            options.setdefault("initial", previous.params)
            options.setdefault("initial_blocks", previous.blocks)
        elif self.intervals:
            start, blocks = _natural_interval_start(objective)
            options.setdefault("initial", start)
            options.setdefault("initial_blocks", blocks)
        return objective, _solve_natural_conic(objective, certify=False, **options)


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


def _posterior(columns, log_weights, observation_weights, /):
    """Return ``(log_likelihood, responsibilities)`` from per-component log values.

    Parameters
    ----------
    columns : sequence of numpy.ndarray, each shape (R,)
        Per-row component log values (densities or masses).
    log_weights : numpy.ndarray, shape (K,)
        Log mixture weights.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    """
    status, log_likelihood, responsibilities, _ = mixture_posterior(
        np.ascontiguousarray(np.column_stack(columns), dtype=np.float64),
        np.ascontiguousarray(log_weights, dtype=np.float64),
        np.ascontiguousarray(observation_weights, dtype=np.float64),
    )
    if status:
        raise FloatingPointError("mixture assigns zero likelihood to an observation")
    return float(log_likelihood), responsibilities


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
    # Interval posteriors are identical for duplicate rows.  Evaluate the
    # mixture once per distinct interval and expand only the responsibilities
    # required by the existing EM state/API.
    distinct = problems[0].distinct_rows if problems and problems[0].intervals else None
    if distinct is not None:
        _, inverse = distinct
        grouped_weights, _ = problems[0]._grouped_observation_moments(
            observation_weights
        )
        columns = []
        for problem, layout, theta in zip(problems, layouts, params, strict=True):
            state = _NaturalCoreState(
                problem.coordinate, layout, theta, problem.z_data_bounds
            )
            columns.append(problem.row_statistics(state, layout, distinct=True)[0])
        log_likelihood, responsibilities = _posterior(
            columns, log_weights, grouped_weights
        )
        return (log_likelihood, responsibilities) if compact else (
            log_likelihood, responsibilities[inverse]
        )

    columns = []
    for problem, layout, theta in zip(problems, layouts, params, strict=True):
        state = _NaturalCoreState(
            problem.coordinate, layout, theta, problem.z_data_bounds
        )
        columns.append(problem.row_statistics(state, layout)[0])
    return _posterior(columns, log_weights, observation_weights)


class _JointMixtureObjective:
    """Observed-data mixture NLL over all parameters, with its safeguarded metric.

    The variable is ``(theta_1, ..., theta_K, eta_1, ..., eta_{K-1})`` with
    mixture weights ``softmax(eta, 0)``.
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
        self.weights = observation_weights
        # Identical interval rows contribute identically to every sum below,
        # so the objective runs on distinct rows with their weights added.
        distinct = self.problems[0].distinct_rows if self.problems else None
        self._row_weights = (
            observation_weights if distinct is None
            else np.bincount(distinct[1], weights=observation_weights,
                             minlength=distinct[0].shape[0])
        )
        self._distinct = distinct is not None
        self.sizes = [layout.n_params for layout in layouts]
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes)))
        self.n_components = len(problems)
        self.n_params = int(self.offsets[-1]) + self.n_components - 1

    def split(self, x, /):
        """Return ``(params, log_weights)`` from the joint variable.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,)
            Joint variable.
        """
        params = [
            np.asarray(x[self.offsets[k]: self.offsets[k + 1]], dtype=np.float64)
            for k in range(self.n_components)
        ]
        eta = np.concatenate((x[self.offsets[-1]:], [0.0]))
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
        return np.concatenate([np.asarray(p, dtype=np.float64) for p in params] + [eta])

    def __call__(self, x, /):
        """Return the observed NLL, gradient and safeguarded Newton metric.

        Parameters
        ----------
        x : numpy.ndarray, shape (N,)
            Joint variable.
        """
        params, log_weights = self.split(np.asarray(x, dtype=np.float64))
        k_count = self.n_components
        n_total = self.n_params
        log_values = []
        centered = []
        within = []
        fishers = []
        for problem, layout, theta in zip(self.problems, self.layouts, params, strict=True):
            state = _NaturalCoreState(
                problem.coordinate, layout, theta, problem.z_data_bounds
            )
            values, means, covariances = problem.row_statistics(
                state, layout, moments=True, distinct=self._distinct
            )
            mu, fisher = _model_first_means_and_fisher(state)
            log_values.append(values)
            centered.append(means - mu)
            within.append(covariances)
            fishers.append(0.5 * (fisher + fisher.T))
        log_likelihood, responsibility = _posterior(log_values, log_weights, self._row_weights)
        w = self._row_weights
        mass = w @ responsibility
        pi = np.exp(log_weights)
        gradient, missing = joint_information(
            responsibility,
            np.ascontiguousarray(np.hstack(centered), dtype=np.float64),
            within,
            np.ascontiguousarray(self.offsets, dtype=np.intp),
            pi,
            np.ascontiguousarray(w, dtype=np.float64),
        )

        complete = np.zeros((n_total, n_total), dtype=np.float64)
        for k in range(k_count):
            block = slice(self.offsets[k], self.offsets[k + 1])
            complete[block, block] = mass[k] * fishers[k]
        logits = slice(self.offsets[-1], n_total)
        complete[logits, logits] = np.diag(pi[:-1]) - np.outer(pi[:-1], pi[:-1])
        complete = 0.5 * (complete + complete.T)
        metric, smallest = _safeguarded_metric(complete, missing)
        return _NaturalIntervalEvaluation(
            nll=-log_likelihood,
            gradient=gradient,
            hessian=metric,
            observed_hessian=complete - missing,
            fisher=complete,
            missing_information=missing,
            smallest_curvature=smallest,
        )


def _joint_representation(representations, n_free, /):
    """Return the product of component cone descriptions plus free coordinates.

    Parameters
    ----------
    representations : sequence of _ConicRepresentation
        Component descriptions, in joint-variable order.
    n_free : int
        Trailing unconstrained coordinates (the mixture logits).
    """
    rows = [rep.n_rows for rep in representations]
    cols = [rep.b_matrix.shape[1] for rep in representations]
    row_offsets = np.concatenate(([0], np.cumsum(rows)))
    col_offsets = np.concatenate(([0], np.cumsum(cols)))
    b = np.zeros((row_offsets[-1], col_offsets[-1] + n_free), dtype=np.float64)
    matrices = []
    for k, rep in enumerate(representations):
        b[row_offsets[k]: row_offsets[k + 1], col_offsets[k]: col_offsets[k + 1]] = rep.b_matrix
        for block in rep.row_matrices:
            padded = np.zeros((row_offsets[-1],) + block.shape[1:], dtype=np.float64)
            padded[row_offsets[k]: row_offsets[k + 1]] = block
            matrices.append(padded)
    return _ConicRepresentation(
        b_matrix=b,
        row_matrices=tuple(matrices),
        reference_dual=np.concatenate([rep.reference_dual for rep in representations]),
        row_degrees=np.concatenate([np.asarray(rep.row_degrees) for rep in representations]),
    )


def _face_representation(layout, result, /):
    """Return the exact-face description a component result lies on.

    Parameters
    ----------
    layout : _NaturalLayout
        Component layout.
    result : _ConicNewtonResult
        Component M-step result.
    """
    return _support_representation(
        layout,
        int(result.effective_curvature_degree),
        bool(result.lower_amplitude_active),
        bool(result.upper_amplitude_active),
    )


def _evaluated_em_state(problems, layouts, results, log_weights, observation_weights, /):
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
        problems, layouts, [r.params for r in results], log_weights, observation_weights,
        compact=compact,
    )
    return _EMState(
        tuple(results),
        np.asarray(log_weights, dtype=np.float64),
        float(log_likelihood),
        responsibilities,
    )


def _em_map(state, problems, observation_weights, /, **options):
    """Apply one EM map using the posterior cached on ``state``.

    Parameters
    ----------
    state : _EMState
        Evaluated input iterate.
    problems : sequence of _ComponentProblem
        Component problems.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    **options : dict
        Passed to every M-step solve.

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
        problems and problems[0].intervals and problems[0].distinct_rows is not None
        and responsibilities.shape[0] == problems[0].distinct_rows[0].shape[0]
    )
    if problems and not problems[0].intervals:
        rows = problems[0].rows
        _check_point_estimability(rows, observation_weights, responsibilities)
    updated = []
    if compact:
        grouped_observation_weights, _ = problems[0]._grouped_observation_moments(
            observation_weights
        )
        for k, problem in enumerate(problems):
            _, result = problem.fit_compact(
                responsibilities[:, k], observation_weights, state.results[k], **options
            )
            updated.append(result)
        mass = grouped_observation_weights @ responsibilities
    else:
        for k, problem in enumerate(problems):
            _, result = problem.fit(
                observation_weights * responsibilities[:, k], state.results[k], **options
            )
            updated.append(result)
        mass = observation_weights @ responsibilities
    return tuple(updated), _log_mixture_weights(mass)


def _stack(results, log_weights, /):
    """Stack component parameters and log weights into one vector.

    Parameters
    ----------
    results : sequence of _ConicNewtonResult
        Component results.
    log_weights : numpy.ndarray
        Log mixture weights.
    """
    return np.concatenate(
        [np.asarray(r.params, dtype=np.float64) for r in results] + [log_weights]
    )


def _unstack(vector, reference, layouts, /):
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
    """
    offset = 0
    results = []
    for result, layout in zip(reference, layouts, strict=True):
        params = np.asarray(vector[offset: offset + layout.n_params], dtype=np.float64)
        offset += layout.n_params
        for index in (layout.lower_a_index, layout.upper_a_index):
            if index is not None and params[index] < 0.0:
                return None
        if not _certify(layout, params).feasible:
            return None
        results.append(replace(result, params=params))
    log_weights = np.asarray(vector[offset:], dtype=np.float64)
    return results, log_weights - _logsumexp(log_weights)


def _em_phase(state, problems, layouts, observation_weights, /, *, max_steps, tolerance,
              accelerate, history, **options):
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
    **options : dict
        Passed to every M-step.

    Returns
    -------
    state : _EMState
        Evaluated point at the end of the phase.
    steps : int
        EM map evaluations used.
    """
    def em(point_state):
        point_results, point_log_weights = _em_map(
            point_state, problems, observation_weights, **options,
        )
        return _evaluated_em_state(
            problems, layouts, point_results, point_log_weights, observation_weights
        )

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
            x0 = _stack(state.results, state.log_weights)
            step = _stack(state1.results, state1.log_weights) - x0
            curvature = _stack(state2.results, state2.log_weights) - 2.0 * (x0 + step) + x0
            norm = float(np.linalg.norm(curvature))
            alpha = -float(np.linalg.norm(step)) / norm if norm > 0.0 else -1.0
            for _ in range(4):
                if alpha > -1.5:
                    break
                point = _unstack(
                    x0 - 2.0 * alpha * step + alpha * alpha * curvature,
                    state2.results, layouts,
                )
                if point is not None:
                    try:
                        point_state = _evaluated_em_state(
                            problems, layouts, point[0], point[1], observation_weights
                        )
                        trial = em(point_state)
                        steps += 1
                    except NUMERIC_FAILURES as exc:
                        _reraise_if_debug(exc, "SQUAREM stabilization step", routine=True)
                        trial = None
                    if trial is not None and trial.log_likelihood >= new.log_likelihood:
                        new = trial
                        break
                alpha = 0.5 * (alpha - 1.0)
        increase = new.log_likelihood - state.log_likelihood
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
    objective : _JointMixtureObjective
        The joint objective (to split the run's parameters).
    """
    results, log_weights = state.results, state.log_weights
    objective = _JointMixtureObjective(problems, layouts, observation_weights)
    representation = _joint_representation(
        [_face_representation(layout, r) for layout, r in zip(layouts, results, strict=True)],
        len(results) - 1,
    )
    x0 = objective.join([r.params for r in results], log_weights)
    blocks = tuple(block for r in results for block in r.blocks)
    try:
        evaluation = objective(x0)
        run = _newton_on_representation(
            objective, representation, x0, blocks, evaluation, options, 0
        )
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "joint mixture polish", routine=True)
        return None, objective
    return run, objective


def _degree_policies(degree, count, /):
    """Return one fixed degree or ``"auto"`` policy per component.

    Parameters
    ----------
    degree : int, str, or sequence
        Shared or per-component degree policy.
    count : int
        Number of mixture components.

    Returns
    -------
    tuple
        One normalized degree policy per component.
    """
    if isinstance(degree, str):
        if degree.lower() != "auto":
            raise ValueError("degree must be an integer, a sequence, or 'auto'")
        return ("auto",) * int(count)
    if np.isscalar(degree):
        return (int(degree),) * int(count)
    values = tuple(degree)
    if len(values) != int(count):
        raise ValueError("per-component degree policies must match the component count")
    out = []
    for value in values:
        if isinstance(value, str):
            if value.lower() != "auto":
                raise ValueError("per-component degree policy must be an integer or 'auto'")
            out.append("auto")
        else:
            out.append(int(value))
    return tuple(out)


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
    ValueError
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
            raise ValueError(
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
        if n_eff <= int(component.layout.n_params):
            return True
    return False


def _policy_initial_components(
    support, rows, policies, lower, upper, observation_weights, responsibilities, /,
    *, degree_config=None, **options,
):
    """Run the first M-step for heterogeneous fixed/automatic degree policies.

    Parameters
    ----------
    support : tuple of (float, float)
        Shared user-coordinate support.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Point or interval observations.
    policies : sequence
        One fixed degree or ``"auto"`` policy per component.
    lower, upper : bool
        Boundary-amplitude flags.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Initial component responsibilities.
    degree_config : _DegreeSelectionConfig or None, optional
        Omitted-information selection policy.
    **options : dict
        Natural conic solver options.

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
    for k, policy in enumerate(policies):
        component_weights = observation_weights * responsibilities[:, k]
        if policy == "auto":
            if rows.shape[1] == 1:
                objective, result = _fit_natural_conic_points_auto(
                    support, rows[:, 0], lower, upper, component_weights,
                    degree_config=degree_config, **options,
                )
            else:
                objective, result = _fit_natural_conic_intervals_auto(
                    support, rows, lower, upper, component_weights,
                    degree_config=degree_config, **options,
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
            objective, result = problem.fit(component_weights, None, **options)
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
    em_tolerance=1e-4,
    max_em_steps=20,
    max_rounds=4,
    accelerate=True,
    initialization="given",
    polish=True,
    tolerance=1e-12,
    certified_tolerance=1e-10,
    accuracy_floor=1e-7,
    degree_config=None,
    _continuation=None,
    _return_continuation=False,
    **options,
):
    """Fit a mixture from initial responsibilities: EM, polish, verify.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Point samples or interval rows.
    degree : int, sequence of int, or ``"auto"``
        Shared polynomial degree, one locked degree per component, or
        omitted-information selection on the first M-step.
    lower, upper : bool
        Boundary amplitude flags.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Initial responsibilities.
    em_tolerance : float, optional
        Relative log-likelihood increase that ends an EM phase.
    max_em_steps : int, optional
        EM map evaluation limit per phase.
    max_rounds : int, optional
        EM-polish-verify rounds.
    accelerate : bool, optional
        Use SQUAREM.
    initialization : str, optional
        Name recorded in the result.
    polish : bool, optional
        ``False`` runs one EM phase only (status ``em_only``), as the
        intermediate rungs of a degree ladder do.
    tolerance, certified_tolerance, accuracy_floor : float, optional
        Newton tolerances of the polish (see ``_solve_natural_conic``).
    degree_config : _DegreeSelectionConfig or None, optional
        Omitted-information policy when ``degree="auto"``.
    _continuation : _EMContinuation or None, optional
        Internal explored state to continue directly into polishing.
    _return_continuation : bool, optional
        Internally return the reusable EM continuation with the fit.
    **options : dict
        Passed to every M-step solve.

    Returns
    -------
    _NaturalMixtureFit or tuple
        Fitted mixture, or ``(fit, continuation)`` when
        ``_return_continuation`` is true.
    """
    w = observation_weights
    if _continuation is None:
        r = np.asarray(responsibilities, dtype=np.float64)
        k_count = r.shape[1]
        _check_point_estimability(rows, w, r)
        policies = _degree_policies(degree, k_count)
        if any(policy == "auto" for policy in policies):
            problems, layouts, results = _policy_initial_components(
                support, rows, policies, lower, upper, w, r,
                degree_config=degree_config, **options,
            )
        else:
            degrees = tuple(int(value) for value in policies)
            problems = [
                _ComponentProblem(support, rows, degrees[k], lower, upper, w * r[:, k])
                for k in range(k_count)
            ]
            results = []
            layouts = []
            for k, problem in enumerate(problems):
                objective, result = problem.fit(w * r[:, k], None, **options)
                results.append(result)
                layouts.append(objective.layout)
        log_weights = _log_mixture_weights(w @ r)
        state = _evaluated_em_state(
            problems, layouts, results, log_weights, w
        )
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
        need_em = False
    newton = _NewtonOptions(
        tolerance=float(tolerance),
        certified_tolerance=float(certified_tolerance),
        accuracy_floor=float(accuracy_floor),
        max_iterations=60,
        armijo=1e-4,
        backtrack=0.5,
        max_line_search=40,
    )

    polish_iterations = 0
    status = "em_only"
    bound = np.inf
    while True:
        if need_em:
            if rounds >= int(max_rounds):
                break
            rounds += 1
            state, steps = _em_phase(
                state, problems, layouts, w,
                max_steps=max_em_steps, tolerance=em_tolerance, accelerate=accelerate,
                history=history, **options,
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
            # The polish never raises the NLL above its start; guard anyway.
            status = "em_only"
            break
        polished = [replace(res, params=p) for res, p in zip(state.results, params, strict=True)]
        history.append(polished_ll)
        _, polished_posterior = _e_step(
            problems, layouts, [x.params for x in polished], log_weights, w
        )
        state = _EMState(
            tuple(polished), np.asarray(log_weights, dtype=np.float64),
            polished_ll, polished_posterior,
        )
        status, bound = run.status, float(run.decrease_bound)
        if status not in _CONVERGED:
            need_em = True
            continue
        # Verification: one EM step from the polished point.
        verified, verified_log_weights = _em_map(
            state, problems, w, **options
        )
        em_steps += 1
        verified_state = _evaluated_em_state(
            problems, layouts, verified, verified_log_weights, w
        )
        scale = max(1.0, abs(polished_ll))
        if verified_state.log_likelihood - polished_ll <= max(
            bound, certified_tolerance * scale
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
    paths=(("ladder", "raw"), ("ladder", "sharpened"), ("direct", "raw"), ("direct", "sharpened")),
    finalists=2,
    degree_config=None,
    **options,
):
    """Fit a mixture of natural log-concave components.

    The observed mixture likelihood has many stationary points and each fit
    is only certified stationary, so this is a multi-start.  Every
    initialization candidate (``mixture._initial_responsibility_candidates``)
    is explored along every path by EM alone; the ``finalists`` best
    explorations are then run to certified convergence, and the best is
    returned.  Explorations and fits are ranked by log likelihood, or by BIC
    when automatic degrees let them lock different degrees.  A resolved KDE
    valley is the only candidate; when its result is doubtful
    (``_needs_wider_search``) the GMM and nested-scale candidates are searched
    too.  Among finished fits a sound one (certified, every component
    determined by more effective observations than it has parameters) is
    preferred to an unsound one, then the higher score.

    A path is a pair ``(schedule, treatment)``.  The ``ladder`` schedule
    climbs the admissible degree ladder: lower-degree mixtures without
    boundary terms are fitted first and their posteriors initialize the next
    rung, so smoother landscapes steer the fit; ``direct`` starts at the
    target degree.  The ``sharpened`` treatment
    squares and renormalizes the responsibilities before each rung; ``raw``
    does not.  No path dominates the others.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support shared by every component.
    samples : array_like, shape (R,), (R, 1) or (R, 2)
        Point samples or interval rows in user coordinates.
    n_components : int
        Number of components.
    poly_degree : int, ``"auto"``, or sequence
        Shared polynomial degree, per-component omitted-information selection,
        or one integer/``"auto"`` policy per component.  Automatic choices
        are locked after the first M-step.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Boundary amplitude flags.
    weights : array_like or None, optional
        Nonnegative observation weights.
    rng : int or numpy.random.Generator, optional
        Randomness for the initialization candidates.
    responsibilities : numpy.ndarray or None, optional
        Explicit initial responsibilities; replaces the candidates.
    paths : sequence of (str, str), optional
        ``(schedule, treatment)`` pairs explored from every candidate.
    finalists : int, optional
        Explorations run to certified convergence.
    degree_config : _DegreeSelectionConfig or None, optional
        Omitted-information policy for ``poly_degree="auto"``.
    **options : dict
        Passed to ``_run_natural_em`` for the finalists.

    Returns
    -------
    _NaturalMixtureFit
        Best fitted mixture.

    Raises
    ------
    RuntimeError
        If no initialization produced a fit.
    """
    rows = np.asarray(samples, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[:, None]
    if rows.ndim != 2 or rows.shape[1] not in (1, 2):
        raise ValueError("samples must have shape (R,), (R, 1) or (R, 2)")
    w = _normalized_observation_weights(rows.shape[0], weights)
    if responsibilities is not None:
        candidates = [("given", np.asarray(responsibilities, dtype=np.float64), None)]
    else:
        representatives = (
            rows[:, 0] if rows.shape[1] == 1 else _interval_initial_representatives(rows, support)
        )
        generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        candidates = _initial_responsibility_candidates(
            representatives, int(n_components), generator, weights=weights
        )
    policies = _degree_policies(poly_degree, int(n_components))
    has_auto_degree = any(policy == "auto" for policy in policies)
    uniform_fixed = (
        not has_auto_degree
        and len({int(policy) for policy in policies}) == 1
    )
    degree = int(policies[0]) if uniform_fixed else policies
    lower, upper = bool(allow_lower_boundary), bool(allow_upper_boundary)
    rungs = (
        [] if not uniform_fixed else
        [int(d) for d in _admissible_degrees(support, degree) if int(d) < degree]
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
        size = sum(int(component.layout.n_params) for component in fit.components)
        return float(fit.log_likelihood) - bic_penalty * size

    def rank(fit):
        # A sound fit (certified, every component determined by more
        # effective observations than it has parameters) outranks any
        # unsound one; the score decides within each class.  Otherwise an
        # uncertified component on a handful of extreme points can win on
        # likelihood over a certified ordinary fit.
        return (not _needs_wider_search(fit, rows, w), score(fit))

    # Exploration uses the default EM/M-step controls.  Its terminal state is
    # therefore exactly the state a default finalist rerun would reproduce.
    # Polish-only tolerances do not affect that state and are safe to change
    # when continuing.  For any custom EM or M-step option, keep the previous
    # restart behavior rather than silently changing its semantics.
    reusable_options = {"tolerance", "certified_tolerance", "accuracy_floor"}
    reuse_exploration = set(options).issubset(reusable_options)
    active_paths = list(paths)
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
                    if schedule == "ladder":
                        for rung in rungs:
                            posterior = _run_natural_em(
                                support, rows, rung, False, False, w,
                                prepare(posterior, treatment), polish=False,
                            ).responsibilities
                    start = prepare(posterior, treatment)
                    if reuse_exploration:
                        exploration, continuation = _run_natural_em(
                            support, rows, degree, lower, upper, w, start, polish=False,
                            degree_config=degree_config, _return_continuation=True,
                        )
                    else:
                        exploration = _run_natural_em(
                            support, rows, degree, lower, upper, w, start, polish=False,
                            degree_config=degree_config,
                        )
                        continuation = None
                except (*NUMERIC_FAILURES, ValueError) as exc:
                    _reraise_if_debug(exc, f"natural mixture EM from {label}", routine=True)
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
        for _, label, start, locked_degrees, continuation in explored[: max(1, int(finalists))]:
            final_degree = locked_degrees if has_auto_degree else degree
            try:
                fit = _run_natural_em(
                    support, rows, final_degree, lower, upper, w, start,
                    initialization=label, _continuation=continuation, **options,
                )
            except (*NUMERIC_FAILURES, ValueError) as exc:
                _reraise_if_debug(exc, f"natural mixture fit from {label}", routine=True)
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
            representatives, int(n_components), generator, weights=weights,
            include_valley=False,
        )
        other, other_failure = search(alternatives)
        if other is not None and (best is None or rank(other) > rank(best)):
            best = other
        failure = failure if failure is not None else other_failure
    if best is None:
        if failure is not None and "point-mixture component is not estimable" in str(failure):
            raise failure
        raise RuntimeError("no mixture initialization produced a fit") from failure
    return best
