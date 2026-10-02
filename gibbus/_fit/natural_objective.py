"""Point and interval objectives in affine natural coordinates.

This module builds the fitting coordinate, model specification and observation
summaries for one fit, evaluates the likelihood geometry of
:mod:`.objective` on natural states, and hands the statistics-driven
objectives to :mod:`.conic_newton`.  It also hosts the fixed-degree and
automatic-degree fit drivers.

Point likelihoods are convex in natural coordinates.  Interval-censored
likelihoods are not: with ``A(theta)`` the log partition function and
``A_r(theta)`` the log mass of row ``r``, the NLL is ``A - sum_r w_r A_r``, a
difference of convex functions.  Its Hessian is ``F - C``: the model Fisher
matrix minus the weighted conditional covariance of the potential partials
within the censoring rows (the missing information).  The EM surrogate at the
current point has the same gradient and Hessian ``F``, and majorizes the NLL.
Interval objectives hand the solver the observed Hessian itself wherever it is
positive definite relative to ``F`` (exact Newton), and otherwise the
saddle-free metric: the observed Hessian with its negative curvature
reflected, ``F V |Lambda| V^T F`` from the generalized eigenproblem
``(F - C) v = lambda F v`` with ``|lambda|`` floored at ``_METRIC_FLOOR``.
Reflecting (rather than damping) negative curvature makes Newton steps move
away from saddle points instead of stalling near them.  The Armijo search runs
on the true observed NLL, so every step descends.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from .._defaults import (
    AUTO_POLY_DEGREE_MAX,
    AUTO_POLY_DEGREE_MIN,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._model.coords import (
    _build_fit_coordinate,
    _build_interval_fit_coordinate,
    _normalised_nonnegative_weights,
    _safe_scaled_difference,
)
from .._model.natural_state import _layout_numerics, _NaturalCoreState
from .._model.spec import _build_model_spec, _ModelSpec
from .._observations.empirical import _EmpiricalStats, _uniform_interval_empirical_stats
from .._observations.intervals import (
    _build_interval_observations,
    _IntervalObservations,
    _row_grouping,
)
from .._observations.points import _PointObservations
from ._mixture_kernels import empirical_point_stats
from .conic_newton import _interior_start, _solve_natural_conic
from .degree import (
    _DegreeSelectionConfig,
    _interval_omitted_statistic_diagnostic,
    _omitted_statistic_diagnostic,
    _probe_orders_for_degree,
)
from .inputs import _admissible_degrees
from .objective import (
    _evaluate_interval_objective,
    _evaluate_point_objective,
    _ObjectiveEvaluation,
)

# Relative eigenvalues (against the Fisher matrix) of the observed Hessian at
# or above _NEWTON_FLOOR make it the metric (exact Newton, however small the
# curvature); otherwise every eigenvalue is reflected and floored at
# _METRIC_FLOOR.
_NEWTON_FLOOR = 1e-8
_METRIC_FLOOR = 1e-3


_POINT_STATS_ERRORS = {
    1: (ValueError, "weights must be finite and non-negative"),
    2: (ValueError, "total observation weight must be positive"),
    3: (FloatingPointError, "empirical power moment is non-finite"),
    4: (ValueError, "active lower log-boundary term requires every positive-weight point above L"),
    5: (ValueError, "active upper log-boundary term requires every positive-weight point below U"),
    6: (FloatingPointError, "boundary-log empirical statistic is non-finite"),
}


def _natural_point_stats(
    points, weights, max_order, support, has_lower, has_upper, /,
    *, lower_distance=None, upper_distance=None,
):
    """Build the point-data sufficient statistics.

    Parameters
    ----------
    points : array_like, shape (n,)
        Canonical point observations.
    weights : array_like or None
        Nonnegative weights (``None``: equal).
    max_order : int
        Highest power moment.
    support : tuple of (float, float)
        Canonical support.
    has_lower, has_upper : bool
        Whether the boundary-log statistics are part of the basis.
    lower_distance, upper_distance : array_like or None, optional
        Preserved canonical distances of physical point observations from
        finite support boundaries.  These prevent affine canonicalization
        from turning a positive sub-ULP physical distance into an apparent
        exact-boundary observation.

    Returns
    -------
    _EmpiricalStats
        The point-data summary.
    """
    z = np.ascontiguousarray(points, dtype=np.float64).reshape(-1)
    if z.size < 1:
        raise ValueError("at least one point observation is required")
    if not np.all(np.isfinite(z)):
        raise ValueError("point observations must be finite")
    lower, upper = map(float, support)
    if not lower < upper:
        raise ValueError("support must satisfy lower < upper")
    if (np.isfinite(lower) and np.any(z < lower)) or (np.isfinite(upper) and np.any(z > upper)):
        raise ValueError("point observations must lie within the support")
    if has_lower and not np.isfinite(lower):
        raise ValueError("active lower log-boundary term requires finite L")
    if has_upper and not np.isfinite(upper):
        raise ValueError("active upper log-boundary term requires finite U")
    # Power moments and participation are reduced in the compiled kernel.
    # Boundary-log statistics are handled below when preserved physical
    # distances are available, because affine canonicalization can round a
    # truly interior point onto a finite canonical endpoint.
    use_preserved_lower = bool(has_lower and lower_distance is not None)
    use_preserved_upper = bool(has_upper and upper_distance is not None)
    status, moments, participation, boundary, total, n_eff = empirical_point_stats(
        z, weights, int(max_order), lower, upper,
        bool(has_lower and not use_preserved_lower),
        bool(has_upper and not use_preserved_upper),
    )
    if status:
        error, message = _POINT_STATS_ERRORS[status]
        raise error(message)

    if use_preserved_lower or use_preserved_upper:
        if weights is None:
            wn = np.full(z.size, 1.0 / z.size, dtype=np.float64)
        else:
            wn = _normalised_nonnegative_weights(weights, z.size)
            if wn is None:
                raise ValueError("weights must be finite and non-negative with positive total")

        boundary = np.asarray(boundary, dtype=np.float64).copy()
        for slot, enabled, distances, fallback, message in (
            (0, use_preserved_lower, lower_distance, z - lower,
             _POINT_STATS_ERRORS[4][1]),
            (1, use_preserved_upper, upper_distance, upper - z,
             _POINT_STATS_ERRORS[5][1]),
        ):
            if not enabled:
                continue
            d = np.asarray(distances, dtype=np.float64).reshape(-1)
            if d.size != z.size:
                raise ValueError("preserved boundary distances must match points")
            d = np.where(np.isfinite(d), d, fallback)
            positive = wn > 0.0
            if np.any(~np.isfinite(d[positive])) or np.any(d[positive] <= 0.0):
                raise ValueError(message)
            with np.errstate(divide="ignore", invalid="ignore"):
                value = float(np.dot(wn[positive], -np.log(d[positive])))
            if not np.isfinite(value):
                raise FloatingPointError(_POINT_STATS_ERRORS[6][1])
            boundary[slot] = value

    return _EmpiricalStats(
        moments=moments,
        boundary_log=boundary,
        support=(lower, upper),
        total_weight=float(total),
        effective_n=float(n_eff),
        n_observations=int(z.size),
        moment_effective_n=participation,
    )


@dataclass(frozen=True)
class _NaturalDegreeDiagnosticFit:
    """Adapter exposing a fitted natural state to degree diagnostics.

    Parameters
    ----------
    spec : _ModelSpec
        Fitted model specification.
    observations : _PointObservations or _IntervalObservations
        Canonical observations used by the fitted objective.
    state : _NaturalCoreState
        Normalized fitted natural state.
    """

    spec: object
    observations: object
    state: object


def _degree_diagnostic_fit(objective, result, /):
    """Return the common diagnostic view of one fitted natural objective.

    Parameters
    ----------
    objective : _NaturalPointObjectiveFunction or _NaturalIntervalObjectiveFunction
        Natural objective used by the solve.
    result : _ConicNewtonResult
        Fitted natural parameters.

    Returns
    -------
    _NaturalDegreeDiagnosticFit
        View consumed by the omitted-information degree diagnostics.
    """
    return _NaturalDegreeDiagnosticFit(
        spec=objective.spec,
        observations=objective.observations,
        state=objective.build_state(result.params),
    )


class _NaturalPointObjectiveFunction:
    """Callable natural-parameter point objective."""

    def __init__(self, spec, observations, z_data_bounds, /):
        """Bind a model spec to fixed point statistics.

        Parameters
        ----------
        spec : _ModelSpec
            Model specification (coordinate, degree, boundary bases).
        observations : _PointObservations
            Fixed sufficient-statistic observation provider.
        z_data_bounds : tuple of (float, float)
            Canonical data range used by quadrature windowing.
        """
        if not isinstance(spec, _ModelSpec):
            raise TypeError("spec must be a _ModelSpec")
        if not isinstance(observations, _PointObservations):
            raise TypeError("observations must be a _PointObservations")
        bounds = np.asarray(z_data_bounds, dtype=np.float64).reshape(-1)
        if bounds.size != 2 or np.any(np.isnan(bounds)) or bounds[0] > bounds[1]:
            raise ValueError("z_data_bounds must be an ordered pair")

        self.spec = spec
        self.observations = observations
        self.z_data_bounds = (float(bounds[0]), float(bounds[1]))
        self.layout = spec.layout
        self._empirical_means = None
        self._coordinate_constant = float(np.log(spec.coordinate.scale))

    def _compiled_point_newton_inputs(self, /):
        """Return fixed arrays consumed by the fused compiled Newton loop.

        The initial Python objective evaluation populates ``_empirical_means``;
        after that, point-data Newton trials depend only on these fixed
        statistics and the shared state-numerics constants.
        """
        if self._empirical_means is None:
            return None
        from .._defaults import QUAD_EPSABS, QUAD_EPSREL, QUAD_LIMIT
        from .._model.natural_state import _MODE_CONTROLS

        numerics = _layout_numerics(self.layout)
        return (
            numerics.support,
            np.asarray(self.z_data_bounds, dtype=np.float64),
            self.layout.lower_a_index is not None,
            self.layout.upper_a_index is not None,
            numerics.kinds,
            numerics.lengths,
            numerics.coefficients,
            _MODE_CONTROLS,
            np.asarray(self._empirical_means, dtype=np.float64),
            self._coordinate_constant,
            self.layout.curvature_degree,
            -1 if self.layout.lower_a_index is None else self.layout.lower_a_index,
            -1 if self.layout.upper_a_index is None else self.layout.upper_a_index,
            QUAD_EPSABS,
            QUAD_EPSREL,
            QUAD_LIMIT,
        )

    def build_state(self, params, /):
        """Build and normalize one natural-coordinate candidate state.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        """
        return _NaturalCoreState(
            self.spec.coordinate,
            self.layout,
            params,
            self.z_data_bounds,
        )

    def __call__(self, params, /):
        """Return exact point NLL, gradient, and PSD Fisher Hessian.

        The potential is linear in the parameters, so the Hessian is the
        Fisher matrix and the empirical partial means are fixed; states from
        the compiled traversal skip the generic evaluator.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        """
        state = self.build_state(params)
        cached = state._first_means_fisher
        if cached is None:
            return _evaluate_point_objective(state, self.observations)
        if not (np.isfinite(state.log_Z) and state.Z > 0.0):
            raise RuntimeError("candidate state is not normalizable")
        if self._empirical_means is None:
            self._empirical_means = self.observations.first_expectations(state.partials)
            self._zero_missing = np.zeros((state.p.size, state.p.size))
        means, fisher = cached
        # q = theta . h exactly (the potential is linear with no constant),
        # so E_hat[q] is the parameter/statistic contraction; the compiled
        # Fisher matrix is exactly symmetric.
        nll = float(self._empirical_means @ state.p) + state.log_Z + self._coordinate_constant
        return _ObjectiveEvaluation(
            nll=nll,
            gradient=self._empirical_means - means,
            hessian=fisher,
            fisher=fisher,
            missing_information=self._zero_missing,
            model_partial_means=means,
        )




def _preserved_point_boundary_distances(coordinate, points, /):
    """Return canonical point-to-boundary distances before affine rounding.

    Distances are computed from the original physical coordinates with the
    same overflow-safe scaled subtraction used by interval observations.
    ``NaN`` marks an infinite side.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed user-to-canonical affine fitting coordinate.
    points : array_like
        Physical point observations.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper canonical boundary distances; unavailable infinite
        sides are filled with ``NaN``.
    """
    x = np.asarray(points, dtype=np.float64).reshape(-1)
    lower_distance = np.full(x.size, np.nan, dtype=np.float64)
    upper_distance = np.full(x.size, np.nan, dtype=np.float64)
    physical = tuple(map(float, coordinate.physical_support))
    canonical = tuple(map(float, coordinate.canonical_support))
    direction = float(coordinate.direction)
    scale = float(coordinate.scale)
    if np.isfinite(canonical[0]):
        endpoint = physical[0] if direction > 0.0 else physical[1]
        lower_distance = direction * _safe_scaled_difference(x, endpoint, scale)
    if np.isfinite(canonical[1]):
        endpoint = physical[1] if direction > 0.0 else physical[0]
        upper_distance = -direction * _safe_scaled_difference(x, endpoint, scale)
    return lower_distance, upper_distance

def _prepare_natural_point_objective(
    support,
    point_samples,
    poly_degree,
    allow_lower_boundary,
    allow_upper_boundary,
    weights,
    /,
    *,
    moment_order=None,
):
    """Build a natural point objective from samples.

    Builds the data-centred fitting coordinate, the model specification and
    the empirical sufficient statistics; the objective depends on the data
    only through those statistics.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    point_samples : array_like, shape (N,)
        Finite point observations in user coordinates.
    poly_degree : int
        Requested maximum polynomial degree.
    allow_lower_boundary, allow_upper_boundary : bool
        Whether the finite physical endpoints may carry logarithmic
        amplitudes.
    weights : array_like or None
        Nonnegative observation weights.
    moment_order : int or None, optional
        Highest empirical power moment to cache.  The default stores at least
        twice the effective fitted degree; automatic degree selection raises
        this to cover the next omitted-statistic probe block.

    Returns
    -------
    _NaturalPointObjectiveFunction
        Statistics-driven natural objective.

    Raises
    ------
    ValueError
        If fewer than two finite samples are given.
    """
    x = np.asarray(point_samples, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("point_samples must contain at least two finite values")
    coordinate = _build_fit_coordinate(support, x, weights, None)
    spec = _build_model_spec(
        coordinate,
        int(poly_degree),
        bool(allow_lower_boundary),
        bool(allow_upper_boundary),
    )
    z = coordinate.to_canonical(x)
    default_order = max(4, 2 * int(spec.effective_poly_degree))
    order = default_order if moment_order is None else int(moment_order)
    if order < spec.effective_poly_degree:
        raise ValueError("moment_order must cover the effective polynomial degree")
    lower_distance, upper_distance = _preserved_point_boundary_distances(
        coordinate, x
    )
    stats = _natural_point_stats(
        z,
        weights,
        order,
        spec.support,
        spec.canonical_lower_a_index is not None,
        spec.canonical_upper_a_index is not None,
        lower_distance=lower_distance,
        upper_distance=upper_distance,
    )
    z_bounds = (float(np.min(z)), float(np.max(z)))
    return _NaturalPointObjectiveFunction(spec, _PointObservations(stats), z_bounds)


def _fit_natural_conic_points(
    support,
    point_samples,
    poly_degree,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    weights=None,
    /,
    *,
    moment_order=None,
    **options,
):
    """Fit point data with the natural conic Newton solver (default settings).

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    point_samples : array_like, shape (N,)
        Finite point observations in user coordinates.
    poly_degree : int
        Requested maximum polynomial degree.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Whether the finite physical endpoints may carry logarithmic
        amplitudes.
    weights : array_like or None, optional
        Nonnegative observation weights.
    moment_order : int or None, optional
        Highest empirical power moment to retain for later diagnostics.
    **options : dict
        Passed to :func:`._solve_natural_conic`.

    Returns
    -------
    objective : _NaturalPointObjectiveFunction
        The fitted objective; ``objective.build_state(result.params)`` gives
        the fitted natural state.
    result : _ConicNewtonResult
        Solver outcome.
    """
    objective = _prepare_natural_point_objective(
        support,
        point_samples,
        poly_degree,
        allow_lower_boundary,
        allow_upper_boundary,
        weights,
        moment_order=moment_order,
    )
    return objective, _solve_natural_conic(objective, **options)


def _fit_natural_conic_points_auto(
    support,
    point_samples,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    weights=None,
    /,
    *,
    degree_config=None,
    **options,
):
    """Select point-data degree with the omitted-information criterion.

    Start at the lowest admissible degree, examine the next
    reliable omitted power block, and increase capacity only when its
    efficient score is statistically resolvable.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    point_samples : array_like, shape (N,)
        Finite point observations.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Whether finite physical endpoints may use logarithmic amplitudes.
    weights : array_like or None, optional
        Nonnegative observation weights.
    degree_config : _DegreeSelectionConfig or None, optional
        Omitted-statistic selection policy.
    **options : dict
        Passed to every natural conic solve.

    Returns
    -------
    objective : _NaturalPointObjectiveFunction
        Objective at the selected degree.
    result : _ConicNewtonResult
        Natural conic fit at the selected degree.

    Raises
    ------
    RuntimeError
        If no admissible natural point model produces a usable fit.
    """
    x = np.asarray(point_samples, dtype=np.float64).reshape(-1)
    cfg = _DegreeSelectionConfig() if degree_config is None else degree_config
    degrees = [
        int(d) for d in _admissible_degrees(support, AUTO_POLY_DEGREE_MAX)
        if int(d) >= int(AUTO_POLY_DEGREE_MIN)
    ]
    if not degrees:
        degrees = [int(AUTO_POLY_DEGREE_MIN)]

    best = None
    max_degree = int(degrees[-1])
    for index, degree in enumerate(degrees):
        probe_orders = _probe_orders_for_degree(
            degree, max_degree, block_size=cfg.probe_block_size
        )
        moment_order = (
            max(2 * int(probe_orders[-1]), 2 * int(degree))
            if probe_orders else 2 * int(degree)
        )
        try:
            candidate = _fit_natural_conic_points(
                support, x, degree,
                allow_lower_boundary, allow_upper_boundary, weights,
                moment_order=moment_order,
                **options,
            )
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(
                exc, f"natural auto-degree point candidate degree={degree}",
                routine=True,
            )
            continue
        best = candidate
        if index == len(degrees) - 1 or not probe_orders:
            return candidate
        diagnostic = _omitted_statistic_diagnostic(
            _degree_diagnostic_fit(*candidate), probe_orders, config=cfg
        )
        if not diagnostic.should_expand:
            return candidate

    if best is None:
        raise RuntimeError(
            "auto poly_degree selection failed: no admissible natural point "
            "model produced a valid fit"
        )
    return best


@dataclass(frozen=True)
class _NaturalIntervalEvaluation:
    """Observed-data interval geometry plus the metric handed to the solver.

    Parameters
    ----------
    nll : float
        Observed-data negative log likelihood per unit weight.
    gradient : numpy.ndarray, shape (n,)
        Exact observed-data gradient.
    hessian : numpy.ndarray, shape (n, n)
        Positive-definite Newton metric: the observed Hessian, or its
        saddle-free reflection where it is not positive definite.
    observed_hessian : numpy.ndarray, shape (n, n)
        Exact observed-data Hessian ``F - C`` (may be indefinite).
    fisher : numpy.ndarray, shape (n, n)
        Model Fisher matrix ``F``, the Hessian of the EM surrogate.
    missing_information : numpy.ndarray, shape (n, n)
        Weighted within-row conditional covariance ``C``.
    smallest_curvature : float
        Smallest eigenvalue of the observed Hessian relative to ``F``;
        at least ``_NEWTON_FLOOR`` means the metric is exact Newton.
    """

    nll: float
    gradient: np.ndarray
    hessian: np.ndarray
    observed_hessian: np.ndarray
    fisher: np.ndarray
    missing_information: np.ndarray
    smallest_curvature: float


def _safeguarded_metric(fisher, missing, /):
    """Return ``(metric, smallest)`` for the observed Hessian ``F - C``.

    Solves ``(F - C) v = lambda F v`` (Jacobi-scaled; the eigenvalues are
    invariant under linear reparameterization).  With ``V^T F V = I`` the
    observed Hessian is ``F V Lambda V^T F``.  When every ``lambda`` is at
    least ``_NEWTON_FLOOR`` the metric is the observed Hessian exactly: near a
    strict local optimum Newton converges quadratically however much
    information is missing.  Otherwise it replaces ``Lambda`` by
    ``max(|Lambda|, _METRIC_FLOOR)``.

    Parameters
    ----------
    fisher : numpy.ndarray, shape (n, n)
        Complete-data Fisher matrix (positive definite).
    missing : numpy.ndarray, shape (n, n)
        Missing information (positive semidefinite).

    Returns
    -------
    metric : numpy.ndarray, shape (n, n)
        Positive-definite Newton metric.
    smallest : float
        Smallest relative eigenvalue of the observed Hessian (``-inf`` when
        ``F`` is numerically singular and the EM metric ``F`` is returned).
    """
    f = 0.5 * (fisher + fisher.T)
    observed = f - 0.5 * (missing + missing.T)
    diagonal = np.sqrt(np.maximum(np.diag(f), np.finfo(float).tiny))
    scale = np.outer(diagonal, diagonal)
    try:
        values, vectors = scipy.linalg.eigh(observed / scale, f / scale)
    except (np.linalg.LinAlgError, ValueError) as exc:
        # F numerically singular: fall back to the EM metric alone.
        _reraise_if_debug(exc, "interval metric eigenvalues", routine=True)
        return f, -np.inf
    if not np.all(np.isfinite(values)):
        return f, -np.inf
    smallest = float(values[0])
    if smallest >= _NEWTON_FLOOR:
        return observed, smallest
    basis = (f / scale) @ vectors
    reflected = (basis * np.maximum(np.abs(values), _METRIC_FLOOR)) @ basis.T
    metric = reflected * scale
    return 0.5 * (metric + metric.T), smallest


def _interval_nll_lower_bound(observations, /, *, max_iterations=20000, tolerance=1e-13):
    """Return a rigorous lower bound on every interval NLL for these rows.

    The censoring endpoints partition the support into atoms; any
    distribution assigns them masses ``p`` and the interval log likelihood
    ``l(p) = sum_r w_r log (A p)_r`` is concave on the simplex.  Turnbull's EM
    update climbs it, and the Frank-Wolfe duality gap
    ``max_j dl/dp_j - p . grad l = max_j cover_j - 1`` bounds the remaining
    ascent, so ``-(l(p) + gap)`` is a lower bound on the NLL of *every*
    distribution, parametric or not, at every iterate.  When a parametric fit
    reaches it, the fit is globally optimal to that accuracy even though the
    parametric likelihood is not convex.

    Parameters
    ----------
    observations : _IntervalObservations
        Canonical interval rows with normalized weights.
    max_iterations : int, optional
        Turnbull iteration limit.
    tolerance : float, optional
        Stop once the duality gap is below this value.

    Returns
    -------
    float or None
        The bound, or ``None`` when zero-width rows are present (they
        contribute densities, and the nonparametric likelihood is unbounded).
    """
    rows = np.asarray(observations.intervals, dtype=np.float64)
    weights = np.asarray(observations.weights, dtype=np.float64)
    if rows.size == 0 or np.any(rows[:, 0] == rows[:, 1]):
        return None
    lower, upper = map(float, observations.support)
    endpoints = np.unique(np.concatenate(([lower, upper], rows.reshape(-1))))
    starts = np.searchsorted(endpoints, rows[:, 0])
    ends = np.searchsorted(endpoints, rows[:, 1])
    atoms = int(endpoints.size - 1)
    mass = np.full(atoms, 1.0 / atoms, dtype=np.longdouble)
    total_weight = float(np.sum(weights))
    best = np.inf
    for _ in range(int(max_iterations)):
        prefix = np.concatenate(([0.0], np.cumsum(mass))).astype(np.longdouble)
        row_mass = prefix[ends] - prefix[starts]
        if np.any(row_mass <= 0.0):
            return None
        log_likelihood = float(np.sum(weights * np.log(row_mass)))
        coefficient = weights / row_mass
        delta = np.zeros(atoms + 1, dtype=np.longdouble)
        np.add.at(delta, starts, coefficient)
        np.add.at(delta, ends, -coefficient)
        cover = np.cumsum(delta[:-1])
        gap = float(np.max(cover)) - total_weight
        best = min(best, log_likelihood + max(gap, 0.0))
        if gap <= tolerance:
            break
        mass = mass * cover
        mass = mass / np.sum(mass)
    # Margin for the rounding of the logarithms and the weighted sum.
    return float(-best - 64.0 * np.finfo(float).eps * (1.0 + abs(best)))


class _NaturalIntervalObjectiveFunction:
    """Callable natural-parameter interval objective.

    ``nll_lower_bound`` is a rigorous lower bound on the NLL of every
    distribution for these rows (``None`` when unavailable).  Fixed-face
    Newton remains a purely local optimizer; the Python fit controller applies
    this bound afterward as a global certificate when an endpoint reaches it.
    """

    def __init__(self, spec, observations, z_data_bounds, /, *, nonparametric_bound=True):
        """Bind a model spec to fixed interval observations.

        Parameters
        ----------
        spec : _ModelSpec
            Model specification (coordinate, degree, boundary bases).
        observations : _IntervalObservations
            Canonical interval rows.
        z_data_bounds : tuple of (float, float)
            Finite canonical observation landmarks used by quadrature windowing.
        nonparametric_bound : bool, optional
            Compute ``nll_lower_bound``.  Mixture M-steps skip it: their
            components rarely saturate the data and are refit every iteration.
        """
        if not isinstance(spec, _ModelSpec):
            raise TypeError("spec must be a _ModelSpec")
        if not isinstance(observations, _IntervalObservations):
            raise TypeError("observations must be an _IntervalObservations")
        bounds = np.asarray(z_data_bounds, dtype=np.float64).reshape(-1)
        if bounds.size != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
            raise ValueError("z_data_bounds must be a finite ordered pair")
        self.spec = spec
        self.observations = observations
        self.z_data_bounds = (float(bounds[0]), float(bounds[1]))
        self.layout = spec.layout
        self.nll_lower_bound = (
            _interval_nll_lower_bound(observations) if nonparametric_bound else None
        )
        support = tuple(map(float, self.spec.support))
        intervals = np.asarray(self.observations.intervals, dtype=np.float64)
        positive_width = intervals[:, 1] > intervals[:, 0]
        finite_rows = np.all(np.isfinite(intervals), axis=1)
        boundary_touch = np.zeros(intervals.shape[0], dtype=bool)
        if np.isfinite(support[0]):
            boundary_touch |= positive_width & (intervals[:, 0] == support[0])
        if np.isfinite(support[1]):
            boundary_touch |= positive_width & (intervals[:, 1] == support[1])
        whole_support = (
            (intervals[:, 0] == support[0])
            & (intervals[:, 1] == support[1])
            & positive_width
        )
        regular_finite = finite_rows & ~boundary_touch
        adaptive = (~finite_rows | boundary_touch) & ~whole_support
        self._compiled_interval_regular_mask = regular_finite
        self._compiled_interval_adaptive_mask = adaptive
        self._compiled_interval_whole_weight = float(
            np.sum(self.observations.weights[whole_support])
        )
        # The fused interval solver is a local fixed-face optimizer.  Global
        # certification from ``nll_lower_bound`` is deliberately handled by
        # the Python controller after the local solve, so standalone and
        # mixture interval objectives share the same compiled Newton engine.
        self._compiled_interval_newton_eligible = True

    def _compiled_interval_newton_inputs(self, /):
        """Return fixed arrays for the fused mixed-geometry Newton loop.

        Standalone fixed-face solves and mixture M-steps stay in compiled code
        across ordinary finite rows, positive-width support-boundary rows, and
        one-/two-sided censoring.  Ordinary rows use the local Gauss--Legendre
        reducer; boundary/tail rows use the compiled adaptive Gauss--Kronrod
        reducer.  Whole-support rows are represented only by their aggregate
        weight.
        """
        if not self._compiled_interval_newton_eligible:
            return None
        from .._defaults import (
            INTERVAL_W_EPS_MULT,
            QUAD_EPSABS,
            QUAD_EPSREL,
            QUAD_LIMIT,
        )
        from .._model.natural_state import _MODE_CONTROLS
        from .._observations.intervals import _GL_LOG_W, _GL_X

        numerics = _layout_numerics(self.layout)
        regular = self._compiled_interval_regular_mask
        adaptive = self._compiled_interval_adaptive_mask
        return (
            numerics.support,
            np.asarray(self.z_data_bounds, dtype=np.float64),
            numerics.kinds,
            numerics.lengths,
            numerics.coefficients,
            _MODE_CONTROLS,
            np.ascontiguousarray(self.observations.intervals[regular], dtype=np.float64),
            np.ascontiguousarray(self.observations.weights[regular], dtype=np.float64),
            np.ascontiguousarray(
                self.observations.point_lower_distance[regular], dtype=np.float64
            ),
            np.ascontiguousarray(
                self.observations.point_upper_distance[regular], dtype=np.float64
            ),
            np.ascontiguousarray(self.observations.intervals[adaptive], dtype=np.float64),
            np.ascontiguousarray(self.observations.weights[adaptive], dtype=np.float64),
            self._compiled_interval_whole_weight,
            float(self.spec.coordinate.scale),
            _GL_X,
            _GL_LOG_W,
            float(INTERVAL_W_EPS_MULT),
            self.layout.curvature_degree,
            -1 if self.layout.lower_a_index is None else self.layout.lower_a_index,
            -1 if self.layout.upper_a_index is None else self.layout.upper_a_index,
            QUAD_EPSABS,
            QUAD_EPSREL,
            QUAD_LIMIT,
        )

    def build_state(self, params, /):
        """Build and normalize one natural-coordinate candidate state.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        """
        return _NaturalCoreState(
            self.spec.coordinate, self.layout, params, self.z_data_bounds
        )

    def __call__(self, params, /):
        """Return the observed NLL, gradient and safeguarded Newton metric.

        Parameters
        ----------
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        """
        evaluation = _evaluate_interval_objective(self.build_state(params), self.observations)
        fisher = evaluation.fisher
        missing = evaluation.missing_information
        metric, smallest = _safeguarded_metric(fisher, missing)
        return _NaturalIntervalEvaluation(
            nll=float(evaluation.nll),
            gradient=np.asarray(evaluation.gradient, dtype=np.float64),
            hessian=metric,
            observed_hessian=fisher - missing,
            fisher=fisher,
            missing_information=missing,
            smallest_curvature=smallest,
        )


def _prepare_natural_interval_objective(
    support,
    intervals,
    poly_degree,
    allow_lower_boundary,
    allow_upper_boundary,
    weights,
    /,
):
    """Build a natural interval objective from censoring rows.

    Finite rows use the midpoint/width fitting coordinate, rows with an
    infinite endpoint the censoring-landmark coordinate.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    intervals : array_like, shape (R, 2)
        Ordered censoring rows in user coordinates; endpoints may be infinite.
    poly_degree : int
        Requested maximum polynomial degree.
    allow_lower_boundary, allow_upper_boundary : bool
        Whether the finite physical endpoints may carry logarithmic
        amplitudes.
    weights : array_like or None
        Nonnegative observation weights.

    Returns
    -------
    _NaturalIntervalObjectiveFunction
        Observed-data natural interval objective.

    Raises
    ------
    ValueError
        If the rows are malformed or carry no finite landmark.
    """
    x = np.asarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError("intervals must have shape (R, 2)")
    if np.any(np.isnan(x)) or np.any(x[:, 0] > x[:, 1]):
        raise ValueError("intervals must be ordered and contain no NaN")
    if np.all(np.isfinite(x)):
        mid = 0.5 * (x[:, 0] + x[:, 1])
        coordinate = _build_fit_coordinate(support, mid, weights, x[:, 1] - x[:, 0])
    else:
        coordinate = _build_interval_fit_coordinate(support, x, weights)
    observations = _build_interval_observations(
        x, weights, coordinate=coordinate,
        grouping_cache={"grouping": _row_grouping(x)},
    )
    spec = _build_model_spec(
        coordinate,
        int(poly_degree),
        bool(allow_lower_boundary),
        bool(allow_upper_boundary),
    )
    finite = observations.intervals[np.isfinite(observations.intervals)]
    if finite.size == 0:
        raise ValueError("whole-support censoring contains no finite landmark")
    z_bounds = (float(np.min(finite)), float(np.max(finite)))
    return _NaturalIntervalObjectiveFunction(spec, observations, z_bounds)


def _natural_interval_start(objective, /):
    """Return a feasible, normalizable start for an interval fit.

    With only finite rows this is the natural point fit to the
    uniform-within-row pseudo-statistics (a convex fit); rows with an
    infinite endpoint admit no uniform, so the start is the generic interior
    point at the coordinate's centre and scale.

    Parameters
    ----------
    objective : _NaturalIntervalObjectiveFunction
        Interval objective to start.

    Returns
    -------
    params : numpy.ndarray, shape (n,)
        Starting natural parameters.
    blocks : tuple of numpy.ndarray or None
        Gram certificate of ``params`` when it came from a conic fit.
    """
    observations = objective.observations
    spec = objective.spec
    if observations.has_infinite_rows:
        return _interior_start(objective), None
    order = max(4, 2 * int(spec.effective_poly_degree))
    stats = _uniform_interval_empirical_stats(
        observations,
        order,
        has_lower_log=spec.canonical_lower_a_index is not None,
        has_upper_log=spec.canonical_upper_a_index is not None,
    )
    pseudo = _NaturalPointObjectiveFunction(
        spec, _PointObservations(stats), objective.z_data_bounds
    )
    result = _solve_natural_conic(pseudo, certify=False)
    return result.params, result.blocks


def _fit_natural_conic_intervals(
    support,
    intervals,
    poly_degree,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    weights=None,
    /,
    **options,
):
    """Fit interval-censored data with the natural conic Newton solver.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support.
    intervals : array_like, shape (R, 2)
        Ordered censoring rows in user coordinates; endpoints may be infinite.
    poly_degree : int
        Requested maximum polynomial degree.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Whether the finite physical endpoints may carry logarithmic
        amplitudes.
    weights : array_like or None, optional
        Nonnegative observation weights.
    **options : dict
        Passed to ``_solve_natural_conic``.

    Returns
    -------
    objective : _NaturalIntervalObjectiveFunction
        The fitted objective.
    result : _ConicNewtonResult
        Solver outcome.  The observed likelihood is not convex, so local
        Newton convergence ordinarily certifies a stationary point.  When the
        endpoint reaches the rigorous nonparametric NLL lower bound, the
        Python controller upgrades that endpoint with a global certificate.
    """
    objective = _prepare_natural_interval_objective(
        support,
        intervals,
        poly_degree,
        allow_lower_boundary,
        allow_upper_boundary,
        weights,
    )
    if "initial" not in options:
        start, blocks = _natural_interval_start(objective)
        options["initial"] = start
        options.setdefault("initial_blocks", blocks)
    return objective, _solve_natural_conic(objective, **options)

def _fit_natural_conic_intervals_auto(
    support,
    intervals,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    weights=None,
    /,
    *,
    degree_config=None,
    **options,
):
    """Select interval degree from omitted observed information.

    Each candidate is an exact natural interval-likelihood fit.  The
    missing-information score decides whether the next omitted power block
    justifies additional polynomial capacity.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate density support.
    intervals : array_like, shape (R, 2)
        Ordered censoring rows; endpoints may be infinite.
    allow_lower_boundary, allow_upper_boundary : bool, optional
        Endpoint logarithmic-amplitude availability.
    weights : array_like or None, optional
        Nonnegative interval weights.
    degree_config : _DegreeSelectionConfig or None, optional
        Omitted-statistic score/rank policy.
    **options : dict
        Passed to every natural conic solve.

    Returns
    -------
    objective : _NaturalIntervalObjectiveFunction
        Exact interval objective at the selected degree.
    result : _ConicNewtonResult
        Natural conic fit at the selected degree.

    Raises
    ------
    RuntimeError
        If no admissible natural interval model produces a usable fit.
    """
    x = np.asarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError("intervals must have shape (R, 2)")
    if np.any(np.isnan(x)) or np.any(x[:, 0] > x[:, 1]):
        raise ValueError("intervals must be ordered and contain no NaN")

    cfg = _DegreeSelectionConfig() if degree_config is None else degree_config
    degrees = [
        int(d) for d in _admissible_degrees(support, AUTO_POLY_DEGREE_MAX)
        if int(d) >= int(AUTO_POLY_DEGREE_MIN)
    ]
    if not degrees:
        degrees = [int(AUTO_POLY_DEGREE_MIN)]

    best = None
    max_degree = int(degrees[-1])
    for index, degree in enumerate(degrees):
        try:
            candidate = _fit_natural_conic_intervals(
                support, x, degree,
                allow_lower_boundary, allow_upper_boundary, weights,
                **options,
            )
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(
                exc, f"natural auto-degree interval candidate degree={degree}",
                routine=True,
            )
            continue
        best = candidate
        probe_orders = _probe_orders_for_degree(
            degree, max_degree, block_size=cfg.probe_block_size
        )
        if index == len(degrees) - 1 or not probe_orders:
            return candidate
        diagnostic = _interval_omitted_statistic_diagnostic(
            _degree_diagnostic_fit(*candidate), probe_orders, config=cfg
        )
        if not diagnostic.should_expand:
            return candidate

    if best is None:
        raise RuntimeError(
            "auto poly_degree selection failed: no admissible natural interval "
            "model produced a valid fit"
        )
    return best
