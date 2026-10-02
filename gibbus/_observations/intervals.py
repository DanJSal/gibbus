"""Canonical interval observations and interval-reduction helpers.

This module owns the observation-side representation of interval-censored data
for the fitting pipeline.  Validation, support canonicalization, normalized
weights, duplicate-row collapse, and deterministic finite-interval
Gauss--Legendre plans are model-independent.

It also provides the model-aware reduction adapters used by the interval
objective.  Ordinary finite rows strictly inside the support are routed to the
end-to-end compiled Gauss--Legendre kernel in ``_finite_reductions``; the plan
object built here serves the degree diagnostics.  Positive-width finite rows
that touch a finite support boundary, and rows with an infinite endpoint, use
a prepared compiled adaptive Gauss--Kronrod reducer bound to the current model
state.  Scalar and batch entry points share that prepared geometry; fitting
batches heterogeneous adaptive rows and accumulates weighted conditional
statistics directly in Cython.  Endpoint-aware mappings damp finite-boundary
log singularities.

Zero-width rows retain the library's documented point-observation limit, and
whole-support rows can be short-circuited by callers to exact model moments.
"""

import weakref
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .._defaults import (
    INTERVAL_GL_ORDER,
    INTERVAL_W_EPS_MULT,
    QUAD_EPSABS,
    QUAD_EPSREL,
    QUAD_LIMIT,
)
from .._model.coords import _safe_scaled_difference
from ._interval_integrals import AdaptiveIntervalIntegrator, statistic_kinds
from .empirical import _normalised_weights

_GL_X, _GL_W = np.polynomial.legendre.leggauss(int(INTERVAL_GL_ORDER))
_GL_LOG_W = np.log(_GL_W)


@dataclass(frozen=True)
class _AdaptiveIntervalReduction:
    """Conditional reduction for one model-aware interval integral.

    Parameters
    ----------
    log_probability : float
        Logarithm of the normalized interval probability in canonical
        coordinates.
    mean : numpy.ndarray
        Conditional mean of the primary vector-valued function.
    covariance : numpy.ndarray
        Conditional covariance of the primary function.
    extra_mean : numpy.ndarray
        Conditional means of optional extra scalar/vector functions.
    """

    log_probability: float
    mean: np.ndarray
    covariance: np.ndarray
    extra_mean: np.ndarray


_STATISTIC_KIND = statistic_kinds()


class _PreparedAdaptiveIntervalReducer:
    """Prepared compiled reducer for repeated intervals under one model state."""

    def __init__(self, integrator, /):
        """Store one prepared compiled quadrature context.

        Parameters
        ----------
        integrator : AdaptiveIntervalIntegrator
            Compiled reducer bound to one live model state and statistic layout.
        """
        self._integrator = integrator

    def reduce(self, interval, /):
        """Reduce one interval with the prepared compiled quadrature context.

        Parameters
        ----------
        interval : array_like, shape (2,)
            Ordered canonical interval endpoints.
        """
        log_probability, mean, covariance, extra_mean = self._integrator.reduce(interval)
        return _AdaptiveIntervalReduction(
            log_probability=float(log_probability),
            mean=np.asarray(mean, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            extra_mean=np.asarray(extra_mean, dtype=np.float64),
        )

    def reduce_many(self, intervals, /):
        """Reduce many intervals in one compiled batch.

        Parameters
        ----------
        intervals : array_like, shape (n, 2)
            Ordered canonical interval endpoints.

        Returns
        -------
        tuple
            ``(log_probability, mean, covariance, extra_mean)`` arrays whose
            leading dimension indexes the input rows.
        """
        return self._integrator.reduce_many(intervals)

    def reduce_weighted(self, intervals, weights, /):
        """Reduce a batch and accumulate weighted conditional statistics.

        Parameters
        ----------
        intervals : array_like, shape (n, 2)
            Ordered canonical interval endpoints.
        weights : array_like, shape (n,)
            Finite row weights.

        Returns
        -------
        tuple
            Per-row log probabilities plus the weighted sums of the primary
            means, primary covariance matrices, and extra-statistic means.
        """
        return self._integrator.reduce_weighted(intervals, weights)


def _prepare_adaptive_interval_reducer(
    state, primary_descriptors, /, *, extra_descriptors=()
):
    """Build a compiled adaptive reducer for fixed statistic descriptors.

    Parameters
    ----------
    state : object
        Normalized state supplying the potential and quadrature geometry.
    primary_descriptors : iterable
        Statistic descriptors whose conditional mean and covariance are needed.
    extra_descriptors : iterable, optional
        Statistic descriptors whose conditional means alone are needed.

    Descriptors are ``("poly", coefficients)``, ``("power", order)``,
    ``("lower_log", None)``, ``("upper_log", None)``, or
    ``("constant", None)``.  All descriptor translation happens once here;
    per-node evaluation is entirely inside the Cython quadrature kernel.
    """
    primary = tuple(primary_descriptors)
    extra = tuple(extra_descriptors)
    descriptors = primary + extra
    if not descriptors:
        raise ValueError("at least one interval statistic descriptor is required")

    max_coeff = max(
        [
            np.asarray(payload, dtype=np.float64).reshape(-1).size
            for kind, payload in descriptors
            if kind == "poly"
        ]
        or [1]
    )
    kinds = np.empty(len(descriptors), dtype=np.int32)
    orders = np.zeros(len(descriptors), dtype=np.int64)
    lengths = np.zeros(len(descriptors), dtype=np.int32)
    coefficients = np.zeros((len(descriptors), max_coeff), dtype=np.float64)

    for i, (kind, payload) in enumerate(descriptors):
        try:
            kinds[i] = int(_STATISTIC_KIND[kind])
        except KeyError as exc:
            raise ValueError(f"unknown adaptive interval statistic kind {kind!r}") from exc
        if kind == "poly":
            coeff = np.asarray(payload, dtype=np.float64).reshape(-1)
            if coeff.size < 1 or not np.all(np.isfinite(coeff)):
                raise ValueError("interval polynomial statistic must be finite and non-empty")
            lengths[i] = int(coeff.size)
            coefficients[i, :coeff.size] = coeff
        elif kind == "power":
            order = int(payload)
            if order < 0:
                raise ValueError("interval power statistic order must be >= 0")
            orders[i] = order
        elif kind not in {"lower_log", "upper_log", "constant"}:
            raise ValueError(f"unknown adaptive interval statistic kind {kind!r}")

    integrator = AdaptiveIntervalIntegrator(
        state.q_poly,
        state.spec.support,
        state.boundary_amplitudes,
        float(state.mode),
        float(state.local_scale),
        float(state.log_Z),
        kinds,
        orders,
        coefficients,
        lengths,
        len(primary),
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=QUAD_LIMIT,
    )
    return _PreparedAdaptiveIntervalReducer(integrator)


def _prepare_partial_interval_reducer(state, /):
    """Prepare compiled first-partial mean/covariance interval reduction.

    Parameters
    ----------
    state : _NaturalCoreState
        Normalized model state.
    """
    primary = []
    for partial in state.partials:
        if partial.kind == "poly":
            primary.append(("poly", partial.coefficients))
        elif partial.boundary_side == "lower":
            primary.append(("lower_log", None))
        elif partial.boundary_side == "upper":
            primary.append(("upper_log", None))
        else:
            raise ValueError("unknown potential-partial descriptor")
    return _prepare_adaptive_interval_reducer(state, primary)


def _prepare_statistic_interval_reducer(state, power_orders, sides, /):
    """Prepare compiled power/log natural-statistic interval reduction.

    Parameters
    ----------
    state : object
        Normalized model state.
    power_orders : iterable of int
        Canonical power-statistic orders.
    sides : iterable of {"lower", "upper"}
        Enabled canonical boundary-log statistic sides.
    """
    primary = [("power", int(k)) for k in power_orders]
    for side in sides:
        if side == "lower":
            primary.append(("lower_log", None))
        elif side == "upper":
            primary.append(("upper_log", None))
        else:
            raise ValueError("unknown boundary statistic side")
    return _prepare_adaptive_interval_reducer(state, primary)


@dataclass(frozen=True)
class _IntervalObservations:
    """Canonical interval rows plus normalized observation weights.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R_unique, 2)
        Ordered canonical lower/upper interval endpoints.  Infinite endpoints
        are allowed; NaN endpoints are not.
    weights : numpy.ndarray, shape (R_unique,)
        Normalized weights of the unique interval rows, summing to one.
    endpoints : numpy.ndarray
        Sorted unique endpoints appearing in ``intervals``.
    support : tuple of (float, float) or None
        Canonical support associated with the fixed fitting coordinate, when
        available.
    total_weight : float
        Sum of input weights before normalization, or the input observation
        count for unweighted data.
    effective_n : float
        Kish effective sample size of the *unmerged* observations.
    n_observations : int
        Number of interval rows before duplicate compression.
    point_lower_distance, point_upper_distance : numpy.ndarray
        Preserved canonical distances from finite support endpoints for exact
        point observations.  Non-point rows and unavailable sides carry NaN.
        These remain physical-distance metadata even when affine rounding maps
        the stored canonical point exactly onto a support endpoint.
    """

    intervals: np.ndarray
    weights: np.ndarray
    endpoints: np.ndarray
    support: tuple[float, float] | None
    total_weight: float
    effective_n: float
    n_observations: int
    point_lower_distance: np.ndarray
    point_upper_distance: np.ndarray

    @property
    def n_unique(self):
        """Return the number of unique stored interval rows.

        Returns
        -------
        int
            Number of rows after exact duplicate compression.
        """
        return int(self.intervals.shape[0])

    @property
    def widths(self):
        """Return interval widths, including ``inf`` where appropriate.

        Returns
        -------
        numpy.ndarray
            ``upper - lower`` for each unique row.
        """
        return self.intervals[:, 1] - self.intervals[:, 0]

    @property
    def point_rows(self):
        """Return a mask for exact zero-width observations.

        Returns
        -------
        numpy.ndarray of bool
            True where lower and upper endpoints are identical.
        """
        return self.intervals[:, 0] == self.intervals[:, 1]

    @property
    def finite_rows(self):
        """Return a mask for intervals with two finite endpoints.

        Returns
        -------
        numpy.ndarray of bool
            True where both endpoints are finite.
        """
        return np.all(np.isfinite(self.intervals), axis=1)

    @property
    def left_infinite_rows(self):
        """Return a mask for intervals unbounded only on the left.

        Returns
        -------
        numpy.ndarray of bool
            Rows of the form ``(-inf, U]`` with finite ``U``.
        """
        return np.isneginf(self.intervals[:, 0]) & np.isfinite(self.intervals[:, 1])

    @property
    def right_infinite_rows(self):
        """Return a mask for intervals unbounded only on the right.

        Returns
        -------
        numpy.ndarray of bool
            Rows of the form ``[L, inf)`` with finite ``L``.
        """
        return np.isfinite(self.intervals[:, 0]) & np.isposinf(self.intervals[:, 1])

    @property
    def whole_support_rows(self):
        """Return a mask for rows unbounded in both directions.

        Returns
        -------
        numpy.ndarray of bool
            True for ``(-inf, inf)`` observation rows.
        """
        return np.isneginf(self.intervals[:, 0]) & np.isposinf(self.intervals[:, 1])

    @property
    def has_infinite_rows(self):
        """Return whether any stored interval has an infinite endpoint.

        Returns
        -------
        bool
            True when finite local Gauss-Legendre quadrature is insufficient.
        """
        return bool(np.any(~self.finite_rows))

    def finite_quadrature(self, mode, /):
        """Build the deterministic local quadrature plan for finite rows.

        Parameters
        ----------
        mode : float
            Current model mode in canonical coordinates.  An interval that
            strictly contains a finite mode is split at the mode before local
            Gauss-Legendre quadrature used by the finite interval path.

        Returns
        -------
        _FiniteIntervalQuadrature
            Model-independent node/weight plan for all stored rows.

        Raises
        ------
        NotImplementedError
            If any row has an infinite endpoint.  Infinite interval integration
            is model/tail dependent and belongs to the later exact interval
            objective rather than this finite quadrature helper.
        """
        if self.has_infinite_rows:
            raise NotImplementedError(
                "finite interval quadrature does not handle infinite censoring rows"
            )
        return _build_finite_interval_quadrature(self.intervals, mode)


@dataclass(frozen=True)
class _FiniteIntervalQuadrature:
    """Deterministic finite-interval quadrature and reduction plan.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Finite ordered canonical interval rows.
    nodes : numpy.ndarray, shape (R, 2*q)
        Local Gauss-Legendre nodes; unused slots carry arbitrary midpoint
        values and ``-inf`` logarithmic weights.
    log_weights : numpy.ndarray, shape (R, 2*q)
        Logarithms of local quadrature weights.
    point_limit : numpy.ndarray of bool, shape (R,)
        Rows sufficiently narrow to use the point/midpoint limit.
    """

    intervals: np.ndarray
    nodes: np.ndarray
    log_weights: np.ndarray
    point_limit: np.ndarray

    @property
    def midpoints(self):
        """Return finite interval midpoints.

        Returns
        -------
        numpy.ndarray
            Midpoint of each represented interval.
        """
        return 0.5 * (self.intervals[:, 0] + self.intervals[:, 1])

    @property
    def widths(self):
        """Return finite interval widths.

        Returns
        -------
        numpy.ndarray
            Upper minus lower endpoint for each row.
        """
        return self.intervals[:, 1] - self.intervals[:, 0]

    def log_integrals(self, log_kernel, /, *, point_log_kernel=None):
        """Reduce current-model log-kernel values to row log integrals.

        The input is an *unnormalized* log density such as ``-q(z)``.  For
        positive-width narrow rows, the point limit is ``log_kernel(mid) +
        log(width)``.  Exact zero-width rows retain the library's point-limit
        convention and return ``log_kernel(mid)`` rather than ``-inf``.

        Parameters
        ----------
        log_kernel : array_like, shape like ``nodes``
            Current unnormalized log-density values at quadrature nodes.
        point_log_kernel : array_like, shape (R,) or shape (n_point_limit,), optional
            Current log-kernel at row midpoints.  Required when any row uses
            the point limit.  A compact array containing only point-limit rows
            is accepted.

        Returns
        -------
        numpy.ndarray, shape (R,)
            Logarithms of unnormalized interval integrals / point-limit
            kernels.

        Raises
        ------
        ValueError
            If input shapes are inconsistent or midpoint values are missing.
        """
        lk = np.asarray(log_kernel, dtype=np.float64)
        if lk.shape != self.nodes.shape:
            raise ValueError("log_kernel must have the same shape as quadrature nodes")

        out = logsumexp(lk + self.log_weights, axis=1)
        if np.any(self.point_limit):
            pm = _point_limit_values(
                point_log_kernel, self.point_limit, self.intervals.shape[0], "point_log_kernel"
            )
            widths = self.widths[self.point_limit]
            log_width = np.zeros_like(widths)
            positive = widths > 0.0
            log_width[positive] = np.log(widths[positive])
            out[self.point_limit] = pm + log_width
        return np.asarray(out, dtype=np.float64)

    def conditional_mean(
        self,
        log_kernel,
        values,
        /,
        *,
        point_values=None,
        log_integrals=None,
    ):
        """Return row-wise conditional means of supplied function values.

        Parameters
        ----------
        log_kernel : array_like, shape ``(R, Q)``
            Current unnormalized log-density values at quadrature nodes.
        values : array_like, shape ``(R, Q, ...)``
            Function values at the same nodes.  Any trailing shape is retained
            in the returned conditional means.
        point_values : array_like, shape ``(R, ...)`` or ``(n_point, ...)``, optional
            Function values at midpoints for point-limit rows.  Required when
            the plan contains such rows.
        log_integrals : array_like, shape ``(R,)``, optional
            Precomputed row log integrals.  Supplying this avoids recomputing
            the log-sum-exp normalization.

        Returns
        -------
        numpy.ndarray, shape ``(R, ...)``
            Conditional mean of each supplied function for every interval.
        """
        lk, val = _validate_node_values(self, log_kernel, values)
        if log_integrals is None:
            # Point-limit rows do not use their quadrature normalization, so
            # a finite placeholder is sufficient here.
            log_i = logsumexp(lk + self.log_weights, axis=1)
        else:
            log_i = np.asarray(log_integrals, dtype=np.float64).reshape(-1)
            if log_i.size != self.intervals.shape[0]:
                raise ValueError("log_integrals must have one entry per interval")

        ordinary = ~self.point_limit
        trailing = val.shape[2:]
        out = np.zeros((self.intervals.shape[0],) + trailing, dtype=np.float64)
        if np.any(ordinary):
            alpha = np.exp(
                lk[ordinary] + self.log_weights[ordinary] - log_i[ordinary, None]
            )
            expand = (slice(None), slice(None)) + (None,) * len(trailing)
            out[ordinary] = np.sum(alpha[expand] * val[ordinary], axis=1)

        if np.any(self.point_limit):
            pv = _point_limit_values(
                point_values, self.point_limit, self.intervals.shape[0], "point_values"
            )
            expected_shape = (int(np.count_nonzero(self.point_limit)),) + trailing
            pv = np.asarray(pv, dtype=np.float64)
            if pv.shape != expected_shape:
                raise ValueError(
                    "point_values trailing shape must match values trailing shape"
                )
            out[self.point_limit] = pv
        return out

    def conditional_mean_covariance(
        self,
        log_kernel,
        values,
        /,
        *,
        point_values=None,
        log_integrals=None,
    ):
        """Return row-wise conditional means and covariance matrices.

        This reduction is intended for vectors of first parameter partials
        ``h``.  Point-limit rows have zero conditional covariance, as required
        by the zero-width limit.

        Parameters
        ----------
        log_kernel : array_like, shape ``(R, Q)``
            Current unnormalized log-density values at quadrature nodes.
        values : array_like, shape ``(R, Q, P)``
            Vector-valued function evaluations at quadrature nodes.
        point_values : array_like, shape ``(R, P)`` or ``(n_point, P)``, optional
            Midpoint function values for point-limit rows.
        log_integrals : array_like, shape ``(R,)``, optional
            Precomputed row log integrals.

        Returns
        -------
        mean : numpy.ndarray, shape ``(R, P)``
            Conditional function means.
        covariance : numpy.ndarray, shape ``(R, P, P)``
            Conditional covariance matrices.

        Raises
        ------
        ValueError
            If ``values`` is not vector-valued with shape ``(R, Q, P)``.
        """
        lk, val = _validate_node_values(self, log_kernel, values)
        if val.ndim != 3:
            raise ValueError("values must have shape (R, Q, P) for covariance")
        if log_integrals is None:
            log_i = logsumexp(lk + self.log_weights, axis=1)
        else:
            log_i = np.asarray(log_integrals, dtype=np.float64).reshape(-1)
            if log_i.size != self.intervals.shape[0]:
                raise ValueError("log_integrals must have one entry per interval")

        mean = self.conditional_mean(
            lk,
            val,
            point_values=point_values,
            log_integrals=log_i,
        )
        cov = np.zeros(
            (self.intervals.shape[0], val.shape[2], val.shape[2]), dtype=np.float64
        )
        ordinary = ~self.point_limit
        if np.any(ordinary):
            alpha = np.exp(
                lk[ordinary] + self.log_weights[ordinary] - log_i[ordinary, None]
            )
            centered = val[ordinary] - mean[ordinary, None, :]
            cov[ordinary] = np.einsum(
                "rq,rqi,rqj->rij", alpha, centered, centered, optimize=True
            )
        return mean, cov


_ROW_GROUPING = None
"""``(weakref to rows, grouping)`` of the last interval-row array grouped by
``_row_grouping``; a fit groups the same rows for every component, start and
M-step.  Only read-only arrays are cached (the fit freezes its private copy of
the samples), so the rows cannot change underneath the cache."""


def _lexicographic_grouping(rows, /):
    """Return exact lexicographic duplicate groups without ``np.unique(axis=0)``.

    Parameters
    ----------
    rows : array_like, shape (R, C)
        Finite-or-infinite numeric rows containing no NaN.

    Returns
    -------
    tuple
        ``(first, inverse, n_unique)`` with the same ordering convention as
        ``np.unique(rows, axis=0, return_index=True, return_inverse=True)``.

    Notes
    -----
    NumPy's structured ``unique(axis=0)`` path is comparatively expensive for
    the narrow two- to four-column arrays used by censored fits.  A stable
    lexicographic indirect sort followed by adjacent-row comparisons produces
    the same groups while avoiding structured-array construction.
    """
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 1:
        raise ValueError("rows must be a non-empty two-dimensional array")
    if np.any(np.isnan(x)):
        raise ValueError("rows must not contain NaN")

    # np.lexsort uses the last key as primary; reverse the columns so column 0
    # is primary, matching np.unique(axis=0)'s lexicographic row order.
    order = np.lexsort(tuple(x[:, j] for j in range(x.shape[1] - 1, -1, -1)))
    ordered = x[order]
    start = np.empty(x.shape[0], dtype=bool)
    start[0] = True
    start[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    ranks = np.cumsum(start, dtype=np.intp) - 1
    first = np.asarray(order[start], dtype=np.intp)
    inverse = np.empty(x.shape[0], dtype=np.intp)
    inverse[order] = ranks
    return first, inverse, int(first.size)


def _row_grouping(rows, /):
    """Group identical interval rows; cached per rows array.

    Identical user-coordinate rows map to identical canonical rows in any
    fitting coordinate, so this grouping serves every component coordinate.

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 2)
        Interval rows in user coordinates.

    Returns
    -------
    tuple
        ``(first, inverse, n_unique)``: the first row of each group (in
        sorted order of the distinct rows), each row's group, and the number
        of groups.
    """
    global _ROW_GROUPING
    entry = _ROW_GROUPING
    if entry is not None and entry[0]() is rows:
        return entry[1]
    first, inverse, _ = _lexicographic_grouping(rows)
    first.setflags(write=False)
    inverse = inverse.reshape(-1)
    inverse.setflags(write=False)
    grouping = (first, inverse, int(first.shape[0]))
    if isinstance(rows, np.ndarray) and not rows.flags.writeable:
        _ROW_GROUPING = (weakref.ref(rows), grouping)
    return grouping


def _merge_duplicate_intervals(intervals, weights, lower_distance=None,
                               upper_distance=None, /, *, grouping_cache=None):
    """Merge identical canonical rows while preserving exact-point geometry.

    Exact physical points can map to the same canonical float even though
    their sub-ulp distances from a support boundary differ.  Such rows are
    therefore duplicates only when both the canonical interval and preserved
    point-boundary distances agree.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Canonical observation intervals.
    weights : numpy.ndarray, shape (R,)
        Weights corresponding to the observation rows.
    lower_distance : numpy.ndarray or None
        Optional preserved distances to the finite lower support boundary.
    upper_distance : numpy.ndarray or None
        Optional preserved distances to the finite upper support boundary.
    grouping_cache : dict or None, optional
        Holds the duplicate grouping between calls on the same rows (a mixture
        refits the same rows under new weights at every M-step); the grouping
        depends on the rows and distances only, never on the weights.
    """
    x = np.asarray(intervals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    ld = (np.full(n, np.nan, dtype=np.float64) if lower_distance is None
          else np.asarray(lower_distance, dtype=np.float64).reshape(-1))
    ud = (np.full(n, np.nan, dtype=np.float64) if upper_distance is None
          else np.asarray(upper_distance, dtype=np.float64).reshape(-1))
    if ld.size != n or ud.size != n:
        raise ValueError("boundary-distance metadata must match interval rows")

    # -1 is an unambiguous missing-value sentinel because valid distances are
    # non-negative.  Using NaN directly in np.unique would prevent equal
    # non-point rows from comparing equal.
    grouping = None if grouping_cache is None else grouping_cache.get("grouping")
    if grouping is None or grouping[1].shape[0] != n:
        key = np.column_stack((
            x,
            np.where(np.isfinite(ld), ld, -1.0),
            np.where(np.isfinite(ud), ud, -1.0),
        ))
        first, inverse, n_unique = _lexicographic_grouping(key)
        grouping = (first, inverse.reshape(-1), int(n_unique))
        if grouping_cache is not None:
            grouping_cache["grouping"] = grouping
    first, inverse, n_unique = grouping
    if n_unique == n:
        return (
            np.ascontiguousarray(x, dtype=np.float64),
            np.ascontiguousarray(w, dtype=np.float64),
            np.ascontiguousarray(ld, dtype=np.float64),
            np.ascontiguousarray(ud, dtype=np.float64),
        )
    merged = np.zeros(n_unique, dtype=np.float64)
    np.add.at(merged, inverse, w)
    return (
        np.ascontiguousarray(x[first], dtype=np.float64),
        np.ascontiguousarray(merged, dtype=np.float64),
        np.ascontiguousarray(ld[first], dtype=np.float64),
        np.ascontiguousarray(ud[first], dtype=np.float64),
    )


def _build_interval_observations(
    intervals, weights=None, /, *, coordinate=None, support=None, deduplicate=True,
    grouping_cache=None,
):
    """Build the canonical interval-observation representation.

    Parameters
    ----------
    intervals : array_like, shape (R, 2)
        Ordered lower/upper endpoints.  They are interpreted as user
        coordinates when ``coordinate`` is supplied and as already canonical
        otherwise.  Infinite endpoints are permitted; NaN is not.
    weights : array_like, shape (R,) or None, optional
        Non-negative observation weights.
    coordinate : _FitCoordinate or None, optional
        Fixed fitting coordinate.  When supplied, intervals are mapped to
        canonical coordinates and its canonical support is recorded.
    support : tuple of (float, float) or None, optional
        Canonical support used when ``coordinate`` is absent.  If provided,
        every interval must lie inside it.
    deduplicate : bool, optional
        Merge identical canonical rows exactly.  Defaults to ``True``.
    grouping_cache : dict or None, optional
        Reuses the duplicate grouping across calls on the same rows (see
        ``_merge_duplicate_intervals``).

    Returns
    -------
    _IntervalObservations
        Immutable observation-side summary.

    Raises
    ------
    ValueError
        If interval shape/order/support or weights are invalid.
    """
    x = np.asarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError("intervals must have shape (R, 2) with R >= 1")
    if np.any(np.isnan(x)):
        raise ValueError("interval endpoints must not contain NaN")
    if np.any(x[:, 0] > x[:, 1]):
        raise ValueError("interval lower endpoints must not exceed upper endpoints")

    n = int(x.shape[0])
    w, total, effective_n = _normalised_weights(n, weights, "interval")

    canonical_support = support
    point_lower_distance = np.full(n, np.nan, dtype=np.float64)
    point_upper_distance = np.full(n, np.nan, dtype=np.float64)
    if coordinate is not None:
        if support is not None:
            raise ValueError("support must be omitted when coordinate is supplied")
        physical = np.asarray(coordinate.physical_support, dtype=np.float64)
        exact = x[:, 0] == x[:, 1]
        point = x[:, 0]
        direction = float(coordinate.direction)
        scale = float(coordinate.scale)
        if np.isfinite(coordinate.canonical_support[0]):
            ep = float(physical[0] if direction > 0.0 else physical[1])
            d = direction * _safe_scaled_difference(point, ep, scale)
            point_lower_distance[exact] = d[exact]
        if np.isfinite(coordinate.canonical_support[1]):
            ep = float(physical[1] if direction > 0.0 else physical[0])
            d = -direction * _safe_scaled_difference(point, ep, scale)
            point_upper_distance[exact] = d[exact]
        x = coordinate.intervals_to_canonical(x)
        canonical_support = tuple(map(float, coordinate.canonical_support))
    else:
        x = np.ascontiguousarray(x, dtype=np.float64)
        if support is not None:
            canonical_support = tuple(map(float, support))
            exact = x[:, 0] == x[:, 1]
            if np.isfinite(canonical_support[0]):
                point_lower_distance[exact] = x[exact, 0] - canonical_support[0]
            if np.isfinite(canonical_support[1]):
                point_upper_distance[exact] = canonical_support[1] - x[exact, 0]

    if canonical_support is not None:
        lower, upper = canonical_support
        if not lower < upper:
            raise ValueError("support lower endpoint must be smaller than upper")
        if np.any(x[:, 0] < lower) or np.any(x[:, 1] > upper):
            raise ValueError("interval observations fall outside the canonical support")

    for distances in (point_lower_distance, point_upper_distance):
        finite_distance = np.isfinite(distances)
        if np.any(distances[finite_distance] < 0.0):
            raise ValueError("exact point has negative preserved boundary distance")

    if deduplicate:
        x, w, point_lower_distance, point_upper_distance = _merge_duplicate_intervals(
            x, w, point_lower_distance, point_upper_distance,
            grouping_cache=grouping_cache,
        )

    return _IntervalObservations(
        intervals=np.ascontiguousarray(x, dtype=np.float64),
        weights=np.ascontiguousarray(w, dtype=np.float64),
        endpoints=np.unique(x.reshape(-1)),
        support=canonical_support,
        total_weight=float(total),
        effective_n=float(effective_n),
        n_observations=n,
        point_lower_distance=np.ascontiguousarray(point_lower_distance, dtype=np.float64),
        point_upper_distance=np.ascontiguousarray(point_upper_distance, dtype=np.float64),
    )


def _build_finite_interval_quadrature(intervals, mode, /):
    """Construct the deterministic local quadrature plan for finite rows.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Finite ordered canonical interval rows.
    mode : float
        Current model mode in canonical coordinates.

    Returns
    -------
    _FiniteIntervalQuadrature
        Deterministic nodes, logarithmic weights, and point-limit mask.
    """
    x = np.asarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("finite interval quadrature requires finite (R, 2) rows")
    if np.any(x[:, 0] > x[:, 1]):
        raise ValueError("interval lower endpoints must not exceed upper endpoints")

    lower = x[:, 0]
    upper = x[:, 1]
    width = upper - lower
    mid = 0.5 * (lower + upper)
    tiny = width <= INTERVAL_W_EPS_MULT * (1.0 + np.abs(mid))

    q = int(INTERVAL_GL_ORDER)
    nodes = np.repeat(mid[:, None], 2 * q, axis=1)
    log_weights = np.full((x.shape[0], 2 * q), -np.inf, dtype=np.float64)

    active = ~tiny
    split = active & np.isfinite(mode) & (lower < float(mode)) & (float(mode) < upper)
    whole = active & ~split

    if np.any(whole):
        h = 0.5 * width[whole]
        m = mid[whole]
        nodes[whole, :q] = m[:, None] + h[:, None] * _GL_X[None, :]
        log_weights[whole, :q] = np.log(h)[:, None] + _GL_LOG_W[None, :]

    if np.any(split):
        ls = lower[split]
        us = upper[split]
        md = float(mode)

        h_left = 0.5 * (md - ls)
        m_left = 0.5 * (ls + md)
        nodes[split, :q] = m_left[:, None] + h_left[:, None] * _GL_X[None, :]
        log_weights[split, :q] = np.log(h_left)[:, None] + _GL_LOG_W[None, :]

        h_right = 0.5 * (us - md)
        m_right = 0.5 * (us + md)
        nodes[split, q:] = m_right[:, None] + h_right[:, None] * _GL_X[None, :]
        log_weights[split, q:] = np.log(h_right)[:, None] + _GL_LOG_W[None, :]

    return _FiniteIntervalQuadrature(
        intervals=np.ascontiguousarray(x, dtype=np.float64),
        nodes=nodes,
        log_weights=log_weights,
        point_limit=tiny,
    )


def _validate_node_values(plan, log_kernel, values, /):
    """Validate quadrature log-kernel and node-value arrays.

    Parameters
    ----------
    plan : _FiniteIntervalQuadrature
        Quadrature plan defining the first two dimensions.
    log_kernel : array_like
        Log-kernel values at plan nodes.
    values : array_like
        Function values whose first two dimensions match plan nodes.

    Returns
    -------
    log_kernel : numpy.ndarray
    values : numpy.ndarray

    Raises
    ------
    ValueError
        If shapes are incompatible.
    """
    lk = np.asarray(log_kernel, dtype=np.float64)
    val = np.asarray(values, dtype=np.float64)
    if lk.shape != plan.nodes.shape:
        raise ValueError("log_kernel must have the same shape as quadrature nodes")
    if val.ndim < 2 or val.shape[:2] != plan.nodes.shape:
        raise ValueError("values must begin with the quadrature node shape")
    return lk, val


def _point_limit_values(values, mask, n_rows, name, /):
    """Select compact or full-row values for point-limit observations.

    Parameters
    ----------
    values : array_like or None
        Full-row values or compact point-limit-row values.
    mask : numpy.ndarray of bool
        Point-limit row mask.
    n_rows : int
        Total number of rows.
    name : str
        Argument name used in error messages.

    Returns
    -------
    numpy.ndarray
        Compact values for rows where ``mask`` is true.

    Raises
    ------
    ValueError
        If values are absent or have an incompatible leading dimension.
    """
    if values is None:
        raise ValueError(f"{name} is required for point-limit rows")
    arr = np.asarray(values, dtype=np.float64)
    n_point = int(np.count_nonzero(mask))
    if arr.ndim == 0:
        if n_point != 1:
            raise ValueError(f"{name} must have one leading entry per point-limit row")
        return arr.reshape(1)
    if arr.shape[0] == n_rows:
        return arr[mask]
    if arr.shape[0] == n_point:
        return arr
    raise ValueError(
        f"{name} must have leading dimension n_rows or n_point_limit"
    )
