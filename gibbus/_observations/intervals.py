"""Canonical interval observations: validated rows, weights, duplicate merge.

G1 owner retained (G3-031): the observation-side representation of
interval-censored data.  Validation, support checks, normalized weights and
exact duplicate-row collapse are model-independent; the G3 interval
likelihood (``_fit.basis_objective``) integrates each merged row with the
normalizer.  Zero-width rows retain the library's documented
point-observation limit.
"""

import weakref
from dataclasses import dataclass

import numpy as np

from .._defaults import INTERVAL_GL_ORDER
from .empirical import _canonical_weights

_GL_X, _GL_W = np.polynomial.legendre.leggauss(int(INTERVAL_GL_ORDER))


@dataclass(frozen=True)
class _IntervalObservations:
    """Canonical interval rows plus normalized observation weights.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R_unique, 2)
        Ordered lower/upper interval endpoints.  Infinite endpoints are
        allowed; NaN endpoints are not.
    weights : numpy.ndarray, shape (R_unique,)
        Normalized weights of the unique interval rows, summing to one.
    support : tuple of (float, float) or None
        Support the rows were checked against, when given.
    total_weight : float
        Sum of input weights before normalization, or the input observation
        count for unweighted data.
    effective_n : float
        Kish effective sample size of the *unmerged* observations.
    n_observations : int
        Number of interval rows before duplicate compression.
    """

    intervals: np.ndarray
    weights: np.ndarray
    support: tuple[float, float] | None
    total_weight: float
    effective_n: float
    n_observations: int

    @property
    def n_unique(self):
        """Return the number of unique stored interval rows.

        Returns
        -------
        int
            Number of rows after exact duplicate compression.
        """
        return int(self.intervals.shape[0])


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


def _merge_duplicate_intervals(intervals, weights, /):
    """Merge identical rows exactly, summing their weights.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Observation intervals.
    weights : numpy.ndarray, shape (R,)
        Weights corresponding to the observation rows.

    Returns
    -------
    tuple
        ``(rows, weights)`` of the distinct rows, in lexicographic order.
    """
    x = np.asarray(intervals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if w.size != n:
        raise ValueError("weights must match the interval rows")
    first, inverse, n_unique = _lexicographic_grouping(x)
    if n_unique == n:
        return np.ascontiguousarray(x), np.ascontiguousarray(w)
    merged = np.zeros(n_unique, dtype=np.float64)
    np.add.at(merged, inverse.reshape(-1), w)
    return np.ascontiguousarray(x[first]), merged


def _build_interval_observations(intervals, weights=None, /, *, support=None):
    """Build the canonical interval-observation representation.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2), dtype float64
        Already-validated ordered lower/upper endpoints.  Infinite endpoints
        are permitted; NaN is not.
    weights : _WeightSummary or None, optional
        Prepared normalized weights and original-row reliability metadata.
        ``None`` creates equal weights; raw relative weights must be prepared
        explicitly before calling this constructor.
    support : tuple of (float, float) or None, optional
        Support every interval must lie inside.

    Returns
    -------
    _IntervalObservations
        Immutable observation-side summary (identical rows merged).

    Raises
    ------
    ValueError
        If the support is empty or a row lies outside it.
    """
    x = np.ascontiguousarray(intervals, dtype=np.float64)
    n = x.shape[0]
    w, total, effective_n = _canonical_weights(n, None) if weights is None else weights
    if support is not None:
        support = tuple(map(float, support))
        lower, upper = support
        if not lower < upper:
            raise ValueError("support lower endpoint must be smaller than upper")
        if np.any(x[:, 0] < lower) or np.any(x[:, 1] > upper):
            raise ValueError("interval observations fall outside the support")
    x, w = _merge_duplicate_intervals(x, w)
    return _IntervalObservations(
        intervals=x,
        weights=np.ascontiguousarray(w, dtype=np.float64),
        support=support,
        total_weight=float(total),
        effective_n=float(effective_n),
        n_observations=n,
    )
