"""Canonical coordinates that define the fixed basis geometry.

Every fit maps user coordinates affinely to ``z = (x - c) / h``, with ``c`` and
``h`` fixed once from the support and the data landmarks ``q10`` and ``q90``:

* full line: ``c = (q10 + q90) / 2``, ``h = (q90 - q10) / 2``, so the landmarks
  sit at ``z = -1`` and ``z = +1``, the fixed gate centers;
* lower half-line ``[L, inf)``: ``c = L``, ``h = q90 - L`` (support ``[0, inf)``);
* upper half-line ``(-inf, U]``: ``c = U``, ``h = U - q10`` (support ``(-inf, 0]``);
* bounded ``[L, U]``: ``c = (L + U) / 2``, ``h = (U - L) / 2`` (support
  ``[-1, 1]``), independent of the data.

The map is part of the model definition: the basis functions and gates are
fixed in ``z``.  Quantiles use one weighted convention (midpoint cumulative
weights, linear interpolation), so equal weights reproduce the unweighted
result.  Landmarks that do not identify a positive scale raise
:class:`ValueError`, and so do data on a finite-ended support whose spread
``q90 - q10`` is below ``COORD_MIN_SPREAD`` times their median's distance from
the nearest finite end (they would occupy a sliver of the coordinate).
Mixture components use their responsibility weights; when a component's
weighted landmarks do not identify a coordinate (it holds one tied value or
one censoring cutpoint), it uses the pooled data's coordinate instead.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import COORD_LOWER_QUANTILE, COORD_MIN_SPREAD, COORD_UPPER_QUANTILE


class _CoordinateNotIdentified(ValueError):
    """The data do not identify a usable canonical coordinate.

    Raised for landmarks that coincide and for data too concentrated for a
    finite-ended support; it is a property of the data, never of one basis
    candidate, so selection loops let it through.  Mixture components never
    raise it: their responsibility-weighted coordinate falls back to the
    pooled data's (the ``fallback`` of the coordinate builders).
    """


_REAL_LINE = "real_line"
_LOWER_HALF_LINE = "lower_half_line"
_UPPER_HALF_LINE = "upper_half_line"
_BOUNDED = "bounded"
# The canonical support of each kind is fixed: the coordinate's endpoints map
# onto it by definition, not by rounding the affine transform.
_CANONICAL_SUPPORT = {
    _REAL_LINE: (-np.inf, np.inf),
    _LOWER_HALF_LINE: (0.0, np.inf),
    _UPPER_HALF_LINE: (-np.inf, 0.0),
    _BOUNDED: (-1.0, 1.0),
}


def _safe_scaled_difference(values, center, scale, /):
    """Return ``(values - center) / scale`` without avoidable overflow.

    Parameters
    ----------
    values : array_like
        Values to center and scale.
    center : float
        Center in the same coordinate system as ``values``.
    scale : float
        Non-zero scale divisor.
    """
    x = np.asarray(values, dtype=np.float64)
    c = float(center)
    s = float(scale)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = (x - c) / s
        mag = np.maximum(np.maximum(np.abs(x), abs(c)), abs(s))
        fallback = ((x / mag) - (c / mag)) / (s / mag)
    bad = ~np.isfinite(out) & np.isfinite(x)
    return np.where(bad, fallback, out)


def _normalized_nonnegative_weights(weights, n, /):
    """Return canonical non-negative weights or ``None`` for unweighted data.

    Parameters
    ----------
    weights : array_like or None
        Non-negative finite weights, or ``None`` for genuinely unweighted data.
    n : int
        Expected number of weights.

    Returns
    -------
    numpy.ndarray or None
        Weights normalized to sum to one, or ``None`` only when ``weights`` is
        ``None``.

    Raises
    ------
    ValueError
        If a supplied weight vector has the wrong length, contains a negative or
        non-finite value, or has zero total mass.
    """
    if weights is None:
        return None
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != int(n):
        raise ValueError(f"weights must have length {int(n)}, got {w.size}")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    wmax = float(np.max(w))
    if not (wmax > 0.0):
        raise ValueError("weights must have positive total mass")
    scaled = w / wmax
    total = float(np.sum(scaled, dtype=np.float64))
    if not (total > 0.0 and np.isfinite(total)):
        raise ValueError("weights must have positive finite total mass")
    return scaled / total


@dataclass(frozen=True)
class _FitCoordinate:
    """One fixed canonical coordinate.

    The forward map is ``z = (x - center) / scale`` with ``scale > 0``.  Finite
    canonical endpoints are ``0`` (half-lines) or ``-1`` and ``1`` (bounded).

    Parameters
    ----------
    support_kind : str
        One of the private support-kind constants in this module.
    center : float
        Coordinate origin in user units.
    scale : float
        Positive fixed numerical scale in user units.
    canonical_support : tuple of (float, float)
        Support after transformation: ``(-inf, inf)``, ``(0, inf)``,
        ``(-inf, 0)`` or ``(-1, 1)``.
    physical_support : tuple of (float, float)
        Original user-coordinate support.  Retained so exact observations can
        preserve sub-ulp physical distances from finite boundaries after the
        affine map.
    """

    support_kind: str
    center: float
    scale: float
    canonical_support: tuple[float, float]
    physical_support: tuple[float, float]

    def to_canonical(self, values, /):
        """Map values from user coordinates to the fixed fitting coordinate.

        Parameters
        ----------
        values : array-like
            Values in user coordinates.

        Returns
        -------
        numpy.ndarray
            Values in the canonical fitting coordinate.
        """
        return _safe_scaled_difference(values, self.center, self.scale)

    def from_canonical(self, values, /):
        """Map values from the fixed fitting coordinate back to user units.

        Parameters
        ----------
        values : array-like
            Values in the canonical fitting coordinate.

        Returns
        -------
        numpy.ndarray
            Values in user coordinates.
        """
        z = np.asarray(values, dtype=np.float64)
        return self.center + self.scale * z

    def intervals_to_canonical(self, intervals, /):
        """Map ordered user-coordinate intervals and preserve endpoint order.

        Parameters
        ----------
        intervals : numpy.ndarray, shape (R, 2)
            Ordered lower/upper endpoints in user coordinates.

        Returns
        -------
        numpy.ndarray, shape (R, 2)
            Ordered lower/upper endpoints in canonical coordinates.
        """
        return np.ascontiguousarray(self.to_canonical(intervals), dtype=np.float64)


def _weighted_median(values, weights, /, *, order=None):
    """Return the median of values under optional non-negative weights.

    Parameters
    ----------
    values : numpy.ndarray, shape (R,)
        Finite sample values.
    weights : numpy.ndarray, shape (R,) or None
        Non-negative finite weights. ``None`` selects the ordinary unweighted
        median; malformed supplied weights are contract errors.
    order : numpy.ndarray or None, optional
        An ascending order of ``values`` when the caller already has one.
        Ties may be ordered either way: the result is the value at which the
        cumulative weight reaches one half, which ties share.

    Returns
    -------
    float
        Smallest value whose cumulative weight reaches half the total.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        return float(np.median(values))

    w = _normalized_nonnegative_weights(weights, values.size)
    if w is None:
        return float(np.median(values))

    if order is None:
        order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(w[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _weighted_quantile(values, weights, levels, /, *, order=None):
    """Return weighted quantiles under the midpoint convention.

    The sorted values carry cumulative probabilities ``(cumsum(w) - w / 2) / W``
    and quantiles interpolate linearly between them (clamped at the extremes).
    ``weights=None`` is the equal-weight case of the same rule.

    Parameters
    ----------
    values : numpy.ndarray, shape (R,)
        Finite values.
    weights : numpy.ndarray, shape (R,) or None
        Non-negative finite weights, or ``None`` for equal weights.
    levels : sequence of float
        Probability levels in ``[0, 1]``.
    order : numpy.ndarray or None, optional
        An ascending order of ``values`` when the caller already has one.

    Returns
    -------
    numpy.ndarray
        Quantiles at ``levels``.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if order is None:
        order = np.argsort(values, kind="stable")
    v = values[order]
    if weights is None:
        w = np.ones(v.size, dtype=np.float64)
    else:
        w = _normalized_nonnegative_weights(weights, values.size)[order]
        keep = w > 0.0
        v, w = v[keep], w[keep]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return np.interp(np.asarray(levels, dtype=np.float64), cdf, v)


def _support_kind(support, /):
    """Classify a validated two-sided support by endpoint finiteness.

    Parameters
    ----------
    support : tuple of (float, float)
        User-coordinate support with lower endpoint smaller than upper.

    Returns
    -------
    str
        One of ``real_line``, ``lower_half_line``, ``upper_half_line``,
        or ``bounded``.

    """
    lower, upper = support

    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    if lower_finite and upper_finite:
        return _BOUNDED
    if lower_finite:
        return _LOWER_HALF_LINE
    if upper_finite:
        return _UPPER_HALF_LINE
    return _REAL_LINE


def _identifies(center, scale, /):
    """Whether ``(center, scale)`` is a usable canonical map (positive finite scale).

    Parameters
    ----------
    center, scale : float
        Coordinate origin and scale in user units.
    """
    return bool(np.isfinite(scale) and scale > 0.0 and np.isfinite(center))


def _spread_suffices(physical_support, q10, q50, q90, /):
    """Whether ``q90 - q10`` is at least ``COORD_MIN_SPREAD`` of the median's distance.

    The distance is to the nearest finite end of ``physical_support``, which
    must have one.

    Parameters
    ----------
    physical_support : tuple of float
        Physical support endpoints (at least one finite).
    q10, q50, q90 : float
        Landmarks and median.
    """
    return _spread(q10, q90) >= COORD_MIN_SPREAD * _end_distance(physical_support, q50)


def _end_distance(physical_support, q50, /):
    """Distance from the median ``q50`` to the nearest finite support end.

    Parameters
    ----------
    physical_support : tuple of float
        Physical support endpoints (at least one finite).
    q50 : float
        Median.
    """
    # Halve before subtracting so landmarks near DBL_MAX do not overflow.
    return min(
        abs(0.5 * q50 - 0.5 * end) * 2.0 for end in physical_support if np.isfinite(end)
    )


def _spread(q10, q90, /):
    """``q90 - q10`` without overflow for landmarks near ``DBL_MAX``.

    Parameters
    ----------
    q10, q90 : float
        Landmarks.
    """
    return 2.0 * (0.5 * q90 - 0.5 * q10)


def _coordinate(kind, lower, upper, center, scale, /):
    """Package a canonical coordinate after checking its scale.

    Parameters
    ----------
    kind : str
        Support kind.
    lower, upper : float
        Physical support endpoints.
    center, scale : float
        Coordinate origin and scale in user units.

    Raises
    ------
    ValueError
        If the scale is not a positive finite number (landmarks do not
        identify the canonical map).
    """
    if not _identifies(center, scale):
        raise _CoordinateNotIdentified(
            "the canonical coordinate is not identified: the data landmarks "
            "q10 and q90 (or the landmark and the support endpoint) coincide"
        )
    canonical = _CANONICAL_SUPPORT[kind]
    return _FitCoordinate(
        support_kind=kind,
        center=float(center),
        scale=float(scale),
        canonical_support=canonical,
        physical_support=(float(lower), float(upper)),
    )


def _build_fit_coordinate(
    support, landmarks, weights=None, /, *, order=None, fallback=None
):
    """Fix the canonical coordinate of one fit.

    Parameters
    ----------
    support : tuple of (float, float)
        Validated support in user coordinates.
    landmarks : numpy.ndarray, shape (R,) or None
        Canonical finite point samples, interval midpoints or censoring
        cutpoints in user coordinates.  On a bounded support they only check
        the spread and may be ``None``.
    weights : numpy.ndarray, shape (R,) or None, optional
        Sample weights or mixture responsibilities.
    order : numpy.ndarray or None, optional
        Ascending order of ``landmarks`` when the caller has it (a mixture
        refits the same samples under many weightings).
    fallback : _FitCoordinate or None, optional
        Coordinate returned when the weighted landmarks do not identify one
        (see :func:`_from_landmarks`); ``None`` raises instead.

    Returns
    -------
    _FitCoordinate
        The canonical coordinate.

    Raises
    ------
    ValueError
        If the landmarks do not identify a positive scale, or the data are too
        concentrated for a finite-ended support, and no ``fallback`` is given.
    """
    if _support_kind(support) == _BOUNDED and landmarks is None:
        return _from_landmarks(support, None, None)
    q10, q50, q90 = _weighted_quantile(
        landmarks,
        weights,
        (COORD_LOWER_QUANTILE, 0.5, COORD_UPPER_QUANTILE),
        order=order,
    )
    return _from_landmarks(support, q10, q90, median=q50, fallback=fallback)


def _check_spread(coordinate, q10, q50, q90, /):
    """Reject data too concentrated far from a finite end.

    Data that hug a finite end are described by its endpoint logarithm; the
    failure is data whose spread is tiny against their distance from every
    finite end (they sit in a sliver of the canonical coordinate, where the
    polynomial coefficients cancel beyond binary64 and the gates dwarf them).

    Parameters
    ----------
    coordinate : _FitCoordinate
        Half-line or bounded coordinate.
    q10, q50, q90 : float
        Data landmarks and median.

    Raises
    ------
    ValueError
        If ``(q90 - q10)`` is below ``COORD_MIN_SPREAD`` times the median's
        distance from the nearest finite end.
    """
    if not _spread_suffices(coordinate.physical_support, q10, q50, q90):
        ends = coordinate.physical_support
        distance = _end_distance(ends, q50)
        spread = _spread(q10, q90)
        raise _CoordinateNotIdentified(
            "the data are too concentrated for the canonical coordinate of the "
            f"support {ends}: q90 - q10 = {spread:.6g} is "
            f"{spread / distance:.3g} of the median's distance {distance:.6g} "
            f"from the nearest finite end (at least {COORD_MIN_SPREAD:g} is "
            "required); fit on the full line or on a support whose finite end "
            "is nearer the data"
        )


def _from_landmarks(support, q10, q90, /, *, median=None, check=True, fallback=None):
    """The canonical coordinate of ``support`` from its landmarks ``q10, q90``.

    Parameters
    ----------
    support : tuple of (float, float)
        Validated support in user coordinates.
    q10, q90 : float or None
        Landmarks (``None`` only on a bounded support, which they do not
        place; they check the spread of finite-ended supports).
    median : float or None
        The data's median, for the spread check (``None``: no check).
    check : bool
        Apply the spread check (off when only one-sided censoring cutpoints
        carry location: they bound the data rather than sample them).
    fallback : _FitCoordinate or None
        Returned instead of raising when the landmarks do not identify a
        coordinate or fail the spread check.  Mixture components pass the
        pooled data's coordinate: responsibilities are derived weights, and
        a component holding one tied value or one censoring cutpoint has
        coincident weighted landmarks although the data identify the map.
    """
    kind = _support_kind(support)
    lower, upper = map(float, support)
    if kind == _BOUNDED:
        # Halve before combining so endpoints near DBL_MAX do not overflow.
        center, scale = 0.5 * lower + 0.5 * upper, 0.5 * upper - 0.5 * lower
    elif kind == _LOWER_HALF_LINE:
        center, scale = lower, float(q90) - lower
    elif kind == _UPPER_HALF_LINE:
        center, scale = upper, upper - float(q10)
    else:
        center, scale = 0.5 * float(q10) + 0.5 * float(q90), 0.5 * float(
            q90
        ) - 0.5 * float(q10)
    if fallback is not None and not _identifies(center, scale):
        return fallback
    out = _coordinate(kind, lower, upper, center, scale)
    if kind == _REAL_LINE or not check or median is None:
        return out
    if fallback is not None and not _spread_suffices(
        out.physical_support, float(q10), float(median), float(q90)
    ):
        return fallback
    _check_spread(out, float(q10), float(median), float(q90))
    return out


def _interval_quantiles(intervals, weights, levels, /):
    """Quantiles of the censored empirical distribution of interval rows.

    Each finite row of positive width spreads its weight uniformly over the
    row, a zero-width row is an atom, and a one-sided row is an atom at its
    finite cutpoint; whole-support rows carry no location and are dropped.
    The quantile at level ``p`` is the generalized inverse ``inf {x : F(x) >=
    p}`` of the resulting distribution function.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Ordered rows (infinite endpoints permitted).
    weights : numpy.ndarray, shape (R,)
        Nonnegative row weights.
    levels : sequence of float
        Probability levels in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray or None
        The quantiles, or ``None`` when no row carries location weight.
    """
    lo, hi = intervals[:, 0], intervals[:, 1]
    w = np.asarray(weights, dtype=np.float64)
    fin_lo, fin_hi = np.isfinite(lo), np.isfinite(hi)
    spread = fin_lo & fin_hi & (hi > lo) & (w > 0.0)
    atom = ((fin_lo & fin_hi & (hi == lo)) | (fin_lo ^ fin_hi)) & (w > 0.0)
    atoms = np.where(fin_lo, lo, hi)[atom]
    total = float(np.sum(w[spread]) + np.sum(w[atom]))
    if not total > 0.0:
        return None
    a, b, ws = lo[spread], hi[spread], w[spread] / total
    wa = w[atom] / total
    knots = np.unique(np.concatenate((a, b, atoms)))
    # F is piecewise linear between knots with jumps at atoms: accumulate the
    # slope changes and jumps knot by knot (O(R log R)).  Half-widths keep
    # the arithmetic finite for endpoints near DBL_MAX.
    half = 0.5 * b - 0.5 * a
    slope_change = np.zeros(knots.size)
    np.add.at(slope_change, np.searchsorted(knots, a), 0.5 * ws / half)
    np.add.at(slope_change, np.searchsorted(knots, b), -0.5 * ws / half)
    jump = np.zeros(knots.size)
    np.add.at(jump, np.searchsorted(knots, atoms), wa)
    slope = np.cumsum(slope_change)
    rise = np.zeros(knots.size)
    rise[1:] = 2.0 * slope[:-1] * (0.5 * knots[1:] - 0.5 * knots[:-1])
    f_right = np.minimum(np.cumsum(rise + jump), 1.0)
    f_left = np.minimum(f_right - jump, 1.0)
    out = []
    for p in levels:
        k = int(np.searchsorted(f_right, p, side="left"))
        k = min(k, knots.size - 1)
        if k == 0 or p > f_left[k]:
            out.append(float(knots[k]))
            continue
        # Linear between the previous knot (after its atom) and this one.
        f0, f1 = f_right[k - 1], f_left[k]
        t = 0.0 if f1 <= f0 else (p - f0) / (f1 - f0)
        out.append(float(knots[k - 1] + t * (knots[k] - knots[k - 1])))
    return np.array(out)


def _build_interval_fit_coordinate(
    support, intervals, weights=None, /, *, fallback=None
):
    """Fix the canonical coordinate from censored interval observations.

    The landmarks are the quantiles of the censored empirical distribution
    (:func:`_interval_quantiles`: each finite row's weight spread uniformly
    over it, zero-width and one-sided rows as atoms), so coarse bins locate
    ``q10`` and ``q90`` inside the bins that hold them rather than at a
    shared midpoint.  When every row is a point the data are point samples
    and the point rule applies (:func:`_build_fit_coordinate`).

    Parameters
    ----------
    support : tuple of (float, float)
        Density support in user coordinates.
    intervals : numpy.ndarray, shape (R, 2), dtype float64
        Boundary-validated ordered censoring intervals.  Infinite endpoints
        are permitted.
    weights : numpy.ndarray, shape (R,) or None, optional
        Canonical nonnegative row masses (sample weights or mixture
        responsibilities).
    fallback : _FitCoordinate or None, optional
        Coordinate returned when the weighted rows carry no finite landmark
        or their landmarks do not identify one (see :func:`_from_landmarks`);
        ``None`` raises instead.

    Returns
    -------
    _FitCoordinate
        The canonical coordinate.

    Raises
    ------
    ValueError
        If no row carries a finite landmark, or the landmarks do not identify
        a positive scale, and no ``fallback`` is given.
    """
    bounded = _support_kind(support) == _BOUNDED
    x = np.asarray(intervals, dtype=np.float64)
    w = (
        np.ones(x.shape[0], dtype=np.float64)
        if weights is None
        else _normalized_nonnegative_weights(weights, x.shape[0])
    )
    points = np.isfinite(x[:, 0]) & (x[:, 0] == x[:, 1])
    if np.all(points):
        return _build_fit_coordinate(support, x[:, 0], weights, fallback=fallback)
    quantiles = _interval_quantiles(
        x, w, (COORD_LOWER_QUANTILE, 0.5, COORD_UPPER_QUANTILE)
    )
    if quantiles is None and bounded:
        # (whole-support rows: a bounded coordinate needs no landmark)
        return _from_landmarks(support, None, None)
    if quantiles is None and fallback is not None:
        return fallback
    if quantiles is None:
        raise ValueError(
            "whole-support censoring contains no finite landmark for the "
            "canonical coordinate"
        )
    fin_lo, fin_hi = np.isfinite(x[:, 0]), np.isfinite(x[:, 1])
    sampled = np.any(fin_lo & fin_hi & (w > 0.0))
    return _from_landmarks(
        support,
        quantiles[0],
        quantiles[2],
        median=quantiles[1],
        check=bool(sampled),
        fallback=fallback,
    )
