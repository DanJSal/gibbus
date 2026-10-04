"""Fixed numerical coordinates used by the fitting pipeline.

The model works on a full line, a lower half-line, or a bounded interval
(an upper half-line is reflected), but the finite canonical endpoints need
not be fixed at ``0`` or ``+-1``.  This module chooses one data-centered
affine coordinate for every support type.  The transform is numerical
preconditioning, not an optimizer parameter.

The origin and robust scale are responsibility-weighted where component
weights are relevant.  The scale uses a Gaussian-consistent weighted MAD
about that origin, with overflow-safe weighted RMS fallback and a weighted
interval-width floor.  On bounded supports an additional support-span floor
prevents the two finite endpoints from being separated by an extreme number
of canonical units; this protects polynomial cone and moment calculations
without changing the affine model family.  Finite physical endpoints are then
mapped into this coordinate and carried by the model specification.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import MAD_TO_SIGMA, UNIFORM_WIDTH_TO_SIGMA

_REAL_LINE = "real_line"
_LOWER_HALF_LINE = "lower_half_line"
_UPPER_HALF_LINE = "upper_half_line"
_BOUNDED = "bounded"

# Bounded polynomial fits use raw canonical power moments internally.  Once
# the finite support spans vastly more than this many fitting units, those
# powers measure the arbitrary affine coordinate more than the model geometry
# and can overflow in otherwise estimable endpoint-concentrated problems.
# Applying a scale floor is an invertible affine reparameterization: it changes
# conditioning, not the represented density family.
_BOUNDED_CANONICAL_SPAN_MAX = 1.0e6


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
    """One fixed affine fitting coordinate.

    The forward map is

    ``z = direction * (x - center) / scale``

    with ``scale > 0`` and ``direction`` equal to ``+1`` or ``-1``.  The
    negative orientation is used only for an upper half-line so that, after
    reflection, every one-sided support is represented as ``[A, +inf)``.
    The finite endpoint ``A`` is generally data-dependent rather than fixed
    at zero.

    Parameters
    ----------
    support_kind : str
        One of the private support-kind constants in this module.
    center : float
        Coordinate origin in user units.
    scale : float
        Positive fixed numerical scale in user units.
    direction : float
        ``+1.0`` for the usual orientation or ``-1.0`` for reflection.
    canonical_support : tuple of (float, float)
        Support after transformation.  Finite endpoints are data-dependent
        under the robust affine coordinate.
    physical_support : tuple of (float, float)
        Original user-coordinate support.  Retained so exact observations can
        preserve sub-ulp physical distances from finite boundaries after the
        affine map.
    """

    support_kind: str
    center: float
    scale: float
    direction: float
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
        return self.direction * _safe_scaled_difference(values, self.center, self.scale)

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
        return self.center + self.direction * self.scale * z

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
        z = self.to_canonical(intervals)
        if self.direction < 0.0:
            z = z[:, ::-1]
        return np.ascontiguousarray(z, dtype=np.float64)


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


def _interval_scale_floor(widths, weights=None, /, *, order=None):
    """Return the within-bin scale floor for interval-censored data.

    Parameters
    ----------
    widths : numpy.ndarray or None
        Interval widths in user coordinates.  ``None`` means point data.

    weights : numpy.ndarray or None
        Optional row weights corresponding to ``widths``.
    order : numpy.ndarray or None, optional
        Ascending order of the finite widths when already available.

    Returns
    -------
    float
        Gaussian-scale floor implied by the median finite interval width,
        or zero when no usable widths are present.
    """
    if widths is None:
        return 0.0
    width = np.asarray(widths, dtype=np.float64).reshape(-1)
    finite = np.isfinite(width)
    width = width[finite]
    if width.size == 0:
        return 0.0
    if weights is None:
        typical = float(np.median(width))
    else:
        raw = np.asarray(weights, dtype=np.float64).reshape(-1)
        if raw.size != finite.size:
            raise ValueError(f"weights must have length {finite.size}, got {raw.size}")
        typical = _weighted_median(width, raw[finite], order=order)
    return UNIFORM_WIDTH_TO_SIGMA * typical


class _RobustScaleZero(ValueError):
    """The robust fitting-coordinate scale is numerically zero."""


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


def _build_fit_coordinate(
    support,
    point_samples,
    weights=None,
    widths=None,
    /,
    *,
    order=None,
    width_order=None,
):
    """Choose the fixed numerical coordinate for one fit.

    Parameters
    ----------
    support : tuple of (float, float)
        Validated support in user coordinates.
    point_samples : numpy.ndarray, shape (R,)
        Canonical finite point samples or derived finite interval midpoints
        in user coordinates.
    weights : numpy.ndarray, shape (R,) or None, optional
        Mixture responsibilities or sample weights.  They affect both the
        robust center and robust scale on every support type.
    widths : numpy.ndarray, shape (R,) or None, optional
        Interval widths.  When present, their within-bin spread supplies a
        lower bound on data-derived scales.
    order : numpy.ndarray or None, optional
        Ascending order of ``point_samples`` when the caller has it (a
        mixture refits the same samples under many weightings).
    width_order : numpy.ndarray or None, optional
        Ascending order of the finite interval widths when already available.
        This is independent of mixture responsibilities and can therefore be
        reused by every component/start for the same interval geometry.

    Returns
    -------
    _FitCoordinate
        Fixed data-centered transform.  Finite support endpoints are mapped
        into the returned canonical support and are not forced to prescribed
        values such as zero or ``+-1``.

    Raises
    ------
    ValueError
        If no positive data-derived scale is available.
    """
    kind = _support_kind(support)
    lower, upper = map(float, support)
    samples = point_samples

    center = _weighted_median(samples, weights, order=order)
    # Compute absolute deviations in a relative coordinate first so opposite
    # large finite values do not overflow during subtraction.  Multiplication
    # back by the reference magnitude is only needed for the weighted MAD.
    reference = max(1.0, abs(center), float(np.max(np.abs(samples))))
    rel_dev = np.abs(_safe_scaled_difference(samples, center, reference))
    deviation_order = None
    if order is not None:
        split = int(np.searchsorted(samples[order], center, side="left"))
        runs = np.concatenate((order[:split][::-1], order[split:]))
        deviation_order = runs[np.argsort(rel_dev[runs], kind="stable")]
    weighted_rel_mad = _weighted_median(rel_dev, weights, order=deviation_order)
    scale = MAD_TO_SIGMA * reference * weighted_rel_mad
    scale = max(scale, _interval_scale_floor(widths, weights, order=width_order))

    # A weighted MAD can vanish for a valid clustered sample when most weight
    # sits on one location.  Use an overflow-safe weighted RMS fallback before
    # declaring the component numerically degenerate.
    if not (np.isfinite(scale) and scale > 0.0):
        wn = _normalized_nonnegative_weights(weights, samples.size)
        if wn is None:
            wn = np.full(samples.size, 1.0 / samples.size, dtype=np.float64)
        max_rel = float(np.max(rel_dev))
        if max_rel > 0.0 and np.isfinite(max_rel):
            unit = rel_dev / max_rel
            rms_rel = max_rel * float(np.sqrt(np.dot(wn, unit * unit)))
            fallback = reference * rms_rel
            if np.isfinite(fallback) and fallback > 0.0:
                scale = fallback

    if not (np.isfinite(scale) and scale > 0.0):
        raise _RobustScaleZero(
            "Cannot construct fitting coordinate: robust scale is zero"
        )

    if kind == _BOUNDED:
        # Keep the complete finite support in a numerically moderate power
        # basis.  Compute (upper - lower) / cap without forming the possibly
        # overflowing physical span itself.  The center remains data-derived;
        # only the scale is enlarged when the robust local coordinate would
        # send the opposite endpoint millions of scales away.
        support_scale = float(
            _safe_scaled_difference(
                np.asarray([upper]), lower, _BOUNDED_CANONICAL_SPAN_MAX
            )[0]
        )
        if np.isfinite(support_scale) and support_scale > scale:
            scale = support_scale

    direction = -1.0 if kind == _UPPER_HALF_LINE else 1.0
    mapped_support = direction * _safe_scaled_difference(
        np.asarray([lower, upper]), center, scale
    )
    canonical_support = tuple(np.sort(mapped_support))

    return _FitCoordinate(
        support_kind=kind,
        center=float(center),
        scale=float(scale),
        direction=float(direction),
        canonical_support=canonical_support,
        physical_support=(float(lower), float(upper)),
    )


def _build_interval_fit_coordinate(support, intervals, weights=None, /):
    """Choose a fixed fitting coordinate from censored interval geometry.

    Finite rows contribute their midpoints, while one-sided infinite rows
    contribute their finite censoring cutpoint.  Whole-support rows carry no
    location/scale information and are ignored for coordinate construction.
    When all informative landmarks coincide, a finite model-support endpoint
    supplies the only principled scale fallback; a lone full-line cutpoint is
    rejected because the censored sample contains no affine scale information.

    Parameters
    ----------
    support : tuple of (float, float)
        Density support in user coordinates.
    intervals : numpy.ndarray, shape (R, 2), dtype float64
        Boundary-validated ordered censoring intervals. Infinite endpoints
        are permitted.
    weights : numpy.ndarray, shape (R,) or None, optional
        Canonical nonnegative row masses used for the robust center and scale.
        Their informative-landmark subset is normalized by the coordinate builder.

    Returns
    -------
    _FitCoordinate
        Data-centered coordinate suitable for exact censored-likelihood fitting.

    Raises
    ------
    ValueError
        If interval geometry contains insufficient finite
        information to define a numerical affine coordinate.
    """
    x = intervals
    n = x.shape[0]
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = weights

    finite_lo = np.isfinite(x[:, 0])
    finite_hi = np.isfinite(x[:, 1])
    finite = finite_lo & finite_hi
    left_inf = (~finite_lo) & finite_hi
    right_inf = finite_lo & (~finite_hi)

    landmarks = []
    landmark_weights = []
    width_floor = []
    if np.any(finite):
        lo_f = x[finite, 0]
        hi_f = x[finite, 1]
        # Halve before adding so same-sign endpoints near DBL_MAX do not
        # overflow while forming an otherwise representable midpoint.
        mid = 0.5 * lo_f + 0.5 * hi_f
        landmarks.extend(mid.tolist())
        landmark_weights.extend(w[finite].tolist())
        # Widths wider than DBL_MAX cannot be represented physically; cap
        # them at the largest finite scale rather than turning a numerical
        # preconditioner into infinity.
        ref = np.maximum(np.maximum(np.abs(lo_f), np.abs(hi_f)), 1.0)
        unit_width = np.abs(lo_f / ref - hi_f / ref)
        max_float = np.finfo(np.float64).max
        widths_f = np.minimum(ref * np.minimum(unit_width, max_float / ref), max_float)
        width_floor.extend(widths_f.tolist())
    if np.any(left_inf):
        landmarks.extend(x[left_inf, 1].tolist())
        landmark_weights.extend(w[left_inf].tolist())
        width_floor.extend([np.nan] * int(np.sum(left_inf)))
    if np.any(right_inf):
        landmarks.extend(x[right_inf, 0].tolist())
        landmark_weights.extend(w[right_inf].tolist())
        width_floor.extend([np.nan] * int(np.sum(right_inf)))

    if not landmarks:
        raise ValueError(
            "whole-support censoring contains no finite landmark for fitting coordinates"
        )

    landmarks = np.asarray(landmarks, dtype=np.float64)
    landmark_weights = np.asarray(landmark_weights, dtype=np.float64)
    widths = np.asarray(width_floor, dtype=np.float64)
    scale_error = None
    try:
        return _build_fit_coordinate(support, landmarks, landmark_weights, widths)
    except _RobustScaleZero as exc:
        scale_error = exc

    kind = _support_kind(support)
    lower, upper = map(float, support)
    center = _weighted_median(landmarks, landmark_weights)
    candidates = []
    finite_widths = widths[np.isfinite(widths) & (widths > 0.0)]
    if finite_widths.size:
        candidates.append(UNIFORM_WIDTH_TO_SIGMA * float(np.median(finite_widths)))
    spread = float(np.ptp(landmarks))
    if np.isfinite(spread) and spread > 0.0:
        candidates.append(spread)
    if np.isfinite(lower):
        distance = abs(center - lower)
        if distance > 0.0:
            candidates.append(distance)
    if np.isfinite(upper):
        distance = abs(upper - center)
        if distance > 0.0:
            candidates.append(distance)
    if not candidates:
        raise ValueError(
            "censoring geometry does not identify a finite numerical scale on this support"
        ) from scale_error

    scale = float(np.median(np.asarray(candidates, dtype=np.float64)))
    if not (np.isfinite(scale) and scale > 0.0):
        raise ValueError(
            "cannot construct a positive censored-data fitting scale"
        ) from scale_error

    direction = -1.0 if kind == _UPPER_HALF_LINE else 1.0
    mapped_support = direction * _safe_scaled_difference(
        np.asarray([lower, upper]), center, scale
    )
    canonical_support = tuple(np.sort(mapped_support))
    return _FitCoordinate(
        support_kind=kind,
        center=float(center),
        scale=float(scale),
        direction=float(direction),
        canonical_support=canonical_support,
        physical_support=(float(lower), float(upper)),
    )
