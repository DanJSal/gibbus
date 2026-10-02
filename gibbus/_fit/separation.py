"""Algebraic primitives for full-curvature separation in natural coordinates.

The separation oracle works with two denominator-cleared polynomials.  ``R``
preserves the sign of the full curvature inside the support, while ``S`` has
the same interior zeros as the derivative of the full curvature.  Boundary
denominators are included only when the corresponding amplitude is strictly
positive; an exact zero amplitude is therefore a distinct algebraic case.
"""

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.polynomial.polynomial import polyadd, polyder, polymul


@dataclass(frozen=True)
class _CurvaturePolynomials:
    """Denominator-cleared full-curvature polynomials.

    Parameters
    ----------
    sign_coefficients : numpy.ndarray
        Ascending coefficients of ``R`` whose sign matches full curvature in
        the support interior.
    stationary_coefficients : numpy.ndarray
        Ascending coefficients of ``S`` whose interior zeros are exactly the
        stationary points of the full curvature.
    lower_active, upper_active : bool
        Whether the corresponding strictly-positive boundary amplitude was
        included in the denominator clearing.
    """

    sign_coefficients: np.ndarray
    stationary_coefficients: np.ndarray
    lower_active: bool
    upper_active: bool


def _trim_exact(coefficients, /):
    """Remove only exact trailing zeros from one ascending polynomial.

    Parameters
    ----------
    coefficients : array_like
        Ascending polynomial coefficients.

    Returns
    -------
    numpy.ndarray
        A nonempty coefficient array.  No magnitude threshold is used.
    """
    coeffs = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if coeffs.size == 0:
        return np.zeros(1, dtype=np.float64)
    stop = coeffs.size
    while stop > 1 and coeffs[stop - 1] == 0.0:
        stop -= 1
    return coeffs[:stop].copy()


def _fraction_trim(coefficients, /):
    """Remove exact trailing zeros from one ascending rational polynomial.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    """
    values = list(coefficients)
    if not values:
        return [Fraction(0)]
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def _as_exact_fraction(value, /):
    """Return the exact rational value represented by one finite float.

    Parameters
    ----------
    value : float
        Finite binary64 value.
    """
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("exact polynomial arithmetic requires finite values")
    return Fraction.from_float(scalar)


def _fraction_polyadd(left, right, /):
    """Add ascending rational polynomials exactly.

    Parameters
    ----------
    left, right : sequence of Fraction
        Ascending exact coefficients.
    """
    size = max(len(left), len(right))
    out = [Fraction(0)] * size
    for index, value in enumerate(left):
        out[index] += value
    for index, value in enumerate(right):
        out[index] += value
    return _fraction_trim(out)


def _fraction_polymul(left, right, /):
    """Multiply ascending rational polynomials exactly.

    Parameters
    ----------
    left, right : sequence of Fraction
        Ascending exact coefficients.
    """
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, left_value in enumerate(left):
        for j, right_value in enumerate(right):
            out[i + j] += left_value * right_value
    return _fraction_trim(out)


def _fraction_polyder(coefficients, /):
    """Differentiate one ascending rational polynomial exactly.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    """
    if len(coefficients) <= 1:
        return [Fraction(0)]
    return _fraction_trim(
        [Fraction(index) * coefficients[index] for index in range(1, len(coefficients))]
    )


def _fraction_polyval(coefficients, point, /):
    """Evaluate one ascending rational polynomial exactly.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    point : Fraction
        Exact evaluation point.
    """
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def _fraction_power_on_interval(coefficients, lower, upper, /):
    """Compose a rational power polynomial with ``z=lower+(upper-lower)t``.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients in ``z``.
    lower, upper : Fraction
        Interval mapped onto ``t in [0, 1]``.
    """
    affine = [lower, upper - lower]
    out = [Fraction(0)]
    for coefficient in reversed(_fraction_trim(coefficients)):
        out = _fraction_polymul(out, affine)
        out[0] += coefficient
    return _fraction_trim(out)


def _fraction_power_to_bernstein(coefficients, degree, /):
    """Convert unit-interval power coefficients to Bernstein form exactly.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending unit-interval power coefficients.
    degree : int
        Bernstein degree, at least the polynomial degree.
    """
    power = _fraction_trim(coefficients)
    n = int(degree)
    if n < len(power) - 1:
        raise ValueError("Bernstein degree is smaller than polynomial degree")
    padded = power + [Fraction(0)] * (n + 1 - len(power))
    result = []
    for i in range(n + 1):
        total = Fraction(0)
        for k in range(i + 1):
            total += padded[k] * Fraction(math.comb(i, k), math.comb(n, k))
        result.append(total)
    return result


def _fraction_split_bernstein_half(coefficients, /):
    """Split exact Bernstein coefficients at the interval midpoint.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Exact Bernstein coefficients on one interval.
    """
    current = list(coefficients)
    if not current:
        raise ValueError("Bernstein coefficients must be nonempty")
    n = len(current)
    left = [Fraction(0)] * n
    right = [Fraction(0)] * n
    left[0] = current[0]
    right[-1] = current[-1]
    for level in range(1, n):
        current = [
            (current[index] + current[index + 1]) / 2
            for index in range(len(current) - 1)
        ]
        left[level] = current[0]
        right[-level - 1] = current[-1]
    return left, right


def _fraction_sign_variations(coefficients, /):
    """Count Bernstein sign changes exactly after deleting exact zeros.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Exact Bernstein coefficients.
    """
    signs = [1 if value > 0 else -1 for value in coefficients if value != 0]
    return sum(left != right for left, right in itertools.pairwise(signs))


def _fraction_to_float_lower(value, /):
    """Convert a finite rational to a binary64 lower enclosure.

    Parameters
    ----------
    value : Fraction
        Finite exact rational.
    """
    try:
        result = float(value)
    except OverflowError:
        return float(np.finfo(np.float64).max if value > 0 else -np.inf)
    if Fraction.from_float(result) > value:
        result = float(np.nextafter(result, -np.inf))
    return result


def _fraction_to_float_upper(value, /):
    """Convert a finite rational to a binary64 upper enclosure.

    Parameters
    ----------
    value : Fraction
        Finite exact rational.
    """
    try:
        result = float(value)
    except OverflowError:
        return float(np.inf if value > 0 else -np.finfo(np.float64).max)
    if Fraction.from_float(result) < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _exact_float_polynomial(coefficients, /):
    """Lift finite binary64 polynomial coefficients to their exact dyadics.

    Parameters
    ----------
    coefficients : array_like
        Finite binary64 ascending coefficients.
    """
    return _fraction_trim(
        [_as_exact_fraction(value) for value in _trim_exact(coefficients)]
    )


def _curvature_polynomials(q_d2, support, boundary_amplitudes, /):
    """Construct the support-specific denominator-cleared ``R`` and ``S``.

    Parameters
    ----------
    q_d2 : array_like
        Ascending coefficients of the ordinary curvature polynomial ``p``.
    support : array_like, shape (2,)
        Canonical support ``[L, U]``.
    boundary_amplitudes : array_like, shape (2,)
        Canonical ``[a_L, a_U]``.  ``NaN`` denotes a disabled boundary basis,
        zero denotes an enabled but inactive bound, and every strictly
        positive finite value activates its reciprocal-square term.

    Returns
    -------
    _CurvaturePolynomials
        Denominator-cleared sign and stationary polynomials.

    Raises
    ------
    ValueError
        If the support/amplitude shapes are invalid, an amplitude is negative,
        or a finite amplitude is supplied for an infinite endpoint.
    """
    p = _trim_exact(q_d2)
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if bounds.size != 2 or np.any(np.isnan(bounds)) or not bounds[0] < bounds[1]:
        raise ValueError("support must contain two increasing non-NaN endpoints")
    if amplitudes.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")

    lower, upper = map(float, bounds)
    a_lower, a_upper = map(float, amplitudes)
    for amplitude in (a_lower, a_upper):
        if np.isfinite(amplitude) and amplitude < 0.0:
            raise ValueError("boundary amplitudes must be >= 0")
    if np.isfinite(a_lower) and not np.isfinite(lower):
        raise ValueError("lower boundary amplitude requires a finite endpoint")
    if np.isfinite(a_upper) and not np.isfinite(upper):
        raise ValueError("upper boundary amplitude requires a finite endpoint")

    lower_active = np.isfinite(a_lower) and a_lower > 0.0
    upper_active = np.isfinite(a_upper) and a_upper > 0.0
    p_d1 = _trim_exact(polyder(p))

    lower_distance = np.array([-lower, 1.0], dtype=np.float64)
    upper_distance = np.array([upper, -1.0], dtype=np.float64)

    if lower_active and upper_active:
        lower_sq = polymul(lower_distance, lower_distance)
        upper_sq = polymul(upper_distance, upper_distance)
        lower_cube = polymul(lower_sq, lower_distance)
        upper_cube = polymul(upper_sq, upper_distance)
        sign = polymul(polymul(lower_sq, upper_sq), p)
        sign = polyadd(sign, a_lower * upper_sq)
        sign = polyadd(sign, a_upper * lower_sq)
        stationary = polymul(polymul(lower_cube, upper_cube), p_d1)
        stationary = polyadd(stationary, -2.0 * a_lower * upper_cube)
        stationary = polyadd(stationary, 2.0 * a_upper * lower_cube)
    elif lower_active:
        lower_sq = polymul(lower_distance, lower_distance)
        lower_cube = polymul(lower_sq, lower_distance)
        sign = polyadd(polymul(lower_sq, p), np.array([a_lower]))
        stationary = polyadd(polymul(lower_cube, p_d1), np.array([-2.0 * a_lower]))
    elif upper_active:
        upper_sq = polymul(upper_distance, upper_distance)
        upper_cube = polymul(upper_sq, upper_distance)
        sign = polyadd(polymul(upper_sq, p), np.array([a_upper]))
        stationary = polyadd(polymul(upper_cube, p_d1), np.array([2.0 * a_upper]))
    else:
        sign = p.copy()
        stationary = p_d1.copy()

    return _CurvaturePolynomials(
        sign_coefficients=_trim_exact(sign),
        stationary_coefficients=_trim_exact(stationary),
        lower_active=bool(lower_active),
        upper_active=bool(upper_active),
    )


def _exact_curvature_polynomials(q_d2, support, boundary_amplitudes, /):
    """Construct exact-dyadic ``R`` and ``S`` from binary64 model inputs.

    The returned rational polynomials represent the mathematical formulas
    applied to the exact real values encoded by the input floats.  They are
    used only for certification decisions; ordinary float polynomials remain
    the evaluation/optimization representation.

    Parameters
    ----------
    q_d2 : numpy.ndarray
        Ascending ordinary curvature coefficients.
    support : numpy.ndarray, shape (2,)
        Canonical support endpoints.
    boundary_amplitudes : numpy.ndarray, shape (2,)
        Lower/upper logarithmic amplitudes; ``nan`` when absent.
    """
    p_float = _trim_exact(q_d2)
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if bounds.size != 2 or np.any(np.isnan(bounds)) or not bounds[0] < bounds[1]:
        raise ValueError("support must contain two increasing non-NaN endpoints")
    if amplitudes.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")

    lower, upper = map(float, bounds)
    a_lower, a_upper = map(float, amplitudes)
    for amplitude in (a_lower, a_upper):
        if np.isfinite(amplitude) and amplitude < 0.0:
            raise ValueError("boundary amplitudes must be >= 0")
    if np.isfinite(a_lower) and not np.isfinite(lower):
        raise ValueError("lower boundary amplitude requires a finite endpoint")
    if np.isfinite(a_upper) and not np.isfinite(upper):
        raise ValueError("upper boundary amplitude requires a finite endpoint")

    lower_active = np.isfinite(a_lower) and a_lower > 0.0
    upper_active = np.isfinite(a_upper) and a_upper > 0.0
    p = [_as_exact_fraction(value) for value in p_float]
    p_d1 = _fraction_polyder(p)

    lower_distance = None
    upper_distance = None
    if np.isfinite(lower):
        lower_distance = [-_as_exact_fraction(lower), Fraction(1)]
    if np.isfinite(upper):
        upper_distance = [_as_exact_fraction(upper), Fraction(-1)]

    if lower_active and upper_active:
        lower_sq = _fraction_polymul(lower_distance, lower_distance)
        upper_sq = _fraction_polymul(upper_distance, upper_distance)
        lower_cube = _fraction_polymul(lower_sq, lower_distance)
        upper_cube = _fraction_polymul(upper_sq, upper_distance)
        sign = _fraction_polymul(_fraction_polymul(lower_sq, upper_sq), p)
        sign = _fraction_polyadd(
            sign, [value * _as_exact_fraction(a_lower) for value in upper_sq]
        )
        sign = _fraction_polyadd(
            sign, [value * _as_exact_fraction(a_upper) for value in lower_sq]
        )
        stationary = _fraction_polymul(_fraction_polymul(lower_cube, upper_cube), p_d1)
        stationary = _fraction_polyadd(
            stationary,
            [value * (-2 * _as_exact_fraction(a_lower)) for value in upper_cube],
        )
        stationary = _fraction_polyadd(
            stationary,
            [value * (2 * _as_exact_fraction(a_upper)) for value in lower_cube],
        )
    elif lower_active:
        lower_sq = _fraction_polymul(lower_distance, lower_distance)
        lower_cube = _fraction_polymul(lower_sq, lower_distance)
        sign = _fraction_polyadd(
            _fraction_polymul(lower_sq, p), [_as_exact_fraction(a_lower)]
        )
        stationary = _fraction_polyadd(
            _fraction_polymul(lower_cube, p_d1),
            [-2 * _as_exact_fraction(a_lower)],
        )
    elif upper_active:
        upper_sq = _fraction_polymul(upper_distance, upper_distance)
        upper_cube = _fraction_polymul(upper_sq, upper_distance)
        sign = _fraction_polyadd(
            _fraction_polymul(upper_sq, p), [_as_exact_fraction(a_upper)]
        )
        stationary = _fraction_polyadd(
            _fraction_polymul(upper_cube, p_d1),
            [2 * _as_exact_fraction(a_upper)],
        )
    else:
        sign = p
        stationary = p_d1

    return (
        _fraction_trim(sign),
        _fraction_trim(stationary),
        bool(lower_active),
        bool(upper_active),
    )


@dataclass(frozen=True)
class _CurvatureBoundaryValues:
    """Full-curvature endpoint and infinite-tail limits.

    Parameters
    ----------
    lower_endpoint, upper_endpoint : float or None
        Full-curvature limits at finite support endpoints.  ``+inf`` denotes
        the reciprocal-square singularity from a positive amplitude; ``None``
        denotes an infinite support endpoint.
    lower_tail, upper_tail : float or None
        Ordinary-curvature limits at ``-inf`` and ``+inf`` respectively.
        Boundary terms vanish at infinity.  A finite value occurs only for an
        effectively constant ordinary curvature; otherwise the value is an
        infinity with the algebraic tail sign.  ``None`` denotes a finite
        support endpoint.
    """

    lower_endpoint: float | None
    upper_endpoint: float | None
    lower_tail: float | None
    upper_tail: float | None


def _polynomial_tail_limit(q_d2, positive_infinity, /):
    """Return one exact-algebraic infinite-tail limit of the curvature polynomial.

    Parameters
    ----------
    q_d2 : array_like
        Ascending ordinary-curvature coefficients.
    positive_infinity : bool
        Select ``+inf`` when true and ``-inf`` otherwise.

    Returns
    -------
    float
        Finite constant for effective degree zero, otherwise ``+/-inf`` from
        the exact nonzero leading coefficient and tail parity.
    """
    p = _trim_exact(q_d2)
    degree = p.size - 1
    if degree == 0:
        return float(p[0])
    leading = float(p[-1])
    sign = np.sign(leading)
    if not positive_infinity and degree % 2:
        sign = -sign
    return float(np.inf if sign > 0.0 else -np.inf)


def _curvature_boundary_values(q_d2, support, boundary_amplitudes, /):
    """Evaluate regular endpoints and certify algebraic infinite-tail signs.

    Parameters
    ----------
    q_d2 : array_like
        Ascending ordinary-curvature coefficients.
    support : array_like, shape (2,)
        Canonical support ``[L, U]``.
    boundary_amplitudes : array_like, shape (2,)
        Canonical boundary amplitudes using the same ``NaN``/zero/positive
        semantics as :func:`_curvature_polynomials`.

    Returns
    -------
    _CurvatureBoundaryValues
        Endpoint and tail limits needed by the separator.
    """
    cleared = _curvature_polynomials(q_d2, support, boundary_amplitudes)
    p = _trim_exact(q_d2)
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    lower, upper = map(float, bounds)
    a_lower, a_upper = map(float, amplitudes)

    lower_endpoint = None
    lower_tail = None
    if np.isfinite(lower):
        if cleared.lower_active:
            lower_endpoint = float(np.inf)
        else:
            lower_endpoint = float(np.polynomial.polynomial.polyval(lower, p))
            if cleared.upper_active:
                lower_endpoint += a_upper / (upper - lower) ** 2
    else:
        lower_tail = _polynomial_tail_limit(p, False)

    upper_endpoint = None
    upper_tail = None
    if np.isfinite(upper):
        if cleared.upper_active:
            upper_endpoint = float(np.inf)
        else:
            upper_endpoint = float(np.polynomial.polynomial.polyval(upper, p))
            if cleared.lower_active:
                upper_endpoint += a_lower / (upper - lower) ** 2
    else:
        upper_tail = _polynomial_tail_limit(p, True)

    return _CurvatureBoundaryValues(
        lower_endpoint=lower_endpoint,
        upper_endpoint=upper_endpoint,
        lower_tail=lower_tail,
        upper_tail=upper_tail,
    )


def _fraction_bernstein_interval_bounds(
    coefficients,
    lower,
    upper,
    max_subdivide=0,
    /,
):
    """Return exact Bernstein convex-hull bounds for a rational polynomial.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    lower, upper : Fraction
        Exact interval endpoints.
    max_subdivide : int
        Number of midpoint subdivision levels used to tighten the hull.
    """
    depth = int(max_subdivide)
    if depth < 0:
        raise ValueError("max_subdivide must be >= 0")
    if not lower < upper:
        raise ValueError("Bernstein interval must have increasing endpoints")
    power = _fraction_power_on_interval(coefficients, lower, upper)
    degree = max(0, len(_fraction_trim(coefficients)) - 1)
    initial = _fraction_power_to_bernstein(power, degree)
    stack = [(initial, 0)]
    lower_bound = None
    upper_bound = None
    while stack:
        current, current_depth = stack.pop()
        if current_depth < depth:
            left, right = _fraction_split_bernstein_half(current)
            stack.append((right, current_depth + 1))
            stack.append((left, current_depth + 1))
            continue
        current_lower = min(current)
        current_upper = max(current)
        lower_bound = (
            current_lower if lower_bound is None else min(lower_bound, current_lower)
        )
        upper_bound = (
            current_upper if upper_bound is None else max(upper_bound, current_upper)
        )
    return lower_bound, upper_bound


@dataclass(frozen=True)
class _RootBracket:
    """One finite interval retained by Bernstein stationary-root isolation.

    Parameters
    ----------
    lower, upper : float
        Finite bracket endpoints.
    sign_variations : int
        Bernstein sign-variation upper bound for roots in the bracket.  One
        certifies a unique simple-count root under exact arithmetic; larger
        values are retained conservatively when a repeated/clustered root
        cannot be separated before the requested width/depth limit.
    exact_lower, exact_upper : fractions.Fraction or None
        Internal exact rational bracket endpoints used by the separator.  The
        public-style float endpoints above are outward-rounded enclosures.
    """

    lower: float
    upper: float
    sign_variations: int
    exact_lower: Fraction | None = field(default=None, repr=False, compare=False)
    exact_upper: Fraction | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class _RootIsolation:
    """Stationary-root isolation result over one support.

    Parameters
    ----------
    brackets : tuple of _RootBracket
        Retained finite root-containing or conservatively ambiguous intervals.
    identically_zero : bool
        Whether the stationary polynomial is exactly the zero polynomial.
    subdivisions : int
        Number of midpoint subdivisions performed.
    """

    brackets: tuple
    identically_zero: bool
    subdivisions: int


def _fraction_polynomial_root_bound(coefficients, /):
    """Return the exact Cauchy radius bound for a rational polynomial.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    """
    coeffs = _fraction_trim(coefficients)
    if len(coeffs) <= 1:
        return Fraction(0)
    leading = abs(coeffs[-1])
    if leading == 0:
        raise ValueError("root-bound polynomial must have nonzero leading coefficient")
    return Fraction(1) + max(
        (abs(value) / leading for value in coeffs[:-1]), default=Fraction(0)
    )


def _stationary_root_brackets_exact(
    coefficients,
    support,
    max_width,
    max_depth,
    /,
):
    """Isolate roots of one exact rational polynomial on a binary64 support.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    support : numpy.ndarray, shape (2,)
        Binary64 support endpoints.
    max_width : float
        Target bracket width.
    max_depth : int
        Maximum bisection depth.
    """
    coeffs = _fraction_trim(coefficients)
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    if bounds.size != 2 or np.any(np.isnan(bounds)) or not bounds[0] < bounds[1]:
        raise ValueError("support must contain two increasing non-NaN endpoints")
    width_limit = float(max_width)
    depth_limit = int(max_depth)
    if not np.isfinite(width_limit) or width_limit <= 0.0:
        raise ValueError("max_width must be positive and finite")
    if depth_limit < 0:
        raise ValueError("max_depth must be >= 0")
    if len(coeffs) == 1:
        return _RootIsolation((), bool(coeffs[0] == 0), 0)

    radius = _fraction_polynomial_root_bound(coeffs)
    support_lower, support_upper = map(float, bounds)
    lower = (
        max(_as_exact_fraction(support_lower), -radius)
        if np.isfinite(support_lower)
        else -radius
    )
    upper = (
        min(_as_exact_fraction(support_upper), radius)
        if np.isfinite(support_upper)
        else radius
    )
    if not lower < upper:
        return _RootIsolation((), False, 0)

    power = _fraction_power_on_interval(coeffs, lower, upper)
    bernstein = _fraction_power_to_bernstein(power, len(coeffs) - 1)
    stack = [(bernstein, lower, upper, 0)]
    exact_brackets = []
    exact_root_points = set()
    if _fraction_polyval(coeffs, lower) == 0:
        exact_root_points.add(lower)
    if _fraction_polyval(coeffs, upper) == 0:
        exact_root_points.add(upper)
    subdivisions = 0
    exact_width_limit = _as_exact_fraction(width_limit)

    while stack:
        current, left_endpoint, right_endpoint, depth = stack.pop()
        variations = _fraction_sign_variations(current)
        if variations == 0:
            continue
        if variations == 1:
            exact_brackets.append((left_endpoint, right_endpoint, 1))
            continue
        if (
            right_endpoint - left_endpoint <= exact_width_limit
        ) or depth >= depth_limit:
            exact_brackets.append((left_endpoint, right_endpoint, int(variations)))
            continue
        left_coeffs, right_coeffs = _fraction_split_bernstein_half(current)
        midpoint = (left_endpoint + right_endpoint) / 2
        if left_coeffs[-1] == 0 and right_coeffs[0] == 0:
            exact_root_points.add(midpoint)
        subdivisions += 1
        stack.append((right_coeffs, midpoint, right_endpoint, depth + 1))
        stack.append((left_coeffs, left_endpoint, midpoint, depth + 1))

    endpoint_candidates = {lower, upper, *exact_root_points}
    for left_endpoint, right_endpoint, _ in exact_brackets:
        endpoint_candidates.add(left_endpoint)
        endpoint_candidates.add(right_endpoint)
    for point in sorted(endpoint_candidates):
        if _fraction_polyval(coeffs, point) != 0:
            continue
        if any(
            (left < point < right) or (left == point == right)
            for left, right, _ in exact_brackets
        ):
            continue
        exact_brackets.append((point, point, 1))

    exact_brackets.sort(key=lambda item: (item[0], item[1]))
    merged = []
    for bracket in exact_brackets:
        if merged and bracket[0] == merged[-1][0] and bracket[1] == merged[-1][1]:
            previous = merged[-1]
            merged[-1] = (previous[0], previous[1], max(previous[2], bracket[2]))
        else:
            merged.append(bracket)

    brackets = tuple(
        _RootBracket(
            _fraction_to_float_lower(left),
            _fraction_to_float_upper(right),
            variations,
            left,
            right,
        )
        for left, right, variations in merged
    )
    return _RootIsolation(brackets, False, subdivisions)


def _refine_unique_root_bracket_exact(
    coefficients,
    bracket,
    max_width,
    max_depth,
    /,
):
    """Refine one certified unique-root bracket by exact Bernstein bisection.

    Root-aware active-set bookkeeping needs a stationary-root location, whereas
    the separator itself may stop as soon as a broad interval is certified to
    contain one root.  Endpoint and midpoint roots are handled explicitly so a
    dyadic root on a subdivision boundary is not lost by sign-variation counts.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    bracket : object
        Certified bracket containing exactly one root.
    max_width : float
        Target bracket width.
    max_depth : int
        Maximum bisection depth.
    """
    if int(bracket.sign_variations) != 1:
        return bracket, False

    width_limit = float(max_width)
    depth_limit = int(max_depth)
    if not np.isfinite(width_limit) or width_limit <= 0.0:
        raise ValueError("max_width must be positive and finite")
    if depth_limit < 0:
        raise ValueError("max_depth must be >= 0")

    coeffs = _fraction_trim(coefficients)
    left = (
        bracket.exact_lower
        if bracket.exact_lower is not None
        else Fraction.from_float(float(bracket.lower))
    )
    right = (
        bracket.exact_upper
        if bracket.exact_upper is not None
        else Fraction.from_float(float(bracket.upper))
    )
    if left > right:
        raise ValueError("root bracket endpoints must be increasing")
    if left == right:
        return bracket, bool(_fraction_polyval(coeffs, left) == 0)

    degree = len(coeffs) - 1
    power = _fraction_power_on_interval(coeffs, left, right)
    current = _fraction_power_to_bernstein(power, degree)
    if _fraction_sign_variations(current) != 1:
        return bracket, False

    exact_width_limit = _as_exact_fraction(width_limit)
    depth = 0
    while right - left > exact_width_limit and depth < depth_limit:
        left_coeffs, right_coeffs = _fraction_split_bernstein_half(current)
        midpoint = (left + right) / 2
        left_variations = _fraction_sign_variations(left_coeffs)
        right_variations = _fraction_sign_variations(right_coeffs)

        if left_variations == 1 and right_variations == 0:
            current = left_coeffs
            right = midpoint
        elif left_variations == 0 and right_variations == 1:
            current = right_coeffs
            left = midpoint
        elif (
            left_variations == 0
            and right_variations == 0
            and _fraction_polyval(coeffs, midpoint) == 0
        ):
            left = midpoint
            right = midpoint
            break
        else:
            refined = _RootBracket(
                _fraction_to_float_lower(left),
                _fraction_to_float_upper(right),
                1,
                left,
                right,
            )
            return refined, False
        depth += 1

    refined = _RootBracket(
        _fraction_to_float_lower(left),
        _fraction_to_float_upper(right),
        1,
        left,
        right,
    )
    return refined, bool(left == right or right - left <= exact_width_limit)


def _stationary_minimum_locations_exact(
    q_d2,
    support,
    boundary_amplitudes,
    root_width,
    max_root_depth=80,
):
    """Return refined isolated stationary minima for active-set bookkeeping.

    Returns ``(locations, complete)``.  ``complete`` is false when a repeated
    or clustered stationary bracket cannot be resolved under the requested
    work limits.  Feasibility never depends on these locations; the certified
    separator remains authoritative.

    Parameters
    ----------
    q_d2 : numpy.ndarray
        Ascending ordinary curvature coefficients.
    support : numpy.ndarray, shape (2,)
        Canonical support endpoints.
    boundary_amplitudes : numpy.ndarray, shape (2,)
        Lower/upper logarithmic amplitudes; ``nan`` when absent.
    root_width : float
        Target root bracket width.
    max_root_depth : int
        Maximum bisection depth.
    """
    _, stationary, _, _ = _exact_curvature_polynomials(
        q_d2, support, boundary_amplitudes
    )
    isolated = _stationary_root_brackets_exact(
        stationary, support, root_width, max_root_depth
    )
    if isolated.identically_zero:
        return (), True

    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    exact_lower = _as_exact_fraction(bounds[0]) if np.isfinite(bounds[0]) else None
    exact_upper = _as_exact_fraction(bounds[1]) if np.isfinite(bounds[1]) else None

    minima = []
    complete = True
    for bracket in isolated.brackets:
        refined, resolved = _refine_unique_root_bracket_exact(
            stationary, bracket, root_width, max_root_depth
        )
        if not resolved:
            complete = False
            continue

        left = refined.exact_lower
        right = refined.exact_upper
        if left is None or right is None:
            complete = False
            continue

        if left == right:
            root = left
            if exact_lower is not None and root == exact_lower:
                continue
            if exact_upper is not None and root == exact_upper:
                continue

            derivative = list(stationary)
            order = 0
            leading = Fraction(0)
            while derivative:
                value = _fraction_polyval(derivative, root)
                if value != 0:
                    leading = value
                    break
                derivative = _fraction_polyder(derivative)
                order += 1
            is_minimum = order % 2 == 1 and leading > 0
            location_exact = root
        else:
            f_left = _fraction_polyval(stationary, left)
            f_right = _fraction_polyval(stationary, right)
            if f_left == 0 or f_right == 0 or f_left * f_right >= 0:
                complete = False
                continue
            is_minimum = f_left < 0 and f_right > 0
            location_exact = (left + right) / 2

        if not is_minimum:
            continue
        location = float(location_exact)
        lower, upper = map(float, bounds)
        if not (location > lower and location < upper):
            complete = False
            continue
        minima.append(location)

    minima.sort()
    return tuple(minima), bool(complete)


@dataclass(frozen=True)
class _SeparationResult:
    """Result of one full-curvature separation pass.

    Parameters
    ----------
    status : str
        ``"feasible"``, ``"violated"``, or ``"uncertain"``.
    violation_kind : str or None
        ``"stationary"``, endpoint, or tail label for a material violation.
    violation_location : float or None
        Finite cut location when one is certified.  Tail violations have no
        finite location.
    violation_value : float or None
        Full curvature at a returned finite violation location, or the finite
        tail value when applicable.
    isolated_roots, discarded_maxima, pruned_intervals, refined_intervals : int
        Separator diagnostics for profiling the lazy-refinement behavior.
    ambiguous_intervals : int
        Number of root brackets that remained sign-ambiguous at the requested
        width/depth limits.
    root_subdivisions : int
        Bernstein subdivisions used by initial root isolation.
    """

    status: str
    violation_kind: str | None
    violation_location: float | None
    violation_value: float | None
    isolated_roots: int
    discarded_maxima: int
    pruned_intervals: int
    refined_intervals: int
    ambiguous_intervals: int
    root_subdivisions: int

    @property
    def feasible(self):
        """Return whether the pass certified feasibility under its tolerance."""
        return self.status == "feasible"


def _exact_denominator_multiplier_polynomial(support, lower_active, upper_active, /):
    """Return the exact squared denominator multiplier for binary64 support.

    Parameters
    ----------
    support : numpy.ndarray, shape (2,)
        Binary64 support endpoints.
    lower_active, upper_active : bool
        Whether each finite endpoint carries a logarithmic term.
    """
    lower, upper = map(float, support)
    multiplier = [Fraction(1)]
    if bool(lower_active):
        distance = [-_as_exact_fraction(lower), Fraction(1)]
        multiplier = _fraction_polymul(
            multiplier, _fraction_polymul(distance, distance)
        )
    if bool(upper_active):
        distance = [_as_exact_fraction(upper), Fraction(-1)]
        multiplier = _fraction_polymul(
            multiplier, _fraction_polymul(distance, distance)
        )
    return _fraction_trim(multiplier)


def _exact_polynomial_tail_limit(coefficients, positive_infinity, /):
    """Return an exact finite tail value or signed infinity for a rational polynomial.

    Parameters
    ----------
    coefficients : sequence of Fraction
        Ascending exact coefficients.
    positive_infinity : bool
        Tail direction: ``+inf`` when true, ``-inf`` otherwise.
    """
    values = _fraction_trim(coefficients)
    degree = len(values) - 1
    if degree == 0:
        return values[0]
    sign = 1 if values[-1] > 0 else -1
    if not positive_infinity and degree % 2:
        sign = -sign
    return np.inf if sign > 0 else -np.inf


def _exact_curvature_endpoint_values(
    q_d2, support, boundary_amplitudes, lower_active, upper_active, /
):
    """Return exact regular-endpoint and algebraic-tail curvature limits.

    Parameters
    ----------
    q_d2 : numpy.ndarray
        Ascending exact-dyadic ordinary curvature coefficients.
    support : numpy.ndarray, shape (2,)
        Canonical support endpoints.
    boundary_amplitudes : numpy.ndarray, shape (2,)
        Lower/upper logarithmic amplitudes; ``nan`` when absent.
    lower_active, upper_active : bool
        Whether each finite endpoint carries a logarithmic term.
    """
    p = _fraction_trim(q_d2)
    lower, upper = map(float, support)
    a_lower, a_upper = map(float, boundary_amplitudes)

    lower_endpoint = None
    lower_tail = None
    if np.isfinite(lower):
        if lower_active:
            lower_endpoint = np.inf
        else:
            z = _as_exact_fraction(lower)
            lower_endpoint = _fraction_polyval(p, z)
            if upper_active:
                width = _as_exact_fraction(upper) - z
                lower_endpoint += _as_exact_fraction(a_upper) / (width * width)
    else:
        lower_tail = _exact_polynomial_tail_limit(p, False)

    upper_endpoint = None
    upper_tail = None
    if np.isfinite(upper):
        if upper_active:
            upper_endpoint = np.inf
        else:
            z = _as_exact_fraction(upper)
            upper_endpoint = _fraction_polyval(p, z)
            if lower_active:
                width = z - _as_exact_fraction(lower)
                upper_endpoint += _as_exact_fraction(a_lower) / (width * width)
    else:
        upper_tail = _exact_polynomial_tail_limit(p, True)

    return lower_endpoint, upper_endpoint, lower_tail, upper_tail


def _float_near_fraction_point(point, lower, upper, /):
    """Return a binary64 point near a rational target and inside an interval.

    Parameters
    ----------
    point : Fraction
        Exact target point.
    lower, upper : Fraction
        Exact interval that must contain the result.
    """
    if lower > point or point > upper:
        raise ValueError("target point must lie inside the rational interval")
    try:
        candidate = float(point)
    except OverflowError:
        return None
    if not np.isfinite(candidate):
        return None
    probes = (
        candidate,
        float(np.nextafter(candidate, -np.inf)),
        float(np.nextafter(candidate, np.inf)),
    )
    for probe in probes:
        if not np.isfinite(probe):
            continue
        exact = Fraction.from_float(probe)
        if lower <= exact <= upper:
            return probe
    return None


def _certified_float_violation_point(sign_polynomial, lower, upper, support, /):
    """Find a representable interior point where an exact sign polynomial is negative.

    Parameters
    ----------
    sign_polynomial : sequence of Fraction
        Exact polynomial whose sign matches the curvature.
    lower, upper : Fraction
        Exact search interval.
    support : numpy.ndarray, shape (2,)
        Binary64 support endpoints.
    """
    support_lower, support_upper = map(float, support)
    candidates = []
    for exact_point in (lower, (lower + upper) / 2, upper):
        candidate = _float_near_fraction_point(exact_point, lower, upper)
        if candidate is not None:
            candidates.append(candidate)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not (support_lower < candidate < support_upper):
            continue
        point = Fraction.from_float(candidate)
        if _fraction_polyval(sign_polynomial, point) < 0:
            return candidate
    return None


def _full_curvature_value(q_d2, support, boundary_amplitudes, point, /):
    """Evaluate full curvature at one finite support point.

    Parameters
    ----------
    q_d2 : array_like
        Ascending ordinary-curvature coefficients.
    support : array_like, shape (2,)
        Canonical support.
    boundary_amplitudes : array_like, shape (2,)
        Canonical boundary amplitudes.
    point : float
        Finite evaluation point.

    Returns
    -------
    float
        Full curvature value.
    """
    z = float(point)
    lower, upper = map(float, support)
    a_lower, a_upper = map(float, boundary_amplitudes)
    value = float(np.polynomial.polynomial.polyval(z, q_d2))
    if np.isfinite(a_lower) and a_lower > 0.0:
        value += a_lower / (z - lower) ** 2
    if np.isfinite(a_upper) and a_upper > 0.0:
        value += a_upper / (upper - z) ** 2
    return float(value)


def _separate_full_curvature(
    q_d2,
    support,
    boundary_amplitudes,
    feasibility_tolerance,
    root_width,
    max_root_depth,
    bernstein_subdivide,
    /,
):
    """Run the lazy full-curvature separator with exact-dyadic certification.

    The model inputs remain binary64, but every sign decision that can declare
    a stationary interval harmless, reject it as a maximum, or certify a
    violation is made on the exact real values encoded by those binary64
    inputs.  Exact rational arithmetic is used for denominator clearing,
    stationary-root Bernstein isolation, and Bernstein sign bounds.  Float
    evaluation is retained only for the returned diagnostic violation value
    and representable cut location.

    Parameters
    ----------
    q_d2 : numpy.ndarray
        Ascending ordinary curvature coefficients.
    support : numpy.ndarray, shape (2,)
        Canonical support endpoints.
    boundary_amplitudes : numpy.ndarray, shape (2,)
        Lower/upper logarithmic amplitudes; ``nan`` when absent.
    feasibility_tolerance : float
        Curvature may be as low as ``-feasibility_tolerance`` (scaled by the endpoint denominator multiplier).
    root_width : float
        Target stationary-root bracket width.
    max_root_depth : int
        Maximum bisection depth for root isolation.
    bernstein_subdivide : int
        Subdivision levels for Bernstein sign bounds.
    """
    tolerance = float(feasibility_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be finite and >= 0")
    bound_depth = int(bernstein_subdivide)
    if bound_depth < 0:
        raise ValueError("bernstein_subdivide must be >= 0")

    p = _trim_exact(q_d2)
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    boundary = _curvature_boundary_values(p, bounds, amplitudes)
    p_exact = _exact_float_polynomial(p)
    sign_exact, stationary_exact, lower_active, upper_active = (
        _exact_curvature_polynomials(p, bounds, amplitudes)
    )
    tolerance_exact = _as_exact_fraction(tolerance)
    exact_boundary = _exact_curvature_endpoint_values(
        p_exact,
        bounds,
        amplitudes,
        lower_active,
        upper_active,
    )

    def result(
        status,
        kind=None,
        location=None,
        value=None,
        *,
        roots=0,
        maxima=0,
        pruned=0,
        refined=0,
        ambiguous=0,
        subdivisions=0,
    ):
        return _SeparationResult(
            status=status,
            violation_kind=kind,
            violation_location=location,
            violation_value=value,
            isolated_roots=roots,
            discarded_maxima=maxima,
            pruned_intervals=pruned,
            refined_intervals=refined,
            ambiguous_intervals=ambiguous,
            root_subdivisions=subdivisions,
        )

    exact_lower_endpoint, exact_upper_endpoint, exact_lower_tail, exact_upper_tail = (
        exact_boundary
    )
    for kind, exact_value, float_value in (
        ("lower_tail", exact_lower_tail, boundary.lower_tail),
        ("upper_tail", exact_upper_tail, boundary.upper_tail),
    ):
        if exact_value is not None and exact_value < -tolerance_exact:
            return result("violated", kind, None, float(float_value))

    for kind, location, exact_value, float_value in (
        ("lower_endpoint", bounds[0], exact_lower_endpoint, boundary.lower_endpoint),
        ("upper_endpoint", bounds[1], exact_upper_endpoint, boundary.upper_endpoint),
    ):
        if exact_value is not None and exact_value < -tolerance_exact:
            return result("violated", kind, float(location), float(float_value))

    isolated = _stationary_root_brackets_exact(
        stationary_exact,
        bounds,
        root_width,
        max_root_depth,
    )
    if isolated.identically_zero:
        return result("feasible", roots=0, subdivisions=isolated.subdivisions)

    multiplier_exact = _exact_denominator_multiplier_polynomial(
        bounds, lower_active, upper_active
    )
    tolerance_sign_exact = _fraction_polyadd(
        sign_exact,
        [tolerance_exact * value for value in multiplier_exact],
    )

    discarded_maxima = 0
    pruned_intervals = 0
    refined_intervals = 0
    ambiguous_intervals = 0
    stationary_derivative_exact = _fraction_polyder(stationary_exact)
    exact_root_width = _as_exact_fraction(root_width)

    for bracket in isolated.brackets:
        if int(bracket.sign_variations) == 1:
            refined_root, resolved_root = _refine_unique_root_bracket_exact(
                stationary_exact, bracket, root_width, max_root_depth
            )
            if resolved_root:
                bracket = refined_root
            else:
                ambiguous_intervals += 1
                continue

        exact_left = (
            bracket.exact_lower
            if bracket.exact_lower is not None
            else Fraction.from_float(float(bracket.lower))
        )
        exact_right = (
            bracket.exact_upper
            if bracket.exact_upper is not None
            else Fraction.from_float(float(bracket.upper))
        )

        if exact_left == exact_right:
            stationary_slope = _fraction_polyval(
                stationary_derivative_exact, exact_left
            )
            if stationary_slope < 0:
                discarded_maxima += 1
                continue
            sign_value = _fraction_polyval(tolerance_sign_exact, exact_left)
            if sign_value >= 0:
                pruned_intervals += 1
                continue
            envelope_left = Fraction.from_float(float(bracket.lower))
            envelope_right = Fraction.from_float(float(bracket.upper))
            location = _certified_float_violation_point(
                tolerance_sign_exact,
                min(envelope_left, exact_left),
                max(envelope_right, exact_right),
                bounds,
            )
            if location is not None:
                curvature = _full_curvature_value(p, bounds, amplitudes, location)
                return result(
                    "violated",
                    "stationary",
                    location,
                    curvature,
                    roots=len(isolated.brackets),
                    maxima=discarded_maxima,
                    pruned=pruned_intervals,
                    refined=refined_intervals,
                    ambiguous=ambiguous_intervals,
                    subdivisions=isolated.subdivisions,
                )
            ambiguous_intervals += 1
            continue

        s_left = _fraction_polyval(stationary_exact, exact_left)
        s_right = _fraction_polyval(stationary_exact, exact_right)
        if s_left > 0 and s_right < 0:
            discarded_maxima += 1
            continue

        local_left = exact_left
        local_right = exact_right
        local_s_left = s_left
        local_s_right = s_right
        local_depth = 0
        bracket_refined = False
        while True:
            sign_lower, sign_upper = _fraction_bernstein_interval_bounds(
                tolerance_sign_exact,
                local_left,
                local_right,
                bound_depth,
            )
            if sign_lower >= 0:
                pruned_intervals += 1
                break
            if sign_upper < 0:
                location = _certified_float_violation_point(
                    tolerance_sign_exact, local_left, local_right, bounds
                )
                if location is not None:
                    curvature = _full_curvature_value(p, bounds, amplitudes, location)
                    return result(
                        "violated",
                        "stationary",
                        location,
                        curvature,
                        roots=len(isolated.brackets),
                        maxima=discarded_maxima,
                        pruned=pruned_intervals,
                        refined=refined_intervals,
                        ambiguous=ambiguous_intervals,
                        subdivisions=isolated.subdivisions,
                    )
                ambiguous_intervals += 1
                break

            location = _certified_float_violation_point(
                tolerance_sign_exact, local_left, local_right, bounds
            )
            if location is not None:
                curvature = _full_curvature_value(p, bounds, amplitudes, location)
                return result(
                    "violated",
                    "stationary",
                    location,
                    curvature,
                    roots=len(isolated.brackets),
                    maxima=discarded_maxima,
                    pruned=pruned_intervals,
                    refined=refined_intervals,
                    ambiguous=ambiguous_intervals,
                    subdivisions=isolated.subdivisions,
                )

            if (
                local_right - local_left <= exact_root_width
                or local_depth >= max_root_depth
            ):
                ambiguous_intervals += 1
                break

            midpoint = (local_left + local_right) / 2
            s_mid = _fraction_polyval(stationary_exact, midpoint)
            if not bracket_refined:
                refined_intervals += 1
                bracket_refined = True
            local_depth += 1
            if s_mid == 0:
                sign_mid = _fraction_polyval(tolerance_sign_exact, midpoint)
                if sign_mid >= 0:
                    pruned_intervals += 1
                    break
                location = _certified_float_violation_point(
                    tolerance_sign_exact, local_left, local_right, bounds
                )
                if location is not None:
                    curvature = _full_curvature_value(p, bounds, amplitudes, location)
                    return result(
                        "violated",
                        "stationary",
                        location,
                        curvature,
                        roots=len(isolated.brackets),
                        maxima=discarded_maxima,
                        pruned=pruned_intervals,
                        refined=refined_intervals,
                        ambiguous=ambiguous_intervals,
                        subdivisions=isolated.subdivisions,
                    )
                ambiguous_intervals += 1
                break

            if local_s_left * s_mid < 0:
                local_right = midpoint
                local_s_right = s_mid
            elif s_mid * local_s_right < 0:
                local_left = midpoint
                local_s_left = s_mid
            else:
                ambiguous_intervals += 1
                break

    status = "uncertain" if ambiguous_intervals else "feasible"
    return result(
        status,
        roots=len(isolated.brackets),
        maxima=discarded_maxima,
        pruned=pruned_intervals,
        refined=refined_intervals,
        ambiguous=ambiguous_intervals,
        subdivisions=isolated.subdivisions,
    )
