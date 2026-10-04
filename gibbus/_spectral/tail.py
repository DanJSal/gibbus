"""Asymptotic inversion of the CDF in the extreme tails.

The spectral CDF stores probability on an absolute ``[0, 1]`` scale.  Once a
tail probability is below that representation's useful absolute precision,
quantiles are solved from ``log F(x) = log p`` directly against the fitted
potential instead of by inverting a numerically zero CDF value.

Asymptotic form
---------------
Write the density as ``f = exp(-Q)``, with ``Q`` the public-coordinate
potential, and let ``L`` be the support endpoint on the queried side with
``d = |x - L|``.  Integration by parts gives the leading approximation

    F(x) ~ f(x) * d / (1 + a),      a = |Q'(x)| * d

which covers both finite and infinite endpoints:

* At a finite endpoint with ``f ~ C * d**a``, the expression reduces to the
  exact leading term ``C * d**(a+1) / (a+1)``.
* At an infinite endpoint, ``d -> inf`` and the expression approaches
  ``F ~ f / |Q'|``, the usual Mills-ratio form for an exponential tail.

Finite-endpoint distances are formed directly in public coordinates.  A zero
endpoint can therefore be approached through essentially the full positive
float64 range; at a nonzero endpoint, resolution is limited only by the spacing
of representable public-coordinate values.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .._defaults import (
    NUMERIC_FAILURES,
    TAIL_ASYMPTOTIC_P,
    TAIL_BRACKET_GROWTH,
    TAIL_BRACKET_MAX_EXPAND,
    TAIL_QUAD_EPSABS,
    TAIL_QUAD_LIMIT,
    TAIL_RATE_TOL,
    TAIL_SOLVE_MAX_ITER,
    TAIL_SOLVE_TOL,
    TINY_FLOAT,
    _reraise_if_debug,
)

__all__ = [
    "tail_log_cdf",
    "invert_tail",
    "needs_asymptotic_tail",
    "refine_tail_quantiles",
    "exact_tail_log_cdf",
    "tail_rate",
]


def _tail_quad(func, /, *, epsrel, context):
    """Run tail quadrature while routing SciPy diagnostics through the ledger.

    ``full_output=1`` returns QUADPACK's message rather than emitting
    ``IntegrationWarning``, so concurrent evaluation never mutates the
    process-global warning filters.

    Parameters
    ----------
    func : callable
        Dimensionless tail integrand on ``[0, inf)``.
    epsrel : float
        Relative integration tolerance.
    context : str
        Failure-ledger context for reported integration problems.
    """
    result = quad(
        func,
        0.0,
        np.inf,
        epsabs=TAIL_QUAD_EPSABS,
        epsrel=epsrel,
        limit=TAIL_QUAD_LIMIT,
        full_output=1,
    )
    if len(result) > 3:
        _reraise_if_debug(RuntimeError(str(result[3])), context)
    return float(result[0]), float(result[1])


def needs_asymptotic_tail(p, /):
    """Whether *p* is beyond what the spectral CDF can resolve.

    Parameters
    ----------
    p : numpy.ndarray
        Probabilities in ``[0, 1]``.

    Returns
    -------
    lower : numpy.ndarray of bool
        ``p`` below the lower-tail threshold (and strictly positive).
    upper : numpy.ndarray of bool
        ``p`` above the corresponding upper-tail threshold (and strictly
        below one).
    """
    lower = (p > 0.0) & (p <= TAIL_ASYMPTOTIC_P)
    upper = (p < 1.0) & (p >= 1.0 - TAIL_ASYMPTOTIC_P)
    return lower, upper


def tail_log_cdf(potential, x, endpoint, /, *, upper):
    """Asymptotic ``log F(x)`` (or ``log(1 - F(x))``) near a support edge.

    Parameters
    ----------
    potential : callable
        ``potential(x, n)`` returning the ``n``-th derivative of
        ``Q = -log f`` at *x*.  ``Distribution.neg_log`` satisfies this for both
        single components and mixtures.
    x : float
        Evaluation point, in the same coordinates as *endpoint*.
    endpoint : float
        Support endpoint on the side being queried; may be infinite.
    upper : bool
        Which tail is being queried.  Accepted for signature symmetry with
        :func:`exact_tail_log_cdf`, so callers can forward it uniformly, but
        it does not change the result: the expression below depends on the
        endpoint only through ``|x - endpoint|`` and on the potential only
        through ``|Q'|``, so it is already symmetric in the two tails.

    Returns
    -------
    float
        ``log F(x)`` for the lower tail, ``log(1 - F(x))`` for the upper.
        ``-inf`` if the point is at or beyond the endpoint.

    Notes
    -----
    Both terms are evaluated in log space, so the result is meaningful
    far below the smallest positive value ``F`` itself could represent.
    """
    q = float(potential(x, 0))
    dq = float(potential(x, 1))
    # A non-finite potential or slope still marks the edge of what the
    # public float64 coordinate can represent.  Finite-endpoint boundary
    # distances themselves are evaluated in public coordinates, so there is
    # no additional O(1)-internal-coordinate cancellation floor here.
    # Moving away from the mode, the potential must be rising; the
    # gradient points back toward it.
    slope = abs(dq)
    if not np.isfinite(q) or not np.isfinite(slope) or slope <= 0.0:
        return -np.inf

    if np.isfinite(endpoint):
        d = abs(x - endpoint)
        if d <= 0.0:
            return -np.inf
        a = slope * d
        # log(d) - log1p(a) is the algebraic-endpoint form; as a grows it
        # becomes log(d) - log(a) = -log(slope), the Mills form.
        return -q + np.log(d) - np.log1p(a)
    return -q - np.log(slope)


def exact_tail_log_cdf(potential, x, endpoint, /, *, upper):
    """Return the tail log-mass by scaled direct quadrature.

    The leading asymptotic expression in :func:`tail_log_cdf` is an excellent
    seed far into a tail, but near the spectral handover its relative mass
    error is still percent-level for ordinary Gaussian tails.  This evaluator
    factors out the density at *x* and integrates only an O(1) relative tail
    profile, so probabilities far below the float64 underflow threshold never
    need to be represented directly.

    Infinite tails are integrated in a local hazard-length coordinate.  Finite
    tails are integrated in log-distance from the support endpoint; that
    coordinate also regularizes algebraic boundary singularities.

    Parameters
    ----------
    potential : callable
        Potential evaluator accepting a coordinate and derivative order.
    x : float
        Tail coordinate at which to evaluate the mass.
    endpoint : float
        Support endpoint in the outward tail direction.
    upper : bool
        Whether to evaluate the upper rather than lower tail.
    """
    qx = float(potential(x, 0))
    if not np.isfinite(qx):
        return -np.inf

    direction = 1.0 if upper else -1.0
    if np.isfinite(endpoint):
        distance = abs(float(x) - float(endpoint))
        if not np.isfinite(distance) or distance <= 0.0:
            return -np.inf

        def relative(v):
            ev = np.exp(-v)
            t = endpoint - direction * distance * ev
            # Once public-coordinate rounding reaches the endpoint there is no
            # additional representable interval to integrate.
            if t == endpoint:
                return 0.0
            qt = float(potential(t, 0))
            if not np.isfinite(qt):
                return 0.0
            exponent = -v + qx - qt
            if exponent <= np.log(TINY_FLOAT):
                return 0.0
            # A valid tail point should keep this O(1).  The cap avoids a
            # spurious overflow if a trial point lies on the wrong side of a
            # remote mode; bidirectional bracketing will move away from it.
            return float(np.exp(min(exponent, 700.0)))

        rel_floor = 8.0 * abs(np.spacing(x)) / max(distance, TINY_FLOAT)
        epsrel = min(0.1, max(1e-11, rel_floor))
        value, _ = _tail_quad(
            relative, epsrel=epsrel, context="finite-endpoint tail quadrature"
        )
        if not np.isfinite(value) or value <= 0.0:
            return -np.inf
        return -qx + np.log(distance) + np.log(value)

    slope = abs(float(potential(x, 1)))
    if not np.isfinite(slope) or slope <= 0.0:
        return -np.inf
    local_scale = 1.0 / slope
    if not np.isfinite(local_scale) or local_scale <= 0.0:
        return -np.inf

    def relative(v):
        t = x + direction * local_scale * v
        qt = float(potential(t, 0))
        if not np.isfinite(qt):
            return 0.0
        exponent = qx - qt
        if exponent <= np.log(TINY_FLOAT):
            return 0.0
        return float(np.exp(min(exponent, 700.0)))

    rel_floor = 8.0 * abs(np.spacing(x)) / max(local_scale, TINY_FLOAT)
    epsrel = min(0.1, max(1e-11, rel_floor))
    value, _ = _tail_quad(
        relative, epsrel=epsrel, context="infinite-endpoint tail quadrature"
    )
    if not np.isfinite(value) or value <= 0.0:
        return -np.inf
    return -qx + np.log(local_scale) + np.log(value)


def _displacement_coordinate(potential, endpoint, direction, start, /):
    """Return a dimensionless displacement coordinate rooted at *start*.

    Absolute public location must not control the root tolerance.  On an
    infinite tail the displacement is scaled by the local hazard length
    ``1 / |Q'(start)|``.  On a finite tail it is displacement in log-distance
    from the endpoint.  In both cases positive coordinate values move outward,
    toward smaller tail mass.

    Parameters
    ----------
    potential : callable
        Potential evaluator accepting a coordinate and derivative order.
    endpoint : float
        Support endpoint in the outward tail direction.
    direction : float
        Signed direction from the distribution interior toward the tail.
    start : float
        Initial public-coordinate tail seed.
    """
    if np.isfinite(endpoint):
        distance0 = abs(float(start) - float(endpoint))
        if not np.isfinite(distance0):
            raise RuntimeError("tail inversion seed has non-finite endpoint distance")
        if distance0 <= 0.0:
            # On very narrow finite supports the handover quantile can round to
            # the endpoint even though the mathematical quantile is interior.
            # Root the displacement coordinate at the nearest representable
            # interior point instead.  The exact correction may still return
            # the endpoint when that is the correctly rounded public quantile.
            inward = np.nextafter(endpoint, -np.inf if direction > 0.0 else np.inf)
            distance0 = abs(float(inward) - float(endpoint))
            if not np.isfinite(distance0) or distance0 <= 0.0:
                raise RuntimeError(
                    "tail inversion seed has no representable interior displacement"
                )

        def to_x(v):
            distance = distance0 * np.exp(-v)
            return endpoint - direction * distance

        def dx_dv(v):
            return direction * distance0 * np.exp(-v)

        return to_x, dx_dv

    slope = abs(float(potential(start, 1)))
    if not np.isfinite(slope) or slope <= 0.0:
        raise RuntimeError("tail inversion seed has no finite outward slope")
    scale = 1.0 / slope
    if not np.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("tail inversion seed has no finite local scale")

    def to_x(v):
        return start + direction * scale * v

    def dx_dv(v):
        return direction * scale

    return to_x, dx_dv


def _bracket(log_target, f_at, /):
    """Bracket a monotone decreasing tail log-mass around displacement zero.

    The spectral seed is normally inward of the requested probability, but
    panel error can put it on the other side.  Expand in whichever direction
    the value at zero requires instead of assuming that only an outward walk
    can succeed.

    Parameters
    ----------
    log_target : float
        Target logarithmic tail mass.
    f_at : callable
        Monotone decreasing log-tail function of displacement.
    """
    f0 = float(f_at(0.0))
    if not np.isfinite(f0):
        raise RuntimeError("tail inversion seed has non-finite tail mass")
    if f0 == log_target:
        return 0.0, 0.0

    step = 1.0
    if f0 > log_target:
        # Need less mass: move outward (positive displacement).
        lo, f_lo = 0.0, f0
        for _ in range(TAIL_BRACKET_MAX_EXPAND):
            hi = step
            f_hi = float(f_at(hi))
            if f_hi <= log_target:
                return lo, hi
            lo, f_lo = hi, f_hi
            step *= TAIL_BRACKET_GROWTH
    else:
        # Seed has too little mass: move inward (negative displacement).
        hi, f_hi = 0.0, f0
        for _ in range(TAIL_BRACKET_MAX_EXPAND):
            lo = -step
            f_lo = float(f_at(lo))
            if f_lo >= log_target:
                return lo, hi
            hi, f_hi = lo, f_lo
            step *= TAIL_BRACKET_GROWTH

    raise RuntimeError(
        f"tail inversion could not bracket log p = {log_target:.6g} "
        f"within {TAIL_BRACKET_MAX_EXPAND} expansions"
    )


def _bisect_tail(log_target, f_at, lo, hi, /):
    """Bisect a decreasing tail log-mass in displacement coordinates.

    Parameters
    ----------
    log_target : float
        Target logarithmic tail mass.
    f_at : callable
        Monotone decreasing log-tail function of displacement.
    lo : float
        Lower displacement bracket.
    hi : float
        Upper displacement bracket.
    """
    if lo == hi:
        return lo
    for _ in range(TAIL_SOLVE_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if hi - lo <= TAIL_SOLVE_TOL * max(abs(mid), 1.0):
            return mid
        if f_at(mid) >= log_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def invert_tail(potential, log_p, endpoint, start, /, *, upper, log_tail_mass):
    """Solve an extreme tail quantile with asymptotic seeding and exact correction.

    The leading asymptotic tail is used only to obtain a robust seed.  The
    returned quantile is corrected against a scaled direct quadrature of the
    actual fitted density, with Newton steps taken in a dimensionless
    displacement-from-seed coordinate.  This keeps the positional tolerance
    independent of a large common translation and removes the percent-level
    tail-mass error of the leading asymptotic near the handover.

    Parameters
    ----------
    potential : callable
        Potential evaluator accepting a coordinate and derivative order.
    log_p : float
        Target logarithmic tail probability.
    endpoint : float
        Support endpoint in the outward tail direction.
    start : float
        Initial asymptotic tail seed.
    upper : bool
        Whether to invert the upper rather than lower tail.
    log_tail_mass : callable or None
        Exact tail-mass evaluator ``f(x, endpoint, upper=...)``.  Supplying
        one lets callers keep component or mixture tail quadrature below the
        Python potential-callback boundary during root correction.
    """
    direction = 1.0 if upper else -1.0
    to_x, dx_dv = _displacement_coordinate(potential, endpoint, direction, float(start))

    def asym_at(v):
        return tail_log_cdf(potential, to_x(v), endpoint, upper=upper)

    lo, hi = _bracket(log_p, asym_at)
    v = _bisect_tail(log_p, asym_at, lo, hi)

    # A handful of exact Newton corrections is enough because the asymptotic
    # root is already close.  Keep a bracket in the same displacement variable
    # so a bad Newton step cannot leave the monotone tail region.
    def exact_at(vv):
        x_eval = to_x(vv)
        if log_tail_mass is not None:
            return float(log_tail_mass(x_eval, endpoint, upper=upper))
        return exact_tail_log_cdf(potential, x_eval, endpoint, upper=upper)

    # Re-bracket the *exact* target.  This is usually a very small adjustment,
    # but it also handles a spectral seed that was on the wrong side.
    f_v = exact_at(v)
    if not np.isfinite(f_v):
        raise RuntimeError("exact tail quadrature failed at asymptotic seed")
    if f_v >= log_p:
        exact_lo = v
        step = 1.0
        for _ in range(TAIL_BRACKET_MAX_EXPAND):
            exact_hi = v + step
            if exact_at(exact_hi) <= log_p:
                break
            exact_lo = exact_hi
            step *= TAIL_BRACKET_GROWTH
        else:
            raise RuntimeError("exact tail correction could not bracket outward")
    else:
        exact_hi = v
        step = 1.0
        for _ in range(TAIL_BRACKET_MAX_EXPAND):
            exact_lo = v - step
            if exact_at(exact_lo) >= log_p:
                break
            exact_hi = exact_lo
            step *= TAIL_BRACKET_GROWTH
        else:
            raise RuntimeError("exact tail correction could not bracket inward")

    for _ in range(12):
        x = float(to_x(v))
        log_mass = float(exact_at(v))
        residual = log_mass - log_p
        if abs(residual) <= 2e-13:
            return x

        qx = float(potential(x, 0))
        # |d log(tail) / dx| = f / tail.  Multiplying by |dx/dv|
        # gives the magnitude in the displacement coordinate; the derivative
        # itself is negative because positive v always moves outward.
        log_hazard = -qx - log_mass
        deriv = -np.exp(min(log_hazard, 700.0)) * abs(float(dx_dv(v)))
        if not np.isfinite(deriv) or deriv >= 0.0:
            candidate = 0.5 * (exact_lo + exact_hi)
        else:
            candidate = v - residual / deriv
            if not (exact_lo < candidate < exact_hi) or not np.isfinite(candidate):
                candidate = 0.5 * (exact_lo + exact_hi)

        if log_mass >= log_p:
            exact_lo = v
        else:
            exact_hi = v
        if float(to_x(candidate)) == x:
            return x
        v = candidate

    v = _bisect_tail(log_p, exact_at, exact_lo, exact_hi)
    return float(to_x(v))


def refine_tail_quantiles(
    potential, raw_ppf, lo_edge, hi_edge, p, base_out, /, *, log_tail_mass
):
    """Re-solve quantiles the spectral CDF cannot resolve.

    Below ``TAIL_ASYMPTOTIC_P`` the CDF panels hold no significant digits,
    and inverting them returns a confident wrong answer rather than a poor
    one.  Those points are recomputed from the potential; see
    :mod:`gibbus._spectral.tail`.

    This lives beside the tail model so every public path that consumes a
    raw spectral inverse applies the same refinement: component ``base``
    and ``exp`` views as well as the top-level ``Distribution`` object.

    Parameters
    ----------
    potential : callable
        Negative log density in base coordinates.
    raw_ppf : callable
        The *unrefined* spectral inverse, in base coordinates.  Must not
        be a refined PPF, or seeding would recurse.
    lo_edge, hi_edge : float
        Support bounds in base coordinates.
    p : numpy.ndarray, shape (R,)
        Requested probabilities.
    base_out : numpy.ndarray, shape (R,)
        Quantiles from the spectral inverse, in base coordinates.
    log_tail_mass : callable or None
        Exact tail-mass evaluator forwarded to :func:`invert_tail`.

    Returns
    -------
    numpy.ndarray, shape (R,)
        *base_out* with the extreme entries replaced.
    """
    lower, upper = needs_asymptotic_tail(p)
    if not (lower.any() or upper.any()):
        return base_out

    seed_lo = float(np.atleast_1d(raw_ppf(TAIL_ASYMPTOTIC_P))[0])
    seed_hi = float(np.atleast_1d(raw_ppf(1.0 - TAIL_ASYMPTOTIC_P))[0])

    out = base_out.copy()
    for idx in np.flatnonzero(lower):
        try:
            out[idx] = invert_tail(
                potential,
                float(np.log(p[idx])),
                lo_edge,
                seed_lo,
                upper=False,
                log_tail_mass=log_tail_mass,
            )
        except NUMERIC_FAILURES as exc:
            # Keep the spectral answer rather than fail a query; it is
            # wrong out here, but raising would be a regression for
            # callers who only wanted a rough tail value.
            _reraise_if_debug(
                exc, f"asymptotic lower-tail quantile at p={p[idx]:.3e}", routine=True
            )
    for idx in np.flatnonzero(upper):
        try:
            out[idx] = invert_tail(
                potential,
                float(np.log1p(-p[idx])),
                hi_edge,
                seed_hi,
                upper=True,
                log_tail_mass=log_tail_mass,
            )
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(
                exc, f"asymptotic upper-tail quantile at p={p[idx]:.3e}", routine=True
            )
    return out


def tail_rate(potential, support, mode, scale, side, /):
    """Estimate the limiting absolute potential slope in one base-space tail.

    Parameters
    ----------
    potential : callable
        Negative-log density evaluator.
    support : array_like, shape (2,)
        Base-space support.
    mode : float
        Distribution mode.
    scale : float
        Positive body scale used to choose outward probes.
    side : {'lower', 'upper'}
        Tail to inspect.
    """
    side = str(side).lower()
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    lo, hi = map(float, support)
    endpoint = lo if side == "lower" else hi
    if np.isfinite(endpoint):
        return np.inf
    direction = -1.0 if side == "lower" else 1.0
    step = max(float(scale), 1.0)
    vals = []
    for k in range(1, 22):
        x = float(mode + direction * step * (2.0**k))
        slope = abs(float(potential(x, 1)))
        if not np.isfinite(slope):
            return np.inf
        vals.append(slope)
        if len(vals) >= 2:
            rel = abs(vals[-1] - vals[-2]) / max(vals[-1], vals[-2], 1.0)
            if rel <= TAIL_RATE_TOL:
                return float(vals[-1])
        if len(vals) >= 4:
            ratios = [
                vals[i + 1] / max(vals[i], np.finfo(float).tiny)
                for i in range(len(vals) - 1)
            ]
            if min(ratios[-3:]) > 1.25:
                return np.inf
    recent = np.asarray(vals[-6:], dtype=np.float64)
    if recent.size >= 4:
        diffs = np.diff(recent)
        if np.all(diffs > 0.0):
            tiny = np.finfo(np.float64).tiny
            growth = recent[-1] / max(recent[0], tiny)
            diff_ratios = diffs[1:] / np.maximum(diffs[:-1], tiny)
            # A polynomially growing potential slope has non-decaying
            # increments on dyadically spaced probes.  A finite limiting
            # rate approaches its plateau with shrinking increments instead.
            if growth > 1.5 or np.min(diff_ratios) >= 0.9:
                return np.inf
    raise RuntimeError("tail-rate estimate did not stabilize")
