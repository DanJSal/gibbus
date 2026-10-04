"""Post-fit analytics: moment computation, affine updates, and mode finding.

This module provides helpers used *after* fitting to compute or update
derived quantities — moments, summary statistics, support bounds, and
exp-space raw moments — without re-running the optimizer.

Pipeline position
-----------------
This module is called from:

* ``gibbus._api.views._ExpSpaceView`` — for exp-space mode and moment computation.
* ``gibbus._api.component._Component._raw_moment_base`` — for on-demand base-space moment
  computation beyond the cached ``raw_moments`` array.

It wraps the Cython kernels ``_state_kernels._valley_q1_shift`` and
``_quad_integrals.quad_integral`` with the package-level defaults from
:mod:`._defaults`, so callers do not need to supply tolerance parameters
directly.
"""

from math import comb, factorial

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .._defaults import (
    BACKTRACK_MAX_ITERS,
    BACKTRACK_REDUCE,
    BOUNDARY_EPS_MULT,
    BRACKET_INIT_STEP,
    BRACKET_MAX_EXPAND,
    BRACKET_STEP_GROWTH,
    EXP_NARROW_D2_THRESHOLD,
    GRAD_TOL,
    HESS_TOL,
    NEWT_MAX,
    NEWT_TOL,
    QUAD_EPSABS,
    QUAD_EPSREL,
    QUAD_LIMIT,
    STATS_VAR_NEG_TOL,
)
from .._model._quad_integrals import quad_integral
from .._model._state_kernels import _valley_q1_shift
from .._model.numerics import _terms_for_quad


def _stats_from_raw_moments(m1, m2, m3, m4, /):
    """Compute summary statistics from the first four raw moments.

    Parameters
    ----------
    m1 : float
        First raw moment (mean).
    m2 : float
        Second raw moment ``E[X^2]``.
    m3 : float
        Third raw moment ``E[X^3]``.
    m4 : float
        Fourth raw moment ``E[X^4]``.

    Returns
    -------
    dict with keys ``mean``, ``var``, ``std``, ``skew``, ``kurt``
        Variance is computed as ``m2 - m1^2``.  When that result is
        within ``STATS_VAR_NEG_TOL`` *relative to* ``max(m2, m1^2)`` the
        two terms agree to their last bit, which is the signature of a
        density that has collapsed onto a spike, and a ``RuntimeError``
        is raised with an explanation.  The tolerance is purely relative
        so the check is scale-free.  ``kurt`` is
        the Pearson (raw) kurtosis, the standardized fourth moment, so a
        Gaussian reads 3.0.  Skewness and kurtosis are set to ``nan``
        when ``m3`` or ``m4`` is non-finite.

    Raises
    ------
    RuntimeError
        If the variance is non-finite, or is zero to within floating
        point.  That means the density has concentrated onto a point:
        the likelihood of a flexible density family is unbounded, and a
        fit can walk into a spike where the log-likelihood is arbitrarily
        high and every moment beyond the first is meaningless.
    """
    mean = float(m1)
    m2f = float(m2)
    var = m2f - mean * mean

    # ``m2`` and ``mean ** 2`` agree to their last bit on a spike, so
    # this subtraction is where the degeneracy shows up.  Scale the
    # tolerance by ``m2``: an absolute threshold means something
    # different at every data scale.
    # No absolute floor here: with one, data at scale 1e-8 has
    # ``m2 ~ 1e-16`` below a tolerance pinned at 1e-15 and is rejected as
    # degenerate even though it is perfectly well-conditioned.  If both
    # terms are zero the ``var <= 0`` check below still fires.
    scale_ref = max(abs(m2f), abs(mean * mean))
    noise = float(STATS_VAR_NEG_TOL) * scale_ref
    if -noise < var < noise:
        raise RuntimeError(
            f"Degenerate variance encountered: E[X^2] = {m2f!r} and "
            f"E[X]^2 = {mean * mean!r} agree to within floating point, so "
            f"the fitted density has concentrated onto a point. The "
            f"likelihood of this family is unbounded, and an optimizer "
            f"can converge onto such a spike at excessive polynomial "
            f"complexity. Try a lower polynomial degree or a more "
            f"appropriate support/model specification."
        )
    if (not np.isfinite(var)) or (var <= 0.0):
        raise RuntimeError(f"Degenerate variance encountered (var={var}).")
    std = float(np.sqrt(var))

    skew = np.nan
    kurt = np.nan
    if np.isfinite(m3) and np.isfinite(m4):
        m3 = float(m3)
        m4 = float(m4)
        mu3 = m3 - 3.0 * float(m2) * mean + 2.0 * (mean**3)
        mu4 = m4 - 4.0 * m3 * mean + 6.0 * float(m2) * (mean**2) - 3.0 * (mean**4)
        # Standardized third and fourth moments.  ``kurt`` is the Pearson
        # (raw) kurtosis, so a Gaussian reads 3.0, not 0.0 -- it is the
        # counterpart of ``skew``, not the Fisher excess.  Subtract 3 if
        # you want excess.
        skew = float(mu3 / (std**3))
        kurt = float(mu4 / (std**4))

    return {"mean": mean, "var": var, "std": std, "skew": skew, "kurt": kurt}


def _stats_from_centered_moments(mean, mu2, mu3, mu4, /):
    """Build summary statistics from a mean and centered moments.

    Parameters
    ----------
    mean : float
        Distribution mean.
    mu2 : float
        Second centered moment.
    mu3 : float
        Third centered moment.
    mu4 : float
        Fourth centered moment.
    """
    mean = float(mean)
    var = float(mu2)
    if not np.isfinite(var) or var <= 0.0:
        raise RuntimeError(f"Degenerate variance encountered (var={var}).")
    std = float(np.sqrt(var))
    skew = float(mu3) / (std**3) if np.isfinite(mu3) else np.nan
    kurt = float(mu4) / (std**4) if np.isfinite(mu4) else np.nan
    return {"mean": mean, "var": var, "std": std, "skew": skew, "kurt": kurt}


def _central_moment_from_raw(get_raw, k, mean, /):
    """Compute a centered moment from raw moments in a well-scaled coordinate.

    Parameters
    ----------
    get_raw : callable
        Callable returning raw moments by non-negative integer order.
    k : int
        Centered-moment order.
    mean : float
        Mean in the same coordinate system as the raw moments.
    """
    kk = k
    m = float(mean)
    out = 0.0
    for i in range(kk + 1):
        raw = 1.0 if i == 0 else float(get_raw(i))
        out += comb(kk, i) * ((-m) ** (kk - i)) * raw
    return float(out)


def _cumulant_from_centered(get_centered, k, mean, /):
    """Compute a cumulant from centered moments.

    Parameters
    ----------
    get_centered : callable
        Callable returning the centered moment of a positive integer order.
    k : int
        Already-validated positive integer cumulant order.
    mean : float
        First cumulant.

    Returns
    -------
    float
        The *k*-th cumulant.

    """
    kk = k
    if kk == 1:
        return float(mean)

    # Centered moments avoid catastrophic cancellation after large affine
    # translations.  For n >= 2, the cumulant recurrence can be written
    # entirely in centered moments because mu_1 = 0.
    cumulants = {1: float(mean)}
    for n in range(2, kk + 1):
        value = float(get_centered(n))
        for j in range(2, n):
            mu = 1.0 if n - j == 0 else float(get_centered(n - j))
            value -= comb(n - 1, j - 1) * cumulants[j] * mu
        cumulants[n] = float(value)
    return cumulants[kk]


def _valley(window, base_support, q_poly, boundary_amplitudes, shift):
    """Find the point where ``q'(z) = shift`` within the fitting window.

    Thin wrapper around :func:`._state_kernels._valley_q1_shift` that
    passes the package-level solver tolerances from :mod:`._defaults`.

    Parameters
    ----------
    window : array_like, shape (2,)
        ``[wL, wR]`` — search bounds (fitting window).
    base_support : array_like, shape (2,)
        ``[L, U]`` — full support of the potential.
    q_poly : array_like, shape (d+1,)
        Polynomial coefficients, constant term first.
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset amplitudes.
    shift : float
        Target value for ``q'(z)``.  ``shift=0`` gives the mode;
        ``shift = -k / sigma_eff`` is used for exp-space moments.

    Returns
    -------
    float
        ``z*`` satisfying ``q'(z*) ≈ shift`` within the window.
    """
    return _valley_q1_shift(
        window,
        base_support,
        q_poly,
        boundary_amplitudes,
        shift,
        GRAD_TOL,
        HESS_TOL,
        NEWT_TOL,
        NEWT_MAX,
        BOUNDARY_EPS_MULT,
        BRACKET_INIT_STEP,
        BRACKET_MAX_EXPAND,
        BRACKET_STEP_GROWTH,
        BACKTRACK_MAX_ITERS,
        BACKTRACK_REDUCE,
    )


def _powaff_moment_from_z_moments(z_mom, alpha, beta, k, /):
    """Compute ``E[(alpha*Z + beta)^k]`` from internal-coordinate moments.

    Uses the binomial expansion::

        E[(alpha*Z + beta)^k] = sum_{i=0}^{k} C(k,i) * alpha^i * beta^(k-i) * E[Z^i]

    Parameters
    ----------
    z_mom : array_like, shape (k+1,)
        ``z_mom[i] = E[Z^i]`` for ``i = 0, ..., k``.
    alpha : float
        Scale coefficient in the affine transform ``alpha * Z + beta``.
    beta : float
        Offset coefficient in the affine transform ``alpha * Z + beta``.
    k : int
        Moment order.

    Returns
    -------
    float
    """
    kk = k
    s = 0.0
    for i in range(kk + 1):
        s += comb(kk, i) * (alpha**i) * (beta ** (kk - i)) * float(z_mom[i])
    return float(s)


def _support_from_base(base_support, mu_eff, sigma_eff, /):
    """Convert support bounds from internal to user coordinates.

    Parameters
    ----------
    base_support : array_like, shape (2,)
        ``[Lz, Uz]`` — bounds in internal coordinates.
    mu_eff : float
        Effective location parameter of the internal-to-user coordinate transform.
    sigma_eff : float
        Effective scale parameter of the internal-to-user coordinate transform.

    Returns
    -------
    numpy.ndarray, shape (2,), dtype float64
        ``[L, U]`` in user coordinates.  Infinite bounds are preserved.
    """
    z = np.asarray(base_support, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        x = (z - float(mu_eff)) / float(sigma_eff)
    return np.sort(np.asarray(x, dtype=np.float64))


def _moment_from_raw(get_raw, k, mean, std, central, standardized, /):
    """Derive a central or standardized moment from raw moments.

    Parameters
    ----------
    get_raw : callable
        ``get_raw(i) -> float`` returning the *i*-th raw moment.
    k : int
        Moment order.
    mean : float
        First raw moment (used for central/standardized computation).
    std : float
        Standard deviation (used for standardization).
    central : bool
        If ``True``, compute the central moment ``E[(X - mean)^k]``.
    standardized : bool
        If ``True``, divide the central moment by ``std**k``.

    Returns
    -------
    float

    Raises
    ------
    RuntimeError
        If ``standardized=True`` and ``std <= 0``.
    """
    kk = k

    if (not central) and (not standardized):
        return float(get_raw(kk))

    cm = 0.0
    m = float(mean)
    for i in range(kk + 1):
        mi = 1.0 if i == 0 else float(get_raw(i))
        cm += comb(kk, i) * ((-m) ** (kk - i)) * mi

    if not standardized:
        return float(cm)

    s = float(std)
    if not (s > 0.0):
        raise RuntimeError("Standardized moment is undefined because std <= 0.")
    return float(cm / (s**kk))


def _raw_moment_identity(
    base_support, q_poly, boundary_amplitudes, window, mu_eff, sigma_eff, k, /
):
    """Compute the *k*-th raw moment in user coordinates by quadrature.

    Evaluates ``E[X^k]`` where ``X = (Z - mu_eff) / sigma_eff`` and *Z*
    is distributed according to the fitted potential, by expressing
    ``E[X^k]`` as a polynomial in the internal moments ``E[Z^i]`` via
    :func:`_powaff_moment_from_z_moments`.

    Parameters
    ----------
    base_support : array_like, shape (2,)
        ``[L, U]`` in internal coordinates.
    q_poly : array_like, shape (d+1,)
        Potential polynomial coefficients.
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset amplitudes.
    window : array_like, shape (2,)
        Integration window ``[wL, wR]``.
    mu_eff : float
        Effective location parameter of the internal-to-user coordinate transform.
    sigma_eff : float
        Effective scale parameter of the internal-to-user coordinate transform.
    k : int
        Moment order.

    Returns
    -------
    float
    """
    kk = k
    if kk == 0:
        return 1.0

    terms = _terms_for_quad(base_support, boundary_amplitudes)

    alpha = 1.0 / float(sigma_eff)
    beta = -float(mu_eff) / float(sigma_eff)

    z_mom = np.empty(kk + 1, dtype=np.float64)
    z_mom[0] = 1.0
    a = float(window[0])
    b = float(window[1])
    for i in range(1, kk + 1):
        z_mom[i] = quad_integral(
            q_poly,
            a,
            b,
            terms,
            mode=0,
            k=i,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
        )

    return _powaff_moment_from_z_moments(z_mom, alpha, beta, kk)


def _polyval_scalar(c, x, /):
    """Evaluate a polynomial at a single scalar point using Horner's method.

    Parameters
    ----------
    c : numpy.ndarray, shape (d+1,), dtype float64
        Canonical polynomial coefficients, constant term first.
    x : float
        Scalar point at which to evaluate the polynomial.

    Returns
    -------
    float
    """
    cc = c
    n = cc.size
    if n == 0:
        return 0.0
    z = float(x)
    out = float(cc[n - 1])
    for i in range(n - 2, -1, -1):
        out = out * z + float(cc[i])
    return out


class _ScalarPotential:
    """Prepared canonical geometry for scalar root/quadrature callbacks.

    Parameters
    ----------
    support : sequence of float, length 2
        Canonical support bounds.
    q_poly : numpy.ndarray, dtype float64
        Installed increasing-power potential coefficients.
    boundary_amplitudes : numpy.ndarray, shape (2,), dtype float64
        Installed canonical lower/upper log amplitudes.
    """

    def __init__(self, support, q_poly, boundary_amplitudes, /):
        """Snapshot scalar geometry and prepare derivative coefficients once."""
        self.q = q_poly
        self.d1 = np.polynomial.polynomial.polyder(q_poly, 1)
        self.d2 = np.polynomial.polynomial.polyder(q_poly, 2)
        self.lower, self.upper = map(float, support)
        self.a_lower, self.a_upper = map(float, boundary_amplitudes)
        self.has_lower = (
            np.isfinite(self.lower) and np.isfinite(self.a_lower) and self.a_lower > 0.0
        )
        self.has_upper = (
            np.isfinite(self.upper) and np.isfinite(self.a_upper) and self.a_upper > 0.0
        )

    def value(self, x):
        """Evaluate the scalar potential, including active endpoint guards.

        Parameters
        ----------
        x : float
            Canonical evaluation coordinate.
        """
        z = float(x)
        out = _polyval_scalar(self.q, z)
        if self.has_lower:
            distance = z - self.lower
            if distance <= 0.0:
                return np.inf
            out -= self.a_lower * np.log(distance)
        if self.has_upper:
            distance = self.upper - z
            if distance <= 0.0:
                return np.inf
            out -= self.a_upper * np.log(distance)
        return out

    def gradient(self, x):
        """Evaluate the scalar first derivative using prepared coefficients.

        Parameters
        ----------
        x : float
            Canonical evaluation coordinate.
        """
        z = float(x)
        out = (
            float(np.polynomial.polynomial.polyval(z, self.d1)) if self.d1.size else 0.0
        )
        if self.has_lower:
            distance = z - self.lower
            if distance <= 0.0:
                return -np.inf
            out -= self.a_lower / distance
        if self.has_upper:
            distance = self.upper - z
            if distance <= 0.0:
                return np.inf
            out += self.a_upper / distance
        return out

    def curvature(self, x):
        """Evaluate scalar curvature using prepared coefficients.

        Parameters
        ----------
        x : float
            Canonical evaluation coordinate.
        """
        z = float(x)
        out = (
            float(np.polynomial.polynomial.polyval(z, self.d2)) if self.d2.size else 0.0
        )
        if self.has_lower:
            distance = z - self.lower
            if distance <= 0.0:
                return np.inf
            out += self.a_lower / (distance * distance)
        if self.has_upper:
            distance = self.upper - z
            if distance <= 0.0:
                return np.inf
            out += self.a_upper / (distance * distance)
        return out


def _poly_degree_exact(q_poly, /):
    """Return the highest exactly nonzero polynomial coefficient index.

    Parameters
    ----------
    q_poly : numpy.ndarray, dtype float64
        Canonical ascending polynomial coefficients.

    Returns
    -------
    int
        Highest index whose coefficient is not exactly zero, or zero if all
        coefficients vanish.
    """
    q = q_poly
    nz = np.flatnonzero(q != 0.0)
    return int(nz[-1]) if nz.size else 0


def _tail_rate_from_geometry(base_support, q_poly, mu_eff, sigma_eff, side, /):
    """Return the exact asymptotic base-space potential slope for one component.

    Finite support endpoints and any polynomial potential of degree at least
    two have infinite limiting absolute slope.  A genuinely linear potential
    on an unbounded side has the finite slope obtained by transporting its
    canonical derivative through the fitted affine coordinate.

    Parameters
    ----------
    base_support : array_like, shape (2,)
        Canonical support of the stored potential.
    q_poly : numpy.ndarray, dtype float64
        Canonical ascending polynomial-potential coefficients.
    mu_eff, sigma_eff : float
        Effective affine parameters satisfying ``z = sigma_eff*x + mu_eff``.
    side : {'lower', 'upper'}
        Public base-space tail to inspect.

    Returns
    -------
    float
        Limiting absolute potential slope; positive infinity for finite
        endpoints or super-linear polynomial tails.
    """
    key = str(side).lower()
    if key not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    support = base_support
    q = q_poly
    sigma_eff = float(sigma_eff)
    if not np.isfinite(sigma_eff) or sigma_eff == 0.0:
        raise RuntimeError("invalid fitted affine scale for tail-rate evaluation")

    # z = sigma_eff * x + mu_eff.  Positive sigma preserves which canonical
    # endpoint corresponds to each public-space tail; negative sigma swaps it.
    upper_index = 1 if sigma_eff > 0.0 else 0
    index = upper_index if key == "upper" else 1 - upper_index
    if np.isfinite(float(support[index])):
        return np.inf

    degree = _poly_degree_exact(q)
    if degree >= 2:
        return np.inf
    if degree == 1:
        return float(abs(sigma_eff * q[1]))
    return 0.0


def _expanded_tilted_log_moment(
    base_support, q_poly, boundary_amplitudes, window, mu_eff, sigma_eff, k, /
):
    """Evaluate an exp-space log moment whose tilted mode lies outside *window*.

    The ordinary fitted-state quadrature window is sized for ``exp(-q)``.
    Multiplication by ``exp(k X)`` can move the tilted mode far outside that
    material window when the fitted tail is almost exponential.  This helper
    brackets the true tilted stationary point on the full support and integrates
    in a curvature-scaled coordinate around it.

    Parameters
    ----------
    base_support : array_like, shape (2,)
        Canonical support.
    q_poly : numpy.ndarray, dtype float64
        Canonical ascending polynomial-potential coefficients.
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper logarithmic-boundary amplitudes.
    window : array_like, shape (2,)
        Stored material quadrature window for the untilted density.
    mu_eff, sigma_eff : float
        Effective affine parameters satisfying ``z = sigma_eff*x + mu_eff``.
    k : int
        Positive exp-space raw-moment order.

    Returns
    -------
    float
        Log raw moment, positive infinity for a divergent moment, or NaN when
        the expanded saddle construction is not applicable.
    """
    support = base_support
    q = q_poly
    amps = boundary_amplitudes
    potential = _ScalarPotential(support, q, amps)
    L, U = map(float, support)
    kk = float(k) / float(sigma_eff)

    def g(z):
        return potential.gradient(z) - kk

    left, right = map(float, window)
    gl, gr = float(g(left)), float(g(right))

    # q' is monotone by construction.  Expand only toward an infinite side
    # when the tilted stationary point lies beyond the original material
    # window.  A truly linear asymptote that never crosses ``kk`` means the
    # requested exponential moment diverges.
    degree = _poly_degree_exact(q)
    if gr < 0.0 and not np.isfinite(U):
        if degree <= 1:
            return np.inf
        step = max(right - left, 1.0)
        for _ in range(BRACKET_MAX_EXPAND * 2):
            candidate = right + step
            if not np.isfinite(candidate):
                return np.inf
            right = candidate
            gr = float(g(right))
            if np.isfinite(gr) and gr >= 0.0:
                break
            step *= BRACKET_STEP_GROWTH
        else:
            return np.inf
    elif gl > 0.0 and not np.isfinite(L):
        if degree <= 1:
            return np.inf
        step = max(right - left, 1.0)
        for _ in range(BRACKET_MAX_EXPAND * 2):
            candidate = left - step
            if not np.isfinite(candidate):
                return np.inf
            left = candidate
            gl = float(g(left))
            if np.isfinite(gl) and gl <= 0.0:
                break
            step *= BRACKET_STEP_GROWTH
        else:
            return np.inf

    gl, gr = float(g(left)), float(g(right))
    if not (not np.isnan(gl) and not np.isnan(gr) and gl <= 0.0 <= gr):
        # A finite endpoint can legitimately own the tilted minimum; those
        # cases are handled by the ordinary full finite-support window.
        return np.nan

    z_star = float(
        brentq(g, left, right, xtol=1e-10, rtol=4 * np.finfo(float).eps, maxiter=200)
    )
    curvature = float(potential.curvature(z_star))
    if not np.isfinite(curvature) or curvature <= 0.0:
        return np.nan
    local_scale = float(1.0 / np.sqrt(curvature))

    # Taylor coefficients of the polynomial part about the tilted mode.
    deriv_coeff = [
        float(
            np.polynomial.polynomial.polyval(
                z_star, np.polynomial.polynomial.polyder(q, order)
            )
        )
        / float(factorial(order))
        for order in range(1, q.size)
    ]

    aL, aU = map(float, amps)

    def delta_tilt(dx):
        # Exact Taylor expansion of the polynomial around z_star avoids
        # subtracting O(1e20) potential values to recover an O(1) local
        # difference when the saddle is remote.
        total = 0.0
        power = float(dx)
        for coeff in deriv_coeff:
            total += coeff * power
            power *= dx
        if np.isfinite(L) and np.isfinite(aL) and aL > 0.0:
            ratio = dx / (z_star - L)
            if ratio <= -1.0:
                return np.inf
            total -= aL * np.log1p(ratio)
        if np.isfinite(U) and np.isfinite(aU) and aU > 0.0:
            ratio = -dx / (U - z_star)
            if ratio <= -1.0:
                return np.inf
            total -= aU * np.log1p(ratio)
        total -= kk * dx
        return float(total)

    def relative(y, sign):
        dx = sign * local_scale * float(y)
        delta = delta_tilt(dx)
        if not np.isfinite(delta):
            return 0.0 if delta > 0.0 else np.inf
        # Convexity makes delta non-negative; tolerate a few ulps of local
        # cancellation at the stationary point.
        if delta < 0.0 and abs(delta) <= 1e-10 * max(1.0, abs(dx)):
            delta = 0.0
        if delta < 0.0:
            raise RuntimeError(
                "tilted moment lost convexity around its stationary point"
            )
        if delta > 745.0:
            return 0.0
        return float(local_scale * np.exp(-delta))

    left_extent = np.inf if not np.isfinite(L) else max(0.0, (z_star - L) / local_scale)
    right_extent = (
        np.inf if not np.isfinite(U) else max(0.0, (U - z_star) / local_scale)
    )

    # ``quad`` can miss an O(1)-wide peak at the origin when asked to
    # integrate over a finite interval tens of thousands of local scales
    # wide.  The integrand is identically zero beyond a finite endpoint, so
    # promoting a very remote endpoint to infinity is both mathematically
    # harmless at float64 precision and much better conditioned numerically.
    if np.isfinite(left_extent) and left_extent > 64.0:
        left_extent = np.inf
    if np.isfinite(right_extent) and right_extent > 64.0:
        right_extent = np.inf

    left_val, _left_err = quad(
        lambda y: relative(y, -1.0),
        0.0,
        left_extent,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=QUAD_LIMIT,
    )
    right_val, _right_err = quad(
        lambda y: relative(y, 1.0),
        0.0,
        right_extent,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=QUAD_LIMIT,
    )
    relative_mass = float(left_val + right_val)
    if not np.isfinite(relative_mass) or relative_mass <= 0.0:
        return np.nan

    q_star = potential.value(z_star)
    tilted_star = float(q_star - kk * z_star + kk * float(mu_eff))
    return float(-tilted_star + np.log(relative_mass))


def _log_raw_moment_exp(
    base_support, q_poly, boundary_amplitudes, window, mu_eff, sigma_eff, k, terms, /
):
    """Return ``log E[exp(k X)]`` using a saddle-point shifted integral.

    Parameters
    ----------
    base_support : tuple of float
        Support of the latent log variable.
    q_poly : numpy.ndarray
        Polynomial potential coefficients.
    boundary_amplitudes : numpy.ndarray
        Finite-boundary logarithmic amplitudes.
    window : array_like, shape (2,)
        Numerical quadrature window.
    mu_eff : float
        Effective affine location of the latent variable.
    sigma_eff : float
        Effective affine scale of the latent variable.
    k : int
        Raw-moment order.
    terms : object
        Precomputed quadrature boundary terms.
    """
    kk_int = k
    if kk_int == 0:
        return 0.0

    # For positive raw moments of Y = exp(X), only the public upper tail can
    # cause divergence.  A genuinely exponential tail with limiting
    # potential slope ``r`` has E[exp(k X)] < inf exactly when k < r; at the
    # threshold k == r the tilted density no longer decays.  Check that
    # asymptotic condition before any finite-window quadrature so a divergent
    # moment cannot masquerade as a finite truncated integral.
    upper_rate = _tail_rate_from_geometry(
        base_support, q_poly, mu_eff, sigma_eff, "upper"
    )
    if np.isfinite(upper_rate) and float(kk_int) >= float(upper_rate):
        return np.inf

    kk = float(kk_int) / float(sigma_eff)
    z_star = _valley(window, base_support, q_poly, boundary_amplitudes, float(kk))

    # ``window`` is sized for the original density, not the exponentially
    # tilted integrand.  If q'(z_star) still misses the requested shift on an
    # unbounded side, the tilted saddle lies outside the stored material
    # window and the ordinary quadrature would silently omit its dominant
    # mass.  Re-solve and integrate on the full support in that case.
    potential = _ScalarPotential(base_support, q_poly, boundary_amplitudes)
    slope_miss = float(potential.gradient(z_star) - kk)
    wL, wU = map(float, window)
    support_L, support_U = map(float, base_support)
    at_left = abs(z_star - wL) <= 16.0 * abs(np.spacing(wL))
    at_right = abs(z_star - wU) <= 16.0 * abs(np.spacing(wU))
    outside_material_window = (
        at_right and slope_miss < -1e-10 and not np.isfinite(support_U)
    ) or (at_left and slope_miss > 1e-10 and not np.isfinite(support_L))
    if outside_material_window:
        expanded = _expanded_tilted_log_moment(
            base_support,
            q_poly,
            boundary_amplitudes,
            window,
            mu_eff,
            sigma_eff,
            kk_int,
        )
        if np.isposinf(expanded):
            return np.inf
        if np.isfinite(expanded):
            return float(expanded)
        raise RuntimeError(
            "exp-space moment could not resolve the tilted tail outside the fitted window"
        )

    q0z = potential.value(z_star)
    m = float((kk_int * (z_star - mu_eff) / sigma_eff) - q0z)
    if not np.isfinite(m):
        m = 0.0

    q_poly_tilt = q_poly.copy()
    if q_poly_tilt.size < 2:
        q_poly_tilt = np.pad(q_poly_tilt, (0, 2 - q_poly_tilt.size))
    q_poly_tilt[0] += kk * float(mu_eff)
    q_poly_tilt[1] -= kk

    q_poly_scaled = q_poly_tilt.copy()
    q_poly_scaled[0] += m
    val_scaled = quad_integral(
        q_poly_scaled,
        float(window[0]),
        float(window[1]),
        terms,
        mode=0,
        k=0,
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=QUAD_LIMIT,
    )
    if not np.isfinite(val_scaled) or val_scaled <= 0.0:
        return -np.inf
    return float(m + np.log(val_scaled))


def _relative_centered_moment_exp(
    base_support, q_poly, boundary_amplitudes, window, mu_eff, sigma_eff, log_mean, k, /
):
    """Return ``E[(exp(X-log_mean)-1)^k]`` by direct centered quadrature.

    This is used only for narrow transformed distributions, where raw log-moment
    combinations cancel at orders three and four.  In that regime
    ``X-log_mean`` is small throughout the material integration window, so
    ``expm1`` resolves the centered transformed variable directly.

    Parameters
    ----------
    base_support : tuple of float
        Support of the latent log variable.
    q_poly : numpy.ndarray
        Polynomial potential coefficients.
    boundary_amplitudes : numpy.ndarray
        Finite-boundary logarithmic amplitudes.
    window : array_like, shape (2,)
        Numerical quadrature window.
    mu_eff : float
        Effective affine location of the latent variable.
    sigma_eff : float
        Effective affine scale of the latent variable.
    log_mean : float
        Logarithm of the dimensional mean.
    k : int
        Centered-moment order.
    """
    kk = k
    potential = _ScalarPotential(base_support, q_poly, boundary_amplitudes)

    # The centered relative variable can be O(1e-8) or smaller in the exact
    # regime where this helper is needed.  Integrating u**k directly would
    # place the whole integral far below QUAD_EPSABS for k=2..4.  Scale u by
    # its material-window magnitude so the adaptive quadrature sees an O(1)
    # integrand, then restore the dimensional relative moment afterwards.
    endpoints = window
    x_end = (endpoints - float(mu_eff)) / float(sigma_eff)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        u_end = np.expm1(x_end - float(log_mean))
    u_scale = float(np.max(np.abs(u_end)))
    if not np.isfinite(u_scale) or u_scale <= 0.0:
        u_scale = 1.0

    def integrand(z):
        q = potential.value(z)
        if not np.isfinite(q):
            return 0.0
        x = (float(z) - float(mu_eff)) / float(sigma_eff)
        u = np.expm1(x - float(log_mean)) / u_scale
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            value = (u**kk) * np.exp(-q)
        return float(value) if np.isfinite(value) else 0.0

    value, _ = quad(
        integrand,
        float(window[0]),
        float(window[1]),
        epsabs=QUAD_EPSABS,
        epsrel=QUAD_EPSREL,
        limit=QUAD_LIMIT,
    )
    return float(value * (u_scale**kk))


def _internal_geometry(struct, /):
    """Return canonical geometry and affine map of one fitted state.

    Parameters
    ----------
    struct : numpy.void or Mapping
        Validated fitted state whose field arrays are already canonical.

    Returns
    -------
    tuple
        ``(support, canonical_boundary_amplitudes, mu_eff, sigma_eff)``.
    """
    support = struct["canonical_support"]
    amps = struct["boundary_amplitudes"]
    direction = float(struct["fit_direction"])
    canonical = amps.copy() if direction > 0.0 else amps[::-1].copy()
    center = float(struct["fit_center"])
    scale = float(struct["fit_scale"])
    mu_eff = -direction * center / scale
    sigma_eff = direction / scale
    return support, canonical, mu_eff, sigma_eff


def _univariate_canonical_raw_moment(struct, k, /):
    """Compute ``E[Z^k]`` directly in the fitted canonical coordinate.

    Parameters
    ----------
    struct : numpy.void or mapping
        Packed fitted univariate state.
    k : int
        Boundary-validated nonnegative canonical raw-moment order.
    """
    kk = k
    if kk == 0:
        return 1.0
    window = struct["window"]
    q_poly = struct["q_poly"]
    support, boundary_amplitudes, _mu_eff, _sigma_eff = _internal_geometry(struct)
    terms = _terms_for_quad(support, boundary_amplitudes)
    return float(
        quad_integral(
            q_poly,
            float(window[0]),
            float(window[1]),
            terms,
            mode=0,
            k=kk,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
        )
    )


def _univariate_raw_moment(struct, k, /):
    """Compute the *k*-th raw moment in user coordinates from a fitted state struct.

    Extracts the necessary arrays and affine parameters from *struct* and
    delegates to :func:`_raw_moment_identity`.

    Parameters
    ----------
    struct : numpy.void or Mapping
        Packed fitted state.
    k : int
        Boundary-validated nonnegative moment order.

    Returns
    -------
    float

    """
    kk = k
    if kk == 0:
        return 1.0

    window = struct["window"]
    q_poly = struct["q_poly"]
    base_support, boundary_amplitudes, mu_eff, sigma_eff = _internal_geometry(struct)
    return float(
        _raw_moment_identity(
            base_support, q_poly, boundary_amplitudes, window, mu_eff, sigma_eff, kk
        )
    )


def _exp_stats_from_log_moments(log_moments, relative_centered_moment, subject, /):
    """Build exp-space summary statistics from the first four log raw moments.

    Shared by the single-component and mixture exp-space paths, which differ
    only in how they obtain ``log E[Y^k]`` and the narrow-law relative
    centered moments.

    Positive infinity is a legitimate log-moment result for transformed laws
    with a divergent moment, or for a finite mathematical moment that exceeds
    float64's representable range.  Such cases propagate to infinite
    dimensional summaries instead of being misclassified as a degenerate
    zero-variance law.  ``NaN`` and negative infinity remain numerical errors.

    For a narrow transformed law the centered combinations are differences of
    nearly equal ``O(1)`` log moments, so ``d2 = l2 - 2*l1`` loses most of its
    significant digits.  Below ``EXP_NARROW_D2_THRESHOLD`` the relative
    centered moments are therefore integrated directly instead of being
    reconstructed from ``expm1`` of canceling differences.

    Parameters
    ----------
    log_moments : sequence of float
        ``(l1, l2, l3, l4)``, the logarithms of ``E[Y^k]`` for ``k = 1..4``.
    relative_centered_moment : callable
        ``f(log_mean, k) -> float`` returning the ``k``-th centered moment in
        units of ``mean**k``.  Called only on the narrow-law branch.
    subject : str
        Noun phrase naming the law, used in error messages (for example
        ``"exp-space"`` or ``"exp-space mixture"``).

    Returns
    -------
    dict
        ``mean``, ``var``, ``std``, ``skew``, ``kurt``, ``log_mean``, ``cv``.

    Raises
    ------
    RuntimeError
        If a log moment is ``NaN`` or negative infinity, or if a finite
        relative variance is non-positive.
    """
    l1, l2, l3, l4 = (float(v) for v in log_moments)
    vals = np.asarray((l1, l2, l3, l4), dtype=np.float64)
    if np.any(np.isnan(vals)) or np.any(np.isneginf(vals)):
        raise RuntimeError(f"invalid log moment in {subject} statistics")

    # Once the first raw moment itself is infinite, finite-dimensional
    # variance and standardized shape are undefined.  Expose that fact
    # directly rather than evaluating inf-inf combinations below.
    if np.isposinf(l1):
        return {
            "mean": np.inf,
            "var": np.inf,
            "std": np.inf,
            "skew": np.nan,
            "kurt": np.nan,
            "log_mean": np.inf,
            "cv": np.nan,
        }

    d2 = float(l2 - 2.0 * l1)
    if d2 < EXP_NARROW_D2_THRESHOLD:
        rel2 = float(relative_centered_moment(l1, 2))
        rel3 = float(relative_centered_moment(l1, 3))
        rel4 = float(relative_centered_moment(l1, 4))
        cv2 = rel2
        cv = _checked_cv(cv2, subject)
        skew = float(rel3 / (cv**3))
        kurt = float(rel4 / (cv**4))
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            cv2 = float(np.expm1(d2))
            cv = _checked_cv(cv2, subject)
            e2 = cv2
            e3 = float(np.expm1(l3 - 3.0 * l1))
            e4 = float(np.expm1(l4 - 4.0 * l1))
            skew = float((e3 - 3.0 * e2) / (cv**3))
            kurt = float((e4 - 4.0 * e3 + 6.0 * e2) / (cv**4))

        # When variance is finite but a higher raw moment diverges, the
        # corresponding standardized moment is +inf.  Evaluating the raw
        # formulas literally can produce inf-inf -> NaN even though the
        # positive exp-space tail makes the central moment unambiguously
        # divergent.
        if np.isfinite(l2) and np.isposinf(l3):
            skew = np.inf
            kurt = np.inf
        elif np.isfinite(l3) and np.isposinf(l4):
            kurt = np.inf

    log_max = float(np.log(np.finfo(np.float64).max))
    with np.errstate(over="ignore", under="ignore"):
        mean = float(np.exp(l1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_var = float(2.0 * l1 + np.log(cv2))
    var = float(np.exp(log_var)) if log_var <= log_max else np.inf
    log_std = 0.5 * log_var
    std = float(np.exp(log_std)) if log_std <= log_max else np.inf

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "log_mean": l1,
        "cv": cv,
    }


def _checked_cv(cv2, subject, /):
    """Return ``sqrt(cv2)`` after rejecting a degenerate relative variance.

    Parameters
    ----------
    cv2 : float
        Squared coefficient of variation.
    subject : str
        Noun phrase naming the law, used in the error message.
    """
    if np.isnan(cv2) or cv2 <= 0.0:
        raise RuntimeError(f"Degenerate {subject} variance (relative var={cv2}).")
    if np.isposinf(cv2):
        return np.inf
    if not np.isfinite(cv2):
        raise RuntimeError(f"Invalid {subject} variance (relative var={cv2}).")
    return float(np.sqrt(cv2))


def _exp_moment_from_stats(k, stats, standardized, /):
    """Return a low-order exp-space central/standardized moment, or ``None``.

    Orders 0--4 are already available in the cached statistics, so they are
    answered directly rather than re-integrated.  ``None`` means the caller
    should fall back to its general moment machinery.

    Parameters
    ----------
    k : int
        Non-negative moment order.
    stats : Mapping
        Cached exp-space statistics with ``var``, ``std``, ``skew``, ``kurt``.
    standardized : bool
        Whether the standardized rather than central moment is requested.

    Returns
    -------
    float or None
    """
    if k > 4:
        return None
    if standardized:
        # Standardized shape is scale-free.  A huge location shift in log
        # space can overflow the dimensionful variance while leaving the
        # coefficient of variation and standardized moments finite.  Treat
        # only a genuinely divergent relative variance as undefined.
        if k >= 2 and not np.isfinite(float(stats["cv"])):
            return np.nan
        return (1.0, 0.0, 1.0, float(stats["skew"]), float(stats["kurt"]))[k]

    var = float(stats["var"])
    if k >= 3 and np.isfinite(float(stats["mean"])) and np.isposinf(var):
        # Exp-space laws are positive.  With a finite mean and infinite
        # variance, every higher central moment is also +inf; avoid the
        # indeterminate ``nan * inf`` reconstruction from standardized shape.
        return np.inf
    return (
        1.0,
        0.0,
        var,
        float(stats["skew"] * (stats["std"] ** 3)),
        float(stats["kurt"] * (stats["std"] ** 4)),
    )[k]
