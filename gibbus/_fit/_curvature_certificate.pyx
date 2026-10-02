# cython: language_level=3
"""Compiled rigorous full-curvature certificate.

Decides the same question as the exact-dyadic separator
(:func:`gibbus._fit.separation._separate_full_curvature`): is the full
curvature

    q''(z) + a_L / (z - L)^2 + a_U / (U - z)^2

at least ``-tau`` on the closed support (tails included), for the exact real
values encoded by the binary64 inputs?  Clearing the denominators of the
strictly positive amplitudes (``D = (z-L)^2 (U-z)^2`` over the active ones)
turns this into nonnegativity of one polynomial ``T = R + tau D`` with
``R = q'' D + a_L D/(z-L)^2 + a_U D/(U-z)^2`` -- exactly the separator's
tolerance-shifted sign polynomial.

Each piece of the support is mapped onto ``[0, 1]`` (``x = z - L`` or
``x = U - z``; the real line is split at 0).  On a half-line
``x = s / (1 - s)`` and ``T(x) (1 - s)^d`` has Bernstein coefficients
``t_j / C(d, j)`` directly (``t_j`` the coefficients of ``T`` in ``x``); on
a bounded interval ``x = w s`` and the usual power-to-Bernstein map applies.
Bernstein coefficients bound a polynomial on its interval and interpolate it
at the ends, so the search subdivides (de Casteljau at ``s = 1/2``) until
every piece has all coefficients certifiably nonnegative (feasible), some
endpoint value is certifiably negative (violated), or the depth budget runs
out (uncertain; the caller then falls back to the exact separator).

Every floating-point operation carries a rigorous running error bound
(Higham's ``gamma_n`` bounds for sums of products, one unit roundoff per
rounded operation otherwise), so "certifiably" means after subtracting the
accumulated bound.  The whole decision runs without the GIL on stack arrays.
"""

from libc.math cimport fabs, isfinite

cdef extern from * nogil:
    """
    #define GIBBUS_CERT_MAXC 64
    #define GIBBUS_CERT_STACK 72
    """
    int MAXC "GIBBUS_CERT_MAXC"
    int STACK "GIBBUS_CERT_STACK"

cdef double _U = 1.1102230246251565e-16          # unit roundoff 2^-53
cdef double _SAFETY = 1.0 + 1e-6


cdef inline double _gamma(int n) noexcept nogil:
    cdef double nu = n * _U
    return nu / (1.0 - nu)


cdef struct Poly:
    int m                       # number of coefficients
    double v[64]
    double e[64]


cdef inline void _set_const(Poly* p, double value, double error) noexcept nogil:
    p.m = 1
    p.v[0] = value
    p.e[0] = error


cdef inline void _mul(const Poly* a, const Poly* b, Poly* out) noexcept nogil:
    """``out = a * b`` with the dot-product error bound per coefficient."""
    cdef int i, j, k, terms
    cdef double s, mag, prop
    out.m = a.m + b.m - 1
    for k in range(out.m):
        s = 0.0
        mag = 0.0
        prop = 0.0
        terms = 0
        for i in range(a.m):
            j = k - i
            if j < 0 or j >= b.m:
                continue
            s += a.v[i] * b.v[j]
            mag += fabs(a.v[i] * b.v[j])
            prop += fabs(a.v[i]) * b.e[j] + a.e[i] * fabs(b.v[j]) + a.e[i] * b.e[j]
            terms += 1
        out.v[k] = s
        out.e[k] = prop + _gamma(terms + 1) * mag


cdef inline void _axpy(
    double alpha, double alpha_err, const Poly* x, Poly* y
) noexcept nogil:
    """``y += alpha x`` (alpha with its own error bound)."""
    cdef int i
    cdef double prod, sumv
    if x.m > y.m:
        for i in range(y.m, x.m):
            y.v[i] = 0.0
            y.e[i] = 0.0
        y.m = x.m
    for i in range(x.m):
        prod = alpha * x.v[i]
        sumv = y.v[i] + prod
        y.e[i] = (y.e[i] + fabs(alpha) * x.e[i] + alpha_err * (fabs(x.v[i]) + x.e[i])
                  + _U * fabs(prod) + _U * fabs(sumv))
        y.v[i] = sumv


cdef inline void _taylor_shift(Poly* p, double h) noexcept nogil:
    """``p(x) <- p(h + x)`` by repeated synthetic division (``h`` exact)."""
    cdef int i, j
    cdef double prod, sumv
    if h == 0.0:
        return
    for i in range(p.m - 1):
        for j in range(p.m - 2, i - 1, -1):
            prod = h * p.v[j + 1]
            sumv = p.v[j] + prod
            p.e[j] = p.e[j] + fabs(h) * p.e[j + 1] + _U * fabs(prod) + _U * fabs(sumv)
            p.v[j] = sumv


cdef inline void _reflect(Poly* p) noexcept nogil:
    """``p(x) <- p(-x)`` (exact)."""
    cdef int i
    for i in range(1, p.m, 2):
        p.v[i] = -p.v[i]


cdef inline void _trim(Poly* p) noexcept nogil:
    """Drop trailing coefficients that are exactly zero with zero error."""
    while p.m > 1 and p.v[p.m - 1] == 0.0 and p.e[p.m - 1] == 0.0:
        p.m -= 1


cdef inline void _two_diff(
    double a, double b, double* value, double* error
) noexcept nogil:
    """``a - b`` rounded, with the exact magnitude of its rounding error."""
    cdef double s = a - b
    cdef double bv = s - a
    value[0] = s
    error[0] = fabs((a - (s - bv)) + (-b - bv))


cdef inline void _affine_square(
    double d, double d_err, double sign, Poly* out
) noexcept nogil:
    """``out = (d + sign x)^2``."""
    cdef Poly lin
    lin.m = 2
    lin.v[0] = d
    lin.v[1] = sign
    lin.e[0] = d_err
    lin.e[1] = 0.0
    _mul(&lin, &lin, out)


cdef int _build_local(const double* q2, int nq, double z0, bint reflect,
                      bint lower_active, double lower, double a_lower,
                      bint upper_active, double upper, double a_upper,
                      double tau, Poly* t) noexcept nogil:
    """``T = R + tau D`` in the local variable ``x = z - z0`` (``z0 - z`` if ``reflect``).

    ``R = q'' D + a_L D/(z-L)^2 + a_U D/(U-z)^2`` over the active amplitudes,
    ``D`` the product of their squared distances; the distances are affine
    in ``x`` with offsets carrying their exact rounding errors.
    """
    cdef Poly p, d, fl, fu, one, tmp
    cdef double dl = 0.0, dl_err = 0.0, du = 0.0, du_err = 0.0
    cdef int i
    if nq > MAXC - 6:
        return 1
    p.m = nq
    for i in range(nq):
        p.v[i] = q2[i]
        p.e[i] = 0.0
    _taylor_shift(&p, z0)
    if reflect:
        _reflect(&p)
    _set_const(&one, 1.0, 0.0)
    _set_const(&d, 1.0, 0.0)
    if lower_active:
        # z - L = (z0 - L) + x  (or - x when reflected)
        _two_diff(z0, lower, &dl, &dl_err)
        _affine_square(dl, dl_err, -1.0 if reflect else 1.0, &fl)
        _mul(&d, &fl, &tmp)
        d = tmp
    if upper_active:
        # U - z = (U - z0) - x  (or + x when reflected)
        _two_diff(upper, z0, &du, &du_err)
        _affine_square(du, du_err, 1.0 if reflect else -1.0, &fu)
        _mul(&d, &fu, &tmp)
        d = tmp
    _mul(&p, &d, t)
    if lower_active:
        _axpy(a_lower, 0.0, &fu if upper_active else &one, t)
    if upper_active:
        _axpy(a_upper, 0.0, &fl if lower_active else &one, t)
    if tau != 0.0:
        _axpy(tau, 0.0, &d, t)
    _trim(t)
    return 0


cdef inline double _binom(int n, int k) noexcept nogil:
    cdef double c = 1.0
    cdef int i
    if k < 0 or k > n:
        return 0.0
    if k > n - k:
        k = n - k
    for i in range(1, k + 1):
        c = c * (n - k + i) / i
    return c


cdef int _to_bernstein_half(const Poly* t, Poly* b) noexcept nogil:
    """Half-line: ``b_j = t_j / C(d, j)`` for ``x = s / (1 - s)``."""
    cdef int d = t.m - 1, j
    cdef double c, v
    b.m = t.m
    for j in range(t.m):
        c = _binom(d, j)            # exact for the degrees used here
        v = t.v[j] / c
        b.v[j] = v
        b.e[j] = t.e[j] / c + _U * fabs(v)
    return 0


cdef int _to_bernstein_bounded(
    const Poly* t, double width, double width_err, Poly* b
) noexcept nogil:
    """Bounded: ``x = width s``; power-to-Bernstein on ``[0, 1]``."""
    cdef Poly u
    cdef int d = t.m - 1, j, k, terms
    cdef double wpow = 1.0, wpow_err = 0.0, prod, s, mag, prop, weight
    u.m = t.m
    for j in range(t.m):
        prod = t.v[j] * wpow
        u.v[j] = prod
        u.e[j] = (
            t.e[j] * wpow
            + fabs(t.v[j]) * wpow_err
            + t.e[j] * wpow_err
            + _U * fabs(prod)
        )
        # next power of the width, with an absolute error bound
        prod = wpow * width
        wpow_err = (
            wpow_err * width + wpow * width_err + wpow_err * width_err + _U * fabs(prod)
        )
        wpow = prod
    b.m = t.m
    for k in range(t.m):
        s = 0.0
        mag = 0.0
        prop = 0.0
        terms = 0
        for j in range(k + 1):
            weight = _binom(k, j) / _binom(d, j)
            s += weight * u.v[j]
            mag += fabs(weight * u.v[j])
            prop += weight * u.e[j]
            terms += 1
        # weight itself carries up to two roundings.
        b.v[k] = s
        b.e[k] = prop + _gamma(terms + 3) * mag
    return 0


cdef int _search(const Poly* root, int max_depth, int max_leaves) noexcept nogil:
    """1 feasible, 0 violated, -1 uncertain."""
    cdef Poly stack[72]
    cdef int depth[72]
    cdef int top = 0, leaves = 0, i, j, m, dep
    cdef double lo, v, e
    cdef Poly node, left, right, work
    stack[0] = root[0]
    depth[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = stack[top]
        dep = depth[top]
        m = node.m
        lo = 0.0
        for i in range(m):
            v = node.v[i] - _SAFETY * node.e[i]
            if i == 0 or v < lo:
                lo = v
        if lo >= 0.0:
            continue
        # Endpoint values are exact Bernstein coefficients: a certified
        # negative one is a certified violation.
        if (
            node.v[0] + _SAFETY * node.e[0] < 0.0
            or node.v[m - 1] + _SAFETY * node.e[m - 1] < 0.0
        ):
            return 0
        leaves += 1
        if dep >= max_depth or leaves >= max_leaves or top + 2 > STACK:
            return -1
        # de Casteljau at 1/2: left = first column, right = last row.
        work = node
        left.m = m
        right.m = m
        left.v[0] = work.v[0]
        left.e[0] = work.e[0]
        right.v[m - 1] = work.v[m - 1]
        right.e[m - 1] = work.e[m - 1]
        for j in range(1, m):
            for i in range(m - j):
                v = 0.5 * (work.v[i] + work.v[i + 1])
                e = 0.5 * (work.e[i] + work.e[i + 1]) + _U * fabs(v)
                work.v[i] = v
                work.e[i] = e
            left.v[j] = work.v[0]
            left.e[j] = work.e[0]
            right.v[m - 1 - j] = work.v[m - 1 - j]
            right.e[m - 1 - j] = work.e[m - 1 - j]
        stack[top] = right
        depth[top] = dep + 1
        top += 1
        stack[top] = left
        depth[top] = dep + 1
        top += 1
    return 1


cdef int _piece(const double* q2, int nq, double z0, double z1, bint reflect,
                bint tail, bint lo_act, double lower, double a_lower, bint hi_act,
                double upper, double a_upper, double tau, int max_depth,
                int max_leaves) noexcept nogil:
    """Certify one piece: ``[z0, z1]`` exactly, or a tail from ``z0``.

    The width ``z1 - z0`` enters with the exact error of its rounding, so
    consecutive pieces cover the support without gaps.
    """
    cdef Poly t, b
    cdef double width, width_err
    if _build_local(q2, nq, z0, reflect, lo_act, lower, a_lower, hi_act, upper, a_upper,
                    tau, &t) != 0:
        return -1
    if tail:
        _to_bernstein_half(&t, &b)
    else:
        _two_diff(z1, z0, &width, &width_err)
        _to_bernstein_bounded(&t, width, width_err, &b)
    return _search(&b, max_depth, max_leaves)


cdef int _certify(
    const double* q2,
    int nq,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double tau,
    int max_depth,
    int max_leaves,
) noexcept nogil:
    """Decide full-curvature feasibility: 1 feasible, 0 violated, -1 uncertain.

    The data region ``[-8, 8]`` of the canonical coordinate (clipped to the
    support; the whole of a bounded support) is covered by pieces of width at
    most 1, each expanded about its own left end, so no Bernstein conversion
    amplifies coefficients by a large width; the unbounded remainders are
    certified as half-lines through ``x = s / (1 - s)``.
    """
    cdef bint lo_fin = isfinite(lower), hi_fin = isfinite(upper)
    cdef bint lo_act = isfinite(a_lower) and a_lower > 0.0
    cdef bint hi_act = isfinite(a_upper) and a_upper > 0.0
    cdef double a, b, width, z0, z1, step
    cdef int status, i, n = nq, pieces
    for i in range(nq):
        if not isfinite(q2[i]):
            return -1
    while n > 1 and q2[n - 1] == 0.0:
        n -= 1
    if (lo_act and not lo_fin) or (hi_act and not hi_fin):
        return -1
    if lo_fin and hi_fin:
        a = lower
        b = upper
    else:
        a = lower if lo_fin else -8.0
        b = upper if hi_fin else 8.0
        if lo_fin and a > b:
            b = a
        if hi_fin and b < a:
            a = b
    width = b - a
    pieces = <int> (width + 0.999999)
    if pieces < 1:
        pieces = 1
    if pieces > 64:
        pieces = 64
    step = width / pieces
    # Finite pieces [z0, z1]; the last ends exactly at b.
    if width > 0.0:
        for i in range(pieces):
            z0 = a + i * step
            z1 = b if i == pieces - 1 else a + (i + 1) * step
            status = _piece(q2, n, z0, z1, False, False, lo_act, lower, a_lower,
                            hi_act, upper, a_upper, tau, max_depth, max_leaves)
            if status != 1:
                return status
    if not hi_fin:
        status = _piece(q2, n, b, 0.0, False, True, lo_act, lower, a_lower,
                        hi_act, upper, a_upper, tau, max_depth, max_leaves)
        if status != 1:
            return status
    if not lo_fin:
        status = _piece(q2, n, a, 0.0, True, True, lo_act, lower, a_lower,
                        hi_act, upper, a_upper, tau, max_depth, max_leaves)
        if status != 1:
            return status
    return 1


def certify_full_curvature(
    q_d2,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double tolerance,
    int max_depth=48,
    int max_leaves=4096,
):
    """Return 1 (feasible), 0 (violated) or -1 (uncertain).

    Parameters
    ----------
    q_d2 : array_like
        Ascending ordinary curvature coefficients (canonical coordinate).
    lower, upper : float
        Canonical support endpoints (may be infinite).
    a_lower, a_upper : float
        Boundary amplitudes; ``nan`` when the basis is absent.
    tolerance : float
        The separator's feasibility tolerance ``tau``.
    max_depth, max_leaves : int, optional
        Subdivision budget before answering "uncertain".
    """
    cdef const double[::1] q = __import__("numpy").ascontiguousarray(
        q_d2, dtype="float64"
    )
    cdef int nq = q.shape[0]
    cdef int status
    if nq == 0:
        return -1
    with nogil:
        status = _certify(&q[0], nq, lower, upper, a_lower, a_upper, tolerance,
                          max_depth, max_leaves)
    return status
