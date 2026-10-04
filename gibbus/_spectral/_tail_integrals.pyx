# cython: language_level=3
"""Compiled exact component tail masses.

``TailIntegrator.log_mass`` integrates one tail through QUADPACK with a
C-level callback.  ``TailIntegrator.log_masses`` evaluates many tail masses
in one nogil call: the most extreme point's mass by an adaptive
Gauss--Kronrod integration of the same relative integrand, then every other
point by adding the (positive) integral between consecutive points in log
space, so each mass costs one short panel instead of a full integration.
Summing positive terms keeps the relative accuracy of the pieces.
"""

import sys
import threading
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad
cimport numpy as cnp

from libc.math cimport exp, fabs, isfinite, log, log1p, INFINITY, nextafter
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_Destructor

cnp.import_array()

cdef const char* _CAPSULE_NAME = b"gibbus._tail_integrals.ud"

cdef struct TailUd:
    Py_ssize_t nq
    double* q
    double lower
    double upper
    double a_lower
    double a_upper
    double mu_eff
    double sigma_eff
    double log_jac
    double x
    double endpoint
    double direction
    double distance
    double local_scale
    double qx
    double log_tiny
    int finite_endpoint


cdef void _noop_destructor(object capsule) noexcept:
    pass


cdef tuple _make_ud(object q_poly):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qarr
    qarr = np.ascontiguousarray(q_poly, dtype=np.float64).ravel()
    cdef Py_ssize_t nq = qarr.shape[0]
    if nq == 0:
        raise ValueError("q_poly must be non-empty")
    if not np.all(np.isfinite(qarr)):
        raise FloatingPointError("q_poly must contain only finite coefficients")
    cdef Py_ssize_t total = sizeof(TailUd) + nq * sizeof(double)
    cdef cnp.ndarray buf = np.zeros(total, dtype=np.uint8)
    cdef TailUd* ud = <TailUd*>buf.data
    memset(ud, 0, sizeof(TailUd))
    ud.nq = nq
    ud.q = <double*>(buf.data + sizeof(TailUd))
    memcpy(ud.q, &qarr[0], nq * sizeof(double))
    cdef object cap = PyCapsule_New(
        <void*>ud, _CAPSULE_NAME, <PyCapsule_Destructor>_noop_destructor
    )
    return buf, cap


cdef inline TailUd* _from_cap(object cap):
    return <TailUd*>PyCapsule_GetPointer(cap, _CAPSULE_NAME)


cdef inline double _polyval(const double* c, Py_ssize_t n, double x) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 0:
        return 0.0
    out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * x + c[i]
    return out


cdef inline double _polyder1(const double* c, Py_ssize_t n, double x) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 1:
        return 0.0
    out = (n - 1) * c[n - 1]
    for i in range(n - 2, 0, -1):
        out = out * x + i * c[i]
    return out


cdef inline double _q_public(TailUd* ud, double x) noexcept nogil:
    cdef double z = ud.sigma_eff * x + ud.mu_eff
    cdef double out = _polyval(ud.q, ud.nq, z) - ud.log_jac
    cdef double d
    if isfinite(ud.lower) and ud.a_lower > 0.0:
        d = x - ud.lower
        if d <= 0.0:
            return INFINITY
        out -= ud.a_lower * (ud.log_jac + log(d))
    if isfinite(ud.upper) and ud.a_upper > 0.0:
        d = ud.upper - x
        if d <= 0.0:
            return INFINITY
        out -= ud.a_upper * (ud.log_jac + log(d))
    return out


cdef inline double _q1_public(TailUd* ud, double x) noexcept nogil:
    cdef double z = ud.sigma_eff * x + ud.mu_eff
    cdef double out = ud.sigma_eff * _polyder1(ud.q, ud.nq, z)
    if isfinite(ud.lower) and ud.a_lower > 0.0:
        out -= ud.a_lower / (x - ud.lower)
    if isfinite(ud.upper) and ud.a_upper > 0.0:
        out += ud.a_upper / (ud.upper - x)
    return out


cdef inline double _q_noise(TailUd* ud, double x) noexcept nogil:
    """First-order rounding noise of ``q`` evaluated at ``x`` (absolute).

    Horner evaluation perturbs ``q`` by about ``nq * eps`` times the sum of
    the magnitudes of its terms, and the rounding of ``x`` itself by
    ``ulp(x) |q'(x)|``.  ``exp(-q)`` carries this as relative noise, so a
    quadrature tolerance below it cannot be met.
    """
    cdef double z = ud.sigma_eff * x + ud.mu_eff
    cdef double az = fabs(z)
    cdef double terms = 0.0
    cdef double d
    cdef Py_ssize_t i
    for i in range(ud.nq - 1, -1, -1):
        terms = terms * az + fabs(ud.q[i])
    terms += fabs(ud.log_jac)
    if isfinite(ud.lower) and ud.a_lower > 0.0:
        d = x - ud.lower
        if d > 0.0:
            terms += ud.a_lower * fabs(ud.log_jac + log(d))
    if isfinite(ud.upper) and ud.a_upper > 0.0:
        d = ud.upper - x
        if d > 0.0:
            terms += ud.a_upper * fabs(ud.log_jac + log(d))
    return (ud.nq * 2.220446049250313e-16 * terms
            + fabs(nextafter(x, INFINITY) - x) * fabs(_q1_public(ud, x)))


cdef api double ll_tail_relative(int n, double* xx, void* user_data) noexcept nogil:
    cdef TailUd* ud = <TailUd*>user_data
    cdef double v = xx[0]
    cdef double t, qt, exponent, ev
    if ud.finite_endpoint:
        ev = exp(-v)
        t = ud.endpoint - ud.direction * ud.distance * ev
        if t == ud.endpoint:
            return 0.0
        qt = _q_public(ud, t)
        if not isfinite(qt):
            return 0.0
        exponent = -v + ud.qx - qt
    else:
        t = ud.x + ud.direction * ud.local_scale * v
        qt = _q_public(ud, t)
        if not isfinite(qt):
            return 0.0
        exponent = ud.qx - qt
    if exponent <= ud.log_tiny:
        return 0.0
    if exponent > 700.0:
        exponent = 700.0
    return exp(exponent)


# ---------------------------------------------------------------------------
# Adaptive Gauss--Kronrod (15/7) integration in C
# ---------------------------------------------------------------------------

cdef double _TX[8]
cdef double _TWK[8]
cdef double _TWG[4]
_TX[:] = [0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
          0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
          0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
          0.207784955007898467600689403773245, 0.0]
_TWK[:] = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
           0.104790010322250183839876322541518, 0.140653259715525918745189703095821,
           0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
           0.204432940075298892414161999234649, 0.209482141084727828012999174891714]
_TWG[:] = [0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
           0.381830050505118944950369775488975, 0.417959183673469387755102040816327]


cdef inline double _relative(TailUd* ud, double v) noexcept nogil:
    """The ``ll_tail_relative`` integrand at ``v``."""
    cdef double t, qt, exponent, ev
    if ud.finite_endpoint:
        ev = exp(-v)
        t = ud.endpoint - ud.direction * ud.distance * ev
        if t == ud.endpoint:
            return 0.0
        qt = _q_public(ud, t)
        if not isfinite(qt):
            return 0.0
        exponent = -v + ud.qx - qt
    else:
        t = ud.x + ud.direction * ud.local_scale * v
        qt = _q_public(ud, t)
        if not isfinite(qt):
            return 0.0
        exponent = ud.qx - qt
    if exponent <= ud.log_tiny:
        return 0.0
    if exponent > 700.0:
        exponent = 700.0
    return exp(exponent)


cdef inline double _shifted_density(TailUd* ud, double t, double q_ref) noexcept nogil:
    """``exp(q_ref - q(t))`` for the piece integrals (0 where q is infinite)."""
    cdef double qt = _q_public(ud, t)
    cdef double exponent
    if not isfinite(qt):
        return 0.0
    exponent = q_ref - qt
    if exponent <= ud.log_tiny:
        return 0.0
    if exponent > 700.0:
        exponent = 700.0
    return exp(exponent)


cdef void _gk15(TailUd* ud, int kind, double q_ref, double a, double b,
                double* value, double* error) noexcept nogil:
    """GK15 on ``[a, b]`` of the relative (kind 0) or shifted (kind 1) integrand."""
    cdef double f[15]
    cdef double mid = 0.5 * (a + b)
    cdef double half = 0.5 * (b - a)
    cdef double kr, ga, t
    cdef int i
    for i in range(7):
        t = half * _TX[i]
        if kind == 0:
            f[2 * i] = _relative(ud, mid - t)
            f[2 * i + 1] = _relative(ud, mid + t)
        else:
            f[2 * i] = _shifted_density(ud, mid - t, q_ref)
            f[2 * i + 1] = _shifted_density(ud, mid + t, q_ref)
    f[14] = _relative(ud, mid) if kind == 0 else _shifted_density(ud, mid, q_ref)
    kr = _TWK[7] * f[14]
    ga = _TWG[3] * f[14]
    for i in range(7):
        kr += _TWK[i] * (f[2 * i] + f[2 * i + 1])
    ga += _TWG[0] * (f[2] + f[3]) + _TWG[1] * (f[6] + f[7]) + _TWG[2] * (f[10] + f[11])
    value[0] = kr * half
    error[0] = fabs((kr - ga) * half)


cdef int _adaptive(
    TailUd* ud,
    int kind,
    double q_ref,
    double* lo,
    double* hi,
    double* val,
    double* err,
    int count,
    int limit,
    double epsabs,
    double epsrel,
    double* total_out,
) noexcept nogil:
    """Global adaptive bisection over ``count`` initial intervals (arrays of ``limit``).

    Returns 0 on convergence, 1 when the interval limit was reached.
    """
    cdef double total = 0.0, total_err = 0.0, worst, a, b, mid, v1, e1, v2, e2
    cdef int i, idx
    for i in range(count):
        total += val[i]
        total_err += err[i]
    while total_err > max(epsabs, epsrel * fabs(total)):
        if count >= limit:
            total_out[0] = total
            return 1
        idx = 0
        worst = -1.0
        for i in range(count):
            if err[i] > worst:
                worst = err[i]
                idx = i
        a = lo[idx]
        b = hi[idx]
        mid = 0.5 * (a + b)
        if not (a < mid and mid < b):
            total_out[0] = total
            return 1
        _gk15(ud, kind, q_ref, a, mid, &v1, &e1)
        _gk15(ud, kind, q_ref, mid, b, &v2, &e2)
        total += v1 + v2 - val[idx]
        total_err += e1 + e2 - err[idx]
        if total_err < 0.0:
            total_err = 0.0
        hi[idx] = mid
        val[idx] = v1
        err[idx] = e1
        lo[count] = mid
        hi[count] = b
        val[count] = v2
        err[count] = e2
        count += 1
    # Re-sum to shed the running-update rounding.
    total = 0.0
    for i in range(count):
        total += val[i]
    total_out[0] = total
    return 0


cdef double _full_log_mass(TailUd* ud, double x, double epsabs, int limit, double* work,
                           int* failed) noexcept nogil:
    """Exact log tail mass beyond ``x`` (the ``log_mass`` integral, adaptively in C)."""
    cdef double qx, slope, scale, distance, rel_floor, epsrel, value, v, e, a, b, total
    cdef double* lo = work
    cdef double* hi = work + limit
    cdef double* val = work + 2 * limit
    cdef double* err = work + 3 * limit
    cdef int count = 0, status
    ud.x = x
    qx = _q_public(ud, x)
    ud.qx = qx
    if not isfinite(qx):
        return -INFINITY
    if isfinite(ud.endpoint):
        distance = fabs(x - ud.endpoint)
        if not isfinite(distance) or distance <= 0.0:
            return -INFINITY
        ud.finite_endpoint = 1
        ud.distance = distance
        ud.local_scale = 0.0
        scale = distance
    else:
        slope = fabs(_q1_public(ud, x))
        if not isfinite(slope) or slope <= 0.0:
            return -INFINITY
        scale = 1.0 / slope
        if not isfinite(scale) or scale <= 0.0:
            return -INFINITY
        ud.finite_endpoint = 0
        ud.distance = 0.0
        ud.local_scale = scale
    # Rounding floor: the query point's resolution against the integration
    # scale (``ulp(x) |q'(x)|`` itself on an infinite tail) plus ``q``'s noise.
    rel_floor = 8.0 * _q_noise(ud, x)
    if ud.finite_endpoint:
        rel_floor += (
            8.0 * fabs(nextafter(x, INFINITY) - x) / max(scale, 2.2250738585072014e-308)
        )
    epsrel = min(0.1, max(1e-11, rel_floor))
    # Doubling panels [0, 1], [1, 2], [2, 4], ... until they stop contributing.
    a = 0.0
    b = 1.0
    total = 0.0
    while count < limit // 2:
        _gk15(ud, 0, 0.0, a, b, &v, &e)
        lo[count] = a
        hi[count] = b
        val[count] = v
        err[count] = e
        count += 1
        total += v
        if b >= 32.0 and v + e <= 1e-18 * total:
            break
        if b >= 4096.0:
            break
        a = b
        b = 2.0 * b
    status = _adaptive(
        ud, 0, 0.0, lo, hi, val, err, count, limit, epsabs, epsrel, &value
    )
    if status != 0:
        failed[0] += 1
    if not isfinite(value) or value <= 0.0:
        return -INFINITY
    return -qx + log(scale) + log(value)


cdef double _log_piece(TailUd* ud, double a, double b, int limit, double* work,
                       int* failed) noexcept nogil:
    """``log int_a^b exp(-q(t)) dt`` for ``a < b`` inside the support.

    The relative tolerance is ``1e-13`` or eight times the rounding noise of
    the integrand, whichever is larger (``q`` is convex, so its slope and
    term magnitudes on the piece peak at an endpoint).
    """
    cdef double qa, qb, q_ref, v, e, value, epsrel
    cdef double* lo = work
    cdef double* hi = work + limit
    cdef double* val = work + 2 * limit
    cdef double* err = work + 3 * limit
    if not (a < b):
        return -INFINITY
    qa = _q_public(ud, a)
    qb = _q_public(ud, b)
    q_ref = qa if qa < qb else qb
    if not isfinite(q_ref):
        return -INFINITY
    _gk15(ud, 1, q_ref, a, b, &v, &e)
    lo[0] = a
    hi[0] = b
    val[0] = v
    err[0] = e
    epsrel = min(0.1, max(1e-13, 8.0 * max(_q_noise(ud, a), _q_noise(ud, b))))
    if _adaptive(ud, 1, q_ref, lo, hi, val, err, 1, limit, 0.0, epsrel, &value) != 0:
        failed[0] += 1
    if not (value > 0.0) or not isfinite(value):
        return -INFINITY
    return -q_ref + log(value)


cdef inline double _logaddexp(double a, double b) noexcept nogil:
    if a == -INFINITY:
        return b
    if b == -INFINITY:
        return a
    if a > b:
        return a + log1p(exp(b - a))
    return b + log1p(exp(a - b))


cdef class TailIntegrator:
    """Reusable component tail context with a C-level QUADPACK callback."""

    cdef object _buf
    cdef object _cap
    cdef object _llc
    cdef object _refs
    cdef object _lock
    cdef TailUd* _ud

    def __init__(self, q_poly):
        """Prepare owned polynomial data and a reusable tail-quadrature callback.

        Parameters
        ----------
        q_poly : array_like
            Nonempty potential coefficients in increasing power order.
        """
        self._buf, self._cap = _make_ud(q_poly)
        self._ud = _from_cap(self._cap)
        self._llc = LowLevelCallable.from_cython(
            sys.modules[__name__], "ll_tail_relative", self._cap
        )
        self._refs = (self._buf, self._cap, self._llc)
        self._lock = threading.Lock()

    def log_mass(
        self,
        double x,
        double endpoint,
        bint upper,
        support,
        boundary_amplitudes,
        double mu_eff,
        double sigma_eff,
        *,
        double epsabs,
        int limit,
    ):
        """Return ``(log_tail_mass, quad_message_or_None)`` for one query.

        The prepared callback data are reused between calls.  A per-context
        lock keeps the mutable scalar query fields coherent when a fitted
        component is evaluated concurrently from multiple threads.

        Parameters
        ----------
        x : float
            Tail anchor in public base coordinates.
        endpoint : float
            Outward support endpoint in the same coordinates as ``x``.
        upper : bool
            Whether the upper rather than lower tail is requested.
        support : array_like, shape (2,)
            Support bounds in public base coordinates.
        boundary_amplitudes : array_like, shape (2,)
            Lower and upper physical-side logarithmic amplitudes.
        mu_eff, sigma_eff : float
            Affine map ``z = mu_eff + sigma_eff * x`` to polynomial coordinates;
            the scale is finite and nonzero and may be negative.
        epsabs : float
            Absolute tolerance for the scaled relative-tail integral.
        limit : int
            Maximum adaptive quadrature panel count.
        """
        with self._lock:
            return self._log_mass_unlocked(
                x, endpoint, upper, support, boundary_amplitudes,
                mu_eff, sigma_eff, epsabs, limit,
            )

    cdef void _setup(self, double endpoint, bint upper, support, boundary_amplitudes,
                     double mu_eff, double sigma_eff) except *:
        cdef cnp.ndarray[cnp.float64_t, ndim=1] supp = np.asarray(
            support, dtype=np.float64
        ).reshape(-1)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] amps = np.asarray(
            boundary_amplitudes, dtype=np.float64
        ).reshape(-1)
        if supp.shape[0] != 2 or amps.shape[0] != 2:
            raise ValueError("support and boundary_amplitudes must have length 2")
        if not isfinite(sigma_eff) or sigma_eff == 0.0:
            raise ValueError("sigma_eff must be finite and nonzero")
        self._ud.lower = float(supp[0])
        self._ud.upper = float(supp[1])
        self._ud.a_lower = float(amps[0])
        self._ud.a_upper = float(amps[1])
        self._ud.mu_eff = mu_eff
        self._ud.sigma_eff = sigma_eff
        self._ud.log_jac = log(fabs(sigma_eff))
        self._ud.endpoint = endpoint
        self._ud.direction = 1.0 if upper else -1.0
        self._ud.log_tiny = -708.3964185322641   # log(float64 tiny)

    def log_masses(
        self,
        x,
        double endpoint,
        bint upper,
        support,
        boundary_amplitudes,
        double mu_eff,
        double sigma_eff,
        *,
        double epsabs,
        int limit,
    ):
        """Return ``(log_tail_masses, n_unconverged)`` for many anchors at once.

        Same values as :meth:`log_mass` at every anchor (up to the quadrature
        tolerance).  The anchors are ordered toward ``endpoint``; the most
        extreme gets a full adaptive integration and every other one adds the
        integral up to its neighbor in log space.

        Parameters
        ----------
        x : array_like
            Tail anchors in public base coordinates, flattened in the result.
        endpoint : float
            Common outward support endpoint in public base coordinates.
        upper : bool
            Whether the upper rather than lower tail is requested.
        support : array_like, shape (2,)
            Support bounds in public base coordinates.
        boundary_amplitudes : array_like, shape (2,)
            Lower and upper physical-side logarithmic amplitudes.
        mu_eff, sigma_eff : float
            Affine map from public base to polynomial coordinates; scale may be
            negative but must be finite and nonzero.
        epsabs : float
            Absolute tolerance for the initial scaled tail integral.
        limit : int
            Workspace/adaptive-panel budget, at least four.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] xs = np.ascontiguousarray(
            x, dtype=np.float64
        ).reshape(-1)
        cdef Py_ssize_t n = xs.shape[0]
        cdef cnp.ndarray[cnp.intp_t, ndim=1] order
        cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
        cdef double* work
        cdef double* op = &out[0] if n > 0 else NULL
        cdef double* xp = &xs[0] if n > 0 else NULL
        cdef Py_ssize_t* ip
        cdef Py_ssize_t i, j, prev
        cdef int failed = 0
        cdef double current, piece
        if n == 0:
            return out, 0
        if limit < 4:
            raise ValueError("limit must be at least 4")
        # Farthest anchor first: descending for the upper tail.
        order = np.argsort(-xs if upper else xs, kind="stable")
        ip = <Py_ssize_t*> &order[0]
        work = <double*> malloc(4 * limit * sizeof(double))
        if work == NULL:
            raise MemoryError("tail mass workspace")
        try:
            with self._lock:
                self._setup(
                    endpoint, upper, support, boundary_amplitudes, mu_eff, sigma_eff
                )
                with nogil:
                    j = ip[0]
                    current = _full_log_mass(
                        self._ud, xp[j], epsabs, limit, work, &failed
                    )
                    op[j] = current
                    prev = j
                    for i in range(1, n):
                        j = ip[i]
                        if xp[j] == xp[prev]:
                            op[j] = current
                            continue
                        if upper:
                            piece = _log_piece(
                                self._ud, xp[j], xp[prev], limit, work, &failed
                            )
                        else:
                            piece = _log_piece(
                                self._ud, xp[prev], xp[j], limit, work, &failed
                            )
                        current = _logaddexp(current, piece)
                        op[j] = current
                        prev = j
        finally:
            free(work)
        len(self._refs)
        return out, failed

    cdef object _log_mass_unlocked(
        self,
        double x,
        double endpoint,
        bint upper,
        support,
        boundary_amplitudes,
        double mu_eff,
        double sigma_eff,
        double epsabs,
        int limit,
    ):
        cdef double qx, slope, scale, distance, rel_floor, epsrel
        self._setup(endpoint, upper, support, boundary_amplitudes, mu_eff, sigma_eff)
        self._ud.x = x
        qx = _q_public(self._ud, x)
        self._ud.qx = qx
        if not isfinite(qx):
            return -np.inf, None

        if isfinite(endpoint):
            distance = fabs(x - endpoint)
            if not isfinite(distance) or distance <= 0.0:
                return -np.inf, None
            self._ud.finite_endpoint = 1
            self._ud.distance = distance
            self._ud.local_scale = 0.0
            rel_floor = 8.0 * (fabs(float(np.spacing(x))) / max(
                distance, float(np.finfo(np.float64).tiny)
            ) + _q_noise(self._ud, x))
            epsrel = min(0.1, max(1e-11, rel_floor))
            scale = distance
        else:
            slope = fabs(_q1_public(self._ud, x))
            if not isfinite(slope) or slope <= 0.0:
                return -np.inf, None
            scale = 1.0 / slope
            if not isfinite(scale) or scale <= 0.0:
                return -np.inf, None
            self._ud.finite_endpoint = 0
            self._ud.distance = 0.0
            self._ud.local_scale = scale
            rel_floor = 8.0 * _q_noise(self._ud, x)
            epsrel = min(0.1, max(1e-11, rel_floor))

        result = quad(
            self._llc,
            0.0,
            np.inf,
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=int(limit),
            full_output=1,
        )
        len(self._refs)
        value = float(result[0])
        message = str(result[3]) if len(result) > 3 else None
        if not np.isfinite(value) or value <= 0.0:
            return -np.inf, message
        return float(-qx + log(scale) + log(value)), message
