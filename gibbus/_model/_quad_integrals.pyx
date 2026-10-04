# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""Cython extension providing adaptive Gaussian quadrature over the log-concave potential.

This module constructs ``scipy.integrate.quad``-compatible
``LowLevelCallable`` objects that wrap the fitted unnormalized density
``exp(-q(x))`` (and weighted variants) in a C struct, then delegates
integration to SciPy's QUADPACK routines.

The one-shot :func:`quad_integral` function computes::

    integral_{L}^{U}  x^k * w(x) * exp(-q(x))  dx

where ``w(x)`` is one of:

* ``mode=0`` — constant 1 (plain moment integral)
* ``mode=1`` — ``log(d(x))`` (log-distance weight; used for boundary-term
  gradient integrals)
* ``mode=2`` — ``log(d_i(x)) * log(d_j(x))`` (boundary log-product
  weights used by the Fisher matrix)
and distances are selected by ``t_index`` / ``t_index2`` entries of *terms*.

For repeated model-moment work, :class:`PreparedQuad` retains the potential,
boundary descriptors, and ``LowLevelCallable`` objects across synchronous
``scipy.integrate.quad`` calls while changing only the per-integral mode/order
fields.  A prepared instance is intentionally not shareable by concurrent
quadrature calls because those fields are mutable callback state.

Pipeline position
-----------------
Called from model/post-fit code for moment computation and normalization.

Rebuild notes
-------------
After editing this file, rebuild the extension::

    python setup.py build_ext --inplace

and run the package smoke tests before committing.
"""

import sys
import numpy as np
from scipy.integrate import quad
from scipy import LowLevelCallable
cimport numpy as cnp

from libc.math cimport exp, log
from libc.string cimport memcpy, memset
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_GetPointer, PyCapsule_Destructor

cnp.import_array()

cdef const char* _UD_CAPSULE_NAME = b"gibbus._quad_integrals.ud"

cdef struct Ud:
    Py_ssize_t nq
    double* q

    int n_terms
    double nu[2]
    double sign[2]
    double a[2]

    int mode
    long k
    int t_index
    int t_index2


# ---------- numpy-backed Ud allocation (no malloc/free) ----------
#
# The Ud struct and its variable-length q[] data are stored together
# inside a single contiguous numpy byte array.  Layout:
#
#     [ Ud struct  |  q[0] q[1] ... q[nq-1] ]
#
# ud.q points into the same buffer, right after the struct.
# The numpy array is kept alive by the caller (stored alongside the
# LowLevelCallable), so the memory is never freed while in use.
# A PyCapsule wraps the raw pointer for SciPy's LowLevelCallable API
# but has NO destructor — the numpy array owns the memory.

cdef void _ud_noop_destructor(object capsule) noexcept:
    pass  # numpy array owns the memory; nothing to free

cdef tuple _ud_make(object q_poly):
    """Allocate a Ud struct backed by a numpy byte array.

    Returns ``(backing_array, capsule)`` where:
    * ``backing_array`` — the numpy array that owns the memory
    * ``capsule`` — a PyCapsule wrapping a Ud* (for LowLevelCallable)

    Use ``_ud_from_cap(capsule)`` to obtain the ``Ud*``.
    The caller must keep ``backing_array`` alive for as long as the
    capsule or pointer may be accessed.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qarr
    qarr = np.ascontiguousarray(q_poly, dtype=np.float64).ravel()
    cdef Py_ssize_t nq = qarr.shape[0]
    if nq == 0:
        raise ValueError("q_poly must be non-empty")

    # A non-finite coefficient here is fatal, not merely inaccurate.
    # ``exp(-polyval(q, x))`` with an ``+inf`` coefficient evaluates to NaN
    # on part of the window and to a finite value on the rest, and QUADPACK
    # can terminate the interpreter inside QUADPACK rather than producing a
    # Python exception. A non-finite model state is a numerical failure, so
    # use ``FloatingPointError`` rather than the contract-oriented
    # ``ValueError``; trust-region trial evaluation can then reject it without
    # hiding shape or API defects.
    if not np.all(np.isfinite(qarr)):
        raise FloatingPointError(
            "q_poly must be finite; got "
            f"{int(np.count_nonzero(~np.isfinite(qarr)))} non-finite "
            f"coefficient(s) of {nq}")

    cdef Py_ssize_t total = sizeof(Ud) + nq * sizeof(double)
    cdef cnp.ndarray buf = np.zeros(total, dtype=np.uint8)
    cdef Ud* ud = <Ud*> buf.data
    memset(ud, 0, sizeof(Ud))
    ud.nq = nq
    ud.q = <double*> (buf.data + sizeof(Ud))
    memcpy(ud.q, &qarr[0], nq * sizeof(double))

    cdef object cap = PyCapsule_New(<void*>ud, _UD_CAPSULE_NAME,
                                    <PyCapsule_Destructor>_ud_noop_destructor)
    return (buf, cap)


cdef inline Ud* _ud_from_cap(object cap):
    return <Ud*>PyCapsule_GetPointer(cap, _UD_CAPSULE_NAME)


cdef inline double _polyval_ptr(const double* c, Py_ssize_t n, double x) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 0:
        return 0.0
    out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * x + c[i]
    return out


cdef inline double _powi(double x, long k) noexcept nogil:
    cdef double out = 1.0
    cdef double base = x
    cdef long e = k
    while e > 0:
        if e & 1:
            out *= base
        base *= base
        e >>= 1
    return out


cdef inline double _kern_pdf(Ud* ud, double x) noexcept nogil:
    cdef int i
    cdef double expo = -_polyval_ptr(ud.q, ud.nq, x)
    cdef double d
    for i in range(ud.n_terms):
        d = ud.sign[i] * (ud.nu[i] - x)
        expo += ud.a[i] * log(d)
    return exp(expo)


cdef inline double _d_term(Ud* ud, int idx, double x) noexcept nogil:
    return ud.sign[idx] * (ud.nu[idx] - x)


cdef api double ll_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    return _kern_pdf(ud, xx[0])


cdef api double ll_xpow_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    return _powi(x, ud.k) * _kern_pdf(ud, x)


cdef api double ll_x_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    return x * _kern_pdf(ud, x)


cdef api double ll_logdist_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    cdef double d = _d_term(ud, ud.t_index, x)
    return log(d) * _kern_pdf(ud, x)


cdef api double ll_x_logdist_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    cdef double d = _d_term(ud, ud.t_index, x)
    return x * log(d) * _kern_pdf(ud, x)


cdef api double ll_xpow_logdist_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    cdef double d = _d_term(ud, ud.t_index, x)
    return _powi(x, ud.k) * log(d) * _kern_pdf(ud, x)


cdef api double ll_logprod_pdf(int n, double* xx, void* user_data) noexcept nogil:
    cdef Ud* ud = <Ud*>user_data
    cdef double x = xx[0]
    cdef double da = _d_term(ud, ud.t_index, x)
    cdef double db = _d_term(ud, ud.t_index2, x)
    return log(da) * log(db) * _kern_pdf(ud, x)


cdef object _llc(object name, object user_data_capsule):
    return LowLevelCallable.from_cython(sys.modules[__name__], name, user_data_capsule)


cdef tuple _make_llc(Ud* ud, object cap):
    # Return (LowLevelCallable, capsule) so the caller keeps the backing
    # numpy array alive for the full duration of the quad() call.
    cdef object llc
    if ud.mode == 0:
        if ud.k == 0:
            llc = _llc("ll_pdf", cap)
        elif ud.k == 1:
            llc = _llc("ll_x_pdf", cap)
        else:
            llc = _llc("ll_xpow_pdf", cap)
    elif ud.mode == 1:
        if ud.k == 0:
            llc = _llc("ll_logdist_pdf", cap)
        elif ud.k == 1:
            llc = _llc("ll_x_logdist_pdf", cap)
        else:
            llc = _llc("ll_xpow_logdist_pdf", cap)
    elif ud.mode == 2:
        llc = _llc("ll_logprod_pdf", cap)
    else:
        raise ValueError("mode must be 0, 1, or 2")
    return llc, cap


def quad_integral(
    q_poly,
    L,
    U,
    terms=None,
    *,
    mode: int = 0,
    k: int = 0,
    t_index: int = 0,
    t_index2: int = 0,
    full_output: int = 0,
    epsabs: float,
    epsrel: float,
    limit: int,
    points=None,
    weight=None,
    wvar=None,
    wopts=None,
    maxp1: int = 50,
    limlst: int = 50,
    dmin_log: float | None = None,
):
    """Compute a weighted moment integral of the unnormalized log-concave density.

    Evaluates::

        integral_{L}^{U}  x^k * w(x) * exp(-q(x))  dx

    where the weight ``w(x)`` depends on *mode*:

    * ``mode=0`` — ``w(x) = 1``
    * ``mode=1`` — ``w(x) = log(d(x))``  (log-distance weight)
    * ``mode=2`` — ``w(x) = log(d_i(x)) * log(d_j(x))``

    where each distance is selected by a row of *terms*. The integrand is wrapped in a ``LowLevelCallable`` and
    passed to ``scipy.integrate.quad``. For ``mode=1`` the integration
    endpoint is inset slightly to avoid evaluating ``log(0)``.

    Parameters
    ----------
    q_poly : array_like, shape (d+1,)
        Polynomial coefficients of the potential ``q``, constant term
        first.
    L : float
        Left endpoint of the integration interval.  Must be finite.
    U : float
        Right endpoint.  Must be finite with ``L < U``.
    terms : array_like, shape (n_terms, 3) or None, optional
        Zero-offset boundary descriptors ``[endpoint, sign, amplitude]``.
        At most 2 rows are allowed.
        ``None`` means no boundary terms (pure polynomial potential).
    mode : {0, 1, 2}, optional
        Weight type (default ``0``).
    k : int, optional
        Power of *x* in the integrand (default ``0``, i.e. plain density
        integral).  Must be ``>= 0``.
    t_index : int, optional
        First row index into *terms* selecting a boundary distance for
        ``mode=1`` or ``mode=2`` (default ``0``).
    t_index2 : int, optional
        Second row index used for ``mode=2`` (default ``0``).
    full_output : int, optional
        Passed directly to ``scipy.integrate.quad``.  When non-zero,
        the full ``quad`` output tuple is returned instead of just the
        scalar integral (default ``0``).
    epsabs : float
        Absolute error tolerance for ``quad``.  Required rather than
        defaulted: :mod:`._defaults` owns the value and every caller
        passes it, so a default here would be a second source of truth.
    epsrel : float
        Relative error tolerance.  Required, as for *epsabs*.
    limit : int
        Maximum number of sub-intervals for ``quad``.  Required, as for
        *epsabs*.
    points : array_like or None, optional
        Passed to ``scipy.integrate.quad`` as break-points.
    weight : str or None, optional
        Passed to ``scipy.integrate.quad``.
    wvar : object or None, optional
        Passed to ``scipy.integrate.quad``.
    wopts : object or None, optional
        Passed to ``scipy.integrate.quad``.
    maxp1 : int, optional
        Passed to ``scipy.integrate.quad`` (default ``50``).
    limlst : int, optional
        Passed to ``scipy.integrate.quad`` (default ``50``).
    dmin_log : float or None, optional
        Minimum distance from the boundary singularity used to inset the
        integration limit for ``mode=1``.  Defaults to
        ``numpy.nextafter(0.0, 1.0)``.

    Returns
    -------
    float or tuple
        The integral value as a ``float`` when ``full_output=0``;
        the raw ``scipy.integrate.quad`` output tuple otherwise.

    Raises
    ------
    ValueError
        If *L* or *U* are non-finite or ``L >= U``.
    ValueError
        If *mode* is not 0, 1, or 2.
    ValueError
        If *k* is negative.
    ValueError
        If *terms* does not have shape ``(n, 3)`` with ``n <= 2``.
    ValueError
        If *t_index* is out of range for the provided *terms*.
    FloatingPointError
        If *q_poly* contains a non-finite coefficient.
    ValueError
        If the inset integration interval is empty after boundary
        buffering.
    """
    L = float(L)
    U = float(U)
    if not (np.isfinite(L) and np.isfinite(U) and (L < U)):
        raise ValueError("L and U must be finite with L < U")

    mode = int(mode)
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")

    k = int(k)
    if k < 0:
        raise ValueError("k must be >= 0")

    if dmin_log is None:
        dmin_log_val = float(np.nextafter(0.0, 1.0))
    else:
        dmin_log_val = float(dmin_log)
        if not (dmin_log_val > 0.0 and np.isfinite(dmin_log_val)):
            raise ValueError("dmin_log must be a finite positive float")

    if terms is None:
        T = np.empty((0, 3), dtype=np.float64)
    else:
        T = np.asarray(terms, dtype=np.float64)
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError(
                "terms must have shape (n_terms, 3) with columns (endpoint, sign, amplitude)"
            )
        if T.shape[0] > 2:
            raise ValueError("terms must have at most 2 rows")

    n_terms = int(T.shape[0])

    if mode in (1, 2):
        t_index = int(t_index)
        if not (0 <= t_index < n_terms):
            raise ValueError("t_index out of range for provided terms")
    if mode == 2:
        t_index2 = int(t_index2)
        if not (0 <= t_index2 < n_terms):
            raise ValueError("t_index2 out of range for provided terms")

    cdef int i

    # Allocate Ud + q[] in a single numpy byte array (no malloc/free).
    _buf, _cap = _ud_make(q_poly)
    cdef Ud* ud = _ud_from_cap(_cap)
    ud.mode = mode
    ud.k = k
    ud.t_index = t_index
    ud.t_index2 = t_index2
    ud.n_terms = n_terms

    if n_terms:
        for i in range(n_terms):
            ud.nu[i] = float(T[i, 0])
            ud.sign[i] = float(T[i, 1])
            ud.a[i] = float(T[i, 2])

    f, _cap2 = _make_llc(ud, _cap)  # _buf + _cap2 keep the memory alive

    # ---- IMPORTANT: prevent early collection of _buf ----
    # _buf is the numpy byte array that backs the Ud struct.
    # The capsule (_cap/_cap2) has a no-op destructor, so the ONLY
    # thing keeping the memory valid is _buf's refcount.  Cython may
    # decrement the refcount of a local variable once it determines the
    # name is dead (no subsequent reads).  Stash _buf in a tuple that
    # remains referenced through the end of the function to guarantee
    # the backing array survives the entire quad() call.
    _refs = (_buf, _cap2)

    a = L
    b = U
    if mode in (1, 2):
        indices = (t_index,) if mode == 1 else (t_index, t_index2)
        for idx in indices:
            sig = float(T[idx, 1])
            if sig < 0.0:
                a = max(a, L + dmin_log_val, float(np.nextafter(L, np.inf)))
            else:
                b = min(b, U - dmin_log_val, float(np.nextafter(U, -np.inf)))

    if not (a < b):
        raise ValueError(
            "Buffered integration interval is empty; check L/U and boundary parameters"
        )

    out = quad(
        f, a, b,
        full_output=int(full_output),
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        limit=int(limit),
        points=points,
        weight=weight,
        wvar=wvar,
        wopts=wopts,
        maxp1=int(maxp1),
        limlst=int(limlst),
    )

    # Prevent Cython from collecting _refs (and thus _buf) before quad()
    # completes.  The `len()` call is opaque to the Cython optimizer
    # and forces _refs to be alive at this program point.
    len(_refs)

    if int(full_output) != 0:
        return out
    return float(out[0])


cdef class PreparedQuad:
    """Reusable LowLevelCallable context for repeated model-moment integrals.

    The object owns one immutable copy of the potential coefficients and
    boundary descriptors.  Integral-specific fields are mutated immediately
    before a synchronous ``scipy.integrate.quad`` call, so one instance must
    not be shared by concurrent quadrature calls.
    """

    cdef object _buf
    cdef object _cap
    cdef object _refs
    cdef object _terms
    cdef object _ll_pdf
    cdef object _ll_x
    cdef object _ll_xpow
    cdef object _ll_log
    cdef object _ll_xlog
    cdef object _ll_xpowlog
    cdef object _ll_logprod
    cdef Ud* _ud
    cdef double _L
    cdef double _U
    cdef int _n_terms

    def __init__(self, q_poly, L, U, terms=None):
        """Prepare reusable quadrature data for one finite integration window.

        Parameters
        ----------
        q_poly : array_like
            Nonempty potential coefficients in increasing power order.
        L, U : float
            Finite, strictly ordered integration bounds in fitting coordinates.
        terms : array_like, shape (n_terms, 3), or None, optional
            At most two logarithmic boundary descriptors with columns
            ``(endpoint, sign, amplitude)``. ``None`` supplies no boundary terms.
        """
        self._L = float(L)
        self._U = float(U)
        if not (np.isfinite(self._L) and np.isfinite(self._U) and self._L < self._U):
            raise ValueError("L and U must be finite with L < U")
        if terms is None:
            T = np.empty((0, 3), dtype=np.float64)
        else:
            T = np.asarray(terms, dtype=np.float64)
            if T.ndim != 2 or T.shape[1] != 3:
                raise ValueError(
                    "terms must have shape (n_terms, 3) with columns "
                    "(endpoint, sign, amplitude)"
                )
            if T.shape[0] > 2:
                raise ValueError("terms must have at most 2 rows")
        self._terms = np.ascontiguousarray(T, dtype=np.float64)
        self._n_terms = int(T.shape[0])
        self._buf, self._cap = _ud_make(q_poly)
        self._ud = _ud_from_cap(self._cap)
        self._ud.n_terms = self._n_terms
        cdef int i
        for i in range(self._n_terms):
            self._ud.nu[i] = float(T[i, 0])
            self._ud.sign[i] = float(T[i, 1])
            self._ud.a[i] = float(T[i, 2])
        self._ll_pdf = _llc("ll_pdf", self._cap)
        self._ll_x = _llc("ll_x_pdf", self._cap)
        self._ll_xpow = _llc("ll_xpow_pdf", self._cap)
        self._ll_log = _llc("ll_logdist_pdf", self._cap)
        self._ll_xlog = _llc("ll_x_logdist_pdf", self._cap)
        self._ll_xpowlog = _llc("ll_xpow_logdist_pdf", self._cap)
        self._ll_logprod = _llc("ll_logprod_pdf", self._cap)
        self._refs = (self._buf, self._cap, self._ll_pdf, self._ll_x, self._ll_xpow,
                      self._ll_log, self._ll_xlog, self._ll_xpowlog, self._ll_logprod)

    def integrate(
        self,
        *,
        mode: int = 0,
        k: int = 0,
        t_index: int = 0,
        t_index2: int = 0,
        full_output: int = 0,
        epsabs: float,
        epsrel: float,
        limit: int,
        points=None,
        weight=None,
        wvar=None,
        wopts=None,
        maxp1: int = 50,
        limlst: int = 50,
        dmin_log: float | None = None,
    ):
        """Integrate a power/log statistic using the prepared density callback.

        Parameters
        ----------
        mode : int, optional
            Statistic family: zero for powers, one for a power times one
            boundary log distance, two for a product of two log distances.
        k : int, optional
            Nonnegative power order; mode two requires zero.
        t_index, t_index2 : int, optional
            Boundary descriptor indices for logarithmic modes.
        full_output : int, optional
            Nonzero returns SciPy's diagnostic tuple instead of just the value.
        epsabs, epsrel : float
            Explicit absolute and relative quadrature tolerances.
        limit : int
            Explicit adaptive quadrature panel budget.
        points : sequence or None, optional
            Interior integration breakpoints forwarded to SciPy.
        weight : str or None, optional
            SciPy quadrature weighting mode.
        wvar, wopts : object or None, optional
            Weight parameters and cached quadrature information forwarded to
            SciPy.
        maxp1, limlst : int, optional
            SciPy weighted-quadrature moment and cycle budgets.
        dmin_log : float or None, optional
            Positive endpoint-distance floor for logarithmic statistics.
            ``None`` uses the smallest positive float64.
        """
        cdef int mode_i = int(mode)
        cdef long k_i = int(k)
        cdef int idx1 = int(t_index)
        cdef int idx2 = int(t_index2)
        cdef double dmin
        cdef double a = self._L
        cdef double b = self._U
        cdef double sig
        if mode_i not in (0, 1, 2):
            raise ValueError("mode must be 0, 1, or 2")
        if k_i < 0:
            raise ValueError("k must be >= 0")
        if mode_i in (1, 2) and not (0 <= idx1 < self._n_terms):
            raise ValueError("t_index out of range for provided terms")
        if mode_i == 2 and not (0 <= idx2 < self._n_terms):
            raise ValueError("t_index2 out of range for provided terms")
        if dmin_log is None:
            dmin = float(np.nextafter(0.0, 1.0))
        else:
            dmin = float(dmin_log)
            if not (dmin > 0.0 and np.isfinite(dmin)):
                raise ValueError("dmin_log must be a finite positive float")

        self._ud.mode = mode_i
        self._ud.k = k_i
        self._ud.t_index = idx1
        self._ud.t_index2 = idx2

        if mode_i == 0:
            if k_i == 0:
                f = self._ll_pdf
            elif k_i == 1:
                f = self._ll_x
            else:
                f = self._ll_xpow
        elif mode_i == 1:
            if k_i == 0:
                f = self._ll_log
            elif k_i == 1:
                f = self._ll_xlog
            else:
                f = self._ll_xpowlog
        else:
            if k_i != 0:
                raise ValueError("mode=2 currently supports k=0 only")
            f = self._ll_logprod

        if mode_i in (1, 2):
            sig = float(self._terms[idx1, 1])
            if sig < 0.0:
                a = max(a, self._L + dmin, float(np.nextafter(self._L, np.inf)))
            else:
                b = min(b, self._U - dmin, float(np.nextafter(self._U, -np.inf)))
        if mode_i == 2 and idx2 != idx1:
            sig = float(self._terms[idx2, 1])
            if sig < 0.0:
                a = max(a, self._L + dmin, float(np.nextafter(self._L, np.inf)))
            else:
                b = min(b, self._U - dmin, float(np.nextafter(self._U, -np.inf)))
        if not a < b:
            raise ValueError(
                "Buffered integration interval is empty; check L/U and boundary parameters"
            )
        out = quad(
            f, a, b,
            full_output=int(full_output),
            epsabs=float(epsabs),
            epsrel=float(epsrel),
            limit=int(limit),
            points=points,
            weight=weight,
            wvar=wvar,
            wopts=wopts,
            maxp1=int(maxp1),
            limlst=int(limlst),
        )
        len(self._refs)
        if int(full_output) != 0:
            return out
        return float(out[0])
