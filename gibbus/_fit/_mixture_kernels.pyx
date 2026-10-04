# cython: language_level=3
"""Compiled row reductions for natural-coordinate point fits.

* ``empirical_point_stats``: weight normalization, power moments,
  participation effective sizes and boundary-log statistics of point data in
  one pass per statistic (the arithmetic of
  ``_observations.empirical._build_empirical_stats``).

Every loop runs without the GIL over contiguous rows; the long loops (over
observations) are the inner ones, written as restrict-qualified reductions
with ``omp simd`` hints (vectorized under ``-fopenmp-simd``).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport log, fabs, NAN, isfinite
from libc.stdlib cimport malloc, free

cnp.import_array()

cdef extern from * nogil:
    """
    #include <stddef.h>
    #if defined(_MSC_VER)
      #define GIBBUS_MIX_RESTRICT __restrict
      #define GIBBUS_MIX_SIMD
      #define GIBBUS_MIX_SIMD_SUM
      #define GIBBUS_MIX_SIMD_SUM2
    #else
      #define GIBBUS_MIX_RESTRICT __restrict__
      #define GIBBUS_MIX_SIMD _Pragma("omp simd")
      #define GIBBUS_MIX_SIMD_SUM _Pragma("omp simd reduction(+:s)")
      #define GIBBUS_MIX_SIMD_SUM2 _Pragma("omp simd reduction(+:s, q)")
    #endif

    /* sum_i x[i], then x[i] *= z[i] */
    static inline double gibbus_mix_sum_advance(double *GIBBUS_MIX_RESTRICT x,
                                                const double *GIBBUS_MIX_RESTRICT z,
                                                ptrdiff_t n, int advance)
    {
        double s = 0.0;
        GIBBUS_MIX_SIMD_SUM
        for (ptrdiff_t i = 0; i < n; ++i) s += x[i];
        if (advance) {
            GIBBUS_MIX_SIMD
            for (ptrdiff_t i = 0; i < n; ++i) x[i] *= z[i];
        }
        return s;
    }

    /* (sum_i x[i], sum_i x[i]^2) */
    static inline void gibbus_mix_sum_square(const double *GIBBUS_MIX_RESTRICT x,
                                             ptrdiff_t n, double *out_s, double *out_q)
    {
        double s = 0.0, q = 0.0;
        GIBBUS_MIX_SIMD_SUM2
        for (ptrdiff_t i = 0; i < n; ++i) { s += x[i]; q += x[i] * x[i]; }
        *out_s = s;
        *out_q = q;
    }
    """
    double gibbus_mix_sum_advance(double* x, const double* z, Py_ssize_t n, int advance)
    void gibbus_mix_sum_square(
        const double* x, Py_ssize_t n, double* out_s, double* out_q
    )


# ---------------------------------------------------------------------------
# Empirical point statistics
# ---------------------------------------------------------------------------

cdef int _point_stats(
    const double* z,
    const double* raw,
    bint weighted,
    Py_ssize_t n,
    int order,
    double L,
    double U,
    bint has_lower,
    bint has_upper,
    double* w,
    double* work,
    double* moments,
    double* participation,
    double* boundary,
    double* total_out,
    double* n_eff_out,
) noexcept nogil:
    """Status: 0 ok; 1 bad weights; 2 zero total; 3 non-finite moment;
    4 lower distance; 5 upper distance; 6 non-finite boundary statistic."""
    cdef Py_ssize_t i
    cdef int k
    cdef double wmax = 0.0, total = 0.0, sq = 0.0, s, q, d, acc
    cdef double* pw = work
    cdef double* az = work + n
    if weighted:
        for i in range(n):
            if not (isfinite(raw[i]) and raw[i] >= 0.0):
                return 1
            if raw[i] > wmax:
                wmax = raw[i]
        if not (wmax > 0.0):
            return 2
        for i in range(n):
            w[i] = raw[i] / wmax
        total = gibbus_mix_sum_advance(w, NULL, n, 0)
        if not (total > 0.0 and isfinite(total)):
            return 2
        for i in range(n):
            w[i] = w[i] / total
        gibbus_mix_sum_square(w, n, &s, &sq)
        if not (sq > 0.0 and isfinite(sq)):
            return 2
        total_out[0] = wmax * total
        n_eff_out[0] = 1.0 / sq
    else:
        for i in range(n):
            w[i] = 1.0 / n
        total_out[0] = <double>n
        n_eff_out[0] = <double>n

    # Power moments E[z^k].
    for i in range(n):
        pw[i] = w[i]
    moments[0] = 1.0
    gibbus_mix_sum_advance(pw, z, n, 1)
    for k in range(1, order + 1):
        s = gibbus_mix_sum_advance(pw, z, n, 1)
        if not isfinite(s):
            return 3
        moments[k] = s

    # Participation of |z|^k: (sum c)^2 / sum c^2, capped at n.
    for i in range(n):
        pw[i] = w[i]
        az[i] = fabs(z[i])
    for k in range(order + 1):
        gibbus_mix_sum_square(pw, n, &s, &q)
        if q > 0.0 and isfinite(s) and isfinite(q):
            d = (s * s) / q
            participation[k] = d if d < <double>n else <double>n
        else:
            participation[k] = 0.0
        if k < order:
            for i in range(n):
                pw[i] *= az[i]

    boundary[0] = NAN
    boundary[1] = NAN
    if has_lower:
        if not isfinite(L):
            return 4
        acc = 0.0
        for i in range(n):
            if w[i] > 0.0:
                d = z[i] - L
                if not (d > 0.0):
                    return 4
                acc += w[i] * (-log(d))
        if not isfinite(acc):
            return 6
        boundary[0] = acc
    if has_upper:
        if not isfinite(U):
            return 5
        acc = 0.0
        for i in range(n):
            if w[i] > 0.0:
                d = U - z[i]
                if not (d > 0.0):
                    return 5
                acc += w[i] * (-log(d))
        if not isfinite(acc):
            return 6
        boundary[1] = acc
    return 0


def empirical_point_stats(const double[::1] z, weights, int max_order, double lower,
                          double upper, bint has_lower, bint has_upper):
    """Normalized weights and point statistics in one compiled pass.

    Parameters
    ----------
    z : ndarray, shape (n,)
        Finite canonical points inside the support (checked by the caller).
    weights : ndarray, shape (n,) or None
        Raw nonnegative weights; ``None`` means equal weights.
    max_order : int
        Highest power moment.
    lower, upper : float
        Canonical support.
    has_lower, has_upper : bool
        Whether the boundary-log statistics are part of the basis.

    Returns
    -------
    status : int
        0 ok; 1 invalid weights; 2 no positive total; 3 non-finite moment;
        4/5 a positive-weight point on or outside the lower/upper boundary;
        6 non-finite boundary statistic.
    moments, participation, boundary_log, total_weight, effective_n
        As in ``_EmpiricalStats``.
    """
    cdef Py_ssize_t n = z.shape[0]
    cdef const double[::1] raw
    cdef bint weighted = weights is not None
    if n < 1 or max_order < 0:
        raise ValueError("invalid point statistics request")
    if weighted:
        raw = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        if raw.shape[0] != n:
            raise ValueError("weights must have one entry per point observation")
    else:
        raw = z
    cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.empty(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] moments = np.empty(max_order + 1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] participation = np.empty(max_order + 1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] boundary = np.empty(2)
    cdef double total = 0.0, n_eff = 0.0
    cdef int status
    cdef const double* z_ptr = &z[0]
    cdef const double* raw_ptr = &raw[0]
    cdef double* w_ptr = &w[0]
    cdef double* moments_ptr = &moments[0]
    cdef double* participation_ptr = &participation[0]
    cdef double* boundary_ptr = &boundary[0]
    cdef double* work = <double*> malloc(2 * n * sizeof(double))
    if work == NULL:
        raise MemoryError("point statistics workspace")
    with nogil:
        status = _point_stats(
            z_ptr, raw_ptr, weighted, n, max_order, lower, upper,
            has_lower, has_upper, w_ptr, work, moments_ptr,
            participation_ptr, boundary_ptr, &total, &n_eff,
        )
    free(work)
    return status, moments, participation, boundary, total, n_eff
