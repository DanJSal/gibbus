# cython: language_level=3
"""Compiled row reductions for natural-coordinate mixture and point fits.

* ``empirical_point_stats``: weight normalization, power moments,
  participation effective sizes and boundary-log statistics of point data in
  one pass per statistic (the arithmetic of
  ``_observations.empirical._build_empirical_stats``).
* ``mixture_posterior``: stable per-row log-sum-exp over components, the
  weighted log likelihood and the responsibilities.
* ``joint_information``: gradient and missing information of the joint
  mixture likelihood from responsibilities, centered complete-data scores
  and within-row covariances.

Every loop runs without the GIL over contiguous rows; the long loops (over
observations) are the inner ones, written as restrict-qualified reductions
with ``omp simd`` hints (vectorized under ``-fopenmp-simd``).
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, fabs, INFINITY, isfinite
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

    /* a += alpha * x x^T on the upper triangle of an n x n row-major matrix */
    static inline void gibbus_mix_syr(double alpha, const double *GIBBUS_MIX_RESTRICT x,
                                      double *GIBBUS_MIX_RESTRICT a, ptrdiff_t n)
    {
        for (ptrdiff_t r = 0; r < n; ++r) {
            const double xr = alpha * x[r];
            double *row = a + r * n;
            GIBBUS_MIX_SIMD
            for (ptrdiff_t c = r; c < n; ++c) row[c] += xr * x[c];
        }
    }
    """
    double gibbus_mix_sum_advance(double* x, const double* z, Py_ssize_t n, int advance)
    void gibbus_mix_sum_square(const double* x, Py_ssize_t n, double* out_s, double* out_q)
    void gibbus_mix_syr(double alpha, const double* x, double* a, Py_ssize_t n)


# ---------------------------------------------------------------------------
# Empirical point statistics
# ---------------------------------------------------------------------------

cdef int _point_stats(const double* z, const double* raw, bint weighted, Py_ssize_t n,
                      int order, double L, double U, bint has_lower, bint has_upper,
                      double* w, double* work, double* moments, double* participation,
                      double* boundary, double* total_out, double* n_eff_out) noexcept nogil:
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

    boundary[0] = 0.0 / 0.0
    boundary[1] = 0.0 / 0.0
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
    cdef double* work = <double*> malloc(2 * n * sizeof(double))
    if work == NULL:
        raise MemoryError("point statistics workspace")
    with nogil:
        status = _point_stats(&z[0], &raw[0], weighted, n, max_order, lower, upper,
                              has_lower, has_upper, &w[0], work, &moments[0],
                              &participation[0], &boundary[0], &total, &n_eff)
    free(work)
    return status, moments, participation, boundary, total, n_eff


# ---------------------------------------------------------------------------
# Mixture posterior
# ---------------------------------------------------------------------------

def mixture_posterior(const double[:, ::1] log_values, const double[::1] log_weights,
                      const double[::1] observation_weights):
    """Return ``(status, log_likelihood, responsibilities, row_log)``.

    ``row_log[i] = log sum_k exp(log_values[i, k] + log_weights[k])``,
    ``log_likelihood = sum_i observation_weights[i] row_log[i]`` and
    ``responsibilities[i, k] = exp(log_values[i, k] + log_weights[k] -
    row_log[i])``.  ``status`` is 1 when some row has zero likelihood (or a
    non-finite value); the outputs are then unspecified.
    """
    cdef Py_ssize_t R = log_values.shape[0], K = log_values.shape[1]
    if log_weights.shape[0] != K or observation_weights.shape[0] != R:
        raise ValueError("mixture posterior shapes do not match")
    cdef cnp.ndarray[cnp.float64_t, ndim=2] resp = np.empty((R, K))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] row_log = np.empty(R)
    cdef double* rp = &resp[0, 0] if R > 0 and K > 0 else NULL
    cdef double total = 0.0, m, s, v
    cdef Py_ssize_t i, k
    cdef int status = 0
    if R == 0 or K == 0:
        return (1 if K == 0 and R > 0 else 0), 0.0, resp, row_log
    with nogil:
        for i in range(R):
            m = -INFINITY
            for k in range(K):
                v = log_values[i, k] + log_weights[k]
                rp[i * K + k] = v
                if v > m:
                    m = v
            if not isfinite(m):
                status = 1
                break
            s = 0.0
            for k in range(K):
                v = exp(rp[i * K + k] - m)
                rp[i * K + k] = v
                s += v
            for k in range(K):
                rp[i * K + k] /= s
            v = m + log(s)
            row_log[i] = v
            total += observation_weights[i] * v
        if status == 0 and not isfinite(total):
            status = 1
    return status, total, resp, row_log


# ---------------------------------------------------------------------------
# Joint mixture information
# ---------------------------------------------------------------------------

def joint_information(const double[:, ::1] responsibilities, const double[:, ::1] centered,
                      within, const Py_ssize_t[::1] offsets, const double[::1] pi,
                      const double[::1] observation_weights):
    """Gradient and missing information of the joint mixture NLL.

    The joint variable is ``(theta_1, ..., theta_K, eta_1, ..., eta_{K-1})``.
    Row ``i``'s complete-data score of ``-log(pi_k f_k)`` for label ``k``,
    ``s_ik``, has block ``k`` equal to ``centered[i, block k]`` (the negated
    potential-partial mean minus its model mean), logit part ``pi[:-1] -
    e_k``, and zeros elsewhere.  With ``m_i = sum_k r_ik s_ik``:

        gradient = sum_i w_i m_i
        missing  = sum_i w_i sum_k r_ik (s_ik - m_i)(s_ik - m_i)^T
                   + sum_k sum_i w_i r_ik W_ik          (block k)

    Parameters
    ----------
    responsibilities : ndarray, shape (R, K)
    centered : ndarray, shape (R, offsets[K])
        Conditional partial means minus model means, blocks side by side.
    within : sequence of (ndarray of shape (R, n_k, n_k) or None)
        Within-row conditional covariances per component (``None`` for
        point data).
    offsets : ndarray, shape (K + 1,)
        Block offsets of the component parameters.
    pi : ndarray, shape (K,)
        Mixture weights.
    observation_weights : ndarray, shape (R,)

    Returns
    -------
    gradient : ndarray, shape (N,)
    missing : ndarray, shape (N, N)
    """
    cdef Py_ssize_t R = responsibilities.shape[0], K = responsibilities.shape[1]
    cdef Py_ssize_t P = offsets[K]
    cdef Py_ssize_t N = P + K - 1
    if centered.shape[0] != R or centered.shape[1] != P or pi.shape[0] != K:
        raise ValueError("joint information shapes do not match")
    if observation_weights.shape[0] != R or len(within) != K:
        raise ValueError("joint information shapes do not match")
    cdef cnp.ndarray[cnp.float64_t, ndim=1] gradient = np.zeros(N)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] missing = np.zeros((N, N))
    cdef double* g = &gradient[0]
    cdef double* M = &missing[0, 0]
    cdef double* work = <double*> malloc((2 * N + 1) * sizeof(double))
    if work == NULL:
        raise MemoryError("joint information workspace")
    cdef double* mean = work
    cdef double* d = work + N
    cdef Py_ssize_t i, k, j, a, b, lo, hi, nk
    cdef double r, wi, alpha
    cdef const double* cr
    with nogil:
        for i in range(R):
            wi = observation_weights[i]
            cr = &centered[i, 0]
            # m_i
            for k in range(K):
                r = responsibilities[i, k]
                for j in range(offsets[k], offsets[k + 1]):
                    mean[j] = r * cr[j]
            for k in range(K - 1):
                mean[P + k] = pi[k] - responsibilities[i, k]
            for j in range(N):
                g[j] += wi * mean[j]
            if wi == 0.0:
                continue
            # sum_k r_ik (s_ik - m_i)(s_ik - m_i)^T
            for k in range(K):
                r = responsibilities[i, k]
                if r == 0.0:
                    continue
                for j in range(N):
                    d[j] = -mean[j]
                for j in range(offsets[k], offsets[k + 1]):
                    d[j] += cr[j]
                for j in range(K - 1):
                    d[P + j] += pi[j]
                if k < K - 1:
                    d[P + k] -= 1.0
                gibbus_mix_syr(wi * r, d, M, N)
    free(work)
    # Within-row covariances (interval components).
    cdef const double[:, :, ::1] cov
    for k in range(K):
        if within[k] is None:
            continue
        cov = within[k]
        lo = offsets[k]
        nk = offsets[k + 1] - lo
        if cov.shape[0] != R or cov.shape[1] != nk or cov.shape[2] != nk:
            raise ValueError("within-row covariance shape does not match its block")
        with nogil:
            for i in range(R):
                alpha = observation_weights[i] * responsibilities[i, k]
                if alpha == 0.0:
                    continue
                for a in range(nk):
                    for b in range(a, nk):
                        M[(lo + a) * N + lo + b] += alpha * cov[i, a, b]
    # Mirror the upper triangle.
    with nogil:
        for a in range(N):
            for b in range(a + 1, N):
                M[b * N + a] = M[a * N + b]
    return gradient, missing
