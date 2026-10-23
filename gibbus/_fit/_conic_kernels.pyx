# cython: language_level=3
"""Compiled interior-point solver and face Newton loop of the conic subproblems.

G1 owner extended: ``_fit/_conic_kernels.pyx``.  G1's natural-coordinate point
and interval objectives and their Newton loops are replaced by the G3
objectives of ``_basis_newton_kernels`` (G3-039, G3-044); kept are the
interior-point solver, the saddle-free metric and the generic face Newton loop
those objectives drive through ``FaceObjective``.

The test suite keeps a pure-Python translation (``tests/conic_reference.py``)
and compares the two.  One call solves

    minimize  g.(theta - theta0) + 1/2 (theta - theta0)^T H (theta - theta0)
    subject to  B theta = sum_b L_b(Q_b),  Q_b PSD,  L_b(Q)_i = <A_{b,i}, Q>

by the same infeasible-start primal-dual interior point: HKM direction,
Mehrotra predictor-corrector, one step of iterative refinement, minimum-norm
Gram correction of primal drift outside ``range(B)``, a weak-duality
certificate refused unless the reconstructed endpoint satisfies the
description to roundoff, and best-iterate selection by ``(gap, residual)``.
Everything after argument unpacking runs without the GIL.

The matrices are tiny (Gram blocks of at most about 8 x 8, KKT systems of a
few dozen rows), so the dense kernels are written out here instead of being
routed through LAPACK: Cholesky, partial-pivoting LU, cyclic Jacobi for
symmetric eigenproblems and one-sided (Hestenes) Jacobi for the SVD of
``B``.  Jacobi methods have high relative accuracy, which the certificate
(smallest eigenvalues near zero) needs.  Inner loops run over contiguous rows
through restrict-qualified helpers so compilers can vectorize them;
reductions carry ``omp simd`` hints, honored under ``-fopenmp-simd`` (no
OpenMP runtime is linked).

Layout: ``B`` is ``(r, n)`` row-major.  Block ``b`` of size ``k_b`` stores
its ``r`` matrices ``A_{b,i}`` contiguously as ``(r, k_b, k_b)`` starting at
``aoff[b]`` of one packed array; Gram blocks and dual slacks are packed as
consecutive ``k_b * k_b`` matrices starting at ``qoff[b]``.
"""

import numpy as np
cimport numpy as cnp

from .._defaults import (
    CONIC_QP_GAP_TOL,
    CONIC_QP_MAX_ITER,
    CONIC_QP_STEP_FRACTION,
)
from libc.math cimport sqrt, fabs, exp, log, copysign, isinf, NAN, INFINITY
from libc.float cimport DBL_MIN
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

cnp.import_array()


cdef double _QP_GAP_TOL = CONIC_QP_GAP_TOL
cdef int _QP_MAX_ITER = CONIC_QP_MAX_ITER
# A repeated step hint in one line search must shorten the step by at least
# this factor: a moving curvature minimum otherwise shortened it by
# 1e-4 per objective evaluation, doubling the evaluations (G3-060).
cdef double _HINT_SHRINK = 0.9
cdef double _QP_STEP_FRACTION = CONIC_QP_STEP_FRACTION

cdef extern from * nogil:
    """
    #include <stddef.h>
    #if defined(_MSC_VER)
      #define GIBBUS_CONIC_RESTRICT __restrict
      #define GIBBUS_CONIC_SIMD
      #define GIBBUS_CONIC_SIMD_SUM
    #else
      #define GIBBUS_CONIC_RESTRICT __restrict__
      #define GIBBUS_CONIC_SIMD _Pragma("omp simd")
      #define GIBBUS_CONIC_SIMD_SUM _Pragma("omp simd reduction(+:s)")
    #endif

    static inline double gibbus_conic_dot(const double *GIBBUS_CONIC_RESTRICT a,
                                          const double *GIBBUS_CONIC_RESTRICT b,
                                          ptrdiff_t n)
    {
        double s = 0.0;
        GIBBUS_CONIC_SIMD_SUM
        for (ptrdiff_t i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }

    static inline void gibbus_conic_axpy(double alpha,
                                         const double *GIBBUS_CONIC_RESTRICT x,
                                         double *GIBBUS_CONIC_RESTRICT y,
                                         ptrdiff_t n)
    {
        GIBBUS_CONIC_SIMD
        for (ptrdiff_t i = 0; i < n; ++i) y[i] += alpha * x[i];
    }

    static inline void gibbus_conic_scale(double alpha, double *GIBBUS_CONIC_RESTRICT x,
                                          ptrdiff_t n)
    {
        GIBBUS_CONIC_SIMD
        for (ptrdiff_t i = 0; i < n; ++i) x[i] *= alpha;
    }

    static inline double gibbus_conic_absmax(const double *GIBBUS_CONIC_RESTRICT x,
                                             ptrdiff_t n)
    {
        double m = 0.0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            double v = x[i] < 0.0 ? -x[i] : x[i];
            m = v > m ? v : m;
        }
        return m;
    }
    """
    double c_dot "gibbus_conic_dot"(const double* a, const double* b, Py_ssize_t n)
    void c_axpy "gibbus_conic_axpy"(
        double alpha, const double* x, double* y, Py_ssize_t n
    )
    void c_scale "gibbus_conic_scale"(double alpha, double* x, Py_ssize_t n)
    double c_absmax "gibbus_conic_absmax"(const double* x, Py_ssize_t n)


cdef double _EPS = 2.220446049250313e-16


# ---------------------------------------------------------------------------
# Dense kernels (row-major, caller-owned storage)
# ---------------------------------------------------------------------------

cdef inline void _symmetrize(double* a, int k) noexcept nogil:
    cdef int i, j
    cdef double v
    for i in range(k):
        for j in range(i + 1, k):
            v = 0.5 * (a[i * k + j] + a[j * k + i])
            a[i * k + j] = v
            a[j * k + i] = v


cdef inline void _matmul(
    const double* a, const double* b, double* out, int k
) noexcept nogil:
    """``out = a @ b`` for k x k row-major matrices (out must not alias)."""
    cdef int i, p
    memset(out, 0, k * k * sizeof(double))
    for i in range(k):
        for p in range(k):
            c_axpy(a[i * k + p], b + p * k, out + i * k, k)


cdef inline int _cholesky(double* a, int k) noexcept nogil:
    """In-place lower Cholesky of a symmetric matrix; 1 unless positive definite."""
    cdef int i, j
    cdef double s, d
    for j in range(k):
        s = a[j * k + j] - c_dot(a + j * k, a + j * k, j)
        if not (s > 0.0):
            return 1
        d = sqrt(s)
        a[j * k + j] = d
        for i in range(j + 1, k):
            a[i * k + j] = (a[i * k + j] - c_dot(a + i * k, a + j * k, j)) / d
    for i in range(k):
        for j in range(i + 1, k):
            a[i * k + j] = 0.0
    return 0


cdef inline void _lower_solve_rows(
    const double* l, const double* d, double* y, int k
) noexcept nogil:
    """``y = L^-1 d`` for a lower-triangular ``L`` and a k x k right side."""
    cdef int i, p
    for i in range(k):
        memcpy(y + i * k, d + i * k, k * sizeof(double))
        for p in range(i):
            c_axpy(-l[i * k + p], y + p * k, y + i * k, k)
        c_scale(1.0 / l[i * k + i], y + i * k, k)


cdef inline void _transpose(const double* a, double* out, int k) noexcept nogil:
    cdef int i, j
    for i in range(k):
        for j in range(k):
            out[j * k + i] = a[i * k + j]


cdef int _jacobi(
    double* a, int n, double* d, double* v, double* b, double* z, bint vectors
) noexcept nogil:
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix (upper triangle used).

    ``a`` is destroyed.  ``d`` receives the eigenvalues; with ``vectors`` the
    columns of ``v`` (row-major ``v[i * n + j]``) are the eigenvectors.
    Returns 0 on convergence.
    """
    cdef int ip, iq, j, sweep
    cdef double sm, tresh, g, h, t, theta, c, s, tau, x, y
    if vectors:
        memset(v, 0, n * n * sizeof(double))
        for ip in range(n):
            v[ip * n + ip] = 1.0
    for ip in range(n):
        b[ip] = a[ip * n + ip]
        d[ip] = b[ip]
        z[ip] = 0.0
    for sweep in range(1, 61):
        sm = 0.0
        for ip in range(n - 1):
            for iq in range(ip + 1, n):
                sm += fabs(a[ip * n + iq])
        if sm == 0.0:
            return 0
        tresh = 0.2 * sm / (n * n) if sweep < 4 else 0.0
        for ip in range(n - 1):
            for iq in range(ip + 1, n):
                g = 100.0 * fabs(a[ip * n + iq])
                if (
                    sweep > 4
                    and fabs(d[ip]) + g == fabs(d[ip])
                    and fabs(d[iq]) + g == fabs(d[iq])
                ):
                    a[ip * n + iq] = 0.0
                elif fabs(a[ip * n + iq]) > tresh:
                    h = d[iq] - d[ip]
                    if fabs(h) + g == fabs(h):
                        t = a[ip * n + iq] / h
                    else:
                        theta = 0.5 * h / a[ip * n + iq]
                        t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta))
                        if theta < 0.0:
                            t = -t
                    c = 1.0 / sqrt(1.0 + t * t)
                    s = t * c
                    tau = s / (1.0 + c)
                    h = t * a[ip * n + iq]
                    z[ip] -= h
                    z[iq] += h
                    d[ip] -= h
                    d[iq] += h
                    a[ip * n + iq] = 0.0
                    for j in range(ip):
                        x = a[j * n + ip]
                        y = a[j * n + iq]
                        a[j * n + ip] = x - s * (y + x * tau)
                        a[j * n + iq] = y + s * (x - y * tau)
                    for j in range(ip + 1, iq):
                        x = a[ip * n + j]
                        y = a[j * n + iq]
                        a[ip * n + j] = x - s * (y + x * tau)
                        a[j * n + iq] = y + s * (x - y * tau)
                    for j in range(iq + 1, n):
                        x = a[ip * n + j]
                        y = a[iq * n + j]
                        a[ip * n + j] = x - s * (y + x * tau)
                        a[iq * n + j] = y + s * (x - y * tau)
                    if vectors:
                        for j in range(n):
                            x = v[j * n + ip]
                            y = v[j * n + iq]
                            v[j * n + ip] = x - s * (y + x * tau)
                            v[j * n + iq] = y + s * (x - y * tau)
        for ip in range(n):
            b[ip] += z[ip]
            d[ip] = b[ip]
            z[ip] = 0.0
    return 1


cdef double _min_eig(const double* m, int k, double* work) noexcept nogil:
    """Smallest eigenvalue of a symmetric k x k matrix (``work``: k*k + 3k)."""
    cdef double* a = work
    cdef double* d = work + k * k
    cdef int i
    cdef double lo
    memcpy(a, m, k * k * sizeof(double))
    _jacobi(a, k, d, NULL, d + k, d + 2 * k, False)
    lo = d[0]
    for i in range(1, k):
        if d[i] < lo:
            lo = d[i]
    return lo


cdef int _lu_factor(double* a, int n, int* piv) noexcept nogil:
    """In-place LU with partial pivoting; 1 on an exactly zero pivot."""
    cdef int i, j, p, best
    cdef double m, v, f
    for j in range(n):
        best = j
        m = fabs(a[j * n + j])
        for i in range(j + 1, n):
            v = fabs(a[i * n + j])
            if v > m:
                m = v
                best = i
        piv[j] = best
        if m == 0.0 or m != m:
            return 1
        if best != j:
            for p in range(n):
                v = a[j * n + p]
                a[j * n + p] = a[best * n + p]
                a[best * n + p] = v
        for i in range(j + 1, n):
            f = a[i * n + j] / a[j * n + j]
            a[i * n + j] = f
            if f != 0.0:
                c_axpy(-f, a + j * n + j + 1, a + i * n + j + 1, n - j - 1)
    return 0


cdef void _lu_solve(const double* lu, int n, const int* piv, double* x) noexcept nogil:
    """Solve in place with a factorization from ``_lu_factor``."""
    cdef int i, j
    cdef double v
    for i in range(n):
        if piv[i] != i:
            v = x[i]
            x[i] = x[piv[i]]
            x[piv[i]] = v
    for i in range(n):
        x[i] -= c_dot(lu + i * n, x, i)
    for i in range(n - 1, -1, -1):
        v = x[i]
        for j in range(i + 1, n):
            v -= lu[i * n + j] * x[j]
        x[i] = v / lu[i * n + i]


cdef void _svd_rows(
    const double* bmat, int r, int n, double* x, double* u, double* sigma
) noexcept nogil:
    """One-sided Jacobi SVD of ``B`` (r x n) through its rows.

    On return ``x`` (r rows of length n) holds ``sigma_j * v_j`` and ``u``
    (r x r, column j at ``u + j * r``) the left singular vectors, so
    ``B = sum_j sigma_j u_j v_j^T``.
    """
    cdef int p, q, i, _sweep
    cdef bint rotated
    cdef double alpha, beta, gamma, zeta, t, c, s, xp, xq
    memcpy(x, bmat, r * n * sizeof(double))
    memset(u, 0, r * r * sizeof(double))
    for p in range(r):
        u[p * r + p] = 1.0
    for _sweep in range(80):
        rotated = False
        for p in range(r - 1):
            for q in range(p + 1, r):
                alpha = c_dot(x + p * n, x + p * n, n)
                beta = c_dot(x + q * n, x + q * n, n)
                gamma = c_dot(x + p * n, x + q * n, n)
                if gamma == 0.0 or fabs(gamma) <= _EPS * sqrt(alpha * beta):
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                t = copysign(1.0, zeta) / (fabs(zeta) + sqrt(1.0 + zeta * zeta))
                c = 1.0 / sqrt(1.0 + t * t)
                s = c * t
                for i in range(n):
                    xp = x[p * n + i]
                    xq = x[q * n + i]
                    x[p * n + i] = c * xp - s * xq
                    x[q * n + i] = s * xp + c * xq
                for i in range(r):
                    xp = u[p * r + i]
                    xq = u[q * r + i]
                    u[p * r + i] = c * xp - s * xq
                    u[q * r + i] = s * xp + c * xq
                rotated = True
        if not rotated:
            break
    for p in range(r):
        sigma[p] = sqrt(c_dot(x + p * n, x + p * n, n))


# ---------------------------------------------------------------------------
# Representation with the per-solve precomputations
# ---------------------------------------------------------------------------

cdef struct Rep:
    int n
    int r
    int nb
    int kmax
    const int* k
    const Py_ssize_t* aoff
    const Py_ssize_t* qoff
    Py_ssize_t qtot
    const double* b          # r x n
    const double* a          # packed row matrices
    const double* ref        # r
    # derived
    double* pinv             # n x r, B^+ over the resolved range (1e-12)
    double* complement       # r x r projector onto range(B)^perp
    bint has_complement
    double* gg_vec           # r x r, eigenvectors of the Gram of the rows (transposed)
    double* gg_val           # r
    double gg_cut
    int* exact_row
    int* exact_col
    int n_exact
    double margin            # smallest eigenvalue of the reference slacks
    # Rows with a nonzero operator in each block (CSR): block bb reads rows
    # blk_rows[blk_start[bb]:blk_start[bb + 1]].  Every block loop skips the
    # exact zeros (G3 descriptions: one row per 1 x 1 block, G3-040).
    int* blk_start
    int* blk_rows


cdef void _gram_map(const Rep* rep, const double* q, double* out) noexcept nogil:
    """``out_i = sum_b <A_{b,i}, Q_b>``."""
    cdef int i, bb, kk, t
    cdef const double* ab
    memset(out, 0, rep.r * sizeof(double))
    for bb in range(rep.nb):
        kk = rep.k[bb]
        ab = rep.a + rep.aoff[bb]
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            out[i] += c_dot(ab + i * kk * kk, q + rep.qoff[bb], kk * kk)


cdef void _slacks(const Rep* rep, const double* y, double* s) noexcept nogil:
    """``S_b = -sum_i y_i A_{b,i}`` for every block."""
    cdef int i, bb, kk, t
    cdef const double* ab
    memset(s, 0, rep.qtot * sizeof(double))
    for bb in range(rep.nb):
        kk = rep.k[bb]
        ab = rep.a + rep.aoff[bb]
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            if y[i] != 0.0:
                c_axpy(-y[i], ab + i * kk * kk, s + rep.qoff[bb], kk * kk)


cdef double _residual(
    const Rep* rep, const double* theta, const double* q, double* work
) noexcept nogil:
    """``max |B theta - sum_b L_b(Q_b)|`` (``work``: 2 r)."""
    cdef int i
    cdef double m = 0.0, v
    _gram_map(rep, q, work)
    for i in range(rep.r):
        v = fabs(c_dot(rep.b + i * rep.n, theta, rep.n) - work[i])
        if v > m:
            m = v
    return m


cdef void _reconstruct(
    const Rep* rep, double* theta, const double* q, double* work
) noexcept nogil:
    """Move ``theta`` onto ``B theta = L(Q)``: minimum-norm, then exact coordinates."""
    cdef int i, j
    cdef double* target = work
    cdef double* diff = work + rep.r
    _gram_map(rep, q, target)
    for i in range(rep.r):
        diff[i] = target[i] - c_dot(rep.b + i * rep.n, theta, rep.n)
    for j in range(rep.n):
        theta[j] += c_dot(rep.pinv + j * rep.r, diff, rep.r)
    for i in range(rep.n_exact):
        theta[rep.exact_col[i]] = target[rep.exact_row[i]] / rep.b[
            rep.exact_row[i] * rep.n + rep.exact_col[i]
        ]


cdef int _prepare(Rep* rep, bint need_complement, double* work) noexcept nogil:
    """Derived quantities of a description (``work``: see ``_prepare_size``)."""
    cdef int n = rep.n, r = rep.r
    cdef int i, j, bb, kk, count, col, t, u_
    cdef bint separable
    cdef double smax, thr, v
    cdef double* x = work
    cdef double* u = x + r * n
    cdef double* sigma = u + r * r
    cdef double* g = sigma + r
    cdef double* scratch = g + r * r
    cdef const double* ab
    # Row lists of the blocks; a description is row-separable when every row
    # reads exactly one block (G3: pointwise curvature and structural rows).
    count = 0
    for bb in range(rep.nb):
        kk = rep.k[bb]
        ab = rep.a + rep.aoff[bb]
        rep.blk_start[bb] = count
        for i in range(r):
            for j in range(kk * kk):
                if ab[i * kk * kk + j] != 0.0:
                    rep.blk_rows[count] = i
                    count += 1
                    break
    rep.blk_start[rep.nb] = count
    separable = count == r
    if separable:
        memset(scratch, 0, r * sizeof(double))
        for t in range(count):
            scratch[rep.blk_rows[t]] += 1.0
        for i in range(r):
            if scratch[i] != 1.0:
                separable = False
    # SVD of B for B^+ and the range complement by one-sided Jacobi: over the
    # rows (G1), or over the n columns for a row-separable description with
    # more rows than columns (G3's cut rows: O(n^2 r) per sweep instead of
    # O(r^2 (n + r)), G3-040).  G1's polynomial descriptions keep the row
    # sweep, so their arithmetic is unchanged.
    rep.has_complement = False
    if not (separable and r > n):
        _svd_rows(rep.b, r, n, x, u, sigma)
        smax = 0.0
        for j in range(r):
            if sigma[j] > smax:
                smax = sigma[j]
        # One rank decision for B^+ and the range complement: a direction
        # outside the resolved range is not inverted (a near-null singular
        # value of nearly parallel rows amplified the rounding of L(Q) into a
        # large endpoint move, G3-050).
        thr = 1e-12 * (smax if smax > 1.0 else 1.0)
        memset(rep.pinv, 0, n * r * sizeof(double))
        for j in range(r):
            if sigma[j] > thr:
                # pinv += (x_j / sigma_j) u_j^T / sigma_j
                for i in range(n):
                    v = x[j * n + i] / (sigma[j] * sigma[j])
                    c_axpy(v, u + j * r, rep.pinv + i * r, r)
        if need_complement:
            memset(rep.complement, 0, r * r * sizeof(double))
            for j in range(r):
                if not (sigma[j] > thr):
                    rep.has_complement = True
                    for i in range(r):
                        c_axpy(u[j * r + i], u + j * r, rep.complement + i * r, r)
    else:
        # B^T = sum_j sigma_j v_j u_j^T: x row j = sigma_j u_j (length r), u
        # column j (at u + j * n) = v_j.  B^T is staged in g (r x r >= n x r).
        for i in range(r):
            for j in range(n):
                g[j * r + i] = rep.b[i * n + j]
        _svd_rows(g, n, r, x, u, sigma)
        smax = 0.0
        for j in range(n):
            if sigma[j] > smax:
                smax = sigma[j]
        thr = 1e-12 * (smax if smax > 1.0 else 1.0)  # as above
        memset(rep.pinv, 0, n * r * sizeof(double))
        for j in range(n):
            if sigma[j] > thr:
                # pinv += v_j (sigma_j u_j)^T / sigma_j^2
                for i in range(n):
                    v = u[j * n + i] / (sigma[j] * sigma[j])
                    c_axpy(v, x + j * r, rep.pinv + i * r, r)
        if need_complement:
            # Complement of the range, I - B B_thr^+ with B_thr^+ over
            # sigma_j > thr (the r - n missing directions are null, so it is
            # never empty).  Formed with B itself, the complement annihilates
            # the computed range to roundoff; I - sum u_j u_j^T from the
            # normalized x_j / sigma_j carries their eps / sigma_j direction
            # error and stalled the primal residual near 1e-12 (G3-040).
            rep.has_complement = True
            memset(g, 0, n * r * sizeof(double))
            for j in range(n):
                if sigma[j] > thr:
                    for i in range(n):
                        v = u[j * n + i] / (sigma[j] * sigma[j])
                        c_axpy(v, x + j * r, g + i * r, r)
            memset(rep.complement, 0, r * r * sizeof(double))
            for i in range(r):
                rep.complement[i * r + i] = 1.0
                for j in range(n):
                    if rep.b[i * n + j] != 0.0:
                        c_axpy(-rep.b[i * n + j], g + j * r, rep.complement + i * r, r)
    if need_complement:
        # Gram of the rows, G_ij = sum_b <A_{b,i}, A_{b,j}>, for lstsq(G, .).
        memset(g, 0, r * r * sizeof(double))
        for bb in range(rep.nb):
            kk = rep.k[bb]
            ab = rep.a + rep.aoff[bb]
            for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
                i = rep.blk_rows[t]
                for u_ in range(t, rep.blk_start[bb + 1]):
                    j = rep.blk_rows[u_]
                    v = c_dot(ab + i * kk * kk, ab + j * kk * kk, kk * kk)
                    g[i * r + j] += v
                    if j != i:
                        g[j * r + i] += v
        if (
            _jacobi(
                g, r, rep.gg_val, scratch, scratch + r * r, scratch + r * r + r, True
            )
            != 0
        ):
            return 1
        # Store eigenvectors transposed: row j = vector j.
        _transpose(scratch, rep.gg_vec, r)
        smax = 0.0
        for j in range(r):
            if fabs(rep.gg_val[j]) > smax:
                smax = fabs(rep.gg_val[j])
        rep.gg_cut = _EPS * r * smax
    # Exact coordinates: rows reading one column that no other row reads.
    rep.n_exact = 0
    for i in range(r):
        count = 0
        col = -1
        for j in range(n):
            if rep.b[i * n + j] != 0.0:
                count += 1
                col = j
        if count != 1:
            continue
        count = 0
        for j in range(r):
            if rep.b[j * n + col] != 0.0:
                count += 1
        if count == 1:
            rep.exact_row[rep.n_exact] = i
            rep.exact_col[rep.n_exact] = col
            rep.n_exact += 1
    # Reference margin.
    _slacks(rep, rep.ref, scratch)
    rep.margin = INFINITY
    for bb in range(rep.nb):
        kk = rep.k[bb]
        v = _min_eig(scratch + rep.qoff[bb], kk, scratch + rep.qtot)
        if v < rep.margin:
            rep.margin = v
    return 0


cdef Py_ssize_t _prepare_size(int n, int r, Py_ssize_t qtot, int kmax) noexcept nogil:
    """Scratch doubles needed by ``_prepare``."""
    cdef Py_ssize_t a = r * n + r * r + r + r * r
    cdef Py_ssize_t b = r * r + 3 * r
    cdef Py_ssize_t c = qtot + kmax * kmax + 3 * kmax
    if b < c:
        b = c
    return a + b + 8


cdef Py_ssize_t _rep_storage(int n, int r) noexcept nogil:
    """Doubles owned by a prepared ``Rep``: pinv, complement, eigen of the row Gram."""
    return n * r + r * r + r * r + r


# ---------------------------------------------------------------------------
# The interior point
# ---------------------------------------------------------------------------

cdef struct Hessian:
    int n
    const double* h        # symmetrized n x n
    double* vec            # n x n, row j = eigenvector j
    double* val            # n
    double cut


cdef int _hessian_eigen(Hessian* hs, double* work) noexcept nogil:
    """Eigen-decomposition of the model Hessian (``work``: 2 n^2 + 2 n)."""
    cdef int n = hs.n, j
    cdef double smax = 0.0
    memcpy(work, hs.h, n * n * sizeof(double))
    if (
        _jacobi(
            work, n, hs.val, work + n * n, work + 2 * n * n, work + 2 * n * n + n, True
        )
        != 0
    ):
        return 1
    _transpose(work + n * n, hs.vec, n)
    for j in range(n):
        if fabs(hs.val[j]) > smax:
            smax = fabs(hs.val[j])
    hs.cut = _EPS * n * smax
    return 0


cdef double _dual_value(
    const Rep* rep,
    const Hessian* hs,
    const double* g,
    const double* theta0,
    const double* y,
    double* work,
) noexcept nogil:
    """Lagrangian dual ``D(y)``; ``-inf`` when the stationary system is inconsistent.

    ``work``: 4 n.  The stationary point is the least-squares (pseudo-inverse)
    solution of ``H step = -(g + B^T y)`` with ``numpy.linalg.lstsq``'s cutoff.
    """
    cdef int n = rep.n, r = rep.r, i, j
    cdef double* w = work
    cdef double* step = work + n
    cdef double* hs_step = work + 2 * n
    cdef double* theta = work + 3 * n
    cdef double scale, coef, value, check, v
    for j in range(n):
        w[j] = -g[j]
    for i in range(r):
        if y[i] != 0.0:
            c_axpy(-y[i], rep.b + i * n, w, n)
    memset(step, 0, n * sizeof(double))
    for j in range(n):
        if fabs(hs.val[j]) > hs.cut:
            coef = c_dot(hs.vec + j * n, w, n) / hs.val[j]
            c_axpy(coef, hs.vec + j * n, step, n)
    scale = 1.0
    v = c_absmax(g, n)
    if v > scale:
        scale = v
    v = c_absmax(w, n)
    if v > scale:
        scale = v
    check = 0.0
    for i in range(n):
        hs_step[i] = c_dot(hs.h + i * n, step, n)
        v = fabs(hs_step[i] - w[i])
        if v > check:
            check = v
    if check > 1e-10 * scale:
        return -INFINITY
    for j in range(n):
        theta[j] = theta0[j] + step[j]
    value = c_dot(g, step, n) + 0.5 * c_dot(step, hs_step, n)
    for i in range(r):
        value += y[i] * c_dot(rep.b + i * n, theta, n)
    return value


cdef double _model(
    const Hessian* hs,
    const double* g,
    const double* theta0,
    const double* theta,
    double* work,
) noexcept nogil:
    """Newton model value ``g.d + 1/2 d^T H d`` (``work``: 2 n)."""
    cdef int n = hs.n, i
    for i in range(n):
        work[i] = theta[i] - theta0[i]
    for i in range(n):
        work[n + i] = c_dot(hs.h + i * n, work, n)
    return c_dot(g, work, n) + 0.5 * c_dot(work, work + n, n)


cdef double _certified_gap(
    const Rep* rep,
    const Hessian* hs,
    const double* g,
    const double* theta0,
    const double* theta,
    const double* q,
    const double* y,
    double* endpoint,
    double* model_value,
    double* work,
) noexcept nogil:
    """Reconstruct the endpoint and bound its suboptimality (``inf`` if uncertified).

    ``work``: 2 r + qtot + max(2 r, 4 n, kmax^2 + 3 kmax).
    """
    cdef int n = rep.n, r = rep.r, bb, kk, i, j
    cdef double* image = work
    cdef double* shifted = work + r
    cdef double* slack = shifted + r
    cdef double* scratch = slack + rep.qtot
    cdef double deficit = 0.0, v, lower, big, floor_, row_mag
    memcpy(endpoint, theta, n * sizeof(double))
    _reconstruct(rep, endpoint, q, scratch)
    model_value[0] = _model(hs, g, theta0, endpoint, scratch)
    # Weak duality bounds only feasible points: refuse an endpoint that
    # misses the description by more than roundoff -- 1e-13 of the image, or
    # the rounding of the row product itself when that is larger (a far
    # coefficient vector, |theta| ~ 1e4 on a near-unidentified span, cannot
    # meet 1e-13 in B theta whatever its blocks, G3-050).
    _gram_map(rep, q, image)
    big = c_absmax(image, r)
    floor_ = 1e-13 * (big if big > 1.0 else 1.0)
    for i in range(r):
        v = fabs(c_dot(rep.b + i * n, endpoint, n) - image[i])
        row_mag = 0.0
        for j in range(n):
            row_mag += fabs(rep.b[i * n + j] * endpoint[j])
        if v > floor_ and v > 8.0 * n * _EPS * row_mag:
            return INFINITY
    # Shift a boundary dual along the strictly feasible reference until every
    # slack is PSD; weak duality then holds exactly at the shifted multiplier.
    _slacks(rep, y, slack)
    for bb in range(rep.nb):
        kk = rep.k[bb]
        v = -_min_eig(slack + rep.qoff[bb], kk, scratch)
        if v > deficit:
            deficit = v
    memcpy(shifted, y, r * sizeof(double))
    if deficit > 0.0:
        c_axpy(2.0 * deficit / rep.margin, rep.ref, shifted, r)
        _slacks(rep, shifted, slack)
        for bb in range(rep.nb):
            kk = rep.k[bb]
            if _min_eig(slack + rep.qoff[bb], kk, scratch) < 0.0:
                return INFINITY
    lower = _dual_value(rep, hs, g, theta0, shifted, scratch)
    return model_value[0] - lower


cdef void _spd_inverse(
    const double* m, int k, double* out, double* work, bint* failed
) noexcept nogil:
    """``out = m^-1`` for a symmetric positive-definite ``m`` (``work``: 3 k^2)."""
    cdef int i
    memcpy(work, m, k * k * sizeof(double))
    if _cholesky(work, k) != 0:
        failed[0] = True
        return
    memset(work + k * k, 0, k * k * sizeof(double))
    for i in range(k):
        work[k * k + i * k + i] = 1.0
    _lower_solve_rows(work, work + k * k, work + 2 * k * k, k)   # X = L^-1
    _transpose(work + 2 * k * k, work + k * k, k)                  # X^T
    _matmul(work + k * k, work + 2 * k * k, out, k)                # X^T X


cdef void _schur(const Rep* rep, const double* q, const double* sinv, double* schur,
                 double* work) noexcept nogil:
    """``-M`` with ``M_ij = sum_b tr(A_i Q A_j S^-1)``, symmetrized (``work``: (2 r + 1) kmax^2)."""
    cdef int r = rep.r, i, j, bb, kk, k2, t, u_
    cdef const double* ab
    cdef double* xa
    cdef double* xb
    cdef double* tmp
    cdef double v
    memset(schur, 0, r * r * sizeof(double))
    for bb in range(rep.nb):
        kk = rep.k[bb]
        k2 = kk * kk
        ab = rep.a + rep.aoff[bb]
        xa = work
        xb = work + r * k2
        tmp = xb + r * k2
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            _matmul(ab + i * k2, q + rep.qoff[bb], xa + i * k2, kk)
            _matmul(ab + i * k2, sinv + rep.qoff[bb], tmp, kk)
            # tr(X_i Y_j) = <X_i, Y_j^T>: store Y transposed.
            _transpose(tmp, xb + i * k2, kk)
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            for u_ in range(t, rep.blk_start[bb + 1]):
                j = rep.blk_rows[u_]
                v = c_dot(xa + i * k2, xb + j * k2, k2)
                schur[i * r + j] -= v
                if j != i:
                    schur[j * r + i] -= c_dot(xa + j * k2, xb + i * k2, k2)
    for i in range(r):
        for j in range(i + 1, r):
            v = 0.5 * (schur[i * r + j] + schur[j * r + i])
            schur[i * r + j] = v
            schur[j * r + i] = v


cdef struct Kkt:
    int nn
    double* mat            # nn x nn, LU factors after factoring
    int* piv
    double* schur          # r x r, the -M block (kept for refinement)
    double* dual_res       # n
    double* primal_res     # r
    double* sinv           # packed S_b^-1


cdef int _kkt_build(
    const Rep* rep,
    const Hessian* hs,
    const double* g,
    const double* theta0,
    const double* theta,
    const double* q,
    const double* y,
    const double* s,
    Kkt* kkt,
    double* work,
) noexcept nogil:
    """Assemble and factor ``[[H, B^T], [B, -M]]``; 1 on a singular system.

    ``work``: max((2 r + 1) kmax^2, 3 kmax^2, n + r).
    """
    cdef int n = rep.n, r = rep.r, nn = n + r, i, j, bb
    cdef bint failed = False
    for bb in range(rep.nb):
        _spd_inverse(
            s + rep.qoff[bb], rep.k[bb], kkt.sinv + rep.qoff[bb], work, &failed
        )
        if failed:
            return 1
    _schur(rep, q, kkt.sinv, kkt.schur, work)
    for i in range(n):
        memcpy(kkt.mat + i * nn, hs.h + i * n, n * sizeof(double))
        for j in range(r):
            kkt.mat[i * nn + n + j] = rep.b[j * n + i]
    for i in range(r):
        memcpy(kkt.mat + (n + i) * nn, rep.b + i * n, n * sizeof(double))
        memcpy(kkt.mat + (n + i) * nn + n, kkt.schur + i * r, r * sizeof(double))
    for i in range(n):
        work[i] = theta[i] - theta0[i]
    for i in range(n):
        kkt.dual_res[i] = g[i] + c_dot(hs.h + i * n, work, n)
    for j in range(r):
        if y[j] != 0.0:
            c_axpy(y[j], rep.b + j * n, kkt.dual_res, n)
    _gram_map(rep, q, work)
    for i in range(r):
        kkt.primal_res[i] = c_dot(rep.b + i * n, theta, n) - work[i]
    return _lu_factor(kkt.mat, nn, kkt.piv)


cdef void _direction(
    const Rep* rep,
    const Hessian* hs,
    const Kkt* kkt,
    const double* q,
    const double* targets,
    double* d_theta,
    double* d_dual,
    double* d_q,
    double* d_s,
    double* work,
) noexcept nogil:
    """One search direction for complementarity ``targets`` (packed k x k per block).

    ``work``: 3 nn + 4 r + 2 kmax^2.
    """
    cdef int n = rep.n, r = rep.r, nn = n + r, i, j, bb, kk, k2, t
    cdef int km2 = rep.kmax * rep.kmax
    cdef double* rhs = work
    cdef double* sol = rhs + nn
    cdef double* res = sol + nn
    cdef double* ell = res + nn
    cdef double* mismatch = ell + r
    cdef double* coef = mismatch + r
    cdef double* vec = coef + r
    cdef double* tmp = vec + r
    cdef double* tmp2 = tmp + km2
    cdef const double* ab
    cdef bint nonzero
    # ell_i = sum_b tr(A_{b,i} (T_b S_b^-1)) = sum_b <A_{b,i}, (T_b S_b^-1)^T>.
    memset(ell, 0, r * sizeof(double))
    for bb in range(rep.nb):
        kk = rep.k[bb]
        k2 = kk * kk
        ab = rep.a + rep.aoff[bb]
        _matmul(targets + rep.qoff[bb], kkt.sinv + rep.qoff[bb], tmp, kk)
        _transpose(tmp, tmp2, kk)
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            ell[i] += c_dot(ab + i * k2, tmp2, k2)
    for i in range(n):
        rhs[i] = -kkt.dual_res[i]
    for i in range(r):
        rhs[n + i] = -kkt.primal_res[i] + ell[i]
    memcpy(sol, rhs, nn * sizeof(double))
    _lu_solve(kkt.mat, nn, kkt.piv, sol)
    # One step of iterative refinement: res = rhs - K sol from the unfactored blocks.
    for i in range(n):
        res[i] = rhs[i] - c_dot(hs.h + i * n, sol, n)
        for j in range(r):
            res[i] -= rep.b[j * n + i] * sol[n + j]
    for i in range(r):
        res[n + i] = (
            rhs[n + i]
            - c_dot(rep.b + i * n, sol, n)
            - c_dot(kkt.schur + i * r, sol + n, r)
        )
    _lu_solve(kkt.mat, nn, kkt.piv, res)
    for i in range(nn):
        sol[i] += res[i]
    memcpy(d_theta, sol, n * sizeof(double))
    memcpy(d_dual, sol + n, r * sizeof(double))
    _slacks(rep, d_dual, d_s)
    # dQ = sym((T - Q dS) S^-1).
    for bb in range(rep.nb):
        kk = rep.k[bb]
        k2 = kk * kk
        _matmul(q + rep.qoff[bb], d_s + rep.qoff[bb], tmp, kk)
        for i in range(k2):
            tmp[i] = targets[rep.qoff[bb] + i] - tmp[i]
        _matmul(tmp, kkt.sinv + rep.qoff[bb], d_q + rep.qoff[bb], kk)
        _symmetrize(d_q + rep.qoff[bb], kk)
    if not rep.has_complement:
        return
    # Close the linearized primal equation outside range(B) with the
    # minimum-norm Gram correction (lstsq on the Gram of the rows).
    _gram_map(rep, d_q, vec)
    for i in range(r):
        vec[i] = c_dot(rep.b + i * n, d_theta, n) - vec[i] + kkt.primal_res[i]
    nonzero = False
    for i in range(r):
        mismatch[i] = c_dot(rep.complement + i * r, vec, r)
        if mismatch[i] != 0.0:
            nonzero = True
    if not nonzero:
        return
    memset(coef, 0, r * sizeof(double))
    for i in range(r):
        if fabs(rep.gg_val[i]) > rep.gg_cut:
            c_axpy(
                c_dot(rep.gg_vec + i * r, mismatch, r) / rep.gg_val[i],
                rep.gg_vec + i * r,
                coef,
                r,
            )
    for bb in range(rep.nb):
        kk = rep.k[bb]
        k2 = kk * kk
        ab = rep.a + rep.aoff[bb]
        for t in range(rep.blk_start[bb], rep.blk_start[bb + 1]):
            i = rep.blk_rows[t]
            if coef[i] != 0.0:
                c_axpy(coef[i], ab + i * k2, d_q + rep.qoff[bb], k2)


cdef double _max_step(
    const double* m, const double* d, int k, double* work, bint* failed
) noexcept nogil:
    """Largest ``alpha`` keeping ``m + alpha d`` PSD (``work``: 5 k^2 + 3 k)."""
    cdef double* chol = work
    cdef double* y = chol + k * k
    cdef double* yt = y + k * k
    cdef double* x = yt + k * k
    cdef double* eig = x + k * k
    cdef double smallest
    memcpy(chol, m, k * k * sizeof(double))
    if _cholesky(chol, k) != 0:
        failed[0] = True
        return 0.0
    _lower_solve_rows(chol, d, y, k)
    _transpose(y, yt, k)
    _lower_solve_rows(chol, yt, x, k)
    _symmetrize(x, k)
    smallest = _min_eig(x, k, eig)
    if smallest >= 0.0:
        return INFINITY
    return -1.0 / smallest


cdef double _step_limit(
    const Rep* rep,
    const double* q,
    const double* dq,
    const double* s,
    const double* ds,
    double* work,
    bint* failed,
) noexcept nogil:
    cdef int bb, kk
    cdef double limit = INFINITY, v
    for bb in range(rep.nb):
        kk = rep.k[bb]
        v = _max_step(q + rep.qoff[bb], dq + rep.qoff[bb], kk, work, failed)
        if failed[0]:
            return 0.0
        if v < limit:
            limit = v
        v = _max_step(s + rep.qoff[bb], ds + rep.qoff[bb], kk, work, failed)
        if failed[0]:
            return 0.0
        if v < limit:
            limit = v
    return limit


cdef bint _all_pd(const Rep* rep, const double* packed, double* work) noexcept nogil:
    cdef int bb, kk
    for bb in range(rep.nb):
        kk = rep.k[bb]
        memcpy(work, packed + rep.qoff[bb], kk * kk * sizeof(double))
        if _cholesky(work, kk) != 0:
            return False
    return True


cdef Py_ssize_t _ipm_work(int n, int r, Py_ssize_t qtot, int kmax) noexcept nogil:
    """Scratch doubles for any helper called from the interior-point loop."""
    cdef Py_ssize_t nn = n + r
    cdef Py_ssize_t km2 = kmax * kmax
    cdef Py_ssize_t need = 3 * nn + 4 * r + 2 * km2  # _direction
    cdef Py_ssize_t other
    other = 2 * r + qtot + 4 * n + 2 * r + km2 + 3 * kmax  # _certified_gap
    if other > need:
        need = other
    other = (2 * r + 1) * km2 + 3 * km2 + nn  # _kkt_build
    if other > need:
        need = other
    other = 5 * km2 + 3 * kmax  # _step_limit
    if other > need:
        need = other
    return need + 16


cdef int _solve_ipm(
    const Rep* rep,
    const Hessian* hs,
    const double* g,
    const double* theta0,
    const double* start,
    double gap_tolerance,
    int max_iterations,
    double step_fraction,
    double* out_params,
    double* out_blocks,
    double* out_dual,
    double* out_model,
    double* out_gap,
    int* out_iterations,
) noexcept nogil:
    """The interior-point loop; 0 on success, -1 allocation failure, -2 start not PD."""
    cdef int n = rep.n, r = rep.r, nn = n + r, nb = rep.nb
    cdef Py_ssize_t qtot = rep.qtot
    cdef Py_ssize_t size, wsize
    cdef double* arena
    cdef double* p
    cdef double* theta
    cdef double* dual
    cdef double* q
    cdef double* s
    cdef double* endpoint
    cdef double* best_endpoint
    cdef double* best_q
    cdef double* best_dual
    cdef double* history
    cdef double* dq_aff
    cdef double* ds_aff
    cdef double* dq
    cdef double* ds
    cdef double* d_theta
    cdef double* d_dual
    cdef double* targets
    cdef double* new_q
    cdef double* new_dual
    cdef double* new_s
    cdef double* work
    cdef int* piv
    cdef Kkt kkt
    cdef int iterations = 0, n_history = 0, bb, kk, i, off
    cdef double nu = 0.0, pairing, target_mu, mu, mu_aff, sigma, alpha_aff, alpha, v
    cdef double gap, model_value = 0.0, residual
    cdef double best_gap = INFINITY, best_residual = INFINITY
    cdef double best_model = 0.0, scale
    cdef bint have_best = False, failed

    for bb in range(nb):
        nu += rep.k[bb]
    wsize = _ipm_work(n, r, qtot, rep.kmax)
    size = 4 * n + 4 * r + 13 * qtot + max_iterations + nn * nn + r * r + n + r + wsize
    arena = <double*> malloc(size * sizeof(double))
    piv = <int*> malloc((nn + 1) * sizeof(int))
    if arena == NULL or piv == NULL:
        free(arena)
        free(piv)
        return -1
    p = arena
    theta = p
    p += n
    endpoint = p
    p += n
    best_endpoint = p
    p += n
    d_theta = p
    p += n
    dual = p
    p += r
    best_dual = p
    p += r
    d_dual = p
    p += r
    new_dual = p
    p += r
    q = p
    p += qtot
    s = p
    p += qtot
    best_q = p
    p += qtot
    dq_aff = p
    p += qtot
    ds_aff = p
    p += qtot
    dq = p
    p += qtot
    ds = p
    p += qtot
    targets = p
    p += qtot
    new_q = p
    p += qtot
    new_s = p
    p += qtot
    kkt.sinv = p
    p += qtot
    p += 2 * qtot                          # spare, keeps offsets stable
    history = p
    p += max_iterations
    kkt.mat = p
    p += nn * nn
    kkt.schur = p
    p += r * r
    kkt.dual_res = p
    p += n
    kkt.primal_res = p
    p += r
    work = p
    kkt.nn = nn
    kkt.piv = piv

    memcpy(theta, theta0, n * sizeof(double))
    memcpy(q, start, qtot * sizeof(double))
    if not _all_pd(rep, q, work):
        free(arena)
        free(piv)
        return -2
    # Interior dual start: a positive multiple of the reference dual, scaled
    # so that the initial complementarity is comparable to the model scale.
    _slacks(rep, rep.ref, s)
    pairing = c_dot(q, s, qtot)
    target_mu = c_absmax(g, n)
    if target_mu < 1e-8:
        target_mu = 1e-8
    v = target_mu * nu / (pairing if pairing > 1e-300 else 1e-300)
    for i in range(r):
        dual[i] = v * rep.ref[i]

    while iterations < max_iterations:
        iterations += 1
        gap = _certified_gap(
            rep, hs, g, theta0, theta, q, dual, endpoint, &model_value, work
        )
        residual = _residual(rep, endpoint, q, work)
        # Rank by certified gap; among uncertified iterates prefer the one
        # closest to primal feasibility, never the starting guess.
        if (
            (not have_best)
            or gap < best_gap
            or (gap == best_gap and residual < best_residual)
        ):
            have_best = True
            best_gap = gap
            best_residual = residual
            best_model = model_value
            memcpy(best_endpoint, endpoint, n * sizeof(double))
            memcpy(best_q, q, qtot * sizeof(double))
            memcpy(best_dual, dual, r * sizeof(double))
        scale = fabs(model_value) if fabs(model_value) > 1.0 else 1.0
        if gap <= gap_tolerance * scale:
            break
        _slacks(rep, dual, s)
        mu = c_dot(q, s, qtot) / nu
        # Stall detection applies only near the roundoff floor.
        if mu <= 1e-9 * scale:
            history[n_history] = best_gap
            n_history += 1
            if n_history > 5 and history[n_history - 1] > 0.5 * history[n_history - 6]:
                break
        if _kkt_build(rep, hs, g, theta0, theta, q, dual, s, &kkt, work) != 0:
            break
        # Predictor: targets -Q S.
        for bb in range(nb):
            kk = rep.k[bb]
            off = rep.qoff[bb]
            _matmul(q + off, s + off, targets + off, kk)
            c_scale(-1.0, targets + off, kk * kk)
        _direction(rep, hs, &kkt, q, targets, d_theta, d_dual, dq_aff, ds_aff, work)
        failed = False
        alpha_aff = _step_limit(rep, q, dq_aff, s, ds_aff, work, &failed)
        if failed:
            break
        if alpha_aff > 1.0:
            alpha_aff = 1.0
        mu_aff = 0.0
        for i in range(qtot):
            mu_aff += (q[i] + alpha_aff * dq_aff[i]) * (s[i] + alpha_aff * ds_aff[i])
        mu_aff /= nu
        sigma = 0.0
        if mu > 0.0:
            sigma = mu_aff / mu
            if sigma < 0.0:
                sigma = 0.0
            if sigma > 1.0:
                sigma = 1.0
            sigma = sigma * sigma * sigma
        # Corrector: sigma mu I - Q S - dQ_aff dS_aff.
        for bb in range(nb):
            kk = rep.k[bb]
            off = rep.qoff[bb]
            _matmul(q + off, s + off, targets + off, kk)
            _matmul(dq_aff + off, ds_aff + off, work, kk)
            for i in range(kk * kk):
                targets[off + i] = -targets[off + i] - work[i]
            for i in range(kk):
                targets[off + i * kk + i] += sigma * mu
        _direction(rep, hs, &kkt, q, targets, d_theta, d_dual, dq, ds, work)
        failed = False
        alpha = _step_limit(rep, q, dq, s, ds, work, &failed)
        if failed:
            break
        alpha *= step_fraction
        if alpha > 1.0:
            alpha = 1.0
        if alpha < 1e-10:
            break
        for i in range(qtot):
            new_q[i] = q[i] + alpha * dq[i]
        for bb in range(nb):
            _symmetrize(new_q + rep.qoff[bb], rep.k[bb])
        for i in range(r):
            new_dual[i] = dual[i] + alpha * d_dual[i]
        if not _all_pd(rep, new_q, work):
            break
        _slacks(rep, new_dual, new_s)
        if not _all_pd(rep, new_s, work):
            break
        c_axpy(alpha, d_theta, theta, n)
        memcpy(dual, new_dual, r * sizeof(double))
        memcpy(q, new_q, qtot * sizeof(double))

    memcpy(out_params, best_endpoint, n * sizeof(double))
    memcpy(out_blocks, best_q, qtot * sizeof(double))
    memcpy(out_dual, best_dual, r * sizeof(double))
    out_model[0] = best_model
    out_gap[0] = best_gap
    out_iterations[0] = iterations
    free(arena)
    free(piv)
    return 0


# ---------------------------------------------------------------------------
# Owned representations and the Python entry points
# ---------------------------------------------------------------------------

cdef struct OwnedRep:
    Rep rep
    double* storage
    int* ints


cdef int _own(
    OwnedRep* own,
    int n,
    int r,
    int nb,
    int kmax,
    const int* k,
    const Py_ssize_t* aoff,
    const Py_ssize_t* qoff,
    Py_ssize_t qtot,
    const double* b,
    const double* a,
    const double* ref,
    bint need_complement,
) noexcept nogil:
    """Prepare a description (allocating its derived storage); 0 on success."""
    cdef Py_ssize_t storage = _rep_storage(n, r) + _prepare_size(n, r, qtot, kmax)
    own.storage = <double*> malloc(storage * sizeof(double))
    own.ints = <int*> malloc((2 * r + 2 + nb + 1 + nb * r) * sizeof(int))
    if own.storage == NULL or own.ints == NULL:
        free(own.storage)
        free(own.ints)
        own.storage = NULL
        own.ints = NULL
        return -1
    own.rep.n = n
    own.rep.r = r
    own.rep.nb = nb
    own.rep.kmax = kmax
    own.rep.k = k
    own.rep.aoff = aoff
    own.rep.qoff = qoff
    own.rep.qtot = qtot
    own.rep.b = b
    own.rep.a = a
    own.rep.ref = ref
    own.rep.pinv = own.storage
    own.rep.complement = own.rep.pinv + n * r
    own.rep.gg_vec = own.rep.complement + r * r
    own.rep.gg_val = own.rep.gg_vec + r * r
    own.rep.exact_row = own.ints
    own.rep.exact_col = own.ints + r
    own.rep.blk_start = own.ints + 2 * r + 2
    own.rep.blk_rows = own.rep.blk_start + nb + 1
    if _prepare(&own.rep, need_complement, own.rep.gg_val + r) != 0:
        return -3
    return 0


cdef void _disown(OwnedRep* own) noexcept nogil:
    free(own.storage)
    free(own.ints)
    own.storage = NULL
    own.ints = NULL


def _as_c(value, dtype=np.float64):
    """Adapt a compiled-entry buffer to C-contiguous storage.

    Parameters
    ----------
    value : array_like
        Buffer to bind; existing compatible arrays may be reused.
    dtype : numpy dtype, optional
        Required element type, defaulting to float64.
    """
    return np.ascontiguousarray(value, dtype=dtype)


def solve_conic_qp(
    hessian, gradient, params, b_matrix, a_packed, sizes, a_offsets, q_offsets,
    reference_dual, start_packed, double gap_tolerance, int max_iterations,
    double step_fraction,
):
    """Solve one unscaled Newton model over a packed cone description.

    The fitting path always goes through :func:`solve_preconditioned`; this
    entry exposes the bare interior-point solve to the tests.

    Returns ``(params, blocks_packed, dual, model_value, gap, iterations)``.

    Parameters
    ----------
    hessian, gradient, params : array_like
        Quadratic Hessian ``(n, n)``, gradient ``(n,)`` and starting parameters.
    b_matrix : array_like, shape (r, n)
        Parameter-to-constraint row map.
    a_packed, sizes, a_offsets, q_offsets : array_like
        Packed cone row operators, Gram block sizes and row/Gram offsets.
    reference_dual : array_like, shape (r,)
        Interior reference equality multipliers.
    start_packed : array_like
        Positive-definite starting Gram blocks in the solver's packed ordering.
    gap_tolerance : float
        Interior-point duality-gap target.
    max_iterations : int
        Interior-point iteration budget.
    step_fraction : float
        Fraction of the maximal feasible step to take.

    Raises
    ------
    numpy.linalg.LinAlgError
        If a starting Gram block is not positive definite.
    """
    cdef const double[:, ::1] h = _as_c(hessian)
    cdef const double[::1] g = _as_c(gradient)
    cdef const double[::1] theta0 = _as_c(params)
    cdef const double[:, ::1] b = _as_c(b_matrix)
    cdef const double[::1] a = _as_c(a_packed)
    cdef const int[::1] k = _as_c(sizes, np.intc)
    cdef const Py_ssize_t[::1] aoff = _as_c(a_offsets, np.intp)
    cdef const Py_ssize_t[::1] qoff = _as_c(q_offsets, np.intp)
    cdef const double[::1] ref = _as_c(reference_dual)
    cdef const double[::1] start = _as_c(start_packed)
    cdef int n = b.shape[1], r = b.shape[0], nb = k.shape[0], kmax = 1, i
    cdef Py_ssize_t qtot = start.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_params = np.empty(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_blocks = np.empty(qtot)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_dual = np.empty(r)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] hsym = np.ascontiguousarray(
        0.5 * (np.asarray(h) + np.asarray(h).T)
    )
    cdef double[:, ::1] hv = hsym
    cdef double model_value = 0.0, gap = INFINITY
    cdef int iterations = 0, status = 0
    cdef OwnedRep own
    cdef Hessian hs
    cdef double* hbuf = NULL
    cdef double* hwork = NULL
    cdef const int* k_ptr = &k[0] if nb > 0 else NULL
    cdef const Py_ssize_t* aoff_ptr = &aoff[0] if nb > 0 else NULL
    cdef const Py_ssize_t* qoff_ptr = &qoff[0] if nb > 0 else NULL
    cdef const double* b_ptr = &b[0, 0] if r > 0 and n > 0 else NULL
    cdef const double* a_ptr = &a[0] if a.shape[0] > 0 else NULL
    cdef const double* ref_ptr = &ref[0] if r > 0 else NULL
    cdef double* hv_ptr = &hv[0, 0] if n > 0 else NULL
    cdef const double* g_ptr = &g[0] if n > 0 else NULL
    cdef const double* theta0_ptr = &theta0[0] if n > 0 else NULL
    cdef const double* start_ptr = &start[0] if qtot > 0 else NULL
    cdef double* out_params_ptr = &out_params[0] if n > 0 else NULL
    cdef double* out_blocks_ptr = &out_blocks[0] if qtot > 0 else NULL
    cdef double* out_dual_ptr = &out_dual[0] if r > 0 else NULL
    own.storage = NULL
    own.ints = NULL
    for i in range(nb):
        if k[i] > kmax:
            kmax = k[i]
    try:
        with nogil:
            status = _own(
                &own,
                n,
                r,
                nb,
                kmax,
                k_ptr,
                aoff_ptr,
                qoff_ptr,
                qtot,
                b_ptr,
                a_ptr,
                ref_ptr,
                True,
            )
            if status == 0:
                hbuf = <double*> malloc((n * n + n) * sizeof(double))
                hwork = <double*> malloc((2 * n * n + 2 * n + 8) * sizeof(double))
                if hbuf == NULL or hwork == NULL:
                    status = -1
                else:
                    hs.n = n
                    hs.h = hv_ptr
                    hs.vec = hbuf
                    hs.val = hbuf + n * n
                    if _hessian_eigen(&hs, hwork) != 0:
                        status = -3
            if status == 0:
                status = _solve_ipm(
                    &own.rep,
                    &hs,
                    g_ptr,
                    theta0_ptr,
                    start_ptr,
                    gap_tolerance,
                    max_iterations,
                    step_fraction,
                    out_params_ptr,
                    out_blocks_ptr,
                    out_dual_ptr,
                    &model_value,
                    &gap,
                    &iterations,
                )
    finally:
        with nogil:
            _disown(&own)
            free(hbuf)
            free(hwork)
    if status == -2:
        raise np.linalg.LinAlgError("starting Gram block is not positive definite")
    if status == -1:
        raise MemoryError("conic QP workspace")
    if status != 0:
        raise np.linalg.LinAlgError("conic QP precomputation did not converge")
    return out_params, out_blocks, out_dual, model_value, gap, iterations


def solve_preconditioned(
    hessian, gradient, params, b_matrix, a_packed, sizes, a_offsets, q_offsets,
    reference_dual, row_degrees, blocks_packed, double gap_tolerance,
    int max_iterations, double step_fraction, column_scale=None,
):
    """Solve one Newton model in Jacobi/geometric scaled variables.

    Compiled ``_preconditioned_subproblem``: ``theta = D phi`` with Jacobi
    ``D = diag(H)^-1/2``; polynomial rows of power ``i`` scaled by
    ``sigma^i``, Gram bases by ``v(t / sigma)``, other rows equilibrated;
    the start is the given blocks shifted inward by 1e-3 of their mean
    eigenvalue.

    Parameters
    ----------
    hessian, gradient, params : array_like
        Quadratic Hessian ``(n, n)``, gradient ``(n,)`` and starting parameters.
    b_matrix : array_like, shape (r, n)
        Parameter-to-constraint row map.
    a_packed, sizes, a_offsets, q_offsets : array_like
        Packed cone row operators, Gram block sizes and row/Gram offsets.
    reference_dual, row_degrees : array_like, shape (r,)
        Reference equality multipliers and polynomial powers for row scaling.
    blocks_packed : array_like
        Starting Gram blocks in the solver's packed ordering.
    gap_tolerance : float
        Interior-point duality-gap target.
    max_iterations : int
        Interior-point iteration budget.
    step_fraction : float
        Fraction of the maximal feasible step to take.
    column_scale : array_like or None
        Positive column scales replacing Jacobi (``None``: Jacobi).

    Returns
    -------
    tuple
        ``(endpoint, endpoint_blocks_packed, model_value, gap, iterations,
        dual_scaled, scaled_model_value)``: the endpoint reconstructed in
        natural coordinates against the original description, its blocks in
        the original basis and Newton model value, and the scaled solve's
        certified gap, iteration count, multipliers and model value.
    """
    cdef const double[:, ::1] h = _as_c(hessian)
    cdef const double[::1] g = _as_c(gradient)
    cdef const double[::1] theta = _as_c(params)
    cdef const double[:, ::1] b = _as_c(b_matrix)
    cdef const double[::1] a = _as_c(a_packed)
    cdef const int[::1] k = _as_c(sizes, np.intc)
    cdef const Py_ssize_t[::1] aoff = _as_c(a_offsets, np.intp)
    cdef const Py_ssize_t[::1] qoff = _as_c(q_offsets, np.intp)
    cdef const double[::1] ref = _as_c(reference_dual)
    cdef const Py_ssize_t[::1] deg = _as_c(row_degrees, np.intp)
    cdef const double[::1] blocks = _as_c(blocks_packed)
    return _solve_preconditioned_buffers(
        h, g, theta, b, a, k, aoff, qoff, ref, deg, blocks,
        gap_tolerance, max_iterations, step_fraction,
        None if column_scale is None else _as_c(column_scale),
    )


cdef tuple _solve_preconditioned_buffers(
    const double[:, ::1] h,
    const double[::1] g,
    const double[::1] theta,
    const double[:, ::1] b,
    const double[::1] a,
    const int[::1] k,
    const Py_ssize_t[::1] aoff,
    const Py_ssize_t[::1] qoff,
    const double[::1] ref,
    const Py_ssize_t[::1] deg,
    const double[::1] blocks,
    double gap_tolerance,
    int max_iterations,
    double step_fraction,
    const double[::1] col_scale=None,
):
    """Solve a quadratic from bound contiguous buffers without recoercion.

    ``col_scale`` (length ``n``, positive) replaces the Jacobi column scaling
    when given; ``None`` keeps G1's Jacobi scaling.

    The caller retains the compatible quadratic/cone buffer owners and supplies
    explicit interior-point controls. Only the result buffers are allocated here.
    """
    cdef int n = b.shape[1], r = b.shape[0], nb = k.shape[0]
    cdef Py_ssize_t qtot = blocks.shape[0], na = a.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] endpoint = np.empty(n)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_blocks = np.empty(qtot)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out_dual = np.empty(r)
    cdef double model_value = 0.0, gap = INFINITY, scaled_model = 0.0
    cdef int iterations = 0, status = 0
    cdef const int* k_ptr = &k[0] if nb > 0 else NULL
    cdef const Py_ssize_t* aoff_ptr = &aoff[0] if nb > 0 else NULL
    cdef const Py_ssize_t* qoff_ptr = &qoff[0] if nb > 0 else NULL
    cdef const double* h_ptr = &h[0, 0] if n > 0 else NULL
    cdef const double* g_ptr = &g[0] if n > 0 else NULL
    cdef const double* theta_ptr = &theta[0] if n > 0 else NULL
    cdef const double* b_ptr = &b[0, 0] if r > 0 and n > 0 else NULL
    cdef const double* a_ptr = &a[0] if na > 0 else NULL
    cdef const double* ref_ptr = &ref[0] if r > 0 else NULL
    cdef const Py_ssize_t* deg_ptr = &deg[0] if r > 0 else NULL
    cdef const double* blocks_ptr = &blocks[0] if qtot > 0 else NULL
    cdef const double* col_ptr = NULL
    if col_scale is not None and n > 0:
        if col_scale.shape[0] != n:
            raise ValueError("column scale must have one entry per variable")
        col_ptr = &col_scale[0]
    cdef double* endpoint_ptr = &endpoint[0] if n > 0 else NULL
    cdef double* out_blocks_ptr = &out_blocks[0] if qtot > 0 else NULL
    cdef double* out_dual_ptr = &out_dual[0] if r > 0 else NULL
    with nogil:
        status = _preconditioned(
            n,
            r,
            nb,
            k_ptr,
            aoff_ptr,
            qoff_ptr,
            qtot,
            na,
            h_ptr,
            g_ptr,
            theta_ptr,
            b_ptr,
            a_ptr,
            ref_ptr,
            deg_ptr,
            blocks_ptr,
            col_ptr,
            gap_tolerance,
            max_iterations,
            step_fraction,
            endpoint_ptr,
            out_blocks_ptr,
            out_dual_ptr,
            &model_value,
            &gap,
            &iterations,
            &scaled_model,
        )
    if status == -2:
        raise np.linalg.LinAlgError("starting Gram block is not positive definite")
    if status == -1:
        raise MemoryError("conic QP workspace")
    if status != 0:
        raise np.linalg.LinAlgError("conic QP precomputation did not converge")
    return endpoint, out_blocks, model_value, gap, iterations, out_dual, scaled_model


cdef int _preconditioned(int n, int r, int nb, const int* k, const Py_ssize_t* aoff,
                         const Py_ssize_t* qoff, Py_ssize_t qtot, Py_ssize_t na,
                         const double* h, const double* g, const double* theta,
                         const double* b, const double* a, const double* ref,
                         const Py_ssize_t* deg, const double* blocks,
                         const double* col_scale,
                         double gap_tolerance, int max_iterations, double step_fraction,
                         double* endpoint, double* out_blocks, double* out_dual,
                         double* model_value, double* gap, int* iterations,
                         double* scaled_model) noexcept nogil:
    cdef int kmax = 1, i, j, bb, kk, aa, cc, m, status
    cdef Py_ssize_t size
    cdef double* arena
    cdef double* p
    cdef double* column
    cdef double* magnitude
    cdef double* row_scale
    cdef double* gscale
    cdef double* bs
    cdef double* as_
    cdef double* refs
    cdef double* hs_mat
    cdef double* gs
    cdef double* phi
    cdef double* start
    cdef double* sol
    cdef double* hbuf
    cdef double* hwork
    cdef double* work
    cdef double sigma, mean_x, mean_y, sxx, sxy, v, tr, shift, dmin, dmax
    cdef OwnedRep scaled
    cdef OwnedRep original
    cdef Hessian hs
    cdef Hessian hs_orig
    cdef Py_ssize_t off
    cdef int gscale_len = 0
    for bb in range(nb):
        if k[bb] > kmax:
            kmax = k[bb]
        gscale_len += k[bb]
    size = (n + r + r + gscale_len + r * n + na + r + n * n + n + n + qtot + n
            + n * n + n + 2 * n * n + 2 * n + 16 + 4 * n + 4 * r + 16)
    arena = <double*> malloc(size * sizeof(double))
    if arena == NULL:
        return -1
    p = arena
    column = p
    p += n
    magnitude = p
    p += r
    row_scale = p
    p += r
    gscale = p
    p += gscale_len
    bs = p
    p += r * n
    as_ = p
    p += na
    refs = p
    p += r
    hs_mat = p
    p += n * n
    gs = p
    p += n
    phi = p
    p += n
    start = p
    p += qtot
    sol = p
    p += n
    hbuf = p
    p += n * n + n
    hwork = p
    p += 2 * n * n + 2 * n + 16
    work = p
    # Column scaling: Jacobi, unless the caller supplies explicit positive
    # scales (G3 row-flattening scales for 1 x 1 row descriptions).
    for j in range(n):
        if col_scale != NULL:
            column[j] = col_scale[j]
        else:
            v = h[j * n + j]
            column[j] = 1.0 / sqrt(v) if v > 0.0 else 1.0
    # Row magnitudes after column scaling; geometric rate over polynomial rows.
    m = 0
    mean_x = 0.0
    mean_y = 0.0
    dmin = INFINITY
    dmax = -INFINITY
    for i in range(r):
        v = 0.0
        for j in range(n):
            if fabs(b[i * n + j] * column[j]) > v:
                v = fabs(b[i * n + j] * column[j])
        magnitude[i] = v if v != 0.0 else 1.0
        if deg[i] >= 0:
            m += 1
            mean_x += <double> deg[i]
            mean_y += -log(magnitude[i])
            if deg[i] < dmin:
                dmin = deg[i]
            if deg[i] > dmax:
                dmax = deg[i]
    sigma = 1.0
    if m >= 2 and dmax > dmin:
        mean_x /= m
        mean_y /= m
        sxx = 0.0
        sxy = 0.0
        for i in range(r):
            if deg[i] >= 0:
                sxx += (deg[i] - mean_x) * (deg[i] - mean_x)
                sxy += (deg[i] - mean_x) * (-log(magnitude[i]) - mean_y)
        sigma = exp(sxy / sxx)
    for i in range(r):
        row_scale[i] = sigma ** deg[i] if deg[i] >= 0 else 1.0 / magnitude[i]
    off = 0
    for bb in range(nb):
        for aa in range(k[bb]):
            gscale[off + aa] = sigma ** aa
        off += k[bb]
    # Scaled description.
    for i in range(r):
        for j in range(n):
            bs[i * n + j] = row_scale[i] * b[i * n + j] * column[j]
        refs[i] = ref[i] / row_scale[i]
    off = 0
    for bb in range(nb):
        kk = k[bb]
        for i in range(r):
            for aa in range(kk):
                for cc in range(kk):
                    as_[aoff[bb] + (i * kk + aa) * kk + cc] = (
                        row_scale[i] * a[aoff[bb] + (i * kk + aa) * kk + cc]
                        / (gscale[off + aa] * gscale[off + cc])
                    )
        # Start: blocks in the scaled basis, shifted inward by 1e-3 of their mean eigenvalue.
        tr = 0.0
        for aa in range(kk):
            for cc in range(kk):
                start[qoff[bb] + aa * kk + cc] = (
                    blocks[qoff[bb] + aa * kk + cc]
                    * gscale[off + aa]
                    * gscale[off + cc]
                )
            tr += start[qoff[bb] + aa * kk + aa]
        shift = tr / kk
        if shift < 1e-300:
            shift = 1e-300
        for aa in range(kk):
            start[qoff[bb] + aa * kk + aa] += 1e-3 * shift
        off += kk
    for i in range(n):
        for j in range(n):
            hs_mat[i * n + j] = (
                column[i] * 0.5 * (h[i * n + j] + h[j * n + i]) * column[j]
            )
        gs[i] = column[i] * g[i]
        phi[i] = theta[i] / column[i]
    status = _own(&scaled, n, r, nb, kmax, k, aoff, qoff, qtot, bs, as_, refs, True)
    if status != 0:
        _disown(&scaled)
        free(arena)
        return status
    hs.n = n
    hs.h = hs_mat
    hs.vec = hbuf
    hs.val = hbuf + n * n
    if _hessian_eigen(&hs, hwork) != 0:
        _disown(&scaled)
        free(arena)
        return -3
    status = _solve_ipm(
        &scaled.rep,
        &hs,
        gs,
        phi,
        start,
        gap_tolerance,
        max_iterations,
        step_fraction,
        sol,
        out_blocks,
        out_dual,
        scaled_model,
        gap,
        iterations,
    )
    _disown(&scaled)
    if status != 0:
        free(arena)
        return status
    # Back to natural coordinates and the original Gram basis.
    off = 0
    for bb in range(nb):
        kk = k[bb]
        for aa in range(kk):
            for cc in range(kk):
                out_blocks[qoff[bb] + aa * kk + cc] /= gscale[off + aa] * gscale[
                    off + cc
                ]
        off += kk
    for j in range(n):
        endpoint[j] = column[j] * sol[j]
    status = _own(&original, n, r, nb, kmax, k, aoff, qoff, qtot, b, a, ref, False)
    if status != 0:
        _disown(&original)
        free(arena)
        return status
    _reconstruct(&original.rep, endpoint, out_blocks, work)
    _disown(&original)
    # Model value in natural coordinates (H symmetrized as in the reference).
    for i in range(n):
        for j in range(n):
            hs_mat[i * n + j] = 0.5 * (h[i * n + j] + h[j * n + i])
    hs_orig.n = n
    hs_orig.h = hs_mat
    model_value[0] = _model(&hs_orig, g, theta, endpoint, work)
    free(arena)
    return 0


# ---------------------------------------------------------------------------
# Fused Newton support for finite real-line interval objectives
# ---------------------------------------------------------------------------

cdef int _in_safeguarded_metric(
    int n, const double* fisher, const double* missing,
    double* metric, double* work, double* smallest_out,
) noexcept nogil:
    """Build the interval saddle-free metric in small dense C kernels.

    This is the compiled equivalent of ``natural_objective._safeguarded_metric``.
    The generalized eigenproblem is whitened by the Fisher matrix after the
    same diagonal Jacobi scaling used by the Python/SciPy implementation.
    """
    cdef Py_ssize_t n2 = n * n
    cdef double* f = work
    cdef double* observed = f + n2
    cdef double* b = observed + n2
    cdef double* a = b + n2
    cdef double* y = a + n2
    cdef double* c = y + n2
    cdef double* v = c + n2
    cdef double* diag = v + n2
    cdef double* eig = diag + n
    cdef double* jac_b = eig + n
    cdef double* jac_z = jac_b + n
    cdef int i, j, k, status
    cdef double fi, oi, d, value, lo

    for i in range(n):
        for j in range(n):
            fi = 0.5 * (fisher[i * n + j] + fisher[j * n + i])
            oi = fi - 0.5 * (missing[i * n + j] + missing[j * n + i])
            f[i * n + j] = fi
            observed[i * n + j] = oi
    for i in range(n):
        d = f[i * n + i]
        if d < DBL_MIN:
            d = DBL_MIN
        diag[i] = sqrt(d)
    for i in range(n):
        for j in range(n):
            d = diag[i] * diag[j]
            b[i * n + j] = f[i * n + j] / d
            a[i * n + j] = observed[i * n + j] / d

    # B = L L^T.  If the Fisher metric is numerically singular, use Fisher
    # itself, exactly as ``_safeguarded_metric`` does.
    if _cholesky(b, n) != 0:
        memcpy(metric, f, n2 * sizeof(double))
        smallest_out[0] = -INFINITY
        return 0

    # C = L^-1 A L^-T.
    _lower_solve_rows(b, a, y, n)
    _transpose(y, a, n)
    _lower_solve_rows(b, a, c, n)
    _transpose(c, a, n)
    _symmetrize(a, n)
    status = _jacobi(a, n, eig, v, jac_b, jac_z, True)
    if status != 0:
        memcpy(metric, f, n2 * sizeof(double))
        smallest_out[0] = -INFINITY
        return 0

    lo = eig[0]
    for k in range(1, n):
        if eig[k] < lo:
            lo = eig[k]
    smallest_out[0] = lo
    if lo >= 1e-8:
        memcpy(metric, observed, n2 * sizeof(double))
        return 0

    # ``basis = B_scaled @ V = L @ U`` in the notation of the Python code.
    # Reuse y for the basis and c for the reflected scaled metric.
    for i in range(n):
        for k in range(n):
            value = 0.0
            for j in range(i + 1):
                value += b[i * n + j] * v[j * n + k]
            y[i * n + k] = value
    for i in range(n):
        for j in range(n):
            value = 0.0
            for k in range(n):
                d = fabs(eig[k])
                if d < 1e-3:
                    d = 1e-3
                value += y[i * n + k] * d * y[j * n + k]
            c[i * n + j] = value * diag[i] * diag[j]
    _symmetrize(c, n)
    memcpy(metric, c, n2 * sizeof(double))
    return 0


def safeguarded_metric(fisher, missing):
    """Saddle-free interval Newton metric (``_in_safeguarded_metric``).

    Parameters
    ----------
    fisher : numpy.ndarray
        Model Fisher matrix ``F``, shape ``(n, n)``.
    missing : numpy.ndarray
        Missing information ``C``, shape ``(n, n)``.

    Returns
    -------
    metric : numpy.ndarray
        ``F - C`` when its smallest eigenvalue relative to ``F`` is at least
        ``1e-8``, else its reflection ``F V max(|Lambda|, 1e-3) V^T F``
        (``F`` itself when ``F`` is numerically singular).
    smallest : float
        Smallest relative eigenvalue (``-inf`` when ``F`` is singular).
    """
    f = np.ascontiguousarray(fisher, dtype=np.float64)
    c = np.ascontiguousarray(missing, dtype=np.float64)
    cdef int n = f.shape[0]
    if f.ndim != 2 or f.shape[1] != n or c.shape != (n, n) or n < 1:
        raise ValueError("fisher and missing must be matching square matrices")
    metric = np.zeros((n, n))
    work = np.zeros(7 * n * n + 4 * n)
    cdef double[:, ::1] fv = f
    cdef double[:, ::1] cv = c
    cdef double[:, ::1] mv = metric
    cdef double[::1] wv = work
    cdef double smallest = NAN
    with nogil:
        _in_safeguarded_metric(n, &fv[0, 0], &cv[0, 0], &mv[0, 0], &wv[0], &smallest)
    return metric, float(smallest)


# ---------------------------------------------------------------------------
# Fused Newton loop for natural point objectives
# ---------------------------------------------------------------------------

cdef inline bint _pn_finite(double x) noexcept nogil:
    return (x == x) and (not isinf(x))


# ---------------------------------------------------------------------------
# Generic fixed-face Newton loop over a compiled objective callback
# ---------------------------------------------------------------------------

cdef int _face_newton_loop(
    int n, int r, int nb, const int* sizes, const Py_ssize_t* aoff,
    const Py_ssize_t* qoff, Py_ssize_t qtot, Py_ssize_t na,
    const double* b, const double* a, const double* ref, const Py_ssize_t* degrees,
    const double* col_scale,
    FaceObjective* objective,
    double* theta, double* blocks, double* current_nll, double* gradient,
    double* hessian, double* current_smallest, double* dual,
    double tolerance, double certified_tolerance, double accuracy_floor,
    int max_iterations, double armijo, double backtrack, int max_line_search,
    int min_steps, int* iterations_out, int* evaluations_out,
    int* subproblem_iterations_out, double* bound_out,
) noexcept nogil:
    """Run one fixed-face Newton solve over a compiled objective without the GIL.

    The control flow is the point/interval loops' line for line: warm then
    cold conic subproblems, the carried certificate, Armijo with the roundoff
    rule, and backtracking on any nonzero trial status.  The caller has
    evaluated and accepted ``theta``; ``objective.accept`` commits each
    accepted trial's raw geometry inside the objective context.

    Return codes 1--5 are the public solver statuses, 6 an objective's
    ``FACE_STOP``; negative codes come from the conic subproblem kernel (-1
    allocation failure).  ``col_scale`` (or NULL for Jacobi) fixes the
    subproblem column scales (G3-034).
    """
    cdef Py_ssize_t n2 = n * n
    cdef Py_ssize_t total = n + qtot + r + n + qtot + r + n + qtot + n + n2 + qtot
    cdef double* arena = <double*>malloc(total * sizeof(double))
    cdef double* p
    cdef double* endpoint
    cdef double* endpoint_blocks
    cdef double* endpoint_dual
    cdef double* chosen
    cdef double* chosen_blocks
    cdef double* chosen_dual
    cdef double* trial_theta
    cdef double* trial_blocks
    cdef double* trial_gradient
    cdef double* trial_hessian
    cdef double* cold_blocks
    cdef int iteration, _line_it, qp_iterations, status, bb, aa, kk, i
    cdef int evaluations = 0, sub_iterations = 0, cold_starts = 0
    cdef int first_solve, accepted, any_step, hinted
    cdef double bound = INFINITY, carried = INFINITY, scale, solved_bound
    cdef double model_value = 0.0, candidate_model = 0.0
    cdef double gap = INFINITY, scaled_model = 0.0
    cdef double directional, alpha, trial_nll, trial_smallest, decrease, cert, hint

    if arena == NULL:
        return -1
    p = arena
    endpoint = p
    p += n
    endpoint_blocks = p
    p += qtot
    endpoint_dual = p
    p += r
    chosen = p
    p += n
    chosen_blocks = p
    p += qtot
    chosen_dual = p
    p += r
    trial_theta = p
    p += n
    trial_blocks = p
    p += qtot
    trial_gradient = p
    p += n
    trial_hessian = p
    p += n2
    cold_blocks = p

    memset(cold_blocks, 0, qtot * sizeof(double))
    for bb in range(nb):
        kk = sizes[bb]
        for aa in range(kk):
            cold_blocks[qoff[bb] + aa * kk + aa] = 1.0

    for iteration in range(max_iterations):
        # Roundoff of the NLL value.  Tolerances on NLL decreases are absolute
        # (nats per observation): a change of data units shifts the NLL by
        # log(scale), so a tolerance relative to it moved the stopping point
        # with the units (G3-050).
        scale = fabs(current_nll[0])
        if scale < 1.0:
            scale = 1.0
        first_solve = 1
        while True:
            qp_iterations = 0
            status = _preconditioned(
                n, r, nb, sizes, aoff, qoff, qtot, na,
                hessian, gradient, theta, b, a, ref, degrees,
                blocks if first_solve else cold_blocks, col_scale,
                _QP_GAP_TOL, _QP_MAX_ITER, _QP_STEP_FRACTION,
                endpoint, endpoint_blocks, endpoint_dual,
                &candidate_model, &gap, &qp_iterations, &scaled_model,
            )
            # A resolved scalar projection may improve the conic endpoint.
            # Both kernels use the same Newton traversal and original-model
            # certificate; no basis-family dispatch is involved.
            if objective.refine != NULL and nb == r and qtot == r:
                if status != 0:
                    candidate_model = 0.0
                    gap = INFINITY
                if objective.refine(objective.ctx, n, r, b, hessian, gradient,
                                    theta, endpoint, endpoint_dual,
                                    &candidate_model, &gap):
                    status = 0
                    for bb in range(nb):
                        endpoint_blocks[bb] = 0.0
                        for i in range(n):
                            endpoint_blocks[bb] += b[bb*n+i] * endpoint[i]
                        if endpoint_blocks[bb] <= 0.0:
                            endpoint_blocks[bb] = 1.0
            if status != 0:
                free(arena)
                return status
            sub_iterations += qp_iterations
            solved_bound = (-candidate_model if candidate_model < 0.0 else 0.0) \
                + (gap if gap > 0.0 else 0.0)
            if first_solve or solved_bound < bound:
                bound = solved_bound
                model_value = candidate_model
                memcpy(chosen, endpoint, n * sizeof(double))
                memcpy(chosen_blocks, endpoint_blocks, qtot * sizeof(double))
                memcpy(chosen_dual, endpoint_dual, r * sizeof(double))
            directional = 0.0
            any_step = 0
            for i in range(n):
                directional += gradient[i] * (chosen[i] - theta[i])
                if chosen[i] != theta[i]:
                    any_step = 1
            if (
                (_pn_finite(directional) and directional < 0.0 and any_step)
                or (not first_solve)
                or bound <= certified_tolerance
                or cold_starts >= 2
            ):
                break
            cold_starts += 1
            first_solve = 0

        memcpy(dual, chosen_dual, r * sizeof(double))
        if bound <= tolerance and (
            iteration >= min_steps or model_value >= 0.0
        ):
            iterations_out[0] = iteration
            evaluations_out[0] = evaluations
            subproblem_iterations_out[0] = sub_iterations
            bound_out[0] = bound
            free(arena)
            return 1

        if not (_pn_finite(directional) and directional < 0.0 and any_step):
            if carried < bound:
                bound = carried
            iterations_out[0] = iteration
            evaluations_out[0] = evaluations
            subproblem_iterations_out[0] = sub_iterations
            bound_out[0] = bound
            free(arena)
            if bound <= certified_tolerance:
                return 1
            if bound <= accuracy_floor:
                return 2
            return 3

        alpha = 1.0
        accepted = 0
        hinted = 0
        for _line_it in range(max_line_search):
            for i in range(n):
                trial_theta[i] = theta[i] + alpha * (chosen[i] - theta[i])
            for i in range(qtot):
                trial_blocks[i] = (1.0 - alpha) * blocks[i] + alpha * chosen_blocks[i]
            evaluations += 1
            status = objective.evaluate(
                objective.ctx, trial_theta, &trial_nll, trial_gradient,
                trial_hessian, &trial_smallest,
            )
            if status == FACE_STOP:
                # The objective ends the run at the accepted iterate (G3).
                iterations_out[0] = iteration
                evaluations_out[0] = evaluations
                subproblem_iterations_out[0] = sub_iterations
                bound_out[0] = bound
                free(arena)
                return 6
            if status != 0:
                # A non-normalizable or numerically unresolved trial is a
                # line-search rejection, not a reason to abandon the solve.
                # A shape rejection by rows linear in theta may give the
                # largest step that keeps them feasible: halving discarded
                # half of a step that missed a moving curvature minimum by
                # 1e-9, so the cut exchange converged at a rate of 0.6
                # (G3-060).  The shortened trial is gated again.
                hint = -1.0
                if objective.step != NULL:
                    hint = objective.step(
                        objective.ctx, theta, trial_theta, alpha
                    )
                    if hinted and hint > _HINT_SHRINK * alpha:
                        hint = _HINT_SHRINK * alpha
                if hint > 0.0 and hint < alpha:
                    alpha = hint
                    hinted = 1
                else:
                    alpha *= backtrack
                continue
            if (
                trial_nll <= current_nll[0] + armijo * alpha * directional
                or (
                    -model_value <= 1e-9
                    and trial_nll <= current_nll[0] + 8.0 * _EPS * scale
                )
            ):
                accepted = 1
                break
            alpha *= backtrack

        if not accepted:
            cert = bound if bound < carried else carried
            if cert <= accuracy_floor:
                bound = cert
                iterations_out[0] = iteration
                evaluations_out[0] = evaluations
                subproblem_iterations_out[0] = sub_iterations
                bound_out[0] = bound
                free(arena)
                if cert <= certified_tolerance:
                    return 1
                return 2
            iterations_out[0] = iteration
            evaluations_out[0] = evaluations
            subproblem_iterations_out[0] = sub_iterations
            bound_out[0] = bound
            free(arena)
            return 4

        decrease = current_nll[0] - trial_nll
        if bound <= accuracy_floor and decrease <= bound:
            carried = bound - (decrease if decrease > 0.0 else 0.0)
            if carried < 0.0:
                carried = 0.0
        else:
            carried = INFINITY
        memcpy(theta, trial_theta, n * sizeof(double))
        memcpy(blocks, trial_blocks, qtot * sizeof(double))
        memcpy(gradient, trial_gradient, n * sizeof(double))
        memcpy(hessian, trial_hessian, n2 * sizeof(double))
        objective.accept(objective.ctx)
        current_smallest[0] = trial_smallest
        current_nll[0] = trial_nll

    cert = bound if bound < carried else carried
    if cert <= accuracy_floor:
        bound = cert
        iterations_out[0] = max_iterations
        evaluations_out[0] = evaluations
        subproblem_iterations_out[0] = sub_iterations
        bound_out[0] = bound
        free(arena)
        if cert <= certified_tolerance:
            return 1
        return 2
    iterations_out[0] = max_iterations
    evaluations_out[0] = evaluations
    subproblem_iterations_out[0] = sub_iterations
    bound_out[0] = bound
    free(arena)
    return 5
