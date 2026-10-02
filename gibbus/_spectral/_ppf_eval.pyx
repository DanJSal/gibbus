# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""Compiled evaluator for adaptive spectral PPF representations.

Construction is handled in Python.  This module owns the hot interior query path:
probability -> logit coordinate -> inverse panel lookup -> local Chebyshev
inverse -> compact support coordinate z.

The fitted inverse panels cover the numerically useful interior probability
range.  More extreme probabilities are handled entirely inside this compiled
module by monotone bisection of the packed spectral CDF in compact ``z`` space;
there is no Python/SciPy tail fallback and no separate asymptotic tail model.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport hypot, isnan, log, log1p
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free

cnp.import_array()

cdef extern from * nogil:
    """
    #include <stddef.h>
    #include <math.h>

    #if defined(_MSC_VER)
      #define GIBBUS_RESTRICT __restrict
      #define GIBBUS_VECTOR_HINT __pragma(loop(ivdep))
    #elif defined(__clang__)
      #define GIBBUS_RESTRICT __restrict__
      #define GIBBUS_VECTOR_HINT _Pragma("clang loop vectorize(enable)")
    #elif defined(__GNUC__)
      #define GIBBUS_RESTRICT __restrict__
      #define GIBBUS_VECTOR_HINT _Pragma("GCC ivdep")
    #else
      #define GIBBUS_RESTRICT
      #define GIBBUS_VECTOR_HINT
    #endif

    static void gibbus_ppf_cheb_batch(
        const double * GIBBUS_RESTRICT u,
        const size_t * GIBBUS_RESTRICT index,
        size_t n,
        const double * GIBBUS_RESTRICT c,
        int ncoeff,
        double za,
        double zb,
        double * GIBBUS_RESTRICT out,
        double * GIBBUS_RESTRICT work0,
        double * GIBBUS_RESTRICT work1,
        double * GIBBUS_RESTRICT work2)
    {
        size_t i;
        int k;
        const double zmid = 0.5 * (za + zb);
        const double zscale = 0.5 * (zb - za);
        double *b0 = work0;
        double *b1 = work1;
        double *b2 = work2;
        double *tmp;

        GIBBUS_VECTOR_HINT
        for (i = 0; i < n; ++i) {
            b1[i] = 0.0;
            b2[i] = 0.0;
        }
        for (k = ncoeff - 1; k > 0; --k) {
            const double ck = c[k];
            GIBBUS_VECTOR_HINT
            for (i = 0; i < n; ++i) {
                b0[i] = 2.0 * u[i] * b1[i] - b2[i] + ck;
            }
            tmp = b2; b2 = b1; b1 = b0; b0 = tmp;
        }
        GIBBUS_VECTOR_HINT
        for (i = 0; i < n; ++i) {
            double v = u[i] * b1[i] - b2[i] + c[0];
            double z;
            if (v <= -1.0) z = za;
            else if (v >= 1.0) z = zb;
            else z = zmid + zscale * v;
            b0[i] = z;
        }
        for (i = 0; i < n; ++i) {
            out[index[i]] = b0[i];
        }
    }

    static void gibbus_ppf_cheb_batch_contiguous(
        const double * GIBBUS_RESTRICT u,
        size_t n,
        const double * GIBBUS_RESTRICT c,
        int ncoeff,
        double za,
        double zb,
        double * GIBBUS_RESTRICT out,
        double * GIBBUS_RESTRICT work0,
        double * GIBBUS_RESTRICT work1,
        double * GIBBUS_RESTRICT work2)
    {
        size_t i;
        int k;
        const double zmid = 0.5 * (za + zb);
        const double zscale = 0.5 * (zb - za);
        double *b0 = work0;
        double *b1 = work1;
        double *b2 = work2;
        double *tmp;

        GIBBUS_VECTOR_HINT
        for (i = 0; i < n; ++i) {
            b1[i] = 0.0;
            b2[i] = 0.0;
        }
        for (k = ncoeff - 1; k > 0; --k) {
            const double ck = c[k];
            GIBBUS_VECTOR_HINT
            for (i = 0; i < n; ++i) {
                b0[i] = 2.0 * u[i] * b1[i] - b2[i] + ck;
            }
            tmp = b2; b2 = b1; b1 = b0; b0 = tmp;
        }
        GIBBUS_VECTOR_HINT
        for (i = 0; i < n; ++i) {
            double v = u[i] * b1[i] - b2[i] + c[0];
            double z;
            if (v <= -1.0) z = za;
            else if (v >= 1.0) z = zb;
            else z = zmid + zscale * v;
            out[i] = z;
        }
    }
    """
    void gibbus_ppf_cheb_batch(
        const double* u,
        const size_t* index,
        size_t n,
        const double* c,
        int ncoeff,
        double za,
        double zb,
        double* out,
        double* work0,
        double* work1,
        double* work2,
    ) noexcept
    void gibbus_ppf_cheb_batch_contiguous(
        const double* u,
        size_t n,
        const double* c,
        int ncoeff,
        double za,
        double zb,
        double* out,
        double* work0,
        double* work1,
        double* work2,
    ) noexcept


cdef inline double _cheb17(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(16, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]

cdef inline double _cheb25(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(24, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]

cdef inline double _cheb33(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(32, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]

cdef inline double _cheb_generic(const double* c, int ncoeff, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(ncoeff - 1, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]


cdef class SpectralPPFEvaluator:
    """Compiled bulk evaluator for the full compact-coordinate PPF."""

    cdef double pmin, pmax
    cdef int map_kind
    cdef double map_L, map_U, map_center, map_scale
    cdef Py_ssize_t npanels, stride

    cdef cnp.ndarray _breaks_arr
    cdef cnp.ndarray _zbreaks_arr
    cdef cnp.ndarray _coeff_arr
    cdef cnp.ndarray _ncoeff_arr

    cdef Py_ssize_t cdf_npanels, cdf_stride
    cdef cnp.ndarray _cdf_breaks_arr
    cdef cnp.ndarray _cdf_offsets_arr
    cdef cnp.ndarray _cdf_coeff_arr
    cdef cnp.ndarray _cdf_ncoeff_arr

    cdef const double* breaks
    cdef const double* zbreaks
    cdef const double* coeff
    cdef const int32_t* ncoeff
    cdef const double* cdf_breaks
    cdef const double* cdf_offsets
    cdef const double* cdf_coeff
    cdef const int32_t* cdf_ncoeff

    def __cinit__(
        self,
        double pmin,
        double pmax,
        int map_kind,
        double map_L,
        double map_U,
        double map_center,
        double map_scale,
        object breaks_r,
        object breaks_z,
        object coeffs,
        object ncoeff,
        object cdf_breaks,
        object cdf_offsets,
        object cdf_coeffs,
        object cdf_ncoeff,
    ):
        """Bind the packed inverse panels and source CDF for query time.

        The arrays are stored as contiguous attributes and raw pointers
        are cached from them; the instance keeps the arrays alive, so the
        pointers stay valid for its lifetime.

        Parameters
        ----------
        pmin, pmax : float
            Probability range covered by the fitted inverse panels.
            Outside it, quantiles are found by bisecting the source CDF.
        map_kind : int
            Support type of the underlying distribution: 0 finite, 1
            lower-bounded, 2 upper-bounded, 3 unbounded, 4 lower-bounded
            centered near the density, 5 upper-bounded centered near the
            density.
        map_L, map_U : float
            Support endpoints in physical coordinates.
        map_center, map_scale : float
            Affine parameters of the compact-to-physical map.
        breaks_r : array_like, shape (npanels + 1,)
            Inverse-panel boundaries in logit coordinates, ascending.
        breaks_z : array_like, shape (npanels + 1,)
            The same boundaries in compact coordinates.
        coeffs : array_like, shape (npanels, stride)
            Chebyshev coefficients of the inverse, one row per panel.
        ncoeff : array_like, shape (npanels,)
            Live coefficient count per inverse panel.
        cdf_breaks : array_like, shape (cdf_npanels + 1,)
            Source-CDF panel boundaries in compact coordinates.
        cdf_offsets : array_like, shape (cdf_npanels,)
            CDF value at each source panel's left edge.
        cdf_coeffs : array_like, shape (cdf_npanels, stride)
            Chebyshev coefficients of the source CDF.
        cdf_ncoeff : array_like, shape (cdf_npanels,)
            Live coefficient count per source panel.

        Raises
        ------
        ValueError
            If either coefficient array is not 2-D, if either panel set
            is empty, if any panel arrays disagree on length, if *map_kind*
            is outside 0..5, or if a live coefficient count is outside its
            row stride.
        """
        cdef Py_ssize_t j

        if map_kind < 0 or map_kind > 5:
            raise ValueError("map_kind must be an integer in 0..5")
        self.pmin = pmin
        self.pmax = pmax
        self.map_kind = map_kind
        self.map_L = map_L
        self.map_U = map_U
        self.map_center = map_center
        self.map_scale = map_scale
        self._breaks_arr = np.ascontiguousarray(breaks_r, dtype=np.float64)
        self._zbreaks_arr = np.ascontiguousarray(breaks_z, dtype=np.float64)
        self._coeff_arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        self._ncoeff_arr = np.ascontiguousarray(ncoeff, dtype=np.int32)
        self._cdf_breaks_arr = np.ascontiguousarray(cdf_breaks, dtype=np.float64)
        self._cdf_offsets_arr = np.ascontiguousarray(cdf_offsets, dtype=np.float64)
        self._cdf_coeff_arr = np.ascontiguousarray(cdf_coeffs, dtype=np.float64)
        self._cdf_ncoeff_arr = np.ascontiguousarray(cdf_ncoeff, dtype=np.int32)

        if self._coeff_arr.ndim != 2:
            raise ValueError("coeffs must be a 2-D array")
        self.npanels = self._coeff_arr.shape[0]
        self.stride = self._coeff_arr.shape[1]
        if self.npanels <= 0:
            raise ValueError("at least one panel is required")
        if self.stride <= 0:
            raise ValueError("coefficient stride must be positive")
        if self._breaks_arr.size != self.npanels + 1:
            raise ValueError("breaks_r must contain npanels + 1 entries")
        if self._zbreaks_arr.size != self.npanels + 1:
            raise ValueError("breaks_z must contain npanels + 1 entries")
        if self._ncoeff_arr.size != self.npanels:
            raise ValueError("ncoeff must contain npanels entries")
        for j in range(self.npanels):
            if self._ncoeff_arr[j] < 1 or self._ncoeff_arr[j] > self.stride:
                raise ValueError("ncoeff entries must be in 1..stride")
        if self._cdf_coeff_arr.ndim != 2:
            raise ValueError("cdf_coeffs must be a 2-D array")
        self.cdf_npanels = self._cdf_coeff_arr.shape[0]
        self.cdf_stride = self._cdf_coeff_arr.shape[1]
        if self.cdf_npanels <= 0:
            raise ValueError("at least one CDF panel is required")
        if self.cdf_stride <= 0:
            raise ValueError("CDF coefficient stride must be positive")
        if self._cdf_breaks_arr.size != self.cdf_npanels + 1:
            raise ValueError("cdf_breaks must contain cdf_npanels + 1 entries")
        if (self._cdf_offsets_arr.size != self.cdf_npanels or
                self._cdf_ncoeff_arr.size != self.cdf_npanels):
            raise ValueError("CDF panel arrays must all have cdf_npanels entries")
        for j in range(self.cdf_npanels):
            if self._cdf_ncoeff_arr[j] < 1 or self._cdf_ncoeff_arr[j] > self.cdf_stride:
                raise ValueError("cdf_ncoeff entries must be in 1..cdf_stride")

        self.breaks = <const double*> self._breaks_arr.data
        self.zbreaks = <const double*> self._zbreaks_arr.data
        self.coeff = <const double*> self._coeff_arr.data
        self.ncoeff = <const int32_t*> self._ncoeff_arr.data
        self.cdf_breaks = <const double*> self._cdf_breaks_arr.data
        self.cdf_offsets = <const double*> self._cdf_offsets_arr.data
        self.cdf_coeff = <const double*> self._cdf_coeff_arr.data
        self.cdf_ncoeff = <const int32_t*> self._cdf_ncoeff_arr.data

    cdef inline Py_ssize_t _panel_index(self, double r) noexcept nogil:
        """Index of the inverse panel containing logit coordinate *r*."""
        cdef Py_ssize_t lo = 0
        cdef Py_ssize_t hi = self.npanels
        cdef Py_ssize_t mid
        while lo < hi:
            mid = (lo + hi) >> 1
            if r < self.breaks[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        if lo >= self.npanels:
            return self.npanels - 1
        return lo

    cdef inline Py_ssize_t _cdf_panel_index(self, double z) noexcept nogil:
        """Index of the source-CDF panel containing compact coordinate *z*."""
        cdef Py_ssize_t lo = 0
        cdef Py_ssize_t hi = self.cdf_npanels
        cdef Py_ssize_t mid
        while lo < hi:
            mid = (lo + hi) >> 1
            if z < self.cdf_breaks[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        if lo >= self.cdf_npanels:
            return self.cdf_npanels - 1
        return lo

    cdef inline double _cdf_z_one(self, double z) noexcept nogil:
        """Evaluate the packed source CDF at compact coordinate *z*."""
        cdef Py_ssize_t j
        cdef double u, val
        cdef const double* c
        cdef int nc
        if z <= -1.0:
            return 0.0
        if z >= 1.0:
            return 1.0
        j = self._cdf_panel_index(z)
        u = (2.0 * z - (self.cdf_breaks[j] + self.cdf_breaks[j + 1])) / (
            self.cdf_breaks[j + 1] - self.cdf_breaks[j]
        )
        c = self.cdf_coeff + j * self.cdf_stride
        nc = self.cdf_ncoeff[j]
        val = _cheb_generic(c, nc, u) + self.cdf_offsets[j]
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    cdef inline double _eval_interior_one(self, double p) noexcept nogil:
        """Quantile for a probability inside the fitted interior range."""
        cdef double r, u, v, z
        cdef Py_ssize_t j
        cdef const double* c
        cdef int nc
        r = log(p) - log1p(-p)
        j = self._panel_index(r)
        u = (2.0 * r - (self.breaks[j] + self.breaks[j + 1])) / (
            self.breaks[j + 1] - self.breaks[j]
        )
        c = self.coeff + j * self.stride
        nc = self.ncoeff[j]
        if nc == 17:
            v = _cheb17(c, u)
        elif nc == 25:
            v = _cheb25(c, u)
        elif nc == 33:
            v = _cheb33(c, u)
        else:
            v = _cheb_generic(c, nc, u)
        if v <= -1.0:
            z = self.zbreaks[j]
        elif v >= 1.0:
            z = self.zbreaks[j + 1]
        else:
            z = 0.5 * (
                (self.zbreaks[j + 1] - self.zbreaks[j]) * v
                + (self.zbreaks[j] + self.zbreaks[j + 1])
            )
        return z

    cdef inline double _invert_tail(self, double p) noexcept nogil:
        """Quantile outside the fitted range, by bisecting the source CDF.

        Inverts the packed spectral CDF in compact ``z`` by monotone
        bisection.  There is no Python or SciPy fallback behind this:
        extreme probabilities are resolved entirely here.
        """
        cdef double a = -1.0
        cdef double b = 1.0
        cdef double m, fm, fa, fb
        cdef int _it
        fa = -p
        fb = 1.0 - p
        # The endpoint values are exact by construction.  Bisection over the
        # full compact interval is deliberately used rather than an approximate
        # PPF-panel endpoint, so a spectral inverse residual can never destroy
        # the bracket for probabilities just outside [pmin, pmax].
        for _it in range(80):
            m = 0.5 * (a + b)
            if m == a or m == b:
                break
            fm = self._cdf_z_one(m) - p
            if fm <= 0.0:
                a = m
                fa = fm
            else:
                b = m
                fb = fm
        if -fa <= fb:
            return a
        return b

    cdef inline double _eval_one(self, double p) noexcept nogil:
        """Quantile at a single probability, interior or tail."""
        if isnan(p):
            return p
        if p == 0.0:
            return -1.0
        if p == 1.0:
            return 1.0
        if p < self.pmin or p > self.pmax:
            return self._invert_tail(p)
        return self._eval_interior_one(p)

    cdef inline double _x_from_z(self, double z) noexcept nogil:
        """Map a compact coordinate back to physical coordinates."""
        cdef double y_edge, t_edge, t
        if self.map_kind == 0:
            if z <= -1.0:
                return self.map_L
            if z >= 1.0:
                return self.map_U
            return self.map_center + self.map_scale * z
        if self.map_kind == 1:
            return self.map_L + self.map_scale * (1.0 + z) / (1.0 - z)
        if self.map_kind == 2:
            return self.map_U - self.map_scale * (1.0 - z) / (1.0 + z)
        if self.map_kind == 4:
            if z <= -1.0:
                return self.map_L
            y_edge = (self.map_L - self.map_center) / self.map_scale
            t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
            t = t_edge + 0.5 * (z + 1.0) * (1.0 - t_edge)
            return self.map_center + (2.0 * self.map_scale * t) / (1.0 - t * t)
        if self.map_kind == 5:
            if z >= 1.0:
                return self.map_U
            y_edge = (self.map_U - self.map_center) / self.map_scale
            t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
            t = -1.0 + 0.5 * (z + 1.0) * (t_edge + 1.0)
            return self.map_center + (2.0 * self.map_scale * t) / (1.0 - t * t)
        return self.map_center + (2.0 * self.map_scale * z) / (1.0 - z * z)

    cdef void _map_many_to_x(self, double* out, Py_ssize_t n) noexcept nogil:
        """Map an array of compact coordinates to physical, in place."""
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self._x_from_z(out[i])

    cdef void _eval_many(
        self, const double* p, double* out, Py_ssize_t n
    ) noexcept nogil:
        """Scalar-per-observation loop; the fallback for small arrays."""
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self._eval_one(p[i])

    cdef bint _is_nondecreasing(self, const double* p, Py_ssize_t n) noexcept nogil:
        """Whether *p* is sorted ascending and NaN-free, so runs are contiguous."""
        cdef Py_ssize_t i
        cdef double prev, cur
        if n <= 1:
            return True
        prev = p[0]
        if isnan(prev):
            return False
        for i in range(1, n):
            cur = p[i]
            if isnan(cur) or cur < prev:
                return False
            prev = cur
        return True

    cdef void _eval_many_runs(
        self, const double* p, double* out, Py_ssize_t n
    ) noexcept nogil:
        """SIMD-oriented evaluator for one-panel or sorted inputs; no scatter."""
        cdef Py_ssize_t cap = 16384
        cdef double* u_buf = NULL
        cdef double* work0 = NULL
        cdef double* work1 = NULL
        cdef double* work2 = NULL
        cdef Py_ssize_t i = 0, start, m, j
        cdef double pp, r
        cdef const double* c
        cdef int nc

        if n <= 0:
            return
        if n < cap:
            cap = n
        u_buf = <double*> malloc(cap * sizeof(double))
        work0 = <double*> malloc(cap * sizeof(double))
        work1 = <double*> malloc(cap * sizeof(double))
        work2 = <double*> malloc(cap * sizeof(double))
        if u_buf == NULL or work0 == NULL or work1 == NULL or work2 == NULL:
            if u_buf != NULL:
                free(u_buf)
            if work0 != NULL:
                free(work0)
            if work1 != NULL:
                free(work1)
            if work2 != NULL:
                free(work2)
            self._eval_many(p, out, n)
            return

        while i < n:
            pp = p[i]
            if isnan(pp):
                out[i] = pp
                i += 1
                continue
            if pp == 0.0:
                out[i] = -1.0
                i += 1
                continue
            if pp == 1.0:
                out[i] = 1.0
                i += 1
                continue
            if pp < self.pmin or pp > self.pmax:
                out[i] = self._invert_tail(pp)
                i += 1
                continue

            r = log(pp) - log1p(-pp)
            j = self._panel_index(r)
            start = i
            m = 0
            while i < n and m < cap:
                pp = p[i]
                if (
                    isnan(pp)
                    or pp == 0.0
                    or pp == 1.0
                    or pp < self.pmin
                    or pp > self.pmax
                ):
                    break
                r = log(pp) - log1p(-pp)
                if self.npanels > 1 and (
                    r < self.breaks[j]
                    or (r >= self.breaks[j + 1] and j < self.npanels - 1)
                ):
                    break
                u_buf[m] = (2.0 * r - (self.breaks[j] + self.breaks[j + 1])) / (
                    self.breaks[j + 1] - self.breaks[j]
                )
                m += 1
                i += 1
            if m > 0:
                c = self.coeff + j * self.stride
                nc = self.ncoeff[j]
                gibbus_ppf_cheb_batch_contiguous(
                    u_buf,
                    <size_t> m,
                    c,
                    nc,
                    self.zbreaks[j],
                    self.zbreaks[j + 1],
                    out + start,
                    work0,
                    work1,
                    work2,
                )
            else:
                continue

        free(u_buf)
        free(work0)
        free(work1)
        free(work2)

    cdef void _eval_many_simd(
        self, const double* p, double* out, Py_ssize_t n
    ) noexcept nogil:
        """Cache-blocked panel-bucketed evaluator with transposed Clenshaw."""
        cdef Py_ssize_t block_cap = 16384
        cdef int32_t* panel_of = NULL
        cdef double* u_orig = NULL
        cdef double* u_bucket = NULL
        cdef size_t* index_bucket = NULL
        cdef Py_ssize_t* counts = NULL
        cdef Py_ssize_t* starts = NULL
        cdef Py_ssize_t* pos = NULL
        cdef double* work0 = NULL
        cdef double* work1 = NULL
        cdef double* work2 = NULL
        cdef Py_ssize_t base, m, i, j, q
        cdef double pp, r, u
        cdef const double* c
        cdef int nc

        if n <= 0:
            return
        if n < block_cap:
            block_cap = n
        panel_of = <int32_t*> malloc(block_cap * sizeof(int32_t))
        u_orig = <double*> malloc(block_cap * sizeof(double))
        u_bucket = <double*> malloc(block_cap * sizeof(double))
        index_bucket = <size_t*> malloc(block_cap * sizeof(size_t))
        counts = <Py_ssize_t*> malloc(self.npanels * sizeof(Py_ssize_t))
        starts = <Py_ssize_t*> malloc((self.npanels + 1) * sizeof(Py_ssize_t))
        pos = <Py_ssize_t*> malloc(self.npanels * sizeof(Py_ssize_t))
        work0 = <double*> malloc(block_cap * sizeof(double))
        work1 = <double*> malloc(block_cap * sizeof(double))
        work2 = <double*> malloc(block_cap * sizeof(double))
        if (
            panel_of == NULL
            or u_orig == NULL
            or u_bucket == NULL
            or index_bucket == NULL
            or counts == NULL
            or starts == NULL
            or pos == NULL
            or work0 == NULL
            or work1 == NULL
            or work2 == NULL
        ):
            if panel_of != NULL:
                free(panel_of)
            if u_orig != NULL:
                free(u_orig)
            if u_bucket != NULL:
                free(u_bucket)
            if index_bucket != NULL:
                free(index_bucket)
            if counts != NULL:
                free(counts)
            if starts != NULL:
                free(starts)
            if pos != NULL:
                free(pos)
            if work0 != NULL:
                free(work0)
            if work1 != NULL:
                free(work1)
            if work2 != NULL:
                free(work2)
            self._eval_many(p, out, n)
            return

        base = 0
        while base < n:
            m = n - base
            if m > block_cap:
                m = block_cap
            for j in range(self.npanels):
                counts[j] = 0

            for i in range(m):
                pp = p[base + i]
                panel_of[i] = -1
                if isnan(pp):
                    out[base + i] = pp
                    continue
                if pp == 0.0:
                    out[base + i] = -1.0
                    continue
                if pp == 1.0:
                    out[base + i] = 1.0
                    continue
                if pp < self.pmin or pp > self.pmax:
                    out[base + i] = self._invert_tail(pp)
                    continue
                r = log(pp) - log1p(-pp)
                j = self._panel_index(r)
                u = (2.0 * r - (self.breaks[j] + self.breaks[j + 1])) / (
                    self.breaks[j + 1] - self.breaks[j]
                )
                panel_of[i] = <int32_t> j
                u_orig[i] = u
                counts[j] += 1

            starts[0] = 0
            for j in range(self.npanels):
                starts[j + 1] = starts[j] + counts[j]
                pos[j] = starts[j]

            for i in range(m):
                j = panel_of[i]
                if j >= 0:
                    q = pos[j]
                    pos[j] = q + 1
                    u_bucket[q] = u_orig[i]
                    index_bucket[q] = <size_t> i

            for j in range(self.npanels):
                if counts[j] == 0:
                    continue
                c = self.coeff + j * self.stride
                nc = self.ncoeff[j]
                gibbus_ppf_cheb_batch(
                    u_bucket + starts[j],
                    index_bucket + starts[j],
                    <size_t> counts[j],
                    c,
                    nc,
                    self.zbreaks[j],
                    self.zbreaks[j + 1],
                    out + base,
                    work0,
                    work1,
                    work2,
                )
            base += m

        free(panel_of)
        free(u_orig)
        free(u_bucket)
        free(index_bucket)
        free(counts)
        free(starts)
        free(pos)
        free(work0)
        free(work1)
        free(work2)

    cdef object _call_mode(self, object p, bint use_simd, bint map_to_x):
        """Dispatch to the vectorized or scalar loop and restore *p*'s shape.

        *map_to_x* additionally maps the compact result back to physical
        coordinates.
        """
        cdef object arr_obj = np.asarray(p, dtype=np.float64)
        cdef cnp.ndarray arr = arr_obj
        if np.any((arr_obj < 0.0) | (arr_obj > 1.0)):
            raise ValueError("ppf is defined for p in [0, 1]")
        cdef bint scalar = arr.ndim == 0
        cdef object shape = arr_obj.shape
        cdef cnp.ndarray flat = np.ascontiguousarray(arr_obj).reshape(-1)
        cdef cnp.ndarray out = np.empty(flat.size, dtype=np.float64)
        cdef Py_ssize_t n = flat.size
        cdef const double* pp = <const double*> flat.data
        cdef double* op = <double*> out.data
        with nogil:
            if use_simd and n >= 256:
                if self.npanels == 1 or self._is_nondecreasing(pp, n):
                    self._eval_many_runs(pp, op, n)
                else:
                    self._eval_many_simd(pp, op, n)
            else:
                self._eval_many(pp, op, n)
            if map_to_x:
                self._map_many_to_x(op, n)
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def eval_x_scalar(self, object p):
        """Quantiles in physical coordinates, scalar loop.

        Parameters
        ----------
        p : array_like or float
            Probability value or values in ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
            Quantiles in physical coordinates, matching the input shape.
        """
        return self._call_mode(p, False, True)

    def eval_x(self, object p):
        """Quantiles in physical coordinates, vectorized loop.

        Parameters
        ----------
        p : array_like or float
            Probability value or values in ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
            Quantiles in physical coordinates, matching the input shape.
        """
        return self._call_mode(p, True, True)
