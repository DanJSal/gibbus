# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""Compiled evaluator for adaptive piecewise spectral CDF representations.

Construction is handled in Python.  This module owns only the hot query path:
physical x -> compact coordinate -> panel lookup -> local Chebyshev CDF.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, hypot, isinf, isnan
from libc.stdint cimport int32_t
from libc.float cimport DBL_EPSILON
from libc.stdlib cimport malloc, free

cnp.import_array()

# The recurrence is deliberately transposed relative to the scalar evaluator:
# polynomial order is the outer loop and observations are the inner loop.
# Points passed here all share one panel (hence one coefficient row).  The
# contiguous inner loop is explicitly dependency-free.  Restrict-qualified
# pointers plus compiler-specific vectorization hints make that contract visible
# to GCC, Clang, and MSVC so their auto-vectorizers can use SIMD across observations.
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

    static void gibbus_cheb_batch(
        const double * GIBBUS_RESTRICT u,
        const size_t * GIBBUS_RESTRICT index,
        size_t n,
        const double * GIBBUS_RESTRICT c,
        int ncoeff,
        double offset,
        double * GIBBUS_RESTRICT out,
        double * GIBBUS_RESTRICT work0,
        double * GIBBUS_RESTRICT work1,
        double * GIBBUS_RESTRICT work2)
    {
        size_t i;
        int k;
        double *b0 = work0;
        double *b1 = work1;
        double *b2 = work2;
        double *tmp;

        /* b1=b2=0 for Clenshaw start. */
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
            tmp = b2;
            b2 = b1;
            b1 = b0;
            b0 = tmp;
        }

        /* Reuse the inactive scratch buffer for contiguous final values. */
        GIBBUS_VECTOR_HINT
        for (i = 0; i < n; ++i) {
            b0[i] = u[i] * b1[i] - b2[i] + c[0] + offset;
        }

        /* Scatter is cheap relative to the recurrence and need not vectorize. */
        for (i = 0; i < n; ++i) {
            double v = b0[i];
            if (v <= 0.0) v = 0.0;
            else if (v >= 1.0) v = 1.0;
            out[index[i]] = v;
        }
    }

    static void gibbus_cheb_batch_contiguous(
        const double * GIBBUS_RESTRICT u,
        size_t n,
        const double * GIBBUS_RESTRICT c,
        int ncoeff,
        double offset,
        double * GIBBUS_RESTRICT out,
        double * GIBBUS_RESTRICT work0,
        double * GIBBUS_RESTRICT work1,
        double * GIBBUS_RESTRICT work2)
    {
        size_t i;
        int k;
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
            double v = u[i] * b1[i] - b2[i] + c[0] + offset;
            if (v <= 0.0) v = 0.0;
            else if (v >= 1.0) v = 1.0;
            out[i] = v;
        }
    }
    """
    void gibbus_cheb_batch(
        const double* u,
        const size_t* index,
        size_t n,
        const double* c,
        int ncoeff,
        double offset,
        double* out,
        double* work0,
        double* work1,
        double* work2,
    ) noexcept
    void gibbus_cheb_batch_contiguous(
        const double* u,
        size_t n,
        const double* c,
        int ncoeff,
        double offset,
        double* out,
        double* work0,
        double* work1,
        double* work2,
    ) noexcept

cdef inline double _cheb18(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(17, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]

cdef inline double _cheb26(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(25, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]

cdef inline double _cheb34(const double* c, double u) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(33, 0, -1):
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


cdef inline void _cheb_value_derivative(
    const double* c, int ncoeff, double u, double* value, double* derivative
) noexcept nogil:
    """Evaluate one Chebyshev series and its derivative together."""
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef double d0, d1 = 0.0, d2 = 0.0
    cdef int k
    for k in range(ncoeff - 1, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        d0 = 2.0 * b1 + 2.0 * u * d1 - d2
        b2 = b1
        b1 = b0
        d2 = d1
        d1 = d0
    value[0] = u * b1 - b2 + c[0]
    derivative[0] = b1 + u * d1 - d2


cdef class SpectralEvaluator:
    """Compiled bulk evaluator for a packed spectral CDF."""

    cdef int kind
    cdef double L, U, center, scale
    cdef Py_ssize_t npanels, stride

    cdef cnp.ndarray _breaks_arr
    cdef cnp.ndarray _offset_arr
    cdef cnp.ndarray _coeff_arr
    cdef cnp.ndarray _ncoeff_arr

    cdef const double* breaks
    cdef const double* offset
    cdef const double* coeff
    cdef const int32_t* ncoeff

    def __cinit__(
        self,
        int kind,
        double L,
        double U,
        double center,
        double scale,
        object breaks,
        object offsets,
        object coeffs,
        object ncoeff,
    ):
        """Bind a packed panel representation for query-time evaluation.

        The arrays are stored as contiguous attributes and raw pointers
        are cached from them; the instance keeps the arrays alive, so the
        pointers stay valid for its lifetime.

        Parameters
        ----------
        kind : int
            Support type: 0 finite, 1 lower-bounded, 2 upper-bounded,
            3 unbounded, 4 lower-bounded centered near the density, 5
            upper-bounded centered near the density. Selects the
            physical-to-compact map.
        L, U : float
            Support endpoints in physical coordinates.  The unused side
            is ignored for half-lines.
        center, scale : float
            Affine parameters of the physical-to-compact map.
        breaks : array_like, shape (npanels + 1,)
            Panel boundaries in compact coordinates, ascending.
        offsets : array_like, shape (npanels,)
            CDF value at each panel's left edge.
        coeffs : array_like, shape (npanels, stride)
            Chebyshev coefficients, one row per panel, right-padded.
        ncoeff : array_like, shape (npanels,)
            Live coefficient count per panel.

        Raises
        ------
        ValueError
            If *coeffs* is not 2-D, if there are no panels, or if the
            panel arrays disagree on length; if *kind* is outside 0..5;
            or if any live coefficient count is outside 1..stride.
        """
        cdef Py_ssize_t j

        if kind < 0 or kind > 5:
            raise ValueError("kind must be an integer in 0..5")
        self.kind = kind
        self.L = L
        self.U = U
        self.center = center
        self.scale = scale

        self._breaks_arr = np.ascontiguousarray(breaks, dtype=np.float64)
        self._offset_arr = np.ascontiguousarray(offsets, dtype=np.float64)
        self._coeff_arr = np.ascontiguousarray(coeffs, dtype=np.float64)
        self._ncoeff_arr = np.ascontiguousarray(ncoeff, dtype=np.int32)

        if self._coeff_arr.ndim != 2:
            raise ValueError("coeffs must be a 2-D array")
        self.npanels = self._coeff_arr.shape[0]
        self.stride = self._coeff_arr.shape[1]
        if self.npanels <= 0:
            raise ValueError("at least one panel is required")
        if self.stride <= 0:
            raise ValueError("coefficient stride must be positive")
        if self._breaks_arr.size != self.npanels + 1:
            raise ValueError("breaks must contain npanels + 1 entries")
        if (self._offset_arr.size != self.npanels or
                self._ncoeff_arr.size != self.npanels):
            raise ValueError("panel arrays must all have npanels entries")
        for j in range(self.npanels):
            if self._ncoeff_arr[j] < 1 or self._ncoeff_arr[j] > self.stride:
                raise ValueError("ncoeff entries must be in 1..stride")

        self.breaks = <const double*> self._breaks_arr.data
        self.offset = <const double*> self._offset_arr.data
        self.coeff = <const double*> self._coeff_arr.data
        self.ncoeff = <const int32_t*> self._ncoeff_arr.data

    cdef inline double _z_from_x(self, double x) noexcept nogil:
        """Map a physical coordinate into the compact domain ``[-1, 1]``."""
        cdef double y, t, y_edge, t_edge
        if self.kind == 0:  # finite
            return (x - self.center) / self.scale
        if self.kind == 1:  # lower bounded
            y = (x - self.L) / self.scale
            return (y - 1.0) / (y + 1.0)
        if self.kind == 2:  # upper bounded
            y = (self.U - x) / self.scale
            return (1.0 - y) / (1.0 + y)
        if self.kind == 4:  # lower bounded, centered near the density
            y = (x - self.center) / self.scale
            t = y / (1.0 + hypot(1.0, y))
            y_edge = (self.L - self.center) / self.scale
            t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
            return 2.0 * (t - t_edge) / (1.0 - t_edge) - 1.0
        if self.kind == 5:  # upper bounded, centered near the density
            y = (x - self.center) / self.scale
            t = y / (1.0 + hypot(1.0, y))
            y_edge = (self.U - self.center) / self.scale
            t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
            return 2.0 * (t + 1.0) / (t_edge + 1.0) - 1.0
        # real line
        y = (x - self.center) / self.scale
        return y / (1.0 + hypot(1.0, y))

    cdef inline Py_ssize_t _panel_index(self, double z) noexcept nogil:
        """Return the index of the panel containing compact coordinate *z*."""
        cdef Py_ssize_t lo = 0
        cdef Py_ssize_t hi = self.npanels
        cdef Py_ssize_t mid
        # Equivalent to searchsorted(breaks, z, side='right') - 1,
        # specialized to z strictly inside (-1, 1).
        while lo < hi:
            mid = (lo + hi) >> 1
            if z < self.breaks[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        if lo >= self.npanels:
            return self.npanels - 1
        return lo

    cdef inline double _invert_panel_fraction_one(
        self, Py_ssize_t j, double frac
    ) noexcept nogil:
        """Invert one source panel at a local cumulative-mass fraction.

        A safeguarded Newton step uses the derivative of the same stored
        Chebyshev antiderivative.  The bracketing interval is retained at all
        times, so pathological flat regions fall back to bisection without
        changing the monotone inversion contract.
        """
        cdef double lo = -1.0
        cdef double hi = 1.0
        cdef double u, candidate, value, deriv, f
        cdef double flo, fhi, target, local_mass, scale, tol
        cdef const double* c = self.coeff + j * self.stride
        cdef int nc = self.ncoeff[j]
        cdef int _it

        if frac <= 8.0 * DBL_EPSILON:
            return self.breaks[j]
        if frac >= 1.0 - 8.0 * DBL_EPSILON:
            return self.breaks[j + 1]

        local_mass = _cheb_generic(c, nc, 1.0)
        target = frac * local_mass
        flo = _cheb_generic(c, nc, lo) - target
        fhi = local_mass - target
        u = 2.0 * frac - 1.0
        if u <= lo or u >= hi:
            u = 0.0

        scale = local_mass
        if fabs(target) > scale:
            scale = fabs(target)
        if scale < 1e-300:
            scale = 1e-300
        tol = 8.0 * DBL_EPSILON * scale

        for _it in range(64):
            _cheb_value_derivative(c, nc, u, &value, &deriv)
            f = value - target
            if f <= 0.0:
                lo = u
                flo = f
            else:
                hi = u
                fhi = f

            if fabs(f) <= tol:
                break
            if hi - lo <= 8.0 * DBL_EPSILON * (1.0 + fabs(u)):
                break

            if deriv > 0.0 and not isinf(deriv) and not isnan(deriv):
                candidate = u - f / deriv
                if candidate <= lo or candidate >= hi or isinf(candidate) or isnan(candidate):
                    candidate = 0.5 * (lo + hi)
            else:
                candidate = 0.5 * (lo + hi)
            if candidate == u:
                break
            u = candidate

        # Return the best of the iterate and bracketing endpoints.  This keeps
        # the old endpoint-limit behavior when the local signal is below the
        # representable antiderivative scale.
        _cheb_value_derivative(c, nc, u, &value, &deriv)
        f = value - target
        if fabs(flo) <= fabs(f) and fabs(flo) <= fabs(fhi):
            u = lo
        elif fabs(fhi) < fabs(f):
            u = hi
        return 0.5 * (
            (self.breaks[j + 1] - self.breaks[j]) * u
            + (self.breaks[j] + self.breaks[j + 1])
        )

    cdef void _invert_panel_fraction_many(
        self, Py_ssize_t j, const double* frac, double* out, Py_ssize_t n
    ) noexcept nogil:
        """Invert many local cumulative-mass fractions on one source panel."""
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self._invert_panel_fraction_one(j, frac[i])

    cdef inline double _eval_z_one(self, double z) noexcept nogil:
        """Evaluate the packed CDF at one compact support coordinate."""
        cdef double u, val
        cdef Py_ssize_t j
        cdef const double* c
        cdef int nc

        if isnan(z):
            return z
        if z <= -1.0:
            return 0.0
        if z >= 1.0:
            return 1.0

        j = self._panel_index(z)
        u = (2.0 * z - (self.breaks[j] + self.breaks[j + 1])) / (self.breaks[j + 1] - self.breaks[j])
        c = self.coeff + j * self.stride
        nc = self.ncoeff[j]
        if nc == 18:
            val = _cheb18(c, u)
        elif nc == 26:
            val = _cheb26(c, u)
        elif nc == 34:
            val = _cheb34(c, u)
        else:
            val = _cheb_generic(c, nc, u)
        val += self.offset[j]
        # Guard only against final floating-point roundoff.
        if val <= 0.0:
            return 0.0
        if val >= 1.0:
            return 1.0
        return val

    cdef inline double _eval_one(self, double x) noexcept nogil:
        """Evaluate the CDF at a single physical point."""
        cdef double z

        if isnan(x):
            return x

        if self.kind == 0:
            if x <= self.L:
                return 0.0
            if x >= self.U:
                return 1.0
        elif self.kind == 1 or self.kind == 4:
            if x <= self.L:
                return 0.0
            if isinf(x) and x > 0.0:
                return 1.0
        elif self.kind == 2 or self.kind == 5:
            if x >= self.U:
                return 1.0
            if isinf(x) and x < 0.0:
                return 0.0
        else:
            if isinf(x):
                return 1.0 if x > 0.0 else 0.0

        z = self._z_from_x(x)
        return self._eval_z_one(z)

    cdef void _eval_many(self, const double* x, double* out, Py_ssize_t n) noexcept nogil:
        """Scalar-per-observation loop; the fallback for small arrays."""
        cdef Py_ssize_t i
        for i in range(n):
            out[i] = self._eval_one(x[i])

    cdef inline bint _is_nondecreasing(self, const double* x, Py_ssize_t n) noexcept nogil:
        """Whether *x* is sorted ascending and NaN-free, so runs are contiguous."""
        cdef Py_ssize_t i
        cdef double prev, cur
        if n <= 1:
            return True
        prev = x[0]
        if isnan(prev):
            return False
        for i in range(1, n):
            cur = x[i]
            if isnan(cur) or cur < prev:
                return False
            prev = cur
        return True

    cdef void _eval_many_runs(self, const double* x, double* out, Py_ssize_t n) noexcept nogil:
        """SIMD-oriented evaluator for one-panel or sorted inputs; no scatter."""
        cdef Py_ssize_t cap = 16384
        cdef double* u_buf = NULL
        cdef double* work0 = NULL
        cdef double* work1 = NULL
        cdef double* work2 = NULL
        cdef Py_ssize_t i = 0, start, m, j
        cdef double xx, z
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
            if u_buf != NULL: free(u_buf)
            if work0 != NULL: free(work0)
            if work1 != NULL: free(work1)
            if work2 != NULL: free(work2)
            self._eval_many(x, out, n)
            return

        while i < n:
            xx = x[i]
            if isnan(xx):
                out[i] = xx; i += 1; continue
            if self.kind == 0:
                if xx <= self.L: out[i] = 0.0; i += 1; continue
                if xx >= self.U: out[i] = 1.0; i += 1; continue
            elif self.kind == 1 or self.kind == 4:
                if xx <= self.L: out[i] = 0.0; i += 1; continue
                if isinf(xx) and xx > 0.0: out[i] = 1.0; i += 1; continue
            elif self.kind == 2 or self.kind == 5:
                if xx >= self.U: out[i] = 1.0; i += 1; continue
                if isinf(xx) and xx < 0.0: out[i] = 0.0; i += 1; continue
            else:
                if isinf(xx): out[i] = 1.0 if xx > 0.0 else 0.0; i += 1; continue

            z = self._z_from_x(xx)
            if z <= -1.0: out[i] = 0.0; i += 1; continue
            if z >= 1.0: out[i] = 1.0; i += 1; continue
            j = self._panel_index(z)
            start = i
            m = 0

            # For multiple panels this assumes nondecreasing x/z, so membership
            # in panel j forms one contiguous run. For one panel, ordering is irrelevant.
            while i < n and m < cap:
                xx = x[i]
                if isnan(xx):
                    break
                if self.kind == 0:
                    if xx <= self.L or xx >= self.U: break
                elif self.kind == 1 or self.kind == 4:
                    if xx <= self.L or (isinf(xx) and xx > 0.0): break
                elif self.kind == 2 or self.kind == 5:
                    if xx >= self.U or (isinf(xx) and xx < 0.0): break
                else:
                    if isinf(xx): break
                z = self._z_from_x(xx)
                if z <= -1.0 or z >= 1.0:
                    break
                if self.npanels > 1 and (z < self.breaks[j] or z >= self.breaks[j + 1]):
                    break
                u_buf[m] = (2.0 * z - (self.breaks[j] + self.breaks[j + 1])) / (self.breaks[j + 1] - self.breaks[j])
                m += 1
                i += 1

            if m > 0:
                c = self.coeff + j * self.stride
                nc = self.ncoeff[j]
                gibbus_cheb_batch_contiguous(
                    u_buf, <size_t> m, c, nc, self.offset[j], out + start,
                    work0, work1, work2,
                )
            else:
                # Unreachable in practice: the inner loop re-tests the very
                # point that just passed the outer checks, so it accepts at
                # least one.  Handle it defensively and advance ``i`` explicitly
                # so this branch cannot loop indefinitely if the assumption breaks.
                out[i] = self._eval_one(x[i])
                i += 1

        free(u_buf); free(work0); free(work1); free(work2)

    cdef void _eval_many_simd(self, const double* x, double* out, Py_ssize_t n) noexcept nogil:
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
        cdef double xx, z, u
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
        if (panel_of == NULL or u_orig == NULL or u_bucket == NULL or
                index_bucket == NULL or counts == NULL or starts == NULL or
                pos == NULL or work0 == NULL or work1 == NULL or work2 == NULL):
            if panel_of != NULL: free(panel_of)
            if u_orig != NULL: free(u_orig)
            if u_bucket != NULL: free(u_bucket)
            if index_bucket != NULL: free(index_bucket)
            if counts != NULL: free(counts)
            if starts != NULL: free(starts)
            if pos != NULL: free(pos)
            if work0 != NULL: free(work0)
            if work1 != NULL: free(work1)
            if work2 != NULL: free(work2)
            self._eval_many(x, out, n)
            return

        base = 0
        while base < n:
            m = n - base
            if m > block_cap:
                m = block_cap
            for j in range(self.npanels):
                counts[j] = 0

            # First pass: preserve endpoint/NaN semantics and classify interiors.
            for i in range(m):
                xx = x[base + i]
                panel_of[i] = -1
                if isnan(xx):
                    out[base + i] = xx
                    continue
                if self.kind == 0:
                    if xx <= self.L:
                        out[base + i] = 0.0
                        continue
                    if xx >= self.U:
                        out[base + i] = 1.0
                        continue
                elif self.kind == 1 or self.kind == 4:
                    if xx <= self.L:
                        out[base + i] = 0.0
                        continue
                    if isinf(xx) and xx > 0.0:
                        out[base + i] = 1.0
                        continue
                elif self.kind == 2 or self.kind == 5:
                    if xx >= self.U:
                        out[base + i] = 1.0
                        continue
                    if isinf(xx) and xx < 0.0:
                        out[base + i] = 0.0
                        continue
                else:
                    if isinf(xx):
                        out[base + i] = 1.0 if xx > 0.0 else 0.0
                        continue

                z = self._z_from_x(xx)
                if z <= -1.0:
                    out[base + i] = 0.0
                    continue
                if z >= 1.0:
                    out[base + i] = 1.0
                    continue
                j = self._panel_index(z)
                u = (2.0 * z - (self.breaks[j] + self.breaks[j + 1])) / (self.breaks[j + 1] - self.breaks[j])
                panel_of[i] = <int32_t> j
                u_orig[i] = u
                counts[j] += 1

            starts[0] = 0
            for j in range(self.npanels):
                starts[j + 1] = starts[j] + counts[j]
                pos[j] = starts[j]

            # Stable counting-sort into panel-contiguous buffers.
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
                gibbus_cheb_batch(
                    u_bucket + starts[j], index_bucket + starts[j],
                    <size_t> counts[j], c, nc, self.offset[j], out + base,
                    work0, work1, work2,
                )
            base += m

        free(panel_of); free(u_orig); free(u_bucket); free(index_bucket)
        free(counts); free(starts); free(pos)
        free(work0); free(work1); free(work2)

    cdef object _call_mode(self, object x, bint use_simd):
        """Dispatch to the vectorized or scalar loop and restore *x*'s shape."""
        cdef object arr_obj = np.asarray(x, dtype=np.float64)
        cdef cnp.ndarray arr = arr_obj
        cdef bint scalar = arr.ndim == 0
        cdef object shape = arr_obj.shape
        cdef cnp.ndarray flat = np.ascontiguousarray(arr_obj).reshape(-1)
        cdef cnp.ndarray out = np.empty(flat.size, dtype=np.float64)
        cdef Py_ssize_t n = flat.size
        cdef const double* xp = <const double*> flat.data
        cdef double* op = <double*> out.data
        with nogil:
            if use_simd and n >= 256:
                if self.npanels == 1 or self._is_nondecreasing(xp, n):
                    self._eval_many_runs(xp, op, n)
                else:
                    self._eval_many_simd(xp, op, n)
            else:
                self._eval_many(xp, op, n)
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def invert_panel_fraction(self, int panel, object fraction):
        """Invert one CDF panel at local cumulative-mass fractions.

        Parameters
        ----------
        panel : int
            Source panel index.
        fraction : array_like or float
            Local cumulative-mass fractions in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray or float
            Compact coordinates in the selected source panel.
        """
        cdef object arr_obj
        cdef cnp.ndarray arr
        cdef cnp.ndarray flat
        cdef cnp.ndarray out
        cdef bint scalar
        cdef object shape
        cdef Py_ssize_t n
        cdef const double* fp
        cdef double* op

        if panel < 0 or panel >= self.npanels:
            raise ValueError("panel index is out of range")
        arr_obj = np.asarray(fraction, dtype=np.float64)
        if np.any(~np.isfinite(arr_obj)) or np.any((arr_obj < 0.0) | (arr_obj > 1.0)):
            raise ValueError("fraction must contain finite values in [0, 1]")
        arr = arr_obj
        scalar = arr.ndim == 0
        shape = arr_obj.shape
        flat = np.ascontiguousarray(arr_obj).reshape(-1)
        out = np.empty(flat.size, dtype=np.float64)
        n = flat.size
        fp = <const double*> flat.data
        op = <double*> out.data
        with nogil:
            self._invert_panel_fraction_many(panel, fp, op, n)
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def eval_compact(self, object z):
        """Evaluate the packed CDF directly in compact support coordinates.

        This construction-time entry point avoids mapping compact validation
        nodes back through physical coordinates during inverse-panel fitting.

        Parameters
        ----------
        z : array_like or float
            Compact coordinates. Values at or beyond ``[-1, 1]`` saturate to
            0 or 1; NaN propagates.

        Returns
        -------
        numpy.ndarray or float
            CDF values with the shape of *z*.
        """
        cdef object arr_obj = np.asarray(z, dtype=np.float64)
        cdef cnp.ndarray arr = arr_obj
        cdef bint scalar = arr.ndim == 0
        cdef object shape = arr_obj.shape
        cdef cnp.ndarray flat = np.ascontiguousarray(arr_obj).reshape(-1)
        cdef cnp.ndarray out = np.empty(flat.size, dtype=np.float64)
        cdef Py_ssize_t i, n = flat.size
        cdef const double* zp = <const double*> flat.data
        cdef double* op = <double*> out.data
        with nogil:
            for i in range(n):
                op[i] = self._eval_z_one(zp[i])
        if scalar:
            return float(out[0])
        return out.reshape(shape)

    def eval_scalar(self, object x):
        """Evaluate the CDF with the compiled scalar-per-observation loop.

        This diagnostic entry point bypasses the SIMD-oriented large-array
        dispatch while retaining compiled execution.  It is used to check
        numerical parity and benchmark the transposed Clenshaw path.

        Parameters
        ----------
        x : array_like or float
            Query points in physical coordinates.

        Returns
        -------
        numpy.ndarray or float
            CDF values with the shape of *x*.
        """
        return self._call_mode(x, False)

    def __call__(self, object x):
        """Evaluate the CDF at *x*, elementwise.

        Parameters
        ----------
        x : array_like or float
            Query points in physical coordinates.  Values outside
            ``[L, U]`` saturate to 0 or 1; NaN propagates.

        Returns
        -------
        numpy.ndarray or float
            CDF values with the shape of *x*; a Python float when *x* is
            a scalar.
        """
        return self._call_mode(x, True)
