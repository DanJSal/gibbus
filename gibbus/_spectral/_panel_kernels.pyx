# cython: language_level=3
"""Tiny Chebyshev panel-construction kernels used during post-fit builds.

These routines target the small fixed-degree arrays used by the adaptive CDF
and PPF constructors.  Keeping the transforms, Clenshaw evaluations, and
coefficient calculus in C avoids repeated NumPy dispatch and temporary arrays
for each trial panel.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cnp.import_array()


cdef inline double _chebval_one(double x, const double[::1] coeff) noexcept nogil:
    cdef Py_ssize_t n = coeff.shape[0]
    cdef Py_ssize_t i
    cdef double c0, c1, tmp
    if n == 0:
        return 0.0
    if n == 1:
        return coeff[0]
    c0 = coeff[n - 2]
    c1 = coeff[n - 1]
    for i in range(n - 3, -1, -1):
        tmp = c0
        c0 = coeff[i] - c1
        c1 = tmp + 2.0 * x * c1
    return c0 + x * c1


def lobatto_coefficients(const double[::1] values, const double[:, ::1] transform):
    """Apply one cached Lobatto value-to-coefficient transform."""
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t i, j
    cdef double total
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    if transform.shape[0] != n or transform.shape[1] != n:
        raise ValueError("transform shape must match values length")
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += transform[i, j] * values[j]
        out[i] = total
    return out


def chebval_many(const double[::1] x, const double[::1] coeff):
    """Evaluate one Chebyshev series at many points with scalar Clenshaw loops."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)
    with nogil:
        for i in range(n):
            out[i] = _chebval_one(x[i], coeff)
    return out


def cdf_panel_metrics(
    const double[::1] values,
    const double[::1] exact,
    const double[:, ::1] transform,
    const double[::1] validation_nodes,
):
    """Return coefficients and error diagnostics for one forward CDF panel."""
    cdef Py_ssize_t n = values.shape[0]
    cdef Py_ssize_t m = exact.shape[0]
    cdef Py_ssize_t i, j
    cdef double total, pred, err, max_err = 0.0
    cdef double max_exact = 0.0, max_values = 0.0
    cdef double max_coeff = 0.0, tail_abs = 0.0, av
    cdef Py_ssize_t tail_start
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coeff
    cdef double[::1] cv
    if transform.shape[0] != n or transform.shape[1] != n:
        raise ValueError("transform shape must match values length")
    if validation_nodes.shape[0] != m:
        raise ValueError("validation_nodes must match exact length")

    coeff = np.empty(n, dtype=np.float64)
    cv = coeff
    for i in range(n):
        total = 0.0
        for j in range(n):
            total += transform[i, j] * values[j]
        cv[i] = total
        av = fabs(total)
        if av > max_coeff:
            max_coeff = av

    for i in range(n):
        av = fabs(values[i])
        if av > max_values:
            max_values = av
    for i in range(m):
        av = fabs(exact[i])
        if av > max_exact:
            max_exact = av
        pred = _chebval_one(validation_nodes[i], cv)
        err = fabs(exact[i] - pred)
        if err > max_err:
            max_err = err

    tail_start = n - 4
    if tail_start < 0:
        tail_start = 0
    for i in range(tail_start, n):
        av = fabs(cv[i])
        if av > tail_abs:
            tail_abs = av

    return coeff, max_err, max(max_exact, max_values), tail_abs, max_coeff


def chebint_scaled(const double[::1] coeff, double scl=1.0):
    """Integrate one Chebyshev series once, matching ``numpy.chebint`` at lbnd=0."""
    cdef Py_ssize_t n = coeff.shape[0]
    cdef Py_ssize_t j
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef double[::1] ov
    cdef double left, right
    if n == 0:
        return np.empty(0, dtype=np.float64), 0.0
    out = np.zeros(n + 1, dtype=np.float64)
    ov = out
    if n == 1 and coeff[0] == 0.0:
        return out, 0.0
    ov[1] = coeff[0] * scl
    if n > 1:
        ov[2] = coeff[1] * scl / 4.0
    for j in range(2, n):
        ov[j + 1] = coeff[j] * scl / (2.0 * (j + 1))
        ov[j - 1] -= coeff[j] * scl / (2.0 * (j - 1))
    # numpy.chebint with lbnd=0 chooses the constant so the antiderivative is
    # zero at x=0.  The panel mass is invariant to that constant, but matching
    # NumPy exactly keeps packed coefficients bitwise-close to the old path.
    ov[0] = -_chebval_one(0.0, ov)
    left = _chebval_one(-1.0, ov)
    right = _chebval_one(1.0, ov)
    return out, right - left


def chebder(const double[::1] coeff):
    """Differentiate one Chebyshev series once."""
    cdef Py_ssize_t n = coeff.shape[0]
    cdef Py_ssize_t j
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef double[::1] ov
    if n <= 1:
        return np.zeros(1, dtype=np.float64)
    out = np.zeros(n - 1, dtype=np.float64)
    ov = out
    if n == 2:
        ov[0] = coeff[1]
        return out
    ov[n - 2] = 2.0 * (n - 1) * coeff[n - 1]
    ov[n - 3] = 2.0 * (n - 2) * coeff[n - 2]
    for j in range(n - 4, -1, -1):
        ov[j] = ov[j + 2] + 2.0 * (j + 1) * coeff[j + 1]
    ov[0] *= 0.5
    return out
