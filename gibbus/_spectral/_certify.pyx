# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""Compiled Bernstein subdivision used by spectral construction certificates."""

import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY, fabs

cdef double _EPS = 2.220446049250313e-16

cnp.import_array()


cdef double _bernstein_lower_bound(
    const double* coeff,
    Py_ssize_t ncoeff,
    int max_subdivide,
    double certified,
    double* current,
    double* temp,
    double* stack,
    int* depth_stack,
) noexcept nogil:
    """Return the smallest leaf Bernstein coefficient of a subdivision.

    A subtree stops splitting once its coefficients are all at least
    ``certified`` (nonnegativity is then proved) or at ``max_subdivide``.
    """
    cdef Py_ssize_t i, r, live
    cdef int depth = 0
    cdef int top = 0
    cdef double lower = INFINITY
    cdef double mn
    cdef double* right

    for i in range(ncoeff):
        current[i] = coeff[i]

    while True:
        mn = current[0]
        for i in range(1, ncoeff):
            if current[i] < mn:
                mn = current[i]

        if mn >= certified or depth >= max_subdivide:
            if mn < lower:
                lower = mn
            if top == 0:
                break
            top -= 1
            depth = depth_stack[top]
            right = stack + top * ncoeff
            for i in range(ncoeff):
                current[i] = right[i]
            continue

        # Split at t=1/2 with de Casteljau.  Continue immediately with the
        # left child and push only the right child, so DFS needs O(depth*n)
        # scratch rather than an exponentially large explicit tree.
        for i in range(ncoeff):
            temp[i] = current[i]
        right = stack + top * ncoeff
        current[0] = temp[0]
        right[ncoeff - 1] = temp[ncoeff - 1]
        live = ncoeff
        for r in range(1, ncoeff):
            live -= 1
            for i in range(live):
                temp[i] = 0.5 * (temp[i] + temp[i + 1])
            current[r] = temp[0]
            right[ncoeff - 1 - r] = temp[live - 1]
        depth_stack[top] = depth + 1
        top += 1
        depth += 1

    return lower



cdef double _chebyshev_lower_bound_c(
    const double* coeff,
    Py_ssize_t ncoeff,
    const double* matrix,
    int max_subdivide,
    double* work,
    int* depths,
) noexcept nogil:
    """``chebyshev_lower_bound`` on raw pointers.

    ``work`` holds at least ``(4 + max_subdivide) * ncoeff`` doubles and
    ``depths`` ``max_subdivide + 1`` ints.
    """
    cdef double* bernstein = work
    cdef double* current = bernstein + ncoeff
    cdef double* temp = current + ncoeff
    cdef double* stack = temp + ncoeff
    cdef Py_ssize_t j, k
    cdef double total, magnitude, value, conversion = 0.0, largest = 0.0
    cdef double margin, lower
    for j in range(ncoeff):
        total = 0.0
        magnitude = 0.0
        for k in range(ncoeff):
            value = matrix[j * ncoeff + k] * coeff[k]
            total += value
            magnitude += fabs(value)
        bernstein[j] = total
        if magnitude > conversion:
            conversion = magnitude
        if fabs(total) > largest:
            largest = fabs(total)
    # Rounding of the conversion (a dot product of ncoeff rounded terms, each
    # matrix entry itself rounded) and of the de Casteljau averages.
    margin = ((ncoeff + 2) * conversion + 2.0 * (max_subdivide + 1) * largest) * _EPS
    lower = _bernstein_lower_bound(
        bernstein, ncoeff, max_subdivide, margin, current, temp, stack, depths
    )
    return lower - margin


def chebyshev_lower_bound(object coeff, object matrix, int max_subdivide):
    """Return a rigorous lower bound of a Chebyshev series on ``[-1, 1]``.

    The series is converted to Bernstein form with the exact
    Chebyshev-to-Bernstein matrix (entries correctly rounded) and bounded by
    recursive de Casteljau subdivision: on every subinterval the polynomial
    lies in the convex hull of its Bernstein coefficients.  The returned value
    subtracts a floating-point error bound for the conversion and the
    subdivision, so it never exceeds the true minimum.

    Parameters
    ----------
    coeff : array_like
        Chebyshev coefficients on ``[-1, 1]``, lowest order first.
    matrix : array_like, shape (ncoeff, ncoeff)
        ``chebyshev.chebyshev_bernstein_matrix(ncoeff - 1)``.
    max_subdivide : int
        Maximum recursive subdivision depth.

    Returns
    -------
    float
        Certified lower bound.
    """
    cdef cnp.ndarray arr = np.ascontiguousarray(coeff, dtype=np.float64)
    cdef cnp.ndarray matrix_arr = np.ascontiguousarray(matrix, dtype=np.float64)
    cdef Py_ssize_t ncoeff
    cdef cnp.ndarray work, depths
    cdef double result

    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("coeff must be a non-empty one-dimensional array")
    if max_subdivide < 0:
        raise ValueError("max_subdivide must be >= 0")
    ncoeff = arr.size
    if matrix_arr.ndim != 2 or matrix_arr.shape[0] != ncoeff or matrix_arr.shape[1] != ncoeff:
        raise ValueError("matrix must be a square matrix matching coeff")
    work = np.empty((4 + max_subdivide) * ncoeff, dtype=np.float64)
    depths = np.empty(max_subdivide + 1, dtype=np.intc)
    cdef const double* cp = <const double*> arr.data
    cdef const double* mp = <const double*> matrix_arr.data
    cdef double* wp = <double*> work.data
    cdef int* dp = <int*> depths.data
    with nogil:
        result = _chebyshev_lower_bound_c(cp, ncoeff, mp, max_subdivide, wp, dp)
    return result
