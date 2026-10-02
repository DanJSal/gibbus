# cython: language_level=3
"""C-level Bernstein certificate shared with the compiled spectral builders."""

cdef double _chebyshev_lower_bound_c(
    const double* coeff,
    Py_ssize_t ncoeff,
    const double* matrix,
    int max_subdivide,
    double* work,
    int* depths,
) noexcept nogil
