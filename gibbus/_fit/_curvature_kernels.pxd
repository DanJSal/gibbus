# cython: language_level=3

from .._model._basis_kernels cimport BasisTable, TailResult, TailTable

cdef enum:
    SEARCH_NONNEGATIVE = 0
    SEARCH_VIOLATED = 1
    SEARCH_INCONCLUSIVE = 2
    MIN_INTERIOR = 0
    MIN_ENDPOINT = 1
    MIN_AFFINE = 2
    MIN_UNRESOLVED = 3


cdef struct SearchOptions:
    double rho
    int initial_cells
    int max_cells
    double min_width
    double newton_tol
    int newton_max
    double zeta_cap
    double fallback_zeta
    int max_cuts
    double feasibility_tol


cdef struct Minimum:
    double x
    double value
    double lower
    double upper
    int kind


cdef struct SearchResult:
    int status
    double tau
    double a
    double b
    int n_bound
    int* bound_cols
    int n_min
    Minimum* minima
    int n_tails
    TailResult tails[2]
    int tail_side[2]
    int n_cuts
    double* cut_points
    double* cut_rows
    int cells
    int unresolved


cdef int separate_c(
    const BasisTable* t,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
) noexcept nogil
cdef void search_free(SearchResult* r) noexcept nogil
cdef int valleys_c(
    const BasisTable* t,
    const double* theta,
    double a,
    double b,
    const SearchOptions* o,
    double* out,
    int* n,
) noexcept nogil
