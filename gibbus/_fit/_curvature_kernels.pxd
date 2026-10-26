# cython: language_level=3

from .._model._basis_kernels cimport BasisTable, TailResult, TailTable

cdef enum:
    SEARCH_VERIFIED = 0
    SEARCH_VIOLATED = 1
    SEARCH_UNVERIFIED = 2
    MIN_INTERIOR = 0
    MIN_ENDPOINT = 1
    MIN_TAIL = 2
    MIN_UNRESOLVED = 3


cdef struct SearchOptions:
    double tau
    int degree
    double res_frac
    int max_depth
    int jmin
    int jmax
    int stop_shells
    int jmin_end
    int end_shells
    int nsamp_factor
    int n_extra
    int max_cuts
    int max_new_panels
    int max_cached_panels
    double zeta_cap
    double fallback_zeta


cdef struct Minimum:
    double x
    double z
    double value
    int kind


cdef struct Segment:
    double z_lo
    double z_hi
    double z_min
    double h_min
    int to_inf
    int touches_end


cdef struct SearchResult:
    int status
    double tau
    double a
    double b
    int n_bound
    int* bound_cols
    int n_seg
    Segment* segments
    int n_min
    Minimum* minima
    int n_tails
    TailResult tails[2]
    int tail_side[2]
    int n_cuts
    double* cut_points
    double* cut_rows
    int panels
    int unresolved


cdef struct Panel:
    int side
    int kind
    int depth
    int shell
    double a
    double b
    int child[2]
    Py_ssize_t rowoff


# Per-basis search state: the fixed panel hierarchy and the theta-free rows
# cached at each panel's Chebyshev nodes (grown lazily, kept across thetas).
cdef struct CurvSearch:
    const BasisTable* t
    int support
    int A
    int n
    int np
    double* nodes
    double* Tm
    double* Dv
    double pmax
    int nside
    int sides[2]
    Panel* P
    int nP
    int np_start
    int capP
    double* R
    double* NR
    double* Z
    double* DZ
    Py_ssize_t nrow
    Py_ssize_t caprow
    int inner_fixed[2][64]
    int n_inner_fixed[2]
    int* inner_end[2]
    int n_inner_end[2]
    int cap_inner_end[2]
    int* outer[2]
    int n_outer[2]
    int cap_outer[2]
    long kernel_points
    long panel_evals
    long direct_panels
    long resets


cdef class CurvatureSearch:
    cdef CurvSearch s
    cdef object _keep


cdef int separate_c(
    CurvSearch* S,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
) noexcept nogil
cdef void search_free(SearchResult* r) noexcept nogil
cdef void search_options(object options, SearchOptions* o)
