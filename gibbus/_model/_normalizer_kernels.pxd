# cython: language_level=3

from ._basis_kernels cimport BasisTable, TailTable

# Normalizer statuses: 0 proper, 1 improper, >= 10 failures (fail loudly).
cdef enum:
    NORM_PROPER = 0
    NORM_IMPROPER = 1
    NORM_UNDECIDED = 10
    NORM_TAIL_CAP = 11
    NORM_TAIL_NOCONV = 12
    NORM_TAIL_NONFINITE = 13
    NORM_CELLS_RESOLVE = 14
    NORM_CELLS_INTEGRATE = 15
    NORM_ACCURACY = 16
    NORM_NO_REFERENCE = 17
    NORM_ROW_POINT = 18


cdef struct GrowthTable:
    int d
    const int* start
    const double* coef
    const double* power
    const double* rate


cdef class GrowthKernel:
    cdef GrowthTable t
    cdef object _keep


cdef struct NormOptions:
    double rel_tol
    int initial_cells
    int max_cells
    int max_level
    double variation
    double drop_tol
    double zeta_cap


cdef struct NormResult:
    int status
    int stage
    int side
    double log_z
    double rel_error
    double cond_error
    int cells
    double diag[6]
    Py_ssize_t n_nodes
    double* nodes
    double* weights
    double part_v0
    double part_step


cdef int basis_state_c(
    const BasisTable* t,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const GrowthTable* growth_lo,
    const GrowthTable* growth_hi,
    const double* column_scale,
    const NormOptions* o,
    bint keep_nodes,
    double* mean,
    double* cov,
    NormResult* r,
) noexcept nogil


cdef int interval_states_c(
    const BasisTable* t,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const GrowthTable* growth_lo,
    const GrowthTable* growth_hi,
    const double* column_scale,
    const NormOptions* o,
    Py_ssize_t n_rows,
    const double* row_lo,
    const double* row_hi,
    const double* row_w,
    double* mean,
    double* cov,
    double* agg,
    double* row_logp,
    double* row_mean,
    double* row_cov,
    NormResult* r,
    NormResult* rr,
    Py_ssize_t* fail_row,
) noexcept nogil
