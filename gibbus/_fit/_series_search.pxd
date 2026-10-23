cdef struct SeriesSearchTable:
    int panels
    int order
    int d
    int samples
    int max_steps
    double tolerance
    double threshold
    const double* geometry
    const double* tensor
    const double* scan_points
    const double* scan_rows
    double* scan_values
    double* coefficients
    double* warm
    double* warm_scores
    int* warm_counts
    double* scores
    double* last_theta
    double* last_points
    int last_count
    int has_last
    int replays
    int capacity
    int calls
    int failures
    int refinements
    int scan_nodes
    int q2_builds
    int q3_builds
    int q4_builds
    int warm_probes

cdef class SeriesSearchData:
    cdef object _keep

cdef class SeriesSearchKernel:
    cdef SeriesSearchTable t
    cdef object _keep

# Returns candidate count, or -1 on numerical failure. Caller owns out[capacity].
cdef int series_candidates_c(SeriesSearchTable* t,const double* theta,
                             double* out,int capacity) noexcept nogil
