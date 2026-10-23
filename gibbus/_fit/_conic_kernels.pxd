ctypedef int (*face_evaluate_fn)(
    void* ctx,
    const double* x,
    double* nll,
    double* gradient,
    double* metric,
    double* smallest,
) noexcept nogil

ctypedef void (*face_accept_fn)(void* ctx) noexcept nogil

# After ``evaluate`` rejects a trial for shape, ``step`` may return a smaller
# step fraction (``0 < hint < alpha``) that keeps the rejecting rows feasible;
# any other value falls back to the loop's backtracking factor.  ``NULL``
# always backtracks.
ctypedef double (*face_step_fn)(
    void* ctx, const double* x, const double* trial, double alpha
) noexcept nogil

# An objective may return FACE_STOP from ``evaluate``: the loop then ends with
# status 6 at its current accepted iterate (G3 cut exchange, G3-038).
cdef enum:
    FACE_STOP = -100


# Optional scalar-inequality QP refinement. It replaces an endpoint only
# after checking its bound in the original quadratic model.
ctypedef int (*face_refine_fn)(
    void* ctx, int n, int r, const double* b, const double* h,
    const double* g, const double* theta, double* endpoint, double* dual,
    double* model, double* gap
) noexcept nogil

cdef struct FaceObjective:
    void* ctx
    face_evaluate_fn evaluate
    face_accept_fn accept
    face_step_fn step
    face_refine_fn refine


cdef int _in_safeguarded_metric(
    int n,
    const double* fisher,
    const double* missing,
    double* metric,
    double* work,
    double* smallest_out,
) noexcept nogil


cdef int _face_newton_loop(
    int n,
    int r,
    int nb,
    const int* sizes,
    const Py_ssize_t* aoff,
    const Py_ssize_t* qoff,
    Py_ssize_t qtot,
    Py_ssize_t na,
    const double* b,
    const double* a,
    const double* ref,
    const Py_ssize_t* degrees,
    const double* col_scale,
    FaceObjective* objective,
    double* theta,
    double* blocks,
    double* current_nll,
    double* gradient,
    double* hessian,
    double* current_smallest,
    double* dual,
    double tolerance,
    double certified_tolerance,
    double accuracy_floor,
    int max_iterations,
    double armijo,
    double backtrack,
    int max_line_search,
    int min_steps,
    int* iterations_out,
    int* evaluations_out,
    int* subproblem_iterations_out,
    double* bound_out,
) noexcept nogil
