# cython: language_level=3
"""Fused compiled objectives for natural-mixture Newton solves.

Two exact-face objectives drive the generic compiled Newton loop of
``_conic_kernels`` so that every trial evaluation and the whole fixed-face
solve run without the GIL:

* ``SharedMStepObjective`` -- one EM M-step of a shared-boundary mixture:
  the responsibility-mass-weighted sum of the component natural objectives
  in the free-face coordinates, where every component reads the single
  amplitude of each shared physical side.
* ``JointMixtureObjective`` -- the observed-data mixture likelihood over
  private shapes, shared amplitudes and ``K - 1`` mixture logits, with its
  complete (Fisher) and missing (Louis) information.

Component objectives reuse the single-component point and interval
evaluators.  Joint per-row arrays are feature-major (rows contiguous), so the
posterior and the information reductions are unit-stride loops over rows.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, isfinite, log, INFINITY, NAN
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset

from .._model._state_kernels cimport _state_numerics_c
from .._observations._finite_reductions cimport (
    finite_natural_objective_c,
    finite_natural_real_line_objective_c,
)
from .._observations._interval_integrals cimport adaptive_natural_objective_c
from ._conic_kernels cimport (
    FaceObjective,
    _face_newton_loop,
    _in_evaluate_raw,
    _in_safeguarded_metric,
    _pn_evaluate,
)

cnp.import_array()

cdef extern from * nogil:
    """
    #include <stddef.h>
    #if defined(_MSC_VER)
      #define GIBBUS_SMK_RESTRICT __restrict
      #define GIBBUS_SMK_SIMD
      #define GIBBUS_SMK_SIMD_SUM
    #else
      #define GIBBUS_SMK_RESTRICT __restrict__
      #define GIBBUS_SMK_SIMD _Pragma("omp simd")
      #define GIBBUS_SMK_SIMD_SUM _Pragma("omp simd reduction(+:s)")
    #endif

    /* sum_i x[i] y[i] */
    static inline double gibbus_smk_dot(const double *GIBBUS_SMK_RESTRICT x,
                                        const double *GIBBUS_SMK_RESTRICT y,
                                        ptrdiff_t n)
    {
        double s = 0.0;
        GIBBUS_SMK_SIMD_SUM
        for (ptrdiff_t i = 0; i < n; ++i) s += x[i] * y[i];
        return s;
    }

    /* y[i] += a x[i] */
    static inline void gibbus_smk_axpy(double a, const double *GIBBUS_SMK_RESTRICT x,
                                       double *GIBBUS_SMK_RESTRICT y, ptrdiff_t n)
    {
        GIBBUS_SMK_SIMD
        for (ptrdiff_t i = 0; i < n; ++i) y[i] += a * x[i];
    }

    /* out[i] = x[i] y[i] */
    static inline void gibbus_smk_mul(const double *GIBBUS_SMK_RESTRICT x,
                                      const double *GIBBUS_SMK_RESTRICT y,
                                      double *GIBBUS_SMK_RESTRICT out, ptrdiff_t n)
    {
        GIBBUS_SMK_SIMD
        for (ptrdiff_t i = 0; i < n; ++i) out[i] = x[i] * y[i];
    }

    /* out[i] = x[i] - c */
    static inline void gibbus_smk_shift(const double *GIBBUS_SMK_RESTRICT x, double c,
                                        double *GIBBUS_SMK_RESTRICT out, ptrdiff_t n)
    {
        GIBBUS_SMK_SIMD
        for (ptrdiff_t i = 0; i < n; ++i) out[i] = x[i] - c;
    }
    """
    double gibbus_smk_dot(const double* x, const double* y, Py_ssize_t n)
    void gibbus_smk_axpy(double a, const double* x, double* y, Py_ssize_t n)
    void gibbus_smk_mul(const double* x, const double* y, double* out, Py_ssize_t n)
    void gibbus_smk_shift(const double* x, double c, double* out, Py_ssize_t n)


cdef enum:
    _POINT = 0
    _INTERVAL = 1

# Evaluation failures (0 is success).  Component failures are reported as
# ``_COMPONENT_FAILURE * (k + 1) + status`` of component ``k``.
cdef enum:
    _BAD_NLL = 2
    _BAD_INFORMATION = 3
    _BAD_METRIC = 4
    _ZERO_LIKELIHOOD = 5
    _COMPONENT_FAILURE = 1000

_STATUSES = {
    1: "converged",
    2: "converged_approximately",
    3: "non_descent",
    4: "line_search_failed",
    5: "iteration_limit",
}


cdef struct Component:
    # Fixed natural-state inputs (as in the single-component kernels).
    int kind
    int n
    int curvature_degree
    int lower_index
    int upper_index
    int n_power
    int n_log
    int F
    Py_ssize_t width
    Py_ssize_t state_size
    const double* support
    const double* data_bounds
    const int* kinds
    const int* lengths
    const double* coefficients
    const double* controls
    double epsabs
    double epsrel
    int limit
    # Point data: empirical partial means and the log coordinate scale.
    const double* empirical
    double coordinate_constant
    # Interval data, partitioned as in the single-component solver.
    Py_ssize_t Rf
    Py_ssize_t Ra
    const double* finite_intervals
    const double* finite_weights
    const double* point_lower_distance
    const double* point_upper_distance
    const double* adaptive_intervals
    const double* adaptive_weights
    double whole_weight
    double log_coordinate_scale
    Py_ssize_t G
    const double* gl_nodes
    const double* gl_log_weights
    double width_eps_mult
    # Coupling into the face coordinates (-1: the coordinate is exactly 0).
    double mass
    const int* columns
    int logit_column
    # Joint rows: the point basis (n x N) or the joint row of each interval
    # row of the finite, adaptive and whole-support partitions.
    const double* basis
    const Py_ssize_t* finite_rows
    const Py_ssize_t* adaptive_rows
    const Py_ssize_t* whole_rows
    Py_ssize_t Rw
    # Workspace.
    double* theta
    double* q_poly
    double* amplitudes
    double* state_work
    double* geometry
    double* points
    double* moments
    double* natural_scale
    double* log_probability
    double* sum_h
    double* sum_second
    double* hbuf
    double* obs_h
    # Raw geometry of the latest evaluation.
    double nll
    double* gradient
    double* means
    double* fisher
    double* missing
    # Raw geometry at the accepted iterate.
    double accepted_nll
    double* accepted_gradient
    double* accepted_means
    double* accepted_fisher
    double* accepted_missing
    # Joint per-row outputs.
    double* adaptive_log_probability
    double* finite_row_means
    double* finite_row_cov
    double* adaptive_row_means
    double* adaptive_row_cov
    double* log_values
    double* centered


cdef struct MStep:
    int K
    int nf
    bint raw
    Component* comps
    double* fisher
    double* missing
    double* metric_work
    double* accepted_fisher
    double* accepted_missing


cdef struct Joint:
    int K
    int nf
    bint raw
    Py_ssize_t N
    const double* row_weights
    double weight_total
    Component* comps
    double* log_pi
    double* pi
    double* mass
    double* row_max
    double* row_sum
    double* resp
    double* scaled
    double* v
    double* tmp
    double* gathered
    double* minus_one
    double* fisher
    double* missing
    double* metric_work
    double* accepted_fisher
    double* accepted_missing
    double log_likelihood


# ---------------------------------------------------------------------------
# Component binding and evaluation
# ---------------------------------------------------------------------------

cdef const double* _doubles(list refs, object value, Py_ssize_t size) except? NULL:
    """Keep a C-contiguous float64 copy of ``value`` alive and return its data."""
    cdef cnp.ndarray array = np.ascontiguousarray(value, dtype=np.float64)
    if size >= 0 and array.size != size:
        raise ValueError("compiled mixture input has the wrong size")
    refs.append(array)
    return <const double*>cnp.PyArray_DATA(array)


cdef const int* _ints(list refs, object value, Py_ssize_t size) except? NULL:
    """Keep a C-contiguous C-int copy of ``value`` alive and return its data."""
    cdef cnp.ndarray array = np.ascontiguousarray(value, dtype=np.intc)
    if size >= 0 and array.size != size:
        raise ValueError("compiled mixture input has the wrong size")
    refs.append(array)
    return <const int*>cnp.PyArray_DATA(array)


cdef const Py_ssize_t* _indices(list refs, object value, Py_ssize_t N) except? NULL:
    """Keep validated joint row indices alive and return their data."""
    cdef cnp.ndarray array = np.ascontiguousarray(value, dtype=np.intp).reshape(-1)
    if array.size and (array.min() < 0 or array.max() >= N):
        raise ValueError("compiled mixture row index is out of range")
    refs.append(array)
    return <const Py_ssize_t*>cnp.PyArray_DATA(array)


cdef int _bind_state(
    Component* c,
    list refs,
    object support,
    object data_bounds,
    object kinds,
    object lengths,
    object coefficients,
    object controls,
    int curvature_degree,
    int lower_index,
    int upper_index,
    double epsabs,
    double epsrel,
    int limit,
) except -1:
    """Bind the natural-state inputs shared by point and interval components."""
    cdef cnp.ndarray packed = np.ascontiguousarray(coefficients, dtype=np.float64)
    cdef int n
    cdef int nq = curvature_degree + 3
    if packed.ndim != 2 or packed.shape[1] != nq or limit < 1:
        raise ValueError("invalid compiled mixture state geometry")
    n = <int>packed.shape[0]
    if (
        n < 1
        or lower_index < -1
        or lower_index >= n
        or upper_index < -1
        or upper_index >= n
    ):
        raise ValueError("invalid compiled mixture state geometry")
    refs.append(packed)
    c.n = n
    c.curvature_degree = curvature_degree
    c.lower_index = lower_index
    c.upper_index = upper_index
    c.width = nq
    c.n_power = <int>(2 * nq - 1)
    c.n_log = nq
    c.F = (
        c.n_power
        + (c.n_log if lower_index >= 0 else 0)
        + (c.n_log if upper_index >= 0 else 0)
        + (1 if lower_index >= 0 else 0)
        + (1 if upper_index >= 0 else 0)
        + (1 if (lower_index >= 0 and upper_index >= 0) else 0)
    )
    c.state_size = 3 * nq + 2 * limit * c.F + 2 * limit + 8 * c.F
    c.support = _doubles(refs, support, 2)
    if not (c.support[0] < c.support[1]):
        raise ValueError("invalid compiled mixture support")
    c.data_bounds = _doubles(refs, data_bounds, 2)
    c.kinds = _ints(refs, kinds, n)
    c.lengths = _ints(refs, lengths, n)
    c.coefficients = <const double*>cnp.PyArray_DATA(packed)
    c.controls = _doubles(refs, controls, 11)
    c.epsabs = epsabs
    c.epsrel = epsrel
    c.limit = limit
    return 0


cdef int _bind_component(Component* c, list refs, int kind, tuple packed) except -1:
    """Bind one component's packed compiled-objective inputs."""
    cdef cnp.ndarray finite
    cdef cnp.ndarray adaptive
    cdef double scale
    memset(c, 0, sizeof(Component))
    c.logit_column = -1
    if kind == _POINT:
        if len(packed) != 16:
            raise ValueError("invalid compiled point inputs")
        _bind_state(
            c,
            refs,
            packed[0],
            packed[1],
            packed[4],
            packed[5],
            packed[6],
            packed[7],
            packed[10],
            packed[11],
            packed[12],
            packed[13],
            packed[14],
            packed[15],
        )
        c.kind = _POINT
        c.empirical = _doubles(refs, packed[8], c.n)
        c.coordinate_constant = packed[9]
        return 0
    if kind != _INTERVAL or len(packed) != 23:
        raise ValueError("invalid compiled interval inputs")
    _bind_state(
        c,
        refs,
        packed[0],
        packed[1],
        packed[2],
        packed[3],
        packed[4],
        packed[5],
        packed[17],
        packed[18],
        packed[19],
        packed[20],
        packed[21],
        packed[22],
    )
    c.kind = _INTERVAL
    finite = np.ascontiguousarray(packed[6], dtype=np.float64).reshape(-1, 2)
    adaptive = np.ascontiguousarray(packed[10], dtype=np.float64).reshape(-1, 2)
    refs.append(finite)
    refs.append(adaptive)
    c.Rf = finite.shape[0]
    c.Ra = adaptive.shape[0]
    c.finite_intervals = <const double*>cnp.PyArray_DATA(finite)
    c.finite_weights = _doubles(refs, packed[7], c.Rf)
    c.point_lower_distance = _doubles(refs, packed[8], c.Rf)
    c.point_upper_distance = _doubles(refs, packed[9], c.Rf)
    c.adaptive_intervals = <const double*>cnp.PyArray_DATA(adaptive)
    c.adaptive_weights = _doubles(refs, packed[11], c.Ra)
    c.whole_weight = packed[12]
    scale = packed[13]
    if not (scale > 0.0 and isfinite(scale)):
        raise ValueError("invalid compiled interval coordinate scale")
    if not (c.whole_weight >= 0.0 and isfinite(c.whole_weight)):
        raise ValueError("invalid compiled interval whole-support weight")
    c.log_coordinate_scale = log(scale)
    c.G = np.asarray(packed[14]).size
    if c.G < 1:
        raise ValueError("invalid compiled interval quadrature rule")
    c.gl_nodes = _doubles(refs, packed[14], c.G)
    c.gl_log_weights = _doubles(refs, packed[15], c.G)
    c.width_eps_mult = packed[16]
    return 0


cdef Py_ssize_t _component_size(const Component* c, Py_ssize_t N, bint joint) noexcept:
    """Doubles of workspace used by one component."""
    cdef Py_ssize_t n = c.n
    cdef Py_ssize_t n2 = n * n
    cdef Py_ssize_t rows = c.Rf if c.Rf > c.Ra else c.Ra
    cdef Py_ssize_t size
    if rows < 1:
        rows = 1
    size = (
        n + c.width + 2 + c.state_size + 6 + 16 + c.F + n + rows
        + n + n2 + n + n
        + n + n + n2 + n2
        + n + n + n2 + n2
    )
    if joint:
        size += c.Ra + (n + n2) * (c.Rf + c.Ra) + N + n * N
    return size


cdef double* _carve(Component* c, double* p, Py_ssize_t N, bint joint) noexcept:
    """Assign one component's workspace from ``p`` and return the next slot."""
    cdef Py_ssize_t n = c.n
    cdef Py_ssize_t n2 = n * n
    cdef Py_ssize_t rows = c.Rf if c.Rf > c.Ra else c.Ra
    if rows < 1:
        rows = 1
    c.theta = p
    p += n
    c.q_poly = p
    p += c.width
    c.amplitudes = p
    p += 2
    c.state_work = p
    p += c.state_size
    c.geometry = p
    p += 6
    c.points = p
    p += 16
    c.moments = p
    p += c.F
    c.natural_scale = p
    p += n
    c.log_probability = p
    p += rows
    c.sum_h = p
    p += n
    c.sum_second = p
    p += n2
    c.hbuf = p
    p += n
    c.obs_h = p
    p += n
    c.gradient = p
    p += n
    c.means = p
    p += n
    c.fisher = p
    p += n2
    c.missing = p
    p += n2
    c.accepted_gradient = p
    p += n
    c.accepted_means = p
    p += n
    c.accepted_fisher = p
    p += n2
    c.accepted_missing = p
    p += n2
    # Point components never write missing information.
    memset(c.missing, 0, n2 * sizeof(double))
    if joint:
        c.adaptive_log_probability = p
        p += c.Ra
        c.finite_row_means = p
        p += n * c.Rf
        c.finite_row_cov = p
        p += n2 * c.Rf
        c.adaptive_row_means = p
        p += n * c.Ra
        c.adaptive_row_cov = p
        p += n2 * c.Ra
        c.log_values = p
        p += N
        c.centered = p
        p += n * N
    return p


cdef int _component_evaluate(Component* c, bint rows) noexcept nogil:
    """Evaluate the raw geometry of one component at ``c.theta``."""
    if c.kind == _POINT:
        return _pn_evaluate(
            c.n,
            c.curvature_degree,
            c.lower_index,
            c.upper_index,
            c.theta,
            c.empirical,
            c.coordinate_constant,
            c.support,
            c.data_bounds,
            c.lower_index >= 0,
            c.upper_index >= 0,
            c.kinds,
            c.lengths,
            c.coefficients,
            c.width,
            c.controls,
            c.epsabs,
            c.epsrel,
            c.limit,
            c.q_poly,
            c.amplitudes,
            c.state_work,
            c.geometry,
            c.points,
            c.moments,
            c.n_power,
            c.n_log,
            c.F,
            &c.nll,
            c.gradient,
            c.means,
            c.fisher,
        )
    return _in_evaluate_raw(
        c.n,
        c.curvature_degree,
        c.theta,
        c.Rf,
        c.finite_intervals,
        c.finite_weights,
        c.point_lower_distance,
        c.point_upper_distance,
        c.Ra,
        c.adaptive_intervals,
        c.adaptive_weights,
        c.whole_weight,
        c.log_coordinate_scale,
        c.support,
        c.data_bounds,
        c.kinds,
        c.lengths,
        c.coefficients,
        c.width,
        c.controls,
        c.epsabs,
        c.epsrel,
        c.limit,
        c.G,
        c.gl_nodes,
        c.gl_log_weights,
        c.width_eps_mult,
        c.lower_index,
        c.upper_index,
        c.q_poly,
        c.amplitudes,
        c.state_work,
        c.geometry,
        c.points,
        c.moments,
        c.means,
        c.fisher,
        c.natural_scale,
        c.log_probability,
        c.obs_h,
        c.missing,
        c.sum_h,
        c.sum_second,
        c.hbuf,
        &c.nll,
        c.gradient,
        c.adaptive_log_probability,
        c.finite_row_means if rows else NULL,
        c.finite_row_cov if rows else NULL,
        c.adaptive_row_means if rows else NULL,
        c.adaptive_row_cov if rows else NULL,
    )


cdef int _component_log_masses(Component* c) noexcept nogil:
    """Normalize one joint component and reduce only its row log probabilities.

    The E-step needs no information, but its log likelihood must equal the
    joint objective's bit for bit.  The state therefore integrates every
    feature (they drive its adaptive subdivision) and only skips assembling
    the model moments; finite rows, whose fixed Gauss-Legendre mass does not
    depend on the statistics, are reduced without partial statistics; adaptive
    rows keep theirs for the same reason as the state.  Point components set
    ``c.nll`` to ``log Z`` plus the log coordinate scale; interval components
    fill ``c.log_probability`` and ``c.adaptive_log_probability``.  Status
    codes match ``_component_evaluate``.
    """
    cdef int k, status, npts = 0
    cdef Py_ssize_t nq = c.curvature_degree + 3
    cdef double shifted_z = 0.0, log_z, nll = 0.0
    cdef bint real_line = (
        not isfinite(c.support[0])
        and not isfinite(c.support[1])
        and c.lower_index < 0
        and c.upper_index < 0
    )
    c.q_poly[0] = 0.0
    c.q_poly[1] = c.theta[0]
    for k in range(c.curvature_degree + 1):
        c.q_poly[k + 2] = c.theta[k + 1] / ((k + 1.0) * (k + 2.0))
    c.amplitudes[0] = c.theta[c.lower_index] if c.lower_index >= 0 else NAN
    c.amplitudes[1] = c.theta[c.upper_index] if c.upper_index >= 0 else NAN
    status = _state_numerics_c(
        c.support,
        c.q_poly,
        nq,
        c.amplitudes,
        c.data_bounds,
        c.lower_index >= 0,
        c.upper_index >= 0,
        c.n_power,
        c.n_log,
        c.F,
        c.kinds,
        c.lengths,
        c.coefficients,
        0,
        c.width,
        c.controls,
        c.epsabs,
        c.epsrel,
        c.limit,
        c.state_work,
        c.geometry,
        c.points,
        &npts,
        &shifted_z,
        c.moments,
        NULL,
        NULL,
    )
    if c.kind == _POINT:
        if status != 0:
            return status
        c.nll = -c.geometry[3] + log(shifted_z) + c.coordinate_constant
        return 0 if isfinite(c.nll) else 1
    if status != 0 or not (shifted_z > 0.0 and isfinite(shifted_z)):
        return 10 + status
    log_z = log(shifted_z)
    if c.Rf > 0:
        if real_line:
            status = finite_natural_real_line_objective_c(
                c.Rf,
                0,
                c.G,
                nq,
                c.finite_intervals,
                c.finite_weights,
                c.q_poly,
                c.geometry[3],
                log_z,
                c.geometry[2],
                c.log_coordinate_scale,
                c.natural_scale,
                c.gl_nodes,
                c.gl_log_weights,
                c.width_eps_mult,
                c.log_probability,
                c.obs_h,
                c.missing,
                c.sum_h,
                c.sum_second,
                c.hbuf,
                &nll,
                NULL,
                NULL,
            )
        else:
            status = finite_natural_objective_c(
                c.Rf,
                0,
                c.G,
                c.width,
                nq,
                c.finite_intervals,
                c.finite_weights,
                c.point_lower_distance,
                c.point_upper_distance,
                c.q_poly,
                c.amplitudes[0],
                c.amplitudes[1],
                c.geometry[3],
                log_z,
                c.geometry[2],
                c.log_coordinate_scale,
                c.kinds,
                c.lengths,
                c.coefficients,
                c.support[0],
                c.support[1],
                c.gl_nodes,
                c.gl_log_weights,
                c.width_eps_mult,
                c.log_probability,
                c.obs_h,
                c.missing,
                c.sum_h,
                c.sum_second,
                c.hbuf,
                &nll,
                NULL,
                NULL,
            )
        if status != 0 or not isfinite(nll):
            return 30 + status
    if c.Ra > 0:
        status = adaptive_natural_objective_c(
            c.Ra,
            c.n,
            c.width,
            nq,
            c.adaptive_intervals,
            c.adaptive_weights,
            c.q_poly,
            c.amplitudes[0],
            c.amplitudes[1],
            c.geometry[2],
            c.geometry[5],
            -c.geometry[3] + log_z,
            c.kinds,
            c.lengths,
            c.coefficients,
            c.support[0],
            c.support[1],
            c.epsabs,
            c.epsrel,
            c.limit,
            c.adaptive_log_probability,
            c.obs_h,
            c.missing,
            &nll,
            NULL,
            NULL,
        )
        if status != 0 or not isfinite(nll):
            return 50 + status
    return 0


cdef inline void _gather(Component* c, const double* x) noexcept nogil:
    """Read one component's parameters from the face coordinates."""
    cdef int j, column
    for j in range(c.n):
        column = c.columns[j]
        c.theta[j] = x[column] if column >= 0 else 0.0


cdef void _component_accept(Component* c) noexcept nogil:
    """Commit the latest component evaluation as the accepted geometry."""
    cdef Py_ssize_t n = c.n
    c.accepted_nll = c.nll
    memcpy(c.accepted_gradient, c.gradient, n * sizeof(double))
    memcpy(c.accepted_means, c.means, n * sizeof(double))
    memcpy(c.accepted_fisher, c.fisher, n * n * sizeof(double))
    memcpy(c.accepted_missing, c.missing, n * n * sizeof(double))


cdef int _finish(
    int nf,
    double nll,
    const double* gradient,
    const double* fisher,
    const double* missing,
    double* metric,
    double* work,
    double* smallest,
    bint raw,
) noexcept nogil:
    """Reject invalid raw geometry, then build the safeguarded Newton metric.

    With ``raw`` only finiteness is checked and no metric is built.
    """
    cdef Py_ssize_t i, nf2 = nf * nf
    if not isfinite(nll):
        return _BAD_NLL
    for i in range(nf):
        if not (isfinite(gradient[i]) and (raw or fisher[i * nf + i] > 0.0)):
            return _BAD_INFORMATION
    for i in range(nf2):
        if not (isfinite(fisher[i]) and isfinite(missing[i])):
            return _BAD_INFORMATION
    if raw:
        return 0
    _in_safeguarded_metric(nf, fisher, missing, metric, work, smallest)
    for i in range(nf2):
        if not isfinite(metric[i]):
            return _BAD_METRIC
    return 0


# ---------------------------------------------------------------------------
# Shared-boundary M-step objective
# ---------------------------------------------------------------------------

cdef int _mstep_evaluate(
    void* raw,
    const double* x,
    double* nll,
    double* gradient,
    double* metric,
    double* smallest,
) noexcept nogil:
    """Mass-weighted component geometry scattered into the face coordinates."""
    cdef MStep* ctx = <MStep*>raw
    cdef Component* c
    cdef int nf = ctx.nf, k, j, m, a, b, n, status
    cdef double total = 0.0, mass
    memset(gradient, 0, nf * sizeof(double))
    memset(ctx.fisher, 0, nf * nf * sizeof(double))
    memset(ctx.missing, 0, nf * nf * sizeof(double))
    for k in range(ctx.K):
        c = &ctx.comps[k]
        n = c.n
        _gather(c, x)
        status = _component_evaluate(c, False)
        if status != 0:
            return _COMPONENT_FAILURE * (k + 1) + status
        mass = c.mass
        total += mass * c.nll
        for j in range(n):
            a = c.columns[j]
            if a < 0:
                continue
            gradient[a] += mass * c.gradient[j]
            for m in range(n):
                b = c.columns[m]
                if b < 0:
                    continue
                ctx.fisher[a * nf + b] += mass * c.fisher[j * n + m]
                ctx.missing[a * nf + b] += mass * c.missing[j * n + m]
    nll[0] = total
    return _finish(
        nf,
        total,
        gradient,
        ctx.fisher,
        ctx.missing,
        metric,
        ctx.metric_work,
        smallest,
        ctx.raw,
    )


cdef void _mstep_accept(void* raw) noexcept nogil:
    cdef MStep* ctx = <MStep*>raw
    cdef int k
    memcpy(ctx.accepted_fisher, ctx.fisher, ctx.nf * ctx.nf * sizeof(double))
    memcpy(ctx.accepted_missing, ctx.missing, ctx.nf * ctx.nf * sizeof(double))
    for k in range(ctx.K):
        _component_accept(&ctx.comps[k])


# ---------------------------------------------------------------------------
# Joint observed-data mixture objective
# ---------------------------------------------------------------------------

cdef void _joint_rows(Joint* ctx, Component* c, bint rows) noexcept nogil:
    """Write one component's per-row log values and centered partial means."""
    cdef Py_ssize_t N = ctx.N, t, i
    cdef int j, n = c.n
    cdef double value, mu
    cdef double* centered
    if c.kind == _POINT:
        # With zero empirical means the point NLL is log Z + log scale.
        for i in range(N):
            c.log_values[i] = -c.nll
        for j in range(n):
            value = c.theta[j]
            if value != 0.0:
                gibbus_smk_axpy(-value, &c.basis[j * N], c.log_values, N)
            if rows:
                gibbus_smk_shift(&c.basis[j * N], c.means[j], &c.centered[j * N], N)
        return
    for t in range(c.Rf):
        c.log_values[c.finite_rows[t]] = c.log_probability[t]
    for t in range(c.Ra):
        c.log_values[c.adaptive_rows[t]] = c.adaptive_log_probability[t]
    for t in range(c.Rw):
        c.log_values[c.whole_rows[t]] = 0.0
    if not rows:
        return
    for j in range(n):
        centered = &c.centered[j * N]
        mu = c.means[j]
        for t in range(c.Rf):
            centered[c.finite_rows[t]] = c.finite_row_means[j * c.Rf + t] - mu
        for t in range(c.Ra):
            centered[c.adaptive_rows[t]] = c.adaptive_row_means[j * c.Ra + t] - mu
        for t in range(c.Rw):
            centered[c.whole_rows[t]] = 0.0


cdef inline const double* _feature(Joint* ctx, Component* c, int j) noexcept nogil:
    """Row values of complete-data score feature ``j`` (the logit last)."""
    if j < c.n:
        return &c.centered[j * ctx.N]
    return ctx.minus_one


cdef inline int _feature_column(Component* c, int j) noexcept nogil:
    return c.columns[j] if j < c.n else c.logit_column


cdef int _joint_posterior(Joint* ctx, const double* x, bint rows) noexcept nogil:
    """Mixture log likelihood and responsibilities at the face point ``x``.

    Sets ``ctx.log_likelihood`` and ``ctx.resp``.  With ``rows`` every
    component evaluates its full geometry and keeps its centered per-row
    partial means; otherwise it reduces only its row log probabilities, which
    gives the same log likelihood bit for bit.
    """
    cdef Component* c
    cdef int K = ctx.K, k, status
    cdef Py_ssize_t N = ctx.N, i
    cdef double top, total, value
    cdef double* resp

    # Mixture weights from the free logits (the last logit is fixed at 0).
    top = 0.0
    for k in range(K - 1):
        value = x[ctx.comps[k].logit_column]
        if value > top:
            top = value
    total = 0.0
    for k in range(K - 1):
        total += exp(x[ctx.comps[k].logit_column] - top)
    total += exp(-top)
    top += log(total)
    for k in range(K - 1):
        ctx.log_pi[k] = x[ctx.comps[k].logit_column] - top
    ctx.log_pi[K - 1] = -top
    for k in range(K):
        ctx.pi[k] = exp(ctx.log_pi[k])

    for k in range(K):
        c = &ctx.comps[k]
        _gather(c, x)
        status = _component_evaluate(c, True) if rows else _component_log_masses(c)
        if status != 0:
            return _COMPONENT_FAILURE * (k + 1) + status
        _joint_rows(ctx, c, rows)

    # Posterior: per-row log-sum-exp over the components.
    for i in range(N):
        ctx.row_max[i] = -INFINITY
        ctx.row_sum[i] = 0.0
    for k in range(K):
        resp = &ctx.resp[k * N]
        value = ctx.log_pi[k]
        for i in range(N):
            resp[i] = ctx.comps[k].log_values[i] + value
            if resp[i] > ctx.row_max[i]:
                ctx.row_max[i] = resp[i]
    for i in range(N):
        if not isfinite(ctx.row_max[i]):
            return _ZERO_LIKELIHOOD
    for k in range(K):
        resp = &ctx.resp[k * N]
        for i in range(N):
            resp[i] = exp(resp[i] - ctx.row_max[i])
            ctx.row_sum[i] += resp[i]
    for k in range(K):
        resp = &ctx.resp[k * N]
        for i in range(N):
            resp[i] /= ctx.row_sum[i]
    total = 0.0
    for i in range(N):
        total += ctx.row_weights[i] * (ctx.row_max[i] + log(ctx.row_sum[i]))
    if not isfinite(total):
        return _ZERO_LIKELIHOOD
    ctx.log_likelihood = total
    return 0


cdef int _joint_evaluate(
    void* raw,
    const double* x,
    double* nll,
    double* gradient,
    double* metric,
    double* smallest,
) noexcept nogil:
    """Observed mixture NLL with complete and missing information.

    Row ``i``'s complete-data score for label ``k`` has the centered partial
    means ``c_ik`` in block ``k`` and ``pi - e_k`` in the logits.  The Louis
    missing information ``sum_i w_i sum_k r_ik (s_ik - m_i)(s_ik - m_i)^T``
    therefore equals ``sum_i w_i sum_kl (delta_kl r_ik - r_ik r_il)
    phi_ik phi_il^T`` with ``phi_ik = (c_ik, -e_k)``, plus the
    responsibility-weighted within-row covariances.
    """
    cdef Joint* ctx = <Joint*>raw
    cdef Component* c
    cdef Component* d
    cdef int K = ctx.K, nf = ctx.nf, k, other, j, m, a, b, status
    cdef int features_k, features_l
    cdef Py_ssize_t N = ctx.N, i, t
    cdef double value, mass
    cdef double* resp
    cdef double* scaled

    status = _joint_posterior(ctx, x, True)
    if status != 0:
        return status
    nll[0] = -ctx.log_likelihood

    # Gradient and the complete (Fisher) information.
    memset(gradient, 0, nf * sizeof(double))
    memset(ctx.fisher, 0, nf * nf * sizeof(double))
    memset(ctx.missing, 0, nf * nf * sizeof(double))
    for k in range(K):
        c = &ctx.comps[k]
        scaled = &ctx.scaled[k * N]
        gibbus_smk_mul(ctx.row_weights, &ctx.resp[k * N], scaled, N)
        mass = 0.0
        for i in range(N):
            mass += scaled[i]
        ctx.mass[k] = mass
        for j in range(c.n):
            a = c.columns[j]
            if a < 0:
                continue
            gradient[a] += gibbus_smk_dot(scaled, &c.centered[j * N], N)
            for m in range(c.n):
                b = c.columns[m]
                if b >= 0:
                    ctx.fisher[a * nf + b] += mass * c.fisher[j * c.n + m]
        if k < K - 1:
            gradient[c.logit_column] += ctx.weight_total * ctx.pi[k] - mass
    for k in range(K - 1):
        a = ctx.comps[k].logit_column
        for other in range(K - 1):
            b = ctx.comps[other].logit_column
            ctx.fisher[a * nf + b] += (
                (ctx.pi[k] if k == other else 0.0) - ctx.pi[k] * ctx.pi[other]
            )

    # Missing information between complete-data score features.
    for k in range(K):
        c = &ctx.comps[k]
        features_k = c.n + (1 if k < K - 1 else 0)
        scaled = &ctx.scaled[k * N]
        for other in range(k, K):
            d = &ctx.comps[other]
            features_l = d.n + (1 if other < K - 1 else 0)
            resp = &ctx.resp[other * N]
            if other == k:
                for i in range(N):
                    ctx.v[i] = scaled[i] * (1.0 - resp[i])
            else:
                for i in range(N):
                    ctx.v[i] = -scaled[i] * resp[i]
            for j in range(features_k):
                a = _feature_column(c, j)
                if a < 0:
                    continue
                gibbus_smk_mul(ctx.v, _feature(ctx, c, j), ctx.tmp, N)
                for m in range(j if other == k else 0, features_l):
                    b = _feature_column(d, m)
                    if b < 0:
                        continue
                    value = gibbus_smk_dot(ctx.tmp, _feature(ctx, d, m), N)
                    ctx.missing[a * nf + b] += value
                    if other != k or m != j:
                        ctx.missing[b * nf + a] += value

        # Responsibility-weighted within-row covariances of interval rows.
        if c.kind != _INTERVAL:
            continue
        for t in range(c.Rf):
            ctx.gathered[t] = scaled[c.finite_rows[t]]
        for t in range(c.Ra):
            ctx.gathered[c.Rf + t] = scaled[c.adaptive_rows[t]]
        mass = 0.0
        for t in range(c.Rw):
            mass += scaled[c.whole_rows[t]]
        for j in range(c.n):
            a = c.columns[j]
            if a < 0:
                continue
            for m in range(j, c.n):
                b = c.columns[m]
                if b < 0:
                    continue
                value = mass * c.fisher[j * c.n + m]
                if c.Rf > 0:
                    value += gibbus_smk_dot(
                        ctx.gathered, &c.finite_row_cov[(j * c.n + m) * c.Rf], c.Rf
                    )
                if c.Ra > 0:
                    value += gibbus_smk_dot(
                        &ctx.gathered[c.Rf],
                        &c.adaptive_row_cov[(j * c.n + m) * c.Ra],
                        c.Ra,
                    )
                ctx.missing[a * nf + b] += value
                if m != j:
                    ctx.missing[b * nf + a] += value

    return _finish(
        nf,
        nll[0],
        gradient,
        ctx.fisher,
        ctx.missing,
        metric,
        ctx.metric_work,
        smallest,
        ctx.raw,
    )


cdef void _joint_accept(void* raw) noexcept nogil:
    cdef Joint* ctx = <Joint*>raw
    memcpy(ctx.accepted_fisher, ctx.fisher, ctx.nf * ctx.nf * sizeof(double))
    memcpy(ctx.accepted_missing, ctx.missing, ctx.nf * ctx.nf * sizeof(double))


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------

cdef tuple _solve_face(
    FaceObjective* objective,
    int nf,
    double* accepted_fisher,
    double* accepted_missing,
    object params,
    object blocks_packed,
    object b_matrix,
    object a_packed,
    object sizes,
    object a_offsets,
    object q_offsets,
    object reference_dual,
    object row_degrees,
    double tolerance,
    double certified_tolerance,
    double accuracy_floor,
    int max_iterations,
    double armijo,
    double backtrack,
    int max_line_search,
    int min_steps,
):
    """Evaluate the start and run one fixed-face Newton solve without the GIL."""
    cdef cnp.ndarray[cnp.float64_t, ndim=1] theta = np.array(
        params, dtype=np.float64
    ).reshape(-1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] blocks = np.array(
        blocks_packed, dtype=np.float64
    ).reshape(-1)
    cdef const double[:, ::1] b = np.ascontiguousarray(b_matrix, dtype=np.float64)
    cdef const double[::1] a = np.ascontiguousarray(a_packed, dtype=np.float64)
    cdef const int[::1] k = np.ascontiguousarray(sizes, dtype=np.intc)
    cdef const Py_ssize_t[::1] aoff = np.ascontiguousarray(a_offsets, dtype=np.intp)
    cdef const Py_ssize_t[::1] qoff = np.ascontiguousarray(q_offsets, dtype=np.intp)
    cdef const double[::1] ref = np.ascontiguousarray(reference_dual, dtype=np.float64)
    cdef const Py_ssize_t[::1] degrees = np.ascontiguousarray(
        row_degrees, dtype=np.intp
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] gradient = np.zeros(nf)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] hessian = np.zeros((nf, nf))
    cdef int r = b.shape[0], nb = k.shape[0]
    cdef Py_ssize_t qtot = blocks.shape[0], na = a.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dual = np.zeros(r)
    cdef int iterations = 0, evaluations = 0, sub_iterations = 0, code = 0, start
    cdef double bound = INFINITY, nll = 0.0, smallest = 0.0
    cdef cnp.ndarray fisher = np.empty((nf, nf))
    cdef cnp.ndarray missing = np.empty((nf, nf))
    if (
        nf < 1
        or theta.shape[0] != nf
        or b.shape[1] != nf
        or r < 1
        or nb < 1
        or ref.shape[0] != r
        or degrees.shape[0] != r
        or aoff.shape[0] != nb
        or qoff.shape[0] != nb
    ):
        raise ValueError("invalid compiled mixture face geometry")
    with nogil:
        start = objective.evaluate(
            objective.ctx, &theta[0], &nll, &gradient[0], &hessian[0, 0], &smallest
        )
        if start == 0:
            objective.accept(objective.ctx)
            code = _face_newton_loop(
                nf,
                r,
                nb,
                &k[0],
                &aoff[0],
                &qoff[0],
                qtot,
                na,
                &b[0, 0],
                &a[0],
                &ref[0],
                &degrees[0],
                objective,
                &theta[0],
                &blocks[0],
                &nll,
                &gradient[0],
                &hessian[0, 0],
                &smallest,
                &dual[0],
                tolerance,
                certified_tolerance,
                accuracy_floor,
                max_iterations,
                armijo,
                backtrack,
                max_line_search,
                min_steps,
                &iterations,
                &evaluations,
                &sub_iterations,
                &bound,
            )
    if start != 0:
        raise FloatingPointError(
            f"compiled mixture start evaluation failed (status {start})"
        )
    if code == -2:
        raise np.linalg.LinAlgError("starting Gram block is not positive definite")
    if code == -1:
        raise MemoryError("compiled mixture Newton workspace")
    if code < 0:
        raise np.linalg.LinAlgError(
            "compiled mixture Newton precomputation did not converge"
        )
    memcpy(cnp.PyArray_DATA(fisher), accepted_fisher, nf * nf * sizeof(double))
    memcpy(cnp.PyArray_DATA(missing), accepted_missing, nf * nf * sizeof(double))
    return (
        _STATUSES[code],
        theta,
        blocks,
        dual,
        float(nll),
        gradient,
        hessian,
        fisher,
        missing,
        float(smallest),
        int(iterations),
        int(evaluations),
        int(sub_iterations),
        float(bound),
    )


cdef class _Workspace:
    """Owns the component table, its arena and the bound input arrays."""

    cdef Component* comps
    cdef double* arena
    cdef int K
    cdef list refs

    def __cinit__(self):
        self.comps = NULL
        self.arena = NULL
        self.K = 0
        self.refs = []

    def __dealloc__(self):
        free(self.comps)
        free(self.arena)

    cdef int allocate(self, int K) except -1:
        """Allocate the component table; ``carve`` assigns the arena later."""
        self.K = K
        self.comps = <Component*>malloc(K * sizeof(Component))
        if self.comps == NULL:
            raise MemoryError("compiled mixture component table")
        memset(self.comps, 0, K * sizeof(Component))
        return 0

    cdef double* carve(self, Py_ssize_t N, bint joint, Py_ssize_t extra) except NULL:
        """Allocate every component workspace; return the ``extra`` tail."""
        cdef Py_ssize_t total = extra
        cdef int k
        cdef double* p
        for k in range(self.K):
            total += _component_size(&self.comps[k], N, joint)
        self.arena = <double*>malloc((total if total > 0 else 1) * sizeof(double))
        if self.arena == NULL:
            raise MemoryError("compiled mixture workspace")
        p = self.arena
        for k in range(self.K):
            p = _carve(&self.comps[k], p, N, joint)
        return p


cdef class SharedMStepObjective:
    """Compiled responsibility-weighted M-step objective of a mixture.

    Parameters
    ----------
    components : sequence of (int, float, tuple)
        ``(kind, mass, inputs)`` per component: kind 0 for point data with the
        packed point-kernel inputs, kind 1 for interval data with the packed
        interval-kernel inputs; ``mass`` is the responsibility mass.
    """

    cdef _Workspace workspace
    cdef MStep ctx

    def __cinit__(self, components):
        cdef int K = len(components), k
        cdef Component* c
        if K < 1:
            raise ValueError("a compiled M-step needs at least one component")
        self.workspace = _Workspace()
        self.workspace.allocate(K)
        for k, (kind, mass, packed) in enumerate(components):
            c = &self.workspace.comps[k]
            _bind_component(c, self.workspace.refs, int(kind), tuple(packed))
            c.mass = float(mass)
            if not (c.mass >= 0.0 and isfinite(c.mass)):
                raise ValueError("component responsibility mass must be finite")
        self.workspace.carve(0, False, 0)
        memset(&self.ctx, 0, sizeof(MStep))
        self.ctx.K = K
        self.ctx.comps = self.workspace.comps

    cdef int _bind_face(self, list refs, object columns) except -1:
        """Point every component at its free-face columns."""
        cdef int k, nf = self.ctx.nf
        cdef Component* c
        if len(columns) != self.ctx.K:
            raise ValueError("one column map per component is required")
        for k in range(self.ctx.K):
            c = &self.ctx.comps[k]
            array = np.ascontiguousarray(columns[k], dtype=np.intc).reshape(-1)
            if array.size != c.n or (array.size and array.max() >= nf):
                raise ValueError("invalid component column map")
            refs.append(array)
            c.columns = <const int*>cnp.PyArray_DATA(array)
        return 0

    def _components(self):
        """Per-component accepted ``(nll, gradient, fisher, missing, means)``."""
        cdef int k
        cdef Py_ssize_t n
        cdef Component* c
        out = []
        for k in range(self.ctx.K):
            c = &self.ctx.comps[k]
            n = c.n
            out.append(
                (
                    float(c.accepted_nll),
                    np.asarray(<double[:n]>c.accepted_gradient).copy(),
                    np.asarray(<double[:n, :n]>c.accepted_fisher).copy(),
                    np.asarray(<double[:n, :n]>c.accepted_missing).copy(),
                    np.asarray(<double[:n]>c.accepted_means).copy(),
                )
            )
        return tuple(out)

    cdef cnp.ndarray _bind_work(self, int nf):
        """Allocate the information buffers of an ``nf``-dimensional face."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] work = np.empty(
            4 * nf * nf + 7 * nf * nf + 4 * nf
        )
        self.ctx.fisher = &work[0]
        self.ctx.missing = self.ctx.fisher + nf * nf
        self.ctx.accepted_fisher = self.ctx.missing + nf * nf
        self.ctx.accepted_missing = self.ctx.accepted_fisher + nf * nf
        self.ctx.metric_work = self.ctx.accepted_missing + nf * nf
        return work

    def evaluate(self, columns, int n_free, params):
        """Evaluate every component's raw geometry at one face point.

        Returns
        -------
        tuple
            Per-component ``(nll, gradient, fisher, missing, means)``.

        Raises
        ------
        FloatingPointError
            If the point is not a valid evaluation point.
        """
        cdef list refs = []
        cdef cnp.ndarray _work
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.ascontiguousarray(
            params, dtype=np.float64
        ).reshape(-1)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] gradient = np.zeros(n_free)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] metric = np.zeros((n_free, n_free))
        cdef double nll = 0.0, smallest = 0.0
        cdef int status
        if n_free < 1 or x.shape[0] != n_free:
            raise ValueError("invalid compiled M-step evaluation point")
        self.ctx.nf = n_free
        self._bind_face(refs, columns)
        _work = self._bind_work(n_free)
        self.ctx.raw = True
        with nogil:
            status = _mstep_evaluate(
                &self.ctx, &x[0], &nll, &gradient[0], &metric[0, 0], &smallest
            )
            if status == 0:
                _mstep_accept(&self.ctx)
        if status != 0:
            raise FloatingPointError(
                f"compiled M-step evaluation failed (status {status})"
            )
        return self._components()

    def solve(
        self,
        columns,
        int n_free,
        params,
        blocks_packed,
        b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        reference_dual,
        row_degrees,
        double tolerance,
        double certified_tolerance,
        double accuracy_floor,
        int max_iterations,
        double armijo,
        double backtrack,
        int max_line_search,
        int min_steps=0,
    ):
        """Optimize one exact face of the coupled M-step objective.

        Returns
        -------
        tuple
            ``(status, theta, blocks, dual, nll, gradient, metric, fisher,
            missing, smallest, iterations, evaluations, subproblem_iterations,
            bound, components)`` where ``components`` holds the per-component
            raw geometry at ``theta``.

        Raises
        ------
        FloatingPointError
            If the start is not a valid evaluation point.
        """
        cdef list refs = []
        cdef FaceObjective objective
        cdef cnp.ndarray _work
        self.ctx.nf = n_free
        self._bind_face(refs, columns)
        _work = self._bind_work(n_free)
        self.ctx.raw = False
        objective.ctx = &self.ctx
        objective.evaluate = _mstep_evaluate
        objective.accept = _mstep_accept
        result = _solve_face(
            &objective,
            n_free,
            self.ctx.accepted_fisher,
            self.ctx.accepted_missing,
            params,
            blocks_packed,
            b_matrix,
            a_packed,
            sizes,
            a_offsets,
            q_offsets,
            reference_dual,
            row_degrees,
            tolerance,
            certified_tolerance,
            accuracy_floor,
            max_iterations,
            armijo,
            backtrack,
            max_line_search,
            min_steps,
        )
        return result + (self._components(),)


cdef class JointMixtureObjective:
    """Compiled observed-data mixture likelihood on exact faces.

    Parameters
    ----------
    components : sequence of (int, tuple, object)
        ``(kind, inputs, rows)`` per component.  Point components pass the
        packed point-kernel inputs with zero empirical means and their natural
        basis ``h(z_i)`` as an ``(n, N)`` feature-major array.  Interval
        components pass packed interval-kernel inputs with unit row weights
        and no whole-support weight, and ``(finite, adaptive, whole)`` arrays
        giving the joint row of every row in each partition.
    row_weights : numpy.ndarray, shape (N,)
        Observation weight of every joint row.
    """

    cdef _Workspace workspace
    cdef Joint ctx
    cdef cnp.ndarray weights

    def __cinit__(self, components, row_weights):
        cdef int K = len(components), k
        cdef Py_ssize_t N, i, gathered = 1, extra
        cdef Component* c
        cdef double* p
        if K < 1:
            raise ValueError("a joint mixture needs at least one component")
        self.weights = np.ascontiguousarray(row_weights, dtype=np.float64).reshape(-1)
        N = self.weights.shape[0]
        if N < 1 or not np.all(np.isfinite(self.weights)):
            raise ValueError("joint mixture row weights must be finite")
        self.workspace = _Workspace()
        self.workspace.allocate(K)
        for k, (kind, packed, rows) in enumerate(components):
            c = &self.workspace.comps[k]
            _bind_component(c, self.workspace.refs, int(kind), tuple(packed))
            if c.kind == _POINT:
                c.basis = _doubles(self.workspace.refs, rows, c.n * N)
            else:
                finite, adaptive, whole = rows
                if (
                    np.size(finite) != c.Rf
                    or np.size(adaptive) != c.Ra
                    or c.Rf + c.Ra + np.size(whole) != N
                ):
                    raise ValueError("interval partitions must cover the joint rows")
                c.finite_rows = _indices(self.workspace.refs, finite, N)
                c.adaptive_rows = _indices(self.workspace.refs, adaptive, N)
                c.whole_rows = _indices(self.workspace.refs, whole, N)
                c.Rw = np.size(whole)
                if c.Rf + c.Ra > gathered:
                    gathered = c.Rf + c.Ra
        extra = 3 * K + 2 * N + 2 * K * N + 3 * N + gathered
        p = self.workspace.carve(N, True, extra)
        memset(&self.ctx, 0, sizeof(Joint))
        self.ctx.K = K
        self.ctx.N = N
        self.ctx.comps = self.workspace.comps
        self.ctx.row_weights = <const double*>cnp.PyArray_DATA(self.weights)
        self.ctx.weight_total = float(np.sum(self.weights))
        self.ctx.log_pi = p
        p += K
        self.ctx.pi = p
        p += K
        self.ctx.mass = p
        p += K
        self.ctx.row_max = p
        p += N
        self.ctx.row_sum = p
        p += N
        self.ctx.resp = p
        p += K * N
        self.ctx.scaled = p
        p += K * N
        self.ctx.v = p
        p += N
        self.ctx.tmp = p
        p += N
        self.ctx.minus_one = p
        p += N
        self.ctx.gathered = p
        for i in range(N):
            self.ctx.minus_one[i] = -1.0

    cdef int _bind_face(
        self, list refs, object columns, object logit_columns, int nf
    ) except -1:
        """Point every component and logit at its free-face column."""
        cdef int k
        cdef Component* c
        if len(columns) != self.ctx.K or len(logit_columns) != self.ctx.K - 1:
            raise ValueError("one column map per component and logit is required")
        for k in range(self.ctx.K):
            c = &self.ctx.comps[k]
            array = np.ascontiguousarray(columns[k], dtype=np.intc).reshape(-1)
            if array.size != c.n or (array.size and array.max() >= nf):
                raise ValueError("invalid component column map")
            refs.append(array)
            c.columns = <const int*>cnp.PyArray_DATA(array)
            c.logit_column = -1
            if k < self.ctx.K - 1:
                c.logit_column = int(logit_columns[k])
                if not 0 <= c.logit_column < nf:
                    raise ValueError("mixture logits must be free coordinates")
        self.ctx.nf = nf
        return 0

    cdef cnp.ndarray _bind_work(self, int nf):
        """Allocate the information buffers of an ``nf``-dimensional face."""
        cdef cnp.ndarray[cnp.float64_t, ndim=1] work = np.empty(
            4 * nf * nf + 7 * nf * nf + 4 * nf
        )
        self.ctx.fisher = &work[0]
        self.ctx.missing = self.ctx.fisher + nf * nf
        self.ctx.accepted_fisher = self.ctx.missing + nf * nf
        self.ctx.accepted_missing = self.ctx.accepted_fisher + nf * nf
        self.ctx.metric_work = self.ctx.accepted_missing + nf * nf
        return work

    def evaluate(self, columns, logit_columns, int n_free, params, bint rows=False):
        """Evaluate the joint objective's raw geometry at one point.

        Only finiteness is checked; no Newton metric is built, so the point
        may lie in any coordinate system, including all reduced coordinates.

        Returns
        -------
        tuple
            ``(nll, gradient, fisher, missing)``; with ``rows`` also the
            responsibilities ``(N, K)`` and the centered per-row partial
            means ``(N, n_k)`` of every component.

        Raises
        ------
        FloatingPointError
            If the point is not a valid evaluation point.
        """
        cdef list refs = []
        cdef cnp.ndarray _work
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.ascontiguousarray(
            params, dtype=np.float64
        ).reshape(-1)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] gradient = np.zeros(n_free)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] metric = np.zeros((n_free, n_free))
        cdef double nll = 0.0, smallest = 0.0
        cdef int status, k
        cdef Py_ssize_t N = self.ctx.N, n
        if n_free < 1 or x.shape[0] != n_free:
            raise ValueError("invalid joint mixture evaluation point")
        self._bind_face(refs, columns, logit_columns, n_free)
        _work = self._bind_work(n_free)
        self.ctx.raw = True
        with nogil:
            status = _joint_evaluate(
                &self.ctx, &x[0], &nll, &gradient[0], &metric[0, 0], &smallest
            )
        if status != 0:
            raise FloatingPointError(
                f"compiled joint mixture evaluation failed (status {status})"
            )
        fisher = np.asarray(<double[:n_free, :n_free]>self.ctx.fisher).copy()
        missing = np.asarray(<double[:n_free, :n_free]>self.ctx.missing).copy()
        if not rows:
            return float(nll), gradient, fisher, missing
        responsibilities = np.asarray(<double[:self.ctx.K, :N]>self.ctx.resp).T.copy()
        centered = []
        for k in range(self.ctx.K):
            n = self.ctx.comps[k].n
            centered.append(
                np.asarray(<double[:n, :N]>self.ctx.comps[k].centered).T.copy()
            )
        return float(nll), gradient, fisher, missing, responsibilities, tuple(centered)

    def posterior(self, columns, logit_columns, int n_free, params):
        """Return the mixture log likelihood and responsibilities at one point.

        This is the compiled E-step: component normalizers and per-row log
        values feed a per-row log-sum-exp; no moments or information are
        accumulated.  The result equals ``evaluate`` bit for bit.

        Returns
        -------
        tuple
            ``(log_likelihood, responsibilities)`` with responsibilities of
            shape ``(N, K)``.

        Raises
        ------
        FloatingPointError
            If a component is not normalizable or a row has zero likelihood.
        """
        cdef list refs = []
        cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.ascontiguousarray(
            params, dtype=np.float64
        ).reshape(-1)
        cdef int status
        cdef Py_ssize_t N = self.ctx.N
        if n_free < 1 or x.shape[0] != n_free:
            raise ValueError("invalid joint mixture evaluation point")
        self._bind_face(refs, columns, logit_columns, n_free)
        with nogil:
            status = _joint_posterior(&self.ctx, &x[0], False)
        if status != 0:
            raise FloatingPointError(
                f"compiled mixture posterior failed (status {status})"
            )
        responsibilities = np.asarray(<double[:self.ctx.K, :N]>self.ctx.resp).T.copy()
        return float(self.ctx.log_likelihood), responsibilities

    def solve(
        self,
        columns,
        logit_columns,
        int n_free,
        params,
        blocks_packed,
        b_matrix,
        a_packed,
        sizes,
        a_offsets,
        q_offsets,
        reference_dual,
        row_degrees,
        double tolerance,
        double certified_tolerance,
        double accuracy_floor,
        int max_iterations,
        double armijo,
        double backtrack,
        int max_line_search,
        int min_steps=0,
    ):
        """Optimize the joint objective on one exact face.

        Returns
        -------
        tuple
            ``(status, theta, blocks, dual, nll, gradient, metric, fisher,
            missing, smallest, iterations, evaluations, subproblem_iterations,
            bound)``.

        Raises
        ------
        FloatingPointError
            If the start is not a valid evaluation point.
        """
        cdef list refs = []
        cdef FaceObjective objective
        cdef cnp.ndarray _work
        self._bind_face(refs, columns, logit_columns, n_free)
        _work = self._bind_work(n_free)
        self.ctx.raw = False
        objective.ctx = &self.ctx
        objective.evaluate = _joint_evaluate
        objective.accept = _joint_accept
        return _solve_face(
            &objective,
            n_free,
            self.ctx.accepted_fisher,
            self.ctx.accepted_missing,
            params,
            blocks_packed,
            b_matrix,
            a_packed,
            sizes,
            a_offsets,
            q_offsets,
            reference_dual,
            row_degrees,
            tolerance,
            certified_tolerance,
            accuracy_floor,
            max_iterations,
            armijo,
            backtrack,
            max_line_search,
            min_steps,
        )
