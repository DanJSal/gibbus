# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""Compiled adaptive reductions for model-aware interval censoring.

The fitting pipeline repeatedly needs vectors of integrals over censored
intervals: probability mass, centered first moments, centered second moments,
and sometimes extra power moments.  ``scipy.integrate.quad_vec`` can evaluate
that vector, but a Python callback then runs at every adaptive quadrature node.

This module keeps the complete node path in Cython.  Statistics are described by
small fixed descriptors (polynomial, power, lower/upper endpoint log, or
constant), and a globally adaptive Gauss-Kronrod 15/7 rule accumulates the whole
vector in one traversal.  Prepared contexts expose scalar, multi-row, and
weighted-batch entry points; the fitting objective therefore keeps heterogeneous
censoring-row dispatch and weighted statistic accumulation below Python. Infinite
intervals are mapped to ``t in (0, 1)`` by a rational transformation using the
state's local scale.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport (
    cos, exp, INFINITY, isfinite, log, nextafter, sin, sqrt, tan,
)
from libc.stdlib cimport free, malloc

cnp.import_array()

cdef int _STAT_POLY = 0
cdef int _STAT_POWER = 1
cdef int _STAT_LOG_LOWER = 2
cdef int _STAT_LOG_UPPER = 3
cdef int _STAT_CONSTANT = 4

# Positive Kronrod abscissae plus zero, with matching Kronrod weights.
# Gauss-7 nodes are the entries with indices 1, 3, 5, and 7.
cdef double _XGK[8]
_XGK[0] = 0.991455371120812639206854697526329
_XGK[1] = 0.949107912342758524526189684047851
_XGK[2] = 0.864864423359769072789712788640926
_XGK[3] = 0.741531185599394439863864773280788
_XGK[4] = 0.586087235467691130294144838258730
_XGK[5] = 0.405845151377397166906606412076961
_XGK[6] = 0.207784955007898467600689403773245
_XGK[7] = 0.0

# Distance of the outermost Kronrod node from its piece's end, in piece units.
cdef double _GK15_FIRST_NODE = 0.5 * (1.0 - _XGK[0])

cdef double _WGK[8]
_WGK[0] = 0.022935322010529224963732008058970
_WGK[1] = 0.063092092629978553290700663189204
_WGK[2] = 0.104790010322250183839876322541518
_WGK[3] = 0.140653259715525918745189590510238
_WGK[4] = 0.169004726639267902826583426598550
_WGK[5] = 0.190350578064785409913256402421014
_WGK[6] = 0.204432940075298892414161999234649
_WGK[7] = 0.209482141084727828012999174891714

cdef double _WG[4]
_WG[0] = 0.129484966168869693270611432679082
_WG[1] = 0.279705391489276667901467771423780
_WG[2] = 0.381830050505118944950369775488975
_WG[3] = 0.417959183673469387755102040816327


cdef inline double _polyval_ptr(const double* c, Py_ssize_t n, double x) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 0:
        return 0.0
    out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * x + c[i]
    return out


cdef inline double _powi(double x, long k) noexcept nogil:
    cdef double out = 1.0
    cdef double base = x
    cdef long e = k
    while e > 0:
        if e & 1:
            out *= base
        base *= base
        e >>= 1
    return out


cdef inline double _q_eval_ptr(
    const double* q,
    Py_ssize_t nq,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double z,
) noexcept nogil:
    cdef double out = _polyval_ptr(q, nq, z)
    cdef double d
    if isfinite(lower) and isfinite(a_lower) and a_lower > 0.0:
        d = z - lower
        if d <= 0.0:
            return INFINITY
        out -= a_lower * log(d)
    if isfinite(upper) and isfinite(a_upper) and a_upper > 0.0:
        d = upper - z
        if d <= 0.0:
            return INFINITY
        out -= a_upper * log(d)
    return out


cdef inline double _q_d1_eval_ptr(
    const double* q,
    Py_ssize_t nq,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double z,
) noexcept nogil:
    """Evaluate the first derivative of the natural potential."""
    cdef double out = 0.0
    cdef double d
    cdef Py_ssize_t i
    if nq > 1:
        out = (nq - 1) * q[nq - 1]
        for i in range(nq - 2, 0, -1):
            out = out * z + i * q[i]
    if isfinite(lower) and isfinite(a_lower) and a_lower > 0.0:
        d = z - lower
        if d <= 0.0:
            return -INFINITY
        out -= a_lower / d
    if isfinite(upper) and isfinite(a_upper) and a_upper > 0.0:
        d = upper - z
        if d <= 0.0:
            return INFINITY
        out += a_upper / d
    return out


cdef inline double _stat_eval(
    int kind,
    long order,
    const double* coeffs,
    Py_ssize_t ncoeff,
    double lower,
    double upper,
    double z,
) noexcept nogil:
    cdef double d
    if kind == _STAT_POLY:
        return _polyval_ptr(coeffs, ncoeff, z)
    if kind == _STAT_POWER:
        return _powi(z, order)
    if kind == _STAT_LOG_LOWER:
        d = z - lower
        if d <= 0.0:
            return INFINITY
        return -log(d)
    if kind == _STAT_LOG_UPPER:
        d = upper - z
        if d <= 0.0:
            return INFINITY
        return -log(d)
    return 1.0


cdef inline void _map_t(
    double t,
    int transform,
    double lo,
    double hi,
    double scale,
    double* z,
    double* jac,
) noexcept nogil:
    cdef double om
    cdef double angle, c
    if transform == 0:
        z[0] = lo + (hi - lo) * t
        jac[0] = hi - lo
    elif transform == 3:
        angle = 3.141592653589793238462643383279502884 * (t - 0.5)
        c = cos(angle)
        z[0] = lo + scale * tan(angle)
        jac[0] = scale * 3.141592653589793238462643383279502884 / (c * c)
    elif transform == 4:
        z[0] = lo + (hi - lo) * t * t
        jac[0] = 2.0 * (hi - lo) * t
    elif transform == 5:
        om = 1.0 - t
        z[0] = hi - (hi - lo) * om * om
        jac[0] = 2.0 * (hi - lo) * om
    elif transform == 6:
        angle = 0.5 * 3.141592653589793238462643383279502884 * t
        c = sin(angle)
        z[0] = lo + (hi - lo) * c * c
        jac[0] = (
            (hi - lo) * 0.5 * 3.141592653589793238462643383279502884 * sin(2.0 * angle)
        )
    else:
        om = 1.0 - t
        if transform == 1:
            z[0] = lo + scale * t / om
        else:
            z[0] = hi - scale * t / om
        jac[0] = scale / (om * om)


cdef int _node_accumulate(
    double t,
    double wk,
    double wg,
    int transform,
    double lo,
    double hi,
    double scale,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    double q_ref,
    const cnp.int32_t* kinds,
    const cnp.int64_t* orders,
    const double* coeffs,
    const cnp.int32_t* lengths,
    Py_ssize_t coeff_width,
    Py_ssize_t n_primary,
    Py_ssize_t n_extra,
    const double* refs,
    double* centered,
    double* kronrod,
    double* gauss,
) noexcept nogil:
    cdef double z, jac, qz, log_weight, weight, value, cval, common
    cdef Py_ssize_t i, j, off
    _map_t(t, transform, lo, hi, scale, &z, &jac)

    # Roundoff in the finite/rational transforms can land exactly on a support
    # endpoint.  Such a point has zero measure but endpoint-log statistics are
    # singular there, so move to the nearest representable interior point.
    if isfinite(support_lower) and z <= support_lower:
        z = nextafter(
            support_lower, support_upper if isfinite(support_upper) else INFINITY
        )
    if isfinite(support_upper) and z >= support_upper:
        z = nextafter(
            support_upper, support_lower if isfinite(support_lower) else -INFINITY
        )

    qz = _q_eval_ptr(q, nq, support_lower, support_upper, a_lower, a_upper, z)
    if not isfinite(qz):
        # A transformed tail node may overflow far beyond all relevant mass.
        # Treat +inf potential as exact underflow, matching the Python path.
        return 0
    log_weight = q_ref - qz
    if log_weight > 1e-9:
        return 1
    if log_weight > 0.0:
        log_weight = 0.0
    if log_weight < -745.0:
        return 0
    weight = exp(log_weight)
    common = weight * jac
    if not isfinite(common):
        return 2

    kronrod[0] += wk * common
    if wg != 0.0:
        gauss[0] += wg * common

    for i in range(n_primary + n_extra):
        value = _stat_eval(
            kinds[i], orders[i], coeffs + i * coeff_width, lengths[i],
            support_lower, support_upper, z,
        )
        if not isfinite(value):
            return 3
        centered[i] = value - refs[i]

    off = 1
    for i in range(n_primary):
        cval = common * centered[i]
        kronrod[off + i] += wk * cval
        if wg != 0.0:
            gauss[off + i] += wg * cval
    off += n_primary

    for i in range(n_primary):
        for j in range(n_primary):
            cval = common * centered[i] * centered[j]
            kronrod[off + i * n_primary + j] += wk * cval
            if wg != 0.0:
                gauss[off + i * n_primary + j] += wg * cval
    off += n_primary * n_primary

    for i in range(n_extra):
        cval = common * centered[n_primary + i]
        kronrod[off + i] += wk * cval
        if wg != 0.0:
            gauss[off + i] += wg * cval
    return 0


cdef int _gk15(
    double ta,
    double tb,
    int transform,
    double lo,
    double hi,
    double scale,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    double q_ref,
    const cnp.int32_t* kinds,
    const cnp.int64_t* orders,
    const double* coeffs,
    const cnp.int32_t* lengths,
    Py_ssize_t coeff_width,
    Py_ssize_t n_primary,
    Py_ssize_t n_extra,
    const double* refs,
    double* centered,
    double* kronrod,
    double* gauss,
    Py_ssize_t m,
    double* err,
) noexcept nogil:
    cdef double mid = 0.5 * (ta + tb)
    cdef double half = 0.5 * (tb - ta)
    cdef double t, diff, err2 = 0.0
    cdef int i, status
    cdef Py_ssize_t k

    for k in range(m):
        kronrod[k] = 0.0
        gauss[k] = 0.0

    # Center node.
    status = _node_accumulate(
        mid, _WGK[7], _WG[3], transform, lo, hi, scale,
        support_lower, support_upper, q, nq, a_lower, a_upper, q_ref,
        kinds, orders, coeffs, lengths, coeff_width, n_primary, n_extra,
        refs, centered, kronrod, gauss,
    )
    if status != 0:
        return status

    for i in range(7):
        t = half * _XGK[i]
        status = _node_accumulate(
            mid - t,
            _WGK[i],
            _WG[0] if i == 1 else (_WG[1] if i == 3 else (_WG[2] if i == 5 else 0.0)),
            transform,
            lo,
            hi,
            scale,
            support_lower,
            support_upper,
            q,
            nq,
            a_lower,
            a_upper,
            q_ref,
            kinds,
            orders,
            coeffs,
            lengths,
            coeff_width,
            n_primary,
            n_extra,
            refs,
            centered,
            kronrod,
            gauss,
        )
        if status != 0:
            return status
        status = _node_accumulate(
            mid + t,
            _WGK[i],
            _WG[0] if i == 1 else (_WG[1] if i == 3 else (_WG[2] if i == 5 else 0.0)),
            transform,
            lo,
            hi,
            scale,
            support_lower,
            support_upper,
            q,
            nq,
            a_lower,
            a_upper,
            q_ref,
            kinds,
            orders,
            coeffs,
            lengths,
            coeff_width,
            n_primary,
            n_extra,
            refs,
            centered,
            kronrod,
            gauss,
        )
        if status != 0:
            return status

    err2 = 0.0
    for k in range(m):
        kronrod[k] *= half
        gauss[k] *= half
        diff = kronrod[k] - gauss[k]
        err2 += diff * diff
    err[0] = sqrt(err2)
    return 0


# Failure codes for the allocation-free/GIL-free adaptive natural reducer.
cdef enum:
    _ADAPT_BAD_ROW = 10
    _ADAPT_OUTSIDE_SUPPORT = 11
    _ADAPT_BAD_REFERENCE = 12
    _ADAPT_BAD_CENTER = 13
    _ADAPT_BAD_STATISTIC = 14
    _ADAPT_BAD_SHIFT = 15
    _ADAPT_NONFINITE_WEIGHT = 16
    _ADAPT_BAD_MASS = 17
    _ADAPT_BAD_KIND = 18
    _ADAPT_ALLOC = 19


cdef int _adaptive_natural_row(
    double lo,
    double hi,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    double mode,
    double local_scale,
    double log_Z,
    const cnp.int32_t* kinds,
    const cnp.int64_t* orders,
    const double* coeffs,
    const cnp.int32_t* lengths,
    Py_ssize_t coeff_width,
    Py_ssize_t P,
    double epsabs,
    double epsrel,
    int limit,
    double* refs,
    double* centered,
    double* tmp_k,
    double* tmp_g,
    double* vals,
    double* errors,
    double* bounds_a,
    double* bounds_b,
    double* total,
    double* left,
    double* right,
    double* centered_mean,
    double* log_probability_out,
    double* mean_out,
    double* covariance_out,
) noexcept nogil:
    """Reduce one adaptive row for natural first-partial statistics."""
    cdef double map_lo = lo
    cdef double map_hi = hi
    cdef double q_min_x, q_ref, center_x, step, mass, tol
    cdef double total_err, old_err, e1, e2, norm2, mid, old_b
    cdef double tau, node_t, node_z, node_jac, node_q, piece_a, piece_b
    cdef int transform, status, count, idx, i, zoom_end, levels, _rep
    cdef Py_ssize_t j, k, m, off

    if lo != lo or hi != hi or not lo < hi:
        return _ADAPT_BAD_ROW
    if lo < support_lower or hi > support_upper:
        return _ADAPT_OUTSIDE_SUPPORT

    q_min_x = mode
    if isfinite(lo) and q_min_x < lo:
        q_min_x = lo
    if isfinite(hi) and q_min_x > hi:
        q_min_x = hi
    q_ref = _q_eval_ptr(q, nq, support_lower, support_upper, a_lower, a_upper, q_min_x)
    if not isfinite(q_ref):
        return _ADAPT_BAD_REFERENCE

    if isfinite(lo) and isfinite(hi):
        center_x = 0.5 * lo + 0.5 * hi
        if (
            isfinite(support_lower)
            and lo == support_lower
            and isfinite(support_upper)
            and hi == support_upper
        ):
            transform = 6
        elif isfinite(support_lower) and lo == support_lower:
            transform = 4
        elif isfinite(support_upper) and hi == support_upper:
            transform = 5
        else:
            transform = 0
        step = 1.0
    elif isfinite(lo):
        step = local_scale
        if q_min_x == lo:
            node_q = _q_d1_eval_ptr(
                q, nq, support_lower, support_upper, a_lower, a_upper, lo,
            )
            if isfinite(node_q) and node_q > 0.0 and 1.0 / node_q < step:
                step = 1.0 / node_q
        center_x = lo + step
        transform = 1
    elif isfinite(hi):
        step = local_scale
        if q_min_x == hi:
            node_q = _q_d1_eval_ptr(
                q, nq, support_lower, support_upper, a_lower, a_upper, hi,
            )
            if isfinite(node_q) and node_q < 0.0 and -1.0 / node_q < step:
                step = -1.0 / node_q
        center_x = hi - step
        transform = 2
    else:
        step = local_scale
        center_x = mode
        map_lo = mode
        map_hi = mode
        transform = 3
    if isfinite(lo) and not center_x > lo:
        center_x = nextafter(lo, hi if isfinite(hi) else INFINITY)
    if isfinite(hi) and not center_x < hi:
        center_x = nextafter(hi, lo if isfinite(lo) else -INFINITY)
    if not isfinite(center_x):
        return _ADAPT_BAD_CENTER

    for j in range(P):
        refs[j] = _stat_eval(
            kinds[j], orders[j], coeffs + j * coeff_width, lengths[j],
            support_lower, support_upper, center_x,
        )
        if not isfinite(refs[j]):
            return _ADAPT_BAD_STATISTIC

    m = 1 + P + P * P
    zoom_end = -1
    if transform == 0 or transform == 4 or transform == 5 or transform == 6:
        if q_min_x == lo:
            zoom_end = 0
        elif q_min_x == hi:
            zoom_end = 1
    elif transform == 1 and q_min_x == lo:
        zoom_end = 0
    elif transform == 2 and q_min_x == hi:
        zoom_end = 0
    levels = 0
    if zoom_end >= 0:
        tau = 1.0
        while levels + 1 < limit and levels < 64:
            node_t = tau * _GK15_FIRST_NODE
            if zoom_end == 1:
                node_t = 1.0 - node_t
            _map_t(node_t, transform, map_lo, map_hi, step, &node_z, &node_jac)
            node_q = _q_eval_ptr(
                q, nq, support_lower, support_upper, a_lower, a_upper, node_z,
            )
            if isfinite(node_q) and node_q - q_ref <= 1.0:
                break
            tau *= 0.0625
            levels += 1

    count = 0
    total_err = 0.0
    for k in range(m):
        total[k] = 0.0
    for i in range(levels + 1):
        piece_b = 1.0
        for _rep in range(i):
            piece_b *= 0.0625
        piece_a = piece_b * 0.0625 if i < levels else 0.0
        if zoom_end == 1:
            piece_a, piece_b = 1.0 - piece_b, 1.0 - piece_a
        if not piece_a < piece_b:
            continue
        status = _gk15(
            piece_a, piece_b, transform, map_lo, map_hi, step,
            support_lower, support_upper, q, nq, a_lower, a_upper, q_ref,
            kinds, orders, coeffs, lengths, coeff_width, P, 0,
            refs, centered, tmp_k, tmp_g, m, &errors[count],
        )
        if status == 1:
            return _ADAPT_BAD_SHIFT
        if status == 2:
            return _ADAPT_NONFINITE_WEIGHT
        if status == 3:
            return _ADAPT_BAD_STATISTIC
        for k in range(m):
            vals[count * m + k] = tmp_k[k]
            total[k] += tmp_k[k]
        bounds_a[count] = piece_a
        bounds_b[count] = piece_b
        total_err += errors[count]
        count += 1

    while count < limit:
        norm2 = 0.0
        for k in range(m):
            norm2 += total[k] * total[k]
        tol = epsabs
        if epsrel * sqrt(norm2) > tol:
            tol = epsrel * sqrt(norm2)
        if total_err <= tol:
            break

        idx = 0
        for i in range(1, count):
            if errors[i] > errors[idx]:
                idx = i
        old_b = bounds_b[idx]
        mid = 0.5 * (bounds_a[idx] + old_b)
        old_err = errors[idx]

        status = _gk15(
            bounds_a[idx], mid, transform, map_lo, map_hi, step,
            support_lower, support_upper, q, nq, a_lower, a_upper, q_ref,
            kinds, orders, coeffs, lengths, coeff_width, P, 0,
            refs, centered, left, tmp_g, m, &e1,
        )
        if status == 1:
            return _ADAPT_BAD_SHIFT
        if status == 2:
            return _ADAPT_NONFINITE_WEIGHT
        if status == 3:
            return _ADAPT_BAD_STATISTIC
        status = _gk15(
            mid, old_b, transform, map_lo, map_hi, step,
            support_lower, support_upper, q, nq, a_lower, a_upper, q_ref,
            kinds, orders, coeffs, lengths, coeff_width, P, 0,
            refs, centered, right, tmp_g, m, &e2,
        )
        if status == 1:
            return _ADAPT_BAD_SHIFT
        if status == 2:
            return _ADAPT_NONFINITE_WEIGHT
        if status == 3:
            return _ADAPT_BAD_STATISTIC

        for k in range(m):
            total[k] += left[k] + right[k] - vals[idx * m + k]
            vals[idx * m + k] = left[k]
            vals[count * m + k] = right[k]
        bounds_b[idx] = mid
        bounds_a[count] = mid
        bounds_b[count] = old_b
        errors[idx] = e1
        errors[count] = e2
        total_err += e1 + e2 - old_err
        if total_err < 0.0:
            total_err = 0.0
        count += 1

    mass = total[0]
    if not (mass > 0.0 and isfinite(mass)):
        return _ADAPT_BAD_MASS

    off = 1
    for j in range(P):
        centered_mean[j] = total[off + j] / mass
        mean_out[j] = refs[j] + centered_mean[j]
    off += P
    for j in range(P):
        for k in range(P):
            covariance_out[j * P + k] = (
                total[off + j * P + k] / mass
                - centered_mean[j] * centered_mean[k]
            )
    for j in range(P):
        for k in range(j + 1, P):
            covariance_out[j * P + k] = covariance_out[k * P + j] = 0.5 * (
                covariance_out[j * P + k] + covariance_out[k * P + j]
            )
    log_probability_out[0] = log(mass) - q_ref - log_Z
    return 0


cdef api int adaptive_natural_objective_c(
    Py_ssize_t R,
    Py_ssize_t P,
    Py_ssize_t W,
    Py_ssize_t nq,
    const double* intervals,
    const double* row_weights,
    const double* q_poly,
    double a_lower,
    double a_upper,
    double mode,
    double local_scale,
    double log_Z,
    const int* natural_kinds,
    const int* natural_lengths,
    const double* coefficients,
    double support_lower,
    double support_upper,
    double epsabs,
    double epsrel,
    int limit,
    double* log_probability,
    double* obs_h,
    double* obs_cov,
    double* nll_io,
    double* row_means,
    double* row_cov,
) noexcept nogil:
    """Accumulate adaptive interval likelihood geometry without Python objects.

    ``obs_h``, ``obs_cov`` and ``nll_io`` are additive accumulators so callers
    can combine ordinary finite rows, adaptive rows and whole-support rows in
    one objective evaluation.  Optional feature-major ``row_means``
    (``P x R``) and ``row_cov`` (``P*P x R``) receive each row's conditional
    moments; pass ``NULL`` to skip them.
    """
    cdef Py_ssize_t m = 1 + P + P * P
    cdef Py_ssize_t doubles_n = (
        2 * P + 2 * m + limit * m + 3 * limit + 3 * m + P + P + P * P
    )
    cdef double* arena = NULL
    cdef double* ptr
    cdef double* refs
    cdef double* centered
    cdef double* tmp_k
    cdef double* tmp_g
    cdef double* vals
    cdef double* errors
    cdef double* bounds_a
    cdef double* bounds_b
    cdef double* total
    cdef double* left
    cdef double* right
    cdef double* centered_mean
    cdef double* mean_tmp
    cdef double* cov_tmp
    cdef cnp.int32_t* kinds = NULL
    cdef cnp.int32_t* lengths = NULL
    cdef cnp.int64_t* orders = NULL
    cdef Py_ssize_t r, i, j
    cdef int status
    cdef double w

    if R < 1 or P < 1 or W < 1 or nq < 1 or limit < 1:
        return _ADAPT_BAD_ROW
    arena = <double*>malloc(doubles_n * sizeof(double))
    kinds = <cnp.int32_t*>malloc(P * sizeof(cnp.int32_t))
    lengths = <cnp.int32_t*>malloc(P * sizeof(cnp.int32_t))
    orders = <cnp.int64_t*>malloc(P * sizeof(cnp.int64_t))
    if arena == NULL or kinds == NULL or lengths == NULL or orders == NULL:
        if arena != NULL:
            free(arena)
        if kinds != NULL:
            free(kinds)
        if lengths != NULL:
            free(lengths)
        if orders != NULL:
            free(orders)
        return _ADAPT_ALLOC

    ptr = arena
    refs = ptr
    ptr += P
    centered = ptr
    ptr += P
    tmp_k = ptr
    ptr += m
    tmp_g = ptr
    ptr += m
    vals = ptr
    ptr += limit * m
    errors = ptr
    ptr += limit
    bounds_a = ptr
    ptr += limit
    bounds_b = ptr
    ptr += limit
    total = ptr
    ptr += m
    left = ptr
    ptr += m
    right = ptr
    ptr += m
    centered_mean = ptr
    ptr += P
    mean_tmp = ptr
    ptr += P
    cov_tmp = ptr

    for i in range(P):
        if natural_kinds[i] == 0:
            kinds[i] = _STAT_POLY
        elif natural_kinds[i] == 1:
            kinds[i] = _STAT_LOG_LOWER
        elif natural_kinds[i] == 2:
            kinds[i] = _STAT_LOG_UPPER
        else:
            free(arena)
            free(kinds)
            free(lengths)
            free(orders)
            return _ADAPT_BAD_KIND
        lengths[i] = <cnp.int32_t>natural_lengths[i]
        orders[i] = 0

    if not (isfinite(local_scale) and local_scale > 0.0):
        local_scale = 1.0
    for r in range(R):
        status = _adaptive_natural_row(
            intervals[2 * r], intervals[2 * r + 1],
            support_lower, support_upper,
            q_poly, nq, a_lower, a_upper, mode, local_scale, log_Z,
            kinds, orders, coefficients, lengths, W, P, epsabs, epsrel, limit,
            refs, centered, tmp_k, tmp_g, vals, errors, bounds_a, bounds_b,
            total, left, right, centered_mean,
            &log_probability[r], mean_tmp, cov_tmp,
        )
        if status != 0:
            free(arena)
            free(kinds)
            free(lengths)
            free(orders)
            return status
        w = row_weights[r]
        nll_io[0] -= w * log_probability[r]
        for i in range(P):
            obs_h[i] += w * mean_tmp[i]
            if row_means != NULL:
                row_means[i * R + r] = mean_tmp[i]
            for j in range(P):
                obs_cov[i * P + j] += w * cov_tmp[i * P + j]
                if row_cov != NULL:
                    row_cov[(i * P + j) * R + r] = cov_tmp[i * P + j]

    free(arena)
    free(kinds)
    free(lengths)
    free(orders)
    return 0

cdef class AdaptiveIntervalIntegrator:
    """Prepared compiled interval-reduction context for one live model state."""

    cdef object _q_arr
    cdef object _kinds_arr
    cdef object _orders_arr
    cdef object _coeffs_arr
    cdef object _lengths_arr
    cdef double* _q
    cdef cnp.int32_t* _kinds
    cdef cnp.int64_t* _orders
    cdef double* _coeffs
    cdef cnp.int32_t* _lengths
    cdef Py_ssize_t _nq
    cdef Py_ssize_t _n_total
    cdef Py_ssize_t _n_primary
    cdef Py_ssize_t _n_extra
    cdef Py_ssize_t _coeff_width
    cdef double _lower
    cdef double _upper
    cdef double _a_lower
    cdef double _a_upper
    cdef double _mode
    cdef double _local_scale
    cdef double _log_Z
    cdef double _epsabs
    cdef double _epsrel
    cdef int _limit

    def __init__(
        self,
        q_poly,
        support,
        boundary_amplitudes,
        double mode,
        double local_scale,
        double log_Z,
        kinds,
        orders,
        coefficients,
        lengths,
        int n_primary,
        *,
        double epsabs,
        double epsrel,
        int limit,
    ):
        """Prepare adaptive conditional-statistic reduction for one model state.

        Parameters
        ----------
        q_poly : array_like
            Nonempty finite potential coefficients in increasing power order.
        support, boundary_amplitudes : array_like, shape (2,)
            Support bounds and lower/upper logarithmic amplitudes in the same
            coordinates as the polynomial and censoring rows.
        mode : float
            Model mode used to anchor the adaptive integration coordinates.
        local_scale : float
            Positive local scale for tail integration.
        log_Z : float
            Log normalizer of the described unnormalized density.
        kinds, orders : array_like
            Statistic kind codes and their corresponding power orders.
            Kind codes are exposed by ``statistic_kinds``.
        coefficients : array_like, shape (n_statistics, width)
            Padded increasing-power polynomial coefficient rows.
        lengths : array_like, shape (n_statistics,)
            Valid polynomial length of each descriptor row.
        n_primary : int
            Leading descriptors whose conditional means/covariance are returned;
            remaining descriptors produce extra conditional means only.
        epsabs, epsrel : float
            Absolute and relative adaptive integration tolerances.
        limit : int
            Maximum adaptive panel count.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] qarr
        cdef cnp.ndarray[cnp.int32_t, ndim=1] karr
        cdef cnp.ndarray[cnp.int64_t, ndim=1] oarr
        cdef cnp.ndarray[cnp.float64_t, ndim=2] carr
        cdef cnp.ndarray[cnp.int32_t, ndim=1] larr
        cdef cnp.ndarray[cnp.float64_t, ndim=1] supp
        cdef cnp.ndarray[cnp.float64_t, ndim=1] amps

        qarr = np.ascontiguousarray(q_poly, dtype=np.float64).reshape(-1)
        karr = np.ascontiguousarray(kinds, dtype=np.int32).reshape(-1)
        oarr = np.ascontiguousarray(orders, dtype=np.int64).reshape(-1)
        carr = np.ascontiguousarray(coefficients, dtype=np.float64)
        larr = np.ascontiguousarray(lengths, dtype=np.int32).reshape(-1)
        supp = np.ascontiguousarray(support, dtype=np.float64).reshape(-1)
        amps = np.ascontiguousarray(boundary_amplitudes, dtype=np.float64).reshape(-1)

        if qarr.shape[0] < 1 or not np.all(np.isfinite(qarr)):
            raise ValueError("q_poly must be a non-empty finite vector")
        if supp.shape[0] != 2 or amps.shape[0] != 2:
            raise ValueError("support and boundary_amplitudes must have length 2")
        if karr.shape[0] != oarr.shape[0] or karr.shape[0] != larr.shape[0]:
            raise ValueError("statistic descriptor arrays must have matching lengths")
        if carr.ndim != 2 or carr.shape[0] != karr.shape[0] or carr.shape[1] < 1:
            raise ValueError("coefficients must be a non-empty 2-D descriptor matrix")
        if n_primary < 0 or n_primary > karr.shape[0]:
            raise ValueError("n_primary is out of range")
        if limit < 1 or epsabs < 0.0 or epsrel < 0.0:
            raise ValueError("invalid adaptive quadrature controls")

        self._q_arr = qarr
        self._kinds_arr = karr
        self._orders_arr = oarr
        self._coeffs_arr = carr
        self._lengths_arr = larr
        self._q = &qarr[0]
        self._kinds = &karr[0] if karr.shape[0] else NULL
        self._orders = &oarr[0] if oarr.shape[0] else NULL
        self._coeffs = &carr[0, 0]
        self._lengths = &larr[0] if larr.shape[0] else NULL
        self._nq = qarr.shape[0]
        self._n_total = karr.shape[0]
        self._n_primary = n_primary
        self._n_extra = self._n_total - n_primary
        self._coeff_width = carr.shape[1]
        self._lower = float(supp[0])
        self._upper = float(supp[1])
        self._a_lower = float(amps[0])
        self._a_upper = float(amps[1])
        self._mode = mode
        self._local_scale = (
            local_scale if isfinite(local_scale) and local_scale > 0.0 else 1.0
        )
        self._log_Z = log_Z
        self._epsabs = epsabs
        self._epsrel = epsrel
        self._limit = limit

    cdef int _reduce_bounds_into(
        self,
        double lo,
        double hi,
        double* log_probability_out,
        double* mean_out,
        double* covariance_out,
        double* extra_mean_out,
    ) except -1:
        """Reduce one validated endpoint pair into caller-owned output buffers."""
        cdef double map_lo = lo
        cdef double map_hi = hi
        cdef double q_min_x, q_ref, center_x, step, mass, tol
        cdef double total_err, old_err, e1, e2, norm2, mid, old_b
        cdef double tau, node_t, node_z, node_jac, node_q, piece_a, piece_b
        cdef int transform, status, count, idx, i, zoom_end, levels, _rep
        cdef Py_ssize_t j, k, m, off, alloc_n
        cdef double* refs = NULL
        cdef double* centered = NULL
        cdef double* tmp_k = NULL
        cdef double* tmp_g = NULL
        cdef double* vals = NULL
        cdef double* errors = NULL
        cdef double* bounds_a = NULL
        cdef double* bounds_b = NULL
        cdef double* total = NULL
        cdef double* left = NULL
        cdef double* right = NULL
        cdef double* centered_mean = NULL

        if lo != lo or hi != hi or not lo < hi:
            raise ValueError("adaptive interval must be a positive-width ordered pair")
        if lo < self._lower or hi > self._upper:
            raise ValueError("adaptive interval lies outside the model support")

        q_min_x = self._mode
        if isfinite(lo) and q_min_x < lo:
            q_min_x = lo
        if isfinite(hi) and q_min_x > hi:
            q_min_x = hi
        q_ref = _q_eval_ptr(
            self._q, self._nq, self._lower, self._upper,
            self._a_lower, self._a_upper, q_min_x,
        )
        if not isfinite(q_ref):
            raise RuntimeError("interval constrained potential minimum is non-finite")

        if isfinite(lo) and isfinite(hi):
            center_x = 0.5 * lo + 0.5 * hi
            if (
                isfinite(self._lower)
                and lo == self._lower
                and isfinite(self._upper)
                and hi == self._upper
            ):
                transform = 6
            elif isfinite(self._lower) and lo == self._lower:
                transform = 4
            elif isfinite(self._upper) and hi == self._upper:
                transform = 5
            else:
                transform = 0
            step = 1.0
        elif isfinite(lo):
            step = self._local_scale
            if q_min_x == lo:
                node_q = _q_d1_eval_ptr(
                    self._q, self._nq, self._lower, self._upper,
                    self._a_lower, self._a_upper, lo,
                )
                if isfinite(node_q) and node_q > 0.0 and 1.0 / node_q < step:
                    step = 1.0 / node_q
            center_x = lo + step
            transform = 1
        elif isfinite(hi):
            step = self._local_scale
            if q_min_x == hi:
                node_q = _q_d1_eval_ptr(
                    self._q, self._nq, self._lower, self._upper,
                    self._a_lower, self._a_upper, hi,
                )
                if isfinite(node_q) and node_q < 0.0 and -1.0 / node_q < step:
                    step = -1.0 / node_q
            center_x = hi - step
            transform = 2
        else:
            step = self._local_scale
            center_x = self._mode
            map_lo = self._mode
            map_hi = self._mode
            transform = 3
        if isfinite(lo) and not center_x > lo:
            center_x = nextafter(lo, hi if isfinite(hi) else INFINITY)
        if isfinite(hi) and not center_x < hi:
            center_x = nextafter(hi, lo if isfinite(lo) else -INFINITY)
        if not isfinite(center_x):
            raise RuntimeError("interval interior reference is non-finite")

        m = 1 + self._n_primary + self._n_primary * self._n_primary + self._n_extra
        alloc_n = self._n_total if self._n_total > 0 else 1
        refs = <double*>malloc(alloc_n * sizeof(double))
        centered = <double*>malloc(alloc_n * sizeof(double))
        tmp_k = <double*>malloc(m * sizeof(double))
        tmp_g = <double*>malloc(m * sizeof(double))
        vals = <double*>malloc(self._limit * m * sizeof(double))
        errors = <double*>malloc(self._limit * sizeof(double))
        bounds_a = <double*>malloc(self._limit * sizeof(double))
        bounds_b = <double*>malloc(self._limit * sizeof(double))
        total = <double*>malloc(m * sizeof(double))
        left = <double*>malloc(m * sizeof(double))
        right = <double*>malloc(m * sizeof(double))
        centered_mean = <double*>malloc(
            (self._n_primary if self._n_primary > 0 else 1) * sizeof(double)
        )
        if (
            refs == NULL
            or centered == NULL
            or tmp_k == NULL
            or tmp_g == NULL
            or vals == NULL
            or errors == NULL
            or bounds_a == NULL
            or bounds_b == NULL
            or total == NULL
            or left == NULL
            or right == NULL
            or centered_mean == NULL
        ):
            if refs != NULL:
                free(refs)
            if centered != NULL:
                free(centered)
            if tmp_k != NULL:
                free(tmp_k)
            if tmp_g != NULL:
                free(tmp_g)
            if vals != NULL:
                free(vals)
            if errors != NULL:
                free(errors)
            if bounds_a != NULL:
                free(bounds_a)
            if bounds_b != NULL:
                free(bounds_b)
            if total != NULL:
                free(total)
            if left != NULL:
                free(left)
            if right != NULL:
                free(right)
            if centered_mean != NULL:
                free(centered_mean)
            raise MemoryError("adaptive interval quadrature allocation failed")

        try:
            for j in range(self._n_total):
                refs[j] = _stat_eval(
                    self._kinds[j], self._orders[j],
                    self._coeffs + j * self._coeff_width, self._lengths[j],
                    self._lower, self._upper, center_x,
                )
                if not isfinite(refs[j]):
                    raise RuntimeError(
                        "interval statistic is non-finite at interior reference"
                    )

            # When the constrained minimum is an endpoint, the potential can
            # rise so steeply that the Gauss-Kronrod nodes, none of which sits
            # on the endpoint, see only e^-hundreds of the peak (or underflow
            # entirely).  The rule then reports a tiny mass with a tiny error
            # and is accepted.  Seed the partition with pieces graded
            # geometrically toward that endpoint until the innermost piece's
            # first node lies within one unit of potential of the minimum.
            zoom_end = -1
            if transform == 0 or transform == 4 or transform == 5 or transform == 6:
                if q_min_x == lo:
                    zoom_end = 0
                elif q_min_x == hi:
                    zoom_end = 1
            elif transform == 1 and q_min_x == lo:
                zoom_end = 0
            elif transform == 2 and q_min_x == hi:
                zoom_end = 0
            levels = 0
            if zoom_end >= 0:
                tau = 1.0
                while levels + 1 < self._limit and levels < 64:
                    node_t = tau * _GK15_FIRST_NODE
                    if zoom_end == 1:
                        node_t = 1.0 - node_t
                    _map_t(node_t, transform, map_lo, map_hi, step, &node_z, &node_jac)
                    node_q = _q_eval_ptr(
                        self._q, self._nq, self._lower, self._upper,
                        self._a_lower, self._a_upper, node_z,
                    )
                    if isfinite(node_q) and node_q - q_ref <= 1.0:
                        break
                    tau *= 0.0625
                    levels += 1

            count = 0
            total_err = 0.0
            for k in range(m):
                total[k] = 0.0
            for i in range(levels + 1):
                # Piece i spans [16^-(i+1), 16^-i] measured from the zoom end,
                # except the innermost, which reaches the endpoint itself.
                piece_b = 1.0
                for _rep in range(i):
                    piece_b *= 0.0625
                piece_a = piece_b * 0.0625 if i < levels else 0.0
                if zoom_end == 1:
                    piece_a, piece_b = 1.0 - piece_b, 1.0 - piece_a
                if not piece_a < piece_b:
                    continue
                with nogil:
                    status = _gk15(
                        piece_a, piece_b, transform, map_lo, map_hi, step,
                        self._lower, self._upper, self._q, self._nq,
                        self._a_lower, self._a_upper, q_ref,
                        self._kinds, self._orders, self._coeffs, self._lengths,
                        self._coeff_width, self._n_primary, self._n_extra,
                        refs, centered, tmp_k, tmp_g, m, &errors[count],
                    )
                if status == 1:
                    raise RuntimeError(
                        "adaptive interval shift is not a potential minimum"
                    )
                if status == 2:
                    raise RuntimeError(
                        "adaptive interval integration produced non-finite weight"
                    )
                if status == 3:
                    raise RuntimeError("interval statistic is non-finite")
                for k in range(m):
                    vals[count * m + k] = tmp_k[k]
                    total[k] += tmp_k[k]
                bounds_a[count] = piece_a
                bounds_b[count] = piece_b
                total_err += errors[count]
                count += 1

            while count < self._limit:
                norm2 = 0.0
                for k in range(m):
                    norm2 += total[k] * total[k]
                tol = self._epsabs
                if self._epsrel * sqrt(norm2) > tol:
                    tol = self._epsrel * sqrt(norm2)
                if total_err <= tol:
                    break

                idx = 0
                for i in range(1, count):
                    if errors[i] > errors[idx]:
                        idx = i
                old_b = bounds_b[idx]
                mid = 0.5 * (bounds_a[idx] + old_b)
                old_err = errors[idx]

                with nogil:
                    status = _gk15(
                        bounds_a[idx], mid, transform, map_lo, map_hi, step,
                        self._lower, self._upper, self._q, self._nq,
                        self._a_lower, self._a_upper, q_ref,
                        self._kinds, self._orders, self._coeffs, self._lengths,
                        self._coeff_width, self._n_primary, self._n_extra,
                        refs, centered, left, tmp_g, m, &e1,
                    )
                if status == 1:
                    raise RuntimeError(
                        "adaptive interval shift is not a potential minimum"
                    )
                if status == 2:
                    raise RuntimeError(
                        "adaptive interval integration produced non-finite weight"
                    )
                if status == 3:
                    raise RuntimeError("interval statistic is non-finite")

                with nogil:
                    status = _gk15(
                        mid, old_b, transform, map_lo, map_hi, step,
                        self._lower, self._upper, self._q, self._nq,
                        self._a_lower, self._a_upper, q_ref,
                        self._kinds, self._orders, self._coeffs, self._lengths,
                        self._coeff_width, self._n_primary, self._n_extra,
                        refs, centered, right, tmp_g, m, &e2,
                    )
                if status == 1:
                    raise RuntimeError(
                        "adaptive interval shift is not a potential minimum"
                    )
                if status == 2:
                    raise RuntimeError(
                        "adaptive interval integration produced non-finite weight"
                    )
                if status == 3:
                    raise RuntimeError("interval statistic is non-finite")

                for k in range(m):
                    total[k] += left[k] + right[k] - vals[idx * m + k]
                    vals[idx * m + k] = left[k]
                    vals[count * m + k] = right[k]
                bounds_b[idx] = mid
                bounds_a[count] = mid
                bounds_b[count] = old_b
                errors[idx] = e1
                errors[count] = e2
                total_err += e1 + e2 - old_err
                if total_err < 0.0:
                    total_err = 0.0
                count += 1

            mass = total[0]
            if not (mass > 0.0 and isfinite(mass)):
                raise RuntimeError(
                    "adaptive interval integration produced invalid mass"
                )

            off = 1
            for j in range(self._n_primary):
                centered_mean[j] = total[off + j] / mass
                mean_out[j] = refs[j] + centered_mean[j]
            off += self._n_primary
            for j in range(self._n_primary):
                for k in range(self._n_primary):
                    covariance_out[j * self._n_primary + k] = (
                        total[off + j * self._n_primary + k] / mass
                        - centered_mean[j] * centered_mean[k]
                    )
            for j in range(self._n_primary):
                for k in range(j + 1, self._n_primary):
                    covariance_out[j * self._n_primary + k] = covariance_out[k * self._n_primary + j] = 0.5 * (
                        covariance_out[j * self._n_primary + k]
                        + covariance_out[k * self._n_primary + j]
                    )
            off += self._n_primary * self._n_primary
            for j in range(self._n_extra):
                extra_mean_out[j] = refs[self._n_primary + j] + total[off + j] / mass

            log_probability_out[0] = log(mass) - q_ref - self._log_Z
            return 0
        finally:
            free(refs)
            free(centered)
            free(tmp_k)
            free(tmp_g)
            free(vals)
            free(errors)
            free(bounds_a)
            free(bounds_b)
            free(total)
            free(left)
            free(right)
            free(centered_mean)

    def reduce(self, interval):
        """Return ``(log_probability, mean, covariance, extra_mean)``.

        Parameters
        ----------
        interval : array_like, shape (2,)
            Ordered censoring bounds in the prepared model's coordinates;
            endpoints may be infinite.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] row = np.asarray(
            interval, dtype=np.float64
        ).reshape(-1)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] mean_arr
        cdef cnp.ndarray[cnp.float64_t, ndim=2] cov_arr
        cdef cnp.ndarray[cnp.float64_t, ndim=1] extra_arr
        cdef double log_probability
        cdef double* mean_ptr = NULL
        cdef double* cov_ptr = NULL
        cdef double* extra_ptr = NULL

        if row.shape[0] != 2:
            raise ValueError("adaptive interval must contain two endpoints")
        mean_arr = np.empty(self._n_primary, dtype=np.float64)
        cov_arr = np.empty((self._n_primary, self._n_primary), dtype=np.float64)
        extra_arr = np.empty(self._n_extra, dtype=np.float64)
        if self._n_primary:
            mean_ptr = &mean_arr[0]
            cov_ptr = &cov_arr[0, 0]
        if self._n_extra:
            extra_ptr = &extra_arr[0]
        self._reduce_bounds_into(
            float(row[0]), float(row[1]), &log_probability,
            mean_ptr, cov_ptr, extra_ptr,
        )
        return float(log_probability), mean_arr, cov_arr, extra_arr

    def reduce_many(self, intervals):
        """Reduce many intervals in one Cython dispatch.

        Returns ``(log_probability, mean, covariance, extra_mean)`` with the
        leading dimension indexing input intervals.  Adaptive row traversal and
        statistic reduction stay below the Python boundary for the whole batch.

        Parameters
        ----------
        intervals : array_like, shape (R, 2)
            Ordered censoring rows in the prepared model's coordinates.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rows = np.ascontiguousarray(
            intervals, dtype=np.float64
        )
        cdef Py_ssize_t n, i
        cdef cnp.ndarray[cnp.float64_t, ndim=1] logp
        cdef cnp.ndarray[cnp.float64_t, ndim=2] mean_arr
        cdef cnp.ndarray[cnp.float64_t, ndim=3] cov_arr
        cdef cnp.ndarray[cnp.float64_t, ndim=2] extra_arr
        cdef double* mean_ptr = NULL
        cdef double* cov_ptr = NULL
        cdef double* extra_ptr = NULL

        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError("adaptive intervals must have shape (n, 2)")
        n = rows.shape[0]
        logp = np.empty(n, dtype=np.float64)
        mean_arr = np.empty((n, self._n_primary), dtype=np.float64)
        cov_arr = np.empty((n, self._n_primary, self._n_primary), dtype=np.float64)
        extra_arr = np.empty((n, self._n_extra), dtype=np.float64)
        for i in range(n):
            mean_ptr = &mean_arr[i, 0] if self._n_primary else NULL
            cov_ptr = &cov_arr[i, 0, 0] if self._n_primary else NULL
            extra_ptr = &extra_arr[i, 0] if self._n_extra else NULL
            self._reduce_bounds_into(
                rows[i, 0], rows[i, 1], &logp[i], mean_ptr, cov_ptr, extra_ptr,
            )
        return logp, mean_arr, cov_arr, extra_arr

    def reduce_weighted(self, intervals, weights):
        """Reduce many intervals and accumulate weighted conditional statistics.

        Returns ``(log_probability, weighted_mean, weighted_covariance,
        weighted_extra_mean)``.  Only the per-row log probabilities are
        materialized; conditional statistic arrays are accumulated directly.

        Parameters
        ----------
        intervals : array_like, shape (R, 2)
            Ordered censoring rows in the prepared model's coordinates.
        weights : array_like, shape (R,)
            Finite row weights used as supplied, without normalization.
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rows = np.ascontiguousarray(
            intervals, dtype=np.float64
        )
        cdef cnp.ndarray[cnp.float64_t, ndim=1] w = np.ascontiguousarray(
            weights, dtype=np.float64
        ).reshape(-1)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] logp
        cdef cnp.ndarray[cnp.float64_t, ndim=1] mean_sum
        cdef cnp.ndarray[cnp.float64_t, ndim=2] cov_sum
        cdef cnp.ndarray[cnp.float64_t, ndim=1] extra_sum
        cdef Py_ssize_t n, i, j, k
        cdef double weight
        cdef double* mean_tmp = NULL
        cdef double* cov_tmp = NULL
        cdef double* extra_tmp = NULL

        if rows.ndim != 2 or rows.shape[1] != 2:
            raise ValueError("adaptive intervals must have shape (n, 2)")
        n = rows.shape[0]
        if w.shape[0] != n:
            raise ValueError("adaptive interval weights must match the row count")
        if not np.all(np.isfinite(w)):
            raise ValueError("adaptive interval weights must be finite")

        logp = np.empty(n, dtype=np.float64)
        mean_sum = np.zeros(self._n_primary, dtype=np.float64)
        cov_sum = np.zeros((self._n_primary, self._n_primary), dtype=np.float64)
        extra_sum = np.zeros(self._n_extra, dtype=np.float64)
        if self._n_primary:
            mean_tmp = <double*>malloc(self._n_primary * sizeof(double))
            cov_tmp = <double*>malloc(
                self._n_primary * self._n_primary * sizeof(double)
            )
        if self._n_extra:
            extra_tmp = <double*>malloc(self._n_extra * sizeof(double))
        if ((self._n_primary and (mean_tmp == NULL or cov_tmp == NULL)) or
                (self._n_extra and extra_tmp == NULL)):
            if mean_tmp != NULL:
                free(mean_tmp)
            if cov_tmp != NULL:
                free(cov_tmp)
            if extra_tmp != NULL:
                free(extra_tmp)
            raise MemoryError("adaptive interval batch allocation failed")
        try:
            for i in range(n):
                self._reduce_bounds_into(
                    rows[i, 0], rows[i, 1], &logp[i],
                    mean_tmp, cov_tmp, extra_tmp,
                )
                weight = w[i]
                for j in range(self._n_primary):
                    mean_sum[j] += weight * mean_tmp[j]
                    for k in range(self._n_primary):
                        cov_sum[j, k] += weight * cov_tmp[j * self._n_primary + k]
                for j in range(self._n_extra):
                    extra_sum[j] += weight * extra_tmp[j]
            return logp, mean_sum, cov_sum, extra_sum
        finally:
            if mean_tmp != NULL:
                free(mean_tmp)
            if cov_tmp != NULL:
                free(cov_tmp)
            if extra_tmp != NULL:
                free(extra_tmp)


def statistic_kinds():
    """Return the integer descriptor codes used by the Python adapter."""
    return {
        "poly": _STAT_POLY,
        "power": _STAT_POWER,
        "lower_log": _STAT_LOG_LOWER,
        "upper_log": _STAT_LOG_UPPER,
        "constant": _STAT_CONSTANT,
    }
