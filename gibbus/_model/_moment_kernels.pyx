# cython: language_level=3
"""Small fused moment contractions used by the fit-time Fisher geometry."""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def product_moment(double[::1] a, double[::1] b, double[::1] moments):
    """Return ``sum_ij a[i] * b[j] * moments[i+j]`` without convolution."""
    cdef Py_ssize_t na = a.shape[0]
    cdef Py_ssize_t nb = b.shape[0]
    cdef Py_ssize_t i, j
    cdef double total = 0.0
    if na == 0 or nb == 0:
        return 0.0
    if moments.shape[0] < na + nb - 1:
        raise ValueError("moments array is too short for coefficient product")
    for i in range(na):
        for j in range(nb):
            total += a[i] * b[j] * moments[i + j]
    return total


# Gauss--Kronrod 15/7 constants for simultaneous power-moment integration.
from libc.math cimport exp, fabs, INFINITY, isfinite, log, nextafter
from libc.stdlib cimport free, malloc

cdef double _PM_XGK[8]
_PM_XGK[0] = 0.991455371120812639206854697526329
_PM_XGK[1] = 0.949107912342758524526189684047851
_PM_XGK[2] = 0.864864423359769072789712788640926
_PM_XGK[3] = 0.741531185599394439863864773280788
_PM_XGK[4] = 0.586087235467691130294144838258730
_PM_XGK[5] = 0.405845151377397166906606412076961
_PM_XGK[6] = 0.207784955007898467600689403773245
_PM_XGK[7] = 0.0
cdef double _PM_WGK[8]
_PM_WGK[0] = 0.022935322010529224963732008058970
_PM_WGK[1] = 0.063092092629978553290700663189204
_PM_WGK[2] = 0.104790010322250183839876322541518
_PM_WGK[3] = 0.140653259715525918745189703095821
_PM_WGK[4] = 0.169004726639267902826583426598550
_PM_WGK[5] = 0.190350578064785409913256402421014
_PM_WGK[6] = 0.204432940075298892414161999234649
_PM_WGK[7] = 0.209482141084727828012999174891714
cdef double _PM_WG[4]
_PM_WG[0] = 0.129484966168869693270611432679082
_PM_WG[1] = 0.279705391489276667901467771423780
_PM_WG[2] = 0.381830050505118944950369775488975
_PM_WG[3] = 0.417959183673469387755102040816327


cdef inline double _pm_polyval(const double* c, Py_ssize_t n, double z) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * z + c[i]
    return out


cdef inline int _pm_node(
    double t,
    double wk,
    double wg,
    int transform,
    double lo,
    double hi,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    Py_ssize_t m,
    double* kronrod,
    double* gauss,
) noexcept nogil:
    cdef double z, jac, om, qz, d, weight, power, common
    cdef Py_ssize_t k
    if transform == 1:
        z = lo + (hi - lo) * t * t
        jac = 2.0 * (hi - lo) * t
    elif transform == 2:
        om = 1.0 - t
        z = hi - (hi - lo) * om * om
        jac = 2.0 * (hi - lo) * om
    else:
        z = lo + (hi - lo) * t
        jac = hi - lo

    if isfinite(support_lower) and z <= support_lower:
        z = nextafter(
            support_lower, support_upper if isfinite(support_upper) else INFINITY
        )
    if isfinite(support_upper) and z >= support_upper:
        z = nextafter(
            support_upper, support_lower if isfinite(support_lower) else -INFINITY
        )

    qz = _pm_polyval(q, nq, z)
    if isfinite(support_lower) and isfinite(a_lower) and a_lower > 0.0:
        d = z - support_lower
        if d <= 0.0:
            return 1
        qz -= a_lower * log(d)
    if isfinite(support_upper) and isfinite(a_upper) and a_upper > 0.0:
        d = support_upper - z
        if d <= 0.0:
            return 1
        qz -= a_upper * log(d)
    if not isfinite(qz):
        return 0
    if qz < -1e-8:
        return 2
    if qz < 0.0:
        qz = 0.0
    if qz > 745.0:
        return 0
    weight = exp(-qz)
    common = weight * jac
    if not isfinite(common):
        return 3
    power = 1.0
    for k in range(m):
        kronrod[k] += wk * common * power
        if wg != 0.0:
            gauss[k] += wg * common * power
        power *= z
    return 0


cdef int _pm_gk15(
    double ta,
    double tb,
    int transform,
    double lo,
    double hi,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    Py_ssize_t m,
    double* kronrod,
    double* gauss,
    double* error,
) noexcept nogil:
    cdef double mid = 0.5 * (ta + tb)
    cdef double half = 0.5 * (tb - ta)
    cdef double dt, wg
    cdef Py_ssize_t k
    cdef int i, status
    for k in range(m):
        kronrod[k] = 0.0
        gauss[k] = 0.0
    status = _pm_node(
        mid, _PM_WGK[7], _PM_WG[3], transform, lo, hi,
        support_lower, support_upper, q, nq, a_lower, a_upper,
        m, kronrod, gauss,
    )
    if status != 0:
        return status
    for i in range(7):
        dt = half * _PM_XGK[i]
        wg = (
            _PM_WG[0]
            if i == 1
            else (_PM_WG[1] if i == 3 else (_PM_WG[2] if i == 5 else 0.0))
        )
        status = _pm_node(
            mid - dt, _PM_WGK[i], wg, transform, lo, hi,
            support_lower, support_upper, q, nq, a_lower, a_upper,
            m, kronrod, gauss,
        )
        if status != 0:
            return status
        status = _pm_node(
            mid + dt, _PM_WGK[i], wg, transform, lo, hi,
            support_lower, support_upper, q, nq, a_lower, a_upper,
            m, kronrod, gauss,
        )
        if status != 0:
            return status
    for k in range(m):
        kronrod[k] *= half
        gauss[k] *= half
        error[k] = fabs(kronrod[k] - gauss[k])
    return 0


cdef inline void _pm_cleanup(
    double* vals, double* errs, double* bounds_a, double* bounds_b,
    double* total, double* total_err, double* left, double* right,
    double* err_left, double* err_right, double* tmp_g,
) noexcept nogil:
    if vals != NULL:
        free(vals)
    if errs != NULL:
        free(errs)
    if bounds_a != NULL:
        free(bounds_a)
    if bounds_b != NULL:
        free(bounds_b)
    if total != NULL:
        free(total)
    if total_err != NULL:
        free(total_err)
    if left != NULL:
        free(left)
    if right != NULL:
        free(right)
    if err_left != NULL:
        free(err_left)
    if err_right != NULL:
        free(err_right)
    if tmp_g != NULL:
        free(tmp_g)

cdef int _pm_segment(
    double lo,
    double hi,
    int transform,
    double support_lower,
    double support_upper,
    const double* q,
    Py_ssize_t nq,
    double a_lower,
    double a_upper,
    Py_ssize_t m,
    double epsabs,
    double epsrel,
    int limit,
    double* result,
) noexcept nogil:
    cdef double* vals = NULL
    cdef double* errs = NULL
    cdef double* bounds_a = NULL
    cdef double* bounds_b = NULL
    cdef double* total = NULL
    cdef double* total_err = NULL
    cdef double* left = NULL
    cdef double* right = NULL
    cdef double* err_left = NULL
    cdef double* err_right = NULL
    cdef double* tmp_g = NULL
    cdef Py_ssize_t k
    cdef int count = 1, idx, i, status
    cdef double mid, old_b, score, best, tol, denom

    vals = <double*>malloc(limit * m * sizeof(double))
    errs = <double*>malloc(limit * m * sizeof(double))
    bounds_a = <double*>malloc(limit * sizeof(double))
    bounds_b = <double*>malloc(limit * sizeof(double))
    total = <double*>malloc(m * sizeof(double))
    total_err = <double*>malloc(m * sizeof(double))
    left = <double*>malloc(m * sizeof(double))
    right = <double*>malloc(m * sizeof(double))
    err_left = <double*>malloc(m * sizeof(double))
    err_right = <double*>malloc(m * sizeof(double))
    tmp_g = <double*>malloc(m * sizeof(double))
    if (vals == NULL or errs == NULL or bounds_a == NULL or bounds_b == NULL or
            total == NULL or total_err == NULL or left == NULL or right == NULL or
            err_left == NULL or err_right == NULL or tmp_g == NULL):
        _pm_cleanup(vals, errs, bounds_a, bounds_b, total, total_err,
                    left, right, err_left, err_right, tmp_g)
        return 9

    status = _pm_gk15(
        0.0, 1.0, transform, lo, hi, support_lower, support_upper,
        q, nq, a_lower, a_upper, m, vals, tmp_g, errs,
    )
    if status != 0:
        _pm_cleanup(vals, errs, bounds_a, bounds_b, total, total_err,
                    left, right, err_left, err_right, tmp_g)
        return status
    bounds_a[0] = 0.0
    bounds_b[0] = 1.0
    for k in range(m):
        total[k] = vals[k]
        total_err[k] = errs[k]

    while count < limit:
        status = 0
        for k in range(m):
            tol = epsabs + epsrel * fabs(total[k])
            if total_err[k] > tol:
                status = 1
                break
        if status == 0:
            break

        idx = 0
        best = -1.0
        for i in range(count):
            score = 0.0
            for k in range(m):
                denom = epsabs + epsrel * fabs(total[k])
                if denom <= 0.0:
                    denom = 1e-300
                if errs[i * m + k] / denom > score:
                    score = errs[i * m + k] / denom
            if score > best:
                best = score
                idx = i
        old_b = bounds_b[idx]
        mid = 0.5 * (bounds_a[idx] + old_b)
        status = _pm_gk15(
            bounds_a[idx], mid, transform, lo, hi, support_lower, support_upper,
            q, nq, a_lower, a_upper, m, left, tmp_g, err_left,
        )
        if status != 0:
            _pm_cleanup(vals, errs, bounds_a, bounds_b, total, total_err,
                        left, right, err_left, err_right, tmp_g)
            return status
        status = _pm_gk15(
            mid, old_b, transform, lo, hi, support_lower, support_upper,
            q, nq, a_lower, a_upper, m, right, tmp_g, err_right,
        )
        if status != 0:
            _pm_cleanup(vals, errs, bounds_a, bounds_b, total, total_err,
                        left, right, err_left, err_right, tmp_g)
            return status
        for k in range(m):
            total[k] += left[k] + right[k] - vals[idx * m + k]
            total_err[k] += err_left[k] + err_right[k] - errs[idx * m + k]
            if total_err[k] < 0.0:
                total_err[k] = 0.0
            vals[idx * m + k] = left[k]
            errs[idx * m + k] = err_left[k]
            vals[count * m + k] = right[k]
            errs[count * m + k] = err_right[k]
        bounds_b[idx] = mid
        bounds_a[count] = mid
        bounds_b[count] = old_b
        count += 1

    for k in range(m):
        result[k] = total[k]
    _pm_cleanup(vals, errs, bounds_a, bounds_b, total, total_err,
                left, right, err_left, err_right, tmp_g)
    return 0


def power_moments(
    const double[::1] q_poly,
    const double[::1] support,
    const double[::1] boundary_amplitudes,
    const double[::1] window,
    const double[::1] points,
    int max_order,
    *,
    double epsabs,
    double epsrel,
    int limit,
):
    """Integrate all shifted raw power moments in shared adaptive traversals."""
    cdef Py_ssize_t m, k, i, nseg
    cdef double lower, upper, lo, hi, a_lower, a_upper
    cdef int transform, status
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out
    cdef cnp.ndarray[cnp.float64_t, ndim=1] seg
    cdef cnp.ndarray[cnp.float64_t, ndim=1] edges
    if (
        q_poly.shape[0] < 1
        or support.shape[0] != 2
        or boundary_amplitudes.shape[0] != 2
        or window.shape[0] != 2
    ):
        raise ValueError("invalid power-moment geometry")
    if max_order < 0 or limit < 1 or epsabs < 0.0 or epsrel < 0.0:
        raise ValueError("invalid power-moment controls")
    lower = support[0]
    upper = support[1]
    a_lower = boundary_amplitudes[0]
    a_upper = boundary_amplitudes[1]
    m = max_order + 1
    edges = np.empty(points.shape[0] + 2, dtype=np.float64)
    edges[0] = window[0]
    for i in range(points.shape[0]):
        edges[i + 1] = points[i]
    edges[points.shape[0] + 1] = window[1]
    out = np.zeros(m, dtype=np.float64)
    seg = np.empty(m, dtype=np.float64)
    nseg = edges.shape[0] - 1
    for i in range(nseg):
        lo = edges[i]
        hi = edges[i + 1]
        if not hi > lo:
            continue
        transform = 0
        if isfinite(lower) and lo == lower:
            transform = 1
        elif isfinite(upper) and hi == upper:
            transform = 2
        with nogil:
            status = _pm_segment(
                lo, hi, transform, lower, upper, &q_poly[0], q_poly.shape[0],
                a_lower, a_upper, m, epsabs / nseg, epsrel, limit, &seg[0],
            )
        if status == 9:
            raise MemoryError("power-moment adaptive quadrature allocation failed")
        if status == 2:
            raise RuntimeError("shifted potential is negative at a quadrature node")
        if status != 0:
            raise RuntimeError("power-moment adaptive quadrature failed")
        for k in range(m):
            out[k] += seg[k]
    return out
