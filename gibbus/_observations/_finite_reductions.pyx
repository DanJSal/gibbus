# cython: language_level=3
"""Compiled finite-interval objective kernel.

One traversal constructs local Gauss--Legendre nodes, evaluates the current
potential, normalizes each interval stably, and reduces the conditional
sufficient-statistic means and covariances.  The row loop runs without the
GIL.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, fabs, isfinite, log, INFINITY
from libc.stdlib cimport free, malloc

cnp.import_array()

cdef int _PARTIAL_POLY = 0
cdef int _PARTIAL_LOG_LOWER = 1
cdef int _PARTIAL_LOG_UPPER = 2


cdef inline double _polyval(const double* c, Py_ssize_t n, double z) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 0:
        return 0.0
    out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * z + c[i]
    return out


cdef inline double _partial_value(
    int kind,
    const double* coeff,
    Py_ssize_t ncoeff,
    double lower,
    double upper,
    double z,
) noexcept nogil:
    if kind == _PARTIAL_POLY:
        return _polyval(coeff, ncoeff, z)
    if kind == _PARTIAL_LOG_LOWER:
        return -log(z - lower)
    return -log(upper - z)


cdef inline double _potential_value(
    const double* q_poly,
    Py_ssize_t nq,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double z,
) noexcept nogil:
    """Evaluate the canonical potential at one interior point."""
    cdef double out = _polyval(q_poly, nq, z)
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


cdef inline double _point_potential_value(
    const double* q_poly,
    Py_ssize_t nq,
    double lower,
    double upper,
    double a_lower,
    double a_upper,
    double z,
    double preserved_lower,
    double preserved_upper,
) noexcept nogil:
    """Evaluate the potential at an exact point using preserved edge distances."""
    cdef double out = _polyval(q_poly, nq, z)
    cdef double d
    if isfinite(lower) and isfinite(a_lower) and a_lower > 0.0:
        d = preserved_lower if isfinite(preserved_lower) else z - lower
        if d <= 0.0:
            return INFINITY
        out -= a_lower * log(d)
    if isfinite(upper) and isfinite(a_upper) and a_upper > 0.0:
        d = preserved_upper if isfinite(preserved_upper) else upper - z
        if d <= 0.0:
            return INFINITY
        out -= a_upper * log(d)
    return out


cdef inline double _point_partial_value(
    int kind,
    const double* coeff,
    Py_ssize_t ncoeff,
    double lower,
    double upper,
    double z,
    double preserved_lower,
    double preserved_upper,
) noexcept nogil:
    """Evaluate one potential partial at an exact point."""
    cdef double d
    if kind == _PARTIAL_POLY:
        return _polyval(coeff, ncoeff, z)
    if kind == _PARTIAL_LOG_LOWER:
        d = preserved_lower if isfinite(preserved_lower) else z - lower
    else:
        d = preserved_upper if isfinite(preserved_upper) else upper - z
    if d <= 0.0:
        return INFINITY
    return -log(d)


# Failure codes of ``_finite_rows`` (0 is success).
cdef enum:
    _BAD_ROW = 1
    _BAD_POINT_PARTIAL = 2
    _BAD_PANEL = 3
    _BAD_POTENTIAL = 4
    _BAD_PARTIAL = 5
    _BAD_MASS = 6


cdef int _finite_rows(
    Py_ssize_t R, Py_ssize_t P, Py_ssize_t G, Py_ssize_t W, Py_ssize_t nq,
    const double* intervals, const double* row_weights,
    const double* point_lower_distance, const double* point_upper_distance,
    const double* q_poly, double a_lower, double a_upper, double q_shift,
    double shifted_log_Z, double mode, double log_scale,
    const int* partial_kinds, const int* partial_lengths,
    const double* partial_coeffs, double support_lower, double support_upper,
    const double* gl_nodes, const double* gl_log_weights, double width_eps_mult,
    double* log_probability, double* obs_h, double* obs_cov,
    double* sum_h, double* sum_second, double* hbuf,
    bint natural_real_line, const double* natural_scale,
    double* row_means, double* row_cov,
) noexcept nogil:
    """Reduce finite rows; optionally keep each row's conditional moments.

    ``row_means`` (``P x R``) and ``row_cov`` (``P*P x R``) are feature-major
    per-row outputs, skipped when ``NULL``.
    """
    cdef Py_ssize_t r, i, j, g, panel, n_panels
    cdef double lo, hi, width, mid, hpanel, mpanel, log_hpanel, z, zpow
    cdef double qv, term, max_log, scaled, scale, total
    cdef double wrow, mean_i, mean_j, d_lower, d_upper, log_integral
    cdef bint have_term, split

    for r in range(R):
        lo = intervals[2 * r]
        hi = intervals[2 * r + 1]
        if not (isfinite(lo) and isfinite(hi)) or lo > hi:
            return _BAD_ROW
        width = hi - lo
        mid = 0.5 * (lo + hi)
        wrow = row_weights[r]
        if natural_real_line:
            d_lower = 0.0
            d_upper = 0.0
        else:
            d_lower = point_lower_distance[r]
            d_upper = point_upper_distance[r]

        if width <= width_eps_mult * (1.0 + fabs(mid)):
            if natural_real_line:
                qv = _polyval(q_poly, nq, mid)
            else:
                qv = _point_potential_value(
                    q_poly, nq, support_lower, support_upper,
                    a_lower, a_upper, mid, d_lower, d_upper,
                )
            log_integral = q_shift - qv
            if width > 0.0:
                log_integral += log(width)
            log_probability[r] = log_integral - shifted_log_Z
            if width == 0.0:
                log_probability[r] -= log_scale
            if natural_real_line:
                if P > 0:
                    obs_h[0] += wrow * mid
                    if row_means != NULL:
                        row_means[r] = mid
                    zpow = mid * mid
                    for i in range(1, P):
                        obs_h[i] += wrow * zpow * natural_scale[i]
                        if row_means != NULL:
                            row_means[i * R + r] = zpow * natural_scale[i]
                        zpow *= mid
            else:
                for i in range(P):
                    mean_i = _point_partial_value(
                        partial_kinds[i], &partial_coeffs[i * W], partial_lengths[i],
                        support_lower, support_upper, mid, d_lower, d_upper,
                    )
                    if not isfinite(mean_i):
                        return _BAD_POINT_PARTIAL
                    obs_h[i] += wrow * mean_i
                    if row_means != NULL:
                        row_means[i * R + r] = mean_i
            if row_cov != NULL:
                for i in range(P * P):
                    row_cov[i * R + r] = 0.0
            continue

        for i in range(P):
            sum_h[i] = 0.0
        for i in range(P * P):
            sum_second[i] = 0.0
        max_log = 0.0
        total = 0.0
        have_term = False

        split = isfinite(mode) and lo < mode and mode < hi
        n_panels = 2 if split else 1
        for panel in range(n_panels):
            if split and panel == 0:
                hpanel = 0.5 * (mode - lo)
                mpanel = 0.5 * (lo + mode)
            elif split:
                hpanel = 0.5 * (hi - mode)
                mpanel = 0.5 * (mode + hi)
            else:
                hpanel = 0.5 * width
                mpanel = mid
            if not (hpanel > 0.0 and isfinite(hpanel)):
                return _BAD_PANEL
            log_hpanel = log(hpanel)

            for g in range(G):
                z = mpanel + hpanel * gl_nodes[g]
                if natural_real_line:
                    qv = _polyval(q_poly, nq, z)
                else:
                    qv = _potential_value(
                        q_poly, nq, support_lower, support_upper, a_lower, a_upper, z,
                    )
                if not isfinite(qv):
                    return _BAD_POTENTIAL
                term = log_hpanel + gl_log_weights[g] + q_shift - qv

                if not have_term:
                    max_log = term
                    total = 1.0
                    scaled = 1.0
                    have_term = True
                elif term > max_log:
                    scale = exp(max_log - term)
                    total = total * scale + 1.0
                    for i in range(P):
                        sum_h[i] *= scale
                    for i in range(P * P):
                        sum_second[i] *= scale
                    max_log = term
                    scaled = 1.0
                else:
                    scaled = exp(term - max_log)
                    total += scaled

                if natural_real_line:
                    hbuf[0] = z
                    sum_h[0] += scaled * z
                    zpow = z * z
                    for i in range(1, P):
                        mean_i = zpow * natural_scale[i]
                        hbuf[i] = mean_i
                        sum_h[i] += scaled * mean_i
                        zpow *= z
                else:
                    for i in range(P):
                        mean_i = _partial_value(
                            partial_kinds[i],
                            &partial_coeffs[i * W],
                            partial_lengths[i],
                            support_lower,
                            support_upper,
                            z,
                        )
                        if not isfinite(mean_i):
                            return _BAD_PARTIAL
                        hbuf[i] = mean_i
                        sum_h[i] += scaled * mean_i
                for i in range(P):
                    for j in range(P):
                        sum_second[i * P + j] += scaled * hbuf[i] * hbuf[j]

        if not have_term or not (total > 0.0 and isfinite(total)):
            return _BAD_MASS
        log_integral = max_log + log(total)
        log_probability[r] = log_integral - shifted_log_Z
        for i in range(P):
            mean_i = sum_h[i] / total
            obs_h[i] += wrow * mean_i
            if row_means != NULL:
                row_means[i * R + r] = mean_i
            for j in range(P):
                mean_j = sum_h[j] / total
                obs_cov[i * P + j] += wrow * (
                    sum_second[i * P + j] / total - mean_i * mean_j
                )
                if row_cov != NULL:
                    row_cov[(i * P + j) * R + r] = (
                        sum_second[i * P + j] / total - mean_i * mean_j
                    )
    return 0


cdef api int finite_natural_objective_c(
    Py_ssize_t R, Py_ssize_t P, Py_ssize_t G, Py_ssize_t W, Py_ssize_t nq,
    const double* intervals, const double* row_weights,
    const double* point_lower_distance, const double* point_upper_distance,
    const double* q_poly, double a_lower, double a_upper,
    double q_shift, double shifted_log_Z, double mode, double log_coordinate_scale,
    const int* partial_kinds, const int* partial_lengths,
    const double* partial_coeffs, double support_lower, double support_upper,
    const double* gl_nodes, const double* gl_log_weights, double width_eps_mult,
    double* log_probability, double* obs_h, double* obs_cov,
    double* sum_h, double* sum_second, double* hbuf,
    double* nll_out, double* row_means, double* row_cov,
) noexcept nogil:
    """Reduce ordinary finite interval rows for an arbitrary natural basis.

    This C-API entry point mirrors :func:`evaluate_finite_objective` without
    allocating Python objects.  The caller owns all workspaces and must route
    positive-width rows that touch a finite support boundary to the adaptive
    reducer, exactly as the Python objective does.  Optional feature-major
    ``row_means`` (``P x R``) and ``row_cov`` (``P*P x R``) receive each row's
    conditional moments; pass ``NULL`` to skip them.
    """
    cdef Py_ssize_t i
    cdef int status
    for i in range(P):
        obs_h[i] = 0.0
    for i in range(P * P):
        obs_cov[i] = 0.0
    status = _finite_rows(
        R, P, G, W, nq,
        intervals, row_weights,
        point_lower_distance, point_upper_distance,
        q_poly, a_lower, a_upper, q_shift,
        shifted_log_Z, mode, log_coordinate_scale,
        partial_kinds, partial_lengths, partial_coeffs,
        support_lower, support_upper,
        gl_nodes, gl_log_weights, width_eps_mult,
        log_probability, obs_h, obs_cov,
        sum_h, sum_second, hbuf,
        False, NULL, row_means, row_cov,
    )
    if status != 0:
        return status
    nll_out[0] = 0.0
    for i in range(R):
        nll_out[0] -= row_weights[i] * log_probability[i]
    if not isfinite(nll_out[0]):
        return _BAD_MASS
    return 0


cdef api int finite_natural_real_line_objective_c(
    Py_ssize_t R, Py_ssize_t P, Py_ssize_t G, Py_ssize_t nq,
    const double* intervals, const double* row_weights,
    const double* q_poly, double q_shift, double shifted_log_Z,
    double mode, double log_coordinate_scale,
    const double* natural_scale,
    const double* gl_nodes, const double* gl_log_weights,
    double width_eps_mult,
    double* log_probability, double* obs_h, double* obs_cov,
    double* sum_h, double* sum_second, double* hbuf,
    double* nll_out, double* row_means, double* row_cov,
) noexcept nogil:
    """Reduce a finite real-line natural interval objective without Python objects.

    This is the C-API entry point used by the fused interval Newton loop.  The
    caller owns every workspace.  ``obs_h`` and ``obs_cov`` are zeroed here;
    ``nll_out`` receives the negative weighted interval log likelihood.
    Optional per-row moments are written as in
    :c:func:`finite_natural_objective_c`.
    """
    cdef Py_ssize_t i
    cdef int status
    for i in range(P):
        obs_h[i] = 0.0
    for i in range(P * P):
        obs_cov[i] = 0.0
    status = _finite_rows(
        R, P, G, 0, nq,
        intervals, row_weights,
        NULL, NULL,
        q_poly, 0.0, 0.0, q_shift,
        shifted_log_Z, mode, log_coordinate_scale,
        NULL, NULL, NULL, -INFINITY, INFINITY,
        gl_nodes, gl_log_weights, width_eps_mult,
        log_probability, obs_h, obs_cov,
        sum_h, sum_second, hbuf,
        True, natural_scale, row_means, row_cov,
    )
    if status != 0:
        return status
    nll_out[0] = 0.0
    for i in range(R):
        nll_out[0] -= row_weights[i] * log_probability[i]
    if not isfinite(nll_out[0]):
        return _BAD_MASS
    return 0


def evaluate_finite_objective(
    const double[:, ::1] intervals,
    const double[::1] row_weights,
    const double[::1] point_lower_distance,
    const double[::1] point_upper_distance,
    const double[::1] q_poly,
    const double[::1] boundary_amplitudes,
    double q_shift,
    double shifted_log_Z,
    double mode,
    double coordinate_scale,
    const cnp.int32_t[::1] partial_kinds,
    const cnp.int32_t[::1] partial_lengths,
    const double[:, ::1] partial_coeffs,
    double support_lower,
    double support_upper,
    const double[::1] gl_nodes,
    const double[::1] gl_log_weights,
    double width_eps_mult,
):
    """Evaluate ordinary finite interval rows entirely in compiled code.

    The kernel constructs local Gauss--Legendre nodes on demand, evaluates the
    potential, performs a stable streaming log-sum-exp normalization, and
    accumulates conditional first-partial means and covariance terms without
    materializing row-by-node temporary arrays.  Narrow rows use the
    point-limit convention of the quadrature plan, including preserved sub-ulp
    distances for exact boundary-adjacent points.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Finite canonical interval endpoints.  Positive-width rows touching a
        finite support boundary must be routed to the adaptive reducer instead.
    row_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    point_lower_distance, point_upper_distance : numpy.ndarray, shape (R,)
        Preserved exact-point distances from finite support boundaries; NaN
        denotes unavailable metadata.
    q_poly : numpy.ndarray
        Potential polynomial coefficients, constant term first.
    boundary_amplitudes : numpy.ndarray, shape (2,)
        Lower and upper logarithmic boundary amplitudes.
    q_shift : float
        Potential shift used by the state normalization integral.
    shifted_log_Z : float
        Logarithm of the shifted normalizer ``state.Z``.
    mode : float
        Current canonical mode; intervals containing it are split there.
    coordinate_scale : float
        User-to-canonical affine scale, used by exact-point density rows.
    partial_kinds, partial_lengths, partial_coeffs : numpy.ndarray
        Packed first-potential partial descriptors.
    support_lower, support_upper : float
        Canonical support endpoints.
    gl_nodes, gl_log_weights : numpy.ndarray
        Symmetric Gauss--Legendre rule on ``[-1, 1]``.
    width_eps_mult : float
        Relative width threshold selecting the point-limit convention.

    Returns
    -------
    log_probability : numpy.ndarray, shape (R,)
        Normalized row log probabilities (or exact-point log densities).
    observed_h : numpy.ndarray, shape (P,)
        Observation-weighted conditional first-partial means.
    observed_cov : numpy.ndarray, shape (P, P)
        Observation-weighted conditional partial covariance.
    """
    cdef Py_ssize_t R = intervals.shape[0]
    cdef Py_ssize_t P = partial_kinds.shape[0]
    cdef Py_ssize_t W = partial_coeffs.shape[1]
    cdef Py_ssize_t G = gl_nodes.shape[0]
    cdef Py_ssize_t nq = q_poly.shape[0]
    cdef int status
    cdef bint natural_real_line
    cdef Py_ssize_t i
    cdef double* work = NULL
    cdef cnp.ndarray[cnp.float64_t, ndim=1] log_probability
    cdef cnp.ndarray[cnp.float64_t, ndim=1] obs_h
    cdef cnp.ndarray[cnp.float64_t, ndim=2] obs_cov
    cdef double[::1] lp_view
    cdef double[::1] h_view
    cdef double[:, ::1] cov_view

    if intervals.shape[1] != 2:
        raise ValueError("intervals must have shape (R, 2)")
    if (
        row_weights.shape[0] != R
        or point_lower_distance.shape[0] != R
        or point_upper_distance.shape[0] != R
    ):
        raise ValueError("row arrays must match intervals")
    if q_poly.shape[0] < 1 or boundary_amplitudes.shape[0] != 2:
        raise ValueError("potential descriptors are invalid")
    if P < 1 or partial_lengths.shape[0] != P or partial_coeffs.shape[0] != P or W < 1:
        raise ValueError("partial descriptor arrays must match")
    if G < 1 or gl_log_weights.shape[0] != G:
        raise ValueError("Gauss--Legendre rule arrays must have equal nonzero length")
    if not (coordinate_scale > 0.0 and isfinite(coordinate_scale)):
        raise ValueError("coordinate_scale must be finite and positive")
    if not (width_eps_mult >= 0.0 and isfinite(width_eps_mult)):
        raise ValueError("width_eps_mult must be finite and non-negative")

    log_probability = np.empty(R, dtype=np.float64)
    obs_h = np.zeros(P, dtype=np.float64)
    obs_cov = np.zeros((P, P), dtype=np.float64)
    if R == 0:
        return log_probability, obs_h, obs_cov
    lp_view = log_probability
    h_view = obs_h
    cov_view = obs_cov
    natural_real_line = (
        (not isfinite(support_lower)) and (not isfinite(support_upper)) and nq == P + 1
    )
    work = <double*>malloc((3 * P + P * P) * sizeof(double))
    if work == NULL:
        raise MemoryError("finite interval objective allocation failed")
    if natural_real_line:
        work[2 * P + P * P] = 0.0
        for i in range(1, P):
            work[2 * P + P * P + i] = 1.0 / (i * (i + 1.0))
    with nogil:
        status = _finite_rows(
            R,
            P,
            G,
            W,
            nq,
            &intervals[0, 0],
            &row_weights[0],
            &point_lower_distance[0],
            &point_upper_distance[0],
            &q_poly[0],
            boundary_amplitudes[0],
            boundary_amplitudes[1],
            q_shift,
            shifted_log_Z,
            mode,
            log(coordinate_scale),
            <const int*>&partial_kinds[0],
            <const int*>&partial_lengths[0],
            &partial_coeffs[0, 0],
            support_lower,
            support_upper,
            &gl_nodes[0],
            &gl_log_weights[0],
            width_eps_mult,
            &lp_view[0],
            &h_view[0],
            &cov_view[0, 0],
            work,
            work + P,
            work + P + P * P,
            natural_real_line,
            work + 2 * P + P * P,
            NULL,
            NULL,
        )
    free(work)
    if status == _BAD_ROW:
        raise ValueError("finite objective rows must have ordered finite endpoints")
    if status == _BAD_POINT_PARTIAL:
        raise RuntimeError("finite interval point partial is non-finite")
    if status == _BAD_PANEL:
        raise RuntimeError("finite interval quadrature panel is invalid")
    if status == _BAD_POTENTIAL:
        raise RuntimeError("finite interval potential is non-finite")
    if status == _BAD_PARTIAL:
        raise RuntimeError("finite interval partial is non-finite")
    if status == _BAD_MASS:
        raise RuntimeError("finite interval quadrature produced invalid mass")
    return log_probability, obs_h, obs_cov


def partial_kinds():
    """Return integer tags understood by the finite objective kernels."""
    return {
        "poly": _PARTIAL_POLY,
        "lower_log": _PARTIAL_LOG_LOWER,
        "upper_log": _PARTIAL_LOG_UPPER,
    }
