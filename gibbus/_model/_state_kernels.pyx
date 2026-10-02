# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled state numerics shared by the model and post-fit layers.

The module contains no parameterization or empirical likelihood code.  It
provides the compiled numerical primitives of the model state and post-fit
evaluation:

* ``state_numerics`` (and ``_state_numerics_c`` for other kernels) for the
  mode, window, normalizer, moments and Fisher statistics in one pass;
* ``_valley_q1_shift`` for shifted stationary-point solves;
* ``_q_window_and_mode`` for the mode and stable integration window;
* ``_pdf_vec`` for bulk density evaluation;
* ``_polyval_vec`` for bulk polynomial evaluation.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs, log, sqrt, isinf, isnan, exp as c_exp, NAN, INFINITY

cnp.import_array()


cdef inline double _nan() noexcept nogil:
    return (<double>0.0) / (<double>0.0)


cdef inline bint _finite(double x) noexcept nogil:
    return (x == x) and (not isinf(x))


cdef inline double _poly_eval(const double* c, Py_ssize_t n, double x) noexcept nogil:
    cdef Py_ssize_t i
    cdef double out
    if n <= 0:
        return 0.0
    out = c[n - 1]
    for i in range(n - 2, -1, -1):
        out = out * x + c[i]
    return out


cdef inline double _agm(double a, double b) noexcept nogil:
    cdef double x = a
    cdef double y = b
    cdef double xn, yn
    cdef int it
    if (not _finite(x)) or (not _finite(y)):
        return 0.0
    if x <= 0.0 or y <= 0.0:
        return 0.0
    for it in range(64):
        xn = 0.5 * (x + y)
        yn = sqrt(x * y)
        if fabs(xn - yn) <= 1e-15 * (fabs(xn) + fabs(yn) + 1.0):
            return xn
        x = xn
        y = yn
    return 0.5 * (x + y)


cdef struct _QCtx:
    double L
    double U
    bint finL
    bint finU

    double aL
    double aU

    bint hasL
    bint hasU

    Py_ssize_t nq
    const double* q

    Py_ssize_t nd1
    const double* d1

    Py_ssize_t nd2
    const double* d2


cdef inline double _q0(_QCtx* ctx, double x) noexcept nogil:
    cdef double out = _poly_eval(ctx.q, ctx.nq, x)
    cdef double dL, dU
    if ctx.hasL:
        dL = x - ctx.L
        if dL <= 0.0:
            return _nan()
        out -= ctx.aL * log(dL)
    if ctx.hasU:
        dU = ctx.U - x
        if dU <= 0.0:
            return _nan()
        out -= ctx.aU * log(dU)
    return out


cdef inline double _q1(_QCtx* ctx, double x) noexcept nogil:
    cdef double out = _poly_eval(ctx.d1, ctx.nd1, x)
    cdef double dL, dU
    if ctx.hasL:
        dL = x - ctx.L
        if dL <= 0.0:
            return _nan()
        out += (-ctx.aL) / dL
    if ctx.hasU:
        dU = ctx.U - x
        if dU <= 0.0:
            return _nan()
        out += (ctx.aU) / dU
    return out


cdef inline double _q2(_QCtx* ctx, double x) noexcept nogil:
    cdef double out = _poly_eval(ctx.d2, ctx.nd2, x)
    cdef double dL, dU
    if ctx.hasL:
        dL = x - ctx.L
        if dL <= 0.0:
            return _nan()
        out += ctx.aL / (dL * dL)
    if ctx.hasU:
        dU = ctx.U - x
        if dU <= 0.0:
            return _nan()
        out += ctx.aU / (dU * dU)
    return out


cdef inline double _safe(double x, double L, double U, bint finL, bint finU, double eps) noexcept nogil:
    if finL:
        if x < L + eps:
            x = L + eps
    if finU:
        if x > U - eps:
            x = U - eps
    return x


cdef inline double _f(_QCtx* ctx, double x, int kind, double param) noexcept nogil:
    if kind == 0:
        return _q1(ctx, x) - param
    if kind == 1:
        return _q0(ctx, x) - param
    return (-_q0(ctx, x)) + param


cdef inline double _fp(_QCtx* ctx, double x, int kind) noexcept nogil:
    if kind == 0:
        return _q2(ctx, x)
    if kind == 1:
        return _q1(ctx, x)
    return -_q1(ctx, x)


cdef inline bint _try_bracket(
    _QCtx* ctx,
    int kind,
    double param,
    double x0,
    double L,
    double U,
    bint finL,
    bint finU,
    double eps,
    double bracket_init_step,
    int bracket_max_expand,
    double bracket_step_growth,
    double* a_out,
    double* b_out,
) noexcept nogil:
    cdef double a, b, at, bt
    cdef double ga, gb, gat, gbt
    cdef double step
    cdef int it

    if finL and finU:
        a = _safe(L, L, U, finL, finU, eps)
        b = _safe(U, L, U, finL, finU, eps)
        ga = _f(ctx, a, kind, param)
        gb = _f(ctx, b, kind, param)
        if _finite(ga) and _finite(gb) and (ga * gb <= 0.0):
            a_out[0] = a
            b_out[0] = b
            return 1

    x0 = _safe(x0, L, U, finL, finU, eps)
    gb = _f(ctx, x0, kind, param)
    if not _finite(gb):
        return 0

    step = bracket_init_step
    if fabs(gb) > step:
        step = fabs(gb)

    a = x0
    b = x0
    ga = gb

    for it in range(bracket_max_expand):
        at = _safe(a - step, L, U, finL, finU, eps)
        gat = _f(ctx, at, kind, param)
        if _finite(gat) and (gat * gb <= 0.0):
            a_out[0] = at
            b_out[0] = b
            return 1

        bt = _safe(b + step, L, U, finL, finU, eps)
        gbt = _f(ctx, bt, kind, param)
        if _finite(gbt) and (ga * gbt <= 0.0):
            a_out[0] = a
            b_out[0] = bt
            return 1

        if _finite(gat) and ((not _finite(ga)) or (fabs(gat) < fabs(ga))):
            a = at
            ga = gat
        if _finite(gbt) and ((not _finite(gb)) or (fabs(gbt) < fabs(gb))):
            b = bt
            gb = gbt

        if finL and (a <= L + eps) and finU and (b >= U - eps):
            break

        step *= bracket_step_growth

    return 0


cdef inline double _bisect(
    _QCtx* ctx,
    int kind,
    double param,
    double a,
    double b,
    double fa,
    double fb,
    int maxiter,
    double newt_tol,
) noexcept nogil:
    cdef int it
    cdef double m, fm
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    for it in range(maxiter):
        m = 0.5 * (a + b)
        fm = _f(ctx, m, kind, param)
        if (not _finite(fm)):
            b = m
            continue
        if fm == 0.0:
            return m
        if (b - a) <= newt_tol:
            return m
        if fa * fm <= 0.0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


cdef double _valley_solve(
    _QCtx* ctx,
    int kind,
    double param,
    double L,
    double U,
    bint finL,
    bint finU,
    double grad_tol,
    double hess_tol,
    double newt_tol,
    int newt_max,
    double boundary_eps_mult,
    double bracket_init_step,
    int bracket_max_expand,
    double bracket_step_growth,
    int backtrack_max_iters,
    double backtrack_reduce,
) noexcept nogil:
    cdef double epsL, epsU, eps, width, s
    cdef double a, b, x, gL, gU
    cdef double g, h, step, xn, gn, gx
    cdef double lmbda
    cdef int it, bt
    cdef bint newton_ok = 0
    cdef double ba, bb, fa, fb

    if finL and finU:
        epsL = newt_tol
        s = boundary_eps_mult * (1.0 + fabs(L))
        if s > epsL:
            epsL = s
        epsU = newt_tol
        s = boundary_eps_mult * (1.0 + fabs(U))
        if s > epsU:
            epsU = s

        width = U - L
        if epsL + epsU >= 0.5 * width:
            s = 0.25 * width / (epsL + epsU if (epsL + epsU) > 1.0 else 1.0)
            epsL *= s
            epsU *= s

        a = _safe(L + epsL, L, U, finL, finU, epsL if epsL > epsU else epsU)
        b = _safe(U - epsU, L, U, finL, finU, epsL if epsL > epsU else epsU)

        gL = _f(ctx, a, kind, param)
        gU = _f(ctx, b, kind, param)

        if _finite(gL) and _finite(gU):
            if gL >= 0.0 and gU >= 0.0:
                return L
            if gL <= 0.0 and gU <= 0.0:
                return U

        x = 0.5 * (L + U)
    elif finL:
        epsL = newt_tol
        s = boundary_eps_mult * (1.0 + fabs(L))
        if s > epsL:
            epsL = s
        x = L + (bracket_init_step if bracket_init_step > epsL else epsL)
        # For mode searches, a non-negative derivative immediately inside a
        # finite lower endpoint certifies that convex q is minimized at the
        # endpoint itself.  Return the endpoint rather than the interior probe
        # so the mode is not displaced by the arbitrary bracket step.
        if kind == 0:
            gL = _f(ctx, L + epsL, kind, param)
            if _finite(gL) and gL >= 0.0:
                return L
    elif finU:
        epsU = newt_tol
        s = boundary_eps_mult * (1.0 + fabs(U))
        if s > epsU:
            epsU = s
        x = U - (bracket_init_step if bracket_init_step > epsU else epsU)
        if kind == 0:
            gU = _f(ctx, U - epsU, kind, param)
            if _finite(gU) and gU <= 0.0:
                return U
    else:
        x = 0.0

    epsL = newt_tol
    s = boundary_eps_mult * (0.0 if (not finL) else (1.0 + fabs(L)))
    if s > epsL:
        epsL = s
    epsU = newt_tol
    s = boundary_eps_mult * (0.0 if (not finU) else (1.0 + fabs(U)))
    if s > epsU:
        epsU = s

    if finL and (x < L + epsL):
        x = L + epsL
    if finU and (x > U - epsU):
        x = U - epsU

    for it in range(newt_max):
        g = _f(ctx, x, kind, param)
        if not _finite(g):
            break
        if fabs(g) < grad_tol:
            newton_ok = 1
            break

        h = _fp(ctx, x, kind)
        if (not _finite(h)) or (fabs(h) < hess_tol):
            break

        step = g / h
        if not _finite(step):
            break

        xn = _safe(x - step, L, U, finL, finU, epsL if epsL > epsU else epsU)
        gn = fabs(_f(ctx, xn, kind, param))
        gx = fabs(g)

        lmbda = 1.0
        for bt in range(backtrack_max_iters):
            if (not _finite(gn)):
                pass
            elif (gn < gx) or (fabs(xn - x) < newt_tol):
                break
            lmbda *= backtrack_reduce
            xn = _safe(x - lmbda * step, L, U, finL, finU, epsL if epsL > epsU else epsU)
            gn = fabs(_f(ctx, xn, kind, param))

        x = xn

        g = _f(ctx, x, kind, param)
        if _finite(g) and (fabs(g) < grad_tol or fabs(step) * lmbda < newt_tol):
            newton_ok = 1
            break

    if newton_ok:
        return x

    eps = newt_tol
    s = boundary_eps_mult * (1.0 + (0.0 if not finL else fabs(L)) + (0.0 if not finU else fabs(U)))
    if s > eps:
        eps = s

    ba = 0.0
    bb = 0.0
    if _try_bracket(
        ctx, kind, param, x, L, U, finL, finU, eps,
        bracket_init_step, bracket_max_expand, bracket_step_growth,
        &ba, &bb
    ):
        fa = _f(ctx, ba, kind, param)
        fb = _f(ctx, bb, kind, param)
        if _finite(fa) and _finite(fb) and (fa * fb <= 0.0):
            return _bisect(ctx, kind, param, ba, bb, fa, fb, newt_max, newt_tol)

    return x


cpdef double _valley_q1_shift(
    bounds,
    support,
    q_poly,
    boundary_amplitudes,
    shift,
    grad_tol,
    hess_tol,
    newt_tol,
    newt_max,
    boundary_eps_mult,
    bracket_init_step,
    bracket_max_expand,
    bracket_step_growth,
    backtrack_max_iters,
    backtrack_reduce,
):
    """Find the point where ``q'(z) = shift`` inside *bounds*.

    Uses Newton-Raphson with a backtracking line search, falling back to
    bracketing and bisection when Newton does not converge.  Setting
    ``shift=0`` yields the mode of ``exp(-q)``.

    All tolerances are required rather than defaulted: they live in
    :mod:`._defaults` and every caller passes them explicitly, so a
    default here would be a second source of truth.

    Parameters
    ----------
    bounds : array_like, shape (2,)
        Search interval in internal ``z`` coordinates.
    support : array_like, shape (2,)
        Distribution support in ``z`` coordinates; may be infinite.
    q_poly : array_like, shape (d+1,)
        Coefficients of the potential ``q``, constant term first.
    boundary_amplitudes : array_like, shape (2,)
        Canonical boundary amplitudes ``[aL, aU]``; unavailable sides are NaN.
    shift : float
        Target value of ``q'(z)``.
    grad_tol, hess_tol, newt_tol : float
        Convergence tolerances on the gradient, the Hessian magnitude
        below which a Newton step is rejected, and the step size.
    newt_max : int
        Maximum Newton iterations before falling back to bisection.
    boundary_eps_mult : float
        Multiplier setting how close an iterate may approach a boundary
        singularity.
    bracket_init_step, bracket_max_expand, bracket_step_growth : float, int, float
        Initial step, expansion limit, and growth factor for the
        bisection bracket search.
    backtrack_max_iters : int
        Maximum backtracking steps per Newton iteration.
    backtrack_reduce : float
        Step-size reduction factor applied when backtracking.

    Returns
    -------
    float
        ``z*`` with ``q'(z*) approximately equal to shift``, or the best
        iterate found if neither Newton nor bisection converged.

    Raises
    ------
    ValueError
        If *q_poly* is empty.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] bnd = np.ascontiguousarray(bounds, dtype=np.float64).reshape(2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sup = np.ascontiguousarray(support, dtype=np.float64).reshape(2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qp = np.ascontiguousarray(q_poly, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qb = np.ascontiguousarray(boundary_amplitudes, dtype=np.float64).reshape(2)
    if qp.size < 1:
        raise ValueError("q_poly must be non-empty")
    cdef double c_shift = float(shift)
    cdef double c_grad_tol = float(grad_tol)
    cdef double c_hess_tol = float(hess_tol)
    cdef double c_newt_tol = float(newt_tol)
    cdef int c_newt_max = int(newt_max)
    cdef double c_boundary_eps_mult = float(boundary_eps_mult)
    cdef double c_bracket_init_step = float(bracket_init_step)
    cdef int c_bracket_max_expand = int(bracket_max_expand)
    cdef double c_bracket_step_growth = float(bracket_step_growth)
    cdef int c_backtrack_max_iters = int(backtrack_max_iters)
    cdef double c_backtrack_reduce = float(backtrack_reduce)
    cdef Py_ssize_t nq = qp.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d1 = np.empty(max(nq - 1, 0), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d2 = np.empty(max(nq - 2, 0), dtype=np.float64)
    cdef Py_ssize_t i
    for i in range(nq - 1):
        d1[i] = (i + 1) * qp[i + 1]
    for i in range(nq - 2):
        d2[i] = (i + 2) * (i + 1) * qp[i + 2]
    cdef _QCtx ctx
    ctx.L = float(sup[0])
    ctx.U = float(sup[1])
    ctx.finL = _finite(ctx.L)
    ctx.finU = _finite(ctx.U)
    ctx.aL = float(qb[0])
    ctx.aU = float(qb[1])
    ctx.hasL = _finite(ctx.aL) and (ctx.aL > 0.0) and ctx.finL
    ctx.hasU = _finite(ctx.aU) and (ctx.aU > 0.0) and ctx.finU
    ctx.nq = nq
    ctx.q = <const double*> &qp[0]
    ctx.nd1 = d1.shape[0]
    ctx.d1 = (<const double*> &d1[0]) if ctx.nd1 > 0 else <const double*> NULL
    ctx.nd2 = d2.shape[0]
    ctx.d2 = (<const double*> &d2[0]) if ctx.nd2 > 0 else <const double*> NULL
    cdef double L = float(bnd[0])
    cdef double U = float(bnd[1])
    cdef bint finL = _finite(L)
    cdef bint finU = _finite(U)
    if finL and finU and not (L < U):
        raise ValueError("bounds must satisfy bounds[0] < bounds[1]")
    cdef double out
    with nogil:
        out = _valley_solve(
            &ctx, 0, c_shift,
            L, U, finL, finU,
            c_grad_tol, c_hess_tol, c_newt_tol, c_newt_max,
            c_boundary_eps_mult,
            c_bracket_init_step, c_bracket_max_expand, c_bracket_step_growth,
            c_backtrack_max_iters, c_backtrack_reduce,
        )
    return out
cdef void _ctx_setup(_QCtx* ctx, double L, double U, double aL, double aU,
                     const double* q, Py_ssize_t nq, double* d1, double* d2) noexcept nogil:
    """Fill a potential context; ``d1``/``d2`` receive the derivative coefficients."""
    cdef Py_ssize_t i
    for i in range(nq - 1):
        d1[i] = (i + 1) * q[i + 1]
    for i in range(nq - 2):
        d2[i] = (i + 2) * (i + 1) * q[i + 2]
    ctx.L = L
    ctx.U = U
    ctx.finL = _finite(L)
    ctx.finU = _finite(U)
    ctx.aL = aL
    ctx.aU = aU
    ctx.hasL = _finite(aL) and (aL > 0.0) and ctx.finL
    ctx.hasU = _finite(aU) and (aU > 0.0) and ctx.finU
    ctx.nq = nq
    ctx.q = q
    ctx.nd1 = nq - 1 if nq > 1 else 0
    ctx.d1 = d1 if ctx.nd1 > 0 else NULL
    ctx.nd2 = nq - 2 if nq > 2 else 0
    ctx.d2 = d2 if ctx.nd2 > 0 else NULL


cdef void _window_mode_c(_QCtx* ctx, double xb0, double xb1, const double* ctl,
                         double* lower_out, double* upper_out, double* mode_out,
                         double* q_mode_out, double* q2_mode_out,
                         double* core_lower_out, double* core_upper_out) noexcept nogil:
    """Mode, tail points and data-padded window (see ``_q_window_and_mode``).

    ``ctl`` holds, in order: log_thresh, grad_tol, hess_tol, newt_tol,
    newt_max, boundary_eps_mult, bracket_init_step, bracket_max_expand,
    bracket_step_growth, backtrack_max_iters, backtrack_reduce.
    """
    cdef double log_thresh = ctl[0]
    cdef double grad_tol = ctl[1]
    cdef double hess_tol = ctl[2]
    cdef double newt_tol = ctl[3]
    cdef int newt_max = <int>ctl[4]
    cdef double boundary_eps_mult = ctl[5]
    cdef double bracket_init_step = ctl[6]
    cdef int bracket_max_expand = <int>ctl[7]
    cdef double bracket_step_growth = ctl[8]
    cdef int backtrack_max_iters = <int>ctl[9]
    cdef double backtrack_reduce = ctl[10]
    cdef double L = ctx.L
    cdef double U = ctx.U
    cdef bint finL = ctx.finL
    cdef bint finU = ctx.finU
    cdef double mode, q_mode, q2_mode, lower, upper, pad_num, pad_data, pad, lo, hi
    mode = _valley_solve(
        ctx, 0, 0.0,
        L, U, finL, finU,
        grad_tol, hess_tol, newt_tol, newt_max,
        boundary_eps_mult,
        bracket_init_step, bracket_max_expand, bracket_step_growth,
        backtrack_max_iters, backtrack_reduce,
    )
    q_mode = _q0(ctx, mode)
    q2_mode = _q2(ctx, mode)
    lower = L
    upper = U
    if not (finL and finU):
        if not finL:
            lower = _valley_solve(
                ctx, 2, q_mode + log_thresh,
                L, mode, 0, 1,
                grad_tol, hess_tol, newt_tol, newt_max,
                boundary_eps_mult,
                bracket_init_step, bracket_max_expand, bracket_step_growth,
                backtrack_max_iters, backtrack_reduce,
            )
        if not finU:
            upper = _valley_solve(
                ctx, 1, q_mode + log_thresh,
                mode, U, 1, 0,
                grad_tol, hess_tol, newt_tol, newt_max,
                boundary_eps_mult,
                bracket_init_step, bracket_max_expand, bracket_step_growth,
                backtrack_max_iters, backtrack_reduce,
            )
    # Preserve the density-defined tail window before padding out to the
    # observed data range.  The padded window is useful for diagnostics, but
    # quadrature needs these inner breakpoints explicitly: otherwise a narrow
    # component with tiny responsibility on far-away observations can create
    # a panel hundreds of thousands of local widths long, on which both the
    # Gauss and Kronrod nodes miss the density-bearing edge.
    core_lower_out[0] = lower
    core_upper_out[0] = upper

    pad_num = 0.5 * (upper - lower)
    pad_data = 0.5 * (xb1 - xb0)
    pad = _agm(pad_num, pad_data)
    if pad <= 0.0:
        pad = pad_num if pad_num > pad_data else pad_data
        if pad < 0.0:
            pad = 0.0
    lo = (xb0 if xb0 < lower else lower) - pad
    hi = (xb1 if xb1 > upper else upper) + pad
    if not finL:
        lower = lo
    if not finU:
        upper = hi
    lower_out[0] = lower
    upper_out[0] = upper
    mode_out[0] = mode
    q_mode_out[0] = q_mode
    q2_mode_out[0] = q2_mode


cpdef tuple _q_window_and_mode(
    support,
    q_poly,
    boundary_amplitudes,
    x_data_bounds,
    log_thresh,
    grad_tol,
    hess_tol,
    newt_tol,
    newt_max,
    boundary_eps_mult,
    bracket_init_step,
    bracket_max_expand,
    bracket_step_growth,
    backtrack_max_iters,
    backtrack_reduce,
    bint include_core=False,
):
    """Compute the fitting window and the mode of the potential.

    Locates the mode of ``exp(-q)`` and the two tail points where
    ``q(z) - q(mode) = log_thresh``, then expands the result to cover the
    observed data range.  The window is the working domain for
    normalisation, quadrature, and post-fit diagnostics.

    As with :func:`_valley_q1_shift`, every tolerance is required; the
    values live in :mod:`._defaults`.

    Parameters
    ----------
    support : array_like, shape (2,)
        Distribution support in ``z`` coordinates; may be infinite.
    q_poly : array_like, shape (d+1,)
        Coefficients of the potential ``q``, constant term first.
    boundary_amplitudes : array_like, shape (2,)
        Canonical boundary amplitudes ``[aL, aU]``; unavailable sides are NaN.
    x_data_bounds : array_like, shape (2,)
        Observed data range in ``z`` coordinates; the window is expanded
        to cover it.
    log_thresh : float
        Drop in log-density, relative to the peak, defining the tail
        points.  ``_defaults.LOG_THRESH`` is ``-log(float64 eps)``.
    grad_tol, hess_tol, newt_tol : float
        Convergence tolerances passed to the Newton solver.
    newt_max : int
        Maximum Newton iterations before falling back to bisection.
    boundary_eps_mult : float
        Multiplier setting how close an iterate may approach a boundary
        singularity.
    bracket_init_step, bracket_max_expand, bracket_step_growth : float, int, float
        Initial step, expansion limit, and growth factor for the
        bisection bracket search.
    backtrack_max_iters : int
        Maximum backtracking steps per Newton iteration.
    backtrack_reduce : float
        Step-size reduction factor applied when backtracking.
    include_core : bool, optional
        If true, append the density-defined pre-padding window to the return
        tuple.  The scalar reference path uses these points as explicit
        quadrature breakpoints when the working window is data-padded.

    Returns
    -------
    tuple
        ``(window, mode, q_mode, q2_mode)`` by default.  With
        ``include_core=True`` the tuple has a fifth entry,
        ``core_window``, containing the density-defined tail points before
        data padding.

    Raises
    ------
    ValueError
        If *q_poly* is empty.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] sup = np.ascontiguousarray(support, dtype=np.float64).reshape(2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qp = np.ascontiguousarray(q_poly, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qb = np.ascontiguousarray(boundary_amplitudes, dtype=np.float64).reshape(2)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] xb = np.ascontiguousarray(x_data_bounds, dtype=np.float64).reshape(2)
    if qp.size < 1:
        raise ValueError("q_poly must be non-empty")
    cdef double ctl[11]
    ctl[0] = float(log_thresh)
    ctl[1] = float(grad_tol)
    ctl[2] = float(hess_tol)
    ctl[3] = float(newt_tol)
    ctl[4] = float(int(newt_max))
    ctl[5] = float(boundary_eps_mult)
    ctl[6] = float(bracket_init_step)
    ctl[7] = float(int(bracket_max_expand))
    ctl[8] = float(bracket_step_growth)
    ctl[9] = float(int(backtrack_max_iters))
    ctl[10] = float(backtrack_reduce)
    cdef Py_ssize_t nq = qp.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d1 = np.empty(max(nq, 1), dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] d2 = np.empty(max(nq, 1), dtype=np.float64)
    cdef _QCtx ctx
    cdef double L = float(sup[0])
    cdef double U = float(sup[1])
    if _finite(L) and _finite(U) and not (L < U):
        raise ValueError("support must satisfy support[0] < support[1]")
    _ctx_setup(&ctx, L, U, float(qb[0]), float(qb[1]), &qp[0], nq, &d1[0], &d2[0])
    cdef double mode, q_mode, q2_mode, lower, upper, core_lower, core_upper
    cdef double xb0 = float(xb[0])
    cdef double xb1 = float(xb[1])
    with nogil:
        _window_mode_c(
            &ctx, xb0, xb1, ctl, &lower, &upper, &mode, &q_mode, &q2_mode,
            &core_lower, &core_upper,
        )
    cdef cnp.ndarray[cnp.float64_t, ndim=1] win = np.empty(2, dtype=np.float64)
    win[0] = lower
    win[1] = upper
    if include_core:
        core = np.array([core_lower, core_upper], dtype=np.float64)
        return win, float(mode), float(q_mode), float(q2_mode), core
    return win, float(mode), float(q_mode), float(q2_mode)


# ==========================================================================
# Vectorised PDF and basis-function evaluation kernels
# ==========================================================================
cpdef cnp.ndarray _pdf_vec(
    cnp.ndarray x_arr,
    tuple support,
    cnp.ndarray q_poly_arr,
    cnp.ndarray boundary_amplitudes_arr,
    double log_norm,
):
    """Fused PDF evaluation: exp(-(polyval(x) + boundary_terms + log_norm)).
    Replaces the Python ``gibbus._model.vec`` PDF path with a single C loop that fuses
    Horner polynomial evaluation, boundary log terms, exponentiation,
    and support masking.  Avoids all intermediate array allocations.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.ascontiguousarray(x_arr, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] poly = np.ascontiguousarray(q_poly_arr, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] qb = np.ascontiguousarray(boundary_amplitudes_arr, dtype=np.float64).ravel()
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t npoly = poly.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(N, dtype=np.float64)
    cdef double Lx = <double>support[0]
    cdef double Ux = <double>support[1]
    cdef bint finL = _finite(Lx)
    cdef bint finU = _finite(Ux)
    if qb.shape[0] != 2:
        raise ValueError("boundary_amplitudes must have length 2")
    cdef double aL = qb[0], aU = qb[1]
    cdef bint hasL = _finite(aL) and (aL > 0.0) and finL
    cdef bint hasU = _finite(aU) and (aU > 0.0) and finU
    if N == 0:
        # ``&x[0]`` on an empty buffer is out of bounds (flagged by checked builds).
        return out
    cdef double* xp = <double*>&x[0]
    cdef double* pp = <double*>&poly[0]
    cdef double* op = <double*>&out[0]
    cdef Py_ssize_t i, k
    cdef double xi, q, dL, dU, val
    with nogil:
        for i in range(N):
            xi = xp[i]
            # NaN propagates, matching cdf() and ppf(); it is not "outside
            # the support".
            if isnan(xi):
                op[i] = NAN
                continue
            # Support masking
            if finL and xi < Lx:
                op[i] = 0.0
                continue
            if finU and xi > Ux:
                op[i] = 0.0
                continue
            # Horner polynomial evaluation
            q = pp[npoly - 1]
            for k in range(npoly - 2, -1, -1):
                q = q * xi + pp[k]
            # Boundary log terms
            if hasL:
                dL = xi - Lx
                if dL <= 0.0:
                    op[i] = 0.0
                    continue
                q = q - aL * log(dL)
            if hasU:
                dU = Ux - xi
                if dU <= 0.0:
                    op[i] = 0.0
                    continue
                q = q - aU * log(dU)
            # exp(-(q + log_norm))
            q = q + log_norm
            val = 0.0
            if q < 700.0:  # avoid overflow in exp
                val = c_exp(-q)
            if not _finite(val):
                val = 0.0
            op[i] = val
    return out
cpdef cnp.ndarray _polyval_vec(
    cnp.ndarray x_arr,
    cnp.ndarray poly_arr,
):
    """Vectorised Horner polynomial evaluation.
    Replacement for numpy.polynomial.polynomial.polyval when called
    on a 1-D array with a small coefficient vector.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x = np.ascontiguousarray(x_arr, dtype=np.float64).ravel()
    cdef cnp.ndarray[cnp.float64_t, ndim=1] poly = np.ascontiguousarray(poly_arr, dtype=np.float64).ravel()
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t npoly = poly.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(N, dtype=np.float64)
    if N == 0:
        return out
    if npoly == 0:
        out.fill(0.0)
        return out
    cdef double* xp = <double*>&x[0]
    cdef double* pp = <double*>&poly[0]
    cdef double* op = <double*>&out[0]
    cdef Py_ssize_t i, k
    cdef double xi, q
    with nogil:
        for i in range(N):
            xi = xp[i]
            if npoly == 0:
                op[i] = 0.0
            else:
                q = pp[npoly - 1]
                for k in range(npoly - 2, -1, -1):
                    q = q * xi + pp[k]
                op[i] = q
    return out


# ==========================================================================
# Complete state numerics in one nogil call
# ==========================================================================
#
# ``state_numerics`` does everything ``_NaturalCoreState._init_numerics`` and
# the fit-time moment requests do, in one pass: mode/window, the mode-aware
# breakpoints of ``numerics._mode_quad_points``, and one shared adaptive
# Gauss--Kronrod traversal integrating every statistic the Fisher geometry
# needs (normalizer, ordinary powers, power-weighted boundary logs, squared
# and crossed logs) against the shifted density.  The per-feature
# convergence test and the segment transforms match the compiled
# ``power_moments`` traversal.  At each panel the fifteen nodes are
# evaluated once into contiguous arrays; every feature is then two dot
# products over those arrays (SIMD reductions under ``-fopenmp-simd``).

from libc.stdlib cimport malloc, free
from libc.math cimport nextafter

cdef extern from * nogil:
    """
    #if defined(_MSC_VER)
      #define GIBBUS_SN_RESTRICT __restrict
      #define GIBBUS_SN_SIMD_SUM2
      #define GIBBUS_SN_SIMD
    #else
      #define GIBBUS_SN_RESTRICT __restrict__
      #define GIBBUS_SN_SIMD_SUM2 _Pragma("omp simd reduction(+:kr, ga)")
      #define GIBBUS_SN_SIMD _Pragma("omp simd")
    #endif

    /* Kronrod and Gauss sums of one feature over the panel nodes, then
       optionally advance the feature by one power of z. */
    static inline void gibbus_sn_feature(const double *GIBBUS_SN_RESTRICT wk,
                                         const double *GIBBUS_SN_RESTRICT wg,
                                         double *GIBBUS_SN_RESTRICT v,
                                         const double *GIBBUS_SN_RESTRICT z,
                                         int n, int advance,
                                         double *out_k, double *out_g)
    {
        double kr = 0.0, ga = 0.0;
        GIBBUS_SN_SIMD_SUM2
        for (int i = 0; i < n; ++i) { kr += wk[i] * v[i]; ga += wg[i] * v[i]; }
        if (advance) {
            GIBBUS_SN_SIMD
            for (int i = 0; i < n; ++i) v[i] *= z[i];
        }
        *out_k = kr;
        *out_g = ga;
    }

    static inline void gibbus_sn_product(const double *GIBBUS_SN_RESTRICT a,
                                         const double *GIBBUS_SN_RESTRICT b,
                                         double *GIBBUS_SN_RESTRICT out, int n)
    {
        GIBBUS_SN_SIMD
        for (int i = 0; i < n; ++i) out[i] = a[i] * b[i];
    }
    """
    void gibbus_sn_feature(const double* wk, const double* wg, double* v, const double* z,
                           int n, int advance, double* out_k, double* out_g)
    void gibbus_sn_product(const double* a, const double* b, double* out, int n)


# 15-point Kronrod nodes on [-1, 1] in ascending order with their Kronrod
# weights and the embedded 7-point Gauss weights (zero off the Gauss nodes).
cdef double _SN_X[15]
cdef double _SN_WK[15]
cdef double _SN_WG[15]


def _sn_init_rule():
    cdef double xgk[8]
    cdef double wgk[8]
    cdef double wg[4]
    cdef int i
    xgk[:] = [0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
              0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
              0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
              0.207784955007898467600689403773245, 0.0]
    wgk[:] = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
              0.104790010322250183839876322541518, 0.140653259715525918745189703095821,
              0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
              0.204432940075298892414161999234649, 0.209482141084727828012999174891714]
    wg[:] = [0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
             0.381830050505118944950369775488975, 0.417959183673469387755102040816327]
    for i in range(7):
        _SN_X[i] = -xgk[i]
        _SN_X[14 - i] = xgk[i]
        _SN_WK[i] = wgk[i]
        _SN_WK[14 - i] = wgk[i]
        _SN_WG[i] = 0.0
        _SN_WG[14 - i] = 0.0
    _SN_X[7] = 0.0
    _SN_WK[7] = wgk[7]
    _SN_WG[7] = wg[3]
    _SN_WG[1] = wg[0]
    _SN_WG[13] = wg[0]
    _SN_WG[3] = wg[1]
    _SN_WG[11] = wg[1]
    _SN_WG[5] = wg[2]
    _SN_WG[9] = wg[2]


_sn_init_rule()


cdef struct _SNGeom:
    double lo                   # segment
    double hi
    int transform               # 0 affine, 1 quadratic at lo, 2 quadratic at hi
    double L                    # support
    double U
    bint finL
    bint finU
    const double* q             # shifted potential polynomial
    Py_ssize_t nq
    double aL                   # density amplitudes (used when dens_L / dens_U)
    double aU
    bint dens_L
    bint dens_U
    bint feat_L                 # boundary-log features present
    bint feat_U
    int n_power                 # number of power features (orders 0..n_power-1)
    int n_log                   # power-weighted log features per side
    int n_features


cdef int _sn_panel(const _SNGeom* g, double ta, double tb,
                   double* kr, double* ga, double* err) noexcept nogil:
    """One vector GK15 panel on ``[ta, tb]`` of the segment parameter."""
    cdef double z[15]
    cdef double w[15]
    cdef double v[15]
    cdef double u[15]
    cdef double lL[15]
    cdef double lU[15]
    cdef double mid = 0.5 * (ta + tb)
    cdef double half = 0.5 * (tb - ta)
    cdef double width = g.hi - g.lo
    cdef double t, om, zz, jac, qz, up_lim, lo_lim
    cdef int n, k, f
    up_lim = g.U if g.finU else INFINITY
    lo_lim = g.L if g.finL else -INFINITY
    for n in range(15):
        t = mid + half * _SN_X[n]
        if g.transform == 1:
            zz = g.lo + width * t * t
            jac = 2.0 * width * t
        elif g.transform == 2:
            om = 1.0 - t
            zz = g.hi - width * om * om
            jac = 2.0 * width * om
        else:
            zz = g.lo + width * t
            jac = width
        if g.finL and zz <= g.L:
            zz = nextafter(g.L, up_lim)
        if g.finU and zz >= g.U:
            zz = nextafter(g.U, lo_lim)
        qz = _poly_eval(g.q, g.nq, zz)
        lL[n] = log(zz - g.L) if g.finL else 0.0
        lU[n] = log(g.U - zz) if g.finU else 0.0
        if g.dens_L:
            qz -= g.aL * lL[n]
        if g.dens_U:
            qz -= g.aU * lU[n]
        if qz < -1e-8:
            return 2
        if not (qz <= 745.0):       # +inf, NaN or negligible: no contribution
            w[n] = 0.0
        else:
            if qz < 0.0:
                qz = 0.0
            w[n] = c_exp(-qz) * jac * half
            if not _finite(w[n]):
                return 3
        z[n] = zz
    f = 0
    for n in range(15):
        v[n] = w[n]
    for k in range(g.n_power):
        gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 1, &kr[f], &ga[f])
        f += 1
    if g.feat_L:
        gibbus_sn_product(w, lL, v, 15)
        for k in range(g.n_log):
            gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 1, &kr[f], &ga[f])
            f += 1
    if g.feat_U:
        gibbus_sn_product(w, lU, v, 15)
        for k in range(g.n_log):
            gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 1, &kr[f], &ga[f])
            f += 1
    if g.feat_L:
        gibbus_sn_product(lL, lL, u, 15)
        gibbus_sn_product(w, u, v, 15)
        gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 0, &kr[f], &ga[f])
        f += 1
    if g.feat_U:
        gibbus_sn_product(lU, lU, u, 15)
        gibbus_sn_product(w, u, v, 15)
        gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 0, &kr[f], &ga[f])
        f += 1
    if g.feat_L and g.feat_U:
        gibbus_sn_product(lL, lU, u, 15)
        gibbus_sn_product(w, u, v, 15)
        gibbus_sn_feature(_SN_WK, _SN_WG, v, z, 15, 0, &kr[f], &ga[f])
        f += 1
    for k in range(f):
        err[k] = fabs(kr[k] - ga[k])
    return 0


cdef struct _SNWork:
    double* vals        # (limit, F) panel values
    double* errs        # (limit, F) panel error estimates
    double* ta          # (limit,) panel parameter bounds
    double* tb
    double* total       # (F,)
    double* total_err
    double* left
    double* right
    double* err_left
    double* err_right
    double* scratch


cdef int _sn_segment(const _SNGeom* g, double epsabs, double epsrel, int limit,
                     _SNWork* w, double* result) noexcept nogil:
    """Adaptive vector GK15 over one segment (the ``_pm_segment`` scheme)."""
    cdef int F = g.n_features
    cdef int count = 1, idx, i, k, status, pending
    cdef double mid, old_b, score, best, tol, denom, ratio
    status = _sn_panel(g, 0.0, 1.0, w.vals, w.scratch, w.errs)
    if status != 0:
        return status
    w.ta[0] = 0.0
    w.tb[0] = 1.0
    for k in range(F):
        w.total[k] = w.vals[k]
        w.total_err[k] = w.errs[k]
    while count < limit:
        pending = 0
        for k in range(F):
            tol = epsabs + epsrel * fabs(w.total[k])
            if w.total_err[k] > tol:
                pending = 1
                break
        if not pending:
            break
        idx = 0
        best = -1.0
        for i in range(count):
            score = 0.0
            for k in range(F):
                denom = epsabs + epsrel * fabs(w.total[k])
                if denom <= 0.0:
                    denom = 1e-300
                ratio = w.errs[i * F + k] / denom
                if ratio > score:
                    score = ratio
            if score > best:
                best = score
                idx = i
        old_b = w.tb[idx]
        mid = 0.5 * (w.ta[idx] + old_b)
        status = _sn_panel(g, w.ta[idx], mid, w.left, w.scratch, w.err_left)
        if status != 0:
            return status
        status = _sn_panel(g, mid, old_b, w.right, w.scratch, w.err_right)
        if status != 0:
            return status
        for k in range(F):
            w.total[k] += w.left[k] + w.right[k] - w.vals[idx * F + k]
            w.total_err[k] += w.err_left[k] + w.err_right[k] - w.errs[idx * F + k]
            if w.total_err[k] < 0.0:
                w.total_err[k] = 0.0
            w.vals[idx * F + k] = w.left[k]
            w.errs[idx * F + k] = w.err_left[k]
            w.vals[count * F + k] = w.right[k]
            w.errs[count * F + k] = w.err_right[k]
        w.tb[idx] = mid
        w.ta[count] = mid
        w.tb[count] = old_b
        count += 1
    for k in range(F):
        result[k] = w.total[k]
    return 0


cdef int _sn_points(double lo, double hi, double mode, double scale,
                    double core_lo, double core_hi, double L, double U,
                    double* out) noexcept nogil:
    """Port of ``numerics._mode_quad_points``: sorted unique interior breakpoints."""
    cdef double vals[16]
    cdef int n = 0, i, i2, j, m
    cdef double span = hi - lo
    cdef double v, frac
    if _finite(scale) and scale > 0.0 and span / scale > 128.0:
        vals[0] = mode - 8.0 * scale
        vals[1] = mode
        vals[2] = mode + 8.0 * scale
        n = 3
        # The density-defined tail points are essential only when the working
        # window is very wide in local component scales.  In that regime data
        # padding can create enormous outer panels whose quadrature nodes all
        # miss the density-bearing inner edge.  Keeping ordinary windows on
        # their historical breakpoint set avoids needless numerical drift.
        if _finite(core_lo):
            vals[n] = core_lo
            n += 1
        if _finite(core_hi):
            vals[n] = core_hi
            n += 1
    else:
        vals[0] = mode
        n = 1

    if _finite(span) and span > 0.0:
        if _finite(L) and lo == L:
            frac = 1.0
            for i in range(5):
                frac *= 0.015625            # 2^-6, 2^-12, ..., 2^-30
                vals[n] = lo + span * frac
                n += 1
        if _finite(U) and hi == U:
            frac = 1.0
            for i in range(5):
                frac *= 0.015625
                vals[n] = hi - span * frac
                n += 1
    m = 0
    for i in range(n):
        v = vals[i]
        if not (_finite(v) and lo < v and v < hi):
            continue
        # insertion into the sorted unique output
        j = m
        while j > 0 and out[j - 1] > v:
            j -= 1
        if j > 0 and out[j - 1] == v:
            continue
        if j < m and out[j] == v:
            continue
        for i2 in range(m, j, -1):
            out[i2] = out[i2 - 1]
        out[j] = v
        m += 1
    return m



def state_numerics(
    const double[::1] support,
    const double[::1] q_poly,
    const double[::1] amplitudes,
    const double[::1] data_bounds,
    bint lower_basis,
    bint upper_basis,
    const int[::1] kinds,
    const int[::1] lengths,
    const double[:, ::1] coefficients,
    const double[::1] controls,
    double epsabs,
    double epsrel,
    int limit,
):
    """Normalize a state and integrate its Fisher statistics in one call.

    Parameters
    ----------
    support : ndarray, shape (2,)
        Canonical support.
    q_poly : ndarray, shape (nq,)
        Ascending potential polynomial (before the mode shift).
    amplitudes : ndarray, shape (2,)
        Canonical boundary amplitudes (NaN when the basis is absent).
    data_bounds : ndarray, shape (2,)
        Canonical data range for the window.
    lower_basis, upper_basis : bool
        Whether the boundary-log bases (hence log features) are present.
    kinds, lengths, coefficients : ndarray
        First-partial descriptors as packed by
        ``objective._finite_partial_descriptors`` (0 polynomial with
        ``lengths[i]`` ascending coefficients, 1 lower log, 2 upper log);
        a partial of kind 1 is ``-log(z - L)``, of kind 2 ``-log(U - z)``.
    controls : ndarray, shape (11,)
        Mode-search controls in ``_window_mode_c`` order.
    epsabs, epsrel : float
        Per-feature quadrature tolerances (``epsabs`` split over segments).
    limit : int
        Maximum panels per segment.

    Returns
    -------
    status : int
        0 success; 1 non-finite shifted potential; 2 negative shifted
        potential at a node; 3 non-finite weight; 9 allocation failure.  On
        a nonzero status only the geometry entries are meaningful.
    geometry : ndarray, shape (6,)
        ``window[0], window[1], mode, q_min, q2_mode, local_scale``.
    points : ndarray
        Interior quadrature breakpoints.
    shifted_z : float
        Shifted normalizer ``Z``; ``log Z = -q_min + log(shifted_z)``.
    moments : ndarray
        Normalized features: ``E[z^k]`` (``k < 2 w - 1``), ``E[z^k log(z-L)]``
        and ``E[z^k log(U-z)]`` (``k < w``, present sides), then
        ``E[log^2(z-L)]``, ``E[log^2(U-z)]``, ``E[log(z-L) log(U-z)]``
        (present sides), with ``w`` the descriptor width.
    means : ndarray, shape (n,)
        Model means of the partials.
    fisher : ndarray, shape (n, n)
        Model covariance of the partials.
    """
    cdef Py_ssize_t nq = q_poly.shape[0]
    cdef Py_ssize_t n = kinds.shape[0]
    cdef Py_ssize_t width = coefficients.shape[1] if n > 0 else 1
    if nq < 1 or support.shape[0] != 2 or amplitudes.shape[0] != 2 or data_bounds.shape[0] != 2:
        raise ValueError("invalid state geometry")
    if controls.shape[0] != 11 or limit < 1 or n != lengths.shape[0] or n != coefficients.shape[0]:
        raise ValueError("invalid state numerics controls")
    cdef int n_power = <int>(2 * width - 1)
    cdef int n_log = <int>width
    cdef bint featL = lower_basis and _finite(support[0])
    cdef bint featU = upper_basis and _finite(support[1])
    cdef int F = n_power + (n_log if featL else 0) + (n_log if featU else 0) \
        + (1 if featL else 0) + (1 if featU else 0) + (1 if (featL and featU) else 0)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] geometry = np.empty(6, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] points_buf = np.empty(16, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] moments = np.zeros(F, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] means = np.zeros(n, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] fisher = np.zeros((n, n), dtype=np.float64)
    cdef double* work = <double*>malloc((3 * nq + 2 * limit * F + 2 * limit + 8 * F) * sizeof(double))
    if work == NULL:
        raise MemoryError("state numerics allocation failed")
    cdef int status = 0, npts = 0
    cdef double shifted_z = 0.0
    try:
        with nogil:
            status = _state_numerics_c(
                &support[0], &q_poly[0], nq, &amplitudes[0], &data_bounds[0],
                featL, featU, n_power, n_log, F,
                &kinds[0] if n > 0 else NULL, &lengths[0] if n > 0 else NULL,
                &coefficients[0, 0] if n > 0 else NULL, n, width,
                &controls[0], epsabs, epsrel, limit, work,
                &geometry[0], &points_buf[0], &npts, &shifted_z,
                &moments[0], &means[0] if n > 0 else NULL,
                &fisher[0, 0] if n > 0 else NULL,
            )
    finally:
        free(work)
    return status, geometry, points_buf[:npts].copy(), shifted_z, moments, means, fisher


cdef int _state_numerics_c(
    const double* support, const double* q_poly, Py_ssize_t nq, const double* amplitudes,
    const double* data_bounds, bint featL, bint featU, int n_power, int n_log, int F,
    const int* kinds, const int* lengths, const double* coefficients, Py_ssize_t n,
    Py_ssize_t width, const double* controls, double epsabs, double epsrel, int limit,
    double* work, double* geometry, double* points, int* npts, double* shifted_z,
    double* moments, double* means, double* fisher,
) noexcept nogil:
    cdef double L = support[0]
    cdef double U = support[1]
    cdef double aL = amplitudes[0]
    cdef double aU = amplitudes[1]
    cdef double* shifted = work
    cdef double* d1 = work + nq
    cdef double* d2 = work + 2 * nq
    cdef double* seg_out
    cdef _SNWork w
    cdef _QCtx ctx
    cdef _SNGeom g
    cdef double lower, upper, core_lower, core_upper
    cdef double mode, q_min, q2_mode, span, local_scale, lo, hi
    cdef double regular_L, regular_U, Z, a, value
    cdef double edges[18]
    cdef int status, i, j, k, nseg, ai, bi, a_i, b_i
    cdef int f_log_L, f_log_U, f_sq_L, f_sq_U, f_cross
    cdef Py_ssize_t ii, jj
    # Workspace: shifted, d1, d2 (nq each); panel values and errors
    # (limit x F each); panel bounds (limit each); seven F-vectors; output.
    w.vals = work + 3 * nq
    w.errs = w.vals + limit * F
    w.ta = w.errs + limit * F
    w.tb = w.ta + limit
    w.total = w.tb + limit
    w.total_err = w.total + F
    w.left = w.total_err + F
    w.right = w.left + F
    w.err_left = w.right + F
    w.err_right = w.err_left + F
    w.scratch = w.err_right + F
    seg_out = w.scratch + F

    # Mode and window, with the collapsed-endpoint retry of _init_numerics.
    _ctx_setup(&ctx, L, U, aL, aU, q_poly, nq, d1, d2)
    _window_mode_c(
        &ctx, data_bounds[0], data_bounds[1], controls,
        &lower, &upper, &mode, &q_min, &q2_mode, &core_lower, &core_upper,
    )
    if (not _finite(q_min)) and ((_finite(aL) and aL > 0.0) or (_finite(aU) and aU > 0.0)):
        span = fabs(mode)
        if span < 1.0:
            span = 1.0
        regular_L = aL
        regular_U = aU
        if _finite(L) and mode - L <= 1e-8 * span:
            regular_L = 0.0
        if _finite(U) and U - mode <= 1e-8 * span:
            regular_U = 0.0
        _ctx_setup(&ctx, L, U, regular_L, regular_U, q_poly, nq, d1, d2)
        _window_mode_c(
            &ctx, data_bounds[0], data_bounds[1], controls,
            &lower, &upper, &mode, &q_min, &q2_mode, &core_lower, &core_upper,
        )
    local_scale = 1.0
    if _finite(q2_mode) and q2_mode > 0.0:
        local_scale = 1.0 / sqrt(q2_mode)
    if not _finite(q_min):
        q_min = 0.0
    geometry[0] = lower
    geometry[1] = upper
    geometry[2] = mode
    geometry[3] = q_min
    geometry[4] = q2_mode
    geometry[5] = local_scale
    npts[0] = _sn_points(
        lower, upper, mode, local_scale, core_lower, core_upper, L, U, points
    )

    for ii in range(nq):
        shifted[ii] = q_poly[ii]
        if not _finite(shifted[ii]):
            return 1
    shifted[0] -= q_min

    # Shared adaptive traversal over [lower, points..., upper].
    edges[0] = lower
    for i in range(npts[0]):
        edges[i + 1] = points[i]
    edges[npts[0] + 1] = upper
    nseg = npts[0] + 1
    g.L = L
    g.U = U
    g.finL = _finite(L)
    g.finU = _finite(U)
    g.q = shifted
    g.nq = nq
    g.aL = aL
    g.aU = aU
    g.dens_L = g.finL and _finite(aL) and aL > 0.0
    g.dens_U = g.finU and _finite(aU) and aU > 0.0
    g.feat_L = featL
    g.feat_U = featU
    g.n_power = n_power
    g.n_log = n_log
    g.n_features = F
    for k in range(F):
        moments[k] = 0.0
    for i in range(nseg):
        lo = edges[i]
        hi = edges[i + 1]
        if not hi > lo:
            continue
        g.lo = lo
        g.hi = hi
        g.transform = 0
        if g.finL and lo == L:
            g.transform = 1
        elif g.finU and hi == U:
            g.transform = 2
        status = _sn_segment(&g, epsabs / nseg, epsrel, limit, &w, seg_out)
        if status != 0:
            return status
        for k in range(F):
            moments[k] += seg_out[k]
    Z = moments[0]
    shifted_z[0] = Z
    if not (Z > 0.0 and _finite(Z)):
        return 1
    for k in range(F):
        moments[k] /= Z
    moments[0] = 1.0

    # Feature offsets.
    f_log_L = n_power
    f_log_U = f_log_L + (n_log if featL else 0)
    f_sq_L = f_log_U + (n_log if featU else 0)
    f_sq_U = f_sq_L + (1 if featL else 0)
    f_cross = f_sq_U + (1 if featU else 0)

    # Means of the partials: polynomial c . E[z^k]; logs -E[log d].
    for ii in range(n):
        if kinds[ii] == 0:
            value = 0.0
            for k in range(lengths[ii]):
                value += coefficients[ii * width + k] * moments[k]
        elif kinds[ii] == 1:
            value = -moments[f_log_L] if featL else 0.0
        else:
            value = -moments[f_log_U] if featU else 0.0
        means[ii] = value

    # Second moments, then covariance.
    for ii in range(n):
        for jj in range(ii, n):
            value = 0.0
            if kinds[ii] == 0 and kinds[jj] == 0:
                for a_i in range(lengths[ii]):
                    a = coefficients[ii * width + a_i]
                    for b_i in range(lengths[jj]):
                        value += a * coefficients[jj * width + b_i] * moments[a_i + b_i]
            elif kinds[ii] == 0 or kinds[jj] == 0:
                # polynomial x log: -(c . E[z^k log d])
                if kinds[ii] == 0:
                    ai = <int>ii
                    bi = kinds[jj]
                else:
                    ai = <int>jj
                    bi = kinds[ii]
                j = f_log_L if bi == 1 else f_log_U
                for k in range(lengths[ai]):
                    value -= coefficients[ai * width + k] * moments[j + k]
            elif kinds[ii] == kinds[jj]:
                value = moments[f_sq_L] if kinds[ii] == 1 else moments[f_sq_U]
            else:
                value = moments[f_cross]
            value -= means[ii] * means[jj]
            fisher[ii * n + jj] = value
            fisher[jj * n + ii] = value
    return 0
