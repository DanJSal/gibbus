# cython: language_level=3
"""Compiled construction of the spectral CDF and PPF representations.

Compiled implementation of the spectral CDF/PPF construction used by fitted
models.  Runtime densities are described by data and built here; the test-only
Python construction harness exercises the same control flow for arbitrary
callables and forced-failure scenarios.  Everything after argument unpacking
runs without the GIL: the map selection, the adaptive
Chebyshev partition of the transformed density with Bernstein positivity
lifts, the quadrature re-measurement of uncertified panels, the packed
evaluator arrays, and the monotone inverse in log-odds with its
root-inversion validation.

The density is described by data, not a Python callable: a weighted sum of
fitted components, each the canonical kernel ``exp(-(q(z) - a_L log(z - L)
- a_U log(U - z) + c))`` of :func:`gibbus._model._state_kernels._pdf_vec`
at ``z = sigma x + mu``, times ``|sigma|`` and masked to the component's
public support when it is a public-coordinate view.  One component with
``sigma = 1, mu = 0`` is the canonical state density used at packing time;
several are the base-space mixture density.

Chebyshev series are evaluated with the same two Clenshaw variants as the
Python construction harness: NumPy's (``chebval``, panel metrics, integration)
and the packed evaluator's (inversion, validation), so each quantity reproduces
the harness arithmetic.  Node sets, value transforms and Chebyshev-Bernstein
matrices come from the shared cached helper tables.
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport exp, log, log1p, fabs, hypot, isfinite, isnan, isinf, nextafter
from libc.math cimport INFINITY, NAN
from libc.stdlib cimport malloc, free
from libc.string cimport memmove, memset

from ._certify cimport _chebyshev_lower_bound_c

cnp.import_array()

cdef double _EPS = 2.220446049250313e-16
cdef double _DBL_EPSILON = 2.220446049250313e-16
cdef double _TINY = 2.2250738585072014e-308
cdef double _PMIN = 5e-324
cdef double _PMAX = 0.9999999999999999
cdef int _MAXN = 72               # largest node/coefficient count handled


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

cdef struct Density:
    int K
    bint view                     # public-coordinate views: mask to [lo, hi]
    const double* q               # concatenated normalized polynomials
    const Py_ssize_t* qoff
    const Py_ssize_t* nq
    const double* par             # per component: Lz Uz aL aU log_norm mu sigma jf weight lo hi


cdef inline double _kernel(
    const double* q, Py_ssize_t nq, double z, const double* p
) noexcept nogil:
    """``_pdf_vec`` at one point (``p``: Lz, Uz, aL, aU, log_norm)."""
    cdef double Lz = p[0], Uz = p[1], aL = p[2], aU = p[3]
    cdef bint finL = isfinite(Lz)
    cdef bint finU = isfinite(Uz)
    cdef double v, d, out
    cdef Py_ssize_t k
    if isnan(z):
        return NAN
    if finL and z < Lz:
        return 0.0
    if finU and z > Uz:
        return 0.0
    v = q[nq - 1]
    for k in range(nq - 2, -1, -1):
        v = v * z + q[k]
    if isfinite(aL) and aL > 0.0 and finL:
        d = z - Lz
        if d <= 0.0:
            return 0.0
        v = v - aL * log(d)
    if isfinite(aU) and aU > 0.0 and finU:
        d = Uz - z
        if d <= 0.0:
            return 0.0
        v = v - aU * log(d)
    v = v + p[4]
    out = 0.0
    if v < 700.0:
        out = exp(-v)
    if not isfinite(out):
        out = 0.0
    return out


cdef inline double _density(const Density* d, double x) noexcept nogil:
    cdef double total = 0.0, v, z
    cdef const double* p
    cdef int k
    for k in range(d.K):
        p = d.par + 11 * k
        z = p[6] * x + p[5]
        v = p[7] * _kernel(d.q + d.qoff[k], d.nq[k], z, p)
        if d.view:
            if not (isfinite(x) and x >= p[9] and x <= p[10]):
                v = NAN if isnan(x) else 0.0
        total += p[8] * v
    return total


# ---------------------------------------------------------------------------
# Compactifying maps (``cdf._Map``)
# ---------------------------------------------------------------------------

cdef struct Map:
    int kind                      # 0 finite 1 lower 2 upper 3 real 4 lower_c 5 upper_c
    double L
    double U
    double center
    double scale


cdef inline double _x_from_z(const Map* mp, double z) noexcept nogil:
    cdef double x, y_edge, t_edge, t
    if mp.kind == 0:
        x = mp.center + mp.scale * z
        if z <= -1.0:
            x = mp.L
        if z >= 1.0:
            x = mp.U
        return x
    if mp.kind == 1:
        return mp.L + mp.scale * (1.0 + z) / (1.0 - z)
    if mp.kind == 2:
        return mp.U - mp.scale * (1.0 - z) / (1.0 + z)
    if mp.kind == 3:
        return mp.center + (2.0 * mp.scale * z) / (1.0 - z * z)
    if mp.kind == 4:
        y_edge = (mp.L - mp.center) / mp.scale
        t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
        t = t_edge + 0.5 * (z + 1.0) * (1.0 - t_edge)
    else:
        y_edge = (mp.U - mp.center) / mp.scale
        t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
        t = -1.0 + 0.5 * (z + 1.0) * (t_edge + 1.0)
    x = mp.center + (2.0 * mp.scale * t) / (1.0 - t * t)
    if mp.kind == 4 and z <= -1.0:
        x = mp.L
    if mp.kind == 5 and z >= 1.0:
        x = mp.U
    return x


cdef inline double _jac(const Map* mp, double z) noexcept nogil:
    cdef double y_edge, t_edge, dt_dz, t, w
    if mp.kind == 0:
        return mp.scale
    if mp.kind == 1:
        w = 1.0 - z
        return 2.0 * mp.scale / (w * w)
    if mp.kind == 2:
        w = 1.0 + z
        return 2.0 * mp.scale / (w * w)
    if mp.kind == 3:
        w = 1.0 - z * z
        return 2.0 * mp.scale * (1.0 + z * z) / (w * w)
    if mp.kind == 4:
        y_edge = (mp.L - mp.center) / mp.scale
        t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
        dt_dz = 0.5 * (1.0 - t_edge)
        t = t_edge + (z + 1.0) * dt_dz
    else:
        y_edge = (mp.U - mp.center) / mp.scale
        t_edge = y_edge / (1.0 + hypot(1.0, y_edge))
        dt_dz = 0.5 * (t_edge + 1.0)
        t = -1.0 + (z + 1.0) * dt_dz
    w = 1.0 - t * t
    return (2.0 * mp.scale * (1.0 + t * t) / (w * w)) * dt_dz


cdef inline double _g(const Density* d, const Map* mp, double z) noexcept nogil:
    """Transformed density ``f(x(z)) dx/dz`` (``SpectralCDF._g_with_map``)."""
    cdef double xx, jj, v
    if (mp.kind == 3 or mp.kind == 2 or mp.kind == 5) and not (z > -1.0):
        return 0.0
    if (mp.kind == 3 or mp.kind == 1 or mp.kind == 4) and not (z < 1.0):
        return 0.0
    xx = _x_from_z(mp, z)
    if isfinite(mp.L) and z <= -1.0:
        xx = nextafter(mp.L, INFINITY)
    if isfinite(mp.U) and z >= 1.0:
        xx = nextafter(mp.U, -INFINITY)
    jj = _jac(mp, z)
    v = _density(d, xx) * jj
    if not (isfinite(v) and v >= 0.0):
        v = 0.0
    return v


# ---------------------------------------------------------------------------
# Chebyshev helpers
# ---------------------------------------------------------------------------

cdef inline double _chebval_np(double x, const double* c, int n) noexcept nogil:
    """NumPy's ``chebval`` recurrence (``_panel_kernels._chebval_one``)."""
    cdef double c0, c1, tmp
    cdef int i
    if n == 0:
        return 0.0
    if n == 1:
        return c[0]
    c0 = c[n - 2]
    c1 = c[n - 1]
    for i in range(n - 3, -1, -1):
        tmp = c0
        c0 = c[i] - c1
        c1 = tmp + 2.0 * x * c1
    return c0 + x * c1


cdef inline double _chebval_ev(const double* c, int nc, double u) noexcept nogil:
    """The packed evaluator's recurrence (``_cdf_eval._cheb_generic``)."""
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef int k
    for k in range(nc - 1, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        b2 = b1
        b1 = b0
    return u * b1 - b2 + c[0]


cdef inline void _cheb_value_derivative(
    const double* c, int nc, double u, double* value, double* derivative
) noexcept nogil:
    cdef double b0, b1 = 0.0, b2 = 0.0
    cdef double d0, d1 = 0.0, d2 = 0.0
    cdef int k
    for k in range(nc - 1, 0, -1):
        b0 = 2.0 * u * b1 - b2 + c[k]
        d0 = 2.0 * b1 + 2.0 * u * d1 - d2
        b2 = b1
        b1 = b0
        d2 = d1
        d1 = d0
    value[0] = u * b1 - b2 + c[0]
    derivative[0] = b1 + u * d1 - d2


cdef inline double _chebint(
    const double* c, int n, double scl, double* out
) noexcept nogil:
    """``_panel_kernels.chebint_scaled`` (``out``: ``n + 1``); returns the mass."""
    cdef int j
    memset(out, 0, (n + 1) * sizeof(double))
    if n == 1 and c[0] == 0.0:
        return 0.0
    out[1] = c[0] * scl
    if n > 1:
        out[2] = c[1] * scl / 4.0
    for j in range(2, n):
        out[j + 1] = c[j] * scl / (2.0 * (j + 1))
        out[j - 1] -= c[j] * scl / (2.0 * (j - 1))
    out[0] = -_chebval_np(0.0, out, n + 1)
    return _chebval_np(1.0, out, n + 1) - _chebval_np(-1.0, out, n + 1)


cdef inline int _chebder(const double* c, int n, double* out) noexcept nogil:
    """``_panel_kernels.chebder``; returns the output length."""
    cdef int j
    if n <= 1:
        out[0] = 0.0
        return 1
    memset(out, 0, (n - 1) * sizeof(double))
    if n == 2:
        out[0] = c[1]
        return 1
    out[n - 2] = 2.0 * (n - 1) * c[n - 1]
    out[n - 3] = 2.0 * (n - 2) * c[n - 2]
    for j in range(n - 4, -1, -1):
        out[j] = out[j + 2] + 2.0 * (j + 1) * c[j + 1]
    out[0] *= 0.5
    return n - 1


cdef inline double _expit(double x) noexcept nogil:
    return 1.0 / (1.0 + exp(-x))


cdef inline double _logit(double p) noexcept nogil:
    return log(p) - log1p(-p)


# ---------------------------------------------------------------------------
# Cached shared tables (nodes, value transforms, Bernstein matrices)
# ---------------------------------------------------------------------------

cdef class Tables:
    """Pointers to the shared cached node/transform/Bernstein arrays."""

    cdef object _keep
    cdef const double* lob[73]
    cdef const double* trans[73]
    cdef const double* mid[160]
    cdef const double* ratio[73]
    cdef const double* gl_nodes
    cdef const double* gl_weights

    def __init__(
        self, lobatto, transforms, midpoints, ratios, gauss_nodes, gauss_weights
    ):
        """Bind cached tables.

        Parameters
        ----------
        lobatto, transforms, ratios : dict
            Degree -> Lobatto nodes, value transform, Chebyshev-Bernstein matrix.
        midpoints : dict
            Count -> midpoint validation nodes.
        gauss_nodes, gauss_weights : numpy.ndarray
            Gauss--Legendre rule for the uncertified-mass re-measurement.
        """
        cdef const double[::1] v
        cdef const double[:, ::1] m
        keep = []
        for i in range(73):
            self.lob[i] = NULL
            self.trans[i] = NULL
            self.ratio[i] = NULL
        for i in range(160):
            self.mid[i] = NULL
        for n, arr in lobatto.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keep.append(a)
            v = a
            self.lob[int(n)] = &v[0]
        for n, arr in transforms.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keep.append(a)
            m = a
            self.trans[int(n)] = &m[0, 0]
        for n, arr in ratios.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keep.append(a)
            m = a
            self.ratio[int(n)] = &m[0, 0]
        for n, arr in midpoints.items():
            a = np.ascontiguousarray(arr, dtype=np.float64)
            keep.append(a)
            v = a
            self.mid[int(n)] = &v[0]
        a = np.ascontiguousarray(gauss_nodes, dtype=np.float64)
        keep.append(a)
        v = a
        self.gl_nodes = &v[0]
        a = np.ascontiguousarray(gauss_weights, dtype=np.float64)
        keep.append(a)
        v = a
        self.gl_weights = &v[0]
        self._keep = keep


cdef struct Tab:
    const double** lob
    const double** trans
    const double** mid
    const double** ratio
    const double* gl_nodes
    const double* gl_weights


cdef inline double _lift(
    const Tab* t,
    const double* coeff,
    int ncoeff,
    int max_subdivide,
    double* work,
    int* depths,
) noexcept nogil:
    """``SpectralCDF._positivity_lift_bernstein``."""
    cdef double lower, scale
    cdef int i
    lower = _chebyshev_lower_bound_c(coeff, ncoeff, t.ratio[ncoeff - 1], max_subdivide,
                                     work, depths)
    if not isfinite(lower) or lower >= 0.0:
        return 0.0
    scale = 0.0
    for i in range(ncoeff):
        if fabs(coeff[i]) > scale:
            scale = fabs(coeff[i])
    if scale < 1.0:
        scale = 1.0
    return -lower + 32.0 * _EPS * scale


# ---------------------------------------------------------------------------
# Forward CDF
# ---------------------------------------------------------------------------

cdef struct CdfOpts:
    const int* degrees
    int n_degrees
    double rel_tol
    double abs_tol
    double coeff_tol
    int max_depth
    int max_panels


cdef struct CPanel:
    double a
    double b
    double mass
    double fit_error
    double error_mass
    double tail_ratio
    double lift
    int depth
    int certified
    int ncoeff                  # coeff length (degree + 1); icoeff has ncoeff + 1


cdef double _trial_score(
    const Density* d, const Map* mp, const Tab* t, int degree
) noexcept nogil:
    cdef int n = 20 if degree > 20 else degree
    cdef int m = 2 * n + 3
    cdef double vals[72]
    cdef double coeff[72]
    cdef double exact
    cdef const double* u = t.lob[n]
    cdef const double* T = t.trans[n]
    cdef const double* uv = t.mid[m]
    cdef double total, err = 0.0, scale = 0.0, cmax = 0.0, tail = 0.0, v
    cdef int i, j
    for i in range(n + 1):
        vals[i] = _g(d, mp, u[i])
        if fabs(vals[i]) > scale:
            scale = fabs(vals[i])
    for i in range(n + 1):
        total = 0.0
        for j in range(n + 1):
            total += T[i * (n + 1) + j] * vals[j]
        coeff[i] = total
    for i in range(m):
        exact = _g(d, mp, uv[i])
        if fabs(exact) > scale:
            scale = fabs(exact)
        v = fabs(exact - _chebval_np(uv[i], coeff, n + 1))
        if v > err or isnan(v):
            err = v
    if scale < 1e-15:
        scale = 1e-15
    for i in range(n + 1):
        if fabs(coeff[i]) > cmax:
            cmax = fabs(coeff[i])
    for i in range(n - 3 if n >= 3 else 0, n + 1):
        if fabs(coeff[i]) > tail:
            tail = fabs(coeff[i])
    if cmax < 1e-15:
        cmax = 1e-15
    return err / scale + tail / cmax


cdef int _fit_cdf_panel(
    const Density* d,
    const Map* mp,
    const Tab* t,
    const CdfOpts* o,
    double a,
    double b,
    int depth,
    int n,
    CPanel* out,
    double* coeff,
    double* icoeff,
    double* work,
    int* depths,
) noexcept nogil:
    """``SpectralCDF._fit_panel``; returns whether the panel converged."""
    cdef int m = 2 * n + 3
    cdef double vals[72]
    cdef double exact[72]
    cdef const double* u = t.lob[n]
    cdef const double* T = t.trans[n]
    cdef const double* uv = t.mid[m]
    cdef double total, fit_error = 0.0, data_scale = 0.0
    cdef double tail_abs = 0.0, coeff_scale = 0.0
    cdef double av, pred, err, scale, width, mass_scale
    cdef double tol_mass, tail_tol_mass, lift, mass
    cdef int i, j, converged
    for i in range(n + 1):
        vals[i] = _g(d, mp, 0.5 * ((b - a) * u[i] + (a + b)))
    for i in range(m):
        exact[i] = _g(d, mp, 0.5 * ((b - a) * uv[i] + (a + b)))
    # cdf_panel_metrics
    for i in range(n + 1):
        total = 0.0
        for j in range(n + 1):
            total += T[i * (n + 1) + j] * vals[j]
        coeff[i] = total
        av = fabs(total)
        if av > coeff_scale:
            coeff_scale = av
    for i in range(n + 1):
        av = fabs(vals[i])
        if av > data_scale:
            data_scale = av
    for i in range(m):
        av = fabs(exact[i])
        if av > data_scale:
            data_scale = av
        pred = _chebval_np(uv[i], coeff, n + 1)
        err = fabs(exact[i] - pred)
        if err > fit_error:
            fit_error = err
    for i in range(n - 3 if n >= 3 else 0, n + 1):
        av = fabs(coeff[i])
        if av > tail_abs:
            tail_abs = av
    scale = data_scale if data_scale > 1e-300 else 1e-300
    out.tail_ratio = tail_abs / (coeff_scale if coeff_scale > 1e-300 else 1e-300)
    width = b - a
    mass_scale = width * scale
    tol_mass = o.abs_tol + o.rel_tol * mass_scale
    tail_tol_mass = o.abs_tol + o.coeff_tol * mass_scale
    converged = (width * fit_error <= tol_mass) and (width * tail_abs <= tail_tol_mass)
    lift = _lift(t, coeff, n + 1, 8, work, depths)
    if lift > 0.0:
        coeff[0] += lift
    if width * lift > tol_mass:
        converged = 0
    mass = _chebint(coeff, n + 1, 0.5 * (b - a), icoeff)
    out.a = a
    out.b = b
    out.mass = mass
    out.fit_error = fit_error
    out.error_mass = width * (fit_error + lift)
    out.lift = lift
    out.depth = depth
    out.certified = converged
    out.ncoeff = n + 1
    return converged


cdef inline double _cdf_priority(const CPanel* p, const double* coeff) noexcept nogil:
    cdef double width = p.b - p.a, tail = 0.0, priority
    cdef int i, start = p.ncoeff - 4 if p.ncoeff > 4 else 0
    for i in range(start, p.ncoeff):
        if fabs(coeff[i]) > tail:
            tail = fabs(coeff[i])
    priority = p.error_mass
    if width * tail > priority:
        priority = width * tail
    if isnan(p.error_mass) or isnan(width * tail):
        priority = NAN
    return priority if isfinite(priority) else INFINITY


cdef struct CdfBuild:
    int n_leaves
    int* leaf                   # record index per leaf, in order
    CPanel* rec
    double* coeff               # record r: coeff + r * S
    double* icoeff              # record r: icoeff + r * (S + 1)
    int S
    int n_rec
    int exhausted


cdef int _fit_best(const Density* d, const Map* mp, const Tab* t, const CdfOpts* o,
                   CdfBuild* bd, double a, double b, int depth, double* work,
                   int* depths) noexcept nogil:
    """Fit into a fresh record, p-refining over the degree options; returns it."""
    cdef int r = bd.n_rec, k
    bd.n_rec += 1
    for k in range(o.n_degrees):
        if _fit_cdf_panel(
            d,
            mp,
            t,
            o,
            a,
            b,
            depth,
            o.degrees[k],
            &bd.rec[r],
            bd.coeff + r * bd.S,
            bd.icoeff + r * (bd.S + 1),
            work,
            depths,
        ):
            return r
    return r


cdef int _build_partition(
    const Density* d,
    const Map* mp,
    const Tab* t,
    const CdfOpts* o,
    CdfBuild* bd,
    const double* breaks,
    int nbreaks,
    double* work,
    int* depths,
) noexcept nogil:
    cdef int i, j, best_j, r, left, right
    cdef double priority, best, mid, min_width = 5e-13
    cdef CPanel* p
    bd.n_leaves = 0
    for i in range(nbreaks - 1):
        bd.leaf[bd.n_leaves] = _fit_best(
            d, mp, t, o, bd, breaks[i], breaks[i + 1], 0, work, depths
        )
        bd.n_leaves += 1
    bd.exhausted = 0
    while True:
        best_j = -1
        best = -INFINITY
        for j in range(bd.n_leaves):
            p = &bd.rec[bd.leaf[j]]
            if p.certified or p.depth >= o.max_depth or not ((p.b - p.a) >= min_width):
                continue
            priority = _cdf_priority(p, bd.coeff + bd.leaf[j] * bd.S)
            if best_j < 0 or priority > best:
                best = priority
                best_j = j
        if best_j < 0:
            break
        if bd.n_leaves >= o.max_panels:
            bd.exhausted = 1
            break
        r = bd.leaf[best_j]
        mid = 0.5 * (bd.rec[r].a + bd.rec[r].b)
        left = _fit_best(
            d, mp, t, o, bd, bd.rec[r].a, mid, bd.rec[r].depth + 1, work, depths
        )
        right = _fit_best(
            d, mp, t, o, bd, mid, bd.rec[r].b, bd.rec[r].depth + 1, work, depths
        )
        memmove(bd.leaf + best_j + 2, bd.leaf + best_j + 1,
                (bd.n_leaves - best_j - 1) * sizeof(int))
        bd.leaf[best_j] = left
        bd.leaf[best_j + 1] = right
        bd.n_leaves += 1
    return 0


cdef int _recertify(
    const Density* d, const Map* mp, const Tab* t, CdfBuild* bd
) noexcept nogil:
    """``SpectralCDF._recertify_panel_masses``; returns the count replaced."""
    cdef int j, r, s, g, fixed = 0, i
    cdef double half, midp, total, lo, hi, sh, sc, zz
    cdef double val, acc, true_mass, old, ratio, height
    cdef CPanel* p
    cdef double* coeff
    cdef double* icoeff
    for j in range(bd.n_leaves):
        r = bd.leaf[j]
        p = &bd.rec[r]
        if p.certified:
            continue
        half = 0.5 * (p.b - p.a)
        if not (half > 0.0):
            continue
        midp = 0.5 * (p.a + p.b)
        total = 0.0
        for s in range(8):
            lo = -1.0 + 0.25 * s
            hi = -1.0 + 0.25 * (s + 1)
            if s == 7:
                hi = 1.0
            sh = 0.5 * (hi - lo)
            sc = 0.5 * (hi + lo)
            acc = 0.0
            for g in range(24):
                zz = midp + half * (sc + sh * t.gl_nodes[g])
                val = _g(d, mp, zz)
                if not isfinite(val):
                    val = 0.0
                acc += t.gl_weights[g] * val
            total += acc * sh
        true_mass = total * half
        if not isfinite(true_mass) or true_mass < 0.0:
            true_mass = 0.0
        old = p.mass
        coeff = bd.coeff + r * bd.S
        icoeff = bd.icoeff + r * (bd.S + 1)
        if old > 0.0 and isfinite(old):
            ratio = true_mass / old
            for i in range(p.ncoeff + 1):
                icoeff[i] = icoeff[i] * ratio
            for i in range(p.ncoeff):
                coeff[i] = coeff[i] * ratio
        else:
            height = true_mass / (2.0 * half) if half > 0.0 else 0.0
            coeff[0] = height
            p.ncoeff = 1
            icoeff[0] = 0.0
            icoeff[1] = height * half
        p.mass = true_mass
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Packed forward evaluator (``SpectralCDF._build_cython_evaluator``) and the
# inversion helpers the inverse construction needs
# ---------------------------------------------------------------------------

cdef struct CdfEval:
    int m
    int stride
    const double* breaks        # m + 1
    const double* offsets       # m (cum_mass[:-1])
    const double* cum           # m + 1
    const double* coeffs        # m x stride, normalized, A_j(-1) = 0
    const int* ncoeff
    const double* a             # raw panel edges
    const double* b
    const double* icoeff        # raw antiderivatives, m x istride
    const int* nicoeff
    int istride
    double total


cdef inline int _panel_index(const CdfEval* ev, double z) noexcept nogil:
    cdef int lo = 0, hi = ev.m, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if z < ev.breaks[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    if lo >= ev.m:
        return ev.m - 1
    return lo


cdef inline double _eval_compact(const CdfEval* ev, double z) noexcept nogil:
    cdef int j
    cdef double u, val
    if isnan(z):
        return z
    if z <= -1.0:
        return 0.0
    if z >= 1.0:
        return 1.0
    j = _panel_index(ev, z)
    u = (2.0 * z - (ev.breaks[j] + ev.breaks[j + 1])) / (
        ev.breaks[j + 1] - ev.breaks[j]
    )
    val = _chebval_ev(ev.coeffs + j * ev.stride, ev.ncoeff[j], u) + ev.offsets[j]
    if val <= 0.0:
        return 0.0
    if val >= 1.0:
        return 1.0
    return val


cdef double _invert_fraction(const CdfEval* ev, int j, double frac) noexcept nogil:
    """``SpectralEvaluator._invert_panel_fraction_one``."""
    cdef double lo = -1.0, hi = 1.0, u, candidate, value, deriv, f
    cdef double flo, fhi, target, local_mass, scale, tol
    cdef const double* c = ev.coeffs + j * ev.stride
    cdef int nc = ev.ncoeff[j], _it
    if frac <= 8.0 * _DBL_EPSILON:
        return ev.breaks[j]
    if frac >= 1.0 - 8.0 * _DBL_EPSILON:
        return ev.breaks[j + 1]
    local_mass = _chebval_ev(c, nc, 1.0)
    target = frac * local_mass
    flo = _chebval_ev(c, nc, lo) - target
    fhi = local_mass - target
    u = 2.0 * frac - 1.0
    if u <= lo or u >= hi:
        u = 0.0
    scale = local_mass
    if fabs(target) > scale:
        scale = fabs(target)
    if scale < 1e-300:
        scale = 1e-300
    tol = 8.0 * _DBL_EPSILON * scale
    for _it in range(64):
        _cheb_value_derivative(c, nc, u, &value, &deriv)
        f = value - target
        if f <= 0.0:
            lo = u
            flo = f
        else:
            hi = u
            fhi = f
        if fabs(f) <= tol:
            break
        if hi - lo <= 8.0 * _DBL_EPSILON * (1.0 + fabs(u)):
            break
        if deriv > 0.0 and not isinf(deriv) and not isnan(deriv):
            candidate = u - f / deriv
            if (
                candidate <= lo
                or candidate >= hi
                or isinf(candidate)
                or isnan(candidate)
            ):
                candidate = 0.5 * (lo + hi)
        else:
            candidate = 0.5 * (lo + hi)
        if candidate == u:
            break
        u = candidate
    _cheb_value_derivative(c, nc, u, &value, &deriv)
    f = value - target
    if fabs(flo) <= fabs(f) and fabs(flo) <= fabs(fhi):
        u = lo
    elif fabs(fhi) < fabs(f):
        u = hi
    return 0.5 * (
        (ev.breaks[j + 1] - ev.breaks[j]) * u + (ev.breaks[j] + ev.breaks[j + 1])
    )


cdef inline double _clip01(double v) noexcept nogil:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


cdef double _invert_source(const CdfEval* ev, int j, double p) noexcept nogil:
    cdef double pa = ev.cum[j], pb = ev.cum[j + 1]
    if p <= pa:
        return ev.breaks[j]
    if p >= pb:
        return ev.breaks[j + 1]
    if not pb > pa:
        return ev.breaks[j]
    return _invert_fraction(ev, j, _clip01((p - pa) / (pb - pa)))


cdef double _invert_global(
    const CdfEval* ev, double p, bint prefer_left
) noexcept nogil:
    cdef int j, lo, hi, mid, k, found
    if p <= 0.0:
        return -1.0
    if p >= 1.0:
        return 1.0
    # searchsorted(cum, p, side) - 1 over m + 1 entries
    lo = 0
    hi = ev.m + 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if (ev.cum[mid] < p) if prefer_left else (ev.cum[mid] <= p):
            lo = mid + 1
        else:
            hi = mid
    j = lo - 1
    if j < 0:
        j = 0
    if j > ev.m - 1:
        j = ev.m - 1
    if not (ev.cum[j] <= p and p <= ev.cum[j + 1]) or not (ev.cum[j + 1] > ev.cum[j]):
        found = -1
        for k in range(ev.m):
            if ev.cum[k] <= p and ev.cum[k + 1] >= p and ev.cum[k + 1] > ev.cum[k]:
                found = k
                if prefer_left:
                    break
        if found >= 0:
            j = found
    return _invert_source(ev, j, p)


cdef inline double _clip_p(double p) noexcept nogil:
    if p < _PMIN:
        return _PMIN
    if p > _PMAX:
        return _PMAX
    return p


cdef double _exact_z_for_r(const CdfEval* ev, int j, double r) noexcept nogil:
    cdef double p = _clip_p(_expit(r))
    if ev.cum[j] <= p and p <= ev.cum[j + 1] and ev.cum[j + 1] > ev.cum[j]:
        return _invert_source(ev, j, p)
    return _invert_global(ev, p, False)


cdef double _exact_z_for_r_many_one(const CdfEval* ev, int j, double r) noexcept nogil:
    """One element of ``_exact_z_for_r_many`` (inside: clipped-fraction path)."""
    cdef double p = _clip_p(_expit(r)), pa, pb
    if p >= ev.cum[j] and p <= ev.cum[j + 1] and ev.cum[j + 1] > ev.cum[j]:
        pa = ev.cum[j]
        pb = ev.cum[j + 1]
        return _invert_fraction(ev, j, _clip01((p - pa) / (pb - pa)))
    return _invert_global(ev, p, False)


# ---------------------------------------------------------------------------
# Inverse (PPF)
# ---------------------------------------------------------------------------

cdef struct PpfOpts:
    const int* degrees
    int n_degrees
    double prob_tol
    int max_depth
    int max_panels
    int certify_subdivide


cdef struct IPanel:
    double ra
    double rb
    double pa
    double pb
    double za
    double zb
    double fit_error
    double logit_residual
    double prob_residual
    double tail_abs
    double derivative_lower
    int depth
    int source
    int ncoeff
    int ok


cdef inline double _spacing(double x) noexcept nogil:
    return (
        fabs(nextafter(x, INFINITY) - x)
        if x >= 0.0
        else fabs(nextafter(x, -INFINITY) - x)
    )


cdef void _fail_ipanel(IPanel* out, double* coeff) noexcept nogil:
    out.fit_error = INFINITY
    out.logit_residual = INFINITY
    out.prob_residual = INFINITY
    out.tail_abs = INFINITY
    out.derivative_lower = -INFINITY
    out.ncoeff = 2
    coeff[0] = -1.0
    coeff[1] = 1.0
    out.ok = 0


cdef int _fit_ppf_panel(
    const Density* d,
    const Map* mp,
    const Tab* t,
    const PpfOpts* o,
    const CdfEval* ev,
    int src,
    double ra,
    double rb,
    double pa,
    double pb,
    double za,
    double zb,
    int depth,
    int degree,
    IPanel* out,
    double* coeff,
    double* work,
    int* depths,
) noexcept nogil:
    """``SpectralPPF._fit_panel``; returns whether the panel certified."""
    cdef int n = degree - 1, m = 2 * degree + 5, i, k, nd, ndc, nic
    cdef double r
    cdef double probs[72]
    cdef double z[72]
    cdef double qvals[72]
    cdef double qcoeff[72]
    cdef double icoeff[73]
    cdef double dicoeff[73]
    cdef double dcoeff[73]
    cdef const double* u = t.lob[n]
    cdef const double* T = t.trans[n]
    cdef const double* uv = t.mid[m]
    cdef double sa = ev.a[src], sb = ev.b[src], usrc, dens, factor, total
    cdef double lift, imass, ileft, rv, pv, ztrue, vtrue, vpred, zpred, pback, psafe
    cdef double fit_error = 0.0, prob_resid = 0.0, logit_resid = 0.0, tail_abs = 0.0
    cdef double prob_floor = 0.0, dlower
    cdef bint bad = False
    out.ra = ra
    out.rb = rb
    out.pa = pa
    out.pb = pb
    out.za = za
    out.zb = zb
    out.depth = depth
    out.source = src
    factor = (rb - ra) / (zb - za)
    nd = _chebder(ev.icoeff + src * ev.istride, ev.nicoeff[src], dicoeff)
    for i in range(n + 1):
        r = 0.5 * ((rb - ra) * u[i] + (ra + rb))
        probs[i] = _expit(r)
        z[i] = _exact_z_for_r_many_one(ev, src, r)
        usrc = (2.0 * z[i] - (sa + sb)) / (sb - sa)
        dens = (_chebval_np(usrc, dicoeff, nd) / ev.total) * (2.0 / (sb - sa))
        qvals[i] = factor * probs[i] * (1.0 - probs[i]) / dens
        if not isfinite(qvals[i]) or qvals[i] < 0.0:
            bad = True
    if bad:
        # Rare fallback to the exact transformed density.
        bad = False
        for i in range(n + 1):
            dens = _g(d, mp, z[i])
            qvals[i] = factor * probs[i] * (1.0 - probs[i]) / dens
            if not isfinite(qvals[i]) or qvals[i] < 0.0:
                bad = True
    if bad:
        _fail_ipanel(out, coeff)
        return 0
    # ``degree_options`` are validated to be >= 2, so the loop below always
    # initializes every coefficient that is subsequently consumed.  Zero the
    # fixed workspace as well so checked/optimized C compilers do not have to
    # infer that non-empty-loop invariant across the helper call.
    memset(qcoeff, 0, sizeof(qcoeff))
    for i in range(n + 1):
        total = 0.0
        for k in range(n + 1):
            total += T[i * (n + 1) + k] * qvals[k]
        qcoeff[i] = total
    lift = _lift(t, qcoeff, n + 1, o.certify_subdivide, work, depths)
    if lift > 0.0:
        qcoeff[0] += lift
    imass = _chebint(qcoeff, n + 1, 1.0, icoeff)
    nic = n + 2
    ileft = _chebval_np(-1.0, icoeff, nic)
    if not isfinite(imass) or imass <= 0.0:
        _fail_ipanel(out, coeff)
        return 0
    for i in range(nic):
        coeff[i] = icoeff[i] * (2.0 / imass)
    coeff[0] += -1.0 - (2.0 * ileft / imass)
    out.ncoeff = nic
    # Interlaced validation against accurate root inversions.
    for i in range(m):
        rv = 0.5 * ((rb - ra) * uv[i] + (ra + rb))
        pv = _expit(rv)
        ztrue = _exact_z_for_r_many_one(ev, src, rv)
        vtrue = (2.0 * ztrue - (za + zb)) / (zb - za)
        vpred = _chebval_np(uv[i], coeff, nic)
        zpred = 0.5 * ((zb - za) * vpred + (za + zb))
        if fabs(vpred - vtrue) > fit_error:
            fit_error = fabs(vpred - vtrue)
        pback = _eval_compact(ev, zpred)
        if fabs(pback - pv) > prob_resid:
            prob_resid = fabs(pback - pv)
        psafe = _clip_p(pback)
        if fabs(_logit(psafe) - rv) > logit_resid:
            logit_resid = fabs(_logit(psafe) - rv)
        if 64.0 * _spacing(pv) > prob_floor:
            prob_floor = 64.0 * _spacing(pv)
    for i in range(nic - 4 if nic > 4 else 0, nic):
        if fabs(coeff[i]) > tail_abs:
            tail_abs = fabs(coeff[i])
    ndc = _chebder(coeff, nic, dcoeff)
    dlower = _chebyshev_lower_bound_c(
        dcoeff, ndc, t.ratio[ndc - 1], o.certify_subdivide, work, depths
    )
    out.fit_error = fit_error
    out.logit_residual = logit_resid
    out.prob_residual = prob_resid
    out.tail_abs = tail_abs
    out.derivative_lower = dlower
    out.ok = (prob_resid <= (o.prob_tol if o.prob_tol > prob_floor else prob_floor)
              and dlower >= -128.0 * _EPS)
    return out.ok


cdef inline double _ppf_priority(const IPanel* p) noexcept nogil:
    if (not isfinite(p.prob_residual) or not isfinite(p.derivative_lower)
            or p.derivative_lower < -128.0 * _EPS):
        return INFINITY
    return p.prob_residual if p.prob_residual > 0.0 else 0.0


cdef struct PpfBuild:
    int n_leaves
    int* leaf
    IPanel* rec
    double* coeff               # record r: coeff + r * S
    int S
    int n_rec
    int fail                    # 0 ok, 1 depth/spacing, 2 budget
    int fail_rec


cdef int _ppf_fit_best(
    const Density* d,
    const Map* mp,
    const Tab* t,
    const PpfOpts* o,
    const CdfEval* ev,
    PpfBuild* bd,
    int src,
    double ra,
    double rb,
    double pa,
    double pb,
    double za,
    double zb,
    int depth,
    double* work,
    int* depths,
) noexcept nogil:
    cdef int r = bd.n_rec, k
    bd.n_rec += 1
    for k in range(o.n_degrees):
        if _fit_ppf_panel(d, mp, t, o, ev, src, ra, rb, pa, pb, za, zb, depth,
                          o.degrees[k], &bd.rec[r], bd.coeff + r * bd.S, work, depths):
            return r
    return r


cdef int _ppf_partition(
    const Density* d,
    const Map* mp,
    const Tab* t,
    const PpfOpts* o,
    const CdfEval* ev,
    PpfBuild* bd,
    int n_seed,
    const int* seed_src,
    const double* seed,
    double* work,
    int* depths,
) noexcept nogil:
    """``SpectralPPF._build_partition``; sets ``bd.fail`` instead of raising."""
    cdef int i, j, best_j, r, left, right, n_failed
    cdef double best, priority, rm, pm, zm
    cdef IPanel* p
    bd.n_leaves = 0
    bd.fail = 0
    for i in range(n_seed):
        bd.leaf[bd.n_leaves] = _ppf_fit_best(
            d,
            mp,
            t,
            o,
            ev,
            bd,
            seed_src[i],
            seed[6 * i],
            seed[6 * i + 1],
            seed[6 * i + 2],
            seed[6 * i + 3],
            seed[6 * i + 4],
            seed[6 * i + 5],
            0,
            work,
            depths,
        )
        bd.n_leaves += 1
    while True:
        best_j = -1
        best = -INFINITY
        n_failed = 0
        for j in range(bd.n_leaves):
            p = &bd.rec[bd.leaf[j]]
            if p.ok:
                continue
            n_failed += 1
            priority = _ppf_priority(p)
            if best_j < 0 or priority > best:
                best = priority
                best_j = j
        if best_j < 0:
            return 0
        r = bd.leaf[best_j]
        p = &bd.rec[r]
        if p.depth >= o.max_depth or not (p.rb > nextafter(p.ra, INFINITY)):
            bd.fail = 1
            bd.fail_rec = r
            return 1
        if bd.n_leaves >= o.max_panels:
            bd.fail = 2
            bd.fail_rec = n_failed
            return 1
        rm = 0.5 * (p.ra + p.rb)
        pm = _clip_p(_expit(rm))
        zm = _exact_z_for_r(ev, p.source, rm)
        left = _ppf_fit_best(
            d,
            mp,
            t,
            o,
            ev,
            bd,
            p.source,
            p.ra,
            rm,
            p.pa,
            pm,
            p.za,
            zm,
            p.depth + 1,
            work,
            depths,
        )
        p = &bd.rec[r]
        right = _ppf_fit_best(
            d,
            mp,
            t,
            o,
            ev,
            bd,
            p.source,
            rm,
            p.rb,
            pm,
            p.pb,
            zm,
            p.zb,
            p.depth + 1,
            work,
            depths,
        )
        memmove(bd.leaf + best_j + 2, bd.leaf + best_j + 1,
                (bd.n_leaves - best_j - 1) * sizeof(int))
        bd.leaf[best_j] = left
        bd.leaf[best_j + 1] = right
        bd.n_leaves += 1


# ---------------------------------------------------------------------------
# Python entry points
# ---------------------------------------------------------------------------

cdef class DensitySpec:
    """Owns the arrays of one density description (see the module docstring)."""

    cdef object _keep
    cdef Density d

    def __init__(self, components, bint view):
        """Pack components.

        Parameters
        ----------
        components : sequence of tuple
            ``(q_poly, Lz, Uz, aL, aU, log_norm, mu, sigma, jf, weight, lo, hi)``
            per component.
        view : bool
            Mask each component to ``[lo, hi]`` (public-coordinate views).
        """
        cdef const double[::1] qv
        cdef const Py_ssize_t[::1] ov
        cdef const Py_ssize_t[::1] nv
        cdef const double[::1] pv
        k = len(components)
        if k < 1:
            raise ValueError("at least one component is required")
        polys = [
            np.ascontiguousarray(c[0], dtype=np.float64).reshape(-1) for c in components
        ]
        if any(p.size < 1 or not np.all(np.isfinite(p)) for p in polys):
            raise ValueError("component polynomials must be finite and non-empty")
        q = np.ascontiguousarray(np.concatenate(polys))
        nq = np.array([p.size for p in polys], dtype=np.intp)
        off = np.zeros(k, dtype=np.intp)
        off[1:] = np.cumsum(nq[:k - 1])
        par = np.ascontiguousarray(
            np.array(
                [[float(v) for v in c[1:]] for c in components], dtype=np.float64
            ).reshape(-1)
        )
        if par.size != 11 * k:
            raise ValueError("each component needs eleven scalar parameters")
        self._keep = (q, nq, off, par)
        qv = q
        ov = off
        nv = nq
        pv = par
        self.d.K = k
        self.d.view = view
        self.d.q = &qv[0]
        self.d.qoff = &ov[0]
        self.d.nq = &nv[0]
        self.d.par = &pv[0]

    def pdf(self, x):
        """Evaluate the described density (for tests)."""
        xs = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        out = np.empty_like(xs)
        cdef double[::1] xv = xs
        cdef double[::1] ov = out
        cdef Py_ssize_t i
        with nogil:
            for i in range(xv.shape[0]):
                ov[i] = _density(&self.d, xv[i])
        return out


cdef void _tab_from(Tables tables, Tab* t):
    t.lob = tables.lob
    t.trans = tables.trans
    t.mid = tables.mid
    t.ratio = tables.ratio
    t.gl_nodes = tables.gl_nodes
    t.gl_weights = tables.gl_weights


def build_cdf(
    DensitySpec density,
    Tables tables,
    int map_kind,
    double L,
    double U,
    double center,
    double scale,
    bint choose_scale,
    double s0,
    const double[::1] breaks,
    const int[::1] degrees,
    double rel_tol,
    double abs_tol,
    double coeff_tol,
    int max_depth,
    int max_panels,
):
    """Build the adaptive spectral CDF (``SpectralCDF`` construction).

    Parameters
    ----------
    density : DensitySpec
        Density description.
    tables : Tables
        Cached shared tables for every degree used.
    map_kind, L, U, center, scale : int, float
        Compactifying map; ``scale`` is ignored when ``choose_scale``.
    choose_scale : bool
        Select the map scale from ``s0`` times the baseline multipliers.
    s0 : float
        Baseline scale for ``choose_scale``.
    breaks : ndarray
        Sorted unique seed breaks including -1 and 1.
    degrees : ndarray of int32
        Sorted degree options.
    rel_tol, abs_tol, coeff_tol : float
        Panel tolerances.
    max_depth, max_panels : int
        Refinement limits.

    Returns
    -------
    tuple
        ``(scale, panels, coeff, ncoeff, icoeff, exhausted, n_recertified)``
        where ``panels`` is ``(m, 7)`` float (a, b, mass, fit_error,
        error_mass, tail_ratio, lift) plus depth/certified in ``ncoeff``'s
        companion array (see the Python wrapper).
    """
    cdef Map mp
    cdef Tab t
    cdef CdfOpts o
    cdef CdfBuild bd
    cdef int maxdeg = degrees[degrees.shape[0] - 1], i, j, r, nrec_cap, fixed = 0
    cdef int ms = 10
    cdef double best_score = INFINITY, best_s = 0.0, sc, trial
    cdef double mults[7]
    cdef double* work = NULL
    cdef int* depths = NULL
    mults[:] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    _tab_from(tables, &t)
    if maxdeg + 2 > _MAXN or 2 * maxdeg + 3 > 150:
        raise ValueError("degree options exceed the compiled table size")
    nrec_cap = 2 * max_panels + breaks.shape[0] + 4
    bd.S = maxdeg + 1
    bd.leaf = <int*> malloc((max_panels + breaks.shape[0] + 2) * sizeof(int))
    bd.rec = <CPanel*> malloc(nrec_cap * sizeof(CPanel))
    bd.coeff = <double*> malloc(nrec_cap * bd.S * sizeof(double))
    bd.icoeff = <double*> malloc(nrec_cap * (bd.S + 1) * sizeof(double))
    work = <double*> malloc(((13 + ms) * (_MAXN + 1) + 8) * sizeof(double))
    depths = <int*> malloc((ms + 2) * sizeof(int))
    bd.n_rec = 0
    if (bd.leaf == NULL or bd.rec == NULL or bd.coeff == NULL or bd.icoeff == NULL
            or work == NULL or depths == NULL):
        free(bd.leaf)
        free(bd.rec)
        free(bd.coeff)
        free(bd.icoeff)
        free(work)
        free(depths)
        raise MemoryError("spectral CDF workspace")
    o.degrees = &degrees[0]
    o.n_degrees = degrees.shape[0]
    o.rel_tol = rel_tol
    o.abs_tol = abs_tol
    o.coeff_tol = coeff_tol
    o.max_depth = max_depth
    o.max_panels = max_panels
    mp.kind = map_kind
    mp.L = L
    mp.U = U
    mp.center = center
    mp.scale = scale
    with nogil:
        if choose_scale:
            for i in range(7):
                sc = s0 * mults[i]
                if sc < _TINY:
                    sc = _TINY
                mp.scale = sc
                trial = _trial_score(&density.d, &mp, &t, maxdeg)
                if i == 0 or trial < best_score:
                    best_score = trial
                    best_s = sc
            mp.scale = best_s
        _build_partition(
            &density.d, &mp, &t, &o, &bd, &breaks[0], breaks.shape[0], work, depths
        )
        fixed = _recertify(&density.d, &mp, &t, &bd)
    m = bd.n_leaves
    panels = np.empty((m, 7), dtype=np.float64)
    ints = np.empty((m, 3), dtype=np.int32)
    coeff = np.zeros((m, bd.S), dtype=np.float64)
    icoeff = np.zeros((m, bd.S + 1), dtype=np.float64)
    cdef double[:, ::1] pv = panels
    cdef int[:, ::1] iv = ints
    cdef double[:, ::1] cv = coeff
    cdef double[:, ::1] icv = icoeff
    cdef CPanel* p
    for j in range(m):
        r = bd.leaf[j]
        p = &bd.rec[r]
        pv[j, 0] = p.a
        pv[j, 1] = p.b
        pv[j, 2] = p.mass
        pv[j, 3] = p.fit_error
        pv[j, 4] = p.error_mass
        pv[j, 5] = p.tail_ratio
        pv[j, 6] = p.lift
        iv[j, 0] = p.depth
        iv[j, 1] = p.certified
        iv[j, 2] = p.ncoeff
        for i in range(p.ncoeff):
            cv[j, i] = bd.coeff[r * bd.S + i]
        for i in range(p.ncoeff + 1):
            icv[j, i] = bd.icoeff[r * (bd.S + 1) + i]
    exhausted = bd.exhausted
    free(bd.leaf)
    free(bd.rec)
    free(bd.coeff)
    free(bd.icoeff)
    free(work)
    free(depths)
    return mp.scale, panels, ints, coeff, icoeff, bool(exhausted), fixed


def build_ppf(
    DensitySpec density,
    Tables tables,
    int map_kind,
    double L,
    double U,
    double center,
    double scale,
    const double[::1] breaks,
    const double[::1] offsets,
    const double[::1] cum,
    const double[:, ::1] ev_coeffs,
    const int[::1] ev_ncoeff,
    const double[::1] a,
    const double[::1] b,
    const double[:, ::1] icoeff,
    const int[::1] nicoeff,
    double total,
    const int[::1] seed_src,
    const double[:, ::1] seed,
    const int[::1] degrees,
    double prob_tol,
    int max_depth,
    int max_panels,
    int certify_subdivide,
):
    """Build the monotone spectral inverse (``SpectralPPF`` construction).

    Returns
    -------
    tuple
        ``(status, panels, ints, coeff, detail)``: ``status`` 0 success, 1 a
        leaf reached the depth/spacing limit (``detail`` its row), 2 the
        panel budget ran out (``detail`` the uncertified-leaf count).
        ``panels`` is ``(m, 11)`` float (ra, rb, pa, pb, za, zb, fit_error,
        logit_residual, prob_residual, tail_abs, derivative_lower) and
        ``ints`` ``(m, 3)`` (depth, source, ncoeff).  On failure the arrays
        hold every record so the caller can format the message.
    """
    cdef Map mp
    cdef Tab t
    cdef PpfOpts o
    cdef PpfBuild bd
    cdef CdfEval ev
    cdef int maxdeg = degrees[degrees.shape[0] - 1], n_seed = seed_src.shape[0]
    cdef int i, j, r, m, nrec_cap, status = 0, ms = certify_subdivide
    cdef double* work = NULL
    cdef int* depths = NULL
    if 2 * maxdeg + 5 > 150 or maxdeg + 2 > _MAXN:
        raise ValueError("degree options exceed the compiled table size")
    _tab_from(tables, &t)
    mp.kind = map_kind
    mp.L = L
    mp.U = U
    mp.center = center
    mp.scale = scale
    ev.m = a.shape[0]
    ev.stride = ev_coeffs.shape[1]
    ev.breaks = &breaks[0]
    ev.offsets = &offsets[0]
    ev.cum = &cum[0]
    ev.coeffs = &ev_coeffs[0, 0]
    ev.ncoeff = &ev_ncoeff[0]
    ev.a = &a[0]
    ev.b = &b[0]
    ev.icoeff = &icoeff[0, 0]
    ev.nicoeff = &nicoeff[0]
    ev.istride = icoeff.shape[1]
    ev.total = total
    o.degrees = &degrees[0]
    o.n_degrees = degrees.shape[0]
    o.prob_tol = prob_tol
    o.max_depth = max_depth
    o.max_panels = max_panels
    o.certify_subdivide = certify_subdivide
    nrec_cap = 2 * max_panels + n_seed + 4
    bd.S = maxdeg + 2
    bd.n_rec = 0
    bd.leaf = <int*> malloc((max_panels + n_seed + 2) * sizeof(int))
    bd.rec = <IPanel*> malloc(nrec_cap * sizeof(IPanel))
    bd.coeff = <double*> malloc(nrec_cap * bd.S * sizeof(double))
    work = <double*> malloc(((13 + ms) * (_MAXN + 1) + 8) * sizeof(double))
    depths = <int*> malloc((ms + 2) * sizeof(int))
    if (
        bd.leaf == NULL
        or bd.rec == NULL
        or bd.coeff == NULL
        or work == NULL
        or depths == NULL
    ):
        free(bd.leaf)
        free(bd.rec)
        free(bd.coeff)
        free(work)
        free(depths)
        raise MemoryError("spectral PPF workspace")
    with nogil:
        status = _ppf_partition(&density.d, &mp, &t, &o, &ev, &bd, n_seed, &seed_src[0],
                                &seed[0, 0], work, depths)
    if status == 0:
        order = [bd.leaf[j] for j in range(bd.n_leaves)]
    else:
        order = list(range(bd.n_rec))
    m = len(order)
    panels = np.empty((m, 11), dtype=np.float64)
    ints = np.empty((m, 3), dtype=np.int32)
    coeff = np.zeros((m, bd.S), dtype=np.float64)
    cdef double[:, ::1] pv = panels
    cdef int[:, ::1] iv = ints
    cdef double[:, ::1] cv = coeff
    cdef IPanel* p
    for j in range(m):
        r = order[j]
        p = &bd.rec[r]
        pv[j, 0] = p.ra
        pv[j, 1] = p.rb
        pv[j, 2] = p.pa
        pv[j, 3] = p.pb
        pv[j, 4] = p.za
        pv[j, 5] = p.zb
        pv[j, 6] = p.fit_error
        pv[j, 7] = p.logit_residual
        pv[j, 8] = p.prob_residual
        pv[j, 9] = p.tail_abs
        pv[j, 10] = p.derivative_lower
        iv[j, 0] = p.depth
        iv[j, 1] = p.source
        iv[j, 2] = p.ncoeff
        for i in range(p.ncoeff):
            cv[j, i] = bd.coeff[r * bd.S + i]
    detail = bd.fail_rec if status != 0 else 0
    free(bd.leaf)
    free(bd.rec)
    free(bd.coeff)
    free(work)
    free(depths)
    return int(status), panels, ints, coeff, int(detail)
