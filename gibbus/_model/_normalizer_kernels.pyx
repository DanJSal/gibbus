# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled normalizer: ``log Z``, ``E[Phi]``, ``Cov[Phi]`` with an error bound.

``basis_state_c`` is the compiled body of ``basis_normalizer._basis_state``
(see that module for the method): properness from order-1 tail atoms,
reference potential and tail truncation, cell resolution by interval
enclosures, Gauss--Kronrod on the kept cells (nested tanh-sinh on cells
ending at an endpoint singularity) in batches, the error budget, and centered
moments.  Failures return a status (the Python wrapper raises
``_QuadratureFailure``); there is no fallback path.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport (
    INFINITY,
    NAN,
    M_PI,
    asinh,
    ceil,
    copysign,
    cosh,
    exp,
    fabs,
    isfinite,
    floor,
    log,
    log1p,
    nextafter,
    sinh,
)
from libc.stdlib cimport calloc, free, malloc, qsort, realloc
from libc.string cimport memcpy

from ._basis_kernels cimport (
    GROWTH_IMPROPER,
    GROWTH_INCONCLUSIVE,
    BasisKernel,
    BasisTable,
    Iv,
    PotentialPrep,
    TailKernel,
    TailResult,
    TailTable,
    basis_enclosure_cell,
    eval_point,
    growth_status,
    logaddexp,
    npmax,
    npmin,
    potential_enclosure_cell,
    potential_point,
    prep_size,
    prepare_potential,
    tail_bound,
    tail_verdict,
    tail_work_size,
)

from .gauss_kronrod import WG10, WK21, XK21

cnp.import_array()

cdef double _EPS = 2.220446049250313e-16
cdef double _TS_TMAX = 3.6
cdef int _TS_LEVEL0 = 2
cdef int _GRID = 257
# Below this log weight a node's moment products are formed in logs.
cdef double _LOG_FAR = -600.0
# Above this magnitude a column's moment products are formed in logs.
cdef double _BIG_COLUMN = 1e150
# Highest tanh-sinh level of the cached node table (``Ws.ts``).
cdef int _TS_LEVELS = 30
# The Gauss--Kronrod 10/21 rule (``gauss_kronrod``): abscissae ``x >= 0``
# descending, Kronrod weights, Gauss weights (0 at Kronrod-only nodes).
cdef double _GK_X[11]
cdef double _GK_WK[11]
cdef double _GK_WG[11]
_GK_X[:] = [float(v) for v in XK21]
_GK_WK[:] = [float(v) for v in WK21]
_GK_WG[:] = [float(WG10[j // 2]) if j % 2 else 0.0 for j in range(11)]


# --------------------------------------------------------------------------
# growable buffers
# --------------------------------------------------------------------------

cdef struct DArr:
    double* p
    Py_ssize_t n
    Py_ssize_t cap


cdef inline int _reserve(DArr* a, Py_ssize_t need) noexcept nogil:
    cdef Py_ssize_t cap
    cdef double* q
    if need <= a.cap:
        return 0
    cap = a.cap * 2 if a.cap else 64
    while cap < need:
        cap *= 2
    q = <double*> realloc(a.p, cap * sizeof(double))
    if q == NULL:
        return -1
    a.p = q
    a.cap = cap
    return 0


cdef inline int _push(DArr* a, double v) noexcept nogil:
    if _reserve(a, a.n + 1) != 0:
        return -1
    a.p[a.n] = v
    a.n += 1
    return 0


cdef inline void _swap(DArr* a, DArr* b) noexcept nogil:
    cdef DArr tmp = a[0]
    a[0] = b[0]
    b[0] = tmp


cdef inline void _release(DArr* a) noexcept nogil:
    free(a.p)
    a.p = NULL
    a.n = 0
    a.cap = 0


cdef class GrowthKernel:
    """Per-column growth atoms ``(C, b, r, c)``.

    ``|Phi_c| <= sum C zeta^b e^{r zeta + c}``.

    Parameters
    ----------
    growth : list of list of tuple
        ``basis_normalizer._column_growth`` output.
    """

    def __cinit__(self, growth):
        starts, coef, power, rate, lconst = [0], [], [], [], []
        for atoms in growth:
            for c_, b_, r_, k_ in atoms:
                coef.append(c_)
                power.append(b_)
                rate.append(r_)
                lconst.append(k_)
            starts.append(len(coef))
        arrays = [
            np.array(starts, dtype=np.intc),
            np.array(coef + [0.0], dtype=np.float64),
            np.array(power + [0.0], dtype=np.float64),
            np.array(rate + [0.0], dtype=np.float64),
            np.array(lconst + [0.0], dtype=np.float64),
        ]
        self._keep = arrays
        self.t.d = len(growth)
        self.t.start = <const int*> cnp.PyArray_DATA(arrays[0])
        self.t.coef = <const double*> cnp.PyArray_DATA(arrays[1])
        self.t.power = <const double*> cnp.PyArray_DATA(arrays[2])
        self.t.rate = <const double*> cnp.PyArray_DATA(arrays[3])
        self.t.lconst = <const double*> cnp.PyArray_DATA(arrays[4])


# --------------------------------------------------------------------------
# pieces
# --------------------------------------------------------------------------

cdef inline double _expit_ts(double x) noexcept nogil:
    """``exp(-logaddexp(0, -x))`` (``basis_normalizer._expit``)."""
    return exp(-logaddexp(0.0, -x))


cdef void _tail_bounds(
    const BasisTable* t,
    const GrowthTable* g,
    const TailResult* slope,
    double zeta_t,
    double q_t,
    double q_ref,
    double* mass,
    double* moments,
    bint* ok,
) noexcept nogil:
    """Bounds beyond ``zeta_t`` on the shifted mass and column moments."""
    cdef double h = t.h
    cdef double s = tail_bound(slope, zeta_t)
    cdef double shift, total, k
    cdef int col, a
    if not s > 0.0:
        mass[0] = INFINITY
        ok[0] = False
        if moments != NULL:
            for col in range(t.d):
                moments[col] = INFINITY
        return
    shift = -(q_t - q_ref)
    # A truncation point far below the reference has an infinite (valid) bound.
    mass[0] = exp(shift) * h / s
    ok[0] = True
    for col in range(g.d):
        total = 0.0
        for a in range(g.start[col], g.start[col + 1]):
            k = s - g.rate[a]
            if k <= 0.0 or zeta_t < 2.0 * g.power[a] / k:
                ok[0] = False
                break
            total += g.coef[a] * exp(g.power[a] * log(zeta_t) + g.rate[a] * zeta_t
                                     + g.lconst[a] + shift) * 2.0 * h / k
        if moments != NULL:
            moments[col] = total


cdef void _q_range(
    const BasisTable* t,
    const PotentialPrep* p,
    double lo,
    double hi,
    Iv* ivwork,
    double* qlo,
    double* qhi,
) noexcept nogil:
    """Natural enclosure of ``q`` intersected with the mean-value form."""
    cdef Iv nat[2]
    cdef Iv mid[1]
    cdef double m = 0.5 * (lo + hi)
    cdef double r = 0.5 * (hi - lo)
    cdef double slope, spread, mv_lo, mv_hi
    potential_enclosure_cell(t, p, lo, hi, 1, ivwork, nat)
    potential_enclosure_cell(t, p, m, m, 0, ivwork, mid)
    slope = npmax(fabs(nat[1].lo), fabs(nat[1].hi))
    spread = slope * r * (1.0 + 8.0 * _EPS)
    mv_lo = mid[0].lo - spread
    mv_hi = mid[0].hi + spread
    qlo[0] = npmax(nat[0].lo, mv_lo) if isfinite(mv_lo) else nat[0].lo
    qhi[0] = npmin(nat[0].hi, mv_hi) if isfinite(mv_hi) else nat[0].hi


cdef inline double _x_of(const BasisTable* t, int upper, double zeta) noexcept nogil:
    return t.c + (zeta if upper else -zeta) * t.h


cdef void _sort_unique(double* v, Py_ssize_t* n) noexcept nogil:
    cdef Py_ssize_t i, j, m
    cdef double a
    for i in range(1, n[0]):
        a = v[i]
        j = i - 1
        while j >= 0 and v[j] > a:
            v[j + 1] = v[j]
            j -= 1
        v[j + 1] = a
    m = 0
    for i in range(n[0]):
        if m == 0 or v[i] != v[m - 1]:
            v[m] = v[i]
            m += 1
    n[0] = m


# --------------------------------------------------------------------------
# Fitting-coordinate integration
# --------------------------------------------------------------------------

cdef int _integration_table(
    const BasisTable* physical, BasisTable* local, double** storage
) noexcept nogil:
    """Re-express the same basis in ``z=(x-c)/h`` for quadrature.

    The potential columns remain identical except that each ungated endpoint
    log gains ``log(h)``.  That constant is removed from the moments, while
    its contribution and the Jacobian are restored to the physical ``log Z``.
    Tail atoms and curvature constraints already use zeta=(x-c)/h, so neither
    needs reinterpretation.  Memory belongs to the caller.
    """
    cdef double h = physical.h, c = physical.c
    # The family is fixed in z (gates, soft powers and canonical rates), so only
    # the coordinate map and the endpoint-log distances change.  No storage is
    # needed; the caller's free(NULL) is a no-op.
    storage[0] = NULL
    local[0] = physical[0]
    local.c = 0.0
    local.h = 1.0
    local.orth_scale = physical.orth_scale / h
    local.orth_origin = (physical.orth_origin - c) / h
    local.lower = (physical.lower - c) / h if isfinite(physical.lower) else physical.lower
    local.upper = (physical.upper - c) / h if isfinite(physical.upper) else physical.upper
    return 0


cdef inline double _physical_log_shift(
    const BasisTable* t, const double* theta
) noexcept nogil:
    """``log Z_x - log Z_z`` including the log-basis constant and dx=h dz."""
    cdef double a = 1.0
    if t.log_col_lower >= 0:
        a += theta[t.log_col_lower]
    if t.log_col_upper >= 0:
        a += theta[t.log_col_upper]
    return a * log(t.h)


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------

cdef struct Ws:
    double* fwork
    double* colwork
    Iv* ivwork
    double* prep
    double* prep_rest
    double* tailwork
    double* theta_rest
    double* grid
    double* qgrid
    int* used
    int nused
    DArr lo
    DArr hi
    DArr nlo
    DArr nhi
    DArr cmag
    DArr flt
    # next generation of ``flt``/``cmag`` while resolving cells
    DArr nflt
    DArr ncmag
    DArr dropped_mom
    DArr bt
    DArr pool
    DArr done
    # tanh-sinh nodes ``(dist, wt)`` for ``k >= 0`` of levels ``< ts_levels``;
    # level ``l`` starts at pair ``ts_off[l]``
    DArr ts
    Py_ssize_t ts_off[31]
    int ts_levels
    # the last successful domain: reference, mass (e-units) and the error
    # budget outside its settled cells (dropped cells and tails; per column)
    double q_ref
    double z_s
    double budget_mass
    DArr budget_mom


cdef void _ws_free(Ws* w) noexcept nogil:
    free(w.fwork)
    free(w.colwork)
    free(w.ivwork)
    free(w.prep)
    free(w.prep_rest)
    free(w.tailwork)
    free(w.theta_rest)
    free(w.grid)
    free(w.qgrid)
    free(w.used)
    _release(&w.lo)
    _release(&w.hi)
    _release(&w.nlo)
    _release(&w.nhi)
    _release(&w.cmag)
    _release(&w.flt)
    _release(&w.nflt)
    _release(&w.ncmag)
    _release(&w.dropped_mom)
    _release(&w.bt)
    _release(&w.pool)
    _release(&w.done)
    _release(&w.ts)
    _release(&w.budget_mom)


cdef int _ws_alloc(Ws* w, const BasisTable* t, Py_ssize_t tailwork) noexcept nogil:
    cdef Py_ssize_t ps = prep_size(t)
    w.fwork = <double*> malloc((5 * t.nf + 1) * sizeof(double))
    w.colwork = <double*> malloc((5 * t.d + 1) * sizeof(double))
    w.ivwork = <Iv*> malloc((5 * t.nf + 1) * sizeof(Iv))
    w.prep = <double*> malloc(ps * sizeof(double))
    w.prep_rest = <double*> malloc(ps * sizeof(double))
    w.tailwork = <double*> malloc((tailwork + 1) * sizeof(double))
    w.theta_rest = <double*> malloc((t.d + 1) * sizeof(double))
    w.grid = <double*> malloc(_GRID * sizeof(double))
    w.qgrid = <double*> malloc(_GRID * sizeof(double))
    w.used = <int*> malloc((t.d + 1) * sizeof(int))
    if (w.used == NULL or w.fwork == NULL or w.colwork == NULL or w.ivwork == NULL
            or w.prep == NULL
            or w.prep_rest == NULL or w.tailwork == NULL or w.theta_rest == NULL
            or w.grid == NULL or w.qgrid == NULL):
        return -1
    return 0


cdef inline double pow2m(int level) noexcept nogil:
    cdef double s = 1.0
    cdef int _i
    for _i in range(level):
        s *= 0.5
    return s


cdef const double* _ts_level(Ws* w, int level) noexcept nogil:
    """Pairs ``(dist, wt)`` of the level's nodes ``t = k 2**-level``, ``k >= 0``.

    ``x = a + r dist`` (``b - r dist`` for ``t > 0``) and weight ``r wt`` on
    ``[a, b]``; both are even in ``t``.  Cached per workspace; ``NULL`` on
    allocation failure.
    """
    cdef int lv
    cdef Py_ssize_t kk, kmax, base
    cdef double step, tt, u
    while w.ts_levels <= level:
        lv = w.ts_levels
        step = pow2m(lv)
        kmax = <Py_ssize_t> ceil(_TS_TMAX / step)
        base = w.ts_off[lv]
        if _reserve(&w.ts, 2 * (base + kmax + 1)) != 0:
            return NULL
        for kk in range(kmax + 1):
            tt = kk * step
            u = 0.5 * M_PI * sinh(tt)
            w.ts.p[2 * (base + kk)] = 2.0 * _expit_ts(-2.0 * u)
            w.ts.p[2 * (base + kk) + 1] = (
                0.5 * M_PI * cosh(tt) * 4.0 * _expit_ts(2.0 * u) * _expit_ts(-2.0 * u)
            )
        w.ts_off[lv + 1] = base + kmax + 1
        w.ts_levels = lv + 1
    return &w.ts.p[2 * w.ts_off[level]]


# Fields of one settled cell in ``Ws.done``: z, dz, nz, step, id, lo, hi,
# then from ``_DONE_S1``: s1[d], a1[d], ds1[d], ns1[d].
cdef enum:
    _DONE_S1 = 7


cdef inline Py_ssize_t _done_width(int d) noexcept nogil:
    return _DONE_S1 + 4 * d


# --------------------------------------------------------------------------
# integration batch
# --------------------------------------------------------------------------

cdef int _node(
    const BasisTable* t,
    const double* theta,
    const int* used,
    int nused,
    double q_ref,
    double ident,
    double x,
    double wn,
    Ws* w,
    double* nb,
    double* gb,
    double gf,
) noexcept nogil:
    """Evaluate one quadrature node, pool it and add its terms to the sums.

    The node ``(ident, x, wn, e, phi[d])`` is appended to ``w.pool``.  Its
    terms at weight ``wn`` are added to ``nb`` (``z, nz, s1[d], s2[d], a1[d],
    ns1[d], ns2[d]``) and, at weight ``gf wn``, to the embedded rule's ``gb``
    (``z, s1[d], s2[d]``) unless ``gb`` is ``NULL``.  Returns ``-1`` on
    allocation failure.
    """
    cdef int d = t.d
    cdef int col, i
    cdef double q, dq, term, le, e, contrib, lw, ph, v, v2
    cdef double* phi
    cdef bint far
    if _reserve(&w.pool, w.pool.n + 4 + d) != 0:
        return -1
    phi = &w.pool.p[w.pool.n + 4]
    q = 0.0
    dq = 0.0
    le = -INFINITY
    e = 0.0
    if x > t.lower and x < t.upper:
        eval_point(t, x, 0, w.fwork, phi)
        for i in range(nused):
            term = phi[used[i]] * theta[used[i]]
            q = q + term
            dq = dq + fabs(term)
        dq = (d + 4) * _EPS * dq
        # Positional rounding: x itself is known to |x| eps, which an endpoint
        # log turns into |x| eps / (x - end) (a node 1e-9 below a finite end
        # at 1: 1e-7 relative); the floor no quadrature in binary64 x can beat.
        if t.log_col_lower >= 0 and theta[t.log_col_lower] > 0.0:
            dq = dq + theta[t.log_col_lower] * _EPS * fabs(x) / (x - t.lower)
        if t.log_col_upper >= 0 and theta[t.log_col_upper] > 0.0:
            dq = dq + theta[t.log_col_upper] * _EPS * fabs(x) / (t.upper - x)
        le = -(q - q_ref)
        e = exp(le)
        if not isfinite(e):
            e = 0.0
            le = -INFINITY
    if not (le > -INFINITY):
        for col in range(d):
            phi[col] = 0.0
    if not isfinite(dq):
        dq = 0.0
    contrib = wn * e
    # Where e^{-(q - q_ref)} underflows, a moment integrand e^{-q} Phi can
    # still be large (an exponential column under a flat potential), and
    # where a column is huge its square overflows before the weight scales
    # it: those products are formed in logs.
    far = le < _LOG_FAR
    lw = log(wn) + le if le > -INFINITY else -INFINITY
    w.pool.p[w.pool.n] = ident
    w.pool.p[w.pool.n + 1] = x
    w.pool.p[w.pool.n + 2] = wn
    w.pool.p[w.pool.n + 3] = le
    w.pool.n += 4 + d
    nb[0] += contrib
    nb[1] += contrib * dq
    if gb != NULL:
        gb[0] += gf * contrib
    for col in range(d):
        ph = phi[col]
        if (far or fabs(ph) > _BIG_COLUMN) and ph != 0.0:
            v = copysign(exp(lw + log(fabs(ph))), ph)
            v2 = exp(lw + 2.0 * log(fabs(ph)))
        else:
            v = contrib * ph
            v2 = v * ph
        nb[2 + col] += v
        nb[2 + d + col] += v2
        nb[2 + 2 * d + col] += fabs(v)
        nb[2 + 3 * d + col] += dq * fabs(v)
        nb[2 + 4 * d + col] += dq * v2
        if gb != NULL:
            gb[1 + col] += gf * v
            gb[1 + d + col] += gf * v2
    return 0


# Per-cell accumulator in ``Ws.bt`` (``_integrate_batch``): totals ``z, nz``,
# the rule (``_RULE_*``, ``0`` once settled), the cell id, totals ``s1, s2,
# a1, ns1, ns2`` (``d`` each) from ``_ACC_S1``; then the previous estimate
# (``z, s1[d], s2[d]``) at ``_acc_old(d)`` and the current level's new
# sums (``_node`` layout) at ``_acc_new(d)``.
cdef enum:
    _ACC_S1 = 4
    _RULE_TS = 1
    _RULE_GK = 2


cdef inline Py_ssize_t _acc_old(int d) noexcept nogil:
    return _ACC_S1 + 5 * d


cdef inline Py_ssize_t _acc_new(int d) noexcept nogil:
    return _acc_old(d) + 1 + 2 * d


cdef inline Py_ssize_t _acc_width(int d) noexcept nogil:
    return _acc_new(d) + 2 + 5 * d


cdef int _settle(
    double* acc,
    int d,
    double lo,
    double hi,
    double step,
    double tol,
    double share_z,
    const double* column_scale,
    Ws* w,
) noexcept nogil:
    """Settle the cell of ``acc`` if its estimates agree, recording it.

    The cell answers for its own share (its magnitude, plus ``share_z``, an
    equal slice of the total) and for its rounding-noise floor.  A settled
    cell is appended to ``w.done`` with the node-weight multiplier ``step``.
    Returns ``1`` when settled, ``0`` when not, ``-1`` on allocation failure.
    """
    cdef Py_ssize_t old = _acc_old(d)
    cdef int col
    cdef double dzv, cs
    cdef double* ids
    dzv = fabs(acc[0] - acc[old])
    if not (dzv <= tol * (acc[0] + share_z) + 2.0 * acc[1]):
        return 0
    for col in range(d):
        cs = column_scale[col]
        if not (fabs(acc[_ACC_S1 + col] - acc[old + 1 + col])
                <= tol * (acc[_ACC_S1 + 2 * d + col] + acc[0] * cs + share_z * cs)
                + 2.0 * acc[_ACC_S1 + 3 * d + col]):
            return 0
        if not (fabs(acc[_ACC_S1 + d + col] - acc[old + 1 + d + col])
                <= tol * (acc[_ACC_S1 + d + col] + acc[0] * (cs * cs)
                          + share_z * (cs * cs))
                + 2.0 * acc[_ACC_S1 + 4 * d + col]):
            return 0
    acc[2] = 0.0
    if _reserve(&w.done, w.done.n + _done_width(d)) != 0:
        return -1
    ids = &w.done.p[w.done.n]
    ids[0] = acc[0]
    ids[1] = dzv
    ids[2] = 2.0 * acc[1]
    ids[3] = step
    ids[4] = acc[3]
    ids[5] = lo
    ids[6] = hi
    ids = &ids[_DONE_S1]
    for col in range(d):
        ids[col] = acc[_ACC_S1 + col]
        ids[d + col] = acc[_ACC_S1 + 2 * d + col]
        ids[2 * d + col] = fabs(acc[_ACC_S1 + col] - acc[old + 1 + col])
        ids[3 * d + col] = 2.0 * acc[_ACC_S1 + 3 * d + col]
    w.done.n += _done_width(d)
    return 1


cdef inline bint _end_singular(
    const BasisTable* t, const double* theta, double lo, double hi
) noexcept nogil:
    """Whether the cell ends at a finite end with an endpoint log column.

    There ``e^{-q}`` behaves as ``dist**gamma`` and the log column's moments
    as ``log dist`` (even at ``gamma = 0``): not analytic at the cell's end,
    which tanh-sinh absorbs (its nodes carry the endpoint distance exactly)
    and Gauss--Kronrod does not.
    """
    return ((lo == t.lower and t.log_col_lower >= 0)
            or (hi == t.upper and t.log_col_upper >= 0))


cdef int _integrate_batch(
    const BasisTable* t,
    const double* theta,
    const int* used,
    int nused,
    double q_ref,
    const double* column_scale,
    const NormOptions* o,
    Ws* w,
    Py_ssize_t ndone_before,
    double base_z,
    Py_ssize_t* next_id,
    DArr* failed_lo,
    DArr* failed_hi,
) noexcept nogil:
    """Integrate the batch ``(w.lo, w.hi)``, settling each cell or failing it.

    A cell whose integrand is analytic on it (all but those ``_end_singular``)
    takes the Gauss--Kronrod 10/21 rule: the Kronrod value, settled when it
    agrees with the embedded Gauss value (that difference is its error).  A
    cell ending at an endpoint singularity takes nested tanh-sinh from level
    ``_TS_LEVEL0``, settled when consecutive levels agree, up to
    ``o.max_level``.  Settled cells are appended to ``w.done`` and their
    nodes to ``w.pool`` (``id, x, wnode, e, phi[d]``; a cell's final weights
    are ``wnode`` times its recorded step, 1 for Gauss--Kronrod); unsettled
    cells go to ``failed_*`` to be split.  Returns ``-1`` on allocation
    failure.
    """
    cdef int d = t.d
    cdef Py_ssize_t n = w.lo.n
    cdef Py_ssize_t ci, i, kk, kmax, k_cells, j
    cdef int level = _TS_LEVEL0
    cdef int out
    cdef double step, dist, wt, a, b, r, x, tot_z, tol, share_z
    cdef bint any_active
    cdef Py_ssize_t stride = _acc_width(d)
    cdef Py_ssize_t old = _acc_old(d)
    cdef Py_ssize_t nk = _acc_new(d)
    cdef double* acc
    cdef double* gb
    cdef const double* nodes
    if _reserve(&w.bt, n * stride) != 0:
        return -1
    for ci in range(n):
        acc = &w.bt.p[ci * stride]
        for i in range(stride):
            acc[i] = 0.0
        acc[2] = (
            _RULE_TS if _end_singular(t, theta, w.lo.p[ci], w.hi.p[ci]) else _RULE_GK
        )
        acc[3] = <double> next_id[0]
        next_id[0] += 1
    # Gauss--Kronrod cells: one pass; the Gauss value is the previous estimate.
    for ci in range(n):
        acc = &w.bt.p[ci * stride]
        if acc[2] != _RULE_GK:
            continue
        a = w.lo.p[ci]
        b = w.hi.p[ci]
        r = 0.5 * (b - a)
        for j in range(11):
            gb = &acc[old] if _GK_WG[j] != 0.0 else NULL
            for kk in range(2 if j < 10 else 1):
                if j == 10:
                    x = a + r
                elif kk == 0:
                    x = a + r * (1.0 - _GK_X[j])
                else:
                    x = b - r * (1.0 - _GK_X[j])
                if _node(t, theta, used, nused, q_ref, acc[3], x, r * _GK_WK[j], w,
                         &acc[nk], gb, _GK_WG[j] / _GK_WK[j]) != 0:
                    return -1
        acc[0] = acc[nk]
        acc[1] = acc[nk + 1]
        for i in range(5 * d):
            acc[_ACC_S1 + i] = acc[nk + 2 + i]
    while True:
        step = pow2m(level)
        kmax = <Py_ssize_t> ceil(_TS_TMAX / step)
        nodes = _ts_level(w, level)
        if nodes == NULL:
            return -1
        for ci in range(n):
            acc = &w.bt.p[ci * stride]
            if acc[2] != _RULE_TS:
                continue
            for i in range(2 + 5 * d):
                acc[nk + i] = 0.0
            a = w.lo.p[ci]
            b = w.hi.p[ci]
            r = 0.5 * (b - a)
            for kk in range(-kmax, kmax + 1):
                if level > _TS_LEVEL0 and kk % 2 == 0:
                    continue
                dist = nodes[2 * (kk if kk >= 0 else -kk)]
                wt = nodes[2 * (kk if kk >= 0 else -kk) + 1]
                if kk < 0:
                    x = a + r * dist
                elif kk > 0:
                    x = b - r * dist
                else:
                    x = 0.5 * (a + b)
                if _node(t, theta, used, nused, q_ref, acc[3], x, r * wt, w,
                         &acc[nk], NULL, 0.0) != 0:
                    return -1
            for i in range(2 + 5 * d):
                acc[nk + i] *= step
            # old <- totals; totals <- 0.5 totals + new (level0: totals <- new)
            if level == _TS_LEVEL0:
                acc[0] = acc[nk]
                acc[1] = acc[nk + 1]
                for i in range(5 * d):
                    acc[_ACC_S1 + i] = acc[nk + 2 + i]
            else:
                acc[old] = acc[0]
                for i in range(2 * d):
                    acc[old + 1 + i] = acc[_ACC_S1 + i]
                acc[0] = 0.5 * acc[0] + acc[nk]
                acc[1] = 0.5 * acc[1] + acc[nk + 1]
                for i in range(5 * d):
                    acc[_ACC_S1 + i] = 0.5 * acc[_ACC_S1 + i] + acc[nk + 2 + i]
        tot_z = base_z
        for ci in range(n):
            tot_z = tot_z + w.bt.p[ci * stride]
        k_cells = n + ndone_before
        if k_cells < 1:
            k_cells = 1
        tol = 0.25 * o.rel_tol
        share_z = tot_z / k_cells
        any_active = False
        for ci in range(n):
            acc = &w.bt.p[ci * stride]
            if acc[2] == _RULE_GK:
                # decided at the first check: settled or split
                out = _settle(acc, d, w.lo.p[ci], w.hi.p[ci], 1.0, tol, share_z,
                              column_scale, w)
                if out < 0:
                    return -1
                if out == 0:
                    acc[2] = 0.0
                    if (_push(failed_lo, w.lo.p[ci]) != 0
                            or _push(failed_hi, w.hi.p[ci]) != 0):
                        return -1
            elif acc[2] == _RULE_TS and level > _TS_LEVEL0:
                out = _settle(acc, d, w.lo.p[ci], w.hi.p[ci], step, tol, share_z,
                              column_scale, w)
                if out < 0:
                    return -1
            if acc[2] != 0.0:
                any_active = True
        if not any_active:
            break
        if level >= o.max_level:
            for ci in range(n):
                if w.bt.p[ci * stride + 2] != 0.0:
                    if (_push(failed_lo, w.lo.p[ci]) != 0
                            or _push(failed_hi, w.hi.p[ci]) != 0):
                        return -1
            break
        level += 1
    return 0


# --------------------------------------------------------------------------
# domain stages
# --------------------------------------------------------------------------

cdef bint _reference(
    const BasisTable* t,
    const double* theta,
    double lo_x,
    double hi_x,
    int npts,
    bint any_log,
    Ws* w,
    double* q_ref,
    double* x_star,
    double* z_est,
) noexcept nogil:
    """Smallest potential on an asinh grid of ``[lo_x, hi_x]``; False if none.

    ``z_est`` is the trapezoid estimate of ``int exp(-(q - q_ref))`` on the
    grid: it only scales the truncation's moment criterion (a poor estimate
    moves the truncation, never the error budget).
    """
    cdef double v0 = asinh((lo_x - t.c) / t.h)
    cdef double v1 = asinh((hi_x - t.c) / t.h)
    cdef double stepv, x, prev_x = NAN, prev_e = 0.0, e
    cdef double qv[1]
    cdef int i, m = 0
    cdef bint found = False
    if npts < 2:
        npts = 2
    if npts > _GRID:
        npts = _GRID
    stepv = (v1 - v0) / (npts - 1)
    # A domain narrow against its distance from the center (a censored row
    # at a finite end, 1e-93 wide) collapses in the asinh coordinate: the
    # grid is then linear in x.
    cdef bint linear = not (v1 - v0 > 1e-9 * npmax(1.0, fabs(v0)))
    for i in range(npts):
        if linear:
            x = lo_x + (hi_x - lo_x) * i / (npts - 1) if i < npts - 1 else hi_x
        else:
            x = t.c + t.h * sinh(i * stepv + v0 if i < npts - 1 else v1)
        if any_log and not (x > t.lower and x < t.upper):
            continue
        w.grid[m] = npmin(npmax(x, t.lower), t.upper)
        m += 1
    q_ref[0] = INFINITY
    x_star[0] = NAN
    for i in range(m):
        potential_point(t, theta, w.grid[i], 0, w.fwork, w.colwork, qv)
        w.qgrid[i] = qv[0]
        if isfinite(qv[0]) and (not found or qv[0] < q_ref[0]):
            q_ref[0] = qv[0]
            x_star[0] = w.grid[i]
            found = True
    z_est[0] = 0.0
    if not found:
        return False
    for i in range(m):
        e = exp(-(w.qgrid[i] - q_ref[0])) if isfinite(w.qgrid[i]) else 0.0
        if i > 0:
            z_est[0] += 0.5 * (e + prev_e) * (w.grid[i] - prev_x)
        prev_x = w.grid[i]
        prev_e = e
    return True


cdef int _truncate(
    const BasisTable* t,
    const double* theta,
    int side,
    double zeta_t,
    const TailResult* growth,
    const GrowthTable* g,
    double q_ref,
    double z_est,
    const double* column_scale,
    const NormOptions* o,
    Ws* w,
    double* zeta_out,
    double* q_out,
    NormResult* r,
) noexcept nogil:
    """Move a tail truncation point outward until the tail bounds are negligible.

    Negligible: the potential has risen 80 above the reference, and the
    moment tails are below a quarter of the tolerance of ``z_est`` times the
    column scales (the moments of exponential columns under a flat potential
    live far beyond the mass: ``exp(-b z**2 + l z)`` peaks at ``z = l / 2b``).
    Returns ``0``, or ``1`` with ``r.status`` set.
    """
    cdef double q_t = NAN, mass, deficit, stp, a, b
    cdef double qv[1]
    cdef double* tmom = &w.colwork[3 * t.d]
    cdef double cap = 0.25 * o.rel_tol * (z_est if z_est > 0.0 else 0.0)
    cdef bint okb = False
    cdef int _it, col
    for _it in range(200):
        potential_point(t, theta, _x_of(t, side, zeta_t), 0, w.fwork, w.colwork, qv)
        q_t = qv[0]
        if not isfinite(q_t):
            # Columns overflow (inf - inf) before the potential has grown
            # enough to truncate: not representable in binary64.
            r.status = NORM_TAIL_NONFINITE
            r.side = side
            r.diag[0] = zeta_t
            return 1
        _tail_bounds(t, g, growth, zeta_t, q_t, q_ref, &mass, tmom, &okb)
        deficit = 80.0 - (q_t - q_ref)
        if okb and deficit <= 0.0:
            for col in range(t.d):
                if not (tmom[col] <= cap * column_scale[col]):
                    okb = False
                    break
            if okb:
                break
        okb = False
        # The slope bound can be very weak where a tiny leader has not yet
        # taken over: advance by at most a doubling (at least x1.25).
        stp = (1.0 if 1.0 > deficit else deficit) / tail_bound(growth, zeta_t)
        a = zeta_t + stp
        b = zeta_t * 1.25
        a = b if b > a else a
        b = 2.0 * zeta_t
        zeta_t = b if b < a else a
        if not zeta_t <= o.zeta_cap:
            r.status = NORM_TAIL_CAP
            r.side = side
            r.diag[0] = zeta_t
            return 1
    if not okb:
        r.status = NORM_TAIL_NOCONV
        r.side = side
        return 1
    zeta_out[0] = zeta_t
    q_out[0] = q_t
    return 0


cdef inline void _add_outer(
    double* acc, const double* dev, double lstp, int d
) noexcept nogil:
    """``acc += exp(lstp) dev dev^T`` on the upper triangle.

    Below ``_LOG_FAR`` the products are formed in logs: the weight can be
    subnormal where ``dev`` is huge.
    """
    cdef double stp = exp(lstp)
    cdef double v0
    cdef int col, c2
    if lstp >= _LOG_FAR:
        for col in range(d):
            for c2 in range(col, d):
                acc[col * d + c2] += dev[col] * stp * dev[c2]
        return
    for col in range(d):
        for c2 in range(col, d):
            v0 = dev[col] * dev[c2]
            if v0 != 0.0:
                acc[col * d + c2] += copysign(
                    exp(lstp + log(fabs(dev[col])) + log(fabs(dev[c2]))), v0
                )


cdef int _domain_body(
    const BasisTable* t,
    const double* theta,
    const PotentialPrep* prep,
    const TailResult* growth,
    const GrowthTable** gt,
    const bint* tail_on,
    double lo_end,
    double hi_end,
    double part_v0,
    double part_step,
    int ref_npts,
    const double* column_scale,
    const NormOptions* o,
    bint keep_nodes,
    double* mean,
    double* cov,
    NormResult* r,
    Ws* w,
) noexcept nogil:
    """Mass, mean and centered covariance of ``exp(-q)`` over one domain.

    The domain is ``[lo_end, hi_end]`` inside the support; an end with
    ``tail_on`` reaches an infinite support end and is truncated by the tail
    bounds.  ``r.log_z`` receives the log of the domain's (unnormalized) mass,
    ``mean``/``cov`` its normalized moments, all to ``o.rel_tol`` relative to
    the domain itself.  The initial cells are the asinh partition
    ``part_v0 + k part_step`` cut to the domain (``part_step <= 0``: a new
    partition of ``o.initial_cells`` cells over the truncated domain, then
    reported in ``r.part_v0``/``r.part_step``; ``part_step = inf``: the
    domain is one initial cell).  On success the reference, mass and the
    error budget outside the settled cells are left in ``w`` with the
    settled cells (``_reuse_build``).  Returns ``-1`` on allocation failure,
    else ``0`` with ``r.status`` set.
    """
    cdef int d = t.d
    cdef int side, col, c2
    cdef double lo_x, hi_x, v0, v1, stepv, q_ref, x_star, z_est, zeta0, mass
    cdef double stp
    cdef double tail_zeta[2]
    cdef double tail_q[2]
    cdef double zeta_start[2]
    cdef bint okb, any_log, found
    cdef bint prep_on[2]
    cdef PotentialPrep prep_rest
    cdef int log_col[2]
    cdef bint log_on[2]
    cdef Py_ssize_t i, n, k, nsplit, cells_used, ndone_before, next_id, m, kmin, kmax
    cdef double q_lo, q_hi, width, log_z_lo, rel_mag, weight, ratio, scale_c
    cdef double dropped_mass = 0.0
    # Mass bound of cells at binary64 resolution (no representable midpoint):
    # reported with the conditioning floor, not counted against the target.
    cdef double resolution_mass = 0.0
    cdef double new_ref, factor, base_z, z_s, quad_err, cond_err, tail_mass, lstp
    cdef double extra[4]
    cdef double* rec
    cdef double* tmom
    cdef DArr flo, fhi
    flo.p = NULL
    flo.n = 0
    flo.cap = 0
    fhi.p = NULL
    fhi.n = 0
    fhi.cap = 0
    r.status = NORM_PROPER
    r.log_z = NAN
    r.rel_error = NAN
    r.cond_error = NAN
    r.cells = 0
    r.stage = 1
    r.side = -1
    r.n_nodes = 0
    r.nodes = NULL
    r.weights = NULL
    for i in range(6):
        r.diag[i] = NAN

    log_col[0] = t.log_col_lower
    log_col[1] = t.log_col_upper
    any_log = False
    for side in range(2):
        log_on[side] = log_col[side] >= 0 and theta[log_col[side]] > 0.0
        if log_on[side]:
            any_log = True

    # A tail end starts at the growth cutoff, or one h beyond a finite end
    # already past it (a far censored row).
    lo_x = lo_end
    hi_x = hi_end
    if tail_on[0]:
        zeta0 = growth[0].zeta
        if isfinite(hi_end) and (t.c - hi_end) / t.h + 1.0 > zeta0:
            zeta0 = (t.c - hi_end) / t.h + 1.0
        zeta_start[0] = zeta0
        lo_x = _x_of(t, 0, zeta0)
    if tail_on[1]:
        zeta0 = growth[1].zeta
        if isfinite(lo_end) and (lo_end - t.c) / t.h + 1.0 > zeta0:
            zeta0 = (lo_end - t.c) / t.h + 1.0
        zeta_start[1] = zeta0
        hi_x = _x_of(t, 1, zeta0)
    if part_step > 0.0:
        # About eight reference points per partition cell of the domain.
        v0 = (asinh((hi_x - t.c) / t.h) - asinh((lo_x - t.c) / t.h)) / part_step
        ref_npts = 8 * (<int> ceil(npmin(v0, <double> _GRID))) + 1
        if ref_npts < 9:
            ref_npts = 9

    # --- reference potential and tail truncation -----------------------------
    if not _reference(t, theta, lo_x, hi_x, ref_npts, any_log, w, &q_ref, &x_star,
                      &z_est):
        r.status = NORM_NO_REFERENCE
        return 0
    for side in range(2):
        if not tail_on[side]:
            continue
        if _truncate(t, theta, side, zeta_start[side], &growth[side], gt[side],
                     q_ref, z_est, column_scale, o, w, &tail_zeta[side],
                     &tail_q[side], r) != 0:
            return 0
        if side == 0:
            lo_x = _x_of(t, 0, tail_zeta[side])
        else:
            hi_x = _x_of(t, 1, tail_zeta[side])

    # --- cells: resolve by enclosures --------------------------------------
    r.stage = 2
    v0 = asinh((lo_x - t.c) / t.h)
    v1 = asinh((hi_x - t.c) / t.h)
    if part_step <= 0.0:
        stepv = (v1 - v0) / o.initial_cells
        r.part_v0 = v0
        r.part_step = stepv
        kmin = 1
        kmax = o.initial_cells - 1
        part_v0 = v0
        part_step = stepv
    elif not isfinite(part_step):
        r.part_v0 = part_v0
        r.part_step = part_step
        kmin = 1
        kmax = 0
    else:
        r.part_v0 = part_v0
        r.part_step = part_step
        kmin = <Py_ssize_t> ceil((v0 - part_v0) / part_step)
        kmax = <Py_ssize_t> floor((v1 - part_v0) / part_step)
    if kmax < kmin:
        kmax = kmin - 1
    if _reserve(&w.flt, (kmax - kmin + 1) + 8) != 0:
        return -1
    m = 0
    w.flt.p[m] = lo_x
    m += 1
    for i in range(kmin, kmax + 1):
        w.flt.p[m] = npmin(npmax(t.c + t.h * sinh(i * part_step + part_v0), lo_x), hi_x)
        m += 1
    w.flt.p[m] = hi_x
    m += 1
    extra[0] = t.c
    extra[1] = x_star
    # The fixed gate centers z = -1 and z = +1.
    extra[2] = t.c - t.h if t.has_l else NAN
    extra[3] = t.c + t.h if t.has_u else NAN
    # (a one-cell domain is split by the variation rule alone)
    for i in range(4 if isfinite(part_step) else 0):
        if isfinite(extra[i]) and lo_x < extra[i] and extra[i] < hi_x:
            w.flt.p[m] = extra[i]
            m += 1
    _sort_unique(w.flt.p, &m)
    w.flt.p[0] = lo_x
    w.flt.p[m - 1] = hi_x
    w.lo.n = 0
    w.hi.n = 0
    if _reserve(&w.lo, m) != 0 or _reserve(&w.hi, m) != 0:
        return -1
    for i in range(m - 1):
        w.lo.p[i] = w.flt.p[i]
        w.hi.p[i] = w.flt.p[i + 1]
    w.lo.n = m - 1
    w.hi.n = m - 1
    if _reserve(&w.dropped_mom, d) != 0:
        return -1
    for col in range(d):
        w.dropped_mom.p[col] = 0.0
    cells_used = w.lo.n
    # Per cell, in w.flt: q_lo, q_hi, var, log_upper, log_lower, state; and
    # cmag[d] in w.cmag.  The enclosures (q_lo, q_hi, var, cmag) depend on the
    # cell alone, so they are computed once per cell (state -1: fresh) and
    # carried while the cell lives; the log bounds and the drop decision
    # (state 1) follow the current reference and are redone every pass.
    if _reserve(&w.flt, 6 * w.lo.n) != 0 or _reserve(&w.cmag, w.lo.n * d) != 0:
        return -1
    for i in range(w.lo.n):
        w.flt.p[6 * i + 5] = -1.0
    while True:
        n = w.lo.n
        for side in range(2):
            prep_on[side] = False
        for i in range(n):
            if w.flt.p[6 * i + 5] != -1.0:
                continue
            _q_range(t, prep, w.lo.p[i], w.hi.p[i], w.ivwork, &q_lo, &q_hi)
            w.flt.p[6 * i] = q_lo
            w.flt.p[6 * i + 1] = q_hi
            w.flt.p[6 * i + 2] = q_hi - q_lo
            # At an enabled endpoint log the variation is that of the rest
            # of the potential (the log's own is absorbed by the end cell's
            # tanh-sinh).
            for side in range(2):
                if not (log_on[side] and ((side == 0 and w.lo.p[i] == t.lower)
                                          or (side == 1 and w.hi.p[i] == t.upper))):
                    continue
                if not prep_on[side]:
                    for col in range(d):
                        w.theta_rest[col] = theta[col]
                    w.theta_rest[log_col[side]] = 0.0
                    prepare_potential(t, w.theta_rest, w.prep_rest, &prep_rest)
                    prep_on[side] = True
                    prep_on[1 - side] = False
                _q_range(t, &prep_rest, w.lo.p[i], w.hi.p[i], w.ivwork, &q_lo, &q_hi)
                w.flt.p[6 * i + 2] = q_hi - q_lo
            basis_enclosure_cell(t, w.lo.p[i], w.hi.p[i], 0, NULL, w.ivwork,
                                 <Iv*> w.colwork)
            for col in range(d):
                v0 = npmax(
                    fabs((<Iv*> w.colwork)[col].lo), fabs((<Iv*> w.colwork)[col].hi)
                )
                w.cmag.p[i * d + col] = v0 if isfinite(v0) else INFINITY
            w.flt.p[6 * i + 5] = 0.0
        # Keep the reference at or below every cell's upper bound, so a deep
        # narrow well missed by the coarse grid cannot overflow e^{-(q-q_ref)}.
        found = False
        new_ref = INFINITY
        for i in range(n):
            if isfinite(w.flt.p[6 * i + 1]) and w.flt.p[6 * i + 1] < new_ref:
                new_ref = w.flt.p[6 * i + 1]
                found = True
        if found and new_ref < q_ref:
            factor = exp(-(q_ref - new_ref))
            dropped_mass *= factor
            for col in range(d):
                w.dropped_mom.p[col] *= factor
            q_ref = new_ref
        log_z_lo = -INFINITY
        for i in range(n):
            width = w.hi.p[i] - w.lo.p[i]
            w.flt.p[6 * i + 3] = log(width) - (w.flt.p[6 * i] - q_ref)
            if isfinite(w.flt.p[6 * i + 1]):
                w.flt.p[6 * i + 4] = log(width) - (w.flt.p[6 * i + 1] - q_ref)
            else:
                w.flt.p[6 * i + 4] = -INFINITY
            if i == 0:
                log_z_lo = w.flt.p[6 * i + 4]
            else:
                log_z_lo = logaddexp(log_z_lo, w.flt.p[6 * i + 4])
        nsplit = 0
        w.nlo.n = 0
        w.nhi.n = 0
        for col in range(d):
            scale_c = 0.0
            for i in range(n):
                if w.flt.p[6 * i + 4] >= log_z_lo - 5.0:
                    scale_c = npmax(scale_c, w.cmag.p[i * d + col])
            w.colwork[2 * d + col] = npmax(scale_c, column_scale[col])
        for i in range(n):
            rel_mag = -INFINITY
            for col in range(d):
                ratio = w.cmag.p[i * d + col] / w.colwork[2 * d + col]
                if ratio != ratio:
                    ratio = INFINITY
                rel_mag = ratio if col == 0 or ratio > rel_mag else rel_mag
            if rel_mag == INFINITY:
                weight = INFINITY
            else:
                weight = 2.0 * log1p(rel_mag) + log1p(
                    -rel_mag / ((1.0 + rel_mag) * (1.0 + rel_mag))
                )
            if w.flt.p[6 * i + 3] + weight < log_z_lo + log(
                o.drop_tol / (n if n > 1 else 1)
            ):
                # dropped
                dropped_mass += exp(w.flt.p[6 * i + 3])
                for col in range(d):
                    w.dropped_mom.p[col] += (
                        exp(w.flt.p[6 * i + 3]) * w.cmag.p[i * d + col]
                    )
                w.flt.p[6 * i + 5] = 1.0
            elif w.flt.p[6 * i + 2] > o.variation:
                v1 = 0.5 * (w.lo.p[i] + w.hi.p[i])
                if not (w.lo.p[i] < v1 and v1 < w.hi.p[i]):
                    resolution_mass += exp(w.flt.p[6 * i + 3])
                    w.flt.p[6 * i + 5] = 1.0
                else:
                    nsplit += 1
        if nsplit == 0:
            k = 0
            for i in range(n):
                if w.flt.p[6 * i + 5] == 0.0:
                    w.lo.p[k] = w.lo.p[i]
                    w.hi.p[k] = w.hi.p[i]
                    k += 1
            w.lo.n = k
            w.hi.n = k
            break
        cells_used += nsplit
        if cells_used > o.max_cells:
            r.status = NORM_CELLS_RESOLVE
            r.diag[0] = cells_used
            r.diag[1] = nsplit
            return 0
        # Children are fresh; a kept cell carries its enclosures.  Cells stay
        # in ascending order (a split emits its halves in place).
        if (_reserve(&w.nlo, n + nsplit) != 0 or _reserve(&w.nhi, n + nsplit) != 0
                or _reserve(&w.nflt, 6 * (n + nsplit)) != 0
                or _reserve(&w.ncmag, (n + nsplit) * d) != 0):
            return -1
        k = 0
        for i in range(n):
            if w.flt.p[6 * i + 5] != 0.0:
                continue
            if w.flt.p[6 * i + 2] > o.variation:
                v1 = 0.5 * (w.lo.p[i] + w.hi.p[i])
                w.nlo.p[k] = w.lo.p[i]
                w.nhi.p[k] = v1
                w.nflt.p[6 * k + 5] = -1.0
                k += 1
                w.nlo.p[k] = v1
                w.nhi.p[k] = w.hi.p[i]
                w.nflt.p[6 * k + 5] = -1.0
                k += 1
            else:
                w.nlo.p[k] = w.lo.p[i]
                w.nhi.p[k] = w.hi.p[i]
                memcpy(&w.nflt.p[6 * k], &w.flt.p[6 * i], 6 * sizeof(double))
                memcpy(&w.ncmag.p[k * d], &w.cmag.p[i * d], d * sizeof(double))
                k += 1
        w.nlo.n = k
        w.nhi.n = k
        _swap(&w.lo, &w.nlo)
        _swap(&w.hi, &w.nhi)
        _swap(&w.flt, &w.nflt)
        _swap(&w.cmag, &w.ncmag)

    # --- integration -----------------------------------------------------------
    r.stage = 3
    w.done.n = 0
    w.pool.n = 0
    next_id = 0
    while w.lo.n > 0:
        ndone_before = w.done.n // _done_width(d)
        base_z = 0.0
        for i in range(ndone_before):
            base_z = base_z + w.done.p[i * _done_width(d)]
        flo.n = 0
        fhi.n = 0
        if _integrate_batch(t, theta, w.used, w.nused, q_ref, column_scale, o, w,
                            ndone_before, base_z, &next_id, &flo, &fhi) != 0:
            _release(&flo)
            _release(&fhi)
            return -1
        if flo.n == 0:
            break
        cells_used += flo.n
        if cells_used > o.max_cells:
            r.status = NORM_CELLS_INTEGRATE
            r.diag[0] = cells_used
            r.diag[1] = flo.n
            _release(&flo)
            _release(&fhi)
            return 0
        n = flo.n
        if _reserve(&w.lo, 2 * n) != 0 or _reserve(&w.hi, 2 * n) != 0:
            _release(&flo)
            _release(&fhi)
            return -1
        m = 0
        for i in range(n):
            v0 = 0.5 * (flo.p[i] + fhi.p[i])
            if not (flo.p[i] < v0 and v0 < fhi.p[i]):
                # binary64 resolution (a singular endpoint 1e-16 from a
                # finite end at 1): bound the cell instead of splitting it.
                _q_range(t, prep, flo.p[i], fhi.p[i], w.ivwork, &q_lo, &q_hi)
                resolution_mass += (fhi.p[i] - flo.p[i]) * exp(-(q_lo - q_ref))
                continue
            w.lo.p[m] = flo.p[i]
            w.hi.p[m] = v0
            m += 1
            w.lo.p[m] = v0
            w.hi.p[m] = fhi.p[i]
            m += 1
        w.lo.n = m
        w.hi.n = m
    _release(&flo)
    _release(&fhi)

    # --- error budget ------------------------------------------------------------
    r.stage = 4
    k = w.done.n // _done_width(d)
    z_s = 0.0
    quad_err = 0.0
    cond_err = 0.0
    for col in range(d):
        mean[col] = 0.0
        w.colwork[col] = 0.0          # a1
        w.colwork[d + col] = 0.0      # mom_err
    for i in range(k):
        rec = &w.done.p[i * _done_width(d)]
        z_s = z_s + rec[0]
        quad_err = quad_err + npmax(rec[1] - rec[2], 0.0)
        cond_err = cond_err + rec[2]
        for col in range(d):
            mean[col] = mean[col] + rec[_DONE_S1 + col]
            w.colwork[col] = w.colwork[col] + rec[_DONE_S1 + d + col]
            w.colwork[d + col] = w.colwork[d + col] + npmax(
                rec[_DONE_S1 + 2 * d + col] - rec[_DONE_S1 + 3 * d + col], 0.0
            )
    tail_mass = 0.0
    if _reserve(&w.flt, 2 * d) != 0:
        return -1
    tmom = w.flt.p
    for col in range(d):
        tmom[d + col] = 0.0
    for side in range(2):
        if not tail_on[side]:
            continue
        _tail_bounds(t, gt[side], &growth[side], tail_zeta[side], tail_q[side], q_ref,
                     &mass, tmom, &okb)
        tail_mass += mass
        for col in range(d):
            tmom[d + col] += tmom[col]
    r.rel_error = (quad_err + dropped_mass + tail_mass) / z_s
    okb = r.rel_error <= o.rel_tol
    v1 = 0.0
    for col in range(d):
        v0 = (w.colwork[d + col] + w.dropped_mom.p[col] + tmom[d + col]) / (
            w.colwork[col] + z_s * column_scale[col])
        if not (v0 <= o.rel_tol):
            okb = False
        if col == 0 or not (v0 <= v1):
            v1 = v0
    if not okb:
        r.status = NORM_ACCURACY
        r.diag[0] = r.rel_error
        r.diag[1] = v1
        r.diag[2] = quad_err / z_s
        r.diag[3] = dropped_mass / z_s
        r.diag[4] = tail_mass / z_s
        return 0
    if _reserve(&w.budget_mom, d) != 0:
        return -1
    w.q_ref = q_ref
    w.z_s = z_s
    w.budget_mass = dropped_mass + tail_mass
    for col in range(d):
        w.budget_mom.p[col] = w.dropped_mom.p[col] + tmom[d + col]

    # --- moments (centered second pass) -----------------------------------------
    for col in range(d):
        mean[col] = mean[col] / z_s
    for i in range(d * d):
        cov[i] = 0.0
    # step per settled cell id
    if _reserve(&w.cmag, next_id + 1) != 0:
        return -1
    for i in range(next_id):
        w.cmag.p[i] = 0.0
    for i in range(k):
        rec = &w.done.p[i * _done_width(d)]
        w.cmag.p[<Py_ssize_t> rec[4]] = rec[3]
    m = 0
    i = 0
    while i < w.pool.n:
        lstp = (
            log(w.cmag.p[<Py_ssize_t> w.pool.p[i]] * w.pool.p[i + 2] / z_s)
            + w.pool.p[i + 3]
        )
        stp = exp(lstp)
        if lstp > -INFINITY:
            for col in range(d):
                w.colwork[col] = w.pool.p[i + 4 + col] - mean[col]
            _add_outer(cov, w.colwork, lstp, d)
            if keep_nodes:
                # compact live nodes to the front of the pool: x, weight
                w.pool.p[2 * m] = w.pool.p[i + 1]
                w.pool.p[2 * m + 1] = stp
            m += 1
        i += 4 + d
    for col in range(d):
        for c2 in range(col + 1, d):
            cov[c2 * d + col] = cov[col * d + c2]
    r.log_z = log(z_s) - q_ref
    r.cond_error = (cond_err + resolution_mass) / z_s
    r.cells = <int> k
    if keep_nodes:
        r.n_nodes = m
        r.nodes = <double*> malloc((m + 1) * sizeof(double))
        r.weights = <double*> malloc((m + 1) * sizeof(double))
        if r.nodes == NULL or r.weights == NULL:
            free(r.nodes)
            free(r.weights)
            r.nodes = NULL
            r.weights = NULL
            return -1
        for i in range(m):
            r.nodes[i] = w.pool.p[2 * i]
            r.weights[i] = w.pool.p[2 * i + 1]
    return 0


# --------------------------------------------------------------------------
# settled cells of the whole support, reused by interval rows
# --------------------------------------------------------------------------

cdef struct LoIdx:
    double lo
    Py_ssize_t i


cdef int _cmp_lo(const void* a, const void* b) noexcept nogil:
    cdef double x = (<const LoIdx*> a).lo
    cdef double y = (<const LoIdx*> b).lo
    return -1 if x < y else (1 if x > y else 0)


cdef struct Reuse:
    # Settled cells of the whole support sorted by ``lo`` (disjoint, so ``hi``
    # is sorted too), and prefix/suffix sums of per-cell records of width
    # ``kw = 2 + 3 d + d d``: z, err(z), s1[d], err(s1)[d], a1[d] and
    # M2[d d] (upper triangle of sum e (Phi - mean)(Phi - mean)^T), all in
    # units of ``exp(-(q - q_ref))``; ``budget_*`` is the error outside the
    # settled cells (dropped cells and tails).
    Py_ssize_t nc
    int kw
    double* clo
    double* chi
    double* pre
    double* suf
    double q_ref
    double z_s
    double budget_mass
    double* budget_mom
    # row workspace: summed record (kw), piece mean, cov and deviation
    double* scratch


cdef void _reuse_init(Reuse* u) noexcept nogil:
    # Every field is defined before ``_reuse_build`` fills the table.
    u.nc = 0
    u.kw = 0
    u.q_ref = 0.0
    u.z_s = 0.0
    u.budget_mass = 0.0
    u.clo = NULL
    u.chi = NULL
    u.pre = NULL
    u.suf = NULL
    u.budget_mom = NULL
    u.scratch = NULL


cdef void _reuse_free(Reuse* u) noexcept nogil:
    free(u.scratch)
    free(u.clo)
    free(u.chi)
    free(u.pre)
    free(u.suf)
    free(u.budget_mom)


cdef int _reuse_build(Ws* w, int d, const double* mean, Reuse* u) noexcept nogil:
    """The ``Reuse`` table of the domain just integrated into ``w``.

    Uses ``w.done``, ``w.pool`` and the per-id steps in ``w.cmag`` left by
    ``_domain_body`` (``keep_nodes`` false).  Returns ``-1`` on allocation
    failure.
    """
    cdef Py_ssize_t dw = _done_width(d)
    cdef Py_ssize_t k = w.done.n // dw
    cdef Py_ssize_t kw = 2 + 3 * d + d * d
    cdef Py_ssize_t i, p, ident, max_id = -1
    cdef int col
    cdef double lstp
    cdef double* rec
    cdef double* cv
    cdef LoIdx* order = NULL
    cdef Py_ssize_t* pos = NULL
    u.nc = k
    u.kw = <int> kw
    u.q_ref = w.q_ref
    u.z_s = w.z_s
    u.budget_mass = w.budget_mass
    u.clo = <double*> malloc((k + 1) * sizeof(double))
    u.chi = <double*> malloc((k + 1) * sizeof(double))
    u.pre = <double*> calloc((k + 1) * kw, sizeof(double))
    u.suf = <double*> calloc((k + 1) * kw, sizeof(double))
    u.budget_mom = <double*> malloc((d + 1) * sizeof(double))
    u.scratch = <double*> malloc((kw + 2 * d + d * d) * sizeof(double))
    order = <LoIdx*> malloc((k + 1) * sizeof(LoIdx))
    if (u.clo == NULL or u.chi == NULL or u.pre == NULL or u.suf == NULL
            or u.budget_mom == NULL or u.scratch == NULL or order == NULL):
        free(order)
        return -1
    for col in range(d):
        u.budget_mom[col] = w.budget_mom.p[col]
    for i in range(k):
        order[i].lo = w.done.p[i * dw + 5]
        order[i].i = i
        if <Py_ssize_t> w.done.p[i * dw + 4] > max_id:
            max_id = <Py_ssize_t> w.done.p[i * dw + 4]
    qsort(order, k, sizeof(LoIdx), _cmp_lo)
    pos = <Py_ssize_t*> malloc((max_id + 2) * sizeof(Py_ssize_t))
    if pos == NULL:
        free(order)
        return -1
    for i in range(max_id + 1):
        pos[i] = -1
    # per-cell records in u.pre[1:], summed into prefix/suffix below
    for p in range(k):
        rec = &w.done.p[order[p].i * dw]
        pos[<Py_ssize_t> rec[4]] = p
        u.clo[p] = rec[5]
        u.chi[p] = rec[6]
        cv = &u.pre[(p + 1) * kw]
        cv[0] = rec[0]
        cv[1] = npmax(rec[1] - rec[2], 0.0)
        for col in range(d):
            cv[2 + col] = rec[_DONE_S1 + col]
            cv[2 + d + col] = npmax(
                rec[_DONE_S1 + 2 * d + col] - rec[_DONE_S1 + 3 * d + col], 0.0
            )
            cv[2 + 2 * d + col] = rec[_DONE_S1 + d + col]
    free(order)
    i = 0
    while i < w.pool.n:
        ident = <Py_ssize_t> w.pool.p[i]
        if ident <= max_id and pos[ident] >= 0:
            lstp = log(w.cmag.p[ident] * w.pool.p[i + 2]) + w.pool.p[i + 3]
            if lstp > -INFINITY:
                for col in range(d):
                    w.colwork[col] = w.pool.p[i + 4 + col] - mean[col]
                _add_outer(&u.pre[(pos[ident] + 1) * kw + 2 + 3 * d], w.colwork,
                           lstp, d)
        i += 4 + d
    free(pos)
    for p in range(k - 1, -1, -1):
        for i in range(kw):
            u.suf[p * kw + i] = u.suf[(p + 1) * kw + i] + u.pre[(p + 1) * kw + i]
    for p in range(k):
        for i in range(kw):
            u.pre[(p + 1) * kw + i] += u.pre[p * kw + i]
    return 0


cdef inline Py_ssize_t _first_at_least(
    const double* a, Py_ssize_t n, double v
) noexcept nogil:
    """First ``i`` with ``a[i] >= v`` (``n`` if none); ``a`` ascending."""
    cdef Py_ssize_t lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] >= v:
            hi = mid
        else:
            lo = mid + 1
    return lo


cdef int _reuse_row(
    const BasisTable* t,
    const double* theta,
    const PotentialPrep* prep,
    const TailResult* growth,
    const GrowthTable** gt,
    Reuse* u,
    double lo,
    double hi,
    double log_z,
    const double* mean,
    double part_v0,
    const double* column_scale,
    const NormOptions* o,
    double* logp,
    double* m_i,
    double* c_i,
    NormResult* rr,
    Ws* w,
) noexcept nogil:
    """Row ``[lo, hi]`` from the settled cells it contains, if they suffice.

    The cells inside the row are summed (prefix/suffix sums); the pieces of
    the row in the cells cut by its finite ends are integrated as domains at
    half the tolerance.  Used only when the settled cells' error budget
    (theirs, the dropped cells', the tails' and the summation rounding) is
    within half the tolerance relative to the row's mass, so the row meets
    ``o.rel_tol`` relative to itself as on its own domain.  Returns ``1``
    when the row is done (``rr.status`` proper or a piece's failure), ``0``
    when it must be integrated as its own domain, ``-1`` on allocation
    failure.
    """
    cdef int d = t.d
    cdef Py_ssize_t kw = u.kw
    cdef Py_ssize_t i0, j1, nfull, j
    cdef int col, c2, piece, out
    cdef double* sm = u.scratch
    cdef double* pm = &u.scratch[kw]
    cdef double* pc = &u.scratch[kw + d]
    cdef double* dev = &u.scratch[kw + d + d * d]
    cdef const double* pa = NULL
    cdef const double* pb = NULL
    cdef double rnd_z, err, a1, pk, ptot, lim, plo, phi_
    cdef bint finite_lo = isfinite(lo), finite_hi = isfinite(hi)
    cdef bint tail_off[2]
    cdef NormOptions oh
    cdef NormResult rp
    i0 = _first_at_least(u.clo, u.nc, lo) if finite_lo else 0
    # cells with chi <= hi: first index with chi > hi
    j1 = _first_at_least(u.chi, u.nc, nextafter(hi, INFINITY)) if finite_hi else u.nc
    nfull = j1 - i0
    if nfull < 1 or (finite_lo and finite_hi and nfull < 3):
        return 0
    if not finite_lo:
        pa = &u.pre[j1 * kw]
        for j in range(kw):
            sm[j] = pa[j]
        rnd_z = 0.0
    elif not finite_hi:
        pa = &u.suf[i0 * kw]
        for j in range(kw):
            sm[j] = pa[j]
        rnd_z = 0.0
    else:
        pa = &u.pre[j1 * kw]
        pb = &u.pre[i0 * kw]
        for j in range(kw):
            sm[j] = pa[j] - pb[j]
        rnd_z = 2.0 * _EPS * (pa[0] + pb[0])
    # Accumulated summation rounding over the table.
    rnd_z = rnd_z + (u.nc + 1) * _EPS * sm[0]
    err = sm[1] + u.budget_mass + rnd_z
    if not (err <= 0.5 * o.rel_tol * sm[0]):
        return 0
    for col in range(d):
        a1 = sm[2 + 2 * d + col]
        err = sm[2 + d + col] + u.budget_mom[col] + (u.nc + 3) * _EPS * a1
        if finite_lo and finite_hi:
            err = err + 2.0 * _EPS * (pa[2 + 2 * d + col] + pb[2 + 2 * d + col])
        if not (err <= 0.5 * o.rel_tol * (a1 + sm[0] * column_scale[col])):
            return 0
    # Probabilities relative to the whole support: full cells, then pieces.
    ptot = sm[0] / u.z_s
    for col in range(d):
        m_i[col] = sm[2 + col] / u.z_s
    for j in range(d * d):
        c_i[j] = 0.0
    for col in range(d):
        for c2 in range(col, d):
            c_i[col * d + c2] = sm[2 + 3 * d + col * d + c2] / u.z_s
    oh = o[0]
    oh.rel_tol = 0.5 * o.rel_tol
    tail_off[0] = False
    tail_off[1] = False
    pk = 0.0
    for piece in range(2):
        if piece == 0:
            if not (finite_lo and lo < u.clo[i0]):
                continue
            plo = lo
            phi_ = u.clo[i0]
        else:
            if not (finite_hi and hi > u.chi[j1 - 1]):
                continue
            plo = u.chi[j1 - 1]
            phi_ = hi
        out = _domain_body(t, theta, prep, growth, gt, tail_off, plo, phi_, part_v0,
                           INFINITY, 9, column_scale, &oh, False, pm, pc, &rp, w)
        if out != 0:
            return -1
        if rp.status != NORM_PROPER:
            rr[0] = rp
            return 1
        pk = exp(rp.log_z - log_z)
        ptot = ptot + pk
        for col in range(d):
            m_i[col] = m_i[col] + pk * pm[col]
            dev[col] = pm[col] - mean[col]
        for col in range(d):
            for c2 in range(col, d):
                c_i[col * d + c2] += pk * (pc[col * d + c2] + dev[col] * dev[c2])
    logp[0] = log(ptot)
    for col in range(d):
        m_i[col] = m_i[col] / ptot
        dev[col] = m_i[col] - mean[col]
    for col in range(d):
        for c2 in range(col, d):
            lim = c_i[col * d + c2] / ptot - dev[col] * dev[c2]
            c_i[col * d + c2] = lim
            c_i[c2 * d + col] = lim
    rr.status = NORM_PROPER
    return 1


# --------------------------------------------------------------------------
# whole support and interval rows
# --------------------------------------------------------------------------

cdef void _ws_init(Ws* w) noexcept nogil:
    w.fwork = NULL
    w.colwork = NULL
    w.ivwork = NULL
    w.prep = NULL
    w.prep_rest = NULL
    w.tailwork = NULL
    w.theta_rest = NULL
    w.grid = NULL
    w.qgrid = NULL
    w.used = NULL
    w.lo.p = NULL
    w.lo.n = 0
    w.lo.cap = 0
    w.hi = w.lo
    w.nlo = w.lo
    w.nhi = w.lo
    w.cmag = w.lo
    w.flt = w.lo
    w.nflt = w.lo
    w.ncmag = w.lo
    w.dropped_mom = w.lo
    w.bt = w.lo
    w.pool = w.lo
    w.done = w.lo
    w.ts = w.lo
    w.budget_mom = w.lo
    w.ts_levels = 0
    w.ts_off[0] = 0


cdef int _ws_open(
    Ws* w, const BasisTable* t, const TailTable* tail_lo, const TailTable* tail_hi
) noexcept nogil:
    cdef Py_ssize_t tw = 3
    _ws_init(w)
    if tail_lo != NULL and tail_work_size(tail_lo) > tw:
        tw = tail_work_size(tail_lo)
    if tail_hi != NULL and tail_work_size(tail_hi) > tw:
        tw = tail_work_size(tail_hi)
    return _ws_alloc(w, t, tw)


cdef int _state_body(
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
    TailResult* growth,
    PotentialPrep* prep,
    Ws* w,
) noexcept nogil:
    """Properness, then the whole support as one domain (``_domain_body``).

    ``growth`` (two entries) and ``prep`` are left set for interval rows of
    the same ``theta``.
    """
    cdef int side, st, col
    cdef const TailTable* tails[2]
    cdef const GrowthTable* gt[2]
    cdef bint has_tail[2]
    cdef int i
    r.status = NORM_PROPER
    r.stage = 0
    r.side = -1
    r.log_z = NAN
    r.rel_error = NAN
    r.cond_error = NAN
    r.cells = 0
    r.n_nodes = 0
    r.nodes = NULL
    r.weights = NULL
    r.part_v0 = NAN
    r.part_step = NAN
    for i in range(6):
        r.diag[i] = NAN
    tails[0] = tail_lo
    tails[1] = tail_hi
    gt[0] = growth_lo
    gt[1] = growth_hi
    # --- 1. properness and growth cutoffs ----------------------------------
    for side in range(2):
        has_tail[side] = tails[side] != NULL
        if not has_tail[side]:
            continue
        growth[side] = tail_verdict(tails[side], theta, o.zeta_cap, 0.5, True,
                                    w.tailwork)
        st = growth_status(&growth[side])
        if st == GROWTH_IMPROPER:
            r.status = NORM_IMPROPER
            return 0
        if st == GROWTH_INCONCLUSIVE:
            r.status = NORM_UNDECIDED
            r.side = side
            return 0
    prepare_potential(t, theta, w.prep, prep)
    w.nused = 0
    for col in range(t.d):
        if theta[col] != 0.0:
            w.used[w.nused] = col
            w.nused += 1
    return _domain_body(t, theta, prep, growth, gt, has_tail, t.lower, t.upper, 0.0,
                        0.0, _GRID, column_scale, o, keep_nodes, mean, cov, r, w)


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
) noexcept nogil:
    """Normalized state of ``theta``; ``0`` or ``-1`` on allocation failure.

    ``r.status`` reports proper, improper or the failure; with
    ``keep_nodes`` the live quadrature nodes and probability weights are
    returned in ``r.nodes``/``r.weights`` (malloc'ed; the caller frees).
    """
    cdef Ws w
    cdef BasisTable local
    cdef double* storage = NULL
    cdef int col
    # At >= 2**24 origin/scale ratios, user-coordinate quadrature nodes
    # lose too many bits to meet the accuracy budget. Ordinary fits retain
    # the original path to avoid perturbing already-resolved tail rows.
    cdef bint canonical = t.orth_degree > 0 or fabs(t.c / t.h) > 16777216.0
    cdef TailResult growth[2]
    cdef PotentialPrep prep
    cdef int out
    if canonical:
        if _integration_table(t, &local, &storage) != 0:
            return -1
    else:
        local = t[0]
    if _ws_open(&w, &local, tail_lo, tail_hi) != 0:
        free(storage)
        _ws_free(&w)
        return -1
    out = _state_body(&local, theta, tail_lo, tail_hi, growth_lo, growth_hi, column_scale,
                      o, keep_nodes, mean, cov, r, growth, &prep, &w)
    if canonical and out == 0 and r.status == NORM_PROPER:
        r.log_z += _physical_log_shift(t, theta)
        if t.log_col_lower >= 0:
            mean[t.log_col_lower] -= log(t.h)
        if t.log_col_upper >= 0:
            mean[t.log_col_upper] -= log(t.h)
        if r.nodes != NULL:
            for col in range(r.n_nodes):
                r.nodes[col] = t.c + t.h * r.nodes[col]
    _ws_free(&w)
    free(storage)
    return out


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
) noexcept nogil:
    """Normalized state plus the weighted conditional moments of interval rows.

    Rows are ``[row_lo, row_hi]`` inside the support (clipping and
    duplicate merging upstream).  A zero-width row is a point observation
    (its log density and ``Phi(x)``); a row covering the support is the
    whole model; any other row is one domain of ``_domain_body`` with its
    own reference, truncation and tolerance, so a small mass far in a tail
    is resolved to ``o.rel_tol`` relative to itself.  ``agg`` receives
    ``sum w log P``, ``sum w E[Phi|B]`` (``d``) and ``sum w Cov(Phi|B)``
    (``d x d``); per-row outputs are written where not NULL.  ``r`` is the
    normalizer's result; a failing row stops the pass with its status in
    ``rr`` and its index in ``fail_row``.  Returns ``-1`` on allocation
    failure, else ``0``.
    """
    cdef Ws w
    cdef BasisTable local
    cdef double* storage = NULL
    cdef double* canonical_rows = NULL
    cdef double* zlo
    cdef double* zhi
    cdef double sw = 0.0
    cdef double point_w = 0.0
    cdef double lh = log(t.h)
    # At >= 2**24 origin/scale ratios, user-coordinate quadrature nodes
    # lose too many bits to meet the accuracy budget. Ordinary fits retain
    # the original path to avoid perturbing already-resolved tail rows.
    cdef bint canonical = fabs(t.c / t.h) > 16777216.0
    cdef TailResult growth[2]
    cdef PotentialPrep prep
    cdef const GrowthTable* gt[2]
    cdef bint tail_on[2]
    cdef int d = t.d
    cdef Py_ssize_t i, j
    cdef int out, col
    cdef double lo, hi, logp, q, wi, step
    cdef double* m_i
    cdef double* c_i
    cdef double* buf = NULL
    cdef Reuse u
    _reuse_init(&u)
    fail_row[0] = -1
    rr.status = NORM_PROPER
    for j in range(1 + d + d * d):
        agg[j] = 0.0
    if canonical:
        if _integration_table(t, &local, &storage) != 0:
            return -1
    else:
        local = t[0]
    canonical_rows = <double*> malloc(2 * (n_rows if n_rows > 0 else 1) * sizeof(double))
    if canonical_rows == NULL:
        free(storage)
        return -1
    zlo = canonical_rows
    zhi = &canonical_rows[n_rows]
    for i in range(n_rows):
        zlo[i] = ((row_lo[i] - t.c) / t.h if isfinite(row_lo[i])
                  else row_lo[i]) if canonical else row_lo[i]
        zhi[i] = ((row_hi[i] - t.c) / t.h if isfinite(row_hi[i])
                  else row_hi[i]) if canonical else row_hi[i]
    if _ws_open(&w, &local, tail_lo, tail_hi) != 0:
        free(storage)
        free(canonical_rows)
        _ws_free(&w)
        return -1
    buf = <double*> malloc((d + d * d + 1) * sizeof(double))
    if buf == NULL:
        _ws_free(&w)
        free(storage)
        free(canonical_rows)
        return -1
    out = _state_body(&local, theta, tail_lo, tail_hi, growth_lo, growth_hi, column_scale,
                      o, False, mean, cov, r, growth, &prep, &w)
    if out == 0 and r.status == NORM_PROPER and n_rows > 0:
        out = _reuse_build(&w, d, mean, &u)
    if out != 0 or r.status != NORM_PROPER:
        free(buf)
        _reuse_free(&u)
        _ws_free(&w)
        free(storage)
        free(canonical_rows)
        return out
    gt[0] = growth_lo
    gt[1] = growth_hi
    for i in range(n_rows):
        lo = zlo[i]
        hi = zhi[i]
        wi = row_w[i]
        logp = NAN
        m_i = &row_mean[i * d] if row_mean != NULL else buf
        c_i = &row_cov[i * d * d] if row_cov != NULL else &buf[d]
        if lo == hi:
            # A point observation: log density and Phi(x).
            eval_point(&local, lo, 0, w.fwork, m_i)
            q = 0.0
            for col in range(d):
                if theta[col] != 0.0:
                    q = q + m_i[col] * theta[col]
            for j in range(d * d):
                c_i[j] = 0.0
            if not isfinite(q):
                rr.status = NORM_ROW_POINT
                fail_row[0] = i
                break
            logp = -q - r.log_z
        elif lo <= local.lower and hi >= local.upper:
            memcpy(m_i, mean, d * sizeof(double))
            memcpy(c_i, cov, d * d * sizeof(double))
            logp = 0.0
        else:
            # The whole support's settled cells when their budget allows,
            # else the row as its own domain.
            out = _reuse_row(&local, theta, &prep, growth, gt, &u, lo, hi, r.log_z, mean,
                             r.part_v0, column_scale, o, &logp, m_i, c_i, rr, &w)
            if out < 0:
                break
            if out == 0:
                tail_on[0] = tail_lo != NULL and not isfinite(lo)
                tail_on[1] = tail_hi != NULL and not isfinite(hi)
                # A finite row starts as one cell (split by the variation
                # rule); a row reaching a tail is cut by the partition.
                step = r.part_step if tail_on[0] or tail_on[1] else INFINITY
                out = _domain_body(&local, theta, &prep, growth, gt, tail_on,
                                   npmax(lo, local.lower), npmin(hi, local.upper), r.part_v0,
                                   step, 9, column_scale, o, False, m_i, c_i, rr, &w)
                if out != 0:
                    break
                logp = rr.log_z - r.log_z
            else:
                out = 0
            if rr.status != NORM_PROPER:
                fail_row[0] = i
                break
        if row_logp != NULL:
            row_logp[i] = logp
        agg[0] += wi * logp
        for col in range(d):
            agg[1 + col] += wi * m_i[col]
        for j in range(d * d):
            agg[1 + d + j] += wi * c_i[j]
    if canonical and out == 0 and r.status == NORM_PROPER:
        # Return all moments in physical basis coordinates.  Interval masses
        # are invariant under the affine change; point rows are *densities*
        # and require the inverse Jacobian 1/h.
        r.log_z += _physical_log_shift(t, theta)
        for i in range(n_rows):
            sw += row_w[i]
            if row_lo[i] == row_hi[i]:
                point_w += row_w[i]
                if row_logp != NULL:
                    row_logp[i] -= lh
            if row_mean != NULL:
                if t.log_col_lower >= 0:
                    row_mean[i * d + t.log_col_lower] -= lh
                if t.log_col_upper >= 0:
                    row_mean[i * d + t.log_col_upper] -= lh
        agg[0] -= point_w * lh
        if t.log_col_lower >= 0:
            mean[t.log_col_lower] -= lh
            agg[1 + t.log_col_lower] -= sw * lh
        if t.log_col_upper >= 0:
            mean[t.log_col_upper] -= lh
            agg[1 + t.log_col_upper] -= sw * lh
    free(buf)
    _reuse_free(&u)
    _ws_free(&w)
    free(storage)
    free(canonical_rows)
    return out


def basis_state(BasisKernel kernel, const double[::1] theta, tail_lo, tail_hi,
                growth_lo, growth_hi, const double[::1] column_scale, options,
                bint keep_nodes=True):
    """Python entry of ``basis_state_c``.

    Returns ``(status, stage, side, log_z, mean, cov, rel_error, cond_error,
    nodes, weights, cells, diag)``.
    """
    cdef const BasisTable* t = &kernel.t
    cdef int d = t.d
    cdef NormOptions o
    cdef NormResult r
    cdef const TailTable* tl = NULL
    cdef const TailTable* tu = NULL
    cdef const GrowthTable* gl = NULL
    cdef const GrowthTable* gu = NULL
    cdef int code
    if theta.shape[0] != d or column_scale.shape[0] != d:
        raise ValueError("theta or column_scale has the wrong length")
    if tail_lo is not None:
        tl = &(<TailKernel> tail_lo).t
        gl = &(<GrowthKernel> growth_lo).t
    if tail_hi is not None:
        tu = &(<TailKernel> tail_hi).t
        gu = &(<GrowthKernel> growth_hi).t
    _options(options, &o)
    mean = np.zeros(d)
    cov = np.zeros((d, d))
    cdef double[::1] mv = mean
    cdef double[:, ::1] cv = cov
    code = basis_state_c(t, &theta[0], tl, tu, gl, gu, &column_scale[0], &o, keep_nodes,
                         &mv[0], &cv[0, 0], &r)
    if code != 0:
        raise MemoryError("normalizer workspace")
    nodes = weights = None
    if r.nodes != NULL:
        nodes = np.zeros(0)
        weights = np.zeros(0)
        if r.n_nodes:
            nodes = np.array(<double[:r.n_nodes]> r.nodes, copy=True)
            weights = np.array(<double[:r.n_nodes]> r.weights, copy=True)
        free(r.nodes)
        free(r.weights)
    return (r.status, r.stage, r.side, r.log_z, mean, cov, r.rel_error, r.cond_error,
            nodes, weights, r.cells, [r.diag[i] for i in range(6)])


cdef void _options(object options, NormOptions* o):
    o.rel_tol = options.rel_tol
    o.initial_cells = options.initial_cells
    o.max_cells = options.max_cells
    o.max_level = options.max_level
    if not 3 <= o.max_level <= _TS_LEVELS:
        raise ValueError(f"max_level must be in [3, {_TS_LEVELS}]")
    o.variation = options.variation
    o.drop_tol = options.drop_tol
    o.zeta_cap = options.zeta_cap


def interval_states(BasisKernel kernel, const double[::1] theta, tail_lo, tail_hi,
                    growth_lo, growth_hi, const double[::1] column_scale, options,
                    const double[::1] row_lo, const double[::1] row_hi,
                    const double[::1] row_w, bint per_row=False):
    """Python entry of ``interval_states_c``.

    Returns ``(status, fail_row, row_status, side, row_side, log_z, mean, cov,
    agg_logp, agg_mean, agg_cov, row_logp, row_mean, row_cov, diag, row_diag)``;
    the per-row arrays are ``None`` unless ``per_row``.
    """
    cdef const BasisTable* t = &kernel.t
    cdef int d = t.d
    cdef Py_ssize_t n = row_lo.shape[0], fail = -1
    cdef NormOptions o
    cdef NormResult r, rr
    cdef const TailTable* tl = NULL
    cdef const TailTable* tu = NULL
    cdef const GrowthTable* gl = NULL
    cdef const GrowthTable* gu = NULL
    cdef int code
    if theta.shape[0] != d or column_scale.shape[0] != d:
        raise ValueError("theta or column_scale has the wrong length")
    if row_hi.shape[0] != n or row_w.shape[0] != n:
        raise ValueError("row arrays have different lengths")
    if tail_lo is not None:
        tl = &(<TailKernel> tail_lo).t
        gl = &(<GrowthKernel> growth_lo).t
    if tail_hi is not None:
        tu = &(<TailKernel> tail_hi).t
        gu = &(<GrowthKernel> growth_hi).t
    _options(options, &o)
    mean = np.zeros(d)
    cov = np.zeros((d, d))
    agg = np.zeros(1 + d + d * d)
    logp = np.zeros(max(n, 1)) if per_row else None
    rmean = np.zeros((max(n, 1), d)) if per_row else None
    rcov = np.zeros((max(n, 1), d, d)) if per_row else None
    cdef double[::1] mv = mean
    cdef double[:, ::1] cv = cov
    cdef double[::1] av = agg
    cdef double* lp = NULL
    cdef double* rm = NULL
    cdef double* rc = NULL
    cdef double[::1] lpv
    cdef double[:, ::1] rmv
    cdef double[:, :, ::1] rcv
    if per_row:
        lpv = logp
        rmv = rmean
        rcv = rcov
        lp = &lpv[0]
        rm = &rmv[0, 0]
        rc = &rcv[0, 0, 0]
    rr.status = NORM_PROPER
    rr.side = -1
    for i in range(6):
        rr.diag[i] = NAN
    cdef const double* lo_p = &row_lo[0] if n > 0 else NULL
    cdef const double* hi_p = &row_hi[0] if n > 0 else NULL
    cdef const double* w_p = &row_w[0] if n > 0 else NULL
    with nogil:
        code = interval_states_c(t, &theta[0], tl, tu, gl, gu, &column_scale[0], &o, n,
                                 lo_p, hi_p, w_p, &mv[0], &cv[0, 0], &av[0], lp, rm, rc,
                                 &r, &rr, &fail)
    if code != 0:
        raise MemoryError("interval normalizer workspace")
    if per_row:
        logp, rmean, rcov = logp[:n], rmean[:n], rcov[:n]
    return (r.status, fail, rr.status, r.side, rr.side, r.log_z, mean, cov, agg[0],
            agg[1:1 + d],
            agg[1 + d:].reshape(d, d), logp, rmean, rcov,
            [r.diag[i] for i in range(6)], [rr.diag[i] for i in range(6)])


# Python-visible statuses of ``basis_state``.
STATUS_PROPER = NORM_PROPER
STATUS_IMPROPER = NORM_IMPROPER
STATUS_UNDECIDED = NORM_UNDECIDED
STATUS_TAIL_CAP = NORM_TAIL_CAP
STATUS_TAIL_NOCONV = NORM_TAIL_NOCONV
STATUS_TAIL_NONFINITE = NORM_TAIL_NONFINITE
STATUS_CELLS_RESOLVE = NORM_CELLS_RESOLVE
STATUS_CELLS_INTEGRATE = NORM_CELLS_INTEGRATE
STATUS_ACCURACY = NORM_ACCURACY
STATUS_NO_REFERENCE = NORM_NO_REFERENCE
STATUS_ROW_POINT = NORM_ROW_POINT
