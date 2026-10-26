# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled curvature separation: every region where ``q''`` falls below 0.

``separate_c`` is the compiled body of ``curvature_search._separate_curvature``.
It works on the scaled curvature ``H = q'' / ||pieces||`` of
``_basis_kernels.curvature_rows`` in the canonical coordinate:

1. **Structural signs**: an enabled log amplitude below zero is reported
   as a violated bound.
2. **Domain**: each infinite end is searched out to the cutoff its tail
   verdict proves (``basis_tails``), or to the configured cap when the tail is
   violated or unproven; each finite end with a positive log amplitude has an
   excluded neighborhood where ``q'' >= 0`` is enclosed.
3. **Panels**: a fixed hierarchy per basis -- dyadic panels in ``r = |z|``
   toward the origin, shells in ``d = 1 / (1 + r)`` toward infinity and in the
   endpoint distance ``e = 1 - r`` on a bounded support -- whose theta-free
   per-atom rows are cached at each panel's Chebyshev nodes (``CurvatureSearch``).
   For one ``theta``, ``H`` at the nodes is a matrix-vector product and its
   Chebyshev coefficients a DCT.  A panel is resolved when its last
   coefficients fall below ``res_frac * tau`` (or the rounding floor), and
   bisected otherwise.  A resolved panel whose lower bound ``c_0 - sum |c_k|``
   reaches below ``-tau`` is sampled for its roots and the critical points of
   the interpolant; every subinterval with minimum below ``-tau`` is a
   negative segment.
4. **Cuts**: each negative segment contributes its minimum and ``n_extra``
   Chebyshev points (geometric points beyond an unbounded one), as scaled
   column rows ``R @ atom_rows`` (``q''`` up to a positive factor).

The verdict is ``verified`` when no segment reaches below ``-tau``, every
tail is proven and every panel resolved; ``violated`` when a segment, a tail
or a log amplitude is negative; ``unverified`` otherwise (fail closed).
"""

import cython
import numpy as np
cimport numpy as cnp

from libc.math cimport INFINITY, M_PI, cos, fabs, isfinite, isnan, ldexp, sqrt
from libc.stdlib cimport free, malloc, realloc

from .._model._basis_kernels cimport (
    TAIL_NONNEGATIVE,
    TAIL_VIOLATED,
    BasisKernel,
    BasisTable,
    Iv,
    TailKernel,
    TailTable,
    curvature_rows,
    curvature_scaled,
    endpoint_exclusion,
    prep_size,
    tail_verdict,
    tail_work_size,
)

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def same_constraint_rows(
    const double[:, :] units, const double[:, :] rows, double tolerance, /
):
    """Compare rows entry by entry, stopping at the first mismatch.

    Parameters
    ----------
    units, rows : numpy.ndarray
        Finite unit rows with matching column counts, shape (k, d) and (m, d).
        Strided and read-only arrays are supported.
    tolerance : float
        Relative entry tolerance, measured against the row in units.

    Returns
    -------
    numpy.ndarray
        Boolean comparisons, shape (k, m), with identical zero patterns.

    Raises
    ------
    ValueError
        If the column counts differ.
    """
    cdef Py_ssize_t k = units.shape[0], m = rows.shape[0], d = units.shape[1]
    cdef Py_ssize_t i, j, column
    cdef double u, r
    if rows.shape[1] != d:
        raise ValueError("constraint column counts must match")
    result = np.empty((k, m), dtype=np.bool_)
    cdef cnp.npy_bool[:, ::1] matches = result
    with nogil:
        for i in range(k):
            for j in range(m):
                matches[i, j] = True
                for column in range(d):
                    u = units[i, column]
                    r = rows[j, column]
                    if (r != 0.0) != (u != 0.0) or not (
                        fabs(r - u) <= tolerance * fabs(u)
                    ):
                        matches[i, j] = False
                        break
    return result


cdef double _EPS = 1.1102230246251565e-16

cdef enum:
    SUP_FULL = 0
    SUP_LOWER = 1       # [L, inf): the origin z = 0 is the endpoint
    SUP_UPPER = 2       # (-inf, U]
    SUP_BOUNDED = 3
    K_R = 0             # r linear
    K_D = 1             # d = 1 / (1 + r) linear (toward infinity)
    K_E = 2             # e = 1 - r linear (toward a bounded endpoint)
    _MAXN = 64          # nodes per panel
    _MAXNP = 64


# --------------------------------------------------------------------------
# growable arrays (every allocation failure is returned, never aborted)
# --------------------------------------------------------------------------

cdef inline int _grow(
    void** p, Py_ssize_t* cap, Py_ssize_t need, size_t item
) noexcept nogil:
    cdef Py_ssize_t c = cap[0]
    cdef void* q
    if need <= c:
        return 0
    if c < 64:
        c = 64
    while c < need:
        c *= 2
    q = realloc(p[0], <size_t> c * item)
    if q == NULL:
        return -1
    p[0] = q
    cap[0] = c
    return 0


cdef struct IArr:
    int* p
    Py_ssize_t n
    Py_ssize_t cap


cdef struct SArr:
    Segment* p
    Py_ssize_t n
    Py_ssize_t cap


cdef inline int _ipush(IArr* a, int v) noexcept nogil:
    if _grow(<void**> &a.p, &a.cap, a.n + 1, sizeof(int)) != 0:
        return -1
    a.p[a.n] = v
    a.n += 1
    return 0


cdef inline int _spush(SArr* a, Segment v) noexcept nogil:
    if _grow(<void**> &a.p, &a.cap, a.n + 1, sizeof(Segment)) != 0:
        return -1
    a.p[a.n] = v
    a.n += 1
    return 0


# --------------------------------------------------------------------------
# Chebyshev helpers
# --------------------------------------------------------------------------

cdef inline double _cheb(const double* c, int n, double t) noexcept nogil:
    cdef double b1 = 0.0, b2 = 0.0, b0
    cdef int j
    for j in range(n - 1, 0, -1):
        b0 = 2.0 * t * b1 - b2 + c[j]
        b2 = b1
        b1 = b0
    return t * b1 - b2 + c[0]


cdef void _chebder(const double* c, int n, double* d) noexcept nogil:
    """``d`` (``n - 1`` coefficients) of the derivative of a Chebyshev series."""
    cdef double dp1 = 0.0, dp2 = 0.0, dj
    cdef int j
    if n < 2:
        d[0] = 0.0
        return
    for j in range(n - 1):
        d[j] = 0.0
    for j in range(n - 1, 0, -1):
        dj = dp2 + 2.0 * j * c[j]
        d[j - 1] = dj
        dp2 = dp1
        dp1 = dj
    d[0] *= 0.5


cdef double _bisect_root(
    const double* c, int n, double a, double b, double fa
) noexcept nogil:
    cdef double m, fm
    cdef int _k
    for _k in range(80):
        m = 0.5 * (a + b)
        if m == a or m == b:
            break
        fm = _cheb(c, n, m)
        if (fm < 0) == (fa < 0):
            a = m
            fa = fm
        else:
            b = m
    return 0.5 * (a + b)


# --------------------------------------------------------------------------
# panel hierarchy and cached rows
# --------------------------------------------------------------------------

cdef int _new_panel(CurvSearch* S, int side, int kind, double a, double b, int depth,
                    int shell) noexcept nogil:
    cdef Py_ssize_t cap = S.capP
    cdef Panel* p
    if _grow(<void**> &S.P, &cap, S.nP + 1, sizeof(Panel)) != 0:
        return -1
    S.capP = <int> cap
    p = &S.P[S.nP]
    p.side = side
    p.kind = kind
    p.a = a
    p.b = b
    p.depth = depth
    p.shell = shell
    p.child[0] = -1
    p.child[1] = -1
    p.rowoff = -1
    S.nP += 1
    return S.nP - 1


cdef inline void _node_r(
    const Panel* p, double t, double* r, double* drdt
) noexcept nogil:
    cdef double mid = 0.5 * (p.a + p.b)
    cdef double half = 0.5 * (p.b - p.a)
    cdef double d, e
    if p.kind == K_R:
        r[0] = mid + half * t
        drdt[0] = half
    elif p.kind == K_D:
        d = mid - half * t
        r[0] = (1.0 - d) / d
        drdt[0] = half / (d * d)
    else:
        e = mid - half * t
        r[0] = 1.0 - e
        drdt[0] = half


cdef inline double _edge_r(const Panel* p, int hi) noexcept nogil:
    if p.kind == K_R:
        return p.b if hi else p.a
    if p.kind == K_D:
        return (1.0 - p.a) / p.a if hi else (1.0 - p.b) / p.b
    return 1.0 - p.a if hi else 1.0 - p.b


cdef int _split(CurvSearch* S, int ip) noexcept nogil:
    cdef Panel p = S.P[ip]
    cdef double m = 0.5 * (p.a + p.b)
    cdef int c0, c1
    if p.kind == K_R:
        c0 = _new_panel(S, p.side, p.kind, p.a, m, p.depth + 1, p.shell)
        c1 = _new_panel(S, p.side, p.kind, m, p.b, p.depth + 1, p.shell)
    else:
        # d and e decrease with r: the first child is nearer the origin.
        c0 = _new_panel(S, p.side, p.kind, m, p.b, p.depth + 1, p.shell)
        c1 = _new_panel(S, p.side, p.kind, p.a, m, p.depth + 1, p.shell)
    if c0 < 0 or c1 < 0:
        return -1
    S.P[ip].child[0] = c0
    S.P[ip].child[1] = c1
    return 0


cdef inline int _resize(double** p, Py_ssize_t n) noexcept nogil:
    cdef void* q = realloc(p[0], sizeof(double) * n)
    if q == NULL:
        return -1
    p[0] = <double*> q
    return 0


cdef int _ensure_rows(CurvSearch* S, const int* idx, int cnt) noexcept nogil:
    """Compute and cache the per-atom rows of the listed panels lacking them."""
    cdef int need = 0, i, k, a, q
    cdef int n = S.n, np_ = S.np, A = S.A
    cdef Py_ssize_t off, add, zo, zc
    cdef double r, drdt
    cdef double* z
    cdef double* rt
    cdef double* nt
    cdef double* ra
    cdef double* rr
    cdef int* rk
    cdef int* who
    cdef Panel* p
    cdef int rc = 0
    for i in range(cnt):
        if S.P[idx[i]].rowoff < 0:
            need += 1
    if need == 0:
        return 0
    z = <double*> malloc(sizeof(double) * need * n)
    rt = <double*> malloc(sizeof(double) * need * n * (A if A else 1))
    nt = <double*> malloc(sizeof(double) * need * n * (A if A else 1))
    ra = <double*> malloc(sizeof(double) * 2 * need * n)
    rr = <double*> malloc(sizeof(double) * 2 * need * n)
    rk = <int*> malloc(sizeof(int) * need * n)
    who = <int*> malloc(sizeof(int) * need)
    if (z == NULL or rt == NULL or nt == NULL or ra == NULL or rr == NULL or rk == NULL
            or who == NULL):
        rc = -1
    if rc == 0:
        q = 0
        for i in range(cnt):
            p = &S.P[idx[i]]
            if p.rowoff >= 0:
                continue
            who[q] = idx[i]
            for k in range(n):
                _node_r(p, S.nodes[k], &r, &drdt)
                z[q * n + k] = p.side * r
            q += 1
        if A > 0:
            rc = curvature_rows(S.t, need * n, z, True, rt, nt, rk, ra, rr)
    if rc == 0:
        add = <Py_ssize_t> need * np_ * (A if A else 1)
        if S.nrow + add > S.caprow:
            # R, NR, Z and DZ share one offset space and one capacity.
            zc = 2 * S.caprow if S.caprow else 4096
            while zc < S.nrow + add:
                zc *= 2
            rc = _resize(&S.R, zc)
            if rc == 0:
                rc = _resize(&S.NR, zc)
            if rc == 0:
                rc = _resize(&S.Z, zc)
            if rc == 0:
                rc = _resize(&S.DZ, zc)
            if rc == 0:
                S.caprow = zc
    if rc == 0:
        for i in range(need):
            p = &S.P[who[i]]
            off = S.nrow
            p.rowoff = off
            for a in range(A):
                for k in range(np_):
                    if k < n:
                        S.R[off + a * np_ + k] = rt[(i * n + k) * A + a]
                        # An out-of-range atom keeps its negative marker.
                        S.NR[off + a * np_ + k] = nt[(i * n + k) * A + a]
                    else:
                        S.R[off + a * np_ + k] = 0.0
                        S.NR[off + a * np_ + k] = 0.0
            # Node locations and dz/dt share the row offset (one per node).
            zo = off
            for k in range(np_):
                r = 0.0
                drdt = 1.0
                if k < n:
                    _node_r(p, S.nodes[k], &r, &drdt)
                S.Z[zo + k] = p.side * r
                S.DZ[zo + k] = drdt
            S.nrow += np_ * (A if A else 1)
        S.kernel_points += need * n
    free(z)
    free(rt)
    free(nt)
    free(ra)
    free(rr)
    free(rk)
    free(who)
    return rc


cdef struct Leafs:
    IArr idx
    double* H
    double* c
    Py_ssize_t cap
    int unresolved


cdef int _leaf_push(Leafs* L, int np_, int n, int ip) noexcept nogil:
    cdef Py_ssize_t cap
    cdef void* q
    if L.idx.n + 1 > L.cap:
        cap = L.cap * 2 if L.cap else 256
        q = realloc(L.H, sizeof(double) * cap * np_)
        if q == NULL:
            return -1
        L.H = <double*> q
        q = realloc(L.c, sizeof(double) * cap * n)
        if q == NULL:
            return -1
        L.c = <double*> q
        L.cap = cap
    return _ipush(&L.idx, ip)


cdef int _eval_panel(
    CurvSearch* S, const Panel* p, const double* coef,
    double* H, double* c, double* tail, double* noise
) noexcept nogil:
    """``H`` at the nodes, its Chebyshev coefficients, tail and rounding floor.

    ``H = sum_i c_i R_i / ||c N||`` from the cached rows, with the norm scaled
    by its largest term so that tiny rows do not underflow when squared.  On a
    panel where the atoms with nonzero coefficients lie beyond the rows'
    dynamic range at some node (all within ``1e-288`` of an out-of-range one,
    ``N < 0`` in the rows) the rows do not determine ``H``, and the panel is
    evaluated directly (``curvature_scaled``).  Returns ``-1`` on allocation
    failure.
    """
    cdef int n = S.n, np_ = S.np, A = S.A
    cdef const double* R = S.R + p.rowoff
    cdef const double* NR = S.NR + p.rowoff
    cdef const double* Ra
    cdef const double* Na
    cdef double V[_MAXNP]
    cdef double W[_MAXNP]
    cdef double M[_MAXNP]
    cdef double D[_MAXNP]
    cdef double E[_MAXNP]
    cdef double L[_MAXNP]
    cdef double ta, acc, tl, cond, d, v, err
    cdef int a, k, j
    cdef bint direct = False
    for k in range(np_):
        V[k] = 0.0
        W[k] = 0.0
        M[k] = 0.0
        D[k] = 0.0
    # Pass 1: the largest weighted norm per node (and out-of-range atoms).
    for a in range(A):
        ta = coef[a]
        if ta == 0.0:
            continue
        Na = NR + a * np_
        for k in range(np_):
            if Na[k] < 0.0:
                if fabs(ta) > D[k]:
                    D[k] = fabs(ta)
            elif fabs(ta) * Na[k] > M[k]:
                M[k] = fabs(ta) * Na[k]
    # Pass 2: numerator and the scaled squared norm.
    for a in range(A):
        ta = coef[a]
        if ta == 0.0:
            continue
        Ra = R + a * np_
        Na = NR + a * np_
        for k in range(np_):
            if Na[k] < 0.0 or M[k] == 0.0:
                continue
            V[k] += ta * Ra[k] / M[k]
            v = ta * Na[k] / M[k]
            W[k] += v * v
    for k in range(n):
        if D[k] > 0.0 and not (M[k] > 1e-288 * D[k]):
            direct = True
    err = 0.0
    if direct:
        for k in range(n):
            D[k] = S.Z[p.rowoff + k]
        if curvature_scaled(S.t, coef, n, D, True, H, E, L, NULL, NULL, NULL,
                            NULL) != 0:
            return -1
        for k in range(n):
            if E[k] > err:
                err = E[k]
        for k in range(n, np_):
            H[k] = 0.0
        S.direct_panels += 1
    else:
        for k in range(np_):
            H[k] = V[k] / sqrt(W[k]) if W[k] > 0.0 else 0.0
    for j in range(n):
        acc = 0.0
        for k in range(n):
            acc += S.Tm[j * n + k] * H[k]
        c[j] = acc
    tl = 0.0
    for j in range(n - 3, n):
        if j >= 0 and fabs(c[j]) > tl:
            tl = fabs(c[j])
    tail[0] = tl
    # Rounding floor: the kernel's relative accuracy plus the effect of
    # rounding the node location (|z| eps |dH/dz|).
    cond = 0.0
    for k in range(n):
        d = 0.0
        for j in range(n):
            d += S.Dv[k * n + j] * c[j]
        v = fabs(S.Z[p.rowoff + k]) * _EPS * fabs(d) / S.DZ[p.rowoff + k]
        if v > cond:
            cond = v
    noise[0] = 2e-14 * (1.0 + S.pmax / 50.0) + cond + err
    return 0


cdef int _resolve(
    CurvSearch* S, const int* roots, int nroots, const double* coef,
    const SearchOptions* o, Leafs* L
) noexcept nogil:
    """Resolve the subtrees under ``roots``; leaves are appended to ``L``."""
    cdef IArr work, cur, stack
    cdef int i, q, rc = 0, top
    cdef double thr = o.res_frac * o.tau
    cdef double lim, tail, noise
    cdef double* H
    cdef double* c
    work.p = NULL
    work.n = 0
    work.cap = 0
    cur.p = NULL
    cur.n = 0
    cur.cap = 0
    stack.p = NULL
    stack.n = 0
    stack.cap = 0
    for i in range(nroots):
        if _ipush(&work, roots[i]) != 0:
            rc = -1
            break
    while rc == 0 and work.n:
        cur.n = 0
        for i in range(<int> work.n):
            stack.n = 0
            if _ipush(&stack, work.p[i]) != 0:
                rc = -1
                break
            while stack.n and rc == 0:
                stack.n -= 1
                top = stack.p[stack.n]
                if S.P[top].child[0] >= 0:
                    if (_ipush(&stack, S.P[top].child[0]) != 0
                            or _ipush(&stack, S.P[top].child[1]) != 0):
                        rc = -1
                elif _ipush(&cur, top) != 0:
                    rc = -1
            if rc != 0:
                break
        if rc != 0:
            break
        if _ensure_rows(S, cur.p, <int> cur.n) != 0:
            rc = -1
            break
        work.n = 0
        for i in range(<int> cur.n):
            q = cur.p[i]
            if _leaf_push(L, S.np, S.n, q) != 0:
                rc = -1
                break
            L.idx.n -= 1        # capacity reserved; pushed below if resolved
            H = L.H + L.idx.n * S.np
            c = L.c + L.idx.n * S.n
            if _eval_panel(S, &S.P[q], coef, H, c, &tail, &noise) != 0:
                rc = -1
                break
            S.panel_evals += 1
            lim = thr if thr > 10.0 * noise else 10.0 * noise
            # Fail closed at the depth limit or the per-search panel budget
            # (the cached rows of every new panel are kept).
            if (tail <= lim or S.P[q].depth >= o.max_depth
                    or S.nP - S.np_start >= o.max_new_panels):
                if tail > lim:
                    L.unresolved += 1
                L.idx.n += 1
            else:
                if _split(S, q) != 0 or _ipush(&work, q) != 0:
                    rc = -1
                    break
    free(work.p)
    free(cur.p)
    free(stack.p)
    return rc


cdef int _collect_leaf(
    const CurvSearch* S, const Leafs* L, int li, const SearchOptions* o, SArr* segs
) noexcept nogil:
    """Append the negative segments (minimum below ``-tau``) of one leaf."""
    cdef int n = S.n
    cdef const double* c = L.c + li * n
    cdef const Panel* p = &S.P[L.idx.p[li]]
    cdef double tau = o.tau
    cdef double lb = c[0]
    cdef int j, i, m, ncrit = 0, nb = 0, sgi, rc = 0
    cdef double dcoef[_MAXN]
    cdef double* ts
    cdef double* ps
    cdef double* ds
    cdef double* crit
    cdef double* brk
    cdef double a, b, fa, mm, fm, sm, tsm, v, rlo, rhi, rmn, drdt
    cdef Segment g
    for j in range(1, n):
        lb -= fabs(c[j])
    if lb >= -tau:
        return 0
    m = o.nsamp_factor * n
    if m < 64:
        m = 64
    ts = <double*> malloc(sizeof(double) * m)
    ps = <double*> malloc(sizeof(double) * m)
    ds = <double*> malloc(sizeof(double) * m)
    crit = <double*> malloc(sizeof(double) * m)
    brk = <double*> malloc(sizeof(double) * (m + 2))
    if ts == NULL or ps == NULL or ds == NULL or crit == NULL or brk == NULL:
        free(ts)
        free(ps)
        free(ds)
        free(crit)
        free(brk)
        return -1
    _chebder(c, n, dcoef)
    for i in range(m):
        ts[i] = -cos(M_PI * i / (m - 1))
        ps[i] = _cheb(c, n, ts[i])
        ds[i] = _cheb(dcoef, n - 1, ts[i])
    ts[0] = -1.0
    ts[m - 1] = 1.0
    # Critical points of the interpolant (minima: derivative from - to +).
    for i in range(m - 1):
        if ds[i] < 0 and ds[i + 1] >= 0:
            a = ts[i]
            b = ts[i + 1]
            fa = ds[i]
            for _ in range(80):
                mm = 0.5 * (a + b)
                if mm == a or mm == b:
                    break
                fm = _cheb(dcoef, n - 1, mm)
                if (fm < 0) == (fa < 0):
                    a = mm
                    fa = fm
                else:
                    b = mm
            crit[ncrit] = 0.5 * (a + b)
            ncrit += 1
    # Roots split the panel into sign-constant subintervals.
    brk[nb] = -1.0
    nb += 1
    for i in range(m - 1):
        if ps[i] == 0.0 and i > 0:
            brk[nb] = ts[i]
            nb += 1
        elif (ps[i] < 0) != (ps[i + 1] < 0) and ps[i + 1] != 0.0:
            brk[nb] = _bisect_root(c, n, ts[i], ts[i + 1], ps[i])
            nb += 1
    brk[nb] = 1.0
    nb += 1
    for sgi in range(nb - 1):
        a = brk[sgi]
        b = brk[sgi + 1]
        if not (b > a):
            continue
        tsm = 0.5 * (a + b)
        sm = _cheb(c, n, tsm)
        for i in range(m):
            if ts[i] >= a and ts[i] <= b and ps[i] < sm:
                sm = ps[i]
                tsm = ts[i]
        for i in range(ncrit):
            if crit[i] >= a and crit[i] <= b:
                v = _cheb(c, n, crit[i])
                if v < sm:
                    sm = v
                    tsm = crit[i]
        if not (sm < -tau):
            continue
        _node_r(p, a, &rlo, &drdt)
        _node_r(p, b, &rhi, &drdt)
        _node_r(p, tsm, &rmn, &drdt)
        if a == -1.0:
            rlo = _edge_r(p, 0)
        if b == 1.0:
            rhi = _edge_r(p, 1)
        if rlo > rhi:
            v = rlo
            rlo = rhi
            rhi = v
        g.z_lo = rlo
        g.z_hi = rhi
        g.z_min = rmn
        g.h_min = sm
        g.to_inf = 0
        g.touches_end = 0
        if _spush(segs, g) != 0:
            rc = -1
            break
    free(ts)
    free(ps)
    free(ds)
    free(crit)
    free(brk)
    return rc


cdef int _shell_panel(
    CurvSearch* S, int si, int side, int j, int bounded
) noexcept nogil:
    """Outer shell ``j >= 1`` of a branch (created on first use)."""
    cdef int jj, ip
    cdef Py_ssize_t cap
    while S.n_outer[si] < j:
        jj = S.n_outer[si] + 1
        ip = _new_panel(S, side, K_E if bounded else K_D, ldexp(1.0, -(jj + 1)),
                        ldexp(1.0, -jj), 0, jj)
        if ip < 0:
            return -1
        cap = S.cap_outer[si]
        if _grow(<void**> &S.outer[si], &cap, S.n_outer[si] + 1, sizeof(int)) != 0:
            return -1
        S.cap_outer[si] = <int> cap
        S.outer[si][S.n_outer[si]] = ip
        S.n_outer[si] += 1
    return S.outer[si][j - 1]


cdef int _end_panel(CurvSearch* S, int si, int side, int j) noexcept nogil:
    """Inner panel ``[2**j, 2**(j+1)]``, ``j <= -1``, toward an endpoint origin."""
    cdef int want = -j, jj, ip
    cdef Py_ssize_t cap
    while S.n_inner_end[si] < want:
        jj = -(S.n_inner_end[si] + 1)
        ip = _new_panel(S, side, K_R, ldexp(1.0, jj), ldexp(1.0, jj + 1), 0, jj)
        if ip < 0:
            return -1
        cap = S.cap_inner_end[si]
        if _grow(<void**> &S.inner_end[si], &cap, S.n_inner_end[si] + 1,
                 sizeof(int)) != 0:
            return -1
        S.cap_inner_end[si] = <int> cap
        S.inner_end[si][S.n_inner_end[si]] = ip
        S.n_inner_end[si] += 1
    return S.inner_end[si][want - 1]


cdef void _leaf_range(const CurvSearch* S, const Leafs* L, int l0, double* lo,
                      double* hi) noexcept nogil:
    cdef int li, k
    lo[0] = INFINITY
    hi[0] = -INFINITY
    for li in range(l0, <int> L.idx.n):
        for k in range(S.n):
            if L.H[li * S.np + k] < lo[0]:
                lo[0] = L.H[li * S.np + k]
            if L.H[li * S.np + k] > hi[0]:
                hi[0] = L.H[li * S.np + k]


# --------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------

cdef void search_free(SearchResult* r) noexcept nogil:
    free(r.bound_cols)
    free(r.segments)
    free(r.minima)
    free(r.cut_points)
    free(r.cut_rows)
    r.bound_cols = NULL
    r.segments = NULL
    r.minima = NULL
    r.cut_points = NULL
    r.cut_rows = NULL


cdef int _segment_cmp(const Segment* x, const Segment* y) noexcept nogil:
    return (x.z_lo > y.z_lo) - (x.z_lo < y.z_lo)


cdef void _sort_segments(Segment* s, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef Segment v
    for i in range(1, n):
        v = s[i]
        j = i - 1
        while j >= 0 and _segment_cmp(&s[j], &v) > 0:
            s[j + 1] = s[j]
            j -= 1
        s[j + 1] = v


cdef int _branch(
    CurvSearch* S, int si, int side, const double* coef,
    double r_cap, double r_excl, bint tail_violated, const SearchOptions* o,
    Leafs* L, SArr* segs, int* unconverged
) noexcept nogil:
    """Search one branch ``z = side r``; negative segments go to ``segs`` in ``r``."""
    cdef int support = S.support
    cdef int i, j, ip, streak
    cdef double lo, hi, prev_lo = 0.0, prev_hi = 0.0
    cdef bint good, origin_end = support == SUP_LOWER or support == SUP_UPPER
    cdef Segment g
    cdef double r_stop
    cdef int* roots
    cdef int nroots
    # --- inner -------------------------------------------------------------
    if not origin_end:
        L.idx.n = 0
        if _resolve(S, S.inner_fixed[si], S.n_inner_fixed[si], coef, o, L) != 0:
            return -1
        for i in range(<int> L.idx.n):
            if _collect_leaf(S, L, i, o, segs) != 0:
                return -1
    else:
        # The origin is the endpoint: scan toward it until it is excluded
        # (log barrier) or H has converged over consecutive panels.
        streak = 0
        unconverged[0] = 1
        j = -1
        while j >= o.jmin_end:
            ip = _end_panel(S, si, side, j)
            if ip < 0:
                return -1
            L.idx.n = 0
            if _resolve(S, &ip, 1, coef, o, L) != 0:
                return -1
            for i in range(<int> L.idx.n):
                if _collect_leaf(S, L, i, o, segs) != 0:
                    return -1
            if ldexp(1.0, j) <= r_excl:
                unconverged[0] = 0
                break
            _leaf_range(S, L, 0, &lo, &hi)
            # Converged: H is constant to a tenth of the threshold over the
            # shell and from the previous one (its sign is then settled).
            good = (fabs(lo - prev_lo) < 0.1 * o.tau
                    and fabs(hi - prev_hi) < 0.1 * o.tau and hi - lo < 0.1 * o.tau)
            streak = streak + 1 if good else 0
            prev_lo = lo
            prev_hi = hi
            if streak >= o.stop_shells:
                unconverged[0] = 0
                if hi < -o.tau:
                    # Negative right up to the endpoint.
                    g.z_lo = 0.0
                    g.z_hi = ldexp(1.0, j + 1)
                    g.z_min = 0.0
                    g.h_min = lo
                    g.to_inf = 0
                    g.touches_end = 1
                    if _spush(segs, g) != 0:
                        return -1
                break
            j -= 1
    # --- outer -------------------------------------------------------------
    if support == SUP_BOUNDED:
        nroots = o.end_shells
        roots = <int*> malloc(sizeof(int) * (nroots if nroots > 0 else 1))
        if roots == NULL:
            return -1
        nroots = 0
        for j in range(1, o.end_shells + 1):
            if ldexp(1.0, -j) <= r_excl:
                break       # this and every later shell lie inside the exclusion
            ip = _shell_panel(S, si, side, j, True)
            if ip < 0:
                free(roots)
                return -1
            roots[nroots] = ip
            nroots += 1
        L.idx.n = 0
        if _resolve(S, roots, nroots, coef, o, L) != 0:
            free(roots)
            return -1
        free(roots)
        for i in range(<int> L.idx.n):
            if _collect_leaf(S, L, i, o, segs) != 0:
                return -1
        return 0
    r_stop = 1.0
    for j in range(1, o.jmax + 1):
        if ldexp(1.0, j) - 1.0 >= r_cap:
            break
        ip = _shell_panel(S, si, side, j, False)
        if ip < 0:
            return -1
        L.idx.n = 0
        if _resolve(S, &ip, 1, coef, o, L) != 0:
            return -1
        for i in range(<int> L.idx.n):
            if _collect_leaf(S, L, i, o, segs) != 0:
                return -1
        r_stop = ldexp(1.0, j + 1) - 1.0
    if tail_violated:
        # The tail is negative far out: one segment to infinity.
        g.z_lo = r_stop
        g.z_hi = INFINITY
        g.z_min = r_stop
        g.h_min = -1.0
        g.to_inf = 1
        g.touches_end = 0
        if _spush(segs, g) != 0:
            return -1
    return 0


cdef int _search_body(
    CurvSearch* S,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
    double* coef,
    double* work,
    double* prep,
    double* rest,
    Iv* ivwork,
    Leafs* L,
    SArr* segs,
    SArr* all_segs,
) noexcept nogil:
    cdef const BasisTable* t = S.t
    cdef int d = t.d, A = S.A
    cdef int i, j, k, nb, si, side, unconverged, base, npts, q, m
    cdef double h = t.h
    cdef double r_cap[2]
    cdef double r_excl[2]
    cdef bint violated[2]
    cdef bint unverified = False, any_violation = False
    cdef const TailTable* tt
    cdef double width, r0, f, big, v, r_excl_b
    cdef Segment g
    cdef int* order
    cdef double* zc
    cdef double* rr
    cdef double* nn
    cdef double* ra
    cdef double* rm
    cdef int* rk
    cdef int rc

    r.tau = o.tau
    # --- structural sign of log amplitudes --------------------------------
    nb = 0
    for i in range(d):
        if t.nonneg[i] and theta[i] < 0.0:
            nb += 1
    if nb:
        r.bound_cols = <int*> malloc(nb * sizeof(int))
        if r.bound_cols == NULL:
            return -1
        nb = 0
        for i in range(d):
            if t.nonneg[i] and theta[i] < 0.0:
                r.bound_cols[nb] = i
                nb += 1
        r.n_bound = nb
        r.status = SEARCH_VIOLATED
        r.a = t.lower
        r.b = t.upper
        return 0

    # --- domain: tail cutoffs and endpoint exclusions (canonical r) ----------
    for side in range(2):
        r_cap[side] = 1.0
        r_excl[side] = 0.0
        violated[side] = False
        tt = tail_hi if side == 1 else tail_lo
        if tt == NULL:
            if (side == 1 and isfinite(t.upper)) or (side == 0 and isfinite(t.lower)):
                width = endpoint_exclusion(t, theta, side, 0.0, rest, prep, ivwork)
                r_excl[side] = width / h
            continue
        r.tails[r.n_tails] = tail_verdict(tt, theta, o.zeta_cap, 0.0, False, work)
        r.tail_side[r.n_tails] = side
        if r.tails[r.n_tails].status == TAIL_NONNEGATIVE:
            r_cap[side] = r.tails[r.n_tails].zeta
        else:
            if r.tails[r.n_tails].status == TAIL_VIOLATED:
                violated[side] = True
                any_violation = True
                r_cap[side] = o.fallback_zeta
            else:
                unverified = True
                # A positive leader too weak to dominate before the cap: search
                # the whole range the tail analysis could not cover.
                if r.tails[r.n_tails].has_lead and r.tails[r.n_tails].lead_value > 0.0:
                    r_cap[side] = o.zeta_cap
                else:
                    r_cap[side] = o.fallback_zeta
        r.n_tails += 1

    # --- atom coefficients ------------------------------------------------------
    # A coefficient within rounding of its terms is zero: on a face whose
    # functional cancels exactly (a suppressed leader), the residue has no
    # sign, and its summation order must not decide the far curvature.
    for i in range(A):
        v = 0.0
        f = 0.0
        for j in range(d):
            v += t.atom_row[i * d + j] * theta[j]
            f += fabs(t.atom_row[i * d + j] * theta[j])
        if fabs(v) <= 4.0 * d * _EPS * f:
            v = 0.0
        coef[i] = v

    # --- branches ---------------------------------------------------------------
    for k in range(S.nside):
        side = S.sides[k]
        si = 0 if side > 0 else 1
        segs.n = 0
        unconverged = 0
        # Branch side +1 meets the upper end, -1 the lower end; on a half-line
        # the origin is the finite end.
        if S.support == SUP_LOWER:
            r_excl_b = r_excl[0]
        elif S.support == SUP_UPPER:
            r_excl_b = r_excl[1]
        else:
            r_excl_b = r_excl[1] if side > 0 else r_excl[0]
        if _branch(S, si, side, coef, r_cap[1] if side > 0 else r_cap[0],
                   r_excl_b, violated[1] if side > 0 else violated[0], o, L, segs,
                   &unconverged) != 0:
            return -1
        if unconverged:
            r.unresolved += 1
        _sort_segments(segs.p, segs.n)
        # Merge touching segments, then map r to z.
        m = 0
        for i in range(<int> segs.n):
            g = segs.p[i]
            if m and g.z_lo <= segs.p[m - 1].z_hi * (1 + 1e-12) + 1e-300:
                if g.z_hi > segs.p[m - 1].z_hi:
                    segs.p[m - 1].z_hi = g.z_hi
                if g.h_min < segs.p[m - 1].h_min:
                    segs.p[m - 1].h_min = g.h_min
                    segs.p[m - 1].z_min = g.z_min
                segs.p[m - 1].to_inf = segs.p[m - 1].to_inf or g.to_inf
                segs.p[m - 1].touches_end = segs.p[m - 1].touches_end or g.touches_end
            else:
                segs.p[m] = g
                m += 1
        for i in range(m):
            g = segs.p[i]
            if side < 0:
                v = g.z_lo
                g.z_lo = -g.z_hi
                g.z_hi = -v
                g.z_min = -g.z_min
            v = g.z_hi if side > 0 else g.z_lo
            if S.support == SUP_BOUNDED and fabs(v) >= 1.0 - 1e-15:
                g.touches_end = 1
            if _spush(all_segs, g) != 0:
                return -1
    _sort_segments(all_segs.p, all_segs.n)
    # Join across the interior origin (both branches stop at +-2**jmin).
    m = 0
    width = 4.0 * ldexp(1.0, o.jmin if o.jmin < 0 else -1)
    for i in range(<int> all_segs.n):
        g = all_segs.p[i]
        if (m and S.nside == 2 and all_segs.p[m - 1].z_hi <= 0.0 <= g.z_lo
                and g.z_lo - all_segs.p[m - 1].z_hi <= width):
            if g.h_min < all_segs.p[m - 1].h_min:
                all_segs.p[m - 1].h_min = g.h_min
                all_segs.p[m - 1].z_min = g.z_min
            all_segs.p[m - 1].z_hi = g.z_hi
            all_segs.p[m - 1].to_inf = all_segs.p[m - 1].to_inf or g.to_inf
            all_segs.p[m - 1].touches_end = (all_segs.p[m - 1].touches_end
                                             or g.touches_end)
        else:
            all_segs.p[m] = g
            m += 1
    all_segs.n = m
    r.unresolved += L.unresolved
    r.panels = S.nP

    # --- report segments, minima and cuts ------------------------------------------
    r.n_seg = m
    r.segments = <Segment*> malloc(sizeof(Segment) * (m + 1))
    r.minima = <Minimum*> malloc(sizeof(Minimum) * (m + 1))
    order = <int*> malloc(sizeof(int) * (m + 1))
    if r.segments == NULL or r.minima == NULL or order == NULL:
        free(order)
        return -1
    for i in range(m):
        g = all_segs.p[i]
        r.segments[i] = g
        r.minima[i].z = g.z_min
        r.minima[i].x = t.c + h * g.z_min if isfinite(g.z_min) else g.z_min
        r.minima[i].value = g.h_min
        r.minima[i].kind = (MIN_TAIL if g.to_inf else
                            (MIN_ENDPOINT if g.touches_end else MIN_INTERIOR))
        any_violation = True
        # Lowest minima first for the cut budget.
        j = i - 1
        while j >= 0 and all_segs.p[order[j]].h_min > g.h_min:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = i
    r.n_min = m
    # Cut points: minimum plus n_extra Chebyshev (finite) or geometric (unbounded).
    k = m if m < o.max_cuts else o.max_cuts
    npts = k * (o.n_extra + 1)
    zc = <double*> malloc(sizeof(double) * (npts + 1))
    if zc == NULL:
        free(order)
        return -1
    npts = 0
    for q in range(k):
        g = all_segs.p[order[q]]
        zc[npts] = g.z_min
        npts += 1
        if isfinite(g.z_lo) and isfinite(g.z_hi):
            for j in range(o.n_extra):
                zc[npts] = (0.5 * (g.z_lo + g.z_hi) + 0.5 * (g.z_hi - g.z_lo)
                            * cos(M_PI * (j + 0.5) / o.n_extra))
                npts += 1
        else:
            r0 = fabs(g.z_hi if not isfinite(g.z_lo) else g.z_lo)
            if r0 < 1e-3:
                r0 = 1e-3
            f = 4.0
            base = 1 if isfinite(g.z_lo) else -1
            for _ in range(o.n_extra):
                zc[npts] = base * r0 * f
                f *= 4.0
                npts += 1
    free(order)
    # Points must lie inside the support (finite canonical endpoints are 0, +-1).
    rr = <double*> malloc(sizeof(double) * (npts * A + 1))
    nn = <double*> malloc(sizeof(double) * (npts * A + 1))
    ra = <double*> malloc(sizeof(double) * (2 * npts + 1))
    rm = <double*> malloc(sizeof(double) * (2 * npts + 1))
    rk = <int*> malloc(sizeof(int) * (npts + 1))
    r.cut_points = <double*> malloc(sizeof(double) * (npts + 1))
    r.cut_rows = <double*> malloc(sizeof(double) * (npts * d + 1))
    rc = 0
    if (rr == NULL or nn == NULL or ra == NULL or rm == NULL or rk == NULL
            or r.cut_points == NULL or r.cut_rows == NULL):
        rc = -1
    elif npts and A:
        rc = curvature_rows(t, npts, zc, True, rr, nn, rk, ra, rm)
    if rc == 0:
        q = 0
        for i in range(npts):
            big = 0.0
            for j in range(d):
                v = 0.0
                for k in range(A):
                    v += rr[i * A + k] * t.atom_row[k * d + j]
                r.cut_rows[q * d + j] = v
                if fabs(v) > big:
                    big = fabs(v)
            if not (big > 0.0) or not isfinite(big):
                continue
            for j in range(d):
                r.cut_rows[q * d + j] /= big
            r.cut_points[q] = t.c + h * zc[i]
            q += 1
        r.n_cuts = q
    free(zc)
    free(rr)
    free(nn)
    free(ra)
    free(rm)
    free(rk)
    if rc != 0:
        return -1

    # --- verdict -----------------------------------------------------------------
    r.a = t.lower if not isfinite(r_cap[0]) or tail_lo == NULL else t.c - h * r_cap[0]
    r.b = t.upper if not isfinite(r_cap[1]) or tail_hi == NULL else t.c + h * r_cap[1]
    if any_violation:
        r.status = SEARCH_VIOLATED
    elif unverified or r.unresolved:
        r.status = SEARCH_UNVERIFIED
    else:
        r.status = SEARCH_VERIFIED
    return 0


cdef int _reset_cache(CurvSearch* S) noexcept nogil:
    """Drop every split panel and every cached row, keeping the base panels.

    The base panels (depth 0) are compacted in order and the branch lists
    remapped; rows are recomputed on demand.  Returns ``-1`` on allocation
    failure.
    """
    cdef int* where = <int*> malloc(sizeof(int) * (S.nP + 1))
    cdef int i, m = 0, si, k
    if where == NULL:
        return -1
    for i in range(S.nP):
        where[i] = -1
        if S.P[i].depth == 0:
            S.P[m] = S.P[i]
            S.P[m].child[0] = -1
            S.P[m].child[1] = -1
            S.P[m].rowoff = -1
            where[i] = m
            m += 1
    S.nP = m
    for si in range(2):
        for k in range(S.n_inner_fixed[si]):
            S.inner_fixed[si][k] = where[S.inner_fixed[si][k]]
        for k in range(S.n_inner_end[si]):
            S.inner_end[si][k] = where[S.inner_end[si][k]]
        for k in range(S.n_outer[si]):
            S.outer[si][k] = where[S.outer[si][k]]
    free(where)
    free(S.R)
    free(S.NR)
    free(S.Z)
    free(S.DZ)
    S.R = NULL
    S.NR = NULL
    S.Z = NULL
    S.DZ = NULL
    S.nrow = 0
    S.caprow = 0
    S.resets += 1
    return 0


cdef int separate_c(
    CurvSearch* S,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
) noexcept nogil:
    """Curvature separation for ``theta``; ``0`` or ``-1`` on allocation failure.

    The caller releases ``r`` with ``search_free``.  The options' degree and
    ``jmin`` must be those the state ``S`` was built with.
    """
    cdef const BasisTable* t = S.t
    cdef Py_ssize_t tw = 3
    cdef int out
    cdef double* coef
    cdef double* work
    cdef double* prep
    cdef double* rest
    cdef Iv* ivwork
    cdef Leafs L
    cdef SArr segs, all_segs
    r.status = SEARCH_UNVERIFIED
    r.n_bound = 0
    r.bound_cols = NULL
    r.n_seg = 0
    r.segments = NULL
    r.n_min = 0
    r.minima = NULL
    r.n_tails = 0
    r.n_cuts = 0
    r.cut_points = NULL
    r.cut_rows = NULL
    r.panels = 0
    r.unresolved = 0
    if S.nP > o.max_cached_panels and _reset_cache(S) != 0:
        return -1
    S.np_start = S.nP
    if tail_lo != NULL and tail_work_size(tail_lo) > tw:
        tw = tail_work_size(tail_lo)
    if tail_hi != NULL and tail_work_size(tail_hi) > tw:
        tw = tail_work_size(tail_hi)
    coef = <double*> malloc(sizeof(double) * (S.A + 1))
    work = <double*> malloc(sizeof(double) * (tw + 1))
    prep = <double*> malloc(sizeof(double) * (prep_size(t) + 1))
    rest = <double*> malloc(sizeof(double) * (t.d + 1))
    ivwork = <Iv*> malloc(sizeof(Iv) * (5 * t.nf + 1))
    L.idx.p = NULL
    L.idx.n = 0
    L.idx.cap = 0
    L.H = NULL
    L.c = NULL
    L.cap = 0
    L.unresolved = 0
    segs.p = NULL
    segs.n = 0
    segs.cap = 0
    all_segs.p = NULL
    all_segs.n = 0
    all_segs.cap = 0
    out = -1
    if (coef != NULL and work != NULL and prep != NULL
            and rest != NULL and ivwork != NULL):
        out = _search_body(S, theta, tail_lo, tail_hi, o, r, coef, work, prep,
                           rest, ivwork, &L, &segs, &all_segs)
    free(coef)
    free(work)
    free(prep)
    free(rest)
    free(ivwork)
    free(L.idx.p)
    free(L.H)
    free(L.c)
    free(segs.p)
    free(all_segs.p)
    if out != 0:
        search_free(r)
    return out


# --------------------------------------------------------------------------
# per-basis state
# --------------------------------------------------------------------------

cdef class CurvatureSearch:
    """Fixed panel hierarchy and cached rows of one compiled basis.

    Built once per basis (or face); every search for a new ``theta`` reuses
    the rows already computed and adds rows only for newly split panels.

    Parameters
    ----------
    kernel : BasisKernel
        Compiled column table of the basis.
    degree : int
        Chebyshev degree per panel (``degree + 1`` nodes, at most 63).
    jmin : int
        Inner panels reach ``r = 2**jmin`` (full line and bounded supports).
    """

    def __cinit__(self, BasisKernel kernel, int degree, int jmin):
        cdef CurvSearch* S = &self.s
        cdef const BasisTable* t = &kernel.t
        cdef int n = degree + 1, j, k, i, side, si, jtop, ip
        cdef double e[_MAXN]
        cdef double dd[_MAXN]
        if not 2 <= degree < _MAXN:
            raise ValueError("degree must be in 2..63")
        self._keep = kernel
        S.t = t
        S.A = t.natom
        S.n = n
        S.np = (n + 3) // 4 * 4
        S.P = NULL
        S.nP = 0
        S.capP = 0
        S.R = NULL
        S.NR = NULL
        S.Z = NULL
        S.DZ = NULL
        S.nrow = 0
        S.caprow = 0
        S.kernel_points = 0
        S.panel_evals = 0
        S.direct_panels = 0
        S.resets = 0
        for i in range(2):
            S.inner_end[i] = NULL
            S.n_inner_end[i] = 0
            S.cap_inner_end[i] = 0
            S.outer[i] = NULL
            S.n_outer[i] = 0
            S.cap_outer[i] = 0
            S.n_inner_fixed[i] = 0
        if t.has_l and t.has_u:
            S.support = SUP_FULL
        elif t.has_u:
            S.support = SUP_LOWER
        elif t.has_l:
            S.support = SUP_UPPER
        else:
            S.support = SUP_BOUNDED
        S.nodes = <double*> malloc(sizeof(double) * n)
        S.Tm = <double*> malloc(sizeof(double) * n * n)
        S.Dv = <double*> malloc(sizeof(double) * n * n)
        if S.nodes == NULL or S.Tm == NULL or S.Dv == NULL:
            raise MemoryError("curvature search tables")
        for k in range(n):
            S.nodes[k] = cos(M_PI * (k + 0.5) / n)
        for j in range(n):
            for k in range(n):
                S.Tm[j * n + k] = ((1.0 if j == 0 else 2.0) / n
                                   * cos(j * M_PI * (k + 0.5) / n))
        for j in range(n):
            for k in range(n):
                e[k] = 0.0
            e[j] = 1.0
            _chebder(e, n, dd)
            for k in range(n):
                S.Dv[k * n + j] = _cheb(dd, n - 1, S.nodes[k])
        S.pmax = 1.0
        for i in range(S.A):
            if t.kind[t.atom_fn[i]] == 2 and t.atom_par[i] > S.pmax:
                S.pmax = t.atom_par[i]
        if S.support == SUP_FULL or S.support == SUP_BOUNDED:
            S.nside = 2
            S.sides[0] = 1
            S.sides[1] = -1
        else:
            S.nside = 1
            S.sides[0] = 1 if S.support == SUP_LOWER else -1
        for i in range(S.nside):
            side = S.sides[i]
            si = 0 if side > 0 else 1
            if S.support == SUP_FULL or S.support == SUP_BOUNDED:
                jtop = -2 if S.support == SUP_BOUNDED else -1
                j = jmin
                while j <= jtop and S.n_inner_fixed[si] < 64:
                    ip = _new_panel(S, side, K_R, ldexp(1.0, j), ldexp(1.0, j + 1),
                                    0, j)
                    if ip < 0:
                        raise MemoryError("curvature search panels")
                    S.inner_fixed[si][S.n_inner_fixed[si]] = ip
                    S.n_inner_fixed[si] += 1
                    j += 1

    def __dealloc__(self):
        cdef CurvSearch* S = &self.s
        cdef int i
        free(S.nodes)
        free(S.Tm)
        free(S.Dv)
        free(S.P)
        free(S.R)
        free(S.NR)
        free(S.Z)
        free(S.DZ)
        for i in range(2):
            free(S.inner_end[i])
            free(S.outer[i])

    @property
    def statistics(self):
        """``(kernel_points, panel_evals, panels)`` accumulated over searches."""
        return int(self.s.kernel_points), int(self.s.panel_evals), int(self.s.nP)

    def panel_values(self, int index, const double[::1] theta):
        """Node locations ``z`` and cached-row ``H`` of panel ``index`` (tests)."""
        cdef CurvSearch* S = &self.s
        cdef int k, i
        cdef double v
        cdef double H[_MAXNP]
        cdef double c[_MAXN]
        cdef double tl, noise
        if not 0 <= index < S.nP or theta.shape[0] != S.t.d:
            raise ValueError("no such panel or theta of the wrong length")
        coef = np.zeros(max(S.A, 1))
        cdef double[::1] cv = coef
        for i in range(S.A):
            v = 0.0
            for k in range(S.t.d):
                v += S.t.atom_row[i * S.t.d + k] * theta[k]
            cv[i] = v
        if _ensure_rows(S, &index, 1) != 0:
            raise MemoryError("curvature search rows")
        if _eval_panel(S, &S.P[index], &cv[0], H, c, &tl, &noise) != 0:
            raise MemoryError("curvature search rows")
        off = S.P[index].rowoff
        z = np.array([S.Z[off + k] for k in range(S.n)])
        h = np.array([H[k] for k in range(S.n)])
        rows = np.array([[S.R[off + i * S.np + k] for i in range(S.A)]
                         for k in range(S.n)])
        return z, h, tl, noise, rows

    @property
    def resets(self):
        """Times the cached hierarchy was dropped for its size (tests)."""
        return int(self.s.resets)

    @property
    def direct_panels(self):
        """Panel evaluations that fell back to the direct kernel (tests)."""
        return int(self.s.direct_panels)

    @property
    def panels(self):
        """Every panel as ``(side, kind, depth, r_lo, r_hi, leaf)`` (tests)."""
        cdef int i
        cdef Panel* p
        out = []
        for i in range(self.s.nP):
            p = &self.s.P[i]
            out.append((p.side, p.kind, p.depth, _edge_r(p, 0), _edge_r(p, 1),
                        p.child[0] < 0, p.rowoff))
        return out


cdef void search_options(object options, SearchOptions* o):
    """Copy a ``_CurvatureSearchOptions`` record into ``o``."""
    o.tau = options.tau
    o.degree = options.degree
    o.res_frac = options.res_frac
    o.max_depth = options.max_depth
    o.jmin = options.jmin
    o.jmax = options.jmax
    o.stop_shells = options.stop_shells
    o.jmin_end = options.jmin_end
    o.end_shells = options.end_shells
    o.nsamp_factor = options.nsamp_factor
    o.n_extra = options.n_extra
    o.max_cuts = options.max_cuts
    o.max_new_panels = options.max_new_panels
    o.max_cached_panels = options.max_cached_panels
    o.zeta_cap = options.zeta_cap
    o.fallback_zeta = options.fallback_zeta


def separate(CurvatureSearch search, const double[::1] theta, tail_lo, tail_hi,
             options):
    """Python entry of ``separate_c``.

    Returns ``(status, tau, (a, b), bound_cols, segments, minima, tails,
    cut_points, cut_rows, panels, unresolved)``; ``segments`` rows are ``(z_lo,
    z_hi, z_min, h_min, to_inf, touches_end)``, ``minima`` rows ``(x, z, value,
    kind)``, ``tails`` items ``(side, status, cutoff, zeta, lead_value,
    lead_power, lead_rate, lead_kappa, lead_const, margin, cut_atom)``.
    """
    cdef CurvSearch* S = &search.s
    cdef SearchOptions o
    cdef SearchResult r
    cdef const TailTable* tl = NULL
    cdef const TailTable* tu = NULL
    cdef const double* th
    cdef int i, j, rc
    cdef int d = S.t.d
    if theta.shape[0] != d:
        raise ValueError("theta has the wrong length")
    if tail_lo is not None:
        tl = &(<TailKernel> tail_lo).t
    if tail_hi is not None:
        tu = &(<TailKernel> tail_hi).t
    search_options(options, &o)
    if o.degree + 1 != S.n:
        raise ValueError("options.degree differs from the search state's")
    if o.n_extra < 1:
        raise ValueError("n_extra must be at least 1")
    th = &theta[0] if d else NULL
    with nogil:
        rc = separate_c(S, th, tl, tu, &o, &r)
    if rc != 0:
        raise MemoryError("curvature search workspace")
    try:
        status = ("verified", "violated", "unverified")[r.status]
        bound = [r.bound_cols[i] for i in range(r.n_bound)]
        segments = [
            (r.segments[i].z_lo, r.segments[i].z_hi, r.segments[i].z_min,
             r.segments[i].h_min, r.segments[i].to_inf, r.segments[i].touches_end)
            for i in range(r.n_seg)
        ]
        minima = [
            (r.minima[i].x, r.minima[i].z, r.minima[i].value,
             ("interior", "endpoint", "tail", "unresolved")[r.minima[i].kind])
            for i in range(r.n_min)
        ]
        tails = []
        for i in range(r.n_tails):
            res = r.tails[i]
            tails.append((
                ("lower", "upper")[r.tail_side[i]],
                ("verified", "violated", "unverified")[res.status],
                res.cutoff if res.has_cutoff else None,
                res.zeta if res.has_cutoff else None,
                None if isnan(res.lead_value) else res.lead_value,
                res.lead_power if res.has_lead else None,
                res.lead_rate if res.has_lead else None,
                res.lead_kappa if res.has_lead else None,
                res.lead_const if res.has_lead else None,
                res.margin if res.has_margin else None,
                res.cut_atom,
            ))
        points = np.array([r.cut_points[i] for i in range(r.n_cuts)], dtype=np.float64)
        rows = np.array(
            [[r.cut_rows[i * d + j] for j in range(d)] for i in range(r.n_cuts)],
            dtype=np.float64,
        ).reshape(r.n_cuts, d)
        return (status, r.tau, (r.a, r.b), bound, segments, minima, tails, points,
                rows, r.panels, r.unresolved)
    finally:
        search_free(&r)
