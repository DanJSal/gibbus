# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled G3 curvature separation: where ``q''`` may fall below the threshold.

G1 owner replaced: the exact separator ``_fit/separation.py`` and the
post-solve ``_certify`` of ``_fit/conic_newton.py``.

``separate_c`` is the compiled body of ``curvature_search._separate_curvature``
(see that module for the method): compact domain from tail verdicts and log
endpoint exclusions, asinh-spaced cells classified by second-order Taylor and
direct enclosures (discard, monotone, convex, affine, bisect), safeguarded
Newton/bisection on ``q'''`` in convex cells, deduplicated minima, the lowest
``max_cuts`` cut rows and the shape verdict.  ``valleys_c`` locates the local
minima of ``q''`` on the search grid (step-model anchors, G3-038).
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport INFINITY, NAN, asinh, fabs, isfinite, sinh
from libc.stdlib cimport free, malloc, realloc

from .._model._basis_kernels cimport (
    TAIL_NONNEGATIVE,
    TAIL_VIOLATED,
    BasisKernel,
    BasisTable,
    Iv,
    PotentialPrep,
    TailKernel,
    TailTable,
    endpoint_exclusion,
    eval_point,
    npmax,
    npmin,
    potential_enclosure_cell,
    potential_point,
    prep_size,
    prepare_potential,
    tail_verdict,
    tail_work_size,
)

cnp.import_array()

cdef double _U = 1.1102230246251565e-16


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


cdef inline int _push2(DArr* a, double u, double v) noexcept nogil:
    if _reserve(a, a.n + 2) != 0:
        return -1
    a.p[a.n] = u
    a.p[a.n + 1] = v
    a.n += 2
    return 0


cdef inline void _release(DArr* a) noexcept nogil:
    free(a.p)
    a.p = NULL
    a.n = 0
    a.cap = 0


cdef inline void _zero_arr(DArr* a) noexcept nogil:
    a.p = NULL
    a.n = 0
    a.cap = 0


cdef int _breakpoints(
    const BasisTable* t, double a, double b, int n, double* pts
) noexcept nogil:
    """Asinh-spaced points plus centre and gate locations; returns the count.

    ``pts`` holds ``n + 4`` doubles.
    """
    cdef double va = asinh((a - t.c) / t.h)
    cdef double vb = asinh((b - t.c) / t.h)
    cdef double step = (vb - va) / n
    cdef int i, j, m = 0
    cdef double v
    cdef double extra[3]
    for i in range(n + 1):
        pts[m] = npmin(npmax(t.c + t.h * sinh(i * step + va if i < n else vb), a), b)
        m += 1
    extra[0] = t.c
    extra[1] = t.xl if t.has_l else NAN
    extra[2] = t.xu if t.has_u else NAN
    for i in range(3):
        if isfinite(extra[i]) and a < extra[i] and extra[i] < b:
            pts[m] = extra[i]
            m += 1
    # sort + unique
    for i in range(1, m):
        v = pts[i]
        j = i - 1
        while j >= 0 and pts[j] > v:
            pts[j + 1] = pts[j]
            j -= 1
        pts[j + 1] = v
    j = 0
    for i in range(m):
        if j == 0 or pts[i] != pts[j - 1]:
            pts[j] = pts[i]
            j += 1
    pts[0] = a
    pts[j - 1] = b
    return j


cdef inline double _taylor_lower(
    double a_lo, double b_lo, double b_hi, double c_lo, double r
) noexcept nogil:
    """Lower bound of ``A + B d + C/2 d**2`` over ``|d| <= r``."""
    cdef double cands[5]
    cdef double best = INFINITY
    cdef double dd, lin, val, mag
    cdef int i
    cands[0] = -1.0
    cands[1] = 0.0
    cands[2] = 1.0
    cands[3] = 0.0
    cands[4] = 0.0
    if c_lo > 0.0:
        cands[3] = npmin(npmax(-b_lo / (c_lo * r), 0.0), 1.0)
        cands[4] = npmin(npmax(-b_hi / (c_lo * r), -1.0), 0.0)
    mag = fabs(a_lo) + npmax(fabs(b_lo), fabs(b_hi)) * r + 0.5 * fabs(c_lo) * r * r
    for i in range(5):
        dd = cands[i] * r
        lin = b_lo * dd if dd >= 0.0 else b_hi * dd
        val = a_lo + lin + 0.5 * c_lo * dd * dd
        best = npmin(best, val - 16.0 * _U * mag)
    return best


cdef void _refine(
    const BasisTable* t,
    const double* theta,
    double lo,
    double hi,
    const SearchOptions* o,
    double* fwork,
    double* colwork,
    double* xout,
) noexcept nogil:
    """Safeguarded Newton/bisection for the minimum of ``q''`` on ``[lo, hi]``.

    Requires ``q'''`` nondecreasing on the cell.
    """
    cdef double h = t.h
    cdef double a = lo
    cdef double b = hi
    cdef double q[5]
    cdef double fa, fb, x, f, fp, aa, bb, newton, nx
    cdef bint done, ok, conv
    cdef int _it
    potential_point(t, theta, a, 3, fwork, colwork, q)
    fa = q[3]
    potential_point(t, theta, b, 3, fwork, colwork, q)
    fb = q[3]
    x = 0.5 * (a + b)
    if fa >= 0.0:
        x = a
    if fb <= 0.0:
        x = a if fa >= 0.0 else b
    done = fa >= 0.0 or fb <= 0.0
    for _it in range(o.newton_max):
        if done:
            break
        potential_point(t, theta, x, 4, fwork, colwork, q)
        f = q[3]
        fp = q[4]
        aa = x if f < 0.0 else a
        bb = x if f > 0.0 else b
        newton = x - f / fp
        ok = fp > 0.0 and newton > aa and newton < bb
        nx = newton if ok else 0.5 * (aa + bb)
        conv = (
            f == 0.0 or bb - aa <= o.newton_tol * h or fabs(nx - x) <= o.newton_tol * h
        )
        a = aa
        b = bb
        if f != 0.0:
            x = nx
        if conv:
            done = True
    xout[0] = x


cdef int valleys_c(
    const BasisTable* t,
    const double* theta,
    double a,
    double b,
    const SearchOptions* o,
    double* out,
    int* n,
) noexcept nogil:
    """Local minima of ``q''`` on ``[a, b]`` (``curvature_search._valleys``).

    ``out`` holds ``initial_cells + 4`` doubles.  Returns ``-1`` on
    allocation failure.
    """
    cdef int m, i, j, k = 0
    cdef double q[5]
    cdef double v
    cdef double* pts = <double*> malloc((o.initial_cells + 5) * sizeof(double))
    cdef double* q3 = <double*> malloc((o.initial_cells + 5) * sizeof(double))
    cdef double* fwork = <double*> malloc((5 * t.nf + 1) * sizeof(double))
    cdef double* colwork = <double*> malloc((5 * t.d + 1) * sizeof(double))
    if pts == NULL or q3 == NULL or fwork == NULL or colwork == NULL:
        free(pts)
        free(q3)
        free(fwork)
        free(colwork)
        return -1
    m = _breakpoints(t, a, b, o.initial_cells, pts)
    j = 0
    for i in range(m):
        potential_point(t, theta, pts[i], 3, fwork, colwork, q)
        if isfinite(q[3]):
            pts[j] = pts[i]
            q3[j] = q[3]
            j += 1
    for i in range(j - 1):
        if q3[i] < 0.0 and q3[i + 1] > 0.0:
            _refine(t, theta, pts[i], pts[i + 1], o, fwork, colwork, &out[k])
            k += 1
    for i in range(1, k):
        v = out[i]
        j = i - 1
        while j >= 0 and out[j] > v:
            out[j + 1] = out[j]
            j -= 1
        out[j + 1] = v
    n[0] = k
    free(pts)
    free(q3)
    free(fwork)
    free(colwork)
    return 0


cdef void search_free(SearchResult* r) noexcept nogil:
    free(r.bound_cols)
    free(r.minima)
    free(r.cut_points)
    free(r.cut_rows)
    r.bound_cols = NULL
    r.minima = NULL
    r.cut_points = NULL
    r.cut_rows = NULL


cdef struct Sws:
    double* fwork
    double* colwork
    Iv* ivwork
    double* prep
    double* prep2
    double* rest
    double* tailwork
    double* pts
    DArr cells
    DArr nxt
    DArr minima
    DArr convex
    DArr unres
    DArr rec


cdef void _sws_free(Sws* w) noexcept nogil:
    free(w.fwork)
    free(w.colwork)
    free(w.ivwork)
    free(w.prep)
    free(w.prep2)
    free(w.rest)
    free(w.tailwork)
    free(w.pts)
    _release(&w.cells)
    _release(&w.nxt)
    _release(&w.minima)
    _release(&w.convex)
    _release(&w.unres)
    _release(&w.rec)


cdef int _search_body(
    const BasisTable* t,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
    Sws* w,
) noexcept nogil:
    cdef int d = t.d
    cdef double h = t.h
    cdef double c = t.c
    cdef double tau = o.rho / (h * h)
    cdef double ends[2]
    cdef bint real_end[2]
    cdef bint flag_violated = False, flag_inconclusive = False
    cdef const TailTable* tails[2]
    cdef double sign, width, a, b, lo, hi, mid, rr, lower, cmag, q3_lo, q3_hi
    cdef double f_lo, f_hi, x, floor_, prev
    cdef Iv pe[4]
    cdef Iv ce[5]
    cdef Iv enc[3]
    cdef double q[5]
    cdef int side, i, j, k, m, nb, take, kind, attempt
    cdef Py_ssize_t processed = 0, n, ncut
    cdef PotentialPrep p
    cdef bint discard, increasing, decreasing, convex, flat
    cdef bint split, narrow, keep, nonzero
    cdef int* order

    r.tau = tau
    r.n_tails = 0
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

    # --- compact domain -----------------------------------------------------
    tails[0] = tail_lo
    tails[1] = tail_hi
    for side in range(2):
        sign = 1.0 if side else -1.0
        if tails[side] == NULL:
            width = endpoint_exclusion(t, theta, side, tau, w.rest, w.prep2, w.ivwork)
            ends[side] = (t.upper if side else t.lower) - sign * width
            real_end[side] = width == 0.0
            continue
        r.tails[r.n_tails] = tail_verdict(tails[side], theta, o.zeta_cap, 0.0, False,
                                          w.tailwork)
        r.tail_side[r.n_tails] = side
        real_end[side] = r.tails[r.n_tails].status != TAIL_NONNEGATIVE
        if r.tails[r.n_tails].status == TAIL_NONNEGATIVE:
            ends[side] = r.tails[r.n_tails].cutoff
        else:
            if r.tails[r.n_tails].status == TAIL_VIOLATED:
                flag_violated = True
            else:
                flag_inconclusive = True
            if r.tails[r.n_tails].has_lead and r.tails[r.n_tails].lead_value > 0.0:
                # A positive leader too weak to dominate before the cap: search
                # the whole range the tail analysis could not cover (G3-037).
                ends[side] = c + sign * o.zeta_cap * h
            else:
                ends[side] = c + sign * o.fallback_zeta * h
        r.n_tails += 1
    a = ends[0]
    b = ends[1]
    r.a = a
    r.b = b

    # --- cell search ----------------------------------------------------------
    prepare_potential(t, theta, w.prep, &p)
    m = _breakpoints(t, a, b, o.initial_cells, w.pts)
    w.cells.n = 0
    for i in range(m - 1):
        if _push2(&w.cells, w.pts[i], w.pts[i + 1]) != 0:
            return -1
    while w.cells.n:
        n = w.cells.n // 2
        if processed + n > o.max_cells:
            take = o.max_cells - processed
            if take < 0:
                take = 0
            for i in range(take, n):
                if _push2(&w.unres, w.cells.p[2 * i], w.cells.p[2 * i + 1]) != 0:
                    return -1
            n = take
            w.cells.n = 2 * take
            if n == 0:
                break
        processed += n
        w.nxt.n = 0
        # endpoint candidates are appended after the affine ones of a pass
        for i in range(n):
            lo = w.cells.p[2 * i]
            hi = w.cells.p[2 * i + 1]
            mid = 0.5 * (lo + hi)
            rr = 0.5 * (hi - lo)
            potential_enclosure_cell(t, &p, mid, mid, 3, w.ivwork, pe)
            potential_enclosure_cell(t, &p, lo, hi, 4, w.ivwork, ce)
            lower = npmax(
                ce[2].lo, _taylor_lower(pe[2].lo, pe[3].lo, pe[3].hi, ce[4].lo, rr)
            )
            cmag = npmax(fabs(ce[4].lo), fabs(ce[4].hi))
            q3_lo = npmax(ce[3].lo, pe[3].lo - cmag * rr * (1.0 + 4.0 * _U))
            q3_hi = npmin(ce[3].hi, pe[3].hi + cmag * rr * (1.0 + 4.0 * _U))
            discard = lower > tau
            increasing = not discard and q3_lo > 0.0
            decreasing = not discard and not increasing and q3_hi < 0.0
            convex = (
                not discard and not increasing and not decreasing and ce[4].lo >= 0.0
            )
            # Affine within negligible error: the minimum is at a cell end.
            flat = (not discard and not increasing and not decreasing and not convex
                    and 0.5 * cmag * rr * rr <= 1e-9 * tau)
            split = not (discard or increasing or decreasing or convex or flat)
            if flat:
                potential_point(t, theta, lo, 2, w.fwork, w.colwork, q)
                f_lo = q[2]
                potential_point(t, theta, hi, 2, w.fwork, w.colwork, q)
                f_hi = q[2]
                if _push2(
                    &w.minima, hi if f_hi < f_lo else lo, <double> MIN_AFFINE
                ) != 0:
                    return -1
            if convex:
                if _push2(&w.convex, lo, hi) != 0:
                    return -1
            if split:
                narrow = hi - lo <= o.min_width * h
                if narrow:
                    if _push2(&w.unres, lo, hi) != 0:
                        return -1
                else:
                    if _push2(&w.nxt, lo, mid) != 0 or _push2(&w.nxt, mid, hi) != 0:
                        return -1
        # Monotone cells: only a real support endpoint becomes a candidate
        # (lower ends of increasing cells, then upper ends of decreasing ones).
        for i in range(n):
            lo = w.cells.p[2 * i]
            hi = w.cells.p[2 * i + 1]
            if real_end[0] and lo == a:
                if _endpoint_monotone(t, &p, lo, hi, tau, w.ivwork, 1):
                    if _push2(&w.minima, lo, <double> MIN_ENDPOINT) != 0:
                        return -1
        for i in range(n):
            lo = w.cells.p[2 * i]
            hi = w.cells.p[2 * i + 1]
            if real_end[1] and hi == b:
                if _endpoint_monotone(t, &p, lo, hi, tau, w.ivwork, -1):
                    if _push2(&w.minima, hi, <double> MIN_ENDPOINT) != 0:
                        return -1
        # next pass: sorted by lo
        k = <int> (w.nxt.n // 2)
        for i in range(1, k):
            lo = w.nxt.p[2 * i]
            hi = w.nxt.p[2 * i + 1]
            j = i - 1
            while j >= 0 and w.nxt.p[2 * j] > lo:
                w.nxt.p[2 * j + 2] = w.nxt.p[2 * j]
                w.nxt.p[2 * j + 3] = w.nxt.p[2 * j + 1]
                j -= 1
            w.nxt.p[2 * j + 2] = lo
            w.nxt.p[2 * j + 3] = hi
        if _reserve(&w.cells, w.nxt.n) != 0:
            return -1
        for i in range(<int> w.nxt.n):
            w.cells.p[i] = w.nxt.p[i]
        w.cells.n = w.nxt.n

    for i in range(<int> (w.convex.n // 2)):
        lo = w.convex.p[2 * i]
        hi = w.convex.p[2 * i + 1]
        _refine(t, theta, lo, hi, o, w.fwork, w.colwork, &x)
        potential_point(t, theta, x, 3, w.fwork, w.colwork, q)
        # A refinement that stops at an inner cell boundary with q'' still
        # decreasing into the neighbour is not a local minimum (G3-038).
        if (x == hi and hi < b and q[3] < 0.0) or (x == lo and lo > a and q[3] > 0.0):
            continue
        kind = MIN_INTERIOR
        if (
            (x == lo and lo == a and real_end[0])
            or (x == hi and hi == b and real_end[1])
        ):
            kind = MIN_ENDPOINT
        if _push2(&w.minima, x, <double> kind) != 0:
            return -1
    for i in range(<int> (w.unres.n // 2)):
        if _push2(&w.minima, 0.5 * (w.unres.p[2 * i] + w.unres.p[2 * i + 1]),
                  <double> MIN_UNRESOLVED) != 0:
            return -1
    r.cells = <int> processed
    r.unresolved = <int> (w.unres.n // 2)

    # --- evaluate, deduplicate, select cuts ----------------------------------
    m = <int> (w.minima.n // 2)
    # stable insertion sort by x
    for i in range(1, m):
        x = w.minima.p[2 * i]
        lo = w.minima.p[2 * i + 1]
        j = i - 1
        while j >= 0 and w.minima.p[2 * j] > x:
            w.minima.p[2 * j + 2] = w.minima.p[2 * j]
            w.minima.p[2 * j + 3] = w.minima.p[2 * j + 1]
            j -= 1
        w.minima.p[2 * j + 2] = x
        w.minima.p[2 * j + 3] = lo
    r.minima = <Minimum*> malloc((m + 1) * sizeof(Minimum))
    if r.minima == NULL:
        return -1
    k = 0
    for i in range(m):
        x = w.minima.p[2 * i]
        if i > 0:
            prev = w.minima.p[2 * i - 2]
            if not (x - prev > 4.0 * _U * npmax(fabs(x), h)):
                continue
        potential_enclosure_cell(t, &p, x, x, 2, w.ivwork, enc)
        potential_point(t, theta, x, 2, w.fwork, w.colwork, q)
        r.minima[k].x = x
        r.minima[k].value = q[2]
        # A tight Horner enclosure can be more accurate than the separately
        # summed point value. Report its midpoint when that dot product has
        # rounded outside the enclosure, including for cut ordering below.
        if ((q[2] < enc[2].lo or q[2] > enc[2].hi)
                and isfinite(enc[2].lo) and isfinite(enc[2].hi)):
            r.minima[k].value = 0.5 * enc[2].lo + 0.5 * enc[2].hi
        r.minima[k].lower = enc[2].lo
        r.minima[k].upper = enc[2].hi
        r.minima[k].kind = <int> w.minima.p[2 * i + 1]
        k += 1
    r.n_min = k
    # cut candidates: below tau with a nonzero row, lowest values first
    order = <int*> malloc((k + 1) * sizeof(int))
    if order == NULL:
        return -1
    ncut = 0
    for i in range(k):
        if not (r.minima[i].lower < tau):
            continue
        eval_point(t, r.minima[i].x, 2, w.fwork, w.colwork)
        nonzero = False
        for j in range(d):
            if w.colwork[2 * d + j] != 0.0:
                nonzero = True
                break
        if not nonzero:
            continue
        # stable insertion by value
        j = <int> ncut - 1
        while j >= 0 and r.minima[order[j]].value > r.minima[i].value:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = i
        ncut += 1
    if ncut > o.max_cuts:
        ncut = o.max_cuts
    r.n_cuts = <int> ncut
    r.cut_points = <double*> malloc((ncut + 1) * sizeof(double))
    r.cut_rows = <double*> malloc((ncut * d + 1) * sizeof(double))
    if r.cut_points == NULL or r.cut_rows == NULL:
        free(order)
        return -1
    for i in range(<int> ncut):
        r.cut_points[i] = r.minima[order[i]].x
    free(order)
    for i in range(1, <int> ncut):
        x = r.cut_points[i]
        j = i - 1
        while j >= 0 and r.cut_points[j] > x:
            r.cut_points[j + 1] = r.cut_points[j]
            j -= 1
        r.cut_points[j + 1] = x
    for i in range(<int> ncut):
        eval_point(t, r.cut_points[i], 2, w.fwork, w.colwork)
        keep = True
        for j in range(d):
            if not isfinite(w.colwork[2 * d + j]):
                keep = False
        if not keep:
            # A finite endpoint may have a divergent log derivative.  Far
            # out on an infinite end, an *inactive* exponential column can
            # instead overflow even though the candidate's polynomial
            # curvature is finite.  Never substitute +/-infinity for a point
            # cut: move the point towards the fitting centre until every
            # column is representable.  Each resulting finite point row is
            # still a necessary global convexity constraint.  This does not
            # change the search verdict or certify the skipped interval.
            x = r.cut_points[i]
            if isfinite(t.lower) and x <= t.lower:
                x = t.lower + 1e-9 * h
            elif isfinite(t.upper) and x >= t.upper:
                x = t.upper - 1e-9 * h
            for attempt in range(80):
                eval_point(t, x, 2, w.fwork, w.colwork)
                keep = True
                for j in range(d):
                    if not isfinite(w.colwork[2 * d + j]):
                        keep = False
                        break
                if keep:
                    break
                x = 0.5 * (x + t.c)
            if not keep:
                # Even the centre cannot be evaluated.  Fail closed rather
                # than emitting a nonfinite row or an invalid certificate.
                return -1
            r.cut_points[i] = x
        for j in range(d):
            r.cut_rows[i * d + j] = w.colwork[2 * d + j]

    floor_ = -o.feasibility_tol / (h * h)
    r.status = SEARCH_NONNEGATIVE
    for i in range(k):
        if r.minima[i].upper < floor_:
            flag_violated = True
    if flag_violated:
        r.status = SEARCH_VIOLATED
    else:
        for i in range(k):
            if r.minima[i].lower < floor_:
                flag_inconclusive = True
        if flag_inconclusive or r.unresolved:
            r.status = SEARCH_INCONCLUSIVE
    return 0


cdef bint _endpoint_monotone(
    const BasisTable* t, const PotentialPrep* p, double lo, double hi, double tau,
    Iv* ivwork, int direction
) noexcept nogil:
    """Whether the cell is classified increasing (``direction = 1``) or
    decreasing (``-1``) by the cell test (same arithmetic as the main loop)."""
    cdef Iv pe[4]
    cdef Iv ce[5]
    cdef double mid = 0.5 * (lo + hi)
    cdef double rr = 0.5 * (hi - lo)
    cdef double lower, cmag, q3_lo, q3_hi
    cdef bint discard, increasing
    potential_enclosure_cell(t, p, mid, mid, 3, ivwork, pe)
    potential_enclosure_cell(t, p, lo, hi, 4, ivwork, ce)
    lower = npmax(ce[2].lo, _taylor_lower(pe[2].lo, pe[3].lo, pe[3].hi, ce[4].lo, rr))
    cmag = npmax(fabs(ce[4].lo), fabs(ce[4].hi))
    q3_lo = npmax(ce[3].lo, pe[3].lo - cmag * rr * (1.0 + 4.0 * _U))
    q3_hi = npmin(ce[3].hi, pe[3].hi + cmag * rr * (1.0 + 4.0 * _U))
    discard = lower > tau
    increasing = not discard and q3_lo > 0.0
    if direction > 0:
        return increasing
    return not discard and not increasing and q3_hi < 0.0


cdef int separate_c(
    const BasisTable* t,
    const double* theta,
    const TailTable* tail_lo,
    const TailTable* tail_hi,
    const SearchOptions* o,
    SearchResult* r,
) noexcept nogil:
    """Curvature separation for ``theta``; ``0`` or ``-1`` on allocation failure.

    The caller releases ``r`` with ``search_free``.
    """
    cdef Sws w
    cdef Py_ssize_t tw = 3
    cdef int out
    r.status = SEARCH_INCONCLUSIVE
    r.n_bound = 0
    r.bound_cols = NULL
    r.n_min = 0
    r.minima = NULL
    r.n_tails = 0
    r.n_cuts = 0
    r.cut_points = NULL
    r.cut_rows = NULL
    r.cells = 0
    r.unresolved = 0
    if tail_lo != NULL and tail_work_size(tail_lo) > tw:
        tw = tail_work_size(tail_lo)
    if tail_hi != NULL and tail_work_size(tail_hi) > tw:
        tw = tail_work_size(tail_hi)
    w.fwork = <double*> malloc((5 * t.nf + 1) * sizeof(double))
    w.colwork = <double*> malloc((5 * t.d + 1) * sizeof(double))
    w.ivwork = <Iv*> malloc((5 * t.nf + 1) * sizeof(Iv))
    w.prep = <double*> malloc(prep_size(t) * sizeof(double))
    w.prep2 = <double*> malloc(prep_size(t) * sizeof(double))
    w.rest = <double*> malloc((t.d + 1) * sizeof(double))
    w.tailwork = <double*> malloc((tw + 1) * sizeof(double))
    w.pts = <double*> malloc((o.initial_cells + 5) * sizeof(double))
    _zero_arr(&w.cells)
    _zero_arr(&w.nxt)
    _zero_arr(&w.minima)
    _zero_arr(&w.convex)
    _zero_arr(&w.unres)
    _zero_arr(&w.rec)
    if (w.fwork == NULL or w.colwork == NULL or w.ivwork == NULL or w.prep == NULL
            or w.prep2 == NULL or w.rest == NULL or w.tailwork == NULL
            or w.pts == NULL):
        _sws_free(&w)
        return -1
    out = _search_body(t, theta, tail_lo, tail_hi, o, r, &w)
    _sws_free(&w)
    if out != 0:
        search_free(r)
    return out


cdef void _options(object options, SearchOptions* o):
    o.rho = options.rho
    o.initial_cells = options.initial_cells
    o.max_cells = options.max_cells
    o.min_width = options.min_width
    o.newton_tol = options.newton_tol
    o.newton_max = options.newton_max
    o.zeta_cap = options.zeta_cap
    o.fallback_zeta = options.fallback_zeta
    o.max_cuts = options.max_cuts
    o.feasibility_tol = options.feasibility_tol


def separate(BasisKernel kernel, const double[::1] theta, tail_lo, tail_hi, options):
    """Python entry of ``separate_c``.

    Returns ``(status, tau, (a, b), bound_cols, minima, tails, cut_points,
    cut_rows, cells, unresolved)``; ``minima`` rows are ``(x, value, lower,
    upper, kind)``, ``tails`` items ``(side, status, cutoff, zeta, lead_value,
    lead_power, lead_rate, margin, cut_atom)``.
    """
    cdef const BasisTable* t = &kernel.t
    cdef SearchOptions o
    cdef SearchResult r
    cdef const TailTable* tl = NULL
    cdef const TailTable* tu = NULL
    cdef int i, j
    if theta.shape[0] != t.d:
        raise ValueError("theta has the wrong length")
    if tail_lo is not None:
        tl = &(<TailKernel> tail_lo).t
    if tail_hi is not None:
        tu = &(<TailKernel> tail_hi).t
    _options(options, &o)
    if separate_c(t, &theta[0], tl, tu, &o, &r) != 0:
        raise MemoryError("G3 curvature search workspace")
    try:
        bound = [r.bound_cols[i] for i in range(r.n_bound)]
        minima = [
            (r.minima[i].x, r.minima[i].value, r.minima[i].lower, r.minima[i].upper,
             ("interior", "endpoint", "affine", "unresolved")[r.minima[i].kind])
            for i in range(r.n_min)
        ]
        tails = []
        for i in range(r.n_tails):
            res = r.tails[i]
            tails.append((
                ("lower", "upper")[r.tail_side[i]],
                ("nonnegative", "violated", "inconclusive")[res.status],
                res.cutoff if res.has_cutoff else None,
                res.zeta if res.has_cutoff else None,
                None if res.lead_value != res.lead_value else res.lead_value,
                res.lead_power if res.has_lead else None,
                res.lead_rate if res.has_lead else None,
                res.margin if res.has_margin else None,
                res.cut_atom,
            ))
        points = np.array([r.cut_points[i] for i in range(r.n_cuts)], dtype=np.float64)
        rows = np.array(
            [[r.cut_rows[i * t.d + j] for j in range(t.d)] for i in range(r.n_cuts)],
            dtype=np.float64,
        ).reshape(r.n_cuts, t.d)
        return (
            ("nonnegative", "violated", "inconclusive")[r.status],
            r.tau,
            (r.a, r.b),
            bound,
            minima,
            tails,
            points,
            rows,
            r.cells,
            r.unresolved,
        )
    finally:
        search_free(&r)
