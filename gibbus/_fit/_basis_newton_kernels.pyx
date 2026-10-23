# cython: language_level=3

"""Compiled G3 face Newton: G1's fused loop over a compiled basis objective.

G1 owner replaced: the Python objective callback of
``conic_newton._newton_on_representation`` (``solve_callback_newton``) for
G3 point fits, G1's compiled interval loop (``_conic_kernels.
solve_interval_newton``) for G3 interval fits (G3-044), and G1's compiled
shared-mixture objectives (``_shared_mixture_kernels.SharedMStepObjective``
and ``JointMixtureObjective``) for G3 mixtures (G3-045).  Retained from G1:
the fused nogil Newton loop ``_conic_kernels._face_newton_loop`` (warm/cold
conic subproblems, carried certificate, Armijo line search with the roundoff
rule, backtracking on any rejected trial) and the saddle-free metric
``_conic_kernels._in_safeguarded_metric``.

``BasisFaceObjective`` evaluates one objective on a face ``x -> theta = N x``
of joint coordinates.  The coefficients form blocks over one basis,
``theta_k = theta[index[k]]`` (one block for a single fit, one per mixture
component; shared columns are joint coordinates of every block).  Kinds:

* ``KIND_POINT`` (one block): ``nll = theta . Phi_bar + log Z``, gradient
  ``Phi_bar - E[Phi]``, ``F = Cov[Phi]``, ``C = 0``
  (``_normalizer_kernels.basis_state_c``);
* ``KIND_INTERVAL`` (one block): ``nll = -sum w log P(B)``, gradient
  ``sum w E[Phi | B] - E[Phi]``, ``F``, ``C = sum w Cov(Phi | B)``
  (``interval_states_c``);
* ``KIND_MSTEP``: the EM surrogate ``sum_k m_k nll_k(theta_k)`` of point or
  interval rows with per-component weights (the coupled M-step);
* ``KIND_JOINT``: the observed mixture NLL ``-sum_i w_i log sum_k pi_k
  p_k(row_i)`` over the blocks and ``K - 1`` logits (``pi_K`` the
  reference): gradient ``m_k (t_bar_k - E_k[Phi])`` per block and
  ``pi - m`` per logit; complete-data information ``G`` (``m_k F_k`` per
  block, ``diag pi - pi pi^T`` for the logits) and missing information
  ``M`` = within-row covariances ``sum_i w_i r_ik Cov_k(Phi | B_i)`` plus the
  covariance of the complete-data scores over the labels ``sum_i w_i
  sum_k r_ik (g_ik - g_bar_i)(g_ik - g_bar_i)^T``.

The metric is ``F`` when ``C`` is identically zero, else G1's saddle-free
metric of the restricted ``F`` and ``C``.  While its gate is on (inside a run)
every trial must first be globally convex in every block (G3-038): a block
is rejected cheaply when ``q''`` at one of its valleys lies below the shape
verdict's floor, else after the full curvature search; the first violating
trial of a run queues its block's cut rows, and the first call after the
next convex evaluation returns ``FACE_STOP``, ending the run at its accepted
iterate.

Rejections: a non-convex or improper trial is routine; a trial the
normalizer cannot resolve (status >= 10) or an allocation failure is
counted, with the last one's details, for the Python layer's failure ledger
(``basis_newton``).  When ``stop_on_failure`` is set (``GIBBUS_DEBUG``) such
a trial ends the run instead, so the ledger can raise.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport INFINITY, NAN, exp, fabs, isfinite, log, sqrt
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy

from .._model._basis_kernels cimport (
    BasisKernel,
    BasisTable,
    TailKernel,
    TailTable,
    eval_point,
)
from .._model._normalizer_kernels cimport (
    NORM_PROPER,
    GrowthKernel,
    GrowthTable,
    NormOptions,
    NormResult,
    basis_state_c,
    interval_states_c,
)
from ._conic_kernels cimport (
    FACE_STOP,
    FaceObjective,
    _face_newton_loop,
    _in_safeguarded_metric,
)
from ._curvature_kernels cimport (
    SEARCH_NONNEGATIVE,
    SEARCH_INCONCLUSIVE,
    SearchOptions,
    SearchResult,
    search_free,
    separate_c,
    valleys_c,
)

cnp.import_array()

# ``basis_constraints._same_constraint``: four ulp of each entry.
cdef double _SAME = 4.0 * 2.220446049250313e-16

# Rejection codes of ``_evaluate`` (any nonzero code backtracks in G1's loop).
cdef enum:
    REJECT_SHAPE = 1
    REJECT_IMPROPER = 2
    REJECT_FAILURE = 3
    FAIL_ALLOCATION = -1

# Objective kinds.
cdef enum:
    K_POINT = 0
    K_INTERVAL = 1
    K_MSTEP = 2
    K_JOINT = 3

# Python-visible codes.
ACCEPTED = 0
STOP = FACE_STOP
SHAPE = REJECT_SHAPE
IMPROPER = REJECT_IMPROPER
FAILURE = REJECT_FAILURE
# ``failures()`` status of an allocation failure (normalizer statuses are >= 10).
ALLOCATION = FAIL_ALLOCATION
KIND_POINT = K_POINT
KIND_INTERVAL = K_INTERVAL
KIND_MSTEP = K_MSTEP
KIND_JOINT = K_JOINT


from ._series_search cimport SeriesSearchTable, SeriesSearchKernel, series_candidates_c

cdef struct Ctx:
    # Blocks: table and size per block; per-block slots have stride ``d``
    # (the largest size), the block's own arrays contiguous inside them.
    SeriesSearchTable** series
    int series_hits
    int full_searches
    const BasisTable** tables
    const int* dims
    const double** cscale
    int d
    int n
    int k
    int nb
    int kind
    int n_logit
    const double* basis
    const Py_ssize_t* index
    const TailTable** ntail_lo
    const TailTable** ntail_hi
    const GrowthTable** growth_lo
    const GrowthTable** growth_hi
    const TailTable** ctail_lo
    const TailTable** ctail_hi
    NormOptions nopt
    SearchOptions sopt
    double floor_
    double series_margin
    int cap_rows
    # Data.
    const double* stats
    const double* mass
    Py_ssize_t n_rows
    const double* row_lo
    const double* row_hi
    const double* row_w
    const double* phi_rows
    int intervals
    # Run state.
    int gate
    void* curvature_contexts
    void* curvature_options
    int stop_on_failure
    double* dom_a
    double* dom_b
    const double* known
    int nknown
    int armed
    int has_pending
    # Rows of the latest shape rejection (``c.rows``, block ``r_block``):
    # ``_step``'s ratio test; zero when the rejection was not by such rows.
    int r_block
    int r_nrows
    int p_block
    int p_nrows
    int p_npoints
    double p_a
    double p_b
    int failures
    int f_status
    int f_side
    double f_diag[6]
    Py_ssize_t f_row
    int f_block
    int evaluations
    int trace_cap
    int trace_n
    # Work buffers (owned by the Python object).
    double* theta
    double* part
    double* parts
    double* mean
    double* cov
    double* logz
    double* fwork
    double* colwork
    double* vbuf
    double* rows
    double* unit
    double* p_theta
    double* p_rows
    double* f_theta
    double* trace
    double* gfull
    double* ffull
    double* cfull
    double* fface
    double* cface
    double* mwork
    double* acc_mean
    double* acc_fisher
    double* acc_missing
    double* agg
    double* row_logp
    double* row_mean
    double* row_cov
    double* resp
    double* gvec
    double* nll_part
    double* scores


cdef inline void _to_full(Ctx* c, const double* x) noexcept nogil:
    cdef int i, j
    cdef double v
    for i in range(c.n):
        v = 0.0
        for j in range(c.k):
            v = v + c.basis[i * c.k + j] * x[j]
        c.theta[i] = v


cdef inline void _gather(Ctx* c, int b, double* out) noexcept nogil:
    """Block ``b``'s coefficients ``theta[index[b]]``."""
    cdef int j
    for j in range(c.dims[b]):
        out[j] = c.theta[c.index[b * c.d + j]]


cdef bint _unit_row(const double* row, int n, double* out) noexcept nogil:
    """``_ConstraintRows.merged`` normalization; False for an unusable row."""
    cdef double peak = 0.0, nrm = 0.0
    cdef int i
    for i in range(n):
        if not isfinite(row[i]):
            return False
        if fabs(row[i]) > peak:
            peak = fabs(row[i])
    if not peak > 0.0:
        return False
    for i in range(n):
        out[i] = row[i] / peak
        nrm += out[i] * out[i]
    nrm = sqrt(nrm)
    for i in range(n):
        out[i] = out[i] / nrm
    return True


cdef bint _known(const Ctx* c, const double* u) noexcept nogil:
    """Whether joint unit row ``u`` is a known row (``_same_constraint``)."""
    cdef int r, i
    cdef const double* v
    cdef bint same
    for r in range(c.nknown):
        v = &c.known[r * c.n]
        same = True
        for i in range(c.n):
            if (u[i] != 0.0) != (v[i] != 0.0) or fabs(u[i] - v[i]) > _SAME * fabs(u[i]):
                same = False
                break
        if same:
            return True
    return False


cdef void _queue(
    Ctx* c, int b, int nrows, int npoints, double a, double bb
) noexcept nogil:
    """Keep ``c.rows`` as block ``b``'s cuts of the first violating trial, if new.

    Only the first violating trial of a run (normally the full step)
    supplies curvature cuts; backtracked trials are only tested.
    """
    cdef int r, j, db = c.dims[b]
    cdef bint new = False
    if c.has_pending or nrows == 0:
        return
    for r in range(nrows):
        # Embed the block row (shared coordinates summed), then normalize.
        for j in range(c.n):
            c.gvec[j] = 0.0
        for j in range(db):
            c.gvec[c.index[b * c.d + j]] += c.rows[r * db + j]
        if _unit_row(c.gvec, c.n, c.unit) and not _known(c, c.unit):
            new = True
            break
    if not new:
        return
    memcpy(c.p_theta, c.theta, c.n * sizeof(double))
    memcpy(c.p_rows, c.rows, nrows * db * sizeof(double))
    c.p_block = b
    c.p_nrows = nrows
    c.p_npoints = npoints
    c.p_a = a
    c.p_b = bb
    c.has_pending = 1


cdef int _resolve_contact(Ctx* c, int b) noexcept nogil:
    """Ask the shared separator to refine a numerically ambiguous contact."""
    if c.curvature_contexts == NULL:
        return 0
    with gil:
        try:
            from .curvature_search import _separate_curvature
            contexts = <object> c.curvature_contexts
            options = <object> c.curvature_options
            result = _separate_curvature(contexts[b],
                np.array(<double[:c.dims[b]]> c.part, copy=True), options)
            return result.status == "nonnegative"
        except (ArithmeticError, ValueError, OverflowError):
            return 0


cdef int _gate(Ctx* c, int b) noexcept nogil:
    """Block ``b``: ``0`` globally convex, ``1`` rejection, ``-1`` allocation failure.

    A block is rejected outright when ``q''`` at one of its valleys (the
    local minima over the round's search domain) lies below the shape
    verdict's floor: that point proves the violation and ``Phi''(x)`` there
    is a valid cut.  Only a block passing this test pays for the full
    curvature search, whose ``nonnegative`` verdict accepts it.  Columns
    overflowing far out give no usable row: such a point is left to the
    full search.
    """
    cdef const BasisTable* t = c.tables[b]
    cdef int d = c.dims[b]
    cdef int nv = 0, i, j, nbad = 0, m = 0, npts
    cdef double q2
    cdef bint usable, used_series=False
    cdef double rejection_floor
    cdef SearchResult sr
    cdef const TailTable* tt
    _gather(c, b, c.part)
    nv=-1
    if c.series[b]!=NULL:
        nv=series_candidates_c(c.series[b],c.part,c.vbuf,c.sopt.initial_cells+4)
        used_series=nv>=0
    if nv<0:
        if valleys_c(t, c.part, c.dom_a[b], c.dom_b[b], &c.sopt, c.vbuf, &nv) != 0:
            return -1
    rejection_floor=-c.series_margin/(t.h*t.h) if used_series else c.floor_
    for i in range(nv):
        eval_point(t, c.vbuf[i], 2, c.fwork, c.colwork)
        usable = False
        q2 = 0.0
        for j in range(d):
            if not isfinite(c.colwork[2 * d + j]):
                usable = False
                break
            if c.colwork[2 * d + j] != 0.0:
                usable = True
            q2 = q2 + c.colwork[2 * d + j] * c.part[j]
        if usable and q2 < rejection_floor:
            memcpy(&c.rows[nbad * d], &c.colwork[2 * d], d * sizeof(double))
            nbad += 1
    if nbad:
        if used_series: c.series_hits+=1
        _queue(c, b, nbad, nbad, c.dom_a[b], c.dom_b[b])
        c.r_block = b
        c.r_nrows = nbad
        return 1
    c.full_searches+=1
    if separate_c(t, c.part, c.ctail_lo[b], c.ctail_hi[b], &c.sopt, &sr) != 0:
        return -1
    if sr.n_bound == 0:
        if sr.status == SEARCH_NONNEGATIVE:
            search_free(&sr)
            return 0
        if sr.status == SEARCH_INCONCLUSIVE and sr.unresolved == 0 and _resolve_contact(c, b):
            search_free(&sr)
            return 0
    if sr.n_bound:
        # gamma < 0 sends q'' to -inf at an endpoint: the bound rows.
        for i in range(sr.n_bound):
            for j in range(d):
                c.rows[m * d + j] = 1.0 if j == sr.bound_cols[i] else 0.0
            m += 1
        _queue(c, b, m, 0, sr.a, sr.b)
    else:
        for i in range(sr.n_cuts):
            memcpy(&c.rows[m * d], &sr.cut_rows[i * d], d * sizeof(double))
            m += 1
        npts = m
        for i in range(sr.n_tails):
            if sr.tails[i].status == 1 and sr.tails[i].cut_atom >= 0:
                tt = c.ctail_lo[b] if sr.tail_side[i] == 0 else c.ctail_hi[b]
                memcpy(&c.rows[m * d], &tt.acut[sr.tails[i].cut_atom * d],
                       d * sizeof(double))
                m += 1
        _queue(c, b, m, npts, sr.a, sr.b)
        if m == npts:
            # Point cuts only: a violated tail's leading functional belongs to
            # its asymptotic hierarchy, and a step to that boundary is left to
            # the plain backtracking.
            c.r_block = b
            c.r_nrows = npts
    search_free(&sr)
    return 1


cdef int _fail(
    Ctx* c, int status, const NormResult* res, Py_ssize_t row, int block
) noexcept nogil:
    """Record a non-routine trial failure; the run's rejection or stop code."""
    cdef int i
    c.failures += 1
    c.f_status = status
    c.f_row = row
    c.f_block = block
    c.f_side = -1 if res == NULL else res.side
    for i in range(6):
        c.f_diag[i] = 0.0 if res == NULL else res.diag[i]
    memcpy(c.f_theta, c.theta, c.n * sizeof(double))
    if c.stop_on_failure:
        return FACE_STOP
    return REJECT_FAILURE


cdef void _restrict(Ctx* c, const double* full, double* out) noexcept nogil:
    """``out = N^T full N`` (``(k, k)``), a column at a time through colwork."""
    cdef int n = c.n, k = c.k, i, j, a, b
    cdef double v
    for a in range(k):
        for i in range(n):
            v = 0.0
            for j in range(n):
                v = v + full[i * n + j] * c.basis[j * k + a]
            c.colwork[i] = v
        for b in range(k):
            v = 0.0
            for i in range(n):
                v = v + c.basis[i * k + b] * c.colwork[i]
            out[b * k + a] = v


cdef inline void _add_vec(Ctx* c, int b, double s, const double* v) noexcept nogil:
    """``gfull[index[b]] += s v``."""
    cdef int j
    for j in range(c.dims[b]):
        c.gfull[c.index[b * c.d + j]] += s * v[j]


cdef inline void _add_mat(
    Ctx* c, int b, double s, const double* m, double* full
) noexcept nogil:
    """``full[index[b], index[b]] += s m`` (``m`` block-sized)."""
    cdef int i, j, n = c.n, db = c.dims[b]
    cdef const Py_ssize_t* idx = &c.index[b * c.d]
    for i in range(db):
        for j in range(db):
            full[idx[i] * n + idx[j]] += s * m[i * db + j]


cdef int _normalize(
    Ctx* c, int b, const double* weights, bint per_row, double* nll, double* g
) noexcept nogil:
    """Component ``b`` at ``theta[index[b]]``: its NLL and gradient (block coords).

    Point data: ``stats[b]``; interval rows: ``weights`` (sum one).  Fills
    ``mean[b]``, ``cov[b]``, ``logz[b]`` and (interval rows) ``agg``; with
    ``per_row`` (interval rows) the per-row ``row_logp/row_mean/row_cov`` of
    block ``b``.  Returns ``0``, a rejection, ``FACE_STOP`` or the failure
    code.
    """
    cdef int d = c.dims[b], i, code
    cdef Py_ssize_t R = c.n_rows
    cdef NormResult res, rres
    cdef Py_ssize_t row = -1
    cdef double v
    cdef const BasisTable* t = c.tables[b]
    cdef double* mb = &c.mean[b * c.d]
    cdef double* cb = &c.cov[b * c.d * c.d]
    cdef const double* st
    _gather(c, b, c.part)
    if not c.intervals:
        code = basis_state_c(t, c.part, c.ntail_lo[b], c.ntail_hi[b], c.growth_lo[b],
                             c.growth_hi[b], c.cscale[b], &c.nopt, False, mb, cb,
                             &res)
        if code != 0:
            return _fail(c, FAIL_ALLOCATION, NULL, -1, b)
        if res.status != NORM_PROPER:
            if res.status >= 10:
                return _fail(c, res.status, &res, -1, b)
            return REJECT_IMPROPER
        c.logz[b] = res.log_z
        if nll != NULL:
            st = &c.stats[b * c.d]
            v = 0.0
            for i in range(d):
                v = v + c.part[i] * st[i]
            nll[0] = v + res.log_z
            for i in range(d):
                g[i] = st[i] - mb[i]
        return 0
    code = interval_states_c(
        t, c.part, c.ntail_lo[b], c.ntail_hi[b], c.growth_lo[b], c.growth_hi[b],
        c.cscale[b], &c.nopt, R, c.row_lo, c.row_hi, weights, mb, cb,
        c.agg,
        &c.row_logp[b * R] if per_row else NULL,
        &c.row_mean[b * R * c.d] if per_row else NULL,
        &c.row_cov[b * R * c.d * c.d] if per_row and c.row_cov != NULL else NULL,
        &res, &rres, &row,
    )
    if code != 0:
        return _fail(c, FAIL_ALLOCATION, NULL, -1, b)
    if res.status != NORM_PROPER:
        if res.status >= 10:
            return _fail(c, res.status, &res, -1, b)
        return REJECT_IMPROPER
    if row >= 0:
        # Not routine: a row of a proper trial could not be resolved.
        return _fail(c, rres.status, &rres, row, b)
    c.logz[b] = res.log_z
    if nll != NULL:
        nll[0] = -c.agg[0]
        for i in range(d):
            g[i] = c.agg[1 + i] - mb[i]
    return 0


cdef int _single(Ctx* c, double* nll) noexcept nogil:
    """``KIND_POINT`` / ``KIND_INTERVAL`` / ``KIND_MSTEP`` into gfull/ffull/cfull."""
    cdef int d = c.d, b, code
    cdef double nb_, s
    cdef double* g = c.gvec
    nll[0] = 0.0
    for b in range(c.nb):
        code = _normalize(
            c, b, &c.row_w[b * c.n_rows] if c.intervals else NULL, False, &nb_, g
        )
        if code != 0:
            return code
        s = 1.0 if c.kind != K_MSTEP else c.mass[b]
        nll[0] += s * nb_
        _add_vec(c, b, s, g)
        _add_mat(c, b, s, &c.cov[b * d * d], c.ffull)
        if c.intervals:
            _add_mat(c, b, s, &c.agg[1 + c.dims[b]], c.cfull)
    return 0


cdef inline const double* _row_stat(
    const Ctx* c, int b, Py_ssize_t i
) noexcept nogil:
    """Row ``i``'s statistic under block ``b``: ``E_b[Phi | B_i]`` or ``Phi(x_i)``."""
    if c.intervals:
        return &c.row_mean[b * c.n_rows * c.d + i * c.dims[b]]
    return &c.phi_rows[b * c.n_rows * c.d + i * c.dims[b]]


cdef int _joint(Ctx* c, double* nll) noexcept nogil:
    """``KIND_JOINT`` into gfull/ffull/cfull (see the module docstring).

    With ``c.scores`` the per-row NLL scores ``-g_bar_i`` (joint
    coordinates) are kept for the selection diagnostics.
    """
    cdef int d = c.d, n = c.n, nb = c.nb, b, j, i2, j2, code, db
    cdef Py_ssize_t i, R = c.n_rows
    cdef double lse, top, w, r, v, s
    cdef double* logpi = c.nll_part
    cdef double* lterm
    cdef const double* t_i
    cdef double* g
    cdef int l0 = n - c.n_logit
    for b in range(nb):
        code = _normalize(c, b, c.row_w, True, NULL, NULL)
        if code != 0:
            return code
    # log pi from the logits (the last component is the reference).
    top = 0.0
    for b in range(nb - 1):
        if c.theta[l0 + b] > top:
            top = c.theta[l0 + b]
    lse = exp(-top)
    for b in range(nb - 1):
        lse += exp(c.theta[l0 + b] - top)
    lse = top + log(lse)
    for b in range(nb):
        logpi[b] = (c.theta[l0 + b] if b < nb - 1 else 0.0) - lse
    # Per-row log terms (point rows from Phi_b(x_i)) and responsibilities.
    for b in range(nb):
        _gather(c, b, &c.parts[b * d])
    nll[0] = 0.0
    for i in range(R):
        top = -INFINITY
        for b in range(nb):
            if c.intervals:
                v = c.row_logp[b * R + i]
            else:
                t_i = _row_stat(c, b, i)
                v = 0.0
                for j in range(c.dims[b]):
                    v = v + c.parts[b * d + j] * t_i[j]
                v = -v - c.logz[b]
                c.row_logp[b * R + i] = v
            c.resp[i * nb + b] = logpi[b] + v
            if c.resp[i * nb + b] > top:
                top = c.resp[i * nb + b]
        if not isfinite(top):
            return REJECT_IMPROPER
        s = 0.0
        for b in range(nb):
            s += exp(c.resp[i * nb + b] - top)
        lse = top + log(s)
        for b in range(nb):
            c.resp[i * nb + b] = exp(c.resp[i * nb + b] - lse)
        nll[0] -= c.row_w[i] * lse
    # Gradient and complete-data information.
    for b in range(nb):
        db = c.dims[b]
        lterm = &c.agg[b * d]  # m_b t_bar_b (agg reused as scratch, nb d)
        for j in range(db):
            lterm[j] = 0.0
        s = 0.0
        for i in range(R):
            w = c.row_w[i] * c.resp[i * nb + b]
            s += w
            t_i = _row_stat(c, b, i)
            for j in range(db):
                lterm[j] += w * t_i[j]
        c.nll_part[nb + b] = s  # m_b
        for j in range(db):
            c.gvec[j] = lterm[j] - s * c.mean[b * d + j]
        _add_vec(c, b, 1.0, c.gvec)
        _add_mat(c, b, s, &c.cov[b * d * d], c.ffull)
    for b in range(nb - 1):
        c.gfull[l0 + b] += exp(logpi[b]) - c.nll_part[nb + b]
        for j2 in range(nb - 1):
            c.ffull[(l0 + b) * n + l0 + j2] += (
                (exp(logpi[b]) if b == j2 else 0.0) - exp(logpi[b] + logpi[j2])
            )
    # Missing information: within-row covariances, then label covariance.
    for i in range(R):
        for b in range(nb):
            w = c.row_w[i] * c.resp[i * nb + b]
            if w == 0.0:
                continue
            if c.intervals:
                db = c.dims[b]
                _add_mat(
                    c, b, w, &c.row_cov[b * R * d * d + i * db * db], c.cfull
                )
        # g_ib (joint) and g_bar_i, in gvec rows: nb + 1 vectors of length n.
        g = &c.gvec[n]
        for j in range((nb + 1) * n):
            g[j] = 0.0
        for b in range(nb):
            t_i = _row_stat(c, b, i)
            for j in range(c.dims[b]):
                g[b * n + c.index[b * d + j]] += c.mean[b * d + j] - t_i[j]
            for j in range(nb - 1):
                g[b * n + l0 + j] = (1.0 if j == b else 0.0) - exp(logpi[j])
            r = c.resp[i * nb + b]
            for j in range(n):
                g[nb * n + j] += r * g[b * n + j]
        if c.scores != NULL:
            for j in range(n):
                c.scores[i * n + j] = -g[nb * n + j]
        for b in range(nb):
            w = c.row_w[i] * c.resp[i * nb + b]
            if w == 0.0:
                continue
            for j in range(n):
                g[b * n + j] -= g[nb * n + j]
            for i2 in range(n):
                v = w * g[b * n + i2]
                if v == 0.0:
                    continue
                for j2 in range(n):
                    c.cfull[i2 * n + j2] += v * g[b * n + j2]
    return 0


cdef int _objective(Ctx* c, double* nll) noexcept nogil:
    """Joint NLL, gradient ``gfull``, ``F`` and ``C`` (``ffull``/``cfull``)."""
    cdef int i
    for i in range(c.n):
        c.gfull[i] = 0.0
    for i in range(c.n * c.n):
        c.ffull[i] = 0.0
        c.cfull[i] = 0.0
    if c.kind == K_JOINT:
        return _joint(c, nll)
    return _single(c, nll)


cdef int _evaluate(
    void* vctx,
    const double* x,
    double* nll,
    double* gradient,
    double* metric,
    double* smallest,
) noexcept nogil:
    """``FaceObjective.evaluate``: 0 accepted, ``FACE_STOP``, else a rejection."""
    cdef Ctx* c = <Ctx*> vctx
    cdef int n = c.n, k = c.k, i, j, b, code
    cdef double v
    c.r_nrows = 0
    _to_full(c, x)
    if c.gate:
        if c.armed:
            return FACE_STOP
        for b in range(c.nb):
            code = _gate(c, b)
            if code < 0:
                return _fail(c, FAIL_ALLOCATION, NULL, -1, b)
            if code:
                return REJECT_SHAPE
    for b in range(c.nb):
        for i in range(c.dims[b]):
            if c.tables[b].nonneg[i] and c.theta[c.index[b * c.d + i]] < 0.0:
                return REJECT_SHAPE  # outside the model (a negative log amplitude)
    c.evaluations += 1
    if c.trace_n < c.trace_cap:
        memcpy(&c.trace[c.trace_n * n], c.theta, n * sizeof(double))
        c.trace_n += 1
    code = _objective(c, nll)
    if code != 0:
        return code
    for j in range(k):
        v = 0.0
        for i in range(n):
            v = v + c.basis[i * k + j] * c.gfull[i]
        gradient[j] = v
    _restrict(c, c.ffull, c.fface)
    if c.intervals or c.kind == K_JOINT:
        _restrict(c, c.cfull, c.cface)
        _in_safeguarded_metric(k, c.fface, c.cface, metric, c.mwork, smallest)
    else:
        memcpy(metric, c.fface, k * k * sizeof(double))
        smallest[0] = NAN
    if c.gate and c.has_pending:
        c.armed = 1
    return 0


cdef int _refine_scalar_qp(
    void* vctx, int n, int r, const double* b, const double* h,
    const double* g, const double* theta, double* endpoint, double* dual,
    double* model, double* gap,
) noexcept nogil:
    cdef int i
    cdef double candidate_bound, incumbent_bound
    with gil:
        try:
            from .linear_qp import _projection_candidate
            result = _projection_candidate(
                np.array(<double[:n*n]> h, copy=True).reshape(n,n),
                np.array(<double[:n]> g, copy=True),
                np.array(<double[:n]> theta, copy=True),
                np.array(<double[:r*n]> b, copy=True).reshape(r,n),
            )
            if result is None:
                return 0
            candidate_bound = max(0.0, -result.model_value) + max(0.0, result.gap)
            incumbent_bound = max(0.0, -model[0]) + max(0.0, gap[0])
            if isfinite(incumbent_bound) and candidate_bound >= incumbent_bound:
                return 0
            for i in range(n):
                endpoint[i] = result.params[i]
            for i in range(r):
                dual[i] = result.dual[i]
            model[0] = result.model_value
            gap[0] = result.gap
            return 1
        except (np.linalg.LinAlgError, RuntimeError, ValueError):
            return 0


cdef double _step(
    void* vctx, const double* x, const double* trial, double alpha
) noexcept nogil:
    """``FaceObjective.step``: the ratio-test step fraction after a shape rejection.

    Every rejecting point cut ``r = Phi''(x)`` (a violated valley or search
    minimum; tail and bound rejections are not hinted) is linear in
    ``theta``, so along the step ``r . theta`` is linear in the fraction: the
    largest fraction keeping each row nonnegative is exact for that row.  The rows were
    evaluated at the rejected trial (``c.theta``, set by ``_evaluate``); the
    accepted point ``x`` must lie strictly inside each violated row, else
    ``-1`` (backtrack).  The result is shortened by a relative ``1e-4`` and
    gated again by the loop, so no constraint test is relaxed.

    Parameters
    ----------
    vctx : void*
        ``Ctx``.
    x, trial : const double*
        Accepted point and rejected trial (face coordinates).
    alpha : double
        Step fraction of the rejected trial.
    """
    cdef Ctx* c = <Ctx*> vctx
    cdef int b = c.r_block, db, j, jj, r, idx
    cdef double v0, v1, a0, frac, best = alpha
    if c.r_nrows <= 0:
        return -1.0
    db = c.dims[b]
    _gather(c, b, c.part)  # the trial's block coefficients
    for r in range(c.r_nrows):
        v0 = 0.0
        v1 = 0.0
        for jj in range(db):
            idx = c.index[b * c.d + jj]
            a0 = 0.0
            for j in range(c.k):
                a0 = a0 + c.basis[idx * c.k + j] * x[j]
            v0 = v0 + c.rows[r * db + jj] * a0
            v1 = v1 + c.rows[r * db + jj] * c.part[jj]
        if not v1 < 0.0:
            continue
        if not v0 > 0.0:
            return -1.0
        frac = v0 / (v0 - v1)
        if alpha * frac < best:
            best = alpha * frac
    if not best < alpha:
        return -1.0
    return best * (1.0 - 1e-4)


cdef void _accept(void* vctx) noexcept nogil:
    """Keep the accepted trial's ``E[Phi]``, ``F`` and ``C`` (just evaluated)."""
    cdef Ctx* c = <Ctx*> vctx
    memcpy(c.acc_mean, c.mean, c.nb * c.d * sizeof(double))
    memcpy(c.acc_fisher, c.fface, c.k * c.k * sizeof(double))
    if c.intervals or c.kind == K_JOINT:
        memcpy(c.acc_missing, c.cface, c.k * c.k * sizeof(double))


cdef inline double* _ptr(object array):
    return <double*> cnp.PyArray_DATA(array)


cdef class BasisFaceObjective:
    """Compiled objective on one face of joint coordinates, with the convexity gate.

    Parameters
    ----------
    kernels : sequence of BasisKernel
        Column table of each block's basis (one geometry).
    basis : numpy.ndarray
        Face basis ``N``, shape ``(n, k)``.
    index : sequence of numpy.ndarray
        Per block, the joint coordinate of each of its coefficients
        (``theta_b = theta[index[b]]``).
    normalizer : tuple
        ``(blocks, options)``: per block ``(tail_lo, tail_hi, growth_lo,
        growth_hi, column_scale)`` (order-1 ``TailKernel`` / ``GrowthKernel``
        per infinite end, ``None`` otherwise, face-restricted, and the moment
        column scales) and ``_NormalizerOptions``.
    search : tuple
        ``(blocks, options)``: per block ``(tail_lo, tail_hi)`` order-2
        ``TailKernel`` and ``_CurvatureOptions``.
    kind : int
        ``KIND_POINT``, ``KIND_INTERVAL``, ``KIND_MSTEP`` or ``KIND_JOINT``.
    data : dict
        ``stats`` (per block ``Phi_bar_b``: point data of
        ``KIND_POINT``/``KIND_MSTEP``), ``rows`` (``(lower, upper)``
        interval rows), ``weights`` (row weights: ``(R,)``, or ``(nb, R)``
        for ``KIND_MSTEP``), ``mass`` (``(nb,)``, ``KIND_MSTEP``), ``phi``
        (per block ``Phi_b(x_i)``, ``(R, d_b)``: point rows of
        ``KIND_JOINT``), ``n_logit`` (``KIND_JOINT``) and ``scores`` (keep
        the per-row joint scores of ``KIND_JOINT``).
    trace : int
        Record the first ``trace`` normalized points (tests).
    """

    cdef Ctx c
    cdef object _keep
    cdef void* _ptrs

    def __cinit__(self, kernels, basis, index, normalizer, search, int kind, data,
                  int trace=0, series_models=None, curvature_contexts=None):
        cdef int nb, b, n_logit = 0
        cdef Py_ssize_t R = 0
        cdef BasisKernel kb
        self._ptrs = NULL
        self.c.curvature_contexts = NULL
        self.c.curvature_options = NULL
        self.c.series=NULL
        self.c.series_hits=0;self.c.full_searches=0
        kernels = list(kernels)
        nb = len(kernels)
        if nb < 1 or len(index) != nb:
            raise ValueError("one kernel and one index per block")
        dims = np.array([(<BasisKernel> kern).t.d for kern in kernels], dtype=np.intc)
        d = int(dims.max())
        basis = np.ascontiguousarray(basis, dtype=np.float64)
        if basis.ndim != 2:
            raise ValueError("face basis must be a matrix")
        n = basis.shape[0]
        k = basis.shape[1]
        padded = np.zeros((nb, d), dtype=np.intp)
        for b in range(nb):
            idx = np.asarray(index[b], dtype=np.intp)
            if idx.shape != (dims[b],) or idx.min() < 0 or idx.max() >= n:
                raise ValueError("block index outside the joint coordinates")
            padded[b, : dims[b]] = idx
        nblocks, nopts = normalizer
        sblocks, sopts = search
        if len(nblocks) != nb or len(sblocks) != nb:
            raise ValueError("one normalizer and search entry per block")
        scales = []
        for b in range(nb):
            cs = np.ascontiguousarray(nblocks[b][4], dtype=np.float64)
            if cs.shape != (dims[b],):
                raise ValueError("column_scale has the wrong length")
            scales.append(cs)
        intervals = "rows" in data
        stats = np.zeros((nb, d))
        lower = upper = np.zeros(1)
        weights = np.zeros(1)
        mass = np.ones(nb)
        phi = np.zeros(1)
        if kind in (K_POINT, K_INTERVAL) and nb != 1:
            raise ValueError("point and interval objectives have one block")
        if kind == K_POINT or (kind == K_MSTEP and not intervals):
            for b in range(nb):
                sb = np.asarray(data["stats"][b], dtype=np.float64).reshape(-1)
                if sb.shape != (dims[b],):
                    raise ValueError("statistics have the wrong length")
                stats[b, : dims[b]] = sb
        if intervals:
            lower, upper = (
                np.ascontiguousarray(a, dtype=np.float64) for a in data["rows"]
            )
            R = lower.shape[0]
            if R < 1 or upper.shape != lower.shape:
                raise ValueError("interval rows have mismatched shapes")
        if kind == K_INTERVAL or kind == K_JOINT or (kind == K_MSTEP and intervals):
            weights = np.ascontiguousarray(data["weights"], dtype=np.float64)
            if kind == K_MSTEP:
                weights = weights.reshape(nb, -1)
        if kind == K_MSTEP:
            mass = np.ascontiguousarray(data["mass"], dtype=np.float64)
            if mass.shape != (nb,):
                raise ValueError("one mass per block")
        if kind == K_JOINT:
            n_logit = int(data["n_logit"])
            if n_logit != nb - 1:
                raise ValueError("a joint mixture has K - 1 logits")
            if not intervals:
                R = np.asarray(data["phi"][0]).shape[0]
                phi = np.zeros(nb * R * d)
                for b in range(nb):
                    pb = np.ascontiguousarray(data["phi"][b], dtype=np.float64)
                    if pb.shape != (R, dims[b]):
                        raise ValueError("phi must have shape (R, d_b)")
                    phi[b * R * d : b * R * d + R * dims[b]] = pb.reshape(-1)
            if weights.shape != (R,):
                raise ValueError("one weight per row")
        elif kind == K_INTERVAL and weights.shape != (R,):
            raise ValueError("one weight per row")
        elif kind == K_MSTEP and intervals and weights.shape != (nb, R):
            raise ValueError("one weight vector per block")
        # Valleys (initial_cells + 4) or a search's cuts, tails and bound rows.
        cap = max(int(sopts.max_cuts) + d + 2, int(sopts.initial_cells) + 4)
        trace = max(trace, 0)
        wide = max(5 * d, n) + 1
        joint = kind == K_JOINT
        nf = max((<BasisKernel> kern).t.nf for kern in kernels)
        bufs = dict(
            theta=np.zeros(n), part=np.zeros(d), parts=np.zeros(nb * d),
            mean=np.zeros(nb * d),
            cov=np.zeros(nb * d * d), logz=np.zeros(nb),
            fwork=np.zeros(5 * nf + 1), colwork=np.zeros(wide),
            vbuf=np.zeros(int(sopts.initial_cells) + 4), rows=np.zeros((cap, d)),
            unit=np.zeros(n), p_theta=np.zeros(n), p_rows=np.zeros((cap, d)),
            f_theta=np.zeros(n), trace=np.zeros((max(trace, 1), n)),
            gfull=np.zeros(n), ffull=np.zeros((n, n)), cfull=np.zeros((n, n)),
            fface=np.zeros((k, k)), cface=np.zeros((k, k)),
            mwork=np.zeros(7 * k * k + 4 * k), acc_mean=np.zeros(nb * d),
            acc_fisher=np.zeros((k, k)), acc_missing=np.zeros((k, k)),
            agg=np.zeros(max(1 + d + d * d, nb * d)),
            row_logp=np.zeros(max(R * nb, 1)),
            row_mean=np.zeros(max(R * nb * d, 1) if joint and intervals else 1),
            row_cov=np.zeros(max(R * nb * d * d, 1) if joint and intervals else 1),
            resp=np.zeros(max(R * nb, 1)),
            gvec=np.zeros(max(d, n) + (nb + 1) * n + 1),
            nll_part=np.zeros(2 * nb + 1),
            scores=np.zeros(max(R * n, 1) if joint and data.get("scores") else 1),
            dom=np.zeros((2, nb)), known=np.zeros((1, n)),
        )
        self._keep = dict(kernels=kernels, basis=basis, index=padded, dims=dims,
                          stats=stats, scales=scales, nblocks=nblocks,
                          sblocks=sblocks, bufs=bufs, rows=(lower, upper),
                          weights=weights, mass=mass, phi=phi)
        if curvature_contexts is not None:
            if len(curvature_contexts) != nb:
                raise ValueError("one curvature context per block")
            self._keep["curvature_contexts"] = tuple(curvature_contexts)
            self._keep["curvature_options"] = sopts
            self.c.curvature_contexts = <void*> self._keep["curvature_contexts"]
            self.c.curvature_options = <void*> sopts
        self.c.series=<SeriesSearchTable**>calloc(nb,sizeof(SeriesSearchTable*))
        if self.c.series==NULL: raise MemoryError("series search pointer table")
        caches=[None]*nb
        if getattr(sopts,"series_candidates",False) and series_models is not None:
            from .series_search import _SeriesCandidateCache
            if len(series_models)!=nb: raise ValueError("one series model per block")
            caches=[_SeriesCandidateCache(m,sopts) for m in series_models]
        self._keep["series_caches"]=caches
        self._keep["series_kernels"]=[None]*nb
        # Per-block pointers: tables, six table arrays and column scales.
        self._ptrs = calloc(8 * nb + 1, sizeof(void*))
        if self._ptrs == NULL:
            raise MemoryError("compiled G3 face objective")
        cdef const TailTable** tp = <const TailTable**> self._ptrs
        cdef const GrowthTable** gp = <const GrowthTable**> self._ptrs
        cdef const BasisTable** bp = <const BasisTable**> self._ptrs
        cdef const double** sp = <const double**> self._ptrs
        for b in range(nb):
            ntl, ntu, gl, gu, _ = nblocks[b]
            ctl, ctu = sblocks[b]
            kb = kernels[b]
            tp[b] = NULL if ntl is None else &(<TailKernel> ntl).t
            tp[nb + b] = NULL if ntu is None else &(<TailKernel> ntu).t
            gp[2 * nb + b] = NULL if gl is None else &(<GrowthKernel> gl).t
            gp[3 * nb + b] = NULL if gu is None else &(<GrowthKernel> gu).t
            tp[4 * nb + b] = NULL if ctl is None else &(<TailKernel> ctl).t
            tp[5 * nb + b] = NULL if ctu is None else &(<TailKernel> ctu).t
            bp[6 * nb + b] = &kb.t
            sp[7 * nb + b] = _ptr(scales[b])
        kb = kernels[0]
        self.c.tables = &bp[6 * nb]
        self.c.dims = <const int*> cnp.PyArray_DATA(dims)
        self.c.cscale = &sp[7 * nb]
        self.c.d = d
        self.c.n = n
        self.c.k = k
        self.c.nb = nb
        self.c.kind = kind
        self.c.n_logit = n_logit
        self.c.basis = _ptr(basis)
        self.c.index = <const Py_ssize_t*> cnp.PyArray_DATA(padded)
        self.c.ntail_lo = tp
        self.c.ntail_hi = &tp[nb]
        self.c.growth_lo = &gp[2 * nb]
        self.c.growth_hi = &gp[3 * nb]
        self.c.ctail_lo = &tp[4 * nb]
        self.c.ctail_hi = &tp[5 * nb]
        self.c.nopt.rel_tol = nopts.rel_tol
        self.c.nopt.initial_cells = nopts.initial_cells
        self.c.nopt.max_cells = nopts.max_cells
        self.c.nopt.max_level = nopts.max_level
        self.c.nopt.variation = nopts.variation
        self.c.nopt.drop_tol = nopts.drop_tol
        self.c.nopt.zeta_cap = nopts.zeta_cap
        self.c.sopt.rho = sopts.rho
        self.c.sopt.initial_cells = sopts.initial_cells
        self.c.sopt.max_cells = sopts.max_cells
        self.c.sopt.min_width = sopts.min_width
        self.c.sopt.newton_tol = sopts.newton_tol
        self.c.sopt.newton_max = sopts.newton_max
        self.c.sopt.zeta_cap = sopts.zeta_cap
        self.c.sopt.fallback_zeta = sopts.fallback_zeta
        self.c.sopt.max_cuts = sopts.max_cuts
        self.c.sopt.feasibility_tol = sopts.feasibility_tol
        self.c.floor_ = -float(sopts.feasibility_tol) / float(kb.t.h) ** 2
        self.c.series_margin=max(float(sopts.feasibility_tol),float(getattr(sopts,"series_violation_tol",1e-10)))
        self.c.cap_rows = cap
        self.c.stats = _ptr(stats)
        self.c.mass = _ptr(mass)
        self.c.n_rows = R
        self.c.row_lo = _ptr(lower)
        self.c.row_hi = _ptr(upper)
        self.c.row_w = _ptr(weights)
        self.c.phi_rows = _ptr(phi)
        self.c.intervals = intervals
        self.c.theta = _ptr(bufs["theta"])
        self.c.part = _ptr(bufs["part"])
        self.c.parts = _ptr(bufs["parts"])
        self.c.mean = _ptr(bufs["mean"])
        self.c.cov = _ptr(bufs["cov"])
        self.c.logz = _ptr(bufs["logz"])
        self.c.fwork = _ptr(bufs["fwork"])
        self.c.colwork = _ptr(bufs["colwork"])
        self.c.vbuf = _ptr(bufs["vbuf"])
        self.c.rows = _ptr(bufs["rows"])
        self.c.unit = _ptr(bufs["unit"])
        self.c.p_theta = _ptr(bufs["p_theta"])
        self.c.p_rows = _ptr(bufs["p_rows"])
        self.c.f_theta = _ptr(bufs["f_theta"])
        self.c.trace = _ptr(bufs["trace"])
        self.c.gfull = _ptr(bufs["gfull"])
        self.c.ffull = _ptr(bufs["ffull"])
        self.c.cfull = _ptr(bufs["cfull"])
        self.c.fface = _ptr(bufs["fface"])
        self.c.cface = _ptr(bufs["cface"])
        self.c.mwork = _ptr(bufs["mwork"])
        self.c.acc_mean = _ptr(bufs["acc_mean"])
        self.c.acc_fisher = _ptr(bufs["acc_fisher"])
        self.c.acc_missing = _ptr(bufs["acc_missing"])
        self.c.agg = _ptr(bufs["agg"])
        self.c.row_logp = _ptr(bufs["row_logp"])
        self.c.row_mean = _ptr(bufs["row_mean"])
        self.c.row_cov = _ptr(bufs["row_cov"]) if joint and intervals else NULL
        self.c.resp = _ptr(bufs["resp"])
        self.c.gvec = _ptr(bufs["gvec"])
        self.c.nll_part = _ptr(bufs["nll_part"])
        self.c.scores = _ptr(bufs["scores"]) if joint and data.get("scores") else NULL
        self.c.dom_a = _ptr(bufs["dom"])
        self.c.dom_b = &self.c.dom_a[nb]
        self.c.trace_cap = trace
        self.c.trace_n = 0
        self.c.known = _ptr(bufs["known"])
        self.c.nknown = 0
        self.c.gate = 0
        self.c.stop_on_failure = 0
        self.c.armed = 0
        self.c.has_pending = 0
        self.c.failures = 0
        self.c.f_status = 0
        self.c.f_side = -1
        self.c.f_row = -1
        self.c.f_block = -1
        self.c.evaluations = 0

    def __dealloc__(self):
        free(self.c.series)
        free(self._ptrs)

    def begin(self, known_rows, domains, bint stop_on_failure=False):
        """Start a gated run: known joint unit rows ``(m, n)`` and per-block domains."""
        cdef int b
        cdef SeriesSearchKernel sk
        known = np.ascontiguousarray(known_rows, dtype=np.float64)
        if known.ndim != 2 or known.shape[1] != self.c.n:
            raise ValueError("known rows have the wrong shape")
        if len(domains) != self.c.nb:
            raise ValueError("one search domain per block")
        if known.shape[0] == 0:
            known = np.zeros((1, self.c.n))
            self.c.nknown = 0
        else:
            self.c.nknown = known.shape[0]
        self._keep["known_now"] = known
        self.c.known = _ptr(known)
        for b in range(self.c.nb):
            self.c.dom_a[b] = float(domains[b][0])
            self.c.dom_b[b] = float(domains[b][1])
            cache=self._keep["series_caches"][b]
            kernel=None if cache is None else cache.prepare(domains[b])
            self._keep["series_kernels"][b]=kernel
            self.c.series[b]=NULL
            if kernel is not None:
                sk=kernel
                self.c.series[b]=&sk.t
        self.c.armed = 0
        self.c.has_pending = 0
        self.c.stop_on_failure = stop_on_failure
        self.c.gate = 1

    def series_statistics(self):
        """Candidate-search reuse and gate counts for experiments."""
        return dict(hits=self.c.series_hits,full_searches=self.c.full_searches,
                    blocks=[None if c is None else dict(builds=c.builds,
                        tensor_builds=c.tensor_builds,tensor_reuses=c.tensor_reuses,reason=c.reason,
                        kernel=None if c.kernel is None else c.kernel.statistics())
                        for c in self._keep["series_caches"]])

    def end(self):
        """End the gated run."""
        self.c.gate = 0

    def trial(self, x):
        """One evaluation as the Newton loop makes it (tests).

        Returns
        -------
        tuple
            ``(code, nll)``: ``ACCEPTED``, ``STOP``, ``SHAPE``, ``IMPROPER``
            or ``FAILURE``, and the NLL when accepted.
        """
        out = self.evaluate(x)
        return out[0], (out[1] if out[0] == 0 else None)

    def evaluate(self, x):
        """Evaluate at face coordinates ``x`` (gated only inside a run).

        Returns
        -------
        tuple
            ``(code, nll, gradient, metric, fisher, missing, smallest)``;
            the arrays are face-restricted (``missing`` zero for point data).
        """
        cdef int k = self.c.k
        xv = np.ascontiguousarray(x, dtype=np.float64)
        if xv.shape != (k,):
            raise ValueError("x has the wrong length")
        grad = np.zeros(k)
        metric = np.zeros((k, k))
        cdef double nll = NAN, smallest = NAN
        cdef int code = _evaluate(
            &self.c, _ptr(xv), &nll, _ptr(grad), _ptr(metric), &smallest
        )
        if code != 0:
            return code, None, None, None, None, None, None
        bufs = self._keep["bufs"]
        missing = (
            bufs["cface"].copy()
            if self.c.intervals or self.c.kind == K_JOINT
            else np.zeros((k, k))
        )
        return code, nll, grad, metric, bufs["fface"].copy(), missing, smallest

    def _blocks_of(self, flat):
        d, dims = self.c.d, self._keep["dims"]
        return [flat[b * d : b * d + dims[b]].copy() for b in range(self.c.nb)]

    def means(self):
        """``E_b[Phi_b]`` per block (list) and ``log Z_b`` of the last evaluation."""
        bufs = self._keep["bufs"]
        return self._blocks_of(bufs["mean"]), bufs["logz"].copy()

    def scores(self):
        """Per-row joint NLL scores ``(R, n)`` of the last ``KIND_JOINT`` evaluation."""
        if self.c.scores == NULL:
            raise RuntimeError("the objective was built without row scores")
        R, n = self.c.n_rows, self.c.n
        return self._keep["bufs"]["scores"][: R * n].reshape(R, n).copy()

    def row_terms(self):
        """Per-row log densities/masses ``(R, nb)`` and responsibilities.

        Of the last ``KIND_JOINT`` evaluation (``log p_k(row_i)`` normalized
        per component, and ``r_ik``).
        """
        if self.c.kind != K_JOINT:
            raise RuntimeError("row terms belong to the joint objective")
        bufs = self._keep["bufs"]
        R, nb = self.c.n_rows, self.c.nb
        return (
            bufs["row_logp"][: R * nb].reshape(nb, R).T.copy(),
            bufs["resp"][: R * nb].reshape(R, nb).copy(),
        )

    def newton(self, x0, double nll0, gradient0, hessian0, b_matrix, a_packed, sizes,
               a_offsets, q_offsets, reference_dual, row_degrees, blocks0, column_scale,
               options):
        """Run G1's fused Newton loop from ``x0`` (face coordinates).

        ``begin`` must have been called.  ``x0`` is the accepted start with
        NLL ``nll0``, gradient ``gradient0`` ``(k,)`` and metric
        ``hessian0`` ``(k, k)``; the cone description is G1's packed one
        and ``blocks0`` its packed start Gram blocks; ``column_scale`` fixes
        the subproblem column scales (G3-034), ``None`` takes G1's Jacobi
        scaling of each subproblem's Hessian (G3-050); ``options`` is G1's
        ``_NewtonOptions``.

        Returns
        -------
        tuple
            ``(code, x, blocks, dual, iterations, evaluations,
            subproblem_iterations, bound, nll, gradient, metric, mean, fisher,
            missing)``; ``code`` 1-5 are G1's statuses, 6 an objective stop;
            ``nll``, ``gradient`` and ``metric`` are those of ``x`` (face
            coordinates), ``mean`` its ``E_b[Phi_b]`` per block (list) and
            ``fisher``/``missing`` its restricted ``F`` and ``C`` (meaningful
            only when ``x`` moved).

        Raises
        ------
        MemoryError, numpy.linalg.LinAlgError
            From the loop's conic subproblems (as G1's callback loop).
        """
        cdef int k = self.c.k
        if not self.c.gate:
            raise RuntimeError("begin() was not called")
        theta = np.ascontiguousarray(x0, dtype=np.float64).copy()
        grad = np.ascontiguousarray(gradient0, dtype=np.float64).copy()
        hess = np.ascontiguousarray(hessian0, dtype=np.float64).copy()
        blocks = np.ascontiguousarray(blocks0, dtype=np.float64).copy()
        cdef const double[:, ::1] bm = np.ascontiguousarray(b_matrix, dtype=np.float64)
        cdef const double[::1] av = np.ascontiguousarray(a_packed, dtype=np.float64)
        cdef const int[::1] sv = np.ascontiguousarray(sizes, dtype=np.intc)
        cdef const Py_ssize_t[::1] aov = np.ascontiguousarray(a_offsets, dtype=np.intp)
        cdef const Py_ssize_t[::1] qov = np.ascontiguousarray(q_offsets, dtype=np.intp)
        cdef const double[::1] rv = np.ascontiguousarray(
            reference_dual, dtype=np.float64
        )
        cdef const Py_ssize_t[::1] dv = np.ascontiguousarray(row_degrees, dtype=np.intp)
        cdef const double[::1] csv = np.ascontiguousarray(
            np.ones(k) if column_scale is None else column_scale, dtype=np.float64
        )
        cdef const double* csp = NULL
        if column_scale is not None:
            csp = &csv[0]
        cdef int r = bm.shape[0], nb = sv.shape[0]
        cdef Py_ssize_t qtot = blocks.shape[0], na = av.shape[0]
        if (
            theta.shape != (k,) or grad.shape != (k,) or hess.shape != (k, k)
            or bm.shape[1] != k or csv.shape[0] != k or r < 1 or nb < 1
            or rv.shape[0] != r or dv.shape[0] != r
        ):
            raise ValueError("start, scales or cone description do not match the face")
        dual = np.zeros(r)
        cdef double nll = nll0, smallest = NAN, bound = INFINITY
        cdef int its = 0, evals = 0, subits = 0, code
        cdef FaceObjective fo
        cdef double* tp = _ptr(theta)
        cdef double* bp = _ptr(blocks)
        cdef double* gp = _ptr(grad)
        cdef double* hp = _ptr(hess)
        cdef double* dp = _ptr(dual)
        cdef int max_iterations = options.max_iterations
        cdef int max_line_search = options.max_line_search
        cdef double tolerance = options.tolerance
        cdef double certified = options.certified_tolerance
        cdef double floor_ = options.accuracy_floor
        cdef double armijo = options.armijo
        cdef double backtrack = options.backtrack
        fo.ctx = <void*> &self.c
        fo.evaluate = _evaluate
        fo.accept = _accept
        fo.step = _step
        fo.refine = _refine_scalar_qp
        with nogil:
            code = _face_newton_loop(
                k, r, nb, &sv[0], &aov[0], &qov[0], qtot, na,
                &bm[0, 0], &av[0], &rv[0], &dv[0], csp, &fo,
                tp, bp, &nll, gp, hp, &smallest, dp,
                tolerance, certified, floor_, max_iterations, armijo, backtrack,
                max_line_search, 0, &its, &evals, &subits, &bound,
            )
        if code == -2:
            raise np.linalg.LinAlgError("starting Gram block is not positive definite")
        if code == -1:
            raise MemoryError("compiled G3 Newton workspace")
        if code < 0:
            raise np.linalg.LinAlgError("conic QP precomputation did not converge")
        bufs = self._keep["bufs"]
        mean = self._blocks_of(bufs["acc_mean"])
        fisher = bufs["acc_fisher"].copy()
        missing = bufs["acc_missing"].copy()
        return (
            code, theta, blocks, dual, its, evals, subits, bound, nll, grad, hess, mean,
            fisher, missing,
        )

    def pending(self):
        """Cuts queued by the run.

        Returns
        -------
        tuple or None
            ``(block, rows, n_points, theta, domain)``: ``rows[:n_points]``
            are point cuts (``Phi''(x)``, block coordinates), the rest tail
            or bound rows; ``theta`` is the violating trial (joint
            coordinates).
        """
        if not self.c.has_pending:
            return None
        d = self._keep["dims"][self.c.p_block]
        rows = np.array(<double[:self.c.p_nrows * d]> self.c.p_rows, copy=True)
        theta = np.array(<double[:self.c.n]> self.c.p_theta, copy=True)
        domain = (self.c.p_a, self.c.p_b)
        return (
            int(self.c.p_block), rows.reshape(self.c.p_nrows, d), self.c.p_npoints,
            theta, domain,
        )

    def failures(self):
        """Non-routine trial failures since the last call, then reset.

        Returns
        -------
        tuple or None
            ``(count, status, side, diag, theta, row, block)`` of the last
            one (``status`` a normalizer status or ``ALLOCATION``; ``row`` the
            failing interval row, ``-1`` for the normalization itself;
            ``block`` the failing block).
        """
        if not self.c.failures:
            return None
        out = (
            self.c.failures,
            self.c.f_status,
            self.c.f_side,
            tuple(self.c.f_diag[i] for i in range(6)),
            np.array(<double[:self.c.n]> self.c.f_theta, copy=True),
            int(self.c.f_row),
            int(self.c.f_block),
        )
        self.c.failures = 0
        return out

    def traced(self):
        """Normalized points recorded so far (``trace`` of the constructor)."""
        n = self.c.n
        if self.c.trace_n == 0:
            return np.zeros((0, n))
        out = np.array(<double[:self.c.trace_n * n]> self.c.trace, copy=True)
        return out.reshape(-1, n)

    @property
    def evaluations(self):
        """Normalizer evaluations so far."""
        return self.c.evaluations
