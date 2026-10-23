# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled G3 basis kernels: column table, evaluation, enclosures, tail verdicts.

G1 owner replaced: the evaluation role of ``_model/_state_kernels.pyx``
(``_q0``/``_q1``/``_q2`` and ``_polyval_vec``) for the G3 column basis.

* ``BasisKernel`` -- the flat column table of one compiled ``_BasisModel``
  (functions, gated terms, the exact map ``full = M theta`` and its gate-limit
  rows, geometry), built once per model.
* ``eval_point`` / ``potential_point`` -- ``Phi`` and ``q`` with ``x``
  derivatives to order four.  Gates use the cancellation-free lower/upper
  forms of ``basis_eval``.
* ``basis_enclosure_cell`` / ``potential_enclosure_cell`` -- outward-rounded
  interval enclosures over a cell: every arithmetic result widened by a
  relative ``4u`` plus one ulp, every transcendental by ``8u`` plus one ulp
  (``8 n u`` for integer powers ``n``), exact zeros kept exact.  Numerical
  evidence (G3-028), not an arbitrary-precision certificate.
* ``TailKernel`` / ``tail_verdict`` -- the sign of ``d^k q / d zeta^k`` beyond
  a cutoff from coefficient-linear asymptotic atoms (``basis_tails``).

The module must be built without floating-point contraction (``setup.py``
passes ``-ffp-contract=off``): the rounding margins assume one rounding per
operation.
"""

import numpy as np
from fractions import Fraction
cimport numpy as cnp
from libc.stdint cimport uint64_t
from libc.string cimport memcpy

from libc.math cimport (
    INFINITY,
    NAN,
    sqrt,
    exp,
    fabs,
    fabsl,
    isfinite,
    isnan,
    log,
    log1p,
    pow,
    tanh,
)
from libc.float cimport LDBL_EPSILON

cnp.import_array()

cdef double _U = 1.1102230246251565e-16  # eps / 2
cdef double _ARITH = 4.0
cdef double _TRANS = 8.0
cdef double _LOG_UNDERFLOW = -575.0
cdef double _LOGE2 = 0.6931471805599453

cdef struct LongIv:
    long double lo
    long double hi

cdef inline Iv _polynomial_curvature_point(
    const BasisTable* t, const PotentialPrep* p, double x
) noexcept nogil:
    """Tightly enclose the ungated polynomial part of ``q''(x)``.

    Evaluate the polynomial as a single Horner chain in extended precision.
    Coefficients are accumulated by degree first, so cancellation between
    columns happens before interval rounding.  The explicit long-double
    allowance covers each arithmetic operation; conversion back to binary64
    is widened by one ulp.
    """
    cdef int degree, j, col, max_degree = 0, map_count, unit_col
    cdef long double eps = <long double>LDBL_EPSILON
    cdef long double scale, e, p0, p1, p2, p3, mag, coeff_mag
    cdef long double slot_lo, slot_hi, mapped, mapped_abs, mapped_error, term_ld
    cdef double map_value, unit_sign
    cdef LongIv z, acc, coeff, nxt
    cdef Iv slot, out
    cdef bint have = False
    z.lo = ((<long double>x - <long double>t.c) / <long double>t.h)
    e = <long double>8.0 * eps * (
        fabsl(z.lo) + (fabsl(<long double>x) + fabsl(<long double>t.c))
        / fabsl(<long double>t.h) + 1.0
    )
    z.lo -= e
    z.hi = ((<long double>x - <long double>t.c) / <long double>t.h) + e
    for j in range(t.nf):
        if (t.kind[j] == FN_MONOMIAL and not p.gated[j]
                and t.degree[j] > max_degree):
            max_degree = t.degree[j]
    acc.lo = 0.0
    acc.hi = 0.0
    for degree in range(max_degree, 1, -1):
        coeff.lo = 0.0
        coeff.hi = 0.0
        coeff_mag = 0.0
        for j in range(t.nf):
            if (t.kind[j] == FN_MONOMIAL and t.degree[j] == degree
                    and not p.gated[j]):
                slot = p.slot[3 * j]
                slot_lo = <long double>slot.lo
                slot_hi = <long double>slot.hi
                if t.orth_degree:
                    mapped = 0.0
                    mapped_abs = 0.0
                    mapped_error = 0.0
                    for col in range(t.d):
                        term_ld = (<long double>t.cmap[3*j*t.d + col]
                                   + <long double>t.cmap_low[3*j*t.d + col])
                        term_ld *= <long double>p.theta[col]
                        mapped += term_ld
                        mapped_abs += fabsl(term_ld)
                        mapped_error += (<long double>2.0 * _U
                            * fabsl(<long double>t.cmap_low[3*j*t.d + col]
                                    * <long double>p.theta[col]))
                    mapped_error += (<long double>(8 * (t.d + 2)) * eps * mapped_abs)
                    slot_lo = mapped - mapped_error
                    slot_hi = mapped + mapped_error
                else:
                    map_count = 0
                    unit_col = -1
                    unit_sign = 0.0
                    for col in range(t.d):
                        map_value = t.cmap[3*j*t.d + col]
                        if map_value != 0.0:
                            map_count += 1
                            unit_col = col
                            unit_sign = map_value
                    if map_count == 1 and (unit_sign == 1.0 or unit_sign == -1.0):
                        slot_lo = <long double>unit_sign * <long double>p.theta[unit_col]
                        slot_hi = slot_lo
                scale = (<long double>degree * <long double>(degree - 1)
                         / (<long double>t.h * <long double>t.h))
                e = <long double>8.0 * eps * fabsl(scale)
                p0 = slot_lo * (scale - e)
                p1 = slot_lo * (scale + e)
                p2 = slot_hi * (scale - e)
                p3 = slot_hi * (scale + e)
                e = p0
                if p1 < e: e = p1
                if p2 < e: e = p2
                if p3 < e: e = p3
                coeff.lo += e
                e = p0
                if p1 > e: e = p1
                if p2 > e: e = p2
                if p3 > e: e = p3
                coeff.hi += e
                coeff_mag += fabsl(p0) + fabsl(p1) + fabsl(p2) + fabsl(p3)
        e = <long double>8.0 * eps * coeff_mag
        coeff.lo -= e
        coeff.hi += e
        if degree == max_degree and not have:
            acc = coeff
            have = True
            continue
        p0 = acc.lo * z.lo
        p1 = acc.lo * z.hi
        p2 = acc.hi * z.lo
        p3 = acc.hi * z.hi
        nxt.lo = (p0 if p0 < p1 else p1)
        if p2 < nxt.lo: nxt.lo = p2
        if p3 < nxt.lo: nxt.lo = p3
        nxt.hi = (p0 if p0 > p1 else p1)
        if p2 > nxt.hi: nxt.hi = p2
        if p3 > nxt.hi: nxt.hi = p3
        mag = fabsl(nxt.lo) + fabsl(nxt.hi) + fabsl(coeff.lo) + fabsl(coeff.hi)
        e = <long double>8.0 * eps * mag
        acc.lo = nxt.lo + coeff.lo - e
        acc.hi = nxt.hi + coeff.hi + e
    if not have:
        return _iv(0.0, 0.0)
    out.lo = _next_down(<double>acc.lo)
    out.hi = _next_up(<double>acc.hi)
    return out


# --------------------------------------------------------------------------
# scalar helpers (NumPy semantics: NaN propagates through max/min)
# --------------------------------------------------------------------------

cdef inline double npmax(double a, double b) noexcept nogil:
    if isnan(a) or isnan(b):
        return NAN
    return a if a >= b else b


cdef inline double npmin(double a, double b) noexcept nogil:
    if isnan(a) or isnan(b):
        return NAN
    return a if a <= b else b


cdef inline double expit(double x) noexcept nogil:
    """``1 / (1 + exp(-x))`` (SciPy's formula)."""
    return 1.0 / (1.0 + exp(-x))


cdef inline double logaddexp(double x, double y) noexcept nogil:
    """NumPy's ``logaddexp``."""
    cdef double tmp
    if x == y:
        return x + _LOGE2
    tmp = x - y
    if tmp > 0.0:
        return x + log1p(exp(-tmp))
    if tmp <= 0.0:
        return y + log1p(exp(tmp))
    return tmp


cdef inline double _falling(int n, int k) noexcept nogil:
    cdef double out = 1.0
    cdef int i
    for i in range(n - k + 1, n + 1):
        out *= i
    return out


cdef inline double _comb(int k, int i) noexcept nogil:
    cdef double out = 1.0
    cdef int j
    for j in range(1, i + 1):
        out = out * (k - i + j) / j
    return out


# --------------------------------------------------------------------------
# interval arithmetic
# --------------------------------------------------------------------------

cdef inline Iv _iv(double lo, double hi) noexcept nogil:
    cdef Iv r
    r.lo = lo
    r.hi = hi
    return r


cdef inline double _next_up(double x) noexcept nogil:
    """``nextafter(x, +inf)`` by stepping the bit pattern (inlined).

    The enclosure arithmetic widens every result by one ulp; glibc's
    ``nextafter`` call was 6.5% of a degree-6 fit (callgrind, G3-050).
    """
    cdef uint64_t bits
    if x != x or x == INFINITY:
        return x
    if x == 0.0:
        return 4.9406564584124654e-324
    memcpy(&bits, &x, sizeof(double))
    if x > 0.0:
        bits += 1
    else:
        bits -= 1
    memcpy(&x, &bits, sizeof(double))
    return x


cdef inline double _next_down(double x) noexcept nogil:
    """``nextafter(x, -inf)`` (``_next_up`` mirrored)."""
    return -_next_up(-x)


cdef inline Iv _widen(double lo, double hi, double f) noexcept nogil:
    cdef Iv r
    r.lo = _next_down(lo - f * _U * fabs(lo))
    r.hi = _next_up(hi + f * _U * fabs(hi))
    return r


cdef inline bint _zero(Iv a) noexcept nogil:
    return a.lo == 0.0 and a.hi == 0.0


cdef inline Iv _add(Iv a, Iv b) noexcept nogil:
    if _zero(a) and _zero(b):
        return _iv(0.0, 0.0)
    return _widen(a.lo + b.lo, a.hi + b.hi, _ARITH)


cdef inline Iv _add_const(Iv a, double c) noexcept nogil:
    if _zero(a) and c == 0.0:
        return _iv(0.0, 0.0)
    return _widen(a.lo + c, a.hi + c, _ARITH)


cdef inline Iv _scale(double c, Iv a) noexcept nogil:
    if _zero(a) or c == 0.0:
        return _iv(0.0, 0.0)
    if c >= 0.0:
        return _widen(c * a.lo, c * a.hi, _ARITH)
    return _widen(c * a.hi, c * a.lo, _ARITH)


cdef inline double _nz(double v) noexcept nogil:
    # 0 * inf only arises with an exactly zero factor interval; the product is 0.
    return 0.0 if isnan(v) else v


cdef inline Iv _mul(Iv a, Iv b) noexcept nogil:
    cdef double p1, p2, p3, p4, lo, hi
    if _zero(a) or _zero(b):
        return _iv(0.0, 0.0)
    p1 = _nz(a.lo * b.lo)
    p2 = _nz(a.lo * b.hi)
    p3 = _nz(a.hi * b.lo)
    p4 = _nz(a.hi * b.hi)
    lo = p1
    hi = p1
    if p2 < lo:
        lo = p2
    if p3 < lo:
        lo = p3
    if p4 < lo:
        lo = p4
    if p2 > hi:
        hi = p2
    if p3 > hi:
        hi = p3
    if p4 > hi:
        hi = p4
    return _widen(lo, hi, _ARITH)


cdef inline Iv _neg(Iv a) noexcept nogil:
    return _iv(-a.hi, -a.lo)


cdef inline Iv _ipow(Iv a, int n) noexcept nogil:
    cdef double plo, phi, lo, hi
    if n == 0:
        return _iv(1.0, 1.0)
    if n == 1:
        return a
    plo = pow(a.lo, n)
    phi = pow(a.hi, n)
    if n % 2 == 1:
        return _widen(plo, phi, _TRANS * n)
    if a.lo >= 0.0:
        lo = plo
        hi = phi
    elif a.hi <= 0.0:
        lo = phi
        hi = plo
    else:
        lo = 0.0
        hi = npmax(plo, phi)
    return _widen(lo, hi, _TRANS * n)


cdef inline Iv _fpow_positive(Iv a, double b) noexcept nogil:
    cdef double plo = pow(a.lo, b)
    cdef double phi = pow(a.hi, b)
    if b >= 0.0:
        return _widen(plo, phi, _TRANS)
    return _widen(phi, plo, _TRANS)


cdef inline Iv _tanh_iv(Iv a) noexcept nogil:
    return _widen(tanh(a.lo), tanh(a.hi), _TRANS)


cdef inline Iv _p_iv(Iv a) noexcept nogil:
    return _widen(2.0 * expit(2.0 * a.lo), 2.0 * expit(2.0 * a.hi), _TRANS)


cdef inline Iv _m_iv(Iv a) noexcept nogil:
    return _widen(2.0 * expit(-2.0 * a.hi), 2.0 * expit(-2.0 * a.lo), _TRANS)


cdef inline Iv _log_iv(Iv a) noexcept nogil:
    return _widen(log(a.lo), log(a.hi), _TRANS)


cdef inline Iv _exp_iv(Iv a) noexcept nogil:
    return _widen(exp(a.lo), exp(a.hi), _TRANS)


# --------------------------------------------------------------------------
# column table
# --------------------------------------------------------------------------

cdef class BasisKernel:
    """Flat column table of one compiled ``_BasisModel`` (built once).

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    """

    def __cinit__(self, model):
        geometry, request = model.geometry, model.request
        functions = model.functions
        nf = len(functions)
        d = model.n_coefficients
        kinds = {"monomial": FN_MONOMIAL, "log": FN_LOG, "soft_power": FN_SOFT,
                 "exponential": FN_EXP}
        kind = np.array([kinds[f.kind] for f in functions], dtype=np.intc)
        side = np.array(
            [SIDE_UPPER if f.side == "upper" else SIDE_LOWER for f in functions],
            dtype=np.intc,
        )
        degree = np.array([f.degree or 0 for f in functions], dtype=np.intc)
        power = np.array([f.power or 0.0 for f in functions], dtype=np.float64)
        delta = np.array([f.delta or 0.0 for f in functions], dtype=np.float64)
        rate = np.array([f.rate or 0.0 for f in functions], dtype=np.float64)
        location = np.array([f.location or 0.0 for f in functions], dtype=np.float64)
        starts, fns, g0, g1, g2, glow, gup = [0], [], [], [], [], [], []
        log_cols = {"lower": [], "upper": []}
        for col, column in enumerate(model.columns):
            first = functions[column.terms[0][0]]
            if first.kind == "log":
                log_cols[first.side].append(col)
            for j, gate in column.terms:
                fns.append(j)
                g0.append(float(gate[0]))
                g1.append(float(gate[1]))
                g2.append(float(gate[2]))
                glow.append(float(gate[0] - gate[1] + gate[2]))
                gup.append(float(gate[0] + gate[1] - gate[2]))
            starts.append(len(fns))
        for s in ("lower", "upper"):
            if len(log_cols[s]) > 1:
                raise ValueError(f"more than one {s} log column")
        cmap = np.ascontiguousarray(model.coefficient_map, dtype=np.float64)
        plan = getattr(model, "polynomial_coordinates", None)
        cmap_low = np.zeros_like(cmap)
        if plan is not None:
            for i, row in enumerate(model.coefficient_map_exact):
                for j, value in enumerate(row):
                    cmap_low[i, j] = float(value - Fraction(float(cmap[i, j])))
        orth_degree = (np.zeros(d, dtype=np.intc) if plan is None
                       else np.ascontiguousarray(plan.column_degree))
        orth_fn = (np.zeros(d, dtype=np.intc) if plan is None
                   else np.ascontiguousarray(plan.column_fn))
        orth_norm = np.ones(1) if plan is None else np.ascontiguousarray(plan.norms)
        orth_reference = np.zeros(1) if plan is None else np.ascontiguousarray(plan.reference)
        orth_recurrence = np.zeros((1, 3)) if plan is None else plan.recurrence
        orth_map = np.zeros((1, d)) if plan is None else plan.polynomial_map
        ma, mb, mc = cmap[0::3], cmap[1::3], cmap[2::3]
        lim_low = np.ascontiguousarray(ma + -1.0 * mb + 1.0 * mc)
        lim_up = np.ascontiguousarray(ma + 1.0 * mb + -1.0 * mc)
        absrow = np.abs(ma) + np.abs(mb) + np.abs(mc)
        lim_low_abs = np.ascontiguousarray(np.where(lim_low != 0.0, absrow, 0.0))
        lim_up_abs = np.ascontiguousarray(np.where(lim_up != 0.0, absrow, 0.0))
        nonneg = np.array([bool(c.nonnegative) for c in model.columns], dtype=np.intc)
        fn_used = np.zeros(nf, dtype=np.intc)
        fn_used[np.unique(np.array(fns, dtype=np.intc))] = 1
        arrays = dict(
            kind=kind, side=side, degree=degree, power=power, delta=delta,
            rate=rate, location=location,
            col_start=np.array(starts, dtype=np.intc),
            term_fn=np.array(fns, dtype=np.intc),
            term_g0=np.array(g0, dtype=np.float64),
            term_g1=np.array(g1, dtype=np.float64),
            term_g2=np.array(g2, dtype=np.float64),
            term_glow=np.array(glow, dtype=np.float64),
            term_gup=np.array(gup, dtype=np.float64),
            cmap=cmap, cmap_low=cmap_low, orth_degree=orth_degree,
            orth_fn=orth_fn, orth_norm=orth_norm, orth_reference=orth_reference,
            orth_recurrence=orth_recurrence, orth_map=orth_map,
            lim_low=lim_low, lim_up=lim_up,
            lim_low_abs=lim_low_abs, lim_up_abs=lim_up_abs, nonneg=nonneg,
            fn_used=fn_used,
        )
        self._keep = arrays
        self.t.nf = nf
        self.t.d = d
        self.t.kind = <const int*> cnp.PyArray_DATA(arrays["kind"])
        self.t.side = <const int*> cnp.PyArray_DATA(arrays["side"])
        self.t.degree = <const int*> cnp.PyArray_DATA(arrays["degree"])
        self.t.power = <const double*> cnp.PyArray_DATA(arrays["power"])
        self.t.delta = <const double*> cnp.PyArray_DATA(arrays["delta"])
        self.t.rate = <const double*> cnp.PyArray_DATA(arrays["rate"])
        self.t.location = <const double*> cnp.PyArray_DATA(arrays["location"])
        self.t.col_start = <const int*> cnp.PyArray_DATA(arrays["col_start"])
        self.t.term_fn = <const int*> cnp.PyArray_DATA(arrays["term_fn"])
        self.t.term_g0 = <const double*> cnp.PyArray_DATA(arrays["term_g0"])
        self.t.term_g1 = <const double*> cnp.PyArray_DATA(arrays["term_g1"])
        self.t.term_g2 = <const double*> cnp.PyArray_DATA(arrays["term_g2"])
        self.t.term_glow = <const double*> cnp.PyArray_DATA(arrays["term_glow"])
        self.t.term_gup = <const double*> cnp.PyArray_DATA(arrays["term_gup"])
        self.t.cmap = <const double*> cnp.PyArray_DATA(arrays["cmap"])
        self.t.cmap_low = <const double*> cnp.PyArray_DATA(arrays["cmap_low"])
        self.t.orth_degree = 0 if plan is None else plan.degree
        self.t.polynomial_point = all(
            functions[index].kind in ("monomial", "log") and not gate[1] and not gate[2]
            for column in model.columns for index, gate in column.terms
        )
        self.t.orth_scale = 1.0 if plan is None else plan.scale
        self.t.orth_alpha = 0.0 if plan is None else plan.alpha
        self.t.orth_origin = 0.0 if plan is None else plan.origin
        self.t.orth_recurrence = <const double*> cnp.PyArray_DATA(arrays["orth_recurrence"])
        self.t.orth_map = <const double*> cnp.PyArray_DATA(arrays["orth_map"])
        self.t.orth_column_degree = <const int*> cnp.PyArray_DATA(arrays["orth_degree"])
        self.t.orth_column_fn = <const int*> cnp.PyArray_DATA(arrays["orth_fn"])
        self.t.orth_norm = <const double*> cnp.PyArray_DATA(arrays["orth_norm"])
        self.t.orth_reference = <const double*> cnp.PyArray_DATA(arrays["orth_reference"])
        self.t.lim_low = <const double*> cnp.PyArray_DATA(arrays["lim_low"])
        self.t.lim_up = <const double*> cnp.PyArray_DATA(arrays["lim_up"])
        self.t.lim_low_abs = <const double*> cnp.PyArray_DATA(arrays["lim_low_abs"])
        self.t.lim_up_abs = <const double*> cnp.PyArray_DATA(arrays["lim_up_abs"])
        self.t.nonneg = <const int*> cnp.PyArray_DATA(arrays["nonneg"])
        self.t.fn_used = <const int*> cnp.PyArray_DATA(arrays["fn_used"])
        self.t.c = float(geometry.center)
        self.t.h = float(geometry.scale)
        self.t.lower = float(request.lower)
        self.t.upper = float(request.upper)
        self.t.has_l = geometry.lower_location is not None
        self.t.has_u = geometry.upper_location is not None
        self.t.xl = float(geometry.lower_location) if self.t.has_l else 0.0
        self.t.al = float(geometry.lower_steepness) if self.t.has_l else 0.0
        self.t.xu = float(geometry.upper_location) if self.t.has_u else 0.0
        self.t.au = float(geometry.upper_steepness) if self.t.has_u else 0.0
        if self.t.has_l and self.t.has_u:
            self.t.mid = 0.5 * (self.t.xl + self.t.xu)
        elif self.t.has_l:
            self.t.mid = self.t.xl
        elif self.t.has_u:
            self.t.mid = self.t.xu
        else:
            self.t.mid = INFINITY
        self.t.log_col_lower = log_cols["lower"][0] if log_cols["lower"] else -1
        self.t.log_col_upper = log_cols["upper"][0] if log_cols["upper"] else -1

    @property
    def n_coefficients(self):
        """Number of columns ``d``."""
        return self.t.d


# --------------------------------------------------------------------------
# point evaluation
# --------------------------------------------------------------------------

cdef inline double scaled_z(const BasisTable* t, double x) noexcept nogil:
    """``(x - c) / h`` without avoidable overflow (G1 ``_safe_scaled_difference``)."""
    cdef double out = (x - t.c) / t.h
    cdef double mag
    if not isfinite(out) and isfinite(x):
        mag = npmax(npmax(fabs(x), fabs(t.c)), fabs(t.h))
        out = ((x / mag) - (t.c / mag)) / (t.h / mag)
    return out


# Powers ``z**i`` tabulated per point for monomials (``_function_point``).
cdef enum:
    _ZP = 64


cdef inline void _tanh_point(
    double u, double c, int order, double* tv
) noexcept nogil:
    """``tv = [t, p, m, t', t'', t''', t'''']`` for ``t = tanh(u)``, ``u' = c``."""
    cdef double p = 2.0 * expit(2.0 * u)
    cdef double m = 2.0 * expit(-2.0 * u)
    cdef double t, w, t2
    tv[1] = p
    tv[2] = m
    if order == 0:
        return
    t = tanh(u)
    w = p * m
    t2 = t * t
    tv[0] = t
    tv[3] = c * w
    tv[4] = -2.0 * (c * c) * t * w
    tv[5] = -2.0 * (c * c * c) * w * (1.0 - 3.0 * t2)
    tv[6] = 8.0 * (c * c * c * c) * t * w * (2.0 - 3.0 * t2)


cdef inline void _flat_point(double* tv) noexcept nogil:
    tv[0] = 1.0
    tv[1] = 2.0
    tv[2] = 0.0
    tv[3] = 0.0
    tv[4] = 0.0
    tv[5] = 0.0
    tv[6] = 0.0


cdef void _function_point(
    const BasisTable* t,
    int j,
    double x,
    double z,
    const double* zp,
    int order,
    double* f,
) noexcept nogil:
    """``f[0..order]``: ``x`` derivatives of function ``j``.

    ``zp[i] = z**i`` for ``i < _ZP`` (higher monomial powers use ``pow``).
    """
    cdef int k, n
    cdef double h = t.h
    cdef double dist, inv, lam, base, a, s, z2, dz, ih
    cdef double v[5]
    if t.kind[j] == FN_MONOMIAL:
        n = t.degree[j]
        f[0] = zp[n] if n < _ZP else pow(z, n)
        ih = 1.0
        for k in range(1, order + 1):
            ih = ih / h
            if k > n:
                f[k] = 0.0
            else:
                f[k] = _falling(n, k) * (
                    zp[n - k] if n - k < _ZP else pow(z, n - k)
                ) * ih
        return
    if t.kind[j] == FN_LOG:
        if t.side[j] == SIDE_LOWER:
            dist = x - t.lower
            inv = -1.0 / dist
        else:
            dist = t.upper - x
            inv = 1.0 / dist
        f[0] = -log(dist)
        # d^k/dx^k of -log(dist): (k - 1)! inv**k
        base = 1.0
        for k in range(1, order + 1):
            base = base * inv * (k - 1 if k > 1 else 1)
            f[k] = base
        return
    if t.kind[j] == FN_EXP:
        lam = t.rate[j]
        if t.side[j] == SIDE_LOWER:
            base = exp(-lam * (x - t.location[j]))
            lam = -lam
        else:
            base = exp(lam * (x - t.location[j]))
        f[0] = base
        for k in range(1, order + 1):
            base = base * lam
            f[k] = base
        return
    a = 0.5 * t.power[j]
    dz = t.delta[j] / (h * h)
    s = z * z + dz
    z2 = z * z
    v[0] = pow(s, a)
    if order == 0:
        f[0] = v[0]
        return
    v[1] = 2.0 * a * z * pow(s, a - 1.0)
    v[2] = 2.0 * a * pow(s, a - 1.0) + 4.0 * a * (a - 1.0) * z2 * pow(s, a - 2.0)
    v[3] = (
        12.0 * a * (a - 1.0) * z * pow(s, a - 2.0)
        + 8.0 * a * (a - 1.0) * (a - 2.0) * z * z2 * pow(s, a - 3.0)
    )
    v[4] = (
        12.0 * a * (a - 1.0) * pow(s, a - 2.0)
        + 48.0 * a * (a - 1.0) * (a - 2.0) * z2 * pow(s, a - 3.0)
        + 16.0 * a * (a - 1.0) * (a - 2.0) * (a - 3.0) * z2 * z2 * pow(s, a - 4.0)
    )
    for k in range(order + 1):
        f[k] = v[k] / pow(h, k)


cdef void eval_point(
    const BasisTable* t, double x, int order, double* fwork, double* out
) noexcept nogil:
    """``out[k * d + c] = Phi_c^(k)(x)`` for ``k <= order``.

    ``fwork`` holds ``5 nf`` doubles.
    """
    cdef int d = t.d
    cdef int j, k, i, col, term, n
    cdef double z = scaled_z(t, x)
    cdef double tl[7]
    cdef double tu[7]
    cdef double g[5]
    cdef double zp[_ZP]
    cdef double a1, a2, acc
    cdef bint lower_side = x < t.mid
    cdef bint gated, need_l = False, need_u = False
    cdef int nmax = 0
    cdef double argument
    for term in range(t.col_start[d]):
        if t.term_g1[term] != 0.0:
            need_l = True
        if t.term_g2[term] != 0.0:
            need_u = True
    if t.has_l and need_l:
        _tanh_point(t.al * (x - t.xl), t.al, order, tl)
    else:
        _flat_point(tl)
    if t.has_u and need_u:
        _tanh_point(t.au * (t.xu - x), -t.au, order, tu)
    else:
        _flat_point(tu)
    for k in range((order + 1) * d):
        out[k] = 0.0
    if t.orth_degree:
        argument = (x - t.orth_origin) / t.orth_scale
        for k in range(order + 1):
            fwork[k] = 1.0 if k == 0 else 0.0
        for n in range(t.orth_degree):
            for k in range(order + 1):
                fwork[5*(n+1) + k] = (
                    (t.orth_recurrence[3*n] * argument + t.orth_recurrence[3*n+1]) * fwork[5*n+k]
                    + (k * t.orth_recurrence[3*n] * fwork[5*n+k-1] if k else 0.0)
                    - (t.orth_recurrence[3*n+2] * fwork[5*(n-1)+k] if n else 0.0)
                )
        for col in range(d):
            if t.orth_column_degree[col] == 0:
                continue
            for n in range(t.orth_degree + 1):
                a1 = t.orth_map[(3*n+1)*d+col]
                a2 = t.orth_map[(3*n+2)*d+col]
                if a1 == 0.0 and a2 == 0.0:
                    acc = t.orth_map[3*n*d+col]
                    if acc != 0.0:
                        for k in range(order + 1):
                            out[k*d+col] += fwork[5*n+k] * acc / pow(t.orth_scale,k)
                    continue
                if lower_side:
                    g[0] = (t.orth_map[3*n*d+col] - a1 + a2) + a1 * tl[1] - a2 * tu[2]
                else:
                    g[0] = (t.orth_map[3*n*d+col] + a1 - a2) - a1 * tl[2] + a2 * tu[1]
                for k in range(order):
                    g[k+1] = a1 * tl[3+k] + a2 * tu[3+k]
                for k in range(order + 1):
                    for i in range(k + 1):
                        out[k*d+col] += _comb(k,i) * fwork[5*n+i] / pow(t.orth_scale,i) * g[k-i]
    for j in range(t.nf):
        if t.fn_used[j] and t.kind[j] == FN_MONOMIAL and t.degree[j] > nmax:
            nmax = t.degree[j]
    if nmax >= _ZP:
        nmax = _ZP - 1
    zp[0] = 1.0
    for i in range(1, nmax + 1):
        zp[i] = zp[i - 1] * z
    for j in range(t.nf):
        if t.fn_used[j]:
            _function_point(t, j, x, z, zp, order, &fwork[5 * j])
    for col in range(d):
        for term in range(t.col_start[col], t.col_start[col + 1]):
            j = t.term_fn[term]
            if t.orth_degree and t.orth_column_degree[col] > 0 and t.kind[j] == FN_MONOMIAL:
                continue
            a1 = t.term_g1[term]
            a2 = t.term_g2[term]
            gated = a1 != 0.0 or a2 != 0.0
            if lower_side:
                g[0] = t.term_glow[term] + a1 * tl[1] - a2 * tu[2]
            else:
                g[0] = t.term_gup[term] - a1 * tl[2] + a2 * tu[1]
            for k in range(order):
                g[k + 1] = a1 * tl[3 + k] + a2 * tu[3 + k]
            for k in range(order + 1):
                if gated:
                    acc = 0.0
                    for i in range(k + 1):
                        acc = acc + _comb(k, i) * fwork[5 * j + i] * g[k - i]
                    out[k * d + col] += acc
                else:
                    out[k * d + col] += fwork[5 * j + k] * g[0]


cdef void potential_point(
    const BasisTable* t,
    const double* theta,
    double x,
    int order,
    double* fwork,
    double* colwork,
    double* q,
) noexcept nogil:
    """``q[k] = sum_c theta_c Phi_c^(k)(x)`` over columns with ``theta_c != 0``.

    ``fwork`` holds ``5 nf`` doubles, ``colwork`` ``(order + 1) d``.
    """
    cdef int d = t.d
    cdef int k, col
    cdef double acc
    eval_point(t, x, order, fwork, colwork)
    for k in range(order + 1):
        acc = 0.0
        for col in range(d):
            if theta[col] != 0.0:
                acc = acc + colwork[k * d + col] * theta[col]
        q[k] = acc


# --------------------------------------------------------------------------
# cell enclosures
# --------------------------------------------------------------------------

cdef inline void _tanh_cell(Iv u, double c, int order, Iv* tv) noexcept nogil:
    """``tv = [t, p, m, t', ..., t^(order)]`` enclosures over ``u``."""
    cdef Iv t, p, m, w, t2
    t = _tanh_iv(u)
    p = _p_iv(u)
    m = _m_iv(u)
    t = _iv(npmax(t.lo, -1.0), npmin(t.hi, 1.0))
    p = _iv(npmax(p.lo, 0.0), npmin(p.hi, 2.0))
    m = _iv(npmax(m.lo, 0.0), npmin(m.hi, 2.0))
    w = _mul(p, m)
    w = _iv(npmax(w.lo, 0.0), npmin(w.hi, 1.0))
    t2 = _ipow(t, 2)
    tv[0] = t
    tv[1] = p
    tv[2] = m
    if order >= 1:
        tv[3] = _scale(c, w)
    if order >= 2:
        tv[4] = _scale(-2.0 * pow(c, 2), _mul(t, w))
    if order >= 3:
        tv[5] = _scale(-2.0 * pow(c, 3), _mul(w, _add_const(_scale(-3.0, t2), 1.0)))
    if order >= 4:
        tv[6] = _scale(
            8.0 * pow(c, 4), _mul(_mul(t, w), _add_const(_scale(-3.0, t2), 2.0))
        )


cdef inline void _flat_cell(Iv* tv) noexcept nogil:
    tv[0] = _iv(1.0, 1.0)
    tv[1] = _iv(2.0, 2.0)
    tv[2] = _iv(0.0, 0.0)
    tv[3] = _iv(0.0, 0.0)
    tv[4] = _iv(0.0, 0.0)
    tv[5] = _iv(0.0, 0.0)
    tv[6] = _iv(0.0, 0.0)


cdef void _function_cell(
    const BasisTable* t, int j, Iv x, Iv z, int order, Iv* f
) noexcept nogil:
    """Enclosures ``f[0..order]`` of function ``j`` over the cell ``x``."""
    cdef int k, n
    cdef double h = t.h
    cdef double lam, rate, a, dz
    cdef double signs[5]
    cdef Iv d, arg, base, z2, z3, z4, s
    if t.kind[j] == FN_MONOMIAL:
        n = t.degree[j]
        for k in range(order + 1):
            if k > n:
                f[k] = _iv(0.0, 0.0)
            else:
                f[k] = _scale(
                    (_falling(n, k) if k else 1.0) / pow(h, k), _ipow(z, n - k)
                )
        return
    if t.kind[j] == FN_LOG:
        if t.side[j] == SIDE_LOWER:
            d = _widen(x.lo - t.lower, x.hi - t.lower, _ARITH)
            signs[0] = -1.0
            signs[1] = -1.0
            signs[2] = 1.0
            signs[3] = -2.0
            signs[4] = 6.0
        else:
            d = _widen(t.upper - x.hi, t.upper - x.lo, _ARITH)
            signs[0] = -1.0
            signs[1] = 1.0
            signs[2] = 1.0
            signs[3] = 2.0
            signs[4] = 6.0
        d = _iv(npmax(d.lo, 0.0), d.hi)
        f[0] = _scale(-1.0, _log_iv(d))
        for k in range(1, order + 1):
            f[k] = _scale(signs[k], _fpow_positive(d, -<double> k))
        return
    if t.kind[j] == FN_EXP:
        lam = t.rate[j]
        if t.side[j] == SIDE_LOWER:
            arg = _scale(-lam, _add_const(x, -t.location[j]))
            rate = -lam
        else:
            arg = _scale(lam, _add_const(x, -t.location[j]))
            rate = lam
        base = _exp_iv(arg)
        for k in range(order + 1):
            f[k] = _scale(pow(rate, k), base)
        return
    a = 0.5 * t.power[j]
    dz = t.delta[j] / (h * h)
    z2 = _ipow(z, 2)
    s = _add_const(z2, dz)
    s = _iv(npmax(s.lo, dz), s.hi)  # s >= dz > 0 exactly
    f[0] = _fpow_positive(s, a)
    if order >= 1:
        f[1] = _scale(2.0 * a, _mul(z, _fpow_positive(s, a - 1.0)))
    if order >= 2:
        f[2] = _add(
            _scale(2.0 * a, _fpow_positive(s, a - 1.0)),
            _scale(4.0 * a * (a - 1.0), _mul(z2, _fpow_positive(s, a - 2.0))),
        )
    if order >= 3:
        z3 = _ipow(z, 3)
        f[3] = _add(
            _scale(12.0 * a * (a - 1.0), _mul(z, _fpow_positive(s, a - 2.0))),
            _scale(
                8.0 * a * (a - 1.0) * (a - 2.0), _mul(z3, _fpow_positive(s, a - 3.0))
            ),
        )
    if order >= 4:
        z4 = _ipow(z, 4)
        f[4] = _add(
            _add(
                _scale(12.0 * a * (a - 1.0), _fpow_positive(s, a - 2.0)),
                _scale(
                    48.0 * a * (a - 1.0) * (a - 2.0),
                    _mul(z2, _fpow_positive(s, a - 3.0)),
                ),
            ),
            _scale(
                16.0 * a * (a - 1.0) * (a - 2.0) * (a - 3.0),
                _mul(z4, _fpow_positive(s, a - 4.0)),
            ),
        )
    for k in range(order + 1):
        f[k] = _scale(1.0 / pow(h, k), f[k])


cdef inline void _cell_setup(
    const BasisTable* t, double lo, double hi, int top, Iv* x, Iv* z, Iv* tl, Iv* tu
) noexcept nogil:
    x[0] = _iv(lo, hi)
    z[0] = _scale(1.0 / t.h, _add_const(x[0], -t.c))
    if t.has_l:
        _tanh_cell(_scale(t.al, _add_const(x[0], -t.xl)), t.al, top, tl)
    else:
        _flat_cell(tl)
    if t.has_u:
        _tanh_cell(_scale(-t.au, _add_const(x[0], -t.xu)), -t.au, top, tu)
    else:
        _flat_cell(tu)


cdef void basis_enclosure_cell(
    const BasisTable* t,
    double lo,
    double hi,
    int top,
    const int* want,
    Iv* fwork,
    Iv* out,
) noexcept nogil:
    """``out[k * d + c]`` encloses ``Phi_c^(k)`` over ``[lo, hi]``, ``k <= top``.

    Columns with ``want[c] == 0`` are left untouched; ``want`` may be NULL
    (all).  ``fwork`` holds ``5 nf`` intervals.
    """
    cdef int d = t.d
    cdef int j, k, i, col, term
    cdef Iv x, z, total, low, up
    cdef Iv tl[7]
    cdef Iv tu[7]
    cdef Iv g[5]
    cdef Iv acc[5]
    cdef bint have[5]
    cdef double a1, a2
    cdef bint gated
    cdef bint lower_side = 0.5 * (lo + hi) < t.mid
    _cell_setup(t, lo, hi, top, &x, &z, tl, tu)
    for j in range(t.nf):
        if t.fn_used[j]:
            _function_cell(t, j, x, z, top, &fwork[5 * j])
    for col in range(d):
        if want != NULL and not want[col]:
            continue
        for k in range(top + 1):
            have[k] = False
        for term in range(t.col_start[col], t.col_start[col + 1]):
            j = t.term_fn[term]
            if t.orth_degree and t.orth_column_degree[col] > 0 and t.kind[j] == FN_MONOMIAL:
                continue
            a1 = t.term_g1[term]
            a2 = t.term_g2[term]
            gated = a1 != 0.0 or a2 != 0.0
            if gated:
                low = _add_const(_add(_scale(a1, tl[1]), _scale(-a2, tu[2])),
                                 t.term_glow[term])
                up = _add_const(_add(_scale(-a1, tl[2]), _scale(a2, tu[1])),
                                t.term_gup[term])
                g[0] = low if lower_side else up
                for k in range(top):
                    g[k + 1] = _add(_scale(a1, tl[3 + k]), _scale(a2, tu[3 + k]))
            for k in range(top + 1):
                if gated:
                    total = _scale(_comb(k, 0), _mul(fwork[5 * j], g[k]))
                    for i in range(1, k + 1):
                        total = _add(
                            total, _scale(_comb(k, i), _mul(fwork[5 * j + i], g[k - i]))
                        )
                else:
                    total = _scale(t.term_g0[term], fwork[5 * j + k])
                if have[k]:
                    acc[k] = _add(acc[k], total)
                else:
                    acc[k] = total
                    have[k] = True
        for k in range(top + 1):
            out[k * d + col] = acc[k]


cdef Py_ssize_t prep_size(const BasisTable* t) noexcept nogil:
    """Doubles needed by ``prepare_potential``."""
    return 10 * t.nf + (5 * t.nf + 1) // 2 + 1


cdef inline Iv _limit(const double* row, const double* absrow, const double* theta,
                      int d) noexcept nogil:
    """Enclosure of one gate-limit functional ``row . theta`` (``_limit_slot``)."""
    cdef double v = 0.0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double part
    cdef bint any_nz = False
    cdef int col
    for col in range(d):
        part = row[col] * theta[col]
        if part != 0.0:
            any_nz = True
        v = v + part
        a = a + fabs(part)
        b = b + absrow[col] * fabs(theta[col])
    if not any_nz:
        return _iv(0.0, 0.0)
    cdef double e = (d + 2) * _U * a + 2.0 * _U * b
    return _iv(_next_down(v - e), _next_up(v + e))


cdef void prepare_potential(
    const BasisTable* t, const double* theta, double* work, PotentialPrep* p
) noexcept nogil:
    """Slots of ``full = M theta`` with rounding bounds, and gate limits.

    Each slot carries ``(d + 1) u sum_c |M_sc theta_c|``; exactly zero slots
    stay exactly zero, so a function without a coefficient is skipped.
    Specialized polynomial point bounds handle exact unit maps separately.
    ``work`` holds ``prep_size(t)`` doubles.
    """
    cdef int nf = t.nf
    cdef int d = t.d
    cdef int s, col, j
    cdef double full, acc, term, err
    cdef bint any_nz
    p.theta = theta
    p.slot = <Iv*> work
    p.k_low = <Iv*> (work + 6 * nf)
    p.k_up = <Iv*> (work + 8 * nf)
    p.slot_zero = <int*> (work + 10 * nf)
    p.skip = p.slot_zero + 3 * nf
    p.gated = p.skip + nf
    p.all_zero = 1
    for s in range(3 * nf):
        full = 0.0
        acc = 0.0
        any_nz = False
        for col in range(d):
            term = t.cmap[s * d + col] * theta[col]
            if term != 0.0:
                any_nz = True
            full = full + term
            acc = acc + fabs(term)
        p.slot_zero[s] = not any_nz
        if any_nz:
            p.all_zero = 0
            err = (d + 1) * _U * acc
            p.slot[s] = _iv(_next_down(full - err), _next_up(full + err))
        else:
            p.slot[s] = _iv(0.0, 0.0)
    for j in range(nf):
        p.skip[j] = (
            p.slot_zero[3 * j] and p.slot_zero[3 * j + 1] and p.slot_zero[3 * j + 2]
        )
        p.gated[j] = not (p.slot_zero[3 * j + 1] and p.slot_zero[3 * j + 2])
        if p.gated[j]:
            p.k_low[j] = _limit(&t.lim_low[j * d], &t.lim_low_abs[j * d], theta, d)
            p.k_up[j] = _limit(&t.lim_up[j * d], &t.lim_up_abs[j * d], theta, d)


cdef void potential_enclosure_cell(
    const BasisTable* t,
    const PotentialPrep* p,
    double lo,
    double hi,
    int top,
    Iv* fwork,
    Iv* out,
) noexcept nogil:
    """``out[k]`` encloses ``q^(k)`` over ``[lo, hi]``, ``k <= top``.

    ``q`` is enclosed function by function through ``full = M theta``, one
    combined gate per function in the lower/upper forms, so coefficients that
    cancel across columns cancel before any interval product
    (``basis_interval._potential_enclosure``).  ``fwork`` holds ``5 nf``
    intervals.
    """
    cdef int j, k, i
    cdef Iv x, z, total, low, up, ca, cb, cc
    cdef Iv tl[7]
    cdef Iv tu[7]
    cdef Iv g[5]
    cdef Iv acc[5]
    cdef bint have = False
    cdef bint lower_side
    cdef Iv* f
    cdef Iv poly
    if p.all_zero:
        for k in range(top + 1):
            out[k] = _iv(0.0, 0.0)
        return
    lower_side = 0.5 * (lo + hi) < t.mid
    acc[2] = _iv(0.0, 0.0)
    _cell_setup(t, lo, hi, top, &x, &z, tl, tu)
    for j in range(t.nf):
        if p.skip[j]:
            continue
        f = &fwork[5 * j]
        _function_cell(t, j, x, z, top, f)
        ca = p.slot[3 * j]
        cb = p.slot[3 * j + 1]
        cc = p.slot[3 * j + 2]
        if p.gated[j]:
            low = _add(_add(p.k_low[j], _mul(cb, tl[1])), _neg(_mul(cc, tu[2])))
            up = _add(_add(p.k_up[j], _neg(_mul(cb, tl[2]))), _mul(cc, tu[1]))
            g[0] = low if lower_side else up
            for k in range(top):
                g[k + 1] = _add(_mul(cb, tl[3 + k]), _mul(cc, tu[3 + k]))
        for k in range(top + 1):
            if (t.polynomial_point and lo == hi and top == 2 and k == 2
                    and t.kind[j] == FN_MONOMIAL and not p.gated[j]):
                continue
            if p.gated[j]:
                total = _scale(_comb(k, 0), _mul(f[0], g[k]))
                for i in range(1, k + 1):
                    total = _add(total, _scale(_comb(k, i), _mul(f[i], g[k - i])))
            else:
                total = _mul(f[k], ca)
            acc[k] = _add(acc[k], total) if have else total
        have = True
    if t.polynomial_point and lo == hi and top == 2:
        poly = _polynomial_curvature_point(t, p, lo)
        if poly.lo != 0.0 or poly.hi != 0.0:
            if have:
                acc[2] = _add(acc[2], poly)
            else:
                acc[2] = poly
                have = True
    for k in range(top + 1):
        out[k] = acc[k]


cdef double endpoint_exclusion(
    const BasisTable* t,
    const double* theta,
    int upper,
    double threshold,
    double* rest,
    double* prepwork,
    Iv* ivwork,
) noexcept nogil:
    """Width of the neighbourhood of a finite endpoint where ``q'' >= threshold``.

    Only an endpoint with an enabled logarithm whose amplitude ``gamma`` is
    positive is excluded (``basis_tails._endpoint_exclusion``).  ``rest``
    holds ``d`` doubles, ``prepwork`` ``prep_size(t)``, ``ivwork`` ``5 nf``
    intervals.
    """
    cdef int col = t.log_col_upper if upper else t.log_col_lower
    cdef double gamma, span, delta0, rest_lo, need
    cdef int i
    cdef PotentialPrep p
    cdef Iv enc[3]
    if col < 0:
        return 0.0
    gamma = theta[col]
    if gamma <= 0.0:
        return 0.0
    span = t.upper - t.lower
    delta0 = npmin(t.h, 0.25 * span) if isfinite(span) else t.h
    for i in range(t.d):
        rest[i] = theta[i]
    rest[col] = 0.0
    prepare_potential(t, rest, prepwork, &p)
    if upper:
        potential_enclosure_cell(t, &p, t.upper - delta0, t.upper, 2, ivwork, enc)
    else:
        potential_enclosure_cell(t, &p, t.lower, t.lower + delta0, 2, ivwork, enc)
    rest_lo = enc[2].lo
    need = threshold - rest_lo
    if not isfinite(rest_lo):
        return 0.0
    if need <= 0.0:
        return delta0
    return npmin(delta0, sqrt(gamma / need) * (1.0 - 1e-9))


# --------------------------------------------------------------------------
# tail atoms and verdicts
# --------------------------------------------------------------------------

cdef class TailKernel:
    """Flat atoms of one ``_TailModel`` (``basis_tails``).

    Parameters
    ----------
    tail : _TailModel
        Tail atoms of one infinite end (possibly face-restricted).
    center, scale : float
        Fitting coordinate ``c``, ``h``.
    d : int
        Number of coefficients.
    """

    def __cinit__(self, tail, double center, double scale, int d):
        na = len(tail.atoms)
        ne = len(tail.envelopes)
        akind = np.array(
            [1 if a.kind == "log" else 0 for a in tail.atoms], dtype=np.intc
        )
        apower = np.array([a.power for a in tail.atoms], dtype=np.float64)
        arate = np.array([a.rate for a in tail.atoms], dtype=np.float64)
        aoffset = np.array([a.offset for a in tail.atoms], dtype=np.float64)
        arow = np.zeros((max(na, 1), d))
        acut = np.zeros((max(na, 1), d))
        for i, a in enumerate(tail.atoms):
            arow[i] = a.row
            acut[i] = a.row if a.cut is None else a.cut
        ecol = np.array([e.column for e in tail.envelopes], dtype=np.intc)
        ecoef = np.array([e.coef for e in tail.envelopes], dtype=np.float64)
        epower = np.array([e.power for e in tail.envelopes], dtype=np.float64)
        erate = np.array([e.rate for e in tail.envelopes], dtype=np.float64)
        arrays = [akind, apower, arate, aoffset, arow, acut, ecol, ecoef, epower, erate]
        arrays = [
            np.ascontiguousarray(a) if a.size else np.zeros(1, a.dtype) for a in arrays
        ]
        self._keep = arrays
        self.t.na = na
        self.t.ne = ne
        self.t.d = d
        self.t.sign = float(tail.sign)
        self.t.zeta_min = float(tail.zeta_min)
        self.t.c = center
        self.t.h = scale
        self.t.akind = <const int*> cnp.PyArray_DATA(arrays[0])
        self.t.apower = <const double*> cnp.PyArray_DATA(arrays[1])
        self.t.arate = <const double*> cnp.PyArray_DATA(arrays[2])
        self.t.aoffset = <const double*> cnp.PyArray_DATA(arrays[3])
        self.t.arow = <const double*> cnp.PyArray_DATA(arrays[4])
        self.t.acut = <const double*> cnp.PyArray_DATA(arrays[5])
        self.t.ecol = <const int*> cnp.PyArray_DATA(arrays[6])
        self.t.ecoef = <const double*> cnp.PyArray_DATA(arrays[7])
        self.t.epower = <const double*> cnp.PyArray_DATA(arrays[8])
        self.t.erate = <const double*> cnp.PyArray_DATA(arrays[9])

    def cut_row(self, int atom):
        """Cut row of a violating atom (its own row, or the full-space row)."""
        return np.array(self._keep[5][atom], dtype=np.float64)


cdef inline TailResult _empty_result() noexcept nogil:
    cdef TailResult r
    r.status = TAIL_INCONCLUSIVE
    r.cutoff = NAN
    r.zeta = NAN
    r.lead_value = NAN
    r.lead_power = NAN
    r.lead_rate = NAN
    r.margin = NAN
    r.has_lead = False
    r.has_margin = False
    r.has_cutoff = False
    r.cut_atom = -1
    return r


cdef TailResult _lead_proof(
    const TailTable* tt,
    const double* values,
    const double* slacks,
    int lead,
    const double* env,
    int nenv,
    double zeta_cap,
    double dominance,
    double* pieces,
) noexcept nogil:
    """Cutoff beyond which atom ``lead`` dominates every competitor."""
    cdef double lead_lo = values[lead] - slacks[lead]
    cdef int npc = 0
    cdef int i
    cdef double v, sl, mag, m, big_b, big_r, zeta, keep, factor, total
    cdef double b, r
    cdef TailResult res = _empty_result()
    for i in range(nenv):
        pieces[3 * npc] = env[3 * i]
        pieces[3 * npc + 1] = env[3 * i + 1]
        pieces[3 * npc + 2] = env[3 * i + 2]
        npc += 1
    for i in range(lead + 1, tt.na):
        v = values[i]
        sl = slacks[i]
        if v - sl > 0.0:
            continue  # nonnegative contribution
        mag = fabs(v) + sl
        if mag == 0.0:
            continue
        if tt.akind[i] == 1:
            # 1/(zeta + c0)^m <= zeta^-m (c0 >= 0) or 2^m zeta^-m (zeta >= 2|c0|).
            m = -tt.apower[i]
            if tt.aoffset[i] < 0.0:
                mag *= pow(2.0, m)
        pieces[3 * npc] = tt.apower[i]
        pieces[3 * npc + 1] = tt.arate[i]
        pieces[3 * npc + 2] = mag
        npc += 1
    big_b = tt.apower[lead]
    big_r = tt.arate[lead]
    res.status = TAIL_INCONCLUSIVE
    res.lead_value = lead_lo
    res.lead_power = big_b
    res.lead_rate = big_r
    res.has_lead = True
    zeta = tt.zeta_min
    for i in range(npc):
        b = pieces[3 * i]
        r = pieces[3 * i + 1]
        if r > big_r or (r == big_r and b > big_b):
            return res
        if r < big_r and b > big_b:
            zeta = npmax(zeta, (b - big_b) / (big_r - r))
    keep = 1.0 - npmax(dominance, 1e-12)
    while zeta <= zeta_cap:
        if big_b * log(zeta) + big_r * zeta < _LOG_UNDERFLOW:
            break
        if tt.akind[lead] == 1 and tt.aoffset[lead] > 0.0:
            factor = pow(zeta / (zeta + tt.aoffset[lead]), -tt.apower[lead])
        else:
            factor = 1.0
        total = 0.0
        for i in range(npc):
            total += exp(
                log(pieces[3 * i + 2])
                + (pieces[3 * i] - big_b) * log(zeta)
                + (pieces[3 * i + 1] - big_r) * zeta
            )
        if total * (1.0 + 1e-12) < lead_lo * factor * keep:
            res.status = TAIL_NONNEGATIVE
            res.cutoff = tt.c + tt.sign * zeta * tt.h
            res.zeta = zeta
            res.has_cutoff = True
            res.margin = lead_lo * factor - total * (1.0 + 1e-12)
            res.has_margin = True
            return res
        zeta *= 2.0
    return res


cdef Py_ssize_t tail_work_size(const TailTable* tt) noexcept nogil:
    """Doubles needed by ``tail_verdict``: values, slacks, envelopes, pieces."""
    return 5 * tt.na + 6 * tt.ne + 3


cdef TailResult tail_verdict(
    const TailTable* tt,
    const double* theta,
    double zeta_cap,
    double dominance,
    bint growth,
    double* work,
) noexcept nogil:
    """Sign of ``d^k q / d zeta^k`` beyond a cutoff (``basis_tails._tail_verdict``).

    The first nonzero atom decides a violation (its cut is the first
    stronger atom that is exactly zero at ``theta``, if any); any positive
    atom may lead the proof when every stronger atom is a nonnegative
    contribution, the smallest proven cutoff wins.  ``work`` holds
    ``tail_work_size(tt)`` doubles.
    """
    cdef int na = tt.na
    cdef int d = tt.d
    cdef double* values = work
    cdef double* slacks = work + na
    cdef double* env = work + 2 * na
    cdef double* pieces = work + 2 * na + 3 * tt.ne
    cdef int i, col, first = -1, nenv = 0, k, zero_lead = -1
    cdef double v, a, term, mag
    cdef bint nonzero_row
    cdef TailResult res = _empty_result()
    cdef TailResult best, cand
    for i in range(na):
        v = 0.0
        a = 0.0
        nonzero_row = False
        for col in range(d):
            if tt.arow[i * d + col] != 0.0:
                nonzero_row = True
            term = tt.arow[i * d + col] * theta[col]
            v = v + term
            a = a + fabs(term)
        values[i] = v
        slacks[i] = (d + 2) * _U * a
        if first < 0:
            if v == 0.0 and slacks[i] == 0.0:
                if nonzero_row and zero_lead < 0:
                    zero_lead = i
                continue
            if fabs(v) <= slacks[i]:
                res.lead_value = v
                return res
            if v < 0.0:
                res.status = TAIL_VIOLATED
                res.lead_value = v
                # A stronger atom exactly zero at theta was skipped: near
                # theta the feasible set is lexicographic (that atom > 0, or
                # = 0 and the weaker ones >= 0), whose closure is that atom
                # >= 0.  Its row is the valid cut (the weaker atom's row
                # excluded every member with linear tails at both ends,
                # G3-049); its margin moves the point to a positive leader.
                res.cut_atom = i if zero_lead < 0 else zero_lead
                return res
            first = i
    for i in range(tt.ne):
        mag = fabs(theta[tt.ecol[i]]) * tt.ecoef[i]
        if mag > 0.0:
            env[3 * nenv] = tt.epower[i]
            env[3 * nenv + 1] = tt.erate[i]
            env[3 * nenv + 2] = mag
            nenv += 1
    if first < 0:
        if nenv == 0:
            # No atom and no remainder survives: the derivative is zero there.
            res.status = TAIL_NONNEGATIVE
            res.cutoff = tt.c + tt.sign * tt.zeta_min * tt.h
            res.zeta = tt.zeta_min
            res.has_cutoff = True
            res.lead_value = 0.0
            return res
        res.lead_value = 0.0
        return res
    best = _lead_proof(
        tt, values, slacks, first, env, nenv, zeta_cap, dominance, pieces
    )
    for k in range(first + 1, na):
        if not (values[k - 1] - slacks[k - 1] > 0.0
                or (values[k - 1] == 0.0 and slacks[k - 1] == 0.0)):
            break  # a stronger atom that is not a nonnegative contribution
        if values[k] - slacks[k] <= 0.0:
            continue
        if growth and (tt.apower[k] < 0.0 or tt.arate[k] < 0.0):
            continue
        cand = _lead_proof(
            tt, values, slacks, k, env, nenv, zeta_cap, dominance, pieces
        )
        if cand.status == TAIL_NONNEGATIVE and (
            best.status != TAIL_NONNEGATIVE or cand.zeta < best.zeta
        ):
            best = cand
    return best


cdef int growth_status(const TailResult* res) noexcept nogil:
    """Properness from an order-1 verdict (``basis_tails._growth_verdict``)."""
    if res.status == TAIL_VIOLATED:
        return GROWTH_IMPROPER
    if res.has_lead and (res.lead_power < 0.0 or res.lead_rate < 0.0):
        # A positive but decaying leading slope is integrable.
        return GROWTH_IMPROPER
    if res.status == TAIL_INCONCLUSIVE:
        return GROWTH_INCONCLUSIVE
    if not res.has_margin or res.lead_power < 0.0 or res.lead_rate < 0.0:
        # Identically zero or decaying slope: q tends to a finite limit.
        return GROWTH_IMPROPER
    return GROWTH_PROPER


cdef double tail_bound(const TailResult* res, double zeta) noexcept nogil:
    """``margin zeta**lead_power exp(lead_rate zeta)`` (``_TailResult.bound``)."""
    cdef double log_w
    if not res.has_margin:
        return 0.0
    log_w = res.lead_power * log(zeta) + res.lead_rate * zeta
    if log_w > 700.0:
        return INFINITY
    return res.margin * exp(log_w)


# --------------------------------------------------------------------------
# Python entry points
# --------------------------------------------------------------------------

def evaluate(BasisKernel kernel, const double[::1] x, int max_order):
    """``Phi`` and ``x`` derivatives: array ``(max_order + 1, R, d)``."""
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = x.shape[0]
    cdef int d = t.d
    cdef int k, col
    cdef Py_ssize_t i
    if not 0 <= max_order <= 4:
        raise ValueError("max_order must be in 0..4")
    out = np.empty((max_order + 1, n, d))
    cdef double[:, :, ::1] o = out
    cdef double[::1] fwork = np.empty(5 * t.nf + 1)
    cdef double[::1] buf = np.empty((max_order + 1) * d + 1)
    for i in range(n):
        eval_point(t, x[i], max_order, &fwork[0], &buf[0])
        for k in range(max_order + 1):
            for col in range(d):
                o[k, i, col] = buf[k * d + col]
    return out


def potential(BasisKernel kernel, const double[::1] theta, const double[::1] x,
              int max_order):
    """``q`` and ``x`` derivatives: array ``(max_order + 1, R)``."""
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = x.shape[0]
    cdef Py_ssize_t i
    cdef int k
    if theta.shape[0] != t.d:
        raise ValueError("theta has the wrong length")
    if not 0 <= max_order <= 4:
        raise ValueError("max_order must be in 0..4")
    out = np.empty((max_order + 1, n))
    cdef double[:, ::1] o = out
    cdef double[::1] fwork = np.empty(5 * t.nf + 1)
    cdef double[::1] cwork = np.empty((max_order + 1) * t.d + 1)
    cdef double q[5]
    for i in range(n):
        potential_point(t, &theta[0], x[i], max_order, &fwork[0], &cwork[0], q)
        for k in range(max_order + 1):
            o[k, i] = q[k]
    return out


def basis_enclosure(BasisKernel kernel, const double[::1] lo, const double[::1] hi,
                    int top, want=None):
    """Enclosures of ``Phi^(k)``, ``k <= top``: arrays ``(top + 1, N, d)`` (lo, hi).

    Columns not in ``want`` are NaN.
    """
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = lo.shape[0]
    cdef int d = t.d
    cdef int k, col
    cdef Py_ssize_t i
    if hi.shape[0] != n:
        raise ValueError("cell arrays have different lengths")
    if not 0 <= top <= 4:
        raise ValueError("top must be in 0..4")
    out_lo = np.full((top + 1, n, d), np.nan)
    out_hi = np.full((top + 1, n, d), np.nan)
    cdef double[:, :, ::1] ol = out_lo
    cdef double[:, :, ::1] oh = out_hi
    mask = np.ones(d, dtype=np.intc)
    if want is not None:
        mask[:] = 0
        mask[np.asarray(list(want), dtype=np.intp)] = 1
    cdef int[::1] wm = mask
    fwork = np.empty(10 * t.nf + 2)
    cdef double[::1] fwd = fwork
    buf = np.empty(2 * (top + 1) * d + 2)
    cdef double[::1] bd = buf
    for i in range(n):
        basis_enclosure_cell(t, lo[i], hi[i], top, &wm[0], <Iv*> &fwd[0], <Iv*> &bd[0])
        for k in range(top + 1):
            for col in range(d):
                if wm[col]:
                    ol[k, i, col] = (<Iv*> &bd[0])[k * d + col].lo
                    oh[k, i, col] = (<Iv*> &bd[0])[k * d + col].hi
    return out_lo, out_hi


def potential_enclosure(BasisKernel kernel, const double[::1] theta,
                        const double[::1] lo, const double[::1] hi, int top):
    """Enclosures of ``q^(k)``, ``k <= top``: arrays ``(top + 1, N)`` (lo, hi)."""
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = lo.shape[0]
    cdef Py_ssize_t i
    cdef int k
    cdef PotentialPrep p
    if theta.shape[0] != t.d or hi.shape[0] != n:
        raise ValueError("theta or cell arrays have the wrong length")
    if not 0 <= top <= 4:
        raise ValueError("top must be in 0..4")
    prep = np.empty(prep_size(t))
    cdef double[::1] pw = prep
    prepare_potential(t, &theta[0], &pw[0], &p)
    out_lo = np.empty((top + 1, n))
    out_hi = np.empty((top + 1, n))
    cdef double[:, ::1] ol = out_lo
    cdef double[:, ::1] oh = out_hi
    fwork = np.empty(10 * t.nf + 2)
    cdef double[::1] fwd = fwork
    cdef Iv res[5]
    for i in range(n):
        potential_enclosure_cell(t, &p, lo[i], hi[i], top, <Iv*> &fwd[0], res)
        for k in range(top + 1):
            ol[k, i] = res[k].lo
            oh[k, i] = res[k].hi
    return out_lo, out_hi


def tail_result(TailKernel tail, const double[::1] theta, double zeta_cap,
                double dominance, bint growth):
    """Tail verdict as a tuple ``(status, cutoff, zeta, lead_value, lead_power,
    lead_rate, margin, cut_atom)`` (``None`` for absent values)."""
    if theta.shape[0] != tail.t.d:
        raise ValueError("theta has the wrong length")
    work = np.empty(tail_work_size(&tail.t))
    cdef double[::1] w = work
    cdef TailResult r = tail_verdict(
        &tail.t, &theta[0], zeta_cap, dominance, growth, &w[0]
    )
    status = ("nonnegative", "violated", "inconclusive")[r.status]
    return (
        status,
        r.cutoff if r.has_cutoff else None,
        r.zeta if r.has_cutoff else None,
        None if isnan(r.lead_value) else r.lead_value,
        r.lead_power if r.has_lead else None,
        r.lead_rate if r.has_lead else None,
        r.margin if r.has_margin else None,
        r.cut_atom,
    )


def endpoint_width(BasisKernel kernel, const double[::1] theta, bint upper,
                   double threshold):
    """Python entry of ``endpoint_exclusion`` (user units)."""
    cdef const BasisTable* t = &kernel.t
    if theta.shape[0] != t.d:
        raise ValueError("theta has the wrong length")
    rest = np.empty(t.d + 1)
    prep = np.empty(prep_size(t))
    iv = np.empty(10 * t.nf + 2)
    cdef double[::1] rv = rest
    cdef double[::1] pv = prep
    cdef double[::1] iw = iv
    return endpoint_exclusion(
        t, &theta[0], upper, threshold, &rv[0], &pv[0], <Iv*> &iw[0]
    )


def _ulp_steps(const double[::1] x):
    """``(next_up, next_down)`` of ``x`` (test hook for the inlined steps)."""
    up = np.empty(x.shape[0])
    down = np.empty(x.shape[0])
    cdef double[::1] u = up
    cdef double[::1] w = down
    cdef Py_ssize_t i
    for i in range(x.shape[0]):
        u[i] = _next_up(x[i])
        w[i] = _next_down(x[i])
    return up, down
