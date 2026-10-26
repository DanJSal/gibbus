# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""Compiled basis kernels: column table, evaluation, enclosures, curvature, tails.

* ``BasisKernel`` -- the flat column table of one compiled ``_BasisModel``
  (functions, gated terms, the exact map ``full = M theta`` and its gate-limit
  rows, the family constants and the curvature atoms), built once per model.
* ``eval_point`` / ``potential_point`` -- ``Phi`` and ``q`` with ``x``
  derivatives to order four.  Each gate ``alpha_0 + alpha_1 t_L + alpha_2 t_U``
  is evaluated in the exact partition form ``g_L W_L + kappa W_C + g_U W_U``
  (the gate's lower, central and upper limits times the nonnegative weights
  ``W_L = 1 - G_L``, ``W_U = 1 - G_U``, ``W_C = G_L + G_U - 1``), so an exactly
  suppressed limit contributes an exact zero and the rest are products of
  accurately computed small factors.
* ``basis_enclosure_cell`` / ``potential_enclosure_cell`` -- outward-rounded
  interval enclosures over a cell: every arithmetic result widened by a
  relative ``4u`` plus one ulp, every transcendental by ``8u`` plus one ulp
  (``8 n u`` for integer powers ``n``), exact zeros kept exact.  Numerical
  evidence, not an arbitrary-precision certificate.  Soft-power derivative
  enclosures beyond order two are the whole line (valid, uninformative).
* ``curvature_scaled`` / ``curvature_rows`` -- the structured scaled curvature
  ``H = q'' / ||pieces||`` with an error bound and the exact scale, from any
  ``z`` in binary64 range, and its theta-free per-atom rows (``R``, ``N``).
* ``TailKernel`` / ``tail_verdict`` -- the sign of ``d^k q / d zeta^k`` beyond
  a cutoff from coefficient-linear asymptotic atoms (``basis_tails``).

The module must be built without floating-point contraction (``setup.py``
passes ``-ffp-contract=off``): the rounding margins assume one rounding per
operation, and the double-double primitives of the curvature kernel must not
be fused.
"""

import numpy as np
from fractions import Fraction
cimport numpy as cnp
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

from libc.math cimport (
    INFINITY,
    NAN,
    sqrt,
    exp,
    erfc,
    expm1,
    fabs,
    fabsl,
    fma,
    frexp,
    ilogb,
    isfinite,
    isinf,
    isnan,
    log,
    log1p,
    pow,
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
    cdef long double slot_lo, slot_hi
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
    ``nextafter`` call was a measurable share of a fit.
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


cdef inline Iv _fpow_positive_zero(Iv a, double b) noexcept nogil:
    """``a**b`` for ``a >= 0`` and ``b > 0`` (increasing; ``0**b = 0`` exactly)."""
    cdef double plo = 0.0 if a.lo <= 0.0 else pow(a.lo, b)
    cdef Iv r = _widen(plo, pow(a.hi, b), _TRANS)
    return _iv(npmax(r.lo, 0.0), r.hi)


cdef inline Iv _log_iv(Iv a) noexcept nogil:
    return _widen(log(a.lo), log(a.hi), _TRANS)


cdef inline Iv _exp_iv(Iv a) noexcept nogil:
    return _widen(exp(a.lo), exp(a.hi), _TRANS)


# --------------------------------------------------------------------------
# column table
# --------------------------------------------------------------------------

# Curvature-atom gate codes: the single gates, the partition weights, and the
# central weight of two gates.
cdef enum:
    GATE_NONE = 0
    GATE_GL = 1
    GATE_GU = 2
    GATE_WL = 3
    GATE_WU = 4
    GATE_WC = 5


def _curvature_atoms(model, lim_low, lim_mid, lim_up):
    """Curvature atoms ``(function, gate code, coefficient row)`` of a model.

    Each function enters ``q`` as ``f (g_L W_L + kappa W_C + g_U W_U)`` with
    coefficient rows ``g_L = a - b + c``, ``kappa = a + b + c`` and
    ``g_U = a + b - c`` over ``theta``; an ungated function as ``f a``.  On a
    half-line the missing gate is identically one, so the central weight is the
    single gate (``W_C = G_U`` or ``G_L``).  Exactly zero rows are omitted.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    lim_low, lim_mid, lim_up : numpy.ndarray
        Gate-limit rows per function, shape ``(nf, d)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(fn, gate, rows)`` with shapes ``(A,)``, ``(A,)`` and ``(A, d)``.
    """
    request = model.request
    lower_gate, upper_gate = request.lower_infinite, request.upper_infinite
    exact = model.coefficient_map_exact
    cmap = np.asarray(model.coefficient_map, dtype=np.float64)
    fns, gates, rows = [], [], []
    for j in range(len(model.functions)):
        slots = exact[3 * j : 3 * j + 3]
        if not any(v for slot in slots for v in slot):
            continue
        gated = any(v for v in slots[1]) or any(v for v in slots[2])
        if not gated:
            fns.append(j)
            gates.append(GATE_NONE)
            rows.append(cmap[3 * j])
            continue
        if lower_gate and upper_gate:
            pieces = (
                (GATE_WL, lim_low[j]), (GATE_WC, lim_mid[j]), (GATE_WU, lim_up[j])
            )
        elif upper_gate:
            pieces = ((GATE_GU, lim_mid[j]), (GATE_WU, lim_up[j]))
        else:
            pieces = ((GATE_GL, lim_mid[j]), (GATE_WL, lim_low[j]))
        for code, row in pieces:
            if np.any(row != 0.0):
                fns.append(j)
                gates.append(code)
                rows.append(row)
    d = model.n_coefficients
    return (
        np.array(fns, dtype=np.intc),
        np.array(gates, dtype=np.intc),
        np.array(rows, dtype=np.float64).reshape(len(rows), d),
    )


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
        rate = np.array([f.rate or 0.0 for f in functions], dtype=np.float64)
        starts, fns, g0, g1, g2, glow, gmid, gup = [0], [], [], [], [], [], [], []
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
                gmid.append(float(gate[0] + gate[1] + gate[2]))
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
        # Exact gate-limit rows per function (small rationals: exact in binary64
        # whenever the map entries are).
        exact = model.coefficient_map_exact
        lim_rows = []
        for signs in ((1, -1, 1), (1, 1, 1), (1, 1, -1)):
            lim_rows.append(np.array(
                [[float(sum(sg * exact[3 * j + k][col] for k, sg in enumerate(signs)))
                  for col in range(d)] for j in range(nf)],
                dtype=np.float64,
            ).reshape(nf, d))
        lim_low, lim_mid, lim_up = (np.ascontiguousarray(r) for r in lim_rows)
        ma, mb, mc = cmap[0::3], cmap[1::3], cmap[2::3]
        absrow = np.abs(ma) + np.abs(mb) + np.abs(mc)
        lim_low_abs = np.ascontiguousarray(np.where(lim_low != 0.0, absrow, 0.0))
        lim_mid_abs = np.ascontiguousarray(np.where(lim_mid != 0.0, absrow, 0.0))
        lim_up_abs = np.ascontiguousarray(np.where(lim_up != 0.0, absrow, 0.0))
        nonneg = np.array([bool(c.nonnegative) for c in model.columns], dtype=np.intc)
        fn_used = np.zeros(nf, dtype=np.intc)
        fn_used[np.unique(np.array(fns, dtype=np.intc))] = 1
        atom_fn, atom_gate, atom_row = _curvature_atoms(model, lim_low, lim_mid, lim_up)
        # Per-atom parameter: soft-power exponent, canonical rate, or the exact
        # canonical endpoint of a log (0 or +-1 by construction of z).
        z_lower, z_upper = geometry.coordinate.canonical_support
        atom_par = np.array(
            [
                functions[j].power if functions[j].kind == "soft_power"
                else functions[j].rate if functions[j].kind == "exponential"
                else (z_lower if functions[j].side == "lower" else z_upper)
                if functions[j].kind == "log" else 0.0
                for j in atom_fn
            ],
            dtype=np.float64,
        )
        arrays = dict(
            kind=kind, side=side, degree=degree, power=power, rate=rate,
            col_start=np.array(starts, dtype=np.intc),
            term_fn=np.array(fns, dtype=np.intc),
            term_g0=np.array(g0, dtype=np.float64),
            term_g1=np.array(g1, dtype=np.float64),
            term_g2=np.array(g2, dtype=np.float64),
            term_glow=np.array(glow, dtype=np.float64),
            term_gmid=np.array(gmid, dtype=np.float64),
            term_gup=np.array(gup, dtype=np.float64),
            cmap=cmap, cmap_low=cmap_low, orth_degree=orth_degree,
            orth_fn=orth_fn, orth_norm=orth_norm, orth_reference=orth_reference,
            orth_recurrence=orth_recurrence, orth_map=orth_map,
            lim_low=lim_low, lim_mid=lim_mid, lim_up=lim_up,
            lim_low_abs=lim_low_abs, lim_mid_abs=lim_mid_abs, lim_up_abs=lim_up_abs,
            nonneg=nonneg, fn_used=fn_used,
            atom_fn=(np.ascontiguousarray(atom_fn) if atom_fn.size
                     else np.zeros(1, np.intc)),
            atom_gate=(np.ascontiguousarray(atom_gate) if atom_gate.size
                       else np.zeros(1, np.intc)),
            atom_par=np.ascontiguousarray(atom_par) if atom_par.size else np.zeros(1),
            atom_row=(np.ascontiguousarray(atom_row) if atom_row.size
                      else np.zeros((1, max(d, 1)))),
        )
        self._keep = arrays
        self.t.nf = nf
        self.t.d = d
        self.t.kind = <const int*> cnp.PyArray_DATA(arrays["kind"])
        self.t.side = <const int*> cnp.PyArray_DATA(arrays["side"])
        self.t.degree = <const int*> cnp.PyArray_DATA(arrays["degree"])
        self.t.power = <const double*> cnp.PyArray_DATA(arrays["power"])
        self.t.rate = <const double*> cnp.PyArray_DATA(arrays["rate"])
        self.t.col_start = <const int*> cnp.PyArray_DATA(arrays["col_start"])
        self.t.term_fn = <const int*> cnp.PyArray_DATA(arrays["term_fn"])
        self.t.term_g0 = <const double*> cnp.PyArray_DATA(arrays["term_g0"])
        self.t.term_g1 = <const double*> cnp.PyArray_DATA(arrays["term_g1"])
        self.t.term_g2 = <const double*> cnp.PyArray_DATA(arrays["term_g2"])
        self.t.term_glow = <const double*> cnp.PyArray_DATA(arrays["term_glow"])
        self.t.term_gmid = <const double*> cnp.PyArray_DATA(arrays["term_gmid"])
        self.t.term_gup = <const double*> cnp.PyArray_DATA(arrays["term_gup"])
        self.t.cmap = <const double*> cnp.PyArray_DATA(arrays["cmap"])
        self.t.cmap_low = <const double*> cnp.PyArray_DATA(arrays["cmap_low"])
        if plan is not None and plan.degree > 64:
            raise ValueError("recurrence coordinates support degrees up to 64")
        self.t.orth_degree = 0 if plan is None else plan.degree
        self.t.polynomial_point = plan is None and all(
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
        self.t.lim_mid = <const double*> cnp.PyArray_DATA(arrays["lim_mid"])
        self.t.lim_up = <const double*> cnp.PyArray_DATA(arrays["lim_up"])
        self.t.lim_low_abs = <const double*> cnp.PyArray_DATA(arrays["lim_low_abs"])
        self.t.lim_mid_abs = <const double*> cnp.PyArray_DATA(arrays["lim_mid_abs"])
        self.t.lim_up_abs = <const double*> cnp.PyArray_DATA(arrays["lim_up_abs"])
        self.t.nonneg = <const int*> cnp.PyArray_DATA(arrays["nonneg"])
        self.t.fn_used = <const int*> cnp.PyArray_DATA(arrays["fn_used"])
        self.t.c = float(geometry.center)
        self.t.h = float(geometry.scale)
        self.t.lower = float(request.lower)
        self.t.upper = float(request.upper)
        # Canonical support ends: exact by definition (-1, 0 or 1), so a
        # canonical node's distance to a finite end is not rounded through
        # the physical coordinate.
        self.t.zlower = float(geometry.coordinate.canonical_support[0])
        self.t.zupper = float(geometry.coordinate.canonical_support[1])
        self.t.has_l = bool(geometry.has_lower_gate)
        self.t.has_u = bool(geometry.has_upper_gate)
        self.t.gate_s = float(geometry.gate_sharpness)
        self.t.root = int(geometry.soft_power_root)
        self.t.log_col_lower = log_cols["lower"][0] if log_cols["lower"] else -1
        self.t.log_col_upper = log_cols["upper"][0] if log_cols["upper"] else -1
        self.t.natom = int(atom_fn.size)
        self.t.atom_fn = <const int*> cnp.PyArray_DATA(arrays["atom_fn"])
        self.t.atom_gate = <const int*> cnp.PyArray_DATA(arrays["atom_gate"])
        self.t.atom_par = <const double*> cnp.PyArray_DATA(arrays["atom_par"])
        self.t.atom_row = <const double*> cnp.PyArray_DATA(arrays["atom_row"])

    @property
    def n_coefficients(self):
        """Number of columns ``d``."""
        return self.t.d

    @property
    def atom_rows(self):
        """Coefficient rows of the curvature atoms over ``theta``, ``(A, d)``."""
        return np.array(self._keep["atom_row"][: self.t.natom], dtype=np.float64)

    @property
    def atoms(self):
        """``(function index, gate code)`` of every curvature atom."""
        return tuple(
            (int(self._keep["atom_fn"][i]), int(self._keep["atom_gate"][i]))
            for i in range(self.t.natom)
        )


# --------------------------------------------------------------------------
# point evaluation
# --------------------------------------------------------------------------

cdef double _SQRT1_2 = 0.70710678118654752440
cdef double _INV_SQRT_2PI = 0.39894228040143267794


cdef inline double scaled_z(const BasisTable* t, double x) noexcept nogil:
    """``(x - c) / h`` without avoidable overflow (as ``coords``)."""
    cdef double out = (x - t.c) / t.h
    cdef double mag
    if not isfinite(out) and isfinite(x):
        mag = npmax(npmax(fabs(x), fabs(t.c)), fabs(t.h))
        out = ((x / mag) - (t.c / mag)) / (t.h / mag)
    return out


# Powers ``z**i`` tabulated per point for monomials (``_function_point``).
cdef enum:
    _ZP = 64


cdef inline void _gaussian_jet(
    double w, double s, double sign, int order, double* out
) noexcept nogil:
    """``out[k] = sign s**k phi^(k-1)(w)`` for ``1 <= k <= order``.

    ``phi^(j)(w) = (-1)**j He_j(w) phi(w)`` with the probabilists' Hermite
    polynomials ``He_0..He_3``.
    """
    cdef double phi = _INV_SQRT_2PI * exp(-0.5 * w * w)
    cdef double he[4]
    cdef double sk = sign
    cdef int k
    he[0] = 1.0
    he[1] = w
    he[2] = w * w - 1.0
    he[3] = w * (w * w - 3.0)
    for k in range(1, order + 1):
        sk = sk * s
        out[k] = sk * (he[k - 1] if (k - 1) % 2 == 0 else -he[k - 1]) * phi


cdef inline void _gate_point(
    const BasisTable* t, double z, int order, double* wl, double* wu, double* wc
) noexcept nogil:
    """``z`` derivatives of the partition weights ``W_L``, ``W_U``, ``W_C``.

    ``W_L = Phi(-s (z + 1))`` and ``W_U = Phi(s (z - 1))`` are evaluated with
    ``erfc``, accurate down to underflow; ``W_C = 1 - W_L - W_U`` is the
    difference of a gate and the far partition weight (``G_U - W_L`` for
    ``z >= 0``, ``G_L - W_U`` below), free of cancellation.  A missing gate is
    identically one (its partition weight zero).
    """
    cdef double s = t.gate_s
    cdef double w = s * (z + 1.0)
    cdef double v = s * (z - 1.0)
    cdef double gl, gu
    cdef int k
    for k in range(order + 1):
        wl[k] = 0.0
        wu[k] = 0.0
        wc[k] = 0.0
    gl = 1.0
    gu = 1.0
    if t.has_l:
        wl[0] = 0.5 * erfc(w * _SQRT1_2)
        gl = 0.5 * erfc(-w * _SQRT1_2)
        _gaussian_jet(w, s, -1.0, order, wl)
    if t.has_u:
        wu[0] = 0.5 * erfc(-v * _SQRT1_2)
        gu = 0.5 * erfc(v * _SQRT1_2)
        _gaussian_jet(v, s, 1.0, order, wu)
    if t.has_l and t.has_u:
        wc[0] = gu - wl[0] if z >= 0.0 else gl - wu[0]
    elif t.has_u:
        wc[0] = gu
    elif t.has_l:
        wc[0] = gl
    else:
        wc[0] = 1.0
    for k in range(1, order + 1):
        wc[k] = -wl[k] - wu[k]


cdef inline void _tps_mul(
    const double* a, const double* b, double* out, int n
) noexcept nogil:
    """Truncated Taylor product ``out = a b`` (coefficients ``0..n``)."""
    cdef int k, i
    cdef double acc
    for k in range(n + 1):
        acc = 0.0
        for i in range(k + 1):
            acc = acc + a[i] * b[k - i]
        out[k] = acc


cdef inline void _tps_pow(
    const double* a, double p, double* out, int n
) noexcept nogil:
    """Truncated Taylor power ``out = a**p`` for ``a[0] > 0``.

    ``k a_0 out_k = sum_{j=1..k} (p j - (k - j)) a_j out_{k-j}``.
    """
    cdef int k, j
    cdef double acc
    out[0] = pow(a[0], p)
    for k in range(1, n + 1):
        acc = 0.0
        for j in range(1, k + 1):
            acc = acc + (p * j - (k - j)) * a[j] * out[k - j]
        out[k] = acc / (k * a[0])


cdef inline double _binom(double q, int k) noexcept nogil:
    """Generalized binomial coefficient ``q (q - 1) ... (q - k + 1) / k!``."""
    cdef double out = 1.0
    cdef int i
    for i in range(k):
        out = out * (q - i) / (i + 1)
    return out


cdef void _soft_power_point(
    double z, double p, int root, int order, double* f
) noexcept nogil:
    """``f[k] = d^k/dz^k A(z)**p``, ``A = (1 + z**m)**(1/m) - 1``, ``k <= order``.

    Taylor arithmetic in ``z``.  Below ``|z| = 1`` the identity
    ``A = z**m / D`` with ``D = sum_{i<m} R**i`` and ``R = (1 + z**m)**(1/m)``
    gives ``A**p = |z|**(m p) D**(-p)``: the power factor has closed-form
    Taylor coefficients (no ``0 * inf`` at the origin, where every derivative
    of order ``<= 4 < m p`` vanishes) and ``D`` is smooth and near ``m``.
    Above, ``R = |z| (1 + z**-m)**(1/m)`` avoids overflow of ``z**m`` and
    ``A >= 2**(1/m) - 1`` is bounded away from zero.
    """
    cdef int _k
    cdef double c0[5]
    cdef double c1[5]
    cdef double c2[5]
    cdef double c3[5]
    cdef double az = fabs(z)
    cdef double sg = -1.0 if z < 0.0 else 1.0
    cdef double q, fact
    cdef int k, n = order
    for k in range(5):
        c0[k] = 0.0
        c1[k] = 0.0
        c2[k] = 0.0
        c3[k] = 0.0
    if az < 1.0:
        q = root * p
        # c0: |z0 + d|**q;  c1: 1 + (z0 + d)**m;  R = c1**(1/m).
        for k in range(n + 1):
            c0[k] = _binom(q, k) * pow(az, q - k) * (sg if k % 2 == 1 else 1.0)
        for k in range(min(n, root) + 1):
            c1[k] = _binom(<double> root, k) * pow(z, <double> (root - k))
        c1[0] += 1.0
        _tps_pow(c1, 1.0 / root, c2, n)
        # D = 1 + R + ... + R**(m-1) by Horner: D = 1 + R (1 + R (...)).
        for k in range(n + 1):
            c3[k] = 0.0
        c3[0] = 1.0
        for _k in range(root - 1):
            _tps_mul(c3, c2, c1, n)
            for k in range(n + 1):
                c3[k] = c1[k]
            c3[0] += 1.0
        _tps_pow(c3, -p, c1, n)
        _tps_mul(c0, c1, c2, n)
    else:
        # c0: (z0 + d)**-m;  R = |z| (1 + c0)**(1/m);  A = R - 1.
        for k in range(n + 1):
            c0[k] = _binom(-<double> root, k) * pow(z, <double> (-root - k))
        c0[0] += 1.0
        _tps_pow(c0, 1.0 / root, c1, n)
        c3[0] = az
        c3[1] = sg
        _tps_mul(c3, c1, c0, n)
        c0[0] -= 1.0
        _tps_pow(c0, p, c2, n)
    fact = 1.0
    for k in range(n + 1):
        if k > 0:
            fact = fact * k
        f[k] = fact * c2[k]


cdef void _function_point(
    const BasisTable* t,
    int j,
    double dlower,
    double dupper,
    double z,
    const double* zp,
    int order,
    double* f,
) noexcept nogil:
    """``f[0..order]``: ``x`` derivatives of function ``j``.

    ``dlower``/``dupper`` are the point's distances to the finite ends in
    user units (endpoint logs); ``zp[i] = z**i`` for ``i < _ZP`` (higher
    monomial powers use ``pow``).
    """
    cdef int k, n
    cdef double h = t.h
    cdef double dist, inv, lam, base, ih
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
            dist = dlower
            inv = -1.0 / dist
        else:
            dist = dupper
            inv = 1.0 / dist
        f[0] = -log(dist)
        # d^k/dx^k of -log(dist): (k - 1)! inv**k
        base = 1.0
        for k in range(1, order + 1):
            base = base * inv * (k - 1 if k > 1 else 1)
            f[k] = base
        return
    if t.kind[j] == FN_EXP:
        # Centered at the own gate: exp(lam (z - 1)) or exp(-lam (z + 1)).
        lam = t.rate[j]
        if t.side[j] == SIDE_LOWER:
            base = exp(-lam * (z + 1.0))
            lam = -lam
        else:
            base = exp(lam * (z - 1.0))
        f[0] = base
        for k in range(1, order + 1):
            base = base * lam / h
            f[k] = base
        return
    _soft_power_point(z, t.power[j], t.root, order, f)
    ih = 1.0
    for k in range(1, order + 1):
        ih = ih / h
        f[k] = f[k] * ih


cdef inline void _gate_jet(
    double glow, double gmid, double gup, const double* wl, const double* wu,
    const double* wc, double ih, int order, double* g
) noexcept nogil:
    """``x`` derivatives of one gate ``g_L W_L + kappa W_C + g_U W_U``."""
    cdef int k
    cdef double scale = 1.0
    for k in range(order + 1):
        g[k] = (glow * wl[k] + gmid * wc[k] + gup * wu[k]) * scale
        scale = scale * ih


cdef void eval_point(
    const BasisTable* t, double x, int order, double* fwork, double* out
) noexcept nogil:
    """``out[k * d + c] = Phi_c^(k)(x)`` for ``k <= order``.

    ``fwork`` holds ``5 nf`` doubles.
    """
    _eval_point(t, x - t.lower, t.upper - x, scaled_z(t, x),
                (x - t.orth_origin) / t.orth_scale if t.orth_degree else 0.0,
                order, fwork, out)


cdef void eval_point_z(
    const BasisTable* t, double z, int order, double* fwork, double* out
) noexcept nogil:
    """``eval_point`` at the canonical point ``z``, never rounded to user units.

    The endpoint distances are ``h (z - z_L)`` and ``h (z_U - z)``, and the
    recurrence argument ``((c - origin) + h z) / scale``: on a support a few
    ulps wide the physical point ``c + h z`` would quantize the density.
    """
    _eval_point(t, t.h * (z - t.zlower), t.h * (t.zupper - z), z,
                ((t.c - t.orth_origin) + t.h * z) / t.orth_scale
                if t.orth_degree else 0.0,
                order, fwork, out)


cdef void _eval_point(
    const BasisTable* t, double dlower, double dupper, double z, double argument,
    int order, double* fwork, double* out
) noexcept nogil:
    """``eval_point`` from the point's end distances, ``z`` and recurrence argument."""
    cdef int d = t.d
    cdef int j, k, i, col, term, n
    cdef double wl[5]
    cdef double wu[5]
    cdef double wc[5]
    cdef double g[5]
    cdef double zp[_ZP]
    cdef double a0, a1, a2, acc, rk, rate = 0.0
    cdef double ih = 1.0 / t.h
    cdef bint gated
    cdef int nmax = 0
    _gate_point(t, z, order, wl, wu, wc)
    for k in range((order + 1) * d):
        out[k] = 0.0
    if t.orth_degree:
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
                a0 = t.orth_map[3*n*d+col]
                a1 = t.orth_map[(3*n+1)*d+col]
                a2 = t.orth_map[(3*n+2)*d+col]
                if a1 == 0.0 and a2 == 0.0:
                    if a0 != 0.0:
                        for k in range(order + 1):
                            out[k*d+col] += fwork[5*n+k] * a0 / pow(t.orth_scale, k)
                    continue
                _gate_jet(a0 - a1 + a2, a0 + a1 + a2, a0 + a1 - a2, wl, wu, wc, ih,
                          order, g)
                for k in range(order + 1):
                    for i in range(k + 1):
                        if g[k - i] != 0.0 and fwork[5*n+i] != 0.0:
                            out[k*d+col] += (_comb(k, i) * fwork[5*n+i]
                                             / pow(t.orth_scale, i) * g[k-i])
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
            _function_point(t, j, dlower, dupper, z, zp, order, &fwork[5 * j])
    for col in range(d):
        for term in range(t.col_start[col], t.col_start[col + 1]):
            j = t.term_fn[term]
            if t.orth_degree and t.orth_column_degree[col] > 0 and t.kind[j] == FN_MONOMIAL:
                continue
            a1 = t.term_g1[term]
            a2 = t.term_g2[term]
            gated = a1 != 0.0 or a2 != 0.0
            if gated:
                _gate_jet(t.term_glow[term], t.term_gmid[term], t.term_gup[term],
                          wl, wu, wc, ih, order, g)
            if t.kind[j] == FN_EXP:
                rate = (t.rate[j] if t.side[j] == SIDE_UPPER else -t.rate[j]) * ih
            for k in range(order + 1):
                if gated and t.kind[j] == FN_EXP and isinf(fwork[5 * j]):
                    # Overflowed exponential: f^(i) = f r**i exactly, so factor
                    # f out and keep the sign of the finite Leibniz sum.
                    acc = 0.0
                    rk = 1.0
                    for i in range(k + 1):
                        acc = acc + _comb(k, i) * rk * g[k - i]
                        rk = rk * rate
                    if acc != 0.0:
                        out[k * d + col] += fwork[5 * j] * acc
                elif gated:
                    # An exactly zero factor (an underflowed gate derivative)
                    # contributes nothing, even beside an overflowed function.
                    acc = 0.0
                    for i in range(k + 1):
                        if g[k - i] != 0.0 and fwork[5 * j + i] != 0.0:
                            acc = acc + _comb(k, i) * fwork[5 * j + i] * g[k - i]
                    out[k * d + col] += acc
                else:
                    out[k * d + col] += fwork[5 * j + k] * t.term_g0[term]


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
    eval_point(t, x, order, fwork, colwork)
    _contract(t, theta, order, colwork, q)


cdef void potential_point_z(
    const BasisTable* t,
    const double* theta,
    double z,
    int order,
    double* fwork,
    double* colwork,
    double* q,
) noexcept nogil:
    """``potential_point`` at the canonical point ``z`` (``eval_point_z``)."""
    eval_point_z(t, z, order, fwork, colwork)
    _contract(t, theta, order, colwork, q)


cdef inline void _contract(
    const BasisTable* t, const double* theta, int order, const double* colwork,
    double* q
) noexcept nogil:
    """``q[k] = sum_c theta_c colwork[k d + c]`` over columns with ``theta_c != 0``."""
    cdef int k, col
    cdef double acc
    for k in range(order + 1):
        acc = 0.0
        for col in range(t.d):
            if theta[col] != 0.0:
                acc = acc + colwork[k * t.d + col] * theta[col]
        q[k] = acc


# --------------------------------------------------------------------------
# cell enclosures
# --------------------------------------------------------------------------

cdef inline Iv _whole() noexcept nogil:
    return _iv(-INFINITY, INFINITY)


cdef inline Iv _phi_iv(Iv w) noexcept nogil:
    """Enclosure of the standard normal density over ``w``."""
    cdef double near, far
    if w.lo <= 0.0 <= w.hi:
        near = 0.0
    else:
        near = npmin(fabs(w.lo), fabs(w.hi))
    far = npmax(fabs(w.lo), fabs(w.hi))
    return _widen(_INV_SQRT_2PI * exp(-0.5 * far * far),
                  _INV_SQRT_2PI * exp(-0.5 * near * near), _TRANS)


cdef inline Iv _ncdf_iv(Iv w) noexcept nogil:
    """Enclosure of ``Phi(w) = erfc(-w / sqrt 2) / 2`` (increasing) over ``w``."""
    cdef Iv r = _widen(
        0.5 * erfc(-w.lo * _SQRT1_2), 0.5 * erfc(-w.hi * _SQRT1_2), _TRANS
    )
    return _iv(npmax(r.lo, 0.0), npmin(r.hi, 1.0))


cdef inline void _gaussian_cell(
    Iv w, double s, double sign, int order, Iv* out
) noexcept nogil:
    """``out[k]`` encloses ``sign s**k phi^(k-1)(w)`` for ``1 <= k <= order``."""
    cdef Iv phi = _phi_iv(w)
    cdef Iv he
    cdef double sk = sign
    cdef int k
    for k in range(1, order + 1):
        sk = sk * s
        if k == 1:
            he = _iv(1.0, 1.0)
        elif k == 2:
            he = _neg(w)
        elif k == 3:
            he = _add_const(_ipow(w, 2), -1.0)
        else:
            he = _neg(_mul(w, _add_const(_ipow(w, 2), -3.0)))
        out[k] = _scale(sk, _mul(he, phi))


cdef inline void _gate_cell(
    const BasisTable* t, Iv z, int order, Iv* wl, Iv* wu, Iv* wc
) noexcept nogil:
    """Enclosures of the partition weights and their ``z`` derivatives."""
    cdef double s = t.gate_s
    cdef Iv w = _scale(s, _add_const(z, 1.0))
    cdef Iv v = _scale(s, _add_const(z, -1.0))
    cdef Iv gl, gu
    cdef int k
    for k in range(order + 1):
        wl[k] = _iv(0.0, 0.0)
        wu[k] = _iv(0.0, 0.0)
        wc[k] = _iv(0.0, 0.0)
    gl = _iv(1.0, 1.0)
    gu = _iv(1.0, 1.0)
    if t.has_l:
        wl[0] = _ncdf_iv(_neg(w))
        gl = _ncdf_iv(w)
        _gaussian_cell(w, s, -1.0, order, wl)
    if t.has_u:
        wu[0] = _ncdf_iv(v)
        gu = _ncdf_iv(_neg(v))
        _gaussian_cell(v, s, 1.0, order, wu)
    if t.has_l and t.has_u:
        wc[0] = _add(gu, _neg(wl[0])) if z.lo + z.hi >= 0.0 else _add(gl, _neg(wu[0]))
        wc[0] = _iv(npmax(wc[0].lo, 0.0), npmin(wc[0].hi, 1.0))
    elif t.has_u:
        wc[0] = gu
    elif t.has_l:
        wc[0] = gl
    else:
        wc[0] = _iv(1.0, 1.0)
    for k in range(1, order + 1):
        wc[k] = _neg(_add(wl[k], wu[k]))


cdef inline double _soft_r(double m, int root) noexcept nogil:
    """``R(m) = (1 + m**root)**(1/root)`` for ``m >= 0`` without overflow."""
    if m <= 1.0:
        return pow(1.0 + pow(m, root), 1.0 / root)
    return m * pow(1.0 + pow(m, -root), 1.0 / root)


cdef void _soft_power_cell(Iv z, double p, int root, int order, Iv* f) noexcept nogil:
    """Enclosures of ``d^k/dz^k A**p`` over ``z`` for ``k <= order``.

    ``A**p = |z|**(m p) D**-p``, ``(A**p)' = sgn(z) p |z|**(m p - 1) D**(1-p)
    R**(1-m)`` and ``(A**p)'' = p (p - 1) |z|**(m p - 2) D**(2-p) R**(2-2m)
    + p (m - 1) |z|**(m p - 2) D**(1-p) R**(1-2m)`` follow from ``A = z**m / D``,
    ``A' = sgn(z) |z|**(m-1) R**(1-m)`` and ``A'' = (m-1) |z|**(m-2)
    R**(1-2m)``; every factor is a positive power of a positive monotone
    quantity, so the enclosures are tight and finite at the origin.  Orders
    three and four are the whole line.

    For ``|z| >= 1`` the bounded forms ``A' = sgn(z) u**(m-1)`` and
    ``A'' = (m - 1) u**(m-2) R**(-m-1)`` with ``u = |z| / R`` in ``(0, 1]`` and
    ``A = R - 1 >= 2**(1/m) - 1`` avoid overflow of ``|z|**(m p)``.
    """
    cdef int _k
    cdef double lo = npmin(fabs(z.lo), fabs(z.hi))
    cdef double hi = npmax(fabs(z.lo), fabs(z.hi))
    cdef double q = root * p
    cdef Iv m, r, dd_, t1, t2, a, u
    cdef int k
    if z.lo <= 0.0 <= z.hi:
        lo = 0.0
    m = _iv(lo, hi)
    r = _widen(_soft_r(lo, root), _soft_r(hi, root), _TRANS)
    if lo >= 1.0:
        a = _add_const(r, -1.0)
        a = _iv(npmax(a.lo, 0.0), a.hi)
        u = _widen(pow(1.0 + pow(lo, -root), -1.0 / root),
                   pow(1.0 + pow(hi, -root), -1.0 / root), _TRANS)
        u = _iv(u.lo, npmin(u.hi, 1.0))
        f[0] = _fpow_positive(a, p)
        if order >= 1:
            t1 = _scale(p, _mul(_fpow_positive(a, p - 1.0), _ipow(u, root - 1)))
            f[1] = t1 if z.lo >= 0.0 else _neg(t1)
        if order >= 2:
            t1 = _mul(_fpow_positive(a, p - 2.0), _ipow(u, 2 * root - 2))
            t2 = _mul(_mul(_fpow_positive(a, p - 1.0), _ipow(u, root - 2)),
                      _fpow_positive(r, -1.0 - root))
            f[2] = _add(_scale(p * (p - 1.0), t1), _scale(p * (root - 1.0), t2))
        for k in range(3, order + 1):
            f[k] = _whole()
        return
    # D = (R**m - 1) / A is 1 + R + ... + R**(m-1), increasing in |z|.
    dd_ = _iv(1.0, 1.0)
    for _k in range(root - 1):
        dd_ = _add_const(_mul(dd_, r), 1.0)
    f[0] = _mul(_fpow_positive_zero(m, q), _fpow_positive(dd_, -p))
    if order >= 1:
        t1 = _mul(_mul(_fpow_positive_zero(m, q - 1.0), _fpow_positive(dd_, 1.0 - p)),
                  _fpow_positive(r, 1.0 - root))
        t1 = _scale(p, t1)
        if z.lo >= 0.0:
            f[1] = t1
        elif z.hi <= 0.0:
            f[1] = _neg(t1)
        else:
            f[1] = _iv(-t1.hi, t1.hi)
    if order >= 2:
        t1 = _mul(_mul(_fpow_positive_zero(m, q - 2.0), _fpow_positive(dd_, 2.0 - p)),
                  _fpow_positive(r, 2.0 - 2.0 * root))
        t2 = _mul(_mul(_fpow_positive_zero(m, q - 2.0), _fpow_positive(dd_, 1.0 - p)),
                  _fpow_positive(r, 1.0 - 2.0 * root))
        f[2] = _add(_scale(p * (p - 1.0), t1), _scale(p * (root - 1.0), t2))
    for k in range(3, order + 1):
        f[k] = _whole()


cdef void _function_cell(
    const BasisTable* t, int j, Iv x, Iv z, int order, Iv* f
) noexcept nogil:
    """Enclosures ``f[0..order]`` of function ``j`` over the cell ``x``."""
    cdef int k, n
    cdef double h = t.h
    cdef double lam, rate
    cdef double signs[5]
    cdef Iv d, arg, base
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
            arg = _scale(-lam, _add_const(z, 1.0))
            rate = -lam / h
        else:
            arg = _scale(lam, _add_const(z, -1.0))
            rate = lam / h
        base = _exp_iv(arg)
        for k in range(order + 1):
            f[k] = _scale(pow(rate, k), base)
        return
    _soft_power_cell(z, t.power[j], t.root, order, f)
    for k in range(1, order + 1):
        f[k] = _scale(1.0 / pow(h, k), f[k])


cdef inline void _cell_setup(
    const BasisTable* t, double lo, double hi, int top, Iv* x, Iv* z, Iv* wl, Iv* wu,
    Iv* wc
) noexcept nogil:
    """Cell ``x``, its ``z`` image and the partition weights in ``x`` derivatives."""
    cdef int k
    cdef double ih = 1.0 / t.h
    cdef double scale = 1.0
    x[0] = _iv(lo, hi)
    z[0] = _scale(1.0 / t.h, _add_const(x[0], -t.c))
    _gate_cell(t, z[0], top, wl, wu, wc)
    for k in range(1, top + 1):
        scale = scale * ih
        wl[k] = _scale(scale, wl[k])
        wu[k] = _scale(scale, wu[k])
        wc[k] = _scale(scale, wc[k])


cdef inline Iv _gate_cell_jet(
    Iv glow, Iv gmid, Iv gup, const Iv* wl, const Iv* wu, const Iv* wc, int k
) noexcept nogil:
    """Enclosure of the ``k``-th derivative of ``g_L W_L + kappa W_C + g_U W_U``."""
    return _add(_add(_mul(glow, wl[k]), _mul(gmid, wc[k])), _mul(gup, wu[k]))


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
    cdef Iv x, z, total
    cdef Iv wl[5]
    cdef Iv wu[5]
    cdef Iv wc[5]
    cdef Iv g[5]
    cdef Iv acc[5]
    cdef bint have[5]
    cdef double a1, a2
    cdef bint gated
    _cell_setup(t, lo, hi, top, &x, &z, wl, wu, wc)
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
                for k in range(top + 1):
                    g[k] = _gate_cell_jet(
                        _iv(t.term_glow[term], t.term_glow[term]),
                        _iv(t.term_gmid[term], t.term_gmid[term]),
                        _iv(t.term_gup[term], t.term_gup[term]), wl, wu, wc, k,
                    )
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
    return 12 * t.nf + (5 * t.nf + 1) // 2 + 1 + 6 * (t.orth_degree + 1)


cdef inline Iv _limit(const double* row, const double* absrow, const double* theta,
                      int d, const int* skip_col) noexcept nogil:
    """Enclosure of one gate-limit functional ``row . theta`` (``_limit_slot``).

    Columns with ``skip_col[col]`` set (recurrence columns of a monomial
    function, enclosed in their own form) are left out; ``skip_col`` may be
    NULL.
    """
    cdef double v = 0.0
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double part
    cdef bint any_nz = False
    cdef int col
    for col in range(d):
        if skip_col != NULL and skip_col[col]:
            continue
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


cdef void _prepare_orth(
    const BasisTable* t, const double* theta, PotentialPrep* p
) noexcept nogil:
    """Recurrence coefficients of q per gate-limit combination, with bounds.

    ``orth_beta[L (n + 1) + n']`` encloses ``sum_c m_L[n', c] theta_c`` over the
    recurrence columns, where ``m_L`` combines the slot maps as the gate
    limits do (low ``a0 - a1 + a2``, mid ``a0 + a1 + a2``, up ``a0 + a1 -
    a2``); the bound covers the rounded map entries and the summation.
    """
    cdef int d = t.d, N = t.orth_degree, n, col, L
    cdef double a0, a1, a2, m, v, acc, gated_mag
    cdef double sg1[3]
    cdef double sg2[3]
    sg1[0] = -1.0
    sg1[1] = 1.0
    sg1[2] = 1.0
    sg2[0] = 1.0
    sg2[1] = 1.0
    sg2[2] = -1.0
    gated_mag = 0.0
    for L in range(3):
        for n in range(N + 1):
            v = 0.0
            acc = 0.0
            for col in range(d):
                if t.orth_column_degree[col] == 0 or theta[col] == 0.0:
                    continue
                a0 = t.orth_map[3 * n * d + col]
                a1 = t.orth_map[(3 * n + 1) * d + col]
                a2 = t.orth_map[(3 * n + 2) * d + col]
                if L == 0:
                    gated_mag += fabs(a1 * theta[col]) + fabs(a2 * theta[col])
                m = a0 + sg1[L] * a1 + sg2[L] * a2
                v += m * theta[col]
                acc += (fabs(a0) + fabs(a1) + fabs(a2)) * fabs(theta[col])
            if acc != 0.0:
                p.orth_on = 1
                acc = (d + 4) * 2.0 * _U * acc
                p.orth_beta[L * (N + 1) + n] = _iv(_next_down(v - acc), _next_up(v + acc))
            else:
                p.orth_beta[L * (N + 1) + n] = _iv(0.0, 0.0)
    p.orth_gated = gated_mag != 0.0
    if p.orth_on:
        p.all_zero = 0


cdef enum:
    _OMAX = 64


cdef void _orth_cell(
    const BasisTable* t, const PotentialPrep* p, double lo, double hi, int top,
    const Iv* wl, const Iv* wu, const Iv* wc, Iv* acc, bint* have
) noexcept nogil:
    """Add the recurrence-coordinate polynomial part of ``q^(k)`` over a cell.

    Each limit combination's polynomial is expanded in Taylor form about the
    cell midpoint ``u_m`` (``u = (x - origin) / scale``): its exact jets come
    from the three-term recurrence on the point ``u_m``, and the remainder
    over ``|u - u_m| <= r`` is bounded by the jets' magnitudes, so no
    monomial cancellation and no interval recurrence over a wide ``u``
    enters.  Gated parts combine with the partition weights by Leibniz.
    """
    cdef int N = t.orth_degree, n, k, j, L, i, nL
    cdef double xm = 0.5 * lo + 0.5 * hi
    cdef double r = 0.5 * hi - 0.5 * lo
    cdef double sc = t.orth_scale
    cdef double ru, fact, tail, mag, pw
    cdef Iv um, a, b, c, term
    cdef Iv prev[_OMAX + 1]
    cdef Iv cur[_OMAX + 1]
    cdef Iv nxt[_OMAX + 1]
    cdef Iv J[3][_OMAX + 1]
    cdef Iv poly[3][5]
    cdef Iv total
    um = _scale(1.0 / sc, _add_const(_iv(xm, xm), -t.orth_origin))
    um = _widen(um.lo, um.hi, _ARITH)
    ru = _next_up(_next_up(r / fabs(sc)) * (1.0 + 4.0 * _U))
    nL = 3 if p.orth_gated else 1
    for L in range(3):
        for k in range(N + 1):
            J[L][k] = _iv(0.0, 0.0)
    # Jets P_n^(k)(u_m) by the recurrence, accumulated into each combination.
    for k in range(N + 1):
        prev[k] = _iv(0.0, 0.0)
        cur[k] = _iv(0.0, 0.0)
    cur[0] = _iv(1.0, 1.0)
    for n in range(N + 1):
        for L in range(nL):
            b = p.orth_beta[(1 if nL == 1 else L) * (N + 1) + n]
            if _zero(b):
                continue
            for k in range(n + 1):
                J[L][k] = _add(J[L][k], _mul(b, cur[k]))
        if n == N:
            break
        a = _iv(t.orth_recurrence[3 * n], t.orth_recurrence[3 * n])
        b = _iv(t.orth_recurrence[3 * n + 1], t.orth_recurrence[3 * n + 1])
        c = _iv(t.orth_recurrence[3 * n + 2], t.orth_recurrence[3 * n + 2])
        for k in range(n + 2):
            term = _mul(_add(_mul(a, um), b), cur[k]) if k <= n else _iv(0.0, 0.0)
            if k >= 1:
                term = _add(term, _scale(<double> k, _mul(a, cur[k - 1])))
            if n >= 1 and k <= n - 1:
                term = _add(term, _scale(-1.0, _mul(c, prev[k])))
            nxt[k] = term
        for k in range(n + 2):
            prev[k] = cur[k]
            cur[k] = nxt[k]
    # Taylor enclosure of each x-derivative over the cell.
    for L in range(nL):
        for k in range(top + 1):
            if k > N:
                poly[L][k] = _iv(0.0, 0.0)
                continue
            tail = 0.0
            fact = 1.0
            pw = 1.0
            for j in range(k + 1, N + 1):
                fact = fact * (j - k)
                pw = pw * ru
                mag = npmax(fabs(J[L][j].lo), fabs(J[L][j].hi))
                tail = tail + mag * pw / fact
            tail = _next_up(tail * (1.0 + 4.0 * (N + 2) * _U))
            total = _iv(_next_down(J[L][k].lo - tail), _next_up(J[L][k].hi + tail))
            poly[L][k] = _scale(1.0 / pow(sc, k), total)
    for k in range(top + 1):
        if nL == 1:
            total = poly[0][k]
        else:
            # low, mid, up combinations on W_L, W_C, W_U.
            total = _iv(0.0, 0.0)
            for i in range(k + 1):
                total = _add(total, _scale(_comb(k, i), _add(_add(
                    _mul(poly[0][i], wl[k - i]), _mul(poly[1][i], wc[k - i])),
                    _mul(poly[2][i], wu[k - i]))))
        acc[k] = _add(acc[k], total) if have[0] else total
    have[0] = True


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
    cdef const int* skip_col
    p.theta = theta
    p.slot = <Iv*> work
    p.k_low = <Iv*> (work + 6 * nf)
    p.k_mid = <Iv*> (work + 8 * nf)
    p.k_up = <Iv*> (work + 10 * nf)
    p.slot_zero = <int*> (work + 12 * nf)
    p.skip = p.slot_zero + 3 * nf
    p.gated = p.skip + nf
    p.orth_beta = <Iv*> (work + 12 * nf + (5 * nf + 1) // 2 + 1)
    p.orth_on = 0
    p.orth_gated = 0
    p.all_zero = 1
    for s in range(3 * nf):
        full = 0.0
        acc = 0.0
        any_nz = False
        for col in range(d):
            if (t.orth_degree and t.orth_column_degree[col] > 0
                    and t.kind[s // 3] == FN_MONOMIAL):
                continue        # enclosed in its recurrence form instead
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
            skip_col = (t.orth_column_degree
                        if t.orth_degree and t.kind[j] == FN_MONOMIAL else NULL)
            p.k_low[j] = _limit(&t.lim_low[j * d], &t.lim_low_abs[j * d], theta, d,
                                skip_col)
            p.k_mid[j] = _limit(&t.lim_mid[j * d], &t.lim_mid_abs[j * d], theta, d,
                                skip_col)
            p.k_up[j] = _limit(&t.lim_up[j * d], &t.lim_up_abs[j * d], theta, d,
                               skip_col)
    if t.orth_degree:
        _prepare_orth(t, theta, p)


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
    combined gate per function in the partition form, so coefficients that
    cancel across columns cancel before any interval product
    (``basis_interval._potential_enclosure``).  ``fwork`` holds ``5 nf``
    intervals.
    """
    cdef int j, k, i
    cdef Iv x, z, total, ca
    cdef Iv wl[5]
    cdef Iv wu[5]
    cdef Iv wc[5]
    cdef Iv g[5]
    cdef Iv acc[5]
    cdef bint have = False
    cdef Iv* f
    cdef Iv poly
    if p.all_zero:
        for k in range(top + 1):
            out[k] = _iv(0.0, 0.0)
        return
    acc[2] = _iv(0.0, 0.0)
    _cell_setup(t, lo, hi, top, &x, &z, wl, wu, wc)
    for j in range(t.nf):
        if p.skip[j]:
            continue
        f = &fwork[5 * j]
        _function_cell(t, j, x, z, top, f)
        ca = p.slot[3 * j]
        if p.gated[j]:
            for k in range(top + 1):
                g[k] = _gate_cell_jet(p.k_low[j], p.k_mid[j], p.k_up[j], wl, wu, wc, k)
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
    if p.orth_on:
        _orth_cell(t, p, lo, hi, top, wl, wu, wc, acc, &have)
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
    """Width of the neighborhood of a finite endpoint where ``q'' >= threshold``.

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
# structured scaled curvature
# --------------------------------------------------------------------------
#
# Every curvature atom is ``coef * f(z) * W(z)``: ``f`` a monomial ``z**n``, a
# soft power ``A**p``, an exponential ``exp(lam (dir z - 1))``, or an endpoint
# log ``-log|z - e|`` (``q'' = coef / (z - e)**2``), and ``W`` one of the gates
# ``1, G_L, G_U, W_L, W_U, W_C``.  ``q''`` expands by the Leibniz rule into
# signed pieces, each a product of single-signed factor-jet terms held in
# structured log form
#
#     log|piece| = -k (s**2 / 2) z**2 + a1 z + rem      (k in {0, 1})
#
# with ``a1`` and ``rem`` in double-double, so no piece overflows or
# underflows, Gaussian, exponential and algebraic magnitudes compare exactly,
# and huge exponents keep full relative accuracy.  Per point the kernel
# returns ``H = q'' / ||pieces||_2`` (bounded, smooth, sign exact), an error
# bound ``err_h`` at the backward-stable level and the exact scale
# ``log M = -k (s**2/2) z**2 + a1 z + rem + log ||pieces|| / M_ref``.  A fast
# path (``1e-30 <= |z| <= 1e30``) carries each factor as one structured value
# times double ratios ``f'/f`` and ``f''/f`` (one ``exp`` per atom); the log
# path handles every other point.  Both give the same pieces.

cdef double _DD_EPS = 1.1102230246251565e-16
cdef double _SKIP_LOG = -60.0       # pieces below e**-60 of the reference are dropped
cdef double _FAST_ZMIN = 1e-30
cdef double _FAST_ZMAX = 1e30
cdef double _LN2_HI = 6.93147180369123816490e-01
cdef double _LN2_LO = 1.90821492927058770002e-10

cdef enum:
    _MAXP_PER_ATOM = 14     # Leibniz pieces of one atom (6 orders x <= 2 terms)
    _XPOLY = 0
    _XSOFT = 1
    _XEXP = 2
    _XLOG = 3


cdef struct dd:
    double hi
    double lo


cdef inline dd _dd(double hi, double lo) noexcept nogil:
    cdef dd r
    r.hi = hi
    r.lo = lo
    return r


cdef dd _DD0 = _dd(0.0, 0.0)
cdef dd _LN2_DD = _dd(0.6931471805599453, 2.3190468138462996e-17)
cdef dd _HALF_LOG_2PI = _dd(0.9189385332046728, -3.8782941580672414e-17)


cdef inline dd _two_sum(double a, double b) noexcept nogil:
    cdef double s = a + b
    cdef double bb = s - a
    return _dd(s, (a - (s - bb)) + (b - bb))


cdef inline dd _quick_two_sum(double a, double b) noexcept nogil:
    cdef double s = a + b
    return _dd(s, b - (s - a))


cdef inline dd _two_prod(double a, double b) noexcept nogil:
    cdef double p = a * b
    return _dd(p, fma(a, b, -p))


cdef inline dd _dd_neg(dd a) noexcept nogil:
    return _dd(-a.hi, -a.lo)


cdef inline dd _dd_add(dd a, dd b) noexcept nogil:
    cdef dd s = _two_sum(a.hi, b.hi)
    cdef dd t = _two_sum(a.lo, b.lo)
    s.lo += t.hi
    s = _quick_two_sum(s.hi, s.lo)
    s.lo += t.lo
    return _quick_two_sum(s.hi, s.lo)


cdef inline dd _dd_add_d(dd a, double b) noexcept nogil:
    cdef dd s = _two_sum(a.hi, b)
    s.lo += a.lo
    return _quick_two_sum(s.hi, s.lo)


cdef inline dd _dd_mul_d(dd a, double b) noexcept nogil:
    cdef dd p = _two_prod(a.hi, b)
    p.lo += a.lo * b
    return _quick_two_sum(p.hi, p.lo)


cdef inline dd _dd_mul(dd a, dd b) noexcept nogil:
    cdef dd p = _two_prod(a.hi, b.hi)
    p.lo += a.hi * b.lo + a.lo * b.hi
    return _quick_two_sum(p.hi, p.lo)


cdef inline dd _log_dd(double x) noexcept nogil:
    """``log(x)`` for finite ``x > 0`` to about one ulp absolute accuracy."""
    cdef int e
    cdef double f = frexp(x, &e)
    cdef double lf
    cdef dd r
    if f < _SQRT1_2:
        f *= 2.0
        e -= 1
    lf = log1p(f - 1.0)                 # f - 1 is exact (Sterbenz)
    r = _two_sum(<double> e * _LN2_HI, lf)
    r.lo += <double> e * _LN2_LO
    return _quick_two_sum(r.hi, r.lo)


cdef inline dd _log_of_dd(dd x) noexcept nogil:
    return _dd_add_d(_log_dd(x.hi), x.lo / x.hi)


cdef double _erfcx_pos(double x) noexcept nogil:
    """``exp(x**2) erfc(x)`` for ``x >= 0`` to a few ulp."""
    cdef dd x2
    cdef double t, total, inv
    cdef int k
    if x < 25.0:
        x2 = _two_prod(x, x)
        return exp(x2.hi) * erfc(x) * (1.0 + x2.lo)
    # Asymptotic series; term ratios (2k - 1) / (2 x**2) <= 1/1250.
    t = 1.0
    total = 1.0
    inv = 1.0 / (2.0 * x * x)
    for k in range(1, 13):
        t *= -(2.0 * k - 1.0) * inv
        total += t
    return total / (x * 1.7724538509055160273)


# One signed term of a factor jet: sgn * exp(-k (s**2/2) z**2 + a1 z + rem).
cdef struct term:
    double sgn
    int k
    dd a1
    dd rem
    double err


# A factor: derivative orders 0..2, each a sum of at most two terms.
cdef struct factor:
    int nt[3]
    term t[3][2]


cdef inline term _term(double sgn, int k, dd a1, dd rem, double err) noexcept nogil:
    cdef term r
    r.sgn = sgn
    r.k = k
    r.a1 = a1
    r.rem = rem
    r.err = err
    return r


cdef inline void _factor_one(factor* f) noexcept nogil:
    memset(f, 0, sizeof(factor))
    f.nt[0] = 1
    f.t[0][0] = _term(1.0, 0, _DD0, _DD0, 0.0)


cdef struct fam:
    double s
    dd s2
    dd s2half
    dd logs
    dd log_s2
    int m
    dd log_m1


cdef struct ptstate:
    double z
    double az
    double sz
    int zzero
    int lz_ok
    dd Lz
    dd z2
    int sp_ok
    dd logA
    dd logAp
    dd logApp
    int g_ok[4]
    factor g[4]


cdef struct atomc:
    int xtype
    int n
    int gate
    int dir
    double par
    double sgnth
    double theta
    dd lth
    dd logp
    dd logpm1
    double sgnpm1
    dd loglam
    dd logn
    dd lognn1


cdef struct piece:
    double sgn
    int k
    dd a1
    dd rem
    double err


cdef inline double _gate_sig(int gi) noexcept nogil:
    # G_L, G_U, W_L = 1 - G_L, W_U = 1 - G_U as Phi(sig s (z + c)).
    return 1.0 if gi == 0 or gi == 3 else -1.0


cdef inline double _gate_c(int gi) noexcept nogil:
    return 1.0 if gi == 0 or gi == 2 else -1.0


cdef void _softpower_base(ptstate* P, const fam* F) noexcept nogil:
    cdef int _k
    cdef int m = F.m
    cdef double az = P.az
    cdef double y, l1p, g, aoz, R, D, Rk
    cdef dd lzr, logR
    if az >= 1.0:
        y = pow(az, -<double> m)
        l1p = log1p(y)
        lzr = _dd(-l1p / m, 0.0)
        logR = _dd_add_d(P.Lz, l1p / m)
        g = expm1(l1p / m)                 # R / |z| - 1
        aoz = (az - 1.0) / az + g          # A / |z|
        P.logA = _dd_add(P.Lz, _log_dd(aoz))
    else:
        y = pow(az, <double> m)
        l1p = log1p(y)
        logR = _dd(l1p / m, 0.0)
        lzr = _dd_add_d(P.Lz, -l1p / m)
        R = exp(l1p / m)
        D = 0.0
        Rk = 1.0
        for _k in range(m):
            D += Rk
            Rk *= R
        P.logA = _dd_add(_dd_mul_d(P.Lz, <double> m), _dd_neg(_log_dd(D)))
    P.logAp = _dd_mul_d(lzr, <double> (m - 1))                       # log|A'|
    P.logApp = _dd_add(_dd_add(F.log_m1, _dd_mul_d(lzr, <double> (m - 2))),
                       _dd_neg(_dd_mul_d(logR, <double> (m + 1))))   # log A''
    P.sp_ok = 1


cdef void _gate_single(
    factor* f, const ptstate* P, const fam* F, double sig, double c
) noexcept nogil:
    """Jet of ``Phi(w)``, ``w = sig s (z + c)``, ``c = +-1``."""
    cdef dd zc = _two_sum(P.z, c)                 # exact z + c
    cdef double w = sig * F.s * (zc.hi + zc.lo)
    cdef dd a1 = _dd_mul_d(F.s2, -c)              # -s**2 c
    cdef dd r0 = _dd_neg(F.s2half)                # -(s**2/2) c**2, c**2 = 1
    cdef dd rphi, labsw
    cdef double ph, x, ex
    memset(f, 0, sizeof(factor))
    f.nt[0] = 1
    if w >= -1.0:
        ph = 0.5 * erfc(-w * _SQRT1_2)
        f.t[0][0] = _term(1.0, 0, _DD0, _log_dd(ph), 4.0)
    else:
        x = -w * _SQRT1_2
        ex = _erfcx_pos(x)
        f.t[0][0] = _term(1.0, 1, a1, _dd_add(r0, _log_dd(0.5 * ex)), 6.0)
    # Order 1: sig s phi(w).
    rphi = _dd_add(r0, _dd_neg(_HALF_LOG_2PI))
    f.nt[1] = 1
    f.t[1][0] = _term(sig, 1, a1, _dd_add(rphi, F.logs), 3.0)
    # Order 2: -s**2 w phi(w).
    if zc.hi != 0.0:
        labsw = _dd_add(F.logs, _log_of_dd(zc if zc.hi > 0 else _dd_neg(zc)))
        f.nt[2] = 1
        f.t[2][0] = _term(-1.0 if w > 0 else 1.0, 1, a1,
                          _dd_add(_dd_add(rphi, F.log_s2), labsw), 4.0)


cdef inline const factor* _get_gate(ptstate* P, const fam* F, int gi) noexcept nogil:
    if not P.g_ok[gi]:
        _gate_single(&P.g[gi], P, F, _gate_sig(gi), _gate_c(gi))
        P.g_ok[gi] = 1
    return &P.g[gi]


cdef void _poly_factor(factor* f, const ptstate* P, const atomc* a) noexcept nogil:
    cdef int n = a.n
    cdef double sz
    memset(f, 0, sizeof(factor))
    if n == 0:
        _factor_one(f)
        return
    if P.zzero:
        # Only the order-n derivative survives at the origin.
        if n == 1:
            f.nt[1] = 1
            f.t[1][0] = _term(1.0, 0, _DD0, _DD0, 0.0)
        if n == 2:
            f.nt[2] = 1
            f.t[2][0] = _term(1.0, 0, _DD0, _LN2_DD, 0.0)
        return
    sz = P.sz
    f.nt[0] = 1
    f.t[0][0] = _term(-1.0 if (n & 1) and sz < 0 else 1.0, 0, _DD0,
                      _dd_mul_d(P.Lz, n), 2.0)
    f.nt[1] = 1
    f.t[1][0] = _term(-1.0 if ((n - 1) & 1) and sz < 0 else 1.0, 0, _DD0,
                      _dd_add(a.logn, _dd_mul_d(P.Lz, n - 1)), 2.0)
    if n >= 2:
        f.nt[2] = 1
        f.t[2][0] = _term(-1.0 if ((n - 2) & 1) and sz < 0 else 1.0, 0, _DD0,
                          _dd_add(a.lognn1, _dd_mul_d(P.Lz, n - 2)), 2.0)


cdef void _softpower_factor(
    factor* f, ptstate* P, const fam* F, const atomc* a
) noexcept nogil:
    cdef double p = a.par
    cdef double er = 6.0 + 2.0 * fabs(p)
    cdef int j = 0
    memset(f, 0, sizeof(factor))
    if P.zzero:
        return                  # S, S', S'' vanish at the origin for p > 1/2
    if not P.sp_ok:
        _softpower_base(P, F)
    f.nt[0] = 1
    f.t[0][0] = _term(1.0, 0, _DD0, _dd_mul_d(P.logA, p), er)
    f.nt[1] = 1
    f.t[1][0] = _term(P.sz, 0, _DD0,
                      _dd_add(_dd_add(a.logp, _dd_mul_d(P.logA, p - 1.0)), P.logAp), er)
    if a.sgnpm1 != 0.0:
        f.t[2][j] = _term(a.sgnpm1, 0, _DD0, _dd_add(
            _dd_add(_dd_add(a.logp, a.logpm1), _dd_mul_d(P.logA, p - 2.0)),
            _dd_mul_d(P.logAp, 2.0)), er)
        j += 1
    f.t[2][j] = _term(1.0, 0, _DD0, _dd_add(
        _dd_add(a.logp, _dd_mul_d(P.logA, p - 1.0)), P.logApp), er)
    j += 1
    f.nt[2] = j


cdef void _exp_factor(factor* f, const atomc* a) noexcept nogil:
    # exp(lam (dir z - 1)): centered at the own gate.
    cdef double lam = a.par
    cdef dd a1 = _dd(a.dir * lam, 0.0)
    cdef dd r = _dd(-lam, 0.0)
    memset(f, 0, sizeof(factor))
    f.nt[0] = 1
    f.t[0][0] = _term(1.0, 0, a1, r, 1.0)
    f.nt[1] = 1
    f.t[1][0] = _term(<double> a.dir, 0, a1, _dd_add(r, a.loglam), 2.0)
    f.nt[2] = 1
    f.t[2][0] = _term(1.0, 0, a1, _dd_add(r, _dd_mul_d(a.loglam, 2.0)), 2.0)


cdef inline void _emit(
    piece* pc, int* npiece, double sgn, const term* x, const term* y, const term* w,
    int mult2, const atomc* a
) noexcept nogil:
    cdef piece* q = &pc[npiece[0]]
    cdef dd r
    npiece[0] += 1
    q.sgn = sgn * a.sgnth * x.sgn * y.sgn * w.sgn
    q.k = x.k + y.k + w.k
    q.a1 = _dd_add(_dd_add(x.a1, y.a1), w.a1)
    r = _dd_add(_dd_add(x.rem, y.rem), _dd_add(w.rem, a.lth))
    if mult2:
        r = _dd_add(r, _LN2_DD)
    q.rem = r
    q.err = x.err + y.err + w.err + 1.0


cdef void _leibniz(
    piece* pc, int* npiece, const factor* f0, const factor* f1, const factor* f2,
    double sgn, const atomc* a
) noexcept nogil:
    """Pieces of ``(f0 f1 f2)''``: three second-order and three mixed terms."""
    cdef int o, i0, i1, i2, a0, b0, c0
    for o in range(6):
        if o == 0:
            i0, i1, i2 = 2, 0, 0
        elif o == 1:
            i0, i1, i2 = 0, 2, 0
        elif o == 2:
            i0, i1, i2 = 0, 0, 2
        elif o == 3:
            i0, i1, i2 = 1, 1, 0
        elif o == 4:
            i0, i1, i2 = 1, 0, 1
        else:
            i0, i1, i2 = 0, 1, 1
        for a0 in range(f0.nt[i0]):
            for b0 in range(f1.nt[i1]):
                for c0 in range(f2.nt[i2]):
                    _emit(pc, npiece, sgn, &f0.t[i0][a0], &f1.t[i1][b0],
                          &f2.t[i2][c0], o >= 3, a)


cdef inline double _approx_d(
    const piece* p, const piece* r, double z, double z2d, double s2h
) noexcept nogil:
    """Approximate ``log|p| - log|r|`` (for ordering and skipping)."""
    cdef int dk = p.k - r.k
    cdef double t2 = 0.0 if dk == 0 else -<double> dk * s2h * z2d
    cdef double t1 = (p.a1.hi - r.a1.hi) * z
    if isinf(t2):
        return t2
    if isinf(t1):
        return t1
    return t2 + t1 + (p.rem.hi - r.rem.hi)


# Fast-path gate.  The value is ``mult * exp(structured part)``.  In the
# Gaussian branch (``w < -1``) the derivatives are ratios ``r1 = G'/G`` and
# ``r2 = G''/G`` of the same structure.  In the plain branch (``w >= -1``, the
# gate is order one) the derivatives ``G' = sig s phi(w)`` and
# ``G'' = -s**2 w phi(w)`` are carried separately as multipliers ``d1``, ``d2``
# of the exact structure of ``phi`` (``k = 1``, ``da1``, ``drem``), so they keep
# their relative accuracy where ``phi`` itself underflows.
cdef struct gfast:
    int k
    dd a1
    dd rem
    double mult
    double r1
    double r2
    int plain
    dd da1
    dd drem
    double d1
    double d2


cdef void _gate_fast(
    gfast* g, const ptstate* P, const fam* F, double sig, double c
) noexcept nogil:
    cdef dd zc = _two_sum(P.z, c)
    cdef dd w = _dd_mul_d(zc, sig * F.s)
    cdef double wd = w.hi
    cdef double ratio, x, ex
    if wd >= -1.0:
        g.plain = 1
        g.k = 0
        g.a1 = _DD0
        g.rem = _DD0
        g.mult = 0.5 * erfc(-wd * _SQRT1_2)
        g.r1 = 0.0
        g.r2 = 0.0
        # phi(w) = exp(-(s**2/2) z**2 - s**2 c z - s**2/2) / sqrt(2 pi), c**2 = 1.
        g.da1 = _dd_mul_d(F.s2, -c)
        g.drem = _dd_add(_dd_neg(F.s2half), _dd_neg(_HALF_LOG_2PI))
        g.d1 = sig * F.s
        g.d2 = -F.s2.hi * wd
        return
    g.plain = 0
    x = -wd * _SQRT1_2
    ex = _erfcx_pos(x)
    g.k = 1
    g.a1 = _dd_mul_d(F.s2, -c)
    g.rem = _dd_neg(F.s2half)
    g.mult = 0.5 * ex
    ratio = 0.79788456080286535588 / ex          # sqrt(2/pi) / erfcx
    g.r1 = sig * F.s * ratio
    g.r2 = -F.s2.hi * wd * ratio


cdef struct spfast:
    int ok
    dd logA
    double rA1
    double rA2


cdef void _softpower_fast(spfast* S, ptstate* P, const fam* F) noexcept nogil:
    cdef int _k
    cdef int m = F.m
    cdef double az = P.az
    cdef double A, R, t, gm1, iz2, y, l1p, aoz, D, Rk, q, qm2, Rp
    if not P.lz_ok:
        P.Lz = _log_dd(az)
        P.lz_ok = 1
    if az >= 1.0:
        # t = R / |z| = (1 + |z|**-m)**(1/m).
        if m == 4:
            iz2 = 1.0 / (az * az)
            y = iz2 * iz2
            t = sqrt(sqrt(1.0 + y))
            gm1 = y / ((1.0 + t) * (1.0 + t * t))       # t - 1, exactly rearranged
        else:
            y = pow(az, -<double> m)
            l1p = log1p(y)
            t = exp(l1p / m)
            gm1 = expm1(l1p / m)
        aoz = (az - 1.0) / az + gm1
        A = az * aoz
        R = az * t
        S.logA = _dd_add(P.Lz, _log_dd(aoz))
    else:
        y = 1.0
        for _k in range(m):
            y *= az
        R = sqrt(sqrt(1.0 + y)) if m == 4 else exp(log1p(y) / m)
        D = 0.0
        Rk = 1.0
        for _k in range(m):
            D += Rk
            Rk *= R
        A = y / D
        S.logA = _dd_add(_dd_mul_d(P.Lz, <double> m), _dd_neg(_log_dd(D)))
    q = az / R
    qm2 = 1.0
    for _k in range(m - 2):
        qm2 *= q
    Rp = 1.0
    for _k in range(m + 1):
        Rp *= R
    S.rA1 = P.sz * (qm2 * q) / A                   # A'/A,  A' = sgn(z) q**(m-1)
    # A''/A with A'' = (m - 1) q**(m-2) R**-(m+1).
    S.rA2 = ((m - 1) * qm2 / Rp) / A
    S.ok = 1


cdef struct fpieces:
    double c[8]
    int n


cdef struct curvwork:
    atomc* atoms
    piece* pc
    int* akey_k
    dd* akey_a1
    dd* akey_rem
    double* akey
    double* aerr
    fpieces* apc


cdef inline void _store_reference(
    Py_ssize_t ip, int k, dd a1, dd rem, dd hl, int* refk, double* ref_a1,
    double* ref_rem, double* half_log_norm
) noexcept nogil:
    if refk == NULL:
        return
    refk[ip] = k
    ref_a1[2 * ip] = a1.hi
    ref_a1[2 * ip + 1] = a1.lo
    ref_rem[2 * ip] = rem.hi
    ref_rem[2 * ip + 1] = rem.lo
    half_log_norm[2 * ip] = hl.hi
    half_log_norm[2 * ip + 1] = hl.lo


cdef inline void _store_zero(
    Py_ssize_t ip, double* H, double* err_h, double* log_m, int* refk,
    double* ref_a1, double* ref_rem, double* half_log_norm
) noexcept nogil:
    H[ip] = 0.0
    err_h[ip] = 0.0
    log_m[ip] = -INFINITY
    _store_reference(ip, 0, _DD0, _dd(-INFINITY, 0.0), _DD0, refk, ref_a1, ref_rem,
                     half_log_norm)


cdef int _fast_group(
    curvwork* W, int u, int k, dd a1, dd rem, double mult, const double* tv, int nt,
    double er, double z, double z2d, double s2h
) noexcept nogil:
    """Store one fast-path group ``u``: multipliers ``mult tv`` at one structure.

    Returns 0 when a multiplier is not finite or near underflow (the caller
    then uses the log path).
    """
    cdef fpieces* pc = &W.apc[u]
    cdef double pm = 0.0, v
    cdef int i
    pc.n = 0
    for i in range(nt):
        if tv[i] != 0.0:
            v = mult * tv[i]
            pc.c[pc.n] = v
            pc.n += 1
            if fabs(v) > pm:
                pm = fabs(v)
    if not isfinite(pm):
        return 0
    if pc.n > 0 and not (pm > 1e-280):
        return 0            # multiplier at or near underflow: use the log path
    W.akey_k[u] = k
    W.akey_a1[u] = a1
    W.akey_rem[u] = rem
    W.aerr[u] = er
    if pc.n == 0 or pm == 0.0:
        pc.n = 0
        W.akey[u] = -INFINITY
        return 1
    # Only the multiplier's own magnitude: groups are compared through their
    # structural differences (``_fast_d``), never through absolute keys whose
    # rounding grows with ``z**2``.
    W.akey[u] = ilogb(pm) * _LN2_HI
    return 1


cdef inline double _fast_d(
    const curvwork* W, int u, int r, double z, double z2d, double s2h
) noexcept nogil:
    """Approximate ``log|group u| - log|group r|`` (as ``_approx_d``)."""
    cdef int dk = W.akey_k[u] - W.akey_k[r]
    cdef double t2 = 0.0 if dk == 0 else -<double> dk * s2h * z2d
    cdef double t1 = (W.akey_a1[u].hi - W.akey_a1[r].hi) * z
    if isinf(t2):
        return t2
    if isinf(t1):
        return t1
    return t2 + t1 + (W.akey_rem[u].hi - W.akey_rem[r].hi) + (W.akey[u] - W.akey[r])


cdef int _fast_point(
    Py_ssize_t ip, const atomc* A, int na, const fam* F, ptstate* P,
    double* H, double* err_h, double* log_m, int* refk, double* ref_a1,
    double* ref_rem, double* half_log_norm, curvwork* W
) noexcept nogil:
    """Fast path at one point; returns 1 if handled, 0 to use the log path."""
    cdef int _k
    cdef double z = P.z
    cdef gfast G[4]
    cdef int gok[4]
    cdef spfast S
    cdef double s2h = F.s2half.hi
    cdef double z2d = z * z
    cdef double iz = 1.0 / z
    cdef int nu = 0, j, nsub, sub, u, k, gi, i, r, dk
    cdef const atomc* a
    cdef fpieces* pc
    cdef dd a1, rem, ze, D, da1, dr, tdd, hl
    cdef double mult, er, d_, p1, p2, x1, x2a, x2b, g1, g2, p, lam, zn
    cdef double S_, comp, ss, ea, sc, mag, rel, c, tt, nrm, lref
    cdef double tv[7]
    cdef const gfast* g
    if not (P.az >= _FAST_ZMIN and P.az <= _FAST_ZMAX):
        return 0
    gok[0] = gok[1] = gok[2] = gok[3] = 0
    S.ok = 0
    for j in range(na):
        a = &A[j]
        nsub = 2 if a.gate == GATE_WC else 1
        for sub in range(nsub):
            u = nu
            nu += 1
            pc = &W.apc[u]
            pc.n = 0
            k = 0
            a1 = _DD0
            rem = _DD0
            mult = a.theta
            er = 4.0
            if a.xtype == _XLOG:
                ze = _two_sum(z, -a.par)
                if ze.hi == 0.0:
                    return 0
                d_ = ze.hi + ze.lo
                pc.c[0] = mult / (d_ * d_)
                pc.n = 1
                if not isfinite(pc.c[0]) or not (fabs(pc.c[0]) > 1e-280):
                    return 0
                W.akey_k[u] = 0
                W.akey_a1[u] = _DD0
                W.akey_rem[u] = _DD0
                W.aerr[u] = 4.0
                W.akey[u] = ilogb(pc.c[0]) * _LN2_HI
                continue
            p1 = 0.0
            p2 = 0.0
            if a.n > 0:
                zn = z
                for _k in range(1, a.n):
                    zn *= z
                mult *= zn
                p1 = a.n * iz
                p2 = <double> a.n * (a.n - 1) * iz * iz
                er += a.n
            x1 = 0.0
            x2a = 0.0
            x2b = 0.0
            if a.xtype == _XSOFT:
                if not S.ok:
                    _softpower_fast(&S, P, F)
                p = a.par
                rem = _dd_mul_d(S.logA, p)
                x1 = p * S.rA1
                x2a = p * (p - 1.0) * S.rA1 * S.rA1
                x2b = p * S.rA2
                er += 6.0 + 2.0 * fabs(p)
            elif a.xtype == _XEXP:
                lam = a.par
                a1 = _dd(a.dir * lam, 0.0)
                rem = _dd(-lam, 0.0)
                x1 = a.dir * lam
                x2a = lam * lam
                er += 3.0
            g1 = 0.0
            g2 = 0.0
            gi = -1
            if a.gate >= GATE_GL and a.gate <= GATE_WU:
                gi = a.gate - 1
            elif a.gate == GATE_WC:
                # W_C = G_U - W_L for z >= 0, G_L - W_U below (exact identities).
                if z >= 0:
                    gi = 1 if sub == 0 else 2
                else:
                    gi = 0 if sub == 0 else 3
                if sub == 1:
                    mult = -mult
            g = NULL
            if gi >= 0:
                if not gok[gi]:
                    _gate_fast(&G[gi], P, F, _gate_sig(gi), _gate_c(gi))
                    gok[gi] = 1
                g = &G[gi]
                er += 6.0
            if g != NULL and g.plain:
                # Value pieces at the atom's own structure ...
                tv[0] = p2 * g.mult
                tv[1] = x2a * g.mult
                tv[2] = x2b * g.mult
                tv[3] = 2.0 * p1 * x1 * g.mult
                if not _fast_group(W, u, k, a1, rem, mult, tv, 4, er, z, z2d, s2h):
                    return 0
                # ... and derivative pieces at the structure of phi.
                u = nu
                nu += 1
                tv[0] = g.d2
                tv[1] = 2.0 * p1 * g.d1
                tv[2] = 2.0 * x1 * g.d1
                if not _fast_group(W, u, k + 1, _dd_add(a1, g.da1),
                                   _dd_add(rem, g.drem), mult, tv, 3, er, z, z2d,
                                   s2h):
                    return 0
                continue
            if g != NULL:
                k = g.k
                if g.k:
                    a1 = _dd_add(a1, g.a1)
                    rem = _dd_add(rem, g.rem)
                mult *= g.mult
                g1 = g.r1
                g2 = g.r2
            tv[0] = p2
            tv[1] = x2a
            tv[2] = x2b
            tv[3] = g2
            tv[4] = 2.0 * p1 * x1
            tv[5] = 2.0 * p1 * g1
            tv[6] = 2.0 * x1 * g1
            if not _fast_group(W, u, k, a1, rem, mult, tv, 7, er, z, z2d, s2h):
                return 0
    r = -1
    for u in range(nu):
        if W.akey[u] == -INFINITY:
            continue
        if r < 0 or _fast_d(W, u, r, z, z2d, s2h) > 0.0:
            r = u
    if r < 0:
        _store_zero(ip, H, err_h, log_m, refk, ref_a1, ref_rem, half_log_norm)
        return 1
    S_ = 0.0
    comp = 0.0
    ss = 0.0
    ea = 0.0
    for u in range(nu):
        if W.akey[u] == -INFINITY:
            continue
        if u != r and not (_fast_d(W, u, r, z, z2d, s2h) > _SKIP_LOG):
            continue
        sc = 1.0
        mag = 0.0
        if u != r:
            D = _DD0
            dk = W.akey_k[u] - W.akey_k[r]
            if dk != 0:
                tdd = _dd_mul(F.s2half, P.z2)
                D = _dd_neg(tdd) if dk > 0 else tdd
                mag += fabs(tdd.hi)
            da1 = _dd_add(W.akey_a1[u], _dd_neg(W.akey_a1[r]))
            if da1.hi != 0.0:
                tdd = _dd_mul_d(da1, z)
                D = _dd_add(D, tdd)
                mag += fabs(tdd.hi)
            dr = _dd_add(W.akey_rem[u], _dd_neg(W.akey_rem[r]))
            D = _dd_add(D, dr)
            mag += fabs(dr.hi)
            sc = exp(D.hi) * (1.0 + D.lo)
        rel = _DD_EPS * (W.aerr[u] + 1.0) + 8.0 * _DD_EPS * _DD_EPS * mag
        for i in range(W.apc[u].n):
            c = W.apc[u].c[i] * sc
            tt = S_ + c
            comp += (S_ - tt) + c if fabs(S_) >= fabs(c) else (c - tt) + S_
            S_ = tt
            ss += c * c
            ea += fabs(c) * rel
    S_ += comp
    if not (ss > 0.0) or not isfinite(ss):
        return 0
    nrm = sqrt(ss)
    H[ip] = S_ / nrm
    err_h[ip] = ea / nrm + 2.0 * _DD_EPS * fabs(H[ip])
    hl = _log_dd(nrm)
    lref = (-s2h * z2d if W.akey_k[r] else 0.0) + W.akey_a1[r].hi * z + W.akey_rem[r].hi
    log_m[ip] = lref + hl.hi
    _store_reference(ip, W.akey_k[r], W.akey_a1[r], W.akey_rem[r], hl, refk, ref_a1,
                     ref_rem, half_log_norm)
    return 1


cdef void _log_point(
    Py_ssize_t ip, const atomc* A, int na, const fam* F, ptstate* P,
    double* H, double* err_h, double* log_m, int* refk, double* ref_a1,
    double* ref_rem, double* half_log_norm, piece* pc
) noexcept nogil:
    """Log path at one point (every piece in structured log form)."""
    cdef double z = P.z
    cdef double z2d = z * z
    cdef double s2h = F.s2half.hi
    cdef int npiece = 0, j, i, r, dk
    cdef const atomc* a
    cdef factor fp, fx, one
    cdef piece* q
    cdef const piece* R
    cdef dd ze, D, tdd, da1, dr, hl
    cdef double Sm, comp, ss, ea, Da, mag, c, t, nrm, lref
    P.sp_ok = 0
    P.g_ok[0] = P.g_ok[1] = P.g_ok[2] = P.g_ok[3] = 0
    if not P.zzero and not P.lz_ok:
        P.Lz = _log_dd(P.az)
        P.lz_ok = 1
    _factor_one(&one)
    for j in range(na):
        a = &A[j]
        if a.xtype == _XLOG:
            # coef * (-log|z - e|): q'' = coef / (z - e)**2.
            ze = _two_sum(z, -a.par)
            if ze.hi == 0.0:
                continue
            q = &pc[npiece]
            npiece += 1
            q.sgn = a.sgnth
            q.k = 0
            q.a1 = _DD0
            q.rem = _dd_add(a.lth, _dd_mul_d(
                _log_of_dd(ze if ze.hi > 0 else _dd_neg(ze)), -2.0))
            q.err = 3.0
            continue
        _poly_factor(&fp, P, a)
        if a.xtype == _XSOFT:
            _softpower_factor(&fx, P, F, a)
        elif a.xtype == _XEXP:
            _exp_factor(&fx, a)
        else:
            fx = one
        if a.gate == GATE_NONE:
            _leibniz(pc, &npiece, &fp, &fx, &one, 1.0, a)
        elif a.gate <= GATE_WU:
            _leibniz(pc, &npiece, &fp, &fx, _get_gate(P, F, a.gate - 1), 1.0, a)
        elif z >= 0:
            # W_C = G_U - W_L for z >= 0, G_L - W_U below (exact identities).
            _leibniz(pc, &npiece, &fp, &fx, _get_gate(P, F, 1), 1.0, a)
            _leibniz(pc, &npiece, &fp, &fx, _get_gate(P, F, 2), -1.0, a)
        else:
            _leibniz(pc, &npiece, &fp, &fx, _get_gate(P, F, 0), 1.0, a)
            _leibniz(pc, &npiece, &fp, &fx, _get_gate(P, F, 3), -1.0, a)
    if npiece == 0:
        _store_zero(ip, H, err_h, log_m, refk, ref_a1, ref_rem, half_log_norm)
        return
    # Reference: the largest piece (exact order on extended log magnitudes).
    r = 0
    for i in range(1, npiece):
        if _approx_d(&pc[i], &pc[r], z, z2d, s2h) > 0.0:
            r = i
    R = &pc[r]
    Sm = 0.0
    comp = 0.0
    ss = 0.0
    ea = 0.0
    for i in range(npiece):
        q = &pc[i]
        Da = 0.0 if i == r else _approx_d(q, R, z, z2d, s2h)
        if not (Da > _SKIP_LOG):
            continue
        D = _DD0
        mag = 0.0
        dk = q.k - R.k
        if dk != 0:
            tdd = _dd_mul(F.s2half, P.z2)
            D = _dd_neg(tdd) if dk > 0 else tdd
            mag += fabs(tdd.hi)
        da1 = _dd_add(q.a1, _dd_neg(R.a1))
        if da1.hi != 0.0:
            tdd = _dd_mul_d(da1, z)
            D = _dd_add(D, tdd)
            mag += fabs(tdd.hi)
        dr = _dd_add(q.rem, _dd_neg(R.rem))
        D = _dd_add(D, dr)
        mag += fabs(dr.hi)
        c = q.sgn * exp(D.hi) * (1.0 + D.lo)
        t = Sm + c                                  # Neumaier summation
        comp += (Sm - t) + c if fabs(Sm) >= fabs(c) else (c - t) + Sm
        Sm = t
        ss += c * c
        ea += fabs(c) * (_DD_EPS * q.err + 8.0 * _DD_EPS * _DD_EPS * mag)
    Sm += comp
    nrm = sqrt(ss)
    H[ip] = Sm / nrm
    err_h[ip] = ea / nrm + 2.0 * _DD_EPS * fabs(H[ip])
    hl = _log_dd(nrm)
    lref = (-s2h * z2d if R.k else 0.0) + R.a1.hi * z + R.rem.hi
    log_m[ip] = lref + hl.hi
    _store_reference(ip, R.k, R.a1, R.rem, hl, refk, ref_a1, ref_rem, half_log_norm)


cdef inline void _family(const BasisTable* t, fam* F) noexcept nogil:
    F.s = t.gate_s
    F.m = t.root
    F.s2 = _two_prod(F.s, F.s)
    F.s2half = _dd_mul_d(F.s2, 0.5)
    F.logs = _log_dd(F.s)
    F.log_s2 = _dd_mul_d(F.logs, 2.0)
    F.log_m1 = _log_dd(<double> (F.m - 1))


cdef void _atom_constants(
    const BasisTable* t, int i, double coef, atomc* a
) noexcept nogil:
    """Per-atom constants of curvature atom ``i`` with coefficient ``coef``."""
    cdef int j = t.atom_fn[i]
    cdef double p
    memset(a, 0, sizeof(atomc))
    a.gate = t.atom_gate[i]
    a.par = t.atom_par[i]
    a.dir = 1 if t.side[j] == SIDE_UPPER else -1
    if t.kind[j] == FN_MONOMIAL:
        a.xtype = _XPOLY
        a.n = t.degree[j]
    elif t.kind[j] == FN_SOFT:
        a.xtype = _XSOFT
    elif t.kind[j] == FN_EXP:
        a.xtype = _XEXP
    else:
        a.xtype = _XLOG
    a.sgnth = 1.0 if coef > 0 else -1.0
    a.theta = coef
    a.lth = _log_dd(fabs(coef))
    if a.n >= 1:
        a.logn = _log_dd(<double> a.n)
    if a.n >= 2:
        a.lognn1 = _log_dd(<double> a.n * (a.n - 1))
    if a.xtype == _XSOFT:
        p = a.par
        a.logp = _log_dd(p)
        a.sgnpm1 = 1.0 if p > 1.0 else (-1.0 if p < 1.0 else 0.0)
        if p != 1.0:
            a.logpm1 = _log_dd(fabs(p - 1.0))
    elif a.xtype == _XEXP:
        a.loglam = _log_dd(a.par)


cdef void _free_work(curvwork* W) noexcept nogil:
    free(W.atoms)
    free(W.pc)
    free(W.akey_k)
    free(W.akey_a1)
    free(W.akey_rem)
    free(W.akey)
    free(W.aerr)
    free(W.apc)


cdef int _alloc_work(curvwork* W, int na) noexcept nogil:
    """Allocate the per-call buffers for ``na`` atoms; ``-1`` on failure."""
    cdef size_t n = <size_t> (na if na > 0 else 1)
    W.atoms = <atomc*> malloc(sizeof(atomc) * n)
    W.pc = <piece*> malloc(sizeof(piece) * n * _MAXP_PER_ATOM)
    # Fast groups: two sub-gates of W_C, each split into value and derivative.
    W.akey_k = <int*> malloc(sizeof(int) * 4 * n)
    W.akey_a1 = <dd*> malloc(sizeof(dd) * 4 * n)
    W.akey_rem = <dd*> malloc(sizeof(dd) * 4 * n)
    W.akey = <double*> malloc(sizeof(double) * 4 * n)
    W.aerr = <double*> malloc(sizeof(double) * 4 * n)
    W.apc = <fpieces*> malloc(sizeof(fpieces) * 4 * n)
    if (W.atoms == NULL or W.pc == NULL or W.akey_k == NULL or W.akey_a1 == NULL
            or W.akey_rem == NULL or W.akey == NULL or W.aerr == NULL or W.apc == NULL):
        _free_work(W)
        return -1
    return 0


cdef void _scaled_points(
    const BasisTable* t, curvwork* W, int na, const fam* F, Py_ssize_t npts,
    const double* zin, bint use_fast, double* H, double* err_h, double* log_m,
    int* refk, double* ref_a1, double* ref_rem, double* half_log_norm
) noexcept nogil:
    """Evaluate the ``na`` prepared atoms of ``W.atoms`` at every point."""
    cdef Py_ssize_t ip
    cdef ptstate P
    cdef double z
    for ip in range(npts):
        z = zin[ip]
        P.z = z
        P.az = fabs(z)
        P.sz = -1.0 if z < 0 else 1.0
        P.zzero = z == 0.0
        P.lz_ok = 0
        P.z2 = _two_prod(z, z)
        if use_fast and not P.zzero and _fast_point(
            ip, W.atoms, na, F, &P, H, err_h, log_m, refk, ref_a1, ref_rem,
            half_log_norm, W,
        ):
            continue
        _log_point(ip, W.atoms, na, F, &P, H, err_h, log_m, refk, ref_a1, ref_rem,
                   half_log_norm, W.pc)


cdef int curvature_scaled(
    const BasisTable* t,
    const double* acoef,
    Py_ssize_t npts,
    const double* z,
    bint use_fast,
    double* H,
    double* err_h,
    double* log_m,
    int* refk,
    double* ref_a1,
    double* ref_rem,
    double* half_log_norm,
) noexcept nogil:
    """Structured scaled curvature at ``npts`` canonical points.

    ``acoef[i]`` is the coefficient of curvature atom ``i`` (``atom_row @
    theta``); zero atoms are skipped.  ``refk`` may be NULL (no reference
    output; then ``ref_a1``, ``ref_rem`` and ``half_log_norm`` are unused,
    otherwise each holds ``2 npts`` doubles as ``(hi, lo)`` pairs).
    ``use_fast`` selects the fast path where it applies (both paths give the
    same result to rounding).  Returns ``-1`` on allocation failure.
    """
    cdef curvwork W
    cdef fam F
    cdef int i, na = 0
    if _alloc_work(&W, t.natom) != 0:
        return -1
    _family(t, &F)
    for i in range(t.natom):
        if acoef[i] != 0.0:
            _atom_constants(t, i, acoef[i], &W.atoms[na])
            na += 1
    _scaled_points(t, &W, na, &F, npts, z, use_fast, H, err_h, log_m, refk, ref_a1,
                   ref_rem, half_log_norm)
    _free_work(&W)
    return 0


cdef int curvature_rows(
    const BasisTable* t,
    Py_ssize_t npts,
    const double* z,
    bint use_fast,
    double* R,
    double* N,
    int* refk,
    double* ref_a1,
    double* ref_rem,
) noexcept nogil:
    """Theta-free per-atom curvature rows at ``npts`` canonical points.

    ``R[ip * A + i] = phi_i''(z) / S(z)`` and ``N[ip * A + i] = ||pieces_i|| /
    S(z)`` for every curvature atom ``i`` with unit coefficient, where the
    common positive scale ``S = exp(-k (s**2/2) z**2 + a1 z + rem)`` (returned
    as ``refk``, ``ref_a1``, ``ref_rem``, double-double pairs) is the largest
    atom magnitude at the point.  For any atom coefficients ``c``,
    ``q'' = S sum_i c_i R_i`` and ``H = sum_i c_i R_i / ||c N||``.  An atom
    whose pieces lie below ``exp(-700)`` of the largest atom is beyond the
    rows' dynamic range at that point: it is stored as ``R = 0`` and
    ``N = -1`` (an atom without pieces has ``R = N = 0``), so a caller can
    tell when the atoms it weights are all out of range and the rows do not
    determine ``H``.  Returns ``-1`` on allocation failure.
    """
    cdef Py_ssize_t A = t.natom
    cdef Py_ssize_t ip, q, p
    cdef int i, r, dk
    cdef curvwork W
    cdef fam F
    cdef double* hj
    cdef double* eh
    cdef double* lm
    cdef int* kj
    cdef double* a1j
    cdef double* remj
    cdef double* hl
    cdef double zz, z2d, t2, t1, Da, nj
    cdef dd rr, hh, z2, D, pa1, prem, qa1, qrem, da1
    if A == 0:
        for ip in range(npts):
            refk[ip] = 0
            ref_a1[2 * ip] = 0.0
            ref_a1[2 * ip + 1] = 0.0
            ref_rem[2 * ip] = -INFINITY
            ref_rem[2 * ip + 1] = 0.0
        return 0
    if _alloc_work(&W, 1) != 0:
        return -1
    hj = <double*> malloc(sizeof(double) * npts * A)
    eh = <double*> malloc(sizeof(double) * npts)
    lm = <double*> malloc(sizeof(double) * npts)
    kj = <int*> malloc(sizeof(int) * npts * A)
    a1j = <double*> malloc(sizeof(double) * 2 * npts * A)
    remj = <double*> malloc(sizeof(double) * 2 * npts * A)
    hl = <double*> malloc(sizeof(double) * 2 * npts)
    if (hj == NULL or eh == NULL or lm == NULL or kj == NULL or a1j == NULL
            or remj == NULL or hl == NULL):
        free(hj)
        free(eh)
        free(lm)
        free(kj)
        free(a1j)
        free(remj)
        free(hl)
        _free_work(&W)
        return -1
    _family(t, &F)
    for i in range(A):
        _atom_constants(t, i, 1.0, &W.atoms[0])
        _scaled_points(t, &W, 1, &F, npts, z, use_fast, &hj[i * npts], eh, lm,
                       &kj[i * npts], &a1j[2 * i * npts], &remj[2 * i * npts], hl)
        for ip in range(npts):
            # Fold the piece norm into rem.  An atom without pieces already has
            # rem = -inf; one whose pieces cancel exactly (H = 0) keeps its norm.
            q = i * npts + ip
            if not isfinite(remj[2 * q]):
                continue        # (-inf + x in double-double would give NaN)
            rr = _dd_add(_dd(remj[2 * q], remj[2 * q + 1]),
                         _dd(hl[2 * ip], hl[2 * ip + 1]))
            remj[2 * q] = rr.hi
            remj[2 * q + 1] = rr.lo
    for ip in range(npts):
        zz = z[ip]
        z2d = zz * zz
        z2 = _two_prod(zz, zz)
        r = -1
        for i in range(A):
            q = i * npts + ip
            if remj[2 * q] == -INFINITY:
                continue
            if r < 0:
                r = i
                continue
            p = r * npts + ip
            dk = kj[q] - kj[p]
            t2 = 0.0 if dk == 0 else -<double> dk * F.s2half.hi * z2d
            t1 = (a1j[2 * q] - a1j[2 * p]) * zz
            Da = t2 if isinf(t2) else (
                t1 if isinf(t1) else t2 + t1 + (remj[2 * q] - remj[2 * p]))
            if Da > 0:
                r = i
        if r < 0:
            for i in range(A):
                R[ip * A + i] = 0.0
                N[ip * A + i] = 0.0
            refk[ip] = 0
            ref_a1[2 * ip] = 0.0
            ref_a1[2 * ip + 1] = 0.0
            ref_rem[2 * ip] = -INFINITY
            ref_rem[2 * ip + 1] = 0.0
            continue
        p = r * npts + ip
        pa1 = _dd(a1j[2 * p], a1j[2 * p + 1])
        prem = _dd(remj[2 * p], remj[2 * p + 1])
        for i in range(A):
            q = i * npts + ip
            if remj[2 * q] == -INFINITY:
                R[ip * A + i] = 0.0
                N[ip * A + i] = 0.0
                continue
            dk = kj[q] - kj[p]
            t2 = 0.0 if dk == 0 else -<double> dk * F.s2half.hi * z2d
            t1 = (a1j[2 * q] - a1j[2 * p]) * zz
            Da = t2 if isinf(t2) else (
                t1 if isinf(t1) else t2 + t1 + (remj[2 * q] - remj[2 * p]))
            if not (Da > -700.0):
                R[ip * A + i] = 0.0
                N[ip * A + i] = -1.0
                continue
            D = _DD0
            if dk != 0:
                hh = _dd_mul(F.s2half, z2)
                D = _dd_neg(hh) if dk > 0 else hh
            qa1 = _dd(a1j[2 * q], a1j[2 * q + 1])
            qrem = _dd(remj[2 * q], remj[2 * q + 1])
            da1 = _dd_add(qa1, _dd_neg(pa1))
            if da1.hi != 0.0:
                D = _dd_add(D, _dd_mul_d(da1, zz))
            D = _dd_add(D, _dd_add(qrem, _dd_neg(prem)))
            nj = exp(D.hi) * (1.0 + D.lo)          # ||pieces_i|| / S
            N[ip * A + i] = nj
            R[ip * A + i] = hj[q] * nj
        refk[ip] = kj[p]
        ref_a1[2 * ip] = pa1.hi
        ref_a1[2 * ip + 1] = pa1.lo
        ref_rem[2 * ip] = prem.hi
        ref_rem[2 * ip + 1] = prem.lo
    free(hj)
    free(eh)
    free(lm)
    free(kj)
    free(a1j)
    free(remj)
    free(hl)
    _free_work(&W)
    return 0


# --------------------------------------------------------------------------
# tail atoms and verdicts
# --------------------------------------------------------------------------

cdef class TailKernel:
    """Flat atoms of one ``_TailModel`` (``basis_tails``).

    Every atom and envelope weight is ``zeta**power exp(rate zeta - kappa
    zeta**2 + const)``; ``kappa > 0`` for the Gaussian gate deviations, and
    the constant is kept in log form so no row underflows.

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
        akappa = np.array([a.kappa for a in tail.atoms], dtype=np.float64)
        aconst = np.array([a.const for a in tail.atoms], dtype=np.float64)
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
        ekappa = np.array([e.kappa for e in tail.envelopes], dtype=np.float64)
        econst = np.array([e.const for e in tail.envelopes], dtype=np.float64)
        arrays = [akind, apower, arate, akappa, aconst, aoffset, arow, acut, ecol,
                  ecoef, epower, erate, ekappa, econst]
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
        self.t.akappa = <const double*> cnp.PyArray_DATA(arrays[3])
        self.t.aconst = <const double*> cnp.PyArray_DATA(arrays[4])
        self.t.aoffset = <const double*> cnp.PyArray_DATA(arrays[5])
        self.t.arow = <const double*> cnp.PyArray_DATA(arrays[6])
        self.t.acut = <const double*> cnp.PyArray_DATA(arrays[7])
        self.t.ecol = <const int*> cnp.PyArray_DATA(arrays[8])
        self.t.ecoef = <const double*> cnp.PyArray_DATA(arrays[9])
        self.t.epower = <const double*> cnp.PyArray_DATA(arrays[10])
        self.t.erate = <const double*> cnp.PyArray_DATA(arrays[11])
        self.t.ekappa = <const double*> cnp.PyArray_DATA(arrays[12])
        self.t.econst = <const double*> cnp.PyArray_DATA(arrays[13])

    def cut_row(self, int atom):
        """Cut row of a violating atom (its own row, or the full-space row)."""
        return np.array(self._keep[7][atom], dtype=np.float64)


cdef inline TailResult _empty_result() noexcept nogil:
    cdef TailResult r
    r.status = TAIL_INCONCLUSIVE
    r.cutoff = NAN
    r.zeta = NAN
    r.lead_value = NAN
    r.lead_power = NAN
    r.lead_rate = NAN
    r.lead_kappa = NAN
    r.lead_const = NAN
    r.margin = NAN
    r.has_lead = False
    r.has_margin = False
    r.has_cutoff = False
    r.cut_atom = -1
    return r


cdef enum:
    _PIECE = 5      # power, rate, kappa, const, magnitude


cdef inline bint _stronger(
    double kappa, double rate, double power, double big_k, double big_r, double big_b
) noexcept nogil:
    """Whether a weight outgrows the leader's (lexicographic in ``-kappa, rate,
    power``)."""
    if kappa != big_k:
        return kappa < big_k
    if rate != big_r:
        return rate > big_r
    return power > big_b


cdef inline double _crossover(
    double db, double dr, double dk
) noexcept nogil:
    """Smallest ``zeta >= 0`` beyond which a weaker ratio is nonincreasing.

    The log ratio has derivative ``db / zeta + dr - 2 dk zeta`` (``db``, ``dr``
    are the competitor's power and rate excess, ``dk >= 0`` its extra
    Gaussian rate); it is nonpositive beyond the positive root of
    ``2 dk zeta**2 - dr zeta - db``.
    """
    if dk > 0.0:
        if db <= 0.0 and dr <= 0.0:
            return 0.0
        return (dr + sqrt(dr * dr + 8.0 * dk * npmax(db, 0.0))) / (4.0 * dk)
    if dr < 0.0 and db > 0.0:
        return db / (-dr)
    return 0.0


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
    cdef int i, q
    cdef double v, sl, mag, m, big_b, big_r, big_k, big_c, zeta, keep, factor, total
    cdef double b, r, kk, log_w
    cdef TailResult res = _empty_result()
    for i in range(nenv):
        for q in range(_PIECE):
            pieces[_PIECE * npc + q] = env[_PIECE * i + q]
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
        pieces[_PIECE * npc] = tt.apower[i]
        pieces[_PIECE * npc + 1] = tt.arate[i]
        pieces[_PIECE * npc + 2] = tt.akappa[i]
        pieces[_PIECE * npc + 3] = tt.aconst[i]
        pieces[_PIECE * npc + 4] = mag
        npc += 1
    big_b = tt.apower[lead]
    big_r = tt.arate[lead]
    big_k = tt.akappa[lead]
    big_c = tt.aconst[lead]
    res.status = TAIL_INCONCLUSIVE
    res.lead_value = lead_lo
    res.lead_power = big_b
    res.lead_rate = big_r
    res.lead_kappa = big_k
    res.lead_const = big_c
    res.has_lead = True
    zeta = tt.zeta_min
    for i in range(npc):
        b = pieces[_PIECE * i]
        r = pieces[_PIECE * i + 1]
        kk = pieces[_PIECE * i + 2]
        if _stronger(kk, r, b, big_k, big_r, big_b):
            return res
        zeta = npmax(zeta, _crossover(b - big_b, r - big_r, kk - big_k))
    keep = 1.0 - npmax(dominance, 1e-12)
    while zeta <= zeta_cap:
        # Leader weights below exp(-575) are not trusted, except Gaussian
        # leaders, whose comparisons are relative and exact in log form.
        log_w = big_b * log(zeta) + big_r * zeta
        if big_k == 0.0 and log_w < _LOG_UNDERFLOW:
            break
        if tt.akind[lead] == 1 and tt.aoffset[lead] > 0.0:
            factor = pow(zeta / (zeta + tt.aoffset[lead]), -tt.apower[lead])
        else:
            factor = 1.0
        total = 0.0
        for i in range(npc):
            total += exp(
                log(pieces[_PIECE * i + 4])
                + (pieces[_PIECE * i] - big_b) * log(zeta)
                + (pieces[_PIECE * i + 1] - big_r) * zeta
                - (pieces[_PIECE * i + 2] - big_k) * zeta * zeta
                + (pieces[_PIECE * i + 3] - big_c)
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
    return 2 * tt.na + _PIECE * (tt.na + 2 * tt.ne) + _PIECE


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
    cdef double* pieces = work + 2 * na + _PIECE * tt.ne
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
                # excluded every member with linear tails at both ends); its
                # margin moves the point to a positive leader.
                res.cut_atom = i if zero_lead < 0 else zero_lead
                return res
            first = i
    for i in range(tt.ne):
        mag = fabs(theta[tt.ecol[i]]) * tt.ecoef[i]
        if mag > 0.0:
            env[_PIECE * nenv] = tt.epower[i]
            env[_PIECE * nenv + 1] = tt.erate[i]
            env[_PIECE * nenv + 2] = tt.ekappa[i]
            env[_PIECE * nenv + 3] = tt.econst[i]
            env[_PIECE * nenv + 4] = mag
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
        if growth and _decaying(tt.apower[k], tt.arate[k], tt.akappa[k]):
            continue
        cand = _lead_proof(
            tt, values, slacks, k, env, nenv, zeta_cap, dominance, pieces
        )
        if cand.status == TAIL_NONNEGATIVE and (
            best.status != TAIL_NONNEGATIVE or cand.zeta < best.zeta
        ):
            best = cand
    return best


cdef inline bint _decaying(double power, double rate, double kappa) noexcept nogil:
    """Whether a leading slope weight decays (gives no outward growth)."""
    return kappa > 0.0 or rate < 0.0 or (rate == 0.0 and power < 0.0)


cdef int growth_status(const TailResult* res) noexcept nogil:
    """Properness from an order-1 verdict (``basis_tails._growth_verdict``)."""
    if res.status == TAIL_VIOLATED:
        return GROWTH_IMPROPER
    if res.has_lead and _decaying(res.lead_power, res.lead_rate, res.lead_kappa):
        # A positive but decaying leading slope is integrable.
        return GROWTH_IMPROPER
    if res.status == TAIL_INCONCLUSIVE:
        return GROWTH_INCONCLUSIVE
    if not res.has_margin:
        # Identically zero slope: q tends to a finite limit.
        return GROWTH_IMPROPER
    return GROWTH_PROPER


cdef double tail_bound(const TailResult* res, double zeta) noexcept nogil:
    """``margin zeta**b exp(r zeta - kappa zeta**2 + const)`` (``_TailResult.bound``)."""
    cdef double log_w
    if not res.has_margin:
        return 0.0
    log_w = (res.lead_power * log(zeta) + res.lead_rate * zeta
             - res.lead_kappa * zeta * zeta + res.lead_const)
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


def potential_z(BasisKernel kernel, const double[::1] theta, const double[::1] z,
                int max_order):
    """``q`` and its ``z`` derivatives at canonical points: ``(max_order + 1, R)``.

    The points are never rounded through user units (``potential_point_z``),
    so a support a few ulps wide is resolved.
    """
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = z.shape[0]
    cdef Py_ssize_t i
    cdef int k
    cdef double hk
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
        potential_point_z(t, &theta[0], z[i], max_order, &fwork[0], &cwork[0], q)
        hk = 1.0
        for k in range(max_order + 1):
            o[k, i] = q[k] * hk
            hk = hk * t.h
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
    lead_rate, lead_kappa, lead_const, margin, cut_atom)`` (``None`` for absent
    values)."""
    if theta.shape[0] != tail.t.d:
        raise ValueError("theta has the wrong length")
    work = np.empty(tail_work_size(&tail.t))
    cdef double[::1] w = work
    cdef TailResult r = tail_verdict(
        &tail.t, &theta[0], zeta_cap, dominance, growth, &w[0]
    )
    status = ("verified", "violated", "unverified")[r.status]
    return (
        status,
        r.cutoff if r.has_cutoff else None,
        r.zeta if r.has_cutoff else None,
        None if isnan(r.lead_value) else r.lead_value,
        r.lead_power if r.has_lead else None,
        r.lead_rate if r.has_lead else None,
        r.lead_kappa if r.has_lead else None,
        r.lead_const if r.has_lead else None,
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


def curvature(BasisKernel kernel, const double[::1] theta, const double[::1] z,
              bint use_fast=True, bint reference=False):
    """Structured scaled curvature ``(H, err_h, log_m)`` at canonical points ``z``.

    ``H = q'' / ||pieces||`` in ``z`` units has the exact sign of ``q''``;
    ``log_m`` is the natural log of the piece norm (``q''_zz = H exp(log_m)``),
    possibly infinite when the scale itself leaves binary64 range.
    ``use_fast=False`` forces the log path (a test hook; both agree).  With
    ``reference=True`` the exact scale is appended as ``(refk, ref_a1,
    ref_rem, half_log_norm)``: ``log_m = -refk (s**2/2) z**2 + a1 z + rem +
    half_log_norm`` with double-double ``(n, 2)`` parts.
    """
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = z.shape[0]
    cdef int rc
    if theta.shape[0] != t.d:
        raise ValueError("theta has the wrong length")
    acoef = np.ascontiguousarray(kernel.atom_rows @ np.asarray(theta))
    if acoef.size == 0:
        acoef = np.zeros(1)
    H = np.empty(max(n, 1))
    err = np.empty(max(n, 1))
    logm = np.empty(max(n, 1))
    cdef double[::1] av = acoef
    cdef double[::1] hv = H
    cdef double[::1] ev = err
    cdef double[::1] lv = logm
    refk = np.zeros(max(n, 1), dtype=np.intc)
    a1 = np.zeros((max(n, 1), 2))
    rem = np.zeros((max(n, 1), 2))
    hl = np.zeros((max(n, 1), 2))
    cdef int[::1] kv = refk
    cdef double[:, ::1] a1v = a1
    cdef double[:, ::1] remv = rem
    cdef double[:, ::1] hlv = hl
    cdef int* kp = &kv[0] if reference else NULL
    cdef const double* zp = &z[0] if n else NULL
    with nogil:
        rc = curvature_scaled(t, &av[0], n, zp, use_fast, &hv[0], &ev[0], &lv[0],
                              kp, &a1v[0, 0], &remv[0, 0], &hlv[0, 0])
    if rc != 0:
        raise MemoryError("curvature work buffers")
    if reference:
        return H[:n], err[:n], logm[:n], refk[:n], a1[:n], rem[:n], hl[:n]
    return H[:n], err[:n], logm[:n]


def curvature_atom_rows(BasisKernel kernel, const double[::1] z, bint use_fast=True):
    """Theta-free per-atom curvature rows at canonical points ``z``.

    Returns ``(R, N, refk, ref_a1, ref_rem)``: ``R`` and ``N`` of shape
    ``(n, A)`` (see ``curvature_rows``), ``refk`` of shape ``(n,)`` and the
    double-double scale parts ``ref_a1``, ``ref_rem`` of shape ``(n, 2)``.
    """
    cdef const BasisTable* t = &kernel.t
    cdef Py_ssize_t n = z.shape[0]
    cdef Py_ssize_t A = t.natom
    cdef int rc
    R = np.zeros((n, A))
    N = np.zeros((n, A))
    refk = np.zeros(max(n, 1), dtype=np.intc)
    a1 = np.zeros((max(n, 1), 2))
    rem = np.zeros((max(n, 1), 2))
    if n == 0:
        return R, N, refk[:0], a1[:0], rem[:0]
    Rb = np.zeros(n * A + 1)
    Nb = np.zeros(n * A + 1)
    cdef double[::1] rv = Rb
    cdef double[::1] nv = Nb
    cdef int[::1] kv = refk
    cdef double[:, ::1] av = a1
    cdef double[:, ::1] mv = rem
    cdef const double* zp = &z[0]
    with nogil:
        rc = curvature_rows(t, n, zp, use_fast, &rv[0], &nv[0], &kv[0], &av[0, 0],
                            &mv[0, 0])
    if rc != 0:
        raise MemoryError("curvature work buffers")
    if A:
        R = Rb[: n * A].reshape(n, A).copy()
        N = Nb[: n * A].reshape(n, A).copy()
    return R, N, refk[:n], a1[:n], rem[:n]


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
