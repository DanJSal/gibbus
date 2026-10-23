"""Outward-rounded interval enclosures of basis columns and their derivatives.

G1 owner replaced: the rounding-margin arithmetic of the Bernstein
curvature certificate (``_fit/_curvature_certificate.pyx``), which applies only
to polynomials; G3 encloses general columns cell by cell instead.

For cells ``X_i = [lo_i, hi_i]`` (degenerate cells give point enclosures) this
module returns intervals guaranteed to contain ``Phi_c^(k)(x)`` for every
``x`` in ``X_i``, built by interval arithmetic on the same closed forms the
evaluator uses: monotone ranges for ``z``, ``exp``, ``log``, ``tanh`` and
``expit``, interval powers, and the tanh derivative polynomials in ``t`` and
``w = 1 - t**2 = p m``.

Rounding: every arithmetic result is widened by a relative ``4u`` plus one
ulp (``numpy.nextafter``) -- covering correct rounding and the rounding of
float scalar constants -- and every transcendental (``exp``, ``log``,
``tanh``, ``expit``, non-integer powers) by ``8u`` plus one ulp, to cover
correctly rounded arithmetic and the accuracy of the platform's libm.  These
are floating-point enclosures, recorded as numerical evidence (G3-028): no
arbitrary-precision verification is claimed.

The arithmetic runs in the compiled ``_basis_kernels``
(``basis_enclosure_cell``, ``potential_enclosure_cell``), built without
floating-point contraction so that each operation rounds once. Ungated
polynomial/log curvature at a point is grouped into an extended-precision
Horner chain when the whole potential has that ungated form. In this point
calculation, exact single-entry +/-1 coefficient maps carry no formation error;
general maps retain their rounding bounds. The curvature search can resolve
ambiguous ungated polynomial/log contact points using exact rational arithmetic.
"""

from fractions import Fraction

import numpy as np

from ._basis_kernels import basis_enclosure, potential_enclosure


def _exact_ungated_curvature(model, theta, points):
    """Directed binary64 point bounds from exact rational polynomial arithmetic.

    This is a precision fallback for resolved but ambiguous contact points.
    Unsupported functions or nonzero gates return ``None``.  The exact model
    map, stored coefficients and stored geometry define the rational input.
    """
    coefficients = [
        sum((v * Fraction(float(a)) for v, a in zip(row, theta)), Fraction(0))
        for row in model.coefficient_map_exact
    ]
    if any(
        coefficients[3 * j + 1] or coefficients[3 * j + 2]
        for j in range(len(model.functions))
    ):
        return None
    polynomial = {}
    logarithms = []
    for fn in model.functions:
        coefficient = coefficients[3 * fn.index]
        if not coefficient:
            continue
        if fn.kind == "monomial":
            if fn.degree >= 2:
                polynomial[fn.degree - 2] = coefficient * fn.degree * (fn.degree - 1)
        elif fn.kind == "log":
            endpoint = (
                model.request.lower if fn.side == "lower" else model.request.upper
            )
            logarithms.append((Fraction(float(endpoint)), coefficient))
        else:
            return None
    h, c = Fraction(model.geometry.scale), Fraction(model.geometry.center)
    degree = max(polynomial, default=0)
    result = []
    for point in points:
        x = Fraction(float(point))
        z = (x - c) / h
        value = Fraction(0)
        for n in range(degree, -1, -1):
            value = value * z + polynomial.get(n, Fraction(0))
        value /= h * h
        for endpoint, coefficient in logarithms:
            if x == endpoint:
                return None
            value += coefficient / (x - endpoint) ** 2
        nearest = float(value)
        if not np.isfinite(nearest):
            return None
        lower = np.nextafter(nearest, -np.inf) if Fraction(nearest) > value else nearest
        upper = np.nextafter(nearest, np.inf) if Fraction(nearest) < value else nearest
        result.append((nearest, float(lower), float(upper)))
    return result


def _cells(lo, hi, /):
    """Contiguous ``float64`` cell endpoint arrays.

    Parameters
    ----------
    lo, hi : array_like
        Cell endpoints.
    """
    lo = np.ascontiguousarray(np.asarray(lo, dtype=np.float64).reshape(-1))
    hi = np.ascontiguousarray(np.asarray(hi, dtype=np.float64).reshape(-1))
    return lo, hi


def _basis_enclosure(model, lo, hi, orders, /, columns=None):
    """Enclose ``Phi_c^(k)`` over cells ``[lo_i, hi_i]`` for each ``k`` in ``orders``.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    lo, hi : numpy.ndarray
        Cell endpoints in user units, shape ``(N,)``, ``lo <= hi``, inside the
        closed support.  Cells touching an endpoint with an enabled log give
        infinite enclosures for that column.
    orders : iterable of int
        Derivative orders in ``0..4``.
    columns : iterable of int or None
        Columns to enclose (default all); the rest are left as ``nan``.

    Returns
    -------
    dict
        ``{k: (LO, HI)}`` with arrays of shape ``(N, d)``.
    """
    orders = sorted(set(orders))
    lo, hi = _cells(lo, hi)
    out_lo, out_hi = basis_enclosure(model.kernel, lo, hi, orders[-1], columns)
    return {k: (out_lo[k], out_hi[k]) for k in orders}


def _potential_enclosure(model, theta, lo, hi, orders, /):
    """Enclose ``q^(k)`` over cells for coefficient vector ``theta``.

    ``q`` is enclosed function by function through the exact map
    ``full = M theta``: ``q = sum_j f_j (a_j + b_j t_L + c_j t_U)``, one
    combined gate per function, in the same cancellation-free lower/upper
    forms the evaluator uses.  Coefficients that cancel across columns (for
    example several gated exponentials whose leading amplitudes nearly sum to
    zero) therefore cancel before any interval product, instead of adding
    their widths.  Each slot of ``full`` carries the dot-product rounding
    bound ``(d + 1) u sum_c |M_sc theta_c|``; exactly zero slots stay exactly
    zero, so a function with no coefficient (e.g. a disabled endpoint log) is
    skipped and never produces ``0 * inf``.  Each gate limit
    ``a + sb b + sc c`` is enclosed as one sum per function, so a structurally
    exact zero (e.g. the upper limit of ``tL + tU``) stays exact.
    Specialized polynomial point bounds handle exact unit maps separately.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``.
    lo, hi : numpy.ndarray
        Cell endpoints, shape ``(N,)``.
    orders : iterable of int
        Derivative orders.

    Returns
    -------
    dict
        ``{k: (qlo, qhi)}`` with arrays of shape ``(N,)``.
    """
    orders = sorted(set(orders))
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    lo, hi = _cells(lo, hi)
    out_lo, out_hi = potential_enclosure(model.kernel, theta, lo, hi, orders[-1])
    return {k: (out_lo[k], out_hi[k]) for k in orders}
