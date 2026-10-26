"""Outward-rounded interval enclosures of basis columns and their derivatives.

For cells ``X_i = [lo_i, hi_i]`` (degenerate cells give point enclosures) this
module returns intervals guaranteed to contain ``Phi_c^(k)(x)`` for every
``x`` in ``X_i``, built by interval arithmetic on the same closed forms the
evaluator uses: monotone ranges for ``z``, ``exp``, ``log`` and the gate
weights (``erfc``), Hermite polynomials times the Gaussian density for gate
derivatives, and positive powers of monotone quantities for soft powers.
Soft-power derivative enclosures beyond order two are the whole line.

Rounding: every arithmetic result is widened by a relative ``4u`` plus one
ulp -- covering correct rounding and the rounding of float scalar constants
-- and every transcendental (``exp``, ``log``, ``erfc``, non-integer powers)
by ``8u`` plus one ulp, to cover correctly rounded arithmetic and the
accuracy of the platform's libm.  These are floating-point enclosures,
recorded as numerical evidence: no arbitrary-precision verification is
claimed.

The arithmetic runs in the compiled ``_basis_kernels``
(``basis_enclosure_cell``, ``potential_enclosure_cell``), built without
floating-point contraction so that each operation rounds once. Ungated
polynomial/log curvature at a point is grouped into an extended-precision
Horner chain when the whole potential has that ungated form. In this point
calculation, exact single-entry +/-1 coefficient maps carry no formation error;
general maps retain their rounding bounds.
"""

import numpy as np

from ._basis_kernels import basis_enclosure, potential_enclosure


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
    combined gate per function, in the same cancellation-free partition
    form the evaluator uses.  Coefficients that cancel across columns (for
    example several gated exponentials whose leading amplitudes nearly sum to
    zero) therefore cancel before any interval product, instead of adding
    their widths.  Each slot of ``full`` carries the dot-product rounding
    bound ``(d + 1) u sum_c |M_sc theta_c|``; exactly zero slots stay exactly
    zero, so a function with no coefficient (e.g. a disabled endpoint log) is
    skipped and never produces ``0 * inf``.  Each gate limit
    ``a + sb b + sc c`` (lower, central and upper) is enclosed as one sum per
    function, so a structurally exact zero (e.g. the upper limit of
    ``tL + tU``) stays exact.
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
