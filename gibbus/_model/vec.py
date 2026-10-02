"""Vectorised low-level evaluators for polynomial potentials.

Array polynomial evaluation is delegated to the compiled Horner kernel
``_state_kernels._polyval_vec``; potential evaluation combines those
polynomials with the fixed zero-offset endpoint-log terms of the model.
"""

from math import factorial

import numpy as np
from numpy.polynomial.polynomial import polyval

from .._defaults import MAX_FAC
from ._state_kernels import _polyval_vec

_FACTORIAL = np.empty(int(MAX_FAC) + 1, dtype=np.float64)
_FACTORIAL[0] = 1.0
for _i in range(1, int(MAX_FAC) + 1):
    _FACTORIAL[_i] = _FACTORIAL[_i - 1] * _i
"""Module-level factorial table for orders 0 … MAX_FAC (inclusive).

Avoids repeated ``math.factorial`` calls in tight loops.  Orders above
``MAX_FAC`` fall back to ``math.factorial`` at runtime.
"""


def _polyval(xx, poly, /):
    """Evaluate a polynomial at *xx*, dispatching on input shape.

    Arrays go to the compiled Horner kernel
    :func:`._state_kernels._polyval_vec`, which is roughly eight times
    faster than ``numpy.polynomial.polynomial.polyval`` on the small
    coefficient vectors used here (it avoids building one temporary
    per coefficient).  Scalars are evaluated inline, since the array
    machinery costs more than the arithmetic saves.

    Parameters
    ----------
    xx : numpy.ndarray
        Evaluation point(s), any shape including 0-d.
    poly : array_like, shape (d+1,)
        Coefficients, constant term first.

    Returns
    -------
    numpy.ndarray
        Same shape as *xx*.  An empty *poly* gives zeros, matching the
        mathematical zero polynomial.
    """
    coef = np.asarray(poly, dtype=np.float64).ravel()
    if coef.shape[0] == 0:
        # An empty coefficient vector is the zero polynomial, which is
        # what a derivative past the polynomial's degree produces.
        # ``numpy.polynomial.polyval`` raises IndexError on this input.
        return (np.float64(0.0) if xx.ndim == 0
                else np.zeros(xx.shape, dtype=np.float64))
    if xx.ndim == 0:
        return polyval(xx, coef)
    return _polyval_vec(xx, coef).reshape(xx.shape)


def _q_eval(x, support, poly, boundary_amplitudes, order, /):
    """Evaluate the potential ``q`` (or a derivative) at *x*.

    The boundary terms use fixed zero-offset endpoint distances::

        q(x) = poly(x)
               - aL * log(x - L)    if aL > 0
               - aU * log(U - x)    if aU > 0

    Higher-order derivatives are computed analytically:

        q^(n)(x) = poly^(n)(x)
                   + (-1)^n * (n-1)! * aL / (x - L)^n   if aL > 0
                   + (n-1)! * aU / (U - x)^n             if aU > 0

    Parameters
    ----------
    x : array_like
        Evaluation point(s).
    support : array_like, shape (2,)
        ``[L, U]`` canonical support bounds.
    poly : array_like, shape (d+1,)
        Polynomial coefficients, constant term first. For derivative order
        *n*, pass the *n*-times-differentiated polynomial.
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset log amplitudes ``[aL, aU]``.
        ``NaN`` or zero denotes an inactive side.
    order : int
        Derivative order of *poly*.

    Returns
    -------
    float or numpy.ndarray
        Evaluated potential or derivative. Floating-point errors in endpoint
        terms are suppressed; callers mask out-of-support values as needed.
    """
    Lx, Ux = map(float, support)
    amps = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")
    aL, aU = map(float, amps)

    xx = np.asarray(x, dtype=np.float64)
    scalar = (xx.ndim == 0)

    hasL = np.isfinite(Lx) and np.isfinite(aL) and (aL > 0.0)
    hasU = np.isfinite(Ux) and np.isfinite(aU) and (aU > 0.0)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        out = _polyval(xx, poly)

        if int(order) == 0:
            if hasL:
                out -= aL * np.log(xx - Lx)
            if hasU:
                out -= aU * np.log(Ux - xx)
        else:
            n = int(order)
            if n - 1 <= int(MAX_FAC):
                fac = float(_FACTORIAL[n - 1])
            else:
                fac = float(factorial(n - 1))

            if hasL:
                sgn = -1.0 if (n & 1) else 1.0
                out += (sgn * fac * aL) / ((xx - Lx) ** n)
            if hasU:
                out += (fac * aU) / ((Ux - xx) ** n)

    return float(out) if scalar else out
