"""Cached evaluation closures for PDF and potential functions.

This module constructs and returns *callable closures* over a fitted
polynomial potential ``q(z)``.  Each factory function captures the
relevant arrays at construction time so that repeated evaluations are
cheap.

Pipeline position
-----------------
``gibbus._postfit.evaluators`` sits just above the raw vectorized kernels in
``_model/vec.py``.  It is called from ``_api/component.py`` (via ``_assign_from_struct``)
to rebuild the evaluation stack whenever a new fit state is loaded.

Conventions
-----------
* All closures accept both scalars and arrays.  Scalar inputs produce
  ``float`` outputs; array inputs produce ``numpy.ndarray`` outputs.
* Internal coordinate *z* is related to user coordinate *x* by
  ``z = sigma_eff * x + mu_eff``; the affine mapping is handled by the
  caller (``_BaseSpaceView``), not here.
* Boundary log-singularity terms use only the canonical amplitude pair
  ``[aL, aU]``. Distances are fixed at the actual support endpoints.
"""

from functools import cache
from math import factorial

import numpy as np
from numpy.polynomial.polynomial import polyder

from .._model._state_kernels import _pdf_vec
from .._model.vec import _polyval, _q_eval


def _potential_base_func(support, q_poly, boundary_amplitudes):
    """Build a closure that evaluates the potential ``q(z)`` or its derivatives.

    The potential is defined as::

        q(z) = poly(z) - aL*log(z-L) - aU*log(U-z)

    where each log term is present only when its amplitude is positive.

    Parameters
    ----------
    support : array_like, shape (2,)
        ``[L, U]`` — lower and upper bounds of the internal coordinate.
        Infinite values indicate an unbounded side.
    q_poly : array_like, shape (d+1,)
        Polynomial coefficients in NumPy/``polyder`` convention
        (constant term first).
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset amplitudes ``[aL, aU]``.

    Returns
    -------
    callable
        ``pot(z, n)`` — evaluates the *n*-th derivative of ``q`` at *z*.

        Parameters
        ----------
        z : float or array_like
            Evaluation point(s) in internal coordinates.
        n : int
            Derivative order (``>= 0``).

        Returns
        -------
        float or numpy.ndarray
            ``q^(n)(z)``.  For ``n == 0``, points outside the support or
            with non-finite values return ``inf``.  For ``n >= 1``,
            such points return ``nan``.

        Raises
        ------
        ValueError
            If *n* is negative.
    """
    support = np.asarray(support, dtype=np.float64)
    q_poly = np.asarray(q_poly, dtype=np.float64)
    boundary_amplitudes = np.asarray(boundary_amplitudes, dtype=np.float64)

    L = float(support[0])
    U = float(support[1])

    def pot(z, n):
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")

        zz = np.asarray(z, dtype=np.float64)
        scalar = (zz.ndim == 0)

        mask = np.isfinite(zz)
        if np.isfinite(L):
            mask &= (zz >= L)
        if np.isfinite(U):
            mask &= (zz <= U)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            if n == 0:
                out = _q_eval(zz, support, q_poly, boundary_amplitudes, 0)
                out = np.where(mask & np.isfinite(out), out, np.inf)
                return float(out) if scalar else out

            dp = polyder(q_poly, n)
            out = _q_eval(zz, support, dp, boundary_amplitudes, n)
            out = np.where(mask & np.isfinite(out), out, np.nan)
            return float(out) if scalar else out

    return pot




@cache
def _stirling1(n):
    """Compute signed Stirling numbers of the first kind ``s(n, k)`` for
    ``k = 0 ... n``.

    Parameters
    ----------
    n : int
        Row index (non-negative).

    Returns
    -------
    tuple of float
        ``(s(n,0), s(n,1), ..., s(n,n))``.
    """
    n = int(n)
    s = [0.0] * (n + 1)
    s[0] = 1.0
    for m in range(1, n + 1):
        new = [0.0] * (n + 1)
        for k in range(1, m + 1):
            new[k] = s[k - 1] - (m - 1) * s[k]
        s = new
    return tuple(s)


def _potential_exp_from_x_potential(pot_x):
    """Build a closure for the exp-space potential from a base-space potential.

    Given a base-space potential callable ``pot_x(x, n)`` this factory
    returns ``pot_y(y, n)`` where *y = exp(x)* and the change-of-variables
    formula is applied via unsigned Stirling numbers of the first kind.

    The relationship is::

        pot_y(y, 0) = pot_x(log y, 0) + log y
        pot_y(y, n) = (1/y^n) * sum_{k=1}^{n} s(n,k) * q^(k)(log y)
                      + (-1)^(n-1) * (n-1)! / y^n       for n >= 1

    where ``s(n, k)`` are (signed) Stirling numbers of the first kind,
    cached at module level via :func:`_stirling1`.

    Parameters
    ----------
    pot_x : callable
        Base-space potential, signature ``pot_x(x, n) -> float | ndarray``.

    Returns
    -------
    callable
        ``pot_y(y, n)`` — exp-space potential.

        Parameters
        ----------
        y : float or array_like
            Evaluation point(s).  Non-positive values return ``inf``
            (for ``n == 0``) or ``nan`` (for ``n >= 1``).
        n : int
            Derivative order (``>= 0``).

        Returns
        -------
        float or numpy.ndarray

        Raises
        ------
        ValueError
            If *n* is negative.
    """

    def pot_y(y, n):
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")

        yy = np.asarray(y, dtype=np.float64)
        scalar = (yy.ndim == 0)

        out = np.empty_like(yy, dtype=np.float64)
        out.fill(np.inf if n == 0 else np.nan)

        m = (yy > 0.0) & np.isfinite(yy)
        if not np.any(m):
            return float(out) if scalar else out

        x = np.log(yy[m])

        if n == 0:
            out[m] = pot_x(x, 0) + x
            return float(out) if scalar else out

        s = _stirling1(n)

        num = np.zeros_like(x, dtype=np.float64)
        for k in range(1, n + 1):
            num += float(s[k]) * pot_x(x, k)

        num += float(((-1) ** (n - 1)) * factorial(n - 1))
        out[m] = num / (yy[m] ** n)

        return float(out) if scalar else out

    return pot_y


def _pdf_func(support, q_poly, boundary_amplitudes, /):
    """Build a closure that evaluates ``exp(-q(z))`` (unnormalized PDF) in internal coordinates.

    The closure is a thin wrapper over the compiled kernel
    :func:`._state_kernels._pdf_vec`, which is the single PDF
    implementation in the package; it converts the inputs once so that
    every call is a direct kernel invocation.

    Parameters
    ----------
    support : array_like, shape (2,)
        ``[L, U]`` in internal coordinates.
    q_poly : array_like, shape (d+1,)
        Polynomial coefficients (constant term first).
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset amplitudes.

    Returns
    -------
    callable
        ``pdf(x)`` — returns ``exp(-q(x))`` for *x* inside the support,
        zero outside it, and NaN where *x* is NaN.  Accepts scalars or
        arrays; a scalar in gives a ``float`` out.
    """
    supp = (float(support[0]), float(support[1]))
    qp = np.ascontiguousarray(q_poly, dtype=np.float64)
    qb = np.ascontiguousarray(boundary_amplitudes, dtype=np.float64)

    def pdf(x, /):
        x_arr = np.asarray(x, dtype=np.float64)
        scalar = (x_arr.ndim == 0)
        out = _pdf_vec(np.ascontiguousarray(np.atleast_1d(x_arr)).reshape(-1),
                       supp, qp, qb, 0.0)
        if scalar:
            return float(out[0])
        return out.reshape(x_arr.shape)

    return pdf


def _potential_oriented_affine_eval(
    x, support, mu_eff, sigma_eff, q_poly, boundary_amplitudes, n, /
):
    """Evaluate a fitted potential in an oriented public affine coordinate.

    Parameters
    ----------
    x : float or array_like
        Public-coordinate evaluation points.
    support : array_like, shape (2,)
        Public lower/upper support.
    mu_eff, sigma_eff : float
        Effective affine parameters satisfying ``z = sigma_eff*x + mu_eff``.
        ``sigma_eff`` may be negative for a reflected upper half-line.
    q_poly : array_like
        Normalized canonical polynomial-potential coefficients.
    boundary_amplitudes : array_like, shape (2,)
        Physical lower/upper zero-offset log amplitudes.
    n : int
        Derivative order.

    Returns
    -------
    float or numpy.ndarray
        Potential or derivative in public coordinates.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    xx = np.asarray(x, dtype=np.float64)
    scalar = xx.ndim == 0
    support = np.asarray(support, dtype=np.float64)
    amps = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")
    L, U = map(float, support)
    aL, aU = map(float, amps)
    sig = float(sigma_eff)
    jac = abs(sig)
    if not np.isfinite(jac) or jac <= 0.0:
        raise ValueError("effective affine scale must be finite and nonzero")
    mu = float(mu_eff)

    mask = np.isfinite(xx)
    if np.isfinite(L):
        mask &= xx >= L
    if np.isfinite(U):
        mask &= xx <= U

    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        z = sig * xx + mu
        if n == 0:
            out = _polyval(z, q_poly) - np.log(jac)
            if np.isfinite(L) and aL > 0.0:
                out -= aL * (np.log(jac) + np.log(xx - L))
            if np.isfinite(U) and aU > 0.0:
                out -= aU * (np.log(jac) + np.log(U - xx))
            out = np.where(mask & np.isfinite(out), out, np.inf)
            return float(out) if scalar else out

        out = (sig ** n) * _polyval(z, polyder(q_poly, n))
        fac = float(factorial(n - 1))
        if np.isfinite(L) and aL > 0.0:
            sign = -1.0 if (n & 1) else 1.0
            out += sign * fac * aL / ((xx - L) ** n)
        if np.isfinite(U) and aU > 0.0:
            out += fac * aU / ((U - xx) ** n)
        out = np.where(mask & np.isfinite(out), out, np.nan)
        return float(out) if scalar else out
