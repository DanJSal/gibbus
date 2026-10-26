"""The exp-space potential of a base-space potential.

``_potential_exp_from_x_potential`` turns a base-space potential callable
``pot_x(x, n)`` into ``pot_y(y, n)`` for ``y = exp(x)`` by the change of
variables (Stirling numbers of the first kind give the derivatives).  The
closure accepts scalars and arrays; scalar inputs produce ``float``.
"""

from functools import cache
from math import factorial

import numpy as np


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
        y : numpy.ndarray, dtype float64
            Boundary-prepared points. Non-positive values return ``inf``
            (for ``n == 0``) or ``nan`` (for ``n >= 1``).
        n : int
            Boundary-validated derivative order (``>= 0``).

        Returns
        -------
        float or numpy.ndarray

    """

    def pot_y(y, n):
        yy = y
        scalar = yy.ndim == 0

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
