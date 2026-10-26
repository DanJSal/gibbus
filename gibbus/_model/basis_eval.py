"""Numerical evaluation of a compiled basis and its ``x`` derivatives.

Every retained column is ``sum_j f_j(x) G_j(x)`` with an exact gate
``G = alpha_0 + alpha_1 t_L + alpha_2 t_U``.  Derivatives (up to order four)
are taken with respect to the user coordinate ``x`` from the *same* columns,
so ``q`` and all its derivatives share one coefficient vector by
construction.

Gates are evaluated without cancellation toward suppressed ends through the
exact partition ``W_L + W_C + W_U = 1`` of the Gaussian gates:

``G = g_L W_L + kappa W_C + g_U W_U``

with the gate's limits ``g_L = alpha_0 - alpha_1 + alpha_2``,
``kappa = alpha_0 + alpha_1 + alpha_2`` and ``g_U = alpha_0 + alpha_1 -
alpha_2`` as exact constants and the nonnegative weights
``W_L = Phi(-s (z + 1))``, ``W_U = Phi(s (z - 1))`` (``erfc``, accurate down to
underflow) and ``W_C`` (a gate minus the far weight on each side).  An exactly
suppressed limit contributes an exact zero, and every derivative of a weight
is a Hermite polynomial times the Gaussian density.

Internal functions follow the canonical coordinate ``z = (x - c) / h``:
monomials are ``z**n``, soft powers ``A(z)**p`` (truncated Taylor arithmetic,
finite to fourth order at the origin) and exponentials are centered at their
own gate; logarithms use physical distances directly.  Finite endpoints with an
enabled logarithm are singular: the log column is ``+inf`` there.  Points must
lie in the closed support; the evaluator performs no input validation beyond
that numerical invariant.  Values beyond binary64 range overflow to the
correctly signed infinity.

The arithmetic runs in the compiled ``_basis_kernels`` (``eval_point``,
``potential_point``) on the model's column table ``model.kernel``.  The scaled
curvature for shape checks far outside binary64 range is the structured
kernel ``_basis_kernels.curvature``.
"""

from dataclasses import dataclass

import numpy as np

from ._basis_kernels import evaluate, potential


@dataclass(frozen=True)
class _BasisValues:
    """Basis columns and their ``x`` derivatives.

    Parameters
    ----------
    phi, dphi, d2phi : numpy.ndarray
        Arrays of shape ``(R, d)``.
    d3phi, d4phi : numpy.ndarray or None
        Third and fourth derivatives when requested, else ``None``.
    """

    phi: np.ndarray
    dphi: np.ndarray | None = None
    d2phi: np.ndarray | None = None
    d3phi: np.ndarray | None = None
    d4phi: np.ndarray | None = None

    def derivative(self, order, /):
        """Return the array of the requested derivative order.

        Parameters
        ----------
        order : int
            Derivative order, 0 to 4.
        """
        return (self.phi, self.dphi, self.d2phi, self.d3phi, self.d4phi)[order]


def _points(model, x, /):
    """Canonical evaluation points (closed support checked).

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    x : array_like
        Evaluation points (user units).
    """
    request = model.request
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
    if np.any(x < request.lower) or np.any(x > request.upper):
        raise ValueError("evaluation points must lie in the closed support")
    return x


def _evaluate_basis(model, x, /, max_order=2):
    """Evaluate all retained columns and their derivatives.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    x : numpy.ndarray
        Canonical finite ``float64`` points in the closed support, shape
        ``(R,)``.
    max_order : int
        Highest derivative order, ``0`` to ``4`` (default ``2``).

    Returns
    -------
    _BasisValues
        ``Phi`` and derivatives, each of shape ``(R, d)``; orders above
        ``max_order`` are ``None``.

    Raises
    ------
    ValueError
        If a point lies outside the closed support or ``max_order`` is not
        in ``0..4``.
    """
    if max_order not in (0, 1, 2, 3, 4):
        raise ValueError("max_order must be in 0..4")
    out = evaluate(model.kernel, _points(model, x), max_order)
    return _BasisValues(*out, *([None] * (4 - max_order)))


def _potential(model, theta, x, /, max_order=2):
    """Evaluate ``q`` and its derivatives for independent coefficients.

    Columns with an exactly zero coefficient are skipped, so a disabled
    endpoint logarithm never produces ``0 * inf`` at its endpoint.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Independent coefficients, shape ``(d,)``.
    x : numpy.ndarray
        Canonical finite points in the closed support, shape ``(R,)``.
    max_order : int
        Highest derivative order, ``0`` to ``4`` (default ``2``).

    Returns
    -------
    tuple of numpy.ndarray
        ``(q, q', ..., q^(max_order))``, each of shape ``(R,)``; ``q`` is
        defined up to the normalization gauge (no constant term).
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    return tuple(potential(model.kernel, theta, _points(model, x), max_order))
