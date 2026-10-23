"""Numerical evaluation of a compiled G3 basis and its ``x`` derivatives.

G1 owner replaced: ``_model/vec.py`` and the potential-partial evaluation of
``_model/spec.py``/``_model/_state_kernels.pyx``.  Retained from G1: endpoint
distances in user units at finite boundaries (no reconstruction from two
fitting coordinates).

Every retained column is ``sum_j f_j(x) G_j(x)`` with an exact gate
``G = alpha_0 + alpha_1 t_L + alpha_2 t_U``.  Derivatives (up to order four)
are taken with respect to the user coordinate ``x`` from the *same* columns,
so ``q`` and all its derivatives share one coefficient vector by
construction.

Gates are evaluated without cancellation toward suppressed ends.  Writing
``p = 1 + t`` and ``m = 1 - t`` (each ``2 expit(+-2u)``, accurate down to
underflow), a gate has the exact rearrangements

* lower form: ``(alpha_0 - alpha_1 + alpha_2) + alpha_1 p_L - alpha_2 m_U``,
* upper form: ``(alpha_0 + alpha_1 - alpha_2) - alpha_1 m_L + alpha_2 p_U``,

whose constants are the exact gate limits at ``-inf`` / ``+inf``.  The lower
form is used below the gate midpoint and the upper form above it, so an
exactly suppressed limit contributes an exact zero constant and the residual
is a product of accurately computed small factors.  Gate derivatives are
polynomials in ``t`` times ``1 - t**2 = p m``, also free of cancellation.

Internal functions follow the fitting coordinate ``z = (x - c) / h``:
monomials are ``z**n`` and soft powers ``(z**2 + delta / h**2)**(p/2)``;
logarithms and exponentials use physical distances directly.  Finite
endpoints with an enabled logarithm are singular: the log column is ``+inf``
there.  Points must lie in the closed support; the evaluator performs no
input validation beyond that numerical invariant.

The arithmetic runs in the compiled ``_basis_kernels`` (``eval_point``,
``potential_point``) on the model's column table ``model.kernel``; the
tanh derivatives are ``t' = c w``, ``t'' = -2 c^2 t w``,
``t''' = -2 c^3 w (1 - 3 t^2)``, ``t'''' = 8 c^4 t w (2 - 3 t^2)`` with
``w = p m`` and ``c = du/dx``.
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
