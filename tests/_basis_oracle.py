"""Independent high-precision reference for compiled basis evaluation (test only).

The oracle evaluates a compiled ``_BasisModel`` from its definition: every
retained column is ``sum_j f_j (alpha_0 + alpha_1 t_L + alpha_2 t_U)`` over
the canonical functions (``z**n``, endpoint logs, ``A(z)**p`` with
``A = (1 + z**m)**(1/m) - 1``, the gate-centered exponentials) and the erf
gates ``t_L = erf(a (1 + z))`` and ``t_U = erf(a (1 - z))``.  Derivatives
are symbolic (sympy) and evaluated in mpmath at adaptive precision, so the
oracle shares no derivative formulas, Taylor arithmetic, double-double code
or rescaling with the compiled kernels.  Family constants and parameters
enter as the exact binary64 values the kernels use.

Each gate is written in the partition form ``g_L W_L + kappa W_C + g_U W_U``
(an exact identity of the definition, with ``W_C`` written per side as a
difference of a gate and the far partition weight), so far-tail values keep
their relative accuracy without astronomic working precision.  Next to each
value the oracle returns the absolute Leibniz magnitude
``sum |coef| C(k, i) |f^(i)| |w^(k-i)|``: the natural scale of a
backward-stable evaluation.
"""

import math

import mpmath as mp
import sympy as sp

from gibbus._defaults import BASIS_GATE_SHARPNESS, BASIS_SOFT_POWER_ROOT

_V = sp.Symbol("v", real=True)


def _rational(value):
    """Exact rational of a binary64 value."""
    return sp.Rational(float(value))


class _Phi(sp.Function):
    """Standard normal CDF whose derivative is the density."""

    def fdiff(self, argindex=1):
        w = self.args[0]
        return sp.exp(-(w**2) / 2) / sp.sqrt(2 * sp.pi)


def _mp_phi(w):
    """``Phi(w)`` in mpmath, with the Mills series far in the tails."""
    if w < -1e8:
        t = mp.mpf(1)
        total = mp.mpf(1)
        w2 = w * w
        for k in range(1, 12):
            t *= -(2 * k - 1) / w2
            total += t
        return mp.exp(-w2 / 2) / (-w * mp.sqrt(2 * mp.pi)) * total
    if w > 1e8:
        return 1 - _mp_phi(-w)
    return mp.ncdf(w)


def _lambdify(expr):
    return sp.lambdify(_V, expr, modules=[{"_Phi": _mp_phi}, "mpmath"])


class BasisOracle:
    """High-precision columns of one compiled basis.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    units : {"x", "z"}, optional
        Differentiate with respect to the user coordinate ``x`` (logs use the
        physical distance to the endpoint, as the evaluator does) or the
        canonical ``z``.
    max_order : int, optional
        Highest derivative order provided.
    """

    def __init__(self, model, units="x", max_order=5):
        self.model = model
        self.units = units
        self.max_order = max_order
        g = model.geometry
        self.c = _rational(g.center)
        self.h = _rational(g.scale)
        self.has_l = g.has_lower_gate
        self.has_u = g.has_upper_gate
        s = _rational(BASIS_GATE_SHARPNESS)
        m = BASIS_SOFT_POWER_ROOT
        z = (_V - self.c) / self.h if units == "x" else _V
        lo_z, up_z = g.coordinate.canonical_support
        req = model.request
        self._functions = {}
        for fn in model.functions:
            if fn.kind == "monomial":
                expr = z**fn.degree
            elif fn.kind == "log":
                if units == "x":
                    expr = (
                        -sp.log(_V - _rational(req.lower))
                        if fn.side == "lower"
                        else -sp.log(_rational(req.upper) - _V)
                    )
                else:
                    expr = (
                        -sp.log(_V - _rational(lo_z))
                        if fn.side == "lower"
                        else -sp.log(_rational(up_z) - _V)
                    )
            elif fn.kind == "exponential":
                if not fn.active:
                    continue
                lam = _rational(fn.rate)
                expr = (
                    sp.exp(lam * (z - 1))
                    if fn.side == "upper"
                    else sp.exp(-lam * (z + 1))
                )
            else:
                if not fn.active:
                    continue
                expr = ((1 + z**m) ** sp.Rational(1, m) - 1) ** _rational(fn.power)
            self._functions[fn.index] = [
                _lambdify(sp.diff(expr, _V, k)) for k in range(max_order + 1)
            ]
        w_l = _Phi(-s * (z + 1))
        w_u = _Phi(s * (z - 1))
        g_l = _Phi(s * (z + 1))
        g_u = _Phi(s * (1 - z))
        # Partition weights per side: lists of (sign, expression) summands.
        weights = {}
        for side in (1, -1):
            if self.has_l and self.has_u:
                central = [(1, g_u), (-1, w_l)] if side > 0 else [(1, g_l), (-1, w_u)]
            elif self.has_u:
                central = [(1, g_u)]
            elif self.has_l:
                central = [(1, g_l)]
            else:
                central = [(1, sp.Integer(1))]
            weights[side] = {
                "low": [(1, w_l)] if self.has_l else [],
                "mid": central,
                "up": [(1, w_u)] if self.has_u else [],
            }
        self._weights = {
            side: {
                key: [
                    (
                        sign,
                        [_lambdify(sp.diff(expr, _V, k)) for k in range(max_order + 1)],
                    )
                    for sign, expr in parts
                ]
                for key, parts in table.items()
            }
            for side, table in weights.items()
        }

    def _z(self, point):
        return (point - mp.mpf(self.c)) / mp.mpf(self.h) if self.units == "x" else point

    def column(self, col, point, order, dps=None):
        """``(value, magnitude)`` of derivative ``order`` of one column.

        Parameters
        ----------
        col : int
            Column index.
        point : float or mpmath.mpf
            Evaluation point (``x`` or ``z`` per ``units``).
        order : int
            Derivative order.
        dps : int or None, optional
            Working digits; ``None`` adapts them to the point.
        """
        point_f = float(point)
        with mp.workdps(dps or self.digits(point_f)):
            # An mpmath point is used as given (quadrature nodes are not
            # rounded to binary64).
            v = mp.mpf(point)
            side = 1 if self._z(v) >= 0 else -1
            value = mp.mpf(0)
            mag = mp.mpf(0)
            for j, gate in self.model.columns[col].terms:
                if self.model.functions[j].kind == "soft_power" and self._z(v) == 0:
                    # Every derivative of order <= 4 < m p vanishes at the origin.
                    f = [mp.mpf(0)] * (order + 1)
                else:
                    f = [fk(v) for fk in self._functions[j][: order + 1]]
                a0, a1, a2 = (mp.mpf(g.numerator) / g.denominator for g in gate)
                if a1 == 0 and a2 == 0:
                    value += a0 * f[order]
                    mag += abs(a0 * f[order])
                    continue
                coefs = {"low": a0 - a1 + a2, "mid": a0 + a1 + a2, "up": a0 + a1 - a2}
                for key, parts in self._weights[side].items():
                    coef = coefs[key]
                    if coef == 0:
                        continue
                    for sign, wd in parts:
                        for i in range(order + 1):
                            piece = math.comb(order, i) * f[i] * wd[order - i](v)
                            value += sign * coef * piece
                            mag += abs(coef * piece)
            return value, mag

    def potential(self, theta, point, order, dps=None):
        """``(value, magnitude)`` of derivative ``order`` of ``q = theta . Phi``.

        Parameters
        ----------
        theta : sequence of float or Fraction
            Coefficients.
        point : float
            Evaluation point.
        order : int
            Derivative order.
        dps : int or None, optional
            Working digits.
        """
        point_f = float(point)
        with mp.workdps(dps or self.digits(point_f)):
            value = mp.mpf(0)
            mag = mp.mpf(0)
            for col, t in enumerate(theta):
                if t == 0:
                    continue
                tv = (
                    mp.mpf(t.numerator) / t.denominator
                    if hasattr(t, "numerator")
                    else mp.mpf(t)
                )
                v, m = self.column(col, point, order, dps=mp.mp.dps)
                value += tv * v
                mag += abs(tv) * m
            return value, mag

    def digits(self, point):
        """Working digits for a point: extra digits where ``A = R - 1`` cancels.

        Parameters
        ----------
        point : float
            Evaluation point.
        """
        z = abs(float(self._z(mp.mpf(float(point)))))
        extra = 4.0 * max(0.0, -math.log10(z)) if z > 0 else 0.0
        return int(50 + extra)
