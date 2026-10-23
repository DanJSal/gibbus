"""Local Taylor evaluators for q'', q''' and q'''' in physical coordinates.

Internal API, independent of minima search. A plan binds fixed basis geometry;
a cache also binds centers/order. u=(x-center)/scale; derivative rows already
have physical x units. All numeric construction/contraction/Horner loops are
compiled and release the GIL. Python validates arguments and allocates buffers.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np

from . import _series_kernels as _kernels


def _frozen(a, dtype=None):
    a = np.ascontiguousarray(a, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _theta(theta, d):
    a = np.array(theta, dtype=np.float64, order="C", copy=True)
    if a.shape != (d,) or not np.isfinite(a).all():
        raise ValueError(f"theta must be a finite vector of length {d}")
    return a


def _geometry_signature(model):
    """Value key for all inputs to local-series geometry, never model identity.

    Rational numerator/denominator pairs give a canonical representation even
    when independently constructed models do not share Fraction objects.
    Labels, data, and theta do not affect the series. Float repr preserves every
    binary64 value, so nearby geometries never accidentally share a tensor.
    """
    g = model.geometry
    functions = tuple(
        (f.index, f.kind, f.side, f.degree, f.power, f.delta, f.rate, f.location)
        for f in model.functions
    )
    exact = tuple(
        tuple((x.numerator, x.denominator) for x in row)
        for row in model.coefficient_map_exact
    )
    return repr(
        (
            1,
            model.request.lower,
            model.request.upper,
            g.center,
            g.scale,
            g.lower_location,
            g.upper_location,
            g.lower_steepness,
            g.upper_steepness,
            functions,
            exact,
        )
    ).encode("ascii")


@dataclass(frozen=True, init=False)
class SeriesPlan:
    """Reusable exact sign maps and fixed geometry for a compiled basis.

    Rational map algebra runs once at initialization. Calls use independent
    workspaces; immutable plans and caches may be shared by threads.
    """

    _model: object
    _kernel: object
    _maps: np.ndarray

    def __init__(self, model):
        nf, d = len(model.functions), model.n_coefficients
        if not d:
            raise ValueError("a series plan requires at least one basis coefficient")
        exact = model.coefficient_map_exact
        maps = np.zeros((4, 3 * nf, d))
        for j in range(nf):
            a, b, c = exact[3 * j : 3 * j + 3]
            for k, (x, y, z) in enumerate(zip(a, b, c)):
                # Keep cancellation exact before rounding, but do not repeat
                # rational arithmetic for structural zeros or ungated terms.
                if not y and not z:
                    if x:
                        maps[:, 3 * j, k] = float(x)
                    continue
                minus, plus = x - y, x + y
                maps[:, 3 * j, k] = (
                    float(minus - z),
                    float(minus + z),
                    float(plus - z),
                    float(plus + z),
                )
                if y:
                    twice = float(2 * y)
                    maps[:, 3 * j + 1, k] = (twice, twice, -twice, -twice)
                if z:
                    twice = float(2 * z)
                    maps[:, 3 * j + 2, k] = (twice, -twice, twice, -twice)
        if not np.isfinite(maps).all():
            raise OverflowError("basis map is not representable as float64")
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_kernel", model.kernel)
        object.__setattr__(self, "_maps", _frozen(maps))

    def _centers(self, centers, order, scale):
        if isinstance(order, (bool, np.bool_)):
            raise TypeError("order must be an integer")
        order = operator.index(order)
        if order < 0 or order > 4096:
            raise ValueError("order must be between 0 and 4096")
        x = np.array(centers, dtype=np.float64, ndmin=1, copy=True)
        if x.ndim != 1 or not x.size or not np.isfinite(x).all():
            raise ValueError("centers must be a nonempty finite scalar or vector")
        model, g = self._model, self._model.geometry
        if np.any(x <= model.request.lower) or np.any(x >= model.request.upper):
            raise ValueError("centers must lie strictly inside the support")
        radius = np.full(x.shape, np.inf)
        used = {j for column in model.columns for j, _ in column.terms}
        for j in used:
            fn = model.functions[j]
            if fn.kind == "log":
                distance = (
                    x - model.request.lower
                    if fn.side == "lower"
                    else model.request.upper - x
                )
                radius = np.minimum(radius, distance)
            elif fn.kind == "soft_power":
                radius = np.minimum(radius, np.hypot(x - g.center, math.sqrt(fn.delta)))
        signs = np.ones((x.size, 2), dtype=np.int32)
        for i, (a, b) in enumerate(
            (
                (g.lower_steepness, g.lower_location),
                (g.upper_steepness, g.upper_location),
            )
        ):
            if a is None:
                continue
            signed_a = a if i == 0 else -a
            signs[:, i] = np.where((x >= b) if signed_a > 0 else (x <= b), 1, -1)
            radius = np.minimum(radius, np.hypot(x - b, (math.pi / 2) / abs(a)))
        if scale is None:
            ell = np.where(np.isfinite(radius), 0.4 * radius, g.scale)
        else:
            ell = np.broadcast_to(np.asarray(scale, dtype=np.float64), x.shape).copy()
        if not np.isfinite(ell).all() or np.any(ell <= 0):
            raise ValueError("scale must be finite and strictly positive")
        if np.any(ell >= radius):
            raise ValueError(
                "scale must be strictly smaller than every active singularity distance"
            )
        packed = np.column_stack((x, ell, np.full(x.size, order)))
        return _frozen(packed), _frozen(signs), _frozen(radius), order

    def coefficients(self, theta, centers, order, *, scale=None):
        """Build [center, derivative 2/3/4, power] coefficients for one theta.

        No primitive cache is retained. Scale sets series units, not an error
        guarantee. Order zero is supported. New centers always rebuild jets.
        """
        packed, signs, radius, order = self._centers(centers, order, scale)
        theta = _theta(theta, self._model.n_coefficients)
        m, e, fallbacks = _kernels._construct(
            self._kernel, packed, signs, self._maps, theta
        )
        return SeriesBatch._make(
            packed, radius, order, m, e, fallbacks, self._model.request
        )

    def cache(self, centers, order, *, scale=None):
        """Cache primitive jets for repeated theta. Storage O(centers*atoms*order)."""
        packed, signs, radius, order = self._centers(centers, order, scale)
        m, e, fallbacks = _kernels._construct(
            self._kernel, packed, signs, self._maps, np.empty(0), True
        )
        return SeriesCache(
            self, packed, signs, radius, order, _frozen(m), _frozen(e), fallbacks
        )


@dataclass(frozen=True)
class SeriesCache:
    _plan: SeriesPlan
    _packed: np.ndarray
    _signs: np.ndarray
    radius: np.ndarray
    order: int
    _mantissa: np.ndarray
    _exponent: np.ndarray
    range_fallbacks: int

    def coefficients(self, theta):
        """Combine cached jets without transcendental evaluations."""
        p = self._plan
        theta = _theta(theta, p._model.n_coefficients)
        m, e = _kernels._combine(
            p._maps,
            theta,
            self._signs,
            self._mantissa,
            self._exponent,
            len(p._model.functions),
            self.order,
        )
        return SeriesBatch._make(
            self._packed,
            self.radius,
            self.order,
            m,
            e,
            self.range_fallbacks,
            p._model.request,
        )


@dataclass(frozen=True)
class SeriesBatch:
    centers: np.ndarray
    scales: np.ndarray
    radius: np.ndarray
    order: int
    mantissa: np.ndarray
    exponent: np.ndarray
    range_fallbacks: int
    _scaled: np.ndarray
    _support: tuple

    @classmethod
    def _make(cls, packed, radius, order, m, e, fallbacks, request):
        shape = (packed.shape[0], 3, order + 1)
        return cls(
            _frozen(packed[:, 0]),
            _frozen(packed[:, 1]),
            radius,
            order,
            _frozen(m.reshape(shape)),
            _frozen(e.reshape(shape)),
            fallbacks,
            _frozen(np.any(e != 0, axis=1), np.uint8),
            (request.lower, request.upper),
        )

    @property
    def coefficients(self):
        """Float64 coefficients or OverflowError; tiny coefficients may underflow.

        mantissa/exponent retain coefficients outside float64 range. Evaluation
        supports these coefficients without first exporting them to float64.
        """
        with np.errstate(over="ignore", under="ignore"):
            out = np.ldexp(self.mantissa, self.exponent)
        if not np.isfinite(out).all():
            raise OverflowError(
                "coefficients are not representable as float64; use mantissa/exponent"
            )
        return out

    def _evaluate(self, values, center_index, mode):
        values = np.asarray(values, dtype=np.float64)
        raw = np.asarray(center_index)
        if raw.dtype.kind not in "iu":
            raise TypeError("center_index must contain integers")
        values, idx = np.broadcast_arrays(values, raw)
        if np.any(idx < 0) or np.any(idx >= self.centers.size):
            raise IndexError("center_index outside the center batch")
        idx = np.ascontiguousarray(idx.ravel(), dtype=np.intp)
        v = np.ascontiguousarray(values.ravel())
        if not np.isfinite(v).all():
            raise ValueError("query values must be finite")
        if mode == "x":
            dx = v - self.centers[idx]
            u = dx / self.scales[idx]
        elif mode == "dx":
            dx = v
            u = dx / self.scales[idx]
        else:
            u = v
            dx = u * self.scales[idx]
        if np.any(np.abs(dx) >= self.radius[idx]) or not np.isfinite(u).all():
            raise ValueError("query must lie strictly inside its Taylor radius")
        lo, hi = self._support
        if np.any(dx <= lo - self.centers[idx]) or np.any(dx >= hi - self.centers[idx]):
            raise ValueError("query must lie strictly inside the support")
        result = _kernels._evaluate(
            self.mantissa, self.exponent, self._scaled, np.ascontiguousarray(u), idx
        )
        return result.reshape(values.shape + (3,))

    def evaluate(self, x, *, center_index=0):
        """Return (...,3) q'', q''' and q'''' values at physical x."""
        return self._evaluate(x, center_index, "x")

    def evaluate_offsets(self, dx, *, center_index=0):
        """Use physical offsets to avoid subtracting two large coordinates."""
        return self._evaluate(dx, center_index, "dx")

    def evaluate_scaled(self, u, *, center_index=0):
        """Evaluate at dimensionless offsets u=(x-center)/scale."""
        return self._evaluate(u, center_index, "u")
