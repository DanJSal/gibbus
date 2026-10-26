"""Shared numerical expectation engine for fitted densities."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .._defaults import (
    EXPECT_EPSABS,
    EXPECT_EPSREL,
    EXPECT_MAX_RELATIVE_ERROR,
    EXPECT_SCALAR_LIMIT,
    EXPECT_VECTORIZED_LIMIT,
)
from .._model.gauss_kronrod import WG10 as _WG10
from .._model.gauss_kronrod import WK21 as _WK21
from .._model.gauss_kronrod import XK21 as _XK21


def expect(potential, support, func, /, *, points):
    """Compute ``E[g(X)]`` under a normalized fitted potential.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator ``potential(x, 0)``.
    support : array_like, shape (2,)
        Integration support.
    func : callable
        Scalar function ``g``. Non-finite interior values are rejected.
    points : sequence of float or None
        Interior breakpoints (typically the mode or modes); the integral is
        split into the displacement pieces of :func:`_pieces`.
    """
    lo, hi = map(float, support)

    def integrand(x):
        # Evaluate the fitted density before the user function.  Infinite-range
        # quadrature deliberately probes very remote coordinates; for light
        # tails the density may already underflow to exactly zero there while
        # a transformed user function (for example ``exp(x)`` in exp space)
        # would overflow.  Such a point contributes exactly zero and the user
        # function must not be asked to represent a mathematically irrelevant
        # value.
        q = float(potential(float(x), 0))
        if not np.isfinite(q):
            return 0.0
        with np.errstate(under="ignore"):
            density = float(np.exp(-q))
        if density == 0.0:
            return 0.0

        gx = func(float(x))
        arr = np.asarray(gx)
        if arr.ndim != 0:
            raise RuntimeError("expect function must return a scalar for scalar input")
        gx = float(arr)
        if not np.isfinite(gx):
            raise RuntimeError(
                f"expect function returned a non-finite value at x={x!r}"
            )
        return float(gx * density)

    pts, width = _anchors(potential, lo, hi, points)
    value = error = 0.0
    for anchor, sign, _a, top in zip(*_pieces(pts, lo, hi, width), strict=True):

        def piece(u, anchor=anchor, sign=sign):
            x = min(max(anchor + sign * width * (u / (1.0 - u)), lo), hi)
            return integrand(x) * width / (1.0 - u) ** 2

        part, part_error = quad(
            piece,
            0.0,
            top,
            epsabs=EXPECT_EPSABS,
            epsrel=EXPECT_EPSREL,
            limit=EXPECT_SCALAR_LIMIT,
        )
        value += part
        error += part_error
    scale = max(abs(value), 1.0)
    if (
        not np.isfinite(value)
        or not np.isfinite(error)
        or error > EXPECT_MAX_RELATIVE_ERROR * scale
    ):
        raise RuntimeError(
            f"expect quadrature did not converge: value={value!r}, error={error!r}"
        )
    return float(value)


# The Gauss--Kronrod 21/10 rule on [-1, 1], nodes ascending.
_NODES = np.concatenate([-_XK21[:-1], [0.0], _XK21[-2::-1]])
_KRONROD = np.concatenate([_WK21[:-1], [_WK21[-1]], _WK21[-2::-1]])
_GAUSS = np.zeros(21)
_GAUSS[[1, 3, 5, 7, 9]] = _WG10
_GAUSS[[19, 17, 15, 13, 11]] = _WG10


def _anchors(potential, lo, hi, points):
    """Breakpoints anchoring the displacement pieces, and the local width.

    Parameters
    ----------
    potential : callable
        Vectorized normalized negative-log density ``potential(x, 0)``.
    lo, hi : float
        Integration support.
    points : sequence of float or None
        Caller breakpoints; those strictly inside the support are kept.
    """
    pts = np.asarray(points if points is not None else (), dtype=np.float64).reshape(-1)
    pts = np.unique(pts[np.isfinite(pts) & (pts > lo) & (pts < hi)])
    if pts.size == 0:
        if np.isfinite(lo) and np.isfinite(hi):
            default = 0.5 * (lo + hi)
        elif np.isfinite(lo):
            default = lo + 1.0
        elif np.isfinite(hi):
            default = hi - 1.0
        else:
            default = 0.0
        pts = np.array([default])
    width = 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        # Pointwise, so scalar-only potentials are accepted too.
        q = np.array([float(potential(float(p), 0)) for p in pts])
        peak = np.exp(-q)
    peak = peak[np.isfinite(peak) & (peak > 0.0)]
    if peak.size:
        # The narrowest density feature sets the map scale everywhere: a
        # finer scale only spends more nodes near the anchors, while a coarse
        # one can step over a narrow peak entirely.
        width = 0.5 / float(np.max(peak))
    return pts, width


def _pieces(pts, lo, hi, width):
    """Displacement pieces ``x = anchor + sign * width * u / (1 - u)``.

    Every anchor owns the stretch to each side up to the support end or the
    midpoint towards its neighbor; a stretch of length ``d`` maps onto
    ``u in [0, d / (d + width)]`` (``[0, 1]`` when unbounded).  Finite and
    infinite stretches therefore share one coordinate, measured in density
    widths from the anchor, so mass next to an anchor is resolved however
    far away the stretch's other end lies.

    Parameters
    ----------
    pts : ndarray
        Sorted interior anchors.
    lo, hi : float
        Integration support.
    width : float
        Positive displacement scale.
    """
    cuts = np.concatenate(([lo], 0.5 * (pts[:-1] + pts[1:]), [hi]))
    anchor, sign, top = [], [], []
    for i, p in enumerate(pts):
        for direction, end in ((-1.0, cuts[i]), (1.0, cuts[i + 1])):
            distance = abs(end - p)
            if not distance > 0.0:
                continue
            anchor.append(p)
            sign.append(direction)
            top.append(distance / (distance + width) if np.isfinite(distance) else 1.0)
    n = len(anchor)
    return (
        np.array(anchor, dtype=np.float64),
        np.array(sign, dtype=np.float64),
        np.zeros(n),
        np.array(top, dtype=np.float64),
    )


def expect_vectorized(potential, support, func, /, *, points, noise=0.0):
    """Compute ``E[g(X)]`` with a vectorized adaptive Gauss--Kronrod rule.

    Same contract as :func:`expect` for integrands that accept arrays:
    ``potential(x, 0)`` and ``func(x)`` are called on whole node arrays, so a
    compiled potential is evaluated once per refinement round instead of
    once per node through Python.  Each piece is a displacement
    ``x = a +/- s u / (1 - u)`` from a breakpoint ``a`` (see :func:`_pieces`),
    with ``s`` the density's width at its highest breakpoint; refinement
    bisects the panels carrying the largest errors until the global error
    meets the package expectation tolerances.

    Parameters
    ----------
    potential : callable
        Vectorized normalized negative-log density ``potential(x, 0)``.
    support : array_like, shape (2,)
        Integration support.
    func : callable
        Vectorized ``g``; non-finite values where the density is positive
        are rejected.
    points : sequence of float or None
        Interior breakpoints (typically the mode or modes).
    noise : float, default 0.0
        Relative rounding noise of the integrand values (an absolute error
        of the potential).  The Kronrod--Gauss difference cannot fall below
        it, so the error target is never set beneath ``4 * noise`` times the
        integral of ``|g|`` times the density.
    """
    lo, hi = map(float, support)
    pts, width = _anchors(potential, lo, hi, points)
    anchor, sign, pa, pb = _pieces(pts, lo, hi, width)

    def evaluate(anchor, sign, pa, pb):
        mid = 0.5 * (pa + pb)
        half = 0.5 * (pb - pa)
        u = mid[:, None] + half[:, None] * _NODES[None, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            x = anchor[:, None] + sign[:, None] * width * (u / (1.0 - u))
            jac = width / (1.0 - u) ** 2
        x = np.clip(x, lo, hi)
        flat = x.reshape(-1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            q = np.asarray(potential(flat, 0), dtype=np.float64).reshape(x.shape)
            density = np.where(np.isfinite(q), np.exp(-q), 0.0)
        positive = density > 0.0
        g = np.zeros_like(density)
        if np.any(positive):
            gx = np.asarray(func(flat[positive.reshape(-1)]), dtype=np.float64).reshape(
                -1
            )
            if not np.all(np.isfinite(gx)):
                bad = flat[positive.reshape(-1)][~np.isfinite(gx)][0]
                raise RuntimeError(
                    f"expect function returned a non-finite value at x={bad!r}"
                )
            g[positive] = gx
        values = g * density * jac
        values[~positive] = 0.0
        kr = (values @ _KRONROD) * half
        ga = (values @ _GAUSS) * half
        return kr, np.abs(kr - ga), (np.abs(values) @ _KRONROD) * half

    value, error, magnitude = evaluate(anchor, sign, pa, pb)
    while True:
        total = float(np.sum(value))
        total_error = float(np.sum(error))
        floor = 4.0 * noise * float(np.sum(magnitude))
        tolerance = max(EXPECT_EPSABS, EXPECT_EPSREL * abs(total), floor)
        if total_error <= tolerance or value.size >= EXPECT_VECTORIZED_LIMIT:
            break
        order = np.argsort(-error)
        needed = (
            np.searchsorted(np.cumsum(error[order]), total_error - 0.5 * tolerance) + 1
        )
        split = order[: min(int(needed), EXPECT_VECTORIZED_LIMIT - value.size)]
        if split.size == 0:
            break
        mid = 0.5 * (pa[split] + pb[split])
        keep = np.ones(value.size, dtype=bool)
        keep[split] = False
        new_anchor = np.concatenate([anchor[split], anchor[split]])
        new_sign = np.concatenate([sign[split], sign[split]])
        new_a = np.concatenate([pa[split], mid])
        new_b = np.concatenate([mid, pb[split]])
        new_value, new_error, new_magnitude = evaluate(
            new_anchor, new_sign, new_a, new_b
        )
        anchor = np.concatenate([anchor[keep], new_anchor])
        sign = np.concatenate([sign[keep], new_sign])
        pa = np.concatenate([pa[keep], new_a])
        pb = np.concatenate([pb[keep], new_b])
        value = np.concatenate([value[keep], new_value])
        error = np.concatenate([error[keep], new_error])
        magnitude = np.concatenate([magnitude[keep], new_magnitude])
    total = float(np.sum(value))
    total_error = float(np.sum(error))
    scale = max(abs(total), 1.0)
    floor = 4.0 * noise * float(np.sum(magnitude))
    if (
        not np.isfinite(total)
        or not np.isfinite(total_error)
        or (total_error > max(EXPECT_MAX_RELATIVE_ERROR * scale, floor))
    ):
        raise RuntimeError(
            f"expect quadrature did not converge: value={total!r}, error={total_error!r}"
        )
    return total
