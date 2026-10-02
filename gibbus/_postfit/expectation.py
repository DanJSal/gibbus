"""Shared numerical expectation engine for fitted densities."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.integrate import quad

from .._defaults import EXPECT_MAX_RELATIVE_ERROR


def expect(potential, support, func, /, *, points=None):
    """Compute ``E[g(X)]`` under a normalized fitted potential.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator ``potential(x, 0)``.
    support : array_like, shape (2,)
        Integration support.
    func : callable
        Scalar function ``g``. Non-finite interior values are rejected.
    points : sequence of float or None, optional
        Interior quadrature breakpoints for finite integration ranges.
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
            raise RuntimeError(f"expect function returned a non-finite value at x={x!r}")
        return float(gx * density)

    kwargs = {"epsabs": 1e-10, "epsrel": 1e-10, "limit": 300}
    if np.isfinite(lo) and np.isfinite(hi) and points is not None:
        pts = np.asarray(points, dtype=np.float64).reshape(-1)
        pts = pts[np.isfinite(pts) & (pts > lo) & (pts < hi)]
        if pts.size:
            kwargs["points"] = np.unique(pts)
    value, error = quad(integrand, lo, hi, **kwargs)
    scale = max(abs(value), 1.0)
    if not np.isfinite(value) or not np.isfinite(error) or error > EXPECT_MAX_RELATIVE_ERROR * scale:
        raise RuntimeError(
            f"expect quadrature did not converge: value={value!r}, error={error!r}"
        )
    return float(value)


# Gauss--Kronrod 21/10 rule (QUADPACK qk21) on [-1, 1].
_XK21 = np.array([
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
    0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
    0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
    0.294392862701460198131126603103866, 0.148874338981631210884826001129720,
    0.0,
])
_WK21 = np.array([
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
    0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
    0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
    0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
])
_WG10 = np.array([
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
    0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
])
_NODES = np.concatenate([-_XK21[:-1], [0.0], _XK21[-2::-1]])
_KRONROD = np.concatenate([_WK21[:-1], [_WK21[-1]], _WK21[-2::-1]])
_GAUSS = np.zeros(21)
_GAUSS[[1, 3, 5, 7, 9]] = _WG10
_GAUSS[[19, 17, 15, 13, 11]] = _WG10


def expect_vectorized(potential, support, func, /, *, points=None, epsabs=1e-10,
                      epsrel=1e-10, limit=4000):
    """Compute ``E[g(X)]`` with a vectorized adaptive Gauss--Kronrod rule.

    Same contract as :func:`expect` for integrands that accept arrays:
    ``potential(x, 0)`` and ``func(x)`` are called on whole node arrays, so a
    compiled potential is evaluated once per refinement round instead of
    once per node through Python.  Semi-infinite pieces use
    ``x = a +/- s u / (1 - u)`` with ``s`` the density's width at its highest
    breakpoint; refinement bisects the panels carrying the largest errors
    until the global error meets ``max(epsabs, epsrel |E|)``.

    Parameters
    ----------
    potential : callable
        Vectorized normalized negative-log density ``potential(x, 0)``.
    support : array_like, shape (2,)
        Integration support.
    func : callable
        Vectorized ``g``; non-finite values where the density is positive
        are rejected.
    points : sequence of float or None, optional
        Interior breakpoints (typically the mode or modes).
    epsabs, epsrel : float, optional
        Global error targets.
    limit : int, optional
        Maximum number of panels.
    """
    lo, hi = map(float, support)
    pts = np.asarray(points if points is not None else (), dtype=np.float64).reshape(-1)
    pts = np.unique(pts[np.isfinite(pts) & (pts > lo) & (pts < hi)])
    if pts.size == 0 and not (np.isfinite(lo) and np.isfinite(hi)):
        pts = np.array([0.0 if not (np.isfinite(lo) or np.isfinite(hi))
                        else (lo + 1.0 if np.isfinite(lo) else hi - 1.0)])
    width = 1.0
    if pts.size:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            peak = np.exp(-np.asarray(potential(pts, 0), dtype=np.float64))
        peak = peak[np.isfinite(peak) & (peak > 0.0)]
        if peak.size:
            width = 0.5 / float(np.max(peak))
    edges = np.concatenate(([lo], pts, [hi]))
    # Segment kinds: 0 finite [a, b]; 1 [a, inf); 2 (-inf, b].
    kinds, anchors, a_par, b_par = [], [], [], []
    for a, b in itertools.pairwise(edges):
        if not b > a:
            continue
        if np.isfinite(a) and np.isfinite(b):
            kinds.append(0)
            anchors.append(0.0)
            a_par.append(a)
            b_par.append(b)
        elif np.isfinite(a):
            kinds.append(1)
            anchors.append(a)
            a_par.append(0.0)
            b_par.append(1.0)
        else:
            kinds.append(2)
            anchors.append(b)
            a_par.append(0.0)
            b_par.append(1.0)
    kind = np.array(kinds, dtype=np.int64)
    anchor = np.array(anchors, dtype=np.float64)
    pa = np.array(a_par, dtype=np.float64)
    pb = np.array(b_par, dtype=np.float64)

    def evaluate(kind, anchor, pa, pb):
        mid = 0.5 * (pa + pb)
        half = 0.5 * (pb - pa)
        u = mid[:, None] + half[:, None] * _NODES[None, :]
        x = np.where(kind[:, None] == 0, u, 0.0)
        jac = np.where(kind[:, None] == 0, 1.0, 0.0)
        semi = kind[:, None] != 0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio = u / (1.0 - u)
            sign = np.where(kind[:, None] == 1, 1.0, -1.0)
            x = np.where(semi, anchor[:, None] + sign * width * ratio, x)
            jac = np.where(semi, width / (1.0 - u) ** 2, jac)
        flat = x.reshape(-1)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            q = np.asarray(potential(flat, 0), dtype=np.float64).reshape(x.shape)
            density = np.where(np.isfinite(q), np.exp(-q), 0.0)
        positive = density > 0.0
        g = np.zeros_like(density)
        if np.any(positive):
            gx = np.asarray(func(flat[positive.reshape(-1)]), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(gx)):
                bad = flat[positive.reshape(-1)][~np.isfinite(gx)][0]
                raise RuntimeError(f"expect function returned a non-finite value at x={bad!r}")
            g[positive] = gx
        values = g * density * jac
        values[~positive] = 0.0
        kr = (values @ _KRONROD) * half
        ga = (values @ _GAUSS) * half
        return kr, np.abs(kr - ga)

    value, error = evaluate(kind, anchor, pa, pb)
    while True:
        total = float(np.sum(value))
        total_error = float(np.sum(error))
        tolerance = max(epsabs, epsrel * abs(total))
        if total_error <= tolerance or value.size >= limit:
            break
        order = np.argsort(-error)
        needed = np.searchsorted(np.cumsum(error[order]), total_error - 0.5 * tolerance) + 1
        split = order[: min(int(needed), limit - value.size)]
        if split.size == 0:
            break
        mid = 0.5 * (pa[split] + pb[split])
        keep = np.ones(value.size, dtype=bool)
        keep[split] = False
        new_kind = np.concatenate([kind[split], kind[split]])
        new_anchor = np.concatenate([anchor[split], anchor[split]])
        new_a = np.concatenate([pa[split], mid])
        new_b = np.concatenate([mid, pb[split]])
        new_value, new_error = evaluate(new_kind, new_anchor, new_a, new_b)
        kind = np.concatenate([kind[keep], new_kind])
        anchor = np.concatenate([anchor[keep], new_anchor])
        pa = np.concatenate([pa[keep], new_a])
        pb = np.concatenate([pb[keep], new_b])
        value = np.concatenate([value[keep], new_value])
        error = np.concatenate([error[keep], new_error])
    total = float(np.sum(value))
    total_error = float(np.sum(error))
    scale = max(abs(total), 1.0)
    if not np.isfinite(total) or not np.isfinite(total_error) or (
        total_error > EXPECT_MAX_RELATIVE_ERROR * scale
    ):
        raise RuntimeError(
            f"expect quadrature did not converge: value={total!r}, error={total_error!r}"
        )
    return total
