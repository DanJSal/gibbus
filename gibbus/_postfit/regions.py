"""Equal-tailed and highest-density regions for fitted distributions."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .._defaults import (
    HPD_FINAL_MASS_TOL,
    HPD_LEVEL_TOL,
    HPD_MERGE_TOL,
    HPD_MONOTONICITY_TOL,
    QFLAT_TOL,
)
from .logspace import log_mass_between


def interval(ppf, isf, support, level, /):
    """Return an equal-tailed interval.

    Parameters
    ----------
    ppf, isf : callable
        Lower- and upper-tail quantile functions.
    support : array_like, shape (2,)
        Distribution support.
    level : float
        Boundary-validated probability mass in ``(0, 1]``.
    """
    if level == 1.0:
        lo, hi = map(float, support)
        return (lo, hi)
    alpha = 0.5 * (1.0 - level)
    return (float(ppf(alpha)), float(isf(alpha)))


def _mass(logcdf, logsf, a, b):
    """Return probability mass between two endpoints.

    Parameters
    ----------
    logcdf, logsf : callable
        Tail-accurate log-probability evaluators.
    a, b : float
        Interval endpoints.
    """
    lm = log_mass_between(logcdf(a), logcdf(b), logsf(a), logsf(b))
    with np.errstate(under="ignore"):
        return float(np.exp(lm))


def hpd(logpdf, logcdf, logsf, ppf, isf, support, level, /, *, modes):
    """Compute a highest-density region as disjoint intervals.

    Parameters
    ----------
    logpdf, logcdf, logsf : callable
        Log density and tail-accurate log probability evaluators.
    ppf, isf : callable
        Quantile evaluators.
    support : array_like, shape (2,)
        Distribution support.
    level : float
        Boundary-validated probability mass in ``(0, 1]``.
    modes : sequence of float
        Known local modes, used to enrich the crossing grid.
    """
    lo, hi = map(float, support)
    if level == 1.0:
        return np.array([[lo, hi]], dtype=np.float64)

    # Quantile spacing samples equal chunks of probability mass and therefore
    # remains useful for separated mixtures and long tails alike.
    eps = min(1e-10, max(1e-14, (1.0 - level) * 1e-3))
    probs = np.linspace(eps, 1.0 - eps, 4097)
    xs = np.asarray(ppf(probs), dtype=np.float64).reshape(-1)
    extra = [float(m) for m in modes if np.isfinite(m)]
    if np.isfinite(lo):
        extra.append(lo)
    if np.isfinite(hi):
        extra.append(hi)
    if extra:
        xs = np.unique(np.concatenate([xs, np.asarray(extra, dtype=np.float64)]))
    xs = xs[np.isfinite(xs)]
    xs.sort()
    if xs.size < 3:
        return np.array([interval(ppf, isf, support, level)], dtype=np.float64)

    lp = np.asarray(logpdf(xs), dtype=np.float64)
    finite_lp = lp[np.isfinite(lp)]
    if finite_lp.size == 0:
        raise RuntimeError("HPD computation found no finite density values")
    lp_max = float(np.max(finite_lp))
    lp_min = float(np.min(finite_lp))
    if lp_max - lp_min <= QFLAT_TOL * max(1.0, abs(lp_max)):
        return np.array([interval(ppf, isf, support, level)], dtype=np.float64)

    def regions_at(threshold):
        vals = lp - threshold
        inside = vals >= 0.0
        intervals = []
        start = None
        for i in range(xs.size - 1):
            if inside[i] and start is None:
                start = float(xs[i])
            if inside[i] != inside[i + 1]:
                a, b = float(xs[i]), float(xs[i + 1])
                root = float(
                    brentq(
                        lambda z: float(logpdf(z)) - threshold,
                        a,
                        b,
                        xtol=1e-13,
                        rtol=4 * np.finfo(float).eps,
                        maxiter=100,
                    )
                )
                if inside[i]:
                    if start is None:
                        start = a
                    intervals.append([start, root])
                    start = None
                else:
                    start = root
        if inside[-1]:
            if start is None:
                start = float(xs[-1])
            end = (
                hi
                if np.isfinite(hi) and float(logpdf(hi)) >= threshold
                else float(xs[-1])
            )
            intervals.append([start, end])

        if intervals and np.isfinite(lo) and float(logpdf(lo)) >= threshold:
            intervals[0][0] = lo

        # Merge numerical slivers near a grazing valley.
        if len(intervals) > 1:
            span = max(float(intervals[-1][1] - intervals[0][0]), np.finfo(float).tiny)
            merged = [intervals[0]]
            for item in intervals[1:]:
                if item[0] - merged[-1][1] <= HPD_MERGE_TOL * span:
                    merged[-1][1] = item[1]
                else:
                    merged.append(item)
            intervals = merged
        mass = sum(_mass(logcdf, logsf, a, b) for a, b in intervals)
        return intervals, float(mass)

    # Threshold below all sampled log densities gives essentially full mass;
    # threshold at the maximum gives zero mass. Bisect in log-density space.
    lower_thr = lp_min - max(4.0, abs(lp_max - lp_min))
    upper_thr = lp_max
    best = None
    seen = []
    for _ in range(100):
        mid = 0.5 * (lower_thr + upper_thr)
        regs, mass = regions_at(mid)
        for old_thr, old_mass in seen:
            if mid > old_thr and mass > old_mass + HPD_MONOTONICITY_TOL:
                raise RuntimeError("HPD level-set mass was not monotone")
            if mid < old_thr and mass < old_mass - HPD_MONOTONICITY_TOL:
                raise RuntimeError("HPD level-set mass was not monotone")
        seen.append((mid, mass))
        best = regs
        if abs(mass - level) <= HPD_LEVEL_TOL:
            break
        if mass > level:
            lower_thr = mid
        else:
            upper_thr = mid

    if not best:
        raise RuntimeError("HPD computation failed to locate a non-empty region")
    out = np.asarray(best, dtype=np.float64).reshape(-1, 2)
    total = sum(_mass(logcdf, logsf, float(a), float(b)) for a, b in out)
    if abs(total - level) > HPD_FINAL_MASS_TOL:
        raise RuntimeError(
            f"HPD mass did not converge: requested {level!r}, obtained {total!r}"
        )
    return out
