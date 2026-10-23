"""Empirical within-family tail parameters for G3 point-data fits.

The family is always user-selected. A fixed power/rate bypasses this module.
For each requested automatic end, choose a parameter from conditional
upper exceedances (reflect the data for the lower end). Only the empirical
model is optimized; global G3 fitting is not an outer optimization loop.

This is an approximate statistical screening procedure, not an estimator of
unobservable ultimate asymptotics. Numerical certification remains the
responsibility of the ordinary G3 solver. A second proposal is retained
solely for a single numerical-failure recovery attempt.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .._model.basis_spec import (
    Exponential,
    Polynomial,
    SoftPower,
    _needs_tail_estimation,
)
from ._empirical_tail_likelihood import (
    _h_exp,
    _h_power,
    potential_fixed,
    prepare_tail,
    tail_data,
)
from .budget import _check_fit_time


@dataclass(frozen=True)
class TailEstimate:
    """Selected physical parameter, competitive backup, and tail sample size."""

    primary: float
    alternative: float | None
    n_tail: int
    family: str


def _prepare_values(samples, weights):
    """Point observations; replicate weighted observations deterministically.

    A deterministic weighted quantile grid is used only for nonuniform weights.
    This avoids randomized tail proposals and prevents a single massive weight
    from creating an arbitrarily large replicate array.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size < 20 or not np.all(np.isfinite(x)):
        raise ValueError(
            "automatic tail parameters require >=20 finite point observations"
        )
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if (
            w.shape != x.shape
            or np.any(w < 0)
            or not np.all(np.isfinite(w))
            or not np.any(w > 0)
        ):
            raise ValueError("invalid weights for empirical tail estimation")
        w = w / np.max(w)
        w = w / np.sum(w)
        if not np.allclose(w, 1 / x.size, rtol=1e-12, atol=1e-15):
            ids = np.argsort(x)
            xs, ws = x[ids], w[ids]
            positive = ws > 0
            xs, ws = xs[positive], ws[positive]
            cdf = np.cumsum(ws)
            count = max(64, min(2048, x.size))
            magnitude = max(float(np.max(np.abs(xs))), np.finfo(float).tiny)
            x = magnitude * np.interp(
                (np.arange(count) + 0.5) / count,
                np.r_[0.0, cdf],
                np.r_[xs[0], xs] / magnitude,
            )
    if not np.any(x != x[0]):
        raise ValueError("automatic tail parameters require nonzero sample dispersion")
    return x


def estimate_tail(x, *, family, side, fraction=0.4):
    """Profile fixed-family conditional-tail likelihood; return top candidates.

    Polynomial candidates are integers 1..6. SoftPower first screens those
    integers, then optimizes in the best integer's clipped +/-1 interval.
    Exponential rates are optimized in standardized inverse-data units.
    """
    if family not in {"polynomial", "soft_power", "exponential"}:
        raise ValueError(f"unknown tail family {family!r}")
    x = np.asarray(x, dtype=np.float64)
    oriented, scale = tail_data(x, side, return_scale=True)
    prepared = None
    for f in (fraction, 0.3, 0.2, 0.1):
        try:
            y, u, med = prepare_tail(oriented, f)
            if y.size >= 8:
                prepared = y, u, med
                break
        except (ValueError, FloatingPointError):
            continue
    if prepared is None:
        raise ValueError(
            "insufficient informative tail observations for automatic tail parameter"
        )
    y, u, med = prepared
    kind = "exp" if family == "exponential" else "power"
    cache = {}

    def model(v):
        _check_fit_time("tail parameter candidate")
        key = round(float(v), 5)
        if key not in cache:
            try:
                vfit = potential_fixed(y, u, med, kind=kind, p=float(v))
                if not math.isfinite(vfit["ll"]):
                    vfit = None
            except (ArithmeticError, ValueError, OverflowError, FloatingPointError):
                vfit = None
            cache[key] = vfit
        return cache[key]

    def score(v):
        m = model(v)
        return -math.inf if m is None else m["ll"]

    integers = tuple(float(i) for i in range(1, 7))
    if family == "polynomial":
        primary = max(integers, key=score)
        trials = integers
    elif family == "soft_power":
        n = max(integers, key=score)
        low, high = max(1.0, n - 1.0), min(6.0, n + 1.0)
        fit = minimize_scalar(
            lambda p: -score(p),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 0.02, "maxiter": 18},
        )
        if not fit.success:
            raise RuntimeError("soft-power tail parameter search did not converge")
        primary = max((n, float(fit.x), low, high), key=score)
        trials = sorted(
            {primary, n, low, high, max(1.0, primary - 0.15), min(6.0, primary + 0.15)}
        )
    else:
        fit = minimize_scalar(
            lambda p: -score(p),
            bounds=(0.12, 5.0),
            method="bounded",
            options={"xatol": 0.03, "maxiter": 18},
        )
        if not fit.success:
            raise RuntimeError("exponential tail parameter search did not converge")
        primary = float(fit.x)
        trials = sorted(
            {
                primary,
                max(0.12, primary * 0.75),
                min(5.0, primary * 1.25),
                max(0.12, primary * 0.9),
                min(5.0, primary * 1.1),
            }
        )
    if model(primary) is None:
        raise RuntimeError(
            "automatic tail estimation could not fit a finite conditional likelihood"
        )
    # The backup is a different parameter with a statistically competitive
    # conditional-tail likelihood, not the candidate with largest coefficient.
    best = model(primary)
    primary_h = (
        _h_exp(y, primary, med) if kind == "exp" else _h_power(y, primary, u, med)
    )
    backups = []
    for v in trials:
        if v == primary or model(v) is None:
            continue
        m = model(v)
        h = _h_exp(y, v, med) if kind == "exp" else _h_power(y, v, u, med)
        diff = -(best["b"] - m["b"]) * y - (best["a"] * primary_h - m["a"] * h)
        if not np.all(np.isfinite(diff)):
            continue
        se = float(np.std(diff, ddof=1) / math.sqrt(len(y)))
        loss = float(score(primary) - score(v))
        if loss <= max(1e-4, se):
            backups.append((loss, float(v)))
    backups.sort()
    physical = (lambda p: p / scale) if family == "exponential" else float
    if not math.isfinite(physical(primary)) or physical(primary) <= 0:
        raise ValueError(
            "estimated tail parameter is not representable in physical units; rescale the observations"
        )
    return TailEstimate(
        float(physical(primary)),
        float(physical(backups[0][1])) if backups else None,
        int(y.size),
        family,
    )


def resolve_auto_tails(spec, samples, weights=None):
    """Replace omitted powers/rates once for a fit; return spec and proposals.

    ``samples`` is a canonical (R,1) point sample. Interval-censored data
    require a separate censoring-aware empirical estimator; never substitute
    interval midpoints silently.
    """
    from dataclasses import replace

    required = [
        (side, getattr(spec, side + "_tail"))
        for side in ("lower", "upper")
        if _needs_tail_estimation(getattr(spec, side + "_tail"))
    ]
    if not required:
        return spec, {}
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 1:
        raise ValueError(
            "automatic tail parameters currently require point observations; specify explicit parameters for interval-censored data"
        )
    x = _prepare_values(data[:, 0], weights)
    estimates = {}
    result = spec
    for side, term in required:
        _check_fit_time("automatic tail parameters")
        if isinstance(term, Polynomial):
            family = "polynomial"

            def ctor(v, old):
                return Polynomial(int(round(v)))

        elif isinstance(term, SoftPower):
            family = "soft_power"

            def ctor(v, old):
                return SoftPower(float(v), delta=old.delta)

        else:
            family = "exponential"

            def ctor(v, old):
                return Exponential(float(v))

        est = estimate_tail(x, family=family, side=side)
        estimates[side] = (term, est, ctor)
        result = replace(result, **{side + "_tail": ctor(est.primary, term)})
    return result, estimates


def alternative_spec(spec, estimates):
    """One backup configuration, prioritizing the first side with a backup."""
    from dataclasses import replace

    for side, (term, est, ctor) in estimates.items():
        if est.alternative is not None:
            return replace(spec, **{side + "_tail": ctor(est.alternative, term)})
    return None
