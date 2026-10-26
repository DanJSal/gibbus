"""Empirical within-family tail parameters for point-data fits.

The family is always user-selected. A fixed power/rate bypasses this module.
For each requested automatic end, choose a parameter from conditional
upper exceedances (reflect the data for the lower end). Only the empirical
model is optimized; the fit is not an outer optimization loop.

This is an approximate statistical screening procedure, not an estimator of
unobservable ultimate asymptotics; the fit itself verifies the shape, and a
fit that fails with the estimate fails.
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
    potential_fixed,
    prepare_tail,
    tail_data,
)
from .budget import _check_fit_time

# Integer powers screened by the empirical tail likelihood (1 through this).
_MAX_SCREENED_POWER = 6
# The open soft-power range p > 1 is searched from here.
_MIN_SOFT_POWER = 1.0 + 1e-6
# An estimate this close to p = 1 means the tail is too heavy for the family.
_SOFT_POWER_BOUNDARY = 1e-3
# Decade widenings of the exponential-rate bracket.
_BRACKET_WIDENINGS = 8


@dataclass(frozen=True)
class TailEstimate:
    """Selected physical parameter, tail sample size and family."""

    primary: float
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

    Polynomial candidates are the integers 1 through 6.  A soft power takes
    the best of those integers and is refined continuously within one of it
    (``p > 1``; the refinement may leave the screened range).  An exponential
    rate is optimized in log standardized inverse-data units from 0.12,
    widening the bracket upward until the optimum is interior.
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

    # Integer powers are screened over a fixed range: the conditional
    # likelihood of a few dozen exceedances is nearly flat in the power for
    # light tails, and an open scan climbs to spurious high powers there
    # (Gaussian data: p = 10).  This is a screening policy, not a limit of
    # the family, which accepts any explicit p > 1.
    integers = tuple(float(i) for i in range(1, _MAX_SCREENED_POWER + 1))
    if family == "polynomial":
        primary = max(integers, key=score)
    elif family == "soft_power":
        n = max(integers, key=score)
        low, high = max(_MIN_SOFT_POWER, n - 1.0), n + 1.0
        fit = minimize_scalar(
            lambda p: -score(p),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 0.02, "maxiter": 18},
        )
        if not fit.success:
            raise RuntimeError("soft-power tail parameter search did not converge")
        primary = max((n, float(fit.x), low, high), key=score)
        if primary < 1.0 + _SOFT_POWER_BOUNDARY:
            # The likelihood climbs toward p = 1: the tail is at least as
            # heavy as linear growth, which no soft power p > 1 follows (the
            # fit would run into the improper boundary).
            raise ValueError(
                f"the {side} tail looks heavier than any SoftPower: the "
                "automatic estimate reaches p = 1; give an explicit power or "
                "another tail family"
            )
    else:
        # The bracket widens upward (steeper tails) until the optimum is
        # interior.  Its lower end stays: as lambda -> 0 the empirical model
        # tends to a quadratic potential, a different family whose
        # likelihood the profile would otherwise follow (Gumbel's upper tail
        # chose lambda ~ 1e-4).
        lo_l, hi_l = math.log(0.12), math.log(5.0)
        for _ in range(_BRACKET_WIDENINGS):
            fit = minimize_scalar(
                lambda t: -score(math.exp(t)),
                bounds=(lo_l, hi_l),
                method="bounded",
                options={"xatol": 0.006, "maxiter": 24},
            )
            if not fit.success:
                raise RuntimeError("exponential tail parameter search did not converge")
            t = float(fit.x)
            if hi_l - t < 0.05:
                hi_l += math.log(10.0)
            else:
                break
        primary = math.exp(t)
    if model(primary) is None:
        raise RuntimeError(
            "automatic tail estimation could not fit a finite conditional likelihood"
        )
    physical = (lambda p: p / scale) if family == "exponential" else float
    if not math.isfinite(physical(primary)) or physical(primary) <= 0:
        raise ValueError(
            "estimated tail parameter is not representable in physical units; rescale the observations"
        )
    return TailEstimate(float(physical(primary)), int(y.size), family)


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
                return SoftPower(float(v))

        else:
            family = "exponential"

            def ctor(v, old):
                return Exponential(float(v))

        est = estimate_tail(x, family=family, side=side)
        estimates[side] = (term, est, ctor)
        result = replace(result, **{side + "_tail": ctor(est.primary, term)})
    return result, estimates
