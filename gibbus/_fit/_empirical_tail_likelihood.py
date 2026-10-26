"""Conditional tail likelihood with analytic gradients and 96-node quadrature."""

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

from . import _tail_likelihood_kernels as _tail_kernels
from .budget import _check_fit_time

GX, GW = roots_laguerre(96)
LOGW = np.log(GW)


def tail_data(raw, side, *, return_scale=False):
    """Standardize finite observations without squaring their physical scale."""
    x = np.asarray(raw, dtype=float)
    if side not in {"lower", "upper"}:
        raise ValueError("side must be 'lower' or 'upper'")
    z = -x if side == "lower" else x
    magnitude = float(np.max(np.abs(z)))
    if not math.isfinite(magnitude) or magnitude == 0:
        raise ValueError("tail observations require finite nonzero dispersion")
    scaled = z / magnitude
    centered = scaled - float(np.median(scaled))
    spread = float(np.std(centered))
    scale = magnitude * spread
    if not (math.isfinite(scale) and scale > 0 and spread > 0):
        raise ValueError(
            "tail sample dispersion is not representable; rescale the observations"
        )
    result = centered / spread
    return (result, scale) if return_scale else result


def _h_power(y, p, u, med):
    if p <= 1 + 1e-9:
        return np.zeros_like(y)
    y = np.asarray(y)
    v = y / u
    vm = med / u
    raw = np.expm1(p * np.log1p(v)) - p * v
    dm = float(np.expm1(p * np.log1p(vm)) - p * vm)
    if dm <= 0 or not np.isfinite(dm):
        return np.full_like(y, np.inf)
    return raw / dm


def _h_exp(y, lam, med):
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.expm1(lam * np.asarray(y)) - lam * np.asarray(y)
    den = float(np.expm1(lam * med) - lam * med)
    return raw / den


def prepare_tail(x, fraction):
    x = np.asarray(x, float)
    u = float(np.quantile(x, 1 - fraction))
    y = x[x > u] - u
    med = float(np.median(y))
    if not (u > 0 and med > 0):
        raise ValueError("positive origin required")
    return y, u, med


def potential_fixed(y, u, med, kind="power", p=2, starts=None):
    ym = float(np.mean(y))
    if kind == "power" and p == 1:
        b = 1 / ym
        return dict(
            kind="P1", p=1.0, b=float(b), a=0.0, ll=float(np.log(b) - 1), npar=1
        )
    geth = (
        (lambda z: _h_power(z, p, u, med))
        if kind == "power"
        else (lambda z: _h_exp(z, p, med))
    )
    mh = float(np.mean(geth(y)))
    if not np.isfinite(mh):
        raise ArithmeticError("nonfinite mean h")
    rate = 1 / ym
    hnodes = geth(GX / rate)
    if np.isnan(hnodes).any():
        raise ArithmeticError("nan quadrature basis")
    ynodes = np.ascontiguousarray(GX / rate, dtype=np.float64)
    hnodes = np.ascontiguousarray(hnodes, dtype=np.float64)
    logwr = np.ascontiguousarray(LOGW + GX, dtype=np.float64)

    def obj_and_grad(z):
        _check_fit_time("tail likelihood optimization")
        return _tail_kernels.objective_gradient(
            z[0], z[1], ym, mh, rate, ynodes, hnodes, logwr
        )

    bounds = [(-8, 5), (-13, 7)]
    xstarts = (
        starts
        if starts is not None
        else [
            (math.log(max(1 / ym, 1e-5)), math.log(0.1)),
            (math.log(max(0.6 / ym, 1e-5)), math.log(0.5)),
        ]
    )
    fits = []
    for s in xstarts:
        _check_fit_time("tail likelihood start")
        opt = minimize(
            obj_and_grad,
            s,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 100, "ftol": 3e-11, "gtol": 1e-7},
        )
        if (
            opt.success
            and np.isfinite(opt.fun)
            and opt.fun < 1e20
            and np.isfinite(opt.x).all()
        ):
            fits.append(opt)
            break
    if not fits:
        raise ArithmeticError("conditional tail optimization did not converge")
    o = min(fits, key=lambda v: v.fun)
    return dict(
        kind="EXP" if kind == "exp" else f"P{p}",
        p=float(p),
        b=float(np.exp(o.x[0])),
        a=float(np.exp(o.x[1])),
        ll=float(-o.fun),
        npar=2,
    )
