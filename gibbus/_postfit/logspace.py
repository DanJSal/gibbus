"""Stable log-space probability primitives used by post-fit APIs."""

import numpy as np

from .._defaults import LOG_HALF


def _return_shape(value, scalar):
    """Return a Python float for scalar input and an array otherwise.

    Parameters
    ----------
    value : array_like
        Computed scalar or array value.
    scalar : bool
        Whether the original input was scalar.
    """
    return float(value) if scalar else value


def log1mexp(a, /):
    """Compute ``log(1 - exp(a))`` stably for ``a <= 0``.

    Parameters
    ----------
    a : float or array_like
        Log-probability values, which must be no greater than zero.
    """
    arr = np.asarray(a, dtype=np.float64)
    scalar = arr.ndim == 0
    if np.any((arr > 0.0) & ~np.isnan(arr)):
        raise ValueError("log1mexp requires a <= 0")
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        out = np.where(
            arr > LOG_HALF,
            np.log(-np.expm1(arr)),
            np.log1p(-np.exp(arr)),
        )
    return _return_shape(out, scalar)


def log_diff_exp(a, b, /):
    """Compute ``log(exp(a) - exp(b))`` stably for ``a >= b``.

    Parameters
    ----------
    a, b : float or array_like
        Broadcastable logarithms with ``a >= b`` elementwise.
    """
    aa, bb = np.broadcast_arrays(
        np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    )
    scalar = aa.ndim == 0
    invalid = (aa < bb) & ~np.isnan(aa) & ~np.isnan(bb)
    if np.any(invalid):
        raise ValueError("log_diff_exp requires a >= b")
    both_neg_inf = np.isneginf(aa) & np.isneginf(bb)
    with np.errstate(invalid="ignore"):
        delta = bb - aa
    out = aa + np.asarray(log1mexp(delta))
    out = np.where(both_neg_inf, -np.inf, out)
    return _return_shape(out, scalar)


def log_mass_between(log_cdf_lo, log_cdf_hi, log_sf_lo, log_sf_hi, /):
    """Return the log probability of ``(lo, hi]`` from the accurate side.

    Parameters
    ----------
    log_cdf_lo, log_cdf_hi : float or array_like
        Log CDF values at the lower and upper interval endpoints.
    log_sf_lo, log_sf_hi : float or array_like
        Log survival values at the lower and upper interval endpoints.
    """
    lc_lo, lc_hi, ls_lo, ls_hi = np.broadcast_arrays(
        np.asarray(log_cdf_lo, dtype=np.float64),
        np.asarray(log_cdf_hi, dtype=np.float64),
        np.asarray(log_sf_lo, dtype=np.float64),
        np.asarray(log_sf_hi, dtype=np.float64),
    )
    scalar = lc_hi.ndim == 0
    use_cdf = lc_hi <= LOG_HALF
    if scalar:
        if bool(use_cdf):
            return float(log_diff_exp(lc_hi, lc_lo))
        return float(log_diff_exp(ls_lo, ls_hi))

    out = np.empty(lc_hi.shape, dtype=np.float64)
    if np.any(use_cdf):
        out[use_cdf] = np.asarray(
            log_diff_exp(lc_hi[use_cdf], lc_lo[use_cdf]), dtype=np.float64
        )
    use_sf = ~use_cdf
    if np.any(use_sf):
        out[use_sf] = np.asarray(
            log_diff_exp(ls_lo[use_sf], ls_hi[use_sf]), dtype=np.float64
        )
    return out
