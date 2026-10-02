"""Held-out likelihood and probability-integral-transform helpers."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import ndtri_exp

from .._defaults import EXPECT_MAX_RELATIVE_ERROR, LOG_HALF, NARROW_LOG_MASS_GAP
from .._fit.inputs import _canon_univariate_samples, _to_generator
from .logspace import log_diff_exp, log_mass_between
from .survival import _quad_with_ledger


def canonical_scoring_rows(x, /):
    """Canonicalize exact or interval scoring rows without a fit-size minimum.

    Parameters
    ----------
    x : array_like
        Point observations or ``(n, 2)`` intervals.
    """
    return _canon_univariate_samples(x, min_samples=1)[0]


def scoring_weights(n, sample_weight, /):
    """Validate scoring weights without normalizing them.

    Parameters
    ----------
    n : int
        Number of observations.
    sample_weight : array_like or None
        Optional non-negative finite row weights.
    """
    if sample_weight is None:
        return np.ones(int(n), dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if w.size != int(n) or np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("sample_weight must be finite, non-negative, and length n")
    return w


def warn_out_of_support(rows, support, /):
    """Warn when scoring rows have zero mass because they miss the support.

    Parameters
    ----------
    rows : numpy.ndarray, shape (n, 1) or (n, 2)
        Canonical exact or interval observations.
    support : array_like, shape (2,)
        Active distribution support.
    """
    lo, hi = map(float, support)
    if rows.shape[1] == 1:
        outside = (rows[:, 0] < lo) | (rows[:, 0] > hi)
    else:
        outside = (rows[:, 1] <= lo) | (rows[:, 0] >= hi)
    count = int(np.count_nonzero(outside))
    if count:
        warnings.warn(
            f"{count} scoring row(s) lie outside the fitted support and contribute "
            "zero probability",
            RuntimeWarning,
            stacklevel=3,
        )
    return count


def _direct_interval_log_mass(logpdf, lo, hi, /):
    """Integrate a finite interval mass after factoring out local log density.

    Parameters
    ----------
    logpdf : callable
        Active-space log-density evaluator.
    lo, hi : float
        Finite interval endpoints with ``lo < hi``.
    """
    width = float(hi - lo)
    if not np.isfinite(width) or width <= 0.0:
        return np.nan
    mid = float(lo + 0.5 * width)
    probes = np.asarray(
        logpdf(np.array([lo, mid, hi], dtype=np.float64)), dtype=np.float64
    )
    finite = probes[np.isfinite(probes)]
    if finite.size == 0:
        return np.nan
    anchor = float(np.max(finite))

    def relative(u):
        t = lo + width * u
        lp = float(logpdf(t))
        if not np.isfinite(lp):
            return 0.0
        return float(np.exp(lp - anchor))

    value, error = _quad_with_ledger(
        relative,
        0.0,
        1.0,
        epsabs=1e-12,
        epsrel=1e-11,
        limit=100,
        context="direct interval log-mass quadrature",
    )
    scale = max(abs(value), 1.0)
    if (
        not np.isfinite(value)
        or not np.isfinite(error)
        or value <= 0.0
        or error > EXPECT_MAX_RELATIVE_ERROR * scale
    ):
        return np.nan
    return float(np.log(width) + anchor + np.log(value))


def interval_loglik(rows, weights, logpdf, logcdf, logsf, /):
    """Evaluate weighted log likelihood for canonical interval rows.

    Parameters
    ----------
    rows : numpy.ndarray, shape (n, 2)
        Canonical observation intervals.
    weights : numpy.ndarray, shape (n,)
        Unnormalized scoring weights.
    logpdf, logcdf, logsf : callable
        Active-space probability evaluators.
    """
    lo = rows[:, 0]
    hi = rows[:, 1]
    exact = lo == hi
    terms = np.empty(rows.shape[0], dtype=np.float64)
    if np.any(exact):
        terms[exact] = np.asarray(logpdf(lo[exact]), dtype=np.float64)
    if np.any(~exact):
        idxs = np.flatnonzero(~exact)
        lc_lo = np.asarray(logcdf(lo[idxs]), dtype=np.float64)
        lc_hi = np.asarray(logcdf(hi[idxs]), dtype=np.float64)
        ls_lo = np.asarray(logsf(lo[idxs]), dtype=np.float64)
        ls_hi = np.asarray(logsf(hi[idxs]), dtype=np.float64)
        masses = np.asarray(
            log_mass_between(lc_lo, lc_hi, ls_lo, ls_hi), dtype=np.float64
        )
        use_cdf = lc_hi <= LOG_HALF
        # Rows with zero mass give -inf - (-inf); the NaN gap is correctly
        # rejected by ``gaps >= 0`` below, so silence only that arithmetic.
        # np.errstate is thread-local, unlike warnings.catch_warnings.
        with np.errstate(invalid="ignore"):
            gaps = np.where(use_cdf, lc_hi - lc_lo, ls_lo - ls_hi)
        narrow = (
            np.isfinite(lo[idxs])
            & np.isfinite(hi[idxs])
            & (gaps >= 0.0)
            & (gaps < NARROW_LOG_MASS_GAP)
        )
        for local_index in np.flatnonzero(narrow):
            direct = _direct_interval_log_mass(
                logpdf, float(lo[idxs[local_index]]), float(hi[idxs[local_index]])
            )
            if np.isfinite(direct):
                masses[local_index] = direct
        terms[idxs] = masses
    positive = weights > 0.0
    return float(np.sum(weights[positive] * terms[positive], dtype=np.float64))


def _ndtri_from_log(logp):
    """Evaluate the normal quantile from logarithmic probabilities.

    Parameters
    ----------
    logp : array_like
        Natural logarithms of lower-tail probabilities.
    """
    return ndtri_exp(np.asarray(logp, dtype=np.float64))


def randomized_quantile_residuals(rows, logcdf, logsf, /, *, rng=None):
    """Return Dunn-Smyth randomized quantile residuals for canonical rows.

    Parameters
    ----------
    rows : numpy.ndarray, shape (n, 1) or (n, 2)
        Canonical exact or interval observations.
    logcdf, logsf : callable
        Tail-accurate active-space probability evaluators.
    rng : optional
        Random-number source accepted by gibbus sampling methods.
    """
    gen = _to_generator(rng)
    if rows.shape[1] == 1:
        lc = np.asarray(logcdf(rows[:, 0]), dtype=np.float64)
        ls = np.asarray(logsf(rows[:, 0]), dtype=np.float64)
        lower = lc <= LOG_HALF
        out = np.empty(rows.shape[0], dtype=np.float64)
        out[lower] = _ndtri_from_log(lc[lower])
        out[~lower] = -_ndtri_from_log(ls[~lower])
        return out

    lo, hi = rows[:, 0], rows[:, 1]
    exact = lo == hi
    out = np.empty(rows.shape[0], dtype=np.float64)
    if np.any(exact):
        single = rows[exact, :1]
        out[exact] = randomized_quantile_residuals(single, logcdf, logsf, rng=gen)
    if np.any(~exact):
        idxs = np.flatnonzero(~exact)
        lc_lo = np.asarray(logcdf(lo[idxs]), dtype=np.float64)
        lc_hi = np.asarray(logcdf(hi[idxs]), dtype=np.float64)
        ls_lo = np.asarray(logsf(lo[idxs]), dtype=np.float64)
        ls_hi = np.asarray(logsf(hi[idxs]), dtype=np.float64)
        r = gen.random(idxs.size)
        use_cdf = lc_hi <= LOG_HALF
        vals = np.empty(idxs.size, dtype=np.float64)
        if np.any(use_cdf):
            j = use_cdf
            lm = np.asarray(log_diff_exp(lc_hi[j], lc_lo[j]))
            logu = np.logaddexp(lc_lo[j], np.log(r[j]) + lm)
            vals[j] = _ndtri_from_log(logu)
        if np.any(~use_cdf):
            j = ~use_cdf
            lm = np.asarray(log_diff_exp(ls_lo[j], ls_hi[j]))
            logs = np.logaddexp(ls_hi[j], np.log(r[j]) + lm)
            vals[j] = -_ndtri_from_log(logs)
        out[idxs] = vals
    return out
