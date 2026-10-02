"""Empirical-distribution-function goodness-of-fit statistics on PIT values.

All three statistics measure the same thing -- how far a sample of
probability-integral-transform values sits from uniformity -- but weight the
distance differently.  Kolmogorov-Smirnov uses the supremum deviation and is
least sensitive in the tails; Cramer-von Mises integrates the squared
deviation with uniform weight; Anderson-Darling integrates it with weight
``1 / (u (1 - u))``, which concentrates attention exactly where a
log-concave tail model is most likely to be wrong.

The statistics here are computed from PIT values alone and know nothing
about how those values were produced.  Whether a resulting *p*-value is
meaningful depends entirely on that provenance, which is the caller's
responsibility; see :meth:`gibbus.Distribution.goodness_of_fit`.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import cramervonmises, kstwo

from .._defaults import PIT_CLIP

GOF_STATISTICS = ("ks", "cvm", "ad")
"""Statistic keys accepted by :func:`gof_statistic`."""


def validate_statistic(statistic, /):
    """Validate and normalize a goodness-of-fit statistic key.

    Parameters
    ----------
    statistic : str
        Candidate key; compared case-insensitively after stripping
        surrounding whitespace.

    Returns
    -------
    str
        The normalized key, one of :data:`GOF_STATISTICS`.

    Raises
    ------
    ValueError
        If *statistic* is not a recognised key.
    """
    key = str(statistic).strip().lower()
    if key not in GOF_STATISTICS:
        raise ValueError(
            f"statistic must be one of {GOF_STATISTICS}, got {statistic!r}"
        )
    return key


def canonical_pit(u, /):
    """Validate PIT values and clip them away from the open-interval endpoints.

    Exact ``0.0`` and ``1.0`` are legitimate outputs of a fitted CDF for
    observations at or beyond a bounded support, but they send the
    Anderson-Darling weight to infinity.  Clipping by :data:`PIT_CLIP`
    bounds the statistic while leaving any genuinely interior value
    untouched at float64 resolution.

    Parameters
    ----------
    u : array_like
        Probability-integral-transform values, expected in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Sorted, clipped float64 PIT values.

    Raises
    ------
    ValueError
        If *u* is empty, contains NaN, or contains a value outside
        ``[0, 1]``.
    """
    arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("goodness-of-fit requires at least one PIT value")
    if np.any(np.isnan(arr)):
        raise ValueError("PIT values must not contain NaN")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("PIT values must lie in [0, 1]")
    return np.sort(np.clip(arr, PIT_CLIP, 1.0 - PIT_CLIP))


def gof_statistic(u, statistic, /):
    """Evaluate one EDF goodness-of-fit statistic against uniformity.

    Parameters
    ----------
    u : array_like
        Probability-integral-transform values in ``[0, 1]``.
    statistic : str
        One of :data:`GOF_STATISTICS`: ``"ks"`` for the two-sided
        Kolmogorov-Smirnov supremum, ``"cvm"`` for the Cramer-von Mises
        integral, or ``"ad"`` for the Anderson-Darling integral.

    Returns
    -------
    float
        The statistic value; larger means further from uniform.
    """
    key = validate_statistic(statistic)
    us = canonical_pit(u)
    n = us.size
    i = np.arange(1, n + 1, dtype=np.float64)
    if key == "ks":
        d_plus = float(np.max(i / n - us))
        d_minus = float(np.max(us - (i - 1.0) / n))
        return max(d_plus, d_minus)
    if key == "cvm":
        return float(
            np.sum((us - (2.0 * i - 1.0) / (2.0 * n)) ** 2) + 1.0 / (12.0 * n)
        )
    terms = (2.0 * i - 1.0) * (np.log(us) + np.log1p(-us[::-1]))
    return float(-n - np.sum(terms) / n)


def asymptotic_pvalue(u, statistic, /):
    """Return the null *p*-value for PIT values independent of the fitted model.

    The reference distributions used here assume the fit did not see the
    data being tested, so they are calibrated only for genuinely held-out
    observations.  Anderson-Darling has no closed-form null distribution
    exposed by SciPy, so it returns ``None`` rather than an approximation
    whose error would be invisible to the caller; use Monte Carlo
    calibration for that statistic instead.

    Parameters
    ----------
    u : array_like
        Probability-integral-transform values in ``[0, 1]``.
    statistic : str
        One of :data:`GOF_STATISTICS`.

    Returns
    -------
    float or None
        Upper-tail probability under the null, or ``None`` when no
        calibrated asymptotic reference exists for *statistic*.
    """
    key = validate_statistic(statistic)
    us = canonical_pit(u)
    if key == "ks":
        return float(np.clip(kstwo.sf(gof_statistic(us, "ks"), us.size), 0.0, 1.0))
    if key == "cvm":
        return float(np.clip(cramervonmises(us, "uniform").pvalue, 0.0, 1.0))
    return None


def monte_carlo_pvalue(observed, replicates, /):
    """Return a bias-corrected Monte Carlo *p*-value.

    Uses the ``(1 + #{replicate >= observed}) / (1 + B)`` estimator, which
    never returns exactly zero.  A plain proportion can report ``p = 0``
    from a finite simulation, overstating the evidence; that is a
    well-documented reporting error in simulated tests.

    Parameters
    ----------
    observed : float
        Statistic computed on the real sample.
    replicates : array_like
        Statistics computed on samples simulated under the null.

    Returns
    -------
    float
        Estimated upper-tail probability in ``(0, 1]``.

    Raises
    ------
    ValueError
        If *replicates* contains no finite value.
    """
    reps = np.asarray(replicates, dtype=np.float64).reshape(-1)
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        raise ValueError("Monte Carlo p-value requires at least one finite replicate")
    exceed = int(np.count_nonzero(reps >= float(observed)))
    return float((1.0 + exceed) / (1.0 + reps.size))
