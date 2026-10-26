"""Canonical observation weights and Kish effective counts.

The likelihood's sufficient statistics are the basis statistics of
``_fit.basis_objective``.  Weights are validated, normalized to sum to one and summarized once here for every
point and interval likelihood.
"""

from typing import NamedTuple

import numpy as np


class _WeightSummary(NamedTuple):
    """Normalized row weights with their original reliability metadata.

    Parameters
    ----------
    weights : numpy.ndarray, shape (R,), dtype float64
        Nonnegative row masses normalized to sum to one.
    total_weight : float
        Total mass before normalization; it may overflow for raw relative weights.
    effective_n : float
        Original-row Kish effective sample size, before duplicate compression.
    """

    weights: np.ndarray
    total_weight: float
    effective_n: float


def _canonical_weights(n, weights, /):
    """Describe boundary-normalized weights without converting or normalizing.

    Parameters
    ----------
    n : int
        Positive number of original observation rows.
    weights : numpy.ndarray, shape (n,), dtype float64 or None
        Already-validated weights summing to one, or ``None`` for equal weights.

    Returns
    -------
    _WeightSummary
        Original reliability metadata, sharing a supplied canonical array.
    """
    if weights is None:
        return _WeightSummary(np.full(n, 1.0 / n, dtype=np.float64), float(n), float(n))
    return _WeightSummary(weights, 1.0, float(1.0 / np.dot(weights, weights)))


def _normalized_weights(n, weights, subject="point", /):
    """Return normalized weights plus total and effective sample size.

    This prepares newly derived responsibility masses or subsets. Original
    observation weights are normalized at the API boundary and described by
    :func:`_canonical_weights` instead.

    Normalization runs in units of the largest weight so that neither the raw
    sum nor the sum of squares can overflow merely because every relative
    weight shares a large common scale.

    Parameters
    ----------
    n : int
        Number of observations.
    weights : array_like or None
        Non-negative finite weights.  ``None`` means equal weights.
    subject : str, optional
        Noun naming the observation kind in error messages, for example
        ``"point"`` or ``"interval"``.

    Returns
    -------
    _WeightSummary
        Normalized weights, pre-normalization total and original-row Kish size.

    Raises
    ------
    ValueError
        If the weight vector is invalid or has no positive total mass.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"at least one {subject} observation is required")
    if weights is None:
        return _canonical_weights(n, None)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n:
        raise ValueError(f"weights must have one entry per {subject} observation")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    wmax = float(np.max(w))
    if not (wmax > 0.0):
        raise ValueError("total observation weight must be positive")
    scaled = w / wmax
    scaled_total = float(np.sum(scaled, dtype=np.float64))
    if not (scaled_total > 0.0 and np.isfinite(scaled_total)):
        raise ValueError("total observation weight must be positive")
    norm = np.ascontiguousarray(scaled / scaled_total, dtype=np.float64)
    sum_sq_norm = float(np.dot(norm, norm))
    if not (sum_sq_norm > 0.0 and np.isfinite(sum_sq_norm)):
        raise ValueError("squared observation weight must be positive and finite")
    with np.errstate(over="ignore"):
        total = float(wmax * scaled_total)
    return _WeightSummary(norm, total, float(1.0 / sum_sq_norm))
