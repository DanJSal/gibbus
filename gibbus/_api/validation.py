"""Canonical probability and region inputs for public query boundaries."""

import numpy as np


def _validate_probabilities(p, name):
    """Prepare ordinary probabilities, preserving NaN query semantics.

    Parameters
    ----------
    p : float or array_like
        Candidate probabilities in ``[0, 1]``.
    name : str
        Public method name used in range errors.
    """
    arr = np.asarray(p, dtype=np.float64)
    if np.any(((arr < 0.0) | (arr > 1.0) | np.isinf(arr)) & ~np.isnan(arr)):
        raise ValueError(f"{name} is defined for probabilities in [0, 1]")
    return arr


def _validate_log_probabilities(log_p, name):
    """Prepare logarithmic probabilities, retaining NaN and negative infinity.

    Parameters
    ----------
    log_p : float or array_like
        Candidate logarithmic probabilities no greater than zero.
    name : str
        Public method name used in range errors.
    """
    arr = np.asarray(log_p, dtype=np.float64)
    if np.any((arr > 0.0) & ~np.isnan(arr)):
        raise ValueError(f"{name} requires log_p <= 0")
    return arr


def _validate_level(level, /):
    """Prepare a probability-region mass at a public query boundary.

    Parameters
    ----------
    level : float
        Requested finite probability mass in ``(0, 1]``.
    """
    value = float(level)
    if not np.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError("level must be in (0, 1]")
    return value
