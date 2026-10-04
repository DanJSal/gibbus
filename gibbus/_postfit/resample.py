"""Nonparametric and parametric resampling for fitted-model uncertainty.

Both public entry points take the per-replicate work as a callable.  That
keeps the resampling loop -- index generation, failure accounting, band
aggregation -- independent of the fitting machinery, so it can be exercised
directly without building a model.

A replicate is counted as failed when the refit raises one of
:data:`gibbus._defaults.NUMERIC_FAILURES`, explicitly declines by returning
``None``, or returns non-finite numerical output.  A result with the wrong
shape is a callback contract error and is raised immediately.
Failures are expected at low rates: a resample can omit enough of a mode's
support that the log-concave fit for that component degenerates.  They are
tolerated up to a fraction of the requested total and reported in the
result, because silently averaging over a run in which most replicates
failed would turn a broken uncertainty estimate into a confident one.
"""

from __future__ import annotations

import numpy as np

from .._defaults import NUMERIC_FAILURES, _reraise_if_debug


def validate_resample_count(n_resamples, /):
    """Validate the requested number of resampling replicates.

    Parameters
    ----------
    n_resamples : int
        Requested replicate count.  Integral floats such as ``200.0`` are
        accepted; ``bool`` is rejected because ``True`` almost certainly
        means a mistyped argument rather than one replicate.

    Returns
    -------
    int
        The validated count.

    Raises
    ------
    TypeError
        If *n_resamples* is a ``bool`` or not an integral number.
    ValueError
        If *n_resamples* is less than one.
    """
    if isinstance(n_resamples, bool):
        raise TypeError("n_resamples must be an integer, not a bool")
    value = np.asarray(n_resamples)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.number):
        raise TypeError("n_resamples must be a scalar integer")
    as_float = float(value)
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise TypeError(f"n_resamples must be integral, got {n_resamples!r}")
    count = int(as_float)
    if count < 1:
        raise ValueError("n_resamples must be at least 1")
    return count


def validate_confidence(level, /):
    """Validate a two-sided confidence level.

    Parameters
    ----------
    level : float
        Requested coverage, strictly between zero and one.

    Returns
    -------
    float
        The validated level.

    Raises
    ------
    ValueError
        If *level* is not a finite number in the open interval ``(0, 1)``.
    """
    value = float(level)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"level must lie strictly in (0, 1), got {level!r}")
    return value


def percentile_bands(curves, level, /):
    """Aggregate replicate curves into pointwise percentile bands.

    The bands are *pointwise*: each abscissa is covered at the requested
    level in isolation.  A band that covers the whole curve simultaneously
    is wider, and these must not be read as one.

    Parameters
    ----------
    curves : array_like, shape (B, M)
        One row per successful replicate, one column per abscissa.
    level : float
        Already-validated two-sided coverage in ``(0, 1)``.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper band arrays, each of shape ``(M,)``.

    Raises
    ------
    ValueError
        If *curves* is not two-dimensional or has no rows.
    """
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("curves must be two-dimensional, shape (B, M)")
    if arr.shape[0] == 0:
        raise ValueError("percentile bands require at least one replicate curve")
    tail = 50.0 * (1.0 - level)
    lower, upper = np.percentile(arr, [tail, 100.0 - tail], axis=0)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _collect_replicates(
    make_replicate, n_resamples, width, context, /, *, max_failure_fraction
):
    """Run replicate work, discarding failures and enforcing a failure budget.

    Parameters
    ----------
    make_replicate : callable
        Zero-argument callable returning either a length-*width* array of
        replicate values, or ``None`` to mark the replicate as failed.
    n_resamples : int
        Number of replicates to attempt.
    width : int
        Expected length of each replicate result.
    context : str
        Short description used in the failure ledger and error messages.
    max_failure_fraction : float
        Largest tolerated share of failed replicates.

    Returns
    -------
    tuple
        ``(rows, n_failed)`` where ``rows`` is a ``(B, width)`` float64
        array of successful replicates.

    Raises
    ------
    RuntimeError
        If the failed share exceeds *max_failure_fraction*, or if every
        replicate failed.
    """
    rows = []
    n_failed = 0
    for _ in range(n_resamples):
        try:
            value = make_replicate()
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, context, routine=True)
            n_failed += 1
            continue
        if value is None:
            n_failed += 1
            continue
        row = np.asarray(value, dtype=np.float64).reshape(-1)
        if row.size != width:
            raise ValueError(
                f"{context} replicate returned {row.size} values; expected {width}"
            )
        if not np.all(np.isfinite(row)):
            exc = FloatingPointError(f"{context} replicate returned non-finite values")
            _reraise_if_debug(exc, context, routine=True)
            n_failed += 1
            continue
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"every {context} replicate failed ({n_failed} of {n_resamples}); "
            "the fit is too fragile under resampling to quantify this way"
        )
    if n_failed > max_failure_fraction * n_resamples:
        raise RuntimeError(
            f"{n_failed} of {n_resamples} {context} replicates failed, above the "
            f"tolerated fraction {max_failure_fraction:.2f}; treat the reported "
            "uncertainty as unreliable rather than widening the budget"
        )
    return np.asarray(rows, dtype=np.float64), n_failed


def bootstrap_curves(
    evaluate_resample,
    n_rows,
    n_points,
    /,
    *,
    n_resamples,
    level,
    rng,
    max_failure_fraction,
):
    """Build pointwise bands by nonparametric resampling of observation rows.

    Rows are drawn with replacement, which is the right unit for both point
    and interval observations: an interval row is one observation whose
    endpoints must travel together.

    Parameters
    ----------
    evaluate_resample : callable
        Called as ``evaluate_resample(indices)`` with an integer index array
        of length *n_rows*, returning a length-*n_points* array of curve
        values, or ``None`` to mark the replicate as failed.
    n_rows : int
        Number of observation rows available to resample.
    n_points : int
        Number of abscissae each replicate evaluates.
    n_resamples : int
        Already-validated positive replicate count.
    level : float
        Already-validated two-sided pointwise coverage in ``(0, 1)``.
    rng : numpy.random.Generator
        Source of resampling indices.
    max_failure_fraction : float
        Largest tolerated share of failed replicates.

    Returns
    -------
    dict
        Keys ``lower``, ``upper``, ``n_resamples``, ``n_failed``, and
        ``level``.
    """
    rows = n_rows
    width = n_points
    if rows < 1:
        raise ValueError("bootstrap requires at least one observation row")
    if width < 1:
        raise ValueError("bootstrap requires at least one evaluation point")

    def make_replicate():
        return evaluate_resample(rng.integers(0, rows, size=rows))

    curves, n_failed = _collect_replicates(
        make_replicate,
        n_resamples,
        width,
        "bootstrap",
        max_failure_fraction=max_failure_fraction,
    )
    lower, upper = percentile_bands(curves, level)
    return {
        "lower": lower,
        "upper": upper,
        "n_resamples": n_resamples,
        "n_failed": n_failed,
        "level": level,
    }


def simulated_statistics(
    simulate_statistic,
    /,
    *,
    n_resamples,
    max_failure_fraction,
):
    """Collect a null distribution of scalar statistics by simulation.

    Parameters
    ----------
    simulate_statistic : callable
        Zero-argument callable returning one statistic drawn under the
        null, or ``None`` to mark the replicate as failed.
    n_resamples : int
        Already-validated positive replicate count.
    max_failure_fraction : float
        Largest tolerated share of failed replicates.

    Returns
    -------
    tuple
        ``(statistics, n_failed)`` where ``statistics`` is a one-dimensional
        float64 array of successful draws.
    """

    def make_replicate():
        value = simulate_statistic()
        return None if value is None else np.asarray([value], dtype=np.float64)

    rows, n_failed = _collect_replicates(
        make_replicate,
        n_resamples,
        1,
        "parametric bootstrap",
        max_failure_fraction=max_failure_fraction,
    )
    return rows.reshape(-1), n_failed
