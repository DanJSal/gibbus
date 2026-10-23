"""Selection policy and PSD helpers of the omitted-statistic score test.

G1 owner extended: ``_fit/degree.py``.  G1's polynomial power-moment
diagnostics are replaced by ``basis_selection._span_diagnostic`` (one score
test for degrees, spans and mixtures, G3-035/045); this module keeps the
immutable selection policy (``_DegreeSelectionConfig``) and the
positive-semidefinite projections the test uses.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _DegreeSelectionConfig:
    """Immutable omitted-statistic policy owned by one fit request.

    The API boundary creates this record once. Candidate fits, component probes,
    shared degree growth and boundary-removal refits consume the same record
    rather than resolving their own defaults.

    Parameters
    ----------
    probe_block_size : int
        Number of consecutive omitted power statistics examined at once.
    alpha : float
        Upper-tail score-test probability required before model capacity is
        increased.
    min_participation : float
        Minimum contribution-participation effective sample size for an
        omitted moment to enter the score test.
    covariance_rtol : float
        Relative eigenvalue tolerance used by rank-aware covariance solves.
    max_variance_inflation : float
        Maximum conservative inflation applied when empirical variability of
        an omitted statistic exceeds the current model's variability.
    """

    probe_block_size: int = 2
    alpha: float = 0.05
    min_participation: float = 8.0
    covariance_rtol: float = 1e-10
    max_variance_inflation: float = 1e6


def _psd_pinv(matrix, rtol, /):
    """Return a symmetric PSD pseudoinverse and numerical rank.

    Parameters
    ----------
    matrix : array_like
        Symmetric positive-semidefinite matrix up to numerical roundoff.
    rtol : float
        Relative eigenvalue rank tolerance.

    Returns
    -------
    pinv : numpy.ndarray
        Rank-aware symmetric pseudoinverse.
    rank : int
        Numerical rank.
    """
    a = _project_psd(matrix, rtol)
    if a.size == 0:
        return a.copy(), 0
    values, vectors = np.linalg.eigh(a)
    vmax = float(np.max(values)) if values.size else 0.0
    cutoff = max(float(rtol) * vmax, np.finfo(np.float64).eps)
    keep = values > cutoff
    if not np.any(keep):
        return np.zeros_like(a), 0
    inv = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    return inv, int(np.count_nonzero(keep))


def _project_psd(matrix, rtol, /):
    """Symmetrize a covariance matrix and clip roundoff-negative eigenvalues.

    Parameters
    ----------
    matrix : array_like
        Candidate symmetric covariance/information matrix.
    rtol : float
        Relative tolerance distinguishing roundoff-negative eigenvalues.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-semidefinite projection.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.size == 0:
        return a.reshape((0, 0)).copy()
    a = 0.5 * (a + a.T)
    values, vectors = np.linalg.eigh(a)
    vmax = max(float(np.max(np.abs(values))), 1.0)
    floor = -float(rtol) * vmax
    if np.min(values) < floor:
        # A covariance becoming materially indefinite indicates accumulated
        # quadrature/cancellation error; clipping is still safer than allowing
        # an invalid score metric to propagate.
        values = np.maximum(values, 0.0)
    else:
        values = np.maximum(values, 0.0)
    return (vectors * values) @ vectors.T
