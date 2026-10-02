"""Stable compiled mixture log-sum-exp and derivative reductions."""

import numpy as np

from ._mix_kernels import neg_log_mix_derivs_batch as _compiled_mix_derivs
from ._mix_kernels import neg_logsumexp_batch as _compiled_neg_logsumexp


def _neg_logsumexp_batch(ell0):
    """Return stable negative log-sum-exp values over the component axis.

    Parameters
    ----------
    ell0 : array_like, shape (K, R)
        Component log values for ``R`` evaluation points.
    """
    arr = np.ascontiguousarray(ell0, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("ell0 must have shape (K, R)")
    return np.asarray(_compiled_neg_logsumexp(arr), dtype=np.float64)


def _neg_log_mix_derivs_batch(ell_jets, max_order):
    """Compute derivatives of ``-log(sum_j exp(ell_j(x)))`` through one order.

    Parameters
    ----------
    ell_jets : array_like, shape (K, N+1, R)
        Component log-value derivative jets.
    max_order : int
        Highest derivative order to return.
    """
    arr = np.ascontiguousarray(ell_jets, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("ell_jets must have shape (K, N+1, R)")
    return np.asarray(_compiled_mix_derivs(arr, int(max_order)), dtype=np.float64)
