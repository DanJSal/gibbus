"""Information measures built on the vectorized expectation engine.

Potentials are vectorized evaluators, so every measure integrates whole
node arrays per refinement round (``expect_vectorized``).
"""

from __future__ import annotations

import numpy as np

from .._defaults import EXPECT_MAX_RELATIVE_ERROR
from .expectation import expect_vectorized


def entropy(potential, support, /, *, points=None):
    """Return differential entropy ``E[-log f(X)]``.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    support : array_like, shape (2,)
        Integration support.
    points : sequence of float or None, optional
        Quadrature breakpoints.
    """
    return expect_vectorized(potential, support, lambda x: potential(x, 0), points=points)


def cross_entropy(potential, support, other_potential, other_support, /, *, points=None):
    """Return ``E_self[-log f_other(X)]``.

    Parameters
    ----------
    potential : callable
        Self normalized negative-log density evaluator.
    support : array_like, shape (2,)
        Self integration support.
    other_potential : callable
        Other distribution's negative-log density evaluator.
    other_support : array_like, shape (2,)
        Other distribution support.
    points : sequence of float or None, optional
        Quadrature breakpoints.
    """
    lo, hi = map(float, support)
    olo, ohi = map(float, other_support)
    if lo < olo or hi > ohi:
        return np.inf
    return expect_vectorized(
        potential, support, lambda x: other_potential(x, 0), points=points
    )


def kl_divergence(potential, support, other_potential, other_support, /, *, points=None):
    """Return KL divergence ``D_KL(self || other)``.

    Parameters
    ----------
    potential : callable
        Self normalized negative-log density evaluator.
    support : array_like, shape (2,)
        Self support.
    other_potential : callable
        Other negative-log density evaluator.
    other_support : array_like, shape (2,)
        Other support.
    points : sequence of float or None, optional
        Quadrature breakpoints.
    """
    lo, hi = map(float, support)
    olo, ohi = map(float, other_support)
    if lo < olo or hi > ohi:
        return np.inf
    value = expect_vectorized(
        potential, support,
        lambda x: other_potential(x, 0) - potential(x, 0),
        points=points,
    )
    # Quadrature roundoff can produce a tiny negative value for nearly equal fits.
    if value < 0.0 and abs(value) <= EXPECT_MAX_RELATIVE_ERROR:
        value = 0.0
    return float(value)
