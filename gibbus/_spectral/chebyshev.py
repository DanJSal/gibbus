"""Small cached helpers for Chebyshev interpolation during spectral builds.

The spectral constructors repeatedly interpolate values sampled at
Chebyshev--Lobatto nodes.  ``numpy.polynomial.chebyshev.chebfit`` solves a
small dense least-squares problem even when the sampling matrix is exactly the
square interpolation matrix.  Here the corresponding cosine transform is
cached once per degree and then applied by a matrix-vector product.
"""

import math
from functools import lru_cache

import numpy as np

_CACHE_MAXSIZE = 64
"""Maximum number of distinct spectral degrees/node counts cached per helper."""


@lru_cache(maxsize=_CACHE_MAXSIZE)
def lobatto_nodes(degree):
    """Return descending Chebyshev--Lobatto nodes for one polynomial degree.

    Parameters
    ----------
    degree : int
        Polynomial degree, at least one.

    Returns
    -------
    numpy.ndarray
        Read-only nodes ``cos(pi*j/degree)``, ``j=0,...,degree``.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1")
    nodes = np.cos(np.pi * np.arange(degree + 1, dtype=np.float64) / degree)
    nodes.setflags(write=False)
    return nodes


@lru_cache(maxsize=_CACHE_MAXSIZE)
def midpoint_nodes(count):
    """Return Chebyshev midpoint nodes used for interlaced validation.

    Parameters
    ----------
    count : int
        Number of nodes.

    Returns
    -------
    numpy.ndarray
        Read-only nodes ``cos(pi*(j + 1/2)/count)``.
    """
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    nodes = np.cos(np.pi * (np.arange(count, dtype=np.float64) + 0.5) / count)
    nodes.setflags(write=False)
    return nodes


@lru_cache(maxsize=_CACHE_MAXSIZE)
def _lobatto_transform(degree):
    """Return the value-to-Chebyshev coefficient transform for one degree.

    Parameters
    ----------
    degree : int
        Polynomial degree, at least one.

    Returns
    -------
    numpy.ndarray
        Read-only square transform matrix.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be >= 1")
    j = np.arange(degree + 1, dtype=np.float64)
    k = j[:, None]
    transform = np.cos(np.pi * k * j / degree)
    weights = np.full(degree + 1, 2.0 / degree, dtype=np.float64)
    weights[0] = 1.0 / degree
    weights[-1] = 1.0 / degree
    transform *= weights
    transform[0] *= 0.5
    transform[-1] *= 0.5
    transform.setflags(write=False)
    return transform


def lobatto_coefficients(values, /):
    """Interpolate Lobatto-sampled values as Chebyshev coefficients.

    Parameters
    ----------
    values : array_like, shape (degree + 1,)
        Function values ordered consistently with :func:`lobatto_nodes`.

    Returns
    -------
    numpy.ndarray
        Chebyshev coefficients, lowest order first.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a one-dimensional array of length >= 2")
    return _lobatto_transform(values.size - 1) @ values


@lru_cache(maxsize=_CACHE_MAXSIZE)
def chebyshev_bernstein_matrix(degree):
    """Return the Chebyshev-to-Bernstein basis change, correctly rounded.

    Row ``j`` maps Chebyshev coefficients on ``[-1, 1]`` to the ``j``-th
    degree-``degree`` Bernstein coefficient on ``[0, 1]`` (``u = 2 t - 1``):

        b_j = sum_k c_k sum_i (-1)^(k-i) C(2k, 2i) C(n-k, j-i) / C(n, j)

    The inner sums are exact integers, so every entry is the correctly
    rounded double of its exact rational value.  Converting through the
    power basis instead loses all accuracy by degree 24.

    Parameters
    ----------
    degree : int
        Polynomial degree ``n``.

    Returns
    -------
    numpy.ndarray, shape (n + 1, n + 1)
        Read-only matrix ``M`` with ``b = M c``.
    """
    n = int(degree)
    if n < 0:
        raise ValueError("degree must be >= 0")
    out = np.empty((n + 1, n + 1), dtype=np.float64)
    for j in range(n + 1):
        denominator = math.comb(n, j)
        for k in range(n + 1):
            numerator = sum(
                (-1) ** (k - i) * math.comb(2 * k, 2 * i) * math.comb(n - k, j - i)
                for i in range(max(0, j + k - n), min(j, k) + 1)
            )
            out[j, k] = numerator / denominator
    out.setflags(write=False)
    return out
