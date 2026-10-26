# cython: language_level=3
"""Compactifying support maps shared by the spectral builders and evaluators.

Every map sends the compact coordinate ``z in [-1, 1]`` onto the support.
``kind`` 0 is affine (a finite support the density fills).  The others are
rational maps of a coordinate ``t in (-1, 1)`` onto the whole line or a
half-line, restricted to the window ``[t_lo, t_hi]`` that the support
occupies:

* 1, anchored at the lower end: ``x = L + s (1 + t) / (1 - t)``;
* 2, anchored at the upper end: ``x = U - s (1 - t) / (1 + t)``;
* 3, centered on the density: ``x = c + 2 s t / (1 - t^2)``.

A finite end far from the density thus clips the window instead of
stretching an affine map across it, so the body keeps its resolution
whatever the support length.  ``z`` is affine in ``t`` on the window and
equals ``t`` when the window is ``[-1, 1]``.  Finite ends are returned
exactly at ``z = +/- 1``.
"""

from libc.math cimport hypot, isfinite, isinf


cdef struct Map:
    int kind
    double L
    double U
    double center
    double scale
    double t_lo
    double t_hi
    bint full


cdef inline double _rational_t(int kind, double center, double scale, double x) noexcept nogil:
    """Rational coordinate ``t`` of a physical point (``+/- 1`` at infinity)."""
    cdef double y
    if isinf(x):
        return 1.0 if x > 0.0 else -1.0
    if kind == 1:
        y = (x - center) / scale
        return (y - 1.0) / (y + 1.0)
    if kind == 2:
        y = (center - x) / scale
        return (1.0 - y) / (1.0 + y)
    y = (x - center) / scale
    return y / (1.0 + hypot(1.0, y))


cdef inline void map_set(
    Map* mp, int kind, double L, double U, double center, double scale
) noexcept nogil:
    """Fill ``mp`` and its window for the given parameters."""
    mp.kind = kind
    mp.L = L
    mp.U = U
    mp.center = center
    mp.scale = scale
    if kind == 0:
        mp.t_lo = -1.0
        mp.t_hi = 1.0
    else:
        mp.t_lo = -1.0 if kind == 1 else _rational_t(kind, center, scale, L)
        mp.t_hi = 1.0 if kind == 2 else _rational_t(kind, center, scale, U)
    mp.full = mp.t_lo == -1.0 and mp.t_hi == 1.0


cdef inline double map_t(const Map* mp, double z) noexcept nogil:
    """Window coordinate ``t`` of a compact coordinate."""
    if mp.full:
        return z
    return mp.t_lo + 0.5 * (z + 1.0) * (mp.t_hi - mp.t_lo)


cdef inline double map_x(const Map* mp, double z) noexcept nogil:
    """Physical coordinate of a compact coordinate."""
    cdef double t
    if z <= -1.0 and isfinite(mp.L):
        return mp.L
    if z >= 1.0 and isfinite(mp.U):
        return mp.U
    if mp.kind == 0:
        return mp.center + mp.scale * z
    t = map_t(mp, z)
    if mp.kind == 1:
        return mp.center + mp.scale * (1.0 + t) / (1.0 - t)
    if mp.kind == 2:
        return mp.center - mp.scale * (1.0 - t) / (1.0 + t)
    return mp.center + (2.0 * mp.scale * t) / (1.0 - t * t)


cdef inline double map_jac(const Map* mp, double z) noexcept nogil:
    """``dx/dz`` of the map."""
    cdef double t, w, dt
    if mp.kind == 0:
        return mp.scale
    t = map_t(mp, z)
    dt = 1.0 if mp.full else 0.5 * (mp.t_hi - mp.t_lo)
    if mp.kind == 1:
        w = 1.0 - t
        return 2.0 * mp.scale / (w * w) * dt
    if mp.kind == 2:
        w = 1.0 + t
        return 2.0 * mp.scale / (w * w) * dt
    w = 1.0 - t * t
    return 2.0 * mp.scale * (1.0 + t * t) / (w * w) * dt


cdef inline double map_z(const Map* mp, double x) noexcept nogil:
    """Compact coordinate of a physical point inside the support."""
    cdef double t
    if mp.kind == 0:
        return (x - mp.center) / mp.scale
    t = _rational_t(mp.kind, mp.center, mp.scale, x)
    if mp.full:
        return t
    return 2.0 * (t - mp.t_lo) / (mp.t_hi - mp.t_lo) - 1.0
