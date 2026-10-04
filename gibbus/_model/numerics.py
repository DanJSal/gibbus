"""Shared model-state numerical helpers independent of observation type.

These helpers belong to the retained numerical core: boundary descriptors for
quadrature, shifted quadrature-kernel preparation, and mode-aware local
breakpoints.  They are deliberately separated from both the
parameterization and point/interval objective evaluation.
"""

import numpy as np


def _terms_for_quad(support, boundary_amplitudes, /):
    """Build active zero-offset boundary descriptors for quadrature.

    Parameters
    ----------
    support : array_like, shape (2,)
        Canonical support ``[L, U]``.
    boundary_amplitudes : numpy.ndarray, shape (2,), dtype float64
        Canonical lower/upper amplitudes ``[aL, aU]``. ``NaN`` or zero
        denotes an inactive side.

    Returns
    -------
    numpy.ndarray, shape (n_terms, 3), dtype float64
        Rows ``[endpoint, sign, amplitude]`` for active terms, where
        ``sign=-1`` denotes the lower distance ``x-L`` and ``sign=+1``
        denotes the upper distance ``U-x``.
    """
    L, U = map(float, support)
    aL, aU = boundary_amplitudes

    terms = []
    if np.isfinite(L) and np.isfinite(aL) and aL > 0.0:
        terms.append((L, -1.0, aL))
    if np.isfinite(U) and np.isfinite(aU) and aU > 0.0:
        terms.append((U, +1.0, aU))
    return (
        np.asarray(terms, dtype=np.float64)
        if terms
        else np.empty((0, 3), dtype=np.float64)
    )


def _complete_boundary_terms(
    support, boundary_amplitudes, lower_enabled, upper_enabled, /
):
    """Build boundary descriptors for every enabled canonical endpoint basis.

    Unlike :func:`_terms_for_quad`, this retains enabled terms whose current
    amplitude is zero.  A complete descriptor set can therefore be reused for
    normalization and generalized log-distance moments without rebuilding the
    low-level quadrature context.

    Parameters
    ----------
    support : array_like, shape (2,)
        Canonical support ``[L, U]``.
    boundary_amplitudes : numpy.ndarray, shape (2,), dtype float64
        Current canonical lower/upper amplitudes.
    lower_enabled, upper_enabled : bool
        Whether the corresponding endpoint-log basis belongs to the model.

    Returns
    -------
    terms : numpy.ndarray, shape (R, 3)
        Rows ``[endpoint, sign, amplitude]`` for every enabled boundary basis.
    side_to_term : dict
        Mapping from ``"lower"``/``"upper"`` to rows in *terms*.
    """
    L, U = map(float, support)
    aL, aU = boundary_amplitudes
    rows = []
    mapping = {}
    if bool(lower_enabled):
        mapping["lower"] = len(rows)
        rows.append((L, -1.0, aL))
    if bool(upper_enabled):
        mapping["upper"] = len(rows)
        rows.append((U, 1.0, aU))
    terms = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, 3), dtype=np.float64)
    )
    return terms, mapping


def _build_quad_kernel(q_poly, support, boundary_amplitudes, /):
    """Prepare the shifted polynomial and active boundary quadrature terms.

    Parameters
    ----------
    q_poly : array_like, shape (d+1,)
        Shifted potential polynomial coefficients.
    support : array_like, shape (2,)
        Canonical support.
    boundary_amplitudes : array_like, shape (2,)
        Canonical zero-offset boundary amplitudes.

    Returns
    -------
    tuple
        Copied polynomial and ``(n_terms, 3)`` boundary descriptor array.
    """
    qp = np.asarray(q_poly, dtype=np.float64).copy()
    return qp, _terms_for_quad(support, boundary_amplitudes)


_MODE_QUAD_SPAN_TRIGGER = 128.0
_MODE_QUAD_RADIUS = 8.0


def _mode_quad_points(window, mode, local_scale, support=None, core_window=None, /):
    """Return interior quadrature breakpoints around a narrow density mode.

    ``points=[mode]`` is insufficient when the integration window spans
    thousands of component widths: QUADPACK may sample neither side close
    enough to the mode and can miss an order-one peak entirely.  Bracketing
    the known mode by eight affine scales creates a finite local panel while
    leaving the tail panels adaptive.  The extra points are used by both the
    normalizer and moment/derivative quadratures so they share the same local
    resolution.

    Parameters
    ----------
    window : array_like, shape (2,)
        Finite integration window in fitting coordinates.
    mode : float
        Known potential minimizer within the integration window.
    local_scale : float
        Affine scale associated with the fitted component.
    support : tuple of (float, float) or None, optional
        Canonical model support.  When a finite support endpoint coincides
        with the integration-window endpoint, geometric interior panels are
        added to resolve integrable logarithmic boundary singularities.
    core_window : array_like, shape (2,), or None, optional
        Density-defined tail points before any data-range padding.  Supplying
        them prevents a very large padded outer panel from missing a narrow
        density-bearing edge.

    Returns
    -------
    list of float
        Interior breakpoints for adaptive quadrature.
    """
    lo, hi = map(float, window)
    m = float(mode)
    s = float(local_scale)
    vals = [m]
    span = hi - lo
    wide = np.isfinite(s) and s > 0.0 and span / s > _MODE_QUAD_SPAN_TRIGGER
    if wide:
        vals = [m - _MODE_QUAD_RADIUS * s, m, m + _MODE_QUAD_RADIUS * s]
        if core_window is not None:
            core = np.asarray(core_window, dtype=np.float64).reshape(2)
            vals.extend(float(v) for v in core if np.isfinite(v))

    if support is not None and np.isfinite(span) and span > 0.0:
        L, U = map(float, support)
        # Log-distance moments remain integrable at an enabled boundary but
        # vary on every logarithmic distance scale.  A handful of geometric
        # panels gives QUADPACK those scales explicitly without placing the
        # innermost point so close that it rounds back onto the endpoint.
        fractions = (2.0**-6, 2.0**-12, 2.0**-18, 2.0**-24, 2.0**-30)
        if np.isfinite(L) and lo == L:
            vals.extend(lo + span * f for f in fractions)
        if np.isfinite(U) and hi == U:
            vals.extend(hi - span * f for f in fractions)

    clean = sorted({float(v) for v in vals if np.isfinite(v) and lo < float(v) < hi})
    return clean
