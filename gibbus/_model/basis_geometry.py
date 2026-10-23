"""Fixed nonlinear geometry of a G3 basis.

G1 owner extended: ``_model/coords.py``.  The G1 ``_FitCoordinate`` is
reused unchanged; this module adds the fixed gate, soft-power and exponential
geometry on top of it.

The basis lives in the G1 robust fitting coordinate ``z = (x - c) / h``
(:mod:`gibbus._model.coords`): weighted median ``c`` and robust scale ``h``,
responsibility-weighted per mixture component, never reflected for G3.
This module resolves the remaining geometry once — tanh locations and
steepnesses, soft-power smoothings and exponential rates — from explicit
request values or representative data.  Geometry stays fixed during every
coefficient fit.

Under tanh gating the origin ``c`` is part of the represented family (for
example ``z**3 (t_L + t_U)`` is not affine invariant); the scale ``h`` only
rescales coefficients.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import (
    BASIS_SOFT_POWER_ETA,
    BASIS_TANH_LOWER_QUANTILE,
    BASIS_TANH_STEEPNESS_KAPPA,
    BASIS_TANH_UPPER_QUANTILE,
)
from .basis_spec import _EXPONENTIAL, _POLYNOMIAL, _SOFT_POWER
from .coords import _FitCoordinate


@dataclass(frozen=True)
class _GeometryPolicy:
    """Automatic-geometry constants resolved at the owning boundary.

    Parameters
    ----------
    lower_quantile, upper_quantile : float
        Data quantiles for automatic tanh locations.
    steepness_kappa : float
        Dimensionless automatic steepness ``alpha = kappa / h``.
    soft_power_eta : float
        Dimensionless automatic smoothing ``delta = (eta h)**2``.
    """

    lower_quantile: float = BASIS_TANH_LOWER_QUANTILE
    upper_quantile: float = BASIS_TANH_UPPER_QUANTILE
    steepness_kappa: float = BASIS_TANH_STEEPNESS_KAPPA
    soft_power_eta: float = BASIS_SOFT_POWER_ETA


@dataclass(frozen=True)
class _BasisGeometry:
    """Resolved fixed geometry, physical (user) units.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed non-reflected fitting coordinate supplying ``c`` and ``h``.
    lower_location, upper_location : float or None
        Tanh centers ``x_L``, ``x_U`` (``None`` when unavailable).
    lower_steepness, upper_steepness : float or None
        Tanh steepnesses ``alpha_L``, ``alpha_U``.
    lower_delta, upper_delta : float or None
        Soft-power smoothings for active noninteger soft-power tails.
    lower_rate, upper_rate : float or None
        Exponential rates for active exponential tails.
    """

    coordinate: _FitCoordinate
    lower_location: float | None
    upper_location: float | None
    lower_steepness: float | None
    upper_steepness: float | None
    lower_delta: float | None
    upper_delta: float | None
    lower_rate: float | None
    upper_rate: float | None

    @property
    def center(self):
        """Basis origin ``c`` in user units."""
        return self.coordinate.center

    @property
    def scale(self):
        """Internal coordinate scale ``h`` in user units."""
        return self.coordinate.scale

    def to_internal(self, x, /):
        """Map user coordinates to ``z = (x - c) / h``.

        Parameters
        ----------
        x : array_like or float
            User-coordinate values.
        """
        return self.coordinate.to_canonical(x)


def _weighted_quantile(values, weights, q, /):
    """Return an interpolated weighted quantile.

    Parameters
    ----------
    values : numpy.ndarray
        Canonical finite ``float64`` values, shape ``(R,)``.
    weights : numpy.ndarray or None
        Nonnegative weights, shape ``(R,)``, or ``None``.
    q : float
        Quantile level in ``[0, 1]``.
    """
    if weights is None:
        return float(np.quantile(values, q))
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = np.asarray(weights, dtype=np.float64)[order]
    keep = w > 0.0
    v, w = v[keep], w[keep]
    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def _skew_aware_gate_settings(values, /):
    """Automatic full-line gate quantiles and dimensionless steepnesses.

    Return ``None`` for small samples, nearly symmetric data, or unresolved
    quantile spacings.  The sample-size shrinkage and constants are the fixed
    result of the 2026-10-19 independent holdout geometry experiments.  The
    optimization of basis coefficients remains separate and convex.
    """
    n = int(values.size)
    if n <= 150:
        return None
    q10, q50, q90 = np.quantile(values, (0.10, 0.50, 0.90))
    left, right = float(q50 - q10), float(q90 - q50)
    if not (np.isfinite(left) and np.isfinite(right) and left > 0 and right > 0):
        return None
    asymmetry = float(np.log(right / left))
    strength = max(0.0, abs(asymmetry) - 0.12)
    if strength <= 0.0:
        return None
    shrink = min(1.0, max(0.0, (n - 150) / 550.0))
    shift = shrink * min(0.07, 0.22 * strength)
    slow = np.clip(2.0 * np.exp(-5.0 * strength), 0.4, 2.0)
    fast = np.clip(2.0 * np.exp(0.75 * strength), 2.0, 5.0)
    slow = 2.0 * (slow / 2.0) ** shrink
    fast = 2.0 * (fast / 2.0) ** shrink
    if asymmetry > 0.0:
        return 0.10, 0.90 - shift, float(fast), float(slow)
    return 0.10 + shift, 0.90, float(slow), float(fast)


def _resolve_basis_geometry(
    request,
    coordinate,
    points=None,
    weights=None,
    policy=None,
    /,
    *,
    adaptive_geometry=True,
):
    """Resolve all fixed geometry for one compiled basis.

    Parameters
    ----------
    request : _BasisRequest
        Canonical basis request.
    coordinate : _FitCoordinate
        Fixed fitting coordinate.
    points : numpy.ndarray or None
        Canonical representative data locations in user units, finite
        ``float64`` of shape ``(R,)``, or ``None`` when every tanh location
        needed is explicit.
    weights : numpy.ndarray or None
        Nonnegative weights of shape ``(R,)``, or ``None``.
    policy : _GeometryPolicy or None
        Automatic-geometry constants; ``None`` uses the package defaults.
    adaptive_geometry : bool
        Whether to adapt full-line gate geometry to independent unweighted
        point samples. Disabled for mixtures, censored and weighted fits.

    Returns
    -------
    _BasisGeometry
        Resolved geometry.

    Raises
    ------
    ValueError
        If automatic locations are requested without data, or the resolved
        tanh locations are not strictly ordered.
    """
    policy = _GeometryPolicy() if policy is None else policy
    has_data = points is not None and np.asarray(points).size > 0
    values = np.asarray(points, dtype=np.float64) if has_data else None
    lower_inf, upper_inf = request.lower_infinite, request.upper_infinite
    scale = coordinate.scale

    def auto_location(explicit, q, available):
        if not available:
            return None
        if explicit is not None:
            return explicit
        if not has_data:
            raise ValueError(
                "automatic tanh location requires data; specify it explicitly"
            )
        return _weighted_quantile(values, weights, q)

    q_lower, q_upper = policy.lower_quantile, policy.upper_quantile
    k_lower = k_upper = policy.steepness_kappa
    if adaptive_geometry and lower_inf and upper_inf and has_data and weights is None:
        adaptive = _skew_aware_gate_settings(values)
        if adaptive is not None:
            q_lower, q_upper, k_lower, k_upper = adaptive

    x_lower = auto_location(request.lower_location, q_lower, lower_inf)
    x_upper = auto_location(request.upper_location, q_upper, upper_inf)
    if x_lower is not None and x_upper is not None and not x_lower < x_upper:
        if request.lower_location is not None or request.upper_location is not None:
            raise ValueError(
                f"tanh locations must satisfy x_L < x_U (resolved {x_lower:g}, {x_upper:g})"
            )
        # Degenerate automatic quantiles (heavily tied data): spread about the tie.
        mid = 0.5 * (x_lower + x_upper)
        x_lower, x_upper = mid - scale, mid + scale

    # A pair of linear polynomial tails requires a curvature-compatible
    # window for gated bulk powers.  If the two gate decay rates differ,
    # (t_L + t_U) changes sign between the far ends: the quadratic window
    # then has opposing leading curvature signs and is forced onto an exact
    # zero-coefficient face.  The remaining degree-2 columns may have no
    # globally convex, outward-growing member.  Equal steepnesses ensure
    # t_L + t_U >= 0 when x_L < x_U and preserve the useful quadratic span.
    # Keep explicitly requested steepnesses unchanged; match an automatic
    # partner to a single explicit value, or use the common policy default.
    both_linear = all(
        tail is not None and tail.kind == _POLYNOMIAL and tail.value == 1.0
        for tail in (request.lower_tail, request.upper_tail)
    )
    if lower_inf and upper_inf and both_linear:
        if request.lower_steepness is None and request.upper_steepness is None:
            k_lower = k_upper = policy.steepness_kappa
        elif request.lower_steepness is None:
            k_lower = request.upper_steepness * scale
        elif request.upper_steepness is None:
            k_upper = request.lower_steepness * scale

    alpha_lower = (
        (
            request.lower_steepness
            if request.lower_steepness is not None
            else k_lower / scale
        )
        if lower_inf
        else None
    )
    alpha_upper = (
        (
            request.upper_steepness
            if request.upper_steepness is not None
            else k_upper / scale
        )
        if upper_inf
        else None
    )

    def tail_param(tail, kind):
        if tail is None or tail.kind != kind:
            return None
        if kind == _EXPONENTIAL:
            return tail.value
        return (
            tail.delta
            if tail.delta is not None
            else (policy.soft_power_eta * scale) ** 2
        )

    return _BasisGeometry(
        coordinate=coordinate,
        lower_location=None if x_lower is None else float(x_lower),
        upper_location=None if x_upper is None else float(x_upper),
        lower_steepness=None if alpha_lower is None else float(alpha_lower),
        upper_steepness=None if alpha_upper is None else float(alpha_upper),
        lower_delta=tail_param(request.lower_tail, _SOFT_POWER),
        upper_delta=tail_param(request.upper_tail, _SOFT_POWER),
        lower_rate=tail_param(request.lower_tail, _EXPONENTIAL),
        upper_rate=tail_param(request.upper_tail, _EXPONENTIAL),
    )
