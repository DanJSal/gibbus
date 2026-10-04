"""Affine natural-coordinate layout of convex candidate potentials.

The natural parameters are ``theta = (gamma, c_0, ..., c_d, a_L, a_U)``: the
potential is

    q(z) = gamma z + sum_k c_k z^(k+2) / ((k+1)(k+2)) - a_L log(z - L) - a_U log(U - z)

so ``q'' = sum_k c_k z^k`` is the ordinary curvature polynomial (ascending
power order) and the finite-endpoint logarithmic amplitudes are direct
nonnegative coordinates.  The potential is linear in ``theta``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyder, polyval

_REAL_LINE = "real_line"
_LOWER_HALF_LINE = "lower_half_line"
_UPPER_HALF_LINE = "upper_half_line"
_BOUNDED = "bounded"


@dataclass(frozen=True)
class _NaturalLayout:
    """Fixed affine natural-parameter layout for one support and degree.

    Parameters
    ----------
    support_kind : str
        Canonical support geometry.
    support_lower, support_upper : float
        Support endpoints.
    requested_poly_degree : int
        Requested maximum polynomial degree of the potential.
    effective_poly_degree : int
        Highest polynomial degree represented by this layout.
    curvature_degree : int
        Highest represented degree of the ordinary curvature polynomial.
    gamma_index : int
        Index of the genuine linear potential coefficient.
    curvature_slice : slice
        Slice containing ``c_0, ..., c_d``.
    lower_a_index, upper_a_index : int or None
        Direct boundary-log amplitude indices when those bases are enabled.
    n_params : int
        Total natural-parameter count.
    """

    support_kind: str
    support_lower: float
    support_upper: float
    requested_poly_degree: int
    effective_poly_degree: int
    curvature_degree: int
    gamma_index: int
    curvature_slice: slice
    lower_a_index: int | None
    upper_a_index: int | None
    n_params: int

    @property
    def support(self):
        """Return the support endpoints as a tuple."""
        return self.support_lower, self.support_upper

    @property
    def lower_boundary_enabled(self):
        """Return whether the lower logarithmic boundary basis is present."""
        return self.lower_a_index is not None

    @property
    def upper_boundary_enabled(self):
        """Return whether the upper logarithmic boundary basis is present."""
        return self.upper_a_index is not None

    def validate_params(self, params, /):
        """Validate and return one natural-parameter vector.

        Parameters
        ----------
        params : array_like
            Natural vector in layout order.

        Returns
        -------
        numpy.ndarray
            One-dimensional ``float64`` parameter vector.

        Raises
        ------
        ValueError
            If the vector has the wrong length, contains non-finite values, or
            contains a negative enabled boundary amplitude.
        """
        raw = np.asarray(params, dtype=np.float64).reshape(-1)
        return self._validate_canonical_params(raw)

    def _validate_canonical_params(self, raw, /):
        """Check feasibility without reformatting an owned/canonical vector.

        Parameters
        ----------
        raw : numpy.ndarray, dtype float64
            One-dimensional natural vector in this layout's ordering.
        """
        if raw.shape != (self.n_params,):
            raise ValueError(
                f"expected {self.n_params} natural parameters, got {raw.size}"
            )
        if not np.all(np.isfinite(raw)):
            raise ValueError("natural parameters must be finite")
        for index in (self.lower_a_index, self.upper_a_index):
            if index is not None and raw[index] < 0.0:
                raise ValueError("boundary amplitudes must be >= 0")
        return raw

    def unpack(self, params, /):
        """Return the affine sectors of one natural-parameter vector.

        Parameters
        ----------
        params : array_like
            Natural vector in layout order.

        Returns
        -------
        tuple
            ``(gamma, curvature_coefficients, boundary_amplitudes)`` where the
            boundary array is canonical ``[a_L, a_U]`` and disabled entries
            are ``NaN``.
        """
        return self._unpack_canonical(self.validate_params(params))

    def _unpack_canonical(self, raw, /):
        """Extract affine sectors from an already validated natural vector.

        Parameters
        ----------
        raw : numpy.ndarray, shape (n_params,), dtype float64
            Natural vector whose layout feasibility checks have already passed.
        """
        amplitudes = np.full(2, np.nan, dtype=np.float64)
        if self.lower_a_index is not None:
            amplitudes[0] = raw[self.lower_a_index]
        if self.upper_a_index is not None:
            amplitudes[1] = raw[self.upper_a_index]
        return (
            float(raw[self.gamma_index]),
            raw[self.curvature_slice].copy(),
            amplitudes,
        )

    def pack(self, gamma, curvature_coefficients, boundary_amplitudes=None, /):
        """Pack affine model sectors into one validated natural vector.

        Parameters
        ----------
        gamma : float
            Genuine linear potential coefficient.
        curvature_coefficients : array_like
            Ascending coefficients ``c_0, ..., c_d`` of the ordinary
            curvature polynomial.
        boundary_amplitudes : array_like or None, optional
            Canonical ``[a_L, a_U]`` amplitudes.  Disabled entries are ignored;
            enabled entries must be finite and nonnegative.

        Returns
        -------
        numpy.ndarray
            Natural vector in layout order.
        """
        coeffs = np.asarray(curvature_coefficients, dtype=np.float64).reshape(-1)
        expected = self.curvature_degree + 1
        if coeffs.size != expected:
            raise ValueError(
                f"expected {expected} curvature coefficients, got {coeffs.size}"
            )
        amps = (
            np.full(2, np.nan, dtype=np.float64)
            if boundary_amplitudes is None
            else np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
        )
        if amps.size != 2:
            raise ValueError("boundary_amplitudes must have length 2")

        raw = np.empty(self.n_params, dtype=np.float64)
        raw[self.gamma_index] = float(gamma)
        raw[self.curvature_slice] = coeffs
        if self.lower_a_index is not None:
            raw[self.lower_a_index] = amps[0]
        if self.upper_a_index is not None:
            raw[self.upper_a_index] = amps[1]
        return self._validate_canonical_params(raw)

    def build_candidate(self, params, /):
        """Build the affine polynomial and boundary representation.

        Parameters
        ----------
        params : array_like
            Natural vector in layout order.

        Returns
        -------
        _NaturalCandidate
            Candidate potential with exact ordinary polynomial derivatives.
        """
        raw = self.validate_params(params)
        gamma, curvature, amplitudes = self._unpack_canonical(raw)
        q_poly = np.zeros(self.effective_poly_degree + 1, dtype=np.float64)
        q_poly[1] = gamma
        orders = np.arange(curvature.size, dtype=np.float64)
        q_poly[2:] = curvature / ((orders + 1.0) * (orders + 2.0))
        q_d1 = np.asarray(polyder(q_poly, 1), dtype=np.float64)
        q_d2 = np.asarray(polyder(q_poly, 2), dtype=np.float64)
        return _NaturalCandidate(
            layout=self,
            params=raw.copy(),
            q_poly=q_poly,
            q_d1=q_d1,
            q_d2=q_d2,
            boundary_amplitudes=amplitudes,
        )


@dataclass(frozen=True)
class _NaturalCandidate:
    """Algebraic natural-coordinate candidate before normalization.

    Parameters
    ----------
    layout : _NaturalLayout
        Fixed natural parameter layout.
    params : numpy.ndarray
        Validated natural parameter vector.
    q_poly, q_d1, q_d2 : numpy.ndarray
        Ascending polynomial coefficients of the ordinary potential and its
        first two derivatives.
    boundary_amplitudes : numpy.ndarray
        Canonical lower/upper logarithmic amplitudes; disabled sides are NaN.
    """

    layout: _NaturalLayout
    params: np.ndarray
    q_poly: np.ndarray
    q_d1: np.ndarray
    q_d2: np.ndarray
    boundary_amplitudes: np.ndarray

    def q(self, values, /):
        """Evaluate the full potential up to its omitted additive constant.

        Parameters
        ----------
        values : array_like
            Canonical-coordinate evaluation points.

        Returns
        -------
        numpy.ndarray
            Full potential values.
        """
        z = np.asarray(values, dtype=np.float64)
        out = np.asarray(polyval(z, self.q_poly), dtype=np.float64)
        lower, upper = self.layout.support
        a_lower, a_upper = self.boundary_amplitudes
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isfinite(a_lower) and a_lower > 0.0:
                out = out - a_lower * np.log(z - lower)
            if np.isfinite(a_upper) and a_upper > 0.0:
                out = out - a_upper * np.log(upper - z)
        return out

    def q_d1_full(self, values, /):
        """Evaluate the full first derivative of the potential.

        Parameters
        ----------
        values : array_like
            Canonical-coordinate evaluation points.

        Returns
        -------
        numpy.ndarray
            Full first-derivative values.
        """
        z = np.asarray(values, dtype=np.float64)
        out = np.asarray(polyval(z, self.q_d1), dtype=np.float64)
        lower, upper = self.layout.support
        a_lower, a_upper = self.boundary_amplitudes
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isfinite(a_lower) and a_lower > 0.0:
                out = out - a_lower / (z - lower)
            if np.isfinite(a_upper) and a_upper > 0.0:
                out = out + a_upper / (upper - z)
        return out

    def q_d2_full(self, values, /):
        """Evaluate the full curvature including finite-boundary terms.

        Parameters
        ----------
        values : array_like
            Canonical-coordinate evaluation points.

        Returns
        -------
        numpy.ndarray
            Full curvature values.
        """
        z = np.asarray(values, dtype=np.float64)
        out = np.asarray(polyval(z, self.q_d2), dtype=np.float64)
        lower, upper = self.layout.support
        a_lower, a_upper = self.boundary_amplitudes
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isfinite(a_lower) and a_lower > 0.0:
                out = out + a_lower / (z - lower) ** 2
            if np.isfinite(a_upper) and a_upper > 0.0:
                out = out + a_upper / (upper - z) ** 2
        return out

    def q_d3_full(self, values, /):
        """Evaluate the derivative of the full curvature.

        Parameters
        ----------
        values : array_like
            Canonical-coordinate evaluation points.

        Returns
        -------
        numpy.ndarray
            Full third-derivative values.
        """
        z = np.asarray(values, dtype=np.float64)
        out = np.asarray(polyval(z, polyder(self.q_d2)), dtype=np.float64)
        lower, upper = self.layout.support
        a_lower, a_upper = self.boundary_amplitudes
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isfinite(a_lower) and a_lower > 0.0:
                out = out - 2.0 * a_lower / (z - lower) ** 3
            if np.isfinite(a_upper) and a_upper > 0.0:
                out = out + 2.0 * a_upper / (upper - z) ** 3
        return out


def _natural_layout(
    support,
    poly_degree,
    lower_boundary_enabled=False,
    upper_boundary_enabled=False,
    /,
):
    """Return the affine natural layout for one support geometry.

    Parameters
    ----------
    support : array_like, shape (2,)
        Canonical support endpoints.
    poly_degree : int
        Requested maximum polynomial degree of the potential.  On the real
        line, an odd request canonicalizes to the preceding even degree.
    lower_boundary_enabled, upper_boundary_enabled : bool, optional
        Whether the corresponding finite-endpoint logarithmic basis belongs
        to the model.

    Returns
    -------
    _NaturalLayout
        Fixed natural-coordinate layout.

    Raises
    ------
    ValueError
        If the support or degree is invalid or a boundary basis is requested
        at an infinite endpoint.
    """
    bounds = np.asarray(support, dtype=np.float64).reshape(-1)
    if bounds.size != 2 or np.any(np.isnan(bounds)):
        raise ValueError("support must contain two non-NaN endpoints")
    lower, upper = map(float, bounds)
    if np.isposinf(lower) or np.isneginf(upper):
        raise ValueError("support infinities must point outward")
    if not lower < upper:
        raise ValueError("support must contain two increasing non-NaN endpoints")

    degree = int(poly_degree)
    if degree < 2:
        raise ValueError("poly_degree must be >= 2")

    if np.isneginf(lower) and np.isposinf(upper):
        support_kind = _REAL_LINE
        effective_degree = degree - (degree % 2)
    elif np.isfinite(lower) and np.isposinf(upper):
        support_kind = _LOWER_HALF_LINE
        effective_degree = degree
    elif np.isneginf(lower) and np.isfinite(upper):
        support_kind = _UPPER_HALF_LINE
        effective_degree = degree
    elif np.isfinite(lower) and np.isfinite(upper):
        support_kind = _BOUNDED
        effective_degree = degree
    else:
        raise ValueError("unsupported support geometry")

    lower_enabled = bool(lower_boundary_enabled)
    upper_enabled = bool(upper_boundary_enabled)
    if lower_enabled and not np.isfinite(lower):
        raise ValueError("lower boundary basis requires a finite lower endpoint")
    if upper_enabled and not np.isfinite(upper):
        raise ValueError("upper boundary basis requires a finite upper endpoint")

    curvature_degree = effective_degree - 2
    gamma_index = 0
    curvature_slice = slice(1, curvature_degree + 2)
    next_index = curvature_slice.stop
    lower_index = next_index if lower_enabled else None
    if lower_enabled:
        next_index += 1
    upper_index = next_index if upper_enabled else None
    if upper_enabled:
        next_index += 1

    return _NaturalLayout(
        support_kind=support_kind,
        support_lower=lower,
        support_upper=upper,
        requested_poly_degree=degree,
        effective_poly_degree=effective_degree,
        curvature_degree=curvature_degree,
        gamma_index=gamma_index,
        curvature_slice=curvature_slice,
        lower_a_index=lower_index,
        upper_a_index=upper_index,
        n_params=next_index,
    )
