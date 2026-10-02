"""Model specification: one fitting coordinate bound to one natural layout.

A model specification binds a fixed
:class:`gibbus._model.coords._FitCoordinate` to the affine natural-parameter
layout of :mod:`gibbus._model.natural`, translating the user's physical
boundary-term flags into canonical-coordinate flags (reflection of an upper
half-line swaps the sides).  It carries no live optimizer state and performs
no model quadrature.

The module also defines the first-partial descriptors of the potential.  The
potential ``q = theta . t(z)`` is linear in the natural parameters, so every
partial is a fixed polynomial or a fixed zero-offset endpoint logarithm.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyval

from .coords import (
    _BOUNDED,
    _LOWER_HALF_LINE,
    _UPPER_HALF_LINE,
    _FitCoordinate,
)
from .natural import _natural_layout, _NaturalLayout

_POLY = "poly"
_LOGDIST = "logdist"
_LOWER = "lower"
_UPPER = "upper"


@dataclass(frozen=True)
class _PotentialPartial:
    """One first parameter partial of the candidate potential.

    Parameters
    ----------
    kind : str
        ``"poly"`` for an ascending power-basis polynomial or ``"logdist"``
        for a fixed zero-offset endpoint logarithm.
    coefficients : numpy.ndarray or None
        Polynomial coefficients for ``kind == "poly"``.
    boundary_side : str or None
        ``"lower"`` or ``"upper"`` for a logarithmic boundary partial.
    """

    kind: str
    coefficients: np.ndarray | None = None
    boundary_side: str | None = None

    def evaluate(self, values, support, /):
        """Evaluate this partial on canonical-coordinate values.

        Parameters
        ----------
        values : array_like
            Canonical-coordinate evaluation points.
        support : tuple of (float, float)
            Canonical support.

        Returns
        -------
        numpy.ndarray
            Partial-potential values with the same shape as ``values``.
        """
        z = np.asarray(values, dtype=np.float64)
        if self.kind == _POLY:
            return np.asarray(polyval(z, self.coefficients), dtype=np.float64)
        lower, upper = map(float, support)
        with np.errstate(divide="ignore", invalid="ignore"):
            if self.boundary_side == _LOWER:
                return -np.log(z - lower)
            if self.boundary_side == _UPPER:
                return -np.log(upper - z)
        raise ValueError("unknown potential-partial descriptor")


@dataclass(frozen=True)
class _ModelSpec:
    """Fixed coordinate and natural-parameter layout for one fit.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed user-to-canonical affine fitting coordinate.
    layout : _NaturalLayout
        Natural-parameter layout on the canonical support.
    """

    coordinate: _FitCoordinate
    layout: _NaturalLayout

    @property
    def support(self):
        """Canonical support used by model numerics."""
        return self.coordinate.canonical_support

    @property
    def requested_poly_degree(self):
        """Requested maximum polynomial degree of the potential."""
        return int(self.layout.requested_poly_degree)

    @property
    def effective_poly_degree(self):
        """Highest polynomial degree the layout represents."""
        return int(self.layout.effective_poly_degree)

    @property
    def n_params(self):
        """Complete natural-parameter count."""
        return int(self.layout.n_params)

    @property
    def canonical_lower_a_index(self):
        """Parameter index of the canonical lower-endpoint amplitude, or None."""
        return self.layout.lower_a_index

    @property
    def canonical_upper_a_index(self):
        """Parameter index of the canonical upper-endpoint amplitude, or None."""
        return self.layout.upper_a_index

    @property
    def physical_lower_a_index(self):
        """Parameter index of the user-support lower amplitude, or None."""
        if self.coordinate.direction > 0.0:
            return self.layout.lower_a_index
        return self.layout.upper_a_index

    @property
    def physical_upper_a_index(self):
        """Parameter index of the user-support upper amplitude, or None."""
        if self.coordinate.direction > 0.0:
            return self.layout.upper_a_index
        return self.layout.lower_a_index


def _physical_finite_sides(coordinate, /):
    """Return whether the original support has finite lower/upper endpoints.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed fitting coordinate whose support kind encodes endpoint geometry.

    Returns
    -------
    tuple of bool
        ``(lower_finite, upper_finite)`` in user-coordinate orientation.
    """
    kind = coordinate.support_kind
    return (
        kind in (_LOWER_HALF_LINE, _BOUNDED),
        kind in (_UPPER_HALF_LINE, _BOUNDED),
    )


def _build_model_spec(
    coordinate,
    poly_degree,
    allow_lower_boundary=False,
    allow_upper_boundary=False,
    /,
):
    """Construct the complete model specification.

    Boundary flags refer to the original physical/user-coordinate lower and
    upper endpoints.  Reflection of an upper half-line maps its physical upper
    endpoint to the canonical lower endpoint automatically.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed fitting coordinate.
    poly_degree : int
        Requested maximum polynomial degree of ``q``.
    allow_lower_boundary, allow_upper_boundary : bool
        Whether the corresponding finite physical endpoint carries a direct
        nonnegative zero-offset logarithmic amplitude.

    Returns
    -------
    _ModelSpec
        Complete model layout.

    Raises
    ------
    TypeError
        If ``coordinate`` is not a ``_FitCoordinate``.
    ValueError
        If a boundary term is requested on an infinite physical endpoint.
    """
    if not isinstance(coordinate, _FitCoordinate):
        raise TypeError("coordinate must be a _FitCoordinate")

    lower_finite, upper_finite = _physical_finite_sides(coordinate)
    if allow_lower_boundary and not lower_finite:
        raise ValueError("lower boundary term requires a finite lower endpoint")
    if allow_upper_boundary and not upper_finite:
        raise ValueError("upper boundary term requires a finite upper endpoint")

    if coordinate.direction > 0.0:
        canonical_lower, canonical_upper = allow_lower_boundary, allow_upper_boundary
    else:
        canonical_lower, canonical_upper = allow_upper_boundary, allow_lower_boundary
    layout = _natural_layout(
        coordinate.canonical_support,
        int(poly_degree),
        bool(canonical_lower),
        bool(canonical_upper),
    )
    return _ModelSpec(coordinate=coordinate, layout=layout)
