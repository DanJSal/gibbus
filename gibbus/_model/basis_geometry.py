"""Fixed nonlinear geometry of the basis family.

The basis lives in the canonical coordinate ``z = (x - c) / h`` of
:mod:`gibbus._model.coords`.  Every other piece of geometry is a family
constant in ``z``:

* Gaussian gates at ``z = -1`` and ``z = +1`` on each infinite side, with
  ``G_L = Phi(s (z + 1))``, ``G_U = Phi(s (1 - z))`` and the signed forms
  ``t_L = erf(s (1 + z) / sqrt(2))``, ``t_U = erf(s (1 - z) / sqrt(2))``;
* the soft power ``A**p`` with ``A = (1 + z**m)**(1/m) - 1``;
* the exponentials ``exp(lambda (z - 1))`` (upper) and ``exp(-lambda (z + 1))``
  (lower), each centered at its own gate.

Only the exponential rates depend on the request.  A rate ``r`` in user units
becomes ``lambda = r h`` in ``z``.  The geometry is fixed before any
coefficient fit and never changes during one.
"""

from dataclasses import dataclass

from .._defaults import BASIS_GATE_SHARPNESS, BASIS_SOFT_POWER_ROOT
from .basis_spec import _EXPONENTIAL
from .coords import _FitCoordinate


@dataclass(frozen=True)
class _BasisGeometry:
    """Resolved fixed geometry of one compiled basis.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Canonical coordinate supplying ``c`` and ``h``.
    lower_rate, upper_rate : float or None
        Canonical exponential rates ``lambda = r h`` of active exponential
        tails, or ``None``.
    """

    coordinate: _FitCoordinate
    lower_rate: float | None
    upper_rate: float | None

    gate_sharpness = BASIS_GATE_SHARPNESS
    """Gate sharpness ``s`` (a family constant)."""

    soft_power_root = BASIS_SOFT_POWER_ROOT
    """Soft-power root order ``m`` (a family constant)."""

    @property
    def center(self):
        """Basis origin ``c`` in user units."""
        return self.coordinate.center

    @property
    def scale(self):
        """Canonical coordinate scale ``h`` in user units."""
        return self.coordinate.scale

    @property
    def has_lower_gate(self):
        """Whether the lower gate ``t_L`` exists (the lower side is infinite)."""
        return not self.coordinate.physical_support[0] > -float("inf")

    @property
    def has_upper_gate(self):
        """Whether the upper gate ``t_U`` exists (the upper side is infinite)."""
        return not self.coordinate.physical_support[1] < float("inf")

    def to_internal(self, x, /):
        """Map user coordinates to ``z = (x - c) / h``.

        Parameters
        ----------
        x : array_like or float
            User-coordinate values.
        """
        return self.coordinate.to_canonical(x)


def _resolve_basis_geometry(request, coordinate, /):
    """Resolve the fixed geometry of one compiled basis.

    Parameters
    ----------
    request : _BasisRequest
        Canonical basis request.
    coordinate : _FitCoordinate
        Canonical coordinate.

    Returns
    -------
    _BasisGeometry
        Resolved geometry.
    """
    scale = coordinate.scale

    def rate(tail):
        if tail is None or tail.kind != _EXPONENTIAL:
            return None
        return float(tail.value) * scale

    return _BasisGeometry(
        coordinate=coordinate,
        lower_rate=rate(request.lower_tail),
        upper_rate=rate(request.upper_tail),
    )
