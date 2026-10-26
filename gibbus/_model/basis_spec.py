"""Public basis configuration and its one-time canonicalization.

The potential is ``q(z) = sum_j f_j(z) [a_j + b_j t_L(z) + c_j t_U(z)]`` over
named underlying functions of the canonical coordinate ``z`` (monomials,
finite-endpoint logarithms, soft powers and directional exponentials), with
the fixed Gaussian gates ``t_L`` and ``t_U`` of
:mod:`gibbus._model.basis_geometry`.  This module defines the user-facing
configuration records and converts one :class:`BasisSpec` into a canonical,
fully validated :class:`_BasisRequest`.  It performs no data analysis and no
linear algebra; see :mod:`gibbus._model.basis_geometry` and
:mod:`gibbus._model.basis`.

An integer ``SoftPower`` request is converted to the :class:`Polynomial` tail
of the same power.  The soft power exists to give non-integer tail orders
smoothly; at an integer order the monomial gives the same tail with a simpler
function.
"""

import math
import numbers
import warnings
from dataclasses import dataclass

from .._defaults import BASIS_DEFAULT_BULK_DEGREE

_AUTO = "auto"
_MINIMAL = "minimal"
_FULL = "full"
_AUTO_EXPANSION = "auto"

_POLYNOMIAL = "polynomial"
_SOFT_POWER = "soft_power"
_EXPONENTIAL = "exponential"


class GibbusBasisWarning(UserWarning):
    """Warning about a basis request that was reinterpreted or degraded."""


def _is_real_number(value, /):
    """Return whether ``value`` is a real, non-boolean number.

    Parameters
    ----------
    value : object
        Candidate value.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class Polynomial:
    """Integer polynomial tail: ``q ~ C |x|**power`` at that infinite end.

    Parameters
    ----------
    power : int or "auto", optional
        Integer tail power, at least 1; omitted or ``"auto"`` estimates it
        empirically at fit time (integer search 1 through 6).
    """

    power: int | str = _AUTO

    def __post_init__(self):
        """Validate the fields."""
        if self.power == _AUTO:
            return
        if not isinstance(self.power, numbers.Integral) or isinstance(self.power, bool):
            raise TypeError("Polynomial power must be an integer")
        if int(self.power) < 1:
            raise ValueError("Polynomial power must be at least 1")


@dataclass(frozen=True)
class SoftPower:
    """Soft-power tail ``A(z)**power``, so ``q ~ C |x|**power`` at that end.

    ``A(z) = (1 + z**4)**(1/4) - 1`` in the canonical coordinate ``z`` is
    smooth, vanishes to fourth order at the origin and grows like ``|z|``.
    An integer ``power`` is converted to :class:`Polynomial` with a
    :class:`GibbusBasisWarning`.

    Parameters
    ----------
    power : float or "auto", optional
        Tail power, greater than 1 (``1`` itself converts to
        ``Polynomial(1)``); omitted or ``"auto"`` estimates a continuous
        exponent empirically at fit time.
    """

    power: float | str = _AUTO

    def __post_init__(self):
        """Validate the fields."""
        if self.power != _AUTO and (
            not _is_real_number(self.power) or not math.isfinite(float(self.power))
        ):
            raise ValueError("SoftPower power must be a finite real number")
        if self.power != _AUTO and float(self.power) < 1.0:
            raise ValueError("SoftPower power must be at least 1")


@dataclass(frozen=True)
class Exponential:
    """Directional exponential tail: ``q ~ C exp(rate |x|)`` at that end.

    Parameters
    ----------
    rate : float or "auto", optional
        Positive rate in inverse user units; omitted or ``"auto"`` estimates
        it empirically at fit time.
    """

    rate: float | str = _AUTO

    def __post_init__(self):
        """Validate the fields."""
        if self.rate == _AUTO:
            return
        if not _is_real_number(self.rate) or not math.isfinite(float(self.rate)):
            raise ValueError("Exponential rate must be a finite real number")
        if self.rate != _AUTO and not float(self.rate) > 0.0:
            raise ValueError("Exponential rate must be positive")


_TAIL_TYPES = (Polynomial, SoftPower, Exponential)


@dataclass(frozen=True)
class BasisSpec:
    """Potential-basis configuration.

    Parameters
    ----------
    support : tuple of (float, float), optional
        Support ``(L, U)`` with ``L < U``; either end may be infinite.
    bulk_degree : int or "auto", optional
        Highest bulk monomial degree, at least 1.  ``"auto"`` lets the fitter
        choose it by the omitted-statistic degree ladder.
    log_lower, log_upper : bool or "auto", optional
        Enable the ungated ``-log(x - L)`` / ``-log(U - x)`` term; ``True``
        requires the corresponding finite endpoint.  ``"auto"`` lets the
        fitter decide by a one-sided likelihood-ratio test (no term at an
        infinite end).
    lower_tail, upper_tail : Polynomial, SoftPower, Exponential or None
        Requested leading tail class at an infinite end.  Omitting that
        class's power/rate (e.g. ``Exponential()``) requests empirical
        estimation during fitting; explicitly supplied values are fixed.
        ``None`` keeps the bulk-determined behavior.
    independent_equal_rate_coefficients : bool, optional
        Give equal-rate tails independent leading amplitudes instead of one
        shared physical amplitude.
    expansion : {"auto", "minimal", "full"}, optional
        Coefficient-span policy.  ``"auto"`` (default) starts from the
        minimal span and adds slots of the full span only when the
        omitted-statistic score test resolves them; ``"minimal"`` and
        ``"full"`` fix the span.

    Notes
    -----
    The basis is defined in the canonical coordinate ``z = (x - c) / h`` fixed
    by the support and the data quantiles ``q10`` and ``q90`` (per component
    in mixtures; see :mod:`gibbus._model.coords`).  On the full line the
    quantiles sit at ``z = -1`` and ``z = 1``, the centers of the fixed gates.
    Soft-power and exponential tails are gated toward their own end, so they
    shape only that tail; monomials shape the bulk and give subleading tail
    corrections.
    """

    support: tuple = (-math.inf, math.inf)
    bulk_degree: int = BASIS_DEFAULT_BULK_DEGREE
    log_lower: bool = False
    log_upper: bool = False
    lower_tail: Polynomial | SoftPower | Exponential | None = None
    upper_tail: Polynomial | SoftPower | Exponential | None = None
    independent_equal_rate_coefficients: bool = False
    expansion: str = _AUTO_EXPANSION

    def __post_init__(self):
        """Validate the fields."""
        try:
            lower, upper = self.support
        except (TypeError, ValueError):
            raise ValueError("support must be a pair (L, U)") from None
        if not (_is_real_number(lower) and _is_real_number(upper)):
            raise ValueError("support endpoints must be real numbers")
        lower, upper = float(lower), float(upper)
        if math.isnan(lower) or math.isnan(upper) or not lower < upper:
            raise ValueError("support must satisfy L < U")
        if lower == math.inf or upper == -math.inf:
            raise ValueError("support must satisfy L < U")
        if self.bulk_degree != _AUTO:
            if not isinstance(self.bulk_degree, numbers.Integral) or isinstance(
                self.bulk_degree, bool
            ):
                raise TypeError("bulk_degree must be an integer or 'auto'")
            if int(self.bulk_degree) < 1:
                raise ValueError("bulk_degree must be at least 1")
        for name in ("log_lower", "log_upper"):
            value = getattr(self, name)
            if not (isinstance(value, bool) or value == _AUTO):
                raise TypeError(f"{name} must be a bool or 'auto'")
        if not isinstance(self.independent_equal_rate_coefficients, bool):
            raise TypeError("independent_equal_rate_coefficients must be a bool")
        if self.log_lower is True and not math.isfinite(lower):
            raise ValueError("log_lower requires a finite lower support endpoint")
        if self.log_upper is True and not math.isfinite(upper):
            raise ValueError("log_upper requires a finite upper support endpoint")
        for name, tail, finite in (
            ("lower_tail", self.lower_tail, math.isfinite(lower)),
            ("upper_tail", self.upper_tail, math.isfinite(upper)),
        ):
            if tail is None:
                continue
            if not isinstance(tail, _TAIL_TYPES):
                raise TypeError(
                    f"{name} must be Polynomial, SoftPower, Exponential or None"
                )
            if finite:
                raise ValueError(f"{name} requires an infinite support end")
        if self.expansion not in (_AUTO_EXPANSION, _MINIMAL, _FULL):
            raise ValueError("expansion must be 'auto', 'minimal' or 'full'")


@dataclass(frozen=True)
class _TailRequest:
    """Canonical requested tail class at one infinite end.

    Parameters
    ----------
    kind : str
        ``"polynomial"``, ``"soft_power"`` or ``"exponential"``.
    value : float
        Integer power, noninteger soft power, or exponential rate in user
        units.
    converted_from_soft_power : bool
        Whether an integer ``SoftPower`` request was converted.
    """

    kind: str
    value: float
    converted_from_soft_power: bool = False

    @property
    def order(self):
        """Asymptotic growth-order key of the requested leader."""
        if self.kind == _EXPONENTIAL:
            return (2, float(self.value), 0)
        return (1, float(self.value), 0)


@dataclass(frozen=True)
class _BasisRequest:
    """Canonical validated basis request (no data-dependent values).

    Parameters
    ----------
    lower, upper : float
        Support endpoints.
    bulk_degree : int
        Highest bulk monomial degree.
    log_lower, log_upper : bool
        Endpoint-log enable flags.
    lower_tail, upper_tail : _TailRequest or None
        Canonical requested tails.
    independent : bool
        Independent equal-rate leading amplitudes.
    expansion : str
        ``"minimal"`` or ``"full"``.
    """

    lower: float
    upper: float
    bulk_degree: int
    log_lower: bool
    log_upper: bool
    lower_tail: _TailRequest | None
    upper_tail: _TailRequest | None
    independent: bool
    expansion: str

    @property
    def lower_infinite(self):
        """Whether the lower support end is ``-inf``."""
        return self.lower == -math.inf

    @property
    def upper_infinite(self):
        """Whether the upper support end is ``+inf``."""
        return self.upper == math.inf


def _needs_tail_estimation(tail, /):
    """Whether a requested family has an unresolved empirical parameter."""
    return (
        isinstance(tail, Polynomial)
        and tail.power == _AUTO
        or (isinstance(tail, SoftPower) and tail.power == _AUTO)
        or (isinstance(tail, Exponential) and tail.rate == _AUTO)
    )


def _canonical_tail(tail, side, /):
    """Convert one validated public tail to a :class:`_TailRequest`.

    Parameters
    ----------
    tail : Polynomial, SoftPower, Exponential or None
        Validated public tail.
    side : str
        ``"lower"`` or ``"upper"``, for the warning text.

    Returns
    -------
    _TailRequest or None
        Canonical tail; integer soft powers become polynomial requests.
    """
    if tail is None:
        return None
    if _needs_tail_estimation(tail):
        raise ValueError(
            f"{side} tail has an automatically estimated parameter; "
            "fit observations with Distribution.fit() before compiling the basis"
        )
    if isinstance(tail, Polynomial):
        return _TailRequest(_POLYNOMIAL, float(int(tail.power)))
    if isinstance(tail, Exponential):
        return _TailRequest(_EXPONENTIAL, float(tail.rate))
    power = float(tail.power)
    if power.is_integer():
        warnings.warn(
            f"{side} SoftPower(power={power:g}) has an integer exponent and is "
            f"treated as the polynomial tail Polynomial({int(power)})",
            GibbusBasisWarning,
            stacklevel=4,
        )
        return _TailRequest(_POLYNOMIAL, power, converted_from_soft_power=True)
    return _TailRequest(_SOFT_POWER, power)


def _canonicalize_basis_spec(spec, /):
    """Validate and canonicalize a public :class:`BasisSpec` once.

    Parameters
    ----------
    spec : BasisSpec
        Public basis configuration.

    Returns
    -------
    _BasisRequest
        Canonical request.

    Raises
    ------
    TypeError
        If ``spec`` is not a :class:`BasisSpec`.
    """
    if not isinstance(spec, BasisSpec):
        raise TypeError("spec must be a BasisSpec")
    automatic = [
        name
        for name in ("bulk_degree", "log_lower", "log_upper")
        if getattr(spec, name) == _AUTO
    ]
    if automatic:
        raise ValueError(
            f"{', '.join(automatic)}='auto' is resolved by the fitter; a compiled "
            "basis needs concrete values"
        )
    lower, upper = (float(v) for v in spec.support)
    return _BasisRequest(
        lower=lower,
        upper=upper,
        bulk_degree=int(spec.bulk_degree),
        log_lower=bool(spec.log_lower),
        log_upper=bool(spec.log_upper),
        lower_tail=_canonical_tail(spec.lower_tail, "lower"),
        upper_tail=_canonical_tail(spec.upper_tail, "upper"),
        independent=bool(spec.independent_equal_rate_coefficients),
        expansion=str(spec.expansion),
    )
