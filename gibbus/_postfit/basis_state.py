"""Fitted G3 components: durable model record, potential and packed state.

G1 owners replaced (G3-046): G1's natural-state packer (packing),
``evaluators`` (potential and density closures) and
``_spectral._tail_integrals.TailIntegrator`` (exact tail masses) for G3
components.  A G3 component is ``p(x) = exp(-theta . Phi(x)) / Z`` over a
compiled basis; everything a fitted state needs is rebuilt from the basis
request, its geometry and the slot choice (no data), the coefficients and
``log Z``.

Coordinates follow G1's component contract: the internal coordinate is the
fitting coordinate ``z = (x - c) / h`` (never reflected for G3), so the
G1 views, affine presentations and spectral evaluators apply unchanged.

* **Potential** (``_BasisPotential``): ``q`` and its derivatives up to order
  four from the compiled basis evaluator (G3-046(c): higher orders fail
  loudly).
* **Exact masses**: ``log P(a < X < b)`` from the G3 normalizer's interval
  rows (log space, certified tails), the reference for CDF tails and
  quantile refinement.
* **Packed state**: G1's runtime fields (support, window, mode, median,
  moments, spectral CDF/PPF) plus the durable G3 record; the spectral CDF is
  G1's compiled builder over the G3 density, whose panel mass must equal the
  normalizer's ``Z``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import numbers

import numpy as np

from .._defaults import (
    LOG_THRESH,
    MAX_CACHED_MOMENTS,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._fit.basis_mixture import _mixture_blocks
from .._fit.basis_newton import _matching_face
from .._model._basis_kernels import BasisKernel, potential
from .._model.basis import _compile_basis, _face_model
from .._model.basis_geometry import _BasisGeometry
from .._model.basis_normalizer import _basis_state, _interval_state, _NormalizerContext
from .._model.basis_spec import (
    BasisSpec,
    Exponential,
    Polynomial,
    SoftPower,
    TanhSpec,
    _BasisRequest,
    _TailRequest,
)
from .._model.basis_tails import _build_tail_model, _growth_verdict
from .._model.coords import _FitCoordinate, _support_kind
from .._spectral.cdf import (
    SpectralCDF,
    boundary_aware_breaks_from_amplitudes,
    density_spec,
)
from .._spectral.config import _SpectralCDFOptions, _SpectralPPFOptions
from .._spectral.ppf import SpectralPPF
from .._spectral.runtime import fallback_ppf_state, pack_cdf_state, pack_ppf_state
from .analytics import _powaff_moment_from_z_moments, _stats_from_raw_moments
from .expectation import expect_vectorized
from .fitted_state import _shared_model_geometry, _structured_scalar

_RECORD_TYPES = ("_BasisRequest", "_TailRequest", "_BasisGeometry", "_FitCoordinate")
_RECORD_VERSION = 2


# ---------------------------------------------------------------------------
# Durable model record
# ---------------------------------------------------------------------------


def _record_types():
    """Record dataclasses by name (``_encode``/``_decode``)."""
    return {
        "_BasisRequest": _BasisRequest,
        "_TailRequest": _TailRequest,
        "_BasisGeometry": _BasisGeometry,
        "_FitCoordinate": _FitCoordinate,
    }


def _encode(value, /):
    """JSON-ready value; floats as exact hexadecimal strings.

    Parameters
    ----------
    value : object
        A record value (float, str, tuple, dataclass, ...).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _RECORD_TYPES:
            raise TypeError(f"{name} is not part of the G3 model record")
        return {
            "type": name,
            "fields": {
                f.name: _encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return {"float": float(value).hex()}
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, tuple):
        return {"tuple": [_encode(v) for v in value]}
    raise TypeError(f"cannot record a {type(value).__name__}")


def _decode(value, types, /):
    """Inverse of ``_encode``.

    Parameters
    ----------
    value : object
        An encoded value.
    types : dict
        Record dataclasses by name.
    """
    if isinstance(value, dict):
        if "float" in value:
            return float.fromhex(value["float"])
        if "tuple" in value:
            return tuple(_decode(v, types) for v in value["tuple"])
        cls = types[value["type"]]
        return cls(**{k: _decode(v, types) for k, v in value["fields"].items()})
    return value


def _model_record(model, suppressed=(), /):
    """Durable JSON record of a fitted basis (request, geometry, slots, face).

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    suppressed : tuple of str
        Infinite ends whose requested leading tail functional is fixed at
        zero by the fit's face (``"lower"``/``"upper"``).

    Returns
    -------
    str
        Canonical JSON (floats exact); ``_model_from_record`` recompiles it.
    """
    payload = {
        "version": _RECORD_VERSION,
        "request": _encode(model.request),
        "geometry": _encode(model.geometry),
        "slots": _encode(tuple(tuple(int(i) for i in s) for s in model.slot_choice)),
        "columns": [c.label for c in model.columns],
        "suppressed": sorted(_check_suppressed(suppressed)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _model_from_record(text, /):
    """Recompile a basis from ``_model_record`` and check its columns.

    Parameters
    ----------
    text : str
        Canonical JSON record.

    Returns
    -------
    tuple
        ``(model, suppressed)``: the compiled basis and the ends whose
        leading tail functional the fit's face fixes at zero.

    Raises
    ------
    ValueError
        If the record is malformed, of another version, or recompiles to
        different columns.
    """

    try:
        payload = json.loads(str(text))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("malformed G3 model record") from exc
    required = {"version", "request", "geometry", "slots", "suppressed", "columns"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError("malformed G3 model record: missing fields")
    version = payload["version"]
    if type(version) is not int:
        raise ValueError("G3 model record version must be an integer")
    if version != _RECORD_VERSION:
        raise ValueError(f"unsupported G3 model record version {version}")
    types = _record_types()
    try:
        request = _decode(payload["request"], types)
        geometry = _decode(payload["geometry"], types)
        slots = _decode(payload["slots"], types)
        suppressed = _check_suppressed(tuple(payload["suppressed"]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("malformed G3 model record") from exc
    try:
        _validate_record_geometry(request, geometry)
    except (TypeError, OverflowError) as exc:
        raise ValueError("malformed G3 model record") from exc
    if not isinstance(slots, tuple) or any(
        not isinstance(group, tuple) or any(type(i) is not int or i < 0 for i in group)
        for group in slots
    ):
        raise ValueError("malformed G3 model record: invalid slots")
    if not isinstance(payload["columns"], list) or not all(
        isinstance(c, str) for c in payload["columns"]
    ):
        raise ValueError("malformed G3 model record: invalid columns")
    model = _compile_basis(request, geometry, slots=slots or None)
    labels = [c.label for c in model.columns]
    if labels != list(payload["columns"]):
        raise ValueError(
            "G3 model record recompiles to different columns: "
            f"{labels} != {payload['columns']}"
        )
    return model, suppressed


def _validate_record_geometry(request, geometry):
    """Validate decoded records before compiling a native basis table."""
    if not isinstance(request, _BasisRequest) or not isinstance(
        geometry, _BasisGeometry
    ):
        raise ValueError("malformed G3 model record: invalid request or geometry type")
    if not isinstance(geometry.coordinate, _FitCoordinate):
        raise ValueError("malformed G3 model record: invalid coordinate type")

    def tail(value):
        if value is None:
            return None
        if not isinstance(value, _TailRequest):
            raise ValueError("malformed G3 model record: invalid tail type")
        if value.kind == "polynomial":
            if (
                not isinstance(value.value, numbers.Real)
                or not math.isfinite(value.value)
                or not float(value.value).is_integer()
            ):
                raise ValueError("malformed G3 polynomial tail")
            return Polynomial(int(value.value))
        if value.kind == "soft_power":
            return SoftPower(
                value.value, "auto" if value.delta is None else value.delta
            )
        if value.kind == "exponential":
            return Exponential(value.value)
        raise ValueError("malformed G3 model record: unknown tail kind")

    def supplied(value):
        return "auto" if value is None else value

    BasisSpec(
        support=(request.lower, request.upper),
        bulk_degree=request.bulk_degree,
        log_lower=request.log_lower,
        log_upper=request.log_upper,
        lower_tail=tail(request.lower_tail),
        upper_tail=tail(request.upper_tail),
        independent_equal_rate_coefficients=request.independent,
        expansion=request.expansion,
        tanh=TanhSpec(
            *(
                supplied(getattr(request, field))
                for field in (
                    "lower_location",
                    "upper_location",
                    "lower_steepness",
                    "upper_steepness",
                )
            )
        ),
    )
    coordinate = geometry.coordinate
    if coordinate.support_kind not in {
        "real_line",
        "lower_half_line",
        "upper_half_line",
        "bounded",
    }:
        raise ValueError("malformed G3 coordinate support kind")
    for value in (coordinate.center, coordinate.scale):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError("malformed G3 coordinate scale or center")
    if coordinate.scale <= 0:
        raise ValueError("G3 coordinate scale must be positive")
    for bounds in (coordinate.canonical_support, coordinate.physical_support):
        if (
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or not all(isinstance(v, numbers.Real) for v in bounds)
            or not bounds[0] < bounds[1]
        ):
            raise ValueError("malformed G3 coordinate support")
    if coordinate.physical_support != (request.lower, request.upper):
        raise ValueError("G3 coordinate support disagrees with request")
    if coordinate.support_kind != _support_kind(coordinate.physical_support):
        raise ValueError("G3 coordinate support kind disagrees with physical support")
    if not np.array_equal(
        coordinate.canonical_support,
        coordinate.to_canonical(coordinate.physical_support),
    ):
        raise ValueError("G3 canonical support disagrees with coordinate transform")
    for side in ("lower", "upper"):
        required = []
        if math.isinf(getattr(request, side)):
            required.extend((side + "_location", side + "_steepness"))
        term = getattr(request, side + "_tail")
        if term is not None and term.kind in {"soft_power", "exponential"}:
            required.append(side + ("_delta" if term.kind == "soft_power" else "_rate"))
        if any(getattr(geometry, name) is None for name in required):
            raise ValueError(f"malformed G3 geometry: unresolved {side} parameters")
    for field in dataclasses.fields(geometry):
        if field.name == "coordinate":
            continue
        value = getattr(geometry, field.name)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"malformed G3 geometry {field.name}")
        if not field.name.endswith("location") and value <= 0:
            raise ValueError(f"G3 geometry {field.name} must be positive")


def _check_suppressed(suppressed, /):
    """Validated tuple of suppressed tail ends.

    Parameters
    ----------
    suppressed : iterable of str
        Ends named ``"lower"`` or ``"upper"``, each at most once.

    Raises
    ------
    ValueError
        If an end is unknown or repeated.
    """
    out = tuple(str(s) for s in suppressed)
    if len(set(out)) != len(out) or not set(out) <= {"lower", "upper"}:
        raise ValueError(f"invalid suppressed tail ends {out!r}")
    return out


def _check_on_face(model, theta, suppressed, /):
    """Reject coefficients whose suppressed leaders are not zero to rounding.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Coefficients.
    suppressed : tuple of str
        Suppressed infinite ends.

    Raises
    ------
    ValueError
        If a suppressed leading functional exceeds 64 ulps of its terms.
    """
    for side in suppressed:
        if math.isfinite(getattr(model.request, side)):
            raise ValueError(f"suppressed tail end {side!r} is a finite endpoint")
        row = np.array(
            [float(v) for v in model.side_asymptotics(side).leading.row_exact]
        )
        terms = np.abs(row * theta)
        if abs(float(row @ theta)) > 64.0 * np.finfo(float).eps * float(terms.sum()):
            raise ValueError(
                f"coefficients are not on the recorded face: the {side} leading "
                f"tail functional is {float(row @ theta)!r}"
            )


# ---------------------------------------------------------------------------
# Potential and exact masses
# ---------------------------------------------------------------------------


class _BasisPotential:
    """Normalized potential of one fitted G3 component.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis (the parent of a face).
    theta : numpy.ndarray
        Coefficients of ``model``.
    log_z : float
        ``log Z`` in user units over the active support.
    log_z_model : float or None
        ``log Z`` over the basis support (default ``log_z``).
    suppressed : tuple of str
        Infinite ends whose leading tail functional the fit's face fixes at
        zero.  The potential then evaluates the exact face model
        (``basis._face_model``): ``model`` and ``theta`` become the face basis
        and its coefficients, ``parent`` keeps the compiled basis (G3-049).
    """

    MAX_ORDER = 4

    def __init__(self, model, theta, log_z, /, log_z_model=None, suppressed=()):
        """Bind the component (see class docstring)."""

        self.suppressed = _check_suppressed(suppressed)
        self.parent = model
        theta = np.ascontiguousarray(theta, dtype=np.float64)
        model, free = _face_model(model, self.suppressed)
        self.model = model
        self.theta = np.ascontiguousarray(theta[list(free)]).copy()
        self.log_z = float(log_z)
        # ``log Z`` over the basis support; ``log_z`` is over the active
        # support (they differ after a truncation).
        self.log_z_model = self.log_z if log_z_model is None else float(log_z_model)
        self.active = (float(model.request.lower), float(model.request.upper))
        self.kernel = BasisKernel(model)
        coordinate = model.geometry.coordinate
        self.c = float(coordinate.center)
        self.h = float(coordinate.scale)
        self.lower = float(model.request.lower)
        self.upper = float(model.request.upper)
        self.canonical_support = (
            (self.lower - self.c) / self.h,
            (self.upper - self.c) / self.h,
        )
        self._context = None

    def _inside(self, x, /):
        """Mask of points in the closed active support.

        Parameters
        ----------
        x : array_like
            Evaluation points.
        """
        # Closed at finite ends (G1): a potential without an endpoint log is
        # finite there; with one it is +inf, from the basis itself.
        x = np.asarray(x, dtype=np.float64)
        return np.isfinite(x) & (x >= self.lower) & (x <= self.upper)

    def q(self, x, order=0, /):
        """``q^(k)(x)``, ``k <= order``, at points of the closed support.

        Parameters
        ----------
        x : array_like
            Points of the closed support (user units).
        order : int
            Highest derivative order, up to 4.

        Returns
        -------
        numpy.ndarray
            Shape ``(order + 1, R)``.
        """

        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        return potential(self.kernel, self.theta, x, int(order))

    def neg_log(self, x, shift, scale, n, /):
        """Normalized potential (or derivative) in a public affine coordinate.

        G1 ``_potential_oriented_affine_eval`` contract for the presentation
        ``x = shift + scale u`` of the basis's user units ``u``: order ``n``
        returns ``d^n/dx^n`` of the public negative log density; outside the
        support ``inf`` for ``n = 0`` and ``nan`` otherwise.  ``u`` is formed
        as ``(x - shift) / scale`` (exact without a presentation): through
        the fitting coordinate it would carry the centre's rounding, which
        swamps points within 1e-15 of a finite end at zero.

        Parameters
        ----------
        x : array_like
            Public points.
        shift, scale : float
            Presentation ``x = shift + scale u``.
        n : int
            Derivative order, up to 4.

        Raises
        ------
        ValueError
            If ``n > 4`` (the basis evaluator's highest order).
        """
        n = int(n)
        if n > self.MAX_ORDER:
            raise ValueError(
                f"G3 potentials provide derivatives up to order {self.MAX_ORDER}"
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError("presentation scale must be finite and nonzero")
        xx = np.asarray(x, dtype=np.float64)
        scalar = xx.ndim == 0
        flat = np.atleast_1d(xx).reshape(-1)
        with np.errstate(over="ignore", invalid="ignore"):
            u = (flat - float(shift)) / scale if (shift, scale) != (0.0, 1.0) else flat
        inside = self._inside(u)
        out = np.full(flat.shape, np.inf if n == 0 else np.nan)
        if np.any(inside):
            values = self.q(u[inside], n)[n]
            if n == 0:
                values = values + self.log_z + math.log(abs(scale))
            else:
                values = values / scale**n
            out[inside] = np.where(
                np.isfinite(values), values, np.inf if n == 0 else np.nan
            )
        out[np.isnan(flat)] = np.nan
        out = out.reshape(xx.shape)
        return float(out) if scalar else out

    def potential_z(self, z, n=0, /):
        """Normalized internal-coordinate potential (``expect`` callable).

        Parameters
        ----------
        z : array_like
            Internal coordinates.
        n : int
            Derivative order.
        """
        return self.neg_log(z, -self.c / self.h, 1.0 / self.h, n)

    def log_mass(self, lower, upper, /):
        """``log P(lower < X < upper)`` in user units (exact, log space).

        Rows are clipped to the active support; an empty row has ``-inf``.  One
        normalizer call resolves all rows (``_interval_state``).

        Parameters
        ----------
        lower, upper : array_like
            Row bounds in user units.

        Raises
        ------
        _QuadratureFailure
            If a row cannot be resolved to the normalizer's accuracy.
        """

        lo = np.maximum(np.asarray(lower, dtype=np.float64).reshape(-1), self.active[0])
        hi = np.minimum(np.asarray(upper, dtype=np.float64).reshape(-1), self.active[1])
        out = np.full(lo.shape, -np.inf)
        rows = (lo < hi) & ~np.isnan(lo) & ~np.isnan(hi)
        out[np.isnan(lo) | np.isnan(hi)] = np.nan
        if np.any(rows):
            if self._context is None:
                self._context = _NormalizerContext(self.model)
            state = _interval_state(
                self._context,
                self.theta,
                lo[rows],
                hi[rows],
                np.full(int(np.sum(rows)), 1.0 / np.sum(rows)),
                per_row=True,
            )
            # Renormalized to the active support (a truncated component).
            out[rows] = state.row_log_p + (self.log_z_model - self.log_z)
        return out

    def log_masses_from(self, end, points, /):
        """``log P`` of the interval between ``end`` and each point (exact).

        The masses share their end, so the points are sorted away from it and
        each mass is its predecessor's plus the piece between them: one
        normalizer call resolves the short consecutive pieces, each to
        ``rel_tol`` relative to itself, instead of every long row from the
        end.  The pieces are positive, so their compensated running sum
        (``_log_cumsum``) keeps that relative accuracy (G3-050).

        Parameters
        ----------
        end : float
            Common end of the intervals, in user units.
        points : array_like
            Other ends, in user units; a point at ``end`` has mass zero
            (``-inf``) and a NaN point gives NaN.
        """
        x = np.asarray(points, dtype=np.float64).reshape(-1)
        end = float(end)
        out = np.full(x.shape, np.nan)
        out[x == end] = -np.inf
        pieces_lo, pieces_hi, sides = [], [], []
        for above in (True, False):
            sel = (x > end) if above else (x < end)
            if not np.any(sel):
                continue
            # Unique points ordered away from the end, and the pieces between.
            far, inverse = np.unique(x[sel], return_inverse=True)
            if not above:
                far, inverse = far[::-1], far.size - 1 - inverse
            near = np.concatenate([[end], far[:-1]])
            pieces_lo.append(near if above else far)
            pieces_hi.append(far if above else near)
            sides.append((sel, inverse, far.size))
        if not sides:
            return out
        logs = self.log_mass(np.concatenate(pieces_lo), np.concatenate(pieces_hi))
        start = 0
        for sel, inverse, count in sides:
            out[sel] = _log_cumsum(logs[start : start + count])[inverse]
            start += count
        return out


def _log_cumsum(log_terms, /):
    """Running ``log sum exp`` of nonnegative terms given as logs.

    ``numpy.logaddexp.accumulate`` rounds at every step (relative error up to
    ``n`` ulps after ``n`` terms).  Here the sum is held as ``exp(ref) * s``
    with Neumaier compensation, ``ref`` moving only when a term exceeds the
    scale by ``e**600``, so the error stays a few ulps whatever ``n``.

    Parameters
    ----------
    log_terms : numpy.ndarray
        Log terms, ``-inf`` allowed (a zero term).
    """
    out = np.empty(log_terms.shape)
    ref, s, c = -math.inf, 0.0, 0.0
    for i, t in enumerate(log_terms.tolist()):
        if t > ref + 600.0 or ref == -math.inf and t > ref:
            if ref > -math.inf:
                f = math.exp(ref - t)
                s, c = s * f, c * f
            ref = t
        if t > -math.inf:
            v = math.exp(t - ref)
            total = s + v
            c += (s - total) + v if abs(s) >= v else (v - total) + s
            s = total
        out[i] = ref + math.log(s + c) if s + c > 0.0 else -math.inf
    return out


# ---------------------------------------------------------------------------
# Mode and window
# ---------------------------------------------------------------------------


def _bracket_root(f, x0, step, lo, hi, sign, /, *, max_steps=200):
    """Expand from ``x0`` towards ``sign`` until ``f(b) * sign >= 0`` or the end.

    ``f`` increases along the search direction's crossing (pass ``sign * g``
    for a ``g`` increasing away from ``x0``).  Returns ``(a, b)`` with the
    crossing between them, or ``(x_last, None)`` when the support end is
    reached first.

    Parameters
    ----------
    f : callable
        Scalar function.
    x0 : float
        Start.
    step : float
        Initial step.
    lo, hi : float
        Support bounds.
    sign : float
        Search direction, ``+1`` or ``-1``.
    max_steps : int
        Doubling limit.
    """
    a = x0
    for k in range(max_steps):
        b = x0 + sign * step * 2.0**k
        end = hi if sign > 0 else lo
        if (sign > 0 and b >= hi) or (sign < 0 and b <= lo):
            if not math.isfinite(end):
                continue
            # Last point inside a finite end (an endpoint log diverges there).
            b = end - sign * 1e-15 * max(1.0, abs(end))
            if (b - a) * sign > 0.0 and f(b) * sign >= 0.0:
                return a, b
            return a, None
        if f(b) * sign >= 0.0:
            return a, b
        a = b
    raise RuntimeError("could not bracket a root of the fitted potential")


def _bisect(f, a, b, /, *, tol):
    """Root of a monotone ``f`` between ``a`` and ``b`` (sign change).

    Parameters
    ----------
    f : callable
        Monotone scalar function.
    a, b : float
        Bracket with a sign change.
    tol : float
        Absolute width tolerance.
    """
    fa = f(a)
    for _ in range(200):
        m = 0.5 * (a + b)
        if abs(b - a) <= tol or m in (a, b):
            return m
        fm = f(m)
        if (fm < 0.0) == (fa < 0.0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def _mode_and_window(pot, threshold, /):
    """Internal-coordinate mode and mass window of a fitted convex potential.

    G1 ``_q_window_and_mode`` contract: the mode minimizes ``q``; the window
    is where ``q - q(mode) <= threshold`` (clipped to the support).  The
    fitted potential is convex on the support (shape verdict of the fit), so
    ``q'`` is monotone and each search is a bracket and bisection.

    Parameters
    ----------
    pot : _BasisPotential
        Fitted potential.
    threshold : float
        Potential rise that bounds the window.
    """
    lo, hi = pot.lower, pot.upper
    tol = 1e-15 * pot.h

    def slope(x):
        return float(pot.q(np.array([x]), 1)[1, 0])

    x0 = min(max(pot.c, np.nextafter(lo, hi)), np.nextafter(hi, lo))
    s0 = slope(x0)
    if s0 == 0.0:
        mode = x0
    else:
        sign = -1.0 if s0 > 0.0 else 1.0
        a, b = _bracket_root(slope, x0, pot.h, lo, hi, sign)
        if b is None:
            # The minimum is the support end (no log term there).
            mode = lo if sign < 0 else hi
        else:
            mode = _bisect(slope, a, b, tol=tol)
    q_mode = float(pot.q(np.array([mode]))[0, 0]) if lo < mode < hi else -math.inf
    if not math.isfinite(q_mode):
        # Boundary mode: anchor the window at the end itself.
        q_mode = float(
            pot.q(np.array([np.nextafter(mode, hi if mode == lo else lo)]))[0, 0]
        )
    window = []
    for sign in (-1.0, 1.0):
        end = lo if sign < 0 else hi
        if mode == end:
            window.append(end)
            continue

        def excess(x):
            return float(pot.q(np.array([x]))[0, 0]) - q_mode - threshold

        a, b = _bracket_root(
            lambda x, sign=sign: sign * excess(x), mode, pot.h, lo, hi, sign
        )
        if b is None:
            window.append(end)
        else:
            window.append(_bisect(excess, a, b, tol=tol))
    to_z = (np.asarray([mode, *window]) - pot.c) / pot.h
    return float(to_z[0]), np.asarray(to_z[1:], dtype=np.float64)


# ---------------------------------------------------------------------------
# Packed component state
# ---------------------------------------------------------------------------


def _log_amplitudes(model, theta, /):
    """Physical ``(lower, upper)`` endpoint-log amplitudes and their presence.

    Parameters
    ----------
    model : _BasisModel
        Fitted basis.
    theta : numpy.ndarray
        Its coefficients.
    """
    amps = np.zeros(2)
    allowed = np.zeros(2, dtype=bool)
    for i, column in enumerate(model.columns):
        if not column.nonnegative:
            continue
        side = model.functions[column.terms[0][0]].side
        j = 0 if side == "lower" else 1
        amps[j] = float(theta[i])
        allowed[j] = True
    return amps, allowed


def _amplitude_standard_errors(model, theta, information, face_basis, n_eff, /):
    """Standard errors of the log amplitudes from the observed information.

    ``Cov = N (N' I N)^-1 N' / n_eff`` on the fit's face ``N``; an amplitude
    fixed by the face (zero) or absent reports ``nan``.

    Parameters
    ----------
    model : _BasisModel
        Fitted basis.
    theta : numpy.ndarray
        Its coefficients.
    information : numpy.ndarray
        Observed information per unit weight.
    face_basis : numpy.ndarray
        Face basis ``N``.
    n_eff : float
        Kish effective sample size.
    """
    out = np.full(2, np.nan)
    if information is None or not (math.isfinite(n_eff) and n_eff > 0.0):
        return out
    n = face_basis
    reduced = n.T @ information @ n
    try:
        cov = n @ np.linalg.solve(reduced, n.T) / n_eff
    except np.linalg.LinAlgError as exc:
        # A singular face information leaves the endpoint coefficients
        # unresolved: their standard errors are reported as NaN.
        _reraise_if_debug(exc, "endpoint standard errors")
        return out
    for i, column in enumerate(model.columns):
        if not column.nonnegative:
            continue
        j = 0 if model.functions[column.terms[0][0]].side == "lower" else 1
        var = float(cov[i, i])
        if var > 1e-14 * max(1.0, abs(float(theta[i]))) ** 2:
            out[j] = math.sqrt(var)
    return out


def _canonical_raw_moment(pot, anchor_z, k, /, *, potential=None):
    """``E[z^k]`` by G1's vectorized expectation engine (breakpoint ``anchor_z``).

    Parameters
    ----------
    pot : _BasisPotential
        Fitted potential.
    anchor_z : float
        Breakpoint (canonical coordinate).
    k : int
        Moment order.
    """

    if k == 0:
        return 1.0
    return float(
        expect_vectorized(
            pot.potential_z if potential is None else potential,
            np.asarray(pot.canonical_support, dtype=np.float64),
            lambda z: np.asarray(z, dtype=np.float64) ** k,
            points=[anchor_z],
        )
    )


def _canonical_moments(pot, mode_z, /):
    """``E[z^k]``, ``k = 0..4``.

    Parameters
    ----------
    pot : _BasisPotential
        Fitted potential.
    mode_z : float
        Canonical mode (breakpoint).
    """
    evaluator = None
    if abs(pot.c / pot.h) > 16777216.0:
        # Physical-coordinate round-trips lose significant bits here.  The
        # spectral density's compiled fitting-coordinate basis already
        # evaluates the same model without that loss; use it for raw moments.
        lo, hi = pot.canonical_support
        density = density_spec(
            [
                (
                    pot.kernel,
                    pot.theta,
                    lo,
                    hi,
                    pot.log_z - math.log(pot.h),
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    -np.inf,
                    np.inf,
                )
            ],
            view=False,
        )

        def evaluator(z, order=0):
            if order != 0:
                raise ValueError(
                    "moment integrator requires the density, not derivatives"
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                return -np.log(density.pdf(z))

    return np.array(
        [_canonical_raw_moment(pot, mode_z, k, potential=evaluator) for k in range(5)]
    )


def _spectral_state(pot, mode_z, std_z, /):
    """G1's compiled spectral CDF/PPF over the G3 density (packed fields).

    Parameters
    ----------
    pot : _BasisPotential
        Fitted potential.
    mode_z : float
        Canonical mode.
    std_z : float
        Canonical standard deviation.

    Raises
    ------
    RuntimeError
        If the panel mass disagrees with the normalizer's ``Z``.
    """

    zs = pot.canonical_support
    amps, _ = _log_amplitudes(pot.model, pot.theta)
    density = density_spec(
        [
            (
                pot.kernel,
                pot.theta,
                zs[0],
                zs[1],
                pot.log_z - math.log(pot.h),
                0.0,
                1.0,
                1.0,
                1.0,
                -np.inf,
                np.inf,
            )
        ],
        view=False,
    )
    cdf = SpectralCDF(
        zs,
        density=density,
        mode=float(mode_z),
        std=float(std_z),
        initial_breaks=boundary_aware_breaks_from_amplitudes(zs, amps),
        map_scale=None,
        config=_SpectralCDFOptions(),
    )
    defect = abs(float(cdf.total_mass) - 1.0)
    if not defect <= 1e-9:
        raise RuntimeError(
            f"spectral CDF mass {cdf.total_mass!r} disagrees with the G3 "
            "normalizer's Z"
        )
    state = dict(pack_cdf_state(cdf))
    try:
        ppf = SpectralPPF(cdf, config=_SpectralPPFOptions())
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "spectral PPF construction", routine=True)
        ppf = None
        state.update(fallback_ppf_state())
        state["ppf_fallback"] = np.int32(1)
    else:
        state.update(pack_ppf_state(ppf))
        state["ppf_fallback"] = np.int32(0)
    return state, cdf, ppf


def _pack_basis_state(
    model,
    theta,
    /,
    *,
    result=None,
    n_eff=math.nan,
    boundary_p_values=(math.nan, math.nan),
    face_rank=None,
    suppressed=None,
):
    """Pack one fitted G3 component for public evaluation.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Fitted coefficients.
    result : _BasisNewtonResult or None
        The fit (diagnostics, face and observed information); ``None`` for
        a state rebuilt from a durable record.
    n_eff : float
        Effective sample size (amplitude standard errors).
    boundary_p_values : tuple of float
        Endpoint-log likelihood-ratio p-values.
    face_rank : int or None
        Free private coefficients on the fit's face (mixture components:
        the rank of the component's private rows of the joint face); by
        default from ``result``'s face, else all private coefficients.
    suppressed : tuple of str or None
        Infinite ends whose leading tail functional the fit's face fixes at
        zero; by default from ``result``'s face, else none.

    Returns
    -------
    numpy.void
        Structured fitted-state scalar (G1 runtime fields plus the G3
        record).

    Raises
    ------
    RuntimeError
        If the density is not normalizable or its moments overflow.
    """

    theta = np.asarray(theta, dtype=np.float64)
    if suppressed is None:
        suppressed = (
            ()
            if result is None
            else tuple(side for block, side in result.face.suppressed if block == 0)
        )
    suppressed = _check_suppressed(suppressed)
    _check_on_face(model, theta, suppressed)
    pot = _BasisPotential(model, theta, 0.0, suppressed=suppressed)
    state = _basis_state(_NormalizerContext(pot.model), pot.theta)
    if state.status != "proper":
        raise RuntimeError("cannot pack a non-normalizable G3 fit")
    pot.log_z = pot.log_z_model = float(state.log_z)
    mode_z, window = _mode_and_window(pot, float(LOG_THRESH))
    z_mom = _canonical_moments(pot, mode_z)
    coordinate = model.geometry.coordinate
    alpha, beta = pot.h, pot.c
    try:
        x_mom = np.array(
            [_powaff_moment_from_z_moments(z_mom, alpha, beta, k) for k in range(5)]
        )
    except OverflowError as exc:
        raise RuntimeError(
            "cannot represent fitted raw moments at this data scale; rescale the "
            "observations to a numerically moderate range and refit"
        ) from exc
    z_stats = _stats_from_raw_moments(*map(float, z_mom[1:5]))
    spectral, _, ppf = _spectral_state(pot, mode_z, z_stats["std"])
    z_median = float(ppf.ppf(0.5)) if ppf is not None else math.nan
    raw = np.full(int(MAX_CACHED_MOMENTS), np.nan)
    raw[:5] = x_mom
    canonical_raw = np.full(int(MAX_CACHED_MOMENTS), np.nan)
    canonical_raw[:5] = z_mom
    amps, allowed = _log_amplitudes(model, theta)
    private = [i for i, c in enumerate(model.columns) if not c.nonnegative]
    if result is not None:
        n_face = _private_face_rank(result.face.basis(model.n_coefficients), private)
    else:
        n_face = len(private)
    if face_rank is not None:
        n_face = int(face_rank)
    if result is not None:
        ev = result.evaluation
        information = ev.fisher - ev.missing_information
        errors = _amplitude_standard_errors(
            model, theta, information, result.face.basis(model.n_coefficients), n_eff
        )
        status = str(result.status)
        nll = float(result.objective_value)
        iterations = int(result.newton_iterations)
        evaluations = int(result.objective_evaluations)
        sub = int(result.subproblem_iterations)
        bound = float(result.final_decrease_bound)
        shape = (
            "nonnegative"
            if result.final_separation is None
            else str(result.final_separation.status)
        )
    else:
        errors = np.full(2, np.nan)
        status, nll, shape = "rebuilt", math.nan, "unavailable"
        iterations = evaluations = sub = 0
        bound = math.nan
    data = {
        "basis_model": np.str_(_model_record(model, suppressed)),
        "basis_theta": theta.copy(),
        "basis_log_z": float(state.log_z),
        "basis_log_z_model": float(state.log_z),
        "basis_log_z_rel_error": float(state.rel_error),
        "basis_shape": np.array(shape, dtype="<U32"),
        "basis_n_face": np.int64(n_face),
        "bulk_degree": np.int64(model.request.bulk_degree),
        "boundary_amplitudes": amps,
        "boundary_allowed": allowed,
        "support": np.asarray(coordinate.physical_support, dtype=np.float64),
        "canonical_support": np.asarray(pot.canonical_support, dtype=np.float64),
        "window": window,
        "canonical_mode": float(mode_z),
        "mode": float(beta + alpha * mode_z),
        "median": float(beta + alpha * z_median),
        "mean": float(beta + alpha * z_stats["mean"]),
        "var": float(alpha * alpha * z_stats["var"]),
        "std": float(alpha * z_stats["std"]),
        "skew": float(z_stats["skew"]),
        "kurt": float(z_stats["kurt"]),
        "raw_moments": raw,
        "canonical_raw_moments": canonical_raw,
        "fit_center": float(beta),
        "fit_scale": float(alpha),
        "nll": nll,
        "optimizer_success": np.int8(
            status in ("converged", "converged_approximately")
        ),
        "optimizer_status": np.array(status, dtype="<U64"),
        "optimizer_message": np.array(status.replace("_", " "), dtype="<U256"),
        "optimizer_n_iterations": np.int64(iterations),
        "optimizer_n_evaluations": np.int64(evaluations),
        "optimizer_subproblem_iterations": np.int64(sub),
        "optimizer_decrease_bound": bound,
        "boundary_standard_errors": errors,
        "boundary_p_values": np.asarray(boundary_p_values, dtype=np.float64).reshape(2),
    }
    data.update(spectral)
    return _structured_scalar(data)


def _private_face_rank(face_basis, private, /):
    """Rank of a face basis restricted to the private (non-log) rows.

    Parameters
    ----------
    face_basis : numpy.ndarray
        Joint face basis ``N``.
    private : sequence of int
        Joint rows of the component's private coefficients.
    """
    rows = np.asarray(face_basis)[private]
    if rows.size == 0:
        return 0
    sv = np.linalg.svd(rows, compute_uv=False)
    return int(np.sum(sv > 1e-10 * max(1.0, float(sv[0]))))


def _is_basis_state(struct, /):
    """Whether a packed component state is a G3 basis state.

    Parameters
    ----------
    struct : numpy.void
        Packed component state.
    """
    names = getattr(getattr(struct, "dtype", None), "names", None) or ()
    return "basis_model" in names


def _potential_of(struct, /):
    """``_BasisPotential`` of a packed G3 state (recompiled from its record).

    Parameters
    ----------
    struct : numpy.void
        Packed G3 component state.
    """
    model, suppressed = _model_from_record(struct["basis_model"])
    pot = _BasisPotential(
        model,
        np.asarray(struct["basis_theta"]),
        float(struct["basis_log_z"]),
        log_z_model=float(struct["basis_log_z_model"]),
        suppressed=suppressed,
    )
    pot.active = tuple(map(float, struct["support"]))
    return pot


# ---------------------------------------------------------------------------
# Tail rates and exp-space quantities (G1 ``analytics`` roles)
# ---------------------------------------------------------------------------


def _tail_rate(pot, mu_eff, sigma_eff, side, /):
    """Limiting absolute public-coordinate potential slope toward ``side``.

    G1 ``_tail_rate_from_geometry`` contract: ``inf`` at a finite endpoint
    and for superlinear growth; a linear tail has its slope.  The G3 leading
    slope atom (``basis_tails._growth_verdict``) is ``C zeta^b e^{r zeta}``
    in outward fitting units: growing (``b > 0`` or ``r > 0``) means
    ``inf``, constant (``b = r = 0``) the slope ``C / h`` per user unit.

    Parameters
    ----------
    pot : _BasisPotential
        Fitted potential.
    mu_eff, sigma_eff : float
        Affine map ``z = sigma_eff x + mu_eff`` of the presentation.
    side : str
        ``"lower"`` or ``"upper"``.
    """

    key = str(side).lower()
    if key not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")
    sig = float(sigma_eff)
    if not math.isfinite(sig) or sig == 0.0:
        raise RuntimeError("invalid fitted affine scale for tail-rate evaluation")
    model_side = key if sig > 0.0 else ("lower" if key == "upper" else "upper")
    end = pot.upper if model_side == "upper" else pot.lower
    if math.isfinite(end):
        return math.inf
    verdict = _growth_verdict(
        _build_tail_model(pot.model, model_side, order=1), pot.model, pot.theta
    )
    lead = verdict.slope
    if verdict.status != "proper" or lead.lead_power is None:
        raise RuntimeError(f"the {model_side} tail growth of the fit is undecided")
    if lead.lead_power > 0.0 or lead.lead_rate > 0.0:
        return math.inf
    return float(abs(lead.lead_value * sig))


def _view_root(f, lo, hi, x0, scale, /):
    """Root of an increasing ``f`` on ``(lo, hi)``; an end when none.

    Parameters
    ----------
    f : callable
        Increasing scalar function.
    lo, hi : float
        Interval.
    x0 : float
        Start.
    scale : float
        Initial bracketing step.
    """
    v0 = f(x0)
    if v0 == 0.0:
        return x0
    sign = -1.0 if v0 > 0.0 else 1.0
    a, b = _bracket_root(f, x0, scale, lo, hi, sign)
    if b is None:
        return lo if sign < 0 else hi
    lo_, hi_ = (a, b) if a < b else (b, a)
    return _bisect(f, lo_, hi_, tol=1e-15 * max(scale, abs(x0)))


def _log_exp_moment(neg_log, support, rate, k, x0, scale, /):
    """``log E[exp(k X)]`` for the public potential ``neg_log(x, n)``.

    Saddle-shifted (G1 ``_log_raw_moment_exp``): with ``x*`` the minimizer
    of ``V(x) - k x``, ``log E = k x* - V(x*) - log h + log I`` where ``I``
    integrates ``h exp(-(V(x) - V(x*) - k (x - x*)))``.  The peak ``h`` is
    the height of a normal law of the tilted curvature at ``x*``, which the
    expectation engine reads as the width of its infinite pieces (a unit
    peak made a law 9e-8 wide look 0.5 wide and integrate to zero).
    ``inf`` when ``k`` reaches the upper tail rate.

    Parameters
    ----------
    neg_log : callable
        Public potential ``neg_log(x, n)``.
    support : array_like
        Public support ``(lo, hi)``.
    rate : float
        Upper tail rate (``inf`` for superlinear growth).
    k : int
        Moment order.
    x0 : float
        Start of the saddle search.
    scale : float
        Its initial step.
    """

    if k == 0:
        return 0.0
    if math.isfinite(rate) and float(k) >= rate:
        return math.inf
    lo, hi = map(float, support)
    star = _view_root(lambda x: float(neg_log(x, 1)) - k, lo, hi, x0, scale)
    if star in (lo, hi):
        # Boundary saddle (no log at that end): anchor just inside it.
        star = star - math.copysign(1e-15 * max(1.0, abs(star)), star - x0)
    if not math.isfinite(star):
        raise RuntimeError("exp-space moment saddle is not finite")

    v_star = float(neg_log(star, 0))
    curvature = float(neg_log(star, 2))
    log_h = (
        0.5 * math.log(curvature / (2.0 * math.pi))
        if math.isfinite(curvature) and curvature > 0.0
        else 0.0
    )

    def tilted(x, n=0):
        v = np.asarray(neg_log(x, n), dtype=np.float64)
        if n == 0:
            return v - v_star - log_h - k * (np.asarray(x, dtype=np.float64) - star)
        if n == 1:
            return v - k
        return v

    total = float(
        expect_vectorized(
            tilted, support, lambda x: np.ones_like(np.asarray(x, float)), points=[star]
        )
    )
    if not (math.isfinite(total) and total > 0.0):
        raise RuntimeError(f"exp-space moment integral is not positive ({total!r})")
    return float(k * star - v_star - log_h + math.log(total))


def _relative_centered_exp_moment(neg_log, support, log_mean, k, x0, /):
    """``E[(exp(X) / m - 1)^k]`` with ``log m = log_mean`` (no cancellation).

    Integrated as ``E[f(d)]``, ``d = X - log m``, in a tilted form whose
    integrand is bounded: above ``d = 0`` the factor ``e^{k d}`` moves into
    the potential.

    Parameters
    ----------
    neg_log : callable
        Public potential ``neg_log(x, n)``.
    support : array_like
        Public support ``(lo, hi)``.
    log_mean : float
        Logarithm of the exp-space mean.
    k : int
        Moment order.
    x0 : float
        Breakpoint (the mode).
    """
    log_mean = float(log_mean)

    def shifted(x, n=0):
        v = np.asarray(neg_log(x, n), dtype=np.float64)
        d = np.asarray(x, dtype=np.float64) - log_mean
        if n == 0:
            return v - k * np.maximum(d, 0.0)
        return v - (k * (d > 0.0) if n == 1 else 0.0)

    # In units of the law's width (curvature at the mode): the engine's
    # absolute tolerance would otherwise decide an O(width^k) integral (a
    # 9e-8 law's variance came out 2e-5 off).
    curvature = float(neg_log(x0, 2))
    unit = (
        1.0 / math.sqrt(curvature)
        if math.isfinite(curvature) and curvature > 0
        else 1.0
    )

    def func(x):
        d = np.asarray(x, dtype=np.float64) - log_mean
        with np.errstate(over="ignore"):
            out = np.where(d > 0.0, -np.expm1(-np.abs(d)), np.expm1(np.minimum(d, 0)))
        return (out / unit) ** k

    scaled = expect_vectorized(shifted, support, func, points=[x0, log_mean])
    return float(scaled) * unit**k


def _exp_log_mode(neg_log, support, x0, scale, /):
    """Log-coordinate mode of ``Y = exp(X)``: the minimizer of ``V(x) + x``.

    Parameters
    ----------
    neg_log : callable
        Public potential ``neg_log(x, n)``.
    support : array_like
        Public support ``(lo, hi)``.
    x0 : float
        Start.
    scale : float
        Initial bracketing step.
    """
    lo, hi = map(float, support)
    return _view_root(lambda x: float(neg_log(x, 1)) + 1.0, lo, hi, x0, scale)


# ---------------------------------------------------------------------------
# Fitted models (single fits and mixtures)
# ---------------------------------------------------------------------------


def _fit_metadata(states, n_face, errors, p_values, /):
    """Model metadata of fitted G3 states (``fitted_state._model_metadata``).

    Parameters
    ----------
    states : sequence of numpy.void
        Packed component states.
    n_face : int
        Model face dimension.
    errors : array_like
        Shared log-amplitude standard errors.
    p_values : array_like
        Shared log-term p-values.
    """

    geometry = _shared_model_geometry(states)
    return {
        "provenance": "fitted",
        "n_parameters": int(geometry["n_parameters"]),
        "n_face_parameters": int(n_face),
        "shared_boundary": {
            "allowed": geometry["allowed"],
            "amplitudes": geometry["amplitudes"],
            "active": geometry["active"],
            "standard_errors": np.asarray(errors, dtype=np.float64),
            "p_values": np.asarray(p_values, dtype=np.float64),
        },
    }


def _pack_basis_fit(fit, n_eff, /):
    """``(weights, states, fit_metadata)`` of a single G3 fit.

    Parameters
    ----------
    fit : basis_selection._BasisFit
        Completed fit.
    n_eff : float
        Effective sample size (standard errors).
    """
    result = fit.result
    state = _pack_basis_state(
        fit.model,
        result.params,
        result=result,
        n_eff=float(n_eff),
        boundary_p_values=fit.boundary_p_values,
    )
    n_face = int(result.face.basis(fit.model.n_coefficients).shape[1])
    metadata = _fit_metadata(
        [state],
        n_face,
        state["boundary_standard_errors"],
        state["boundary_p_values"],
    )
    return np.ones(1), [state], metadata


def _pack_basis_mixture(fit, n_eff, /, boundary_p_values=(math.nan, math.nan)):
    """``(weights, states, fit_metadata)`` of a G3 mixture fit.

    Each component is packed from its coefficients; its private face rank is
    the rank of its private rows of the joint face, so the model dimension
    is the joint face's (G1 counting).  Solver records are the mixture's:
    status, EM plus polish iterations and the joint shape verdict.  The
    shared amplitudes' standard errors come from the joint observed
    information on the face.

    Parameters
    ----------
    fit : basis_mixture._MixtureFit
        Completed mixture (responsibilities on any rows).
    n_eff : float
        Effective sample size.
    boundary_p_values : tuple of float
        Shared endpoint-log likelihood-ratio p-values.
    """

    blocks = _mixture_blocks(fit.models, logits=True)
    face_basis = _matching_face(blocks, *fit.face).basis(blocks.n)
    errors = np.full(2, np.nan)
    if fit.observed_information is not None and math.isfinite(n_eff):
        info = np.asarray(fit.observed_information)
        if info.shape == (blocks.n, blocks.n):
            reduced = face_basis.T @ info @ face_basis
            try:
                cov = face_basis @ np.linalg.solve(reduced, face_basis.T) / n_eff
            except np.linalg.LinAlgError as exc:
                # A singular face information leaves the endpoint
                # coefficients unresolved: standard errors are NaN.
                _reraise_if_debug(exc, "mixture endpoint standard errors")
                cov = None
            model = fit.models[0]
            for i, column in enumerate(model.columns):
                if cov is None or not column.nonnegative:
                    continue
                j = 0 if model.functions[column.terms[0][0]].side == "lower" else 1
                coord = int(blocks.index[0][i])
                var = float(cov[coord, coord])
                if var > 1e-14 * max(1.0, abs(float(fit.thetas[0][i]))) ** 2:
                    errors[j] = math.sqrt(var)
    shape = (
        "nonnegative"
        if fit.result is None or fit.result.final_separation is None
        else str(fit.result.final_separation.status)
    )
    states = []
    for k, (model, theta) in enumerate(zip(fit.models, fit.thetas, strict=True)):
        private = [i for i, c in enumerate(model.columns) if not c.nonnegative]
        rank = _private_face_rank(face_basis[blocks.index[k]], private)
        state = _pack_basis_state(
            model,
            theta,
            face_rank=rank,
            boundary_p_values=boundary_p_values,
            suppressed=tuple(side for block, side in fit.face[0] if block == k),
        )
        status = str(fit.status)
        state["nll"] = -float(fit.log_likelihood)
        state["optimizer_success"] = np.int8(
            status in ("converged", "converged_approximately")
        )
        state["optimizer_status"] = status
        state["optimizer_message"] = status.replace("_", " ")
        state["optimizer_n_iterations"] = np.int64(
            fit.em_iterations + fit.polish_iterations
        )
        state["optimizer_n_evaluations"] = np.int64(-1)
        state["optimizer_subproblem_iterations"] = np.int64(-1)
        state["optimizer_decrease_bound"] = (
            math.nan if fit.result is None else float(fit.result.final_decrease_bound)
        )
        state["basis_shape"] = shape
        state["boundary_standard_errors"] = errors
        states.append(state)
    metadata = _fit_metadata(
        states, int(fit.n_face_parameters), errors, boundary_p_values
    )
    return np.asarray(fit.weights, dtype=np.float64), states, metadata


def _pack_selected(fit, n_eff, boundary_p_values, /):
    """Pack a ``_MixtureFit`` of any count; one component as a single fit.

    Parameters
    ----------
    fit : _MixtureFit
        Selected fit.
    n_eff : float
        Kish effective sample size.
    boundary_p_values : tuple
        Shared log-term p-values.
    """
    if len(fit.models) == 1 and fit.result is not None:
        model, result = fit.models[0], fit.result
        state = _pack_basis_state(
            model,
            fit.thetas[0],
            result=result,
            n_eff=float(n_eff),
            boundary_p_values=boundary_p_values,
        )
        metadata = _fit_metadata(
            [state],
            int(result.face.basis(model.n_coefficients).shape[1]),
            state["boundary_standard_errors"],
            state["boundary_p_values"],
        )
        return np.ones(1), [state], metadata
    return _pack_basis_mixture(fit, n_eff, boundary_p_values)
