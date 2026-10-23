"""Exact compilation of a G3 coefficient-linear potential basis.

G1 owner replaced: ``_model/natural.py`` (``_NaturalLayout``) and
``_model/spec.py`` (``_ModelSpec``), plus the degree-parity admissibility of
``_fit/inputs.py`` (``_is_poly_degree_admissible``), which becomes the exact
structural admissibility below.  Retained from G1: a fixed, immutable layout of
independent coefficients bound to one fitting coordinate; enabled endpoint-log
amplitudes as structurally nonnegative coordinates.

The full G3 potential is

``q(x) = sum_j f_j(x) [a_j + b_j t_L(x) + c_j t_U(x)]``

over the canonical underlying functions ``x^0..x^N``, ``-log(x - L)``,
``-log(U - x)``, lower/upper noninteger soft powers and lower/upper
exponentials (``N + 7`` functions, ``3 (N + 7)`` slots).  Compilation:

1. activates functions and fixes unavailable slots to zero (``a_0``, ``b``
   on a finite lower end, ``c`` on a finite upper end, gated logarithms);
2. assigns every infinite end a target leading growth order (requested
   tail, else bulk-determined; the full-line odd-degree rule of spec §7);
3. imposes exact suppression rows ``g_s = 0`` on functions growing faster
   than the target, and the shared physical-amplitude row when both ends
   request the same order;
4. selects the retained span (``"full"``: every admissible slot;
   ``"minimal"``: the deterministic smallest slot subset that preserves the
   requested leading amplitudes, outward-positive feasibility and the
   presence of every active nonconstant function);
5. reconstructs the full coefficients *algebraically* from independent
   coefficients ``theta`` through an exact rational map ``full = M theta``
   (G3-006/M-O04), so suppressions and ties hold exactly rather than as
   optimizer residuals.

Each retained column is ``sum_j f_j(x) * (alpha_0 + alpha_1 t_L + alpha_2 t_U)``
with exact rational gate coefficients.  Numerical evaluation lives in the
basis evaluator; this module performs no floating-point model evaluation.

Asymptotic growth orders are tuples compared lexicographically:
``(2, rate, 0)`` for an exponential, ``(1, p, 0)`` for ``|x|**p``,
``(1, 0, 1)`` for a logarithm and ``(0, 0, 0)`` for a decaying function.
"""

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property, lru_cache

import numpy as np

from ._basis_kernels import BasisKernel
from .basis_geometry import _resolve_basis_geometry
from .basis_spec import (
    _EXPONENTIAL,
    _FULL,
    _POLYNOMIAL,
    _SOFT_POWER,
    _canonicalize_basis_spec,
)
from .coords import _build_fit_coordinate

_MONOMIAL = "monomial"
_LOG = "log"
_LOWER = "lower"
_UPPER = "upper"
_SIDES = (_LOWER, _UPPER)

_A, _B, _C = 0, 1, 2
_SLOT_NAMES = ("a", "b", "c")

_DECAY = (0, 0.0, 0)
_CONSTANT = (1, 0.0, 0)
_LINEAR = (1, 1.0, 0)
_LOG_ORDER = (1, 0.0, 1)

# Largest |log| of the centering ratio in a shared exponential-amplitude row.
_MAX_SHARED_EXP_LOG_RATIO = 700.0


@dataclass(frozen=True)
class _BasisFunction:
    """One canonical underlying function ``f_j``.

    Parameters
    ----------
    index : int
        Canonical function index ``j``.
    kind : str
        ``"monomial"``, ``"log"``, ``"soft_power"`` or ``"exponential"``.
    side : str or None
        ``"lower"``/``"upper"`` for side-attached functions, else ``None``.
    degree : int or None
        Monomial degree.
    power : float or None
        Noninteger soft-power exponent.
    delta : float or None
        Soft-power smoothing in squared user units.
    rate : float or None
        Exponential rate in inverse user units.
    location : float or None
        Exponential centering location (the side's tanh location).
    active : bool
        Whether the function carries free admissible slots (bulk, requested
        tail, enabled log).  An inactive intermediate monomial can still
        appear inside a lower-order completion column (G3-025), always tied
        to a gate.
    label : str
        Human-readable name.
    """

    index: int
    kind: str
    side: str | None
    degree: int | None
    power: float | None
    delta: float | None
    rate: float | None
    location: float | None
    active: bool
    label: str


@dataclass(frozen=True)
class _BasisColumn:
    """One retained coefficient direction (column of the reduced basis).

    Parameters
    ----------
    terms : tuple of (int, tuple of Fraction)
        ``(function index, (alpha_0, alpha_1, alpha_2))`` pairs; the column
        is ``sum f_j (alpha_0 + alpha_1 t_L + alpha_2 t_U)``.
    label : str
        Human-readable name.
    nonnegative : bool
        Whether the coefficient is structurally constrained ``>= 0``
        (finite-endpoint logarithm amplitudes).
    """

    terms: tuple
    label: str
    nonnegative: bool


@dataclass(frozen=True)
class _AsymptoticTerm:
    """Coefficient functional of one surviving growth order at one end.

    The physical contribution is ``exp(log_factor) * (row @ theta) * shape``
    where ``shape`` is ``|x - c|**p``, ``exp(rate |x - c|)`` or
    ``-log|x - c|`` in user units.

    Parameters
    ----------
    order : tuple
        Growth-order key.
    shape : str
        ``"power"``, ``"exponential"`` or ``"log"``.
    row_exact : tuple of Fraction
        Exact functional in internal-coordinate coefficient units.
    row : numpy.ndarray
        ``float64`` copy of ``row_exact`` converted to physical units.
    log_factor : float
        Exponential centering factor (``0`` for nonexponential shapes).
    functions : tuple of int
        Contributing function indices.
    """

    order: tuple
    shape: str
    row_exact: tuple
    row: np.ndarray
    log_factor: float
    functions: tuple


@dataclass(frozen=True)
class _CurvatureTerm:
    """One term of the leading ``q''`` expansion at a linear-target end.

    The contribution to ``q''`` is ``(row_exact @ theta) * w(z)`` with a
    positive weight ``w``: ``1 / d**2`` for an opposite-endpoint logarithm
    (``shape == "log"``, distance ``d`` in user units) or
    ``|z|**order * exp(-rate |z|)`` up to a positive constant
    (``shape == "power"``, internal units ``z = (x - c) / h``).  The sign
    convention is already applied: a positive value means positive curvature.

    Parameters
    ----------
    shape : str
        ``"log"`` or ``"power"``.
    rate : float
        Exponential decay rate in internal units (``0`` for ``"log"``).
    order : float
        Power of ``|z|`` multiplying the exponential.
    row_exact : tuple of Fraction
        Exact coefficient functional over ``theta``.
    """

    shape: str
    rate: float
    order: float
    row_exact: tuple


@dataclass(frozen=True)
class _SideAsymptotics:
    """Asymptotic metadata of one infinite support end.

    Parameters
    ----------
    side : str
        ``"lower"`` or ``"upper"``.
    requested : _TailRequest or None
        Requested tail class.
    target_order : tuple
        Leading order the construction preserves at this end.
    suppressed : tuple of int
        Functions exactly suppressed at this end.
    terms : tuple of _AsymptoticTerm
        Surviving growth orders, strongest first; ``terms[0]`` is the target.
    curvature : tuple of _CurvatureTerm
        For a linear target only (whose own curvature is zero): the leading
        ``q''`` terms, strongest first -- an opposite-endpoint logarithm, then
        the slowest exponentially small gate/exponential corrections ordered
        by the power of ``|z|``.  Empty for other targets, whose leading term
        has curvature of the same sign as its outward-positive amplitude.
    """

    side: str
    requested: object
    target_order: tuple
    suppressed: tuple
    terms: tuple
    curvature: tuple = ()

    @property
    def leading(self):
        """Asymptotic term of the target leading order."""
        return self.terms[0]


@dataclass(frozen=True)
class _BasisModel:
    """Immutable compiled G3 basis.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Fixed nonlinear geometry.
    functions : tuple of _BasisFunction
        All ``N + 7`` canonical functions.
    available : numpy.ndarray
        Boolean ``(3 (N + 7),)`` mask of slots that may be nonzero: free
        admissible slots plus slots tied in by lower-order completion.
    equality_rows : tuple of dict
        Exact full-slot equalities ``{slot: Fraction}``, each ``== 0``.
    columns : tuple of _BasisColumn
        Retained columns, in ``theta`` order.
    coefficient_map_exact : tuple of tuple of Fraction
        Exact ``(3 (N + 7), d)`` reconstruction ``full = M theta``.
    coefficient_map : numpy.ndarray
        ``float64`` copy of the reconstruction map.
    asymptotics : tuple of _SideAsymptotics
        Metadata of each infinite end, lower first.
    merged_soft_power : bool
        Whether equal lower/upper soft powers share the lower slot.
    slot_choice : tuple of tuple of int
        Retained free slots of each constraint-coupled group, in canonical
        group order (the span is their exact nullspace under the equalities,
        plus lower-order completion).  ``_grow_basis`` extends it by one slot.
    face : tuple of str
        Infinite ends whose requested leader a fit's face fixes at zero
        (``_face_model``); empty for a compiled basis.  A face model is a
        runtime object: its durable record is the parent basis plus ``face``.
    polynomial_coordinates : object or None
        Private runtime recurrence plan for an equivalent polynomial fitting
        basis. Public and serialized models use the canonical columns.
    """

    request: object
    geometry: object
    functions: tuple
    available: np.ndarray
    equality_rows: tuple
    columns: tuple
    coefficient_map_exact: tuple
    coefficient_map: np.ndarray
    asymptotics: tuple
    merged_soft_power: bool
    slot_choice: tuple = ()
    face: tuple = ()
    polynomial_coordinates: object = None

    @property
    def n_coefficients(self):
        """Number of independent coefficients ``d``."""
        return len(self.columns)

    @property
    def monomial_degree(self):
        """Highest enumerated monomial degree ``N``."""
        return len(self.functions) - 7

    @property
    def slot_names(self):
        """Names of all ``3 (N + 7)`` full slots, e.g. ``"b[x^2]"``."""
        return tuple(f"{s}[{f.label}]" for f in self.functions for s in _SLOT_NAMES)

    @cached_property
    def kernel(self):
        """Compiled column table (``_basis_kernels.BasisKernel``), built once."""
        return BasisKernel(self)

    @cached_property
    def series_plan(self):
        """Immutable local-series geometry and exact sign maps, built once."""
        from .basis_series import SeriesPlan

        return SeriesPlan(self)

    @cached_property
    def series_signature(self):
        """Exact value key for reuse of immutable local-series geometry."""
        from .basis_series import _geometry_signature

        return _geometry_signature(self)

    def side_asymptotics(self, side, /):
        """Return the asymptotic metadata of an infinite end, or ``None``.

        Parameters
        ----------
        side : str
            ``"lower"`` or ``"upper"``.
        """
        for entry in self.asymptotics:
            if entry.side == side:
                return entry
        return None

    def full_coefficients(self, theta, /):
        """Reconstruct the full ``(a, b, c)`` slot vector from ``theta``.

        Parameters
        ----------
        theta : array_like
            Independent coefficients, shape ``(d,)``.

        Returns
        -------
        numpy.ndarray
            Full slot coefficients, shape ``(3 (N + 7),)``.
        """
        return self.coefficient_map @ np.asarray(theta, dtype=np.float64)


# ---------------------------------------------------------------------------
# Exact rational linear algebra
# ---------------------------------------------------------------------------


def _rref(rows, ncols, /):
    """Return the reduced row-echelon form and pivot columns.

    Parameters
    ----------
    rows : list of list of Fraction
        Matrix rows of length ``ncols``.
    ncols : int
        Number of columns.

    Returns
    -------
    tuple
        ``(reduced_rows, pivots)``; only the first ``len(pivots)`` rows are
        meaningful.
    """
    a = [list(r) for r in rows]
    pivots = []
    r = 0
    for c in range(ncols):
        if r == len(a):
            break
        piv = next((i for i in range(r, len(a)) if a[i][c] != 0), None)
        if piv is None:
            continue
        a[r], a[piv] = a[piv], a[r]
        inv = 1 / a[r][c]
        a[r] = [v * inv for v in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [vi - f * vr for vi, vr in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
    return a, pivots


def _nullspace(rows, ncols, /):
    """Return a deterministic exact nullspace basis.

    One basis vector per free (non-pivot) column, with a unit entry there.

    Parameters
    ----------
    rows : list of list of Fraction
        Equality rows of length ``ncols``.
    ncols : int
        Number of variables.
    """
    a, pivots = _rref(rows, ncols)
    pivot_set = set(pivots)
    basis = []
    for f in range(ncols):
        if f in pivot_set:
            continue
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -a[i][f]
        basis.append(v)
    return basis


def _rank(matrix, /):
    """Exact rank of a Fraction matrix given as rows.

    Parameters
    ----------
    matrix : list of list of Fraction
        Matrix rows (may be empty).
    """
    if not matrix or not matrix[0]:
        return 0
    return len(_rref(matrix, len(matrix[0]))[1])


def _image(functionals, vectors, /):
    """Return ``P N`` for functional rows ``P`` and basis vectors ``N``.

    Parameters
    ----------
    functionals : list of list of Fraction
        Rows over the same variables as ``vectors``.
    vectors : list of list of Fraction
        Nullspace basis vectors.
    """
    return [
        [sum(p * v for p, v in zip(row, vec)) for vec in vectors] for row in functionals
    ]


def _positive_feasible(matrix, /):
    """Whether the column span of a ``k x d`` matrix (``k <= 2``) meets ``R^k_{>0}``.

    Parameters
    ----------
    matrix : list of list of Fraction
        At most two rows.
    """
    k = len(matrix)
    if k == 0:
        return True
    r = _rank(matrix)
    if r == k:
        return True
    if r == 0:
        return False
    # k == 2, rank one: the span is a line through a nonzero column u.
    for j in range(len(matrix[0])):
        u = (matrix[0][j], matrix[1][j])
        if u != (0, 0):
            return u[0] * u[1] > 0
    return False


# ---------------------------------------------------------------------------
# Function enumeration and asymptotic orders
# ---------------------------------------------------------------------------


def _enumerate_functions(request, geometry, /):
    """Enumerate the ``N + 7`` canonical functions with activation flags.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved geometry.

    Returns
    -------
    tuple
        ``(functions, merged_soft_power)``.
    """
    lt, ut = request.lower_tail, request.upper_tail
    tail_degrees = {
        int(t.value) for t in (lt, ut) if t is not None and t.kind == _POLYNOMIAL
    }
    # Lower-order completion (G3-025) multiplies the gate of a soft power of
    # order p by z**k for every integer k < p, so those monomials must exist.
    completion_degrees = {
        math.ceil(t.value) - 1
        for t in (lt, ut)
        if t is not None and t.kind == _SOFT_POWER
    }
    n = max([1, request.bulk_degree, *tail_degrees, *completion_degrees])
    funcs = []
    for k in range(n + 1):
        active = k <= request.bulk_degree or k in tail_degrees
        label = "1" if k == 0 else ("x" if k == 1 else f"x^{k}")
        funcs.append(
            _BasisFunction(k, _MONOMIAL, None, k, None, None, None, None, active, label)
        )
    funcs.append(
        _BasisFunction(
            n + 1, _LOG, _LOWER, None, None, None, None, None, request.log_lower, "logL"
        )
    )
    funcs.append(
        _BasisFunction(
            n + 2, _LOG, _UPPER, None, None, None, None, None, request.log_upper, "logU"
        )
    )
    pow_l = lt is not None and lt.kind == _SOFT_POWER
    pow_u = ut is not None and ut.kind == _SOFT_POWER
    merged = (
        pow_l
        and pow_u
        and lt.value == ut.value
        and geometry.lower_delta == geometry.upper_delta
    )
    funcs.append(
        _BasisFunction(
            n + 3,
            _SOFT_POWER,
            _LOWER,
            None,
            lt.value if pow_l else None,
            geometry.lower_delta if pow_l else None,
            None,
            None,
            pow_l,
            "powL",
        )
    )
    funcs.append(
        _BasisFunction(
            n + 4,
            _SOFT_POWER,
            _UPPER,
            None,
            ut.value if pow_u else None,
            geometry.upper_delta if pow_u else None,
            None,
            None,
            pow_u and not merged,
            "powU",
        )
    )
    exp_l = lt is not None and lt.kind == _EXPONENTIAL
    exp_u = ut is not None and ut.kind == _EXPONENTIAL
    funcs.append(
        _BasisFunction(
            n + 5,
            _EXPONENTIAL,
            _LOWER,
            None,
            None,
            None,
            geometry.lower_rate if exp_l else None,
            geometry.lower_location if exp_l else None,
            exp_l,
            "expL",
        )
    )
    funcs.append(
        _BasisFunction(
            n + 6,
            _EXPONENTIAL,
            _UPPER,
            None,
            None,
            None,
            geometry.upper_rate if exp_u else None,
            geometry.upper_location if exp_u else None,
            exp_u,
            "expU",
        )
    )
    return tuple(funcs), bool(merged)


def _slot_available(fn, slot, request, /):
    """Whether one ``(function, slot)`` pair is structurally admissible.

    Parameters
    ----------
    fn : _BasisFunction
        Canonical function.
    slot : int
        ``0`` (a), ``1`` (b, lower gate) or ``2`` (c, upper gate).
    request : _BasisRequest
        Canonical request.
    """
    if not fn.active:
        return False
    if slot == _A:
        return not (fn.kind == _MONOMIAL and fn.degree == 0)
    if fn.kind == _LOG:
        return False
    if slot == _B:
        return request.lower_infinite
    return request.upper_infinite


def _order(fn, side, /):
    """Asymptotic growth order of ``f_j`` toward an infinite end.

    Parameters
    ----------
    fn : _BasisFunction
        Active canonical function.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    if fn.kind == _MONOMIAL:
        return (1, float(fn.degree), 0)
    if fn.kind == _LOG:
        return _LOG_ORDER
    if fn.kind == _SOFT_POWER:
        return (1, float(fn.power), 0)
    if fn.side == side:
        return (2, float(fn.rate), 0)
    return _DECAY


def _limit_functional(fn, side, request, available=None, /):
    """Exact gate-limit functional ``g_s`` over the function's three slots.

    ``g_L = a - b + c`` (``t_L -> -1``, ``t_U -> +1``) and
    ``g_U = a + b - c``; unavailable slots receive zero weight.

    Parameters
    ----------
    fn : _BasisFunction
        Canonical function.
    side : str
        ``"lower"`` or ``"upper"``.
    request : _BasisRequest
        Canonical request.
    available : numpy.ndarray or None
        Compiled slot mask; ``None`` uses free structural admissibility.
    """
    signs = (1, -1, 1) if side == _LOWER else (1, 1, -1)
    if available is None:
        ok = [_slot_available(fn, k, request) for k in range(3)]
    else:
        ok = [bool(available[3 * fn.index + k]) for k in range(3)]
    return [Fraction(signs[k]) if ok[k] else Fraction(0) for k in range(3)]


def _parity_sign(fn, side, /):
    """Sign converting ``g_s`` to the coefficient of ``|z|**order``.

    Parameters
    ----------
    fn : _BasisFunction
        Canonical function.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    if fn.kind == _MONOMIAL and side == _LOWER and fn.degree % 2 == 1:
        return -1
    return 1


def _exp_log_factor(fn, geometry, /):
    """Log of the factor relating ``g_s`` to the amplitude of ``exp(rate |x - c|)``.

    ``exp(-r (x - x_L)) = exp(r (x_L - c)) exp(r |x - c|)`` as ``x -> -inf``
    and ``exp(r (x - x_U)) = exp(-r (x_U - c)) exp(r |x - c|)`` as ``x -> +inf``.

    Parameters
    ----------
    fn : _BasisFunction
        Active exponential function.
    geometry : _BasisGeometry
        Resolved geometry.
    """
    if fn.side == _LOWER:
        return fn.rate * (fn.location - geometry.center)
    return -fn.rate * (fn.location - geometry.center)


def _target_order(request, side, /):
    """Leading growth order preserved at an infinite end.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    tail = request.lower_tail if side == _LOWER else request.upper_tail
    if tail is not None:
        return tail.order
    nb = request.bulk_degree
    full_line = request.lower_infinite and request.upper_infinite
    if (
        full_line
        and request.lower_tail is None
        and request.upper_tail is None
        and nb % 2 == 1
    ):
        # Spec §7: an odd top bulk monomial cannot be outward on both ends.
        return (1, float(nb - 1), 0)
    return (1, float(nb), 0)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _gate_label(gate, /):
    """Human-readable gate ``alpha_0 + alpha_1 tL + alpha_2 tU``.

    Parameters
    ----------
    gate : tuple of Fraction
        Gate coefficients.
    """
    parts = []
    for coef, name in zip(gate, ("1", "tL", "tU")):
        if coef == 0:
            continue
        mag = abs(coef)
        text_mag = str(mag) if mag.denominator == 1 else f"{float(mag):.6g}"
        if mag == 1:
            body = name
        else:
            body = text_mag if name == "1" else f"{text_mag}*{name}"
        parts.append(("-" if coef < 0 else "+") + body)
    text = "".join(parts).lstrip("+")
    return text


def _column_label(terms, functions, /):
    """Human-readable label of a retained column.

    Parameters
    ----------
    terms : tuple
        ``(function index, gate)`` pairs.
    functions : tuple of _BasisFunction
        Canonical functions.
    """
    pieces = []
    for j, gate in terms:
        g = _gate_label(gate)
        name = functions[j].label
        if g == "1":
            pieces.append(name)
        elif functions[j].kind == _MONOMIAL and functions[j].degree == 0:
            pieces.append(f"({g})")
        else:
            pieces.append(f"{name}*({g})")
    return " + ".join(pieces)


def _group_functions(active, shared_members, /):
    """Partition active functions into constraint-coupled groups.

    Parameters
    ----------
    active : list of int
        Active function indices, ascending.
    shared_members : set of int
        Functions coupled by the shared-amplitude row.
    """
    groups = []
    shared = sorted(shared_members)
    if shared:
        groups.append(tuple(shared))
    groups.extend((j,) for j in active if j not in shared_members)
    return sorted(groups, key=lambda g: g[0])


# ---------------------------------------------------------------------------
# Exact tail curvature at linear-target ends (G3-025, rule 2)
# ---------------------------------------------------------------------------


def _tail_curvature(side, functions, used, exact, request, geometry, available, /):
    """Leading ``q''`` terms at an infinite end whose target is linear.

    At such an end every power term above linear is exactly suppressed and the
    surviving algebraic part (``z``, constants) has zero curvature, so the sign
    of ``q''`` is decided by (i) the logarithm of an opposite finite endpoint,
    whose curvature ``gamma / d**2`` decays only algebraically, then (ii) the
    exponentially small corrections.  Writing the gate tails exactly,
    ``1 + t_L = 2 e^{2 u_L} + O(e^{4 u_L})`` toward ``-inf`` (and analogously
    for ``t_U`` and toward ``+inf``), each term ``f G`` contributes
    ``f(z) w e^{-r |z|}`` with ``r = 2 alpha h`` and ``w > 0`` from the gate
    location; exponential functions of the opposite side contribute their own
    decay.  Only the slowest rate is kept (all exactly-equal rates combined,
    ``w`` normalized exactly from the float geometry).  With prefactor
    ``F(z)``, the curvature is ``e^{sigma r z} (F'' + 2 sigma r F' + r^2 F)``,
    expanded in descending powers of ``|z|`` with the end's parity signs.

    Parameters
    ----------
    side : str
        ``"lower"`` or ``"upper"`` (infinite, linear target).
    functions : tuple of _BasisFunction
        Canonical functions.
    used : list of int
        Functions appearing in any column.
    exact : tuple of tuple of Fraction
        Exact ``(3 (N + 7), d)`` reconstruction map.
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved geometry.
    available : numpy.ndarray
        Compiled slot mask.

    Returns
    -------
    tuple of _CurvatureTerm
        Strongest first; terms whose functional is identically zero are
        omitted.
    """
    d = len(exact[0]) if exact else 0
    h, c = geometry.scale, geometry.center
    lower_end = side == _LOWER
    sigma = 1 if lower_end else -1

    def slot_row(slot, scale):
        return [scale * v for v in exact[slot]]

    gates = []  # (slot offset, rate, log weight, coefficient sign)
    if geometry.lower_location is not None:
        rate = 2.0 * geometry.lower_steepness * h
        zl = (geometry.lower_location - c) / h
        gates.append((1, rate, -sigma * rate * zl, 2 * sigma))
    if geometry.upper_location is not None:
        rate = 2.0 * geometry.upper_steepness * h
        zu = (geometry.upper_location - c) / h
        gates.append((2, rate, -sigma * rate * zu, -2 * sigma))

    terms = []
    pieces = []  # (rate, log weight, kind, key, row)
    for j in used:
        fn = functions[j]
        if fn.kind == _LOG:
            row = list(exact[3 * j])
            if any(v != 0 for v in row):
                terms.append(_CurvatureTerm("log", 0.0, 0.0, tuple(row)))
            continue
        if fn.kind == _EXPONENTIAL:
            if fn.side == side:
                raise RuntimeError(
                    "internal error: growing exponential at a linear end"
                )
            base_rate = fn.rate * h
            base_log = fn.rate * (c - fn.location) * sigma
            g = _limit_functional(fn, side, request, available)
            row = [
                sum(g[k] * exact[3 * j + k][col] for k in range(3)) for col in range(d)
            ]
            pieces.append((base_rate, base_log, _MONOMIAL, 0, row))
            kind, key = _MONOMIAL, 0
        else:
            base_rate, base_log = 0.0, 0.0
            kind = fn.kind
            key = fn.degree if fn.kind == _MONOMIAL else fn.power
        for offset, rate, log_w, sign in gates:
            row = slot_row(3 * j + offset, sign)
            pieces.append((base_rate + rate, base_log + log_w, kind, key, row))

    pieces = [p for p in pieces if any(v != 0 for v in p[4])]
    if pieces:
        rate = min(p[0] for p in pieces)
        group = [p for p in pieces if p[0] == rate]
        top = max(p[1] for p in group)
        beta, soft = {}, {}
        for _, log_w, kind, key, row in group:
            w = Fraction(math.exp(log_w - top))
            target = beta if kind == _MONOMIAL else soft
            acc = target.setdefault(key, [Fraction(0)] * d)
            target[key] = [a + w * v for a, v in zip(acc, row)]
        r = Fraction(rate)
        zero = [Fraction(0)] * d
        ordered = []
        if beta:
            for n in range(max(beta), -1, -1):
                b0, b1, b2 = (
                    beta.get(n, zero),
                    beta.get(n + 1, zero),
                    beta.get(n + 2, zero),
                )
                parity = -1 if (lower_end and n % 2 == 1) else 1
                row = [
                    parity
                    * (
                        r * r * v0
                        + 2 * sigma * r * (n + 1) * v1
                        + (n + 2) * (n + 1) * v2
                    )
                    for v0, v1, v2 in zip(b0, b1, b2)
                ]
                ordered.append((float(n), row))
        for p, row in soft.items():
            ordered.append((float(p), [r * r * v for v in row]))
        ordered.sort(key=lambda t: -t[0])
        for order, row in ordered:
            if any(v != 0 for v in row):
                terms.append(_CurvatureTerm("power", rate, order, tuple(row)))
    return tuple(terms)


def _cone_feasible(equal, strict, weak, /):
    """Exact test for ``theta`` with ``E theta = 0``, ``S theta > 0``, ``W theta >= 0``.

    The system is homogeneous, so only the image ``y = M theta`` of the
    stacked rows matters.  The image is parametrized exactly, equalities are
    eliminated by substitution, and the remaining few inequalities by
    Fourier-Motzkin elimination with strictness tracking.

    Parameters
    ----------
    equal, strict, weak : list of list of Fraction
        Functional rows over ``theta``.
    """
    if not strict:
        return True
    rows = [*equal, *strict, *weak]
    k, d = len(rows), len(rows[0])
    transposed = [[rows[i][j] for i in range(k)] for j in range(d)]
    reduced, pivots = _rref(transposed, k)
    basis = reduced[: len(pivots)]
    if not basis:
        return False

    def image_row(i):
        return [b[i] for b in basis]

    eqs = [image_row(i) for i in range(len(equal))]
    ineqs = [(image_row(len(equal) + i), True) for i in range(len(strict))]
    ineqs += [
        (image_row(len(equal) + len(strict) + i), False) for i in range(len(weak))
    ]
    for e in eqs:
        v = next((i for i, a in enumerate(e) if a != 0), None)
        if v is None:
            continue
        ineqs = [
            ([a - row[v] / e[v] * b for a, b in zip(row, e)], st) for row, st in ineqs
        ]
        eqs = [[a - row[v] / e[v] * b for a, b in zip(row, e)] for row in eqs]
    for v in range(len(basis)):
        pos = [(r, st) for r, st in ineqs if r[v] > 0]
        neg = [(r, st) for r, st in ineqs if r[v] < 0]
        new = [(r, st) for r, st in ineqs if r[v] == 0]
        for rp, sp in pos:
            for rn, sn in neg:
                new.append(
                    ([a * (-rn[v]) + b * rp[v] for a, b in zip(rp, rn)], sp or sn)
                )
        seen, ineqs = set(), []
        for r, st in new:
            scale = max(abs(a) for a in r) or Fraction(1)
            key = (tuple(a / scale for a in r), st)
            if key not in seen:
                seen.add(key)
                ineqs.append((r, st))
    return not any(st for _, st in ineqs)


def _tail_curvature_feasible(asymptotics, columns, d, /):
    """Whether outward growth and eventually nonnegative ``q''`` are compatible.

    Requires, jointly over all infinite ends, outward-positive leading
    amplitudes, nonnegative logarithm amplitudes, and at every linear-target
    end a lexicographically nonnegative curvature expansion (the first nonzero
    term positive, or all terms zero).  Each is necessary for a convex proper
    member, so a ``False`` result is an exact proof that none exists.

    Parameters
    ----------
    asymptotics : list of _SideAsymptotics
        Per-end metadata.
    columns : list of _BasisColumn
        Retained columns.
    d : int
        Number of coefficients.
    """
    strict = [list(side.leading.row_exact) for side in asymptotics]
    weak = []
    for i, column in enumerate(columns):
        if column.nonnegative:
            unit = [Fraction(0)] * d
            unit[i] = Fraction(1)
            weak.append(unit)
    choices = []
    for side in asymptotics:
        seq = [list(t.row_exact) for t in side.curvature]
        options = [(seq[:i], [seq[i]]) for i in range(len(seq))]
        options.append((seq, []))
        choices.append(options)
    for combo in itertools.product(*choices):
        equal = [row for eq, _ in combo for row in eq]
        positive = [row for _, st in combo for row in st]
        if _cone_feasible(equal, strict + positive, weak):
            return True
    return False


def _lower_order_completion(map_columns, functions, /):
    """Lower-order completion of gated power terms (G3-025, rule 1).

    For every retained column and every term ``f G`` in it whose gate
    ``G = alpha_0 + alpha_1 t_L + alpha_2 t_U`` is gated (``alpha_1`` or
    ``alpha_2`` nonzero) and whose ``f`` is a monomial ``z**n`` or a soft power
    of order ``p``, the columns ``z**k G`` for every integer ``0 <= k < n``
    (resp. ``k < p``) are added.  For ``k = 0`` the ungated constant
    ``alpha_0`` is the normalization gauge and is dropped.  For monomials this
    makes every gated span translation invariant, so the family does not
    depend on the origin of the fitting coordinate.  Each added column keeps
    the gate of its parent, hence vanishes wherever the parent was suppressed
    and never outgrows a target order.

    Parameters
    ----------
    map_columns : list of list of Fraction
        Retained full-slot columns (linearly independent).
    functions : tuple of _BasisFunction
        Canonical functions.

    Returns
    -------
    list of list of Fraction
        New full-slot columns, independent of each other and of
        ``map_columns``, in deterministic order.
    """
    n_slots = 3 * len(functions)
    # Echelon rows are kept sparse ({slot: value}, nonzeros only): columns are
    # gate triples of a few functions, and exact rational arithmetic on the
    # zeros dominated basis assembly (G3-041).  The elimination is unchanged.
    echelon = []

    def reduce(vec):
        v = dict(vec)
        for pivot, row in echelon:
            f = v.get(pivot)
            if f is not None:
                for i, b in row.items():
                    a = v.get(i, 0) - f * b
                    if a != 0:
                        v[i] = a
                    else:
                        v.pop(i, None)
        return v

    def insert(vec):
        v = reduce(vec)
        if not v:
            return False
        pivot = min(v)
        inv = 1 / v[pivot]
        echelon.append((pivot, {i: a * inv for i, a in v.items()}))
        return True

    for col in map_columns:
        insert({i: a for i, a in enumerate(col) if a})
    added = []
    for col in map_columns:
        for fn in functions:
            j = fn.index
            gate = col[3 * j : 3 * j + 3]
            if not gate[1] and not gate[2]:
                continue
            if fn.kind == _MONOMIAL:
                top = fn.degree
            elif fn.kind == _SOFT_POWER:
                top = math.ceil(fn.power)
            else:
                continue
            for k in range(top - 1, -1, -1):
                sparse = {3 * k + i: gate[i] for i in range(3) if gate[i] and (i or k)}
                if insert(sparse):
                    vec = [Fraction(0)] * n_slots
                    vec[3 * k] = gate[0] if k > 0 else Fraction(0)
                    vec[3 * k + 1] = gate[1]
                    vec[3 * k + 2] = gate[2]
                    added.append(vec)
    return added


@dataclass(frozen=True)
class _BasisLayout:
    """Slot structure of one request and geometry, before a span is chosen.

    Parameters
    ----------
    functions : tuple of _BasisFunction
        Canonical functions.
    merged : bool
        Whether equal soft powers share the lower slot.
    rows : tuple of dict
        Exact full-slot equalities.
    entries : tuple
        Per group ``(group, group_slots, span, acceptable, full_vecs,
        ref_rank, pref)``.
    assemble : callable
        Maps a per-group choice of direction vectors to
        ``(columns, map_columns, available, exact, asymptotics)``.
    """

    functions: tuple
    merged: bool
    rows: tuple
    entries: tuple
    assemble: object
    # Assembled parts per slot choice (``select`` is a pure function of it).
    selected: dict = field(default_factory=dict, compare=False, repr=False)

    def full_slots(self):
        """Every group's admissible slots (the full span)."""
        return tuple(tuple(sorted(e[6])) for e in self.entries)

    def select(self, slots, /):
        """Assemble a slot choice, or ``None`` if it is not admissible.

        Parameters
        ----------
        slots : tuple of tuple of int
            Retained slots per group.
        """
        key = tuple(tuple(sorted(subset)) for subset in slots)
        if key not in self.selected:
            self.selected[key] = self._select(key)
        return self.selected[key]

    def _select(self, slots, /):
        """Uncached ``select``.

        Parameters
        ----------
        slots : tuple
            Chosen slot labels.
        """
        choice = []
        for entry, subset in zip(self.entries, slots, strict=True):
            vecs = entry[2](sorted(subset))
            if not entry[3](vecs, entry[5]):
                return None
            choice.append(vecs)
        parts = self.assemble(choice)
        if not _tail_curvature_feasible(parts[4], parts[0], len(parts[1])):
            return None
        return parts


@lru_cache(maxsize=32)
def _basis_layout(request, geometry, /):
    """Slot structure of a canonical request and fixed geometry.

    A pure function of its (hashable, immutable) arguments, cached: the
    automatic pipeline compiles, grows and completes the same layout
    repeatedly (G3-041).

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved fixed geometry.

    Returns
    -------
    _BasisLayout
        Groups, admissibility tests and the assembler; the full span is
        verified admissible.

    Raises
    ------
    ValueError
        If the family cannot represent a proper log-concave density (no
        surviving outward growth of at least linear order at an infinite end,
        or no outward-positive leading amplitude), or a shared exponential
        amplitude is numerically unrepresentable.
    """
    functions, merged = _enumerate_functions(request, geometry)
    available = np.array(
        [_slot_available(f, k, request) for f in functions for k in range(3)],
        dtype=bool,
    )
    active = [f.index for f in functions if f.active]
    sides = [
        s
        for s in _SIDES
        if (request.lower_infinite if s == _LOWER else request.upper_infinite)
    ]

    targets = {}
    for s in sides:
        target = _target_order(request, s)
        if target < _LINEAR:
            raise ValueError(
                f"structurally improper basis: no admissible term can grow at least "
                f"linearly toward the {s} infinite end (full-line bulk_degree="
                f"{request.bulk_degree} with both tails unspecified); increase "
                "bulk_degree or request a tail"
            )
        targets[s] = target

    suppressed = {s: [] for s in sides}
    leaders = {s: [] for s in sides}
    rows = []  # (dict slot -> Fraction)
    for j in active:
        fn = functions[j]
        for s in sides:
            order = _order(fn, s)
            if order > targets[s]:
                if fn.kind == _EXPONENTIAL:
                    raise RuntimeError(
                        "internal error: an exponential requires suppression"
                    )
                suppressed[s].append(j)
                g = _limit_functional(fn, s, request)
                rows.append({3 * j + k: g[k] for k in range(3) if g[k] != 0})
            elif order == targets[s]:
                leaders[s].append(j)

    lt, ut = request.lower_tail, request.upper_tail
    shared = (
        len(sides) == 2
        and lt is not None
        and ut is not None
        and lt.order == ut.order
        and not request.independent
    )
    shared_members = set()
    if shared:
        weights = {_LOWER: Fraction(1), _UPPER: Fraction(1)}
        if lt.kind == _EXPONENTIAL:
            (jl,), (ju,) = leaders[_LOWER], leaders[_UPPER]
            log_ratio = _exp_log_factor(functions[jl], geometry) - _exp_log_factor(
                functions[ju], geometry
            )
            if abs(log_ratio) > _MAX_SHARED_EXP_LOG_RATIO:
                raise ValueError(
                    "shared exponential amplitude is unrepresentable: tanh locations "
                    "are too asymmetric about the basis center"
                )
            # exp(fL) gL = exp(fU) gU, scaled so the larger factor is one.
            factor = Fraction(math.exp(-abs(log_ratio)))
            weights[_LOWER if log_ratio < 0 else _UPPER] = factor
        row = {}
        for s, sign in ((_LOWER, 1), (_UPPER, -1)):
            for j in leaders[s]:
                g = _limit_functional(functions[j], s, request)
                p = _parity_sign(functions[j], s) * sign * weights[s]
                for k in range(3):
                    if g[k] != 0:
                        row[3 * j + k] = row.get(3 * j + k, Fraction(0)) + p * g[k]
                shared_members.add(j)
        rows.append({k: v for k, v in row.items() if v != 0})

    def physical_functional(group_slots, j, s):
        g = _limit_functional(functions[j], s, request)
        p = _parity_sign(functions[j], s)
        return [
            Fraction(p) * g[slot % 3] if slot // 3 == j else Fraction(0)
            for slot in group_slots
        ]

    def leader_functionals(group, group_slots, sides_subset):
        out = []
        for s in sides_subset:
            members = [j for j in group if j in leaders[s]]
            if not members:
                continue
            total = [Fraction(0)] * len(group_slots)
            for j in members:
                total = [
                    t + v for t, v in zip(total, physical_functional(group_slots, j, s))
                ]
            out.append(total)
        return out

    entries = []
    for group in _group_functions(active, shared_members):
        group_slots = [
            3 * j + k for j in group for k in range(3) if available[3 * j + k]
        ]
        group_rows = [
            [row.get(slot, Fraction(0)) for slot in group_slots]
            for row in rows
            if any(slot // 3 in group for slot in row)
        ]
        specified = [s for s in sides if (lt if s == _LOWER else ut) is not None]
        spec_funcs = leader_functionals(group, group_slots, specified)
        lead_funcs = leader_functionals(group, group_slots, sides)
        required = [
            j
            for j in group
            if not (functions[j].kind == _MONOMIAL and functions[j].degree == 0)
        ]

        def span(subset, group_slots=group_slots, group_rows=group_rows):
            idx = [group_slots.index(s) for s in subset]
            sub_rows = [[r[i] for i in idx] for r in group_rows]
            vecs = _nullspace(sub_rows, len(idx))
            lifted = []
            for v in vecs:
                full = [Fraction(0)] * len(group_slots)
                for i, val in zip(idx, v):
                    full[i] = val
                lifted.append(full)
            return lifted

        def acceptable(
            vecs,
            ref_rank,
            group_slots=group_slots,
            required=required,
            spec_funcs=spec_funcs,
            lead_funcs=lead_funcs,
        ):
            for j in required:
                pos = [i for i, slot in enumerate(group_slots) if slot // 3 == j]
                if not any(v[i] != 0 for v in vecs for i in pos):
                    return False
            if spec_funcs and _rank(_image(spec_funcs, vecs)) != ref_rank:
                return False
            return _positive_feasible(_image(lead_funcs, vecs)) if lead_funcs else True

        full_vecs = span(group_slots)
        ref_rank = _rank(_image(spec_funcs, full_vecs)) if spec_funcs else 0
        if not acceptable(full_vecs, ref_rank):
            names = ", ".join(functions[j].label for j in group)
            raise ValueError(
                f"structurally improper basis: terms {{{names}}} cannot supply an "
                "outward-positive leading amplitude under the requested constraints"
            )
        pref = []
        for j in group:
            order = [_A, _B, _C]
            in_l = j in suppressed.get(_LOWER, [])
            in_u = j in suppressed.get(_UPPER, [])
            if in_u and not in_l:
                order = [_A, _C, _B]
            pref.extend(3 * j + k for k in order if available[3 * j + k])
        entries.append(
            (group, group_slots, span, acceptable, full_vecs, ref_rank, pref)
        )

    def assemble(choice):
        return _assemble_basis(
            request,
            geometry,
            functions,
            available,
            sides,
            targets,
            suppressed,
            [(e[0], e[1]) for e in entries],
            choice,
        )

    # Admissibility = outward-positive leading amplitudes, nonnegative log
    # amplitudes and eventually nonnegative q'' at linear tails, jointly (exact).
    choice = [e[4] for e in entries]
    parts = assemble(choice)
    if not _tail_curvature_feasible(parts[4], parts[0], len(parts[1])):
        raise ValueError(
            "structurally improper basis: no coefficient vector makes q'' "
            "eventually nonnegative together with outward growth at every "
            "linear tail (tanh-gated terms of degree <= 1 cannot form an "
            "exactly convex linear tail); raise bulk_degree or request a "
            "faster tail"
        )
    return _BasisLayout(functions, merged, tuple(rows), tuple(entries), assemble)


def _compile_basis(request, geometry, /, slots=None):
    """Compile a canonical request and fixed geometry into a :class:`_BasisModel`.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved fixed geometry.
    slots : tuple of tuple of int or None
        Explicit slot choice per group (expansion growth); ``None`` applies the
        request's policy: ``"full"`` keeps every admissible slot, ``"minimal"``
        and ``"auto"`` the minimal span (``"auto"`` grows it in the fitter).

    Returns
    -------
    _BasisModel
        Compiled immutable basis.

    Raises
    ------
    ValueError
        If the family cannot represent a proper log-concave density, or an
        explicit slot choice is not admissible.
    """
    layout = _basis_layout(request, geometry)
    entries = layout.entries
    combos = list(layout.full_slots())
    parts = layout.select(tuple(combos))
    if slots is not None:
        combos = [tuple(sorted(c)) for c in slots]
        parts = layout.select(tuple(combos))
        if parts is None:
            raise ValueError("slot choice is not admissible")
    elif request.expansion != _FULL:
        # Minimal: groups in canonical order, each reduced to its smallest
        # admissible slot subset given the earlier choices (later groups full).
        for gi, (_, _, span, acceptable, _, ref_rank, pref) in enumerate(entries):
            found = False
            for size in range(len(pref) + 1):
                for combo in itertools.combinations(pref, size):
                    trial = (*combos[:gi], tuple(sorted(combo)), *combos[gi + 1 :])
                    candidate = layout.select(trial)
                    if candidate is not None:
                        combos, parts, found = list(trial), candidate, True
                        break
                if found:
                    break
    return _model_from_parts(request, geometry, layout, tuple(combos), parts)


def _model_from_parts(request, geometry, layout, combos, parts, /):
    """Package assembled parts as a :class:`_BasisModel`.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Fixed nonlinear geometry.
    layout : _BasisLayout
        Slot layout.
    combos : tuple
        Chosen slot combination per function.
    parts : tuple
        ``(columns, map_columns, available, exact, asymptotics)`` from the
        assembly.
    """
    columns, map_columns, available, exact, asymptotics = parts
    n_slots = 3 * len(layout.functions)
    d = len(map_columns)
    coefficient_map = np.array(
        [[float(v) for v in row] for row in exact], dtype=np.float64
    ).reshape(n_slots, d)
    return _BasisModel(
        request=request,
        geometry=geometry,
        functions=layout.functions,
        available=available,
        equality_rows=layout.rows,
        columns=tuple(columns),
        coefficient_map_exact=exact,
        coefficient_map=coefficient_map,
        asymptotics=tuple(asymptotics),
        merged_soft_power=layout.merged,
        slot_choice=combos,
    )


def _grow_basis(model, /):
    """Admissible one-slot extensions of a basis inside its full span.

    Each candidate adds one free slot of one group (spec section 8: the
    minimal span grows toward the full span); its span strictly contains the
    current one and keeps the exact admissibility rules (G3-025).

    Parameters
    ----------
    model : _BasisModel
        Current basis.

    Returns
    -------
    list of tuple
        ``(label, candidate)`` per admissible extension, ``label`` the slot
        name, in canonical group/preference order.
    """
    layout = _basis_layout(model.request, model.geometry)
    current = model.slot_choice
    out = []
    for gi, entry in enumerate(layout.entries):
        for slot in entry[6]:
            if slot in current[gi]:
                continue
            trial = tuple(
                tuple(sorted((*c, slot))) if i == gi else c
                for i, c in enumerate(current)
            )
            parts = layout.select(trial)
            if parts is None or len(parts[1]) <= model.n_coefficients:
                continue
            candidate = _model_from_parts(
                model.request, model.geometry, layout, trial, parts
            )
            out.append((model.slot_names[slot], candidate))
    return out


def _full_basis(model, /):
    """The full-span basis of a model's request and geometry.

    Parameters
    ----------
    model : _BasisModel
        A compiled basis.
    """
    layout = _basis_layout(model.request, model.geometry)
    slots = layout.full_slots()
    return _model_from_parts(
        model.request, model.geometry, layout, slots, layout.select(slots)
    )


def _assemble_basis(
    request,
    geometry,
    functions,
    available,
    sides,
    targets,
    suppressed,
    groups,
    choice,
    /,
):
    """Assemble columns, the exact map and per-end asymptotics for a selection.

    Parameters
    ----------
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved geometry.
    functions : tuple of _BasisFunction
        Canonical functions.
    available : numpy.ndarray
        Free admissible slot mask.
    sides : list of str
        Infinite ends.
    targets, suppressed : dict
        Per-end target orders and exactly suppressed functions.
    groups : list of (tuple, list)
        ``(group functions, group slots)`` per constraint-coupled group.
    choice : list of list of list of Fraction
        Selected group-local direction vectors per group.

    Returns
    -------
    tuple
        ``(columns, map_columns, available, exact, asymptotics)`` after
        lower-order completion.
    """
    m = len(functions)
    n_slots = 3 * m
    columns, map_columns = [], []
    for (group, group_slots), vecs in zip(groups, choice):
        for vec in vecs:
            full = [Fraction(0)] * n_slots
            for i, slot in enumerate(group_slots):
                full[slot] = vec[i]
            terms = tuple(
                (j, tuple(full[3 * j : 3 * j + 3]))
                for j in group
                if full[3 * j] or full[3 * j + 1] or full[3 * j + 2]
            )
            nonneg = len(terms) == 1 and functions[terms[0][0]].kind == _LOG
            columns.append(_BasisColumn(terms, _column_label(terms, functions), nonneg))
            map_columns.append(full)

    for full in _lower_order_completion(map_columns, functions):
        terms = tuple(
            (j, tuple(full[3 * j : 3 * j + 3]))
            for j in range(m)
            if full[3 * j] or full[3 * j + 1] or full[3 * j + 2]
        )
        columns.append(_BasisColumn(terms, _column_label(terms, functions), False))
        map_columns.append(full)
    nonzero = {slot for col in map_columns for slot, v in enumerate(col) if v}
    available = available | np.array(
        [slot in nonzero for slot in range(n_slots)], dtype=bool
    )
    used = sorted({j for column in columns for j, _ in column.terms})
    d = len(map_columns)
    exact = tuple(tuple(map_columns[c][i] for c in range(d)) for i in range(n_slots))

    asymptotics = [
        _end_asymptotics(
            s,
            request,
            geometry,
            functions,
            available,
            used,
            exact,
            targets[s],
            suppressed[s],
        )
        for s in sides
    ]
    available.setflags(write=False)
    return (
        tuple(columns),
        tuple(tuple(col) for col in map_columns),
        available,
        exact,
        tuple(asymptotics),
    )


def _end_asymptotics(
    s, request, geometry, functions, available, used, exact, target, suppressed, /
):
    """Asymptotic metadata of one infinite end from an exact coefficient map.

    Parameters
    ----------
    s : str
        ``"lower"`` or ``"upper"``.
    request : _BasisRequest
        Canonical request.
    geometry : _BasisGeometry
        Resolved geometry.
    functions : tuple of _BasisFunction
        Canonical functions.
    available : numpy.ndarray
        Compiled slot mask.
    used : list of int
        Functions appearing in any column.
    exact : tuple of tuple of Fraction
        Exact ``(3 (N + 7), d)`` reconstruction map.
    target : tuple or None
        Target leading order (a completion term outgrowing it is an internal
        error); ``None`` takes the strongest order whose functional is not
        identically zero (a face of a compiled basis, ``_face_model``).
    suppressed : tuple of int
        Functions exactly suppressed at this end.

    Returns
    -------
    _SideAsymptotics

    Raises
    ------
    RuntimeError
        If the target leader is missing (internal error).
    """
    d = len(exact[0]) if exact else 0
    by_order = {}
    for j in used:
        fn = functions[j]
        order = _order(fn, s)
        if j in suppressed or order <= _CONSTANT:
            continue
        g = _limit_functional(fn, s, request, available)
        p = _parity_sign(fn, s)
        # Only nonzero functional and map entries contribute (exact).
        weights = [(Fraction(p) * g[k], exact[3 * j + k]) for k in range(3) if g[k]]
        row = [Fraction(0)] * d
        for w, slot_row in weights:
            for c in range(d):
                if slot_row[c]:
                    row[c] += w * slot_row[c]
        if target is not None and order > target and any(v != 0 for v in row):
            raise RuntimeError(
                f"internal error: a completion term outgrows the {s} target"
            )
        entry = by_order.setdefault(order, {"row": [Fraction(0)] * d, "funcs": []})
        entry["row"] = [a + b for a, b in zip(entry["row"], row)]
        entry["funcs"].append(j)
    terms = []
    for order in sorted(by_order, reverse=True):
        entry = by_order[order]
        if all(v == 0 for v in entry["row"]):
            continue
        funcs = tuple(entry["funcs"])
        if order[0] == 2:
            shape, log_factor, unit = (
                "exponential",
                _exp_log_factor(functions[funcs[0]], geometry),
                1.0,
            )
        elif order == _LOG_ORDER:
            shape, log_factor, unit = "log", 0.0, 1.0
        else:
            # Internal monomials/soft powers are in z = (x - c)/h units.
            shape, log_factor, unit = "power", 0.0, geometry.scale ** (-order[1])
        row_float = np.array([float(v) * unit for v in entry["row"]], dtype=np.float64)
        terms.append(
            _AsymptoticTerm(
                order, shape, tuple(entry["row"]), row_float, log_factor, funcs
            )
        )
    if target is None and terms:
        target = terms[0].order
    if not terms or terms[0].order != target:
        raise RuntimeError(f"internal error: target leader missing at the {s} end")
    requested = request.lower_tail if s == _LOWER else request.upper_tail
    curvature = (
        _tail_curvature(s, functions, used, exact, request, geometry, available)
        if target == _LINEAR
        else ()
    )
    return _SideAsymptotics(
        s, requested, target, tuple(suppressed), tuple(terms), curvature
    )


def _face_model(model, suppressed, /):
    """The basis restricted to a fit's leader face, in exact arithmetic.

    A fit on a leader face (``basis_newton``) fixes the requested leading
    functional of each ``suppressed`` end at zero.  In the parent basis that
    zero is a cancellation between columns (``x^2`` against its gated copies,
    ``e^x`` against ``e^x t_L``): evaluated in binary64 it leaves a rounding
    residue that grows with the leading function and makes the far tail
    improper.  The face model substitutes the face's exact nullspace
    ``theta = E phi`` into the columns, so cancelling gate coefficients cancel
    as rationals before evaluation, and recomputes each end's asymptotics
    from the face map (the next surviving order leads).  Two ends whose new
    leaders demand opposite signs of one functional force it to zero, as in
    ``basis_newton._face_growth``.

    Parameters
    ----------
    model : _BasisModel
        Compiled parent basis (``face`` empty).
    suppressed : tuple of str
        Infinite ends whose leader is fixed at zero.

    Returns
    -------
    tuple
        ``(face_model, free)``: the face basis and the parent coefficient
        indices of its coefficients (``phi = theta[free]``, exactly).

    Raises
    ------
    ValueError
        If an end is not an infinite end of the basis, or the face has no
        member with outward growth at some end.
    """
    if model.face:
        raise ValueError("a face model has no further faces")
    d = model.n_coefficients
    rows = []
    for side in suppressed:
        entry = model.side_asymptotics(side)
        if entry is None:
            raise ValueError(f"{side!r} is not an infinite end of the basis")
        rows.append(list(entry.leading.row_exact))
    if not rows:
        return model, tuple(range(d))
    for _ in range(d + 1):
        # One direction per free (non-pivot) column, unit there: phi is
        # exactly theta at the free columns.
        vectors = _nullspace(rows, d)
        pivots = set(_rref(rows, d)[1])
        free = tuple(c for c in range(d) if c not in pivots)
        face = _restrict_to(model, vectors, tuple(suppressed))
        lead = [face.side_asymptotics(s) for s in _SIDES]
        lead = [e.leading for e in lead if e is not None]
        forced = None
        if len(lead) == 2:
            a, b = lead[0].row_exact, lead[1].row_exact
            pairs = [(x, y) for x, y in zip(a, b) if x or y]
            ratio = pairs[0][1] / pairs[0][0] if pairs and pairs[0][0] else None
            if (
                lead[0].order == lead[1].order
                and ratio is not None
                and ratio < 0
                and all(y == ratio * x for x, y in pairs)
            ):
                forced = a
        if forced is None:
            for entry in face.asymptotics:
                if entry.leading.order <= _CONSTANT:
                    raise ValueError(
                        f"the leader face has no outward growth at the {entry.side} end"
                    )
            return face, free
        row = [Fraction(0)] * d
        for i, c in enumerate(free):
            row[c] = forced[i]
        rows.append(row)
    raise RuntimeError("internal error: leader face did not stabilize")


def _restrict_to(model, vectors, face, /):
    """The parent basis on ``theta = sum_f phi_f vectors[f]`` (exact).

    Parameters
    ----------
    model : _BasisModel
        Parent basis.
    vectors : list of list of Fraction
        Exact face directions in parent coefficients.
    face : tuple of str
        Suppressed ends (recorded on the result).
    """
    functions = model.functions
    exact = model.coefficient_map_exact
    n_slots = len(exact)
    k = len(vectors)
    face_exact = tuple(
        tuple(
            sum((exact[i][c] * v[c] for c in range(len(v)) if v[c]), Fraction(0))
            for v in vectors
        )
        for i in range(n_slots)
    )
    columns = []
    for f in range(k):
        full = [face_exact[i][f] for i in range(n_slots)]
        terms = tuple(
            (j, tuple(full[3 * j : 3 * j + 3]))
            for j in range(len(functions))
            if full[3 * j] or full[3 * j + 1] or full[3 * j + 2]
        )
        if not terms:
            raise RuntimeError("internal error: a face direction has no terms")
        nonneg = len(terms) == 1 and functions[terms[0][0]].kind == _LOG
        if nonneg != any(
            vectors[f][c] and model.columns[c].nonnegative
            for c in range(len(vectors[f]))
        ):
            raise ValueError("a leader face mixes an endpoint log with other columns")
        columns.append(_BasisColumn(terms, _column_label(terms, functions), nonneg))
    used = sorted({j for column in columns for j, _ in column.terms})
    asymptotics = tuple(
        _end_asymptotics(
            entry.side,
            model.request,
            model.geometry,
            functions,
            model.available,
            used,
            face_exact,
            None,
            entry.suppressed,
        )
        for entry in model.asymptotics
    )
    return dataclasses.replace(
        model,
        columns=tuple(columns),
        coefficient_map_exact=face_exact,
        coefficient_map=np.array(
            [[float(v) for v in row] for row in face_exact], dtype=np.float64
        ).reshape(n_slots, k),
        asymptotics=asymptotics,
        face=tuple(face),
    )


def _build_basis(
    spec,
    points=None,
    weights=None,
    policy=None,
    coordinate=None,
    /,
    *,
    adaptive_geometry=True,
):
    """Canonicalize a public spec, fix the coordinate and geometry, compile.

    Parameters
    ----------
    spec : BasisSpec
        Public basis configuration.
    points : numpy.ndarray or None
        Canonical representative data locations in user units, shape ``(R,)``
        (point samples or interval midpoints).
    weights : numpy.ndarray or None
        Nonnegative weights or mixture responsibilities, shape ``(R,)``.
    policy : _GeometryPolicy or None
        Automatic-geometry constants.
    coordinate : _FitCoordinate or None
        Previously fixed non-reflected coordinate (continuation, probes or
        interval-built coordinates); ``None`` builds one from ``points``.
    adaptive_geometry : bool
        Allow skew-aware defaults for full-line unweighted point data. Mixture
        responsibility fits disable this even for uniform responsibilities.

    Returns
    -------
    _BasisModel
        Compiled basis.

    Raises
    ------
    ValueError
        If neither a coordinate nor data are supplied.
    """
    request = _canonicalize_basis_spec(spec)
    if coordinate is None:
        if points is None or np.asarray(points).size == 0:
            raise ValueError(
                "a fitting coordinate requires data or an explicit coordinate"
            )
        coordinate = _build_fit_coordinate(
            (request.lower, request.upper),
            np.asarray(points, dtype=np.float64),
            weights,
            None,
        )
    geometry = _resolve_basis_geometry(
        request,
        coordinate,
        points,
        weights,
        policy,
        adaptive_geometry=adaptive_geometry,
    )
    return _compile_basis(request, geometry)
