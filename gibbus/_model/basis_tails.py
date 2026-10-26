"""Tail derivative bounds: where ``d^k q / d zeta^k > 0`` provably holds toward an end.

The leading tail behavior of a basis that mixes powers, logs, Gaussian gates
and exponentials is derived here from coefficient-linear asymptotic atoms.

**Infinite ends.**  Measure outward with ``zeta = +z`` (upper) or ``-z``
(lower), ``z = (x - c) / h``, so ``q_xx = q_zz / h**2`` and ``q_zz =
q_zeta zeta``.  Toward an end the own gate tends to ``-1`` and the opposite
gate to ``+1``, and every gate splits exactly into its limit and the erfc
deviations ``G = G_inf + alpha_own e_own - alpha_opp e_opp`` with
``e_o = erfc(a (zeta + o))``, ``o = -1`` for the own gate and ``+1`` for the
opposite one.  The deviation derivatives are exact Gaussians,

* ``e' = -(2 a / sqrt(pi)) g`` and ``e'' = (4 a**3 / sqrt(pi)) (zeta + o) g``
  with ``g = exp(-a**2 (zeta + o)**2)``,

and the deviation itself obeys the Mills bound ``0 < e <= g / (a sqrt(pi)
(zeta + o))``.  So ``d^k q / d zeta^k`` is a finite sum of *main atoms*
``(row . theta) w(zeta)`` with ``w = zeta**b exp(r zeta - kappa zeta**2 +
c)`` (limit derivatives of monomials, soft-power leaders and exponentials,
opposite-endpoint logarithms, and the Gaussian gate terms with
``kappa = a**2``), plus a remainder bounded by *envelope atoms*
``|theta_c| C w(zeta)`` (erfc value terms and soft-power deviations from
``zeta**p``), all valid for ``zeta >= zeta_min``.  The constant ``c`` stays in
log form, so factors such as ``exp(-lambda)`` never underflow a row.

Soft powers use, on ``zeta >= 2`` with ``x = 1 / zeta`` and ``M(q) = max(1,
2**-q)`` (a bound of ``y**q`` on ``[1/2, 1]``) and ``A = R - 1`` with
``zeta - 1 <= A <= zeta``:

* ``|A**p - zeta**p| <= p M(p - 1) zeta**(p-1)``,
* ``|(A**p)' - p zeta**(p-1)| <= p (|p - 1| M(p - 2) + 3/32) zeta**(p-2)``,
* ``|(A**p)'' - p (p-1) zeta**(p-2)| <= (p |p - 1| (|p - 2| M(p - 3) + 3/16)
  + 3 p M(p - 1) / 8) zeta**(p-3)``,

from ``A' = (1 + x**4)**(-3/4)`` and ``0 <= A'' <= 3 zeta**-5``.

For a coefficient vector the strongest atom with a nonzero value leads.  A
negative leader is a tail violation; its row gives a coefficient-linear cut
``row . theta >= 0``.  A positive leader is compared with every other
non-positive or uncertain main atom and every envelope atom: once each ratio
to the leader is nonincreasing beyond ``R`` and their sum at ``R`` is below
the leader, the sign holds on ``zeta >= R``.  ``R`` doubles until that holds or
a cap is reached (then the tail is unverified -- never assumed fine).

**Finite ends with a logarithm** ``gamma > 0``: on ``(L, L + delta]`` the
curvature is ``gamma / d**2`` plus the rest of ``q''``, enclosed by interval
arithmetic on ``[L, L + delta0]``; ``delta`` is chosen so the total stays at
or above the cut threshold there.

All arithmetic is binary64 with explicit rounding slack: numerical evidence,
not an arbitrary-precision certificate.
"""

import dataclasses
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .._defaults import BASIS_GATE_SHARPNESS
from ._basis_kernels import TailKernel, endpoint_width, tail_result
from .basis import _EXPONENTIAL, _LOG, _LOWER, _MONOMIAL, _UPPER

_ZETA_MIN = 2.0  # soft-power bounds and the own-gate Mills bound hold beyond


@dataclass(frozen=True)
class _MainAtom:
    """Coefficient-linear atom ``(row . theta) w(zeta)`` of ``d^k q / d zeta^k``.

    ``w = zeta**power exp(rate zeta - kappa zeta**2 + const)`` for
    ``kind == "power"``; for ``kind == "log"`` the exact term is ``(row .
    theta) / (zeta + offset)**k`` up to sign (an opposite-endpoint logarithm,
    coefficient constrained ``>= 0``).
    """

    kind: str
    power: float
    rate: float
    row: np.ndarray
    offset: float = 0.0
    cut: np.ndarray | None = None
    kappa: float = 0.0
    const: float = 0.0


@dataclass(frozen=True)
class _Envelope:
    """Remainder bound ``|theta_column| coef zeta**power exp(rate zeta -
    kappa zeta**2 + const)``."""

    column: int
    coef: float
    power: float
    rate: float
    kappa: float = 0.0
    const: float = 0.0


@dataclass(frozen=True)
class _TailModel:
    """Theta-independent tail atoms of one infinite end.

    Parameters
    ----------
    side : str
        ``"lower"`` or ``"upper"``.
    sign : float
        ``zeta = sign * z``.
    zeta_min : float
        Start of the validity range of all atoms and envelopes.
    atoms : tuple of _MainAtom
        Main atoms, strongest first.
    envelopes : tuple of _Envelope
        Remainder envelopes.
    order : int
        Derivative order ``k``.
    center, scale : float
        Fitting coordinate ``c``, ``h`` (user-unit cutoffs).
    n_coefficients : int
        Length ``d`` of the atom rows.
    """

    side: str
    sign: float
    zeta_min: float
    atoms: tuple
    envelopes: tuple
    order: int = 2
    center: float = 0.0
    scale: float = 1.0
    n_coefficients: int = 0

    @cached_property
    def kernel(self):
        """Compiled atoms (``_basis_kernels.TailKernel``), built once."""
        return TailKernel(self, self.center, self.scale, self.n_coefficients)


@dataclass(frozen=True)
class _TailResult:
    """Tail verdict for one infinite end and one coefficient vector.

    Parameters
    ----------
    side : str
        ``"lower"`` or ``"upper"``.
    status : str
        ``"verified"`` (the derivative is positive beyond the cutoff, or
        identically zero when no term survives there), ``"violated"`` (the
        leading term is negative far out) or ``"unverified"``.
    cutoff : float or None
        Cutoff in user units when verified.
    zeta : float or None
        Cutoff in outward fitting units.
    lead_value : float or None
        Value of the leading atom's functional.
    cut_row : numpy.ndarray or None
        For a violation, the leading functional: ``cut_row . theta >= 0`` is
        necessary for the sign far out.
    lead_power, lead_rate, lead_kappa, lead_const : float or None
        Weight of the leading atom.
    margin : float or None
        Guaranteed fraction of the leader (see ``bound``).
    """

    side: str
    status: str
    cutoff: float | None
    zeta: float | None
    lead_value: float | None
    cut_row: np.ndarray | None
    lead_power: float | None = None
    lead_rate: float | None = None
    lead_kappa: float | None = None
    lead_const: float | None = None
    margin: float | None = None

    def bound(self, zeta, /):
        """Guaranteed lower bound of the derivative at ``zeta >= self.zeta``.

        ``margin * w(zeta)`` with the leader's weight ``w`` (internal units);
        valid because every competing ratio is nonincreasing beyond the
        cutoff.  Zero when no leader exists (identically zero derivative).

        Parameters
        ----------
        zeta : float
            Tail coordinate, at least ``self.zeta``.
        """
        if self.margin is None:
            return 0.0
        log_w = (
            self.lead_power * math.log(zeta)
            + self.lead_rate * zeta
            - self.lead_kappa * zeta * zeta
            + self.lead_const
        )
        if log_w > 700.0:
            return math.inf
        return self.margin * math.exp(log_w)


def _falling(n, k, /):
    """Falling factorial ``n (n-1) ... (n-k+1)``.

    Parameters
    ----------
    n : float
        Top factor.
    k : int
        Number of factors.
    """
    out = 1.0
    for i in range(k):
        out *= n - i
    return out


def _unit_bound(q, /):
    """``max(1, 2**-q)``: a bound of ``y**q`` for ``y`` in ``[1/2, 1]``.

    Parameters
    ----------
    q : float
        Exponent.
    """
    return max(1.0, 2.0 ** (-q))


def _soft_power_jets(p, /):
    """Exact leaders and envelope constants of ``A(zeta)**p`` derivatives.

    Returns ``[(lead, envelope), ...]`` for orders ``0, 1, 2`` with
    ``|f^(i) - lead zeta**(p-i)| <= envelope zeta**(p-i-1)`` on ``zeta >= 2``.

    Parameters
    ----------
    p : float
        Soft-power exponent (``> 1``).
    """
    m = _unit_bound
    return [
        (1.0, p * m(p - 1.0)),
        (p, p * (abs(p - 1.0) * m(p - 2.0) + 3.0 / 32.0)),
        (
            p * (p - 1.0),
            p * abs(p - 1.0) * (abs(p - 2.0) * m(p - 3.0) + 3.0 / 16.0)
            + 3.0 * p * m(p - 1.0) / 8.0,
        ),
    ]


def _function_jets(fn, sgn, k, /):
    """Derivatives ``f^(i)``, ``i <= k``, of one function along ``zeta``.

    Returns per order ``(exact, envelopes)``: ``exact`` a list of terms
    ``(coef, power, rate, const)`` (``coef zeta**power exp(rate zeta +
    const)``) and ``envelopes`` a list of nonnegative bounds of the same form
    on the remainder.

    Parameters
    ----------
    fn : _BasisFunction
        Monomial, soft power or exponential.
    sgn : float
        ``+1`` (upper end) or ``-1`` (lower end).
    k : int
        Highest order.
    """
    out = []
    for i in range(k + 1):
        if fn.kind == _MONOMIAL:
            n = fn.degree
            coef = sgn**n * _falling(n, i)
            out.append(([(coef, float(n - i), 0.0, 0.0)] if coef else [], []))
        elif fn.kind == _EXPONENTIAL:
            lam = fn.rate
            # Own side: exp(lam (zeta - 1)); opposite side: exp(-lam (zeta + 1)).
            rate = lam if (fn.side == _UPPER) == (sgn > 0) else -lam
            out.append(([(rate**i, 0.0, rate, -lam)], []))
        else:
            lead, env = _soft_power_jets(fn.power)[i]
            p = fn.power
            out.append(
                (
                    [(lead, p - i, 0.0, 0.0)] if lead else [],
                    [(env, p - i - 1.0, 0.0, 0.0)],
                )
            )
    return out


def _gaussian_jets(o, k, /):
    """Exact ``e^(j)``, ``1 <= j <= k``, and the Mills envelope of ``e``.

    ``e = erfc(a (zeta + o))``; ``e^(j)`` is returned as terms ``(coef, power,
    rate, kappa, const)``.  The envelope of ``e`` itself is
    ``g / (a sqrt(pi) (zeta + o)) <= g / (a sqrt(pi) zeta (1 - 1/zeta_min))``.

    Parameters
    ----------
    o : int
        ``-1`` (own gate) or ``+1`` (opposite gate).
    k : int
        Highest order.
    """
    a = BASIS_GATE_SHARPNESS / math.sqrt(2.0)
    a2 = a * a
    g = (-2.0 * a2 * o, a2, -a2)  # rate, kappa, const of g
    root_pi = math.sqrt(math.pi)
    jets = {1: [(-2.0 * a / root_pi, 0.0, *g)]}
    jets[2] = [(4.0 * a**3 / root_pi, 1.0, *g), (4.0 * a**3 / root_pi * o, 0.0, *g)]
    shrink = 1.0 - 1.0 / _ZETA_MIN if o < 0 else 1.0
    value_env = (1.0 / (a * root_pi * shrink), -1.0, *g)
    return {j: jets[j] for j in range(1, k + 1)}, value_env


def _build_tail_model(model, side, /, order=2):
    """Precompute main atoms and envelopes of ``d^k q / d zeta^k`` at an infinite end.

    ``order = 2`` gives the curvature; ``order = 1`` gives the outward slope
    used for properness and normalizer tail bounds.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    side : str
        ``"lower"`` or ``"upper"`` (must be infinite).
    order : int
        Derivative order ``k`` (1 or 2).
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    k = order
    geometry, request = model.geometry, model.request
    h, c = geometry.scale, geometry.center
    d = model.n_coefficients
    sgn = 1.0 if side == _UPPER else -1.0
    own_slot, opp_slot = (2, 1) if side == _UPPER else (1, 2)
    has_own = geometry.has_upper_gate if side == _UPPER else geometry.has_lower_gate
    has_opp = geometry.has_lower_gate if side == _UPPER else geometry.has_upper_gate
    # (slot, sign of the deviation, gate offset o)
    gates = [
        g
        for g, ok in (((own_slot, 1.0, -1), has_own), ((opp_slot, -1.0, 1), has_opp))
        if ok
    ]

    log_offsets = {}
    for fn in model.functions:
        if fn.kind == _LOG and fn.active:
            log_offsets[fn.index] = (
                -(request.lower - c) / h
                if fn.side == _LOWER
                else (request.upper - c) / h
            )
    zeta_min = max(
        [_ZETA_MIN, *(2.0 * abs(o) + 1.0 for o in log_offsets.values() if o < 0.0)]
    )

    atoms = {}  # (kind, power, rate, kappa, offset) -> {const: row}
    envelopes = []

    def add_atom(key, const, col, value):
        rows = atoms.setdefault(key, {})
        row = rows.setdefault(const, np.zeros(d))
        row[col] += value

    for col, column in enumerate(model.columns):
        for j, gate in column.terms:
            fn = model.functions[j]
            g_inf = float(
                gate[0] - gate[1] + gate[2]
                if side == _LOWER
                else gate[0] + gate[1] - gate[2]
            )
            if fn.kind == _LOG:
                if g_inf != 0.0:
                    # f = -log(h (zeta + offset)): f' = -1/(.), f'' = +1/(.)^2.
                    value = -g_inf if k == 1 else g_inf
                    add_atom(
                        ("log", -float(k), 0.0, 0.0, log_offsets[j]), 0.0, col, value
                    )
                continue
            jets = _function_jets(fn, sgn, k)
            # --- limit part: f^(k) G_inf ------------------------------------
            if g_inf != 0.0:
                exact, env = jets[k]
                for coef, b, r, cst in exact:
                    add_atom(("power", b, r, 0.0, 0.0), cst, col, g_inf * coef)
                for coef, b, r, cst in env:
                    envelopes.append(_Envelope(col, abs(g_inf) * coef, b, r, 0.0, cst))
            # --- gate deviations: sum_i C(k, i) f^(i) (+-alpha e)^(k - i) -----
            for slot, dev_sign, o in gates:
                alpha = float(gate[slot])
                if alpha == 0.0:
                    continue
                weight = dev_sign * alpha
                e_jets, value_env = _gaussian_jets(o, k)
                for i in range(k + 1):
                    exact, env = jets[i]
                    binom = math.comb(k, i)
                    if i < k:
                        for ce, be, re, ke, cse in e_jets[k - i]:
                            for coef, b, r, cst in exact:
                                add_atom(
                                    ("power", b + be, r + re, ke, 0.0),
                                    cst + cse,
                                    col,
                                    weight * binom * coef * ce,
                                )
                            for coef, b, r, cst in env:
                                envelopes.append(
                                    _Envelope(
                                        col,
                                        abs(alpha * binom * coef * ce),
                                        b + be,
                                        r + re,
                                        ke,
                                        cst + cse,
                                    )
                                )
                    else:
                        ve, vb, vr, vk, vc = value_env
                        for coef, b, r, cst in exact + env:
                            envelopes.append(
                                _Envelope(
                                    col,
                                    abs(alpha * coef) * ve,
                                    b + vb,
                                    r + vr,
                                    vk,
                                    cst + vc,
                                )
                            )

    merged = []
    for (kind, power, rate, kappa, offset), rows in atoms.items():
        top = max(rows)
        row = sum(math.exp(cst - top) * r for cst, r in rows.items())
        if np.any(row != 0.0):
            merged.append(_MainAtom(kind, power, rate, row, offset, None, kappa, top))
    ordered = sorted(merged, key=lambda a: (a.kappa, -a.rate, -a.power))
    return _TailModel(side, sgn, zeta_min, tuple(ordered), tuple(envelopes), k, c, h, d)


def _tail_verdict(tail, model, theta, /, zeta_cap=1e8, dominance=0.0, growth=False):
    """Decide the sign of ``d^k q / d zeta^k`` beyond a cutoff, for one ``theta``.

    The first nonzero atom decides a violation.  Any positive atom may lead
    the proof when every stronger atom is a nonnegative contribution (they
    only add to the derivative); among those leads the one with the smallest
    proven cutoff is used.  A tiny but positive leading amplitude (a fit
    approaching the boundary where it vanishes) would otherwise have to
    dominate the remainder alone, pushing the cutoff to ``|zeta| ~ 1e11``
    where the cancellation of large columns leaves enclosures that never
    narrow, while the next atom (e.g. a robust linear slope) proves the same
    sign a few scales out.  For a lead, the cutoff ``R`` doubles from
    the largest crossover of the competing ratios until their sum is below
    the leader (``_basis_kernels.tail_verdict``).

    Parameters
    ----------
    tail : _TailModel
        Precomputed atoms.
    model : _BasisModel
        Compiled basis (unused; kept for the call shape of the verdicts).
    theta : numpy.ndarray
        Coefficients, shape ``(d,)``.
    zeta_cap : float
        Largest cutoff tried (fitting units).
    dominance : float
        Required fraction of the leader left after subtracting every competing
        bound, in ``[0, 1)``: the verdict then guarantees
        ``d^k q / d zeta^k >= margin * w(zeta)`` beyond the cutoff with
        ``margin >= dominance * lead`` (``w`` the leader's weight).
    growth : bool
        Properness use: later leads must be non-decaying (a decaying lead
        proves no outward growth).

    Returns
    -------
    _TailResult
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    status, cutoff, zeta, lead, power, rate, kappa, const, margin, atom = tail_result(
        tail.kernel, theta, float(zeta_cap), float(dominance), bool(growth)
    )
    cut = None
    if status == "violated":
        a = tail.atoms[atom]
        cut = (a.row if a.cut is None else a.cut).copy()
    return _TailResult(
        tail.side,
        status,
        cutoff,
        zeta,
        lead,
        cut,
        lead_power=power,
        lead_rate=rate,
        lead_kappa=kappa,
        lead_const=const,
        margin=margin,
    )


def _endpoint_exclusion(model, theta, side, threshold, /):
    """Width of the neighborhood of a finite endpoint where ``q'' >= threshold``.

    Only an endpoint with an enabled logarithm whose coefficient ``gamma`` is
    positive is excluded; otherwise the width is ``0`` and the endpoint is an
    ordinary point of the closed search interval.  On ``(L, L + delta]`` the
    curvature is ``gamma / d**2`` plus the rest of ``q''``, whose interval
    enclosure on ``[L, L + delta0]`` (``delta0 = min(h, span / 4)``) fixes
    ``delta = min(delta0, sqrt(gamma / (threshold - rest)))``
    (``_basis_kernels.endpoint_exclusion``).

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Coefficients.
    side : str
        ``"lower"`` or ``"upper"`` (finite).
    threshold : float
        Curvature level (user units) the excluded region must respect.

    Returns
    -------
    float
        Width ``delta >= 0`` in user units.
    """
    theta = np.ascontiguousarray(theta, dtype=np.float64)
    return float(endpoint_width(model.kernel, theta, side == _UPPER, float(threshold)))


@dataclass(frozen=True)
class _GrowthResult:
    """Outward growth of ``q`` toward one infinite end (properness).

    Parameters
    ----------
    side : str
        ``"lower"`` or ``"upper"``.
    status : str
        ``"proper"`` (``dq/dzeta >= slope.bound(zeta) > 0``, nondecreasing,
        beyond the cutoff: at least linear growth), ``"improper"`` (the
        leading slope term is negative, decays, or vanishes identically, so
        ``e^{-q}`` is not integrable toward this end) or ``"unverified"``.
    slope : _TailResult
        First-derivative verdict carrying the cutoff and the bound.
    """

    side: str
    status: str
    slope: object

    @property
    def zeta(self):
        """Cutoff in outward fitting units (``proper`` only)."""
        return self.slope.zeta


def _growth_verdict(tail, model, theta, /, zeta_cap=1e8):
    """Decide properness toward an infinite end from first-derivative atoms.

    Parameters
    ----------
    tail : _TailModel
        Order-1 tail model of this end.
    model : _BasisModel
        Compiled basis.
    theta : numpy.ndarray
        Coefficients.
    zeta_cap : float
        Largest cutoff tried.

    Returns
    -------
    _GrowthResult
    """
    if tail.order != 1:
        raise ValueError("properness needs an order-1 tail model")
    res = _tail_verdict(
        tail, model, theta, zeta_cap=zeta_cap, dominance=0.5, growth=True
    )
    if res.status == "violated":
        return _GrowthResult(tail.side, "improper", res)
    decaying = res.lead_power is not None and (
        res.lead_kappa > 0.0
        or res.lead_rate < 0.0
        or (res.lead_rate == 0.0 and res.lead_power < 0.0)
    )
    if decaying:
        # A positive but decaying leading slope (all else weaker) is integrable:
        # q tends to a finite limit whether or not a cutoff was established.
        return _GrowthResult(tail.side, "improper", res)
    if res.status == "unverified":
        return _GrowthResult(tail.side, "unverified", res)
    if res.margin is None:
        # Identically zero slope: q tends to a finite limit.
        return _GrowthResult(tail.side, "improper", res)
    return _GrowthResult(tail.side, "proper", res)


def _tail_on_face(tail, basis, /):
    """Tail atoms on a face ``theta = N phi`` of the coefficient space.

    An atom whose functional vanishes on the face is identically zero there
    and is dropped (in floating point a lifted or projected ``theta`` would
    leave a rounding-level value that the sign test must call undecided).
    The others are evaluated through their projection ``N N^T row``, exact on
    the face; a violation still reports the original row as its cut, which
    is valid on every face.

    Parameters
    ----------
    tail : _TailModel
        Tail atoms of the full coefficient space.
    basis : numpy.ndarray
        Orthonormal face basis ``N``, shape ``(d, k)``.

    Returns
    -------
    _TailModel
    """
    atoms = []
    for atom in tail.atoms:
        reduced = atom.row @ basis
        if np.linalg.norm(reduced) <= 1e-12 * np.linalg.norm(atom.row):
            continue
        atoms.append(
            dataclasses.replace(
                atom,
                row=basis @ reduced,
                cut=atom.row if atom.cut is None else atom.cut,
            )
        )
    return dataclasses.replace(tail, atoms=tuple(atoms))
