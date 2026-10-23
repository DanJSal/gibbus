"""Tail curvature bounds: where ``q'' > 0`` provably holds toward an end.

G1 owner replaced: G1 decides tail behavior from the leading polynomial
coefficient (even degree, positive sign) inside its cone description; G3 tails
mix powers, logs, gates and exponentials, so the leading behavior is derived
here from coefficient-linear asymptotic atoms.

**Infinite ends.**  Measure outward with ``zeta = +z`` (upper) or ``-z``
(lower), ``z = (x - c) / h``, so ``q_xx = q_zz / h**2`` and ``q_zz =
q_zeta zeta``.  Every term ``f G`` of a column splits into its limit part
``f G_inf`` and the gate deviation ``f (G - G_inf)``.  With
``E = exp(-rho (zeta - zeta_g))`` per tanh gate (``rho = 2 alpha h``), the
deviation of each gate is ``sigma alpha D(E)``, ``D = 2E / (1 + E)``, and
exactly

* ``D = 2E + R0``, ``D' = -2 rho E + R1``, ``D'' = 2 rho**2 E + R2`` with
  ``|R0| <= 2E**2``, ``|R1| <= 6 rho E**2``, ``|R2| <= 8 rho**2 E**2`` for
  ``zeta >= zeta_g`` (``0 < E <= 1``).

So ``q_zz`` is a finite sum of *main atoms* ``(row . theta) zeta**b e^{r zeta}``
(limit curvatures of monomials, soft-power leaders and exponentials,
opposite-endpoint logarithms, and the first-order gate terms
``2 sigma alpha E (f'' - 2 rho f' + rho**2 f)``), plus a remainder bounded by
*envelope atoms* ``|theta_c| C zeta**b e^{r zeta}`` (second-order gate terms,
soft-power deviations from ``zeta**p``), all valid for ``zeta >= zeta_min``.

For a coefficient vector the strongest atom with a nonzero value leads.  A
negative leader is a tail violation; its row gives a coefficient-linear cut
``row . theta >= 0``.  A positive leader is compared with every other
non-positive or uncertain main atom and every envelope atom: once each ratio
to the leader is nonincreasing beyond ``R`` and their sum at ``R`` is below
the leader, ``q'' > 0`` on ``zeta >= R``.  ``R`` doubles until that holds or
a cap is reached (then the tail is inconclusive -- never assumed fine).

**Finite ends with a logarithm** ``gamma > 0``: on ``(L, L + delta]`` the
curvature is ``gamma / d**2`` plus the rest of ``q''``, enclosed by interval
arithmetic on ``[L, L + delta0]``; ``delta`` is chosen so the total stays at
or above the cut threshold there.

All arithmetic is binary64 with explicit rounding slack: numerical evidence,
not an arbitrary-precision certificate (G3-028).
"""

import dataclasses
import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ._basis_kernels import TailKernel, endpoint_width, tail_result
from .basis import _EXPONENTIAL, _LOG, _LOWER, _MONOMIAL, _UPPER

_U = np.finfo(np.float64).eps / 2.0
_LOG_UNDERFLOW = -575.0  # leader weights below exp(-575) ~ 1e-250 are not trusted


@dataclass(frozen=True)
class _MainAtom:
    """Coefficient-linear atom ``(row . theta) w(zeta)`` of ``q_zz``.

    ``w = zeta**power exp(rate zeta)`` for ``kind == "power"``; for
    ``kind == "log"`` the exact term is ``(row . theta) / (zeta + offset)**2``
    (an opposite-endpoint logarithm, coefficient constrained ``>= 0``).
    """

    kind: str
    power: float
    rate: float
    row: np.ndarray
    offset: float = 0.0
    cut: np.ndarray | None = None


@dataclass(frozen=True)
class _Envelope:
    """Remainder bound ``|theta_column| * coef * zeta**power exp(rate zeta)``."""

    column: int
    coef: float
    power: float
    rate: float


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
        ``"nonnegative"`` (``q'' > 0`` beyond the cutoff, or identically zero
        when no term survives there), ``"violated"`` (the
        leading term is negative: ``q'' < 0`` far out) or ``"inconclusive"``.
    cutoff : float or None
        Cutoff in user units (``q'' >= 0`` beyond it) when nonnegative.
    zeta : float or None
        Cutoff in outward fitting units.
    lead_value : float or None
        Value of the leading atom's functional.
    cut_row : numpy.ndarray or None
        For a violation, the leading functional: ``cut_row . theta >= 0`` is
        necessary for ``q'' >= 0`` far out.
    """

    side: str
    status: str
    cutoff: float | None
    zeta: float | None
    lead_value: float | None
    cut_row: np.ndarray | None
    lead_power: float | None = None
    lead_rate: float | None = None
    margin: float | None = None

    def bound(self, zeta, /):
        """Guaranteed lower bound of the derivative at ``zeta >= self.zeta``.

        ``margin * zeta**lead_power * exp(lead_rate * zeta)`` (internal units);
        valid because every competing ratio is nonincreasing beyond the
        cutoff.  Zero when no leader exists (identically zero derivative).

        Parameters
        ----------
        zeta : float or numpy.ndarray
            Tail coordinate, at least ``self.zeta``.
        """
        if self.margin is None:
            return 0.0
        log_w = self.lead_power * math.log(zeta) + self.lead_rate * zeta
        if log_w > 700.0:
            return math.inf
        return self.margin * math.exp(log_w)


def _soft_constants(power, dz, zeta_min, /):
    """Bounds for ``f = (zeta**2 + dz)**(p/2)`` on ``zeta >= zeta_min``.

    Returns ``(F0, F1, F2, DEV_F, DEV_F2)`` with ``|f| <= F0 zeta**p``,
    ``|f'| <= F1 zeta**(p-1)``, ``|f''| <= F2 zeta**(p-2)``,
    ``|f - zeta**p| <= DEV_F zeta**(p-2)`` and
    ``|f'' - p(p-1) zeta**(p-2)| <= DEV_F2 zeta**(p-4)``.

    Parameters
    ----------
    power : float
        Soft-power exponent ``p``.
    dz : float
        Smoothing offset ``dz`` of ``(zeta**2 + dz)``.
    zeta_min : float
        Smallest tail coordinate.
    """
    a = 0.5 * power
    eps = dz / zeta_min**2
    g = 1.0 + eps
    f0 = g ** max(a, 0.0)
    f1 = power * g ** max(a - 1.0, 0.0)
    f2 = power * (power - 1.0 + eps) * g ** max(a - 2.0, 0.0)
    dev_f = a * dz * g ** max(a - 1.0, 0.0)
    dphi = abs(a - 2.0) * g ** max(a - 3.0, 0.0) * (power - 1.0 + eps) + g ** max(
        a - 2.0, 0.0
    )
    dev_f2 = power * dphi * dz
    return f0, f1, f2, dev_f, dev_f2


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


# Second-order gate remainder constants: |(f R)^(k)| <= |alpha| E^2 sum_i K[k][i] |f^(i)|
# from |R0| <= 2E^2, |R1| <= 6 rho E^2, |R2| <= 8 rho^2 E^2 (powers of rho added below).
_REMAINDER = {1: (6.0, 2.0), 2: (8.0, 12.0, 2.0)}
_REMAINDER_RHO_POWER = {1: (1, 0), 2: (2, 1, 0)}


def _build_tail_model(model, side, /, order=2):
    """Precompute main atoms and envelopes of ``d^k q / d zeta^k`` at an infinite end.

    ``order = 2`` gives the curvature (Stage 2); ``order = 1`` gives the
    outward slope used for properness and normalizer tail bounds (Stage 3).
    The gate deviation of each tanh enters exactly to first order through
    ``(d/dzeta - rho)^k f`` and is bounded to second order.

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

    gates = []  # (slot offset, sigma, rho, zeta_g)
    if geometry.lower_location is not None:
        zl = (geometry.lower_location - c) / h
        rho = 2.0 * geometry.lower_steepness * h
        gates.append((1, 1.0 if side == _LOWER else -1.0, rho, sgn * zl))
    if geometry.upper_location is not None:
        zu = (geometry.upper_location - c) / h
        rho = 2.0 * geometry.upper_steepness * h
        gates.append((2, -1.0 if side == _LOWER else 1.0, rho, sgn * zu))

    soft_dz = [
        fn.delta / h**2
        for fn in model.functions
        if fn.kind == "soft_power" and fn.active
    ]
    log_offsets = []
    for fn in model.functions:
        if fn.kind == _LOG and fn.active:
            log_offsets.append(
                -(request.lower - c) / h
                if fn.side == _LOWER
                else (request.upper - c) / h
            )
    zeta_min = max(
        [
            1.0,
            *(zg + 1.0 for _, _, _, zg in gates),
            *(2.0 * math.sqrt(v) for v in soft_dz),
            *(2.0 * abs(o) + 1.0 for o in log_offsets if o < 0.0),
        ]
    )

    atoms = {}  # (kind, power, rate, offset) -> row
    envelopes = []

    def add_atom(key, col, value):
        row = atoms.setdefault(key, np.zeros(d))
        row[col] += value

    for col, column in enumerate(model.columns):
        for j, gate in column.terms:
            fn = model.functions[j]
            g_inf = float(
                gate[0] - gate[1] + gate[2]
                if side == _LOWER
                else gate[0] + gate[1] - gate[2]
            )
            # --- limit part: f^(k) * G_inf ------------------------------------
            if fn.kind == _LOG:
                if g_inf != 0.0:
                    # f = -log(h (zeta + offset)): f' = -1/(.), f'' = +1/(.)^2.
                    offset = (
                        -(request.lower - c) / h
                        if fn.side == _LOWER
                        else (request.upper - c) / h
                    )
                    value = -g_inf if k == 1 else g_inf
                    add_atom(("log", -float(k), 0.0, offset), col, value)
                continue
            if fn.kind == _MONOMIAL:
                n = fn.degree
                par = sgn**n
                if n >= k and g_inf != 0.0:
                    add_atom(
                        ("power", float(n - k), 0.0, 0.0),
                        col,
                        g_inf * par * _falling(n, k),
                    )
                # |f^(i)| = falling(n, i) zeta^(n-i)
                f_abs = [(_falling(n, i), float(n - i)) for i in range(k + 1)]
            elif fn.kind == _EXPONENTIAL:
                lam = fn.rate * h
                zloc = (fn.location - c) / h
                if fn.side == _UPPER:  # exp(lam (z - zloc)) = K exp(r zeta)
                    r_f, log_k = sgn * lam, -lam * zloc
                else:
                    r_f, log_k = -sgn * lam, lam * zloc
                k_f = math.exp(log_k)
                if g_inf != 0.0:
                    add_atom(("power", 0.0, r_f, 0.0), col, g_inf * r_f**k * k_f)
            else:  # soft power
                p = fn.power
                dz = fn.delta / h**2
                f0, f1, f2, dev_f, dev_f2 = _soft_constants(p, dz, zeta_min)
                a = 0.5 * p
                eps = dz / zeta_min**2
                dev_f1 = p * abs(a - 1.0) * dz * (1.0 + eps) ** max(a - 2.0, 0.0)
                soft_abs = (f0, f1, f2)
                if g_inf != 0.0:
                    if k == 1:
                        add_atom(("power", p - 1.0, 0.0, 0.0), col, g_inf * p)
                        envelopes.append(
                            _Envelope(col, abs(g_inf) * dev_f1, p - 3.0, 0.0)
                        )
                    else:
                        add_atom(
                            ("power", p - 2.0, 0.0, 0.0), col, g_inf * p * (p - 1.0)
                        )
                        envelopes.append(
                            _Envelope(col, abs(g_inf) * dev_f2, p - 4.0, 0.0)
                        )
            # --- gate deviation parts -----------------------------------------
            for offset, sigma, rho, zeta_g in gates:
                alpha = float(gate[offset])
                if alpha == 0.0:
                    continue
                w = 2.0 * sigma * alpha * math.exp(rho * zeta_g)
                w2 = abs(alpha) * math.exp(2.0 * rho * zeta_g)
                consts = [
                    kk * rho**pp
                    for kk, pp in zip(
                        _REMAINDER[k], _REMAINDER_RHO_POWER[k], strict=True
                    )
                ]
                if fn.kind == _MONOMIAL:
                    n = fn.degree
                    par = sgn**n
                    # main: w E (d/dzeta - rho)^k (par zeta^n)
                    for i in range(min(k, n) + 1):
                        coef = math.comb(k, i) * (-rho) ** (k - i) * _falling(n, i)
                        add_atom(
                            ("power", float(n - i), -rho, 0.0), col, w * par * coef
                        )
                    for i in range(min(k, n) + 1):
                        cf, b = f_abs[i]
                        envelopes.append(
                            _Envelope(col, w2 * consts[i] * cf, b, -2.0 * rho)
                        )
                elif fn.kind == _EXPONENTIAL:
                    add_atom(
                        ("power", 0.0, r_f - rho, 0.0), col, w * k_f * (r_f - rho) ** k
                    )
                    rem = sum(consts[i] * abs(r_f) ** i for i in range(k + 1))
                    envelopes.append(
                        _Envelope(col, w2 * k_f * rem, 0.0, r_f - 2.0 * rho)
                    )
                else:  # soft power: leader zeta**p kept exact, the rest bounded
                    aw = abs(w)
                    add_atom(("power", p, -rho, 0.0), col, w * (-rho) ** k)
                    envelopes.append(_Envelope(col, aw * rho**k * dev_f, p - 2.0, -rho))
                    for i in range(1, k + 1):
                        envelopes.append(
                            _Envelope(
                                col,
                                aw * math.comb(k, i) * rho ** (k - i) * soft_abs[i],
                                p - i,
                                -rho,
                            )
                        )
                    rem = sum(consts[i] * soft_abs[i] for i in range(k + 1))
                    envelopes.append(_Envelope(col, w2 * rem, p, -2.0 * rho))

    ordered = sorted(
        (
            _MainAtom(kind, power, rate, row, offset)
            for (kind, power, rate, offset), row in atoms.items()
            if np.any(row != 0.0)
        ),
        key=lambda a: (-a.rate, -a.power),
    )
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
    sign a few scales out (G3-035).  For a lead, the cutoff ``R`` doubles from
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
    status, cutoff, zeta, lead, power, rate, margin, atom = tail_result(
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
        margin=margin,
    )


def _endpoint_exclusion(model, theta, side, threshold, /):
    """Width of the neighbourhood of a finite endpoint where ``q'' >= threshold``.

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
        ``e^{-q}`` is not integrable toward this end) or ``"inconclusive"``.
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
        res.lead_power < 0.0 or res.lead_rate < 0.0
    )
    if decaying:
        # A positive but decaying leading slope (all else weaker) is integrable:
        # q tends to a finite limit whether or not a cutoff was established.
        return _GrowthResult(tail.side, "improper", res)
    if res.status == "inconclusive":
        return _GrowthResult(tail.side, "inconclusive", res)
    if res.margin is None or res.lead_power < 0.0 or res.lead_rate < 0.0:
        # Identically zero or decaying slope: q tends to a finite limit.
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
