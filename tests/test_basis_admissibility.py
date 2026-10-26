"""Lower-order completion and exact tail curvature of the compiled basis.

Completion is checked as exact span closure, and the compiled family as
exactly affine equivariant in the data.  The tail-curvature expansion is
checked against an independent high-precision ``q''`` far out in linear
tails (analytic derivatives of the defining formulas; the algebraic part is
combined in exact rationals so exactly suppressed terms cancel exactly), and
its feasibility test against hand-derived cases.  An independent grid LP (own
evaluator, not used by the library) supplies numerical evidence that every
compiled battery family has a convex proper member on a +-8 grid in ``z``,
and that the rejected families have none even there.
"""

import math
import warnings
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from scipy.optimize import linprog
from scipy.special import erfc
from test_basis_construction import (
    BATTERY,
    NORMAL,
    _compile_quiet,
    _exact_matrix,
    _is_expected_rejection,
    _rank,
)

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._defaults import BASIS_GATE_SHARPNESS
from gibbus._model.basis import _build_basis, _cone_feasible, _nullspace

INF = math.inf
RNG = np.random.default_rng(1610202625)
VALID = [s for s in BATTERY if not _is_expected_rejection(s)]
GATE_RATE = BASIS_GATE_SHARPNESS / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Lower-order completion and equivariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", VALID, ids=lambda s: "")
def test_gated_power_terms_are_completed_by_lower_monomials(spec):
    model = _compile_quiet(spec)
    exact = _exact_matrix(model)  # slots x d
    n_slots, d = len(exact), model.n_coefficients
    span = [[exact[i][c] for i in range(n_slots)] for c in range(d)]
    rank = _rank(span)
    for column in model.columns:
        for j, gate in column.terms:
            fn = model.functions[j]
            if gate[1] == 0 and gate[2] == 0:
                continue
            if fn.kind == "monomial":
                top = fn.degree
            elif fn.kind == "soft_power":
                top = math.ceil(fn.power)
            else:
                continue
            for k in range(top):
                vec = [Fraction(0)] * n_slots
                vec[3 * k] = gate[0] if k > 0 else Fraction(0)
                vec[3 * k + 1], vec[3 * k + 2] = gate[1], gate[2]
                assert _rank([*span, vec]) == rank, (column.label, k)


@pytest.mark.parametrize(
    "tails",
    [
        (Polynomial(1), Polynomial(1)),
        (None, Polynomial(3)),
        (Polynomial(2), Polynomial(4)),
        (Exponential(0.7), Polynomial(1)),
        (SoftPower(1.5), SoftPower(2.5)),
    ],
    ids=lambda t: "",
)
@pytest.mark.parametrize("bulk", [2, 3])
def test_compiled_family_is_exactly_affine_equivariant(tails, bulk):
    """Shifted and rescaled data give the same canonical family exactly.

    The canonical coordinate absorbs the affine map, so every column, map and
    asymptotic row agrees exactly; only the user-unit exponential rates and
    physical rows scale."""
    spec = BasisSpec(bulk_degree=bulk, lower_tail=tails[0], upper_tail=tails[1])
    a = _compile_quiet(spec)
    lower = tails[0]
    if isinstance(lower, Exponential):
        lower = Exponential(lower.rate / 4.0)  # the same canonical rate
    scaled = BasisSpec(bulk_degree=bulk, lower_tail=lower, upper_tail=tails[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        b = _build_basis(scaled, 4.0 * NORMAL + 0.9)
    assert b.geometry.scale == pytest.approx(4.0 * a.geometry.scale, rel=1e-14)
    assert b.geometry.center == pytest.approx(4.0 * a.geometry.center + 0.9, rel=1e-14)
    assert [c.terms for c in a.columns] == [c.terms for c in b.columns]
    assert a.coefficient_map_exact == b.coefficient_map_exact
    for sa, sb in zip(a.asymptotics, b.asymptotics, strict=True):
        assert [t.row_exact for t in sa.terms] == [t.row_exact for t in sb.terms]
        assert [t.row_exact for t in sa.curvature] == [
            t.row_exact for t in sb.curvature
        ]


def test_cubic_bulk_with_linear_upper_tail_gates_the_cubic_toward_the_lower_end():
    # The cubic leads the lower end and is suppressed at the linear upper end.
    # Its Gaussian gate leaves only opposite-gate curvature in the upper tail,
    # which the own-gate rows of the remaining columns dominate, so minimal
    # keeps the single direction z^3 (-1 + tL) = -2 z^3 W_L.
    model = _build_basis(BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)), NORMAL)
    cubic = next(f.index for f in model.functions if f.label == "z^3")
    gates = [g for c in model.columns for j, g in c.terms if j == cubic]
    assert gates == [(-1, 1, 0)]
    assert _grid_lp(model, 8.0, 1601) is not None


# ---------------------------------------------------------------------------
# Exact tail curvature
# ---------------------------------------------------------------------------


def _mp(value):
    """Exact ``mpf`` of a Fraction or float."""
    if isinstance(value, (Fraction, int)):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _oracle_d2q(model, theta, z):
    """High-precision ``q''(z)`` in canonical units from the defining formulas.

    Each gate is split exactly into its limit at the end ``z`` lies toward and
    an ``erfc`` deviation; the limit parts combine in exact rationals, so a
    suppressed function contributes exactly zero there.
    """
    a = _mp(Fraction(BASIS_GATE_SHARPNESS)) / mp.sqrt(2)
    z = mp.mpf(z)
    sqrt_pi = mp.sqrt(mp.pi)
    lo = model.geometry.to_internal(model.request.lower)
    up = model.geometry.to_internal(model.request.upper)

    def f(fn):
        """``(f, f', f'')`` at ``z``."""
        if fn.kind == "monomial":
            k = fn.degree
            return (
                z**k,
                k * z ** (k - 1) if k else mp.mpf(0),
                k * (k - 1) * z ** (k - 2) if k > 1 else mp.mpf(0),
            )
        if fn.kind == "log":
            d = z - _mp(float(lo)) if fn.side == "lower" else _mp(float(up)) - z
            sign = 1 if fn.side == "lower" else -1
            return -mp.log(d), -sign / d, 1 / d**2
        if fn.kind == "exponential":
            lam = mp.mpf(fn.rate)
            if fn.side == "upper":
                e = mp.exp(lam * (z - 1))
                return e, lam * e, lam**2 * e
            e = mp.exp(-lam * (z + 1))
            return e, -lam * e, lam**2 * e
        p = mp.mpf(fn.power)
        r = 1 + z**4
        big_a = r ** mp.mpf(0.25) - 1
        d1 = z**3 * r ** mp.mpf(-0.75)
        d2 = 3 * z**2 * r ** mp.mpf(-1.75)
        return (
            big_a**p,
            p * big_a ** (p - 1) * d1,
            p * (p - 1) * big_a ** (p - 2) * d1**2 + p * big_a ** (p - 1) * d2,
        )

    def deviation(which):
        """``(g, g', g'')`` of ``t - t_limit`` for one gate (exact sign)."""
        # t_L = erf(a (1 + z)), t_U = erf(a (1 - z)).
        arg_sign = 1 if which == "L" else -1
        u = a * (1 + arg_sign * z)
        limit = 1 if u > 0 else -1
        # t - limit = -limit * erfc(|u|).
        g = -limit * mp.erfc(abs(u))
        dt = arg_sign * 2 * a / sqrt_pi * mp.exp(-(u**2))
        d2t = -2 * u * a * arg_sign * dt
        return limit, g, dt, d2t

    lim_l, gl, dl, d2l = deviation("L")
    lim_u, gu, du, d2u = deviation("U")
    total = mp.mpf(0)
    for coef, column in zip(theta, model.columns):
        c = _mp(coef)
        for j, gate in column.terms:
            v, v1, v2 = f(model.functions[j])
            limit = gate[0] + gate[1] * lim_l + gate[2] * lim_u  # exact rational
            total += c * _mp(limit) * v2
            for alpha, g, g1, g2 in ((gate[1], gl, dl, d2l), (gate[2], gu, du, d2u)):
                if alpha:
                    total += c * _mp(alpha) * (v2 * g + 2 * v1 * g1 + v * g2)
    return total


def _prefactor(term, zeta):
    """Exact gated prefactor function ``F(zeta)`` of one curvature term."""
    if term.rate:
        return mp.exp(-mp.mpf(term.rate) * (zeta + 1))
    order = mp.mpf(term.order)
    if float(term.order).is_integer():
        return zeta**order
    return ((1 + zeta**4) ** mp.mpf(0.25) - 1) ** order


def _predicted(side, offset, theta, zeta):
    """Leading-order ``q''`` of one gate group at ``zeta = |z|``.

    ``(F e)'' = F e'' (1 + O(1 / zeta))`` with ``e'' = 4 a**2 u exp(-u**2) /
    sqrt(pi)`` and the group's exact prefactor ``F``.
    """
    zeta = mp.mpf(zeta)
    a = _mp(Fraction(BASIS_GATE_SHARPNESS)) / mp.sqrt(2)
    u = a * (zeta + offset)
    prefactor = sum(
        (
            _mp(sum(r * t for r, t in zip(term.row_exact, theta)))
            * _prefactor(term, zeta)
            for term in side.curvature
            if term.shape == "gate" and term.offset == offset
        ),
        mp.mpf(0),
    )
    return prefactor * 4 * a**2 * u * mp.exp(-(u**2)) / mp.sqrt(mp.pi)


LINEAR_TAIL_SPECS = [
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=3, upper_tail=Polynomial(1), expansion="full"),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(3), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Exponential(0.7)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=SoftPower(1.5)),
    BasisSpec(
        bulk_degree=4,
        lower_tail=SoftPower(2.7),
        upper_tail=Polynomial(1),
        expansion="full",
    ),
    BasisSpec(support=(0.0, INF), bulk_degree=1),
    BasisSpec(support=(0.0, INF), bulk_degree=3, upper_tail=Polynomial(1)),
]


@pytest.mark.parametrize("spec", LINEAR_TAIL_SPECS, ids=lambda s: "")
def test_tail_curvature_expansion_matches_high_precision_q2(spec):
    """Every term of the expansion leads once the earlier ones vanish.

    For each ``k``, an exact rational ``theta`` is drawn from the nullspace of
    the first ``k`` curvature rows.  Far out, ``q''`` then has the sign of row
    ``k`` and approaches the leading form of its gate group, with an
    ``O(1 / zeta)`` relative error.  Once every own-gate row vanishes, the
    opposite gate is smaller by ``exp(4 a**2 zeta)``; it is checked at
    moderate ``zeta`` with enough digits to resolve that ratio.
    """
    model = _compile_quiet(spec)
    d = model.n_coefficients
    for side in model.asymptotics:
        if side.target_order != (1, 1.0, 0):
            assert side.curvature == ()
            continue
        sign = -1 if side.side == "lower" else 1
        rows = [list(t.row_exact) for t in side.curvature]
        if not rows:
            # No correction survives: q'' vanishes identically far out.
            theta = [Fraction(float(v)) for v in RNG.normal(size=d)]
            with mp.workdps(60):
                assert _oracle_d2q(model, theta, sign * 32.0) == 0
            continue
        for k, term in enumerate(side.curvature):
            if k:
                basis = _nullspace(rows[:k], d)
            else:
                basis = [[Fraction(int(i == c)) for i in range(d)] for c in range(d)]
            if not basis:
                break
            weights = [Fraction(float(w)) for w in RNG.normal(size=len(basis))]
            theta = [sum(w * v[c] for w, v in zip(weights, basis)) for c in range(d)]
            value = sum(r * t for r, t in zip(rows[k], theta))
            if value == 0:
                # Row k is implied zero by the earlier rows.
                continue
            zetas = (16.0, 64.0, 256.0) if term.offset == 1 else (32.0, 128.0, 512.0)
            digits = int(6 * zetas[-1]) + 60 if term.offset == 1 else 60
            ratios = []
            with mp.workdps(digits):
                for zeta in zetas:
                    d2 = _oracle_d2q(model, theta, sign * zeta)
                    if term.shape == "log":
                        end = model.request.lower if sign > 0 else model.request.upper
                        zero = mp.mpf(float(model.geometry.to_internal(end)))
                        expected = _mp(value) / (zeta - sign * zero) ** 2
                    else:
                        expected = _predicted(side, term.offset, theta, zeta)
                    ratios.append(float(d2 / expected))
            # The relative error decays at least like 1 / zeta (zeta grows 16x).
            errors = [abs(r - 1.0) for r in ratios]
            assert errors[-1] < 0.1, (side.side, k, ratios)
            assert errors[-1] <= 0.3 * errors[0] + 1e-40, (side.side, k, ratios)


def test_opposite_endpoint_log_dominates_linear_tail_curvature():
    model = _compile_quiet(
        BasisSpec(
            support=(-INF, 0.0), bulk_degree=2, lower_tail=Polynomial(1), log_upper=True
        )
    )
    (side,) = model.asymptotics
    assert side.curvature[0].shape == "log"
    assert [t.offset for t in side.curvature[1:]] == sorted(
        t.offset for t in side.curvature[1:]
    )
    theta = [Fraction(float(v)) for v in RNG.normal(size=model.n_coefficients)]
    gamma = float(sum(r * t for r, t in zip(side.curvature[0].row_exact, theta)))
    with mp.workdps(60):
        z = mp.mpf(-20.0)
        d2 = _oracle_d2q(model, theta, z)
        assert float(d2 / (gamma / z**2)) == pytest.approx(1.0, rel=1e-12)


def test_own_gate_terms_precede_the_opposite_gate():
    model = _compile_quiet(
        BasisSpec(
            bulk_degree=4,
            lower_tail=SoftPower(2.7),
            upper_tail=Polynomial(1),
            expansion="full",
        )
    )
    upper = model.side_asymptotics("upper")
    offsets = [t.offset for t in upper.curvature]
    assert offsets == sorted(offsets) and offsets[0] == -1 and offsets[-1] == 1
    # The lower soft power reaches the upper tail only through its own gate.
    other = [t for t in upper.curvature if t.offset == 1]
    assert 2.7 in [t.order for t in other]


@pytest.mark.parametrize("expansion", ["minimal", "full"])
@pytest.mark.parametrize(
    "tails",
    [(None, Polynomial(1)), (Polynomial(1), None), (Polynomial(1), Polynomial(1))],
    ids=lambda t: "",
)
def test_degree_one_linear_tails_are_rejected_exactly(tails, expansion):
    spec = BasisSpec(
        bulk_degree=1, lower_tail=tails[0], upper_tail=tails[1], expansion=expansion
    )
    with pytest.raises(ValueError, match="eventually nonnegative"):
        _build_basis(spec, NORMAL)


def test_cone_feasibility_small_cases():
    one, zero = Fraction(1), Fraction(0)
    # x > 0, y > 0 in R^2: feasible.
    assert _cone_feasible([], [[one, zero], [zero, one]], [])
    # x > 0 and -x > 0: infeasible.
    assert not _cone_feasible([], [[one, zero], [-one, zero]], [])
    # x - y = 0, x > 0, -y >= 0: infeasible (strictness tracked).
    assert not _cone_feasible([[one, -one]], [[one, zero]], [[zero, -one]])
    # x - y = 0, x > 0, y >= 0: feasible.
    assert _cone_feasible([[one, -one]], [[one, zero]], [[zero, one]])
    # Three variables, rank-deficient image: x + y > 0, -x - y >= 0: infeasible.
    assert not _cone_feasible([], [[one, one, zero]], [[-one, -one, zero]])
    # No strict rows: theta = 0 works.
    assert _cone_feasible([[one, zero]], [], [[-one, zero]])


def test_hand_derived_linear_tail_contradiction():
    """bulk 1, both linear (full): ``q = a z + b0 tL + c0 tU + b1 z tL + c1 z tU``.

    Outward slopes need ``a + b1 - c1 > 0`` and ``a - b1 + c1 < 0``, hence
    ``b1 - c1 > |a| >= 0``; the own-gate curvature rows need ``c1 >= 0`` at the
    upper end and ``-b1 >= 0`` at the lower end, hence ``b1 - c1 <= 0``.  The
    exact test must reject every option."""
    spec = BasisSpec(
        bulk_degree=1,
        lower_tail=Polynomial(1),
        upper_tail=Polynomial(1),
        expansion="full",
    )
    with pytest.raises(ValueError, match="eventually nonnegative"):
        _build_basis(spec, NORMAL)
    # Adding a faster opposite tail removes one tail condition: accepted.
    _build_basis(
        BasisSpec(bulk_degree=1, lower_tail=Polynomial(1), upper_tail=Polynomial(2)),
        NORMAL,
    )


# ---------------------------------------------------------------------------
# Independent numerical evidence (not used by the library)
# ---------------------------------------------------------------------------


def _function_values(fn, z, model):
    """``(f, f', f'')`` of one canonical function on a grid (own evaluator)."""
    if fn.kind == "monomial":
        k = fn.degree
        zero = np.zeros_like(z)
        return (
            z**k,
            k * z ** (k - 1) if k else zero,
            k * (k - 1) * z ** (k - 2) if k > 1 else zero,
        )
    if fn.kind == "log":
        if fn.side == "lower":
            d = z - float(model.geometry.to_internal(model.request.lower))
            return -np.log(d), -1.0 / d, 1.0 / d**2
        d = float(model.geometry.to_internal(model.request.upper)) - z
        return -np.log(d), 1.0 / d, 1.0 / d**2
    if fn.kind == "exponential":
        lam = fn.rate
        if fn.side == "upper":
            e = np.exp(lam * (z - 1.0))
            return e, lam * e, lam**2 * e
        e = np.exp(-lam * (z + 1.0))
        return e, -lam * e, lam**2 * e
    p = fn.power
    r = 1.0 + z**4
    big_a = r**0.25 - 1.0
    d1 = z**3 * r**-0.75
    d2 = 3.0 * z**2 * r**-1.75
    with np.errstate(divide="ignore", invalid="ignore"):
        second = (
            p * (p - 1.0) * big_a ** (p - 2.0) * d1**2 + p * big_a ** (p - 1.0) * d2
        )
    return big_a**p, p * big_a ** (p - 1.0) * d1, np.where(big_a > 0, second, 0.0)


def _gate_parts(z, sign, /):
    """``(limit, t - limit, t', t'')`` of ``t = erf(a (1 + sign z))``.

    The deviation from the region's limit is an ``erfc``, so gate
    combinations that cancel at an end keep their Gaussian tail exactly.
    """
    a = GATE_RATE
    u = a * (1.0 + sign * z)
    limit = np.where(u > 0.0, 1.0, -1.0)
    first = sign * 2.0 * a / math.sqrt(math.pi) * np.exp(-(u**2))
    return limit, -limit * erfc(np.abs(u)), first, -2.0 * a * sign * u * first


def _columns(model, z):
    """``(phi', phi'')`` of every column on a grid in ``z``."""
    lim_l, dev_l, dl, d2l = _gate_parts(z, 1.0)
    lim_u, dev_u, du, d2u = _gate_parts(z, -1.0)
    d1 = np.zeros((z.size, model.n_coefficients))
    d2 = np.zeros((z.size, model.n_coefficients))
    for c, column in enumerate(model.columns):
        for j, gate in column.terms:
            v, v1, v2 = _function_values(model.functions[j], z, model)
            a0, a1, a2 = (float(g) for g in gate)
            # Exact limit part (small rationals) plus erfc deviations.
            g = (a0 + a1 * lim_l + a2 * lim_u) + (a1 * dev_l + a2 * dev_u)
            g1 = a1 * dl + a2 * du
            g2 = a1 * d2l + a2 * d2u
            d1[:, c] += v1 * g + v * g1
            d2[:, c] += v2 * g + 2.0 * v1 * g1 + v * g2
    return d1, d2


def _grid(model, horizon, n):
    lo = float(model.geometry.to_internal(model.request.lower))
    up = float(model.geometry.to_internal(model.request.upper))
    z = np.linspace(max(lo, -horizon), min(up, horizon), n)
    for end, sign in ((lo, 1.0), (up, -1.0)):
        if math.isfinite(end):
            z = np.concatenate([z, end + sign * np.geomspace(1e-7, horizon, 60)])
    return np.unique(z[(z > lo) & (z < up)])


def _scaled(rows):
    peak = np.abs(rows).max(axis=1, keepdims=True)
    reference = np.median(peak[peak > 0]) if np.any(peak > 0) else 1.0
    return rows / np.maximum(peak, 1e-6 * reference)


def _grid_lp(model, horizon, n):
    """``theta`` with grid ``q'' >= 0`` and unit outward far slopes, or ``None``."""
    _, d2 = _columns(model, _grid(model, horizon, n))
    finite = np.all(np.isfinite(d2), axis=1)
    rows = [_scaled(d2[finite])]
    for end, sign in ((model.request.lower, -1.0), (model.request.upper, 1.0)):
        if not math.isfinite(end):
            d1, _ = _columns(model, np.array([sign * horizon]))
            rows.append(sign * _scaled(d1))
    a_ub = -np.vstack(rows)
    b_ub = np.concatenate([np.zeros(rows[0].shape[0]), -np.ones(len(rows) - 1)])
    bounds = [(0.0, None) if c.nonnegative else (None, None) for c in model.columns]
    res = linprog(np.zeros(model.n_coefficients), A_ub=a_ub, b_ub=b_ub, bounds=bounds)
    return res.x if res.status == 0 else None


@pytest.mark.parametrize("spec", VALID, ids=lambda s: "")
def test_every_compiled_family_is_grid_feasible(spec):
    model = _compile_quiet(spec)
    assert _grid_lp(model, 8.0, 1601) is not None


@pytest.mark.parametrize(
    "tails",
    [(None, Polynomial(1)), (Polynomial(1), None), (Polynomial(1), Polynomial(1))],
    ids=lambda t: "",
)
def test_rejected_families_are_grid_infeasible_even_without_the_tail_test(
    tails, monkeypatch
):
    import gibbus._model.basis as basis_module

    monkeypatch.setattr(basis_module, "_tail_curvature_feasible", lambda *a: True)
    spec = BasisSpec(
        bulk_degree=1, lower_tail=tails[0], upper_tail=tails[1], expansion="full"
    )
    assert _grid_lp(_build_basis(spec, NORMAL), 6.0, 1201) is None
