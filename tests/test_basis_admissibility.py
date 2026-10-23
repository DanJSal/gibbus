"""G3-025: lower-order completion (rule 1) and exact tail curvature (rule 2).

Rule 1 is checked as exact span closure and, numerically, as invariance of
the basis span under a shift of the fitting-coordinate origin with fixed gate
geometry.  Rule 2's tail expansion is checked against a 150-digit mpmath
evaluation of ``q''`` far out in linear tails, and its feasibility test
against hand-derived cases.  An independent grid LP (own code, not used by the
library) supplies numerical evidence that every compiled battery family has a
convex proper member on a +-8-scale grid, and that the rejected families have
none even there.
"""

import math
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from scipy.optimize import linprog
from test_basis_construction import (
    BATTERY,
    NORMAL,
    _compile_quiet,
    _exact_matrix,
    _is_expected_rejection,
    _rank,
)

from gibbus import BasisSpec, Exponential, Polynomial, SoftPower, TanhSpec
from gibbus._model.basis import _build_basis, _cone_feasible
from gibbus._model.basis_eval import _evaluate_basis

INF = math.inf
RNG = np.random.default_rng(1610202625)
VALID = [s for s in BATTERY if not _is_expected_rejection(s)]
mp.mp.dps = 150


# ---------------------------------------------------------------------------
# Rule 1: lower-order completion
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
    ],
    ids=lambda t: "",
)
@pytest.mark.parametrize("bulk", [2, 3])
def test_basis_span_does_not_depend_on_the_coordinate_origin(tails, bulk):
    """Fixed gates, shifted data median: the function space is unchanged."""
    tanh = TanhSpec(
        lower_location=-1.0,
        upper_location=3.0,
        lower_steepness=0.8,
        upper_steepness=0.8,
    )
    spec = BasisSpec(
        bulk_degree=bulk, lower_tail=tails[0], upper_tail=tails[1], tanh=tanh
    )
    a = _build_basis(spec, NORMAL)
    b = _build_basis(spec, NORMAL + 0.9)
    assert a.geometry.center != pytest.approx(b.geometry.center)
    x = np.linspace(-6.0, 9.0, 400)
    pa = np.column_stack([np.ones_like(x), _evaluate_basis(a, x).phi])
    pb = _evaluate_basis(b, x).phi

    def rank(m):
        s = np.linalg.svd(m / np.abs(m).max(axis=0), compute_uv=False)
        return int(np.sum(s > 1e-9 * s[0]))

    assert rank(pa) == a.n_coefficients + 1
    assert rank(np.column_stack([pa, pb])) == rank(pa)


def test_cubic_bulk_with_linear_tail_keeps_both_gates_of_the_suppressed_cubic():
    # x^3 (1 + tU) alone leaves -z^3 exp(-r z) curvature in the linear upper
    # tail once the lower end forces its sign; minimal must keep a second gate.
    model = _build_basis(BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)), NORMAL)
    cubic = next(f.index for f in model.functions if f.label == "x^3")
    gates = [g for c in model.columns for j, g in c.terms if j == cubic]
    assert len(gates) == 2


# ---------------------------------------------------------------------------
# Rule 2: exact tail curvature
# ---------------------------------------------------------------------------


def _oracle_d2q(model, theta, x):
    """150-digit ``q''(x)`` from the defining formulas (mpmath.diff)."""
    g, req = model.geometry, model.request
    c, h = mp.mpf(g.center), mp.mpf(g.scale)

    def gate_value(a, xv):
        val = mp.mpf(a[0].numerator) / a[0].denominator
        if a[1] != 0:
            t = mp.tanh(mp.mpf(g.lower_steepness) * (xv - mp.mpf(g.lower_location)))
            val += mp.mpf(a[1].numerator) / a[1].denominator * t
        if a[2] != 0:
            t = mp.tanh(mp.mpf(g.upper_steepness) * (mp.mpf(g.upper_location) - xv))
            val += mp.mpf(a[2].numerator) / a[2].denominator * t
        return val

    def f(fn, xv):
        if fn.kind == "monomial":
            return ((xv - c) / h) ** fn.degree
        if fn.kind == "log":
            if fn.side == "lower":
                return -mp.log(xv - mp.mpf(req.lower))
            return -mp.log(mp.mpf(req.upper) - xv)
        if fn.kind == "exponential":
            lam, loc = mp.mpf(fn.rate), mp.mpf(fn.location)
            sign = -1 if fn.side == "lower" else 1
            return mp.exp(sign * lam * (xv - loc))
        p = mp.mpf(fn.power)
        return ((xv - c) ** 2 + mp.mpf(fn.delta)) ** (p / 2) / h**p

    def q(xv):
        total = mp.mpf(0)
        for coef, column in zip(theta, model.columns):
            for j, gate in column.terms:
                total += (
                    mp.mpf(float(coef))
                    * f(model.functions[j], xv)
                    * gate_value(gate, xv)
                )
        return total

    return mp.diff(q, mp.mpf(float(x)), 2)


LINEAR_TAIL_SPECS = [
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=3, upper_tail=Polynomial(1), expansion="full"),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(3), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Exponential(0.7)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=SoftPower(1.5)),
    BasisSpec(support=(0.0, INF), bulk_degree=1),
]


@pytest.mark.parametrize("spec", LINEAR_TAIL_SPECS, ids=lambda s: "")
def test_tail_curvature_expansion_matches_high_precision_q2(spec):
    """``q''(z) e^{rate |z|}`` equals ``K * sum_k (row_k . theta) |z|**order_k``
    with one positive constant ``K`` (the common weight normalization), up to
    the next, exponentially smaller rate group.  Checked at several far points,
    this validates every term of the expansion, not only its sign.  Soft-power
    sub-orders are not part of the expansion, so with a soft power only the
    leading term is compared, much farther out."""
    model = _compile_quiet(spec)
    g = model.geometry
    for trial in range(3):
        theta = RNG.normal(size=model.n_coefficients)
        for side in model.asymptotics:
            if side.target_order != (1, 1.0, 0):
                assert side.curvature == ()
                continue
            sign = -1.0 if side.side == "lower" else 1.0
            terms = [
                (t, float(np.dot([float(v) for v in t.row_exact], theta)))
                for t in side.curvature
            ]
            if not terms:
                # No correction survives: q'' vanishes far out.
                d2 = _oracle_d2q(model, theta, g.center + sign * 12.0 * g.scale)
                assert abs(d2) < mp.mpf(10) ** -100, (trial, side.side)
                continue
            rate = terms[0][0].rate
            assert all(t.rate == rate for t, _ in terms)
            integer = all(float(t.order).is_integer() for t, _ in terms)
            # Far enough that the next rate group (gap >= ~2.4 here) is
            # negligible; soft-power sub-orders (relative z**-1.5) need more.
            zs = (20.0, 30.0, 40.0, 50.0) if integer else (100.0, 200.0, 400.0)
            ks = []
            with mp.workdps(300 if integer else 900):
                for z in zs:
                    d2 = _oracle_d2q(model, theta, g.center + sign * z * g.scale)
                    poly = sum(mp.mpf(a) * mp.mpf(z) ** t.order for t, a in terms)
                    ks.append(d2 * mp.exp(mp.mpf(rate) * z) / poly)
            ks = [float(k) for k in ks]
            assert all(k > 0 for k in ks), (trial, side.side, ks)
            tol = 1e-8 if integer else 1e-2
            np.testing.assert_allclose(ks, ks[0], rtol=tol, err_msg=side.side)


def test_opposite_endpoint_log_dominates_linear_tail_curvature():
    model = _compile_quiet(
        BasisSpec(
            support=(-INF, 0.0), bulk_degree=2, lower_tail=Polynomial(1), log_upper=True
        )
    )
    (side,) = model.asymptotics
    assert side.curvature[0].shape == "log"
    theta = RNG.normal(size=model.n_coefficients)
    gamma = float(np.dot([float(v) for v in side.curvature[0].row_exact], theta))
    x = model.geometry.center - 20.0 * model.geometry.scale
    d2 = _oracle_d2q(model, theta, x)
    assert float(d2 / (gamma / mp.mpf(x) ** 2)) == pytest.approx(1.0, rel=1e-8)


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
    """bulk 1, both linear (full): k = exp(-2 a (xU - xL)) < 1 makes the slope
    condition b > c incompatible with the two tail conditions b < k c and
    b < c / k, so the exact test must reject every option."""
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


def _grid(model, horizon, n):
    g, req = model.geometry, model.request
    x = g.center + g.scale * np.linspace(-horizon, horizon, n)
    for end, sign in ((req.lower, 1.0), (req.upper, -1.0)):
        if math.isfinite(end):
            x = np.concatenate(
                [x, end + sign * g.scale * np.geomspace(1e-7, horizon, 60)]
            )
    return np.unique(x[(x > req.lower) & (x < req.upper)])


def _scaled(rows):
    peak = np.abs(rows).max(axis=1, keepdims=True)
    reference = np.median(peak[peak > 0]) if np.any(peak > 0) else 1.0
    return rows / np.maximum(peak, 1e-6 * reference)


def _grid_lp(model, horizon, n):
    """``theta`` with grid ``q'' >= 0`` and unit outward far slopes, or ``None``."""
    g, req = model.geometry, model.request
    vals = _evaluate_basis(model, _grid(model, horizon, n))
    finite = np.all(np.isfinite(vals.d2phi), axis=1)
    rows = [_scaled(vals.d2phi[finite])]
    for end, sign in ((req.lower, -1.0), (req.upper, 1.0)):
        if not math.isfinite(end):
            far = np.array([g.center + sign * horizon * g.scale])
            rows.append(sign * _scaled(_evaluate_basis(model, far).dphi))
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
def test_rejected_families_are_grid_infeasible_even_without_rule_two(
    tails, monkeypatch
):
    import gibbus._model.basis as basis_module

    monkeypatch.setattr(basis_module, "_tail_curvature_feasible", lambda *a: True)
    spec = BasisSpec(
        bulk_degree=1, lower_tail=tails[0], upper_tail=tails[1], expansion="full"
    )
    assert _grid_lp(_build_basis(spec, NORMAL), 6.0, 1201) is None
