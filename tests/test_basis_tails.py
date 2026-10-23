"""G3 Stage 2b: tail cutoffs and finite-endpoint exclusion.

Every verdict is checked against an analytic 400-digit mpmath ``q''``:
``nonnegative`` cutoffs must have ``q'' > 0`` (or exactly 0) at several points
beyond them, ``violated`` tails must have ``q'' < 0`` far out, and an
artificially small cutoff cap must never yield ``nonnegative``.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from test_basis_admissibility import VALID, _grid_lp
from test_basis_construction import _compile_quiet

from gibbus import BasisSpec, Exponential, Polynomial, SoftPower
from gibbus._model.basis_tails import (
    _build_tail_model,
    _endpoint_exclusion,
    _tail_verdict,
)

RNG = np.random.default_rng(1610202602)


def _mp(v):
    return mp.mpf(v.numerator) / v.denominator if hasattr(v, "numerator") else mp.mpf(v)


def _oracle_q2(model, theta, x):
    """Analytic ``q''(x)`` (user units) at the current mpmath precision."""
    g, req = model.geometry, model.request
    c, h = mp.mpf(g.center), mp.mpf(g.scale)
    x = mp.mpf(x)

    def tanh_parts(alpha, loc, sign):
        u = sign * mp.mpf(alpha) * (x - mp.mpf(loc))
        t = mp.tanh(u)
        w = mp.sech(u) ** 2
        cc = sign * mp.mpf(alpha)
        return t, cc * w, -2 * cc**2 * t * w

    tl = (
        tanh_parts(g.lower_steepness, g.lower_location, 1)
        if g.lower_location is not None
        else (1, 0, 0)
    )
    tu = (
        tanh_parts(g.upper_steepness, g.upper_location, -1)
        if g.upper_location is not None
        else (1, 0, 0)
    )

    def f(fn):
        if fn.kind == "monomial":
            n = fn.degree
            z = (x - c) / h
            v = z**n
            d1 = n * z ** (n - 1) / h if n >= 1 else 0
            d2 = n * (n - 1) * z ** (n - 2) / h**2 if n >= 2 else 0
            return v, d1, d2
        if fn.kind == "log":
            if fn.side == "lower":
                d = x - mp.mpf(req.lower)
                return -mp.log(d), -1 / d, 1 / d**2
            d = mp.mpf(req.upper) - x
            return -mp.log(d), 1 / d, 1 / d**2
        if fn.kind == "exponential":
            lam = mp.mpf(fn.rate) * (-1 if fn.side == "lower" else 1)
            v = mp.exp(lam * (x - mp.mpf(fn.location)))
            return v, lam * v, lam**2 * v
        p, a = mp.mpf(fn.power), mp.mpf(fn.power) / 2
        dl = mp.mpf(fn.delta)
        s = (x - c) ** 2 + dl
        v = s**a / h**p
        d1 = 2 * a * (x - c) * s ** (a - 1) / h**p
        d2 = (
            2 * a * s ** (a - 1) + 4 * a * (a - 1) * (x - c) ** 2 * s ** (a - 2)
        ) / h**p
        return v, d1, d2

    total = mp.mpf(0)
    for coef, column in zip(theta, model.columns):
        if coef == 0.0:
            continue
        col = mp.mpf(0)
        for j, gate in column.terms:
            a0, a1, a2 = (_mp(v) for v in gate)
            g0 = a0 + a1 * tl[0] + a2 * tu[0]
            g1 = a1 * tl[1] + a2 * tu[1]
            g2 = a1 * tl[2] + a2 * tu[2]
            v, d1, d2 = f(model.functions[j])
            col += d2 * g0 + 2 * d1 * g1 + v * g2
        total += mp.mpf(float(coef)) * col
    return total


TAIL_SPECS = [s for s in VALID if math.isinf(s.support[0]) or math.isinf(s.support[1])][
    ::7
]


def _thetas(model, seed):
    # Per-case generator: the module RNG made the draws depend on which tests
    # ran earlier in the same worker (an order-dependent oracle miss, G3-049).
    rng = np.random.default_rng(seed)
    out = [rng.normal(size=model.n_coefficients) for _ in range(2)]
    feasible = _grid_lp(model, 8.0, 801)
    if feasible is not None:
        out.append(feasible)
    for t in out:
        t[[i for i, c in enumerate(model.columns) if c.nonnegative]] = np.abs(
            t[[i for i, c in enumerate(model.columns) if c.nonnegative]]
        )
    return out


@pytest.mark.parametrize("spec", TAIL_SPECS, ids=lambda s: "")
def test_tail_verdicts_agree_with_high_precision_curvature(spec):
    model = _compile_quiet(spec)
    g = model.geometry
    for side in ("lower", "upper"):
        if not math.isinf(getattr(model.request, side)):
            continue
        tail = _build_tail_model(model, side)
        for theta in _thetas(model, 1000 + TAIL_SPECS.index(spec)):
            res = _tail_verdict(tail, model, theta)
            if res.status == "nonnegative":
                lead = next(
                    (a for a in tail.atoms if float(a.row @ theta) != 0.0), None
                )
                for mult in (1.0, 1.5, 3.0, 10.0, 40.0):
                    zeta = res.zeta * mult
                    if lead is not None and lead.kind == "power":
                        log_w = lead.power * math.log(zeta) + lead.rate * zeta
                        if log_w < -900.0:
                            break
                    dps = (
                        60 + int(abs(lead.rate) * zeta / 2.0)
                        if lead is not None
                        else 60
                    )
                    with mp.workdps(dps):
                        q2 = _oracle_q2(
                            model, theta, g.center + tail.sign * zeta * g.scale
                        )
                    if lead is None:
                        assert abs(q2) < mp.mpf(10) ** (-dps // 2)
                    else:
                        assert q2 > 0, (side, res, mult, float(q2))
            elif res.status == "violated":
                lead = next(a for a in tail.atoms if float(a.row @ theta) != 0.0)
                # The cut is the leader's row, unless a stronger atom is
                # exactly zero at theta: then that atom's row (G3-049; the
                # leader's row is not a valid cut there).
                stronger = tail.atoms[: tail.atoms.index(lead)]
                zero = [
                    a
                    for a in stronger
                    if np.any(a.row != 0.0) and not np.any(a.row * theta != 0.0)
                ]
                if zero:
                    expected = zero[0].row if zero[0].cut is None else zero[0].cut
                    np.testing.assert_array_equal(res.cut_row, expected)
                    assert float(res.cut_row @ theta) == 0.0
                else:
                    assert float(res.cut_row @ theta) < 0.0
                found = False
                # A negative leader wins eventually, possibly very far out when
                # large subordinate terms compete (algebraic leaders: up to 1e12).
                zetas = [30.0 * 4.0**k for k in range(18)]
                for zeta in zetas:
                    dps = 60 + int(abs(lead.rate) * zeta / 2.0)
                    if dps > 2000:
                        break
                    with mp.workdps(dps):
                        q2 = _oracle_q2(
                            model, theta, g.center + tail.sign * zeta * g.scale
                        )
                    if q2 < 0:
                        found = True
                        break
                assert found, (side, res)
            # Fail closed: a cap below the first admissible cutoff never proves.
            capped = _tail_verdict(tail, model, theta, zeta_cap=0.5 * tail.zeta_min)
            assert capped.status != "nonnegative" or capped.lead_value == 0.0


def test_violated_linear_tail_gives_leading_functional_as_cut():
    model = _compile_quiet(BasisSpec(bulk_degree=3, upper_tail=Polynomial(1)))
    tail = _build_tail_model(model, "upper")
    lead = tail.atoms[0]
    theta = RNG.normal(size=model.n_coefficients)
    if float(lead.row @ theta) > 0:
        theta = theta - 2.0 * float(lead.row @ theta) * lead.row / float(
            lead.row @ lead.row
        )
    res = _tail_verdict(tail, model, theta)
    assert res.status == "violated"
    np.testing.assert_array_equal(res.cut_row, lead.row)


@pytest.mark.parametrize(
    "spec,weights",
    [
        (BasisSpec(bulk_degree=2), {"x^2": 1.0}),
        (
            BasisSpec(
                bulk_degree=2, lower_tail=Exponential(0.7), upper_tail=SoftPower(1.5)
            ),
            {"expL": 1.0, "powU": 1.0},
        ),
        (BasisSpec(support=(0.0, math.inf), bulk_degree=1), {"x": 1.0}),
    ],
    ids=lambda s: "",
)
def test_hand_built_convex_members_have_nonnegative_tails(spec, weights):
    # (A grid LP is no substitute: its vectors can be improper beyond the grid.)
    model = _compile_quiet(spec)
    theta = np.zeros(model.n_coefficients)
    for i, column in enumerate(model.columns):
        theta[i] = weights.get(column.label, 0.0)
    assert np.count_nonzero(theta) == len(weights)
    for side in ("lower", "upper"):
        if math.isinf(getattr(model.request, side)):
            res = _tail_verdict(_build_tail_model(model, side), model, theta)
            assert res.status == "nonnegative", (side, res)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_endpoint_exclusion_respects_threshold(side):
    model = _compile_quiet(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=3, log_lower=True, log_upper=True)
    )
    for _ in range(5):
        theta = RNG.normal(size=model.n_coefficients)
        logs = [i for i, c in enumerate(model.columns) if c.nonnegative]
        theta[logs] = np.abs(theta[logs]) * 0.05
        tau = 1e-3 / model.geometry.scale**2
        delta = _endpoint_exclusion(model, theta, side, tau)
        assert delta > 0.0
        end = model.request.lower if side == "lower" else model.request.upper
        step = 1.0 if side == "lower" else -1.0
        with mp.workdps(50):
            for frac in np.geomspace(1e-9, 1.0, 25):
                q2 = _oracle_q2(model, theta, end + step * frac * delta)
                assert q2 >= tau, (frac, float(q2))
    theta[logs[0 if side == "lower" else 1]] = 0.0
    assert _endpoint_exclusion(model, theta, side, tau) == 0.0
