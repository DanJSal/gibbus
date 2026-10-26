"""Tail atoms, tail cutoffs and finite-endpoint exclusion.

The reference is the independent high-precision oracle of ``_basis_oracle``.
The atoms themselves are checked rigorously: at every ``zeta >= zeta_min`` the
oracle's ``d^k q / d zeta^k`` lies within the sum of the main atoms plus or
minus the sum of the envelopes.  Every verdict is checked too:
``nonnegative`` cutoffs must have ``q'' > 0`` (or exactly 0) at several points
beyond them, ``violated`` tails must have ``q'' < 0`` far out, and an
artificially small cutoff cap must never yield ``nonnegative``.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from _basis_oracle import BasisOracle
from test_basis_admissibility import VALID, _grid_lp
from test_basis_construction import _compile_quiet

from gibbus import BasisSpec, Exponential, Polynomial, SoftPower
from gibbus._model.basis_tails import (
    _build_tail_model,
    _endpoint_exclusion,
    _tail_verdict,
)

RNG = np.random.default_rng(1610202602)


_ORACLES = {}


def _oracle(model):
    key = id(model)
    if key not in _ORACLES:
        _ORACLES[key] = (model, BasisOracle(model, max_order=2))
    return _ORACLES[key][1]


def _oracle_q2(model, theta, x):
    """High-precision ``q''(x)`` in user units."""
    return _oracle(model).potential(theta, x, 2, dps=mp.mp.dps)[0]


def _weight(atom_or_env, zeta):
    z = mp.mpf(zeta)
    return z**atom_or_env.power * mp.exp(
        atom_or_env.rate * z - atom_or_env.kappa * z * z + atom_or_env.const
    )


ENVELOPE_SPECS = [
    BasisSpec(bulk_degree=3),
    BasisSpec(
        bulk_degree=4,
        lower_tail=Polynomial(1),
        upper_tail=Polynomial(1),
        expansion="full",
    ),
    BasisSpec(
        bulk_degree=2,
        lower_tail=SoftPower(1.01),
        upper_tail=SoftPower(7.3),
        expansion="full",
    ),
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(2.5), upper_tail=SoftPower(2.5)),
    BasisSpec(
        bulk_degree=3,
        lower_tail=Exponential(0.7),
        upper_tail=Exponential(5.0),
        expansion="full",
    ),
    BasisSpec(
        support=(0.0, math.inf),
        bulk_degree=2,
        log_lower=True,
        upper_tail=SoftPower(1.4),
        expansion="full",
    ),
    BasisSpec(
        support=(-math.inf, 1.0),
        bulk_degree=3,
        log_upper=True,
        lower_tail=Polynomial(1),
    ),
]


@pytest.mark.parametrize("spec", ENVELOPE_SPECS, ids=lambda s: "")
@pytest.mark.parametrize("order", [1, 2])
def test_main_atoms_and_envelopes_enclose_the_tail_derivative(spec, order):
    model = _compile_quiet(spec)
    oracle = BasisOracle(model, units="z", max_order=2)
    for side in [s.side for s in model.asymptotics]:
        sgn = 1.0 if side == "upper" else -1.0
        tail = _build_tail_model(model, side, order=order)
        for _ in range(2):
            theta = RNG.normal(size=model.n_coefficients)
            for zeta in (tail.zeta_min, 2.5, 3.3, 5.0, 8.0, 15.0, 40.0, 300.0):
                if zeta < tail.zeta_min:
                    continue
                with mp.workdps(60):
                    value = oracle.potential(theta, sgn * zeta, order, dps=60)[0]
                    value *= sgn**order
                    main = mp.mpf(0)
                    for atom in tail.atoms:
                        coef = mp.mpf(float(atom.row @ theta))
                        if atom.kind == "log":
                            main += coef / (mp.mpf(zeta) + atom.offset) ** order
                        else:
                            main += coef * _weight(atom, zeta)
                    envelope = sum(
                        (
                            abs(theta[e.column]) * e.coef * _weight(e, zeta)
                            for e in tail.envelopes
                        ),
                        mp.mpf(0),
                    )
                    slack = envelope + mp.mpf(1e-12) * (abs(value) + abs(main))
                    assert abs(value - main) <= slack + mp.mpf(10) ** -300, (
                        side,
                        zeta,
                        float(value),
                        float(main),
                        float(envelope),
                    )


TAIL_SPECS = [s for s in VALID if math.isinf(s.support[0]) or math.isinf(s.support[1])][
    ::7
]


def _thetas(model, seed):
    # Per-case generator: the module RNG would make the draws depend on which
    # tests ran earlier in the same worker.
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
            if res.status == "verified":
                lead = next(
                    (a for a in tail.atoms if float(a.row @ theta) != 0.0), None
                )
                for mult in (1.0, 1.5, 3.0, 10.0, 40.0):
                    zeta = res.zeta * mult
                    if lead is not None and lead.kind == "power":
                        log_w = (
                            lead.power * math.log(zeta)
                            + lead.rate * zeta
                            - lead.kappa * zeta * zeta
                            + lead.const
                        )
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
                # exactly zero at theta: then that atom's row (the leader's row
                # is not a valid cut there).
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
            assert capped.status != "verified" or capped.lead_value == 0.0


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
        (BasisSpec(bulk_degree=2), {"z^2": 1.0}),
        (
            BasisSpec(
                bulk_degree=2, lower_tail=Exponential(0.7), upper_tail=SoftPower(1.5)
            ),
            # (-1 + t) = -2 W: negative weights give outward-convex tails.
            {"expL*(-1+tL)": -1.0, "powU*(-1+tU)": -1.0},
        ),
        (BasisSpec(support=(0.0, math.inf), bulk_degree=1), {"z": 1.0}),
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
            assert res.status == "verified", (side, res)


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
