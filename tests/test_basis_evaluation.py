"""Numerical evaluation of ``Phi`` and its ``x`` derivatives to order four.

The reference is the independent high-precision oracle of ``_basis_oracle``
(symbolic derivatives of the defining formulas, mpmath at adaptive
precision).  It shares only the compiled gate coefficients and family
constants with the evaluator, not its Taylor arithmetic or rearrangements.
Agreement is required at the backward-stable level: a few ``eps`` times the
absolute Leibniz magnitude of each column plus the effect of rounding ``z``.
Far-tail points check that exactly suppressed gates are evaluated with
*relative* accuracy rather than as cancelling differences.
"""

import itertools
import math
import warnings

import numpy as np
import pytest
from _basis_oracle import BasisOracle
from scipy.special import erf

from gibbus import (
    BasisSpec,
    Exponential,
    GibbusBasisWarning,
    Polynomial,
    SoftPower,
)
from gibbus._fit.basis_objective import _point_statistics
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis, _potential

INF = math.inf
RNG = np.random.default_rng(161020261)
NORMAL = RNG.normal(1.5, 2.0, size=400)
HALF = RNG.gamma(2.0, 1.5, size=400) + 0.25
BOUNDED = RNG.beta(2.0, 3.0, size=400) * 4.0 - 1.0

EPS = 2.220446049250313e-16


def _data_for(support):
    lo, up = support
    if lo == -INF and up == INF:
        return NORMAL
    if up == INF:
        return HALF + lo
    if lo == -INF:
        return up - HALF
    return BOUNDED


def _compile(spec):
    lo, up = spec.support
    data = None if math.isfinite(lo) and math.isfinite(up) else _data_for(spec.support)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, data)


ORACLE_SPECS = [
    # All nine support/log configurations.
    BasisSpec(bulk_degree=3),
    BasisSpec(support=(0.0, INF), bulk_degree=3),
    BasisSpec(support=(0.0, INF), bulk_degree=2, log_lower=True, expansion="full"),
    BasisSpec(support=(-INF, 0.0), bulk_degree=3),
    BasisSpec(support=(-INF, 0.0), bulk_degree=2, log_upper=True, expansion="full"),
    BasisSpec(support=(-1.0, 3.0), bulk_degree=3),
    BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True),
    BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_upper=True),
    BasisSpec(support=(-1.0, 3.0), bulk_degree=4, log_lower=True, log_upper=True),
    # Tail families, expansions, shared/independent leaders.
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(2), upper_tail=Polynomial(2)),
    BasisSpec(
        bulk_degree=4,
        lower_tail=Polynomial(2),
        upper_tail=Polynomial(2),
        expansion="full",
    ),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(3), upper_tail=Polynomial(3)),
    BasisSpec(
        bulk_degree=2,
        lower_tail=Polynomial(3),
        upper_tail=Polynomial(3),
        independent_equal_rate_coefficients=True,
        expansion="full",
    ),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1)),
    BasisSpec(bulk_degree=3, lower_tail=SoftPower(1.5), upper_tail=SoftPower(2.5)),
    BasisSpec(
        bulk_degree=2,
        lower_tail=SoftPower(2.5),
        upper_tail=SoftPower(2.5),
        expansion="full",
    ),
    BasisSpec(bulk_degree=2, upper_tail=SoftPower(3.0)),
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.01), upper_tail=SoftPower(37.5)),
    BasisSpec(bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Exponential(0.7)),
    BasisSpec(
        bulk_degree=3,
        lower_tail=Exponential(0.7),
        upper_tail=Exponential(1.3),
        expansion="full",
    ),
    BasisSpec(bulk_degree=4, lower_tail=Exponential(0.5), upper_tail=Polynomial(2)),
    BasisSpec(support=(0.0, INF), bulk_degree=4, upper_tail=Polynomial(2)),
    BasisSpec(support=(0.0, INF), bulk_degree=2, upper_tail=Exponential(0.9)),
    BasisSpec(support=(-INF, 0.0), bulk_degree=3, lower_tail=SoftPower(1.5)),
    BasisSpec(
        bulk_degree=2, lower_tail=Exponential(1e-5), upper_tail=Exponential(40.0)
    ),
]


def _test_points(model):
    req, g = model.request, model.geometry
    c, h = g.center, g.scale
    pts = []
    lo = req.lower if math.isfinite(req.lower) else c - 6.0 * h
    up = req.upper if math.isfinite(req.upper) else c + 6.0 * h
    span = up - lo
    pts.extend(lo + span * np.array([0.03, 0.2, 0.41, 0.5, 0.67, 0.85, 0.97]))
    pts.extend([c, c - h, c + h, c + 1e-7 * h])
    tail = []
    if not math.isfinite(req.lower):
        tail.extend(c - h * np.array([8.0, 15.0, 25.0]))
    if not math.isfinite(req.upper):
        tail.extend(c + h * np.array([8.0, 15.0, 25.0]))
    # Keep the added points clear of finite endpoints (derivative stencils).
    inside = [p for p in pts if req.lower + 1e-3 * h < p < req.upper - 1e-3 * h]
    return np.array(inside), np.array(tail)


@pytest.mark.parametrize("spec", ORACLE_SPECS, ids=lambda s: "")
def test_columns_and_derivatives_match_high_precision_oracle(spec):
    model = _compile(spec)
    oracle = BasisOracle(model)
    h = model.geometry.scale
    bulk, tail = _test_points(model)
    points = np.concatenate([bulk, tail])
    values = _evaluate_basis(model, points, max_order=4)
    for i, x in enumerate(points):
        # Rounding z = (x - c) / h moves the evaluation by about eps |z| h.
        shift = 2.0 * EPS * (abs(float(model.geometry.to_internal(x))) + 1.0) * h
        for col, column in enumerate(model.columns):
            refs = [oracle.column(col, x, k) for k in range(6)]
            for order in range(5):
                got = values.derivative(order)[i, col]
                ref, mag = refs[order]
                if got == float(ref):
                    continue  # includes a matching overflow to +-inf
                tol = (
                    16.0 * EPS * float(mag) + shift * float(refs[order + 1][1]) + 1e-300
                )
                assert (
                    abs(got - float(ref)) <= tol
                ), f"{column.label} d{order} at x={x}: got {got}, ref {float(ref)}"


def test_suppressed_gate_is_relatively_accurate_far_into_the_tail():
    # z^3 (tL + tU) on the full line: exactly zero gate limit at both ends.
    model = _compile(BasisSpec(bulk_degree=3))
    col = next(i for i, c in enumerate(model.columns) if c.label == "z^3*(tL+tU)")
    g = model.geometry
    x = g.center + g.scale * np.array([-12.0, -8.0, -5.0, 5.0, 8.0, 12.0])
    got = _evaluate_basis(model, x).phi[:, col]
    oracle = BasisOracle(model, max_order=0)
    ref = np.array([float(oracle.column(col, v, 0)[0]) for v in x])
    assert np.all(ref != 0.0) and np.all(np.isfinite(got))
    np.testing.assert_allclose(got, ref, rtol=1e-12)


def test_soft_power_columns_vanish_to_fourth_order_at_the_origin():
    model = _compile(BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.01)))
    col = next(i for i, c in enumerate(model.columns) if c.label.startswith("powL"))
    values = _evaluate_basis(model, np.array([model.geometry.center]), max_order=4)
    for order in range(5):
        assert values.derivative(order)[0, col] == 0.0


def _broad_battery():
    tails = [None, Polynomial(1), Polynomial(3), SoftPower(1.5), Exponential(0.7)]
    for lt, ut, bulk, expansion in itertools.product(
        tails, tails, (2, 3), ("minimal", "full")
    ):
        yield BasisSpec(
            bulk_degree=bulk, lower_tail=lt, upper_tail=ut, expansion=expansion
        )
    for support, ll, lu in (
        ((0.0, INF), True, False),
        ((-INF, 0.0), False, True),
        ((-1.0, 3.0), True, True),
    ):
        for bulk, expansion in itertools.product((1, 2, 4), ("minimal", "full")):
            yield BasisSpec(
                support=support,
                bulk_degree=bulk,
                log_lower=ll,
                log_upper=lu,
                expansion=expansion,
            )


@pytest.mark.parametrize("spec", list(_broad_battery()), ids=lambda s: "")
def test_derivatives_consistent_by_finite_differences(spec):
    model = _compile(spec)
    bulk, _ = _test_points(model)
    step = 2e-5 * model.geometry.scale
    vals = _evaluate_basis(model, bulk)
    plus = _evaluate_basis(model, bulk + step)
    minus = _evaluate_basis(model, bulk - step)
    fd1 = (plus.phi - minus.phi) / (2 * step)
    fd2 = (plus.dphi - minus.dphi) / (2 * step)
    scale1 = np.max(np.abs(vals.dphi), axis=0) + 1e-300
    scale2 = np.max(np.abs(vals.d2phi), axis=0) + 1e-300
    assert np.all(np.abs(fd1 - vals.dphi) <= 1e-6 * scale1)
    assert np.all(np.abs(fd2 - vals.d2phi) <= 1e-6 * scale2)
    # Third and fourth derivatives, and agreement of the order-2 prefix.
    high = _evaluate_basis(model, bulk, max_order=4)
    hp = _evaluate_basis(model, bulk + step, max_order=4)
    hm = _evaluate_basis(model, bulk - step, max_order=4)
    for k in range(3):
        np.testing.assert_array_equal(high.derivative(k), vals.derivative(k))
    for k in (3, 4):
        fd = (hp.derivative(k - 1) - hm.derivative(k - 1)) / (2 * step)
        scale = np.max(np.abs(high.derivative(k)), axis=0) + 1e-300
        # Truncation error ~ (step / distance-to-log-endpoint)**2 near endpoints.
        assert np.all(np.abs(fd - high.derivative(k)) <= 2e-5 * scale), k


@pytest.mark.parametrize("spec", list(_broad_battery()), ids=lambda s: "")
def test_columns_are_functionally_independent_of_each_other_and_the_gauge(spec):
    model = _compile(spec)
    req, g = model.request, model.geometry
    lo = req.lower if math.isfinite(req.lower) else g.center - 8.0 * g.scale
    up = req.upper if math.isfinite(req.upper) else g.center + 8.0 * g.scale
    x = np.linspace(lo, up, 402)[1:-1]
    phi = _evaluate_basis(model, x).phi
    mat = np.column_stack([np.ones_like(x), phi])
    mat = mat / np.linalg.norm(mat, axis=0)
    sv = np.linalg.svd(mat, compute_uv=False)
    assert sv[-1] / sv[0] > 1e-12, f"near-dependent columns: {sv[-1] / sv[0]:.3g}"


@pytest.mark.parametrize("spec", ORACLE_SPECS, ids=lambda s: "")
def test_reduced_columns_reproduce_full_slot_potential(spec):
    model = _compile(spec)
    bulk, _ = _test_points(model)
    theta = RNG.normal(size=model.n_coefficients)
    full = model.full_coefficients(theta)
    g, req = model.geometry, model.request
    z = g.to_internal(bulk)
    a = g.gate_sharpness / math.sqrt(2.0)
    tl = erf(a * (1.0 + z)) if g.has_lower_gate else np.zeros_like(bulk)
    tu = erf(a * (1.0 - z)) if g.has_upper_gate else np.zeros_like(bulk)
    q_full = np.zeros_like(bulk)
    for fn in model.functions:
        a0, b, c = full[3 * fn.index : 3 * fn.index + 3]
        if a0 == b == c == 0.0:
            continue
        if fn.kind == "monomial":
            f = z**fn.degree
        elif fn.kind == "log":
            f = -np.log(bulk - req.lower if fn.side == "lower" else req.upper - bulk)
        elif fn.kind == "exponential":
            f = (
                np.exp(fn.rate * (z - 1.0))
                if fn.side == "upper"
                else np.exp(-fn.rate * (z + 1.0))
            )
        else:
            f = ((1.0 + z**4) ** 0.25 - 1.0) ** fn.power
        q_full += f * (a0 + b * tl + c * tu)
    q, _, _ = _potential(model, theta, bulk)
    np.testing.assert_allclose(q, q_full, rtol=1e-12, atol=1e-12 * np.max(np.abs(q)))


ASYMPTOTIC_SPECS = [
    BasisSpec(bulk_degree=3),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(2), upper_tail=Polynomial(2)),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(3), upper_tail=Polynomial(3)),
    BasisSpec(
        bulk_degree=2,
        lower_tail=Polynomial(3),
        upper_tail=Polynomial(3),
        independent_equal_rate_coefficients=True,
    ),
    BasisSpec(bulk_degree=2, lower_tail=Polynomial(1)),
    BasisSpec(bulk_degree=1, lower_tail=SoftPower(2.5), upper_tail=SoftPower(1.5)),
    BasisSpec(bulk_degree=3, lower_tail=SoftPower(2.5), upper_tail=SoftPower(2.5)),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(2), expansion="full"),
    BasisSpec(support=(0.0, INF), bulk_degree=3, log_lower=True),
    BasisSpec(support=(-INF, 0.0), bulk_degree=2, upper_tail=None, log_upper=True),
]


@pytest.mark.parametrize("spec", ASYMPTOTIC_SPECS, ids=lambda s: "")
def test_power_leading_coefficient_metadata_matches_evaluation(spec):
    model = _compile(spec)
    g = model.geometry
    theta = RNG.normal(size=model.n_coefficients)
    for side in model.asymptotics:
        lead = side.leading
        assert lead.shape == "power"
        p = lead.order[1]
        nxt = side.terms[1].order[1] if len(side.terms) > 1 else p - 1.0
        coef = float(lead.row @ theta)
        sign = -1.0 if side.side == "lower" else 1.0
        errs = []
        for dist in (1e3, 1e4, 1e5):
            x = np.array([g.center + sign * dist * g.scale])
            q, _, _ = _potential(model, theta, x)
            r = q[0] / ((dist * g.scale) ** p)
            errs.append(abs(r - coef))
        # Converges to the metadata amplitude at the next-order rate.
        assert errs[-1] <= max(errs[0], 1e-300)
        bound = 50.0 * np.sum(np.abs(theta)) * (1e5 ** (nxt - p)) * g.scale ** (
            nxt - p
        ) + 1e-9 * abs(coef)
        assert errs[-1] <= bound, (side.side, errs, coef)


@pytest.mark.parametrize(
    "spec",
    [
        BasisSpec(
            bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Exponential(0.7)
        ),
        BasisSpec(
            bulk_degree=2,
            lower_tail=Exponential(0.5),
            upper_tail=Exponential(1.1),
            expansion="full",
        ),
        BasisSpec(support=(0.0, INF), upper_tail=Exponential(0.9), log_lower=True),
    ],
    ids=lambda s: "",
)
def test_exponential_leading_coefficient_metadata_matches_evaluation(spec):
    model = _compile(spec)
    g = model.geometry
    theta = RNG.normal(size=model.n_coefficients)
    for side in model.asymptotics:
        lead = side.leading
        assert lead.shape == "exponential"
        rate = lead.order[1] / g.scale  # canonical lambda to user units
        coef = math.exp(lead.log_factor) * float(lead.row @ theta)
        sign = -1.0 if side.side == "lower" else 1.0
        dist = 60.0 / rate
        x = np.array([g.center + sign * dist])
        q, _, _ = _potential(model, theta, x)
        np.testing.assert_allclose(q[0] / math.exp(rate * dist), coef, rtol=1e-9)


def test_shared_exponential_amplitudes_are_equal_in_evaluation():
    model = _compile(
        BasisSpec(lower_tail=Exponential(0.8), upper_tail=Exponential(0.8))
    )
    g = model.geometry
    theta = RNG.normal(size=model.n_coefficients)
    dist = 60.0 / 0.8
    x = np.array([g.center - dist, g.center + dist])
    q, _, _ = _potential(model, theta, x)
    np.testing.assert_allclose(q[0], q[1], rtol=1e-9)


def test_finite_endpoint_log_behavior():
    model = _compile(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True, log_upper=True)
    )
    vals = _evaluate_basis(model, np.array([-1.0, -1.0 + 1e-8, 3.0 - 1e-8, 3.0]))
    log_cols = [i for i, c in enumerate(model.columns) if c.nonnegative]
    assert len(log_cols) == 2
    lo_col, up_col = log_cols
    assert vals.phi[0, lo_col] == np.inf and vals.phi[3, up_col] == np.inf
    np.testing.assert_allclose(vals.d2phi[1, lo_col], 1e16, rtol=1e-6)
    np.testing.assert_allclose(vals.d2phi[2, up_col], 1e16, rtol=1e-6)
    # A zero log coefficient never produces 0 * inf at its endpoint.
    theta = np.ones(model.n_coefficients)
    theta[lo_col] = 0.0
    q, dq, d2q = _potential(model, theta, np.array([-1.0]))
    assert np.isfinite(q[0]) and np.isfinite(dq[0]) and np.isfinite(d2q[0])


def test_points_outside_support_rejected():
    model = _compile(BasisSpec(support=(0.0, INF)))
    with pytest.raises(ValueError, match="closed support"):
        _evaluate_basis(model, np.array([-0.5, 1.0]))


def test_potential_derivatives_are_the_same_linear_map():
    model = _compile(
        BasisSpec(bulk_degree=4, lower_tail=SoftPower(1.5), upper_tail=Exponential(0.6))
    )
    theta = RNG.normal(size=model.n_coefficients)
    x = np.linspace(-5.0, 8.0, 50)
    vals = _evaluate_basis(model, x)
    q, dq, d2q = _potential(model, theta, x)
    for got, ref in (
        (q, vals.phi @ theta),
        (dq, vals.dphi @ theta),
        (d2q, vals.d2phi @ theta),
    ):
        # Same linear map up to summation order; zero crossings need a scale.
        np.testing.assert_allclose(
            got, ref, rtol=1e-13, atol=1e-13 * np.max(np.abs(ref))
        )


def test_tiny_endpoint_distances_stay_exact_in_point_statistics():
    # Distances from subnormal up are formed physically, not after rounding
    # through the fitting coordinate.
    x = np.array([np.nextafter(0.0, 1.0), 1e-300, 1e-100, 1e-20, 1e-5, 0.01, 0.4])
    spec = BasisSpec(support=(0.0, math.inf), bulk_degree=2, log_lower=True)
    model = _build_basis(spec, x)
    i = [c.label for c in model.columns].index("logL")
    phi = _evaluate_basis(model, x, max_order=0).phi
    np.testing.assert_allclose(phi[:, i], -np.log(x), rtol=2e-14)
    stats = _point_statistics(model, x)
    assert np.all(np.isfinite(stats.mean))
    assert stats.mean[i] == pytest.approx(float(np.mean(-np.log(x))), rel=2e-14)
