"""Independent Decimal jets of the original q, not production recurrences."""

import math
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal as D
from decimal import localcontext

import numpy as np
import pytest
from _series_reference import conv, oracle_atom, oracle_derivatives, oracle_gate
from test_basis_evaluation import ORACLE_SPECS, _compile, _test_points

from gibbus._model.basis_eval import _potential
from gibbus._model.basis_series import SeriesPlan


def reference(model, theta, x, ell, N):
    g = model.geometry
    with localcontext() as ctx:
        ctx.prec = 150
        gates = []
        for a, b in (
            (g.lower_steepness, g.lower_location),
            (g.upper_steepness, g.upper_location),
        ):
            if a is None:
                gates.append([D(0)] * (N + 5))
            else:
                if len(gates) == 1:
                    a = -a
                gates.append(oracle_gate(a, b, x, ell, N + 4))
        q = [D(0)] * (N + 5)
        for j, fn in enumerate(model.functions):
            slots = [
                sum(
                    (
                        D(v.numerator) / D(v.denominator) * D(float(t))
                        for v, t in zip(row, theta)
                    ),
                    D(0),
                )
                for row in model.coefficient_map_exact[3 * j : 3 * j + 3]
            ]
            if not any(slots):
                continue
            if fn.kind == "monomial":
                kind, pars = "poly", (fn.degree,)
            elif fn.kind == "log":
                kind, pars = "log", (
                    (1, model.request.lower)
                    if fn.side == "lower"
                    else (-1, model.request.upper)
                )
            elif fn.kind == "soft_power":
                kind, pars = "soft", (fn.power, fn.delta)
            else:
                kind, pars = "exp", (
                    -fn.rate if fn.side == "lower" else fn.rate,
                    fn.location,
                )
            f = oracle_atom(kind, pars, x, g.scale, g.center, ell, N + 4)
            weight = [
                slots[1] * gates[0][n] + slots[2] * gates[1][n] for n in range(N + 5)
            ]
            weight[0] += slots[0]
            term = conv(f, weight, N + 4)
            q = [a + b for a, b in zip(q, term)]
        return np.array(
            [[float(v) for v in row] for row in oracle_derivatives(q, ell, N)[2:]]
        )


@pytest.mark.parametrize("spec", ORACLE_SPECS)
def test_all_basis_families(spec):
    model = _compile(spec)
    plan = SeriesPlan(model)
    theta = np.random.default_rng(32).normal(size=model.n_coefficients)
    bulk, tail = _test_points(model)
    centers = np.concatenate((bulk[2:4], tail[:1]))
    series = plan.coefficients(theta, centers, 24)
    cached = plan.cache(centers, 24).coefficients(theta)
    np.testing.assert_allclose(
        cached.coefficients, series.coefficients, rtol=2e-13, atol=1e-13
    )
    for i, x in enumerate(centers):
        expected = reference(model, theta, float(x), float(series.scales[i]), 24)
        scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1e-300)
        assert np.max(np.abs(series.coefficients[i] - expected) / scale) < 3e-12
        dx = series.scales[i] * np.array([-0.2, 0, 0.2])
        direct = np.array(_potential(model, theta, x + dx, max_order=4)[2:]).T
        np.testing.assert_allclose(
            series.evaluate_offsets(dx, center_index=i), direct, rtol=3e-11, atol=1e-11
        )


@pytest.mark.parametrize("order", [0, 1, 2, 8, 48])
def test_orders_and_gate_centers(order):
    model = _compile(ORACLE_SPECS[14])
    g = model.geometry
    plan = SeriesPlan(model)
    theta = np.arange(model.n_coefficients) * 0.125
    centers = [g.center, g.lower_location, g.upper_location]
    s = plan.coefficients(theta, centers, order)
    for i, x in enumerate(centers):
        expected = reference(model, theta, x, float(s.scales[i]), order)
        scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1e-300)
        assert np.max(np.abs(s.coefficients[i] - expected) / scale) < 5e-12


def test_validation_and_immutability():
    model = _compile(ORACLE_SPECS[8])
    p = SeriesPlan(model)
    theta = np.ones(model.n_coefficients)
    for bad in [-1, 4097]:
        with pytest.raises(ValueError):
            p.coefficients(theta, 1, bad)
    with pytest.raises(TypeError):
        p.coefficients(theta, 1, 2.0)
    with pytest.raises(ValueError):
        p.coefficients(theta, [], 4)
    with pytest.raises(ValueError):
        p.coefficients(theta, -1, 4)
    with pytest.raises(ValueError):
        p.coefficients(theta, 1, 4, scale=2)
    with pytest.raises(ValueError):
        p.coefficients(theta[:-1], 1, 4)
    s = p.coefficients(theta, 1, 24)
    with pytest.raises(ValueError):
        s.mantissa.setflags(write=True)
    with pytest.raises(IndexError):
        s.evaluate(1, center_index=1)
    with pytest.raises(TypeError):
        s.evaluate(1, center_index=0.5)
    with pytest.raises(ValueError):
        s.evaluate_offsets(s.radius[0])
    with pytest.raises(ValueError):
        s.evaluate(np.nan)
    assert s.evaluate([]).shape == (0, 3)
    assert s.evaluate([[1, 1], [1, 1]]).shape == (2, 2, 3)


def test_shared_cache_threads():
    model = _compile(ORACLE_SPECS[18])
    p = SeriesPlan(model)
    cache = p.cache([model.geometry.center], 24)
    ts = np.random.default_rng(11).normal(size=(16, model.n_coefficients))

    def run(t):
        return cache.coefficients(t).evaluate_scaled(np.linspace(-0.2, 0.2, 200))

    serial = [run(t) for t in ts]
    with ThreadPoolExecutor(4) as pool:
        parallel = list(pool.map(run, ts))
    for a, b in zip(serial, parallel):
        np.testing.assert_array_equal(a, b)


def single_atom(
    kind,
    *,
    slots=(1, 0, 0),
    side="upper",
    degree=None,
    power=None,
    delta=None,
    rate=None,
    location=0.0,
    a=1.0,
    b=0.0,
):
    """Isolate an atomic formula using the real compiled BasisKernel table."""
    from dataclasses import replace
    from fractions import Fraction
    from types import SimpleNamespace

    from gibbus._model._basis_kernels import BasisKernel

    from gibbus._model.basis import _BasisColumn, _BasisFunction

    base = _compile(ORACLE_SPECS[0])
    geometry = replace(base.geometry, lower_steepness=a, lower_location=b)
    f = _BasisFunction(
        0, kind, side, degree, power, delta, rate, location, True, "test"
    )
    slot = tuple(map(Fraction, slots))
    model = SimpleNamespace(
        geometry=geometry,
        request=base.request,
        functions=(f,),
        columns=(_BasisColumn(((0, slot),), "test", False),),
        n_coefficients=1,
        coefficient_map_exact=tuple((v,) for v in slot),
        coefficient_map=np.array(slot, dtype=float)[:, None],
    )
    model.kernel = BasisKernel(model)
    return model


@pytest.mark.parametrize("theta", [1.0, 1e308])
def test_fused_exponential_underflow_gate(theta):
    # exp(800)*(1-tanh(450)) ~ 2 exp(-100), including physical derivatives.
    model = single_atom("exponential", slots=(1, -1, 0), rate=1.0, location=-350.0)
    p = SeriesPlan(model)
    s = p.coefficients([theta], 450.0, 16, scale=0.1)
    cached = p.cache(450.0, 16, scale=0.1).coefficients([theta])
    x = np.array([-0.01, 0, 0.01])
    expected = (
        theta * (2 * math.exp(-100)) * np.exp(-x)[:, None] * np.array([1.0, -1.0, 1.0])
    )
    np.testing.assert_allclose(s.evaluate_offsets(x), expected, rtol=1e-13)
    np.testing.assert_allclose(cached.evaluate_offsets(x), expected, rtol=1e-13)
    assert s.range_fallbacks == 1


def test_underflow_seed_recovered_by_derivative_rates():
    model = single_atom("exponential", rate=1e100)
    s = SeriesPlan(model).coefficients([1.0], -8e-98, 8, scale=1e-101)
    with localcontext() as ctx:
        ctx.prec = 80
        x = D(-8e-98)
        lam = D(1e100)
        expected = np.array([float((lam * x).exp() * lam**r) for r in (2, 3, 4)])
    np.testing.assert_allclose(s.evaluate_offsets(0), expected, rtol=2e-13, atol=0)
    assert s.range_fallbacks == 1


def test_extended_horner_cancellation():
    from gibbus._model import _series_kernels as k

    # Three derivative rows of 1 + A*(u**2-u), A outside float64 range.
    m = np.array([[[1.0, -0.5, 0.5]] * 3])
    e = np.array([[[0, 1030, 1030]] * 3], dtype=np.int64)
    out = k._evaluate(
        m,
        e,
        np.array([1], dtype=np.uint8),
        np.array([1.0]),
        np.array([0], dtype=np.intp),
    )
    np.testing.assert_array_equal(out, np.ones((1, 3)))
    with pytest.raises(OverflowError):
        k._evaluate(
            m,
            e,
            np.array([1], dtype=np.uint8),
            np.array([2.0]),
            np.array([0], dtype=np.intp),
        )


def test_log_boundary_and_fractional_radius():
    p = SeriesPlan(_compile(ORACLE_SPECS[8]))
    model = p._model
    x = np.nextafter(model.request.lower, model.request.upper)
    s = p.coefficients(np.ones(model.n_coefficients), x, 12)
    assert s.radius[0] == x - model.request.lower
    np.testing.assert_allclose(
        s.evaluate_offsets(0), s.coefficients[0, :, 0], rtol=1e-14
    )
    m = single_atom("soft_power", power=1.5, delta=0.01)
    s = SeriesPlan(m).coefficients([1.0], m.geometry.center, 12, scale=0.01)
    assert s.radius[0] <= 0.1
    with pytest.raises(ValueError):
        s.evaluate_offsets(0.1)


def test_gated_constant_and_linear():
    for degree in (0, 1):
        m = single_atom("monomial", degree=degree, slots=(0, 1, 0))
        s = SeriesPlan(m).coefficients([1.0], 0.25, 24, scale=0.1)
        expected = reference(m, [1.0], 0.25, 0.1, 24)
        np.testing.assert_allclose(s.coefficients[0], expected, rtol=3e-12, atol=1e-14)
        assert np.any(s.coefficients[0, :, 0] != 0)


def test_flat_fused_rate_preserves_tiny_derivatives():
    # exp(2x)*(1-tanh(x)) = 2/(1+exp(-2x)). Its nonconstant
    # derivatives must survive even when exp(-2x) itself underflows.
    model = single_atom("exponential", slots=(1, -1, 0), rate=2.0)
    p = SeriesPlan(model)
    x = 450.0
    theta = 1e308
    s = p.coefficients([theta], x, 16, scale=0.1)
    cached = p.cache(x, 16, scale=0.1).coefficients([theta])
    with localcontext() as ctx:
        ctx.prec = 100
        amplitude = float(D(theta) * D(-900).exp())
    expected = amplitude * np.array([-8.0, 16.0, -32.0])
    np.testing.assert_allclose(s.evaluate_offsets(0), expected, rtol=2e-13, atol=0)
    np.testing.assert_allclose(cached.evaluate_offsets(0), expected, rtol=2e-13, atol=0)


def test_face_and_recenter():
    from gibbus._model.basis import _face_model

    model = _compile(ORACLE_SPECS[10])
    model, _ = _face_model(model, ("lower",))
    p = SeriesPlan(model)
    theta = np.ones(model.n_coefficients)
    centers = [model.geometry.center - 10 * model.geometry.scale, model.geometry.center]
    batch = p.coefficients(theta, centers, 32)
    for i, x in enumerate(centers):
        single = p.coefficients(theta, x, 32)
        np.testing.assert_array_equal(single.coefficients[0], batch.coefficients[i])
        expected = reference(model, theta, x, float(single.scales[0]), 32)
        scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1e-300)
        assert np.max(np.abs(single.coefficients[0] - expected) / scale) < 5e-12


def test_polynomial_exact_at_origin_and_no_tiny_theta_pruning():
    m = single_atom("monomial", degree=4)
    x = m.geometry.center
    s = SeriesPlan(m).coefficients([1e-300], x, 6, scale=0.1)
    expected = reference(m, [1e-300], x, 0.1, 6)
    np.testing.assert_allclose(s.coefficients[0], expected, rtol=1e-14, atol=0)
    assert s.coefficients[0, 2, 0] != 0


def test_high_order_near_singularities():
    for index in (8, 14, 18):
        m = _compile(ORACLE_SPECS[index])
        p = SeriesPlan(m)
        theta = np.linspace(-1, 1, m.n_coefficients)
        x = m.geometry.center
        s = p.coefficients(theta, x, 48)
        expected = reference(m, theta, x, float(s.scales[0]), 48)
        scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1e-300)
        assert np.max(np.abs(s.coefficients[0] - expected) / scale) < 5e-12
        # A modest offset makes finite-degree truncation negligible here.
        dx = 0.3 * s.scales[0]
        exact = np.array(_potential(m, theta, np.array([x + dx]), max_order=4)[2:])[
            :, 0
        ]
        np.testing.assert_allclose(
            s.evaluate_offsets(dx), exact, rtol=2e-11, atol=1e-11
        )


@pytest.mark.parametrize("index", [0, 8, 14, 18])
def test_nogil_construction_api(index):
    from gibbus._model import _series_kernels as k

    model = _compile(ORACLE_SPECS[index])
    p = SeriesPlan(model)
    theta = np.linspace(0.1, 1.0, model.n_coefficients)
    s = p.coefficients(theta, model.geometry.center, 24)
    m, e, status = k._construct_capi(
        model.kernel, p._maps, theta, float(s.centers[0]), float(s.scales[0]), 24
    )
    np.testing.assert_array_equal(np.ldexp(m, e), s.coefficients[0])
    assert status == 0
    with pytest.raises(ValueError):
        k._construct_capi(model.kernel, p._maps, theta, float(s.centers[0]), 0.0, 24)


def test_nogil_api_scaled_branch():
    from gibbus._model import _series_kernels as k

    model = single_atom("exponential", slots=(1, -1, 0), rate=2.0)
    p = SeriesPlan(model)
    theta = np.array([1e308])
    s = p.coefficients(theta, 450.0, 16, scale=0.1)
    m, e, status = k._construct_capi(model.kernel, p._maps, theta, 450.0, 0.1, 16)
    assert status == 1
    np.testing.assert_allclose(np.ldexp(m, e), s.coefficients[0], rtol=1e-14)


def test_mixed_center_query_batches():
    m = _compile(ORACLE_SPECS[14])
    p = SeriesPlan(m)
    s = p.coefficients(
        np.ones(m.n_coefficients), [m.geometry.center, m.geometry.center + 0.1], 24
    )
    for Q in range(1, 17):
        u = np.linspace(-0.2, 0.2, Q)
        for ids in (np.zeros(Q, dtype=int), np.arange(Q) % 2, np.arange(Q) // 4 % 2):
            out = s.evaluate_scaled(u, center_index=ids)
            for i in range(Q):
                np.testing.assert_array_equal(
                    out[i], s.evaluate_scaled(u[i], center_index=int(ids[i]))
                )


def test_soft_power_range_fallback():
    from dataclasses import replace

    from gibbus._model._basis_kernels import BasisKernel

    m = single_atom("soft_power", power=1.5, delta=1e-300)
    m.geometry = replace(
        m.geometry, coordinate=replace(m.geometry.coordinate, center=0.0, scale=1e150)
    )
    m.kernel = BasisKernel(m)
    p = SeriesPlan(m)
    s = p.coefficients([1.0], 0.0, 12, scale=1e-151)
    expected = reference(m, [1.0], 0.0, 1e-151, 12)
    scale = np.maximum(np.max(np.abs(expected), axis=1, keepdims=True), 1e-300)
    assert np.max(np.abs(s.coefficients[0] - expected) / scale) < 5e-13
    assert s.range_fallbacks == 1
    np.testing.assert_allclose(
        p.cache(0.0, 12, scale=1e-151).coefficients([1.0]).coefficients,
        s.coefficients,
        rtol=1e-13,
        atol=0,
    )


def test_log_derivatives_recovered_by_small_theta():
    model = _compile(ORACLE_SPECS[2])
    theta = np.zeros(model.n_coefficients)
    for i, c in enumerate(model.columns):
        if model.functions[c.terms[0][0]].kind == "log":
            theta[i] = 1e-200
    s = SeriesPlan(model).coefficients(theta, 1e-100, 12, scale=1e-101)
    np.testing.assert_allclose(
        s.evaluate_offsets(0), [1.0, -2e100, 6e200], rtol=2e-14, atol=0
    )
    assert s.range_fallbacks == 1
