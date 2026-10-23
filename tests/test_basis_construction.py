"""G3 Stage 1a: BasisSpec canonicalization, geometry and exact compilation.

Defining properties checked here are exact (rational arithmetic) wherever the
compiler claims exactness: structural zeros, suppression and shared-amplitude
equalities hold identically for every reduced coefficient vector.  Feasibility
of outward-positive leading amplitudes is checked with an independent LP.
Numerical basis evaluation is covered by the Stage 1b evaluator tests.
"""

import itertools
import math
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import linprog

from gibbus import (
    BasisSpec,
    Exponential,
    GibbusBasisWarning,
    Polynomial,
    SoftPower,
    TanhSpec,
)
from gibbus._model.basis import _build_basis, _limit_functional, _rank
from gibbus._model.basis_geometry import _GeometryPolicy
from gibbus._model.basis_spec import _canonicalize_basis_spec
from gibbus._model.coords import _build_fit_coordinate

INF = math.inf
RNG = np.random.default_rng(20261016)
NORMAL = RNG.normal(1.5, 2.0, size=400)
HALF = RNG.gamma(2.0, 1.5, size=400) + 0.25
BOUNDED = RNG.beta(2.0, 3.0, size=400) * 4.0 - 1.0

SUPPORT_CASES = [
    # (id, support, log_lower, log_upper, data)
    ("S01", (-INF, INF), False, False, NORMAL),
    ("S02", (0.0, INF), False, False, HALF),
    ("S03", (0.0, INF), True, False, HALF),
    ("S04", (-INF, 0.0), False, False, -HALF),
    ("S05", (-INF, 0.0), False, True, -HALF),
    ("S06", (-1.0, 3.0), False, False, BOUNDED),
    ("S07", (-1.0, 3.0), True, False, BOUNDED),
    ("S08", (-1.0, 3.0), False, True, BOUNDED),
    ("S09", (-1.0, 3.0), True, True, BOUNDED),
]

TAILS = [
    None,
    Polynomial(1),
    Polynomial(2),
    Polynomial(3),
    SoftPower(1.5),
    SoftPower(2.5),
    Exponential(0.7),
]


def _exact_matrix(model):
    """Exact reconstruction map as a list of rows."""
    return [list(r) for r in model.coefficient_map_exact]


def _apply(functional, model, j):
    """Exact ``functional . M`` restricted to function ``j`` slots."""
    exact = model.coefficient_map_exact
    return [
        sum(functional[k] * exact[3 * j + k][c] for k in range(3))
        for c in range(model.n_coefficients)
    ]


def _full_line_battery():
    """Every full-line tail/bulk/expansion/independence combination."""
    for lt, ut, bulk, expansion, indep in itertools.product(
        TAILS, TAILS, (1, 2, 3, 4), ("minimal", "full"), (False, True)
    ):
        yield BasisSpec(
            bulk_degree=bulk,
            lower_tail=lt,
            upper_tail=ut,
            expansion=expansion,
            independent_equal_rate_coefficients=indep,
        )


def _half_line_battery():
    """Half-line and bounded combinations with optional logs."""
    for (_, support, ll, lu, _data), bulk, expansion in itertools.product(
        SUPPORT_CASES[1:], (1, 2, 3, 4), ("minimal", "full")
    ):
        tails = [None]
        if support[1] == INF or support[0] == -INF:
            tails = TAILS
        for tail in tails:
            kwargs = {}
            if support[1] == INF:
                kwargs["upper_tail"] = tail
            elif support[0] == -INF:
                kwargs["lower_tail"] = tail
            yield BasisSpec(
                support=support,
                bulk_degree=bulk,
                log_lower=ll,
                log_upper=lu,
                expansion=expansion,
                **kwargs,
            )


def _data_for(spec):
    lo, up = spec.support
    if lo == -INF and up == INF:
        return NORMAL
    if up == INF:
        return HALF + lo
    if lo == -INF:
        return up - HALF
    return BOUNDED


def _compile_quiet(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, _data_for(spec))


def _is_linear(tail):
    return isinstance(tail, Polynomial) and tail.power == 1


def _is_expected_rejection(spec):
    """Full-line bulk degree 1 without a faster tail is never proper and convex.

    Either no growth reaches linear order (both tails unspecified), or both
    ends are linear and the gate corrections force ``q''`` negative far out at
    one of them (G3-025 rule 2).
    """
    if spec.support != (-INF, INF) or spec.bulk_degree != 1:
        return False
    lt, ut = spec.lower_tail, spec.upper_tail
    return (lt is None or _is_linear(lt)) and (ut is None or _is_linear(ut))


BATTERY = list(_full_line_battery()) + list(_half_line_battery())


# ---------------------------------------------------------------------------
# Validation and canonicalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Polynomial(0),
        lambda: SoftPower(0.5),
        lambda: SoftPower(float("nan")),
        lambda: SoftPower(1.5, delta=0.0),
        lambda: SoftPower(1.5, delta="smooth"),
        lambda: Exponential(0.0),
        lambda: Exponential(-1.0),
        lambda: TanhSpec(lower_location=1.0, upper_location=1.0),
        lambda: TanhSpec(lower_steepness=0.0),
        lambda: BasisSpec(support=(1.0, 1.0)),
        lambda: BasisSpec(support=(INF, INF)),
        lambda: BasisSpec(support=(0.0, float("nan"))),
        lambda: BasisSpec(bulk_degree=0),
        lambda: BasisSpec(log_lower=True),
        lambda: BasisSpec(support=(0.0, INF), log_upper=True),
        lambda: BasisSpec(support=(0.0, INF), lower_tail=Polynomial(2)),
        lambda: BasisSpec(support=(-INF, 0.0), upper_tail=Exponential(1.0)),
        lambda: BasisSpec(expansion="maximal"),
        lambda: BasisSpec(support=(0.0, INF), tanh=TanhSpec(lower_location=1.0)),
        lambda: BasisSpec(support=(-INF, 0.0), tanh=TanhSpec(upper_steepness=1.0)),
    ],
)
def test_invalid_specs_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Polynomial(2.0),
        lambda: Polynomial(True),
        lambda: BasisSpec(bulk_degree=2.5),
        lambda: BasisSpec(upper_tail=2),
        lambda: BasisSpec(log_lower=1),
        lambda: BasisSpec(tanh={}),
    ],
)
def test_wrongly_typed_specs_rejected(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0, 4.0])
def test_integer_soft_power_becomes_monomial_with_warning(power):
    spec = BasisSpec(bulk_degree=2, upper_tail=SoftPower(power, delta=0.3))
    with pytest.warns(GibbusBasisWarning, match="delta is ignored"):
        request = _canonicalize_basis_spec(spec)
    assert request.upper_tail.kind == "polynomial"
    assert request.upper_tail.value == power
    assert request.upper_tail.converted_from_soft_power
    with pytest.warns(GibbusBasisWarning):
        model = _build_basis(spec, NORMAL)
    funcs = {f.label: f for f in model.functions}
    assert not funcs["powU"].active and not funcs["powL"].active
    k = int(power)
    assert funcs["x" if k == 1 else f"x^{k}"].active
    assert model.monomial_degree == max(2, k)


def test_noninteger_soft_power_keeps_soft_slot_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", GibbusBasisWarning)
        model = _build_basis(BasisSpec(upper_tail=SoftPower(2.5)), NORMAL)
    funcs = {f.label: f for f in model.functions}
    assert funcs["powU"].active and funcs["powU"].power == 2.5
    assert model.monomial_degree == 2


def test_function_enumeration_has_n_plus_seven_slots():
    model = _build_basis(BasisSpec(bulk_degree=3, lower_tail=Polynomial(5)), NORMAL)
    assert model.monomial_degree == 5
    assert len(model.functions) == 5 + 7
    assert len(model.slot_names) == 3 * (5 + 7)
    labels = [f.label for f in model.functions]
    assert labels == [
        "1",
        "x",
        "x^2",
        "x^3",
        "x^4",
        "x^5",
        "logL",
        "logU",
        "powL",
        "powU",
        "expL",
        "expU",
    ]
    # Intermediate monomial x^4 is above the bulk and not requested: inactive.
    assert not model.functions[4].active and model.functions[5].active


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_auto_geometry_matches_policy_and_g1_coordinate():
    policy = _GeometryPolicy()
    model = _build_basis(
        BasisSpec(lower_tail=SoftPower(1.5), upper_tail=Exponential(0.5)), NORMAL
    )
    g = model.geometry
    # Center/scale are exactly G1's robust fitting coordinate (median, MAD).
    ref = _build_fit_coordinate((-INF, INF), NORMAL, None, None)
    assert (g.center, g.scale) == (ref.center, ref.scale)
    assert g.center == pytest.approx(np.median(NORMAL))
    mad = np.median(np.abs(NORMAL - np.median(NORMAL)))
    assert g.scale == pytest.approx(1.4826 * mad)
    s = g.scale
    assert g.lower_location == pytest.approx(np.quantile(NORMAL, policy.lower_quantile))
    assert g.upper_location == pytest.approx(np.quantile(NORMAL, policy.upper_quantile))
    assert g.lower_steepness == pytest.approx(policy.steepness_kappa / s)
    assert g.lower_delta == pytest.approx((policy.soft_power_eta * s) ** 2)
    assert g.upper_delta is None and g.upper_rate == 0.5 and g.lower_rate is None


def test_weighted_geometry_follows_component_responsibilities():
    w = np.where(NORMAL > np.median(NORMAL), 1.0, 0.05)
    model = _build_basis(BasisSpec(), NORMAL, w)
    ref = _build_fit_coordinate((-INF, INF), NORMAL, w, None)
    assert (model.geometry.center, model.geometry.scale) == (ref.center, ref.scale)
    assert model.geometry.center > np.median(NORMAL)


@pytest.mark.parametrize("support,data", [((-INF, 0.0), -HALF), ((0.0, INF), HALF)])
def test_half_line_coordinate_keeps_the_user_orientation(support, data):
    model = _build_basis(BasisSpec(support=support), data)
    coordinate = model.geometry.coordinate
    assert coordinate.canonical_support == pytest.approx(
        tuple((np.asarray(support) - coordinate.center) / coordinate.scale)
    )


def test_explicit_geometry_with_fixed_coordinate_needs_no_data():
    coord = _build_fit_coordinate((-INF, INF), NORMAL, None, None)
    spec = BasisSpec(
        lower_tail=SoftPower(1.5, delta=0.25),
        tanh=TanhSpec(
            lower_location=-1.0,
            upper_location=2.0,
            lower_steepness=3.0,
            upper_steepness=0.5,
        ),
    )
    model = _build_basis(spec, None, None, None, coord)
    g = model.geometry
    assert g.coordinate is coord
    assert (g.lower_location, g.upper_location) == (-1.0, 2.0)
    assert (g.lower_steepness, g.upper_steepness) == (3.0, 0.5)
    assert g.lower_delta == 0.25


def test_missing_data_rejected():
    with pytest.raises(ValueError, match="requires data or an explicit coordinate"):
        _build_basis(BasisSpec())
    coord = _build_fit_coordinate((-INF, INF), NORMAL, None, None)
    with pytest.raises(ValueError, match="tanh location requires data"):
        _build_basis(BasisSpec(), None, None, None, coord)


def test_explicit_location_order_checked_after_auto_fill():
    spec = BasisSpec(tanh=TanhSpec(lower_location=50.0))
    with pytest.raises(ValueError, match="x_L < x_U"):
        _build_basis(spec, NORMAL)


def test_tied_data_get_ordered_tanh_locations():
    data = np.array([3.0] * 18 + [0.0, 6.0])
    model = _build_basis(BasisSpec(), data)
    g = model.geometry
    assert g.lower_location < g.upper_location
    assert g.lower_location == pytest.approx(3.0 - g.scale)


def test_unavailable_tanh_geometry_is_none():
    model = _build_basis(BasisSpec(support=(0.0, INF)), HALF)
    assert (
        model.geometry.lower_location is None and model.geometry.lower_steepness is None
    )
    assert model.geometry.upper_location is not None


# ---------------------------------------------------------------------------
# Structural zeros and exactness (all nine support/log configurations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", SUPPORT_CASES, ids=[c[0] for c in SUPPORT_CASES])
@pytest.mark.parametrize("expansion", ["minimal", "full"])
def test_nine_support_log_configurations(case, expansion):
    _, support, ll, lu, data = case
    model = _build_basis(
        BasisSpec(
            support=support,
            bulk_degree=3,
            log_lower=ll,
            log_upper=lu,
            expansion=expansion,
        ),
        data,
    )
    exact = _exact_matrix(model)
    funcs = {f.label: f for f in model.functions}
    # Unavailable slots are identically zero in every column.
    for slot, ok in enumerate(model.available):
        if not ok:
            assert all(v == 0 for v in exact[slot])
    # a_0 = 0; finite lower end => no b; finite upper end => no c; logs ungated.
    assert not model.available[0]
    for f in model.functions:
        if support[0] != -INF:
            assert not model.available[3 * f.index + 1]
        if support[1] != INF:
            assert not model.available[3 * f.index + 2]
    for name, enabled in (("logL", ll), ("logU", lu)):
        f = funcs[name]
        assert f.active == enabled
        assert (
            not model.available[3 * f.index + 1]
            and not model.available[3 * f.index + 2]
        )
        if enabled:
            cols = [c for c in model.columns if any(t[0] == f.index for t in c.terms)]
            assert len(cols) == 1 and cols[0].nonnegative
            assert cols[0].terms == ((f.index, (1, 0, 0)),)
    # Exactly one asymptotic record per infinite end.
    assert [a.side for a in model.asymptotics] == [
        s for s, e in (("lower", support[0]), ("upper", support[1])) if math.isinf(e)
    ]
    # Bounded supports: the plain monomial/log family (no gating, nothing suppressed).
    if all(math.isfinite(e) for e in support):
        assert model.n_coefficients == 3 + int(ll) + int(lu)


@pytest.mark.parametrize("spec", BATTERY, ids=lambda s: repr(s)[:0])
def test_battery_exact_invariants(spec):
    if _is_expected_rejection(spec):
        with pytest.raises(ValueError, match="structurally improper"):
            _compile_quiet(spec)
        return
    model = _compile_quiet(spec)
    exact = _exact_matrix(model)
    d = model.n_coefficients
    # Exact full column rank of the reconstruction map.
    assert _rank([list(r) for r in exact]) == d
    # Every recorded equality holds identically.
    for row in model.equality_rows:
        for c in range(d):
            assert sum(v * exact[slot][c] for slot, v in row.items()) == 0
    # Suppressed functions have an exactly vanishing gate limit at that end.
    for side in model.asymptotics:
        for j in side.suppressed:
            g = _limit_functional(model.functions[j], side.side, model.request)
            assert all(v == 0 for v in _apply(g, model, j))
        # Surviving orders never exceed the target; the target is first.
        assert side.terms[0].order == side.target_order
        assert all(t.order <= side.target_order for t in side.terms)
    # Float map is the rounded exact map.
    assert np.array_equal(
        model.coefficient_map,
        np.array([[float(v) for v in r] for r in exact]).reshape(-1, d),
    )
    # Independent LP: outward-positive leading amplitudes are jointly feasible.
    rows = [side.leading.row for side in model.asymptotics]
    if rows:
        a = -np.vstack(rows)
        res = linprog(
            np.zeros(d), A_ub=a, b_ub=-np.ones(len(rows)), bounds=[(None, None)] * d
        )
        assert res.status == 0, "no outward-positive leading amplitude"


@pytest.mark.parametrize(
    "spec", [s for s in BATTERY if s.expansion == "minimal"], ids=lambda s: ""
)
def test_minimal_span_is_contained_in_full_span(spec):
    if _is_expected_rejection(spec):
        return
    minimal = _compile_quiet(spec)
    full = _compile_quiet(BasisSpec(**{**spec.__dict__, "expansion": "full"}))
    mf = _exact_matrix(full)
    mm = _exact_matrix(minimal)
    stacked = [rf + rm for rf, rm in zip(mf, mm)]
    assert _rank(stacked) == full.n_coefficients
    assert minimal.n_coefficients <= full.n_coefficients


# ---------------------------------------------------------------------------
# Specification examples and resolved policies
# ---------------------------------------------------------------------------


def _gate_of(model, label):
    j = next(f.index for f in model.functions if f.label == label)
    gates = [t[1] for c in model.columns for t in c.terms if t[0] == j]
    return gates


def test_odd_degree_rule_full_line_unspecified_tails():
    model = _build_basis(BasisSpec(bulk_degree=3), NORMAL)
    assert _gate_of(model, "x^3") == [(0, 1, 1)]  # x^3 (tL + tU)
    # Lower-order completion (G3-025): x^k (tL + tU) for k < 3, ungated x^2 kept.
    assert sorted(_gate_of(model, "x^2")) == [(0, 1, 1), (1, 0, 0)]
    assert sorted(_gate_of(model, "x")) == [(0, 1, 1), (1, 0, 0)]
    assert _gate_of(model, "1") == [(0, 1, 1)]
    for side in model.asymptotics:
        assert side.target_order == (1, 2.0, 0)


def test_degree_one_full_line_is_rejected_unless_a_faster_tail_or_half_line_exists():
    with pytest.raises(ValueError, match="structurally improper"):
        _build_basis(BasisSpec(bulk_degree=1), NORMAL)
    # Linear tails from gated polynomials of degree <= 1 are never convex.
    with pytest.raises(ValueError, match="eventually nonnegative"):
        _build_basis(
            BasisSpec(
                bulk_degree=1, lower_tail=Polynomial(1), upper_tail=Polynomial(1)
            ),
            NORMAL,
        )
    # M-O01: degree 1 is fine once a faster tail is requested or on a half-line.
    _build_basis(
        BasisSpec(bulk_degree=1, lower_tail=Polynomial(2), upper_tail=Polynomial(2)),
        NORMAL,
    )
    _build_basis(BasisSpec(support=(0.0, INF), bulk_degree=1), HALF)


def test_minimal_lower_suppression_matches_spec_example():
    # x^4 above a lower Poly(2) tail; the upper end requests degree 4.
    model = _build_basis(
        BasisSpec(bulk_degree=2, lower_tail=Polynomial(2), upper_tail=Polynomial(4)),
        NORMAL,
    )
    assert _gate_of(model, "x^4") == [(1, 1, 0)]  # a = b, c = 0: x^4 (1 + tL)


def test_minimal_upper_suppression_matches_spec_example():
    model = _build_basis(
        BasisSpec(bulk_degree=2, lower_tail=Polynomial(4), upper_tail=Polynomial(2)),
        NORMAL,
    )
    assert _gate_of(model, "x^4") == [(1, 0, 1)]  # a = c, b = 0: x^4 (1 + tU)


def test_full_expansion_lower_suppression_keeps_both_gates():
    model = _build_basis(
        BasisSpec(
            bulk_degree=2,
            lower_tail=Polynomial(2),
            upper_tail=Polynomial(4),
            expansion="full",
        ),
        NORMAL,
    )
    assert sorted(_gate_of(model, "x^4")) == sorted([(1, 1, 0), (-1, 0, 1)])


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_shared_polynomial_amplitude_is_exactly_equal(k):
    model = _build_basis(
        BasisSpec(bulk_degree=2, lower_tail=Polynomial(k), upper_tail=Polynomial(k)),
        NORMAL,
    )
    lower, upper = model.asymptotics
    assert lower.leading.row_exact == upper.leading.row_exact
    assert any(v != 0 for v in lower.leading.row_exact)
    if k % 2 == 1:
        assert all(
            g[0] == 0 for g in _gate_of(model, "x" if k == 1 else f"x^{k}")
        )  # a_k = 0


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("expansion", ["minimal", "full"])
def test_independent_polynomial_amplitudes_have_rank_two(k, expansion):
    model = _build_basis(
        BasisSpec(
            bulk_degree=2,
            lower_tail=Polynomial(k),
            upper_tail=Polynomial(k),
            independent_equal_rate_coefficients=True,
            expansion=expansion,
        ),
        NORMAL,
    )
    lower, upper = model.asymptotics
    assert _rank([list(lower.leading.row_exact), list(upper.leading.row_exact)]) == 2


def test_shared_exponential_amplitude_includes_centering_factors():
    model = _build_basis(
        BasisSpec(lower_tail=Exponential(0.8), upper_tail=Exponential(0.8)), NORMAL
    )
    lower, upper = model.asymptotics
    phys_l = math.exp(lower.leading.log_factor) * lower.leading.row
    phys_u = math.exp(upper.leading.log_factor) * upper.leading.row
    np.testing.assert_allclose(phys_l, phys_u, rtol=1e-14, atol=0)
    g = model.geometry
    assert lower.leading.log_factor == pytest.approx(
        0.8 * (g.lower_location - g.center)
    )
    assert upper.leading.log_factor == pytest.approx(
        -0.8 * (g.upper_location - g.center)
    )


def test_unequal_exponential_rates_are_not_shared():
    model = _build_basis(
        BasisSpec(lower_tail=Exponential(0.8), upper_tail=Exponential(1.2)), NORMAL
    )
    lower, upper = model.asymptotics
    assert _rank([list(lower.leading.row_exact), list(upper.leading.row_exact)]) == 2


def test_equal_soft_powers_share_one_function():
    model = _build_basis(
        BasisSpec(lower_tail=SoftPower(2.5), upper_tail=SoftPower(2.5)), NORMAL
    )
    assert model.merged_soft_power
    funcs = {f.label: f for f in model.functions}
    assert funcs["powL"].active and not funcs["powU"].active
    lower, upper = model.asymptotics
    assert lower.leading.row_exact == upper.leading.row_exact


def test_equal_soft_powers_with_distinct_delta_share_combined_amplitude():
    model = _build_basis(
        BasisSpec(
            lower_tail=SoftPower(2.5, delta=0.5), upper_tail=SoftPower(2.5, delta=2.0)
        ),
        NORMAL,
    )
    assert not model.merged_soft_power
    lower, upper = model.asymptotics
    assert set(lower.leading.functions) == {
        model.monomial_degree + 3,
        model.monomial_degree + 4,
    }
    assert lower.leading.row_exact == upper.leading.row_exact


def test_mixed_tail_classes_suppress_dominating_terms_exactly():
    # Lower soft power 1.5, upper polynomial 3: x^2 and x^3 dominate the lower end.
    model = _build_basis(
        BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.5), upper_tail=Polynomial(3)),
        NORMAL,
    )
    lower, upper = model.asymptotics
    labels = {model.functions[j].label for j in lower.suppressed}
    assert labels == {"x^2", "x^3"}
    assert upper.suppressed == ()
    assert lower.target_order == (1, 1.5, 0) and upper.target_order == (1, 3.0, 0)
    # x^2 must still lead nothing at the upper end beyond its own order.
    assert [t.order for t in upper.terms][:2] == [(1, 3.0, 0), (1, 2.0, 0)]


def test_linear_tail_with_unspecified_opposite_end_is_outward_feasible():
    # M-O06: leading affine growth must be outward on both ends.
    model = _build_basis(BasisSpec(bulk_degree=2, lower_tail=Polynomial(1)), NORMAL)
    rows = np.vstack([side.leading.row for side in model.asymptotics])
    res = linprog(
        np.zeros(model.n_coefficients),
        A_ub=-rows,
        b_ub=-np.ones(2),
        bounds=[(None, None)] * model.n_coefficients,
    )
    assert res.status == 0


def test_exponential_with_unspecified_opposite_end():
    model = _build_basis(BasisSpec(bulk_degree=3, lower_tail=Exponential(1.0)), NORMAL)
    lower, upper = model.asymptotics
    assert lower.target_order == (2, 1.0, 0)
    assert upper.target_order == (1, 3.0, 0)
    # The upper bulk leader x^3 is subordinate (not suppressed) at the lower end.
    assert lower.suppressed == () and upper.suppressed == ()


def test_physical_power_rows_are_in_user_units():
    spec = BasisSpec(bulk_degree=2)
    m1 = _build_basis(spec, NORMAL)
    m2 = _build_basis(spec, 10.0 * NORMAL)
    # Internal coefficient vectors differ by the scale, physical leading rows by h^-2.
    h1, h2 = m1.geometry.scale, m2.geometry.scale
    assert m1.asymptotics[1].leading.row[1] == pytest.approx(h1**-2)
    assert m2.asymptotics[1].leading.row[1] == pytest.approx(h2**-2)


def test_full_coefficients_reconstruct_exactly():
    model = _build_basis(
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(2), upper_tail=Polynomial(2)),
        NORMAL,
    )
    theta = RNG.normal(size=model.n_coefficients)
    full = model.full_coefficients(theta)
    exact_theta = [Fraction(float(t)) for t in theta]
    for row in model.equality_rows:
        val = sum(
            v
            * sum(
                model.coefficient_map_exact[slot][c] * exact_theta[c]
                for c in range(model.n_coefficients)
            )
            for slot, v in row.items()
        )
        assert val == 0
    assert full.shape == (3 * len(model.functions),)


def test_cached_layout_and_selection_are_the_uncached_computation():
    # _basis_layout is cached per (request, geometry) and select per slot
    # choice (G3-041); both must return exactly what a fresh computation does.
    from gibbus._model.basis import _basis_layout

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        model = _build_basis(
            BasisSpec(
                bulk_degree=3, lower_tail=SoftPower(1.5), upper_tail=SoftPower(1.5)
            ),
            NORMAL,
        )
    cached = _basis_layout(model.request, model.geometry)
    assert _basis_layout(model.request, model.geometry) is cached
    fresh = _basis_layout.__wrapped__(model.request, model.geometry)
    for slots in (cached.full_slots(), model.slot_choice):
        a, b = cached.select(slots), fresh._select(tuple(slots))
        assert cached.select(slots) is a
        assert a[1] == b[1] and a[3] == b[3]
        assert [c.label for c in a[0]] == [c.label for c in b[0]]
        assert np.array_equal(a[2], b[2]) and not a[2].flags.writeable
        for sa, sb in zip(a[4], b[4], strict=True):
            assert [t.row_exact for t in sa.terms] == [t.row_exact for t in sb.terms]
