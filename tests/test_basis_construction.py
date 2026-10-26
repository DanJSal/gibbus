"""BasisSpec canonicalization, fixed geometry and exact compilation.

Defining properties checked here are exact (rational arithmetic) wherever the
compiler claims exactness: structural zeros, tail-function gating,
suppression and shared-amplitude equalities hold identically for every
reduced coefficient vector.  Feasibility of outward-positive leading
amplitudes is checked with an independent LP.  Numerical basis evaluation is
covered by the evaluator tests.
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
)
from gibbus._defaults import BASIS_GATE_SHARPNESS, BASIS_SOFT_POWER_ROOT
from gibbus._model.basis import _build_basis, _limit_functional, _rank, _tail_row
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
    one of them.
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
        lambda: SoftPower(0.999),
        lambda: Exponential(0.0),
        lambda: Exponential(-1.0),
        lambda: BasisSpec(support=(1.0, 1.0)),
        lambda: BasisSpec(support=(INF, INF)),
        lambda: BasisSpec(support=(0.0, float("nan"))),
        lambda: BasisSpec(bulk_degree=0),
        lambda: BasisSpec(log_lower=True),
        lambda: BasisSpec(support=(0.0, INF), log_upper=True),
        lambda: BasisSpec(support=(0.0, INF), lower_tail=Polynomial(2)),
        lambda: BasisSpec(support=(-INF, 0.0), upper_tail=Exponential(1.0)),
        lambda: BasisSpec(expansion="maximal"),
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
    ],
)
def test_wrongly_typed_specs_rejected(factory):
    with pytest.raises(TypeError):
        factory()


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0, 4.0])
def test_integer_soft_power_becomes_polynomial_with_warning(power):
    spec = BasisSpec(bulk_degree=2, upper_tail=SoftPower(power))
    with pytest.warns(GibbusBasisWarning, match=rf"Polynomial\({int(power)}\)"):
        request = _canonicalize_basis_spec(spec)
    assert request.upper_tail.kind == "polynomial"
    assert request.upper_tail.value == power
    assert request.upper_tail.converted_from_soft_power
    with pytest.warns(GibbusBasisWarning):
        model = _build_basis(spec, NORMAL)
    funcs = {f.label: f for f in model.functions}
    assert not funcs["powU"].active and not funcs["powL"].active
    k = int(power)
    assert funcs["z" if k == 1 else f"z^{k}"].active
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
        "z",
        "z^2",
        "z^3",
        "z^4",
        "z^5",
        "logL",
        "logU",
        "powL",
        "powU",
        "expL",
        "expU",
    ]
    # Intermediate monomial z^4 is above the bulk and not requested: inactive.
    assert not model.functions[4].active and model.functions[5].active
    # The requested z^5 is a tail function of the lower end only.
    assert model.functions[5].tail_sides == ("lower",)
    assert model.functions[3].tail_sides == ()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_is_the_canonical_coordinate_plus_family_constants():
    model = _build_basis(
        BasisSpec(lower_tail=SoftPower(1.5), upper_tail=Exponential(0.5)), NORMAL
    )
    g = model.geometry
    q10, q90 = np.quantile(NORMAL, (0.1, 0.9), method="hazen")
    assert g.center == pytest.approx(0.5 * (q10 + q90), rel=1e-15)
    assert g.scale == pytest.approx(0.5 * (q90 - q10), rel=1e-15)
    # The data landmarks sit exactly on the gate centers z = -1 and z = +1.
    np.testing.assert_allclose(g.to_internal([q10, q90]), [-1.0, 1.0], rtol=1e-14)
    assert g.gate_sharpness == BASIS_GATE_SHARPNESS == 2.5066282746310007
    assert g.soft_power_root == BASIS_SOFT_POWER_ROOT == 4
    assert g.has_lower_gate and g.has_upper_gate
    # Exponential rates are canonical: lambda = r h.
    assert g.upper_rate == 0.5 * g.scale and g.lower_rate is None


def test_weighted_geometry_follows_component_responsibilities():
    w = np.where(NORMAL > np.median(NORMAL), 1.0, 0.05)
    model = _build_basis(BasisSpec(), NORMAL, w)
    ref = _build_fit_coordinate((-INF, INF), NORMAL, w)
    assert (model.geometry.center, model.geometry.scale) == (ref.center, ref.scale)
    assert model.geometry.center > np.median(NORMAL)


@pytest.mark.parametrize(
    "support,data,canonical",
    [((-INF, 0.0), -HALF, (-INF, 0.0)), ((0.0, INF), HALF, (0.0, INF))],
)
def test_half_line_coordinate_anchors_the_endpoint_at_zero(support, data, canonical):
    model = _build_basis(BasisSpec(support=support), data)
    coordinate = model.geometry.coordinate
    assert coordinate.canonical_support == canonical
    assert coordinate.center == 0.0
    # The far landmark (q90 on [0, inf), q10 on (-inf, 0]) is |z| = 1.
    level = 0.9 if support[0] == 0.0 else 0.1
    q = np.quantile(data, level, method="hazen")
    assert abs(float(coordinate.to_canonical(q))) == pytest.approx(1.0, rel=1e-15)
    assert model.geometry.has_lower_gate == (support[0] == -INF)
    assert model.geometry.has_upper_gate == (support[1] == INF)


def test_bounded_coordinate_needs_no_data():
    model = _build_basis(BasisSpec(support=(-1.0, 3.0), bulk_degree=2))
    coordinate = model.geometry.coordinate
    assert (coordinate.center, coordinate.scale) == (1.0, 2.0)
    assert coordinate.canonical_support == (-1.0, 1.0)


def test_fixed_coordinate_needs_no_data():
    coord = _build_fit_coordinate((-INF, INF), NORMAL)
    model = _build_basis(BasisSpec(lower_tail=SoftPower(1.5)), None, None, coord)
    assert model.geometry.coordinate is coord


def test_missing_data_rejected():
    with pytest.raises(ValueError, match="requires data or an explicit coordinate"):
        _build_basis(BasisSpec())


def test_tied_landmarks_do_not_identify_a_coordinate():
    data = np.array([3.0] * 18 + [0.0, 6.0])
    with pytest.raises(ValueError, match="not identified"):
        _build_basis(BasisSpec(), data)


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
    # Tail functions stay out of the bulk: every gate of one lies in the span
    # of W_L and W_U (a + b + c = 0), in every column.
    for fn in model.functions:
        if not fn.tail_sides:
            continue
        for c in range(d):
            assert sum(v * exact[slot][c] for slot, v in _tail_row(fn).items()) == 0
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
    assert _gate_of(model, "z^3") == [(0, 1, 1)]  # z^3 (tL + tU)
    # Lower-order completion: z^k (tL + tU) for k < 3, ungated z^2 kept.
    assert sorted(_gate_of(model, "z^2")) == [(0, 1, 1), (1, 0, 0)]
    assert sorted(_gate_of(model, "z")) == [(0, 1, 1), (1, 0, 0)]
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
    # Degree 1 is fine once a faster tail is requested or on a half-line.
    _build_basis(
        BasisSpec(bulk_degree=1, lower_tail=Polynomial(2), upper_tail=Polynomial(2)),
        NORMAL,
    )
    _build_basis(BasisSpec(support=(0.0, INF), bulk_degree=1), HALF)


def test_bulk_monomial_above_a_tail_is_gated_toward_the_bulk():
    # z^3 is a bulk monomial above the lower Polynomial(2) target.
    model = _build_basis(
        BasisSpec(bulk_degree=3, lower_tail=Polynomial(2), upper_tail=Polynomial(3)),
        NORMAL,
    )
    assert _gate_of(model, "z^3") == [(1, 1, 0)]  # a = b, c = 0: z^3 (1 + tL)


@pytest.mark.parametrize("expansion", ["minimal", "full"])
def test_tail_monomial_above_the_bulk_is_gated_toward_its_end(expansion):
    # z^4 is requested only by the upper end: z^4 (-1 + tU) = -2 z^4 W_U in
    # every span; its completion supplies the subleading z^k W_U.
    model = _build_basis(
        BasisSpec(
            bulk_degree=2,
            lower_tail=Polynomial(2),
            upper_tail=Polynomial(4),
            expansion=expansion,
        ),
        NORMAL,
    )
    assert _gate_of(model, "z^4") == [(-1, 0, 1)]
    assert (-1, 0, 1) in _gate_of(model, "z^3")


@pytest.mark.parametrize("expansion", ["minimal", "full"])
def test_soft_power_and_exponential_are_gated_toward_their_end(expansion):
    model = _build_basis(
        BasisSpec(
            lower_tail=SoftPower(2.5), upper_tail=Exponential(1.0), expansion=expansion
        ),
        NORMAL,
    )
    assert _gate_of(model, "powL") == [(-1, 1, 0)]  # -2 A^p W_L
    assert _gate_of(model, "expU") == [(-1, 0, 1)]  # -2 E_U W_U
    if expansion == "minimal":
        # Lower-order completion of the soft power: z^k W_L for k < 3; the
        # gauge drops the constant of W_L.
        assert (-1, 1, 0) in _gate_of(model, "z^2")
        assert (-1, 1, 0) in _gate_of(model, "z")
        assert (0, 1, 0) in _gate_of(model, "1")


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
            g[0] == 0 for g in _gate_of(model, "z" if k == 1 else f"z^{k}")
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
    # Each exponential is centered at its own gate: exp(-lambda) on both ends.
    lam = 0.8 * model.geometry.scale
    assert lower.leading.log_factor == upper.leading.log_factor == -lam
    assert lower.leading.row_exact == upper.leading.row_exact


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


def test_unequal_soft_powers_stay_separate_one_sided_functions():
    model = _build_basis(
        BasisSpec(lower_tail=SoftPower(1.5), upper_tail=SoftPower(2.5)), NORMAL
    )
    assert not model.merged_soft_power
    funcs = {f.label: f for f in model.functions}
    assert funcs["powL"].tail_sides == ("lower",)
    assert funcs["powU"].tail_sides == ("upper",)
    lower, upper = model.asymptotics
    assert _rank([list(lower.leading.row_exact), list(upper.leading.row_exact)]) == 2


def test_mixed_tail_classes_suppress_dominating_terms_exactly():
    # Lower soft power 1.5, upper polynomial 3: z^2 and z^3 dominate the lower end.
    model = _build_basis(
        BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.5), upper_tail=Polynomial(3)),
        NORMAL,
    )
    lower, upper = model.asymptotics
    labels = {model.functions[j].label for j in lower.suppressed}
    assert labels == {"z^2", "z^3"}
    assert upper.suppressed == ()
    assert lower.target_order == (1, 1.5, 0) and upper.target_order == (1, 3.0, 0)
    # z^2 must still lead nothing at the upper end beyond its own order.
    assert [t.order for t in upper.terms][:2] == [(1, 3.0, 0), (1, 2.0, 0)]


def test_linear_tail_with_unspecified_opposite_end_is_outward_feasible():
    # Leading affine growth must be outward on both ends.
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
    assert lower.target_order == (2, 1.0 * model.geometry.scale, 0)
    assert upper.target_order == (1, 3.0, 0)
    # The upper bulk leader z^3 is subordinate (not suppressed) at the lower end.
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
    # choice; both must return exactly what a fresh computation does.
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


# ---------------------------------------------------------------------------
# Nested lifts and faces
# ---------------------------------------------------------------------------


GROWTH_SPECS = [
    BasisSpec(bulk_degree=3),
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.5), upper_tail=Exponential(0.7)),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, upper_tail=Polynomial(5)),
    BasisSpec(support=(0.0, INF), bulk_degree=3, upper_tail=SoftPower(2.5)),
    BasisSpec(support=(-INF, 0.0), bulk_degree=2, lower_tail=Exponential(1.3)),
]


def _contains(big, small):
    """Whether the column span of ``small`` lies in that of ``big`` (exact)."""
    mb, ms = _exact_matrix(big), _exact_matrix(small)
    return _rank([rb + rs for rb, rs in zip(mb, ms)]) == big.n_coefficients


@pytest.mark.parametrize("spec", GROWTH_SPECS, ids=lambda s: "")
def test_growth_is_a_nested_chain_from_the_minimal_to_the_full_span(spec):
    from gibbus._model.basis import _full_basis, _grow_basis

    model = _compile_quiet(spec)
    full = _full_basis(model)
    while True:
        candidates = _grow_basis(model)
        for _, candidate in candidates:
            # Each extension strictly enlarges the span and contains it exactly.
            assert candidate.n_coefficients > model.n_coefficients
            assert _contains(candidate, model)
            assert _contains(full, candidate)
        if not candidates:
            break
        model = candidates[0][1]
    assert model.n_coefficients == full.n_coefficients
    assert _contains(model, full)


@pytest.mark.parametrize("spec", GROWTH_SPECS, ids=lambda s: "")
@pytest.mark.parametrize("expansion", ["minimal", "full"])
def test_leader_faces_are_exact_restrictions(spec, expansion):
    from gibbus._model.basis import _face_model

    model = _compile_quiet(BasisSpec(**{**spec.__dict__, "expansion": expansion}))
    for entry in model.asymptotics:
        try:
            face, free = _face_model(model, (entry.side,))
        except ValueError as error:
            assert "no outward growth" in str(error)
            continue
        # The face's columns lie in the parent span, and the face's own leading
        # functional at the suppressed end is a strictly lower order.
        assert _contains(model, face)
        assert face.n_coefficients == model.n_coefficients - 1
        assert len(free) == face.n_coefficients
        lowered = face.side_asymptotics(entry.side)
        assert lowered.leading.order < entry.leading.order
