"""Definition-level tests for natural full-curvature separation algebra."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
from numpy.polynomial.polynomial import polyval

from gibbus._fit.separation import (
    _as_exact_fraction,
    _curvature_boundary_values,
    _curvature_polynomials,
    _exact_float_polynomial,
    _fraction_bernstein_interval_bounds,
    _fraction_to_float_lower,
    _fraction_to_float_upper,
    _separate_full_curvature,
    _stationary_root_brackets_exact,
    _trim_exact,
)
from gibbus._model.natural import _natural_layout

_REAL_LINE = "real_line"
_HALF_LINE = "half_line"
_BOUNDED = "bounded"


def _bernstein_interval_bounds(coefficients, lower, upper, max_subdivide=0, /):
    """Certified binary64 Bernstein bounds via the exact rational kernel."""
    exact_lower, exact_upper = _fraction_bernstein_interval_bounds(
        _exact_float_polynomial(coefficients),
        _as_exact_fraction(float(lower)),
        _as_exact_fraction(float(upper)),
        max_subdivide,
    )
    return _fraction_to_float_lower(exact_lower), _fraction_to_float_upper(exact_upper)


def _stationary_root_brackets(coefficients, support, max_width, max_depth, /):
    return _stationary_root_brackets_exact(
        _exact_float_polynomial(coefficients), support, max_width, max_depth
    )


def _naive_float_bernstein(coefficients, lower, upper, /):
    """Bernstein coefficients on ``[lower, upper]`` in plain binary64."""
    source = _trim_exact(coefficients)
    affine = np.array([float(lower), float(upper) - float(lower)])
    power = np.zeros(source.size)
    basis = np.array([1.0])
    for coefficient in source:
        power[:basis.size] += coefficient * basis
        if basis.size < source.size:
            basis = np.polynomial.polynomial.polymul(basis, affine)
    n = source.size - 1
    return np.array([
        sum(power[k] * math.comb(i, k) / math.comb(n, k) for k in range(i + 1))
        for i in range(n + 1)
    ])


def _random_nonnegative_curvature(support, degree, rng, /):
    """Random degree-``degree`` polynomial nonnegative on ``support``.

    Markov--Lukacs form: ``a^2 + b^2`` on the line, ``a^2 + (z - L) b^2`` on a
    lower half-line (mirrored for an upper one), ``a^2 + (z - L)(U - z) b^2``
    or ``(z - L) a^2 + (U - z) b^2`` on an interval.
    """
    P = np.polynomial.Polynomial
    lower, upper = support

    def square(d):
        return P(rng.normal(size=d + 1)) ** 2

    if np.isneginf(lower) and np.isposinf(upper):
        result = square(degree // 2) + square(degree // 2)
    elif np.isfinite(lower) and np.isposinf(upper):
        result = square(degree // 2) + P([-lower, 1.0]) * square((degree - 1) // 2)
    elif np.isneginf(lower) and np.isfinite(upper):
        result = square(degree // 2) + P([upper, -1.0]) * square((degree - 1) // 2)
    elif degree % 2 == 0:
        result = square(degree // 2) + P([-lower, 1.0]) * P([upper, -1.0]) * square(
            degree // 2 - 1
        )
    else:
        result = P([-lower, 1.0]) * square(degree // 2) + P([upper, -1.0]) * square(
            degree // 2
        )
    coefficients = np.zeros(degree + 1)
    coefficients[: result.coef.size] = result.coef
    return coefficients


def _fraction_trim(coefficients):
    values = list(coefficients)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return values


def _fraction_derivative(coefficients):
    values = [coefficients[k] * k for k in range(1, len(coefficients))]
    return values or [Fraction(0)]


def _fraction_divrem(dividend, divisor):
    remainder = _fraction_trim(dividend)
    divisor = _fraction_trim(divisor)
    if len(divisor) == 1 and divisor[0] == 0:
        raise ZeroDivisionError
    quotient = [Fraction(0)] * max(1, len(remainder) - len(divisor) + 1)
    while len(remainder) >= len(divisor) and not (
        len(remainder) == 1 and remainder[0] == 0
    ):
        offset = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        quotient[offset] = factor
        for j, value in enumerate(divisor):
            remainder[offset + j] -= factor * value
        remainder = _fraction_trim(remainder)
    return _fraction_trim(quotient), remainder


def _fraction_sturm_sequence(coefficients):
    polynomial = _fraction_trim(
        [Fraction.from_float(float(value)) for value in coefficients]
    )
    derivative = _fraction_derivative(polynomial)
    sequence = [polynomial]
    if not (len(derivative) == 1 and derivative[0] == 0):
        sequence.append(derivative)
    while len(sequence) >= 2:
        _, remainder = _fraction_divrem(sequence[-2], sequence[-1])
        if len(remainder) == 1 and remainder[0] == 0:
            break
        sequence.append([-value for value in remainder])
    return sequence


def _fraction_polyval(coefficients, point):
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = value * point + coefficient
    return value


def _fraction_variations(sequence, point):
    signs = []
    for polynomial in sequence:
        value = _fraction_polyval(polynomial, point)
        if value > 0:
            signs.append(1)
        elif value < 0:
            signs.append(-1)
    return sum(left != right for left, right in itertools.pairwise(signs))


def _fraction_root_count(sequence, lower, upper):
    left = Fraction.from_float(float(lower))
    right = Fraction.from_float(float(upper))
    return _fraction_variations(sequence, left) - _fraction_variations(sequence, right)


def _interior_points(support, /):
    """Return deterministic interior points spanning one support."""
    lower, upper = support
    if np.isneginf(lower) and np.isposinf(upper):
        return np.linspace(-3.0, 3.0, 37)
    if np.isfinite(lower) and np.isposinf(upper):
        return lower + np.geomspace(1e-4, 12.0, 43)
    if np.isneginf(lower) and np.isfinite(upper):
        return upper - np.geomspace(1e-4, 12.0, 43)
    width = upper - lower
    return np.linspace(lower + 1e-4 * width, upper - 1e-4 * width, 47)


def _clearing_factors(points, support, lower_active, upper_active, /):
    """Return squared/cubed denominator-clearing factors on test points."""
    z = np.asarray(points, dtype=np.float64)
    lower, upper = support
    squared = np.ones_like(z)
    cubed = np.ones_like(z)
    if lower_active:
        squared *= (z - lower) ** 2
        cubed *= (z - lower) ** 3
    if upper_active:
        squared *= (upper - z) ** 2
        cubed *= (upper - z) ** 3
    return squared, cubed


@pytest.mark.parametrize(
    "support,lower_enabled,upper_enabled,amplitudes",
    [
        ((-np.inf, np.inf), False, False, (np.nan, np.nan)),
        ((-1.0, np.inf), True, False, (0.0, np.nan)),
        ((-1.0, np.inf), True, False, (0.7, np.nan)),
        ((-np.inf, 2.0), False, True, (np.nan, 0.0)),
        ((-np.inf, 2.0), False, True, (np.nan, 0.6)),
        ((-1.0, 2.0), True, True, (0.0, 0.0)),
        ((-1.0, 2.0), True, True, (0.7, 0.0)),
        ((-1.0, 2.0), True, True, (0.0, 0.6)),
        ((-1.0, 2.0), True, True, (0.7, 0.6)),
    ],
)
def test_denominator_clearing_matches_direct_full_curvature_and_derivative(
    support, lower_enabled, upper_enabled, amplitudes
):
    layout = _natural_layout(support, 8, lower_enabled, upper_enabled)
    rng = np.random.default_rng(31000 + layout.n_params)
    curvature = rng.normal(size=layout.curvature_degree + 1)
    raw = layout.pack(0.2, curvature, np.asarray(amplitudes, dtype=float))
    candidate = layout.build_candidate(raw)
    cleared = _curvature_polynomials(
        candidate.q_d2, support, candidate.boundary_amplitudes
    )
    points = _interior_points(support)
    squared, cubed = _clearing_factors(
        points, support, cleared.lower_active, cleared.upper_active
    )

    expected_sign = squared * candidate.q_d2_full(points)
    expected_stationary = cubed * candidate.q_d3_full(points)
    actual_sign = polyval(points, cleared.sign_coefficients)
    actual_stationary = polyval(points, cleared.stationary_coefficients)

    sign_scale = max(1.0, float(np.max(np.abs(expected_sign))))
    stationary_scale = max(1.0, float(np.max(np.abs(expected_stationary))))
    np.testing.assert_allclose(actual_sign, expected_sign, rtol=2e-12, atol=2e-12 * sign_scale)
    np.testing.assert_allclose(
        actual_stationary,
        expected_stationary,
        rtol=4e-12,
        atol=4e-12 * stationary_scale,
    )


def test_exact_zero_and_tiny_positive_amplitudes_use_different_polynomials():
    layout = _natural_layout((-1.0, 2.0), 6, True, True)
    curvature = np.array([0.8, -0.2, 0.1, 0.03, 0.01])

    zero = layout.build_candidate(layout.pack(0.0, curvature, np.array([0.0, 0.0])))
    tiny = layout.build_candidate(layout.pack(0.0, curvature, np.array([1e-300, 0.0])))
    zero_cleared = _curvature_polynomials(
        zero.q_d2, layout.support, zero.boundary_amplitudes
    )
    tiny_cleared = _curvature_polynomials(
        tiny.q_d2, layout.support, tiny.boundary_amplitudes
    )

    assert not zero_cleared.lower_active
    assert tiny_cleared.lower_active
    assert zero_cleared.sign_coefficients.size == curvature.size
    assert tiny_cleared.sign_coefficients.size >= curvature.size + 2
    assert zero_cleared.stationary_coefficients.size <= curvature.size - 1
    assert tiny_cleared.stationary_coefficients.size >= curvature.size + 2


@pytest.mark.parametrize(
    "support_kind,support,degree,lower_enabled,upper_enabled,amplitudes",
    [
        (_REAL_LINE, (-np.inf, np.inf), 8, False, False, (np.nan, np.nan)),
        (_HALF_LINE, (-1.5, np.inf), 7, True, False, (0.0, np.nan)),
        (_HALF_LINE, (-1.5, np.inf), 7, True, False, (0.4, np.nan)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.0, 0.0)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.4, 0.0)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.0, 0.5)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.4, 0.5)),
    ],
)
def test_feasible_states_keep_nonnegative_cleared_sign(
    support_kind, support, degree, lower_enabled, upper_enabled, amplitudes
):
    natural_layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    rng = np.random.default_rng(32000 + degree + natural_layout.n_params)
    points = _interior_points(support)

    for _ in range(24):
        curvature = _random_nonnegative_curvature(
            support, natural_layout.curvature_degree, rng
        )
        raw = natural_layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
        candidate = natural_layout.build_candidate(raw)
        cleared = _curvature_polynomials(
            candidate.q_d2, support, candidate.boundary_amplitudes
        )
        direct = candidate.q_d2_full(points)
        cleared_values = polyval(points, cleared.sign_coefficients)
        scale = max(1.0, float(np.max(np.abs(cleared_values))))

        assert float(np.min(direct)) >= -2e-11 * max(1.0, float(np.max(np.abs(direct))))
        assert float(np.min(cleared_values)) >= -2e-11 * scale


@pytest.mark.parametrize(
    "support,lower_enabled,upper_enabled,amplitudes",
    [
        ((-np.inf, np.inf), False, False, (np.nan, np.nan)),
        ((0.0, np.inf), True, False, (0.8, np.nan)),
        ((-np.inf, 1.0), False, True, (np.nan, 0.7)),
        ((-1.0, 2.0), True, True, (0.5, 0.9)),
    ],
)
def test_real_stationary_polynomial_roots_are_full_curvature_stationary_points(
    support, lower_enabled, upper_enabled, amplitudes
):
    layout = _natural_layout(support, 6, lower_enabled, upper_enabled)
    curvature = np.array([0.7, -0.8, -0.1, 0.3, 0.08])
    candidate = layout.build_candidate(
        layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
    )
    cleared = _curvature_polynomials(
        candidate.q_d2, support, candidate.boundary_amplitudes
    )
    roots = np.polynomial.Polynomial(cleared.stationary_coefficients).roots()
    lower, upper = support
    real_roots = [
        float(root.real)
        for root in roots
        if abs(root.imag) <= 2e-8 * max(1.0, abs(root.real))
        and root.real > lower
        and root.real < upper
    ]

    for root in real_roots:
        value = float(candidate.q_d3_full(root))
        local_scale = max(
            1.0,
            abs(float(polyval(root, candidate.q_d2))),
            abs(root) ** max(1, candidate.q_d2.size - 1),
        )
        assert abs(value) <= 2e-7 * local_scale


def test_separation_algebra_rejects_invalid_boundary_amplitudes():
    with pytest.raises(ValueError, match="boundary amplitudes"):
        _curvature_polynomials(np.array([1.0]), (0.0, np.inf), np.array([-1e-12, np.nan]))
    with pytest.raises(ValueError, match="finite endpoint"):
        _curvature_polynomials(np.array([1.0]), (-np.inf, np.inf), np.array([0.0, np.nan]))


@pytest.mark.parametrize(
    "q_d2,support,amplitudes,expected_lower,expected_upper",
    [
        (np.array([2.0]), (-np.inf, np.inf), (np.nan, np.nan), 2.0, 2.0),
        (np.array([0.0, 1.0]), (-np.inf, np.inf), (np.nan, np.nan), -np.inf, np.inf),
        (np.array([0.0, -1.0]), (-np.inf, np.inf), (np.nan, np.nan), np.inf, -np.inf),
        (np.array([0.0, 0.0, 1.0]), (-np.inf, np.inf), (np.nan, np.nan), np.inf, np.inf),
        (np.array([0.0, 0.0, -1.0]), (-np.inf, np.inf), (np.nan, np.nan), -np.inf, -np.inf),
        (np.array([0.0, 0.0, 0.0]), (-np.inf, np.inf), (np.nan, np.nan), 0.0, 0.0),
    ],
)
def test_infinite_tail_limits_follow_exact_effective_degree(
    q_d2, support, amplitudes, expected_lower, expected_upper
):
    values = _curvature_boundary_values(q_d2, support, np.asarray(amplitudes, dtype=float))
    assert values.lower_tail == expected_lower
    assert values.upper_tail == expected_upper
    assert values.lower_endpoint is None
    assert values.upper_endpoint is None


def test_half_line_tail_and_regular_or_singular_endpoint_are_distinct():
    q_d2 = np.array([-0.3, 0.2, 0.04])
    regular = _curvature_boundary_values(q_d2, (1.0, np.inf), np.array([0.0, np.nan]))
    singular = _curvature_boundary_values(q_d2, (1.0, np.inf), np.array([1e-300, np.nan]))

    assert regular.lower_endpoint == pytest.approx(float(polyval(1.0, q_d2)))
    assert singular.lower_endpoint == np.inf
    assert regular.upper_tail == np.inf
    assert singular.upper_tail == np.inf


def test_upper_half_line_tail_uses_parity_at_negative_infinity():
    even_degree_positive = _curvature_boundary_values(
        np.array([0.0, 0.0, 0.5]), (-np.inf, 2.0), np.array([np.nan, 0.0])
    )
    odd_degree_positive = _curvature_boundary_values(
        np.array([0.0, 0.0, 0.5, 0.2]), (-np.inf, 2.0), np.array([np.nan, 0.0])
    )
    assert even_degree_positive.lower_tail == np.inf
    assert odd_degree_positive.lower_tail == -np.inf


def test_bounded_mixed_amplitude_endpoint_values_include_opposite_boundary_term():
    support = (-1.0, 3.0)
    q_d2 = np.array([0.4, -0.2, 0.1])
    values = _curvature_boundary_values(q_d2, support, np.array([0.0, 0.8]))
    expected_lower = float(polyval(support[0], q_d2)) + 0.8 / (support[1] - support[0]) ** 2
    assert values.lower_endpoint == pytest.approx(expected_lower)
    assert values.upper_endpoint == np.inf
    assert values.lower_tail is None
    assert values.upper_tail is None


@pytest.mark.parametrize(
    "coefficients,interval",
    [
        (np.array([1.0]), (-3.0, 4.0)),
        (np.array([-0.4, 1.2]), (-2.0, 3.0)),
        (np.array([0.7, -1.1, 0.3]), (-1.5, 2.2)),
        (np.array([0.2, 0.8, -0.6, 0.1, 0.05]), (-3.0, -0.2)),
        (np.array([-0.1, 0.3, 0.5, -0.2, 0.04, 0.01]), (0.1, 4.0)),
    ],
)
def test_bernstein_interval_bounds_enclose_dense_polynomial_values(coefficients, interval):
    lower, upper = interval
    grid = np.linspace(lower, upper, 4001)
    values = polyval(grid, coefficients)
    bound_lower, bound_upper = _bernstein_interval_bounds(
        coefficients, lower, upper, 3
    )
    scale = max(1.0, float(np.max(np.abs(values))))
    assert bound_lower <= float(np.min(values)) + 2e-13 * scale
    assert bound_upper >= float(np.max(values)) - 2e-13 * scale


def test_bernstein_subdivision_monotonically_tightens_enclosure():
    coefficients = np.array([0.2, -1.7, 0.4, 1.2, -0.3, 0.04])
    bounds = [
        _bernstein_interval_bounds(coefficients, -1.3, 2.1, depth)
        for depth in range(6)
    ]
    for earlier, later in itertools.pairwise(bounds):
        assert later[0] >= earlier[0] - 2e-14
        assert later[1] <= earlier[1] + 2e-14


def test_certified_bernstein_bounds_prevent_false_negative_cut_from_cancellation():
    center = 33554432.75
    coefficients = np.array([1125899957174272.5, -67108865.5, 1.0])
    lower = center - 0.25
    upper = center + 0.25

    naive = _naive_float_bernstein(coefficients, lower, upper)
    assert float(np.max(naive)) < 0.0

    certified_lower, certified_upper = _bernstein_interval_bounds(
        coefficients, lower, upper, 0
    )
    assert certified_lower == -0.125
    assert certified_upper == 0.0


def test_certified_bernstein_bounds_prevent_false_positive_prune_from_cancellation():
    center = 33554432.75
    coefficients = np.array([1125899957174273.5, -67108865.5, 1.0])
    lower = center - 1.0
    upper = center + 1.0

    naive = _naive_float_bernstein(coefficients, lower, upper)
    assert float(np.min(naive)) >= 0.0

    certified_lower, certified_upper = _bernstein_interval_bounds(
        coefficients, lower, upper, 0
    )
    assert certified_lower == -0.0625
    assert certified_upper == 1.9375


def test_certified_bernstein_outward_conversion_handles_binary64_overflow():
    maximum = np.finfo(np.float64).max
    positive = np.array([maximum, maximum])
    negative = -positive

    positive_bounds = _bernstein_interval_bounds(positive, 1.0, 2.0, 0)
    negative_bounds = _bernstein_interval_bounds(negative, 1.0, 2.0, 0)

    assert positive_bounds[0] == maximum
    assert positive_bounds[1] == np.inf
    assert negative_bounds[0] == -np.inf
    assert negative_bounds[1] == -maximum


def test_bernstein_bounds_prune_strictly_positive_and_negative_intervals():
    positive = np.array([0.14, -0.4, 1.0])  # (z - 0.2)^2 + 0.1
    negative = -positive
    positive_bounds = _bernstein_interval_bounds(positive, -1.0, 1.5, 4)
    negative_bounds = _bernstein_interval_bounds(negative, -1.0, 1.5, 4)
    assert positive_bounds[0] > 0.0
    assert negative_bounds[1] < 0.0


def test_bernstein_root_isolation_finds_multiple_simple_roots():
    roots = np.array([-2.5, -0.7, 0.4, 1.8])
    coefficients = np.polynomial.polynomial.polyfromroots(roots)
    isolated = _stationary_root_brackets(
        coefficients, (-np.inf, np.inf), 1e-8, 80
    )
    assert not isolated.identically_zero
    for root in roots:
        assert any(bracket.lower <= root <= bracket.upper for bracket in isolated.brackets)
    assert sum(bracket.sign_variations == 1 for bracket in isolated.brackets) >= roots.size


def test_bernstein_root_isolation_retains_repeated_root_conservatively():
    roots = np.array([-1.2, 0.137, 0.137, 1.6])
    coefficients = np.polynomial.polynomial.polyfromroots(roots)
    isolated = _stationary_root_brackets(coefficients, (-2.0, 2.0), 2e-7, 80)
    assert any(bracket.lower <= 0.137 <= bracket.upper for bracket in isolated.brackets)
    repeated = [
        bracket for bracket in isolated.brackets
        if bracket.lower <= 0.137 <= bracket.upper
    ]
    assert min(bracket.upper - bracket.lower for bracket in repeated) <= 2.1e-7


def test_bernstein_root_isolation_prunes_polynomial_with_no_real_roots():
    isolated = _stationary_root_brackets(np.array([1.0, 0.0, 1.0]), (-np.inf, np.inf), 1e-8, 80)
    assert isolated.brackets == ()
    assert not isolated.identically_zero


def test_bernstein_root_isolation_marks_identically_zero_polynomial():
    isolated = _stationary_root_brackets(np.array([0.0, 0.0]), (-1.0, 1.0), 1e-8, 20)
    assert isolated.brackets == ()
    assert isolated.identically_zero


def test_bernstein_root_isolation_respects_support_restriction():
    coefficients = np.polynomial.polynomial.polyfromroots([-4.0, -0.5, 1.0, 5.0])
    isolated = _stationary_root_brackets(coefficients, (0.0, 3.0), 1e-8, 80)
    assert isolated.brackets
    assert all(0.0 <= bracket.lower <= bracket.upper <= 3.0 for bracket in isolated.brackets)
    assert any(bracket.lower <= 1.0 <= bracket.upper for bracket in isolated.brackets)
    assert not any(bracket.lower <= -0.5 <= bracket.upper for bracket in isolated.brackets)


@pytest.mark.parametrize("degree", [3, 5, 7, 9, 11, 13])
def test_bernstein_root_isolation_contains_random_companion_real_roots(degree):
    rng = np.random.default_rng(41000 + degree)
    for _ in range(30):
        coefficients = rng.normal(size=degree + 1)
        isolated = _stationary_root_brackets(
            coefficients, (-3.0, 2.0), 2e-8, 90
        )
        roots = np.polynomial.Polynomial(coefficients).roots()
        real_roots = [
            float(root.real)
            for root in roots
            if abs(root.imag) <= 2e-9 * max(1.0, abs(root.real))
            and -3.0 < root.real < 2.0
        ]
        for root in real_roots:
            assert any(
                bracket.lower - 5e-8 <= root <= bracket.upper + 5e-8
                for bracket in isolated.brackets
            )


def test_bernstein_root_isolation_recovers_repeated_root_on_split_boundary():
    isolated = _stationary_root_brackets(
        np.array([0.0, 0.0, 1.0]), (-1.0, 1.0), 1e-10, 40
    )
    assert any(bracket.lower == 0.0 and bracket.upper == 0.0 for bracket in isolated.brackets)


@pytest.mark.parametrize("degree", [3, 5, 7, 9, 11, 13])
def test_bernstein_root_isolation_matches_exact_fraction_sturm_counts(degree):
    rng = np.random.default_rng(43000 + degree)
    for _ in range(12):
        coefficients = rng.normal(size=degree + 1)
        coefficients *= 2.0 ** int(rng.integers(-12, 13))
        isolated = _stationary_root_brackets(
            coefficients, (-3.0, 2.0), 5e-9, 100
        )
        sturm = _fraction_sturm_sequence(coefficients)
        exact_total = _fraction_root_count(sturm, -3.0, 2.0)
        bracket_total = sum(
            _fraction_root_count(sturm, bracket.lower, bracket.upper)
            for bracket in isolated.brackets
            if bracket.lower < bracket.upper
        )
        assert bracket_total == exact_total
        for bracket in isolated.brackets:
            if bracket.sign_variations == 1 and bracket.lower < bracket.upper:
                assert _fraction_root_count(sturm, bracket.lower, bracket.upper) == 1


def _run_separator(candidate, tolerance=1e-10):
    return _separate_full_curvature(
        candidate.q_d2,
        candidate.layout.support,
        candidate.boundary_amplitudes,
        tolerance,
        2e-8,
        90,
        2,
    )


def test_separator_certifies_real_line_double_root_minima_with_tolerance():
    layout = _natural_layout((-np.inf, np.inf), 6, False, False)
    curvature = np.polynomial.polynomial.polymul(
        np.array([1.0, 0.0, -2.0, 0.0, 1.0]), np.array([1.0])
    )
    candidate = layout.build_candidate(layout.pack(0.0, curvature, np.array([np.nan, np.nan])))
    result = _run_separator(candidate, 1e-9)
    assert result.status == "feasible"
    assert result.isolated_roots >= 3
    assert result.pruned_intervals + result.discarded_maxima >= 3


def test_separator_finds_negative_stationary_minimum():
    layout = _natural_layout((-np.inf, np.inf), 6, False, False)
    curvature = np.array([0.999, 0.0, -2.0, 0.0, 1.0])
    candidate = layout.build_candidate(layout.pack(0.0, curvature, np.array([np.nan, np.nan])))
    result = _run_separator(candidate, 1e-10)
    assert result.status == "violated"
    assert result.violation_kind == "stationary"
    assert result.violation_location is not None
    assert result.violation_value < -1e-10


def test_separator_finds_regular_endpoint_violation():
    layout = _natural_layout((0.0, 3.0), 4, True, False)
    curvature = np.array([-0.2, 0.5, 0.2])
    candidate = layout.build_candidate(layout.pack(0.0, curvature, np.array([0.0, np.nan])))
    result = _run_separator(candidate)
    assert result.status == "violated"
    assert result.violation_kind == "lower_endpoint"
    assert result.violation_location == 0.0


def test_separator_tiny_positive_amplitude_removes_endpoint_candidate():
    layout = _natural_layout((0.0, 3.0), 4, True, False)
    curvature = np.array([-0.2, 0.5, 0.2])
    candidate = layout.build_candidate(layout.pack(0.0, curvature, np.array([1e-8, np.nan])))
    result = _run_separator(candidate)
    assert result.violation_kind != "lower_endpoint"


def test_separator_finds_unbounded_tail_violation():
    layout = _natural_layout((0.0, np.inf), 3, False, False)
    candidate = layout.build_candidate(
        layout.pack(0.0, np.array([1.0, -0.1]), np.array([np.nan, np.nan]))
    )
    result = _run_separator(candidate)
    assert result.status == "violated"
    assert result.violation_kind == "upper_tail"
    assert result.violation_location is None


@pytest.mark.parametrize(
    "support_kind,support,degree,lower_enabled,upper_enabled,amplitudes",
    [
        (_REAL_LINE, (-np.inf, np.inf), 8, False, False, (np.nan, np.nan)),
        (_HALF_LINE, (-1.0, np.inf), 7, True, False, (0.0, np.nan)),
        (_HALF_LINE, (-1.0, np.inf), 7, True, False, (0.2, np.nan)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.0, 0.0)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.2, 0.0)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.0, 0.3)),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.2, 0.3)),
    ],
)
def test_separator_certifies_random_feasible_states(
    support_kind, support, degree, lower_enabled, upper_enabled, amplitudes
):
    natural_layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    rng = np.random.default_rng(45000 + degree + natural_layout.n_params)
    for _ in range(20):
        curvature = _random_nonnegative_curvature(
            support, natural_layout.curvature_degree, rng
        )
        candidate = natural_layout.build_candidate(
            natural_layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
        )
        result = _run_separator(candidate, 2e-9)
        assert result.status == "feasible"


@pytest.mark.parametrize(
    "support_kind,support,degree,lower_enabled,upper_enabled,amplitudes,point",
    [
        (_REAL_LINE, (-np.inf, np.inf), 8, False, False, (np.nan, np.nan), 0.25),
        (_HALF_LINE, (-1.0, np.inf), 7, True, False, (0.2, np.nan), 0.5),
        (_BOUNDED, (-2.0, 3.0), 7, True, True, (0.2, 0.3), 0.4),
    ],
)
def test_separator_detects_constant_shift_forced_interior_violations(
    support_kind, support, degree, lower_enabled, upper_enabled, amplitudes, point
):
    natural_layout = _natural_layout(support, degree, lower_enabled, upper_enabled)
    rng = np.random.default_rng(46000 + degree)
    for _ in range(15):
        curvature = _random_nonnegative_curvature(
            support, natural_layout.curvature_degree, rng
        )
        candidate = natural_layout.build_candidate(
            natural_layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
        )
        shift = float(candidate.q_d2_full(point)) + 0.05
        shifted = curvature.copy()
        shifted[0] -= shift
        bad = natural_layout.build_candidate(
            natural_layout.pack(0.0, shifted, np.asarray(amplitudes, dtype=float))
        )
        assert bad.q_d2_full(point) < -0.049999
        result = _run_separator(bad, 1e-9)
        assert result.status == "violated"


@pytest.mark.parametrize(
    "amplitudes",
    [
        (0.0, 0.0),
        (0.2, 0.0),
        (0.0, 0.3),
        (0.2, 0.3),
        (1e-300, 0.3),
        (0.2, 1e-300),
    ],
)
def test_separator_bounded_presence_cases_find_known_interior_violation(amplitudes):
    support = (0.0, 2.0)
    a_lower, a_upper = amplitudes
    # Make z=1 an exact stationary point of the full curvature.  The ordinary
    # polynomial derivative cancels the reciprocal-square derivatives there.
    curvature = np.array([
        1.0 - 2.0 * (a_lower - a_upper) - (a_lower + a_upper) - 0.05,
        -2.0 + 2.0 * (a_lower - a_upper),
        1.0,
    ])
    layout = _natural_layout(support, 4, True, True)
    candidate = layout.build_candidate(
        layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
    )
    assert candidate.q_d3_full(1.0) == pytest.approx(0.0, abs=2e-14)
    assert candidate.q_d2_full(1.0) == pytest.approx(-0.05, abs=2e-14)
    result = _run_separator(candidate, 1e-10)
    assert result.status == "violated"
    assert result.violation_kind == "stationary"
    assert result.violation_value < -1e-10


@pytest.mark.parametrize(
    "support,amplitudes,stationary_point",
    [
        ((0.0, np.inf), (0.25, np.nan), 1.0),
        ((-np.inf, 2.0), (np.nan, 0.25), 1.0),
    ],
)
def test_separator_half_line_positive_amplitude_known_stationary_violation(
    support, amplitudes, stationary_point
):
    lower, _upper = support
    a_lower, a_upper = amplitudes
    amplitude = a_lower if np.isfinite(a_lower) else a_upper
    if np.isfinite(lower):
        # At z=1, p'(1)=2a cancels -2a/(z-L)^3.
        curvature = np.array([0.95 - 3.0 * amplitude,
                              -2.0 + 2.0 * amplitude, 1.0])
    else:
        # Mirror around U=2; at z=1, p'(1)=-2a cancels +2a/(U-z)^3.
        curvature = np.array([-0.05 - amplitude + 1.0 + 2.0 * amplitude,
                              -2.0 - 2.0 * amplitude, 1.0])
    layout = _natural_layout(
        support,
        4,
        bool(np.isfinite(a_lower)),
        bool(np.isfinite(a_upper)),
    )
    candidate = layout.build_candidate(
        layout.pack(0.0, curvature, np.asarray(amplitudes, dtype=float))
    )
    assert candidate.q_d3_full(stationary_point) == pytest.approx(0.0, abs=2e-14)
    assert candidate.q_d2_full(stationary_point) == pytest.approx(-0.05, abs=2e-14)
    result = _run_separator(candidate, 1e-10)
    assert result.status == "violated"
    assert result.violation_kind == "stationary"


def test_separator_repeated_stationary_root_is_handled_conservatively():
    layout = _natural_layout((-np.inf, np.inf), 6, False, False)
    candidate = layout.build_candidate(
        layout.pack(0.0, np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
                    np.array([np.nan, np.nan]))
    )
    result = _run_separator(candidate, 1e-12)
    assert result.status == "feasible"
    assert result.isolated_roots == 1
    assert result.ambiguous_intervals == 0


@pytest.mark.parametrize("leading", [1e-280, -1e-280])
def test_separator_effective_degree_uses_exact_nonzero_leading_term(leading):
    layout = _natural_layout((-np.inf, np.inf), 4, False, False)
    candidate = layout.build_candidate(
        layout.pack(0.0, np.array([1.0, 0.0, leading]),
                    np.array([np.nan, np.nan]))
    )
    result = _run_separator(candidate, 1e-12)
    if leading > 0.0:
        assert result.status == "feasible"
    else:
        assert result.status == "violated"
        assert result.violation_kind in {"lower_tail", "upper_tail"}


def test_separator_bounded_extreme_support_scale_constant_curvature():
    support = (-1e6, 1e6)
    layout = _natural_layout(support, 4, True, True)
    candidate = layout.build_candidate(
        layout.pack(0.0, np.array([0.75, 0.0, 0.0]),
                    np.array([1e8, 1e8]))
    )
    result = _run_separator(candidate, 1e-10)
    assert result.status == "feasible"


def test_separator_random_bounded_states_agree_with_dense_reference():
    rng = np.random.default_rng(47001)
    support = (-2.0, 3.0)
    grid = np.linspace(support[0], support[1], 20001)
    for amplitudes in (
        np.array([0.0, 0.0]),
        np.array([0.2, 0.0]),
        np.array([0.0, 0.3]),
        np.array([0.2, 0.3]),
        np.array([1e-12, 0.3]),
    ):
        layout = _natural_layout(support, 8, True, True)
        for _ in range(20):
            curvature = rng.normal(scale=0.4, size=7)
            candidate = layout.build_candidate(
                layout.pack(0.0, curvature, amplitudes)
            )
            values = np.asarray(candidate.q_d2_full(grid), dtype=float)
            reference_minimum = float(np.nanmin(values))
            result = _run_separator(candidate, 1e-10)
            if reference_minimum < -1e-4:
                assert result.status == "violated"
            if result.status == "feasible":
                assert reference_minimum >= -1e-7
            if result.status == "violated" and result.violation_location is not None:
                assert candidate.q_d2_full(result.violation_location) < -1e-10


def test_separator_certifies_mirrored_upper_half_line_states():
    source_support = (0.0, np.inf)
    target_support = (-np.inf, 2.0)
    natural_layout = _natural_layout(target_support, 7, False, True)
    rng = np.random.default_rng(49007)
    affine = np.polynomial.Polynomial([2.0, -1.0])
    for _ in range(30):
        source_curvature = _random_nonnegative_curvature(
            source_support, natural_layout.curvature_degree, rng
        )
        mirrored_curvature = np.polynomial.Polynomial(source_curvature)(affine).coef
        candidate = natural_layout.build_candidate(
            natural_layout.pack(
                0.0,
                mirrored_curvature,
                np.array([np.nan, 0.2]),
            )
        )
        result = _run_separator(candidate, 2e-9)
        assert result.status == "feasible"


def test_separator_refines_unique_root_bracket_that_shares_exact_endpoint_root():
    # S(z) has roots at approximately +/-0.953 and exactly at zero.  Initial
    # Bernstein isolation returns the negative root in a broad interval whose
    # upper endpoint is the separate exact zero root.  The separator must
    # refine the unique interior root instead of treating the zero endpoint as
    # an ambiguous sign-bisection boundary.
    q_d2 = np.array([
        0.876612938,
        0.0,
        -1.93056952,
        -4.13726538e-18,
        1.06292137,
    ])
    result = _separate_full_curvature(
        q_d2,
        (-np.inf, np.inf),
        np.array([np.nan, np.nan]),
        1e-12,
        1e-10,
        80,
        2,
    )
    assert result.status == "violated"
    assert result.violation_kind == "stationary"
    assert result.violation_location == pytest.approx(-0.95296545, abs=2e-7)
    assert result.violation_value < -1e-7
