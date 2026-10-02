"""Tests for the natural-coordinate conic Newton solver."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from gibbus._fit.conic_newton import _amplitude_release_gain, _solve_natural_conic
from gibbus._fit.conic_qp import _support_representation
from gibbus._fit.natural_objective import (
    _fit_natural_conic_points,
    _prepare_natural_point_objective,
)
from gibbus._model.natural import _natural_layout

_DATA = Path(__file__).parent / "data"
_REAL_LINE = (-np.inf, np.inf)

# Certified optimal NLLs of the cases below.  Each was cross-checked, when
# recorded, against an independent local fitter that could only match or sit
# above it; a fit must now reproduce it within the certified bounds.
_BASELINE_NLL = {
    "small_2": 1.8534070509500151,
    "small_4": 1.8922241355039233,
    "small_6": 1.9779854769646048,
    "natural_degree8_seed270904_nll_gap_1e3.npz": 1.6442830096522811,
    "natural_degree8_seed270909_nll_gap_5e5.npz": 1.6783932548868694,
    "natural_degree8_seed270914_nll_gap_1e4.npz": 1.7621020831590126,
    "natural_degree8_seed270901_latency_pathology.npz": 1.6037446221427547,
    "natural_degree8_seed270929_exchange_limit.npz": 1.6539743717365971,
    "weighted_exchange": 1.366685990797411,
    "degree10": 1.4686043875955574,
    "seeded_2_normal": 1.667449861396986,
    "seeded_2_weighted_gumbel": 1.5227947792305754,
    "seeded_4_normal": 1.785546920901547,
    "seeded_4_weighted_gumbel": 1.4381303068444433,
    "seeded_6_normal": 1.7333885061430157,
    "seeded_6_weighted_gumbel": 1.4869021995889817,
    "geom_bounded_lower_amplitude_11_3": -0.5414302275432681,
    "geom_bounded_lower_amplitude_11_4": -0.543989815093189,
    "geom_bounded_lower_amplitude_11_6": -0.5443424168488928,
    "geom_bounded_lower_amplitude_1_4": -0.7727064495889071,
    "geom_bounded_plain_11_3": -0.276463749393576,
    "geom_bounded_plain_11_4": -0.28540443551244166,
    "geom_bounded_plain_11_6": -0.28671536822592336,
    "geom_bounded_two_amplitudes_11_3": -0.29164653773088767,
    "geom_bounded_two_amplitudes_11_4": -0.2916894123022644,
    "geom_bounded_two_amplitudes_11_6": -0.3009073274629175,
    "geom_bounded_upper_amplitude_11_3": -0.27724727952100614,
    "geom_bounded_upper_amplitude_11_4": -0.27733888733023115,
    "geom_bounded_upper_amplitude_11_6": -0.29207738197929123,
    "geom_half_line_amplitude_11_3": 1.8089375874178497,
    "geom_half_line_amplitude_11_4": 1.8085921545932382,
    "geom_half_line_amplitude_11_6": 1.8018059164265536,
    "geom_half_line_amplitude_1_4": 1.7180032168933839,
    "geom_half_line_exponential_11_3": 0.9826826353106333,
    "geom_half_line_exponential_11_4": 0.980314203501115,
    "geom_half_line_exponential_11_6": 0.9768816626903309,
    "geom_half_line_exponential_3_4": 1.1395996041395287,
    "geom_half_line_plain_11_3": 1.8402564809994417,
    "geom_half_line_plain_11_4": 1.8121363737010843,
    "geom_half_line_plain_11_6": 1.808034434634822,
    "geom_reflected_half_line_11_3": 1.5619026816996557,
    "geom_reflected_half_line_11_4": 1.5601395010442247,
    "geom_reflected_half_line_11_6": 1.556445397551446,
    "geom_reflected_plain_11_3": 1.585254902345996,
    "geom_reflected_plain_11_4": 1.562467328607341,
    "geom_reflected_plain_11_6": 1.5609381546915024,
}


def _assert_baseline(result, key, /):
    """Assert a certified fit reproduces its recorded optimum."""
    assert result.final_separation.feasible
    slack = 2.0 * max(1e-10, result.final_decrease_bound)
    assert result.objective_value == pytest.approx(_BASELINE_NLL[key], abs=slack)


def _assert_certified_baseline(samples, degree, key, weights=None, /):
    objective, result = _fit_natural_conic_points(
        _REAL_LINE, samples, degree, False, False, weights
    )
    assert result.status == "converged"
    _assert_baseline(result, key)
    return objective, result


_SMALL_CASES = {
    2: np.array([-2.4, -1.3, -0.7, -0.1, 0.2, 0.9, 1.6, 2.8]),
    4: np.array([-2.5, -1.7, -0.9, -0.25, 0.1, 0.45, 0.95, 1.8, 3.2]),
    6: np.array([-2.8, -1.9, -1.1, -0.55, -0.1, 0.25, 0.6, 1.05, 1.7, 2.6, 3.8]),
}


@pytest.mark.parametrize("degree", [2, 4, 6])
def test_small_cases_reach_their_certified_optima(degree):
    _assert_certified_baseline(_SMALL_CASES[degree], degree, f"small_{degree}")


@pytest.mark.parametrize(
    "name",
    [
        # Silent suboptimal convergence of the cutting-plane solver.
        "natural_degree8_seed270904_nll_gap_1e3.npz",
        "natural_degree8_seed270909_nll_gap_5e5.npz",
        # Far-out tangent QP defect of the cutting-plane solver.
        "natural_degree8_seed270914_nll_gap_1e4.npz",
        # Latency pathology (>300 s) of the cutting-plane solver.
        "natural_degree8_seed270901_latency_pathology.npz",
        # Exchange-limit regression of the cutting-plane solver.
        "natural_degree8_seed270929_exchange_limit.npz",
    ],
)
def test_cutting_plane_campaign_fixtures_reach_their_optima(name):
    samples = np.load(_DATA / name)["samples"]
    _, result = _assert_certified_baseline(samples, 8, name)
    assert result.objective_evaluations <= 20


def test_weighted_exchange_exhaustion_fixture_reaches_its_optimum():
    fixture = np.load(_DATA / "natural_degree8_weighted_exchange_exhaustion.npz")
    _assert_certified_baseline(
        fixture["samples"], 8, "weighted_exchange", fixture["weights"]
    )


def test_fit_depends_on_data_only_through_sufficient_statistics():
    """Integer weights and duplicated observations give the same statistics,
    hence the same objective and the same fit (the EM/mixture interface)."""
    rng = np.random.default_rng(11)
    base = rng.normal(size=25)
    counts = rng.integers(1, 4, size=base.size)
    first, weighted = _fit_natural_conic_points(
        _REAL_LINE, base, 6, False, False, counts.astype(float)
    )
    second, duplicated = _fit_natural_conic_points(
        _REAL_LINE, np.repeat(base, counts), 6
    )
    np.testing.assert_allclose(
        first.observations.stats.moments,
        second.observations.stats.moments,
        rtol=1e-13,
    )
    assert weighted.objective_value == pytest.approx(
        duplicated.objective_value, abs=1e-12
    )
    # Parameters agree to the resolution an objective tolerance of 1e-12
    # implies along weakly curved Fisher directions.
    np.testing.assert_allclose(weighted.params, duplicated.params, atol=1e-5)


def test_returns_warm_startable_certificate():
    rng = np.random.default_rng(5)
    samples = rng.gamma(3.0, size=80)
    objective, cold = _fit_natural_conic_points(_REAL_LINE, samples, 8)
    warm = _solve_natural_conic(
        objective, initial=cold.params, initial_blocks=cold.blocks
    )
    assert warm.status == "converged"
    assert warm.objective_evaluations <= 3
    assert warm.objective_value == pytest.approx(cold.objective_value, abs=1e-12)


@dataclass(frozen=True)
class _Evaluation:
    nll: float
    gradient: np.ndarray
    hessian: np.ndarray


class _QuadraticObjective:
    """Convex quadratic in natural coordinates, for exact-face semantics."""

    def __init__(self, layout, target, weights):
        self.layout = layout
        self.target = np.asarray(target, dtype=float)
        self.weights = np.asarray(weights, dtype=float)

    def __call__(self, params):
        d = np.asarray(params, dtype=float) - self.target
        return _Evaluation(
            nll=float(0.5 * np.sum(self.weights * d * d)),
            gradient=self.weights * d,
            hessian=np.diag(self.weights),
        )


def _curvature_target(layout, curvature, /):
    target = np.zeros(layout.n_params)
    target[layout.gamma_index] = 0.4
    target[layout.curvature_slice] = curvature
    return target


def test_leading_coefficient_reaches_exact_zero_through_the_face_step():
    """A target with negative leading curvature is optimal on the exact
    lower-degree face; the fit returns exact zeros, not tiny positives."""
    layout = _natural_layout(_REAL_LINE, 6)  # curvature degree 4
    target = _curvature_target(layout, [1.0, 0.0, 1.0, 0.0, -1.0])
    objective = _QuadraticObjective(layout, target, np.ones(layout.n_params))
    result = _solve_natural_conic(objective)
    curvature = result.params[layout.curvature_slice]
    assert result.status == "converged"
    assert result.effective_curvature_degree == 2
    assert curvature[3] == 0.0 and curvature[4] == 0.0
    np.testing.assert_allclose(curvature[:3], [1.0, 0.0, 1.0], atol=1e-9)
    assert result.final_separation.feasible


def test_small_positive_leading_coefficient_is_not_snapped():
    """A face is taken only when it is optimal: a genuinely positive small
    leading coefficient whose removal costs likelihood is kept."""
    layout = _natural_layout(_REAL_LINE, 6)
    leading = 5e-9
    target = _curvature_target(layout, [1.0, 0.0, 1.0, 0.0, leading])
    weights = np.ones(layout.n_params)
    weights[layout.curvature_slice.start + 4] = 1e8  # removal costs 1.25e-9
    result = _solve_natural_conic(_QuadraticObjective(layout, target, weights))
    curvature = result.params[layout.curvature_slice]
    assert result.status == "converged"
    assert result.effective_curvature_degree == 4
    assert curvature[4] == pytest.approx(leading, rel=1e-4)


def test_warm_start_without_certificate_is_projected():
    rng = np.random.default_rng(5)
    samples = rng.gamma(3.0, size=80)
    objective, cold = _fit_natural_conic_points(_REAL_LINE, samples, 8)
    warm = _solve_natural_conic(objective, initial=cold.params)
    assert warm.status == "converged"
    # Both are certified within 1e-10; a restart may only polish further.
    assert (
        cold.objective_value - 1e-10
        <= warm.objective_value
        <= cold.objective_value + 1e-12
    )


def test_prepared_objective_holds_canonical_sufficient_statistics():
    rng = np.random.default_rng(2)
    samples = rng.logistic(size=60)
    objective = _prepare_natural_point_objective(
        _REAL_LINE, samples, 6, False, False, None
    )
    z = objective.spec.coordinate.to_canonical(samples)
    stats = objective.observations.stats
    np.testing.assert_allclose(
        stats.moments,
        [np.mean(z**k) for k in range(stats.moments.size)],
        rtol=1e-13,
        atol=1e-15,
    )
    assert stats.effective_n == pytest.approx(samples.size, rel=1e-14)
    assert objective.z_data_bounds == (float(z.min()), float(z.max()))
    representation = _support_representation(objective.layout)
    assert representation.n_rows == objective.layout.curvature_degree + 1


def test_degenerate_degree_ten_fit_reports_a_truthful_certificate():
    """Degree 10 on 30 weighted points has a rank-one optimal Gram matrix and
    weakly active contacts; interior-point certificates floor near 1e-9 there.
    The fit must say so honestly: its reported bound covers the true gap."""
    rng = np.random.default_rng(270901)
    samples = rng.gumbel(loc=-0.3, scale=0.9, size=30)
    weights = np.linspace(0.3, 2.0, samples.size)
    _, result = _fit_natural_conic_points(
        _REAL_LINE, samples, 10, False, False, weights
    )
    assert result.status in ("converged", "converged_approximately")
    assert result.final_separation.feasible
    assert result.final_decrease_bound <= 1e-7
    _assert_baseline(result, "degree10")


@pytest.mark.parametrize(
    "degree,kind",
    [
        (2, "normal"),
        (2, "weighted_gumbel"),
        (4, "normal"),
        (4, "weighted_gumbel"),
        (6, "normal"),
        (6, "weighted_gumbel"),
    ],
)
def test_seeded_validation_cases_reach_their_optima(degree, kind):
    seed = 270927 + 10 * int(degree) + (1 if kind == "weighted_gumbel" else 0)
    rng = np.random.default_rng(seed)
    if kind == "normal":
        samples, weights = rng.normal(loc=0.2, scale=1.3, size=120), None
    else:
        samples = rng.gumbel(loc=-0.3, scale=0.9, size=120)
        weights = np.linspace(0.3, 2.0, samples.size)
    _assert_certified_baseline(samples, degree, f"seeded_{degree}_{kind}", weights)


# --- half-line and bounded supports ----------------------------------------

_GEOMETRY_CASES = {
    "half_line_amplitude": (
        (0.0, np.inf),
        lambda rng: rng.gamma(3.0, 1.0, 120),
        True,
        False,
    ),
    "half_line_exponential": (
        (0.0, np.inf),
        lambda rng: rng.exponential(1.0, 120),
        True,
        False,
    ),
    "reflected_half_line": (
        (-np.inf, 0.0),
        lambda rng: -rng.gamma(2.0, 1.0, 120),
        False,
        True,
    ),
    "reflected_plain": (
        (-np.inf, 0.0),
        lambda rng: -rng.gamma(2.0, 1.0, 120),
        False,
        False,
    ),
    "half_line_plain": (
        (0.0, np.inf),
        lambda rng: rng.gamma(3.0, 1.0, 120),
        False,
        False,
    ),
    "bounded_upper_amplitude": (
        (0.0, 1.0),
        lambda rng: rng.beta(3.0, 2.0, 120),
        False,
        True,
    ),
    "bounded_two_amplitudes": (
        (0.0, 1.0),
        lambda rng: rng.beta(2.0, 3.0, 120),
        True,
        True,
    ),
    "bounded_plain": ((0.0, 1.0), lambda rng: rng.beta(2.0, 3.0, 120), False, False),
    "bounded_lower_amplitude": (
        (0.0, 1.0),
        lambda rng: rng.beta(0.8, 3.0, 120),
        True,
        False,
    ),
}


def _geometry_fits(name, seed, degree, /):
    support, sampler, lower, upper = _GEOMETRY_CASES[name]
    samples = sampler(np.random.default_rng(seed))
    objective = _prepare_natural_point_objective(
        support, samples, degree, lower, upper, None
    )
    return objective, _solve_natural_conic(objective)


@pytest.mark.parametrize("name", sorted(_GEOMETRY_CASES))
@pytest.mark.parametrize("degree", [3, 4, 6])
def test_geometry_fits_reach_their_certified_optima(name, degree):
    _, full = _geometry_fits(name, 11, degree)
    assert full.status in ("converged", "converged_approximately")
    _assert_baseline(full, f"geom_{name}_11_{degree}")


def test_full_cone_admits_negative_polynomial_part_where_it_pays():
    """The log-concave family is ``p + a/(z-L)^2 >= 0``, not ``p >= 0``: on
    this sample the optimum has ``p < 0`` near the endpoint and beats the
    best fit with ``p >= 0`` (NLL 1.72139736083864, recorded) by more than
    1e-3 per observation."""
    objective, full = _geometry_fits("half_line_amplitude", 1, 4)
    _assert_baseline(full, "geom_half_line_amplitude_1_4")
    assert full.objective_value < 1.7213973608386444 - 1e-3
    layout = objective.layout
    candidate = layout.build_candidate(full.params)
    z = np.linspace(layout.support[0] + 1e-6, objective.z_data_bounds[1], 2001)
    assert np.polynomial.polynomial.polyval(z, candidate.q_d2).min() < 0.0


def test_zero_amplitude_is_exact_and_kkt_optimal():
    objective, full = _geometry_fits("bounded_lower_amplitude", 1, 4)
    layout = objective.layout
    assert not full.lower_amplitude_active
    assert full.params[layout.lower_a_index] == 0.0
    free = list(range(layout.curvature_slice.start, layout.curvature_slice.stop))
    gain, residual = _amplitude_release_gain(
        layout, full.params, full.evaluation.gradient, "lower", free
    )
    assert gain >= 0.0
    assert residual < 1e-6
    _assert_baseline(full, "geom_bounded_lower_amplitude_1_4")


def test_exponential_half_line_is_not_fooled_by_the_amplitude_boundary():
    """Regression: a normalization bug made tiny positive amplitudes look
    2.7e-4 better than a = 0."""
    _, full = _geometry_fits("half_line_exponential", 3, 4)
    _assert_baseline(full, "geom_half_line_exponential_3_4")


def test_stranded_warm_start_is_resolved_from_a_centered_start():
    """Regression (geometry campaign): Beta(5, 1.5) on (0, 1), both
    amplitudes, degree 8.  The warm-started subproblem inherited a badly
    centered certificate, stalled with Gram block and dual slack singular in
    the same direction, and Newton stopped ``non_descent`` with bound 1.3e-6.
    Re-solving from the default start certifies the optimum."""
    samples = np.random.default_rng(270903).beta(5.0, 1.5, 120)
    objective = _prepare_natural_point_objective(
        (0.0, 1.0), samples, 8, True, True, None
    )
    full = _solve_natural_conic(objective)
    assert full.status in ("converged", "converged_approximately")
    assert full.final_decrease_bound <= 1e-8
    assert full.final_separation.feasible


@pytest.mark.parametrize("support", [(-1.3, np.inf), (-np.inf, 0.7)])
def test_half_line_tail_degree_face_is_exact(support):
    """On a half-line the tail needs a nonnegative leading term (sign-adjusted
    for an upper half-line); a target beyond it lands on the exact face with
    the effective degree lowered by one."""
    layout = _natural_layout(support, 5)  # curvature degree 3
    sign = 1.0 if np.isfinite(support[0]) else -1.0
    target = _curvature_target(layout, [2.0, 0.0, 0.5, -sign * 1.0])
    result = _solve_natural_conic(
        _QuadraticObjective(layout, target, np.ones(layout.n_params))
    )
    assert result.status == "converged"
    assert result.effective_curvature_degree == 2
    assert result.params[layout.curvature_slice][3] == 0.0
    assert result.final_separation.feasible
