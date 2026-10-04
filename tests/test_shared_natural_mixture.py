"""End-to-end shared physical amplitudes through EM, faces and joint polish."""

import numpy as np
import pytest

from gibbus._fit.conic_qp import _support_representation
from gibbus._fit.natural_mixture import (
    _CompiledJointMixture,
    _ComponentProblem,
    _continue_natural_mixture,
    _degree_probe_problems,
    _fit_natural_mixture,
    _run_natural_em,
)


def _sample(intervals=False):
    rng = np.random.default_rng(4)
    x = np.r_[rng.beta(2, 8, 80), rng.beta(8, 2, 120)]
    rows = (
        np.column_stack((np.floor(x * 30) / 30, np.ceil(x * 30) / 30))
        if intervals
        else x[:, None]
    )
    r = np.column_stack((x < 0.5, x >= 0.5)).astype(float)
    w = np.linspace(0.8, 1.2, len(x))
    w /= w.sum()
    return rows, r, w


def _assert_matching_fit(fit, rows, w):
    assert fit.separator_certified
    assert fit.status in ("converged", "converged_approximately")
    assert min(np.diff(fit.history), default=0.0) >= -2e-12
    for side in ("lower", "upper"):
        values = [
            c.params[getattr(c.spec, f"physical_{side}_a_index")]
            for c in fit.components
            if getattr(c.spec, f"physical_{side}_a_index") is not None
        ]
        assert not values or all(value == values[0] for value in values)
    for component in fit.components:
        result = component.solver_result
        np.testing.assert_array_equal(result.params, component.params)
        rep = _support_representation(
            component.layout,
            component.effective_curvature_degree,
            component.lower_amplitude_active,
            component.upper_amplitude_active,
        )
        assert rep.residual(result.params, result.blocks) <= 2e-7 * max(
            1.0, np.max(np.abs(rep.b_matrix @ result.params))
        )
        assert result.final_separation.feasible
        assert result.status == "joint_feasible"
        assert np.isnan(result.final_decrease_bound)
    degrees = tuple(c.spec.requested_poly_degree for c in fit.components)
    problems, layouts, params = _degree_probe_problems(fit, rows, w, degrees)
    objective = _CompiledJointMixture(problems, layouts, w)
    evaluation = objective(objective.join(params, np.log(fit.weights)))
    np.testing.assert_allclose(
        fit.observed_information, evaluation.observed_hessian, rtol=0, atol=1e-12
    )
    assert fit.n_parameters == objective.n_params
    assert fit.n_face_parameters == len(fit.free_parameter_indices)
    assert fit.log_likelihood == pytest.approx(-evaluation.nll, abs=1e-12)


@pytest.mark.parametrize("intervals", [False, True])
def test_all_shared_phases_use_coupled_solver_and_matching_certificates(
    intervals, monkeypatch
):
    rows, r, w = _sample(intervals)

    def independent_fit(*args, **kwargs):
        raise AssertionError("a shared M-step must not fit components independently")

    monkeypatch.setattr(_ComponentProblem, "fit", independent_fit)
    monkeypatch.setattr(_ComponentProblem, "fit_compact", independent_fit)
    fit = _run_natural_em(
        (0, 1), rows, 4, True, True, w, r, max_em_steps=4, max_rounds=3
    )
    _assert_matching_fit(fit, rows, w)
    assert fit.n_parameters == 11


def test_global_zero_faces_are_exact_and_released_by_joint_refits():
    rng = np.random.default_rng(35)
    x = rng.beta(0.5, 0.5, 180)
    rows = x[:, None]
    r = np.column_stack((x < np.median(x), x >= np.median(x))).astype(float)
    w = np.full(x.size, 1 / x.size)
    fit = _run_natural_em(
        (0, 1), rows, 2, True, True, w, r, max_em_steps=3, max_rounds=3
    )
    _assert_matching_fit(fit, rows, w)
    for component in fit.components:
        np.testing.assert_array_equal(component.params[-2:], 0.0)
    assert fit.n_parameters == 7
    assert fit.n_face_parameters == 5
    assert all(
        i not in fit.free_parameter_indices for i in fit.shared_parameter_indices
    )

    # A previously zero shared side is not locked out when responsibilities or
    # the observations change. The continuation must actually optimize its release.
    changed = np.r_[rng.beta(3, 9, 90), rng.beta(9, 3, 90)][:, None]
    released = _continue_natural_mixture(
        fit, (0, 1), changed, (2, 2), True, True, w, max_em_steps=4, max_rounds=3
    )
    _assert_matching_fit(released, changed, w)
    assert any(released.components[0].params[-2:] > 0.1)


def test_degree_probe_and_continuation_preserve_fixed_coordinates():
    rows, r, w = _sample()
    fit = _run_natural_em(
        (0, 1), rows, 2, True, True, w, r, max_em_steps=3, max_rounds=3
    )
    problems, layouts, params = _degree_probe_problems(fit, rows, w, (4, 2))
    for old, problem, layout, theta in zip(
        fit.components, problems, layouts, params, strict=True
    ):
        assert problem.coordinate == old.coordinate
        np.testing.assert_array_equal(theta[:2], old.params[:2])
        np.testing.assert_array_equal(theta[-2:], old.params[-2:])
        np.testing.assert_array_equal(theta[2 : layout.curvature_slice.stop], 0.0)
    objective = _CompiledJointMixture(problems, layouts, w)
    assert objective(objective.join(params, np.log(fit.weights))).nll == pytest.approx(
        -fit.log_likelihood, abs=2e-12
    )
    lifted = _fit_natural_mixture(
        (0, 1),
        rows,
        2,
        (4, 2),
        True,
        True,
        w,
        initial_fit=fit,
        max_em_steps=3,
        max_rounds=3,
    )
    _assert_matching_fit(lifted, rows, w)
    assert lifted.log_likelihood >= fit.log_likelihood - 1e-10
    assert [c.coordinate for c in lifted.components] == [
        c.coordinate for c in fit.components
    ]


@pytest.mark.parametrize("intervals", [False, True])
def test_single_automatic_warmstart_reselects_degree(intervals):
    from scipy.stats import gennorm

    x = gennorm.rvs(4.0, size=1000, random_state=np.random.default_rng(6))
    lower = np.floor(x / 0.1) * 0.1
    rows = np.column_stack((lower, lower + 0.1)) if intervals else x[:, None]
    cold = _fit_natural_mixture(
        (-np.inf, np.inf),
        rows,
        1,
        "auto",
        responsibilities=np.ones((len(rows), 1)),
        paths=(("direct", "raw"),),
    )
    warm = _fit_natural_mixture((-np.inf, np.inf), rows, 1, "auto", initial_fit=cold)
    assert cold.components[0].spec.requested_poly_degree == 4
    assert warm.components[0].spec.requested_poly_degree == 4
    assert warm.components[0].coordinate == cold.components[0].coordinate
    assert warm.log_likelihood >= cold.log_likelihood - 1e-9
    assert warm.degree_diagnostics
    assert warm.status in ("converged", "converged_approximately")


def test_infinite_censoring_releases_shared_physical_upper_after_reflection():
    rng = np.random.default_rng(16)
    x = np.r_[rng.gamma(3, 0.5, 75), rng.gamma(3, 2, 75)]
    r = np.column_stack((x < 2.5, x >= 2.5)).astype(float)
    w = np.full(x.size, 1 / x.size)
    rows = np.column_stack((np.floor(x * 5) / 5, np.floor(x * 5) / 5 + 0.2))
    rows[rows[:, 0] > 9, 1] = np.inf
    lower = _run_natural_em(
        (0, np.inf), rows, 2, True, False, w, r, max_em_steps=4, max_rounds=3
    )
    reflected_rows = -rows[:, ::-1]
    upper = _run_natural_em(
        (-np.inf, 0),
        reflected_rows,
        2,
        False,
        True,
        w,
        r,
        max_em_steps=4,
        max_rounds=3,
    )
    _assert_matching_fit(lower, rows, w)
    _assert_matching_fit(upper, reflected_rows, w)
    assert upper.shared_parameter_indices == (None, lower.shared_parameter_indices[0])
    assert upper.components[0].params[-1] > 0.1
    assert upper.log_likelihood == pytest.approx(lower.log_likelihood, abs=2e-10)


def test_boundary_policy_warmstarts_preserve_coordinates_across_basis_changes():
    rows, r, w = _sample()
    fit = _run_natural_em(
        (0, 1), rows, 2, True, True, w, r, max_em_steps=3, max_rounds=3
    )
    coordinates = [component.coordinate for component in fit.components]
    for lower, upper, degree in (
        (False, True, (2, 2)),
        (False, True, "auto"),
        (False, False, (2, 2)),
    ):
        fit = _fit_natural_mixture(
            (0, 1),
            rows,
            2,
            degree,
            lower,
            upper,
            w,
            initial_fit=fit,
            max_em_steps=3,
            max_rounds=3,
        )
        _assert_matching_fit(fit, rows, w)
        assert [component.coordinate for component in fit.components] == coordinates
        for component in fit.components:
            assert (component.spec.physical_lower_a_index is not None) == lower
            assert (component.spec.physical_upper_a_index is not None) == upper


@pytest.mark.parametrize("intervals", [False, True])
def test_subsample_warmstart_recomputes_full_data_posteriors(intervals):
    rows, r, w = _sample(intervals)
    subset = np.arange(0, rows.shape[0], 2)
    sample_weights = w[subset] / w[subset].sum()
    screened = _run_natural_em(
        (0, 1),
        rows[subset],
        4,
        True,
        True,
        sample_weights,
        r[subset],
        max_em_steps=3,
        max_rounds=3,
    )
    fit = _fit_natural_mixture(
        (0, 1),
        rows,
        2,
        "auto",
        True,
        True,
        w,
        initial_fit=screened,
        max_em_steps=3,
        max_rounds=3,
    )
    _assert_matching_fit(fit, rows, w)
    assert fit.responsibilities.shape == r.shape
    np.testing.assert_allclose(fit.responsibilities.sum(axis=1), 1.0, atol=1e-14)
    assert [component.coordinate for component in fit.components] == [
        component.coordinate for component in screened.components
    ]
    assert fit.degree_diagnostics[0]["degrees"] == (2, 2)


@pytest.mark.filterwarnings("error:overflow encountered:RuntimeWarning")
def test_invalid_trial_covariance_backtracks_without_scaling_overflow():
    rng = np.random.default_rng(5)
    x = np.r_[rng.gamma(3, 0.5, 150), rng.normal(8, 1, 150)]
    fit = _fit_natural_mixture((0, np.inf), x, 2, 4, True, False, rng=0)
    assert fit.status in ("converged", "converged_approximately")
    assert fit.separator_certified
