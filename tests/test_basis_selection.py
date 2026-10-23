"""G3 Stage 4c: automatic expansion by G1's omitted-statistic score test (G3-035).

Defining properties: one-slot growth strictly enlarges the span and the lift
is exact; the joint gate does not add slots to data from the minimal family
and is calibrated near its level; skewed data gain slots, every accepted
extension converges with a nonnegative shape and is no worse than its lifted
start; an explicit full expansion that cannot be fitted fails loudly with the
span the score test supports; supporting driver properties (exact face bases,
restoration toward a convex anchor, scaled polish, best-lead tail verdicts).
"""

import math
import warnings

import numpy as np
import pytest

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial
from gibbus._fit.basis_constraints import _ConstraintRows
from gibbus._fit.basis_newton import _Face, _polish, _solve_basis, _toward_anchor
from gibbus._fit.basis_selection import (
    _fit_basis,
    _lift_matrix,
    _PointData,
    _span_diagnostic,
)
from gibbus._fit.conic_newton import _NewtonOptions
from gibbus._fit.degree import _DegreeSelectionConfig
from gibbus._fit.inputs import _normalize_basis_inputs
from gibbus._model.basis import _build_basis, _full_basis, _grow_basis
from gibbus._model.basis_tails import _build_tail_model, _growth_verdict

EXP = {"bulk_degree": 2, "lower_tail": Exponential(0.5), "upper_tail": Exponential(0.5)}


def _normal(seed, n=400):
    return np.random.default_rng(seed).normal(1.5, 2.0, n)


def _model(spec, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, x)


def test_auto_is_the_default_expansion():
    assert BasisSpec().expansion == "auto"
    with pytest.raises(ValueError):
        BasisSpec(expansion="maximal")


def test_growth_strictly_enlarges_the_span_with_an_exact_lift():
    x = _normal(0)
    model = _model(BasisSpec(**EXP), x)
    full = _full_basis(model)
    candidates = _grow_basis(model)
    assert len(candidates) == 10
    for _, larger in candidates:
        assert model.n_coefficients < larger.n_coefficients <= full.n_coefficients
        lift = _lift_matrix(model, larger)
        np.testing.assert_allclose(
            larger.coefficient_map @ lift, model.coefficient_map, atol=1e-13
        )
    # The minimal model reproduces the "auto" start exactly.
    minimal = _model(BasisSpec(**EXP, expansion="minimal"), x)
    assert minimal.slot_choice == model.slot_choice


@pytest.mark.parametrize("seed", range(4))
def test_minimal_family_data_keep_the_minimal_span(seed):
    x = _normal(seed)
    fit = _fit_basis(BasisSpec(**EXP), x)
    assert fit.result.status == "converged"
    assert fit.model.n_coefficients == 3
    step = fit.history[0]
    assert step.kind == "expansion" and step.chosen is None
    assert step.diagnostic.added == 10 and not step.diagnostic.should_expand


def test_joint_gate_is_calibrated_on_minimal_family_data():
    # 40 Gaussian samples: the gate should reject rarely (level 0.05, G1's
    # conservative variance inflation makes it at most nominal).
    cfg = _DegreeSelectionConfig()
    spec = BasisSpec(**EXP, expansion="minimal")
    p = []
    for s in range(40):
        x = _normal(1000 + s)
        model = _model(spec, x)
        result = _solve_basis(_PointData(x, None, x.size).objective(model))
        gate = _span_diagnostic(
            model,
            result.params,
            _full_basis(model),
            _PointData(x, None, x.size),
            cfg,
            "joint",
        )
        p.append(gate.p_value)
    p = np.array(p)
    assert np.mean(p < 0.05) <= 0.15
    assert 0.3 <= np.median(p) <= 0.85


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_skewed_data_gain_converged_convex_extensions(seed):
    x = np.random.default_rng(seed).gumbel(0.0, 1.0, 400)
    minimal = _fit_basis(BasisSpec(**EXP, expansion="minimal"), x)
    fit = _fit_basis(BasisSpec(**EXP), x)
    assert fit.model.n_coefficients > 3
    assert fit.result.status == "converged"
    assert fit.result.final_separation.status == "nonnegative"
    assert fit.result.objective_value <= minimal.result.objective_value + 1e-12
    assert all(h.chosen is not None for h in fit.history[:-1])
    assert fit.history[-1].chosen is None


def test_warm_started_extensions_are_no_worse_than_their_start():
    x = np.random.default_rng(0).laplace(0.0, 1.0, 400)
    model = _model(BasisSpec(**EXP, expansion="minimal"), x)
    result = _solve_basis(_PointData(x, None, x.size).objective(model))
    for label, larger in _grow_basis(model):
        start = _lift_matrix(model, larger) @ result.params
        grown = _solve_basis(
            _PointData(x, None, x.size).objective(larger),
            start=start,
            start_face=(result.face.suppressed, ()),
        )
        assert grown.status == "converged", label
        assert grown.final_separation.status == "nonnegative", label
        assert grown.objective_value <= result.objective_value + 1e-12, label


def test_explicit_full_is_returned_when_it_converges():
    # With globally convex Newton iterates (G3-038) the 13-column full MLE
    # (the near-unidentified noise fit of G3-035) is reached and certified.
    x = _normal(0)
    full = _fit_basis(BasisSpec(**EXP, expansion="full"), x)
    minimal = _fit_basis(BasisSpec(**EXP, expansion="minimal"), x)
    assert full.model.n_coefficients == 13
    assert full.result.status == "converged"
    assert full.result.final_separation.status == "nonnegative"
    assert full.result.objective_value <= minimal.result.objective_value


def test_explicit_full_fails_loudly_with_the_supported_span():
    x = _normal(0)
    with pytest.raises(RuntimeError, match="expansion='full' could not be fitted"):
        _fit_basis(
            BasisSpec(**EXP, expansion="full"),
            x,
            options=_NewtonOptions(max_iterations=0),
        )


def test_face_basis_keeps_exact_unit_columns():
    d = 5
    eq = np.array([[0.0, 0.0, 0.3, -0.7, 0.0]])
    n = _Face(eq, (), (), np.zeros((0, d))).basis(d)
    np.testing.assert_allclose(n.T @ n, np.eye(4), atol=1e-15)
    assert np.max(np.abs(eq @ n)) <= 1e-16
    for j in (0, 1, 4):
        column = n[:, np.flatnonzero(n[j])[0]]
        assert column[j] == 1.0 and np.count_nonzero(column) == 1


def test_toward_anchor_reaches_feasibility_on_the_segment():
    rng = np.random.default_rng(3)
    for _ in range(50):
        d = int(rng.integers(2, 7))
        anchor = rng.normal(size=d)
        a = rng.normal(size=(2 * d, d))
        a[(a @ anchor) < 0.0] *= -1.0
        rows = _ConstraintRows(np.zeros((0, d)), ()).merged(a, "cut")
        phi = rng.normal(size=d) * 3.0
        if np.min(rows.slack(phi)) >= 0.0:
            continue
        out = _toward_anchor(rows, phi, anchor, np.ones(d))
        assert np.min(rows.slack(out)) >= -1e-14
        t = np.linalg.lstsq((anchor - phi)[:, None], out - phi, rcond=None)[0][0]
        assert -1e-12 <= t <= 1.0 + 1e-12
        np.testing.assert_allclose(out, phi + t * (anchor - phi), atol=1e-12)


def test_polish_measures_violations_in_the_solver_scales():
    # A far exponential-tail cut: e_exp plus polynomial entries ~1e-17 of it.
    far = np.array([0.0, 0.4856006, -44.6269, 7.135548e17])
    rows = _ConstraintRows(np.zeros((0, 4)), ()).merged(far, "cut")
    scales = np.array([1.0, 1.0, 1.0, 1e-18])
    poly = np.array([-0.0325, 0.52, 0.00644])
    exact = -(far[:3] @ poly) / far[3]
    # Rounding-level in the solver's scales (relative 1e-11): polished.
    phi = np.append(poly, exact * (1.0 - 1e-11))
    assert rows.rows[0] @ phi > -1e-30  # invisible to an unscaled test
    out = _polish(rows, phi, scales)
    assert out is not phi
    assert far @ out >= 0.0
    np.testing.assert_allclose(out, phi, rtol=1e-10)
    # A genuine violation (q'' = -0.03 at the contact) is left to the solver.
    real = np.append(poly, 9.2584e-21)
    assert far @ real < -0.02
    assert _polish(rows, real, scales) is real


def test_polish_respects_exact_log_domain_below_the_roundoff_floor():
    rows = _ConstraintRows(np.eye(2), ("cut", "log"))
    phi = np.array([1.0, -np.finfo(np.float64).eps / 4.0])
    out = _polish(rows, phi, np.array([1.0, 100.0]))
    assert rows.slack(out)[1] >= 0.0
    np.testing.assert_allclose(out, phi, atol=1e-12, rtol=0.0)


def test_growth_verdict_uses_a_robust_lead_beyond_a_tiny_one():
    # q = a z^2 + b z with a tiny positive: the linear slope proves outward
    # growth a few scales out; the tiny quadratic alone would need |z|~1e16.
    x = _normal(0)
    model = _model(BasisSpec(bulk_degree=2, expansion="minimal"), x)
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("x^2")] = 1e-17
    theta[labels.index("x")] = -0.3
    tail = _build_tail_model(model, "lower", order=1)
    res = _growth_verdict(tail, model, theta)
    assert res.status == "proper"
    assert res.slope.zeta <= 100.0
    assert math.isfinite(res.slope.cutoff)


def test_explicit_minimal_and_polynomial_tails_are_untouched():
    x = _normal(1)
    spec = BasisSpec(bulk_degree=3, upper_tail=Polynomial(1), expansion="minimal")
    fit = _fit_basis(spec, x)
    assert fit.history == ()
    assert fit.model.slot_choice == _model(spec, x).slot_choice


# --- degree ladder, automatic endpoint logs, inputs (G3-035) ----------------


_RNG = np.random.default_rng(5)
_LADDER = {
    "normal-line": ((-math.inf, math.inf), False, False, _RNG.normal(1.5, 2.0, 600)),
    "bimodal-line": (
        (-math.inf, math.inf),
        False,
        False,
        np.r_[_RNG.normal(-2.0, 1.0, 300), _RNG.normal(2.5, 0.8, 300)],
    ),
    "gumbel-line": ((-math.inf, math.inf), False, False, _RNG.gumbel(0.0, 1.0, 600)),
    "gamma-half-log": ((0.0, math.inf), "auto", False, _RNG.gamma(2.5, 1.0, 600)),
    "beta-bounded-logs": ((0.0, 1.0), "auto", "auto", _RNG.beta(2.0, 5.0, 600)),
    "uniform-bounded-logs": ((0.0, 1.0), "auto", "auto", _RNG.uniform(0.0, 1.0, 600)),
}


def test_auto_fields_are_resolved_by_the_fitter_only():
    spec = BasisSpec(bulk_degree="auto", log_lower="auto")
    with pytest.raises(ValueError, match="resolved by the fitter"):
        _model(spec, _normal(0))
    with pytest.raises(ValueError):
        BasisSpec(log_lower=True)  # infinite lower end
    with pytest.raises(TypeError):
        BasisSpec(bulk_degree="two")


def test_inputs_resolve_log_policy_with_g1_endpoint_rules():
    x = np.random.default_rng(0).gamma(2.0, 1.0, 50)
    half = (0.0, math.inf)
    auto = BasisSpec(support=half, log_lower="auto")
    assert _normalize_basis_inputs(auto, x).log_policy == ("auto", False)
    # An exact endpoint observation settles an automatic term to "no term".
    assert _normalize_basis_inputs(auto, np.r_[x, 0.0]).log_policy == (
        False,
        False,
    )
    with pytest.raises(ValueError):
        _normalize_basis_inputs(BasisSpec(support=half, log_lower=True), np.r_[x, 0.0])
    with pytest.raises(ValueError, match="inside the closed support"):
        _normalize_basis_inputs(BasisSpec(support=half), np.r_[x, -1.0])
    rows = _normalize_basis_inputs(BasisSpec(support=half), np.c_[x, x + 1.0])
    assert rows.intervals and rows.samples.shape == (x.size, 2)
    with pytest.raises(ValueError, match="inside the closed support"):
        _normalize_basis_inputs(BasisSpec(support=half), np.c_[x - 5.0, x])


@pytest.mark.parametrize("name", list(_LADDER))
def test_automatic_degree_and_logs_converge_convex(name):
    support, lower, upper, x = _LADDER[name]
    spec = BasisSpec(
        support=support,
        bulk_degree="auto",
        log_lower=lower,
        log_upper=upper,
        expansion="minimal",
    )
    fit = _fit_basis(spec, x)
    assert fit.result.status == "converged"
    assert fit.result.final_separation.status == "nonnegative"


def test_boundary_terms_are_kept_or_dropped_by_the_lr_test():
    _, _, _, beta = _LADDER["beta-bounded-logs"]
    _, _, _, uniform = _LADDER["uniform-bounded-logs"]
    spec = BasisSpec(support=(0.0, 1.0), log_lower="auto", log_upper="auto")
    kept = _fit_basis(spec, beta)
    assert kept.model.request.log_lower and kept.model.request.log_upper
    assert all(p < 0.05 for p in kept.boundary_p_values)
    dropped = _fit_basis(spec, uniform)
    assert not dropped.model.request.log_lower and not dropped.model.request.log_upper


def test_fully_automatic_tail_fit_is_converged_and_convex():
    x = np.random.default_rng(5).gumbel(0.0, 1.0, 600)
    spec = BasisSpec(
        bulk_degree="auto", lower_tail=Exponential(0.5), upper_tail=Exponential(0.5)
    )
    fit = _fit_basis(spec, x)
    assert fit.result.status == "converged"
    assert fit.result.final_separation.status == "nonnegative"
    kinds = [h.kind for h in fit.history]
    assert kinds[0] == "degree" and kinds[-1] == "expansion"
