"""G3 Stage 5 Phase B: interval/censored likelihood and fits (G3-044).

Defining properties of the interval objective: the gradient and observed
Hessian ``F - C`` are the derivatives of ``-sum w log P(B)``; the metric is
``F - C`` where that is positive definite relative to ``F``, else positive
definite; zero-width rows are the point objective; whole-support rows add
nothing; a face's metric is formed from the restricted ``F`` and ``C``.
"""

import math

import numpy as np
import pytest
from test_basis_intervals import _fitted, _model, _theta

from gibbus import BasisSpec, Exponential
from gibbus._fit.basis_newton import _FaceObjective
from gibbus._fit.basis_objective import (
    _BasisIntervalObjective,
    _BasisPointObjective,
    _interval_rows,
    _point_statistics,
)
from gibbus._model.basis_normalizer import _NormalizerContext

INF = math.inf
RNG = np.random.default_rng(2044)


def _objective(model, rows, weights=None):
    support = (model.request.lower, model.request.upper)
    return _BasisIntervalObjective(
        _NormalizerContext(model), _interval_rows(support, rows, weights)
    )


def _censored_rows(model, n=60):
    """Finite, half-infinite and point rows around the model's data."""
    g = model.geometry
    c = g.center + g.scale * RNG.normal(size=n)
    w = g.scale * RNG.exponential(0.4, size=n)
    lo, hi = c - w, c + w
    lo[: n // 6] = -INF
    hi[n // 6 : n // 3] = INF
    hi[n // 3 : n // 3 + 5] = lo[n // 3 : n // 3 + 5]
    return np.column_stack([lo, hi])


def _fd(f, theta, step=1e-5):
    out = []
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = step * max(1.0, abs(theta[j]))
        out.append((f(theta + e) - f(theta - e)) / (2.0 * e[j]))
    return np.array(out)


MODELS = [
    (BasisSpec(bulk_degree=4), {"x^4": 0.05, "x^2": 0.5, "x": 0.2}),
    (BasisSpec(bulk_degree=4), {"x^4": 1.0, "x^2": -4.0, "x": 0.5}),
]


@pytest.mark.parametrize("spec, weights", MODELS)
def test_gradient_and_observed_hessian_are_the_derivatives(spec, weights):
    model = _model(spec)
    theta, _ = _theta(model, weights)
    obj = _objective(model, _censored_rows(model))
    ev = obj(theta)
    grad = _fd(lambda t: obj(t).nll, theta)
    np.testing.assert_allclose(ev.gradient, grad, rtol=1e-6, atol=1e-8)
    hess = np.array(_fd(lambda t: obj(t).gradient, theta))
    observed = ev.fisher - ev.missing_information
    np.testing.assert_allclose(observed, hess.T, rtol=1e-5, atol=1e-7)


def test_metric_is_the_observed_hessian_when_positive_definite():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 0.05, "x^2": 0.5})
    g = model.geometry
    narrow = np.column_stack(
        [
            g.center + g.scale * np.linspace(-2, 2, 30),
            g.center + g.scale * np.linspace(-2, 2, 30) + 1e-3,
        ]
    )
    ev = _objective(model, narrow)(theta)
    assert ev.smallest_curvature >= 1e-8
    np.testing.assert_array_equal(
        ev.hessian,
        0.5 * (ev.fisher + ev.fisher.T)
        - 0.5 * (ev.missing_information + ev.missing_information.T),
    )


def test_metric_reflects_negative_curvature():
    # Two wide overlapping rows of a bimodal model: F - C is indefinite.
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1.0, "x^2": -4.0, "x": 0.5})
    g = model.geometry
    rows = np.array([[-INF, g.center], [g.center - 0.1 * g.scale, INF]])
    ev = _objective(model, rows)(theta)
    assert ev.smallest_curvature < 0.0
    assert np.all(np.linalg.eigvalsh(ev.hessian) > 0.0)


def test_zero_width_rows_are_the_point_objective():
    model = _model(BasisSpec(bulk_degree=4, lower_tail=Exponential(0.7)))
    theta = _fitted(model)
    g = model.geometry
    x = g.center + g.scale * RNG.normal(size=40)
    w = RNG.random(40)
    ev = _objective(model, np.column_stack([x, x]), w)(theta)
    point = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, x, w)
    )(theta)
    assert ev.nll == pytest.approx(point.nll, rel=1e-13, abs=1e-13)
    np.testing.assert_allclose(ev.gradient, point.gradient, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(ev.missing_information, 0.0)
    np.testing.assert_allclose(ev.hessian, point.fisher, rtol=1e-14, atol=1e-14)


def test_whole_support_rows_add_nothing():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 0.05, "x^2": 0.5, "x": 0.2})
    rows = _censored_rows(model, 30)
    base = _objective(model, rows)(theta)
    extra = np.vstack([rows, [[-INF, INF]] * 30])
    both = _objective(model, extra)(theta)
    # The whole-support rows carry half the weight and probability one.
    assert both.nll == pytest.approx(0.5 * base.nll, rel=1e-14)
    np.testing.assert_allclose(both.gradient, 0.5 * base.gradient, atol=1e-14)


def test_face_metric_uses_the_restricted_matrices():
    model = _model(BasisSpec(bulk_degree=4))
    theta, _ = _theta(model, {"x^4": 1.0, "x^2": -4.0, "x": 0.5})
    g = model.geometry
    rows = np.array([[-INF, g.center], [g.center - 0.1 * g.scale, INF]])
    obj = _objective(model, rows)
    # A two-dimensional face containing theta.
    basis = np.linalg.qr(np.column_stack([theta, RNG.normal(size=theta.size)]))[0]
    face = _FaceObjective(obj, basis)
    ev = face(basis.T @ theta)
    f = basis.T @ obj(theta).fisher @ basis
    c = basis.T @ obj(theta).missing_information @ basis
    np.testing.assert_allclose(ev.fisher, f, atol=1e-14)
    np.testing.assert_allclose(ev.missing_information, c, atol=1e-14)
    np.testing.assert_allclose(ev.hessian, obj.metric(f, c)[0], atol=1e-14)


# ---------------------------------------------------------------------------
# Fits: closed-form maximum likelihood, G1 parity, global certificate
# ---------------------------------------------------------------------------


def _censored_normal_nll(params, rows):
    """Exact censored-normal NLL per row (scipy log-cdf differences)."""
    from scipy.special import log_ndtr

    mu, log_sd = params
    sd = math.exp(log_sd)
    a, b = (rows[:, 0] - mu) / sd, (rows[:, 1] - mu) / sd
    point = rows[:, 0] == rows[:, 1]
    out = np.empty(rows.shape[0])
    out[point] = -0.5 * a[point] ** 2 - 0.5 * math.log(2 * math.pi) - log_sd
    # log(Phi(b) - Phi(a)) on the side with less cancellation.
    upper = ~point & (a > 0.0)
    lower = ~point & ~upper
    lb, la = log_ndtr(b[lower]), log_ndtr(a[lower])
    out[lower] = lb + np.log1p(-np.exp(la - lb))
    lb, la = log_ndtr(-a[upper]), log_ndtr(-b[upper])
    out[upper] = lb + np.log1p(-np.exp(la - lb))
    return -float(np.mean(out))


@pytest.mark.parametrize("kind", ["binned", "censored", "mixed"])
def test_normal_fit_is_the_censored_normal_mle(kind):
    from scipy.optimize import minimize

    from gibbus._fit.basis_selection import _fit_basis

    rng = np.random.default_rng(11)
    x = rng.normal(1.5, 2.0, 300)
    if kind == "binned":
        e = np.floor(x)
        rows = np.column_stack([e, e + 1.0])
    elif kind == "censored":
        rows = np.column_stack([x, x])
        rows[x > 3.0] = [3.0, INF]
        rows[x < -1.0] = [-INF, -1.0]
    else:
        h = np.abs(rng.normal(0.0, 0.5, (300, 2)))
        rows = np.column_stack([x - h[:, 0], x + h[:, 1]])
        rows[:40, 1] = rows[:40, 0]
        rows[x > 4.0, 1] = INF
    fit = _fit_basis(BasisSpec(bulk_degree=2, expansion="minimal"), rows)
    assert fit.result.status == "converged"
    ref = minimize(
        _censored_normal_nll,
        [1.5, math.log(2.0)],
        args=(rows,),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 20000},
    )
    assert fit.result.objective_value == pytest.approx(ref.fun, abs=1e-11)


@pytest.mark.parametrize("censor", [0.0, 0.3])
def test_exponential_fit_is_the_closed_form_mle(censor):
    from gibbus._fit.basis_selection import _fit_basis

    rng = np.random.default_rng(12)
    t = rng.exponential(2.0, 400)
    c = rng.exponential(2.0 / censor, 400) if censor else np.full(400, INF)
    obs = np.minimum(t, c)
    event = t <= c
    rows = np.column_stack([obs, np.where(event, obs, INF)])
    fit = _fit_basis(BasisSpec(support=(0.0, INF), bulk_degree=1), rows)
    assert fit.result.status == "converged"
    rate = event.sum() / obs.sum()
    # NLL per row: -(events log rate - rate sum t) / n.
    nll = -(event.sum() * math.log(rate) - rate * obs.sum()) / obs.size
    assert fit.result.objective_value == pytest.approx(nll, abs=1e-12)


G1_CASES = {
    "binned normal": ((-INF, INF), 4),
    "binned censored": ((-INF, INF), 4),
    "random intervals": ((-INF, INF), 4),
    "censored exponential": ((0.0, INF), 2),
    "binned bounded": ((0.0, 1.0), 2),
}


def test_a_saturated_fit_reaches_the_nonparametric_bound():
    from gibbus._fit.basis_selection import _fit_basis

    rng = np.random.default_rng(15)
    x = rng.normal(0.0, 1.0, 500)
    # Three bins: the normal family attains every bin-probability vector.
    rows = np.where(
        x[:, None] < -0.3,
        [-INF, -0.3],
        np.where(x[:, None] < 0.8, [-0.3, 0.8], [0.8, INF]),
    )
    fit = _fit_basis(BasisSpec(bulk_degree=2, expansion="minimal"), rows)
    assert fit.result.status == "converged"
    assert abs(fit.result.global_gap) <= 1e-10


def test_interval_inputs_follow_g1_endpoint_rules():
    from gibbus._fit.basis_selection import _fit_basis

    g = np.random.default_rng(16).gamma(2.0, 1.0, 100)
    rows = np.column_stack([np.floor(g), np.floor(g) + 1.0])
    # Positive-width rows touching a log endpoint are fine...
    fit = _fit_basis(BasisSpec(support=(0.0, INF), log_lower=True), rows)
    assert fit.result.status == "converged"
    # ...an exact observation there is not.
    with pytest.raises(ValueError, match="lower support endpoint"):
        _fit_basis(
            BasisSpec(support=(0.0, INF), log_lower=True), np.vstack([rows, [0.0, 0.0]])
        )
    with pytest.raises(ValueError, match="inside the closed support"):
        _fit_basis(BasisSpec(support=(0.0, INF)), np.vstack([rows, [-INF, 1.0]]))


def test_an_expansion_that_cannot_lower_the_nll_is_not_kept():
    # Censored exponential data: the fit sits on the active curvature row
    # (zero z**2 coefficient), the joint score points out of the shape cone
    # (p = 0.006) and the extension gains nothing; the likelihood ratio keeps
    # the minimal span.
    from gibbus._fit.basis_selection import _fit_basis

    rng = np.random.default_rng(1)
    rng.normal(size=400)
    t, c = rng.exponential(2.0, 400), rng.exponential(3.0, 400)
    rows = np.column_stack([np.minimum(t, c), np.where(t <= c, t, INF)])
    fit = _fit_basis(
        BasisSpec(support=(0.0, INF), bulk_degree="auto", log_lower="auto"), rows
    )
    step = fit.history[-1]
    assert step.kind == "expansion" and step.diagnostic.should_expand
    assert step.chosen is None and step.lr_p_value > 0.05
    assert fit.model.n_coefficients == 2
