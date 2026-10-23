"""Direct orthogonal evaluation and equivalence of the runtime fitting basis."""

from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from scipy.special import eval_genlaguerre

from gibbus import BasisSpec
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._fit.laguerre import _laguerre_model
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_interval import _exact_ungated_curvature, _potential_enclosure
from gibbus._model.basis_normalizer import _NormalizerContext


def _models(degree=12):
    data = np.random.default_rng(7).lognormal(0.0, 0.6, 400)
    model = _build_basis(
        BasisSpec(support=(0.0, np.inf), bulk_degree=degree, log_lower=True), data
    )
    stats = _point_statistics(model, data)
    theta = np.zeros(model.n_coefficients)
    labels = [c.label for c in model.columns]
    theta[labels.index("x")] = model.geometry.scale
    theta[labels.index("logL")] = 0.35
    orthogonal = _laguerre_model(model, stats, theta)
    return model, orthogonal, data, theta


@pytest.mark.parametrize("order", range(5))
def test_recurrence_derivatives_agree_with_laguerre_identity(order):
    _, model, _, _ = _models()
    plan = model.polynomial_coordinates
    points = plan.scale * np.array([0.01, 0.5, 1.0, 2.0, 7.0, 20.0])
    values = _evaluate_basis(model, points, max_order=order).derivative(order)
    for col, n in enumerate(plan.column_degree):
        if not n:
            continue
        reference = (
            np.zeros(len(points))
            if n < order
            else (-1) ** order
            * eval_genlaguerre(n - order, plan.alpha + order, points / plan.scale)
            / plan.norms[n]
            / plan.scale**order
        )
        if order == 0:
            reference -= plan.reference[n]
        np.testing.assert_allclose(values[:, col], reference, rtol=3e-12, atol=3e-12)


def test_runtime_map_and_recurrence_represent_the_same_polynomial():
    _, model, _, _ = _models()
    plan = model.polynomial_coordinates
    points = plan.scale * np.array([0.2, 1.0, 5.0, 15.0])
    values = _evaluate_basis(model, points, max_order=0).phi
    with mp.workdps(80):
        for row, x in enumerate(points):
            z = (mp.mpf(float(x)) - mp.mpf(model.geometry.center)) / mp.mpf(
                model.geometry.scale
            )
            for col, n in enumerate(plan.column_degree):
                if not n:
                    continue
                expected = sum(
                    mp.mpf(v.numerator) / v.denominator * z**fn.degree
                    for fn in model.functions
                    if fn.kind == "monomial"
                    for v in [model.coefficient_map_exact[3 * fn.index][col]]
                )
                assert float(values[row, col]) == pytest.approx(
                    float(expected), rel=3e-12, abs=3e-12
                )


def test_direct_objective_matches_canonical_density_and_fisher():
    original, model, data, theta = _models()
    plan = model.polynomial_coordinates
    theta[1] = 0.1
    theta[11] = 1e-16
    beta = plan.from_canonical(theta)
    np.testing.assert_allclose(plan.to_canonical(beta), theta, atol=2e-14, rtol=2e-12)
    first = _BasisPointObjective(
        _NormalizerContext(original), _point_statistics(original, data)
    )(theta)
    second = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )(beta)
    assert second.nll == pytest.approx(first.nll, abs=2e-11)
    np.testing.assert_allclose(
        second.gradient, plan.transform.T @ first.gradient, atol=2e-10, rtol=2e-9
    )
    expected = plan.transform.T @ first.fisher @ plan.transform
    scale = np.sqrt(np.diag(second.fisher))
    assert (
        np.max(np.abs(second.fisher - expected) / scale[:, None] / scale[None, :])
        < 2e-8
    )


def test_mapped_point_bounds_contain_exact_curvature():
    _, model, _, theta = _models()
    plan = model.polynomial_coordinates
    beta = plan.from_canonical(theta)
    beta += np.random.default_rng(9).normal(scale=1e-5, size=len(beta))
    points = plan.scale * np.array([0.3, 1.0, 3.0, 10.0, 20.0])
    lower, upper = _potential_enclosure(model, beta, points, points, (2,))[2]
    for row, point in enumerate(points):
        z = (Fraction(float(point)) - Fraction(model.geometry.center)) / Fraction(
            model.geometry.scale
        )
        exact = Fraction(0)
        for fn in model.functions:
            coefficient = sum(
                v * Fraction(float(b))
                for v, b in zip(model.coefficient_map_exact[3 * fn.index], beta)
            )
            if fn.kind == "monomial" and fn.degree >= 2:
                exact += (
                    coefficient
                    * fn.degree
                    * (fn.degree - 1)
                    * z ** (fn.degree - 2)
                    / Fraction(model.geometry.scale) ** 2
                )
            elif fn.kind == "log":
                exact += coefficient / Fraction(float(point)) ** 2
        assert Fraction(float(lower[row])) <= exact <= Fraction(float(upper[row]))


@pytest.mark.parametrize(
    "unresolved,tail_status",
    [(0, "nonnegative"), (1, "nonnegative"), (0, "inconclusive")],
)
def test_exact_contact_fallback_preserves_unresolved_verdicts(
    monkeypatch, unresolved, tail_status
):
    from gibbus._fit import curvature_search as search

    model, _, _, theta = _models()
    point = 2.0
    exact = Fraction(float(theta[-1])) / Fraction(point) ** 2
    bounds = _exact_ungated_curvature(model, theta, [point])[0]
    assert Fraction(bounds[1]) <= exact <= Fraction(bounds[2])
    row = _evaluate_basis(model, [point]).d2phi
    tail = ("upper", tail_status, 10.0, 10.0, 1.0, 1.0, 0.0, 1.0, -1)
    monkeypatch.setattr(
        search,
        "separate",
        lambda *args: (
            "inconclusive",
            1e-3,
            (0.1, 10.0),
            [],
            [(point, float(exact), -1.0, 1.0, "interior")],
            [tail],
            np.array([point]),
            row,
            1,
            unresolved,
        ),
    )
    result = search._separate_curvature(search._CurvatureContext(model), theta)
    expected = (
        "nonnegative"
        if unresolved == 0 and tail_status == "nonnegative"
        else "inconclusive"
    )
    assert result.status == expected
