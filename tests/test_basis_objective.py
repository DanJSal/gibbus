"""Weighted point objective (NLL, gradient, Fisher Hessian).

Finite differences check the gradient against the NLL and the Hessian
against the gradient; the Gaussian maximum-likelihood point makes the
gradient vanish; weights behave (zero weights neutral, scaling invariant);
the evaluation-record contract holds; points on an enabled log endpoint
are rejected and improper vectors raise a ``RuntimeError``.
"""

import math
import warnings

import numpy as np
import pytest

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._model.basis import _build_basis
from gibbus._model.basis_normalizer import _ImproperPotential, _NormalizerContext

RNG = np.random.default_rng(2026)
NORMAL = RNG.normal(1.5, 2.0, size=500)
HALF = RNG.gamma(2.0, 1.5, size=500) + 0.25
BOUNDED = RNG.beta(2.0, 3.0, size=500) * 4.0 - 1.0


def _model(spec, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, data)


CASES = [
    (BasisSpec(bulk_degree=2), NORMAL, {"z^2": 0.6, "z": -0.2}),
    (
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(1.5)),
        NORMAL,
        {"z": -0.5, "powU*(-1+tU)": -0.5, "z^2*(tL+tU)": 0.3, "z^4*(tL+tU)": 0.01},
    ),
    (
        BasisSpec(bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Polynomial(3)),
        NORMAL,
        {"expL*(-1+tL)": -0.3, "z^3": 0.02, "z^2": 0.4, "z": 0.1},
    ),
    (
        BasisSpec(support=(0.0, math.inf), bulk_degree=2, log_lower=True),
        HALF,
        {"logL": 1.3, "z^2": 0.2, "z": 0.5},
    ),
    (
        BasisSpec(support=(-1.0, 3.0), bulk_degree=3, log_lower=True, log_upper=True),
        BOUNDED,
        {"logL": 1.1, "logU": 2.2, "z^3": 0.1, "z^2": 0.3, "z": -0.2},
    ),
]


def _hand_theta(model, weights):
    """Comfortably proper hand-built vector (grid-LP vectors sit on boundaries)."""
    labels = [c.label for c in model.columns]
    assert set(weights) <= set(labels), labels
    theta = np.zeros(model.n_coefficients)
    for label, value in weights.items():
        theta[labels.index(label)] = value
    return theta


@pytest.mark.parametrize("spec,data,weights", CASES, ids=lambda s: "")
def test_gradient_and_hessian_match_finite_differences(spec, data, weights):
    model = _model(spec, data)
    ctx = _NormalizerContext(model)
    stats = _point_statistics(model, data)
    theta = _hand_theta(model, weights)
    objective = _BasisPointObjective(ctx, stats)
    obj = objective(theta)
    assert np.array_equal(obj.hessian, obj.fisher)
    assert not np.any(obj.missing_information)
    np.testing.assert_array_equal(obj.gradient, stats.mean - obj.model_partial_means)
    d = model.n_coefficients
    for c in range(d):
        step = 1e-5 * max(1.0, abs(theta[c]))
        e = np.zeros(d)
        e[c] = step
        plus = objective(theta + e)
        minus = objective(theta - e)
        fd = (plus.nll - minus.nll) / (2 * step)
        assert fd == pytest.approx(obj.gradient[c], rel=1e-6, abs=1e-7)
        fd_h = (plus.gradient - minus.gradient) / (2 * step)
        scale = np.sqrt(np.abs(np.diag(obj.hessian)) * abs(obj.hessian[c, c])) + 1e-12
        assert np.all(np.abs(fd_h - obj.hessian[:, c]) <= 1e-5 * scale + 1e-9)
    np.testing.assert_array_equal(obj.hessian, obj.hessian.T)
    assert np.min(np.linalg.eigvalsh(obj.hessian)) >= -1e-10 * np.max(
        np.abs(obj.hessian)
    )


def test_gaussian_mle_has_zero_gradient():
    model = _model(BasisSpec(bulk_degree=2), NORMAL)
    labels = [c.label for c in model.columns]
    z = model.geometry.to_internal(NORMAL)
    m, v = z.mean(), z.var()
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("z^2")] = 1.0 / (2.0 * v)
    theta[labels.index("z")] = -m / v
    obj = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, NORMAL)
    )(theta)
    assert np.max(np.abs(obj.gradient)) < 1e-10 * max(1.0, v)
    # nll at the MLE: 0.5 log(2 pi v h^2) + 0.5, shifted by the gauge
    # (theta . Phi_bar contains -m^2 / (2v) relative to the centered form).
    h = model.geometry.scale
    exact = 0.5 * math.log(2 * math.pi * v * h * h) + 0.5
    assert obj.nll == pytest.approx(exact, rel=1e-12)


def test_zero_weights_are_neutral_and_scaling_invariant():
    spec, data, weights = CASES[1]
    model = _model(spec, data)
    ctx = _NormalizerContext(model)
    theta = _hand_theta(model, weights)
    w = RNG.uniform(0.5, 2.0, size=data.size)
    base = _BasisPointObjective(ctx, _point_statistics(model, data, w))(theta)
    extra = np.concatenate([data, RNG.normal(size=7)])
    padded = _BasisPointObjective(
        ctx, _point_statistics(model, extra, np.concatenate([w, np.zeros(7)]))
    )(theta)
    scaled = _BasisPointObjective(ctx, _point_statistics(model, data, 3.7 * w))(theta)
    assert padded.nll == base.nll and np.array_equal(padded.gradient, base.gradient)
    assert scaled.nll == pytest.approx(base.nll, rel=1e-14)


def test_point_on_a_log_endpoint_is_rejected():
    # The public boundary rejects these;
    # the statistic itself refuses a non-finite value.
    data = np.concatenate([HALF, [0.0]])
    model = _model(
        BasisSpec(support=(0.0, math.inf), bulk_degree=2, log_lower=True), data
    )
    with pytest.raises(ValueError):
        _point_statistics(model, data)
    w = np.ones(data.size)
    w[-1] = 0.0
    assert np.all(np.isfinite(_point_statistics(model, data, w).mean))


def test_improper_and_invalid_inputs():
    model = _model(BasisSpec(bulk_degree=2), NORMAL)
    stats = _point_statistics(model, NORMAL)
    labels = [c.label for c in model.columns]
    theta = np.zeros(model.n_coefficients)
    theta[labels.index("z^2")] = -1.0
    objective = _BasisPointObjective(_NormalizerContext(model), stats)
    with pytest.raises(_ImproperPotential):
        objective(theta)
    assert issubclass(_ImproperPotential, RuntimeError)
    assert objective.build_state(theta).status == "improper"
    with pytest.raises(ValueError):
        _point_statistics(model, NORMAL, np.zeros(NORMAL.size))
    with pytest.raises(ValueError):
        _point_statistics(model, NORMAL, -np.ones(NORMAL.size))
