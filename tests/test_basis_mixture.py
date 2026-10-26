"""Shared-support mixtures.

Defining properties: the joint observed NLL's gradient and observed Hessian
``G - M`` are its derivatives; one component is the single-fit objective;
the EM surrogate has the observed gradient at its expansion point; the
stacked M-step without shared columns equals independent fits; fitted
mixtures equal the exact two-normal mixture MLE.  Selection: one
component's joint diagnostic is the single fit's; a dropped shared log is
the likelihood-ratio decision; every kept extension passed its likelihood
ratio; the component count's score is the BIC.
"""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

from gibbus import BasisSpec, Exponential, GibbusBasisWarning
from gibbus._fit.basis_mixture import (
    _fit_mixture_auto,
    _joint_objective,
    _mixture_blocks,
    _mixture_data,
    _mixture_span_diagnostic,
    _MixtureRows,
    _mstep_objective,
)
from gibbus._fit.basis_newton import _solve_basis
from gibbus._fit.basis_objective import (
    _BasisIntervalObjective,
    _BasisPointObjective,
    _interval_rows,
    _point_statistics,
    _PointStatistics,
)
from gibbus._model.basis import _build_basis
from gibbus._model.basis_normalizer import _NormalizerContext

INF = math.inf
RNG = np.random.default_rng(2045)
TWO = np.r_[RNG.normal(-2.0, 1.0, 300), RNG.normal(3.0, 1.5, 300)]
GAMMAS = np.r_[RNG.gamma(2.0, 1.0, 200), RNG.gamma(9.0, 1.0, 200)]


def _model(spec, x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, x)


def _theta(model, values):
    labels = [c.label for c in model.columns]
    out = np.zeros(model.n_coefficients)
    for label, value in values.items():
        out[labels.index(label)] = value
    return out


def _data(kind, x, support=(-INF, INF)):
    if kind == "points":
        return _MixtureRows(np.full(x.size, 1.0 / x.size), x=x)
    e = np.floor(x * 2.0) / 2.0
    rows = np.column_stack([e, e + 0.5])
    if math.isinf(support[1]):
        rows[x > np.quantile(x, 0.95)] = [np.quantile(x, 0.95), INF]
    r = _interval_rows(support, rows)
    return _MixtureRows(r.weights, rows=r)


def _fd(f, x0, step=1e-6):
    out = []
    for j in range(x0.size):
        e = np.zeros_like(x0)
        e[j] = step * max(1.0, abs(x0[j]))
        out.append((f(x0 + e) - f(x0 - e)) / (2.0 * e[j]))
    return np.array(out)


CASES = [
    ("points", BasisSpec(bulk_degree=4), TWO, (-INF, INF)),
    ("intervals", BasisSpec(bulk_degree=4), TWO, (-INF, INF)),
    (
        "points",
        BasisSpec(support=(0.0, INF), bulk_degree=2, log_lower=True),
        GAMMAS,
        (0.0, INF),
    ),
    (
        "intervals",
        BasisSpec(support=(0.0, INF), bulk_degree=2, log_lower=True),
        GAMMAS,
        (0.0, INF),
    ),
]


def _start(model):
    labels = [c.label for c in model.columns]
    if "logL" in labels:
        a = _theta(model, {"z": 1.0, "z^2": 0.05, "logL": 1.0})
        b = _theta(model, {"z": 0.6, "z^2": 0.01, "logL": 1.0})
        shared = [labels.index("logL")]
    else:
        a = _theta(model, {"z^2": 0.4, "z": 1.0, "z^4": 0.02})
        b = _theta(model, {"z^2": 0.3, "z": -0.5, "z^4": 0.02})
        shared = []
    keep = [j for j in range(model.n_coefficients) if j not in shared]
    return np.concatenate([a[keep], b[keep], a[shared], [0.2]])


@pytest.mark.parametrize("kind, spec, x, support", CASES)
def test_joint_gradient_and_observed_hessian_are_the_derivatives(
    kind, spec, x, support
):
    model = _model(spec, x)
    obj = _joint_objective((model,) * 2, _data(kind, x, support))
    t = _start(model)
    ev = obj(t)
    np.testing.assert_allclose(
        ev.gradient, _fd(lambda z: obj(z).nll, t), rtol=1e-6, atol=1e-8
    )
    hess = _fd(lambda z: obj(z).gradient, t)
    observed = ev.fisher - ev.missing_information
    np.testing.assert_allclose(observed, hess.T, rtol=1e-5, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(ev.hessian) > 0.0)


@pytest.mark.parametrize("kind", ["points", "intervals"])
def test_one_component_is_the_single_fit_objective(kind):
    model = _model(BasisSpec(bulk_degree=4), TWO)
    data = _data(kind, TWO)
    theta = _theta(model, {"z^2": 0.4, "z": 1.0, "z^4": 0.02})
    ev = _joint_objective((model,), data)(theta)
    ctx = _NormalizerContext(model)
    single = (
        _BasisPointObjective(ctx, _point_statistics(model, TWO))
        if kind == "points"
        else _BasisIntervalObjective(ctx, data.rows)
    )(theta)
    assert ev.nll == pytest.approx(single.nll, abs=1e-13)
    np.testing.assert_allclose(ev.gradient, single.gradient, atol=1e-12)
    np.testing.assert_allclose(ev.fisher, single.fisher, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        ev.missing_information, single.missing_information, atol=1e-12
    )


@pytest.mark.parametrize("kind, spec, x, support", CASES)
def test_em_surrogate_has_the_observed_gradient(kind, spec, x, support):
    model = _model(spec, x)
    data = _data(kind, x, support)
    joint = _joint_objective((model,) * 2, data)
    t = _start(model)
    _, resp = joint.row_terms(t)
    surrogate = _mstep_objective((model,) * 2, data, resp)
    np.testing.assert_allclose(
        surrogate(t[:-1]).gradient, joint(t).gradient[:-1], atol=1e-14
    )


def test_stacked_m_step_without_shared_columns_is_independent_fits():
    model = _model(BasisSpec(bulk_degree=4), TWO)
    data = _data("points", TWO)
    resp = np.column_stack([TWO < 0.5, TWO >= 0.5]).astype(float) * 0.9 + 0.05
    mstep = _mstep_objective((model,) * 2, data, resp)
    stacked = _solve_basis(mstep, blocks=_mixture_blocks((model,) * 2))
    assert stacked.status == "converged"
    total = 0.0
    for k in range(2):
        objective = _BasisPointObjective(
            _NormalizerContext(model),
            _PointStatistics(mstep.data["stats"][k], 1.0, TWO.size),
        )
        total += mstep.data["mass"][k] * _solve_basis(objective).objective_value
    # Fits on an active curvature row are verified within tau: cut sets that
    # differ between the runs move the optimum by tau-level amounts.
    assert stacked.objective_value == pytest.approx(total, abs=1e-8)


def test_stacked_m_step_is_invariant_to_component_order():
    """The first shape-rejected block must not govern cut discovery."""
    model = _model(BasisSpec(bulk_degree=4), TWO)
    data = _data("points", TWO)
    responsibilities = (
        np.column_stack([TWO < 0.5, TWO >= 0.5]).astype(float) * 0.9 + 0.05
    )
    results = []
    blocks = _mixture_blocks((model,) * 2)
    for resp in (responsibilities, responsibilities[:, ::-1]):
        objective = _mstep_objective((model,) * 2, data, resp)
        result = _solve_basis(objective, blocks=blocks)
        assert result.status == "converged"
        assert result.final_separation.status == "verified"
        results.append(result)
    assert results[0].objective_value == pytest.approx(
        results[1].objective_value, abs=1e-8
    )
    # Weakly identified directions at the shape-active optimum move by more
    # than the tau-level NLL difference suggests.
    for k in range(2):
        np.testing.assert_allclose(
            blocks.part(k, results[0].params),
            blocks.part(1 - k, results[1].params),
            rtol=2e-3,
            atol=2e-3,
        )


def _two_normal_nll(p, lo, hi):
    m1, m2, ls1, ls2, a = p
    w = 1.0 / (1.0 + math.exp(-a))

    def logp(m, ls):
        s = math.exp(ls)
        if lo is hi:
            return -0.5 * ((lo - m) / s) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        a_, b_ = (lo - m) / s, (hi - m) / s
        # Both cancellation-free forms, each used where it is finite.
        with np.errstate(divide="ignore"):
            up = log_ndtr(-a_) + np.log1p(-np.exp(log_ndtr(-b_) - log_ndtr(-a_)))
            down = log_ndtr(b_) + np.log1p(-np.exp(log_ndtr(a_) - log_ndtr(b_)))
        return np.where(a_ > 0, up, down)

    terms = np.c_[math.log(w) + logp(m1, ls1), math.log(1 - w) + logp(m2, ls2)]
    return -float(np.mean(logsumexp(terms, axis=1)))


@pytest.mark.parametrize("kind", ["points", "binned"])
def test_two_normal_fit_is_the_exact_mixture_mle(kind):
    if kind == "points":
        samples, lo, hi = TWO, TWO, TWO
    else:
        e = np.floor(TWO * 2.0) / 2.0
        samples, lo, hi = np.column_stack([e, e + 0.5]), e, e + 0.5
    fit, _, _ = _fit_mixture_auto(
        BasisSpec(bulk_degree=2, expansion="minimal"), samples, 2
    )
    assert fit.status == "converged"
    ref = minimize(
        _two_normal_nll,
        [-2.0, 3.0, 0.0, 0.4, 0.0],
        args=(lo, hi),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 40000, "maxfev": 80000},
    )
    ref = minimize(_two_normal_nll, ref.x, args=(lo, hi), method="BFGS")
    assert fit.log_likelihood == pytest.approx(-ref.fun, abs=1e-9)
    assert fit.responsibilities.shape == (TWO.size, 2)


G1_MIXTURES = [
    ("three normals", (-INF, INF), 3, 2, False, False),
    ("two normals censored", (-INF, INF), 2, 2, False, False),
    ("two gammas shared log", (0.0, INF), 2, 2, True, False),
    ("two betas shared logs", (0.0, 1.0), 2, 2, True, True),
]


# ---------------------------------------------------------------------------
# Selection inside a mixture (C5) and the component count (C6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["points", "intervals"])
def test_one_component_joint_diagnostic_is_the_single_fit_diagnostic(kind):
    from gibbus._fit.basis_selection import (
        _face_of,
        _IntervalData,
        _PointData,
        _span_diagnostic,
        _warm_fit,
    )
    from gibbus._fit.conic_newton import _NewtonOptions
    from gibbus._fit.degree import _DegreeSelectionConfig
    from gibbus._fit.inputs import _normalize_basis_inputs
    from gibbus._model.basis import _full_basis

    config = _DegreeSelectionConfig()
    x = np.random.default_rng(3).gumbel(0.0, 1.0, 400)
    spec = BasisSpec(
        bulk_degree=2,
        lower_tail=Exponential(0.7),
        upper_tail=Exponential(0.7),
        expansion="minimal",
    )
    if kind == "points":
        samples, single = x, _PointData(x, None, 400.0)
    else:
        e = np.floor(x * 2.0) / 2.0
        samples = np.column_stack([e, e + 0.5])
        single = _IntervalData((-INF, INF), samples, None, 400.0)
    model = single.build(spec)
    result = _warm_fit(model, single, _NewtonOptions(), None)
    expected = _span_diagnostic(
        model,
        result.params,
        _full_basis(model),
        single,
        config,
        "joint",
        face=_face_of(model, result),
    )
    fit, _, _ = _fit_mixture_auto(spec, samples, 1)
    assert fit.log_likelihood == pytest.approx(-result.objective_value, abs=1e-12)
    data, _, _ = _mixture_data(_normalize_basis_inputs(spec, samples, None))
    got = _mixture_span_diagnostic(
        fit, 0, _full_basis(fit.models[0]), data, config, "joint"
    )
    assert got.rank == expected.rank
    tolerance = 1e-8 if kind == "points" else 1e-5
    assert got.score == pytest.approx(expected.score, rel=tolerance)


def test_one_component_automatic_fit_is_the_single_fit():
    from gibbus._fit.basis_selection import _fit_basis

    x = np.random.default_rng(4).gumbel(0.0, 1.0, 300)
    spec = BasisSpec(bulk_degree="auto")
    fit, steps, _ = _fit_mixture_auto(spec, x, 1)
    single = _fit_basis(spec, x)
    assert fit.log_likelihood == -single.result.objective_value
    assert [(s.kind, s.chosen, s.diagnostic.p_value) for s in steps] == [
        (s.kind, s.chosen, s.diagnostic.p_value) for s in single.history
    ]


AUTO_MIXTURES = [
    ("two normals", lambda r: np.r_[r.normal(-2, 1, 300), r.normal(3, 1.5, 300)]),
    ("gumbel and normal", lambda r: np.r_[r.gumbel(-3, 1, 300), r.normal(3, 1, 300)]),
]


def test_shared_log_is_kept_or_dropped_by_its_likelihood_ratio():
    from gibbus._defaults import BOUNDARY_ALPHA

    spec = BasisSpec(
        support=(0.0, INF), bulk_degree=2, log_lower="auto", expansion="minimal"
    )
    fit, _, p_values = _fit_mixture_auto(spec, GAMMAS, 2)
    kept = fit.models[0].request.log_lower
    assert math.isfinite(p_values[0]) and math.isnan(p_values[1])
    assert kept == (p_values[0] < BOUNDARY_ALPHA)
    assert all(m.request.log_lower == kept for m in fit.models)


def test_every_kept_mixture_extension_passed_its_likelihood_ratio():
    from gibbus._fit.degree import _DegreeSelectionConfig

    x = np.random.default_rng(1).gumbel(0.0, 1.0, 600)
    x = np.r_[x - 4.0, -x[:300] + 4.0]
    fit, steps, _ = _fit_mixture_auto(BasisSpec(bulk_degree=2), x, 2)
    alpha = _DegreeSelectionConfig().alpha
    expansions = [s for s in steps if s.kind == "expansion"]
    assert expansions and expansions[-1].chosen is None
    for step in expansions[:-1]:
        assert step.chosen is not None and step.lr_p_value < alpha
    assert fit.status == "converged"
