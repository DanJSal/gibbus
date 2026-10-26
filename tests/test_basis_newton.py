"""Newton fit over constraint rows, cut exchange and faces.

Defining properties: the Gaussian MLE in closed form; first-order KKT
conditions over the retained rows (gradient a nonnegative combination of the
near-active rows); interior, proper and shape-nonnegative results; exact
faces where a leader or amplitude is optimal at zero; the improper-boundary
logic behind the improper-boundary error (for point data an improper face
has ``log Z = inf``, so the error is reachable only with interval data); the
trial-failure ledger.
"""

import math
import warnings

import numpy as np
import pytest
from gibbus._fit._basis_newton_kernels import FAILURE
from scipy.optimize import nnls

from gibbus import (
    BasisSpec,
    Exponential,
    GibbusBasisWarning,
    Polynomial,
    SoftPower,
    clear_suppressed_failures,
    suppressed_failures,
)
from gibbus._fit.basis_constraints import _column_scales, _interior_start
from gibbus._fit.basis_newton import (
    _Blocks,
    _face_growth,
    _FaceObjective,
    _JointCurvature,
    _solve_basis,
)
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._model.basis import _build_basis
from gibbus._model.basis_normalizer import _NormalizerContext, _NormalizerOptions

RNG = np.random.default_rng(2027)
NORMAL = RNG.normal(1.5, 2.0, 400)
GUMBEL = RNG.gumbel(0.0, 1.0, 400)
HALF = RNG.gamma(2.0, 1.5, 400) + 0.2
BOUNDED = RNG.beta(2.0, 3.0, 400) * 4.0 - 1.0
LAPLACE = RNG.laplace(0.0, 1.0, 400)
STUDENT = RNG.standard_t(5, 400)
CAUCHY = RNG.standard_cauchy(400)
INF = math.inf

# (support, bulk degree, log lower, log upper, data)
POLY = {
    "line-b2": ((-INF, INF), 2, False, False, NORMAL),
    "line-b4-gumbel": ((-INF, INF), 4, False, False, GUMBEL),
    "half-b2-log": ((0.0, INF), 2, True, False, HALF),
    "half-b3-nolog": ((0.0, INF), 3, False, False, HALF),
    "bounded-b3-logs": ((-1.0, 3.0), 3, True, True, BOUNDED),
    "bounded-b2-nolog": ((-1.0, 3.0), 2, False, False, BOUNDED),
    "bounded-b4-lower": ((-1.0, 3.0), 4, True, False, BOUNDED),
}

TAILS = {
    "student-softpower": (
        BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.5), upper_tail=SoftPower(1.5)),
        STUDENT,
    ),
    "half-exponential-tail": (
        BasisSpec(
            support=(0.0, INF),
            bulk_degree=2,
            log_lower=True,
            upper_tail=Exponential(0.5),
        ),
        HALF,
    ),
    "line-odd-b3": (BasisSpec(bulk_degree=3), NORMAL),
    # Convexity contact ~30 h out in the exponential tail: needs the
    # entrywise row identity and row-flattening column scales.
    "normal-exp-cubic": (
        BasisSpec(bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Polynomial(3)),
        NORMAL,
    ),
}


def _objective(spec, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        model = _build_basis(spec, data)
    return _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )


def _poly(name):
    support, k, lo, up, data = POLY[name]
    spec = BasisSpec(support=support, bulk_degree=k, log_lower=lo, log_upper=up)
    return _objective(spec, data), (support, k, lo, up, data)


def _assert_kkt(result):
    """Gradient = sum lambda_j a_j over near-active rows, lambda >= 0.

    Measured in the solver's row-flattening column scales: rows whose entries
    span many orders of magnitude (far exponential-tail cuts) make an
    unscaled residual meaningless.
    """
    theta, g = result.params, result.evaluation.gradient
    rows = result.rows.rows
    face_eq = result.face.equalities
    slack = rows @ theta
    active = slack <= 1e-7 * max(1.0, float(np.linalg.norm(theta)))
    scale = _column_scales(result.rows, result.evaluation.hessian)
    # Equalities of the face are free-sign constraints.
    design = scale[:, None] * np.vstack([rows[active], face_eq, -face_eq]).T
    gs = scale * g
    if design.size == 0:
        residual = float(np.linalg.norm(gs))
    else:
        design = design / np.maximum(np.linalg.norm(design, axis=0), 1e-300)
        _, residual = nnls(design, gs, maxiter=10_000)
    # Convergence certifies a predicted decrease of 1e-12: the
    # stationarity residual may be up to sqrt(2e-12 * lambda_max(H)).
    hs = scale[:, None] * result.evaluation.hessian * scale[None, :]
    lam = max(1.0, float(np.max(np.linalg.eigvalsh(hs))))
    nll = max(1.0, abs(result.objective_value))
    assert residual <= 10.0 * np.sqrt(2e-12 * nll * lam)


def test_gaussian_mle_in_closed_form():
    objective, _ = _poly("line-b2")
    result = _solve_basis(objective)
    model = objective.model
    z = model.geometry.to_internal(NORMAL)
    labels = [c.label for c in model.columns]
    m, v = z.mean(), z.var()
    assert result.status == "converged"
    assert result.final_separation.status == "verified"
    # A predicted-decrease certificate of 1e-12 bounds the parameters to ~1e-6.
    assert result.params[labels.index("z^2")] == pytest.approx(1 / (2 * v), rel=1e-5)
    assert result.params[labels.index("z")] == pytest.approx(-m / v, rel=1e-5)


@pytest.mark.parametrize("name", list(POLY))
def test_polynomial_family_fits_are_kkt_interior_and_convex(name):
    objective, _ = _poly(name)
    result = _solve_basis(objective)
    assert result.status == "converged"
    assert result.final_separation.status == "verified"
    assert result.evaluation.state.status == "proper"
    assert np.min(result.rows.slack(result.params)) >= -1e-12
    start = _interior_start(objective, result.rows)
    assert result.objective_value <= start.evaluation.nll + 1e-12
    _assert_kkt(result)


def test_gamma_like_data_fit_on_the_exact_zero_quadratic_face():
    objective, _ = _poly("half-b2-log")
    result = _solve_basis(objective)
    assert result.face.suppressed == ((0, "upper"),)
    labels = [c.label for c in objective.model.columns]
    assert abs(result.params[labels.index("z^2")]) <= 1e-14
    assert result.params[labels.index("z")] > 0.0


def test_laplace_data_suppress_both_exponential_leaders():
    spec = BasisSpec(
        bulk_degree=2, lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)
    )
    result = _solve_basis(_objective(spec, LAPLACE))
    assert result.status == "converged"
    assert result.final_separation.status == "verified"
    assert set(result.face.suppressed) == {(0, "lower"), (0, "upper")}


@pytest.mark.parametrize("name", list(TAILS))
def test_tail_classes_converge_convex(name):
    spec, data = TAILS[name]
    result = _solve_basis(_objective(spec, data))
    assert result.status == "converged"
    assert result.final_separation.status == "verified"
    _assert_kkt(result)


def test_heavy_tails_give_a_proper_fit_with_a_tiny_leader():
    # Cauchy outliers make the exponential statistic ~1e34: the MLE has a tiny
    # but positive leader (a proper density), not an improper limit.
    spec = BasisSpec(
        bulk_degree=1, lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)
    )
    result = _solve_basis(_objective(spec, CAUCHY))
    assert result.status in ("converged", "converged_approximately")
    assert result.evaluation.state.status == "proper"


def test_face_growth_detects_faces_without_a_proper_member():
    # Full line, bulk 4: suppressing x^4 forces x^3 = 0 (opposite outward
    # signs at the two ends) and leaves x^2 as the growth row; suppressing
    # x^2 as well leaves only x, which cannot grow outward at both ends.
    objective, _ = _poly("line-b4-gumbel")
    model = objective.model
    labels = [c.label for c in model.columns]
    unit = np.eye(model.n_coefficients)
    eq, growth = _face_growth(_Blocks.single(model), unit[[labels.index("z^4")]])
    assert growth.shape[0] == 1
    assert np.argmax(np.abs(growth[0])) == labels.index("z^2")
    assert any(np.argmax(np.abs(r)) == labels.index("z^3") for r in eq)
    assert (
        _face_growth(
            _Blocks.single(model), unit[[labels.index("z^4"), labels.index("z^2")]]
        )
        is None
    )


def test_trial_quadrature_failure_is_recorded_not_routine():
    # A two-cell budget cannot resolve a normal potential: the compiled trial
    # is rejected and recorded, and the ledger entry is not routine.
    objective = _objective(BasisSpec(bulk_degree=2), NORMAL)
    good = _solve_basis(objective).params
    model = objective.model
    starved = _BasisPointObjective(
        objective.context,
        objective.stats,
        _NormalizerOptions(initial_cells=2, max_cells=2),
    )
    face = _FaceObjective(
        starved, np.eye(model.n_coefficients), _JointCurvature(_Blocks.single(model))
    )
    compiled = face.compiled()
    assert compiled.trial(good) == (FAILURE, None)
    failures = compiled.failures()
    assert failures[0] == 1 and compiled.failures() is None
    clear_suppressed_failures()
    face._ledger(failures)
    ledger = suppressed_failures()
    assert len(ledger) == 1
    assert ledger[0]["context"] == "Newton trial evaluation"
    assert ledger[0]["type"] == "_QuadratureFailure"
    assert ledger[0]["routine"] is False


@pytest.mark.parametrize("sides", [("lower",), ("upper",), ("lower", "upper")])
def test_a_wrong_leader_face_start_is_released(sides):
    # The cold optimum has both leaders positive; a start on a face that
    # suppresses either (as a warm start inherits) must be released to it
    # Without release the fit stayed 7e-5 to 2.8e-3 nats worse.
    spec = BasisSpec(
        bulk_degree=3,
        lower_tail=Exponential(0.7),
        upper_tail=Polynomial(3),
        expansion="minimal",
    )
    objective = _objective(spec, NORMAL)
    cold = _solve_basis(objective)
    assert cold.status == "converged" and cold.face.suppressed == ()
    warm = _solve_basis(objective, start_face=(tuple((0, s) for s in sides), ()))
    assert warm.status == "converged"
    assert warm.final_separation.status == "verified"
    assert warm.face.suppressed == ()
    scale = max(1.0, abs(cold.objective_value))
    assert abs(warm.objective_value - cold.objective_value) <= 1e-10 * scale


@pytest.mark.parametrize(
    ("spec", "data"),
    [
        (BasisSpec(support=(0.0, INF), bulk_degree=3, expansion="minimal"), HALF),
        (
            BasisSpec(
                bulk_degree=2,
                lower_tail=Exponential(1.0),
                upper_tail=Exponential(1.0),
                expansion="minimal",
            ),
            LAPLACE,
        ),
    ],
)
def test_an_optimal_leader_face_is_kept(spec, data):
    # Where the cold fit itself collapses a leader, a start on that face stays
    # there and reaches the same optimum (the shared Laplace amplitude starts
    # from a projected point without growth: the face's own cold start).
    objective = _objective(spec, data)
    cold = _solve_basis(objective)
    assert cold.status == "converged" and cold.face.suppressed
    warm = _solve_basis(objective, start_face=(cold.face.suppressed, ()))
    assert warm.status == "converged"
    assert warm.face.suppressed == cold.face.suppressed
    scale = max(1.0, abs(cold.objective_value))
    assert abs(warm.objective_value - cold.objective_value) <= 1e-10 * scale


@pytest.mark.parametrize(
    "spec,draw",
    [
        (BasisSpec(bulk_degree=6), lambda r: r.standard_t(5, size=600)),
        (
            BasisSpec(support=(0.0, np.inf), bulk_degree=6, upper_tail=SoftPower(1.5)),
            lambda r: r.gamma(2.0, size=600),
        ),
    ],
)
def test_seeded_cuts_reach_the_same_optimum(spec, draw):
    # A cut is q''(x*) >= 0 at some point: valid for every convex potential,
    # so seeding an earlier solve's cuts changes the exchange path only.
    objective = _objective(spec, draw(np.random.default_rng(7)))
    cold = _solve_basis(objective)
    cuts = cold.rows.rows[[kind == "cut" for kind in cold.rows.kinds]]
    assert len(cuts)
    seeded = _solve_basis(objective, start_cuts=cuts)
    assert cold.status == seeded.status == "converged"
    assert seeded.final_separation.status == "verified"
    assert seeded.objective_value == pytest.approx(cold.objective_value, abs=1e-9)
    assert seeded.exchanges <= cold.exchanges
