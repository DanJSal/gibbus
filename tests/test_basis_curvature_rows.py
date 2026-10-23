"""G3 scalar curvature inequalities and globally convex Newton iterates."""

import math
import warnings

import numpy as np
from gibbus._fit._basis_newton_kernels import ACCEPTED, SHAPE, STOP

from gibbus import BasisSpec, GibbusBasisWarning
from gibbus._fit import basis_newton
from gibbus._fit.basis_constraints import (
    _ConstraintRows,
    _linear_representation,
)
from gibbus._fit.basis_newton import (
    _Blocks,
    _FaceObjective,
    _JointCurvature,
    _solve_basis,
)
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._fit.curvature_search import (
    _CurvatureContext,
    _CurvatureOptions,
    _separate_curvature,
)
from gibbus._model.basis import _build_basis
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_normalizer import _NormalizerContext

NORMAL = np.random.default_rng(0).normal(1.5, 2.0, 400)


def _model(spec, x=NORMAL):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, x)


def _objective(model, x=NORMAL):
    return _BasisPointObjective(_NormalizerContext(model), _point_statistics(model, x))


def _theta_for(model, curvature):
    """Coefficients with ``q''(z) = (z**2 - 1)**2 + eps`` on a quartic bulk.

    ``q`` is a polynomial in the fitting coordinate ``z`` with
    ``q'' = z**4 - 2 z**2 + 1 + eps``; returned in the model's columns.
    """
    labels = [c.label for c in model.columns]
    h = model.geometry.scale
    c = model.geometry.center
    # Solve for the coefficients by least squares on q'' samples (exact:
    # q'' lies in the span of the columns' second derivatives).
    x = c + h * np.linspace(-3.0, 3.0, 41)
    z = (x - c) / h
    target = (z**4 - 2.0 * z**2 + 1.0 + curvature) / h**2
    d2 = _evaluate_basis(model, x).d2phi
    theta, *_ = np.linalg.lstsq(d2, target, rcond=None)
    assert len(labels) == theta.size
    return theta


QUARTIC = BasisSpec(bulk_degree=6, expansion="minimal")


def test_newton_constraints_are_scalar_curvature_rows():
    model = _model(QUARTIC)
    d = model.n_coefficients
    rng = np.random.default_rng(1)
    rows = _ConstraintRows(np.zeros((0, d)), ()).merged(rng.normal(size=(3, d)), "cut")
    rep = _linear_representation(rows)
    assert [m.shape[1] for m in rep.row_matrices] == [1, 1, 1]
    for block in rep.row_matrices:
        slack = -np.tensordot(rep.reference_dual, block, axes=(0, 0))
        assert float(slack[0, 0]) > 0.0
    # Feasible row slacks are the only Newton cone inequalities.
    interior = rows.rows.T @ np.ones(rows.size)
    if np.all(rows.slack(interior) > 0.0):
        blocks = tuple(np.array([[v]]) for v in rows.slack(interior))
        assert rep.residual(interior, blocks) < 1e-13


def test_search_reports_each_minimum_once():
    model = _model(QUARTIC)
    theta = _theta_for(model, -1e-4)  # two shallow violations at z = +-1
    res = _separate_curvature(
        _CurvatureContext(model), theta, _CurvatureOptions(rho=1.0)
    )
    interior = [m for m in res.minima if m.kind == "interior" and m.value < 0.0]
    z = sorted((m.x - model.geometry.center) / model.geometry.scale for m in interior)
    np.testing.assert_allclose(z, [-1.0, 1.0], atol=1e-8)
    assert res.status == "violated"


def test_rows_with_tiny_entries_merge():
    rows = _ConstraintRows(np.zeros((0, 3)), ())
    out = rows.merged(1e-170 * np.array([1.0, 2.0, 2.0]), "cut")
    np.testing.assert_allclose(out.rows[0], [1 / 3, 2 / 3, 2 / 3], rtol=1e-15)


def test_gate_queues_cuts_and_ends_the_run_after_a_convex_point():
    model = _model(QUARTIC)
    d = model.n_coefficients
    curvature = _CurvatureContext(model)
    joint = _JointCurvature(_Blocks.single(model))
    obj = _FaceObjective(_objective(model), np.eye(d), joint, None).compiled()
    convex = _theta_for(model, 1.0)
    concave = _theta_for(model, -1e-2)
    domains = (_separate_curvature(curvature, convex).domain,)
    obj.begin(np.zeros((0, d)), domains)
    assert obj.trial(concave) == (SHAPE, None)
    block, rows, n_points, theta, _ = obj.pending()
    assert block == 0
    assert n_points > 0 and np.all(rows[:n_points] @ concave < 0.0)
    np.testing.assert_array_equal(theta, concave)
    code, nll = obj.trial(convex)  # a convex trial is evaluated ...
    assert code == ACCEPTED and math.isfinite(nll)
    assert obj.trial(convex)[0] == STOP  # ... and the next one ends the run
    obj.end()
    # A trial adding no new row queues nothing.
    known = _ConstraintRows(np.zeros((0, d)), ()).merged(rows, "cut")
    obj.begin(known.rows, domains)
    assert obj.trial(concave) == (SHAPE, None)
    assert obj.pending() is None
    obj.end()


def test_newton_runs_evaluate_only_convex_points():
    # Normal data at bulk degree 4 (seed 0): the MLE touches q'' = 0 at
    # x = -116, far outside the data; before G3-038 76% of the evaluations
    # were non-convex and 17 of 99 exhausted the normalizer (E-G3-016).
    model = _model(BasisSpec(bulk_degree=4, expansion="minimal"))
    objective = _objective(model)
    curvature = _CurvatureContext(model)
    basis_newton._TRACE = seen = []
    try:
        result = _solve_basis(objective)
    finally:
        basis_newton._TRACE = None
    assert result.status == "converged"
    assert result.final_separation.status == "nonnegative"
    assert seen
    for theta in seen:
        assert _separate_curvature(curvature, theta).status == "nonnegative"
    assert result.objective_evaluations <= 40
    assert math.isfinite(result.objective_value)


def test_newton_only_constructs_scalar_conic_blocks(monkeypatch):
    """Full fitting uses linear curvature cuts, never 2-by-2 contact blocks."""
    original = basis_newton._linear_representation
    observed = []

    def checked(rows):
        representation = original(rows)
        sizes = [m.shape[1] for m in representation.row_matrices]
        assert all(size == 1 for size in sizes)
        observed.append(len(sizes))
        return representation

    monkeypatch.setattr(basis_newton, "_linear_representation", checked)
    objective = _objective(_model(BasisSpec(bulk_degree=4, expansion="minimal")))
    result = _solve_basis(objective)
    assert result.status in {"converged", "converged_approximately"}
    assert result.final_separation.status == "nonnegative"
    assert observed and min(observed) > 0
