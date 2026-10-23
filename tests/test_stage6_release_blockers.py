"""Reproductions for internal row errors and optional face-start failures."""

import numpy as np
import pytest

from gibbus import BasisSpec, Distribution, Exponential, Polynomial, SoftPower, TanhSpec
from gibbus._fit.basis_constraints import _ConstraintRows
from gibbus._model.basis import _build_basis
from tests.g3_casegen import distributions as dist
from tests.g3_casegen.cases import (
    CaseSpec,
    FitSpec,
    ObservationSpec,
    materialize,
)


def _observations(family, n, seed, **params):
    """Return the deterministic Stage 6 sample for a named distribution."""
    spec = CaseSpec(
        dist.Named(family, **params), ObservationSpec("points"), FitSpec(), n
    )
    return materialize(spec, seed=seed).observations


def test_generated_invalid_constraints_raise_runtime_error():
    """An internally generated row is not a bad public argument."""
    rows = _ConstraintRows(np.zeros((0, 2)), ())
    for invalid in ([float("inf"), 1.0], [float("nan"), 1.0], [0.0, 0.0]):
        with pytest.raises(RuntimeError, match="generated constraint rows"):
            rows.merged(np.array([invalid]), "cut")
    with pytest.raises(ValueError, match="unknown constraint kind"):
        rows.merged(np.array([[1.0, 0.0]]), "invalid")


def test_exponential_tail_far_cut_recovers_converged_certificate():
    """A cut past the exponential overflow range must not abort fitting."""
    sample = _observations("gumbel_r", 100, 1, mu=0, scale=1)
    fitted = Distribution().fit(
        sample,
        basis=BasisSpec(
            bulk_degree=8,
            lower_tail=Exponential(1.0),
            upper_tail=Exponential(1.0),
        ),
    )
    assert fitted.fit_diagnostics["converged"]
    assert fitted.fit_diagnostics["components"][0]["shape"] == "nonnegative"


def test_exponential_infinite_end_cut_stays_finite_and_violating():
    """Exponential overflow must not replace a finite cut with infinity.

    This fixed trial has an inactive exponential and a negative degree-seven
    polynomial curvature at the far upper search boundary.  Its raw
    exponential basis column overflows at the original point, but a finite
    inward point remains a necessary, violated curvature constraint.
    """
    from gibbus._fit.curvature_search import _CurvatureContext, _separate_curvature

    sample = _observations("gumbel_r", 100, 1, mu=0, scale=1)
    model = _build_basis(
        BasisSpec(
            bulk_degree=8,
            lower_tail=Exponential(1.0),
            upper_tail=Exponential(1.0),
        ),
        sample,
    )
    theta = np.array(
        [
            2.40398321e-1,
            6.17368949e-1,
            -2.50672072e-1,
            -3.99961370e-2,
            4.45481663e-2,
            4.95151277e-4,
            -2.15244144e-3,
            2.27506273e-16,
            0.0,
        ]
    )
    ctx = _CurvatureContext(model).on_face(np.eye(model.n_coefficients)[:, :-1])
    sep = _separate_curvature(ctx, theta)
    assert sep.status == "violated"
    assert sep.domain[1] > 100_000.0
    assert np.isfinite(sep.cut_points).all()
    assert np.isfinite(sep.cut_rows).all()
    assert np.any((sep.cut_points > 100.0) & (sep.cut_points < sep.domain[1]))
    assert np.min(sep.cut_rows @ theta) < 0.0


def test_face_preserves_cut_lost_beneath_inactive_exponential_column():
    """A far-point inequality must survive numerical projection to a face."""
    from types import SimpleNamespace

    from gibbus._fit.basis_newton import (
        _active_rows,
        _Blocks,
        _Face,
        _merge_face_part,
    )

    sample = np.random.default_rng(7).normal(size=30)
    model = _build_basis(
        BasisSpec(
            bulk_degree=2, lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)
        ),
        sample,
    )
    d = model.n_coefficients
    face = _Face(np.eye(d)[[-1]], (), (), np.zeros((0, d)))
    released = _Face(np.zeros((0, d)), (), (), np.zeros((0, d)))
    # This is a valid point-curvature row at an extreme scale: the active
    # coefficient is 1, but the fixed exponential column is 1e200.
    raw = np.zeros(d)
    raw[1] = 1.0
    raw[-1] = 1e200
    part = SimpleNamespace(cut_points=np.array([400.0]), cut_rows=raw.reshape(1, -1))
    problem = SimpleNamespace(
        rows=_ConstraintRows(np.zeros((0, d)), ()),
        blocks=_Blocks.single(model),
        scoped_tails=[],
        scoped_points=[],
    )
    _merge_face_part(problem, face, 0, part)
    assert len(problem.scoped_points) == 1
    on_face = _active_rows(problem, face, face.basis(d))
    assert np.max(np.abs(on_face.rows[:, 1])) > 0.9
    off_face = _active_rows(problem, released, released.basis(d))
    assert all(abs(row[1]) < 1e-12 for row in off_face.rows)


def test_softpower_candidate_face_start_retains_certified_incumbent():
    """A failed optional face start must not invalidate a converged fit."""
    sample = _observations("generalized_gamma", 100, 0, a=2, c=2, scale=1)
    fitted = Distribution().fit(
        sample,
        basis=BasisSpec(
            support=(0.0, float("inf")),
            bulk_degree=4,
            log_lower="auto",
            log_upper=False,
            upper_tail=SoftPower(1.5),
        ),
    )
    assert fitted.fit_diagnostics["converged"]
    assert all(
        component["shape"] == "nonnegative"
        for component in fitted.fit_diagnostics["components"]
    )


def test_automatic_two_linear_tails_keep_compatible_curvature_gates():
    """Unequal skew-adaptive gate rates forced the gated quadratic to zero.

    On a linear-tail full line the quadratic ``z**2 * (t_L + t_U)`` is
    exponentially small at both ends.  It must have the same leading
    curvature sign there to provide a useful convex bulk direction.
    """
    x = np.random.default_rng(0).laplace(size=800)
    model = _build_basis(
        BasisSpec(
            bulk_degree=2,
            lower_tail=Polynomial(1),
            upper_tail=Polynomial(1),
        ),
        x,
    )
    g = model.geometry
    assert g.lower_steepness == g.upper_steepness
    j = next(i for i, col in enumerate(model.columns) if col.label == "x^2*(tL+tU)")
    for side in model.asymptotics:
        leading_curvature = side.curvature[0]
        assert leading_curvature.order == 2.0
        assert float(leading_curvature.row_exact[j]) > 0.0


def test_two_linear_tails_match_one_explicit_steepness_without_overriding_two():
    """An unspecified gate may follow its explicit partner; two explicit gates stand."""
    x = np.random.default_rng(0).laplace(size=800)
    common = dict(bulk_degree=2, lower_tail=Polynomial(1), upper_tail=Polynomial(1))
    matched = _build_basis(BasisSpec(**common, tanh=TanhSpec(lower_steepness=1.25)), x)
    assert matched.geometry.lower_steepness == matched.geometry.upper_steepness == 1.25
    specified = _build_basis(
        BasisSpec(**common, tanh=TanhSpec(lower_steepness=1.25, upper_steepness=1.4)),
        x,
    )
    assert specified.geometry.lower_steepness == 1.25
    assert specified.geometry.upper_steepness == 1.4


def test_face_tail_cut_does_not_exclude_globally_convex_release():
    """An asymptotic cut after suppressing a leader is only face-valid.

    The potential q(z)=10z²-z³+z⁴ has curvature 20-6z+12z²>0.
    On the face with the quartic coefficient fixed to zero, however, the
    negative cubic coefficient violates upper-tail curvature. Reusing that
    face cut after releasing the quartic would exclude a convex potential.
    """
    from types import SimpleNamespace

    from gibbus._fit.basis_newton import (
        _active_rows,
        _Blocks,
        _Face,
        _merge_face_part,
    )
    from gibbus._fit.curvature_search import _CurvatureContext, _separate_curvature

    sample = np.random.default_rng(7).normal(size=80)
    model = _build_basis(
        BasisSpec(
            bulk_degree=4,
            lower_tail=Polynomial(4),
            upper_tail=Polynomial(4),
            expansion="minimal",
        ),
        sample,
    )
    assert [col.label for col in model.columns] == ["x", "x^2", "x^3", "x^4"]
    d = model.n_coefficients
    full = np.array([0.0, 10.0, -1.0, 1.0])
    face_point = full.copy()
    face_point[-1] = 0.0
    eq = np.eye(d)[[3]]
    face = _Face(eq, (), (), np.zeros((0, d)))
    unrestricted = _Face(np.zeros((0, d)), (), (), np.zeros((0, d)))
    ctx = _CurvatureContext(model)
    certified = _separate_curvature(ctx, full)
    assert certified.status == "nonnegative"
    cut = _separate_curvature(ctx.on_face(face.basis(d)), face_point)
    assert cut.status == "violated"
    tail_rows = cut.cut_rows[cut.cut_points.size :]
    assert tail_rows.shape[0] > 0
    assert np.min(tail_rows @ full) < -1.0

    problem = SimpleNamespace(
        rows=_ConstraintRows(np.zeros((0, d)), ()),
        blocks=_Blocks.single(model),
        scoped_tails=[],
    )
    _merge_face_part(problem, face, 0, cut)
    assert len(problem.scoped_tails) > 0
    # Finite q''(x*) cuts remain global. Only the conditional tail cuts are scoped.
    assert all(kind == "cut" for kind in problem.rows.kinds)
    full_rows = _active_rows(problem, unrestricted, unrestricted.basis(d))
    assert all(kind != "tail" for kind in full_rows.kinds)
    assert np.min(full_rows.slack(full)) >= 0.0
    face_rows = _active_rows(problem, face, face.basis(d))
    assert "tail" in face_rows.kinds
    assert (
        min(
            row @ face.basis(d).T @ face_point
            for row, kind in zip(face_rows.rows, face_rows.kinds, strict=True)
            if kind == "tail"
        )
        < 0.0
    )


def test_conditional_tail_cuts_apply_to_subfaces_not_other_faces():
    """A conditional cut must be enforced exactly when its source holds."""
    from types import SimpleNamespace

    from gibbus._fit.basis_newton import _active_rows, _Face

    d = 4
    source = np.eye(d)[[3]]
    cut = np.array([0.0, 0.0, 1.0, 0.0])
    problem = SimpleNamespace(
        rows=_ConstraintRows(np.zeros((0, d)), ()),
        scoped_tails=[(source, cut)],
    )
    subface = _Face(np.eye(d)[[0, 3]], (), (), np.zeros((0, d)))
    different = _Face(np.eye(d)[[0]], (), (), np.zeros((0, d)))
    assert "tail" in _active_rows(problem, subface, subface.basis(d)).kinds
    assert "tail" not in _active_rows(problem, different, different.basis(d)).kinds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_degree3_two_linear_tails_use_exact_curvature_face(seed):
    """Opposite cubic tail curvatures require a zero cubic coefficient.

    Degree 3 has the same convex span as degree 2 for two matched linear
    tails.  The optimizer must solve on that exact face instead of requiring
    positive slack on two incompatible asymptotic curvature inequalities.
    """
    from gibbus._fit.basis_newton import (
        _Blocks,
        _forced_curvature_equalities,
    )
    from gibbus._fit.basis_selection import _fit_basis

    sample = np.random.default_rng(seed).laplace(size=150)
    common = dict(
        lower_tail=Polynomial(1),
        upper_tail=Polynomial(1),
        expansion="minimal",
    )
    requested = BasisSpec(bulk_degree=3, **common)
    model = _build_basis(requested, sample)
    assert "x^3*(tL+tU)" in [column.label for column in model.columns]
    forced = _forced_curvature_equalities(_Blocks.single(model))
    assert forced.shape == (1, model.n_coefficients)
    assert np.count_nonzero(forced) == 1

    fit = _fit_basis(requested, sample)
    assert fit.result.status == "converged"
    assert fit.result.final_separation.parts[0].status == "nonnegative"
    assert fit.result.face.equalities.shape[0] >= 1
    np.testing.assert_allclose(forced @ fit.result.params, 0.0, atol=1e-12)

    # A higher nominal degree does not silently change the requested tails;
    # the exact curvature equality simply removes a nonconvex direction.
    baseline = Distribution().fit(sample, basis=BasisSpec(bulk_degree=2, **common))
    candidate = Distribution().fit(sample, basis=requested)
    assert candidate.fit_diagnostics["converged"]
    xs = np.linspace(-4.0, 4.0, 25)
    np.testing.assert_allclose(
        candidate.logpdf(xs), baseline.logpdf(xs), rtol=1e-6, atol=1e-6
    )
    rebuilt = Distribution(candidate.data)
    np.testing.assert_allclose(rebuilt.logpdf(xs), candidate.logpdf(xs), atol=1e-10)


def test_curvature_equalities_are_only_applied_when_proven():
    """Even degrees and explicitly unequal rates are not collapsed."""
    from gibbus._fit.basis_newton import (
        _Blocks,
        _forced_curvature_equalities,
        _matching_face,
    )

    sample = np.random.default_rng(0).laplace(size=120)
    common = dict(lower_tail=Polynomial(1), upper_tail=Polynomial(1))
    for spec in (
        BasisSpec(bulk_degree=2, **common),
        BasisSpec(bulk_degree=4, **common),
        BasisSpec(
            bulk_degree=3,
            tanh=TanhSpec(lower_steepness=1.25, upper_steepness=1.4),
            **common,
        ),
    ):
        model = _build_basis(spec, sample)
        assert _forced_curvature_equalities(_Blocks.single(model)).size == 0

    model = _build_basis(BasisSpec(bulk_degree=3, **common), sample)
    blocks = _Blocks.single(model)
    face = _matching_face(blocks, (), ())
    assert face.equalities.shape == (1, model.n_coefficients)
    np.testing.assert_allclose(
        face.equalities @ face.basis(model.n_coefficients),
        0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("expansion", ["minimal", "full", "auto"])
def test_degree3_linear_tails_remain_certified_across_expansions(expansion):
    """The forced curvature face is independent of the span policy."""
    sample = np.random.default_rng(4).laplace(size=120)
    fitted = Distribution().fit(
        sample,
        basis=BasisSpec(
            bulk_degree=3,
            lower_tail=Polynomial(1),
            upper_tail=Polynomial(1),
            expansion=expansion,
        ),
    )
    assert fitted.fit_diagnostics["converged"]
    assert all(
        part["shape"] == "nonnegative" for part in fitted.fit_diagnostics["components"]
    )
    basis = fitted.components[0]._basis
    cubic = next(
        index
        for index, column in enumerate(basis.model.columns)
        if column.label == "x^3*(tL+tU)"
    )
    assert basis.theta[cubic] == 0.0


def test_degree3_linear_tails_certify_interval_and_mixture_fits():
    """Both observation paths retain the exact face in joint fitting."""
    sample = np.random.default_rng(42).laplace(size=100)
    intervals = np.floor(sample * 4.0) / 4.0
    intervals = np.column_stack((intervals, intervals + 0.25))
    spec = BasisSpec(
        bulk_degree=3,
        lower_tail=Polynomial(1),
        upper_tail=Polynomial(1),
        expansion="minimal",
    )
    cases = ((intervals, 1), (np.r_[sample - 3, sample + 3], 2))
    for observations, components in cases:
        fitted = Distribution().fit(
            observations, basis=spec, n_components=components, rng=0
        )
        assert fitted.fit_diagnostics["converged"]
        assert all(
            part["shape"] == "nonnegative"
            for part in fitted.fit_diagnostics["components"]
        )
