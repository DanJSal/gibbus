"""G3 Stage 4a: constraint rows, their 1x1-block cone description, cold start.

Defining properties: row bookkeeping (unit norm, duplicate directions merged,
invalid rows rejected); structural rows per support/log/tail category; the G1
interior-point solver on the description agrees with exact solutions (single
half-space projection in closed form, exhaustive active sets on small
problems) and returns strictly interior points; the cold start is strictly
interior, proper and converged along its ray (G1's predicted-decrease
criterion).
"""

import itertools
import math
import warnings

import numpy as np
import pytest

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._fit.basis_constraints import (
    _column_scales,
    _ConstraintRows,
    _interior_direction,
    _interior_start,
    _linear_representation,
    _structural_rows,
    _unit_blocks,
)
from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
from gibbus._fit.conic_newton import _preconditioned_subproblem
from gibbus._model.basis import _build_basis
from gibbus._model.basis_normalizer import _NormalizerContext

RNG = np.random.default_rng(41)
NORMAL = RNG.normal(1.5, 2.0, 400)
HALF = RNG.gamma(2.0, 1.5, 400) + 0.2
BOUNDED = RNG.beta(2.0, 3.0, 400) * 4.0 - 1.0
LAPLACE = RNG.laplace(0.0, 1.0, 400)

CASES = {
    "line-b2": (BasisSpec(bulk_degree=2), NORMAL, 0, 1),
    "line-b4": (BasisSpec(bulk_degree=4), NORMAL, 0, 1),
    "line-b3-odd": (BasisSpec(bulk_degree=3), NORMAL, 0, 1),
    "half-log-b2": (
        BasisSpec(support=(0.0, math.inf), bulk_degree=2, log_lower=True),
        HALF,
        1,
        1,
    ),
    "half-log-b1": (
        BasisSpec(support=(0.0, math.inf), bulk_degree=1, log_lower=True),
        HALF,
        1,
        1,
    ),
    "bounded-logs": (
        BasisSpec(support=(-1.0, 3.0), bulk_degree=3, log_lower=True, log_upper=True),
        BOUNDED,
        2,
        0,
    ),
    "laplace-exp": (
        BasisSpec(
            bulk_degree=1, lower_tail=Exponential(1.0), upper_tail=Exponential(1.0)
        ),
        LAPLACE,
        0,
        1,
    ),
    "softpower": (
        BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.5), upper_tail=SoftPower(1.5)),
        LAPLACE,
        0,
        1,
    ),
    "mixed": (
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(1.5)),
        NORMAL,
        0,
        2,
    ),
    "exp-poly3": (
        BasisSpec(bulk_degree=3, lower_tail=Exponential(0.7), upper_tail=Polynomial(3)),
        NORMAL,
        0,
        2,
    ),
}


def _model(spec, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, data)


def _empty(d):
    return _ConstraintRows(np.zeros((0, d)), ())


def _qp(rows, hessian, gradient, theta):
    result, endpoint, _, _ = _preconditioned_subproblem(
        hessian, gradient, theta, _linear_representation(rows), _unit_blocks(rows)
    )
    assert result.converged
    return endpoint


def test_rows_are_unit_norm_and_duplicates_merge():
    rows = _empty(3).merged(np.array([[3.0, 0.0, 4.0], [0.0, 2.0, 0.0]]), "cut")
    np.testing.assert_allclose(np.linalg.norm(rows.rows, axis=1), 1.0)
    again = rows.merged(np.array([6.0, 0.0, 8.0]), "tail")
    assert again.size == 2 and again.kinds == ("cut", "cut")
    opposite = rows.merged(np.array([-3.0, 0.0, -4.0]), "cut")
    assert opposite.size == 3
    with pytest.raises(RuntimeError):
        rows.merged(np.zeros(3), "cut")
    with pytest.raises(RuntimeError):
        rows.merged(np.array([np.inf, 0.0, 0.0]), "cut")
    with pytest.raises(ValueError):
        rows.merged(np.ones(3), "bogus")
    with pytest.raises(ValueError):
        _linear_representation(_empty(3))


@pytest.mark.parametrize("name", list(CASES))
def test_structural_rows_per_category(name):
    spec, data, n_log, n_growth = CASES[name]
    model = _model(spec, data)
    rows = _structural_rows(model)
    assert rows.kinds.count("log") == n_log
    assert rows.kinds.count("growth") == n_growth
    for i in np.flatnonzero([k == "log" for k in rows.kinds]):
        assert model.columns[int(np.argmax(rows.rows[i]))].nonnegative


def test_single_half_space_projection_is_exact():
    rng = np.random.default_rng(7)
    for _ in range(50):
        d = int(rng.integers(2, 9))
        a = rng.normal(size=d)
        p = rng.normal(size=d)
        rows = _empty(d).merged(a, "cut")
        unit = a / np.linalg.norm(a)
        expected = p - min(0.0, float(unit @ p)) * unit
        got = _qp(rows, np.eye(d), np.zeros(d), p)
        # The solver certifies its objective to the duality gap (1e-12
        # relative); for a unit metric the point error is at most its root.
        value = 0.5 * float((got - p) @ (got - p))
        exact = 0.5 * float((expected - p) @ (expected - p))
        assert float(unit @ got) >= -1e-12
        assert value - exact <= 1e-11 * max(1.0, exact)
        assert np.linalg.norm(got - expected) <= np.sqrt(2e-11 * max(1.0, exact))


def test_small_problems_match_exhaustive_active_sets():
    rng = np.random.default_rng(8)
    for _ in range(60):
        d = int(rng.integers(2, 7))
        r = int(rng.integers(1, d + 2))
        a = rng.normal(size=(r, d))
        inside = rng.normal(size=d)
        a[(a @ inside) < 0.0] *= -1.0
        rows = _empty(d).merged(a, "cut")
        m = rng.normal(size=(d, d))
        h = m @ m.T + 1e-2 * np.eye(d)
        g, t0 = rng.normal(size=d), rng.normal(size=d)
        got = _qp(rows, h, g, t0)

        def model(y, g=g, h=h, t0=t0):
            return float(g @ (y - t0) + 0.5 * (y - t0) @ h @ (y - t0))

        best = math.inf
        b = rows.rows
        for k in range(rows.size + 1):
            for act in itertools.combinations(range(rows.size), k):
                ba = b[list(act)]
                kkt = np.block([[h, -ba.T], [ba, np.zeros((k, k))]])
                rhs = np.concatenate([h @ t0 - g, np.zeros(k)])
                sol = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
                y, lam = sol[:d], sol[d:]
                if np.all(b @ y >= -1e-10) and np.all(lam >= -1e-10):
                    best = min(best, model(y))
        assert np.min(rows.slack(got)) >= -1e-12
        assert model(got) <= best + 1e-9 * max(1.0, abs(best))


@pytest.mark.parametrize("name", list(CASES))
def test_interior_direction_is_strictly_interior(name):
    spec, data, _, _ = CASES[name]
    rows = _structural_rows(_model(spec, data))
    w = _interior_direction(rows)
    assert np.all(rows.slack(w) > 1e-6)


@pytest.mark.parametrize("name", list(CASES))
def test_cold_start_is_interior_proper_and_ray_stationary(name):
    spec, data, _, _ = CASES[name]
    model = _model(spec, data)
    rows = _structural_rows(model)
    objective = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )
    start = _interior_start(objective, rows)
    assert start.evaluation.state.status == "proper"
    assert np.all(rows.slack(start.theta) > 0.0)
    w = start.direction
    a = start.scale
    d1 = a * float(start.evaluation.gradient @ w)
    d2 = d1 + a * a * float(w @ start.evaluation.hessian @ w)
    assert d2 > 0.0
    assert 0.5 * d1 * d1 / d2 <= 1e-12 * max(1.0, abs(start.evaluation.nll))
    assert start.evaluation.nll <= objective(w).nll + 1e-12


def test_bounded_support_without_logs_starts_uniform():
    spec = BasisSpec(support=(-1.0, 3.0), bulk_degree=2)
    model = _model(spec, BOUNDED)
    rows = _structural_rows(model)
    assert rows.size == 0
    objective = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, BOUNDED)
    )
    start = _interior_start(objective, rows)
    assert not np.any(start.theta)
    assert start.evaluation.nll == pytest.approx(math.log(4.0), rel=1e-12)


def test_far_tail_cut_is_not_a_duplicate_of_its_dominant_column():
    # A far cut row is e_exp plus polynomial entries ~1e-17 of it: different
    # constraint, though its direction agrees with e_exp to 1e-33.
    d = 4
    rows = _empty(d).merged(np.eye(d)[3], "growth")
    far = np.array([0.0, 0.4856, -44.894, 9.2588e17])
    merged = rows.merged(far, "cut")
    assert merged.size == 2
    assert merged.rows[1][2] != 0.0
    again = merged.merged(3.0 * far, "cut")
    assert again.size == 2


def test_column_scales_leave_solutions_unchanged_and_flatten_rows():
    rng = np.random.default_rng(11)
    for _ in range(30):
        d = int(rng.integers(2, 7))
        # Columns badly scaled consistently in rows and Hessian, as a basis
        # column that is large in its cut rows is large in the Fisher matrix.
        s = 10.0 ** rng.uniform(-8, 8, size=d)
        a = rng.normal(size=(d + 2, d)) * s
        inside = rng.normal(size=d) / s
        a[(a @ inside) < 0.0] *= -1.0
        rows = _empty(d).merged(a, "cut")
        m = rng.normal(size=(d, d))
        h = s[:, None] * (m @ m.T + np.eye(d)) * s[None, :]
        g, t0 = s * rng.normal(size=d), rng.normal(size=d) / s
        scales = _column_scales(rows, h)
        assert np.all(scales > 0.0) and np.all(np.isfinite(scales))
        rep = _linear_representation(rows)
        r1, _, _, v1 = _preconditioned_subproblem(h, g, t0, rep, _unit_blocks(rows))
        r2, _, _, v2 = _preconditioned_subproblem(
            h, g, t0, rep, _unit_blocks(rows), column_scale=scales
        )
        assert r1.converged and r2.converged
        assert v2 == pytest.approx(v1, rel=1e-8, abs=1e-10)
    # The far-tail example: flattening scales separate the exponential column.
    rows = _empty(4).merged(np.eye(4)[3], "growth").merged(np.eye(4)[2], "growth")
    rows = rows.merged(np.array([0.0, 0.4856, -44.894, 9.2588e17]), "cut")
    scales = _column_scales(rows, np.eye(4))
    assert scales[3] < 1e-6 * scales[2]


def test_nearly_parallel_rows_do_not_spoil_a_certified_endpoint():
    """Regression (G3-050): cuts restricted to a leader face are nearly parallel.

    Rows differing by 1e-14 gave ``B`` a singular value the range complement
    treated as null but ``B^+`` inverted: the endpoint reconstruction turned
    rounding into a large move, and "certified" endpoints were 10-30% worse
    than the unconstrained optimum, which is strictly feasible here.
    """
    checked = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        lead = np.array([0.0, 1.0, 1.0, 0.0, 0.0]) / math.sqrt(2.0)
        tiny = rng.uniform(0.5, 2.0, size=2)
        rows = np.array(
            [
                rng.normal(size=5),
                lead + tiny[0] * np.array([0.0, 0.0, 0.0, -1e-14, -3e-15]),
                lead + tiny[1] * np.array([0.0, 0.0, 0.0, -3e-15, -4e-16]),
                np.array([0.0, 1.0, -1.0, 0.0, 0.0]),
                lead,
            ]
        )
        rows /= np.linalg.norm(rows, axis=1)[:, None]
        q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
        h = q @ np.diag(10.0 ** rng.uniform(-4.0, 1.2, 5)) @ q.T
        phi = rng.normal(size=5)
        phi += (0.5 - min(0.0, float(np.min(rows @ phi)))) * np.linalg.lstsq(
            rows, np.ones(5), rcond=None
        )[0]
        step = 0.01 * rng.normal(size=5)
        if np.min(rows @ (phi + step)) <= 0.0:
            continue
        g = -h @ step
        optimum = float(g @ step + 0.5 * step @ h @ step)
        cone = _ConstraintRows(rows, ("cut",) * 5)
        blocks = tuple(np.array([[float(v)]]) for v in rows @ phi)
        result, _, _, value = _preconditioned_subproblem(
            h, g, phi, _linear_representation(cone), blocks
        )
        if result.converged:
            checked += 1
            assert value <= optimum + 1e-9 * abs(optimum), seed
    assert checked >= 20


def test_interior_direction_with_cuts_is_strictly_interior():
    rng = np.random.default_rng(12)
    for _ in range(30):
        d = int(rng.integers(2, 7))
        inside = rng.normal(size=d)
        a = rng.normal(size=(3 * d, d))
        a[(a @ inside) < 0.0] *= -1.0
        rows = _empty(d).merged(a, "cut")
        w = _interior_direction(rows)
        assert np.min(rows.slack(w)) > 0.0
