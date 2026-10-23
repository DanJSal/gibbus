"""G3 Stage 2a: outward-rounded interval enclosures of basis derivatives.

Point enclosures must contain the 150-digit mpmath value of every column and
derivative (orders 0-4) at bulk and far-tail points; cell enclosures must
contain dense float samples; and enclosures must shrink with the cell.
"""

import math
import warnings
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from test_basis_evaluation import (
    ORACLE_SPECS,
    _broad_battery,
    _compile,
    _oracle_column,
    _test_points,
)

from gibbus import BasisSpec, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._model.basis_eval import _evaluate_basis
from gibbus._model.basis_interval import _basis_enclosure, _potential_enclosure

RNG = np.random.default_rng(20261016)
mp.mp.dps = 150


@pytest.mark.parametrize("spec", ORACLE_SPECS, ids=lambda s: "")
def test_point_enclosures_contain_high_precision_values(spec):
    model = _compile(spec)
    bulk, tail = _test_points(model)
    points = np.concatenate([bulk, tail])
    enc = _basis_enclosure(model, points, points, range(5))
    for col, column in enumerate(model.columns):
        oracle = _oracle_column(model, column)
        for k in range(5):
            lo, hi = enc[k][0][:, col], enc[k][1][:, col]
            for i, x in enumerate(points):
                ref = mp.diff(oracle, mp.mpf(float(x)), k)
                if not mp.isfinite(ref):
                    continue
                # mpmath.diff leaves ~1e-160 noise on exactly-zero derivatives.
                noise = mp.mpf(10) ** -100
                assert lo[i] - noise <= ref <= hi[i] + noise, (
                    column.label,
                    k,
                    x,
                    lo[i],
                    float(ref),
                    hi[i],
                )


def test_point_potential_enclosure_uses_stable_polynomial_horner():
    """A tiny positive curvature survives cancellation among degree-12 terms."""
    model = _compile(BasisSpec(support=(0.0, math.inf), bulk_degree=12, log_lower=True))
    labels = [column.label for column in model.columns]
    theta = np.zeros(model.n_coefficients)
    root, curvature = 13.8, 8e-11
    theta[labels.index("x^2")] = (root**2 + curvature) / 2.0
    theta[labels.index("x^3")] = -root / 3.0
    theta[labels.index("x^4")] = 1.0 / 12.0
    x = model.geometry.center + model.geometry.scale * root

    enclosure = _potential_enclosure(model, theta, [x], [x], (2,))[2]
    z = (Fraction(x) - Fraction(model.geometry.center)) / Fraction(model.geometry.scale)
    exact = sum(
        n
        * (n - 1)
        * Fraction(theta[labels.index(f"x^{n}")])
        * z ** (n - 2)
        / Fraction(model.geometry.scale) ** 2
        for n in range(2, 5)
    )

    assert Fraction(enclosure[0][0]) <= exact <= Fraction(enclosure[1][0])
    assert enclosure[0][0] > 0.0


def _cells(model, n, rng):
    """Random cells of widths from 1e-9 to ~1 scale, inside the support."""
    req, g = model.request, model.geometry
    lo_b = req.lower if math.isfinite(req.lower) else g.center - 12.0 * g.scale
    up_b = req.upper if math.isfinite(req.upper) else g.center + 12.0 * g.scale
    span = up_b - lo_b
    centers = lo_b + span * rng.uniform(0.02, 0.98, size=n)
    widths = g.scale * 10.0 ** rng.uniform(-9.0, 0.0, size=n)
    lo = np.maximum(centers - widths / 2, lo_b + 0.01 * span)
    hi = np.minimum(centers + widths / 2, up_b - 0.01 * span)
    return lo, np.maximum(hi, lo)


@pytest.mark.parametrize("spec", list(_broad_battery()), ids=lambda s: "")
def test_cell_enclosures_contain_dense_samples(spec):
    model = _compile(spec)
    lo, hi = _cells(model, 40, RNG)
    enc = _basis_enclosure(model, lo, hi, range(5))
    frac = np.linspace(0.0, 1.0, 17)
    for i in range(lo.size):
        x = lo[i] + (hi[i] - lo[i]) * frac
        vals = _evaluate_basis(model, x, max_order=4)
        for k in range(5):
            v = vals.derivative(k)
            scale = np.max(np.abs(v), axis=0) + 1e-300
            tol = 1e-12 * scale
            assert np.all(v >= enc[k][0][i] - tol), (k, i)
            assert np.all(v <= enc[k][1][i] + tol), (k, i)


@pytest.mark.parametrize(
    "spec",
    [
        BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=SoftPower(2.5)),
        BasisSpec(support=(-1.0, 3.0), bulk_degree=3, log_lower=True, log_upper=True),
    ],
    ids=lambda s: "",
)
def test_potential_enclosure_contains_samples_and_shrinks_linearly(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        model = _compile(spec)
    theta = RNG.normal(size=model.n_coefficients)
    g = model.geometry
    center = g.center + 0.37 * g.scale
    widths = []
    for w in (1e-1, 1e-2, 1e-3, 1e-4):
        lo, hi = np.array([center - w * g.scale]), np.array([center + w * g.scale])
        enc = _potential_enclosure(model, theta, lo, hi, (2, 3, 4))
        x = np.linspace(lo[0], hi[0], 33)
        vals = _evaluate_basis(model, x, max_order=4)
        for k in (2, 3, 4):
            q = vals.derivative(k) @ theta
            tol = 1e-12 * (np.abs(q).max() + 1.0)
            assert np.all(q >= enc[k][0][0] - tol) and np.all(q <= enc[k][1][0] + tol)
        widths.append(enc[2][1][0] - enc[2][0][0])
    ratios = np.array(widths[1:]) / np.array(widths[:-1])
    assert np.all(ratios < 0.2), widths


def test_zero_coefficient_log_column_is_skipped_at_its_endpoint():
    model = _compile(
        BasisSpec(support=(-1.0, 3.0), bulk_degree=2, log_lower=True, log_upper=True)
    )
    theta = np.ones(model.n_coefficients)
    lo_col = next(i for i, c in enumerate(model.columns) if c.label == "logL")
    theta[lo_col] = 0.0
    enc = _potential_enclosure(model, theta, np.array([-1.0]), np.array([-0.9]), (2,))
    assert np.isfinite(enc[2][0][0]) and np.isfinite(enc[2][1][0])


def test_inlined_ulp_steps_match_nextafter():
    from gibbus._model._basis_kernels import _ulp_steps

    rng = np.random.default_rng(7)
    mags = 10.0 ** rng.uniform(-310.0, 308.0, 2000)
    tiny = np.finfo(float).smallest_subnormal
    big = np.finfo(float).max
    x = np.concatenate(
        [mags, -mags, [0.0, -0.0, tiny, -tiny, 3 * tiny, big, -big]]
        + [[math.inf, -math.inf]]
    )
    up, down = _ulp_steps(x)
    with np.errstate(over="ignore"):  # +-DBL_MAX steps to +-inf
        assert np.array_equal(up, np.nextafter(x, math.inf))
        assert np.array_equal(down, np.nextafter(x, -math.inf))
    nan_up, nan_down = _ulp_steps(np.array([math.nan]))
    assert math.isnan(nan_up[0]) and math.isnan(nan_down[0])
