"""Structured scaled curvature kernel and its theta-free per-atom rows.

``curvature`` returns ``H = q''_zz / ||pieces||`` with an error bound
``err_h`` and the exact scale; ``curvature_atom_rows`` returns per-atom rows
``R``, ``N`` with ``q'' = S sum_i c_i R_i``.  The reference is the independent
high-precision oracle of ``_basis_oracle`` in canonical units.  ``H`` must
match the oracle's ``q''`` divided by the kernel's own scale (rebuilt exactly
from the double-double reference parts) to the kernel's error bound, at
points from the origin to ``|z| = 1e12`` and next to finite endpoints; the
fast and log paths must agree; and the rows must reproduce ``H`` for any
coefficients.
"""

import math
import warnings
from fractions import Fraction

import mpmath as mp
import numpy as np
import pytest
from _basis_oracle import BasisOracle
from gibbus._model._basis_kernels import curvature, curvature_atom_rows

from gibbus import BasisSpec, Exponential, GibbusBasisWarning, Polynomial, SoftPower
from gibbus._defaults import BASIS_GATE_SHARPNESS
from gibbus._model.basis import _build_basis

INF = math.inf
RNG = np.random.default_rng(20261024)
NORMAL = RNG.normal(1.5, 2.0, size=400)
HALF = RNG.gamma(2.0, 1.5, size=400) + 0.25

SPECS = [
    BasisSpec(bulk_degree=3),
    BasisSpec(bulk_degree=4, lower_tail=Polynomial(1), upper_tail=Polynomial(1)),
    BasisSpec(bulk_degree=2, upper_tail=Polynomial(5)),
    BasisSpec(
        bulk_degree=3,
        lower_tail=SoftPower(1.5),
        upper_tail=SoftPower(2.5),
        expansion="full",
    ),
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(1.01), upper_tail=SoftPower(1.01)),
    BasisSpec(bulk_degree=2, lower_tail=SoftPower(7.3), upper_tail=SoftPower(40.5)),
    BasisSpec(
        bulk_degree=4,
        lower_tail=Exponential(0.7),
        upper_tail=Exponential(1.3),
        expansion="full",
    ),
    BasisSpec(
        bulk_degree=2, lower_tail=Exponential(1e-6), upper_tail=Exponential(50.0)
    ),
    BasisSpec(
        support=(0.0, INF), bulk_degree=2, log_lower=True, upper_tail=SoftPower(1.2)
    ),
    BasisSpec(
        support=(0.0, INF), bulk_degree=3, log_lower=True, upper_tail=Polynomial(1)
    ),
    BasisSpec(
        support=(-INF, 0.0), bulk_degree=2, log_upper=True, lower_tail=Exponential(2.0)
    ),
    BasisSpec(support=(-2.0, 3.0), bulk_degree=4, log_lower=True, log_upper=True),
]

FAR = (1e12, 1e6, 300.0, 40.0, 7.0, 1.0, 0.3, 2e-3, 1e-9, 1e-200)


def _compile(spec):
    lo, up = spec.support
    data = (
        NORMAL if lo == -INF and up == INF else (HALF + lo if up == INF else up - HALF)
    )
    if math.isfinite(lo) and math.isfinite(up):
        data = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GibbusBasisWarning)
        return _build_basis(spec, data)


def _points(model):
    lo, up = model.geometry.coordinate.canonical_support
    z = np.array(sorted({0.0, *FAR, *(-v for v in FAR)}))
    z = z[(z > lo) & (z < up)]
    for end, sign in ((lo, 1.0), (up, -1.0)):
        if math.isfinite(end):
            z = np.concatenate([z, end + sign * np.array([1e-12, 1e-6, 1e-3, 0.1])])
    return z


def _scale(z, refk, a1, rem, hl):
    """Exact ``log M`` from the double-double reference parts."""
    s = (
        mp.mpf(Fraction(BASIS_GATE_SHARPNESS).numerator)
        / Fraction(BASIS_GATE_SHARPNESS).denominator
    )
    zz = mp.mpf(float(z))
    total = mp.mpf(a1[0]) + mp.mpf(a1[1])
    total = total * zz + mp.mpf(rem[0]) + mp.mpf(rem[1]) + mp.mpf(hl[0]) + mp.mpf(hl[1])
    if refk:
        total -= s * s / 2 * zz * zz
    return total


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "")
def test_scaled_curvature_matches_the_oracle_to_its_error_bound(spec):
    model = _compile(spec)
    oracle = BasisOracle(model, units="z", max_order=2)
    z = _points(model)
    for _ in range(2):
        theta = RNG.normal(size=model.n_coefficients)
        h, err, _, refk, a1, rem, hl = curvature(model.kernel, theta, z, reference=True)
        assert np.all(np.isfinite(h)) and np.all(np.abs(h) <= np.sqrt(14 * 4 * len(h)))
        for i, zi in enumerate(z):
            value, _ = oracle.potential(theta, zi, 2)
            with mp.workdps(oracle.digits(zi)):
                if value == 0:
                    assert h[i] == 0.0, zi
                    continue
                ref = value / mp.exp(_scale(zi, refk[i], a1[i], rem[i], hl[i]))
                gap = abs(mp.mpf(h[i]) - ref)
                assert gap <= 4 * err[i] + 1e-300, (zi, h[i], float(ref), err[i])
                if abs(ref) > 4 * err[i]:
                    assert np.sign(h[i]) == np.sign(float(ref)), zi


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "")
def test_fast_and_log_paths_agree(spec):
    model = _compile(spec)
    lo, up = model.geometry.coordinate.canonical_support
    z = np.concatenate([-np.logspace(-35, 35, 300), np.logspace(-35, 35, 300), [0.0]])
    z = z[(z > lo) & (z < up)]
    for _ in range(3):
        theta = RNG.normal(size=model.n_coefficients)
        fast, ef, lf = curvature(model.kernel, theta, z)
        slow, es, ls = curvature(model.kernel, theta, z, use_fast=False)
        assert np.all(np.abs(fast - slow) <= 2 * (ef + es) + 1e-300)
        finite = np.isfinite(lf)
        assert np.allclose(lf[finite], ls[finite], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: "")
def test_atom_rows_reproduce_the_scaled_curvature(spec):
    model = _compile(spec)
    z = _points(model)
    rows, norms, refk, a1, rem = curvature_atom_rows(model.kernel, z)
    atoms = model.kernel.atom_rows
    assert rows.shape == norms.shape == (z.size, atoms.shape[0])
    # N = -1 marks an atom beyond the rows' dynamic range (its R is zero).
    out = norms < 0.0
    assert np.all(norms[out] == -1.0) and np.all(rows[out] == 0.0)
    norms = np.where(out, 0.0, norms)
    assert np.all(np.abs(rows) <= norms * (1 + 1e-12) * 4)
    for _ in range(3):
        theta = RNG.normal(size=model.n_coefficients)
        coef = atoms @ theta
        h, err, logm = curvature(model.kernel, theta, z)
        norm = np.sqrt(((norms * coef) ** 2).sum(axis=1))
        live = norm > 0
        # Every point with a resolved curvature has a nonzero row norm.
        assert np.all(live | (np.abs(h) <= 8 * err)), z[~live]
        from_rows = (rows @ coef)[live] / norm[live]
        assert np.all(
            np.abs(from_rows - h[live]) <= 8 * err[live] + 1e-12 * np.abs(h[live])
        )
        # The scales differ only by the piece norm: log M = log S + log ||c N||.
        log_s = np.array(
            [
                float(_scale(zi, k, a, r, (0.0, 0.0)))
                for zi, k, a, r in zip(z[live], refk[live], a1[live], rem[live])
            ]
        )
        expected = log_s + np.log(norm[live])
        good = np.isfinite(logm[live]) & (np.abs(expected) < 1e12)
        np.testing.assert_allclose(
            logm[live][good], expected[good], rtol=1e-12, atol=1e-10
        )


def test_column_curvature_rows_combine_atoms_exactly():
    model = _compile(SPECS[3])
    z = np.array([-30.0, -2.0, -0.5, 0.0, 0.7, 3.0, 25.0])
    rows, _, _, _, _ = curvature_atom_rows(model.kernel, z)
    columns = rows @ model.kernel.atom_rows  # (n, d): q''/S per unit column
    theta = RNG.normal(size=model.n_coefficients)
    np.testing.assert_allclose(
        columns @ theta, rows @ (model.kernel.atom_rows @ theta), rtol=1e-12, atol=1e-14
    )


def test_atoms_partition_every_gated_function():
    # Full-line gated functions carry W_L, W_C and W_U atoms; half-line ones the
    # single gate and its complement; ungated ones a single atom.
    model = _compile(SPECS[0])
    codes = {}
    for j, code in model.kernel.atoms:
        codes.setdefault(j, set()).add(code)
    for j, found in codes.items():
        assert found in ({0}, {3, 5, 4}, {5}, {3, 5}, {5, 4}), (j, found)
    half = _compile(SPECS[8])
    for j, code in half.kernel.atoms:
        assert code in (0, 2, 4)


def test_empty_inputs_and_zero_coefficients():
    model = _compile(SPECS[0])
    h, err, logm = curvature(
        model.kernel, np.zeros(model.n_coefficients), np.array([0.5])
    )
    assert h[0] == 0.0 and err[0] == 0.0 and logm[0] == -np.inf
    h, err, logm = curvature(model.kernel, np.ones(model.n_coefficients), np.zeros(0))
    assert h.size == err.size == logm.size == 0
    rows, norms, refk, a1, rem = curvature_atom_rows(model.kernel, np.zeros(0))
    assert rows.shape == (0, len(model.kernel.atoms)) and refk.size == 0
