"""Reduced constraint balancing against the full incidence least squares."""

import numpy as np
import pytest

from gibbus._fit.basis_constraints import _column_scales, _ConstraintRows


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("disconnected", [False, True])
def test_reduced_balancing_preserves_full_minimum_norm_scales(seed, disconnected):
    """Eliminating row offsets preserves scales and disconnected gauges.

    Parameters
    ----------
    seed : int
        Random problem seed.
    disconnected : bool
        Split the constraint incidence graph into independent groups.
    """
    rng = np.random.default_rng(seed)
    values = np.exp(rng.uniform(-100, 100, (35, 8)))
    values[rng.random(values.shape) < 0.35] = 0.0
    if disconnected:
        values[:18, 4:] = 0.0
        values[18:, :4] = 0.0
    hessian = np.diag(np.exp(rng.uniform(-50, 50, 8)))
    rows = _ConstraintRows(values, ("cut",) * len(values))
    entries = []
    for i, row in enumerate(values):
        for j in np.flatnonzero(row):
            entries.append((i, j, np.log(row[j])))
    for j, value in enumerate(np.diag(hessian)):
        entries.append((len(values) + j, j, 0.5 * np.log(value)))
    offsets = len(values) + 8
    design = np.zeros((len(entries), offsets + 8))
    target = np.zeros(len(entries))
    for index, (i, j, value) in enumerate(entries):
        design[index, i] = 1.0
        design[index, offsets + j] = 1.0
        target[index] = -value
    gamma = np.linalg.lstsq(design, target, rcond=None)[0][offsets:]
    gamma -= gamma.mean()
    actual = _column_scales(rows, hessian)
    np.testing.assert_allclose(np.log(actual), gamma, atol=2e-11, rtol=2e-11)
