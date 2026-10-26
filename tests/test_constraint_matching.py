"""Compiled constraint comparisons against the entrywise NumPy predicate."""

import numpy as np
import pytest

from gibbus._fit.basis_constraints import _same_constraint


@pytest.mark.parametrize("layout", ["contiguous", "strided", "readonly"])
@pytest.mark.parametrize("seed", range(8))
def test_constraint_matching_preserves_entrywise_comparisons(seed, layout):
    """Preserve zero patterns, relative thresholds and input layouts.

    Parameters
    ----------
    seed : int
        Random row seed.
    layout : str
        Buffer layout and write permission.
    """
    rng = np.random.default_rng(seed)
    units = rng.normal(size=(24, 12)) * np.exp(rng.uniform(-300, 300, (24, 12)))
    units[rng.random(units.shape) < 0.25] = 0.0
    rows = np.r_[units.copy(), -units, units * (1 + 4 * np.finfo(float).eps)]
    rows[0, 0] = np.nextafter(units[0, 0], np.inf)
    rows[1, 1] = np.nextafter(0.0, 1.0)
    if layout == "strided":
        units = np.repeat(units, 2, axis=1)[:, ::2]
        rows = np.repeat(rows, 2, axis=1)[:, ::2]
    elif layout == "readonly":
        units.flags.writeable = False
        rows.flags.writeable = False
    u, r = units[:, None, :], rows[None, :, :]
    expected = np.all(
        ((r != 0.0) == (u != 0.0))
        & (np.abs(r - u) <= 4 * np.finfo(float).eps * np.abs(u)),
        axis=2,
    )
    np.testing.assert_array_equal(_same_constraint(units, rows), expected)


@pytest.mark.parametrize("shape", [(0, 3, 4), (3, 0, 4), (2, 3, 0)])
def test_constraint_matching_handles_empty_axes(shape):
    """Empty row sets and empty coefficient spaces retain Boolean semantics.

    Parameters
    ----------
    shape : tuple
        The two row counts and the common coefficient count.
    """
    k, m, d = shape
    result = _same_constraint(np.empty((k, d)), np.empty((m, d)))
    np.testing.assert_array_equal(result, np.ones((k, m), dtype=bool))
