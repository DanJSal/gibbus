"""Constraint retirement preserves active rows and permanent inequalities."""

import numpy as np

from gibbus._fit.basis_constraints import _ConstraintRows
from gibbus._fit.basis_newton import _retire_inactive_cuts
from gibbus._fit.conic_newton import _NewtonOptions


def test_inactive_cuts_expire_while_structural_and_binding_rows_persist():
    """Repeated inactivity expires only point cuts with positive slack."""
    rows = _ConstraintRows(np.eye(5), ("cut", "cut", "log", "growth", "tail"))
    theta = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    options = _NewtonOptions(cut_retention_exchanges=2)
    ages = np.zeros(5, dtype=np.int64)
    original = rows
    for _ in range(2):
        rows, ages = _retire_inactive_cuts(rows, ages, theta, options)
        assert rows is original
    rows, ages = _retire_inactive_cuts(rows, ages, theta, options)
    np.testing.assert_array_equal(rows.rows, np.eye(5)[1:])
    assert rows.kinds == ("cut", "log", "growth", "tail")
    np.testing.assert_array_equal(ages, 0)


def test_activity_resets_age_and_rediscovered_cuts_start_fresh():
    """A returning contact survives and a regenerated point cut can reenter."""
    rows = _ConstraintRows(np.eye(2), ("cut", "cut"))
    options = _NewtonOptions(cut_retention_exchanges=1)
    ages = np.zeros(2, dtype=np.int64)
    rows, ages = _retire_inactive_cuts(rows, ages, np.ones(2), options)
    rows, ages = _retire_inactive_cuts(rows, ages, np.array([0.0, 1.0]), options)
    np.testing.assert_array_equal(rows.rows, [[1.0, 0.0]])
    rows = rows.merged(np.array([[0.0, 1.0]]), "cut")
    rows, ages = _retire_inactive_cuts(rows, ages, np.zeros(2), options)
    np.testing.assert_array_equal(rows.rows, np.eye(2))
    np.testing.assert_array_equal(ages, 0)


def test_cancellation_margin_protects_near_binding_cuts():
    """A small relative residual keeps a cut despite large coefficients."""
    row = np.array([[1.0, -1.0]]) / np.sqrt(2.0)
    rows = _ConstraintRows(row, ("cut",))
    options = _NewtonOptions(cut_retention_exchanges=1)
    ages = np.zeros(1, dtype=np.int64)
    for _ in range(4):
        rows, ages = _retire_inactive_cuts(
            rows, ages, np.array([1e10 + 1.0, 1e10]), options
        )
    assert rows.size == 1
    np.testing.assert_array_equal(ages, 0)


def test_retirement_can_be_disabled():
    """The reference policy preserves every row and the age state."""
    rows = _ConstraintRows(np.eye(2), ("cut", "tail"))
    ages = np.zeros(0, dtype=np.int64)
    kept, updated = _retire_inactive_cuts(
        rows, ages, np.ones(2), _NewtonOptions(cut_retention_exchanges=0)
    )
    assert kept is rows
    assert updated is ages
