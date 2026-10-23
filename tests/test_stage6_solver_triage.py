"""Structured diagnostics for nonconverged Stage 6 fitting cases.

A numerical curvature verdict and a small optimization bound are separate
requirements.  An explicit request with an unresolved Newton certificate must
fail, retaining enough context to reproduce and assess the failure.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from gibbus._fit import basis_selection as bs


def test_explicit_nonconvergence_reports_last_solver_certificate(monkeypatch):
    result = SimpleNamespace(
        status="iteration_limit",
        objective_value=1.325,
        final_decrease_bound=2.1e-5,
        exchanges=30,
        newton_iterations=120,
        objective_evaluations=190,
        subproblem_iterations=800,
        final_separation=SimpleNamespace(
            parts=(SimpleNamespace(status="nonnegative"),)
        ),
    )
    fit = SimpleNamespace(
        model=SimpleNamespace(request=SimpleNamespace(bulk_degree=4)),
        result=result,
    )
    monkeypatch.setattr(bs, "_search_data", lambda *args: fit)
    with pytest.raises(bs._BasisConvergenceError, match="did not converge") as exc:
        bs._fit_data(None, None, None, None, None)
    details = exc.value.diagnostics
    assert details == {
        "bulk_degree": 4,
        "status": "iteration_limit",
        "objective_value": 1.325,
        "final_decrease_bound": 2.1e-5,
        "curvature_exchanges": 30,
        "newton_iterations": 120,
        "objective_evaluations": 190,
        "subproblem_iterations": 800,
        "shape_status": ("nonnegative",),
    }
    assert "predicted-decrease bound=2.1e-05" in str(exc.value)
    assert "shape=('nonnegative',)" in str(exc.value)


def test_nonconverged_fits_preserve_nonfinite_bounds(monkeypatch):
    result = SimpleNamespace(
        status="non_descent",
        objective_value=0.3,
        final_decrease_bound=np.inf,
        exchanges=3,
        newton_iterations=4,
        objective_evaluations=5,
        subproblem_iterations=20,
        final_separation=None,
    )
    fit = SimpleNamespace(
        model=SimpleNamespace(request=SimpleNamespace(bulk_degree=12)),
        result=result,
    )
    monkeypatch.setattr(bs, "_search_data", lambda *args: fit)
    with pytest.raises(bs._BasisConvergenceError) as exc:
        bs._fit_data(None, None, None, None, None)
    assert np.isinf(exc.value.diagnostics["final_decrease_bound"])
    assert exc.value.diagnostics["shape_status"] == ()


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason=(
        "OI-E2-6: two degree-4 linear tails on Laplace(0,1) data still "
        "exhaust curvature cut exchange; increasing exchanges to 65 did not "
        "resolve its optimization certificate"
    ),
)
def test_degree4_two_linear_tails_reach_a_converged_certificate():
    """Acceptance target: certify the full explicit degree-4 linear-tail fit."""
    from gibbus import BasisSpec, Distribution, Polynomial

    samples = np.random.default_rng(0).laplace(size=800)
    fitted = Distribution().fit(
        samples,
        basis=BasisSpec(
            bulk_degree=4,
            lower_tail=Polynomial(1),
            upper_tail=Polynomial(1),
            expansion="minimal",
        ),
        rng=0,
    )
    assert fitted.fit_diagnostics["converged"]
    assert all(
        c["shape"] == "nonnegative" for c in fitted.fit_diagnostics["components"]
    )
