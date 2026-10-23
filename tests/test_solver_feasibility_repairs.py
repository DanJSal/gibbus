"""Row restoration must preserve objective progress and certify its result."""

from types import SimpleNamespace

import numpy as np

from gibbus._fit.basis_constraints import _ConstraintRows
from gibbus._fit.basis_newton import _checked_row_polish


def test_tiny_coefficient_repair_cannot_hide_large_likelihood_cost():
    rows = _ConstraintRows(np.array([[1.0, 0.0]]), ("cut",))
    phi = np.array([-1e-14, 1.0])
    original = SimpleNamespace(nll=0.0)

    def objective(candidate):
        # The first coordinate is highly informative despite its tiny size.
        return SimpleNamespace(nll=0.5e28 * (candidate[0] - phi[0]) ** 2)

    out, evaluation, changed = _checked_row_polish(
        objective, rows, phi, original, np.ones(2)
    )
    assert out is phi
    assert evaluation is original
    assert not changed


def test_likelihood_improving_row_repair_returns_its_own_evaluation():
    rows = _ConstraintRows(np.eye(2), ("cut", "log"))
    phi = np.array([1.0, -np.finfo(float).eps / 4])
    original = SimpleNamespace(nll=-phi[1])

    def objective(candidate):
        assert candidate[1] >= 0.0
        return SimpleNamespace(nll=-candidate[1])

    out, evaluation, changed = _checked_row_polish(
        objective, rows, phi, original, np.array([1.0, 100.0])
    )
    assert changed
    assert out[1] >= 0.0
    assert evaluation.nll == -out[1]
    assert evaluation.nll <= original.nll


def test_repaired_endpoint_cannot_keep_a_pre_repair_certificate(monkeypatch):
    from gibbus import BasisSpec
    from gibbus._fit import basis_newton as bn
    from gibbus._fit.basis_objective import _BasisPointObjective, _point_statistics
    from gibbus._fit.conic_newton import _NewtonRun
    from gibbus._model.basis import _build_basis
    from gibbus._model.basis_normalizer import _NormalizerContext

    data = np.linspace(-0.9, 0.9, 21)
    model = _build_basis(
        BasisSpec(
            support=(-1.0, 1.0),
            bulk_degree=2,
            log_lower=False,
            log_upper=False,
            expansion="minimal",
        ),
        data,
    )
    objective = _BasisPointObjective(
        _NormalizerContext(model), _point_statistics(model, data)
    )

    def supposedly_converged(self, rep, phi, blocks, evaluation, *args):
        point = np.array([0.0, -1e-14])
        state = self(point)
        return (
            _NewtonRun(
                "converged", point, blocks, np.zeros(rep.n_rows), state, 0, 1, 0, 1e-14
            ),
            [],
        )

    monkeypatch.setattr(bn._FaceObjective, "run", supposedly_converged)
    monkeypatch.setattr(
        bn,
        "_preconditioned_subproblem",
        lambda *args: (SimpleNamespace(gap=np.inf, iterations=0), None, None, 0.0),
    )
    # One exchange allows the first solve and repair, but not certification
    # of the changed endpoint. Its inherited tiny bound must be discarded.
    result = bn._solve_basis(objective, max_exchanges=1)
    assert result.status not in bn._CONVERGED
    assert np.isinf(result.final_decrease_bound)
