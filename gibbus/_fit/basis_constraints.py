"""Coefficient-linear constraint rows of a fit and their cone description.

The solver's description is ``B theta = sum_b L_b(Q_b)``, ``Q_b`` PSD,
consumed by the compiled interior-point solver; every constraint is a
homogeneous row ``a . theta >= 0``, so each row is one ``1 x 1`` block (a
nonnegative scalar).  A strictly interior cold start is built here too.

Rows, by kind:

* ``"log"`` -- an enabled finite-endpoint log amplitude is nonnegative;
* ``"growth"`` -- at each infinite end the requested leading amplitude is
  outward-nonnegative (``_SideAsymptotics.leading``).  Interior-point endpoints
  lie strictly inside the description, and the segment between two interior
  points does too, so every iterate keeps a strictly positive leader and is
  proper; a collapse toward zero is a face decision;
* ``"cut"`` -- ``Phi''(x*) . theta >= 0`` at a curvature minimum;
* ``"tail"`` -- the leading curvature functional of a violated tail.

Cut and tail rows come from ``curvature_search`` and are permanent once added.
Each row is scaled to unit norm, which leaves the cone unchanged.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._defaults import _reraise_if_debug
from .._model.basis_normalizer import _QuadratureFailure
from ._curvature_kernels import same_constraint_rows
from .conic_newton import _preconditioned_subproblem
from .conic_qp import _ConicRepresentation

_KINDS = ("log", "growth", "cut", "tail")
_START_TOL = 1e-12
_START_STEP = 8.0  # largest log-scale step of the ray search
_SAME_DIRECTION = 4.0 * np.finfo(np.float64).eps


def _same_constraint(units, rows, /):
    """Which unit rows ``units[i]`` are the same constraint as ``rows[j]``.

    A direction test (cosine) cannot separate rows that differ only in
    entries far below the largest one (a far-tail cut is ``e_exp`` plus
    polynomial entries ``1e-17`` of it), yet those entries are exact and carry
    the constraint.  Rows are the same only if their zero patterns agree and
    every entry agrees to four ulp of itself (of ``units[i]``'s entry).

    Parameters
    ----------
    units : numpy.ndarray
        Unit-norm rows, shape ``(k, d)``.
    rows : numpy.ndarray
        Unit-norm rows, shape ``(m, d)``.

    Returns
    -------
    numpy.ndarray
        Boolean, shape ``(k, m)``.
    """
    return same_constraint_rows(units, rows, _SAME_DIRECTION)


@dataclass(frozen=True)
class _ConstraintRows:
    """Homogeneous constraint rows ``rows @ theta >= 0``.

    Parameters
    ----------
    rows : numpy.ndarray
        Unit-norm rows, shape ``(r, d)``.
    kinds : tuple of str
        Kind of each row (``"log"``, ``"growth"``, ``"cut"``, ``"tail"``).
    """

    rows: np.ndarray
    kinds: tuple

    @property
    def size(self):
        """Number of rows."""
        return len(self.kinds)

    def slack(self, theta, /):
        """Return ``rows @ theta``.

        Parameters
        ----------
        theta : numpy.ndarray
            Coefficients, shape ``(d,)``.
        """
        return self.rows @ np.asarray(theta, dtype=np.float64)

    def merged(self, rows, kind, /):
        """Return the rows with new ones of one kind appended.

        A new row equal to an existing row entry by entry (to four ulp of each
        entry, same zero pattern) is the same constraint and is not added
        again.

        Parameters
        ----------
        rows : numpy.ndarray
            New rows, shape ``(k, d)``; any nonzero scale.
        kind : str
            Their kind.

        Raises
        ------
        ValueError
            If ``kind`` is unknown.
        RuntimeError
            If a generated constraint row is zero or not finite.
        """
        if kind not in _KINDS:
            raise ValueError(f"unknown constraint kind {kind!r}")
        d = self.rows.shape[1]
        new = np.atleast_2d(np.asarray(rows, dtype=np.float64)).reshape(-1, d)
        # Scale by the largest entry first: a far-out row (entries ~1e-170)
        # has a 2-norm that underflows to zero.
        peak = np.max(np.abs(new), axis=1) if d else np.zeros(len(new))
        if not (np.all(np.isfinite(new)) and np.all(peak > 0.0)):
            raise RuntimeError("generated constraint rows must be finite and nonzero")
        unit = new / peak[:, None]
        unit /= np.sqrt(np.einsum("ij,ij->i", unit, unit))[:, None]
        # A new row is dropped if it is an existing row or an earlier added one.
        known = np.any(_same_constraint(unit, self.rows), axis=1)
        among = _same_constraint(unit, unit)
        added = []
        for i in range(len(unit)):
            if not known[i]:
                added.append(i)
                known |= among[:, i]
        if not added:
            return self
        return _ConstraintRows(
            np.vstack([self.rows, unit[added]]),
            self.kinds + (kind,) * len(added),
        )


def _structural_rows(model, /):
    """Rows every fit carries: log-amplitude bounds and leading growth.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.

    Returns
    -------
    _ConstraintRows
    """
    d = model.n_coefficients
    rows = _ConstraintRows(np.zeros((0, d)), ())
    logs = [i for i, column in enumerate(model.columns) if column.nonnegative]
    if logs:
        rows = rows.merged(np.eye(d)[logs], "log")
    if model.asymptotics:
        lead = [
            [float(v) for v in side.leading.row_exact] for side in model.asymptotics
        ]
        rows = rows.merged(np.array(lead), "growth")
    return rows


def _linear_representation(rows, /):
    """Cone description of ``rows @ theta >= 0`` for the conic solver.

    Each row is a ``1 x 1`` block; ``y = -1`` is a strictly dual-feasible
    reference multiplier (every slack ``-y_b`` is positive).

    Parameters
    ----------
    rows : _ConstraintRows
        Constraint rows (at least one).

    Returns
    -------
    _ConicRepresentation
    """
    r = rows.size
    if r == 0:
        raise ValueError("a cone description needs at least one row")
    unit = np.eye(r)
    matrices = tuple(unit[:, b].reshape(r, 1, 1).copy() for b in range(r))
    return _ConicRepresentation(
        np.ascontiguousarray(rows.rows),
        matrices,
        -np.ones(r),
        -np.ones(r, dtype=np.intp),
    )


def _column_scales(rows, hessian, /):
    """Row-flattening column scales: where row feasibility is measured (Curtis-Reid).

    Rows are point evaluations of general columns (an exponential column can
    exceed the others by ``1e18`` at a far curvature cut), so the scales are
    least-squares balanced in log magnitude: scales ``c_j = exp(gamma_j)`` minimizing
    ``sum (log|a_ij| + rho_i + gamma_j)**2`` over the nonzero row entries
    together with the Jacobi targets ``log sqrt(H_jj)`` (one pseudo-row per
    column), so likelihood curvature and constraint geometry are balanced.
    In these coordinates every row is resolvable, so feasibility of the
    subproblem endpoints is measured and restored here (``_polish``, the
    KKT tests).  The interior-point solves themselves run in the Jacobi
    scaling: flattening the rows can spread the scaled Hessian over
    1e18 (a near-zero leader face whose cuts carry 1e-14 gate entries), past
    what the solver's dual certificate resolves (``non_descent``, bound
    ``inf``).

    Parameters
    ----------
    rows : _ConstraintRows
        Constraint rows (any coordinates matching ``hessian``).
    hessian : numpy.ndarray
        Newton-model Hessian, shape ``(n, n)``.

    Returns
    -------
    numpy.ndarray
        Positive column scales, shape ``(n,)``.
    """
    h = np.asarray(hessian, dtype=np.float64)
    n = h.shape[0]
    diag = np.diag(h)
    # Eliminate each row offset by centering its nonzero entries. The
    # remaining least-squares problem has n columns, independent of the
    # number of cuts. Preserve the original minimum-norm gauge separately
    # on every connected component of the row/column incidence graph.
    values = np.vstack([rows.rows, np.diag(np.sqrt(np.maximum(diag, 0.0)))])
    mask = values != 0.0
    counts = mask.sum(axis=1)
    mask = mask[counts > 0]
    values = values[counts > 0]
    counts = counts[counts > 0]
    logs = np.zeros_like(values)
    np.log(np.abs(values), out=logs, where=mask)
    means = logs.sum(axis=1) / counts
    incidence = mask.astype(float)
    centered = incidence / counts[:, None]
    ii, jj = np.nonzero(mask)
    design = -centered[ii].copy()
    design[np.arange(len(ii)), jj] += 1.0
    target = means[ii] - logs[ii, jj]
    adjacency = incidence.T @ incidence > 0
    remaining = set(range(n))
    gauges, targets = [], []
    while remaining:
        component = {min(remaining)}
        pending = list(component)
        while pending:
            column = pending.pop()
            neighbors = set(np.flatnonzero(adjacency[column])) - component
            component.update(neighbors)
            pending.extend(neighbors)
        remaining.difference_update(component)
        columns = sorted(component)
        selected = np.any(mask[:, columns], axis=1)
        gauge = centered[selected].sum(axis=0)
        gauge[columns] += 1.0
        gauges.append(gauge)
        targets.append(-float(means[selected].sum()))
    design = np.vstack([design, gauges])
    target = np.concatenate([target, targets])
    gamma = np.linalg.lstsq(design, target, rcond=None)[0]
    # Fix the free row/column gauge: geometric mean of the scales is one.
    gamma -= gamma.mean()
    return np.exp(gamma)


def _unit_blocks(rows, /):
    """Interior ``1 x 1`` blocks (all ones) for ``rows``.

    Parameters
    ----------
    rows : _ConstraintRows
        Constraint rows.
    """
    return tuple(np.ones((1, 1)) for _ in range(rows.size))


def _interior_direction(rows, /):
    """Strictly interior direction of the row cone (max-margin).

    With a margin variable ``t`` the homogeneous system ``a_i . theta >= t``
    (unit rows ``a_i``), ``t >= 0`` is solved for
    ``min 1/2 (|theta|^2 + t^2) - t`` by the interior-point solver: the
    result has slack at least ``t > 0`` in every row whenever the cone has an
    interior.  (Minimizing a pull alone
    can stop on a face once cut rows are present.)

    Parameters
    ----------
    rows : _ConstraintRows
        Rows (structural, optionally with cuts).

    Returns
    -------
    numpy.ndarray
        Direction ``w``, shape ``(d,)``; zero when there are no rows.

    Raises
    ------
    RuntimeError
        If the endpoint is not strictly interior (the cone has no interior to
        the solver's resolution).
    """
    d = rows.rows.shape[1]
    if rows.size == 0:
        return np.zeros(d)
    augmented = _ConstraintRows(np.zeros((0, d + 1)), ())
    augmented = augmented.merged(
        np.hstack([rows.rows, -np.ones((rows.size, 1))]), "cut"
    )
    augmented = augmented.merged(np.eye(d + 1)[d], "cut")
    hessian = np.eye(d + 1)
    gradient = np.zeros(d + 1)
    gradient[d] = -1.0
    # Only strict interiority matters here, not an optimal margin, so the
    # subproblem's own convergence is not required.
    _, x, _, _ = _preconditioned_subproblem(
        hessian,
        gradient,
        np.zeros(d + 1),
        _linear_representation(augmented),
        _unit_blocks(augmented),
    )
    w, margin = x[:d], float(x[d])
    slack = rows.slack(w)
    if not margin > 0.0 or not np.all(slack >= 0.5 * margin):
        raise RuntimeError(
            "no strictly interior direction "
            f"(margin={margin:.3g}, min slack={float(slack.min()):.3g})"
        )
    return w


@dataclass(frozen=True)
class _StartResult:
    """Cold start of a fit.

    Parameters
    ----------
    theta : numpy.ndarray
        Starting coefficients.
    evaluation : _ObjectiveEvaluation
        Objective at ``theta``.
    direction : numpy.ndarray
        Interior direction ``w`` (``theta = scale * w``).
    scale : float
        Fitted ray scale.
    iterations : int
        Ray Newton iterations.
    """

    theta: np.ndarray
    evaluation: object
    direction: np.ndarray
    scale: float
    iterations: int


def _interior_start(objective, rows, /, max_iterations=60, direction=None):
    """Strictly interior cold start: the best point on the ray ``a w``.

    A basis need not contain a free quadratic (no Gaussian start), so the
    start is the NLL minimizer along the strictly interior structural
    direction ``w`` (convex in ``a > 0``; every ``a w`` is proper because the
    growth rows are homogeneous and strictly positive).  Newton steps in
    ``a``, bounded in ``log a``, with Armijo backtracking; stopped by the
    predicted-decrease criterion.

    Parameters
    ----------
    objective : _BasisPointObjective
        Objective callable.
    rows : _ConstraintRows
        Structural rows.
    max_iterations : int
        Ray Newton budget.
    direction : numpy.ndarray or None
        Interior direction to use (default: that of ``rows``).

    Returns
    -------
    _StartResult

    Raises
    ------
    RuntimeError
        If the ray minimization does not converge.
    _QuadratureFailure
        At the starting point ``w`` (an unresolved backtracking trial is
        recorded and rejected instead).
    """
    w = _interior_direction(rows) if direction is None else direction
    if not np.any(w):
        # No structural rows: a bounded support without logs, where theta = 0
        # (the uniform density) is proper and interior.
        if any(
            not np.isfinite(v)
            for v in (objective.model.request.lower, objective.model.request.upper)
        ):
            raise RuntimeError("an infinite support needs growth rows to start from")
        theta = np.zeros_like(w)
        return _StartResult(theta, objective(theta), w, 0.0, 0)
    s = 0.0
    current = objective(w)
    for iteration in range(max_iterations):
        a = np.exp(s)
        # f(s) = nll(e^s w): f' = a g.w, f'' = a g.w + a^2 w.H.w
        gw = float(current.gradient @ w)
        whw = float(w @ current.hessian @ w)
        d1 = a * gw
        d2 = a * gw + a * a * whw
        # Convergence measure: the predicted decrease of the Newton model
        # (1e-12 nats, ``_NewtonOptions.tolerance``); a gradient test alone
        # would ask for more than the NLL can resolve.
        predicted = 0.5 * d1 * d1 / d2 if d2 > 0.0 else np.inf
        if predicted <= _START_TOL:
            return _StartResult(a * w, current, w, a, iteration)
        # Newton in ``a`` (exact for the linear part of the NLL), taken as a
        # bounded step in ``log a`` so ``a`` stays positive.
        target = a - gw / whw if whw > 0.0 else 0.0
        step = math.log(target / a) if target > 0.0 else -_START_STEP
        step = float(np.clip(step, -_START_STEP, _START_STEP))
        for _ in range(40):
            try:
                trial = objective(np.exp(s + step) * w)
            except _QuadratureFailure as exc:
                # A trial the normalizer cannot resolve (a nearly flat
                # potential far from the minimizer: a z**2 face start stepped
                # 0.5 -> 1.7e-4) is a line-search rejection, recorded as not
                # routine, as for Newton trials.
                _reraise_if_debug(exc, "interior-start ray trial")
                step *= 0.5
                continue
            if trial.nll <= current.nll + 1e-4 * step * d1:
                break
            step *= 0.5
        else:
            raise RuntimeError("interior start: ray line search failed")
        s += step
        current = trial
    raise RuntimeError("interior start: ray minimization did not converge")
