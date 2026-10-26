"""Newton fit of a basis over its constraint rows, with faces and cuts.

All objectives share a coordinate preparation boundary and one compiled Newton
lifecycle. Scalar projection and interior-point quadratic kernels supply steps
through the same interface. Faces, line search, curvature exchange, rejected
trial handling and convergence rules are independent of polynomial family.
Runtime coordinate plans evaluate sufficient statistics directly and restore
canonical coefficients, faces and likelihood accounting at the output boundary.

**Faces.**  A face fixes linear functionals of ``theta`` at zero: an enabled
log amplitude, or a collapsed leading growth amplitude.  It is entered only
when its row is *active*: a positive KKT multiplier (recovered by NNLS over
the near-active rows), so a merely small leader at
an interior optimum is kept.  On a leader face the next surviving growth
term of each end becomes its growth row; an end left with no outward growth,
or two ends demanding opposite signs of the same term, adds an equality.  An
active leader whose face has no proper member means the likelihood supremum
lies on the improper boundary: the fit raises.  A face is left again when
the point is not a KKT point of a larger face (zero log amplitudes by the
KKT gain; collapsed leaders from a strictly interior convex start), which
matters for warm starts that inherit a smaller fit's face.

**Trial failures.**  An improper trial is a routine line-search rejection.
A trial the normalizer cannot resolve is recorded as a non-routine
``_QuadratureFailure`` (fatal under ``GIBBUS_DEBUG``) and the step
backtracks; at the start, an accepted iterate or the final state it
propagates.
"""

import itertools
import math
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from types import SimpleNamespace

import numpy as np
from scipy.optimize import nnls

from .._defaults import (
    CURVATURE_SEARCH_OPTIONS,
    DEBUG,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._model.basis_eval import _evaluate_basis
from .._model.basis_normalizer import (
    _ImproperPotential,
    _NormalizerOptions,
    _QuadratureFailure,
)
from ._basis_newton_kernels import ALLOCATION, BasisFaceObjective
from .basis_constraints import (
    _column_scales,
    _ConstraintRows,
    _interior_direction,
    _interior_start,
    _linear_representation,
    _same_constraint,
    _structural_rows,
    _unit_blocks,
)
from .basis_objective import _ObjectiveEvaluation
from .budget import _check_fit_time
from .conic_newton import (
    _CONVERGED,
    _UNVERIFIED,
    _apply_global_objective_certificate,
    _NewtonOptions,
    _NewtonRun,
    _preconditioned_subproblem,
)
from .curvature_search import (
    _CurvatureContext,
    _separate_curvature,
)

_ZERO_ROW = 1e-12
_ACTIVE_SLACK = 1e-7
_RELEASE_START = 1e-3


@dataclass(frozen=True)
class _Face:
    """Linear face ``E theta = 0`` of the coefficient space.

    Parameters
    ----------
    equalities : numpy.ndarray
        Rows of ``E``, shape ``(m, d)``.
    zero_logs : tuple of int
        Log columns fixed at zero.
    suppressed : tuple of tuple
        ``(block, end)`` pairs whose requested leader is fixed at zero.
    growth : numpy.ndarray
        Growth rows on the face, shape ``(g, d)``.
    """

    equalities: np.ndarray
    zero_logs: tuple
    suppressed: tuple
    growth: np.ndarray

    def basis(self, d, /):
        """Orthonormal basis ``N`` of the face, shape ``(d, k)``.

        Coordinates that no equality involves keep exact unit columns; only
        the involved block goes through the SVD.  A full SVD leaves
        rounding-level entries (~1e-16) in every column, which become
        spurious nonzeros of the face rows and wreck the row-flattening
        column scales (a scale of 2.7e10 from one 1.6e-16 entry).

        Parameters
        ----------
        d : int
            Number of joint coefficients.
        """
        if self.equalities.shape[0] == 0:
            return np.eye(d)
        involved = np.flatnonzero(np.any(self.equalities != 0.0, axis=0))
        free = np.setdiff1d(np.arange(d), involved)
        _, s, vt = np.linalg.svd(self.equalities[:, involved])
        rank = int(np.sum(s > _ZERO_ROW * max(1.0, float(s[0]))))
        null = vt[rank:].T
        out = np.zeros((d, free.size + null.shape[1]))
        out[free, np.arange(free.size)] = 1.0
        out[np.ix_(involved, np.arange(free.size, out.shape[1]))] = null
        return out


@dataclass(frozen=True)
class _Blocks:
    """Coefficient blocks: ``theta_k = theta[index[k]]`` in block ``k``'s basis.

    A single fit is one block, the identity.  A mixture has one block per
    component, each in its own span of one
    geometry (one coordinate, gates and centers); a shared column is one
    joint coordinate in every block, and coordinates in no block (mixture
    logits) are unconstrained.  Constraint rows, faces, cuts and
    curvature searches are per block, embedded in the joint coordinates.

    Parameters
    ----------
    models : tuple of _BasisModel
        The basis of each block (one geometry).
    index : tuple of numpy.ndarray
        Per block, the joint coordinate of each of its basis coefficients.
    n : int
        Number of joint coordinates.
    labels : tuple of str
        Joint coordinate labels (warm-start face matching).
    """

    models: tuple
    index: tuple
    n: int
    labels: tuple

    @classmethod
    def single(cls, model, /):
        """The one-block identity of a single fit.

        Parameters
        ----------
        model : _BasisModel
            The fit's basis.
        """
        d = model.n_coefficients
        return cls((model,), (np.arange(d),), d, tuple(c.label for c in model.columns))

    @property
    def model(self):
        """The first block's basis (geometry, support and requests shared)."""
        return self.models[0]

    @property
    def count(self):
        """Number of blocks."""
        return len(self.index)

    def part(self, k, theta, /):
        """Block ``k``'s coefficients ``theta[index[k]]``.

        Parameters
        ----------
        k : int
            Block index.
        theta : numpy.ndarray
            Joint coefficients, shape ``(n,)``.
        """
        return np.asarray(theta, dtype=np.float64)[self.index[k]]

    def embed(self, k, rows, /):
        """Rows over block ``k``'s coefficients as joint rows (shared summed).

        Parameters
        ----------
        k : int
            Block index.
        rows : array_like
            Rows over the block's coefficients, shape ``(m, d_k)`` or ``(d_k,)``.
        """
        rows = np.asarray(rows, dtype=np.float64)
        out = np.zeros(rows.shape[:-1] + (self.n,))
        np.add.at(out, (..., self.index[k]), rows)
        return out

    def range_basis(self, k, basis, /):
        """Orthonormal basis of block ``k``'s coefficients on a face ``N``.

        On the face, ``theta_k`` ranges over the span of ``N[index[k]]``
        (shape ``(d, r)``); a single fit's ``N`` itself.

        Parameters
        ----------
        k : int
            Block index.
        basis : numpy.ndarray
            Face basis ``N``, shape ``(n, r)``.
        """
        part = basis[self.index[k]]
        if (self.count == 1 and self.n == part.shape[0]) or part.shape[1] == 0:
            return part
        u, sv, _ = np.linalg.svd(part, full_matrices=False)
        rank = int(np.sum(sv > _ZERO_ROW * max(1.0, float(sv[0]) if sv.size else 1.0)))
        return u[:, :rank]

    def structural_rows(self):
        """Every block's structural rows (``_structural_rows``), embedded."""
        rows = _ConstraintRows(np.zeros((0, self.n)), ())
        for k in range(self.count):
            base = _structural_rows(self.models[k])
            for kind, run in itertools.groupby(
                zip(base.rows, base.kinds, strict=True), key=lambda item: item[1]
            ):
                rows = rows.merged(self.embed(k, np.array([r for r, _ in run])), kind)
        return rows


@dataclass(frozen=True)
class _JointSeparation:
    """Curvature searches of every block (``_CurvatureSearchResult`` each).

    ``status`` is ``verified`` when every block's is, ``violated`` when one
    block's is, else ``unverified``.
    """

    parts: tuple

    @property
    def status(self):
        """Joint shape verdict."""
        statuses = [p.status for p in self.parts]
        if all(st == "verified" for st in statuses):
            return "verified"
        if "violated" in statuses:
            return "violated"
        return "unverified"

    @property
    def feasible(self):
        """Whether ``q'' >= 0`` was verified for every block."""
        return self.status == "verified"

    @property
    def threshold(self):
        """The curvature threshold ``tau`` the verdict is relative to."""
        return max(float(p.threshold) for p in self.parts)


class _JointCurvature:
    """Curvature search data per block (``_CurvatureContext``).

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    contexts : tuple of _CurvatureContext or None
        Per block (default: the basis's own context for every block).
    """

    def __init__(self, blocks, contexts=None, /):
        """Bind per-block search data (see class docstring)."""
        self.blocks = blocks
        if contexts is None:
            cache = {}
            for m in blocks.models:
                cache.setdefault(id(m), _CurvatureContext(m))
            contexts = tuple(cache[id(m)] for m in blocks.models)
        self.contexts = tuple(contexts)

    def on_face(self, basis, /):
        """The same search for points of a face ``theta = N phi``.

        Parameters
        ----------
        basis : numpy.ndarray
            Face basis ``N``, shape ``(n, r)``.
        """
        blocks = self.blocks
        return _JointCurvature(
            blocks,
            tuple(
                c.on_face(blocks.range_basis(k, basis))
                for k, c in enumerate(self.contexts)
            ),
        )

    def separate(self, theta, options, /):
        """Search every block at joint coefficients ``theta``.

        Parameters
        ----------
        theta : numpy.ndarray
            Joint coefficients, shape ``(n,)``.
        options : _CurvatureSearchOptions
            Search settings.
        """
        return _JointSeparation(
            tuple(
                _separate_curvature(c, self.blocks.part(k, theta), options)
                for k, c in enumerate(self.contexts)
            )
        )


@dataclass(frozen=True)
class _BasisNewtonResult:
    """Outcome of one basis Newton fit.

    Parameters
    ----------
    status : str
        ``converged``, ``converged_approximately``, ``non_descent``,
        ``line_search_failed``, ``iteration_limit`` or ``unverified`` (a
        converged point whose shape the curvature search did not verify
        within ``tau``: its verdict is ``violated`` or ``unverified``).
    params : numpy.ndarray
        Fitted coefficients.
    objective_value : float
        NLL per unit weight.
    evaluation : _ObjectiveEvaluation
        Objective at ``params`` (full coordinates).
    rows : _ConstraintRows
        All retained constraint rows.
    face : _Face
        Face the fit lies on.
    newton_iterations, objective_evaluations, subproblem_iterations : int
        Work counters.
    exchanges : int
        Cut exchange rounds.
    final_decrease_bound : float
        Predicted-decrease certificate of the last Newton run.
    final_separation : _JointSeparation
        Shape evidence at ``params`` per block (``verified``, ``violated``
        or ``unverified``).
    global_gap : float
        NLL minus the nonparametric interval bound (``lower_bound``; run
        only until the certificate is decided, so a large gap is an upper
        estimate) when the fit's shape is verified, else ``nan``.
    """

    status: str
    params: np.ndarray
    objective_value: float
    evaluation: object
    rows: object
    face: object
    newton_iterations: int
    objective_evaluations: int
    subproblem_iterations: int
    exchanges: int
    final_decrease_bound: float
    final_separation: object
    global_gap: float = math.nan
    coordinate_system: str = "canonical"


# Tests: when a list, every point a compiled face run normalizes is appended
# (full coordinates).
_TRACE = None
_TRACE_CAPACITY = 4096

# Solver statuses of ``_face_newton_loop``; 6 is the gate's stop.
_RUN_STATUS = {
    1: "converged",
    2: "converged_approximately",
    3: "non_descent",
    4: "line_search_failed",
    5: "iteration_limit",
    6: "cut_required",
}


class _FaceObjective:
    """Objective restricted to a face, ``theta = N phi``.

    Calling it evaluates in Python (starts, restored points, final states);
    ``run`` makes one Newton run in the fused loop over the compiled face
    objective (``_basis_newton_kernels``).  Inside a run every trial point
    is first checked by the global curvature search: the normalizer
    only ever integrates log-concave potentials. A non-convex trial is a
    routine line-search
    rejection; if it reveals new cuts they are queued, the line search
    continues to a convex trial, and the next trial after that ends the run
    with its accepted iterate (status ``cut_required``), so every exchange
    round keeps one step of progress and re-solves with the new cuts.

    Parameters
    ----------
    objective : _BasisPointObjective or _BasisIntervalObjective
        Full objective (its compiled form evaluates the same objective).
    basis : numpy.ndarray
        Face basis ``N``, shape ``(d, k)``.
    curvature : _CurvatureContext or None
        Curvature search of the face (required by ``run``).
    options : _CurvatureSearchOptions or None
        Search settings.
    """

    def __init__(self, objective, basis, /, curvature=None, options=None):
        """Bind the objective to a face basis (see class docstring)."""
        self.objective = objective
        self.model = getattr(objective, "model", None)
        self.basis = basis
        self.curvature = curvature
        self.curvature_options = (
            CURVATURE_SEARCH_OPTIONS if options is None else options
        )
        self._calls = 0
        self._compiled = None

    @property
    def evaluations(self):
        """Objective evaluations, Python and compiled."""
        compiled = 0 if self._compiled is None else self._compiled.evaluations
        return self._calls + compiled

    def full(self, phi, /):
        """Map face coordinates to full coefficients.

        Parameters
        ----------
        phi : numpy.ndarray
            Face coordinates, shape ``(k,)``.
        """
        return self.basis @ np.asarray(phi, dtype=np.float64)

    def restrict(self, evaluation, /):
        """Restrict a full evaluation to the face.

        Parameters
        ----------
        evaluation : _ObjectiveEvaluation
            Full-coordinate evaluation.
        """
        n = self.basis
        fisher = n.T @ evaluation.fisher @ n
        missing = n.T @ evaluation.missing_information @ n
        # The metric of the restricted F and C (a reflection does not commute
        # with restriction); F itself for point data.
        metric, smallest = self.objective.metric(fisher, missing)
        return _ObjectiveEvaluation(
            nll=evaluation.nll,
            gradient=n.T @ evaluation.gradient,
            hessian=metric,
            fisher=fisher,
            missing_information=missing,
            model_partial_means=n.T @ evaluation.model_partial_means,
            state=evaluation,
            smallest_curvature=smallest,
        )

    def __call__(self, phi, /):
        """Evaluate on the face (failures propagate).

        Parameters
        ----------
        phi : numpy.ndarray
            Face coordinates.
        """
        self._calls += 1
        return self.restrict(self.objective(self.full(phi)))

    def compiled(self):
        """The compiled face objective (built once)."""
        if self._compiled is None:
            objective, joint = self.objective, self.curvature
            blocks = joint.blocks
            kind, data, nblocks = objective.compiled_spec()
            nopts = objective.options or _NormalizerOptions()
            sblocks = [
                (
                    c.tails["lower"].kernel if "lower" in c.tails else None,
                    c.tails["upper"].kernel if "upper" in c.tails else None,
                    c.search,
                )
                for c in joint.contexts
            ]
            self._compiled = BasisFaceObjective(
                [m.kernel for m in blocks.models],
                self.basis,
                blocks.index,
                (nblocks, nopts),
                (sblocks, self.curvature_options),
                kind,
                data,
                trace=0 if _TRACE is None else _TRACE_CAPACITY,
            )
        return self._compiled

    def run(self, representation, phi, blocks, evaluation, options, known):
        """One gated Newton run in the fused compiled loop.

        Parameters
        ----------
        representation : _ConicRepresentation
            Cone description (scalar inequality rows in face coordinates).
        phi : numpy.ndarray
            Feasible start.
        blocks : tuple of numpy.ndarray
            Start Gram blocks.
        evaluation : _ObjectiveEvaluation
            Face evaluation at ``phi``.
        options : _NewtonOptions
            Newton policy.
        known : _ConstraintRows
            Retained rows (a trial's cuts are queued only if one is new).

        Returns
        -------
        tuple
            ``(run, queued)``: the ``_NewtonRun`` (its evaluation without a
            normalizer state unless the run did not move) and the cuts of
            its first violating trial
            (``[]`` or ``[(_TrialCuts, theta)]``).
        """
        _check_fit_time("Newton run")
        compiled = self.compiled()
        coefficient_blocks = self.curvature.blocks
        a_packed, sizes, a_offsets, q_offsets = representation.packed
        compiled.begin(known.rows, stop_on_failure=DEBUG)
        try:
            (
                code,
                x,
                packed,
                dual,
                its,
                evals,
                subits,
                bound,
                nll,
                grad,
                hess,
                mean,
                fisher,
                missing,
            ) = compiled.newton(
                phi,
                float(evaluation.nll),
                evaluation.gradient,
                evaluation.hessian,
                representation.b_matrix,
                a_packed,
                sizes,
                a_offsets,
                q_offsets,
                representation.reference_dual,
                representation.row_degrees,
                representation.pack_blocks(blocks),
                # Jacobi scaling of each subproblem's Hessian: the
                # row-flattening scales serve feasibility measurement only
                # (``_polish``).
                None,
                options,
            )
        finally:
            compiled.end()
            if _TRACE is not None:
                _TRACE.extend(compiled.traced())
        _check_fit_time("Newton run")
        self._ledger(compiled.failures())
        queued = []
        pending = compiled.pending()
        if pending is not None:
            block, rows, n_points, theta = pending
            # The gate stops at the first rejected block: search every block
            # at the violating trial so all of its cuts are queued together.
            separation = self.curvature.separate(theta, self.curvature_options)
            queued = [
                (_TrialCuts(k, part.cut_points, part.cut_rows), theta)
                for k, part in enumerate(separation.parts)
                if part.status != "verified" and len(part.cut_rows)
            ]
            if not queued and not separation.feasible:
                queued = [(_TrialCuts(block, np.zeros(n_points), rows), theta)]
        final = evaluation
        if not np.array_equal(x, phi):
            # The loop's accepted iterate, as the compiled objective evaluated
            # it (face coordinates; no state: ``_solve_face`` evaluates the
            # answer it returns).
            final = _ObjectiveEvaluation(
                nll=float(nll),
                gradient=grad,
                hessian=hess,
                fisher=fisher,
                missing_information=missing,
                model_partial_means=self.basis.T
                @ sum(coefficient_blocks.embed(b, m) for b, m in enumerate(mean)),
                state=None,
            )
        run = _NewtonRun(
            status=_RUN_STATUS[code],
            params=x,
            blocks=representation.unpack_blocks(packed),
            dual=dual,
            evaluation=final,
            iterations=int(its),
            evaluations=int(evals),
            subproblem_iterations=int(subits),
            decrease_bound=float(bound),
        )
        return run, queued

    def _ledger(self, failures, /):
        """Record the run's unresolved trials, fatal under ``GIBBUS_DEBUG``.

        Parameters
        ----------
        failures : tuple
            ``(count, status, side, diag, theta, row, block)`` of the compiled run's
            unresolved trials.
        """
        if failures is None:
            return
        count, status, side, diag, theta, row, block = failures
        if status == ALLOCATION:
            exc = MemoryError("Newton trial workspace")
        else:
            exc = self.objective.trial_failure(status, side, diag, theta, row, block)
        for _ in range(count):
            # Not routine: the primary integrator could not resolve a trial
            # point; the compiled line search backtracked on it.
            _reraise_if_debug(exc, "Newton trial evaluation")


@dataclass(frozen=True)
class _TrialCuts:
    """Cuts queued at a violating trial of one block (separation-like record).

    ``cut_rows[:cut_points.size]`` are point cuts, the rest tail or bound
    rows (block coordinates).
    """

    block: int
    cut_points: np.ndarray
    cut_rows: np.ndarray


def _merge_part(rows, blocks, k, part, /):
    """``rows`` with block ``k``'s search cut and tail rows merged (embedded).

    Parameters
    ----------
    rows : _ConstraintRows
        Retained rows (joint coordinates).
    blocks : _Blocks
        Coefficient blocks.
    k : int
        Block the search ran on.
    part : _Separation or _TrialCuts
        The block's cut and tail rows.
    """
    m = part.cut_points.size
    if m:
        rows = rows.merged(blocks.embed(k, part.cut_rows[:m]), "cut")
    if part.cut_rows.shape[0] > m:
        rows = rows.merged(blocks.embed(k, part.cut_rows[m:]), "tail")
    return rows


def _merge_separation(rows, blocks, separation, /):
    """Return ``rows`` with every block's cut and tail rows merged.

    Parameters
    ----------
    rows : _ConstraintRows
        Retained rows (joint coordinates).
    blocks : _Blocks
        Coefficient blocks.
    separation : _JointSeparation
        Search result of every block.
    """
    for k, part in enumerate(separation.parts):
        rows = _merge_part(rows, blocks, k, part)
    return rows


def _merge_face_part(problem, face, k, part, /):
    """Retain global point cuts and tail cuts valid on ``face`` only.

    A tail atom exposed after the face suppresses stronger atoms is not a
    global inequality: convex points off that face may have a positive
    stronger atom and a negative coefficient for the exposed one.

    Parameters
    ----------
    problem : _Problem
        Joint rows and scoped tail-cut ledger.
    face : _Face
        Face on which the separator found the cuts.
    k : int
        Component block.
    part : _CurvatureSearchResult or _TrialCuts
        Curvature cuts, with point rows preceding tail rows.
    """
    if not face.equalities.size:
        problem.rows = _merge_part(problem.rows, problem.blocks, k, part)
        return
    m = part.cut_points.size
    if m:
        point_rows = problem.blocks.embed(k, part.cut_rows[:m])
        problem.rows = problem.rows.merged(point_rows, "cut")
        # The full row at a far point can be dominated by a column fixed
        # exactly to zero on this face (e.g. exp(500) vs polynomial terms).
        # Normalizing it globally then projecting drops the useful active
        # entries below _ZERO_ROW.  Preserve the *face-restricted* point
        # inequality separately, without extending it to released faces.
        basis = face.basis(problem.blocks.n)
        for raw in point_rows:
            restricted = raw @ basis
            full_peak = float(np.max(np.abs(raw)))
            active_peak = float(np.max(np.abs(restricted))) if restricted.size else 0.0
            if not (active_peak > 0.0 and active_peak < _ZERO_ROW * full_peak):
                continue
            # Only reconstruct an underflowed point row when the columns
            # dominating its full representation vanish *exactly* on the
            # face.  An approximate null-space cancellation at scale 1e200
            # cannot recover a reliable 1e-12 relative component.
            dominating = np.abs(raw) >= _ZERO_ROW * full_peak
            if np.any(basis[dominating] != 0.0):
                continue
            local = basis @ (restricted / active_peak)
            if any(
                np.array_equal(scope, face.equalities)
                and _same_constraint(local[None, :], prior[None, :])[0, 0]
                for scope, prior in getattr(problem, "scoped_points", ())
            ):
                continue
            if not hasattr(problem, "scoped_points"):
                problem.scoped_points = []
            problem.scoped_points.append((face.equalities.copy(), local))
    if part.cut_rows.shape[0] == m:
        return
    tail = problem.blocks.embed(k, part.cut_rows[m:])
    unit = _ConstraintRows(np.zeros((0, problem.blocks.n)), ()).merged(tail, "tail")
    for row in unit.rows:
        if any(
            np.array_equal(scope, face.equalities)
            and _same_constraint(row[None, :], existing[None, :])[0, 0]
            for scope, existing in problem.scoped_tails
        ):
            continue
        problem.scoped_tails.append((face.equalities.copy(), row.copy()))


def _merge_face_separation(problem, face, separation, /):
    """Merge all component cuts with their originating face's scope.

    Parameters
    ----------
    problem : _Problem
        Joint rows and scoped tail-cut ledger.
    face : _Face
        Source face for conditional tail cuts.
    separation : _JointSeparation
        Component curvature searches.
    """
    for k, part in enumerate(separation.parts):
        _merge_face_part(problem, face, k, part)


def _proper_capable(term, /):
    """Whether an asymptotic term can carry outward growth (``>=`` linear).

    Parameters
    ----------
    term : _AsymptoticTerm
        One surviving asymptotic term of a side.
    """
    kind, power = term.order[0], term.order[1]
    return kind == 2 or (kind == 1 and power >= 1.0)


def _face_growth(blocks, equalities, /):
    """Growth rows and forced equalities of a face, or ``None`` if improper.

    For each infinite end of each block the first asymptotic term not
    identically zero on the face supplies its growth row.  Two ends demanding
    opposite signs of the same functional force it to zero (one more
    equality); an end whose first surviving term cannot grow outward leaves
    no proper member.

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    equalities : numpy.ndarray
        Face equalities, shape ``(m, n)``.

    Returns
    -------
    tuple of numpy.ndarray or None
        ``(equalities, growth)`` after forced equalities, else ``None``.
    """
    d = blocks.n
    eq = np.asarray(equalities, dtype=np.float64).reshape(-1, d)
    for _ in range(2 * d + 2):
        n = _Face(eq, (), (), np.zeros((0, d))).basis(d)
        rows = []
        for k in range(blocks.count):
            for side in blocks.models[k].asymptotics:
                for term in side.terms:
                    row = blocks.embed(k, [float(v) for v in term.row_exact])
                    restricted = row @ n
                    if np.linalg.norm(restricted) <= _ZERO_ROW * np.linalg.norm(row):
                        continue
                    if not _proper_capable(term):
                        return None
                    rows.append(n @ restricted)
                    break
                else:
                    return None
        forced = None
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                cos = float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
                if cos <= -1.0 + 1e-12:
                    forced = a
        if forced is None:
            unique = []
            for row in rows:
                unit = row / np.linalg.norm(row)
                if not any(float(unit @ u) >= 1.0 - 1e-12 for u in unique):
                    unique.append(unit)
            return eq, np.array(unique).reshape(-1, d)
        eq = np.vstack([eq, forced])
    return None


def _suppressed_ends(blocks, equalities, /):
    """``(block, end)`` pairs whose requested leading functional vanishes on a face.

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    equalities : numpy.ndarray
        Face equalities.
    """
    d = blocks.n
    basis = _Face(equalities, (), (), np.zeros((0, d))).basis(d)
    out = []
    for k in range(blocks.count):
        for side in blocks.models[k].asymptotics:
            lead = blocks.embed(k, [float(v) for v in side.leading.row_exact])
            if np.linalg.norm(lead @ basis) <= _ZERO_ROW * np.linalg.norm(lead):
                out.append((k, side.side))
    return tuple(out)


def _restricted_rows(rows, face, basis, /, scoped_tails=(), scoped_points=()):
    """Constraint rows of a face in face coordinates (zero rows dropped).

    Parameters
    ----------
    rows : _ConstraintRows
        Full-coordinate rows (log bounds, cuts, tails; growth replaced by the
        face's own).
    face : _Face
        The face.
    basis : numpy.ndarray
        Face basis ``N``.
    scoped_tails : sequence of tuple, optional
        ``(equalities, row)`` for a tail cut valid only while those
        equalities hold. Cuts from a different face are not enforced.
    scoped_points : sequence of tuple, optional
        ``(equalities, row)`` for numerically rescaled finite-point cuts
        valid on the generating face and its subfaces.
    """
    kept = [i for i, kind in enumerate(rows.kinds) if kind != "growth"]
    kinds = [rows.kinds[i] for i in kept] + ["growth"] * len(face.growth)
    restricted = (
        np.vstack([rows.rows[kept], np.reshape(face.growth, (-1, basis.shape[0]))])
        @ basis
    )
    for kind, records in (("tail", scoped_tails), ("cut", scoped_points)):
        for equalities, row in records:
            # Enforce the row only where its source equalities hold.
            projected = equalities @ basis
            norms = np.linalg.norm(equalities, axis=1)
            if np.all(np.linalg.norm(projected, axis=1) <= _ZERO_ROW * norms):
                restricted = np.vstack([restricted, row @ basis])
                kinds.append(kind)
    nonzero = np.linalg.norm(restricted, axis=1) > _ZERO_ROW
    out = _ConstraintRows(np.zeros((0, basis.shape[1])), ())
    # Merged in runs of one kind, in order (the same rows as one at a time).
    for kind, run in itertools.groupby(
        (i for i in range(len(kinds)) if nonzero[i]), key=kinds.__getitem__
    ):
        out = out.merged(restricted[list(run)], kind)
    return out


def _active_rows(problem, face, basis, /):
    """Constraint rows valid on the current face only.

    Parameters
    ----------
    problem : _Problem
        Shared global rows and conditional tail-cut ledger.
    face : _Face
        Active face.
    basis : numpy.ndarray
        Orthonormal face basis.
    """
    return _restricted_rows(
        problem.rows,
        face,
        basis,
        problem.scoped_tails,
        getattr(problem, "scoped_points", ()),
    )


def _multipliers(rows, gradient, theta, /, hessian=None):
    """KKT multipliers of the near-active inequality rows by NNLS.

    Parameters
    ----------
    rows : _ConstraintRows
        Rows in the coordinates of ``gradient``/``theta``.
    gradient : numpy.ndarray
        NLL gradient.
    theta : numpy.ndarray
        Point.

    Returns
    -------
    numpy.ndarray
        One multiplier per row (zero for inactive rows).
    """
    scale = np.ones_like(theta)
    if hessian is not None:
        diagonal = np.diag(hessian)
        resolved = np.isfinite(diagonal) & (diagonal > 0.0)
        scale[resolved] = 1.0 / np.sqrt(diagonal[resolved])
    design = rows.rows * scale
    norms = np.linalg.norm(design, axis=1)
    norms = np.where(norms > 0.0, norms, 1.0)
    slack = rows.slack(theta) / norms
    tol = _ACTIVE_SLACK * max(1.0, float(np.linalg.norm(theta / scale)))
    active = np.flatnonzero(slack <= tol)
    lam = np.zeros(rows.size)
    if active.size:
        dual, _ = nnls(
            (design[active] / norms[active, None]).T,
            gradient * scale,
            maxiter=50 * rows.size,
        )
        lam[active] = dual / norms[active]
    return lam


def _polish(rows, phi, scales, /):
    """Restore exact feasibility of rows the solver met only to its tolerance.

    An interior-point endpoint reconstructed from its Gram certificate
    satisfies its rows by construction.  With more ``1 x 1`` rows than
    coefficients the infeasible-start solver meets them only to its primal
    residual (``~1e-12`` relative), which is a visible ``q'' < 0`` at an
    active cut.  The minimum-norm correction onto the violated rows restores
    them to rounding; it is applied only when it is that small.  Violation,
    correction and budget are measured in the solver's row-flattening column
    scales, where every row is resolvable: a far exponential-tail cut is
    ``e_exp`` plus polynomial entries ~1e-17 of it, so an unscaled test
    accepts a violation of ``4e-20`` that is ``q'' = -0.03`` there.

    Parameters
    ----------
    rows : _ConstraintRows
        Unit-norm rows.
    phi : numpy.ndarray
        Point from the solver.
    scales : numpy.ndarray
        Subproblem column scales ``D`` (``phi = D psi``).

    Returns
    -------
    numpy.ndarray
        ``phi`` itself when nothing is violated or the correction would not be
        a polish; otherwise the corrected point.
    """
    a = rows.rows * scales[None, :]
    a = a / np.linalg.norm(a, axis=1)[:, None]
    psi0 = np.asarray(phi, dtype=np.float64) / scales
    out = psi0.copy()
    norm = max(1.0, float(np.linalg.norm(psi0)))
    floor = 8.0 * np.finfo(np.float64).eps * norm
    budget = 1e-8 * norm
    # Log amplitudes have an exact domain boundary in the normalizer. A
    # curvature correction must not leave one slightly negative even when
    # its scaled residual is below the general row-roundoff floor.
    exact = np.array([kind == "log" for kind in rows.kinds], dtype=bool)
    moved = False
    for _ in range(4):
        slack = a @ out
        bad = slack < floor
        violated = (slack < -floor) | (exact & (slack < 0.0))
        if not np.any(violated):
            return out * scales if moved else phi
        # Correct onto the binding rows with a margin of the floor, so the
        # unscaled rows hold as well.
        step = np.linalg.lstsq(a[bad], slack[bad] - 2.0 * floor, rcond=None)[0]
        if np.linalg.norm(out - step - psi0) > budget:
            return phi
        out = out - step
        moved = True
    return phi


def _checked_row_polish(objective, rows, phi, evaluation, scales, /):
    """Keep a row repair only when its actual likelihood does not worsen.

    A coefficient-small correction can be large in the Fisher metric.
    Returning a changed point requires a fresh optimization certificate;
    the caller must not carry the incumbent's Newton bound to that point.
    """
    candidate = _polish(rows, phi, scales)
    if candidate is phi:
        return phi, evaluation, False
    try:
        trial = objective(candidate)
    except (_ImproperPotential, _QuadratureFailure, ValueError, OverflowError) as exc:
        _reraise_if_debug(exc, "row feasibility polish", routine=True)
        return phi, evaluation, False
    roundoff = 8.0 * np.finfo(float).eps * max(1.0, abs(evaluation.nll))
    if not math.isfinite(trial.nll) or trial.nll > evaluation.nll + roundoff:
        return phi, evaluation, False
    return candidate, trial, True


@dataclass
class _Problem:
    """Mutable bookkeeping of one fit (rows grow, counters accumulate)."""

    objective: object
    blocks: object
    curvature: object
    curvature_options: object
    options: object
    max_exchanges: int
    rows: object = None
    scoped_tails: list = field(default_factory=list)
    scoped_points: list = field(default_factory=list)
    newton_iterations: int = 0
    evaluations: int = 0
    subproblem_iterations: int = 0
    exchanges: int = 0


def _toward_anchor(rows, phi, anchor, scales, /, *, objective=None, tau=None):
    """Restore new rows exactly by moving toward a globally convex anchor.

    A point whose curvature was verified (``H >= -tau`` everywhere) satisfies
    every cut found later up to that threshold (each cut is ``q''(x*) >= 0``
    at some point), so the segment from the current point to that anchor
    reaches feasibility, at worst at the anchor's own level.
    The smallest feasible step is an exact ratio test because rows are
    linear; the NLL is convex along the segment, so the restored point is no
    worse than the anchor.  A Fisher-metric projection, a Newton-model or a
    fixed-metric nearest point would all move far along poorly determined
    directions.

    Parameters
    ----------
    rows : _ConstraintRows
        Face rows.
    phi : numpy.ndarray
        Current point (violating some rows).
    anchor : numpy.ndarray
        Globally convex point of the face.
    scales : numpy.ndarray
        Subproblem column scales (for the final polish).
    objective : callable or None
        When supplied, assess a final rounding repair in the actual objective.
    tau : float or None
        Curvature threshold the anchor was verified within (default: the
        search default).

    Returns
    -------
    numpy.ndarray
        The restored point (feasible, possibly on active rows).

    Raises
    ------
    RuntimeError
        If the anchor violates a row far beyond its verification threshold
        (an internal inconsistency).
    """
    tau = CURVATURE_SEARCH_OPTIONS.tau if tau is None else tau
    slack = rows.slack(phi)
    anchor_slack = rows.slack(anchor)
    # The anchor is verified within tau (H = q'' / ||pieces|| >= -tau), so a
    # unit cut row may hold it at -tau ||pieces|| / ||row||, of order
    # tau ||anchor||; each row is restored to the anchor's own level at
    # worst.  A violation far beyond that is an inconsistency.
    if np.min(anchor_slack) < -1e3 * tau * max(1.0, float(np.linalg.norm(anchor))):
        raise RuntimeError("internal error: a convex anchor violates a row")
    target = np.minimum(anchor_slack, 0.0)
    bad = slack < target
    if not np.any(bad):
        return phi
    t = np.max((target[bad] - slack[bad]) / (anchor_slack[bad] - slack[bad]))
    t = min(1.0, float(t))
    out = phi + t * (anchor - phi)
    # The ratio test lands on the binding rows up to rounding.
    if np.min(rows.slack(out)) < 0.0:
        if objective is None:
            out = _polish(rows, out, scales)
        else:
            evaluation = objective(out)
            out, _, _ = _checked_row_polish(objective, rows, out, evaluation, scales)
    return out


def _retire_inactive_cuts(rows, ages, theta, options, /):
    """Retire point cuts after consecutive slack exchange iterates.

    Structural rows, tail rows, and face-scoped records are retained. The
    global curvature gate can regenerate retired point cuts before any
    nonconvex trial reaches the normalizer. Activity uses a relative
    cancellation margin in each row, with an absolute floor for small
    coefficients; it does not change the shape or convergence tolerance.

    Parameters
    ----------
    rows : _ConstraintRows
        Global unit rows, with new rows appended after the tracked prefix.
    ages : numpy.ndarray
        Consecutive inactive exchange counts for that prefix, shape ``(m,)``.
    theta : numpy.ndarray
        Current full-coordinate coefficients, shape ``(d,)``.
    options : _NewtonOptions
        Cut retention and activity policy; zero retention disables retirement.

    Returns
    -------
    retained : _ConstraintRows
        Rows still enforced by the next Newton relaxation.
    ages : numpy.ndarray
        Updated counts in retained-row order.
    """
    if not options.cut_retention_exchanges:
        return rows, ages
    ages = np.pad(ages, (0, rows.size - ages.size))
    slack = rows.slack(theta)
    magnitude = np.abs(rows.rows) @ np.abs(theta)
    inactive = (np.array(rows.kinds) == "cut") & (
        slack > options.cut_activity_margin * np.maximum(1.0, magnitude)
    )
    ages = np.where(inactive, ages + 1, 0)
    keep = ages <= options.cut_retention_exchanges
    if np.all(keep):
        return rows, ages
    return (
        _ConstraintRows(
            rows.rows[keep],
            tuple(kind for kind, kept in zip(rows.kinds, keep, strict=True) if kept),
        ),
        ages[keep],
    )


def _solve_face(problem, face, theta, /, *, interior_lift=False, restore_start=False):
    """Newton with cut exchange on one face from ``theta``.

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping.
    face : _Face
        Face.
    theta : numpy.ndarray
        Start (full coordinates); projected onto the face and its rows.
    interior_lift : bool
        A feasible, globally convex start without strict growth is moved
        into the interior along the face's convex direction instead of being
        replaced by the face's cold start (see ``_solve_basis``).
    restore_start : bool
        A start with strict growth that violates some of the face's rows is
        restored toward the face's cold start (``_toward_anchor``) instead of
        being replaced by it (``_candidate_face``).

    Returns
    -------
    tuple
        ``(status, theta, evaluation, bound, separation)``.
    """
    blocks = problem.blocks
    d = blocks.n
    basis = face.basis(d)
    if face.equalities.shape[0]:
        # Functionals fixed at zero by the face are exactly zero on it: the
        # properness and tail sign tests use face-restricted atoms.
        objective = problem.objective.on_face(basis)
        curvature = problem.curvature.on_face(basis)
    else:
        objective, curvature = problem.objective, problem.curvature
    face_obj = _FaceObjective(objective, basis, curvature, problem.curvature_options)
    phi = basis.T @ theta
    start_rows = _active_rows(problem, face, basis)
    floor = 8.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(phi)))
    slack = start_rows.slack(phi)
    growth = np.array([k == "growth" for k in start_rows.kinds], dtype=bool)
    if (
        interior_lift
        and start_rows.size
        and np.min(slack) >= -floor
        and np.any(slack[growth] <= 0.0)
        and curvature.separate(basis @ phi, problem.curvature_options).status
        == "verified"
    ):
        # A lifted smaller fit (feasible and globally convex, its new leader
        # zero): moved into the interior along the face's verified convex
        # direction, as a released leader, giving up at most
        # ``options.lift_nll_slack``, so it stays near the smaller answer.
        w = _face_direction(problem, face, basis, curvature)
        t = _RELEASE_START * max(1.0, float(np.linalg.norm(phi))) / np.linalg.norm(w)
        base_nll = float(face_obj(phi).nll)
        allowed = problem.options.lift_nll_slack
        for _ in range(int(problem.options.lift_halvings)):
            if float(face_obj(phi + t * w).nll) <= base_nll + allowed:
                break
            t *= 0.5
        phi = phi + t * w
        start_rows = _active_rows(problem, face, basis)
        slack = start_rows.slack(phi)
        growth = np.array([k == "growth" for k in start_rows.kinds], dtype=bool)
    anchor = None
    if (
        restore_start
        and start_rows.size
        and np.min(slack) < -floor
        and np.all(slack[growth] > 0.0)
    ):
        # A proper start a hair outside the face's rows (an incumbent
        # projected onto a leader face: 1e-11 against a floor of 1e-14): the
        # cold start sat 1.5-3 nats above it, and from there 30 exchanges of
        # 60 Newton iterations each crawled 0.03-0.06 nats per run without
        # converging (105 s per candidate).  The NLL is convex along
        # the segment to the anchor, so the restored start stays near phi.
        anchor = _face_anchor(problem, face, basis, face_obj, curvature)
        start_rows = _active_rows(problem, face, basis)
        scales = _column_scales(start_rows, face_obj(phi).hessian)
        phi = _toward_anchor(
            start_rows,
            phi,
            anchor,
            scales,
            objective=face_obj,
            tau=problem.curvature_options.tau,
        )
    elif start_rows.size and (np.min(slack) < -floor or np.any(slack[growth] <= 0.0)):
        # A start outside the face's rows or without strict outward growth
        # (a cold start projected onto a leader face can lose its growth,
        # hence properness): the face's own convex cold start.
        phi = _face_anchor(problem, face, basis, face_obj, curvature)
    evaluation = face_obj(phi)
    status, bound = "iteration_limit", np.inf
    separation = None
    queued = []
    repair_pending = False
    cut_ages = np.zeros(problem.rows.size, dtype=np.int64)
    for exchange in range(problem.max_exchanges + 1):
        _check_fit_time("curvature exchange")
        theta = face_obj.full(phi)
        separation = curvature.separate(theta, problem.curvature_options)
        if separation.status == "verified":
            anchor = phi.copy()
        before = (
            problem.rows.size + len(problem.scoped_tails) + len(problem.scoped_points)
        )
        # Cuts found at the trial that ended the last run, then at the
        # accepted iterate itself (all valid: q'' >= 0 at a point).
        for trial_sep, _ in queued:
            _merge_face_part(problem, face, trial_sep.block, trial_sep)
        _merge_face_separation(problem, face, separation)
        added = (
            problem.rows.size
            + len(problem.scoped_tails)
            + len(problem.scoped_points)
            - before
        )
        must_recertify = repair_pending
        repair_pending = False
        if exchange > 0 and added == 0 and not must_recertify:
            # No new inequality can improve the relaxation. Retain the
            # certified classification at its unchanged numerical floor.
            if (
                separation.status == "verified"
                and bound <= problem.options.certified_tolerance
            ):
                status = "converged"
            break
        # A run stopped for a cut at a feasible point whose model bound over
        # the relaxation is already certified: the kernel's rule for a run
        # that cannot step.  Cutting planes approach a degenerate
        # contact at a linear rate (a log-concave fit of multimodal data:
        # thirty exchanges at a bound below 1e-10), so the bound, not the
        # exchange, ends the fit.
        feasible = exchange > 0 and separation.status == "verified"
        if feasible and status not in _CONVERGED and bound <= problem.options.tolerance:
            status = "converged"
            break
        if exchange == problem.max_exchanges:
            if feasible and bound <= problem.options.accuracy_floor:
                status = (
                    "converged"
                    if bound <= problem.options.certified_tolerance
                    else "converged_approximately"
                )
            else:
                status = "iteration_limit"
            break
        problem.exchanges += 1
        problem.rows, cut_ages = _retire_inactive_cuts(
            problem.rows, cut_ages, theta, problem.options
        )
        rows = _active_rows(problem, face, basis)
        if rows.size == 0:
            # No structural row and no cut yet (a bounded support without
            # logs, started at q'' = 0): any curvature cut is a valid row
            # (q'' >= 0 everywhere); take the one at the center.
            problem.rows = problem.rows.merged(_center_cut(blocks), "cut")
            rows = _active_rows(problem, face, basis)
        if rows.size == 0:
            raise RuntimeError("a face has no constraint rows")
        # Newton constraints are exactly the coefficient-linear rows for
        # q''(x) >= 0 at the known cut locations, plus structural growth/log
        # inequalities.  Strict positivity is required in the interior-point
        # subproblem, not of the final shape-constrained optimum.
        representation = _linear_representation(rows)
        scales = _column_scales(rows, evaluation.hessian)
        slack = rows.slack(phi)
        floor = 8.0 * np.finfo(np.float64).eps * max(1.0, float(np.linalg.norm(phi)))
        if np.min(slack) < -floor:
            if anchor is None:
                # The face's own convex cold start (verified, ray-optimal).
                anchor = _face_anchor(problem, face, basis, face_obj, curvature)
                rows = _active_rows(problem, face, basis)
                representation = _linear_representation(rows)
                scales = _column_scales(rows, evaluation.hessian)
                slack = rows.slack(phi)
            phi = _toward_anchor(
                rows,
                phi,
                anchor,
                scales,
                objective=face_obj,
                tau=problem.curvature_options.tau,
            )
            evaluation = face_obj(phi)
            slack = rows.slack(phi)
        # All Gram blocks are scalar slacks.  A boundary or slightly
        # infeasible start uses strictly positive reference blocks; the
        # interior-point subproblem handles restoration as before.
        gram = (
            tuple(np.array([[v]]) for v in slack)
            if np.all(slack > 0.0)
            else _unit_blocks(rows)
        )
        run, queued = face_obj.run(
            representation,
            phi,
            gram,
            evaluation,
            problem.options,
            problem.rows,
        )
        problem.newton_iterations += run.iterations
        problem.subproblem_iterations += run.subproblem_iterations
        phi, evaluation = run.params, run.evaluation
        status, bound = run.status, run.decrease_bound
        phi, evaluation, repair_pending = _checked_row_polish(
            face_obj, rows, phi, evaluation, scales
        )
        if repair_pending:
            status, bound = "iteration_limit", np.inf
            if run.status in _CONVERGED:
                # Certify the repaired point without taking another step:
                # another full Newton run can return the same boundary
                # rounding error and repeat an otherwise harmless repair.
                slack = rows.slack(phi)
                gram = (
                    tuple(np.array([[v]]) for v in slack)
                    if np.all(slack > 0.0)
                    else _unit_blocks(rows)
                )
                sub, _, _, model = _preconditioned_subproblem(
                    evaluation.hessian, evaluation.gradient, phi, representation, gram
                )
                problem.subproblem_iterations += sub.iterations
                bound = max(0.0, -model) + max(0.0, sub.gap)
                if bound <= problem.options.accuracy_floor:
                    status = (
                        "converged"
                        if bound <= problem.options.certified_tolerance
                        else "converged_approximately"
                    )
                    repair_pending = False
        # A run that did not converge continues the exchange: it may have
        # failed in a region the missing cuts exclude (a degree-8 fit with 7
        # rows reached a bimodal q, where the subproblem broke down).  The
        # loop stops once the curvature search adds no row.
    problem.evaluations += face_obj.evaluations
    full = (
        evaluation.state if isinstance(evaluation.state, _ObjectiveEvaluation) else None
    )
    if full is None:
        full = objective(face_obj.full(phi))
    theta = face_obj.full(phi)
    return status, theta, full, bound, separation


class _NoConvexFaceStart(RuntimeError):
    """A candidate face's cut-exchange could not verify a convex start."""


def _face_direction(problem, face, basis, curvature, /):
    """Strictly interior, globally convex direction of a face (face coordinates).

    The max-margin interior direction of the face rows, cut-exchanged until
    the curvature search verifies it (``q''`` is homogeneous, so the whole
    ray is then convex).  The search adds rows for newly discovered cuts.

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping (its rows grow with the cuts found).
    face : _Face
        Face.
    basis : numpy.ndarray
        Face basis ``N``, shape ``(n, r)``.
    curvature : _JointCurvature
        Curvature search restricted to the face.

    Raises
    ------
    RuntimeError
        If the direction cannot be verified nonnegative within the exchange
        budget.
    """
    rows = _active_rows(problem, face, basis)
    direction = _interior_direction(rows)
    for _ in range(problem.max_exchanges):
        separation = curvature.separate(basis @ direction, problem.curvature_options)
        if separation.status == "verified":
            return direction
        before = (
            problem.rows.size + len(problem.scoped_tails) + len(problem.scoped_points)
        )
        _merge_face_separation(problem, face, separation)
        if (
            problem.rows.size + len(problem.scoped_tails) + len(problem.scoped_points)
            == before
        ):
            break
        rows = _active_rows(problem, face, basis)
        direction = _interior_direction(rows)
    raise _NoConvexFaceStart("no verified convex point on the face")


def _face_anchor(problem, face, basis, face_obj, curvature, /):
    """Globally convex point of a face: its convex cold start.

    The same construction as the fit's cold start (``_convex_start``),
    restricted to the face: the ray minimizer along ``_face_direction``.

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping.
    face : _Face
        Face.
    basis : numpy.ndarray
        Face basis ``N``, shape ``(n, r)``.
    face_obj : _FaceObjective
        Objective on the face.
    curvature : _JointCurvature
        Curvature search restricted to the face.

    Raises
    ------
    RuntimeError
        If the direction cannot be verified nonnegative within the exchange
        budget.
    """
    direction = _face_direction(problem, face, basis, curvature)
    rows = _active_rows(problem, face, basis)
    return _interior_start(face_obj, rows, direction=direction).theta


def _candidate_face(problem, face, theta, /):
    """``_solve_face`` of a face to enter, ``None`` if its start fails.

    A face to enter (a collapsed leader or a zero log amplitude) is an
    alternative to a valid incumbent: when the normalizer cannot resolve its
    start the candidate is not adopted, and the fit keeps the incumbent.
    Leaving a face is different: an unresolvable release start fails the
    solve (a mixture M-step then stalls, ending growth), since continuing
    past it ran release solves of 30 s that never converged.

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping.
    face : _Face
        Candidate face.
    theta : numpy.ndarray
        Start (full coordinates).
    """
    try:
        return _solve_face(problem, face, theta, restore_start=True)
    except _QuadratureFailure as exc:
        # Routine: a candidate start on the face's boundary can leave
        # exponentially cancelling columns the normalizer cannot evaluate.
        _reraise_if_debug(exc, "candidate face start", routine=True)
        return None
    except _NoConvexFaceStart as exc:
        # An optional active-set face without a verified convex start cannot
        # displace the current run.  Preserve that run and let the caller's
        # normal convergence check decide whether it is acceptable.
        _reraise_if_debug(exc, "candidate face start", routine=True)
        return None


def _leader_release(problem, face, theta, evaluation, sides, gscale, /):
    """A released-leader face and its start, or ``None``.

    Releasing the collapsed leaders of ``sides`` is the face without those
    ends' suppression (``_face_from``; its forced equalities are recomputed,
    and a face no larger than the current one is not a release).  It is
    tried only when ``theta`` is not a KKT point of the released face: the
    NNLS residual ``r`` of the gradient over the near-active rows (the
    released leader's own growth row among them) is nonzero, so ``-r`` is a
    feasible descent direction (``g . (-r) = -|r|^2``); a gain along the
    leader functional alone misses descent into the released face along other
    directions.  The start is ``theta + t N w`` with ``w``
    the released face's verified convex interior direction: ``theta``
    satisfies every row and ``q''(theta) >= 0`` (a converged shape verdict),
    ``N w`` strictly satisfies every row with ``q''(N w) >= 0``, and rows and
    ``q''`` are linear in the coefficients, so the start is strictly inside,
    globally convex, and proper (the released leader is positive).

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping.
    face : _Face
        Current face.
    theta : numpy.ndarray
        Converged point on ``face`` (joint coordinates).
    evaluation : _ObjectiveEvaluation
        Objective at ``theta``.
    sides : tuple
        ``(block, end)`` pairs whose leaders are released.
    gscale : float
        Gradient scale of the KKT test.

    Returns
    -------
    tuple or None
        ``(released_face, start)``.
    """
    blocks = problem.blocks
    d = blocks.n
    rest = tuple(e for e in face.suppressed if e not in sides)
    released = _face_from(blocks, rest, face.zero_logs)
    if released is None or released.suppressed == face.suppressed:
        return None
    if released.equalities.shape[0] and np.max(
        np.abs(released.equalities @ theta)
    ) > _ACTIVE_SLACK * max(1.0, float(np.linalg.norm(theta))):
        return None  # the current point is not on the released face
    rbasis = released.basis(d)
    rrows = _active_rows(problem, released, rbasis)
    rphi = rbasis.T @ theta
    rgrad = rbasis.T @ evaluation.gradient
    lam = _multipliers(rrows, rgrad, rphi, rbasis.T @ evaluation.hessian @ rbasis)
    residual = rgrad - rrows.rows.T @ lam
    if float(np.linalg.norm(residual)) <= 1e-9 * gscale:
        return None
    curvature = problem.curvature.on_face(rbasis)
    w = rbasis @ _face_direction(problem, released, rbasis, curvature)
    t = _RELEASE_START * max(1.0, float(np.linalg.norm(theta))) / np.linalg.norm(w)
    return released, theta + t * w


def _center_cut(blocks, /):
    """Curvature cuts ``Phi''(c) . theta_k >= 0`` at the fitting center, per block.

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    """
    center = np.array([blocks.model.geometry.center])
    return np.array(
        [
            blocks.embed(k, _evaluate_basis(m, center, max_order=2).d2phi[0])
            for k, m in enumerate(blocks.models)
        ]
    )


def _leading_row(blocks, k, side, /):
    """Requested leading functional of one block's infinite end (joint row).

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    k : int
        Block index.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    lead = blocks.models[k].side_asymptotics(side).leading
    return blocks.embed(k, [float(v) for v in lead.row_exact])


def _forced_curvature_equalities(blocks, /):
    """Exact tail-curvature equalities required on every feasible face.

    A linear-tail curvature leader with opposite coefficient functionals at
    the two infinities must vanish: neither sign is globally admissible.
    This is an algebraic face condition, not a pair of strict inequalities.
    Compare the exact coefficients and asymptotic orders before restricting
    the optimizer; finite-point cuts are still certified independently.

    Parameters
    ----------
    blocks : _Blocks
        Component bases in joint coefficient coordinates.

    Returns
    -------
    numpy.ndarray
        Forced coefficient equalities, shape ``(n_equalities, n)``.
    """
    equalities = []
    for k, model in enumerate(blocks.models):
        if len(model.asymptotics) != 2:
            continue
        lower, upper = model.asymptotics
        if (lower.side, upper.side) != ("lower", "upper"):
            continue
        if not lower.curvature or not upper.curvature:
            continue
        a, b = lower.curvature[0], upper.curvature[0]
        if (a.shape, a.rate, a.order) != (b.shape, b.rate, b.order):
            continue
        if not any(a.row_exact) or any(
            x != -y for x, y in zip(a.row_exact, b.row_exact, strict=True)
        ):
            continue
        equalities.append(blocks.embed(k, [float(v) for v in a.row_exact]))
    return np.asarray(equalities, dtype=np.float64).reshape(-1, blocks.n)


def _face_from(blocks, suppressed, zero_logs, /):
    """Face fixing the leaders of ``suppressed`` ``(block, end)`` pairs and ``zero_logs``.

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks.
    suppressed : tuple
        ``(block, end)`` pairs whose requested leaders are fixed at zero.
    zero_logs : tuple of int
        Joint columns of endpoint logs fixed at zero.
    """
    d = blocks.n
    rows = [_leading_row(blocks, k, side) for k, side in suppressed]
    rows += [np.eye(d)[col] for col in zero_logs]
    forced = _forced_curvature_equalities(blocks)
    eq = np.vstack([forced, np.asarray(rows, dtype=np.float64).reshape(-1, d)])
    grown = _face_growth(blocks, eq)
    if grown is None:
        return None
    eq, growth = grown
    return _Face(eq, tuple(zero_logs), _suppressed_ends(blocks, eq), growth)


def _matching_face(blocks, suppressed, zero_log_labels, /):
    """The face matching a smaller fit's face (warm starts).

    Parameters
    ----------
    blocks : _Blocks
        Coefficient blocks of the larger fit.
    suppressed : tuple of tuple
        ``(block, end)`` pairs.
    zero_log_labels : tuple of str
        Joint labels of the zero log columns.

    Raises
    ------
    RuntimeError
        If the smaller fit's face has no proper member here (impossible for
        a span that contains it).
    """
    labels = list(blocks.labels)
    zero_logs = tuple(labels.index(label) for label in zero_log_labels)
    face = _face_from(blocks, tuple(suppressed), zero_logs)
    if face is None:
        raise RuntimeError("internal error: a lifted face has no proper member")
    return face


def _convex_start(problem, structural, /):
    """Cold start strictly inside the structural *and* curvature rows.

    The cut exchange runs on the interior direction itself: recompute the strictly interior direction with the cuts
    found so far until the curvature search adds none.  ``q''`` is linear and
    homogeneous in ``theta``, so every point ``a w`` of the ray is then
    convex as well, and the ray minimizer is a feasible start.

    Parameters
    ----------
    problem : _Problem
        Fit bookkeeping (its rows grow with the start's cuts).
    structural : _ConstraintRows
        Structural rows.

    Returns
    -------
    _StartResult
    """
    direction = _interior_direction(problem.rows)
    if np.any(direction):
        for _ in range(problem.max_exchanges):
            separation = problem.curvature.separate(
                direction, problem.curvature_options
            )
            before = problem.rows.size
            problem.rows = _merge_separation(problem.rows, problem.blocks, separation)
            if problem.rows.size == before:
                break
            direction = _interior_direction(problem.rows)
    return _interior_start(problem.objective, structural, direction=direction)


def _better(candidate, incumbent, tolerance, /, *, strictly=False):
    """Face acceptance: a verified optimum no worse than the incumbent.

    A face solve can end converged on its relaxation while its shape is still
    violated (a violation its exchange could not cut); such a point is not
    an optimum of the shape-constrained problem and never displaces the
    incumbent, however low its NLL.

    Parameters
    ----------
    candidate, incumbent : tuple
        ``_solve_face`` results ``(status, theta, evaluation, bound, separation)``.
    tolerance : float
        NLL tolerance (nats per observation).
    strictly : bool, default False
        Require an NLL decrease of more than ``tolerance`` (leaving a face)
        instead of allowing an increase up to it (entering one).
    """
    separation = candidate[4]
    if candidate[0] not in _CONVERGED or separation is None or not separation.feasible:
        return False
    if strictly:
        return float(candidate[2].nll) < float(incumbent[2].nll) - tolerance
    return float(candidate[2].nll) <= float(incumbent[2].nll) + tolerance


def _solve_coordinate_basis(
    objective,
    /,
    *,
    options=None,
    curvature_options=None,
    max_exchanges=30,
    start=None,
    start_face=None,
    blocks=None,
    interior_lift=False,
    start_cuts=None,
):
    """Fit a basis point or interval objective over ``q'' >= 0`` with exact faces.

    Parameters
    ----------
    objective : _BasisPointObjective or _BasisIntervalObjective
        Objective; a ``lower_bound(nll, options)`` method supplies the
        global certificate, applied after the local
        solve.
    options : _NewtonOptions or None
        Newton policy.
    curvature_options : _CurvatureSearchOptions or None
        Curvature search settings.
    max_exchanges : int
        Cut exchange rounds per face solve.  With convex trial points a round
        ends after one accepted step whenever a step reveals new cuts; every
        fit of a reference grid needs fewer than 30 rounds per face, and a
        budget of 60 only slowed the loud failures of the explicit 13-column
        ``full`` expansion (71 -> 787 s) without rescuing one.
    start : numpy.ndarray or None
        Warm start (a lifted fit of a smaller span: proper and globally
        convex, possibly on active rows); ``None`` builds the convex cold
        start.
    blocks : _Blocks or None
        Coefficient blocks (default: the single fit's identity block).
    start_face : tuple or None
        ``(suppressed, zero_log_labels)`` of the fit ``start`` was lifted
        from (``(block, end)`` pairs and joint labels): the warm start begins on the matching face of this basis (its
        suppressed functionals are exact zeros there).  Both leader faces
        and zero-log faces are released when the KKT test says so;
        a start without strict growth on that face is replaced by the face's
        own convex cold start.
    interior_lift : bool
        Instead, move such a start (feasible and globally convex) into the
        face's interior along its verified convex direction (the face's cold
        start lost a degree-7 lognormal fit at NLL 2.44 against the nested
        1.45).
    start_cuts : numpy.ndarray or None
        Curvature cut rows in this objective's coordinates, shape ``(k, d)``,
        seeded with the structural rows (see :func:`_solve_basis`); slack
        ones are retired like any other cut.

    Returns
    -------
    _BasisNewtonResult

    Raises
    ------
    RuntimeError
        If the likelihood supremum lies on the improper boundary (an active
        leading-growth row whose face has no proper member), or the start
        cannot be built.
    _QuadratureFailure
        At the start, an accepted iterate or the final state.
    """
    _check_fit_time("component solve")
    options = _NewtonOptions() if options is None else options
    if not isinstance(options, _NewtonOptions):
        raise TypeError("options must be _NewtonOptions or None")
    blocks = _Blocks.single(objective.model) if blocks is None else blocks
    d = blocks.n
    structural = blocks.structural_rows()
    problem = _Problem(
        objective,
        blocks,
        _JointCurvature(blocks),
        CURVATURE_SEARCH_OPTIONS if curvature_options is None else curvature_options,
        options,
        int(max_exchanges),
        rows=(
            structural if start_cuts is None else structural.merged(start_cuts, "cut")
        ),
    )
    if start_face is not None:
        face = _matching_face(blocks, *start_face)
    elif _forced_curvature_equalities(blocks).size:
        face = _face_from(blocks, (), ())
    else:
        face = _Face(
            np.zeros((0, d)),
            (),
            (),
            structural.rows[[k == "growth" for k in structural.kinds]],
        )
    if face is None:
        raise RuntimeError("the requested basis has no proper coefficient face")
    if start is None:
        if face.equalities.size:
            basis = face.basis(d)
            curvature = problem.curvature.on_face(basis)
            objective = problem.objective.on_face(basis)
            face_obj = _FaceObjective(
                objective, basis, curvature, problem.curvature_options
            )
            start_theta = basis @ _face_anchor(
                problem, face, basis, face_obj, curvature
            )
        else:
            start_theta = _convex_start(problem, structural).theta
    else:
        start_theta = np.asarray(start, dtype=np.float64).reshape(-1)
        if start_theta.shape != (d,) or not np.all(np.isfinite(start_theta)):
            raise ValueError("start must be finite with one entry per coefficient")
    run = _solve_face(problem, face, start_theta, interior_lift=interior_lift)

    for _ in range(8):
        changed = False
        status, theta, evaluation, _, _ = run
        if status not in _CONVERGED:
            break
        basis = face.basis(d)
        face_rows = _active_rows(problem, face, basis)
        phi = basis.T @ theta
        lam = _multipliers(
            face_rows,
            basis.T @ evaluation.gradient,
            phi,
            basis.T @ evaluation.hessian @ basis,
        )
        gscale = max(1.0, float(np.max(np.abs(evaluation.gradient))))
        active = lam > 1e-9 * gscale
        # Leader faces: an active growth row is a collapsing leader.
        for idx in np.flatnonzero(
            active & np.array([k == "growth" for k in face_rows.kinds])
        ):
            lead = basis @ face_rows.rows[idx]
            grown = _face_growth(blocks, np.vstack([face.equalities, lead]))
            if grown is None:
                raise RuntimeError(
                    "the maximum-likelihood fit lies on the improper boundary: a "
                    "leading tail amplitude collapses to zero and the reduced "
                    "family has no proper member (the MLE does not exist in the "
                    "requested proper family)"
                )
            eq, growth = grown
            candidate_face = _Face(
                eq, face.zero_logs, _suppressed_ends(blocks, eq), growth
            )
            candidate = _candidate_face(problem, candidate_face, theta)
            if candidate is not None and _better(candidate, run, options.tolerance):
                run, face, changed = candidate, candidate_face, True
                break
        if changed:
            continue
        # Zero-amplitude faces: an active log bound.
        for idx in np.flatnonzero(
            active & np.array([k == "log" for k in face_rows.kinds])
        ):
            col = int(np.argmax(np.abs(basis @ face_rows.rows[idx])))
            if col in face.zero_logs:
                continue
            eq = np.vstack([face.equalities, np.eye(d)[col]])
            candidate_face = _Face(
                eq, (*face.zero_logs, col), face.suppressed, face.growth
            )
            candidate = _candidate_face(problem, candidate_face, theta)
            if candidate is not None and _better(candidate, run, options.tolerance):
                run, face, changed = candidate, candidate_face, True
                break
        if changed:
            continue
        # Release a collapsed leader when the active-row KKT gain says so: a
        # warm start carries the incumbent's suppressed leaders into a larger
        # span, where they need not stay zero.
        # One end at a time, then both: suppressing one end can force the
        # other (opposite signs of a shared functional), so only releasing
        # both leaves the face.
        release_sets = [(s,) for s in face.suppressed]
        if len(face.suppressed) > 1:
            release_sets.append(tuple(face.suppressed))
        for sides in release_sets:
            release = _leader_release(problem, face, theta, evaluation, sides, gscale)
            if release is None:
                continue
            released_face, start_theta = release
            candidate = _solve_face(problem, released_face, start_theta)
            if _better(candidate, run, options.tolerance, strictly=True):
                run, face, changed = candidate, released_face, True
                break
        if changed:
            continue
        # Release a zero amplitude when the active-row KKT gain says so.
        for col in face.zero_logs:
            rest = tuple(c for c in face.zero_logs if c != col)
            eq = np.array(
                [r for r in face.equalities if not np.allclose(r, np.eye(d)[col])]
            ).reshape(-1, d)
            released_face = _Face(eq, rest, face.suppressed, face.growth)
            rbasis = released_face.basis(d)
            rrows = _active_rows(problem, released_face, rbasis)
            rphi = rbasis.T @ theta
            rgrad = rbasis.T @ evaluation.gradient
            # Reduced gradient along the amplitude, net of active inequality rows.
            lam_r = _multipliers(
                rrows, rgrad, rphi, rbasis.T @ evaluation.hessian @ rbasis
            )
            direction = rbasis.T @ np.eye(d)[col]
            gain = float(direction @ (rgrad - rrows.rows.T @ lam_r))
            if gain >= -1e-9 * gscale:
                continue
            start_theta = theta.copy()
            start_theta[col] = _RELEASE_START
            candidate = _solve_face(problem, released_face, start_theta)
            if _better(candidate, run, options.tolerance, strictly=True):
                run, face, changed = candidate, released_face, True
                break
        if not changed:
            break
    status, theta, evaluation, bound, separation = run
    if status in _CONVERGED and (separation is None or separation.status != "verified"):
        # Convergence is claimed only with a conclusive shape verdict.
        status = _UNVERIFIED
    gap = math.nan
    bound_of = getattr(objective, "lower_bound", None)
    lower = None
    if bound_of is not None and (
        separation is not None and separation.status == "verified"
    ):
        lower = bound_of(float(evaluation.nll), options)
    if lower is not None:
        # The global certificate after the local solve: an endpoint at the
        # nonparametric bound is globally optimal.
        local = _NewtonRun(
            status=status,
            params=theta,
            blocks=None,
            dual=None,
            evaluation=evaluation,
            iterations=0,
            evaluations=0,
            subproblem_iterations=0,
            decrease_bound=bound,
        )
        certified = _apply_global_objective_certificate(
            SimpleNamespace(nll_lower_bound=lower), local, options
        )
        status = certified.status
        gap = float(evaluation.nll) - float(lower)
    return _BasisNewtonResult(
        status=status,
        params=theta,
        objective_value=float(evaluation.nll),
        evaluation=evaluation,
        rows=problem.rows,
        face=face,
        newton_iterations=problem.newton_iterations,
        objective_evaluations=problem.evaluations,
        subproblem_iterations=problem.subproblem_iterations,
        exchanges=problem.exchanges,
        final_decrease_bound=float(bound),
        final_separation=separation,
        global_gap=gap,
    )


def _solve_basis(
    objective,
    /,
    *,
    options=None,
    curvature_options=None,
    max_exchanges=30,
    start=None,
    start_face=None,
    blocks=None,
    interior_lift=False,
    start_cuts=None,
):
    """Fit through one controller with validated runtime coordinate candidates.

    Parameters
    ----------
    objective : _BasisPointObjective, _BasisIntervalObjective or joint objective
        Objective in canonical coordinates.
    options, curvature_options, max_exchanges, start, start_face, blocks
        As for :func:`_solve_coordinate_basis`.
    interior_lift : bool
        As for :func:`_solve_coordinate_basis`.
    start_cuts : numpy.ndarray or None
        Canonical curvature cut rows ``r . theta >= 0`` of an earlier solve of
        the same basis, shape ``(k, d)``: a mixture's previous M-step.  Each
        is ``q''(x*) >= 0`` at some point, valid for every convex potential,
        so seeding them changes the exchange path and not the feasible set.
    """
    # Deferred: ``coordinates`` imports ``basis_newton`` (``_Blocks``,
    # ``_face_from``, ``_JointCurvature``).
    from .coordinates import _prepare_coordinates

    _check_fit_time("component solve")
    options = _NewtonOptions() if options is None else options
    if not isinstance(options, _NewtonOptions):
        raise TypeError("options must be _NewtonOptions or None")
    prepared = _prepare_coordinates(objective, blocks, start)
    best = None
    totals = dict(
        newton_iterations=0,
        objective_evaluations=0,
        subproblem_iterations=0,
        exchanges=0,
    )
    for binding in prepared.candidates(start):
        _check_fit_time("component solve")
        try:
            result = _solve_coordinate_basis(
                binding.objective,
                options=options,
                curvature_options=curvature_options,
                max_exchanges=max_exchanges,
                start=binding.start,
                start_face=start_face,
                blocks=binding.blocks,
                interior_lift=interior_lift,
                start_cuts=(
                    None
                    if start_cuts is None or not len(start_cuts)
                    else np.asarray(start_cuts, dtype=np.float64) @ binding.transform
                ),
            )
            for name in totals:
                totals[name] += getattr(result, name)
            result = binding.restore(result, curvature_options)
        except NUMERIC_FAILURES as exc:
            if binding.original is binding.objective:
                raise
            # Not routine: the runtime coordinates failed numerically or did
            # not reproduce the canonical likelihood and shape; the same
            # family is retried in canonical coordinates.
            _reraise_if_debug(exc, "runtime-coordinate solve")
            continue
        if result.status in _CONVERGED:
            return dataclass_replace(result, **totals)
        score = (result.status == _UNVERIFIED, -result.objective_value)
        if best is None or score > (best.status == _UNVERIFIED, -best.objective_value):
            best = result
    return dataclass_replace(best, **totals)
