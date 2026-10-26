"""Shared-support mixtures of basis densities.

Three phases: SQUAREM-accelerated EM, a joint Newton polish of the observed
likelihood with the saddle-free metric, and one verifying EM step.
Components have private shapes with one shared variable per enabled
endpoint log amplitude; the EM surrogate is the M-step objective; a
multistart search with initializers and soundness guards picks the fit.

**Coordinates.**  Each component is compiled in its own
responsibility-weighted coordinate and geometry, fixed by its start (a
narrow or remote component in a pooled coordinate needs coefficients of
order ``(scale ratio)^2`` and loses its digits to cancellation); a
component is a coefficient block ``theta_k`` of the stacked bases
(``basis_newton._Blocks``).  The endpoint log columns are joint coordinates
shared by every block (shared physical amplitudes); all other columns,
including tail columns, are private.
The joint polish appends ``K - 1`` logits (``pi_K`` the reference), which
are in no block and unconstrained.

**Objectives** (``_basis_newton_kernels``, one implementation for the face
Newton loop and Python evaluations).  The coupled M-step is the EM surrogate
``sum_k m_k nll_k(theta_k)`` with responsibility weights; the joint objective
is the observed NLL ``-sum_i w_i log sum_k pi_k p_k(row_i)`` with
complete-data information ``G`` and missing information ``M`` (label
uncertainty plus the within-row covariances of interval rows); the Newton
metric is the saddle-free metric of ``G`` and ``M``.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .._defaults import (
    AUTO_K_MAX,
    AUTO_LC_MIN_COMPONENT_N,
    AUTO_LC_SCREEN_STOP_ABOVE_MODES,
    AUTO_LC_SWEEP_EM_MAX_ITER,
    AUTO_LC_SWEEP_NEWTON_TOL,
    BOUNDARY_ALPHA,
    CURVATURE_SEARCH_OPTIONS,
    EM_MIN_EFFECTIVE_DISTINCT_N,
    NUMERIC_FAILURES,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
    _reraise_if_debug,
)
from .._model.basis import _compile_basis, _full_basis, _grow_basis
from .._model.basis_eval import _evaluate_basis
from .._model.basis_normalizer import (
    _ImproperPotential,
    _NormalizerContext,
    _NormalizerOptions,
)
from .._model.basis_spec import _AUTO, _AUTO_EXPANSION, _MINIMAL
from ._basis_newton_kernels import KIND_JOINT, KIND_MSTEP, BasisFaceObjective
from ._conic_kernels import safeguarded_metric
from .basis_newton import _Blocks, _matching_face, _solve_basis
from .basis_objective import (
    _BasisIntervalObjective,
    _BasisPointObjective,
    _interval_rows,
    _normalizer_block,
    _ObjectiveEvaluation,
    _PointStatistics,
    _trial_failure,
)
from .basis_selection import (
    _bulk_degrees,
    _fit_basis,
    _IntervalData,
    _lift_matrix,
    _PointData,
    _score_test,
    _tested,
    _UntestableSpan,
)
from .boundary import _effective_n, _select_boundary_terms
from .budget import _check_fit_time
from .conic_newton import _CONVERGED, _NewtonOptions
from .degree import _DegreeSelectionConfig
from .inputs import _normalize_basis_inputs, _prepare_component_selection_policy
from .mixture import (
    _distinct_location_index,
    _effective_distinct_point_count,
    _initial_responsibility_candidates,
    _interval_identifiability_diagnostic,
    _interval_initial_representatives,
    _propose_n_components,
    _stratified_subsample,
)


@dataclass(frozen=True)
class _EMOptions:
    """Immutable EM scheduling policy for one mixture fit.

    Parameters
    ----------
    tolerance : float, optional
        Relative log-likelihood increase that ends an EM phase.
    max_steps : int, optional
        EM map evaluation limit per phase.
    max_rounds : int, optional
        EM-polish-verify round limit.
    accelerate : bool, optional
        Whether to use SQUAREM acceleration.
    """

    tolerance: float = 1e-4
    max_steps: int = 20
    max_rounds: int = 4
    accelerate: bool = True


@dataclass(frozen=True)
class _MixtureSearchOptions:
    """Immutable multistart search policy for one mixture fit.

    Parameters
    ----------
    treatments : tuple of str, optional
        Responsibility treatments explored from every initializer
        (``"raw"``, ``"sharpened"``). A failed fixed-degree finalist may receive
        one retry from the nearest smaller admissible degree; the ladder is
        not added to the default exploration of every initializer.
    finalists : int, optional
        Number of best explorations run to convergence.  One: each finalist
        costs a full EM and joint polish (a third of mixture fit time), and
        a second finalist changed no selected component count over 122
        paired benchmark fits.
    """

    treatments: tuple = ("raw", "sharpened")
    finalists: int = 1


def _resolve_search_options(search_options, /):
    """Return one validated immutable multistart search policy.

    Parameters
    ----------
    search_options : _MixtureSearchOptions or None
        Explicit policy, or ``None`` for the mixture defaults.

    Returns
    -------
    _MixtureSearchOptions
    """
    if search_options is None:
        return _MixtureSearchOptions()
    if not isinstance(search_options, _MixtureSearchOptions):
        raise TypeError("search_options must be _MixtureSearchOptions or None")
    return search_options


def _resolve_em_options(em_options, /):
    """Return one validated immutable EM policy.

    Parameters
    ----------
    em_options : _EMOptions or None
        Explicit policy, or ``None`` for the mixture defaults.

    Returns
    -------
    _EMOptions
    """
    if em_options is None:
        return _EMOptions()
    if not isinstance(em_options, _EMOptions):
        raise TypeError("em_options must be _EMOptions or None")
    return em_options


class _PointMixtureEstimabilityFailure(RuntimeError):
    """Expected rejection of a singular point-mixture candidate."""


def _check_point_estimability(rows, observation_weights, responsibilities, /):
    """Reject the singular one-location boundary of point-mixture likelihood.

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical observation rows.
    observation_weights : numpy.ndarray, shape (R,)
        Normalized observation weights.
    responsibilities : numpy.ndarray, shape (R, K)
        Current component responsibilities.

    Raises
    ------
    _PointMixtureEstimabilityFailure
        If a point-data component has fewer than the required effective
        distinct sample locations.
    """
    if rows.shape[1] != 1 or responsibilities.shape[1] <= 1:
        return
    x = rows[:, 0]
    inverse = _distinct_location_index(rows)
    for k in range(responsibilities.shape[1]):
        n_eff = _effective_distinct_point_count(
            x, observation_weights * responsibilities[:, k], inverse=inverse
        )
        if n_eff < EM_MIN_EFFECTIVE_DISTINCT_N:
            raise _PointMixtureEstimabilityFailure(
                "point-mixture component is not estimable: component "
                f"{k + 1}/{responsibilities.shape[1]} has only {n_eff:.6g} "
                "effective distinct sample locations. The unconstrained "
                "point-mixture likelihood is singular when a component "
                "concentrates onto one observed location."
            )


def _mixture_blocks(models, /, *, logits=False):
    """Blocks of one component per model, endpoint logs shared.

    Joint coordinates: each component's private columns in turn, then the
    shared endpoint log columns (matched by label: the same function in
    every span of one geometry), then (``logits``) ``K - 1`` logits.

    Parameters
    ----------
    models : sequence of _BasisModel
        Each component's basis (one geometry; spans may differ).
    logits : bool
        Append the mixture logits.

    Returns
    -------
    _Blocks
    """
    models = tuple(models)
    k = len(models)
    shared = []
    for model in models:
        for c in model.columns:
            if c.nonnegative and c.label not in shared:
                shared.append(c.label)
    labels, index, offset = [], [], 0
    for comp, model in enumerate(models):
        idx = np.empty(model.n_coefficients, dtype=np.intp)
        for j, c in enumerate(model.columns):
            if not c.nonnegative:
                idx[j] = offset
                labels.append(f"{comp}:{c.label}")
                offset += 1
        index.append(idx)
    for comp, model in enumerate(models):
        for j, c in enumerate(model.columns):
            if c.nonnegative:
                index[comp][j] = offset + shared.index(c.label)
    labels += shared
    n = offset + len(shared)
    if logits:
        labels += [f"logit{comp}" for comp in range(k - 1)]
        n += k - 1
    return _Blocks(models, tuple(index), n, tuple(labels))


@dataclass(frozen=True)
class _MixtureRows:
    """Observations of a mixture fit (points or canonical interval rows).

    Parameters
    ----------
    weights : numpy.ndarray
        Normalized row weights (sum one), shape ``(R,)``.
    x : numpy.ndarray or None
        Point observations, shape ``(R,)``.
    rows : _IntervalRows or None
        Interval rows (``basis_objective._interval_rows``).
    effective_n : float or None
        Kish effective size of the original rows (default: of ``weights``;
        merged interval rows must give it).
    """

    weights: np.ndarray
    x: np.ndarray | None = None
    rows: object = None
    effective_n: float | None = None

    @property
    def n_eff(self):
        """Kish effective sample size of the original rows."""
        if self.effective_n is not None:
            return float(self.effective_n)
        w = self.weights
        return float(w.sum() ** 2 / np.sum(w * w))

    @property
    def intervals(self):
        """Whether the observations are interval rows."""
        return self.rows is not None

    @property
    def size(self):
        """Number of rows."""
        return int(self.weights.size)

    def phi(self, model, /):
        """``Phi(x_i)`` of point rows, shape ``(R, d)``.

        Parameters
        ----------
        model : _BasisModel
            Component basis.
        """
        return _evaluate_basis(model, self.x, max_order=0).phi


class _MixtureObjective:
    """A mixture objective over joint coordinates (``KIND_MSTEP``/``KIND_JOINT``).

    Parameters
    ----------
    blocks : _Blocks
        Component blocks (``_mixture_blocks``).
    contexts : tuple of _NormalizerContext
        Per component (face-restricted on a face).
    kind : int
        ``KIND_MSTEP`` or ``KIND_JOINT``.
    data : dict
        Compiled data (``BasisFaceObjective``).
    options : _NormalizerOptions or None
        Normalizer settings.
    """

    def __init__(self, blocks, contexts, kind, data, options=None, /):
        """Bind blocks, contexts and data (see class docstring)."""
        self.blocks = blocks
        self.model = blocks.model
        self.contexts = tuple(contexts)
        self.context = self.contexts[0]
        self.kind = kind
        self.data = data
        self.options = options
        self.intervals = "rows" in data
        self._evaluator = None

    def coordinate_observations(self, block):
        """Per-component observations used to rebuild point statistics directly."""
        if self.intervals:
            return None, None
        if "points" not in self.data:
            return None
        weights = self.data.get("point_weights")
        return self.data["points"], None if weights is None else weights[block]

    def in_coordinates(self, blocks):
        """Rebind all component kernels while preserving shared coordinates."""
        data = dict(self.data)
        if not self.intervals:
            phis = [
                _evaluate_basis(m, data["points"], max_order=0).phi
                for m in blocks.models
            ]
            if "stats" in data:
                data["stats"] = [w @ phi for w, phi in zip(data["point_weights"], phis)]
            else:
                data["phi"] = phis
        return _MixtureObjective(
            blocks, _contexts(blocks.models), self.kind, data, self.options
        )

    def on_face(self, basis, /):
        """The same objective for points of a face ``theta = N phi``.

        Parameters
        ----------
        basis : numpy.ndarray
            Face basis ``N`` of the joint coordinates.
        """
        blocks = self.blocks
        contexts = tuple(
            c.on_face(blocks.range_basis(k, basis)) for k, c in enumerate(self.contexts)
        )
        return _MixtureObjective(blocks, contexts, self.kind, self.data, self.options)

    def metric(self, fisher, missing, /):
        """Newton metric of restricted ``F`` and ``C`` (``F`` for point M-steps).

        Parameters
        ----------
        fisher, missing : numpy.ndarray
            Complete-data information ``G`` and missing information ``M``.
        """
        if self.kind == KIND_MSTEP and not self.intervals:
            return fisher, math.nan
        return safeguarded_metric(fisher, missing)

    def compiled_spec(self):
        """``(kind, data, normalizer_blocks)`` of the compiled face objective."""
        return self.kind, self.data, [_normalizer_block(c) for c in self.contexts]

    def trial_failure(self, status, side, diag, theta, row, block, /):
        """The exception of a compiled trial the normalizer could not resolve.

        Parameters
        ----------
        status : int
            Normalizer status of the failed trial.
        side : int
            Side code of the failure.
        diag : float
            Normalizer diagnostic value.
        theta : numpy.ndarray
            Trial coefficients of the block.
        row : int
            Interval row of the failure, or ``-1``.
        block : int
            Component block.
        """
        k = max(int(block), 0)
        return _trial_failure(
            self.contexts[k],
            self.options,
            status,
            side,
            diag,
            self.blocks.part(k, theta),
            row,
            self.data.get("interval_rows"),
        )

    def evaluator(self):
        """Compiled evaluator on the identity face (built once, gate off)."""
        if self._evaluator is None:
            kind, data, nblocks = self.compiled_spec()
            self._evaluator = BasisFaceObjective(
                [m.kernel for m in self.blocks.models],
                np.eye(self.blocks.n),
                self.blocks.index,
                (nblocks, self.options or _NormalizerOptions()),
                ([(None, None, None)] * self.blocks.count, CURVATURE_SEARCH_OPTIONS),
                kind,
                data,
            )
        return self._evaluator

    def __call__(self, theta, /):
        """Return the NLL, gradient, ``F``, ``C`` and metric (joint coordinates).

        Raises
        ------
        _ImproperPotential
            If a component is improper.
        _QuadratureFailure
            If a component or a row cannot be resolved (fails loudly).
        """

        ev = self.evaluator()
        code, nll, grad, metric, fisher, missing, smallest = ev.evaluate(theta)
        if code != 0:
            failures = ev.failures()
            if failures is not None:
                _, status, side, diag, full, row, block = failures
                raise self.trial_failure(status, side, diag, full, row, block)
            raise _ImproperPotential("a mixture component is not normalizable")
        means, _ = ev.means()
        return _ObjectiveEvaluation(
            nll=float(nll),
            gradient=grad,
            hessian=metric,
            fisher=fisher,
            missing_information=missing,
            model_partial_means=sum(
                self.blocks.embed(k, m) for k, m in enumerate(means)
            ),
            state=None,
            smallest_curvature=smallest,
        )

    def row_terms(self, theta, /):
        """``(log p_k(row_i), r_ik)`` at joint ``theta`` (``KIND_JOINT``).

        Parameters
        ----------
        theta : numpy.ndarray
            Joint coordinates.
        """
        self(theta)
        return self.evaluator().row_terms()


def _contexts(models, /):
    """One normalizer context per component (shared between equal models).

    Parameters
    ----------
    models : sequence of _BasisModel
        Component bases.
    """
    cache = {}
    for m in models:
        cache.setdefault(id(m), _NormalizerContext(m))
    return tuple(cache[id(m)] for m in models)


def _phis(models, data, /):
    """``Phi_k(x_i)`` per component of point rows (shared between equal models).

    Parameters
    ----------
    models : sequence of _BasisModel
        Component bases.
    data : _MixtureRows
        Point rows.
    """
    cache = {}
    for m in models:
        if id(m) not in cache:
            cache[id(m)] = np.ascontiguousarray(data.phi(m))
    return [cache[id(m)] for m in models]


def _joint_objective(models, data, /, options=None, scores=False):
    """The observed mixture NLL of one component per model (joint coordinates).

    ``scores`` keeps the per-row joint scores (selection diagnostics).

    Parameters
    ----------
    models : sequence of _BasisModel
        Component bases.
    data : _MixtureRows
        Observations.
    options : _NormalizerOptions or None
        Normalizer policy.
    scores : bool
        Keep per-row joint scores.
    """
    models = tuple(models)
    blocks = _mixture_blocks(models, logits=True)
    payload = {"weights": data.weights, "n_logit": len(models) - 1, "scores": scores}
    if data.intervals:
        payload["rows"] = (data.rows.lower, data.rows.upper)
        payload["interval_rows"] = data.rows
    else:
        payload["phi"] = _phis(models, data)
        payload["points"] = data.x
    return _MixtureObjective(blocks, _contexts(models), KIND_JOINT, payload, options)


def _mstep_objective(models, data, responsibilities, /, options=None):
    """The coupled M-step (EM surrogate) of given responsibilities.

    Parameters
    ----------
    models : sequence of _BasisModel
        Each component's basis.
    data : _MixtureRows
        Observations.
    responsibilities : numpy.ndarray
        ``r_ik``, shape ``(R, K)``.
    options : _NormalizerOptions or None
        Normalizer settings.
    """
    models = tuple(models)
    r = np.asarray(responsibilities, dtype=np.float64)
    wr = data.weights[:, None] * r
    mass = wr.sum(axis=0)
    if r.shape[1] != len(models) or not np.all(mass > 0.0):
        raise ValueError("every component needs positive responsibility mass")
    blocks = _mixture_blocks(models)
    payload = {"mass": mass}
    if data.intervals:
        payload["rows"] = (data.rows.lower, data.rows.upper)
        payload["weights"] = np.ascontiguousarray((wr / mass).T)
        payload["interval_rows"] = data.rows
    else:
        phis = _phis(models, data)
        payload["points"] = data.x
        payload["point_weights"] = np.ascontiguousarray((wr / mass).T)
        payload["stats"] = [(wr[:, c] / mass[c]) @ phis[c] for c in range(len(models))]
    return _MixtureObjective(blocks, _contexts(models), KIND_MSTEP, payload, options)


# ---------------------------------------------------------------------------
# EM, polish and verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _MixtureState:
    """An evaluated mixture iterate.

    Parameters
    ----------
    theta : numpy.ndarray
        Joint component coefficients (``_mixture_blocks`` without logits).
    log_weights : numpy.ndarray
        Log mixture weights, shape ``(K,)``.
    face : tuple
        ``(suppressed, zero_log_labels)`` of the component fits (warm starts).
    log_likelihood : float
        Observed log likelihood per unit weight.
    responsibilities : numpy.ndarray
        ``r_ik`` at this iterate, shape ``(R, K)``.
    statuses : tuple of str
        Status of the M-step solves that produced ``theta``.
    """

    theta: np.ndarray
    log_weights: np.ndarray
    face: tuple
    log_likelihood: float
    responsibilities: np.ndarray
    statuses: tuple = ()


@dataclass(frozen=True)
class _MixtureFit:
    """A fitted mixture.

    Parameters
    ----------
    models : tuple of _BasisModel
        Each component's basis.
    thetas : tuple of numpy.ndarray
        Component coefficients.
    weights : numpy.ndarray
        Mixture weights.
    log_likelihood : float
        Observed log likelihood per unit weight.
    status : str
        Status of the joint polish (``em_only`` if it did not finish).
    responsibilities : numpy.ndarray
        Final ``r_ik``.
    face : tuple
        ``(suppressed, zero_log_labels)`` of the final face.
    n_face_parameters : int
        Free joint parameters on that face (logits included).
    observed_information : numpy.ndarray or None
        ``G - M`` of the polished fit (joint coordinates with logits).
    em_iterations, polish_iterations, rounds : int
        Work counters.
    history : tuple of float
        Log likelihood after every accepted EM cycle and polish.
    initialization : str
        Initializer label.
    result : _BasisNewtonResult or None
        The final joint polish.
    selection : tuple of _MixtureStep
        Automatic growth decisions (``_grow``).
    """

    models: tuple
    thetas: tuple
    weights: np.ndarray
    log_likelihood: float
    status: str
    responsibilities: np.ndarray
    face: tuple
    n_face_parameters: int
    observed_information: object
    em_iterations: int
    polish_iterations: int
    rounds: int
    history: tuple
    initialization: str
    result: object = None
    selection: tuple = ()


class _MixtureProblem:
    """Fixed model, data and component count of one mixture fit.

    Parameters
    ----------
    models : sequence of _BasisModel
        Each component's basis (one geometry).
    data : _MixtureRows
        Observations.
    options : _NewtonOptions
        Newton policy of every M-step and polish.
    """

    def __init__(self, models, data, options, /):
        """Bind the fit (see class docstring)."""
        self.models = tuple(models)
        self.model = self.models[0]
        self.data = data
        self.k = len(self.models)
        self.options = options
        # Growth candidates (``_refit(..., strict=True)``): an M-step that does
        # not converge ends the candidate (``_StalledRefit``).
        self.strict = False
        self.blocks = _mixture_blocks(self.models)
        self.joint_blocks = _mixture_blocks(self.models, logits=True)
        self.joint = _joint_objective(self.models, data)
        self.shared = any(c.nonnegative for m in self.models for c in m.columns)
        self._phis = None if data.intervals else _phis(self.models, data)
        self._contexts = _contexts(self.models)
        self._faces = {}
        # Latest canonical curvature cuts of each component's M-step (key
        # ``None``: the coupled M-step), seeding the next one (``_cut_rows``).
        self._cuts = {}

    def logits(self, log_weights, /):
        """Logits of log weights (``pi_K`` the reference).

        Parameters
        ----------
        log_weights : numpy.ndarray
            Log mixture weights, shape ``(K,)``.
        """
        return np.asarray(log_weights[:-1]) - float(log_weights[-1])

    def joint_point(self, theta, log_weights, /):
        """Joint coordinates ``(theta, logits)``.

        Parameters
        ----------
        theta : numpy.ndarray
            Stacked component coefficients.
        log_weights : numpy.ndarray
            Log mixture weights.
        """
        return np.concatenate([theta, self.logits(log_weights)])

    def on_face(self, face, /):
        """The joint objective on ``face`` (``(suppressed, zero_log_labels)``).

        A suppressed leader is an exact zero only on its face: off it, the
        roundoff of a cancelled leading functional (``-2e-15``) leaves
        properness undecided.

        Parameters
        ----------
        face : tuple
            ``(suppressed, zero_log_labels)`` of the joint fit.
        """

        if not (face[0] or face[1]):
            return self.joint
        key = (tuple(face[0]), tuple(face[1]))
        if key not in self._faces:
            basis = _matching_face(self.joint_blocks, *key).basis(self.joint_blocks.n)
            self._faces[key] = self.joint.on_face(basis)
        return self._faces[key]

    def evaluate(self, theta, log_weights, face, statuses=(), /):
        """The evaluated state of ``(theta, log_weights)`` (E-step included).

        Parameters
        ----------
        theta : numpy.ndarray
            Stacked component coefficients.
        log_weights : numpy.ndarray
            Log mixture weights.
        face : tuple
            Face the state lies on.
        statuses : tuple of str
            Statuses of the solves that produced it.
        """
        x = self.joint_point(theta, log_weights)
        objective = self.on_face(face)
        _, resp = objective.row_terms(x)
        ll = -float(objective(x).nll)
        return _MixtureState(theta, np.asarray(log_weights), face, ll, resp, statuses)

    def m_step(self, responsibilities, previous, /):
        """Component refits to the EM surrogate of ``responsibilities``.

        Coupled over the shared log columns (``KIND_MSTEP`` on the stacked
        blocks); otherwise one independent single fit per component, when
        no amplitude is shared.  Warm
        started from ``previous`` and its face when given.

        Parameters
        ----------
        responsibilities : numpy.ndarray
            Responsibilities, shape ``(R, K)``.
        previous : _MixtureState or None
            Warm start.

        Returns
        -------
        tuple
            ``(theta, log_weights, face, statuses)``.
        """

        _check_fit_time("mixture M-step")
        data, k = self.data, self.k
        r = np.asarray(responsibilities, dtype=np.float64)
        if not data.intervals and k > 1:
            # Every M-step is checked: EM can drive a component onto one
            # observed location, where the point likelihood is singular.
            _check_point_estimability(data.x[:, None], data.weights, r)
        wr = data.weights[:, None] * r
        mass = wr.sum(axis=0)
        if not np.all(mass > 0.0):
            raise RuntimeError("a mixture component lost all responsibility mass")
        start = None if previous is None else previous.theta
        face = None if previous is None else previous.face
        labels = self.blocks.labels
        if k > 1 and self.shared:
            objective = _mstep_objective(self.models, data, r)
            result = _solve_basis(
                objective,
                options=self.options,
                blocks=self.blocks,
                start=start,
                start_face=face,
                start_cuts=self._cuts.get(None),
            )
            self._cuts[None] = _cut_rows(result)
            new_face = (
                result.face.suppressed,
                tuple(labels[i] for i in result.face.zero_logs),
            )
            return result.params, np.log(mass), new_face, (result.status,)
        theta = np.zeros(self.blocks.n)
        suppressed, statuses = [], []
        for comp in range(k):
            ctx = self._contexts[comp]
            v = wr[:, comp] / mass[comp]
            if data.intervals:
                rows = data.rows
                objective = _BasisIntervalObjective(
                    ctx,
                    type(rows)(
                        rows.lower,
                        rows.upper,
                        v,
                        rows.effective_n,
                        rows.count,
                        rows.support,
                    ),
                )
            else:
                objective = _BasisPointObjective(
                    ctx, _PointStatistics(v @ self._phis[comp], 1.0, data.size)
                )
            kwargs = {}
            if start is not None:
                kwargs["start"] = self.blocks.part(comp, start)
                kwargs["start_face"] = (
                    tuple((0, side) for b, side in face[0] if b == comp),
                    (),
                )
            result = _solve_basis(
                objective,
                options=self.options,
                start_cuts=self._cuts.get(comp),
                **kwargs,
            )
            self._cuts[comp] = _cut_rows(result)
            theta[self.blocks.index[comp]] = result.params
            suppressed += [(comp, side) for _, side in result.face.suppressed]
            statuses.append(result.status)
        return theta, np.log(mass), (tuple(suppressed), ()), tuple(statuses)

    def em_map(self, state, /):
        """One EM step from an evaluated state.

        Parameters
        ----------
        state : _MixtureState
            Evaluated state.
        """

        theta, log_weights, face, statuses = self.m_step(state.responsibilities, state)
        if self.strict and any(status not in _CONVERGED for status in statuses):
            raise _StalledRefit(statuses)
        return self.evaluate(theta, log_weights, face, statuses)

    def polish(self, state, /):
        """Joint Newton polish of the observed likelihood from ``state``.

        Parameters
        ----------
        state : _MixtureState
            Evaluated state.
        """

        return _solve_basis(
            self.joint,
            options=self.options,
            blocks=self.joint_blocks,
            start=self.joint_point(state.theta, state.log_weights),
            start_face=state.face,
        )

    def split(self, x, /):
        """``(theta, log_weights)`` of joint coordinates with logits.

        Parameters
        ----------
        x : numpy.ndarray
            Joint coordinates with logits.
        """
        n = self.blocks.n
        eta = np.concatenate([x[n:], [0.0]])
        top = float(np.max(eta))
        lse = top + math.log(float(np.sum(np.exp(eta - top))))
        return np.asarray(x[:n]), eta - lse


def _cut_rows(result, /):
    """Canonical curvature cut rows of a solve, shape ``(k, d)``.

    Parameters
    ----------
    result : _BasisNewtonResult
        Completed solve.
    """
    rows = result.rows
    return rows.rows[[kind == "cut" for kind in rows.kinds]]


def _stack(state, /):
    """Coefficients and log weights of a state as one vector (SQUAREM).

    Parameters
    ----------
    state : _MixtureState
        Evaluated state.
    """
    return np.concatenate([state.theta, state.log_weights])


def _em_phase(problem, state, max_steps, tolerance, accelerate, history, /):
    """SQUAREM-accelerated EM until the likelihood stalls.

    Parameters
    ----------
    problem : _MixtureProblem
        Model, data and component count.
    state : _MixtureState
        Start.
    max_steps : int
        EM map evaluation limit.
    tolerance : float
        Relative log-likelihood increase that ends the phase.
    accelerate : bool
        Use SQUAREM.
    history : list
        Log-likelihood history, appended in place.

    Returns
    -------
    tuple
        ``(state, steps, settled)``: ``settled`` is true when the phase
        ended on its stall or tolerance test rather than its step budget.
    """

    n = problem.blocks.n
    steps = 0
    while steps < int(max_steps):
        _check_fit_time("EM exploration")
        state1 = problem.em_map(state)
        steps += 1
        if not accelerate:
            new = state1
        else:
            state2 = problem.em_map(state1)
            steps += 1
            new = state2
            # SQUAREM (scheme S3).  alpha = -1 reproduces the second EM step;
            # steeper extrapolations are tried first, halved toward it, and
            # one is kept only if its stabilizing EM step beats two plain EM
            # steps, so the iteration stays monotone.
            x0 = _stack(state)
            step = _stack(state1) - x0
            curvature = _stack(state2) - 2.0 * (x0 + step) + x0
            norm = float(np.linalg.norm(curvature))
            alpha = -float(np.linalg.norm(step)) / norm if norm > 0.0 else -1.0
            for _ in range(4):
                if alpha > -1.5:
                    break
                point = x0 - 2.0 * alpha * step + alpha * alpha * curvature
                lw = point[n:] - np.max(point[n:])
                lw = lw - math.log(float(np.sum(np.exp(lw))))
                try:
                    trial = problem.em_map(problem.evaluate(point[:n], lw, state2.face))
                    steps += 1
                except NUMERIC_FAILURES as exc:
                    # An extrapolation outside the proper family (or one the
                    # normalizer cannot resolve) declines this acceleration.
                    _reraise_if_debug(exc, "SQUAREM stabilization step", routine=True)
                    trial = None
                if trial is not None and trial.log_likelihood >= new.log_likelihood:
                    new = trial
                    break
                alpha = 0.5 * (alpha - 1.0)
        increase = new.log_likelihood - state.log_likelihood
        if increase < 0.0:
            return state, steps, True
        state = new
        history.append(state.log_likelihood)
        if increase <= tolerance * (1.0 + abs(state.log_likelihood)):
            return state, steps, True
    return state, steps, False


@dataclass(frozen=True)
class _EMExploration:
    """The first EM phase of a fit, kept so a finalist can continue it.

    Restarting a finalist from the same responsibilities would recompute
    this phase exactly before polishing (39% of a 490 s automatic-count fit
    in a profile).  Continuing is exact when the finalist's first
    phase extends this one (``extends``): EM is deterministic, and the phase
    loop carries only the iterate, its step count and the history.

    Parameters
    ----------
    state : _MixtureState
        Iterate at the end of the phase.
    history : tuple of float
        Log likelihood at the start and after every accepted EM cycle.
    steps : int
        EM map evaluations of the phase.
    settled : bool
        Whether the phase ended on its stall or tolerance test (a longer
        budget ends it at the same iterate).
    schedule : _EMOptions
        EM schedule of the phase.
    """

    state: _MixtureState
    history: tuple
    steps: int
    settled: bool
    schedule: _EMOptions

    def extends(self, em, /):
        """Whether the first phase of schedule ``em`` continues this phase exactly.

        Parameters
        ----------
        em : _EMOptions
            Finalist schedule.
        """
        return (
            int(em.max_rounds) >= 1
            and int(em.max_steps) >= int(self.schedule.max_steps)
            and float(em.tolerance) == float(self.schedule.tolerance)
            and bool(em.accelerate) == bool(self.schedule.accelerate)
        )


def _initial_state(problem, responsibilities, /):
    """The evaluated cold M-step from ``responsibilities``.

    Parameters
    ----------
    problem : _MixtureProblem
        Model, data and component count.
    responsibilities : numpy.ndarray
        Initial ``r_ik``, shape ``(R, K)``.
    """
    theta, log_weights, face, statuses = problem.m_step(responsibilities, None)
    return problem.evaluate(theta, log_weights, face, statuses)


def _explore_em(problem, responsibilities, em, /):
    """One EM phase from ``responsibilities``, resumable by ``_run_em``.

    Parameters
    ----------
    problem : _MixtureProblem
        Model, data and component count.
    responsibilities : numpy.ndarray
        Initial ``r_ik``, shape ``(R, K)``.
    em : _EMOptions
        Exploration schedule (``max_rounds >= 1``).

    Returns
    -------
    _EMExploration
    """
    state = _initial_state(problem, responsibilities)
    history = [state.log_likelihood]
    state, steps, settled = _em_phase(
        problem, state, em.max_steps, em.tolerance, em.accelerate, history
    )
    return _EMExploration(state, tuple(history), steps, settled, em)


def _run_em(
    problem,
    responsibilities,
    /,
    *,
    em_options=None,
    polish=True,
    initialization="given",
    start=None,
    resume=None,
):
    """Fit a mixture from initial responsibilities: EM, polish, verify.

    Parameters
    ----------
    problem : _MixtureProblem
        Model, data and component count.
    responsibilities : numpy.ndarray
        Initial ``r_ik``, shape ``(R, K)``.
    em_options : _EMOptions or None
        EM schedule.
    polish : bool
        ``False`` runs one EM phase only (status ``em_only``).
    initialization : str
        Label recorded in the fit.
    start : _MixtureState or None
        Evaluated state to continue from instead of an M-step from
        ``responsibilities`` (warm refits after a span grows).
    resume : _EMExploration or None
        First EM phase already run on ``problem`` from ``responsibilities``
        (multistart exploration); the fit continues it, exactly as if the
        phase had been run here.  Its schedule must be extended by
        ``em_options`` (``_EMExploration.extends``).

    Returns
    -------
    _MixtureFit

    Raises
    ------
    ValueError
        If ``resume`` is combined with ``start`` or not extended by
        ``em_options``.
    """

    em = _resolve_em_options(em_options)
    options = problem.options
    if resume is not None:
        if start is not None or not resume.extends(em):
            raise ValueError(
                "resume must be a first EM phase that em_options extends, without start"
            )
        state, history = resume.state, list(resume.history)
        em_steps, rounds, need_em = int(resume.steps), 1, False
        if not resume.settled:
            # The phase stopped on its budget: continue it on the remainder.
            state, steps, _ = _em_phase(
                problem,
                state,
                int(em.max_steps) - em_steps,
                em.tolerance,
                em.accelerate,
                history,
            )
            em_steps += steps
    else:
        state = _initial_state(problem, responsibilities) if start is None else start
        history = [state.log_likelihood]
        em_steps = rounds = 0
        need_em = True
    polish_iterations = 0
    status, result = "em_only", None
    while True:
        if need_em:
            if rounds >= int(em.max_rounds):
                break
            rounds += 1
            state, steps, _ = _em_phase(
                problem, state, em.max_steps, em.tolerance, em.accelerate, history
            )
            em_steps += steps
        if not polish:
            break
        result = problem.polish(state)
        polish_iterations += result.newton_iterations
        polished_ll = -float(result.objective_value)
        # Two evaluations of the same mean likelihood can differ by a few
        # ulps. Preserve the strict decrease guard beyond that rounding
        # envelope; an equal-likelihood polished point must still pass its
        # optimizer/shape certificate and the subsequent EM verification.
        roundoff = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(polished_ll), abs(state.log_likelihood))
        )
        # Two shape-constrained optima (the M-steps' and the polish's) are
        # each verified within the curvature threshold on their own cut sets,
        # so they agree only to the accuracy floor.
        floor = max(roundoff, float(options.accuracy_floor))
        if polished_ll < state.log_likelihood - floor:
            status = "em_only"
            break
        theta, log_weights = problem.split(result.params)
        face = (
            result.face.suppressed,
            tuple(problem.joint_blocks.labels[i] for i in result.face.zero_logs),
        )
        state = problem.evaluate(theta, log_weights, face, (result.status,))
        history.append(state.log_likelihood)
        status = result.status
        if status not in _CONVERGED:
            need_em = True
            continue
        # Verification: one EM step from the polished point.
        verified = problem.em_map(state)
        em_steps += 1
        climb = verified.log_likelihood - polished_ll
        if climb <= max(
            float(result.final_decrease_bound), options.certified_tolerance, roundoff
        ):
            break
        if climb <= floor:
            # Within the accuracy floor: converged, but not to the certificate.
            status = "converged_approximately"
            break
        # EM still climbs: the polish sat on the wrong face.
        state = verified
        history.append(state.log_likelihood)
        status, need_em = "em_only", True
    joint = problem.joint_blocks
    n_face = int(_matching_face(joint, *state.face).basis(joint.n).shape[1])
    information = None
    if result is not None and status in _CONVERGED:
        ev = result.evaluation
        information = ev.fisher - ev.missing_information
    return _MixtureFit(
        models=problem.models,
        thetas=tuple(problem.blocks.part(c, state.theta) for c in range(problem.k)),
        weights=np.exp(state.log_weights),
        log_likelihood=float(state.log_likelihood),
        status=status,
        responsibilities=state.responsibilities,
        face=state.face,
        n_face_parameters=n_face,
        observed_information=information,
        em_iterations=em_steps,
        polish_iterations=polish_iterations,
        rounds=rounds,
        history=tuple(history),
        initialization=initialization,
        result=result,
    )


# ---------------------------------------------------------------------------
# Multistart fit
# ---------------------------------------------------------------------------


def _mixture_data(inputs, /):
    """Rows of a mixture fit and the expansion from merged to original rows.

    Interval rows are merged exactly (summed weights), grouping duplicate
    rows; point rows are kept.

    Parameters
    ----------
    inputs : _BasisInputs
        Validated observations.

    Returns
    -------
    tuple
        ``(data, inverse, representatives)``: ``_MixtureRows``, the merged
        row of each original row (``None`` for points) and the initializer
        representatives per original row.
    """

    s = inputs.samples
    n = s.shape[0]
    w = np.full(n, 1.0 / n) if inputs.weights is None else np.asarray(inputs.weights)
    support = tuple(map(float, inputs.spec.support))
    if not inputs.intervals:
        data = _MixtureRows(
            w / w.sum(), x=s[:, 0].copy(), effective_n=inputs.effective_n
        )
        return data, None, s[:, 0]
    unique, inverse = np.unique(s, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    merged = np.bincount(inverse, weights=w, minlength=unique.shape[0])
    rows = _interval_rows(support, unique, merged)
    if rows.lower.size != unique.shape[0]:
        raise RuntimeError("internal error: merged interval rows lost a row")
    return (
        _MixtureRows(rows.weights, rows=rows, effective_n=inputs.effective_n),
        inverse,
        _interval_initial_representatives(s, support),
    )


def _merge_responsibilities(r, w, inverse, size, /):
    """Original-row responsibilities averaged within merged rows.

    Parameters
    ----------
    r : numpy.ndarray
        Responsibilities of the original rows, shape ``(R, K)``.
    w : numpy.ndarray
        Original row weights.
    inverse : numpy.ndarray
        Merged-row index of each original row.
    size : int
        Number of merged rows.
    """
    if inverse is None:
        return r
    out = np.zeros((size, r.shape[1]))
    np.add.at(out, inverse, w[:, None] * r)
    return out / out.sum(axis=1, keepdims=True)


def _sound(fit, data, /):
    """Soundness: converged, and point components determined by enough points.

    A point component resting on at most its coefficient count plus two
    effective distinct locations is not determined by the data.

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    data : _MixtureRows
        Observations.
    """

    if fit is None or fit.status not in _CONVERGED:
        return False
    if data.intervals:
        return True
    rows = data.x[:, None]
    inverse = _distinct_location_index(rows)
    for comp in range(len(fit.thetas)):
        d = fit.models[comp].n_coefficients
        n_eff = _effective_distinct_point_count(
            data.x, data.weights * fit.responsibilities[:, comp], inverse=inverse
        )
        if n_eff <= d + 2:
            return False
    return True


@dataclass
class _Prepared:
    """Validated inputs of a mixture fit (pooled basis data and mixture rows)."""

    inputs: object
    pooled: object
    data: _MixtureRows
    inverse: object
    representatives: np.ndarray
    weights: np.ndarray


def _prepare(spec, samples, weights, /):
    """Input rules, the pooled-data basis builder and the mixture rows.

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    samples : array_like
        Point or interval observations.
    weights : array_like or None
        Observation weights.
    """

    inputs = _normalize_basis_inputs(spec, samples, weights)
    if inputs.intervals:
        pooled = _IntervalData(
            tuple(map(float, spec.support)),
            inputs.samples,
            inputs.weights,
            inputs.effective_n,
        )
    else:
        pooled = _PointData(inputs.samples[:, 0], inputs.weights, inputs.effective_n)
    data, inverse, representatives = _mixture_data(inputs)
    n = inputs.samples.shape[0]
    w = np.full(n, 1.0 / n) if inputs.weights is None else np.asarray(inputs.weights)
    return _Prepared(inputs, pooled, data, inverse, representatives, w)


def _recompile(model, /, **changes):
    """``model``'s request with ``changes`` in the same coordinate and geometry.

    Parameters
    ----------
    model : _BasisModel
        A compiled basis.
    **changes :
        ``_BasisRequest`` fields to replace.
    """

    return _compile_basis(dataclasses.replace(model.request, **changes), model.geometry)


def _component_models(prepared, specs, posterior, /):
    """One basis per component in its own responsibility-weighted coordinate.

    Each component is fitted in the coordinate of its responsibility-weighted
    data, which keeps a narrow or remote component's potential well
    conditioned (a pooled coordinate needs coefficients of order
    ``(scale ratio)^2`` and loses digits to cancellation).  ``posterior``
    holds the original rows' responsibilities; ``specs`` one concrete
    request per component.

    Parameters
    ----------
    prepared : _Prepared
        Prepared data and basis builder.
    specs : sequence of BasisSpec
        One concrete request per component.
    posterior : numpy.ndarray
        Responsibilities fixing each component's coordinate, shape ``(R, K)``.
    """
    r = np.asarray(posterior, dtype=np.float64)
    return tuple(
        prepared.pooled.build_weighted(spec, r[:, k]) for k, spec in enumerate(specs)
    )


def _multistart(
    prepared, specs, rng, responsibilities, search, final_em, options, /, grow=None
):
    """Multistart over initializers for concrete component ``specs``.

    Every initializer candidate is explored by EM alone with raw and
    sharpened responsibilities; the ``finalists`` best explorations are run
    to convergence (continuing their exploration phase when ``final_em``
    extends it, ``_EMExploration``) and, with automatic selection, grown
    (``grow``, every finalist grows); a sound fit outranks an unsound
    one, then the higher score: the log likelihood, BIC-penalized by the
    free parameters when finalists grow to different sizes.  A lone
    valley seed whose best fit is unsound widens the search to the other
    families.

    Each start's components are compiled in its own responsibility-weighted
    coordinates (``_component_models``).

    Parameters
    ----------
    prepared : _Prepared
        Prepared data.
    specs : sequence of BasisSpec
        One concrete request per component.
    rng : numpy.random.Generator
        Initializer randomness.
    responsibilities : numpy.ndarray or None
        Given starts replacing the initializer candidates.
    search : _MixtureSearchOptions
        Multistart policy.
    final_em : _EMOptions
        EM policy of the finalists.
    options : _NewtonOptions
        Newton policy.
    grow : callable or None
        Automatic growth applied to each finalist.

    Returns
    -------
    _MixtureFit
        On the merged rows.
    """

    data, k = prepared.data, len(specs)
    penalty = 0.0
    if grow is not None:
        n_eff = data.n_eff
        penalty = 0.5 * math.log(n_eff) / n_eff

    def prepare(posterior, treatment):
        if treatment == "sharpened":
            posterior = np.square(posterior)
            posterior = posterior / posterior.sum(axis=1, keepdims=True)
        start = _merge_responsibilities(
            posterior, prepared.weights, prepared.inverse, data.size
        )
        if not data.intervals:
            # Before compiling: a component on one location has no coordinate.
            _check_point_estimability(data.x[:, None], data.weights, start)
        problem = _MixtureProblem(
            _component_models(prepared, specs, posterior), data, options
        )
        return problem, start

    explore_em = _EMOptions()

    def finish(problem, start, exploration, label):
        if exploration.extends(final_em):
            return _run_em(
                problem,
                None,
                em_options=final_em,
                initialization=label,
                resume=exploration,
            )
        return _run_em(problem, start, em_options=final_em, initialization=label)

    def rank(fit):
        score = float(fit.log_likelihood) - penalty * fit.n_face_parameters
        return (_sound(fit, data), score)

    treatments = tuple(search.treatments)

    def explore(candidates):
        explored, failure = [], None
        for name, initial, _ in candidates:
            for treatment in treatments:
                label = f"{name}/direct/{treatment}"
                try:
                    problem, start = prepare(
                        np.asarray(initial, dtype=np.float64), treatment
                    )
                    exploration = _explore_em(problem, start, explore_em)
                    fit = _run_em(
                        problem,
                        None,
                        em_options=explore_em,
                        polish=False,
                        initialization=label,
                        resume=exploration,
                    )
                except _PointMixtureEstimabilityFailure as exc:
                    failure = exc
                    continue
                except NUMERIC_FAILURES as exc:
                    _reraise_if_debug(exc, f"mixture EM from {label}", routine=True)
                    failure = exc
                    continue
                explored.append(
                    (fit.log_likelihood, label, problem, start, exploration)
                )
        explored.sort(key=lambda item: -item[0])
        best = None
        for _, label, problem, start, exploration in explored[
            : max(1, int(search.finalists))
        ]:
            try:
                fit = finish(problem, start, exploration, label)
                if grow is not None:
                    fit = grow(fit)
            except _PointMixtureEstimabilityFailure as exc:
                failure = exc
                continue
            except NUMERIC_FAILURES as exc:
                _reraise_if_debug(exc, f"mixture fit from {label}", routine=True)
                failure = exc
                continue
            if best is None or rank(fit) > rank(best):
                best = fit
        return best, failure

    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    n_original = prepared.inputs.samples.shape[0]
    weights = prepared.inputs.weights
    if responsibilities is not None:
        candidates = [("given", np.asarray(responsibilities, dtype=np.float64), None)]
    elif k == 1:
        candidates = [("single", np.ones((n_original, 1)), None)]
    else:
        candidates = _initial_responsibility_candidates(
            prepared.representatives, k, generator, weights=weights
        )
    best, failure = explore(candidates)
    if (
        responsibilities is None
        and k > 1
        and candidates[0][0] == "valley"
        and not _sound(best, data)
    ):
        alternatives = _initial_responsibility_candidates(
            prepared.representatives,
            k,
            generator,
            weights=weights,
            include_valley=False,
        )
        other, other_failure = explore(alternatives)
        if other is not None and (best is None or rank(other) > rank(best)):
            best = other
        failure = failure if failure is not None else other_failure
    if best is None:
        if isinstance(failure, _PointMixtureEstimabilityFailure):
            raise failure
        raise RuntimeError("no mixture initialization produced a fit") from failure
    return best


def _expanded(fit, prepared, /):
    """``fit`` with responsibilities of the original rows.

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    prepared : _Prepared
        Prepared data (merged-row map).
    """
    if prepared.inverse is None:
        return fit
    return dataclasses.replace(
        fit, responsibilities=fit.responsibilities[prepared.inverse]
    )


# ---------------------------------------------------------------------------
# Selection inside a mixture
# ---------------------------------------------------------------------------


def _lifted_point(fit, models, /):
    """``(theta, log_weights, face)`` of ``fit`` lifted into ``models``.

    Each component's coefficients are lifted into its (equal or larger)
    span of the same geometry (``basis_selection._lift_matrix``); the face's
    suppressed ends and zero log labels carry over.

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    models : sequence of _BasisModel
        Larger component bases nesting ``fit.models``.
    """

    blocks = _mixture_blocks(models)
    theta = np.zeros(blocks.n)
    for comp, (old, new) in enumerate(zip(fit.models, models, strict=True)):
        lifted = (
            fit.thetas[comp]
            if old is new
            else _lift_matrix(old, new) @ fit.thetas[comp]
        )
        theta[blocks.index[comp]] = lifted
    return theta, np.log(fit.weights), fit.face


class _StalledRefit(RuntimeError):
    """A growth candidate's M-step did not converge."""


def _refit(fit, models, data, options, em_options, /, *, strict=False):
    """Warm refit of ``fit`` with components in ``models`` (no multistart).

    ``strict`` (growth candidates) raises ``_StalledRefit`` at the first
    M-step that does not converge from its lifted warm start.

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    models : sequence of _BasisModel
        Component bases of the refit.
    data : _MixtureRows
        Observations.
    options : _NewtonOptions
        Newton policy.
    em_options : _EMOptions or None
        EM policy.
    strict : bool
        End at the first M-step that does not converge.
    """
    problem = _MixtureProblem(models, data, options)
    problem.strict = bool(strict)
    theta, log_weights, face = _lifted_point(fit, problem.models)
    start = problem.evaluate(theta, log_weights, face)
    return _run_em(
        problem,
        None,
        em_options=em_options,
        initialization=fit.initialization,
        start=start,
    )


def _mixture_span_diagnostic(fit, comp, larger, data, config, label, /):
    """Score test of component ``comp`` in ``larger`` at the fitted mixture.

    The efficient score in the joint coordinates (with every other parameter,
    the shared amplitudes and the logits as nuisance; observed information
    ``G - M`` of the joint likelihood).  The test itself is the single fits'
    ``basis_selection._score_test``: statistics are the joint coordinates,
    the probes the component's coefficients in ``larger`` and the nuisance
    every other coordinate plus its lifted current span; per-row values are
    the joint scores, with the point policy for point rows and the
    interval policy (no guards) for merged interval rows.

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    comp : int
        Tested component.
    larger : _BasisModel
        Larger span of that component.
    data : _MixtureRows
        Observations.
    config : _DegreeSelectionConfig
        Selection policy.
    label : str
        Name of the tested extension.

    Returns
    -------
    _SpanDiagnostic
    """

    models = list(fit.models)
    models[comp] = larger
    models = tuple(models)
    blocks = _mixture_blocks(models, logits=True)
    theta, log_weights, face = _lifted_point(fit, models)
    x = np.concatenate([theta, np.asarray(log_weights[:-1]) - float(log_weights[-1])])
    objective = _joint_objective(models, data, scores=True)
    if face[0] or face[1]:
        basis = _matching_face(blocks, *face).basis(blocks.n)
        objective = objective.on_face(basis)
    ev = objective(x)
    rows = objective.evaluator().scores()
    lift = _lift_matrix(fit.models[comp], larger)
    private = [j for j, c in enumerate(larger.columns) if not c.nonnegative]
    probe_coords = blocks.index[comp][private]
    others = np.setdiff1d(np.arange(blocks.n), probe_coords)
    eye = np.eye(blocks.n)
    nuisance = np.hstack([eye[:, others], blocks.embed(comp, lift.T).T])
    # Only the private part of the lifted span is new information.
    nuisance[others, others.size :] = 0.0
    probe = eye[:, probe_coords]
    return _score_test(
        ev.gradient,
        ev.fisher - ev.missing_information,
        rows,
        data.weights,
        nuisance,
        probe,
        data.n_eff,
        not data.intervals,
        config,
        label,
        larger.n_coefficients - fit.models[comp].n_coefficients,
    )


@dataclass(frozen=True)
class _MixtureStep:
    """One automatic decision inside a mixture (degree-growth history).

    Parameters
    ----------
    kind : str
        ``"degree"`` or ``"expansion"``.
    diagnostics : tuple
        ``(component, _SpanDiagnostic)`` pairs tested at this step.
    chosen : tuple or None
        ``(component, label)`` grown, or ``None`` when selection stopped.
    lr_p_value : float
        Expansion steps that refitted: the likelihood-ratio confirmation.
    """

    kind: str
    diagnostics: tuple
    chosen: object
    lr_p_value: float = math.nan
    failure: str = ""


def _grow(
    fit, spec, prepared, config, options, em_options, /, *, degree, span, growable=None
):
    """Per-component growth of a fitted mixture.

    Degree (``bulk_degree="auto"``): each component starts at the minimum
    admissible degree; the component whose next probe block (the degree
    ``d + probe_block_size`` minimal span) has the smallest supported p-value
    in the joint diagnostic grows to its next admissible degree by a warm
    lifted refit, which must not lose likelihood.  Span
    (``expansion="auto"``): per component a joint gate of its
    full span; if one rejects, the one-slot extension (``_grow_basis``) with
    the smallest p-value over the rejecting components is refitted and kept
    only if its likelihood ratio also rejects.

    Parameters
    ----------
    degree, span : bool
        Grow degrees (``spec`` gives the degree ladder's basis requests) and
        spans.
    growable : tuple of bool or None
        Components whose degree is automatic (default all); the others keep
        their requested degree (per-component degree policies).

    Returns
    -------
    _MixtureFit
        With the decisions appended to ``selection``.
    """

    data = prepared.data
    k = len(fit.models)
    steps = []
    cache = {}

    def at_degree(comp, degree):
        # The component's own coordinate and geometry (its fit's model).
        key = (comp, degree)
        if key not in cache:
            cache[key] = _recompile(
                fit.models[comp], bulk_degree=degree, expansion=_MINIMAL
            )
        return cache[key]

    if degree:
        degrees = _bulk_degrees(spec)
        growable = (True,) * k if growable is None else tuple(growable)
        current = [int(m.request.bulk_degree) for m in fit.models]
        while True:
            diagnostics = []
            untestable = None
            for comp in range(k):
                if not growable[comp]:
                    continue
                index = degrees.index(current[comp])
                if index == len(degrees) - 1:
                    continue
                probe_degree = min(
                    current[comp] + int(config.probe_block_size), degrees[-1]
                )
                try:
                    diagnostics.append(
                        (
                            comp,
                            _tested(
                                _mixture_span_diagnostic,
                                fit,
                                comp,
                                at_degree(comp, probe_degree),
                                data,
                                config,
                                f"{comp}:bulk_degree={probe_degree}",
                            ),
                        )
                    )
                except _UntestableSpan as exc:
                    untestable = (
                        f"{comp}:bulk_degree={probe_degree} is untestable ({exc})"
                    )
                    break
            if untestable is not None:
                # As the single-fit ladder: the growth ends at the fitted
                # degrees, recorded.
                steps.append(
                    _MixtureStep("degree", tuple(diagnostics), None, failure=untestable)
                )
                break
            supported = [(c, dg) for c, dg in diagnostics if dg.should_expand]
            if not supported:
                steps.append(_MixtureStep("degree", tuple(diagnostics), None))
                break
            comp, _ = min(
                supported, key=lambda item: (item[1].p_value, -item[1].score, item[0])
            )
            new_degree = degrees[degrees.index(current[comp]) + 1]
            models = list(fit.models)
            models[comp] = at_degree(comp, new_degree)
            try:
                candidate = _refit(
                    fit, tuple(models), data, options, em_options, strict=True
                )
            except _StalledRefit as exc:
                # As the single-fit ladder: a rung whose warm refit
                # does not converge ends the growth at the converged smaller
                # degree, recorded.  (A log-concave component spanning two of
                # three modes grew to degree 10, where each M-step took 15-65
                # s without converging: 20 minutes for one selection.)
                steps.append(
                    _MixtureStep(
                        "degree",
                        tuple(diagnostics),
                        None,
                        failure=(
                            f"{comp}:bulk_degree={new_degree} M-step did not "
                            f"converge ({', '.join(exc.args[0])})"
                        ),
                    )
                )
                break
            if candidate.log_likelihood + 1e-8 * (1.0 + abs(fit.log_likelihood)) < (
                fit.log_likelihood
            ):
                steps.append(_MixtureStep("degree", tuple(diagnostics), None))
                break
            steps.append(
                _MixtureStep(
                    "degree", tuple(diagnostics), (comp, f"bulk_degree={new_degree}")
                )
            )
            fit, current[comp] = candidate, new_degree
    if span:
        while True:
            gates = []
            try:
                for comp in range(k):
                    full = _full_basis(fit.models[comp])
                    if full.n_coefficients == fit.models[comp].n_coefficients:
                        continue
                    gates.append(
                        (
                            comp,
                            _tested(
                                _mixture_span_diagnostic,
                                fit,
                                comp,
                                full,
                                data,
                                config,
                                f"{comp}:joint",
                            ),
                        )
                    )
                rejecting = [c for c, dg in gates if dg.should_expand]
                candidates = []
                for comp in rejecting:
                    for label, larger in _grow_basis(fit.models[comp]):
                        dg = _tested(
                            _mixture_span_diagnostic,
                            fit,
                            comp,
                            larger,
                            data,
                            config,
                            f"{comp}:{label}",
                        )
                        if dg.rank > 0:
                            candidates.append((comp, larger, dg))
            except _UntestableSpan as exc:
                steps.append(
                    _MixtureStep(
                        "expansion",
                        tuple(gates),
                        None,
                        failure=f"a larger span is untestable ({exc})",
                    )
                )
                break
            if not rejecting:
                steps.append(_MixtureStep("expansion", tuple(gates), None))
                break
            if not candidates:
                steps.append(_MixtureStep("expansion", tuple(gates), None))
                break
            comp, larger, chosen = min(candidates, key=lambda c: c[2].p_value)
            models = list(fit.models)
            models[comp] = larger
            try:
                grown = _refit(
                    fit, tuple(models), data, options, em_options, strict=True
                )
            except _StalledRefit as exc:
                steps.append(
                    _MixtureStep(
                        "expansion",
                        tuple(gates),
                        None,
                        failure=(
                            f"{comp}:{chosen.label} M-step did not converge "
                            f"({', '.join(exc.args[0])})"
                        ),
                    )
                )
                break
            lr = 2.0 * data.n_eff * (grown.log_likelihood - fit.log_likelihood)
            lr_p = float(chi2.sf(max(lr, 0.0), chosen.rank))
            tested = tuple(gates) + tuple((c, d) for c, _, d in candidates)
            if not lr_p < float(config.alpha):
                steps.append(_MixtureStep("expansion", tested, None, lr_p))
                break
            steps.append(_MixtureStep("expansion", tested, (comp, chosen.label), lr_p))
            fit = grown
    return dataclasses.replace(fit, selection=fit.selection + tuple(steps))


def _shared_log_amplitude(fit, side, /):
    """Fitted shared amplitude of a side's endpoint log term (0 if absent).

    Parameters
    ----------
    fit : _MixtureFit
        A fitted mixture.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    model = fit.models[0]
    for i, column in enumerate(model.columns):
        if column.nonnegative and model.functions[column.terms[0][0]].side == side:
            return float(fit.thetas[0][i])
    return 0.0


def _single_mixture_fit(single, n_rows, /):
    """A single fit (``basis_selection._BasisFit``) as a one-component mixture.

    Parameters
    ----------
    single : _BasisFit
        A single fit.
    n_rows : int
        Number of observation rows.
    """
    r = single.result
    return _MixtureFit(
        models=(single.model,),
        thetas=(r.params,),
        weights=np.ones(1),
        log_likelihood=-float(r.objective_value),
        status=r.status,
        responsibilities=np.ones((n_rows, 1)),
        face=(
            r.face.suppressed,
            tuple(single.model.columns[i].label for i in r.face.zero_logs),
        ),
        n_face_parameters=int(r.face.basis(single.model.n_coefficients).shape[1]),
        observed_information=r.evaluation.fisher - r.evaluation.missing_information,
        em_iterations=0,
        polish_iterations=0,
        rounds=0,
        history=(),
        initialization="single",
        result=r,
    )


def _fit_mixture_auto(
    spec,
    samples,
    k,
    weights=None,
    /,
    *,
    rng=0,
    responsibilities=None,
    degrees=None,
    config=None,
    search_options=None,
    em_options=None,
    options=None,
):
    """Fit a ``k``-component mixture, resolving every ``"auto"`` field.

    Order (as single fits): per-component degree growth with every
    automatic endpoint log enabled (``_grow``, each multistart finalist
    grown, BIC-ranked); backward elimination of automatic shared log
    terms (``_select_boundary_terms``: one-sided likelihood ratio at the same
    degrees, the reduced fit from the larger fit's responsibilities, then
    the growth rerun with the final terms); finally per-component span
    growth.  One component is the single fit (``basis_selection._fit_basis``).

    ``responsibilities`` (original rows), when given, replace the
    initializer candidates of every multistart (refinement of a screened
    component count).  ``degrees`` gives one policy per component (an
    integer, or ``"auto"`` to grow from the lowest admissible degree; the
    public ``component_options``) in place of ``spec.bulk_degree``.

    Parameters
    ----------
    spec : BasisSpec
        Basis request (``"auto"`` fields resolved).
    samples : array_like
        Point or interval observations.
    k : int
        Component count.
    weights : array_like or None
        Observation weights.
    rng : int or numpy.random.Generator
        Initializer randomness.
    responsibilities : numpy.ndarray or None
        Starting responsibilities of the original rows.
    degrees : tuple or None
        One degree policy per component.
    config : _DegreeSelectionConfig or None
        Selection policy.
    search_options : _MixtureSearchOptions or None
        Multistart policy.
    em_options : _EMOptions or None
        EM policy.
    options : _NewtonOptions or None
        Newton policy.

    Returns
    -------
    tuple
        ``(fit, steps, boundary_p_values)``; the fit's responsibilities
        refer to the original rows.
    """

    k = int(k)
    config = _DegreeSelectionConfig() if config is None else config
    options = _NewtonOptions() if options is None else options
    search = _resolve_search_options(search_options)
    final_em = _resolve_em_options(em_options)
    prepared = _prepare(spec, samples, weights)
    if k == 1:
        single = _fit_basis(spec, samples, weights, config=config, options=options)
        fit = _single_mixture_fit(single, prepared.inputs.samples.shape[0])
        return fit, single.history, single.boundary_p_values
    lower, upper = prepared.inputs.log_policy
    degree_spec = dataclasses.replace(
        spec,
        expansion=_MINIMAL if spec.expansion == _AUTO_EXPANSION else spec.expansion,
    )

    policies = (spec.bulk_degree,) * k if degrees is None else tuple(degrees)
    if len(policies) != k:
        raise ValueError("one degree policy per component is required")
    growable = tuple(p == _AUTO for p in policies)
    any_auto = any(growable)

    def fit_logs(lo, up):
        concrete = dataclasses.replace(degree_spec, log_lower=lo, log_upper=up)
        lowest = _bulk_degrees(spec)[0]
        specs = tuple(
            dataclasses.replace(concrete, bulk_degree=lowest if p == _AUTO else int(p))
            for p in policies
        )
        grow_spec = dataclasses.replace(concrete, expansion=_MINIMAL)

        def grow(fit):
            return _grow(
                fit,
                grow_spec,
                prepared,
                config,
                options,
                final_em,
                degree=True,
                span=False,
                growable=growable,
            )

        return _multistart(
            prepared,
            specs,
            rng,
            responsibilities,
            search,
            final_em,
            options,
            grow=grow if any_auto else None,
        )

    def fit_reduced(fit, lo, up):
        models = tuple(_recompile(m, log_lower=lo, log_upper=up) for m in fit.models)
        problem = _MixtureProblem(models, prepared.data, options)
        return _run_em(problem, fit.responsibilities, em_options=final_em)

    if _AUTO in (lower, upper):
        fit, flags, p_values = _select_boundary_terms(
            fit_logs,
            lambda f: -float(f.log_likelihood),
            _shared_log_amplitude,
            lower,
            upper,
            prepared.data.n_eff,
            fit_reduced=fit_reduced,
            refit=fit_logs if any_auto else None,
            alpha=BOUNDARY_ALPHA,
        )
        lower, upper = flags
    else:
        fit, p_values = fit_logs(lower, upper), (math.nan, math.nan)
    if spec.expansion == _AUTO_EXPANSION:
        fit = _grow(
            fit, spec, prepared, config, options, final_em, degree=False, span=True
        )
    return _expanded(fit, prepared), fit.selection, p_values


# ---------------------------------------------------------------------------
# Component count
# ---------------------------------------------------------------------------


class _CandidateRejected(RuntimeError):
    """A component-count candidate the BIC search does not score."""


class _ComponentSelectionError(RuntimeError):
    """No component count produced a scored fit; carries the diagnostics."""

    def __init__(self, diagnostics, /):
        """Store the selection diagnostics."""
        super().__init__("no component count produced a scored mixture fit")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class _ComponentSelection:
    """Selected component count and its fit.

    Parameters
    ----------
    n_components : int
        BIC-selected count.
    fit : _MixtureFit
        The selected count's full-data fit (original rows).
    steps : tuple
        Its automatic decisions (``_fit_mixture_auto``).
    boundary_p_values : tuple of float
        Its endpoint-log likelihood-ratio p-values.
    diagnostics : dict
        Search record: screens, refinements, admissions.
    """

    n_components: int
    fit: _MixtureFit
    steps: tuple
    boundary_p_values: tuple
    diagnostics: dict


def _select_components(
    spec,
    samples,
    weights=None,
    /,
    *,
    k_max=None,
    subsample="auto",
    rng=0,
    config=None,
    search_options=None,
    em_options=None,
    options=None,
):
    """Select the component count by a staged shared BIC search.

    Steps: the KDE mode proposal and count ceiling; shared screens of ``K = 1 ..
    initial_hi`` (stopping above the mode proposal at the first count whose
    best screen BIC is worse than its predecessor's,
    ``AUTO_LC_SCREEN_STOP_ABOVE_MODES``) at every screening degree with every automatic endpoint log
    enabled (``K = 1`` a single fit, ``K > 1`` a short unaccelerated EM from
    each initializer's sharpened responsibilities, no polish); the BIC
    ``-2 n_eff LL + dim log n_eff`` on the face dimension, rejecting a
    component below ``AUTO_LC_MIN_COMPONENT_N`` effective rows and an
    interval fit saturating the censored likelihood with excess parameters;
    admission of each degree's winner and neighbors, refinement under the
    request's actual policies (``_fit_mixture_auto`` from the best screen's
    responsibilities) and opening the refined winner's edge until closed;
    purpose-indexed random streams; optional stratified subsampling with the
    mean likelihood extrapolated to the full effective size.  The search is
    heuristic, not globally optimal.  With a subsample the winner is
    refitted on all rows from its screen's responsibilities.

    Parameters
    ----------
    spec : BasisSpec
        Basis request of every component (``"auto"`` fields allowed).
    samples, weights : array_like
        Observations and observation weights.
    k_max : int or None
        Largest count considered (``AUTO_K_MAX`` when ``None``).
    subsample : str, int or bool
        Selection-sample policy.
    rng : int or numpy.random.Generator
        Randomness of the proposal, initializers and refinements.

    Returns
    -------
    _ComponentSelection

    Raises
    ------
    _ComponentSelectionError
        If every refinement failed.
    """

    config = _DegreeSelectionConfig() if config is None else config
    options = _NewtonOptions() if options is None else options
    failures_caught = (*NUMERIC_FAILURES, _PointMixtureEstimabilityFailure)
    full = _prepare(spec, samples, weights)
    rows = full.inputs.samples
    intervals = full.inputs.intervals
    support = tuple(map(float, spec.support))
    n_rows = rows.shape[0]
    full_w = full.weights / float(np.sum(full.weights))
    effective_n = float(full.inputs.effective_n)
    lower, upper = full.inputs.log_policy
    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    with _maybe_suppress(True, SUPPRESSED_WARNINGS):
        k_modes, effective_k_max = _propose_n_components(
            full.representatives,
            AUTO_K_MAX if k_max is None else int(k_max),
            generator,
        )
    policy = _prepare_component_selection_policy(
        n_rows,
        support,
        k_modes,
        effective_k_max,
        subsample,
        spec.bulk_degree,
        lower,
        upper,
    )
    ceiling, initial_hi, degrees = policy.ceiling, policy.initial_hi, policy.degrees
    entropy = generator.integers(0, 2**32, size=4, dtype=np.uint32)

    def rng_for(stage, count):
        return np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(stage, count))
        )

    # The full data's endpoint policies apply to the selection sample.
    resolved = dataclasses.replace(spec, log_lower=lower, log_upper=upper)
    sub_idx = None
    sample, sample_rows, sample_w = full, samples, weights
    if policy.subsample_size < n_rows:
        sub_idx = _stratified_subsample(
            full.representatives, policy.subsample_size, rng_for(0, 0)
        )
        picked = rows[sub_idx]
        sample_rows = picked if intervals else picked[:, 0]
        sample_w = full_w[sub_idx] / float(np.sum(full_w[sub_idx]))
        sample = _prepare(resolved, sample_rows, sample_w)
    fit_rows = sample.inputs.samples
    fit_w = sample.weights / float(np.sum(sample.weights))
    sample_effective_n = _effective_n(fit_rows.shape[0], fit_w)
    enabled = tuple(flag is True or flag == _AUTO for flag in (lower, upper))
    screen_expansion = _MINIMAL if spec.expansion == _AUTO_EXPANSION else spec.expansion
    screen_em = _EMOptions(max_steps=AUTO_LC_SWEEP_EM_MAX_ITER, accelerate=False)
    # Screens rank short EM explorations. Their component M-steps use a
    # coarser certified bound; the single-component screens and all admitted
    # refinements retain the requested optimizer policy and shape checks.
    screen_options = dataclasses.replace(
        options,
        tolerance=max(options.tolerance, AUTO_LC_SWEEP_NEWTON_TOL),
        certified_tolerance=max(options.certified_tolerance, AUTO_LC_SWEEP_NEWTON_TOL),
        accuracy_floor=max(options.accuracy_floor, AUTO_LC_SWEEP_NEWTON_TOL),
    )
    log_n = math.log(max(effective_n, 1.0))
    screens, screening_scores, refined, scores = {}, [], {}, []
    admissions = {1: "mandatory_single"}

    def score(fit):
        dimension = int(fit.n_face_parameters)
        likelihood = float(fit.log_likelihood)
        if dimension < 1 or not math.isfinite(likelihood):
            raise _CandidateRejected(
                "candidate has invalid likelihood or face dimension"
            )
        if fit.weights.size > 1:
            if float(np.min(fit.weights)) * sample_effective_n < (
                AUTO_LC_MIN_COMPONENT_N
            ):
                raise _CandidateRejected("degenerate_component")
            if (
                intervals
                and _interval_identifiability_diagnostic(
                    fit_rows,
                    None,
                    likelihood,
                    support,
                    obs_weights=fit_w,
                    n_parameters=dimension,
                )
                is not None
            ):
                raise _CandidateRejected("unidentified_interval_components")
        return -2.0 * likelihood * effective_n + dimension * log_n

    def screen_fit(count, degree, name, resp):
        concrete = dataclasses.replace(
            spec,
            bulk_degree=degree,
            log_lower=enabled[0],
            log_upper=enabled[1],
            expansion=screen_expansion,
        )
        if count == 1:
            single = _fit_basis(
                concrete, sample_rows, sample_w, config=config, options=options
            )
            return _single_mixture_fit(single, fit_rows.shape[0])
        r = resp if sub_idx is None else resp[sub_idx]
        r = np.square(np.asarray(r, dtype=np.float64))
        r /= r.sum(axis=1, keepdims=True)
        start = _merge_responsibilities(r, fit_w, sample.inverse, sample.data.size)
        if not intervals:
            _check_point_estimability(
                sample.data.x[:, None], sample.data.weights, start
            )
        models = _component_models(sample, (concrete,) * count, r)
        problem = _MixtureProblem(models, sample.data, screen_options)
        return _run_em(
            problem, start, em_options=screen_em, polish=False, initialization=name
        )

    def screen(count):
        _check_fit_time("component-count screening")
        if count in screens:
            return
        screens[count] = {}
        candidates = ((None, None, None),)
        if count > 1:
            try:
                with _maybe_suppress(True, SUPPRESSED_WARNINGS):
                    candidates = _initial_responsibility_candidates(
                        full.representatives, count, rng_for(1, count), weights=full_w
                    )
            except NUMERIC_FAILURES as exc:
                _reraise_if_debug(
                    exc, f"K screen initialization K={count}", routine=True
                )
                screens[count] = dict.fromkeys(degrees)
                screening_scores.extend(
                    {
                        "n_components": count,
                        "degree": int(degree),
                        "bic": None,
                        "initializer": None,
                        "status": "initialization_failed",
                        "failures": (f"{type(exc).__name__}: {exc}",),
                    }
                    for degree in degrees
                )
                return
        for degree in degrees:
            best, failures = None, []
            for name, resp, mix_weights in candidates:
                try:
                    fitted = screen_fit(count, degree, name, resp)
                    bic = score(fitted)
                except _CandidateRejected as exc:
                    # Expected control flow, not a numerical fallback.
                    failures.append(f"{type(exc).__name__}: {exc}")
                    continue
                except failures_caught as exc:
                    _reraise_if_debug(
                        exc, f"K screen K={count}, d={degree}", routine=True
                    )
                    failures.append(f"{type(exc).__name__}: {exc}")
                    continue
                if best is None or bic < best["bic"]:
                    best = {
                        "bic": float(bic),
                        "degree": int(degree),
                        "initializer": name,
                        "fit": fitted,
                        "responsibilities": resp,
                        "weights": mix_weights,
                    }
            screens[count][degree] = best
            screening_scores.append(
                {
                    "n_components": count,
                    "degree": int(degree),
                    "bic": None if best is None else best["bic"],
                    "initializer": None if best is None else best["initializer"],
                    "status": "failed" if best is None else "ok",
                    "failures": tuple(failures),
                }
            )

    def screen_bic(count):
        values = [e["bic"] for e in screens[count].values() if e is not None]
        return min(values) if values else math.inf

    screen_stop = None
    for count in range(1, initial_hi + 1):
        screen(count)
        if (
            AUTO_LC_SCREEN_STOP_ABOVE_MODES
            and count > max(k_modes, 1)
            and math.isfinite(screen_bic(count - 1))
            and screen_bic(count) > screen_bic(count - 1)
        ):
            screen_stop = count
            break
    for degree in degrees:
        viable = [
            (entries[degree]["bic"], count)
            for count, entries in screens.items()
            if entries[degree] is not None
        ]
        if viable:
            winner = min(viable)[1]
            for count in (winner - 1, winner, winner + 1):
                if 1 <= count <= ceiling:
                    admissions.setdefault(
                        count, f"screen_degree_{degree}_winner_or_neighbor"
                    )

    def refine(count):
        _check_fit_time("component-count refinement")
        record = {"n_components": count, "admission": admissions[count]}
        try:
            screen(count)
            viable = [e for e in screens[count].values() if e is not None]
            start = min(viable, key=lambda e: e["bic"]) if viable else None
            resp = None if start is None else start["responsibilities"]
            if resp is not None and sub_idx is not None:
                resp = resp[sub_idx]
            fit, steps, p_values = _fit_mixture_auto(
                resolved,
                sample_rows,
                count,
                sample_w,
                rng=rng_for(2, count),
                responsibilities=resp,
                config=config,
                search_options=search_options,
                em_options=em_options,
                options=options,
            )
            bic = score(fit)
        except _CandidateRejected as exc:
            scores.append(
                record
                | {"bic": None, "status": "failed:_CandidateRejected"}
                | {"reason": str(exc)}
            )
            return
        except failures_caught as exc:
            _reraise_if_debug(exc, f"K refinement K={count}", routine=True)
            scores.append(
                record
                | {"bic": None, "status": f"failed:{type(exc).__name__}"}
                | {"reason": str(exc)}
            )
            return
        refined[count] = (bic, fit, steps, p_values, start)
        scores.append(
            record
            | {
                "bic": float(bic),
                "status": "ok",
                "degree": tuple(m.request.bulk_degree for m in fit.models),
                "n_parameters": int(fit.n_face_parameters),
                "mean_log_likelihood": float(fit.log_likelihood),
                "initializer": None if start is None else start["initializer"],
            }
        )

    for count in sorted(admissions):
        refine(count)
    while refined:
        winner = min(refined, key=lambda c: (refined[c][0], c))
        neighbors = [
            c
            for c in (winner - 1, winner + 1)
            if 1 <= c <= ceiling and c not in admissions
        ]
        if not neighbors:
            break
        for count in neighbors:
            admissions[count] = "open_refined_winner_edge"
            refine(count)
    diagnostics = {
        "method": "log_concave_bic",
        "search": "staged_shared_structural_shortlist",
        "pruning": "heuristic_not_globally_certified",
        "kde_modes": int(k_modes),
        "subsample_size": int(policy.subsample_size),
        "full_sample_size": int(n_rows),
        "subsampled": sub_idx is not None,
        # The selected candidate is the returned fit only without a
        # subsample; a subsampled winner is refitted on every row.
        "reuse_selected_fit": sub_idx is None,
        "score_effective_n": effective_n,
        "selection_sample_effective_n": float(sample_effective_n),
        "likelihood_scale": (
            "subsample_mean_extrapolated_to_full_effective_n"
            if sub_idx is not None
            else "full_data_effective_n"
        ),
        "screen_boundary_enabled": enabled,
        "screen_degrees": degrees,
        # The count at which a worse screen BIC above the mode proposal ended
        # the upward sweep (``None``: every count to the ceiling was screened).
        "screen_stopped_at": screen_stop,
        "screen_newton_tolerance": float(screen_options.certified_tolerance),
        "screening_scores": tuple(screening_scores),
        "scores": tuple(scores),
        "admissions": tuple(sorted(admissions.items())),
        "excluded_components": tuple(
            c for c in range(1, ceiling + 1) if c not in admissions
        ),
        "termination": (
            "closed_winner_neighborhood_or_k_limit"
            if refined
            else "all_refinements_failed"
        ),
    }
    if not refined:
        raise _ComponentSelectionError(diagnostics)
    winner = min(refined, key=lambda c: (refined[c][0], c))
    bic, fit, steps, p_values, start = refined[winner]
    diagnostics.update(selected_n_components=winner, selected_bic=float(bic))
    if sub_idx is not None:
        fit, steps, p_values = _fit_mixture_auto(
            resolved,
            samples,
            winner,
            weights,
            rng=rng_for(3, winner),
            responsibilities=None if start is None else start["responsibilities"],
            config=config,
            search_options=search_options,
            em_options=em_options,
            options=options,
        )
    return _ComponentSelection(winner, fit, steps, p_values, diagnostics)
