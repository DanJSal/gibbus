"""Expansion selection for basis fits: the omitted-statistic score test.

A larger model is adopted only when its omitted statistics are resolvable.
The policy is ``_DegreeSelectionConfig`` (``alpha = 0.05``, participation
floor 8, variance inflation cap ``1e6``, ``covariance_rtol = 1e-10``); the
test is the efficient score with the fitted statistics as nuisance
coordinates, the Schur-complement conditional covariance, a conservative
empirical/model variance inflation, the participation reliability guard,
the chi-square calibration (a calibration device, not exact boundary
asymptotics) and lifted warm starts.  The probes are the columns of a larger
span of the same basis; participation of a probe is measured per
eigen-direction of the standardized conditional covariance, on the efficient
(nuisance-residualized) statistic itself, and so is the variance inflation.
Inflating per probe statistic (``S C S``) instead would, with nearly
singular ``C``, move its near-null space and divide the residual by spurious
near-zero eigenvalues (a joint score of 1.2e4 for 10 directions of Gaussian
data whose uninflated score is 8.9).  Per direction the guards are
invariant to the probe representation.  A direction is resolvable only when
its conditional variance exceeds ``covariance_rtol`` of the unconditional
variance of a standardized probe (a cut relative to the largest conditional
eigenvalue would be meaningless when every new direction is nearly
redundant; the normalizer's ``1e-12`` moment accuracy leaves a hundredfold
margin).

**Policy.**  ``expansion="auto"`` (default) fits the minimal span,
then repeats: a joint score test of every omitted full-span direction at
``alpha``; only if it rejects, the one-slot extension (``_grow_basis``) with
the smallest p-value is fitted from the lifted current fit and kept only if
its likelihood ratio also rejects at ``alpha`` (a score pointing out
of the shape cone at an active curvature row gains nothing).  The joint gate
holds the probability of a first spurious addition at ``alpha``.  A selected
extension that does not converge fails loudly.

**Interval data.**  The score is ``sum w E[Phi | B] - E[Phi]`` and the
information the missing-information identity ``Cov[Phi] - sum w
Cov(Phi | B)``; there is no variance inflation or participation guard.  ``expansion="full"`` fits the
full span and fails loudly when that fit fails or does not converge; the
error reports the span the score test supports.  (Full-span directions with
conditional information ~1e-9 of their variance are numerically resolvable
but statistically undetermined at ordinary sample sizes; their MLE sits at
|theta| ~ 1e3-1e5 and is fitted noise.)
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .._defaults import (
    AUTO_POLY_DEGREE_MAX,
    AUTO_POLY_DEGREE_MIN,
    BOUNDARY_ALPHA,
    _reraise_if_debug,
)
from .._model.basis import (
    _build_basis,
    _full_basis,
    _grow_basis,
    _StructurallyImproper,
)
from .._model.basis_eval import _evaluate_basis
from .._model.basis_normalizer import (
    _basis_state,
    _interval_state,
    _NormalizerContext,
    _QuadratureFailure,
)
from .._model.basis_spec import _AUTO, _AUTO_EXPANSION, _FULL, _MINIMAL
from .._model.coords import (
    _build_fit_coordinate,
    _build_interval_fit_coordinate,
)
from .basis_newton import _Blocks, _matching_face, _solve_basis
from .basis_objective import (
    _BasisIntervalObjective,
    _BasisPointObjective,
    _interval_rows,
    _point_statistics,
    _uniform_statistics,
)
from .boundary import _select_boundary_terms
from .budget import _check_fit_time
from .conic_newton import _CONVERGED, _UNVERIFIED, _NewtonOptions
from .degree import _DegreeSelectionConfig, _project_psd, _psd_pinv
from .inputs import (
    _admissible_degrees,
    _is_poly_degree_admissible,
    _normalize_basis_inputs,
)

_LIFT_RTOL = 1e-12


@dataclass(frozen=True)
class _SpanDiagnostic:
    """Score test of a larger span at the current fit.

    Parameters
    ----------
    label : str
        ``"joint"`` or the added slot name.
    added : int
        Number of additional coefficients of the larger span.
    resolvable : int
        Directions with conditional variance above ``covariance_rtol`` of
        their unconditional variance (before the reliability mask).
    reliable_mask : numpy.ndarray
        Resolvable directions with adequate participation.
    participation : numpy.ndarray
        Participation effective sample size per resolvable direction.
    variance_inflation : numpy.ndarray
        Empirical/model variance inflation per resolvable direction.
    score : float
        Rank-aware quadratic score on reliable probes.
    rank : int
        Rank of the reliable test covariance.
    p_value : float
        Chi-square reference upper-tail probability.
    should_expand : bool
        ``p_value < alpha``.
    """

    label: str
    added: int
    resolvable: int
    reliable_mask: np.ndarray
    participation: np.ndarray
    variance_inflation: np.ndarray
    score: float
    rank: int
    p_value: float
    should_expand: bool


@dataclass(frozen=True)
class _BasisFit:
    """A fitted point basis and how it was chosen.

    Parameters
    ----------
    model : _BasisModel
        Fitted basis (selected degree, log terms and span).
    result : _BasisNewtonResult
        Fit on that basis.
    history : tuple of _SelectionStep
        Degree steps, then expansion rounds.
    boundary_p_values : tuple of float
        One-sided likelihood-ratio p-values per side (``nan`` when the
        side was not decided automatically or needed no test).
    """

    model: object
    result: object
    history: tuple
    boundary_p_values: tuple = (math.nan, math.nan)


class _BasisConvergenceError(RuntimeError):
    """An explicit fit whose solver did not establish convergence.

    ``diagnostics`` records the last attempted model, optimization counters,
    predicted-decrease bound and numerical shape verdict.  The bound is a
    *certificate target*, not a measured likelihood error.  A nonnegative
    curvature verdict does not by itself establish an optimal fit.
    """

    def __init__(self, fit):
        result = fit.result
        degree = int(fit.model.request.bulk_degree)
        parts = getattr(result.final_separation, "parts", ())
        diagnostics = {
            "bulk_degree": degree,
            "status": str(result.status),
            "objective_value": float(result.objective_value),
            "final_decrease_bound": float(result.final_decrease_bound),
            "curvature_exchanges": int(result.exchanges),
            "newton_iterations": int(result.newton_iterations),
            "objective_evaluations": int(result.objective_evaluations),
            "subproblem_iterations": int(result.subproblem_iterations),
            "shape_status": tuple(part.status for part in parts),
        }
        self.diagnostics = diagnostics
        super().__init__(
            f"the fit at bulk_degree={degree} did not converge "
            f"({result.status}); no converged fit of the requested family; "
            f"exchanges={diagnostics['curvature_exchanges']}, "
            f"Newton iterations={diagnostics['newton_iterations']}, "
            f"predicted-decrease bound={diagnostics['final_decrease_bound']:.6g}, "
            f"shape={diagnostics['shape_status']}"
        )


def _function_key(fn, /):
    """Identity of a basis function independent of its index.

    Parameters
    ----------
    fn : _BasisFunction
        A compiled basis function.
    """
    return (fn.kind, fn.side, fn.degree, fn.power, fn.rate, fn.tail_sides)


def _lift_matrix(current, larger, /):
    """``L`` with ``M_larger L = M_current`` (the smaller span inside the larger).

    Slots are matched by function identity, so bases of different bulk
    degree (different function lists) compare in the larger slot space; the
    geometry must be the same, since the gates and centers define the
    functions.

    Parameters
    ----------
    current : _BasisModel
        Smaller span.
    larger : _BasisModel
        Larger span in the same geometry.

    Raises
    ------
    RuntimeError
        If the geometries differ or the current span is not contained in the
        larger one.
    """
    if repr(current.geometry) != repr(larger.geometry):
        raise RuntimeError("internal error: spans with different geometry")
    where = {_function_key(fn): fn.index for fn in larger.functions}
    m_c = np.zeros((larger.coefficient_map.shape[0], current.n_coefficients))
    for fn in current.functions:
        rows = current.coefficient_map[3 * fn.index : 3 * fn.index + 3]
        if not np.any(rows):
            continue
        j = where.get(_function_key(fn))
        if j is None:
            raise RuntimeError(
                "internal error: span is not contained in the larger span"
            )
        m_c[3 * j : 3 * j + 3] = rows
    m_l = larger.coefficient_map
    lift = np.linalg.lstsq(m_l, m_c, rcond=None)[0]
    residual = np.max(np.abs(m_l @ lift - m_c), initial=0.0)
    if residual > _LIFT_RTOL * max(1.0, float(np.max(np.abs(m_c), initial=0.0))):
        raise RuntimeError("internal error: span is not contained in the larger span")
    return lift


class _PointData:
    """Point observations of a fit (normalized weights or ``None``).

    Parameters
    ----------
    x : numpy.ndarray
        Observations, shape ``(R,)``.
    weights : numpy.ndarray or None
        Weights.
    effective_n : float
        Kish effective sample size.
    """

    # The point policy: empirical variance inflation and participation.
    guards = True

    def __init__(self, x, weights, effective_n, /):
        """Bind point data (see class docstring)."""
        self.x = np.asarray(x, dtype=np.float64)
        self.weights = weights
        self.effective_n = float(effective_n)
        # Pooled coordinate per support: the fallback of component coordinates.
        self._pooled = {}

    def build(self, spec, /):
        """Compile ``spec`` in the coordinate and geometry of these data.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        """
        return _build_basis(spec, self.x, self.weights)

    def build_weighted(self, spec, responsibilities, /):
        """Compile ``spec`` in the coordinate of the data weighted by ``r``.

        A mixture component's own coordinate and geometry (each component
        is fitted in its responsibility-weighted coordinate).  A component
        whose weighted landmarks do not identify a coordinate (most of its
        weight on one tied value) uses the pooled data's coordinate.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        responsibilities : numpy.ndarray
            Component responsibilities ``r``, shape ``(R,)``.
        """
        w = np.full(self.x.size, 1.0) if self.weights is None else self.weights
        cw = w * np.asarray(responsibilities, dtype=np.float64)
        if np.all(cw == cw[0]):
            # Equal weights: the unweighted coordinate, exactly.
            return _build_basis(spec, self.x, None)
        support = tuple(map(float, spec.support))
        if support not in self._pooled:
            self._pooled[support] = _build_fit_coordinate(support, self.x, self.weights)
        coordinate = _build_fit_coordinate(
            support, self.x, cw / cw.sum(), fallback=self._pooled[support]
        )
        return _build_basis(spec, self.x, cw / cw.sum(), coordinate)

    def objective(self, model, /):
        """The point objective of ``model``.

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        """
        return _BasisPointObjective(
            _NormalizerContext(model), _point_statistics(model, self.x, self.weights)
        )

    def start(self, model, options, /):
        """Cold fits start from the convex cold start (returns ``None``).

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        options : _NewtonOptions
            Newton policy.
        """
        del model, options

    def score_statistics(self, context, theta, /):
        """``(state, information, row_statistics, row_weights)`` at ``theta``.

        For points the information is ``Cov[Phi]`` and each row's statistic
        ``Phi(x_r)``.

        Parameters
        ----------
        context : _NormalizerContext
            Normalizer of the larger span.
        theta : numpy.ndarray
            Coefficients in that span.
        """
        state = _basis_state(context, theta)
        if state.status != "proper":
            raise RuntimeError("internal error: a lifted proper fit is not proper")
        w = np.ones_like(self.x) if self.weights is None else self.weights
        keep = w > 0.0
        phi = _evaluate_basis(context.model, self.x[keep], max_order=0).phi
        return state, state.cov, phi, w[keep] / w[keep].sum()


class _IntervalData:
    """Interval rows of a fit and their canonical coordinate.

    The coordinate comes from the quantiles of the censored empirical
    distribution of the rows (``_build_interval_fit_coordinate``).  The
    representative points (midpoints of finite rows, cutpoints of one-sided
    rows; whole-support rows carry none) serve the runtime coordinates.

    Parameters
    ----------
    support : tuple of float
        Closed user support.
    samples : numpy.ndarray
        Ordered rows, shape ``(R, 2)``.
    weights : numpy.ndarray or None
        Normalized weights.
    effective_n : float
        Kish effective sample size.
    """

    # The interval policy: the
    # observed information alone, no variance inflation or participation
    # guard (rows are merged, so participation would count distinct rows,
    # not observations: binned data have only their bins).
    guards = False

    def __init__(self, support, samples, weights, effective_n, /):
        """Bind interval data (see class docstring)."""
        x = np.asarray(samples, dtype=np.float64)
        w = np.ones(x.shape[0]) if weights is None else np.asarray(weights)
        self.rows = _interval_rows(support, x, weights)
        self.effective_n = float(effective_n)
        self._support, self._x, self._w = support, x, w
        self.coordinate, self.points, self.point_weights = self._geometry(
            w, weights, None
        )

    def _geometry(self, w, weights, fallback, /):
        """``(coordinate, points, point_weights)`` of the rows weighted by ``w``.

        ``weights`` is what the coordinate builders see (``None``: unweighted).

        Parameters
        ----------
        w : numpy.ndarray
            Row weights of the coordinate, shape ``(R,)``.
        weights : numpy.ndarray or None
            Weights passed to the coordinate builders (``None``: unweighted).
        fallback : _FitCoordinate or None
            Coordinate used when the weighted rows do not identify one
            (``None``: raise).
        """
        support, x = self._support, self._x
        lo, hi = x[:, 0], x[:, 1]
        finite = np.isfinite(lo) & np.isfinite(hi)
        coordinate = _build_interval_fit_coordinate(
            support, x, weights, fallback=fallback
        )
        if np.all(finite):
            points, pw = 0.5 * lo + 0.5 * hi, w
        else:
            one_sided = np.isfinite(lo) ^ np.isfinite(hi)
            cut = np.where(np.isfinite(lo), lo, hi)
            keep = finite | one_sided
            points = np.where(finite, 0.5 * lo + 0.5 * hi, cut)[keep]
            pw = w[keep]
        return coordinate, points, pw / pw.sum()

    def build_weighted(self, spec, responsibilities, /):
        """Compile ``spec`` in the coordinate of the rows weighted by ``r``.

        A component whose weighted rows do not identify a coordinate (most
        of its weight on one censoring cutpoint) uses the pooled rows'
        coordinate.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        responsibilities : numpy.ndarray
            Component responsibilities ``r``, shape ``(R,)``.
        """
        cw = self._w * np.asarray(responsibilities, dtype=np.float64)
        uniform = np.all(cw == cw[0])
        coordinate, points, pw = self._geometry(
            cw, None if uniform else cw / cw.sum(), self.coordinate
        )
        return _build_basis(spec, points, pw, coordinate)

    def build(self, spec, /):
        """Compile ``spec`` in the interval coordinate and landmark geometry.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        """
        return _build_basis(spec, self.points, self.point_weights, self.coordinate)

    def objective(self, model, /):
        """The interval objective of ``model``.

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        """
        return _BasisIntervalObjective(_NormalizerContext(model), self.rows)

    def start(self, model, options, /):
        """Interval start: the point fit to uniform-within-row statistics.

        Only with all rows finite; otherwise ``None``, the convex cold start
        (the generic interior start).
        Returns ``None`` or ``(theta, face)``.

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        options : _NewtonOptions
            Newton policy.
        """
        if self.rows.has_infinite_rows:
            return None
        pseudo = _BasisPointObjective(
            _NormalizerContext(model), _uniform_statistics(model, self.rows)
        )
        fit = _solve_basis(pseudo, options=options)
        if fit.status not in _CONVERGED:
            return None
        return fit.params, _face_of(model, fit)

    def score_statistics(self, context, theta, /):
        """``(state, information, row_statistics, row_weights)`` at ``theta``.

        The information is the missing-information identity ``Cov[Phi] -
        sum w Cov(Phi | B)`` (broad and censored rows inform less) and each
        row's statistic ``E[Phi | B_r]``.

        Parameters
        ----------
        context : _NormalizerContext
            Normalizer of the larger span.
        theta : numpy.ndarray
            Coefficients in that span.
        """
        rows = self.rows
        reduced = _interval_state(
            context, theta, rows.lower, rows.upper, rows.weights, None, True
        )
        state = reduced.state
        information = state.cov - reduced.cov
        return (
            state,
            0.5 * (information + information.T),
            reduced.row_mean,
            rows.weights,
        )


class _UntestableSpan(Exception):
    """The score statistics of a larger span cannot be computed at the fit.

    Parameters
    ----------
    reason : str
        The normalizer's failure.
    """


def _tested(diagnostic, /, *args, **kwargs):
    """Run a span diagnostic; an uncomputable one raises ``_UntestableSpan``.

    A larger span can hold columns without finite moments under the fitted
    density (an exponential of rate 2 against a fitted rate-1 tail, after the
    fit's own leader collapsed): its score test is undefined, and an
    automatic search stops at the fitted span, recorded.

    Parameters
    ----------
    diagnostic : callable
        ``_span_diagnostic`` or ``basis_mixture._mixture_span_diagnostic``.
    *args, **kwargs
        Its arguments.

    Raises
    ------
    _UntestableSpan
        If the normalizer cannot resolve the lifted state.
    """
    try:
        return diagnostic(*args, **kwargs)
    except _QuadratureFailure as exc:
        # Expected on healthy fits (above): the search records it and stops.
        _reraise_if_debug(exc, "score statistics of a larger span", routine=True)
        raise _UntestableSpan(str(exc)) from exc


def _span_diagnostic(current, theta, larger, data, config, label, /, face=None):
    """Score test of ``larger`` against the fitted ``current`` span.

    Parameters
    ----------
    current : _BasisModel
        Fitted span.
    theta : numpy.ndarray
        Fitted coefficients of ``current``.
    larger : _BasisModel
        Span containing ``current``.
    data : _PointData or _IntervalData
        Observations (interval information by the missing-information
        identity).
    config : _DegreeSelectionConfig
        Selection policy.
    label : str
        Diagnostic label.
    face : tuple or None
        ``(suppressed, zero_log_labels)`` of the fit: its suppressed
        functionals are exact zeros, so the lifted state is evaluated with the
        matching face-restricted tail atoms (a lifted zero leader is a
        rounding-level cancellation otherwise).

    Returns
    -------
    _SpanDiagnostic
    """
    lift = _lift_matrix(current, larger)
    context = _NormalizerContext(larger)
    if face is not None and (face[0] or face[1]):
        basis = _matching_face(_Blocks.single(larger), *face).basis(
            larger.n_coefficients
        )
        context = context.on_face(basis)
    state, sigma, phi, w = data.score_statistics(context, lift @ theta)
    return _score_test(
        w @ phi - state.mean,
        sigma,
        phi,
        w,
        lift,
        None,
        data.effective_n,
        data.guards,
        config,
        label,
        larger.n_coefficients - current.n_coefficients,
    )


def _score_test(
    residual, sigma, phi, w, nuisance, probe, effective_n, guards, config, label, added
):
    """Efficient score test of probe statistics given nuisance statistics.

    The statistics are coordinates of one space with score ``residual``,
    information ``sigma`` and per-row values ``phi`` (weights ``w``); the
    probes are the columns of ``probe`` (``None``: every coordinate) and the
    nuisance the columns of ``nuisance`` (the fitted statistics).  A
    single fit's statistics are the larger span's columns with the lifted
    current span as nuisance; a mixture's are its joint coordinates with
    every other coordinate and the probed component's lifted span as
    nuisance (``basis_mixture._mixture_span_diagnostic``).

    Parameters
    ----------
    residual : numpy.ndarray
        Score of the statistics, shape ``(m,)``.
    sigma : numpy.ndarray
        Information of the statistics, shape ``(m, m)``.
    phi : numpy.ndarray
        Per-row statistic values, shape ``(R, m)``.
    w : numpy.ndarray
        Row weights, shape ``(R,)``.
    nuisance : numpy.ndarray
        Nuisance directions (columns), shape ``(m, q)``.
    probe : numpy.ndarray or None
        Probe directions (columns); ``None`` probes every coordinate.
    effective_n : float
        Kish effective sample size.
    guards : bool
        Apply the point-data participation and variance-inflation guards.
    config : _DegreeSelectionConfig
        Selection policy.
    label : str
        Name of the tested extension.
    added : tuple
        Labels of the probed slots.

    Returns
    -------
    _SpanDiagnostic
    """
    sigma = 0.5 * (sigma + sigma.T)
    lift = nuisance
    if probe is not None:
        # Probe coordinates only: sigma_pp, the cross block and the probe rows.
        residual_p = probe.T @ residual
        sigma_pp = probe.T @ sigma @ probe
        sigma_pt = probe.T @ (sigma @ lift)
        phi_p = phi @ probe
    else:
        residual_p, sigma_pp, sigma_pt, phi_p = residual, sigma, sigma @ lift, phi
    sd = np.sqrt(np.maximum(np.diag(sigma_pp), 0.0))
    inv_sd = np.divide(1.0, sd, out=np.zeros_like(sd), where=sd > 0.0)

    # Efficient residual and conditional covariance, fitted statistics
    # (t = L^T Phi) as nuisance coordinates.
    sigma_tt = lift.T @ sigma @ lift
    sigma_ft = inv_sd[:, None] * sigma_pt
    tt_pinv, _ = _psd_pinv(sigma_tt, config.covariance_rtol)
    projection = sigma_ft @ tt_pinv
    efficient = inv_sd * residual_p - projection @ (lift.T @ residual)
    standardized = inv_sd[:, None] * sigma_pp * inv_sd[None, :]
    conditional = _project_psd(
        standardized - projection @ sigma_ft.T, config.covariance_rtol
    )
    values, vectors = np.linalg.eigh(conditional)
    # Resolvable = conditional variance above covariance_rtol of the (unit)
    # unconditional variance of a standardized probe.
    keep = values > float(config.covariance_rtol)
    resolvable = int(np.count_nonzero(keep))
    values, vectors = values[keep], vectors[:, keep]

    # The guards per tested direction: the efficient statistic of direction
    # k is s_k(x) = u_k . (D Phi(x) - P L^T Phi(x)), with model variance
    # values[k].  Its empirical variance (conservative inflation) and the
    # participation of its centered contributions are measured on the data.
    efficient_stats = (phi_p * inv_sd[None, :] - (phi @ lift) @ projection.T) @ vectors
    centered_stats = efficient_stats - w @ efficient_stats
    emp_var = w @ (centered_stats * centered_stats)
    tiny = np.finfo(np.float64).tiny
    inflation = np.clip(
        np.maximum(1.0, emp_var / np.maximum(values, tiny)),
        1.0,
        float(config.max_variance_inflation),
    )
    contribution = w[:, None] * np.abs(centered_stats)
    total = contribution.sum(axis=0)
    square = np.sum(contribution * contribution, axis=0)
    participation = np.divide(
        total * total, square, out=np.zeros_like(total), where=square > 0.0
    )
    participation = np.minimum(participation, float(w.size))
    reliable = participation >= float(config.min_participation)
    if not guards:
        inflation = np.ones_like(inflation)
        reliable = np.ones_like(reliable)

    score, rank, p_value = 0.0, int(np.count_nonzero(reliable)), 1.0
    if rank:
        coords = vectors[:, reliable].T @ efficient
        score = float(
            effective_n * np.sum(coords**2 / (values[reliable] * inflation[reliable]))
        )
        p_value = float(chi2.sf(max(score, 0.0), rank))
    return _SpanDiagnostic(
        label=label,
        added=int(added),
        resolvable=resolvable,
        reliable_mask=reliable,
        participation=participation,
        variance_inflation=inflation,
        score=score,
        rank=rank,
        p_value=p_value,
        should_expand=bool(rank > 0 and p_value < float(config.alpha)),
    )


@dataclass(frozen=True)
class _SelectionStep:
    """One automatic decision of a fit.

    Parameters
    ----------
    kind : str
        ``"degree"`` or ``"expansion"``.
    diagnostic : _SpanDiagnostic or None
        Score test that decided the step (the joint gate for an expansion);
        ``None`` when the test could not be computed (``failure`` says why).
    candidates : tuple of _SpanDiagnostic
        Per-slot tests (expansion rounds whose gate rejected), else empty.
    chosen : int, str or None
        The next bulk degree or the added slot; ``None`` when selection
        stopped at this step.
    lr_p_value : float
        Expansion steps that fitted ``chosen``: the likelihood-ratio
        confirmation's p-value (``nan`` otherwise).
    failure : str
        Why the selected degree was not adopted (it converged from neither
        start), else empty.
    """

    kind: str
    diagnostic: object
    candidates: tuple
    chosen: object
    lr_p_value: float = math.nan
    failure: str = ""


@dataclass(frozen=True)
class _Stage:
    model: object
    result: object
    history: tuple


def _face_of(model, result, /):
    """``(suppressed_ends, zero_log_labels)`` of a fit, for warm starts.

    Parameters
    ----------
    model : _BasisModel
        Fitted basis.
    result : _BasisNewtonResult
        Its fit.
    """
    return (
        result.face.suppressed,
        tuple(model.columns[i].label for i in result.face.zero_logs),
    )


def _warm_fit(model, data, options, previous, /):
    """Fit ``model``, warm-started from a smaller nested fit when given (a lift).

    A cold fit starts where the data say (``data.start``: the interval
    start, else the convex cold start).

    Parameters
    ----------
    model : _BasisModel
        Basis to fit.
    data : _PointData or _IntervalData
        Observations.
    options : _NewtonOptions
        Newton policy.
    previous : _Stage or None
        Smaller nested fit to lift, or ``None`` for a cold fit.

    Returns
    -------
    _BasisNewtonResult
        The fit (:func:`_fit_rung` compares it with ``previous``).
    """
    objective = data.objective(model)
    if previous is None:
        start = data.start(model, options)
        if start is None:
            return _solve_basis(objective, options=options)
        return _solve_basis(
            objective,
            options=options,
            start=start[0],
            start_face=start[1],
        )
    # A lifted fit has zero amplitudes on the larger basis's new directions;
    # it is moved into the face interior along a verified convex direction
    # (``interior_lift``) rather than replaced by the face's cold start.
    return _solve_basis(
        objective,
        options=options,
        start=_lift_matrix(previous.model, model) @ previous.result.params,
        start_face=_face_of(previous.model, previous.result),
        interior_lift=True,
    )


def _worse_than_nested(result, smaller, options, /):
    """Whether a converged fit is worse than a fit of a family it contains.

    The larger family's optimum is at most the smaller fit's NLL. An excess
    beyond the solver's status tolerance and likelihood roundoff therefore
    signals unresolved optimization, rather than a distinct local minimum.
    The comparison uses ``certified_tolerance`` for ``converged`` and
    ``accuracy_floor`` for ``converged_approximately``; the Newton stopping
    criterion alone is not a global likelihood-error bound.

    Parameters
    ----------
    result : _BasisNewtonResult
        Converged fit of the larger family.
    smaller : _BasisNewtonResult
        Fit of the nested smaller family.
    options : _NewtonOptions
        The fit's tolerances.
    """
    value = float(result.objective_value)
    promise = (
        options.certified_tolerance
        if result.status == "converged"
        else options.accuracy_floor
    )
    roundoff = 8.0 * np.finfo(np.float64).eps * max(1.0, abs(value))
    return value > float(smaller.objective_value) + promise + roundoff


def _bulk_degrees(spec, /):
    """Admissible automatic degrees (2-12; even only on the full line).

    Parameters
    ----------
    spec : BasisSpec
        Basis request (its support).
    """
    degrees = [
        int(d)
        for d in _admissible_degrees(tuple(spec.support), AUTO_POLY_DEGREE_MAX)
        if int(d) >= int(AUTO_POLY_DEGREE_MIN)
    ]
    return degrees or [int(AUTO_POLY_DEGREE_MIN)]


def _fit_rung(model, data, options, nested, /):
    """Fit one basis from its starts: cold, then the lifted ``nested`` fit.

    The cold start comes first: a lifted smaller fit has a zero leading
    amplitude, i.e. it lies on the larger basis's improper boundary, where a
    first Newton step can pair a 1e-14 leader with an O(0.1) next term.  With
    a nested fit, a cold result that does not converge, or converges above
    the nested fit, is retried from the lift (the cold start can stall on a
    badly scaled rung, a cluster 40 scale units out, or stop at a point the
    nested fit beats), and the better converged answer is kept.  A larger
    family cannot converge above a fit it contains, so such an answer is a
    stall and is reported as ``"non_descent"``.

    Parameters
    ----------
    model : _BasisModel
        Basis to fit.
    data : _PointData or _IntervalData
        Observations.
    options : _NewtonOptions
        Newton policy.
    nested : _Stage or None
        Converged fit of a basis nested in ``model``.

    Returns
    -------
    _BasisNewtonResult
    """
    result = _warm_fit(model, data, options, None)
    if nested is None:
        return result
    if result.status not in _CONVERGED or _worse_than_nested(
        result, nested.result, options
    ):
        warm = _warm_fit(model, data, options, nested)
        if warm.status in _CONVERGED and (
            result.status not in _CONVERGED
            or warm.objective_value < result.objective_value
        ):
            result = warm
    return _against_nested(result, nested, options)


def _against_nested(result, nested, options, /):
    """``result``, reported ``"non_descent"`` if it converged above ``nested``.

    A larger family cannot converge above a fit it contains: such an answer
    is a stall (an ill-conditioned rung that kept the lifted fit's zero
    leader).

    Parameters
    ----------
    result : _BasisNewtonResult
        Fit of the larger family.
    nested : _Stage
        Fit of a family it contains.
    options : _NewtonOptions
        The fit's tolerances.
    """
    if result.status in _CONVERGED and _worse_than_nested(
        result, nested.result, options
    ):
        return dataclasses.replace(result, status="non_descent")
    return result


def _rung_model(data, base, degree, /):
    """The degree-``degree`` minimal basis, or ``None`` when it is improper.

    Parameters
    ----------
    data : _PointData or _IntervalData
        Observations.
    base : BasisSpec
        Concrete minimal request.
    degree : int
        Bulk degree.
    """
    try:
        return data.build(dataclasses.replace(base, bulk_degree=degree))
    except _StructurallyImproper:
        # A property of the request at this degree, not a numerical failure:
        # the degree is not admissible.
        return None


_SEED_CLIMB_PATIENCE = 2
"""Consecutive non-converged rungs that end an explicit degree's seed climb."""


def _seed_rungs(spec, base, data, config, options, /):
    """Converged rungs below an explicit degree, the seeds of its lifts.

    The climb fits the admissible degrees below the request upward as rungs
    (:func:`_fit_rung`), each nested on the largest converged one, and ends
    after ``_SEED_CLIMB_PATIENCE`` consecutive rungs that do not converge:
    one failed rung (an odd degree on a half-line) can sit below rungs that
    converge, while a run of failures continues to the target at many
    seconds per rung.  A lift from one seed can fail where another's
    succeeds, so every converged rung is a seed.

    Parameters
    ----------
    spec : BasisSpec
        Request with an explicit ``bulk_degree``.
    base : BasisSpec
        Its concrete minimal request.
    data : _PointData or _IntervalData
        Observations.
    config : _DegreeSelectionConfig
        Selection policy.
    options : _NewtonOptions
        Newton policy.

    Returns
    -------
    list of _Stage
        Converged rungs, ascending in degree.
    """
    support = tuple(spec.support)
    seeds, misses = [], 0
    for degree in range(int(AUTO_POLY_DEGREE_MIN), int(spec.bulk_degree)):
        if not _is_poly_degree_admissible(degree, support):
            continue
        rung = _rung_model(data, base, degree)
        if rung is None:
            continue
        fit = _fit_rung(rung, data, options, seeds[-1] if seeds else None)
        if fit.status in _CONVERGED:
            seeds.append(_Stage(rung, fit, ()))
            misses = 0
            continue
        misses += 1
        if misses >= _SEED_CLIMB_PATIENCE:
            break
    return seeds


def _fit_degree(spec, lower, upper, data, config, options, /):
    """Minimal-span fit at the requested or automatically selected bulk degree.

    Every degree is fitted as a rung (:func:`_fit_rung`) nested on the
    largest converged smaller degree.  The automatic ladder fits each
    admissible degree, probes the next ``probe_block_size`` bulk powers (the
    degree-``d + 2`` basis, nested in the same geometry) with the
    omitted-statistic score test and moves to the next admissible degree only
    when the test rejects; a selected rung that does not converge ends the
    ladder at the smaller degree with the failure recorded (the search's
    answer is the largest converged degree).  An explicit degree is fitted
    cold; if that does not converge, it is lifted from the converged rungs
    below it (:func:`_seed_rungs`), largest first, until a lift converges
    no worse than the largest.
    An explicit degree, or a
    ladder whose first rung does not converge, returns the unconverged fit
    and ``_fit_data`` raises.

    Parameters
    ----------
    spec : BasisSpec
        Basis request (its ``bulk_degree`` explicit or ``"auto"``).
    lower, upper : bool
        Endpoint logs of this fit.
    data : _PointData or _IntervalData
        Observations.
    config : _DegreeSelectionConfig
        Selection policy.
    options : _NewtonOptions
        Newton policy.

    Returns
    -------
    _Stage
    """
    _check_fit_time("degree selection")
    base = dataclasses.replace(
        spec, log_lower=lower, log_upper=upper, expansion=_MINIMAL
    )
    if spec.bulk_degree != _AUTO:
        model = data.build(base)
        result = _warm_fit(model, data, options, None)
        if result.status not in _CONVERGED:
            seeds = _seed_rungs(spec, base, data, config, options)
            # Every lift answers to the best nested fit, the largest converged
            # rung (a degree-3 lift into degree 12 "converged" above the
            # degree-10 fit); lifts are tried from the largest seed down.
            best = seeds[-1] if seeds else None
            for seed in reversed(seeds):
                lifted = _against_nested(
                    _warm_fit(model, data, options, seed), best, options
                )
                if lifted.status in _CONVERGED:
                    result = lifted
                    break
        return _Stage(model, result, ())
    degrees = _bulk_degrees(spec)
    history, current = [], None
    for index, degree in enumerate(degrees):
        model = _rung_model(data, base, degree)
        if model is None:
            continue
        result = _fit_rung(model, data, options, current)
        if result.status not in _CONVERGED:
            if current is None:
                return _Stage(model, result, tuple(history))
            # The search's answer is the largest converged degree; the record
            # says why it stopped.
            last = history.pop()
            history.append(
                dataclasses.replace(
                    last,
                    chosen=None,
                    failure=(
                        f"bulk_degree={degree} converged from neither start "
                        f"({result.status})"
                    ),
                )
            )
            break
        current = _Stage(model, result, ())
        if index == len(degrees) - 1:
            break
        probe_degree = min(degree + int(config.probe_block_size), degrees[-1])
        probe = data.build(dataclasses.replace(base, bulk_degree=probe_degree))
        try:
            diagnostic = _tested(
                _span_diagnostic,
                model,
                result.params,
                probe,
                data,
                config,
                f"bulk_degree={probe_degree}",
                face=_face_of(model, result),
            )
        except _UntestableSpan as exc:
            history.append(
                _SelectionStep(
                    "degree",
                    None,
                    (),
                    None,
                    failure=f"bulk_degree={probe_degree} is untestable ({exc})",
                )
            )
            break
        chosen = degrees[index + 1] if diagnostic.should_expand else None
        history.append(_SelectionStep("degree", diagnostic, (), chosen))
        if chosen is None:
            break
    if current is None:
        raise ValueError("no admissible bulk degree for this basis request")
    return _Stage(current.model, current.result, tuple(history))


def _log_amplitude(stage, side, /):
    """Fitted amplitude of a side's endpoint log term (0 if absent).

    Parameters
    ----------
    stage : _Stage
        A fitted stage.
    side : str
        ``"lower"`` or ``"upper"``.
    """
    model = stage.model
    for i, column in enumerate(model.columns):
        if column.nonnegative and model.functions[column.terms[0][0]].side == side:
            return float(stage.result.params[i])
    return 0.0


def _fit_basis(spec, samples, weights=None, /, *, config=None, options=None):
    """Fit a basis to point or interval data, resolving every ``"auto"`` field.

    Order: the bulk degree ladder at the minimal span with every
    automatic endpoint log enabled; backward elimination of automatic
    log terms (``_fit/boundary.py`` ``_select_boundary_terms``: one-sided
    likelihood-ratio test at the same degree, then the degree ladder rerun
    with the final terms); finally the span by ``spec.expansion``.

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    samples : array_like
        Point observations, shape ``(R,)``, or interval rows, shape
        ``(R, 2)`` (infinite ends are censoring).
    weights : array_like or None
        Observation weights.
    config : _DegreeSelectionConfig or None
        Selection policy.
    options : _NewtonOptions or None
        Newton policy.

    Returns
    -------
    _BasisFit

    Raises
    ------
    RuntimeError
        If a degree or extension selected by the score test does not
        converge, or an explicit full expansion fails or does not converge
        (the message reports the span the score test supports).
    """
    config = _DegreeSelectionConfig() if config is None else config
    options = _NewtonOptions() if options is None else options
    inputs = _normalize_basis_inputs(spec, samples, weights)
    if inputs.intervals:
        data = _IntervalData(
            tuple(map(float, spec.support)),
            inputs.samples,
            inputs.weights,
            inputs.effective_n,
        )
    else:
        data = _PointData(inputs.samples[:, 0], inputs.weights, inputs.effective_n)
    return _fit_data(spec, data, inputs.log_policy, config, options)


def _fit_data(spec, data, log_policy, config, options, /):
    """``_fit_basis`` on validated data (see there), failing loudly.

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    data : _PointData or _IntervalData
        Observations.
    log_policy : tuple
        Resolved endpoint-log policies ``(lower, upper)``.
    config : _DegreeSelectionConfig
        Selection policy.
    options : _NewtonOptions
        Newton policy.

    Returns
    -------
    _BasisFit

    Raises
    ------
    RuntimeError
        If the returned fit's optimizer did not converge (an explicit degree,
        or an automatic search with no converged member).
    """
    fit = _search_data(spec, data, log_policy, config, options)
    status = fit.result.status
    if status not in _CONVERGED and status != _UNVERIFIED:
        raise _BasisConvergenceError(fit)
    return fit


def _search_data(spec, data, log_policy, config, options, /):
    """Degree, endpoint-log and span searches of ``_fit_data`` (see there).

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    data : _PointData or _IntervalData
        Observations.
    log_policy : tuple
        Resolved endpoint-log policies ``(lower, upper)``.
    config : _DegreeSelectionConfig
        Selection policy.
    options : _NewtonOptions
        Newton policy.
    """
    lower, upper = log_policy
    # An endpoint log at a distance >1e8 times the full observed span is
    # numerically indistinguishable from its local polynomial Taylor series.
    # In particular its non-polynomial residual is below float resolution for
    # degree >= 2, while the redundant log amplitude can grow without bound
    # and destroy line-search accuracy.  Exclude only *automatic* log terms;
    # a user-requested term is never silently changed.  Point data only:
    # interval/censoring observations have different identifiability rules.
    unresolved = [False, False]
    if isinstance(data, _PointData):
        xlo, xhi = float(np.min(data.x)), float(np.max(data.x))
        width = xhi - xlo
        for index, (policy, bound, gap) in enumerate(
            (
                (lower, float(spec.support[0]), xlo - float(spec.support[0])),
                (upper, float(spec.support[1]), float(spec.support[1]) - xhi),
            )
        ):
            if policy == _AUTO and math.isfinite(bound) and gap > 0.0:
                unresolved[index] = width / gap < 1.0e-8
        if unresolved[0]:
            lower = False
        if unresolved[1]:
            upper = False

    def fit(lo, up):
        return _fit_degree(spec, lo, up, data, config, options)

    def fit_reduced(stage, lo, up):
        fixed = dataclasses.replace(spec, bulk_degree=stage.model.request.bulk_degree)
        return _fit_degree(fixed, lo, up, data, config, options)

    if _AUTO in (lower, upper):
        stage, _, p_values = _select_boundary_terms(
            fit,
            lambda st: float(st.result.objective_value),
            _log_amplitude,
            lower,
            upper,
            data.effective_n,
            fit_reduced=fit_reduced,
            refit=fit if spec.bulk_degree == _AUTO else None,
            alpha=BOUNDARY_ALPHA,
        )
    else:
        stage, p_values = fit(lower, upper), (math.nan, math.nan)
    if any(unresolved):
        p_values = tuple(
            1.0 if dropped else p
            for dropped, p in zip(unresolved, p_values, strict=True)
        )
    model, result = stage.model, stage.result
    history = list(stage.history)
    concrete = dataclasses.replace(
        spec,
        bulk_degree=model.request.bulk_degree,
        log_lower=model.request.log_lower,
        log_upper=model.request.log_upper,
    )
    if spec.expansion == _FULL:
        full = data.build(concrete)
        try:
            result = _warm_fit(full, data, options, None)
        except RuntimeError as exc:
            _raise_full_failure(concrete, data, config, options, full, str(exc), exc)
        if result.status not in _CONVERGED:
            _raise_full_failure(
                concrete, data, config, options, full, result.status, None
            )
        return _BasisFit(full, result, tuple(history), p_values)
    if spec.expansion == _MINIMAL:
        return _BasisFit(model, result, tuple(history), p_values)
    if spec.expansion != _AUTO_EXPANSION:
        raise ValueError("unknown expansion policy")
    while result.status in _CONVERGED:
        full = _full_basis(model)
        if full.n_coefficients == model.n_coefficients:
            break
        face = _face_of(model, result)
        try:
            gate = _tested(
                _span_diagnostic,
                model,
                result.params,
                full,
                data,
                config,
                "joint",
                face=face,
            )
            if not gate.should_expand:
                history.append(_SelectionStep("expansion", gate, (), None))
                break
            candidates = [
                (
                    larger,
                    _tested(
                        _span_diagnostic,
                        model,
                        result.params,
                        larger,
                        data,
                        config,
                        label,
                        face=face,
                    ),
                )
                for label, larger in _grow_basis(model)
            ]
        except _UntestableSpan as exc:
            history.append(
                _SelectionStep(
                    "expansion",
                    None,
                    (),
                    None,
                    failure=f"the larger span is untestable ({exc})",
                )
            )
            break
        tests = tuple(c[1] for c in candidates)
        ranked = [c for c in candidates if c[1].rank > 0]
        if not ranked:
            history.append(_SelectionStep("expansion", gate, tests, None))
            break
        larger, chosen = min(ranked, key=lambda c: c[1].p_value)
        history.append(_SelectionStep("expansion", gate, tests, chosen.label))
        nested = _Stage(model, result, ())
        grown = _against_nested(
            _warm_fit(larger, data, options, nested), nested, options
        )
        if grown.status not in _CONVERGED:
            # The search's answer is the largest converged span.
            history[-1] = dataclasses.replace(
                history[-1],
                chosen=None,
                failure=(
                    f"the score-selected extension {chosen.label} "
                    f"(p = {chosen.p_value:.3g}) did not converge ({grown.status})"
                ),
            )
            break
        # Likelihood-ratio confirmation: at a fit on an active
        # curvature row the score can point out of the shape cone, where the
        # larger span cannot follow (a censored exponential: score p = 0.006,
        # no change in the NLL).  The chi-square reference with the score's
        # rank is conservative for the constrained (chi-bar-square) ratio.
        lr = 2.0 * data.effective_n * (result.objective_value - grown.objective_value)
        lr_p = float(chi2.sf(max(lr, 0.0), chosen.rank))
        history[-1] = dataclasses.replace(history[-1], lr_p_value=lr_p)
        if not lr_p < float(config.alpha):
            history[-1] = dataclasses.replace(history[-1], chosen=None)
            break
        model, result = larger, grown
    return _BasisFit(model, result, tuple(history), p_values)


def _raise_full_failure(spec, data, config, options, full, reason, cause, /):
    """Fail loudly for an explicit full expansion that could not be fitted.

    The message reports what the omitted-statistic score test supports (the
    automatic selection), so no separate identifiability threshold is needed:
    full-span directions the data do not determine show up as directions the
    test does not add.

    Parameters
    ----------
    spec : BasisSpec
        The concrete request whose full span failed.
    data : _PointData or _IntervalData
        Observations.
    config : _DegreeSelectionConfig
        Selection policy.
    options : _NewtonOptions
        Newton policy.
    full : _BasisModel
        The full span.
    reason : str
        Status or failure message.
    cause : BaseException or None
        Original failure, chained.
    """
    try:
        auto = _search_data(
            dataclasses.replace(spec, expansion=_AUTO_EXPANSION),
            data,
            (spec.log_lower, spec.log_upper),
            config,
            options,
        )
    except RuntimeError as exc:
        # Diagnostic only: the full-expansion failure below is raised either
        # way; the automatic search merely names the reason in its message.
        _reraise_if_debug(exc, "automatic search for a full-expansion failure")
        auto = None
    if auto is None or auto.result.status not in _CONVERGED:
        raise RuntimeError(
            f"expansion='full' could not be fitted on these data ({reason})"
        ) from cause
    extra = full.n_coefficients - auto.model.n_coefficients
    raise RuntimeError(
        f"expansion='full' could not be fitted on these data ({reason}); the "
        f"omitted-statistic score test supports {auto.model.n_coefficients} of "
        f"its {full.n_coefficients} coefficients and leaves {extra} undetermined "
        "by the data; use expansion='auto' (the default) or 'minimal'"
    ) from cause
