"""Expansion selection for G3 point fits: G1's omitted-statistic score test.

G1 owner adapted: ``_fit/degree.py`` (``_omitted_statistic_diagnostic``, the
degree ladder's "grow only when the omitted statistics are resolvable" rule).
Retained from G1 unchanged: ``_DegreeSelectionConfig`` (``alpha = 0.05``,
participation floor 8, variance inflation cap ``1e6``, ``covariance_rtol =
1e-10``), the efficient score with the fitted statistics as nuisance
coordinates, the Schur-complement conditional covariance, the conservative
empirical/model variance inflation, the participation reliability guard, the
chi-square calibration (a calibration device, not exact boundary asymptotics)
and lifted warm starts.  Adapted: the probes are the columns of a larger span
of the same basis instead of the next power moments; participation of a probe
is measured per eigen-direction of the standardized conditional covariance,
on the efficient (nuisance-residualized) statistic itself, and so is the
variance inflation.  G1 inflates per probe statistic (``S C S``); with
nearly singular ``C`` that moves its near-null space and divides the residual
by spurious near-zero eigenvalues (a joint score of 1.2e4 for 10 directions
of Gaussian data whose uninflated score is 8.9).  Per direction the guards
are invariant to the probe representation.  A direction is resolvable only
when its conditional variance exceeds ``covariance_rtol`` of the
unconditional variance of a standardized probe (G1 cuts relative to the
largest conditional eigenvalue, which is meaningless when every new direction
is nearly redundant; the normalizer's ``1e-12`` moment accuracy leaves a
hundredfold margin).

**Policy (G3-035).**  ``expansion="auto"`` (default) fits the minimal span,
then repeats: a joint score test of every omitted full-span direction at
``alpha``; only if it rejects, the one-slot extension (``_grow_basis``) with
the smallest p-value is fitted from the lifted current fit and kept only if
its likelihood ratio also rejects at ``alpha`` (G3-044: a score pointing out
of the shape cone at an active curvature row gains nothing).  The joint gate
holds the probability of a first spurious addition at ``alpha``.  A selected
extension that does not converge fails loudly.

**Interval data (G3-044).**  The score is ``sum w E[Phi | B] - E[Phi]`` and
the information G1's missing-information identity ``Cov[Phi] - sum w
Cov(Phi | B)``; as G1's ``_interval_omitted_statistic_diagnostic`` there is
no variance inflation or participation guard.  ``expansion="full"`` fits the
full span and fails loudly when that fit fails or does not converge; the
error reports the span the score test supports.  (Full-span directions with
conditional information ~1e-9 of their variance are numerically resolvable
but statistically undetermined at ordinary sample sizes; their MLE sits at
|theta| ~ 1e3-1e5 and is fitted noise, G3-035.)
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
from .._model.basis import _build_basis, _full_basis, _grow_basis
from .._model.basis_eval import _evaluate_basis
from .._model.basis_normalizer import (
    _basis_state,
    _interval_state,
    _NormalizerContext,
    _QuadratureFailure,
)
from .._model.basis_spec import _AUTO, _AUTO_EXPANSION, _FULL, _MINIMAL
from .._model.coords import _build_fit_coordinate, _build_interval_fit_coordinate
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
from .conic_newton import _CONVERGED, _UNCERTIFIED, _NewtonOptions
from .degree import _DegreeSelectionConfig, _project_psd, _psd_pinv
from .inputs import _admissible_degrees, _normalize_basis_inputs

_LIFT_RTOL = 1e-12


@dataclass(frozen=True)
class _SpanDiagnostic:
    """Score test of a larger span at the current fit (G1 ``_DegreeDiagnostic`` role).

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
    """A fitted G3 point basis and how it was chosen.

    Parameters
    ----------
    model : _BasisModel
        Fitted basis (selected degree, log terms and span).
    result : _BasisNewtonResult
        Fit on that basis.
    history : tuple of _SelectionStep
        Degree steps, then expansion rounds.
    boundary_p_values : tuple of float
        G1 one-sided likelihood-ratio p-values per side (``nan`` when the
        side was not decided automatically or needed no test).
    """

    model: object
    result: object
    history: tuple
    boundary_p_values: tuple = (math.nan, math.nan)


class _BasisConvergenceError(RuntimeError):
    """An explicit G3 fit whose solver did not establish convergence.

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
            f"the G3 fit at bulk_degree={degree} did not converge "
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
    return (fn.kind, fn.side, fn.degree, fn.power, fn.delta, fn.rate, fn.location)


def _lift_matrix(current, larger, /):
    """``L`` with ``M_larger L = M_current`` (the smaller span inside the larger).

    Slots are matched by function identity, so bases of different bulk
    degree (different function lists) compare in the larger slot space; the
    geometry must be the same, since the gates and centres define the
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

    # G1's point policy: empirical variance inflation and participation.
    guards = True

    def __init__(self, x, weights, effective_n, /):
        """Bind point data (see class docstring)."""
        self.x = np.asarray(x, dtype=np.float64)
        self.weights = weights
        self.effective_n = float(effective_n)

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

        A mixture component's own coordinate and geometry (G1: each
        component is fitted in its responsibility-weighted coordinate).

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
            return _build_basis(spec, self.x, None, adaptive_geometry=False)
        return _build_basis(spec, self.x, cw / cw.sum(), adaptive_geometry=False)

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
            G1 Newton policy.
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
    """Interval rows of a fit, with G1's interval fitting coordinate.

    The coordinate is G1's: midpoints and widths when every row is finite
    (``_build_fit_coordinate``), else the censoring-landmark coordinate
    (``_build_interval_fit_coordinate``); the basis geometry is placed by
    the same landmarks (midpoints of finite rows, cutpoints of one-sided
    rows; whole-support rows carry none).

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

    # G1's interval policy (``_interval_omitted_statistic_diagnostic``): the
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
        self.coordinate, self.points, self.point_weights = self._geometry(w, weights)

    def _geometry(self, w, weights, /):
        """``(coordinate, points, point_weights)`` of the rows weighted by ``w``.

        ``weights`` is what the coordinate builders see (``None``: unweighted).

        Parameters
        ----------
        w : numpy.ndarray
            Row weights of the coordinate, shape ``(R,)``.
        weights : numpy.ndarray or None
            Weights passed to the coordinate builders (``None``: unweighted).
        """
        support, x = self._support, self._x
        lo, hi = x[:, 0], x[:, 1]
        finite = np.isfinite(lo) & np.isfinite(hi)
        if np.all(finite):
            mid = 0.5 * lo + 0.5 * hi
            coordinate = _build_fit_coordinate(support, mid, weights, hi - lo)
            points, pw = mid, w
        else:
            coordinate = _build_interval_fit_coordinate(support, x, weights)
            one_sided = np.isfinite(lo) ^ np.isfinite(hi)
            cut = np.where(np.isfinite(lo), lo, hi)
            keep = finite | one_sided
            points = np.where(finite, 0.5 * lo + 0.5 * hi, cut)[keep]
            pw = w[keep]
        return coordinate, points, pw / pw.sum()

    def build_weighted(self, spec, responsibilities, /):
        """Compile ``spec`` in the coordinate of the rows weighted by ``r``.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        responsibilities : numpy.ndarray
            Component responsibilities ``r``, shape ``(R,)``.
        """
        cw = self._w * np.asarray(responsibilities, dtype=np.float64)
        uniform = np.all(cw == cw[0])
        coordinate, points, pw = self._geometry(cw, None if uniform else cw / cw.sum())
        return _build_basis(spec, points, pw, None, coordinate)

    def build(self, spec, /):
        """Compile ``spec`` in the interval coordinate and landmark geometry.

        Parameters
        ----------
        spec : BasisSpec
            Basis request.
        """
        return _build_basis(
            spec, self.points, self.point_weights, None, self.coordinate
        )

    def objective(self, model, /):
        """The interval objective of ``model``.

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        """
        return _BasisIntervalObjective(_NormalizerContext(model), self.rows)

    def start(self, model, options, /):
        """G1's interval start: the point fit to uniform-within-row statistics.

        Only with all rows finite (G1 ``_natural_interval_start``); otherwise
        ``None``, the convex cold start (G1's generic interior start).
        Returns ``None`` or ``(theta, face)``.

        Parameters
        ----------
        model : _BasisModel
            Compiled basis.
        options : _NewtonOptions
            G1 Newton policy.
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

        The information is G1's missing-information identity ``Cov[Phi] -
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
    automatic search stops at the fitted span, recorded (G3-049).

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
        identity, G1 ``_interval_omitted_statistic_diagnostic``).
    config : _DegreeSelectionConfig
        G1 policy.
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
    """G1's efficient score test of probe statistics given nuisance statistics.

    The statistics are coordinates of one space with score ``residual``,
    information ``sigma`` and per-row values ``phi`` (weights ``w``); the
    probes are the columns of ``probe`` (``None``: every coordinate) and the
    nuisance the columns of ``nuisance`` (G1: the fitted statistics).  A
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
        Apply G1's point-data participation and variance-inflation guards.
    config : _DegreeSelectionConfig
        G1 selection policy.
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
    # (t = L^T Phi) as nuisance coordinates (G1).
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

    # G1's guards per tested direction: the efficient statistic of direction
    # k is s_k(x) = u_k . (D Phi(x) - P L^T Phi(x)), with model variance
    # values[k].  Its empirical variance (conservative inflation) and the
    # participation of its centred contributions are measured on the data.
    efficient_stats = (phi_p * inv_sd[None, :] - (phi @ lift) @ projection.T) @ vectors
    centred_stats = efficient_stats - w @ efficient_stats
    emp_var = w @ (centred_stats * centred_stats)
    tiny = np.finfo(np.float64).tiny
    inflation = np.clip(
        np.maximum(1.0, emp_var / np.maximum(values, tiny)),
        1.0,
        float(config.max_variance_inflation),
    )
    contribution = w[:, None] * np.abs(centred_stats)
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
    """One automatic decision of a G3 fit.

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
        start, G3-047), else empty.
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
    """Fit ``model``, warm-started from a smaller nested fit when given (G1 lift).

    A cold fit starts where the data say (``data.start``: G1's interval
    start, else the convex cold start).

    Parameters
    ----------
    model : _BasisModel
        Basis to fit.
    data : _PointData or _IntervalData
        Observations.
    options : _NewtonOptions
        G1 Newton policy.
    previous : _Stage or None
        Smaller nested fit to lift, or ``None`` for a cold fit.
    """
    objective = data.objective(model)
    if previous is None:
        start = data.start(model, options)
        if start is None:
            return _solve_basis(objective, options=options, leader_restart=True)
        return _solve_basis(
            objective,
            options=options,
            start=start[0],
            start_face=start[1],
            leader_restart=True,
        )
    start = _lift_matrix(previous.model, model) @ previous.result.params
    face = _face_of(previous.model, previous.result)
    interior_first = (
        isinstance(data, _PointData)
        and model.request.bulk_degree >= 10
        and model.request.lower_infinite != model.request.upper_infinite
        and model.request.lower_tail is None
        and model.request.upper_tail is None
    )
    result = _solve_basis(
        objective,
        options=options,
        start=start,
        start_face=face,
        interior_lift=interior_first,
        leader_restart=True,
    )
    if result.status not in _CONVERGED:
        # Try the alternative treatment of a zero new leader: a bounded
        # interior lift or the face's own cold start. Only a converged
        # second start replaces the first.
        lifted = _solve_basis(
            objective,
            options=options,
            start=start,
            start_face=face,
            interior_lift=not interior_first,
            leader_restart=True,
        )
        if lifted.status in _CONVERGED:
            result = lifted
    if result.status in _CONVERGED and _worse_than_nested(
        result, previous.result, options
    ):
        raise RuntimeError(
            "internal error: a converged larger span is worse than its lifted start"
        )
    return result


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
    """G1 admissible automatic degrees (2-12; even only on the full line).

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


def _fit_degree(spec, lower, upper, data, config, options, /):
    """Minimal-span fit at the requested or automatically selected bulk degree.

    G1 owner adapted: ``natural_objective._fit_natural_conic_points_auto``.
    The ladder fits each admissible degree, probes the next
    ``probe_block_size`` bulk powers (the degree-``d + 2`` basis, nested in
    the same geometry) with the omitted-statistic score test and moves to the
    next admissible degree only when the test rejects.  Retained from G1:
    admissible degrees, probe blocks, the stopping rule, cold fits per rung
    (a lifted smaller fit has a zero leading amplitude: it lies on the larger
    basis's improper boundary, where a first Newton step can pair a 1e-14
    leader with an O(0.1) next term).  Changed: a rung whose cold start does
    not converge is restarted from the lifted smaller fit; a rung the test
    selected that converges from neither start ends the ladder at the smaller
    degree with the failure recorded (the search's answer is the largest
    converged degree, G3-047), and a converged rung must be no worse than the
    nested smaller degree.  An explicit degree whose cold start does not
    converge is refitted from the lifted fit of the next smaller admissible
    degree (fitted by the same rule); an explicit degree, or a ladder whose
    first rung does not converge, returns the unconverged fit and
    ``_fit_data`` raises.

    Parameters
    ----------
    spec : BasisSpec
        Basis request (its ``bulk_degree`` explicit or ``"auto"``).
    lower, upper : bool
        Endpoint logs of this fit.
    data : _PointData or _IntervalData
        Observations.
    config : _DegreeSelectionConfig
        G1 selection policy.
    options : _NewtonOptions
        G1 Newton policy.

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
        seed = None
        if (
            isinstance(data, _PointData)
            and int(spec.bulk_degree) >= 10
            and math.isinf(float(spec.support[0])) != math.isinf(float(spec.support[1]))
            and spec.lower_tail is None
            and spec.upper_tail is None
        ):
            # Keep a well-resolved nested seed even when intermediate high
            # degrees cannot be fitted. A failed rung must not prevent a
            # larger requested family from using that valid starting density.
            seed = _fit_degree(
                dataclasses.replace(spec, bulk_degree=6),
                lower,
                upper,
                data,
                config,
                options,
            )
            if seed.result.status in _CONVERGED:
                result = _warm_fit(model, data, options, seed)
                if result.status in _CONVERGED:
                    return _Stage(model, result, ())
        result = _warm_fit(model, data, options, None)
        smaller = [
            d
            for d in _admissible_degrees(tuple(spec.support), int(spec.bulk_degree) - 1)
            if int(d) >= int(AUTO_POLY_DEGREE_MIN)
        ]
        if (
            result.status not in _CONVERGED
            and smaller
            and (seed is None or seed.result.status not in _CONVERGED)
        ):
            # Second start (G3-047): the lifted fit of the next smaller
            # admissible degree, itself fitted by this rule.  A cold degree-8
            # fit of 600 normal draws stalled at NLL 1.39480, above the
            # converged degree 6 (1.39479).  Only a converged answer replaces
            # the cold one.
            nested = _fit_degree(
                dataclasses.replace(spec, bulk_degree=int(smaller[-1])),
                lower,
                upper,
                data,
                config,
                options,
            )
            if nested.result.status in _CONVERGED:
                lifted = _warm_fit(model, data, options, nested)
                if lifted.status in _CONVERGED:
                    result = lifted
        return _Stage(model, result, ())
    degrees = _bulk_degrees(spec)
    history, current = [], None
    for index, degree in enumerate(degrees):
        try:
            model = data.build(dataclasses.replace(base, bulk_degree=degree))
        except ValueError as exc:
            # Structurally improper at this degree (G3-025): not admissible.
            _reraise_if_debug(exc, "G3 automatic degree rung", routine=True)
            continue
        # Cold, as G1: the lifted smaller fit has a zero leading amplitude,
        # i.e. it lies on this basis's improper boundary (the new leader is
        # not one of the smaller fit's suppressed ends, so no face carries it;
        # released leader faces, G3-042, need a proper start on a face).
        result = _warm_fit(model, data, options, None)
        if result.status not in _CONVERGED and current is not None:
            # A second start of the same problem (G3-047): the lifted smaller
            # fit.  The cold start can stall on a badly scaled rung (degree 6
            # with a cluster 40 scale units out: non_descent at NLL 2.97, the
            # warm start converges to 2.47); either answer is certified.
            result = _warm_fit(model, data, options, current)
        if (
            current is not None
            and result.status in _CONVERGED
            and _worse_than_nested(result, current.result, options)
        ):
            raise RuntimeError(
                f"automatic degree: bulk_degree={degree} converged worse than "
                "the nested smaller degree"
            )
        if result.status not in _CONVERGED:
            if current is None:
                return _Stage(model, result, tuple(history))
            # The search's answer is the largest converged degree (G3-047;
            # G1 skipped a failing candidate); the record says why it stopped.
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
    """Fit a G3 basis to point or interval data, resolving every ``"auto"`` field.

    Order (G1): the bulk degree ladder at the minimal span with every
    automatic endpoint log enabled; G1's backward elimination of automatic
    log terms (``_fit/boundary.py`` ``_select_boundary_terms``: one-sided
    likelihood-ratio test at the same degree, then the degree ladder rerun
    with the final terms); finally the span by ``spec.expansion``.

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    samples : array_like
        Point observations, shape ``(R,)``, or interval rows, shape
        ``(R, 2)`` (infinite ends are censoring; G1 layouts and rules).
    weights : array_like or None
        Observation weights.
    config : _DegreeSelectionConfig or None
        G1 selection policy.
    options : _NewtonOptions or None
        G1 Newton policy.

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
        G1 selection policy.
    options : _NewtonOptions
        G1 Newton policy.

    Returns
    -------
    _BasisFit

    Raises
    ------
    RuntimeError
        If the returned fit's optimizer did not converge (an explicit degree,
        or an automatic search with no converged member, G3-047).
    """
    fit = _search_data(spec, data, log_policy, config, options)
    status = fit.result.status
    if status not in _CONVERGED and status != _UNCERTIFIED:
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
        G1 selection policy.
    options : _NewtonOptions
        G1 Newton policy.
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
        grown = _warm_fit(larger, data, options, _Stage(model, result, ()))
        if grown.status not in _CONVERGED:
            # The search's answer is the largest converged span (G3-047).
            history[-1] = dataclasses.replace(
                history[-1],
                chosen=None,
                failure=(
                    f"the score-selected extension {chosen.label} "
                    f"(p = {chosen.p_value:.3g}) did not converge ({grown.status})"
                ),
            )
            break
        # Likelihood-ratio confirmation (G3-044): at a fit on an active
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
        G1 selection policy.
    options : _NewtonOptions
        G1 Newton policy.
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
