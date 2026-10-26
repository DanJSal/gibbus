"""Weighted point and interval observation objectives of the basis.

An objective is a callable (``build_state`` and ``__call__``) returning one
evaluation record; the Newton controller, interval EM and mixture M-steps
consume it.  The NLL is per unit weight in user coordinates, and a
non-normalizable candidate raises ``RuntimeError``.  The sufficient
statistics are the basis columns ``Phi``.

With fixed sufficient statistics ``Phi_bar = sum_i w_i Phi(x_i) / sum_i w_i``
the negative log likelihood per unit weight is

    ``nll(theta) = theta . Phi_bar + log Z(theta)``

with gradient ``Phi_bar - E_theta[Phi]`` and Hessian ``Cov_theta[Phi]`` (the
exact Fisher matrix, positive semidefinite; the missing information of point
data is zero).  All model quantities come from one normalized state
(``basis_normalizer``).

Point observations exactly at a finite endpoint with an enabled logarithm are
rejected at the public boundary (``_fit/inputs.py``; an automatic log term
resolves to no term), so the statistics here are always finite.

**Interval rows.**  With row masses ``P(B_r)`` and normalized
weights ``w_r`` the NLL per unit weight is ``-sum_r w_r log P(B_r)``, a
difference of convex functions: gradient ``sum_r w_r E[Phi | B_r] -
E[Phi]``, observed Hessian ``F - C`` with ``F = Cov[Phi]`` and the missing
information ``C = sum_r w_r Cov(Phi | B_r)`` (``basis_normalizer.
_interval_state``).  ``F`` is the Hessian of the EM surrogate, which has the
same gradient.  The Newton metric is ``F - C`` wherever it is positive
definite relative to ``F`` and otherwise its saddle-free reflection
(compiled); on a face the
metric is formed from the restricted ``F`` and ``C``.  Zero-width rows are
point observations (log densities), rows covering the support contribute
nothing.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._model.basis_eval import _evaluate_basis
from .._model.basis_normalizer import (
    _basis_state,
    _failure,
    _ImproperPotential,
    _interval_state,
    _NormalizerContext,
    _NormalizerOptions,
)
from .._observations.empirical import _normalized_weights
from .._observations.intervals import _GL_W, _GL_X, _build_interval_observations
from ._basis_newton_kernels import KIND_INTERVAL, KIND_POINT
from ._conic_kernels import safeguarded_metric


@dataclass(frozen=True)
class _ObjectiveEvaluation:
    """One exact objective/derivative evaluation.

    Parameters
    ----------
    nll : float
        Per-unit-weight negative log likelihood in user coordinates.
    gradient : numpy.ndarray
        Exact gradient with respect to the coefficients, shape ``(d,)``.
    hessian : numpy.ndarray
        Newton metric: the exact observed Hessian ``fisher -
        missing_information``, or its saddle-free reflection where that is not
        positive definite relative to ``fisher`` (interval data).
    fisher : numpy.ndarray
        Model covariance of the sufficient statistics, shape ``(d, d)``.
    missing_information : numpy.ndarray
        Weighted within-row conditional covariance of interval data (zero for
        exact points).
    model_partial_means : numpy.ndarray
        Current model expectations ``E_theta[Phi]``.
    state : _BasisState
        The normalized state (error estimates, nodes, weights).
    smallest_curvature : float
        Smallest eigenvalue of the observed Hessian relative to ``fisher``
        (``nan`` for point data, whose observed Hessian is ``fisher``).
    """

    nll: float
    gradient: np.ndarray
    hessian: np.ndarray
    fisher: np.ndarray
    missing_information: np.ndarray
    model_partial_means: np.ndarray
    state: object
    smallest_curvature: float = math.nan


@dataclass(frozen=True)
class _PointStatistics:
    """Weighted sufficient statistics of point observations.

    Parameters
    ----------
    mean : numpy.ndarray
        ``Phi_bar``, shape ``(d,)``.
    total_weight : float
        ``sum_i w_i``.
    count : int
        Number of observations with positive weight.
    points, weights : numpy.ndarray or None
        Retained positive-weight observations for direct recomputation in an
        equivalent runtime basis. Synthetic moment objectives may omit them.
    """

    mean: np.ndarray
    total_weight: float
    count: int
    points: object = None
    weights: object = None


def _point_statistics(model, x, weights=None, /):
    """Compute ``Phi_bar`` for point observations.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    x : numpy.ndarray
        Canonical finite observations in user units, shape ``(n,)``, inside
        the closed support and not on an enabled log endpoint.
    weights : numpy.ndarray or None
        Nonnegative finite weights (default all one).

    Returns
    -------
    _PointStatistics

    Raises
    ------
    ValueError
        If no observation has positive weight, a weight is negative or not
        finite, or a statistic is not finite (a positive-weight point on an
        enabled log endpoint, which the public boundary rejects, or an
        exponential column overflowing at an extreme observation).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != x.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite, nonnegative and match x")
    keep = w > 0.0
    if not np.any(keep):
        raise ValueError("at least one observation needs positive weight")
    x, w = x[keep], w[keep]
    phi = _evaluate_basis(model, x, max_order=0).phi
    if not np.all(np.isfinite(phi)):
        raise ValueError(
            "point statistics are not finite: a point on an enabled log endpoint, "
            "or an exponential column overflowing binary64 at an extreme observation"
        )
    total = float(w.sum())
    return _PointStatistics((w @ phi) / total, total, int(x.size), x.copy(), w.copy())


def _normalizer_block(context, /):
    """``(tail_lo, tail_hi, growth_lo, growth_hi, column_scale)`` of a context.

    Parameters
    ----------
    context : _NormalizerContext
        Normalizer of the basis.
    """
    tails = context.tails
    return (
        tails["lower"].kernel if "lower" in tails else None,
        tails["upper"].kernel if "upper" in tails else None,
        context.growth_kernels.get("lower"),
        context.growth_kernels.get("upper"),
        context.column_scale,
    )


def _trial_failure(context, options, status, side, diag, theta, row, rows, /):
    """The ``_QuadratureFailure`` of a compiled trial (row named when given).

    Parameters
    ----------
    context : _NormalizerContext
        Normalizer of the basis.
    options : _NormalizerOptions
        Normalizer policy.
    status : int
        Normalizer status.
    side : int
        Side code.
    diag : float
        Diagnostic value.
    theta : numpy.ndarray
        Trial coefficients.
    row : int
        Interval row, or -1.
    rows : _IntervalRows or None
        Interval rows that name the failing row.
    """
    exc = _failure(context, theta, options or _NormalizerOptions(), status, side, diag)
    if row >= 0:
        exc.diagnostics["row"] = int(row)
        exc.diagnostics["interval"] = (float(rows.lower[row]), float(rows.upper[row]))
    return exc


class _BasisPointObjective:
    """Callable point objective bound to one basis and fixed statistics.

    Parameters
    ----------
    context : _NormalizerContext
        Compiled-basis normalizer data.
    stats : _PointStatistics
        Fixed sufficient statistics.
    options : _NormalizerOptions or None
        Normalizer settings.
    """

    def __init__(self, context, stats, options=None, /):
        """Bind a basis to fixed point statistics (see class docstring)."""
        self.context = context
        self.model = context.model
        self.stats = stats
        self.options = options
        self._zero_missing = np.zeros((stats.mean.size, stats.mean.size))

    def coordinate_observations(self, block):
        """Retained observations for direct statistics in runtime coordinates."""
        del block
        if self.stats.points is None:
            return None
        return self.stats.points, self.stats.weights

    def in_coordinates(self, blocks):
        """Recompute sufficient statistics from observations in the new basis."""
        model = blocks.model
        return _BasisPointObjective(
            _NormalizerContext(model),
            _point_statistics(model, self.stats.points, self.stats.weights),
            self.options,
        )

    def on_face(self, basis, /):
        """The same objective for points of a face ``theta = N phi``.

        Parameters
        ----------
        basis : numpy.ndarray
            Orthonormal face basis ``N``.
        """
        return _BasisPointObjective(
            self.context.on_face(basis), self.stats, self.options
        )

    def metric(self, fisher, missing, /):
        """Newton metric of restricted ``F`` and ``C``: ``F`` (point data).

        Parameters
        ----------
        fisher, missing : numpy.ndarray
            Restricted Fisher matrix and missing information.

        Returns
        -------
        tuple
            ``(metric, smallest_curvature)``.
        """
        return fisher, math.nan

    def compiled_spec(self):
        """``(kind, data, normalizer_blocks)`` of the compiled face objective."""
        data = {"stats": [self.stats.mean]}
        return KIND_POINT, data, [_normalizer_block(self.context)]

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
            Trial coefficients.
        row : int
            Interval row of the failure, or ``-1``.
        block : int
            Coefficient block (``0`` for a single fit).
        """
        del row, block
        return _trial_failure(
            self.context, self.options, status, side, diag, theta, -1, None
        )

    def build_state(self, theta, /):
        """Build and normalize one candidate state.

        Parameters
        ----------
        theta : numpy.ndarray
            Coefficients, shape ``(d,)``.
        """
        return _basis_state(self.context, theta, self.options)

    def __call__(self, theta, /):
        """Return the exact point NLL, gradient and Fisher Hessian.

        Parameters
        ----------
        theta : numpy.ndarray
            Coefficients, shape ``(d,)``.

        Returns
        -------
        _ObjectiveEvaluation

        Raises
        ------
        _ImproperPotential
            If ``Z = inf`` (a ``RuntimeError``).
        _QuadratureFailure
            Propagated from the normalizer (fails loudly).
        """
        state = self.build_state(theta)
        if state.status != "proper":
            raise _ImproperPotential("candidate state is not normalizable")
        fisher = state.cov
        return _ObjectiveEvaluation(
            nll=float(state.theta @ self.stats.mean) + state.log_z,
            gradient=self.stats.mean - state.mean,
            hessian=fisher,
            fisher=fisher,
            missing_information=self._zero_missing,
            model_partial_means=state.mean,
            state=state,
        )


@dataclass(frozen=True)
class _IntervalRows:
    """Canonical interval rows in user units.

    Duplicates merged and weights normalized (``_observations.intervals``);
    a zero-width row is a point
    observation, an infinite end a censoring row.

    Parameters
    ----------
    lower, upper : numpy.ndarray
        Row endpoints, shape ``(R,)``, inside the closed support.
    weights : numpy.ndarray
        Normalized weights (sum one), shape ``(R,)``.
    effective_n : float
        Kish effective size of the original rows.
    count : int
        Number of original rows.
    support : tuple of float
        Closed user support.
    """

    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray
    effective_n: float
    count: int
    support: tuple

    @property
    def intervals(self):
        """Rows as ``(R, 2)``."""
        return np.column_stack([self.lower, self.upper])

    @property
    def has_infinite_rows(self):
        """Whether a row has an infinite end (censoring)."""
        return bool(
            np.any(~np.isfinite(self.lower)) or np.any(~np.isfinite(self.upper))
        )


def _interval_rows(support, intervals, weights=None, /):
    """Canonical interval rows of user-unit observations.

    Parameters
    ----------
    support : tuple of float
        Closed user support.
    intervals : array_like
        Ordered rows ``(lower, upper)``, shape ``(R, 2)``, inside the support;
        endpoints may be infinite.
    weights : array_like or None
        Nonnegative finite weights (default all one).

    Returns
    -------
    _IntervalRows

    Raises
    ------
    ValueError
        If the rows are not ordered, contain NaN, leave the support or no
        row has positive weight.
    """
    x = np.ascontiguousarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] < 1:
        raise ValueError("intervals must have shape (R, 2)")
    if np.any(np.isnan(x)) or np.any(x[:, 0] > x[:, 1]):
        raise ValueError("intervals must be ordered and contain no NaN")
    obs = _build_interval_observations(
        x,
        _normalized_weights(x.shape[0], weights, "interval"),
        support=tuple(map(float, support)),
    )
    keep = obs.weights > 0.0
    return _IntervalRows(
        np.ascontiguousarray(obs.intervals[keep, 0]),
        np.ascontiguousarray(obs.intervals[keep, 1]),
        np.ascontiguousarray(obs.weights[keep] / np.sum(obs.weights[keep])),
        float(obs.effective_n),
        int(obs.n_observations),
        tuple(map(float, support)),
    )


def _uniform_statistics(model, rows, /):
    """Point statistics of uniforms within finite rows (the interval start).

    Each row's ``Phi`` mean by the ``INTERVAL_GL_ORDER``-point Gauss-Legendre rule (``Phi(x)`` for a point
    row).  Only a starting point is built from it; the fit is the interval
    likelihood's.

    Parameters
    ----------
    model : _BasisModel
        Compiled basis.
    rows : _IntervalRows
        Rows, all finite.

    Returns
    -------
    _PointStatistics

    Raises
    ------
    ValueError
        If a row is infinite or a statistic is not finite.
    """
    lo, hi = rows.lower, rows.upper
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("uniform statistics need finite rows")
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    nodes = mid[:, None] + half[:, None] * _GL_X[None, :]
    phi = _evaluate_basis(model, nodes.reshape(-1), max_order=0).phi
    phi = phi.reshape(lo.size, _GL_X.size, -1)
    row_mean = 0.5 * np.einsum("k,rkc->rc", _GL_W, phi)
    if not np.all(np.isfinite(row_mean)):
        raise ValueError("uniform interval statistics are not finite")
    return _PointStatistics(rows.weights @ row_mean, 1.0, int(lo.size))


class _BasisIntervalObjective:
    """Callable interval objective bound to one basis and fixed rows.

    Parameters
    ----------
    context : _NormalizerContext
        Compiled-basis normalizer data.
    rows : _IntervalRows
        Fixed observation rows.
    options : _NormalizerOptions or None
        Normalizer settings.
    """

    def __init__(self, context, rows, options=None, /):
        """Bind a basis to fixed interval rows (see class docstring)."""
        self.context = context
        self.model = context.model
        self.rows = rows
        self.options = options

    def coordinate_observations(self, block):
        """Intervals are reduced directly by the normalizer in either basis."""
        del block
        return None, None

    def in_coordinates(self, blocks):
        """Bind the original interval rows to the numerical basis."""
        return _BasisIntervalObjective(
            _NormalizerContext(blocks.model), self.rows, self.options
        )

    def on_face(self, basis, /):
        """The same objective for points of a face ``theta = N phi``.

        Parameters
        ----------
        basis : numpy.ndarray
            Orthonormal face basis ``N``.
        """
        return _BasisIntervalObjective(
            self.context.on_face(basis), self.rows, self.options
        )

    def lower_bound(self, nll, options, /):
        """Nonparametric NLL lower bound, run until a fit's certificate is decided.

        Turnbull EM with its Frank-Wolfe duality gap (a bound on the NLL of
        every distribution), stopped as soon as the certificate rule
        (``conic_newton._apply_global_objective_certificate``) is decided for
        a fit of NLL ``nll``; ``None`` with point rows, whose
        nonparametric likelihood is unbounded.

        Parameters
        ----------
        nll : float
            NLL of the fit to certify.
        options : _NewtonOptions
            Certificate tolerances.
        """
        return _interval_nll_lower_bound(
            self.rows,
            decide=(
                float(nll),
                options.certified_tolerance,
                options.accuracy_floor,
            ),
        )

    def metric(self, fisher, missing, /):
        """Saddle-free Newton metric of restricted ``F`` and ``C``.

        Parameters
        ----------
        fisher, missing : numpy.ndarray
            Restricted Fisher matrix and missing information.

        Returns
        -------
        tuple
            ``(metric, smallest_curvature)``.
        """
        return safeguarded_metric(fisher, missing)

    def compiled_spec(self):
        """``(kind, data, normalizer_blocks)`` of the compiled face objective."""
        rows = self.rows
        data = {"rows": (rows.lower, rows.upper), "weights": rows.weights}
        return KIND_INTERVAL, data, [_normalizer_block(self.context)]

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
            Trial coefficients.
        row : int
            Interval row of the failure, or ``-1``.
        block : int
            Coefficient block (``0`` for a single fit).
        """
        del block
        return _trial_failure(
            self.context, self.options, status, side, diag, theta, row, self.rows
        )

    def build_state(self, theta, /):
        """Normalize one candidate and reduce the rows (``_IntervalState``).

        Parameters
        ----------
        theta : numpy.ndarray
            Coefficients, shape ``(d,)``.
        """
        rows = self.rows
        return _interval_state(
            self.context, theta, rows.lower, rows.upper, rows.weights, self.options
        )

    def __call__(self, theta, /):
        """Return the observed NLL, gradient and Newton metric.

        Parameters
        ----------
        theta : numpy.ndarray
            Coefficients, shape ``(d,)``.

        Returns
        -------
        _ObjectiveEvaluation
            ``state`` is the normalized ``_BasisState``.

        Raises
        ------
        _ImproperPotential
            If ``Z = inf``.
        _QuadratureFailure
            Propagated from the normalizer or a row (fails loudly).
        """
        reduced = self.build_state(theta)
        state = reduced.state
        fisher = state.cov
        missing = reduced.cov
        metric, smallest = safeguarded_metric(fisher, missing)
        return _ObjectiveEvaluation(
            nll=-float(reduced.log_p),
            gradient=reduced.mean - state.mean,
            hessian=metric,
            fisher=fisher,
            missing_information=missing,
            model_partial_means=state.mean,
            state=state,
            smallest_curvature=smallest,
        )


def _interval_nll_lower_bound(
    observations, /, *, max_iterations=20000, tolerance=1e-13, decide=None
):
    """Return a rigorous lower bound on every interval NLL for these rows.

    The censoring endpoints partition the support into atoms; any
    distribution assigns them masses ``p`` and the interval log likelihood
    ``l(p) = sum_r w_r log (A p)_r`` is concave on the simplex.  Turnbull's EM
    update climbs it, and the Frank-Wolfe duality gap
    ``max_j dl/dp_j - p . grad l = max_j cover_j - 1`` bounds the remaining
    ascent, so ``-(l(p) + gap)`` is a lower bound on the NLL of *every*
    distribution, parametric or not, at every iterate.  When a parametric fit
    reaches it, the fit is globally optimal to that accuracy even though the
    parametric likelihood is not convex.

    Parameters
    ----------
    observations : _IntervalObservations
        Canonical interval rows with normalized weights.
    max_iterations : int, optional
        Turnbull iteration limit.
    tolerance : float, optional
        Stop once the duality gap is below this value.
    decide : tuple or None, optional
        ``(nll, certified, floor)``: stop as soon as the certificate of a fit
        with this NLL is decided: the bound is within ``certified``
        of ``nll`` (certified), or the current iterate's NLL ``-l(p)``, which
        the nonparametric optimum cannot exceed, lies more than ``floor``
        below ``nll`` (no bound can come within ``floor``).  Either way the
        returned value is a valid bound.

    Returns
    -------
    float or None
        The bound, or ``None`` when zero-width rows are present (they
        contribute densities, and the nonparametric likelihood is unbounded).
    """
    rows = np.asarray(observations.intervals, dtype=np.float64)
    weights = np.asarray(observations.weights, dtype=np.float64)
    if rows.size == 0 or np.any(rows[:, 0] == rows[:, 1]):
        return None
    lower, upper = map(float, observations.support)
    endpoints = np.unique(np.concatenate(([lower, upper], rows.reshape(-1))))
    starts = np.searchsorted(endpoints, rows[:, 0])
    ends = np.searchsorted(endpoints, rows[:, 1])
    atoms = int(endpoints.size - 1)
    mass = np.full(atoms, 1.0 / atoms, dtype=np.longdouble)
    total_weight = float(np.sum(weights))
    best = np.inf
    for _ in range(int(max_iterations)):
        prefix = np.concatenate(([0.0], np.cumsum(mass))).astype(np.longdouble)
        row_mass = prefix[ends] - prefix[starts]
        if np.any(row_mass <= 0.0):
            return None
        log_likelihood = float(np.sum(weights * np.log(row_mass)))
        coefficient = weights / row_mass
        delta = np.zeros(atoms + 1, dtype=np.longdouble)
        np.add.at(delta, starts, coefficient)
        np.add.at(delta, ends, -coefficient)
        cover = np.cumsum(delta[:-1])
        gap = float(np.max(cover)) - total_weight
        best = min(best, log_likelihood + max(gap, 0.0))
        if gap <= tolerance:
            break
        if decide is not None:
            nll, certified, floor = decide
            margin = 64.0 * np.finfo(float).eps * (1.0 + abs(best))
            if nll - (-best - margin) <= certified or nll + log_likelihood > floor:
                break
        mass = mass * cover
        mass = mass / np.sum(mass)
    # Margin for the rounding of the logarithms and the weighted sum.
    return float(-best - 64.0 * np.finfo(float).eps * (1.0 + abs(best)))
