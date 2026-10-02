"""Exact likelihood geometry of one normalized natural state.

The potential ``q = theta . t(z)`` is linear in the natural parameters, so
the first partials are the fixed basis functions ``t`` and the model
information is the covariance ``Cov_theta(t)``.

* Point data enter only through fixed sufficient statistics: the NLL is
  ``theta . E_hat[t] + log Z`` and the Hessian is exactly the Fisher matrix.
* Interval data use current-model conditional expectations over each
  censoring row.  The observed Hessian is the Fisher matrix minus the
  weighted within-row conditional covariance (the missing information), so
  it can be indefinite.  Ordinary finite rows are constructed, normalized
  and reduced in one compiled traversal; rows touching a finite support
  boundary or with an infinite endpoint use the batched adaptive reducer.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyval

from .._defaults import INTERVAL_W_EPS_MULT
from .._model._moment_kernels import product_moment
from .._model.natural_state import _layout_numerics
from .._model.spec import _LOGDIST, _LOWER, _POLY, _UPPER
from .._model.vec import _q_eval
from .._observations._finite_reductions import (
    evaluate_finite_objective as _evaluate_finite_objective_kernel,
)
from .._observations.intervals import (
    _GL_LOG_W,
    _GL_X,
    _IntervalObservations,
    _prepare_partial_interval_reducer,
)
from .._observations.points import _PointObservations


@dataclass(frozen=True)
class _ObjectiveEvaluation:
    """One exact objective/derivative evaluation.

    Parameters
    ----------
    nll : float
        Per-unit-weight negative log likelihood in user coordinates.
    gradient : numpy.ndarray
        Exact gradient with respect to the natural parameters.
    hessian : numpy.ndarray
        Exact observed Hessian ``fisher - missing_information``.
    fisher : numpy.ndarray
        Model covariance of the sufficient statistics.
    missing_information : numpy.ndarray
        Weighted within-row conditional covariance of interval data (zero for
        exact points).
    model_partial_means : numpy.ndarray
        Current model expectations ``E_theta[t_i]``.
    """

    nll: float
    gradient: np.ndarray
    hessian: np.ndarray
    fisher: np.ndarray
    missing_information: np.ndarray
    model_partial_means: np.ndarray


def _finite_interval_log_kernel(state, support, plan, point_q, /):
    """Evaluate the unnormalized log kernel over a finite-interval plan.

    Used by the degree diagnostics, which build the Gauss--Legendre reduction
    of the interval objective.  Exact-point rows evaluate the potential through
    *point_q* so that preserved sub-ulp boundary distances are honored.

    Parameters
    ----------
    state : _NaturalCoreState
        Normalized model state supplying ``q_poly``, ``boundary_amplitudes``
        and ``q_shift``.
    support : tuple of (float, float)
        Canonical support.
    plan : _FiniteIntervalQuadrature
        Reduction plan built by
        :func:`._observations.intervals._build_finite_interval_quadrature`.
    point_q : callable
        ``f(z) -> ndarray`` giving the potential at zero-width rows.  Called
        only when the plan contains such rows.

    Returns
    -------
    nodes : numpy.ndarray, shape (R, G)
        Quadrature nodes.
    log_kernel : numpy.ndarray, shape (R, G)
        ``q_shift - q(nodes)``.
    point_mid : numpy.ndarray, shape (P,)
        Midpoints of the zero-width rows, possibly empty.
    log_integrals : numpy.ndarray, shape (R,)
        Per-row log integrals of the kernel.
    """
    nodes = np.asarray(plan.nodes, dtype=np.float64)
    q_nodes = _q_eval(
        nodes.reshape(-1), support, state.q_poly, state.boundary_amplitudes, 0
    ).reshape(nodes.shape)
    log_kernel = float(state.q_shift) - np.asarray(q_nodes, dtype=np.float64)

    point_mid = plan.midpoints[plan.point_limit]
    point_log_kernel = None
    if point_mid.size:
        point_log_kernel = float(state.q_shift) - np.asarray(
            point_q(point_mid), dtype=np.float64)
    log_integrals = plan.log_integrals(
        log_kernel, point_log_kernel=point_log_kernel
    )
    return nodes, log_kernel, point_mid, log_integrals


def _point_boundary_distance(z, support, side, preserved, /):
    """Return an exact-point boundary distance, preferring physical metadata.

    Parameters
    ----------
    z : float
        Canonical exact-point coordinate.
    support : tuple of float
        Canonical support endpoints.
    side : {'lower', 'upper'}
        Boundary side whose distance is requested.
    preserved : float or None
        Preserved physical boundary distance when available.
    """
    if np.isfinite(preserved):
        return float(preserved)
    lower, upper = map(float, support)
    return float(z - lower) if side == _LOWER else float(upper - z)


def _point_q_with_boundary_distances(state, z, lower_distance, upper_distance, /):
    """Evaluate q at an exact point without losing sub-ulp boundary distance.

    Parameters
    ----------
    state : _NaturalCoreState
        Current model state.
    z : float
        Canonical exact-point coordinate.
    lower_distance : float or None
        Preserved distance to the lower support endpoint.
    upper_distance : float or None
        Preserved distance to the upper support endpoint.
    """
    zz = float(z)
    support = tuple(map(float, state.spec.support))
    q = float(polyval(zz, state.q_poly))
    a_lower, a_upper = map(float, state.boundary_amplitudes)
    if np.isfinite(a_lower) and a_lower > 0.0:
        d = _point_boundary_distance(zz, support, _LOWER, lower_distance)
        if not d > 0.0:
            return np.inf
        q -= a_lower * np.log(d)
    if np.isfinite(a_upper) and a_upper > 0.0:
        d = _point_boundary_distance(zz, support, _UPPER, upper_distance)
        if not d > 0.0:
            return np.inf
        q -= a_upper * np.log(d)
    return float(q)


def _evaluate_point_objective(state, observations, /):
    """Evaluate exact point-data NLL, gradient, and Fisher Hessian.

    The fixed affine-coordinate Jacobian is included in ``nll`` so the value is
    the absolute per-unit-weight negative log likelihood in original user
    coordinates.  It is constant in the parameters and therefore does not
    enter the gradient or Hessian.

    Parameters
    ----------
    state : _NaturalCoreState
        Fully normalized candidate state.
    observations : _PointObservations
        Fixed sufficient-statistic point observation provider.

    Returns
    -------
    _ObjectiveEvaluation
        Exact likelihood geometry for the current parameter vector.

    Raises
    ------
    TypeError
        If ``observations`` is not a point observation provider.
    RuntimeError
        If the candidate state is not normalizable.
    ValueError
        If empirical summary support does not match the model support.
    """
    if not isinstance(observations, _PointObservations):
        raise TypeError("observations must be a _PointObservations")
    if tuple(map(float, observations.stats.support)) != tuple(map(float, state.spec.support)):
        raise ValueError("empirical statistics and model support do not match")
    if not (state.Z > 0.0 and np.isfinite(state.Z) and np.isfinite(state.log_Z)):
        raise RuntimeError("candidate state is not normalizable")

    empirical_q = observations.potential_expectation(state)
    coordinate_constant = float(np.log(state.spec.coordinate.scale))
    nll = float(empirical_q + state.log_Z + coordinate_constant)

    empirical_means = observations.first_expectations(state.partials)
    model_means, fisher = _model_first_means_and_fisher(state)
    # Symmetrize away only roundoff from independent contractions.
    fisher = 0.5 * (fisher + fisher.T)
    return _ObjectiveEvaluation(
        nll=nll,
        gradient=np.asarray(empirical_means - model_means, dtype=np.float64),
        hessian=fisher,
        fisher=fisher,
        missing_information=np.zeros_like(fisher),
        model_partial_means=np.asarray(model_means, dtype=np.float64),
    )


def _model_first_means_and_fisher(state, /):
    """Return model first-partial means and their covariance matrix.

    Parameters
    ----------
    state : _NaturalCoreState
        Current normalized candidate state.

    Returns
    -------
    means : numpy.ndarray
        ``E_theta[t_i]``.
    fisher : numpy.ndarray
        ``Cov_theta(t_i, t_j)``.
    """
    # States normalized by the compiled traversal carry both already.
    cached = state._first_means_fisher
    if cached is not None:
        return cached[0].copy(), cached[1].copy()
    partials = state.partials
    n = len(partials)
    max_poly_degree = max(
        (len(p.coefficients) - 1 for p in partials if p.kind == _POLY),
        default=0,
    )
    power = state.moments.power(2 * max_poly_degree)

    log_power = {}
    for partial in partials:
        if partial.kind == _LOGDIST and partial.boundary_side not in log_power:
            log_power[partial.boundary_side] = state.moments.log_power(
                partial.boundary_side, max_poly_degree
            )

    means = np.empty(n, dtype=np.float64)
    for i, partial in enumerate(partials):
        if partial.kind == _POLY:
            c = np.asarray(partial.coefficients, dtype=np.float64)
            means[i] = float(np.dot(c, power[:c.size]))
        else:
            means[i] = -float(log_power[partial.boundary_side][0])

    second = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            value = _model_partial_product_expectation(
                state, partials[i], partials[j], power, log_power
            )
            second[i, j] = value
            second[j, i] = value
    fisher = second - means[:, None] * means[None, :]
    return means, fisher


def _model_partial_product_expectation(
    state, partial_a, partial_b, power, log_power, /
):
    """Return ``E_theta[t_a t_b]`` for two first-partial descriptors.

    Parameters
    ----------
    state : _NaturalCoreState
        Current normalized candidate state.
    partial_a, partial_b : _PotentialPartial
        First-partial descriptors.
    power : numpy.ndarray
        Cached ordinary moments through the maximum required product degree.
    log_power : dict
        Cached ``E[Z**k log d]`` arrays keyed by canonical boundary side.

    Returns
    -------
    float
        Model expectation of the partial product.
    """
    if partial_a.kind == _POLY and partial_b.kind == _POLY:
        a = np.ascontiguousarray(partial_a.coefficients, dtype=np.float64)
        b = np.ascontiguousarray(partial_b.coefficients, dtype=np.float64)
        m = np.ascontiguousarray(power, dtype=np.float64)
        return float(product_moment(a, b, m))

    if partial_a.kind == _POLY and partial_b.kind == _LOGDIST:
        c = np.asarray(partial_a.coefficients, dtype=np.float64)
        return -float(np.dot(c, log_power[partial_b.boundary_side][:c.size]))
    if partial_a.kind == _LOGDIST and partial_b.kind == _POLY:
        c = np.asarray(partial_b.coefficients, dtype=np.float64)
        return -float(np.dot(c, log_power[partial_a.boundary_side][:c.size]))

    if partial_a.boundary_side == partial_b.boundary_side:
        return float(state.moments.log_square(partial_a.boundary_side))
    return float(state.moments.log_cross())


def _evaluate_interval_objective(state, observations, /):
    """Evaluate exact interval NLL, gradient, Fisher and observed Hessian.

    Interior finite rows use one end-to-end compiled Gauss--Legendre traversal
    for node construction, potential evaluation, stable row normalization, and
    conditional-statistic reduction.  Positive-width rows touching a finite
    support boundary, and rows with an infinite censoring endpoint, use
    interval-local adaptive quadrature shifted at the constrained potential
    minimum, so small tail probabilities are integrated directly rather than
    obtained by subtracting nearly equal CDF values.  An interval that exactly
    equals the model support contributes probability one and zero observed
    information.

    Parameters
    ----------
    state : _NaturalCoreState
        Fully normalized current candidate.
    observations : _IntervalObservations
        Canonical interval rows, including finite, point-limit, and one-/two-
        sided infinite censoring rows.

    Returns
    -------
    _ObjectiveEvaluation
        Exact interval observed-data likelihood geometry.

    Raises
    ------
    TypeError
        If ``observations`` has the wrong type.
    RuntimeError
        If the current model is not normalizable or an interval mass is
        numerically invalid.
    ValueError
        If the observation/model supports disagree.
    """
    if not isinstance(observations, _IntervalObservations):
        raise TypeError("observations must be an _IntervalObservations")
    if observations.support is None or tuple(map(float, observations.support)) != tuple(
        map(float, state.spec.support)
    ):
        raise ValueError("interval observations and model support do not match")
    if not (state.Z > 0.0 and np.isfinite(state.Z) and np.isfinite(state.log_Z)):
        raise RuntimeError("candidate state is not normalizable")

    support = tuple(map(float, state.spec.support))
    n_rows = observations.n_unique
    n_params = state.spec.n_params
    model_h, fisher = _model_first_means_and_fisher(state)

    # The common real-line case has no support-boundary rows and no infinite
    # censoring.  Pass the immutable observation buffers straight to the
    # compiled reducer: constructing masks, flat indices and fancy-index copies
    # on every Newton evaluation is pure overhead.
    full_real_line_finite = (
        np.isneginf(support[0])
        and np.isposinf(support[1])
        and not observations.has_infinite_rows
    )
    if full_real_line_finite:
        numerics = _layout_numerics(state.layout)
        log_probability, observed_h, observed_cov = _evaluate_finite_objective_kernel(
            observations.intervals,
            observations.weights,
            observations.point_lower_distance,
            observations.point_upper_distance,
            state.q_poly,
            state.boundary_amplitudes,
            float(state.q_shift),
            float(np.log(state.Z)),
            float(state.mode),
            float(state.spec.coordinate.scale),
            numerics.kinds,
            numerics.lengths,
            numerics.coefficients,
            float(support[0]),
            float(support[1]),
            _GL_X,
            _GL_LOG_W,
            float(INTERVAL_W_EPS_MULT),
        )
        positive_width = observations.intervals[:, 1] > observations.intervals[:, 0]
    else:
        log_probability = np.empty(n_rows, dtype=np.float64)
        observed_h = np.zeros(n_params, dtype=np.float64)
        observed_cov = np.zeros((n_params, n_params), dtype=np.float64)

        finite_mask = np.asarray(observations.finite_rows, dtype=bool).copy()
        positive_width = observations.widths > 0.0
        boundary_touch = np.zeros(n_rows, dtype=bool)
        if np.isfinite(support[0]):
            boundary_touch |= observations.intervals[:, 0] == support[0]
        if np.isfinite(support[1]):
            boundary_touch |= observations.intervals[:, 1] == support[1]
        adaptive_boundary_mask = finite_mask & positive_width & boundary_touch
        regular_finite_mask = finite_mask & ~adaptive_boundary_mask

        finite_idx = np.flatnonzero(regular_finite_mask)
        if finite_idx.size:
            numerics = _layout_numerics(state.layout)
            finite_logp, finite_h, finite_cov = (
                _evaluate_finite_objective_kernel(
                    np.ascontiguousarray(observations.intervals[finite_idx], dtype=np.float64),
                    np.ascontiguousarray(observations.weights[finite_idx], dtype=np.float64),
                    np.ascontiguousarray(
                        observations.point_lower_distance[finite_idx], dtype=np.float64
                    ),
                    np.ascontiguousarray(
                        observations.point_upper_distance[finite_idx], dtype=np.float64
                    ),
                    state.q_poly,
                    state.boundary_amplitudes,
                    float(state.q_shift),
                    float(np.log(state.Z)),
                    float(state.mode),
                    float(state.spec.coordinate.scale),
                    numerics.kinds,
                    numerics.lengths,
                    numerics.coefficients,
                    float(support[0]),
                    float(support[1]),
                    _GL_X,
                    _GL_LOG_W,
                    float(INTERVAL_W_EPS_MULT),
                )
            )
            log_probability[finite_idx] = finite_logp
            observed_h += finite_h
            observed_cov += finite_cov

        adaptive_idx = np.flatnonzero((~observations.finite_rows) | adaptive_boundary_mask)
        if adaptive_idx.size:
            adaptive_rows = np.ascontiguousarray(
                observations.intervals[adaptive_idx], dtype=np.float64
            )
            whole_support = (
                (adaptive_rows[:, 0] == float(support[0]))
                & (adaptive_rows[:, 1] == float(support[1]))
            )
            if np.any(whole_support):
                whole_idx = adaptive_idx[whole_support]
                log_probability[whole_idx] = 0.0
                whole_weight = float(np.sum(observations.weights[whole_idx]))
                observed_h += whole_weight * model_h
                observed_cov += whole_weight * fisher

            reduce_mask = ~whole_support
            if np.any(reduce_mask):
                reduce_idx = adaptive_idx[reduce_mask]
                adaptive_reducer = _prepare_partial_interval_reducer(state)
                batch = adaptive_reducer.reduce_weighted(
                    np.ascontiguousarray(adaptive_rows[reduce_mask], dtype=np.float64),
                    np.ascontiguousarray(observations.weights[reduce_idx], dtype=np.float64),
                )
                log_probability[reduce_idx] = batch[0]
                observed_h += np.asarray(batch[1], dtype=np.float64)
                observed_cov += np.asarray(batch[2], dtype=np.float64)

    if (
        not np.all(np.isfinite(log_probability))
        or np.any(log_probability[positive_width] > 1e-7)
    ):
        raise RuntimeError("candidate interval probabilities are numerically invalid")
    nll = -float(np.dot(observations.weights, log_probability))

    fisher = 0.5 * (fisher + fisher.T)
    missing = 0.5 * (observed_cov + observed_cov.T)
    return _ObjectiveEvaluation(
        nll=nll,
        gradient=np.asarray(observed_h - model_h, dtype=np.float64),
        hessian=fisher - missing,
        fisher=fisher,
        missing_information=missing,
        model_partial_means=np.asarray(model_h, dtype=np.float64),
    )
