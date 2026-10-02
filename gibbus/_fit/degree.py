"""Information-based polynomial-degree diagnostics for point and interval fits.

The selector asks whether power statistics omitted by the current maximum
polynomial degree contain statistically resolvable information after the
statistics already represented by the fitted model are treated as nuisance
coordinates.  Model covariance supplies the score geometry; observation-level
moment participation and empirical/model variance mismatch provide a
conservative reliability guard against tail-dominated high-order moments.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .._model.moments import _ModelMoments
from .._observations.intervals import (
    _build_finite_interval_quadrature,
    _prepare_statistic_interval_reducer,
)
from .objective import _finite_interval_log_kernel, _point_q_with_boundary_distances

_LOWER = "lower"
_UPPER = "upper"


@dataclass(frozen=True)
class _DegreeSelectionConfig:
    """Policy controlling one omitted-statistic degree diagnostic.

    Parameters
    ----------
    probe_block_size : int
        Number of consecutive omitted power statistics examined at once.
    alpha : float
        Upper-tail score-test probability required before model capacity is
        increased.
    min_participation : float
        Minimum contribution-participation effective sample size for an
        omitted moment to enter the score test.
    covariance_rtol : float
        Relative eigenvalue tolerance used by rank-aware covariance solves.
    max_variance_inflation : float
        Maximum conservative inflation applied when empirical variability of
        an omitted statistic exceeds the current model's variability.
    """

    probe_block_size: int = 2
    alpha: float = 0.05
    min_participation: float = 8.0
    covariance_rtol: float = 1e-10
    max_variance_inflation: float = 1e6


@dataclass(frozen=True)
class _DegreeDiagnostic:
    """Information diagnostic for one fitted degree and omitted power block.

    Parameters
    ----------
    fitted_degree : int
        Current requested polynomial degree.
    probe_orders : tuple of int
        Consecutive omitted power orders examined.
    reliable_mask : numpy.ndarray
        Boolean mask selecting omitted statistics with adequate empirical
        participation.
    participation : numpy.ndarray
        Observation-level participation effective sample sizes.
    residual : numpy.ndarray
        Raw empirical-minus-model omitted power residuals.
    efficient_residual : numpy.ndarray
        Residual after projecting current fitted-statistic mismatch out through
        the model covariance geometry.
    conditional_covariance : numpy.ndarray
        Model Schur-complement covariance of omitted powers after conditioning
        on fitted polynomial/log statistics.
    test_covariance : numpy.ndarray
        Conservative covariance used by the score test after empirical/model
        variance inflation.
    variance_inflation : numpy.ndarray
        Per-statistic empirical/model unconditional variance inflation factors.
    standardized_residual : numpy.ndarray
        Efficient residuals in sampling-standard-error units.
    score : float
        Rank-aware quadratic score statistic on reliable omitted directions.
    rank : int
        Numerical rank of the reliable test covariance.
    p_value : float
        Chi-square reference upper-tail probability.  This is a calibration
        device, not a claim of exact boundary asymptotics.
    should_expand : bool
        Whether the current policy recommends increasing maximum degree.
    stopped_for_reliability : bool
        True when no omitted statistic in the probe block is sufficiently
        supported by the empirical sample.
    """

    fitted_degree: int
    probe_orders: tuple[int, ...]
    reliable_mask: np.ndarray
    participation: np.ndarray
    residual: np.ndarray
    efficient_residual: np.ndarray
    conditional_covariance: np.ndarray
    test_covariance: np.ndarray
    variance_inflation: np.ndarray
    standardized_residual: np.ndarray
    score: float
    rank: int
    p_value: float
    should_expand: bool
    stopped_for_reliability: bool


def _omitted_statistic_diagnostic(fit, probe_orders, /, *, config=None):
    """Diagnose unresolved omitted power statistics for one point fit.

    Parameters
    ----------
    fit : _NaturalDegreeDiagnosticFit
        Converged point fit whose empirical summary contains moments
        through twice the largest probe order.
    probe_orders : iterable of int
        Strictly increasing omitted power orders, all above the fitted degree.
    config : _DegreeSelectionConfig or None, optional
        Diagnostic policy.

    Returns
    -------
    _DegreeDiagnostic
        Residual, covariance, reliability, and score-test information.

    Raises
    ------
    ValueError
        If probe orders are invalid or the empirical summary lacks required
        moments.
    """
    cfg = _DegreeSelectionConfig() if config is None else config
    orders = tuple(int(k) for k in probe_orders)
    if not orders or any(k < 1 for k in orders):
        raise ValueError("probe_orders must contain positive power orders")
    if tuple(sorted(set(orders))) != orders:
        raise ValueError("probe_orders must be strictly increasing")

    degree = int(fit.spec.requested_poly_degree)
    if orders[0] <= degree:
        raise ValueError("probe orders must lie above the fitted degree")
    max_order = int(orders[-1])
    stats = fit.observations.stats
    if stats.max_order < 2 * max_order:
        raise ValueError(
            "empirical summary needs moments through twice the probe order"
        )

    moments = _ModelMoments(fit.state)
    model_power = moments.power(2 * max_order)
    nuisance_orders = tuple(range(1, degree + 1))
    sides = _enabled_boundary_sides(fit.spec)

    nuisance_mean, nuisance_empirical, sigma_tt = _nuisance_geometry(
        fit, moments, model_power, nuisance_orders, sides, max_order
    )
    probe_mean = model_power[np.asarray(orders, dtype=int)]
    probe_empirical = stats.moments[np.asarray(orders, dtype=int)]
    residual = np.asarray(probe_empirical - probe_mean, dtype=np.float64)

    sigma_uu = _power_covariance_block(model_power, orders, orders)
    sigma_ut = _probe_nuisance_covariance(
        moments, model_power, orders, nuisance_orders, sides, max_order
    )

    if sigma_tt.size:
        tt_pinv, _ = _psd_pinv(sigma_tt, cfg.covariance_rtol)
        projection = sigma_ut @ tt_pinv
        nuisance_residual = nuisance_empirical - nuisance_mean
        efficient_residual = residual - projection @ nuisance_residual
        conditional = sigma_uu - projection @ sigma_ut.T
    else:
        efficient_residual = residual.copy()
        conditional = sigma_uu.copy()
    conditional = _project_psd(conditional, cfg.covariance_rtol)

    empirical_cov = stats.power_covariance(max_order)
    empirical_var = np.diag(empirical_cov)[np.asarray(orders, dtype=int)]
    model_var = np.diag(_power_covariance_block(model_power, orders, orders))
    tiny = np.finfo(np.float64).tiny
    ratio = np.divide(
        empirical_var,
        np.maximum(model_var, tiny),
        out=np.ones_like(empirical_var),
        where=np.isfinite(empirical_var),
    )
    inflation = np.clip(np.maximum(1.0, ratio), 1.0, float(cfg.max_variance_inflation))
    scale = np.sqrt(inflation)
    test_cov = conditional * scale[:, None] * scale[None, :]
    test_cov = _project_psd(test_cov, cfg.covariance_rtol)

    participation = np.array(
        [stats.moment_participation(k) for k in orders], dtype=np.float64
    )
    reliable = (~np.isfinite(participation)) | (
        participation >= float(cfg.min_participation)
    )

    diag = np.maximum(np.diag(test_cov), 0.0)
    se = np.sqrt(diag / max(float(stats.effective_n), 1.0))
    standardized = np.divide(
        efficient_residual,
        se,
        out=np.zeros_like(efficient_residual),
        where=se > 0.0,
    )

    stopped = not bool(np.any(reliable))
    score = 0.0
    rank = 0
    p_value = 1.0
    should_expand = False
    if not stopped:
        rr = efficient_residual[reliable]
        ss = test_cov[np.ix_(reliable, reliable)]
        ss_pinv, rank = _psd_pinv(ss, cfg.covariance_rtol)
        if rank > 0:
            score = float(stats.effective_n * (rr @ ss_pinv @ rr))
            p_value = float(chi2.sf(max(score, 0.0), rank))
            should_expand = bool(p_value < float(cfg.alpha))

    return _DegreeDiagnostic(
        fitted_degree=degree,
        probe_orders=orders,
        reliable_mask=np.asarray(reliable, dtype=bool),
        participation=participation,
        residual=residual,
        efficient_residual=np.asarray(efficient_residual, dtype=np.float64),
        conditional_covariance=conditional,
        test_covariance=test_cov,
        variance_inflation=inflation,
        standardized_residual=standardized,
        score=float(score),
        rank=int(rank),
        p_value=float(p_value),
        should_expand=bool(should_expand),
        stopped_for_reliability=bool(stopped),
    )


def _probe_orders_for_degree(degree, max_degree, /, *, block_size=2):
    """Return the next consecutive omitted power block within a probe ceiling.

    Parameters
    ----------
    degree : int
        Current fitted polynomial degree.
    max_degree : int
        Highest power order that may be probed.
    block_size : int, optional
        Maximum number of consecutive omitted powers.

    Returns
    -------
    tuple of int
        Orders ``degree+1, ...`` up to the requested block size/ceiling.
    """
    d = int(degree)
    ceiling = int(max_degree)
    count = max(1, int(block_size))
    hi = min(ceiling, d + count)
    if hi <= d:
        return ()
    return tuple(range(d + 1, hi + 1))


def _enabled_boundary_sides(spec, /):
    """Return canonical fixed log-boundary sides present in one model spec.

    Parameters
    ----------
    spec : _ModelSpec
        Model specification whose enabled canonical boundary amplitudes are
        inspected.

    Returns
    -------
    tuple of str
        Enabled canonical side names.
    """
    sides = []
    if spec.canonical_lower_a_index is not None:
        sides.append(_LOWER)
    if spec.canonical_upper_a_index is not None:
        sides.append(_UPPER)
    return tuple(sides)


def _nuisance_geometry(fit, moments, model_power, power_orders, sides, max_order, /):
    """Return nuisance model/empirical means and covariance.

    Parameters
    ----------
    fit : _NaturalDegreeDiagnosticFit
        Current point fit.
    moments : _ModelMoments
        Current model-moment service.
    model_power : numpy.ndarray
        Cached model power moments.
    power_orders : tuple of int
        Fitted polynomial-statistic orders.
    sides : tuple of str
        Enabled canonical boundary-log sides.
    max_order : int
        Highest power order required by generalized moments.

    Returns
    -------
    tuple
        Model means, empirical means, and nuisance covariance matrix.
    """
    stats = fit.observations.stats
    n_power = len(power_orders)
    n_total = n_power + len(sides)
    if n_total == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, np.empty((0, 0), dtype=np.float64)

    mean = np.empty(n_total, dtype=np.float64)
    empirical = np.empty(n_total, dtype=np.float64)
    if n_power:
        idx = np.asarray(power_orders, dtype=int)
        mean[:n_power] = model_power[idx]
        empirical[:n_power] = stats.moments[idx]
    for j, side in enumerate(sides, start=n_power):
        mean[j] = -moments.log_power(side, 0)[0]
        slot = 0 if side == _LOWER else 1
        empirical[j] = stats.boundary_log[slot]

    cov = np.empty((n_total, n_total), dtype=np.float64)
    if n_power:
        cov[:n_power, :n_power] = _power_covariance_block(
            model_power, power_orders, power_orders
        )
    for p, order in enumerate(power_orders):
        for j, side in enumerate(sides, start=n_power):
            e_product = -moments.log_power(side, max_order)[order]
            value = e_product - mean[p] * mean[j]
            cov[p, j] = value
            cov[j, p] = value
    for a, side_a in enumerate(sides, start=n_power):
        for b, side_b in enumerate(sides, start=n_power):
            if b < a:
                continue
            if side_a == side_b:
                e_product = moments.log_square(side_a)
            else:
                e_product = moments.log_cross()
            value = e_product - mean[a] * mean[b]
            cov[a, b] = value
            cov[b, a] = value
    return mean, empirical, _project_psd(cov, 1e-12)


def _probe_nuisance_covariance(
    moments, model_power, probe_orders, nuisance_orders, sides, max_order, /
):
    """Return covariance between omitted powers and fitted nuisance statistics.

    Parameters
    ----------
    moments : _ModelMoments
        Current model-moment service.
    model_power : numpy.ndarray
        Cached model power moments.
    probe_orders : tuple of int
        Omitted power-statistic orders.
    nuisance_orders : tuple of int
        Fitted polynomial-statistic orders.
    sides : tuple of str
        Enabled canonical boundary-log sides.
    max_order : int
        Highest generalized power/log order required.

    Returns
    -------
    numpy.ndarray
        Omitted-by-nuisance covariance block.
    """
    n_u = len(probe_orders)
    n_t = len(nuisance_orders) + len(sides)
    out = np.empty((n_u, n_t), dtype=np.float64)
    if nuisance_orders:
        out[:, : len(nuisance_orders)] = _power_covariance_block(
            model_power, probe_orders, nuisance_orders
        )
    for row, order in enumerate(probe_orders):
        mean_u = model_power[order]
        for col, side in enumerate(sides, start=len(nuisance_orders)):
            mean_log_basis = -moments.log_power(side, 0)[0]
            e_product = -moments.log_power(side, max_order)[order]
            out[row, col] = e_product - mean_u * mean_log_basis
    return out


def _power_covariance_block(moments, rows, cols, /):
    """Return one covariance block for selected power orders.

    Parameters
    ----------
    moments : numpy.ndarray
        Power moments through every required summed order.
    rows, cols : iterable of int
        Power orders indexing covariance rows and columns.

    Returns
    -------
    numpy.ndarray
        Requested covariance block.
    """
    r = np.asarray(rows, dtype=int)
    c = np.asarray(cols, dtype=int)
    second = moments[r[:, None] + c[None, :]]
    return second - moments[r][:, None] * moments[c][None, :]


def _psd_pinv(matrix, rtol, /):
    """Return a symmetric PSD pseudoinverse and numerical rank.

    Parameters
    ----------
    matrix : array_like
        Symmetric positive-semidefinite matrix up to numerical roundoff.
    rtol : float
        Relative eigenvalue rank tolerance.

    Returns
    -------
    pinv : numpy.ndarray
        Rank-aware symmetric pseudoinverse.
    rank : int
        Numerical rank.
    """
    a = _project_psd(matrix, rtol)
    if a.size == 0:
        return a.copy(), 0
    values, vectors = np.linalg.eigh(a)
    vmax = float(np.max(values)) if values.size else 0.0
    cutoff = max(float(rtol) * vmax, np.finfo(np.float64).eps)
    keep = values > cutoff
    if not np.any(keep):
        return np.zeros_like(a), 0
    inv = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    return inv, int(np.count_nonzero(keep))


def _project_psd(matrix, rtol, /):
    """Symmetrize a covariance matrix and clip roundoff-negative eigenvalues.

    Parameters
    ----------
    matrix : array_like
        Candidate symmetric covariance/information matrix.
    rtol : float
        Relative tolerance distinguishing roundoff-negative eigenvalues.

    Returns
    -------
    numpy.ndarray
        Symmetric positive-semidefinite projection.
    """
    a = np.asarray(matrix, dtype=np.float64)
    if a.size == 0:
        return a.reshape((0, 0)).copy()
    a = 0.5 * (a + a.T)
    values, vectors = np.linalg.eigh(a)
    vmax = max(float(np.max(np.abs(values))), 1.0)
    floor = -float(rtol) * vmax
    if np.min(values) < floor:
        # A covariance becoming materially indefinite indicates accumulated
        # quadrature/cancellation error; clipping is still safer than allowing
        # an invalid score metric to propagate.
        values = np.maximum(values, 0.0)
    else:
        values = np.maximum(values, 0.0)
    return (vectors * values) @ vectors.T


@dataclass(frozen=True)
class _IntervalDegreeDiagnostic:
    """Omitted-statistic information diagnostic for finite interval data.

    Parameters
    ----------
    fitted_degree : int
        Current requested polynomial degree.
    probe_orders : tuple of int
        Omitted power orders examined.
    residual : numpy.ndarray
        Exact interval-conditional-minus-unconditional model mean residuals.
    efficient_residual : numpy.ndarray
        Residual after nuisance-score projection.
    observed_information : numpy.ndarray
        Schur-complement observed information for omitted powers.
    standardized_residual : numpy.ndarray
        Efficient residuals in observed-information sampling units.
    score : float
        Rank-aware score statistic using Kish effective sample size.
    rank : int
        Numerical rank of the omitted observed-information block.
    p_value : float
        Chi-square reference upper-tail probability.
    should_expand : bool
        Whether the current policy recommends more polynomial capacity.
    stopped_for_reliability : bool
        True when censoring leaves no numerically resolvable omitted direction.
    """

    fitted_degree: int
    probe_orders: tuple[int, ...]
    residual: np.ndarray
    efficient_residual: np.ndarray
    observed_information: np.ndarray
    standardized_residual: np.ndarray
    score: float
    rank: int
    p_value: float
    should_expand: bool
    stopped_for_reliability: bool


def _interval_omitted_statistic_diagnostic(fit, probe_orders, /, *, config=None):
    """Diagnose omitted natural power statistics for one finite interval fit.

    The natural-coordinate observed information uses the missing-information
    identity: unconditional model covariance minus the weighted conditional
    covariance remaining inside each censoring interval.  Finite rows use local
    Gauss--Legendre reductions and infinite rows use the same adaptive tail
    reductions as the exact likelihood.  Thus broad or whole-support censoring
    automatically loses information/rank rather than being treated like precise
    point observations.

    Parameters
    ----------
    fit : _NaturalDegreeDiagnosticFit
        Converged interval fit.
    probe_orders : iterable of int
        Strictly increasing omitted power orders above the fitted degree.
    config : _DegreeSelectionConfig or None, optional
        Rank tolerance and score threshold policy.

    Returns
    -------
    _IntervalDegreeDiagnostic
        Exact interval omitted-score diagnostic.
    """

    cfg = _DegreeSelectionConfig() if config is None else config
    orders = tuple(int(k) for k in probe_orders)
    if not orders or tuple(sorted(set(orders))) != orders:
        raise ValueError("probe_orders must be a nonempty strictly increasing sequence")
    degree = int(fit.spec.requested_poly_degree)
    if orders[0] <= degree:
        raise ValueError("probe orders must lie above the fitted degree")
    max_order = int(orders[-1])
    nuisance_orders = tuple(range(1, degree + 1))
    sides = _enabled_boundary_sides(fit.spec)
    all_power_orders = nuisance_orders + orders
    n_power = len(all_power_orders)
    n_stats = n_power + len(sides)

    state = fit.state
    moments = _ModelMoments(state)
    model_power = moments.power(2 * max_order)
    model_mean = np.empty(n_stats, dtype=np.float64)
    if n_power:
        model_mean[:n_power] = model_power[np.asarray(all_power_orders, dtype=int)]
    for j, side in enumerate(sides, start=n_power):
        model_mean[j] = -moments.log_power(side, 0)[0]

    model_cov = np.empty((n_stats, n_stats), dtype=np.float64)
    if n_power:
        model_cov[:n_power, :n_power] = _power_covariance_block(
            model_power, all_power_orders, all_power_orders
        )
    for p, order in enumerate(all_power_orders):
        for j, side in enumerate(sides, start=n_power):
            e_product = -moments.log_power(side, max_order)[order]
            value = e_product - model_mean[p] * model_mean[j]
            model_cov[p, j] = value
            model_cov[j, p] = value
    for a, side_a in enumerate(sides, start=n_power):
        for b, side_b in enumerate(sides, start=n_power):
            if b < a:
                continue
            e_product = (
                moments.log_square(side_a) if side_a == side_b else moments.log_cross()
            )
            value = e_product - model_mean[a] * model_mean[b]
            model_cov[a, b] = value
            model_cov[b, a] = value
    model_cov = _project_psd(model_cov, cfg.covariance_rtol)

    support = tuple(map(float, fit.spec.support))
    observations = fit.observations
    row_mean = np.empty((observations.n_unique, n_stats), dtype=np.float64)
    row_cov = np.empty((observations.n_unique, n_stats, n_stats), dtype=np.float64)

    finite_idx = np.flatnonzero(observations.finite_rows)
    if finite_idx.size:
        finite_intervals = observations.intervals[finite_idx]
        plan = _build_finite_interval_quadrature(finite_intervals, state.mode)

        point_global_rows = finite_idx[plan.point_limit]

        def _point_q(z_values):
            """Potential at exact-point rows, keeping sub-ulp edge distances."""
            out = np.empty(z_values.size, dtype=np.float64)
            for j, (z, r) in enumerate(zip(z_values, point_global_rows, strict=True)):
                out[j] = _point_q_with_boundary_distances(
                    state,
                    z,
                    observations.point_lower_distance[r],
                    observations.point_upper_distance[r],
                )
            return out

        nodes, log_kernel, point_mid, log_integrals = _finite_interval_log_kernel(
            state, support, plan, _point_q
        )
        node_values = np.empty(nodes.shape + (n_stats,), dtype=np.float64)
        ordinary = ~plan.point_limit
        if np.any(ordinary):
            node_values[ordinary] = _natural_statistic_values(
                nodes[ordinary], all_power_orders, sides, support
            )
        point_values = None
        if point_mid.size:
            point_values = _point_natural_statistic_values(
                point_mid,
                all_power_orders,
                sides,
                support,
                observations.point_lower_distance[point_global_rows],
                observations.point_upper_distance[point_global_rows],
            )
            node_values[plan.point_limit] = point_values[:, None, :]
        finite_mean, finite_cov = plan.conditional_mean_covariance(
            log_kernel,
            node_values,
            point_values=point_values,
            log_integrals=log_integrals,
        )
        row_mean[finite_idx] = finite_mean
        row_cov[finite_idx] = finite_cov

    infinite_idx = np.flatnonzero(~observations.finite_rows)
    if infinite_idx.size:
        infinite_rows = np.ascontiguousarray(
            observations.intervals[infinite_idx], dtype=np.float64
        )
        whole_support = (infinite_rows[:, 0] == float(support[0])) & (
            infinite_rows[:, 1] == float(support[1])
        )
        if np.any(whole_support):
            whole_idx = infinite_idx[whole_support]
            row_mean[whole_idx] = model_mean
            row_cov[whole_idx] = model_cov
        reduce_mask = ~whole_support
        if np.any(reduce_mask):
            reduce_idx = infinite_idx[reduce_mask]
            adaptive_reducer = _prepare_statistic_interval_reducer(
                fit.state, all_power_orders, sides
            )
            _, batch_mean, batch_cov, _ = adaptive_reducer.reduce_many(
                np.ascontiguousarray(infinite_rows[reduce_mask], dtype=np.float64)
            )
            row_mean[reduce_idx] = np.asarray(batch_mean, dtype=np.float64)
            row_cov[reduce_idx] = np.asarray(batch_cov, dtype=np.float64)

    observed_mean = np.einsum("r,ri->i", observations.weights, row_mean, optimize=True)
    missing_cov = np.einsum("r,rij->ij", observations.weights, row_cov, optimize=True)
    information = 0.5 * ((model_cov - missing_cov) + (model_cov - missing_cov).T)

    # The statistic order is powers first then logs; nuisance logs must be moved
    # next to nuisance powers before taking the Schur complement.
    nuisance_idx = list(range(len(nuisance_orders))) + list(range(n_power, n_stats))
    probe_idx = list(range(len(nuisance_orders), n_power))
    residual_all = observed_mean - model_mean
    residual = residual_all[probe_idx]
    sigma_uu = information[np.ix_(probe_idx, probe_idx)]
    if nuisance_idx:
        sigma_tt = information[np.ix_(nuisance_idx, nuisance_idx)]
        sigma_ut = information[np.ix_(probe_idx, nuisance_idx)]
        tt_pinv, _ = _psd_pinv(sigma_tt, cfg.covariance_rtol)
        projection = sigma_ut @ tt_pinv
        efficient = residual - projection @ residual_all[nuisance_idx]
        efficient_info = sigma_uu - projection @ sigma_ut.T
    else:
        efficient = residual.copy()
        efficient_info = sigma_uu.copy()
    efficient_info = _project_psd(efficient_info, cfg.covariance_rtol)

    info_pinv, rank = _psd_pinv(efficient_info, cfg.covariance_rtol)
    stopped = rank == 0
    score = 0.0
    p_value = 1.0
    should_expand = False
    if rank > 0:
        score = float(
            fit.observations.effective_n * (efficient @ info_pinv @ efficient)
        )
        p_value = float(chi2.sf(max(score, 0.0), rank))
        should_expand = bool(p_value < float(cfg.alpha))

    diag = np.maximum(np.diag(efficient_info), 0.0)
    se = np.sqrt(diag / max(float(fit.observations.effective_n), 1.0))
    standardized = np.divide(
        efficient, se, out=np.zeros_like(efficient), where=se > 0.0
    )
    return _IntervalDegreeDiagnostic(
        fitted_degree=degree,
        probe_orders=orders,
        residual=np.asarray(residual, dtype=np.float64),
        efficient_residual=np.asarray(efficient, dtype=np.float64),
        observed_information=np.asarray(efficient_info, dtype=np.float64),
        standardized_residual=np.asarray(standardized, dtype=np.float64),
        score=float(score),
        rank=int(rank),
        p_value=float(p_value),
        should_expand=bool(should_expand),
        stopped_for_reliability=bool(stopped),
    )


def _point_natural_statistic_values(
    points, power_orders, sides, support, lower_distance, upper_distance, /
):
    """Evaluate exact-point natural statistics with preserved edge distances.

    Canonicalization can round a physical point onto a finite support endpoint
    even when the original point remains a positive sub-ulp distance inside the
    support.  Power statistics use the canonical coordinate, while enabled
    boundary-log statistics use the preserved physical distance when available.

    Parameters
    ----------
    points : array_like
        Canonical exact-point coordinates.
    power_orders : iterable of int
        Power statistics to evaluate.
    sides : iterable of {"lower", "upper"}
        Enabled fixed boundary-log statistics.
    support : tuple of (float, float)
        Canonical support endpoints.
    lower_distance, upper_distance : array_like
        Preserved exact-point distances to the corresponding finite boundary.

    Returns
    -------
    numpy.ndarray
        Statistic values with one trailing statistic axis.
    """
    z = np.asarray(points, dtype=np.float64)
    lower_distance = np.asarray(lower_distance, dtype=np.float64)
    upper_distance = np.asarray(upper_distance, dtype=np.float64)
    if z.shape != lower_distance.shape or z.shape != upper_distance.shape:
        raise ValueError("point boundary distances must match exact-point coordinates")

    out = _natural_statistic_values(z, power_orders, (), support)
    if not sides:
        return out

    shape = z.shape + (len(power_orders) + len(sides),)
    values = np.empty(shape, dtype=np.float64)
    if len(power_orders):
        values[..., : len(power_orders)] = out

    lower, upper = map(float, support)
    for j, side in enumerate(sides, start=len(power_orders)):
        if side == _LOWER:
            distance = np.where(np.isfinite(lower_distance), lower_distance, z - lower)
        else:
            distance = np.where(np.isfinite(upper_distance), upper_distance, upper - z)
        values[..., j] = -np.log(distance)
    return values


def _natural_statistic_values(points, power_orders, sides, support, /):
    """Evaluate selected natural statistics on scalar/array coordinates.

    Parameters
    ----------
    points : array_like
        Canonical coordinates.
    power_orders : iterable of int
        Power statistics to evaluate.
    sides : iterable of {"lower", "upper"}
        Enabled fixed boundary-log statistics.
    support : tuple of (float, float)
        Canonical support endpoints.

    Returns
    -------
    numpy.ndarray
        Statistic values with one trailing statistic axis.
    """
    z = np.asarray(points, dtype=np.float64)
    out = np.empty(z.shape + (len(power_orders) + len(sides),), dtype=np.float64)
    for j, order in enumerate(power_orders):
        out[..., j] = z ** int(order)
    lower, upper = map(float, support)
    for j, side in enumerate(sides, start=len(power_orders)):
        if side == _LOWER:
            out[..., j] = -np.log(z - lower)
        else:
            out[..., j] = -np.log(upper - z)
    return out
