"""Observed-mixture omitted-power diagnostics and shared degree growth.

Probes are zero-coefficient degree lifts, not independent component fits.
Every score conditions on all fitted private coefficients, the free shared
physical amplitudes, and the mixture logits.  Indefinite observed information
is reported as unresolved rather than reflected into artificial evidence.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import chi2

from .._defaults import AUTO_POLY_DEGREE_MAX, AUTO_POLY_DEGREE_MIN
from .boundary import _effective_n
from .degree import (
    _DegreeDiagnostic,
    _interval_omitted_statistic_diagnostic,
    _omitted_statistic_diagnostic,
    _probe_orders_for_degree,
)
from .inputs import _admissible_degrees
from .natural_objective import _degree_diagnostic_fit


@dataclass(frozen=True)
class _MixtureDegreeDiagnostic(_DegreeDiagnostic):
    """Component probe result in shared observed-mixture information geometry.

    Parameters
    ----------
    component_index : int
        Component whose omitted power block was probed.
    information_status : str
        Resolution or identifiability status of the observed information solve.
    """

    component_index: int
    information_status: str


def _strict_information_pinv(matrix, rtol):
    """Equilibrated rank solve without clipping materially negative curvature.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square observed-information block in joint natural coordinates.
    rtol : float
        Shared relative tolerance for numerical rank and negative curvature.

    Returns
    -------
    tuple
        Pseudoinverse, numerical rank and explicit information status.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    if not matrix.size:
        return matrix.copy(), 0, "resolved"
    if not np.all(np.isfinite(matrix)):
        return np.zeros_like(matrix), 0, "nonfinite_information"
    diagonal = np.diag(matrix)
    floor = max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny) * rtol
    if np.any(diagonal < -floor):
        return np.zeros_like(matrix), 0, "indefinite_information"
    scale = np.sqrt(np.maximum(np.abs(diagonal), np.finfo(float).tiny))
    normalized = matrix / scale[:, None] / scale[None, :]
    values, vectors = np.linalg.eigh(normalized)
    cutoff = rtol * max(float(np.max(np.abs(values))), 1.0)
    if np.any(values < -cutoff):
        return np.zeros_like(matrix), 0, "indefinite_information"
    keep = values > cutoff
    if not np.any(keep):
        return np.zeros_like(matrix), 0, "unidentified_information"
    inverse = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    return inverse / scale[:, None] / scale[None, :], int(np.sum(keep)), "resolved"


def _joint_omitted_statistic_diagnostic(
    *,
    gradient,
    observed_information,
    row_scores,
    weights,
    nuisance_indices,
    probe_indices,
    fitted_degree,
    probe_orders,
    component_index,
    config,
    nuisance_solution=None,
):
    """Efficient score and original-row reliability in joint coordinates.

    Parameters
    ----------
    gradient : numpy.ndarray
        Joint objective gradient at the zero-coefficient degree lift.
    observed_information : numpy.ndarray
        Joint observed-information matrix.
    row_scores : numpy.ndarray
        Original-row joint scores, before duplicate compression.
    weights : numpy.ndarray
        Canonical normalized original-row reliability weights.
    nuisance_indices, probe_indices : numpy.ndarray
        Joint coordinates for fitted nuisance parameters and omitted powers.
    fitted_degree : int
        Current requested degree of the probed component.
    probe_orders : tuple of int
        Omitted power orders represented by the probe coordinates.
    component_index : int
        Probed component's index in solver order.
    config : _DegreeSelectionConfig
        Explicit policy shared by this fit's probes and growth steps.
    nuisance_solution : tuple or None, optional
        Previously computed nuisance inverse, rank and information status;
        ``None`` requests the same solve locally.
    """
    cfg = config
    w = weights
    effective_n = _effective_n(len(w), w)
    gradient = np.asarray(gradient, dtype=np.float64)
    information = np.asarray(observed_information, dtype=np.float64)
    row_scores = np.asarray(row_scores, dtype=np.float64)
    nuisance = np.asarray(nuisance_indices, dtype=int)
    probe = np.asarray(probe_indices, dtype=int)
    residual = gradient[probe]
    efficient = residual.copy()
    conditional = information[np.ix_(probe, probe)].copy()
    efficient_rows = row_scores[:, probe].copy()
    status = "resolved"
    if nuisance.size:
        nuisance_information = information[np.ix_(nuisance, nuisance)]
        inverse, _, status = (
            _strict_information_pinv(nuisance_information, cfg.covariance_rtol)
            if nuisance_solution is None
            else nuisance_solution
        )
        cross = information[np.ix_(probe, nuisance)]
        projection = cross @ inverse
        if status == "resolved":
            # A cross block outside the nuisance range cannot be profiled by a
            # pseudoinverse: the apparent efficient score is not identified.
            remainder = cross - projection @ nuisance_information
            denominator = np.maximum(
                np.linalg.norm(cross, axis=1), np.finfo(float).tiny
            )
            if np.any(
                np.linalg.norm(remainder, axis=1)
                > np.sqrt(cfg.covariance_rtol) * denominator
            ):
                status = "unresolved_nuisance_range"
        efficient -= projection @ gradient[nuisance]
        conditional -= projection @ cross.T
        efficient_rows -= row_scores[:, nuisance] @ projection.T
    conditional = 0.5 * (conditional + conditional.T)
    centered = efficient_rows - w @ efficient_rows
    empirical_variance = np.einsum("r,ri,ri->i", w, centered, centered)
    model_variance = np.diag(conditional)
    with np.errstate(over="ignore"):
        variance_ratio = empirical_variance / np.maximum(
            model_variance, np.finfo(float).tiny
        )
    inflation = np.clip(
        np.maximum(1.0, variance_ratio), 1.0, cfg.max_variance_inflation
    )
    inflation_scale = np.sqrt(inflation)
    test_covariance = conditional * inflation_scale[:, None] * inflation_scale[None, :]
    # Keep original rows here even when the interval likelihood is deduplicated.
    contributions = w[:, None] * np.abs(efficient_rows)
    top = np.max(contributions, axis=0)
    scaled = np.divide(
        contributions, top, out=np.zeros_like(contributions), where=top > 0
    )
    squares = np.sum(scaled * scaled, axis=0)
    participation = np.divide(
        np.sum(scaled, axis=0) ** 2,
        squares,
        out=np.zeros_like(squares),
        where=squares > 0,
    )
    reliable = np.isfinite(participation) & (participation >= cfg.min_participation)
    score, rank, p_value = 0.0, 0, 1.0
    if status == "resolved":
        _, _, status = _strict_information_pinv(conditional, cfg.covariance_rtol)
    if status == "resolved" and np.any(reliable):
        inverse, rank, status = _strict_information_pinv(
            test_covariance[np.ix_(reliable, reliable)], cfg.covariance_rtol
        )
        if status == "resolved" and rank:
            values = efficient[reliable]
            covariance = test_covariance[np.ix_(reliable, reliable)]
            unresolved = values - covariance @ inverse @ values
            if np.linalg.norm(unresolved) > np.sqrt(cfg.covariance_rtol) * max(
                np.linalg.norm(values), np.finfo(float).tiny
            ):
                status = "unresolved_probe_range"
            else:
                score = max(0.0, float(effective_n * (values @ inverse @ values)))
                p_value = float(chi2.sf(score, rank))
    if not np.any(reliable) and status == "resolved":
        status = "insufficient_participation"
    se = np.sqrt(np.maximum(np.diag(test_covariance), 0.0) / max(effective_n, 1.0))
    standardized = np.divide(efficient, se, out=np.zeros_like(efficient), where=se > 0)
    return _MixtureDegreeDiagnostic(
        fitted_degree=int(fitted_degree),
        probe_orders=tuple(probe_orders),
        reliable_mask=reliable,
        participation=participation,
        residual=residual,
        efficient_residual=efficient,
        conditional_covariance=conditional,
        test_covariance=test_covariance,
        variance_inflation=inflation,
        standardized_residual=standardized,
        score=score,
        rank=rank,
        p_value=p_value,
        should_expand=status == "resolved" and rank > 0 and p_value < cfg.alpha,
        stopped_for_reliability=status != "resolved" or not np.any(reliable),
        component_index=int(component_index),
        information_status=status,
    )


def _mixture_probe_geometry(fit, rows, observation_weights, degrees):
    """Evaluate all zero-power lifts in one shared observed-mixture geometry.

    Parameters
    ----------
    fit : _NaturalMixtureFit
        Optimized shared mixture supplying the fixed fitting coordinates.
    rows : numpy.ndarray
        Canonical point or interval observation rows.
    observation_weights : numpy.ndarray
        Normalized original-row weights.
    degrees : tuple of int
        Derived probe degrees, one per component, at least their fitted degrees.

    Returns
    -------
    tuple
        Joint evaluation, original-row scores, nuisance indices and component
        probe-index blocks.
    """
    # Mixture fitting invokes degree growth; defer the reciprocal fit dependency.
    from .natural_mixture import _CompiledJointMixture, _degree_probe_problems

    problems, layouts, params = _degree_probe_problems(
        fit, rows, observation_weights, degrees
    )
    objective = _CompiledJointMixture(problems, layouts, observation_weights)
    point = objective.join(params, np.log(fit.weights))
    evaluation, posterior, centered = objective.evaluate(point, rows=True)
    mapping = objective.natural_map
    expanded = np.column_stack(
        [
            *(posterior[:, k, None] * values for k, values in enumerate(centered)),
            fit.weights[:-1] - posterior[:, :-1],
        ]
    )
    row_scores = expanded @ mapping.joint_matrix(len(fit.components) - 1)
    nuisance = []
    for component, indices in zip(fit.components, mapping.local_indices, strict=True):
        nuisance.append(indices[component.layout.gamma_index])
        start = component.layout.curvature_slice.start
        nuisance.extend(
            indices[start : start + component.effective_curvature_degree + 1]
        )
    nuisance.extend(
        index
        for index in mapping.shared_parameter_indices
        if index is not None and point[index] > 0.0
    )
    nuisance.extend(range(mapping.n_params, objective.n_params))
    probes = tuple(
        indices[old.layout.curvature_slice.stop : new.curvature_slice.stop]
        for old, new, indices in zip(
            fit.components, layouts, mapping.local_indices, strict=True
        )
    )
    return evaluation, row_scores, np.asarray(nuisance, dtype=int), probes


def _shared_degree_diagnostics(fit, policies, rows, observation_weights, config):
    """Batch eligible component probes while keeping the represented density.

    Parameters
    ----------
    fit : _NaturalMixtureFit
        Current optimized shared mixture.
    policies : tuple of int or str
        Canonical requested policies; only ``"auto"`` entries can expand.
    rows : numpy.ndarray
        Canonical observations used by the owning fit.
    observation_weights : numpy.ndarray
        Normalized original-row reliability weights.
    config : _DegreeSelectionConfig
        Explicit policy reused by every component diagnostic.

    Returns
    -------
    tuple
        Component-index/diagnostic pairs for eligible omitted power blocks.
    """
    cfg = config
    current = tuple(int(c.spec.requested_poly_degree) for c in fit.components)
    support = fit.components[0].coordinate.physical_support
    admissible = tuple(
        int(d)
        for d in _admissible_degrees(support, AUTO_POLY_DEGREE_MAX)
        if d >= AUTO_POLY_DEGREE_MIN
    )
    orders = tuple(
        (
            _probe_orders_for_degree(d, admissible[-1], block_size=cfg.probe_block_size)
            if policy == "auto" and d < admissible[-1]
            else ()
        )
        for d, policy in zip(current, policies, strict=True)
    )
    if not any(orders):
        return ()
    augmented = tuple(
        max(block) if block else degree
        for degree, block in zip(current, orders, strict=True)
    )
    if len(fit.components) == 1:
        # This shares the same deferred mixture/degree dependency as joint probes.
        from .natural_mixture import _degree_probe_problems

        problems, _, _ = _degree_probe_problems(fit, rows, observation_weights, current)
        problem = problems[0]
        problem.order = max(problem.order, 2 * max(orders[0]))
        objective = problem.objective(observation_weights)
        adapted = _degree_diagnostic_fit(objective, fit.components[0].solver_result)
        diagnostic = (
            _omitted_statistic_diagnostic
            if rows.shape[1] == 1
            else _interval_omitted_statistic_diagnostic
        )(adapted, orders[0], config=cfg)
        return ((0, diagnostic),)
    evaluation, row_scores, nuisance, probes = _mixture_probe_geometry(
        fit, rows, observation_weights, augmented
    )
    nuisance_solution = _strict_information_pinv(
        evaluation.observed_hessian[np.ix_(nuisance, nuisance)], cfg.covariance_rtol
    )
    return tuple(
        (
            k,
            _joint_omitted_statistic_diagnostic(
                gradient=evaluation.gradient,
                observed_information=evaluation.observed_hessian,
                row_scores=row_scores,
                weights=observation_weights,
                nuisance_indices=nuisance,
                probe_indices=probe,
                fitted_degree=current[k],
                probe_orders=orders[k],
                component_index=k,
                config=cfg,
                nuisance_solution=nuisance_solution,
            ),
        )
        for k, probe in enumerate(probes)
        if len(probe)
    )


def _fit_shared_degree_growth(
    initial_fit,
    policies,
    *,
    support,
    rows,
    observation_weights,
    refit,
    degree_config,
):
    """Grow one strongest supported private block, with shared warm refits.

    ``initial_fit`` is an optimized shared mixture with automatic components
    initialized at the existing minimum admissible degree.  ``refit(degrees,
    previous_fit)`` performs the fixed-degree shared fit, preserving numerical
    coordinates and using a genuine lifted warm start.  It does not repeat the
    multistart search.  Fixed policy entries never grow.

    Parameters
    ----------
    initial_fit : _NaturalMixtureFit
        Initial optimized shared mixture.
    policies : tuple of int or str
        Canonical fixed or automatic policy per component.
    support : tuple of float
        Validated physical support of the owning fit.
    rows : numpy.ndarray
        Canonical point or interval observation rows.
    observation_weights : numpy.ndarray
        Normalized original-row reliability weights.
    refit : callable
        Fixed-degree warm-refit callback accepting degrees and the previous fit.
    degree_config : _DegreeSelectionConfig
        One explicit policy reused throughout growth and component diagnostics.
    """
    cfg = degree_config
    admissible = tuple(
        int(d)
        for d in _admissible_degrees(support, AUTO_POLY_DEGREE_MAX)
        if d >= AUTO_POLY_DEGREE_MIN
    )
    current = initial_fit
    history = []
    while True:
        diagnostics = _shared_degree_diagnostics(
            current, policies, rows, observation_weights, cfg
        )
        degrees = tuple(int(c.spec.requested_poly_degree) for c in current.components)
        supported = [(index, item) for index, item in diagnostics if item.should_expand]
        history.append(
            {"degrees": degrees, "diagnostics": diagnostics, "expanded_component": None}
        )
        if not supported:
            break
        index, _ = min(
            supported, key=lambda item: (item[1].p_value, -item[1].score, item[0])
        )
        next_degree = next((d for d in admissible if d > degrees[index]), None)
        if next_degree is None:
            break
        new_degrees = list(degrees)
        new_degrees[index] = next_degree
        candidate = refit(tuple(new_degrees), current)
        if (
            candidate.log_likelihood + 1e-8 * (1 + abs(current.log_likelihood))
            < current.log_likelihood
        ):
            history[-1]["termination"] = "warm_refit_lost_likelihood"
            break
        history[-1]["expanded_component"] = index
        current = candidate
    return replace(current, degree_diagnostics=tuple(history))
