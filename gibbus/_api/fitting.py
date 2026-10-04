"""Fit-request orchestration behind the public :class:`~gibbus.Distribution` API.

This module owns the controller logic that translates one public ``Distribution.fit``
call into either a single-component fit or a multi-component EM fit.
It deliberately contains no persistent estimator state: callers prepare a
request, run it to obtain an immutable result record, then install that result
on the public estimator.

Keeping this policy here makes :mod:`gibbus._api.distribution` a thin public facade
without violating the package dependency graph: the controller may compose
``_Component``, EM, selection, and lower-level ``_fit`` helpers because it
lives in the API layer rather than importing API objects from ``_fit``.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._defaults import (
    AUTO_K_MAX,
    EM_MAX_ITER,
    EM_TOL,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
)
from .._fit.boundary import AUTO, _effective_n, _select_boundary_terms
from .._fit.inputs import (
    _boundary_policy,
    _normalize_mixture_fit_inputs,
    _normalize_univariate_fit_inputs,
    _resolve_endpoint_observations,
    _to_generator,
)
from .._fit.mixture import (
    _e_step,
    _e_step_intervals,
    _initial_responsibility_candidates,
    _interval_identifiability_diagnostic,
    _interval_initial_representatives,
    _propose_n_components,
    _sort_components_by_mode,
)
from .._fit.natural_mixture import (
    _degree_policies,
    _fit_natural_mixture,
    _NaturalComponent,
)
from .._postfit.fitted_state import _pack_natural_component, _pack_natural_fit
from .._postfit.mixture_inference import _mixture_fit_metadata, _single_fit_metadata
from .component import _Component, _run_natural_fit
from .errors import _PointMixtureEstimabilityError
from .selection import _SingleSelectionFit, select_n_components


@dataclass(frozen=True)
class _FitRequest:
    """All options needed to execute one public fit request.

    Parameters
    ----------
    samples : array_like
        Raw point or interval observations supplied by the caller.
    n_components : int or str
        Explicit component count or ``"auto"``.
    poly_degree : int, str, or None
        Global polynomial-degree request.
    support : tuple of (float, float) or None
        User-coordinate support.
    log_boundary_lower, log_boundary_upper : bool or None
        Optional endpoint-log basis policy.
    verbose : int
        Verbosity level.
    suppress_warnings : bool
        Whether selected numerical warnings are suppressed.
    init_from : object or None
        Public estimator warm-start source before preparation.
    sample_weights : array_like or None
        Optional observation weights.
    component_options : list of dict or None
        Per-component fitting overrides.
    em_max_iter : int or None
        Requested EM iteration limit.
    em_tol : float or None
        Requested EM convergence tolerance.
    rng : object
        Random-number source for initialization and selection.
    k_max : int or None
        Maximum component count considered during automatic selection.
    progressive : bool
        Whether explicit high-degree mixture fits use a degree ladder.
    auto_k_subsample : str, int, or bool
        Automatic-component-count scoring subsample policy.
    seed_components : list of _Component or None
        Prepared component warm starts, populated from ``init_from``.
    seed_weights : numpy.ndarray or None
        Prepared mixture weights matching ``seed_components``.
    """

    samples: ArrayLike
    n_components: int | str
    poly_degree: int | str | None
    support: tuple[float, float] | None
    log_boundary_lower: bool | None
    log_boundary_upper: bool | None
    verbose: int
    suppress_warnings: bool
    init_from: Any
    sample_weights: ArrayLike | None
    component_options: list[dict[str, Any]] | None
    em_max_iter: int | None
    em_tol: float | None
    rng: Any
    k_max: int | None
    progressive: bool
    auto_k_subsample: str | int | bool
    seed_components: list[_Component] | None = None
    seed_weights: np.ndarray | None = None


@dataclass(frozen=True)
class _FitResult:
    """Completed fit state ready for installation on a public estimator.

    Parameters
    ----------
    components : list of _Component
        Fitted components in canonical mixture order.
    weights : numpy.ndarray
        Normalized mixture weights matching ``components``.
    selection_diagnostics : dict or None
        Automatic component-count diagnostics, when selection ran.
    em_diagnostics : dict or None
        Diagnostics from the final EM fit, when applicable.
    fit_metadata : dict or None
        Portable model dimensions and shared-boundary inference.
    """

    components: list[_Component]
    weights: np.ndarray
    selection_diagnostics: dict[str, Any] | None = None
    em_diagnostics: dict[str, Any] | None = None
    fit_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class _MixtureContext:
    """Canonicalized shared inputs for one mixture fit.

    Parameters
    ----------
    samples_rk : numpy.ndarray
        Canonical point ``(R, 1)`` or interval ``(R, 2)`` observations.
    n_components : int or str
        Explicit component count or ``"auto"`` after input normalization.
    support : tuple of (float, float)
        Validated shared support.
    component_options : list of dict or None
        Validated per-component options.
    verbose : int
        Normalized verbosity level.
    suppress_warnings : bool
        Normalized warning-suppression policy.
    observation_weights : numpy.ndarray or None
        Normalized observation weights.
    generator : numpy.random.Generator
        Deterministic or caller-provided random stream for component-count
        selection and initialization.
    fit_generator : numpy.random.Generator
        Stream for the final multi-start fit, seeded from ``generator``
        before anything else draws from it, so the final fit does not depend
        on how much randomness the selection consumed (for instance whether
        it subsampled).
    samples_1d : numpy.ndarray
        Point observations or interval representatives used by initialization.
    n_columns : int
        Number of canonical sample columns.
    log_boundary_lower, log_boundary_upper : bool or str
        Resolved endpoint-log basis policy (``"auto"`` lets the data decide).
    em_max_iter : int
        Effective EM iteration limit.
    em_tol : float
        Effective EM convergence tolerance.
    """

    samples_rk: np.ndarray
    n_components: int | str
    support: tuple[float, float]
    component_options: list[dict[str, Any]] | None
    verbose: int
    suppress_warnings: bool
    observation_weights: np.ndarray | None
    generator: np.random.Generator
    fit_generator: np.random.Generator
    samples_1d: np.ndarray
    n_columns: int
    log_boundary_lower: bool | str
    log_boundary_upper: bool | str
    em_max_iter: int
    em_tol: float


@dataclass(frozen=True)
class _MixtureInitialization:
    """Initial responsibilities and policy resolved before final EM.

    Parameters
    ----------
    n_components : int
        Effective component count.
    responsibilities : numpy.ndarray or None
        Initial row responsibilities; ``None`` when auto-selection chose one
        component and the caller should use the single-component path.
    weights : numpy.ndarray or None
        Initial mixture weights matching ``responsibilities``.
    component_options : list of dict
        Effective per-component options.
    candidates : list or None
        Alternative explicit-K initializers to compare by converged likelihood.
    selection_diagnostics : dict or None
        Automatic-K diagnostic record.
    completed_fit : dict or None
        Private refinement payload, separate from exposed selection diagnostics.
    """

    n_components: int
    responsibilities: np.ndarray | None
    weights: np.ndarray | None
    component_options: list[dict[str, Any]]
    candidates: list | None
    selection_diagnostics: dict[str, Any] | None
    completed_fit: dict[str, Any] | None = None


def _as_integer(value, name, /):
    """Return *value* as an ``int``, rejecting non-integral input.

    Parameters
    ----------
    value : object
        Candidate count supplied by the caller.
    name : str
        Parameter name, used in the error message.

    Returns
    -------
    int
        The value as an integer.

    Raises
    ------
    ValueError
        If the value is a bool, is not numeric, or has a fractional part.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}.") from None
    if np.isnan(as_float) or as_float in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite integer, got {value!r}.")
    if as_float != int(as_float):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return int(as_float)


def _check_optional_fit_controls(request, /):
    """Validate the optional controls that silently changed a fit when wrong.

    ``n_components`` is checked by the caller.  These are the remaining
    knobs where an out-of-range value used to be accepted: a non-positive
    ``k_max`` capped automatic selection at one component without saying
    so, and negative EM controls were passed through to the solver.

    Parameters
    ----------
    request : _FitRequest
        Raw request as supplied by the caller.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``k_max``, ``em_max_iter``, or ``em_tol`` is out of range or is
        not of an admissible type.
    """
    if request.k_max is not None:
        k_max = _as_integer(request.k_max, "k_max")
        if k_max < 1:
            raise ValueError(f"k_max must be >= 1 or None, got {k_max}.")
    if request.em_max_iter is not None:
        em_max_iter = _as_integer(request.em_max_iter, "em_max_iter")
        if em_max_iter < 0:
            raise ValueError(f"em_max_iter must be >= 0 or None, got {em_max_iter}.")
    if request.em_tol is not None:
        try:
            em_tol = float(request.em_tol)
        except (TypeError, ValueError):
            raise ValueError(
                f"em_tol must be a positive float or None, got {request.em_tol!r}."
            ) from None
        if not em_tol > 0.0 or em_tol in (float("inf"),) or np.isnan(em_tol):
            raise ValueError(f"em_tol must be finite and > 0 or None, got {em_tol}.")


def _prepare_fit_request(model_type, request, /):
    """Resolve estimator warm starts and component-count policy.

    Parameters
    ----------
    model_type : type
        Public estimator type used for the runtime warm-start type check.  It
        is supplied by ``Distribution.fit`` to avoid importing ``Distribution`` here.
    request : _FitRequest
        Raw public fit request.

    Returns
    -------
    _FitRequest
        Prepared request with canonical component-count policy and any warm
        start unpacked into component/weight state.

    Raises
    ------
    TypeError
        If ``init_from`` is not an instance of ``model_type``.
    ValueError
        If the warm start is unfitted or structurally invalid, or if the
        requested component count is invalid.
    """
    n_components = request.n_components
    support = request.support
    log_boundary_lower = request.log_boundary_lower
    log_boundary_upper = request.log_boundary_upper
    seed_components = None
    seed_weights = None

    if request.init_from is not None:
        init_from = request.init_from
        if not isinstance(init_from, model_type):
            raise TypeError("init_from must be a fitted Distribution instance.")
        if not init_from.is_fitted:
            raise ValueError(
                "init_from must be a fitted Distribution (call .fit() first)."
            )

        seed_components = init_from.components
        seed_weights = init_from.weights
        n_components = init_from.n_components
        supports = np.asarray(
            [component.base.support for component in seed_components],
            dtype=np.float64,
        )
        support = (float(np.min(supports[:, 0])), float(np.max(supports[:, 1])))

        seed_data = seed_components[0].data
        allowed = np.asarray(seed_data["boundary_allowed"], dtype=bool).reshape(-1)
        if allowed.size != 2:
            raise ValueError("seed has invalid boundary_allowed field")
        if log_boundary_lower is None:
            log_boundary_lower = bool(allowed[0])
        if log_boundary_upper is None:
            log_boundary_upper = bool(allowed[1])

        if (
            request.component_options is not None
            and len(request.component_options) != n_components
        ):
            raise ValueError(
                f"component_options length ({len(request.component_options)}) "
                f"must match init_from.n_components ({n_components})."
            )
    else:
        is_auto = isinstance(n_components, str) and n_components.lower() == "auto"
        if not is_auto:
            try:
                n_components = _as_integer(n_components, "n_components")
            except (TypeError, ValueError):
                raise ValueError(
                    "n_components must be a positive integer or 'auto', "
                    f"got {n_components!r}."
                ) from None
            if n_components < 1:
                raise ValueError("n_components must be >= 1 or 'auto'.")
        else:
            n_components = "auto"

    _check_optional_fit_controls(request)

    return replace(
        request,
        n_components=n_components,
        support=support,
        log_boundary_lower=log_boundary_lower,
        log_boundary_upper=log_boundary_upper,
        init_from=None,
        seed_components=seed_components,
        seed_weights=seed_weights,
    )


def _run_fit_request(request, /):
    """Execute one prepared fit request without mutating a ``Distribution``.

    Parameters
    ----------
    request : _FitRequest
        Prepared fit request returned by :func:`_prepare_fit_request`.

    Returns
    -------
    _FitResult
        Completed component state and diagnostics.
    """
    if request.n_components != "auto" and request.n_components == 1:
        seed = request.seed_components[0] if request.seed_components else None
        return _run_single_fit(
            request,
            poly_degree=request.poly_degree,
            support=request.support,
            log_boundary_lower=request.log_boundary_lower,
            log_boundary_upper=request.log_boundary_upper,
            verbose=request.verbose,
            suppress_warnings=request.suppress_warnings,
            init_from=seed,
        )
    return _run_mixture_fit(request)


def _run_single_fit(
    request,
    /,
    *,
    poly_degree,
    support,
    log_boundary_lower,
    log_boundary_upper,
    verbose,
    suppress_warnings,
    init_from,
):
    """Fit one component and package it as a fit result.

    Parameters
    ----------
    request : _FitRequest
        Prepared public request supplying samples and observation weights.
    poly_degree : int, str, or None
        Effective polynomial degree for this component.
    support : tuple of (float, float) or None
        Effective user-coordinate support.
    log_boundary_lower, log_boundary_upper : bool or None
        Effective endpoint-log basis policy.
    verbose : int
        Effective verbosity level.
    suppress_warnings : bool
        Effective warning-suppression policy.
    init_from : _Component or None
        Optional component warm start.

    Returns
    -------
    _FitResult
        Single fitted component with unit weight.
    """
    comp = _Component._fit(
        request.samples,
        poly_degree=poly_degree,
        support=support,
        log_boundary_lower=log_boundary_lower,
        log_boundary_upper=log_boundary_upper,
        verbose=verbose,
        suppress_warnings=suppress_warnings,
        init_from=init_from,
        sample_weights=request.sample_weights,
    )
    return _FitResult(
        components=[comp],
        weights=np.array([1.0], dtype=np.float64),
        fit_metadata=_single_fit_metadata(comp.data),
    )


def _prepare_mixture_context(request, /):
    """Normalize shared mixture inputs and resolve fitting defaults.

    Parameters
    ----------
    request : _FitRequest
        Prepared public fit request.

    Returns
    -------
    _MixtureContext
        Canonicalized mixture inputs shared by selection, initialization, and
        final EM.
    """
    norm = _normalize_mixture_fit_inputs(
        request.samples,
        request.n_components,
        request.support,
        request.component_options,
        request.verbose,
        request.suppress_warnings,
        sample_weights=request.sample_weights,
    )
    samples_rk = norm["samples_rk"]
    support = tuple(map(float, norm["support"]))
    n_columns = int(samples_rk.shape[1])
    samples_1d = (
        samples_rk[:, 0]
        if n_columns == 1
        else _interval_initial_representatives(samples_rk, support)
    )
    log_boundary_lower = _boundary_policy(request.log_boundary_lower, support[0])
    log_boundary_upper = _boundary_policy(request.log_boundary_upper, support[1])
    if log_boundary_lower is True and not np.isfinite(support[0]):
        raise ValueError(
            "log_boundary_lower=True requires a finite lower support endpoint"
        )
    if log_boundary_upper is True and not np.isfinite(support[1]):
        raise ValueError(
            "log_boundary_upper=True requires a finite upper support endpoint"
        )
    log_boundary_lower, log_boundary_upper = _resolve_endpoint_observations(
        samples_rk,
        support,
        log_boundary_lower,
        log_boundary_upper,
        norm["weights"],
    )

    generator = _to_generator(0 if request.rng is None else request.rng)
    fit_generator = np.random.default_rng(int(generator.integers(0, 2**63 - 1)))
    return _MixtureContext(
        samples_rk=samples_rk,
        n_components=norm["n_components"],
        support=support,
        component_options=norm["component_options"],
        verbose=norm["verbose"],
        suppress_warnings=norm["suppress_warnings"],
        observation_weights=norm["weights"],
        generator=generator,
        fit_generator=fit_generator,
        samples_1d=samples_1d,
        n_columns=n_columns,
        log_boundary_lower=log_boundary_lower,
        log_boundary_upper=log_boundary_upper,
        em_max_iter=(
            EM_MAX_ITER if request.em_max_iter is None else request.em_max_iter
        ),
        em_tol=EM_TOL if request.em_tol is None else request.em_tol,
    )


def _initialize_mixture(request, context, /):
    """Resolve seeded, automatic-K, or explicit-K initialization.

    Parameters
    ----------
    request : _FitRequest
        Prepared public fit request.
    context : _MixtureContext
        Canonicalized mixture inputs.

    Returns
    -------
    _MixtureInitialization
        Effective component count, responsibilities, options, candidates, and
        any automatic-selection diagnostics.
    """
    is_seeded = request.seed_components is not None
    if is_seeded:
        n_components = int(context.n_components)
        component_options = context.component_options
        if component_options is None:
            component_options = [{} for _ in range(n_components)]
        if context.n_columns == 2:
            responsibilities, _ = _e_step_intervals(
                context.samples_rk,
                request.seed_components,
                request.seed_weights,
                obs_weights=context.observation_weights,
            )
        else:
            responsibilities, _ = _e_step(
                context.samples_1d,
                request.seed_components,
                request.seed_weights,
                "base",
                obs_weights=context.observation_weights,
            )
        return _MixtureInitialization(
            n_components=n_components,
            responsibilities=responsibilities,
            weights=request.seed_weights.copy(),
            component_options=component_options,
            candidates=None,
            selection_diagnostics=None,
        )

    if context.n_components == "auto":
        requested_k_max = (
            int(request.k_max) if request.k_max is not None else AUTO_K_MAX
        )
        with _maybe_suppress(
            context.suppress_warnings,
            SUPPRESSED_WARNINGS,
        ):
            k_modes, effective_k_max = _propose_n_components(
                context.samples_1d,
                requested_k_max,
                context.generator,
                verbose=context.verbose,
            )

        def refine_candidate(
            *,
            S,
            samples_1d,
            obs_w,
            n_components,
            responsibilities,
            rng,
            initial_fit,
        ):
            return _refine_component_count(
                request,
                context,
                S,
                samples_1d,
                obs_w,
                n_components,
                responsibilities,
                rng,
                initial_fit,
            )

        n_components, responsibilities, weights, diagnostics = select_n_components(
            S=context.samples_rk,
            samples_1d=context.samples_1d,
            supp=context.support,
            k_modes=k_modes,
            effective_k_max=effective_k_max,
            gen=context.generator,
            obs_w=context.observation_weights,
            verb=context.verbose,
            subsample=request.auto_k_subsample,
            lower_boundary=context.log_boundary_lower,
            upper_boundary=context.log_boundary_upper,
            degree_policy=(
                "auto" if request.poly_degree is None else request.poly_degree
            ),
            refine_candidate=refine_candidate,
        )
        completed_fit = diagnostics.pop("_selected_fit")
        return _MixtureInitialization(
            n_components=int(n_components),
            responsibilities=responsibilities,
            weights=weights,
            component_options=[{} for _ in range(int(n_components))],
            candidates=None,
            selection_diagnostics=diagnostics,
            completed_fit=completed_fit,
        )

    n_components = int(context.n_components)
    component_options = context.component_options
    with _maybe_suppress(
        context.suppress_warnings,
        SUPPRESSED_WARNINGS,
    ):
        candidates = _initial_responsibility_candidates(
            context.samples_1d,
            n_components,
            context.generator,
            weights=request.sample_weights,
        )
    _, responsibilities, weights = candidates[0]
    return _MixtureInitialization(
        n_components=n_components,
        responsibilities=responsibilities,
        weights=weights,
        component_options=component_options,
        candidates=candidates,
        selection_diagnostics=None,
    )


def _refine_component_count(
    request,
    context,
    rows,
    representatives,
    observation_weights,
    n_components,
    responsibilities,
    generator,
    initial_fit,
    /,
):
    """Refine a screening candidate using the actual final fitting policies."""
    degree = "auto" if request.poly_degree is None else request.poly_degree
    if n_components == 1:
        norm = _normalize_univariate_fit_inputs(
            _Component,
            rows,
            degree,
            context.support,
            context.log_boundary_lower,
            context.log_boundary_upper,
            0,
            context.suppress_warnings,
            None,
            observation_weights,
        )
        with _maybe_suppress(context.suppress_warnings, SUPPRESSED_WARNINGS):
            objective, result, effective_n, p_values = _run_natural_fit(norm)
        component = _NaturalComponent(
            coordinate=objective.spec.coordinate,
            spec=objective.spec,
            layout=objective.layout,
            z_data_bounds=objective.z_data_bounds,
            params=result.params,
            effective_curvature_degree=result.effective_curvature_degree,
            lower_amplitude_active=result.lower_amplitude_active,
            upper_amplitude_active=result.upper_amplitude_active,
            solver_result=result,
        )
        dimension = (
            2
            + int(result.effective_curvature_degree)
            + int(result.lower_amplitude_active)
            + int(result.upper_amplitude_active)
        )
        fitted = _SingleSelectionFit(
            objective=objective,
            result=result,
            components=(component,),
            weights=np.ones(1),
            log_likelihood=-float(result.objective_value),
            n_face_parameters=dimension,
        )
        return {
            "fit": fitted,
            "log_likelihood": fitted.log_likelihood,
            "n_face_parameters": dimension,
            "weights": fitted.weights,
            "components": fitted.components,
            "boundary_p_values": p_values,
            "status": str(result.status),
            "effective_n": float(effective_n),
            "degrees": (int(objective.spec.requested_poly_degree),),
        }

    candidate_context = replace(
        context,
        samples_rk=rows,
        samples_1d=representatives,
        observation_weights=observation_weights,
        n_components=int(n_components),
        fit_generator=generator,
        verbose=0,
    )
    fitted, p_values = _fit_mixture_with_boundary_policy(
        candidate_context,
        n_components,
        degree,
        responsibilities,
        (("direct", "raw"),),
        initial_fit=initial_fit,
    )
    return {
        "fit": fitted,
        "log_likelihood": float(fitted.log_likelihood),
        "n_face_parameters": int(fitted.n_face_parameters),
        "weights": fitted.weights,
        "components": fitted.components,
        "boundary_p_values": p_values,
        "status": str(fitted.status),
        "degrees": tuple(
            int(component.spec.requested_poly_degree) for component in fitted.components
        ),
    }


def _resolve_seeded_component_options(request, initialization, /):
    """Apply per-component degree precedence for a warm-started mixture.

    Parameters
    ----------
    request : _FitRequest
        Prepared request containing component seeds and the global degree.
    initialization : _MixtureInitialization
        Seeded mixture initialization with validated component options.

    Returns
    -------
    list of dict
        Independent per-component option dictionaries with ``poly_degree``
        resolved for every component.
    """
    resolved = [dict(opts) for opts in initialization.component_options]
    for index in range(initialization.n_components):
        if "poly_degree" in resolved[index]:
            continue
        if request.poly_degree is not None:
            resolved[index]["poly_degree"] = request.poly_degree
            continue
        seed_data = request.seed_components[index].data
        resolved[index]["poly_degree"] = int(seed_data["requested_poly_degree"])
    return resolved


def _natural_degree_policy(global_degree, component_options, /):
    """Resolve public global/per-component degree precedence for natural EM.

    Parameters
    ----------
    global_degree : int, str, or None
        Global polynomial-degree policy.
    component_options : sequence of Mapping
        Per-component options that may override ``poly_degree``.

    Returns
    -------
    int, str, or tuple
        Shared policy when all components agree, otherwise one policy per component.
    """
    default = "auto" if global_degree is None else global_degree
    policies = tuple(
        options.get("poly_degree", default) for options in component_options
    )
    first = policies[0]
    if all(value == first for value in policies):
        return first
    return policies


def _fit_mixture_with_boundary_policy(
    context,
    n_components,
    degree_policy,
    responsibilities,
    paths,
    /,
    *,
    initial_fit=None,
):
    """Fit the final mixture, deciding ``"auto"`` boundary terms from the data.

    Automatic sides are tested globally at locked degrees, using the fitted
    mixture as a numerical warm start. After removing a side, automatic
    degrees are reselected under the remaining allowed boundary basis.

    Parameters
    ----------
    context : _MixtureContext
        Shared mixture inputs.
    n_components : int
        Number of components.
    degree_policy : int, str or sequence
        Degree policy of the multi-start.
    responsibilities : numpy.ndarray or None
        Explicit initial responsibilities, or ``None`` for the candidates.
    paths : tuple
        Multi-start paths.
    initial_fit : _NaturalMixtureFit or None, optional
        Shared numerical fit supplying fixed coordinates and a warm start.

    Returns
    -------
    fitted : _NaturalMixtureFit
        Final mixture.
    p_values : tuple of float
        Boundary-term test p-values per physical side (``nan`` where no test
        ran).
    """
    rows = context.samples_rk
    options = {"max_em_steps": context.em_max_iter, "em_tolerance": context.em_tol}
    reduced_fits = {}

    def fit(lo, up):
        return _fit_natural_mixture(
            context.support,
            rows,
            n_components,
            degree_policy,
            lo,
            up,
            context.observation_weights,
            rng=context.fit_generator,
            responsibilities=responsibilities,
            paths=paths,
            initial_fit=initial_fit,
            **options,
        )

    lower, upper = context.log_boundary_lower, context.log_boundary_upper
    if lower != AUTO and upper != AUTO:
        return fit(bool(lower), bool(upper)), (np.nan, np.nan)

    def fit_reduced(model, lo, up):
        locked = tuple(int(c.spec.requested_poly_degree) for c in model.components)
        reduced = _fit_natural_mixture(
            context.support,
            rows,
            n_components,
            locked,
            lo,
            up,
            context.observation_weights,
            rng=context.fit_generator,
            responsibilities=model.responsibilities,
            paths=(("direct", "raw"),),
            initial_fit=model,
            **options,
        )
        reduced_fits[(bool(lo), bool(up))] = reduced
        return reduced

    def amplitude(model, side):
        component = model.components[0]
        index = (
            component.spec.physical_lower_a_index
            if side == "lower"
            else component.spec.physical_upper_a_index
        )
        return 0.0 if index is None else float(component.params[index])

    def refit(lo, up):
        previous = reduced_fits[(bool(lo), bool(up))]
        return _fit_natural_mixture(
            context.support,
            rows,
            n_components,
            degree_policy,
            lo,
            up,
            context.observation_weights,
            rng=context.fit_generator,
            responsibilities=previous.responsibilities,
            paths=(("direct", "raw"),),
            initial_fit=previous,
            **options,
        )

    automatic_degree = "auto" in _degree_policies(degree_policy, n_components)
    fitted, _, p_values = _select_boundary_terms(
        fit,
        lambda model: -float(model.log_likelihood),
        amplitude,
        lower,
        upper,
        _effective_n(rows.shape[0], context.observation_weights),
        fit_reduced=fit_reduced,
        refit=refit if automatic_degree else None,
    )
    return fitted, p_values


def _components_from_natural_mixture(fitted, observation_weights, boundary_p_values, /):
    """Finalize a natural mixture into public query-time component states.

    Parameters
    ----------
    fitted : _NaturalMixtureResult
        Completed natural mixture fit.
    observation_weights : numpy.ndarray or None
        Observation weights (``None`` for equal weights).
    boundary_p_values : tuple of float
        Automatic boundary-term test p-values per physical side.

    Returns
    -------
    components : list of _Component
        Finalized public components.
    weights : numpy.ndarray
        Mixture weights.
    diagnostics : dict
        In-process EM/polish diagnostic record.
    """
    r = np.asarray(fitted.responsibilities, dtype=np.float64)
    w = (
        np.ones(r.shape[0], dtype=np.float64)
        if observation_weights is None
        else np.asarray(observation_weights, dtype=np.float64)
    )
    components = [
        _Component(
            _pack_natural_component(
                component,
                effective_n=_effective_n(r.shape[0], w * r[:, k]),
                boundary_p_values=(np.nan, np.nan),
            )
        )
        for k, component in enumerate(fitted.components)
    ]
    em_diag = {
        "converged": str(fitted.status) in {"converged", "converged_approximately"},
        "termination_reason": str(fitted.status),
        "n_iterations": int(fitted.em_iterations),
        "polish_iterations": int(fitted.polish_iterations),
        "rounds": int(fitted.rounds),
        "last_relative_change": None,
        "n_failed_component_refits": 0,
        "final_log_likelihood": float(fitted.log_likelihood),
        "finalize_rewound": False,
        "decrease_bound": float(fitted.decrease_bound),
        "separator_certified": bool(fitted.separator_certified),
        "initialization": str(fitted.initialization),
        "degree_diagnostics": tuple(
            {
                "degrees": tuple(map(int, record["degrees"])),
                "diagnostics": tuple(
                    {"component": int(index), **asdict(diagnostic)}
                    for index, diagnostic in record["diagnostics"]
                ),
                "expanded_component": record["expanded_component"],
            }
            for record in fitted.degree_diagnostics
        ),
        "degree_component_order": "solver",
    }
    return components, np.asarray(fitted.weights, dtype=np.float64), em_diag


def _run_mixture_fit(request, /):
    """Execute automatic or explicit multi-component fitting policy.

    Parameters
    ----------
    request : _FitRequest
        Prepared public fit request.

    Returns
    -------
    _FitResult
        Completed single- or multi-component fit.  Automatic selection may
        return a single component while retaining its selection diagnostics.
    """
    context = _prepare_mixture_context(request)
    initialization = _initialize_mixture(request, context)
    completed = initialization.completed_fit
    if completed is not None and initialization.selection_diagnostics is None:
        raise RuntimeError("a completed selection fit must include its diagnostics")
    reuse_completed = bool(
        completed is not None
        and initialization.selection_diagnostics["reuse_selected_fit"]
    )

    if initialization.n_components == 1:
        if reuse_completed:
            selected = completed["fit"]
            state = _pack_natural_fit(
                selected.objective,
                selected.result,
                effective_n=completed["effective_n"],
                boundary_p_values=completed["boundary_p_values"],
            )
            result = _FitResult(
                components=[_Component(state)],
                weights=np.ones(1),
                fit_metadata=_single_fit_metadata(state),
            )
        else:
            result = _run_single_fit(
                request,
                poly_degree=(
                    "auto" if request.poly_degree is None else request.poly_degree
                ),
                support=context.support,
                log_boundary_lower=context.log_boundary_lower,
                log_boundary_upper=context.log_boundary_upper,
                verbose=context.verbose,
                suppress_warnings=context.suppress_warnings,
                init_from=None,
            )
        return replace(
            result,
            selection_diagnostics=initialization.selection_diagnostics,
        )

    component_options = initialization.component_options
    if request.seed_components is not None:
        component_options = _resolve_seeded_component_options(request, initialization)

    poly_degree = request.poly_degree
    if poly_degree is None:
        poly_degree = "auto"

    degree_policy = _natural_degree_policy(poly_degree, component_options)
    responsibilities = initialization.responsibilities
    if initialization.candidates is not None and request.seed_components is None:
        # Let the multi-start explore the complete initializer set rather
        # than restricting it to the first candidate.
        responsibilities = None
    paths = (
        (
            ("ladder", "raw"),
            ("ladder", "sharpened"),
            ("direct", "raw"),
            ("direct", "sharpened"),
        )
        if request.progressive
        else (("direct", "raw"), ("direct", "sharpened"))
    )
    try:
        if reuse_completed:
            fitted = completed["fit"]
            boundary_p_values = completed["boundary_p_values"]
        else:
            fitted, boundary_p_values = _fit_mixture_with_boundary_policy(
                context,
                initialization.n_components,
                degree_policy,
                responsibilities,
                paths,
            )
    except ValueError as exc:
        if "point-mixture component is not estimable" in str(exc):
            k = int(initialization.n_components)
            raise _PointMixtureEstimabilityError(
                f"{exc} No initialization of the requested {k}-component "
                "mixture produced an estimable fit, so the data do not support "
                f"n_components={k}; use n_components='auto' or fewer "
                "components."
            ) from exc
        raise
    components, weights, em_diagnostics = _components_from_natural_mixture(
        fitted,
        context.observation_weights,
        boundary_p_values,
    )
    if context.samples_rk.shape[1] == 2:
        ident = _interval_identifiability_diagnostic(
            context.samples_rk,
            components,
            fitted.log_likelihood,
            context.support,
            obs_weights=context.observation_weights,
            n_parameters=fitted.n_face_parameters,
        )
        if ident is not None:
            pattern = "overlapping" if ident["has_overlap"] else "endpoint-partition"
            raise RuntimeError(
                "interval-censored fit is non-identifiable at the observed "
                "resolution: the fitted likelihood reaches the nonparametric "
                f"interval-likelihood bound (gap={ident['gap']:.3g}) with "
                f"{ident['n_params']} free parameters but only "
                f"{ident['observable_dim']} independent probability coordinates "
                f"in the {pattern} censoring pattern. Component shapes within "
                "the censoring intervals are not determined by the data. Use "
                "finer censoring intervals, reduce model complexity, or fit "
                "n_components=1 if only the coarse distribution is needed."
            )
    solver_order = {id(component): index for index, component in enumerate(components)}
    components, weights = _sort_components_by_mode(components, weights)
    public_to_solver = tuple(solver_order[id(component)] for component in components)
    em_diagnostics["solver_to_public"] = tuple(
        public_to_solver.index(index) for index in range(len(components))
    )
    return _FitResult(
        components=components,
        weights=weights,
        selection_diagnostics=initialization.selection_diagnostics,
        em_diagnostics=em_diagnostics,
        fit_metadata=_mixture_fit_metadata(
            fitted,
            _effective_n(context.samples_rk.shape[0], context.observation_weights),
            boundary_p_values,
        ),
    )
