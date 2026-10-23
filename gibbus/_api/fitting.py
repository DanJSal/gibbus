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

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .._defaults import (
    EM_MAX_ITER,
    EM_TOL,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
)
from .._fit.basis_mixture import (
    _EMOptions,
    _fit_mixture_auto,
    _PointMixtureEstimabilityFailure,
    _select_components,
)
from .._fit.basis_selection import _fit_basis
from .._fit.boundary import _effective_n
from .._fit.conic_newton import _CONVERGED, _UNCERTIFIED
from .._fit.degree import _DegreeSelectionConfig
from .._fit.inputs import (
    _boundary_policy,
    _coerce_poly_degree,
    _is_poly_degree_admissible,
    _normalize_degree_policies,
    _normalize_mixture_fit_inputs,
    _resolve_endpoint_observations,
    _to_generator,
)
from .._fit.mixture import (
    _e_step,
    _e_step_intervals,
    _interval_identifiability_diagnostic,
    _interval_initial_representatives,
    _sort_components_by_mode,
)
from .._fit.tail_estimation import alternative_spec, resolve_auto_tails
from .._model.basis import _compile_basis
from .._model.basis_spec import (
    _EXPONENTIAL,
    _FULL,
    _POLYNOMIAL,
    BasisSpec,
    Exponential,
    Polynomial,
    SoftPower,
    TanhSpec,
)
from .._postfit.basis_state import _pack_basis_fit, _pack_selected
from .component import _Component
from .errors import _PointMixtureEstimabilityError


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
    sample_weight : array_like or None
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
    auto_k_subsample : str, int, or bool
        Automatic-component-count scoring subsample policy.
    seed_components : list of _Component or None
        Prepared component warm starts, populated from ``init_from``.
    seed_weights : numpy.ndarray or None
        Prepared mixture weights matching ``seed_components``.
    basis : BasisSpec or None
        Full G3 basis request (tails, spans) in place of ``poly_degree``,
        ``support`` and ``log_boundary_*``.
    seed_basis : BasisSpec or None
        The warm-start seed's basis family in public units (tails, span
        policy, tanh geometry), used when ``basis`` is not given.
    degree_config : _DegreeSelectionConfig
        Immutable diagnostic policy created once for this request and reused by
        automatic candidates, component selection, degree growth and warm refits.
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
    sample_weight: ArrayLike | None
    component_options: list[dict[str, Any]] | None
    em_max_iter: int | None
    em_tol: float | None
    rng: Any
    k_max: int | None
    auto_k_subsample: str | int | bool
    seed_components: list[_Component] | None = None
    seed_weights: np.ndarray | None = None
    basis: Any = None
    seed_basis: Any = None
    degree_config: _DegreeSelectionConfig = field(
        default_factory=_DegreeSelectionConfig
    )


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
    tail_parameter_diagnostics: dict[str, Any] | None = None


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
    degree_config : _DegreeSelectionConfig
        One immutable diagnostic policy reused by screening, growth and refits.
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
    degree_config: _DegreeSelectionConfig


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
        if request.basis is not None and tuple(
            map(float, request.basis.support)
        ) != tuple(support):
            raise ValueError(
                f"init_from fixes the support {support}; the basis support "
                f"{tuple(request.basis.support)} differs"
            )

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

    poly_degree = request.poly_degree
    if isinstance(poly_degree, str):
        if poly_degree.lower() != "auto":
            raise ValueError(
                "poly_degree must be an integer >= 2, 'auto', or None, "
                f"got {poly_degree!r}."
            )
        poly_degree = "auto"
    elif poly_degree is not None:
        poly_degree = _coerce_poly_degree(poly_degree)

    return replace(
        request,
        n_components=n_components,
        poly_degree=poly_degree,
        support=support,
        log_boundary_lower=log_boundary_lower,
        log_boundary_upper=log_boundary_upper,
        init_from=None,
        seed_components=seed_components,
        seed_weights=seed_weights,
        seed_basis=(
            None
            if seed_components is None or request.basis is not None
            else _seed_basis_spec(seed_components[0])
        ),
    )


def _seed_basis_spec(component, /):
    """The basis family of a warm-start seed, in its public units.

    The seed's canonical request lives in model units; the presentation
    ``x = shift + scale u`` maps tanh locations, steepnesses, exponential
    rates and soft-power smoothings into public units (a negative scale
    swaps the ends).  The span policy is ``"full"`` for a full-span seed,
    ``"auto"`` when the seed's span grew beyond the minimal one, else
    ``"minimal"``.  Support, degree and endpoint logs are set by the caller.

    Parameters
    ----------
    component : _Component
        First seed component (the family is shared by all components).
    """
    parent = component._basis.parent
    req = parent.request
    shift, scale = map(float, component._affine())
    size = abs(scale)

    def tail(t):
        if t is None:
            return None
        if t.kind == _POLYNOMIAL:
            return Polynomial(int(t.value))
        if t.kind == _EXPONENTIAL:
            return Exponential(float(t.value) / size)
        return SoftPower(
            float(t.value), "auto" if t.delta is None else float(t.delta) * size**2
        )

    def location(v):
        return "auto" if v is None else shift + scale * float(v)

    def steepness(v):
        return "auto" if v is None else float(v) / size

    tails = (tail(req.lower_tail), tail(req.upper_tail))
    tanh = (
        (location(req.lower_location), steepness(req.lower_steepness)),
        (location(req.upper_location), steepness(req.upper_steepness)),
    )
    if scale < 0.0:
        tails, tanh = tails[::-1], tanh[::-1]
    if req.expansion == _FULL:
        expansion = _FULL
    else:
        minimal = _compile_basis(req, parent.geometry)
        expansion = (
            "auto" if parent.n_coefficients > minimal.n_coefficients else "minimal"
        )
    return BasisSpec(
        lower_tail=tails[0],
        upper_tail=tails[1],
        independent_equal_rate_coefficients=req.independent,
        expansion=expansion,
        tanh=TanhSpec(
            lower_location=tanh[0][0],
            upper_location=tanh[1][0],
            lower_steepness=tanh[0][1],
            upper_steepness=tanh[1][1],
        ),
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
    return _run_basis_request(request)


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
        sample_weight=request.sample_weight,
        allow_odd_degree=request.basis is not None or request.seed_basis is not None,
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
        degree_config=request.degree_config,
    )


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
    tuple of int or str
        Canonical policy per component, including homogeneous mixtures.
    """
    default = "auto" if global_degree is None else global_degree
    policies = tuple(
        options.get("poly_degree", default) for options in component_options
    )
    return _normalize_degree_policies(policies, len(component_options))


# ----------------------------------------------------------------------
# G3 fit controller (G3-047): public requests on the basis fitters
# ----------------------------------------------------------------------


def _basis_spec_of(request, support, degree, lower, upper, /):
    """The fit's ``BasisSpec``: the caller's, or G1's arguments mapped onto one.

    G1's arguments describe G1's family exactly: ``poly_degree`` is the bulk
    degree (``"auto"`` by default, as G1), ``log_boundary_*`` the endpoint-log
    policies (``"auto"`` lets the data decide) and the span is minimal (bulk
    monomials and logs: no gated slots, which only a ``basis`` request with
    tails or an expansion policy adds).  A warm start without ``basis``
    keeps its seed's family (``_seed_basis_spec``) with the request's
    support, degree and logs.

    Parameters
    ----------
    request : _FitRequest
        Prepared public request.
    support : tuple of float
        Fit support.
    degree : int or "auto"
        Bulk degree.
    lower : bool or "auto"
        Lower endpoint-log policy.
    upper : bool or "auto"
        Upper endpoint-log policy.
    """

    if request.basis is not None:
        return replace(
            request.basis, bulk_degree=degree, log_lower=lower, log_upper=upper
        )
    if request.seed_basis is not None:
        return replace(
            request.seed_basis,
            support=support,
            bulk_degree=degree,
            log_lower=lower,
            log_upper=upper,
        )
    return BasisSpec(
        support=support,
        bulk_degree=degree,
        log_lower=lower,
        log_upper=upper,
        expansion="minimal",
    )


def _em_diagnostics(fit, /):
    """In-process EM record of a G3 mixture (G1 ``em_diagnostics`` keys).

    Parameters
    ----------
    fit : _MixtureFit
        Fitted G3 mixture.
    """
    status = str(fit.status)
    return {
        "converged": status in {"converged", "converged_approximately"},
        "termination_reason": status,
        "n_iterations": int(fit.em_iterations),
        "polish_iterations": int(fit.polish_iterations),
        "rounds": int(fit.rounds),
        "final_log_likelihood": float(fit.log_likelihood),
        "decrease_bound": (
            float("nan")
            if fit.result is None
            else float(fit.result.final_decrease_bound)
        ),
        "initialization": str(fit.initialization),
        "selection": tuple(
            {
                "kind": step.kind,
                "chosen": step.chosen,
                "lr_p_value": float(step.lr_p_value),
                "failure": step.failure,
                "diagnostics": tuple(
                    {"component": int(c), "label": d.label, "p_value": d.p_value}
                    for c, d in step.diagnostics
                ),
            }
            for step in fit.selection
        ),
    }


def _seed_responsibilities(context, seeds, weights, /):
    """Responsibilities of the seed mixture at the data (G1 warm E-step).

    Parameters
    ----------
    context : _MixtureContext
        Prepared fit context.
    seeds : sequence of Component
        Seed mixture components.
    weights : ndarray
        Seed mixture weights.
    """
    if context.n_columns == 2:
        resp, _ = _e_step_intervals(
            context.samples_rk, seeds, weights, obs_weights=context.observation_weights
        )
    else:
        resp, _ = _e_step(
            context.samples_1d,
            seeds,
            weights,
            "base",
            obs_weights=context.observation_weights,
        )
    return resp


def _check_identifiable(context, log_likelihood, n_parameters, /):
    """G1's interval identifiability rule on a mixture (fails loudly).

    ``log_likelihood`` is the mixture's mean interval log-likelihood, or a
    nested model's (a mixture contains its one-component model): reaching
    the nonparametric bound with more parameters than independent interval
    probabilities leaves the components undetermined.

    Parameters
    ----------
    context : _MixtureContext
        Prepared fit context.
    log_likelihood : float
        Mean interval log-likelihood of the model.
    n_parameters : int
        Free parameters of the mixture.
    """
    if context.samples_rk.shape[1] != 2:
        return
    ident = _interval_identifiability_diagnostic(
        context.samples_rk,
        None,
        float(log_likelihood),
        context.support,
        obs_weights=context.observation_weights,
        n_parameters=int(n_parameters),
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


def _run_basis_request(request, /):
    """Execute a prepared public request on the G3 fitters.

    Inputs are validated by G1's public rules (``_prepare_mixture_context``:
    sample layouts, support, weights, endpoint observations, component
    options); the request becomes a ``BasisSpec``; one component is the
    single fit, an explicit count ``_fit_mixture_auto`` (a warm start seeds
    its EM with the seed's responsibilities at the data), ``"auto"``
    ``_select_components``.  Results are packed G3 component states.

    Parameters
    ----------
    request : _FitRequest
        Prepared public request.
    """

    context = _prepare_mixture_context(request)
    seeds = request.seed_components
    k = context.n_components
    options = context.component_options
    degree = request.poly_degree
    if request.basis is not None:
        degree = request.basis.bulk_degree
        degrees = (
            tuple(opts.get("poly_degree", degree) for opts in options)
            if options is not None and any("poly_degree" in opts for opts in options)
            else None
        )
    elif seeds is not None and degree is None:
        degrees = tuple(
            opts.get("poly_degree", s._basis.model.request.bulk_degree)
            for opts, s in zip(options, seeds, strict=True)
        )
    else:
        degrees = (
            None
            if options is None or k == "auto"
            else _natural_degree_policy(degree, options)
        )
    global_degree = "auto" if degree is None else degree
    for policy in (
        ()
        if request.basis is not None or request.seed_basis is not None
        else (global_degree, *(() if degrees is None else degrees))
    ):
        if policy != "auto" and not _is_poly_degree_admissible(policy, context.support):
            raise ValueError(
                f"poly_degree={policy} is odd, which is inadmissible on "
                "full-infinite support (-inf, +inf). Use an even degree."
            )
    if degrees is not None and len(set(degrees)) == 1:
        global_degree, degrees = degrees[0], None
    spec = _basis_spec_of(
        request,
        context.support,
        "auto" if degrees is not None else global_degree,
        context.log_boundary_lower,
        context.log_boundary_upper,
    )
    weights = context.observation_weights
    samples = context.samples_rk
    spec, tail_proposals = resolve_auto_tails(spec, samples, weights)
    tail_diagnostics = {
        side: {
            "family": est.family,
            "estimated_parameter": float(est.primary),
            "backup_parameter": (
                None if est.alternative is None else float(est.alternative)
            ),
            "tail_n": est.n_tail,
            "fallback_used": False,
        }
        for side, (_, est, _) in tail_proposals.items()
    }
    n_eff = _effective_n(samples.shape[0], weights)
    em = _EMOptions(tolerance=context.em_tol, max_steps=context.em_max_iter)
    config = context.degree_config
    selection = None
    with _maybe_suppress(context.suppress_warnings, SUPPRESSED_WARNINGS):
        if k == 1:
            backup = alternative_spec(spec, tail_proposals) if tail_proposals else None
            original_error = None
            fit = None
            try:
                fit = _fit_basis(spec, samples, weights, config=config)
                if tail_proposals and fit.result.status not in _CONVERGED:
                    original_error = RuntimeError(
                        f"automatic tail primary fit did not converge ({fit.result.status})"
                    )
            except (RuntimeError, ArithmeticError, FloatingPointError) as exc:
                if not tail_proposals or backup is None:
                    raise
                original_error = exc
            if original_error is not None:
                if backup is None:
                    raise original_error
                try:
                    recovered = _fit_basis(backup, samples, weights, config=config)
                    if recovered.result.status not in _CONVERGED:
                        raise RuntimeError(
                            "automatic tail backup did not converge "
                            f"({recovered.result.status})"
                        )
                except (
                    RuntimeError,
                    ArithmeticError,
                    FloatingPointError,
                ) as retry_error:
                    raise RuntimeError(
                        "automatic tail fit failed with the empirical parameter "
                        f"({original_error}); single backup failed ({retry_error})"
                    ) from original_error
                fit = recovered
                for side, data in tail_diagnostics.items():
                    old = getattr(spec, side + "_tail")
                    new = getattr(backup, side + "_tail")
                    if old != new:
                        data["fallback_used"] = True
                        data["estimated_parameter"] = float(
                            new.rate if isinstance(new, Exponential) else new.power
                        )
            if fit.result.status == _UNCERTIFIED:
                warnings.warn(
                    "fit returned an uncertified shape verdict; inspect fit_diagnostics before quantitative use",
                    RuntimeWarning,
                    stacklevel=3,
                )
            packed = _pack_basis_fit(fit, n_eff)
            em_diag = None
        else:
            try:
                if k == "auto":
                    chosen = _select_components(
                        spec,
                        samples,
                        weights,
                        k_max=request.k_max,
                        subsample=request.auto_k_subsample,
                        rng=context.generator,
                        config=config,
                        em_options=em,
                    )
                    fit, p_values = chosen.fit, chosen.boundary_p_values
                    selection = chosen.diagnostics
                else:
                    resp = (
                        None
                        if seeds is None
                        else _seed_responsibilities(
                            context, seeds, request.seed_weights
                        )
                    )
                    fit, _, p_values = _fit_mixture_auto(
                        spec,
                        samples,
                        int(k),
                        weights,
                        rng=context.fit_generator,
                        responsibilities=resp,
                        degrees=degrees,
                        config=config,
                        em_options=em,
                    )
            except _PointMixtureEstimabilityFailure as exc:
                raise _PointMixtureEstimabilityError(
                    f"{exc} No initialization of the requested {k}-component "
                    "mixture produced an estimable fit, so the data do not "
                    f"support n_components={k}; use n_components='auto' or "
                    "fewer components."
                ) from exc
            except RuntimeError:
                if k == "auto" or samples.shape[1] != 2:
                    raise
                # No start of an explicit interval mixture converged.  If the
                # one-component fit already reaches the nonparametric bound,
                # the mixture (which contains it) does too: G1's
                # non-identifiability error, which G1 reported after its EM
                # reached the bound (G3-047).
                nested = _fit_basis(spec, samples, weights, config=config)
                _check_identifiable(
                    context,
                    -nested.result.objective_value,
                    int(k) * nested.model.n_coefficients + int(k) - 1,
                )
                raise
            if str(fit.status) not in _CONVERGED and str(fit.status) != _UNCERTIFIED:
                # Explicit counts and a selected count's final fit fail loudly
                # (G3-047); automatic searches end at their last converged
                # member inside the fitters.
                raise RuntimeError(
                    f"the {len(fit.models)}-component G3 mixture did not converge "
                    f"({fit.status})"
                )
            if len(fit.models) > 1:
                _check_identifiable(context, fit.log_likelihood, fit.n_face_parameters)
            if fit.status == _UNCERTIFIED:
                warnings.warn(
                    "fit returned an uncertified shape verdict; inspect fit_diagnostics before quantitative use",
                    RuntimeWarning,
                    stacklevel=3,
                )
            packed = _pack_selected(fit, n_eff, p_values)
            em_diag = _em_diagnostics(fit) if len(fit.models) > 1 else None
    mix_weights, states, metadata = packed
    components = [_Component(state) for state in states]
    if len(components) > 1:
        solver_order = {id(c): i for i, c in enumerate(components)}
        components, mix_weights = _sort_components_by_mode(components, mix_weights)
        public = tuple(solver_order[id(c)] for c in components)
        em_diag["solver_to_public"] = tuple(
            public.index(i) for i in range(len(components))
        )
    return _FitResult(
        components=components,
        weights=np.asarray(mix_weights, dtype=np.float64),
        selection_diagnostics=selection,
        em_diagnostics=em_diag,
        fit_metadata=metadata,
        tail_parameter_diagnostics=tail_diagnostics or None,
    )
