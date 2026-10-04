"""Staged, shared-model component-count selection.

Short shared fits screen a degree portfolio.  The structural shortlist contains
K=1, each degree's winner, and its immediate neighbors.  Only fits refined under
the requested degree and boundary policies compete for the final BIC.  This is
a search heuristic, not a safe global pruning rule.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import (
    AUTO_LC_MIN_COMPONENT_N,
    AUTO_LC_SWEEP_EM_MAX_ITER,
    BOUNDARY_ALPHA,
    NUMERIC_FAILURES,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
    _reraise_if_debug,
)
from .._fit.boundary import AUTO, _effective_n, _select_boundary_terms
from .._fit.mixture import (
    _initial_responsibility_candidates,
    _interval_identifiability_diagnostic,
    _stratified_subsample,
)
from .._fit.natural_mixture import (
    _EMOptions,
    _fit_natural_mixture,
    _NaturalComponent,
    _observation_weights,
    _run_natural_em,
)
from .._fit.natural_objective import (
    _fit_natural_conic_intervals,
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points,
    _fit_natural_conic_points_auto,
)
from .._observations.empirical import _normalized_weights


class _ComponentSelectionError(RuntimeError):
    """No refined shared-model candidate could be scored.

    Parameters
    ----------
    diagnostics : dict
        Search diagnostics, including individual candidate failure reasons.
    """

    def __init__(self, diagnostics):
        """Retain the failed search's diagnostics for the caller."""
        super().__init__("Shared-model component selection failed for every candidate")
        self.diagnostics = diagnostics


class _CandidateRejected(RuntimeError):
    """Expected candidate rejection during component-count selection."""


@dataclass(frozen=True)
class _SingleSelectionFit:
    """A single fit retained without constructing public spectral state.

    Parameters
    ----------
    objective, result : object
        Prepared natural objective and its optimized solver result.
    components : tuple
        One numerical component, without postfit spectral tables.
    weights : numpy.ndarray
        Unit mixture weight for the single component.
    log_likelihood : float
        Weighted mean observation log likelihood.
    n_face_parameters : int
        Number of parameters on the fitted numerical face.
    """

    objective: object
    result: object
    components: tuple
    weights: np.ndarray
    log_likelihood: float
    n_face_parameters: int


@dataclass(frozen=True)
class _SelectionCandidate:
    """Canonical inputs and numerical seed for one shortlist refinement.

    Parameters
    ----------
    rows, samples_1d : numpy.ndarray
        Selection observation rows and their initializer representatives.
    observation_weights : numpy.ndarray
        Normalized selection-row weights.
    n_components : int
        Candidate component count.
    responsibilities : numpy.ndarray or None
        Screening initializer restricted to the selection rows.
    rng : numpy.random.Generator
        Candidate-specific refinement stream.
    initial_fit : object or None
        Best screening fit, when one succeeded.
    """

    rows: np.ndarray
    samples_1d: np.ndarray
    observation_weights: np.ndarray
    n_components: int
    responsibilities: np.ndarray | None
    rng: np.random.Generator
    initial_fit: object | None


def _single_selection_fit(support, rows, degree, lower, upper, weights, degree_config):
    """Fit one candidate without constructing public spectral state.

    Parameters
    ----------
    support : tuple of float
        Validated physical support.
    rows : numpy.ndarray
        Canonical float64 point or interval rows.
    degree : int or str
        Fixed admissible degree or ``"auto"``.
    lower, upper : bool or str
        Prepared boundary policies.
    weights : numpy.ndarray
        Normalized selection-row weights.
    degree_config : _DegreeSelectionConfig
        Explicit omitted-information policy shared by the selection search.
    """
    intervals = rows.shape[1] == 2
    automatic = degree is None or degree == "auto"
    if automatic:
        fitter = (
            _fit_natural_conic_intervals_auto
            if intervals
            else _fit_natural_conic_points_auto
        )
        objective, result = fitter(
            support,
            rows if intervals else rows[:, 0],
            lower,
            upper,
            weights,
            degree_config=degree_config,
        )
    else:
        fitter = (
            _fit_natural_conic_intervals if intervals else _fit_natural_conic_points
        )
        objective, result = fitter(
            support,
            rows if intervals else rows[:, 0],
            int(degree),
            lower,
            upper,
            weights,
        )
    component = _NaturalComponent(
        objective.spec.coordinate,
        objective.spec,
        objective.layout,
        objective.z_data_bounds,
        result.params,
        result.effective_curvature_degree,
        result.lower_amplitude_active,
        result.upper_amplitude_active,
        result,
    )
    dimension = (
        2
        + int(result.effective_curvature_degree)
        + int(result.lower_amplitude_active)
        + int(result.upper_amplitude_active)
    )
    return _SingleSelectionFit(
        objective, result, (component,), np.ones(1), -result.objective_value, dimension
    )


def _selection_payload(fitted):
    """Normalize the state-free refinement callback's numerical result.

    Parameters
    ----------
    fitted : _SingleSelectionFit or _NaturalMixtureFit
        Numerical fit exposing components, weights, mean likelihood and face
        dimension. Policy metadata is attached explicitly by the caller.
    """
    return {
        "fit": fitted,
        "log_likelihood": float(fitted.log_likelihood),
        "n_face_parameters": int(fitted.n_face_parameters),
        "weights": fitted.weights,
        "components": fitted.components,
    }


def _candidate_degrees(payload):
    """Return explicit degrees or recover them from the fitted components.

    Parameters
    ----------
    payload : dict
        Refinement result with components and optional explicit ``degrees``.
    """
    if "degrees" in payload:
        return tuple(map(int, payload["degrees"]))
    return tuple(
        int(component.spec.requested_poly_degree) for component in payload["components"]
    )


def _default_refinement(candidate, supp, policy, degree_config, /):
    """Private numerical fallback for direct users of the selection helper.

    Parameters
    ----------
    candidate : _SelectionCandidate
        Prepared selection rows, weights and numerical initialization.
    supp : tuple of float
        Validated physical support.
    policy : _ComponentSelectionPolicy
        Resolved degree and full-data boundary policies.
    degree_config : _DegreeSelectionConfig
        Explicit omitted-information policy reused by all refinement fits.
    """
    rows = candidate.rows
    weights = candidate.observation_weights
    count = candidate.n_components
    previous = candidate.initial_fit
    degree_policy = (policy.degree_policy,) * count
    reduced = {}

    def fit_at(degree, lo, up, initial):
        if count == 1:
            return _single_selection_fit(
                supp, rows, degree[0], lo, up, weights, degree_config
            )
        return _fit_natural_mixture(
            supp,
            rows,
            count,
            degree,
            lo,
            up,
            weights,
            rng=candidate.rng,
            responsibilities=candidate.responsibilities,
            initial_fit=initial,
            degree_config=degree_config,
        )

    def nested(model, lo, up):
        locked = tuple(c.spec.requested_poly_degree for c in model.components)
        fitted = fit_at(locked, lo, up, model)
        reduced[(lo, up)] = fitted
        return fitted

    def amplitude(model, side):
        component = model.components[0]
        index = getattr(component.spec, f"physical_{side}_a_index")
        return 0.0 if index is None else float(component.params[index])

    def reselect(lo, up):
        return fit_at(degree_policy, lo, up, reduced[(lo, up)])

    fitted, flags, p_values = _select_boundary_terms(
        lambda lo, up: fit_at(degree_policy, lo, up, previous),
        lambda model: -model.log_likelihood,
        amplitude,
        policy.lower_boundary,
        policy.upper_boundary,
        _effective_n(len(rows), weights),
        fit_reduced=nested,
        refit=reselect if "auto" in degree_policy else None,
        alpha=BOUNDARY_ALPHA,
    )
    payload = _selection_payload(fitted)
    payload.update(boundary_p_values=p_values, boundary_flags=flags)
    return payload


def select_n_components(
    *,
    S,
    samples_1d,
    supp,
    k_modes,
    gen,
    obs_w,
    verb,
    policy,
    degree_config,
    refine_candidate,
):
    """Select K using shared screens and an explicitly refined shortlist.

    The policy's boundary flags must already incorporate endpoint
    exclusions from the full data.  AUTO bases are enabled in every screen;
    their LR policy runs only during refinement.  An explicit degree replaces
    the automatic screening portfolio.

    ``refine_candidate`` is a state-free callback accepting a
    :class:`_SelectionCandidate` record.
    It returns a dictionary with ``fit, log_likelihood`` (weighted mean),
    ``n_face_parameters, weights, components`` and optional policy metadata.
    It must apply the caller's actual degree/boundary policies and fit controls.

    The four-tuple return retains full-data initial responsibilities.  The
    private diagnostics entry ``_selected_fit`` carries the completed callback
    payload; callers should remove it before publishing diagnostics and reuse
    it when ``reuse_selected_fit`` is true.  Subsample BIC explicitly
    extrapolates its mean likelihood to the full-data Kish effective size.

    Parameters
    ----------
    S : numpy.ndarray
        Canonical float64 observations shaped ``(R, 1)`` or ``(R, 2)``.
    samples_1d : numpy.ndarray
        Canonical full-data representatives used by the initializer.
    supp : tuple of float
        Validated physical support.
    k_modes : int
        Full-data mode proposal retained in diagnostics.
    gen : numpy.random.Generator
        Generator supplying purpose- and candidate-specific streams.
    obs_w : numpy.ndarray or None
        Boundary-normalized original row weights, or uniform weighting.
    verb : int
        Prepared verbosity level.
    policy : _ComponentSelectionPolicy
        Explicit search limits, subsample size and degree/boundary policies.
    degree_config : _DegreeSelectionConfig
        One explicit diagnostic policy reused by screening and refinement.
    refine_candidate : callable or None
        Callback consuming one candidate record and returning the numerical
        payload described above. Explicit ``None`` uses private refinement.

    Returns
    -------
    tuple
        Selected count, full-data initial responsibilities, initial mixture
        weights and search diagnostics.
    """
    n_rows = len(S)
    full_w = _observation_weights(n_rows, obs_w)
    effective_n = _effective_n(n_rows, full_w)
    ceiling = policy.ceiling
    initial_hi = policy.initial_hi
    degrees = policy.degrees
    sub_m = policy.subsample_size
    # Streams are indexed by purpose and K, never by candidate execution order.
    entropy = gen.integers(0, 2**32, size=4, dtype=np.uint32)

    def rng_for(stage, count):
        return np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(stage, count))
        )

    sub_idx = (
        _stratified_subsample(samples_1d, sub_m, rng_for(0, 0))
        if sub_m < n_rows
        else None
    )
    fit_S = S if sub_idx is None else np.ascontiguousarray(S[sub_idx])
    fit_x = samples_1d if sub_idx is None else np.ascontiguousarray(samples_1d[sub_idx])
    fit_w = (
        full_w
        if sub_idx is None
        else _normalized_weights(sub_m, full_w[sub_idx], "selection").weights
    )
    sample_effective_n = _effective_n(sub_m, fit_w)
    enabled = tuple(
        flag is True or flag == AUTO
        for flag in (policy.lower_boundary, policy.upper_boundary)
    )
    screens = {}
    screening_scores = []
    refined = {}
    scores = []
    admissions = {1: "mandatory_single"}
    log_n = float(np.log(max(effective_n, 1.0)))

    def score(fitted):
        dimension = int(fitted["n_face_parameters"])
        likelihood = float(fitted["log_likelihood"])
        if dimension < 1 or not np.isfinite(likelihood):
            raise _CandidateRejected(
                "candidate has invalid likelihood or face dimension"
            )
        if len(fitted["weights"]) > 1:
            if (
                float(np.min(fitted["weights"])) * sample_effective_n
                < AUTO_LC_MIN_COMPONENT_N
            ):
                raise _CandidateRejected("degenerate_component")
            if (
                S.shape[1] == 2
                and _interval_identifiability_diagnostic(
                    fit_S,
                    fitted["components"],
                    likelihood,
                    supp,
                    obs_weights=fit_w,
                    n_parameters=dimension,
                )
                is not None
            ):
                raise _CandidateRejected("unidentified_interval_components")
        return -2.0 * likelihood * effective_n + dimension * log_n

    def screen(count):
        if count in screens:
            return
        screens[count] = {}
        if count == 1:
            candidates = ((None, None, None),)
        else:
            try:
                with _maybe_suppress(True, SUPPRESSED_WARNINGS):
                    candidates = _initial_responsibility_candidates(
                        samples_1d, count, rng_for(1, count), weights=full_w
                    )
            except NUMERIC_FAILURES as exc:
                _reraise_if_debug(
                    exc, f"shared K screen initialization K={count}", routine=True
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
            best = None
            failures = []
            for name, resp, mix_weights in candidates:
                try:
                    r = (
                        None
                        if resp is None
                        else np.asarray(
                            resp if sub_idx is None else resp[sub_idx], dtype=np.float64
                        )
                    )
                    if count == 1:
                        fitted = _single_selection_fit(
                            supp, fit_S, degree, *enabled, fit_w, degree_config
                        )
                    else:
                        sharpened = np.square(r)
                        sharpened /= sharpened.sum(axis=1, keepdims=True)
                        fitted = _run_natural_em(
                            supp,
                            fit_S,
                            (degree,) * count,
                            *enabled,
                            fit_w,
                            sharpened,
                            em_options=_EMOptions(
                                max_steps=AUTO_LC_SWEEP_EM_MAX_ITER,
                                accelerate=False,
                            ),
                            initialization=str(name),
                            polish=False,
                            degree_config=degree_config,
                        )
                    payload = _selection_payload(fitted)
                    bic = score(payload)
                    record = {
                        "bic": float(bic),
                        "degree": int(degree),
                        "initializer": name,
                        "fit": fitted,
                        "responsibilities": resp,
                        "weights": mix_weights,
                    }
                    if best is None or bic < best["bic"]:
                        best = record
                except _CandidateRejected as exc:
                    # Candidate rejection is expected control flow, not a numerical
                    # fallback and therefore is not recorded in the failure ledger.
                    failures.append(f"{type(exc).__name__}: {exc}")
                except NUMERIC_FAILURES as exc:
                    _reraise_if_debug(
                        exc, f"shared K screen K={count}, d={degree}", routine=True
                    )
                    failures.append(f"{type(exc).__name__}: {exc}")
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

    for count in range(1, initial_hi + 1):
        screen(count)
    for degree in degrees:
        viable = [
            (entries[degree]["bic"], count)
            for count, entries in screens.items()
            if entries[degree] is not None
        ]
        if not viable:
            continue
        winner = min(viable)[1]
        for count in (winner - 1, winner, winner + 1):
            if 1 <= count <= ceiling:
                admissions.setdefault(
                    count, f"screen_degree_{degree}_winner_or_neighbor"
                )

    def refine(count):
        try:
            screen(count)
            viable = [entry for entry in screens[count].values() if entry is not None]
            start = min(viable, key=lambda item: item["bic"]) if viable else None
            resp = None if start is None else start["responsibilities"]
            sample_resp = (
                None if resp is None else (resp if sub_idx is None else resp[sub_idx])
            )
            candidate = _SelectionCandidate(
                fit_S,
                fit_x,
                fit_w,
                count,
                sample_resp,
                rng_for(2, count),
                None if start is None else start["fit"],
            )
            payload = (
                _default_refinement(candidate, supp, policy, degree_config)
                if refine_candidate is None
                else refine_candidate(candidate)
            )
            bic = score(payload)
            refined[count] = (bic, payload, start)
            scores.append(
                {
                    "n_components": count,
                    "bic": float(bic),
                    "status": "ok",
                    "degree": _candidate_degrees(payload),
                    "n_parameters": int(payload["n_face_parameters"]),
                    "mean_log_likelihood": float(payload["log_likelihood"]),
                    "initializer": None if start is None else start["initializer"],
                    "admission": admissions[count],
                }
            )
        except _CandidateRejected as exc:
            # Candidate rejection is expected control flow, not a numerical
            # fallback and therefore is not recorded in the failure ledger.
            scores.append(
                {
                    "n_components": count,
                    "bic": None,
                    "status": f"failed:{type(exc).__name__}",
                    "reason": str(exc),
                    "admission": admissions[count],
                }
            )
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, f"shared K refinement K={count}", routine=True)
            scores.append(
                {
                    "n_components": count,
                    "bic": None,
                    "status": f"failed:{type(exc).__name__}",
                    "reason": str(exc),
                    "admission": admissions[count],
                }
            )

    for count in sorted(admissions):
        refine(count)
    while refined:
        winner = min(refined, key=lambda count: (refined[count][0], count))
        neighbors = [
            count
            for count in (winner - 1, winner + 1)
            if 1 <= count <= ceiling and count not in admissions
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
        "subsample_size": sub_m,
        "full_sample_size": n_rows,
        "subsampled": sub_idx is not None,
        "score_effective_n": float(effective_n),
        "selection_sample_effective_n": float(sample_effective_n),
        "likelihood_scale": (
            "subsample_mean_extrapolated_to_full_effective_n"
            if sub_idx is not None
            else "full_data_effective_n"
        ),
        "screen_boundary_enabled": enabled,
        "screen_degrees": degrees,
        "screening_scores": tuple(screening_scores),
        "scores": tuple(scores),
        "admissions": tuple(sorted(admissions.items())),
        "excluded_components": tuple(
            count for count in range(1, ceiling + 1) if count not in admissions
        ),
        "termination": (
            "closed_winner_neighborhood_or_k_limit"
            if refined
            else "all_refinements_failed"
        ),
        "reuse_selected_fit": sub_idx is None,
    }
    if not refined:
        raise _ComponentSelectionError(diagnostics)
    winner = min(refined, key=lambda count: (refined[count][0], count))
    bic, payload, start = refined[winner]
    diagnostics.update(
        selected_n_components=winner, selected_bic=float(bic), _selected_fit=payload
    )
    if verb >= 1:
        print(f"  -> shared staged BIC selected K={winner} (BIC={bic:.2f})")
    if winner == 1:
        return 1, None, None, diagnostics
    if sub_idx is None:
        resp = payload["fit"].responsibilities
        mix_weights = payload["weights"]
    elif start is not None:
        resp, mix_weights = start["responsibilities"], start["weights"]
    else:
        with _maybe_suppress(True, SUPPRESSED_WARNINGS):
            _, resp, mix_weights = _initial_responsibility_candidates(
                samples_1d, winner, rng_for(1, winner), weights=full_w
            )[0]
    return winner, resp, mix_weights, diagnostics
