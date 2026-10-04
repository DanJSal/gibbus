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
    AUTO_LC_SUBSAMPLE_MIN_N,
    AUTO_LC_SUBSAMPLE_PER_K,
    AUTO_LC_SUBSAMPLE_SIZE,
    AUTO_LC_SWEEP_EM_MAX_ITER,
    AUTO_LC_SWEEP_MAX_K,
    AUTO_LC_VALIDATION_DEGREES,
    NUMERIC_FAILURES,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
    _reraise_if_debug,
)
from .._fit.boundary import AUTO, _effective_n, _select_boundary_terms
from .._fit.inputs import _is_poly_degree_admissible
from .._fit.mixture import (
    _initial_responsibility_candidates,
    _interval_identifiability_diagnostic,
    _stratified_subsample,
)
from .._fit.natural_mixture import (
    _fit_natural_mixture,
    _NaturalComponent,
    _normalized_observation_weights,
    _run_natural_em,
)
from .._fit.natural_objective import (
    _fit_natural_conic_intervals,
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points,
    _fit_natural_conic_points_auto,
)


class _ComponentSelectionError(RuntimeError):
    """No refined shared-model candidate could be scored."""

    def __init__(self, diagnostics):
        super().__init__("Shared-model component selection failed for every candidate")
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class _SingleSelectionFit:
    """A single fit retained without constructing public spectral state."""

    objective: object
    result: object
    components: tuple
    weights: np.ndarray
    log_likelihood: float
    n_face_parameters: int


def _single_selection_fit(support, rows, degree, lower, upper, weights):
    intervals = rows.shape[1] == 2
    automatic = degree is None or degree == "auto"
    if automatic:
        fitter = (
            _fit_natural_conic_intervals_auto
            if intervals
            else _fit_natural_conic_points_auto
        )
        objective, result = fitter(
            support, rows if intervals else rows[:, 0], lower, upper, weights
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


def _selection_payload(fitted, **extra):
    """Normalize the state-free refinement callback's numerical result."""
    return {
        "fit": fitted,
        "log_likelihood": float(fitted.log_likelihood),
        "n_face_parameters": int(fitted.n_face_parameters),
        "weights": fitted.weights,
        "components": fitted.components,
        **extra,
    }


def _candidate_degrees(payload):
    if "degrees" in payload:
        return tuple(map(int, payload["degrees"]))
    return tuple(
        int(component.spec.requested_poly_degree) for component in payload["components"]
    )


def _default_refinement(
    *, supp, degree_policy, lower_boundary, upper_boundary, **candidate
):
    """Private numerical fallback for direct users of the selection helper."""
    rows = candidate["S"]
    weights = candidate["obs_w"]
    count = candidate["n_components"]
    previous = candidate["initial_fit"]
    reduced = {}

    def fit_at(degree, lo, up, initial=None):
        if count == 1:
            return _single_selection_fit(supp, rows, degree, lo, up, weights)
        return _fit_natural_mixture(
            supp,
            rows,
            count,
            degree,
            lo,
            up,
            weights,
            rng=candidate["rng"],
            responsibilities=candidate["responsibilities"],
            initial_fit=initial,
        )

    def nested(model, lo, up):
        locked = tuple(c.spec.requested_poly_degree for c in model.components)
        fitted = fit_at(locked[0] if count == 1 else locked, lo, up, model)
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
        lower_boundary,
        upper_boundary,
        _effective_n(len(rows), weights),
        fit_reduced=nested,
        refit=reselect if degree_policy is None or degree_policy == "auto" else None,
    )
    return _selection_payload(fitted, boundary_p_values=p_values, boundary_flags=flags)


def select_n_components(
    *,
    S,
    samples_1d,
    supp,
    k_modes,
    effective_k_max,
    gen,
    obs_w,
    verb,
    subsample="auto",
    lower_boundary=False,
    upper_boundary=False,
    degree_policy="auto",
    refine_candidate=None,
):
    """Select K using shared screens and an explicitly refined shortlist.

    ``lower_boundary`` and ``upper_boundary`` must already incorporate endpoint
    exclusions from the full data.  AUTO bases are enabled in every screen;
    their LR policy runs only during refinement.  An explicit degree replaces
    the automatic screening portfolio.

    ``refine_candidate`` is a state-free callback accepting keyword arguments
    ``S, samples_1d, obs_w, n_components, responsibilities, rng, initial_fit``.
    It returns a dictionary with ``fit, log_likelihood`` (weighted mean),
    ``n_face_parameters, weights, components`` and optional policy metadata.
    It must apply the caller's actual degree/boundary policies and fit controls.

    The four-tuple return retains full-data initial responsibilities.  The
    private diagnostics entry ``_selected_fit`` carries the completed callback
    payload; callers should remove it before publishing diagnostics and reuse
    it when ``reuse_selected_fit`` is true.  Subsample BIC explicitly
    extrapolates its mean likelihood to the full-data Kish effective size.
    """
    S = np.ascontiguousarray(S, dtype=np.float64)
    samples_1d = np.asarray(samples_1d, dtype=np.float64)
    n_rows = len(S)
    full_w = _normalized_observation_weights(n_rows, obs_w)
    effective_n = _effective_n(n_rows, full_w)
    ceiling = max(
        1,
        min(int(effective_k_max), AUTO_LC_SWEEP_MAX_K, max(n_rows // 50, 2)),
    )
    initial_hi = min(ceiling, max(2 * int(k_modes), 1))
    if degree_policy is None or degree_policy == "auto":
        degrees = tuple(
            d for d in AUTO_LC_VALIDATION_DEGREES if _is_poly_degree_admissible(d, supp)
        )
    else:
        degrees = (int(degree_policy),)
        if not _is_poly_degree_admissible(degrees[0], supp):
            raise ValueError("requested screening degree is not admissible on support")
    if not degrees:
        raise ValueError("no admissible screening degrees")

    if subsample is False or subsample == 0:
        sub_m = n_rows
    elif isinstance(subsample, str):
        if subsample != "auto":
            raise ValueError(
                f"subsample must be 'auto', an int, or False, got {subsample!r}."
            )
        sub_m = (
            max(AUTO_LC_SUBSAMPLE_SIZE, initial_hi * AUTO_LC_SUBSAMPLE_PER_K)
            if n_rows > AUTO_LC_SUBSAMPLE_MIN_N
            else n_rows
        )
    else:
        sub_m = int(subsample)
        if sub_m < 2:
            raise ValueError(f"subsample must be at least 2, got {sub_m}.")
    sub_m = min(sub_m, n_rows)
    # Streams are indexed by purpose and K, never by candidate execution order.
    entropy = gen.integers(0, 2**32, size=4, dtype=np.uint32)

    def rng_for(stage, count=0):
        return np.random.default_rng(
            np.random.SeedSequence(entropy, spawn_key=(stage, count))
        )

    sub_idx = (
        _stratified_subsample(samples_1d, sub_m, rng_for(0)) if sub_m < n_rows else None
    )
    fit_S = S if sub_idx is None else np.ascontiguousarray(S[sub_idx])
    fit_x = samples_1d if sub_idx is None else np.ascontiguousarray(samples_1d[sub_idx])
    fit_w = (
        full_w
        if sub_idx is None
        else _normalized_observation_weights(sub_m, full_w[sub_idx])
    )
    sample_effective_n = _effective_n(sub_m, fit_w)
    enabled = tuple(
        flag is True or flag == AUTO for flag in (lower_boundary, upper_boundary)
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
            raise ValueError("candidate has invalid likelihood or face dimension")
        if len(fitted["weights"]) > 1:
            if (
                float(np.min(fitted["weights"])) * sample_effective_n
                < AUTO_LC_MIN_COMPONENT_N
            ):
                raise ValueError("degenerate_component")
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
                raise ValueError("unidentified_interval_components")
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
            except (*NUMERIC_FAILURES, ValueError) as exc:
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
                            supp, fit_S, degree, *enabled, fit_w
                        )
                    else:
                        sharpened = np.square(r)
                        sharpened /= sharpened.sum(axis=1, keepdims=True)
                        fitted = _run_natural_em(
                            supp,
                            fit_S,
                            degree,
                            *enabled,
                            fit_w,
                            sharpened,
                            max_em_steps=AUTO_LC_SWEEP_EM_MAX_ITER,
                            accelerate=False,
                            initialization=str(name),
                            polish=False,
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
                except (*NUMERIC_FAILURES, ValueError) as exc:
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
            candidate = {
                "S": fit_S,
                "samples_1d": fit_x,
                "obs_w": fit_w,
                "n_components": count,
                "responsibilities": sample_resp,
                "rng": rng_for(2, count),
                "initial_fit": None if start is None else start["fit"],
            }
            payload = (
                _default_refinement(
                    supp=supp,
                    degree_policy=degree_policy,
                    lower_boundary=lower_boundary,
                    upper_boundary=upper_boundary,
                    **candidate,
                )
                if refine_candidate is None
                else refine_candidate(**candidate)
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
        except (*NUMERIC_FAILURES, ValueError) as exc:
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
        "likelihood_scale": "subsample_mean_extrapolated_to_full_effective_n"
        if sub_idx is not None
        else "full_data_effective_n",
        "screen_boundary_enabled": enabled,
        "screen_degrees": degrees,
        "screening_scores": tuple(screening_scores),
        "scores": tuple(scores),
        "admissions": tuple(sorted(admissions.items())),
        "excluded_components": tuple(
            count for count in range(1, ceiling + 1) if count not in admissions
        ),
        "termination": "closed_winner_neighborhood_or_k_limit"
        if refined
        else "all_refinements_failed",
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
