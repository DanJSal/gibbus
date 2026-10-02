"""Automatic component-count selection.

This is where *K* is decided.  :func:`._fit.mixture._propose_n_components`
supplies only the KDE mode count that centres the search; candidates are then
scored with lightweight log-concave fits, so the screening model belongs to
the same family as the final fit rather than to a Gaussian surrogate.

The natural-coordinate sweep is deliberately side-effect free: it returns its
diagnostic record rather than writing onto the model, so it can be exercised
in isolation and so :class:`~gibbus._api.distribution.Distribution` owns all of
its own state mutation.
"""

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
from .._fit.inputs import _is_poly_degree_admissible
from .._fit.mixture import (
    _init_responsibilities,
    _initial_responsibility_candidates,
    _interval_identifiability_diagnostic,
    _stratified_subsample,
)
from .._fit.natural_mixture import (
    _normalized_observation_weights,
    _run_natural_em,
)
from .._fit.natural_objective import (
    _fit_natural_conic_intervals,
    _fit_natural_conic_points,
)


def select_n_components(*, S, samples_1d, supp, k_modes,
                                effective_k_max, gen, obs_w, verb,
                                subsample="auto"):
    """Choose the component count by BIC.

    The sweep searches a KDE-centred range of candidate counts over validation
    degrees, scores candidates on a stratified subsample when the data are
    large, requires minimum effective support per component, counts BIC
    parameters from the fitted faces, and stops early once BIC stops
    improving.  Screening mixtures run short ordinary EM without joint
    polishing; the returned full-data initializer is the same initializer
    whose screening fit attained the winning BIC.

    Parameters
    ----------
    S : numpy.ndarray, shape (R, k)
        Point rows ``(R, 1)`` or interval rows ``(R, 2)``.
    samples_1d : numpy.ndarray, shape (R,)
        Point samples or interval representatives used only for KDE and
        responsibility initialization.
    supp : tuple of (float, float)
        Shared support.
    k_modes : int
        KDE mode-count proposal centring the BIC search.
    effective_k_max : int
        Clamped upper bound on candidate component count.
    gen : numpy.random.Generator
        Random generator for initialization candidates and subsampling.
    obs_w : numpy.ndarray, shape (R,) or None
        Normalized observation weights.
    verb : int
        Internal verbosity level.
    subsample : {"auto"}, int, or False, optional
        Candidate-scoring subsample policy.

    Returns
    -------
    K : int
        Selected component count.
    resp : numpy.ndarray or None
        Full-data responsibilities of the winning initializer, or ``None``
        for ``K=1``.
    mix_weights : numpy.ndarray or None
        Initial mixture weights matching ``resp``.
    diagnostics : dict
        Selection record (per-candidate BIC, degrees and support).
    """
    lc_k_lo = max(int(k_modes) - 2, 1)
    n_val = int(S.shape[0])
    lc_k_hi = min(
        int(effective_k_max), 2 * int(k_modes), AUTO_LC_SWEEP_MAX_K
    )
    lc_k_hi = min(lc_k_hi, max(n_val // 50, 2))
    lc_k_hi = max(lc_k_hi, lc_k_lo)
    # The KDE proposal only centres the range.  When the best score sits on
    # its upper edge the sweep keeps going, one K at a time, until the score
    # stops improving or the hard ceiling is reached: a KDE that smooths two
    # overlapping log-concave components into one mode must not cap K at 2.
    lc_k_ceiling = max(
        min(int(effective_k_max), AUTO_LC_SWEEP_MAX_K, max(n_val // 50, 2)),
        lc_k_hi,
    )

    log_n_val = float(np.log(n_val))
    val_degrees = tuple(
        d for d in AUTO_LC_VALIDATION_DEGREES
        if _is_poly_degree_admissible(d, supp)
    ) or (AUTO_LC_VALIDATION_DEGREES[0],)

    if subsample is False or subsample == 0:
        sub_m = n_val
    elif isinstance(subsample, str):
        if subsample != "auto":
            raise ValueError(
                f"subsample must be 'auto', an int, or False, got {subsample!r}."
            )
        sub_m = (
            max(AUTO_LC_SUBSAMPLE_SIZE, lc_k_hi * AUTO_LC_SUBSAMPLE_PER_K)
            if n_val > AUTO_LC_SUBSAMPLE_MIN_N else n_val
        )
    else:
        sub_m = int(subsample)
        if sub_m < 2:
            raise ValueError(f"subsample must be at least 2, got {sub_m}.")
    sub_m = min(sub_m, n_val)

    if sub_m < n_val:
        sub_idx = _stratified_subsample(samples_1d, sub_m, gen)
        fit_x = np.ascontiguousarray(samples_1d[sub_idx])
        fit_S = np.ascontiguousarray(S[sub_idx])
        if obs_w is None:
            fit_w = None
        else:
            fit_w = np.asarray(obs_w[sub_idx], dtype=np.float64)
            fit_w = fit_w / fit_w.sum()
    else:
        sub_idx = None
        fit_x = samples_1d
        fit_S = S
        fit_w = obs_w

    is_interval = int(S.shape[1]) == 2
    natural_w = _normalized_observation_weights(sub_m, fit_w)

    if verb >= 1:
        scope = (
            f"on {sub_m}/{n_val} stratified subsample"
            if sub_idx is not None else f"on all {n_val} samples"
        )
        print(
            f"  natural LC BIC sweep: K={lc_k_lo}..{lc_k_hi} "
            f"(K_modes={k_modes}) {scope}"
        )

    best_lc_bic = np.inf
    best_lc_k = int(k_modes)
    lc_scores = []
    best_lc_resp = None
    best_lc_wts = None
    lc_rising = 0

    for ck in range(lc_k_lo, lc_k_ceiling + 1):
        if ck > lc_k_hi and best_lc_k != ck - 1:
            break
        try:
            if ck == 1:
                lc_bic = np.inf
                lc_degree = None
                degree_exc = None
                for degree in val_degrees:
                    try:
                        if is_interval:
                            objective, result = _fit_natural_conic_intervals(
                                supp, fit_S, degree, False, False, fit_w
                            )
                        else:
                            objective, result = _fit_natural_conic_points(
                                supp, fit_x, degree, False, False, fit_w
                            )
                    except NUMERIC_FAILURES as exc:
                        _reraise_if_debug(
                            exc,
                            f"natural LC BIC sweep K=1 at degree {degree}",
                        )
                        degree_exc = exc
                        continue
                    nll = float(result.objective_value)
                    params = int(objective.layout.n_params)
                    bic = 2.0 * nll * n_val + params * log_n_val
                    if bic < lc_bic:
                        lc_bic = bic
                        lc_degree = int(degree)
                if lc_degree is None:
                    raise (
                        degree_exc if degree_exc is not None
                        else RuntimeError("natural LC BIC sweep K=1 failed")
                    )

                lc_scores.append({
                    "n_components": 1,
                    "degree": lc_degree,
                    "initializer": None,
                    "bic": float(lc_bic),
                    "status": "ok",
                })
                if verb >= 1:
                    print(f"    natural LC(K=1,d={lc_degree})  BIC={lc_bic:.2f}")
                if lc_bic < best_lc_bic:
                    best_lc_bic = lc_bic
                    best_lc_k = 1
                    best_lc_resp = None
                    best_lc_wts = None
                    lc_rising = 0
                else:
                    lc_rising += 1
            else:
                with _maybe_suppress(
                    True, SUPPRESSED_WARNINGS
                ):
                    candidates = _initial_responsibility_candidates(
                        samples_1d, ck, gen, weights=obs_w
                    )
                lc_bic = np.inf
                lc_degree = None
                lc_initializer = None
                lc_resp = None
                lc_wts = None
                degree_exc = None
                degenerate = False

                for init_name, init_resp, init_wts in candidates:
                    if sub_idx is None:
                        init_resp_fit = np.asarray(init_resp, dtype=np.float64)
                    else:
                        init_resp_fit = np.asarray(
                            init_resp[sub_idx], dtype=np.float64
                        )
                        init_resp_fit = init_resp_fit / init_resp_fit.sum(
                            axis=1, keepdims=True
                        )
                    # Screening EM sharpens its initial responsibilities once
                    # before the first M-step.
                    screen_resp = np.square(init_resp_fit)
                    screen_resp /= screen_resp.sum(axis=1, keepdims=True)

                    for degree in val_degrees:
                        try:
                            fitted = _run_natural_em(
                                supp,
                                fit_S,
                                degree,
                                False,
                                False,
                                natural_w,
                                screen_resp,
                                max_em_steps=AUTO_LC_SWEEP_EM_MAX_ITER,
                                accelerate=False,
                                initialization=str(init_name),
                                polish=False,
                            )
                        except (*NUMERIC_FAILURES, ValueError) as exc:
                            _reraise_if_debug(
                                exc,
                                f"natural LC BIC sweep K={ck}, "
                                f"init={init_name}, degree {degree}",
                                routine=True,
                            )
                            degree_exc = exc
                            continue
                        if (
                            float(np.min(fitted.weights)) * n_val
                            < AUTO_LC_MIN_COMPONENT_N
                        ):
                            degenerate = True
                            continue
                        if is_interval and _interval_identifiability_diagnostic(
                            fit_S, fitted.components, fitted.log_likelihood, supp,
                            obs_weights=natural_w,
                        ) is not None:
                            # A saturated richer censored candidate has no
                            # identified component decomposition and is not a
                            # valid BIC competitor.
                            continue
                        params = sum(
                            int(component.layout.n_params)
                            for component in fitted.components
                        ) + (ck - 1)
                        bic = (
                            -2.0 * float(fitted.log_likelihood) * n_val
                            + params * log_n_val
                        )
                        if bic < lc_bic:
                            lc_bic = bic
                            lc_degree = int(degree)
                            lc_initializer = str(init_name)
                            lc_resp = init_resp
                            lc_wts = init_wts

                if lc_degree is None:
                    if degenerate:
                        lc_scores.append({
                            "n_components": int(ck),
                            "degree": None,
                            "initializer": None,
                            "bic": None,
                            "status": "degenerate_component",
                        })
                        if verb >= 1:
                            print(
                                f"    natural LC(K={ck}) rejected: component "
                                f"below {AUTO_LC_MIN_COMPONENT_N} effective samples"
                            )
                        continue
                    raise (
                        degree_exc if degree_exc is not None
                        else RuntimeError(f"natural LC BIC sweep K={ck} failed")
                    )

                lc_scores.append({
                    "n_components": int(ck),
                    "degree": lc_degree,
                    "initializer": lc_initializer,
                    "bic": float(lc_bic),
                    "status": "ok",
                })
                if verb >= 1:
                    print(
                        f"    natural LC(K={ck},d={lc_degree},"
                        f"init={lc_initializer})  BIC={lc_bic:.2f}"
                    )
                if lc_bic < best_lc_bic:
                    best_lc_bic = lc_bic
                    best_lc_k = ck
                    best_lc_resp = lc_resp
                    best_lc_wts = lc_wts
                    lc_rising = 0
                else:
                    lc_rising += 1
        except (*NUMERIC_FAILURES, ValueError) as exc:
            _reraise_if_debug(
                exc, f"natural LC BIC sweep candidate K={ck}", routine=True
            )
            lc_scores.append({
                "n_components": int(ck),
                "degree": None,
                "initializer": None,
                "bic": None,
                "status": f"failed:{type(exc).__name__}",
            })
            if verb >= 1:
                print(f"    natural LC(K={ck}) failed -- skipping")
            lc_rising += 1

        if lc_rising >= 2:
            if verb >= 1:
                print(
                    "    (natural LC early stop -- BIC rising for "
                    f"{lc_rising} consecutive K values)"
                )
            break

    if not np.isfinite(best_lc_bic):
        diagnostics = {
            "method": "kde_fallback",
            "kde_modes": int(k_modes),
            "selected_n_components": int(max(1, k_modes)),
            "subsample_size": int(sub_m),
            "full_sample_size": int(n_val),
            "subsampled": bool(sub_idx is not None),
            "scores": tuple(lc_scores),
        }
        if k_modes <= 1:
            return 1, None, None, diagnostics
        with _maybe_suppress(
            True, SUPPRESSED_WARNINGS
        ):
            fb_resp, fb_wts = _init_responsibilities(
                samples_1d, k_modes, gen, weights=obs_w
            )
        return k_modes, fb_resp, fb_wts, diagnostics

    diagnostics = {
        "method": "log_concave_bic",
        "kde_modes": int(k_modes),
        "selected_n_components": int(best_lc_k),
        "selected_bic": float(best_lc_bic),
        "subsample_size": int(sub_m),
        "full_sample_size": int(n_val),
        "subsampled": bool(sub_idx is not None),
        "scores": tuple(lc_scores),
    }
    if verb >= 1:
        print(
            f"  -> natural LC sweep selected K={best_lc_k} "
            f"(BIC={best_lc_bic:.2f})"
        )
    if best_lc_k == 1:
        return 1, None, None, diagnostics
    return best_lc_k, best_lc_resp, best_lc_wts, diagnostics
