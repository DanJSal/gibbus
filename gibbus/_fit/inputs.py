"""Public-boundary input validation and normalization for ``Distribution.fit``.

This module is the sole entry-point for converting raw user arguments
into the canonical internal representation used by the fitting pipeline.
All type coercion, shape validation, support defaulting/validation, and seed
extraction happen here; downstream fitting helpers assume they receive
already-validated canonical arrays.

Pipeline position
-----------------
Called from ``gibbus._api.distribution.Distribution.fit`` (via
``_Component.fit``) before the point/interval fitting pipelines.  Also
validates mixture-level inputs for multi-component fitting.  No other module
should perform user-level input validation.

Canonical sample representations
---------------------------------
After passing through this module, samples always have one of two shapes:

* **Point samples** — ``(R, 1)`` float64
* **Interval samples** — ``(R, 2)`` float64, with ``col0 <= col1``
  (swapped silently if reversed)

Sample weights (when provided) are normalized to sum to one and stored
as a 1-D float64 array of length *R*.
"""

from collections.abc import Mapping

import numpy as np

from .boundary import AUTO

# Fitted-state fields a warm-start seed must provide.
_SEED_FIELDS = (
    "q_poly",
    "boundary_amplitudes",
    "boundary_allowed",
    "support",
    "requested_poly_degree",
)


def _to_generator(rng, /):
    """Convert *rng* to a ``numpy.random.Generator``.

    Parameters
    ----------
    rng : None, int, numpy.random.Generator, or numpy.random.RandomState
        * ``None`` or ``int`` — passed to ``numpy.random.default_rng``.
        * ``Generator`` — returned unchanged.
        * ``RandomState`` — bridged by drawing a 32-bit seed from it and
          constructing a new ``Generator``.

    Returns
    -------
    numpy.random.Generator
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng.randint(0, 2 ** 32 - 1))
    return np.random.default_rng(rng)


def _normalize_sample_weights_1d(R, w, /):
    """Validate and normalize a 1-D sample-weight array.

    Parameters
    ----------
    R : int
        Expected number of weights (must match the number of samples).
    w : array_like
        Raw weights.  Must be non-negative, finite, length *R*, and not
        all zero.

    Returns
    -------
    numpy.ndarray, shape (R,), dtype float64
        Weights normalized to sum to one.

    Raises
    ------
    ValueError
        If *w* has the wrong length, contains non-finite or negative
        values, or is all zero.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if (
        w.size != int(R)
        or not np.all(np.isfinite(w))
        or np.any(w < 0)
        or np.all(w == 0)
    ):
        raise ValueError("sample_weights must be non-negative, finite, length R, and not all zero.")
    return w / np.sum(w)


def _check_spread(S, /):
    """Reject only genuinely point-degenerate datasets before fitting.

    Positive-width or censored intervals contain uncertainty even when their
    midpoints/cutpoints coincide, so midpoint spread is not a valid rejection
    criterion.  Exact point data are rejected only when every row is the same
    finite point; more subtle identifiability/coordinate failures for censored
    data are handled by the censored fitting-coordinate constructor.

    Parameters
    ----------
    S : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical samples.

    Raises
    ------
    ValueError
        If every observation is the same exact finite point.
    """
    if S.shape[1] == 1:
        x = S[:, 0]
        degenerate = float(np.ptp(x)) == 0.0
        value = float(x[0])
    else:
        point_rows = np.isfinite(S[:, 0]) & (S[:, 0] == S[:, 1])
        degenerate = bool(np.all(point_rows) and np.ptp(S[:, 0]) == 0.0)
        value = float(S[0, 0]) if degenerate else np.nan
    if degenerate:
        raise ValueError(
            f"all samples are identical (value {value!r}); a "
            "log-concave density cannot be fitted to data with zero "
            "spread. Check for a constant or placeholder column."
        )


def _canon_univariate_samples(samples, /, *, min_samples=2):
    """Convert user samples to the canonical ``(R, k)`` float64 layout.

    Accepted input shapes:

    * ``(R,)`` or ``(R, 1)`` → point samples, output shape ``(R, 1)``
    * ``(R, 2)`` → interval samples, output shape ``(R, 2)``
    * ``(R, 1, 1)`` or ``(R, 1, 2)`` → squeezed to ``(R, 1)`` or ``(R, 2)``

    For interval samples, rows with ``lower > upper`` are silently
    swapped so that ``col0 <= col1`` always holds.

    Parameters
    ----------
    samples : array_like
        Candidate univariate observations, flattened to a validated finite one-dimensional array.
    min_samples : int, optional
        Minimum accepted row count. Fitting uses two; scoring may use one.

    Returns
    -------
    S : numpy.ndarray, shape (R, 1) or (R, 2), dtype float64
        Canonical sample array.
    R : int
        Number of observations.

    Raises
    ------
    ValueError
        For unsupported shapes, complex input, NaN values, or non-finite
        point samples.
    """
    raw = np.asarray(samples)
    if raw.dtype.kind == "c":
        raise ValueError(
            "samples must be real-valued; complex input is not supported."
        )
    # The fit works on its own contiguous copy, frozen below: the caller's
    # array may be modified after (or during) the call, and fit-time caches
    # key on the array object.
    x = np.array(raw, dtype=np.float64, copy=True, order="C")

    if x.ndim == 1:
        S = x.reshape(-1, 1)
    elif x.ndim == 2:
        if x.shape[1] in (1, 2):
            S = x
        else:
            raise ValueError("samples must be (R,), (R,1), or (R,2)")
    elif x.ndim == 3:
        if x.shape[1] != 1 or x.shape[2] not in (1, 2):
            raise ValueError("samples must be (R,1,1) or (R,1,2)")
        S = np.ascontiguousarray(x[:, 0, :])
    else:
        raise ValueError("samples must be (R,), (R,1), (R,2), (R,1,1), or (R,1,2)")

    R = int(S.shape[0])
    k = int(S.shape[1])
    if k == 1:
        if not np.all(np.isfinite(S)):
            raise ValueError("point samples must contain only finite values.")
    else:
        if np.any(np.isnan(S)):
            raise ValueError("interval endpoints must not contain NaN.")

    minimum = int(min_samples)
    if minimum < 1:
        raise ValueError("min_samples must be >= 1")
    if R < minimum:
        raise ValueError(
            f"at least {minimum} samples are required, got {R}. "
            "A fitted density requires enough observations to identify scale."
        )

    if k == 2:
        L = S[:, 0]
        U = S[:, 1]
        swap = L > U
        if np.any(swap):
            S = S.copy()
            S[swap, 0] = U[swap]
            S[swap, 1] = L[swap]
        invalid_infinite_point = (~np.isfinite(S[:, 0]) | ~np.isfinite(S[:, 1])) & (S[:, 0] == S[:, 1])
        if np.any(invalid_infinite_point):
            raise ValueError("infinite censoring intervals must have positive width")

    # Read-only: the per-array grouping caches only trust frozen arrays.
    S.setflags(write=False)
    return S, R


def _default_univariate_support():
    """Return the neutral support used when the caller passes ``support=None``.

    Structural support is not identifiable from a finite sample: observing only
    positive values does not distinguish a density on ``(0, +inf)`` from one on
    the real line whose current sample happened to be positive.  Inferring a
    hard boundary from sample signs therefore makes the fitted model class jump
    when an extreme observation crosses zero.

    ``support=None`` consequently means *no support constraint*: the full real
    line.  Callers that know a structural domain such as ``(0, +inf)`` or
    ``(0, 1)`` must state it explicitly.
    """
    return (-np.inf, np.inf)


def _seed_user_support(seed, /):
    """Return a fitted-state seed's user-coordinate support.

    Parameters
    ----------
    seed : Mapping or numpy.void
        Fitted state.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Physical lower and upper support bounds.

    Raises
    ------
    ValueError
        If the seed has no valid support field.
    """
    if not _state_has_field(seed, "support"):
        raise ValueError("seed is not a fitted state (no support field)")
    support = np.asarray(seed["support"], dtype=np.float64).reshape(-1)
    if support.size != 2:
        raise ValueError("seed has invalid support field")
    return support


def _validate_support(support, /):
    """Check that *support* is a valid ``(lower, upper)`` pair.

    Parameters
    ----------
    support : array_like, shape (2,)
        ``[lower, upper]``.

    Returns
    -------
    tuple of (float, float)
        ``(lower, upper)`` as Python floats.

    Raises
    ------
    ValueError
        If either endpoint is NaN or if ``lower >= upper``.
    """
    L = float(support[0])
    U = float(support[1])
    if np.isnan(L) or np.isnan(U):
        raise ValueError("support endpoints must not be NaN")
    if L >= U:
        raise ValueError("support[0] must be < support[1]")
    return L, U


def _validate_endpoint_observations(
    samples_rk, support, log_boundary_lower, log_boundary_upper, weights, /
):
    """Reject positive-weight exact endpoint observations for enabled log bases.

    A zero-offset boundary basis has ``-log(distance)`` as its sufficient
    statistic. At an exact endpoint that statistic diverges. Positive-width
    intervals touching an endpoint are valid because the logarithm is
    integrable; only exact point observations (including zero-width interval
    rows) are rejected when the corresponding basis is enabled.

    Parameters
    ----------
    samples_rk : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical user observations before fitting-coordinate mapping.
    support : tuple of (float, float)
        User-coordinate support.
    log_boundary_lower, log_boundary_upper : bool
        Whether each zero-offset boundary basis is allowed.
    weights : numpy.ndarray or None
        Normalized observation weights. Zero-weight rows are ignored.

    Raises
    ------
    ValueError
        If a positive-weight exact endpoint observation conflicts with an
        enabled zero-offset logarithmic boundary basis.
    """
    S = np.asarray(samples_rk, dtype=np.float64)
    active = np.ones(S.shape[0], dtype=bool) if weights is None else np.asarray(weights) > 0.0
    if S.shape[1] == 1:
        point = active
        value = S[:, 0]
    else:
        point = active & (S[:, 0] == S[:, 1])
        value = S[:, 0]
    L, U = map(float, support)
    if bool(log_boundary_lower) and np.isfinite(L) and np.any(point & (value == L)):
        raise ValueError(
            "exact observations at the lower support endpoint are incompatible "
            "with log_boundary_lower=True; disable that boundary basis or use "
            "a positive-width censoring interval"
        )
    if bool(log_boundary_upper) and np.isfinite(U) and np.any(point & (value == U)):
        raise ValueError(
            "exact observations at the upper support endpoint are incompatible "
            "with log_boundary_upper=True; disable that boundary basis or use "
            "a positive-width censoring interval"
        )


def _is_poly_degree_admissible(poly_degree, support, /):
    """Check whether *poly_degree* is admissible for the given *support*.

    On full-infinite support ``(-inf, +inf)``, odd polynomial degrees are
    structurally inadmissible because ``q''`` would have odd degree, whose
    leading term eventually dominates and becomes negative.

    Parameters
    ----------
    poly_degree : int
        Polynomial degree used for the component potential fit.
    support : tuple of (float, float)
        Support bounds used to determine admissible polynomial degrees.

    Returns
    -------
    bool
    """
    L, U = float(support[0]), float(support[1])
    if np.isneginf(L) and np.isposinf(U):
        return int(poly_degree) % 2 == 0
    return True


def _admissible_degrees(support, target_degree, /):
    """Return the sorted list of admissible degrees from 2 up to *target_degree*.

    Parameters
    ----------
    support : tuple of (float, float)
        Support bounds used to determine admissible polynomial degrees.
    target_degree : int
        Requested polynomial degree around which admissible degrees are enumerated.

    Returns
    -------
    list of int
    """
    L, U = float(support[0]), float(support[1])
    full_infinite = np.isneginf(L) and np.isposinf(U)
    if full_infinite:
        return [d for d in range(2, int(target_degree) + 1) if d % 2 == 0]
    else:
        return list(range(2, int(target_degree) + 1))


def _state_has_field(state, key, /):
    """Return whether a mapping or structured scalar contains a field.

    Parameters
    ----------
    state : Mapping or numpy.void
        Candidate fitted state.
    key : str
        Field name.

    Returns
    -------
    bool
    """
    names = getattr(getattr(state, "dtype", None), "names", None)
    if names is not None:
        return key in names
    if isinstance(state, Mapping):
        return key in state
    return False


def _boundary_policy(caller, endpoint, /):
    """Resolve one side's boundary-term policy for a cold fit.

    Parameters
    ----------
    caller : bool, ``"auto"`` or None
        The caller's flag; ``None`` or ``"auto"`` leaves the choice open.
    endpoint : float
        The support endpoint on that side.

    Returns
    -------
    bool or str
        The caller's flag, else ``"auto"`` on a finite endpoint and ``False``
        on an infinite one.
    """
    if caller is not None and not (isinstance(caller, str) and caller == AUTO):
        return bool(caller)
    return AUTO if np.isfinite(float(endpoint)) else False


def _resolve_endpoint_observations(samples_rk, support, lower, upper, weights, /):
    """Validate explicit terms against endpoint observations; settle automatic ones.

    An exact observation at a finite endpoint has zero density under that
    side's term, so an explicit term raises (see
    ``_validate_endpoint_observations``) and an automatic one resolves to no
    term.

    Parameters
    ----------
    samples_rk : numpy.ndarray, shape (R, 1) or (R, 2)
        Canonical sample rows.
    support : tuple of (float, float)
        User support.
    lower, upper : bool or ``"auto"``
        Boundary policies.
    weights : numpy.ndarray or None
        Observation weights.

    Returns
    -------
    tuple
        ``(lower, upper)`` policies.
    """
    _validate_endpoint_observations(
        samples_rk, support, lower is True, upper is True, weights
    )
    if lower == AUTO:
        try:
            _validate_endpoint_observations(samples_rk, support, True, False, weights)
        except ValueError:
            lower = False
    if upper == AUTO:
        try:
            _validate_endpoint_observations(samples_rk, support, False, True, weights)
        except ValueError:
            upper = False
    return lower, upper


def _normalize_univariate_fit_inputs(
    ComponentType,
    samples,
    poly_degree,
    support,
    log_boundary_lower,
    log_boundary_upper,
    verbose,
    suppress_warnings,
    init_from,
    sample_weights,
    /,
):
    """Validate and normalize inputs for one component fit.

    ``log_boundary_lower`` and ``log_boundary_upper`` mean *allow* the
    corresponding fixed zero-offset logarithmic basis. Their amplitudes are
    direct nonnegative natural parameters and may fit to exactly zero.

    Parameters
    ----------
    ComponentType : type
        Internal component class, passed to avoid a circular import.
    samples : array_like
        Point or interval observations.
    poly_degree : int, ``"auto"``, or None
        ``None`` selects automatic degree without a seed and inherits the
        seed's requested degree with a seed.
    support : tuple of (float, float) or None
        Structural user-coordinate support. A seed overrides it.
    log_boundary_lower, log_boundary_upper : bool or None
        Whether each finite endpoint may carry a zero-offset logarithmic
        boundary term. ``None`` lets the data decide on a finite endpoint of
        a cold fit (``"auto"``; see :mod:`gibbus._fit.boundary`), means no
        term on an infinite one, and inherits ``boundary_allowed`` from a
        seed.
    verbose : int
        Verbosity level for fitting progress and diagnostics.
    suppress_warnings : bool
        Whether numerical fitting warnings should be suppressed.
    init_from : _Component, Mapping, or None
        Fitted-state warm-start seed.
    sample_weights : array_like or None
        Optional nonnegative observation weights.

    Returns
    -------
    dict
        Canonical validated fitting inputs.

    Raises
    ------
    ValueError
        For invalid shapes/support/degrees/seeds, out-of-support observations,
        or exact endpoint observations conflicting with an enabled log basis.
    """
    seed_state = None
    caller_log_lower = log_boundary_lower
    caller_log_upper = log_boundary_upper
    is_auto_degree = isinstance(poly_degree, str) and poly_degree.lower() == "auto"

    if init_from is not None:
        seed = init_from.data if isinstance(init_from, ComponentType) else init_from
        missing = [
            name for name in _SEED_FIELDS if not _state_has_field(seed, name)
        ]
        if missing:
            raise ValueError(
                "init_from must be a fitted state; missing fields: "
                + ", ".join(missing)
            )
        seed_degree = int(seed["requested_poly_degree"])
        if seed_degree < 2:
            raise ValueError("seed has invalid polynomial degree (<2)")
        if poly_degree is None:
            poly_degree = seed_degree
        elif not is_auto_degree:
            poly_degree = int(poly_degree)
            if poly_degree < 2:
                raise ValueError("poly_degree must be >= 2")

        supp = _validate_support(_seed_user_support(seed))
        allowed = np.asarray(seed["boundary_allowed"], dtype=bool).reshape(-1)
        if allowed.size != 2:
            raise ValueError("seed has invalid boundary_allowed field")
        log_boundary_lower = bool(allowed[0]) if caller_log_lower is None else bool(caller_log_lower)
        log_boundary_upper = bool(allowed[1]) if caller_log_upper is None else bool(caller_log_upper)
        seed_state = seed
    else:
        if poly_degree is None:
            poly_degree = "auto"
            is_auto_degree = True
        elif not is_auto_degree:
            poly_degree = int(poly_degree)
            if poly_degree < 2:
                raise ValueError("poly_degree must be >= 2")
        supp = _default_univariate_support() if support is None else _validate_support(support)
        log_boundary_lower = _boundary_policy(caller_log_lower, supp[0])
        log_boundary_upper = _boundary_policy(caller_log_upper, supp[1])

    S, R = _canon_univariate_samples(samples)
    L, U = map(float, supp)
    if S.shape[1] == 1:
        x = S[:, 0]
        if (np.isfinite(L) and np.any(x < L)) or (np.isfinite(U) and np.any(x > U)):
            origin = "seed's" if init_from is not None else "provided"
            raise ValueError(f"samples fall outside the {origin} support.")
    else:
        Ls, Us = S[:, 0], S[:, 1]
        if (np.isfinite(L) and np.any(Ls < L)) or (np.isfinite(U) and np.any(Us > U)):
            origin = "seed's" if init_from is not None else "provided"
            raise ValueError(f"interval samples fall outside the {origin} support.")

    _check_spread(S)
    weights = None if sample_weights is None else _normalize_sample_weights_1d(R, sample_weights)

    if log_boundary_lower is True and not np.isfinite(L):
        raise ValueError("log_boundary_lower=True requires a finite lower support endpoint")
    if log_boundary_upper is True and not np.isfinite(U):
        raise ValueError("log_boundary_upper=True requires a finite upper support endpoint")
    log_boundary_lower, log_boundary_upper = _resolve_endpoint_observations(
        S, supp, log_boundary_lower, log_boundary_upper, weights
    )

    if not is_auto_degree and not _is_poly_degree_admissible(poly_degree, supp):
        raise ValueError(
            f"poly_degree={poly_degree} is odd, which is inadmissible on "
            "full-infinite support (-inf, +inf). Use an even degree."
        )

    return {
        "samples_rk": S,
        "R": R,
        "support": supp,
        "poly_degree": poly_degree,
        "log_boundary_lower": log_boundary_lower,
        "log_boundary_upper": log_boundary_upper,
        "verbose": int(verbose),
        "suppress_warnings": bool(suppress_warnings),
        "weights": weights,
        "seed_state": seed_state,
    }


def _normalize_mixture_fit_inputs(
    samples,
    n_components,
    support,
    component_options,
    verbose,
    suppress_warnings,
    /,
    sample_weights=None,
):
    """Validate and normalize all inputs for multi-component fitting.

    Parameters
    ----------
    samples : array_like
        Raw samples — see :func:`_canon_univariate_samples`.
    n_components : int or ``"auto"``
        Number of mixture components.  Must be >= 1 or the string
        ``"auto"`` to enable automatic BIC-based selection.
    support : tuple of (float, float) or None
        Shared support for every component.  ``None`` means ``(-inf, +inf)``.
    component_options : list of dict or None
        Per-component keyword arguments forwarded to component fitting.
        Only ``poly_degree`` is allowed as a per-component override;
        keys ``'support'``, ``'sample_weights'``, ``'init_from'``,
        ``'log_boundary_lower'`` and ``'log_boundary_upper'`` are forbidden
        (these are global across all components).  Must be ``None``
        when ``n_components="auto"``.
    verbose : int
        Verbosity level for fitting progress and diagnostics.
    suppress_warnings : bool
        Whether numerical fitting warnings should be suppressed.
    sample_weights : array_like or None, optional
        Non-negative observation weights of length *R*.  Normalized to
        sum to one and returned under the ``weights`` key.

    Returns
    -------
    dict
        ``samples_rk``, ``R``, ``n_components``, ``support``,
        ``component_options``, ``verbose``, ``suppress_warnings``,
        ``weights``.
        When automatic selection is active, ``n_components`` is the
        string ``"auto"`` and ``component_options`` is ``None``.

    Raises
    ------
    ValueError
        For any invalid input.
    """
    is_auto = isinstance(n_components, str) and n_components.lower() == "auto"
    if not is_auto:
        n_components = int(n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1 or 'auto'.")

    S, R = _canon_univariate_samples(samples)

    if support is None:
        supp = _default_univariate_support()
    else:
        supp = _validate_support(support)

    L, U = float(supp[0]), float(supp[1])
    if S.shape[1] == 1:
        x = S[:, 0]
        if (np.isfinite(L) and np.any(x < L)) or (np.isfinite(U) and np.any(x > U)):
            raise ValueError("samples fall outside the provided support.")
    else:
        Ls, Us = S[:, 0], S[:, 1]
        if (np.isfinite(L) and np.any(Ls < L)) or (np.isfinite(U) and np.any(Us > U)):
            raise ValueError("interval samples fall outside the provided support.")

    _check_spread(S)

    # Each component needs enough data to be identifiable at all;
    # without this the failure surfaces deep inside the initializers.
    if not is_auto and n_components > 1 and R < 2 * n_components:
        raise ValueError(
            f"n_components={n_components} requires at least "
            f"{2 * n_components} samples, got {R}. Reduce "
            "n_components or supply more data."
        )

    if is_auto:
        if component_options is not None:
            raise ValueError(
                "component_options must be None when n_components='auto' "
                "(the number of components is not known in advance)."
            )
        # Leave component_options as None; the API-layer fitting controller
        # creates the list once K has been determined.
    else:
        if component_options is None:
            component_options = [{}] * n_components
        else:
            component_options = list(component_options)

        if len(component_options) != n_components:
            raise ValueError(
                f"component_options must be a list of length n_components "
                f"({n_components}), got {len(component_options)}."
            )

        _forbidden_keys = {
            "support", "sample_weights", "init_from",
            "log_boundary_lower", "log_boundary_upper",
        }
        for i, opts in enumerate(component_options):
            if not isinstance(opts, dict):
                raise ValueError(
                    f"component_options[{i}] must be a dict, got {type(opts).__name__}."
                )
            bad = _forbidden_keys & set(opts)
            if bad:
                raise ValueError(
                    f"component_options[{i}] must not contain {bad}; "
                    f"boundary flags are global, 'support' is shared, "
                    f"'sample_weights' is managed by the EM loop, and "
                    f"'init_from' is handled at the Distribution level."
                )

    w = None if sample_weights is None else _normalize_sample_weights_1d(
        R, sample_weights
    )

    return {
        "samples_rk": S,
        "R": R,
        "n_components": "auto" if is_auto else n_components,
        "support": supp,
        "component_options": component_options,
        "verbose": int(verbose),
        "suppress_warnings": bool(suppress_warnings),
        "weights": w,
    }


def _coerce_sample_size(size, /):
    """Validate a scalar ``size`` argument and return it as a non-negative int.

    Integrality is checked before integer conversion so non-integral floats
    and numeric-looking strings cannot be silently truncated or coerced.

    Integral floats (``4.0``) are accepted: that is a common and
    unambiguous idiom.  ``bool`` is rejected despite being an ``int``
    subclass, because ``sample(size=True)`` returning one sample is far
    more likely to be a bug at the call site than an intention.

    Parameters
    ----------
    size : object
        The user-supplied ``size`` argument (never ``None``; callers
        handle that case before calling).

    Returns
    -------
    int
        The validated non-negative sample count.

    Raises
    ------
    TypeError
        If *size* is a bool, a string, or any non-numeric type.
    ValueError
        If *size* is not scalar, not integral, or negative.
    """
    _MSG = "size must be None or a non-negative integer"

    if isinstance(size, bool):
        raise TypeError(f"{_MSG}, not bool")

    size_arr = np.asarray(size)
    if size_arr.ndim != 0:
        raise ValueError(_MSG)
    if size_arr.dtype.kind == "b":
        raise TypeError(f"{_MSG}, not bool")
    if size_arr.dtype.kind not in ("i", "u", "f"):
        raise TypeError(f"{_MSG}, not {type(size).__name__}")

    value = size_arr.item()
    if size_arr.dtype.kind == "f" and (not np.isfinite(value) or value != int(value)):
        raise ValueError(f"{_MSG}, got {value!r}")

    n = int(value)
    if n < 0:
        raise ValueError(_MSG)
    return n
