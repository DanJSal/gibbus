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

from dataclasses import dataclass

import numpy as np

from .._defaults import (
    AUTO_LC_SUBSAMPLE_MIN_N,
    AUTO_LC_SUBSAMPLE_PER_K,
    AUTO_LC_SUBSAMPLE_SIZE,
    AUTO_LC_SWEEP_MAX_K,
    AUTO_LC_VALIDATION_DEGREES,
)
from .._model.basis_spec import BasisSpec
from .boundary import AUTO


@dataclass(frozen=True)
class _ComponentSelectionPolicy:
    """Resolved component-count search controls.

    Parameters
    ----------
    ceiling, initial_hi : int
        Maximum candidate count and upper end of the initial screening sweep.
    subsample_size : int
        Number of canonical observation rows used for selection.
    degrees : tuple of int
        Admissible fixed degrees used for screening.
    degree_policy : int or str
        Requested refinement degree, with automatic selection spelled ``"auto"``.
    lower_boundary, upper_boundary : bool or str
        Boundary policies after full-data endpoint exclusions.
    """

    ceiling: int
    initial_hi: int
    subsample_size: int
    degrees: tuple[int, ...]
    degree_policy: int | str
    lower_boundary: bool | str
    upper_boundary: bool | str


def _prepare_component_selection_policy(
    n_rows,
    support,
    k_modes,
    effective_k_max,
    subsample,
    degree_policy,
    lower_boundary,
    upper_boundary,
    /,
):
    """Resolve raw selection controls before entering the numerical search.

    Parameters
    ----------
    n_rows : int
        Full canonical observation count.
    support : tuple of float
        Validated physical support.
    k_modes, effective_k_max : int
        Mode proposal and its component-count limit.
    subsample : str, int or bool
        Public selection-sample policy.
    degree_policy : int, str or None
        Public polynomial-degree policy.
    lower_boundary, upper_boundary : bool or str
        Prepared full-data boundary policies.
    """
    ceiling = max(1, min(effective_k_max, AUTO_LC_SWEEP_MAX_K, max(n_rows // 50, 2)))
    initial_hi = min(ceiling, max(2 * k_modes, 1))
    if degree_policy is None or degree_policy == "auto":
        degree_policy = "auto"
        degrees = tuple(
            d
            for d in AUTO_LC_VALIDATION_DEGREES
            if _is_poly_degree_admissible(d, support)
        )
    else:
        degree_policy = _coerce_poly_degree(degree_policy)
        degrees = (degree_policy,)
        if not _is_poly_degree_admissible(degree_policy, support):
            raise ValueError("requested screening degree is not admissible on support")
    if not degrees:
        raise ValueError("no admissible screening degrees")
    if subsample is False or subsample == 0:
        size = n_rows
    elif isinstance(subsample, str):
        if subsample != "auto":
            raise ValueError(
                f"subsample must be 'auto', an int, or False, got {subsample!r}."
            )
        size = (
            max(AUTO_LC_SUBSAMPLE_SIZE, initial_hi * AUTO_LC_SUBSAMPLE_PER_K)
            if n_rows > AUTO_LC_SUBSAMPLE_MIN_N
            else n_rows
        )
    else:
        size = int(subsample)
        if size < 2:
            raise ValueError(f"subsample must be at least 2, got {size}.")
    return _ComponentSelectionPolicy(
        ceiling,
        initial_hi,
        min(size, n_rows),
        degrees,
        degree_policy,
        lower_boundary,
        upper_boundary,
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
        return np.random.default_rng(rng.randint(0, 2**32 - 1))
    return np.random.default_rng(rng)


def _normalize_degree_policies(degree, count, /):
    """Canonicalize a raw shared or per-component polynomial-degree policy.

    Parameters
    ----------
    degree : int, str, None or sequence
        Shared degree policy or one entry per component. ``None`` and
        case-insensitive ``"auto"`` select automatic degree growth.
    count : int
        Prepared positive component count.

    Returns
    -------
    tuple of int or str
        Exactly one integral degree or canonical ``"auto"`` per component.
    """
    if degree is None or isinstance(degree, str) or np.isscalar(degree):
        values = (degree,) * count
    else:
        values = tuple(degree)
        if len(values) != count:
            raise ValueError(
                "per-component degree policies must match the component count"
            )
    policies = []
    for value in values:
        if value is None:
            policies.append("auto")
        elif isinstance(value, str):
            if value.lower() != "auto":
                raise ValueError("degree policy must be an integer or 'auto'")
            policies.append("auto")
        else:
            policies.append(_coerce_poly_degree(value))
    return tuple(policies)


def _coerce_poly_degree(poly_degree, /):
    """Return one finite integral polynomial degree of at least two.

    Parameters
    ----------
    poly_degree : object
        Candidate polynomial degree supplied at a user-input boundary.

    Returns
    -------
    int
        Canonical integral degree.

    Raises
    ------
    ValueError
        If the value is boolean, non-numeric, non-finite, non-integral, or less
        than two.
    """
    if isinstance(poly_degree, bool):
        raise ValueError(f"poly_degree must be an integer >= 2, got {poly_degree!r}")
    try:
        value = float(poly_degree)
    except (TypeError, ValueError):
        raise ValueError(
            f"poly_degree must be an integer >= 2, got {poly_degree!r}"
        ) from None
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"poly_degree must be an integer >= 2, got {poly_degree!r}")
    degree = int(value)
    if degree < 2:
        raise ValueError("poly_degree must be >= 2")
    return degree


def _normalize_sample_weight_1d(R, w, /):
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
        raise ValueError(
            "sample_weight must be non-negative, finite, length R, and not all zero."
        )
    scale = float(np.max(w))
    scaled = w / scale
    return scaled / np.sum(scaled)


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
        Candidate univariate point or interval observations. Point observations
        must be finite; interval endpoints may be infinite for censoring but
        must not contain NaN.
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
        raise ValueError("samples must be real-valued; complex input is not supported.")
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
        invalid_infinite_point = (~np.isfinite(S[:, 0]) | ~np.isfinite(S[:, 1])) & (
            S[:, 0] == S[:, 1]
        )
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
    raw = np.asarray(support, dtype=np.float64).reshape(-1)
    if raw.size != 2:
        raise ValueError("support must contain exactly two endpoints")
    L = float(raw[0])
    U = float(raw[1])
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
    active = (
        np.ones(S.shape[0], dtype=bool)
        if weights is None
        else np.asarray(weights) > 0.0
    )
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


def _normalize_mixture_fit_inputs(
    samples,
    n_components,
    support,
    component_options,
    verbose,
    suppress_warnings,
    /,
    sample_weight=None,
    allow_odd_degree=False,
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
        keys ``'support'``, ``'sample_weight'``, ``'init_from'``,
        ``'log_boundary_lower'`` and ``'log_boundary_upper'`` are forbidden
        (these are global across all components).  Must be ``None``
        when ``n_components="auto"``.
    verbose : int
        Verbosity level for fitting progress and diagnostics.
    suppress_warnings : bool
        Whether numerical fitting warnings should be suppressed.
    sample_weight : array_like or None, optional
        Non-negative observation weights of length *R*.  Normalized to
        sum to one and returned under the ``weights`` key.
    allow_odd_degree : bool, optional
        Let a general basis establish properness of odd component degrees.
        The convenience polynomial-only interface keeps its parity rule.

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
            component_options = [{} for _ in range(n_components)]
        else:
            component_options = list(component_options)

        if len(component_options) != n_components:
            raise ValueError(
                f"component_options must be a list of length n_components "
                f"({n_components}), got {len(component_options)}."
            )

        _allowed_keys = {"poly_degree"}
        normalized_options = []
        for i, opts in enumerate(component_options):
            if not isinstance(opts, dict):
                raise ValueError(
                    f"component_options[{i}] must be a dict, got {type(opts).__name__}."
                )
            unknown = set(opts) - _allowed_keys
            if unknown:
                raise ValueError(
                    f"component_options[{i}] must not contain unsupported keys {unknown}; "
                    "only 'poly_degree' may be set per component."
                )
            normalized = dict(opts)
            if "poly_degree" in normalized:
                value = normalized["poly_degree"]
                if isinstance(value, str) and value.lower() == "auto":
                    normalized["poly_degree"] = "auto"
                else:
                    degree = _coerce_poly_degree(value)
                    if not allow_odd_degree and not _is_poly_degree_admissible(
                        degree, supp
                    ):
                        raise ValueError(
                            f"component_options[{i}].poly_degree={degree} is odd, "
                            "which is inadmissible on support infinite at both ends."
                        )
                    normalized["poly_degree"] = degree
            normalized_options.append(normalized)
        component_options = normalized_options

    w = None if sample_weight is None else _normalize_sample_weight_1d(R, sample_weight)

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


@dataclass(frozen=True)
class _BasisInputs:
    """Validated G3 fit inputs (G1 ``_normalize_univariate_fit_inputs`` role).

    Parameters
    ----------
    spec : BasisSpec
        Basis request (``"auto"`` fields still open).
    samples : numpy.ndarray
        Canonical observations, shape ``(R, 1)`` (points) or ``(R, 2)``
        (ordered interval rows; infinite ends are censoring), read-only.
    weights : numpy.ndarray or None
        Weights normalized to sum to one (``None`` for equal weights).
    log_policy : tuple
        ``(lower, upper)`` boundary-term policy: ``True``, ``False`` or
        ``"auto"`` after G1's endpoint-observation rule (an automatic term
        at an infinite end, or at a finite end carrying an exact point
        observation, resolves to ``False``).
    effective_n : float
        Kish effective sample size.
    """

    spec: object
    samples: np.ndarray
    weights: np.ndarray | None
    log_policy: tuple
    effective_n: float

    @property
    def intervals(self):
        """Whether the observations are interval rows."""
        return self.samples.shape[1] == 2


def _normalize_basis_inputs(spec, samples, weights=None, /):
    """Validate a G3 fit request with G1's input rules.

    G1 owner adapted: ``_normalize_univariate_fit_inputs`` (sample layout,
    weights, spread, support and endpoint-observation rules, boundary
    policy) for a :class:`BasisSpec` instead of G1's degree/boundary flags.

    Parameters
    ----------
    spec : BasisSpec
        Basis request.
    samples : array_like
        Point observations, shape ``(R,)`` or ``(R, 1)``, or interval rows,
        shape ``(R, 2)`` (G1 layouts; reversed rows are swapped).
    weights : array_like or None
        Nonnegative observation weights.

    Returns
    -------
    _BasisInputs

    Raises
    ------
    TypeError
        If ``spec`` is not a ``BasisSpec``.
    ValueError
        If the observations leave the closed support, points are not finite
        or all coincide, an infinite row has zero width, the weights are
        invalid, or an explicit log term meets an exact observation at its
        endpoint.
    """

    if not isinstance(spec, BasisSpec):
        raise TypeError("spec must be a BasisSpec")
    s, r = _canon_univariate_samples(samples)
    _check_spread(s)
    w = None if weights is None else _normalize_sample_weight_1d(r, weights)
    if w is not None and np.all(w == w[0]):
        # Equal weights carry no information: the unweighted fit, exactly
        # (the weighted median is the lower median, the unweighted one the
        # average of the middle pair).
        w = None
    lower, upper = (float(v) for v in spec.support)
    if np.any(s[:, 0] < lower) or np.any(s[:, -1] > upper):
        kind = "point samples" if s.shape[1] == 1 else "interval samples"
        raise ValueError(f"{kind} must lie inside the closed support")
    policy = _resolve_endpoint_observations(
        s,
        (lower, upper),
        _boundary_policy(spec.log_lower, lower),
        _boundary_policy(spec.log_upper, upper),
        w,
    )
    n_eff = float(r) if w is None else float(1.0 / np.dot(w, w))
    return _BasisInputs(spec, s, w, tuple(policy), n_eff)
