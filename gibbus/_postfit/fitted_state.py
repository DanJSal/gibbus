"""Shared geometry, metadata and validation of fitted component states.

The state records the fixed fitting coordinate, the basis record and
coefficients, solver diagnostics and the downstream numerical
representations used by public evaluation and warm starts; ``basis_state._pack_basis_state`` packs it.  It is a runtime
representation, not the durable persistence schema; that boundary is defined
in :mod:`gibbus._serialization`.
"""

import math

import numpy as np

from .._fit.boundary import _weakly_identified_sides


def _shared_model_geometry(comp_states, /):
    """Validate shared component geometry and return model-level dimensions.

    Parameters
    ----------
    comp_states : sequence of numpy.void
        Packed component states sharing support and boundary amplitudes.

    Returns
    -------
    dict
        ``allowed``, ``amplitudes`` and ``active`` boundary arrays in public
        lower/upper order, plus the integer ``n_parameters`` and
        ``n_face_parameters`` implied by the components.

    Raises
    ------
    ValueError
        If the component states are invalid or disagree on shared geometry.
    """
    if not comp_states:
        raise ValueError("model state must contain at least one component")
    for state in comp_states:
        _check_state_invariants(state)
    first = comp_states[0]
    allowed = np.asarray(first["boundary_allowed"], dtype=bool)
    amplitudes = np.asarray(first["boundary_amplitudes"], dtype=float)

    # Endpoint-log amplitudes are shared, so every component carries the same
    # values; a log is active when its amplitude is positive (a zero amplitude
    # lies on its zero face).  Dimensions count each component's private
    # coefficients (``basis_n_face`` on the fit's face), the shared amplitudes
    # once and ``K - 1`` weights.
    def anchors(state):
        """Potential anchors of ``state`` in fitted base coordinates."""
        return float(state["fit_center"]) + float(state["fit_scale"]) * np.asarray(
            state["canonical_support"], dtype=float
        )

    for state in comp_states:
        if (
            not np.array_equal(state["boundary_allowed"], allowed)
            or not np.array_equal(state["boundary_amplitudes"], amplitudes)
            or not np.array_equal(state["support"], first["support"])
            or not np.allclose(anchors(state), anchors(first), rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("model components disagree on shared boundary geometry")
    active = allowed & (amplitudes > 0.0)
    shared = int(allowed.sum())
    k = len(comp_states)
    n_parameters = (
        sum(np.asarray(state["basis_theta"]).size - shared for state in comp_states)
        + shared
        + k
        - 1
    )
    n_face = (
        sum(int(state["basis_n_face"]) for state in comp_states)
        + int(active.sum())
        + k
        - 1
    )
    return {
        "allowed": allowed,
        "amplitudes": amplitudes,
        "active": active,
        "n_parameters": n_parameters,
        "n_face_parameters": n_face,
    }


def _metadata_conflict(geometry, metadata, /):
    """Return why declared model metadata disagrees with the components.

    Parameters
    ----------
    geometry : Mapping
        Result of :func:`_shared_model_geometry` for the model's components.
    metadata : Mapping
        Declared provenance, dimensions and shared-boundary inference with the
        keys produced by :func:`_model_metadata`.

    Returns
    -------
    str or None
        A description of the first disagreement, or ``None`` when the
        metadata is consistent with the components.
    """
    if str(metadata["provenance"]) not in ("fitted", "derived"):
        return "model provenance must be 'fitted' or 'derived'"
    for key in ("n_parameters", "n_face_parameters"):
        value = metadata[key]
        if not np.isfinite(value) or int(value) != value or int(value) != geometry[key]:
            return f"model {key} disagrees with fitted component dimensions"
    allowed = geometry["allowed"]
    shared = metadata["shared_boundary"]
    for key in ("allowed", "active"):
        value = np.asarray(shared[key])
        if value.shape != (2,) or not np.array_equal(value, geometry[key]):
            return f"shared boundary {key} disagrees with components"
    declared = np.asarray(shared["amplitudes"], dtype=float)
    if declared.shape != (2,) or not np.array_equal(
        declared[allowed], geometry["amplitudes"][allowed]
    ):
        return "shared boundary amplitudes disagree with components"
    if np.any(~(np.isnan(declared[~allowed]) | (declared[~allowed] == 0.0))):
        return "excluded shared amplitudes must be zero or unavailable"
    errors = np.asarray(shared["standard_errors"], dtype=float)
    p_values = np.asarray(shared["p_values"], dtype=float)
    if errors.shape != (2,) or np.any(errors < 0):
        return "shared standard errors must be nonnegative or unavailable"
    if p_values.shape != (2,) or np.any((p_values < 0) | (p_values > 1)):
        return "shared p-values must lie in [0, 1] or be unavailable"
    return None


def _model_metadata(comp_states, metadata=None):
    """Validate shared fitted geometry and return detached model metadata.

    Parameters
    ----------
    comp_states : sequence of numpy.void
        Packed component states sharing support and boundary amplitudes.
    metadata : Mapping or None, optional
        Declared model dimensions and shared inference. ``None`` creates
        derived-state metadata with unavailable inferential errors and p-values.

    Returns
    -------
    dict
        Detached model dimensions, provenance and shared-boundary inference.

    Raises
    ------
    ValueError
        If component geometry or declared metadata is inconsistent.
    """
    geometry = _shared_model_geometry(comp_states)
    allowed = geometry["allowed"]
    amplitudes = geometry["amplitudes"]
    active = geometry["active"]
    if metadata is None:
        metadata = {
            "provenance": "derived",
            "n_parameters": geometry["n_parameters"],
            "n_face_parameters": geometry["n_face_parameters"],
            "shared_boundary": {
                "allowed": allowed,
                "amplitudes": amplitudes,
                "active": active,
                "standard_errors": (np.nan, np.nan),
                "p_values": (np.nan, np.nan),
            },
        }
    conflict = _metadata_conflict(geometry, metadata)
    if conflict is not None:
        raise ValueError(conflict)
    provenance = str(metadata["provenance"])
    errors = np.asarray(metadata["shared_boundary"]["standard_errors"], dtype=float)
    p_values = np.asarray(metadata["shared_boundary"]["p_values"], dtype=float)
    if provenance == "derived":
        errors = np.full(2, np.nan)
        p_values = np.full(2, np.nan)
    return {
        "provenance": provenance,
        "n_parameters": int(metadata["n_parameters"]),
        "n_face_parameters": int(metadata["n_face_parameters"]),
        "shared_boundary": {
            "allowed": tuple(map(bool, allowed)),
            "amplitudes": tuple(map(float, amplitudes)),
            "active": tuple(map(bool, active)),
            "standard_errors": tuple(map(float, errors)),
            "p_values": tuple(map(float, p_values)),
            "weakly_identified": _weakly_identified_sides(amplitudes, errors),
        },
    }


# Fields every fitted state must carry; the spectral CDF/PPF evaluators check
# their own array geometry when they are rebuilt.
_STATE_FIELDS = (
    "boundary_amplitudes",
    "boundary_allowed",
    "support",
    "canonical_support",
    "window",
    "canonical_mode",
    "mode",
    "median",
    "mean",
    "var",
    "std",
    "skew",
    "kurt",
    "raw_moments",
    "canonical_raw_moments",
    "fit_center",
    "fit_scale",
    "nll",
    "cdf_map_params",
    "cdf_npanels",
    "cdf_coeff_stride",
    "cdf_breaks",
    "cdf_offsets",
    "cdf_coeffs",
    "ppf_pmin",
    "ppf_pmax",
    "ppf_npanels",
    "ppf_coeff_stride",
    "ppf_breaks_r",
    "ppf_breaks_z",
    "ppf_coeffs",
    "boundary_standard_errors",
    "boundary_p_values",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "optimizer_n_iterations",
    "optimizer_n_evaluations",
    "optimizer_subproblem_iterations",
    "optimizer_decrease_bound",
    "basis_model",
    "basis_theta",
    "basis_log_z",
    "basis_log_z_model",
    "basis_log_z_canonical",
    "basis_location",
    "basis_shape",
    "basis_shape_tau",
    "basis_n_face",
)


def _structured_scalar(data, /):
    """Convert a mapping to a structured NumPy scalar.

    Parameters
    ----------
    data : Mapping
        Scalar and fixed-shape array fields.

    Returns
    -------
    numpy.void
        Packed structured scalar.
    """
    dtype = []
    for key, value in data.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            dtype.append((key, arr.dtype))
        else:
            dtype.append((key, arr.dtype, arr.shape))
    out = np.zeros((), dtype=dtype)
    for key, value in data.items():
        out[key] = value
    return out


def _check_finite_increasing(values, name, /):
    """Raise unless *values* are finite and strictly increasing.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate breakpoints, already trimmed to the live panel count.
    name : str
        Field name, used in the error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any entry is non-finite or the sequence does not increase.
    """
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")


def _check_state_invariants(struct, /):
    """Reject a fitted state whose stored values cannot describe a density.

    The compiled evaluators already reject panel counts, strides, and array
    lengths that would not reconstruct, which is what keeps a corrupted file
    from indexing out of bounds.  This adds the value-level invariants: a
    state can be structurally intact and still hold a non-positive scale,
    non-monotone panel breakpoints, or non-finite coefficients, and such a
    state evaluates silently to nonsense rather than failing.

    Parameters
    ----------
    struct : numpy.ndarray or numpy.void
        Packed single-component fitted state.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a required field is missing or any stored value violates an
        invariant of a fitted state.
    """
    names = getattr(getattr(struct, "dtype", None), "names", None) or ()
    missing = [name for name in _STATE_FIELDS if name not in names]
    if missing:
        raise ValueError(
            "not a gibbus fitted state; missing fields: " + ", ".join(missing)
        )

    def field(name):
        return np.asarray(struct[name], dtype=np.float64).ravel()

    if np.asarray(struct).shape != () or struct.dtype.hasobject:
        raise ValueError("component state must be a non-object structured scalar")
    forbidden = {"mu", "sigma", "pullback", "default_space"}.intersection(names)
    if forbidden:
        raise ValueError("component state cannot contain presentation transforms")
    scale = float(struct["fit_scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"state fit_scale must be finite and positive, got {scale}")
    if not np.isfinite(float(struct["fit_center"])):
        raise ValueError("state fit_center must be finite")
    theta = field("basis_theta")
    if not (np.all(np.isfinite(theta)) and math.isfinite(float(struct["basis_log_z"]))):
        raise ValueError("state coefficients and log Z must be finite")
    for name in (
        "boundary_amplitudes",
        "boundary_allowed",
        "boundary_standard_errors",
        "boundary_p_values",
        "support",
        "canonical_support",
        "window",
    ):
        if np.asarray(struct[name]).shape != (2,):
            raise ValueError(f"state {name} must have shape (2,)")
    amplitudes = field("boundary_amplitudes")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("state boundary_amplitudes must be finite and non-negative")
    for name in ("support", "canonical_support", "window"):
        bounds = field(name)
        if np.any(np.isnan(bounds)) or not bounds[0] < bounds[1]:
            raise ValueError(f"state {name} must satisfy lower < upper")
    if not np.all(np.isfinite(field("window"))):
        raise ValueError("state window must be finite")
    support = field("support")
    center = float(struct["fit_center"])
    active_support = (support - center) / scale
    anchors = field("canonical_support")
    # The active ends are differences of public coordinates: their rounding
    # scales with the magnitudes that cancel, not with the result.
    cancelled = (np.abs(support) + abs(center)) / scale
    tolerance = 1e-12 * max(
        1.0,
        float(np.max(np.abs(anchors[np.isfinite(anchors)]), initial=0.0)),
        float(np.max(cancelled[np.isfinite(cancelled)], initial=0.0)),
    )
    if (
        active_support[0] < anchors[0] - tolerance
        or active_support[1] > anchors[1] + tolerance
    ):
        raise ValueError("active support lies outside the original potential anchors")

    npanels = int(struct["cdf_npanels"])
    stride = int(struct["cdf_coeff_stride"])
    if npanels > 0 and stride > 0:
        _check_finite_increasing(field("cdf_breaks")[: npanels + 1], "cdf_breaks")
        if not np.all(np.isfinite(field("cdf_offsets")[:npanels])):
            raise ValueError("cdf_offsets must be finite")
        if not np.all(np.isfinite(field("cdf_coeffs")[: npanels * stride])):
            raise ValueError("cdf_coeffs must be finite")
        params = field("cdf_map_params")
        if params.size != 4 or not np.allclose(
            params[:2], active_support, rtol=1e-12, atol=tolerance
        ):
            raise ValueError("cdf_map_params disagree with the active support")
        if np.any(np.isnan(params)) or not np.all(np.isfinite(params[2:])):
            raise ValueError(
                "cdf_map_params must be finite apart from support endpoints"
            )

    pmin = float(struct["ppf_pmin"])
    pmax = float(struct["ppf_pmax"])
    ppf_panels = int(struct["ppf_npanels"])
    ppf_stride = int(struct["ppf_coeff_stride"])
    if pmin < pmax and ppf_panels > 0 and ppf_stride > 0:
        # An empty [pmin, pmax] is the documented fallback that routes every
        # query through CDF inversion, and carries no usable panel geometry.
        if not (0.0 <= pmin < pmax <= 1.0):
            raise ValueError("ppf_pmin and ppf_pmax must satisfy 0 <= pmin < pmax <= 1")
        _check_finite_increasing(
            field("ppf_breaks_r")[: ppf_panels + 1], "ppf_breaks_r"
        )
        _check_finite_increasing(
            field("ppf_breaks_z")[: ppf_panels + 1], "ppf_breaks_z"
        )
        if not np.all(np.isfinite(field("ppf_coeffs")[: ppf_panels * ppf_stride])):
            raise ValueError("ppf_coeffs must be finite")
