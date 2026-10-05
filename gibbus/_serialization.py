"""Versioned durable serialization for fitted :class:`Distribution` models.

Serialization format v1 deliberately separates the mathematical model from
optional provenance and runtime caches.  Only the ``model`` section is frozen
as the durable cross-version contract.  ``provenance`` may grow by adding
ignorable fields, and ``cache`` is an implementation detail that is reused
only when it is known to match the running Gibbus build and the durable model.

Optional sections are validated explicitly, field by field, before any value
is read.  A section that does not match the recognized layout is treated as
absent; nothing here detects malformed data by catching exceptions.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from importlib.metadata import distributions

import numpy as np

from ._fit.mixture import _pack_mixture_struct, _unpack_mixture_struct
from ._postfit.fitted_state import (
    _metadata_conflict,
    _model_metadata,
    _shared_model_geometry,
)

FORMAT_NAME = "gibbus"
FORMAT_VERSION = 1

_ENVELOPE_FIELDS = ("format", "format_version", "model", "provenance", "cache")

_MODEL_FIELDS = (
    "n_components",
    "weights",
    "q_poly_values",
    "q_poly_offsets",
    "canonical_support",
    "boundary_amplitudes",
    "boundary_allowed",
    "fit_center",
    "fit_scale",
    "fit_direction",
    "support",
    "requested_poly_degree",
    "mu",
    "sigma",
    "default_space",
)

# Sentinel written into component states whose historical solver record is
# unavailable (see ``_clear_historical_component_provenance``).  States carrying
# it are serialized with an absent provenance section rather than re-labelling
# the placeholder values as history.
_MISSING_HISTORY = "missing"

# Recognized provenance layout: field -> (base dtype, trailing shape).  Model
# fields are scalars or fixed vectors; component fields gain a leading K axis.
# Strings match any ``U`` width.  The writer emits exactly these dtypes and the
# reader accepts only them, so a recognized value always fits the internal
# component-state field it fills.
_PROVENANCE_MODEL_SPEC = {
    "writer_version": ("<U64", ()),
    "model_provenance": ("<U16", ()),
    "n_parameters": ("<i8", ()),
    "n_face_parameters": ("<i8", ()),
    "shared_boundary_allowed": ("?", (2,)),
    "shared_boundary_amplitudes": ("<f8", (2,)),
    "shared_boundary_active": ("?", (2,)),
    "shared_boundary_standard_errors": ("<f8", (2,)),
    "shared_boundary_p_values": ("<f8", (2,)),
}
_PROVENANCE_COMPONENT_SPEC = {
    "nll": ("<f8", ()),
    "optimizer_success": ("i1", ()),
    "optimizer_status": ("<U64", ()),
    "optimizer_message": ("<U256", ()),
    "optimizer_n_iterations": ("<i8", ()),
    "optimizer_n_evaluations": ("<i8", ()),
    "optimizer_subproblem_iterations": ("<i8", ()),
    "optimizer_decrease_bound": ("<f8", ()),
    "effective_curvature_degree": ("<i8", ()),
    "lower_amplitude_active": ("i1", ()),
    "upper_amplitude_active": ("i1", ()),
    "separator_certified": ("i1", ()),
    "boundary_standard_errors": ("<f8", (2,)),
    "boundary_p_values": ("<f8", (2,)),
}
_PROVENANCE_FLAGS = (
    "optimizer_success",
    "lower_amplitude_active",
    "upper_amplitude_active",
    "separator_certified",
)
# Iteration counters use -1 for "unavailable"; anything lower is malformed.
_PROVENANCE_COUNTERS = (
    "optimizer_n_iterations",
    "optimizer_n_evaluations",
    "optimizer_subproblem_iterations",
)

_CACHE_HEADER_SPEC = {
    "present": ("?", ()),
    "writer_version": ("<U64", ()),
    "model_sha256": ("<U64", ()),
    "runtime_sha256": ("<U64", ()),
}


@dataclass(frozen=True)
class _LoadedState:
    """Validated ingredients needed to install one serialized distribution."""

    weights: np.ndarray
    default_space: str
    comp_states: tuple[np.ndarray, ...]
    mu: float
    sigma: float
    fit_metadata: dict | None
    base_modes: tuple[float, ...] | None


@dataclass(frozen=True)
class _History:
    """Recognized optional provenance, detached from the serialized section."""

    fit_metadata: dict
    records: tuple[dict, ...]


@functools.cache
def _package_version() -> str:
    """Return the installed Gibbus version used to key optional runtime caches.

    The metadata lookup scans installed distributions, so it is resolved once
    per process; the installed version cannot change underneath a running
    interpreter.  A source tree without installed metadata reports
    ``"0+unknown"``.
    """
    found = next(iter(distributions(name="gibbus")), None)
    return "0+unknown" if found is None else str(found.version)


# ----------------------------------------------------------------------
# Explicit structural checks for optional sections
# ----------------------------------------------------------------------


def _structured_fields(section, /):
    """Return the field names of a structured scalar, or ``None``.

    Parameters
    ----------
    section : object
        Candidate section taken from a serialized envelope.

    Returns
    -------
    tuple of str or None
        Field names when *section* is a non-object structured scalar.
    """
    dtype = getattr(section, "dtype", None)
    if not isinstance(dtype, np.dtype) or dtype.names is None or dtype.hasobject:
        return None
    if np.ndim(section) != 0:
        return None
    return dtype.names


def _matches_spec(section, names, spec, k, /):
    """Return whether every field in *spec* is present with its exact layout.

    Parameters
    ----------
    section : numpy.void
        Structured scalar holding the fields.
    names : tuple of str
        Field names of *section*.
    spec : Mapping
        Field name -> ``(base dtype, trailing shape)``.  String dtypes match
        any ``U`` width.
    k : int or None
        Component count prepended to every shape, or ``None`` for model-level
        fields.

    Returns
    -------
    bool
    """
    for name, (dtype, shape) in spec.items():
        if name not in names:
            return False
        field = section.dtype.fields[name][0]
        base, subshape = field.subdtype or (field, ())
        expected = np.dtype(dtype)
        if expected.kind == "U":
            if base.kind != "U":
                return False
        elif base != expected:
            return False
        full_shape = tuple(shape) if k is None else (k, *shape)
        if tuple(subshape) != full_shape:
            return False
    return True


def _section_present(section, names, /):
    """Return whether an optional section's ``present`` flag is a true bool.

    Parameters
    ----------
    section : numpy.void
        Structured optional section.
    names : tuple of str
        Field names of *section*.

    Returns
    -------
    bool
    """
    if not _matches_spec(section, names, {"present": ("?", ())}, None):
        return False
    return bool(section["present"])


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------


def _model_dtype(k, q_size, /):
    """Return the exact frozen v1 dtype for one durable model geometry.

    Parameters
    ----------
    k : int
        Number of mixture components.
    q_size : int
        Total number of potential coefficients across components.

    Returns
    -------
    numpy.dtype
    """
    return np.dtype(
        [
            ("n_components", "<i8"),
            ("weights", "<f8", (k,)),
            ("q_poly_values", "<f8", (q_size,)),
            ("q_poly_offsets", "<i8", (k + 1,)),
            ("canonical_support", "<f8", (k, 2)),
            ("boundary_amplitudes", "<f8", (k, 2)),
            ("boundary_allowed", "?", (k, 2)),
            ("fit_center", "<f8", (k,)),
            ("fit_scale", "<f8", (k,)),
            ("fit_direction", "<f8", (k,)),
            ("support", "<f8", (k, 2)),
            ("requested_poly_degree", "<i8", (k,)),
            ("mu", "<f8"),
            ("sigma", "<f8"),
            ("default_space", "<U4"),
        ]
    )


def _pack_model(weights, default_space, comp_states, /, *, mu, sigma):
    """Pack the frozen format-v1 mathematical model section.

    Parameters
    ----------
    weights : array_like, shape (K,)
        Mixture weights.
    default_space : str
        Active query space, ``"base"`` or ``"exp"``.
    comp_states : sequence of numpy.void
        Internal fitted component states.
    mu, sigma : float
        Accumulated public affine transform.

    Returns
    -------
    numpy.void
        Validated durable ``model`` section.

    Raises
    ------
    ValueError
        If the packed model violates the v1 schema or its invariants.
    """
    k = len(comp_states)
    q_parts = [
        np.asarray(state["q_poly"], dtype="<f8").reshape(-1) for state in comp_states
    ]
    offsets = np.zeros(k + 1, dtype="<i8")
    for j, part in enumerate(q_parts):
        offsets[j + 1] = offsets[j] + part.size
    values = np.concatenate(q_parts).astype("<f8", copy=False)

    out = np.zeros((), dtype=_model_dtype(k, values.size))
    out["n_components"] = np.int64(k)
    out["weights"] = np.asarray(weights, dtype="<f8")
    out["q_poly_values"] = values
    out["q_poly_offsets"] = offsets
    for name in (
        "canonical_support",
        "boundary_amplitudes",
        "boundary_allowed",
        "support",
    ):
        dtype = "?" if name == "boundary_allowed" else "<f8"
        out[name] = np.stack(
            [np.asarray(state[name], dtype=dtype) for state in comp_states]
        )
    for name in ("fit_center", "fit_scale", "fit_direction"):
        out[name] = np.asarray([state[name] for state in comp_states], dtype="<f8")
    out["requested_poly_degree"] = np.asarray(
        [state["requested_poly_degree"] for state in comp_states], dtype="<i8"
    )
    out["mu"] = np.float64(mu)
    out["sigma"] = np.float64(sigma)
    out["default_space"] = np.asarray(str(default_space), dtype="<U4")
    _validate_model_v1(out)
    return out


def _provenance_dtype(k, /):
    """Return the provenance dtype this writer emits for *k* components.

    Parameters
    ----------
    k : int
        Number of mixture components.

    Returns
    -------
    numpy.dtype
    """
    fields = [("present", "?")]
    fields += [
        (name, dtype, shape) for name, (dtype, shape) in _PROVENANCE_MODEL_SPEC.items()
    ]
    fields += [
        (name, dtype, (k, *shape))
        for name, (dtype, shape) in _PROVENANCE_COMPONENT_SPEC.items()
    ]
    return np.dtype(fields)


def _pack_provenance(comp_states, fit_metadata, /):
    """Pack optional historical fit metadata without making it durable state.

    Parameters
    ----------
    comp_states : sequence of numpy.void
        Internal fitted component states carrying solver history.
    fit_metadata : Mapping or None
        Model-level metadata; ``None`` derives it from the components.

    Returns
    -------
    numpy.void
        Provenance section marked present.
    """
    metadata = _model_metadata(comp_states, fit_metadata)
    out = np.zeros((), dtype=_provenance_dtype(len(comp_states)))
    out["present"] = True
    out["writer_version"] = _package_version()
    out["model_provenance"] = metadata["provenance"]
    out["n_parameters"] = np.int64(metadata["n_parameters"])
    out["n_face_parameters"] = np.int64(metadata["n_face_parameters"])
    shared = metadata["shared_boundary"]
    for name in ("allowed", "amplitudes", "active", "standard_errors", "p_values"):
        out[f"shared_boundary_{name}"] = shared[name]
    for name, (dtype, _) in _PROVENANCE_COMPONENT_SPEC.items():
        if np.dtype(dtype).kind == "U":
            out[name] = [str(state[name]) for state in comp_states]
        else:
            out[name] = np.stack([np.asarray(state[name]) for state in comp_states])
    return out


def _absent_section():
    """Return the minimal optional-section stub marked as not present."""
    out = np.zeros((), dtype=[("present", "?")])
    out["present"] = False
    return out


def _history_unavailable(comp_states, /):
    """Return whether any component carries the missing-history sentinel.

    Such states come from loading a model whose optional provenance was absent
    or unusable.  Writing them back as present provenance would turn
    placeholder values into apparent history.

    Parameters
    ----------
    comp_states : sequence of numpy.void
        Internal fitted component states.

    Returns
    -------
    bool
    """
    return any(
        "optimizer_status" in (state.dtype.names or ())
        and str(state["optimizer_status"]) == _MISSING_HISTORY
        for state in comp_states
    )


def _canonical_model_digest(model, /):
    """Return SHA-256 of the frozen model values in a platform-stable encoding.

    Parameters
    ----------
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    h = hashlib.sha256()
    for name in _MODEL_FIELDS:
        h.update(name.encode("ascii"))
        value = np.asarray(model[name])
        if value.dtype.kind == "U":
            encoded = str(value.item()).encode("utf-8")
            h.update(len(encoded).to_bytes(8, "little"))
            h.update(encoded)
            continue
        if value.dtype.kind == "b":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("?"))
        elif value.dtype.kind in "iu":
            canonical = np.ascontiguousarray(value, dtype=np.dtype("<i8"))
        else:
            canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
        h.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        h.update(canonical.tobytes(order="C"))
    return h.hexdigest()


def _runtime_digest(runtime, /):
    """Return SHA-256 of a runtime cache payload's dtype description and bytes.

    The cache is only reused by the exact writer version, so hashing the
    stored bytes directly is sufficient: ``np.save``/``np.load`` preserve them.
    A mismatch identifies a corrupted or edited cache before it is parsed.

    Parameters
    ----------
    runtime : numpy.void
        Structured runtime cache payload.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    raw = np.asarray(runtime)
    h = hashlib.sha256()
    h.update(repr(raw.dtype.descr).encode("utf-8"))
    h.update(np.ascontiguousarray(raw).tobytes())
    return h.hexdigest()


def _pack_cache(
    model,
    weights,
    default_space,
    comp_states,
    /,
    *,
    base_modes,
    mu,
    sigma,
    fit_metadata,
):
    """Pack the optional current-implementation runtime cache.

    Parameters
    ----------
    model : numpy.void
        Durable ``model`` section the cache belongs to.
    weights : array_like, shape (K,)
        Mixture weights.
    default_space : str
        Active query space.
    comp_states : sequence of numpy.void
        Internal fitted component states, including spectral panels.
    base_modes : sequence of float or None
        Cached mixture modes in base coordinates, when known.
    mu, sigma : float
        Accumulated public affine transform.
    fit_metadata : Mapping or None
        Model-level metadata stored with the runtime state.

    Returns
    -------
    numpy.void
        Cache section marked present.
    """
    runtime = _pack_mixture_struct(
        weights,
        default_space,
        comp_states,
        base_modes=base_modes,
        mu=mu,
        sigma=sigma,
        fit_metadata=fit_metadata,
    )
    dtype = np.dtype(
        [(name, dtype) for name, (dtype, _) in _CACHE_HEADER_SPEC.items()]
        + [("runtime_state", runtime.dtype)]
    )
    out = np.zeros((), dtype=dtype)
    out["present"] = True
    out["writer_version"] = _package_version()
    out["model_sha256"] = _canonical_model_digest(model)
    out["runtime_sha256"] = _runtime_digest(runtime)
    out["runtime_state"] = runtime
    return out


def pack_distribution_state(
    weights,
    default_space,
    comp_states,
    /,
    *,
    base_modes=None,
    mu=0.0,
    sigma=1.0,
    fit_metadata=None,
):
    """Pack one versioned, non-object durable Distribution state.

    Parameters
    ----------
    weights : array_like, shape (K,)
        Mixture weights.
    default_space : str
        Active query space, ``"base"`` or ``"exp"``.
    comp_states : sequence of numpy.void
        Internal fitted component states.
    base_modes : sequence of float or None, optional
        Cached mixture modes in base coordinates, when known.
    mu, sigma : float, optional
        Accumulated public affine transform.
    fit_metadata : Mapping or None, optional
        Model-level metadata; ``None`` derives it from the components.

    Returns
    -------
    numpy.void
        Serialization format v1 envelope.
    """
    model = _pack_model(weights, default_space, comp_states, mu=mu, sigma=sigma)
    if _history_unavailable(comp_states):
        provenance = _absent_section()
    else:
        provenance = _pack_provenance(comp_states, fit_metadata)
    cache = _pack_cache(
        model,
        weights,
        default_space,
        comp_states,
        base_modes=base_modes,
        mu=mu,
        sigma=sigma,
        fit_metadata=fit_metadata,
    )
    dtype = np.dtype(
        [
            ("format", "<U8"),
            ("format_version", "<i8"),
            ("model", model.dtype),
            ("provenance", provenance.dtype),
            ("cache", cache.dtype),
        ]
    )
    out = np.zeros((), dtype=dtype)
    out["format"] = FORMAT_NAME
    out["format_version"] = np.int64(FORMAT_VERSION)
    out["model"] = model
    out["provenance"] = provenance
    out["cache"] = cache
    return out


# ----------------------------------------------------------------------
# Durable model validation
# ----------------------------------------------------------------------


def _expect_field_dtype(model, name, dtype, shape, /):
    """Require one model field to have the exact v1 base dtype and shape.

    Parameters
    ----------
    model : numpy.void
        Candidate durable ``model`` section.
    name : str
        Field to check.
    dtype : str
        Required base dtype.
    shape : tuple of int
        Required trailing shape.

    Raises
    ------
    ValueError
        If the field's dtype or shape differs.
    """
    field_dtype = model.dtype.fields[name][0]
    base, subshape = field_dtype.subdtype or (field_dtype, ())
    if base != np.dtype(dtype) or tuple(subshape) != tuple(shape):
        raise ValueError(
            f"serialization v1 model field {name!r} must have dtype {np.dtype(dtype)} "
            f"and shape {tuple(shape)}, got {field_dtype}"
        )


def _validate_model_v1(model, /):
    """Validate the frozen v1 model schema and mathematical invariants.

    Parameters
    ----------
    model : numpy.void
        Candidate durable ``model`` section.

    Raises
    ------
    ValueError
        If the section does not match the frozen v1 schema or describes no
        valid distribution.
    """
    if _structured_fields(model) is None:
        raise ValueError(
            "serialization v1 model must be a non-object structured scalar"
        )
    names = tuple(model.dtype.names)
    if names != _MODEL_FIELDS:
        missing = [name for name in _MODEL_FIELDS if name not in names]
        unknown = [name for name in names if name not in _MODEL_FIELDS]
        details = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        if not details:
            details.append("field order does not match the frozen v1 schema")
        raise ValueError("invalid serialization v1 model; " + "; ".join(details))

    _expect_field_dtype(model, "n_components", "<i8", ())
    k = int(model["n_components"])
    if k < 1:
        raise ValueError("serialization v1 n_components must be a positive integer")

    offsets = np.asarray(model["q_poly_offsets"])
    q_values = np.asarray(model["q_poly_values"])
    if offsets.ndim != 1 or q_values.ndim != 1:
        raise ValueError("serialization v1 q_poly storage must be one-dimensional")
    expected = {
        "weights": ("<f8", (k,)),
        "q_poly_values": ("<f8", q_values.shape),
        "q_poly_offsets": ("<i8", (k + 1,)),
        "canonical_support": ("<f8", (k, 2)),
        "boundary_amplitudes": ("<f8", (k, 2)),
        "boundary_allowed": ("?", (k, 2)),
        "fit_center": ("<f8", (k,)),
        "fit_scale": ("<f8", (k,)),
        "fit_direction": ("<f8", (k,)),
        "support": ("<f8", (k, 2)),
        "requested_poly_degree": ("<i8", (k,)),
        "mu": ("<f8", ()),
        "sigma": ("<f8", ()),
        "default_space": ("<U4", ()),
    }
    for name, (dtype, shape) in expected.items():
        _expect_field_dtype(model, name, dtype, shape)

    if offsets[0] != 0 or offsets[-1] != q_values.size:
        raise ValueError("serialization v1 q_poly_offsets do not cover q_poly_values")
    lengths = np.diff(offsets)
    if np.any(lengths < 3):
        raise ValueError(
            "serialization v1 q_poly entries must have at least three coefficients"
        )
    if not np.all(np.isfinite(q_values)):
        raise ValueError("serialization v1 q_poly data are malformed")

    weights = np.asarray(model["weights"], dtype=np.float64)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("serialization v1 model has invalid component weights")

    centers = np.asarray(model["fit_center"], dtype=np.float64)
    scales = np.asarray(model["fit_scale"], dtype=np.float64)
    directions = np.asarray(model["fit_direction"], dtype=np.float64)
    if not np.all(np.isfinite(centers)):
        raise ValueError("serialization v1 fit_center must be finite")
    if not np.all(np.isfinite(scales) & (scales > 0.0)):
        raise ValueError("serialization v1 fit_scale must be finite and positive")
    if not np.all(np.isin(directions, (-1.0, 1.0))):
        raise ValueError("serialization v1 fit_direction must be -1 or +1")

    supports = np.asarray(model["support"], dtype=np.float64)
    anchors = np.asarray(model["canonical_support"], dtype=np.float64)
    if np.any(np.isnan(supports)) or np.any(supports[:, 0] >= supports[:, 1]):
        raise ValueError("serialization v1 support must satisfy lower < upper")
    if np.any(np.isnan(anchors)) or np.any(anchors[:, 0] >= anchors[:, 1]):
        raise ValueError(
            "serialization v1 canonical_support must satisfy lower < upper"
        )

    amplitudes = np.asarray(model["boundary_amplitudes"], dtype=np.float64)
    allowed = np.asarray(model["boundary_allowed"], dtype=bool)
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError(
            "serialization v1 boundary amplitudes must be finite and non-negative"
        )
    if np.any((~allowed) & (amplitudes != 0.0)):
        raise ValueError("serialization v1 disabled boundary amplitudes must be zero")

    requested = np.asarray(model["requested_poly_degree"], dtype=np.int64)
    effective = lengths - 1
    if np.any(requested < 2) or np.any(requested < effective):
        raise ValueError(
            "serialization v1 requested polynomial degrees are inconsistent"
        )

    for j in range(k):
        active = np.sort(directions[j] * (supports[j] - centers[j]) / scales[j])
        finite_anchor = anchors[j][np.isfinite(anchors[j])]
        tol = 1e-12 * max(1.0, float(np.max(np.abs(finite_anchor), initial=0.0)))
        if active[0] < anchors[j, 0] - tol or active[1] > anchors[j, 1] + tol:
            raise ValueError(
                "serialization v1 active support lies outside potential anchors"
            )

    if k > 1:
        if not all(np.array_equal(supports[j], supports[0]) for j in range(1, k)):
            raise ValueError("serialization v1 mixture components disagree on support")
        if not all(np.array_equal(allowed[j], allowed[0]) for j in range(1, k)):
            raise ValueError(
                "serialization v1 mixture components disagree on boundary policy"
            )
        if not all(np.array_equal(amplitudes[j], amplitudes[0]) for j in range(1, k)):
            raise ValueError(
                "serialization v1 mixture components disagree on boundary amplitudes"
            )

    if not np.isfinite(float(model["mu"])):
        raise ValueError("serialization v1 mu must be finite")
    sigma = float(model["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("serialization v1 sigma must be finite and positive")
    if str(model["default_space"]) not in ("base", "exp"):
        raise ValueError("serialization v1 default_space must be 'base' or 'exp'")


def _component_models(model, /):
    """Return detached per-component durable model mappings.

    Parameters
    ----------
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    list of dict
        One mapping per component, in stored order.
    """
    offsets = np.asarray(model["q_poly_offsets"], dtype=np.int64)
    values = np.asarray(model["q_poly_values"], dtype=np.float64)
    items = []
    for j in range(int(model["n_components"])):
        items.append(
            {
                "q_poly": values[offsets[j] : offsets[j + 1]].copy(),
                "canonical_support": np.asarray(
                    model["canonical_support"][j], dtype=np.float64
                ).copy(),
                "boundary_amplitudes": np.asarray(
                    model["boundary_amplitudes"][j], dtype=np.float64
                ).copy(),
                "boundary_allowed": np.asarray(
                    model["boundary_allowed"][j], dtype=bool
                ).copy(),
                "fit_center": float(model["fit_center"][j]),
                "fit_scale": float(model["fit_scale"][j]),
                "fit_direction": float(model["fit_direction"][j]),
                "support": np.asarray(model["support"][j], dtype=np.float64).copy(),
                "requested_poly_degree": int(model["requested_poly_degree"][j]),
            }
        )
    return items


# ----------------------------------------------------------------------
# Optional provenance
# ----------------------------------------------------------------------


def _history_values_valid(provenance, model, /):
    """Return whether recognized provenance values are admissible for *model*.

    Values that could not have been written for this model, such as an
    out-of-range flag or an active boundary face the model does not allow,
    make the whole record unusable rather than partially applied.

    Parameters
    ----------
    provenance : numpy.void
        Provenance section whose layout already matches the recognized spec.
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    bool
    """
    for name in _PROVENANCE_FLAGS:
        if not np.all(np.isin(provenance[name], (0, 1))):
            return False
    for name in _PROVENANCE_COUNTERS:
        if np.any(np.asarray(provenance[name]) < -1):
            return False
    if np.any(np.asarray(provenance["effective_curvature_degree"]) < 0):
        return False
    for name in ("boundary_standard_errors", "shared_boundary_standard_errors"):
        if np.any(np.asarray(provenance[name]) < 0.0):
            return False
    for name in ("boundary_p_values", "shared_boundary_p_values"):
        values = np.asarray(provenance[name])
        if np.any((values < 0.0) | (values > 1.0)):
            return False

    # Active-face flags are stored per component in canonical z order; the
    # model stores boundary policy and amplitudes in public x order.
    allowed = np.asarray(model["boundary_allowed"], dtype=bool)
    amplitudes = np.asarray(model["boundary_amplitudes"], dtype=np.float64)
    directions = np.asarray(model["fit_direction"], dtype=np.float64)
    first_side = None
    for j in range(int(model["n_components"])):
        side = np.array(
            [
                provenance["lower_amplitude_active"][j],
                provenance["upper_amplitude_active"][j],
            ],
            dtype=bool,
        )
        if directions[j] < 0.0:
            side = side[::-1]
        if np.any(side & ~allowed[j]) or np.any((~side) & (amplitudes[j] != 0.0)):
            return False
        if first_side is not None and not np.array_equal(side, first_side):
            return False
        first_side = side
    return True


def _read_provenance(provenance, model, /):
    """Return recognized optional history, or ``None`` when it is unusable.

    Provenance is all-or-nothing: a missing, partial, re-typed, or
    inadmissible record yields ``None`` so no stale history is ever mixed with
    a rebuilt or cached state.  Unknown extra fields are ignored.

    Parameters
    ----------
    provenance : object
        The envelope's ``provenance`` section.
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    _History or None
    """
    names = _structured_fields(provenance)
    if names is None or not _section_present(provenance, names):
        return None
    k = int(model["n_components"])
    if not _matches_spec(provenance, names, _PROVENANCE_MODEL_SPEC, None):
        return None
    if not _matches_spec(provenance, names, _PROVENANCE_COMPONENT_SPEC, k):
        return None
    if not _history_values_valid(provenance, model):
        return None

    fit_metadata = {
        "provenance": str(provenance["model_provenance"]),
        "n_parameters": int(provenance["n_parameters"]),
        "n_face_parameters": int(provenance["n_face_parameters"]),
        "shared_boundary": {
            name: np.array(provenance[f"shared_boundary_{name}"], copy=True)
            for name in (
                "allowed",
                "amplitudes",
                "active",
                "standard_errors",
                "p_values",
            )
        },
    }
    records = []
    for j in range(k):
        record = {}
        for name, (dtype, shape) in _PROVENANCE_COMPONENT_SPEC.items():
            value = provenance[name][j]
            kind = np.dtype(dtype).kind
            if shape:
                record[name] = np.array(value, dtype=np.float64, copy=True)
            elif kind == "U":
                record[name] = str(value)
            elif kind == "i":
                record[name] = int(value)
            else:
                record[name] = float(value)
        records.append(record)
    return _History(fit_metadata=fit_metadata, records=tuple(records))


def _clear_historical_component_provenance(states, /):
    """Return states with historical fields unavailable and geometry derived.

    Parameters
    ----------
    states : sequence of numpy.void
        Internal component states.

    Returns
    -------
    list of numpy.void
        Detached copies carrying the missing-history sentinel.
    """
    out = []
    for state in states:
        item = np.array(state, copy=True)
        names = set(item.dtype.names or ())
        assignments = {
            "nll": np.nan,
            "optimizer_success": np.int8(0),
            "optimizer_status": _MISSING_HISTORY,
            "optimizer_message": _MISSING_HISTORY,
            "optimizer_n_iterations": np.int64(-1),
            "optimizer_n_evaluations": np.int64(-1),
            "optimizer_subproblem_iterations": np.int64(-1),
            "optimizer_decrease_bound": np.nan,
            "separator_certified": np.int8(0),
        }
        if "effective_curvature_degree" in names:
            degree = int(np.asarray(item["q_poly"]).size - 1)
            assignments["effective_curvature_degree"] = np.int64(max(0, degree - 2))
        if {"boundary_amplitudes", "fit_direction"} <= names:
            active = np.asarray(item["boundary_amplitudes"], dtype=np.float64) > 0.0
            if float(item["fit_direction"]) < 0.0:
                active = active[::-1]
            assignments["lower_amplitude_active"] = np.int8(active[0])
            assignments["upper_amplitude_active"] = np.int8(active[1])
        for name, value in assignments.items():
            if name in names:
                item[name] = value
        for name in ("boundary_standard_errors", "boundary_p_values"):
            if name in names:
                item[name] = np.full(2, np.nan, dtype=np.float64)
        out.append(item)
    return out


def _apply_history(states, history, /):
    """Overlay recognized history onto component states, or clear it.

    Parameters
    ----------
    states : sequence of numpy.void
        Internal component states, rebuilt or taken from a verified cache.
    history : _History or None
        Recognized provenance from :func:`_read_provenance`.

    Returns
    -------
    states : list of numpy.void
        Detached states carrying either the recorded or the missing history.
    fit_metadata : dict or None
        Model metadata consistent with *states*, or ``None`` when the history
        is unavailable or disagrees with the components.
    """
    if history is None:
        return _clear_historical_component_provenance(states), None
    out = []
    for state, record in zip(states, history.records, strict=True):
        item = np.array(state, copy=True)
        names = set(item.dtype.names or ())
        for name, value in record.items():
            if name in names:
                item[name] = value
        out.append(item)
    conflict = _metadata_conflict(_shared_model_geometry(out), history.fit_metadata)
    if conflict is not None:
        return _clear_historical_component_provenance(states), None
    return out, history.fit_metadata


# ----------------------------------------------------------------------
# Optional runtime cache
# ----------------------------------------------------------------------


def _verified_runtime(cache, model, /):
    """Return the cache payload when every integrity gate passes, else ``None``.

    The gates are: recognized header layout, ``present``, exact writer
    version, durable-model digest, and runtime-content digest.  Failing a gate
    is ordinary (absent, stale, edited, or corrupted cache) and selects a
    rebuild.

    Parameters
    ----------
    cache : object
        The envelope's ``cache`` section.
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    numpy.void or None
        The ``runtime_state`` payload, byte-identical to what this Gibbus
        build wrote for exactly this model.
    """
    names = _structured_fields(cache)
    if names is None or not _section_present(cache, names):
        return None
    if not _matches_spec(cache, names, _CACHE_HEADER_SPEC, None):
        return None
    if "runtime_state" not in names:
        return None
    runtime = cache["runtime_state"]
    if _structured_fields(runtime) is None:
        return None
    if str(cache["writer_version"]) != _package_version():
        return None
    if str(cache["model_sha256"]) != _canonical_model_digest(model):
        return None
    if str(cache["runtime_sha256"]) != _runtime_digest(runtime):
        return None
    return runtime


def _unpack_verified_runtime(runtime, model, /):
    """Unpack a digest-verified runtime cache and check it against *model*.

    Parameters
    ----------
    runtime : numpy.void
        Payload returned by :func:`_verified_runtime`.
    model : numpy.void
        Validated durable ``model`` section.

    Returns
    -------
    comp_states : list of numpy.void
    weights : numpy.ndarray
    default_space : str
    base_modes : tuple of float or None

    Raises
    ------
    ValueError
        If the payload is inconsistent.  A digest-verified cache holds exactly
        the bytes this build wrote for this model, so this indicates a Gibbus
        defect and is not hidden behind a silent rebuild.
    """
    weights, default_space, comp_states = _unpack_mixture_struct(runtime)
    defect = (
        "serialized runtime cache passed its integrity checks but {}; "
        "this indicates a Gibbus defect"
    )
    if not np.array_equal(np.asarray(weights, dtype=np.float64), model["weights"]):
        raise ValueError(defect.format("disagrees with the model weights"))
    if str(default_space) != str(model["default_space"]):
        raise ValueError(defect.format("disagrees with the model default space"))
    if float(runtime["mu"]) != float(model["mu"]) or float(runtime["sigma"]) != float(
        model["sigma"]
    ):
        raise ValueError(defect.format("disagrees with the model transform"))
    durable = _component_models(model)
    if len(comp_states) != len(durable):
        raise ValueError(defect.format("has the wrong component count"))
    for state, item in zip(comp_states, durable, strict=True):
        for name in (
            "q_poly",
            "canonical_support",
            "boundary_amplitudes",
            "boundary_allowed",
            "support",
        ):
            if not np.array_equal(np.asarray(state[name]), np.asarray(item[name])):
                raise ValueError(defect.format(f"disagrees with model field {name}"))
        for name in (
            "fit_center",
            "fit_scale",
            "fit_direction",
            "requested_poly_degree",
        ):
            if np.asarray(state[name]).item() != item[name]:
                raise ValueError(defect.format(f"disagrees with model field {name}"))

    raw_modes = np.asarray(runtime["base_modes"], dtype=np.float64).reshape(-1)
    if int(runtime["n_modes"]) != raw_modes.size or not np.all(np.isfinite(raw_modes)):
        raise ValueError(defect.format("holds inconsistent cached modes"))
    base_modes = tuple(map(float, raw_modes)) if raw_modes.size else None
    weights = np.asarray(weights, dtype=np.float64)
    return comp_states, weights, str(default_space), base_modes


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------


def load_distribution_state(state, /, *, rebuild_component):
    """Validate and unpack one versioned durable Distribution state.

    Parameters
    ----------
    state : numpy.void
        Serialized envelope, typically from :attr:`Distribution.data` or
        ``np.load(..., allow_pickle=False)``.
    rebuild_component : callable
        ``rebuild_component(durable_component, provenance)`` returning an
        internal component state.  Injected by the API layer to keep this
        module independent of component construction details.

    Returns
    -------
    _LoadedState

    Raises
    ------
    ValueError
        If the envelope or durable model is invalid or uses an unsupported
        format version, or if a digest-verified cache is inconsistent.
    """
    raw = np.array(state, copy=True)
    names = _structured_fields(raw)
    if names is None:
        raise ValueError(
            "serialized Gibbus state must be a non-object structured scalar"
        )
    if names != _ENVELOPE_FIELDS:
        if "format_version" not in names:
            raise ValueError(
                "unversioned Gibbus state is not supported; refit or resave it "
                "with a released version"
            )
        raise ValueError("invalid Gibbus serialization envelope")
    if raw.dtype["format"] != np.dtype("<U8"):
        raise ValueError("serialization format must have dtype <U8")
    if raw.dtype["format_version"] != np.dtype("<i8"):
        raise ValueError("serialization format_version must have dtype <i8")
    if str(raw["format"]) != FORMAT_NAME:
        raise ValueError("serialized state is not a Gibbus model")
    version = int(raw["format_version"])
    if version > FORMAT_VERSION:
        raise ValueError(
            f"serialization format {version} is newer than this Gibbus build supports "
            f"(maximum {FORMAT_VERSION})"
        )
    if version < FORMAT_VERSION:
        raise ValueError(
            f"serialization format {version} is no longer supported by this "
            "Gibbus build"
        )

    model = raw["model"]
    _validate_model_v1(model)

    runtime = _verified_runtime(raw["cache"], model)
    if runtime is not None:
        comp_states, weights, default_space, base_modes = _unpack_verified_runtime(
            runtime, model
        )
    else:
        comp_states = [
            rebuild_component(item, None) for item in _component_models(model)
        ]
        weights = np.asarray(model["weights"], dtype=np.float64).copy()
        default_space = str(model["default_space"])
        base_modes = None

    comp_states, fit_metadata = _apply_history(
        comp_states, _read_provenance(raw["provenance"], model)
    )
    return _LoadedState(
        weights=weights,
        default_space=default_space,
        comp_states=tuple(comp_states),
        mu=float(model["mu"]),
        sigma=float(model["sigma"]),
        fit_metadata=fit_metadata,
        base_modes=base_modes,
    )


# ----------------------------------------------------------------------
# Test and fixture helpers
# ----------------------------------------------------------------------


def _replace_optional_section(state, section, /):
    """Return a copy of a v1 envelope with one optional section marked absent.

    Parameters
    ----------
    state : numpy.void
        Versioned Gibbus serialization envelope.
    section : {"provenance", "cache"}
        Optional section to replace with the absent stub.

    Returns
    -------
    numpy.void

    Raises
    ------
    ValueError
        If *state* is not a versioned envelope.
    """
    raw = np.array(state, copy=True)
    if _structured_fields(raw) != _ENVELOPE_FIELDS:
        raise ValueError("expected a versioned Gibbus serialization envelope")
    stub = _absent_section()
    dtype = np.dtype(
        [
            (name, stub.dtype if name == section else raw.dtype[name])
            for name in _ENVELOPE_FIELDS
        ]
    )
    out = np.zeros((), dtype=dtype)
    for name in _ENVELOPE_FIELDS:
        out[name] = stub if name == section else raw[name]
    return out


def strip_cache(state, /):
    """Return a v1 state whose optional runtime cache is marked absent.

    This private helper exists for compatibility fixtures and tests.  It does
    not alter the durable model or provenance sections.

    Parameters
    ----------
    state : numpy.void
        Versioned Gibbus serialization envelope.

    Returns
    -------
    numpy.void
    """
    return _replace_optional_section(state, "cache")


def strip_provenance(state, /):
    """Return a v1 state whose optional provenance section is marked absent.

    Parameters
    ----------
    state : numpy.void
        Versioned Gibbus serialization envelope.

    Returns
    -------
    numpy.void
    """
    return _replace_optional_section(state, "provenance")
