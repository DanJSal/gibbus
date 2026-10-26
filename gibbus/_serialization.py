"""Durable serialization of fitted :class:`Distribution` models.

The envelope carries ``format`` and ``format_version`` and separates the
mathematical model from optional provenance and runtime caches.  Only the
``model`` section is the durable contract: per component, the basis record,
the canonical coefficients, the canonical potential's ``log Z`` and the
active support.  ``provenance`` may grow by adding ignorable fields, and
``cache`` is an implementation detail that is reused only when it is known to
match the running Gibbus build and the durable model.

Optional sections are validated explicitly, field by field, before any value
is read.  A section that does not match the recognized layout is treated as
absent; nothing here detects malformed data by catching exceptions.
"""

from __future__ import annotations

import functools
import hashlib
import sys
from dataclasses import dataclass
from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
import scipy

from ._fit.mixture import _pack_mixture_struct, _unpack_mixture_struct
from ._postfit.basis_state import _model_from_record, _pack_basis_state
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
    "basis_models",
    "theta_values",
    "theta_offsets",
    "log_z",
    "support",
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
    "basis_shape": ("<U32", ()),
    "basis_shape_tau": ("<f8", ()),
    "basis_n_face": ("<i8", ()),
    "boundary_standard_errors": ("<f8", (2,)),
    "boundary_p_values": ("<f8", (2,)),
}
# Iteration counters use -1 for "unavailable"; anything lower is malformed.
_PROVENANCE_COUNTERS = (
    "optimizer_n_iterations",
    "optimizer_n_evaluations",
    "optimizer_subproblem_iterations",
)

_CACHE_HEADER_SPEC = {
    "present": ("?", ()),
    "writer_version": ("<U64", ()),
    "implementation_sha256": ("<U64", ()),
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


@functools.cache
def _implementation_digest():
    """Identify runtime code and numerical dependencies once per process.

    Content hashes distinguish editable snapshots and rebuilt extensions even
    when their package versions agree. This cache identity is not durable schema.
    """
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                sys.implementation.cache_tag,
                sys.byteorder,
                np.__version__,
                scipy.__version__,
            )
        ).encode()
    )
    root = Path(__file__).resolve().parent
    for path in sorted(root.rglob("*")):
        if path.is_file() and (
            path.suffix == ".py"
            or any(str(path).endswith(s) for s in EXTENSION_SUFFIXES)
        ):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


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


def _pack_model(weights, default_space, comp_states, /, *, mu, sigma):
    """Pack the durable model section of basis components.

    Each component is its basis record (request, geometry, slot choice;
    ``basis_state._model_record``), its canonical coefficients, the canonical
    potential's ``log Z`` over the basis support and the active support (a
    truncation keeps the potential and narrows the support).

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
        If the packed model violates the schema or its invariants.
    """
    k = len(comp_states)
    thetas = [
        np.asarray(s["basis_theta"], dtype="<f8").reshape(-1) for s in comp_states
    ]
    offsets = np.zeros(k + 1, dtype="<i8")
    for j, part in enumerate(thetas):
        offsets[j + 1] = offsets[j] + part.size
    records = [str(s["basis_model"]) for s in comp_states]
    width = max(len(r) for r in records)
    dtype = np.dtype(
        [
            ("n_components", "<i8"),
            ("weights", "<f8", (k,)),
            ("basis_models", f"<U{width}", (k,)),
            ("theta_values", "<f8", (int(offsets[-1]),)),
            ("theta_offsets", "<i8", (k + 1,)),
            ("log_z", "<f8", (k,)),
            ("support", "<f8", (k, 2)),
            ("mu", "<f8"),
            ("sigma", "<f8"),
            ("default_space", "<U4"),
        ]
    )
    out = np.zeros((), dtype=dtype)
    out["n_components"] = np.int64(k)
    out["weights"] = np.asarray(weights, dtype="<f8")
    out["basis_models"] = records
    out["theta_values"] = np.concatenate(thetas)
    out["theta_offsets"] = offsets
    out["log_z"] = [float(s["basis_log_z_canonical"]) for s in comp_states]
    out["support"] = np.stack(
        [np.asarray(s["support"], dtype="<f8") for s in comp_states]
    )
    out["mu"] = np.float64(mu)
    out["sigma"] = np.float64(sigma)
    out["default_space"] = np.asarray(str(default_space), dtype="<U4")
    _validate_model(out)
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
    # The validated layout ``_MODEL_FIELDS``.
    for name in model.dtype.names:
        h.update(name.encode("ascii"))
        value = np.asarray(model[name])
        if value.dtype.kind == "U":
            h.update(np.asarray(value.shape, dtype="<i8").tobytes())
            for item in value.reshape(-1):
                encoded = str(item).encode("utf-8")
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
    out["implementation_sha256"] = _implementation_digest()
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
    """Pack one non-object durable Distribution state.

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
        Serialization envelope.
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
    """Require one model field to have the exact base dtype and shape.

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
            f"serialization model field {name!r} must have dtype {np.dtype(dtype)} "
            f"and shape {tuple(shape)}, got {field_dtype}"
        )


def _validate_model(model, /):
    """Reject a model section that cannot describe a distribution.

    Parameters
    ----------
    model : numpy.void
        Candidate durable ``model`` section.

    Raises
    ------
    ValueError
        If the section does not match the schema or its invariants.
    """
    if _structured_fields(model) is None:
        raise ValueError("serialization model must be a non-object structured scalar")
    names = tuple(model.dtype.names)
    if names != _MODEL_FIELDS:
        raise ValueError(
            "invalid serialization model; fields "
            f"{names} do not match the frozen schema {_MODEL_FIELDS}"
        )
    _expect_field_dtype(model, "n_components", "<i8", ())
    k = int(model["n_components"])
    if k < 1:
        raise ValueError("serialization n_components must be a positive integer")
    theta = np.asarray(model["theta_values"])
    record_type = model.dtype["basis_models"].subdtype
    if theta.ndim != 1 or record_type is None or record_type[0].kind != "U":
        raise ValueError("serialization component storage is malformed")
    for name, (dtype, shape) in {
        "weights": ("<f8", (k,)),
        "theta_values": ("<f8", theta.shape),
        "theta_offsets": ("<i8", (k + 1,)),
        "log_z": ("<f8", (k,)),
        "support": ("<f8", (k, 2)),
        "mu": ("<f8", ()),
        "sigma": ("<f8", ()),
        "default_space": ("<U4", ()),
    }.items():
        _expect_field_dtype(model, name, dtype, shape)
    if tuple(model.dtype["basis_models"].shape) != (k,):
        raise ValueError("serialization needs one basis record per component")
    offsets = np.asarray(model["theta_offsets"])
    if offsets[0] != 0 or offsets[-1] != theta.size or np.any(np.diff(offsets) < 1):
        raise ValueError("serialization theta_offsets do not cover theta_values")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(model["log_z"]))):
        raise ValueError("serialization coefficients and log Z must be finite")
    weights = np.asarray(model["weights"], dtype=np.float64)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("serialization model has invalid component weights")
    supports = np.asarray(model["support"], dtype=np.float64)
    if np.any(np.isnan(supports)) or np.any(supports[:, 0] >= supports[:, 1]):
        raise ValueError("serialization support must satisfy lower < upper")
    if k > 1 and not all(np.array_equal(supports[j], supports[0]) for j in range(k)):
        raise ValueError("serialization mixture components disagree on support")
    if not np.isfinite(float(model["mu"])):
        raise ValueError("serialization mu must be finite")
    sigma = float(model["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("serialization sigma must be finite and positive")
    if str(model["default_space"]) not in ("base", "exp"):
        raise ValueError("serialization default_space must be 'base' or 'exp'")


# ----------------------------------------------------------------------
# Optional provenance
# ----------------------------------------------------------------------


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
    if not np.all(np.isin(provenance["optimizer_success"], (0, 1))):
        return None
    for name in _PROVENANCE_COUNTERS:
        if np.any(np.asarray(provenance[name]) < -1):
            return None
    if np.any(np.asarray(provenance["basis_n_face"]) < 0):
        return None
    if np.any(np.asarray(provenance["basis_shape_tau"]) <= 0.0):
        # A recorded threshold is positive; NaN means none was recorded.
        return None
    for name in ("boundary_standard_errors", "shared_boundary_standard_errors"):
        if np.any(np.asarray(provenance[name]) < 0.0):
            return None
    for name in ("boundary_p_values", "shared_boundary_p_values"):
        values = np.asarray(provenance[name])
        if np.any((values < 0.0) | (values > 1.0)):
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
        }
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
    if str(cache["implementation_sha256"]) != _implementation_digest():
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
    _check_runtime_components(comp_states, model, defect)
    raw_modes = np.asarray(runtime["base_modes"], dtype=np.float64).reshape(-1)
    if int(runtime["n_modes"]) != raw_modes.size or not np.all(np.isfinite(raw_modes)):
        raise ValueError(defect.format("holds inconsistent cached modes"))
    base_modes = tuple(map(float, raw_modes)) if raw_modes.size else None
    weights = np.asarray(weights, dtype=np.float64)
    return comp_states, weights, str(default_space), base_modes


def _check_runtime_components(comp_states, model, defect, /):
    """Require a verified cache to hold exactly the model's components.

    Parameters
    ----------
    comp_states : sequence of numpy.void
        Component states unpacked from the cache.
    model : numpy.void
        Validated durable ``model`` section.
    defect : str
        Message template (one ``{}`` slot) for a Gibbus defect.

    Raises
    ------
    ValueError
        If a component disagrees with the model.
    """
    k = int(model["n_components"])
    if len(comp_states) != k:
        raise ValueError(defect.format("has the wrong component count"))
    offsets = np.asarray(model["theta_offsets"])
    for j, state in enumerate(comp_states):
        theta = np.asarray(model["theta_values"][offsets[j] : offsets[j + 1]])
        if (
            str(state["basis_model"]) != str(model["basis_models"][j])
            or not np.array_equal(np.asarray(state["basis_theta"]), theta)
            or float(state["basis_log_z_canonical"]) != float(model["log_z"][j])
            or not np.array_equal(np.asarray(state["support"]), model["support"][j])
        ):
            raise ValueError(defect.format(f"disagrees with model component {j}"))


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------


# ``log Z`` recomputed at load must reproduce the stored value (normalizer
# accuracy).
_LOG_Z_TOLERANCE = 1e-9


def _rebuild_basis_component(model, j, /):
    """Rebuild component ``j``'s runtime state from the durable model.

    The basis is recompiled from its record and ``log Z`` recomputed; a
    disagreement with the stored value fails loudly (the record does not
    reproduce its density).  An active support narrower than the basis
    support is the truncation of the rebuilt state.

    Parameters
    ----------
    model : numpy.void
        Validated durable ``model`` section.
    j : int
        Component index.

    Returns
    -------
    numpy.void
        Internal component state.

    Raises
    ------
    ValueError
        If the record does not reproduce its coefficients, normalization or
        support.
    """
    # Deferred: ``_api.distribution`` imports this module, and ``_api.component``
    # is loaded through the ``_api`` package that imports ``distribution``.
    from ._api.component import _Component

    basis, suppressed = _model_from_record(model["basis_models"][j])
    offsets = np.asarray(model["theta_offsets"])
    theta = np.asarray(model["theta_values"][offsets[j] : offsets[j + 1]])
    if theta.size != basis.n_coefficients:
        raise ValueError("serialization coefficients do not match their basis")
    state = _pack_basis_state(basis, theta, suppressed=suppressed)
    stored = float(model["log_z"][j])
    rebuilt = float(state["basis_log_z_canonical"])
    if abs(rebuilt - stored) > _LOG_Z_TOLERANCE * max(1.0, abs(stored)):
        raise ValueError(
            f"serialization component {j} does not reproduce its normalization "
            f"(log Z {rebuilt!r} vs stored {stored!r})"
        )
    support = np.asarray(model["support"][j], dtype=np.float64)
    full = np.asarray(state["support"], dtype=np.float64)
    if np.array_equal(support, full):
        return state
    if support[0] < full[0] or support[1] > full[1]:
        raise ValueError("serialization active support lies outside its basis")
    lo = None if support[0] == full[0] else float(support[0])
    hi = None if support[1] == full[1] else float(support[1])
    return _Component(state)._truncate_base(lo, hi).data


def load_distribution_state(state, /):
    """Validate and unpack one durable Distribution state.

    Parameters
    ----------
    state : numpy.void
        Serialized envelope, typically from :attr:`Distribution.data` or
        ``np.load(..., allow_pickle=False)``.

    Returns
    -------
    _LoadedState

    Raises
    ------
    ValueError
        If the envelope or durable model is invalid, or if a digest-verified
        cache is inconsistent.
    """
    raw = np.array(state, copy=True)
    names = _structured_fields(raw)
    if names is None:
        raise ValueError(
            "serialized Gibbus state must be a non-object structured scalar"
        )
    if names != _ENVELOPE_FIELDS:
        raise ValueError(
            f"invalid Gibbus serialization envelope: fields {names} must be "
            f"{_ENVELOPE_FIELDS}"
        )
    if raw.dtype["format"] != np.dtype("<U8"):
        raise ValueError("serialization format must have dtype <U8")
    if raw.dtype["format_version"] != np.dtype("<i8"):
        raise ValueError("serialization format_version must have dtype <i8")
    if str(raw["format"]) != FORMAT_NAME:
        raise ValueError("serialized state is not a Gibbus model")
    if int(raw["format_version"]) != FORMAT_VERSION:
        raise ValueError(
            f"invalid Gibbus serialization envelope: format_version must be "
            f"{FORMAT_VERSION}"
        )
    model = raw["model"]
    _validate_model(model)
    runtime = _verified_runtime(raw["cache"], model)
    if runtime is not None:
        comp_states, weights, default_space, base_modes = _unpack_verified_runtime(
            runtime, model
        )
    else:
        k = int(model["n_components"])
        comp_states = [_rebuild_basis_component(model, j) for j in range(k)]
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
    """Return a copy of an envelope with one optional section marked absent.

    Parameters
    ----------
    state : numpy.void
        Gibbus serialization envelope.
    section : {"provenance", "cache"}
        Optional section to replace with the absent stub.

    Returns
    -------
    numpy.void

    Raises
    ------
    ValueError
        If *state* is not a Gibbus serialization envelope.
    """
    raw = np.array(state, copy=True)
    if _structured_fields(raw) != _ENVELOPE_FIELDS:
        raise ValueError("expected a Gibbus serialization envelope")
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
    """Return a state whose optional runtime cache is marked absent.

    This private helper exists for tests of the cacheless rebuild.  It does
    not alter the durable model or provenance sections.

    Parameters
    ----------
    state : numpy.void
        Gibbus serialization envelope.

    Returns
    -------
    numpy.void
    """
    return _replace_optional_section(state, "cache")


def strip_provenance(state, /):
    """Return a state whose optional provenance section is marked absent.

    Parameters
    ----------
    state : numpy.void
        Gibbus serialization envelope.

    Returns
    -------
    numpy.void
    """
    return _replace_optional_section(state, "provenance")
