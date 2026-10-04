"""Model-owned dimensions and shared-boundary inference after fitting."""

import numpy as np

from .._fit.boundary import _weakly_identified_sides


def _shared_standard_errors(information, free_indices, shared_indices, effective_n, /):
    """Marginal shared-parameter errors on the joint selected face.

    Information is per unit observation weight. All free private parameters
    and mixture logits remain nuisance coordinates in the covariance solve.
    A shared direction coupled to nonpositive or unresolved information has
    infinite uncertainty; a direction outside the face has no ordinary
    interior standard error.

    Parameters
    ----------
    information : numpy.ndarray, shape (P, P)
        Joint observed information per unit observation weight.
    free_indices : sequence of int
        Distinct joint-coordinate indices belonging to the selected face.
    shared_indices : tuple of (int or None, int or None)
        Physical lower/upper shared-amplitude indices; ``None`` excludes a side.
    effective_n : float
        Positive effective observation count used to scale the covariance.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Physical-side standard errors, with infinity for unresolved directions
        and NaN for amplitudes outside the face.
    """
    h = np.asarray(information, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.all(np.isfinite(h)):
        raise ValueError("joint information must be a finite square matrix")
    free = np.asarray(free_indices, dtype=np.intp)
    if (
        free.ndim != 1
        or np.unique(free).size != free.size
        or np.any(free < 0)
        or np.any(free >= h.shape[0])
    ):
        raise ValueError("free parameter indices must be distinct and in range")
    if len(shared_indices) != 2:
        raise ValueError("shared indices must describe both physical sides")
    if not np.isfinite(effective_n) or effective_n <= 0.0:
        raise ValueError("effective sample size must be finite and positive")
    out = np.full(2, np.nan, dtype=np.float64)
    positions = {}
    for side, index in enumerate(shared_indices):
        if index is None:
            continue
        if index < 0 or index >= h.shape[0]:
            raise ValueError("shared parameter index is out of range")
        found = np.flatnonzero(free == index)
        if found.size:
            positions[side] = int(found[0])
    if not positions:
        return out

    h = h[np.ix_(free, free)]
    h = 0.5 * (h + h.T)
    scale = np.sqrt(np.maximum(np.abs(np.diag(h)), np.finfo(np.float64).tiny))
    scaled = h / scale[:, None] / scale[None, :]
    eigenvalues, vectors = np.linalg.eigh(scaled)
    top = float(np.max(np.abs(eigenvalues), initial=0.0))
    positive = eigenvalues > 1e-12 * max(top, np.finfo(np.float64).tiny)
    for side, position in positions.items():
        projection = vectors[position]
        if np.any(np.abs(projection[~positive]) > 1e-8):
            out[side] = np.inf
            continue
        variance = float(
            np.sum(np.square(projection[positive]) / eigenvalues[positive])
        )
        out[side] = np.sqrt(variance / effective_n) / scale[position]
    return out


def _mixture_fit_metadata(fitted, effective_n, p_values, /):
    """Build portable shared inference from a completed numerical mixture.

    Parameters
    ----------
    fitted : _NaturalMixtureFit
        Completed joint fit with components, selected face and information.
    effective_n : float
        Positive effective observation count for shared standard errors.
    p_values : numpy.ndarray, shape (2,)
        Physical-side boundary-selection p-values.

    Returns
    -------
    dict
        Fitted provenance, model dimensions and shared-boundary inference.
    """
    allowed = None
    active = None
    amplitudes = None
    for component in fitted.components:
        indices = (
            component.spec.physical_lower_a_index,
            component.spec.physical_upper_a_index,
        )
        component_allowed = tuple(index is not None for index in indices)
        component_amplitudes = np.array(
            [
                np.nan if index is None else float(component.params[index])
                for index in indices
            ],
            dtype=np.float64,
        )
        component_active = tuple(
            (
                False
                if index is None
                else bool(
                    component.lower_amplitude_active
                    if index == component.layout.lower_a_index
                    else component.upper_amplitude_active
                )
            )
            for index in indices
        )
        if allowed is None:
            allowed = component_allowed
            active = component_active
            amplitudes = component_amplitudes
        elif (
            allowed != component_allowed
            or active != component_active
            or not np.array_equal(amplitudes, component_amplitudes, equal_nan=True)
        ):
            raise ValueError("mixture components disagree on shared boundary state")
    if allowed is None:
        raise ValueError("a mixture fit must contain at least one component")
    for enabled, free, amplitude in zip(allowed, active, amplitudes, strict=True):
        if enabled and (not np.isfinite(amplitude) or amplitude < 0.0):
            raise ValueError(
                "shared boundary amplitudes must be finite and nonnegative"
            )
        if not free and enabled and amplitude != 0.0:
            raise ValueError("an inactive shared amplitude must be exactly zero")
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.shape != (2,):
        raise ValueError("boundary p-values must describe both physical sides")
    shared_indices = tuple(
        index if free and amplitude > 0.0 else None
        for index, free, amplitude in zip(
            fitted.shared_parameter_indices, active, amplitudes, strict=True
        )
    )
    errors = _shared_standard_errors(
        fitted.observed_information,
        fitted.free_parameter_indices,
        shared_indices,
        effective_n,
    )
    return {
        "provenance": "fitted",
        "n_parameters": int(fitted.n_parameters),
        "n_face_parameters": int(fitted.n_face_parameters),
        "shared_boundary": {
            "allowed": allowed,
            "amplitudes": tuple(map(float, amplitudes)),
            "active": active,
            "standard_errors": tuple(map(float, errors)),
            "p_values": tuple(map(float, p_values)),
            "weakly_identified": _weakly_identified_sides(amplitudes, errors),
        },
    }


def _single_fit_metadata(state, /):
    """Describe a standalone fitted face in the common model envelope.

    Parameters
    ----------
    state : numpy.void or Mapping
        Packed single-component state with physical boundary inference.

    Returns
    -------
    dict
        Fitted provenance, model dimensions and shared-boundary inference.
    """
    allowed = tuple(map(bool, state["boundary_allowed"]))
    active = (
        bool(state["lower_amplitude_active"]),
        bool(state["upper_amplitude_active"]),
    )
    if float(state["fit_direction"]) < 0.0:
        active = active[::-1]
    amplitudes = tuple(map(float, state["boundary_amplitudes"]))
    errors = tuple(map(float, state["boundary_standard_errors"]))
    return {
        "provenance": "fitted",
        "n_parameters": int(np.asarray(state["optimizer_params"]).size),
        "n_face_parameters": (
            2 + int(state["effective_curvature_degree"]) + sum(active)
        ),
        "shared_boundary": {
            "allowed": allowed,
            "amplitudes": amplitudes,
            "active": active,
            "standard_errors": errors,
            "p_values": tuple(map(float, state["boundary_p_values"])),
            "weakly_identified": _weakly_identified_sides(amplitudes, errors),
        },
    }
