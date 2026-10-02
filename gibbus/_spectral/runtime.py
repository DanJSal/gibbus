"""Packed runtime helpers for the spectral CDF and PPF.

The fitting pipeline stores only NumPy-compatible scalar/1-D fields in the
structured fitted state.  This module reconstructs the compiled evaluators
from those fields.  Both ordinary and extreme-tail CDF/PPF queries are resolved
by the compiled spectral kernels directly; the fitted runtime stores no interpolation grid.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as C

from ._cdf_eval import SpectralEvaluator
from ._ppf_eval import SpectralPPFEvaluator

_KIND_TO_CODE = {"finite": 0, "lower": 1, "upper": 2, "real": 3, "lower_centered": 4, "upper_centered": 5}


def pack_cdf_state(rep):
    """Return structured-state fields for a built ``SpectralCDF``.

    Panel antiderivatives are shifted to vanish at each panel's left edge
    and divided by the total mass, so the packed coefficients describe a
    normalized CDF directly and the evaluator needs no runtime scaling.

    Parameters
    ----------
    rep : gibbus._spectral.cdf.SpectralCDF
        Built spectral CDF representation.

    Returns
    -------
    dict
        Field name to value, ready to merge into the fitted structured
        scalar.  Array shapes vary with the panel count, so the enclosing
        dtype is built per fit.
    """
    m = len(rep.panels)
    max_ncoeff = max(p.icoeff.size for p in rep.panels)
    coeffs = np.zeros((m, max_ncoeff), dtype=np.float64)
    ncoeff = np.empty(m, dtype=np.int32)
    offsets = np.asarray(rep.cum_mass[:-1], dtype=np.float64).copy()

    for j, panel in enumerate(rep.panels):
        ic = np.asarray(panel.icoeff, dtype=np.float64).copy()
        ic[0] -= C.chebval(-1.0, ic)
        ic /= rep.total_mass
        coeffs[j, :ic.size] = ic
        ncoeff[j] = ic.size

    # Convergence diagnostics. Construction degrades rather than raises
    # when a density resists refinement, so without these a silently
    # under-resolved CDF is indistinguishable from a converged one.
    # ``refinement_capped`` reports only the mechanism.  Public severity is
    # expressed on the probability scale: uncertified mass, normalization
    # defect, certified local error, and their combined CDF-health estimate.
    worst_error = float(rep.worst_panel_error)
    hit_cap = bool(rep.max_depth_used >= int(rep.max_depth)
                   or rep.panel_budget_exhausted)

    mp = rep.map
    return {
        "cdf_max_depth_used": np.int32(rep.max_depth_used),
        "cdf_refinement_capped": np.int32(1 if hit_cap else 0),
        "cdf_worst_panel_error": np.float64(worst_error),
        "cdf_uncertified_mass": np.float64(rep.uncertified_mass_fraction),
        "cdf_mass_defect": np.float64(rep.mass_defect),
        "cdf_error_estimate": np.float64(rep.cdf_error_estimate),
        "cdf_map_kind": np.int32(_KIND_TO_CODE[mp.kind]),
        "cdf_map_params": np.array([mp.L, mp.U, mp.center, mp.scale], dtype=np.float64),
        "cdf_npanels": np.int32(m),
        "cdf_coeff_stride": np.int32(max_ncoeff),
        "cdf_breaks": np.asarray(rep.breaks, dtype=np.float64).copy(),
        "cdf_offsets": offsets,
        "cdf_ncoeff": ncoeff,
        "cdf_coeffs": coeffs.reshape(-1),
    }


def pack_ppf_state(rep):
    """Return structured-state fields for a built ``SpectralPPF``.

    Parameters
    ----------
    rep : gibbus._spectral.ppf.SpectralPPF
        Built spectral inverse representation.

    Returns
    -------
    dict
        Field name to value, ready to merge into the fitted structured
        scalar.
    """
    m = len(rep.panels)
    max_ncoeff = max(p.coeff.size for p in rep.panels)
    coeffs = np.zeros((m, max_ncoeff), dtype=np.float64)
    ncoeff = np.empty(m, dtype=np.int32)
    for j, panel in enumerate(rep.panels):
        c = np.asarray(panel.coeff, dtype=np.float64)
        coeffs[j, :c.size] = c
        ncoeff[j] = c.size
    breaks_z = np.array(
        [rep.panels[0].za] + [panel.zb for panel in rep.panels],
        dtype=np.float64,
    )

    return {
        "ppf_pmin": np.float64(rep.pmin),
        "ppf_pmax": np.float64(rep.pmax),
        "ppf_npanels": np.int32(m),
        "ppf_coeff_stride": np.int32(max_ncoeff),
        "ppf_breaks_r": np.asarray(rep.breaks_r, dtype=np.float64).copy(),
        "ppf_breaks_z": breaks_z,
        "ppf_ncoeff": ncoeff,
        "ppf_coeffs": coeffs.reshape(-1),
    }


def fallback_ppf_state():
    """Return PPF fields that force every query through CDF inversion.

    ``SpectralPPF`` raises rather than degrades when it cannot certify an
    inverse panel monotone, which is the right call for the panels
    themselves -- an uncertified inverse could be non-monotone -- but a
    fit should not die because its quantile *representation* failed while
    its density is perfectly good.

    The compiled evaluator already resolves probabilities outside
    ``[pmin, pmax]`` by monotone bisection of the packed spectral CDF,
    which is exact if slower.  Packing an empty probability range routes
    every query down that path, so quantiles stay correct and the fit
    survives.

    Returns
    -------
    dict
        Field name to value, shaped like :func:`pack_ppf_state` output.
    """
    return {
        # Empty range: pmin > pmax, so no p is ever "inside".
        "ppf_pmin": np.float64(1.0),
        "ppf_pmax": np.float64(0.0),
        "ppf_npanels": np.int32(1),
        "ppf_coeff_stride": np.int32(2),
        "ppf_breaks_r": np.array([0.0, 0.0], dtype=np.float64),
        "ppf_breaks_z": np.array([-1.0, 1.0], dtype=np.float64),
        "ppf_ncoeff": np.array([2], dtype=np.int32),
        "ppf_coeffs": np.zeros(2, dtype=np.float64),
    }


def build_cdf_evaluator(data):
    """Reconstruct the compiled CDF evaluator from a fitted state.

    Parameters
    ----------
    data : numpy.void
        Fitted structured scalar containing the ``cdf_*`` fields written
        by :func:`pack_cdf_state`.

    Returns
    -------
    _cdf_eval.SpectralEvaluator
        Compiled evaluator for bulk CDF queries.
    """
    m = int(data["cdf_npanels"])
    stride = int(data["cdf_coeff_stride"])
    kind = int(data["cdf_map_kind"])
    if m <= 0:
        raise ValueError("cdf_npanels must be positive")
    if stride <= 0:
        raise ValueError("cdf_coeff_stride must be positive")
    if not 0 <= kind <= 5:
        raise ValueError("cdf_map_kind must be an integer in 0..5")
    coeffs = np.asarray(data["cdf_coeffs"], dtype=np.float64).reshape(m, stride)
    params = np.asarray(data["cdf_map_params"], dtype=np.float64)
    return SpectralEvaluator(
        kind,
        float(params[0]), float(params[1]), float(params[2]), float(params[3]),
        np.asarray(data["cdf_breaks"], dtype=np.float64),
        np.asarray(data["cdf_offsets"], dtype=np.float64),
        coeffs,
        np.asarray(data["cdf_ncoeff"], dtype=np.int32),
    )


def build_ppf_evaluator(data):
    """Reconstruct the compiled full compact-coordinate PPF evaluator.

    The PPF evaluator also receives the packed CDF panels, because
    probabilities outside the fitted logit range are resolved by direct
    inversion of the spectral CDF rather than by the inverse panels.

    Parameters
    ----------
    data : numpy.void
        Fitted structured scalar containing the ``ppf_*`` and ``cdf_*``
        fields.

    Returns
    -------
    _ppf_eval.SpectralPPFEvaluator
        Compiled evaluator for bulk PPF queries.
    """
    m = int(data["ppf_npanels"])
    stride = int(data["ppf_coeff_stride"])
    if m <= 0:
        raise ValueError("ppf_npanels must be positive")
    if stride <= 0:
        raise ValueError("ppf_coeff_stride must be positive")
    coeffs = np.asarray(data["ppf_coeffs"], dtype=np.float64).reshape(m, stride)
    cdf_m = int(data["cdf_npanels"])
    cdf_stride = int(data["cdf_coeff_stride"])
    kind = int(data["cdf_map_kind"])
    if cdf_m <= 0:
        raise ValueError("cdf_npanels must be positive")
    if cdf_stride <= 0:
        raise ValueError("cdf_coeff_stride must be positive")
    if not 0 <= kind <= 5:
        raise ValueError("cdf_map_kind must be an integer in 0..5")
    cdf_coeffs = np.asarray(data["cdf_coeffs"], dtype=np.float64).reshape(
        cdf_m, cdf_stride
    )
    params = np.asarray(data["cdf_map_params"], dtype=np.float64)
    return SpectralPPFEvaluator(
        float(data["ppf_pmin"]), float(data["ppf_pmax"]),
        kind,
        float(params[0]), float(params[1]), float(params[2]), float(params[3]),
        np.asarray(data["ppf_breaks_r"], dtype=np.float64),
        np.asarray(data["ppf_breaks_z"], dtype=np.float64),
        coeffs,
        np.asarray(data["ppf_ncoeff"], dtype=np.int32),
        np.asarray(data["cdf_breaks"], dtype=np.float64),
        np.asarray(data["cdf_offsets"], dtype=np.float64),
        cdf_coeffs,
        np.asarray(data["cdf_ncoeff"], dtype=np.int32),
    )
