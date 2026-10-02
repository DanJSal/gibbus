"""Packing and validation of the portable fitted-state record.

The state records the fixed fitting coordinate, normalized natural potential,
solver diagnostics, and downstream numerical representations used by public
evaluation and warm starts.
"""

import numpy as np

from .._defaults import (
    MAX_CACHED_MOMENTS,
    NUMERIC_FAILURES,
    PPF_BISECT_MAX_ITER,
    PPF_BISECT_Z_TOL,
    _reraise_if_debug,
)
from .._fit.boundary import _amplitude_standard_errors
from .._model.coords import _FitCoordinate
from .._spectral.cdf import (
    SpectralCDF,
    boundary_aware_breaks_from_amplitudes,
    density_spec,
)
from .._spectral.ppf import SpectralPPF
from .._spectral.runtime import fallback_ppf_state, pack_cdf_state, pack_ppf_state
from .analytics import _powaff_moment_from_z_moments, _stats_from_raw_moments

# Fields every fitted state must carry; the spectral CDF/PPF evaluators check
# their own array geometry when they are rebuilt.
_STATE_FIELDS = (
    "q_poly", "boundary_amplitudes", "boundary_allowed", "support",
    "canonical_support", "window", "canonical_mode", "mode", "median", "mean",
    "var", "std", "skew", "kurt", "raw_moments", "canonical_raw_moments",
    "fit_center", "fit_scale", "fit_direction", "mu", "sigma", "pullback",
    "default_space", "optimizer_params", "requested_poly_degree",
    "effective_poly_degree", "nll", "cdf_map_params", "cdf_npanels",
    "cdf_coeff_stride", "cdf_breaks", "cdf_offsets", "cdf_coeffs", "ppf_pmin",
    "ppf_pmax", "ppf_npanels", "ppf_coeff_stride", "ppf_breaks_r",
    "ppf_breaks_z", "ppf_coeffs", "boundary_standard_errors",
    "boundary_p_values", "optimizer_success", "optimizer_status",
    "optimizer_message", "optimizer_n_iterations", "optimizer_n_evaluations",
    "optimizer_subproblem_iterations", "optimizer_decrease_bound",
    "effective_curvature_degree", "lower_amplitude_active",
    "upper_amplitude_active", "separator_certified",
)


def _median_z_from_cdf(cdf_rep, /):
    """Return the compact-coordinate median of a spectral CDF.

    Parameters
    ----------
    cdf_rep : SpectralCDF
        Built forward spectral CDF representation.

    Returns
    -------
    float
        Compact coordinate in ``[-1, 1]`` whose CDF is one half.
    """
    lo, hi = -1.0, 1.0
    for _ in range(PPF_BISECT_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi or (hi - lo) <= PPF_BISECT_Z_TOL:
            break
        if float(np.asarray(cdf_rep.cdf_z(mid)).reshape(-1)[0]) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _canonical_boundary_amplitudes(boundary_amplitudes, direction, /):
    """Map physical boundary amplitudes into canonical lower/upper order.

    Parameters
    ----------
    boundary_amplitudes : array_like, shape (2,)
        Physical lower/upper zero-offset log amplitudes.
    direction : float
        Fitting-coordinate orientation, ``+1`` or ``-1``.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Canonical lower/upper amplitudes; reflection swaps the physical
        endpoint order.
    """
    amps = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")
    return amps.copy() if float(direction) > 0.0 else amps[::-1].copy()


def _user_support(coordinate, /):
    """Return the fixed coordinate's support in user coordinates.

    The support the user asked for, exactly: mapping the canonical endpoints
    back through the affine coordinate can land an ulp outside it (a lower
    endpoint of ``0.0`` coming back as ``-1.1e-16``).

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed fitting coordinate.

    Returns
    -------
    numpy.ndarray
        Ordered physical support.
    """
    if not isinstance(coordinate, _FitCoordinate):
        raise TypeError("coordinate must be a _FitCoordinate")
    return np.asarray(coordinate.physical_support, dtype=np.float64)


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


def _pack_natural_state(state, spec, coord, result, /, *, effective_n=np.nan,
                        boundary_p_values=(np.nan, np.nan)):
    """Pack a normalized natural state plus its conic-solver record.

    Parameters
    ----------
    state : _NaturalCoreState
        Normalized model state built from natural parameters.
    spec : _ModelSpec
        Model specification used by the fit.
    coord : _FitCoordinate
        Fixed fitting coordinate.
    result : _ConicNewtonResult
        Completed natural conic solve.
    effective_n : float, optional
        Effective sample size behind the fit; ``nan`` skips the amplitude
        standard errors.
    boundary_p_values : tuple of float, optional
        Physical-side p-values of the automatic boundary-term tests (``nan``
        where no test ran).

    Returns
    -------
    numpy.void
        Portable fitted-state scalar.
    """
    if not np.isfinite(state.log_Z):
        raise RuntimeError("cannot pack a nonnormalizable natural fit")

    q_poly = np.asarray(state.q_poly, dtype=np.float64).copy()
    q_poly[0] += float(state.log_Z)

    canonical_amps = np.nan_to_num(
        np.asarray(state.boundary_amplitudes, dtype=np.float64), nan=0.0
    )
    amps = _canonical_boundary_amplitudes(canonical_amps, coord.direction)
    allowed = np.array([
        spec.physical_lower_a_index is not None,
        spec.physical_upper_a_index is not None,
    ], dtype=bool)

    z_mom = np.asarray(state.moments.power(4), dtype=np.float64)
    alpha = float(coord.direction * coord.scale)
    beta = float(coord.center)
    try:
        x_mom = np.array(
            [_powaff_moment_from_z_moments(z_mom, alpha, beta, k) for k in range(5)],
            dtype=np.float64,
        )
    except OverflowError as exc:
        _reraise_if_debug(exc, "post-fit raw-moment conversion")
        raise RuntimeError(
            "cannot represent fitted raw moments at this data scale; "
            "rescale the observations to a numerically moderate range and refit"
        ) from exc
    z_stats = _stats_from_raw_moments(*map(float, z_mom[1:5]))
    alpha_sign = -1.0 if alpha < 0.0 else 1.0
    stats = {
        "mean": float(beta + alpha * z_stats["mean"]),
        "var": float((alpha * alpha) * z_stats["var"]),
        "std": float(abs(alpha) * z_stats["std"]),
        "skew": float(alpha_sign * z_stats["skew"]),
        "kurt": float(z_stats["kurt"]),
    }

    z_std_sq = max(0.0, float(z_mom[2] - z_mom[1] * z_mom[1]))
    z_std = float(np.sqrt(z_std_sq))
    initial_breaks = boundary_aware_breaks_from_amplitudes(
        spec.support, canonical_amps
    )
    support_z = np.asarray(spec.support, dtype=np.float64)
    density = density_spec(
        [(state.q_poly, support_z[0], support_z[1], canonical_amps[0],
          canonical_amps[1], float(state.log_Z), 0.0, 1.0, 1.0, 1.0,
          -np.inf, np.inf)],
        view=False,
    )
    cdf_rep = SpectralCDF(
        spec.support,
        density=density,
        mode=float(state.mode),
        std=z_std,
        initial_breaks=initial_breaks,
    )
    spectral_state = dict(pack_cdf_state(cdf_rep))
    try:
        ppf_rep = SpectralPPF(cdf_rep)
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "spectral PPF construction", routine=True)
        ppf_rep = None
        spectral_state.update(fallback_ppf_state())
        spectral_state["ppf_fallback"] = np.int32(1)
    else:
        spectral_state.update(pack_ppf_state(ppf_rep))
        spectral_state["ppf_fallback"] = np.int32(0)

    z_median = (
        float(ppf_rep.ppf(0.5)) if ppf_rep is not None
        else float(_median_z_from_cdf(cdf_rep))
    )
    support = _user_support(coord)
    mode = float(coord.from_canonical(float(state.mode)))
    median = float(coord.from_canonical(z_median))
    raw_moments = np.full(int(MAX_CACHED_MOMENTS), np.nan, dtype=np.float64)
    raw_moments[:5] = x_mom
    canonical_raw_moments = np.full(
        int(MAX_CACHED_MOMENTS), np.nan, dtype=np.float64
    )
    canonical_raw_moments[:5] = z_mom

    converged = str(result.status) in {"converged", "converged_approximately"}
    data = {
        "q_poly": q_poly,
        "boundary_amplitudes": amps,
        "boundary_allowed": allowed,
        "support": support,
        "canonical_support": np.asarray(spec.support, dtype=np.float64),
        "window": np.asarray(state.window, dtype=np.float64),
        "canonical_mode": float(state.mode),
        "mode": mode,
        "median": median,
        "mean": float(stats["mean"]),
        "var": float(stats["var"]),
        "std": float(stats["std"]),
        "skew": float(stats["skew"]),
        "kurt": float(stats["kurt"]),
        "raw_moments": raw_moments,
        "canonical_raw_moments": canonical_raw_moments,
        "fit_center": float(coord.center),
        "fit_scale": float(coord.scale),
        "fit_direction": float(coord.direction),
        "mu": 0.0,
        "sigma": 1.0,
        "pullback": True,
        "default_space": np.str_("base"),
        "optimizer_params": np.asarray(result.params, dtype=np.float64),
        "requested_poly_degree": int(spec.requested_poly_degree),
        "effective_poly_degree": int(spec.effective_poly_degree),
        "nll": float(result.objective_value),
        "optimizer_success": np.int8(converged),
        "optimizer_status": np.str_(result.status),
        "optimizer_message": np.str_(result.status.replace("_", " ")),
        "optimizer_n_iterations": np.int64(result.newton_iterations),
        "optimizer_n_evaluations": np.int64(result.objective_evaluations),
        "optimizer_subproblem_iterations": np.int64(result.subproblem_iterations),
        "optimizer_decrease_bound": float(result.final_decrease_bound),
        "effective_curvature_degree": np.int64(result.effective_curvature_degree),
        "lower_amplitude_active": np.int8(bool(result.lower_amplitude_active)),
        "upper_amplitude_active": np.int8(bool(result.upper_amplitude_active)),
        "separator_certified": np.int8(
            result.final_separation is None or bool(result.final_separation.feasible)
        ),
    }
    evaluation = result.evaluation
    information = getattr(evaluation, "observed_hessian", None)
    if information is None and evaluation is not None:
        information = evaluation.hessian
    standard_errors = (
        np.full(2, np.nan, dtype=np.float64)
        if information is None or not np.isfinite(effective_n)
        else _amplitude_standard_errors(
            result.params, information, spec.layout, spec,
            int(result.effective_curvature_degree), float(effective_n),
        )
    )
    data["boundary_standard_errors"] = standard_errors
    data["boundary_p_values"] = np.asarray(boundary_p_values, dtype=np.float64).reshape(2)
    data.update(spectral_state)
    return _structured_scalar(data)


def _pack_natural_fit(objective, result, /, *, effective_n=np.nan,
                      boundary_p_values=(np.nan, np.nan)):
    """Pack one completed natural-coordinate fit for public evaluation.

    Parameters
    ----------
    objective : _NaturalPointObjectiveFunction or _NaturalIntervalObjectiveFunction
        Objective that fixed the fitting coordinate and natural layout.
    result : _ConicNewtonResult
        Completed conic Newton solve.
    effective_n : float, optional
        Effective sample size behind the fit.
    boundary_p_values : tuple of float, optional
        Automatic boundary-term test p-values per physical side.

    Returns
    -------
    numpy.void
        Portable fitted-state scalar.
    """
    return _pack_natural_state(
        objective.build_state(result.params),
        objective.spec,
        objective.spec.coordinate,
        result,
        effective_n=effective_n,
        boundary_p_values=boundary_p_values,
    )


def _pack_natural_component(component, /, *, effective_n=np.nan,
                            boundary_p_values=(np.nan, np.nan)):
    """Pack one finalized component of a natural mixture fit.

    Parameters
    ----------
    component : _NaturalMixtureComponent
        Final natural mixture component.
    effective_n : float, optional
        Effective sample size behind the component (its responsibility-
        weighted observations).
    boundary_p_values : tuple of float, optional
        Automatic boundary-term test p-values of the mixture per physical side.

    Returns
    -------
    numpy.void
        Portable fitted-state scalar.
    """
    result = component.solver_result
    return _pack_natural_state(
        component.state(), component.spec, component.coordinate, result,
        effective_n=effective_n, boundary_p_values=boundary_p_values,
    )


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

    sigma = float(struct["sigma"])
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"state sigma must be finite and positive, got {sigma}")
    scale = float(struct["fit_scale"])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"state fit_scale must be finite and positive, got {scale}")
    if not np.isfinite(float(struct["mu"])) or not np.isfinite(float(struct["fit_center"])):
        raise ValueError("state mu and fit_center must be finite")
    if float(struct["fit_direction"]) not in (-1.0, 1.0):
        raise ValueError("state fit_direction must be -1 or +1")
    if str(struct["default_space"]) not in ("base", "exp"):
        raise ValueError("state default_space must be 'base' or 'exp'")

    q_poly = field("q_poly")
    if q_poly.size < 3 or not np.all(np.isfinite(q_poly)):
        raise ValueError("state q_poly must hold at least three finite coefficients")
    amplitudes = field("boundary_amplitudes")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("state boundary_amplitudes must be finite and non-negative")
    for name in ("support", "canonical_support", "window"):
        bounds = field(name)
        if np.any(np.isnan(bounds)) or not bounds[0] < bounds[1]:
            raise ValueError(f"state {name} must satisfy lower < upper")
    if not np.all(np.isfinite(field("window"))):
        raise ValueError("state window must be finite")

    npanels = int(struct["cdf_npanels"])
    stride = int(struct["cdf_coeff_stride"])
    if npanels > 0 and stride > 0:
        _check_finite_increasing(field("cdf_breaks")[: npanels + 1], "cdf_breaks")
        if not np.all(np.isfinite(field("cdf_offsets")[:npanels])):
            raise ValueError("cdf_offsets must be finite")
        if not np.all(np.isfinite(field("cdf_coeffs")[: npanels * stride])):
            raise ValueError("cdf_coeffs must be finite")
        params = field("cdf_map_params")
        if np.any(np.isnan(params)) or not np.all(np.isfinite(params[2:])):
            raise ValueError("cdf_map_params must be finite apart from support endpoints")

    pmin = float(struct["ppf_pmin"])
    pmax = float(struct["ppf_pmax"])
    ppf_panels = int(struct["ppf_npanels"])
    ppf_stride = int(struct["ppf_coeff_stride"])
    if pmin < pmax and ppf_panels > 0 and ppf_stride > 0:
        # An empty [pmin, pmax] is the documented fallback that routes every
        # query through CDF inversion, and carries no usable panel geometry.
        if not (0.0 <= pmin < pmax <= 1.0):
            raise ValueError("ppf_pmin and ppf_pmax must satisfy 0 <= pmin < pmax <= 1")
        _check_finite_increasing(field("ppf_breaks_r")[: ppf_panels + 1], "ppf_breaks_r")
        _check_finite_increasing(field("ppf_breaks_z")[: ppf_panels + 1], "ppf_breaks_z")
        if not np.all(np.isfinite(field("ppf_coeffs")[: ppf_panels * ppf_stride])):
            raise ValueError("ppf_coeffs must be finite")
