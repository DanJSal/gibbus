"""A single fitted log-concave component.

:class:`_Component` exposes an immutable fitted payload and read-only queries.
The fixed fitting coordinate belongs to that payload; the common presentation
map is supplied by its owning model. Evaluation and moment caches are separate
from the bytes-backed fitted record.

Users never touch this class directly; they interact with
:class:`~gibbus._api.distribution.Distribution`.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Union

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from numpy.typing import ArrayLike, NDArray

from .._defaults import (
    BOUNDARY_ALPHA,
    NUMERIC_FAILURES,
    SUPPRESSED_WARNINGS,
    _maybe_suppress,
    _reraise_if_debug,
)
from .._fit.boundary import AUTO, _effective_n, _select_boundary_terms
from .._fit.conic_newton import _certify, _solve_natural_conic
from .._fit.degree import _DegreeSelectionConfig
from .._fit.inputs import _admissible_degrees, _normalize_univariate_fit_inputs
from .._fit.natural_objective import (
    _fit_natural_conic_intervals_auto,
    _fit_natural_conic_points_auto,
    _natural_interval_start,
    _prepare_natural_interval_objective,
    _prepare_natural_point_objective,
)
from .._postfit import analytics as _pf
from .._postfit.evaluators import (
    _pdf_func,
    _potential_base_func,
    _potential_exp_from_x_potential,
)
from .._postfit.expectation import expect_vectorized as _expect_vectorized
from .._postfit.fitted_state import (
    _canonical_boundary_amplitudes,
    _check_state_invariants,
    _pack_natural_fit,
    _structured_scalar,
)
from .._postfit.logspace import log_mass_between
from .._spectral._certify import chebyshev_lower_bound
from .._spectral._tail_integrals import TailIntegrator
from .._spectral.cdf import SpectralCDF, density_spec
from .._spectral.chebyshev import chebyshev_bernstein_matrix
from .._spectral.config import _SpectralCDFOptions, _SpectralPPFOptions
from .._spectral.ppf import SpectralPPF
from .._spectral.runtime import (
    build_cdf_evaluator,
    build_ppf_evaluator,
    fallback_ppf_state,
    pack_cdf_state,
    pack_ppf_state,
)
from .views import _BaseSpaceView, _ExpSpaceView


@dataclass(slots=True)
class _Presentation:
    """One model-owned presentation context, shared by its read-only views."""

    affine: tuple[float, float] = (0.0, 1.0)
    default: str = "base"
    version: int = 0


def _natural_seed_params(seed, layout, coordinate, /):
    """Reconstruct affine natural parameters from a portable fitted state.

    The stored normalized potential is sufficient for this conversion.

    Parameters
    ----------
    seed : numpy.void or Mapping
        Portable fitted state.
    layout : _NaturalLayout
        Target natural-coordinate parameter layout.
    coordinate : _FitCoordinate
        Target fitting coordinate for the new objective.

    Returns
    -------
    numpy.ndarray or None
        Reconstructed parameters, or ``None`` when the state cannot be embedded.
    """
    q_poly = np.asarray(seed["q_poly"], dtype=np.float64).reshape(-1)
    if q_poly.size < 2:
        return None
    source_direction = float(seed["fit_direction"])
    source_scale = float(seed["fit_scale"])
    slope = source_direction * coordinate.direction * coordinate.scale / source_scale
    offset = (
        source_direction
        * (coordinate.center - float(seed["fit_center"]))
        / source_scale
    )
    q_poly = Polynomial(q_poly)(Polynomial([offset, slope])).coef
    if q_poly.size < 2:
        q_poly = np.pad(q_poly, (0, 2 - q_poly.size))
    q_d2 = np.asarray(Polynomial(q_poly).deriv(2).coef, dtype=np.float64)
    curvature = np.zeros(layout.curvature_degree + 1, dtype=np.float64)
    curvature[: min(curvature.size, q_d2.size)] = q_d2[: curvature.size]
    physical = np.asarray(seed["boundary_amplitudes"], dtype=np.float64).reshape(-1)
    if physical.size != 2:
        return None
    canonical = _canonical_boundary_amplitudes(physical, coordinate.direction)
    for index, amplitude in zip(
        (layout.lower_a_index, layout.upper_a_index), canonical, strict=True
    ):
        if index is not None and np.isnan(amplitude):
            # The target enables a boundary basis absent from the seed.  This is
            # a normal warm-start incompatibility, so decline the seed explicitly.
            return None
    return layout.pack(float(q_poly[1]), curvature, canonical)


def _lift_natural_params(source_layout, source_params, target_layout, /):
    """Embed a lower-degree natural fit into a higher-degree layout.

    Parameters
    ----------
    source_layout : _NaturalLayout
        Layout that owns ``source_params``.
    source_params : numpy.ndarray
        Natural parameters in the source layout.
    target_layout : _NaturalLayout
        Higher-degree target layout.

    Returns
    -------
    numpy.ndarray
        Parameters embedded in the target layout.
    """
    gamma, curvature, amplitudes = source_layout.unpack(source_params)
    target_curvature = np.zeros(target_layout.curvature_degree + 1, dtype=np.float64)
    target_curvature[: curvature.size] = curvature
    return target_layout.pack(gamma, target_curvature, amplitudes)


def _run_natural_fit(norm, degree_config, /):
    """Dispatch normalized public inputs to the natural conic fitter.

    Boundary terms whose policy is ``"auto"`` are decided by
    :func:`gibbus._fit.boundary._select_boundary_terms`: the fit with the term
    is tested against the nested fit without it at the same degree, and when
    a term is dropped under automatic degree the degree is chosen again.

    Parameters
    ----------
    norm : Mapping
        Canonicalized single-component fit inputs.
    degree_config : _DegreeSelectionConfig
        Explicit policy reused by automatic candidates and boundary refits.

    Returns
    -------
    objective : object
        Natural objective of the final fit.
    result : _ConicNewtonResult
        Completed conic-Newton result.
    effective_n : float
        Effective sample size of the observation weights.
    p_values : tuple of float
        Boundary-term test p-values per physical side (``nan`` where no test
        ran).
    """
    degree = norm["poly_degree"]
    is_auto = isinstance(degree, str) and degree.lower() == "auto"
    support = norm["support"]
    lower = norm["log_boundary_lower"]
    upper = norm["log_boundary_upper"]
    weights = norm["weights"]
    rows = np.asarray(norm["samples_rk"], dtype=np.float64)
    effective_n = _effective_n(rows.shape[0], weights)

    def fit_auto(lo, up):
        if rows.shape[1] == 1:
            return _fit_natural_conic_points_auto(
                support, rows[:, 0], lo, up, weights, degree_config=degree_config
            )
        return _fit_natural_conic_intervals_auto(
            support, rows, lo, up, weights, degree_config=degree_config
        )

    def fit_fixed(target, lo, up):
        return _fit_natural_fixed_degree(norm, int(target), lo, up)

    if lower != AUTO and upper != AUTO:
        objective, result = (
            fit_auto(bool(lower), bool(upper))
            if is_auto
            else fit_fixed(degree, bool(lower), bool(upper))
        )
        return objective, result, effective_n, (np.nan, np.nan)

    def amplitude(model, side):
        objective, result = model
        index = (
            objective.spec.physical_lower_a_index
            if side == "lower"
            else objective.spec.physical_upper_a_index
        )
        return 0.0 if index is None else float(result.params[index])

    (objective, result), _, p_values = _select_boundary_terms(
        (
            (lambda lo, up: fit_auto(lo, up))
            if is_auto
            else (lambda lo, up: fit_fixed(degree, lo, up))
        ),
        lambda model: float(model[1].objective_value),
        amplitude,
        lower,
        upper,
        effective_n,
        fit_reduced=lambda model, lo, up: fit_fixed(
            model[0].spec.requested_poly_degree, lo, up
        ),
        refit=fit_auto if is_auto else None,
        alpha=BOUNDARY_ALPHA,
    )
    return objective, result, effective_n, p_values


def _fit_natural_fixed_degree(norm, degree, lower, upper, /):
    """Fit one fixed degree through its admissible lower-degree ladder.

    Parameters
    ----------
    norm : Mapping
        Canonicalized single-component fit inputs.
    degree : int
        Target polynomial degree.
    lower, upper : bool
        Boundary-term flags.

    Returns
    -------
    tuple
        Natural objective and completed conic-Newton result.
    """
    support = norm["support"]
    weights = norm["weights"]
    rows = np.asarray(norm["samples_rk"], dtype=np.float64)
    degree = int(degree)
    seed = norm.get("seed_state")
    previous = None
    degrees = [degree] if seed is not None else _admissible_degrees(support, degree)
    final = None
    for rung in degrees:
        if rows.shape[1] == 1:
            objective = _prepare_natural_point_objective(
                support, rows[:, 0], rung, lower, upper, weights
            )
        else:
            objective = _prepare_natural_interval_objective(
                support, rows, rung, lower, upper, weights
            )
        initial = initial_blocks = None
        if previous is not None:
            prev_objective, prev_result = previous
            initial = _lift_natural_params(
                prev_objective.layout, prev_result.params, objective.layout
            )
        elif seed is not None:
            initial = _natural_seed_params(
                seed, objective.layout, objective.spec.coordinate
            )
        if rows.shape[1] == 2 and initial is None:
            initial, initial_blocks = _natural_interval_start(objective)
        try:
            result = _solve_natural_conic(
                objective, initial=initial, initial_blocks=initial_blocks
            )
        except RuntimeError as exc:
            if (
                previous is None
                or "cone description failed its rank checks" not in str(exc)
            ):
                raise
            # The requested model contains every lower-degree face.  At
            # extreme endpoint/data-scale ratios the exact shifted monomial
            # description of a higher face can lose numerical rank even though
            # the already-fitted lower face is perfectly valid.  Preserve that
            # certified member of the requested model rather than introducing
            # an optimizer-specific rescue path.
            prev_objective, prev_result = previous
            lifted = _lift_natural_params(
                prev_objective.layout, prev_result.params, objective.layout
            )
            evaluation = objective(lifted)
            result = replace(
                prev_result,
                params=lifted,
                objective_value=float(evaluation.nll),
                evaluation=evaluation,
                blocks=(),
                dual=np.empty(0, dtype=np.float64),
                final_separation=_certify(objective.layout, lifted),
            )
            if rung < degree:
                # Keep attempting later admissible faces; a skipped odd face
                # need not imply the requested target face is unusable.
                previous = (objective, result)
                final = previous
                continue
        previous = (objective, result)
        final = previous
    return final


class _Component:
    """Internal single-component log-concave density estimator.

    This class handles the fitting, state management, and evaluation of
    a single log-concave component.  It is not part of the public API;
    users interact with :class:`Distribution` instead.
    """

    __slots__ = (
        "_base_base_potential",
        "_base_cdf",
        "_base_exp_potential",
        "_base_pdf",
        "_base_ppf",
        "_base_view",
        "_canonical_moments",
        "_center",
        "_context",
        "_data",
        "_direction",
        "_exp_view",
        "_scale",
        "_spectral_cdf_eval",
        "_spectral_ppf_eval",
        "_tail_integrator",
        "_window",
    )

    def __init__(self, uni_state: Mapping[str, float | NDArray] | None = None):
        """Privately construct a component from an immutable fitted payload.

        Parameters
        ----------
        uni_state : Mapping or None, optional
            Structured fit state to load.  ``None`` leaves the
            component unfitted.
        """
        # Fitted storage is backed by immutable bytes, not a reversible
        # ndarray writeability flag.
        self._data: np.ndarray | None = None
        self._window: Any = None
        self._center: float = float("nan")
        self._scale: float = float("nan")
        self._direction: float = 1.0
        self._base_pdf: Any = None
        self._base_cdf: Any = None
        self._base_ppf: Any = None
        self._spectral_cdf_eval: Any = None
        self._spectral_ppf_eval: Any = None
        self._base_base_potential: Any = None
        self._base_exp_potential: Any = None
        self._tail_integrator: Any = None
        self._base_view = _BaseSpaceView(self)
        self._exp_view = _ExpSpaceView(self)
        self._context = None
        self._canonical_moments = {}
        if uni_state is not None:
            self._assign_from_struct(uni_state)

    def __copy__(self):
        """Return a component copy through the standard copy protocol."""
        return self.copy()

    def __deepcopy__(self, memo):
        """Return an independent deep copy.

        Parameters
        ----------
        memo : dict
            Standard :mod:`copy` memo dictionary.

        Returns
        -------
        object
            A copy sharing no mutable state with the original.
        """
        return self.copy()

    @property
    def is_fitted(self) -> bool:
        """Whether the instance holds a fitted density."""
        return self._data is not None

    @property
    def default(self) -> str:
        """Currently active evaluation space."""
        return "base" if self._context is None else self._context.default

    @property
    def _version(self):
        """Return the owner presentation's cache-invalidation revision."""
        return 0 if self._context is None else self._context.version

    def _affine(self):
        """Return the owner presentation's shift/scale, or standalone identity."""
        return (0.0, 1.0) if self._context is None else self._context.affine

    @property
    def base(self) -> _BaseSpaceView:
        """Base-space view."""
        return self._base_view

    @property
    def exp(self) -> _ExpSpaceView:
        """Exp-space view."""
        return self._exp_view

    @classmethod
    def _fit(
        cls,
        samples: ArrayLike,
        *,
        poly_degree: int | str | None,
        support: tuple[float, float] | None,
        log_boundary_lower: bool | None,
        log_boundary_upper: bool | None,
        verbose: int,
        suppress_warnings: bool,
        init_from: Union["_Component", Mapping[str, Any]] | None,
        sample_weights: ArrayLike | None,
        degree_config: _DegreeSelectionConfig,
    ) -> "_Component":
        """Privately fit and return a new completed read-only component.

        Parameters
        ----------
        samples : array_like
            Observations.
        poly_degree : int, ``'auto'``, or None
            Degree of the polynomial potential. Explicit ``None``
            means ``"auto"`` when no seed is given, or inherit from the
            seed when ``init_from`` is provided.  Pass ``'auto'`` to
            select degree by the omitted-information criterion.
        support : tuple of (float, float) or None
            Domain of the density.  ``None`` means the full real line
            ``(-inf, +inf)``; specify structural boundaries explicitly.
            Ignored when ``init_from`` is given (inherited from seed).
        log_boundary_lower, log_boundary_upper : bool or None
            Whether to include a log-singularity boundary term at the
            lower / upper finite endpoint. Explicit ``None`` selects the term
            from the data at a finite unseeded endpoint, excludes it at an
            infinite endpoint, or inherits the seed policy with ``init_from``.
            Explicit ``True``/``False`` overrides the seed.
            An enabled basis has a direct nonnegative fitted amplitude
            that is allowed to optimize to zero.
        verbose : int
            Verbosity level for fitting progress and diagnostics.
        suppress_warnings : bool
            Whether numerical fitting warnings should be suppressed.
        init_from : _Component, Mapping, or None
            Warm-start seed.
        sample_weights : array_like or None
            Optional non-negative relative weight assigned to each observation.
        degree_config : _DegreeSelectionConfig
            Explicit omitted-information policy owned by the fit request.

        Returns
        -------
        _Component
            A new fitted component.

        Raises
        ------
        ValueError
            If inputs are invalid (bad shapes, unsupported ``poly_degree``,
            samples outside support, etc.).
        RuntimeError
            If the fit encounters a degenerate numerical state.
        """
        if isinstance(init_from, _Component):
            init_from = init_from._seed_state()
        norm = _normalize_univariate_fit_inputs(
            _Component,
            samples,
            poly_degree,
            support,
            log_boundary_lower,
            log_boundary_upper,
            verbose,
            suppress_warnings,
            init_from,
            sample_weights,
        )
        with _maybe_suppress(norm["suppress_warnings"], SUPPRESSED_WARNINGS):
            objective, result, effective_n, p_values = _run_natural_fit(
                norm, degree_config
            )
            data = _pack_natural_fit(
                objective, result, effective_n=effective_n, boundary_p_values=p_values
            )
        return cls(data)

    # -- Convenience delegation to active view --

    def pdf(self, x):
        """Evaluate the density in the active space.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).

        Returns
        -------
        float or numpy.ndarray
        """
        return self._active.pdf(x)

    def cdf(self, x):
        """Evaluate the cumulative distribution in the active space.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).

        Returns
        -------
        float or numpy.ndarray
        """
        return self._active.cdf(x)

    def ppf(self, p):
        """Evaluate the quantile function in the active space.

        Parameters
        ----------
        p : array_like
            Probabilities in ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._active.ppf(p)

    def logpdf(self, x):
        """Evaluate the active-space log density.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.logpdf(x)

    def logcdf(self, x):
        """Evaluate the active-space log CDF.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.logcdf(x)

    def sf(self, x):
        """Evaluate the active-space survival function.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.sf(x)

    def logsf(self, x):
        """Evaluate the active-space log survival function.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.logsf(x)

    def isf(self, p):
        """Evaluate the active-space inverse survival function.

        Parameters
        ----------
        p : array_like
            Survival probabilities.
        """
        return self._active.isf(p)

    def logppf(self, log_p):
        """Evaluate active-space quantiles from log CDF probabilities.

        Parameters
        ----------
        log_p : array_like
            Log probabilities.
        """
        return self._active.logppf(log_p)

    def logisf(self, log_p):
        """Evaluate active-space upper quantiles from log survival probabilities.

        Parameters
        ----------
        log_p : array_like
            Log probabilities.
        """
        return self._active.logisf(log_p)

    def log_hazard(self, x):
        """Evaluate the active-space log hazard.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.log_hazard(x)

    def hazard(self, x, n: int = 0):
        """Evaluate the active-space hazard or first derivative.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        n : {0, 1}, optional
            Derivative order.
        """
        return self._active.hazard(x, n=n)

    def cumulative_hazard(self, x):
        """Evaluate the active-space cumulative hazard.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        """
        return self._active.cumulative_hazard(x)

    def mean_residual_life(self, x):
        """Evaluate active-space mean residual life.

        Parameters
        ----------
        x : array_like
            Conditioning threshold(s).
        """
        return self._active.mean_residual_life(x)

    def residual_entropy(self, x):
        """Evaluate active-space residual entropy.

        Parameters
        ----------
        x : array_like
            Conditioning threshold(s).
        """
        return self._active.residual_entropy(x)

    def interval(self, level):
        """Return an active-space equal-tailed interval.

        Parameters
        ----------
        level : float
            Probability mass.
        """
        return self._active.interval(level)

    def hpd(self, level):
        """Return an active-space highest-density region.

        Parameters
        ----------
        level : float
            Probability mass.
        """
        return self._active.hpd(level)

    def expect(self, func):
        """Compute an active-space expectation.

        Parameters
        ----------
        func : callable
            Scalar function of the random variable.
        """
        return self._active.expect(func)

    def entropy(self):
        """Return active-space differential entropy."""
        return self._active.entropy()

    def cross_entropy(self, other):
        """Return active-space cross entropy.

        Parameters
        ----------
        other : _Component
            Other fitted component in the same active space.
        """
        if not isinstance(other, _Component) or other.default != self.default:
            raise ValueError(
                "cross_entropy requires a component in the same active space"
            )
        return self._active.cross_entropy(other._active)

    def kl_divergence(self, other):
        """Return active-space KL divergence.

        Parameters
        ----------
        other : _Component
            Other fitted component in the same active space.
        """
        if not isinstance(other, _Component) or other.default != self.default:
            raise ValueError(
                "kl_divergence requires a component in the same active space"
            )
        return self._active.kl_divergence(other._active)

    def loglik(self, x, sample_weight=None):
        """Return active-space held-out log likelihood.

        Parameters
        ----------
        x : array_like
            Exact or interval observations.
        sample_weight : array_like or None, optional
            Non-negative row weights.
        """
        return self._active.loglik(x, sample_weight=sample_weight)

    def quantile_residuals(self, x, rng=None):
        """Return active-space randomized quantile residuals.

        Parameters
        ----------
        x : array_like
            Exact or interval observations.
        rng : optional
            Random-number source.
        """
        return self._active.quantile_residuals(x, rng=rng)

    def tail_rate(self, side):
        """Return the base-space tail rate.

        Parameters
        ----------
        side : {'lower', 'upper'}
            Tail to inspect.
        """
        return self._active.tail_rate(side)

    def log_concavity_margin(self) -> float:
        """Return a certified base-space lower bound on ``q''``.

        The full curvature ``p(z) + a_L / (z - L)^2 + a_U / (U - z)^2`` was
        certified nonnegative by the exact separator when the fit finished,
        so zero is a valid bound.  The boundary terms are nonnegative, so on a
        bounded support a positive Bernstein-subdivision bound on the
        polynomial part ``p`` sharpens it; on an unbounded support a
        non-constant ``p`` has no finite positive bound.  A fit whose
        certificate failed reports the polynomial bound alone (``-inf`` on an
        unbounded support).
        """
        data = self._ensure_fitted()
        certified = bool(int(data["separator_certified"]))
        floor = 0.0 if certified else -np.inf
        q_poly = np.asarray(data["q_poly"], dtype=np.float64)
        q2 = np.polynomial.polynomial.polyder(q_poly, 2)
        if q2.size == 0:
            return 0.0

        # Drop exact trailing zero coefficients so the certificate sees the
        # true polynomial degree rather than the requested maximum degree.
        while q2.size > 1 and q2[-1] == 0.0:
            q2 = q2[:-1]

        mu_eff, sigma_eff = self._mu_sigma_eff()
        lo, hi = map(float, self.base.support)
        with np.errstate(invalid="ignore"):
            z_bounds = np.sort(
                np.asarray(
                    [
                        sigma_eff * lo + mu_eff,
                        sigma_eff * hi + mu_eff,
                    ],
                    dtype=np.float64,
                )
            )

        if q2.size == 1:
            lower = max(floor, float(q2[0]))
        elif np.all(np.isfinite(z_bounds)):
            cheb = Polynomial(q2).convert(
                kind=Chebyshev, domain=[float(z_bounds[0]), float(z_bounds[1])]
            )
            coeff = np.asarray(cheb.coef, dtype=np.float64)
            lower = max(
                floor,
                float(
                    chebyshev_lower_bound(
                        coeff,
                        chebyshev_bernstein_matrix(coeff.size - 1),
                        12,
                    )
                ),
            )
        else:
            lower = floor
        return float((sigma_eff * sigma_eff) * lower)

    def neg_log(self, x, n: int = 0):
        """Evaluate the negative log-density or one of its derivatives.

        Parameters
        ----------
        x : array_like
            Evaluation point(s).
        n : int, optional
            Derivative order (default ``0``, the potential itself).

        Returns
        -------
        float or numpy.ndarray
        """
        return self._active.neg_log(x, n)

    def sample(self, size=None, rng=None):
        """Draw samples from the fitted density in the active space.

        Parameters
        ----------
        size : int or None, optional
            Number of draws; ``None`` returns a single float.  Must be a
            non-negative integer.  Integral floats such as ``4.0`` are
            accepted; non-integral floats, strings and ``bool`` are
            rejected rather than silently truncated.
        rng : None, int, Generator, or RandomState, optional
            Random-number source.

        Returns
        -------
        float or numpy.ndarray
        """
        return self._active.sample(size=size, rng=rng)

    def moment(
        self, k: int, central: bool = False, standardized: bool = False
    ) -> float:
        """Compute the *k*-th moment in the active space.

        Parameters
        ----------
        k : int
            Moment order.
        central : bool, optional
            Take the moment about the mean.
        standardized : bool, optional
            Divide the central moment by ``std ** k``.

        Returns
        -------
        float
        """
        return self._active.moment(k, central=central, standardized=standardized)

    def cumulant(self, k: int) -> float:
        """Return the *k*-th cumulant in the active space.

        Parameters
        ----------
        k : int
            Positive cumulant order.

        Returns
        -------
        float
            Requested cumulant.

        Raises
        ------
        RuntimeError
            If the component is not fitted.
        ValueError
            If *k* is not a positive integer.
        """
        return self._active.cumulant(k)

    @property
    def support(self):
        """Return support bounds in the active evaluation space."""
        return self._active.support

    @property
    def mode(self):
        """Return the mode in the active evaluation space."""
        return self._active.mode

    @property
    def median(self):
        """Return the median in the active evaluation space."""
        return self._active.median

    @property
    def mean(self):
        """Return the mean in the active evaluation space."""
        return self._active.mean

    @property
    def var(self):
        """Return variance in the active evaluation space."""
        return self._active.var

    @property
    def std(self):
        """Return standard deviation in the active evaluation space."""
        return self._active.std

    @property
    def skew(self):
        """Return skewness of the active-space distribution."""
        return self._active.skew

    @property
    def kurt(self):
        """Return excess kurtosis of the active-space distribution."""
        return self._active.kurt

    def _truncate_base(self, lower, upper):
        """Return this component conditioned to a base-space interval.

        Parameters
        ----------
        lower, upper : float or None
            Base-space truncation bounds. ``None`` keeps the current bound.
        """
        self._ensure_fitted()
        if self._context is not None:
            shift, scale = self._affine()
            return _Component(self._data)._truncate_base(
                None if lower is None else (float(lower) - shift) / scale,
                None if upper is None else (float(upper) - shift) / scale,
            )
        active_lo, active_hi = map(float, self.base.support)
        lo = active_lo if lower is None else max(active_lo, float(lower))
        hi = active_hi if upper is None else min(active_hi, float(upper))
        if not lo < hi:
            raise ValueError(
                f"truncation interval ({lower!r}, {upper!r}) does not overlap "
                f"support ({active_lo!r}, {active_hi!r})"
            )

        log_mass = float(
            log_mass_between(
                self.base.logcdf(lo),
                self.base.logcdf(hi),
                self.base.logsf(lo),
                self.base.logsf(hi),
            )
        )
        if not np.isfinite(log_mass) or log_mass < np.log(np.finfo(np.float64).tiny):
            raise RuntimeError(
                f"truncation retained negligible probability mass (log mass={log_mass!r})"
            )

        old = self.data
        data = {name: np.array(old[name], copy=True) for name in old.dtype.names}
        q_poly = np.asarray(data["q_poly"], dtype=np.float64).copy()
        q_poly[0] += log_mass
        data["q_poly"] = q_poly
        data["support"] = np.array([lo, hi], dtype=np.float64)

        mu_eff, sigma_eff = self._mu_sigma_eff()
        base_bounds = np.array([lo, hi], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            z_bounds = sigma_eff * base_bounds + mu_eff
        trunc_z_support = np.sort(z_bounds)

        original_z_support, canonical_amps = self._internal_model_geometry()
        old_z_mode = float(data["canonical_mode"])
        z_mode = float(np.clip(old_z_mode, trunc_z_support[0], trunc_z_support[1]))
        crm = np.asarray(data["canonical_raw_moments"], dtype=np.float64)
        if crm.size >= 3 and np.all(np.isfinite(crm[1:3])):
            z_std = float(np.sqrt(max(0.0, crm[2] - crm[1] ** 2)))
        else:
            z_std = 1.0
        if np.all(np.isfinite(trunc_z_support)):
            z_std = min(
                max(z_std, np.finfo(float).tiny),
                max(
                    (trunc_z_support[1] - trunc_z_support[0]) / 2.0,
                    np.finfo(float).tiny,
                ),
            )
        z_std = max(z_std, np.sqrt(np.finfo(float).tiny))

        zl, zu = (float(v) for v in original_z_support)
        density = density_spec(
            [
                (
                    q_poly,
                    zl,
                    zu,
                    float(canonical_amps[0]),
                    float(canonical_amps[1]),
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    -np.inf,
                    np.inf,
                )
            ],
            view=False,
        )
        cdf_rep = SpectralCDF(
            trunc_z_support,
            density=density,
            mode=z_mode,
            std=z_std,
            map_scale=None,
            initial_breaks=None,
            config=_SpectralCDFOptions(),
        )
        spectral = dict(pack_cdf_state(cdf_rep))
        try:
            ppf_rep = SpectralPPF(cdf_rep, config=_SpectralPPFOptions())
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, "truncated spectral PPF construction", routine=True)
            ppf_rep = None
            spectral.update(fallback_ppf_state())
            spectral["ppf_fallback"] = np.int32(1)
        else:
            spectral.update(pack_ppf_state(ppf_rep))
            spectral["ppf_fallback"] = np.int32(0)
        data.update(spectral)
        data["canonical_mode"] = z_mode
        data["mode"] = float(np.clip(self.base.mode, lo, hi))

        # First construct a temporary component so its packed spectral inverse
        # can drive the conditional moments and median.
        temp = _Component(_structured_scalar(data))
        base_view = temp.base
        raw = np.full_like(np.asarray(data["raw_moments"], dtype=np.float64), np.nan)
        craw = np.full_like(
            np.asarray(data["canonical_raw_moments"], dtype=np.float64), np.nan
        )
        if raw.size:
            raw[0] = 1.0
        if craw.size:
            craw[0] = 1.0
        upto = min(4, raw.size - 1, craw.size - 1)
        for k in range(1, upto + 1):
            raw[k] = _expect_vectorized(
                base_view.neg_log,
                base_view.support,
                lambda x, kk=k: x**kk,
                points=(base_view.mode,),
            )
            craw[k] = _expect_vectorized(
                base_view.neg_log,
                base_view.support,
                lambda x, kk=k: (sigma_eff * x + mu_eff) ** kk,
                points=(base_view.mode,),
            )
        data["raw_moments"] = raw
        data["canonical_raw_moments"] = craw
        if upto >= 4:
            stats = _pf._stats_from_raw_moments(*map(float, raw[1:5]))
            for name in ("mean", "var", "std", "skew", "kurt"):
                data[name] = float(stats[name])
        data["median"] = float(base_view.ppf(0.5))

        # Keep the moment quadrature window inside the active truncation
        # support. Quantile-defined edges remain finite on unbounded sides.
        try:
            qlo = lo if np.isfinite(lo) else float(base_view.ppf(1e-12))
            qhi = hi if np.isfinite(hi) else float(base_view.isf(1e-12))
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, "truncated moment-window quantiles", routine=True)
        else:
            with np.errstate(invalid="ignore"):
                zwin = np.sort(
                    np.array(
                        [
                            sigma_eff * qlo + mu_eff,
                            sigma_eff * qhi + mu_eff,
                        ],
                        dtype=np.float64,
                    )
                )
            if np.all(np.isfinite(zwin)) and zwin[0] < zwin[1]:
                data["window"] = zwin

        result = _Component(_structured_scalar(data))
        return result

    @property
    def data(self):
        """A deep copy of the structured fitted state."""
        self._ensure_fitted()
        return self._data.copy()

    def _seed_state(self):
        """Materialize seed conditioning in the common current model coordinate."""
        state = self.data
        shift, scale = self._affine()
        state["support"] = self.base.support
        state["fit_center"] = shift + scale * self._center
        state["fit_scale"] = scale * self._scale
        for name in ("mode", "median", "mean", "var", "std", "skew", "kurt"):
            state[name] = self._base_stat(name)
        raw = state["raw_moments"]
        for order in np.flatnonzero(np.isfinite(raw)):
            raw[order] = self._raw_moment_base(int(order))
        return state

    def copy(self) -> "_Component":
        """Create an independent deep copy.

        Returns
        -------
        _Component
            A component sharing no mutable state with this one.
        """
        new = _Component()
        if self._data is None:
            return new
        new._assign_from_struct(self._data)
        new._context = None if self._context is None else replace(self._context)
        return new

    @property
    def _active(self):
        """Return the currently active space view."""
        return self._base_view if self.default == "base" else self._exp_view

    def _raw_moment_base(self, k: int) -> float:
        """Compute (or retrieve cached) *k*-th raw moment in base coordinates.

        Parameters
        ----------
        k : int
            Moment order.

        Returns
        -------
        float
        """
        self._ensure_fitted()
        mu_eff, sigma_eff = self._mu_sigma_eff()
        moments = [self._canonical_raw_moment(j) for j in range(k + 1)]
        return float(
            _pf._powaff_moment_from_z_moments(
                moments, 1.0 / sigma_eff, -mu_eff / sigma_eff, k
            )
        )

    def _canonical_raw_moment(self, k: int) -> float:
        """Compute or retrieve a raw moment in the fitted canonical coordinate.

        Parameters
        ----------
        k : int
            Non-negative canonical raw-moment order.
        """
        data = self._ensure_fitted()
        if k == 0:
            return 1.0
        rm = data["canonical_raw_moments"]
        cap = rm.shape[0]
        if k < cap and np.isfinite(rm[k]):
            return float(rm[k])
        if k in self._canonical_moments:
            return self._canonical_moments[k]
        val = float(_pf._univariate_canonical_raw_moment(data, k))
        self._canonical_moments[k] = val
        return val

    def _base_mean_parts(self):
        """Return ``(common_origin, local_mean)`` for the base affine map."""
        origin, scale = self._affine()
        z_mean = float(self._canonical_raw_moment(1))
        local = (self._center + self._direction * self._scale * z_mean) * scale
        return float(origin), float(local)

    def _support_base(self):
        """Return active base support under the shared presentation transform."""
        shift, scale = self._affine()
        return shift + scale * self._data["support"]

    def _base_stat(self, name):
        """Transform a stored base-space summary statistic.

        Parameters
        ----------
        name : str
            Stored statistic field to express in the presentation coordinate.

        Returns
        -------
        float
            Statistic after applying the appropriate affine transformation.
        """
        value = float(self._data[name])
        shift, scale = self._affine()
        if name in ("mode", "median", "mean"):
            return shift + scale * value
        if name == "var":
            return scale * scale * value
        if name == "std":
            return scale * value
        return value

    def _assign_from_struct(self, struct):
        """Populate evaluation state from one packed fitted state.

        Parameters
        ----------
        struct : Mapping or numpy.void
            Packed single-component fitted state.

        Returns
        -------
        None
        """
        data = np.array(struct, copy=True)
        _check_state_invariants(data)
        self._data = np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(())
        self._window = self._data["window"]
        self._center = float(self._data["fit_center"])
        self._scale = float(self._data["fit_scale"])
        self._direction = float(self._data["fit_direction"])
        internal_support = self._data["canonical_support"]
        kernel_boundary = _canonical_boundary_amplitudes(
            self._data["boundary_amplitudes"], self._direction
        )
        self._base_pdf = _pdf_func(
            internal_support, self._data["q_poly"], kernel_boundary
        )
        self._spectral_cdf_eval = build_cdf_evaluator(self._data)
        self._spectral_ppf_eval = build_ppf_evaluator(self._data)
        self._base_cdf = self._spectral_cdf_eval
        self._base_ppf = self._spectral_ppf_eval.eval_x
        self._base_base_potential = _potential_base_func(
            internal_support, self._data["q_poly"], kernel_boundary
        )
        self._base_exp_potential = _potential_exp_from_x_potential(
            lambda x, n: self._base_view.neg_log(x, n)
        )
        self._tail_integrator = TailIntegrator(self._data["q_poly"])

    def _internal_model_geometry(self):
        """Return the canonical support and kernel-order boundary amplitudes."""
        return (
            np.asarray(self._data["canonical_support"], dtype=np.float64),
            _canonical_boundary_amplitudes(
                self._data["boundary_amplitudes"], self._direction
            ),
        )

    def _ensure_fitted(self) -> np.ndarray:
        """Return the packed fitted record, raising if there is none.

        Returns
        -------
        numpy.ndarray
            The component's packed fitted-state record.

        Raises
        ------
        RuntimeError
            If the component is not fitted.
        """
        if self._data is None:
            raise RuntimeError("Component has no fitted payload.")
        return self._data

    def _mu_sigma_eff(self):
        """Compute the effective affine parameters mapping user *x* to internal *z*."""
        direction = float(self._direction)
        shift, scale = self._affine()
        mu_eff = direction * (-shift / scale - self._center) / self._scale
        sigma_eff = direction / (scale * self._scale)
        return mu_eff, sigma_eff

    def _potential_support_base(self):
        """Return the original potential support in current base coordinates."""
        self._ensure_fitted()
        z_support = np.asarray(self._data["canonical_support"], dtype=np.float64)
        mu_eff, sigma_eff = self._mu_sigma_eff()
        with np.errstate(invalid="ignore"):
            x_support = (z_support - mu_eff) / sigma_eff
        return np.sort(np.asarray(x_support, dtype=np.float64))
