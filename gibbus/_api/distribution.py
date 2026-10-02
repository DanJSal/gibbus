"""The public :class:`Distribution` model.

This module holds the public API surface and nothing else.  The pieces it
composes live beside it:

* :mod:`.views` — ``base`` / ``exp`` coordinate-space views.
* :mod:`.component` — a single fitted component and its lite EM state.
* :mod:`.fitting` — fit-request preparation and fitting orchestration.
* :mod:`.mixture_stats` — mixture moments, potentials and spectral cache.
* :mod:`.diagnostics` — fit, selection and spectral diagnostic records.

Base vs exp space
-----------------
* **base** space operates on the modelled variable *x* directly.
* **exp** space treats the modelled variable as *y = exp(x)*, so the fitted
  log-concave density on *x* induces a density on the positive half-line via
  the standard change-of-variables formula.

The active space, used by convenience methods such as :meth:`Distribution.pdf`, is
controlled by :meth:`Distribution.set_default` and is ``"base"`` after a fresh
:meth:`~Distribution.fit`.

Affine transforms
-----------------
After fitting, the density can be shifted and scaled through
:meth:`Distribution.transform`.  The ``pullback`` flag determines whether
``(mu, sigma)`` are interpreted as pushforward (location-scale) parameters or
as pullback (internal coordinate) parameters; see that method's docstring.
"""

import copy as _copy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from .._defaults import BOOTSTRAP_DEFAULT_RESAMPLES, PROB_EPS
from .._fit.inputs import _coerce_sample_size, _to_generator
from .._fit.mixture import (
    _find_mixture_modes_exp,
    _pack_mixture_struct,
    _sort_components_by_mode,
    _unpack_mixture_struct,
)
from .._postfit.analytics import _cumulant_from_centered, _exp_moment_from_stats
from .._postfit.expectation import expect as _expect
from .._postfit.expectation import expect_vectorized as _expect_vectorized
from .._postfit.gof import asymptotic_pvalue as _asymptotic_gof_pvalue
from .._postfit.gof import gof_statistic as _gof_statistic
from .._postfit.gof import monte_carlo_pvalue as _monte_carlo_pvalue
from .._postfit.gof import validate_statistic as _validate_gof_statistic
from .._postfit.information import kl_divergence as _kl_divergence
from .._postfit.logspace import log_mass_between
from .._postfit.regions import hpd as _hpd
from .._postfit.regions import interval as _interval
from .._postfit.resample import bootstrap_curves as _bootstrap_curves
from .._postfit.resample import simulated_statistics as _simulated_statistics
from .._postfit.scoring import canonical_scoring_rows as _canonical_scoring_rows
from .._postfit.scoring import interval_loglik as _interval_loglik
from .._postfit.scoring import randomized_quantile_residuals as _randomized_quantile_residuals
from .._postfit.scoring import scoring_weights as _scoring_weights
from .._postfit.scoring import warn_out_of_support as _warn_out_of_support
from .._postfit.survival import cdf_hybrid, log_cdf_hybrid, log_sf_hybrid
from .._postfit.survival import isf as _isf
from .._postfit.survival import logisf as _logisf
from .._postfit.survival import logppf as _logppf
from .._postfit.survival import mean_residual_life as _mean_residual_life
from .._postfit.survival import residual_entropy as _residual_entropy
from .._spectral.tail import refine_tail_quantiles
from .component import _Component
from .diagnostics import _DiagnosticsMixin
from .fitting import _FitRequest, _prepare_fit_request, _run_fit_request
from .frozen import FrozenDistribution
from .mixture_stats import _MixtureAnalyticsMixin
from .views import _BaseSpaceView, _ExpSpaceView


def _restore_distribution_pickle(cls, state, em_diagnostics, selection_diagnostics):
    """Reconstruct a pickled Distribution while preserving session diagnostics.

    Parameters
    ----------
    state : numpy.void
        Serialized fitted state.
    em_diagnostics : object
        Stored EM diagnostic record.
    selection_diagnostics : object
        Stored component-selection diagnostic record.
    """
    obj = cls(state)
    obj._em_diagnostics = em_diagnostics
    obj._selection_diagnostics = selection_diagnostics
    return obj


class Distribution(_MixtureAnalyticsMixin, _DiagnosticsMixin):
    """Flexible maximum-likelihood model for univariate probability distributions.

    ``Distribution`` is the sole public class of the ``gibbus`` package.  It fits
    one or more smooth log-concave components to point or interval-censored
    data.  Each component uses an explicit support-aware analytic potential
    representation whose polynomial degree controls shape flexibility and whose
    optional finite-boundary logarithmic terms capture endpoint behavior.
    The fitted object exposes density and distribution evaluation, sampling,
    moments, tail and information quantities, diagnostics, and affine
    transformations in two coordinate views:

    * :attr:`base` -- operates on the modelled variable *x* directly.
    * :attr:`exp`  -- treats *y = exp(x)* as the modelled variable, useful
      when data are naturally positive (e.g. eigenvalues, prices).

    For ``n_components=1``, the density is unimodal.  For
    ``n_components > 1``, a mixture of log-concave components is fitted
    via Expectation-Maximisation (EM), supporting multimodal data.

    The currently active view is selected by :meth:`set_default` (``"base"``
    after a fresh fit) and determines the behaviour of the top-level
    convenience methods :meth:`pdf`, :meth:`cdf`, :meth:`ppf`, etc.

    A fitted instance may be shared between threads after its lazy caches have
    been warmed; concurrent first-use cache construction is not synchronized.

    Parameters
    ----------
    state : numpy.void or None, optional
        A previously saved structured state (as returned by :attr:`data`).
        If provided, the object is initialised as if :meth:`load` had been
        called.  If ``None`` (default), an unfitted instance is created.

    Examples
    --------
    Fit a single-component (unimodal) density::

        import numpy as np
        from gibbus import Distribution

        rng = np.random.default_rng(0)
        samples = rng.normal(size=500)
        c = Distribution().fit(samples, support=(-np.inf, np.inf))
        c.mean

    Fit a multimodal density::

        bimodal = np.concatenate([rng.normal(-2, 0.5, 300),
                                  rng.normal(2, 0.5, 300)])
        c2 = Distribution().fit(bimodal, n_components=2,
                                support=(-np.inf, np.inf))
        c2.pdf(0.0)   # valley between the two modes

    Save and reload state::

        state = c.data
        c3 = Distribution(state)
        np.isclose(c.base.pdf(0.0), c3.base.pdf(0.0))   # True
    """

    def __init__(self, state=None) -> None:
        """Create a Distribution, optionally loading a saved state.

        Parameters
        ----------
        state : Mapping or None, optional
            Structured fit state, single- or multi-component.  ``None``
            leaves the model unfitted.
        """
        # An empty component list is the unfitted state.
        self._components: List[_Component] = []
        self._weights: np.ndarray = np.ones(0, dtype=np.float64)
        self._default: str = "base"
        self._K: int = 0

        # Mixture-level caches (used only when K > 1).
        self._mix_base_cdf = None
        self._mix_base_ppf = None
        self._mix_spectral_cdf_rep = None
        self._mix_spectral_ppf_rep = None
        self._stats_cache: Optional[dict] = None
        self._cumulant_cache: dict[tuple[str, int], float] = {}
        self._fit_diagnostics_cache = None
        self._mode_cache: Optional[dict] = None
        self._spectral_cache_valid: bool = False
        self._em_diagnostics = None
        self._selection_diagnostics = None

        if state is not None:
            self.load(state)

    def __copy__(self):
        """Return an independent copy rather than sharing mutable fit caches."""
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

    def __repr__(self):
        """Return a compact fitted/unfitted model representation."""
        if not self.is_fitted:
            return "Distribution(unfitted)"
        degrees = tuple(
            int(np.asarray(comp.data["q_poly"]).size - 1)
            for comp in self._components
        )
        support = tuple(map(float, self._components[0].base.support))
        return (f"Distribution(n_components={self._K}, support={support}, "
                f"poly_degrees={degrees}, default={self._default!r})")

    def __reduce__(self):
        """Serialize fitted state together with in-process diagnostic records."""
        if not self.is_fitted:
            return (type(self), ())
        return (
            _restore_distribution_pickle,
            (type(self), self.data, self._em_diagnostics, self._selection_diagnostics),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the instance holds a fitted density."""
        return bool(self._components)

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        self._ensure_fitted()
        return self._K

    @property
    def weights(self) -> np.ndarray:
        """Mixture weights, shape ``(K,)``, summing to one."""
        self._ensure_fitted()
        return self._weights.copy()

    @property
    def components(self) -> List[_Component]:
        """Fitted components (read-only list)."""
        self._ensure_fitted()
        return list(self._components)

    @property
    def default(self) -> str:
        """Currently active evaluation space (``'base'`` or ``'exp'``)."""
        return str(self._default)

    @property
    def base(self) -> _BaseSpaceView:
        """Base-space view (single-component only).

        For multi-component models, use top-level methods directly or
        access individual component views via ``c.components[k].base``.

        Returns
        -------
        _BaseSpaceView
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].base
        raise AttributeError(
            "Multi-component Distribution has no single .base view. "
            "Use .pdf(), .cdf() etc. directly, or .components[k].base.")

    @property
    def exp(self) -> _ExpSpaceView:
        """Exp-space view (single-component only).

        For multi-component models, use top-level methods directly or
        access individual component views via ``c.components[k].exp``.

        Returns
        -------
        _ExpSpaceView
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].exp
        raise AttributeError(
            "Multi-component Distribution has no single .exp view. "
            "Use .pdf(), .cdf() etc. directly, or .components[k].exp.")

    # ------------------------------------------------------------------
    # Space selection
    # ------------------------------------------------------------------

    def set_default(self, space: str) -> "Distribution":
        """Set the default evaluation space.

        Parameters
        ----------
        space : {'base', 'exp'}
            Coordinate space to make active: ``"base"`` or ``"exp"``.

        Returns
        -------
        Distribution
            ``self``, for method chaining.
        """
        s = str(space).lower()
        if s not in ("base", "exp"):
            raise ValueError("space must be 'base' or 'exp'")
        self._default = s
        self._stats_cache = None
        self._cumulant_cache.clear()
        self._fit_diagnostics_cache = None
        for comp in self._components:
            comp.set_default(s)
        return self

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        samples: ArrayLike,
        *,
        n_components: Union[int, str] = "auto",
        poly_degree: Optional[Union[int, str]] = None,
        support: Optional[Tuple[float, float]] = None,
        log_boundary_lower: Optional[bool] = None,
        log_boundary_upper: Optional[bool] = None,
        verbose: int = 0,
        suppress_warnings: bool = False,
        init_from: Optional["Distribution"] = None,
        sample_weights: Optional[ArrayLike] = None,
        component_options: Optional[List[Dict[str, Any]]] = None,
        em_max_iter: Optional[int] = None,
        em_tol: Optional[float] = None,
        rng: Any = None,
        k_max: Optional[int] = None,
        progressive: bool = True,
        auto_k_subsample: Union[str, int, bool] = "auto",
    ) -> "Distribution":
        """Fit a log-concave density to the supplied samples.

        Parameters
        ----------
        samples : array_like
            Observations.  Accepted layouts:

            * ``(R,)`` or ``(R, 1)`` -- finite point samples.
            * ``(R, 2)`` -- interval-censored samples.  Endpoints may be
              infinite for one-sided censoring but may not be NaN.

        n_components : int or ``'auto'``, optional
            Number of mixture components (default ``'auto'``).
            Ignored when ``init_from`` is given (inherited from seed).
            ``'auto'`` counts KDE modes to centre a search range, picks
            the best *K* by BIC over lightweight log-concave fits, then
            runs the full log-concave EM only once for the chosen *K*.
            ``1`` fits a single unimodal density directly.  An explicit
            integer ``> 1`` skips selection and fits exactly that many
            components.
        poly_degree : int, ``'auto'``, or None, optional
            Degree of the polynomial potential.  ``None`` (default)
            means ``"auto"`` when no seed is given, or inherit from the
            seed when ``init_from`` is provided.  For mixtures, this
            may be overridden per component via ``component_options``.
        support : tuple of (float, float) or None, optional
            Domain of the density.  ``None`` means the full real line
            ``(-inf, +inf)``.  Specify structural boundaries explicitly.
            Ignored when ``init_from`` is given (inherited from seed).
        log_boundary_lower, log_boundary_upper : bool or None, optional
            Whether to allow log-singularity boundary terms at the lower /
            upper finite endpoint.  With no seed, ``None`` (default) lets the
            data decide: a finite-endpoint term is retained only when the
            one-sided likelihood-ratio test against the nested fit without it
            has ``p < 0.05``, and it is excluded when a positive-weight
            observation lies exactly at that endpoint.  Infinite endpoints
            have no term.  With ``init_from``, ``None`` inherits the seed
            setting.  Explicit ``True``/``False`` overrides the seed.  An
            enabled basis has a direct nonnegative fitted amplitude that may
            optimize to zero.  Global across all components.
        verbose : int, optional
            Verbosity level (default ``0``).
        suppress_warnings : bool, optional
            Suppress selected numerical warnings (default ``False``).
        init_from : Distribution or None, optional
            Warm-start seed from a previously fitted ``Distribution``.  When
            given, ``n_components`` and ``support`` are inherited from
            the seed.  Per-component seeds are threaded automatically
            in seed-component order.
        sample_weights : array_like or None, optional
            Non-negative observation weights.
        component_options : list of dict or None, optional
            Per-component keyword arguments (currently only
            ``poly_degree`` is allowed).  Length must match the
            effective ``n_components``.  Must be ``None`` when
            ``n_components='auto'`` and no seed is given.
        em_max_iter : int or None, optional
            Maximum EM iterations.
        em_tol : float or None, optional
            EM convergence tolerance.
        rng : optional
            Random-number source for the stratified subsampling used by
            automatic selection and for the fallback GMM initialiser. For
            fitting, ``None`` is resolved to deterministic seed 0; pass an
            integer or generator to choose another stream.
        k_max : int or None, optional
            Maximum *K* to consider when ``n_components='auto'``.
            ``None`` (default) uses the library's own ceiling.
            Ignored for explicit *K*.
        progressive : bool, optional
            For mixture fits with an explicit integer ``poly_degree > 2``,
            use a degree ladder from the lowest admissible degree to the
            requested degree, warm-starting each rung.  This option has no
            effect on single-component fits or when ``poly_degree="auto"``.
        auto_k_subsample : {"auto"}, int, or False, optional
            Size of the subsample used to *select* the component count
            when ``n_components="auto"``.  ``"auto"`` (the default)
            subsamples only above ``AUTO_LC_SUBSAMPLE_MIN_N`` samples;
            an integer sets the size directly; ``False`` disables it.
            Candidate scoring may therefore select a different *K* than a
            full-data sweep.  Once *K* is selected, the final model is
            refitted on every observation.  Ignored unless
            ``n_components="auto"``.

        Returns
        -------
        Distribution
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If *samples* has an unsupported shape, holds fewer than two
            observations or no spread, falls outside *support*, or if
            any argument is outside its documented range.
        RuntimeError
            If the data admits no log-concave fit (heavy tails such as
            Cauchy or Pareto), if a point-data mixture component is
            effectively supported on fewer than two distinct observed
            locations (the classical unconstrained mixture-likelihood
            singularity), or if the optimiser reaches a degenerate numerical
            state.  Interval-censored mixtures are not subject to this
            point-density estimability check because their row contributions
            are probability masses rather than point densities.
        """
        self._em_diagnostics = None
        self._selection_diagnostics = None
        request = _FitRequest(
            samples=samples,
            n_components=n_components,
            poly_degree=poly_degree,
            support=support,
            log_boundary_lower=log_boundary_lower,
            log_boundary_upper=log_boundary_upper,
            verbose=verbose,
            suppress_warnings=suppress_warnings,
            init_from=init_from,
            sample_weights=sample_weights,
            component_options=component_options,
            em_max_iter=em_max_iter,
            em_tol=em_tol,
            rng=rng,
            k_max=k_max,
            progressive=progressive,
            auto_k_subsample=auto_k_subsample,
        )
        request = _prepare_fit_request(Distribution, request)
        result = _run_fit_request(request)
        self._install_fit_result(result)
        return self

    def _install_fit_result(self, result, /):
        """Install completed component state and invalidate derived caches.

        Parameters
        ----------
        result : _FitResult
            Completed fitting result returned by the API-internal fitting
            controller.
        """
        self._components = list(result.components)
        self._weights = np.asarray(result.weights, dtype=np.float64).copy()
        self._K = len(self._components)
        self._default = "base"
        self._stats_cache = None
        self._cumulant_cache.clear()
        self._fit_diagnostics_cache = None
        self._mode_cache = None
        self._spectral_cache_valid = False
        self._em_diagnostics = (
            None if result.em_diagnostics is None
            else dict(result.em_diagnostics)
        )
        self._selection_diagnostics = (
            None if result.selection_diagnostics is None
            else dict(result.selection_diagnostics)
        )
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def pdf(self, x) -> Union[float, np.ndarray]:
        """Evaluate the PDF in the currently active space.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.

        Returns
        -------
        float or numpy.ndarray
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].pdf(x)
        x = np.asarray(x, dtype=np.float64)
        scalar = (x.ndim == 0)
        out = np.zeros_like(x, dtype=np.float64)
        for k, comp in enumerate(self._components):
            view = comp.base if self._default == "base" else comp.exp
            out = out + self._weights[k] * view.pdf(x)
        return float(out) if scalar else out

    def cdf(self, x) -> Union[float, np.ndarray]:
        """Evaluate the CDF in the currently active space.

        The spectral representation serves the body while lower-tail values
        below the handover probability are replaced by exact tail quadrature.
        This preserves ordinary CDF speed without imposing the spectral
        absolute-error floor on extreme probabilities.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.

        Returns
        -------
        float or numpy.ndarray
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].cdf(x)
        pot, spectral_cdf, support = self._base_probability_parts()
        if self._default == "base":
            return cdf_hybrid(
                pot, spectral_cdf, x, float(support[0]),
                upper_endpoint=float(support[1]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            )
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.log(arr)
            out = np.asarray(cdf_hybrid(
                pot, spectral_cdf, lx, float(support[0]),
                upper_endpoint=float(support[1]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            ))
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, 0.0))
        return float(out) if scalar else out

    def ppf(self, p) -> Union[float, np.ndarray]:
        """Evaluate the PPF (quantile function) in the currently active space.

        Parameters
        ----------
        p : float or array_like
            CDF probability or probabilities in the closed interval ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray

        Raises
        ------
        ValueError
            If any element of *p* lies outside ``[0, 1]``.
        """
        self._ensure_fitted()
        p_arr = np.asarray(p, dtype=np.float64)
        scalar = (p_arr.ndim == 0)
        if np.any(~np.isfinite(p_arr) | (p_arr < 0.0) | (p_arr > 1.0)):
            raise ValueError("ppf is defined for finite p in [0, 1]")

        if self._K == 1:
            out = np.atleast_1d(
                np.asarray(self._components[0].ppf(p_arr), dtype=np.float64))
            base_out = np.log(out) if self._default == "exp" else out
        else:
            self._ensure_spectral_cache()
            base_out = np.atleast_1d(
                np.asarray(self._mix_base_ppf(p_arr), dtype=np.float64)).copy()

        base_out = self._refine_extreme_quantiles(
            np.atleast_1d(p_arr).reshape(-1),
            np.atleast_1d(base_out).reshape(-1))

        out = np.exp(base_out) if self._default == "exp" else base_out
        out = out.reshape(p_arr.shape)
        return float(out) if scalar else out

    def _refine_extreme_quantiles(self, p, base_out, /):
        """Re-solve quantiles the spectral CDF cannot resolve.

        Below ``TAIL_ASYMPTOTIC_P`` the CDF panels hold no significant
        digits, and inverting them returns a confident wrong answer
        rather than a poor one.  Those points are recomputed from the
        potential; see :mod:`gibbus._spectral.tail`.

        Parameters
        ----------
        p : numpy.ndarray, shape (R,)
            Requested probabilities.
        base_out : numpy.ndarray, shape (R,)
            Quantiles from the spectral inverse, in base coordinates.

        Returns
        -------
        numpy.ndarray, shape (R,)
            *base_out* with the extreme entries replaced.
        """
        # Work in base coordinates throughout: every component shares the
        # support, and the mixture has no single ``.base`` view.
        first = self._components[0].base
        supp = first.support
        if self._K == 1:
            potential = first.neg_log
            # ``first.ppf`` includes tail refinement, so using it as the seed
            # here would recurse.  Seed from the raw spectral inverse.
            raw_ppf = first._raw_ppf
        else:
            potential = self._mix_base_potential
            raw_ppf = self._mix_base_ppf
        tail_mass = (
            first._exact_tail_log_mass if self._K == 1
            else self._mix_base_log_tail_mass
        )
        return refine_tail_quantiles(
            potential, raw_ppf, float(supp[0]), float(supp[1]), p, base_out,
            log_tail_mass=tail_mass,
        )

    def _base_ppf_for_extensions(self, p):
        """Return base-space quantiles without depending on the active view.

        Parameters
        ----------
        p : float or array_like
            CDF probabilities in ``[0, 1]``.
        """
        p_arr = np.asarray(p, dtype=np.float64)
        scalar = p_arr.ndim == 0
        if np.any(~np.isfinite(p_arr) | (p_arr < 0.0) | (p_arr > 1.0)):
            raise ValueError("ppf is defined for finite p in [0, 1]")
        if self._K == 1:
            return self._components[0].base.ppf(p_arr)
        self._ensure_spectral_cache()
        base_out = np.atleast_1d(
            np.asarray(self._mix_base_ppf(p_arr), dtype=np.float64)).copy()
        base_out = self._refine_extreme_quantiles(
            np.atleast_1d(p_arr).reshape(-1), base_out.reshape(-1))
        out = base_out.reshape(p_arr.shape)
        return float(out) if scalar else out

    def _base_probability_parts(self):
        """Return ``(potential, cdf, support)`` in base coordinates."""
        if self._K == 1:
            view = self._components[0].base
            return view.neg_log, view.cdf, np.asarray(view.support, dtype=np.float64)
        self._ensure_spectral_cache()
        return (
            self._mix_base_potential,
            self._mix_base_cdf,
            np.asarray(self._components[0].base.support, dtype=np.float64),
        )

    def _base_log_tail_mass(self, x, endpoint, /, *, upper=False):
        """Return an exact base-space tail mass without mixture-potential quadrature.

        Parameters
        ----------
        x : float
            Tail anchor in base coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        if self._K == 1:
            return self._components[0].base._exact_tail_log_mass(
                x, endpoint, upper=upper
            )
        return self._mix_base_log_tail_mass(x, endpoint, upper=upper)

    def _base_log_tail_masses(self, x, endpoint, /, *, upper=False):
        """Batched ``_base_log_tail_mass`` over an array of anchors.

        Parameters
        ----------
        x : numpy.ndarray
            Tail anchors in base coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        if self._K == 1:
            return self._components[0].base._exact_tail_log_masses(
                x, endpoint, upper=upper
            )
        return self._mix_base_log_tail_masses(x, endpoint, upper=upper)

    def logpdf(self, x) -> Union[float, np.ndarray]:
        """Evaluate the log density in the currently active space.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].logpdf(x)
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        out = -np.asarray(self.neg_log(arr, 0), dtype=np.float64)
        out = np.where(np.isnan(arr), np.nan, out)
        return float(out) if scalar else out

    def logcdf(self, x) -> Union[float, np.ndarray]:
        """Evaluate a tail-accurate log CDF.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].logcdf(x)
        pot, cdf, support = self._base_probability_parts()
        if self._default == "base":
            return log_cdf_hybrid(
                pot, cdf, x, float(support[0]), upper_endpoint=float(support[1]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            )
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.log(arr)
            out = np.asarray(log_cdf_hybrid(
                pot, cdf, lx, float(support[0]), upper_endpoint=float(support[1]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            ))
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, -np.inf))
        return float(out) if scalar else out

    def sf(self, x) -> Union[float, np.ndarray]:
        """Evaluate survival as ``exp(logsf)``.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        scalar = np.asarray(x).ndim == 0
        with np.errstate(under="ignore"):
            out = np.exp(self.logsf(x))
        return float(out) if scalar else out

    def logsf(self, x) -> Union[float, np.ndarray]:
        """Evaluate a tail-accurate log survival function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].logsf(x)
        pot, cdf, support = self._base_probability_parts()
        if self._default == "base":
            return log_sf_hybrid(
                pot, cdf, x, float(support[1]), lower_endpoint=float(support[0]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            )
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            lx = np.log(arr)
            out = np.asarray(log_sf_hybrid(
                pot, cdf, lx, float(support[1]), lower_endpoint=float(support[0]),
                log_tail_mass=self._base_log_tail_mass,
                log_tail_masses=self._base_log_tail_masses,
            ))
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, 0.0))
        return float(out) if scalar else out

    def isf(self, p) -> Union[float, np.ndarray]:
        """Evaluate inverse survival without forming ``1-p`` in deep tails.

        Parameters
        ----------
        p : float or array_like
            Survival probabilities in ``[0, 1]``.
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].isf(p)
        pot, _cdf, support = self._base_probability_parts()
        base = _isf(
            pot, self._base_ppf_for_extensions, support, p,
            log_tail_mass=self._base_log_tail_mass,
        )
        if self._default == "base":
            return base
        scalar = np.asarray(p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(base)
        return float(out) if scalar else out

    def logppf(self, log_p) -> Union[float, np.ndarray]:
        """Evaluate quantiles from logarithmic CDF probabilities.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].logppf(log_p)
        pot, _cdf, support = self._base_probability_parts()
        base = _logppf(
            pot, self._base_ppf_for_extensions, support, log_p,
            log_tail_mass=self._base_log_tail_mass,
        )
        if self._default == "base":
            return base
        scalar = np.asarray(log_p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(base)
        return float(out) if scalar else out

    def logisf(self, log_p) -> Union[float, np.ndarray]:
        """Evaluate upper quantiles from logarithmic survival probabilities.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].logisf(log_p)
        pot, _cdf, support = self._base_probability_parts()
        base = _logisf(
            pot, self._base_ppf_for_extensions, support, log_p,
            log_tail_mass=self._base_log_tail_mass,
        )
        if self._default == "base":
            return base
        scalar = np.asarray(log_p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(base)
        return float(out) if scalar else out

    def log_hazard(self, x) -> Union[float, np.ndarray]:
        """Evaluate log hazard in the active space.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        lo, hi = map(float, self.support)
        with np.errstate(invalid="ignore"):
            out = np.asarray(self.logpdf(arr)) - np.asarray(self.logsf(arr))
        if np.isfinite(lo):
            out = np.where(arr < lo, -np.inf, out)
        if np.isfinite(hi):
            out = np.where(arr >= hi, np.inf, out)
        out = np.where(np.isnan(arr), np.nan, out)
        return float(out) if scalar else out

    def hazard(self, x, n: int = 0) -> Union[float, np.ndarray]:
        """Evaluate the hazard or its first derivative.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        n : {0, 1}, optional
            Derivative order.
        """
        self._ensure_fitted()
        n = int(n)
        if n not in (0, 1):
            raise ValueError("hazard supports only n=0 or n=1")
        scalar = np.asarray(x).ndim == 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            h = np.asarray(np.exp(self.log_hazard(x)), dtype=np.float64)
        out = h if n == 0 else h * (h - np.asarray(self.neg_log(x, 1), dtype=np.float64))
        return float(out) if scalar else out

    def cumulative_hazard(self, x) -> Union[float, np.ndarray]:
        """Evaluate cumulative hazard ``-logsf(x)``.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        self._ensure_fitted()
        scalar = np.asarray(x).ndim == 0
        out = -np.asarray(self.logsf(x), dtype=np.float64)
        return float(out) if scalar else out

    def mean_residual_life(self, x) -> Union[float, np.ndarray]:
        """Evaluate mean residual life in the active space.

        Parameters
        ----------
        x : float or array_like
            Conditioning threshold(s).
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].mean_residual_life(x)
        return _mean_residual_life(
            self.neg_log, self.logsf, self.support, self.mean, self.modes, x
        )

    def residual_entropy(self, x) -> Union[float, np.ndarray]:
        """Evaluate residual entropy in the active space.

        Parameters
        ----------
        x : float or array_like
            Conditioning threshold(s).
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].residual_entropy(x)
        return _residual_entropy(self.neg_log, self.logsf, self.support, x)

    def interval(self, level: float) -> Tuple[float, float]:
        """Return an equal-tailed interval.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        self._ensure_fitted()
        return _interval(self.ppf, self.isf, self.support, level)

    def hpd(self, level: float) -> np.ndarray:
        """Return the highest-density region as an ``(m, 2)`` array.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        self._ensure_fitted()
        return _hpd(
            self.logpdf, self.logcdf, self.logsf, self.ppf, self.isf,
            self.support, level, modes=self.modes,
        )

    def _base_expect(self, func):
        """Evaluate an expectation in base coordinates.

        Parameters
        ----------
        func : callable
            Scalar integrand as a function of the base-space variable.
        """
        pot, _cdf, support = self._base_probability_parts()
        if self._K == 1:
            points = (self._components[0].base.mode,)
        else:
            points = self._ensure_modes().get("base", ())
        return _expect(pot, support, func, points=points)

    def _vector_expect(self, func):
        """``expect`` for an integrand that accepts arrays (internal use).

        Parameters
        ----------
        func : callable
            Vectorized function of the active-space random variable.
        """
        pot, _cdf, support = self._base_probability_parts()
        if self._K == 1:
            points = (self._components[0].base.mode,)
        else:
            points = self._ensure_modes().get("base", ())
        if self._default == "base":
            return _expect_vectorized(pot, support, func, points=points)

        def transformed(x):
            with np.errstate(over="ignore"):
                return func(np.exp(x))

        return _expect_vectorized(pot, support, transformed, points=points)

    def expect(self, func) -> float:
        """Compute ``E[func(X)]`` in the active space.

        Parameters
        ----------
        func : callable
            Scalar function of the active-space random variable.
        """
        self._ensure_fitted()
        if self._default == "base":
            return self._base_expect(func)
        def transformed(x):
            with np.errstate(over="ignore"):
                y = float(np.exp(x))
            return func(y)
        return self._base_expect(transformed)

    def _expect_between(self, func, lower, upper):
        """Integrate an active-space expectation contribution over bounds.

        Parameters
        ----------
        func : callable
            Scalar function of the active-space random variable.
        lower, upper : float
            Active-space integration limits.
        """
        self._ensure_fitted()
        a = float(lower)
        b = float(upper)
        if not a < b:
            return 0.0
        pot, _cdf, base_support = self._base_probability_parts()
        if self._K == 1:
            points = (self._components[0].base.mode,)
        else:
            points = self._ensure_modes().get("base", ())
        if self._default == "base":
            lo = max(float(base_support[0]), a)
            hi = min(float(base_support[1]), b)
            if not lo < hi:
                return 0.0
            return _expect(pot, (lo, hi), func, points=points)

        with np.errstate(divide="ignore", invalid="ignore"):
            lo = float(base_support[0]) if a <= 0.0 else max(float(base_support[0]), float(np.log(a)))
            hi = float(base_support[1]) if np.isposinf(b) else min(float(base_support[1]), float(np.log(b)))
        if not lo < hi:
            return 0.0

        def transformed(x):
            with np.errstate(over="ignore"):
                y = float(np.exp(x))
            return func(y)

        return _expect(pot, (lo, hi), transformed, points=points)

    def entropy(self) -> float:
        """Return differential entropy in the active space."""
        self._ensure_fitted()
        pot, _cdf, support = self._base_probability_parts()
        if self._K == 1:
            points = (self._components[0].base.mode,)
        else:
            points = self._ensure_modes().get("base", ())
        base_entropy = _expect_vectorized(pot, support, lambda x: pot(x, 0), points=points)
        if self._default == "base":
            return float(base_entropy)
        base_mean = float(sum(
            w * comp.base.mean for w, comp in zip(self._weights, self._components, strict=True)
        ))
        return float(base_entropy + base_mean)

    def cross_entropy(self, other) -> float:
        """Return cross entropy against another fitted Distribution in the same view.

        Parameters
        ----------
        other : Distribution
            Other fitted distribution with the same active coordinate view.
        """
        self._ensure_fitted()
        if not isinstance(other, Distribution) or not other.is_fitted:
            raise ValueError("other must be a fitted Distribution")
        if other.default != self.default:
            raise ValueError("cross_entropy requires matching active spaces")
        lo, hi = map(float, self.support)
        olo, ohi = map(float, other.support)
        if lo < olo or hi > ohi:
            return np.inf
        return self._vector_expect(lambda x: other.neg_log(x, 0))

    def kl_divergence(self, other) -> float:
        """Return ``D_KL(self || other)``.

        Parameters
        ----------
        other : Distribution
            Other fitted distribution with the same active coordinate view.
        """
        self._ensure_fitted()
        if not isinstance(other, Distribution) or not other.is_fitted:
            raise ValueError("other must be a fitted Distribution")
        if other.default != self.default:
            raise ValueError("kl_divergence requires matching active spaces")
        pot, _cdf, support = self._base_probability_parts()
        opot, _ocdf, osupport = other._base_probability_parts()
        points = self._mode_cache.get("base", ()) if self._mode_cache else ()
        return _kl_divergence(pot, support, opot, osupport, points=points)

    def _log_mass(self, lo, hi):
        """Return active-space log mass of ``(lo, hi]``.

        Parameters
        ----------
        lo, hi : float
            Interval endpoints in the active coordinate space.
        """
        return float(log_mass_between(
            self.logcdf(lo), self.logcdf(hi), self.logsf(lo), self.logsf(hi)
        ))

    def loglik(self, x, sample_weight=None) -> float:
        """Return total held-out log likelihood.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        sample_weight : array_like or None, optional
            Non-negative row weights; values are not normalized.
        """
        self._ensure_fitted()
        rows = _canonical_scoring_rows(x)
        _warn_out_of_support(rows, self.support)
        weights = _scoring_weights(rows.shape[0], sample_weight)
        if rows.shape[1] == 1:
            positive = weights > 0.0
            terms = np.asarray(self.logpdf(rows[:, 0]), dtype=np.float64)
            return float(np.sum(weights[positive] * terms[positive], dtype=np.float64))
        return _interval_loglik(rows, weights, self.logpdf, self.logcdf, self.logsf)

    def goodness_of_fit(self, x, *, statistic: str = "cvm",
                        calibration: str = "asymptotic",
                        n_resamples: int = BOOTSTRAP_DEFAULT_RESAMPLES,
                        rng=None, fit_kwargs=None) -> Dict[str, Any]:
        """Test the fitted distribution against observations.

        Transforms *x* through the fitted CDF and measures how far the
        result is from uniform.  The statistic is always meaningful; the
        *p*-value is only meaningful under the stated calibration, which
        the returned ``pvalue_valid_for`` field records so it cannot be
        read off without the caveat.

        ``"asymptotic"`` uses the classical null distribution, which
        assumes the model never saw *x*.  Applied to the training sample
        its p-values are miscalibrated toward overly large values because
        the fit has already moved toward those points.  Use it for held-out
        data.  ``"montecarlo"`` simulates from the fitted model, refits each
        simulation, and calibrates against that null, which is intended for
        in-sample checks at the cost of *n_resamples* refits.

        Parameters
        ----------
        x : array_like
            Exact observations to test, in the currently active space.
            Interval rows are not accepted: the probability integral
            transform of a censored observation is not a point.
        statistic : str, optional
            ``"ks"``, ``"cvm"`` (default) or ``"ad"``.  Anderson-Darling
            weights the tails most heavily and is the sharpest check on a
            log-concave tail, but has no asymptotic null available here,
            so it reports ``pvalue`` as ``None`` unless calibrated by
            simulation.
        calibration : str, optional
            ``"asymptotic"`` (default) or ``"montecarlo"``.
        n_resamples : int, optional
            Simulated datasets used when *calibration* is
            ``"montecarlo"``; ignored otherwise.
        rng : optional
            Random-number source used for the simulated null.
        fit_kwargs : dict or None, optional
            Overrides for the refit performed on each simulated dataset.
            Defaults to the fitted component count and support, which
            holds model selection fixed; pass ``{"n_components": "auto"}``
            to propagate selection uncertainty into the null at
            proportionally greater cost.

        Returns
        -------
        dict
            Keys ``statistic``, ``value``, ``pvalue``, ``calibration``,
            ``n``, ``n_resamples``, ``n_failed``, and
            ``pvalue_valid_for``.  ``pvalue`` is ``None`` when no
            calibrated reference is available, and ``n_resamples`` and
            ``n_failed`` are ``0`` under asymptotic calibration.

        Raises
        ------
        ValueError
            If *statistic* or *calibration* is unrecognised, or if *x*
            contains interval rows.
        RuntimeError
            If the model is not fitted, or if too many simulated refits
            fail to produce a usable null.
        """
        self._ensure_fitted()
        key = _validate_gof_statistic(statistic)
        mode = str(calibration).strip().lower()
        if mode not in ("asymptotic", "montecarlo"):
            raise ValueError(
                "calibration must be 'asymptotic' or 'montecarlo', "
                f"got {calibration!r}"
            )
        rows = _canonical_scoring_rows(x)
        if rows.shape[1] != 1:
            raise ValueError(
                "goodness_of_fit requires exact observations; interval rows "
                "have no single probability-integral transform"
            )
        points = rows[:, 0]
        observed = _gof_statistic(self.cdf(points), key)
        n = int(points.size)

        if mode == "asymptotic":
            return {
                "statistic": key,
                "value": observed,
                "pvalue": _asymptotic_gof_pvalue(self.cdf(points), key),
                "calibration": "asymptotic",
                "n": n,
                "n_resamples": 0,
                "n_failed": 0,
                "pvalue_valid_for": "held-out observations only",
            }

        gen = _to_generator(rng)
        refit_kwargs = self._resampling_fit_kwargs(fit_kwargs)

        def simulate():
            draw = np.asarray(self.sample(size=n, rng=gen), dtype=np.float64)
            replica = Distribution().fit(draw, **refit_kwargs)
            return _gof_statistic(replica.cdf(draw), key)

        null, n_failed = _simulated_statistics(simulate, n_resamples=n_resamples)
        return {
            "statistic": key,
            "value": observed,
            "pvalue": _monte_carlo_pvalue(observed, null),
            "calibration": "montecarlo",
            "n": n,
            "n_resamples": int(null.size + n_failed),
            "n_failed": int(n_failed),
            "pvalue_valid_for": "the sample the model was fitted to",
        }

    def bootstrap_bands(self, samples, x, *, quantity: str = "pdf",
                        n_resamples: int = BOOTSTRAP_DEFAULT_RESAMPLES,
                        level: float = 0.95, rng=None,
                        fit_kwargs=None) -> Dict[str, Any]:
        """Estimate pointwise uncertainty bands by nonparametric resampling.

        Resamples observation rows with replacement, refits, and reports
        percentiles of the resulting curves.  Cost is *n_resamples* full
        refits, so this is minutes-to-hours work rather than a property
        read.

        The bands are **pointwise**: each abscissa is covered at *level*
        in isolation.  A band covering the entire curve simultaneously is
        wider, and these must not be reported as one.

        Parameters
        ----------
        samples : array_like
            The observations the model was fitted to, in the same layout
            :meth:`fit` accepts.  They are not recoverable from the fitted
            state, so they must be supplied again.
        x : array_like
            Abscissae at which to evaluate *quantity*, in the currently
            active space.
        quantity : str, optional
            ``"pdf"`` (default), ``"cdf"``, or ``"sf"``.
        n_resamples : int, optional
            Resampling replicates to attempt.
        level : float, optional
            Two-sided pointwise coverage, strictly inside ``(0, 1)``.
        rng : optional
            Random-number source for the resampling indices.
        fit_kwargs : dict or None, optional
            Overrides for each refit.  Defaults to the fitted component
            count and support, which holds model selection fixed and so
            reports uncertainty *conditional* on the selected structure;
            pass ``{"n_components": "auto"}`` to let each replicate
            reselect and widen the bands accordingly.

        Returns
        -------
        dict
            Keys ``x``, ``estimate``, ``lower``, ``upper``, ``level``,
            ``quantity``, ``n_resamples``, ``n_failed``, and
            ``coverage_kind``, which is always ``"pointwise"``.

        Raises
        ------
        ValueError
            If *quantity* is unrecognised, *level* falls outside
            ``(0, 1)``, or *x* is empty.
        RuntimeError
            If the model is not fitted, or if too many replicate refits
            fail to leave a usable resampling distribution.
        """
        self._ensure_fitted()
        name = str(quantity).strip().lower()
        if name not in ("pdf", "cdf", "sf"):
            raise ValueError(
                f"quantity must be 'pdf', 'cdf' or 'sf', got {quantity!r}"
            )
        grid = np.asarray(x, dtype=np.float64).reshape(-1)
        if grid.size == 0:
            raise ValueError("bootstrap_bands requires at least one abscissa")
        if not np.all(np.isfinite(grid)):
            raise ValueError("bootstrap abscissae must be finite")

        rows = _canonical_scoring_rows(samples)
        gen = _to_generator(rng)
        refit_kwargs = self._resampling_fit_kwargs(fit_kwargs)
        active = self._default

        def evaluate(indices):
            replica = Distribution().fit(rows[indices], **refit_kwargs)
            replica.set_default(active)
            return np.asarray(getattr(replica, name)(grid), dtype=np.float64)

        result = _bootstrap_curves(
            evaluate, rows.shape[0], grid.size,
            n_resamples=n_resamples, level=level, rng=gen,
        )
        return {
            "x": grid,
            "estimate": np.asarray(getattr(self, name)(grid), dtype=np.float64),
            "lower": result["lower"],
            "upper": result["upper"],
            "level": result["level"],
            "quantity": name,
            "n_resamples": result["n_resamples"],
            "n_failed": result["n_failed"],
            "coverage_kind": "pointwise",
        }

    def _resampling_fit_kwargs(self, overrides, /):
        """Build refit keywords for a resampling replicate.

        Parameters
        ----------
        overrides : dict or None
            Caller-supplied keywords, which take precedence over the
            inherited defaults.

        Returns
        -------
        dict
            Keyword arguments for :meth:`fit`.

        Raises
        ------
        TypeError
            If *overrides* is neither ``None`` nor a mapping.
        """
        if overrides is not None and not isinstance(overrides, Mapping):
            raise TypeError("fit_kwargs must be a mapping or None")
        base_support = tuple(map(float, self._components[0].base.support))
        kwargs: Dict[str, Any] = {
            "n_components": self._K,
            "support": base_support,
            "verbose": 0,
        }
        if overrides:
            kwargs.update(dict(overrides))
        return kwargs

    def quantile_residuals(self, x, rng=None) -> np.ndarray:
        """Return normal-score residuals, randomized for interval observations.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        rng : optional
            Random-number source.
        """
        self._ensure_fitted()
        rows = _canonical_scoring_rows(x)
        return _randomized_quantile_residuals(rows, self.logcdf, self.logsf, rng=rng)

    def tail_rate(self, side: str) -> float:
        """Return the limiting absolute base-space potential slope.

        Parameters
        ----------
        side : {'lower', 'upper'}
            Tail to inspect.
        """
        self._ensure_fitted()
        if self._default != "base":
            raise ValueError("tail_rate is defined only in base space")
        rates = [
            comp.base.tail_rate(side)
            for weight, comp in zip(self._weights, self._components, strict=True)
            if weight > 0.0
        ]
        if not rates:
            raise RuntimeError("tail_rate requires at least one positive-weight component")
        finite = [rate for rate in rates if np.isfinite(rate)]
        return float(min(finite)) if finite else np.inf

    def frozen(self) -> FrozenDistribution:
        """Return a live SciPy-compatible frozen-distribution adapter."""
        self._ensure_fitted()
        return FrozenDistribution(self)

    def truncate(self, lower=None, upper=None) -> "Distribution":
        """Condition the fitted distribution to an interval.

        The returned model represents ``X | lower < X <= upper`` in the
        currently active coordinate space.  The original potential family and
        serialized state format are preserved; only normalization, active
        support, spectral probability representations, moments, and mixture
        weights are rebuilt.

        Parameters
        ----------
        lower, upper : float or None, optional
            Truncation bounds in the active space. ``None`` keeps the current
            support endpoint on that side.

        Returns
        -------
        Distribution
            An independent fitted model for the conditional distribution.

        Raises
        ------
        ValueError
            If the requested interval is empty or disjoint from the support.
        RuntimeError
            If the retained probability mass is numerically negligible.
        """
        self._ensure_fitted()
        if lower is not None and np.isnan(float(lower)):
            raise ValueError("lower must not be NaN")
        if upper is not None and np.isnan(float(upper)):
            raise ValueError("upper must not be NaN")
        if lower is not None and upper is not None and float(lower) >= float(upper):
            raise ValueError("lower must be strictly less than upper")

        active_lo, active_hi = map(float, self.support)
        lo = active_lo if lower is None else max(active_lo, float(lower))
        hi = active_hi if upper is None else min(active_hi, float(upper))
        if not lo < hi:
            raise ValueError(
                f"truncation interval ({lower!r}, {upper!r}) does not overlap "
                f"support ({active_lo!r}, {active_hi!r})"
            )

        if self._default == "base":
            base_lo, base_hi = lo, hi
        else:
            if hi <= 0.0:
                raise ValueError(
                    f"truncation interval ({lower!r}, {upper!r}) does not overlap "
                    f"support ({active_lo!r}, {active_hi!r})"
                )
            with np.errstate(divide="ignore"):
                base_lo = -np.inf if lo <= 0.0 else float(np.log(lo))
                base_hi = np.inf if np.isposinf(hi) else float(np.log(hi))

        log_masses = np.empty(self._K, dtype=np.float64)
        for j, comp in enumerate(self._components):
            log_masses[j] = float(log_mass_between(
                comp.base.logcdf(base_lo), comp.base.logcdf(base_hi),
                comp.base.logsf(base_lo), comp.base.logsf(base_hi),
            ))

        with np.errstate(divide="ignore"):
            log_weights = np.where(self._weights > 0.0, np.log(self._weights), -np.inf)
        log_terms = log_weights + log_masses
        log_total = float(np.logaddexp.reduce(log_terms))
        log_tiny = float(np.log(np.finfo(np.float64).tiny))
        if not np.isfinite(log_total) or log_total < log_tiny:
            raise RuntimeError(
                "truncation retained negligible probability mass "
                f"(log mass={log_total!r})"
            )

        kept_components = []
        kept_log_terms = []
        for comp, log_mass, log_term in zip(self._components, log_masses, log_terms, strict=True):
            if not np.isfinite(log_term) or log_mass < log_tiny:
                continue
            kept_components.append(comp._truncate_base(base_lo, base_hi))
            kept_log_terms.append(float(log_term))
        if not kept_components:
            raise RuntimeError(
                "truncation retained probability mass but no component could be "
                "represented numerically"
            )

        kept_log_array = np.asarray(kept_log_terms, dtype=np.float64)
        normalizer = float(np.logaddexp.reduce(kept_log_array))
        new_weights = np.exp(kept_log_array - normalizer)
        new_weights /= np.sum(new_weights, dtype=np.float64)

        if len(kept_components) == 1:
            result = Distribution(kept_components[0].data)
        else:
            state = _pack_mixture_struct(
                new_weights, self._default, [comp.data for comp in kept_components]
            )
            result = Distribution(state)
        result.set_default(self._default)
        return result

    def _log_concavity_margin(self) -> float:
        """Return a certified lower bound on component base-space convexity.

        Reported through :attr:`spectral_diagnostics`; for mixtures the
        minimum component certificate is returned, and the mixture density
        itself need not be log-concave.

        Returns
        -------
        float
            Smallest certified convexity margin across components.
        """
        self._ensure_fitted()
        return float(min(comp.log_concavity_margin() for comp in self._components))

    def neg_log(self, x, n: int = 0) -> Union[float, np.ndarray]:
        """Evaluate the potential (or derivatives) in the currently active space.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.
        n : int, optional
            Derivative order; zero requests the potential itself.

        Returns
        -------
        float or numpy.ndarray
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].neg_log(x, n)
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if self._default == "exp":
            return self._mix_exp_potential(x, n)
        return self._mix_base_potential(x, n)

    def sample(self, size=None, rng=None):
        """Draw random samples in the currently active space.

        Parameters
        ----------
        size : int or None, optional
            Number of draws; ``None`` returns a single float.  Must be a
            non-negative integer.  Integral floats such as ``4.0`` are
            accepted; non-integral floats, strings and ``bool`` are
            rejected rather than silently truncated.
        rng : optional
            Random-number source accepted by NumPy generator normalization.

        Returns
        -------
        float or numpy.ndarray
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].sample(size=size, rng=rng)
        gen = _to_generator(rng)
        K = self._K
        if size is None:
            j = int(gen.choice(K, p=self._weights))
            u = float(np.clip(gen.random(), PROB_EPS, 1.0 - PROB_EPS))
            return self._view(self._components[j]).ppf(u)
        nn = _coerce_sample_size(size)
        if nn == 0:
            return np.empty(0, dtype=np.float64)
        assignments = gen.choice(K, size=nn, p=self._weights)
        u = np.clip(gen.random(nn), PROB_EPS, 1.0 - PROB_EPS)
        out = np.empty(nn, dtype=np.float64)
        for j in range(K):
            mask = assignments == j
            if mask.any():
                out[mask] = self._view(self._components[j]).ppf(u[mask])
        return out

    def moment(self, k: int, central: bool = False, standardized: bool = False) -> float:
        """Compute the *k*-th moment in the currently active space.

        Parameters
        ----------
        k : int
            Non-negative integer moment order.
        central : bool, optional
            Whether to compute a moment about the distribution mean.
        standardized : bool, optional
            Whether to divide a central moment by the corresponding power of the standard deviation.

        Returns
        -------
        float
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].moment(k, central=central, standardized=standardized)
        if isinstance(k, bool) or int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        k = int(k)
        if not central and not standardized:
            return float(self._raw_moment(k))
        stats = self._ensure_stats()
        if self._default == "exp":
            low_order = _exp_moment_from_stats(k, stats, standardized)
            if low_order is not None:
                return low_order

        cm = float(self._mixture_centered_moment(k))
        if not standardized:
            return cm
        std = float(stats["std"])
        if not (std > 0.0):
            raise RuntimeError("Standardized moment is undefined because std <= 0.")
        return float(cm / (std ** k))

    def cumulant(self, k: int) -> float:
        """Return the *k*-th cumulant in the currently active space.

        Parameters
        ----------
        k : int
            Positive integer cumulant order.

        Returns
        -------
        float
            Requested cumulant in the active coordinate system.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If *k* is not a positive integer.
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].cumulant(k)
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        order = int(k)
        key = (str(self._default), order)
        if key not in self._cumulant_cache:
            self._cumulant_cache[key] = _cumulant_from_centered(
                self._mixture_centered_moment, order, self.mean
            )
        return float(self._cumulant_cache[key])

    # ------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------

    @property
    def support(self) -> np.ndarray:
        """Support of the density in the currently active space.

        Returns
        -------
        numpy.ndarray
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].support
        view = self._view(self._components[0])
        return np.asarray(view.support, dtype=np.float64)

    @property
    def mode(self) -> float:
        """Global maximum of the PDF in the currently active space.

        For single-component densities this is the unique mode.  For
        multi-component densities this is the location of the tallest
        peak among all local maxima.

        Returns
        -------
        float
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].mode
        all_modes = self.modes
        if len(all_modes) == 1:
            return all_modes[0]
        best = min(all_modes, key=lambda m: float(self.neg_log(m, 0)))
        return float(best)

    @property
    def modes(self) -> Tuple[float, ...]:
        """All local maxima (modes) of the PDF in the currently active space.

        For single-component densities the tuple has length 1.  For
        multi-component densities there may be one or more numerically resolved
        local modes, returned sorted in ascending order.

        Returns
        -------
        tuple of float
        """
        self._ensure_fitted()
        if self._K == 1:
            return (self._components[0].mode,)
        modes = self._ensure_modes()
        if self._default == "exp":
            if "exp" not in modes:
                modes["exp"] = _find_mixture_modes_exp(
                    self._mix_base_potential,
                    [c.exp._log_mode_coordinate() for c in self._components],
                )
            return modes["exp"]
        return modes["base"]

    @property
    def median(self) -> float:
        """Median of the density in the currently active space."""
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].median
        return float(self.ppf(0.5))

    @property
    def mean(self) -> float:
        """Mean of the density in the currently active space."""
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].mean
        return float(self._ensure_stats()["mean"])

    @property
    def var(self) -> float:
        """Variance of the density in the currently active space."""
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].var
        return float(self._ensure_stats()["var"])

    @property
    def std(self) -> float:
        """Standard deviation of the density in the currently active space."""
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].std
        return float(self._ensure_stats()["std"])

    @property
    def skew(self) -> float:
        """Skewness of the density in the currently active space."""
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].skew
        return float(self._ensure_stats()["skew"])

    @property
    def kurt(self) -> float:
        """Kurtosis of the density in the currently active space.

        Pearson (raw) kurtosis, so a Gaussian reads ``3.0``; subtract 3
        for the Fisher excess.
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].kurt
        return float(self._ensure_stats()["kurt"])

    # ------------------------------------------------------------------
    # Transform / copy / serialisation
    # ------------------------------------------------------------------

    def transform(self, *, mu: float | None = None, sigma: float | None = None,
                  pullback: bool,
                  inplace: bool = True) -> "Distribution":
        """Apply an affine location-scale transformation to every component.

        An exact change of variables, not a refit: the PDF, CDF, PPF and
        every summary statistic are updated analytically.

        Parameters
        ----------
        mu : float or None, optional
            Location parameter.  ``None`` leaves it unchanged.
        sigma : float or None, optional
            Scale parameter; must be positive.  ``None`` leaves it
            unchanged.
        pullback : bool
            ``False`` gives the pushforward ``Y = mu + sigma * X``.
            ``True`` gives the pullback ``Y = (X - mu) / sigma``, the
            inverse map.  This argument is required so transform direction
            is never inferred from mutable model state.
        inplace : bool, optional
            Mutate this object (default) or return a transformed copy.

        Returns
        -------
        Distribution
            ``self`` when *inplace*, otherwise the transformed copy.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        TypeError
            If *pullback* is not a boolean.
        ValueError
            If *mu* is non-finite, or *sigma* is non-finite or not positive.
        """
        self._ensure_fitted()
        target = self if inplace else self.copy()
        for comp in target._components:
            comp.transform(mu=mu, sigma=sigma, pullback=pullback, inplace=True)
        target._stats_cache = None
        target._cumulant_cache.clear()
        target._mode_cache = None
        # The diagnostics cache key cannot observe a location-scale change, so
        # clear it here alongside the other state-dependent caches.
        target._fit_diagnostics_cache = None
        if target._K > 1:
            target._rebuild_spectral_cache()
            target._spectral_cache_valid = True
        return target


    @property
    def data(self):
        """A deep copy of the structured fitted state.

        Returns
        -------
        numpy.void
        """
        self._ensure_fitted()
        if self._K == 1:
            return self._components[0].data
        comp_states = [c.data for c in self._components]
        bm = None
        if self._mode_cache is not None and "base" in self._mode_cache:
            bm = self._mode_cache["base"]
        return _pack_mixture_struct(self._weights, self._default, comp_states,
                                    base_modes=bm)

    def load(self, state) -> "Distribution":
        """Load a previously saved fitted state.

        Automatically detects single-component vs multi-component format. The
        current instance is updated only after the complete state has validated
        and all required spectral caches have been reconstructed.

        Parameters
        ----------
        state : numpy.void
            Structured fitted state previously obtained from :attr:`data`
            or loaded from a NumPy file.

        Returns
        -------
        Distribution
            ``self``.
        """
        state = np.array(state, copy=True)
        names = state.dtype.names or ()
        target = Distribution()
        if "n_components" in names:
            weights, default_space, comp_states = _unpack_mixture_struct(state)
            packed = np.asarray(weights, dtype=np.float64).ravel()
            if (
                packed.size != len(comp_states)
                or not np.all(np.isfinite(packed))
                or np.any(packed < 0.0)
                or abs(float(packed.sum()) - 1.0) > 1e-8
            ):
                raise ValueError(
                    "mixture state has invalid component weights: expected one "
                    "finite non-negative weight per component, summing to one"
                )
            if str(default_space) not in ("base", "exp"):
                raise ValueError(
                    "mixture state default space must be 'base' or 'exp'"
                )
            components = [_Component(cs) for cs in comp_states]
            components, weights = _sort_components_by_mode(components, weights)
            target._components = components
            target._weights = weights
            target._K = len(components)
            target._default = default_space
            target._stats_cache = None
            target._mode_cache = None
            if "base_modes" in names:
                n_modes = int(state["n_modes"]) if "n_modes" in names else 0
                raw = np.asarray(state["base_modes"], dtype=np.float64).ravel()
                if n_modes < 0 or n_modes > raw.size:
                    raise ValueError("mixture state has inconsistent n_modes/base_modes")
                base_modes = tuple(
                    float(value) for value in raw[:n_modes] if np.isfinite(value)
                )
                if base_modes:
                    target._mode_cache = {"base": base_modes}
            if target._K > 1:
                target._rebuild_spectral_cache()
                target._spectral_cache_valid = True
        else:
            comp = _Component(state)
            target._components = [comp]
            target._weights = np.array([1.0], dtype=np.float64)
            target._K = 1
            target._default = comp.default
            target._stats_cache = None
            target._mode_cache = None
            target._spectral_cache_valid = False

        self.__dict__.update(target.__dict__)
        return self

    def copy(self) -> "Distribution":
        """Create an independent deep copy, including diagnostic records."""
        if not self.is_fitted:
            return Distribution()
        clone = Distribution(self.data)
        clone._em_diagnostics = _copy.deepcopy(self._em_diagnostics)
        clone._selection_diagnostics = _copy.deepcopy(self._selection_diagnostics)
        return clone

    # ------------------------------------------------------------------
    # Internal helpers
    #
    # Mixture moments, potentials and the spectral cache live in
    # ``_mixture_stats``; diagnostic records live in ``_diagnostics``.
    # ------------------------------------------------------------------

    def _view(self, comp):
        """Return the active-space view for a component.

        Parameters
        ----------
        comp : _Component
            Component whose view should be selected in the active coordinate space.

        Returns
        -------
        _BaseView or _ExpView
        """
        return comp.base if self._default == "base" else comp.exp


    def _ensure_fitted(self):
        """Raise ``RuntimeError`` if the instance has no fitted data."""
        if not self.is_fitted:
            raise RuntimeError("Distribution is not fitted; call .fit(...) or .load(...).")
