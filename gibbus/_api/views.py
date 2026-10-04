"""Coordinate-space views over a fitted component.

A fitted component is evaluated through one of two views:

* :class:`_BaseSpaceView` — the modeled variable *x* directly.
* :class:`_ExpSpaceView` — the induced density on *y = exp(x)*.

Both are thin: they hold no fitted state of their own beyond memoized
exp-space quadrature results, and defer to the parent component for the
potential, the affine map, and the spectral CDF/PPF.

The views are not instantiated directly; reach them through
``Distribution.base`` / ``Distribution.exp`` or a component's ``.base`` / ``.exp``.
"""

from typing import TYPE_CHECKING

import numpy as np

from .._defaults import PROB_EPS, _reraise_if_debug
from .._fit.inputs import _coerce_sample_size, _to_generator
from .._model.numerics import _terms_for_quad
from .._postfit import analytics as _pf
from .._postfit.analytics import (
    _exp_moment_from_stats,
    _exp_stats_from_log_moments,
    _tail_rate_from_geometry,
)
from .._postfit.evaluators import _potential_oriented_affine_eval
from .._postfit.expectation import expect as _expect
from .._postfit.expectation import expect_vectorized as _expect_vectorized
from .._postfit.information import cross_entropy as _cross_entropy
from .._postfit.information import entropy as _entropy
from .._postfit.information import kl_divergence as _kl_divergence
from .._postfit.regions import hpd as _hpd
from .._postfit.regions import interval as _interval
from .._postfit.scoring import canonical_scoring_rows as _canonical_scoring_rows
from .._postfit.scoring import interval_loglik as _interval_loglik
from .._postfit.scoring import (
    randomized_quantile_residuals as _randomized_quantile_residuals,
)
from .._postfit.scoring import scoring_weights as _scoring_weights
from .._postfit.scoring import warn_out_of_support as _warn_out_of_support
from .._postfit.survival import cdf_hybrid, log_cdf_hybrid, log_sf_hybrid
from .._postfit.survival import isf as _isf
from .._postfit.survival import logisf as _logisf
from .._postfit.survival import logppf as _logppf
from .._postfit.survival import mean_residual_life as _mean_residual_life
from .._postfit.survival import residual_entropy as _residual_entropy
from .._spectral.tail import refine_tail_quantiles

if TYPE_CHECKING:  # avoids the views <-> component import cycle at runtime
    from .component import _Component


class _BaseSpaceView:
    """View that evaluates the fitted density in **base** (native) coordinates.

    All evaluation is expressed in terms of the internal fitted potential
    *q(z)*, mapped to the user coordinate *x* via an affine transform
    determined by ``(mu, sigma, center, scale)``.

    This class is not instantiated directly; access it via
    :attr:`Distribution.base` (single-component) or a component's ``.base``
    attribute.

    Notes
    -----
    *  ``mu_eff`` and ``sigma_eff`` denote the *effective* affine parameters
       that map user *x* to internal *z*:  ``z = sigma_eff * x + mu_eff```.
    *  Scalar inputs produce scalar (``float``) outputs; array inputs produce
       ``ndarray`` outputs.
    """

    def __init__(self, parent: "_Component"):
        """Bind this base-space view to its parent :class:`_Component`.

        Parameters
        ----------
        parent : _Component
            Owning fitted object that supplies the structured state and
            cached evaluation closures.
        """
        self._p = parent
        self._cumulant_cache_version = -1
        self._cumulant_cache: dict[int, float] = {}

    @property
    def support(self):
        """Finite or semi-infinite support of the density in base coordinates.

        Returns
        -------
        numpy.ndarray
            Shape ``(2,)`` float64 array ``[lower, upper]``.
        """
        self._p._ensure_fitted()
        return self._p._support_base()

    @property
    def mode(self):
        """Location of the density maximum in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("mode")

    @property
    def median(self):
        """Median of the density in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("median")

    @property
    def mean(self):
        """Mean of the density in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("mean")

    @property
    def var(self):
        """Variance of the density in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("var")

    @property
    def std(self):
        """Standard deviation of the density in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("std")

    @property
    def skew(self):
        """Skewness of the density in base coordinates.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("skew")

    @property
    def kurt(self):
        """Kurtosis of the density in base coordinates.

        This is the Pearson (raw) kurtosis -- the standardized fourth
        moment, the counterpart of :attr:`skew` -- so a Gaussian reads
        ``3.0``.  Subtract 3 for the Fisher excess.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        return self._p._base_stat("kurt")

    def pdf(self, x):
        """Evaluate the probability density function in base coordinates.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        z = sigma_eff * x + mu_eff
        out = abs(sigma_eff) * self._p._base_pdf(z)
        lo, hi = map(float, self.support)
        mask = np.isfinite(x)
        if np.isfinite(lo):
            mask &= x >= lo
        if np.isfinite(hi):
            mask &= x <= hi
        out = np.where(mask, out, np.where(np.isnan(x), np.nan, 0.0))
        return float(out) if scalar else out

    def _spectral_cdf(self, x):
        """Evaluate the body CDF from the packed spectral representation.

        Parameters
        ----------
        x : float or array_like
            Base-space evaluation coordinates.

        Returns
        -------
        float or numpy.ndarray
            Spectral CDF values on an absolute-probability scale.
        """
        self._p._ensure_fitted()
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        z = sigma_eff * x + mu_eff
        base = np.asarray(self._p._base_cdf(z), dtype=np.float64)
        out = base if sigma_eff > 0.0 else (1.0 - base)
        out = np.clip(out, 0.0, 1.0)
        return float(out) if scalar else out

    def cdf(self, x):
        """Evaluate the cumulative distribution function in base coordinates.

        The spectral CDF is used in the body and exact tail quadrature is
        selected automatically through :meth:`logcdf` when the lower-tail
        mass is too small for absolute-probability spectral evaluation.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.

        Returns
        -------
        float or numpy.ndarray
        """
        lo, hi = map(float, self.support)
        return cdf_hybrid(
            self.neg_log,
            self._spectral_cdf,
            x,
            lo,
            upper_endpoint=hi,
            log_tail_mass=self._exact_tail_log_mass,
            log_tail_masses=self._exact_tail_log_masses,
        )

    def ppf(self, p):
        """Evaluate the percent-point (quantile) function in base coordinates.

        Quantiles below ``TAIL_ASYMPTOTIC_P`` (and symmetrically above
        ``1 - TAIL_ASYMPTOTIC_P``) are re-solved from the potential rather
        than read off the spectral CDF, whose panels hold no significant
        digits that far out.  Use :meth:`_raw_ppf` only when an internal
        caller explicitly needs the unrefined spectral inverse.

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
        self._p._ensure_fitted()
        p = np.asarray(p, dtype=np.float64)
        scalar = p.ndim == 0
        if np.any(~np.isfinite(p) | (p < 0.0) | (p > 1.0)):
            raise ValueError("ppf is defined for finite p in [0, 1]")

        out = np.atleast_1d(self._raw_ppf(p)).astype(np.float64, copy=True)

        # Refine at the component/view boundary so every public quantile path
        # has the same extreme-tail semantics.
        supp = self.support
        out = refine_tail_quantiles(
            self.neg_log,
            self._raw_ppf,
            float(supp[0]),
            float(supp[1]),
            np.atleast_1d(p).reshape(-1),
            out.reshape(-1),
            log_tail_mass=self._exact_tail_log_mass,
        )

        out = out.reshape(np.asarray(p).shape)
        return float(out) if scalar else out

    def _raw_ppf(self, p):
        """The unrefined spectral inverse in base coordinates.

        Kept separate from :meth:`ppf` so the tail refinement can seed
        itself without recursing through its own output.

        Parameters
        ----------
        p : float or array_like
            Probabilities in ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
            Unrefined quantiles in base coordinates.
        """
        p = np.asarray(p, dtype=np.float64)
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        internal_p = p if sigma_eff > 0.0 else (1.0 - p)
        z = self._p._base_ppf(internal_p)
        out = (z - mu_eff) / sigma_eff
        support = np.asarray(self.support, dtype=np.float64)
        return np.clip(out, support[0], support[1])

    def neg_log(self, x, n: int = 0):
        """Evaluate the negative-log density (potential) or its derivatives.

        Parameters
        ----------
        x : float or array_like
            Evaluation point or points in the relevant coordinate space.
        n : int, optional
            Derivative order (default ``0``).

        Returns
        -------
        float or numpy.ndarray

        Raises
        ------
        ValueError
            If *n* is negative.
        """
        data = self._p._ensure_fitted()
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        potential_support = self._p._potential_support_base()
        out = _potential_oriented_affine_eval(
            x,
            potential_support,
            mu_eff,
            sigma_eff,
            data["q_poly"],
            data["boundary_amplitudes"],
            n,
        )
        active_lo, active_hi = map(float, self.support)
        active = np.isfinite(x)
        if np.isfinite(active_lo):
            active &= x >= active_lo
        if np.isfinite(active_hi):
            active &= x <= active_hi
        if n == 0:
            out = np.where(active, out, np.where(np.isnan(x), np.nan, np.inf))
        else:
            out = np.where(active, out, np.nan)
        return float(out) if scalar else out

    def _exact_tail_log_mass(self, x, endpoint, /, *, upper=False):
        """Evaluate one exact tail mass with the reusable compiled callback.

        Parameters
        ----------
        x : float
            Tail anchor in this view's coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        self._p._ensure_fitted()
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        value, message = self._p._tail_integrator.log_mass(
            float(x),
            float(endpoint),
            bool(upper),
            self._p._potential_support_base(),
            self._p._data["boundary_amplitudes"],
            float(mu_eff),
            float(sigma_eff),
        )
        if message is not None:
            side = "upper" if upper else "lower"
            _reraise_if_debug(RuntimeError(message), f"{side}-tail compiled quadrature")
        return float(value)

    def _exact_tail_log_masses(self, x, endpoint, /, *, upper=False):
        """Evaluate many exact tail masses in one compiled call.

        Parameters
        ----------
        x : numpy.ndarray
            Tail anchors in this view's coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        self._p._ensure_fitted()
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        values, failed = self._p._tail_integrator.log_masses(
            np.asarray(x, dtype=np.float64),
            float(endpoint),
            bool(upper),
            self._p._potential_support_base(),
            self._p._data["boundary_amplitudes"],
            float(mu_eff),
            float(sigma_eff),
        )
        if failed:
            side = "upper" if upper else "lower"
            _reraise_if_debug(
                RuntimeError(f"{failed} tail integrals reached the subdivision limit"),
                f"{side}-tail compiled quadrature",
            )
        return values

    def logpdf(self, x):
        """Evaluate the log density in base coordinates.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        out = -np.asarray(self.neg_log(arr, 0), dtype=np.float64)
        out = np.where(np.isnan(arr), np.nan, out)
        return float(out) if scalar else out

    def logcdf(self, x):
        """Evaluate the tail-accurate log CDF in base coordinates.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        lo, hi = map(float, self.support)
        return log_cdf_hybrid(
            self.neg_log,
            self._spectral_cdf,
            x,
            lo,
            upper_endpoint=hi,
            log_tail_mass=self._exact_tail_log_mass,
            log_tail_masses=self._exact_tail_log_masses,
        )

    def logsf(self, x):
        """Evaluate the tail-accurate log survival function.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        lo, hi = map(float, self.support)
        return log_sf_hybrid(
            self.neg_log,
            self._spectral_cdf,
            x,
            hi,
            lower_endpoint=lo,
            log_tail_mass=self._exact_tail_log_mass,
            log_tail_masses=self._exact_tail_log_masses,
        )

    def sf(self, x):
        """Evaluate the survival function from its log-space representation.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        scalar = np.asarray(x).ndim == 0
        with np.errstate(under="ignore"):
            out = np.exp(self.logsf(x))
        return float(out) if scalar else out

    def isf(self, p):
        """Evaluate the inverse survival function.

        Parameters
        ----------
        p : float or array_like
            Survival probabilities in ``[0, 1]``.
        """
        return _isf(
            self.neg_log,
            self.ppf,
            self.support,
            p,
            log_tail_mass=self._exact_tail_log_mass,
        )

    def logppf(self, log_p):
        """Evaluate a quantile from a logarithmic CDF probability.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        return _logppf(
            self.neg_log,
            self.ppf,
            self.support,
            log_p,
            log_tail_mass=self._exact_tail_log_mass,
        )

    def logisf(self, log_p):
        """Evaluate an upper quantile from a logarithmic survival probability.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        return _logisf(
            self.neg_log,
            self.ppf,
            self.support,
            log_p,
            log_tail_mass=self._exact_tail_log_mass,
        )

    def log_hazard(self, x):
        """Evaluate log hazard ``log f(x) - log S(x)``.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
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

    def hazard(self, x, n: int = 0):
        """Evaluate the hazard or its first derivative.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        n : {0, 1}, optional
            Derivative order.
        """
        n = int(n)
        if n not in (0, 1):
            raise ValueError("hazard supports only n=0 or n=1")
        scalar = np.asarray(x).ndim == 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            h = np.asarray(np.exp(self.log_hazard(x)), dtype=np.float64)
        out = (
            h if n == 0 else h * (h - np.asarray(self.neg_log(x, 1), dtype=np.float64))
        )
        return float(out) if scalar else out

    def cumulative_hazard(self, x):
        """Evaluate cumulative hazard ``-log S(x)``.

        Parameters
        ----------
        x : float or array_like
            Evaluation coordinate(s).
        """
        scalar = np.asarray(x).ndim == 0
        out = -np.asarray(self.logsf(x), dtype=np.float64)
        return float(out) if scalar else out

    def mean_residual_life(self, x):
        """Return ``E[X-x | X>x]`` in base coordinates.

        Parameters
        ----------
        x : float or array_like
            Conditioning threshold(s).
        """
        return _mean_residual_life(
            self.neg_log, self.logsf, self.support, self.mean, self.mode, x
        )

    def residual_entropy(self, x):
        """Return differential entropy of ``X | X>x``.

        Parameters
        ----------
        x : float or array_like
            Conditioning threshold(s).
        """
        return _residual_entropy(self.neg_log, self.logsf, self.support, x)

    def interval(self, level):
        """Return an equal-tailed probability interval.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        return _interval(self.ppf, self.isf, self.support, level)

    def hpd(self, level):
        """Return the highest-density region as an ``(m, 2)`` array.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        return _hpd(
            self.logpdf,
            self.logcdf,
            self.logsf,
            self.ppf,
            self.isf,
            self.support,
            level,
            modes=(self.mode,),
        )

    def expect(self, func):
        """Compute ``E[func(X)]`` in base coordinates.

        Parameters
        ----------
        func : callable
            Scalar function of the random variable.
        """
        return _expect(self.neg_log, self.support, func, points=(self.mode,))

    def entropy(self):
        """Return differential entropy in base coordinates."""
        return _entropy(self.neg_log, self.support, points=(self.mode,))

    def cross_entropy(self, other):
        """Return cross-entropy against another base-space view.

        Parameters
        ----------
        other : _BaseSpaceView
            Other fitted base-space distribution.
        """
        if not isinstance(other, _BaseSpaceView):
            raise ValueError("cross_entropy requires another base-space view")
        return _cross_entropy(
            self.neg_log,
            self.support,
            other.neg_log,
            other.support,
            points=(self.mode,),
        )

    def kl_divergence(self, other):
        """Return ``D_KL(self || other)`` for another base-space view.

        Parameters
        ----------
        other : _BaseSpaceView
            Other fitted base-space distribution.
        """
        if not isinstance(other, _BaseSpaceView):
            raise ValueError("kl_divergence requires another base-space view")
        return _kl_divergence(
            self.neg_log,
            self.support,
            other.neg_log,
            other.support,
            points=(self.mode,),
        )

    def loglik(self, x, sample_weight=None):
        """Return weighted held-out log likelihood.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        sample_weight : array_like or None, optional
            Non-negative row weights; values are not normalized.
        """
        rows = _canonical_scoring_rows(x)
        _warn_out_of_support(rows, self.support)
        weights = _scoring_weights(rows.shape[0], sample_weight)
        if rows.shape[1] == 1:
            positive = weights > 0.0
            terms = np.asarray(self.logpdf(rows[:, 0]), dtype=np.float64)
            return float(np.sum(weights[positive] * terms[positive], dtype=np.float64))
        return _interval_loglik(rows, weights, self.logpdf, self.logcdf, self.logsf)

    def quantile_residuals(self, x, rng=None):
        """Return normal-score residuals, randomized for interval rows.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        rng : optional
            Random-number source.
        """
        rows = _canonical_scoring_rows(x)
        return _randomized_quantile_residuals(rows, self.logcdf, self.logsf, rng=rng)

    def tail_rate(self, side):
        """Return the exact limiting absolute potential slope in one tail.

        Parameters
        ----------
        side : {'lower', 'upper'}
            Tail to inspect.
        """
        support, _amps = self._p._internal_model_geometry()
        q_poly = np.asarray(self._p.data["q_poly"], dtype=np.float64)
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        return _tail_rate_from_geometry(support, q_poly, mu_eff, sigma_eff, side)

    def sample(self, size=None, rng=None):
        """Draw random samples in base coordinates.

        Parameters
        ----------
        size : int or None, optional
            Number of draws; ``None`` returns a single float.  Must be a
            non-negative integer.  Integral floats such as ``4.0`` are
            accepted; non-integral floats, strings and ``bool`` are
            rejected rather than silently truncated.
        rng : {None, int, numpy.random.Generator, numpy.random.RandomState}, optional
            Random-number source accepted by NumPy generator normalization.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        gen = _to_generator(rng)
        if size is None:
            u = float(np.clip(gen.random(), PROB_EPS, 1.0 - PROB_EPS))
            return self.ppf(u)
        n = _coerce_sample_size(size)
        u = np.clip(gen.random(n), PROB_EPS, 1.0 - PROB_EPS)
        return self.ppf(u)

    def moment(
        self, k: int, central: bool = False, standardized: bool = False
    ) -> float:
        """Compute the *k*-th moment in base coordinates.

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
        self._p._ensure_fitted()
        if isinstance(k, bool) or int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        k = int(k)
        if not central and not standardized:
            return float(self._p._raw_moment_base(k))

        # Centralize in the fitted canonical coordinate, where the mean is O(1),
        # then apply the affine scale analytically.  This avoids subtracting
        # translated raw moments such as 1e24 - 1e24 for a distribution centered
        # near 1e12.
        z_mean = float(self._p._canonical_raw_moment(1))
        z_cm = _pf._central_moment_from_raw(self._p._canonical_raw_moment, k, z_mean)
        _mu_eff, sigma_eff = self._p._mu_sigma_eff()
        alpha = 1.0 / float(sigma_eff)
        cm = float((alpha**k) * z_cm)
        if not standardized:
            return cm
        std = float(self.std)
        if not (std > 0.0):
            raise RuntimeError("Standardized moment is undefined because std <= 0.")
        return float(cm / (std**k))

    def cumulant(self, k: int) -> float:
        """Return the *k*-th cumulant in base coordinates.

        Parameters
        ----------
        k : int
            Positive integer cumulant order.

        Returns
        -------
        float
            Requested cumulant in base coordinates.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If *k* is not a positive integer.
        """
        self._p._ensure_fitted()
        if self._cumulant_cache_version != self._p._version:
            self._cumulant_cache_version = self._p._version
            self._cumulant_cache.clear()
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        order = int(k)
        if order not in self._cumulant_cache:
            self._cumulant_cache[order] = _pf._cumulant_from_centered(
                lambda n: self.moment(n, central=True), order, self.mean
            )
        return float(self._cumulant_cache[order])


class _ExpSpaceView:
    """View that evaluates the fitted density in **exp** (exponentiated) coordinates.

    If the base density is defined on *x*, the ``exp`` view provides the
    induced density on *y = exp(x)* via the standard change-of-variables
    formula  ``pdf_y(y) = pdf_x(log y) / y``.

    This class is not instantiated directly; access it via
    :attr:`Distribution.exp` (single-component) or a component's ``.exp``
    attribute.

    Notes
    -----
    *  Summary statistics in exp space are computed from raw moments
       ``E[y^k]`` obtained by numerical quadrature and cached for reuse.
    *  The cache is automatically invalidated whenever the parent
       :class:`_Component` version counter increments.
    """

    def __init__(self, parent: "_Component"):
        """Bind this exp-space view to its parent :class:`_Component`.

        Parameters
        ----------
        parent : _Component
            Fitted parent object whose distribution state this view exposes.
        """
        self._p = parent
        self._cache_version = -1
        self._raw_cache: dict[int, float] = {}
        self._log_raw_cache: dict[int, float] = {}
        self._cumulant_cache: dict[int, float] = {}
        self._stats_cache = None
        self._q_cache = None

    def _invalidate_if_needed(self):
        """Reset caches if the parent version has changed."""
        if self._cache_version != self._p._version:
            self._cache_version = self._p._version
            self._raw_cache.clear()
            self._log_raw_cache.clear()
            self._cumulant_cache.clear()
            self._stats_cache = None
            self._q_cache = None

    def _ensure_q_cache(self):
        """Return the quadrature-parameter tuple required by moment integrals."""
        data = self._p._ensure_fitted()
        self._invalidate_if_needed()
        if self._q_cache is not None:
            return self._q_cache
        base_support, boundary_amplitudes = self._p._internal_model_geometry()
        q_poly = np.asarray(data["q_poly"], dtype=np.float64)
        window = np.asarray(data["window"], dtype=np.float64)
        mu_eff, sigma_eff = self._p._mu_sigma_eff()
        terms = _terms_for_quad(base_support, boundary_amplitudes)
        self._q_cache = (
            base_support,
            q_poly,
            boundary_amplitudes,
            window,
            mu_eff,
            sigma_eff,
            terms,
        )
        return self._q_cache

    def _log_raw_moment(self, k: int) -> float:
        """Return ``log E[Y^k]`` without materializing the dimensional moment.

        Parameters
        ----------
        k : int
            Non-negative raw-moment order.
        """
        q_cache = self._ensure_q_cache()
        k = int(k)
        if k == 0:
            return 0.0
        hit = self._log_raw_cache.get(k)
        if hit is not None:
            return hit
        (
            base_support,
            q_poly,
            boundary_amplitudes,
            window,
            mu_eff,
            sigma_eff,
            terms,
        ) = q_cache
        value = float(
            _pf._log_raw_moment_exp(
                base_support,
                q_poly,
                boundary_amplitudes,
                window,
                mu_eff,
                sigma_eff,
                k,
                terms,
            )
        )
        self._log_raw_cache[k] = value
        return value

    def _raw_moment(self, k: int) -> float:
        """Compute the *k*-th raw moment ``E[y^k]`` via numerical quadrature.

        Parameters
        ----------
        k : int
            Moment order.

        Returns
        -------
        float
        """
        self._p._ensure_fitted()
        self._ensure_q_cache()
        k = int(k)
        if k == 0:
            return 1.0
        hit = self._raw_cache.get(k)
        if hit is not None:
            return hit
        log_value = self._log_raw_moment(k)
        with np.errstate(over="ignore", under="ignore"):
            val = float(np.exp(log_value))
        self._raw_cache[k] = val
        return val

    def _log_mode_coordinate(self):
        """Return the exp-space component mode in base/log coordinates."""
        (
            base_support,
            q_poly,
            boundary_amplitudes,
            window,
            mu_eff,
            sigma_eff,
            _terms,
        ) = self._ensure_q_cache()
        shift = -1.0 / float(sigma_eff)
        z_star = _pf._valley(
            window, base_support, q_poly, boundary_amplitudes, float(shift)
        )
        return float((z_star - mu_eff) / sigma_eff)

    def _ensure_stats(self):
        """Return exp-space statistics, computing them on first use."""
        self._ensure_q_cache()
        if self._stats_cache is not None:
            return self._stats_cache
        log_mode = self._log_mode_coordinate()
        with np.errstate(over="ignore", under="ignore"):
            mode = float(np.exp(log_mode))
        stats = _exp_stats_from_log_moments(
            [self._log_raw_moment(k) for k in (1, 2, 3, 4)],
            self._relative_centered_moment,
            "exp-space",
        )
        stats["mode"] = mode
        self._stats_cache = stats
        return stats

    def _relative_centered_moment(self, log_mean, k, /):
        """Return the ``k``-th centered exp-space moment in units of ``mean**k``.

        Parameters
        ----------
        log_mean : float
            Logarithm of the exp-space mean.
        k : int
            Centered-moment order.
        """
        (
            base_support,
            q_poly,
            boundary_amplitudes,
            window,
            mu_eff,
            sigma_eff,
            _terms,
        ) = self._ensure_q_cache()
        return float(
            _pf._relative_centered_moment_exp(
                base_support,
                q_poly,
                boundary_amplitudes,
                window,
                mu_eff,
                sigma_eff,
                float(log_mean),
                int(k),
            )
        )

    @property
    def support(self):
        """Support of the density in exp coordinates.

        Returns
        -------
        numpy.ndarray
        """
        self._p._ensure_fitted()
        L, U = np.asarray(self._p.base.support, dtype=np.float64)
        lo = 0.0 if np.isneginf(L) else float(np.exp(L))
        hi = np.inf if np.isposinf(U) else float(np.exp(U))
        return np.array([lo, hi], dtype=np.float64)

    @property
    def median(self):
        """Median in exp coordinates."""
        self._p._ensure_fitted()
        return float(np.exp(self._p.base.median))

    @property
    def mode(self):
        """Mode in exp coordinates."""
        return float(self._ensure_stats()["mode"])

    @property
    def mean(self):
        """Mean in exp coordinates."""
        return float(self._ensure_stats()["mean"])

    @property
    def var(self):
        """Variance in exp coordinates."""
        return float(self._ensure_stats()["var"])

    @property
    def std(self):
        """Standard deviation in exp coordinates."""
        return float(self._ensure_stats()["std"])

    @property
    def skew(self):
        """Skewness in exp coordinates."""
        return float(self._ensure_stats()["skew"])

    @property
    def kurt(self):
        """Kurtosis in exp coordinates (Pearson; 3.0 for a Gaussian)."""
        return float(self._ensure_stats()["kurt"])

    def pdf(self, y):
        """Evaluate the PDF in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Positive evaluation point or points in exp space.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        y = np.asarray(y, dtype=np.float64)
        scalar = y.ndim == 0
        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            x = np.log(y)
            out = self._p.base.pdf(x) / y
            # Zero density at and below zero; NaN input propagates.
            out = np.where(y > 0.0, out, np.where(np.isnan(y), np.nan, 0.0))
        return float(out) if scalar else out

    def cdf(self, y):
        """Evaluate the CDF in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Positive evaluation point or points in exp space.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        y = np.asarray(y, dtype=np.float64)
        scalar = y.ndim == 0
        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            x = np.log(y)
            out = self._p.base.cdf(x)
            out = np.where(y > 0.0, out, np.where(np.isnan(y), np.nan, 0.0))
            out = np.clip(out, 0.0, 1.0)
        return float(out) if scalar else out

    def ppf(self, p):
        """Evaluate the PPF in exp coordinates.

        Parameters
        ----------
        p : float or array_like
            CDF probability or probabilities in the closed interval ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        p = np.asarray(p, dtype=np.float64)
        scalar = p.ndim == 0
        if np.any(~np.isfinite(p) | (p < 0.0) | (p > 1.0)):
            raise ValueError("ppf is defined for finite p in [0, 1]")
        out = np.exp(self._p.base.ppf(p))
        return float(out) if scalar else out

    def neg_log(self, y, n: int = 0):
        """Evaluate the potential or its derivatives in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Positive evaluation point or points in exp space.
        n : int, optional
            Derivative order; zero requests the potential itself.

        Returns
        -------
        float or numpy.ndarray
        """
        self._p._ensure_fitted()
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        y = np.asarray(y, dtype=np.float64)
        scalar = y.ndim == 0
        out = self._p._base_exp_potential(y, n)
        return float(out) if scalar else out

    def logpdf(self, y):
        """Evaluate the log density in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Positive evaluation coordinate(s).
        """
        arr = np.asarray(y, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(arr)
            out = np.asarray(self._p.base.logpdf(x)) - np.log(arr)
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, -np.inf))
        return float(out) if scalar else out

    def logcdf(self, y):
        """Evaluate the tail-accurate log CDF in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        """
        arr = np.asarray(y, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.asarray(self._p.base.logcdf(np.log(arr)))
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, -np.inf))
        return float(out) if scalar else out

    def logsf(self, y):
        """Evaluate the tail-accurate log survival in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        """
        arr = np.asarray(y, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.asarray(self._p.base.logsf(np.log(arr)))
        out = np.where(arr > 0.0, out, np.where(np.isnan(arr), np.nan, 0.0))
        return float(out) if scalar else out

    def sf(self, y):
        """Evaluate survival in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        """
        scalar = np.asarray(y).ndim == 0
        with np.errstate(under="ignore"):
            out = np.exp(self.logsf(y))
        return float(out) if scalar else out

    def isf(self, p):
        """Evaluate the inverse survival function in exp coordinates.

        Parameters
        ----------
        p : float or array_like
            Survival probabilities in ``[0, 1]``.
        """
        scalar = np.asarray(p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(self._p.base.isf(p))
        return float(out) if scalar else out

    def logppf(self, log_p):
        """Evaluate exp-space quantiles from logarithmic CDF probabilities.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        scalar = np.asarray(log_p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(self._p.base.logppf(log_p))
        return float(out) if scalar else out

    def logisf(self, log_p):
        """Evaluate exp-space upper quantiles from log survival probabilities.

        Parameters
        ----------
        log_p : float or array_like
            Log probabilities no greater than zero.
        """
        scalar = np.asarray(log_p).ndim == 0
        with np.errstate(over="ignore"):
            out = np.exp(self._p.base.logisf(log_p))
        return float(out) if scalar else out

    def log_hazard(self, y):
        """Evaluate log hazard in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        """
        arr = np.asarray(y, dtype=np.float64)
        scalar = arr.ndim == 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.asarray(self._p.base.log_hazard(np.log(arr))) - np.log(arr)
        lo, hi = map(float, self.support)
        out = np.where(arr < lo, -np.inf, out)
        if np.isfinite(hi):
            out = np.where(arr >= hi, np.inf, out)
        out = np.where(np.isnan(arr), np.nan, out)
        return float(out) if scalar else out

    def hazard(self, y, n: int = 0):
        """Evaluate the exp-space hazard or its first derivative.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        n : {0, 1}, optional
            Derivative order.
        """
        n = int(n)
        if n not in (0, 1):
            raise ValueError("hazard supports only n=0 or n=1")
        scalar = np.asarray(y).ndim == 0
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            h = np.asarray(np.exp(self.log_hazard(y)), dtype=np.float64)
        out = (
            h if n == 0 else h * (h - np.asarray(self.neg_log(y, 1), dtype=np.float64))
        )
        return float(out) if scalar else out

    def cumulative_hazard(self, y):
        """Evaluate cumulative hazard in exp coordinates.

        Parameters
        ----------
        y : float or array_like
            Evaluation coordinate(s).
        """
        scalar = np.asarray(y).ndim == 0
        out = -np.asarray(self.logsf(y), dtype=np.float64)
        return float(out) if scalar else out

    def mean_residual_life(self, y):
        """Return ``E[Y-y | Y>y]`` computed directly in exp space.

        Parameters
        ----------
        y : float or array_like
            Conditioning threshold(s).
        """
        return _mean_residual_life(
            self.neg_log, self.logsf, self.support, self.mean, self.mode, y
        )

    def residual_entropy(self, y):
        """Return differential entropy of ``Y | Y>y``.

        Parameters
        ----------
        y : float or array_like
            Conditioning threshold(s).
        """
        return _residual_entropy(self.neg_log, self.logsf, self.support, y)

    def interval(self, level):
        """Return an equal-tailed exp-space interval.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        return _interval(self.ppf, self.isf, self.support, level)

    def hpd(self, level):
        """Return the exp-space highest-density region.

        Parameters
        ----------
        level : float
            Probability mass in ``(0, 1]``.
        """
        return _hpd(
            self.logpdf,
            self.logcdf,
            self.logsf,
            self.ppf,
            self.isf,
            self.support,
            level,
            modes=(self.mode,),
        )

    def expect(self, func):
        """Compute ``E[func(Y)]`` using base-space quadrature.

        Parameters
        ----------
        func : callable
            Scalar function of the exp-space random variable.
        """

        def transformed(x):
            with np.errstate(over="ignore"):
                y = float(np.exp(x))
            return func(y)

        return self._p.base.expect(transformed)

    def entropy(self):
        """Return exp-space entropy via ``H(exp X) = H(X) + E[X]``."""
        return float(self._p.base.entropy() + self._p.base.mean)

    def cross_entropy(self, other):
        """Return cross-entropy against another exp-space view.

        Parameters
        ----------
        other : _ExpSpaceView
            Other fitted exp-space distribution.
        """
        if not isinstance(other, _ExpSpaceView):
            raise ValueError("cross_entropy requires another exp-space view")
        lo, hi = map(float, self.support)
        olo, ohi = map(float, other.support)
        if lo < olo or hi > ohi:
            return np.inf
        base = self._p.base

        def transformed(x):
            with np.errstate(over="ignore"):
                return other.neg_log(np.exp(x), 0)

        return _expect_vectorized(
            base.neg_log, base.support, transformed, points=(base.mode,)
        )

    def kl_divergence(self, other):
        """Return KL divergence, invariant under the common exp transform.

        Parameters
        ----------
        other : _ExpSpaceView
            Other fitted exp-space distribution.
        """
        if not isinstance(other, _ExpSpaceView):
            raise ValueError("kl_divergence requires another exp-space view")
        return self._p.base.kl_divergence(other._p.base)

    def loglik(self, x, sample_weight=None):
        """Return weighted held-out log likelihood in exp coordinates.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        sample_weight : array_like or None, optional
            Non-negative row weights; values are not normalized.
        """
        rows = _canonical_scoring_rows(x)
        _warn_out_of_support(rows, self.support)
        weights = _scoring_weights(rows.shape[0], sample_weight)
        if rows.shape[1] == 1:
            positive = weights > 0.0
            terms = np.asarray(self.logpdf(rows[:, 0]), dtype=np.float64)
            return float(np.sum(weights[positive] * terms[positive], dtype=np.float64))
        return _interval_loglik(rows, weights, self.logpdf, self.logcdf, self.logsf)

    def quantile_residuals(self, x, rng=None):
        """Return normal-score residuals, randomized for interval rows.

        Parameters
        ----------
        x : array_like
            Exact observations or ``(n, 2)`` intervals.
        rng : optional
            Random-number source.
        """
        rows = _canonical_scoring_rows(x)
        return _randomized_quantile_residuals(rows, self.logcdf, self.logsf, rng=rng)

    def tail_rate(self, side):
        """Reject tail-rate transport to exp space.

        Parameters
        ----------
        side : {'lower', 'upper'}
            Requested tail.
        """
        raise ValueError("tail_rate is defined only in base space")

    def sample(self, size=None, rng=None):
        """Draw random samples in exp coordinates.

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
        s = self._p.base.sample(size=size, rng=rng)
        return np.exp(s)

    def moment(
        self, k: int, central: bool = False, standardized: bool = False
    ) -> float:
        """Compute the *k*-th moment in exp coordinates.

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
        self._p._ensure_fitted()
        if isinstance(k, bool) or int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        k = int(k)
        if not central and not standardized:
            return float(self._raw_moment(k))
        low_order = _exp_moment_from_stats(k, self._ensure_stats(), standardized)
        if low_order is not None:
            return low_order
        return _pf._moment_from_raw(
            self._raw_moment,
            k,
            float(self.mean),
            float(self.std),
            bool(central),
            bool(standardized),
        )

    def cumulant(self, k: int) -> float:
        """Return the *k*-th cumulant in exp coordinates.

        Parameters
        ----------
        k : int
            Positive integer cumulant order.

        Returns
        -------
        float
            Requested cumulant in exp coordinates.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        ValueError
            If *k* is not a positive integer.
        """
        self._p._ensure_fitted()
        self._invalidate_if_needed()
        if isinstance(k, bool) or int(k) != k or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        order = int(k)
        if order not in self._cumulant_cache:
            self._cumulant_cache[order] = _pf._cumulant_from_centered(
                lambda n: self.moment(n, central=True), order, self.mean
            )
        return float(self._cumulant_cache[order])
