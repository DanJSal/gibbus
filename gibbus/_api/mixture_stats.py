"""Mixture-level analytics: moments, potentials, and the spectral cache.

A ``K > 1`` model is not served by any single component's fitted state, so
everything mixture-specific is gathered here:

* **Moments and summary statistics** — computed from *centered* component
  moments about a shared origin.  Forming ``E[X^2] - E[X]^2`` after a large
  common translation loses every variance bit, so the location is factored
  out before combining.
* **Potentials** — the mixture negative-log density and its derivatives,
  evaluated through a batched log-sum-exp recurrence rather than a Python
  loop over points.
* **Spectral cache** — component spectral states remain the serialized
  source of truth; the mixture-level CDF/PPF is cheap enough to rebuild
  deterministically after fitting, loading, or an affine transform, and so
  is never packed.

This is a mixin because every method needs the model's components, weights
and active space; splitting it out keeps :mod:`gibbus._api.distribution` about the public
contract rather than about mixture numerics.
"""

from math import comb

import numpy as np

from .._defaults import (
    NUMERIC_FAILURES,
    PPF_BISECT_MAX_ITER,
    PPF_BISECT_Z_TOL,
    _reraise_if_debug,
)
from .._fit.mixture import _find_mixture_modes_base
from .._postfit import analytics as _pf
from .._postfit.analytics import _exp_stats_from_log_moments
from .._postfit.evaluators import _potential_exp_from_x_potential
from .._postfit.logsumexp import _neg_log_mix_derivs_batch, _neg_logsumexp_batch
from .._spectral.cdf import SpectralCDF, density_spec
from .._spectral.ppf import SpectralPPF


class _MixtureAnalyticsMixin:
    """Mixture moments, potentials, and CDF/PPF cache management for ``Distribution``."""

    _stats_cache: dict | None
    _mode_cache: dict | None

    def _raw_moment(self, k):
        """Compute the *k*-th raw moment of the mixture in the active space.

        Parameters
        ----------
        k : int
            Moment order.

        Returns
        -------
        float
        """
        k = int(k)
        if k == 0:
            return 1.0
        s = 0.0
        for weight, comp in zip(self._weights, self._components, strict=True):
            if weight <= 0.0:
                continue
            s += float(weight) * self._view(comp).moment(k)
        return float(s)

    def _mixture_exp_log_raw_moment(self, k):
        """Return a mixture exp-space raw moment in a shared logarithmic unit.

        Parameters
        ----------
        k : int
            Non-negative raw-moment order in exp space.
        """
        kk = int(k)
        if kk == 0:
            return 0.0
        logs = np.array(
            [
                np.log(float(w)) + comp.exp._log_raw_moment(kk)
                for w, comp in zip(self._weights, self._components, strict=True)
                if w > 0.0
            ],
            dtype=np.float64,
        )
        if logs.size == 0:
            raise RuntimeError("mixture has no positive-weight components")
        m = float(np.max(logs))
        if not np.isfinite(m):
            return m
        return float(m + np.log(np.sum(np.exp(logs - m))))

    def _mixture_exp_relative_centered_moment(self, log_mean, k):
        """Return a narrow-law exp-mixture centered moment in relative units.

        Parameters
        ----------
        log_mean : float
            Logarithm of the mixture exp-space mean.
        k : int
            Centered-moment order.

        Returns
        -------
        float
        """
        total = 0.0
        for weight, comp in zip(self._weights, self._components, strict=True):
            if weight <= 0.0:
                continue
            view = comp.exp
            view._ensure_q_cache()
            (
                base_support,
                q_poly,
                boundary_amplitudes,
                window,
                mu_eff,
                sigma_eff,
                _terms,
            ) = view._q_cache
            total += float(weight) * _pf._relative_centered_moment_exp(
                base_support,
                q_poly,
                boundary_amplitudes,
                window,
                mu_eff,
                sigma_eff,
                float(log_mean),
                int(k),
            )
        return float(total)

    def _ensure_exp_mixture_stats(self):
        """Build exp-mixture statistics without materializing raw moments."""
        return _exp_stats_from_log_moments(
            [self._mixture_exp_log_raw_moment(k) for k in (1, 2, 3, 4)],
            self._mixture_exp_relative_centered_moment,
            "exp-space mixture",
        )

    def _mixture_location_parts(self):
        """Return ``(origin, local_component_means, views)`` stably."""
        views = [self._view(comp) for comp in self._components]
        active = np.flatnonzero(np.asarray(self._weights, dtype=np.float64) > 0.0)
        if active.size == 0:
            raise RuntimeError("mixture has no positive-weight components")

        if self._default == "base":
            parts = [self._components[j]._base_mean_parts() for j in active]
            origins = np.array([part[0] for part in parts], dtype=np.float64)
            locals_active = np.array([part[1] for part in parts], dtype=np.float64)
            if np.all(origins == origins[0]):
                locals_ = np.zeros(self._K, dtype=np.float64)
                locals_[active] = locals_active
                return float(origins[0]), locals_, views

        # Evaluate component means only where they can contribute.  A dormant
        # zero-weight component is allowed to have infinite transformed
        # moments and must not contaminate the active mixture through 0*inf.
        ref_index = int(active[np.argmax(np.asarray(self._weights)[active])])
        ref = float(views[ref_index].mean)
        locals_ = np.zeros(self._K, dtype=np.float64)
        for j in active:
            locals_[j] = float(views[int(j)].mean) - ref
        return ref, locals_, views

    def _mixture_centered_moment(self, k):
        """Return a centered mixture moment from centered component moments.

        Parameters
        ----------
        k : int
            Non-negative centered-moment order.
        """
        kk = int(k)
        if kk == 0:
            return 1.0
        _origin, local_means, views = self._mixture_location_parts()
        local_mean = float(np.dot(self._weights, local_means))
        deltas = local_means - local_mean

        total = 0.0
        for w, view, delta in zip(self._weights, views, deltas, strict=True):
            if w <= 0.0:
                continue
            comp = 0.0
            for r in range(kk + 1):
                if r == 0:
                    mu_r = 1.0
                elif r == 1:
                    mu_r = 0.0
                else:
                    mu_r = float(view.moment(r, central=True))
                comp += comb(kk, r) * (float(delta) ** (kk - r)) * mu_r
            total += float(w) * comp
        return float(total)

    def _mixture_mean_centered_origin(self):
        """Return the mixture mean without large translated intermediate means."""
        origin, local_means, _views = self._mixture_location_parts()
        return float(origin + np.dot(self._weights, local_means))

    def _ensure_stats(self):
        """Return the mixture statistics, computing them on first use."""
        if self._stats_cache is not None:
            return self._stats_cache
        self._ensure_fitted()
        if self._default == "exp":
            stats = self._ensure_exp_mixture_stats()
        else:
            mean = self._mixture_mean_centered_origin()
            mu2 = self._mixture_centered_moment(2)
            mu3 = self._mixture_centered_moment(3)
            mu4 = self._mixture_centered_moment(4)
            stats = _pf._stats_from_centered_moments(mean, mu2, mu3, mu4)
        self._stats_cache = stats
        return stats

    def _ensure_modes(self):
        """Return the mixture mode cache with base-space modes present."""
        if self._mode_cache is not None and "base" in self._mode_cache:
            return self._mode_cache
        self._ensure_fitted()
        comp_modes = [c.base.mode for c in self._components]
        base_modes = _find_mixture_modes_base(self._mix_base_potential, comp_modes)
        modes = {"base": base_modes}
        self._mode_cache = modes
        return modes

    # ---- Mixture potential helpers ----

    def _mix_base_potential(self, x, n):
        """Evaluate the mixture base-space potential or its *n*-th derivative.

        This method is fully vectorized: all evaluation points are batched
        through the component ``neg_log`` evaluators and the recurrence
        in :func:`_neg_log_mix_derivs_batch`, so evaluation does not require a
        Python-level per-element loop.

        Parameters
        ----------
        x : array_like
            Evaluation point(s) in base coordinates.
        n : int
            Derivative order.

        Returns
        -------
        float or numpy.ndarray
        """
        x_arr = np.asarray(x, dtype=np.float64)
        scalar = x_arr.ndim == 0
        x_arr = np.atleast_1d(x_arr)
        K = self._K
        R = x_arr.shape[0]
        with np.errstate(divide="ignore"):
            log_w = np.where(self._weights > 0.0, np.log(self._weights), -np.inf)

        # Order zero is by far the hottest scalar-quadrature case.  Avoid
        # constructing derivative jets or Taylor work arrays entirely.
        ell0 = np.empty((K, R), dtype=np.float64)
        for j in range(K):
            neg_log_0 = self._components[j].base.neg_log(x_arr, 0)
            ell0[j, :] = log_w[j] - np.asarray(neg_log_0, dtype=np.float64)
        if int(n) == 0:
            out = _neg_logsumexp_batch(np.ascontiguousarray(ell0))
            return out.item() if scalar else out

        # Higher derivative jets use the same recurrence, implemented below
        # the Python boundary in a compact Cython kernel.
        ell_jets = np.empty((K, n + 1, R), dtype=np.float64)
        ell_jets[:, 0, :] = ell0
        for j in range(K):
            for m in range(1, n + 1):
                ell_jets[j, m, :] = -np.asarray(
                    self._components[j].base.neg_log(x_arr, m), dtype=np.float64
                )
        all_derivs = _neg_log_mix_derivs_batch(np.ascontiguousarray(ell_jets), n)
        out = all_derivs[n]
        return out.item() if scalar else out

    def _mix_base_log_tail_mass(self, x, endpoint, /, *, upper=False):
        """Combine exact component tail masses with a compiled stable reduction.

        Parameters
        ----------
        x : float
            Tail anchor in base coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        with np.errstate(divide="ignore"):
            log_w = np.where(self._weights > 0.0, np.log(self._weights), -np.inf)
        logs = np.empty((self._K, 1), dtype=np.float64)
        for j, comp in enumerate(self._components):
            logs[j, 0] = log_w[j] + comp.base._exact_tail_log_mass(
                float(x), float(endpoint), upper=bool(upper)
            )
        return float(-_neg_logsumexp_batch(np.ascontiguousarray(logs))[0])

    def _mix_base_log_tail_masses(self, x, endpoint, /, *, upper=False):
        """Batched ``_mix_base_log_tail_mass`` over an array of anchors.

        Parameters
        ----------
        x : numpy.ndarray
            Tail anchors in base coordinates.
        endpoint : float
            Support endpoint in the requested tail direction.
        upper : bool, optional
            Select the upper tail instead of the lower tail.
        """
        xs = np.asarray(x, dtype=np.float64).reshape(-1)
        with np.errstate(divide="ignore"):
            log_w = np.where(self._weights > 0.0, np.log(self._weights), -np.inf)
        logs = np.empty((self._K, xs.size), dtype=np.float64)
        for j, comp in enumerate(self._components):
            logs[j] = log_w[j] + comp.base._exact_tail_log_masses(
                xs, float(endpoint), upper=bool(upper)
            )
        return -np.asarray(
            _neg_logsumexp_batch(np.ascontiguousarray(logs)), dtype=np.float64
        )

    def _mix_exp_potential(self, y, n):
        """Evaluate the mixture exp-space potential or its *n*-th derivative.

        Parameters
        ----------
        y : array_like
            Evaluation point(s) in exp coordinates.
        n : int
            Derivative order.

        Returns
        -------
        float or numpy.ndarray
        """
        pot_y = _potential_exp_from_x_potential(self._mix_base_potential)
        return pot_y(y, n)

    # ---- Mixture CDF/PPF cache ----

    def _ensure_spectral_cache(self):
        """Lazily build the mixture spectral CDF/PPF on first access."""
        if self._spectral_cache_valid:
            return
        self._rebuild_spectral_cache()
        self._spectral_cache_valid = True

    def _rebuild_spectral_cache(self):
        """Build mixture CDF/PPF from the same spectral machinery as components.

        Component spectral states remain the serialized source of truth.  A
        mixture-level spectral representation is cheap enough to rebuild
        deterministically after fitting, loading, or an affine transform.
        """
        if self._K <= 1:
            self._mix_base_cdf = None
            self._mix_base_ppf = None
            self._mix_spectral_cdf_rep = None
            self._mix_spectral_ppf_rep = None
            return

        weights = np.asarray(self._weights, dtype=np.float64)
        components = self._components
        support = np.asarray(components[0].base.support, dtype=np.float64)

        def mix_pdf(x):
            arr = np.asarray(x, dtype=np.float64)
            scalar = arr.ndim == 0
            out = np.zeros_like(arr, dtype=np.float64)
            for k, comp in enumerate(components):
                out += weights[k] * np.asarray(comp.base.pdf(arr), dtype=np.float64)
            return float(out) if scalar else out

        means = np.array([c.base.mean for c in components], dtype=np.float64)
        vars_ = np.array([c.base.var for c in components], dtype=np.float64)
        # Build the mixture spectral scale from centered component statistics.
        # Forming E[X^2] - E[X]^2 loses every variance bit after a large common
        # translation, which can collapse the spectral coordinate to a step.
        ref = float(means[int(np.argmax(weights))])
        offsets = means - ref
        mean_offset = float(np.dot(weights, offsets))
        centered = offsets - mean_offset
        var = float(np.dot(weights, vars_ + centered * centered))
        std = float(np.sqrt(max(var, np.finfo(np.float64).tiny)))

        comp_modes = np.array([c.base.mode for c in components], dtype=np.float64)
        dens_modes = np.asarray(mix_pdf(comp_modes), dtype=np.float64)
        mode = float(comp_modes[int(np.nanargmax(dens_modes))])

        parts = []
        for w, comp in zip(weights, components, strict=True):
            mu_eff, sigma_eff = comp._mu_sigma_eff()
            z_support, kernel_amps = comp._internal_model_geometry()
            lo, hi = map(float, comp.base.support)
            parts.append(
                (
                    comp._data["q_poly"],
                    float(z_support[0]),
                    float(z_support[1]),
                    float(kernel_amps[0]),
                    float(kernel_amps[1]),
                    0.0,
                    float(mu_eff),
                    float(sigma_eff),
                    abs(float(sigma_eff)),
                    float(w),
                    lo,
                    hi,
                )
            )
        cdf_rep = SpectralCDF(
            support,
            density=density_spec(parts, view=True),
            mode=mode,
            std=std,
        )
        try:
            ppf_rep = SpectralPPF(cdf_rep)
        except NUMERIC_FAILURES as exc:
            # Same contract as the component fitted-state path:
            # an inverse that will not certify monotone must not take the
            # mixture density with it.  Fall back to exact bisection of the
            # spectral CDF, which is slower per query but correct.
            _reraise_if_debug(exc, "mixture spectral PPF construction", routine=True)
            ppf_rep = None

        self._mix_spectral_cdf_rep = cdf_rep
        self._mix_spectral_ppf_rep = ppf_rep
        self._mix_base_cdf = cdf_rep.cdf_cython
        if ppf_rep is not None:
            self._mix_base_ppf = ppf_rep.ppf_cython
        else:

            def _bisect_ppf(p, _cdf=cdf_rep):
                """Quantiles by monotone bisection of the spectral CDF.

                Bisection on ``z in [-1, 1]`` halves the bracket each
                pass, so ``PPF_BISECT_MAX_ITER`` steps reach the double
                precision floor; iterating past that costs a full CDF
                evaluation over the query array per step and cannot
                improve the answer.  Interior points that have already
                converged are dropped from the working set, so a query
                dominated by easy probabilities does not pay for the
                hardest one.
                """
                arr = np.asarray(p, dtype=np.float64)
                scalar = arr.ndim == 0
                flat = np.atleast_1d(arr).astype(np.float64, copy=True).reshape(-1)
                lo = np.full(flat.shape, -1.0)
                hi = np.full(flat.shape, 1.0)
                active = np.flatnonzero((flat > 0.0) & (flat < 1.0))
                for _ in range(PPF_BISECT_MAX_ITER):
                    if active.size == 0:
                        break
                    mid = 0.5 * (lo[active] + hi[active])
                    below = np.asarray(_cdf.cdf_z(mid), dtype=np.float64) < flat[active]
                    lo[active] = np.where(below, mid, lo[active])
                    hi[active] = np.where(below, hi[active], mid)
                    active = active[(hi[active] - lo[active]) > PPF_BISECT_Z_TOL]
                z = 0.5 * (lo + hi)
                z = np.where(flat <= 0.0, -1.0, np.where(flat >= 1.0, 1.0, z))
                out = np.asarray(_cdf.map.x_from_z(z), dtype=np.float64)
                return float(out[0]) if scalar else out.reshape(arr.shape)

            self._mix_base_ppf = _bisect_ppf
