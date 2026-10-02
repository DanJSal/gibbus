"""Diagnostic records exposed by :class:`~gibbus._api.distribution.Distribution`.

Three independent questions, deliberately answered separately:

* **Did the likelihood optimisation converge?** — :attr:`fit_diagnostics`,
  covering every component's natural-conic certificate plus the EM session
  whenever that in-process diagnostic record is available.
* **Why was this component count chosen?** — :attr:`selection_diagnostics`.
* **Was the CDF/PPF that queries actually use resolved?** —
  :attr:`spectral_diagnostics`.

The third matters because spectral construction *degrades rather than
raises*, so an under-resolved representation is indistinguishable from a
converged one unless it is recorded.

These are collected in a mixin rather than as free functions because each is
a public property whose contract belongs to ``Distribution``; keeping them here
separates that reporting surface from the estimator's own behaviour.
"""

import copy

import numpy as np

from .._defaults import (
    HAZARD_MONOTONE_TOL,
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._fit.boundary import _weakly_identified_sides


class _DiagnosticsMixin:
    """Fit, selection and spectral diagnostic properties for ``Distribution``."""

    @property
    def fit_diagnostics(self):
        """Convergence diagnostics for the fitted likelihood optimization.

        Returns
        -------
        dict
            ``converged`` summarizes all component optimizers and, when
            available, the final EM run. ``components`` is a tuple of
            per-component optimizer records with natural-conic termination
            status and certified decrease bounds, plus each boundary term's
            amplitude standard error (``boundary_standard_errors``, lower and
            upper, ``nan`` without a term), the p-value of the test that kept
            or dropped an automatically chosen term (``boundary_p_values``,
            ``nan`` where none ran) and the sides whose amplitude lies within
            two standard errors of zero
            (``weakly_identified_boundary_terms``). ``em`` is ``None`` for a
            single-component fit or for a mixture
            reconstructed only from the portable structured ``data`` state,
            which intentionally omits session-level EM diagnostics. Copies
            and Python pickle preserve the in-process EM record.
            ``hazard_is_monotone`` is ``True`` or ``False`` when the numerical
            body check completes and ``None`` when that check is indeterminate
            because of a numerical failure.
        """
        self._ensure_fitted()
        # The hazard body check is intentionally cached: it requires two inverse
        # probability solves and a 257-point hazard sweep.  Include the relevant
        # class-level callables in the cache key so instrumentation/monkeypatching
        # still causes a fresh diagnostic computation.
        cache_key = (
            self._default,
            type(self).ppf,
            type(self).isf,
            type(self).log_hazard,
        )
        cached = self._fit_diagnostics_cache
        if cached is not None and cached[0] == cache_key:
            return copy.deepcopy(cached[1])

        component_records = []
        for index, comp in enumerate(self._components):
            state = comp.data
            component_records.append({
                "component": int(index),
                "success": bool(int(state["optimizer_success"])),
                "status": str(state["optimizer_status"]),
                "message": str(state["optimizer_message"]),
                "n_iterations": int(state["optimizer_n_iterations"]),
                "n_evaluations": int(state["optimizer_n_evaluations"]),
                "subproblem_iterations": int(
                    state["optimizer_subproblem_iterations"]),
                "decrease_bound": float(state["optimizer_decrease_bound"]),
                "effective_curvature_degree": int(
                    state["effective_curvature_degree"]),
                "lower_amplitude_active": bool(
                    int(state["lower_amplitude_active"])),
                "upper_amplitude_active": bool(
                    int(state["upper_amplitude_active"])),
                "separator_certified": bool(int(state["separator_certified"])),
                "boundary_standard_errors": tuple(
                    float(v) for v in np.asarray(state["boundary_standard_errors"]).ravel()
                ),
                "boundary_p_values": tuple(
                    float(v) for v in np.asarray(state["boundary_p_values"]).ravel()
                ),
                "weakly_identified_boundary_terms": _weakly_identified_sides(
                    state["boundary_amplitudes"], state["boundary_standard_errors"]
                ),
            })
        optimizer_ok = all(r.get("success") is True for r in component_records)
        em = None if self._em_diagnostics is None else dict(self._em_diagnostics)
        em_ok = True if em is None else bool(em.get("converged", False))
        try:
            lo = float(self.ppf(1e-6))
            hi = float(self.isf(1e-6))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                grid = np.linspace(lo, hi, 257)
                log_h = np.asarray(self.log_hazard(grid), dtype=np.float64)
                finite = np.isfinite(log_h)
                vals = log_h[finite]
                hazard_is_monotone = (
                    None if vals.size < 2
                    else bool(np.all(np.diff(vals) >= -float(HAZARD_MONOTONE_TOL)))
                )
            else:
                hazard_is_monotone = None
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, "hazard monotonicity diagnostic")
            hazard_is_monotone = None
        result = {
            "converged": bool(optimizer_ok and em_ok),
            "components": tuple(component_records),
            "em": em,
            "hazard_is_monotone": hazard_is_monotone,
        }
        self._fit_diagnostics_cache = (cache_key, result)
        return copy.deepcopy(result)

    @property
    def selection_diagnostics(self):
        """Automatic component-count selection record for the latest fit.

        Returns
        -------
        dict or None
            ``None`` when component count was explicit or the model was
            reconstructed only from the portable structured ``data`` state.
            Copies and Python pickle preserve this record. Otherwise includes
            the KDE proposal, whether selection was subsampled, the selected
            K/BIC, and all candidate score records visited by the sweep.
        """
        if self._selection_diagnostics is None:
            return None
        out = dict(self._selection_diagnostics)
        out["scores"] = tuple(dict(item) for item in out.get("scores", ()))
        return out

    @property
    def spectral_diagnostics(self):
        """Convergence diagnostics for the CDF/PPF actually used by queries.

        Spectral CDF construction degrades rather than raises: at
        ``max_depth``, a degenerate panel width, or budget exhaustion it
        accepts the best panel so far.  That makes an under-resolved
        representation indistinguishable from a converged one unless it
        is recorded, so this reports the record.

        The point of routing it through one property is that ``K == 1``
        and ``K > 1`` are answered by *different* representations: a
        single component is served by its own packed ``cdf_*`` fields,
        while a mixture is served by a separately built mixture-level
        CDF that is rebuilt on demand and never packed.  Reading the
        component fields directly therefore says nothing about what a
        mixture's :meth:`cdf` and :meth:`ppf` will do.

        Returns
        -------
        dict
            ``refinement_capped`` (bool) -- refinement stopped at the depth
            or panel budget rather than on tolerance.  This reports the
            mechanism and fires readily: one sliver of a panel that will
            not resolve against a boundary singularity is enough, on
            otherwise perfectly good fits.  Prefer the mass fields for
            deciding whether a CDF is trustworthy.
            ``uncertified_mass`` (float) -- share of probability sitting
            under panels that failed tolerance.  This is an exposure
            measure, not a ranking metric; ``0.0`` means every panel met
            its internal certification criteria.
            ``mass_defect`` (float) -- ``|1 - total_mass|`` before
            normalisation; large values mean probability the panels
            never resolved, typically at a boundary singularity.
            ``worst_panel_error`` (float) -- largest local panel error
            estimate after conversion to probability units.
            ``error_estimate`` (float) -- overall probability-scale CDF
            health estimate combining normalization defect, uncertified
            mass, and the largest certified local panel error.
            ``n_panels`` (int) and ``n_masses_recertified`` (int or None) --
            panels whose mass had to be re-measured by quadrature
            because their polynomial was not certified.
            ``ppf_fallback`` (bool) -- the stored spectral inverse uses CDF
            bisection rather than fitted inverse panels. Extreme public tail
            queries may subsequently use direct potential-based inversion.
            ``log_concavity_margin`` (float) -- certified lower bound on
            base-space convexity of the potential, minimised over
            components.  Non-negative means every component's polynomial
            is certified convex; the mixture density itself need not be
            log-concave even when each component is.
            ``scope`` (str) -- ``"component"`` or ``"mixture"``.
        """
        self._ensure_fitted()
        if self._K == 1:
            state = self._components[0].data
            return {
                "scope": "component",
                "refinement_capped": bool(int(state["cdf_refinement_capped"])),
                "uncertified_mass": float(state["cdf_uncertified_mass"]),
                "mass_defect": float(state["cdf_mass_defect"]),
                "worst_panel_error": float(state["cdf_worst_panel_error"]),
                "error_estimate": float(state["cdf_error_estimate"]),
                "n_panels": int(state["cdf_npanels"]),
                "n_masses_recertified": None,
                "ppf_fallback": bool(int(state["ppf_fallback"])),
                "log_concavity_margin": self._log_concavity_margin(),
            }

        self._ensure_spectral_cache()
        rep = self._mix_spectral_cdf_rep
        return {
            "scope": "mixture",
            "refinement_capped": bool(rep.max_depth_used >= int(rep.max_depth)
                                  or rep.panel_budget_exhausted),
            "uncertified_mass": float(rep.uncertified_mass_fraction),
            "mass_defect": float(rep.mass_defect),
            "worst_panel_error": float(rep.worst_panel_error),
            "error_estimate": float(rep.cdf_error_estimate),
            "n_panels": int(len(rep.panels)),
            "n_masses_recertified": int(rep.n_masses_recertified),
            "ppf_fallback": bool(self._mix_spectral_ppf_rep is None),
            "log_concavity_margin": self._log_concavity_margin(),
        }
