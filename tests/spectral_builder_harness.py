"""Pure-Python harness for spectral CDF/PPF construction.

The runtime package builds fitted spectral CDFs and inverses in the compiled
builder (``gibbus._spectral._builders``).  This test-only module keeps the
Python construction control flow so tests can exercise arbitrary callables,
forced failures, panel-budget semantics, and bitwise agreement of builder
orchestration.

This is deliberately *not* an independent numerical oracle: it reuses the
production panel algebra, positivity certificate, packed evaluator classes,
and Chebyshev helper tables.  Independent accuracy checks live in tests that
compare CDF/PPF values with analytic distributions or external quadrature.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
from gibbus._spectral._certify import chebyshev_lower_bound
from gibbus._spectral._panel_kernels import cdf_panel_metrics as _cdf_panel_metrics
from gibbus._spectral._panel_kernels import chebder as _chebder_kernel
from gibbus._spectral._panel_kernels import chebint_scaled as _chebint_scaled
from gibbus._spectral._panel_kernels import chebval_many as _chebval_many
from gibbus._spectral._panel_kernels import (
    lobatto_coefficients as _lobatto_coefficients_kernel,
)
from numpy.polynomial import chebyshev as C
from scipy.special import expit

from gibbus._defaults import SPECTRAL_DEGREE_OPTIONS
from gibbus._spectral.cdf import SpectralCDF, _Map, _Panel
from gibbus._spectral.chebyshev import (
    _lobatto_transform,
    chebyshev_bernstein_matrix,
    lobatto_coefficients,
    lobatto_nodes,
    midpoint_nodes,
)
from gibbus._spectral.ppf import SpectralPPF, _InversePanel, _logit

_EPS = np.finfo(np.float64).eps
_PMIN = np.nextafter(np.float64(0.0), np.float64(1.0))
_PMAX = np.nextafter(np.float64(1.0), np.float64(0.0))

_RECERTIFY_GAUSS_N = 24
"""Gauss-Legendre nodes per subinterval when re-measuring a panel's mass."""

_RECERTIFY_SUBPANELS = 8
"""Subintervals per uncertified panel for the composite mass quadrature."""


class PythonSpectralCDFBuilder(SpectralCDF):
    """Spectral CDF built by the test-only Python construction harness.

    Takes a vectorized density callable, or a density description whose
    ``pdf`` is used.  Panel-construction control flow is Python-level; panel
    algebra, normalization, packing, evaluation, and diagnostics reuse the
    production implementation.
    """

    def __init__(
        self,
        pdf=None,
        support=None,
        *,
        density=None,
        mode=None,
        std=None,
        degree_options=SPECTRAL_DEGREE_OPTIONS,
        **options,
    ):
        """Build the Python-harness representation.

        Parameters
        ----------
        pdf : callable or None
            Vectorized density in physical coordinates (need not be
            normalized); defaults to ``density.pdf``.
        support : sequence of (float, float)
            Density support.
        density : DensitySpec or None, optional
            Density description; kept on the object, so a production
            ``SpectralPPF`` of this CDF uses the compiled inverse builder.
        mode, std, degree_options, **options
            As for :class:`gibbus._spectral.cdf.SpectralCDF`.
        """
        if pdf is None:
            if density is None:
                raise TypeError("a pdf callable or a density description is required")
            pdf = density.pdf
        self.pdf = pdf
        rel_tol = options.pop("rel_tol", None)
        abs_tol = options.pop("abs_tol", None)
        coeff_tol = options.pop("coeff_tol", None)
        max_depth = options.pop("max_depth", None)
        max_panels = options.pop("max_panels", None)
        map_scale = options.pop("map_scale", None)
        initial_breaks = options.pop("initial_breaks", None)
        if options:
            raise TypeError(f"unexpected options {sorted(options)}")
        defaults = SpectralCDF.__init__.__kwdefaults__
        breaks, _ = self._configure(
            support,
            mode,
            std,
            degree_options,
            defaults["rel_tol"] if rel_tol is None else rel_tol,
            defaults["abs_tol"] if abs_tol is None else abs_tol,
            defaults["coeff_tol"] if coeff_tol is None else coeff_tol,
            defaults["max_depth"] if max_depth is None else max_depth,
            defaults["max_panels"] if max_panels is None else max_panels,
            map_scale,
            initial_breaks,
            density,
        )
        panels = self._build_partition(breaks)
        panels.sort(key=lambda p: p.a)
        # Must happen before the masses are summed: an uncertified panel's
        # analytic integral is not trustworthy, and it would otherwise
        # scale every CDF value on every panel.
        self._finish(panels, recertified=self._recertify_panel_masses(panels))

    def _default_map_scale(self, L, U, mode, s0, kind, /):
        """Choose the map scale in Python (the compiled builder chooses its own).

        Parameters
        ----------
        L, U : float
            Support endpoints.
        mode : float or None
            Density mode.
        s0 : float
            Baseline scale.
        kind : str
            Map family.
        """
        return self._choose_scale(L, U, mode, s0, kind=kind)

    def _choose_scale(self, L, U, mode, s0, *, kind=None):
        """Pick the map scale whose transformed density is easiest to fit.

        Trial scales are scored by the Chebyshev fit error of the
        transformed density plus its coefficient tail; the best is kept.
        The choice matters because a badly scaled compactification piles
        resolution where the density is flat.

        Parameters
        ----------
        L, U : float
            Support endpoints, either of which may be infinite.
        mode : float or None
            Density mode, used to center a doubly-infinite map.
        s0 : float
            Baseline scale, typically the fitted standard deviation.
        kind : {"lower", "upper", "real", "lower_centered", "upper_centered"} or None, optional
            Compactification family to score. If ``None``, infer the ordinary
            lower-, upper-, or doubly-infinite map from the finite support
            endpoints. Centered half-line variants are passed explicitly when
            a finite endpoint is extremely remote from the density body.

        Returns
        -------
        float
            Selected map scale.
        """
        multipliers = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
        best = None
        if kind is None:
            kind = (
                "lower" if np.isfinite(L) else ("upper" if np.isfinite(U) else "real")
            )
        c = float(mode) if mode is not None and np.isfinite(mode) else 0.0
        for mult in multipliers:
            s = max(float(s0) * mult, np.finfo(float).tiny)
            if kind == "lower":
                mp = _Map(kind, L, U, L, s)
            elif kind == "upper":
                mp = _Map(kind, L, U, U, s)
            else:
                mp = _Map(kind, L, U, c, s)
            score = self._trial_score(mp)
            if best is None or score < best[0]:
                best = (score, s)
        return best[1]

    def _g_with_map(self, z, mp):
        """Transformed density ``g(z) = f(x(z)) dx/dz`` under a given map.

        Parameters
        ----------
        z : array_like
            Compact coordinates.
        mp : _Map
            Map to evaluate under.

        Returns
        -------
        numpy.ndarray
            Transformed density, zero where non-finite or negative.
        """
        z = np.asarray(z, dtype=np.float64)
        out = np.zeros_like(z)
        finite_mask = np.ones_like(z, dtype=bool)
        if mp.kind in ("real", "upper", "upper_centered"):
            finite_mask &= z > -1.0
        if mp.kind in ("real", "lower", "lower_centered"):
            finite_mask &= z < 1.0
        if np.any(finite_mask):
            zz = z[finite_mask]
            xx = mp.x_from_z(zz)
            # PDF values at finite endpoints are measure-zero for the CDF.
            # Evaluate the interior limit instead: affine roundoff in Distribution's
            # user->internal map can otherwise classify an exact user endpoint
            # as microscopically outside the internal support.
            if np.isfinite(mp.L):
                xx = np.where(zz <= -1.0, np.nextafter(mp.L, np.inf), xx)
            if np.isfinite(mp.U):
                xx = np.where(zz >= 1.0, np.nextafter(mp.U, -np.inf), xx)
            jj = mp.jac(zz)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                vals = np.asarray(self.pdf(xx), dtype=np.float64) * jj
            vals = np.where(np.isfinite(vals) & (vals >= 0.0), vals, 0.0)
            out[finite_mask] = vals
        return out

    def transformed_pdf(self, z):
        """Transformed density under this object's own map.

        Parameters
        ----------
        z : array_like
            Compact coordinates.

        Returns
        -------
        numpy.ndarray
            Transformed density values.
        """
        return self._g_with_map(z, self.map)

    def _trial_score(self, mp):
        """Score a candidate map: lower is easier to approximate.

        Parameters
        ----------
        mp : _Map
            Candidate map.

        Returns
        -------
        float
            Relative interpolation error plus relative coefficient tail.
        """
        n = min(20, self.degree)
        u = lobatto_nodes(n)
        vals = self._g_with_map(u, mp)
        coeff = lobatto_coefficients(vals)
        m = 2 * n + 3
        uv = midpoint_nodes(m)
        exact = self._g_with_map(uv, mp)
        approx = C.chebval(uv, coeff)
        scale = max(float(np.max(np.abs(exact))), float(np.max(np.abs(vals))), 1e-15)
        err = float(np.max(np.abs(exact - approx))) / scale
        tail = float(np.max(np.abs(coeff[-4:]))) / max(
            float(np.max(np.abs(coeff))), 1e-15
        )
        return err + tail

    @staticmethod
    def _u_to_z(u, a, b):
        """Map panel-local ``u`` in ``[-1, 1]`` to compact ``z`` in ``[a, b]``.

        Parameters
        ----------
        u : array_like
            Panel-local coordinates.
        a, b : float
            Panel edges in compact coordinates.

        Returns
        -------
        numpy.ndarray
            Compact coordinates.
        """
        return 0.5 * ((b - a) * u + (a + b))

    def _target(self, z):
        """Function the panels approximate: the transformed density.

        Parameters
        ----------
        z : array_like
            Compact coordinates.

        Returns
        -------
        numpy.ndarray
            Target values.
        """
        return self.transformed_pdf(z)

    def _fit_panel(self, a, b, depth, n):
        """Fit one Chebyshev panel and certify it non-negative.

        The panel polynomial is lifted by a Bernstein-certified constant
        if it dips below zero, so the analytic antiderivative is monotone
        by construction rather than by inspection.

        Parameters
        ----------
        a, b : float
            Panel edges in compact coordinates.
        depth : int
            Current refinement depth, recorded on the panel.
        n : int
            Polynomial degree to attempt.

        Returns
        -------
        ok : bool
            Whether the panel met the error tolerances.
        panel : _Panel
            The fitted panel, returned whether or not it passed.
        """
        n = int(n)
        u = lobatto_nodes(n)
        z = self._u_to_z(u, a, b)
        vals = self._target(z)
        m = 2 * n + 3
        uv = midpoint_nodes(m)
        zv = self._u_to_z(uv, a, b)
        exact = self._target(zv)
        coeff0, fit_error, data_scale, tail_abs, coeff_scale = _cdf_panel_metrics(
            np.ascontiguousarray(vals, dtype=np.float64),
            np.ascontiguousarray(exact, dtype=np.float64),
            np.ascontiguousarray(_lobatto_transform(n), dtype=np.float64),
            np.ascontiguousarray(uv, dtype=np.float64),
        )
        scale = max(float(data_scale), 1e-300)
        tail_ratio = float(tail_abs) / max(float(coeff_scale), 1e-300)

        # This representation is built for the *integral* of the transformed
        # density, not for pointwise PDF evaluation.  Near a finite endpoint
        # with a non-integer boundary power, g(z) can have an algebraic
        # derivative singularity even though the probability mass of the
        # affected panel is tiny.  Refining on pointwise density error alone
        # therefore drives the panel tree to extreme depths for no CDF benefit.
        # Multiplication by the panel width gives a conservative bound on the
        # induced integral error and makes the stopping criterion match the
        # quantity the representation is meant to approximate.
        panel_width = float(b - a)
        mass_scale = panel_width * scale
        tol_mass = self.abs_tol + self.rel_tol * mass_scale
        tail_tol_mass = self.abs_tol + self.coeff_tol * mass_scale
        converged = (
            panel_width * fit_error <= tol_mass
            and panel_width * tail_abs <= tail_tol_mass
        )

        coeff = coeff0.copy()
        lift = self._positivity_lift_bernstein(coeff)
        if lift > 0.0:
            coeff[0] += lift
        if panel_width * lift > tol_mass:
            converged = False

        # dz = (b-a)/2 du.  The tiny-series integration and endpoint
        # evaluation stay in C to avoid repeated NumPy polynomial dispatch.
        icoeff, mass = _chebint_scaled(
            np.ascontiguousarray(coeff, dtype=np.float64),
            0.5 * (b - a),
        )
        mass = float(mass)
        # Convert the sampled pointwise residual to the quantity the CDF
        # actually accumulates.  ``fit_error`` is measured before the
        # positivity lift, so the lift contributes additively to the local
        # density error.  Multiplication by panel width puts the result in
        # probability-mass units.
        error_mass = panel_width * (fit_error + lift)
        return converged, _Panel(
            a=a,
            b=b,
            coeff=coeff,
            icoeff=icoeff,
            mass=mass,
            fit_error=fit_error,
            error_mass=error_mass,
            tail_ratio=tail_ratio,
            lift=lift,
            depth=depth,
            certified=bool(converged),
        )

    @classmethod
    def _positivity_lift_bernstein(cls, coeff, max_subdivide=8):
        """Return a constant lift certified by Bernstein lower bounds.

        On any interval a Bernstein-form polynomial lies in the convex hull
        of its Bernstein coefficients.  Recursively subdividing therefore
        gives a rigorous sufficient lower bound without sampling or root
        finding.  Adding a constant shifts every Bernstein coefficient by
        the same amount.

        Parameters
        ----------
        coeff : array_like
            Chebyshev coefficients of the polynomial to lift.
        max_subdivide : int, optional
            Maximum subdivision depth before accepting the current bound.

        Returns
        -------
        float
            Non-negative constant to add.  Zero when the polynomial is
            already certified non-negative.
        """
        lower = chebyshev_lower_bound(
            coeff, chebyshev_bernstein_matrix(len(coeff) - 1), int(max_subdivide)
        )
        if not math.isfinite(lower) or lower >= 0.0:
            return 0.0
        scale = max(float(np.max(np.abs(coeff))), 1.0)
        return -lower + 32.0 * _EPS * scale

    def _fit_best_panel(self, a, b, depth):
        """Fit one interval, p-refining before any spatial subdivision.

        Parameters
        ----------
        a, b : float
            Interval edges in compact coordinates.
        depth : int
            Current bisection depth.

        Returns
        -------
        ok : bool
            Whether one of the configured degrees met tolerance.
        panel : _Panel
            First certified panel, or the highest-degree attempt when none
            certified.
        """
        last_panel = None
        for n in self.degree_options:
            ok, panel = self._fit_panel(a, b, depth, n)
            last_panel = panel
            if ok:
                return True, panel
        return False, last_panel

    @staticmethod
    def _refinement_priority(panel):
        """Probability-scale priority for an unresolved CDF leaf.

        The interpolation residual and positivity lift already live in
        ``error_mass``.  The coefficient-tail gate is a separate acceptance
        criterion, so include its mass-scale counterpart as well.  Refining
        the largest unresolved probability contribution first prevents a
        difficult low-mass boundary sliver from starving the density body.

        Parameters
        ----------
        panel : _Panel
            Uncertified leaf panel.

        Returns
        -------
        float
            Non-negative refinement priority in transformed-mass units.
        """
        width = float(panel.b - panel.a)
        tail_abs = float(np.max(np.abs(panel.coeff[-min(4, panel.coeff.size) :])))
        priority = max(float(panel.error_mass), width * tail_abs)
        return priority if np.isfinite(priority) else math.inf

    def _build_partition(self, breaks):
        """Build a globally budgeted adaptive partition of compact support.

        Every current leaf is retained, whether certified or not, so the
        leaves always cover the complete support.  A split replaces one leaf
        by two and therefore increases the leaf count by exactly one.  This
        makes ``max_panels`` a strict resource bound because every pending
        branch is represented in the global leaf count.

        Among splittable uncertified leaves, the largest probability-scale
        residual is refined first.  The budget therefore degrades globally
        rather than allowing an early difficult region to consume all work
        before later parts of the support are examined.

        Parameters
        ----------
        breaks : numpy.ndarray
            Sorted compact-coordinate seed breaks including ``-1`` and ``1``.

        Returns
        -------
        list of _Panel
            Complete leaf partition, sorted later by the caller.
        """
        leaves = []
        for a, b in itertools.pairwise(breaks):
            ok, panel = self._fit_best_panel(float(a), float(b), 0)
            leaves.append((bool(ok), panel))

        self._panel_budget_exhausted = False
        min_width = 5e-13

        while True:
            candidates = [
                (self._refinement_priority(panel), j)
                for j, (ok, panel) in enumerate(leaves)
                if (
                    not ok
                    and panel.depth < self.max_depth
                    and (panel.b - panel.a) >= min_width
                )
            ]
            if not candidates:
                break
            if len(leaves) >= self.max_panels:
                self._panel_budget_exhausted = True
                break

            _, j = max(candidates, key=lambda item: (item[0], -item[1]))
            _, panel = leaves[j]
            mid = 0.5 * (panel.a + panel.b)
            left = self._fit_best_panel(panel.a, mid, panel.depth + 1)
            right = self._fit_best_panel(mid, panel.b, panel.depth + 1)
            leaves[j : j + 1] = [left, right]

        return [panel for _, panel in leaves]

    def _recertify_panel_masses(self, panels):
        """Replace uncertified panels' analytic masses with quadrature ones.

        Degradation at the depth, width or panel-budget limit accepts a
        polynomial that did not meet tolerance.  Its *shape* error is
        local to the panel, but its analytic integral feeds
        ``total_mass``, which scales every CDF value on every panel -- so
        without this the failure is global.  It is not a small effect: a
        single budget-starved panel has been observed integrating to 146
        against a true total of 1, crushing the whole CDF by that factor.

        The true mass over the panel is recovered by composite
        Gauss-Legendre quadrature of the transformed density, which is
        cheap (one rule per uncertified panel, and there are usually
        none) and does not depend on the failed fit.  The panel's
        antiderivative is then rescaled to match.  Scaling by a positive
        constant preserves the Bernstein-certified monotonicity, so the
        panel stays a valid CDF segment; only its interior shape remains
        approximate, which is the error that was actually measured.

        A panel whose polynomial integrates to a non-positive value
        cannot be rescaled, so it is replaced by the constant density
        carrying the same quadrature mass.

        Parameters
        ----------
        panels : list of _Panel
            Accepted panels, modified in place.

        Returns
        -------
        int
            Number of panels whose mass was replaced.
        """
        nodes, wts = np.polynomial.legendre.leggauss(_RECERTIFY_GAUSS_N)
        fixed = 0

        for j, panel in enumerate(panels):
            if panel.certified:
                continue

            half = 0.5 * (panel.b - panel.a)
            if not (half > 0.0):
                continue
            mid = 0.5 * (panel.a + panel.b)

            sub = np.linspace(-1.0, 1.0, _RECERTIFY_SUBPANELS + 1)
            total = 0.0
            for lo, hi in itertools.pairwise(sub):
                sh = 0.5 * (hi - lo)
                sc = 0.5 * (hi + lo)
                zz = mid + half * (sc + sh * nodes)
                vals = np.asarray(self._target(zz), dtype=np.float64)
                vals = np.where(np.isfinite(vals), vals, 0.0)
                total += float(np.dot(wts, vals)) * sh
            true_mass = total * half

            if not np.isfinite(true_mass) or true_mass < 0.0:
                true_mass = 0.0

            old = float(panel.mass)
            if old > 0.0 and np.isfinite(old):
                icoeff = panel.icoeff * (true_mass / old)
                coeff = panel.coeff * (true_mass / old)
            else:
                # Unusable polynomial: fall back to a flat panel of the
                # right mass, which is still a valid monotone segment.
                height = true_mass / (2.0 * half) if half > 0.0 else 0.0
                coeff = np.array([height], dtype=np.float64)
                icoeff = C.chebint(coeff) * half

            panels[j] = replace(panel, coeff=coeff, icoeff=icoeff, mass=true_mass)
            fixed += 1

        return fixed


class PythonSpectralPPFBuilder(SpectralPPF):
    """Spectral inverse built by the Python construction."""

    def _build_panels(self, intervals):
        """Build the inverse partition in Python.

        Parameters
        ----------
        intervals : list of tuple
            ``(source_j, ra, rb, pa, pb, za, zb)`` seed intervals.
        """
        return self._build_partition(intervals)

    def _source_local_density(self, j, u):
        """Derivative ``dG/dz`` of the stored spectral CDF panel.

        Parameters
        ----------
        j : int
            Source CDF panel index.
        u : array_like
            Panel-local coordinate in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Density in the compact coordinate.
        """
        sp = self.cdf_rep
        panel = sp.panels[j]
        dicoeff = self._dicoeff_cache.get(j)
        if dicoeff is None:
            dicoeff = np.asarray(
                _chebder_kernel(np.ascontiguousarray(panel.icoeff, dtype=np.float64)),
                dtype=np.float64,
            )
            self._dicoeff_cache[j] = dicoeff
        arr = np.asarray(u, dtype=np.float64)
        flat = np.ascontiguousarray(np.atleast_1d(arr).reshape(-1))
        d_du = (
            np.asarray(
                _chebval_many(flat, np.ascontiguousarray(dicoeff)), dtype=np.float64
            ).reshape(np.atleast_1d(arr).shape)
            / sp.total_mass
        )
        out = d_du * (2.0 / (panel.b - panel.a))
        return float(out[0]) if arr.ndim == 0 else out.reshape(arr.shape)

    def _invert_source_many(self, j, p):
        """Invert many probabilities known to lie within one source panel.

        Parameters
        ----------
        j : int
            Source CDF panel index.
        p : array_like
            Probabilities bracketed by that panel.

        Returns
        -------
        numpy.ndarray
            Compact coordinates with the same shape as *p*.
        """
        sp = self.cdf_rep
        arr = np.asarray(p, dtype=np.float64)
        pa = float(sp.cum_mass[j])
        pb = float(sp.cum_mass[j + 1])
        if not pb > pa:
            return np.full_like(arr, float(sp.breaks[j]))
        frac = np.clip((arr - pa) / (pb - pa), 0.0, 1.0)
        return np.asarray(
            sp._cython_evaluator.invert_panel_fraction(j, frac),
            dtype=np.float64,
        )

    @classmethod
    def _bernstein_lower_bound(cls, coeff, max_subdivide=10):
        """Rigorous lower bound on a Chebyshev polynomial over ``[-1, 1]``.

        A Bernstein-form polynomial lies in the convex hull of its
        coefficients, so recursive subdivision bounds it from below without
        sampling or root finding.  Used here on each inverse panel's
        derivative to certify monotonicity.  The de Casteljau split is
        shared with :class:`gibbus._spectral.cdf.SpectralCDF` rather than
        duplicated.

        Parameters
        ----------
        coeff : numpy.ndarray
            Chebyshev coefficients of the polynomial to bound.
        max_subdivide : int, optional
            Maximum subdivision depth before accepting the current bound.

        Returns
        -------
        float
            Certified lower bound, or ``0.0`` for an empty polynomial.
        """
        coeff = np.asarray(coeff, dtype=np.float64)
        if coeff.size == 0:
            return 0.0
        return float(
            chebyshev_lower_bound(
                coeff, chebyshev_bernstein_matrix(len(coeff) - 1), int(max_subdivide)
            )
        )

    @staticmethod
    def _r_from_u(u, ra, rb):
        """Map panel-local ``u`` to the log-odds coordinate.

        Parameters
        ----------
        u : array_like
            Panel-local coordinates in ``[-1, 1]``.
        ra, rb : float
            Panel edges in log-odds.

        Returns
        -------
        numpy.ndarray
            Log-odds values.
        """
        return 0.5 * ((rb - ra) * np.asarray(u) + (ra + rb))

    @staticmethod
    def _v_from_z(z, za, zb):
        """Map compact ``z`` to panel-local ``v`` in ``[-1, 1]``.

        Parameters
        ----------
        z : array_like
            Compact coordinates.
        za, zb : float
            Panel edges in compact coordinates.

        Returns
        -------
        numpy.ndarray
            Panel-local coordinates.
        """
        return (2.0 * np.asarray(z) - (za + zb)) / (zb - za)

    def _exact_z_for_r(self, source_j, r):
        """Exact compact coordinate for a log-odds value.

        Parameters
        ----------
        source_j : int
            Source CDF panel expected to bracket the probability.
        r : float
            Log-odds value.

        Returns
        -------
        float
            Compact coordinate ``z``.
        """
        p = float(expit(float(r)))
        # Stay inside the representable PPF domain at the extreme endpoint.
        p = min(max(p, float(_PMIN)), float(_PMAX))
        # Usually the interval stays within source_j; if rounded source masses
        # disagree at an extreme boundary, the global inversion is safer.
        cm = self.cdf_rep.cum_mass
        if (
            float(cm[source_j]) <= p <= float(cm[source_j + 1])
            and cm[source_j + 1] > cm[source_j]
        ):
            return self._invert_source(source_j, p)
        return self._invert_global(p)

    def _exact_z_for_r_many(self, source_j, r):
        """Invert many log-odds values, batching the common source panel.

        Parameters
        ----------
        source_j : int
            Expected source CDF panel containing the probabilities.
        r : array_like
            Log-odds values.

        Returns
        -------
        numpy.ndarray
            Compact coordinates with the same shape as *r*.
        """
        rr = np.asarray(r, dtype=np.float64)
        probs = np.clip(expit(rr), float(_PMIN), float(_PMAX))
        out = np.empty_like(probs)
        cm = self.cdf_rep.cum_mass
        inside = (
            (probs >= float(cm[source_j]))
            & (probs <= float(cm[source_j + 1]))
            & bool(cm[source_j + 1] > cm[source_j])
        )
        if np.any(inside):
            out[inside] = self._invert_source_many(source_j, probs[inside])
        if np.any(~inside):
            out[~inside] = np.array(
                [self._invert_global(float(p)) for p in probs[~inside]],
                dtype=np.float64,
            )
        return out

    def _fit_panel(self, source_j, ra, rb, pa, pb, za, zb, depth, degree):
        """Fit one inverse panel and certify it monotone.

        The inverse is represented by approximating its positive
        derivative at degree ``D - 1`` and integrating once, so
        monotonicity follows from a Bernstein lower bound on the
        derivative rather than from sampling the inverse itself.

        Parameters
        ----------
        source_j : int
            Source CDF panel this interval draws from.
        ra, rb : float
            Panel edges in the log-odds coordinate.
        pa, pb : float
            The same edges as probabilities.
        za, zb : float
            Corresponding compact coordinates.
        depth : int
            Current refinement depth.
        degree : int
            Final polynomial degree to attempt.

        Returns
        -------
        ok : bool
            Whether the panel met all tolerances and certified monotone.
        panel : _InversePanel
            The fitted panel, returned whether or not it passed.
        """
        # Store a final inverse polynomial of degree D by approximating its
        # positive derivative with degree D-1, then integrating once.
        final_degree = int(degree)
        n = final_degree - 1
        u = lobatto_nodes(n)
        r = self._r_from_u(u, ra, rb)
        probs = expit(r)
        z = self._exact_z_for_r_many(source_j, r)

        # dv/du = ((rb-ra)/(zb-za)) * dz/dr, with
        # dz/dr = p(1-p)/(dG/dz).  Use the derivative of the stored spectral
        # CDF so the inverse representation is internally consistent with G.
        src = self.cdf_rep.panels[source_j]
        usrc = (2.0 * z - (src.a + src.b)) / (src.b - src.a)
        dens = np.asarray(self._source_local_density(source_j, usrc), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            qvals = ((rb - ra) / (zb - za)) * probs * (1.0 - probs) / dens
        if not np.all(np.isfinite(qvals)) or np.any(qvals < 0.0):
            # Rare fallback to the exact transformed fitted density.
            dens = np.asarray(self.cdf_rep.transformed_pdf(z), dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                qvals = ((rb - ra) / (zb - za)) * probs * (1.0 - probs) / dens

        if not np.all(np.isfinite(qvals)) or np.any(qvals < 0.0):
            return False, _InversePanel(
                ra=float(ra),
                rb=float(rb),
                pa=float(pa),
                pb=float(pb),
                za=float(za),
                zb=float(zb),
                coeff=np.array([-1.0, 1.0]),
                fit_error=math.inf,
                logit_residual=math.inf,
                prob_residual=math.inf,
                tail_abs=math.inf,
                derivative_lower=-math.inf,
                depth=int(depth),
                source_panel=int(source_j),
            )

        qcoeff = _lobatto_coefficients_kernel(
            np.ascontiguousarray(qvals, dtype=np.float64),
            np.ascontiguousarray(_lobatto_transform(n), dtype=np.float64),
        )
        lift = PythonSpectralCDFBuilder._positivity_lift_bernstein(
            qcoeff, max_subdivide=self.certify_subdivide
        )
        if lift > 0.0:
            qcoeff = np.asarray(qcoeff, dtype=np.float64).copy()
            qcoeff[0] += lift

        # Integrate the positive derivative and normalize its total increment
        # to exactly 2, giving v(-1)=-1 and v(1)=1 by construction.
        icoeff, imass = _chebint_scaled(
            np.ascontiguousarray(qcoeff, dtype=np.float64), 1.0
        )
        ileft = float(
            _chebval_many(
                np.array([-1.0], dtype=np.float64),
                np.ascontiguousarray(icoeff, dtype=np.float64),
            )[0]
        )
        imass = float(imass)
        if not np.isfinite(imass) or imass <= 0.0:
            return False, _InversePanel(
                ra=float(ra),
                rb=float(rb),
                pa=float(pa),
                pb=float(pb),
                za=float(za),
                zb=float(zb),
                coeff=np.array([-1.0, 1.0]),
                fit_error=math.inf,
                logit_residual=math.inf,
                prob_residual=math.inf,
                tail_abs=math.inf,
                derivative_lower=-math.inf,
                depth=int(depth),
                source_panel=int(source_j),
            )
        coeff = np.asarray(icoeff, dtype=np.float64) * (2.0 / imass)
        coeff[0] += -1.0 - (2.0 * ileft / imass)

        # Interlaced validation against accurate root inversions.
        m = 2 * final_degree + 5
        uv = midpoint_nodes(m)
        rv = self._r_from_u(uv, ra, rb)
        pv = expit(rv)
        ztrue = self._exact_z_for_r_many(source_j, rv)
        vtrue = self._v_from_z(ztrue, za, zb)
        vpred = _chebval_many(
            np.ascontiguousarray(uv, dtype=np.float64),
            np.ascontiguousarray(coeff, dtype=np.float64),
        )
        zpred = self._z_from_v(vpred, za, zb)

        fit_error = float(np.max(np.abs(vpred - vtrue)))
        pback = np.asarray(
            self.cdf_rep._cython_evaluator.eval_compact(zpred), dtype=np.float64
        )
        prob_resid = float(np.max(np.abs(pback - pv)))
        pback_safe = np.clip(pback, float(_PMIN), float(_PMAX))
        logit_resid = float(np.max(np.abs(_logit(pback_safe) - rv)))
        tail_abs = float(np.max(np.abs(coeff[-min(4, coeff.size) :])))

        dcoeff = _chebder_kernel(np.ascontiguousarray(coeff, dtype=np.float64))
        derivative_lower = self._bernstein_lower_bound(
            dcoeff, max_subdivide=self.certify_subdivide
        )
        # This should be nonnegative by construction; retain the independent
        # certificate as a construction sanity check.
        monotone = derivative_lower >= -128.0 * _EPS

        # In the tails, inverse conditioning makes normalized geometric v
        # error a poor stopping metric.  Judge the inverse by composition in
        # probability space, against an absolute floor no tighter than
        # float64 can resolve there.  Positivity of the fitted
        # derivative prevents hidden non-monotone oscillations, so coefficient
        # tail size and geometric v error remain diagnostics rather than hard
        # acceptance gates.
        ulp_p = np.abs(np.spacing(pv))
        prob_floor = float(64.0 * np.max(ulp_p))
        # Acceptance is based on the quantity this object actually inverts:
        # the stored spectral CDF in probability space.  A relative/logit
        # requirement becomes ill-conditioned near p=0 or p=1 and can force
        # meaningless subdivision below the representable z spacing even when
        # the absolute inverse residual is already at the float64 floor.
        ok = prob_resid <= max(self.prob_tol, prob_floor) and monotone
        return ok, _InversePanel(
            ra=float(ra),
            rb=float(rb),
            pa=float(pa),
            pb=float(pb),
            za=float(za),
            zb=float(zb),
            coeff=np.asarray(coeff, dtype=np.float64),
            fit_error=fit_error,
            logit_residual=logit_resid,
            prob_residual=prob_resid,
            tail_abs=tail_abs,
            derivative_lower=derivative_lower,
            depth=int(depth),
            source_panel=int(source_j),
        )

    def _fit_best_panel(self, source_j, ra, rb, pa, pb, za, zb, depth):
        """Fit one inverse interval, p-refining before subdivision.

        Parameters
        ----------
        source_j : int
            Source CDF panel index.
        ra, rb : float
            Log-odds interval edges.
        pa, pb : float
            Probability interval edges.
        za, zb : float
            Corresponding compact-coordinate edges.
        depth : int
            Current bisection depth.

        Returns
        -------
        ok : bool
            Whether one configured degree met the inverse certification.
        panel : _InversePanel
            First certified panel, or the highest-degree attempt otherwise.
        """
        last = None
        for degree in self.degree_options:
            ok, panel = self._fit_panel(source_j, ra, rb, pa, pb, za, zb, depth, degree)
            last = panel
            if ok:
                return True, panel
        return False, last

    @staticmethod
    def _refinement_priority(panel):
        """Return the failed inverse panel's probability-scale priority.

        Parameters
        ----------
        panel : _InversePanel
            Uncertified inverse leaf.

        Returns
        -------
        float
            Priority, with non-finite or non-monotone attempts ranked first.
        """
        if (
            not np.isfinite(panel.prob_residual)
            or not np.isfinite(panel.derivative_lower)
            or panel.derivative_lower < -128.0 * _EPS
        ):
            return math.inf
        return max(float(panel.prob_residual), 0.0)

    def _build_partition(self, intervals):
        """Build the inverse with a strict global leaf budget.

        Failed candidate leaves remain in the frontier until they certify or
        are split.  A split replaces one leaf by two, increasing the frontier
        size by exactly one, so ``max_panels`` is a deterministic resource
        bound rather than a count of only the leaves that happened to certify
        before a depth-first walk reached them.  The largest probability
        residual is refined first.

        Unlike the forward CDF, the inverse may not degrade by retaining an
        uncertified leaf: monotonicity is part of its contract.  Exhausting
        either the leaf budget or the depth/spacing limit therefore raises and
        lets the caller use spectral-CDF bisection instead.

        Parameters
        ----------
        intervals : sequence of tuple
            ``(source_j, ra, rb, pa, pb, za, zb)`` seed intervals.

        Returns
        -------
        list of _InversePanel
            Fully certified inverse panels.
        """
        leaves = [
            self._fit_best_panel(source_j, ra, rb, pa, pb, za, zb, 0)
            for source_j, ra, rb, pa, pb, za, zb in intervals
        ]

        while True:
            failed = [
                (self._refinement_priority(panel), j)
                for j, (ok, panel) in enumerate(leaves)
                if not ok
            ]
            if not failed:
                return [panel for _, panel in leaves]

            _, j = max(failed, key=lambda item: (item[0], -item[1]))
            _, panel = leaves[j]

            if panel.depth >= self.max_depth or not (
                panel.rb > np.nextafter(panel.ra, np.inf)
            ):
                raise RuntimeError(self._failure_message(panel))

            if len(leaves) >= self.max_panels:
                raise RuntimeError(
                    f"spectral PPF exhausted its {self.max_panels}-panel "
                    f"budget with {len(failed)} uncertified leaf interval(s); "
                    "the inverse could not be certified monotone within budget"
                )

            rm = 0.5 * (panel.ra + panel.rb)
            pm = float(expit(rm))
            pm = min(max(pm, float(_PMIN)), float(_PMAX))
            zm = self._exact_z_for_r(panel.source_panel, rm)
            depth = panel.depth + 1
            left = self._fit_best_panel(
                panel.source_panel, panel.ra, rm, panel.pa, pm, panel.za, zm, depth
            )
            right = self._fit_best_panel(
                panel.source_panel, rm, panel.rb, pm, panel.pb, zm, panel.zb, depth
            )
            leaves[j : j + 1] = [left, right]
