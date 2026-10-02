"""Adaptive spectral CDF construction for fitted density components.

The physical support is mapped monotonically to ``z in [-1, 1]``.  The
transformed density ``g(z) = f(x(z)) dx/dz`` is approximated on adaptive
Chebyshev panels with local degree chosen from a small discrete set.  Each
panel polynomial is certified non-negative with Bernstein lower bounds,
minimally lifted if necessary, and analytically integrated.  The normalized
piecewise antiderivative is therefore a valid monotone CDF by construction.

Construction runs in the compiled builder (:mod:`._builders`) from a data
description of the density (:func:`density_spec`).  Query-time state is packed
into the fitted structured scalar and evaluated by :mod:`._cdf_eval`.  The
tests keep a pure-Python construction harness for builder-control-flow and
bitwise-orchestration tests (``tests/spectral_builder_harness.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import chebyshev as C

from .._defaults import (
    SPECTRAL_CDF_ABS_TOL,
    SPECTRAL_CDF_COEFF_TOL,
    SPECTRAL_CDF_MAX_DEPTH,
    SPECTRAL_CDF_MAX_PANELS,
    SPECTRAL_CDF_REL_TOL,
    SPECTRAL_DEGREE_OPTIONS,
)
from . import _builders  # type: ignore[attr-defined]  # compiled extension
from ._cdf_eval import SpectralEvaluator
from .chebyshev import (
    _lobatto_transform,
    chebyshev_bernstein_matrix,
    lobatto_nodes,
    midpoint_nodes,
)

_RECERTIFY_GAUSS_N = 24
"""Gauss-Legendre nodes per subinterval when the builder re-measures the mass
of a panel it could not certify."""

_KIND_CODE = {"finite": 0, "lower": 1, "upper": 2, "real": 3, "lower_centered": 4, "upper_centered": 5}

_TABLES: dict = {}


def _builder_tables(cdf_degrees, ppf_degrees=(), /):
    """Return the compiled builders' view of the cached reference tables.

    Parameters
    ----------
    cdf_degrees : tuple of int
        Forward degree options (their Lobatto/validation/Bernstein tables, plus
        the degree-20 scale-selection tables).
    ppf_degrees : tuple of int, optional
        Inverse degree options.
    """
    key = (tuple(cdf_degrees), tuple(ppf_degrees))
    tables = _TABLES.get(key)
    if tables is not None:
        return tables
    lob, trans, ratios, mids = {}, {}, {}, {}
    trial = min(20, max(cdf_degrees)) if cdf_degrees else None
    for n in list(cdf_degrees) + ([trial] if trial else []):
        lob[n] = lobatto_nodes(n)
        trans[n] = _lobatto_transform(n)
        ratios[n] = chebyshev_bernstein_matrix(n)
        mids[2 * n + 3] = midpoint_nodes(2 * n + 3)
    for degree in ppf_degrees:
        n = degree - 1
        lob[n] = lobatto_nodes(n)
        trans[n] = _lobatto_transform(n)
        ratios[n] = chebyshev_bernstein_matrix(n)
        mids[2 * degree + 5] = midpoint_nodes(2 * degree + 5)
    nodes, weights = np.polynomial.legendre.leggauss(_RECERTIFY_GAUSS_N)
    tables = _builders.Tables(lob, trans, mids, ratios, nodes, weights)
    _TABLES[key] = tables
    return tables


def density_spec(components, /, *, view):
    """Describe a density for the compiled builders.

    Parameters
    ----------
    components : sequence of tuple
        Per component ``(q_poly, Lz, Uz, aL, aU, log_norm, mu, sigma, jf,
        weight, lo, hi)``: the canonical kernel of ``_pdf_vec`` at
        ``z = sigma x + mu``, times ``jf * weight``, masked to ``[lo, hi]``
        when ``view``.
    view : bool
        Whether the components are public-coordinate views.

    Returns
    -------
    _builders.DensitySpec
    """
    return _builders.DensitySpec(list(components), bool(view))


_REMOTE_BOUNDARY_RATIO = 10.0
"""Distance/std ratio above which a half-line map is centered on the mode.

The endpoint-anchored rational map spends its resolution at the endpoint: a
body ``k`` standard deviations away lands in a compact interval of width
``O(1/k^2)``, which a degree-32 panel can straddle with no node inside it.
That happened for a fitted mixture component at ``k = 29`` (every panel saw
zero density and the construction failed with zero mass).  Beyond ten
standard deviations the endpoint region carries no appreciable mass, and the
centered variant preserves the exact finite endpoint while using real-line
resolution through the body of the density.
"""


@dataclass(frozen=True)
class _Map:
    kind: str
    L: float
    U: float
    center: float
    scale: float

    def x_from_z(self, z):
        """Map compact coordinates to physical ones.

        Parameters
        ----------
        z : array_like
            Compact coordinates in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Physical coordinates.  Exact finite endpoints are preserved
            rather than recomputed, since ``center +/- scale`` can round
            just outside the support and create a spurious jump.
        """
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "finite":
            x = self.center + self.scale * z
            # Preserve exact finite support endpoints; roundoff in center +/-
            # scale can otherwise place a Chebyshev endpoint just outside the
            # support and create an artificial jump in the sampled PDF.
            x = np.where(z <= -1.0, self.L, x)
            x = np.where(z >= 1.0, self.U, x)
            return x
        if self.kind == "lower":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return self.L + self.scale * (1.0 + z) / (1.0 - z)
        if self.kind == "upper":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return self.U - self.scale * (1.0 - z) / (1.0 + z)
        if self.kind == "real":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return self.center + (2.0 * self.scale * z) / (1.0 - z * z)
        if self.kind in ("lower_centered", "upper_centered"):
            if self.kind == "lower_centered":
                y_edge = (self.L - self.center) / self.scale
                t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
                t = t_edge + 0.5 * (z + 1.0) * (1.0 - t_edge)
            else:
                y_edge = (self.U - self.center) / self.scale
                t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
                t = -1.0 + 0.5 * (z + 1.0) * (t_edge + 1.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                x = self.center + (2.0 * self.scale * t) / (1.0 - t * t)
            if self.kind == "lower_centered":
                x = np.where(z <= -1.0, self.L, x)
            else:
                x = np.where(z >= 1.0, self.U, x)
            return x
        raise RuntimeError("unknown map kind")

    def jac(self, z):
        """Return ``dx/dz`` for the density change of variables.

        Parameters
        ----------
        z : array_like
            Compact coordinates in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            Jacobian values.
        """
        z = np.asarray(z, dtype=np.float64)
        if self.kind == "finite":
            return np.full_like(z, self.scale)
        if self.kind == "lower":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return 2.0 * self.scale / (1.0 - z) ** 2
        if self.kind == "upper":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return 2.0 * self.scale / (1.0 + z) ** 2
        if self.kind == "real":
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return 2.0 * self.scale * (1.0 + z * z) / (1.0 - z * z) ** 2
        if self.kind in ("lower_centered", "upper_centered"):
            if self.kind == "lower_centered":
                y_edge = (self.L - self.center) / self.scale
                t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
                dt_dz = 0.5 * (1.0 - t_edge)
                t = t_edge + (z + 1.0) * dt_dz
            else:
                y_edge = (self.U - self.center) / self.scale
                t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
                dt_dz = 0.5 * (t_edge + 1.0)
                t = -1.0 + (z + 1.0) * dt_dz
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return (
                    2.0 * self.scale * (1.0 + t * t)
                    / (1.0 - t * t) ** 2
                ) * dt_dz
        raise RuntimeError("unknown map kind")

    def z_from_x(self, x):
        """Map physical coordinates back to compact ones.

        Parameters
        ----------
        x : array_like
            Physical coordinates.

        Returns
        -------
        numpy.ndarray
            Compact coordinates in ``[-1, 1]``.
        """
        x = np.asarray(x, dtype=np.float64)
        if self.kind == "finite":
            return (x - self.center) / self.scale
        if self.kind == "lower":
            y = (x - self.L) / self.scale
            with np.errstate(divide="ignore", invalid="ignore"):
                return (y - 1.0) / (y + 1.0)
        if self.kind == "upper":
            y = (self.U - x) / self.scale
            with np.errstate(divide="ignore", invalid="ignore"):
                return (1.0 - y) / (1.0 + y)
        if self.kind == "real":
            y = (x - self.center) / self.scale
            # Stable inverse of y = 2 z / (1-z^2).
            return y / (1.0 + np.hypot(1.0, y))
        if self.kind in ("lower_centered", "upper_centered"):
            y = (x - self.center) / self.scale
            t = y / (1.0 + np.hypot(1.0, y))
            if self.kind == "lower_centered":
                y_edge = (self.L - self.center) / self.scale
                t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
                return 2.0 * (t - t_edge) / (1.0 - t_edge) - 1.0
            y_edge = (self.U - self.center) / self.scale
            t_edge = y_edge / (1.0 + np.hypot(1.0, y_edge))
            return 2.0 * (t + 1.0) / (t_edge + 1.0) - 1.0
        raise RuntimeError("unknown map kind")


@dataclass
class _Panel:
    a: float
    b: float
    coeff: np.ndarray
    icoeff: np.ndarray
    mass: float
    fit_error: float
    error_mass: float
    tail_ratio: float
    lift: float
    depth: int
    certified: bool = True
    """Whether this panel met the error tolerances.

    ``False`` marks a panel accepted at the depth, width or budget limit
    rather than on merit.  Its polynomial is not a trustworthy
    approximation, so :meth:`SpectralCDF._recertify_panel_masses` refuses
    to use its analytic integral in the global normalizer.
    """

    @property
    def width(self):
        return self.b - self.a

    @property
    def degree(self):
        # icoeff is one order higher after integration; coeff is the local PDF.
        return int(self.coeff.size - 1)


class SpectralCDF:
    """Adaptive compactified piecewise-Chebyshev CDF representation."""

    def __init__(
        self,
        support,
        *,
        density,
        mode=None,
        std=None,
        degree_options=SPECTRAL_DEGREE_OPTIONS,
        rel_tol=SPECTRAL_CDF_REL_TOL,
        abs_tol=SPECTRAL_CDF_ABS_TOL,
        coeff_tol=SPECTRAL_CDF_COEFF_TOL,
        max_depth=SPECTRAL_CDF_MAX_DEPTH,
        max_panels=SPECTRAL_CDF_MAX_PANELS,
        map_scale=None,
        initial_breaks=None,
    ):
        """Build an adaptive spectral CDF for a density described by data.

        Parameters
        ----------
        support : sequence of (float, float)
            Density support; either endpoint may be infinite.
        density : _builders.DensitySpec
            Data description of the density (see :func:`density_spec`); the
            whole construction runs in the compiled builder.
        mode : float or None, optional
            Density mode, used to center a doubly-infinite map.
        std : float or None, optional
            Scale proxy used to choose the map scale.
        degree_options : sequence of int or None, optional
            Degrees tried in order before an interval is bisected.
        rel_tol, abs_tol : float, optional
            Relative and absolute panel error tolerances.
        coeff_tol : float, optional
            Relative Chebyshev coefficient-tail tolerance.
        max_depth : int, optional
            Maximum bisection depth.  On reaching it the best panel so
            far is accepted rather than raising.
        max_panels : int, optional
            Strict leaf budget for the complete adaptive partition.  Depth
            alone does not bound the work: a density that misses tolerance
            across a wide region can bisect a full tree.  When the leaf
            budget is full, unresolved leaves are retained as-is and exposed
            through the spectral diagnostics.
        map_scale : float or None, optional
            Explicit map scale, bypassing automatic selection.
        initial_breaks : array_like or None, optional
            Seed breaks in compact coordinates, typically from
            :func:`boundary_aware_breaks_from_amplitudes`.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If *density* is not a density description.
        """
        if not isinstance(density, _builders.DensitySpec):
            raise TypeError("density must be a DensitySpec (see density_spec)")
        breaks, s0 = self._configure(
            support, mode, std, degree_options, rel_tol, abs_tol, coeff_tol,
            max_depth, max_panels, map_scale, initial_breaks, density,
        )
        panels = self._build_compiled(
            breaks, choose=map_scale is None and self.map.kind != "finite", s0=s0
        )
        self._finish(panels, recertified=self.n_masses_recertified)

    def _configure(self, support, mode, std, degree_options, rel_tol, abs_tol,
                   coeff_tol, max_depth, max_panels, map_scale, initial_breaks,
                   density, /):
        """Store the options, choose the compactifying map and seed the breaks.

        Parameters
        ----------
        support : sequence of (float, float)
            Density support.
        mode, std : float or None
            Mode and scale proxy for the map choice.
        degree_options : sequence of int
            Panel degrees.
        rel_tol, abs_tol, coeff_tol : float
            Panel tolerances.
        max_depth, max_panels : int
            Refinement limits.
        map_scale : float or None
            Explicit map scale.
        initial_breaks : array_like or None
            Seed breaks in compact coordinates.
        density : DensitySpec or None
            Density description (``None`` only for the test harness).

        Returns
        -------
        breaks : numpy.ndarray
            Seed breaks in compact coordinates, including ``-1`` and ``1``.
        s0 : float
            Baseline scale for automatic map-scale selection.
        """
        self.support = np.asarray(support, dtype=np.float64)
        opts = tuple(sorted({int(v) for v in degree_options}))
        if not opts or opts[0] < 2:
            raise ValueError("degree_options must contain degrees >= 2")
        self.degree_options = opts
        self.degree = max(self.degree_options)
        self.rel_tol = float(rel_tol)
        self.abs_tol = float(abs_tol)
        self.coeff_tol = float(coeff_tol)
        self.max_depth = int(max_depth)
        self.max_panels = int(max_panels)
        self.density = density

        s0 = float(std) if std is not None and np.isfinite(std) and std > 0 else 1.0
        L, U = map(float, self.support)
        if np.isfinite(L) and np.isfinite(U):
            center = 0.5 * (L + U)
            scale = 0.5 * (U - L)
            self.map = _Map("finite", L, U, center, scale)
        else:
            c = float(mode) if mode is not None and np.isfinite(mode) else 0.0
            if np.isfinite(L):
                remote = np.isfinite(c) and c > L and (c - L) > _REMOTE_BOUNDARY_RATIO * s0
                kind = "lower_centered" if remote else "lower"
            elif np.isfinite(U):
                remote = np.isfinite(c) and c < U and (U - c) > _REMOTE_BOUNDARY_RATIO * s0
                kind = "upper_centered" if remote else "upper"
            else:
                kind = "real"
            if map_scale is None:
                map_scale = self._default_map_scale(L, U, mode, s0, kind)
            s = float(map_scale) if map_scale is not None else float("nan")
            if kind == "lower":
                self.map = _Map(kind, L, U, L, s)
            elif kind == "upper":
                self.map = _Map(kind, L, U, U, s)
            else:
                self.map = _Map(kind, L, U, c, s)

        br = [-1.0, 1.0]
        if initial_breaks is not None:
            br.extend(float(v) for v in initial_breaks if -1.0 < float(v) < 1.0)
        br = np.array(sorted(set(br)), dtype=np.float64)
        n_initial = int(br.size - 1)
        if n_initial > self.max_panels:
            raise ValueError(
                f"max_panels={self.max_panels} cannot cover {n_initial} "
                "initial spectral intervals"
            )
        return br, s0

    def _default_map_scale(self, L, U, mode, s0, kind, /):
        """Map scale when none is given: ``nan``, chosen in the compiled builder.

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
        return float("nan")

    def _finish(self, panels, *, recertified):
        """Normalize the leaf masses and pack the evaluator.

        Parameters
        ----------
        panels : list of _Panel
            Leaf partition, sorted, masses already re-measured.
        recertified : int
            Number of re-measured panel masses.
        """
        self.n_masses_recertified = int(recertified)
        self.panels = panels

        masses = np.array([p.mass for p in panels], dtype=np.float64)
        total = float(np.sum(masses))
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(f"invalid spectral mass {total}")
        self.total_mass = total
        self.panel_masses = masses / total
        self.cum_mass = np.concatenate(([0.0], np.cumsum(self.panel_masses)))
        self.cum_mass[-1] = 1.0
        self.breaks = np.array([panels[0].a] + [p.b for p in panels], dtype=np.float64)
        self._cython_evaluator = self._build_cython_evaluator()

    def _build_compiled(self, breaks, *, choose, s0):
        """Run the compiled builder and materialize its panels.

        Parameters
        ----------
        breaks : numpy.ndarray
            Seed breaks including -1 and 1.
        choose : bool
            Select the map scale in the builder.
        s0 : float
            Baseline scale for the selection.

        Returns
        -------
        list of _Panel
        """
        mp = self.map
        scale, rows, ints, coeff, icoeff, exhausted, fixed = _builders.build_cdf(
            self.density,
            _builder_tables(self.degree_options),
            _KIND_CODE[mp.kind], float(mp.L), float(mp.U), float(mp.center),
            float(mp.scale) if not choose else 1.0, bool(choose), float(s0),
            np.ascontiguousarray(breaks, dtype=np.float64),
            np.asarray(self.degree_options, dtype=np.int32),
            self.rel_tol, self.abs_tol, self.coeff_tol, self.max_depth, self.max_panels,
        )
        if choose:
            self.map = replace(mp, scale=float(scale))
        self._panel_budget_exhausted = bool(exhausted)
        self.n_masses_recertified = int(fixed)
        panels = []
        for j in range(rows.shape[0]):
            nc = int(ints[j, 2])
            panels.append(_Panel(
                a=float(rows[j, 0]), b=float(rows[j, 1]), coeff=coeff[j, :nc].copy(),
                icoeff=icoeff[j, :nc + 1].copy(), mass=float(rows[j, 2]),
                fit_error=float(rows[j, 3]), error_mass=float(rows[j, 4]),
                tail_ratio=float(rows[j, 5]), lift=float(rows[j, 6]),
                depth=int(ints[j, 0]), certified=bool(ints[j, 1]),
            ))
        return panels

    # ------------------------------------------------------------------
    # Mapping / transformed density
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------









    def _build_cython_evaluator(self):
        """Pack the immutable query-time representation for the Cython kernel."""
        kind_map = {"finite": 0, "lower": 1, "upper": 2, "real": 3, "lower_centered": 4, "upper_centered": 5}
        m = len(self.panels)
        # D <= 32 implies integrated series length <= 34. Keep the storage
        # generic so construction can still experiment with other degrees.
        max_ncoeff = max(p.icoeff.size for p in self.panels)
        coeffs = np.zeros((m, max_ncoeff), dtype=np.float64)
        ncoeff = np.empty(m, dtype=np.int32)
        offsets = np.asarray(self.cum_mass[:-1], dtype=np.float64).copy()

        for j, p in enumerate(self.panels):
            # Shift each local antiderivative so A_j(-1)=0, then bake the
            # global normalization into the coefficients.
            ic = np.asarray(p.icoeff, dtype=np.float64).copy()
            ic[0] -= C.chebval(-1.0, ic)
            ic /= self.total_mass
            coeffs[j, :ic.size] = ic
            ncoeff[j] = ic.size

        self._packed = (offsets, coeffs, ncoeff)
        return SpectralEvaluator(
            kind_map[self.map.kind], float(self.map.L), float(self.map.U),
            float(self.map.center), float(self.map.scale), self.breaks,
            offsets, coeffs, ncoeff,
        )

    def cdf_cython(self, x):
        """Evaluate the spectral CDF with the compiled bulk kernel.

        Parameters
        ----------
        x : array_like
            Physical coordinates.

        Returns
        -------
        numpy.ndarray
            CDF values in ``[0, 1]``.
        """
        return self._cython_evaluator(x)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def cdf_z(self, z):
        """Evaluate the normalized CDF at compact coordinates.

        Parameters
        ----------
        z : array_like
            Compact coordinates in ``[-1, 1]``.

        Returns
        -------
        numpy.ndarray
            CDF values in ``[0, 1]``.
        """
        arr = np.asarray(z, dtype=np.float64)
        scalar = arr.ndim == 0
        flat = np.atleast_1d(arr).reshape(-1)
        out = np.empty_like(flat)
        out[flat <= -1.0] = 0.0
        out[flat >= 1.0] = 1.0
        midmask = (flat > -1.0) & (flat < 1.0)
        if np.any(midmask):
            zz = flat[midmask]
            idx = np.searchsorted(self.breaks, zz, side="right") - 1
            idx = np.clip(idx, 0, len(self.panels) - 1)
            yy = np.empty_like(zz)
            for j in np.unique(idx):
                m = idx == j
                p = self.panels[int(j)]
                u = (2.0 * zz[m] - (p.a + p.b)) / (p.b - p.a)
                local = C.chebval(u, p.icoeff) - C.chebval(-1.0, p.icoeff)
                yy[m] = self.cum_mass[j] + local / self.total_mass
            # Roundoff guard only; monotonicity comes from panel density.
            out[midmask] = np.clip(yy, 0.0, 1.0)
        if scalar:
            return float(out[0])
        return out.reshape(arr.shape)

    def cdf(self, x):
        """Evaluate the normalized CDF at physical coordinates.

        Parameters
        ----------
        x : array_like
            Physical coordinates.

        Returns
        -------
        numpy.ndarray
            CDF values in ``[0, 1]``.
        """
        arr = np.asarray(x, dtype=np.float64)
        scalar = arr.ndim == 0
        L, U = self.support
        z = self.map.z_from_x(arr)
        out = np.asarray(self.cdf_z(z), dtype=np.float64)
        if np.isfinite(L):
            out = np.where(arr <= L, 0.0, out)
        if np.isfinite(U):
            out = np.where(arr >= U, 1.0, out)
        return float(out) if scalar else out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def max_depth_used(self):
        """Deepest refinement level reached by any accepted panel.

        Equal to ``max_depth`` means at least one interval was accepted
        without meeting tolerance, so the representation is the best
        available rather than a converged one.
        """
        return int(max(p.depth for p in self.panels)) if self.panels else 0

    @property
    def panel_budget_exhausted(self):
        """Whether unresolved refinement stopped because the budget ran out."""
        return bool(getattr(self, "_panel_budget_exhausted", False))

    @property
    def worst_panel_error(self):
        """Largest estimated local CDF error in probability units.

        The raw Chebyshev validation residual is a pointwise density error
        in transformed coordinates, so it is not directly comparable with
        a CDF error.  Each panel stores that residual after multiplication
        by panel width and inclusion of any positivity lift; dividing by
        total mass makes the diagnostic dimensionless.

        Returns
        -------
        float
            Largest local probability-scale error estimate.
        """
        if not self.panels or not (self.total_mass > 0.0):
            return 0.0
        return max(
            (float(p.error_mass) / self.total_mass
             for p in self.panels if p.certified),
            default=0.0,
        )

    @property
    def cdf_error_estimate(self):
        """Conservative overall CDF error estimate in probability units.

        Certified panels contribute the largest local interpolation error;
        uncertified panels contribute their full probability mass because no
        stronger shape guarantee is available.  ``mass_defect`` covers global
        normalization loss.  The three terms describe different mechanisms,
        so they are added and clipped to the probability range.

        This is a health estimate, not a rigorous mathematical bound: panel
        validation samples a finite set of points.

        Returns
        -------
        float
            Estimated absolute CDF error in ``[0, 1]``.
        """
        if not self.panels or not (self.total_mass > 0.0):
            return 0.0
        certified_local = max(
            (float(p.error_mass) / self.total_mass
             for p in self.panels if p.certified),
            default=0.0,
        )
        estimate = (self.mass_defect + self.uncertified_mass_fraction
                    + certified_local)
        return float(min(max(estimate, 0.0), 1.0))

    @property
    def uncertified_mass_fraction(self):
        """Share of the total mass carried by panels that failed tolerance.

        The severity measure that ``refinement_capped`` is not.  Refinement
        routinely drives one panel to ``max_depth`` against an algebraic
        boundary singularity and force-accepts it; that trips the depth
        flag while the panel in question holds a ``2**-22``-wide sliver
        of probability and the CDF is accurate everywhere.  What actually
        matters is how much probability sits under polynomials that were
        never certified, which is this.

        Masses are re-measured by quadrature before normalization (see
        :meth:`_recertify_panel_masses`), so the numerator is a true mass
        even when the panel's own polynomial was not usable.

        Returns
        -------
        float
            Value in ``[0, 1]``; ``0.0`` when every panel converged.
        """
        if not self.panels or not (self.total_mass > 0.0):
            return 0.0
        bad = sum(float(p.mass) for p in self.panels if not p.certified)
        return float(min(max(bad / self.total_mass, 0.0), 1.0))

    @property
    def mass_defect(self):
        """``|1 - total_mass|``: probability the panels never resolved.

        The representation is renormalized by ``total_mass``, so a defect
        here does not make the CDF invalid -- it stays a proper
        distribution -- but it does mean the CDF is not the integral of
        the density it was built from.  Non-trivial values indicate mass
        lost into an unresolved boundary singularity.

        Returns
        -------
        float
        """
        return abs(1.0 - float(self.total_mass))

def boundary_aware_breaks_from_amplitudes(support, boundary_amplitudes):
    """Return compact-coordinate seed breaks near active endpoint logs.

    Zero-offset logarithmic boundary factors create endpoint branch behavior
    that is easier for the adaptive spectral builder to resolve when a few
    near-boundary compact-coordinate breaks are supplied explicitly.

    Parameters
    ----------
    support : sequence of (float, float)
        Canonical support endpoints.
    boundary_amplitudes : array_like, shape (2,)
        Canonical lower/upper zero-offset amplitudes.

    Returns
    -------
    list of float
        Sorted unique seed breaks in compact coordinates.
    """
    supp = np.asarray(support, dtype=np.float64).reshape(2)
    amps = np.asarray(boundary_amplitudes, dtype=np.float64).reshape(-1)
    if amps.size != 2:
        raise ValueError("boundary_amplitudes must have length 2")
    L, U = map(float, supp)
    aL, aU = map(float, amps)
    offsets = (1e-3, 8e-3, 6.4e-2, 0.25)
    breaks = []
    if np.isfinite(L) and np.isfinite(aL) and aL > 0.0:
        breaks.extend(-1.0 + d for d in offsets)
    if np.isfinite(U) and np.isfinite(aU) and aU > 0.0:
        breaks.extend(1.0 - d for d in offsets)
    return sorted({float(z) for z in breaks if -1.0 < z < 1.0})
