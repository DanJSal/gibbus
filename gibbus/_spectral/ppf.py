"""Adaptive spectral PPF representation.

This module inverts an already-built :class:`~gibbus._spectral.cdf.SpectralCDF`
and stores a piecewise Chebyshev approximation to the inverse in the
compactified support coordinate ``z``.  The partition is built by the
compiled builder (:mod:`._builders`) from the source CDF's density
description; the packed inverse representation is evaluated in bulk by
:mod:`._ppf_eval`.  The tests keep a pure-Python construction harness for
builder-control-flow and bitwise-orchestration tests
(``tests/spectral_builder_harness.py``).

Interior probabilities are parameterized by the float64 log-odds coordinate
``r = log(p) - log1p(-p)``.  This avoids fitting directly across the singular
PPF endpoint derivatives at ``p=0`` and ``p=1``.  Exact endpoints are handled
separately.  Local polynomial degrees are selected from ``SPECTRAL_DEGREE_OPTIONS`` by
p-refinement before interval bisection, and monotonicity is certified from
Bernstein lower bounds on each derivative polynomial.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import expit

from . import _builders
from ._ppf_eval import SpectralPPFEvaluator
from .cdf import _KIND_CODE, SpectralCDF, _builder_tables
from .config import _SpectralPPFOptions

# Spectral inverse domain. More extreme probabilities are evaluated by direct
# inversion of the already-spectral CDF; this avoids fitting through the
# discrete/subnormal logit endpoint while retaining exact PPF semantics.
_PSPEC_MIN = np.float64(1e-12)
_PSPEC_MAX = np.float64(1.0 - 1e-12)


def _logit(p):
    """Log-odds coordinate ``log(p) - log1p(-p)``.

    Parameters
    ----------
    p : array_like
        Probabilities in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Log-odds values.
    """
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(p) - np.log1p(-p)


@dataclass
class _InversePanel:
    """Locally normalized inverse panel with monotonicity diagnostics.

    Parameters
    ----------
    ra, rb : float
        Log-odds probability bounds.
    pa, pb : float
        Corresponding probability bounds.
    za, zb : float
        Compact-coordinate quantile bounds.
    coeff : numpy.ndarray
        Chebyshev coefficients for the locally normalized compact quantile.
    fit_error, logit_residual, prob_residual, tail_abs : float
        Fit, log-odds, probability and coefficient-tail diagnostics.
    derivative_lower : float
        Certified lower bound on the local inverse derivative.
    depth, source_panel : int
        Subdivision depth and originating forward-panel index.
    """

    ra: float
    rb: float
    pa: float
    pb: float
    za: float
    zb: float
    coeff: np.ndarray
    fit_error: float
    logit_residual: float
    prob_residual: float
    tail_abs: float
    derivative_lower: float
    depth: int
    source_panel: int

    @property
    def degree(self):
        """Return the local inverse polynomial degree."""
        return int(self.coeff.size - 1)


class SpectralPPF:
    """Adaptive monotone spectral inverse of a :class:`SpectralCDF`.

    The approximation is for the compact coordinate ``z`` as a function of
    log-odds probability ``r``.  On every local panel both input and output are
    affinely normalized to ``[-1,1]``.  Thus all stored Chebyshev targets are
    O(1), even for extreme probabilities and unbounded physical support.
    """

    def __init__(
        self,
        spectral_cdf: SpectralCDF,
        *,
        config: _SpectralPPFOptions,
    ):
        """Invert a built spectral CDF into a piecewise Chebyshev PPF.

        Parameters
        ----------
        spectral_cdf : gibbus._spectral.cdf.SpectralCDF
            Built forward representation to invert.
        config : _SpectralPPFOptions
            Explicit degree/tolerance/depth/panel/certification policy. Budget
            exhaustion raises rather than accepting an uncertified inverse.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If *spectral_cdf* is not a ``SpectralCDF``.
        """
        if not isinstance(spectral_cdf, SpectralCDF):
            raise TypeError("spectral_cdf must be a SpectralCDF")
        opts = tuple(sorted({int(v) for v in config.degree_options}))
        if not opts or opts[0] < 2:
            raise ValueError("degree_options must contain degrees >= 2")

        self.cdf_rep = spectral_cdf
        self._dicoeff_cache: dict[int, np.ndarray] = {}
        self.degree_options = opts
        self.fit_tol = float(config.fit_tol)
        self.logit_tol = float(config.logit_tol)
        self.prob_tol = float(config.prob_tol)
        self.coeff_tol = float(config.coeff_tol)
        self.max_depth = int(config.max_depth)
        self.max_panels = int(config.max_panels)
        self.certify_subdivide = int(config.certify_subdivide)
        self.pmin = float(_PSPEC_MIN)
        self.pmax = float(_PSPEC_MAX)
        self.rmin = float(_logit(self.pmin))
        self.rmax = float(_logit(self.pmax))

        # Seed inverse panels with the CDF panel masses, expressed in logit
        # coordinates.  Zero-mass CDF tail panels are naturally skipped.
        probs = [self.pmin]
        probs.extend(
            float(p)
            for p in spectral_cdf.cum_mass[1:-1]
            if self.pmin < float(p) < self.pmax
        )
        probs.append(self.pmax)
        prob_grid = np.array(sorted(set(probs)), dtype=np.float64)

        intervals = []
        for pa, pb in itertools.pairwise(prob_grid):
            if not pb > pa:
                continue
            pm = self._prob_mid_logit(pa, pb)
            source_j = self._source_panel_for_probability(pm)
            # A CDF cumulative-mass boundary is in probs, so an interval
            # should not cross a positive-mass source panel.  In the presence
            # of rounded zero-mass panels, use the actual inverses at endpoints
            # and let the source selection follow the midpoint.
            za = self._invert_global(pa, prefer_left=True)
            zb = self._invert_global(pb, prefer_left=False)
            ra, rb = float(_logit(pa)), float(_logit(pb))
            intervals.append((source_j, ra, rb, float(pa), float(pb), za, zb))

        if len(intervals) > self.max_panels:
            raise RuntimeError(
                f"spectral PPF needs {len(intervals)} source intervals but its "
                f"budget is {self.max_panels}; use CDF bisection fallback"
            )
        panels = self._build_panels(intervals)
        panels.sort(key=lambda p: p.ra)
        if not panels:
            raise RuntimeError("spectral CDF has no invertible interior panels")
        self.panels = panels
        self.breaks_r = np.array(
            [panels[0].ra] + [p.rb for p in panels], dtype=np.float64
        )
        self.breaks_r[0] = self.rmin
        self.breaks_r[-1] = self.rmax
        self._cython_evaluator = self._build_cython_evaluator()

    def _build_panels(self, intervals):
        """Build the inverse partition from its seed intervals.

        Parameters
        ----------
        intervals : list of tuple
            ``(source_j, ra, rb, pa, pb, za, zb)`` seed intervals.

        Returns
        -------
        list of _InversePanel
        """
        if getattr(self.cdf_rep, "density", None) is None:
            raise TypeError("the spectral CDF carries no density description")
        return self._build_compiled(intervals)

    def _build_compiled(self, intervals):
        """Build the inverse partition in the compiled builder.

        Same contract as :meth:`_build_partition`, including the
        ``RuntimeError`` messages on a depth/spacing or budget failure.

        Parameters
        ----------
        intervals : sequence of tuple
            ``(source_j, ra, rb, pa, pb, za, zb)`` seed intervals.

        Returns
        -------
        list of _InversePanel
        """
        sp = self.cdf_rep
        offsets, ev_coeffs, ev_ncoeff = sp._packed
        m = len(sp.panels)
        istride = max(p.icoeff.size for p in sp.panels)
        icoeff = np.zeros((m, istride), dtype=np.float64)
        nicoeff = np.empty(m, dtype=np.int32)
        for j, panel in enumerate(sp.panels):
            icoeff[j, : panel.icoeff.size] = panel.icoeff
            nicoeff[j] = panel.icoeff.size
        seed_src = np.array([iv[0] for iv in intervals], dtype=np.int32)
        seed = np.array([iv[1:] for iv in intervals], dtype=np.float64).reshape(-1, 6)
        mp = sp.map
        status, rows, ints, coeff, detail = _builders.build_ppf(
            sp.density,
            _builder_tables(sp.degree_options, self.degree_options),
            _KIND_CODE[mp.kind],
            float(mp.L),
            float(mp.U),
            float(mp.center),
            float(mp.scale),
            np.ascontiguousarray(sp.breaks, dtype=np.float64),
            np.ascontiguousarray(offsets, dtype=np.float64),
            np.ascontiguousarray(sp.cum_mass, dtype=np.float64),
            np.ascontiguousarray(ev_coeffs, dtype=np.float64),
            np.ascontiguousarray(ev_ncoeff, dtype=np.int32),
            np.array([p.a for p in sp.panels], dtype=np.float64),
            np.array([p.b for p in sp.panels], dtype=np.float64),
            icoeff,
            nicoeff,
            float(sp.total_mass),
            seed_src,
            seed,
            np.asarray(self.degree_options, dtype=np.int32),
            self.prob_tol,
            self.max_depth,
            self.max_panels,
            self.certify_subdivide,
        )
        panels = [
            _InversePanel(
                ra=float(r[0]),
                rb=float(r[1]),
                pa=float(r[2]),
                pb=float(r[3]),
                za=float(r[4]),
                zb=float(r[5]),
                coeff=coeff[j, : int(ints[j, 2])].copy(),
                fit_error=float(r[6]),
                logit_residual=float(r[7]),
                prob_residual=float(r[8]),
                tail_abs=float(r[9]),
                derivative_lower=float(r[10]),
                depth=int(ints[j, 0]),
                source_panel=int(ints[j, 1]),
            )
            for j, r in enumerate(rows)
        ]
        if status == 1:
            raise RuntimeError(self._failure_message(panels[detail]))
        if status == 2:
            raise RuntimeError(
                f"spectral PPF exhausted its {self.max_panels}-panel "
                f"budget with {detail} uncertified leaf interval(s); "
                "the inverse could not be certified monotone within budget"
            )
        return panels

    # ------------------------------------------------------------------
    # Source spectral-CDF inversion used only during construction
    # ------------------------------------------------------------------

    @staticmethod
    def _prob_mid_logit(pa, pb):
        """Midpoint of two probabilities taken in the log-odds coordinate.

        Parameters
        ----------
        pa, pb : float
            Interval endpoints as probabilities.

        Returns
        -------
        float
            Probability at the log-odds midpoint.
        """
        return float(expit(0.5 * (float(_logit(pa)) + float(_logit(pb)))))

    def _source_panel_for_probability(self, p):
        """Index of the forward CDF panel containing probability *p*.

        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.

        Returns
        -------
        int
            Source panel index, clipped to the valid range.
        """
        cm = self.cdf_rep.cum_mass
        j = int(np.searchsorted(cm, float(p), side="right") - 1)
        return int(np.clip(j, 0, len(self.cdf_rep.panels) - 1))

    def _invert_source(self, j, p):
        """Invert the CDF for *p* known to lie within source panel *j*.

        Parameters
        ----------
        j : int
            Source CDF panel index.
        p : float
            Probability bracketed by that panel.

        Returns
        -------
        float
            Compact coordinate ``z`` with ``CDF(z) = p``.
        """
        sp = self.cdf_rep
        pa = float(sp.cum_mass[j])
        pb = float(sp.cum_mass[j + 1])
        p = float(p)
        if p <= pa:
            return float(sp.breaks[j])
        if p >= pb:
            return float(sp.breaks[j + 1])
        if not pb > pa:
            return float(sp.breaks[j])
        frac = float(np.clip((p - pa) / (pb - pa), 0.0, 1.0))
        return float(sp._cython_evaluator.invert_panel_fraction(j, frac))

    def _invert_global(self, p, *, prefer_left=False):
        """Invert the CDF for an arbitrary probability.

        Locates the bracketing panel first, walking to a non-degenerate
        one if rounding selected a zero-mass panel.

        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        prefer_left : bool, optional
            Which side to favor when *p* falls exactly on a panel edge.

        Returns
        -------
        float
            Compact coordinate ``z``.
        """
        p = float(p)
        if p <= 0.0:
            return -1.0
        if p >= 1.0:
            return 1.0
        cm = self.cdf_rep.cum_mass
        side = "left" if prefer_left else "right"
        j = int(np.searchsorted(cm, p, side=side) - 1)
        j = int(np.clip(j, 0, len(self.cdf_rep.panels) - 1))
        # If rounding selected a zero-mass panel, walk to one that brackets p.
        if not (float(cm[j]) <= p <= float(cm[j + 1])) or not (cm[j + 1] > cm[j]):
            candidates = np.flatnonzero(
                (cm[:-1] <= p) & (cm[1:] >= p) & (cm[1:] > cm[:-1])
            )
            if candidates.size:
                j = int(candidates[0] if prefer_left else candidates[-1])
        return self._invert_source(j, p)

    # ------------------------------------------------------------------
    # Monotonicity certificate
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Panel construction
    # ------------------------------------------------------------------

    @staticmethod
    def _z_from_v(v, za, zb):
        """Inverse of :meth:`_v_from_z`.

        Parameters
        ----------
        v : array_like
            Panel-local coordinates in ``[-1, 1]``.
        za, zb : float
            Panel edges in compact coordinates.

        Returns
        -------
        numpy.ndarray
            Compact coordinates.
        """
        return 0.5 * ((zb - za) * np.asarray(v) + (za + zb))

    @staticmethod
    def _failure_message(panel):
        """Format the terminal inverse-certification failure for one panel.

        Parameters
        ----------
        panel : _InversePanel
            Failed panel at a terminal refinement depth or width.

        Returns
        -------
        str
            Diagnostic exception message.
        """
        return (
            "spectral PPF failed to converge/certify logit panel "
            f"[{panel.ra:.6g}, {panel.rb:.6g}] "
            f"(p=[{panel.pa:.3e},{panel.pb:.3e}]) at depth {panel.depth}; "
            f"D={panel.degree}, fit={panel.fit_error:.3e}, "
            f"logit={panel.logit_residual:.3e}, p={panel.prob_residual:.3e}, "
            f"tail={panel.tail_abs:.3e}, dmin={panel.derivative_lower:.3e}"
        )

    def _build_cython_evaluator(self):
        """Pack the immutable inverse and source CDF for the compiled kernel."""
        m = len(self.panels)
        max_ncoeff = max(p.coeff.size for p in self.panels)
        coeffs = np.zeros((m, max_ncoeff), dtype=np.float64)
        ncoeff = np.empty(m, dtype=np.int32)
        for j, panel in enumerate(self.panels):
            c = np.asarray(panel.coeff, dtype=np.float64)
            coeffs[j, : c.size] = c
            ncoeff[j] = c.size
        breaks_z = np.array(
            [self.panels[0].za] + [panel.zb for panel in self.panels],
            dtype=np.float64,
        )

        sp = self.cdf_rep
        cm = len(sp.panels)
        cmax = max(p.icoeff.size for p in sp.panels)
        ccoeffs = np.zeros((cm, cmax), dtype=np.float64)
        cncoeff = np.empty(cm, dtype=np.int32)
        coffsets = np.asarray(sp.cum_mass[:-1], dtype=np.float64).copy()
        for j, panel in enumerate(sp.panels):
            ic = np.asarray(panel.icoeff, dtype=np.float64).copy()
            ic[0] -= C.chebval(-1.0, ic)
            ic /= sp.total_mass
            ccoeffs[j, : ic.size] = ic
            cncoeff[j] = ic.size

        kind_map = {
            "finite": 0,
            "lower": 1,
            "upper": 2,
            "real": 3,
            "lower_centered": 4,
            "upper_centered": 5,
        }
        mp = sp.map
        return SpectralPPFEvaluator(
            self.pmin,
            self.pmax,
            kind_map[mp.kind],
            float(mp.L),
            float(mp.U),
            float(mp.center),
            float(mp.scale),
            self.breaks_r,
            breaks_z,
            coeffs,
            ncoeff,
            np.asarray(sp.breaks, dtype=np.float64),
            coffsets,
            ccoeffs,
            cncoeff,
        )

    # ------------------------------------------------------------------
    # Runtime evaluation
    # ------------------------------------------------------------------

    def ppf_z(self, p):
        """Compact-coordinate quantile function.

        Probabilities outside the fitted log-odds range are resolved by
        direct inversion of the spectral CDF, which keeps exact PPF
        semantics at the extremes without fitting through subnormal
        log-odds.

        Parameters
        ----------
        p : array_like
            Probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray or float
            Compact coordinates; scalar in, scalar out.
        """
        return self._cython_evaluator.eval_compact(p)

    def ppf_cython(self, p, *, simd=True):
        """Physical-coordinate quantile function via the compiled kernel.

        ``simd=True`` is an intentional compiled-dispatch exception: it chooses
        an equivalent execution path, not statistical or accuracy policy.

        Parameters
        ----------
        p : array_like
            Probabilities in ``[0, 1]``.
        simd : bool, optional
            Use the SIMD-oriented large-array dispatch rather than the compiled
            scalar-per-observation loop. Hardware SIMD selection remains a
            compiler/target decision.

        Returns
        -------
        numpy.ndarray
            Quantiles in physical coordinates.
        """
        if simd:
            return self._cython_evaluator.eval_x(p)
        return self._cython_evaluator.eval_x_scalar(p)

    def ppf(self, p):
        """Physical-coordinate quantile function.

        Parameters
        ----------
        p : array_like
            Probabilities in ``[0, 1]``.

        Returns
        -------
        numpy.ndarray or float
            Quantiles; scalar in, scalar out.
        """
        return self.ppf_cython(p)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def max_depth_used(self):
        """Deepest refinement level reached by any accepted inverse panel."""
        return int(max(p.depth for p in self.panels)) if self.panels else 0
