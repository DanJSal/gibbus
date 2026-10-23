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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .._defaults import (
    NUMERIC_FAILURES,
    _reraise_if_debug,
)
from .._postfit import analytics as _pf
from .._postfit.basis_state import (
    _canonical_raw_moment,
    _is_basis_state,
    _potential_of,
)
from .._postfit.evaluators import (
    _potential_exp_from_x_potential,
)
from .._postfit.expectation import expect_vectorized as _expect_vectorized
from .._postfit.fitted_state import (
    _check_state_invariants,
    _structured_scalar,
)
from .._postfit.logspace import log_mass_between
from .._spectral.cdf import (
    SpectralCDF,
    density_spec,
)
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


class _Component:
    """Internal single-component log-concave density estimator.

    This class handles the fitting, state management, and evaluation of
    a single log-concave component.  It is not part of the public API;
    users interact with :class:`Distribution` instead.
    """

    __slots__ = (
        "_base_cdf",
        "_base_exp_potential",
        "_base_ppf",
        "_base_view",
        "_basis",
        "_canonical_moments",
        "_center",
        "_context",
        "_data",
        "_exp_view",
        "_scale",
        "_spectral_cdf_eval",
        "_spectral_ppf_eval",
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
        self._center: float = float("nan")
        self._scale: float = float("nan")
        self._base_cdf: Any = None
        self._base_ppf: Any = None
        self._spectral_cdf_eval: Any = None
        self._spectral_ppf_eval: Any = None
        self._base_exp_potential: Any = None
        # The component's normalized G3 potential (G3-046); ``None`` unfitted.
        self._basis: Any = None
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
        """Return the smallest base-space ``V''`` found on a quantile grid.

        A fit whose shape verdict is not ``nonnegative`` has no grid margin
        (``-inf``); otherwise the smallest ``V''`` over a quantile grid,
        floored at the verdict's zero (G3-046, G3-047).  On an infinite side
        the grid reaches the quantiles ``1e-300`` (standing for the tail
        limit); a finite side's limit is its endpoint, approached
        geometrically in ``x`` from the grid's innermost quantile, since
        ultratail quantiles need not be representable there (G3-059).
        """
        data = self._ensure_fitted()
        if str(data["basis_shape"]) != "nonnegative":
            return -np.inf
        base = self.base
        lo, hi = map(float, base.support)
        levels = np.linspace(1e-12, 0.5, 257)
        far = np.array([1e-300, 1e-100, 1e-30])
        approach = np.logspace(0.0, -30.0, 31)
        parts = []
        for inverse, end in ((base.ppf, lo), (base.isf, hi)):
            if np.isfinite(end):
                inner = np.asarray(inverse(levels[1:]), dtype=np.float64)
                parts += [inner, end + (inner[0] - end) * approach]
            else:
                parts.append(inverse(np.concatenate((far, levels))))
        x = np.unique(np.concatenate(parts))
        x = x[(x > lo) & (x < hi)]
        curvature = np.asarray(base.neg_log(x, 2), dtype=np.float64)
        curvature = curvature[np.isfinite(curvature)]
        if curvature.size == 0:
            return 0.0
        return max(0.0, float(np.min(curvature)))

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
        data["basis_log_z"] = float(data["basis_log_z"]) + log_mass
        data["support"] = np.array([lo, hi], dtype=np.float64)

        mu_eff, sigma_eff = self._mu_sigma_eff()
        base_bounds = np.array([lo, hi], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            z_bounds = sigma_eff * base_bounds + mu_eff
        trunc_z_support = np.sort(z_bounds)

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

        part = self._density_part(0.0, 1.0, 1.0, 1.0, -np.inf, np.inf)
        # Renormalized by the retained mass (the truncated log Z).
        part = (*part[:4], part[4] + log_mass, *part[5:])
        density = density_spec([part], view=False)
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
        """Return a deep copy of this component's internal runtime state."""
        self._ensure_fitted()
        return self._data.copy()

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
        # The packed moments stop at order 4; higher orders integrate the
        # potential directly (breakpoint at the canonical mean).
        val = _canonical_raw_moment(self._basis, float(rm[1]), int(k))
        self._canonical_moments[k] = val
        return val

    def _base_mean_parts(self):
        """Return ``(common_origin, local_mean)`` for the base affine map."""
        origin, scale = self._affine()
        z_mean = float(self._canonical_raw_moment(1))
        local = (self._center + self._scale * z_mean) * scale
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
        self._center = float(self._data["fit_center"])
        self._scale = float(self._data["fit_scale"])
        self._spectral_cdf_eval = build_cdf_evaluator(self._data)
        self._spectral_ppf_eval = build_ppf_evaluator(self._data)
        self._base_cdf = self._spectral_cdf_eval
        self._base_ppf = self._spectral_ppf_eval.eval_x
        self._base_exp_potential = _potential_exp_from_x_potential(
            lambda x, n: self._base_view.neg_log(x, n)
        )
        if not _is_basis_state(self._data):
            raise ValueError("component state is not a G3 basis state")
        self._basis = _potential_of(self._data)

    def _density_part(self, mu, sigma, jf, weight, lo, hi, /, z_support=None):
        """This component as one entry of a spectral ``density_spec``.

        Parameters
        ----------
        mu, sigma, jf, weight, lo, hi : float
            G1 ``density_spec`` entry fields (affine map, Jacobian, weight
            and public mask).
        z_support : tuple of float or None
            Canonical potential support (default: the stored anchors).
        """
        zl, zu = (
            map(float, self._data["canonical_support"])
            if z_support is None
            else map(float, z_support)
        )
        tail = (float(mu), float(sigma), float(jf), float(weight), lo, hi)
        log_norm = self._basis.log_z - np.log(self._basis.h)
        head = (self._basis.kernel, self._basis.theta, zl, zu)
        return (*head, log_norm, *tail)

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
        shift, scale = self._affine()
        mu_eff = (-shift / scale - self._center) / self._scale
        sigma_eff = 1.0 / (scale * self._scale)
        return mu_eff, sigma_eff
