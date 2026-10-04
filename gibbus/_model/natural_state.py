"""Normalized candidate state in affine natural coordinates.

A state binds one natural-parameter vector to its potential
``q = theta . t(z)`` and owns the model-side numerical quantities needed by
fitting: mode and integration window, shifted normalization, first-partial
means and Fisher covariance, prepared quadrature, and ordinary/generalized
moments.  The potential is linear in the parameters, so the first partials are
the fixed basis functions ``t(z)`` and every second partial is zero.

Normalization and the Fisher statistics come from one compiled traversal
(``_state_kernels.state_numerics``).  When that traversal declines a
potential (for example one that is not normalizable) the scalar QUADPACK path
below computes the same quantities, so the caller can still see and reject the
candidate.
"""

import math
from dataclasses import dataclass

import numpy as np

from .._defaults import (
    BACKTRACK_MAX_ITERS,
    BACKTRACK_REDUCE,
    BOUNDARY_EPS_MULT,
    BRACKET_INIT_STEP,
    BRACKET_MAX_EXPAND,
    BRACKET_STEP_GROWTH,
    GRAD_TOL,
    HESS_TOL,
    LOG_THRESH,
    NEWT_MAX,
    NEWT_TOL,
    QUAD_EPSABS,
    QUAD_EPSREL,
    QUAD_LIMIT,
    _reraise_if_debug,
)
from ._quad_integrals import PreparedQuad
from ._state_kernels import _pdf_vec, _q_window_and_mode, state_numerics
from .moments import _ledger_integral, _ModelMoments
from .natural import _NaturalLayout
from .numerics import (
    _build_quad_kernel,
    _complete_boundary_terms,
    _mode_quad_points,
)
from .spec import _LOGDIST, _LOWER, _POLY, _UPPER, _ModelSpec, _PotentialPartial

# Mode-search controls in the order ``_state_kernels.state_numerics`` expects.
_MODE_CONTROLS = np.array(
    [
        LOG_THRESH,
        GRAD_TOL,
        HESS_TOL,
        NEWT_TOL,
        NEWT_MAX,
        BOUNDARY_EPS_MULT,
        BRACKET_INIT_STEP,
        BRACKET_MAX_EXPAND,
        BRACKET_STEP_GROWTH,
        BACKTRACK_MAX_ITERS,
        BACKTRACK_REDUCE,
    ],
    dtype=np.float64,
)


def _pack_partials(partials, width, /):
    """Pack first potential partials for ``state_numerics``.

    Parameters
    ----------
    partials : sequence of _PotentialPartial
        First partials in parameter order.
    width : int
        Coefficient columns (at least the longest polynomial partial).

    Returns
    -------
    kinds, lengths, coefficients : numpy.ndarray
        Kind codes (0 polynomial, 1 lower log, 2 upper log), polynomial
        lengths and zero-padded ascending coefficients.
    """
    kinds = np.zeros(len(partials), dtype=np.int32)
    lengths = np.zeros(len(partials), dtype=np.int32)
    coefficients = np.zeros((len(partials), int(width)), dtype=np.float64)
    for i, partial in enumerate(partials):
        if partial.kind == _POLY:
            c = np.asarray(partial.coefficients, dtype=np.float64).reshape(-1)
            lengths[i] = c.size
            coefficients[i, : c.size] = c
        else:
            kinds[i] = 1 if partial.boundary_side == _LOWER else 2
    return kinds, lengths, coefficients


def _natural_partials(layout, /):
    """Return fixed first-potential partials in natural parameter order.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    """
    q_size = layout.effective_poly_degree + 1
    partials = []

    gamma = np.zeros(q_size, dtype=np.float64)
    gamma[1] = 1.0
    partials.append(_PotentialPartial(_POLY, gamma, None))

    for order in range(layout.curvature_degree + 1):
        coefficients = np.zeros(q_size, dtype=np.float64)
        coefficients[order + 2] = 1.0 / ((order + 1.0) * (order + 2.0))
        partials.append(_PotentialPartial(_POLY, coefficients, None))

    if layout.lower_a_index is not None:
        partials.append(_PotentialPartial(_LOGDIST, None, _LOWER))
    if layout.upper_a_index is not None:
        partials.append(_PotentialPartial(_LOGDIST, None, _UPPER))

    if len(partials) != layout.n_params:
        raise RuntimeError("natural partial layout does not match parameter layout")
    return tuple(partials)


@dataclass(frozen=True)
class _LayoutNumerics:
    """Per-layout constants of the natural state (built once per layout).

    Parameters
    ----------
    partials : tuple of _PotentialPartial
        First potential partials in parameter order.
    kinds, lengths, coefficients : numpy.ndarray
        The partials packed for ``state_numerics`` (0 polynomial, 1 lower
        log, 2 upper log).
    curvature_scale : numpy.ndarray
        ``1 / ((k + 1)(k + 2))`` mapping curvature to potential coefficients.
    d1_factor, d2_factor : numpy.ndarray
        ``polyder`` factors for the first and (second from first) derivative.
    support : numpy.ndarray, shape (2,)
        Canonical support.
    """

    partials: tuple
    kinds: np.ndarray
    lengths: np.ndarray
    coefficients: np.ndarray
    curvature_scale: np.ndarray
    d1_factor: np.ndarray
    d2_factor: np.ndarray
    support: np.ndarray


_LAYOUT_NUMERICS: dict = {}


def _layout_numerics(layout, /):
    """Return the cached ``_LayoutNumerics`` of one natural layout.

    Parameters
    ----------
    layout : _NaturalLayout
        Natural parameter layout.
    """
    key = (
        layout.support_lower,
        layout.support_upper,
        layout.effective_poly_degree,
        layout.lower_a_index,
        layout.upper_a_index,
    )
    cached = _LAYOUT_NUMERICS.get(key)
    if cached is not None:
        return cached
    partials = _natural_partials(layout)
    kinds, lengths, coefficients = _pack_partials(
        partials, layout.effective_poly_degree + 1
    )
    orders = np.arange(layout.curvature_degree + 1, dtype=np.float64)
    degree = layout.effective_poly_degree
    cached = _LayoutNumerics(
        partials=partials,
        kinds=kinds,
        lengths=lengths,
        coefficients=coefficients,
        curvature_scale=1.0 / ((orders + 1.0) * (orders + 2.0)),
        d1_factor=np.arange(1, degree + 1, dtype=np.float64),
        d2_factor=np.arange(1, degree, dtype=np.float64),
        support=np.array(layout.support, dtype=np.float64),
    )
    _LAYOUT_NUMERICS[key] = cached
    return cached


class _NaturalCoreState:
    """Fully normalized candidate density in natural coordinates.

    Parameters
    ----------
    coordinate : _FitCoordinate
        Fixed affine fitting coordinate.
    layout : _NaturalLayout
        Natural parameter layout.
    params : numpy.ndarray, shape (n,)
        Natural parameters.
    z_data_bounds : tuple of (float, float)
        Canonical data range.  It only guides the integration window; it is
        not a model parameter.
    """

    def __init__(self, coordinate, layout, params, z_data_bounds, /):
        """Build and normalize a state directly from natural parameters.

        Parameters
        ----------
        coordinate : _FitCoordinate
            Fixed affine fitting coordinate.
        layout : _NaturalLayout
            Natural parameter layout.
        params : numpy.ndarray, shape (n,)
            Natural parameters.
        z_data_bounds : tuple of (float, float)
            Canonical data range used by quadrature windowing.
        """
        if not isinstance(layout, _NaturalLayout):
            raise TypeError("layout must be a _NaturalLayout")
        canonical = coordinate.canonical_support
        if (
            float(canonical[0]) != layout.support_lower
            or float(canonical[1]) != layout.support_upper
        ):
            raise ValueError("coordinate and natural layout supports do not match")
        lo, hi = (float(v) for v in z_data_bounds)
        if not lo <= hi:
            raise ValueError("z_data_bounds must be an ordered pair")

        numerics = _layout_numerics(layout)
        p = np.array(params, dtype=np.float64).reshape(-1)
        layout._validate_canonical_params(p)
        amplitudes = np.full(2, np.nan, dtype=np.float64)
        if layout.lower_a_index is not None:
            amplitudes[0] = p[layout.lower_a_index]
        if layout.upper_a_index is not None:
            amplitudes[1] = p[layout.upper_a_index]
        self.layout = layout
        self.spec = _ModelSpec(coordinate, layout)
        self.p = p
        self.z_data_bounds = (lo, hi)

        # The candidate arrays, bit-for-bit as ``layout.build_candidate``.
        q_poly = np.zeros(layout.effective_poly_degree + 1, dtype=np.float64)
        q_poly[1] = p[layout.gamma_index]
        q_poly[2:] = p[layout.curvature_slice] * numerics.curvature_scale
        self.q_poly = q_poly
        self.q_d1 = numerics.d1_factor * q_poly[1:]
        self.q_d2 = numerics.d2_factor * self.q_d1[1:]
        self.boundary_amplitudes = amplitudes
        self.partials = numerics.partials

        self.window = None
        self.mode = None
        self.quad_poly = None
        self.quad_terms = None
        self.quad_points = None
        self._prepared_quad = None
        self._side_to_term = None
        self.Z = None
        self.log_Z = None
        self.q_shift = 0.0
        self.local_scale = 1.0
        self.moments = None
        self._first_means_fisher = None

        if not self._compiled_numerics(numerics):
            self._init_numerics()
            self.moments = _ModelMoments(self)

    @property
    def _quad_context(self):
        """Prepared scalar quadrature, built on first use."""
        if self._prepared_quad is None and self.quad_poly is not None:
            self._prepared_quad = PreparedQuad(
                self.quad_poly,
                float(self.window[0]),
                float(self.window[1]),
                self.quad_terms,
            )
        return self._prepared_quad

    def _compiled_numerics(self, numerics, /):
        """Normalize and integrate the Fisher statistics in one compiled call.

        Returns ``False`` (leaving the state untouched) when the compiled
        traversal declines, e.g. a potential that is not normalizable; the
        scalar path then runs instead.  On success the state carries
        ``_first_means_fisher`` and prefilled moment caches.

        Parameters
        ----------
        numerics : _LayoutNumerics
            Cached per-layout constants.
        """
        spec = self.spec
        lower_basis = spec.canonical_lower_a_index is not None
        upper_basis = spec.canonical_upper_a_index is not None
        coefficients = numerics.coefficients
        status, geometry, points, shifted_z, moments, means, fisher = state_numerics(
            numerics.support,
            self.q_poly,
            self.boundary_amplitudes,
            np.array(self.z_data_bounds, dtype=np.float64),
            lower_basis,
            upper_basis,
            numerics.kinds,
            numerics.lengths,
            coefficients,
            _MODE_CONTROLS,
            QUAD_EPSABS,
            QUAD_EPSREL,
            QUAD_LIMIT,
        )
        if status != 0:
            return False
        self.window = geometry[:2].copy()
        self.mode = float(geometry[2])
        self.q_shift = float(geometry[3])
        self.local_scale = float(geometry[5])
        shifted = self.q_poly.copy()
        shifted[0] -= self.q_shift
        self.quad_poly = shifted
        self.quad_terms, self._side_to_term = _complete_boundary_terms(
            spec.support, self.boundary_amplitudes, lower_basis, upper_basis
        )
        self.quad_points = tuple(points.tolist())
        self.Z = float(shifted_z)
        self.log_Z = -self.q_shift + math.log(self.Z)

        width = coefficients.shape[1]
        n_power = 2 * width - 1
        cache = _ModelMoments(self)
        cache._power = moments[:n_power]
        offset = n_power
        if lower_basis:
            cache._log_power[_LOWER] = moments[offset : offset + width]
            offset += width
        if upper_basis:
            cache._log_power[_UPPER] = moments[offset : offset + width]
            offset += width
        if lower_basis:
            cache._log_square[_LOWER] = float(moments[offset])
            offset += 1
        if upper_basis:
            cache._log_square[_UPPER] = float(moments[offset])
            offset += 1
        if lower_basis and upper_basis:
            cache._log_cross = float(moments[offset])
        self.moments = cache
        self._first_means_fisher = (means, fisher)
        return True

    def _window_and_mode(self, boundary_amplitudes, /, *, include_core=False):
        """Run the compiled mode/window search for given boundary amplitudes.

        Parameters
        ----------
        boundary_amplitudes : numpy.ndarray, shape (2,)
            Canonical lower/upper amplitudes used for the search.
        include_core : bool, optional
            Also return the density-defined pre-padding tail window.
        """
        return _q_window_and_mode(
            self.spec.support,
            self.q_poly,
            boundary_amplitudes,
            self.z_data_bounds,
            LOG_THRESH,
            GRAD_TOL,
            HESS_TOL,
            NEWT_TOL,
            NEWT_MAX,
            BOUNDARY_EPS_MULT,
            BRACKET_INIT_STEP,
            BRACKET_MAX_EXPAND,
            BRACKET_STEP_GROWTH,
            BACKTRACK_MAX_ITERS,
            BACKTRACK_REDUCE,
            include_core,
        )

    def _init_numerics(self, /):
        """Compute mode/window and stable shifted normalization quadrature."""
        amplitudes = np.asarray(self.boundary_amplitudes, dtype=np.float64)
        window, mode, q_min, q2_mode, core_window = self._window_and_mode(
            amplitudes, include_core=True
        )
        if not np.isfinite(float(q_min)) and np.any(amplitudes > 0.0):
            # A vanishingly small positive amplitude puts the mode on its own
            # singular endpoint, where q = +inf.  The tail points are defined
            # relative to q(mode), so they collapse and the window truncates
            # real tail mass (2.7e-4 of log Z at a = 1e-12 on an exponential
            # shape).  Away from an endpoint layer of width ~a / q' the density
            # equals the regular one, so take the window, mode and shift from
            # the potential without the collapsed log terms; the quadrature
            # below still integrates the full density.
            lower, upper = (float(value) for value in self.spec.support)
            regular = amplitudes.copy()
            span = max(1.0, abs(float(mode)))
            if np.isfinite(lower) and float(mode) - lower <= 1e-8 * span:
                regular[0] = 0.0
            if np.isfinite(upper) and upper - float(mode) <= 1e-8 * span:
                regular[1] = 0.0
            window, mode, q_min, q2_mode, core_window = self._window_and_mode(
                regular, include_core=True
            )
        self.window = np.asarray(window, dtype=np.float64)
        self.mode = float(mode)

        q2_mode = float(q2_mode)
        if np.isfinite(q2_mode) and q2_mode > 0.0:
            self.local_scale = float(1.0 / np.sqrt(q2_mode))
        else:
            self.local_scale = 1.0

        q_min = float(q_min)
        if not np.isfinite(q_min):
            q_min = 0.0
        self.q_shift = q_min

        shifted_poly = np.asarray(self.q_poly, dtype=np.float64).copy()
        shifted_poly[0] -= q_min
        self.quad_poly, _ = _build_quad_kernel(
            shifted_poly, self.spec.support, self.boundary_amplitudes
        )
        self.quad_terms, self._side_to_term = _complete_boundary_terms(
            self.spec.support,
            self.boundary_amplitudes,
            self.spec.canonical_lower_a_index is not None,
            self.spec.canonical_upper_a_index is not None,
        )
        # These breakpoints depend only on the completed state geometry, not
        # on the requested moment.  Compute them once instead of rebuilding
        # the same list for every model integral.
        self.quad_points = tuple(
            _mode_quad_points(
                self.window, self.mode, self.local_scale, self.spec.support, core_window
            )
        )
        self._prepared_quad = PreparedQuad(
            self.quad_poly,
            float(self.window[0]),
            float(self.window[1]),
            self.quad_terms,
        )

        if not np.all(np.isfinite(self.quad_poly)):
            _reraise_if_debug(
                ArithmeticError("potential contains non-finite coefficients"),
                "potential polynomial overflow",
                routine=True,
            )
            z_shifted = 0.0
        else:
            z_shifted = self._quad(0, 0, 0)

        self.Z = float(z_shifted)
        self.log_Z = (
            -q_min + math.log(z_shifted)
            if z_shifted > 0.0 and np.isfinite(z_shifted)
            else -np.inf
        )

    def _quad(self, mode, k, t_index, /):
        """Run one shifted model quadrature in canonical coordinates.

        Parameters
        ----------
        mode : int
            Integration mode understood by ``quad_integral``.
        k : int
            Power of the canonical coordinate in the integrand.
        t_index : int
            Boundary-term index for generalized integration modes.

        Returns
        -------
        float
            Shifted unnormalized integral.
        """
        return _ledger_integral(
            self._quad_context,
            "model-state quadrature",
            mode=int(mode),
            k=int(k),
            t_index=int(t_index),
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
            points=self.quad_points,
        )

    def moment_raw(self, k, /):
        """Return one canonical-coordinate raw model moment.

        Parameters
        ----------
        k : int
            Nonnegative moment order.

        Returns
        -------
        float
            Normalized model moment ``E[Z**k]``.
        """
        k = int(k)
        if k < 0:
            raise ValueError("k must be >= 0")
        return float(self.moments.power(k)[k])

    def pdf(self, z, /):
        """Evaluate the normalized density in canonical fitting coordinates.

        Parameters
        ----------
        z : array_like
            Canonical-coordinate values.

        Returns
        -------
        float or numpy.ndarray
            Normalized canonical density values.
        """
        zz = np.asarray(z, dtype=np.float64)
        scalar = zz.ndim == 0
        out = _pdf_vec(
            np.atleast_1d(zz),
            tuple(self.spec.support),
            np.asarray(self.q_poly, dtype=np.float64),
            np.asarray(self.boundary_amplitudes, dtype=np.float64),
            float(self.log_Z),
        )
        return float(out[0]) if scalar else out
