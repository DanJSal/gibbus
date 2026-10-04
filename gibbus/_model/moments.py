"""Batch model moments for a normalized candidate state.

The point objective, exact Hessian, and degree diagnostics repeatedly need
coherent blocks of ordinary and generalized model expectations.  This module
owns those expectations and their caches. Ordinary powers share one compiled
adaptive traversal across requested orders, while generalized log moments use
the retained prepared scalar quadrature context. It deliberately operates on
the shifted quadrature representation rather than on raw observations.

Boundary-log statistics are keyed by canonical endpoint side and remain
available when an allowed direct boundary amplitude is exactly zero.  In that
case the corresponding zero-amplitude descriptor is retained solely as a
weight geometry; it does not change the density kernel.
"""

from dataclasses import dataclass

import numpy as np

from .._defaults import QUAD_EPSABS, QUAD_EPSREL, QUAD_LIMIT, _reraise_if_debug
from ._moment_kernels import power_moments as _power_moments_kernel

_LOWER = "lower"
_UPPER = "upper"


def _ledger_integral(prepared, context, /, **options):
    """Run one prepared QUADPACK integral, recording diagnostics in the ledger.

    ``full_output=1`` makes SciPy return QUADPACK's message instead of
    emitting ``IntegrationWarning``: this scalar path serves states the
    compiled traversal declined, typically far-out line-search trials that
    are then rejected, so a process-wide warning would be noise.

    This is a narrow kwargs exception mirroring the prepared QUADPACK option
    surface. It neither forwards raw fit options nor chooses numerical policy;
    callers supply their already-resolved integration controls.

    Parameters
    ----------
    prepared : PreparedQuad
        Prepared integrand context.
    context : str
        Ledger context for a reported integration problem.
    **options : dict
        Passed to ``PreparedQuad.integrate``.
    """
    out = prepared.integrate(full_output=1, **options)
    if len(out) > 3:
        _reraise_if_debug(RuntimeError(str(out[3])), context, routine=True)
    return float(out[0])


@dataclass
class _ModelMoments:
    """Cached ordinary and generalized moments for one live model state.

    Parameters
    ----------
    state : object
        Candidate state exposing ``spec``, ``q_poly``, ``boundary_amplitudes``,
        ``q_shift``, ``quad_poly``, ``window``, ``mode``, and ``Z``.
    """

    state: object

    def __post_init__(self):
        """Initialize private moment caches after dataclass construction."""
        self._power = None
        self._log_power = {}
        self._log_square = {}
        self._log_cross = None
        self._terms = self.state.quad_terms
        self._side_to_term = self.state._side_to_term

    @property
    def _quad(self):
        """The state's prepared scalar quadrature (built lazily by some states)."""
        return self.state._quad_context

    def power(self, max_order, /):
        """Return normalized power moments through one requested order.

        Parameters
        ----------
        max_order : int
            Highest nonnegative power required.

        Returns
        -------
        numpy.ndarray
            ``[E[Z**0], ..., E[Z**max_order]]``.
        """
        K = int(max_order)
        if K < 0:
            raise ValueError("max_order must be >= 0")
        have = -1 if self._power is None else self._power.size - 1
        if have < K:
            try:
                shifted = _power_moments_kernel(
                    np.ascontiguousarray(self.state.quad_poly, dtype=np.float64),
                    np.ascontiguousarray(self.state.spec.support, dtype=np.float64),
                    np.ascontiguousarray(
                        self.state.boundary_amplitudes, dtype=np.float64
                    ),
                    np.ascontiguousarray(self.state.window, dtype=np.float64),
                    np.ascontiguousarray(self.state.quad_points, dtype=np.float64),
                    int(K),
                    epsabs=QUAD_EPSABS,
                    epsrel=QUAD_EPSREL,
                    limit=QUAD_LIMIT,
                )
                raw = np.asarray(shifted, dtype=np.float64) / self.state.Z
            except (ArithmeticError, RuntimeError) as exc:
                # Preserve the mature scalar QUADPACK path as a numerical
                # fallback for unusual geometries.  Debug mode makes an
                # unexpected shared-traversal failure visible before falling
                # back on ordinary runs.
                _reraise_if_debug(exc, "shared power-moment quadrature")
                raw = np.empty(K + 1, dtype=np.float64)
                if self._power is not None:
                    raw[: self._power.size] = self._power
                start = max(0, have + 1)
                for k in range(start, K + 1):
                    raw[k] = self._integral(0, k, 0) / self.state.Z
            raw[0] = 1.0
            self._power = raw
        return self._power[: K + 1].copy()

    def power_covariance(self, max_order, /):
        """Return covariance of power statistics ``1,Z,...,Z**max_order``.

        Parameters
        ----------
        max_order : int
            Highest power statistic included in the covariance matrix.

        Returns
        -------
        numpy.ndarray
            Matrix with entry ``Cov(Z**i, Z**j)``.
        """
        K = int(max_order)
        if K < 0:
            raise ValueError("max_order must be >= 0")
        moments = self.power(2 * K)
        index = np.arange(K + 1)
        second = moments[index[:, None] + index[None, :]]
        mean = moments[: K + 1]
        return second - mean[:, None] * mean[None, :]

    def log_power(self, side, max_order, /):
        """Return ``E[Z**k log d_side(Z)]`` through one requested order.

        Parameters
        ----------
        side : {"lower", "upper"}
            Canonical endpoint side whose fixed zero-offset distance is used.
        max_order : int
            Highest nonnegative power required.

        Returns
        -------
        numpy.ndarray
            Generalized moments for powers zero through ``max_order``.

        Raises
        ------
        ValueError
            If the requested canonical boundary basis is not present.
        """
        side = _validated_side(side)
        if side not in self._side_to_term:
            raise ValueError(f"{side} boundary basis is not enabled")
        K = int(max_order)
        if K < 0:
            raise ValueError("max_order must be >= 0")
        cached = self._log_power.get(side)
        have = -1 if cached is None else cached.size - 1
        if have < K:
            out = np.empty(K + 1, dtype=np.float64)
            if cached is not None:
                out[: cached.size] = cached
            t_index = self._side_to_term[side]
            for k in range(max(0, have + 1), K + 1):
                out[k] = self._integral(1, k, t_index) / self.state.Z
            self._log_power[side] = out
        return self._log_power[side][: K + 1].copy()

    def log_square(self, side, /):
        """Return ``E[(log d_side(Z))**2]``.

        Parameters
        ----------
        side : {"lower", "upper"}
            Canonical endpoint side.

        Returns
        -------
        float
            Normalized squared-log expectation.
        """
        side = _validated_side(side)
        if side not in self._side_to_term:
            raise ValueError(f"{side} boundary basis is not enabled")
        if side not in self._log_square:
            self._log_square[side] = self._generic_log_product(side, side)
        return float(self._log_square[side])

    def log_cross(self, /):
        """Return ``E[log d_lower(Z) log d_upper(Z)]``.

        Returns
        -------
        float
            Normalized cross-boundary logarithmic expectation.

        Raises
        ------
        ValueError
            If both canonical boundary bases are not enabled.
        """
        if _LOWER not in self._side_to_term or _UPPER not in self._side_to_term:
            raise ValueError("both boundary bases must be enabled")
        if self._log_cross is None:
            self._log_cross = self._generic_log_product(_LOWER, _UPPER)
        return float(self._log_cross)

    def _integral(self, mode, k, t_index, /):
        """Run one shifted integral using complete allowed-boundary terms.

        Parameters
        ----------
        mode : int
            ``quad_integral`` weight mode.
        k : int
            Power of the canonical coordinate.
        t_index : int
            Boundary descriptor index for generalized modes.

        Returns
        -------
        float
            Shifted unnormalized integral.
        """
        return _ledger_integral(
            self._quad,
            "model-moment quadrature",
            mode=int(mode),
            k=int(k),
            t_index=int(t_index),
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
            points=self.state.quad_points,
        )

    def _generic_log_product(self, side_a, side_b, /):
        """Integrate one product of fixed logarithmic boundary functions.

        The two logarithmic factors are evaluated in the Cython callback used
        by the ordinary model-moment quadrature, so QUADPACK never re-enters
        Python at an integration node.

        Parameters
        ----------
        side_a, side_b : {"lower", "upper"}
            Canonical endpoint sides.

        Returns
        -------
        float
            Normalized model expectation of the log product.
        """
        idx_a = self._side_to_term[side_a]
        idx_b = self._side_to_term[side_b]
        value = _ledger_integral(
            self._quad,
            "model log-product quadrature",
            mode=2,
            k=0,
            t_index=idx_a,
            t_index2=idx_b,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
            points=self.state.quad_points,
        )
        return float(value / self.state.Z)


def _complete_boundary_terms(state, /):
    """Return the retained complete boundary descriptors for one state.

    Parameters
    ----------
    state : object
        Candidate state with precomputed complete quadrature descriptors.

    Returns
    -------
    terms : numpy.ndarray, shape (R, 3)
        Complete density/weight boundary descriptors.
    side_to_term : dict
        Mapping from canonical side name to row index in ``terms``.
    """
    return state.quad_terms, state._side_to_term


def _validated_side(side, /):
    """Validate and canonicalize a boundary-side name.

    Parameters
    ----------
    side : str
        Requested canonical side.

    Returns
    -------
    str
        ``"lower"`` or ``"upper"``.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    side = str(side)
    if side not in (_LOWER, _UPPER):
        raise ValueError("side must be 'lower' or 'upper'")
    return side
