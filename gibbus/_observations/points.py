"""Point observation provider for the likelihood geometry.

The point-data provider wraps fixed empirical sufficient statistics and therefore
contains no raw observation array. Interval observations are handled by their
model-dependent conditional-expectation machinery in the shared objective layer.
"""

from dataclasses import dataclass

import numpy as np

from .._model.spec import _LOGDIST, _LOWER, _POLY
from .empirical import _EmpiricalStats


@dataclass(frozen=True)
class _PointObservations:
    """Fixed sufficient-statistic observation provider for point data.

    Parameters
    ----------
    stats : _EmpiricalStats
        Canonical point-data summary built once before optimization.
    """

    stats: _EmpiricalStats

    def __post_init__(self):
        """Validate the provider's immutable empirical summary."""
        if not isinstance(self.stats, _EmpiricalStats):
            raise TypeError("stats must be an _EmpiricalStats")

    @property
    def effective_n(self):
        """Return the empirical effective sample size.

        Returns
        -------
        float
            Kish effective sample size carried by the summary.
        """
        return float(self.stats.effective_n)

    def potential_expectation(self, state, /):
        """Return the exact empirical expectation of the current potential.

        Parameters
        ----------
        state : object
            New-path candidate state.

        Returns
        -------
        float
            ``E_hat[q_theta(Z)]`` from fixed sufficient statistics.
        """
        lower_amp = (
            float(state.boundary_amplitudes[0])
            if state.spec.canonical_lower_a_index is not None else 0.0
        )
        upper_amp = (
            float(state.boundary_amplitudes[1])
            if state.spec.canonical_upper_a_index is not None else 0.0
        )
        return self.stats.potential_expectation(
            state.q_poly,
            lower_amplitude=lower_amp,
            upper_amplitude=upper_amp,
        )

    def first_expectations(self, partials, /):
        """Return empirical expectations of first potential partials.

        Parameters
        ----------
        partials : sequence of _PotentialPartial
            First parameter partial descriptors in optimizer order.

        Returns
        -------
        numpy.ndarray
            ``E_hat[h_i]`` for every supplied partial.
        """
        out = np.empty(len(partials), dtype=np.float64)
        for i, partial in enumerate(partials):
            if partial.kind == _POLY:
                out[i] = self.stats.poly_expectation(partial.coefficients)
            elif partial.kind == _LOGDIST:
                index = 0 if partial.boundary_side == _LOWER else 1
                value = float(self.stats.boundary_log[index])
                if not np.isfinite(value):
                    raise ValueError(
                        f"{partial.boundary_side} boundary-log statistic is unavailable"
                    )
                out[i] = value
            else:
                raise ValueError("unknown potential-partial descriptor")
        return out

    def polynomial_expectation(self, coefficients, /):
        """Return an empirical polynomial expectation.

        Parameters
        ----------
        coefficients : array_like
            Ascending power-basis coefficients.

        Returns
        -------
        float
            Exact coefficient/moment contraction.
        """
        return self.stats.poly_expectation(coefficients)
