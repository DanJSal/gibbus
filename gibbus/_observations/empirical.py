"""Fixed empirical sufficient statistics.

Point-data likelihoods depend on the observations only through power moments
and any active zero-offset boundary-log statistics.  They are stored once, in
the fixed canonical fitting coordinate, by :class:`_EmpiricalStats` (the
compiled builder is ``_fit._mixture_kernels.empirical_point_stats``).

Finite interval rows admit a parameter-free pseudo-summary: treat the latent
value as uniform inside each censoring interval and average the same
statistics exactly.  It seeds interval fits; the final interval likelihood is
the observed-data one.
"""

from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class _EmpiricalStats:
    """Fixed sufficient statistics and sampling metadata for point data.

    Parameters
    ----------
    moments : numpy.ndarray, shape (K + 1,)
        Normalised power moments with ``moments[k] = E_hat[z**k]`` and
        ``moments[0] == 1`` up to floating-point roundoff.
    boundary_log : numpy.ndarray, shape (2,)
        Expectations of the fixed basis functions ``-log(z-L)`` and
        ``-log(U-z)``.  Inactive slots are ``nan``.
    support : tuple of (float, float)
        Canonical fitting-coordinate support.
    total_weight : float
        Sum of the input weights before normalisation, or the number of
        observations for unweighted data.
    effective_n : float
        Kish effective sample size ``(sum w)^2 / sum(w^2)``.  Equal weights
        give the ordinary observation count.
    n_observations : int
        Number of point observations used to build the summary.
    moment_effective_n : numpy.ndarray or None
        Participation effective sample size for each absolute power statistic.
        ``moment_effective_n[k]`` approaches one when ``|z|**k`` is dominated
        by a single weighted observation.  ``None`` is allowed for synthetic
        summaries such as interval pseudo-statistics where observation-level
        participation is not defined.
    """

    moments: np.ndarray
    boundary_log: np.ndarray
    support: tuple[float, float]
    total_weight: float
    effective_n: float
    n_observations: int
    moment_effective_n: np.ndarray | None = None

    @property
    def max_order(self):
        """Return the highest available power-moment order.

        Returns
        -------
        int
            Largest ``k`` for which ``E_hat[z**k]`` is stored.
        """
        return int(self.moments.size - 1)

    def poly_expectation(self, coefficients, /):
        """Return the empirical expectation of a power-basis polynomial.

        Parameters
        ----------
        coefficients : array_like
            Ascending coefficients ``c[k]`` for ``sum_k c[k] z**k``.

        Returns
        -------
        float
            Exact coefficient/moment contraction in floating-point arithmetic.

        Raises
        ------
        ValueError
            If the polynomial degree exceeds the stored moment order.
        """
        c = np.asarray(coefficients, dtype=np.float64).reshape(-1)
        if c.size > self.moments.size:
            raise ValueError(
                "polynomial degree exceeds the available empirical moment order"
            )
        return float(np.dot(c, self.moments[:c.size]))

    def potential_expectation(
        self, coefficients, /, *, lower_amplitude=0.0, upper_amplitude=0.0
    ):
        """Return ``E_hat[q]`` for polynomial plus fixed log-boundary terms.

        Parameters
        ----------
        coefficients : array_like
            Ascending power-basis coefficients of the polynomial potential.
        lower_amplitude : float, optional
            Coefficient multiplying ``-log(z-L)``.
        upper_amplitude : float, optional
            Coefficient multiplying ``-log(U-z)``.

        Returns
        -------
        float
            Empirical expectation of the represented potential.

        Raises
        ------
        ValueError
            If a nonzero boundary amplitude is requested for an inactive
            boundary statistic.
        """
        out = self.poly_expectation(coefficients)
        amplitudes = (float(lower_amplitude), float(upper_amplitude))
        for idx, amp in enumerate(amplitudes):
            if amp == 0.0:
                continue
            stat = float(self.boundary_log[idx])
            if not np.isfinite(stat):
                side = "lower" if idx == 0 else "upper"
                raise ValueError(f"{side} boundary-log statistic is unavailable")
            out += amp * stat
        return float(out)


    def power_covariance(self, max_order, /):
        """Return the empirical covariance of ``1,z,...,z**max_order``.

        Parameters
        ----------
        max_order : int
            Highest power statistic included.  Stored moments through twice
            this order are required.

        Returns
        -------
        numpy.ndarray
            Plug-in covariance matrix of the power statistics.

        Raises
        ------
        ValueError
            If the required double-order moments are unavailable.
        """
        K = int(max_order)
        if K < 0 or 2 * K > self.max_order:
            raise ValueError("need stored moments through order 2*max_order")
        index = np.arange(K + 1)
        second = self.moments[index[:, None] + index[None, :]]
        mean = self.moments[:K + 1]
        return second - mean[:, None] * mean[None, :]

    def moment_participation(self, order, /):
        """Return the participation effective sample size of ``|z|**order``.

        Parameters
        ----------
        order : int
            Nonnegative power order.

        Returns
        -------
        float
            Participation effective sample size, or ``nan`` when this summary
            was constructed without observation-level participation metadata.

        Raises
        ------
        ValueError
            If the order is outside the stored range.
        """
        k = int(order)
        if k < 0 or k > self.max_order:
            raise ValueError("moment order is outside the stored range")
        if self.moment_effective_n is None:
            return float("nan")
        return float(self.moment_effective_n[k])

    def approximate_mean_variance(self, order, /):
        """Estimate the sampling variance of the empirical ``k``-th moment.

        This uses the model-free plug-in quantity

        ``(E_hat[z**(2k)] - E_hat[z**k]**2) / effective_n``.

        It is used only as reliability metadata for degree selection.  For
        nonuniform relative weights, treat it as an effective-sample-size
        approximation rather than a separately calibrated uncertainty model.

        Parameters
        ----------
        order : int
            Moment order ``k``.

        Returns
        -------
        float
            Non-negative approximate variance of the empirical mean statistic.

        Raises
        ------
        ValueError
            If ``2*k`` exceeds the stored moment order or ``k`` is negative.
        """
        k = int(order)
        if k < 0 or 2 * k > self.max_order:
            raise ValueError("need stored moments through order 2*k")
        raw_var = float(self.moments[2 * k] - self.moments[k] ** 2)
        # Roundoff can make an exactly zero variance slightly negative.
        raw_var = max(raw_var, 0.0)
        return float(raw_var / self.effective_n)


def _normalised_weights(n, weights, subject="point", /):
    """Return normalised weights plus total and effective sample size.

    Shared by the point and interval observation builders: the arithmetic is
    identical, only the noun in the error messages differs.

    Normalisation runs in units of the largest weight so that neither the raw
    sum nor the sum of squares can overflow merely because every relative
    weight shares a large common scale.

    Parameters
    ----------
    n : int
        Number of observations.
    weights : array_like or None
        Non-negative finite weights.  ``None`` means equal weights.
    subject : str, optional
        Noun naming the observation kind in error messages, for example
        ``"point"`` or ``"interval"``.

    Returns
    -------
    weights : numpy.ndarray, shape (n,)
        Normalised weights summing to one.
    total_weight : float
        Pre-normalisation total weight.
    effective_n : float
        Kish effective sample size.

    Raises
    ------
    ValueError
        If the weight vector is invalid or has no positive total mass.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"at least one {subject} observation is required")
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64), float(n), float(n)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != n:
        raise ValueError(
            f"weights must have one entry per {subject} observation")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    wmax = float(np.max(w))
    if not (wmax > 0.0):
        raise ValueError("total observation weight must be positive")
    scaled = w / wmax
    scaled_total = float(np.sum(scaled, dtype=np.float64))
    if not (scaled_total > 0.0 and np.isfinite(scaled_total)):
        raise ValueError("total observation weight must be positive")
    norm = np.ascontiguousarray(scaled / scaled_total, dtype=np.float64)
    sum_sq_norm = float(np.dot(norm, norm))
    if not (sum_sq_norm > 0.0 and np.isfinite(sum_sq_norm)):
        raise ValueError("squared observation weight must be positive and finite")
    with np.errstate(over="ignore"):
        total = float(wmax * scaled_total)
    return norm, total, float(1.0 / sum_sq_norm)


def _uniform_power_moments(intervals, weights, max_order, /):
    """Return exact power moments for a mixture of finite uniform intervals.

    The calculation expands each interval around its midpoint.  For
    ``Z = m + Y`` with ``Y ~ Uniform[-h,h]``, odd central moments vanish and
    ``E[Y**j] = h**j/(j+1)`` for even ``j``.  This avoids the catastrophic
    cancellation in ``(U**(k+1)-L**(k+1))/(U-L)`` for narrow intervals.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Finite ordered canonical intervals.
    weights : numpy.ndarray, shape (R,)
        Normalized row weights.
    max_order : int
        Highest requested moment order.

    Returns
    -------
    numpy.ndarray, shape (max_order + 1,)
        Mixture moments from order zero through ``max_order``.

    Raises
    ------
    ValueError
        If shapes/order are invalid or ``max_order`` is negative.
    FloatingPointError
        If a requested moment becomes non-finite.
    """
    x = np.asarray(intervals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] != w.size:
        raise ValueError("intervals/weights have incompatible shapes")
    if not np.all(np.isfinite(x)) or np.any(x[:, 0] > x[:, 1]):
        raise ValueError("uniform pseudo-moments require finite ordered intervals")

    mid = 0.5 * (x[:, 0] + x[:, 1])
    half = 0.5 * (x[:, 1] - x[:, 0])
    out = np.empty(order + 1, dtype=np.float64)
    out[0] = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(1, order + 1):
            row = np.zeros(x.shape[0], dtype=np.float64)
            for j in range(0, k + 1, 2):
                row += (
                    float(comb(k, j))
                    * np.power(mid, k - j)
                    * np.power(half, j)
                    / float(j + 1)
                )
            value = float(np.dot(w, row))
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"uniform interval pseudo-moment of order {k} is non-finite"
                )
            out[k] = value
    return out


def _antiderivative_neg_log_distance(distance, /):
    """Return ``t * (1-log(t))`` with its continuous value zero at ``t=0``.

    Parameters
    ----------
    distance : numpy.ndarray
        Non-negative boundary distances.

    Returns
    -------
    numpy.ndarray
        Antiderivative values for ``-log(t)``.

    Raises
    ------
    ValueError
        If any distance is negative.
    """
    t = np.asarray(distance, dtype=np.float64)
    if np.any(t < 0.0):
        raise ValueError("boundary distance must be non-negative")
    out = np.zeros_like(t)
    positive = t > 0.0
    out[positive] = t[positive] * (1.0 - np.log(t[positive]))
    return out


def _uniform_boundary_log_expectation(
    intervals, weights, endpoint, side, /, *, point_distances=None
):
    """Average one fixed ``-log(distance)`` basis over finite intervals.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Finite ordered canonical rows.
    weights : numpy.ndarray, shape (R,)
        Normalized row weights.
    endpoint : float
        Finite canonical support endpoint.
    side : {"lower", "upper"}
        Which boundary distance to use.

    point_distances : numpy.ndarray or None
        Optional preserved exact-point distances from the active endpoint.

    Returns
    -------
    float
        Weighted uniform-within-interval expectation.

    Raises
    ------
    ValueError
        If the endpoint/interval geometry is invalid or an exact zero-width
        observation lies on the active endpoint, where the point log statistic
        diverges.
    """
    x = np.asarray(intervals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    ep = float(endpoint)
    if not np.isfinite(ep):
        raise ValueError("boundary-log pseudo-statistic requires a finite endpoint")
    width = x[:, 1] - x[:, 0]

    if side == "lower":
        lo = x[:, 0] - ep
        hi = x[:, 1] - ep
    elif side == "upper":
        lo = ep - x[:, 1]
        hi = ep - x[:, 0]
    else:
        raise ValueError("side must be 'lower' or 'upper'")
    if np.any(lo < 0.0) or np.any(hi < lo):
        raise ValueError("interval lies outside the active boundary endpoint")

    row = np.empty(x.shape[0], dtype=np.float64)
    point = width == 0.0
    if np.any(point):
        d = np.asarray(lo[point], dtype=np.float64)
        if point_distances is not None:
            supplied = np.asarray(point_distances, dtype=np.float64).reshape(-1)
            if supplied.size != x.shape[0]:
                raise ValueError("point_distances must match interval rows")
            override = supplied[point]
            use = np.isfinite(override)
            d = np.where(use, override, d)
        if np.any(d <= 0.0):
            raise ValueError(
                "zero-width observation on an active boundary has divergent log statistic"
            )
        row[point] = -np.log(d)
    ordinary = ~point
    if np.any(ordinary):
        F_hi = _antiderivative_neg_log_distance(hi[ordinary])
        F_lo = _antiderivative_neg_log_distance(lo[ordinary])
        row[ordinary] = (F_hi - F_lo) / width[ordinary]

    value = float(np.dot(w, row))
    if not np.isfinite(value):
        raise FloatingPointError("uniform boundary-log pseudo-statistic is non-finite")
    return value


def _uniform_interval_empirical_stats(observations, max_order, /, *,
                                      has_lower_log=False, has_upper_log=False):
    """Build point-style sufficient statistics from finite interval uniforms.

    Parameters
    ----------
    observations : _IntervalObservations
        Canonical finite interval rows and normalized weights.
    max_order : int
        Highest retained power moment.
    has_lower_log, has_upper_log : bool, optional
        Whether the corresponding fixed zero-offset boundary basis is active.

    Returns
    -------
    _EmpiricalStats
        Pseudo-statistics consumable by the ordinary point objective.

    Raises
    ------
    NotImplementedError
        If any interval has an infinite endpoint.
    ValueError
        If active boundary geometry is invalid.
    """
    if observations.has_infinite_rows:
        raise NotImplementedError(
            "uniform interval pseudo-statistics are undefined for infinite censoring rows"
        )
    if observations.support is None:
        raise ValueError("interval pseudo-statistics require a canonical support")

    moments = _uniform_power_moments(
        observations.intervals, observations.weights, max_order
    )
    boundary = np.full(2, np.nan, dtype=np.float64)
    lower, upper = map(float, observations.support)
    if has_lower_log:
        boundary[0] = _uniform_boundary_log_expectation(
            observations.intervals, observations.weights, lower, "lower",
            point_distances=observations.point_lower_distance,
        )
    if has_upper_log:
        boundary[1] = _uniform_boundary_log_expectation(
            observations.intervals, observations.weights, upper, "upper",
            point_distances=observations.point_upper_distance,
        )
    return _EmpiricalStats(
        moments=moments,
        boundary_log=boundary,
        support=(lower, upper),
        total_weight=float(observations.total_weight),
        effective_n=float(observations.effective_n),
        n_observations=int(observations.n_observations),
    )
