"""Post-fit analytics: moment computation, affine updates, and mode finding.

This module provides helpers used *after* fitting to compute or update
derived quantities — moments, summary statistics, support bounds, and
exp-space raw moments — without re-running the optimizer.

Pipeline position
-----------------
This module is called from:

* ``gibbus._api.views._ExpSpaceView`` — for exp-space mode and moment computation.
* ``gibbus._api.component._Component._raw_moment_base`` — for on-demand base-space moment
  computation beyond the cached ``raw_moments`` array.

The helpers are pure functions of moments already computed from the fitted
basis potential; tolerances come from :mod:`._defaults`.
"""

from math import comb

import numpy as np

from .._defaults import EXP_NARROW_D2_THRESHOLD


def _stats_from_centered_moments(mean, mu2, mu3, mu4, /):
    """Build summary statistics from a mean and centered moments.

    Parameters
    ----------
    mean : float
        Distribution mean.
    mu2 : float
        Second centered moment.
    mu3 : float
        Third centered moment.
    mu4 : float
        Fourth centered moment.
    """
    mean = float(mean)
    var = float(mu2)
    if not np.isfinite(var) or var <= 0.0:
        raise RuntimeError(f"Degenerate variance encountered (var={var}).")
    std = float(np.sqrt(var))
    skew = float(mu3) / (std**3) if np.isfinite(mu3) else np.nan
    kurt = float(mu4) / (std**4) if np.isfinite(mu4) else np.nan
    return {"mean": mean, "var": var, "std": std, "skew": skew, "kurt": kurt}


def _central_moment_from_raw(get_raw, k, mean, /):
    """Compute a centered moment from raw moments in a well-scaled coordinate.

    Parameters
    ----------
    get_raw : callable
        Callable returning raw moments by non-negative integer order.
    k : int
        Centered-moment order.
    mean : float
        Mean in the same coordinate system as the raw moments.
    """
    kk = k
    m = float(mean)
    out = 0.0
    for i in range(kk + 1):
        raw = 1.0 if i == 0 else float(get_raw(i))
        out += comb(kk, i) * ((-m) ** (kk - i)) * raw
    return float(out)


def _cumulant_from_centered(get_centered, k, mean, /):
    """Compute a cumulant from centered moments.

    Parameters
    ----------
    get_centered : callable
        Callable returning the centered moment of a positive integer order.
    k : int
        Already-validated positive integer cumulant order.
    mean : float
        First cumulant.

    Returns
    -------
    float
        The *k*-th cumulant.

    """
    kk = k
    if kk == 1:
        return float(mean)

    # Centered moments avoid catastrophic cancellation after large affine
    # translations.  For n >= 2, the cumulant recurrence can be written
    # entirely in centered moments because mu_1 = 0.
    cumulants = {1: float(mean)}
    for n in range(2, kk + 1):
        value = float(get_centered(n))
        for j in range(2, n):
            mu = 1.0 if n - j == 0 else float(get_centered(n - j))
            value -= comb(n - 1, j - 1) * cumulants[j] * mu
        cumulants[n] = float(value)
    return cumulants[kk]


def _powaff_moment_from_z_moments(z_mom, alpha, beta, k, /):
    """Compute ``E[(alpha*Z + beta)^k]`` from internal-coordinate moments.

    Uses the binomial expansion::

        E[(alpha*Z + beta)^k] = sum_{i=0}^{k} C(k,i) * alpha^i * beta^(k-i) * E[Z^i]

    Parameters
    ----------
    z_mom : array_like, shape (k+1,)
        ``z_mom[i] = E[Z^i]`` for ``i = 0, ..., k``.
    alpha : float
        Scale coefficient in the affine transform ``alpha * Z + beta``.
    beta : float
        Offset coefficient in the affine transform ``alpha * Z + beta``.
    k : int
        Moment order.

    Returns
    -------
    float
    """
    kk = k
    s = 0.0
    for i in range(kk + 1):
        s += comb(kk, i) * (alpha**i) * (beta ** (kk - i)) * float(z_mom[i])
    return float(s)


def _moment_from_raw(get_raw, k, mean, std, central, standardized, /):
    """Derive a central or standardized moment from raw moments.

    Parameters
    ----------
    get_raw : callable
        ``get_raw(i) -> float`` returning the *i*-th raw moment.
    k : int
        Moment order.
    mean : float
        First raw moment (used for central/standardized computation).
    std : float
        Standard deviation (used for standardization).
    central : bool
        If ``True``, compute the central moment ``E[(X - mean)^k]``.
    standardized : bool
        If ``True``, divide the central moment by ``std**k``.

    Returns
    -------
    float

    Raises
    ------
    RuntimeError
        If ``standardized=True`` and ``std <= 0``.
    """
    kk = k

    if (not central) and (not standardized):
        return float(get_raw(kk))

    cm = 0.0
    m = float(mean)
    for i in range(kk + 1):
        mi = 1.0 if i == 0 else float(get_raw(i))
        cm += comb(kk, i) * ((-m) ** (kk - i)) * mi

    if not standardized:
        return float(cm)

    s = float(std)
    if not (s > 0.0):
        raise RuntimeError("Standardized moment is undefined because std <= 0.")
    return float(cm / (s**kk))


def _exp_stats_from_log_moments(log_moments, relative_centered_moment, subject, /):
    """Build exp-space summary statistics from the first four log raw moments.

    Shared by the single-component and mixture exp-space paths, which differ
    only in how they obtain ``log E[Y^k]`` and the narrow-law relative
    centered moments.

    Positive infinity is a legitimate log-moment result for transformed laws
    with a divergent moment, or for a finite mathematical moment that exceeds
    float64's representable range.  Such cases propagate to infinite
    dimensional summaries instead of being misclassified as a degenerate
    zero-variance law.  ``NaN`` and negative infinity remain numerical errors.

    For a narrow transformed law the centered combinations are differences of
    nearly equal ``O(1)`` log moments, so ``d2 = l2 - 2*l1`` loses most of its
    significant digits.  Below ``EXP_NARROW_D2_THRESHOLD`` the relative
    centered moments are therefore integrated directly instead of being
    reconstructed from ``expm1`` of canceling differences.

    Parameters
    ----------
    log_moments : sequence of float
        ``(l1, l2, l3, l4)``, the logarithms of ``E[Y^k]`` for ``k = 1..4``.
    relative_centered_moment : callable
        ``f(log_mean, k) -> float`` returning the ``k``-th centered moment in
        units of ``mean**k``.  Called only on the narrow-law branch.
    subject : str
        Noun phrase naming the law, used in error messages (for example
        ``"exp-space"`` or ``"exp-space mixture"``).

    Returns
    -------
    dict
        ``mean``, ``var``, ``std``, ``skew``, ``kurt``, ``log_mean``, ``cv``.

    Raises
    ------
    RuntimeError
        If a log moment is ``NaN`` or negative infinity, or if a finite
        relative variance is non-positive.
    """
    l1, l2, l3, l4 = (float(v) for v in log_moments)
    vals = np.asarray((l1, l2, l3, l4), dtype=np.float64)
    if np.any(np.isnan(vals)) or np.any(np.isneginf(vals)):
        raise RuntimeError(f"invalid log moment in {subject} statistics")

    # Once the first raw moment itself is infinite, finite-dimensional
    # variance and standardized shape are undefined.  Expose that fact
    # directly rather than evaluating inf-inf combinations below.
    if np.isposinf(l1):
        return {
            "mean": np.inf,
            "var": np.inf,
            "std": np.inf,
            "skew": np.nan,
            "kurt": np.nan,
            "log_mean": np.inf,
            "cv": np.nan,
        }

    d2 = float(l2 - 2.0 * l1)
    if d2 < EXP_NARROW_D2_THRESHOLD:
        # The center first: ``l1`` carries the log-moment quadrature's
        # relative error (3e-11), which moves the center of a law 9e-8 wide
        # enough to change its skewness by 1e-3; the first relative moment
        # is O(width) with that relative error.
        l1 = float(l1 + np.log1p(float(relative_centered_moment(l1, 1))))
        rel2 = float(relative_centered_moment(l1, 2))
        rel3 = float(relative_centered_moment(l1, 3))
        rel4 = float(relative_centered_moment(l1, 4))
        cv2 = rel2
        cv = _checked_cv(cv2, subject)
        skew = float(rel3 / (cv**3))
        kurt = float(rel4 / (cv**4))
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            cv2 = float(np.expm1(d2))
            cv = _checked_cv(cv2, subject)
            e2 = cv2
            e3 = float(np.expm1(l3 - 3.0 * l1))
            e4 = float(np.expm1(l4 - 4.0 * l1))
            skew = float((e3 - 3.0 * e2) / (cv**3))
            kurt = float((e4 - 4.0 * e3 + 6.0 * e2) / (cv**4))

        # When variance is finite but a higher raw moment diverges, the
        # corresponding standardized moment is +inf.  Evaluating the raw
        # formulas literally can produce inf-inf -> NaN even though the
        # positive exp-space tail makes the central moment unambiguously
        # divergent.
        if np.isfinite(l2) and np.isposinf(l3):
            skew = np.inf
            kurt = np.inf
        elif np.isfinite(l3) and np.isposinf(l4):
            kurt = np.inf

    log_max = float(np.log(np.finfo(np.float64).max))
    with np.errstate(over="ignore", under="ignore"):
        mean = float(np.exp(l1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_var = float(2.0 * l1 + np.log(cv2))
    var = float(np.exp(log_var)) if log_var <= log_max else np.inf
    log_std = 0.5 * log_var
    std = float(np.exp(log_std)) if log_std <= log_max else np.inf

    return {
        "mean": mean,
        "var": var,
        "std": std,
        "skew": skew,
        "kurt": kurt,
        "log_mean": l1,
        "cv": cv,
    }


def _checked_cv(cv2, subject, /):
    """Return ``sqrt(cv2)`` after rejecting a degenerate relative variance.

    Parameters
    ----------
    cv2 : float
        Squared coefficient of variation.
    subject : str
        Noun phrase naming the law, used in the error message.
    """
    if np.isnan(cv2) or cv2 <= 0.0:
        raise RuntimeError(f"Degenerate {subject} variance (relative var={cv2}).")
    if np.isposinf(cv2):
        return np.inf
    if not np.isfinite(cv2):
        raise RuntimeError(f"Invalid {subject} variance (relative var={cv2}).")
    return float(np.sqrt(cv2))


def _exp_moment_from_stats(k, stats, standardized, /):
    """Return a low-order exp-space central/standardized moment, or ``None``.

    Orders 0--4 are already available in the cached statistics, so they are
    answered directly rather than re-integrated.  ``None`` means the caller
    should fall back to its general moment machinery.

    Parameters
    ----------
    k : int
        Non-negative moment order.
    stats : Mapping
        Cached exp-space statistics with ``var``, ``std``, ``skew``, ``kurt``.
    standardized : bool
        Whether the standardized rather than central moment is requested.

    Returns
    -------
    float or None
    """
    if k > 4:
        return None
    if standardized:
        # Standardized shape is scale-free.  A huge location shift in log
        # space can overflow the dimensionful variance while leaving the
        # coefficient of variation and standardized moments finite.  Treat
        # only a genuinely divergent relative variance as undefined.
        if k >= 2 and not np.isfinite(float(stats["cv"])):
            return np.nan
        return (1.0, 0.0, 1.0, float(stats["skew"]), float(stats["kurt"]))[k]

    var = float(stats["var"])
    if k >= 3 and np.isfinite(float(stats["mean"])) and np.isposinf(var):
        # Exp-space laws are positive.  With a finite mean and infinite
        # variance, every higher central moment is also +inf; avoid the
        # indeterminate ``nan * inf`` reconstruction from standardized shape.
        return np.inf
    return (
        1.0,
        0.0,
        var,
        float(stats["skew"] * (stats["std"] ** 3)),
        float(stats["kurt"] * (stats["std"] ** 4)),
    )[k]
