"""Tail-accurate survival, hazard, and reliability calculations."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .._defaults import EXPECT_MAX_RELATIVE_ERROR, SF_HANDOVER_P, _reraise_if_debug
from .._spectral.tail import exact_tail_log_cdf


def _vectorize_scalar(func, x, *args):
    """Apply a scalar evaluator while preserving scalar/array convention.

    Parameters
    ----------
    func : callable
        Scalar function accepting ``(*args, value)``.
    x : float or array_like
        Input coordinate(s).
    *args : tuple
        Leading arguments forwarded to *func*.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    out = np.array([func(*args, float(v)) for v in flat], dtype=np.float64)
    out = out.reshape(arr.shape)
    return float(out) if scalar else out


def log_sf_hybrid(
    potential,
    spectral_cdf,
    x,
    upper_endpoint,
    /,
    *,
    lower_endpoint=-np.inf,
    log_tail_mass=None,
    log_tail_masses=None,
):
    """Evaluate log survival using spectral body values and exact tail quadrature.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    spectral_cdf : callable
        CDF evaluator accurate on an absolute probability scale.
    x : float or array_like
        Evaluation coordinates.
    upper_endpoint : float
        Upper support endpoint.
    lower_endpoint : float, optional
        Lower support endpoint, used for exact boundary semantics.
    log_tail_mass : callable or None, optional
        Exact tail-mass adapter used below the spectral handover.
    log_tail_masses : callable or None, optional
        Batched form ``(x_array, endpoint, upper=...) -> log masses``; when
        given it evaluates every tail anchor in one call.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    lower = float(lower_endpoint)
    upper = float(upper_endpoint)
    out = np.full(flat.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(flat)
    out[valid & (flat <= lower)] = 0.0
    out[valid & (flat >= upper)] = -np.inf
    interior = np.flatnonzero(valid & (flat > lower) & (flat < upper))
    if interior.size:
        values = np.asarray(spectral_cdf(flat[interior]), dtype=np.float64).reshape(-1)
        body = (1.0 - values) > SF_HANDOVER_P
        if np.any(body):
            out[interior[body]] = np.log1p(-values[body])
        tail = interior[~body]
        if tail.size and log_tail_masses is not None:
            out[tail] = log_tail_masses(flat[tail], upper, upper=True)
            tail = tail[:0]
        for j in tail:
            out[j] = (
                float(log_tail_mass(float(flat[j]), upper, upper=True))
                if log_tail_mass is not None
                else exact_tail_log_cdf(potential, float(flat[j]), upper, upper=True)
            )
    out = out.reshape(arr.shape)
    return float(out) if scalar else out


def cdf_hybrid(
    potential,
    spectral_cdf,
    x,
    lower_endpoint,
    /,
    *,
    upper_endpoint=np.inf,
    log_tail_mass=None,
    log_tail_masses=None,
):
    """Evaluate the CDF with exact lower-tail quadrature below the handover.

    The spectral representation is retained in the body so ordinary CDF
    evaluation keeps its fast absolute-probability path.  Only values at or
    below ``SF_HANDOVER_P`` are replaced by exponentiated exact log-tail mass,
    avoiding the spectral antiderivative's absolute-error floor in the extreme
    lower tail.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    spectral_cdf : callable
        Fast body CDF evaluator.
    x : float or array_like
        Evaluation coordinates.
    lower_endpoint, upper_endpoint : float
        Support endpoints.
    log_tail_mass : callable or None, optional
        Scalar exact lower-tail log-mass evaluator.
    log_tail_masses : callable or None, optional
        Batched exact lower-tail log-mass evaluator.

    Returns
    -------
    float or numpy.ndarray
        CDF values in ``[0, 1]``.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    values = np.asarray(spectral_cdf(arr), dtype=np.float64)
    flat_x = np.atleast_1d(arr).reshape(-1)
    flat_v = np.atleast_1d(values).reshape(-1)
    lower = float(lower_endpoint)
    upper = float(upper_endpoint)
    tail_mask = (
        ~np.isnan(flat_x)
        & (flat_x > lower)
        & (flat_x < upper)
        & (flat_v <= SF_HANDOVER_P)
    )
    if not np.any(tail_mask):
        return float(values) if scalar else values

    out = flat_v.copy()
    tail = np.flatnonzero(tail_mask)
    if log_tail_masses is not None:
        with np.errstate(under="ignore"):
            out[tail] = np.exp(log_tail_masses(flat_x[tail], lower, upper=False))
        tail = tail[:0]
    for j in tail:
        log_mass = (
            float(log_tail_mass(float(flat_x[j]), lower, upper=False))
            if log_tail_mass is not None
            else exact_tail_log_cdf(potential, float(flat_x[j]), lower, upper=False)
        )
        with np.errstate(under="ignore"):
            out[j] = np.exp(log_mass)
    out = np.clip(out, 0.0, 1.0).reshape(arr.shape)
    return float(out) if scalar else out


def log_cdf_hybrid(
    potential,
    spectral_cdf,
    x,
    lower_endpoint,
    /,
    *,
    upper_endpoint=np.inf,
    log_tail_mass=None,
    log_tail_masses=None,
):
    """Evaluate log CDF using spectral body values and exact tail quadrature.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    spectral_cdf : callable
        CDF evaluator accurate on an absolute probability scale.
    x : float or array_like
        Evaluation coordinates.
    lower_endpoint : float
        Lower support endpoint.
    upper_endpoint : float, optional
        Upper support endpoint, used for exact boundary semantics.
    log_tail_mass : callable or None, optional
        Exact tail-mass adapter used below the spectral handover.
    log_tail_masses : callable or None, optional
        Batched form ``(x_array, endpoint, upper=...) -> log masses``; when
        given it evaluates every tail anchor in one call.
    """
    arr = np.asarray(x, dtype=np.float64)
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    lower = float(lower_endpoint)
    upper = float(upper_endpoint)
    out = np.full(flat.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(flat)
    out[valid & (flat <= lower)] = -np.inf
    out[valid & (flat >= upper)] = 0.0
    interior = np.flatnonzero(valid & (flat > lower) & (flat < upper))
    if interior.size:
        values = np.asarray(spectral_cdf(flat[interior]), dtype=np.float64).reshape(-1)
        body = values > SF_HANDOVER_P
        if np.any(body):
            out[interior[body]] = np.log(values[body])
        tail = interior[~body]
        if tail.size and log_tail_masses is not None:
            out[tail] = log_tail_masses(flat[tail], lower, upper=False)
            tail = tail[:0]
        for j in tail:
            out[j] = (
                float(log_tail_mass(float(flat[j]), lower, upper=False))
                if log_tail_mass is not None
                else exact_tail_log_cdf(potential, float(flat[j]), lower, upper=False)
            )
    out = out.reshape(arr.shape)
    return float(out) if scalar else out


def _quad_with_ledger(func, a, b, /, *, epsabs, epsrel, limit, context):
    """Run adaptive quadrature without leaking SciPy integration warnings.

    ``full_output=1`` makes SciPy return QUADPACK's diagnostic message
    instead of emitting ``IntegrationWarning``.  No process-global warning
    state is touched, so this stays safe when several threads evaluate a
    fitted model concurrently (``warnings.catch_warnings`` is not).

    Parameters
    ----------
    func : callable
        Scalar integrand.
    a, b : float
        Integration limits.
    epsabs, epsrel : float
        SciPy quadrature tolerances.
    limit : int
        Maximum number of adaptive subintervals.
    context : str
        Failure-ledger context for any reported integration problem.
    """
    result = quad(func, a, b, epsabs=epsabs, epsrel=epsrel, limit=limit, full_output=1)
    if len(result) > 3:
        _reraise_if_debug(RuntimeError(str(result[3])), context)
    return float(result[0]), float(result[1])


def mean_residual_life(potential, logsf, support, mean, mode, x, /):
    """Compute mean residual life from a single density quadrature.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    logsf : callable
        Tail-accurate log-survival evaluator.
    support : array_like, shape (2,)
        Active support.
    mean : float
        Mean in the same coordinate system.
    mode : float or array_like
        Density mode(s) in the same coordinate system. Modes at or above each
        threshold are candidates for the density anchor used by quadrature.
    x : float or array_like
        Conditioning threshold(s).
    """
    lower, upper = map(float, support)
    modes = np.asarray(mode, dtype=np.float64).reshape(-1)
    if modes.size == 0 or np.any(~np.isfinite(modes)):
        raise ValueError("mode must contain at least one finite coordinate")
    modes = np.clip(modes, lower, upper)

    def survival_fallback(start, ls_start):
        """Retain the old survival integral only for pathological scaling."""
        value, err = _quad_with_ledger(
            lambda t: float(np.exp(float(logsf(t)) - ls_start)),
            start,
            upper,
            epsabs=1e-11,
            epsrel=1e-10,
            limit=200,
            context="mean residual life fallback quadrature",
        )
        if err > EXPECT_MAX_RELATIVE_ERROR * max(abs(value), 1.0):
            raise RuntimeError("mean residual life quadrature did not converge")
        return float(value)

    def one(v):
        if np.isnan(v):
            return np.nan
        if v <= lower:
            return float(mean - v)
        if v >= upper:
            return 0.0

        lower_bound = float(mean - v)
        if np.isposinf(lower_bound):
            return np.inf
        ls = float(logsf(v))

        candidates = np.concatenate(([v], modes[modes >= v]))
        q_candidates = np.asarray(
            [float(potential(candidate, 0)) for candidate in candidates],
            dtype=np.float64,
        )
        finite = np.isfinite(q_candidates)
        if not np.any(finite):
            result = survival_fallback(v, ls)
        else:
            finite_indices = np.flatnonzero(finite)
            best = finite_indices[int(np.argmin(q_candidates[finite]))]
            anchor = float(candidates[best])
            q_anchor = float(q_candidates[best])

            # Let scale = S(v) / f(anchor).  With
            #   t = anchor + scale*z,
            # the conditional tail density becomes simply
            #   exp(log f(t) - log f(anchor)) dz.
            # This avoids evaluating logsf inside the quadrature entirely;
            # for upper-tail points it reduces to reciprocal-hazard scaling.
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scale = float(np.exp(ls + q_anchor))
            if not np.isfinite(scale) or scale <= 0.0:
                result = survival_fallback(v, ls)
            else:
                z0 = float((v - anchor) / scale)
                z1 = (
                    np.inf
                    if not np.isfinite(upper)
                    else float(max(z0, (upper - anchor) / scale))
                )

                def relative_first_moment(z):
                    t = anchor + scale * z
                    q = float(potential(t, 0))
                    if not np.isfinite(q):
                        return 0.0
                    with np.errstate(under="ignore"):
                        relative_density = float(np.exp(q_anchor - q))
                    return float((z - z0) * relative_density)

                value, err = _quad_with_ledger(
                    relative_first_moment,
                    z0,
                    z1,
                    epsabs=1e-11,
                    epsrel=1e-10,
                    limit=200,
                    context="mean residual life density quadrature",
                )
                result = float(scale * value)
                if err * scale > EXPECT_MAX_RELATIVE_ERROR * max(
                    abs(result), scale * 1e-12
                ):
                    raise RuntimeError("mean residual life quadrature did not converge")

        tolerance = EXPECT_MAX_RELATIVE_ERROR * max(1.0, abs(result), abs(lower_bound))
        if result + tolerance < lower_bound:
            raise RuntimeError(
                "mean residual life violates the conditional-mean lower bound"
            )
        return float(result)

    return _vectorize_scalar(lambda v: one(v), x)


def residual_entropy(potential, logsf, support, x, /):
    """Compute residual entropy of the conditional tail distribution.

    Parameters
    ----------
    potential : callable
        Normalized negative-log density evaluator.
    logsf : callable
        Tail-accurate log-survival evaluator.
    support : array_like, shape (2,)
        Active support.
    x : float or array_like
        Conditioning threshold(s).
    """
    lower, upper = map(float, support)

    def one(v):
        if np.isnan(v):
            return np.nan
        if v >= upper:
            return -np.inf
        start = max(v, lower)
        ls = float(logsf(start))
        if np.isneginf(ls):
            return -np.inf

        def integrand(t):
            q = float(potential(t, 0))
            if not np.isfinite(q):
                return 0.0
            return q * float(np.exp(-q - ls))

        value, err = _quad_with_ledger(
            integrand,
            start,
            upper,
            epsabs=1e-10,
            epsrel=1e-9,
            limit=200,
            context="residual entropy quadrature",
        )
        if err > EXPECT_MAX_RELATIVE_ERROR * max(abs(value), 1.0):
            raise RuntimeError("residual entropy quadrature did not converge")
        return float(ls + value)

    return _vectorize_scalar(lambda v: one(v), x)


def _validate_probabilities(p, name):
    """Validate ordinary probabilities.

    Parameters
    ----------
    p : float or array_like
        Candidate probabilities.
    name : str
        Public method name for error messages.
    """
    arr = np.asarray(p, dtype=np.float64)
    if np.any(((arr < 0.0) | (arr > 1.0) | np.isinf(arr)) & ~np.isnan(arr)):
        raise ValueError(f"{name} is defined for probabilities in [0, 1]")
    return arr


def _validate_log_probabilities(log_p, name):
    """Validate logarithmic probabilities.

    Parameters
    ----------
    log_p : float or array_like
        Candidate log probabilities.
    name : str
        Public method name for error messages.
    """
    arr = np.asarray(log_p, dtype=np.float64)
    if np.any((arr > 0.0) & ~np.isnan(arr)):
        raise ValueError(f"{name} requires log_p <= 0")
    return arr


def isf(potential, ppf, support, p, /, *, log_tail_mass=None):
    """Invert the survival probability without forming ``1-p`` in deep tails.

    Parameters
    ----------
    potential : callable
        Negative-log density evaluator.
    ppf : callable
        Ordinary quantile evaluator used for body probabilities and tail seeds.
    support : array_like, shape (2,)
        Distribution support.
    p : float or array_like
        Survival probabilities in ``[0, 1]``.
    log_tail_mass : callable or None, optional
        Exact tail-mass adapter used for deep-tail inversion.
    """
    from .._spectral.tail import invert_tail

    arr = _validate_probabilities(p, "isf")
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    lo, hi = map(float, support)
    seed_hi = None
    out = np.empty_like(flat)
    for i, prob in enumerate(flat):
        if np.isnan(prob):
            out[i] = np.nan
        elif prob == 0.0:
            out[i] = hi
        elif prob == 1.0:
            out[i] = lo
        elif prob < SF_HANDOVER_P:
            if seed_hi is None:
                seed_hi = float(np.asarray(ppf(1.0 - SF_HANDOVER_P)))
            out[i] = invert_tail(
                potential,
                float(np.log(prob)),
                hi,
                seed_hi,
                upper=True,
                log_tail_mass=log_tail_mass,
            )
        else:
            out[i] = float(ppf(1.0 - prob))
    out = out.reshape(arr.shape)
    return float(out) if scalar else out


def logppf(potential, ppf, support, log_p, /, *, log_tail_mass=None):
    """Invert a logarithmic CDF probability directly.

    Parameters
    ----------
    potential : callable
        Negative-log density evaluator.
    ppf : callable
        Ordinary quantile evaluator used in the body and for seeds.
    support : array_like, shape (2,)
        Distribution support.
    log_p : float or array_like
        Log CDF probabilities, no greater than zero.
    log_tail_mass : callable or None, optional
        Exact tail-mass adapter used for deep-tail inversion.
    """
    from .._spectral.tail import invert_tail
    from .logspace import log1mexp

    arr = _validate_log_probabilities(log_p, "logppf")
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    lo, hi = map(float, support)
    seed_lo = None
    seed_hi = None
    log_handover = float(np.log(SF_HANDOVER_P))
    log_upper_body = float(np.log1p(-SF_HANDOVER_P))
    out = np.empty_like(flat)
    for i, lp in enumerate(flat):
        if np.isnan(lp):
            out[i] = np.nan
        elif np.isneginf(lp):
            out[i] = lo
        elif lp == 0.0:
            out[i] = hi
        elif lp < log_handover:
            if seed_lo is None:
                seed_lo = float(np.asarray(ppf(SF_HANDOVER_P)))
            out[i] = invert_tail(
                potential,
                float(lp),
                lo,
                seed_lo,
                upper=False,
                log_tail_mass=log_tail_mass,
            )
        elif lp >= log_upper_body:
            if seed_hi is None:
                seed_hi = float(np.asarray(ppf(1.0 - SF_HANDOVER_P)))
            lsf = float(log1mexp(lp))
            out[i] = invert_tail(
                potential, lsf, hi, seed_hi, upper=True, log_tail_mass=log_tail_mass
            )
        else:
            out[i] = float(ppf(np.exp(lp)))
    out = out.reshape(arr.shape)
    return float(out) if scalar else out


def logisf(potential, ppf, support, log_p, /, *, log_tail_mass=None):
    """Invert a logarithmic survival probability directly.

    Parameters
    ----------
    potential : callable
        Negative-log density evaluator.
    ppf : callable
        Ordinary quantile evaluator used in the body and for seeds.
    support : array_like, shape (2,)
        Distribution support.
    log_p : float or array_like
        Log survival probabilities, no greater than zero.
    log_tail_mass : callable or None, optional
        Exact tail-mass adapter used for deep-tail inversion.
    """
    from .._spectral.tail import invert_tail
    from .logspace import log1mexp

    arr = _validate_log_probabilities(log_p, "logisf")
    scalar = arr.ndim == 0
    flat = np.atleast_1d(arr).reshape(-1)
    lo, hi = map(float, support)
    seed_lo = None
    seed_hi = None
    log_handover = float(np.log(SF_HANDOVER_P))
    log_upper_body = float(np.log1p(-SF_HANDOVER_P))
    out = np.empty_like(flat)
    for i, lp in enumerate(flat):
        if np.isnan(lp):
            out[i] = np.nan
        elif np.isneginf(lp):
            out[i] = hi
        elif lp == 0.0:
            out[i] = lo
        elif lp < log_handover:
            if seed_hi is None:
                seed_hi = float(np.asarray(ppf(1.0 - SF_HANDOVER_P)))
            out[i] = invert_tail(
                potential,
                float(lp),
                hi,
                seed_hi,
                upper=True,
                log_tail_mass=log_tail_mass,
            )
        elif lp >= log_upper_body:
            if seed_lo is None:
                seed_lo = float(np.asarray(ppf(SF_HANDOVER_P)))
            lcdf = float(log1mexp(lp))
            out[i] = invert_tail(
                potential, lcdf, lo, seed_lo, upper=False, log_tail_mass=log_tail_mass
            )
        else:
            out[i] = float(ppf(1.0 - np.exp(lp)))
    out = out.reshape(arr.shape)
    return float(out) if scalar else out
