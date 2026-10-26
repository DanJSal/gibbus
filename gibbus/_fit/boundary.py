"""Data-driven boundary-log terms and their identification diagnostics.

The one-sided likelihood-ratio selection of endpoint logs.

A finite endpoint may carry the term ``-a log(z - L)`` (or ``-a log(U - z)``)
with ``a >= 0``.  When the caller leaves the choice open, the term is kept
only if the data resolve it: the fit with the term is compared with the fit
without it at the same bulk degree by a one-sided likelihood-ratio
test.  Under ``a = 0`` the statistic is asymptotically the 50:50 mixture of
``chi2(0)`` and ``chi2(1)`` (the parameter sits on the boundary of its
range), so ``p = chi2(1).sf(LR) / 2``.  Removing the term also removes the
curvature it lends the potential near the endpoint, so the reference is a
calibration device, like the omitted-information test for the degree, not
an exact law.

A kept amplitude also gets a standard error from the curvature of the
likelihood on the solver's final face (``basis_state``); an amplitude within
two standard errors of zero is reported as weakly identified.
"""

import numpy as np
from scipy.stats import chi2

AUTO = "auto"
"""Boundary policy meaning "decide from the data"."""

_SIDES = ("lower", "upper")


def _effective_n(n_rows, weights, /):
    """Return the Kish effective sample size of normalized row weights.

    Parameters
    ----------
    n_rows : int
        Number of observation rows.
    weights : numpy.ndarray, shape (n_rows,), dtype float64 or None
        Boundary-normalized row weights, or ``None`` for equal weights.
    """
    if weights is None:
        return float(n_rows)
    return float(1.0 / np.dot(weights, weights))


def _boundary_p_value(nll_without, nll_with, effective_n, /):
    """One-sided likelihood-ratio p-value for keeping a boundary term.

    Parameters
    ----------
    nll_without, nll_with : float
        Per-unit-weight negative log likelihoods without and with the term.
    effective_n : float
        Effective sample size converting per-unit-weight NLLs to totals.
    """
    lr = 2.0 * float(effective_n) * (float(nll_without) - float(nll_with))
    if not lr > 0.0:
        return 1.0
    return 0.5 * float(chi2.sf(lr, 1))


def _select_boundary_terms(
    fit,
    nll,
    amplitude,
    lower,
    upper,
    effective_n,
    /,
    *,
    fit_reduced=None,
    refit=None,
    alpha,
):
    """Fit with automatic boundary terms decided by one-sided LR tests.

    Parameters
    ----------
    fit : callable
        ``fit(lower, upper)`` with boolean flags returns a fitted model.
    nll : callable
        ``nll(model)`` returns its per-unit-weight negative log likelihood.
    amplitude : callable
        ``amplitude(model, side)`` returns the physical-side amplitude
        (``0.0`` when the side has no term).
    lower, upper : bool or ``"auto"``
        Boundary policy per physical side.
    effective_n : float
        Effective sample size of the observation weights.
    fit_reduced : callable or None, optional
        ``fit_reduced(model, lower, upper)`` fits the nested model without a
        term, from ``model`` and at its degree; defaults to ``fit``.
    refit : callable or None, optional
        ``refit(lower, upper)`` gives the final fit once a term has been
        dropped (for instance, rerunning degree selection); defaults to the
        nested fit.
    alpha : float
        Explicit test level resolved by the caller.

    Returns
    -------
    model : object
        Final fitted model.
    flags : tuple of bool
        Final ``(lower, upper)`` flags.
    p_values : tuple of float
        Per side, the test's p-value when the side was decided automatically
        and needed a test, else ``nan``.
    """
    policy = {"lower": lower, "upper": upper}
    flags = {side: policy[side] is True or policy[side] == AUTO for side in _SIDES}
    auto = [side for side in _SIDES if policy[side] == AUTO]
    p_values = dict.fromkeys(_SIDES, np.nan)
    current = fit(flags["lower"], flags["upper"])

    def nested(model, lo, up):
        if fit_reduced is None:
            return fit(lo, up)
        return fit_reduced(model, lo, up)

    dropped = False
    while auto:
        trials = []
        for side in auto:
            trial = dict(flags, **{side: False})
            if not amplitude(current, side) > 0.0:
                # The optimum already sits on the face a = 0: the nested
                # model has the same optimum, and nothing is tested.
                trials.append((1.0, side, None))
                continue
            reduced = nested(current, trial["lower"], trial["upper"])
            p = _boundary_p_value(nll(reduced), nll(current), effective_n)
            p_values[side] = p
            trials.append((p, side, reduced))
        p, side, reduced = max(trials, key=lambda item: item[0])
        if p < alpha:
            break
        flags[side] = False
        auto.remove(side)
        if reduced is None:
            reduced = nested(current, flags["lower"], flags["upper"])
        current = reduced
        dropped = True
    if dropped and refit is not None:
        current = refit(flags["lower"], flags["upper"])
    return (
        current,
        (flags["lower"], flags["upper"]),
        (p_values["lower"], p_values["upper"]),
    )


def _weakly_identified_sides(amplitudes, standard_errors, /):
    """Sides whose positive amplitude lies within two standard errors of zero.

    Parameters
    ----------
    amplitudes : array_like, shape (2,)
        Physical ``(lower, upper)`` amplitudes (``nan`` or ``0`` for no term).
    standard_errors : array_like, shape (2,)
        Their standard errors (``nan`` for no term).
    """
    a = np.nan_to_num(np.asarray(amplitudes, dtype=np.float64), nan=0.0)
    se = np.asarray(standard_errors, dtype=np.float64)
    return tuple(
        side
        for side, value, error in zip(_SIDES, a, se, strict=True)
        if value > 0.0 and not np.isnan(error) and value < 2.0 * error
    )
