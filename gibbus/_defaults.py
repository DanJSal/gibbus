"""Package-wide numerical constants, tolerances, and algorithm defaults.

User-visible defaults and cross-cutting numerical tolerances live here.
Implementation-local constants may remain beside the kernel they control when
they are not part of the public contract and are not shared across modules.

Groups
------
* **Floating-point sentinels and mode/root numerics** — ``MAX_FAC``,
  ``PROB_EPS``, ``TINY_FLOAT``, ``LOG_THRESH``, Newton/bracketing/backtracking constants
* **Quadrature and moments** — ``QUAD_EPSABS``, ``QUAD_EPSREL``,
  ``QUAD_LIMIT``, ``INTERVAL_GL_ORDER``, ``MAX_CACHED_MOMENTS``
* **EM and component-count selection** — ``EM_*``, ``AUTO_K_*``,
  ``AUTO_KDE_*``, ``AUTO_LC_*`` and selection-subsampling constants
* **Automatic polynomial degree** — ``AUTO_POLY_DEGREE_MIN``,
  ``AUTO_POLY_DEGREE_MAX``
* **Robust coordinate scaling, mixture modes, spectral state, and tail inversion**

Failure handling
----------------
``DEBUG`` (set by the ``GIBBUS_DEBUG`` environment variable),
``NUMERIC_FAILURES``, ``_reraise_if_debug`` and ``_maybe_suppress``
define how the package degrades: which exception types a numerical
fallback may swallow, how to make those fallbacks fatal for debugging,
and how to scope warning suppression to the user's
``suppress_warnings`` flag.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from threading import local
from typing import Final

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.optimize import OptimizeWarning

# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


SUPPRESSED_WARNINGS: Final = (
    RuntimeWarning,
    OptimizeWarning,
    IntegrationWarning,
)
"""Warning categories silenced when ``suppress_warnings=True``.

``RuntimeWarning`` covers deliberate overflow/underflow in numerical
kernels; ``OptimizeWarning`` and ``IntegrationWarning`` come from SciPy.
"""


def _env_flag(name, /):
    """Return whether an environment variable contains a truthy token.

    Parameters
    ----------
    name : str
        Environment-variable name.

    Returns
    -------
    bool
    """
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG: Final = _env_flag("GIBBUS_DEBUG")
"""When ``GIBBUS_DEBUG`` is set, unexpected numerical fallbacks re-raise.

Fallbacks marked ``routine=True`` represent expected guard paths on healthy
input, such as a screening candidate being declined or a spectral inverse
failing certification.  They are recorded but do not re-raise under this
mode.  Intended for development and test-suite verification."""

DEBUG_STRICT: Final = _env_flag("GIBBUS_DEBUG_STRICT")
"""Whether ``GIBBUS_DEBUG`` should also make routine fallbacks fatal.

``GIBBUS_DEBUG_STRICT`` is an add-on to :data:`DEBUG`: it has effect only when
``GIBBUS_DEBUG`` is also set. Use it when even routine guard paths must be
surfaced as failures."""

NUMERIC_FAILURES: Final = (ArithmeticError, np.linalg.LinAlgError, RuntimeError)
"""Exception types a numerical fallback may legitimately swallow.

Covers arithmetic floating-point failures, ``numpy.linalg.LinAlgError``,
and the explicit degenerate-state ``RuntimeError`` raised inside the fitting
pipeline. Generic ``ValueError`` is deliberately excluded: NumPy, SciPy, and
the package itself use it for shape and contract defects that must reach the
caller rather than being guessed numerical from exception-message text.

``KeyboardInterrupt``, ``MemoryError``, ``TypeError``, ``AttributeError``,
``KeyError`` and ``IndexError`` are likewise excluded because they indicate a
bug, invalid object contract, resource exhaustion, or user interrupt. Every
genuine Python-layer suppression is recorded and can be inspected through
:func:`suppressed_failures`."""

_SUPPRESSED_MAX: Final = 256
"""Cap on the suppressed-failure record, which is a ring buffer."""

_SUPPRESSED_LOCAL = local()


def _suppressed_buffer():
    """Return the current thread's fallback ring buffer."""
    buffer = getattr(_SUPPRESSED_LOCAL, "buffer", None)
    if buffer is None:
        buffer = []
        _SUPPRESSED_LOCAL.buffer = buffer
    return buffer


def suppressed_failures():
    """Return the numerical fallbacks taken since the record was last cleared.

    Fallbacks are normal operation in places (a root-finder failing to
    bracket, a screening fit declining a candidate) and a symptom in
    others. The ``routine`` field marks expected guard paths, while the
    context and frequency still determine whether a recorded fallback is
    worth investigating. The record is thread-local: each thread sees only
    the fallbacks recorded by work executed in that same thread.

    Returns
    -------
    list of dict
        Most recent last, each with ``context``, ``type``, ``message`` and
        ``routine``.  The last field marks expected guard paths. Capped at
        ``_SUPPRESSED_MAX`` entries.

    See Also
    --------
    clear_suppressed_failures : Empty the record.
    """
    return [dict(entry) for entry in _suppressed_buffer()]


def clear_suppressed_failures():
    """Empty the suppressed-failure record.

    Returns
    -------
    None
    """
    _suppressed_buffer().clear()


@contextmanager
def _maybe_suppress(active, categories, /):
    """Suppress *categories* of warning only when *active*.

    Parameters
    ----------
    active : bool
        When false this is a no-op, so the caller's
        ``suppress_warnings=False`` still surfaces everything.
    categories : tuple of type
        Warning classes to ignore.

    Yields
    ------
    None
    """
    if not active:
        yield
        return
    with warnings.catch_warnings():
        for cat in categories:
            warnings.simplefilter("ignore", category=cat)
        yield


def _reraise_if_debug(exc, context, /, *, routine=False):
    """Record a swallowed numerical failure, re-raising under ``GIBBUS_DEBUG``.

    Every Python-layer fallback deliberately guarded by
    :data:`NUMERIC_FAILURES` funnels through here. Compiled ``noexcept nogil``
    allocation fallbacks are outside this Python ledger by construction.

    ``routine`` marks a fallback that is expected on healthy input rather
    than a symptom: for example, a selection sweep declining a screening
    fit, an EM guard keeping the current iterate, or a spectral inverse
    declining certification.  :data:`DEBUG` makes unexpected fallbacks fatal
    while continuing to record routine ones. When :data:`DEBUG` is enabled,
    :data:`DEBUG_STRICT` makes both categories fatal.

    Parameters
    ----------
    exc : BaseException
        The caught exception.
    context : str
        Short description of the operation that failed, used in the
        chained error message and in the record.
    routine : bool, optional
        Whether this fallback is expected on healthy input.

    Raises
    ------
    RuntimeError
        When :data:`DEBUG` is set and the fallback is not routine, or when
        both :data:`DEBUG` and :data:`DEBUG_STRICT` are set.
    """
    if DEBUG and (DEBUG_STRICT or not routine):
        raise RuntimeError(
            f"gibbus: {context} failed and would have been silently "
            f"skipped (GIBBUS_DEBUG is set)."
        ) from exc
    buffer = _suppressed_buffer()
    if len(buffer) >= _SUPPRESSED_MAX:
        del buffer[0]
    buffer.append({
        "context": str(context),
        "type": type(exc).__name__,
        "message": str(exc)[:1000],
        "routine": bool(routine),
    })

_F64_INFO: Final = np.finfo(np.float64)

# ---------------------------------------------------------------------------
# Floating-point sentinels
# ---------------------------------------------------------------------------

MAX_FAC: Final = 64
"""Maximum order for the pre-computed factorial table in ``_model/vec.py``."""

EXP_NARROW_D2_THRESHOLD: Final = 1e-4
"""Switch to direct centered exp-space quadrature below this log-variance."""

LOG_THRESH: Final = -np.log(_F64_INFO.eps)
"""Log-density threshold (≈ 36.04) at which exp(-q) is considered negligible
relative to the peak; used by ``_state_kernels._q_window_and_mode`` to set
the fitting window on unbounded sides."""

PROB_EPS: Final = _F64_INFO.eps
"""Minimum probability value used to clip uniform samples before inversion,
preventing exact 0 or 1 from reaching the PPF."""

TINY_FLOAT: Final = _F64_INFO.tiny
"""Smallest positive normalised float64; used as a floor when clipping
values before taking logarithms -- mixture weights and responsibility row
sums, interval masses, and tail distances.

It is **not** a density floor.  The point-data objective accumulates the
potential ``q`` directly and never round-trips through ``exp``/``log``,
so no PDF value is clipped there.  Note also that
``_state_kernels._points_nll_grad_batch`` declares its own local
``TINY = 1e-300`` which shadows this name and is unrelated to it; the
two differ by eight orders of magnitude."""

# ---------------------------------------------------------------------------
# Post-fit API extensions
# ---------------------------------------------------------------------------

SF_HANDOVER_P: Final = 1e-5
"""Survival/CDF probability below which exact tail quadrature is preferred."""

LOG_HALF: Final = -np.log(2.0)
"""Logarithm of one half, used to select the accurate interval-mass side."""

EXPECT_MAX_RELATIVE_ERROR: Final = 1e-8
"""Largest accepted relative quadrature error for public expectations."""

NARROW_LOG_MASS_GAP: Final = 1e-6
"""Log-probability gap below which interval mass is checked by quadrature."""

PIT_CLIP: Final = _F64_INFO.eps
"""Distance from 0 and 1 at which PIT values are clipped before EDF tests.

A fitted CDF legitimately returns exactly ``0.0`` or ``1.0`` for an
observation at or beyond a bounded support.  The Anderson-Darling weight
``1 / (u (1 - u))`` diverges there, so an unclipped endpoint makes the
statistic infinite regardless of how well the rest of the sample fits.
Clipping at one machine epsilon bounds the statistic while leaving every
genuinely interior value unchanged at float64 resolution."""

BOOTSTRAP_MAX_FAILURE_FRACTION: Final = 0.25
"""Share of resampling replicates allowed to fail before the run is rejected.

Occasional failures are expected: a resample can omit enough of a mode's
support that its log-concave component degenerates.  Past this share the
surviving replicates are no longer a fair sample of the resampling
distribution, so reporting a band from them would present a broken estimate
as a confident one."""

BOOTSTRAP_DEFAULT_RESAMPLES: Final = 200
"""Default replicate count for uncertainty bands.

Enough for stable pointwise percentiles at conventional levels while
keeping a full refit-per-replicate run tractable; raise it for published
figures, where 1000 or more is the usual standard."""

HPD_LEVEL_TOL: Final = 1e-10
"""Probability-mass tolerance used by highest-density-region bisection."""

HPD_MERGE_TOL: Final = 1e-9
"""Relative gap below which adjacent HPD intervals are merged."""

HPD_MONOTONICITY_TOL: Final = 1e-8
"""Allowed mass jitter when checking HPD threshold monotonicity."""

HPD_FINAL_MASS_TOL: Final = 1e-7
"""Maximum final absolute HPD mass error before reporting non-convergence."""

QFLAT_TOL: Final = 1e-12
"""Potential variation below which a density is treated as flat."""

TAIL_RATE_TOL: Final = 1e-8
"""Relative stabilization tolerance for asymptotic tail-rate estimates."""

HAZARD_MONOTONE_TOL: Final = 1e-9
"""Allowed relative hazard descent in the monotonicity diagnostic."""

# ---------------------------------------------------------------------------
# Newton / root-finding solver
# ---------------------------------------------------------------------------

GRAD_TOL: Final = 1e-12
"""Gradient convergence tolerance for the Newton–Raphson solver in
``_state_kernels``."""

HESS_TOL: Final = 1e-16
"""Minimum |Hessian| accepted as a valid Newton step denominator."""

NEWT_TOL: Final = 1e-12
"""Step-size convergence tolerance for Newton iterations."""

NEWT_MAX: Final = 100
"""Maximum Newton iterations per solve."""

# ---------------------------------------------------------------------------
# Bracket expansion (used when Newton does not converge)
# ---------------------------------------------------------------------------

BRACKET_MAX_EXPAND: Final = 30
"""Maximum number of geometric bracket-expansion doublings."""

BRACKET_INIT_STEP: Final = 1.0
"""Initial bracket half-width before expansion."""

BRACKET_STEP_GROWTH: Final = 2.0
"""Multiplicative growth factor for the bracket step each iteration."""

# ---------------------------------------------------------------------------
# Backtracking line search
# ---------------------------------------------------------------------------

BACKTRACK_MAX_ITERS: Final = 20
"""Maximum backtracking steps per Newton iteration."""

BACKTRACK_REDUCE: Final = 0.5
"""Step-size reduction factor during backtracking."""

# ---------------------------------------------------------------------------
# Boundary safety margin
# ---------------------------------------------------------------------------

BOUNDARY_EPS_MULT: Final = 1e-12
"""Multiplicative guard distance from finite support boundaries used by the
Newton solver to avoid singularities in the log-boundary terms."""

# ---------------------------------------------------------------------------
# Quadrature (scipy.integrate.quad)
# ---------------------------------------------------------------------------

QUAD_EPSABS: Final = 1.49e-08
"""Absolute error tolerance passed to ``scipy.integrate.quad``."""

QUAD_EPSREL: Final = 1.49e-08
"""Relative error tolerance passed to ``scipy.integrate.quad``."""

QUAD_LIMIT: Final = 100
"""Maximum sub-interval limit passed to ``scipy.integrate.quad``."""

# ---------------------------------------------------------------------------
# Moment cache
# ---------------------------------------------------------------------------

MAX_CACHED_MOMENTS: Final = 32
"""Number of raw moments stored in the ``raw_moments`` field of the fitted
structured state.  Moments up to this order are memoised across calls."""

# ---------------------------------------------------------------------------
# Mixture PPF bisection fallback
# ---------------------------------------------------------------------------

PPF_BISECT_MAX_ITER: Final = 64
"""Maximum bisection steps for mixture quantile inversion fallback."""

PPF_BISECT_Z_TOL: Final = 1e-16
"""Compact-coordinate bracket width at which fallback bisection stops."""

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Interval-likelihood helpers
# ---------------------------------------------------------------------------

INTERVAL_GL_ORDER: Final = 24
"""Gauss-Legendre order per deterministic local finite-interval panel.

Ordinary finite rows strictly inside the support use one panel, or two panels
split at the current density mode when it lies inside the interval.  Boundary-
touching positive-width rows and rows with infinite endpoints use the prepared
adaptive interval reducer instead.  Order 24 gives near-float64 agreement for
the maintained finite-row gradient stress cases while keeping that path
vectorised.
"""

INTERVAL_W_EPS_MULT: Final = 1e-12
"""Relative width threshold for the point limit on the local finite-row path."""


# ---------------------------------------------------------------------------
# Variance tolerance
# ---------------------------------------------------------------------------

STATS_VAR_NEG_TOL: Final = 1e-15
"""Relative tolerance for detecting a collapsed (spike) density.

If ``|m2 - m1^2|`` is below this fraction of ``max(m2, m1^2)`` the two
moments agree to their last bit and post-fit statistics raise rather
than report a meaningless variance.  Relative, so it is scale-free."""

# ---------------------------------------------------------------------------
# EM mixture fitting
# ---------------------------------------------------------------------------

EM_MAX_ITER: Final = 50
"""Maximum EM iterations for mixture fitting."""

EM_TOL: Final = 1e-4
"""Relative log-likelihood convergence tolerance for EM."""

EM_RESP_FLOOR: Final = 1e-300
"""Floor applied to per-sample responsibilities before normalisation, to
prevent exact-zero weights from reaching the component refit."""

EM_MIN_EFFECTIVE_DISTINCT_N: Final = 2.0
"""Minimum Kish effective count of distinct point locations per EM component.

For point-data mixtures the unconstrained likelihood has the classical
singularity in which one component captures one observation and sends its
scale to zero.  Responsibilities are therefore aggregated over duplicate
sample coordinates and each component must retain at least two effective
distinct locations before its M-step.  Interval-censored observations are
exempt: their likelihood contributions are probability masses bounded by one,
so the same point-density singularity is absent.
"""

# ---------------------------------------------------------------------------
# Automatic component-count selection
# ---------------------------------------------------------------------------

AUTO_K_MIN: Final = 1
"""Minimum number of components tried when ``n_components="auto"``."""

AUTO_K_MAX: Final = 10
"""Maximum number of components tried when ``n_components="auto"``."""

AUTO_GMM_N_INIT: Final = 5
"""Number of k-means++ restarts of the Gaussian-mixture initialization seed;
the restart with the best average log likelihood is kept."""

GMM_SEED_MAX_POINTS: Final = 2000
"""Largest sample the GMM initialization seed is fitted on.  Larger samples
are thinned to this many evenly spaced order statistics (deterministic, and
faithful to the sample's shape); responsibilities are then evaluated for
every point.  A seed only needs the components' rough locations and scales."""

AUTO_KDE_BW_LO: Final = 0.5
"""Lower multiplier for the KDE bandwidth sweep used by
``_count_modes_kde``.  The Silverman bandwidth is multiplied by
log-spaced factors from ``AUTO_KDE_BW_LO`` to ``AUTO_KDE_BW_HI``."""

AUTO_KDE_BW_HI: Final = 3.0
"""Upper multiplier for the KDE bandwidth sweep."""

AUTO_KDE_BW_STEPS: Final = 15
"""Number of log-spaced bandwidth multipliers evaluated by the KDE
mode-counting sweep."""

AUTO_KDE_GRID_POINTS: Final = 2048
"""Number of equally spaced grid points on which each KDE is evaluated
when counting local maxima."""

AUTO_LC_VALIDATION_DEGREES: Final = (4, 6)
"""Polynomial degrees the LC BIC sweep scores each candidate *K* at.

Each candidate is fitted at every degree here and keeps its best BIC, so
that *K* is compared under a degree suited to it.  Scoring every candidate at
a single low degree biases selection toward larger *K*: two low-degree
components together can express more shape than one and can therefore win for
representation rather than genuine multimodality.

Measured over eleven shapes at n=4000 (component-count accuracy, and
end-to-end auto-selection time for the 50 fits):

    fixed d=4     44/50   33s        (4, 6)      50/50   48s
    fixed d=6     49/50   35s        (4, 8)      48/50   53s
                                     (6, 8)      50/50   58s
                                     (4, 6, 8)   50/50   67s

``(4, 6)`` is the cheapest set that scores perfectly.  Dropping 6 for 8
loses two ``gamma`` seeds, so the middle degree is doing real work.  The set is
a calibrated cost/coverage choice; each *K* is judged at its best degree from
that common candidate set rather than all candidates sharing one degree.

Only even degrees are listed.  Odd degrees are inadmissible on an
unbounded support, and on a half line they add the skew factor without
adding a mode, so they rarely win a BIC comparison against the next even
degree; including them would grow the sweep's cost for little return.
Candidates are filtered against
``gibbus._fit.inputs._is_poly_degree_admissible`` regardless, so adding one
stays safe.  The ceiling is 8 because the sweep's few-iteration fits
lose conditioning above it, which makes the BIC values noisy.

Independent boundary-case checks show that ``(4, 6)`` behaves like ``(6,)``
and ``(4, 6, 8)`` on difficult unbounded shapes, while bounded beta shapes
require degree 4 to avoid incorrect component counts.  Bounded-support cases
are therefore essential when recalibrating this candidate set.
"""


AUTO_LC_SWEEP_MAX_K: Final = 8
"""Hard upper bound on the number of candidate *K* values evaluated by
the LC BIC sweep.  Prevents pathological cost when ``_count_modes_kde``
returns a large mode count on noisy data."""

AUTO_LC_MIN_COMPONENT_N: Final = 30
"""Minimum effective sample count a component must hold for its *K* to be
a viable BIC candidate.

BIC penalises parameters, not support: a component fitted to a handful of
tail points is cheap in parameters and can win on likelihood, so without
this a sweep will sometimes prefer a *K* whose extra component carries a
fraction of a percent of the mass.  Effective count is ``w_j * n``.  A
component below this has too few observations to estimate its own
potential, whatever the likelihood says.

The validation potentials carry 5-7 free parameters, so 30 is roughly
five observations per parameter.  The value is not knife-edge: anything
in 25-40 keeps ``beta(2, 5)`` at one component while leaving
``lognormal(0, 0.6)`` its genuine second one.  Below ~25 a
``gamma(2, 1)`` rival resting on 22 effective samples starts winning
converged BIC comparisons; above ~45 the sweep's lite fits, which are
noisier than the final one, transiently produce a small component and
get a whole *K* rejected on that basis."""

AUTO_LC_SWEEP_EM_MAX_ITER: Final = 5
"""Maximum EM iterations for each candidate *K* (>= 2) during the LC BIC
sweep.  The sweep only needs a rough BIC ranking, not converged fits, so
this can be much smaller than ``EM_MAX_ITER`` (50)."""

AUTO_KDE_MIN_PROMINENCE: Final = 0.005
"""Minimum topographic prominence for a KDE local maximum to count as a
mode, as a fraction of the sweep's peak density.

Prominence -- a peak's height above the highest saddle separating it from
any taller peak -- distinguishes a real mode from a ripple riding on a
tail.  Without it, sampling noise in a long right tail registers as
structure: the unfiltered sweep reported up to four modes on a
``lognormal`` sample and two on a ``gamma(2, 1)``, both of which have one.

Chosen from a threshold sweep over eleven shapes.  0.005 is the smallest
value at which spurious modes are fully suppressed on every skewed and
heavy-tailed case tested, and it still resolves a well-separated
component carrying 0.5% of the mass -- an order of magnitude below the
smallest minor component in the selection battery.  Raising it costs
sensitivity quickly: 0.02 loses a 1% component and 0.05 loses a 2% one."""

AUTO_KDE_WEAK_CELL_MASS: Final = 0.10
"""Weighted cell-mass threshold below which a KDE valley must be locally
well resolved before it is trusted as a component initializer.

Small tail cells are where sampling ripples most often masquerade as a
second mode.  Larger cells are left to the bandwidth-persistence criterion,
which avoids rejecting moderately overlapping multi-component shapes.
"""

AUTO_KDE_MIN_LOCAL_PROMINENCE: Final = 0.20
"""Minimum prominence relative to a peak's *own height* for a small KDE cell.

The global prominence threshold remains deliberately permissive so a real
minority component can be found even when the dominant component is much
taller.  For cells below :data:`AUTO_KDE_WEAK_CELL_MASS`, however, a peak
must also rise materially above its adjacent saddle.  This rejects shallow
tail ripples while retaining well-separated components carrying around one
percent of the mass.
"""

AUTO_KDE_SUBSAMPLE_N: Final = 5_000
"""Point budget for KDE mode counting.

The cap regularises the mode-count estimate as well as bounding work.  The
Silverman bandwidth falls as the sample count rises, so on a large heavy-tailed
input the lower multipliers can resolve tail noise as extra modes.  Holding the
count fixed also stabilises the bandwidth scale used by the sweep.

Stratified rather than uniform thinning because the output is a
component *count*, and a low-weight mode dropping out of the draw would
change it."""

AUTO_LC_SUBSAMPLE_MIN_N: Final = 20_000
"""Sample count above which the LC BIC sweep scores candidates on a
subsample rather than the full dataset.

Below this the sweep is already cheap relative to the final fit, and
subsampling would only add variance."""

AUTO_LC_SUBSAMPLE_SIZE: Final = 10_000
"""Base target size of the LC sweep subsample.

Component *count* is a coarse property of a distribution; ten thousand
points distinguish K=2 from K=3 comfortably.  Subsampling affects only the
selection stage.  Once *K* is chosen, its final model is refitted on every
observation; a different selected *K* can therefore still change the final
model."""

AUTO_LC_SUBSAMPLE_PER_K: Final = 1_500
"""Minimum subsample points per candidate component.

The subsample target is ``max(AUTO_LC_SUBSAMPLE_SIZE, k_hi *
AUTO_LC_SUBSAMPLE_PER_K)`` so that a large ``k_max`` does not leave
individual components starved of data."""

# ---------------------------------------------------------------------------
# Mixture mode finding
# ---------------------------------------------------------------------------

MODE_XTOL: Final = 1e-10
"""Absolute x-tolerance for convergence and deduplication when finding
mixture modes via root-finding on the first derivative of ``neg_log``."""

MODE_DERIV_TOL: Final = 1e-8
"""Second-derivative threshold for classifying a critical point of the
mixture ``neg_log`` as a mode (local minimum of ``neg_log``, i.e. local
maximum of the PDF).  A root is kept when ``neg_log(x, 2) > -MODE_DERIV_TOL``."""

# ---------------------------------------------------------------------------
# Automatic poly_degree selection
# ---------------------------------------------------------------------------

AUTO_POLY_DEGREE_MIN: Final = 2
"""Minimum polynomial degree tried when ``poly_degree='auto'``."""

AUTO_POLY_DEGREE_MAX: Final = 12
"""Maximum polynomial degree tried when ``poly_degree='auto'``."""


# ---------------------------------------------------------------------------
# Robust scale
# ---------------------------------------------------------------------------

UNIFORM_WIDTH_TO_SIGMA: Final = 0.28867513459481288
"""Standard deviation of a uniform distribution of unit width (``1/sqrt(12)``).

Used as a floor on the scale estimate for interval-censored data.  The
location and scale are estimated from interval midpoints, which say
nothing about spread *within* a bin; with coarse bins more than half the
midpoints can coincide, giving a zero MAD and rejecting data that is
perfectly informative.  A bin of width ``w`` contributes at least
``w * UNIFORM_WIDTH_TO_SIGMA`` of spread, so that is the floor."""

MAD_TO_SIGMA: Final = 1.4826
"""Factor converting a median absolute deviation to a Gaussian-consistent
standard deviation (``1 / Phi^{-1}(3/4)``).  Used to normalise data before
fitting so the optimiser sees O(1) coordinates."""

# ---------------------------------------------------------------------------
# Spectral CDF / PPF construction
# ---------------------------------------------------------------------------
# These set the accuracy of every ``cdf()`` and ``ppf()`` call, so they
# belong here rather than as signature defaults deep in the spectral
# modules, where they lived until this was noticed.

SPECTRAL_DEGREE_OPTIONS: Final = (16, 24, 32)
"""Chebyshev degrees tried per panel, lowest first; a panel is accepted at
the first degree whose coefficient tail has decayed below tolerance."""

SPECTRAL_CDF_REL_TOL: Final = 2e-12
"""Relative error a CDF panel must reach against a higher-order check."""

SPECTRAL_CDF_ABS_TOL: Final = 5e-15
"""Absolute floor on the CDF panel error, for panels carrying tiny mass."""

SPECTRAL_CDF_COEFF_TOL: Final = 2e-12
"""Relative size of the trailing Chebyshev coefficients at which a CDF
panel counts as resolved."""

SPECTRAL_CDF_MAX_DEPTH: Final = 22
"""Refinement depth at which a CDF panel is accepted as-is."""

SPECTRAL_CDF_MAX_PANELS: Final = 512
"""Strict leaf budget for the adaptive CDF partition.

A split replaces one leaf by two, so a 512-leaf cap implies at most 1023
interval nodes.  With the default degree ladder ``(16, 24, 32)``, fitting an
interval samples the transformed density at at most 228 points across all
three attempts; the construction therefore performs at most 233,244 such
samples before optional mass recertification.  Even if every final leaf needs
recertification, the 8 x 24-point composite rule adds at most 98,304 samples.
The cap is thus a deterministic resource fuse, not an accuracy target;
accuracy is controlled by the spectral tolerances and reported diagnostics.
"""

SPECTRAL_PPF_FIT_TOL: Final = 2e-12
"""Relative Chebyshev fit error an inverse panel must reach."""

SPECTRAL_PPF_LOGIT_TOL: Final = 2e-10
"""Residual tolerance for the inverse in logit coordinates."""

SPECTRAL_PPF_PROB_TOL: Final = 5e-14
"""Residual tolerance for the inverse in probability coordinates."""

SPECTRAL_PPF_COEFF_TOL: Final = 2e-12
"""Coefficient-tail tolerance for an inverse panel."""

SPECTRAL_PPF_MAX_DEPTH: Final = 20
"""Refinement depth at which an inverse panel is accepted as-is."""

SPECTRAL_PPF_MAX_PANELS: Final = 256
"""Strict leaf budget for the adaptive inverse partition.

At most 511 interval nodes can be visited before a 256-leaf frontier is full.
Across the default degree ladder, one interval requires at most 231 exact CDF
inversions for fit and validation, so the cap bounds the expensive inverse
construction to about 1.18e5 root inversions (plus seed-endpoint inversions).
If certification still fails, quantiles fall back to monotone CDF bisection;
there is no need to spend an unbounded panel count chasing an inverse fit.
"""

SPECTRAL_PPF_CERTIFY_SUBDIVIDE: Final = 10
"""Sub-intervals per inverse panel used when certifying monotonicity."""

# ---------------------------------------------------------------------------
# Extreme-tail quantiles
# ---------------------------------------------------------------------------

TAIL_ASYMPTOTIC_P: Final = 1e-10
"""Probability below which quantiles are found asymptotically.

The spectral CDF holds probability as an absolute value in ``[0, 1]``, so
it carries about ``1e-16`` of absolute accuracy and is noise below
``1e-15``.  Switching at ``1e-10`` keeps the exact-tail correction comfortably inside
the regime where direct scaled quadrature is inexpensive and avoids relying
on a leading asymptotic whose relative tail-mass error is still percent-level
near the handover.  See :mod:`gibbus._spectral.tail`."""

TAIL_BRACKET_MAX_EXPAND: Final = 200
"""Outward expansions allowed when bracketing an extreme quantile.

Steps grow geometrically, so this reaches any float64 probability with
room to spare; it exists to bound the loop, not to bind."""

TAIL_BRACKET_GROWTH: Final = 1.6
"""Growth factor per bracket expansion.

Gentler than doubling, which overshoots far enough that the subsequent
bisection spends its first several steps walking back."""

TAIL_SOLVE_MAX_ITER: Final = 200
"""Bisection steps allowed when inverting the asymptotic tail."""

TAIL_SOLVE_TOL: Final = 1e-14
"""Relative bracket width at which an extreme quantile is accepted."""
