"""Multi-component initialization, E-steps, packing, and mode utilities.

This low-level fitting module contains observation-level mixture helpers used
by the API-layer fitting controller, automatic component-count selection, and
natural mixture machinery.  It owns no public estimator state and imports no API-layer
objects.

Contents
--------
* **Initialization**: valley, GMM, and concentric-scale responsibility seeds.
* **EM helpers**: point and interval E-steps plus mixture-weight updates.
* **KDE mode counting**: bandwidth sweeps and automatic-K proposals.
* **Serialization**: mixture structured-state packing and unpacking.
* **Post-fit utilities**: component ordering and mixture mode finding.
"""

import itertools
import weakref

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.optimize import brentq
from scipy.signal import find_peaks

from .._defaults import (
    AUTO_GMM_N_INIT,
    AUTO_K_MAX,
    AUTO_K_MIN,
    AUTO_KDE_BW_HI,
    AUTO_KDE_BW_LO,
    AUTO_KDE_BW_STEPS,
    AUTO_KDE_GRID_MARGIN,
    AUTO_KDE_GRID_POINTS,
    AUTO_KDE_MIN_LOCAL_PROMINENCE,
    AUTO_KDE_MIN_PROMINENCE,
    AUTO_KDE_SUBSAMPLE_N,
    AUTO_KDE_WEAK_CELL_MASS,
    EM_RESP_FLOOR,
    GMM_SEED_MAX_ITER,
    GMM_SEED_MAX_POINTS,
    GMM_SEED_TOL,
    GMM_SEED_VARIANCE_FLOOR,
    MODE_DERIV_TOL,
    MODE_XTOL,
    NUMERIC_FAILURES,
    TINY_FLOAT,
    TURNBULL_GAP_TOL,
    TURNBULL_MAX_ITER,
    _reraise_if_debug,
)
from .._model.coords import _weighted_median
from .._observations.intervals import _row_grouping
from .._postfit.fitted_state import _model_metadata


def _interval_initial_representatives(intervals, support, /):
    """Return finite heuristic representatives for interval EM initialization.

    Finite rows use their midpoint; left/right infinite rows use their finite
    censoring cutpoint.  Whole-support rows carry no component-allocation
    information, so they receive the median of the informative representatives
    (or a finite support midpoint when available).  These values are used only
    for initial responsibility/KDE heuristics; all censored likelihood and
    subsequent E-steps use exact interval probabilities.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2), dtype float64
        Boundary-validated ordered interval observations.
    support : tuple of (float, float)
        Shared density support.

    Returns
    -------
    numpy.ndarray, shape (R,)
        Finite representative values.
    """
    x = intervals
    lo, hi = x[:, 0], x[:, 1]
    finite_lo = np.isfinite(lo)
    finite_hi = np.isfinite(hi)
    out = np.empty(x.shape[0], dtype=np.float64)
    both = finite_lo & finite_hi
    out[both] = 0.5 * lo[both] + 0.5 * hi[both]
    left = (~finite_lo) & finite_hi
    out[left] = hi[left]
    right = finite_lo & (~finite_hi)
    out[right] = lo[right]
    whole = (~finite_lo) & (~finite_hi)
    if np.any(whole):
        informative = out[~whole]
        if informative.size:
            fill = float(np.median(informative))
        else:
            lower, upper = map(float, support)
            if np.isfinite(lower) and np.isfinite(upper):
                fill = 0.5 * (lower + upper)
            elif np.isfinite(lower):
                fill = lower
            elif np.isfinite(upper):
                fill = upper
            else:
                fill = 0.0
        out[whole] = fill
    if not np.all(np.isfinite(out)):
        raise ValueError("could not construct finite interval representatives")
    return np.ascontiguousarray(out, dtype=np.float64)


# ======================================================================
# EM helpers
# ======================================================================


def _valley_init_responsibilities(samples_1d, n_components, /, *, weights=None):
    """Initial responsibilities from the KDE's modes and the valleys between them.

    The bandwidth sweep that counts modes already locates them, so the
    decomposition is nearly free: pick the smoothest bandwidth resolving
    at least *n_components* prominent peaks, keep the *n_components* most
    prominent, split the data at the density minima between consecutive
    peaks, and read each component's mass, location and scale off the
    resulting cells.  Those become Gaussian kernels whose posteriors are
    the returned responsibilities.

    The initializer is deterministic, honors observation weights, and costs
    one binning pass plus a few FFTs.  The Gaussian-mixture seed ignores
    observation weights and is therefore used only when the valley path
    cannot represent the requested component structure.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints).
    n_components : int
        Number of mixture components.
    weights : numpy.ndarray, shape (R,) or None, optional
        Boundary-normalized observation weights.

    Returns
    -------
    resp : numpy.ndarray, shape (R, K), or None
    mix_weights : numpy.ndarray, shape (K,), or None
        ``None`` when the density does not separate into *n_components*
        cells -- nested components sharing a location have no valley at
        any bandwidth -- and the caller should fall back.
    """
    x = samples_1d
    n = int(x.shape[0])
    k = int(n_components)
    w = np.full(n, 1.0 / n, dtype=np.float64) if weights is None else weights

    if k <= 1:
        return np.ones((n, 1), dtype=np.float64), np.ones(1, dtype=np.float64)

    lo, hi = float(x.min()), float(x.max())
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return None, None

    n_eff = min(n, int(AUTO_KDE_SUBSAMPLE_N))
    bw = _silverman_bandwidth(x, n_effective=n_eff, weights=w)
    if not (np.isfinite(bw) and bw > 0.0):
        return None, None

    margin = 0.1 * (hi - lo)
    grid = np.linspace(lo - margin, hi + margin, AUTO_KDE_GRID_POINTS)
    mult = np.logspace(
        np.log10(AUTO_KDE_BW_LO), np.log10(AUTO_KDE_BW_HI), AUTO_KDE_BW_STEPS
    )
    try:
        dens = _binned_kde_sweep(x, grid, bw * mult, weights=w)
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "valley initialization")
        return None, None

    # A genuine valley should survive a material change of bandwidth.  A
    # single narrow-bandwidth ripple is sampling noise, even if its local
    # prominence happens to clear the pointwise threshold.  Record the full
    # bandwidth range on which at least k prominent peaks exist and require
    # persistence over a factor of two before accepting the smoothest one.
    eligible = []
    for i, d in enumerate(dens):
        peak = float(d.max())
        if not (np.isfinite(peak) and peak > 0.0):
            continue
        found, props = find_peaks(d, prominence=AUTO_KDE_MIN_PROMINENCE * peak)
        if found.size >= k:
            eligible.append((i, found, props))
    if not eligible:
        return None, None
    if float(mult[eligible[-1][0]] / mult[eligible[0][0]]) < 2.0:
        return None, None

    i, found, props = eligible[-1]
    order = np.argsort(props["prominences"])[::-1][:k]
    selected = found[order]
    peak_to_local_prominence = {
        int(p): float(props["prominences"][j] / dens[i][p])
        for p, j in zip(selected, order, strict=True)
    }
    peaks = np.sort(selected)
    local_prominence = np.array(
        [peak_to_local_prominence[int(p)] for p in peaks], dtype=np.float64
    )
    chosen = dens[i]

    # Split at the density minimum between consecutive peaks.
    cuts = [
        float(grid[a + int(np.argmin(chosen[a : b + 1]))])
        for a, b in itertools.pairwise(peaks)
    ]
    edges = np.array([-np.inf, *cuts, np.inf], dtype=np.float64)

    cell = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, k - 1)
    mix = np.zeros(k, dtype=np.float64)
    mean = np.zeros(k, dtype=np.float64)
    sd = np.zeros(k, dtype=np.float64)
    for j in range(k):
        sel = cell == j
        mass = float(w[sel].sum())
        if mass <= 0.0 or int(sel.sum()) < 2:
            return None, None
        wj = w[sel] / mass
        xj = x[sel]
        mix[j] = mass
        mean[j] = float(np.dot(wj, xj))
        sd[j] = float(np.sqrt(max(np.dot(wj, (xj - mean[j]) ** 2), 0.0)))

    # A very small cell needs a peak that is prominent relative to its own
    # height, not merely relative to the tallest peak in the density.  This
    # rejects shallow tail ripples without sacrificing genuinely separated
    # minority components whose absolute density is necessarily small.
    weak_small = (mix < AUTO_KDE_WEAK_CELL_MASS) & (
        local_prominence < AUTO_KDE_MIN_LOCAL_PROMINENCE
    )
    if np.any(weak_small):
        return None, None

    # A cell narrower than the kernel is an artifact of a shallow
    # valley, not a component; widen it rather than divide by zero.
    sd = np.maximum(sd, 0.25 * bw)

    z = (x[:, None] - mean[None, :]) / sd[None, :]
    log_p = np.log(mix)[None, :] - np.log(sd)[None, :] - 0.5 * z * z
    log_p -= log_p.max(axis=1, keepdims=True)
    resp = np.exp(log_p)
    resp /= resp.sum(axis=1, keepdims=True)
    if not np.all(np.isfinite(resp)):
        return None, None
    return resp, mix


def _nested_scale_init_responsibilities(samples_1d, n_components, /, *, weights=None):
    """Initialize concentric components by radial scale rather than location.

    The ordinary GMM initializer is location driven and can converge to a
    nearly arbitrary left/right split when components share a center but have
    very different scales.  This initializer keeps one robust common center,
    orders observations by absolute distance from it, divides that radial
    ordering into equal-weight shells, and estimates one Gaussian scale per
    shell.  The resulting Gaussian posteriors seed EM with a genuine
    core-versus-tail decomposition.

    Parameters
    ----------
    samples_1d : numpy.ndarray
        One-dimensional sample values.
    n_components : int
        Number of concentric scale components to initialize.
    weights : numpy.ndarray or None
        Boundary-normalized sample weights, or ``None`` for equal weights.
    """
    x = samples_1d
    n = int(x.size)
    k = int(n_components)
    if n < k or k <= 1 or not np.all(np.isfinite(x)):
        return None, None

    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        w = weights

    center = float(_weighted_median(x, w))
    radius = np.abs(x - center)
    order = np.argsort(radius, kind="mergesort")
    cum = np.cumsum(w[order])
    shell = np.minimum(
        np.searchsorted(np.linspace(1.0 / k, 1.0, k), cum, side="left"), k - 1
    )
    labels = np.empty(n, dtype=np.intp)
    labels[order] = shell

    mix = np.empty(k, dtype=np.float64)
    scales = np.empty(k, dtype=np.float64)
    scale_floor = np.finfo(np.float64).eps * max(
        1.0, abs(center), float(np.max(radius))
    )
    for j in range(k):
        sel = labels == j
        mass = float(np.sum(w[sel]))
        if not (mass > 0.0) or np.count_nonzero(sel) < 2:
            return None, None
        rj = x[sel] - center
        scales[j] = max(float(np.sqrt(np.dot(w[sel], rj * rj) / mass)), scale_floor)
        mix[j] = mass

    # The radial shells should describe successively broader kernels.  Collapse
    # duplicates rather than passing an ill-conditioned candidate to EM.
    if np.any(
        np.diff(scales) <= 8.0 * np.finfo(np.float64).eps * np.maximum(scales[:-1], 1.0)
    ):
        return None, None

    z = (x[:, None] - center) / scales[None, :]
    log_p = np.log(mix)[None, :] - np.log(scales)[None, :] - 0.5 * z * z
    log_p -= np.max(log_p, axis=1, keepdims=True)
    resp = np.exp(log_p)
    resp /= np.sum(resp, axis=1, keepdims=True)
    if not np.all(np.isfinite(resp)):
        return None, None
    return resp, mix


def _initial_responsibility_candidates(
    samples_1d, n_components, rng, /, *, weights=None, include_valley=True
):
    """Return distinct initialization families for likelihood comparison.

    A resolved KDE valley is already strong structural information and is used
    alone.  When no valley exists (or ``include_valley`` is false), return both
    the ordinary location-oriented GMM seed and a concentric radial-scale seed;
    callers that run EM can keep whichever converged solution has the larger
    observed log-likelihood.  ``include_valley=False`` is how a caller widens
    the search when the valley start proved unreliable (see
    ``natural_mixture._fit_natural_mixture``).

    Parameters
    ----------
    samples_1d : numpy.ndarray
        One-dimensional sample values.
    n_components : int
        Requested number of mixture components.
    rng : numpy.random.Generator
        Random generator used by stochastic initialization families.
    weights : numpy.ndarray or None
        Optional sample weights.
    include_valley : bool, optional
        Whether a resolved valley seed may be returned.
    """
    if include_valley:
        valley_resp, valley_mix = _valley_init_responsibilities(
            samples_1d, n_components, weights=weights
        )
        if valley_resp is not None:
            return [("valley", valley_resp, valley_mix)]

    candidates = []
    try:
        resp, mix = _gmm_init_responsibilities(samples_1d, n_components, rng)
        candidates.append(("gmm", resp, mix))
    except NUMERIC_FAILURES as exc:
        _reraise_if_debug(exc, "GMM mixture initialization", routine=True)

    resp, mix = _nested_scale_init_responsibilities(
        samples_1d, n_components, weights=weights
    )
    if resp is not None:
        candidates.append(("nested-scale", resp, mix))
    if not candidates:
        raise RuntimeError("could not construct mixture initial responsibilities")
    return candidates


def _init_responsibilities(samples_1d, n_components, rng, /, *, weights=None):
    """Initial responsibilities: valleys where they exist, GMM otherwise.

    :func:`_valley_init_responsibilities` is preferred -- it is
    deterministic, honors *weights*, and reuses the KDE sweep the
    pipeline already runs.  It declines when no bandwidth in the sweep
    resolves exactly *n_components* peaks, which is a real limitation
    rather than a numerical failure.  In practice that means asking for
    more components than the density has modes: two components sharing
    a location differ only in scale and have no valley between them at
    any bandwidth, and an explicit *K* above the mode count has nothing
    to split on.  Where the modes do exist, valleys resolve them
    reliably, including a component carrying 1% of the mass.
    :func:`_gmm_init_responsibilities` covers the remainder.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints).
    n_components : int
        Number of mixture components.
    rng : numpy.random.Generator
        Only consumed by the GMM fallback.
    weights : numpy.ndarray, shape (R,) or None, optional
        Non-negative observation weights.  Used by the valley path; the
        GMM fallback ignores them.

    Returns
    -------
    resp : numpy.ndarray, shape (R, K)
    mix_weights : numpy.ndarray, shape (K,)
    """
    # This helper returns the first available seed.  EM callers that compare
    # converged candidates use ``_initial_responsibility_candidates`` directly.
    _, resp, mix = _initial_responsibility_candidates(
        samples_1d, n_components, rng, weights=weights
    )[0]
    return resp, mix


def _gmm_init_responsibilities(samples_1d, n_components, rng, /):
    """Compute initial responsibilities from a one-dimensional Gaussian mixture.

    Fits a ``K``-component Gaussian mixture by EM (k-means++ centers, hard
    initial assignment, ``AUTO_GMM_N_INIT`` restarts, the best final average
    log likelihood kept) in the modeled base coordinate and returns its
    posterior responsibilities and weights.  It is a seed only: the natural
    mixture fit refines everything.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints) used for initialization.
    n_components : int
        Number of mixture components.
    rng : numpy.random.Generator
        Random-number generator for the k-means++ centers.

    Returns
    -------
    resp : numpy.ndarray, shape (R, K)
        Responsibility matrix.
    weights : numpy.ndarray, shape (K,)
        Initial mixture weights.

    Notes
    -----
    Samples larger than ``GMM_SEED_MAX_POINTS`` are thinned to that many
    evenly spaced order statistics for the fit; responsibilities are
    evaluated for every point.  Data whose natural model is logarithmic
    should be transformed explicitly by the caller.
    """
    x = np.asarray(samples_1d, dtype=np.float64).reshape(-1)
    k = int(n_components)
    if k < 1 or x.size < k:
        raise ValueError("need at least one sample per mixture component")
    fit_x = x
    if x.size > GMM_SEED_MAX_POINTS:
        # Evenly spaced order statistics: deterministic and faithful to the
        # sample's shape, at a fraction of the cost on large samples.
        order = np.argsort(x, kind="stable")
        ranks = np.linspace(0, x.size - 1, GMM_SEED_MAX_POINTS).round().astype(np.intp)
        fit_x = x[order[ranks]]

    best = None
    for _ in range(int(AUTO_GMM_N_INIT)):
        centers = _kmeans_plusplus_1d(fit_x, k, rng)
        params = _gaussian_mixture_em_1d(
            fit_x,
            centers,
            tol=GMM_SEED_TOL,
            max_iter=GMM_SEED_MAX_ITER,
            reg=GMM_SEED_VARIANCE_FLOOR,
        )
        if params is not None and (best is None or params[3] > best[3]):
            best = params
    if best is None:
        raise FloatingPointError("Gaussian mixture initialization failed")
    means, variances, weights, _ = best
    log_resp, _ = _gaussian_log_posterior_1d(x, means, variances, weights)
    return np.exp(log_resp), weights.copy()


def _kmeans_plusplus_1d(x, k, rng, /):
    """Greedy k-means++ centers for one-dimensional data.

    Each new center is the best of ``2 + floor(log k)`` candidates drawn with
    probability proportional to the squared distance to the chosen centers.

    Parameters
    ----------
    x : numpy.ndarray, shape (R,)
        Samples.
    k : int
        Number of centers.
    rng : numpy.random.Generator
        Random-number generator.
    """
    trials = 2 + int(np.log(k))
    centers = [float(x[rng.integers(x.size)])]
    closest = (x - centers[0]) ** 2
    for _ in range(1, k):
        total = float(np.sum(closest))
        if not total > 0.0:
            centers.append(float(x[rng.integers(x.size)]))
            continue
        cumulative = np.cumsum(closest)
        picks = np.searchsorted(cumulative, rng.random(trials) * total)
        picks = np.minimum(picks, x.size - 1)
        candidate_distances = np.minimum(
            closest[None, :], (x[None, :] - x[picks, None]) ** 2
        )
        chosen = int(np.argmin(np.sum(candidate_distances, axis=1)))
        centers.append(float(x[picks[chosen]]))
        closest = candidate_distances[chosen]
    return np.asarray(centers, dtype=np.float64)


def _gaussian_log_posterior_1d(x, means, variances, weights, /):
    """Return ``(log responsibilities, average log likelihood)`` of a 1-D mixture.

    Parameters
    ----------
    x : numpy.ndarray, shape (R,)
        Samples.
    means, variances, weights : numpy.ndarray, shape (K,)
        Component parameters.
    """
    with np.errstate(divide="ignore"):
        log_w = np.log(weights)
    log_p = (
        log_w[None, :]
        - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
        - 0.5 * (x[:, None] - means[None, :]) ** 2 / variances[None, :]
    )
    top = np.max(log_p, axis=1, keepdims=True)
    log_norm = top[:, 0] + np.log(np.sum(np.exp(log_p - top), axis=1))
    return log_p - log_norm[:, None], float(np.mean(log_norm))


def _gaussian_mixture_em_1d(x, centers, /, *, tol, max_iter, reg):
    """EM for a one-dimensional Gaussian mixture from hard nearest-center labels.

    Parameters
    ----------
    x : numpy.ndarray, shape (R,)
        Samples.
    centers : numpy.ndarray, shape (K,)
        Initial centers.
    tol : float
        Stop when the average log likelihood rises by less than this.
    max_iter : int
        EM iteration limit.
    reg : float
        Variance floor added to every component.

    Returns
    -------
    tuple or None
        ``(means, variances, weights, average log likelihood)`` from the last
        valid iteration.  Returns ``None`` only when no valid parameter update
        was completed before a component lost all its mass.
    """
    k = centers.size
    labels = np.argmin(np.abs(x[:, None] - centers[None, :]), axis=1)
    resp = np.zeros((x.size, k), dtype=np.float64)
    resp[np.arange(x.size), labels] = 1.0
    previous = -np.inf
    params = None
    for _ in range(int(max_iter)):
        mass = np.sum(resp, axis=0) + 10.0 * np.finfo(np.float64).eps
        if np.any(mass <= 10.0 * np.finfo(np.float64).eps):
            return params
        means = (resp.T @ x) / mass
        variances = (
            np.sum(resp * (x[:, None] - means[None, :]) ** 2, axis=0) / mass + reg
        )
        weights = mass / np.sum(mass)
        log_resp, average = _gaussian_log_posterior_1d(x, means, variances, weights)
        params = (means, variances, weights, average)
        if abs(average - previous) < tol:
            break
        previous = average
        resp = np.exp(log_resp)
    return params


# ======================================================================
# KDE mode counting
# ======================================================================


def _silverman_bandwidth(samples_1d, /, *, n_effective=None, weights=None):
    """Silverman's rule-of-thumb kernel width, matching ``gaussian_kde``.

    ``scipy.stats.gaussian_kde`` stores a *relative* factor and multiplies
    it by the sample standard deviation to get the kernel width.
    Using that convention keeps the bandwidth on the same scale as
    ``gaussian_kde`` when counting modes.

    The spread is ``ddof=1`` when *weights* is None.  When weights are
    given there is no count to correct by -- ``sample_weight`` are
    relative, not frequencies -- so the reliability-weight estimator is
    used instead, dividing by ``1 - sum(w^2)`` for normalized ``w``.
    That is Kish's effective sample size ``n_eff = 1 / sum(w^2)`` in the
    form ``(n_eff - 1) / n_eff``, and it reduces exactly to ``ddof=1``
    at uniform weights, so passing uniform weights reproduces passing
    none.  Both forms are invariant to rescaling the weights.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples.
    n_effective : int or None, optional
        Sample count to use in the rule instead of ``len(samples_1d)``.
        Lets a caller working on the full data reproduce the bandwidth
        scale of a capped sweep (see ``AUTO_KDE_SUBSAMPLE_N``).
    weights : numpy.ndarray, shape (R,) or None, optional
        Boundary-normalized sample weights for the standard deviation.

    Returns
    -------
    float
        Kernel standard deviation, or ``0.0`` when the samples have no
        spread.
    """
    n = int(samples_1d.shape[0]) if n_effective is None else int(n_effective)
    if n < 2:
        return 0.0
    if weights is None:
        sd = float(np.std(samples_1d, ddof=1))
    else:
        w = weights
        m = float(np.dot(w, samples_1d))
        ss = float(np.dot(w, (samples_1d - m) ** 2))

        # Divide by ``1 - sum(w^2)`` rather than by 1.  ``sample_weight``
        # are *relative*, so there is no count to correct by; the
        # reliability-weight estimator uses Kish's effective sample size
        # ``n_eff = 1 / sum(w^2)`` instead, and ``1 - sum(w^2)`` is
        # ``(n_eff - 1) / n_eff``.
        #
        # At uniform weights ``sum(w^2) = 1/n``, so this reduces to
        # ``sum((x-m)^2) / (n-1)`` -- exactly the ``ddof=1`` above.
        # This makes uniform relative weights exactly match the unweighted
        # ``ddof=1`` estimator.  Both forms are scale-invariant, as relative
        # weights require.
        sum_w2 = float(np.dot(w, w))
        denom = 1.0 - sum_w2
        if not (denom > 0.0):
            # All the weight sits on one observation: no spread to measure.
            return 0.0
        sd = float(np.sqrt(max(ss / denom, 0.0)))
    if not (np.isfinite(sd) and sd > 0.0):
        return 0.0
    factor = float(np.power(n * 3.0 / 4.0, -1.0 / 5.0))
    return factor * sd


def _binned_kde_sweep(samples_1d, grid, bandwidths, /, *, weights=None):
    """Evaluate a Gaussian KDE on *grid* for every bandwidth, via FFT.

    The samples are linearly binned onto the grid once, then convolved
    with each Gaussian kernel in the Fourier domain.  A Gaussian's
    transform is analytic, so a single forward transform of the binned
    counts serves every bandwidth: the sweep costs one ``O(R)`` binning
    pass plus one ``O(G log G)`` transform per bandwidth, instead of the
    ``O(R x G)`` pairwise evaluation ``gaussian_kde`` performs for each
    one.  The data is zero-padded to at least twice the grid length so
    the circular convolution does not wrap.

    Linear binning (splitting each sample between its two neighboring
    grid points) rather than nearest-bin assignment keeps the
    discretization error well below the kernel width, which matters
    because the output is used to locate local maxima.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples; values outside the grid span are dropped.
    grid : numpy.ndarray, shape (G,)
        Uniformly spaced evaluation points, ascending.
    bandwidths : numpy.ndarray, shape (B,)
        Kernel standard deviations, all strictly positive.
    weights : numpy.ndarray, shape (R,) or None, optional
        Non-negative sample weights.  ``None`` (default) weights every
        sample equally.  Linear binning makes weighting free, which is
        what lets the initializer honor ``sample_weight``.

    Returns
    -------
    numpy.ndarray, shape (B, G)
        Density estimates, one row per bandwidth.
    """
    g_n = int(grid.shape[0])
    lo = float(grid[0])
    dx = float(grid[1] - grid[0])

    t = (samples_1d - lo) / dx
    keep = (t >= 0.0) & (t <= g_n - 1)
    t = t[keep]
    w = (
        np.ones(t.shape[0], dtype=np.float64)
        if weights is None
        else np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)[keep]
    )
    i0 = np.floor(t).astype(np.intp)
    np.clip(i0, 0, g_n - 2, out=i0)
    frac = t - i0

    counts = np.bincount(i0, weights=w * (1.0 - frac), minlength=g_n) + np.bincount(
        i0 + 1, weights=w * frac, minlength=g_n
    )
    total = float(counts.sum())
    if total > 0.0:
        counts = counts / total

    m = int(next_fast_len(2 * g_n))
    spec = rfft(counts, n=m)
    freq = rfftfreq(m, d=dx)

    # Fourier transform of a Gaussian of width h, evaluated per bandwidth.
    decay = np.exp(
        -2.0
        * (np.pi * freq[np.newaxis, :] * np.asarray(bandwidths)[:, np.newaxis]) ** 2
    )
    dens = irfft(spec[np.newaxis, :] * decay, n=m, axis=-1)[:, :g_n]
    return dens / dx


def _count_modes_kde(samples_1d, /, *, verbose, rng):
    """Count the number of modes via a bandwidth-swept Gaussian KDE.

    A Gaussian KDE is evaluated at ``AUTO_KDE_GRID_POINTS`` equally
    spaced points for each of ``AUTO_KDE_BW_STEPS`` log-spaced bandwidth
    multipliers in ``[AUTO_KDE_BW_LO, AUTO_KDE_BW_HI]`` times the
    Silverman bandwidth.  For each bandwidth the number of local maxima
    is recorded, counting only peaks whose topographic prominence
    exceeds ``AUTO_KDE_MIN_PROMINENCE`` of that bandwidth's peak density
    so that tail ripples are not mistaken for structure.  The **statistical mode** (most frequent value) of
    these counts is returned as the estimated number of data modes,
    making the result robust to any single bandwidth choice.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints).  Inputs larger than
        ``AUTO_KDE_SUBSAMPLE_N`` are thinned by
        :func:`_stratified_subsample` first: KDE evaluation is linear
        in the sample count and is swept over many bandwidths, while
        the quantity being estimated is a small integer.  Stratified
        thinning preserves the shape of the empirical distribution, so
        a low-weight mode is not lost.
    verbose : int
        Verbosity.  ``>= 2`` prints per-bandwidth mode counts.
    rng : numpy.random.Generator
        Explicit stream used for thinning. Deterministic seeding is resolved
        by the fit boundary rather than supplied by this helper.

    Returns
    -------
    int
        Estimated number of modes (``>= 1``).
    """
    R = int(samples_1d.shape[0])
    if R > AUTO_KDE_SUBSAMPLE_N:
        idx = _stratified_subsample(samples_1d, AUTO_KDE_SUBSAMPLE_N, rng)
        samples_1d = np.ascontiguousarray(samples_1d[idx])
        if verbose >= 1:
            print(
                f"  KDE mode counting on {AUTO_KDE_SUBSAMPLE_N}/{R} "
                f"stratified subsample"
            )

    silverman_bw = _silverman_bandwidth(samples_1d)
    if not (np.isfinite(silverman_bw) and silverman_bw > 0):
        # No spread (e.g. all samples identical) — one mode by definition.
        return 1

    lo = float(samples_1d.min())
    hi = float(samples_1d.max())
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        return 1
    margin = AUTO_KDE_GRID_MARGIN * (hi - lo)
    grid = np.linspace(lo - margin, hi + margin, AUTO_KDE_GRID_POINTS)

    multipliers = np.logspace(
        np.log10(AUTO_KDE_BW_LO),
        np.log10(AUTO_KDE_BW_HI),
        AUTO_KDE_BW_STEPS,
    )

    try:
        densities = _binned_kde_sweep(
            samples_1d,
            grid,
            silverman_bw * multipliers,
        )
    except NUMERIC_FAILURES as exc:
        # A degenerate grid or transform is a reason to call the data
        # unimodal, not to fail the fit.
        _reraise_if_debug(exc, "KDE mode counting")
        return 1

    counts = []
    for i, m in enumerate(multipliers):
        density = densities[i]
        # Count local maxima, discarding any whose topographic
        # prominence is negligible relative to the peak density.  A bare
        # local-maximum test treats a ripple on a long tail as a mode,
        # which is how an unfiltered sweep reported four modes on a
        # lognormal sample.
        peak = float(density.max())
        if not (np.isfinite(peak) and peak > 0.0):
            n_modes = 1
        else:
            found, _ = find_peaks(
                density,
                prominence=AUTO_KDE_MIN_PROMINENCE * peak,
            )
            n_modes = max(int(found.size), 1)
        counts.append(n_modes)
        if verbose >= 2:
            print(f"    KDE bw_mult={m:.3f}  modes={n_modes}")

    # Statistical mode of the counts (most frequent value).
    counts_arr = np.array(counts, dtype=np.int64)
    values, freqs = np.unique(counts_arr, return_counts=True)
    k_modes = int(values[np.argmax(freqs)])
    k_modes = max(k_modes, 1)

    if verbose >= 1:
        print(
            f"  KDE mode count: K_modes={k_modes}  "
            f"(counts across {AUTO_KDE_BW_STEPS} bandwidths: "
            f"{sorted(counts)})"
        )

    return k_modes


def _stratified_subsample(samples_1d, m, gen, /):
    """Draw *m* indices stratified by quantile.

    The samples are sorted and split into *m* contiguous blocks of
    (near-)equal size, and one index is drawn uniformly at random from
    each block.  Compared with simple random sampling this has the same
    expectation but much lower variance in the tails, so a low-weight
    mixture component covering (say) 1% of the mass is represented in
    proportion to that mass instead of occasionally vanishing from the
    draw entirely.  Preserving rare components matters here because
    losing one changes the selected component count.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples or interval midpoints.
    m : int
        Number of indices to draw.  Values ``>= R`` return all indices.
    gen : numpy.random.Generator
        Random-number generator used to draw the stratified subsample.

    Returns
    -------
    numpy.ndarray, shape (min(m, R),), dtype intp
        Indices into *samples_1d*, sorted ascending by sample value.
    """
    R = int(samples_1d.shape[0])
    m = int(m)
    if m >= R:
        return np.arange(R, dtype=np.intp)

    order = np.argsort(samples_1d, kind="stable")
    # Block b spans order[edges[b]:edges[b + 1]].
    edges = np.linspace(0, R, m + 1).astype(np.intp)
    widths = np.diff(edges)
    offsets = (gen.random(m) * widths).astype(np.intp)
    offsets = np.minimum(offsets, widths - 1)
    return order[edges[:-1] + offsets]


def _propose_n_components(samples_1d, k_max, rng, /, *, verbose=0):
    """Propose a component count from the KDE mode sweep, and clamp *k_max*.

    This is a structural prior, not a decision: ``_count_modes_kde``
    sweeps a Gaussian KDE across bandwidths and reports the most
    frequent number of prominent local maxima.  The actual choice is
    made by :func:`gibbus._api.selection.select_n_components`, which scores candidate *K*
    values with lightweight log-concave fits centered on this proposal.

    Candidate counts are deliberately scored by the log-concave model family
    itself.  KDE mode counting only centers the candidate range, while
    ``_valley_init_responsibilities`` supplies inexpensive responsibilities for
    candidate fitting when the modal geometry admits a valley decomposition.

    Safeguards
    ----------
    * *k_max* is clamped to ``min(AUTO_K_MAX, n_unique, R // 10)`` where
      *n_unique* is the number of unique sample values and ``R`` the
      sample count, then floored at ``AUTO_K_MIN``.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints).
    k_max : int
        Maximum *K* to consider, before clamping.
    rng : numpy.random.Generator
        Passed to the KDE sweep for stratified subsampling.
    verbose : int, optional
        Verbosity.  ``>= 1`` prints the mode count and the clamped range.

    Returns
    -------
    k_modes : int
        Number of modes detected by the KDE sweep.
    k_max : int
        The clamped maximum, for the caller to bound its search with.
    """
    R = samples_1d.shape[0]
    k_max = min(int(k_max), AUTO_K_MAX)

    # Cap at unique values (for binned data) and at R//10 (for small N).
    n_unique = len(np.unique(samples_1d))
    k_max = min(k_max, max(n_unique, 1), max(R // 10, 1))
    k_max = max(k_max, AUTO_K_MIN)

    k_modes = _count_modes_kde(samples_1d, verbose=verbose, rng=rng)

    if verbose >= 1:
        print(f"  KDE component prior: K_modes={k_modes}  (k_max={k_max})")

    return k_modes, k_max


def _e_step(samples_1d, components, weights, space, /, obs_weights=None):
    """Compute the E-step: per-sample responsibilities and log-likelihood.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point samples (or interval midpoints).
    components : list of _Component
        Current fitted components.
    weights : numpy.ndarray, shape (K,)
        Current mixture weights.
    space : str
        ``"base"`` or ``"exp"``.
    obs_weights : numpy.ndarray, shape (R,) or None, optional
        User-supplied observation weights, normalized to sum to one.
        ``None`` means uniform ``1/R``.  Affects only the returned
        log-likelihood; responsibilities are per-observation posteriors
        and are independent of the observation weight.

    Returns
    -------
    resp : numpy.ndarray, shape (R, K)
        Responsibility matrix.
    ll : float
        Weighted mean log-likelihood.
    """
    K = len(components)
    R = samples_1d.shape[0]
    weighted_pdf = np.empty((R, K), dtype=np.float64)

    for k in range(K):
        view = components[k].base if space == "base" else components[k].exp
        weighted_pdf[:, k] = weights[k] * view.pdf(samples_1d)

    row_sums = weighted_pdf.sum(axis=1)
    row_sums = np.clip(row_sums, TINY_FLOAT, None)

    resp = weighted_pdf / row_sums[:, np.newaxis]
    resp = np.clip(resp, EM_RESP_FLOOR, None)
    resp /= resp.sum(axis=1, keepdims=True)

    log_rs = np.log(row_sums)
    if obs_weights is None:
        ll = float(np.mean(log_rs))
    else:
        ll = float(np.dot(obs_weights, log_rs))
    return resp, ll


def _component_interval_masses(intervals, component, /):
    """Return one finalized component's observation masses/density limits.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval endpoints in user coordinates.
    component : _Component
        Finalized fitted component.

    Returns
    -------
    numpy.ndarray, shape (R,)
        Probability mass for positive-width rows and density for exact point
        limits, suitable for mixture posterior ratios.
    """
    x = np.ascontiguousarray(intervals, dtype=np.float64)
    first, inverse, _ = _row_grouping(x)
    unique = x[first]
    lo = unique[:, 0]
    hi = unique[:, 1]
    exact = lo == hi
    mass_unique = np.empty(unique.shape[0], dtype=np.float64)
    if np.any(exact):
        mass_unique[exact] = np.asarray(component.base.pdf(lo[exact]), dtype=np.float64)
    positive = ~exact
    if np.any(positive):
        c_hi = np.asarray(component.base.cdf(hi[positive]), dtype=np.float64)
        c_lo = np.asarray(component.base.cdf(lo[positive]), dtype=np.float64)
        mass_unique[positive] = np.maximum(c_hi - c_lo, 0.0)
    return mass_unique[inverse]


def _e_step_intervals(intervals, components, weights, /, obs_weights=None):
    """Compute interval-censored EM posteriors from component masses.

    For an observation ``[a_i, b_i]``, the component posterior is
    proportional to ``w_k * P_k(a_i <= X <= b_i)``.  Observation weights
    affect the mean log-likelihood and subsequent M-step, but not these
    per-row posterior probabilities.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval-censored observations.
    components : list of _Component
        Current fitted components.
    weights : numpy.ndarray, shape (K,)
        Current mixture weights.
    obs_weights : numpy.ndarray, shape (R,) or None, optional
        User-supplied relative observation weights, normalized to sum to one.

    Returns
    -------
    resp : numpy.ndarray, shape (R, K)
        Responsibility matrix.
    ll : float
        Weighted mean censored-data log-likelihood.
    """
    R = int(intervals.shape[0])
    K = len(components)
    weighted_mass = np.empty((R, K), dtype=np.float64)
    for k in range(K):
        weighted_mass[:, k] = float(weights[k]) * _component_interval_masses(
            intervals, components[k]
        )

    row_sums = np.clip(weighted_mass.sum(axis=1), TINY_FLOAT, None)
    resp = weighted_mass / row_sums[:, np.newaxis]
    resp = np.clip(resp, EM_RESP_FLOOR, None)
    resp /= resp.sum(axis=1, keepdims=True)

    log_rs = np.log(row_sums)
    if obs_weights is None:
        ll = float(np.mean(log_rs))
    else:
        ll = float(np.dot(obs_weights, log_rs))
    return resp, ll


_POINT_ORDER = None
"""``(weakref to rows, order, inverse)`` for the last point-row array seen by
``_point_order``; a mixture fit asks for the same rows at every EM step and
for every component coordinate.  The weak reference keeps the cache from
holding the data; only read-only arrays are cached, so the rows cannot change
underneath it."""


def _point_order(rows, /):
    """Return ``(order, inverse)`` of point rows, cached per rows array.

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 1)
        Point rows.  The cache is keyed by the array object (weakly), so a
        different array is recomputed.

    Returns
    -------
    order : numpy.ndarray, shape (R,)
        Stable ascending order of the values.
    inverse : numpy.ndarray, shape (R,)
        Index of each row's value among the sorted distinct values.
    """
    global _POINT_ORDER
    entry = _POINT_ORDER
    if entry is not None and entry[0]() is rows:
        return entry[1], entry[2]
    x = np.asarray(rows, dtype=np.float64)[:, 0]
    order = np.argsort(x, kind="stable")
    ranks = np.concatenate(([0], np.cumsum(np.diff(x[order]) != 0.0)))
    inverse = np.empty(x.size, dtype=np.intp)
    inverse[order] = ranks
    order.setflags(write=False)
    inverse.setflags(write=False)
    if isinstance(rows, np.ndarray) and not rows.flags.writeable:
        _POINT_ORDER = (weakref.ref(rows), order, inverse)
    return order, inverse


def _distinct_location_index(rows, /):
    """Return each point row's distinct-location index (see ``_point_order``).

    Parameters
    ----------
    rows : numpy.ndarray, shape (R, 1)
        Point rows.
    """
    return _point_order(rows)[1]


def _effective_distinct_point_count(samples_1d, component_weights, /, *, inverse=None):
    """Return the Kish effective count after collapsing duplicate locations.

    A point-mixture component can make the likelihood diverge by assigning
    essentially all of its posterior mass to one observed location and then
    collapsing its scale.  Counting rows is insufficient when a dataset
    contains duplicate coordinates: one location repeated many times is still
    only one support point for this purpose.  This helper first sums the
    component's non-negative fitting weights over identical coordinates, then
    returns the Kish effective sample size ``(sum w)^2 / sum(w^2)``.

    Parameters
    ----------
    samples_1d : numpy.ndarray, shape (R,)
        Point observations.
    component_weights : numpy.ndarray, shape (R,)
        Responsibility times observation weight for one component.  The
        vector need not be normalized.
    inverse : numpy.ndarray or None, optional
        Precomputed distinct-location index of ``samples_1d`` (see
        ``_distinct_location_index``).

    Returns
    -------
    float
        Effective number of distinct locations carrying component weight.
    """
    x = np.asarray(samples_1d, dtype=np.float64).reshape(-1)
    w = np.asarray(component_weights, dtype=np.float64).reshape(-1)
    if x.shape != w.shape:
        raise ValueError("samples_1d and component_weights must have equal length")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("component_weights must be finite and non-negative")

    if inverse is None:
        _, inverse = np.unique(x, return_inverse=True)
        inverse = inverse.reshape(-1)
    by_location = np.bincount(inverse, weights=w)
    total = float(by_location.sum())
    if not (total > 0.0):
        return 0.0
    sq = float(np.dot(by_location, by_location))
    if not (sq > 0.0):
        return 0.0
    return total * total / sq


def _aggregate_interval_weights(intervals, obs_weights=None, /):
    """Collapse duplicate censoring intervals and aggregate their weights.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval-censored observations.
    obs_weights : numpy.ndarray, shape (R,) or None, optional
        Normalized relative observation weights.

    Returns
    -------
    unique : numpy.ndarray, shape (B, 2)
        Unique interval rows.
    weights : numpy.ndarray, shape (B,)
        Aggregated row weights, normalized to sum to one.
    """
    x = intervals
    first, inverse, _ = _row_grouping(x)
    unique = x[first]
    if obs_weights is None:
        row_w = np.full(x.shape[0], 1.0 / float(x.shape[0]), dtype=np.float64)
    else:
        row_w = obs_weights
    weights = np.bincount(inverse, weights=row_w, minlength=unique.shape[0]).astype(
        np.float64
    )
    return unique, weights


def _interval_observable_dimension(intervals, support, /):
    """Return the dimension of interval probabilities observable from censoring.

    Each interval probability is a CDF difference between two observed
    endpoints.  Treating endpoints as graph vertices and intervals as graph
    edges, the rank of all observable CDF differences is the incidence-matrix
    rank.  Adding the support-wide normalization edge fixes one degree of
    freedom, so the number returned here is ``rank(edges + normalization) - 1``.

    This handles overlapping and nested censoring patterns without building a
    potentially large dense interval-by-atom matrix.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval-censored observations.
    support : tuple of (float, float)
        Model support.

    Returns
    -------
    int
        Number of independent interval-probability coordinates available to
        the likelihood.
    """
    x = np.asarray(intervals, dtype=np.float64)
    first, _, _ = _row_grouping(x)
    unique = x[first]
    lo_s, hi_s = map(float, support)
    endpoints = np.unique(
        np.concatenate((np.asarray([lo_s, hi_s], dtype=np.float64), unique.reshape(-1)))
    )
    n = int(endpoints.size)
    if n <= 1:
        return 0

    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(i):
        i = int(i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri == rj:
            return
        if rank[ri] < rank[rj]:
            ri, rj = rj, ri
        parent[rj] = ri
        if rank[ri] == rank[rj]:
            rank[ri] += 1

    starts = np.searchsorted(endpoints, unique[:, 0])
    ends = np.searchsorted(endpoints, unique[:, 1])
    for i, j in zip(starts, ends, strict=True):
        if i != j:
            union(int(i), int(j))

    i_s = int(np.searchsorted(endpoints, lo_s))
    j_s = int(np.searchsorted(endpoints, hi_s))
    if i_s != j_s:
        union(i_s, j_s)

    n_components = len({find(i) for i in range(n)})
    incidence_rank = n - n_components
    return max(0, int(incidence_rank - 1))


def _interval_nonparametric_loglik_bound(
    intervals, support, /, obs_weights=None, *, max_iter, tol
):
    """Return a rigorous upper bound on the nonparametric interval log-likelihood.

    The censoring endpoints partition the support into atomic intervals.  The
    unrestricted likelihood is concave in their probability masses.  At a
    feasible mass vector ``p``, the Frank-Wolfe duality gap
    ``max_j dl/dp_j - p . grad l`` bounds all remaining ascent.  Because the
    aggregated observation weights sum to one, ``p . grad l = 1`` and the
    upper bound is the current log-likelihood plus ``max(cover) - 1``.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval-censored observations.
    support : tuple of (float, float)
        Model support.
    obs_weights : numpy.ndarray, shape (R,) or None, optional
        Normalized relative observation weights.
    max_iter : int
        Maximum Turnbull iterations.
    tol : float
        Stop once the Frank-Wolfe duality gap is at most this value.

    Returns
    -------
    float
        Rigorous upper bound on the maximum weighted mean interval
        log-likelihood over all probability distributions on the
        endpoint-induced atoms.
    """
    unique, row_w = _aggregate_interval_weights(intervals, obs_weights)
    lo_s, hi_s = map(float, support)
    endpoints = np.unique(
        np.concatenate((np.asarray([lo_s, hi_s], dtype=np.float64), unique.reshape(-1)))
    )
    starts = np.searchsorted(endpoints, unique[:, 0]).astype(np.int64)
    ends = np.searchsorted(endpoints, unique[:, 1]).astype(np.int64)
    n_atoms = int(endpoints.size - 1)
    if n_atoms <= 0:
        return 0.0

    p = np.full(n_atoms, 1.0 / float(n_atoms), dtype=np.float64)
    best_upper = np.inf
    for _ in range(int(max_iter)):
        prefix = np.empty(n_atoms + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(p, out=prefix[1:])
        masses = prefix[ends] - prefix[starts]
        if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
            break
        ll = float(np.dot(row_w, np.log(masses)))

        coeff = row_w / masses
        delta = np.zeros(n_atoms + 1, dtype=np.float64)
        np.add.at(delta, starts, coeff)
        np.add.at(delta, ends, -coeff)
        cover = np.cumsum(delta[:-1])
        gap = max(float(np.max(cover)) - 1.0, 0.0)
        best_upper = min(best_upper, ll + gap)
        if gap <= tol:
            break

        # Turnbull EM: each row distributes its observed mass across the
        # atoms lying inside that interval in proportion to the current p.
        p_new = p * cover
        total = float(np.sum(p_new))
        if not (total > 0.0 and np.isfinite(total)):
            break
        p = p_new / total

    if not np.isfinite(best_upper):
        return float(best_upper)
    # Round upward so floating-point evaluation cannot turn the certificate
    # into a numerical under-bound.
    margin = 64.0 * np.finfo(float).eps * (1.0 + abs(best_upper))
    return float(best_upper + margin)


def _interval_identifiability_diagnostic(
    intervals, components, ll, support, /, obs_weights=None, *, n_parameters
):
    """Detect an exactly saturated censored likelihood with excess parameters.

    The fitted mixture is compared with the nonparametric maximum likelihood
    attainable from the same censoring pattern.  When the parametric fit
    reaches that bound while carrying more free parameters than independent
    interval-probability coordinates, the within-interval component structure
    is not identified by the observations.  The calculation applies to
    disjoint, overlapping, and nested censoring intervals.

    Parameters
    ----------
    intervals : numpy.ndarray, shape (R, 2)
        Interval-censored observations.
    components : sequence of _Component
        Current EM components; lite and fully finalized states are accepted.
    ll : float
        Current weighted mean interval log-likelihood.
    support : tuple of (float, float)
        Shared model support.
    obs_weights : numpy.ndarray, shape (R,) or None, optional
        Normalized relative observation weights.
    n_parameters : int
        Authoritative selected-face model dimension, counting shared boundary
        amplitudes once and including fitted mixture-weight degrees of freedom.

    Returns
    -------
    dict or None
        Diagnostic fields when the fit reaches the nonparametric likelihood
        bound with excess model dimension; otherwise ``None``.
    """
    x = np.asarray(intervals, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or x.shape[0] == 0 or not np.isfinite(ll):
        return None

    n_params = int(n_parameters)
    if n_params < 0 or n_params != n_parameters:
        raise ValueError("n_parameters must be a non-negative integer")

    observable_dim = _interval_observable_dimension(x, support)
    if n_params <= observable_dim:
        return None

    bound = _interval_nonparametric_loglik_bound(
        x,
        support,
        obs_weights=obs_weights,
        max_iter=TURNBULL_MAX_ITER,
        tol=TURNBULL_GAP_TOL,
    )
    if not np.isfinite(bound):
        return None

    gap = float(bound - float(ll))
    # At an exact censored-data maximum this gap is zero.  The tolerance is
    # deliberately absolute on the per-observation log-likelihood scale; it
    # is far below ordinary model-misspecification gaps but above the expected
    # final EM/quadrature noise of coarse-bin fits.
    if gap < -1e-6 or gap > 1e-6:
        return None

    first, _, _ = _row_grouping(x)
    unique = x[first]
    # _row_grouping is lexicographic with the lower endpoint primary.
    ordered = unique
    has_overlap = bool(
        ordered.shape[0] > 1 and np.any(ordered[:-1, 1] > ordered[1:, 0])
    )
    return {
        "n_intervals": int(unique.shape[0]),
        "observable_dim": int(observable_dim),
        "n_params": int(n_params),
        "nonparametric_ll": float(bound),
        "fit_ll": float(ll),
        "gap": gap,
        "has_overlap": has_overlap,
    }


# ======================================================================
# Runtime-cache structured-state helpers
# ======================================================================


def _pack_mixture_struct(
    weights,
    default_space,
    comp_states,
    /,
    base_modes=None,
    *,
    mu=0.0,
    sigma=1.0,
    fit_metadata=None,
):
    """Pack the uniform model envelope for every component count, including one.

    Any component field whose one-dimensional shape differs across components
    is padded to the maximum length and accompanied by a ``comp_<name>_len``
    vector.  Spectral coefficient/panel arrays are deliberately stored flat in
    each component state, so this generic rule covers all variable-size fields.

    Parameters
    ----------
    weights : numpy.ndarray, shape (K,)
        Mixture weights, summing to one.
    default_space : str
        Coordinate view the mixture reports in by default.
    comp_states : sequence of numpy.void
        Per-component fitted structured scalars.
    base_modes : sequence of float or None, optional
        Mixture modes in base coordinates, when already known.  Stored as a
        1-D ``base_modes`` array with matching ``n_modes``; ``None`` stores an
        empty array and ``n_modes == 0``.
    mu, sigma : float, optional
        The one accumulated model pushforward map.
    fit_metadata : dict or None, optional
        Validated model dimensions, shared boundary inference and provenance.
        Without supplied fitting metadata, the assembled model is derived;
        equality of expanded amplitudes alone cannot establish fit provenance.

    Returns
    -------
    numpy.void
        Single structured scalar holding the whole mixture.
    """
    metadata = _model_metadata(comp_states, fit_metadata)
    K = len(comp_states)
    field_names = list(comp_states[0].dtype.names)
    fields = {}

    for name in field_names:
        arrs = [np.asarray(cs[name]) for cs in comp_states]
        same_shape = all(a.shape == arrs[0].shape for a in arrs)
        if same_shape:
            fields[f"comp_{name}"] = np.stack(arrs, axis=0)
            continue

        if not all(a.ndim == 1 for a in arrs):
            raise ValueError(
                f"mixture component field {name!r} has incompatible ragged shape; "
                "variable component fields must be stored one-dimensionally"
            )
        max_len = max(a.size for a in arrs)
        dtype = np.result_type(*[a.dtype for a in arrs])
        if np.issubdtype(dtype, np.floating):
            fill = np.nan
        elif np.issubdtype(dtype, np.integer):
            fill = 0
        else:
            fill = 0
        padded = np.full((K, max_len), fill, dtype=dtype)
        lengths = np.empty(K, dtype=np.int64)
        for j, a in enumerate(arrs):
            lengths[j] = a.size
            padded[j, : a.size] = a
        fields[f"comp_{name}"] = padded
        fields[f"comp_{name}_len"] = lengths

    fields["weights"] = np.asarray(weights, dtype=np.float64)
    fields["n_components"] = np.int64(K)
    fields["default_space"] = np.array(str(default_space), dtype="U4")
    fields["mu"] = np.float64(mu)
    fields["sigma"] = np.float64(sigma)
    fields["provenance"] = np.asarray(metadata["provenance"], dtype="U7")
    fields["n_parameters"] = np.int64(metadata["n_parameters"])
    fields["n_face_parameters"] = np.int64(metadata["n_face_parameters"])
    for name in ("allowed", "amplitudes", "active", "standard_errors", "p_values"):
        fields[f"shared_boundary_{name}"] = np.asarray(
            metadata["shared_boundary"][name]
        )

    if base_modes is not None:
        n_modes = len(base_modes)
        mode_arr = np.asarray(base_modes, dtype=np.float64).reshape(-1).copy()
        fields["base_modes"] = mode_arr
        fields["n_modes"] = np.int64(n_modes)
    else:
        fields["base_modes"] = np.empty(0, dtype=np.float64)
        fields["n_modes"] = np.int64(0)

    dtype = []
    for key, value in fields.items():
        arr = np.asarray(value)
        dtype.append((key, arr.dtype) if arr.ndim == 0 else (key, arr.dtype, arr.shape))
    struct = np.zeros((), dtype=dtype)
    for key, value in fields.items():
        struct[key] = value
    return struct


def _unpack_mixture_struct(struct):
    """Unpack weights, default space, and per-component runtime states.

    Inverse of :func:`_pack_mixture_struct`: padded component fields are
    trimmed back to their per-component lengths using the stored
    ``comp_<name>_len`` vectors.

    Parameters
    ----------
    struct : numpy.void
        Packed mixture structured scalar.

    Returns
    -------
    weights : numpy.ndarray, shape (K,)
    default_space : str
    comp_states : list of numpy.void
        Reconstructed per-component structured scalars.
    """
    if np.asarray(struct).shape != () or struct.dtype.hasobject:
        raise ValueError("model state must be a non-object structured scalar")
    names = list(struct.dtype.names or ())
    missing = [
        name
        for name in (
            "n_components",
            "weights",
            "default_space",
            "base_modes",
            "n_modes",
            "mu",
            "sigma",
            "provenance",
            "n_parameters",
            "n_face_parameters",
            "shared_boundary_allowed",
            "shared_boundary_amplitudes",
            "shared_boundary_active",
            "shared_boundary_standard_errors",
            "shared_boundary_p_values",
        )
        if name not in names
    ]
    if missing:
        raise ValueError(
            "not a gibbus mixture state; missing fields: " + ", ".join(missing)
        )
    if any(
        name in names
        for name in (
            "comp_mu",
            "comp_sigma",
            "comp_pullback",
            "comp_default_space",
            "pullback",
        )
    ):
        raise ValueError("model state contains duplicate presentation fields")
    count = np.asarray(struct["n_components"])
    if count.shape != () or count.dtype.kind not in "iu" or int(count) < 1:
        raise ValueError("model n_components must be a positive integer")
    K = int(count)
    mu, sigma = float(struct["mu"]), float(struct["sigma"])
    if not np.isfinite(mu):
        raise ValueError("state mu must be finite")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("state sigma must be finite and positive")
    weights = np.asarray(struct["weights"], dtype=np.float64).reshape(-1)
    default_space = str(struct["default_space"])
    if default_space not in ("base", "exp"):
        raise ValueError("state default_space must be 'base' or 'exp'")
    if (
        np.asarray(struct["weights"]).shape != (K,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("model state has invalid component weights")

    comp_field_names = [
        fn[5:] for fn in names if fn.startswith("comp_") and not fn.endswith("_len")
    ]

    comp_states = []
    for name in comp_field_names:
        raw = np.asarray(struct[f"comp_{name}"])
        if raw.ndim < 1 or raw.shape[0] != K:
            raise ValueError(f"component field {name} must have n_components rows")
        length_name = f"comp_{name}_len"
        if length_name in names:
            lengths = np.asarray(struct[length_name])
            if (
                raw.ndim != 2
                or lengths.shape != (K,)
                or lengths.dtype.kind not in "iu"
                or np.any(lengths < 0)
                or np.any(lengths > raw.shape[1])
            ):
                raise ValueError(f"component field {name} has invalid lengths")
    for j in range(K):
        comp_dict = {}
        for name in comp_field_names:
            raw = np.asarray(struct[f"comp_{name}"])[j]
            len_field = f"comp_{name}_len"
            if len_field in names:
                raw = raw[: int(struct[len_field][j])]
            comp_dict[name] = raw

        dt = []
        for name in comp_field_names:
            arr = np.asarray(comp_dict[name])
            dt.append(
                (name, arr.dtype) if arr.ndim == 0 else (name, arr.dtype, arr.shape)
            )
        cs = np.zeros((), dtype=dt)
        for name in comp_field_names:
            cs[name] = comp_dict[name]
        comp_states.append(cs)

    return weights, default_space, comp_states


# ======================================================================
# Component ordering helper
# ======================================================================


def _sort_components_by_mode(components, weights, /):
    """Re-order *components* and *weights* by ascending base-space mode.

    Parameters
    ----------
    components : list of _Component
        Fitted mixture components to order by their base-space modes.
    weights : numpy.ndarray, shape (K,)
        Mixture weights corresponding one-to-one with ``components``.

    Returns
    -------
    components : list of _Component
    weights : numpy.ndarray, shape (K,)
    """
    modes = [c.base.mode for c in components]
    order = sorted(range(len(modes)), key=lambda i: modes[i])
    components = [components[i] for i in order]
    weights = weights[order]
    return components, weights


# ======================================================================
# Mixture mode finding
# ======================================================================


def _find_mixture_modes(
    neg_log_base_func, component_seed_modes, /, *, space, vectorized
):
    """Find all modes of a mixture PDF, in base or exp coordinates.

    A base-space mode is a local maximum of the PDF: a point where
    ``q'(x) = 0`` and ``q''(x) > 0``, writing ``q`` for the mixture
    negative-log-density.  Under ``y = exp(x)`` the change of variables
    contributes an extra ``-log y`` term, so an exp-space mode instead
    satisfies ``q'(x) + 1 = 0``, with the mode reported at ``exp(x)``.
    The two searches are otherwise identical, so they share this
    implementation.

    Search interval:

    * **base** — ``[min(modes), max(modes)]``.  Outside that interval
      every component density is monotone, so the mixture derivative
      has a definite sign and no root can hide there.
    * **exp** — the same interval widened by half its span on each
      side.  The ``+1`` offset shifts roots away from the component
      modes, so they can fall outside the base-space hull.

    Parameters
    ----------
    neg_log_base_func : callable
        ``neg_log_base_func(x, n)`` returning the *n*-th derivative of
        the mixture negative-log-density in **base** coordinates.
    component_seed_modes : sequence of float
        Per-component stationary-point seeds in base/log coordinates.  For
        base-space searches these are the component base modes; for exp-space
        searches they are the component roots of ``q'(x) + 1 = 0``.
    space : {"base", "exp"}
        Coordinate space of the returned modes.
    vectorized : bool
        Explicitly declare whether the potential accepts an array of scan
        coordinates. False uses scalar evaluations without exception probing.

    Returns
    -------
    tuple of float
        Mode locations in ascending order, in the requested space.

    Raises
    ------
    ValueError
        If *space* is not ``"base"`` or ``"exp"``.
    """
    if space not in ("base", "exp"):
        raise ValueError(f"space must be 'base' or 'exp', got {space!r}.")

    is_exp = space == "exp"
    # Root target: q'(x) in base space, q'(x) + 1 in exp space.
    offset = 1.0 if is_exp else 0.0

    def g(x):
        return float(neg_log_base_func(x, 1)) + offset

    def g_many(x):
        return np.asarray(neg_log_base_func(x, 1), dtype=np.float64) + offset

    def height(x):
        """Value to minimize when breaking ties between near-duplicate roots.

        The exp-space potential is ``q(x) + x`` (from the Jacobian).
        """
        v = float(neg_log_base_func(x, 0))
        return v + x if is_exp else v

    def out(x):
        return float(np.exp(x)) if is_exp else float(x)

    modes_sorted = sorted(set(component_seed_modes))
    K = len(modes_sorted)
    if K == 0:
        return ()

    if K == 1:
        m = modes_sorted[0]
        if not is_exp:
            # q' vanishes at the component mode by construction.
            return (out(m),)
        # In exp space the root is displaced from the component mode,
        # so it must actually be solved for.
        if (
            abs(g(m)) < MODE_XTOL * 100
            and float(neg_log_base_func(m, 2)) > -MODE_DERIV_TOL
        ):
            return (out(m),)
        half = max(1.0, abs(m) * 0.1)
        lo, hi = m - half, m + half
        if g(lo) * g(hi) < 0:
            try:
                root = brentq(g, lo, hi, xtol=MODE_XTOL, maxiter=200)
            except NUMERIC_FAILURES as exc:
                # No bracketed root in the search window, so fall back
                # to the component mode below.
                _reraise_if_debug(exc, "exp-space single-mode brentq")
            else:
                if float(neg_log_base_func(root, 2)) > -MODE_DERIV_TOL:
                    return (out(root),)
        return (out(m),)

    # ---- search interval ----
    search_lo, search_hi = modes_sorted[0], modes_sorted[-1]
    if is_exp:
        span = search_hi - search_lo
        search_lo -= 0.5 * span
        search_hi += 0.5 * span

    # ---- collect all critical-point candidates ----
    candidates = list(modes_sorted)
    brackets = [search_lo] + modes_sorted + [search_hi] if is_exp else modes_sorted
    for i in range(len(brackets) - 1):
        _collect_roots_bisection_func(
            g,
            brackets[i],
            brackets[i + 1],
            candidates,
            g_many if vectorized else None,
        )

    # ---- refine each candidate with brentq ----
    refined = []
    for c in candidates:
        half = max(MODE_XTOL * 10, abs(c) * 1e-8)
        g_c = g(c)
        if abs(g_c) < MODE_XTOL * 100:
            # A candidate inside the stationarity band accepted below can
            # still sit |g| / q'' from the mixture root -- about 1e-8 at
            # O(1) curvature, far outside the fixed half-width for |c| < 1.
            # Cover that Newton step so the candidate (typically a component
            # seed mode) refines onto the root the scan already located,
            # rather than surviving unrefined as a near-duplicate mode.
            curvature = float(neg_log_base_func(c, 2))
            if np.isfinite(curvature) and curvature > MODE_DERIV_TOL:
                half = max(half, 2.0 * abs(g_c) / curvature)
        lo = max(c - half, search_lo)
        hi = min(c + half, search_hi)
        if g(lo) * g(hi) < 0:
            try:
                root = brentq(g, lo, hi, xtol=MODE_XTOL, maxiter=200)
            except NUMERIC_FAILURES as exc:
                # Refinement is optional: the unrefined candidate is
                # still tested for stationarity just below.
                _reraise_if_debug(exc, f"{space}-space mode refinement")
            else:
                refined.append(float(root))
                continue
        if abs(g_c) < MODE_XTOL * 100:
            refined.append(float(c))

    # ---- classify: keep only local minima of the potential ----
    kept = [r for r in refined if float(neg_log_base_func(r, 2)) > -MODE_DERIV_TOL]

    # ---- deduplicate within tolerance, keeping the taller peak ----
    kept.sort()
    deduped = []
    for r in kept:
        # Brent stops once its bracket is narrower than xtol + 4*eps*|x|, so
        # two refinements of one root may land on opposite sides of it: up to
        # twice that apart.  Merge on a window that covers both.
        window = 4.0 * (MODE_XTOL + 4.0 * np.finfo(np.float64).eps * abs(r))
        if deduped and abs(r - deduped[-1]) <= window:
            if height(r) < height(deduped[-1]):
                deduped[-1] = r
        else:
            deduped.append(r)

    if not deduped:
        # No critical point survived; fall back to the tallest
        # component mode so that callers always get at least one.
        deduped = [min(modes_sorted, key=height)]

    return tuple(out(r) for r in deduped)


def _find_mixture_modes_base(neg_log_func, component_base_modes, /, *, vectorized):
    """Find all base-space modes of a mixture PDF.

    Thin wrapper over :func:`_find_mixture_modes`; see that function
    for the algorithm.

    Parameters
    ----------
    neg_log_func : callable
        ``neg_log_func(x, n)`` -- *n*-th derivative of the mixture
        negative-log-density in base space.
    component_base_modes : sequence of float
        Per-component base-space modes used as root-search seeds.
    vectorized : bool
        Whether the supplied potential supports array-valued scan coordinates.

    Returns
    -------
    tuple of float
        Sorted base-space mode locations.
    """
    return _find_mixture_modes(
        neg_log_func, component_base_modes, space="base", vectorized=vectorized
    )


def _find_mixture_modes_exp(neg_log_base_func, component_log_modes, /, *, vectorized):
    """Find all exp-space modes of a mixture PDF.

    Thin wrapper over :func:`_find_mixture_modes`; see that function
    for the algorithm.

    Parameters
    ----------
    neg_log_base_func : callable
        ``neg_log_base_func(x, n)`` -- *n*-th derivative of the mixture
        negative-log-density in **base** space.
    component_log_modes : sequence of float
        Per-component exp-space modes expressed in base/log coordinates.
    vectorized : bool
        Whether the supplied base potential supports array-valued scan coordinates.

    Returns
    -------
    tuple of float
        Sorted exp-space mode locations.
    """
    return _find_mixture_modes(
        neg_log_base_func, component_log_modes, space="exp", vectorized=vectorized
    )


def _collect_roots_bisection_func(g_func, a, b, candidates, g_many, /):
    """Find sign-change roots of ``g_func`` by deterministic scanning.

    Recursive endpoint/midpoint sign tests can miss an even number of roots in
    one subinterval when all three sampled values have the same sign.  Sample a
    fixed dense grid instead, then refine every observed sign change with
    Brent's method.  Mode finding is lazy and low-dimensional, so the extra
    scalar evaluations are preferable to a topology-dependent miss.

    Parameters
    ----------
    g_func : callable
        Scalar function whose sign-change roots are sought.
    a : float
        Lower scan endpoint.
    b : float
        Upper scan endpoint.
    candidates : array_like
        Candidate scan points inside the interval.
    g_many : callable or None
        Explicit vectorized ``g_func`` contract, with matching grid shape.
        ``None`` selects a scalar scan. Supplied evaluator errors propagate.
    """
    a = float(a)
    b = float(b)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return

    grid = np.linspace(a, b, 1025, dtype=np.float64)
    if g_many is None:
        vals = np.array([float(g_func(x)) for x in grid], dtype=np.float64)
    else:
        vals = np.asarray(g_many(grid), dtype=np.float64)
        if vals.shape != grid.shape:
            raise ValueError("vectorized root evaluator must match the scan grid shape")

    for x, fx in zip(grid, vals, strict=True):
        if np.isfinite(fx) and abs(fx) <= MODE_XTOL:
            candidates.append(float(x))

    for i in range(grid.size - 1):
        lo, hi = float(grid[i]), float(grid[i + 1])
        flo, fhi = float(vals[i]), float(vals[i + 1])
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            continue
        if flo == 0.0 or fhi == 0.0 or flo * fhi >= 0.0:
            continue
        try:
            root = brentq(g_func, lo, hi, xtol=MODE_XTOL, maxiter=200)
        except NUMERIC_FAILURES as exc:
            _reraise_if_debug(exc, "mode scan root refinement")
            root = 0.5 * (lo + hi)
        candidates.append(float(root))
