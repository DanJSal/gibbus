# gibbus

**Flexible maximum-likelihood modeling of smooth univariate log-concave distributions.**

`gibbus` fits explicit, support-aware analytic probability distributions to point or interval-censored data without requiring a named parametric family such as Normal or Gamma. A fitted `Distribution` represents a **Gibbus distribution**. A single component is log-concave and therefore unimodal; finite mixtures extend the model to multimodal data.

Within each component, the convex negative-log density (potential) is represented by support-aware polynomial structure together with optional finite-boundary logarithmic terms. Mixture components share one exponent per enabled physical boundary, while retaining their own polynomial shapes and numerical fitting coordinates. Increasing the polynomial degree gives progressively richer smooth shapes, so the model is designed to approximate a broad range of smooth log-concave behavior over the probability-mass region rather than assuming that the data-generating distribution itself has a low-degree polynomial potential.

The fitted distribution has an **explicit analytic representation on the interior of its support**: its density and potential are smooth functions of the fitted parameters, not a grid, histogram, KDE, spline, or piecewise-linear log-density. Fitting uses analytic first- and second-order derivatives, including the full Hessian, while the post-fit numerical layer provides accurate CDFs, quantiles, moments, sampling, survival quantities, information measures, and extreme-tail evaluation from that analytic model.

The public API is the **`Distribution`** class, plus two module-level helpers for inspecting numerical fallbacks (`suppressed_failures`, `clear_suppressed_failures`).

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
  - [Unimodal (single-component)](#unimodal-single-component)
  - [Automatic component selection (opt-in)](#automatic-component-selection-opt-in)
  - [Explicit component count](#explicit-component-count)
- [API Reference](#api-reference)
  - [Construction](#construction)
  - [Module-Level Helpers](#module-level-helpers)
  - [Fitting — `Distribution.fit()`](#fitting--distributionfit)
  - [Evaluation Methods](#evaluation-methods)
  - [Survival, Reliability, and Tail Analysis](#survival-reliability-and-tail-analysis)
  - [Model Checking — `goodness_of_fit()`](#model-checking--goodness_of_fit)
  - [Uncertainty — `bootstrap_bands()`](#uncertainty--bootstrap_bands)
  - [Summary Statistics (Properties)](#summary-statistics-properties)
  - [Coordinate Spaces — Base vs Exp](#coordinate-spaces--base-vs-exp)
  - [Affine Transforms — `Distribution.transform()`](#affine-transforms--distributiontransform)
  - [Serialization — Save & Load](#serialization--save--load)
  - [Copying](#copying)
  - [Mixture-Specific API](#mixture-specific-api)
  - [Fit and Selection Diagnostics](#fit-and-selection-diagnostics)
  - [Diagnostics — `Distribution.spectral_diagnostics`](#diagnostics--distributionspectral_diagnostics)
- [Detailed Usage Examples](#detailed-usage-examples)
  - [Point Samples with Infinite Support](#point-samples-with-infinite-support)
  - [Bounded Support](#bounded-support)
  - [Interval-Censored Data](#interval-censored-data)
  - [Weighted Samples](#weighted-samples)
  - [Warm-Starting from a Previous Fit](#warm-starting-from-a-previous-fit)
  - [Exp-Space for Positive Data](#exp-space-for-positive-data)
  - [Affine Transforms](#affine-transforms)
  - [Multi-Component Mixture Fitting](#multi-component-mixture-fitting)
  - [Accessing Individual Components](#accessing-individual-components)
  - [Saving and Loading Models](#saving-and-loading-models)
  - [Computing Moments](#computing-moments)
  - [Evaluating the Potential (Negative-Log Density)](#evaluating-the-potential-negative-log-density)
- [Input Formats](#input-formats)
- [Support Specification](#support-specification)
- [Error Handling](#error-handling)
- [Performance Notes](#performance-notes)
- [License](#license)

---

## Features

- **Flexible log-concave family**: No named distributional family is required. Log-concavity is the primary per-component shape assumption; increasing polynomial degree provides progressively richer smooth convex potentials, while optional finite-boundary logarithmic terms capture non-polynomial boundary behavior. The polynomial machinery is a flexible representation, not an assumption that the target distribution itself is polynomial.
- **Fully analytic fitted model (on the support interior)**: On the interior of its support, the fitted potential and PDF are analytic functions represented directly by the model rather than by a grid, histogram, KDE, spline, or piecewise-linear log-density. Finite-boundary logarithmic terms provide controlled endpoint behavior when enabled.
- **Globally enforced shape constraint**: Convexity of the negative-log density is enforced through an exact support-specific conic description of the full natural curvature, giving a log-concave and therefore unimodal single component without relying on a finite grid of inequality checks.
- **Second-order maximum-likelihood fitting**: The optimizer uses analytic gradients and the full analytic Hessian, together with exact algebraic sufficient-statistic reduction for point observations.
- **Support-aware modeling**: Full-line, half-line, and bounded supports use representations adapted to their geometry rather than post-hoc clipping.
- **Point and interval data**: Handles exact observations and interval-censored (binned) observations; interval observations enter through interval probability masses rather than midpoint substitutions.
- **Weighted samples**: Supports non-negative observation weights.
- **Mixture models**: Fit multimodal data with `n_components > 1` using an EM algorithm, or opt into automatic component-count selection with `n_components="auto"`. The default, `n_components=1`, fits a single log-concave component. Each mixture component retains the smooth log-concave model; the mixture as a whole need not be log-concave.
- **Full distributional interface**: `pdf`, `logpdf`, `cdf`, `logcdf`, `sf`, `logsf`, `ppf`, `isf`, log-probability quantiles, `sample`, `moment`, `cumulant`, intervals/HPD regions, scoring, information measures, and summary statistics.
- **Tail-aware numerics**: Dedicated log-domain survival and quantile machinery avoids reducing extreme-tail calculations to numerically fragile expressions such as `1 - cdf(x)`.
- **Survival analysis**: Tail-accurate hazard, cumulative hazard, mean residual life, and residual entropy. A single fitted log-concave component is IFR by construction, so its hazard is non-decreasing; mixtures do not inherit that guarantee.
- **Two coordinate views**: Evaluate distributions in base coordinates (modeling *x* directly) or exp coordinates (modeling *y = exp(x)*, useful for positive data such as prices or eigenvalues).
- **Affine transforms**: Shift and scale a fitted distribution without re-fitting.
- **Serialization**: Save a fitted model as a NumPy structured array and reload it later; fitted `Distribution` objects also support ordinary Python pickling through that stable state representation.
- **Warm-starting**: Initialize a new fit from a previously fitted model.
- **Cython-accelerated kernels**: Performance-critical quadrature, polynomial evaluation, spectral evaluation, and certification routines are implemented in compiled Cython extensions.

---

## Installation

Clone the repository and install from the source checkout. Building compiles the Cython extensions and therefore requires a C compiler and Python development headers (`build-essential` and `python3-dev` on Debian/Ubuntu, the Xcode command-line tools on macOS, MSVC Build Tools on Windows). Cython itself is declared as a build dependency.

```bash
git clone https://github.com/DanJSal/gibbus.git
cd gibbus
python -m pip install .
```

For editable source development:

```bash
python -m pip install -e .
```

Free-threaded CPython (the `t` builds, such as 3.14t) is not supported or tested yet. The compiled extensions do not declare free-threading compatibility; a source build may therefore re-enable the GIL when `gibbus` is imported. Do not treat an installation on a `t` build as no-GIL support.

On GCC/Clang builds, set `GIBBUS_NATIVE_ARCH=1` at build time to add `-march=native`. This can enable wider machine-specific SIMD and other target tuning, but produces machine-specific binaries, so it is off by default.

---

## Dependencies

- **Python** ≥ 3.10
- **NumPy**
- **SciPy**

---

## Quick Start

### Unimodal (single-component)

```python
import numpy as np
from gibbus import Distribution

rng = np.random.default_rng(0)
samples = rng.normal(size=500)

c = Distribution().fit(samples, support=(-np.inf, np.inf))

# Summary statistics
c.mean, c.std, c.mode, c.median

# Evaluate the density
c.pdf(0.0)  # scalar → float
c.pdf([-1, 0, 1])  # array  → ndarray

# CDF and quantile function
c.cdf(0.0)  # ≈ 0.5
c.ppf(0.25)  # lower quartile

# Draw random samples
c.sample(100, rng=rng)
```

### Automatic component selection (opt-in)

By default, `fit()` uses `n_components=1` and fits a single unimodal density. Pass `n_components="auto"` to select the number of components. A KDE bandwidth sweep estimates the number of data modes and focuses the candidate range. Lightweight shared-boundary fits screen that range; competitive candidates are then refined with the requested degree and boundary-selection policies before their BIC scores choose *K*. This is a staged search, not an exhaustive or globally certified search. For interval-censored data, candidates are scored on their actual interval probability masses rather than midpoint-density surrogates. When selection uses a subsample, the winner is refitted on all observations:

```python
bimodal = np.concatenate(
    [
        rng.normal(-2, 0.5, 300),
        rng.normal(2, 0.5, 300),
    ]
)

c2 = Distribution().fit(bimodal, n_components="auto", support=(-np.inf, np.inf))

c2.n_components  # automatically chosen
c2.modes  # tuple of mode locations
c2.weights  # array of mixture weights
```

Use `k_max` to limit the search range:

```python
c3 = Distribution().fit(
    bimodal, n_components="auto", support=(-np.inf, np.inf), k_max=5
)
```

### Explicit component count

Specify an explicit number of components to fit a mixture without selection:

```python
c2 = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf))

c2.pdf(0.0)  # valley between the two modes
c2.n_components  # 2
c2.weights  # array of mixture weights
c2.modes  # tuple of mode locations
```

---

## API Reference

The reference snippets below share this setup:

```python
import numpy as np
import gibbus
from gibbus import Distribution

rng = np.random.default_rng(0)
data = rng.normal(size=500)  # training sample
holdout = rng.normal(size=300)  # held-out observations
x = np.array([-1.0, 0.0, 2.5])  # evaluation points

c = Distribution().fit(data, support=(-np.inf, np.inf))
other = Distribution().fit(rng.normal(0.3, 1.2, 500), support=(-np.inf, np.inf))
```

### Construction

```python
Distribution(state=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `numpy.void` or `None` | A previously saved structured state (from `.data`). If `None`, creates an unfitted instance. |

**Properties on an unfitted instance:**

| Property | Description |
|----------|-------------|
| `.is_fitted` | `False` until `.fit()` or `.load()` is called. |

Calling evaluation methods (`.pdf()`, `.cdf()`, etc.) or accessing summary statistics on an unfitted `Distribution` raises `RuntimeError`.

A fitted `Distribution` may be shared between threads after its lazy caches have been
warmed. Concurrent first-use cache construction on the same instance is not
synchronized; use one instance per thread during cold-cache initialization.

---

### Module-Level Helpers

Two functions sit alongside `Distribution` at module level. Both concern numerical fallbacks — steps that degrade gracefully rather than failing — and neither requires an environment variable, so they work on an ordinary run.

```python
gibbus.suppressed_failures()
gibbus.clear_suppressed_failures()
```

| Function | Returns | Description |
|----------|---------|-------------|
| `suppressed_failures()` | `list[dict]` | Fallbacks taken since the record was last cleared, most recent last. Each entry has `context`, `type`, `message`, and `routine`; the last field marks an expected guard path rather than an unexpected degradation. The record is thread-local and capped. |
| `clear_suppressed_failures()` | `None` | Empties the record. Call it before a fit to scope the result to that fit. |

```python
gibbus.clear_suppressed_failures()
c = gibbus.Distribution().fit(data)
for f in gibbus.suppressed_failures():
    print(f["context"], "->", f["type"], f["message"])
```

The `routine` field distinguishes expected guard paths from unexpected degradations, but context and frequency still matter. A non-routine fallback on well-behaved data, or any fallback count that climbs unexpectedly with input size, is worth investigating.

See [Debugging silent fallbacks](#debugging-silent-fallbacks) for how these interact with `GIBBUS_DEBUG` and `GIBBUS_DEBUG_STRICT`.

---

### Fitting — `Distribution.fit()`

```python
c = Distribution().fit(data, support=(-np.inf, np.inf))  # fit(samples, **options)
```

All parameters except `samples` are keyword-only. Returns `self` for method chaining.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `samples` | array_like | *(required)* | Observations. Shape `(R,)` or `(R,1)` for point samples; `(R,2)` for interval-censored samples. Point samples must be finite. Interval endpoints may be `-np.inf` or `np.inf` for one-sided censoring, but may not be NaN; an infinite zero-width row is invalid. Reversed rows (`lo > hi`) are swapped silently, and a finite zero-width row is treated as a point. |
| `n_components` | `int` or `'auto'` | `1` | Number of mixture components. The default `1` fits a single unimodal density; `> 1` fits an EM mixture with exactly that many components. Opt-in `'auto'` uses KDE proposals, short shared-boundary screening fits, and policy-aware refinement of competitive candidates before choosing *K* by BIC. Ignored when `init_from` is given (inherited from seed). |
| `poly_degree` | `int`, `'auto'`, or `None` | `None` | Requested degree of the polynomial potential. Must be ≥ 2. On full-infinite support `(-np.inf, np.inf)`, an explicit odd degree is rejected because it is structurally inadmissible; use an even degree. `'auto'` considers only admissible even degrees on full-infinite support. Odd degrees remain available on one-sided or bounded support. `None` means `'auto'` without a seed, or inheritance from a seed. Automatic mixtures grow component degrees using omitted-information diagnostics of the jointly fitted model, including shared boundary parameters and mixture-weight nuisance directions. Different components can retain different degrees. |
| `support` | `(float, float)` or `None` | `None` | Domain of the density, e.g. `(-np.inf, np.inf)`, `(0, np.inf)`, `(0, 1)`. `None` means the unconstrained real line `(-np.inf, np.inf)`; structural boundaries such as zero must be supplied explicitly. Ignored when `init_from` is given (inherited from seed). |
| `log_boundary_lower` | `bool` or `None` | `None` | Allow the direct zero-offset lower-endpoint log term `-aL log(x - L)`, with `aL >= 0` (it may optimize to zero). Both its presence and its amplitude are shared across mixture components. `None` lets the data decide on a finite lower endpoint: the term is kept only when a calibrated one-sided likelihood-ratio comparison against the fit without it has `p < 0.05`, and never when a positive-weight observation sits exactly at the endpoint. `None` means no term on an infinite endpoint and inherits the seed setting with `init_from`. |
| `log_boundary_upper` | `bool` or `None` | `None` | The same for the upper endpoint, `-aU log(U - x)`. |
| `verbose` | `int` | `0` | Verbosity level for fitting and automatic selection diagnostics. |
| `suppress_warnings` | `bool` | `False` | Suppress selected numerical warnings. Warning filters are process-global on supported Python versions, so leave this `False` for concurrent fitting. |
| `init_from` | `Distribution` or `None` | `None` | Warm-start seed from a previously fitted `Distribution`. When given, `n_components`, `support`, and per-component structure are inherited from the seed. Per-component seeds are threaded automatically in seed-component order. |
| `sample_weights` | array_like or `None` | `None` | Non-negative **relative** observation weights, normalized to sum to 1. A row weighted 6 contributes twice what a row weighted 3 does; the absolute scale carries no meaning, so `[1,1,1]` and `[100,100,100]` are identical. Weights are *not* frequencies: they do not stand for repeated observations, and the model-selection sample size is the number of **rows** regardless of the weights. **Aggregated or binned data is therefore not supported through this argument** — 700 rows representing 1431 observations will be penalized as 700, biasing selection toward under-fitting. Expand such data to one row per observation instead. |
| `component_options` | `list[dict]` or `None` | `None` | Per-component keyword arguments for mixture fitting (length must equal effective `n_components`). Currently only `poly_degree` is allowed per-component; `support`, `sample_weights`, `init_from`, `log_boundary_*` are forbidden (they are global). Must be `None` when `n_components='auto'` and no seed is given. |
| `em_max_iter` | `int` or `None` | `None` (default: 50) | Maximum EM iterations. |
| `em_tol` | `float` or `None` | `None` (default: 1e-4) | EM relative log-likelihood convergence tolerance. |
| `rng` | `None`, `int`, `Generator`, or `RandomState` | `None` | Random-number source for stratified subsampling and the fallback GMM initializer. For fitting, `None` uses deterministic seed 0; pass an explicit source to choose another stream. Sampling has separate semantics: `sample(..., rng=None)` uses NumPy entropy. |
| `k_max` | `int` or `None` | `None` (default: 10) | Maximum number of components to consider when `n_components='auto'`. Ignored when `n_components` is an explicit integer. |
| `progressive` | `bool` | `True` | For mixture fits with an explicit integer `poly_degree > 2`, climb through admissible degrees up to the target and warm-start each rung. Has no effect on single-component fits or when `poly_degree='auto'`. |
| `auto_k_subsample` | `'auto'`, `int`, or `False` | `'auto'` | Size of the subsample used to *select* the component count when `n_components='auto'`. `'auto'` subsamples only above 20,000 samples; an integer sets the size directly; `False` scores every candidate on the full dataset. Subsampling can change which *K* wins; after selection, the winning *K* is refitted on all observations. Ignored when `n_components` is an explicit integer. |

---

### Evaluation Methods

All evaluation methods operate in the **currently active space** (base or exp). Both scalar and array inputs are accepted. Scalar inputs return `float`; array inputs return `numpy.ndarray`.

Density, CDF, and negative-log evaluators propagate NaN in `x`. The PPF is stricter: every probability must be finite and lie in `[0, 1]`; NaN and infinities raise `ValueError`. For evaluation coordinates, `±inf` is handled as a limit — `pdf(±inf)` is 0 and `cdf(-inf)`/`cdf(+inf)` are 0/1. `ppf(0)`/`ppf(1)` return the support endpoints.

#### `pdf(x)`

Evaluate the probability density function.

```python
c.pdf(0.0)  # → float
c.pdf([-1.0, 0.0, 1.0])  # → ndarray of shape (3,)
```

#### `cdf(x)`

Evaluate the cumulative distribution function.

```python
c.cdf(0.0)  # ≈ 0.5 for a symmetric density centered at 0
```

#### `ppf(p)`

Evaluate the percent-point function (quantile / inverse CDF).

```python
c.ppf(0.5)  # median
c.ppf(0.025)  # 2.5th percentile
```

Raises `ValueError` if any element of `p` is non-finite or outside `[0, 1]`.

#### `neg_log(x, n=0)`

Evaluate the negative-log density (the "potential") or its *n*-th derivative.

```python
c.neg_log(0.0)  # -log pdf(0)
c.neg_log(0.0, n=1)  # first derivative of -log pdf at 0
c.neg_log(0.0, n=2)  # second derivative (curvature)
```

Raises `ValueError` if `n < 0`.

#### `sample(size=None, rng=None)`

Draw random samples via inverse-CDF (quantile) sampling.

```python
c.sample()  # single float
c.sample(1000)  # ndarray of shape (1000,)
c.sample(100, rng=42)  # reproducible with an int seed
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` or `None` | Number of samples. `None` returns a single scalar. |
| `rng` | `None`, `int`, `Generator`, or `RandomState` | Random-number source. |

#### `moment(k, central=False, standardized=False)`

Compute the *k*-th moment.

```python
c.moment(1)  # raw first moment (mean)
c.moment(2)  # raw second moment E[X²]
c.moment(2, central=True)  # central second moment (variance)
c.moment(3, standardized=True)  # standardized third moment (skewness)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | *(required)* | Moment order (≥ 0). |
| `central` | `bool` | `False` | Compute `E[(X − mean)^k]`. |
| `standardized` | `bool` | `False` | Compute `E[(X − mean)^k] / std^k`. |

#### `cumulant(k)`

Compute the *k*-th cumulant using centered moments, which keeps the calculation stable under large translations. Orders start at 1.

```python
c.cumulant(1)  # mean
c.cumulant(2)  # variance
c.cumulant(3)  # third central moment
c.cumulant(4)  # fourth central moment - 3 * variance**2
```

For affine pushforwards `Y = a + bX`, cumulants obey `κ₁(Y) = a + b κ₁(X)` and `κₖ(Y) = b^k κₖ(X)` for `k >= 2`.

---

### Survival, Reliability, and Tail Analysis

Tail probabilities are evaluated natively in log space. `sf(x)` is computed as
`exp(logsf(x))`, not as `1 - cdf(x)`, and extreme upper quantiles use direct tail
inversion rather than forming `1 - p`. The tail paths are designed and regression-
tested for probabilities far below the resolution of an ordinary float64 CDF value.

```python
c.logpdf(x)
c.logcdf(x)
c.logsf(x)
c.sf(x)

c.isf(1e-100)  # upper quantile without forming 1 - 1e-100
c.logisf(np.log(1e-300))
c.logppf(np.log(1e-300))
```

For a single component, log-concavity implies an **increasing failure rate (IFR)**:
the hazard is non-decreasing by construction. `hazard(x, n=1)` exposes its first
derivative, while `fit_diagnostics["hazard_is_monotone"]` provides a numerical
check over the fitted body. A mixture of IFR components need not itself be IFR, so
the diagnostic may be false when `n_components > 1`; it is `None` if the numerical
check itself could not be completed.

```python
c.log_hazard(x)
c.hazard(x)
c.hazard(x, n=1)
c.cumulative_hazard(x)
c.mean_residual_life(x)
c.residual_entropy(x)
c.tail_rate("upper")  # base space only
```

The same extension adds equal-tailed intervals and highest-density regions,
expectations and information measures, held-out scoring, and truncation:

```python
c.interval(0.95)
c.hpd(0.95)  # always shape (m, 2)
c.expect(lambda x: x**2)
c.entropy()
c.cross_entropy(other)
c.kl_divergence(other)
c.loglik(holdout)
c.quantile_residuals(holdout, rng=0)
conditional = c.truncate(lower=0.0, upper=2.0)
```

In exp space, probability tails and equal-tailed intervals are exact transforms of
the base-space quantities. Hazard includes the Jacobian factor, entropy obeys
`H(exp X) = H(X) + E[X]`, while mean residual life and HPD regions are computed
independently in exp space because neither is preserved by a nonlinear change of
variables.

### Model Checking — `goodness_of_fit()`

Transforms observations through the fitted CDF and measures how far the result
is from uniform. Three statistics are available, differing in how they weight
the deviation: `"ks"` uses the supremum, `"cvm"` (default) integrates with
uniform weight, and `"ad"` weights by `1 / (u (1 - u))`, concentrating on the
tails — where a log-concave model is most likely to be wrong.

```python
c.goodness_of_fit(holdout)  # Cramer-von Mises, asymptotic
c.goodness_of_fit(holdout, statistic="ks")
```

The statistic is always meaningful. **The p-value is only meaningful under the
calibration you ask for**, and the result records which one applied:

| `calibration` | Valid for | Cost |
| --- | --- | --- |
| `"asymptotic"` (default) | Held-out observations only | One CDF evaluation |
| `"montecarlo"` | The sample the model was fitted to | `n_resamples` refits |

Applying the asymptotic null to training data is miscalibrated toward overly large
p-values — the fit has already moved toward those points, so the test accepts models
it should reject.
Conversely, the current refit-based Monte Carlo null is designed for that in-sample
case and should not be substituted for the asymptotic calibration on independent
held-out observations. Use the calibration that matches how the observations were
obtained. When you must check in sample, calibrate by simulation:

```python
result = c.goodness_of_fit(
    data, statistic="ad", calibration="montecarlo", n_resamples=500, rng=0
)
result["value"]  # Anderson-Darling statistic
result["pvalue"]  # calibrated against simulated refits
result["pvalue_valid_for"]  # 'the sample the model was fitted to'
result["n_failed"]  # simulated refits that declined
```

Anderson-Darling has no closed-form null distribution available here, so
`pvalue` is `None` under asymptotic calibration rather than an approximation
whose error would be invisible. Use Monte Carlo calibration for that statistic.

Interval rows are rejected: a censored observation has no single probability
integral transform.

### Uncertainty — `bootstrap_bands()`

Resamples observation rows with replacement, refits, and reports percentiles of
the resulting curves. The fitted state does not retain the training data, so it
must be supplied again.

```python
band = c.bootstrap_bands(
    data, np.linspace(-3, 3, 200), quantity="pdf", n_resamples=500, level=0.95, rng=0
)

band["x"], band["estimate"]  # the point estimate on the same grid
band["lower"], band["upper"]  # pointwise percentile bands
band["n_failed"]  # replicates whose refit declined
band["coverage_kind"]  # 'pointwise'
```

`quantity` may be `"pdf"`, `"cdf"`, or `"sf"`, evaluated in the currently active
space.

Two things to be careful about:

- **The bands are pointwise, not simultaneous.** Each abscissa is covered at
  `level` in isolation. A band covering the whole curve at once is wider, and
  these must not be reported as one. The returned `coverage_kind` field states
  this so the distinction survives being copied into a figure caption.
- **By default the refit holds the component count and support fixed**, so the
  bands describe uncertainty *conditional* on the selected structure. If `K` was
  chosen automatically and that choice is itself uncertain, pass
  `fit_kwargs={"n_components": "auto"}` to let each replicate reselect. The
  bands widen, and the run costs proportionally more.

Cost is one full refit per replicate, so this is minutes-to-hours work rather
than a property read. Replicates that fail — a resample can omit enough of a
mode's support that its component degenerates — are counted and reported; past
`BOOTSTRAP_MAX_FAILURE_FRACTION` of the total the run raises rather than
reporting a band built from a biased remnant.

### Summary Statistics (Properties)

All properties operate in the currently active space and return `float` (except `support` which returns an `ndarray`).

| Property | Description |
|----------|-------------|
| `.support` | `ndarray` of shape `(2,)` — `[lower, upper]` bounds. |
| `.mode` | Location of the density maximum. For mixtures, the tallest peak. |
| `.modes` | `tuple` of all local maxima (ascending order). Mixtures may have one or more numerically resolved local modes. |
| `.median` | Median (50th percentile). |
| `.mean` | Mean. |
| `.var` | Variance. |
| `.std` | Standard deviation. |
| `.skew` | Skewness. |
| `.kurt` | Kurtosis (Pearson, i.e. the standardized fourth moment; `3.0` for a Gaussian). Subtract 3 for the Fisher excess. |

---

### Coordinate Spaces — Base vs Exp

After fitting, the default active space is `"base"`.

#### Base space

The fitted density models the variable *x* directly. This is the natural choice for real-valued data.

#### Exp space

The density on *x* induces a density on *y = exp(x)* via the change-of-variables formula:

```
pdf_y(y) = pdf_x(log y) / y      for y > 0
```

This is useful when modeling inherently positive quantities. The exp-space support is always a subset of `(0, ∞)`.

#### Switching the active space

```python
c.set_default("exp")  # switch to exp space
c.mean  # now returns E[exp(X)]
c.pdf(1.0)  # evaluates the exp-space PDF

c.set_default("base")  # switch back
```

`set_default()` returns `self` for chaining.

#### Accessing a specific space directly

For single-component models, the `.base` and `.exp` properties give view objects with the same evaluation interface (`pdf`, `cdf`, `ppf`, `neg_log`, `sample`, `moment`, `cumulant`, and all summary statistics):

```python
c.base.pdf(0.0)  # always base-space, regardless of the active default
c.exp.pdf(1.0)  # always exp-space
c.base.mean  # base-space mean
c.exp.mean  # exp-space mean
```

For multi-component models, `.base` and `.exp` are not available at the top level. Use the top-level methods (which respect `set_default`) or access individual components:

```python
c2.components[0].base.pdf(0.0)
c2.components[1].exp.mean
```

---

### Affine Transforms — `Distribution.transform()`

Shift and scale a fitted density without re-fitting. The transform direction is
part of the operation, so `pullback` is a required keyword rather than being
inferred from mutable model state:

```python
c.transform(mu=5.0, sigma=2.0, pullback=False)  # pushforward: Y = 5 + 2 * X
c.transform(mu=5.0, sigma=2.0, pullback=True)  # pullback:    Y = (X - 5) / 2
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mu` | `float` or `None` | `None` (identity: 0) | Finite shift for this operation. |
| `sigma` | `float` or `None` | `None` (identity: 1) | Finite positive scale for this operation. |
| `pullback` | `bool` | *(required)* | Interpretation of `(mu, sigma)`. See below. |
| `relative_to` | `'current'` or `'original'` | `'current'` | Compose with the accumulated transform, or replace it relative to the original fitted density. |
| `inplace` | `bool` | `True` | Modify in-place or return a new `Distribution`. |

Returns `self` (if `inplace=True`) or a new `Distribution`.

**Pullback vs pushforward interpretation:**

- **Pushforward** (`pullback=False`): the transformed variable is `Y = mu + sigma * X`. Use this to shift and scale a fitted density.
- **Pullback** (`pullback=True`): the transformed variable is `Y = (X - mu) / sigma`. This is the inverse map, and is what you want when standardizing: a fit on raw data pulled back through its own `(mean, std)` has mean 0 and variance 1.

The two are exact inverses of each other — pulling back and then pushing forward with the same `(mu, sigma)` reproduces the original density to machine precision — and both are exact change-of-variables, not refits.

```python
# Pushforward: shift the density by +5 and scale by 2
c_shifted = c.transform(mu=5.0, sigma=2.0, pullback=False, inplace=False)

# Pullback of the current distribution
c_pulled = c.transform(mu=0.5, sigma=1.5, pullback=True, inplace=False)

# Replace the accumulated transform relative to the original fitted density
c.transform(mu=10.0, sigma=3.0, pullback=False, relative_to="original")

# Identity relative to the original fit resets the presentation transform
c.transform(pullback=False, relative_to="original")
```

One accumulated affine map belongs to the whole `Distribution` and applies uniformly to every component. It never changes fitted polynomial coefficients or shared boundary exponents. Component-specific fitting centers/scales remain immutable numerical conditioning coordinates, not independently adjustable transforms. Omitted arguments are identity operations relative to whichever reference `relative_to` selects.

---



### Serialization — Save & Load

The fitted state is stored as a NumPy structured scalar with the same model envelope for single-component fits and mixtures, enabling save/load without pickle. The envelope stores the accumulated public `(mu, sigma)` once, together with component payloads, weights, and model metadata. Loading validates the required fields and their values.

```python
# Save
state = c.data  # numpy structured scalar (deep copy)
np.save("model.npy", state)

# Load — from state object
c2 = Distribution(state)

# Load — from file
c3 = Distribution(np.load("model.npy", allow_pickle=False))

# Load — via method
c4 = Distribution()
c4.load(state)
```

**`Distribution.data`** (property): Returns a deep copy of the structured fitted state.

**`Distribution.load(state)`**: Validate and install a saved model envelope, including its component count. Returns `self`. An invalid state leaves the existing model unchanged.

For multi-component models, the structured state includes all components, their weights, and mixture metadata. `pickle.dumps(c)` / `pickle.loads(...)` are also supported; `Distribution.__reduce__` delegates to the same structured state so compiled evaluator objects are never pickled directly.

#### State compatibility

Saved states are validated structurally when loaded: required fields, shapes, values, and reconstructed spectral geometry must all be valid. Incompatible or malformed states fail during `Distribution.load(...)` with `ValueError` rather than being partially accepted.

The structured NumPy state returned by `Distribution.data` is the portable persistence representation within the current codebase. Private Python objects, internal cache layouts, and pickle byte streams should not be treated as a durable cross-version format. The state intentionally has no independent format-version tag.

---

### Copying

```python
c2 = c.copy()  # independent model
```

`Distribution` also supports `copy.copy()` and `copy.deepcopy()`. Copies have independent presentation transforms and mutable caches; immutable fitted payloads may be shared safely.

---

### Mixture-Specific API

These properties and behaviors are specific to multi-component models (`n_components > 1`).

| Property / Method | Description |
|--------------------|-------------|
| `.n_components` | Number of mixture components (`int`). |
| `.weights` | Mixture weights, shape `(K,)`, summing to 1 (`ndarray`). |
| `.components` | Read-only sequence of component query objects. |
| `.modes` | Tuple of all local PDF maxima (1 to *K* modes). |
| `.mode` | Location of the tallest peak among all modes. |

Individual components provide read-only density queries with `.base` and `.exp` views:

```python
comp = c2.components[0]
comp.base.pdf(0.0)
comp.exp.mean
```

Fitted coefficients, boundary amplitudes, and weights cannot be edited in place. Components cannot be independently refitted, loaded into, or transformed. Use `Distribution.fit(...)` or `load(...)` to replace a whole fitted model, and `Distribution.transform(...)` for its common affine map. This ownership rule applies to single-component fits too. Derived moments and caches are maintained by the library rather than set independently of the density.

Top-level evaluation methods (`pdf`, `cdf`, `ppf`, `sample`, `moment`, `cumulant`, and all summary statistics) automatically aggregate over components using the mixture weights.

---

### Fit and Selection Diagnostics

`Distribution.fit_diagnostics` distinguishes component geometry/feasibility from joint mixture convergence. Shared boundary amplitudes, standard errors, automatic-selection p-values, and weak-identification flags are reported once in the model-level `shared_boundary` record. Their errors use joint observed information, accounting for private component parameters and mixture weights, rather than independent component fits. A zero-face amplitude has no ordinary interior standard error; an unresolved positive direction has infinite uncertainty. `converged_approximately` remains distinct from `converged`, with its certified remaining-decrease bound reported. Treat `converged=False` as a reason to inspect the diagnostics before using the fit quantitatively.

```python
d = c.fit_diagnostics
if not d["converged"]:
    print(d)
```

`Distribution.selection_diagnostics` is `None` for the default single-component fit and whenever `n_components` is an explicit integer. After opt-in automatic selection it records the KDE proposal, whether screening was subsampled, the selected component count/BIC, and the candidate-score trace. This is the audit trail to retain when automatic *K* is scientifically material.

```python
s = c.selection_diagnostics
if s is not None:
    print(s["selected_n_components"], s["scores"])
```

The stable NumPy `.data` representation retains per-component optimizer termination records. Session-level EM and automatic-selection traces are intentionally not part of that structured model state, so reconstructing with `Distribution(c.data)` or loading a saved `.npy` state does not restore them. `copy.copy`, `copy.deepcopy`, `Distribution.copy()`, and Python pickle preserve those in-process diagnostic records. Persist them separately when the portable `.data` state is the analysis record.

---

### Diagnostics — `Distribution.spectral_diagnostics`

A property (not a method) returning a `dict` that describes how well the CDF/PPF representation actually serving `cdf()` and `ppf()` converged. This matters because spectral construction degrades rather than raises: at the depth limit, a degenerate panel width, or budget exhaustion it accepts the best panel it has, and an under-resolved representation is otherwise indistinguishable from a converged one.

| Key | Type | Meaning |
|-----|------|---------|
| `scope` | `str` | `"component"` for `K == 1`, `"mixture"` for `K > 1` — which representation these numbers describe. |
| `refinement_capped` | `bool` | Refinement stopped on the depth or panel budget rather than on tolerance. This reports the *mechanism* and fires readily, including on ordinary shapes where one sliver of a panel hits the depth limit against a boundary singularity. It is not a severity signal. |
| `uncertified_mass` | `float` | Share of probability under panels that failed tolerance. This is a failure-exposure measure, not an accuracy ranking; `0.0` means every panel met its internal certification criteria. |
| `mass_defect` | `float` | Departure of total integrated mass from 1. |
| `worst_panel_error` | `float` | Largest local error estimate among certified panels, converted to probability/CDF units. |
| `error_estimate` | `float` | Conservative overall CDF-health estimate combining `mass_defect`, `uncertified_mass`, and the largest certified local panel error. |
| `n_panels` | `int` | Panels in the final representation. |
| `n_masses_recertified` | `int` or `None` | Panel masses re-verified after construction; `None` when the step does not apply. |
| `ppf_fallback` | `bool` | Whether the stored spectral inverse fell back to CDF bisection rather than certified inverse panels. Extreme public tail queries may subsequently use direct potential-based tail inversion. |
| `log_concavity_margin` | `float` | Certified lower bound on base-space convexity of the potential, minimized over components. Non-negative means every component's polynomial is certified convex. The mixture density itself need not be log-concave even when each component is. |

```python
d = c.spectral_diagnostics
if d["error_estimate"] > 1e-6:
    print("CDF/quantiles may be under-resolved:", d)
```

Read it for `K == 1` and `K > 1` alike. A single component is served by its own packed CDF fields, while a mixture is served by a separately built mixture-level CDF, so inspecting component fields directly says nothing about what a mixture's `cdf()` and `ppf()` will do — `scope` tells you which one you are looking at.

#### Accuracy and limitations

**Tail resolution.** Quantiles below the reliable stored CDF range are solved directly from the fitted potential in log space rather than inferred from a CDF value that no longer carries significant tail digits. On unbounded supports this path is designed and regression-tested across the representable float64 probability range. Finite endpoints use the same numerical treatment; near a nonzero endpoint the eventual resolution limit is the spacing of representable public-coordinate floats.

Near a **finite** endpoint, boundary distances are evaluated directly in the public coordinate system rather than reconstructed from two internal coordinates. A lower endpoint at zero can therefore be approached through essentially the full positive float64 range; for nonzero endpoints, the only remaining floor is ordinary floating-point representability of the coordinate itself. Quantiles remain inside the support and monotone.

**Mixtures with narrow minority components.** Each component uses its own responsibility-weighted center and robust scale, and normalization brackets the local mode on that component's numerical scale. This keeps narrow or remote minority components well conditioned even when component widths or locations differ by many orders of magnitude. Mixture CDFs are constructed from the weighted component density, and component/mixture consistency is covered by the numerical regression suite.

Checking `error_estimate` after fitting a mixture is the simplest general guard. `uncertified_mass` remains useful for distinguishing a certification failure from a normalization or certified-panel interpolation error. Note that a `cdf`/`ppf` round-trip is *not* a useful check on its own: the PPF is constructed from the CDF, so the two stay self-consistent even when both are wrong.

---

## Detailed Usage Examples

### Point Samples with Infinite Support

```python
import numpy as np
from gibbus import Distribution

rng = np.random.default_rng(42)
data = rng.normal(loc=3, scale=2, size=1000)

c = Distribution().fit(data, support=(-np.inf, np.inf))
print(f"Mean: {c.mean:.3f}, Std: {c.std:.3f}")
print(f"Mode: {c.mode:.3f}, Median: {c.median:.3f}")
```

### Bounded Support

```python
data = rng.beta(2, 5, size=500)
c = Distribution().fit(data, support=(0, 1))
print(f"Support: {c.support}")
print(f"Mean: {c.mean:.4f}")
```

### Interval-Censored Data

Provide intervals as an `(R, 2)` array where each row is `[lower, upper]`:

```python
# Bin continuous data into intervals
true_data = rng.normal(size=300)
bin_edges = np.linspace(-4, 4, 20)
bin_idx = np.digitize(true_data, bin_edges) - 1
bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
intervals = np.column_stack([bin_edges[bin_idx], bin_edges[bin_idx + 1]])

c = Distribution().fit(intervals, support=(-np.inf, np.inf))
print(f"Mean: {c.mean:.3f}")
```

> **Point and interval fits use different numerical objectives.** Point-data
> fitting evaluates the exact log-likelihood (with exact sufficient-statistic
> compression where available). Interval fitting evaluates each observed
> probability mass directly in log space. Ordinary positive-width finite rows
> strictly inside the support use one compiled Gauss-Legendre traversal,
> splitting at the current density mode when the mode lies inside the row. The
> kernel constructs local nodes on demand, evaluates the potential, performs a
> stable streaming log-sum-exp normalization, and accumulates the conditional
> partial means/Fisher contribution and required power moments without
> materializing row-by-node potential, weight, partial, or covariance tensors.
> Positive-width rows touching a finite support boundary, and rows with an
> infinite endpoint, use a prepared compiled adaptive Gauss-Kronrod reducer with
> endpoint-aware or infinite-interval mappings. Heterogeneous adaptive rows are
> submitted as one compiled batch; the objective accumulates their weighted
> conditional first/second moments and auxiliary moments without a Python call
> per censoring bound. Both paths avoid CDF
> subtraction and far-tail probability underflow, while zero-width intervals
> retain the documented point-observation limit.
>
> Mixture E-steps likewise use component probability mass over each interval,
> not midpoint densities. Very coarse censoring can still leave the component
> decomposition statistically unidentified. The fitter compares the observed
> interval likelihood with the unrestricted Turnbull likelihood on the censoring
> endpoints and counts independent probability coordinates from the endpoint
> graph. If an overparameterized mixture reaches that nonparametric maximum, it
> is rejected explicitly as non-identifiable. This applies to disjoint,
> overlapping, and nested interval patterns. Automatic component selection uses
> the same interval likelihood and excludes such richer candidates rather than
> treating their midpoints as exact observations. Prefer point data where
> available and narrower intervals when within-bin component shape matters.

### Weighted Samples

```python
data = rng.normal(size=500)
weights = rng.exponential(size=500)  # non-negative weights

c = Distribution().fit(data, support=(-np.inf, np.inf), sample_weights=weights)
```

For point-data mixtures, the unconstrained mixture likelihood has the usual
singularity: one component can capture one observed location and drive its
scale to zero. `gibbus` does not return that machine-scale spike as a fitted
distribution. During point-data EM, posterior × observation weights are first
aggregated over duplicate coordinates, and every component must retain at
least two Kish-effective distinct locations. An explicit `n_components > 1`
fit raises `RuntimeError` if that boundary is reached; automatic component
selection can discard such a candidate. This is an estimability check, not a
minimum mixture-weight rule: a genuine low-mass component represented by many
observations is allowed. Single-component weighted fits remain valid whenever
their weighted sample has non-zero spread. Interval-censored mixtures are
exempt because each row contributes a probability mass bounded by one, so the
point-density collapse mechanism does not apply.

### Warm-Starting from a Previous Fit

Use `init_from` to initialize a new fit from a previously fitted `Distribution`. This can speed up convergence when fitting similar data. When `init_from` is given, the number of components, support, boundary-basis configuration, and compatible per-component structure are inherited from the seed (unless explicitly overridden):

```python
# Single-component warm start
c1 = Distribution().fit(rng.normal(size=500), support=(-np.inf, np.inf))

new_data = rng.normal(loc=0.1, size=500)
c2 = Distribution().fit(new_data, init_from=c1)

# Multi-component warm start
bimodal = np.concatenate([rng.normal(-2, 0.5, 300), rng.normal(2, 0.5, 300)])
c3 = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf))

new_bimodal = np.concatenate([rng.normal(-2, 0.5, 300), rng.normal(2, 0.5, 300)])
c4 = Distribution().fit(new_bimodal, init_from=c3)  # inherits K=2, support, etc.

# Override poly_degree while keeping everything else from seed
c5 = Distribution().fit(new_bimodal, init_from=c3, poly_degree=8)

# Per-component poly_degree overrides
c6 = Distribution().fit(
    new_bimodal,
    init_from=c3,
    component_options=[{"poly_degree": 6}, {"poly_degree": 4}],
)
```

### Exp-Space for Positive Data

When your data are inherently positive (e.g. prices, durations), fit the log of the data in base space and evaluate in exp space:

```python
positive_data = rng.lognormal(mean=1.0, sigma=0.5, size=500)

# Fit log(data) in base space
c = Distribution().fit(np.log(positive_data), support=(-np.inf, np.inf))

# Switch to exp space to get the density of the original positive variable
c.set_default("exp")
print(f"Exp-space mean: {c.mean:.3f}")
print(f"Exp-space mode: {c.mode:.3f}")

# Evaluate the density of the positive variable
c.pdf(2.0)  # density at y=2

# Or access exp-space directly without changing the default
c.set_default("base")
c.exp.pdf(2.0)  # same result
c.exp.mean  # E[exp(X)]
```

### Affine Transforms

```python
c = Distribution().fit(rng.normal(size=500), support=(-np.inf, np.inf))

# Pushforward: Y = 10 + 3*X
c_push = c.transform(mu=10, sigma=3, pullback=False, inplace=False)
print(f"Original mean: {c.mean:.3f}")
print(f"Transformed mean: {c_push.mean:.3f}")  # ≈ 10 + 3*c.mean

# In-place transform
c.transform(mu=5, sigma=2, pullback=False)
print(f"Now mean: {c.mean:.3f}")
```

### Multi-Component Mixture Fitting

The default fit has one component. Opt into automatic component-count selection
for multimodal data:

```python
# Generate trimodal data
data = np.concatenate(
    [
        rng.normal(-5, 0.8, 200),
        rng.normal(0, 1.0, 300),
        rng.normal(5, 0.6, 200),
    ]
)

# Auto-select K explicitly
c = Distribution().fit(
    data, n_components="auto", support=(-np.inf, np.inf), rng=42
)

print(f"Components: {c.n_components}")  # automatically chosen
print(f"Weights: {c.weights}")
print(f"Modes: {c.modes}")
print(f"Overall mean: {c.mean:.3f}")

# Evaluate
x = np.linspace(-10, 10, 500)
y = c.pdf(x)

# Sample from the mixture
samples = c.sample(1000, rng=rng)
```

You can also specify the number of components explicitly:

```python
c = Distribution().fit(data, n_components=3, support=(-np.inf, np.inf), rng=42)
```

**Limiting the auto search range:**

```python
# Search only K=1..5 instead of the default K=1..10
c = Distribution().fit(
    data, n_components="auto", support=(-np.inf, np.inf), rng=42, k_max=5
)
```

**Per-component options** (requires explicit `n_components`):

```python
c = Distribution().fit(
    data,
    n_components=2,
    support=(-np.inf, np.inf),
    component_options=[
        {"poly_degree": 8},  # component 0 uses degree 8
        {"poly_degree": 4},  # component 1 uses degree 4
    ],
)
```

### Accessing Individual Components

```python
c = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf))

for i, comp in enumerate(c.components):
    print(f"Component {i}: mode={comp.base.mode:.3f}, std={comp.base.std:.3f}")
    print(f"  weight = {c.weights[i]:.3f}")
```

### Saving and Loading Models

```python
# Fit and save
c = Distribution().fit(rng.normal(size=500), support=(-np.inf, np.inf))
np.save("my_model.npy", c.data)

# Load later
c_loaded = Distribution(np.load("my_model.npy", allow_pickle=False))

# Verify
assert np.isclose(c.pdf(0.0), c_loaded.pdf(0.0))

# Works for mixtures too
c_mix = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf))
np.save("mixture_model.npy", c_mix.data)
c_mix_loaded = Distribution(np.load("mixture_model.npy", allow_pickle=False))
```

### Computing Moments

```python
c = Distribution().fit(rng.normal(size=500), support=(-np.inf, np.inf))

# Raw moments
m1 = c.moment(1)  # E[X]
m2 = c.moment(2)  # E[X²]

# Central moments
mu2 = c.moment(2, central=True)  # E[(X − mean)²] = variance
mu3 = c.moment(3, central=True)  # E[(X − mean)³]

# Standardized moments
s3 = c.moment(3, standardized=True)  # skewness
s4 = c.moment(4, standardized=True)  # kurtosis
```

### Evaluating the Potential (Negative-Log Density)

The "potential" is `-log(pdf(x))`. Access it and its derivatives for diagnostics or analysis:

```python
c = Distribution().fit(rng.normal(size=500), support=(-np.inf, np.inf))

x = np.linspace(-3, 3, 100)

q0 = c.neg_log(x, n=0)  # -log pdf(x) — the potential itself
q1 = c.neg_log(x, n=1)  # first derivative (score-like)
q2 = c.neg_log(x, n=2)  # second derivative (curvature / precision)
```

At the mode, the first derivative is zero and the second derivative gives the local curvature.

---

## Input Formats

### Samples

| Shape | Interpretation |
|-------|---------------|
| `(R,)` | *R* point observations |
| `(R, 1)` | *R* point observations (explicit column) |
| `(R, 2)` | *R* interval-censored observations `[lower, upper]` per row |

For intervals, rows where `lower > upper` are silently swapped. Endpoints may be `-np.inf` or `np.inf` for one-sided censoring, but may not be NaN; finite zero-width rows are treated as points.

### Sample Weights

- 1-D array of length *R*.
- Must be non-negative, finite, and not all zero.
- Automatically normalized to sum to 1 internally.

---

## Support Specification

The `support` parameter defines the domain of the density.

| Support | Example | Description |
|---------|---------|-------------|
| Real line | `(-np.inf, np.inf)` | Unbounded on both sides |
| Half-line | `(0, np.inf)` | Non-negative reals |
| Bounded | `(0, 1)` | Finite interval |
| `None` (default) | — | Unconstrained real line `(-np.inf, np.inf)`. Structural boundaries are never inferred from observed extrema; specify them explicitly. |

`support` is a structural modeling choice, not a sample statistic. In particular, an all-positive finite sample does **not** imply a hard boundary at zero. If the variable is non-negative by construction, pass `support=(0, np.inf)` explicitly; otherwise the default fit remains unconstrained on the real line.

All samples must lie within the specified support. When `support=None`, the support is the full real line.

---

## Limitations

`gibbus` is a univariate density estimator for point or interval-censored observations; it is not a multivariate model. A finite-sample fit can approximate data generated by a heavy-tailed population, but log-concavity is still a modeling restriction and tail extrapolation should not be interpreted as a heavy-tail model. Automatic component-count selection uses an approximate KDE/BIC screening pipeline (deterministic by default, but still sensitive to subsampling); inspect `selection_diagnostics` or disable selection subsampling when that choice is scientifically material. Fitting is not safe to perform concurrently on the same `Distribution` instance, and opt-in `suppress_warnings=True` uses Python's process-global warning filters (context-local filters are the default only on free-threaded CPython builds).


## Error Handling

`gibbus` uses specific exception types:

| Exception | When |
|-----------|------|
| `ValueError` | Invalid user input: bad shapes, fewer than 2 samples, zero-spread data, out-of-support samples, unsupported parameter values, `ppf` called with non-finite `p` or `p` outside `[0, 1]`, etc. |
| `RuntimeError` | The model is not yet fitted (`"not fitted; call .fit(...) or .load(...)"`), or a degenerate numerical state was encountered during fitting. |

### Data that is not log-concave

For finite samples, a log-concave empirical fit may still exist even when the data were generated by a heavy-tailed law such as Cauchy or Pareto. On unbounded supports, extreme draws can nevertheless make the polynomial-potential fit numerically non-normalizable or unstable; in that case `fit()` raises `RuntimeError`. Treat a successful fit as an approximation to the observed finite sample, not evidence that the heavy-tailed population is log-concave. Options include supplying a scientifically justified bounded `support`, using a different model family, or fitting a mixture when multimodality rather than tail weight is the issue.

### Debugging silent fallbacks

Several numerical steps degrade gracefully rather than failing — a root-finder that cannot bracket, a warm start that produces a degenerate seed, a candidate *K* that will not fit. Three tools make those visible.

`gibbus.suppressed_failures()` returns the fallbacks taken in the current thread, most recent last, each with the context in which it happened; `gibbus.clear_suppressed_failures()` empties that thread-local record. Neither requires an environment variable, so they work on an ordinary run:

```python
import gibbus

gibbus.clear_suppressed_failures()
c = gibbus.Distribution().fit(data)
for f in gibbus.suppressed_failures():
    print(f["context"], "->", f["type"], f["message"])
```

`GIBBUS_DEBUG=1` makes *unexpected* fallbacks raise instead of degrading. Fallbacks that are expected on well-behaved data are marked routine and are still only recorded. To make routine fallbacks fatal as well, set `GIBBUS_DEBUG_STRICT=1` **together with** `GIBBUS_DEBUG=1`; strict mode does not enable debug mode by itself.

Generic contract errors such as `ValueError`, `TypeError`, `AttributeError`, `KeyError`, and `IndexError` are not classified as numerical failures and therefore propagate directly. This avoids version-dependent exception-message parsing for array-shape defects. `GIBBUS_DEBUG_STRICT=1` is the strongest Python-layer fallback check; allocation fallbacks inside `noexcept nogil` Cython evaluators cannot enter the Python ledger and are intentionally outside that guarantee.

---

## Concurrency

Evaluation of an already-fitted `Distribution` is read-only at the Python level and the compiled bulk kernels may release the GIL. `gibbus` itself does not create thread pools, process pools, or other worker orchestration during fitting or spectral finalization. Do not mutate, transform, load, or refit the same instance concurrently. Separate `Distribution` instances may be fitted concurrently by the caller, but keep `suppress_warnings=False`: Python warning filters are process-global on standard CPython builds, 3.14 included, so opt-in warning suppression can affect unrelated threads while its context is active. The suppressed-failure ledger itself is thread-local.


## Performance Notes

- **Cython kernels**: Performance-sensitive natural-state normalization/Fisher traversal, conic quadratic subproblems and point Newton loops, full-curvature certification, mixture posterior/information reductions, interval-censoring reductions, polynomial evaluation, spectral construction/query, and exact-tail integration use compiled Cython extensions.
- **Subsampled model selection**: With `n_components="auto"`, the two stages that choose the component count — KDE mode counting and the log-concave BIC sweep — run on a stratified subsample of large datasets. The default fit RNG is seed 0 when `rng=None`, so repeated fits are reproducible; pass a `Generator` or explicit integer to choose another stream. Subsampling can still change the selected *K* relative to a full-data sweep, but the final model for the winning *K* is refitted on every observation. Disable with `auto_k_subsample=False` when exact full-data component-count screening is preferred. Inspect `selection_diagnostics` to see the proposal, subsample size, candidate scores, and winner.
- **Compiled polynomial evaluation**: `pdf` and `neg_log` evaluate the fitted potential through compiled polynomial kernels rather than generic NumPy polynomial helpers.
- **Exact point sufficient statistics**: after the fixed fitting coordinate is chosen, the point-data objective depends on the sample only through weighted power moments and enabled fixed boundary-log expectations. These statistics are computed once; the inner optimizer never loops over raw point observations. This is an exact algebraic reduction, not a quadrature approximation.
- **Interval-row merging**: identical interval rows are merged with summed weights before fitting, which is exact and shrinks binned datasets substantially.
- **Natural-coordinate conic fitting**: the potential is affine in polynomial-curvature coefficients, linear tilt, and enabled boundary-log amplitudes. Log-concavity is enforced on the full curvature by an exact support-specific Markov–Lukacs semidefinite cone; Gram matrices are solver variables rather than fitted parameters. Point likelihoods have analytic gradients and Fisher Hessians. Interval fits use the exact observed Hessian when positive definite and its saddle-free reflection otherwise. A small conic Newton solver handles exact zero-amplitude/lower-degree faces, certifies quadratic subproblems by weak duality, and validates the returned full curvature with a rigorous Bernstein certificate plus exact fallback separator.
- **Spectral CDF/PPF**: Each fitted component stores compact adaptive Chebyshev panels for its CDF and inverse CDF. Finite, one-sided, and doubly unbounded supports are compactified to a finite coordinate; CDF panel densities are positivity-certified before analytic integration, and PPF panels are built from a certified non-negative inverse derivative in logit-probability coordinates. Construction uses cached direct Chebyshev-Lobatto transforms, compiled small-panel transform/Clenshaw/calculus kernels, compiled Bernstein certification (exact Chebyshev-to-Bernstein conversion with a rigorous rounding margin), and batched safeguarded Newton inversion of source CDF panels; endpoint-indistinguishable targets are canonicalized to the panel endpoint to preserve monotonicity at float64 resolution. Query-time evaluation uses compiled Cython kernels with degree classes 16, 24, and 32 and SIMD-oriented bulk paths for large arrays. Those transposed Clenshaw loops carry explicit no-alias and compiler-specific vectorization hints for GCC, Clang, and MSVC; the actual SIMD width and instruction selection remain compiler/target dependent, and portable wheels do not opt into machine-specific ISA flags. Outside the stored inverse-panel range, the raw spectral inverse uses compiled monotone bisection of the packed spectral CDF. For probabilities at or beyond the extreme-tail threshold, the public quantile API then re-solves from the fitted potential in log-probability space, avoiding reliance on CDF digits that are no longer representable. No interpolation grid or retained parametric tail-extension model is stored. Mixtures finalize components serially; the library does not create hidden worker pools.

- **Tail-region cost**: body queries run through the compiled spectral panels at tens of nanoseconds per point. `logsf` (and therefore `sf`, hazard, and other survival quantities) switches to exact upper-tail quadrature once the spectral survival probability falls below `SF_HANDOVER_P`; `logcdf` makes the symmetric switch in the lower tail, while `cdf` itself remains an absolute-probability spectral query. Public quantiles are re-solved from the fitted potential below `TAIL_ASYMPTOTIC_P` (and symmetrically above `1 - TAIL_ASYMPTOTIC_P`). Exact component tails use a reusable compiled QUADPACK callback. For mixtures, gibbus integrates each component tail separately and combines the component log masses with a stable compiled reduction, rather than evaluating the full mixture potential at every quadrature node. Exact-tail work is still per query and scales with the number of components, so expect milliseconds rather than panel-query nanoseconds once the handover is crossed.

- **Moment and cumulant caching**: During fitting, each live model state computes its mode-aware quadrature breakpoints and prepared low-level quadrature context once. Ordinary model power moments requested as a block use one compiled adaptive Gauss-Kronrod traversal across all requested orders, with the scalar prepared-QUADPACK path as the numerical fallback; generalized endpoint-log moments continue through the prepared scalar context. In the packed fitted state, raw moments up to order 32 are cached and reused across calls. Higher raw moments remain supported but are recomputed on demand rather than retained in the fixed-size cache. Computed cumulants are cached separately for the relevant active/view coordinate and invalidated when the fitted affine state changes.


- **Fitting cost**: Point optimization cost is essentially independent of sample count after sufficient-statistic construction; model-side normalization/Fisher work scales primarily with `poly_degree`. Natural state normalization and all required Fisher statistics share one compiled adaptive traversal, while the fixed-face Newton loop and conic quadratic subproblems run in Cython without the GIL. Interval fitting remains observation-dependent because each censoring interval contributes a conditional probability/expectation calculation. Ordinary finite rows strictly inside the support use one end-to-end compiled Gauss-Legendre traversal; rows touching a support boundary or containing infinite endpoints use the prepared compiled adaptive interval reducer in batches. Whole-support rows short-circuit to probability one and model covariance where applicable.
- **Mixture cost**: Each EM iteration re-fits all *K* natural components. Point-component M-steps compress responsibilities into weighted sufficient statistics; interval-component M-steps use the exact weighted censored likelihood. Accepted E-step posteriors are reused by the next M-step, and explored multi-start finalists continue directly into joint Newton polish instead of repeating their EM trajectory. With automatic degrees, explorations that locked different component degrees are ranked by BIC rather than raw likelihood.
- **Auto component selection** (`n_components="auto"`): First estimates the number of data modes via a KDE bandwidth sweep, then fits lightweight log-concave models for a focused range of *K* values centered on the KDE mode count (extended upward while the largest *K* tried still scores best) and picks the lowest BIC. On large inputs the KDE sweep and the log-concave sweep both run on a stratified subsample (see `auto_k_subsample`), leaving screening cheap relative to the single full log-concave EM run for the winning *K*. Use `k_max` to narrow the search range if desired.


---

## License

`gibbus` is distributed under the MIT License. See [`LICENSE`](https://github.com/DanJSal/gibbus/blob/main/LICENSE) for the full terms.


## Testing

Install the test dependencies and run the test suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest
```
