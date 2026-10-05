# API reference

The public top-level API is intentionally small:

```python
from gibbus import Distribution, suppressed_failures, clear_suppressed_failures
```

This page is an index of the supported public surface. Detailed parameter semantics and usage guidance are kept in the topical guides, while the source uses NumPy-style docstrings for callable-level reference material.

## `Distribution`

```python
Distribution(state=None)
```

Construct an unfitted distribution when `state` is `None`, or reconstruct a fitted distribution from a structured state produced by `.data`.

### Fitting and state

| Member | Purpose |
|---|---|
| `fit(samples, **options)` | Fit a single component or finite mixture. |
| `load(state)` | Validate and install a versioned durable serialized state atomically. |
| `data` | Deep copy of the current versioned durable serialization envelope. |
| `copy()` | Independent distribution object sharing only immutable fitted payload where safe. |
| `is_fitted` | Whether the instance contains a fitted model. |

See [Fitting](fitting.md), [Using fitted distributions](using-distributions.md#serialization--save--load), and [Serialization and compatibility](serialization.md).

### Density, probability, and quantiles

| Member | Purpose |
|---|---|
| `pdf(x)` / `logpdf(x)` | Density and log density. |
| `cdf(x)` / `logcdf(x)` | Cumulative probability and log cumulative probability. |
| `sf(x)` / `logsf(x)` | Survival probability and log survival probability. |
| `ppf(p)` / `isf(p)` | Lower- and upper-tail quantiles. |
| `logppf(log_p)` / `logisf(log_p)` | Quantiles from log probabilities. |
| `neg_log(x, n=0)` | Negative-log density or derivative. |
| `loglik(x, sample_weight=None)` | Log-likelihood/scoring for observations. |

### Sampling and moments

| Member | Purpose |
|---|---|
| `sample(size=None, rng=None)` | Draw random values by inverse-CDF sampling. |
| `moment(k, central=False, standardized=False)` | Raw, central, or standardized moments. |
| `cumulant(k)` | Cumulants. |
| `expect(func)` | Expectation of a callable. |
| `entropy()` | Differential entropy. |
| `cross_entropy(other)` | Cross-entropy against another fitted distribution. |
| `kl_divergence(other)` | Kullback-Leibler divergence. |

### Survival and regions

| Member | Purpose |
|---|---|
| `hazard(x, n=0)` / `log_hazard(x)` | Hazard and log hazard. |
| `cumulative_hazard(x)` | Cumulative hazard. |
| `mean_residual_life(x)` | Mean residual life. |
| `residual_entropy(x)` | Residual entropy. |
| `tail_rate(side)` | Base-space asymptotic tail-rate summary. |
| `interval(level)` | Equal-tailed probability interval. |
| `hpd(level)` | Highest-density region(s). |
| `truncate(lower=None, upper=None)` | Distribution truncated to a subinterval. |

### Model checking and uncertainty

| Member | Purpose |
|---|---|
| `goodness_of_fit(...)` | Goodness-of-fit statistics and optional resampling calibration. |
| `bootstrap_bands(...)` | Bootstrap pointwise/simultaneous uncertainty bands. |
| `quantile_residuals(x, rng=None)` | Quantile residuals for observations. |
| `fit_diagnostics` | Optimization and fitted-geometry diagnostics. |
| `selection_diagnostics` | Automatic component-selection audit trail, or `None`. |
| `spectral_diagnostics` | CDF/PPF representation health. |

See [Diagnostics and model checking](diagnostics.md).

### Coordinate views and transformations

| Member | Purpose |
|---|---|
| `base` | Explicit base-space query view. |
| `exp` | Explicit `Y = exp(X)` query view. |
| `default` | Name of the active query space. |
| `set_default(space)` | Switch the active query space. |
| `transform(...)` | Apply an affine pushforward/pullback presentation transform. |
| `mu`, `sigma` | Accumulated public affine transform parameters. |

### Summary properties

`support`, `mode`, `modes`, `median`, `mean`, `var`, `std`, `skew`, and `kurt` provide common scalar summaries in the active coordinate view.

For mixtures, `n_components`, `weights`, and `components` expose the fitted mixture structure. Component objects are read-only query views and cannot be independently refit or transformed.

## Module-level helpers

### `suppressed_failures()`

Return the thread-local record of numerical fallbacks that have been suppressed since the record was last cleared. Each record includes the context, exception type/message, and whether the fallback was an expected routine guard path.

### `clear_suppressed_failures()`

Clear the current thread's suppressed-failure record.

See [Diagnostics and model checking](diagnostics.md#module-level-helpers) for usage and debug-mode behavior.
