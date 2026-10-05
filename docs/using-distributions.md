# Using fitted distributions

This guide covers querying and transforming a fitted `Distribution`. All operations described here require a fitted model. For construction options, see [Fitting](fitting.md).

## Evaluation Methods

All evaluation methods operate in the **currently active space** (base or exp). Both scalar and array inputs are accepted. Scalar inputs return `float`; array inputs return `numpy.ndarray`.

Density, CDF, and negative-log evaluators propagate NaN in `x`. The PPF is stricter: every probability must be finite and lie in `[0, 1]`; NaN and infinities raise `ValueError`. For evaluation coordinates, `±inf` is handled as a limit — `pdf(±inf)` is 0 and `cdf(-inf)`/`cdf(+inf)` are 0/1. `ppf(0)`/`ppf(1)` return the support endpoints.

### `pdf(x)`

Evaluate the probability density function.

```python
c.pdf(0.0)  # → float
c.pdf([-1.0, 0.0, 1.0])  # → ndarray of shape (3,)
```

### `cdf(x)`

Evaluate the cumulative distribution function.

```python
c.cdf(0.0)  # ≈ 0.5 for a symmetric density centered at 0
```

### `ppf(p)`

Evaluate the percent-point function (quantile / inverse CDF).

```python
c.ppf(0.5)  # median
c.ppf(0.025)  # 2.5th percentile
```

Raises `ValueError` if any element of `p` is non-finite or outside `[0, 1]`.

### `neg_log(x, n=0)`

Evaluate the negative-log density (the "potential") or its *n*-th derivative.

```python
c.neg_log(0.0)  # -log pdf(0)
c.neg_log(0.0, n=1)  # first derivative of -log pdf at 0
c.neg_log(0.0, n=2)  # second derivative (curvature)
```

Raises `ValueError` if `n < 0`.

### `sample(size=None, rng=None)`

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

### `moment(k, central=False, standardized=False)`

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

### `cumulant(k)`

Compute the *k*-th cumulant using centered moments, which keeps the calculation stable under large translations. Orders start at 1.

```python
c.cumulant(1)  # mean
c.cumulant(2)  # variance
c.cumulant(3)  # third central moment
c.cumulant(4)  # fourth central moment - 3 * variance**2
```

For affine pushforwards `Y = a + bX`, cumulants obey `κ₁(Y) = a + b κ₁(X)` and `κₖ(Y) = b^k κₖ(X)` for `k >= 2`.

---
## Survival, Reliability, and Tail Analysis

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
## Summary Statistics (Properties)

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
## Coordinate Spaces — Base vs Exp

After fitting, the default active space is `"base"`.

### Base space

The fitted density models the variable *x* directly. This is the natural choice for real-valued data.

### Exp space

The density on *x* induces a density on *y = exp(x)* via the change-of-variables formula:

```
pdf_y(y) = pdf_x(log y) / y      for y > 0
```

This is useful when modeling inherently positive quantities. The exp-space random variable is strictly positive. For a full-real-line base support, the public support is reported as `[0, inf]`, using the closed limiting boundary at zero even though `exp(X)` is nonzero for every finite `X`.

### Switching the active space

```python
c.set_default("exp")  # switch to exp space
c.mean  # now returns E[exp(X)]
c.pdf(1.0)  # evaluates the exp-space PDF

c.set_default("base")  # switch back
```

`set_default()` returns `self` for chaining.

### Accessing a specific space directly

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
## Affine Transforms — `Distribution.transform()`

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
## Serialization — Save & Load

`Distribution.data` returns the versioned durable persistence representation for a fitted model. It is a NumPy structured scalar with no object dtype, so it can be saved and loaded without pickle:

```python
# Save
state = c.data
np.save("model.npy", state)

# Load — from state object
c2 = Distribution(state)

# Load — from file
c3 = Distribution(np.load("model.npy", allow_pickle=False))

# Load — via method
c4 = Distribution()
c4.load(state)
```

**`Distribution.data`** (property): Returns a deep copy of the current versioned serialization envelope.

**`Distribution.load(state)`**: Validate and install a saved model. The current instance changes only after the complete durable state has validated; an invalid state leaves the existing model unchanged.

Serialization format v1 separates three concerns:

- **`model`** is the frozen durable mathematical model and public presentation state.
- **`provenance`** is optional historical fit/optimizer metadata.
- **`cache`** is optional implementation-specific derived numerical state used only when it is compatible with the running Gibbus version and the durable model.

A missing, stale, or invalid cache is rebuilt rather than making the model unreadable. This allows future Gibbus versions to change spectral CDF/PPF implementations without changing the mathematical meaning of serialization format v1.

### State compatibility

The durable format has its own `format_version`, independent of the Gibbus package version. Released serialization formats are never silently reinterpreted: a newer Gibbus version either reads a supported format correctly, explicitly migrates it, or rejects it with a clear compatibility error.

Compatibility means preservation of the fitted mathematical distribution and documented public state within the numerical accuracy of the running implementation. It does not promise bit-for-bit identity of derived panels, cached moments, or floating-point query results across releases.

Python pickle is supported for convenience, but pickle byte streams are not the durable cross-version contract. Persist models with `.data` and `np.save` / `np.load(..., allow_pickle=False)` when cross-version readability matters.

See [Serialization and compatibility](serialization.md) for the exact v1 schema, mathematical interpretation, cache/provenance rules, and format-evolution policy.

---
## Copying

```python
c2 = c.copy()  # independent model
```

`Distribution` also supports `copy.copy()` and `copy.deepcopy()`. Copies have independent presentation transforms and mutable caches; immutable fitted payloads may be shared safely.

---
## Mixture-Specific API

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
