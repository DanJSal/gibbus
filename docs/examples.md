# Examples

## Point Samples with Infinite Support

```python
import numpy as np
from gibbus import Distribution

rng = np.random.default_rng(42)
data = rng.normal(loc=3, scale=2, size=1000)

c = Distribution().fit(data, support=(-np.inf, np.inf))
print(f"Mean: {c.mean:.3f}, Std: {c.std:.3f}")
print(f"Mode: {c.mode:.3f}, Median: {c.median:.3f}")
```

## Bounded Support

```python
data = rng.beta(2, 5, size=500)
c = Distribution().fit(data, support=(0, 1))
print(f"Support: {c.support}")
print(f"Mean: {c.mean:.4f}")
```

## Interval-Censored Data

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

Interval rows contribute their fitted probability masses directly; they are not converted to midpoint observations. Very coarse censoring can make mixture decomposition weakly identified or unidentified. See [Fitting](fitting.md#input-formats) for accepted layouts and [Numerical methods and performance](numerical-methods.md#interval-censored-likelihoods) for the integration and identifiability details.

## Weighted Samples

```python
data = rng.normal(size=500)
weights = rng.exponential(size=500)  # non-negative weights

c = Distribution().fit(data, support=(-np.inf, np.inf), sample_weights=weights)
```

Weights are relative observation weights, not frequency counts. For point-data mixtures, Gibbus also guards against the usual collapsing-component likelihood singularity. See [Fitting](fitting.md#weighted-samples) for the weighting and estimability rules.

## Warm-Starting from a Previous Fit

Use `init_from` to initialize a new fit from a previously fitted `Distribution`. This can speed up convergence when fitting similar data. When `init_from` is given, the component count and support are inherited from the seed. Boundary flags inherit when left as `None` but may be explicitly overridden, and polynomial degree may be overridden globally or per component:

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

## Exp-Space for Positive Data

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

## Affine Transforms

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

## Multi-Component Mixture Fitting

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

## Accessing Individual Components

```python
c = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf))

for i, comp in enumerate(c.components):
    print(f"Component {i}: mode={comp.base.mode:.3f}, std={comp.base.std:.3f}")
    print(f"  weight = {c.weights[i]:.3f}")
```

## Saving and Loading Models

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

The structured state is the current persistence representation, but cross-version compatibility guarantees have not yet been formalized. Treat long-term archival compatibility as version-dependent until that policy is defined.

## Computing Moments

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

## Evaluating the Potential (Negative-Log Density)

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
