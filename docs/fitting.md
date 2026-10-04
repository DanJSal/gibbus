# Fitting

`Distribution.fit()` is the main model-construction interface. This guide collects the complete fit options, accepted observation formats, support rules, weighting semantics, mixture selection, and warm-start behavior. For the statistical interpretation of these choices, see [Modeling concepts](modeling-concepts.md).


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
| `init_from` | `Distribution` or `None` | `None` | Warm-start seed from a previously fitted `Distribution`. When given, `n_components` and `support` are inherited from the seed and cannot be overridden for that fit. Boundary flags inherit when left as `None` but may be explicitly overridden. Polynomial degree may be inherited or overridden globally/per component as documented below. Per-component seeds are threaded automatically in seed-component order. |
| `sample_weights` | array_like or `None` | `None` | Non-negative **relative** observation weights, normalized to sum to 1. A row weighted 6 contributes twice what a row weighted 3 does; the absolute scale carries no meaning, so `[1,1,1]` and `[100,100,100]` are identical. Weights are *not* frequencies: they do not stand for repeated observations, and the model-selection sample size is the number of **rows** regardless of the weights. **Aggregated or binned data is therefore not supported through this argument** — 700 rows representing 1431 observations will be penalized as 700, biasing selection toward under-fitting. Expand such data to one row per observation instead. |
| `component_options` | `list[dict]` or `None` | `None` | Per-component keyword arguments for mixture fitting (length must equal effective `n_components`). Currently only `poly_degree` is allowed per-component; `support`, `sample_weights`, `init_from`, `log_boundary_*` are forbidden (they are global). Must be `None` when `n_components='auto'` and no seed is given. |
| `em_max_iter` | `int` or `None` | `None` (default: 50) | Maximum EM iterations. |
| `em_tol` | `float` or `None` | `None` (default: 1e-4) | EM relative log-likelihood convergence tolerance. |
| `rng` | `None`, `int`, `Generator`, or `RandomState` | `None` | Random-number source for stratified subsampling and the fallback GMM initializer. For fitting, `None` uses deterministic seed 0; pass an explicit source to choose another stream. Sampling has separate semantics: `sample(..., rng=None)` uses NumPy entropy. |
| `k_max` | `int` or `None` | `None` (default: 10) | Maximum number of components to consider when `n_components='auto'`. Ignored when `n_components` is an explicit integer. |
| `progressive` | `bool` | `True` | For mixture fits with an explicit integer `poly_degree > 2`, climb through admissible degrees up to the target and warm-start each rung. Has no effect on single-component fits or when `poly_degree='auto'`. |
| `auto_k_subsample` | `'auto'`, `int`, or `False` | `'auto'` | Size of the subsample used to *select* the component count when `n_components='auto'`. `'auto'` subsamples only above 20,000 samples; an integer sets the size directly; `False` scores every candidate on the full dataset. Subsampling can change which *K* wins; after selection, the winning *K* is refitted on all observations. Ignored when `n_components` is an explicit integer. |

---

## Automatic component-count selection

The default `n_components=1` fits one log-concave component. Passing an explicit integer greater than one fits exactly that many components with EM. Passing `n_components="auto"` opts into staged component-count selection: a KDE bandwidth sweep proposes a mode count, lightweight shared-boundary fits screen a focused candidate range, competitive candidates are refined under the requested degree and boundary-selection policy, and BIC selects the winner.

This procedure is intentionally pragmatic rather than exhaustive or globally certified. On large inputs, selection may use a stratified subsample controlled by `auto_k_subsample`; the selected component count is then refit on all observations. Use `selection_diagnostics` to retain the proposal, candidate scores, subsample information, and selected count when the decision is scientifically material.

For interval-censored observations, candidate models are scored using the actual interval probability masses rather than midpoint-density surrogates.

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

## Boundary terms

Finite endpoints may use optional logarithmic boundary terms controlled by `log_boundary_lower` and `log_boundary_upper`. `True` enables the corresponding basis, `False` disables it, and `None` lets the fit select the term when no warm-start seed is supplied. On an infinite endpoint, no boundary term is present.

When `init_from` is supplied, `None` inherits the seed setting while an explicit boolean overrides it. In mixtures, each enabled physical boundary has one shared fitted amplitude across components.

## Weighted Samples

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

## Warm-Starting from a Previous Fit

Use `init_from` to initialize a new fit from a previously fitted `Distribution`. This can speed up convergence when fitting similar data. When `init_from` is given, the component count and support are inherited from the seed. Boundary flags inherit when left as `None` but can be explicitly overridden. Polynomial degree can also be overridden globally or per component. Other compatible per-component seed structure is carried forward automatically:

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

## Fitting versus querying

Fitting mutates the target `Distribution` and is not safe to perform concurrently on the same instance. After fitting, query methods are read-mostly but some caches are constructed lazily; see [Limitations and concurrency](limitations.md#concurrency) before sharing a newly fitted object across threads.
