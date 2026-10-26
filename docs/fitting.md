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
| `poly_degree` | `int`, `'auto'`, or `None` | `None` | Bulk degree of the potential: the highest monomial power of the fitting coordinate. Must be ≥ 2. On full-infinite support `(-np.inf, np.inf)`, an explicit odd degree is rejected because it is structurally inadmissible; use an even degree. `'auto'` considers only admissible degrees and climbs them by an omitted-statistic score test. `None` means `'auto'` without a seed, or inheritance from a seed. Automatic mixtures grow component degrees using omitted-information diagnostics of the jointly fitted model, including shared boundary parameters and mixture-weight nuisance directions. Different components can retain different degrees. An explicit degree that does not converge raises `RuntimeError` (see [Failures](#failures)). |
| `support` | `(float, float)` or `None` | `None` | Domain of the density, e.g. `(-np.inf, np.inf)`, `(0, np.inf)`, `(0, 1)`. `None` means the unconstrained real line `(-np.inf, np.inf)`; structural boundaries such as zero must be supplied explicitly. Ignored when `init_from` is given (inherited from seed). |
| `log_boundary_lower` | `bool` or `None` | `None` | Allow the direct zero-offset lower-endpoint log term `-aL log(x - L)`, with `aL >= 0` (it may optimize to zero). Both its presence and its amplitude are shared across mixture components. `None` lets the data decide on a finite lower endpoint: the term is kept only when a calibrated one-sided likelihood-ratio comparison against the fit without it has `p < 0.05`, and never when a positive-weight observation sits exactly at the endpoint. `None` means no term on an infinite endpoint and inherits the seed setting with `init_from`. |
| `log_boundary_upper` | `bool` or `None` | `None` | The same for the upper endpoint, `-aU log(U - x)`. |
| `verbose` | `int` | `0` | Verbosity level for fitting and automatic selection diagnostics. |
| `suppress_warnings` | `bool` | `False` | Suppress selected numerical warnings. Warning filters are process-global on supported Python versions, so leave this `False` for concurrent fitting. |
| `init_from` | `Distribution` or `None` | `None` | Warm-start seed from a previously fitted `Distribution`. When given, `n_components` and `support` are inherited from the seed and cannot be overridden for that fit. Boundary flags inherit when left as `None` but may be explicitly overridden. The bulk degree may be inherited or overridden globally/per component as documented below. Per-component seeds are threaded automatically in seed-component order. |
| `sample_weight` | array_like or `None` | `None` | Non-negative **relative** observation weights, normalized to sum to 1. A row weighted 6 contributes twice what a row weighted 3 does; the absolute scale carries no meaning, so `[1,1,1]` and `[100,100,100]` are identical. Weights are *not* frequencies: they do not stand for repeated observations, and selection and inference use the **Kish effective sample size**, `n_eff = 1 / sum(w**2)` for weights normalized to sum to one. Equal weights give the number of rows; concentrated weights reduce the effective size. **Aggregated or binned data is therefore not supported through this argument** — 700 weighted rows representing 1431 observations are not treated as 1431 independent observations; they are scored using their Kish effective size. Expand such data to one row per observation instead. |
| `component_options` | `list[dict]` or `None` | `None` | Per-component keyword arguments for mixture fitting (length must equal effective `n_components`). Currently only `poly_degree` is allowed per-component; `support`, `sample_weight`, `init_from`, `log_boundary_*` are forbidden (they are global). Must be `None` when `n_components='auto'` and no seed is given. |
| `em_max_iter` | `int` or `None` | `None` (default: 50) | Maximum EM iterations. |
| `em_tol` | `float` or `None` | `None` (default: 1e-4) | EM relative log-likelihood convergence tolerance. |
| `rng` | `None`, `int`, `Generator`, or `RandomState` | `None` | Random-number source for stratified subsampling and the fallback GMM initializer. For fitting, `None` uses deterministic seed 0; pass an explicit source to choose another stream. Sampling has separate semantics: `sample(..., rng=None)` uses NumPy entropy. |
| `k_max` | `int` or `None` | `None` (default: 10) | Maximum number of components to consider when `n_components='auto'`. Ignored when `n_components` is an explicit integer. |
| `auto_k_subsample` | `'auto'`, `int`, or `False` | `'auto'` | Size of the subsample used to *select* the component count when `n_components='auto'`. `'auto'` subsamples only above 20,000 samples; an integer sets the size directly; `False` scores every candidate on the full dataset. Subsampling can change which *K* wins; after selection, the winning *K* is refitted on all observations. Ignored when `n_components` is an explicit integer. |
| `basis` | `BasisSpec` or `None` | `None` | Full potential-basis request: support, bulk degree, endpoint logs, tail classes and span expansion (see [Basis requests](#basis-requests)). It replaces `poly_degree`, `support` and `log_boundary_*`, which must then be left at `None`; `component_options` may still set per-component degrees. With `init_from` it replaces the seed's basis family (its support must equal the seed's). |
| `max_fit_seconds` | positive `float` or `None` | `None` | Cooperative wall-time budget shared by selection, degree growth, component fits and retries. Exhaustion raises `FitTimeoutError` and preserves any previously fitted state. A running native operation finishes before the next check; the elapsed time may exceed the budget. |

---

## Automatic component-count selection

The default `n_components=1` fits one log-concave component. Passing an explicit integer greater than one fits exactly that many components with EM. Passing `n_components="auto"` opts into staged component-count selection: a KDE bandwidth sweep proposes a mode count, lightweight shared-boundary fits screen a focused candidate range (stopping above the mode count once a count screens worse than its predecessor), competitive candidates are refined under the requested degree and boundary-selection policy, and BIC selects the winner.

This procedure is intentionally pragmatic rather than exhaustive or globally optimal. On large inputs, selection may use a stratified subsample controlled by `auto_k_subsample`; the selected component count is then refit on all observations. Use `selection_diagnostics` to retain the proposal, candidate scores, subsample information, and selected count when the decision is scientifically material.

For interval-censored observations, candidate models are scored using the actual interval probability masses rather than midpoint-density surrogates.

Mixture screens use a Newton convergence target of `1e-6` nats per observation,
without changing the curvature-feasibility threshold. Admitted candidates use
the final optimizer's requested tolerances. `selection_diagnostics` reports the
screen target as `screen_newton_tolerance`. Screening scores are heuristic and
are not bounds on the corresponding fully refined BIC.

## Runtime and model complexity

Use automatic degree unless a fixed degree is justified by the application.
Increasing degree can add expensive curvature constraints and retries, and can
worsen predictions outside the observed range even when optimization converges.
A convergence certificate describes the fitted objective, not predictive quality.
Check independent log-density scores and relevant tail behavior when comparing
complexity; a close central CDF alone can conceal poor tail predictions.

The default `n_components=1` is appropriate for a single log-concave component.
Supply a known component count when that information is available; otherwise use
`n_components="auto"` when mixture selection is needed. Automatic count selection
has additional screening and refinement cost.

A positive `max_fit_seconds` shares one budget across the entire fitting search:

```python
from gibbus import Distribution, FitTimeoutError

model = Distribution()
try:
    model.fit(data, n_components="auto", max_fit_seconds=3.0)
except FitTimeoutError as exc:
    print(f"Fitting stopped after {exc.elapsed_seconds:.3f} seconds")
```

`FitTimeoutError` is a `TimeoutError`, separate from numerical-fit `RuntimeError`
failures. It escapes candidate-failure recovery: an exhausted request does not
start a fresh retry with a reset clock. An existing fitted `model` retains its
previous state, and an unfitted model remains unfitted. Subsequent calls have
independent budgets. With the default `None`, no time limit is imposed.

The budget is checked between solver runs and fitting stages. Native numerical
operations are not interrupted, so it is not a hard process deadline and does
not guarantee convergence within the requested number of seconds. No partial
candidate is installed or reported as a completed fit on timeout.

## Basis requests

Without `basis`, a fit uses the minimal basis of its arguments: bulk monomials of the fitting coordinate up to `poly_degree` and the selected endpoint logarithms. `basis=BasisSpec(...)` requests the general potential family:

```python
from gibbus import BasisSpec, Distribution, Exponential

spec = BasisSpec(
    support=(0.0, np.inf),
    bulk_degree="auto",
    log_lower="auto",
    upper_tail=Exponential(rate=1.0),
)
c = Distribution().fit(data, basis=spec)
```

| Field | Meaning |
|---|---|
| `support` | `(L, U)`, `L < U`; either end may be infinite. |
| `bulk_degree` | Highest bulk monomial degree, or `"auto"` (degree ladder). |
| `log_lower`, `log_upper` | `-log(x - L)` / `-log(U - x)` terms; `True` requires the finite endpoint; `"auto"` decides by the one-sided likelihood-ratio test. |
| `lower_tail`, `upper_tail` | Leading tail class at an infinite end: `Polynomial(power)` (integer power at least 1), `SoftPower(power)` (power greater than 1; an integer power becomes `Polynomial` with a `GibbusBasisWarning`) or `Exponential(rate)` (rate greater than 0); `None` keeps the bulk behavior. |
| `independent_equal_rate_coefficients` | Independent leading amplitudes for equal-rate tails instead of one shared amplitude. |
| `expansion` | `"auto"` (default) starts from the minimal span and adds slots of the full span only when a score test resolves them; `"minimal"` and `"full"` fix the span. |

`BasisSpec()` defaults to `bulk_degree=2`, no endpoint logs, no explicit tail
families, and `expansion="auto"`. In contrast, fitting without `basis` defaults
to automatic degree and finite-endpoint log selection with a minimal expansion.
Per-component `poly_degree` values override the corresponding basis degree;
unspecified components retain `BasisSpec.bulk_degree`. General-basis properness
is checked for each requested component.

Omitting a selected family's parameter (`Polynomial()`, `SoftPower()`,
`Exponential()`) requests empirical within-family estimation. See
[Automatic tail parameters](automatic-tail-parameters.md) for sample-size,
weighting, censoring and persistence limits. Explicit parameters stay fixed.

The basis is defined in a fitting coordinate `z = (x - c) / h` fixed by the support and the data (per component in mixtures): on the full line the 10th and 90th percentiles sit at `z = -1` and `z = 1`; on a half-line the origin is the finite endpoint and the 90th (or 10th) percentile sits at distance 1; on a bounded support the interval maps to `[-1, 1]`. Soft-power and exponential tail terms are confined to their own end by fixed smooth gates centered at `z = -1` and `z = 1`. Every fit is constrained to a convex potential (`q'' >= 0`, verified numerically within a threshold `tau`) and to proper outward growth at infinite ends.

On a half-line or bounded support, data whose 10–90% spread is below 0.5% of the median's distance from the nearest finite end are rejected with a `ValueError`: the coordinate is anchored at the end and cannot represent them. Fit on the full line instead, or choose a support whose finite end is nearer the data.

### Tail-family selection versus other automatic choices

`BasisSpec.lower_tail` and `BasisSpec.upper_tail` accept an explicitly chosen
`Polynomial`, `SoftPower`, or `Exponential` instance (or `None`, retaining
bulk-determined behavior). Gibbus **does not automatically compare these three
tail families** or choose a different tail class from the one requested.
Automatic degree selection, optional endpoint-log tests, and automatic basis
expansion are separate mechanisms. An explicitly constrained tail model may
extrapolate poorly on incompatible data without indicating a solver error;
Gibbus should not silently replace the user's specification. See [Limitations](limitations.md) for predictive-tail qualifications.

## Failures

Gibbus does not return a fit it could not compute. An explicit request whose fit does not converge raises `RuntimeError`: an explicit `poly_degree` (Gibbus first retries it from the lifted fit of the largest smaller admissible degree that converges), an explicit `n_components`, or `expansion="full"`. Automatic searches (`poly_degree="auto"`, automatic expansion, mixture degree growth, `n_components="auto"`) stop at the last size that converged and record why: for mixtures in `em_diagnostics["selection"][i]["failure"]`; the single-component degree ladder records it in its internal selection history, which is not yet public (see [Limitations](limitations.md#known-open-items)). A fit whose convexity is not verified within the threshold reports the status `unverified` (with a `RuntimeWarning`) instead of raising; its `fit_diagnostics` records the verdict and `shape_tau`.

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

c = Distribution().fit(data, support=(-np.inf, np.inf), sample_weight=weights)
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

Use `init_from` to initialize a new fit from a previously fitted `Distribution`. This can speed up convergence when fitting similar data. When `init_from` is given, the component count, support and basis family (tail classes and span policy, mapped through any presentation transform) are inherited from the seed; `basis=` replaces the family. Boundary flags inherit when left as `None` but can be explicitly overridden. The bulk degree can also be overridden globally or per component. Other compatible per-component seed structure is carried forward automatically:

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
