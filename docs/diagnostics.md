# Diagnostics and model checking

Gibbus exposes fit, model-selection, spectral, and fallback diagnostics so numerical degradation is inspectable rather than silent. This document collects those tools and the package's model-checking utilities.

## Module-Level Helpers

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
## Model Checking — `goodness_of_fit()`

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
## Uncertainty — `bootstrap_bands()`

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
## Fit and Selection Diagnostics

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

The NumPy `.data` representation retains per-component optimizer termination records. Session-level EM and automatic-selection traces are intentionally not part of that structured model state, so reconstructing with `Distribution(c.data)` or loading a saved `.npy` state does not restore them. `copy.copy`, `copy.deepcopy`, `Distribution.copy()`, and Python pickle preserve those in-process diagnostic records. Persist them separately when the `.data` state is the analysis record.

---
## Diagnostics — `Distribution.spectral_diagnostics`

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

### Accuracy and limitations

`spectral_diagnostics` reports whether the representation serving `cdf()` and `ppf()` met its internal certification targets. For tail-resolution limits, finite-endpoint floating-point behavior, and narrow-mixture qualifications, see [Limitations and concurrency](limitations.md#numerical-accuracy-qualifications).

---
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
