# Changelog

This file records notable user-visible changes to Gibbus.

## 0.2.0.dev0 - Unreleased development checkpoint

- Use a common runtime coordinate interface for point, interval and coupled
  objectives, with support-appropriate polynomial recurrences and exact span
  preservation. Consolidate Newton traversal into one compiled controller with
  scalar-projection and interior-point subproblem kernels. Apply rejected-endpoint
  searches uniformly and share contact refinement with the native shape gate.

- Fit supported high-degree half-line point models with directly evaluated generalized Laguerre coordinates, retaining canonical persisted coefficients. Group polynomial curvature point bounds with tighter coefficient arithmetic and resolve ambiguous ungated polynomial/log contacts exactly. The 3,000-point degree-12 lognormal regression passes without an expected-failure marker; optimizer and shape tolerances are unchanged.

- Preserve face-scoped constraints and full shape acceptance during rejected-endpoint curvature-cut discovery; the common controller now replaces approximate gate cuts with fully resolved endpoint cuts for every eligible objective.
- Preserve exact nonnegative endpoint-log amplitudes during feasibility polish and refine well-conditioned interior point fits beyond the conic subproblem gap floor, retaining full shape and likelihood checks.
- Reject serialized coordinate kinds and canonical endpoints inconsistent with physical support and the affine map.

- Honor explicit per-component degrees with a general basis, including valid odd-degree faces, and accept automatic degrees in warm starts.
- Preserve selected model families/degrees during uncertainty refits; support basis overrides and resample supplied row weights with observations. Monte Carlo checks refit the base variable in both coordinate views.
- Use scale-safe empirical tail screening, require converged optimization, retry failed coefficient starts, and check the shared fitting deadline throughout screening.
- Validate tail-kernel dimensions, respect central debug-build directives, and validate decoded model-record types before native compilation.
- Invalidate optional runtime caches when implementation, Python ABI, or numerical dependency fingerprints differ. Durable format 2 is unchanged.
- Emit a warning when a returned fit has an uncertified shape verdict.
- Accept rounding-level equal-likelihood polish steps while retaining optimizer, shape and EM verification gates. This repairs the lower-exponential-tail Laplace-mixture regression.
- Repair CI dependencies/test paths and enable CI on development-branch pushes; reconcile the installed-wheel extension inventory.
- Synchronize current API, weighting, serialization, build, uncertainty and development-status documentation; retain historical evidence separately. Apply consistent Python formatting and import checks.

- Reduce series-search setup through sparse exact maps and batched panel checks;
  retain immutable series plans on each basis model for reuse by objectives.

- Retain immutable series-search tensors across matching fits in a bounded
  process-local LRU; keep private mutable workspaces and expose build/reuse
  statistics, clearing, and a cold-versus-retained benchmark.

- Cache actual basis derivatives at scan points; contract derivative series on
  demand and retain multiple warm minima per panel. Prefer refined constraints
  while retaining useful lower samples. Add candidate speed/discovery benchmarks.


## Component-search accuracy and fitting budgets (2026-10-22; not released)

- Mixture component-count screens use a `1e-6` per-observation Newton bound.
  Candidate degrees, admission rules and strict final refinement are retained;
  selection diagnostics expose the screening target.
- `Distribution.fit(max_fit_seconds=...)` provides an optional cooperative
  wall-time budget shared by selection, component solves and retries.
  Exhaustion raises the public `FitTimeoutError` and preserves existing fitted
  state. Native operations finish before the next check. The default is no
  time limit.
- The fitting guide covers complexity, predictive tail checks and time budgets.

## Cached-series Newton candidate experiment (2026-10-22; not released)

- Adds an opt-in cached Taylor candidate search inside the nogil Newton trial
  gate. Cache construction is shared across coefficient updates; candidate
  refinement uses derivative series, while actual basis rows form cuts.
- Preserves the existing full curvature acceptance check and feasibility
  tolerance. Unsupported caches fall back to direct candidate search.
- Adds search/reuse diagnostics, regression tests, and full-fit comparisons.
  The option stays disabled by default because cache construction is not
  amortized by every fit; see `docs/series-search-experiment.md`.

## Local derivative series working addition (2026-10-22; not released)

- Adds independent local Taylor evaluators for q'', q''', and q'''' with
  compiled nogil construction, cached coefficient recombination, and batched
  Horner evaluation. Includes exact gate-map cancellations, scaled range
  fallback, structural convergence radii, and allocation-free Cython APIs.
- Adds independent high-precision tests and a reproducible benchmark. This
  internal numerical API does not alter Newton descent or minima searching.
- Building the new extension requires a C++ compiler. See
  `docs/local-series.md` for usage and the finite-order accuracy contract.

## G3-064 working candidate (2026-10-20; not released)

- The full-line degree-3 span with two `Polynomial(1)` tails now enforces
  the exact zero-coefficient face implied by opposing leading asymptotic
  curvature rows. The cold start and subsequent face transitions honor this
  necessary equality, instead of trying to enter an impossible strict cone.
  This also applies to full/automatic spans and interval/mixture fits where
  the same algebraic condition is present; explicitly unequal tanh rates are
  not silently changed.
- No curvature, properness, or convergence tolerance was relaxed. Degree-4
  linear-tail cut exchange and the exponential-tail overflow remain open;
  the full regression and cross-platform matrices have not been completed.

## G3-063 working candidate (2026-10-20; not released)

- Curvature tail cuts exposed by suppressing stronger asymptotic atoms on a
  coefficient face are now enforced on that face and its subfaces, not after
  releasing the suppression. Finite-point curvature cuts remain global.
  This prevents a face-derived tail cut from excluding a globally convex
  potential on a larger face. Regression coverage includes an independent
  quartic counterexample and nested-face cut visibility.
- No shape-certification or properness tolerance changed. The historical
  degree-3 failure has not been replayed; full-suite and platform tests remain.

## G3-062 working candidate (2026-10-20; not released)

- For two explicitly requested `Polynomial(1)` tails, the two automatic tanh
  steepnesses are equal, so the quadratic `(t_L + t_U)` gate retains the same
  leading curvature sign at both infinite ends. Automatic gate locations
  remain data-adaptive, and explicit steepness values retain precedence.
  Previously failing degree-2 Laplace fits now converge with certified shape.
  Degree-4 and separate cut/separator diagnostics remain under investigation.
- G3-061 retains a certified incumbent when an optional `SoftPower` face start
  fails, and classifies invalid internally generated rows as `RuntimeError`;
  overflow recovery for `Exponential` tails is still outstanding.

## Unreleased development snapshot — G3-060 (2026-10-20)

- Mixture multistarts run one finalist instead of two by default. Over 122
  paired benchmark fits no selected component count changed; total fit time
  fell 1.25–1.31x (largest fit 83 s to 58 s on the reference machine) and
  held-out density was identical on 112 of 121 completed fits.
- After the convexity gate rejects a Newton trial, the compiled line search
  shortens the step to the largest fraction that keeps the rejecting rows
  feasible (an exact ratio test on the violated point cuts; a repeated hint
  must shorten the step by at least 10%) instead of halving it. The gate,
  Armijo test and certification are unchanged. Two ill-conditioned fits
  that previously failed loudly now certify: an explicit degree-12 fit of a
  1200-point lognormal sample (Hessian condition 1e20) and beta(0.02, 2)
  with an explicit endpoint log (Fisher condition 2.5e37). Paired benchmark
  runtime fell 1.19x (96 cases) and 1.23x (mixture targets); no status or
  selected component count changed.

## Unreleased development snapshot — G3-059 (2026-10-19)

Stage 6 efficiency and diagnostics fixes from profiling the G3-057 benchmark.
No tolerance, shape-certification or selection-policy change.

- Mixture multistart finalists continue their exploration EM phase instead
  of recomputing it from the same responsibilities; results are
  bit-identical, 18–39% faster on profiled multi-component fits.
- A face entered during a fit (a collapsing tail leader or a zero log
  amplitude) now starts from the incumbent restored onto the face's rows,
  instead of the face's cold start 1.5–3 nats away, from which candidate
  solves could spend 100 s without converging. Across 122 paired benchmark
  fits, total fit time fell from 1268 s to 419 s (largest fit 293 s to 50 s);
  held-out log density was unchanged to within 1.4e-5 nats per observation,
  improving by 4.5e-3 on one case.
- `spectral_diagnostics["log_concavity_margin"]` no longer raises on bounded
  supports: a finite end is approached in `x` instead of by inverting
  quantiles near 1e-300 (10 of the 96 benchmark fits). Infinite sides keep
  their grid.

## Unreleased development snapshot — G3-057 (handoff G3-058, 2026-10-19)

This section describes source/handoff work **not established as a published
package release**. G3-058 adds documentation and preserves evidence; it does
not change fitted behavior relative to G3-057.

- Hardened fitting-coordinate normalization, spectral evaluation and post-fit
  moments for extremely narrow or large-offset data (G3-052/G3-053).
- Added structured failure diagnostics without relaxing optimization or
  shape acceptance thresholds (G3-054).
- Added a test-only log-concave distribution case generator and isolated
  fitting benchmark runner (G3-055/G3-056).
- Made the sample-size-shrunk skew-aware full-line tanh geometry the automatic
  default for eligible single-component, unweighted, uncensored point fits;
  explicit settings and other data/support categories retain their behavior
  (G3-057).
- Ran exploratory geometry studies, a runtime sample, and a 96-case G3-057
  benchmark. Current improvement objective: predictive accuracy and runtime
  of automatic model selection, **not** correction of incompatible explicit
  user tail requests. The complete G3-057 suite remains outstanding.

## 0.1.0 - 2026-10-05

### Added

- Maximum-likelihood fitting of smooth univariate log-concave distributions on full-line, half-line, and bounded supports.
- Point and interval-censored observations with sample weights.
- Optional finite mixtures and staged automatic component-count selection.
- Density, log-density, CDF, survival, quantile, moment, sampling, transformation, information, and diagnostic interfaces on fitted distributions.
- Tail-aware numerical evaluation and compiled Cython kernels for performance-sensitive fitting and evaluation paths.
- Versioned, non-pickle model serialization with an explicit durable compatibility contract.
- Cross-platform CI for CPython 3.10 through 3.14 on Linux, Windows, and macOS.
- Release-wheel builds for the supported CPython versions and target platforms.

### Notes

- The `0.x` series is pre-1.0. Minor releases may refine the documented public API as the project matures.
- Free-threaded CPython builds are not supported or tested in `0.1.0`.
