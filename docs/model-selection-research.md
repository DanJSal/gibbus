# Next phase: predictive accuracy and model-selection efficiency

**Current objective (agreed 2026-10-19):** Improve held-out predictive accuracy and computational efficiency of automatically selected G3 models, without overriding explicitly requested families or weakening numerical certification.

## Existing automation versus possible extensions

| Choice | State (G3-057 onward) | Scope of next phase |
| --- | --- | --- |
| Number of mixture components | `n_components="auto"` uses staged KDE proposal and BIC-based screening/selection | Profile proposal, candidate rejection, multistart initialization, EM, and polish; assess selection quality. |
| Polynomial degree | `poly_degree="auto"` / `bulk_degree="auto"` score-test ladder | Measure degree-vs-error/time; look for avoidable repeated work. |
| Endpoint logarithms | Automatic one-sided likelihood-ratio selection | Ensure appropriate comparisons and quantify overhead. |
| Expansion span | `expansion="auto"` | Quantify time and predictive value of expansion. |
| Tail family | **No general automatic selection between Polynomial/SoftPower/Exponential** | Investigate whether an opt-in or new automatic *candidate-selection capability* is justified; do not claim a selection defect where no chooser exists. |
| Tanh geometry | New default G3-057 adaptive full-line rule for eligible unweighted single-component point fits | Treat as fixed; half-line/mixture/weighted/censored preserve earlier behavior. No further geometry tuning planned. |

## Proposed bounded work plan

**Status (G3-060):** steps 1, 3 and 6 are done for the observed hotspots ([G3-059](benchmark-findings.md#g3-059-profiling-and-fixes), [G3-060](benchmark-findings.md#g3-060-mixture-cost-analysis)); screen-BIC pruning of automatic-count refinements, inexact M-steps and shorter exploration were evaluated and rejected. Step 2 was run for single-component degree and tail choices ([experiment](benchmark-findings.md#g3-060-basis-and-tail-selection-experiment)): the ladder is near-optimal on half-line and bounded supports, and on the full line its loss comes from bulk-derived tails. Step 4 (automatic tail-family selection) is deferred beyond G3; step 5's criteria were used for every adopted change.

1. **Instrument without policy changes.** Separate CPU time, candidate counts, convergence attempts and log likelihood at each automatic-selection stage (KDE, component-count BIC screen, initialization, weighted-component fits, EM rounds, Newton polish, degree/span selection, post-fit). Record comparison to the final selected configuration. Keep instrumentation isolated when possible.
2. **Run controlled comparisons.** Fix samples, seed, supports, degrees and tail *requests*; compare explicitly fixed component counts versus automatic count; compare auto degree vs fixed degree; use 3–5 seeds for a targeted subset of the 96 manifest cases, not an unconstrained compute sweep. Use timeouts calibrated by workload tier. Include enough easy controls to catch regressions.
3. **Find wasted work.** Diagnose the >90-second case, repeated near-identical starts, expensive EM polish, and overlarge candidate pools. Evaluate cheap pruning and warm starts only if they retain or improve final held-out quality and convergence.
4. **Evaluate optional automated tail-family selection as a separate feature proposal.** First determine a statistically defensible scoring/validation strategy and complexity cap. Compare candidate families only when the user has *not* explicitly requested one. Avoid post-selection optimism in held-out assessment. No implicit override of a `BasisSpec` tail class.
5. **Choose acceptance criteria before adopting changes.** Track median and upper-tail held-out logscore (when true oracle exists), reference-CDF discrepancy, convergence/shape status, p50/p90/p95 fit time, and timeout fraction; compare paired runs. For sums without PDF/CDF oracles, use independent holdout score and empirically calibrated checks with qualifications.
6. **Keep numerical diagnostics hardening separate.** Address unrepresentable 1e-300 post-fit PPF probes later. A diagnostic failure after successful fitting is not automatically a fitting failure; shape verification must remain intact.

## Guardrails

- Never silently substitute tail families or alter a user's explicitly requested polynomial/exponential/soft-power class merely because it extrapolates poorly.
- Never trade invalid densities or false convergence certificates for speed. Keep global shape constraints, properness, and certification tolerances.
- Do not construe holdout logscore gap as the exact KL divergence on finite samples.
- Avoid unbounded complexity from joint model-count, degree, tail-family and span grids; a staged search must be benchmarked against strong controls.
- Changes adopted under this plan (G3-059, G3-060) are listed in the [changelog](../CHANGELOG.md) with their paired evidence.

## Concrete acceptance deliverables

A profiling report with per-stage runtime breakdown; side-by-side case-level paired fit and held-out results; a documented, bounded selection policy proposal; specific speed/accuracy regressions triaged with case IDs and seeds; and independent numerical regression evidence before any new implementation becomes the default.
