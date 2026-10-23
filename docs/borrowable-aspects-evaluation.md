# Evaluation of borrowable G1 fitting techniques

**Date:** October 22, 2026  
**Scope:** Controlled experiments comparing selected techniques from the older natural-coordinate implementation with the current general-basis fitter. This is development evidence, not a release certification.

## Environment and limits

Experiments ran on CPython 3.13 with NumPy 2.3.5 and SciPy 1.17.0. Building the current Cython extensions at the repository's production `-O3` setting exceeded the interactive command time limit, so the isolated experiment copy was built at `-O1` while retaining compiled objects across runs. All paired A/B timings used that same build. Relative timings below are exploratory and should not be read as production-build estimates. The source archive's build configuration was not changed.

The mixture comparisons use fixed generated samples, one two-component responsibility initializer, and fixed-degree minimal expansions. This isolates the candidate warm-start strategy; it does not measure global mixture-search success over every initializer and data family.

## Decisions

| Candidate | Evidence | Decision |
|---|---|---|
| Retry a failed fixed-degree mixture from the nearest smaller admissible degree, then lift back to the requested degree | Mixed across cases, but it rescued a stalled degree-8 fit and improved its likelihood. The retry is restricted to an unconverged fit, and the existing result is retained unless the retry ranks better. | **Integrated as a failure-only fallback.** Not used as a default extra search path, and not used for automatic degree growth or unequal component degrees. |
| Moment-informed polynomial cold start | Usually produced effectively the same normal-fit likelihood, but was slower or materially worse on other cases. An improved remote-cluster case still did not converge, while the current lower-degree retry produced a better fit. | **Not integrated.** No reliable family-wide improvement, and the proposed start assumes the exact `x` and `x^2` basis labels. |
| Old Newton/constraint mechanics | The transferable principles already exist in the current solver: basis-aware preconditioning/scaling, feasible starts, exact coefficient faces, curvature separation and certification. The old SOS/Gram formulation is tied to polynomial curvature and is not a drop-in solver for the current general basis. | **No solver/backend change.** Keep one general solver; extend it only for a demonstrated general-basis defect. |
| Older omitted-statistic degree test | The current span diagnostic generalizes the test to omitted basis spans and mixture nuisance parameters. Current degree fitting also retains lower-degree warm retries when a higher-degree solve fails and nested-likelihood checks. | **No duplicate degree-selection implementation.** Retain and test the current generalized method. |

## Mixture fallback experiment

NLL is negative log likelihood, so lower is better. Timings vary by run and are included only to show the cost pattern.

| Fixed-degree case | Direct target-degree fit | Lower-degree start | Interpretation |
|---|---|---|---|
| Two-normal mixture, degree 6 | Converged; log likelihood about -1.859981 (NLL 1.859981); 0.48 s | Full ladder converged to essentially the same likelihood in 0.34 s | Easy case; a ladder can be faster, but is not a reason to add work to every fit. |
| Gumbel mixture, degree 6 | `iteration_limit`; NLL 1.947293; about 6.5 s | One-rung result had NLL 1.946527 but remained unconverged; about 1.5 s for the retry alone | Better objective, but no convergence recovery; end-to-end fallback is slower than direct alone. |
| Unbalanced two-component mixture, degree 8 | `iteration_limit`; NLL 1.983340; about 5.5 s | One-rung retry reached `converged_approximately`; NLL 1.982512. The integrated end-to-end attempt took about 9.5–12.6 s across repetitions, versus about 5.5 s for a direct attempt. | Strongest positive result: better likelihood and a usable convergence status, with added runtime only after direct failure. |
| Overlapping mixture, degree 8 | `iteration_limit`; NLL 1.751138; about 3.4 s | One-rung candidate NLL 1.751219; about 2.5 s for the retry | Worse objective; the rank check retained the original result. This confirms why retries must never blindly replace the incumbent. |
| Three-mode data fitted with two components, degree 6 | `converged_approximately`; NLL 2.164843 | One-rung candidate was essentially tied and slightly slower | No rescue was needed because the direct result was accepted. |

A full degree ladder had mixed outcomes as well: on some stalled cases it improved the objective or convergence status; on easy mixtures it added cost, and it did not reliably make hard Gumbel cases converge. Accordingly, the implementation tries **one nearest-lower degree only after an equal, explicit fixed-degree mixture fails to converge**. The candidate is lifted back to the requested degree and compared under the existing fit-ranking rules; a worse candidate cannot displace the incumbent. A successful or approximately converged direct fit pays no retry cost.

Three focused tests cover the new behavior: failure triggers the retry path; the retry lifts coefficients back to the requested degree; and a converged direct fit skips the retry.

## Moment-informed cold-start experiment

For basis models with exact `x` and `x^2` columns, a second candidate was formed from weighted internal-coordinate moments, using the Gaussian-potential coefficients `theta_x = -mu / variance` and `theta_x2 = 1 / (2 variance)`. Across seven usable paired cases:

- Normal degree 4, 8 and 12 fits had near-identical likelihoods; speed differences were small and inconsistent.
- A bimodal degree-8 fit was about 38% slower from the moment start while reaching effectively the same likelihood.
- A Gumbel degree-8 fit had NLL worse by about 0.10 and remained at `iteration_limit`.
- A remote-cluster degree-6 fit improved NLL by about 0.043 and ran faster, but still reached `iteration_limit`.
- A gamma-like half-line fit was effectively unchanged. One SoftPower family did not have the exact quadratic column and could not use this candidate as written.

The existing `_fit_degree` lower-degree retry was then checked on the hard standalone cases: it returned a converged Gumbel degree-8 fit and better results for the bimodal and remote-cluster examples. The moment start is therefore not justified as a general default.

## Newton, constraints, and degree selection

The old solver's exact semidefinite Gram representation certifies a narrower polynomial-curvature cone. Reinstating it would create a second backend and would not cover the current gated, soft-power, exponential and other general-basis terms. The useful preconditioning and face-management principles are already present in the current Newton/cut-exchange implementation, including coordinate scaling for general basis columns and exact structural coefficient faces. No separate old-solver backend or polynomial-only fast path was added.

The current degree-selection code already carries forward the old score-test idea in generalized form. Its checks cover omitted spans and nuisance parameters instead of maintaining a separate natural-polynomial version. Focused validation of the unchanged selection/constraint/Newton path passed **95 tests** across `test_basis_selection.py`, `test_degree_policy.py`, `test_basis_constraints.py`, and `test_basis_newton.py` in this experiment environment.

## Validation of the integrated change

- Three new lower-degree-retry tests: **3 passed**.
- Mixture-state and mixture-numerics suites: **15 passed**.
- Full `test_basis_mixture.py`: **20 passed, 1 failed**. The one failure was reproducible before the patch: `test_stacked_m_step_without_shared_columns_is_independent_fits` receives `iteration_limit` instead of the expected `converged` status.
- `test_stage6_solver_triage.py` plus `test_stage6_release_blockers.py`: **18 passed, 1 xfailed, 1 failed**. The failure is the degree-3 two-linear-tail log-density equivalence assertion, whose maximum absolute difference is about `6.2e-6`, above its `1e-6` tolerance. The mixture-only patch does not edit the solver or basis-selection implementation, but this check remains an open validation failure in the tested build.

The complete test suite was not run to completion. These results support a narrow fallback integration, not a claim that the current development source is release-clean.
