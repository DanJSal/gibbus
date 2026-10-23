# Fitting controller: screening accuracy and time budgets

October 22, 2026. Unreleased working source based on GIBBUS_Borrowable_Aspects_Tested.zip.

## Behavior

Automatic component-count selection uses a default certified Newton target of 1e-6 nats per observation for provisional mixture screens. Single-component screens and all admitted refinements retain the requested final optimizer settings. Screening degrees remain 4 and 6 for the automatic policy. The KDE proposal, initializer families, BIC scoring, minimum component mass, admission rules, winner-neighbor exploration and degree-growth decisions retain their existing definitions. No new BIC-margin pruning rule was introduced.

The purpose is to avoid spending final-fit precision on component M-steps inside five-step screening EM explorations. The screening target is reported as selection_diagnostics['screen_newton_tolerance']. Curvature feasibility, properness and final-fit convergence thresholds are not relaxed. Coarser screens are still a heuristic: their scores are not bounds on the corresponding fully refined model.

Distribution.fit(max_fit_seconds=...) provides an optional shared wall-time budget across preparation, selection, component solves, degree growth, faces and retries. The default None imposes no time limit. Exhaustion raises gibbus.FitTimeoutError, a TimeoutError that is not swallowed as a recoverable numerical candidate failure. Its max_fit_seconds, elapsed_seconds and stage attributes identify the exhausted request. Existing fitted state is installed only after successful completion, so a timeout preserves the previous model. An unfitted model remains unfitted.

The budget uses a context-local monotonic deadline. Nested work cannot extend or reset an outer deadline; shorter nested deadlines are restored correctly. Deadline state is reset when a request exits, including exceptional exits, and is independent across execution contexts. Unlimited fitting does not read the clock at checkpoints.

Budget checks occur between solver runs and fitting stages. A native numerical operation finishes before the next check, so the elapsed time may exceed the requested budget. This is not process-level preemption and is not a guarantee of a converged result within a fixed number of seconds. No partial fit is installed or labelled successful on exhaustion.

```python
from gibbus import Distribution, FitTimeoutError

model = Distribution()
try:
    model.fit(data, n_components="auto", max_fit_seconds=3.0)
except FitTimeoutError as exc:
    print(exc.elapsed_seconds, exc.stage)
```

Use automatic degree unless a fixed degree is justified. Supply a known component count when available, or request automatic count selection when that information is unknown. A high-degree optimizer certificate does not establish predictive quality, especially beyond the observed range. The fitting guide includes these distinctions and examples.

## Selection and prediction comparison

The main comparison repeats the five-family settings study on three training seeds (91, 2045, 20261022), using 600 matched training observations and 20,000 matched held-out observations per family/seed. Both trees run the public API with n_components='auto', rng=0, and default selection policies. Additional single-seed controls cover half-line gamma, bounded beta, a bounded two-beta mixture, uniform data, and a four-mode normal mixture. The four-mode control supplies k_max=4 in both trees; the other cases use the library's default ceiling.

For all 19 cases that completed in both arms:

- Selected component counts and per-component degrees match.
- Training and held-out mean log density match exactly in these recorded runs.
- Log densities on each shared 301-point grid match exactly.

This is empirical agreement on these cases, not a proof of unchanged selection on every dataset. Component selection remains heuristic.

Median public fit seconds across the three main training datasets (one timing per dataset/settings pair, not repeated measurements of identical data):

| Family | Original | Updated |
| --- | ---: | ---: |
| normal | 0.993 | 0.962 |
| gumbel | 3.140 | 3.208 |
| separated | 5.241 | 4.516 |
| overlap | 1.157 | 1.014 |
| unbalanced | 5.860 | 5.291 |

The gains are modest and mixed. For example, the separated mixture is faster, while Gumbel has little change. The single-seed gamma control measured 4.613 to 4.151 seconds; beta measured 6.754 to 7.206 seconds; the bounded two-beta mixture measured 17.075 to 18.566 seconds; the four-mode control measured 2.468 to 2.324 seconds. These differences include solver-path and timing variation, so small changes should not be treated as robust speedups or regressions without repeated measurements.

The additional uniform-data control is not a completed quality comparison: the original source returned one component in an uncapped preliminary run of about 102 seconds; the updated run was interrupted by the benchmark's 30-second timer. It is excluded from timing and quality agreement aggregates. This broader case shows that the earlier five-family study did not establish a general few-second ceiling for automatic selection. The benchmark's POSIX signal timer is delivered at Python checkpoints and can itself be delayed by native work; use an external process watchdog when a hard benchmark deadline is required.

With max_fit_seconds=2 on the updated uniform-data request, the public budget raised FitTimeoutError at about 2.000 seconds in a Newton run. A separate degree-8 overlapping-mixture request with max_fit_seconds=1 stopped at about 1.001 seconds and retained the previous model's density exactly. These are individual smoke checks, not worst-case latency guarantees. Their raw records are included.

A coarser 1e-4 screen target was tried on nine single-seed controls. It retained selected models on those controls and helped some timings, but did not consistently improve the bounded cases. The more conservative 1e-6 default is retained. Neither trial justifies promising a universal runtime ceiling.

## Validation

- Focused fitting/API suite: 127 passed, 1 xfailed, 1 failed. The failure is test_stacked_m_step_without_shared_columns_is_independent_fits, already reproduced on the original source during the earlier investigation: iteration_limit instead of the expected converged status.
- Final budget/screen-policy/default suite: see budget-final-tests.log. It covers shared deadlines, shorter nested deadlines, context isolation, no clock reads for unlimited fits, input validation, pickle round trips, existing-state preservation, later unlimited calls, prevention of expired component solves, and propagation out of automatic screening. It also verifies that screens use the separate target while every refinement receives the original optimizer-policy object.
- The broader auto-K/selection-battery run was stopped after its initial 50 passing tests and three failures in explicit degree-6 lognormal requests with counts 2, 3 and 4. Those requests bypass the changed screening policy. The run did not finish, and these three failures were not separately replayed on the original source in this update. They are unresolved validation failures, not a claim that the broader suite passes. The partial output is retained.
- The complete repository suite, debug extension build and cross-platform matrix were not run. This source is a working update, not a release certification.

All original-source comparison runs use the archive's compiled conic kernel, not the earlier experimental reduced-system patch. This update does not incorporate that experimental kernel change. Source hashes and a patch against the uploaded archive are included.

## Reproduce and review

The archive contains the full updated source tree. Build/install it using the existing project instructions, then run:

```sh
python -m pytest tests/test_fit_budget.py tests/test_auto_k.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_component_selection.py --source . --output selection-runtime.json
```

Run original and updated trees sequentially with the same compiled optimization flags and BLAS limits. Benchmark imports, sample generation and scoring are outside the recorded fit timer. Existing output rows are resumed; use fresh output paths for new timing runs. Raw comparison data and test output live in docs/runtime-evidence/.

Runtime environment for this validation: CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0, GCC 13 at -O3, single-threaded BLAS/OpenMP. Absolute times depend on the environment. No additional runtime dependency was introduced by the budget implementation.
