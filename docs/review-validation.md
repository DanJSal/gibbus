# Implementation validation

## October 23, 2026: common fitting controller

- Complete source sweep: **6,104 passed, 1 regression, 2 skipped, 1 expected failure**, out of 6,108 cases in 746.335 seconds. The final guard fixes the bounded warm-start regression; its affected-case rerun passes **28 tests** in 89.569 seconds.
- Reconciled unique source cases: **6,105 passed, 2 skipped, 1 expected failure**, with no unresolved failure. This combines the full sweep and final affected-case rerun; it is not a fresh all-green full invocation after the final guard.
- Installed wheel outside the source checkout: **182 passed** in 98.579 seconds; all 14 native extensions imported. The broad wheel batch emitted four documented SoftPower conversion warnings and one invalid-arithmetic warning in optional contact inspection. A final finite-derivative guard removes the latter; the affected serialization check passes again with runtime warnings treated as errors in both source and installed wheel.
- Both degree-12 likelihood/acceptance regressions pass with existing numerical assertions. The degree-4 two-linear-tail Laplace test retains its strict expected-failure marker.
- Black and configured Ruff checks pass. Source hashes identify the delivered source verified by the sweep plus final targeted reruns; 81 Python/native package files match the installed wheel byte for byte.

See the [resolution report](review-evidence/coordinate-refactor-resolution.md),
[raw source XML](review-evidence/coordinate-refactor-source.xml) and
[installed XML](review-evidence/coordinate-refactor-wheel-tests.xml).
Historical checkpoint counts below describe their respective older sources.

## Earlier October 23, 2026: production Laguerre coordinates

The direct recurrence objective and tighter polynomial contact arithmetic are
implemented in production. Both degree-12 lognormal regressions pass; the
3,000-point test no longer has an expected-failure marker and also checks
likelihood accounting and a serialization round trip.

| Scope | Result |
| --- | --- |
| Broad source sweep, 6,086 cases | 6,074 passed, nine optional-metadata fixture failures, two skipped, one Laplace xfail; 781.944 seconds |
| Source rerun after the metadata-default correction | 284 passed, including all nine previously failing cases |
| Installed-wheel tests | 136 passed; all 14 extensions imported |
| Combined unique source-case outcomes | 6,083 passed, two skipped, one expected failure; no recorded failure remains unresolved |

The combined count reconciles the broad sweep with the affected-suite rerun;
it is not a fresh single all-green full-suite invocation after the final
one-line compatibility correction. Raw XML and the reconciliation are retained:
[source sweep](review-evidence/laguerre-full-suite.xml),
[source rerun](review-evidence/laguerre-source-postfix.xml),
[combined record](review-evidence/laguerre-validation-summary.json), and
[installed tests](review-evidence/laguerre-wheel-tests.log).

The fixed-family production solve converges with a stopping bound of
5.65e-13. Independent quadrature differs from its NLL by 5.4e-12, and exact
Sturm counting plus 80-digit refinement checks all nine positive stationary
points. The minimum curvature, about -9.7e-17, lies within the unchanged
-1.43e-12 floor. See the [implementation and observations](review-evidence/laguerre-production-resolution.md)
and [independent results](review-evidence/laguerre-production-validation.json).

Black and configured Ruff checks pass. Verification is Linux/CPython 3.12;
the complete supported platform matrix is not established here.

## October 22, 2026: scalar projection and feasibility repair

This is the earlier scalar-projection and feasibility-repair checkpoint.
Source derives from `gibbus_latest.zip`. The abandoned endpoint proposal remains
outside scope. No GitHub repository, branch, issue or release was changed.

### Completed checks at that checkpoint

| Scope | Collected | Passed | Failed/errors | Skipped | Expected failures | Unfinished |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final source suite | 6,074 | 6,070 | 0 | 2 | 2 | 0 |
| Installed-wheel batch | 106 | 106 | 0 | 0 | 0 | 0 |

The source suite took 487.408 seconds. The installed batch took
68.403 seconds. JUnit XML and per-case JSON are retained:
[full source results](review-evidence/solver-projection-final-results.json),
[full source XML](review-evidence/solver-projection-full-final.xml),
[installed results](review-evidence/solver-projection-installed-results.json),
[installed XML](review-evidence/solver-projection-installed-wheel.xml).
The final console progress stream stopped updating before its summary, but the
JUnit record includes every collected case and the command exited successfully.
No failure or timeout is inferred from that stale progress display.

At that checkpoint, the original 1,200-point degree-12 lognormal accounting regression passed with
its request and `2e-9` assertion unchanged. The 3,000-point degree-12 lognormal
and degree-4 two-linear-tail Laplace acceptance targets remain strict expected
failures. No numerical assertion or solver/shape tolerance value was weakened,
and no failing test was converted to an expected failure.

The final source includes 14 additional tests for projection QP bounds,
coordinate scaling, unrelated LP initialization, checked row repairs and
invalidation of a stale pre-repair certificate. A focused 68-case batch passed;
a final 11-case QP check includes the unrelated-LP regression. The full suite
also covers the previously repaired unit-scale, stacked-fit, span-equivalence,
heavy-tail and automatic-selection cases.

## Environment and installed verification

Linux x86-64, CPython 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Cython 3.3.0 and
GCC/G++ 13.3. A fresh wheel built all 14 extensions at the production settings.
Installed smoke imports all 14 extensions and the new projection module.
The test runner comes from the test environment with the wheel installation
first on `PYTHONPATH`, outside the source checkout. The wheel and current source
Python package files compare byte for byte. The installed batch covers records,
serialization, adversarial numerics, new solver regressions, the original
high-degree fixture, unit-scale equivariance and stacked-fit behavior.

Four warnings in the installed batch are the documented integer SoftPower to
polynomial-family conversion. Current full debug, other interpreters/platforms,
minimum dependencies and comparative performance qualification remain open.
Native source experiments were reverted; the retained implementation changes
Python and tests. Packaging and style checks accompany the source archive.

## Intermediate results and prior checkpoints

| Checkpoint | Collected | Passed | Failed | Skipped | Expected failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous endpoint-search checkpoint | 6,060 | 6,055 | 1 | 2 | 2 |
| First projection broad run | 6,073 | 6,062 | 7 | 2 | 2 |
| Final projection checkpoint | 6,074 | 6,070 | 0 | 2 | 2 |

The first projection run completed in 443.44 seconds and exposed overbroad
initialization changes and process-global LP scheduler dependence. Those
regressions were fixed before the final source and installed checks. See the
[resolution report](review-evidence/solver-projection-resolution.md) and the
[earlier validation record](history/endpoint-search-validation.md).

Two full suites were run in this pass: the first established regressions and
the second verified their fixes. Bounded prototypes, interrupted diagnostics
and focused checks are separate from those full-suite results. Timings are
acceptance-run observations, not controlled performance benchmarks.
