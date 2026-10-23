# Current development status — October 23, 2026

**Version:** `0.2.0.dev0`. **Status:** unreleased development checkpoint.

The library uses the general potential basis and coefficient-linear curvature
constraints. It supports weighted point and interval observations, finite
mixtures, automatic component/degree/log/span selection, selected tail families
with optional within-family parameter estimates, post-fit distribution operations
and durable format-2 persistence. Experimental Taylor candidate search remains
optional and retains full shape acceptance.

The [follow-up register](review-follow-up.md) records all original findings and
current dispositions. Completed bounded work covers request precedence,
resampling and exp-view integration, empirical-tail scaling/deadlines, record
validation and cache identity, development-branch CI, documentation/package
hygiene, exact endpoint-log feasibility, interior point stationarity and
uniform rejected-endpoint cut discovery, checked feasibility repairs,
support-appropriate recurrence coordinates, a common Newton controller and
tighter curvature contact arithmetic.

## Verification

The current common-controller and polynomial-coordinate refactor is described in
[fitting coordinates](fitting-coordinates.md) and the
[resolution report](review-evidence/coordinate-refactor-resolution.md).
Production verification uses Linux x86-64, CPython 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, Cython 3.3.0 and GCC/G++ 13.3. All 14 native extensions were
built at `-O3`.

- Complete source sweep: **6,104 passed, 1 regression, 2 skipped, 1 expected failure**, out of 6,108 cases in 746.335 seconds. The final guard fixes the bounded warm-start regression; its affected-case rerun passes **28 tests** in 89.569 seconds.
- Reconciled unique source cases: **6,105 passed, 2 skipped, 1 expected failure**, with no unresolved failure. This combines the full sweep and final affected-case rerun; it is not a fresh all-green full invocation after the final guard.
- Installed wheel outside the source checkout: **182 passed** in 98.579 seconds; all 14 native extensions imported. The broad wheel batch emitted four documented SoftPower conversion warnings and one invalid-arithmetic warning in optional contact inspection. A final finite-derivative guard removes the latter; the affected serialization check passes again with runtime warnings treated as errors in both source and installed wheel.
- Both degree-12 likelihood/acceptance regressions pass with existing numerical assertions. The degree-4 two-linear-tail Laplace test retains its strict expected-failure marker.
- Black and configured Ruff checks pass. Source hashes identify the delivered source verified by the sweep plus final targeted reruns; 81 Python/native package files match the installed wheel byte for byte.

Historical debug verification predates this refactor. Current full debug and
other interpreter/platform matrices are not established.

## Open work

R16 retains the degree-4 two-linear-tail Laplace target OI-E2-6 and broader
face/derivative conditioning work. The 1,200-point accounting regression and
3,000-point degree-12 lognormal target OI-E2-3 are resolved. No numerical assertion or solver/shape threshold was relaxed,
and no failing test was converted to xfail.

An uncertified return warns by default; inspect `fit_diagnostics` and
`spectral_diagnostics` before quantitative use. An explicit tail choice can be
statistically inappropriate even when its fitted model passes numerical shape
acceptance.

The declared CPython 3.10–3.14/Linux/Windows/macOS targets need the current
source's CI matrix. Minimum dependency combinations and changes to 3.10/3.15
support policy remain open. Mixture cost, difficult curvature contacts,
comparative predictive-tail validation, per-fit parallelism, broader architecture
profiling and automatic tail-family comparison remain engineering/research work.

## Historical evidence

- [Original source review](history/source-review-2026-10-22.md).
- [First implementation pass](history/implementation-pass-one.md).
- [Second implementation checkpoint](history/implementation-pass-two.md).
- [G3-064 checkpoint](history/g3-064-status.md).
- [Benchmark findings](benchmark-findings.md),
  [curvature rows](G3-078_CURVATURE_ROWS.md),
  [automatic tail parameters](AUTOMATIC_TAIL_PARAMETERS.md),
  [series search](series-staged-search.md) and
  [runtime update](fit-runtime-update.md) retain earlier experiment evidence;
  current behavior is documented in the user guides.
