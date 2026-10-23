# Review follow-up register

This register tracks the October 22, 2026 review of the general-basis source.
The [full original findings](history/source-review-2026-10-22.md) are retained
with locations, reproduction evidence, rationale and suggested follow-ups.
The old compact endpoint-geometry proposal is outside the implementation scope.
The current package is `0.2.0.dev0`, an unreleased development checkpoint.

“Implemented” means the bounded source/configuration/documentation work is in
this tree. It does not imply that the full Python/platform/release matrix has
passed. See [Development status](development-status.md) for verification.

| ID | Observation | Disposition |
|---|---|---|
| R01 | Removed serialization test paths | Implemented: CI, release sdist checks and cibuildwheel target `test_serialization.py`. |
| R02 | Missing CI test dependency | Implemented: CI installs the built distribution with its declared `[test]` extra, including mpmath. |
| R03 | Branch pushes skip CI | Implemented: push checks cover all branches; PR/manual entry points remain. |
| R04 | Incomplete wheel smoke inventory | Implemented: all 14 extensions are listed, including series and empirical-tail kernels. |
| R05 | Degree overrides ignored with a basis | Implemented: resolve and apply per-component precedence consistently; retain basis-aware odd-degree properness checks and automatic warm-start degrees. |
| R06 | Uncertainty refits lose model structure | Implemented: inherit selected count/degrees, endpoint logs, resolved tails and span policy. Automatic geometry/span expansion rerun; original degree/log/tail-parameter selection requires explicit overrides. Bootstrap row weights follow resampled observations. |
| R07 | Resampling basis override conflicts with support | Implemented: structural overrides replace/update the inherited basis without injecting conflicting legacy keywords. Explicit caller conflicts still fail normally. |
| R08 | Exp-space Monte Carlo refits the wrong variable | Implemented: sample/refit in base coordinates and use PIT invariance under exp. |
| R09 | Empirical tail scaling overflow/underflow | Implemented: standardize before variance calculations/interpolation and check physical rate representability. |
| R10 | Unsuccessful tail optimizer accepted | Implemented: require finite successful results; a failed coefficient start tries the second configured start. Scalar-search failures are explicit. Statistical calibration across families remains research work. |
| R11 | Missing tail-screen deadline checks | Implemented: checkpoints between tail sides/candidates, coefficient starts and optimization callbacks; the deadline remains cooperative. |
| R12 | Tail kernel bypasses debug safety policy | Implemented: remove per-file safety overrides and validate matching nonempty node arrays and positive finite rate. A separate safety build compiled all 14 extensions at `-O0` and passed 34 review/kernel plus 39 solver tests. Optimized debug compilation and the broader debug/platform matrix remain open. |
| R13 | Malformed basis records leak implementation errors | Implemented: validate required fields, strict record version, decoded types, slot/column layout and coordinate/geometry values before native compilation; reject scalar Unicode record storage explicitly. Also verify that coordinate support kind and canonical endpoints agree with physical support and the affine map, and require resolved parameters for active gates/tails. |
| R14 | Wrong inner serialization record documented | Implemented: document basis-record version 2 and suppressed leading-tail faces. |
| R15 | Same-version caches can cross implementations | Implemented: optional cache fingerprint includes Python/native module contents, Python ABI/byte order and NumPy/SciPy versions. Missing/mismatched fingerprints rebuild from the durable model; package development version is distinct from 0.1.0. |
| R16 | Numerical regressions and expanded-suite failures | Partial: checked feasibility repairs, direct Laguerre evaluation and tighter contact arithmetic resolve the 1,200-point accounting and 3,000-point degree-12 lognormal regressions. Exact log-domain polish, interior stationarity and coupled/general endpoint discovery remain covered. The degree-4 linear-tail Laplace target, broader face conditioning and numerical hardening remain open. Solver/shape threshold values and regression accuracy assertions are unchanged. |
| R17 | Contradictory checkpoint status | Implemented: one current status page; earlier G3-064 status and runtime patch are retained under `history/`. Historical experiment notes remain historical evidence. |
| R18 | Public API/defaults documentation incomplete | Implemented: synchronize exports, timeout exception, tail diagnostics, constructor defaults, automatic-parameter limits and basis-vs-convenience behavior. |
| R19 | Row count documented instead of Kish size | Implemented: explain Kish effective size for relative weights and distinguish frequency counts. No statistical selection policy was changed to match old prose. |
| R20 | Diagnostic help overstates guarantees | Implemented: distinguish grid curvature estimates from global shape evidence and finite-sample fitting from population tail assumptions. |
| R21 | C++ prerequisite missing | Implemented: main source-build guides require C and C++ compilers plus Python headers. |
| R22 | Branch docs route readers to old main | Implemented: relative repository documentation links; remove version-mismatched package doc/changelog URLs and the default-main CI badge. |
| R23 | Recent Python code needs formatting | Implemented: Black formatting and explicit Ruff correctness/import rule scope. Numerical assertions were not weakened. |
| R24 | Generated/experimental leftovers | Implemented: ignore generated `.cpp`, preserve authored `.hpp`, archive the runtime patch and reorganize stray changelog bullets. Some G1 owner comments remain useful historical context; broad editorial/refactoring cleanup can continue separately. |
| R25 | Sdist omits referenced evidence | Implemented: include docs JSON/log/patch/XML evidence as well as Markdown. |
| R26 | Statistical sieve explanation missing | Implemented: explain fixed-structure nesting, omitted-information/span tests and their distinction from BIC component selection. |
| R27 | Architecture/numerical consolidation | Started through request/resampling boundary fixes and shared tail-budget handling. Broad eager-import/validation profiling and replacement of external numerical routines remain deferred pending measurements; no second fitting backend is introduced. |
| R28 | Uncertified-return policy unclear | Implemented: retain the documented degraded-result policy and emit a default `RuntimeWarning`; suppression is explicit and diagnostics remain authoritative. |
| R29 | Release/platform/benchmark program unfinished | Deferred explicitly: other interpreters/platforms, full debug matrix, minimum dependency builds, Python 3.10/3.15 support-policy changes, independent sum oracles, comparative accuracy/scaling benchmarks and per-fit parallelism. |

## Numerical fixes and investigations under R16

### Resolved in this checkpoint

- `test_tail_class_fits_answer_queries_and_round_trip[laplace-exp1-lower-mixture]`:
  certified polish lost only `1.38e-14` mean likelihood. The EM monotonicity
  guard now permits its floating-point roundoff envelope while retaining shape,
  Newton and EM verification requirements. Material likelihood losses still fail.
- `test_boundary_amplitudes_are_linear_nonnegative_coordinates`: feasibility
  polish could produce a tiny negative endpoint-log amplitude while considering
  it within the scaled row-roundoff floor. Log amplitudes have an exact domain
  boundary, so polish now requires their nonnegative slack before it returns a
  corrected point. A focused unit test covers a violation below that floor;
  the original beta-data regression passes.
- `test_point_fit_is_invariant_to_extreme_global_weight_scale`: the conic
  subproblem's gap floor stopped an interior point solve before full numerical
  stationarity, amplifying tiny rounding differences in equivalent weights.
  Up to three guarded, well-conditioned Newton corrections reduce the scaled
  gradient. Each must preserve all retained rows exactly, preserve likelihood
  to rounding, pass the full shape search and retain the existing bound target.
  Unresolved or infeasible trials keep the incumbent. No input rounding or
  weight quantization was introduced. The existing `2e-13` CDF assertion passes.
- `test_untestable_larger_span_ends_the_automatic_expansion`: the former fixture
  assumed suppressed exponential leaders implied infinite larger-span moments.
  Its fitted quadratic coefficient is positive (about `0.264`), so every
  exponential statistic has finite moments under that tail. The real fit
  legitimately stops when its score accepts the current span. The failure
  handler now has a deterministic injected quadrature-failure test that checks
  the reason and exact retention of the baseline parameters, plus a separate
  real-data test of finite-moment stopping. This corrects a false fixture premise;
  it does not relax a numerical assertion.

### Earlier rejected-endpoint cut follow-up

The fast Newton gate stops at the first shape-rejected component and can use
valleys inside the incumbent's domain. For coupled fits and bases with general
columns, the coordinator now resolves every component at that rejected endpoint
before the next exchange. Refined point and tail cuts keep their originating
face scope. Non-convex endpoints are not integrated or accepted as fitted points.

The completed 6,060-case intermediate run passed all seven formerly failing
acceptance cases below, while exposing three other regressions investigated in
this follow-up:

- `test_stacked_m_step_without_shared_columns_is_independent_fits`.
- `test_degree3_two_linear_tails_use_exact_curvature_face[1]` (all three seeds pass).
- `TestHeavyTailRobustness.test_every_k_fits[2]`, `[3]`, `[4]`.
- `TestHeavyTailRobustness.test_more_components_do_not_fit_worse`.
- `TestHeavyTailRobustness.test_k2_converges_with_certified_natural_components`.

A new component-order permutation test also passes. The existing numerical,
shape, likelihood and pointwise assertions are unchanged.

Unconditionally adding refined endpoint rows regressed two degree-8 unit-scale
checks and the single-component screen used by
`test_an_unsupported_explicit_component_count_says_what_to_do`. A same-environment
comparison reproduced all three failures with extra searches and passed all four
cases with the previous schedule. At that checkpoint extra searches were therefore confined to
coupled fits and bases containing gated or non-monomial columns. Single ungated
monomial fits retained their established schedule. The four affected cases plus
the five stacked/span/order cases pass together (nine tests). That bounded policy has now been replaced by the common controller: full endpoint
cuts replace approximate gate cuts, avoiding their duplicate accumulation. See
[fitting coordinates](fitting-coordinates.md). General active-contact conditioning
is not thereby solved.
The completed intermediate and final run scopes are recorded in
[implementation validation](review-validation.md).

### Remaining numerical work

The [projection resolution checkpoint](review-evidence/solver-projection-resolution.md)
records the retained feasibility safeguards and scoped scalar-QP recovery. The
1,200-point degree-12 likelihood-accounting regression passes; the distinct
3,000-point acceptance target also passes with production recurrence coordinates.
The current common-controller implementation and verification are recorded in
[the coordinate refactor report](review-evidence/coordinate-refactor-resolution.md).
See the [production resolution](review-evidence/laguerre-production-resolution.md)
for the direct recurrence kernels, contact arithmetic and current verification.

The [high-level review](review-evidence/degree12-high-level-review.md) adds
log-concave population controls, an exact orthogonal-basis comparison, and an
observed feasibility repair that increases NLL materially despite a tiny
coefficient change. It distinguishes the shared numerical weaknesses from
ordinary automatic-degree behavior and records concrete implementation targets.

The [degree-12 conditioning investigation](review-evidence/degree12-conditioning-investigation.md)
records independent high-precision checks of captured Newton subproblems and
bounded warm-start experiments. Those checks narrow the numerical diagnosis but
do not establish a converged full fit; no experimental optimizer change is
retained.

The earlier degree-12 investigation recorded raw Hessian condition estimates
near `1e29` and diagonally normalized examples around `1e14`–`1e16`.
Whitening, regularized one-step recovery and retained M-step cuts remain
excluded experiments. The retained projection path checks its quadratic bound
against the original model and can report approximate convergence under the
existing accuracy floor. A shape verdict alone is never promoted to optimizer
convergence.

Mixture runtime remains material: the intermediate full run's four-component
heavy-tail case took about 213 seconds and its 20,000-row two-component case
about 72 seconds. These are acceptance-run timings, not controlled comparative
benchmarks. A bounded large-mixture trace showed repeated intermediate M-steps
reaching the exchange limit. Per-fit constraint reuse and a general constrained
stationarity polish need further investigation.

The additional high-degree, history and boundary failures were reproduced in
an unchanged latest-source build before fixes, establishing that this pass did
not introduce them, without claiming their original introduction history.

OI-E2-3 (degree-12 half-line lognormal) passes without its expected-failure
marker. OI-E2-6 (degree-4 two-linear-tail Laplace fit) remains open. No test
was converted to xfail. The solver and shape acceptance tolerances remain intact.

## Deliberate scope limits

- No GitHub branch, PR, issue, publication or release was created here.
- No obsolete backend, contact Gram blocks or older persistence reader was
  restored. The abandoned design proposal is not an implementation dependency.
- This is development-branch preparation. The open convergence work and release
  matrix are not represented as complete.

## Open GitHub issue mapping

The orientation review read ten open issues in `DanJSal/gibbus`. No issue was
edited or closed during this implementation pass.

| Issue | Current treatment |
|---|---|
| #17, compact endpoint proposal | Abandoned by the owner; excluded from this work. |
| #8, API boundaries | Concrete request/resampling integration fixes implemented; broader architecture remains R27. |
| #9, fit-statistics interface | Deferred public API design, including weighted/censored/derived-model semantics; no replacement object invented here. |
| #10, durable serialization | R13–R15 implemented without restoring older readers. |
| #15, sieve documentation | Current generalized method documented under R26. |
| #2, validation/benchmarks | R16/R29 remain open; bounded tests are recorded honestly. |
| #14, numerical kernels | Broader consolidation remains measurement-driven R27 work. |
| #13, per-fit parallelism | Deferred; parallel pytest workers do not implement this feature. |
| #11, Python 3.10 policy | No support-policy removal in this checkpoint. |
| #12, Python 3.15 policy | No new support claim without the dependency/build/platform matrix. |
