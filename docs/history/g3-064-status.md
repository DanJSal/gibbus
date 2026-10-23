> Historical G3-064 checkpoint. For the current snapshot see [Development status](../development-status.md).

# Current G3 development status — 2026-10-20 (G3-063 working candidate)

**Source checkpoint:** G3-064 working candidate built on G3-063, G3-062, G3-061 and G3-060. **Release status:** development snapshot, not a published release. Both formerly failing degree-2 `Polynomial(1)` regressions now pass after a targeted automatic-geometry correction; the G3-060 full-suite results were obtained *before* these changes. The complete suite on the new candidate and other interpreters/platforms remain unverified.

G3 stages 1–5 are implemented: the public `Distribution.fit()` uses G3, all nine support/boundary-log configurations are implemented, and the general potential basis supports constrained Newton fitting, interval-censored and weighted data, finite mixtures, automatic component count, degree, endpoint-log and span selection, post-fit distribution operations, and persistence. Stage 6 (hardening, accuracy and cost of automatic selection) has reached its exit point: the remaining items are bounded bug fixes and release verification, listed below, plus documented limitations that need design work beyond G3.

## Work completed since G3-051

| Checkpoint | Implemented work | Evidence / qualification |
| --- | --- | --- |
| G3-052 | Fitting-coordinate normalizer, spectral evaluation at near-ULP supports, identifiability guard for automatic endpoint logs | Original three OI-E2-1 acceptance tests pass. |
| G3-053 | Fitting-coordinate post-fit moments for extremely narrow offset mixtures; optimized Cython build | Focused regressions green. Full suite not completed. |
| G3-054 | Structured Newton/shape/exchange failure diagnostics for beta, high-degree and linear-tail investigations | No convergence thresholds relaxed; adverse cases remain tracked, not universally release-blocking. |
| G3-055 | Log-concave family registry, lazy balanced case generator, truncation, independent sums, mixtures, censoring and stable IDs | 32 named family definitions; optional pairwise expansion gives 3,739 distribution specs; 125 compiled-tree generator checks passed at the original checkpoint. |
| G3-056 | Isolated fit runner with resume, timeouts, accuracy and diagnostics records | 137 tests passed and 11 real smoke fits converged across all nine configurations at the original checkpoint. |
| G3-057 | Sample-size-shrunk skew-aware full-line transition geometry **default** | 1,877 selected regressions passed, 2 skipped. No change to half-line, weighted/censored, mixture-component geometry, coefficient optimizer or certification criteria. |
| G3-058 | Consolidated current source, historical materials, findings, decisions and priorities | Documentation-only; no new fitting change, release claim, or new benchmark. |
| G3-059 | Finalists continue their exploration EM; candidate faces start from the restored incumbent; finite-support log-concavity margins | 122 paired runner fits: total fit time 1268 s → 419 s, largest 293 s → 50 s, held-out density within 1.4e-5 nats/obs (one +4.5e-3). See [Benchmark findings](../benchmark-findings.md#g3-059-profiling-and-fixes). |
| G3-060 | One multistart finalist by default; exact ratio-test step after a shape-rejected Newton trial (compiled kernels rebuilt) | Paired with G3-059 code on one machine: total fit time 478 s to 323 s (96 cases) and 272 s to 169 s (mixture targets), no selected count changed; full suite 5848 passed, 2 pre-existing `Polynomial(1)` failures, 1 xfail (OI-E2-3). See [Benchmark findings](../benchmark-findings.md#g3-060-mixture-cost-analysis). |
| G3-061 working | Nonfinite generated rows raise `RuntimeError`; optional uncertifiable face starts retain certified incumbents | 112 targeted passes, 1 xfail; two degree-2 `Polynomial(1)` tests still failed. `Exponential` overflow itself is not repaired. |
| G3-062 working | Matching automatic gate steepnesses for two `Polynomial(1)` tails | The original two degree-2 failures now pass, and 12 additional seed/family fits certify; release-focused suite 125 passed, 1 xfail. Full suite, paired performance and all other platforms remain unverified. |
| G3-063 working | Scope face-dependent tail curvature cuts to their originating face and subfaces; retain global finite-point cuts | Deterministic quartic counterexample reproduced and repaired; 134 bounded focused tests passed, 1 xfail. The historical degree-3 incident was not replayed, and full-suite and platform checks remain open. |
| G3-064 working | Enforce exact coefficient-face equalities for opposing leading tail curvature rows | Reproduced and resolved degree-3 `Polynomial(1)` strict-interior failures, including the 800-point Laplace case. Degree-4 linear-tail exchange remains an xfail; full release verification remains open. |

## Open issues

**To fix before a G3 release** (each fails loudly; none returns a wrong certified fit):

1. **Remaining linear-tail numerical consistency.** The earlier two degree-2 `Polynomial(1)` failures came from skew-adaptive *unequal* gate steepnesses: the gated quadratic had opposing leading curvature signs at the two infinite ends, requiring its coefficient to vanish and sometimes leaving no strictly convex potential. G3-062 synchronizes automatic steepnesses *only when both ends explicitly request `Polynomial(1)`*, preserving automatic locations and explicit steepnesses. Both original regressions now pass. G3-063 corrected a structural inconsistency: tail cuts derived on a face could be carried into a larger face where a stronger asymptotic atom was again allowed. A globally convex quartic counterexample establishes and tests this failure mode; the exact G3-060 degree-3 separator/cut *trace* has not been replayed. G3-064 additionally enforces the exact curvature face for the reproducible degree-3 strict-interior failures: its two leading asymptotic curvature functionals are opposites, so their coefficient must be zero. Explicit unequal-steepness gate geometry remains user-controlled and is not covered by the same convergence guarantee.
2. **`Exponential` tail generated-row overflow.** G3-061 now raises `RuntimeError` rather than `ValueError` for a nonfinite internally generated row, but has not prevented the overflow; several high-degree fits still fail to converge.
3. **`SoftPower` face starts.** G3-061 retains a converged, shape-certified incumbent when an optional candidate face fails to certify its start; one reproduced case now succeeds. The entire original failure set has not been retested.
4. **Build and test matrix.** Only CPython 3.13 Linux x86_64 has been built and tested. The declared range (CPython 3.10–3.14 on Linux, Windows, macOS; `pyproject.toml` `[tool.cibuildwheel]`) needs the CI workflow, followed by the release checklist in [Releasing](../releasing.md). Building from source requires `setuptools>=77` (PEP 639 license metadata).

**Documented limitations accepted for G3** (design work, not bug fixes):

5. **Mixture cost.** A mixture fit still costs a median of tens of times a single-component fit (largest benchmark fit about 40 s on a 2-core machine). Near a degenerate curvature contact the cut exchange converges linearly; a more effective curvature-cut strategy or a convex-by-construction parameterization may be needed ([analysis](../benchmark-findings.md#g3-060-mixture-cost-analysis)).
6. **Full-line default tails.** With `tail=None` the full-line tails derive from the bulk degree, and raising the degree raises the tail growth order. On linear- or exponential-tailed data (Gumbel, logistic, hyperbolic secant) this costs a median 0.011 nats/observation against the best fixed configuration; decoupled `SoftPower(1.5)` tails cut that about 4x. The default was deliberately kept ([experiment](../benchmark-findings.md#g3-060-basis-and-tail-selection-experiment)).
7. **No automatic tail-family selection.** Explicit tail requests are authoritative; automatic choice among `Polynomial`, `SoftPower` and `Exponential` is a possible later feature, only where no family is requested.
8. **Open convergence items** (strict xfail, fail loudly): OI-E2-3, an explicit degree-12 fit of a 3000-point lognormal sample on `(0, inf)`; OI-E2-6, `Polynomial(1)` tails at both ends with bulk degree 4 on Laplace data, which exhaust the curvature cut exchange (`tests/test_stage6_solver_triage.py`).
9. **Automatic component counts** always refine the screen winner's neighbors; screen BIC is not a safe bound for pruning them.
10. **Benchmark oracles.** Sums of independent variables have no independent truth PDF/CDF oracle in the case generator, so their accuracy is scored on held-out likelihood only.

See [Limitations](../limitations.md) for the user-facing statement of these items.

## Latest benchmark (G3-060)

Paired against G3-059 code on one 2-core machine (CPython 3.13.16, NumPy 2.5.3, SciPy 1.18.1), seed 91, holdout 512, 600 s timeout:

| Manifest | G3-059 total fit time | G3-060 total fit time | Largest fit | Status / selected count |
| --- | ---: | ---: | --- | --- |
| 96 balanced G3-057 cases | 478 s | 323 s | 83 s → 40 s | unchanged: 95 ok, 1 fit error (`G3C-91ce6e608ed06a71d729`, two components fitted to a three-component target) |
| 26 mixture targets with `n_components="auto"` | 272 s | 169 s | 51 s → 19 s | unchanged: 26 ok |

Against the original G3-057 benchmark, all six 28-second timeouts and all ten post-fit diagnostic errors are resolved; the formerly unresolved `G3C-2a91bd6857739ffcc604` (517 s on this machine before G3-059) fits in about 50 s. Held-out density changed by less than 1e-3 nats/observation on all but four of the 122 fits; the two largest changes (-0.061 and +0.61) are censored cases whose held-out score is dominated by a few extreme points, and the other two improved (+0.009, +0.008). Details: [Benchmark findings](../benchmark-findings.md). The G3-057 baseline numbers remain in that document and in the handoff's `benchmark/` folder.

## Verification and scope

- Full suite on G3-060 (CPython 3.13 Linux x86_64, all 11 extensions rebuilt with `setup.py build_ext --inplace`): 5848 passed, 2 skipped, 1 xfailed (OI-E2-3), 2 failed (open issue 1); the two Stage 6 test modules restored afterwards (`test_stage6_coordinate_stress.py`, `test_stage6_solver_triage.py`) give 30 passed and 1 xfailed (OI-E2-6).
- Lint: no new Ruff, Black or `cython-lint` findings in changed files; long-standing findings remain.
- Not verified: other interpreters and platforms, wheel builds, a PyPI release. Nothing has been pushed.
- The historical G3-051 curated archive is retained unchanged in the handoff; its status narratives are superseded by this document.
