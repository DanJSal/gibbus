> Historical review of the uploaded source before this implementation pass.
> Current dispositions and validation are in [the follow-up register](../review-follow-up.md).

# Gibbus development-branch review

Reviewed October 22, 2026. Scope: the uploaded `gibbus_latest.zip`, compared with `gibbus_github.zip`, plus the ten open GitHub issues read during the orientation pass. This is a read-only review of the submitted source. Builds, probes, and tests ran in a separate copy. The 217 files extracted from the latest archive were fingerprinted; their contents were unchanged by this review.

The abandoned endpoint-geometry proposal and the plan to implement it are excluded. Its architecture is not a requirement, a benchmark, or a source of missing-feature findings. No finding below asks to reinstate the old natural-polynomial backend, its Gram/contact formulation, or its old serialization reader.

## Assessment and priorities

The newer tree is a substantial, coherent implementation rather than an incomplete collection of replacement modules. The existing distributional interface is broadly retained, the new basis and fitting paths are wired into the public estimator, the Linux production build works, and the source and wheel packages contain the new compiled sources and header. Nevertheless, it is not yet a clean development checkpoint: there are confirmed public-API integration defects, broken verification configuration, contradictory documentation, and failing regressions.

Use these priorities to plan the next pass:

- **A — before calling the uploaded branch reviewed/clean:** repair a confirmed API contract problem, make branch verification usable, or correct a material documentation claim. These are recommendations for a useful development checkpoint, not technical restrictions on pushing a branch.
- **B — next hardening/cleanup:** actionable defects or maintenance risks that can remain on a development branch if accurately logged and scoped.
- **C — deferred/design/release:** worthwhile work or a policy decision that is not necessary merely to upload a development branch.

Every observation has a stable ID. A confirmed source finding means the relevant code/configuration was inspected; it does not imply a platform-specific run was performed. Historical evidence is explicitly distinguished from this review's fresh results. Fix suggestions are proposed work, not changes made.

## Verification performed

### Environment and inputs

- CPython 3.12.14, Linux x86-64; NumPy 2.3.5; SciPy 1.17.0; Cython 3.3.0; GCC/G++ 13.3.0; pytest 9.1.1.
- Compiled with the checked-in production `-O3` configuration, without native tuning, unrolling, or fast-math. Set `CC=gcc CXX=g++`: this runtime's default compiler setting names a missing `clang` binary. That initial environment failure is not a repository defect.
- BLAS/OpenMP threads limited to one for test/probe runs. The broad test run used two pytest workers.
- `gibbus_latest.zip` SHA-256: `6092f46e992dc59401eb8b7386f98776cdd457d0a18572a611abb81da3aaa8dd`.
- `gibbus_github.zip` SHA-256: `b9a765bb58b406a633d4cf32b2e47cce35718720e281b0859fe8acf2b0874122`.

### Completed checks

1. All Python source files parsed successfully.
2. All 14 Cython extensions built successfully with GCC/G++ at production optimization.
3. Built the sdist. It includes the new `.pyx`, `.pxd`, `_series_number.hpp`, `py.typed`, and current serialization tests.
4. Created a CPython 3.12 Linux wheel from that build and installed it into a separate environment. Confirmed all 14 extensions and all Python/Cython/header source files were present.
5. Ran the shipped installed-wheel smoke test, then explicitly imported the three extensions missing from that smoke list. Imports came from the installed wheel, outside the source checkout.
6. Installed-wheel public smoke: a normal degree-2 fit converged; CDF/PPF round-trip error was `1.93e-15`; serialization round-trip log-density difference was zero on the sample. This checks basic wiring, not universal numerical accuracy.
7. Focused verification: **85 passed, 2 failed, 1 xfailed in 27.34 seconds**. Breakdown: fitting budget 17 pass; empirical-tail integration 9 pass; serialization 35 pass; solver triage 2 pass/1 xfail; Stage 6 blockers 16 pass/1 fail; scalar curvature rows 6 pass; the selected coupled-mixture regression 1 fail.
8. Full collection found **6,015 tests**. Broad-run result: **interrupted by the 600-second process watchdog near 99%; no final pytest summary was produced**.
9. Checked documentation file links: no missing relative target files in the repository's Markdown, excluding the abandoned proposal. Found 21 hardcoded links to main; see R22.
10. Read-only formatting/lint: Black reported 39 files needing formatting, 95 unchanged. Explicit Ruff `F` checks found nine unused imports, all in tests/helpers, and no undefined-name finding. A broader Ruff run reported 183 findings, predominantly import ordering/style; these are not 183 demonstrated runtime bugs.

The full cross-platform/Python matrix, debug extension rebuild, minimum-dependency matrix, independent external-method comparisons, and complete statistical certification were not performed. Build success on this environment does not establish those results.

## Findings register

### R01 — Verification still names a deleted serialization test [A, confirmed source]

**Locations:** `.github/workflows/ci.yml` source-distribution tests; `.github/workflows/release.yml` focused sdist tests; `pyproject.toml` cibuildwheel `test-command`.

All three invoke `test_serialization_v1.py`; it is absent from the newer archive. The current module is `tests/test_serialization.py`. These commands fail on the missing path before the intended focused suite can run. This is stale configuration, not a reason to restore the old tests/reader. Replace the targets with the current test module and check every maintained test command against the actual tree.

**Acceptance:** each configured command collects existing tests successfully from the installed package or extracted sdist as appropriate.

### R02 — Main CI does not install a dependency required during test collection [A, confirmed source]

**Locations:** `.github/workflows/ci.yml`, including the main test job; `tests/test_basis_evaluation.py`, `test_basis_growth.py`, `test_basis_intervals.py`, `test_basis_admissibility.py`, `test_basis_tails.py`, `test_basis_interval.py`.

These modules directly import `mpmath`. CI installs the wheel plus pytest/pytest-cov, without the test extra or explicit mpmath. `pyproject.toml` correctly lists mpmath in the test extra, but installing a plain wheel does not install that extra. A clean runner is therefore not provisioned for the new suite. My audit environment included mpmath deliberately.

**Follow-up:** provision the maintained test dependencies consistently, ideally from the declared extra. Verify fresh-environment collection; do not rely on a runner's incidental packages.

### R03 — A branch-only push does not trigger the main CI workflow [A, confirmed source]

**Location:** `.github/workflows/ci.yml` event configuration.

Push events are restricted to `main`. PRs and manual dispatch are supported, so there is an existing way to test the proposed branch, but simply pushing it will not produce the expected CI result. The wheels workflow also limits PR triggering by file paths.

**Follow-up:** either enable push checks for the intended development branch or document/use a draft PR or manual dispatch against that branch. No branch or PR was created during this review.

### R04 — Installed-wheel smoke inventory trails the build inventory [B, confirmed source and package inspection]

**Locations:** `tests/wheel_smoke.py::EXTENSIONS`; `setup.py::MODULES`.

The smoke list contains 11 of the 14 built extensions. It omits `_fit._series_search`, `_model._series_kernels`, and `_fit._tail_likelihood_kernels`. The reviewed wheel does contain and import all three. This is a verification gap rather than missing source. The optional tail accelerator deserves explicit verification because its import failure otherwise falls back to Python.

**Follow-up:** update or mechanically reconcile the smoke inventory and add a small end-to-end public operation where useful.

### R05 — Equal per-component degree overrides are ignored with an explicit basis [A, reproduced]

**Locations:** `_api/fitting.py::_run_basis_request`, `_basis_spec_of`.

The controller collapses equal per-component degrees into `global_degree`; `_basis_spec_of` then returns `request.basis` without applying that resolved degree. A two-component request using `BasisSpec(bulk_degree=2)` and `component_options=[{'poly_degree': 4}, {'poly_degree': 4}]` fitted degrees **[2, 2]**. The same component options without `basis` fitted **[4, 4]**. A one-component probe showed the same dropped override. The fitting guide explicitly allows component degree overrides alongside a basis.

**Follow-up:** define and apply one precedence rule consistently for equal and unequal per-component degrees, explicit basis degree, and warm starts. Cover the equal-degree collapse specifically. Do not silently fit a different requested degree.

### R06 — Resampling does not preserve the new model's structural request [A, confirmed source and live-object inspection]

**Location:** `_api/distribution.py::_resampling_fit_kwargs`, callers `bootstrap_bands` and Monte Carlo `goodness_of_fit`.

The inherited refit options contain only component count, support, and verbosity. They omit tail family/parameters, tanh settings, expansion policy, selected/fixed degree policy, and endpoint-log choices. A fitted explicit-basis model therefore defaults to a different fitting family in its uncertainty calculations. This also affects old-style fixed degrees: they revert to automatic selection. The docs describe conditional-on-selected-structure uncertainty more broadly than these defaults establish.

**Follow-up:** decide explicitly whether each structural choice is frozen or reselected, carry the needed request/provenance, and expose deliberate override behavior. If some choices cannot be reconstructed, require or clearly document `fit_kwargs` rather than imply they are preserved. Consider weighted-resampling semantics separately; blindly reusing an original weight vector for permuted rows is not sufficient.

### R07 — A basis override cannot be passed through the resampling API [A, reproduced]

**Location:** `_api/distribution.py::_resampling_fit_kwargs`.

Adding `fit_kwargs={'basis': spec}` leaves the helper's inherited `support` in place. The public fitter rejects `basis` together with `support`, so the bootstrap fails immediately with `ValueError: basis replaces poly_degree, support and log_boundary_*; got support`. This prevents the natural workaround for R06.

**Acceptance:** a basis override reaches the fitter without conflicting inherited legacy keywords; explicit conflicts supplied by the caller are handled predictably.

### R08 — Exp-space Monte Carlo goodness-of-fit refits the wrong variable [A, reproduced wiring defect]

**Location:** `_api/distribution.py::goodness_of_fit`, nested `simulate`.

With the model's default set to `exp`, `self.sample` generates positive `Y=exp(X)` values. The simulation feeds those values directly to a fresh base-space `Distribution.fit`, with the original base support, and scores that replica as a base-space fit. A spy at the refit boundary confirmed positive simulated values were passed to a base-default estimator. For bounded base support this can additionally produce out-of-support failures. A Monte Carlo null should repeat the same fitting procedure/coordinate transformation as the observed statistic.

**Follow-up:** refit simulated base/log coordinates and evaluate the same active-space statistic, or otherwise define a consistent transformed-model procedure. Add an exp-view regression with an independent construction, including bounded support. The existing base-space Monte Carlo tests do not cover this wiring.

### R09 — Automatic tail estimation bypasses the robust scaling used by the main fitter [B, reproduced]

**Locations:** `_fit/tail_estimation.py::_prepare_values`, `estimate_tail`; `_fit/_empirical_tail_likelihood.py::tail_data`.

The estimator uses raw `np.std` and `(z-center)/scale`, rather than the library's safe coordinate machinery. The same finite, nondegenerate normal sample resolved a tail parameter at ordinary scale, failed as zero-dispersion at `1e-200`, and produced overflow/empty-tail warnings and a failure at `1e200`. These examples isolate the estimator; they are not a promise that the final model's raw moments can be represented at every such scale. The existing affine-rate test only uses a moderate scale factor.

**Follow-up:** use overflow/underflow-safe centering and scale estimation, test extreme but representable data, and fail with a precise supported-range message when conversion to physical rates is impossible.

### R10 — Conditional tail optimization accepts unsuccessful optimizer output [B, reproduced by injected result]

**Location:** `_fit/_empirical_tail_likelihood.py::potential_fixed`; outer scalar searches in `_fit/tail_estimation.py::estimate_tail`.

The inner search accepts a result based on finite `opt.fun < 1e20`; it does not inspect `success`, status, gradient, or a usable convergence condition. Injecting a finite `success=False` iteration-limit result produced an ordinary returned parameter estimate. The configured starts list has two entries but only `xstarts[:1]` is attempted. The outer scalar searches likewise use `fit.x` without surfacing convergence diagnostics. Subsequent G3 shape checks do not establish the statistical quality of this preliminary estimate.

**Follow-up:** define acceptable estimator convergence, retry or decline failed proposals, and preserve meaningful diagnostics. A narrowly justified approximate solution may be acceptable; it should be deliberate and observable rather than inferred from finite objective value alone.

### R11 — Tail screening has no internal fitting-budget checkpoints [C, confirmed source; small overrun observed]

**Locations:** `_api/fitting.py::_run_basis_request`, `_fit/tail_estimation.py`, `_fit/_empirical_tail_likelihood.py`.

Automatic tail screening runs before the next component-solver budget check, across candidate grids and SciPy optimizations. The shared deadline is preserved and eventually raises correctly; it is not reset or swallowed. A `0.0001` second request on 20,000 points raised at the degree-selection checkpoint after approximately 0.008 seconds. That overrun alone is allowed by the documented cooperative contract. The improvement is checkpoint placement between independent tail sides/candidates, not a claim that the API promises a hard deadline.

### R12 — The new tail accelerator disables debug bounds checking locally [B, confirmed source]

**Locations:** `_fit/_tail_likelihood_kernels.pyx` first line; `setup.py` debug compiler directives and adjacent instruction.

This new `.pyx` file sets `boundscheck=False, wraparound=False` in its header. `setup.py` explicitly explains that such per-file directives override `GIBBUS_DEBUG_BUILD=1` and must not be repeated there. Consequently this kernel is outside the advertised debug bounds-check guarantee. Its entry point also derives iteration length from `ys` without checking that `h` and `logwr` have equal lengths. Current internal callers supply matching 96-node arrays; this review did not demonstrate an out-of-bounds access through the ordinary public fit path.

**Follow-up:** let the central debug policy apply and make the native entry point's dimension invariant explicit. Add a debug-build check for this extension.

### R13 — Malformed durable basis records leak implementation exceptions [B, reproduced]

**Locations:** `_postfit/basis_state.py::_model_from_record`, `_decode`; `_serialization.py::_validate_model`.

Removing `columns` from an otherwise valid basis record raises `KeyError`. Replacing decoded `request` or `geometry` with an integer raises `AttributeError` during compilation. The documented malformed-record contract is `ValueError`. Decoding accepts values without checking their expected dataclass type before passing them to compilation. Separately, `_validate_model` reads `basis_models.subdtype[0]` before proving a subdtype exists; malformed layouts deserve an explicit check.

**Follow-up:** validate required keys, record version/type, decoded object types, and field shapes before compilation/native evaluation. Preserve atomic load behavior. These are malformed-input boundary findings; ordinary current-format round trips passed in the focused suite.

### R14 — Serialization documentation specifies the wrong inner record version [A, confirmed source]

**Locations:** `docs/serialization.md` “Basis record”; `_postfit/basis_state.py::_RECORD_VERSION`, `_model_record`.

The guide says the inner basis-record version is currently **1**, but the writer emits **2** and includes `suppressed` tail ends. The guide's field list also omits that field and its role in rebuilding exact leader faces. This is distinct from the outer durable format, which is correctly documented as 2.

**Follow-up:** document the actual current basis record and face reconstruction semantics. This finding does not request support for any earlier outer or inner format.

### R15 — Development builds share a cache identity that does not identify their implementation [B, confirmed design risk]

**Locations:** `_serialization.py::_package_version`, `_pack_cache`, `_verified_runtime`; `pyproject.toml` version; `docs/development.md` serialization section.

The new snapshot still declares package version `0.1.0`. Cache reuse keys the implementation by that installed version plus model/runtime-content digests. Those digests establish unchanged data bytes, not unchanged builder algorithms or tolerances. Two different development snapshots with the same version can therefore accept the same cache as current even after implementation changes. The development guide already acknowledges this risk. I did not demonstrate a stale-cache numerical error across two installed snapshots in this run.

**Follow-up:** choose a development-version/build-identity or cache-schema policy before treating persisted caches as portable between working snapshots. Keep the mathematical model rebuildable. A new public release is not required just to push a branch, and old-format migration is not requested.

### R16 — The current snapshot has reproducible red tests [A for triage/disposition; not a requirement to solve all numerical cases]

**Locations:** test results below; `docs/G3-078_CURVATURE_ROWS.md`, `docs/borrowable-aspects-evaluation.md`, `docs/fit-runtime-update.md` for historical evidence.

The focused run freshly reproduced:

- `test_degree3_two_linear_tails_use_exact_curvature_face[1]`: degree-3 versus degree-2 equivalent-span log densities differ by up to `6.18759089e-6` against `rtol=atol=1e-6`. Both fit paths report convergence and a nonnegative shape verdict. The archive's curvature-row note already records this regression. Do not simply widen the assertion as cleanup; determine whether the optimizer accuracy, test quantity, or intended guarantee needs a justified change.
- `test_stacked_m_step_without_shared_columns_is_independent_fits`: returns `iteration_limit` instead of `converged`. Earlier experiment notes describe it as pre-existing relative to those particular patches. This run confirms it on the submitted snapshot; it does not establish its first introducing commit.

The degree-4/two-linear-tail solver-triage case remained an expected failure. Broader fresh-run details, including five additional named regressions, are appended under “Broad-run results.” Historical partial-selection logs mentioned degree-6 lognormal failures but provided only dots/`FFF`; this review separately reproduced K=2, K=3, and K=4 with complete failure details.

**Follow-up:** retain exact reproductions and classify each as a correctness issue, convergence limitation, test-contract problem, or environment-sensitive case. For a development branch, an honestly tracked numerical limitation can remain. Avoid claiming a clean full suite or making arbitrary tolerance changes to obtain green output.

### R17 — There is no single current, accurate development-status account [A, confirmed source]

**Locations:** `README.md`, `docs/README.md`, `docs/development-status.md`, newer tail/curvature/series/runtime notes.

The README/index still identify G3-060. The status document heading identifies G3-063 and its opening checkpoint G3-064. Later documents describe automatic tail parameters, G3-078 scalar-row constraints, new series work, and budget/screening changes. The older page's statements about unrepaired exponential overflow and which environments were tested are superseded in part by later source/tests/evidence. Some benchmark narratives refer to handoff/evidence folders not in this archive.

**Follow-up:** publish one current snapshot status, distinguish historical checkpoints from current claims, and link the actual in-repository evidence. Keep unresolved failures visible. Do not copy an old full-suite count forward as evidence for today's tree.

### R18 — Public API and configuration documentation is incomplete [A, confirmed source]

**Locations:** `README.md` public API; `gibbus/__init__.py` public API docstring and `__all__`; `docs/api-reference.md`; `docs/fitting.md`; `_model/basis_spec.py`.

The README and package docstring list only the original three public imports. The actual exports additionally include `BasisSpec`, tail classes, `TanhSpec`, `GibbusBasisWarning`, and `FitTimeoutError`. The API-reference page covers basis classes but omits the new timeout exception and `tail_parameter_diagnostics`. The detailed automatic-tail guide is not discoverable from the main docs index. Constructor defaults now support omitted/`auto` powers and rates, but the index still presents only explicit-parameter forms.

Document the important default distinction as well: `Distribution().fit(...)` maps to the legacy convenience policy (automatic degree/finite-endpoint decisions, minimal expansion), whereas a bare `BasisSpec()` specifies bulk degree 2, endpoint logs disabled, and automatic expansion. These need not be made identical; their different meaning must be clear. Also qualify blanket “odd bulk degrees are inadmissible on the full line” prose: explicit general-basis linear-tail models in the suite support degree 3 through an exact coefficient face, whereas the legacy `poly_degree` route rejects odd degrees.

**Acceptance:** one public-surface inventory matches exports and properties; defaults, units, estimation limits, persistence of diagnostics, and precedence are documented consistently.

### R19 — The weighting guide contradicts the statistical sample size used in code [A, confirmed source]

**Locations:** `docs/fitting.md` `sample_weight` row; `_fit/boundary.py::_effective_n`; `_fit/inputs.py::_normalize_basis_inputs`; `_fit/basis_mixture.py::_select_components`.

The guide says selection sample size is the number of rows regardless of weights. The new code uses Kish effective sample size `1 / sum(w**2)` for normalized relative weights, including BIC scoring and inferential/selection calculations; diagnostics expose `score_effective_n`. These differ substantially for concentrated weights. The warning that relative weights are not frequency counts can remain, but its row-count explanation is inaccurate.

**Follow-up:** agree on the intended statistical weighting contract, document the existing Kish behavior if retained, and cover concentrated/zero weights and subsampling. Do not silently change the statistical policy merely to match stale prose.

### R20 — Source-level diagnostic help overstates guarantees [A, confirmed source]

**Location:** `_api/diagnostics.py::spectral_diagnostics` docstring; `_api/component.py::log_concavity_margin`; `_api/distribution.py::fit` Raises section.

The spectral docstring describes `log_concavity_margin` as a certified lower bound and says components' polynomials are certified convex. The actual margin is derived from a finite quantile/endpoint grid, and the current model is a general basis. The Markdown diagnostics guide correctly qualifies this as numerical grid evidence. Also, the fit docstring cites Cauchy/Pareto as data admitting no log-concave fit, whereas the diagnostics guide correctly explains that a finite sample from a heavy-tailed population may still have a log-concave fitted approximation.

**Follow-up:** align callable help with the actual numerical and statistical contracts. Keep global shape verdict, grid diagnostic, optimizer convergence, spectral accuracy, and population-model adequacy distinct.

### R21 — Ordinary build instructions omit the new C++ compiler requirement [A, confirmed source and build]

**Locations:** `README.md`, `docs/getting-started.md`, `docs/development.md`; `_model/_series_kernels.pyx`; `docs/local-series.md`.

The new series extension is compiled as C++ and uses `_series_number.hpp`. The specialized series guide says a C++ compiler is required, but the README/getting-started instructions mention only a C compiler and Python headers. The production build did invoke G++.

**Follow-up:** update the main source-build prerequisites and generated-artifact guidance. Verify standard/platform settings in the supported wheel environments before making new portability claims.

### R22 — Branch documentation sends readers back to the old main [A, confirmed source]

**Locations:** chiefly README documentation links; package documentation/changelog URLs in `pyproject.toml`.

Found 21 Markdown links hardcoded to `github.com/DanJSal/gibbus/blob/main/...`. On a development branch, readers following those links can see the old implementation's docs. This undermines the purpose of publishing a reviewable replacement branch. Standard repository/homepage/issue URLs are a separate concern.

**Follow-up:** use relative repository-documentation links where appropriate and decide how installed development package metadata should route to the matching docs.

### R23 — Recent Python additions need a coherent formatting pass [B, measured]

**Locations:** especially `_fit/_empirical_tail_likelihood.py`, `tail_estimation.py`, `series_search.py`, `_model/basis_series.py`, recent API edits, and associated tests/benchmarks.

Black's read-only check reported 39 files requiring formatting. Dense one-line statements and unspaced expressions are especially concentrated in the newest numerical/experimental code, making review harder. Explicit Ruff `F` checks found nine unused imports in test/helper files. Broader lint flags include import ordering, stylistic constructors, and closures in benchmarks; those need judgment, not automatic classification as bugs. No undefined-name defect was reported by the explicit `F` scan.

**Follow-up:** apply the repository's chosen style without changing mathematics or numerical tolerances; pin/document lint rule scope so acceptance is reproducible. No formatting was applied during this run.

### R24 — Generated C++ files and development leftovers need repository hygiene [B, confirmed source]

**Locations:** `.gitignore`; root `runtime-update.patch`; `CHANGELOG.md`; module docstrings referring to removed G1 entry points.

`.gitignore` covers generated `.c`/`.h` but not the new generated `_series_kernels.cpp`. `MANIFEST.in` does exclude `.cpp` from sdists, but a local `git add` can still stage that generated file. The root patch duplicates part of the implementation history and should have an explicit keep/archive/remove decision. Three newer series-change bullets sit after the 0.1.0 release notes rather than under an appropriate unreleased heading. Several source comments still refer to removed owners or a `_Component.fit` entry point; the production `_empirical_tail_likelihood.py` header also still says no production source is modified; some are useful migration history, others now misdescribe the live pipeline.

**Follow-up:** ignore generated `.cpp`, keep the authored `.hpp`, clean the changelog structure, and curate development artifacts/comments. Do not delete historical evidence without first retaining what is still useful.

### R25 — Source packages omit evidence files described by included docs [C, confirmed package inspection]

**Locations:** `MANIFEST.in`; benchmark/series/runtime Markdown under `docs`.

The sdist includes Markdown documentation, but its `recursive-include docs *.md` policy excludes the JSON benchmarks and runtime `.log` records. The build inspection confirmed, for example, `local-series-benchmark.json` and `runtime-evidence/budget-final-tests.log` are absent. Some included Markdown directs readers to these records. The repository archive itself does contain them, so this is not missing implementation source.

**Follow-up:** decide whether sdists should ship the referenced evidence or docs should direct readers to a version-matched external location. This is not a reason to block a branch upload.

### R26 — Degree/span-selection concepts still need the explanation requested in issue #15 [C, confirmed documentation gap]

**Locations:** `docs/modeling-concepts.md` “Bulk degree”; fitting/span-selection guides; GitHub issue #15.

The fitter generalizes omitted-statistic testing to larger basis spans, and the guide mentions that test. The conceptual page still mostly calls degree a flexibility parameter; it does not clearly explain a nested statistical sieve for fixed structure or distinguish that omitted-information decision from BIC-based component-count selection. Explain this in the actual general-basis setting, including where changing fixed geometry/tail structure affects nesting. No duplicate old degree-selection implementation is needed.

### R27 — Architecture and numerical consolidation remain maintenance work, not a missing backend [C, source assessment]

**Locations:** `_api/distribution.py`, `fitting.py`, component/views; `_fit/inputs.py`, `basis_selection.py`, `basis_mixture.py`; issues #8 and #14.

The newer tree already splits orchestration, diagnostics, mixture analytics, basis construction, optimization, and post-fit state. Public dispatch/forwarding remains substantial, and fitting can repeat normalization/validation through the controller and lower fitters despite the stated validate-once convention. Importing the public estimator still eagerly loads fitting/statistical/compiled modules. Dedicated internal numerics coexist with SciPy `brentq`, quadrature, NNLS, special functions, and optimization.

**Follow-up:** profile import/validation overhead and consolidate only where it clarifies ownership or improves measured behavior. These observations do not justify reinstating the old solver or indiscriminately replacing mature SciPy routines. Shared read-only fitted payloads, detached diagnostic copies, and model-owned presentation remain useful foundations.

### R28 — Decide and document how an uncertified fit is exposed [C, confirmed existing policy; no new failing public sample claimed]

**Locations:** `_fit/basis_selection.py::_fit_data`; `_api/fitting.py::_run_basis_request`; numerical/diagnostic guides.

These paths intentionally allow `_UNCERTIFIED` results to proceed, with diagnostics showing lack of convergence/shape evidence. Automatic-tail recovery is stricter and retries when the primary status is not in `_CONVERGED`. The numerical-methods guide acknowledges `uncertified`, so this is not presented as an undiscovered guaranteed-shape violation. However, README wording can be read as promising every returned fitted component has established global convexity.

**Follow-up:** settle the user-facing policy: explicit warning/opt-in/exception versus documented degraded result. If retaining the current behavior, make the required diagnostic check prominent and consistent. No policy change was made or assumed in this review.

### R29 — Remaining release/platform and benchmark work can be deferred explicitly [C, scope decision]

**Locations:** declared Python/platform matrix, existing Stage 6 xfails/limitations, GitHub issues #2, #11–14.

This review establishes only one production Linux/Python 3.12 build and its reported test results. It does not establish Windows/macOS behavior, bounds-checked builds, Python 3.10–3.15 support, minimum dependency compatibility, or all extreme-tail cases. The repository still declares 3.10–3.14; dropping 3.10 and adding 3.15 remain separate open decisions/work items. The validation/case-generator infrastructure is substantial, but independent accuracy oracles for sums, external-method comparisons, comprehensive scaling benchmarks, and optional per-fit parallelism remain research/engineering work. Existing numerical xfails and mixture cost are legitimate development-branch limitations if stated accurately.

**Follow-up:** keep these as named follow-up items with explicit verification status. They need not all be solved before an unreleased branch is uploaded.

## GitHub issue disposition for the redesigned library

No GitHub issue was edited or closed. This table is a proposed interpretation, not an action taken.

| Issue | How it applies now |
|---|---|
| #17 — compact endpoint backend | The proposed design is abandoned by instruction. Its implementation plan is obsolete; do not use it to demand features or compatibility. |
| #8 — API boundaries | Still relevant as maintenance; R06–R08 expose concrete integration work and R27 covers architecture. |
| #9 — fit-statistics interface | Still a design follow-up. Existing diagnostics expose parameter counts, convergence and some likelihood metadata, but there is no comprehensive unified public fit-statistics object. Decide meaningful weighted/censored/derived-model semantics before expanding it. Not an upload prerequisite. |
| #10 — durable serialization | Separation of model/provenance/cache and rejection of old formats are substantially implemented. Remaining review items are R13–R15, not restoration of v1. |
| #15 — sieve documentation | R26; explain the generalized current method. |
| #2 — statistical validation/benchmarks | Much of the scaffolding exists. Current failures and coverage limitations remain; R16/R29. |
| #14 — numerical kernels | R27; retain external routines when their sophistication/maintenance value justifies them. |
| #13 — per-fit parallelism | Deferred benchmark-driven work. Test-run parallel workers are not per-fit parallelism. |
| #11 — drop Python 3.10 | Not implemented in the current metadata/matrix; handle as a coordinated support-policy update. |
| #12 — add Python 3.15 | Not implemented in the current metadata/matrix; verify actual build/dependency/platform support when doing it. |

## Suggested implementation sequence

1. Repair R01–R03 so the proposed branch can verify itself. Include R04 while reconciling the build inventory.
2. Repair the silent/request-changing API defects R05–R08 with defining-behavior regressions.
3. Give the fresh failures in R16 explicit dispositions. Fix proven correctness defects; keep genuinely deferred numerical limitations clearly tracked. Do not erase evidence through arbitrary tolerances.
4. Synchronize documentation around the actual snapshot: R14, R17–R22, plus the weighting and uncertainty contracts.
5. Clean formatting and repository packaging/hygiene, R23–R25, without changing numerical behavior incidentally.
6. Continue R09–R13/R15 hardening and the C-priority design/release work as separately scoped tasks. Do not expand the branch-readiness bar into a full release program.

## Useful foundations retained in the new implementation

- The original broad public distribution/query surface remains present; this is not a wholesale loss of post-fit functionality.
- `BasisSpec`/tail configuration has validation, explicit units, and independent endpoint choices; automatic parameter estimation does not silently compare unrelated tail families.
- The scalar-row solver and global numerical separator are intentional current architecture. Optional Taylor candidate search remains off by default and does not replace the full acceptance check.
- Portable state is non-object NumPy data with a mathematical model separated from optional provenance/cache. Current-format ordinary and cacheless round trips were exercised successfully in the focused tests/probes.
- Failed/expired fit requests install results atomically; the shared timeout exception is outside the recoverable numerical-failure tuple, and focused budget tests passed.
- Public diagnostics are copied, weights are detached/read-only internally, and component ownership/presentation restrictions are explicit.
- The case generator, isolated runner, historical benchmark notes, numerical stress tests, and independent mpmath/analytic checks are meaningful assets to retain and consolidate.

## Observations deliberately not classified as defects

- No restoration of the abandoned compact-coordinate proposal, old SOS backend, contact Gram blocks, or v1 persistence support is recommended.
- GCC emitted three maybe-uninitialized warnings around the interval normalizer's reuse table (`kw`, `z_s`, `budget_mass`). Inspection found `_reuse_build` initializes these before reuse; zero-row input skips the row loop, and unsuccessful setup returns before it. They are recorded as compiler warnings, not demonstrated uninitialized reads. A debug/sanitizer pass can confirm them later.
- The initial missing-`clang` build failure was an audit-runtime compiler setting; explicitly selecting available GCC/G++ resolved it.
- Same-version development cache identity, uncertified-result policy, concentrated-weight inference, and empirical tail parameter quality need explicit decisions/validation; they are not claimed to be proven wrong-density outputs in this run.
- The old online version was a reference for interface/completeness and issue context. It was not rebuilt as a numerical baseline, and historical comments about a failure being “pre-existing” were not promoted to an independently proven introduction history.

## Broad-run results

The 600-second broad-run watchdog ended the process with exit code 124. The log emitted **5,977 progress indicators: 5,966 dots, 7 failures, 2 skips, and 2 expected failures**, out of 6,015 collected tests. These are counts of the captured progress stream, **not a final pytest summary**. Thirty-eight collected tests had no completed outcome in that stream. The run does not establish that those remaining tests pass or that there are only seven failures in the complete suite.

Subsequent runs isolated **seven distinct failing tests**:

| Reproduction | Fresh failure and interpretation |
|---|---|
| `tests/test_adversarial_numerics.py::test_point_fit_is_invariant_to_extreme_global_weight_scale` | Maximum CDF difference `1.74698034e-11` against absolute tolerance `2e-13`, after globally rescaling relative weights. This is a very small discrepancy; investigate the intended invariance/accuracy target rather than label it a materially wrong distribution without analysis. |
| `tests/test_basis_mixture.py::test_stacked_m_step_without_shared_columns_is_independent_fits` | Coupled solve status `iteration_limit`, expected `converged`. |
| `tests/test_basis_tail_matrix.py::test_tail_class_fits_answer_queries_and_round_trip[laplace-exp1-lower-mixture]` | Public two-component fit raises `RuntimeError: the 2-component G3 mixture did not converge (em_only)`. This is an explicit lower `Exponential(1.0)` tail on shifted Laplace mixture data, not automatic family selection. |
| `tests/test_selection_battery.py::TestHeavyTailRobustness::test_every_k_fits[2]` | Degree-6, half-line lognormal sample, K=2: public fitter raises `RuntimeError`, status `iteration_limit`. |
| `tests/test_selection_battery.py::TestHeavyTailRobustness::test_every_k_fits[3]` | Same deterministic sample/request, K=3: `iteration_limit`. |
| `tests/test_selection_battery.py::TestHeavyTailRobustness::test_every_k_fits[4]` | Same deterministic sample/request, K=4: `iteration_limit`. |
| `tests/test_stage6_release_blockers.py::test_degree3_two_linear_tails_use_exact_curvature_face[1]` | Equivalent degree-3/degree-2 linear-tail spans differ in log density by up to `6.18759089e-6`, outside the test's `1e-6` relative/absolute thresholds. Both fits report convergence and nonnegative shape. |

The public-request/control-flow bugs R05–R08 were found by separate focused probes; passing existing tests does not cover them.

Additional run accounting, with overlaps (do not sum these as independent test coverage):

- The completed focused run remains **85 passed, 2 failed, 1 xfailed**, 27.34 seconds.
- A narrower six-module rerun with a live result ledger ended at its 240-second watchdog and printed **126 passed, 3 failed**, 240.43 seconds, with `KeyboardInterrupt`. It covered adversarial numerics, API, auto-K, basis mixture, selection battery, and validation; it excluded the already-reproduced coupled-M-step case. This is an interrupted run, not a complete six-module pass/fail count.
- A serial post-fit/tail-matrix run emitted **81 pass indicators and one failure** before the 150-second watchdog and subsequent forced termination. Native work did not yield a final traceback in time. Its failed node was then independently selected and reproduced with a complete traceback; see the lower-exponential-tail row above. The incomplete run is not counted as a successful group.
- The final two-case isolation run (that tail-matrix case and the K=4 lognormal case) completed: **2 failed in 60.99 seconds**.
- An additional serial lognormal subset was stopped at 100 seconds: **1 passed, 1 failed** (K=2), with later cases incomplete in that run. The K=3/K=4 results above come from the other named reruns.

These results explain why the branch should retain a clear failure register rather than copy an older green-suite claim. They do not identify the first introducing commit of every failure, and no assertion, solver budget, or numerical tolerance was modified to make a test pass.


## Reproduction notes

Commands used against an isolated copy of the supplied source:

```sh
CC=gcc CXX=g++ python setup.py build_ext --inplace
python -m build --sdist --no-isolation
python setup.py build_py
python setup.py bdist_wheel --skip-build
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q -n 2 --dist worksteal
```

The wheel reused the completed production extension build; it was not an additional isolated PEP 517 wheel compilation or a rebuild from the extracted sdist. The sdist's source completeness was inspected directly. The actual broad command had a 600-second process watchdog; the final disposition is recorded above.

Focused test selection:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q \
  tests/test_fit_budget.py \
  tests/test_empirical_tail_integration.py \
  tests/test_serialization.py \
  tests/test_stage6_solver_triage.py \
  tests/test_stage6_release_blockers.py \
  tests/test_basis_curvature_rows.py \
  tests/test_basis_mixture.py::test_stacked_m_step_without_shared_columns_is_independent_fits
```

Minimal public reproductions for R05/R07:

```python
import numpy as np
from gibbus import BasisSpec, Distribution, Polynomial

x = np.r_[np.random.default_rng(9).normal(-3, .5, 100),
          np.random.default_rng(10).normal(3, .5, 100)]
d = Distribution().fit(
    x, n_components=2, basis=BasisSpec(bulk_degree=2),
    component_options=[{'poly_degree': 4}, {'poly_degree': 4}],
)
print([int(c.data['bulk_degree']) for c in d.components])  # [2, 2]

sample = np.random.default_rng(31).normal(size=160)
spec = BasisSpec(bulk_degree=2, upper_tail=Polynomial(2), expansion='minimal')
model = Distribution().fit(sample, basis=spec)
model.bootstrap_bands(sample, [0.], n_resamples=1, fit_kwargs={'basis': spec})
# ValueError: basis replaces ...; got support
```

The review log is the deliverable. Temporary probe scripts, builds, and environments were not applied to either uploaded archive or to GitHub.
