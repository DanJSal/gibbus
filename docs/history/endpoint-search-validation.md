# Implementation validation

This records the October 22, 2026 development follow-up. The attached archives
and GitHub branches/issues were not modified. Source derives from
`gibbus_latest.zip`; the abandoned endpoint proposal remains outside scope.

## Environment and build

Linux x86-64, CPython 3.12.14, GCC/G++ 13.3, NumPy 2.3.5, SciPy 1.17.0 and
Cython 3.3.0. All 14 production extensions compiled at `-O3` in the first
implementation pass. This follow-up changes Python and tests. The rebuilt wheel
reuses those native objects and refreshes Python modules and metadata:

```sh
CC=gcc CXX=g++ python setup.py build_py bdist_wheel --skip-build
python -m build --sdist --no-isolation
```

This is not an independent isolated wheel compilation or an install built from
the sdist. Tests ran outside the source tree against the installed wheel; the
pytest runner came from the test environment, with the wheel installation first
on `PYTHONPATH`. Smoke independently imported the installed package and all 14
extensions using the wheel environment's interpreter.

## Completed full-suite runs

| Checkpoint | Collection | Passed | Failed | Skipped | Xfailed | Unfinished |
|---|---:|---:|---:|---:|---:|---:|
| Previous solver checkpoint | 6,038 | 6,026 | 8 | 2 | 2 | 0 |
| Unconditional endpoint-search follow-up | 6,060 | 6,052 | 4 | 2 | 2 | 0 |
| Final scoped endpoint-search follow-up | 6,060 | 6,055 | 1 | 2 | 2 | 0 |

The final run completed in **420.459 seconds**, exit 1. Its sole failure is
`test_reported_nll_matches_high_degree_lognormal`, a retained degree-12 half-line
convergence case. Both strict expected failures remain explicit and separate.
The 22 additional cases relative to the previous solver checkpoint comprise the
21 record-validation/round-trip cases and one component-order permutation test.

The completed JUnit XML and parsed per-case results are authoritative, retained
under `review-evidence/solver-followup-full-final.xml` and
`solver-followup-full-final-results.json`. The intermediate follow-up run and
its four failure traces are also retained, not overwritten. Earlier validation
is preserved in [the second implementation checkpoint](implementation-pass-two.md).

The two follow-up full runs used one BLAS/OpenMP thread per process. The final
run used eight pytest workers:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -v -n 8 tests \
  --durations=20 --junitxml=full-final.xml
```

Initial simultaneous diagnostics inherited nine OpenBLAS threads each. Some
were interrupted or timed out before producing complete outcomes. They do not
establish failures, successes or comparative performance. Production thread
settings were not altered.

## Focused and installed checks

- Initial broader source follow-up: **119 passed** before scoping extra searches
  for single-polynomial fits. The batch covers mixtures, adversarial numerics,
  basis selection, release-blocker cases and review regressions.
- Final scheduling regression batch: **9 passed**. It covers all three degree-8
  scale cases, the automatic fallback for unsupported component requests, the
  stacked-fit comparison, component-order permutation, and three equivalent-span
  seeds. Existing assertions are unchanged.
- Final installed-wheel batch: **95 passed**. It covers serialization,
  review regressions, adversarial numerics and the nine scheduling cases.
  Four warnings are the documented integer SoftPower-to-polynomial conversion.
- Installed-wheel smoke imports **all 14 extensions**. Black leaves 136 files
  unchanged; configured Ruff checks, diff whitespace checks and relative
  documentation-file targets outside the abandoned design pass.
- The sdist includes tests and referenced XML/JSON/log/patch evidence. The source
  ZIP excludes compiled extensions, generated C++/assembly, caches, build outputs
  and git metadata, and retains authored headers.

## Numerical interpretation

The fast native shape gate can stop at the first violating component and use
valleys in the incumbent's domain. For coupled fits and bases with gated or
non-monomial columns, the coordinator now resolves every component at the
rejected endpoint before the next exchange. Valid point/tail cuts retain their
originating-face scope. Non-convex endpoints are never normalized or accepted.

This repairs all seven previously failing stacked, span-equivalence and heavy-tail
acceptance cases. Unconditionally applying the extra search also regressed two
unit-scale tests and an automatic-selection screen. Same-environment comparison
reproduced those regressions and passed the affected cases with the established
single-polynomial schedule. The retained policy preserves that schedule while
completing endpoint discovery for coupled/general bases.

No likelihood/shape/convergence tolerance or numerical assertion was weakened.
The earlier exact endpoint-log domain correction, guarded interior stationarity
polish and corrected expansion-history fixture remain in place. Temporary
whitening, regularized recovery, violated-cut filtering and retained-cut-cache
probes did not establish a reliable general conditioning fix and are excluded.
The [investigation log](../review-evidence/solver-followup-investigation.md) records
those observations and incomplete diagnostic scopes.

## Debug and remaining validation

The earlier separate debug-safety copy compiled all 14 extensions at `-O0 -g0`
and passed 34 review/kernel plus 39 solver tests. It predates the current Python
cut-scheduling follow-up. Runtime `GIBBUS_DEBUG`/`GIBBUS_DEBUG_STRICT` fallback
escalation was not enabled. The earlier optimized debug attempt failed with
truncated assembly and missing `.cfi_endproc`; its cause is unresolved.

Other interpreters/platforms, minimum dependency combinations, optimized debug,
the current full debug suite, comparative accuracy/scaling benchmarks and the
remaining solver conditioning work are open. CPython 3.10–3.14 support policy is
unchanged; neither dropping 3.10 nor supporting 3.15 was asserted. The full
finding/issue register remains in [review follow-up](../review-follow-up.md).
