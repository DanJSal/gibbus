# Historical second implementation checkpoint

This is the validation record before the subsequent cut-exchange follow-up.
Current results are in [implementation validation](../review-validation.md).

# Implementation validation

This records the October 22, 2026 development checkpoint. The uploaded archives
and GitHub branches/issues were not modified. The working source derives from
`gibbus_latest.zip`; the abandoned endpoint proposal remains outside scope.

## Environment and production build

Linux x86-64, CPython 3.12.14, GCC/G++ 13.3, NumPy 2.3.5, SciPy 1.17.0 and
Cython 3.3.0. All 14 production extensions compiled successfully at `-O3` in
the first implementation pass. The continuation changes Python and tests;
the production native sources and compiler settings are unchanged.

```sh
CC=gcc CXX=g++ python setup.py build_ext --inplace
python setup.py build_py
python setup.py bdist_wheel --skip-build
python -m build --sdist --no-isolation
```

The wheel reuses the production extension objects and was installed into a
separate environment for checks outside the source directory. This is not an
independent isolated wheel compilation or a build installed from the sdist.

## Current source verification

- The first 50-test continuation batch covers exact log-domain polish, global
  weight-scale invariance, basis objectives/selection, and the deterministic
  untestable-span guard plus real finite-moment expansion stopping: **50 passed**.
- After coordinate-record consistency and missing-geometry rejection fixes, the
  boundary/serialization/coordinate/adversarial batch passed **114 tests**,
  including 15 model-record round trips across tail families and support types.
  Four warnings were the documented integer SoftPower-to-polynomial conversion.
- The previous 44 tests without outcomes were run separately before the final
  solver checkpoint: **40 passed, 4 failed** in 189 seconds. The four failures
  are separately named heavy-tail mixture cases in R16; this was not an all-green
  completion of that remainder.

The full solver-checkpoint run completed **6,038 tests in 826.938 seconds**:
**6,026 passed, 8 failed, 2 skipped and 2 xfailed** (exit 1). There are no
unfinished cases in that run. Its eight failures match the individually logged
remaining numerical cases under R16. The completed JUnit XML is authoritative:
the append/progress logs omitted three passing entries and must not be used to
reconstruct the final count. Both raw XML and a complete parsed summary are
retained in `../review-evidence/continuation-full-suite.xml` and
`continuation-full-suite-results.json`.

The final collection has **6,059 tests**. The 21 additional coordinate-record
rejection and tail-family record-round-trip cases were added after the full run
started. That independent validation change is covered by the 114-test source
batch and final installed-wheel checks; those 21 cases were not part of the
6,038-test solver-checkpoint collection.

The rebuilt installed wheel imported all 14 native extensions and passed
**86 tests**, including the final model-record cases, serialization and
adversarial numerics. Four warnings were the documented integer SoftPower
conversion. Black, configured Ruff checks and `git diff --check` pass; repository
documentation file targets resolve outside the abandoned design material.
The final sdist includes the regression tests and referenced XML/JSON/log/patch
evidence.

## Debug safety build

A separate copy rebuilt all 14 extensions with `GIBBUS_DEBUG_BUILD=1` at
`-O0 -g0`, enabling the central bounds/wraparound/None/initialization checks.
Its first-pass review-regression and compiled-kernel batch passed **34 tests**.
A second solver batch covering adversarial numerics, basis selection and the
boundary-amplitude regression passed **39 tests**. The native smoke imported all
14 extensions from that copy. Runtime
`GIBBUS_DEBUG`/`GIBBUS_DEBUG_STRICT` fallback escalation was not enabled.

The first `-O3` debug build failed while assembling `_basis_newton_kernels`,
with a truncated assembly file and a missing `.cfi_endproc` error. The cause
was not established. The debug copy alone then used `-O0` and rebuilt every
extension with `--force`; production build settings were not altered. This
is scoped debug evidence, not a completed optimized-debug or platform matrix.

## Numerical interpretation

The endpoint-log correction enforces the normalizer's existing exact domain.
The interior point polish reduces a resolved scaled gradient with at most three
well-conditioned Newton corrections. Every accepted correction retains exact
row feasibility, full numerical shape acceptance and the existing likelihood/
bound target. Failed corrections retain the incumbent. Weights are not rounded
or quantized, and no solver/shape tolerance was relaxed.

The expansion-history test's former data premise was incorrect: its retained
positive quadratic coefficient makes exponential moments finite after stronger
leaders collapse. Its failure handler now has a deterministic quadrature-failure
fixture checking exact retention of the baseline parameters, and real-data
finite-moment stopping is tested separately. No acceptance test was moved to
xfail.

Raw high-degree solver tracing and an equivalent-span precision probe are
retained as investigation evidence; they do not establish fixes. The open cases
and all dispositions are in [the follow-up register](../review-follow-up.md).

## Evidence and remaining validation

Current continuation logs use the `continuation-` prefix under
[`../review-evidence/`](../review-evidence/). Earlier logs remain separately identified;
[the first-pass record](implementation-pass-one.md) documents its bounded
run. Report counts overlap and must not be summed as independent coverage.

Other Python/platform combinations, the minimum dependency matrix, optimized
debug compilation, full debug-suite behavior, comparative statistical/scaling
benchmarks and the remaining solver investigations are open. The declared
CPython 3.10–3.14 support policy is unchanged; neither removal of 3.10 nor support
for 3.15 was asserted.
