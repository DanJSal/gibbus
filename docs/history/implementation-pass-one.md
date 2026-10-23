> Historical first implementation pass; current verification is in
> [the validation record](../review-validation.md).

# Implementation-pass validation

This records the October 22, 2026 development checkpoint. The uploaded source
archives and GitHub branches/issues were not modified. The working source was
copied from `gibbus_latest.zip`; the older online source and its abandoned
endpoint-design proposal did not supply the new architecture.

## Environment and build

Linux x86-64, CPython 3.12.14, GCC/G++ 13.3, NumPy 2.3.5, SciPy 1.17.0,
Cython 3.3.0. All 14 production extensions compiled successfully at `-O3`:

```sh
CC=gcc CXX=g++ python setup.py build_ext --inplace
python -m build --sdist --no-isolation
python setup.py build_py
python setup.py bdist_wheel --skip-build
```

The wheel reused those production extension objects. It was installed into a
separate environment, and smoke/tests ran outside the source directory. This
is not an independent isolated wheel build or an installation compiled from
the source distribution. The declared cross-platform CI matrix remains to run.

## Focused checks

- The final source batch covering the new regressions, empirical tails, fitting
  budgets and case generation passed **172 tests**.
- Installed-wheel smoke imported all 14 native extensions. The new regression
  module plus serialization suite passed **56 tests** against the installed
  package. The final source cleanup replaced three equivalent lambdas with local
  functions and renamed an ambiguous test variable; both the source batch and
  the rebuilt installed wheel cover that cleanup.
- Black, configured Ruff correctness/import checks and `git diff --check` pass.
- Repository documentation file-link targets resolve outside the explicitly
  abandoned design material. Historical content is marked as historical.
- Source-distribution inspection found the new regression tests, authored C++
  headers, and referenced Markdown/JSON/log/patch evidence.

## Numerical checks and limits

The focused mixture/Stage 6 batch produced **39 passed, 2 failed, 1 xfailed**.
The lower-exponential-tail Laplace-mixture test now passes after the equal-
likelihood polish rounding fix. The remaining two failures are retained in
R16; neither assertions nor solver/shape tolerances were loosened.

The expanded run additionally exposed high-degree likelihood, expansion-history
and negative boundary-amplitude failures. Each reproduced in the pristine
latest-source build under the same environment (two-case baseline run: 2 failed;
separate boundary-amplitude baseline run: 1 failed). This comparison establishes
they predate this implementation pass, not their original introduction history.

The full suite collected **6,036 tests** and ran with two pytest workers under
an explicit **900-second watchdog** (exit 124). Captured controller outcomes:
**5,981 passed, 7 failed, 2 skipped, 2 xfailed**; **44 tests had no outcome** at
the cutoff. This is a bounded partial run, not a complete suite pass. The seven
failures comprise global weight invariance, generic stacked M-step convergence,
degree-12 lognormal convergence, expansion-history semantics, negative boundary
amplitude, degree-3 equivalent-span accuracy and the k=2 heavy-tail mixture.
The k=3/k=4 heavy-tail cases remain baseline findings without current-run
outcomes. Expected failures remain strict; no assertions were weakened.

The controller outcome log, reported failures and unreported node IDs are
retained in `review-evidence/full-suite-results.json`, `full-suite-failures.log`
and `full-suite-unreported.log`. Raw progress output can lag controller outcomes
because of output buffering; the controller records determine the counts.


See [the follow-up register](../review-follow-up.md) for the individual cases,
open issue mapping and deliberately deferred work. A green release matrix,
full debug build, minimum dependency validation, comparative statistical
benchmarks, and a comprehensive solver investigation have not been completed.

Focused test, wheel smoke, style, baseline-reproduction and polish-trace logs
are retained in [`review-evidence/`](../review-evidence/). The full-suite outcome
is included there after the bounded run.
