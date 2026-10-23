# Generic fitting-coordinate refactor — October 23, 2026

## Implementation

Point, interval and coupled-mixture objectives use one coordinate protocol and
one compiled Newton lifecycle. Complete polynomial prefixes use Chebyshev
coordinates on finite supports, generalized Laguerre on half-lines, and Hermite
on the full line. Incomplete/coupled/shared columns retain their exact canonical
span; gates and shared equalities are preserved. Direct recurrences evaluate
values and four derivatives; empirical statistics and covariance are rebuilt
in those coordinates.

The native interior-point subproblem may be refined by a scalar projection only
when its original-metric bound is better and resolved. Family-specific solver
and endpoint-cut scheduling are removed. Fully separated rejected-endpoint cuts
replace approximate gate cuts. Exact contact resolution and Hessian-scaled
active-row accounting are shared. An unsuccessful numerical representation can
retry canonical coordinates through the same controller, with a common deadline
and accumulated completed-attempt counters.

Conversion reconstructs the canonical face and rechecks likelihood and shape.
Stationarity polish uses the current moving-contact geometry: eliminating its
location adds the multiplier times the third-derivative outer product divided
by the fourth derivative to the likelihood Hessian. It retains full shape,
likelihood and bound checks, with no model-family dispatch. Degenerate contacts
retain the incumbent.

Format-2 persistence and public statistical families are unchanged. Interval
rows retain their observed widths. No numerical assertion, shape threshold or
solver tolerance value was relaxed, and no failing test was converted to xfail.

See [architecture](../fitting-coordinates.md) and the accompanying patch.

## Verification

- Complete source sweep: **6,104 passed, 1 regression, 2 skipped, 1 expected failure**, out of 6,108 cases in 746.335 seconds. The final guard fixes the bounded warm-start regression; its affected-case rerun passes **28 tests** in 89.569 seconds.
- Reconciled unique source cases: **6,105 passed, 2 skipped, 1 expected failure**, with no unresolved failure. This combines the full sweep and final affected-case rerun; it is not a fresh all-green full invocation after the final guard.
- Installed wheel outside the source checkout: **182 passed** in 98.579 seconds; all 14 native extensions imported. The broad wheel batch emitted four documented SoftPower conversion warnings and one invalid-arithmetic warning in optional contact inspection. A final finite-derivative guard removes the latter; the affected serialization check passes again with runtime warnings treated as errors in both source and installed wheel.
- Both degree-12 likelihood/acceptance regressions pass with existing numerical assertions. The degree-4 two-linear-tail Laplace test retains its strict expected-failure marker.
- Black and configured Ruff checks pass. Source hashes identify the delivered source verified by the sweep plus final targeted reruns; 81 Python/native package files match the installed wheel byte for byte.

Raw source and installed XML/logs, targeted recovery results, style checks,
source hashes and wheel/source agreement accompany this report. The first full refactor sweep had 6,104 passes, one unit-scaling regression,
two skips and the existing expected failure. The moving-contact correction
resolved that regression; 27 focused cases passed. The next full sweep
exposed a bounded warm-start change from polishing an approximate lower-degree
fit. Restricting polish to improvements within the configured Newton tolerance
preserved that warm start; all 28 affected checks then passed. A final
finite-derivative guard skips singular endpoint rows before contact inspection;
the affected serialization case passes with runtime warnings treated as errors
in both source and installed wheel. Intermediate XML is retained separately. Earlier interrupted diagnostic
runs are not included in the final source result.
Timings are acceptance-run observations, not controlled performance benchmarks.

## Limits

The smooth basis has not gained `|x|`, nonsmooth knots or fractional-power bulk
terms. The remaining degree-4 linear-tail Laplace failure is not diagnosed solely
as a conditioning defect; the smooth model approximation and contact geometry
remain relevant. Higher-degree, extreme-coordinate and mixture convergence are
not universally guaranteed. Only Linux x86-64/CPython 3.12 was verified here;
current debug and other platform/interpreter matrices remain open.

No repository branch, pull request or release was created. The development
version remains `0.2.0.dev0`.
