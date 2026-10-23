# Solver follow-up investigation

The retained change completes curvature-cut discovery at the first rejected
Newton endpoint for coupled fits and general columns. Single ungated monomial
fits retain their established schedule after the unconditional variant regressed
affine-unit precision and a component-selection screen. The fast native gate remains a rejection shortcut: it stops
at the first violating component and can use valleys inside the incumbent's
search domain. Before the next exchange, the Python coordinator now searches
that endpoint over every component, retaining the refined point and tail cuts
with the same originating-face scope as other cuts. Non-convex endpoints are
still never normalized or accepted as fitted iterates.

The pre-existing stacked-M-step equivalence test, all three degree-3/degree-2
linear-tail equivalence seeds, and a new component-order permutation regression
pass (five cases). The broader 119-case source batch also passes. All existing
likelihood, convergence, shape and pointwise accuracy assertions remain intact.
No test is converted to an expected failure.

## Conditioning probes excluded from the source

Degree-12 half-line lognormal fits still require separate conditioning work.
Earlier raw Hessians reached condition estimates near 1e29; diagonal-normalized
examples have condition estimates around 1e14 to 1e16. Temporary probes tested
full and bounded whitening of Newton-run coordinates and regularized single-step
recovery followed by the original objective. They did not establish a converged
fit within their bounded runs. Some transformed subproblems produced much worse
bounds (including about 1e12). Regularized steps made limited likelihood progress
but supplied no replacement convergence certificate. None of these probes is
included in the implementation or used to accept a fit.

## Execution observations

The initial simultaneous diagnostics each inherited nine OpenBLAS threads.
Several were interrupted or reached external time limits without a final test
outcome. Their partial progress is not a completed failure or success, and their
timings are not comparative benchmarks. Subsequent bounded diagnostics and the
full suite use OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 to prevent worker
oversubscription. Production thread settings and numerical acceptance policies
are unchanged. Completed JUnit XML is authoritative for the final counts.

## Cut-scheduling regression comparison

The intermediate full run completed 6,060 cases: 6,052 passed, four failed,
two skipped and two expected failures. All seven formerly failing stacked,
span-equivalence and heavy-tail acceptance cases passed. The four failures were
the existing degree-12 lognormal case and three regressions: degree-8 unit-scale
seeds 2 and 3, and the automatic fallback in the unsupported-component-count
case. Restricting retained endpoint cuts to violated parts did not repair those
three regressions. Disabling the extra endpoint search passed all four relevant
cases in the same single-BLAS-thread environment. The retained implementation
therefore enables the extra search only for coupled or general-column bases;
its nine focused scheduling regressions pass.

A retained-point-cut cache was also tested in a temporary large-mixture probe.
Some early M-steps converged, but later steps still reached iteration limits or
had an uncertified shape verdict and the request exhausted its 45-second budget.
That experiment is not part of the source.
