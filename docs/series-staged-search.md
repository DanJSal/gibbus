# Staged series candidate search

The Newton candidate search uses cached actual basis derivatives at fixed scan
points, selectively combines Taylor coefficients for refinement, and tracks
multiple minima between trial vectors. All per-trial work remains allocation-free
`noexcept nogil` Cython. The independent q'', q''', and q'''' coefficient
construction remains the series engine's original derivative-specific algebra.

The internal option remains `_CurvatureOptions(series_candidates=True)` and is
still off by default. [The integration guide](series-search-experiment.md)
describes the solver entry point and full acceptance behavior.

## Numeric tables and trial work

At construction, each panel stores basis-column Taylor coefficients
`T[derivative, power, column]`. It also stores actual basis derivative rows
`B[derivative, scan point, column]` for derivative orders two and three, evaluated
by the existing analytic basis evaluator. With fixed geometry these rows are
independent of theta. A trial scans by compensated dot products `B @ theta`;
it does not run polynomial evaluators or evaluate q'''' at every scan point.

The scan detects negative-to-positive q''' crossings and the lowest sampled q''
value. Panels with neither a sampled crossing nor a tracked minimum require no
Taylor contraction. Other panels first combine q'' coefficients and apply the
approximating polynomial bound `C0 - sum(abs(Cn), n>=1)` on |u| <= 1. If the panel
remains relevant, q''' coefficients are combined for root refinement; q''''
coefficients are combined when Newton steps or minimum classification need them.
Each derivative is combined at most once per panel per trial, with a single-panel
scratch coefficient buffer. The bound describes the polynomial only and never
provides a feasibility certificate.

Bracketed roots use safeguarded Newton/bisection. With u=(x-center)/scale, the
Newton step is `u -= q''' / (scale*q'''')`; the derivative rows are already in
physical units. Up to eight prior minimum locations per panel provide independent
warm starts, each with up to eight Newton probes. Minima are deduplicated at the
requested positional tolerance, and a full cache retains the lowest scores.
A root can remain tracked even when its current value is above the cut threshold.

Refined minima take priority over sampled/probed points. A panel's best fallback
point is kept if refinement supplies no useful cut, or if its q'' value is lower
than the best refined value by more than a small score-relative comparison margin
(`1e-10 * (abs(root score) + abs(fallback score) + abs(candidate threshold))`).
This margin only controls redundant candidate selection; the original feasibility
and rejection tolerances still apply. Avoiding an extra sampled row beside every
refined root matters for exchange convergence: adding more valid cuts can still
make a complete fit slower. The actual basis is evaluated at every proposed cut
before it can reject a trial or supply a constraint row.

## Cache and range behavior

Actual scan points and derivative rows are immutable parts of `SeriesSearchData`
and are included in its retained byte count. The shared-table key includes scan
subinterval count as well as geometry, exact coefficient maps, order, panel budget
and domain. Changing refinement steps, tolerance, rho, theta, or output capacity
uses a private workspace on the matching immutable table. The 32-entry/32-MiB LRU
and ownership rules are described in [Tensor reuse](series-tensor-reuse.md).

Model-backed tables always use actual analytic basis rows. Synthetic callers of
`SeriesSearchData`/`SeriesSearchKernel` can omit scan rows, in which case scan rows
are precomputed from the supplied independent derivative tensors. The `exact_scan`
property distinguishes supplied basis rows from that polynomial fallback. A
kernel's sample count must match its retained grid.

Vanishing-boundary exclusions, fractional-power singularity radii, tail cutoffs,
and full curvature acceptance continue through the existing panel planner and
full search. Scan points are clipped to the planned interior endpoints, avoiding
endpoint roundoff into a logarithmic singularity. Nonfinite cached rows disable
the heuristic for that domain; a nonfinite trial contraction returns the existing
numerical-failure signal and uses the original direct search.

Diagnostics now include `scan_nodes`, `q2_builds`, `q3_builds`, `q4_builds`,
`warm_probes`, and `tracked_roots`, as well as candidate/refinement/replay counts.
No lock, allocation, Python call or transcendental evaluation occurs inside the
candidate C loop.

## Candidate speed and ordinary-case discovery

Linux x86-64, CPython 3.12, GCC/G++. Thirty changing vectors per family (180 total),
with seven timing sweeps. Timings below measure the public candidate wrapper,
including its validation/output allocation; construction and reference search are
outside the timer. Vectors are slightly rescaled between sweeps to avoid exact
replay. The previous kernel is loaded independently from the prior built source.

References use a 4,097-point q''' grid followed by Brent refinement of each
negative-to-positive crossing. Only minima below `-1e-4/h^2` count, and a returned
point must have actual negative curvature and be within `1e-3*h` to count as a
match. This dense reference can itself miss sufficiently narrow root pairs; it
is a comparison tool, not an exhaustive mathematical certificate.

| Basis family | Previous (microseconds) | Staged (microseconds) | Speedup | Negative minima found |
|---|---:|---:|---:|---:|
| Full-line cubic bulk | 14.00 | 9.41 | 1.49x | 79/79 |
| Half-line quadratic + log | 16.56 | 8.60 | 1.93x | 37/37 |
| Bounded quartic + two logs | 18.54 | 9.15 | 2.03x | 28/28 |
| Asymmetric soft-power tails | 18.46 | 13.24 | 1.39x | 65/65 |
| Asymmetric exponential tails | 30.95 | 17.71 | 1.75x | 41/41 |
| Upper half-line soft-power tail | 12.63 | 7.53 | 1.68x | 47/47 |

Both kernels found all 297 reference minima and supplied a negative candidate
on every trial having a negative reference minimum. The largest nearest-candidate
position error among those minima was about `6.5e-6*h` for both versions, below
the default `1e-5*h` refinement tolerance. This ordinary-case sample supports
preserved discovery quality rather than a larger discovery rate.

## Clustered minima that move between trials

A separate degree-six polynomial-curvature sequence has three known negative
minima. They move from separated scan intervals into one interval, ending at
0.085, 0.125, and 0.165 in a panel spanning [-1, 1] with eight scan subintervals.
A fixed scan alone cannot distinguish all stationary points in that cell. Warm
tracking uses the earlier separated minima to follow them.

Across eight vectors, the previous kernel found 15 of 24 minimum instances;
the staged kernel found all 24. The comparison counts a known minimum when a
candidate is within one percent of the current inter-minimum spacing. Raw
positions and the complete coefficient-generating test are supplied. This is a
constructed continuity test, not evidence that fresh clustered minima are always
found. With no useful history, multiple stationary points inside one scan cell
can still be missed. Adaptive isolation would be a separate addition.

## Whole solver runs

Nine triplets per example, with direct-search controls and cold/retained series
runs. The previous and staged programs run separately with the same benchmark
inputs; small timing differences can reflect system noise. Each solver timer
includes search setup, private workspaces, objective/normalizer work, and the full
acceptance checks. Initial model/objective construction is outside the timer.

| Case | Previous cold (ms) | Staged cold (ms) | Previous retained (ms) | Staged retained (ms) |
|---|---:|---:|---:|---:|
| line-b2 | 4.169 | 3.952 | 3.484 | 3.295 |
| line-b4-gumbel | 32.261 | 33.706 | 31.338 | 30.559 |
| half-b2-log | 9.871 | 9.954 | 8.917 | 9.346 |
| student-softpower | 19.411 | 19.821 | 17.181 | 17.475 |

Whole-fit times are roughly similar despite the faster candidate calls: most
remaining time is elsewhere in these solver runs. The staged selection keeps
the previous Newton iteration counts in these four examples. All runs converge
with the existing nonnegative certificate, and cold/retained parameter vectors
match bit for bit within each implementation. The candidate changes can alter
parameters and the Newton trajectory between implementations; their objectives
remain close. Full raw results, direct controls and cache counts are supplied.
The sample does not support a general whole-fit speedup claim or changing the
default option.

## Validation and reproduction

92 targeted tests pass with bounds-checked search code and in release mode. New
coverage checks actual cached scan rows, no-series scan paths, selective higher
order contraction, sample fallback with a misleading polynomial, refined-cut
selection, scan-count cache invalidation, and multiple warm minima in one cell.
The selected existing fitting tests again pass 54 of 55; the same stacked-mixture
test reaches the iteration limit while expecting convergence. That pre-existing
issue remains unresolved.

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_series_search.py tests/test_basis_curvature_rows.py tests/test_basis_series.py -q
python tests/benchmark_series_candidates.py
python tests/benchmark_series_tensor_reuse.py
# Compare with a previously built source tree:
python tests/benchmark_series_candidates.py --baseline-source /path/to/previous/source
```

`series-staged-candidate-benchmark.json` contains discovery, position errors,
timings and work counters. `series-staged-fit-benchmark.json` contains both
programs' nine-triplet solver measurements. Rebuild extensions after applying
this patch: the internal search table layout and its Newton consumer changed.
