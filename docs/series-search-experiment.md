# Cached-series candidate search in Newton descent

The optional series search proposes curvature constraint points during compiled
Newton trial evaluation. It uses fixed-geometry Taylor caches for q'', q''',
and q'''' and evaluates the actual basis at every proposed cut point.

Enable it at the internal solver boundary:

```python
from gibbus._fit.curvature_search import _CurvatureOptions
from gibbus._fit.basis_newton import _solve_basis

result = _solve_basis(
    objective,
    curvature_options=_CurvatureOptions(series_candidates=True),
)
```

The default is `False`: short/simple fits may not amortize construction, and
performance depends on the number and type of rejected trials. This is an
internal experiment, not a new public `Distribution.fit` argument.

## What runs where

`_SeriesCandidateCache` plans panels and constructs a coefficient tensor before
a gated Newton run. Each tensor slice maps the existing independent basis
coefficients directly to the three derivative series. This folds the exact
rational sign maps into fixed geometry once. The cache also stores actual q'' and q''' basis rows at the fixed scan points.
Changing theta requires compiled dot products for scanning, followed by selective
series contraction and local refinement. See [Staged candidate search](series-staged-search.md)
for the algorithm, discovery tests and current measurements.

`_series_search.pyx` performs all per-trial candidate work in `noexcept nogil`
functions, without allocation or Python calls. `BasisFaceObjective` owns the
cache and mutable search buffers for each model block. Workspace is per
objective and must not be used by concurrent calls on the same objective.
The existing fused Newton loop remains nogil throughout its trial gate.

A cache survives Newton iterations, backtracking trials, and exchange restarts
on the same objective. A smaller requested compact domain reuses a larger
cache: points in that larger region are still valid global curvature
constraints. An expanded domain prepares a new table or reuses a matching retained one.
Each model/face objective owns a separate mutable workspace. Matching models
share immutable finished tensors through a bounded process-local LRU; objectives
using the same model also share its immutable `model.series_plan`. See
[Tensor reuse across fits](series-tensor-reuse.md) for keys, bounds and timings.

## Candidate algorithm

1. Batch the singularity-radius checks for each subdivision level. Start with
   16 asinh-spaced intervals plus gate locations and the fitting
   center. Bisect until each half-width is at most 0.35 times the nearest
   structural complex singularity distance. Finite-log endpoints leave a
   strip of width `1e-8*h` to the full search when no wider exclusion applies.
2. Construct order-16 derivative series by default. Precombine these into a
   per-panel, per-basis-column tensor. Remove trailing powers that are exactly
   zero in every tensor row; polynomial models pay only their actual degree.
3. Scan by dotting cached actual basis rows for q'' and q''' with theta.
   Panels without a sampled minimum crossing or tracked minimum do not require
   Taylor coefficients. Keep a low sampled point when refinement provides no
   useful candidate, or when it is lower than the refined candidates.
4. On candidate panels, contract q'' coefficients and apply the polynomial bound
   `C0 - sum(abs(Cn), n>=1)`. If it permits further search, contract q''' for
   root refinement and q'''' when Newton steps or minimum classification need it.
   Track up to eight minima per panel with independent warm starts. Scans use
   eight subintervals by default; bracket refinements have up to 16 iterations
   and warm probes have up to eight.
5. Deduplicate nearby candidates and retain the lowest approximate values up
   to the existing candidate buffer capacity. Repeated useful theta vectors
   reuse their candidate locations, preventing refinement history alone from
   manufacturing new rows on a repeated violating trial.
6. Evaluate `Phi''(x)` directly at each proposed point. Only representable,
   nonzero rows with an actual violation reject a trial and enter the cut
   queue. Approximate Taylor coefficients are never used as constraint rows.

Defaults: `series_order=16`, `series_samples=8`, `series_steps=16`,
`series_tolerance=1e-5` in fitting-scale units, `series_max_panels=128`, and
`series_violation_tol=1e-10` in fitting-coordinate curvature units. All are
internal `_CurvatureOptions` fields. The candidate-selection threshold still
uses the existing `rho`. `series_violation_tol` determines only when a direct
point violation is large enough for early rejection; it never relaxes the
solver's original feasibility tolerance.

## Acceptance and fallback

The existing tail analysis, vanishing-boundary exclusion, interval curvature
search, and final certification are unchanged. A trial that is not rejected
by the heuristic still goes through the existing full curvature check before
normalization/acceptance. Small near-contact violations are deliberately left
to that refined search. The heuristic itself never returns a global
nonnegativity certificate.

Unsupported cache geometry, a panel budget overflow, or a primitive cache
requiring extended-exponent arithmetic disables the heuristic for that domain
and uses the original direct candidate search. A per-trial numerical failure
also falls back to the direct search. The underlying series evaluator still
supports its extended-range path; this experimental search cache intentionally
requires finite ordinary coefficients. Allocation failures propagate.

A missed minimum can reduce speed or yield fewer early cuts, but cannot cause
a trial to bypass the full acceptance test. There is no claim that the coarse
scan finds every local minimum, or that a given Taylor order certifies its
finite-degree remainder.

`BasisFaceObjective.series_statistics()` reports workspace builds, tensor builds/reuses, candidate calls,
scan nodes, per-derivative contractions, warm probes/tracked roots,
refinements, exact-theta replays, early rejection hits, numerical failures,
disabled-cache reasons and full searches. Read it after a run, not concurrently
with the mutable objective.

## Reproduce the experiment

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_series_search.py tests/test_basis_curvature_rows.py tests/test_basis_series.py -q
python tests/benchmark_series_search.py
```

The benchmark compares complete fits with the option off/on in alternating
order, including construction and the unchanged acceptance checks. It also
measures a stream of changing, rejected trial vectors separately. See
`series-search-results.md` and `series-search-benchmark.json` for the cold-search
measurements and baseline failure. The current inter-fit reuse measurements are
in `series-tensor-reuse.md` and `series-tensor-reuse-benchmark.json`.

Current candidate-kernel comparisons and discovery measurements are in
`series-staged-search.md` and the adjacent staged-search JSON files.
