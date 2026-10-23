# Reusing series tensors across fits

The optional series search retains finished, immutable coefficient tensors in a
bounded process-local cache. Repeated fits with matching numerical geometry and
compact domains reuse those tensors. Each fit allocates its own mutable search
workspace; its warm starts, contracted coefficients, candidate scores, replay
state and statistics are independent.

## Matching and ownership

The value key includes support endpoints, fitting center/scale, gate locations
and steepnesses, each atom's kind and fixed parameters, and every exact rational
coefficient-map entry. It also includes requested series order, panel budget,
scan subinterval count, and compact-domain endpoints. The domain and planner inputs determine the
center and scale of every panel. No parameter is rounded for matching, and
independently built matching models can share a table without sharing identity.
A changed geometry, exact map, order, budget, scan count or domain causes a miss. Within one
objective, an already prepared larger domain can still service a smaller domain.

Refinement settings, candidate capacity, tolerances, rho, and theta do
not enter the table key. The scan count determines cached basis-row locations. They are applied to each private workspace. A retained
hit skips exact sign-map construction, panel planning, primitive jets, basis
column contraction, tensor validation, trimming and copying. The retained snapshot includes actual basis derivatives at scan points. It still
computes a value signature for a new model and allocates private work buffers.

The retained arrays are backed by immutable bytes. Exposed views cannot change
their values or the internal array shape metadata. The compiled table stores
const pointers to shared geometry/tensor arrays and separate mutable pointers
for each workspace. Per-trial contraction, scanning and refinement remain
allocation-free noexcept nogil operations; lookup and ownership are prepared
before that loop.

## Bounds and concurrency

The cache retains at most 32 entries and 32 MiB of array/signature bytes, using
least-recently-used eviction. Python object/container overhead is additional.
It holds neither model objects nor mutable search kernels. Oversized tables
remain usable by their owning fit but are not retained globally. Existing
kernels keep their tables alive after eviction or clearing, so the retention
budget does not bound memory held by active fits.

Lookup/publication/eviction use a lock. Numeric construction runs outside it:
concurrent cold misses may compute duplicate tables, then publication selects
one immutable table. Each objective remains unsuitable for concurrent calls
on that same mutable objective; distinct objectives may run concurrently.

Retention lasts within the current Python process. Nothing is serialized to
disk, and a fresh process pays the first construction cost. Unsupported or
unrepresentable geometry follows the existing direct-search fallback; failed
constructions are not retained.

Internal diagnostics and clearing:

```python
from gibbus._fit.series_search import (
    _series_tensor_cache_info, _clear_series_tensor_cache,
)

print(_series_tensor_cache_info())
_clear_series_tensor_cache()   # existing kernels remain valid
```

`BasisFaceObjective.series_statistics()` reports per-block `builds` (private
workspace constructions), `tensor_builds` (new numerical tables actually
computed), and `tensor_reuses` (retained hits), alongside the existing gate
statistics. A workspace can reuse its own domain without increasing any of
these counters. Concurrent duplicate construction counts as a build even if
publication selects a table another thread just published.

## Measurements

The measurements below describe the original shared-tensor kernel. Current
selective construction and exact-scan results are in
[Staged candidate search](series-staged-search.md).

Linux x86-64, CPython 3.12, GCC/G++. Medians over nine triplets for each case.
Each triplet clears retained tensors, runs a cold series fit, then constructs a
new matching model and runs a retained fit. The direct comparison alternates
before/after those two fits. Timings include the internal Newton solve, cache
setup, private workspaces, and full acceptance checks. Initial basis/objective
construction is outside the timer in all modes. This is a small fixed benchmark,
not a guarantee of speed for arbitrary geometry or data.

### Solver time

| Case | Original direct search (ms) | Cold series (ms) | Retained series (ms) |
|---|---:|---:|---:|
| line-b2 | 3.303 | 3.876 | 3.193 |
| line-b4-gumbel | 36.634 | 32.756 | 30.422 |
| half-b2-log | 8.354 | 9.300 | 8.702 |
| student-softpower | 24.242 | 19.291 | 17.207 |

### Setup time inside the solve

| Case | Cold series (ms) | Retained series (ms) | Reduction |
|---|---:|---:|---:|
| line-b2 | 0.579 | 0.061 | 89% |
| line-b4-gumbel | 1.440 | 0.179 | 88% |
| half-b2-log | 0.729 | 0.094 | 87% |
| student-softpower | 1.705 | 0.116 | 93% |

Every retained run computed zero new tensors. Cold and retained fits produced
bit-for-bit identical parameter vectors and objective values in every pair,
and all benchmark fits converged with the existing nonnegative certificate.
The logarithmic cold fit also benefits from table reuse among objectives within
that single fit, so its cold setup can be lower than the earlier uncached
setup measurements.

The simple line case is roughly level with the direct solver; the logarithmic
case retains about 4% overhead in this run. Complex examples benefit more, but
these small timing differences and conditional matching do not justify making
the series option the default. It remains opt-in with
`_CurvatureOptions(series_candidates=True)`.

### Validation and reproduction

83 targeted tests pass with bounds-checked series-search code and in release
mode. The new tests cover independent-model reuse without jet reconstruction,
one-ULP parameter changes, exact-map/domain/order/budget misses, immutable
snapshots, private runtime settings and workspaces, eviction/clear/byte budgets,
concurrent warm searches and concurrent cold publication. The existing selected
fitting comparison again passes 54 of 55; the same stacked-mixture test reaches
the iteration limit with or without the series option. That baseline issue
remains unresolved.

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_series_search.py tests/test_basis_curvature_rows.py tests/test_basis_series.py -q
python tests/benchmark_series_tensor_reuse.py
```

Raw nine-triplet timings and construction/reuse counts are in
`series-tensor-reuse-benchmark.json`. `benchmark_series_search.py` explicitly
clears retained tensors before its series runs to preserve its cold comparison.
The full curvature acceptance and certification checks remain in place.
