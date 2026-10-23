# Series candidate search: measured experiment

Tested on Linux x86-64, CPython 3.12, GCC/G++. This experiment remains opt-in:
the cached heuristic helps repeated rejected trials, but construction costs can
outweigh that saving on short fits.

## Verification

- 69 targeted tests passed in both bounds-checked and release builds: 15 new
  search tests, 6 curvature-row tests, and 48 derivative-series tests.
- The 55 selected existing Newton, interval-fit, and mixture tests were run with
  the option disabled and enabled. Both modes passed 54 and failed the same
  test: `test_stacked_m_step_without_shared_columns_is_independent_fits`.
  It expects convergence, but both modes reach the iteration limit. Both
  return a globally nonnegative curvature certificate. This pre-existing
  convergence issue remains unresolved; this is not a clean full-suite result.
- Search tests cover randomized gated/tail bases (180 theta draws), actual
  basis constraint rows, domain/cache reuse, near-contact handling, warm starts,
  disabled/budget fallbacks, numerical-failure fallback, and a deliberately
  misleading heuristic that cannot bypass the full curvature check.

## Complete fits

Median wall time over seven paired runs, alternating off/on execution order.
Times include cache construction, solver work, and full acceptance checks.
These are small fixed examples, not a universal speedup claim.

| Case | Direct (ms) | Series (ms) | Direct / series | Absolute NLL difference |
|---|---:|---:|---:|---:|
| line-b2 | 3.113 | 3.912 | 0.80x | 0 |
| line-b4-gumbel | 35.650 | 31.131 | 1.15x | 4e-11 |
| half-b2-log | 8.681 | 10.057 | 0.86x | 0 |
| student-softpower | 24.439 | 18.834 | 1.30x | 5.2e-11 |

All four benchmark cases converge in both modes and receive the existing
nonnegative certificate. Different cut points can change the Newton trajectory;
parameters need not match bit for bit. The adjacent raw JSON includes parameter
differences, iterations, cache builds, rejection hits, and full-search counts.

## Rejected-trial kernel

The polynomial test streams 100 changing violating vectors, repeated five times,
with construction excluded. It measures the compiled gate including actual
basis-row evaluation; exact-theta replay is not used.

- Direct: 12.704 microseconds per trial.
- Cached series: 3.604 microseconds per trial.
- Speedup: 3.52x.
- One cache build, 500 series rejection hits, zero numerical failures.

The cache trims structurally zero high powers: this polynomial uses order 4
despite the requested default order 16. Positive approximate panels also skip
scanning. Neither optimization bypasses the unchanged full acceptance check.

The complete-fit result is therefore conditional: promising for repeated
constraint discovery, but not a default replacement for every fit. The plain
polynomial and half-line logarithmic examples did not generate early rejection
hits, and their extra setup made them slower.

See `series-search-experiment.md` for enabling the option, algorithm details,
fallbacks and reproduction commands, and `series-search-benchmark.json` for
the raw seven-run measurements.

## Geometry setup cost

No Cython or machine-code compilation occurs during fitting. Setup constructs
numerical coefficients, exact sign maps, panel geometry, and workspaces.
Sparse exact-map construction skips structural zeros while preserving rational
cancellation before conversion to double. Panel radius checks are batched, and
each model retains an immutable `series_plan` shared by its objectives. Mutable
candidate workspaces remain separate for each objective.

The setup comparison uses 21 alternating before/after pairs on newly constructed
models; the baseline is the previously delivered search experiment. Timers wrap
successful cache builds and include plan construction. Total setup time across
all builds in each fit is reported. These figures are separate measurements
from the seven-pair complete-fit benchmark above.

| Case | Previous setup (ms) | Current setup (ms) | Reduction |
|---|---:|---:|---:|
| line-b2 | 1.571 | 0.572 | 64% |
| half-b2-log | 5.164 | 1.733 | 66% |

The raw setup comparison is in `series-setup-benchmark.json`. Reusing a model's
plan can remove its reconstruction from later objectives, but a new center set
still requires coefficients for those centers. A retained primitive cache or
finished basis tensor avoids those computations when geometry, centers, scales,
and order remain fixed. These measurements precede process-local tensor
retention across fits; see [Tensor reuse](series-tensor-reuse.md) for the current
behavior and measurements. Numerical tensors are not persisted to disk, and
mutable search kernels are not shared between fits.

The 69 targeted release tests pass after these setup changes. The enabled fitting
comparison again passes 54 of 55 tests, with the same baseline iteration-limit
failure described above. These setup edits are Python-only; the compiled
per-trial kernel is unchanged from the bounds-checked verification.
