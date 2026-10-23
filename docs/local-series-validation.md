# Local series implementation validation

Environment: CPython 3.12.14, NumPy 2.3.5, GCC/G++ on Linux-6.18.55-x86_64-with-glibc2.39.

| Check | Result |
|---|---|
| New series tests, release | 48 passed |
| Bounds-checked Cython build | 46 passed (the two additional soft-power/log range tests ran in release) |
| Existing basis evaluation + construction | 1,946 passed |
| Full extension build | Passed |
| Source distribution | Built; new Cython sources, pxd and C++ helper included |

No full optimizer/end-to-end suite or cross-platform wheel matrix was run; the new API is not yet called by those paths.

## Order-24 benchmark

Medians of five repetitions; each repetition makes 25 calls (five for the direct evaluator). Construction and cache timings include validation and allocations. All benchmark centers used ordinary double arithmetic. Times are machine/workload-specific, not release guarantees.

| Basis | Build 16 centers (µs) | Recombine cached 16 centers (µs) | Evaluate 10,000, public (µs) | Evaluate 10,000, kernel wrapper (µs) | Existing direct evaluation (µs) | Public evaluation speedup |
|---|---:|---:|---:|---:|---:|---:|
| polynomial_gates | 314.1 | 64.0 | 253.2 | 128.2 | 2666.0 | 10.5× |
| soft_power_gates | 453.2 | 62.3 | 221.1 | 130.8 | 6047.0 | 27.3× |
| exponential_gates | 604.1 | 135.6 | 251.3 | 162.8 | 4797.0 | 19.1× |
| bounded_logs | 92.7 | 43.7 | 255.9 | 139.2 | 1601.1 | 6.3× |

The direct evaluator computes orders 0–4; the new evaluator returns orders 2–4. The comparison represents the current available baseline, not an equal-operation-count algorithm comparison. Construction is excluded from the evaluation speedup and must be amortized over queries. The kernel-wrapper measurement still includes Python entry/output allocation; the exported raw-pointer C API avoids those costs.

The JSON includes orders 12, 24 and 48, and separate primitive-cache construction timings. Higher order costs more and does not automatically imply a certified error bound. Reproduce with `python tests/benchmark_basis_series.py`.

## Files changed

- New numerical engine: `_model/_series_kernels.pyx`, `_series_kernels.pxd`, `_series_number.hpp`.
- New immutable facade/cache: `_model/basis_series.py`.
- Build registration and source/header packaging: `setup.py`, `MANIFEST.in`, `pyproject.toml`.
- Independent oracle, numerical tests and benchmark under `tests/`.
- Usage/numerical contract documentation, documentation index and changelog.

Existing Newton descent, minima search, tail cutoffs and model evaluation code are unchanged.
