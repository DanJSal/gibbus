# G3 fitting case runner (G3-056)

The G3-055 generator selects **mathematically eligible** target distributions and
produces deterministic case IDs, without fitting. This separate runner executes
Gibbus fits against those manifests and records the results. It changes no
production model, optimization settings, acceptance tolerances, or fitting APIs.

## Basic usage

Run from the project root with compiled Gibbus extensions importable:

```bash
# First create the manifest; this does not fit anything.
python -m tests.g3_casegen --profile standard --count 200 \
    --fixed-core 20 --seed 42 --output cases.json

# Start small; per-case subprocess timeout is a wall-clock limit.
OPENBLAS_NUM_THREADS=1 python -m tests.g3_casegen.runner \
    --manifest cases.json --output benchmark/ --limit 20 \
    --seed 42 --jobs 2 --timeout 120 --holdout 128

# Run the rest of the manifest, reusing previously completed case results.
OPENBLAS_NUM_THREADS=1 python -m tests.g3_casegen.runner \
    --manifest cases.json --output benchmark/ --seed 42 --jobs 2 --timeout 120

# Re-run a particular case with the same data and fitting seed (fresh output).
python -m tests.g3_casegen.runner --manifest cases.json \
    --case-id G3C-REPLACE-WITH-REAL-ID --seed 42 --output replay/

# Repeat only unsuccessful attempts, preserving successful results.
python -m tests.g3_casegen.runner --manifest cases.json \
    --output benchmark/ --seed 42 --rerun-failed
```

`--dry-run` shows how many cases are pending without running fits.
`--limit` restricts the selected prefix; a second invocation without it runs
the rest. `--jobs` controls how many independent subprocesses run concurrently,
so memory and CPU utilization must be considered. Heavy-tail, high-degree,
mixture-selection, and censored cases can be expensive. For resource-limited
machines, start with `--jobs 1`, a small limit, and finite timeouts.

## Results and recovery

- `run_config.json`: manifest SHA-256, sample/fit seed, holdout size, timeout,
  runtime versions, platform information, and SHA-256 fingerprints of both
  the runner and Gibbus source/compiled-extension files. Resuming checks that
  the implementation, dependencies, manifest, seed, schema and holdout agree. Change the output directory to
  change those dimensions; changing only the timeout is permitted.
- `results.jsonl`: one independent JSON object per attempted case, flushed
  immediately. A timeout or worker crash is recorded and doesn't abort the
  batch. Re-running with `--rerun-failed` replaces the prior result for that ID.
- `summary.json`: grouped outcome counts by support, fit category, composition,
  observation mechanism and degree.
- `SUMMARY.md`: compact readable summary of attempts and convergence.

A worker is a fresh Python process for each case; the controlling process does
not call `Distribution.fit`. The worker loads the manifest and checks both
identifiers, then generates observations from the G3-055 sampler. The fitting
RNG stream and independent holdout sample are derived from the case ID and
requested run seed; results do not depend on execution order or job count.

Execution status meanings:

| Status | Meaning |
|---|---|
| `ok` | Fit and diagnostic collection completed without exception; *not* an accuracy certificate |
| `sampling_error` | Distribution/sample materialization failed |
| `fit_error` | Gibbus fit raised an exception |
| `diagnostics_error` | Fit completed but subsequent evaluation or diagnostics failed |
| `timeout` | Wall-clock time limit exceeded; subprocess terminated |
| `worker_error` | Worker process could not provide a valid result, e.g. process crash |

`fit_completed` and `fit_converged` are independent fields. A fit can finish
while reporting no convergence certificate. A converged fit can still have
statistical approximation error, since the true family need not be exactly
representable by the chosen basis. `fit_diagnostics`, `spectral_diagnostics`
and selected-K information from Gibbus are recorded where applicable.

## Interpretation of quantitative metrics

Each run scores an independently generated holdout sample, not the fitting
sample. Holdout samples are selected without access to the fitted model.

- `likelihood.observed_mean_log_likelihood`: per-row average point log density
  or interval log probability mass, including observation weights when present.
  Reported as `null` if any observation probability cannot be resolved.
- `validation.holdout_mean_log_density`: sample mean of the fitted log density
  at *latent holdout points*. Available for both point and censored fits.
- `validation.empirical_ks`: two-sided KS distance between the fitted CDF and
  the holdout empirical CDF. Includes irreducible holdout sampling noise.
- `validation.oracle_cdf_max_abs_error` and `oracle_cdf_mean_abs_error`:
  discrepancy at holdout points against independent SciPy/analytic CDFs.
- `validation.holdout_logscore_gap_vs_truth`: average log density of the true
  law minus the fitted law at holdout points. May be negative due to finite
  sampling and is not a deterministic KL divergence.
- `validation.cdf_ppf_max_abs_error`, `ppf_monotonicity_max_violation`,
  `cdf_monotonicity_max_violation`, `negative_pdf_count`,
  `cdf_outside_unit_interval_count`: numerical sanity measurements.
- `spectral_diagnostics`: built-in spectral mass defects, estimated error,
  refinement caps, shape/curvature margin, and other G3 measurements.

For independent **sums**, G3-055 supplies sampling but *not* an independent
convolution PDF/CDF oracle: the true-CDF and density-gap fields are `null`,
not zero. This also applies to mixtures containing a convolution. Empirical
holdout KS remains available. The oracle values for named/truncated/mixed
families are numerical references, not symbolic proof.

Censored observations are scored by their observed interval probability, not
by latent inaccessible points. Close CDF differences can cancel in extreme
tails; the runner tries a survival-function difference where preferable and
explicitly records unresolved masses rather than silently clipping scores.

**There are deliberately no built-in thresholds marking a fit statistically
"correct."** Establish thresholds empirically by target class, sample size,
model complexity and censoring regime, then implement acceptance tests as a
separate layer. Mixtures need component-level log-concavity, not overall
mixture log-concavity. A nonzero error against a valid target need not indicate
an optimizer bug if the requested finite basis is misspecified.

## Test coverage

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
    tests/test_case_generator.py tests/test_case_runner.py
```

Unit tests cover manifest corruption and duplicate IDs, deterministic seeds,
coarsened observation scoring, unavailable convolution oracles, error
reporting, grouped summaries, resume/provenance safety, and a real compiled
Gibbus smoke fit (skipped if extensions are unavailable). The sample handoff
includes execution evidence for all nine G3 support/boundary-log categories,
plus a heterogeneous two-component mixture and an independent sum. This is a
representative smoke validation, not a full 200-case benchmark or full G3
regression suite.
