# Fitting performance

Fitting is serial. Gibbus creates no thread or process pools. For reproducible
measurements, limit external numerical libraries to one thread with
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before
starting Python.

## Sources of fitting cost

Automatic component selection fits several candidate counts and degrees,
then refines shortlisted models. Mixture EM repeatedly fits its components.
High-degree bases and nearly degenerate curvature contacts can require many
constrained Newton subproblems and global curvature searches. After sufficient
statistics are prepared, point-data fit cost is largely independent of sample
count; interval-data likelihood evaluations integrate each distinct interval.

Constraint rows use compiled entrywise duplicate comparison. Constraint
balancing eliminates row offsets analytically before least squares. The
Newton controller tries scalar projection before the interior-point solver,
subject to the same quadratic-gap and descent checks. Each basis caches one
exact numerical coordinate template; observation statistics are recomputed
for each objective. Runtime caches are excluded from the durable model.

## Constraint retention

Global finite-point curvature cuts are retired after more than four
consecutive inactive exchange iterates within a face solve. Activity resets
the counter. Structural, tail, and face-scoped constraints remain. A normalized
row `r` is inactive at coefficients `theta` when

```text
r @ theta > 1e-5 * max(1, abs(r) @ abs(theta))
```

Every trial still passes the global curvature search before normalization,
and retired cuts can be regenerated. Shape, properness, and convergence
checks apply independently of retention. The internal immutable Newton policy
contains the retention count and activity margin; zero retention disables
retirement. An exchange can contain several Newton steps.

Retiring inactive constraints changes the numerical path. For nonconvex
mixture fitting, identical final coefficients or densities are not guaranteed.
The internal acceptance thresholds and statistical selection criteria apply
to every fit.

## Representative measurements

The environment is listed in [Development status](development-status.md).
These are individual measurements on synthetic data, not latency guarantees
or exhaustive worst-case bounds.

Each row is the fastest and slowest of three samples (seeds 0–2), with
numerical libraries limited to one thread; the slowest normal fit includes
first-call setup.

| Fit | Observations | Seconds |
| --- | ---: | ---: |
| Normal, automatic degree | 1,000 | 0.04 |
| Gamma on a half-line, automatic degree and endpoint log | 1,000 | 0.06–0.07 |
| Binned and right-censored gamma | 1,000 | 0.09–0.11 |
| Two normals, two components | 600 | 0.25–0.27 |
| Two normals, automatic component count | 600 | 2.8 |
| Three normals, automatic component count | 900 | 3.9–4.1 |
| Beta mixture on `(0, 1)`, automatic component count | 600 | 5.4–7.4 |

Runtime depends on geometry and candidate complexity as well as sample size.
Automatic component counts dominate: each candidate count is screened and
the winner refined. The slowest known cases are listed under
[limitations](limitations.md#known-open-items) (interval-censored mixtures
with gated tails, and high explicit degrees on half-lines).

Retiring inactive cuts was checked against full cut retention: across 20
automatic selections every component count and degree agreed, and sampled
log densities differed by at most `1.4e-7`. Explicit-degree mixtures, being
nonconvex, can settle in slightly different optima (up to about 2% in
pointwise density in the cases measured).
