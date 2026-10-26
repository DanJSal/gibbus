# Log-concave case generator

`tests/casegen/` is a **standalone, test-infrastructure-only** module. It
creates reproducible distributions, observation records and fit *requests*;
it never calls `Distribution.fit`. There are no production-package or
optimization changes. Python 3.11+, NumPy and SciPy are sufficient for
registry use and dataset generation. Creating fit kwargs requires an
importable, built version of Gibbus.

## Generate a balanced case manifest

From the project root:

```sh
python -m tests.casegen --profile standard --count 200 \
    --fixed-core 20 --seed 42 --output cases.json
```

This writes a JSON manifest with the full case specification and stable IDs,
**not** potentially large arrays. The default catalog has 32 named family
*definitions* represented by many parameter choices (including four that
require truncation, plus beta's restricted parameter subregions), along with reflection, location/scale changes,
heterogeneous mixtures and convolution examples. `core` is smaller.

The universe is the lazy finite product of:

- each registered distribution specification, with its validated support;
- the supported support/boundary-log variants: S01 through S09;
- configured polynomial degrees and suitable tail classes;
- point, weighted-point, left/right/interval/rounded/binned/mixed observation modes;
- sample sizes and replicate indices;
- the fit component count and expansion policy.

To include every eligible pairwise sum and cross-family shared-support mixture
from the parameter-grid catalog, add `--pairwise`. The
standard profile expands from 81 curated distributions to 3,739
distribution specifications. Explicit two- and three-component
mixtures are included; arbitrary larger valid mixtures can be supplied as
`Mixture` expression trees. The default fit-count axis for mixtures is
`1, 2, 3, "auto"`.

There is no accidental cross-product of unsupported supports and boundary
logs. Fit requests are screened for support/log syntax, **not for all of
the solver's properness/curvature or structural leader conditions**;
the numerical fitter remains responsible for its own validation. Sampling a
case does not imply that its requested finite model can represent the truth
exactly, particularly a multi-component mixture fitted with one component.

## Programmatic example

```python
import numpy as np
from tests.casegen import (
    Named, Truncated, Sum, Mixture, CaseUniverse, ObservationSpec,
    CaseSpec, FitSpec, materialize, default_distributions,
)

# Custom expression tree with independent summands
law = Sum(Named("normal", mu=0, sigma=1), Named("gamma", shape=2, scale=1))

# Materialize one case on demand; no model fitting
case = CaseSpec(law, ObservationSpec("interval", fraction=0.4, width=0.2),
                FitSpec(degree=4), n=500, replicate=0)
record = materialize(case, seed=42)
print(case.scenario_id, case.case_id)
print(record.observations.shape)  # (500, 2)

universe = CaseUniverse(default_distributions("standard"))
chosen = universe.select(200, seed=42, explore_fraction=0.15)
# Independently regenerate one selected case, on another machine or later:
record = materialize(chosen[0], seed=42)

# Only with a compiled, importable Gibbus package:
# fitted = Distribution().fit(record.observations, **record.fit_kwargs)
```

For censored data, the same array contains exact rows `(x,x)` and censored
interval rows `(lo,hi)`; one-sided censoring uses infinite support endpoints
when appropriate. All intervals are checked to contain their latent values
and lie inside the target support. The latent values are returned solely for
verification and scoring; they must never be passed to the censored fit.

## What “log-concave” and “smooth” mean here

Named families are validated under **analytic sufficient parameter
conditions**. A family that is log-concave only on part of its support
(`lognormal_restricted`, `student_t_restricted`, `cauchy_restricted`,
`inverse_gaussian_restricted`, and subunit-shape beta distributions) is
rejected unless explicitly truncated to a valid interval. An affine
transformation (including reflection) and conditioning on an interval
preserve log-concavity. The independent sum of log-concave variables is
log-concave by convolution. **A mixture generally is not log-concave**: the
component-level guarantee, not aggregate log-concavity, is what the mixture
model uses.

`Cinf` or `C2` smoothness refers to the *interior* of the support only.
Finite-boundary discontinuity/endpoint singularity is tracked by the finite
support but is **not** a claim of smoothness across a hard boundary.
Convolution smoothness is conservatively reported as `unknown` except when
one factor is Gaussian, which guarantees a smooth density. Users who require
only proven smooth densities may filter by `smoothness(spec) != 'unknown'`.

The registry is deliberately finite and extensible, not a claim to list
all named log-concave laws. Several entries overlap mathematically (e.g.
chi, chi-square and gamma); the catalog is oriented toward coverage of
family interfaces and geometries, not an independent family census.

## Distributional oracles

Independent SciPy or analytic PDF/CDF operations are available for named,
affine, truncated and simple mixture cases. **Convolution PDF/CDF are not
implemented**; direct independent-summand sampling works, and a mixture
containing a convolution retains its ability to sample. These cases should
not be used for density-error or CDF-error assertions without introducing
an external integration oracle and recording its error estimate.

## Sampling, balance and reproducibility

`CaseUniverse.iter_cases()` enumerates the logical universe without storing
cases or generating observations. `select(count, seed=...)` proposes
configurations by fit category and distribution structure rather than
uniformly from catalog entries. Greedy coverage selects most cases to balance
fit category, single/sum/mixture structure, observation mode, sample size,
family and pairwise interactions; by default the remaining 15% are drawn
randomly from feasible proposals. Selection uses a bounded candidate pool and
256-way tournaments, not exhaustive enumeration. **Balancing is approximate,
not a guarantee of all feasible pairs at small budgets.**

Use `fixed=(...)` for immutable baseline reference cases. The CLI option
`--fixed-core N` selects N deterministic core reference cases. The optional
`--exclude-manifest` removes previously selected IDs. `--completed-manifest`
adds category counts from **completed** previous runs and excludes them,
so cumulative exploration favors under-covered interactions. The CLI does
not assume that a previously selected case was actually executed.

A `scenario_id` hashes a canonical scenario without the replicate index; a
`case_id` includes it. Data generation adds an explicit sample seed, which is
independent of the selection seed and execution order. JSON manifests retain
full specs because hashes alone cannot be decoded. A schema version is
included; a change to generation semantics should increment it.

Censoring cutoff thresholds are estimated from a separate seeded pilot sample,
not the fit sample; discretization uses fixed bins with an independently
randomized phase. The `fraction` parameter is a target tail probability for
left/right censoring, or a coarsening probability for `mixed`, **not a promise
of an exact realized censor fraction**. Always inspect `record.censored.mean()`.
Mixture sampling draws component labels internally and generates component
samples independently. Exact component labels are currently not exported.

Weighted-point records use independent positive random weights with mean one;
weighted interval records are deferred, as is a fully independent convolution
PDF/CDF oracle. This infrastructure should be tested separately from future
fitting-accuracy and runtime tests.

## Module-level sanity checks

```sh
python -m pytest -q tests/test_case_generator.py
```

These test distribution eligibility, support validation, mixture compatibility,
independent sampling and source density normalization, censoring containment,
stable IDs, nine fit categories, and seeded selection. They do **not** count as
Gibbus fit regressions or as a completed full suite.
