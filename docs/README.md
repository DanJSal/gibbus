# Gibbus documentation

This directory contains the user and developer documentation for Gibbus. The repository-level [README](../README.md) is intentionally a concise project landing page; detailed material lives here.

## Start here

If you are new to Gibbus, begin with [Getting started](getting-started.md), then read [Modeling concepts](modeling-concepts.md) for the assumptions behind a Gibbus distribution and the distinction between a single log-concave component and a finite mixture.

## User guide

- [Getting started](getting-started.md) — installation, first fits, and basic workflows.
- [Modeling concepts](modeling-concepts.md) — log-concavity, support, basis potentials, boundary terms, mixtures, and coordinate views.
- [Fitting](fitting.md) — input formats, `Distribution.fit()`, support selection, weights, mixtures, automatic selection, and warm starts.
- [Using fitted distributions](using-distributions.md) — density/CDF/quantile evaluation, moments, survival quantities, transforms, coordinate views, serialization, and components.
- [Serialization](serialization.md) — durable format 2, cache/provenance separation, version policy, and format evolution.
- [Diagnostics and model checking](diagnostics.md) — optimizer diagnostics, selection diagnostics, spectral diagnostics, goodness-of-fit tools, bootstrap bands, and suppressed-failure records.
- [Examples](examples.md) — complete examples for common fitting and evaluation tasks.
- [Limitations and concurrency](limitations.md) — modeling limits, numerical qualifications, and thread/concurrency behavior.

## Current development checkpoint

- [Review follow-up register](review-follow-up.md) — implementation status for R01–R29.
- [Automatic tail parameters](AUTOMATIC_TAIL_PARAMETERS.md) — empirical powers/rates within a requested family.
- [Development status](development-status.md) — code state, verification, open issues before a release, and milestone summary.
- [Benchmark findings](benchmark-findings.md) — the 96-case G3-057 benchmark, the G3-059/G3-060 profiling and fixes, the mixture cost analysis, and the basis/tail selection experiment.
- [Model-selection research plan](model-selection-research.md) — the agreed next objective, existing versus proposed automation, and acceptance guardrails.
- [Transition geometry](transition-geometry.md) — the G3-057 default skew-aware full-line rule and its scope.
- [Case generator](g3-case-generator.md) and [case runner](g3-case-runner.md) — reproducible test universe and benchmark execution.

These are **development snapshot** documents, not published release guarantees.

## Reference and internals

- [Staged candidate search](series-staged-search.md) — cached exact scan values, selective derivative coefficients, multiple warm minima, and discovery measurements.

- [Series tensor reuse](series-tensor-reuse.md) — bounded immutable tables across matching fits, workspace ownership, and cold/retained timings.

- [Cached-series Newton experiment](series-search-experiment.md) — optional candidate search, acceptance safeguards, and fitting benchmarks.

- [Local derivative series](local-series.md) — compiled nogil construction/evaluation of q'', q''', and q''''; internal API and numerical contract.

- [API reference](api-reference.md) — public classes, functions, methods, and properties.
- [Fitting coordinates](fitting-coordinates.md) — equivalent polynomial recurrences, the common Newton controller, and canonical validation.
- [Numerical methods and performance](numerical-methods.md) — optimization, interval likelihoods, spectral CDF/PPF construction, tails, caching, and performance architecture.
- [Development](development.md) — building, testing, verification, coding standards, and documentation standards.
- [Releasing](releasing.md) — version/tag rules, Trusted Publishing setup, release workflow behavior, and the maintainer release checklist.

## Documentation conventions

Repository-maintained documentation uses US English. Public names and terminology defined by external APIs, standards, proper names, or quoted material retain their required spelling.
