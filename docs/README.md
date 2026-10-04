# Gibbus documentation

This directory contains the user and developer documentation for Gibbus. The repository-level [README](../README.md) is intentionally a concise project landing page; detailed material lives here.

## Start here

If you are new to Gibbus, begin with [Getting started](getting-started.md), then read [Modeling concepts](modeling-concepts.md) for the assumptions behind a Gibbus distribution and the distinction between a single log-concave component and a finite mixture.

## User guide

- [Getting started](getting-started.md) — installation, first fits, and basic workflows.
- [Modeling concepts](modeling-concepts.md) — log-concavity, support, polynomial potentials, boundary terms, mixtures, and coordinate views.
- [Fitting](fitting.md) — input formats, `Distribution.fit()`, support selection, weights, mixtures, automatic selection, and warm starts.
- [Using fitted distributions](using-distributions.md) — density/CDF/quantile evaluation, moments, survival quantities, transforms, coordinate views, serialization, and components.
- [Diagnostics and model checking](diagnostics.md) — optimizer diagnostics, selection diagnostics, spectral diagnostics, goodness-of-fit tools, bootstrap bands, and numerical fallback records.
- [Examples](examples.md) — complete examples for common fitting and evaluation tasks.
- [Limitations and concurrency](limitations.md) — modeling limits, numerical qualifications, and thread/concurrency behavior.

## Reference and internals

- [API reference](api-reference.md) — public classes, functions, methods, and properties.
- [Numerical methods and performance](numerical-methods.md) — optimization, interval likelihoods, spectral CDF/PPF construction, tails, caching, and performance architecture.
- [Development](development.md) — building, testing, verification, coding standards, and documentation standards.

## Documentation conventions

Repository-maintained documentation uses US English. Public names and terminology defined by external APIs, standards, proper names, or quoted material retain their required spelling.
