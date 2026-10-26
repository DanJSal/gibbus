# Gibbus documentation

This directory contains user and developer documentation for the current Gibbus source tree. See the repository [README](../README.md) for a short introduction.

## Getting started and user guides

- [Getting started](getting-started.md) — installation, first fits, and basic workflows.
- [Modeling concepts](modeling-concepts.md) — support, log-concavity, basis potentials, and mixtures.
- [Fitting](fitting.md) — observations, support, weighting, fitting options, and model selection.
- [Using fitted distributions](using-distributions.md) — evaluation, moments, transforms, and model operations.
- [Serialization](serialization.md) — the durable format and its cache/provenance separation.
- [Diagnostics](diagnostics.md) — fit results, model checking, and uncertainty.
- [Examples](examples.md) — practical fitting and evaluation examples.
- [Limitations and concurrency](limitations.md) — numerical qualifications and thread safety.

## Technical reference

- [API reference](api-reference.md) — public classes, functions, methods, and properties.
- [Numerical methods](numerical-methods.md) — optimization, normalizers, interval likelihoods, and spectral evaluation.
- [Fitting performance](performance.md) — fitting costs, constraint retention, and representative runtimes.
- [Fitting coordinates](fitting-coordinates.md) — basis recurrences and the Newton controller.
- [Automatic tail parameters](automatic-tail-parameters.md) — within-family parameter estimation.
- [Case generator](case-generator.md) and [case runner](case-runner.md) — repeatable test infrastructure.

## Development and maintenance

- [Development status](development-status.md) — current implementation, known limitations, and verification scope.
- [Development](development.md) — building, testing, conventions, and verification.
- [Releasing](releasing.md) — CI, release automation, version/tag rules, and the release checklist.

Documentation uses US English except where public identifiers or quoted material require a different spelling.
