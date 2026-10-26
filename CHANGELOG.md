# Changelog

This file records notable user-facing changes of Gibbus releases.

## 0.2.0.dev0 — Unreleased

A new model replaces the 0.1 polynomial potential. The public entry point is
still `Distribution().fit`, but options, diagnostics and the saved format
differ; models saved by 0.1 are rejected rather than reinterpreted.

### Changed

- Potentials combine bulk polynomials, finite-endpoint logarithms and optional tail classes at infinite ends (`Polynomial`, `SoftPower` with any power above 1, `Exponential` with any positive rate), described by `BasisSpec`. Omitted tail parameters are estimated within the requested family.
- Log-concavity is enforced by curvature cuts and verified numerically over the whole support, tails included, within a recorded threshold (`shape_tau` in `fit_diagnostics`); a fit whose shape is not verified reports `unverified` and warns.
- Polynomial degrees, endpoint-log terms and the coefficient span are selected by score and likelihood-ratio tests. An explicit degree whose cold start does not converge is lifted from the converged smaller degrees.
- Finite mixtures are fitted by EM; the component count is selected by a staged BIC search whose upward screen stops above the KDE mode count once a count screens worse than its predecessor (`selection_diagnostics["screen_stopped_at"]`).
- Fitted models persist as non-object NumPy structured arrays (`format_version` 1 of this model); derived numerical caches are rebuilt when they do not match the running build.
- Python 3.11 or later is required.

### Added

- Interval-, left- and right-censored observations, including mixtures of them; interval data whose every row reaches the same support end are rejected, since their likelihood has no maximizer.
- Densities, distribution and survival functions, quantiles accurate far into the tails, moments, entropy, hazards, samples, affine transforms, truncations, goodness-of-fit statistics and bootstrap bands, with expectations resolved for narrow mass far from a support end.
- An optional cooperative fitting time budget (`max_fit_seconds`, `FitTimeoutError`).

### Known limitations

- Data concentrated far from a finite support end, which the anchored fitting coordinate cannot represent, are rejected with an explicit error. See [Limitations](docs/limitations.md#known-open-items) for the other open items.

## 0.1.0 - 2026-10-05

### Added

- Maximum-likelihood fitting of smooth univariate log-concave distributions on full-line, half-line, and bounded supports.
- Point and interval-censored observations with sample weights.
- Optional finite mixtures and staged automatic component-count selection.
- Density, log-density, CDF, survival, quantile, moment, sampling, transformation, information, and diagnostic interfaces on fitted distributions.
- Tail-aware numerical evaluation and compiled Cython kernels for performance-sensitive fitting and evaluation paths.
- Versioned, non-pickle model serialization with an explicit durable compatibility contract.
- Cross-platform CI for CPython 3.10 through 3.14 on Linux, Windows, and macOS.
- Release-wheel builds for the supported CPython versions and target platforms.

### Notes

- The `0.x` series is pre-1.0. Minor releases may refine the documented public API as the project matures.
- Free-threaded CPython builds are not supported or tested in `0.1.0`.
