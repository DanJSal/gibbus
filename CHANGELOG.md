# Changelog

This file records notable user-visible changes to Gibbus.

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
