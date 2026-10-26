# Development status

**Current source version:** `0.2.0.dev0` (unreleased).

## Implemented capabilities

The library fits a general potential basis (bulk polynomials, finite-endpoint logarithms, and `Polynomial`, `SoftPower` or `Exponential` tails) under coefficient-linear curvature constraints. It fits weighted point and interval observations and finite mixtures. Automatic selection covers the number of components, polynomial degree, endpoint-log terms, coefficient span, and within-family tail parameters. Fitted distributions expose post-fit statistical operations and durable persistence (`format_version` 1).

The fitter uses support-appropriate polynomial recurrences and one constrained Newton controller. Log-concavity is verified numerically over the whole support within a threshold `tau` (default `1e-6`, recorded per component as `shape_tau`); a fit whose shape is not verified reports `unverified` and warns. Inspect `fit_diagnostics` and `spectral_diagnostics` when numerical accuracy matters.

## Verification

Environment: Linux x86-64, CPython 3.13.17, NumPy 2.5.3, SciPy 1.18.1,
Cython 3.3.0, and GCC 13.3. The source uses 12 native extensions.

- Full suite: **6,379 passed, 1 skipped, 11 warnings**, in 14 minutes with
  two worker processes and numerical libraries limited to one thread each.
  The `test` extra (`mpmath`, `sympy`, `pytest`, `pytest-xdist`) is required.
- Post-fit quantities (density, distribution and survival functions, quantiles,
  tails to `e^-300`, moments, entropy, mode, hazard, residual life, exp-space
  mean, truncation, affine views, sampling, goodness of fit, and the durable
  normalizer) are checked against an independent high-precision oracle on five
  supports/shapes and a mixture (`tests/test_postfit_oracle.py`).
- Black and Ruff pass on the whole tree; `cython-lint`
  reports only long-standing accepted findings.
- Fit coverage was also exercised by 140 generated cases (`tests/casegen`) and
  a set of adversarial point, censored, mixture and tail-family fits.
- A wheel built from the tree and installed outside the checkout imports all
  12 extensions and passes the packaging, serialization and expectation tests.
- CI covers CPython 3.11–3.14 on Linux, Windows and macOS. Debug builds, the
  minimum declared NumPy/SciPy versions and the release wheel matrix are not
  yet qualified.

See [Fitting performance](performance.md) for measured runtimes and the
constraint-retention policy. These measurements do not guarantee latency or
cross-platform correctness.

## Known limitations and further work

- See [Limitations](limitations.md#known-open-items) for the numerical limits (data concentrated far from a finite end, high explicit degrees on half-lines, high-degree polynomial tails, heavy tails and `SoftPower`).
- Mixture fitting, and above all automatic component-count selection, is much more expensive than single-component fitting.
- Automatic tail-family comparison and broader predictive-tail evaluation remain research work. An explicit tail family is not changed automatically, even when it may fit the data poorly.
- Debug builds, the minimum declared dependency versions and the release wheel matrix have not been qualified; CI covers the supported interpreters on Linux, Windows and macOS.

See [Limitations](limitations.md), [Fitting](fitting.md), and [Fitting coordinates](fitting-coordinates.md) for details.
