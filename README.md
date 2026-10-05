# gibbus

[![CI](https://github.com/DanJSal/gibbus/actions/workflows/ci.yml/badge.svg)](https://github.com/DanJSal/gibbus/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/DanJSal/gibbus/blob/main/LICENSE)

**Flexible maximum-likelihood modeling of smooth univariate log-concave distributions.**

`gibbus` fits explicit, support-aware analytic probability distributions to point or interval-censored data without requiring a fixed named family such as Normal or Gamma. A fitted `Distribution` represents a **Gibbus distribution**. A single component is log-concave and therefore unimodal; finite mixtures extend the model to multimodal data.

Unlike a histogram, KDE, spline, or piecewise-linear log-density, a fitted Gibbus distribution has an explicit analytic density and negative-log density on the interior of its support. The model is fit by maximum likelihood under a global convexity constraint on the negative-log density, and the fitted object exposes density, CDF, quantile, moment, sampling, survival, diagnostic, and transformation operations through one distributional interface.

## Why Gibbus?

- **No fixed named family required.** Log-concavity is the primary per-component shape assumption, with progressively richer smooth potentials available through polynomial degree.
- **Support-aware modeling.** Full-line, half-line, and bounded supports use representations adapted to their geometry rather than post-hoc clipping.
- **Point and interval-censored observations.** Interval observations contribute their probability masses directly rather than midpoint substitutions.
- **Optional mixtures.** The default fit is a single log-concave component; explicit or automatically selected finite mixtures support multimodal data.
- **Full distributional interface.** Evaluate densities, probabilities, quantiles, moments, samples, survival quantities, information measures, and diagnostics from the fitted model.
- **Tail-aware numerics.** Extreme-tail probabilities and quantiles use dedicated log-domain and direct-tail machinery instead of numerically fragile `1 - cdf(x)` calculations.
- **Exponential-family perspective.** For fixed support, boundary structure, and polynomial degree, the natural coefficients form an exponential-family representation, linking maximum-likelihood fitting to moment-matching and maximum-entropy geometry under the convexity constraint.
- **Versioned persistence.** Fitted models have a non-pickle NumPy serialization format with an explicit cross-version compatibility contract.
- **Compiled numerical kernels.** Performance-sensitive fitting, quadrature, certification, and spectral evaluation routines are implemented in Cython.

See [Modeling concepts](https://github.com/DanJSal/gibbus/blob/main/docs/modeling-concepts.md) for the statistical model and its relationship to fixed parametric families and nonparametric density estimators.

## Installation

For a published release, install from PyPI:

```bash
python -m pip install gibbus
```

Gibbus 0.1.x is tested on CPython 3.10 through 3.14. Release wheels are intended to cover the supported Linux, Windows, and macOS targets, so a compiler is not required when pip selects a compatible wheel.

For a source checkout:

```bash
git clone https://github.com/DanJSal/gibbus.git
cd gibbus
python -m pip install .
```

Source builds compile the Cython extensions and therefore require a C compiler and Python development headers. For editable development installation and build details, see [Development](https://github.com/DanJSal/gibbus/blob/main/docs/development.md).

## Quick start

```python
import numpy as np
from gibbus import Distribution

rng = np.random.default_rng(0)
samples = rng.normal(size=500)

c = Distribution().fit(samples, support=(-np.inf, np.inf))

c.mean, c.std, c.mode, c.median
c.pdf([-1.0, 0.0, 1.0])
c.cdf(0.0)
c.ppf(0.25)
c.sample(100, rng=rng)
```

The default is a single log-concave component. For multimodal data, specify a component count or opt into automatic selection:

```python
bimodal = np.concatenate(
    [rng.normal(-2, 0.5, 300), rng.normal(2, 0.5, 300)]
)

c2 = Distribution().fit(
    bimodal,
    n_components="auto",
    support=(-np.inf, np.inf),
)

c2.n_components
c2.weights
c2.modes
```

Automatic component selection is a staged KDE/BIC search rather than an exhaustive or globally certified search. See [Fitting](https://github.com/DanJSal/gibbus/blob/main/docs/fitting.md) for the selection policy and fit options.

## Documentation

The documentation is organized by task and subject rather than kept in one large README:

- [Documentation index](https://github.com/DanJSal/gibbus/blob/main/docs/README.md)
- [Getting started](https://github.com/DanJSal/gibbus/blob/main/docs/getting-started.md)
- [Modeling concepts](https://github.com/DanJSal/gibbus/blob/main/docs/modeling-concepts.md)
- [Fitting](https://github.com/DanJSal/gibbus/blob/main/docs/fitting.md)
- [Using fitted distributions](https://github.com/DanJSal/gibbus/blob/main/docs/using-distributions.md)
- [Serialization and compatibility](https://github.com/DanJSal/gibbus/blob/main/docs/serialization.md)
- [Diagnostics and model checking](https://github.com/DanJSal/gibbus/blob/main/docs/diagnostics.md)
- [API reference](https://github.com/DanJSal/gibbus/blob/main/docs/api-reference.md)
- [Examples](https://github.com/DanJSal/gibbus/blob/main/docs/examples.md)
- [Numerical methods and performance](https://github.com/DanJSal/gibbus/blob/main/docs/numerical-methods.md)
- [Limitations and concurrency](https://github.com/DanJSal/gibbus/blob/main/docs/limitations.md)
- [Development](https://github.com/DanJSal/gibbus/blob/main/docs/development.md)

## Compatibility and versioning

The 0.x series is pre-1.0: documented public APIs are intended to remain usable within a release line, but minor releases may still refine the public surface as the project matures. Patch releases are reserved for compatible fixes and maintenance changes.

Durable serialized models use their own explicit format version and compatibility contract; package version numbers do not silently reinterpret an older serialization format. See [Serialization and compatibility](https://github.com/DanJSal/gibbus/blob/main/docs/serialization.md).

Free-threaded CPython builds are not currently supported or tested.

## Public API

The supported top-level API consists of:

```python
from gibbus import Distribution, suppressed_failures, clear_suppressed_failures
```

`Distribution` is the fitted-model interface. The two module-level helper functions expose numerical fallbacks that were suppressed during graceful degradation. Other package names are implementation details unless documented otherwise.

## Contributing and citation

Contribution guidance is in [CONTRIBUTING.md](https://github.com/DanJSal/gibbus/blob/main/CONTRIBUTING.md). Release history is tracked in [CHANGELOG.md](https://github.com/DanJSal/gibbus/blob/main/CHANGELOG.md).

For research use, citation metadata is provided in [CITATION.cff](https://github.com/DanJSal/gibbus/blob/main/CITATION.cff).

## License

`gibbus` is distributed under the MIT License. See [LICENSE](https://github.com/DanJSal/gibbus/blob/main/LICENSE) for the full terms.
