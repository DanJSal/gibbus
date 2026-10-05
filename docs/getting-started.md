# Getting started

This guide covers installation and the shortest path from observations to a fitted `Distribution`. For the model assumptions and terminology behind these examples, see [Modeling concepts](modeling-concepts.md).

## Installation

For a published release, install Gibbus from PyPI:

```bash
python -m pip install gibbus
```

Gibbus 0.1.x is tested on CPython 3.10 through 3.14. When pip selects a compatible release wheel, no compiler is required.

To install from a source checkout instead, clone the repository and build locally. Source builds compile the Cython extensions and therefore require a C compiler and Python development headers (`build-essential` and `python3-dev` on Debian/Ubuntu, the Xcode command-line tools on macOS, or MSVC Build Tools on Windows). Cython itself is declared as a build dependency.

```bash
git clone https://github.com/DanJSal/gibbus.git
cd gibbus
python -m pip install .
```

For editable source development:

```bash
python -m pip install -e .
```

Free-threaded CPython (`t` builds such as 3.14t) is not supported or tested yet. The compiled extensions do not declare free-threading compatibility; a source build may therefore re-enable the GIL when `gibbus` is imported. Do not treat an installation on a `t` build as no-GIL support.

On GCC/Clang builds, set `GIBBUS_NATIVE_ARCH=1` at build time to add `-march=native`. This can enable wider machine-specific SIMD and other target tuning, but produces machine-specific binaries, so it is off by default and is not used for release wheels.

## Dependencies

- CPython 3.10 through 3.14 are tested for the 0.1.x release line
- NumPy
- SciPy

## Fit a single-component distribution

```python
import numpy as np
from gibbus import Distribution

rng = np.random.default_rng(0)
samples = rng.normal(size=500)

c = Distribution().fit(samples, support=(-np.inf, np.inf))
```

A single component is the default. It is log-concave and therefore unimodal.

```python
c.mean, c.std, c.mode, c.median
c.pdf(0.0)
c.pdf([-1.0, 0.0, 1.0])
c.cdf(0.0)
c.ppf(0.25)
c.sample(100, rng=rng)
```

Scalar evaluation inputs return Python `float` values; array inputs return NumPy arrays.

## Fit multimodal data

Use an explicit component count when it is known:

```python
bimodal = np.concatenate(
    [rng.normal(-2, 0.5, 300), rng.normal(2, 0.5, 300)]
)

c2 = Distribution().fit(
    bimodal,
    n_components=2,
    support=(-np.inf, np.inf),
)

c2.weights
c2.modes
```

Or opt into automatic component-count selection:

```python
c_auto = Distribution().fit(
    bimodal,
    n_components="auto",
    support=(-np.inf, np.inf),
)

c_auto.n_components
c_auto.selection_diagnostics
```

Automatic selection uses a staged KDE/BIC search. It is a model-selection aid, not an exhaustive proof that the selected component count is globally optimal. See [Fitting](fitting.md#automatic-component-count-selection) for details.

## Next steps

- Read [Fitting](fitting.md) for supports, interval-censored observations, sample weights, boundary terms, polynomial degree, mixtures, and warm starts.
- Read [Using fitted distributions](using-distributions.md) for the full distributional interface.
- Browse [Examples](examples.md) for complete workflows.
- Check [Diagnostics and model checking](diagnostics.md) when a fit or numerical result needs auditing.
