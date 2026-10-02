"""gibbus — flexible univariate probability-distribution modeling.

Gibbus fits explicit support-aware analytic distributions by maximum likelihood
under per-component log-concavity.  The :class:`Distribution` class handles
point or interval-censored observations, with a single component for a smooth
unimodal log-concave fit and finite mixtures for multimodal data.  Increasing
polynomial degree provides progressively richer smooth convex potentials;
optional finite-boundary logarithmic terms supply additional endpoint behavior.

Quick start — unimodal
----------------------
::

    import numpy as np
    from gibbus import Distribution

    rng = np.random.default_rng(0)
    samples = rng.normal(size=500)
    c = Distribution().fit(samples, support=(-np.inf, np.inf))

    c.mean, c.std           # summary statistics
    c.pdf(0.0)              # evaluate PDF
    c.ppf(0.25)             # lower quartile
    c.sample(100, rng=rng)  # draw samples

Quick start — multimodal
-------------------------
::

    bimodal = np.concatenate([rng.normal(-2, 0.5, 300),
                              rng.normal(2, 0.5, 300)])
    c2 = Distribution().fit(bimodal, n_components=2,
                            support=(-np.inf, np.inf))
    c2.pdf(0.0)             # valley between the two modes

The default view after fitting is ``"base"`` (models *x* directly).
Switch to the exp-space view to model positive variates::

    c.set_default("exp")
    c.mean   # E[exp(X)]

Save and restore a fitted model::

    state = c.data           # numpy structured scalar
    c2 = Distribution(state)        # reconstruct without re-fitting

    np.save('model.npy', c.data)
    c3 = Distribution(np.load('model.npy', allow_pickle=False))

Diagnosing a suspect fit
------------------------
Fitting degrades rather than failing in several places.  Two accessors
make that visible::

    c.fit_diagnostics            # did likelihood optimization converge?
    c.selection_diagnostics      # why was automatic K selected?
    c.spectral_diagnostics       # was the CDF/PPF actually resolved?
    gibbus.suppressed_failures() # which numerical fallbacks were taken

Public API
----------
.. autosummary::

    Distribution
    suppressed_failures
    clear_suppressed_failures

All other names in this package are implementation details and subject
to change without notice.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from ._api.distribution import Distribution
from ._defaults import clear_suppressed_failures, suppressed_failures

try:
    __version__ = _distribution_version("gibbus")
except _PackageNotFoundError:  # source tree without installed metadata
    __version__ = "0+unknown"

__all__ = [
    "Distribution",
    "suppressed_failures",
    "clear_suppressed_failures",
    "__version__",
]

