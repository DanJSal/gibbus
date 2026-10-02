"""Post-fit analytics, evaluation closures, and fitted-state packing.

Everything here runs *after* the optimiser has converged:

* :mod:`.analytics` — moments, summary statistics, affine updates, and mode
  finding without re-running the optimiser.
* :mod:`.evaluators` — cached closures for exact PDF and potential evaluation.
* :mod:`.fitted_state` — fitted-state packing and validation, including
  the packed spectral CDF/PPF fields.
* :mod:`.logsumexp` — batched derivative recurrence for mixture potentials.
* ``_mix_kernels`` — compiled stable mixture log-sum-exp and derivative recurrences.
* :mod:`.logspace` — stable log-space probability primitives used by post-fit APIs.
* :mod:`.survival` — tail-accurate survival, hazard, and reliability calculations.
* :mod:`.regions` — equal-tailed and highest-density regions for fitted distributions.
* :mod:`.expectation` — shared numerical expectation engine for fitted densities.
* :mod:`.information` — information measures built on the shared expectation engine.
* :mod:`.scoring` — held-out likelihood and probability-integral-transform helpers.
* :mod:`.gof` — EDF goodness-of-fit statistics on PIT values.
* :mod:`.resample` — bootstrap and parametric-simulation loops for
  uncertainty bands and simulated null distributions.
"""
