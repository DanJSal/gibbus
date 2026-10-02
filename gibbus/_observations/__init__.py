"""Observation providers: the seam between raw data and the likelihood.

Each module turns validated user observations into the canonical summary the
objective actually consumes, so the inner pipeline never revisits raw data:

* :mod:`.empirical` — exact weighted point sufficient statistics (power
  moments, endpoint-log expectations, Kish effective N, participation
  metadata).
* :mod:`.points` — point observation provider around the empirical statistics.
* :mod:`.intervals` — canonical interval representation, validation, exact
  duplicate collapse, shared Gauss--Legendre rule metadata, and prepared
  model-aware adaptive reducers for support-boundary and infinite rows.
* ``_interval_integrals`` — compiled adaptive Gauss--Kronrod reductions for
  interval mass, conditional first/second moments, and auxiliary statistics,
  with scalar, multi-row, and weighted-batch entry points.
* ``_finite_reductions`` — end-to-end deterministic finite-row objective
  traversal: local Gauss--Legendre node construction, potential evaluation,
  stable normalization, and conditional-statistic reduction.

Weight validation and Kish effective counts are shared through
:func:`gibbus._observations.empirical._normalized_weights`.
"""
