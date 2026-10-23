"""Observation providers: the seam between raw data and the likelihood.

* :mod:`.empirical` — canonical weights and Kish effective counts.
* :mod:`.intervals` — canonical interval representation, validation, exact
  duplicate collapse and the nonparametric (Turnbull) bound helpers.

The G3 likelihoods consume these through :mod:`gibbus._fit.basis_objective`;
sufficient statistics of a basis are computed there.
"""
