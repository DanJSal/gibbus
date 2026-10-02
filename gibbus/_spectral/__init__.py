"""Spectral CDF/PPF representation, query-time runtime, and tail inversion.

* :mod:`.cdf` — adaptive Chebyshev panel CDF with Bernstein positivity
  certification and analytic panel antiderivatives.  Degrades rather than
  raises, and records how well it resolved.
* :mod:`.ppf` — monotone inverse in logit-probability coordinates.  Raises on
  budget exhaustion, because a silently non-monotone quantile function is
  worse than a failure.
* :mod:`.runtime` — packs both into plain scalar/1-D fields for the fitted
  state and rebuilds the compiled evaluators on load.
* :mod:`.tail` — asymptotic inversion below ``TAIL_ASYMPTOTIC_P``, where the
  stored CDF holds no significant digits.
* ``_panel_kernels`` — compiled small-panel transforms, Clenshaw validation,
  integration, and differentiation used during post-fit construction.
* ``_cdf_eval`` / ``_ppf_eval`` / ``_certify`` — compiled spectral query
  and Bernstein-certification kernels.
* ``_tail_integrals`` — reusable compiled exact-tail QUADPACK callback.

This subpackage depends only on :mod:`gibbus._defaults`, so it can be built
and tested against any density callable.
"""
