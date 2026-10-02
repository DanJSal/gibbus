"""The fitting pipeline: input normalization through to converged components.

Ordered roughly by pipeline position:

* :mod:`.inputs` — the public-boundary validation and canonicalization seam.
  No other module performs user-level validation.
* :mod:`.objective` — likelihood geometry (NLL, gradient, Fisher and observed
  Hessian) of one normalized state for point and interval observations.
* :mod:`.natural_objective` — point/interval objectives, fixed-degree and
  automatic-degree fit drivers.
* :mod:`.conic_qp` — finite cone descriptions of nonnegative full curvature.
* :mod:`.conic_newton` — the conic Newton solver and its certificates.
* :mod:`.separation` — exact-arithmetic full-curvature separator.
* :mod:`.degree` — information-based automatic polynomial-degree selection.
* :mod:`.mixture` — multi-component helpers: E-steps, initialization
  candidates, identifiability diagnostics, mixture state packing and modes.
* :mod:`.natural_mixture` — the mixture EM, joint Newton polish and
  multi-start.
* ``_conic_kernels`` / ``_curvature_certificate`` / ``_mixture_kernels`` —
  compiled interior point and Newton loop, curvature certificate, and
  point-statistic/posterior/joint-information reductions.
"""
