"""The fitting pipeline: input normalization through to converged components.

Ordered roughly by pipeline position:

* :mod:`.inputs` — the public-boundary validation and canonicalization seam.
  No other module performs user-level validation.
* :mod:`.budget` — cooperative wall-time budgets shared by one fit.
* :mod:`.tail_estimation` / :mod:`._empirical_tail_likelihood` — empirical
  within-family tail parameters (the quadrature runs in
  ``_tail_likelihood_kernels``).
* :mod:`.basis_objective` — weighted point and interval likelihood geometry
  (NLL, gradient, Fisher and observed Hessian) of the basis.
* :mod:`.coordinates` — runtime recurrence coordinates, direct observation
  rebinding and canonical output validation.
* :mod:`.basis_constraints` — coefficient-linear constraint rows (log bounds,
  curvature cuts, tail rows) and their cone description.
* :mod:`.curvature_search` — where ``q''`` may fall below zero (cut points).
* :mod:`.conic_qp` / :mod:`.conic_newton` — cone descriptions, the shared
  Newton policy and the preconditioned subproblem; :mod:`.linear_qp` — the
  projection solve for scalar-inequality Newton models.
* :mod:`.basis_newton` — the face Newton fit over the constraint rows.
* :mod:`.boundary` — likelihood-ratio selection of endpoint logs.
* :mod:`.degree` — omitted-statistic score-test policy and PSD helpers.
* :mod:`.basis_selection` — degree ladder, endpoint logs and span expansion
  of single fits.
* :mod:`.mixture` — mixture initialization candidates, E-steps,
  identifiability diagnostics, packing and modes.
* :mod:`.basis_mixture` — mixture EM, joint polish, growth inside a mixture
  and component-count selection.
* ``_conic_kernels`` / ``_basis_newton_kernels`` / ``_curvature_kernels`` —
  compiled interior point, fused face Newton loop and curvature separation.
"""
