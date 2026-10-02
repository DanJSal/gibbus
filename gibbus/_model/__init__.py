"""Model representation: coordinates, natural layout, state, moments.

This subpackage owns everything that describes *what* a fitted density is,
independent of any particular dataset:

* :mod:`.coords` — the fixed affine fitting coordinate ``z = d*(x - c)/h``.
* :mod:`.natural` — the affine natural-parameter layout
  ``(gamma, c_0..c_d, a_L, a_U)`` of the potential.
* :mod:`.spec` — the model specification (coordinate plus layout) and the
  first-partial descriptors of the potential.
* :mod:`.natural_state` — the live normalised model state: potential,
  mode/window, normalisation, Fisher statistics, PDF and moment access.
* :mod:`.moments` — ordinary and generalized model moments used by the
  likelihood geometry and the degree diagnostics.
* :mod:`.numerics` — observation-independent numerical helpers.
* :mod:`.vec` — vectorised low-level potential/density/partial evaluators.
* ``_state_kernels`` / ``_quad_integrals`` / ``_moment_kernels`` — compiled
  state numerics, prepared quadrature and shared power-moment/contraction
  kernels.

Nothing here depends on observations or on the fitting pipeline, so this is
the lowest layer above :mod:`gibbus._defaults`.
"""
