"""Model representation: coordinates, the compiled G3 basis and its state.

This subpackage owns everything that describes *what* a fitted density is,
independent of any particular dataset:

* :mod:`.coords` — the fixed affine fitting coordinate ``z = (x - c)/h``.
* :mod:`.basis_spec` — the public ``BasisSpec`` and its canonical request.
* :mod:`.basis_geometry` — the fixed nonlinear geometry (tanh gates, soft
  powers, exponential rates).
* :mod:`.basis` — exact compilation of the coefficient-linear basis, its
  rational coefficient map, per-end asymptotics and leader faces.
* :mod:`.basis_eval` — Python evaluation of the columns and their
  derivatives (the reference of the compiled kernels).
* :mod:`.basis_interval` — outward-rounded interval enclosures of columns.
* :mod:`.basis_tails` — tail curvature and growth bounds toward each end.
* :mod:`.basis_normalizer` — ``log Z``, ``E[Phi]``, ``Cov[Phi]`` and
  interval-row masses of one coefficient vector.
* ``_basis_kernels`` / ``_normalizer_kernels`` — compiled column table,
  evaluation, enclosures, tail verdicts and the normalizer traversal.

Nothing here depends on observations or on the fitting pipeline, so this is
the lowest layer above :mod:`gibbus._defaults`.
"""
