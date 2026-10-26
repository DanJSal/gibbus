"""Model representation: coordinates, the compiled basis and its state.

This subpackage owns everything that describes *what* a fitted density is,
independent of any particular dataset:

* :mod:`.coords` — the canonical coordinate ``z = (x - c)/h`` (q10/q90 rule).
* :mod:`.basis_spec` — the public ``BasisSpec`` and its canonical request.
* :mod:`.basis_geometry` — the fixed family geometry (Gaussian gates, soft
  power root, canonical exponential rates).
* :mod:`.basis` — exact compilation of the coefficient-linear basis, its
  rational coefficient map, per-end asymptotics and leader faces.
* :mod:`.basis_eval` — Python evaluation of the columns and their
  derivatives (the reference of the compiled kernels).
* :mod:`.basis_interval` — outward-rounded interval enclosures of columns.
* :mod:`.basis_tails` — tail curvature and growth bounds toward each end.
* :mod:`.basis_normalizer` — ``log Z``, ``E[Phi]``, ``Cov[Phi]`` and
  interval-row masses of one coefficient vector; :mod:`.gauss_kronrod` — its
  Gauss--Kronrod 10/21 rule.
* ``_basis_kernels`` / ``_normalizer_kernels`` — compiled column table,
  evaluation, enclosures, the structured scaled curvature and its per-atom
  rows, tail verdicts and the normalizer traversal.

Nothing here depends on observations or on the fitting pipeline, so this is
the lowest layer above :mod:`gibbus._defaults`.
"""
