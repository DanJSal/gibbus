# Production Laguerre fitting and curvature contacts

The production degree-12, 3,000-point lognormal request passes with direct
Laguerre evaluation and tighter curvature contact arithmetic. Its original
likelihood and mean assertions remain in place. The expected-failure marker
is removed, and the regression also checks reported likelihood accounting and
a serialization round trip.

## Implementation

- `_fit/laguerre.py` constructs an equivalent generalized Laguerre basis for
  single-component point fits on a lower-bounded half-line, at degree eight
  or higher, with ungated polynomial/log columns and no explicit tail family.
  The family and number of coefficients do not change. The reference measure
  uses the weighted sample mean and the warm start's endpoint-log amplitude.
- `_model/_basis_kernels.pyx` evaluates this basis and its first four
  derivatives by recurrence. The native normalizer computes potentials,
  means and centered Fisher covariances from those values. Empirical
  statistics are recomputed directly from retained observations, rather than
  transformed from monomial moments. Retaining those observations costs two
  sample-sized arrays while the point objective is alive.
- The normal production controller still performs curvature exchanges,
  growth/endpoint constraints, face selection and release, warm starts and
  restarts. Its scalar projection QP operates in the Laguerre coordinates.
  Exact maps reconstruct the polynomial representation for constraints.
- Polynomial curvature point bounds use grouped Horner evaluation and
  extended coefficient-map accumulation. A two-part coefficient map retains
  representation error below the initial binary64 rounding. Exact unit maps
  have no coefficient-formation error in this point calculation.
- An otherwise resolved ungated polynomial/log search can evaluate an
  ambiguous contact point using exact rational arithmetic, with directed
  binary64 conversion. This provides a precision fallback without relying
  on extra `long double` precision. It cannot promote unresolved cells or an
  inconclusive tail. The global verdict remains numerical shape evidence.
- A tightly enclosed minimum can be more accurate than a separately summed
  monomial point value. When that value rounds outside the enclosure, its
  reported value and cut-ranking value use the enclosure midpoint.
- Final coefficients, rows and face records are restored to canonical
  coordinates. The canonical density is normalized again and its likelihood
  must agree within the normalizers' numerical error estimates. Its curvature
  is checked again. Public model records and persistence format 2 are
  unchanged; the runtime Laguerre plan is not serialized.

The optimizer stopping thresholds, likelihood assertions, parameter count,
shape feasibility floor, cell budgets and tail acceptance policy are unchanged.

## Verification

The focused kernel/objective/search checks cover independent Laguerre
derivative identities, exact polynomial equivalence, direct likelihood and
Fisher agreement, rational enclosure containment, and conservative unresolved
verdicts. The existing gated full-span acceptance case is included.

The separately checked fixed-family production solve uses the same stored
degree-6 seed and convex lift as the successful experiment, but now uses the
production normalizer, solver, face machinery and recurrence kernel. Its
canonical result is checked with independent adaptive quadrature, exact
rational Sturm counting and 80-digit stationary-point refinement. The adjacent
`laguerre-production-validation.json` records the measured likelihood,
Fisher difference, stopping bound, curvature minimum and unchanged floor.

Affine objective checks cover unit scale, a million-fold scale change with a
shifted endpoint, and an extreme origin/scale ratio that dispatches through
the normalizer's fitting-coordinate table. Canonical and Laguerre likelihoods
agree in all three cases.

The current full-suite and installed-wheel results are recorded in
`development-status.md` and the adjacent validation logs.

The broad sweep recorded 6,074 passes, nine minimal-kernel fixture failures,
two skips and one expected failure across 6,086 cases. After the optional-field
default was corrected, the affected source suites passed 284 tests, including
all nine failures. The installed wheel passed 136 tests. The reconciled
unique-case outcomes are 6,083 passes, two skips and one Laplace xfail; this
is a sweep plus targeted rerun, not a fresh all-green full-suite invocation.

## Observations and remaining limits

1. The earlier production change addressed point bounds while leaving the
   fitting objective in monomial coordinates. It did not test the successful
   combination. This implementation supplies both parts.
2. Early broad runs were stopped after buffered progress output appeared to
   indicate a stalled gated fit. That output was insufficient to establish a
   per-test runtime regression; later instrumentation showed why it could be
   misleading. Exact coefficient handling was conservatively confined to the
   specialized polynomial point calculation; established general cell and
   normalizer bounds are retained. The gated acceptance test passes.
3. Orthogonality is relative to a reference measure. An exploratory narrow
   lognormal sample initialized from an arbitrary Gamma seed still produced
   a scaled Fisher condition near 9e15 and a non-descent outcome. This does
   not establish failure of its ordinary public warm-start path, but it
   demonstrates that Laguerre coordinates alone do not guarantee good
   conditioning for every initialization. General cold-start improvements
   remain numerical-hardening work.
4. The directly evaluated fitting path is currently scoped to supported
   single-component half-line point models. Gated bases, interval objectives,
   coupled mixtures and other supports retain their existing optimization
   paths. The separate degree-4 two-linear-tail Laplace target remains open.
5. Returned canonical coefficients can still be cancellation-sensitive.
   Final likelihood and shape checks guard the conversion; this work does
   not replace every public evaluator or persistent representation with an
   orthogonal basis.
6. Linux/CPython 3.12 is verified here. Exact contact arithmetic is portable,
   but the full Windows/macOS and supported-interpreter CI matrix has not
   been established by this run.
7. Broad verification found minimal internal kernel fixtures without the
   optional coordinate-plan field. A missing field now selects canonical
   evaluation, preserving that construction contract. The installed wheel's
   series, persistence and new coordinate tests cover the correction.
8. One parallel wheel build produced truncated assembler output. A sequential
   optimized build without debug symbols succeeded; all 14 extensions then
   imported and 136 installed-wheel tests passed. The earlier assembler
   failure's cause was not established.

## Reproduction

Build the extensions, then run:

```bash
python -m pytest -q tests/test_laguerre_coordinates.py
python -m pytest -q tests/test_numerical_regressions.py::test_high_degree_fit_retains_likelihood_accuracy
python -m pytest -q tests/test_basis_selection.py::test_explicit_full_is_returned_when_it_converges
python -m pytest -q -n 2
```

The public numerical regression generates its observations with
`numpy.random.default_rng(0).lognormal(0.0, 1.0, 3000)` and exercises the
complete public fitting request without a saved prototype checkpoint.
