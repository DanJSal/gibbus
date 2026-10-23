# Scalar-projection recovery and checked feasibility repairs

Date: October 22, 2026. Development checkpoint; not a release qualification.

## Retained implementation

- Row-feasibility repairs are evaluated in the actual objective. A correction
  that worsens likelihood beyond rounding is rejected even when its coefficient
  displacement is tiny. A changed endpoint cannot inherit its predecessor's
  convergence certificate: a previously converged endpoint is re-certified at
  its repaired coordinates, and other repairs require another solve.
- Ill-conditioned half-line point fits with ungated monomial/logarithm columns
  have a scalar-projection Newton path. Its dual is nonnegative least squares;
  it avoids a large interior-point system for nearly redundant curvature rows.
  An unresolved projection falls back to the established native solve.
- Explicit polynomial-only half-line requests of degree 10 or greater first
  obtain a degree-6 nested seed. A valid seed is retained across unresolved
  intermediate degrees, and a likelihood-bounded interior lift is tried first.
  This changes initialization, not the requested fitted family.
- Full-line and bounded requests retain their established initialization and
  native Newton path. General gated columns, mixtures and interval objectives
  retain their native subproblem implementation.
- The nested-likelihood guard remains an optimization consistency check. A
  larger family must not report convergence materially worse than a known
  nested smaller fit. References to numerical "basins" in the active regression
  terminology are removed; this fixed-family point problem is convex.

The retained changes are Python and tests. Native kernel source and extension
experiments were restored; neither arbitrary-precision arithmetic nor a new
runtime dependency is introduced. The normalizer and full shape separator are
unchanged.

## Projection and numerical bound

For a positive-definite Newton metric `H = L L.T`, use `x = L.T theta` and
`C = B L**(-T)`. The unconstrained quadratic optimum is `u`; projection onto
`C x >= 0` has dual `nnls(C.T, -u)`. The active constraint span determines the
endpoint, even if the retained constraint list is much larger.

The endpoint is checked against the original stored quadratic rather than
treating the factorization's nearby model as exact. For nonnegative multipliers
`z`, stationarity residual `r`, and a positive numerical lower eigenvalue bound
`lambda`, the dual-gap bound uses

`z . (B theta) + ||r||**2 / (2 lambda)`.

Matrix/eigenvalue rounding allowances, residual arithmetic allowances and model
evaluation allowances are included. Residual and model accumulation use
`numpy.longdouble` where that provides extra precision. If the lower eigenvalue
estimate is unresolved, the gap is infinite; a useful descent candidate is
still distinct from a certified stopping point. These are numerical solver
checks, not formal interval proofs or global likelihood-error guarantees.

A rounding repair of the QP endpoint uses a small max-margin projection solved
by NNLS. It does not call a separate LP solver. Its objective and residual are
then recomputed. Every density-fitting trial still needs the complete curvature
search, normalizability, and the actual likelihood line search. A shape verdict
alone cannot establish optimizer convergence.

The existing solver/shape tolerance values and regression assertions are
unchanged. The projection path can report `converged_approximately` when its
checked predicted-decrease bound is within the existing `1e-7` accuracy floor;
it does not label that result fully converged. Component diagnostics expose the
status and bound. SciPy does not expose NNLS's internal iteration count, so the
subproblem iteration field continues to count native interior-point iterations.

## Validation observations

The original 1,200-point degree-12 likelihood-accounting regression passes with
its request and `2e-9` accounting tolerances unchanged. A focused 68-case batch
covering row repairs, scalar QPs, existing basis solves/selection, unit-scale
equivariance and endpoint concentration also passes. The final 11-case QP batch
includes a regression that initializes an unrelated LP solver first.

The first broad run completed 6,073 cases: 6,062 passed, seven failed, two
skipped and two expected failures, in 443.44 seconds. Its failures exposed:

1. Initialization/projection changes applied too broadly, regressing bounded
   endpoint concentration and full-line degree-8 unit equivariance. Scoping
   recovery to half-lines and retaining other initializations repairs those
   focused cases.
2. A separate LP-based interior helper depended on process-global HiGHS
   scheduler state. Four QP cases passed alone but failed after unrelated
   library tests initialized that state; the degree-12 request could also fall
   back to its old failure. The retained max-margin NNLS projection removes
   that dependency. No production fitting worker pool is introduced.

The full final-run results, installed-wheel checks and source/package checks
are recorded in the current [validation report](../review-validation.md).

The final JUnit record contains all 6,074 cases: 6,070 passed, two skipped and
two expected failures, with no failures or errors. The installed-wheel batch
passes all 106 cases and smoke imports all 14 native extensions. The final
console progress stream stopped updating before its summary; the complete
JUnit result and successful command exit are authoritative. An apparent stall
was therefore not evidence of a performance regression. An extra dispatch
restriction proposed on that basis was reverted; wheel/source Python equality
was checked afterward.

The separate 3,000-point degree-12 lognormal target (OI-E2-3) and degree-4
two-linear-tail Laplace target (OI-E2-6) remain open. The former still fails its
convergence request; its expected-failure assertion is not weakened or removed.
Neither is conflated with the passing 1,200-point accounting fixture.

## Experiments excluded

Early projection prototypes included fixture-specific dimension/likelihood
conditions and a direct degree-6 warm-start hook. Those conditions are not
retained; dispatch uses the objective/basis type, half-line geometry, and a
normalized condition estimate. Prototype success is not substituted for the
production test result.

Ungated line-search experiments allowed non-log-concave trial potentials to be
normalized. They did not supply a reliable improvement, were interrupted or
reached their budgets, and were reverted. The retained path checks full shape
before normalizing trials. Earlier whitening, damping and high-precision probes
remain diagnostic evidence only.

## Remaining work

The degree-12 family is still demanding. More stable derivative coordinates,
conditioning of exact-face/release checks, efficient treatment of moving
curvature contacts and optional face searches, and the two open acceptance
targets need further work. Cross-platform/minimum-dependency and current full
debug qualification remain outside this checkpoint. The fix addresses shared
numerical mechanisms without claiming that every high-degree request succeeds.
