# Degree-12 lognormal investigation

Date: 2026-10-23.

This is an earlier investigation snapshot. The retained implementation and
current outcomes are in the [projection resolution checkpoint](solver-projection-resolution.md).

## Outcome

The full degree-12 fit remains unresolved. No optimizer experiment is retained
in the library implementation. The conic kernel source and extension were
restored byte for byte after the selection-rule experiment. The only test-file
edit corrects an outdated docstring claiming that this case certifies; the test
request and assertions are unchanged. The existing source archive has not been
replaced by this investigation.

The investigation isolates a numerical limitation in captured quadratic
subproblems. It does not establish a converged density fit, a new production
certificate, or a timing improvement for the full request. The two existing
expected-failure targets are still open; no other remaining issue is claimed
resolved.

## Reproduction and scope

The principal regression uses 1,200 observations from
`np.random.default_rng(0).lognormal(0.0, 1.0, 1200)` and requests one component,
polynomial degree 12, and support `(0.0, np.inf)`. Endpoint-log selection remains
automatic. Runs can therefore include 13 coefficients before a zero-log face
reduces the dimension.

The original focused pytest invocation was interrupted without a test outcome.
Subsequent end-to-end diagnostic probes used the unchanged public request with
`max_fit_seconds=45` and logged each completed Newton run. Their final outcomes
are timeouts, not completed regression failures. The budget is cooperative;
one completed probe reported 47.067 seconds at its next checkpoint. BLAS and
OpenMP thread counts were limited to one for all diagnostic processes.

The unchanged-policy trace reached shape-feasible iterates with normalized
Hessian condition estimates around `1e15`, infinite decrease bounds, and
`non_descent`. This corroborates the earlier completed suite's failure, without
providing a new full-suite result.

## Population shape does not establish optimization impossibility

For the generating lognormal population, the second derivative of its
log-density is `log(x) / x**2`. It is positive for `x > 1`, so that population
is not globally log-concave. A finite point sample does not itself define a
smooth density. A constrained fit seeks a member of the requested log-concave
family; population misspecification can affect fit quality and active shape
constraints, but does not justify accepting nonconvergence or treating the
optimization problem as infeasible.

## Frozen quadratic subproblems

Three captured subproblems were solved independently as homogeneous scalar
inequality QPs. The diagnostic uses the captured double-precision Hessian,
gradient, parameters and rows, followed by Jacobi column scaling and positive
row scaling. It checks positive definiteness of the rounded scaled Hessian
before applying a primal-dual predictor-corrector method. Its reported gap is
the quadratic model value minus a Lagrangian dual lower bound, with positive
dual multipliers and positive final scaled constraint slacks.

At a 200-bit significand, the representative captured cases give:

| Snapshot | Dimension | Native directional derivative | Native gap | Diagnostic directional derivative | Diagnostic QP gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 13 | `+2.1274257e-3` | infinite | `-6.6085684` | `2.6349132e-13` |
| 8 | 12 | `+2.0924027e-11` | infinite | `-212.32975` | `6.1968677e-14` |

Negative directional derivatives indicate descent in the frozen model, not
measured decreases in the density-fitting objective. The native results were
recomputed on the same saved inputs after restoring the original extension.

The same independent algorithm hits a numerically singular linear system at
a 53-bit significand. At 80, 128 and 200 bits it solves all three captured
subproblems with positive final scaled slacks and nonnegative gaps below
`1e-12`. The 128-bit and 200-bit results agree to the displayed precision.
Here bit counts refer to significand precision, not hardware storage size.

This controlled comparison supports a precision/conditioning diagnosis. The
independent solver also has a different linear-system formulation from the
native solver, so it does not identify one native line as the sole cause.
These are certificates for frozen rounded QPs only. Their endpoints have not
been established as globally shape-feasible fitted densities, and the
experiment does not replace normalizer, curvature or full-fit certification.
`mpmath` is used only for this diagnostic and is not added as a runtime
dependency.

## Experiments not retained

1. **Ranking uncertified QP iterates by model progress.** The native solver
   ranks equally uncertified iterates by primal residual. A temporary variant
   preferred lower quadratic model values among nearly primal-feasible
   candidates. It recovered descent directions on some captured subproblems,
   but did not certify the full fit. Its 45-second end-to-end probe timed out.
   The source and binary change was reverted. The experimental residual
   threshold is not a proposed production acceptance policy.
2. **Direct warm start from degree 6.** The smaller fit converged at NLL
   `1.4189185937789541`. Lifting it directly to the degree-12 family reached
   NLL about `1.4050` in an interior-lift run, and a reduced face reached about
   `1.4088166`. Bounds remained unresolved; the full request timed out.
   These NLLs describe intermediate iterates, not accepted fitted results.
3. **Warm start plus conditional whitening.** A temporary coordinate change
   was applied only when the normalized Hessian condition estimate was between
   `1e7` and `1e13`. The transformed Hessians could have condition estimates
   near one while subproblem bounds still became very large. The full request
   again timed out. This rules out treating a small transformed Hessian
   condition estimate as sufficient evidence that the whole solve is stable.

## Follow-up implementation targets

- Investigate the scalar-inequality QP linear system and certificate together,
  including nearly dependent curvature rows, rank cutoffs, endpoint
  reconstruction, and multiplier cancellation. Use the captured cases to
  check feasible descent and a sound finite gap before attempting full fits.
- Consider adaptive precision or a more stable factorization/refinement path
  for unresolved small subproblems. The high-precision prototype supplies an
  independent diagnostic reference, not a runtime implementation recommendation
  without further portability and cost analysis.
- Investigate whether explicit high-degree recovery should retain the best
  converged lower-degree seed across unconverged intermediate degrees. The
  current recursive path only uses its immediately returned nested result if
  that result converged. A direct degree-6 seed improves the region reached
  here, but does not by itself resolve the requested degree-12 fit.
- Review fallback selection among uncertified subproblem iterates, preserving
  a distinction between a candidate useful for progress and a point whose
  convergence is certified.

An eventual fix must pass the unchanged 1,200-point likelihood-accounting
regression, the 3,000-point high-degree likelihood-basin acceptance target,
and relevant scale-invariance and shape tests. It must not silently substitute
a lower requested degree, relax the convergence thresholds, or accept shape
feasibility as an optimality certificate. Broader suite and installed-wheel
checks belong after a candidate passes those focused checks.

## Evidence package

The accompanying investigation archive contains this report, the bounded fit
probe, the frozen-QP diagnostic, captured QP inputs, and the completed probe
logs. The selection-rule patch and build helper are included as explicitly
experimental reproduction material, separate from the library implementation.
No full suite was rerun during this investigation because no optimizer change
survived the focused experiments.
