# High-level review of the persistent high-degree failure

Date: 2026-10-23. This extends the
[conditioning investigation](degree12-conditioning-investigation.md).

This is an earlier investigation snapshot. The retained implementation and
current outcomes are in the [projection resolution checkpoint](solver-projection-resolution.md).

## Assessment

The evidence points to shared numerical weaknesses concentrated in demanding
high-degree fits, especially on half-lines. The original lognormal sample is
not the only trigger. This is not evidence of a library-wide breakdown:
automatic-degree controls converged, and some high-degree public requests
recovered through their restart path. No optimizer changes are retained in
this review, and the original degree-12 regression remains unresolved.

Three layers should be separated:

1. Ill-conditioned derivative and quadratic-subproblem representations.
2. Feasibility repairs that are small in coefficient units but materially
   change the likelihood.
3. Finite curvature-exchange budgets at moving active contacts, which can
   stop otherwise numerically resolvable fits.

## The fixed-family point problem is convex

For one component and point data, with the basis and support fixed,
`NLL(theta) = theta . mean(Phi) + log Z(theta)` has Hessian `Cov(Phi)`, which
is positive semidefinite. The log-concavity constraints are coefficient-linear
inequalities `Phi''(x) . theta >= 0`, one for every support point. Endpoint-log
amplitude constraints are also linear. The domain of finite normalizers is
convex (by Holder's inequality).

Consequently, within each fixed requested family, genuinely different local
likelihood minima are not the explanation. Restarts can help a numerical
algorithm, but should not be interpreted as escaping statistical local minima.
Automatic endpoint/span selection is a separate discrete decision, and this
argument does not extend to observed mixture or censored-data objectives.

The non-log-concavity of the generating lognormal population concerns model
misspecification, not convexity of this coefficient-fitting problem.

## Breadth controls

The internal controls use 1,200 seeded samples, a fixed minimal monomial span,
disabled endpoint logarithms, one cold `_solve_basis` call, and no collapsed-
leader restart. Each case has a three-second cooperative budget. They isolate
the solver and do not reproduce the complete public selection/restart path.

For seed 0:

| Population | Degree 4 | Degree 6 | Degree 8 | Degree 10 | Degree 12 |
| --- | --- | --- | --- | --- | --- |
| Normal, full line | converged | converged | converged | converged | exchange limit |
| Exponential, half-line | converged | converged | exchange limit | exchange limit | non-descent, infinite bound |
| Gamma(shape 2), half-line | converged | converged | converged | exchange limit | budget expired |
| Lognormal(0, 1), half-line | converged | converged | exchange limit | non-descent, infinite bound | non-descent, infinite bound |
| Uniform, bounded | converged | converged | converged | converged | exchange limit |

The exponential, gamma and uniform populations are log-concave, as is the
normal population. Thus population non-log-concavity is not necessary for
these difficulties.

A second seed reproduced degree-12 cold non-descent with infinite bounds for
exponential, gamma and lognormal data. All degree-4 and degree-6 cases in both
seeds converged. The lognormal degree-8 seed-1 case also converged, despite an
earlier infinite-bound subproblem. A bad intermediate subproblem does not
necessarily determine the complete solve's outcome.

There are 40 completed bounded internal-control records: 25 for seed 0 and
15 for seed 1. A budget expiration is not a completed convergence failure.
Processes used one BLAS/OpenMP thread each; some independent controls ran
concurrently. Their elapsed times are not comparative performance benchmarks.

## Public-path controls

The public controls keep the same fixed support, minimal expansion and disabled
logs. They use `Distribution.fit`, one component, seed-0 data and a ten-second
budget, allowing the ordinary nested-degree and face restart paths.

| Population | Explicit degree 12 | Automatic degree |
| --- | --- | --- |
| Normal | converged | converged, degree 2 |
| Exponential | budget expired | converged, degree 2 |
| Gamma(shape 2) | budget expired | converged, degree 6 |
| Lognormal(0, 1) | budget expired | converged, degree 2; one-dimensional final face |
| Uniform | converged approximately | converged, degree 2 |

These results show that restarts recover some cold-start limitations and
automatic degree selection avoids difficult high degrees in these fixtures.
They do not show that the three timed-out degree-12 requests can never
converge. They also do not establish behavior of all defaults: automatic
endpoint-log and expansion selection were deliberately disabled here.

## Conditioning is substantially representation-dependent

For an exact Exp(1) population and monomials `x**j`, `j = 1,...,d`, the Fisher
matrix is known analytically:

`H[i,j] = (i+j)! - i! j!`.

After Jacobi normalization its numerical condition estimates are approximately
`2.1e3` at degree 4, `1.3e7` at degree 8 and `7.6e10` at degree 12.
The identical polynomial span expressed in Laguerre polynomials has an exact
identity covariance matrix under Exp(1). The included rational-arithmetic
control verifies that congruence exactly, so its condition number is 1.
Changing coordinates does not change the represented density family.

This demonstrates that a huge monomial Fisher condition number does not by
itself prove statistical non-identifiability or an intrinsically unsuitable
model. It can be a numerical-coordinate problem.

It also explains a limitation of late whitening. Forming the degree-12
Laguerre covariance by transforming the already-rounded monomial covariance
in ordinary double precision gives a maximum error of about `2.6e-7` from the
exact identity in this control. That is far above the `1e-12` Newton target.
A promising implementation direction is to evaluate and accumulate derivative
information directly in stable solver coordinates, rather than relying solely
on rotating an ill-conditioned covariance after its entries have been rounded.
The current compiled face evaluator restricts the assembled full Fisher matrix
afterward. A new coordinate path would need careful gradient, curvature-row,
normalization, and face bookkeeping; an arbitrary rotation is not a complete
fix.

As a counter-control, the native QP solver does certify the analytic
zero-gradient Exp(1) population optimum at all tested degrees 2 through 14.
High conditioning alone is not sufficient to force failure. Gradient
directions, active constraints, initialization and their numerical scales
also matter.

## A rank-cutoff mechanism in the exponential control

The degree-12 cold exponential control has an estimated normalized Hessian
condition number around `3.75e14`. A NumPy reconstruction of the native
eigenvalue cutoff gives a smallest eigenvalue about `2.8055e-14` and cutoff
`2.8210e-14`. One positive direction lies just below the cutoff. Projecting
the returned subproblem stationarity right-hand side onto that discarded
direction gives a residual around `9.23e3`, against a residual threshold
around `0.197`.

The native certificate uses a truncated Hessian pseudoinverse and refuses
the dual bound when the stationary system is inconsistent at its threshold.
The reconstructed values explain how a real but poorly resolved direction can
produce an infinite bound. They are a NumPy reconstruction, not a dump of the
native Jacobi eigenvectors; exact native rounding may differ.

The independently solved frozen exponential QP, using a 200-bit significand,
has positive final scaled slacks, model value about `-117.2441`, directional
derivative about `-234.4882`, and gap about `1.39e-13`. This corroborates the
earlier lognormal subproblem controls: a finite well-defined descent problem
can be lost by the numerical representation. These QP endpoints are not
accepted fitted densities and still require full curvature and likelihood
checks.

This is not a recommendation to lower the rank cutoff blindly. Forcing an
inverse through unresolved directions can generate unreliable steps or bounds.

## Confirmed loss of progress in feasibility polishing

An instrumented eight-second run of the original 1,200-point regression found
the following post-Newton repair:

| Quantity | Value |
| --- | ---: |
| Original run status | `line_search_failed` |
| Original decrease bound | infinite |
| Coefficient movement, Euclidean norm | `2.0868e-13` |
| Movement in the repair's flattening scales | `3.3079e-14` |
| Minimum normalized row slack before | `-6.4525e-15` |
| Minimum normalized row slack after | `+6.4463e-16` |
| NLL before | `2.7589513404857513` |
| NLL after | `2.782025578131127` |
| NLL increase per observation | `0.02307423764537564` |
| Local quadratic term `0.5 delta.T H delta` | `0.013401117531390538` |

Thus a repair accepted as tiny in coefficient/row-flattening units materially
worsens the fitting objective. This is not an NLL rounding-scale difference.
The observed repair enforces the retained rows but undoes likelihood progress.
It supplies a concrete invariant to address before adding more restarts.

The observed run was already unconverged with an infinite bound. This audit
does not demonstrate that an incorrect converged density was accepted.
Separately, the coordinator retains the pre-repair status and bound after
recomputing the objective at a repaired point. Any change here must ensure
that status and bounds are valid for the point actually returned, rather than
assuming coefficient-smallness makes a repair harmless.

A second recorded repair improved NLL by about `1.34e-6`; repairs are not
uniformly harmful. The required safeguard must assess the actual change, not
disable all restoration or tolerate invalid log amplitudes/shape violations.

## Coordinate-presentation probe

An additional frozen-QP probe uses original, diagonal and whitened coordinates,
renormalized rows and unit interior blocks. Whitening can produce a finite
native QP gap for these transformed rounded inputs, while mapping the endpoint
back and evaluating its original quadratic model in double precision gives
noticeably different values. These runs also change scaled interior starts;
they are not a controlled claim that one coordinate change alone fixes the
native solver. Congruence rounding and cancellation in the original model
evaluation both need accounting before interpreting a transformed certificate.
The exact rational Laguerre control is the cleaner evidence for the underlying
representation problem.

## Implementation priorities

1. Guard feasibility repairs using their objective/information effect, and
   recompute or invalidate any affected convergence claim. Cover the observed
   tiny-coefficient/large-likelihood change with a focused regression. Preserve
   exact structural constraints and full shape checking.
2. Develop stable derivative and QP coordinates together, including transformed
   sufficient statistics, covariance accumulation, constraint rows, dual residuals
   and exact-face transitions. Use analytic exponential controls and captured
   QPs as independent references. Do not equate coordinate conditioning with
   statistical identifiability.
3. Treat finite-bound contact/exchange exhaustion as a separate recovery task.
   Preserve the evidence that public restarts already recover some such cases;
   avoid increasing every fit's iteration budget indiscriminately.
4. Validate the requested degree-12 regressions unchanged, the exponential and
   gamma controls, unit-scale invariance, lower-degree fits and general-column/
   mixture cases before a broader suite and installed-wheel run.

The work is justified by shared invariants and representations rather than
special handling of one lognormal sample. Mixture and interval paths share
parts of this machinery, making them relevant validation targets; this review
does not establish the same failures in those paths. No optimizer modification
or full-suite rerun was performed in this high-level pass.
