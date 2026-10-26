# Fitting coordinates and the Newton controller

The statistical basis describes the potential family, its gates, support,
endpoint logarithms, coefficient restrictions and tail asymptotics. Runtime
coordinates describe how the same family is evaluated and optimized. They do
not introduce additional parameters or change the persistence format.

## Coordinate construction

`_fit.coordinates` constructs a polynomial recurrence plan from the support
and the observations:

| Situation | Recurrence | Argument |
| --- | --- | --- |
| Finite interval | Chebyshev | The support's center and half-width |
| Lower half-line | Generalized Laguerre | Scaled distance from the lower endpoint |
| Upper half-line | Generalized Laguerre | Reflected, scaled distance from the upper endpoint |
| Full line | Normalized probabilists' Hermite | The centered fitting coordinate |
| Data occupying a small part of a finite-ended support | Normalized probabilists' Hermite | The data's weighted mean and standard deviation |

"A small part" means a weighted standard deviation below 1/8 of the support's
own scale: the half-width of a bounded support, or the distance from a
half-line's endpoint. The support's own recurrence is badly conditioned there,
while Hermite coordinates at the data span the same polynomials. The half-line
reference scale uses the weighted observation distance from the endpoint when
point observations are available, and its Laguerre parameter can incorporate
the initial endpoint-log amplitude. These reference measures improve
representation; they do not guarantee that the Fisher matrix under every
fitted density is well conditioned.

A complete polynomial degree prefix with a common gate admits an invertible
triangular coefficient map. The planner changes coordinates within that prefix.
For an ungated prefix, a constant potential gauge can be added because the
normalizer cancels it. For gated prefixes, constant terms are retained exactly;
a gated constant is not a normalization gauge. Incomplete, algebraically
coupled or shared polynomial columns retain their canonical evaluator when an
independent change would alter their span or equality constraints.
Nonpolynomial columns retain their evaluators.

The map is constructed with rational arithmetic and retains high and low
binary64 parts for coefficient conversion. Constraint rows, asymptotic
functionals and faces transform with the same map. Native values and their first
four derivatives use the normalized three-term recurrence, with the gate product
rule applied afterward. Curvature enclosures expand each recurrence polynomial
in Taylor form at the cell midpoint, with exact jets from the recurrence and a
remainder from their magnitudes. Point sufficient statistics are recomputed
directly from retained observations. Normalizer means and centered covariances
are accumulated directly in the numerical basis; an already rounded monomial
covariance is not rotated afterward.

Point, interval and coupled mixture objectives expose the same observation and
rebinding protocol. Private component coordinates can change independently;
shared coefficient coordinates remain shared. Joint point likelihoods rebuild
per-observation features, and point M-steps rebuild responsibility-weighted
statistics. Interval probabilities continue to use the original observed rows.

Each basis retains its most recent numerical representation, keyed by the
recurrence family, origin, scale, Laguerre parameter and movable coefficient
indices. Reusing that representation avoids repeating exact basis compilation.
Observation-dependent sufficient statistics are recomputed for each objective.
The cache contains one entry and is excluded from the durable model.

## One Newton lifecycle

`_solve_basis` prepares a coordinate binding and uses one compiled Newton
lifecycle. A failed numerical representation, unresolved optimization or rejected
canonical conversion can retry the original coordinates through that same
controller. No model family is changed. All attempts share face handling,
curvature exchange, line search, rejection handling, deadlines and convergence
classification; completed attempts contribute to the result's work counters.

For scalar inequalities with a factorizable metric, the loop first tries a
projection kernel. Its nonnegative-least-squares dual must meet the common
quadratic gap target in the original metric and constraints and supply a usable
descent step or resolve stationarity. An unresolved projection falls back to
the interior-point kernel. Both kernels use the same line search, shape gate
and convergence checks. Interior-point counters record native work; SciPy does
not expose the NNLS iteration count.

Every trial point is first checked for global convexity by the curvature
search, so the normalizer only integrates log-concave potentials. A non-convex
trial is a routine line-search rejection; the cuts it reveals are queued, and
the run ends at its accepted iterate so the next run includes them. Full
endpoint cuts replace the fast gate's approximate cuts instead of accumulating
nearly duplicate contact rows. Unresolved cells and inconclusive tails remain
inconclusive.

When new contact constraints are still being discovered, the exchange aims for
the configured Newton tolerance; the certified tolerance and the accuracy floor
govern the attainable numerical limit and exchange exhaustion. Active-row
multipliers use Hessian-scaled coordinates, so a tiny coefficient that has a
material likelihood effect is not mistaken for a vanishing tail amplitude.
Likelihood-checked row repairs invalidate stale optimization bounds. Small
feasibility restoration, full shape checks, likelihood accounting and the bound
gate apply to every correction. Separately verified optima (a larger fit and
its nested smaller fit, an M-step and the joint polish) agree only to the
accuracy floor, so comparisons between them use that floor.

## Canonical output

After fitting, coefficients and rows are mapped back to the statistical basis.
The canonical face is reconstructed from its suppressed leaders and fixed log
amplitudes. Both normalizer and curvature search use that face so a coefficient
rounding residual does not invent a surviving asymptotic atom. Canonical
likelihood accounting is checked against the numerical likelihood using the
normalizers' error estimates, and canonical shape is checked again. A failed
shape check cannot retain a converged status.

Runtime recurrence plans are not serialized. A fitted component is evaluated
again in recurrence coordinates placed at its own mean and standard deviation
(stored in the runtime cache); the durable model keeps the canonical
coefficients and the canonical potential's normalizer.
