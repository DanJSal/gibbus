# Fitting coordinates and the Newton controller

The statistical basis describes the potential family, its gates, support,
endpoint logarithms, coefficient restrictions and tail asymptotics. Runtime
coordinates describe how the same family is evaluated and optimized. They do
not introduce additional parameters or change the persistence format.

## Coordinate construction

`_fit.coordinates` constructs a polynomial recurrence plan from the support:

| Support | Recurrence | Argument |
| --- | --- | --- |
| Finite interval | Chebyshev | Centered and scaled to the fitted bulk |
| Lower half-line | Generalized Laguerre | Scaled distance from the lower endpoint |
| Upper half-line | Generalized Laguerre | Reflected, scaled distance from the upper endpoint |
| Full line | Normalized probabilists' Hermite | Existing centered fitting coordinate |

For a bounded support, the reference center and scale follow the fitted bulk
rather than forcing a concentrated sample into a tiny neighborhood of `-1`.
The scale cannot exceed the support half-width. The half-line reference scale
uses the weighted observation distance when point
observations are available. Its Laguerre parameter can incorporate the initial
endpoint-log amplitude. Interval objectives use the fixed geometric scale.
These reference measures improve representation; they do not guarantee that the
Fisher matrix under every fitted density is well conditioned.

A complete polynomial degree prefix with a common gate admits an invertible
triangular coefficient map. The planner changes coordinates within that prefix.
For an ungated prefix, a constant potential gauge can be added because the
normalizer cancels it. For gated prefixes, constant terms are retained exactly;
a gated constant is not a normalization gauge. Incomplete, algebraically
coupled or shared polynomial columns retain their canonical evaluator when an
independent change would alter their span or equality constraints. Nonpolynomial
columns retain their existing evaluators.

The map is constructed with rational arithmetic and retains high and low
binary64 parts for coefficient conversion. Constraint rows, asymptotic
functionals and faces transform with the same map. Native values and their first
four derivatives use the normalized three-term recurrence, with the gate product
rule applied afterward. Point sufficient statistics are recomputed directly from
retained observations. Normalizer means and centered covariances are accumulated
directly in the numerical basis; an already rounded monomial covariance is not
rotated afterward.

Point, interval and coupled mixture objectives expose the same observation and
rebinding protocol. Private component coordinates can change independently;
shared coefficient coordinates remain shared. Joint point likelihoods rebuild
per-observation features, and point M-steps rebuild responsibility-weighted
statistics. Interval probabilities continue to use the original observed rows.
Point normalization uses the dimensionless fitting table for recurrence plans.
Interval integration preserves the established coordinate dispatch so narrow
observed widths are not independently rounded by a new affine transformation.

## One Newton lifecycle

`_solve_basis` prepares a coordinate binding and uses one compiled Newton
lifecycle. A failed numerical representation, unresolved optimization or rejected
canonical conversion can retry the original coordinates through that same
controller. No model family is changed. All attempts share face handling,
curvature exchange, line search, rejection handling, deadlines and convergence
classification; completed attempts contribute to the result's work counters.
There is no recursive dispatch through a separate Laguerre fitting routine.

The loop computes an interior-point quadratic endpoint. For scalar inequalities
with a factorizable metric, a projection kernel can supply a better endpoint.
Its nonnegative-least-squares dual must meet the common quadratic gap target in
the original metric and constraints. The projection replaces the conic endpoint
only when it improves that checked bound and supplies a usable descent step or
resolves stationarity. An infinite or insufficiently resolved bound retains the
conic endpoint. Interior-point counters record native work; SciPy does not expose
the NNLS iteration count.

Rejected endpoints are resolved across all component blocks through the same
separator. Full endpoint cuts replace the fast gate's approximate cuts instead
of accumulating both sets of nearly duplicate contact rows. The former polynomial-versus-general cut scheduling rule is absent.
Ambiguous resolved contacts can use the separator's exact arithmetic capability
inside the native gate as well as in the outer exchange. Unresolved cells and
inconclusive tails remain inconclusive.

When new contact constraints are still being discovered, the exchange aims for
the configured Newton tolerance. The existing certified tolerance and approximate
accuracy floor govern the attainable numerical limit and exchange exhaustion.
Their values and the shape feasibility floor are unchanged. Active-row
multipliers use Hessian-scaled coordinates so a tiny coefficient that has a
material likelihood effect is not mistaken for a vanishing tail amplitude. Likelihood-checked
row repairs invalidate stale optimization bounds. Convex coefficient objectives
can use the shared guarded stationarity polish at interior optima and resolved
curvature contacts. At a moving interior contact, eliminating its stationary
location adds the multiplier times the third-derivative outer product divided
by the fourth derivative to the likelihood Hessian. Degenerate or unresolved
contacts retain the incumbent. Predicted improvements larger than the configured
Newton tolerance remain the responsibility of the main solver, preserving its
warm-start policy. Singular endpoint derivative rows are skipped. Small feasibility restoration, full shape checks,
likelihood accounting and the existing bound gate apply to every correction.

## Canonical output

After fitting, coefficients and rows are mapped back to the statistical basis.
The canonical face is reconstructed from its suppressed leaders and fixed log
amplitudes. Both normalizer and separator use that face so a coefficient
rounding residual does not invent a surviving asymptotic atom. Canonical
likelihood accounting is checked against the numerical likelihood using the
normalizers' error estimates, and canonical shape is checked again. A failed
shape check cannot retain a converged status.

Durable format 2 and public basis requests are unchanged. Runtime recurrence
plans are not serialized. This implementation does not add absolute-power bulk
terms, nonsmooth knots or automatic tail-family comparison.
