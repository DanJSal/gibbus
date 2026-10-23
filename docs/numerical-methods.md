# Numerical methods and performance

This document describes implementation details that matter for accuracy, complexity, or interpretation. Routine users do not need these details to fit or query a distribution.

## Interval-censored likelihoods

Point and interval fits use different numerical objectives. Point-data fitting evaluates the exact log-likelihood, with exact sufficient-statistic compression where available. Interval fitting evaluates each observed probability mass directly in log space rather than subtracting nearby CDF values or substituting interval midpoints.

Each row's log mass `log P(B) = log A_B - log Z` is integrated by the same tail-bounded domain integrator as the normalizer, on that row. The objective accumulates the conditional moments and the observed Hessian `F - C` (with a saddle-free reflection when it is not positive definite) without materializing a large row-by-node tensor. Identical rows are merged with summed weights; a whole-support row short-circuits to probability one.

Mixture E-steps likewise use component probability mass over each interval. Very coarse censoring can leave a component decomposition statistically unidentified. The fitter compares the observed interval likelihood with the unrestricted Turnbull likelihood on censoring endpoints and uses the endpoint graph to count independent probability coordinates. An overparameterized mixture that reaches the nonparametric maximum can therefore be rejected as non-identifiable instead of being treated as a successful decomposition.

Prefer point observations where available, and prefer narrower censoring intervals when within-bin component shape matters.


## The potential and its shape constraint

Each component's negative log density is a coefficient-linear potential `q(x) = Phi(x) . theta` over named functions of a fixed robust fitting coordinate `z = (x - c) / h`: monomials up to the bulk degree, finite-endpoint logarithms with nonnegative amplitudes, and, on request, noninteger soft powers and directional exponentials confined to their side by fixed tanh gates. Columns and their derivatives to order 4 are evaluated directly (compiled), not through a polynomial representation.

Log-concavity (`q'' >= 0` on the whole support) is enforced by cut rows `Phi''(x*) . theta >= 0`: the tails are settled by their leading asymptotic terms with uniform remainder bounds, the compact remainder is searched with outward-inflated interval enclosures of `q''` that fail closed, and each located minimum adds a permanent row. Proper outward growth at infinite ends is enforced on the leading-growth rows. The Newton subproblem uses only scalar cut and structural rows; it does not use quadratic contact blocks or derivatives beyond the second order for its constraints. The interior-point method requires strictly positive row slacks during its solve, while the limit density is permitted to satisfy `q'' = 0` at contact points. The global separator continues to certify nonnegative curvature over the entire support. The shape verdict is numerical evidence, not a proof; a fit whose verdict is inconclusive reports `uncertified`.

## Normalization and moments

`log Z`, `E[Phi]` and `Cov[Phi]` share one compiled quadrature traversal over cells resolved by interval enclosures, with tails bounded analytically. Each cell's rule is fixed by the integrand's analytic structure on it: Gauss--Kronrod 10/21 where `e^{-q}` is analytic, nested tanh-sinh on a cell ending at an enabled endpoint log (an endpoint power singularity). A cell whose rule does not settle within its share of the tolerance is split; an integral that cannot be resolved within its budget raises rather than falling back to another rule. Moments are shifted and centred for stability.

## Fitting

Eligible complete polynomial prefixes use support-appropriate recurrence
coordinates: Chebyshev on finite supports, generalized Laguerre on half-lines,
and Hermite on the full line. Values, four derivatives, empirical statistics,
potentials, means and Fisher covariances are evaluated directly in those
coordinates. Gates, shared coefficients and the requested statistical span are
preserved. Canonical coefficient conversion reconstructs the face and rechecks
likelihood accounting and shape; persistence retains the canonical basis.

Resolved polynomial curvature contact points use a grouped Horner enclosure.
The coefficient map is accumulated with extended precision, including its
representation error. An otherwise resolved search whose point bound still
straddles the feasibility floor can use exact rational polynomial/log arithmetic
at that point. This fallback does not certify unresolved cells or tails and
does not change the feasibility or optimizer stopping tolerances.

The point likelihood depends on the sample through sufficient statistics
`mean(Phi(x_i))`, computed directly in the numerical basis. The fitter uses one
damped Newton controller over exact faces, coefficient-linear curvature cuts
and an Armijo line search on the true likelihood. Runtime polynomial coordinates
use support-appropriate Chebyshev, Laguerre or Hermite recurrences within the
requested span. Point, interval and coupled objectives share this coordinate
interface; [fitting coordinates](fitting-coordinates.md) describes the maps,
gated subspaces and canonical output checks. Convergence targets and the shape
feasibility floor retain their configured values. NLL tolerances are absolute,
so changing data units does not change their scale. Status values are
`converged`, `converged_approximately`, `non_descent`, `line_search_failed`,
`iteration_limit` and `uncertified`.

Automatic degrees climb a ladder of nested bases by an omitted-statistic score test (chi-square calibrated, with nuisance projection); each rung starts from the lifted smaller fit, and a lifted start on the larger basis's boundary is moved into the interior along a certified convex direction at a cost of at most `1e-3` relative likelihood. Endpoint logarithms are selected by a one-sided likelihood-ratio test. A ladder rung that converges from no start ends the ladder at the smaller degree.

A scalar-inequality Newton model admits a projection kernel when its metric
can be factored. A Cholesky coordinate change reduces its dual to nonnegative
least squares. The shared loop compares this endpoint with its interior-point
endpoint using the original-model quadratic bound. Projection must meet the
common gap target and improve that checked bound before replacing the conic
endpoint. Kernel eligibility depends on quadratic problem properties, not the
polynomial family or observation type. All candidates retain the same curvature
and likelihood acceptance rules. Native iteration diagnostics count interior
solves; SciPy does not expose NNLS's iteration count.

Row-feasibility repairs are also checked against the actual likelihood. A small
coefficient displacement does not imply a small density change. A repaired
endpoint must obtain a fresh optimization certificate before it can be reported
as converged. Comparing a larger fit with a known nested smaller fit is an
optimization consistency check: for fixed-basis single-component point data the
likelihood and shape-constrained coefficient problem are convex.

Mixtures run SQUAREM-accelerated EM whose M-steps are the single-component fitter on responsibility-weighted statistics, each component in its own responsibility-weighted coordinate, followed by a joint Newton polish with the observed information. Endpoint-log amplitudes are shared physical parameters across components.

## Performance architecture

- **Compiled kernels**: basis evaluation, the normalizer traversal, the face Newton loop with its objective, the interior-point subproblem, the curvature search, mixture reductions, spectral construction/query and exact tails are Cython; Python orchestrates.
- **Scalar quadratic refinement**: the common native controller can call SciPy NNLS to refine a scalar-inequality subproblem when it supplies a better resolved original-model bound. Eligibility depends on the quadratic metric and constraints. Fitting creates no worker pool.
- **Fitting cost**: point optimization cost is essentially independent of sample count after the sufficient statistics; model-side cost grows with the number of basis columns and cut rows. Interval fitting is observation-dependent (one row integral per merged row). Mixture EM refits all *K* components per iteration. Runtime depends on the basis, curvature contacts, interval rows and mixture complexity. Historical benchmark timings predate the current coordinate refactor and are not current performance guarantees (see [Limitations](limitations.md)).
- **Moment caching**: the packed fitted state caches raw moments up to order 32, log-distance moments and cumulants; higher moments are integrated on demand from the potential.

- **Subsampled model selection**: With `n_components="auto"`, the two stages that choose the component count — KDE mode counting and the log-concave BIC sweep — run on a stratified subsample of large datasets. The default fit RNG is seed 0 when `rng=None`, so repeated fits are reproducible; pass a `Generator` or explicit integer to choose another stream. Subsampling can still change the selected *K* relative to a full-data sweep, but the final model for the winning *K* is refitted on every observation. Disable with `auto_k_subsample=False` when exact full-data component-count screening is preferred. Inspect `selection_diagnostics` to see the proposal, subsample size, candidate scores, and winner.
- **Interval-row merging**: identical interval rows are merged with summed weights before fitting, which is exact and shrinks binned datasets substantially.
- **Spectral CDF/PPF**: Each fitted component stores compact adaptive Chebyshev panels for its CDF and inverse CDF. Finite, one-sided, and doubly unbounded supports are compactified to a finite coordinate; CDF panel densities are positivity-certified before analytic integration, and PPF panels are built from a certified non-negative inverse derivative in logit-probability coordinates. Construction uses cached direct Chebyshev-Lobatto transforms, compiled small-panel transform/Clenshaw/calculus kernels, compiled Bernstein certification (exact Chebyshev-to-Bernstein conversion with a rigorous rounding margin), and batched safeguarded Newton inversion of source CDF panels; endpoint-indistinguishable targets are canonicalized to the panel endpoint to preserve monotonicity at float64 resolution. Query-time evaluation uses compiled Cython kernels with degree classes 16, 24, and 32 and SIMD-oriented bulk paths for large arrays. Those transposed Clenshaw loops carry explicit no-alias and compiler-specific vectorization hints for GCC, Clang, and MSVC; the actual SIMD width and instruction selection remain compiler/target dependent, and portable wheels do not opt into machine-specific ISA flags. Outside the stored inverse-panel range, the raw spectral inverse uses compiled monotone bisection of the packed spectral CDF. For probabilities at or beyond the extreme-tail threshold, the public quantile API then re-solves from the fitted potential in log-probability space, avoiding reliance on CDF digits that are no longer representable. No interpolation grid or retained parametric tail-extension model is stored. Mixtures finalize components serially; the library does not create hidden worker pools.

- **Tail-region cost**: body queries run through compiled spectral panels and are substantially cheaper than direct tail quadrature. `logsf` (and therefore `sf`, hazard, and other survival quantities) switches to exact upper-tail quadrature once the spectral survival probability falls below `SF_HANDOVER_P`; `logcdf` makes the symmetric switch in the lower tail, while `cdf` itself remains an absolute-probability spectral query. Public quantiles are re-solved from the fitted potential below `TAIL_ASYMPTOTIC_P` (and symmetrically above `1 - TAIL_ASYMPTOTIC_P`). Exact component tails integrate the G3 potential in log space. For mixtures, gibbus integrates each component tail separately and combines the component log masses with a stable compiled reduction, rather than evaluating the full mixture potential at every quadrature node. Exact-tail work is still per query and scales with the number of components, so it is substantially more expensive than cached panel evaluation once the handover is crossed.



- **Auto component selection** (`n_components="auto"`): First estimates the number of data modes via a KDE bandwidth sweep, then fits lightweight log-concave models for a focused range of *K* values centered on the KDE mode count (extended upward while the largest *K* tried still scores best) and picks the lowest BIC. On large inputs the KDE sweep and the log-concave sweep both run on a stratified subsample (see `auto_k_subsample`), leaving screening cheap relative to the single full log-concave EM run for the winning *K*. Use `k_max` to narrow the search range if desired.
