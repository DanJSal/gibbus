# G3-078 — curvature-row-only Newton constraints

The fitting Newton solver uses only *coefficient-linear scalar inequality rows*. For a fixed location `x`, the curvature constraint is `Phi''(x) @ theta >= 0`. Structural log-amplitude and proper-tail-growth rows remain. Globally valid curvature cuts are exchanged as the fitted coefficients change; cuts that rely on a coefficient face retain their existing face-local scope. A globally nonnegative curvature verdict is still necessary for convergence.

The quadratic `2 × 2` contact Gram blocks are removed, along with `_ContactBlock`, `_ConeDescription`, `_contacts`/`_joint_contacts`, their derivative-based construction and anchoring, and their accompanying tests. The conic subproblem is built through `_linear_representation`, and each conic block is now `1 × 1`. The fast compiled curvature-valley finder remains because it supplies **pointwise cut locations** to the Newton trial gate, not contact blocks. The generic semidefinite kernel still supports larger blocks for its other callers, but G3 fitting no longer constructs them.

The interior-point iterations operate at strictly positive scalar slacks when possible. An optimum may have `q''(x) = 0`, which is valid for convexity and must remain representable; enforcing a fixed strictly positive global curvature margin would exclude valid log-concave solutions and linear tails.

## Verification and limitations

- Focused conic, Newton, empirical-tail-integration, and constraints checks: **104 passed, 1 expected failure**.
- Independent post-rebuild scalar-row/contact-removal tests: **68 passed, 1 expected failure**.
- Stage 6 numerical blocker group: **16 passed, 1 failed**. The failure is the existing degree-3/two-linear-tail logpdf equivalence test at seed 1: maximum difference approximately `6.19e-6`, compared with its `1e-6` threshold. Both versions of the fit converge, are certified, and differ in mean negative log likelihood by approximately `2e-11`. The unmodified G3-077 baseline passes the test. This small regression is not suppressed by weakening its assertion.
- Additional 9 point fits (normal, Laplace with linear tails, asymmetric Gumbel; 3 seeds each) all converged with nonnegative global curvature verdicts.
- Some larger testing groups exceeded their execution limit and were not counted as passes. The entire regression and cross-platform matrices have not run.

This is a *working checkpoint*, not release-cleared. Unlike the previous contact-block step model, the new exchange algorithm can converge at a different rate near a moving curvature contact. Inspect accuracy, convergence, and runtime broadly before promoting this change.
