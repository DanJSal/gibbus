# Local derivative series kernel

This development addition constructs and evaluates fixed-order Taylor series of
`q''`, `q'''`, and `q''''`. The standalone evaluator is independent of Newton descent and minima search.
An opt-in integration is described in [the Newton candidate-search experiment](series-search-experiment.md).
The numerical core is Cython, compiled as C++; a small header supplies the
extended-exponent scalar used on range fallback. Normal computation uses double.

## Python API

The API is internal (`gibbus._model.basis_series`) and accepts the existing
compiled `_BasisModel`, including face-restricted models.

```python
from gibbus._model.basis_series import SeriesPlan

plan = model.series_plan                 # retained immutable plan for this model
series = plan.coefficients(theta, centers=[x0, x1], order=24)
values = series.evaluate(x, center_index=0)  # shape x.shape + (3,)

# Repeated theta changes at fixed centers/order:
cache = plan.cache([x0, x1], order=24)
series = cache.coefficients(new_theta)   # no transcendental evaluations
values = series.evaluate_offsets(dx, center_index=indices)
values = series.evaluate_scaled(u, center_index=indices)
```

`centers` is a scalar or nonempty vector. `scale` is optional, positive, scalar
or one per center. Queries and center indices broadcast. Orders 0 through 4096
are accepted; numerical regression coverage emphasizes orders up to 48.

The convention is

\[
u=(x-x_0)/\ell,\qquad
q^{(r)}(x_0+\ell u)\approx\sum_{n=0}^{N}C_{r,n}u^n,
\quad r=2,3,4.
\]

`series.coefficients` has shape `(centers, 3, order+1)`. Derivatives are already
in physical x units: do not multiply the rows by inverse powers of scale.
`mantissa` and `exponent` retain coefficients as `mantissa * 2**exponent`
when they exceed ordinary double range. `coefficients` raises on overflow;
exporting extremely tiny coefficients to float64 may round them to zero.
Evaluation uses the retained representation and can recover representable
values from intermediate coefficients outside double range. It raises if a
final value is not finite in float64. Final subnormal results have ordinary
float64 rounding and may underflow to zero.

The model retains one plan through `model.series_plan`. Explicit
`SeriesPlan(model)` constructs an independent plan. Exact map construction
skips structural zeros and preserves rational cancellation before rounding.

Plans, returned coefficients and caches have immutable byte-backed arrays.
Every construction call owns its workspace. Shared plans/caches support
concurrent calls without a worker pool or mutable global state.

## Compiled interface and GIL boundary

`_series_kernels.pxd` exports:

- `series_workspace_new(order, nf)` and `series_workspace_free(workspace)`;
- `series_construct_center(basis, maps, theta, center, scale, workspace, out, exponents)`;
- `series_horner_batch(coeff, exponents, scaled, u, indices, Q, order, out)`.

All are `noexcept nogil`. Allocate one workspace per concurrent caller, reuse it
across centers, then free it. Center construction does not allocate. Batch
Horner does not allocate. Output storage belongs to the caller. The `BasisTable`
and exact-precombined floating sign maps must remain alive during the call.
The map layout is `(4, 3*nf, d)`, with signs `(--),(-+),(+-),(++)` and atom
channels `(H, lower complement, upper complement)`. See `SeriesPlan` for its
one-time exact construction. Raw-pointer callers must supply valid dimensions,
finite inputs, and adequately sized buffers; the constructor checks center,
scale, support, radii and workspace compatibility. Horner callers additionally
validate query radii and indices themselves.

Constructor statuses: `0` ordinary success; `1` extended-exponent success;
`-1` invalid geometry/workspace; `-2` unsupported numerical range. Workspace
allocation returns NULL on failure. Horner returns `0` on success or `1` when
a result is outside finite float64 range. On error, output is unspecified.

The Python facade does rational preparation, validation and buffer allocation
with the GIL, then releases it for all numerical construction, combination,
and Horner loops. No Python callback, object arithmetic, arbitrary precision,
or transcendental evaluation occurs inside ordinary polynomial evaluation.
Cython's debug bounds checks may report an invalid buffer through Python;
the numerical pointer kernels themselves contain no Python operations.

## Construction and reuse

At a center, each signed gate `tanh(a*(x-b))` is written as

\[
\sigma(1-2W),\quad W=E/(1+E),\quad
E=\exp(\kappa(x-b)),\quad\kappa=-2\sigma a,
\]

with `sigma` selected so `E(center) <= 1`. Each atom's ungated coefficient
`a_j + sigma_L*b_j + sigma_U*c_j` is combined **as rational map rows before
conversion to double**. The residual channels are `-2*sigma_L*b_j` and
`-2*sigma_U*c_j`. This preserves structural cancellations, including faces.
Gated constants and linear monomials remain present.

Each atomic derivative is expanded independently:

| Atom | Coefficient construction |
|---|---|
| Monomial | Shift each analytic derivative polynomial by Horner; no division by center coordinate. |
| Endpoint log | Independently seed derivative orders 2, 3, 4 and use their negative-power recurrence; no logarithm calls. |
| Exponential | One exponential seed and geometric/factorial recurrence for each derivative row. |
| Soft power | One fractional-power seed, six analytic derivative seeds, and an O(N) differentiated-ODE recurrence per derivative. |
| Gate complement | One exponential seed (plus `expm1` near its center), stable convolution recurrences for the analytic derivatives. |
| Exponential × gate | Fuse exponential exponents before evaluation; use `Y'=(lambda+kappa-kappa*W)Y` and independently construct `Y*P_r(W)`. |

Soft powers use `R=hypot(x0-c,sqrt(delta))`, `v=(x0-c)/R`,
`e=(sqrt(delta)/R)^2` and `omega=ell/R`; `e` is never computed as `1-v*v`.
For every derivative row r, the recurrence is

\[
F_{r,n+2}=-\frac{(n+1)(2n+2r+2-p)v\omega F_{r,n+1}
 +(n+r+1)(n+r-p)\omega^2F_{r,n}}{(n+2)(n+1)}.
\]

The seeds are the analytic r-th and (r+1)-th derivatives. In particular,
q''' and q'''' are **not obtained by differentiating the computed q'' series**.
Products use the derivative Leibniz coefficients (1,2,1), (1,3,3,1), and
(1,4,6,4,1), with ordinary Taylor convolution within each product. Like powers
are accumulated with compensated summation. Gate products with monomials use
finite derivative degree to limit convolution work.

The direct construction path retains only the three combined derivative rows.
The cache retains each used atom/channel's three derivative rows, independent
of the number of independent basis columns. Recombination contracts the exact
sign maps against theta, then sums these cached rows. Cost is O(d*atoms) for
map contraction plus O(centers*atoms*N) for recombination. Construction costs
O(N) for each ungated nonpolynomial atom and O(N²) for gate/product series.
Geometry-specific algebra is compiled into the kernels; exact maps are reused
by the plan, gate jets by all atoms at one center, and primitive jets by all
coefficient vectors using a cache. New centers rebuild their nonpolynomial
jets; they never shift a previously truncated series.

Horner evaluates four queries together when they share a center, exposing
independent arithmetic for SIMD/instruction-level parallelism. Mixed-center
queries retain their original order and use an interleaved three-row loop.

## Range and accuracy contract

The ordinary path checks floating-point range exceptions and results. When
needed, it repeats the calculation with normalized binary64 significands and
signed 64-bit binary exponents. This extends range, not precision. Each
coefficient carries its own exponent, so a tiny seed can recover through high
derivatives or multiplication by a large amplitude. The fallback also covers
map contraction and Horner intermediates. Exponents are bounded to +/-2**50;
exponential seed arguments beyond +/-2**48 are reported as unsupported.
No long-double dependence, Python fallback, arbitrary precision, silent
clipping, or numerical pruning of tiny nonzero theta is used. The floating
point environment is saved/restored around guarded ordinary computations.

The supported environment is ordinary IEEE float64 arithmetic with gradual
underflow, default rounding, and compiler fast-math disabled. The extension
is built with contraction disabled on GCC/Clang because the compensated sums
assume separate rounding; compiler-specific cross-platform CI remains needed.
Tests in this handoff were run on Linux x86-64 with GCC/G++ and CPython 3.12.

The conservative analytic radius is the minimum over structurally used terms:

| Term | Distance to nearest complex singularity |
|---|---|
| Log | Distance to its finite endpoint |
| Soft power | `hypot(x0-c, sqrt(delta))` |
| Tanh gate | `hypot(x0-b, pi/(2*abs(a)))` |
| Polynomial or exponential | Infinity |

Here delta is in squared physical units, so the closest soft-power branch
points are `c +/- i*sqrt(delta)`. Gating does not remove those singularities;
the radius also includes the gate's poles. All available gates are included
conservatively even when a particular coefficient vector disables them.
The default scale is 0.4 of the finite radius, or the fitting scale for an
entire series. Explicit scales must lie strictly below the radius. Queries
must be strictly inside both the radius and the model support. The radius is
structural, not relaxed based on small or zero current numerical amplitudes.

**Convergence radius is not a finite-order error guarantee.** This API evaluates
a specified polynomial approximation. It does not choose order, certify the
Taylor remainder, guarantee relative accuracy at cancellation zeros, or choose
a finite boundary cutoff for vanishing/infinite tails. Those belong to the
later search/certification layer. Keeping them separate avoids silently treating
an approximation as a certified linear constraint.

## Build, tests and benchmark

A C++ compiler is now required in addition to the existing C compiler.

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_basis_series.py tests/test_basis_evaluation.py tests/test_basis_construction.py -q
python tests/benchmark_basis_series.py
```

`GIBBUS_DEBUG_BUILD=1 python setup.py build_ext --inplace` enables the existing
project-wide Cython checking mode. Restore a release build before benchmarking.
No generated C/C++, shared binaries, or local dependency installations are
included in the handoff archive.

The new tests independently expand the original potential with 150-digit
Decimal arithmetic and differentiate only that oracle. They cover all 23
existing basis fixtures, direct/cache agreement, gate centers, orders 0/1,
higher orders, physical coordinate scaling, near-boundary logs, branch radii,
face restrictions, tiny coefficients, extended-exponent coefficients, fused
flat-rate derivatives, mixed query batches, concurrent cache use, and the
exported nogil C construction API. Row-scaled coefficient comparison tolerances
are 3e-12 to 5e-12; evaluations also have independent direct-evaluator checks.

See `local-series-validation.md` for this build's results and timings.
