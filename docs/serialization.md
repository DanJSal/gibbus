# Serialization

Gibbus provides a non-object NumPy representation for persisting fitted `Distribution` objects. The durable path is:

```python
import numpy as np
from gibbus import Distribution

np.save("model.npy", c.data)
c2 = Distribution(np.load("model.npy", allow_pickle=False))
```

The serialized state is a NumPy structured scalar and does not require pickle. Its `format_version` is independent of the Gibbus package version.

A Gibbus build reads the format it writes, `format_version` **1**. The fitted mathematical distribution and the documented public state are the contract, not bit-for-bit identity of every derived numerical representation: CDF/PPF panels, cached moments and other numerical accelerators may be rebuilt.

Python pickle is supported as a convenience mechanism, but it is **not** the durable persistence contract. Use `Distribution.data` with `np.save` / `np.load(..., allow_pickle=False)` for persisted models.

## Envelope

The envelope has exactly five top-level fields:

| Field | dtype | Meaning |
|---|---|---|
| `format` | `<U8` | Format identifier; must equal `"gibbus"`. |
| `format_version` | `<i8` | Durable serialization version, `1`. |
| `model` | nested structured scalar | Durable mathematical model. |
| `provenance` | nested structured scalar | Optional historical fit metadata. |
| `cache` | nested structured scalar | Optional implementation-specific runtime cache. |

The loader checks the format and version before interpreting the model. A state with any other top-level fields, or another `format_version`, is an invalid envelope (`ValueError`). Optional growth happens inside `provenance` and `cache`, whose unknown fields are ignored. An optional section that is not present is stored as a one-field stub, `present = False`.

## Durable model

The `model` section is the contract. Its field set, base dtypes and meanings are fixed; unknown or missing model fields are rejected.

For a model with `K` components and `N` total basis coefficients:

| Field | dtype / shape | Meaning |
|---|---|---|
| `n_components` | `<i8` scalar | Number of mixture components `K`. |
| `weights` | `<f8`, `(K,)` | Mixture weights, nonnegative and summing to one. |
| `basis_models` | `<U*`, `(K,)` | Each component's basis record (canonical JSON, below). |
| `theta_values` | `<f8`, `(N,)` | Flattened canonical basis coefficients. |
| `theta_offsets` | `<i8`, `(K + 1,)` | Offsets delimiting each component's coefficients. |
| `log_z` | `<f8`, `(K,)` | Log normalizing constant of each canonical potential over its basis support. |
| `support` | `<f8`, `(K, 2)` | Active support in model units; narrower than the basis support after truncation. |
| `mu` | `<f8` scalar | Accumulated public affine shift. |
| `sigma` | `<f8` scalar | Positive accumulated public affine scale. |
| `default_space` | `<U4` scalar | Active query space, `"base"` or `"exp"`. |

All durable numeric dtypes are written explicitly. NumPy's `.npy` format records byte order; Gibbus does not infer integer width from the host platform.

Components may have different bases. Their coefficients are stored without object dtype or padding:

```python
theta_j = theta_values[theta_offsets[j] : theta_offsets[j + 1]]
```

### Basis record

`basis_models[j]` is canonical JSON (sorted keys, no whitespace) with a record `version` (`1`) and:

- `request`: the canonical basis request: support ends `lower`/`upper`, `bulk_degree`, the endpoint-log flags `log_lower`/`log_upper`, the resolved tail requests `lower_tail`/`upper_tail` (kind `polynomial`, `soft_power` or `exponential` with its power or rate), `independent` (independent equal-rate tail amplitudes) and the span policy `expansion`;
- `geometry`: the fitting coordinate (`support_kind`, center `c`, scale `h`, canonical and physical supports) and the resolved exponential rates in canonical units;
- `slots`: the chosen coefficient slots of the span;
- `columns`: the column labels, in coefficient order;
- `suppressed`: the infinite ends whose requested leading tail functional is fixed to zero by the coefficient face. Loading rebuilds that exact reduced face before evaluating or normalizing the density.

Floats are stored as exact hexadecimal strings, so the record reproduces the basis bit for bit. Loading recompiles the basis from `request`, `geometry` and `slots`. A record that is malformed, whose recompiled column labels differ from `columns`, whose coefficient count differs from the basis, or whose recomputed `log Z` differs from `log_z` by more than `1e-9` relative is rejected (`ValueError`): the record does not reproduce its density.

## Mathematical interpretation

For component `j`, the basis columns are functions of the canonical coordinate `z = (u - c) / h` at a model-unit point `u`. Each column is a sum of underlying functions `f(z)` (monomials `z^n`, finite-endpoint logarithms `-log(u - L)` and `-log(U - u)`, soft powers `A(z)^p` with `A = (1 + z^4)^(1/4) - 1`, and directional exponentials `exp(lambda (z - 1))` / `exp(-lambda (z + 1))`) times a fixed gate combination `a + b t_L(z) + c t_U(z)`, where `t_L` and `t_U` are the erf gates centered at `z = -1` and `z = 1`. The potential is

```text
q_j(u) = Phi(u) . theta_j
```

and the model-unit component density is

```text
log p_j(u) = -q_j(u) - log_z[j]
```

on the basis support, renormalized to the active `support` when that is narrower (a truncation keeps the potential and narrows the support), and `-inf` outside it. For a public base-space query `x`, `u = (x - mu) / sigma` and

```text
log p_j(x) = log p_j(u) - log(sigma)
p(x)       = sum_j weights[j] * p_j(x)
```

Component order is the order of the stored arrays (mode-sorted at fit time).

The durable coefficients and `log_z` are canonical. A running build may evaluate the same polynomial in other, better-conditioned coordinates (recurrence polynomials placed at the fitted law); that choice is runtime state and never changes the durable values.

## Optional provenance

The `provenance` section may retain historical information such as the writer Gibbus version, optimizer termination records, parameter counts, and shared-boundary inference. It is useful for auditability but is not required to reconstruct the mathematical distribution.

Unknown provenance fields are ignored. Missing or malformed optional provenance does not invalidate an otherwise valid durable model; unavailable historical fit diagnostics are reported as unavailable/derived rather than invented.

A model loaded without usable provenance is written back with provenance absent (`present = False`), so placeholder values are never re-saved as if they were history.

## Optional cache

The `cache` section exists only to avoid rebuilding derived runtime structures on loads, copies, and pickle round trips by the same build. It may contain spectral CDF/PPF panels, breakpoints, cached moments/statistics, mode caches, the runtime evaluation coordinates, and other current implementation state.

A cache is reused only when all of the following hold:

- its writer Gibbus version exactly matches the running package version;
- its `implementation_sha256` matches the current Python/native code, Python ABI and byte order, and NumPy/SciPy versions;
- its `model_sha256` digest matches the canonical durable `model` values;
- its `runtime_sha256` digest matches the stored runtime cache bytes;
- its runtime structure validates; and
- its semantic model fields agree with the durable model.

If any check fails, the cache is discarded and rebuilt. A missing, stale, unknown, or malformed cache is **not** a reason to reject a valid durable model.

Optional sections are validated explicitly, field by field, before any value is read; Gibbus does not detect malformed data by catching exceptions. The checks fall into two stages:

- **Integrity gates** (writer version and digests). Failing one is ordinary (the cache is absent, stale, edited, or corrupted) and selects a rebuild silently.
- **Consistency of a verified cache.** A cache that passes every gate contains exactly the bytes this Gibbus build wrote for exactly this model. If it is nevertheless inconsistent, that can only be a defect in Gibbus, so `load()` raises `ValueError` rather than hiding the defect behind a rebuild. The durable model remains loadable with the cache removed.

Optional provenance follows the same pattern: a section whose layout, dtypes, or values are not recognized for the model is treated as absent, all or nothing.

Cache reuse is deliberately more conservative than the durable format: another Gibbus build rebuilds cached numerical state from the durable model.

## Diagnostics after loading

Historical fit diagnostics come from optional provenance. If that provenance is unavailable, fields that cannot be reconstructed, such as optimizer status or iteration counts, are reported as unavailable rather than inferred.

`spectral_diagnostics` describes the CDF/PPF representation currently serving queries. If a cache is rejected or absent, spectral structures are rebuilt by the running build, so these diagnostics describe the rebuilt representation.

Rebuilding numerical state fails loudly when a numerical step cannot be resolved; recorded routine events may appear in `suppressed_failures()` during `load()`.
