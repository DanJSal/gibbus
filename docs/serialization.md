# Serialization and compatibility

Gibbus provides a versioned, non-object NumPy representation for persisting fitted `Distribution` objects. The durable path is:

```python
import numpy as np
from gibbus import Distribution

np.save("model.npy", c.data)
c2 = Distribution(np.load("model.npy", allow_pickle=False))
```

The serialized state is a NumPy structured scalar and does not require pickle. Its format version is independent of the Gibbus package version.

## Compatibility policy

Released serialization formats are never silently reinterpreted. A newer Gibbus version either reads a supported format correctly, explicitly migrates it, or rejects it with a clear compatibility error. If support for an old format is ever retired, that retirement will be explicit and documented.

Compatibility applies to the fitted mathematical distribution and documented public state, not to bit-for-bit identity of every derived numerical representation. A newer implementation may rebuild CDF/PPF panels, cached moments, and other numerical accelerators while preserving the fitted distribution to the accuracy of the current implementation.

Python pickle is supported as a convenience mechanism, but it is **not** the durable cross-version persistence contract. Use `Distribution.data` with `np.save` / `np.load(..., allow_pickle=False)` for persisted models.

## Envelope

Serialization format v1 has exactly five top-level fields:

| Field | v1 dtype | Meaning |
|---|---|---|
| `format` | `<U8` | Format identifier; must equal `"gibbus"`. |
| `format_version` | `<i8` | Durable serialization version; v1 is `1`. |
| `model` | nested structured scalar | Frozen durable mathematical model. |
| `provenance` | nested structured scalar | Optional historical fit metadata. |
| `cache` | nested structured scalar | Optional implementation-specific runtime cache. |

The loader checks the format and version before interpreting the model. Unversioned pre-release states are rejected. A format newer than the running Gibbus build is rejected rather than guessed at.

The top-level field set is part of the v1 contract: a state with any other top-level fields is rejected. Adding a new top-level section therefore requires a new `format_version`, even if that section would be optional. Optional growth within v1 happens inside `provenance` and `cache`, whose unknown fields are ignored. An optional section that is not present is stored as a one-field stub, `present = False`.

## Durable v1 model

The `model` section is the compatibility boundary. Its field set, base dtypes, and meanings are frozen for format v1. Unknown or missing model fields are rejected. An incompatible change to this section requires a new serialization format version.

For a model with `K` components and `N` total polynomial coefficients, v1 contains:

| Field | dtype / shape | Meaning |
|---|---|---|
| `n_components` | `<i8` scalar | Number of mixture components `K`. |
| `weights` | `<f8`, `(K,)` | Mixture weights, nonnegative and summing to one. |
| `q_poly_values` | `<f8`, `(N,)` | Flattened normalized potential coefficients. |
| `q_poly_offsets` | `<i8`, `(K + 1,)` | Offsets delimiting each component's coefficients. |
| `canonical_support` | `<f8`, `(K, 2)` | Potential support `(zL, zU)` in canonical coordinates. |
| `boundary_amplitudes` | `<f8`, `(K, 2)` | Lower/upper amplitudes stored in public base-`x` order. |
| `boundary_allowed` | `?`, `(K, 2)` | Boundary policy retained for warm-start inheritance. |
| `fit_center` | `<f8`, `(K,)` | Component fitting-coordinate center. |
| `fit_scale` | `<f8`, `(K,)` | Positive fitting-coordinate scale. |
| `fit_direction` | `<f8`, `(K,)` | Coordinate direction, `-1` or `+1`. |
| `support` | `<f8`, `(K, 2)` | Active base-space support; may be narrower after truncation. |
| `requested_poly_degree` | `<i8`, `(K,)` | Requested degree retained for documented warm-start behavior. |
| `mu` | `<f8` scalar | Accumulated public affine shift. |
| `sigma` | `<f8` scalar | Positive accumulated public affine scale. |
| `default_space` | `<U4` scalar | Active query space, `"base"` or `"exp"`. |

All durable numeric dtypes are written explicitly. NumPy's `.npy` format records byte order; Gibbus does not infer integer width from the host platform.

### Ragged polynomial coefficients

Components may have different polynomial degrees. V1 stores their coefficients without object dtype or padding:

```python
q_j = q_poly_values[q_poly_offsets[j] : q_poly_offsets[j + 1]]
```

For example, degrees 2 and 4 produce coefficient lengths 3 and 5 and offsets `[0, 3, 8]`.

## Mathematical interpretation of v1

For component `j`, let the stored coefficients be `q_k`, canonical support be `(zL, zU)`, fitting coordinate be `(center, scale, direction)`, active base support be `(lo, hi)`, and public affine transform be `(mu, sigma)`.

For a public base-space query `x`:

```text
x0 = (x - mu) / sigma
z  = direction * (x0 - center) / scale
```

Boundary amplitudes are stored in public base-`x` order. They are swapped before application in canonical `z` order when `direction < 0`:

```text
(aL, aU) = stored amplitudes                  if direction > 0
(aL, aU) = reversed stored amplitudes         if direction < 0
```

The component potential is:

```text
V(z) = sum_k q_k z^k
       - aL * log(z - zL)
       - aU * log(zU - z)
```

with absent/infinite-side logarithmic terms omitted. `q_0` already includes the normalizing constant, including any renormalization introduced by truncation.

The public base-space component density is therefore:

```text
log p_j(x) = -V(z) - log(scale) - log(sigma)
```

when `x0` lies inside that component's active `support`, and `-inf` otherwise. A mixture is:

```text
p(x) = sum_j weights[j] * p_j(x)
```

This formula is tested against committed v1 fixtures using a plain-NumPy oracle that does not call Gibbus evaluation internals.

`boundary_allowed` and `requested_poly_degree` do not change the already-fitted density, but they are durable because documented `init_from=` warm-start behavior inherits them from a loaded model.

## Optional provenance

The `provenance` section may retain historical information such as the writer Gibbus version, optimizer termination records, parameter counts, and shared-boundary inference. It is useful for auditability but is not required to reconstruct the mathematical distribution.

Unknown provenance fields are ignored. Missing or malformed optional provenance does not invalidate an otherwise valid durable model; unavailable historical fit diagnostics are reported as unavailable/derived rather than invented.

A model loaded without usable provenance is written back with provenance absent (`present = False`), so placeholder values are never re-saved as if they were history.

A future writer may add ignorable provenance without changing `format_version`.

## Optional cache

The `cache` section exists only to avoid rebuilding derived runtime structures on same-version loads, copies, and pickle round trips. It may contain spectral CDF/PPF panels, breakpoints, cached moments/statistics, mode caches, and other current implementation state.

A cache is reused only when all of the following hold:

- its writer Gibbus version exactly matches the running package version;
- its `model_sha256` digest matches the canonical durable `model` values;
- its `runtime_sha256` digest matches the stored runtime cache bytes;
- its runtime structure validates; and
- its semantic model fields agree with the durable model.

If any check fails, the cache is discarded and rebuilt. A missing, stale, unknown, or malformed cache is **not** a reason to reject a valid durable model.

Optional sections are validated explicitly, field by field, before any value is read; Gibbus does not detect malformed data by catching exceptions. The checks fall into two stages:

- **Integrity gates** (version and both digests). Failing one is ordinary — the cache is absent, stale, edited, or corrupted — and selects a rebuild silently.
- **Consistency of a verified cache.** A cache that passes every gate contains exactly the bytes this Gibbus build wrote for exactly this model. If it is nevertheless inconsistent, that can only be a defect in Gibbus, so `load()` raises `ValueError` rather than hiding the defect behind a rebuild. The durable model remains loadable with the cache removed.

Optional provenance follows the same pattern: a section whose layout, dtypes, or values are not recognized for the model is treated as absent, all or nothing.

This intentionally makes cache compatibility more conservative than durable-format compatibility. Upgrading Gibbus may rebuild cached numerical state even when the underlying format remains v1.

## Diagnostics after loading

Historical fit diagnostics come from optional provenance. If that provenance is unavailable, fields that cannot be reconstructed, such as optimizer status or iteration counts, are reported as unavailable rather than inferred.

`spectral_diagnostics` describes the CDF/PPF representation currently serving queries. If a cache is rejected or absent, spectral structures are rebuilt under the running Gibbus version, so these diagnostics describe the rebuilt representation rather than necessarily the representation present when the model was originally saved.

Rebuilding numerical state can take the same graceful numerical fallback paths used elsewhere in Gibbus. Such events may therefore appear in `suppressed_failures()` during `load()`.

## Format evolution

Format v1 has no predecessor because no public Gibbus release used the earlier unversioned development state.

When an incompatible durable-schema change is eventually required:

1. introduce a new `format_version`;
2. keep the old reader and committed compatibility fixtures for every still-supported format;
3. add an explicit migration only when one is actually needed and well-defined;
4. never change the mathematical meaning of an existing format number; and
5. continue treating cache changes independently of durable-format changes.

Loading an older supported format may produce `.data` in the current writer format. Compatibility therefore means preserving model semantics, not reproducing obsolete serialized bytes forever.
