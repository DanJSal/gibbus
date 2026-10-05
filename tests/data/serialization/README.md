# Serialization compatibility fixtures

These files are immutable compatibility fixtures for durable serialization format v1.

- `v1_single.npy` — full-line single-component model.
- `v1_mixture.npy` — two-component mixture with different polynomial degrees.
- `v1_transformed.npy` — reflected upper-half-line model with an affine presentation transform and exp-space default.
- `v1_bounded.npy` — bounded model with both finite-boundary terms enabled.
- `v1_no_cache.npy` — truncated mixture (originally the only cacheless fixture; see below).
- `v1_reference.npz` — public evaluation/reference values and summary statistics for the five model fixtures.

The `.npy` files are loaded with `allow_pickle=False`. `tests/test_serialization_v1.py` also evaluates their durable model sections with a plain-NumPy density oracle.

The fixtures represent the serialization-v1 writer shipped for the `0.1.0` release line. Their optional provenance writer tag is therefore `0.1.0`; it is not part of the frozen durable `model` schema.

None of the fixtures carries an optional runtime cache. The cache is keyed to the exact writer version, and every pre-release build reports the same version, so a committed cache could be trusted by a build whose cache layout had since changed. The compatibility promise concerns the durable `model` section, so the fixtures exercise the rebuild-from-model path only. Same-version cache reuse is tested with freshly written states instead.

Do **not** regenerate or replace these fixtures merely because a future implementation causes a compatibility test to fail. A fixture change requires an explicit serialization-policy decision. New serialization versions should add new fixture sets rather than rewriting v1 history.
