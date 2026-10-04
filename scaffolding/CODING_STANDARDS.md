# gibbus — Coding Standards

This file records the coding standards used across the `gibbus` library. Keep it short, normative, and easy to scan.

For architecture, pipeline layout, maintainer notes, and exception rationale, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## Core coding standards

- Single, small responsibility per module or function.
- Keep the public API thin and stable; keep internals strict and explicit.
- Validate and convert user input at the public boundary, then operate on one canonical internal representation.
- Keep user-visible defaults and tunables in public signatures or in `_defaults.py`.
- Avoid repeated coercion and repeated validation inside the inner pipeline.
- Cache derived fit-state explicitly, and make cache invalidation obvious.
- Isolate numerically expensive kernels in low-level modules or Cython files.
- Prefer clear, short error messages that explain what failed and, when easy, how to fix it.

---

## More specific rules

- Validation and conversion
  - Public methods (for example, `Distribution.fit`) must validate shapes, types, and argument ranges.
  - Internal helpers should assume canonical `np.float64` arrays and only check strict numerical preconditions.

- Function signatures
  - Public functions may use keyword-only arguments for ergonomics.
  - Internal helpers should prefer explicit ordered parameters and positional-only signatures when practical.

- Defaults and kwargs
  - Do not introduce user-facing defaults deep inside helper functions.
  - Do not propagate generic `**kwargs` through internal numerical helpers; pass explicit parameters instead.

- Imports
  - All imports must appear at the top of each module, not inline inside functions or methods.
  - Group imports in the standard order: stdlib, third-party, intra-package.
    This is checked mechanically by `ruff` (rule `I001`); `ruff check --fix .`
    will reorder them for you.
  - Consolidate multiple imports from the same source into a single `from ... import (...)` block.

- Arrays and shapes
  - Document canonical internal shapes where useful, such as `(n,)` and `(n, 2)`.
  - Use `.reshape(-1)` at the boundary only.

- Error handling
  - Prefer `ValueError` for invalid user inputs.
  - Prefer `RuntimeError` for degenerate numerical states encountered during computation.
  - Never write a bare `except Exception`. Catch `NUMERIC_FAILURES` from `_defaults.py`, which covers the exception types a numerical fallback may legitimately swallow and deliberately excludes generic `ValueError`, `TypeError`, `AttributeError`, `KeyError`, `IndexError`, `MemoryError` and `KeyboardInterrupt`. Use an arithmetic/linear-algebra exception or an explicit typed sentinel for a numerical state that is expected to degrade; never classify a generic `ValueError` by matching its message.
  - Every swallowed **numerical** failure in Python must call `_reraise_if_debug(exc, context)` before taking the fallback and must carry a comment explaining why degrading is correct rather than fatal. Typed control-flow sentinels that are immediately handled are not ledger events. `noexcept nogil` allocation fallbacks in Cython cannot enter the Python ledger and must instead be explicit, local, and documented.

- Memory in Cython
  - Every `malloc` needs a NULL check, and every allocation must be released on all paths. In GIL-held code, prefer `try`/`finally`; in `noexcept nogil` helpers, where Python unwinding is unavailable, free every partial allocation explicitly on each early return and on the success path.
  - Extract raw pointers before entering a `nogil` block; typed-`ndarray` buffer access is not permitted inside one.
  - Keep a checked build profile (`GIBBUS_DEBUG_BUILD=1`) available for CI so bounds, wraparound, `None`, and initialization checks can be exercised separately from release-speed builds.

- Testing and verification
  - Assert on defining properties (a density integrates to one, a CDF is monotone) rather than on numbers captured from a previous run, so tests survive optimiser changes.
  - When a compiled path supersedes a Python one, delete the Python one rather than keeping it as a test oracle. Test the survivor against its own definition instead — a likelihood against the density it is built from, a gradient against finite differences. A retained duplicate is a second implementation to maintain, and it drifts.
  - After edits, run `pytest`. The suite covers point-data fits, interval-data fits, `exp` space usage, `pdf/cdf/ppf`, weighting, serialisation, input validation, automatic component selection, and the compiled kernels.
  - Also run `GIBBUS_DEBUG=1 pytest`: it makes unexpected Python-layer numerical fallbacks fatal. Fallbacks marked `routine=True` are recorded rather than raised; setting `GIBBUS_DEBUG_STRICT=1` together with `GIBBUS_DEBUG=1` makes recorded routine Python fallbacks fatal too. This guarantee does not include `noexcept nogil` compiled allocation fallbacks, which cannot call the Python ledger.
  - Run `ruff check .` to confirm the correctness and import-grouping rules
    hold. The enforced rule set lives in `../pyproject.toml`; purely cosmetic
    rules are deliberately off, and the three exceptions carry a comment
    explaining why. It is enforced by CI. The current state is zero
    problems; keep it there.
  - Run `python tools/check_docstrings.py` to confirm the `Parameters` rule holds. It walks the AST of every module, skips zero-parameter functions and nested closures, and reports both missing sections and parameters that have drifted out of step with their signature. It exits non-zero on any problem and is enforced by CI. The current state is zero problems; keep it there.
  - `../tests` is deliberately outside the checker's scope: a pytest function's parameters are fixture injections rather than an interface, so documenting them would be noise.

---

## Documentation standards

- Use **NumPy-style docstrings** for every module, class, method, and function — public and private alike.
- **Every function or method that takes at least one parameter must have a `Parameters` section documenting all of them**, and the section must stay in step with the signature. This is checked mechanically; see below.
  - Two exemptions, both deliberate: **zero-parameter functions and properties** omit the section entirely (numpydoc convention — there is nothing to document), and **nested closures** defined inside another function need only a one-line docstring or none, since they are implementation detail of the enclosing function and are covered by its documentation.
  - `self` and `cls` are not documented.
  - Parameters that share a type and meaning may be grouped on one line, e.g. `log_boundary_lower, log_boundary_upper : bool`.
- Public methods and nontrivial reusable helpers should include `Returns` when the return value is not obvious from the opening sentence, and `Raises` for user-visible or contract-defining failures. Properties and small private predicates may document the value directly in their summary. The mechanical checker currently enforces parameter/signature fidelity only.
- Private helpers may keep their prose short — intent, preconditions, array shapes, invariants — but still need the full `Parameters` section.
- Module docstrings should describe the module's role in the fitting or evaluation pipeline.
- Keep inline comments short and intent-focused; do not restate obvious code.
- Put long-form maintainer explanations, architecture notes, and exception rationale in [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md), not here.

---

## Exceptions

Deliberate exceptions to these standards should be small, local, commented in code, and easy to justify. Record longer rationale in [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

**Current exceptions:**

- Cython kernels that allocate inside `noexcept nogil` helpers use explicit allocation cleanup rather than Python `try`/`finally`; this is the required pattern in that context.
- The Python docstring checker does not parse `.pyx`; Cython safety is covered by compilation plus the checked-build CI profile instead.
- Implementation-local kernel constants may remain local when they are not shared or user-facing; shared/public tolerances belong in `_defaults.py`.

---

## Practical do / don't checklist

- Do validate at the public API boundary and then operate on canonical arrays.
- Do centralize defaults in `../gibbus/_defaults.py`.
- Do place all imports at the top of each module.
- Do keep numerical kernels isolated and guarded in the Cython sources.
- Do prefer explicit, ordered parameters for internal helpers.
- Do document public behavior with NumPy-style docstrings.
- Do keep docstring `Parameters` entries in step with the signature when changing either; `tools/check_docstrings.py` verifies this.

- Don't re-validate or re-coerce types repeatedly in the inner pipeline.
- Don't put user-friendly defaults deep inside helper functions.
- Don't put inline imports inside functions.
- Don't change `.pyx` files unless you are prepared to rebuild and test them.
- Don't put long-form developer documentation in this file.
