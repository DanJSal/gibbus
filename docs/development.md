# Development

## Building and testing

Install the package in editable mode with its test dependencies, then run the suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Any functional change to a `.pyx` file requires rebuilding the extensions before
tests exercise it:

```bash
python setup.py build_ext --inplace
```

On Windows, run the build from an MSVC developer environment (`vcvars64.bat`).
If the interpreter lacks `Python.h`, pass a matching development-header
directory with `build_ext --include-dirs` rather than committing
machine-specific configuration. Generated `.c`/`.h` files and compiled
extensions are build artifacts and are not versioned.

## Verification workflow

- Black, Ruff, and `cython-lint` are development tools but are not currently included in the `test` extra; install them separately when running the formatting/lint workflow.
- Format Python with Black. Cython is not Black-compatible, so keep `.pyx`
  files manually consistent with the same style and line length.
- Lint Python with `ruff check gibbus tests setup.py` and Cython with
  `cython-lint`. A few long-standing findings are accepted; changes should not
  introduce new ones.
- Prefer targeted test modules while iterating, then run the full suite once at
  a consolidated checkpoint, after rebuilding any changed extensions.
- Test defining properties (normalization, monotonicity, likelihood/gradient
  consistency, analytic oracles) rather than values captured from an earlier run.

Diagnostic modes:

| Variable | Effect |
|----------|--------|
| `GIBBUS_DEBUG=1` | Unexpected swallowed numerical fallbacks re-raise; routine ones are recorded. |
| `GIBBUS_DEBUG_STRICT=1` | Together with `GIBBUS_DEBUG=1`, routine fallbacks are fatal too. |
| `GIBBUS_DEBUG_BUILD=1` | Build time: generate Cython with bounds, wraparound, `None` and initialization checks. |

Allocation fallbacks inside `noexcept nogil` Cython code cannot reach the Python
failure ledger and are outside the debug-mode guarantees.

## Coding standards

**Boundaries and canonical data**

- Validate and convert user input once, at the public boundary
  (`Distribution` methods, the fit-request controller, `_fit/inputs.py`,
  `_api/validation.py`). Internal helpers then consume canonical `float64`
  arrays: point rows `(R, 1)`, interval rows `(R, 2)`, normalized observation
  weights `(R,)`.
- Do not re-coerce, reshape or re-validate already-canonical inputs deeper in
  the pipeline. Genuinely derived values (responsibility-weighted masses,
  selected subsets, newly computed coefficients) are new data and may be
  normalized or checked where they are produced.
- Numerical invariants are not redundant validation: support/feasibility
  checks, positive-mass guards, endpoint-distance handling, coordinate-scale
  guards and certificate checks stay where the mathematics needs them.
- Ownership copies, frozen fitted state and cache-protection copies are
  intentional.

**Explicit policy**

- User-visible defaults live in public signatures or `_defaults.py`. Internal
  helpers take their numerical policy explicitly: tolerances, work budgets,
  RNG streams, test levels, degree policies and side/endpoint choices are
  resolved by the owning boundary and passed down.
- Group related policy in immutable records resolved once and reused (for
  example `_DegreeSelectionConfig`, `_NewtonOptions`, `_EMOptions`,
  `_ComponentSelectionPolicy`, `_SpectralCDFOptions`, `_SpectralPPFOptions`).
- Do not propagate generic `**kwargs` through internal numerical code; use
  explicit parameters or typed records.
- Prefer positional-only, explicitly ordered parameters for internal helpers;
  public methods may use keyword-only arguments for ergonomics.
- Declare capabilities explicitly rather than detecting them by catching
  exceptions (for example, whether an evaluator accepts arrays).

**Error handling**

- `ValueError` for invalid user input; `RuntimeError` for degenerate numerical
  states during computation.
- Never write a bare `except Exception`. A numerical fallback may only catch
  `NUMERIC_FAILURES` (`ArithmeticError`, `LinAlgError`, `RuntimeError`).
  Contract errors such as `ValueError`, `TypeError` and `IndexError` must
  propagate; never classify an error by matching its message.
- Every swallowed numerical failure calls `_reraise_if_debug(exc, context)`
  (with `routine=True` for expected guard paths) and carries a comment saying
  why degrading is correct. Typed control-flow sentinels that are handled
  immediately are not ledger events.

**Imports and structure**

- Keep each module and function to one responsibility; keep the public API thin.
- Imports belong at module top, grouped stdlib, third-party, package.
- A deferred import is acceptable only to break a genuine module cycle, and
  must carry a comment naming the reciprocal dependency.
- Keep one production implementation of each algorithm. A test-only reference
  implementation is acceptable when its module docstring states the narrow
  failures it catches and its limits as an independent oracle.

**Cython**

- Isolate expensive kernels in Cython; Python orchestrates.
- Every `malloc` is NULL-checked and freed on all paths: `try`/`finally` when
  holding the GIL; explicit cleanup on every return in `noexcept nogil` code.
- Bind buffer pointers and sizes before `with nogil:`; never index or take
  addresses of typed `ndarray`s inside the block. Guard empty buffers with
  `NULL` rather than indexing element zero.
- A `nogil` allocation failure may degrade to an equivalent slower path only
  when the fallback is commented locally.
- Accuracy tolerances and work budgets are passed explicitly to compiled entry
  points. Remaining defaults there should select an equivalent execution path
  (such as SIMD versus scalar loops) or mirror a wrapped third-party default.

**Documentation**

- Repository-maintained code and documentation use US English. This includes documentation, docstrings, comments, identifiers, exception/warning text, and other user-facing prose. Preserve another English variant only when required by a proper name, quotation, externally defined API or standard, established technical term, or similarly specific context.
- NumPy-style docstrings for modules, classes and functions, public and
  private. Any callable with parameters needs a `Parameters` section kept in
  step with its signature; `self`/`cls` are omitted and parameters sharing a
  type and meaning may be grouped.
- Zero-parameter functions and properties may document their value in the
  summary line; nested closures need at most a one-line docstring.
- Include `Returns` and `Raises` where they are not obvious or are part of the
  contract. Document canonical shapes and preconditions for private helpers.
- Docstring tooling does not parse `.pyx`; review Cython interface docs when
  changing compiled signatures.
- Keep comments short and about intent.

**Deliberate exceptions**

These are acceptable and should not be "fixed" mechanically:

- Defaults on configuration/record dataclasses and meaningful `None`
  sentinels (warm starts, caches, optional metadata).
- Convenience defaults on public methods and boundary helpers that canonicalize
  raw input.
- Small, local kernel constants that are neither shared nor user-facing.
- Narrow adapters that mirror a third-party option surface (such as the
  QUADPACK ledger adapter) or accept several derived-statistic layouts, when
  documented as such.
