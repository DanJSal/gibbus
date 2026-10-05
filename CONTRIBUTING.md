# Contributing to Gibbus

Bug reports, documentation improvements, tests, and focused code contributions are welcome.

## Development setup

Clone the repository and install an editable build with test dependencies:

```bash
git clone https://github.com/DanJSal/gibbus.git
cd gibbus
python -m pip install -e ".[test]"
```

Changes to Cython sources require rebuilding the extensions before tests exercise the modified code:

```bash
python setup.py build_ext --inplace
```

Run the relevant targeted tests while iterating, then run the full suite before submitting a substantial change:

```bash
python -m pytest
```

The detailed build, lint, numerical-testing, Cython, serialization, and documentation conventions are maintained in [docs/development.md](docs/development.md).

## Changes to public behavior

Keep public signatures, documentation, tests, and release notes synchronized. A change that intentionally alters documented behavior should update `CHANGELOG.md` and the relevant documentation in the same contribution.

Durable serialization is a compatibility surface. Do not reinterpret an existing serialization format in place; follow the versioning and fixture rules in [docs/serialization.md](docs/serialization.md) and [docs/development.md](docs/development.md#serialization-maintenance).

## Numerical changes

Prefer tests based on mathematical invariants or independent oracles rather than values captured from a previous implementation. Numerical tolerance changes should be justified by the scale and conditioning of the quantity being tested, not used to conceal a reproducible defect.

## Security reports

Do not report security-sensitive issues in a public issue. Follow [SECURITY.md](SECURITY.md).
