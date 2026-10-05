# Releasing Gibbus

Gibbus releases are built and published by GitHub Actions. The production tag workflow builds the source distribution and all supported binary wheels, tests those artifacts, publishes them to TestPyPI, verifies the TestPyPI installation, and only then permits publication of the same artifact set to PyPI.

## Version and tag rule

The package version is declared in `pyproject.toml`.

A production release tag must be exactly:

```text
v<project.version>
```

For example, `version = "0.1.0"` must be released from tag `v0.1.0`.

`.github/workflows/release.yml` enforces this relationship. A mismatched tag fails before any artifact is published.

Manual `workflow_dispatch` runs are lightweight preflights only. They verify synchronized release metadata, build and validate the sdist, and report the expected production tag. They do not build the 20-wheel matrix and can never publish to TestPyPI or PyPI.

## Trusted Publishing setup

Publishing uses PyPI Trusted Publishing through GitHub Actions OIDC. Do not add long-lived PyPI or TestPyPI API tokens to GitHub secrets.

Create two GitHub environments:

- `testpypi`
- `pypi`

The `pypi` environment should require manual approval when repository settings permit it. This gives the maintainer an explicit approval point after the TestPyPI publication and installation check have passed.

For the first release, configure pending Trusted Publishers on both package indexes. The publisher identity must match the workflow exactly.

### TestPyPI

Configure a pending GitHub Actions publisher for project `gibbus` with:

- Owner: `DanJSal`
- Repository: `gibbus`
- Workflow: `release.yml`
- Environment: `testpypi`

### PyPI

Configure a pending GitHub Actions publisher for project `gibbus` with:

- Owner: `DanJSal`
- Repository: `gibbus`
- Workflow: `release.yml`
- Environment: `pypi`

A pending publisher does not reserve a project name until the first successful publication. Configure the publishers only when the release is ready to proceed.

Official Trusted Publishing documentation:

- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/

## Release workflow

`.github/workflows/release.yml` has two modes.

### Manual preflight

Run **Release** from the Actions tab on `main`.

The workflow:

1. reads the version from `pyproject.toml`;
2. verifies that `CITATION.cff` and `CHANGELOG.md` contain the same release version;
3. builds the source distribution;
4. runs `twine check --strict` on that sdist;
5. reports the exact production tag expected for that version.

It stops there. No wheel matrix is built and a manual dispatch never publishes.

### Tag release

Pushing the exact release tag performs the authoritative release build:

1. validates that the tag is exactly `v<project.version>`;
2. calls the reusable wheel workflow and builds/tests all 20 supported wheels;
3. builds the sdist, installs it, smoke-tests all compiled extensions, and runs focused packaging/API/serialization tests;
4. combines the tested 20 wheels and one sdist into one 21-file release artifact;
5. verifies the final artifact set and runs `twine check --strict`;
6. publishes the 21 tested files to TestPyPI using the `testpypi` environment;
7. installs the exact release version from TestPyPI on CPython 3.14 and smoke-tests the installed compiled extensions;
8. waits for the `pypi` environment approval if that environment is protected;
9. downloads the same `gibbus-release-artifacts` artifact used for TestPyPI and publishes it to PyPI;
10. installs the release from PyPI and repeats the compiled-extension smoke test.

The publishing jobs do not rebuild the package. Both indexes receive files from the same tested GitHub Actions artifact.

## Release checklist

Before tagging:

1. Ensure `main` is clean and all required milestone issues for the release are closed.
2. Confirm the full CI workflow is green on the intended release commit.
3. Confirm the intended version in `pyproject.toml`, `CITATION.cff`, and `CHANGELOG.md`.
4. Replace `Unreleased` in the `CHANGELOG.md` release heading with the release date.
5. Commit and push any final release-note/version edits, then require CI to pass.
6. Run the **Release** workflow manually and require the lightweight preflight to pass. This does not rebuild the wheel matrix.
7. Confirm the `testpypi` and `pypi` GitHub environments and their corresponding Trusted Publishers are configured.

Create and push the tag:

```bash
git switch main
git pull --ff-only
git tag -a v0.1.0 -m "Gibbus 0.1.0"
git push origin v0.1.0
```

During the tag workflow:

1. Require the final 20-wheel build, sdist build, and artifact-assembly jobs to pass.
2. Require TestPyPI publication and the TestPyPI installation smoke test to pass.
3. Inspect the TestPyPI project page if desired.
4. Approve the protected `pypi` environment.
5. Require PyPI publication and the final PyPI installation smoke test to pass.

After publication:

1. Confirm `python -m pip install gibbus==<version>` succeeds in a clean environment.
2. Confirm the PyPI project metadata, README, release files, and supported Python classifiers render as expected.
3. Create the GitHub Release for the same tag using the corresponding `CHANGELOG.md` entry.
4. Keep the release workflow run and its `gibbus-release-artifacts` artifact until the release has been independently verified.

## Failed releases

Do not move or reuse a release tag after any file has been published to PyPI. PyPI release files are immutable for a given filename/version.

If a failure occurs before PyPI publication, diagnose the failed job before retrying. If TestPyPI already contains the version, do not assume a later rebuild is identical to the files that were tested there. Prefer correcting the workflow before creating the production tag.

If an incorrect version has already reached PyPI, fix the problem in a new package version rather than trying to replace the published files.
