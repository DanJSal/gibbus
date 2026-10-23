"""Executable contract tests for durable serialization.

Gibbus reads and writes one format (``FORMAT_VERSION``, the G3 basis model);
every other version is rejected.  The model shapes are freshly fitted, so
the tests exercise the current schema only.
"""

import ast
import inspect

import numpy as np
import pytest

import gibbus
import gibbus._defaults as defaults
import gibbus._serialization as serialization
from gibbus import Distribution
from gibbus._serialization import (
    _MODEL_FIELDS,
    FORMAT_NAME,
    FORMAT_VERSION,
    _package_version,
    _runtime_digest,
    strip_cache,
    strip_provenance,
)


def _fit_shape(name, /):
    """Return a freshly fitted model of one serialized shape.

    Parameters
    ----------
    name : str
        ``"single"``, ``"mixture"`` (different degrees), ``"transformed"``
        (affine presentation, exp default), ``"bounded"`` (two endpoint logs)
        or ``"truncated"`` (a truncated mixture).

    Returns
    -------
    Distribution
    """
    rng = np.random.default_rng(8200 + list(_SHAPES).index(name))
    if name == "single":
        return Distribution().fit(rng.normal(size=300), poly_degree=2)
    if name == "mixture":
        x = np.concatenate([rng.normal(-2.0, 0.5, 150), rng.normal(2.0, 0.8, 150)])
        return Distribution().fit(
            x,
            n_components=2,
            component_options=[{"poly_degree": 2}, {"poly_degree": 4}],
            rng=0,
        )
    if name == "transformed":
        fitted = Distribution().fit(
            8.0 - rng.gamma(3.0, 1.0, 300), poly_degree=2, support=(-np.inf, 8.0)
        )
        moved = fitted.transform(mu=0.5, sigma=0.25, pullback=False, inplace=False)
        moved.set_default("exp")
        return moved
    if name == "bounded":
        return Distribution().fit(
            rng.beta(2.0, 3.0, 300),
            support=(0.0, 1.0),
            poly_degree=2,
            log_boundary_lower=True,
            log_boundary_upper=True,
        )
    x = np.concatenate([rng.normal(-2.0, 0.6, 150), rng.normal(2.0, 0.6, 150)])
    return (
        Distribution().fit(x, n_components=2, poly_degree=2, rng=0).truncate(-3.0, 3.5)
    )


_SHAPES = ("single", "mixture", "transformed", "bounded", "truncated")


def _replace_section(state, section, value, /):
    """Return a structured envelope with one nested section replaced.

    Parameters
    ----------
    state : numpy.void
        Serialized envelope.
    section : str
        Top-level field to replace.
    value : numpy.void
        Replacement section; its dtype replaces the field's dtype.

    Returns
    -------
    numpy.void
    """
    dtype = [
        (name, value.dtype if name == section else state.dtype[name])
        for name in state.dtype.names
    ]
    out = np.zeros((), dtype=dtype)
    for name in state.dtype.names:
        out[name] = value if name == section else state[name]
    return out


def _drop_top_field(state, field, /):
    """Return an envelope with one top-level field removed.

    Parameters
    ----------
    state : numpy.void
        Serialized envelope.
    field : str
        Top-level field to drop.

    Returns
    -------
    numpy.void
    """
    names = [name for name in state.dtype.names if name != field]
    out = np.zeros((), dtype=[(name, state.dtype[name]) for name in names])
    for name in names:
        out[name] = state[name]
    return out


def _counting_rebuild(monkeypatch, /):
    """Patch the component rebuild hook and return the list of its calls.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to install the counting wrapper for one test.

    Returns
    -------
    list
        Grows by one entry per rebuilt component.
    """
    original = serialization._rebuild_basis_component
    calls = []

    def counted(*args, **kwargs):
        """Record one rebuild, then delegate to the real hook."""
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(serialization, "_rebuild_basis_component", counted)
    return calls


def _fresh_state(name, /):
    """Return a state, with cache, of one freshly fitted model shape.

    Parameters
    ----------
    name : str
        Key into ``_SHAPES``.

    Returns
    -------
    numpy.void
    """
    return _fit_shape(name).data


def _serialization_ledger_entries():
    """Return suppressed-failure records whose context mentions serialization."""
    return [f for f in gibbus.suppressed_failures() if "serializ" in f["context"]]


def test_v2_envelope_and_model_schema_are_explicit():
    model = Distribution().fit(
        np.random.default_rng(8101).normal(size=160), poly_degree=4
    )
    state = model.data
    assert state.dtype.names == (
        "format",
        "format_version",
        "model",
        "provenance",
        "cache",
    )
    assert str(state["format"]) == FORMAT_NAME
    assert int(state["format_version"]) == FORMAT_VERSION == 2
    assert state.dtype.hasobject is False
    durable = state["model"]
    assert durable.dtype.names == _MODEL_FIELDS
    assert durable.dtype.hasobject is False

    k = int(durable["n_components"])
    theta_size = int(np.asarray(durable["theta_values"]).size)
    record = durable.dtype["basis_models"].subdtype[0]
    expected_schema = {
        "n_components": (np.dtype("<i8"), ()),
        "weights": (np.dtype("<f8"), (k,)),
        "basis_models": (record, (k,)),
        "theta_values": (np.dtype("<f8"), (theta_size,)),
        "theta_offsets": (np.dtype("<i8"), (k + 1,)),
        "log_z": (np.dtype("<f8"), (k,)),
        "support": (np.dtype("<f8"), (k, 2)),
        "mu": (np.dtype("<f8"), ()),
        "sigma": (np.dtype("<f8"), ()),
        "default_space": (np.dtype("<U4"), ()),
    }
    assert record.kind == "U"
    for name, (expected_base, expected_shape) in expected_schema.items():
        field_dtype = durable.dtype[name]
        base, shape = field_dtype.subdtype or (field_dtype, ())
        assert base == expected_base, name
        assert shape == expected_shape, name


def test_v2_ragged_coefficients_use_values_and_offsets():
    rng = np.random.default_rng(8102)
    x = np.concatenate([rng.normal(-2.0, 0.5, 140), rng.normal(2.0, 0.7, 140)])
    fitted = Distribution().fit(
        x,
        n_components=2,
        component_options=[{"poly_degree": 2}, {"poly_degree": 4}],
        rng=0,
    )
    model = fitted.data["model"]
    offsets = np.asarray(model["theta_offsets"], dtype=np.int64)
    # Minimal full-line spans: the bulk powers z .. z^d.
    assert sorted(np.diff(offsets)) == [2, 4]
    assert model["theta_values"].shape == (6,)
    assert sorted(int(c.data["bulk_degree"]) for c in fitted.components) == [2, 4]


def test_unversioned_state_is_rejected():
    fitted = Distribution().fit(np.random.default_rng(8103).normal(size=120))
    with pytest.raises(ValueError, match="unversioned"):
        Distribution(fitted.components[0].data)
    with pytest.raises(ValueError, match="unversioned"):
        Distribution(_drop_top_field(fitted.data, "format_version"))


def test_envelope_dtypes_are_pinned():
    fitted = Distribution().fit(np.random.default_rng(8110).normal(size=120))
    state = fitted.data
    dtype = [
        (
            name,
            np.dtype("<i4") if name == "format_version" else state.dtype[name],
        )
        for name in state.dtype.names
    ]
    malformed = np.zeros((), dtype=dtype)
    for name in state.dtype.names:
        malformed[name] = state[name]
    with pytest.raises(ValueError, match="format_version must have dtype <i8"):
        Distribution(malformed)


def test_future_and_other_versions_are_rejected():
    fitted = Distribution().fit(np.random.default_rng(8104).normal(size=120))

    future = np.array(fitted.data, copy=True)
    future["format_version"] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        Distribution(future)

    # No migration readers: any other version is rejected outright.
    for version in (FORMAT_VERSION - 1, 0):
        old = np.array(fitted.data, copy=True)
        old["format_version"] = version
        with pytest.raises(ValueError, match="not supported"):
            Distribution(old)


def test_unknown_model_fields_are_rejected():
    fitted = Distribution().fit(np.random.default_rng(8105).normal(size=120))
    state = fitted.data
    model = state["model"]
    dtype = [(name, model.dtype[name]) for name in model.dtype.names]
    dtype.append(("future_field", "<i8"))
    expanded = np.zeros((), dtype=dtype)
    for name in model.dtype.names:
        expanded[name] = model[name]
    expanded["future_field"] = 1
    malformed = _replace_section(state, "model", expanded)
    with pytest.raises(ValueError, match="do not match the frozen schema"):
        Distribution(malformed)


def test_unknown_provenance_fields_are_ignored():
    fitted = Distribution().fit(np.random.default_rng(8106).normal(size=120))
    state = fitted.data
    provenance = state["provenance"]
    dtype = [(name, provenance.dtype[name]) for name in provenance.dtype.names]
    dtype.append(("future_note", "<U16"))
    expanded = np.zeros((), dtype=dtype)
    for name in provenance.dtype.names:
        expanded[name] = provenance[name]
    expanded["future_note"] = "ignored"
    loaded = Distribution(_replace_section(state, "provenance", expanded))
    assert loaded.fit_diagnostics["provenance"] == fitted.fit_diagnostics["provenance"]


def test_malformed_optional_provenance_is_ignored():
    fitted = Distribution().fit(np.random.default_rng(8111).normal(size=140))
    state = np.array(fitted.data, copy=True)
    state["provenance"]["n_parameters"] += 1
    loaded = Distribution(state)
    assert loaded.fit_diagnostics["provenance"] == "derived"
    assert loaded.fit_diagnostics["components"][0]["status"] is None
    np.testing.assert_allclose(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0]), rtol=0.0, atol=0.0
    )


def test_malformed_provenance_shape_is_nonfatal_and_marks_history_unavailable():
    fitted = Distribution().fit(np.random.default_rng(8113).normal(size=140))
    malformed = np.zeros((), dtype=[("present", "?", (2,))])
    malformed["present"] = (True, True)
    loaded = Distribution(_replace_section(fitted.data, "provenance", malformed))
    diagnostics = loaded.fit_diagnostics
    assert diagnostics["provenance"] == "derived"
    assert diagnostics["components"][0]["status"] is None
    np.testing.assert_array_equal(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0])
    )


def test_partial_component_provenance_does_not_leak_history_from_cache():
    fitted = Distribution().fit(np.random.default_rng(8114).normal(size=140))
    state = fitted.data
    provenance = state["provenance"]
    names = [name for name in provenance.dtype.names if name != "optimizer_status"]
    partial = np.zeros((), dtype=[(name, provenance.dtype[name]) for name in names])
    for name in names:
        partial[name] = provenance[name]
    loaded = Distribution(_replace_section(state, "provenance", partial))
    diagnostics = loaded.fit_diagnostics
    assert diagnostics["provenance"] == "derived"
    assert diagnostics["components"][0]["status"] is None
    assert diagnostics["components"][0]["n_iterations"] is None
    assert diagnostics["n_face_parameters"] is None


def test_semantically_invalid_component_provenance_is_nonfatal():
    fitted = Distribution().fit(np.random.default_rng(8116).normal(size=140))
    state = fitted.data
    provenance = state["provenance"]
    dtype = [
        (
            name,
            (
                ("<i8", provenance["optimizer_success"].shape)
                if name == "optimizer_success"
                else provenance.dtype[name]
            ),
        )
        for name in provenance.dtype.names
    ]
    malformed = np.zeros((), dtype=dtype)
    for name in provenance.dtype.names:
        if name != "optimizer_success":
            malformed[name] = provenance[name]
    malformed["optimizer_success"] = 300
    loaded = Distribution(_replace_section(state, "provenance", malformed))
    diagnostics = loaded.fit_diagnostics
    assert diagnostics["provenance"] == "derived"
    assert diagnostics["components"][0]["status"] is None
    np.testing.assert_array_equal(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0])
    )


def test_malformed_cache_shape_is_nonfatal_and_rebuilds(monkeypatch):
    fitted = Distribution().fit(np.random.default_rng(8115).normal(size=140))
    malformed = np.zeros((), dtype=[("present", "?", (2,))])
    malformed["present"] = (True, True)
    calls = _counting_rebuild(monkeypatch)
    loaded = Distribution(_replace_section(fitted.data, "cache", malformed))
    assert len(calls) == 1
    np.testing.assert_array_equal(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0])
    )


def test_unknown_cache_fields_are_ignored(monkeypatch):
    fitted = Distribution().fit(np.random.default_rng(8112).normal(size=140))
    state = fitted.data
    cache = state["cache"]
    dtype = [(name, cache.dtype[name]) for name in cache.dtype.names]
    dtype.append(("future_note", "<U16"))
    expanded = np.zeros((), dtype=dtype)
    for name in cache.dtype.names:
        expanded[name] = cache[name]
    expanded["future_note"] = "ignored"

    def fail(*args, **kwargs):
        """Fail the test: a compatible cache must not be rebuilt."""
        raise AssertionError("compatible cache should have been reused")

    monkeypatch.setattr(serialization, "_rebuild_basis_component", fail)
    loaded = Distribution(_replace_section(state, "cache", expanded))
    np.testing.assert_array_equal(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0])
    )


def test_cache_version_or_digest_mismatch_rebuilds(monkeypatch):
    fitted = Distribution().fit(np.random.default_rng(8107).normal(size=160))
    calls = _counting_rebuild(monkeypatch)
    Distribution(fitted.data)
    assert not calls

    version_mismatch = np.array(fitted.data, copy=True)
    version_mismatch["cache"]["writer_version"] = "definitely-not-this-version"
    Distribution(version_mismatch)
    assert len(calls) == 1

    calls.clear()
    digest_mismatch = np.array(fitted.data, copy=True)
    digest_mismatch["cache"]["model_sha256"] = "0" * 64
    Distribution(digest_mismatch)
    assert len(calls) == 1


def test_cacheless_rebuild_preserves_distribution_and_warm_start_semantics():
    rng = np.random.default_rng(8108)
    fitted = Distribution().fit(
        rng.beta(0.9, 2.0, 220),
        support=(0.0, 1.0),
        poly_degree=4,
        log_boundary_lower=True,
        log_boundary_upper=True,
    )
    loaded = Distribution(strip_cache(fitted.data))
    x = np.linspace(0.01, 0.99, 61)
    p = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6])
    np.testing.assert_allclose(loaded.pdf(x), fitted.pdf(x), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.cdf(x), fitted.cdf(x), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(loaded.ppf(p), fitted.ppf(p), rtol=2e-10, atol=2e-10)
    assert int(loaded.components[0].data["bulk_degree"]) == 4
    np.testing.assert_array_equal(
        loaded.components[0].data["boundary_allowed"],
        fitted.components[0].data["boundary_allowed"],
    )

    refit = Distribution().fit(rng.beta(1.0, 2.2, 180), init_from=loaded)
    assert int(refit.components[0].data["bulk_degree"]) == 4
    np.testing.assert_array_equal(
        refit.components[0].data["boundary_allowed"],
        loaded.components[0].data["boundary_allowed"],
    )


@pytest.mark.parametrize("drop_cache", [False, True])
def test_absent_provenance_yields_missing_historical_fit_records(drop_cache):
    fitted = Distribution().fit(np.random.default_rng(8109).normal(size=140))
    state = strip_provenance(fitted.data)
    if drop_cache:
        state = strip_cache(state)
    loaded = Distribution(state)
    diagnostics = loaded.fit_diagnostics
    assert diagnostics["provenance"] == "derived"
    assert diagnostics["converged"] is None
    assert diagnostics["n_face_parameters"] is None
    record = diagnostics["components"][0]
    assert record["success"] is None
    assert record["status"] is None
    assert record["message"] is None


@pytest.mark.parametrize("name", _SHAPES)
def test_same_version_cache_is_reused_for_every_model_shape(name, monkeypatch):
    """A freshly written cache is reused for every model shape.

    Parameters
    ----------
    name : str
        Shape key.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to count component rebuilds.
    """
    original = _fit_shape(name)
    state = original.data
    calls = _counting_rebuild(monkeypatch)
    reloaded = Distribution(state)
    copied = reloaded.copy()
    assert not calls, "a valid same-version cache must not be rebuilt"
    lo, hi = (float(original.ppf(q)) for q in (0.02, 0.98))
    x = np.linspace(lo, hi, 25)
    p = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6])
    for model in (reloaded, copied):
        np.testing.assert_array_equal(model.pdf(x), original.pdf(x))
        np.testing.assert_array_equal(model.cdf(x), original.cdf(x))
        np.testing.assert_array_equal(model.ppf(p), original.ppf(p))
    assert reloaded.default == original.default
    assert (reloaded.mu, reloaded.sigma) == (original.mu, original.sigma)


@pytest.mark.parametrize("name", _SHAPES)
def test_cacheless_rebuild_reproduces_every_model_shape(name):
    """The durable model alone rebuilds the same distribution.

    Parameters
    ----------
    name : str
        Shape key.
    """
    original = _fit_shape(name)
    rebuilt = Distribution(strip_cache(original.data))
    lo, hi = (float(original.ppf(q)) for q in (0.02, 0.98))
    x = np.linspace(lo, hi, 25)
    p = np.array([1e-6, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-6])
    np.testing.assert_allclose(rebuilt.pdf(x), original.pdf(x), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(rebuilt.cdf(x), original.cdf(x), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(rebuilt.ppf(p), original.ppf(p), rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(rebuilt.support, original.support)
    np.testing.assert_array_equal(rebuilt.weights, original.weights)
    assert rebuilt.default == original.default


def test_edited_cache_fails_content_digest_and_rebuilds_quietly(monkeypatch):
    """An edited cache is rejected by its digest, even under strict debug mode.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to count rebuilds and enable debug mode.
    """
    state = np.array(_fresh_state("single"), copy=True)
    state["cache"]["runtime_state"]["comp_cdf_npanels"][0] = -1
    calls = _counting_rebuild(monkeypatch)
    monkeypatch.setattr(defaults, "DEBUG", True)
    monkeypatch.setattr(defaults, "DEBUG_STRICT", True)
    gibbus.clear_suppressed_failures()
    loaded = Distribution(state)
    assert len(calls) == 1
    assert not _serialization_ledger_entries()
    x = np.linspace(-2.0, 2.0, 9)
    expected = Distribution(strip_cache(state))
    np.testing.assert_array_equal(loaded.pdf(x), expected.pdf(x))


def test_digest_verified_invalid_cache_raises_instead_of_hiding_a_defect():
    """A cache that passes both digests but is invalid can only be a defect."""
    state = np.array(_fresh_state("single"), copy=True)
    runtime = state["cache"]["runtime_state"]
    runtime["comp_cdf_npanels"][0] = -1
    # Simulate a writer defect: the same build wrote, and hashed, a bad cache.
    state["cache"]["runtime_sha256"] = _runtime_digest(runtime)
    with pytest.raises(ValueError):
        Distribution(state)
    # The durable model itself remains loadable without the cache.
    x = np.linspace(-2.0, 2.0, 9)
    expected = Distribution(_fresh_state("single"))
    np.testing.assert_array_equal(
        Distribution(strip_cache(state)).pdf(x), expected.pdf(x)
    )


def test_digest_verified_cache_disagreeing_with_model_raises():
    """A verified cache whose model fields differ reports a Gibbus defect."""
    state = np.array(_fresh_state("single"), copy=True)
    runtime = state["cache"]["runtime_state"]
    assert str(runtime["default_space"]) == "base"
    runtime["default_space"] = "exp"
    state["cache"]["runtime_sha256"] = _runtime_digest(runtime)
    with pytest.raises(ValueError, match="Gibbus defect"):
        Distribution(state)


def test_inadmissible_active_face_history_is_dropped():
    """History claiming an active log the model does not have is not applied."""
    fitted = Distribution().fit(np.random.default_rng(8119).normal(size=140))
    state = np.array(fitted.data, copy=True)
    assert not np.any(state["provenance"]["shared_boundary_allowed"])
    state["provenance"]["shared_boundary_active"][0] = True
    gibbus.clear_suppressed_failures()
    loaded = Distribution(strip_cache(state))
    diagnostics = loaded.fit_diagnostics
    assert diagnostics["provenance"] == "derived"
    assert diagnostics["components"][0]["status"] is None
    assert not _serialization_ledger_entries()
    np.testing.assert_array_equal(
        loaded.pdf([-1.0, 0.0, 1.0]), fitted.pdf([-1.0, 0.0, 1.0])
    )


def test_serialization_module_detects_malformed_data_without_exception_handling():
    """Optional sections are validated explicitly, per the coding standards."""
    tree = ast.parse(inspect.getsource(serialization))
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert not handlers


def test_missing_history_is_resaved_as_absent_provenance():
    """Placeholder history is never re-saved as if it were recorded."""
    fitted = Distribution().fit(np.random.default_rng(8116).normal(size=140))
    loaded = Distribution(strip_provenance(fitted.data))
    for resaved in (loaded.data, loaded.copy().data):
        assert resaved["provenance"].dtype.names == ("present",)
        assert not bool(resaved["provenance"]["present"])
        assert resaved["model"].tobytes() == fitted.data["model"].tobytes()
    again = Distribution(loaded.data)
    assert again.fit_diagnostics["components"][0]["status"] is None


def test_present_history_is_resaved_as_present_provenance():
    """Recorded history survives a load-and-save round trip."""
    fitted = Distribution().fit(np.random.default_rng(8117).normal(size=140))
    resaved = Distribution(fitted.data).data
    assert bool(resaved["provenance"]["present"])
    assert str(resaved["provenance"]["optimizer_status"][0]) != "missing"


def test_package_version_is_resolved_once_per_process():
    """The installed-metadata scan is not repeated on every pack or load."""
    fitted = Distribution().fit(np.random.default_rng(8118).normal(size=140))
    before = _package_version.cache_info().misses
    for _ in range(3):
        Distribution(fitted.data).copy()
    assert _package_version.cache_info().misses == before
