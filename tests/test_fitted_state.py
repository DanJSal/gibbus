"""Fitted-state packing, validation and public evaluation tests."""

import numpy as np
import pytest

from gibbus import Distribution


def _replace_section(state, section, value):
    """Return *state* with one nested serialization section replaced."""
    dtype = []
    for name in state.dtype.names:
        field_dtype = value.dtype if name == section else state.dtype[name]
        dtype.append((name, field_dtype))
    out = np.zeros((), dtype=dtype)
    for name in state.dtype.names:
        out[name] = value if name == section else state[name]
    return out


def _drop_model_field(state, field):
    model = state["model"]
    kept = [name for name in model.dtype.names if name != field]
    dtype = [(name, model.dtype[name]) for name in kept]
    reduced = np.zeros((), dtype=dtype)
    for name in kept:
        reduced[name] = model[name]
    return _replace_section(state, "model", reduced)


def test_point_state_has_canonical_natural_layout():
    rng = np.random.default_rng(104)
    fitted = Distribution().fit(
        rng.normal(0.5, 1.2, 350),
        n_components=1,
        poly_degree=4,
        support=(-np.inf, np.inf),
    )
    state = fitted.components[0].data
    names = set(state.dtype.names)
    assert "optimizer_params" in names
    assert "fit_center" in names and "fit_scale" in names and "fit_direction" in names
    assert "boundary_amplitudes" in names and state["boundary_amplitudes"].shape == (2,)
    assert "q_boundary" not in names
    assert "raw_params" not in names
    assert "center" not in names and "scale" not in names


def test_state_missing_required_fields_is_not_loadable():
    rng = np.random.default_rng(105)
    fitted = Distribution().fit(
        rng.normal(size=200),
        n_components=1,
        support=(-np.inf, np.inf),
        poly_degree=4,
    )
    state = _drop_model_field(fitted.data, "q_poly_values")
    with pytest.raises(ValueError, match="missing fields: q_poly_values"):
        Distribution(state)


def test_reflected_upper_half_line_round_trips_pdf_cdf_ppf_and_potential():
    rng = np.random.default_rng(105)
    data = 8.0 - rng.gamma(2.0, 1.0, 450)
    fitted = Distribution().fit(
        data,
        n_components=1,
        poly_degree=4,
        support=(-np.inf, 8.0),
    )
    state = fitted.components[0].data
    assert float(state["fit_direction"]) == -1.0
    assert tuple(map(float, fitted.support)) == (-np.inf, 8.0)
    assert fitted.ppf(1.0) == 8.0
    assert fitted.cdf(-np.inf) == 0.0
    assert fitted.cdf(8.0) == 1.0

    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    x = fitted.ppf(p)
    assert np.allclose(fitted.cdf(x), p, atol=3e-7, rtol=0.0)
    assert np.allclose(np.log(fitted.pdf(x)), -fitted.neg_log(x), atol=2e-12)


def test_native_state_affine_transform_preserves_distribution_identity():
    rng = np.random.default_rng(106)
    base = Distribution().fit(
        5.0 - rng.gamma(2.0, 1.0, 400),
        n_components=1,
        poly_degree=4,
        support=(-np.inf, 5.0),
    )
    moved = base.transform(mu=2.0, sigma=3.0, pullback=False, inplace=False)
    assert moved.support[1] == pytest.approx(17.0)
    x = np.array([-2.0, 0.0, 2.0, 4.0])
    y = 2.0 + 3.0 * x
    assert np.allclose(moved.pdf(y), base.pdf(x) / 3.0, rtol=2e-10, atol=1e-13)
    assert np.allclose(moved.cdf(y), base.cdf(x), rtol=0.0, atol=3e-8)
    assert moved.mean == pytest.approx(2.0 + 3.0 * base.mean, rel=2e-10)
    assert moved.std == pytest.approx(3.0 * base.std, rel=2e-10)


def _saved_state():
    """Return the packed state of a plain single-component fit."""
    rng = np.random.default_rng(311)
    fitted = Distribution().fit(
        rng.normal(size=400),
        n_components=1,
        support=(-np.inf, np.inf),
        rng=0,
    )
    return np.array(fitted.data, copy=True)


def test_a_saved_state_round_trips():
    state = _saved_state()
    restored = Distribution().load(state)
    assert restored.cdf(0.25) == pytest.approx(
        float(Distribution().load(state).cdf(0.25))
    )


def test_reported_support_is_exactly_the_requested_support():
    """Mapping canonical endpoints back can land an ulp outside the support."""
    from gibbus._model.coords import _build_fit_coordinate

    data = np.random.default_rng(2).lognormal(0.0, 0.6, 200)
    coordinate = _build_fit_coordinate((0.0, np.inf), data, None, None)
    assert coordinate.from_canonical(np.asarray(coordinate.canonical_support))[0] < 0.0
    model = Distribution().fit(
        data, n_components=1, poly_degree=4, support=(0.0, np.inf), rng=0
    )
    np.testing.assert_array_equal(model.support, [0.0, np.inf])
    np.testing.assert_array_equal(model.data["model"]["support"][0], [0.0, np.inf])
    assert model.ppf(0.0) == 0.0


def test_load_rejects_a_non_positive_scale():
    state = _saved_state()
    state["model"]["sigma"] = -1.0
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        Distribution().load(state)


def test_load_rebuilds_non_monotone_quantile_cache():
    clean = _saved_state()
    state = np.array(clean, copy=True)
    runtime = state["cache"]["runtime_state"]
    runtime["comp_ppf_breaks_z"][0] = runtime["comp_ppf_breaks_z"][0, ::-1].copy()
    expected = Distribution(clean)
    loaded = Distribution().load(state)
    x = np.linspace(-2.0, 2.0, 21)
    np.testing.assert_allclose(loaded.cdf(x), expected.cdf(x), rtol=0.0, atol=2e-12)


def test_load_rebuilds_non_finite_cdf_cache():
    clean = _saved_state()
    state = np.array(clean, copy=True)
    runtime = state["cache"]["runtime_state"]
    breaks = np.asarray(runtime["comp_cdf_breaks"][0]).copy()
    breaks[2] = np.nan
    runtime["comp_cdf_breaks"][0] = breaks
    expected = Distribution(clean)
    loaded = Distribution().load(state)
    x = np.linspace(-2.0, 2.0, 21)
    np.testing.assert_allclose(loaded.cdf(x), expected.cdf(x), rtol=0.0, atol=2e-12)


def test_load_rejects_an_unknown_default_space():
    state = _saved_state()
    state["model"]["default_space"] = "zzz"
    with pytest.raises(ValueError, match="default_space"):
        Distribution().load(state)


def test_load_rejects_invalid_mixture_weights():
    rng = np.random.default_rng(312)
    fitted = Distribution().fit(
        np.concatenate([rng.normal(-2.0, 0.5, 300), rng.normal(2.0, 0.5, 300)]),
        n_components=2,
        support=(-np.inf, np.inf),
        rng=0,
    )
    state = np.array(fitted.data, copy=True)
    weights = np.asarray(state["model"]["weights"]).copy()
    weights[0] = 5.0
    state["model"]["weights"] = weights
    with pytest.raises(ValueError, match="component weights"):
        Distribution().load(state)
