"""Fitted-state packing, validation and public evaluation tests."""

import numpy as np
import pytest

from gibbus import Distribution


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
    state = fitted.data
    kept = [name for name in state.dtype.names if name != "q_poly"]
    truncated = np.zeros((), dtype=[(n, state.dtype.fields[n][0]) for n in kept])
    for name in kept:
        truncated[name] = state[name]
    with pytest.raises(ValueError, match="missing fields: q_poly"):
        Distribution(truncated)


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
    np.testing.assert_array_equal(model.data["support"], [0.0, np.inf])
    assert model.ppf(0.0) == 0.0


def test_load_rejects_a_non_positive_scale():
    state = _saved_state()
    state["sigma"] = -1.0
    with pytest.raises(ValueError, match="sigma must be finite and positive"):
        Distribution().load(state)


def test_load_rejects_non_monotone_quantile_breakpoints():
    state = _saved_state()
    state["ppf_breaks_z"] = np.asarray(state["ppf_breaks_z"])[::-1].copy()
    with pytest.raises(ValueError, match="strictly increasing"):
        Distribution().load(state)


def test_load_rejects_non_finite_cdf_breakpoints():
    state = _saved_state()
    breaks = np.asarray(state["cdf_breaks"]).copy()
    breaks[2] = np.nan
    state["cdf_breaks"] = breaks
    with pytest.raises(ValueError, match="cdf_breaks"):
        Distribution().load(state)


def test_load_rejects_an_unknown_default_space():
    state = _saved_state()
    state["default_space"] = "zzz"
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
    weights = np.asarray(state["weights"]).copy()
    weights[0] = 5.0
    state["weights"] = weights
    with pytest.raises(ValueError, match="component weights"):
        Distribution().load(state)
