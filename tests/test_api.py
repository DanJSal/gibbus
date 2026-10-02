"""Round-tripping, weighting, transforms, and kernel/NumPy agreement."""

import copy
import pickle

import numpy as np
import pytest

from gibbus import Distribution


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(4242)


@pytest.fixture(scope="module")
def bimodal(rng):
    return np.concatenate([rng.normal(-3, 0.7, 500),
                           rng.normal(3, 0.7, 500)])


class TestSampleWeights:
    """Observation weights must affect single and mixture fits alike."""

    def test_weights_shift_single_component_mean(self, rng, bimodal):
        w = np.ones(bimodal.size)
        w[bimodal.size // 2:] = 20.0
        c = Distribution().fit(bimodal, n_components=1, support=(-np.inf, np.inf),
                        sample_weights=w)
        assert c.mean == pytest.approx(np.average(bimodal, weights=w),
                                       abs=0.05)

    def test_weights_shift_mixture_weights(self, bimodal):
        w = np.ones(bimodal.size)
        w[bimodal.size // 2:] = 20.0
        c = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0, sample_weights=w)
        expected = w[bimodal.size // 2:].sum() / w.sum()
        assert c.weights[1] == pytest.approx(expected, abs=0.02)
        assert c.mean == pytest.approx(np.average(bimodal, weights=w),
                                       abs=0.05)

    def test_weighted_mixture_differs_from_unweighted(self, bimodal):
        w = np.ones(bimodal.size)
        w[bimodal.size // 2:] = 20.0
        a = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0)
        b = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0, sample_weights=w)
        assert not np.allclose(a.weights, b.weights)

    def test_uniform_weights_match_unweighted(self, rng):
        data = rng.normal(size=600)
        a = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))
        b = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf),
                        sample_weights=np.ones(600))
        assert b.mean == pytest.approx(a.mean, rel=1e-9)
        assert b.var == pytest.approx(a.var, rel=1e-9)

    def test_weights_respected_at_explicit_k_one(self, rng, bimodal):
        """The explicit n_components=1 path must not drop weights."""
        w = np.ones(bimodal.size)
        w[bimodal.size // 2:] = 20.0
        explicit = Distribution().fit(bimodal, n_components=1,
                               support=(-np.inf, np.inf), sample_weights=w)
        assert explicit.mean == pytest.approx(
            np.average(bimodal, weights=w), abs=0.05)


class TestSerialisation:
    def test_single_roundtrip_is_exact(self, rng, tmp_path):
        c = Distribution().fit(rng.normal(size=600), n_components=1,
                        support=(-np.inf, np.inf))
        path = tmp_path / "m.npy"
        np.save(path, c.data)
        loaded = Distribution(np.load(path, allow_pickle=False))
        xs = np.linspace(-4, 4, 50)
        assert np.array_equal(loaded.pdf(xs), c.pdf(xs))
        assert loaded.mean == c.mean

    def test_mixture_roundtrip_is_exact(self, bimodal, tmp_path):
        c = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0)
        path = tmp_path / "mm.npy"
        np.save(path, c.data)
        loaded = Distribution(np.load(path, allow_pickle=False))
        assert loaded.n_components == 2
        assert np.array_equal(loaded.weights, c.weights)
        xs = np.linspace(-8, 8, 50)
        assert np.array_equal(loaded.pdf(xs), c.pdf(xs))

    def test_load_method_equivalent_to_constructor(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        assert Distribution().load(c.data).mean == c.mean

    def test_pickle_roundtrip_uses_serialized_state(self, bimodal):
        c = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0)
        before_diag = c.fit_diagnostics
        loaded = pickle.loads(pickle.dumps(c))
        xs = np.linspace(-8.0, 8.0, 50)
        assert loaded.n_components == c.n_components
        assert np.array_equal(loaded.weights, c.weights)
        assert np.array_equal(loaded.pdf(xs), c.pdf(xs))
        assert before_diag["em"] is not None
        assert all(item["success"] for item in before_diag["components"])
        assert all(item["success"] for item in loaded.fit_diagnostics["components"])

    def test_repr_summarizes_fitted_state(self, rng):
        c = Distribution().fit(rng.normal(size=200), n_components=1,
                        support=(-np.inf, np.inf), poly_degree=2)
        text = repr(c)
        assert text.startswith("Distribution(")
        assert "components=1" in text
        assert "support=" in text

    def test_no_pickle_required(self, rng, tmp_path):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        path = tmp_path / "np.npy"
        np.save(path, c.data)
        loaded = np.load(path, allow_pickle=False)  # must not raise
        c2 = Distribution(loaded)
        xs = np.linspace(-3.0, 3.0, 50)
        assert np.array_equal(c2.pdf(xs), c.pdf(xs))
        assert c2.n_components == c.n_components

    def test_single_state_is_spectral_only(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        names = set(c.data.dtype.names)
        assert {
            "cdf_map_kind", "cdf_breaks", "cdf_offsets", "cdf_coeffs",
            "ppf_breaks_r", "ppf_breaks_z", "ppf_coeffs",
        } <= names
        assert not any("interp" in name.lower() for name in names)

    def test_mixture_state_is_spectral_only(self, bimodal):
        c = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0)
        names = set(c.data.dtype.names)
        assert "comp_cdf_coeffs" in names
        assert "comp_ppf_coeffs" in names
        assert not any("interp" in name.lower() for name in names)

    def test_roundtrip_preserves_cdf_and_ppf(self, rng, tmp_path):
        c = Distribution().fit(rng.normal(size=500), n_components=1,
                        support=(-np.inf, np.inf))
        path = tmp_path / "spectral.npy"
        np.save(path, c.data)
        loaded = Distribution(np.load(path, allow_pickle=False))
        xs = np.linspace(-4, 4, 101)
        ps = np.array([1e-13, 1e-12, 0.01, 0.5, 0.99, 1.0 - 1e-12,
                       1.0 - 1e-13])
        assert np.array_equal(loaded.cdf(xs), c.cdf(xs))
        assert np.array_equal(loaded.ppf(ps), c.ppf(ps))

    def test_load_rejects_invalid_spectral_state_dimensions(self, rng):
        c = Distribution().fit(rng.normal(size=300), n_components=1,
                        support=(-np.inf, np.inf))
        for field in (
            "cdf_npanels",
            "cdf_coeff_stride",
            "ppf_npanels",
            "ppf_coeff_stride",
        ):
            state = np.array(c.data, copy=True)
            state[field] = -1
            with pytest.raises(ValueError, match="must be positive"):
                Distribution(state)

    def test_load_rejects_invalid_spectral_map_kind(self, rng):
        c = Distribution().fit(rng.normal(size=300), n_components=1,
                        support=(-np.inf, np.inf))
        for kind in (-1, 99):
            state = np.array(c.data, copy=True)
            state["cdf_map_kind"] = kind
            with pytest.raises(ValueError, match=r"cdf_map_kind.*0\.\.5"):
                Distribution(state)

    @pytest.mark.parametrize("field,stride_field", [
        ("cdf_ncoeff", "cdf_coeff_stride"),
        ("ppf_ncoeff", "ppf_coeff_stride"),
    ])
    def test_load_rejects_panel_coefficient_count_outside_stride(
        self, rng, field, stride_field
    ):
        c = Distribution().fit(rng.normal(size=300), n_components=1,
                        support=(-np.inf, np.inf))
        for value in (0, int(c.data[stride_field]) + 1):
            state = np.array(c.data, copy=True)
            counts = np.array(state[field], copy=True)
            counts[-1] = value
            state[field] = counts
            with pytest.raises(ValueError, match=r"ncoeff entries must be in"):
                Distribution(state)


class TestTransform:
    def test_pushforward_shifts_and_scales(self, rng):
        c = Distribution().fit(rng.normal(size=600), n_components=1,
                        support=(-np.inf, np.inf))
        t = c.transform(mu=5.0, sigma=2.0, pullback=False, inplace=False)
        assert t.mean == pytest.approx(5.0 + 2.0 * c.mean)
        assert t.std == pytest.approx(2.0 * c.std)

    def test_pushforward_preserves_density_mass(self, rng):
        c = Distribution().fit(rng.normal(size=600), n_components=1,
                        support=(-np.inf, np.inf))
        t = c.transform(mu=5.0, sigma=2.0, pullback=False, inplace=False)
        assert t.pdf(5.0 + 2.0 * 0.3) * 2.0 == pytest.approx(c.pdf(0.3))

    def test_not_inplace_leaves_original_alone(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        before = c.mean
        c.transform(mu=9.0, sigma=1.0, pullback=False, inplace=False)
        assert c.mean == before

    def test_deepcopy_is_independent(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        before = c.mean
        d = copy.deepcopy(c)
        d.transform(mu=9.0, sigma=1.0, pullback=False)
        assert c.mean == before
        assert d.mean != before

    def test_copy_is_independent(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        assert c.copy().mean == c.mean


class TestRefitting:
    def test_refit_invalidates_caches(self, rng):
        """A second fit() must not serve results from the first."""
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        c.cdf(0.0)          # populate caches
        c.fit(rng.normal(loc=50, scale=1, size=400), n_components=1,
              support=(-np.inf, np.inf))
        assert c.mean == pytest.approx(50.0, abs=0.3)
        assert c.cdf(50.0) == pytest.approx(0.5, abs=0.05)
        assert c.cdf(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_mixture_to_single_refit(self, bimodal, rng):
        c = Distribution().fit(bimodal, n_components=2, support=(-np.inf, np.inf),
                        rng=0)
        c.cdf(0.0)
        c.fit(rng.normal(size=400), n_components=1,
              support=(-np.inf, np.inf))
        assert c.n_components == 1
        assert c.cdf(0.0) == pytest.approx(0.5, abs=0.1)


class TestExpSpace:
    def test_change_of_variables(self, rng):
        c = Distribution().fit(rng.gamma(2.0, 1.0, size=800), n_components=1,
                        support=(0, np.inf))
        y = 2.0
        assert c.exp.pdf(y) == pytest.approx(c.base.pdf(np.log(y)) / y)

    def test_set_default_switches_active_space(self, rng):
        c = Distribution().fit(rng.gamma(2.0, 1.0, size=800), n_components=1,
                        support=(0, np.inf))
        base_mean = c.mean
        c.set_default("exp")
        assert c.mean != base_mean
        assert c.mean == pytest.approx(c.exp.mean)
        c.set_default("base")
        assert c.mean == pytest.approx(base_mean)

    def test_exp_support_is_positive(self, rng):
        c = Distribution().fit(rng.gamma(2.0, 1.0, size=800), n_components=1,
                        support=(0, np.inf))
        assert c.exp.support[0] >= 0.0


class TestIntervalCensored:
    def test_interval_fit_approximates_point_fit(self, rng):
        x = rng.normal(size=800)
        point = Distribution().fit(x, n_components=1, support=(-np.inf, np.inf))
        intervals = np.column_stack([x - 0.05, x + 0.05])
        censored = Distribution().fit(intervals, n_components=1,
                               support=(-np.inf, np.inf))
        assert censored.mean == pytest.approx(point.mean, abs=0.05)
        assert censored.std == pytest.approx(point.std, rel=0.1)

    def test_reversed_intervals_are_swapped(self, rng):
        x = rng.normal(size=400)
        forward = np.column_stack([x - 0.05, x + 0.05])
        reverse = np.column_stack([x + 0.05, x - 0.05])
        a = Distribution().fit(forward, n_components=1, support=(-np.inf, np.inf))
        b = Distribution().fit(reverse, n_components=1, support=(-np.inf, np.inf))
        assert b.mean == pytest.approx(a.mean, rel=1e-9)


class TestWarmStart:
    def test_init_from_reproduces_similar_fit(self, rng):
        x = rng.normal(size=800)
        base = Distribution().fit(x, n_components=1, support=(-np.inf, np.inf))
        seeded = Distribution().fit(x, init_from=base)
        assert seeded.mean == pytest.approx(base.mean, abs=1e-3)

    def test_init_from_requires_fitted(self, rng):
        with pytest.raises(ValueError, match="must be a fitted"):
            Distribution().fit(rng.normal(size=100), init_from=Distribution())

    def test_init_from_rejects_non_distribution(self, rng):
        with pytest.raises(TypeError, match="must be a fitted Distribution"):
            Distribution().fit(rng.normal(size=100), init_from="not a distribution")


class TestScalarVsArray:
    @pytest.mark.parametrize("method,arg", [
        ("pdf", 0.0), ("cdf", 0.0), ("ppf", 0.5), ("neg_log", 0.0),
    ])
    def test_scalar_returns_float(self, method, arg, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        assert isinstance(getattr(c, method)(arg), float)

    @pytest.mark.parametrize("method,arg", [
        ("pdf", [0.0, 1.0]), ("cdf", [0.0, 1.0]),
        ("ppf", [0.25, 0.75]), ("neg_log", [0.0, 1.0]),
    ])
    def test_array_returns_ndarray(self, method, arg, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        out = getattr(c, method)(arg)
        assert isinstance(out, np.ndarray) and out.shape == (2,)

    def test_sample_none_returns_scalar(self, rng):
        c = Distribution().fit(rng.normal(size=400), n_components=1,
                        support=(-np.inf, np.inf))
        assert isinstance(c.sample(rng=1), float)


class TestMomentConventions:
    """Which kurtosis convention the public surface reports.

    ``kurt`` is the Pearson (raw) kurtosis -- the standardized fourth
    moment, the counterpart of ``skew`` -- so a Gaussian reads 3.0.  The
    docstrings said "excess" in four places while the code returned raw,
    and the docs were the side that was wrong: ``skew`` and ``kurt`` are
    written as a parallel pair, and ``kurt`` is a stored field in
    ``.data``, so changing its meaning would have silently changed what
    the serialized format means.

    These tests pin the convention so the two cannot drift apart again.
    """

    def test_kurt_is_the_standardized_fourth_moment(self):
        rng = np.random.default_rng(4)
        data = np.ascontiguousarray(rng.normal(0.0, 1.0, 6000))
        c = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))

        # A Gaussian has raw kurtosis 3 and excess 0.
        assert c.kurt == pytest.approx(3.0, abs=0.05)
        assert c.kurt == pytest.approx(c.moment(4, standardized=True), abs=1e-9)

    def test_skew_and_kurt_are_a_parallel_pair(self):
        """Both are standardized moments, with no Fisher correction."""
        rng = np.random.default_rng(5)
        data = np.ascontiguousarray(rng.gamma(3.0, 1.0, 6000))
        c = Distribution().fit(data, n_components=1, support=(0.0, np.inf))

        assert c.skew == pytest.approx(c.moment(3, standardized=True), abs=1e-9)
        assert c.kurt == pytest.approx(c.moment(4, standardized=True), abs=1e-9)

    def test_uniform_kurtosis_is_below_the_gaussian_value(self):
        rng = np.random.default_rng(6)
        data = np.ascontiguousarray(rng.uniform(0.0, 1.0, 6000))
        c = Distribution().fit(data, n_components=1, support=(0.0, 1.0))
        # Raw kurtosis of a uniform is 1.8 (excess -1.2).
        assert c.kurt == pytest.approx(1.8, abs=0.1)


@pytest.fixture(scope="module")
def fitted():
    """Return one shared single-component fit for the convention tests."""
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(rng.normal(size=500))
    return Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))


class TestEvaluationConventions:
    """Cross-cutting behaviour of the evaluators that is easy to break
    one method at a time: NaN propagation, argument validation, and
    scale invariance of the fit."""

    def test_nan_evaluation_contract(self, fitted):
        """Density/CDF propagate NaN, while PPF rejects non-finite probabilities."""
        for space in (fitted.base, fitted.exp):
            assert np.isnan(space.pdf(np.nan))
            assert np.isnan(space.cdf(np.nan))
            with pytest.raises(ValueError, match="finite p"):
                space.ppf(np.nan)
        with pytest.raises(ValueError, match="finite p"):
            fitted.ppf(np.nan)
        out = fitted.pdf(np.array([0.0, np.nan, 1.0]))
        assert np.isfinite(out[0]) and np.isnan(out[1]) and np.isfinite(out[2])

    def test_pdf_preserves_shape_and_scalar_type(self, fitted):
        assert isinstance(fitted.pdf(0.5), float)
        assert fitted.pdf(np.zeros((3, 4))).shape == (3, 4)
        assert fitted.pdf(np.array([])).shape == (0,)

    @pytest.mark.parametrize("bad", [2.5, -1, True, "2"])
    def test_moment_rejects_non_integer_order(self, fitted, bad):
        """Moment order must be a non-negative integer."""
        with pytest.raises(ValueError, match="non-negative integer"):
            fitted.moment(bad)

    def test_moment_accepts_integral_floats_and_numpy_ints(self, fitted):
        m2 = fitted.moment(2)
        assert fitted.moment(2.0) == m2
        assert fitted.moment(np.int64(2)) == m2

    def test_n_components_rejects_junk_with_a_clear_message(self):
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            Distribution().fit(np.arange(50, dtype=float), n_components="bogus",
                        support=(-np.inf, np.inf))

    @pytest.mark.parametrize("scale", [1e-30, 1e-8, 1.0, 1e8, 1e30])
    def test_fit_is_scale_invariant(self, scale):
        """The spike-detection tolerance had an absolute floor of 1.0 in its
        reference scale, so data at scale < ~1e-7 was rejected as degenerate
        even though it was perfectly well-conditioned."""
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(rng.normal(size=400))
        ref = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf), poly_degree=4)
        c = Distribution().fit(data * scale, n_components=1, support=(-np.inf, np.inf), poly_degree=4)
        assert c.std / scale == pytest.approx(ref.std, rel=3e-6)
        assert c.mean / scale == pytest.approx(ref.mean, abs=1e-6)
        assert c.kurt == pytest.approx(ref.kurt, rel=1e-6)


def test_infinite_censored_single_component_uses_native_natural_path():
    rng = np.random.default_rng(333)
    x = rng.normal(0.3, 1.1, size=500)
    rows = []
    for value in x:
        if value < -1.0:
            rows.append((-np.inf, -1.0))
        elif value > 1.4:
            rows.append((1.4, np.inf))
        else:
            rows.append((value - 0.04, value + 0.04))
    intervals = np.asarray(rows, dtype=float)

    explicit = Distribution().fit(
        intervals, n_components=1, support=(-np.inf, np.inf), poly_degree=2
    )
    auto = Distribution().fit(
        intervals, n_components=1, support=(-np.inf, np.inf), poly_degree="auto"
    )
    for fitted in (explicit, auto):
        assert str(fitted.data["optimizer_status"]) in ("converged", "converged_approximately")
    assert int(auto.data["requested_poly_degree"]) == 2
    assert explicit.mean == pytest.approx(0.3, abs=0.16)
    assert explicit.std == pytest.approx(1.1, rel=0.18)
