"""Subsampled component-count selection.

Subsampling applies only to *selecting* the integer K.  A subsampled winner
is refitted on all rows; a completed full-data winner is reused.  These tests
check agreement up to solver accuracy, rather than identical optimizer paths
or any particular speed-up.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from gibbus import Distribution
from gibbus._api import fitting as _fitting
from gibbus._api import selection as _selection
from gibbus._defaults import AUTO_KDE_GRID_POINTS
from gibbus._fit.mixture import (
    _binned_kde_sweep,
    _count_modes_kde,
    _gmm_init_responsibilities,
    _silverman_bandwidth,
    _stratified_subsample,
)


class TestStratifiedSubsample:
    def test_returns_requested_count(self):
        gen = np.random.default_rng(0)
        x = gen.normal(size=5000)
        assert _stratified_subsample(x, 500, gen).shape == (500,)

    def test_returns_all_when_m_exceeds_n(self):
        gen = np.random.default_rng(0)
        x = gen.normal(size=100)
        assert _stratified_subsample(x, 500, gen).shape == (100,)

    def test_indices_are_unique_and_in_range(self):
        gen = np.random.default_rng(1)
        x = gen.normal(size=5000)
        idx = _stratified_subsample(x, 400, gen)
        assert len(np.unique(idx)) == 400
        assert idx.min() >= 0 and idx.max() < 5000

    def test_preserves_distribution_shape(self):
        gen = np.random.default_rng(2)
        x = gen.normal(loc=3.0, scale=2.0, size=20000)
        sub = x[_stratified_subsample(x, 2000, gen)]
        assert sub.mean() == pytest.approx(x.mean(), abs=0.05)
        assert sub.std() == pytest.approx(x.std(), rel=0.05)

    def test_rare_component_survives(self):
        """A 1% cluster must appear in the subsample, not vanish."""
        gen = np.random.default_rng(3)
        x = np.concatenate([gen.normal(0, 1, 19800), gen.normal(20, 0.3, 200)])
        for seed in range(10):
            idx = _stratified_subsample(x, 2000, np.random.default_rng(seed))
            n_rare = int(np.sum(x[idx] > 15))
            assert n_rare >= 5, f"rare cluster nearly lost (seed {seed})"

    def test_lower_variance_than_simple_random(self):
        """Stratification is chosen for variance reduction; verify it."""
        gen = np.random.default_rng(4)
        x = np.concatenate([gen.normal(0, 1, 19800), gen.normal(20, 0.3, 200)])
        strat, simple = [], []
        for seed in range(30):
            g = np.random.default_rng(seed)
            strat.append(np.sum(x[_stratified_subsample(x, 2000, g)] > 15))
            simple.append(np.sum(x[g.choice(x.size, 2000, replace=False)] > 15))
        assert np.std(strat) < np.std(simple)


@pytest.fixture(scope="module")
def big_bimodal():
    rng = np.random.default_rng(20250101)
    return np.concatenate([rng.normal(-3, 0.7, 15000), rng.normal(3, 0.7, 15000)])


class TestSelectionAgreement:
    def test_subsampled_selection_matches_full(self, big_bimodal):
        # ``rng=None`` for fitting is deliberately equivalent to seed 0.
        sub = Distribution().fit(
            big_bimodal, n_components="auto", support=(-np.inf, np.inf)
        )
        full = Distribution().fit(
            big_bimodal,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
            auto_k_subsample=False,
        )
        assert sub.n_components == full.n_components == 2
        diag = sub.selection_diagnostics
        assert diag["selected_n_components"] == 2
        assert diag["subsampled"] is True
        assert diag["scores"]

    def test_fitted_parameters_are_unaffected(self, big_bimodal):
        """Selection is approximate; the refit that follows is not."""
        sub = Distribution().fit(
            big_bimodal, n_components="auto", support=(-np.inf, np.inf), rng=0
        )
        full = Distribution().fit(
            big_bimodal,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
            auto_k_subsample=False,
        )
        assert sub.mean == pytest.approx(full.mean, abs=1e-7)
        assert sub.var == pytest.approx(full.var, rel=1e-7)
        assert sub.weights == pytest.approx(full.weights, abs=1e-8)
        assert np.mean(sub.neg_log(big_bimodal)) == pytest.approx(
            np.mean(full.neg_log(big_bimodal)), abs=1e-9
        )
        assert not sub.selection_diagnostics["reuse_selected_fit"]
        assert full.selection_diagnostics["reuse_selected_fit"]

    def test_unimodal_still_selects_one(self):
        rng = np.random.default_rng(6)
        c = Distribution().fit(
            rng.normal(size=25000),
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
        )
        assert c.n_components == 1

    def test_small_input_is_not_subsampled(self):
        """Below the threshold, results must be exactly as before."""
        rng = np.random.default_rng(7)
        x = np.concatenate([rng.normal(-3, 0.7, 400), rng.normal(3, 0.7, 400)])
        a = Distribution().fit(x, n_components="auto", support=(-np.inf, np.inf), rng=0)
        b = Distribution().fit(
            x,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
            auto_k_subsample=False,
        )
        assert a.n_components == b.n_components
        assert a.mean == pytest.approx(b.mean, rel=1e-12)
        diag = a.selection_diagnostics
        assert diag["selected_n_components"] == a.n_components
        assert diag["subsampled"] is False
        assert diag["scores"]

    @pytest.mark.parametrize("bimodal", [False, True])
    def test_selection_candidates_remain_lite(self, monkeypatch, bimodal):
        """Discarded BIC candidates must not build spectral CDF/PPF state."""

        rng = np.random.default_rng(17)
        x = (
            np.concatenate([rng.normal(-3, 0.5, 75), rng.normal(3, 0.5, 75)])
            if bimodal
            else rng.normal(size=150)
        )
        original = _fitting._pack_natural_component
        original_single = _fitting._pack_natural_fit
        calls = 0
        single_calls = 0

        def counted_pack(component, **kwargs):
            nonlocal calls
            calls += 1
            return original(component, **kwargs)

        def counted_single(*args, **kwargs):
            nonlocal single_calls
            single_calls += 1
            return original_single(*args, **kwargs)

        monkeypatch.setattr(_fitting, "_pack_natural_component", counted_pack)
        monkeypatch.setattr(_fitting, "_pack_natural_fit", counted_single)
        c = Distribution().fit(x, n_components="auto", support=(-np.inf, np.inf), rng=0)

        # Selection candidates stay in natural solver state; only components
        # of the final selected model build spectral CDF/PPF state.
        assert calls + single_calls == c.n_components
        assert single_calls == int(c.n_components == 1)


class TestSubsampleOption:
    def test_explicit_size_accepted(self, big_bimodal):
        c = Distribution().fit(
            big_bimodal,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
            auto_k_subsample=3000,
        )
        assert c.n_components == 2

    def test_rejects_bad_string(self, big_bimodal):
        with pytest.raises(ValueError, match="must be 'auto'"):
            Distribution().fit(
                big_bimodal,
                n_components="auto",
                support=(-np.inf, np.inf),
                rng=0,
                auto_k_subsample="sometimes",
            )

    def test_rejects_degenerate_size(self, big_bimodal):
        with pytest.raises(ValueError, match="at least 2"):
            Distribution().fit(
                big_bimodal,
                n_components="auto",
                support=(-np.inf, np.inf),
                rng=0,
                auto_k_subsample=1,
            )

    def test_ignored_for_explicit_k(self, big_bimodal):
        """The option only affects automatic selection."""
        a = Distribution().fit(
            big_bimodal, n_components=2, support=(-np.inf, np.inf), rng=0
        )
        b = Distribution().fit(
            big_bimodal,
            n_components=2,
            support=(-np.inf, np.inf),
            rng=0,
            auto_k_subsample=500,
        )
        assert a.mean == pytest.approx(b.mean, rel=1e-12)


class TestWeightedSubsampling:
    def test_weights_are_renormalized_on_subsample(self, big_bimodal):
        w = np.ones(big_bimodal.size)
        w[big_bimodal.size // 2 :] = 5.0
        c = Distribution().fit(
            big_bimodal,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=0,
            sample_weight=w,
        )
        assert c.mean == pytest.approx(np.average(big_bimodal, weights=w), abs=0.1)


@pytest.mark.parametrize("intervals", [False, True])
def test_full_data_endpoint_exclusion_is_resolved_before_auto_k_thinning(
    monkeypatch, intervals
):
    x = np.r_[0.0, np.random.default_rng(96).beta(2, 3, 199)]
    rows = np.column_stack((x, x)) if intervals else x
    monkeypatch.setattr(
        _selection,
        "_stratified_subsample",
        lambda x, count, rng: np.arange(1, count + 1),
    )
    seen = []
    original = _selection._single_selection_fit

    def single(support, sample, degree, lower, upper, weights, degree_config):
        seen.append((lower, upper, sample))
        return original(support, sample, degree, lower, upper, weights, degree_config)

    monkeypatch.setattr(_selection, "_single_selection_fit", single)
    fitted = Distribution().fit(
        rows,
        n_components="auto",
        support=(0.0, 1.0),
        poly_degree=2,
        log_boundary_lower="auto",
        log_boundary_upper=False,
        auto_k_subsample=100,
        k_max=2,
        rng=0,
    )
    assert seen
    assert all(lower is False and upper is False for lower, upper, _ in seen)
    assert all(np.min(sample) > 0 for _, _, sample in seen)
    assert fitted.selection_diagnostics["screen_boundary_enabled"] == (False, False)


class TestBinnedKDE:
    """The FFT sweep must agree with a direct pairwise KDE."""

    @staticmethod
    def _reference(x, bw):
        """Pairwise Gaussian KDE on a grid, for one bandwidth."""
        lo, hi = float(x.min()), float(x.max())
        margin = 0.1 * (hi - lo)
        grid = np.linspace(lo - margin, hi + margin, AUTO_KDE_GRID_POINTS)
        z = (grid[:, None] - x[None, :]) / bw
        return np.exp(-0.5 * z * z).sum(axis=1) / (bw * np.sqrt(2 * np.pi) * x.size)

    def test_matches_pairwise_evaluation(self):

        rng = np.random.default_rng(4)
        x = np.concatenate([rng.normal(-2, 0.6, 400), rng.normal(2, 0.6, 400)])
        lo, hi = float(x.min()), float(x.max())
        margin = 0.1 * (hi - lo)
        grid = np.linspace(lo - margin, hi + margin, AUTO_KDE_GRID_POINTS)
        bws = _silverman_bandwidth(x) * np.array([0.5, 1.0, 3.0])

        got = _binned_kde_sweep(x, grid, bws)
        for i, bw in enumerate(bws):
            ref = self._reference(x, bw)
            # Linear binning leaves an error of order (dx / bandwidth)^2,
            # around 5e-6 relative at this grid resolution. What actually
            # has to survive it is the local-maxima count, which
            # test_recovers_known_mode_count pins separately.
            assert np.max(np.abs(got[i] - ref)) < 1e-4 * ref.max()

    def test_densities_integrate_to_one(self):

        rng = np.random.default_rng(5)
        x = rng.normal(size=1000)
        grid = np.linspace(x.min() - 1.0, x.max() + 1.0, 2048)
        bws = _silverman_bandwidth(x) * np.array([0.5, 1.0, 2.0])
        dens = _binned_kde_sweep(x, grid, bws)
        for row in dens:
            assert abs(trapezoid(row, grid) - 1.0) < 1e-3

    @pytest.mark.parametrize(
        "specs,expected",
        [
            ([(0.0, 1.0, 1.0)], 1),
            ([(-3.0, 0.7, 0.5), (3.0, 0.7, 0.5)], 2),
            ([(-5.0, 0.8, 0.35), (0.0, 1.0, 0.3), (5.0, 0.6, 0.35)], 3),
        ],
    )
    def test_recovers_known_mode_count(self, specs, expected):

        rng = np.random.default_rng(6)
        x = np.concatenate([rng.normal(m, s, int(6000 * w)) for m, s, w in specs])
        assert (
            _count_modes_kde(
                np.ascontiguousarray(x), verbose=0, rng=np.random.default_rng(0)
            )
            == expected
        )


class TestGMMInit:
    """Base-space initialization contract."""

    def test_returns_normalized_responsibilities(self):

        rng = np.random.default_rng(8)
        x = np.concatenate([rng.normal(-3, 0.7, 500), rng.normal(3, 0.7, 500)])
        resp, weights = _gmm_init_responsibilities(
            np.ascontiguousarray(x), 2, np.random.default_rng(0)
        )
        assert resp.shape == (x.size, 2)
        assert np.allclose(resp.sum(axis=1), 1.0)
        assert np.isclose(weights.sum(), 1.0)

    def test_is_invariant_to_monotone_rescaling(self):
        """A scale change must not flip which space the init is built in."""

        rng = np.random.default_rng(9)
        x = np.ascontiguousarray(
            np.concatenate([rng.normal(-3, 0.7, 500), rng.normal(3, 0.7, 500)])
        )
        a, _ = _gmm_init_responsibilities(x, 2, np.random.default_rng(0))
        b, _ = _gmm_init_responsibilities(10.0 * x, 2, np.random.default_rng(0))
        assert np.allclose(np.sort(a, axis=1), np.sort(b, axis=1), atol=1e-6)
