"""Component-count selection battery.

Everything that touches mixture initialization -- the KDE mode sweep, the
GMM screen, the log-concave BIC sweep -- changes which *K* comes out the
other end.  Nothing else in the suite would catch a regression in that,
so this module pins the selected component count across a spread of
shapes: separation, weight imbalance, skew, heavy tails, bounded support,
and genuine unimodality.

The cases span reproducible full-line, bounded, half-line, skewed, and
heavy-tailed shapes.  Exact component counts are asserted where the model
selection policy is deterministic for the fixture.

Sample sizes are kept at a few thousand: the properties under test are
about which shapes are distinguishable, not about asymptotics.
"""

import numpy as np
import pytest

from gibbus import Distribution
from gibbus._defaults import AUTO_LC_MIN_COMPONENT_N
from gibbus._fit.mixture import (
    _count_modes_kde,
    _init_responsibilities,
    _valley_init_responsibilities,
)

N = 4000


def _mixture(rng, specs, n=N):
    """Draw from a Gaussian mixture given ``(mean, sd, weight)`` triples."""
    return np.ascontiguousarray(
        np.concatenate([rng.normal(m, s, int(n * w)) for m, s, w in specs])
    )


def _bic(distribution, data):
    """BIC of a fitted mixture on *data*; lower is better."""
    ll = float(np.mean(np.log(np.clip(distribution.pdf(data), 1e-300, None))))
    p = int(distribution.fit_diagnostics["n_face_parameters"])
    return -2.0 * ll * data.size + p * np.log(data.size)


# (name, builder, support, expected K)
STABLE_CASES = [
    ("unimodal normal", lambda r: _mixture(r, [(0, 1, 1.0)]), (-np.inf, np.inf), 1),
    (
        "bimodal wide",
        lambda r: _mixture(r, [(-4, 0.7, 0.5), (4, 0.7, 0.5)]),
        (-np.inf, np.inf),
        2,
    ),
    (
        "bimodal moderate",
        lambda r: _mixture(r, [(-2, 0.8, 0.5), (2, 0.8, 0.5)]),
        (-np.inf, np.inf),
        2,
    ),
    (
        "trimodal",
        lambda r: _mixture(r, [(-6, 0.8, 0.35), (0, 1, 0.3), (6, 0.6, 0.35)]),
        (-np.inf, np.inf),
        3,
    ),
    (
        "minor component 10%",
        lambda r: _mixture(r, [(0, 1, 0.9), (6, 0.6, 0.1)]),
        (-np.inf, np.inf),
        2,
    ),
    (
        "minor component 5%",
        lambda r: _mixture(r, [(0, 1, 0.95), (6, 0.5, 0.05)]),
        (-np.inf, np.inf),
        2,
    ),
    (
        "beta(2,5) bounded",
        lambda r: np.ascontiguousarray(r.beta(2, 5, N)),
        (0.0, 1.0),
        1,
    ),
]


class TestStableSelection:
    """Shapes where the selected component count must not drift."""

    @pytest.mark.parametrize("name,build,support,expected", STABLE_CASES)
    @pytest.mark.parametrize("seed", [0, 1])
    def test_selects_expected_k(self, name, build, support, expected, seed):
        data = build(np.random.default_rng(seed))
        c = Distribution().fit(data, n_components="auto", support=support, rng=seed)
        assert c.n_components in np.atleast_1d(expected)

    @pytest.mark.parametrize("name,build,support,expected", STABLE_CASES)
    def test_fit_is_a_valid_density(self, name, build, support, expected):
        data = build(np.random.default_rng(0))
        c = Distribution().fit(data, n_components="auto", support=support, rng=0)
        grid = np.linspace(*np.percentile(data, [1, 99]), 50)
        pdf = c.pdf(grid)
        assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)
        assert np.isclose(c.weights.sum(), 1.0)


class TestSelectionIsJustified:
    """The chosen K should be the one BIC actually prefers."""

    def test_bimodal_prefers_two_over_one(self):
        rng = np.random.default_rng(0)
        data = _mixture(rng, [(-4, 0.7, 0.5), (4, 0.7, 0.5)])
        one = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))
        two = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
        assert _bic(two, data) < _bic(one, data)

    def test_unimodal_prefers_one_over_two(self):
        rng = np.random.default_rng(0)
        data = _mixture(rng, [(0, 1, 1.0)])
        one = Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))
        two = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
        assert _bic(one, data) < _bic(two, data)


class TestHeavyTailRobustness:
    """A heavy tail must not make failure depend on which K was asked for.

    A lognormal is not log-concave, so the documented remedy is to fit a
    mixture; every K must fit, and quietly: line-search trials far outside
    the data must not leak integration warnings.
    """

    @pytest.mark.filterwarnings("error::scipy.integrate.IntegrationWarning")
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_every_k_fits(self, k):
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(rng.lognormal(0, 0.6, N))
        c = Distribution().fit(
            data,
            n_components=k,
            support=(0.0, np.inf),
            poly_degree=6,
            progressive=False,
            rng=0,
        )
        assert c.n_components == k
        pdf = c.pdf(np.linspace(0.05, 8.0, 40))
        assert np.all(np.isfinite(pdf)) and np.all(pdf >= 0.0)

    def test_more_components_do_not_fit_worse(self):
        """BIC across K should be well-ordered, not jump around."""
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(rng.lognormal(0, 0.6, N))
        lls = []
        for k in (1, 2, 3):
            c = Distribution().fit(
                data,
                n_components=k,
                support=(0.0, np.inf),
                poly_degree=6,
                progressive=False,
                rng=0,
            )
            lls.append(float(np.mean(np.log(np.clip(c.pdf(data), 1e-300, None)))))
        # Each extra component has strictly more capacity, so the
        # attained log-likelihood should not go backwards by more than
        # EM's own convergence slack.
        assert lls[1] >= lls[0] - 1e-3
        assert lls[2] >= lls[1] - 1e-3

    def test_k2_converges_with_certified_natural_components(self):
        """A well-conditioned K=2 fit should converge and certify both components."""
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(rng.lognormal(0, 0.6, 20_000))
        c = Distribution().fit(
            data,
            n_components=2,
            support=(0.0, np.inf),
            poly_degree=6,
            progressive=False,
            rng=0,
            verbose=1,
        )
        diagnostics = c.fit_diagnostics
        assert c.n_components == 2
        assert diagnostics["converged"]
        assert diagnostics["em"]["converged"]
        assert all(
            record["separator_certified"] for record in diagnostics["components"]
        )


class TestSkewedSelection:
    """Component selection on skewed and bounded-support shapes.

    The KDE proposal stage suppresses weak tail ripples, and candidate models
    are scored at the configured validation degrees.  These fixtures pin the
    resulting selection policy across several deterministic seeds.
    """

    @pytest.mark.parametrize("seed", range(8))
    def test_gamma_selects_one_component(self, seed):
        rng = np.random.default_rng(seed)
        data = np.ascontiguousarray(rng.gamma(2.0, 1.0, N))
        assert (
            Distribution()
            .fit(data, n_components="auto", support=(0.0, np.inf), rng=seed)
            .n_components
            == 1
        )

    @pytest.mark.parametrize(
        "dist,kwargs",
        [
            ("gamma", {"shape": 5.0, "scale": 1.0}),
            ("exponential", {"scale": 1.0}),
        ],
    )
    def test_other_log_concave_shapes_select_one_component(self, dist, kwargs):
        """Exponential is the sharpest boundary case on the half line."""
        for seed in range(3):
            rng = np.random.default_rng(seed)
            data = np.ascontiguousarray(getattr(rng, dist)(size=N, **kwargs))
            assert (
                Distribution()
                .fit(data, n_components="auto", support=(0.0, np.inf), rng=seed)
                .n_components
                == 1
            )

    @pytest.mark.parametrize("shape", [2.0, 5.0])
    def test_gamma_shapes_select_one_component(self, shape):
        for seed in range(3):
            rng = np.random.default_rng(seed)
            data = np.ascontiguousarray(rng.gamma(shape, 1.0, N))
            assert (
                Distribution()
                .fit(data, n_components="auto", support=(0.0, np.inf), rng=seed)
                .n_components
                == 1
            )

    def test_mode_sweep_is_exact_on_gamma(self):
        """Prominence filtering should report one stable mode for gamma data."""
        counts = [
            _count_modes_kde(
                np.ascontiguousarray(np.random.default_rng(s).gamma(2.0, 1.0, N)),
                verbose=0,
                rng=np.random.default_rng(0),
            )
            for s in range(8)
        ]
        assert all(m == 1 for m in counts)

    @pytest.mark.parametrize("seed", range(4))
    def test_gamma_selection_agrees_with_converged_bic(self, seed):
        """The selected K must be the one a converged BIC prefers, among
        candidates that carry enough support to be estimable.

        BIC charges for parameters, not for observations, so a converged
        fit can prefer a K whose extra component is supported by a couple
        of dozen tail points -- fewer than five per free parameter.
        ``AUTO_LC_MIN_COMPONENT_N`` excludes those by policy, so a rival
        below it is not evidence that selection chose wrongly.
        """

        rng = np.random.default_rng(seed)
        data = np.ascontiguousarray(rng.gamma(2.0, 1.0, N))
        chosen = Distribution().fit(
            data, n_components="auto", support=(0.0, np.inf), rng=seed
        )
        rival = Distribution().fit(
            data, n_components=chosen.n_components + 1, support=(0.0, np.inf), rng=seed
        )

        rival_support = float(np.min(rival.weights)) * data.size
        if rival_support < AUTO_LC_MIN_COMPONENT_N:
            # This rival is outside the auto-selection model class by policy,
            # so there is no admissible BIC disagreement to compare.  The
            # correct result is the already-selected one-component model, not
            # an informational pytest skip.
            assert chosen.n_components == 1
            return
        assert _bic(chosen, data) < _bic(rival, data)


class TestValleyInitialization:
    """Mixture initialization from KDE modes and the valleys between them.

    The initializer is deterministic, honors ``sample_weights``, and reuses
    the same KDE structure used by component-count proposal.
    """

    @pytest.mark.parametrize(
        "specs,k",
        [
            ([(-4, 0.7, 0.5), (4, 0.7, 0.5)], 2),
            ([(-6, 0.8, 0.35), (0, 1, 0.3), (6, 0.6, 0.35)], 3),
            ([(-6, 0.5, 0.25), (-2, 0.5, 0.25), (2, 0.5, 0.25), (6, 0.5, 0.25)], 4),
            ([(0, 1, 0.99), (6, 0.5, 0.01)], 2),  # 1% minor component
        ],
    )
    def test_resolves_separable_shapes(self, specs, k):

        resp, mix = _valley_init_responsibilities(
            _mixture(np.random.default_rng(0), specs), k
        )
        assert resp is not None, "a valley exists here; init should not decline"
        assert resp.shape[1] == k
        assert np.allclose(resp.sum(axis=1), 1.0)
        assert np.isclose(mix.sum(), 1.0)

    def test_declines_when_no_valley_exists(self):
        """Components sharing a location have no valley at any bandwidth."""

        data = _mixture(np.random.default_rng(0), [(0, 0.5, 0.5), (0, 3.0, 0.5)])
        resp, _ = _valley_init_responsibilities(data, 2)
        assert resp is None

    def test_falls_back_rather_than_failing(self):
        """The wrapper must still return usable responsibilities there."""

        data = _mixture(np.random.default_rng(0), [(0, 0.5, 0.5), (0, 3.0, 0.5)])
        resp, _mix = _init_responsibilities(data, 2, np.random.default_rng(0))
        assert resp is not None and resp.shape[1] == 2
        assert np.allclose(resp.sum(axis=1), 1.0)

    def test_is_deterministic(self):

        data = _mixture(np.random.default_rng(0), [(-4, 0.7, 0.5), (4, 0.7, 0.5)])
        runs = [_valley_init_responsibilities(data, 2)[1] for _ in range(3)]
        assert all(np.array_equal(runs[0], r) for r in runs)

    def test_honors_sample_weights(self):
        """The gap GaussianMixture could not close."""

        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(
            np.concatenate([rng.normal(-4, 0.7, 2000), rng.normal(4, 0.7, 2000)])
        )
        w = np.concatenate([np.full(2000, 1.0), np.full(2000, 9.0)])
        w /= w.sum()

        _, flat = _valley_init_responsibilities(data, 2)
        _, tilted = _valley_init_responsibilities(data, 2, weights=w)
        assert np.allclose(flat, [0.5, 0.5], atol=0.02)
        assert np.allclose(tilted, [0.1, 0.9], atol=0.02)

    def test_rejects_shallow_small_tail_ripple(self):
        """A broad concentric tail must not manufacture a location split."""

        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(
            np.concatenate(
                [
                    rng.normal(0.0, 1.0, 1400),
                    rng.normal(0.0, 30.0, 600),
                ]
            )
        )
        resp, mix = _valley_init_responsibilities(data, 2)
        assert resp is None
        assert mix is None

    def test_auto_k_recovers_concentric_scale_components(self):
        """The BIC sweep must compare location- and scale-oriented seeds."""
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(
            np.concatenate(
                [
                    rng.normal(0.0, 1.0, 1400),
                    rng.normal(0.0, 30.0, 600),
                ]
            )
        )
        fit = Distribution().fit(
            data,
            n_components="auto",
            support=(-np.inf, np.inf),
            rng=1,
            auto_k_subsample=False,
        )

        assert fit.n_components == 2
        scales = np.sort(np.asarray([c.std for c in fit.components], dtype=float))
        assert scales[0] < 2.0
        assert scales[1] > 20.0


class TestSpectralConvergenceIsRecorded:
    """A degraded spectral CDF must be detectable after the fact.

    Construction accepts its best panel at ``max_depth`` rather than
    raising, so without a stored record an under-resolved CDF looks
    exactly like a converged one.
    """

    @pytest.mark.parametrize(
        "support,builder",
        [
            ((-np.inf, np.inf), lambda r: r.normal(size=2000)),
            ((0.0, np.inf), lambda r: r.gamma(2.0, 1.0, 2000)),
            ((0.0, 1.0), lambda r: r.beta(2.0, 5.0, 2000)),
        ],
    )
    def test_diagnostics_present_and_converged(self, support, builder):
        data = np.ascontiguousarray(builder(np.random.default_rng(0)))
        c = Distribution().fit(data, n_components=1, support=support, rng=0)
        d = c.components[0].data
        assert {
            "cdf_max_depth_used",
            "cdf_refinement_capped",
            "cdf_worst_panel_error",
            "cdf_error_estimate",
        } <= set(d.dtype.names)
        assert int(d["cdf_refinement_capped"]) == 0, "refinement hit its depth cap"
        assert float(d["cdf_worst_panel_error"]) < 1e-6


class TestBoundedSupportSelection:
    """Selection on bounded and half-line supports must be stable across seeds.

    Warm starts retain support in user coordinates.  The log-concave fixtures
    have an exact count; the lognormal is not log-concave, and approximating
    its tail with two or three components scores within a few BIC units, so
    either count is correct there.
    """

    CASES = (
        ("beta interior mode", lambda r: r.beta(2.0, 5.0, 3000), (0.0, 1.0), 1),
        (
            "beta two-sided",
            lambda r: np.concatenate([r.beta(2.0, 5.0, 1500), r.beta(5.0, 2.0, 1500)]),
            (0.0, 1.0),
            2,
        ),
        ("uniform", lambda r: r.uniform(0.0, 1.0, 3000), (0.0, 1.0), 1),
        ("lognormal", lambda r: r.lognormal(0.0, 0.6, 3000), (0.0, np.inf), (2, 3)),
    )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(
        "name,builder,support,expected", CASES, ids=[c[0] for c in CASES]
    )
    def test_selection_is_stable_across_seeds(
        self, name, builder, support, expected, seed
    ):
        data = np.ascontiguousarray(builder(np.random.default_rng(seed)))
        c = Distribution().fit(data, n_components="auto", support=support, rng=seed)
        assert c.n_components in np.atleast_1d(expected)

    @pytest.mark.parametrize(
        "name,builder,support,expected", CASES, ids=[c[0] for c in CASES]
    )
    def test_fitted_density_is_normalized_on_its_support(
        self, name, builder, support, expected
    ):
        """Over-selection was one symptom of the runaway support; this is another."""
        data = np.ascontiguousarray(builder(np.random.default_rng(0)))
        c = Distribution().fit(data, n_components="auto", support=support, rng=0)
        assert c.spectral_diagnostics["mass_defect"] < 1e-4
