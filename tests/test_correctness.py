"""Numerical correctness of the fitted densities.

These tests assert the defining properties of a probability density
rather than fixed numbers, so they stay valid as the optimiser changes.
"""
import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.integrate import quad as _quad

import gibbus._postfit.fitted_state as fitted_state
from gibbus import Distribution
from gibbus._api import component as _component
from gibbus._api import fitting as _fitting


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20240101)


@pytest.fixture(scope="module")
def normal_fit(rng):
    return Distribution().fit(rng.normal(size=1500), n_components=1,
                       support=(-np.inf, np.inf))


@pytest.fixture(scope="module")
def gamma_fit(rng):
    return Distribution().fit(rng.gamma(2.0, 1.0, size=1500), n_components=1,
                       support=(0, np.inf))


@pytest.fixture(scope="module")
def beta_fit(rng):
    return Distribution().fit(rng.beta(2, 3, size=1500), n_components=1,
                       support=(0, 1))


@pytest.fixture(scope="module")
def mixture_fit(rng):
    data = np.concatenate([rng.normal(-3, 0.7, 700),
                           rng.normal(3, 0.7, 700)])
    return Distribution().fit(data, n_components=2, support=(-np.inf, np.inf),
                       rng=0)


def _integrate(c, lo, hi):
    val, _ = quad(lambda t: c.pdf(t), lo, hi, limit=400)
    return val


class TestDensityAxioms:
    """Properties every fitted density must satisfy."""

    @pytest.mark.parametrize("fit,lo,hi", [
        ("normal_fit", -15, 15),
        ("gamma_fit", 0, 60),
        ("beta_fit", 0, 1),
        ("mixture_fit", -20, 20),
    ])
    def test_integrates_to_one(self, fit, lo, hi, request):
        c = request.getfixturevalue(fit)
        assert _integrate(c, lo, hi) == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("fit,lo,hi", [
        ("normal_fit", -15, 15),
        ("gamma_fit", 0, 60),
        ("beta_fit", 0, 1),
        ("mixture_fit", -20, 20),
    ])
    def test_pdf_non_negative(self, fit, lo, hi, request):
        c = request.getfixturevalue(fit)
        xs = np.linspace(lo, hi, 500)
        assert np.all(c.pdf(xs) >= 0.0)

    @pytest.mark.parametrize("fit,lo,hi", [
        ("normal_fit", -15, 15),
        ("gamma_fit", 0, 60),
        ("beta_fit", 0, 1),
        ("mixture_fit", -20, 20),
    ])
    def test_cdf_monotone_and_bounded(self, fit, lo, hi, request):
        c = request.getfixturevalue(fit)
        xs = np.linspace(lo, hi, 500)
        F = c.cdf(xs)
        assert np.all(np.diff(F) >= -1e-12)
        assert F.min() >= 0.0 and F.max() <= 1.0


class TestCdfPpfConsistency:
    @pytest.mark.parametrize("fit", ["normal_fit", "gamma_fit", "beta_fit",
                                     "mixture_fit"])
    def test_cdf_ppf_roundtrip(self, fit, request):
        c = request.getfixturevalue(fit)
        p = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])
        assert c.cdf(c.ppf(p)) == pytest.approx(p, abs=1e-8)

    @pytest.mark.parametrize("fit,lo,hi", [
        ("normal_fit", -15, 15),
        ("beta_fit", 0, 1),
        ("mixture_fit", -20, 20),
    ])
    def test_cdf_matches_integrated_pdf(self, fit, lo, hi, request):
        c = request.getfixturevalue(fit)
        for x in np.linspace(lo + 1e-6, hi - 1e-6, 7):
            assert c.cdf(x) == pytest.approx(_integrate(c, lo, x), abs=1e-5)

    def test_ppf_rejects_out_of_range(self, normal_fit):
        with pytest.raises(ValueError):
            normal_fit.ppf(1.5)
        with pytest.raises(ValueError):
            normal_fit.ppf(-0.1)

    @pytest.mark.parametrize("fit", ["normal_fit", "gamma_fit", "beta_fit",
                                     "mixture_fit"])
    def test_extreme_ppf_uses_spectral_cdf_inverse(self, fit, request):
        c = request.getfixturevalue(fit)
        # These probabilities lie outside the stored logit inverse-panel range
        # and therefore exercise the compiled spectral-CDF bisection path.
        p = np.array([1e-20, 1e-13, 1.0 - 1e-13])
        q = c.ppf(p)
        assert np.all(np.diff(q) >= 0.0)
        assert c.cdf(q) == pytest.approx(p, abs=5e-13)


class TestMoments:
    @pytest.mark.parametrize("fit,lo,hi", [
        ("normal_fit", -15, 15),
        ("gamma_fit", 0, 60),
        ("mixture_fit", -20, 20),
    ])
    def test_mean_and_var_match_quadrature(self, fit, lo, hi, request):
        c = request.getfixturevalue(fit)
        m, _ = quad(lambda t: t * c.pdf(t), lo, hi, limit=400)
        v, _ = quad(lambda t: (t - m) ** 2 * c.pdf(t), lo, hi, limit=400)
        assert c.mean == pytest.approx(m, rel=1e-6)
        assert c.var == pytest.approx(v, rel=1e-5)

    def test_std_is_sqrt_var(self, normal_fit):
        assert normal_fit.std == pytest.approx(np.sqrt(normal_fit.var))

    def test_central_moment_two_is_var(self, normal_fit):
        assert normal_fit.moment(2, central=True) == pytest.approx(
            normal_fit.var, rel=1e-9)

    def test_standardized_moment_three_is_skew(self, gamma_fit):
        assert gamma_fit.moment(3, standardized=True) == pytest.approx(
            gamma_fit.skew, rel=1e-6)

    def test_median_is_half_quantile(self, gamma_fit):
        assert gamma_fit.median == pytest.approx(gamma_fit.ppf(0.5), abs=1e-6)


class TestLogConcavity:
    def test_single_component_potential_is_convex(self, normal_fit):
        """A log-concave density has a convex potential: q'' >= 0."""
        xs = np.linspace(-4, 4, 400)
        assert np.all(normal_fit.neg_log(xs, n=2) > -1e-9)

    def test_mode_is_stationary(self, normal_fit):
        assert normal_fit.neg_log(normal_fit.mode, n=1) == pytest.approx(
            0.0, abs=1e-6)

    def test_mode_maximises_pdf(self, mixture_fit):
        xs = np.linspace(-10, 10, 2000)
        assert mixture_fit.pdf(mixture_fit.mode) >= mixture_fit.pdf(xs).max() - 1e-9


class TestRecovery:
    """The fit should recover the shape of a known log-concave law."""

    def test_normal_moments_recovered(self, rng):
        c = Distribution().fit(rng.normal(loc=3.0, scale=2.0, size=4000),
                        n_components=1, support=(-np.inf, np.inf))
        assert c.mean == pytest.approx(3.0, abs=0.15)
        assert c.std == pytest.approx(2.0, abs=0.15)
        assert abs(c.skew) < 0.2

    def test_mixture_modes_recovered(self, mixture_fit):
        modes = mixture_fit.modes
        assert len(modes) == 2
        assert modes[0] == pytest.approx(-3.0, abs=0.4)
        assert modes[1] == pytest.approx(3.0, abs=0.4)
        assert mixture_fit.weights == pytest.approx([0.5, 0.5], abs=0.05)

    def test_sampling_reproduces_distribution(self, normal_fit):
        drawn = normal_fit.sample(20000, rng=7)
        assert drawn.mean() == pytest.approx(normal_fit.mean, abs=0.05)
        assert drawn.std() == pytest.approx(normal_fit.std, rel=0.05)


class TestSpectralDegradationIsGraceful:
    """A quantile representation that will not certify must not kill the fit.

    ``SpectralPPF`` raises rather than degrading when it cannot certify a
    panel monotone — correct for the panels themselves, since an
    uncertified inverse could be non-monotone.  But the density is fine,
    and the compiled evaluator can already resolve any probability by
    monotone bisection of the packed spectral CDF.  So construction
    failure falls back to that path instead of propagating.

    Regression: ``gamma(2, 1)`` at ``n_components=2`` with ``rng=1`` used
    to run for over 400 seconds and then fail, on data the pre-spectral
    version fitted in 1.7 seconds.
    """

    @staticmethod
    def _gamma_pair():
        rng = np.random.default_rng(1)
        return np.ascontiguousarray(rng.gamma(2.0, 1.0, 4000))

    def test_fit_completes_and_quantiles_are_correct(self):
        c = Distribution().fit(self._gamma_pair(), n_components=2,
                        support=(0.0, np.inf), rng=1)
        assert c.n_components == 2
        p = np.linspace(0.002, 0.998, 60)
        q = c.ppf(p)
        assert np.all(np.diff(q) >= 0.0)
        assert c.cdf(q) == pytest.approx(p, abs=1e-9)

    def test_fallback_is_triggered_recorded_and_usable(self, monkeypatch):
        """A failed inverse build must record and use the CDF fallback.

        Natural data should not be required to remain pathological merely to
        cover this branch.  Instead, keep the real inverse builder but give it
        an intentionally impossible one-leaf, quadratic-only, zero-tolerance
        certification budget.  The resulting numerical construction failure
        must be swallowed at the production fallback boundary, packed into the
        component state, and routed through monotone CDF bisection.
        """

        real_ppf = fitted_state.SpectralPPF

        def constrained_ppf(cdf_rep):
            return real_ppf(
                cdf_rep,
                degree_options=(2,),
                fit_tol=0.0,
                logit_tol=0.0,
                prob_tol=0.0,
                coeff_tol=0.0,
                max_panels=1,
            )

        monkeypatch.setattr(fitted_state, "SpectralPPF", constrained_ppf)

        rng = np.random.default_rng(314159)
        data = np.ascontiguousarray(rng.normal(size=500))
        c = Distribution().fit(data, n_components=1,
                        support=(-np.inf, np.inf), rng=0)
        state = c.components[0].data

        assert int(state["ppf_fallback"]) == 1
        assert float(state["ppf_pmin"]) > float(state["ppf_pmax"])

        p = np.array([1e-4, 0.01, 0.2, 0.5, 0.8, 0.99, 1.0 - 1e-4])
        q = c.ppf(p)
        assert np.all(np.diff(q) > 0.0)
        assert c.cdf(q) == pytest.approx(p, abs=2e-10)

    def test_ordinary_fits_do_not_fall_back(self):
        rng = np.random.default_rng(0)
        for data, support in [(rng.normal(size=2000), (-np.inf, np.inf)),
                              (rng.gamma(2.0, 1.0, 2000), (0.0, np.inf)),
                              (rng.beta(2.0, 5.0, 2000), (0.0, 1.0))]:
            c = Distribution().fit(np.ascontiguousarray(data), n_components=1,
                            support=support, rng=0)
            assert int(c.components[0].data["ppf_fallback"]) == 0


class TestMixtureCdfAgreesWithItsComponents:
    """A mixture CDF must equal the weighted sum of its component CDFs.

    ``Distribution.cdf`` for ``K > 1`` does not evaluate the components: it
    builds a separate mixture-level spectral representation and queries
    that.  So the two can disagree, and nothing else in the suite would
    notice, because the obvious check -- ``cdf(ppf(p)) == p`` -- inverts
    the *same* representation and is satisfied to machine precision by a
    badly wrong CDF.

    That is not hypothetical.  One panel accepted at the budget limit
    contributed an analytic mass of 146 against a true total of 1,
    inflating the normaliser by that factor and crushing every CDF value
    on every panel; the round-trip test passed at 4e-16 throughout.  The
    weighted component sum is an independent reference and needs no
    reference implementation, so it is the check that catches this.
    """

    CASES = [
        ("bimodal", lambda r: np.concatenate([r.normal(-3, 1, 2000),
                                              r.normal(3, 1, 2000)]),
         (-np.inf, np.inf), 2),
        ("overlapping", lambda r: np.concatenate([r.normal(-1, 1, 2000),
                                                  r.normal(1, 1, 2000)]),
         (-np.inf, np.inf), 2),
        ("skewed half-line", lambda r: r.gamma(2.0, 1.0, 4000),
         (0.0, np.inf), 2),
        ("bounded", lambda r: np.concatenate([r.beta(2.0, 5.0, 2000),
                                              r.beta(5.0, 2.0, 2000)]),
         (0.0, 1.0), 2),
    ]

    @pytest.mark.parametrize("name,builder,support,k",
                             CASES, ids=[c[0] for c in CASES])
    def test_mixture_cdf_matches_weighted_components(self, name, builder,
                                                     support, k):
        data = np.ascontiguousarray(builder(np.random.default_rng(1)))
        c = Distribution().fit(data, n_components=k, support=support, rng=1)
        # K is explicit here, so no selection runs; a collapse would be a
        # defect in EM, not a reason to skip.
        assert c.n_components == k, f"{name}: explicit K={k} returned {c.n_components}"

        xs = np.quantile(data, np.linspace(0.02, 0.98, 25))
        mixture = np.asarray(c.cdf(xs), dtype=float)
        weighted = sum(
            w * np.asarray(comp.cdf(xs), dtype=float)
            for w, comp in zip(c.weights, c.components, strict=True)
        )

        # The mixture CDF is renormalised over the declared support, so
        # any probability a component placed outside it is divided back
        # out.  The weighted component sum is not renormalised.  Rescale
        # the reference by the mass actually inside the support so the
        # two use the same convention -- otherwise this test would fail
        # on the escaped-mass defect (see
        # TestMixtureMassStaysInsideSupport) rather than on the
        # normaliser defect it exists to catch.
        inside = float(weighted[-1]) + float(
            sum(w * (1.0 - float(np.asarray(comp.cdf(np.array([xs[-1]])),
                                            dtype=float)[0]))
                for w, comp in zip(c.weights, c.components, strict=True)))
        assert inside > 0.0
        assert mixture == pytest.approx(weighted / inside, abs=2e-3)


class TestUncertifiedMassStaysNegligible:
    """Ordinary log-concave shapes must not leave probability uncertified.

    The counterpart to ``test_ordinary_fits_do_not_fall_back``, for the
    CDF rather than the PPF.  It deliberately asserts on
    ``uncertified_mass`` and not on ``refinement_capped``: the depth flag
    fires on most of these shapes and always has, because refinement
    drives a single sliver of a panel to ``max_depth`` against an
    algebraic boundary singularity and force-accepts it.  That panel
    holds around 1e-10 of the probability and the CDF is accurate
    everywhere, so the depth flag is not the thing to regress on.
    """

    CASES = [
        ("normal", lambda r: r.normal(size=3000), (-np.inf, np.inf)),
        ("gamma near-exponential", lambda r: r.gamma(1.05, 1.0, 3000),
         (0.0, np.inf)),
        ("gamma", lambda r: r.gamma(2.0, 1.0, 3000), (0.0, np.inf)),
        ("weibull", lambda r: r.weibull(1.1, 3000), (0.0, np.inf)),
        ("beta interior mode", lambda r: r.beta(2.0, 5.0, 3000), (0.0, 1.0)),
        ("beta boundary mode", lambda r: r.beta(1.1, 8.0, 3000), (0.0, 1.0)),
    ]

    @pytest.mark.parametrize("name,builder,support",
                             CASES, ids=[c[0] for c in CASES])
    def test_uncertified_mass_is_negligible(self, name, builder, support):
        data = np.ascontiguousarray(builder(np.random.default_rng(0)))
        c = Distribution().fit(data, n_components=1, support=support, rng=0)
        diag = c.spectral_diagnostics

        assert diag["scope"] == "component"
        assert diag["uncertified_mass"] < 1e-7, (
            f"{name}: {diag['uncertified_mass']:.3e} of the probability sits "
            "under panels that never met tolerance")
        assert diag["mass_defect"] < 1e-4

    def test_diagnostics_available_for_mixtures_too(self):
        """A mixture is served by its own CDF, so it needs its own record.

        Component ``cdf_*`` fields describe representations a ``K > 1``
        model never queries, so reading them says nothing about what
        ``cdf`` and ``ppf`` will actually do.
        """
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(
            np.concatenate([rng.normal(-3, 1, 2000), rng.normal(3, 1, 2000)]))
        c = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)

        diag = c.spectral_diagnostics
        assert diag["scope"] == "mixture"
        assert diag["uncertified_mass"] < 1e-7
        assert diag["mass_defect"] < 1e-4
        assert diag["n_panels"] > 0


class TestNearExponentialFitsAreNotDoubled:
    """A gamma with shape near 1 must be fitted, not doubled.

    Regression.  Degree screening runs with the boundary flags off; the
    final refit turns them on and warm-starts from the screening
    solution.  A runaway ``mu`` that is harmless without a boundary term
    lands inside the kernel's divergence cap once one is added, where the
    returned gradient is not the derivative of the capped objective.
    L-BFGS-B's first line search then fails and it reports successful
    convergence having moved nowhere, so the fit is its own seed: for
    gamma shapes at or below about 1.05 that produced a density with
    almost exactly twice the correct mean, silently, on roughly half of
    all seeds.

    The property asserted is agreement with the empirical CDF rather than
    a captured NLL, so it survives retuning.  Deciles are used rather
    than the mean because the log-concave MLE legitimately truncates an
    exponential tail, which moves the mean by a few percent.
    """

    @pytest.mark.parametrize("shape", [1.0, 1.02, 1.05, 1.1, 1.2])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_fit_tracks_the_empirical_distribution(self, shape, seed):
        data = np.ascontiguousarray(
            np.random.default_rng(seed).gamma(shape, 1.0, 3000))
        c = Distribution().fit(data, n_components=1, support=(0.0, np.inf), rng=0)

        xs = np.quantile(data, [0.1, 0.3, 0.5, 0.7, 0.9])
        empirical = np.array([(data <= x).mean() for x in xs])
        fitted = np.asarray(c.cdf(xs), dtype=float)
        assert fitted == pytest.approx(empirical, abs=0.05)

    def test_fit_is_not_worse_than_its_own_seed(self):
        """The stall signature: converged, but no better than where it began.

        ``_restart_if_stalled`` exists so a fit cannot end up worse than
        an unseeded one would have been.  A near-exponential half-line
        fit is the case that exercised it.
        """
        data = np.ascontiguousarray(
            np.random.default_rng(0).gamma(1.0, 1.0, 3000))
        seeded = Distribution().fit(data, n_components=1, support=(0.0, np.inf), rng=0)
        nll = float(seeded.components[0].data["nll"])

        reference = float(np.mean(
            -np.log(np.clip(seeded.pdf(data), 1e-300, None))))
        assert np.isfinite(nll)
        assert reference < 1.2, (
            "fitted density assigns implausibly low likelihood to its own "
            f"training data (mean NLL {reference:.4f})")


class TestMixtureMassStaysInsideSupport:
    """A fitted density must not place probability outside its support.

    ``K == 1`` honours this exactly: a fit on ``(0, 1)`` evaluates to
    zero at -0.5, 1.5 and 5.0.  Mixture components are fitted on that same
    declared support; this regression guards against a degenerate component
    drifting outside it when its responsibility mass collapses.

    The escaping component is degenerate: it collects near-zero
    responsibilities, so its weighted M-step objective is meaningless
    (its recorded NLL is -345), and the resulting fit drifts.  Its weight
    is small, so the leaked mass is small -- but it is not bounded by
    anything, and a change that makes the optimiser more persistent can
    inflate it dramatically.  During this session an optimiser restart
    guard pushed the same component's weight from 7.8e-04 to 0.647,
    which took the in-support mass down to 0.35.  That is the reason to
    fix the leak rather than to tune around it.

    Recorded with numbers rather than as a conclusion, per the project's
    own habit: the fix belongs in the M-step (prune or constrain a
    component whose responsibilities collapse), not in the CDF, which
    renormalises the leak away and therefore hides it.
    """

    @staticmethod
    def _fit():
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(np.concatenate(
            [rng.beta(2.0, 5.0, 2000), rng.beta(5.0, 2.0, 2000)]))
        return data, Distribution().fit(data, n_components=2,
                                 support=(0.0, 1.0), rng=1)

    def test_single_component_honours_the_support(self):
        """The contract, where it currently holds."""
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(np.concatenate(
            [rng.beta(2.0, 5.0, 2000), rng.beta(5.0, 2.0, 2000)]))
        c = Distribution().fit(data, n_components=1, support=(0.0, 1.0), rng=1)
        outside = np.array([-0.5, 1.5, 5.0, 30.7])
        assert np.all(np.asarray(c.pdf(outside), dtype=float) == 0.0)

    def test_leaked_mass_has_not_grown(self):
        """Bound the known leak so a change that inflates it is caught.

        Not an assertion that the behaviour is correct -- it is not.
        The threshold is two orders of magnitude above the measured
        1.7e-03 so that ordinary retuning does not trip it, while the
        failure mode that actually matters (a degenerate component
        acquiring real weight) moves this by a factor of hundreds.
        """
        _, c = self._fit()
        outside = np.array([-0.5, 1.5, 5.0, 30.7])
        leaked = float(np.sum(
            [w for w, comp in zip(c.weights, c.components, strict=True)
             if float(np.asarray(comp.pdf(np.array([30.7])), dtype=float)[0]) > 0.0]
        ))
        assert leaked < 0.1, (
            f"a degenerate component now carries {leaked:.3f} of the weight "
            "outside the declared support")
        del outside

    def test_all_mass_lies_inside_the_support(self):
        """The invariant, now holding.

        Was a strict xfail recording a 1.7e-03 leak.  The cause was
        ``_fit_inputs`` inheriting a seed's support from
        ``base_support`` -- which is in fitting coordinates -- and
        handing it back to a pipeline that standardises whatever support
        it is given.  Every warm start therefore widened the support by
        ``1 / scale``, and after ten EM iterations it was effectively
        unbounded.
        """
        _, c = self._fit()
        inside = _quad(lambda t: float(np.asarray(c.pdf(np.array([t])),
                                                  dtype=float)[0]),
                       0.0, 1.0, limit=400)[0]
        assert inside == pytest.approx(1.0, abs=1e-6)


class TestNormalisationSurvivesSubnormalZ:
    """The normaliser must be exact when ``exp(-q)`` underflows.

    ``Z = int exp(-q)`` is a product of exponentials and lands in the
    subnormal range on ordinary fits -- a beta mixture component was
    measured at ``9.39e-323``, roughly four bits of mantissa.  Callers
    want ``log(Z)`` or a ratio ``I / Z``, and at that precision the log
    was 7.5% wrong: the component integrated to 1.075.

    A constant factor is invisible to every shape-based check.  The
    quantile round-trip, the mode, the empirical-CDF comparison and the
    panel diagnostics were all clean; only integrating the density
    caught it.  ``GIBBUS_NO_SUFFSTAT=1`` was what exposed it, by landing
    the fit on slightly more extreme parameters -- the second time that
    A/B has found a normalisation bug the compression was masking.

    Asserted on shapes whose fits drive ``q`` far enough negative to
    underflow, at one seed each: this is a numerical-conditioning
    property, not a distributional one, so more seeds would cost time
    without testing anything further.
    """

    CASES = [
        ("beta two-sided mixture",
         lambda r: np.concatenate([r.beta(2.0, 5.0, 1200),
                                   r.beta(5.0, 2.0, 1200)]), (0.0, 1.0), 2),
        ("beta boundary mode", lambda r: r.beta(1.1, 8.0, 2000), (0.0, 1.0), 1),
        ("gamma", lambda r: r.gamma(2.0, 1.0, 2000), (0.0, np.inf), 1),
    ]

    @pytest.mark.parametrize("name,builder,support,k",
                             CASES, ids=[c[0] for c in CASES])
    def test_density_integrates_to_one(self, name, builder, support, k):
        data = np.ascontiguousarray(builder(np.random.default_rng(1)))
        c = Distribution().fit(data, n_components=k, support=support, rng=1)
        lo = support[0]
        hi = support[1] if np.isfinite(support[1]) else float(data.max()) * 6.0
        total = quad(
            lambda t: float(np.asarray(c.pdf(np.array([t])), dtype=float)[0]),
            lo, hi, limit=300)[0]
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_log_normaliser_is_finite_where_the_linear_one_underflows(self):
        """``log_Z`` must stay exact where ``Z`` would have lost its bits."""
        data = np.ascontiguousarray(np.concatenate([
            np.random.default_rng(1).beta(2.0, 5.0, 1200),
            np.random.default_rng(2).beta(5.0, 2.0, 1200)]))
        c = Distribution().fit(data, n_components=2, support=(0.0, 1.0), rng=1)
        total = quad(
            lambda t: float(np.asarray(c.pdf(np.array([t])), dtype=float)[0]),
            0.0, 1.0, limit=300)[0]
        assert total == pytest.approx(1.0, abs=1e-4)


class TestShapeBugsAreNotSwallowed:
    """Array-shape defects are not classified as numerical fallbacks.

    Generic ``ValueError`` is intentionally absent from ``NUMERIC_FAILURES``.
    This means NumPy broadcasting/reshape failures and package contract errors
    propagate directly instead of being classified from version-dependent
    exception-message text. Numerical invalid states use explicit arithmetic,
    linear-algebra, or runtime exception classes instead.
    """

    def test_a_shape_bug_in_the_em_refit_reaches_the_caller(self, monkeypatch):
        """The exact failure mode that hid a real bug during this review."""

        def broken(*args, **kwargs):
            return np.zeros(3) + np.zeros(5)

        monkeypatch.setattr(_fitting, "_fit_natural_mixture", broken)
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(np.concatenate(
            [rng.normal(-3, 1, 600), rng.normal(3, 1, 600)]))

        with pytest.raises(ValueError, match="could not be broadcast"):
            Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)

    def test_a_shape_bug_in_the_auto_degree_sweep_reaches_the_caller(
            self, monkeypatch):
        """The sibling path, which the EM test above cannot see.

        The auto-degree sweep guards its screening fits separately from
        the EM refit. Generic ``ValueError`` must bypass both numerical
        guards, so a shape bug in either path reaches the caller directly.
        """

        def broken(*args, **kwargs):
            return np.zeros(3) + np.zeros(5)

        monkeypatch.setattr(_component, "_fit_natural_conic_points_auto", broken)
        rng = np.random.default_rng(0)
        data = np.ascontiguousarray(rng.normal(size=300))

        with pytest.raises(ValueError, match="operands could not be broadcast"):
            Distribution().fit(data, n_components=1, poly_degree="auto",
                        support=(-np.inf, np.inf))


class TestMinorityComponentSpectralAccuracy:
    """A small, narrow, well-separated component must retain spectral accuracy.

    Each mixture component is fitted in its own responsibility-weighted coordinate
    system.  This keeps the component potential well conditioned near its own mode
    and allows the mixture CDF to agree with the weighted component CDFs to numerical
    precision.  Round-trip ``cdf(ppf(p))`` checks alone are insufficient because a
    CDF and inverse built from the same approximation can be self-consistent.
    """

    @staticmethod
    def _minority_mixture():
        rng = np.random.default_rng(7)
        data = np.concatenate([rng.normal(0.0, 1.0, 2900),
                               rng.normal(9.0, 0.4, 100)])
        return np.ascontiguousarray(data)

    def test_components_are_centred_on_themselves(self):
        """Each component normalizes around its own responsibility-weighted data."""
        c = Distribution().fit(self._minority_mixture(), n_components=2,
                        support=(-np.inf, np.inf), rng=0)
        for comp in c.components:
            centre = float(comp.data["fit_center"])
            scale = float(comp.data["fit_scale"])
            direction = float(comp.data["fit_direction"])
            z_at_peak = direction * (float(comp.mean) - centre) / scale
            assert abs(z_at_peak) < 1.0, (
                f"component at {float(comp.mean):.3f} normalised to "
                f"center={centre:.3f}, giving z={z_at_peak:.2f}")

    def test_potential_evaluates_without_catastrophic_cancellation(self):
        c = Distribution().fit(self._minority_mixture(), n_components=2,
                        support=(-np.inf, np.inf), rng=0)
        for comp in c.components:
            d = comp.data
            z = (float(d["fit_direction"])
                 * (float(comp.mean) - float(d["fit_center"]))
                 / float(d["fit_scale"]))
            q = np.asarray(d["q_poly"], dtype=float)
            terms = q * z ** np.arange(q.size)
            total = abs(float(terms.sum()))
            ratio = float(np.abs(terms).max()) / max(total, 1e-300)
            assert ratio < 1e3, f"cancellation ratio {ratio:.2e} at z={z:.2f}"

    def test_mixture_cdf_equals_weighted_component_cdfs(self):
        c = Distribution().fit(self._minority_mixture(), n_components=2,
                        support=(-np.inf, np.inf), rng=0)
        xs = np.concatenate([np.linspace(-4.0, 4.0, 5), np.linspace(7.0, 11.0, 5)])
        w = np.asarray(c.weights, dtype=float)
        mixture = np.asarray(c.cdf(xs), dtype=float)
        weighted = sum(w[j] * np.asarray(c.components[j].cdf(xs), dtype=float)
                       for j in range(c.n_components))
        assert np.max(np.abs(mixture - weighted)) < 1e-12

    def test_the_pdf_identity_still_holds_exactly(self):
        c = Distribution().fit(self._minority_mixture(), n_components=2,
                        support=(-np.inf, np.inf), rng=0)
        xs = np.concatenate([np.linspace(-4.0, 4.0, 5), np.linspace(7.0, 11.0, 5)])
        w = np.asarray(c.weights, dtype=float)
        mixture = np.asarray(c.pdf(xs), dtype=float)
        weighted = sum(w[j] * np.asarray(c.components[j].pdf(xs), dtype=float)
                       for j in range(c.n_components))
        assert np.max(np.abs(mixture - weighted)) == 0.0

    def test_the_cdf_is_now_certified(self):
        c = Distribution().fit(self._minority_mixture(), n_components=2,
                        support=(-np.inf, np.inf), rng=0)
        d = c.spectral_diagnostics
        assert d["uncertified_mass"] < 1e-9, d
        assert d["refinement_capped"] is False, d


class TestDegenerateComponentsAreNotSelected:
    """Auto-*K* must not add a component supported by a few points.

    BIC charges for parameters, not for support, so a component fitted to
    a handful of tail points is cheap and can win on likelihood.  This
    only became reachable once per-component recentring made such a
    component well-conditioned enough to fit well: before that, poor
    conditioning masked it.  ``AUTO_LC_MIN_COMPONENT_N`` rejects any *K*
    whose smallest component holds too few effective samples.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_unimodal_beta_stays_at_one_component(self, seed):
        data = np.ascontiguousarray(np.random.default_rng(seed).beta(2.0, 5.0, 3000))
        c = Distribution().fit(data, support=(0.0, 1.0), rng=seed)
        assert c.n_components == 1

    def test_a_genuine_minority_component_is_still_found(self):
        """The guard must not suppress real structure: 3% of the mass,
        well separated, is ~90 effective samples and must survive."""
        rng = np.random.default_rng(7)
        data = np.ascontiguousarray(np.concatenate([rng.normal(0.0, 1.0, 2900),
                                                    rng.normal(9.0, 0.4, 100)]))
        c = Distribution().fit(data, n_components="auto", support=(-np.inf, np.inf), rng=0)
        assert c.n_components >= 2
        assert float(np.min(c.weights)) * data.size >= 20.0



class TestExtremeTailQuantiles:
    """Extreme quantiles use a log-probability tail solve when needed.

    The spectral CDF stores ordinary probabilities and therefore cannot resolve
    arbitrarily small tail masses.  Extreme queries are solved from ``log F(x)``
    using the fitted potential, avoiding loss of significance in absolute
    probability space.
    """

    @staticmethod
    def _normal_fit():
        data = np.ascontiguousarray(np.random.default_rng(0).normal(0.0, 1.0, 30000))
        return Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))

    @pytest.mark.parametrize("p", [1e-15, 1e-20, 1e-50, 1e-100, 1e-300])
    def test_extreme_lower_quantiles_match_the_true_normal(self, p):
        """Accuracy must not depend on how extreme the query is.

        The tolerance is the *statistical* error of the fit, which is
        ~3e-3 at this sample size and is the same at ``p = 1e-4`` where
        the CDF panels are exact.  The inversion adds essentially
        nothing on top of it.
        """

        c = self._normal_fit()
        got = float(c.ppf(p))
        expected = float(stats.norm(0.0, 1.0).ppf(p))
        assert abs(got - expected) / abs(expected) < 1e-2

    def test_extreme_quantiles_are_monotone_and_finite(self):
        c = self._normal_fit()
        ps = np.array([1e-13, 1e-20, 1e-60, 1e-150, 1e-300])
        got = np.asarray(c.ppf(ps), dtype=float)
        assert np.all(np.isfinite(got))
        assert np.all(np.diff(got) < 0.0)

    def test_bulk_quantiles_are_untouched(self):
        """The asymptotic branch must not perturb ordinary queries."""
        c = self._normal_fit()
        q = np.linspace(1e-6, 1.0 - 1e-6, 500)
        assert np.max(np.abs(np.asarray(c.cdf(c.ppf(q))) - q)) < 1e-9

    def test_shape_and_scalar_handling_survive(self):
        c = self._normal_fit()
        assert isinstance(c.ppf(1e-30), float)
        assert c.ppf(np.array([[1e-20, 0.5], [0.9, 1e-40]])).shape == (2, 2)
        assert np.array_equal(c.ppf(np.array([0.0, 1.0])),
                              np.array([-np.inf, np.inf]))

    def test_mixtures_reach_the_same_depth(self):
        rng = np.random.default_rng(1)
        data = np.ascontiguousarray(np.concatenate([rng.normal(-3.0, 1.0, 2000),
                                                    rng.normal(3.0, 1.0, 2000)]))
        c = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
        got = np.asarray(c.ppf(np.array([1e-13, 1e-20, 1e-60, 1e-200])), dtype=float)
        assert np.all(np.isfinite(got))
        assert np.all(np.diff(got) < 0.0)

    @pytest.mark.parametrize("support,builder", [
        ((0.0, np.inf), lambda r: r.exponential(1.0, 20000)),
        ((0.0, 1.0), lambda r: r.beta(2.0, 5.0, 20000)),
    ])
    def test_finite_lower_endpoint_reaches_extreme_float64_tails(self, support, builder):
        """A zero endpoint must not impose the internal-coordinate ulp floor.

        Boundary distances are evaluated directly in public coordinates, so
        points such as ``1e-100`` remain distinguishable from a lower endpoint
        at zero even though their internal affine coordinates round to the same
        float.  Extreme quantiles should therefore keep moving towards the edge
        instead of saturating around ``1e-16`` of the fitted scale.
        """
        c = Distribution().fit(np.ascontiguousarray(builder(np.random.default_rng(0))),
                        n_components=1, support=support)
        ps = np.array([1e-13, 1e-20, 1e-60, 1e-150, 1e-300])
        got = np.asarray(c.ppf(ps), dtype=float)
        assert np.all(np.isfinite(got))
        assert np.all(got > support[0])
        assert np.all(got < support[1])
        assert np.all(np.diff(got) < 0.0)
        assert got[-1] < 1e-120
        assert np.all(np.isfinite(np.asarray(c.neg_log(got), dtype=float)))
