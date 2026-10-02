"""Public-boundary validation: bad input must fail with a clear ValueError.

Each test asserts on the message as well as the type, because the point
of validating at the boundary is that the user is told what to fix --
a ValueError leaking out of NumPy or SciPy would pass a
type-only assertion while being useless to the caller.
"""
import numpy as np
import pytest

from gibbus import Distribution
from gibbus._fit.inputs import _canon_univariate_samples


@pytest.fixture
def rng():
    return np.random.default_rng(99)


class TestSampleValidation:
    def test_single_sample_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            Distribution().fit(np.array([1.0]), support=(-np.inf, np.inf))

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            Distribution().fit(np.array([]), support=(-np.inf, np.inf))

    def test_zero_spread_rejected(self):
        with pytest.raises(ValueError, match="identical"):
            Distribution().fit(np.full(100, 3.0), support=(-np.inf, np.inf))

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Distribution().fit(np.array([1.0, np.nan, 2.0]),
                        support=(-np.inf, np.inf))

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Distribution().fit(np.array([1.0, np.inf, 2.0]),
                        support=(-np.inf, np.inf))

    def test_bad_shape_rejected(self, rng):
        with pytest.raises(ValueError, match=r"\(R,\)"):
            Distribution().fit(rng.normal(size=(10, 3)), support=(-np.inf, np.inf))

    def test_samples_outside_support_rejected(self):
        with pytest.raises(ValueError, match="outside the provided support"):
            Distribution().fit(np.array([-1.0, 2.0, 3.0]), support=(0, np.inf))

    def test_single_sample_error_names_the_problem(self):
        """Validation errors must say what is wrong, not where it failed."""
        with pytest.raises(ValueError) as exc:
            Distribution().fit(np.array([1.0]), support=(-np.inf, np.inf))
        assert "Traceback" not in str(exc.value)
        assert "sample" in str(exc.value).lower()

    def test_large_offset_tiny_spread_is_not_misclassified_as_degenerate(self):
        """A small physical spread must survive the user-coordinate pullback."""
        rng = np.random.default_rng(0)
        data = 1e6 + 1e-9 * rng.standard_normal(800)

        fitted = Distribution().fit(
            data, n_components=1, poly_degree="auto",
            support=(0.0, np.inf),
        )

        assert fitted.is_fitted
        assert np.isfinite(fitted.std)
        assert 2e-10 < fitted.std < 5e-9
        assert abs(fitted.mean - 1e6) < 5e-9
        assert fitted.ppf(0.0) == 0.0
        assert np.isposinf(fitted.ppf(1.0))
        assert 0.0 < fitted.cdf(fitted.mean) < 1.0

    def test_large_offset_tiny_spread_degree_four_is_stable(self):
        """A direct degree-four fit must remain stable at extreme offset/scale."""
        rng = np.random.default_rng(0)
        data = 1e6 + 1e-9 * rng.standard_normal(800)

        fitted = Distribution().fit(
            data, n_components=1, poly_degree=4,
            support=(0.0, np.inf),
        )

        assert fitted.is_fitted
        assert int(fitted.components[0].data["requested_poly_degree"]) == 4
        assert np.isfinite(fitted.std) and 2e-10 < fitted.std < 5e-9

    def test_single_extreme_outlier_does_not_make_auto_degree_fail(self):
        """Auto-degree screening should retain a valid low-degree candidate."""
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.standard_normal(800), [1e9]])

        fitted = Distribution().fit(
            data, n_components=1, poly_degree="auto",
            support=(-np.inf, np.inf),
        )

        assert fitted.is_fitted
        assert np.isfinite(fitted.mean)
        assert np.isfinite(fitted.std) and fitted.std > 0.0


class TestPolynomialDegreeValidation:
    @pytest.mark.parametrize("degree", [3, 5, 7])
    def test_odd_degree_rejected_on_full_infinite_support(self, degree, rng):
        data = rng.normal(size=300)
        with pytest.raises(ValueError, match="odd.*inadmissible.*even degree"):
            Distribution().fit(
                data, n_components=1, poly_degree=degree,
                support=(-np.inf, np.inf),
            )

    def test_auto_degree_on_full_infinite_support_selects_even_degree(self, rng):
        data = rng.normal(size=400)
        fitted = Distribution().fit(
            data, n_components=1, poly_degree="auto",
            support=(-np.inf, np.inf),
        )
        assert int(fitted.components[0].data["requested_poly_degree"]) % 2 == 0


class TestSupportValidation:
    @pytest.mark.parametrize("support", [
        (np.nan, 1.0),
        (0.0, np.nan),
    ])
    def test_nan_support_endpoint_rejected_directly(self, support, rng):
        with pytest.raises(ValueError, match="support endpoints must not be NaN"):
            Distribution().fit(rng.normal(size=200), support=support)

    @pytest.mark.parametrize(
        "data",
        [
            np.array([1.0, 2.0, 3.0]),
            np.array([-3.0, -2.0, -1.0]),
            np.array([-1.0, 1.0, 2.0]),
        ],
        ids=["positive-only", "negative-only", "mixed-sign"],
    )
    def test_none_support_is_always_full_real_line(self, data):
        """Observed signs must not choose the structural support class."""
        fitted = Distribution().fit(
            data, n_components=1, poly_degree=2, support=None,
            progressive=False,
        )
        assert tuple(float(v) for v in fitted.support) == (-np.inf, np.inf)
        assert np.isneginf(fitted.ppf(0.0))
        assert np.isposinf(fitted.ppf(1.0))

    def test_none_support_is_stable_when_one_observation_crosses_zero(self):
        """A sign-changing extreme observation must not change model structure."""
        rng = np.random.default_rng(16)
        positive = rng.normal(5.0, 1.0, 800)
        assert np.all(positive > 0.0)
        crossed = positive.copy()
        crossed[0] = -0.01

        a = Distribution().fit(
            positive, n_components=1, poly_degree=2, support=None,
            progressive=False,
        )
        b = Distribution().fit(
            crossed, n_components=1, poly_degree=2, support=None,
            progressive=False,
        )

        assert tuple(float(v) for v in a.support) == (-np.inf, np.inf)
        assert tuple(float(v) for v in b.support) == (-np.inf, np.inf)
        assert not np.any(a.components[0].data["boundary_allowed"])
        assert not np.any(b.components[0].data["boundary_allowed"])
        assert np.all(a.components[0].data["boundary_amplitudes"] == 0.0)
        assert np.all(b.components[0].data["boundary_amplitudes"] == 0.0)

    def test_explicit_zero_boundary_remains_available(self, rng):
        data = rng.gamma(2.0, 1.0, size=400)
        fitted = Distribution().fit(
            data, n_components=1, poly_degree=2,
            support=(0.0, np.inf), progressive=False,
        )
        assert tuple(float(v) for v in fitted.support) == (0.0, np.inf)
        assert fitted.ppf(0.0) == 0.0
        state = fitted.components[0].data
        assert bool(state["boundary_allowed"][0])
        assert float(state["boundary_amplitudes"][0]) >= 0.0

    def test_none_support_for_mixture_is_full_real_line(self, rng):
        data = np.concatenate([rng.normal(2.0, 0.4, 250),
                               rng.normal(5.0, 0.5, 250)])
        assert np.all(data > 0.0)
        fitted = Distribution().fit(
            data, n_components=2, poly_degree=2, support=None,
            progressive=False, rng=0,
        )
        assert tuple(float(v) for v in fitted.support) == (-np.inf, np.inf)
        assert all(tuple(float(v) for v in component.support)
                   == (-np.inf, np.inf) for component in fitted.components)

    def test_reversed_support_rejected(self, rng):
        with pytest.raises(ValueError, match=r"support\[0\]"):
            Distribution().fit(rng.normal(size=100), support=(5, -5))

    @pytest.mark.parametrize("support", [(-np.inf, np.inf), (0, np.inf),
                                         (0, 1)])
    def test_valid_supports_accepted(self, support, rng):
        lo, hi = support
        data = rng.uniform(max(lo, -3), min(hi, 3), size=300)
        c = Distribution().fit(data, n_components=1, support=support)
        assert c.is_fitted
        assert c.support[0] == lo and c.support[1] == hi


class TestWeightValidation:
    def test_negative_weights_rejected(self, rng):
        with pytest.raises(ValueError, match="non-negative"):
            Distribution().fit(rng.normal(size=50), support=(-np.inf, np.inf),
                        sample_weights=-np.ones(50))

    def test_wrong_length_rejected(self, rng):
        with pytest.raises(ValueError, match="length R"):
            Distribution().fit(rng.normal(size=50), support=(-np.inf, np.inf),
                        sample_weights=np.ones(49))

    def test_all_zero_rejected(self, rng):
        with pytest.raises(ValueError, match="not all zero"):
            Distribution().fit(rng.normal(size=50), support=(-np.inf, np.inf),
                        sample_weights=np.zeros(50))


class TestComponentValidation:
    def test_zero_components_rejected(self, rng):
        with pytest.raises(ValueError, match="n_components"):
            Distribution().fit(rng.normal(size=100), n_components=0,
                        support=(-np.inf, np.inf))

    def test_too_few_samples_for_k(self, rng):
        with pytest.raises(ValueError, match="requires at least"):
            Distribution().fit(rng.normal(size=10), n_components=8,
                        support=(-np.inf, np.inf), rng=0)

    def test_component_options_length_mismatch(self, rng):
        with pytest.raises(ValueError, match="length"):
            Distribution().fit(rng.normal(size=200), n_components=2,
                        support=(-np.inf, np.inf),
                        component_options=[{"poly_degree": 4}])

    def test_component_options_forbidden_key(self, rng):
        with pytest.raises(ValueError, match="must not contain"):
            Distribution().fit(rng.normal(size=200), n_components=2,
                        support=(-np.inf, np.inf),
                        component_options=[{"support": (0, 1)}, {}])

    def test_component_options_forbidden_under_auto(self, rng):
        with pytest.raises(ValueError, match="must be None"):
            Distribution().fit(rng.normal(size=200), n_components="auto",
                        support=(-np.inf, np.inf),
                        component_options=[{"poly_degree": 4}])


class TestUnfittedAccess:
    @pytest.mark.parametrize("call", [
        lambda c: c.pdf(0.0),
        lambda c: c.cdf(0.0),
        lambda c: c.ppf(0.5),
        lambda c: c.sample(5),
        lambda c: c.mean,
        lambda c: c.var,
        lambda c: c.mode,
        lambda c: c.modes,
        lambda c: c.weights,
        lambda c: c.n_components,
    ])
    def test_raises_runtime_error(self, call):
        with pytest.raises(RuntimeError, match="not fitted"):
            call(Distribution())

    def test_is_fitted_false(self):
        assert Distribution().is_fitted is False


class TestNonLogConcaveData:
    @pytest.mark.parametrize("seed", range(6))
    def test_heavy_tails_fail_with_actionable_message(self, seed):
        """Cauchy data has no log-concave MLE.

        Whether the optimizer gives up depends on the draw, so this
        asserts only that *when* it fails the message is actionable --
        never that it fails on every seed.
        """
        data = np.random.default_rng(seed).standard_cauchy(2000)
        try:
            Distribution().fit(data, n_components=1, support=(-np.inf, np.inf))
        except RuntimeError as exc:
            assert "log-concave" in str(exc)
        except ValueError:
            pytest.skip("rejected at validation rather than fitting")


def test_interval_samples_allow_infinite_censoring_endpoints():

    rows = np.array([[-np.inf, -1.0], [-0.2, 0.4], [1.0, np.inf]])
    got, n = _canon_univariate_samples(rows)
    assert n == 3
    np.testing.assert_array_equal(got, rows)


def test_point_samples_still_reject_infinity():

    with pytest.raises(ValueError, match="point samples"):
        _canon_univariate_samples(np.array([0.0, np.inf]))


def test_zero_width_interval_at_infinity_is_invalid():

    with pytest.raises(ValueError, match="positive width"):
        _canon_univariate_samples(np.array([[np.inf, np.inf], [0.0, 1.0]]))


class TestBoundaryEndpointValidation:
    @pytest.mark.parametrize("side,data", [
        ("lower", np.array([0.0, 0.2, 0.5, 0.8])),
        ("upper", np.array([0.2, 0.5, 0.8, 1.0])),
    ])
    def test_endpoint_point_rejected_when_log_basis_requested(self, side, data):
        with pytest.raises(ValueError, match=f"{side} support endpoint.*log_boundary_{side}=True"):
            Distribution().fit(
                data, n_components=1, poly_degree=2, support=(0.0, 1.0),
                progressive=False, **{f"log_boundary_{side}": True},
            )

    @pytest.mark.parametrize("side,data", [
        ("lower", np.array([0.0, 0.2, 0.5, 0.8])),
        ("upper", np.array([0.2, 0.5, 0.8, 1.0])),
    ])
    def test_endpoint_point_rules_out_an_automatic_term(self, side, data):
        """A point at the endpoint has zero density under that side's term."""
        fitted = Distribution().fit(
            data, n_components=1, poly_degree=2, support=(0.0, 1.0), progressive=False,
        )
        allowed = fitted.components[0].data["boundary_allowed"]
        assert not bool(allowed[0 if side == "lower" else 1])

    def test_disabling_corresponding_log_basis_allows_endpoint_point(self):
        data = np.array([0.0, 0.2, 0.5, 0.8])
        fitted = Distribution().fit(
            data, n_components=1, poly_degree=2, support=(0.0, 1.0),
            log_boundary_lower=False, progressive=False,
        )
        assert fitted.is_fitted
        assert not bool(fitted.components[0].data["boundary_allowed"][0])

    def test_positive_width_intervals_may_touch_both_endpoints(self):
        rows = np.array([[0.0, 0.1], [0.2, 0.45], [0.6, 1.0]])
        fitted = Distribution().fit(
            rows, n_components=1, poly_degree=2, support=(0.0, 1.0),
            progressive=False,
        )
        assert fitted.is_fitted

    def test_zero_width_interval_at_endpoint_is_rejected(self):
        rows = np.array([[0.0, 0.0], [0.2, 0.45], [0.6, 0.9]])
        with pytest.raises(ValueError, match="lower support endpoint.*log_boundary_lower=True"):
            Distribution().fit(
                rows, n_components=1, poly_degree=2, support=(0.0, 1.0),
                log_boundary_lower=True, progressive=False,
            )

    def test_zero_weight_endpoint_point_is_ignored(self):
        data = np.array([0.0, 0.2, 0.5, 0.8])
        weights = np.array([0.0, 1.0, 1.0, 1.0])
        fitted = Distribution().fit(
            data, n_components=1, poly_degree=2, support=(0.0, 1.0),
            sample_weights=weights, progressive=False,
        )
        assert fitted.is_fitted

    def test_log_basis_requires_matching_finite_endpoint(self):
        data = np.array([-1.0, -0.2, 0.4, 1.1])
        with pytest.raises(ValueError, match="finite lower support endpoint"):
            Distribution().fit(
                data, n_components=1, poly_degree=2,
                support=(-np.inf, np.inf), log_boundary_lower=True,
                progressive=False,
            )


class TestFitControlValidation:
    """Options that used to change the fitted model silently when wrong."""

    def test_non_integral_n_components_rejected(self, rng):
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            Distribution().fit(rng.normal(size=200), n_components=2.5)

    def test_non_positive_k_max_rejected(self, rng):
        with pytest.raises(ValueError, match="k_max must be >= 1"):
            Distribution().fit(rng.normal(size=200), k_max=0)

    def test_negative_em_max_iter_rejected(self, rng):
        with pytest.raises(ValueError, match="em_max_iter must be >= 0"):
            Distribution().fit(
                rng.normal(size=200), n_components=2, em_max_iter=-1
            )

    def test_non_positive_em_tol_rejected(self, rng):
        with pytest.raises(ValueError, match="em_tol must be finite"):
            Distribution().fit(
                rng.normal(size=200), n_components=2, em_tol=-1.0
            )

    def test_complex_samples_rejected(self, rng):
        with pytest.raises(ValueError, match="real-valued"):
            Distribution().fit(rng.normal(size=200) + 1j)

    def test_valid_controls_still_fit(self, rng):
        fitted = Distribution().fit(
            rng.normal(size=300), n_components=2, k_max=3,
            em_max_iter=5, em_tol=1e-6, support=(-np.inf, np.inf),
        )
        assert fitted.is_fitted
