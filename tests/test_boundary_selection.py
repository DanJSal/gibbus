"""Data-driven boundary-log terms and their identification diagnostics."""

import numpy as np
import pytest
from scipy.stats import chi2

from gibbus import Distribution
from gibbus._defaults import BOUNDARY_ALPHA
from gibbus._fit.boundary import (
    AUTO,
    _boundary_p_value,
    _select_boundary_terms,
    _weakly_identified_sides,
)


def _record(model):
    return model.fit_diagnostics["shared_boundary"]


def test_p_value_is_the_one_sided_boundary_mixture():
    assert _boundary_p_value(1.0, 1.0, 100.0) == 1.0
    assert _boundary_p_value(1.0, 1.01, 100.0) == 1.0  # the nested fit is better
    lr = 2.0 * 100.0 * 0.02
    assert _boundary_p_value(1.02, 1.0, 100.0) == pytest.approx(0.5 * chi2.sf(lr, 1))


def test_selection_drops_the_least_significant_side_first_and_stops():
    # Toy models: dicts of amplitudes and NLLs keyed by flags.
    nlls = {
        (True, True): 1.000,
        (False, True): 1.001,
        (True, False): 1.100,
        (False, False): 1.200,
    }

    def fit(lo, up):
        return {"flags": (lo, up), "nll": nlls[(lo, up)]}

    model, flags, p_values = _select_boundary_terms(
        fit,
        lambda m: m["nll"],
        lambda m, side: 1.0 if m["flags"][0 if side == "lower" else 1] else 0.0,
        AUTO,
        AUTO,
        400.0,
        alpha=BOUNDARY_ALPHA,
    )
    assert flags == (False, True)
    assert model["flags"] == (False, True)
    assert p_values[0] > 0.05 and p_values[1] < 1e-6


def test_explicit_flags_are_never_tested():
    calls = []

    def fit(lo, up):
        calls.append((lo, up))
        return {"nll": 1.0}

    _, flags, p_values = _select_boundary_terms(
        fit,
        lambda m: m["nll"],
        lambda m, side: 1.0,
        True,
        False,
        100.0,
        alpha=BOUNDARY_ALPHA,
    )
    assert flags == (True, False) and calls == [(True, False)]
    assert np.all(np.isnan(p_values))


def test_a_real_boundary_term_is_kept():
    """Gamma(3) vanishes like x^2 at zero: the lower term (amplitude 2) is resolved."""
    x = np.random.default_rng(3).gamma(3.0, 1.0, 400)
    model = Distribution().fit(x, n_components=1, support=(0.0, np.inf))
    data = model.components[0].data
    assert bool(data["boundary_allowed"][0])
    assert data["boundary_amplitudes"][0] == pytest.approx(2.0, abs=1.0)
    record = _record(model)
    assert record["p_values"][0] < 1e-3
    assert 0.0 < record["standard_errors"][0] < 1.0
    assert record["weakly_identified"] == ()


def test_an_absent_boundary_term_is_dropped():
    """The exponential density is positive at zero: no term."""
    x = np.random.default_rng(3).exponential(1.0, 400)
    model = Distribution().fit(x, n_components=1, support=(0.0, np.inf))
    assert not _record(model)["allowed"][0]
    assert model.fit_diagnostics["converged"]


def test_an_unresolvable_term_is_dropped_and_an_explicit_one_is_flagged():
    """Beta(5, 1.5), n = 30, degree 8: the amplitudes trade against the polynomial."""
    x = np.random.default_rng(270901).beta(5.0, 1.5, 30)
    automatic = Distribution().fit(x, n_components=1, poly_degree=8, support=(0.0, 1.0))
    assert not np.any(_record(automatic)["allowed"])
    assert automatic.fit_diagnostics["converged"]
    assert all(p >= 0.05 for p in _record(automatic)["p_values"])

    explicit = Distribution().fit(
        x,
        n_components=1,
        poly_degree=8,
        support=(0.0, 1.0),
        log_boundary_lower=True,
        log_boundary_upper=True,
    )
    assert set(_record(explicit)["weakly_identified"]) == {
        "lower",
        "upper",
    }


def test_standard_error_shrinks_like_one_over_root_n():
    rng = np.random.default_rng(11)
    errors = []
    for n in (500, 2000):
        x = rng.gamma(3.0, 1.0, n)
        model = Distribution().fit(
            x,
            n_components=1,
            poly_degree=2,
            support=(0.0, np.inf),
            log_boundary_lower=True,
        )
        errors.append(_record(model)["standard_errors"][0])
    assert errors[0] / errors[1] == pytest.approx(2.0, rel=0.35)


def test_weakly_identified_sides_compares_with_two_standard_errors():
    assert _weakly_identified_sides([1.0, 0.0], [0.4, np.nan]) == ()
    assert _weakly_identified_sides([1.0, 0.5], [0.6, np.inf]) == ("lower", "upper")
    assert _weakly_identified_sides([np.nan, 0.0], [np.nan, np.nan]) == ()


def test_mixture_decides_terms_for_all_components_together():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.gamma(3.0, 0.5, 300), rng.normal(8.0, 1.0, 300)])
    model = Distribution().fit(x, n_components=2, support=(0.0, np.inf), rng=0)
    allowed = {bool(c.data["boundary_allowed"][0]) for c in model.components}
    assert len(allowed) == 1
    amplitudes = [c.data["boundary_amplitudes"][0] for c in model.components]
    np.testing.assert_array_equal(amplitudes, [amplitudes[0]] * 2)
    p = _record(model)["p_values"][0]
    assert np.isnan(p) or 0.0 <= p <= 1.0
    for record in model.fit_diagnostics["components"]:
        assert "boundary_p_values" not in record
        assert "boundary_standard_errors" not in record
    assert model.fit_diagnostics["converged"]
