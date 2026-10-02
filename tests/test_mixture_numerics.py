"""Focused numerical regressions for mixture spectral and mode machinery."""

import numpy as np
import pytest
from scipy.special import logsumexp

from gibbus import Distribution, clear_suppressed_failures, suppressed_failures
from gibbus._fit import mixture as mixture_module
from gibbus._fit.mixture import (
    _collect_roots_bisection_func,
    _find_mixture_modes_base,
    _find_mixture_modes_exp,
)


def _normal_mixture_potential(mus, sigmas, weights):
    mus = np.asarray(mus, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    def potential(x, n):
        x = float(x)
        z = (x - mus) / sigmas
        log_terms = (
            np.log(weights) - np.log(sigmas) - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * z * z
        )
        log_pdf = float(logsumexp(log_terms))
        responsibilities = np.exp(log_terms - log_pdf)
        score_terms = -(x - mus) / (sigmas * sigmas)
        score = float(np.dot(responsibilities, score_terms))
        if n == 0:
            return -log_pdf
        if n == 1:
            return -score
        if n == 2:
            second_terms = (
                (x - mus) ** 2 / sigmas ** 4 - 1.0 / sigmas ** 2
            )
            pdf_second_over_pdf = float(
                np.dot(responsibilities, second_terms))
            return score * score - pdf_second_over_pdf
        raise AssertionError("test potential only supplies derivatives through order 2")

    return potential


def test_mode_root_scan_finds_even_number_of_roots_with_same_sign_samples():
    candidates = []
    _collect_roots_bisection_func(
        lambda x: (x - 0.2) * (x - 0.3), 0.0, 1.0, candidates)
    assert sorted(candidates) == pytest.approx([0.2, 0.3], abs=2e-10)


def test_base_mixture_modes_do_not_duplicate_a_slightly_displaced_seed():
    # Each remote component pulls the other's mixture mode about 8e-9 away
    # from its component mode: close enough for the seed to pass the
    # stationarity check, too far for the fixed refinement bracket.  The seed
    # and the scan root must be reported as one mode, not two.
    potential = _normal_mixture_potential([0.05, 6.45], [1.0, 1.0], [0.5, 0.5])
    seeds = [0.05, 6.45]
    assert 1e-9 < abs(potential(seeds[0], 1)) < 1e-8

    modes = _find_mixture_modes_base(potential, seeds)

    assert len(modes) == 2
    assert np.diff(modes).min() > 1.0
    for x in modes:
        assert potential(x, 1) == pytest.approx(0.0, abs=1e-12)


def test_readme_trimodal_example_reports_each_mode_once():
    # The README's trimodal example reported one central mode twice for
    # several data seeds.  Distinct modes of a smooth fit are separated by an
    # antimode, so no two reported modes can be nanometres apart.
    for seed in (2, 5):
        rng = np.random.default_rng(seed)
        data = np.concatenate([
            rng.normal(-5, 0.8, 200), rng.normal(0, 1.0, 300), rng.normal(5, 0.6, 200),
        ])
        c = Distribution().fit(data, n_components=3, support=(-np.inf, np.inf), rng=42)
        modes = np.asarray(c.modes)
        assert np.diff(modes).min() > 1e-3, modes

def test_exp_mixture_modes_use_transformed_component_seeds():
    # For N(0, sigma^2), the exp-space component mode is at log y=-sigma^2.
    # A broad concentric component therefore has a transformed mode far outside
    # the hull of the base-space component modes, which are both zero.
    potential = _normal_mixture_potential([0.0, 0.0], [1.0, 5.0], [0.7, 0.3])
    modes = _find_mixture_modes_exp(potential, [-1.0, -25.0])

    assert len(modes) == 2
    log_modes = np.log(np.asarray(modes))
    assert log_modes[0] == pytest.approx(-25.0, abs=5e-3)
    assert log_modes[1] == pytest.approx(-1.155, abs=2e-2)
    for y in modes:
        x = np.log(y)
        assert potential(x, 1) + 1.0 == pytest.approx(0.0, abs=2e-8)
        assert potential(x, 2) > 0.0


@pytest.mark.parametrize("failure", [ValueError, OverflowError])
def test_gmm_initialisation_failure_degrades_to_nested_scale(monkeypatch, failure):
    data = np.linspace(-2.0, 2.0, 20)
    expected_resp = np.full((data.size, 2), 0.5, dtype=np.float64)
    expected_mix = np.array([0.5, 0.5], dtype=np.float64)

    def no_valley(samples, n_components, *, weights=None):
        return None, None

    def failing_gmm(samples, n_components, rng):
        raise failure("synthetic GMM failure")

    def nested_scale(samples, n_components, *, weights=None):
        return expected_resp, expected_mix

    monkeypatch.setattr(mixture_module, "_valley_init_responsibilities", no_valley)
    monkeypatch.setattr(mixture_module, "_gmm_init_responsibilities", failing_gmm)
    monkeypatch.setattr(mixture_module, "_nested_scale_init_responsibilities", nested_scale)

    clear_suppressed_failures()
    candidates = mixture_module._initial_responsibility_candidates(
        data, 2, np.random.default_rng(0)
    )
    assert len(candidates) == 1
    assert candidates[0][0] == "nested-scale"
    np.testing.assert_array_equal(candidates[0][1], expected_resp)
    np.testing.assert_array_equal(candidates[0][2], expected_mix)
    records = suppressed_failures()
    assert records[-1]["context"] == "GMM mixture initialisation"
    assert records[-1]["type"] == failure.__name__
    clear_suppressed_failures()


def test_mixture_spectral_scale_survives_large_common_translation():
    rng = np.random.default_rng(0)
    data = np.ascontiguousarray(np.concatenate([
        rng.normal(-3.0, 1.0, 800),
        rng.normal(3.0, 1.0, 800),
    ]))
    base = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    probs = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    base_q = np.asarray(base.ppf(probs), dtype=np.float64)

    shift = 1e12
    moved = base.transform(mu=shift, sigma=1.0, pullback=False, inplace=False)
    moved_q = np.asarray(moved.ppf(probs), dtype=np.float64)
    expected_q = shift + base_q

    # Public coordinates at 1e12 resolve only about 1.22e-4, so compare at the
    # representable-coordinate floor rather than to an impossible sub-ulp target.
    ulp = np.abs(np.spacing(expected_q))
    assert np.all(np.abs(moved_q - expected_q) <= 2.0 * ulp)

    # At this translation one public-coordinate ulp moves the CDF by about
    # 1e-5, so a fixed probability tolerance can demand accuracy that no
    # representable x value can attain.  The inverse is correct when the
    # requested probability is bracketed by the CDF at the adjacent floats.
    q_lo = np.nextafter(moved_q, -np.inf)
    q_hi = np.nextafter(moved_q, np.inf)
    cdf_lo = np.asarray(moved.cdf(q_lo), dtype=np.float64)
    cdf_hi = np.asarray(moved.cdf(q_hi), dtype=np.float64)
    assert np.all(cdf_lo <= probs)
    assert np.all(probs <= cdf_hi)
