"""Numerical regressions for centered and transformed fitted moments."""

import numpy as np
import pytest

from gibbus import Distribution


def _normal_fit(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    return Distribution().fit(
        np.ascontiguousarray(rng.normal(size=n)),
        n_components=1,
        support=(-np.inf, np.inf),
        rng=seed,
    )


def test_single_component_centered_moments_are_translation_invariant():
    base = _normal_fit()
    moved = base.transform(mu=1e12, sigma=1.0, pullback=False, inplace=False)

    assert moved.var == base.var
    assert moved.skew == base.skew
    assert moved.kurt == base.kurt
    for k in (2, 3, 4):
        assert moved.moment(k, central=True) == base.moment(k, central=True)
        assert moved.moment(k, standardized=True) == base.moment(k, standardized=True)


def test_mixture_centered_moments_keep_common_translation_separate():
    rng = np.random.default_rng(1)
    data = np.ascontiguousarray(np.concatenate([
        rng.normal(-3.0, 1.0, 1500),
        rng.normal(3.0, 1.0, 1500),
    ]))
    base = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    moved = base.transform(mu=1e12, sigma=1.0, pullback=False, inplace=False)

    assert moved.var == base.var
    assert moved.skew == base.skew
    assert moved.kurt == base.kurt
    for k in (2, 3, 4):
        assert moved.moment(k, central=True) == base.moment(k, central=True)
        assert moved.moment(k, standardized=True) == base.moment(k, standardized=True)


def test_narrow_exp_space_uses_direct_centered_relative_moments():
    base = _normal_fit(n=6000)
    narrow = base.transform(mu=30.0, sigma=9e-8, pullback=False, inplace=False)

    # In the vanishing-log-variance limit a lognormal-like law approaches a
    # symmetric normal shape in relative units: skew -> 0, Pearson kurt -> 3.
    assert abs(narrow.exp.skew) < 2e-6
    assert narrow.exp.kurt == pytest.approx(3.0, abs=2e-5)
    assert narrow.exp.moment(3, standardized=True) == narrow.exp.skew
    assert narrow.exp.moment(4, standardized=True) == narrow.exp.kurt



def test_narrow_exp_mixture_uses_direct_centered_relative_moments():
    rng = np.random.default_rng(7)
    data = np.ascontiguousarray(np.concatenate([
        rng.normal(-2.0, 0.7, 1800),
        rng.normal(2.5, 1.0, 1200),
    ]))
    base = Distribution().fit(
        data, n_components=2, support=(-np.inf, np.inf), rng=0
    )
    base_skew = base.skew
    base_kurt = base.kurt

    narrow = base.transform(
        mu=30.0, sigma=9e-8, pullback=False, inplace=False
    )
    narrow.set_default("exp")

    # exp(mu + sigma X) is affine in X to first order as sigma -> 0, so
    # standardized centered moments approach those of the base-space mixture.
    assert narrow.skew == pytest.approx(base_skew, abs=2e-5)
    assert narrow.kurt == pytest.approx(base_kurt, abs=2e-5)
    assert narrow.moment(3, standardized=True) == narrow.skew
    assert narrow.moment(4, standardized=True) == narrow.kurt

def test_exp_mixture_shape_survives_dimensional_overflow():
    rng = np.random.default_rng(2)
    data = np.ascontiguousarray(np.concatenate([
        rng.normal(-1.0, 0.6, 1200),
        rng.normal(1.0, 0.9, 800),
    ]))
    base = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    base.set_default("exp")
    base_skew = base.skew
    base_kurt = base.kurt

    shifted = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    shifted = shifted.transform(mu=700.0, sigma=1.0, pullback=False, inplace=False)
    shifted.set_default("exp")

    assert np.isinf(shifted.var)
    assert shifted.skew == pytest.approx(base_skew, abs=2e-10)
    assert shifted.kurt == pytest.approx(base_kurt, abs=2e-9)
    assert shifted.moment(3, standardized=True) == shifted.skew
    assert shifted.moment(4, standardized=True) == shifted.kurt
