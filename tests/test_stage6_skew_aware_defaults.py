"""Automatic full-line transition geometry and exclusion safeguards."""

import math

import numpy as np
import pytest

from gibbus import BasisSpec, TanhSpec
from gibbus._fit.basis_selection import _PointData
from gibbus._model.basis import _build_basis
from gibbus._model.basis_geometry import _weighted_quantile


def _sample(n=700, seed=214):
    return np.random.default_rng(seed).gumbel(size=n)


def _expected_geometry(x, h):
    """Independent transcription of the preselected quantile heuristic."""
    n = len(x)
    q10, q50, q90 = np.quantile(x, (0.1, 0.5, 0.9))
    asym = math.log((q90 - q50) / (q50 - q10))
    s = max(0.0, abs(asym) - 0.12)
    w = float(np.clip((n - 150) / 550, 0, 1))
    change = min(0.07, 0.22 * s) * w
    slow = 2 * (np.clip(2 * math.exp(-5 * s), 0.4, 2) / 2) ** w
    fast = 2 * (np.clip(2 * math.exp(0.75 * s), 2, 5) / 2) ** w
    if asym >= 0:
        return np.quantile(x, 0.1), np.quantile(x, 0.9 - change), fast / h, slow / h
    return np.quantile(x, 0.1 + change), np.quantile(x, 0.9), slow / h, fast / h


@pytest.mark.parametrize("n", [150, 350, 700, 1200])
def test_default_matches_fixed_heuristic(n):
    x = _sample(n)
    g = _build_basis(BasisSpec(), x).geometry
    expected = _expected_geometry(x, g.scale)
    assert (
        g.lower_location,
        g.upper_location,
        g.lower_steepness,
        g.upper_steepness,
    ) == pytest.approx(expected, rel=1e-13)
    if n == 150:
        assert g.lower_steepness == pytest.approx(2 / g.scale)
        assert g.upper_steepness == pytest.approx(2 / g.scale)


def test_skewed_sample_has_distinct_steepnesses():
    g = _build_basis(BasisSpec(), _sample()).geometry
    assert g.upper_steepness < 2 / g.scale < g.lower_steepness
    assert g.upper_location < np.quantile(_sample(), 0.9)


def test_reflected_data_exchange_gates():
    x = _sample()
    a = _build_basis(BasisSpec(), x).geometry
    b = _build_basis(BasisSpec(), -x).geometry
    assert b.lower_location == pytest.approx(-a.upper_location)
    assert b.upper_location == pytest.approx(-a.lower_location)
    assert b.lower_steepness == pytest.approx(a.upper_steepness)
    assert b.upper_steepness == pytest.approx(a.lower_steepness)


def test_one_sided_and_weighted_keep_classic_rule():
    x = _sample()
    w = np.ones(len(x))
    a = _build_basis(BasisSpec(), x, w).geometry
    assert a.lower_location == pytest.approx(_weighted_quantile(x, w, 0.1))
    assert a.upper_location == pytest.approx(_weighted_quantile(x, w, 0.9))
    assert a.lower_steepness == pytest.approx(2 / a.scale)
    assert a.upper_steepness == pytest.approx(2 / a.scale)
    y = np.exp(x)
    b = _build_basis(BasisSpec(support=(0, math.inf)), y).geometry
    assert b.lower_location is None
    assert b.upper_location == pytest.approx(np.quantile(y, 0.9))
    assert b.upper_steepness == pytest.approx(2 / b.scale)
    c = _build_basis(BasisSpec(support=(-math.inf, 0)), -y).geometry
    assert c.upper_location is None
    assert c.lower_location == pytest.approx(np.quantile(-y, 0.1))
    assert c.lower_steepness == pytest.approx(2 / c.scale)


def test_mixture_responsibilities_do_not_adapt_even_when_equal():
    x = _sample()
    data = _PointData(x, None, float(x.size))
    g = data.build_weighted(BasisSpec(), np.ones(x.size) / 2).geometry
    assert g.lower_location == pytest.approx(np.quantile(x, 0.1))
    assert g.upper_location == pytest.approx(np.quantile(x, 0.9))
    assert g.lower_steepness == pytest.approx(2 / g.scale)
    assert g.upper_steepness == pytest.approx(2 / g.scale)


def test_explicit_values_take_precedence_individually():
    x = _sample()
    settings = TanhSpec(lower_location=-2.5, upper_steepness=0.75)
    g = _build_basis(BasisSpec(tanh=settings), x).geometry
    assert g.lower_location == -2.5
    assert g.upper_steepness == 0.75
    ref = _build_basis(BasisSpec(), x).geometry
    assert g.upper_location == pytest.approx(ref.upper_location)
    assert g.lower_steepness == pytest.approx(ref.lower_steepness)


def test_symmetric_data_and_unresolved_quantiles_use_defaults():
    x = np.linspace(-2, 2, 700)
    g = _build_basis(BasisSpec(), x).geometry
    assert g.lower_location == pytest.approx(np.quantile(x, 0.1))
    assert g.upper_location == pytest.approx(np.quantile(x, 0.9))
    assert g.lower_steepness == pytest.approx(2 / g.scale)
    assert g.upper_steepness == pytest.approx(2 / g.scale)
    ties = np.r_[np.zeros(600), np.ones(100)]
    tied = _build_basis(BasisSpec(), ties).geometry
    assert np.isfinite(tied.lower_steepness)
    assert np.isfinite(tied.upper_steepness)


def test_public_full_line_fit_uses_auto_settings_by_default():
    """Default fitting and the same explicitly fixed geometry are identical."""
    from gibbus import Distribution, SoftPower

    x = _sample(seed=413)
    common = dict(
        bulk_degree=4,
        expansion="minimal",
        lower_tail=SoftPower(1.5),
        upper_tail=SoftPower(1.5),
    )
    auto_basis = BasisSpec(**common)
    reference = _build_basis(auto_basis, x).geometry
    explicit = BasisSpec(
        **common,
        tanh=TanhSpec(
            lower_location=reference.lower_location,
            upper_location=reference.upper_location,
            lower_steepness=reference.lower_steepness,
            upper_steepness=reference.upper_steepness,
        ),
    )
    automatic = Distribution().fit(x, basis=auto_basis, suppress_warnings=True, rng=413)
    manual = Distribution().fit(x, basis=explicit, suppress_warnings=True, rng=413)
    actual = automatic._components[0]._basis.parent.geometry
    assert actual.lower_location == reference.lower_location
    assert actual.upper_steepness == reference.upper_steepness
    grid = np.linspace(-3, 6, 23)
    np.testing.assert_allclose(automatic.pdf(grid), manual.pdf(grid), rtol=0, atol=0)
