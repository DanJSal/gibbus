"""Expectations and spectral maps for mass far from a support end.

The integration and compactification geometry is measured in density widths
from the density's own breakpoints, so a narrow body far from a finite end
(a truncation that removes no mass, a remote exponential-moment saddle) is
resolved like any other.  Oracles are analytic normal and quadratic-potential
integrals.
"""

import math

import numpy as np
import pytest
from scipy.special import erfcx, log_ndtr

from gibbus import Distribution
from gibbus._postfit.basis_state import _log_exp_moment
from gibbus._postfit.expectation import expect, expect_vectorized
from gibbus._spectral.cdf import _map_kind

MU, SIGMA = 50.0, 1e-3


def _narrow_normal(x, n=0):
    """Normalized ``N(MU, SIGMA^2)`` potential and derivatives."""
    x = np.asarray(x, dtype=np.float64)
    if n == 0:
        return 0.5 * ((x - MU) / SIGMA) ** 2 + math.log(SIGMA * math.sqrt(2 * math.pi))
    if n == 1:
        return (x - MU) / SIGMA**2
    return np.full_like(x, 1.0 / SIGMA**2)


@pytest.mark.parametrize("support", [(0.0, math.inf), (0.0, 1e4), (-1e6, math.inf)])
@pytest.mark.parametrize("engine", [expect, expect_vectorized])
def test_remote_narrow_mass_is_fully_integrated(support, engine):
    def one(x):
        return np.ones_like(np.asarray(x, dtype=np.float64))

    def first(x):
        return np.asarray(x, dtype=np.float64) - MU

    def square(x):
        return (np.asarray(x, dtype=np.float64) - MU) ** 2

    assert engine(_narrow_normal, support, one, points=[MU]) == pytest.approx(
        1.0, abs=1e-10
    )
    assert engine(_narrow_normal, support, first, points=[MU]) == pytest.approx(
        0.0, abs=1e-12
    )
    assert engine(_narrow_normal, support, square, points=[MU]) == pytest.approx(
        SIGMA**2, rel=1e-8
    )


def test_narrow_exponential_moment_is_exact():
    got = _log_exp_moment(_narrow_normal, (0.0, math.inf), math.inf, 1, MU, SIGMA)
    assert got == pytest.approx(MU + 0.5 * SIGMA**2, abs=1e-10)


def test_remote_tilted_saddle_recovers_the_analytic_log_moment():
    # V(x) = a x^2 + b x + log Z on [0, inf): the k = 1 tilted saddle lies at
    # (1 - b) / (2 a) = 5e10, far beyond the mass near 1.1.  Its log moment
    # is finite although exp of it overflows.
    a, b = 1e-12, 0.9
    log_z = (
        0.5 * math.log(math.pi / a)
        - math.log(2.0)
        + math.log(erfcx(b / (2 * math.sqrt(a))))
    )

    def potential(x, n=0):
        x = np.asarray(x, dtype=np.float64)
        if n == 0:
            return a * x * x + b * x + log_z
        if n == 1:
            return 2 * a * x + b
        return np.full_like(x, 2 * a)

    c = b - 1.0
    exact = (
        0.5 * math.log(math.pi / a) + c * c / (4 * a) + log_ndtr(-c / math.sqrt(2 * a))
    ) - log_z
    got = _log_exp_moment(potential, (0.0, math.inf), math.inf, 1, 1.0, 1.0)
    assert got == pytest.approx(exact, abs=2e-6, rel=0.0)


@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_exponential_tail_rate_is_the_exact_moment_threshold(rate):
    def potential(x, n=0):
        x = np.asarray(x, dtype=np.float64)
        if n == 0:
            return rate * x - math.log(rate)
        return np.full_like(x, rate if n == 1 else 0.0)

    assert _log_exp_moment(potential, (0, math.inf), rate, int(rate), 1.0, 1.0) == (
        math.inf
    )
    if rate > 1.0:
        moment = _log_exp_moment(potential, (0, math.inf), rate, 1, 1.0, 1.0)
        assert math.exp(moment) == pytest.approx(rate / (rate - 1.0), rel=1e-11)


@pytest.fixture(scope="module")
def narrow_fit():
    data = np.random.default_rng(0).normal(MU, SIGMA, 500)
    return Distribution().fit(data, poly_degree=2)


@pytest.mark.parametrize("bounds", [(0.0, None), (0.0, 1e4), (-1e3, 1e5)])
def test_truncation_removing_no_mass_preserves_the_law(narrow_fit, bounds):
    lower, upper = bounds
    truncated = narrow_fit.truncate(lower=lower, upper=upper)
    assert truncated.mean == pytest.approx(narrow_fit.mean, abs=1e-10)
    assert truncated.var == pytest.approx(narrow_fit.var, rel=1e-8)
    assert truncated.exp.mean == pytest.approx(narrow_fit.exp.mean, rel=1e-9)
    assert truncated.expect(lambda x: 1.0) == pytest.approx(1.0, abs=1e-10)
    x = narrow_fit.ppf([0.1, 0.5, 0.9])
    np.testing.assert_allclose(truncated.cdf(x), narrow_fit.cdf(x), atol=1e-12)


def test_public_expectation_of_a_narrow_fit_carries_unit_mass(narrow_fit):
    assert narrow_fit.expect(lambda x: 1.0) == pytest.approx(1.0, abs=1e-10)
    assert narrow_fit.expect(lambda x: x) == pytest.approx(narrow_fit.mean, abs=1e-9)


@pytest.mark.parametrize(
    ("support", "mode", "kind"),
    [
        ((0.0, 1.0), 0.5, "finite"),
        ((0.0, 1e4), 0.5, "lower"),
        ((-1e4, 1.0), 0.5, "upper"),
        ((0.0, 1e4), 50.0, "centered"),
        ((0.0, math.inf), 0.5, "lower"),
        ((0.0, math.inf), 50.0, "centered"),
        ((-math.inf, math.inf), 0.0, "centered"),
    ],
)
def test_map_family_follows_the_body_not_the_support_length(support, mode, kind):
    assert _map_kind(*support, mode, 0.3) == kind
