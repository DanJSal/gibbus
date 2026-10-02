"""Focused tests for spectral-construction acceleration helpers."""

import math

import numpy as np
from gibbus._spectral._certify import chebyshev_lower_bound
from gibbus._spectral._panel_kernels import (
    chebder as compiled_chebder,
)
from gibbus._spectral._panel_kernels import (
    chebint_scaled as compiled_chebint_scaled,
)
from gibbus._spectral._panel_kernels import (
    chebval_many as compiled_chebval_many,
)
from gibbus._spectral._panel_kernels import (
    lobatto_coefficients as compiled_lobatto_coefficients,
)
from numpy.polynomial import chebyshev as C
from spectral_builder_harness import PythonSpectralCDFBuilder, PythonSpectralPPFBuilder

from gibbus._spectral import chebyshev as chebyshev_helpers
from gibbus._spectral.cdf import SpectralCDF, density_spec
from gibbus._spectral.chebyshev import (
    _lobatto_transform,
    chebyshev_bernstein_matrix,
    lobatto_coefficients,
    lobatto_nodes,
    midpoint_nodes,
)


def _bernstein_eval(coeff, t):
    n = len(coeff) - 1
    return sum(
        coeff[k] * math.comb(n, k) * t**k * (1.0 - t) ** (n - k) for k in range(n + 1)
    )


def test_lobatto_transform_matches_square_chebfit():
    rng = np.random.default_rng(1234)
    for degree in (16, 24, 32):
        nodes = lobatto_nodes(degree)
        values = rng.normal(size=degree + 1)
        expected = C.chebfit(nodes, values, degree)
        actual = lobatto_coefficients(values)
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)


def test_compiled_panel_algebra_matches_numpy_chebyshev_calculus():
    """Post-fit panel kernels must preserve NumPy Chebyshev conventions."""
    rng = np.random.default_rng(55102)
    for degree in (2, 8, 16, 24, 32):
        values = np.ascontiguousarray(rng.normal(size=degree + 1))
        transform = np.ascontiguousarray(_lobatto_transform(degree))
        coeff = np.asarray(compiled_lobatto_coefficients(values, transform))
        expected = lobatto_coefficients(values)
        np.testing.assert_allclose(coeff, expected, rtol=2e-14, atol=2e-14)

        xx = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=37))
        np.testing.assert_allclose(
            compiled_chebval_many(xx, coeff),
            C.chebval(xx, coeff),
            rtol=3e-14,
            atol=3e-14,
        )

        integ, mass = compiled_chebint_scaled(coeff, 0.37)
        expected_integ = C.chebint(coeff, scl=0.37, lbnd=0.0)
        np.testing.assert_allclose(integ, expected_integ, rtol=3e-14, atol=3e-14)
        expected_mass = C.chebval(1.0, expected_integ) - C.chebval(-1.0, expected_integ)
        assert np.isclose(mass, expected_mass, rtol=2e-14, atol=2e-14)

        np.testing.assert_allclose(
            compiled_chebder(coeff), C.chebder(coeff), rtol=3e-14, atol=3e-14
        )


def test_chebyshev_bernstein_matrix_is_the_exact_basis_change():
    from fractions import Fraction

    for degree in (0, 1, 2, 5, 9, 13):
        matrix = chebyshev_bernstein_matrix(degree)
        for k in range(degree + 1):
            # Exact power coefficients of T_k(2t - 1), then exact Bernstein form.
            unit = [0] * k + [1]
            power_u = [Fraction(int(v)) for v in np.rint(C.cheb2poly(unit))]
            exact_t = [Fraction(0)] * (degree + 1)
            shifted = [Fraction(1)]
            for coefficient in power_u:
                for i, value in enumerate(shifted):
                    exact_t[i] += coefficient * value
                nxt = [Fraction(0)] * (len(shifted) + 1)
                for i, value in enumerate(shifted):
                    nxt[i] -= value
                    nxt[i + 1] += 2 * value
                shifted = nxt
            for j in range(degree + 1):
                exact = sum(
                    Fraction(math.comb(j, i), math.comb(degree, i)) * exact_t[i]
                    for i in range(j + 1)
                )
                assert matrix[j, k] == float(exact)


def test_chebyshev_lower_bound_never_exceeds_the_minimum():
    grid = np.linspace(-1.0, 1.0, 20001)
    rng = np.random.default_rng(67214)
    for degree in (0, 1, 2, 5, 8, 16, 24, 32, 40):
        matrix = chebyshev_bernstein_matrix(degree)
        for scale in (1e-12, 1e-4, 1.0, 1e4, 1e12):
            for decay in (0.0, 0.3, 1.0):
                coeff = rng.normal(scale=scale, size=degree + 1)
                coeff *= np.exp(-decay * np.arange(degree + 1))
                values = C.chebval(grid, coeff)
                lower = chebyshev_lower_bound(coeff, matrix, 8)
                assert lower <= values.min()
                if values.min() < 0.0:
                    # Tight where it matters: near a negative minimum.
                    assert lower >= values.min() - 1e-2 * np.abs(coeff).sum()


def test_chebyshev_lower_bound_certifies_positive_series():
    coeff = np.array([0.4, -0.15, 0.2, -0.05, 0.1])
    assert chebyshev_lower_bound(coeff, chebyshev_bernstein_matrix(4), 8) > 0.0
    shifted = coeff.copy()
    shifted[0] -= C.chebval(np.linspace(-1, 1, 4001), coeff).min() + 1e-3
    assert chebyshev_lower_bound(shifted, chebyshev_bernstein_matrix(4), 8) < 0.0


def test_compiled_panel_inversion_round_trips_source_cdf():
    def pdf(x):
        return np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2)

    rep = PythonSpectralCDFBuilder(pdf, (-np.inf, np.inf), mode=0.0, std=1.0)
    fractions = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float64)
    for j in range(len(rep.panels)):
        z = rep._cython_evaluator.invert_panel_fraction(j, fractions)
        actual = np.asarray(rep.cdf_z(z), dtype=np.float64)
        expected = rep.cum_mass[j] + fractions * rep.panel_masses[j]
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8e-14)


def test_midpoint_nodes_match_direct_construction():
    for count in (7, 35, 53, 69):
        expected = np.cos(np.pi * (np.arange(count) + 0.5) / count)
        np.testing.assert_array_equal(midpoint_nodes(count), expected)


def test_spectral_helper_caches_are_bounded():
    cached_helpers = (
        chebyshev_helpers.lobatto_nodes,
        chebyshev_helpers.midpoint_nodes,
        chebyshev_helpers._lobatto_transform,
        chebyshev_helpers.chebyshev_bernstein_matrix,
    )
    for helper in cached_helpers:
        assert helper.cache_info().maxsize == chebyshev_helpers._CACHE_MAXSIZE
        assert helper.cache_info().maxsize is not None


def _logit_probability(p):
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def _spectral_cdf_cases():
    return (
        (
            lambda x: np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2),
            (-np.inf, np.inf),
            0.0,
            1.0,
        ),
        (
            lambda x: np.exp(-np.asarray(x, dtype=np.float64)),
            (0.0, np.inf),
            0.0,
            1.0,
        ),
        (
            lambda x: np.exp(np.asarray(x, dtype=np.float64)),
            (-np.inf, 0.0),
            0.0,
            1.0,
        ),
        (
            lambda x: np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2),
            (-2.0, 3.0),
            0.0,
            1.0,
        ),
    )


def test_compiled_compact_cdf_matches_construction_evaluator():
    """Packed compact-coordinate validation must match the builder CDF."""
    rng = np.random.default_rng(77123)
    for pdf, support, mode, std in _spectral_cdf_cases():
        rep = PythonSpectralCDFBuilder(pdf, support, mode=mode, std=std)
        z = np.concatenate(
            [
                np.array([-1.0, 1.0]),
                np.linspace(-0.999, 0.999, 257),
                rng.uniform(-1.0, 1.0, size=251),
            ]
        )
        expected = np.asarray(rep.cdf_z(z), dtype=np.float64)
        actual = np.asarray(rep._cython_evaluator.eval_compact(z), dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-15)


def test_lobatto_transform_matches_square_chebfit_across_degrees():
    rng = np.random.default_rng(30191)
    for degree in range(1, 65):
        nodes = lobatto_nodes(degree)
        values = rng.normal(size=degree + 1)
        expected = C.chebfit(nodes, values, degree)
        actual = lobatto_coefficients(values)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


def test_compiled_panel_inversion_extreme_and_random_fractions():
    rng = np.random.default_rng(91827)
    extreme = np.array(
        [
            0.0,
            np.nextafter(0.0, 1.0),
            1e-15,
            1e-12,
            1e-9,
            0.5,
            1.0 - 1e-9,
            1.0 - 1e-12,
            1.0 - 1e-15,
            np.nextafter(1.0, 0.0),
            1.0,
        ],
        dtype=np.float64,
    )
    random_fractions = np.sort(rng.uniform(size=37))
    fractions = np.sort(np.concatenate((extreme, random_fractions)))

    for pdf, support, mode, std in _spectral_cdf_cases():
        rep = PythonSpectralCDFBuilder(pdf, support, mode=mode, std=std)
        for j in range(len(rep.panels)):
            z = np.asarray(
                rep._cython_evaluator.invert_panel_fraction(j, fractions),
                dtype=np.float64,
            )
            actual = np.asarray(rep.cdf_z(z), dtype=np.float64)
            expected = rep.cum_mass[j] + fractions * rep.panel_masses[j]
            assert np.all(np.diff(z) >= 0.0)
            assert z[0] == rep.breaks[j]
            assert z[-1] == rep.breaks[j + 1]
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8e-14)


def test_exact_z_for_r_many_handles_boundaries_and_fallback_panels():
    def pdf(x):
        return np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2)

    cdf = PythonSpectralCDFBuilder(pdf, (-np.inf, np.inf), mode=0.0, std=1.0)
    assert len(cdf.panels) >= 2

    inverse = object.__new__(PythonSpectralPPFBuilder)
    inverse.cdf_rep = cdf

    positive_mass = np.flatnonzero(np.diff(cdf.cum_mass) > 0.0)
    source_j = int(positive_mass[len(positive_mass) // 2])
    pa = float(cdf.cum_mass[source_j])
    pb = float(cdf.cum_mass[source_j + 1])

    inside = pa + (pb - pa) * np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    outside = []
    if source_j > 0:
        outside.append(float(cdf.cum_mass[source_j - 1] + cdf.cum_mass[source_j]) / 2.0)
    if source_j + 1 < len(cdf.panels):
        outside.append(
            float(cdf.cum_mass[source_j + 1] + cdf.cum_mass[source_j + 2]) / 2.0
        )
    probs = np.array(outside + inside.tolist(), dtype=np.float64)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)

    actual = inverse._exact_z_for_r_many(source_j, _logit_probability(probs))
    expected = np.array(
        [inverse._invert_global(float(p)) for p in probs], dtype=np.float64
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-14)

    round_trip = np.asarray(cdf.cdf_z(actual), dtype=np.float64)
    np.testing.assert_allclose(round_trip, probs, rtol=0.0, atol=8e-14)


def test_large_array_simd_dispatch_matches_compiled_scalar_paths():
    """SIMD-oriented CDF/PPF dispatch stays numerically identical to scalar loops."""

    def pdf(x):
        values = np.asarray(x, dtype=np.float64)
        return np.exp(-0.5 * values * values)

    cdf = PythonSpectralCDFBuilder(pdf, (-np.inf, np.inf), mode=0.0, std=1.0)
    ppf = PythonSpectralPPFBuilder(cdf)

    # More than the 256-point dispatch threshold.  Sorted input exercises the
    # contiguous-run path; the deterministic permutation exercises panel
    # bucketing/scatter without relying on a timing threshold.
    x_sorted = np.linspace(-4.0, 4.0, 513, dtype=np.float64)
    p_sorted = np.linspace(1e-4, 1.0 - 1e-4, 513, dtype=np.float64)
    permutation = np.random.default_rng(20260919).permutation(x_sorted.size)

    for x in (x_sorted, x_sorted[permutation]):
        bulk = np.asarray(cdf._cython_evaluator(x), dtype=np.float64)
        scalar = np.asarray(cdf._cython_evaluator.eval_scalar(x), dtype=np.float64)
        np.testing.assert_allclose(bulk, scalar, rtol=2e-15, atol=2e-15)

    for p_values in (p_sorted, p_sorted[permutation]):
        bulk = np.asarray(ppf.ppf_cython(p_values, simd=True), dtype=np.float64)
        scalar = np.asarray(ppf.ppf_cython(p_values, simd=False), dtype=np.float64)
        np.testing.assert_allclose(bulk, scalar, rtol=3e-15, atol=3e-15)


def test_half_line_map_resolves_a_body_far_from_the_endpoint():
    """A body many standard deviations from a finite endpoint is not missed.

    Regression: the endpoint-anchored map squeezed this fitted mixture
    component (quartic potential, endpoint 29 standard deviations below the
    mode) between the nodes of every panel, and the construction failed with
    zero mass.
    """
    from numpy.polynomial.polynomial import polyval
    from scipy.integrate import quad

    q_poly = np.array([0.0, -0.23727801, 0.52290041, 0.54643843, 0.2141385])
    q_poly[0] = np.log(quad(lambda z: np.exp(-polyval(z, q_poly)), -np.inf, np.inf)[0])
    support = (-24.09277201100829, np.inf)

    def pdf(z):
        return np.exp(-polyval(np.asarray(z, dtype=np.float64), q_poly))

    density = density_spec(
        [
            (
                q_poly,
                support[0],
                support[1],
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                -np.inf,
                np.inf,
            )
        ],
        view=False,
    )
    x = np.linspace(-3.0, 3.0, 25)
    expected = np.array(
        [quad(pdf, -np.inf, v, epsabs=1e-15, epsrel=1e-13)[0] for v in x]
    )
    for rep in (
        SpectralCDF(support, density=density, mode=0.1747, std=0.829),
        PythonSpectralCDFBuilder(pdf, support, mode=0.1747, std=0.829),
    ):
        assert rep.map.kind == "lower_centered"
        np.testing.assert_allclose(rep.cdf(x), expected, rtol=0.0, atol=1e-13)
