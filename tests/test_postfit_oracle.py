"""Post-fit quantities against an independent high-precision oracle.

Each fitted component is rebuilt from its stored canonical coefficients and
evaluated by ``_basis_oracle`` (sympy/mpmath, sharing no evaluation code with
the compiled kernels).  The oracle's normalizer, distribution function,
quantiles, moments, entropy, exp-space mean and extreme tails come from
mpmath quadrature of that density, so they share nothing with the library's
normalizer, spectral CDF/PPF, expectation engine or tail inversion.  The
library's own summaries (median, standard deviation) only place quadrature
breakpoints, which changes no converged value.

Cases cover the four support kinds, endpoint logarithms, data far from a
half-line's endpoint (runtime recurrence coordinates) and a mixture.
"""

import math

import mpmath as mp
import numpy as np
import pytest
from _basis_oracle import BasisOracle

from gibbus import Distribution
from gibbus._model.basis import _face_model
from gibbus._postfit.basis_state import _model_from_record

_DPS = 30


class _LawOracle:
    """Independent density of one fitted component.

    Parameters
    ----------
    component : gibbus Component
        Fitted component (its packed state supplies the canonical record).
    """

    def __init__(self, component):
        state = component._data
        model, suppressed = _model_from_record(state["basis_model"])
        face, free = _face_model(model, suppressed)
        self.theta = np.asarray(state["basis_theta"])[list(free)]
        self.oracle = BasisOracle(face, units="x", max_order=1)
        self.lower, self.upper = (float(v) for v in state["support"])
        center, spread = float(component.median), float(component.std)
        offsets = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
        points = {center + k * spread for k in offsets} | {
            center - k * spread for k in offsets
        }
        inside = sorted(p for p in points if self.lower < p < self.upper)
        self.breaks = [self.lower, *inside, self.upper]
        # Integration range: where the density is above e^-1000 of its value
        # at the median (the tests probe tail masses down to e^-300).
        with mp.workdps(_DPS):
            peak = self.q(center)
            self.ends = []
            for end, sign in ((self.lower, -1.0), (self.upper, 1.0)):
                x = center
                while True:
                    x += sign * spread
                    if not (self.lower < x < self.upper):
                        x = end
                        break
                    if self.q(x) - peak > 1000:
                        break
                self.ends.append(x)
            self.log_z = mp.log(self._integral(lambda x: mp.mpf(1), *self._ends()))

    def _ends(self):
        return mp.mpf(self.ends[0]), mp.mpf(self.ends[1])

    def q(self, x, order=0):
        """Unnormalized potential (or its derivative) at ``x``."""
        return self.oracle.potential(self.theta, x, order, dps=_DPS)[0]

    def _density(self, x):
        return mp.exp(-self.q(x))

    def _integral(self, f, a, b):
        """``int_a^b f(x) exp(-q(x)) dx`` over the breakpoints inside.

        Next to a finite limit in a tail the density varies on the scale
        ``1 / |q'|``, far below the law's spread: breakpoints on that scale
        keep the quadrature converged there.
        """
        a = max(mp.mpf(a), mp.mpf(self.ends[0]))
        b = min(mp.mpf(b), mp.mpf(self.ends[1]))
        if not a < b:
            return mp.mpf(0)
        inner = {mp.mpf(p) for p in self.breaks if a < p < b}
        for end, sign in ((a, 1), (b, -1)):
            if not mp.isfinite(end) or end in (self.lower, self.upper):
                continue
            slope = abs(self.q(end, 1))
            if slope > 0:
                for j in (0.5, 1.5, 3, 6, 12, 25, 50, 100, 200, 400):
                    point = end + sign * mp.mpf(j) / slope
                    if a < point < b:
                        inner.add(point)
        nodes = [a, *sorted(inner), b]

        def integrand(x):
            density = self._density(x)
            # A zero density at an endpoint log (q = inf) contributes zero.
            return f(x) * density if density else mp.mpf(0)

        return mp.quad(integrand, nodes)

    def logpdf(self, x):
        with mp.workdps(_DPS):
            return float(-self.q(x) - self.log_z)

    def log_mass(self, a, b):
        """``log P(a < X < b)`` in mpmath (no underflow in far tails)."""
        with mp.workdps(_DPS):
            mass = self._integral(lambda x: mp.mpf(1), mp.mpf(a), mp.mpf(b))
            return mp.log(mass) - self.log_z

    def cdfs(self, points):
        """Distribution function at many points, from consecutive pieces."""
        order = np.argsort(points)
        out = np.empty(len(points))
        with mp.workdps(_DPS):
            total, last = mp.mpf(0), self.lower
            for i in order:
                total += mp.exp(self.log_mass(last, points[i]))
                last = points[i]
                out[i] = float(total)
        return out

    def cdf(self, x):
        return float(mp.exp(self.log_mass(self.lower, x)))

    def sf(self, x):
        return float(mp.exp(self.log_mass(x, self.upper)))

    def expect(self, f):
        """``E[f(X)]`` (``f`` takes and returns mpmath numbers)."""
        with mp.workdps(_DPS):
            return self._integral(f, *self._ends()) / mp.exp(self.log_z)

    def ppf(self, p, start):
        """Quantile by Newton's method on the oracle distribution function."""
        x = mp.mpf(start)
        with mp.workdps(_DPS):
            for _ in range(30):
                excess = mp.exp(self.log_mass(self.lower, x)) - p
                step = excess / mp.exp(-self.q(x) - self.log_z)
                x -= step
                if abs(step) <= mp.mpf(10) ** (-20) * (1 + abs(x)):
                    break
            return float(x)

    def isf(self, p, start):
        """Upper quantile by Newton's method on the oracle survival function."""
        x = mp.mpf(start)
        with mp.workdps(_DPS):
            for _ in range(30):
                excess = mp.exp(self.log_mass(x, self.upper)) - p
                step = -excess / mp.exp(-self.q(x) - self.log_z)
                x -= step
                if abs(step) <= mp.mpf(10) ** (-20) * (1 + abs(x)):
                    break
            return float(x)

    def mode(self, start):
        """Root of the oracle potential's slope (Newton on the slope)."""
        x = mp.mpf(start)
        with mp.workdps(_DPS):
            h = mp.mpf(10) ** (-12) * (1 + abs(x))
            for _ in range(40):
                g = self.q(x, 1)
                curvature = (self.q(x + h, 1) - self.q(x - h, 1)) / (2 * h)
                step = g / curvature
                x -= step
                if abs(step) <= mp.mpf(10) ** (-15) * (1 + abs(x)):
                    break
            return float(x)


def _fit(case):
    rng = np.random.default_rng(case["seed"])
    data = case["data"](rng)
    return (
        Distribution().fit(
            data, n_components=case.get("k", 1), rng=0, **case["options"]
        ),
        data,
    )


_CASES = {
    "real_line": {
        "seed": 3,
        "data": lambda rng: rng.standard_t(8, 400),
        "options": {"support": (-np.inf, np.inf)},
    },
    "lower_half_line_log": {
        "seed": 4,
        "data": lambda rng: rng.gamma(3.0, 1.0, 400),
        "options": {"support": (0.0, np.inf)},
    },
    "upper_half_line": {
        "seed": 5,
        "data": lambda rng: 2.0 - rng.weibull(1.7, 400),
        "options": {"support": (-np.inf, 2.0)},
    },
    "bounded_logs": {
        "seed": 6,
        "data": lambda rng: rng.beta(2.5, 4.0, 400),
        "options": {"support": (0.0, 1.0)},
    },
    "far_from_endpoint": {
        "seed": 7,
        "data": lambda rng: 1000.0 + 8.0 * rng.normal(size=400),
        "options": {"support": (0.0, np.inf), "poly_degree": 4},
    },
}


@pytest.fixture(scope="module", params=sorted(_CASES))
def fitted(request):
    """A fitted single-component law, its oracle and its data."""
    dist, data = _fit(_CASES[request.param])
    return request.param, dist, _LawOracle(dist.components[0]), data


def _probe(oracle, dist):
    """Points through the body and well into both tails."""
    p = np.array([1e-9, 1e-4, 0.02, 0.2, 0.5, 0.8, 0.98, 1 - 1e-4, 1 - 1e-9])
    return np.asarray(dist.ppf(p), dtype=float)


def test_log_density(fitted):
    _, dist, oracle, _ = fitted
    x = _probe(oracle, dist)
    expected = np.array([oracle.logpdf(v) for v in x])
    np.testing.assert_allclose(dist.logpdf(x), expected, rtol=0, atol=2e-11)


def test_distribution_function_and_survival(fitted):
    _, dist, oracle, _ = fitted
    x = _probe(oracle, dist)
    cdf = np.array([oracle.cdf(v) for v in x])
    sf = np.array([oracle.sf(v) for v in x])
    np.testing.assert_allclose(dist.cdf(x), cdf, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(dist.sf(x), sf, rtol=1e-10, atol=1e-14)


def _tail_probability(oracle, x, lower_tail):
    """Oracle ``P(X < x)`` or ``P(X > x)`` (never formed as one minus the other)."""
    if lower_tail:
        return float(mp.exp(oracle.log_mass(oracle.lower, x)))
    return float(mp.exp(oracle.log_mass(x, oracle.upper)))


def test_quantiles(fitted):
    """Quantiles meet the inverse's probability contract on the oracle law.

    The spectral inverse resolves the log-odds of ``p`` to 2e-10
    (``SPECTRAL_PPF_LOGIT_TOL``), so the oracle probability at the returned
    quantile must match ``p`` relative to the smaller tail.
    """
    _, dist, oracle, _ = fitted
    p = np.array([1e-12, 1e-6, 0.05, 0.25, 0.5, 0.75, 0.95, 1 - 1e-6])
    for x, pi in zip(np.asarray(dist.ppf(p), dtype=float), p, strict=True):
        lower_tail = pi <= 0.5
        target = pi if lower_tail else 1.0 - pi
        assert _tail_probability(oracle, x, lower_tail) == pytest.approx(
            target, rel=1e-9
        )
    for x, pi in zip(np.asarray(dist.isf(p), dtype=float), p, strict=True):
        upper_tail = pi <= 0.5
        target = pi if upper_tail else 1.0 - pi
        assert _tail_probability(oracle, x, not upper_tail) == pytest.approx(
            target, rel=1e-9
        )


def test_extreme_tails(fitted):
    _, dist, oracle, _ = fitted
    for side in ("lower", "upper"):
        end = oracle.lower if side == "lower" else oracle.upper
        if math.isfinite(end):
            continue
        # A point whose tail mass is near e^-300, found on the oracle itself.
        x = float(dist.median)
        step = float(dist.std) * (-1.0 if side == "lower" else 1.0)
        while True:
            log_tail = (
                oracle.log_mass(oracle.lower, x)
                if side == "lower"
                else oracle.log_mass(x, oracle.upper)
            )
            if log_tail < -300:
                break
            x += step
        got = dist.logcdf(x) if side == "lower" else dist.logsf(x)
        assert got == pytest.approx(float(log_tail), rel=1e-11)


def test_summaries(fitted):
    _, dist, oracle, _ = fitted
    mean = oracle.expect(lambda x: x)
    centered = [oracle.expect(lambda x, k=k: (x - mean) ** k) for k in (2, 3, 4)]
    var = float(centered[0])
    scale = math.sqrt(var)
    assert dist.mean == pytest.approx(
        float(mean), rel=0, abs=1e-12 * (abs(float(mean)) + scale)
    )
    assert dist.var == pytest.approx(var, rel=1e-10)
    assert dist.skew == pytest.approx(float(centered[1]) / scale**3, rel=0, abs=1e-9)
    # Pearson kurtosis (3 for a Gaussian), the documented convention.
    assert dist.kurt == pytest.approx(float(centered[2]) / var**2, rel=0, abs=1e-9)
    entropy = oracle.expect(lambda x: oracle.q(x) + oracle.log_z)
    assert dist.entropy() == pytest.approx(float(entropy), rel=0, abs=1e-10)
    assert dist.median == pytest.approx(
        oracle.ppf(0.5, dist.median), rel=0, abs=1e-10 * scale
    )
    assert dist.mode == pytest.approx(oracle.mode(dist.mode), rel=0, abs=1e-8 * scale)


def test_hazard_and_residual_life(fitted):
    _, dist, oracle, _ = fitted
    x = float(dist.ppf(0.7))
    survival = oracle.sf(x)
    assert dist.hazard(x) == pytest.approx(
        math.exp(oracle.logpdf(x)) / survival, rel=1e-10
    )
    with mp.workdps(_DPS):
        tail = oracle._integral(
            lambda t: t - x, mp.mpf(x), mp.mpf(oracle.upper)
        ) / mp.exp(oracle.log_z)
    assert dist.mean_residual_life(x) == pytest.approx(float(tail) / survival, rel=1e-9)


def test_exp_space_mean(fitted):
    name, dist, oracle, _ = fitted
    if name == "far_from_endpoint":
        pytest.skip("e^X overflows at X ~ 1000")
    rate = dist.tail_rate("upper")
    if math.isfinite(rate) and rate <= 1.0:
        assert math.isinf(dist.exp.mean)
        return
    want = oracle.expect(lambda x: mp.exp(x))
    assert dist.exp.mean == pytest.approx(float(want), rel=1e-10)


def test_truncation_and_affine_view(fitted):
    _, dist, oracle, _ = fitted
    a, b = (float(v) for v in dist.ppf([0.1, 0.85]))
    x = float(dist.ppf(0.4))
    truncated = dist.truncate(a, b)
    with mp.workdps(_DPS):
        inner = mp.exp(oracle.log_mass(a, x) - oracle.log_mass(a, b))
    assert truncated.cdf(x) == pytest.approx(float(inner), rel=1e-10)
    moved = dist.transform(mu=-3.0, sigma=2.5, pullback=False, inplace=False)
    assert moved.cdf(-3.0 + 2.5 * x) == pytest.approx(oracle.cdf(x), rel=1e-10)
    assert moved.mean == pytest.approx(-3.0 + 2.5 * dist.mean, rel=1e-12, abs=1e-12)


def test_sampling_follows_the_law(fitted):
    _, dist, oracle, _ = fitted
    draws = np.sort(np.asarray(dist.sample(3000, rng=11), dtype=float))
    # Kolmogorov-Smirnov distance to the oracle distribution function on a
    # thinned set of order statistics (each oracle CDF is a quadrature).
    idx = np.linspace(0, draws.size - 1, 60).astype(int)
    f = oracle.cdfs(draws[idx])
    n = draws.size
    distance = max(np.max((idx + 1) / n - f), np.max(f - idx / n))
    # Fixed seed; the 0.999 critical value of the KS distance is 1.95/sqrt(n).
    assert distance < 1.95 / math.sqrt(n)


def test_goodness_of_fit_statistic(fitted):
    _, dist, oracle, _ = fitted
    x = np.sort(np.asarray(dist.sample(40, rng=5), dtype=float))
    f = oracle.cdfs(x)
    n = x.size
    cvm = 1.0 / (12 * n) + float(
        np.sum((f - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2)
    )
    result = dist.goodness_of_fit(x, statistic="cvm")
    assert result["value"] == pytest.approx(cvm, rel=1e-9)


def test_mixture_density_and_distribution_function():
    rng = np.random.default_rng(12)
    data = np.r_[rng.normal(-3.0, 1.0, 300), rng.gamma(4.0, 1.0, 300) + 1.0]
    dist = Distribution().fit(data, n_components=2, support=(-np.inf, np.inf), rng=0)
    weights = np.asarray(dist.weights, dtype=float)
    oracles = [_LawOracle(c) for c in dist.components]
    x = np.asarray(dist.ppf([1e-6, 0.1, 0.4, 0.6, 0.9, 1 - 1e-6]), dtype=float)
    pdf = np.array(
        [
            sum(
                w * math.exp(o.logpdf(v)) for w, o in zip(weights, oracles, strict=True)
            )
            for v in x
        ]
    )
    cdf = np.array(
        [sum(w * o.cdf(v) for w, o in zip(weights, oracles, strict=True)) for v in x]
    )
    np.testing.assert_allclose(dist.pdf(x), pdf, rtol=2e-11)
    np.testing.assert_allclose(dist.cdf(x), cdf, rtol=1e-10, atol=1e-14)
    mean = sum(
        w * float(o.expect(lambda t: t)) for w, o in zip(weights, oracles, strict=True)
    )
    assert dist.mean == pytest.approx(mean, rel=1e-11, abs=1e-12)


def test_durable_normalizer_is_the_canonical_potential_s(fitted):
    """The stored ``log Z`` belongs to the canonical model, not the runtime form."""
    _, dist, oracle, _ = fitted
    stored = float(dist.data["model"]["log_z"][0])
    assert stored == pytest.approx(float(oracle.log_z), rel=1e-12, abs=1e-11)
