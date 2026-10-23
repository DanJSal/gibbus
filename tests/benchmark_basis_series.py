"""Run from repository root: python tests/benchmark_basis_series.py.

Median wall times; includes validation/allocation unless marked kernel.
Microbenchmark only, no timing assertions in CI.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_basis_evaluation import ORACLE_SPECS, _compile

from gibbus._model import _series_kernels
from gibbus._model.basis_eval import _potential
from gibbus._model.basis_series import SeriesPlan


def median_us(fn, loops=25):
    fn()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(loops):
            fn()
        samples.append((time.perf_counter_ns() - start) / loops / 1000)
    return float(np.median(samples))


def main():
    results = []
    for name, spec in [
        ("polynomial_gates", ORACLE_SPECS[10]),
        ("soft_power_gates", ORACLE_SPECS[14]),
        ("exponential_gates", ORACLE_SPECS[18]),
        ("bounded_logs", ORACLE_SPECS[8]),
    ]:
        model = _compile(spec)
        p = SeriesPlan(model)
        g = model.geometry
        theta = np.linspace(0.1, 1.0, model.n_coefficients)
        for N in (12, 24, 48):
            centers = np.linspace(
                g.center - 0.2 * g.scale, g.center + 0.2 * g.scale, 16
            )
            cache = p.cache(centers, N)
            s = cache.coefficients(theta)
            u = np.linspace(-0.3, 0.3, 10000)
            idx = np.zeros(u.size, dtype=np.intp)
            x = s.centers[0] + s.scales[0] * u
            row = dict(
                basis=name,
                order=N,
                centers=16,
                columns=model.n_coefficients,
                fallback_centers=cache.range_fallbacks,
            )
            row["build_16_centers_us"] = median_us(
                lambda: p.coefficients(theta, centers, N)
            )
            row["cache_16_centers_us"] = median_us(lambda: p.cache(centers, N))
            row["recombine_16_centers_us"] = median_us(
                lambda: cache.coefficients(theta)
            )
            row["evaluate_10000_public_us"] = median_us(lambda: s.evaluate_scaled(u))
            row["evaluate_10000_kernel_us"] = median_us(
                lambda: _series_kernels._evaluate(
                    s.mantissa, s.exponent, s._scaled, u, idx
                )
            )
            row["direct_10000_us"] = median_us(
                lambda: _potential(model, theta, x, max_order=4), loops=5
            )
            row["public_evaluation_speedup"] = (
                row["direct_10000_us"] / row["evaluate_10000_public_us"]
            )
            results.append(row)
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                numpy=np.__version__,
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
