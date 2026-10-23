"""Cold versus retained tensors across newly constructed matching fit models.

Run from source/: python tests/benchmark_series_tensor_reuse.py
All fit timings include private workspace allocation and full acceptance checks.
"""

import json
import platform
import sys
import time

import numpy as np
from benchmark_series_search import fit

from gibbus._fit.series_search import (
    _clear_series_tensor_cache,
    _series_tensor_cache_info,
    _SeriesCandidateCache,
)


def measured_fit(name, enabled):
    original = _SeriesCandidateCache.prepare
    setup_ns = 0

    def timed(self, domain):
        nonlocal setup_ns
        start = time.perf_counter_ns()
        try:
            return original(self, domain)
        finally:
            setup_ns += time.perf_counter_ns() - start

    _SeriesCandidateCache.prepare = timed
    try:
        result, params = fit(name, enabled)
    finally:
        _SeriesCandidateCache.prepare = original
    result["series_setup_seconds"] = setup_ns / 1e9
    return result, params


def main():
    results = []
    for name in ("line-b2", "line-b4-gumbel", "half-b2-log", "student-softpower"):
        runs = []
        for repetition in range(9):
            pair = {}
            params = {}
            if repetition % 2 == 0:
                pair["direct"], params["direct"] = measured_fit(name, False)
            _clear_series_tensor_cache()
            pair["cold"], params["cold"] = measured_fit(name, True)
            pair["retained"], params["retained"] = measured_fit(name, True)
            if repetition % 2:
                pair["direct"], params["direct"] = measured_fit(name, False)
            pair["cold_retained_parameter_difference"] = float(
                np.max(np.abs(params["cold"] - params["retained"]))
            )
            pair["cold_retained_nll_difference"] = (
                pair["cold"]["nll"] - pair["retained"]["nll"]
            )
            pair["store"] = _series_tensor_cache_info()
            runs.append(pair)
        medians = {
            k: float(np.median([r[k]["seconds"] for r in runs]))
            for k in ("direct", "cold", "retained")
        }
        setup = {
            k: float(np.median([r[k]["series_setup_seconds"] for r in runs]))
            for k in ("cold", "retained")
        }
        row = dict(
            case=name,
            runs=runs,
            median_fit_seconds=medians,
            median_setup_seconds=setup,
            retained_speedup_over_cold=medians["cold"] / medians["retained"],
            retained_speedup_over_direct=medians["direct"] / medians["retained"],
        )
        results.append(row)
        print(json.dumps(row), file=sys.stderr, flush=True)
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
