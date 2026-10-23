"""Compare full fits with direct versus cached-series candidate generation.

Run from source/: python tests/benchmark_series_search.py
Includes cache construction and all unchanged acceptance checks.
"""

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_basis_newton import TAILS, _objective, _poly

from gibbus._fit.basis_newton import _FaceObjective, _solve_basis
from gibbus._fit.curvature_search import _CurvatureOptions
from gibbus._fit.series_search import _clear_series_tensor_cache


def fit(name, enabled):
    objective = _objective(*TAILS[name]) if name in TAILS else _poly(name)[0]
    seen = {}
    original = _FaceObjective.compiled

    def capture(self):
        obj = original(self)
        seen[id(obj)] = obj
        return obj

    _FaceObjective.compiled = capture
    try:
        t = time.perf_counter()
        result = _solve_basis(
            objective, curvature_options=_CurvatureOptions(series_candidates=enabled)
        )
        seconds = time.perf_counter() - t
    finally:
        _FaceObjective.compiled = original
    stats = [o.series_statistics() for o in seen.values()]
    blocks = [b for s in stats for b in s["blocks"] if b is not None]
    return (
        dict(
            seconds=seconds,
            status=result.status,
            nll=result.objective_value,
            iterations=result.newton_iterations,
            evaluations=result.objective_evaluations,
            exchanges=result.exchanges,
            certification=result.final_separation.status,
            cache_builds=sum(b["builds"] for b in blocks),
            tensor_builds=sum(b["tensor_builds"] for b in blocks),
            tensor_reuses=sum(b["tensor_reuses"] for b in blocks),
            candidate_calls=sum(b["kernel"]["calls"] for b in blocks if b["kernel"]),
            hits=sum(s["hits"] for s in stats),
            full_searches=sum(s["full_searches"] for s in stats),
            fallback_reasons=[b["reason"] for b in blocks if b["reason"]],
        ),
        result.params,
    )


def gate_benchmark():
    from test_series_search import _theta_for, make_face

    results = {}
    for enabled in (False, True):
        model, obj, _, domain = make_face(enabled)
        # Changing vectors avoid measuring exact-theta memoization.
        trials = [_theta_for(model, -0.01 - 0.00001 * i) for i in range(100)]
        times = []
        for _ in range(5):
            start = time.perf_counter_ns()
            for theta in trials:
                obj.trial(theta)
            times.append((time.perf_counter_ns() - start) / len(trials) / 1000)
        results["series" if enabled else "direct"] = dict(
            median_microseconds_per_rejected_trial=float(np.median(times)),
            statistics=obj.series_statistics(),
        )
    results["speedup"] = (
        results["direct"]["median_microseconds_per_rejected_trial"]
        / results["series"]["median_microseconds_per_rejected_trial"]
    )
    return results


def main():
    results = []
    for name in ("line-b2", "line-b4-gumbel", "half-b2-log", "student-softpower"):
        runs = []
        for repetition in range(7):
            pair = {}
            params = {}
            for enabled in ((False, True) if repetition % 2 == 0 else (True, False)):
                key = "series" if enabled else "direct"
                if enabled:
                    _clear_series_tensor_cache()
                pair[key], params[key] = fit(name, enabled)
            pair["nll_difference"] = pair["series"]["nll"] - pair["direct"]["nll"]
            pair["parameter_max_difference"] = float(
                np.max(np.abs(params["series"] - params["direct"]))
            )
            runs.append(pair)
        results.append(
            dict(
                case=name,
                runs=runs,
                speedup=float(
                    np.median([r["direct"]["seconds"] for r in runs])
                    / np.median([r["series"]["seconds"] for r in runs])
                ),
            )
        )
        print(json.dumps(results[-1]), file=sys.stderr, flush=True)
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                results=results,
                rejected_trial_gate=gate_benchmark(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
