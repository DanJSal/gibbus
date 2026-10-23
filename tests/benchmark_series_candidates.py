"""Candidate speed and discovery against dense-grid/bracketed references.

python tests/benchmark_series_candidates.py [--baseline-source PREVIOUS/source]
The optional previous source must have its extensions already built. Its candidate
module is loaded independently; no old table is passed into a new Newton kernel.
"""

import argparse
import importlib.util
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gibbus._fit._series_search import SeriesSearchKernel
from test_basis_evaluation import ORACLE_SPECS, _compile
from test_series_search import clustered_history, clustered_tensor

from gibbus._fit.curvature_search import _CurvatureOptions
from gibbus._fit.series_search import _SeriesCandidateCache
from gibbus._model.basis_eval import _evaluate_basis, _potential


def load_baseline(source):
    source = Path(source).resolve()
    directory = source / "gibbus/_fit"
    paths = list(directory.glob("_series_search*.so")) + list(
        directory.glob("_series_search*.pyd")
    )
    if len(paths) != 1:
        raise ValueError("baseline source needs one built _series_search extension")
    package = types.ModuleType("candidate_baseline")
    package.__path__ = []
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "candidate_baseline._series_search", paths[0]
    )
    extension = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extension
    spec.loader.exec_module(extension)
    module = types.ModuleType("gibbus._fit._candidate_baseline")
    module.__package__ = "gibbus._fit"
    sys.modules[module.__name__] = module
    path = directory / "series_search.py"
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    module.SeriesSearchData = extension.SeriesSearchData
    module.SeriesSearchKernel = extension.SeriesSearchKernel
    return module._SeriesCandidateCache, extension.SeriesSearchKernel


def references(model, theta, grid, d3):
    h = model.geometry.scale
    crossings = np.flatnonzero(
        (d3[:-1] <= 0) & (d3[1:] >= 0) & ((d3[:-1] < 0) | (d3[1:] > 0))
    )
    roots = []
    for i in crossings:
        root = brentq(
            lambda x: float(_potential(model, theta, [x], max_order=3)[3][0]),
            grid[i],
            grid[i + 1],
            xtol=1e-11 * h,
            rtol=1e-14,
        )
        if roots and abs(root - roots[-1]) < 1e-7 * h:
            continue
        value = float(_potential(model, theta, [root], max_order=2)[2][0])
        if value < -1e-4 / (h * h):
            roots.append(root)
    return roots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source")
    args = parser.parse_args()
    factories = {"staged": _SeriesCandidateCache}
    classes = {"staged": SeriesSearchKernel}
    if args.baseline_source:
        factories["previous"], classes["previous"] = load_baseline(args.baseline_source)
    results = []
    for index in (0, 2, 8, 14, 18, 22):
        model = _compile(ORACLE_SPECS[index])
        g = model.geometry
        h = g.scale
        domain = (
            max(model.request.lower + 0.001 * h, g.center - 4 * h),
            min(model.request.upper - 0.001 * h, g.center + 4 * h),
        )
        kernels = {
            name: factory(model, _CurvatureOptions(series_max_panels=256)).prepare(
                domain
            )
            for name, factory in factories.items()
        }
        if any(k is None for k in kernels.values()):
            raise RuntimeError("unsupported benchmark geometry")
        rng = np.random.default_rng(3928 + index)
        vectors = rng.normal(size=(30, model.n_coefficients))
        for j, column in enumerate(model.columns):
            if column.nonnegative:
                vectors[:, j] = abs(vectors[:, j])
        grid = np.linspace(*domain, 4097)
        d3 = _evaluate_basis(model, grid, max_order=3).d3phi
        expected = [references(model, theta, grid, d3 @ theta) for theta in vectors]
        discovery = {}
        for name, kernel in kernels.items():
            hits = 0
            trials_with_negative = 0
            trials_detected = 0
            errors = []
            for theta, roots in zip(vectors, expected):
                points = kernel.candidates(theta)
                if points.size:
                    values = _potential(model, theta, points, max_order=2)[2]
                    points = points[values < 0.0]
                for root in roots:
                    error = (
                        float(np.min(abs(points - root)) / h)
                        if points.size
                        else float("inf")
                    )
                    errors.append(error)
                    hits += error < 1e-3
                if roots:
                    trials_with_negative += 1
                    trials_detected += bool(points.size)
            timings = []
            # Changing theta avoids exact-theta replay in the measured stream.
            streams = [vectors * (1.0 + 1e-6 * (r + 1)) for r in range(7)]
            for stream in streams:
                start = time.perf_counter_ns()
                for theta in stream:
                    kernel.candidates(theta)
                timings.append((time.perf_counter_ns() - start) / len(vectors) / 1000)
            discovery[name] = dict(
                median_microseconds=float(np.median(timings)),
                minima_found=hits,
                reference_minima=sum(map(len, expected)),
                trials_with_negative_minima=trials_with_negative,
                trials_with_negative_candidate=trials_detected,
                max_location_error_in_h=max(errors) if errors else None,
                p95_location_error_in_h=(
                    float(np.quantile(errors, 0.95)) if errors else None
                ),
                statistics=kernel.statistics(),
            )
        results.append(dict(spec_index=index, draws=30, discovery=discovery))
    clustered = {}
    for name, klass in classes.items():
        kernel = klass([[0.0, 1.0]], clustered_tensor(), tolerance=1e-7)
        history = clustered_history(kernel)
        clustered[name] = dict(
            history=history,
            minima_found=sum(r["hits"] for r in history),
            reference_minima=3 * len(history),
            statistics=kernel.statistics(),
        )
    print(
        json.dumps(
            dict(
                python=platform.python_version(),
                platform=platform.platform(),
                results=results,
                clustered=clustered,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
