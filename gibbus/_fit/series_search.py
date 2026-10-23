"""Fixed-geometry panel planning for the compiled Newton candidate search."""

import math
import operator
from collections import OrderedDict
from threading import RLock

import numpy as np

from .._model.basis_eval import _evaluate_basis
from ._series_search import SeriesSearchData, SeriesSearchKernel


class _SeriesTensorStore:
    """Bounded process-local LRU of immutable numeric tables, protected by GIL/lock.

    Construction happens outside the lock. Concurrent misses may construct
    duplicate tables; publication selects one snapshot without sharing mutable
    search state. Eviction cannot invalidate existing kernels' references.
    """

    def __init__(self, max_entries=32, max_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.lock = RLock()
        self.entries = OrderedDict()
        self.nbytes = 0
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                self.hits += 1
                return entry[0]
            self.misses += 1
        return None

    def publish(self, key, data):
        size = data.nbytes + len(key[0])
        with self.lock:
            existing = self.entries.get(key)
            if existing is not None:
                self.entries.move_to_end(key)
                return existing[0]
            if self.max_entries < 1 or size > self.max_bytes:
                return data
            while self.entries and (
                len(self.entries) >= self.max_entries
                or self.nbytes + size > self.max_bytes
            ):
                _, (_, old_size) = self.entries.popitem(last=False)
                self.nbytes -= old_size
                self.evictions += 1
            self.entries[key] = (data, size)
            self.nbytes += size
        return data

    def clear(self):
        with self.lock:
            self.entries.clear()
            self.nbytes = 0
            self.hits = self.misses = self.evictions = 0

    def info(self):
        with self.lock:
            return dict(
                entries=len(self.entries),
                bytes=self.nbytes,
                max_entries=self.max_entries,
                max_bytes=self.max_bytes,
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
            )


_TENSOR_STORE = _SeriesTensorStore()


def _clear_series_tensor_cache():
    """Release retained tensors; kernels already using them remain valid."""
    _TENSOR_STORE.clear()


def _series_tensor_cache_info():
    """Inspect process-local retention, lookup counts and byte budget."""
    return _TENSOR_STORE.info()


class _SeriesCandidateCache:
    """One cache per block/objective; reused across Newton trials and run restarts.

    Failure to build a finite cache disables only this heuristic. The caller
    retains the direct candidate search and the full curvature acceptance test.
    """

    def __init__(self, model, options):
        self.model = model
        self.options = options
        self._plan = None
        self.kernel = None
        self.domain = None
        self.builds = 0
        self.tensor_builds = 0
        self.tensor_reuses = 0
        self.reason = None

    @property
    def plan(self):
        if self._plan is None:
            self._plan = self.model.series_plan
        return self._plan

    def prepare(self, domain):
        a, b = map(float, domain)
        if self.domain == (a, b):
            return self.kernel
        # A previous larger domain still supplies valid global point constraints.
        if self.kernel is not None and self.domain[0] <= a < b <= self.domain[1]:
            return self.kernel
        self.domain = (a, b)
        self.kernel = None
        self.reason = None
        o = self.options
        g = self.model.geometry
        if not (math.isfinite(a) and math.isfinite(b) and a < b):
            self.reason = "nonfinite or empty compact domain"
            return None
        if isinstance(o.series_order, (bool, np.bool_)):
            raise TypeError("order must be an integer")
        order = operator.index(o.series_order)
        if order < 0 or order > 4096:
            raise ValueError("order must be between 0 and 4096")
        # Planner inputs and scan count determine the retained points and rows.
        # Refinement options belong only to the private workspace.
        samples = operator.index(o.series_samples)
        if samples < 2:
            raise ValueError("at least two scan subintervals are required")
        key = (self.model.series_signature, order, o.series_max_panels, samples, a, b)
        data = _TENSOR_STORE.get(key)
        if data is None:
            data = self._construct_data(a, b, order)
            if data is None:
                return None
            self.tensor_builds += 1
            data = _TENSOR_STORE.publish(key, data)
        else:
            self.tensor_reuses += 1
        self.kernel = SeriesSearchKernel(
            data,
            samples=o.series_samples,
            max_steps=o.series_steps,
            tolerance=o.series_tolerance * g.scale,
            threshold=o.rho / (g.scale * g.scale),
            capacity=o.initial_cells + 4,
        )
        self.builds += 1
        return self.kernel

    def _construct_data(self, a, b, order):
        o = self.options
        g = self.model.geometry
        # A numerical log amplitude can vanish while its structural column stays.
        # Leave a thin finite-end strip to the unchanged acceptance check.
        gap = 1e-8 * g.scale
        a = max(a, np.nextafter(self.model.request.lower, math.inf))
        b = min(b, np.nextafter(self.model.request.upper, -math.inf))
        for f in self.model.functions:
            if f.kind == "log" and any(self.model.coefficient_map_exact[3 * f.index]):
                if f.side == "lower":
                    a = max(a, self.model.request.lower + gap)
                else:
                    b = min(b, self.model.request.upper - gap)
        if not a < b:
            self.reason = "no interior panel"
            return None
        z = np.linspace(
            np.arcsinh((a - g.center) / g.scale),
            np.arcsinh((b - g.center) / g.scale),
            17,
        )
        edges = np.clip(g.center + g.scale * np.sinh(z), a, b)
        edges[0] = a
        edges[-1] = b
        extra = [
            v
            for v in (g.center, g.lower_location, g.upper_location)
            if v is not None and a < v < b
        ]
        edges = np.unique(np.r_[edges, extra])
        pending = list(zip(edges[:-1], edges[1:]))
        panels = []
        while pending:
            bounds = np.asarray(pending)
            centers = bounds[:, 0] + 0.5 * (bounds[:, 1] - bounds[:, 0])
            halves = 0.5 * (bounds[:, 1] - bounds[:, 0])
            if np.any(centers <= bounds[:, 0]) or np.any(centers >= bounds[:, 1]):
                self.reason = "unrepresentable panel"
                return None
            try:
                _, _, radius, _ = self.plan._centers(centers, order, None)
            except (ValueError, OverflowError, FloatingPointError):
                self.reason = "unsupported panel geometry"
                return None
            pending = []
            for (lo, hi), center, half, r in zip(bounds, centers, halves, radius):
                if half > 0.35 * r:
                    pending.extend(((lo, center), (center, hi)))
                else:
                    panels.append((center, half))
            if len(pending) + len(panels) > o.series_max_panels:
                self.reason = "panel budget"
                return None
        panels = np.array(sorted(panels))
        # This tensor folds the exact rational sign maps into basis directions
        # once. Every subsequent trial uses one compiled matrix contraction.
        try:
            cache = self.plan.cache(panels[:, 0], order, scale=panels[:, 1])
            if cache.range_fallbacks:
                self.reason = "extended-range primitive cache"
                return None
            d = self.model.n_coefficients
            tensor = np.empty((len(panels), 3, order + 1, d))
            unit = np.zeros(d)
            for j in range(d):
                unit[j] = 1.0
                tensor[:, :, :, j] = cache.coefficients(unit).coefficients
                unit[j] = 0.0
            scan_points = panels[:, 0, None] + panels[:, 1, None] * np.linspace(
                -1.0, 1.0, o.series_samples + 1
            )
            scan_points = np.clip(scan_points, a, b)
            basis = _evaluate_basis(self.model, scan_points.ravel(), max_order=3)
            shape = (len(panels), o.series_samples + 1, d)
            scan_rows = np.stack(
                (basis.d2phi.reshape(shape), basis.d3phi.reshape(shape)), axis=1
            )
            data = SeriesSearchData(
                panels,
                tensor,
                samples=o.series_samples,
                scan_points=scan_points,
                scan_rows=scan_rows,
            )
        except (ValueError, OverflowError, FloatingPointError):
            self.reason = "unrepresentable cached coefficients"
            return None
        return data
