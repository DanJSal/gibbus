"""Deterministic randomized interval cases for solver property tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_GEOMETRIES = ("real", "lower_half", "upper_half", "bounded")


@dataclass(frozen=True)
class IntervalSolverCase:
    """One deterministic randomized interval-fitting problem."""

    seed: int
    geometry: str
    support: tuple[float, float]
    rows: np.ndarray
    weights: np.ndarray | None
    degree: int
    lower_boundary: bool
    upper_boundary: bool

    @property
    def label(self) -> str:
        """Return a compact replay label for assertion messages."""
        return (
            f"seed={self.seed} geometry={self.geometry} degree={self.degree} "
            f"boundaries=({int(self.lower_boundary)},{int(self.upper_boundary)}) "
            f"rows={self.rows.shape[0]} unique={np.unique(self.rows, axis=0).shape[0]} "
            f"weighted={self.weights is not None}"
        )


def _support_for_geometry(geometry, rng, /):
    if geometry == "real":
        return (-np.inf, np.inf)
    if geometry == "lower_half":
        return (float(rng.uniform(-1.0, 1.0)), np.inf)
    if geometry == "upper_half":
        return (-np.inf, float(rng.uniform(-1.0, 1.0)))
    if geometry == "bounded":
        lower = float(rng.uniform(-2.0, 0.0))
        return (lower, lower + float(rng.uniform(1.0, 4.0)))
    raise ValueError(f"unknown geometry {geometry!r}")


def _latent_sample(support, n, rng, /):
    lower, upper = support
    if np.isneginf(lower) and np.isposinf(upper):
        return rng.normal(
            float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0.6, 1.5)), n
        )
    if np.isfinite(lower) and np.isposinf(upper):
        return lower + rng.gamma(
            float(rng.uniform(1.4, 3.5)), float(rng.uniform(0.3, 1.1)), n
        )
    if np.isneginf(lower) and np.isfinite(upper):
        return upper - rng.gamma(
            float(rng.uniform(1.4, 3.5)), float(rng.uniform(0.3, 1.1)), n
        )
    width = upper - lower
    return lower + width * rng.beta(
        float(rng.uniform(1.2, 4.0)), float(rng.uniform(1.2, 4.0)), n
    )


def _finite_scale(x, support, /):
    lower, upper = support
    if np.isfinite(lower) and np.isfinite(upper):
        return float(upper - lower)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return max(float(q75 - q25), float(np.std(x)), 0.25)


def _make_rows(x, support, rng, /):
    lower, upper = support
    scale = _finite_scale(x, support)
    rows = []
    for value in x:
        kind = int(rng.integers(0, 8))
        width = float(rng.uniform(0.015, 0.16) * scale)
        if kind == 0:
            lo = hi = float(value)
        elif kind in (1, 2, 3):
            left = float(rng.uniform(0.05, 0.95) * width)
            lo, hi = float(value - left), float(value + width - left)
            if np.isfinite(lower):
                lo = max(lo, float(lower))
            if np.isfinite(upper):
                hi = min(hi, float(upper))
        elif kind == 4 and np.isfinite(lower):
            lo = float(lower)
            hi = (
                min(float(value + width), float(upper))
                if np.isfinite(upper)
                else float(value + width)
            )
        elif kind == 5 and np.isfinite(upper):
            lo = (
                max(float(value - width), float(lower))
                if np.isfinite(lower)
                else float(value - width)
            )
            hi = float(upper)
        elif kind == 6 and np.isneginf(lower):
            lo, hi = -np.inf, float(value)
        elif kind == 7 and np.isposinf(upper):
            lo, hi = float(value), np.inf
        else:
            lo, hi = float(value - width), float(value + width)
            if np.isfinite(lower):
                lo = max(lo, float(lower))
            if np.isfinite(upper):
                hi = min(hi, float(upper))
        rows.append((lo, hi))

    if rng.random() < 0.55:
        rows.append((float(lower), float(upper)))
    rows = np.asarray(rows, dtype=np.float64)
    duplicate = rows[rng.integers(0, rows.shape[0], size=max(2, rows.shape[0] // 6))]
    rows = np.vstack([rows, duplicate])
    rng.shuffle(rows, axis=0)
    return rows


def make_case(seed, geometry=None, /):
    """Generate one replayable randomized interval case."""
    seed = int(seed)
    rng = np.random.default_rng(seed)
    if geometry is None:
        geometry = _GEOMETRIES[seed % len(_GEOMETRIES)]
    support = _support_for_geometry(geometry, rng)
    x = _latent_sample(support, int(rng.integers(28, 55)), rng)
    rows = _make_rows(x, support, rng)
    degree = int(rng.choice(np.array([2, 4, 6], dtype=np.int64), p=[0.35, 0.45, 0.20]))
    lower_boundary = bool(np.isfinite(support[0]) and rng.random() < 0.65)
    upper_boundary = bool(np.isfinite(support[1]) and rng.random() < 0.65)
    weights = None
    if rng.random() < 0.70:
        weights = rng.lognormal(mean=0.0, sigma=0.65, size=rows.shape[0])
        if rows.shape[0] >= 12:
            zero = rng.choice(
                rows.shape[0], size=max(1, rows.shape[0] // 15), replace=False
            )
            weights[zero] = 0.0
    return IntervalSolverCase(
        seed=seed,
        geometry=geometry,
        support=tuple(map(float, support)),
        rows=np.ascontiguousarray(rows),
        weights=None if weights is None else np.ascontiguousarray(weights),
        degree=degree,
        lower_boundary=lower_boundary,
        upper_boundary=upper_boundary,
    )


def cdf_grid(case, n=41, /):
    """Return a finite grid spanning the informative observations."""
    finite = case.rows[np.isfinite(case.rows)]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = max(hi - lo, 1e-3)
    lower, upper = case.support
    lo = max(lo - 0.2 * span, lower) if np.isfinite(lower) else lo - 0.35 * span
    hi = min(hi + 0.2 * span, upper) if np.isfinite(upper) else hi + 0.35 * span
    if not lo < hi:
        hi = np.nextafter(lo, np.inf)
    return np.linspace(lo, hi, int(n), dtype=np.float64)
