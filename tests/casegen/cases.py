"""Lazy test-case universe, coherent observation mechanisms, balanced sampling.

No fitter imports are needed until FitSpec.fit_kwargs() is explicitly called.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np

from .distributions import (
    DistributionSpec,
    Mixture,
    category,
    sample,
    smoothness,
    support,
    to_dict,
    validate,
)

SCHEMA_VERSION = 1
TAIL_NAMES = ("none", "poly1", "poly2", "poly3", "soft1.5", "soft2.5", "exp1", "exp2")
OBSERVATION_MODES = (
    "points",
    "weighted",
    "left",
    "right",
    "interval",
    "rounded",
    "binned",
    "mixed",
)


def _canonical(payload) -> str:
    def clean(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return "+inf" if obj > 0 else "-inf"
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [clean(v) for v in obj]
        return obj

    return json.dumps(
        clean(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _id(prefix: str, payload) -> str:
    return (
        f"{prefix}-"
        + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[:20]
    )


def support_kind(bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    if lo == -math.inf and hi == math.inf:
        return "full"
    if lo != -math.inf and hi == math.inf:
        return "lower"
    if lo == -math.inf and hi != math.inf:
        return "upper"
    return "bounded"


@dataclass(frozen=True)
class FitSpec:
    """One of nine support/log categories, with basis and K controls."""

    log_lower: bool = False
    log_upper: bool = False
    degree: int | str = 2
    tail: str = "none"
    expansion: str = "minimal"
    n_components: int | str = 1

    def validate(self, bounds):
        kind = support_kind(bounds)
        if (self.log_lower and bounds[0] == -math.inf) or (
            self.log_upper and bounds[1] == math.inf
        ):
            raise ValueError("endpoint logarithm requested at an infinite boundary")
        if self.degree != "auto" and (
            not isinstance(self.degree, int) or self.degree < 2
        ):
            raise ValueError("degree must be >=2 or 'auto'")
        if self.n_components != "auto" and (
            not isinstance(self.n_components, int) or self.n_components < 1
        ):
            raise ValueError("component count must be positive or 'auto'")
        if self.tail not in TAIL_NAMES or (kind == "bounded" and self.tail != "none"):
            raise ValueError("invalid tail specification for support")
        if self.expansion not in ("minimal", "full"):
            raise ValueError("expansion must be minimal or full")
        if self.degree == "auto" and (
            self.tail != "none" or self.expansion != "minimal"
        ):
            raise ValueError(
                "automatic degree cannot also specify tails or full expansion"
            )
        return kind

    def fit_category(self, bounds):
        self.validate(bounds)
        k = support_kind(bounds)
        codes = {"full": "S01", "lower": "S02", "upper": "S04", "bounded": "S06"}
        if k == "lower" and self.log_lower:
            return "S03"
        if k == "upper" and self.log_upper:
            return "S05"
        if k == "bounded":
            return {
                (False, False): "S06",
                (True, False): "S07",
                (False, True): "S08",
                (True, True): "S09",
            }[(self.log_lower, self.log_upper)]
        return codes[k]

    def fit_kwargs(self, bounds: tuple[float, float]) -> dict:
        """Convert to the public Distribution.fit arguments (no fit performed)."""
        self.validate(bounds)
        from gibbus import BasisSpec, Exponential, Polynomial, SoftPower

        tail = None
        if self.tail.startswith("poly"):
            tail = Polynomial(int(self.tail[4:]))
        elif self.tail.startswith("soft"):
            tail = SoftPower(float(self.tail[4:]))
        elif self.tail.startswith("exp"):
            tail = Exponential(float(self.tail[3:]))
        return {
            "n_components": self.n_components,
            "basis": (
                BasisSpec(
                    support=bounds,
                    bulk_degree=self.degree if self.degree != "auto" else 2,
                    log_lower=self.log_lower,
                    log_upper=self.log_upper,
                    lower_tail=tail if bounds[0] == -math.inf else None,
                    upper_tail=tail if bounds[1] == math.inf else None,
                    expansion=self.expansion,
                )
                if self.degree != "auto"
                else None
            ),
            **(
                {
                    "support": bounds,
                    "poly_degree": "auto",
                    "log_boundary_lower": self.log_lower,
                    "log_boundary_upper": self.log_upper,
                }
                if self.degree == "auto"
                else {}
            ),
        }

    def as_dict(self):
        return vars(self).copy()


@dataclass(frozen=True)
class ObservationSpec:
    mode: str = "points"
    fraction: float = 0.3
    width: float = 0.2

    def validate(self):
        if self.mode not in OBSERVATION_MODES:
            raise ValueError(f"unknown observation mode: {self.mode}")
        if not (math.isfinite(self.fraction) and 0 <= self.fraction <= 0.95):
            raise ValueError("fraction must be in [0,0.95]")
        if not (math.isfinite(self.width) and self.width > 0):
            raise ValueError("width must be positive")

    def as_dict(self):
        return vars(self).copy()


def _family_group(spec):
    from .distributions import Affine, Mixture, Named, Sum, Truncated

    if isinstance(spec, Named):
        return spec.family
    if isinstance(spec, (Affine, Truncated)):
        return _family_group(spec.source)
    if isinstance(spec, Sum):
        return "sum:" + "+".join(
            sorted((_family_group(spec.left), _family_group(spec.right)))
        )
    if isinstance(spec, Mixture):
        return "mixture:" + "+".join(
            sorted(_family_group(c) for _, c in spec.components)
        )
    raise TypeError(type(spec))


@dataclass(frozen=True)
class CaseSpec:
    distribution: DistributionSpec
    observation: ObservationSpec
    fit: FitSpec
    n: int
    replicate: int = 0

    def validate(self):
        validate(self.distribution)
        self.observation.validate()
        self.fit.validate(support(self.distribution))
        if not isinstance(self.n, int) or self.n < 2 or self.replicate < 0:
            raise ValueError("n must be >=2 and replicate must be nonnegative")

    def as_dict(self, include_replicate=True):
        d = {
            "schema": SCHEMA_VERSION,
            "distribution": to_dict(self.distribution),
            "observation": self.observation.as_dict(),
            "fit": self.fit.as_dict(),
            "n": self.n,
        }
        if include_replicate:
            d["replicate"] = self.replicate
        return d

    @property
    def scenario_id(self):
        return _id("GBS", self.as_dict(False))

    @property
    def case_id(self):
        return _id("GBC", self.as_dict(True))

    def strata(self) -> dict[str, str]:
        lo, hi = support(self.distribution)
        d = {
            "composition": category(self.distribution),
            "support": support_kind((lo, hi)),
            "fit_category": self.fit.fit_category((lo, hi)),
            "family": _family_group(self.distribution),
            "observation": self.observation.mode,
            "censoring": (
                "none"
                if self.observation.mode in ("points", "weighted")
                else (
                    "light"
                    if self.observation.fraction < 0.2
                    else "moderate" if self.observation.fraction < 0.6 else "heavy"
                )
            ),
            "sample_size": str(self.n),
            "degree": str(self.fit.degree),
            "tail": self.fit.tail,
            "components": str(self.fit.n_components),
            "smoothness": smoothness(self.distribution),
            "component_alignment": (
                "underfit"
                if isinstance(self.distribution, Mixture)
                and isinstance(self.fit.n_components, int)
                and self.fit.n_components < len(self.distribution.components)
                else (
                    "overfit"
                    if isinstance(self.distribution, Mixture)
                    and isinstance(self.fit.n_components, int)
                    and self.fit.n_components > len(self.distribution.components)
                    else "matched_or_selected"
                )
            ),
        }
        return d


@dataclass(frozen=True)
class GeneratedCase:
    spec: CaseSpec
    latent: np.ndarray
    observations: np.ndarray
    weights: np.ndarray | None
    censored: np.ndarray
    seed: int

    @property
    def fit_kwargs(self):
        kw = self.spec.fit.fit_kwargs(support(self.spec.distribution))
        if self.weights is not None:
            kw["sample_weight"] = self.weights
        return kw


def materialize(spec: CaseSpec, *, seed: int = 0) -> GeneratedCase:
    """Generate observations independently of fit code; stable for (case ID, seed)."""
    spec.validate()
    nonce = int.from_bytes(
        hashlib.sha256((spec.case_id + ":" + str(seed)).encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(nonce)
    latent = sample(spec.distribution, spec.n, rng)
    if not np.all(np.isfinite(latent)):
        raise ValueError("sample overflow or nonfinite latent observation")
    bounds = support(spec.distribution)
    lo, hi = bounds
    if np.any(latent < lo) or np.any(latent > hi):
        raise ValueError("sampler produced observation outside its support")
    o = spec.observation
    if o.mode in ("points", "weighted"):
        weights = None if o.mode == "points" else rng.lognormal(0, 0.5, size=spec.n)
        if weights is not None:
            weights /= weights.mean()
        return GeneratedCase(
            spec,
            latent.copy(),
            latent.copy(),
            weights,
            np.zeros(spec.n, dtype=bool),
            nonce,
        )
    # Independent pilot used to set fixed measurement resolution/thresholds.
    pilot_rng = np.random.default_rng(nonce ^ 0xC0A5E52B724D8E19)
    pilot = sample(spec.distribution, 4096, pilot_rng)
    q25, q50, q75 = np.quantile(pilot, [0.25, 0.5, 0.75])
    width = max(float(q75 - q25) * o.width, np.spacing(float(q50)) * 8, 1e-300)
    step = width * (3.0 if o.mode == "binned" else 1.0)
    low_cut = float(np.quantile(pilot, o.fraction))
    high_cut = float(np.quantile(pilot, 1 - o.fraction))
    # The phase is selected independently of the latent observations.
    origin = float(q50) + (rng.uniform(-0.5, 0.5) * step)
    intervals = np.column_stack([latent.copy(), latent.copy()])
    censored = np.zeros(spec.n, dtype=bool)
    assigned = np.full(spec.n, o.mode, dtype="<U12")
    if o.mode == "mixed":
        assigned = rng.choice(["left", "right", "interval", "rounded"], size=spec.n)
        assigned = np.where(rng.random(spec.n) < o.fraction, assigned, "points")
    for kind in ("left", "right", "interval", "rounded", "binned"):
        rows = np.flatnonzero(assigned == kind)
        if not len(rows):
            continue
        x = latent[rows]
        if kind == "left":
            m = x <= low_cut
            intervals[rows[m]] = (lo, low_cut)
        elif kind == "right":
            m = x >= high_cut
            intervals[rows[m]] = (high_cut, hi)
        else:
            w = step if kind != "rounded" else width * 0.5
            # Use fixed global bins; never center an interval at the latent x.
            bucket = np.floor((x - origin) / w)
            lower = np.maximum(lo, origin + bucket * w)
            h = np.minimum(hi, origin + (bucket + 1) * w)
            # Include floating-point boundary errors by using neighboring ULPs.
            lower = np.maximum(lo, np.minimum(lower, x))
            h = np.minimum(hi, np.maximum(h, x))
            intervals[rows, 0], intervals[rows, 1] = lower, h
    censored = intervals[:, 0] < intervals[:, 1]
    if not np.all((intervals[:, 0] <= latent) & (latent <= intervals[:, 1])):
        raise AssertionError("coarsening interval omitted its latent observation")
    if not np.all((intervals[:, 0] >= lo) & (intervals[:, 1] <= hi)):
        raise AssertionError("coarsening interval outside support")
    return GeneratedCase(spec, latent, intervals, None, censored, nonce)


class CaseUniverse:
    """Finite logical product of eligible distributions and test dimensions.

    ``iter_cases`` is exhaustive and lazy. ``select`` is bounded and uses
    distribution/support-aware proposals, so it does not enumerate the product.
    """

    def __init__(
        self,
        distributions,
        observations=None,
        degrees=(2, 4, 6),
        tails=("none", "poly2", "soft1.5"),
        sample_sizes=(100, 500, 2000),
        component_counts=(1, 2, 3, "auto"),
        expansions=("minimal",),
        replicates=(0,),
    ):
        self.distributions = tuple(distributions)
        self.observations = tuple(
            observations
            or (
                ObservationSpec("points"),
                ObservationSpec("weighted"),
                ObservationSpec("left", 0.2),
                ObservationSpec("right", 0.2),
                ObservationSpec("interval", 0.4),
                ObservationSpec("rounded", 0.3),
                ObservationSpec("binned", 0.5),
                ObservationSpec("mixed", 0.5),
            )
        )
        self.degrees = tuple(degrees)
        self.tails = tuple(tails)
        self.sample_sizes = tuple(sample_sizes)
        self.component_counts = tuple(component_counts)
        self.expansions = tuple(expansions)
        self.replicates = tuple(replicates)
        if not self.distributions or not self.observations or not self.sample_sizes:
            raise ValueError("case-universe axes must be nonempty")
        for d in self.distributions:
            validate(d)
        for o in self.observations:
            o.validate()
        for n in self.sample_sizes:
            if not isinstance(n, int) or n < 2:
                raise ValueError("invalid sample size")
        for r in self.replicates:
            if not isinstance(r, int) or r < 0:
                raise ValueError("invalid replicate")
        self._by_kind: dict[str, list[DistributionSpec]] = {}
        self._by_kind_and_structure: dict[tuple[str, str], list[DistributionSpec]] = {}
        for d in self.distributions:
            kind = support_kind(support(d))
            self._by_kind.setdefault(kind, []).append(d)
            self._by_kind_and_structure.setdefault((kind, category(d)), []).append(d)

    def _fits(self, d) -> Iterator[FitSpec]:
        lo, hi = support(d)
        lower = (False, True) if math.isfinite(lo) else (False,)
        upper = (False, True) if math.isfinite(hi) else (False,)
        tails = ("none",) if all(map(math.isfinite, (lo, hi))) else self.tails
        counts = (
            self.component_counts
            if isinstance(d, Mixture)
            else tuple(k for k in self.component_counts if k in (1, "auto"))
        )
        for ll, lu, degree, tail, expansion, count in product(
            lower, upper, self.degrees, tails, self.expansions, counts
        ):
            f = FitSpec(ll, lu, degree, tail, expansion, count)
            f.validate((lo, hi))
            yield f

    def iter_cases(self) -> Iterator[CaseSpec]:
        for dist in self.distributions:
            for fit, obs, n, rep in product(
                self._fits(dist), self.observations, self.sample_sizes, self.replicates
            ):
                yield CaseSpec(dist, obs, fit, n, rep)

    def select(
        self,
        count: int,
        *,
        seed: int = 42,
        candidate_factor: int = 25,
        fixed=(),
        balance_keys=(
            "fit_category",
            "composition",
            "observation",
            "sample_size",
            "degree",
            "support",
            "family",
        ),
        pairwise=True,
        explore_fraction=0.15,
        coverage_counts=None,
        exclude_case_ids=(),
    ) -> list[CaseSpec]:
        """Greedy balanced coverage on reproducible stratified randomized proposals.

        Fixed scenarios are always selected. Other proposals first choose one
        of four support kinds uniformly, then a distribution of that kind,
        avoiding catalog-size bias. All proposed cases are valid by construction.
        """
        if not isinstance(count, int) or count < 1:
            raise ValueError("count must be positive")
        if candidate_factor < 1:
            raise ValueError("candidate_factor must be positive")
        if not 0 <= explore_fraction < 1:
            raise ValueError("explore_fraction must be in [0,1)")
        rng = random.Random(seed)
        pool: dict[str, CaseSpec] = {}
        selected = []
        excluded = set(exclude_case_ids)
        for c in fixed:
            c.validate()
            if c.case_id not in pool and c.case_id not in excluded:
                pool[c.case_id] = c
                selected.append(c)
        if len(selected) > count:
            raise ValueError("fixed cases exceed selection budget")
        cat_to_kind = {
            "S01": "full",
            "S02": "lower",
            "S03": "lower",
            "S04": "upper",
            "S05": "upper",
            "S06": "bounded",
            "S07": "bounded",
            "S08": "bounded",
            "S09": "bounded",
        }
        fit_categories = [c for c, k in cat_to_kind.items() if k in self._by_kind]
        targets = max(count * candidate_factor, count + 100, len(fit_categories) * 100)
        trials = 0
        while len(pool) < targets and trials < targets * 20:
            trials += 1
            target_category = (
                fit_categories[(trials - 1) % len(fit_categories)]
                if trials <= len(fit_categories) * 40
                else rng.choice(fit_categories)
            )
            k = cat_to_kind[target_category]
            structures = sorted(
                c for (kind, c) in self._by_kind_and_structure if kind == k
            )
            structure = rng.choice(structures)
            d = rng.choice(self._by_kind_and_structure[(k, structure)])
            # Balance fit categories before distributing across families.
            fits = [
                f
                for f in self._fits(d)
                if f.fit_category(support(d)) == target_category
            ]
            f = rng.choice(fits)
            c = CaseSpec(
                d,
                rng.choice(self.observations),
                f,
                rng.choice(self.sample_sizes),
                rng.choice(self.replicates),
            )
            if c.case_id not in excluded:
                pool[c.case_id] = c
        pool_ids = [k for k in pool if k not in {c.case_id for c in selected}]
        rng.shuffle(pool_ids)
        candidates = [pool[k] for k in pool_ids]
        if len(pool) < count:
            raise ValueError("not enough distinct eligible cases")
        counts: dict[str, int] = dict(coverage_counts or {})

        def features(c):
            s = c.strata()
            single = [("s", k, s[k]) for k in balance_keys]
            pair_keys = [k for k in balance_keys if k != "family"]
            pairs = (
                [
                    ("p", k, j, s[k], s[j])
                    for i, k in enumerate(pair_keys)
                    for j in pair_keys[i + 1 :]
                ]
                if pairwise
                else []
            )
            return single + pairs

        for c in selected:
            for feature in features(c):
                key = _canonical(feature)
                counts[key] = counts.get(key, 0) + 1
        # Greedy maximum incremental coverage, with deterministic randomized
        # tie order. More weight for unobserved pairs and rare single strata.
        balanced_target = max(
            len(selected), count - int(round(count * explore_fraction))
        )
        # Randomized candidate tournaments bound runtime at O(budget*256),
        # rather than O(budget*entire logical universe).
        feature_cache = {c.case_id: features(c) for c in candidates}
        while len(selected) < balanced_target:
            if not candidates:
                break
            shortlist = rng.sample(range(len(candidates)), min(256, len(candidates)))

            def score(i):
                vals = feature_cache[candidates[i].case_id]
                primary = {
                    "fit_category": 6.0,
                    "composition": 6.0,
                    "observation": 3.0,
                    "sample_size": 1.5,
                    "degree": 1.5,
                    "support": 1.0,
                    "family": 0.20,
                }

                def weight(v):
                    if v[0] == "s":
                        return primary.get(v[1], 1.0)
                    return (
                        0.35
                        if v[1] in ("fit_category", "composition", "observation")
                        and v[2] in ("fit_category", "composition", "observation")
                        else 0.10
                    )

                return sum(weight(v) / (1 + counts.get(_canonical(v), 0)) for v in vals)

            best_i = max(shortlist, key=score)
            pick = candidates.pop(best_i)
            selected.append(pick)
            for feature in feature_cache[pick.case_id]:
                key = _canonical(feature)
                counts[key] = counts.get(key, 0) + 1
        rng.shuffle(candidates)
        selected.extend(candidates[: count - len(selected)])
        if len(selected) < count:
            raise ValueError("insufficient distinct eligible cases")
        return selected


def case_from_dict(d: dict) -> CaseSpec:
    """Recover a case from a manifest record (hashes can be verified by caller)."""
    from .distributions import from_dict

    if d.get("schema") != SCHEMA_VERSION:
        raise ValueError("unknown case schema")
    spec = CaseSpec(
        from_dict(d["distribution"]),
        ObservationSpec(**d["observation"]),
        FitSpec(**d["fit"]),
        int(d["n"]),
        int(d.get("replicate", 0)),
    )
    spec.validate()
    return spec


def coverage_counts(
    cases: list[CaseSpec],
    keys=(
        "fit_category",
        "composition",
        "observation",
        "sample_size",
        "degree",
        "support",
        "family",
    ),
    *,
    pairwise=True,
) -> dict[str, int]:
    """Run-history coverage counts, keyed as JSON to allow persistence."""
    counts = {}
    for case in cases:
        s = case.strata()
        f = [("s", k, s[k]) for k in keys]
        if pairwise:
            pair_keys = [k for k in keys if k != "family"]
            f += [
                ("p", k, j, s[k], s[j])
                for i, k in enumerate(pair_keys)
                for j in pair_keys[i + 1 :]
            ]
        for x in f:
            key = _canonical(x)
            counts[key] = counts.get(key, 0) + 1
    return counts


def manifest(cases: list[CaseSpec], selection_seed: int) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "selection_seed": selection_seed,
        "cases": [
            {"scenario_id": c.scenario_id, "case_id": c.case_id, "spec": c.as_dict()}
            for c in cases
        ],
    }
