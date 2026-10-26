"""Deterministic log-concave distribution specifications; no dependency on Gibbus.

The registry intentionally contains only parameter regions justified by analytic
log-density curvature. A named distribution which is log-concave only on a
subinterval must be enclosed in Truncated before validation/sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import special, stats


@dataclass(frozen=True, init=False)
class Named:
    family: str
    params: tuple[tuple[str, float], ...]

    def __init__(self, family: str, **params: float):
        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "params", tuple(sorted((k, float(v)) for k, v in params.items()))
        )

    def arguments(self) -> dict[str, float]:
        return dict(self.params)


@dataclass(frozen=True)
class Affine:
    source: "DistributionSpec"
    loc: float = 0.0
    scale: float = 1.0


@dataclass(frozen=True)
class Truncated:
    source: "DistributionSpec"
    lower: float
    upper: float


@dataclass(frozen=True)
class Sum:
    left: "DistributionSpec"
    right: "DistributionSpec"


@dataclass(frozen=True)
class Mixture:
    components: tuple[tuple[float, "DistributionSpec"], ...]


DistributionSpec = Union[Named, Affine, Truncated, Sum, Mixture]


# kind: scipy distribution, with full-family log-concavity conditions below.
# Entry format: (parameter names, scipy name, support, smoothness).
# C2 means at least two interior derivatives; Cinf is interior C-infinity.
FAMILIES = {
    "normal": (("mu", "sigma"), "norm", (-math.inf, math.inf), "Cinf"),
    "logistic": (("mu", "scale"), "logistic", (-math.inf, math.inf), "Cinf"),
    "gumbel_r": (("mu", "scale"), "gumbel_r", (-math.inf, math.inf), "Cinf"),
    "gumbel_l": (("mu", "scale"), "gumbel_l", (-math.inf, math.inf), "Cinf"),
    "skew_normal": (
        ("alpha", "mu", "sigma"),
        "skewnorm",
        (-math.inf, math.inf),
        "Cinf",
    ),
    "hyperbolic_secant": (("mu", "scale"), "hypsecant", (-math.inf, math.inf), "Cinf"),
    "generalized_normal": (
        ("p", "mu", "scale"),
        "gennorm",
        (-math.inf, math.inf),
        "C2",
    ),
    "loggamma": (("shape", "mu", "scale"), "loggamma", (-math.inf, math.inf), "Cinf"),
    "beta_logistic": (("a", "b"), None, (-math.inf, math.inf), "Cinf"),
    "exponential": (("scale",), "expon", (0.0, math.inf), "Cinf"),
    "gamma": (("shape", "scale"), "gamma", (0.0, math.inf), "Cinf"),
    "weibull": (("shape", "scale"), "weibull_min", (0.0, math.inf), "Cinf"),
    "generalized_gamma": (("a", "c", "scale"), "gengamma", (0.0, math.inf), "Cinf"),
    "chi": (("df",), "chi", (0.0, math.inf), "Cinf"),
    "chi_square": (("df",), "chi2", (0.0, math.inf), "Cinf"),
    "nakagami": (("shape",), "nakagami", (0.0, math.inf), "Cinf"),
    "gompertz": (("shape",), "gompertz", (0.0, math.inf), "Cinf"),
    "generalized_inverse_gaussian": (
        ("p", "b"),
        "geninvgauss",
        (0.0, math.inf),
        "Cinf",
    ),
    "half_normal": (("scale",), "halfnorm", (0.0, math.inf), "Cinf"),
    "rayleigh": (("scale",), "rayleigh", (0.0, math.inf), "Cinf"),
    "maxwell": (("scale",), "maxwell", (0.0, math.inf), "Cinf"),
    "beta": (("a", "b"), "beta", (0.0, 1.0), "Cinf"),
    "kumaraswamy": (("a", "b"), None, (0.0, 1.0), "Cinf"),
    "uniform": ((), "uniform", (0.0, 1.0), "Cinf"),
    "semicircle": ((), "semicircular", (-1.0, 1.0), "Cinf"),
    "cosine": ((), "cosine", (-math.pi, math.pi), "Cinf"),
    "anglit": ((), "anglit", (-math.pi / 4, math.pi / 4), "Cinf"),
    "generalized_pareto_bounded": (("shape",), "genpareto", (0.0, math.inf), "Cinf"),
    # Only valid under an explicit curvature-contained Truncated node.
    "lognormal_restricted": (("mu", "sigma"), "lognorm", (0.0, math.inf), "Cinf"),
    "student_t_restricted": (("df",), "t", (-math.inf, math.inf), "Cinf"),
    "cauchy_restricted": ((), "cauchy", (-math.inf, math.inf), "Cinf"),
    "inverse_gaussian_restricted": (
        ("mu", "scale"),
        "invgauss",
        (0.0, math.inf),
        "Cinf",
    ),
}
RESTRICTED = frozenset(
    {
        "lognormal_restricted",
        "student_t_restricted",
        "cauchy_restricted",
        "inverse_gaussian_restricted",
    }
)


def _check_named(s: Named, interval: tuple[float, float] | None) -> None:
    if s.family not in FAMILIES:
        raise ValueError(f"unknown family: {s.family!r}")
    expected = set(FAMILIES[s.family][0])
    p = s.arguments()
    if set(p) != expected or not all(math.isfinite(v) for v in p.values()):
        raise ValueError(f"{s.family} requires finite parameters {sorted(expected)}")
    f = s.family
    pos = ("sigma", "scale", "shape", "a", "b", "c", "df", "p")
    if any(p[k] <= 0 for k in pos if k in p):
        # 'shape' in bounded generalized Pareto is intentionally negative.
        if f != "generalized_pareto_bounded":
            raise ValueError("scale and shape parameters must be positive")
    if f == "generalized_pareto_bounded":
        if not -1 <= p["shape"] < 0:
            raise ValueError("bounded generalized Pareto needs -1 <= shape < 0")
    elif f in ("gamma", "weibull") and p["shape"] < 1:
        raise ValueError(f"{f} requires shape >= 1")
    elif f == "generalized_gamma" and (p["c"] < 1 or p["a"] * p["c"] < 1):
        raise ValueError("generalized gamma requires c>=1 and a*c>=1")
    elif f == "generalized_normal" and p["p"] < 2:
        raise ValueError("C2 generalized normal requires p>=2")
    elif f == "chi" and p["df"] < 1:
        raise ValueError("chi requires df>=1")
    elif f == "chi_square" and p["df"] < 2:
        raise ValueError("chi-square requires df>=2")
    elif f == "nakagami" and p["shape"] < 0.5:
        raise ValueError("nakagami requires shape>=1/2")
    elif f == "generalized_inverse_gaussian" and p["p"] < 1:
        raise ValueError("generalized inverse Gaussian requires p>=1")
    elif f == "kumaraswamy" and min(p["a"], p["b"]) < 1:
        raise ValueError("Kumaraswamy requires a,b>=1")
    elif f == "beta" and min(p["a"], p["b"]) < 1:
        a, b = p["a"], p["b"]
        if interval is None or interval[0] < 0 or interval[1] > 1:
            raise ValueError("beta with subunit shape requires a certified truncation")
        lo, hi = interval
        if a < 1 and b > 1:
            cutoff = math.sqrt(1 - a) / (math.sqrt(1 - a) + math.sqrt(b - 1))
            if lo < cutoff:
                raise ValueError("truncation crosses beta's convex lower region")
        elif b < 1 and a > 1:
            cutoff = math.sqrt(a - 1) / (math.sqrt(a - 1) + math.sqrt(1 - b))
            if hi > cutoff:
                raise ValueError("truncation crosses beta's convex upper region")
        else:
            raise ValueError("no log-concave interval for these subunit beta shapes")
    elif f in RESTRICTED:
        if (
            interval is None
            or not interval[0] < interval[1]
            or not all(map(math.isfinite, interval))
        ):
            raise ValueError(f"{f} requires an explicit bounded truncation")
        lo, hi = interval
        if f == "lognormal_restricted" and not (
            lo > 0 and math.log(hi) <= p["mu"] + 1 - p["sigma"] ** 2
        ):
            raise ValueError("lognormal truncation leaves its log-concave region")
        if f == "student_t_restricted" and (
            lo < -math.sqrt(p["df"]) or hi > math.sqrt(p["df"])
        ):
            raise ValueError("Student t truncation leaves its log-concave center")
        if f == "cauchy_restricted" and (lo < -1 or hi > 1):
            raise ValueError("Cauchy truncation leaves its log-concave center")
        if f == "inverse_gaussian_restricted" and (lo < 0 or hi > 2 * p["scale"] / 3):
            raise ValueError(
                "inverse Gaussian truncation crosses its curvature inflection"
            )


def support(s: DistributionSpec) -> tuple[float, float]:
    if isinstance(s, Named):
        lo, hi = FAMILIES[s.family][2]
        return (
            (lo, -1.0 / s.arguments()["shape"])
            if s.family == "generalized_pareto_bounded"
            else (lo, hi)
        )
    if isinstance(s, Affine):
        a, b = support(s.source)
        bounds = (s.loc + s.scale * a, s.loc + s.scale * b)
        return (min(bounds), max(bounds))
    if isinstance(s, Truncated):
        a, b = support(s.source)
        return max(a, s.lower), min(b, s.upper)
    if isinstance(s, Sum):
        a, b = support(s.left), support(s.right)
        return a[0] + b[0], a[1] + b[1]
    if isinstance(s, Mixture):
        if not s.components:
            raise ValueError("mixture must have components")
        return support(s.components[0][1])
    raise TypeError(type(s))


def validate(s: DistributionSpec) -> None:
    """Require a proper log-concave component or a valid component mixture."""
    _validate(s, None)


def _validate(s: DistributionSpec, interval: tuple[float, float] | None) -> None:
    if isinstance(s, Named):
        _check_named(s, interval)
    elif isinstance(s, Affine):
        if not (math.isfinite(s.loc) and math.isfinite(s.scale) and s.scale != 0):
            raise ValueError(
                "affine transform requires finite location and nonzero scale"
            )
        if interval is None:
            inner = None
        else:
            inv = ((interval[0] - s.loc) / s.scale, (interval[1] - s.loc) / s.scale)
            inner = min(inv), max(inv)
        _validate(s.source, inner)
    elif isinstance(s, Truncated):
        if math.isnan(s.lower) or math.isnan(s.upper):
            raise ValueError("NaN truncation endpoints")
        a, b = support(s)
        if not a < b or not (s.lower < s.upper):
            raise ValueError("empty truncation support")
        if interval is not None:
            a, b = max(a, interval[0]), min(b, interval[1])
        _validate(s.source, (a, b))
    elif isinstance(s, Sum):
        _validate(s.left, None)
        _validate(s.right, None)
    elif isinstance(s, Mixture):
        if len(s.components) < 2 or any(
            not math.isfinite(w) or w <= 0 for w, _ in s.components
        ):
            raise ValueError("mixture requires >=2 positive finite weights")
        if not math.isclose(
            sum(w for w, _ in s.components), 1, abs_tol=1e-12, rel_tol=0
        ):
            raise ValueError("mixture weights must sum to one")
        common = support(s.components[0][1])
        for _, comp in s.components:
            _validate(comp, None)
            if support(comp) != common:
                raise ValueError("mixture components must share exact support")
    else:
        raise TypeError(type(s))
    a, b = support(s)
    if not a < b or math.isnan(a) or math.isnan(b):
        raise ValueError("invalid support")


def category(s: DistributionSpec) -> str:
    if isinstance(s, Mixture):
        return "mixture"
    if isinstance(s, Sum):
        return "sum"
    if isinstance(s, (Affine, Truncated)):
        return category(s.source)
    return "named"


def smoothness(s: DistributionSpec) -> str:
    """Conservative interior smoothness guarantee: Cinf, C2, or unknown."""
    if isinstance(s, Named):
        return FAMILIES[s.family][3]
    if isinstance(s, (Affine, Truncated)):
        return smoothness(s.source)
    if isinstance(s, Mixture):
        parts = [smoothness(x) for _, x in s.components]
        return "unknown" if "unknown" in parts else ("C2" if "C2" in parts else "Cinf")
    if isinstance(s, Sum):
        # Gaussian convolution is C-infinity even with discontinuous boundaries.
        def gaussian(node):
            return (
                isinstance(node, Named)
                and node.family == "normal"
                or (isinstance(node, Affine) and gaussian(node.source))
            )

        return "Cinf" if gaussian(s.left) or gaussian(s.right) else "unknown"
    raise TypeError(type(s))


def _frozen(s: Named):
    p = s.arguments()
    f = s.family
    if f in ("kumaraswamy", "beta_logistic"):
        return None
    if f == "normal":
        return stats.norm(loc=p["mu"], scale=p["sigma"])
    if f in ("logistic", "gumbel_r", "gumbel_l", "hyperbolic_secant"):
        return getattr(stats, FAMILIES[f][1])(loc=p["mu"], scale=p["scale"])
    if f == "skew_normal":
        return stats.skewnorm(p["alpha"], loc=p["mu"], scale=p["sigma"])
    if f == "generalized_normal":
        return stats.gennorm(p["p"], loc=p["mu"], scale=p["scale"])
    if f == "loggamma":
        return stats.loggamma(p["shape"], loc=p["mu"], scale=p["scale"])
    if f in ("exponential", "half_normal", "rayleigh", "maxwell"):
        return getattr(stats, FAMILIES[f][1])(scale=p["scale"])
    if f in ("gamma", "weibull"):
        return getattr(stats, FAMILIES[f][1])(p["shape"], scale=p["scale"])
    if f == "generalized_gamma":
        return stats.gengamma(p["a"], p["c"], scale=p["scale"])
    if f in ("chi", "chi_square"):
        return getattr(stats, FAMILIES[f][1])(p["df"])
    if f in ("nakagami", "gompertz"):
        return getattr(stats, FAMILIES[f][1])(p["shape"])
    if f == "generalized_inverse_gaussian":
        return stats.geninvgauss(p["p"], p["b"])
    if f == "beta":
        return stats.beta(p["a"], p["b"])
    if f == "generalized_pareto_bounded":
        return stats.genpareto(p["shape"])
    if f == "lognormal_restricted":
        return stats.lognorm(p["sigma"], scale=math.exp(p["mu"]))
    if f == "student_t_restricted":
        return stats.t(p["df"])
    if f == "inverse_gaussian_restricted":
        return stats.invgauss(p["mu"], scale=p["scale"])
    return getattr(stats, FAMILIES[f][1])()


def cdf(s: DistributionSpec, x):
    arr = np.asarray(x, dtype=float)
    if isinstance(s, Named):
        p = s.arguments()
        if s.family == "beta_logistic":
            return special.betainc(p["a"], p["b"], special.expit(arr))
        if s.family == "kumaraswamy":
            return np.where(
                arr <= 0,
                0.0,
                np.where(
                    arr >= 1,
                    1.0,
                    -np.expm1(p["b"] * np.log1p(-(np.clip(arr, 0, 1) ** p["a"]))),
                ),
            )
        return _frozen(s).cdf(arr)
    if isinstance(s, Affine):
        z = (arr - s.loc) / s.scale
        return cdf(s.source, z) if s.scale > 0 else 1 - cdf(s.source, z)
    if isinstance(s, Truncated):
        a, b = support(s)
        low, high = cdf(s.source, a), cdf(s.source, b)
        mass = high - low
        if mass <= 0:
            raise ValueError("truncation has numerically unresolved mass")
        return np.where(
            arr < a,
            0.0,
            np.where(arr >= b, 1.0, np.clip((cdf(s.source, arr) - low) / mass, 0, 1)),
        )
    if isinstance(s, Mixture):
        return sum(w * cdf(c, arr) for w, c in s.components)
    raise NotImplementedError("CDF oracle for convolutions is not supplied")


def pdf(s: DistributionSpec, x):
    arr = np.asarray(x, dtype=float)
    if isinstance(s, Named):
        p = s.arguments()
        if s.family == "beta_logistic":
            return np.exp(
                p["a"] * special.log_expit(arr)
                + p["b"] * special.log_expit(-arr)
                - special.betaln(p["a"], p["b"])
            )
        if s.family == "kumaraswamy":
            z = np.clip(arr, 1e-300, 1 - 1e-15)
            logval = (
                math.log(p["a"] * p["b"])
                + (p["a"] - 1) * np.log(z)
                + (p["b"] - 1) * np.log1p(-(z ** p["a"]))
            )
            return np.where((arr > 0) & (arr < 1), np.exp(logval), 0.0)
        return _frozen(s).pdf(arr)
    if isinstance(s, Affine):
        return pdf(s.source, (arr - s.loc) / s.scale) / abs(s.scale)
    if isinstance(s, Truncated):
        a, b = support(s)
        mass = cdf(s.source, b) - cdf(s.source, a)
        if mass <= 0:
            raise ValueError("truncation has numerically unresolved mass")
        return np.where((arr > a) & (arr < b), pdf(s.source, arr) / mass, 0.0)
    if isinstance(s, Mixture):
        return sum(w * pdf(c, arr) for w, c in s.components)
    raise NotImplementedError("PDF oracle for convolutions is not supplied")


def _ppf_named(s: Named, u):
    p = s.arguments()
    if s.family == "beta_logistic":
        return special.logit(stats.beta(p["a"], p["b"]).ppf(u))
    if s.family == "kumaraswamy":
        return (-np.expm1(np.log1p(-u) / p["b"])) ** (1 / p["a"])
    return _frozen(s).ppf(u)


def _ppf(s: DistributionSpec, u):
    if isinstance(s, Named):
        return _ppf_named(s, u)
    if isinstance(s, Affine):
        return s.loc + s.scale * _ppf(s.source, u if s.scale > 0 else 1 - u)
    if isinstance(s, Truncated):
        a, b = support(s)
        low, hi = cdf(s.source, a), cdf(s.source, b)
        return _ppf(
            s.source,
            np.clip(low + u * (hi - low), np.nextafter(0, 1), np.nextafter(1, 0)),
        )
    raise NotImplementedError("quantile oracle unavailable")


def sample(s: DistributionSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Independent samples; does not consult the Gibbus fit implementation."""
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be nonnegative integer")
    if isinstance(s, Mixture):
        choices = rng.choice(len(s.components), size=n, p=[w for w, _ in s.components])
        out = np.empty(n)
        for i, (_, part) in enumerate(s.components):
            mask = choices == i
            out[mask] = sample(part, int(mask.sum()), rng)
        return out
    if isinstance(s, Sum):
        return sample(s.left, n, rng) + sample(s.right, n, rng)
    if isinstance(s, Affine):
        return s.loc + s.scale * sample(s.source, n, rng)
    if isinstance(s, Truncated):
        try:
            u = rng.uniform(size=n)
            result = np.asarray(_ppf(s, u), dtype=float)
            if np.all(np.isfinite(result)):
                return result
        except (NotImplementedError, ValueError):
            pass
        # Rejection sampling is a fallback only for compositions without ppf.
        # It is deliberately capped: tiny truncation events must not hang.
        lo, hi = support(s)
        out, filled, attempts = np.empty(n), 0, 0
        while filled < n and attempts < 100:
            batch = sample(s.source, max(128, 2 * (n - filled)), rng)
            accepted = batch[(batch >= lo) & (batch <= hi)]
            count = min(len(accepted), n - filled)
            out[filled : filled + count] = accepted[:count]
            filled += count
            attempts += 1
        if filled != n:
            raise ValueError(
                "truncation rejection sampler exhausted; use a less rare event"
            )
        return out
    if isinstance(s, Named):
        if s.family in ("beta_logistic", "kumaraswamy"):
            u = rng.uniform(np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0), size=n)
            return np.asarray(_ppf_named(s, u), dtype=float)
        return np.asarray(_frozen(s).rvs(size=n, random_state=rng), dtype=float)
    raise TypeError(type(s))


def _json_endpoint(value: float):
    return ("+inf" if value > 0 else "-inf") if math.isinf(value) else value


def to_dict(s: DistributionSpec) -> dict:
    if isinstance(s, Named):
        return {"op": "named", "family": s.family, "parameters": s.arguments()}
    if isinstance(s, Affine):
        return {
            "op": "affine",
            "source": to_dict(s.source),
            "loc": float(s.loc),
            "scale": float(s.scale),
        }
    if isinstance(s, Truncated):
        return {
            "op": "truncate",
            "source": to_dict(s.source),
            "lower": _json_endpoint(float(s.lower)),
            "upper": _json_endpoint(float(s.upper)),
        }
    if isinstance(s, Sum):
        return {"op": "sum", "left": to_dict(s.left), "right": to_dict(s.right)}
    if isinstance(s, Mixture):
        return {
            "op": "mixture",
            "components": [
                {"weight": float(w), "distribution": to_dict(c)}
                for w, c in s.components
            ],
        }
    raise TypeError(type(s))


def from_dict(d: dict) -> DistributionSpec:
    op = d["op"]
    if op == "named":
        return Named(d["family"], **d["parameters"])
    if op == "affine":
        return Affine(from_dict(d["source"]), d["loc"], d["scale"])
    if op == "truncate":
        return Truncated(from_dict(d["source"]), float(d["lower"]), float(d["upper"]))
    if op == "sum":
        return Sum(from_dict(d["left"]), from_dict(d["right"]))
    if op == "mixture":
        return Mixture(
            tuple((c["weight"], from_dict(c["distribution"])) for c in d["components"])
        )
    raise ValueError(f"unknown distribution operation: {op}")
