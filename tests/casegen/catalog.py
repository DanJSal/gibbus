"""Curated parameter grids and valid composition examples for case generation."""

from __future__ import annotations

from itertools import combinations, combinations_with_replacement

from .cases import _canonical
from .distributions import Affine as A
from .distributions import Mixture, Sum, support, to_dict, validate
from .distributions import Named as N
from .distributions import Truncated as T


def iter_pairwise_compositions(
    distributions, *, sums=True, mixtures=True, mixture_weights=(0.25, 0.5, 0.75)
):
    """Yield all finite pairwise sums and heterogeneous shared-support mixtures.

    The caller decides whether to materialize this larger catalog; the
    combinatorial enumeration never samples any observations or runs fits.
    """
    base = [d for d in distributions if not isinstance(d, (Sum, Mixture))]
    if sums:
        for a, b in combinations_with_replacement(base, 2):
            yield Sum(a, b)
    if mixtures:
        for a, b in combinations(base, 2):
            # Require different underlying families for cross-family mixtures.
            from .cases import _family_group

            if support(a) != support(b) or _family_group(a) == _family_group(b):
                continue
            for w in mixture_weights:
                if not 0 < w < 1:
                    raise ValueError("mixture weights must be in (0,1)")
                yield Mixture(((float(w), a), (float(1 - w), b)))


def default_distributions(profile: str = "standard", *, pairwise: bool = False):
    """A diverse, deterministic finite catalog, not a list of sampled datasets.

    ``core`` has a compact breadth-first catalog; ``standard`` adds more
    parameters, restricted families, reflections, sums and heterogeneous
    shared-support mixtures. New catalogs do not change the case ID schema.
    """
    if profile not in ("core", "standard"):
        raise ValueError("profile must be 'core' or 'standard'")
    d = [
        N("normal", mu=0, sigma=1),
        N("normal", mu=3, sigma=0.5),
        N("logistic", mu=0, scale=1),
        N("gumbel_r", mu=0, scale=1),
        N("gumbel_l", mu=0, scale=1),
        N("skew_normal", alpha=4, mu=0, sigma=1),
        N("hyperbolic_secant", mu=0, scale=1),
        N("generalized_normal", p=4, mu=0, scale=1),
        N("loggamma", shape=2, mu=0, scale=1),
        N("beta_logistic", a=2, b=4),
        N("exponential", scale=1),
        N("gamma", shape=2, scale=1),
        N("weibull", shape=2, scale=1),
        N("generalized_gamma", a=2, c=2, scale=1),
        N("chi", df=3),
        N("chi_square", df=3),
        N("nakagami", shape=1),
        N("gompertz", shape=2),
        N("generalized_inverse_gaussian", p=2, b=2),
        N("half_normal", scale=1),
        N("rayleigh", scale=1),
        N("maxwell", scale=1),
        N("beta", a=2, b=5),
        N("beta", a=5, b=2),
        N("beta", a=1, b=1),
        N("kumaraswamy", a=2, b=3),
        N("uniform"),
        N("semicircle"),
        N("cosine"),
        N("anglit"),
        N("generalized_pareto_bounded", shape=-0.5),
    ]
    if profile == "standard":
        d += [
            N("normal", mu=-5, sigma=3),
            N("normal", mu=0, sigma=0.1),
            N("logistic", mu=2, scale=0.3),
            N("skew_normal", alpha=-5, mu=0, sigma=1),
            N("skew_normal", alpha=0, mu=0, sigma=1),
            N("generalized_normal", p=2.5, mu=0, scale=1),
            N("generalized_normal", p=6, mu=0, scale=2),
            N("loggamma", shape=0.5, mu=0, scale=1),
            N("beta_logistic", a=0.5, b=3),
            N("beta_logistic", a=5, b=0.5),
            N("gamma", shape=1, scale=2),
            N("gamma", shape=8, scale=0.3),
            N("weibull", shape=1, scale=2),
            N("weibull", shape=4, scale=0.8),
            N("generalized_gamma", a=0.5, c=2, scale=1),
            N("chi", df=1),
            N("chi_square", df=10),
            N("nakagami", shape=0.5),
            N("generalized_inverse_gaussian", p=1, b=0.25),
            N("beta", a=1, b=4),
            N("beta", a=4, b=1),
            N("beta", a=10, b=10),
            N("kumaraswamy", a=1, b=2),
            N("generalized_pareto_bounded", shape=-0.25),
            # Spatial transforms and support-bounded conditional targets.
            A(N("gamma", shape=3, scale=1), loc=0, scale=-1),
            A(N("weibull", shape=2, scale=1), loc=0, scale=-1),
            A(N("exponential", scale=1), loc=0, scale=-1),
            A(N("beta", a=3, b=2), loc=-2, scale=4),
            A(N("normal", mu=0, sigma=1), loc=1e3, scale=0.1),
            T(N("normal", mu=0, sigma=1), -2, 2),
            T(N("normal", mu=0, sigma=1), 0, 3),
            T(N("lognormal_restricted", mu=0, sigma=1), 0.1, 0.9),
            T(N("student_t_restricted", df=5), -2, 2),
            T(N("cauchy_restricted"), -0.75, 0.75),
            T(N("beta", a=0.5, b=2), 0.6, 1),
            T(N("beta", a=2, b=0.5), 0, 0.4),
            T(N("inverse_gaussian_restricted", mu=1, scale=1), 0.05, 0.5),
            T(N("logistic", mu=0, scale=1), -1, 1),
            # Heterogeneous independent sums: log-concavity by convolution.
            Sum(N("normal", mu=0, sigma=1), N("gamma", shape=2, scale=1)),
            Sum(N("gamma", shape=2, scale=1), N("weibull", shape=2, scale=1)),
            Sum(N("beta", a=2, b=4), N("beta", a=4, b=2)),
            Sum(
                A(N("gamma", shape=2, scale=1), scale=-1),
                A(N("weibull", shape=2, scale=1), scale=-1),
            ),
            Sum(N("normal", mu=1, sigma=0.5), N("logistic", mu=0, scale=1)),
            # Component supports exactly agree; families differ.
            Mixture(
                (
                    (0.35, N("normal", mu=-2, sigma=0.7)),
                    (0.65, N("logistic", mu=2, scale=0.9)),
                )
            ),
            Mixture(
                (
                    (0.4, N("gamma", shape=2, scale=1)),
                    (0.6, N("weibull", shape=3, scale=1)),
                )
            ),
            Mixture(((0.25, N("beta", a=2, b=5)), (0.75, N("kumaraswamy", a=3, b=2)))),
            Mixture(
                (
                    (0.3, A(N("gamma", shape=2, scale=1), scale=-1)),
                    (0.7, A(N("weibull", shape=2, scale=1), scale=-1)),
                )
            ),
            Mixture(
                (
                    (
                        0.5,
                        Sum(N("normal", mu=0, sigma=1), N("gamma", shape=2, scale=1)),
                    ),
                    (0.5, N("gumbel_r", mu=0, scale=1)),
                )
            ),
            Mixture(
                (
                    (0.2, N("normal", mu=-2, sigma=0.5)),
                    (0.5, N("logistic", mu=0, scale=1)),
                    (0.3, N("gumbel_r", mu=2, scale=0.7)),
                )
            ),
            Mixture(
                (
                    (0.2, N("beta", a=2, b=4)),
                    (0.3, N("kumaraswamy", a=4, b=2)),
                    (0.5, N("uniform")),
                )
            ),
        ]
    if pairwise:
        d.extend(iter_pairwise_compositions(d))
    seen, result = set(), []
    for item in d:
        validate(item)
        key = _canonical(to_dict(item))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return tuple(result)
