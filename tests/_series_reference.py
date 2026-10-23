"""Independent numerical checks for the G3 local-series specification.

The Decimal oracle expands the original functions to order N+4 and extracts
derivative coefficients at high precision. Production instead builds three
independent derivative jets. This module shares no production recurrences.
"""

from decimal import Decimal as D
from math import comb, factorial


def conv(a, b, n):
    return [
        sum((a[j] * b[k - j] for j in range(k + 1)), a[0] * 0) for k in range(n + 1)
    ]


def div_series(a, b, n):
    out = []
    for k in range(n + 1):
        out.append(
            (a[k] - sum((b[j] * out[k - j] for j in range(1, k + 1)), a[0] * 0)) / b[0]
        )
    return out


def d_exp_series(seed, slope, n):
    out = [seed.exp()]
    for k in range(1, n + 1):
        out.append(out[-1] * slope / k)
    return out


def oracle_atom(kind, pars, x, h, c, ell, n):
    x, h, c, ell = map(D, (x, h, c, ell))
    z, s = (x - c) / h, ell / h
    out = [D(0)] * (n + 1)
    if kind == "poly":
        m = pars[0]
        for k in range(min(n, m) + 1):
            out[k] = comb(m, k) * (z ** (m - k) if m != k else D(1)) * s**k
    elif kind == "exp":
        lam, loc = map(D, pars)
        out = d_exp_series(lam * (x - loc), lam * ell, n)
    elif kind == "log":
        eps, endpoint = map(D, pars)
        dist = eps * (x - endpoint)
        out[0] = -dist.ln()
        v = eps * ell / dist
        for k in range(1, n + 1):
            out[k] = (-1) ** k * v**k / k
    elif kind == "soft":
        p, delta = map(D, pars)
        a = p / 2
        base = z * z + delta / (h * h)
        u, v = 2 * z * s / base, s * s / base
        weight = base**a
        # Direct generalized-binomial expansion of the quadratic perturbation.
        for j in range(n + 1):
            for b in range(min(j, n - j) + 1):
                out[j + b] += (
                    weight * comb(j, b) * (u ** (j - b) if j != b else D(1)) * (v**b)
                )
            weight *= (a - j) / (j + 1)
    else:
        raise ValueError(kind)
    return out


def oracle_gate(a, b, x, ell, n, complement=False):
    a, b, x, ell = map(D, (a, b, x, ell))
    v0 = a * (x - b)
    sigma = D(1) if v0 >= 0 else D(-1)
    es = d_exp_series(-2 * abs(v0), -2 * sigma * a * ell, n)
    den = list(es)
    den[0] += 1
    w = div_series(es, den, n)
    if complement:
        return w
    t = [-2 * sigma * v for v in w]
    t[0] += sigma
    return t


def oracle_derivatives(series, ell, n):
    ell = D(ell)
    return [
        [
            series[k + r] * D(factorial(k + r) // factorial(k)) / (ell**r)
            for k in range(n + 1)
        ]
        for r in range(5)
    ]
