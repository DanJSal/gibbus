# Modeling concepts

Gibbus fits smooth univariate probability distributions by maximum likelihood under a per-component log-concavity constraint. This document describes the model at a conceptual level; [Numerical methods and performance](numerical-methods.md) covers implementation details.

## A Gibbus distribution

For a single component, the density can be written conceptually as

```text
p(x) ∝ exp(-V(x))
```

on a specified support, where the potential `V(x)` is convex. Convexity of the negative-log density is equivalent to log-concavity of the density, so every single component is unimodal.

Within a component, Gibbus represents the convex potential as a linear combination of named basis functions of a fixed fitting coordinate: monomials up to a bulk degree, optional logarithmic terms at finite support boundaries and, on request, noninteger soft-power or exponential tail terms (see [Fitting](fitting.md#basis-requests)). Increasing the bulk degree permits progressively richer smooth shapes. The representation is a flexible model for the potential; it does **not** assume that the data-generating distribution itself has a low-degree polynomial negative-log density.

The fitted potential and density are explicit analytic functions on the interior of the support. They are not stored as a histogram, interpolation grid, KDE, spline, or piecewise-linear log-density.

## Relationship to named parametric families

A fixed named family such as Normal, Gamma, or Beta chooses a small predetermined functional form and estimates its parameters. Gibbus instead keeps the structural assumption of log-concavity while allowing the smooth potential to adapt through its polynomial degree and boundary structure.

This makes Gibbus useful when a smooth unimodal distribution is appropriate but selecting a specific named family would be unnecessarily restrictive. The tradeoff is that Gibbus is still a model: log-concavity and the declared support are substantive assumptions, especially for extrapolation into the tails.

## Relationship to nonparametric density estimators

A KDE constructs a density from localized kernels and a bandwidth. Gibbus instead fits a normalized analytic distribution by maximum likelihood under global shape and support constraints. The result therefore comes with a coherent fitted CDF, quantile function, moments, survival quantities, and related operations derived from the same model.

Gibbus does use KDE mode counting as one proposal mechanism during **optional automatic mixture selection**, but the final fitted components are Gibbus distributions rather than KDE components.

## Support is part of the model

The `support` argument defines the domain of the density:

| Support | Example | Meaning |
|---|---|---|
| Full line | `(-np.inf, np.inf)` | Unbounded on both sides |
| Half-line | `(0, np.inf)` | Finite lower boundary |
| Bounded | `(0, 1)` | Finite lower and upper boundaries |
| `None` | — | Equivalent to `(-np.inf, np.inf)` |

Support is a structural modeling choice, not a statistic inferred from the observed minimum and maximum. An all-positive sample does not by itself imply a hard boundary at zero. If the variable is non-negative by construction, specify `support=(0, np.inf)` explicitly.

## Finite-boundary terms

At a finite endpoint, Gibbus can include an optional logarithmic boundary term. These terms allow controlled singular behavior near a physical boundary while retaining the global convexity constraint. The fitting API can enable or disable them explicitly or select them from the data; see [Fitting](fitting.md#boundary-terms).

For mixtures, the presence and fitted amplitude of each enabled physical-boundary term are shared across components. Each component retains its own bulk shape and numerical fitting coordinates.

## Bulk degree

The bulk degree (`poly_degree`) controls the flexibility of the smooth component potential. Higher degree is not automatically better: it adds flexibility and parameters, and therefore requires model selection and numerical certification.

The convenience `poly_degree` route rejects odd degrees on full-infinite support. A general `BasisSpec` can admit odd bulk degrees with independently constrained tails; some directions may be fixed to zero on an exact coefficient face. Automatic degree selection considers admissible rungs. Mixtures may retain different bulk degrees by component.

For fixed support, geometry, tail structure and boundary features, nested
polynomial spans form a statistical sieve: successive degrees offer increasingly
flexible log-concave potentials. Automatic growth tests omitted sufficient
statistics after projecting out directions already represented by the fit. It
adds complexity when those statistics carry resolvable omitted information,
with likelihood-ratio confirmation where appropriate. General-basis span
expansion uses the same idea beyond polynomial powers. The nesting statement
holds for fixed structural choices; changing the geometry or tail family may
change the represented family instead.

This degree/span decision differs from automatic mixture-count selection,
which compares refined candidate component counts using BIC. Neither procedure
certifies population tail accuracy or a globally optimal mixture search.

## Mixtures and multimodality

A single component is log-concave and therefore unimodal. A finite mixture of such components can be multimodal:

```python
c = Distribution().fit(data, n_components=3)
```

or the component count can be selected automatically with `n_components="auto"`.

Each mixture component remains log-concave, but the mixture density as a whole generally is not. Properties that follow from single-component log-concavity, such as a non-decreasing hazard rate, therefore do not automatically carry over to mixtures.

## Base and exp coordinate views

A fitted distribution has two coordinate views:

- **Base space** models the fitted variable `X` directly.
- **Exp space** represents `Y = exp(X)`.

Exp space is useful when positive observations are naturally modeled on a logarithmic scale. A typical workflow is to fit `log(data)` in base space and query the corresponding positive variable through `c.exp` or by setting the active space to `"exp"`.

The random variable in exp space is strictly positive. For a full-real-line base support, the public support is reported as the closed interval `[0, inf]`, reflecting the limiting boundary at zero even though `exp(X)` never equals zero for finite `X`.

## Exponential-family and maximum-entropy perspective

For a fixed support and basis, the potential coefficients enter linearly in the log density. This gives the fitted component an exponential-family structure, with normalization and moment quantities tied to derivatives of the log-partition function. The maximum-likelihood fit can therefore also be viewed through the corresponding moment-matching / maximum-entropy geometry subject to the global convexity constraint.

That perspective is useful for understanding why point-data fitting admits exact sufficient-statistic compression and why the Hessian has a Fisher-information interpretation. It is not necessary for routine use of the package.
