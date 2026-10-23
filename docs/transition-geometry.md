# Automatic transition geometry (G3-057)

G3 fixes the nonlinear tanh gate geometry before optimizing potential
coefficients. This keeps the single-component coefficient problem convex.
The automatic full-line default uses the skew-aware rule selected by
out-of-sample experiments on smooth log-concave reference families (October
19, 2026). It is a fixed heuristic, not a maximum-likelihood estimate or a
proof of globally optimal gate placement.

## Eligible data

The rule applies to independent, **unweighted, uncensored single-component
point observations** on support `(-inf, +inf)`. It does **not** apply to
weighted/interval-censored samples, mixture-component responsibility fits,
or one-sided support. Bounded support has no tanh gates. For these excluded
cases, each available automatic center remains at empirical `q10` or `q90`
and the automatic steepness is `2/h` where `h` is the pre-existing robust
fitting-coordinate scale. All explicitly specified `TanhSpec` locations and
steepnesses override the corresponding automatic field. No new public option
or optimization mode is introduced.

## Formula

For the observations `x`, let `Q_p` denote the empirical `p` quantile and
write:

- `dL = Q_0.50 - Q_0.10`, `dU = Q_0.90 - Q_0.50`
- `A = log(dU/dL)` and `s = max(0, |A| - 0.12)`
- `w(n) = clip((n - 150)/550, 0, 1)`
- `u = min(0.07, 0.22 s)`, `k_slow = clip(2 exp(-5s), 0.4, 2)`
- `k_fast = clip(2 exp(0.75s), 2, 5)`

When the *upper* quantile spread is longer (`A > 0`), choose automatic
center levels `pL=0.10`, `pU=0.90-wu`; when the *lower* spread is longer
(`A < 0`), choose `pL=0.10+wu`, `pU=0.90`.

The shorter-tail gate takes `k_fast`, the longer-tail gate takes `k_slow`,
but shrink each toward baseline dimensionless steepness 2:
`k_adjusted = 2 * (k_target/2)**w`. The physical gate steepness is then
`alpha = k_adjusted/h`. Centers are `xL=Q_pL` and `xU=Q_pU`.

If `n <= 150`, either spacing is zero/unresolvable, or `|A| <= 0.12`, the
original `Q_0.10/Q_0.90` and `2/h` defaults are retained. Full strength
begins at `n >= 700`.

## Scientific scope

The rule showed held-out gains on asymmetric full-line log-concave targets
while leaving symmetric targets nearly unchanged in the isolated exploratory
experiments. This evidence does **not** extend to arbitrary mixtures,
weighted/censored observations or one-sided supports. Explicit values can
still be supplied through `TanhSpec` for reproducible geometry comparisons.
No optimizer convergence threshold or curvature certification condition was
altered in G3-057.
