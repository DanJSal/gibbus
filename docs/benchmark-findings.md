# G3-057 benchmark findings and interpretation

**Scope:** 96-case balanced exploratory benchmark run on 2026-10-19 using the G3-056 runner against G3-057 source/compiled kernels. Source library unchanged by this benchmark. Read the raw `benchmark/REPORT.md`, `benchmark/case_metrics.csv`, `benchmark/diagnostic_classification.csv`, and run manifests in the accompanying handoff for exact case IDs.

## Outcomes

| Category | Count | What it establishes |
| --- | ---: | --- |
| Fit + diagnostics successful | 79 | Runner completed, not necessarily a good statistical model. |
| Post-fit diagnostics failed | 10 | Numerical fit returned before diagnostic evaluation threw; cannot assume the convergence certificate was captured. |
| Timeout at 28 s | 6 | Runtime budget hit, not a convergence failure. |
| Explicit fit error | 1 | Degree-6, two-component fit to a three-component mixture, sample-dependent; three-component refit succeeded. |

Accuracy based on **54** successfully scored cases with independent PDF/CDF oracles: median maximum reference-CDF gap 0.02863 (p90 0.07549); median holdout log-density gap 0.01381 nats per point (p90 0.13440). Of the 79 successes, **25** were convolution targets without independent PDF/CDF oracle; empirical KS cannot replace ground truth. The original holdout size was 512, and the targeted refits sometimes used 2,048. All metric thresholds are diagnostic screens, not pass/fail correctness criteria.

## Anomalies and what they imply

1. **Explicit tail-model mismatch:** The largest logscore losses (roughly 52–69 nats/obs on the original holdout, greater on certain targeted replays) occurred when the request explicitly imposed polynomial tails on Gamma–Weibull mixtures. Soft-power alternatives dramatically improved logscore, often at higher cost. Do **not** silently alter a user-specified tail class or label the incompatibility an automatic-selection defect. Retain these as evidence of model-specification sensitivity.
2. **Automatic/mixture selection cost:** The dominant *open performance* issue. Typical single-component explicit fits are fast (~0.054 s median), whereas explicit two-component and auto-component fits had medians ~4.09 s and ~3.64 s over noncomparable workloads. Five of six timeouts eventually converged with 28.7–84.7 s fits; case `G3C-2a91bd6857739ffcc604` remained unresolved after a 90-second budget.
3. **Post-fit extreme quantiles:** Seven diagnostics errors requested probabilities near 1e-300, outside useful floating-point coordinate resolution for some bounded supports; three exhausted quadrature cells. Fitting had returned. This deserves a separate robust/representable diagnostic policy, without weakening shape certification.
4. **Basis capacity:** The worst full-line degree-2 Gumbel CDF gap (~0.1026) dropped at degrees 4 (~0.0456) and 6 (~0.0247). This is a model-capacity issue, not a sampled invalid-density defect.
5. **Component underspecification:** One explicit two-component, degree-6 fit failed on data from a three-component mixture; three components fit successfully. More seeds are needed to generalize.
6. **Statistical uncertainty:** A purposely enriched 24-case second-seed check showed 16 fully successful, 5 timeouts and 3 diagnostic errors. This is not an unbiased repeat-sample success rate.

## Runtime measurement guidance

Compare identical manifests, training/holdout observations and resource limits before attributing differences to algorithm changes. Split wall time (worker startup/import/metrics) from fitting duration; profile KDE proposal, model-count screening, candidate initializations, EM SQUAREM, Newton M-steps, joint polish, degree/expansion, normalization and post-fit independently. Record timeouts as right-censored durations, not completed fits. Time-limit increases are not speedups.

**Current priority:** diagnose time spent in automatic component selection and assess predictive accuracy for *automatically chosen* model configurations, with explicit and statistically fair controls. Do not optimize away deliberate user-specified tail misspecification.

## G3-059 profiling and fixes

**Environment:** CPython 3.13.16, NumPy 2.5.3, SciPy 1.18.1, 2 cores, single-threaded BLAS; roughly 2–5x slower than the G3-057 benchmark machine. Runner seed 91, holdout 512, 600 s timeout, one job per tree with both trees run concurrently.

**Where the time went.** Profiling `G3C-2a91bd6857739ffcc604` (517 s here, selecting one component) found two causes, both inside mixture fits:

1. *Duplicated exploration.* Each multistart finalist restarted EM from its exploration's responsibilities and recomputed the exploration phase exactly before polishing (190 of 490 s). Finalists now continue that phase (`_EMExploration`); results are bit-identical on all checked cases.
2. *Cold-started candidate faces.* When an M-step or polish enters a face (a collapsing tail leader), the incumbent projected onto the face keeps strict growth but violates a row by about 1e-11, against a feasibility floor of about 1e-14. The start was replaced by the face's cold start, 1.5–3 nats/obs above the incumbent, from which every Newton run hit its 60-iteration limit and each of 30 exchanges gained only 0.03–0.06 nats (about 105 s per candidate). The start is now restored toward the face anchor (`_toward_anchor`), and candidates converge in 2–4 runs at the incumbent's NLL.

**Paired result** (fix 1 alone versus fix 1 plus fix 2; status, held-out density and oracle CDF from the runner):

| Manifest | Status (both trees) | Fit time, total | p95 | Max | Held-out log density, largest change |
| --- | --- | ---: | ---: | ---: | --- |
| 96 balanced G3-057 cases | 95 ok, 1 fit error | 680 s → 271 s | 24.6 s → 9.8 s | 293 s → 50 s | −1.4e-5 to +4.5e-3 nats/obs (85 of 95 identical) |
| 26 mixture targets as `n_components="auto"` | 26 ok | 588 s → 148 s | 141 s → 16.6 s | 293 s → 31 s | within 9.1e-8 (22 of 26 identical) |

The six G3-057 timeout cases now fit in 4.2–50.1 s (previously 24.6–293.2 s with fix 1 alone). The fit error is the known `G3C-91ce6e608ed06a71d729` (two components fitted to a three-component target). One case became slower (`G3C-051c12835b78645b33cd`, 25 s → 38 s): its candidate solves got cheaper (9.1 s → 0.5 s), but it now adopts eight faces instead of four, and the different EM path costs more polish.

**Rejected: pruning auto-K refinements by screen BIC.** Every auto-K case in the 96-case manifest selects one component, and its two-component refinement recovers at most 13 BIC of a 9–63 BIC screen deficit. But on mixture targets a four-component screen 63 BIC *behind* one component refined to the winner (−251), so screen BIC is not a bound and a margin rule would lose correct selections.

**Post-fit margins.** The ten G3-057 diagnostic errors were all on bounded supports (`ppf`/`isf` at 1e-300 or 1e-12, or far-tail quadrature). The margin now approaches a finite end geometrically in `x`; all ten return in about 10 ms.

## G3-060 mixture cost analysis

**Where the time goes** (68 mixture and automatic-count fits, after G3-059): a mixture fit costs a median 55x (p90 140x) a single-component fit of the same data. M-steps are 60% of it and joint polishes 34%; a median fit runs 40 M-steps, each costing about one single-component fit, and 2–3 polishes of about three each.

**Historical G3-060 analysis (contact blocks removed in G3-078). Why a warm M-step cost a full fit.** Each M-step and polish is a cutting-plane convex solve. Near the optimum `q''` touches zero at a point that moves with the coefficients (a moving contact). The 2x2 contact block's local quadratic model is slightly optimistic, so every full Newton step overshoots by 1e-10 to 1e-6 (fitting units, against a 1e-12 shape tolerance); the compiled loop correctly rejects that trial, accepts a backtracked step and ends the run for a new cut. Runs then take one Newton step each (70–92% of runs), the gap closes by a near-constant 0.6–0.7 per round, and polishes routinely use all 30 exchange rounds. Given the cuts, Newton converges in one run (a candidate face solve inside a polish that inherited ~120 cuts: 2 rounds, 16 iterations).

**Tried and rejected** (paired, same cases):

| Change | Effect |
| --- | --- |
| Carry cuts between M-steps | Rounds unchanged; rows accumulate (600+) and runs slow 100x |
| Stencil cuts at `x* ± δ` around each cut | Rounds unchanged; 4–7x slower |
| Cap M-step exchanges at 3 (generalized EM) | 2x faster on average, but polishes from less converged EM states stop converging (one fit 16 s to 260 s) and one new fit error |
| Shorter exploration EM (2–6 maps) | 1.1–1.2x; one case loses 0.0005 nats/obs training likelihood |
| One finalist instead of two | **Adopted** (1.25–1.31x, no selected count changed) |

**Ratio-test step (adopted).** The halving line search was part of the slow rate: a trial that missed the contact by 1e-9 lost half its step. After a shape rejection by point cuts (`Phi''` at violated valleys), the compiled loop now shortens the trial to the exact largest feasible fraction for those rows; a repeated hint in one line search must shorten by at least 10% (unrestricted hints crept by 1e-4 per evaluation and doubled the objective evaluations). Tail and bound rejections still halve: hinting them steered one `Polynomial(1)` fit onto a point the separator certified while it violated an earlier cut by 1.2e-4. Polish rounds fell from 30 to 18 on the traced case (per-round ratio 0.7 to 0.45). Paired runtime: 96 cases 383 s to 323 s (largest 57.5 s to 40.2 s), mixture targets 207 s to 169 s (largest 32.5 s to 18.7 s); individual fits range from 1.8x faster to 1.4x slower; no status or selected count changed. Open items OI-E2-2 and OI-E2-4 now certify; OI-E2-3 remains.

**Remaining limit.** At the ratio-limited step both slacks are about 1e-9, so the feasible fraction is still about one half: the Newton direction overshoots a degenerate contact (`q''` flat to higher order, where the 2x2 contact model is skipped or uninformative) by about the remaining distance. Superlinear convergence there needs a higher-order contact model (derivatives beyond the fourth order the basis evaluators provide) or a parameterization that is convex by construction; both are design projects beyond Stage 6.

## G3-060 basis and tail selection experiment

**Question:** how far is the automatic single-component configuration from the best available one, and would decoupled tails help? 19 named families (9 full-line, 6 half-line, 4 bounded), n = 100, 500, 2000, three seeds for five families and one for the rest (87 datasets), point observations, held-out sample of 4000. Arms: today's default (`poly_degree="auto"`); the existing degree ladder with `SoftPower(1.5)` tails; and fixed bulk degrees 2–12 with bulk-derived, `Polynomial(2)`, `SoftPower(1.5)` and `Exponential(1)` tails (the same family at both ends). The excess is held-out loss against the best converged fit of each dataset, in nats per observation.

| Support | Default: median / p90 excess | Comment |
| --- | --- | --- |
| Half-line | 0.000 / 0.005 | No headroom |
| Bounded | 0.000 / 0.008 | No headroom |
| Full line | **0.011 / 0.061** | Headroom |

On the full line the loss concentrates in linear- or exponential-tailed families (hyperbolic secant 0.032, Gumbel 0.022–0.027, logistic 0.016; normal, generalized normal and log-gamma about 0). Cause: bulk-derived tails tie the tail growth order to the bulk degree, so the ladder's move from degree 2 to 4 also makes the tails quartic. Remedies: the existing ladder with `SoftPower(1.5)` tails (median 0.003, p90 0.022, max 0.031; tail-region gap p90 1.23 against 2.69) or BIC over {bulk-derived, soft-power} × degree {2, 4, 6} (median 0.0008 but p90 0.032, 0.5 s per fit). BIC chose bulk degree 2 in 60 of 87 datasets, so bulk complexity barely matters for these families; median fit time rose from 0.06 s at degree 2 to about 2 s at degree 12, with timeouts at degrees 10–12. `Exponential(1)` tails were rarely best and often failed; `Polynomial(1)` tails fail in this environment. **Decision:** keep the defaults; decoupled default tails and automatic tail selection are possible later work. Data and scripts are in the handoff's `evidence/G3-060/stage1_basis_selection/`.
