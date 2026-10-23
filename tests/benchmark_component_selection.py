"""Benchmark automatic component selection on fixed train/holdout samples.

Run original and updated source trees sequentially with the same BLAS limits:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/benchmark_component_selection.py --source . --output selection-runtime.json

Existing output rows are resumed. Use a new output path for fresh timings.
On POSIX a 30-second signal timer limits individual fits cooperatively; native
operations can delay delivery. Use an external process deadline for hard limits.
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--output", required=True)
p.add_argument("--screen-tolerance", type=float)
p.add_argument("--seed", type=int)
p.add_argument("--skip-uniform", action="store_true")
p.add_argument("--family")
p.add_argument("--fit-budget", type=float)
a = p.parse_args()
sys.path.insert(0, str(Path(a.source).resolve()))
import numpy as np  # noqa: E402 — source path is selected by CLI

import gibbus._fit.basis_mixture as mixture  # noqa: E402
from gibbus import Distribution  # noqa: E402

if a.screen_tolerance is not None:
    mixture.AUTO_LC_SWEEP_NEWTON_TOL = a.screen_tolerance


class BenchmarkTimeout(TimeoutError):
    pass


def expired(*args):
    raise BenchmarkTimeout("30-second benchmark budget exhausted")


if hasattr(signal, "SIGALRM"):
    signal.signal(signal.SIGALRM, expired)
records = json.loads(Path(a.output).read_text()) if Path(a.output).exists() else []
finished = {(r["seed"], r["family"]) for r in records}

for seed in (91, 2045, 20261022):
    if a.seed is not None and seed != a.seed:
        continue
    for index, name in enumerate(
        (
            "normal",
            "gumbel",
            "separated",
            "overlap",
            "unbalanced",
            "gamma",
            "beta",
            "beta_mixture",
            "uniform",
            "four_mode",
        )
    ):
        if index >= 5 and seed != 91:
            continue
        if a.skip_uniform and name == "uniform":
            continue
        if a.family is not None and name != a.family:
            continue
        if (seed, name) in finished:
            continue
        support = None
        extra = {}
        if a.fit_budget is not None:
            extra["max_fit_seconds"] = a.fit_budget
        r = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
        h = np.random.default_rng(np.random.SeedSequence([seed, index, 1]))
        if name == "normal":
            x = r.normal(size=600)
            test = h.normal(size=20000)
        elif name == "gumbel":
            x = r.gumbel(size=600)
            test = h.gumbel(size=20000)
        elif name == "gamma":
            x = r.gamma(2.0, 1.0, 600)
            test = h.gamma(2.0, 1.0, 20000)
            support = (0.0, np.inf)
        elif name == "beta":
            x = r.beta(2.0, 5.0, 600)
            test = h.beta(2.0, 5.0, 20000)
            support = (0.0, 1.0)
        elif name == "beta_mixture":
            x = np.r_[r.beta(2.0, 5.0, 300), r.beta(5.0, 2.0, 300)]
            z = h.random(20000) < 0.5
            test = h.beta(np.where(z, 2.0, 5.0), np.where(z, 5.0, 2.0))
            support = (0.0, 1.0)
        elif name == "uniform":
            x = r.uniform(size=600)
            test = h.uniform(size=20000)
            support = (0.0, 1.0)
        elif name == "four_mode":
            w = np.full(4, 0.25)
            m = np.array([-6.0, -2.0, 2.0, 6.0])
            s = np.array([0.4, 0.6, 0.5, 0.4])
            x = np.concatenate([r.normal(mu, sd, 150) for mu, sd in zip(m, s)])
            z = h.choice(4, size=20000, p=w)
            test = h.normal(m[z], s[z])
            extra.update(k_max=4)
        else:
            weights, means, scales = {
                "separated": ([0.5, 0.5], [-2, 3], [1, 1.5]),
                "overlap": ([0.5, 0.5], [-0.5, 0.5], [1, 1]),
                "unbalanced": ([0.9, 0.1], [-2, 3], [0.6, 1.5]),
            }[name]
            w, m, s = map(np.array, (weights, means, scales))
            x = np.concatenate(
                [r.normal(mu, sd, int(600 * mass)) for mass, mu, sd in zip(w, m, s)]
            )
            z = h.choice(2, size=20000, p=w)
            test = h.normal(m[z], s[z])
        start = time.perf_counter()
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 30)
            f = Distribution().fit(
                x, n_components="auto", rng=0, support=support, **extra
            )
            seconds = time.perf_counter() - start
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            row = dict(
                seed=seed,
                family=name,
                seconds=seconds,
                k=f.n_components,
                degrees=[int(c._basis.model.request.bulk_degree) for c in f.components],
                train_logpdf=float(np.mean(f.logpdf(x))),
                holdout_logpdf=float(np.mean(f.logpdf(test))),
                em=f._em_diagnostics,
                selection=f.selection_diagnostics,
                grid_logpdf=f.logpdf(
                    np.linspace(np.min(test), np.max(test), 301)
                ).tolist(),
            )
        except Exception as e:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0)
            row = dict(
                seed=seed,
                family=name,
                seconds=time.perf_counter() - start,
                error=repr(e),
            )
        records.append(row)
        Path(a.output).write_text(json.dumps(records, indent=2, default=str))
        print(
            seed,
            name,
            round(row["seconds"], 3),
            row.get("k"),
            row.get("error"),
            flush=True,
        )
