"""Generate docs/images/readme-examples.png for the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

from gibbus import Distribution

FIT, TRUE, HIST, CENS, COMP = "#1f5fa8", "#444444", "#d9d9d9", "#efd9b8", "#7aa6d6"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, ax = plt.subplots(2, 2, figsize=(10, 6.8))


def draw(a, grid, fit, true, title):
    a.plot(grid, true, color=TRUE, ls="--", lw=1.3)
    a.plot(grid, fit, color=FIT, lw=2)
    a.set_title(title, loc="left", fontsize=10.5)
    a.set_yticks([])


# Flat-topped, light-tailed: generalized normal with shape 4
dist = stats.gennorm(4)
x = dist.rvs(1000, random_state=np.random.default_rng(4))
model = Distribution().fit(x)
g = np.linspace(-2, 2, 400)
a = ax[0, 0]
a.hist(x, bins=30, range=(-2, 2), density=True, color=HIST)
draw(
    a, g, model.pdf(g), dist.pdf(g), "Flat-topped: generalized normal (shape 4), n=1000"
)

# Mixture of left-skewed, right-skewed and symmetric components
parts = [
    (stats.skewnorm(-7, loc=-2.5, scale=1.6), 450),
    (stats.gumbel_r(0.8, 0.7), 600),
    (stats.norm(5.5, 0.8), 450),
]
rng = np.random.default_rng(4)
x = np.concatenate([d.rvs(n, random_state=rng) for d, n in parts])
model = Distribution().fit(x, n_components="auto", rng=0)
g = np.linspace(-8, 9, 500)
true = sum(n * d.pdf(g) for d, n in parts) / sum(n for _, n in parts)
a = ax[0, 1]
a.hist(x, bins=60, range=(-8, 9), density=True, color=HIST)
for w, comp in zip(model.weights, model.components):
    a.plot(g, w * comp.pdf(g), color=COMP, lw=1.2, ls=(0, (3, 2)))
draw(
    a,
    g,
    model.pdf(g),
    true,
    f"Mixture: skew-normal + Gumbel + normal, n=1500, K={model.n_components} (auto)",
)
a.legend(
    [Line2D([], [], color=COMP, lw=1.2, ls=(0, (3, 2)))],
    ["fitted components"],
    frameon=False,
    fontsize=8.5,
    loc="upper left",
)

# Bounded, left-skewed, steep drop at the upper endpoint
dist = stats.beta(5, 1.4)
x = dist.rvs(1000, random_state=np.random.default_rng(1))
model = Distribution().fit(
    x, support=(0, 1), log_boundary_lower=True, log_boundary_upper=True
)
g = np.linspace(0.0005, 0.9995, 600)
a = ax[1, 0]
a.hist(x, bins=30, range=(0, 1), density=True, color=HIST)
draw(a, g, model.pdf(g), dist.pdf(g), "Bounded on (0, 1): beta(5, 1.4), n=1000")
a.set_xlim(0, 1)

# Interval-censored Weibull with heavy right censoring
weibull = stats.weibull_min(1.5, scale=3)
t = weibull.rvs(1000, random_state=np.random.default_rng(2))
width, cutoff, xmax = 0.5, 3.5, 9.0
lower = np.floor(t / width) * width
intervals = np.column_stack([lower, lower + width])
censored = t > cutoff
intervals[censored] = [cutoff, np.inf]
model = Distribution().fit(intervals, support=(0, np.inf), log_boundary_lower=True)
g = np.linspace(0, xmax, 400)
a = ax[1, 1]
edges = np.arange(0, cutoff + width, width)
counts = np.histogram(lower[~censored], bins=edges)[0]
a.bar(
    edges[:-1] + width / 2,
    counts / (len(t) * width),
    width=width,
    color=HIST,
    edgecolor="white",
    label="binned observations",
)
a.bar(
    (cutoff + xmax) / 2,
    censored.mean() / (xmax - cutoff),
    width=xmax - cutoff,
    color=CENS,
    edgecolor="white",
    label=f"right-censored at {cutoff} ({censored.mean():.0%})",
)
draw(
    a,
    g,
    model.pdf(g),
    weibull.pdf(g),
    f"Interval-censored: Weibull, n=1000, bins of {width}",
)
a.legend(frameon=False, fontsize=8.5)
a.set_xlim(0, xmax)

fig.legend(
    [Line2D([], [], color=TRUE, ls="--", lw=1.3), Line2D([], [], color=FIT, lw=2)],
    ["true distribution", "gibbus fit"],
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.0),
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("readme-examples.png", dpi=150)
