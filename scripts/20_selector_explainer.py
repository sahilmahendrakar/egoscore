"""Teach what each selector does, on a toy cloud where the right answer is visible.

The real 41-dimensional embedding is impossible to reason about by eye. A synthetic 2D
cloud with deliberately uneven cluster sizes makes each algorithm's *character* obvious:
random over-samples whatever is dense, facility location and k-center spread out for
different reasons, and log-det chases the extremes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from egoscore.select import dpp_logdet, facility_location, kcenter, random_select

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

SEL, POOL = "#2a78d6", "#c9c9c4"

# A dense blob, a medium blob, and two small outlying pockets -- the same imbalance a
# real dataset has when one operator recorded far more episodes than the rest.
rng = np.random.default_rng(3)
Z = np.vstack([
    rng.normal([0.0, 0.0], 0.42, (240, 2)),     # dominant cluster
    rng.normal([2.6, 1.4], 0.30, (70, 2)),      # medium cluster
    rng.normal([-2.2, 1.9], 0.18, (22, 2)),     # small pocket
    rng.normal([1.4, -2.3], 0.16, (14, 2)),     # small pocket
])
K = 26

PANELS = [
    ("random", random_select,
     "Samples uniformly. Whatever is\nover-represented stays\nover-represented."),
    ("facility location", facility_location,
     "Picks representatives so every\npoint has one nearby.\nCoverage, weighted by density."),
    ("k-center", kcenter,
     "Repeatedly takes the point\nfurthest from everything chosen.\nMinimises the worst-case gap."),
    ("dpp (log-det)", dpp_logdet,
     "Maximises the volume spanned\nby the chosen set.\nRewards mutual dissimilarity."),
]

fig, axes = plt.subplots(1, 4, figsize=(16.5, 5.9), facecolor="white")
for ax, (name, fn, blurb) in zip(axes, PANELS):
    idx = fn(Z, K, seed=0)
    mask = np.zeros(len(Z), bool)
    mask[idx] = True

    ax.scatter(Z[~mask, 0], Z[~mask, 1], s=17, c=POOL, linewidths=0, zorder=1)
    ax.scatter(Z[mask, 0], Z[mask, 1], s=74, c=SEL, edgecolors="white", linewidths=1.4, zorder=3)

    # How many of the K landed in the dominant cluster, i.e. wasted on redundancy.
    in_dense = int(mask[:240].sum())
    ax.set_title(name, fontsize=15, fontweight="bold", color="#14161a", pad=26)
    ax.text(0.5, 1.035, f"{in_dense} of {K} picks in the dense cluster",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, color="#2a78d6", fontweight="bold")
    ax.text(0.5, -0.06, blurb, transform=ax.transAxes, ha="center", va="top",
            fontsize=10.8, color="#52514e", linespacing=1.55)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(Z[:, 0].min() - 0.4, Z[:, 0].max() + 0.4)
    ax.set_ylim(Z[:, 1].min() - 0.4, Z[:, 1].max() + 0.4)
    for s in ax.spines.values():
        s.set_color("#e0e0dc")

fig.suptitle(f"Four ways to spend the same budget of {K} picks on {len(Z)} points",
             fontsize=16.5, fontweight="bold", color="#14161a", y=0.995)
fig.tight_layout(rect=[0, 0.13, 1, 0.93])
out = FIGS / "selectors_explained.png"
fig.savefig(out, dpi=145, facecolor="white")
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
