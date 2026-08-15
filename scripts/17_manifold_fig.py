"""Side-by-side of the episode manifold under a diversity selector vs the positive control.

This is the single most explanatory picture in the project: the same budget, spent two
ways, and you can see the difference before reading a single number.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

D = json.loads((REPORTS / "demo_data.json").read_text())["datasets"]["rl2"]
eps = D["episodes"]
xy = np.array([[e["x"], e["y"]] for e in eps])

SEL, POOL, DROP = "#2a78d6", "#c9c9c4", "#e34948"
PANELS = [("kcenter", "k-center — coverage-maximizing"),
          ("degenerate", "degenerate — the positive control")]

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor="white")
for ax, (cond, title) in zip(axes, PANELS):
    idx = set(D["selections"][cond]["idx"])
    mask = np.array([i in idx for i in range(len(eps))])

    ax.scatter(xy[~mask, 0], xy[~mask, 1], s=26, c=POOL, linewidths=0, zorder=1)
    ax.scatter(xy[mask, 0], xy[mask, 1], s=54, c=SEL, edgecolors="white",
               linewidths=1.1, zorder=3)

    g = D["selections"][cond]["n_groups"]
    pct = D["mse"][cond]["pct_op"]
    ax.set_title(title, fontsize=14, fontweight="bold", color="#14161a", pad=12)
    ax.text(0.5, -0.045, f"{g} of 190 operator×scene groups covered",
            transform=ax.transAxes, ha="center", va="top", fontsize=11.5,
            color="#52514e")
    ax.text(0.5, -0.115, f"Avg-MSE {pct:+.2f}% vs random",
            transform=ax.transAxes, ha="center", va="top", fontsize=13,
            color="#1baf7a" if pct < 0 else "#e34948", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e0e0dc")

fig.suptitle("Same budget — 135 of 543 episodes — spent two ways",
             fontsize=16, fontweight="bold", color="#14161a", y=0.99)
fig.text(0.5, 0.015,
         "Each dot is one fold_clothes episode, positioned by 41 trajectory features (t-SNE). "
         "Axes carry no units: read clustering, not position.",
         ha="center", fontsize=10.5, color="#7c7b76")
fig.tight_layout(rect=[0, 0.10, 1, 0.96])
out = FIGS / "manifold_compare.png"
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
