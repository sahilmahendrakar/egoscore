"""Paired analysis + plots.

The across-seed standard deviation is dominated by *split* variance: each seed draws a
different held-out operator/scene set, which shifts absolute MSE for every condition at
once. Comparing raw means with those error bars would understate the effect badly.

Every condition within a seed sees the identical split and the identical eval windows,
so the correct statistic is the paired difference against `random` at the same
(seed, k_frac). That is what we report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = ROOT / "reports" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

res = pd.read_csv(REPORTS / "results.csv")
METRICS = ["mse_unseen_operator", "mse_unseen_scene"]
SUBSET_CONDS = ["random", "curated", "dpp", "kcenter", "degenerate", "random_nogate"]

# ------------------------------------------------------------------ paired deltas
base = (
    res[res.condition == "random"]
    .set_index(["seed", "k_frac"])[METRICS]
    .rename(columns={m: f"base_{m}" for m in METRICS})
)
sub = res[res.condition.isin(SUBSET_CONDS)].join(base, on=["seed", "k_frac"])

rows = []
for m in METRICS:
    sub[f"delta_{m}"] = sub[m] - sub[f"base_{m}"]
    sub[f"pct_{m}"] = 100.0 * sub[f"delta_{m}"] / sub[f"base_{m}"]

for (cond, kf), g in sub.groupby(["condition", "k_frac"]):
    r = {"condition": cond, "k_frac": kf, "n_pairs": len(g)}
    for m in METRICS:
        r[f"{m}_mean_pct"] = g[f"pct_{m}"].mean()
        r[f"{m}_wins"] = int((g[f"delta_{m}"] < 0).sum())
    rows.append(r)
paired = pd.DataFrame(rows).sort_values(["k_frac", "condition"])
paired.to_csv(REPORTS / "paired_deltas.csv", index=False)

print("=== paired vs random, same seed & budget (negative % = better) ===")
print(paired.to_string(index=False))

# Sign test + Wilcoxon signed-rank across all paired comparisons.
from scipy.stats import binomtest, wilcoxon

n_seeds = res["seed"].nunique()
print(f"\n=== overall vs random (both metrics, both budgets, {n_seeds} seeds) ===")
overall = []
for cond in ["curated", "dpp", "kcenter", "degenerate", "random_nogate"]:
    g = sub[sub.condition == cond]
    deltas = np.concatenate([g[f"delta_{m}"].to_numpy() for m in METRICS])
    pcts = np.concatenate([g[f"pct_{m}"].to_numpy() for m in METRICS])
    wins = int((deltas < 0).sum())
    tot = len(deltas)
    p_sign = binomtest(wins, tot, 0.5).pvalue
    # Two-sided Wilcoxon on the paired differences; more powerful than the sign test
    # because it uses the magnitudes, not just the directions.
    p_wil = wilcoxon(deltas).pvalue if np.any(deltas != 0) else float("nan")
    overall.append({"condition": cond, "wins": wins, "n": tot,
                    "mean_pct": pcts.mean(), "p_sign": p_sign, "p_wilcoxon": p_wil})
    print(f"  {cond:14s} {wins:2d}/{tot:2d} wins   mean {pcts.mean():+.2f}%   "
          f"sign p={p_sign:.2e}   wilcoxon p={p_wil:.2e}")
pd.DataFrame(overall).to_csv(REPORTS / "significance.csv", index=False)

# ------------------------------------------------------------------ figure 1
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
order = ["degenerate", "random_nogate", "random", "curated", "dpp", "kcenter"]
colors = {"degenerate": "#c0392b", "random_nogate": "#e67e22", "random": "#7f8c8d",
          "curated": "#2980b9", "dpp": "#16a085", "kcenter": "#8e44ad"}

for ax, m in zip(axes, METRICS):
    for kf, marker in [(0.25, "o"), (0.5, "s")]:
        d = res[(res.k_frac == kf) & res.condition.isin(order)]
        means = [d[d.condition == c][m].mean() for c in order]
        errs = [d[d.condition == c][m].std() for c in order]
        x = np.arange(len(order)) + (0.16 if kf == 0.5 else -0.16)
        ax.errorbar(x, means, yerr=errs, fmt=marker, capsize=3, ms=6,
                    label=f"K = {kf:.0%} of gated pool",
                    color="#2c3e50" if kf == 0.25 else "#95a5a6")
    ceiling = res[res.condition == "all_gated"][m].mean()
    ax.axhline(ceiling, ls="--", lw=1.2, color="#27ae60", label="all gated data (100%)")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Avg-MSE (lower is better)")
    ax.set_title(m.replace("mse_", "").replace("_", " "))
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8)
fig.suptitle("EgoScore — subset selection vs. Avg-MSE proxy (rl2 fold_clothes, 3 seeds)", y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "conditions.png", dpi=150, bbox_inches="tight")
print(f"\nwrote {FIGS/'conditions.png'}")

# ------------------------------------------------------------------ figure 2: paired
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, m in zip(axes, METRICS):
    conds = ["degenerate", "random_nogate", "curated", "dpp", "kcenter"]
    for i, cond in enumerate(conds):
        g = sub[sub.condition == cond]
        vals = g[f"pct_{m}"].to_numpy()
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.1, 0.1, len(vals)), vals,
                   s=28, color=colors[cond], alpha=0.8, zorder=3)
        ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=colors[cond], lw=2.5, zorder=4)
    ax.axhline(0, color="#7f8c8d", lw=1.2, ls="-")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=20, ha="right")
    ax.set_ylabel("% change in Avg-MSE vs random\n(negative = better)")
    ax.set_title(m.replace("mse_", "").replace("_", " "))
    ax.grid(alpha=0.25, axis="y")
fig.suptitle("Paired against random at identical seed and budget (each dot = one seed x budget)", y=1.03)
fig.tight_layout()
fig.savefig(FIGS / "paired.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGS/'paired.png'}")

# ------------------------------------------------------------------ figure 3: diversity
fig, ax = plt.subplots(figsize=(6.4, 4.4))
d = res[res.condition.isin(["random", "curated", "dpp", "kcenter", "degenerate"])]
for cond in ["degenerate", "random", "curated", "dpp", "kcenter"]:
    g = d[d.condition == cond]
    ax.scatter(g["n_groups"], g["mse_unseen_operator"], s=55, label=cond,
               color=colors[cond], alpha=0.85, edgecolor="white", lw=0.6)
ax.set_xlabel("distinct operator x scene groups in the training subset")
ax.set_ylabel("Avg-MSE, unseen operators")
ax.set_title("Group coverage tracks generalization")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "coverage.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGS/'coverage.png'}")

# ------------------------------------------------------------------ gate figure
feat = pd.read_csv(REPORTS / "keep_drop.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(feat["oof_max"], bins=50, color="#2980b9")
axes[0].axvline(0.5, color="#c0392b", ls="--", label="drop threshold")
axes[0].set_xlabel("fraction of frames with a hand outside the image")
axes[0].set_ylabel("episodes")
axes[0].set_title("hands out of frame")
axes[0].legend(fontsize=8)
axes[1].hist(feat["duration_s"], bins=50, color="#16a085")
axes[1].axvline(feat["duration_s"].quantile(0.99), color="#c0392b", ls="--", label="p99 threshold")
axes[1].set_xlabel("episode duration (s)")
axes[1].set_title("duration")
axes[1].legend(fontsize=8)
fig.suptitle(f"Quality gate on rl2 fold_clothes — {(~feat['keep']).sum()}/{len(feat)} episodes dropped", y=1.02)
fig.tight_layout()
fig.savefig(FIGS / "gate.png", dpi=150, bbox_inches="tight")
print(f"wrote {FIGS/'gate.png'}")
