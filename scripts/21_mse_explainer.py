"""Explain Avg-MSE concretely, on a real held-out episode.

Avg-MSE is abstract until you see what is actually being predicted. This draws the real
task: ten frames of hand-pose history go in, a thirty-step future trajectory comes out,
and the error is the average squared gap between the predicted trajectory and what the
human actually did.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from egoscore.gate import apply_gate
from egoscore.proxy import HISTORY, HORIZON, RidgeChunkPolicy, episode_windows

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"
POSES = ROOT / "data" / "poses"

INK, MUTED = "#14161a", "#52514e"
TRUE, PRED, HIST, ERR = "#14161a", "#2a78d6", "#1baf7a", "#e34948"

feat = pd.read_csv(REPORTS / "features.csv")
gated = apply_gate(feat)
pool = gated[gated["keep"]]["episode_hash"].tolist()

# Train on most of the pool, then illustrate on a genuinely held-out episode.
rng = np.random.default_rng(0)
rng.shuffle(pool)
train_h, demo_h = pool[:400], pool[400]

Xs, Ys = [], []
for h in train_h:
    f = POSES / f"{h}.npz"
    if not f.exists():
        continue
    with np.load(f, allow_pickle=True) as z:
        X, Y = episode_windows({k: z[k] for k in z.files})
    if len(X):
        Xs.append(X)
        Ys.append(Y)
model = RidgeChunkPolicy(alpha=1.0, seed=0).fit(np.vstack(Xs).astype(np.float32),
                                                np.vstack(Ys).astype(np.float32))

with np.load(POSES / f"{demo_h}.npz", allow_pickle=True) as z:
    ep = {k: z[k] for k in z.files}
X, Y = episode_windows(ep)
P = model.predict(X)

# Pick a window with visible motion, so the figure shows a trajectory rather than a flat line.
motion = np.abs(Y).mean(axis=1)
w = int(np.argsort(motion)[int(0.85 * len(motion))])

# Y layout: HORIZON steps x 14 dims (left xyz+quat, right xyz+quat). Show left-hand x/y/z.
true_chunk = Y[w].reshape(HORIZON, 14)[:, :3]
pred_chunk = P[w].reshape(HORIZON, 14)[:, :3]

raw = np.nan_to_num(ep["left__obs_ee_pose"][:, :3])
start = HISTORY + w * 15
hist = raw[start - HISTORY:start] - raw[start - 1]

fig = plt.figure(figsize=(14, 6.4), facecolor="white")
gs = fig.add_gridspec(3, 2, width_ratios=[2.05, 1], hspace=0.32, wspace=0.22)

axis_names = ["x", "y", "z"]
t_hist = np.arange(-HISTORY, 0)
t_fut = np.arange(0, HORIZON)

for i in range(3):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(t_hist, hist[:, i], color=HIST, lw=2.4, label="history in (10 frames)" if i == 0 else None)
    ax.plot(t_fut, true_chunk[:, i], color=TRUE, lw=2.4, label="what the human did" if i == 0 else None)
    ax.plot(t_fut, pred_chunk[:, i], color=PRED, lw=2.4, ls="--", label="what the model predicted" if i == 0 else None)
    ax.fill_between(t_fut, true_chunk[:, i], pred_chunk[:, i], color=ERR, alpha=0.18,
                    label="squared error lives here" if i == 0 else None)
    ax.axvline(0, color="#c8c8c2", lw=1)
    ax.set_ylabel(f"left hand {axis_names[i]}  (m)", fontsize=10.5, color=MUTED)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=9, colors=MUTED)
    for s in ax.spines.values():
        s.set_color("#e0e0dc")
    if i == 0:
        ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    if i < 2:
        ax.set_xticklabels([])
ax.set_xlabel("frames from the prediction point  (30 fps)", fontsize=10.5, color=MUTED)

# ---- right: the arithmetic ----
ax = fig.add_subplot(gs[:, 1])
ax.axis("off")
err = float(np.mean((Y[w] - P[w]) ** 2))
ep_err = float(np.mean((Y - P) ** 2))

lines = [
    ("The task", "From 10 frames of hand-pose history,\npredict the next 30 frames (1 second)\nof both hands' motion.", INK),
    ("The error", "For every one of the 30 × 14 numbers,\ntake (predicted − actual), square it,\nand average.", INK),
    ("This window", f"{err:.5f}", PRED),
    ("This episode", f"{ep_err:.5f}   ({len(X)} windows)", PRED),
    ("Avg-MSE", "the same average over every window\nof every held-out episode.", INK),
]
y = 0.97
for label, body, color in lines:
    ax.text(0, y, label.upper(), transform=ax.transAxes, fontsize=10, color="#7c7b76",
            fontweight="bold", family="monospace", va="top")
    y -= 0.055
    ax.text(0, y, body, transform=ax.transAxes, fontsize=12.5 if body[0].isdigit() else 11.5,
            color=color, va="top", linespacing=1.5,
            fontweight="bold" if body[0].isdigit() else "normal")
    y -= 0.075 + 0.05 * body.count("\n")

ax.text(0, y - 0.02,
        "Lower is better. It is a proxy: it measures how\nwell a policy reproduces human motion, not\nwhether a robot would succeed.",
        transform=ax.transAxes, fontsize=11, color=MUTED, va="top", linespacing=1.5, style="italic")

fig.suptitle("What Avg-MSE actually measures", fontsize=17, fontweight="bold", color=INK, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.945])
out = FIGS / "mse_explained.png"
fig.savefig(out, dpi=145, facecolor="white")
print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
print(f"window MSE={err:.5f}  episode MSE={ep_err:.5f}")
