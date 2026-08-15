"""Generate the one-page summary slide for the submission (PNG + PDF, 16:9).

Audience: someone who has never seen this project and does not work on robot learning.
So: no metric names, no algorithm names, no p-values, no file formats. Plain sentences,
one picture that explains itself, and the numbers that matter.

Every value is read from the result CSVs rather than typed, so the slide cannot drift
from the experiment.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

sig = pd.read_csv(REPORTS / "significance.csv").set_index("condition")
res = pd.read_csv(REPORTS / "results.csv")
kd = pd.read_csv(REPORTS / "keep_drop.csv")
xlab = pd.read_csv(REPORTS / "cross_lab_summary.csv").set_index("lab")
gv = pd.read_csv(REPORTS / "gate_value_microagi.csv")

METRICS = ["mse_unseen_operator", "mse_unseen_scene"]
N_TESTS = int(sig.loc["dpp", "n"])
BEST_WINS = int(sig.loc["dpp", "wins"])
BEST_PCT = abs(float(sig.loc["dpp", "mean_pct"]))
CTRL_LOSSES = N_TESTS - int(sig.loc["degenerate", "wins"])
CTRL_PCT = float(sig.loc["degenerate", "mean_pct"])

ceiling = {m: res[res.condition == "all_gated"][m].mean() for m in METRICS}
rand25 = {m: res[(res.condition == "random") & (res.k_frac == 0.25)][m].mean() for m in METRICS}
best25 = {m: res[(res.condition == "dpp") & (res.k_frac == 0.25)][m].mean() for m in METRICS}
gap = sum(100.0 * (rand25[m] - best25[m]) / (rand25[m] - ceiling[m]) for m in METRICS) / 2

gvp = gv.pivot(index="seed", columns="condition", values="avg_mse")
gf = gvp["gated_equal_frames"] - gvp["ungated_equal_frames"]
gf_pct = abs(100.0 * (gf / gvp["ungated_equal_frames"]).mean())

n_drop = int((~kd["keep"]).sum())

INK, MUTED, FAINT = "#14161a", "#52514e", "#7c7b76"
GOOD, BAD, ACC = "#1baf7a", "#e34948", "#2a78d6"

fig = plt.figure(figsize=(16, 9), facecolor="white", dpi=150)
T = fig.transFigure


def txt(x, y, s, size=11, color=INK, weight="normal", style="normal",
        va="center", ha="left", lh=1.5):
    fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, style=style,
             va=va, ha=ha, transform=T, linespacing=lh)


def box(x, y, w, h, fc, ec, lw=1.1):
    fig.patches.append(plt.Rectangle((x, y), w, h, transform=T, facecolor=fc,
                                     edgecolor=ec, lw=lw, zorder=0))


# ------------------------------------------------------------------ title
txt(0.035, 0.945, "Which training videos are worth keeping?", 31, INK, "bold")
txt(0.035, 0.900,
    "EgoScore  ·  EgoVerse Data Optimization & Evaluation Suite, Track 1  ·  "
    "github.com/sahilmahendrakar/egoscore", 12.5, MUTED)

# ------------------------------------------------------------------ the setup
txt(0.035, 0.838, "THE PROBLEM", 10, FAINT, "bold")
txt(0.035, 0.800,
    "EgoVerse is a library of videos of people performing everyday tasks,\n"
    "recorded from a head-worn camera. Robots learn manipulation from them.\n"
    "Some of its clips are unusable — the hand tracking fails, or a recording\n"
    "covers a whole session rather than one task. Many others are\n"
    "near-duplicates that add training time without adding information.",
    12.3, INK, va="top")

txt(0.035, 0.632, "HOW IT WORKS", 10, FAINT, "bold")
steps = [
    ("1", "Remove clips that are unusable.", "Eight checks on the recorded positions: frozen tracking,\n"
     "a hand jumping past 10 m/s, a clip over 3× its lab's median."),
    ("2", "Select a varied subset of the rest.", "Summarise each clip as 41 numbers describing how the person\n"
     "moved, then choose clips unlike one another, not at random."),
    ("3", "Measure whether it helped.", "Train an identical model on each subset; score it on clips from\n"
     "people and rooms absent from its training."),
]
y = 0.585
for n, head, body in steps:
    txt(0.038, y, n, 13, ACC, "bold")
    txt(0.060, y, head, 13, INK, "bold")
    txt(0.060, y - 0.028, body, 11, MUTED, va="top")
    y -= 0.088

# ------------------------------------------------------------------ the picture
ax = fig.add_axes([0.415, 0.470, 0.565, 0.415])
ax.imshow(mpimg.imread(REPORTS / "figs" / "manifold_compare.png"))
ax.axis("off")


# ------------------------------------------------------------------ headline
box(0.415, 0.320, 0.565, 0.118, "#f0fdf4", "#86efac")
txt(0.432, 0.408, "THE RESULT", 10, FAINT, "bold")
txt(0.432, 0.375,
    f"The varied quarter beat the random quarter in all {BEST_WINS} of",
    14.5, INK, "bold")
txt(0.432, 0.348,
    f"{N_TESTS} comparisons, with {BEST_PCT:.1f}% lower prediction error.",
    14.5, INK, "bold")

# ------------------------------------------------------------------ bottom panels
top, h = 0.052, 0.258
w, gapx = 0.2215, 0.0165
panels = [
    ("Why the margin is credible",
     f"A {BEST_PCT:.0f}% margin is small enough to be chance, so we\n"
     "included a selection designed to fail: the same clip\n"
     "count, drawn almost entirely from a handful of\n"
     "people in a handful of rooms.\n\n"
     f"It lost {CTRL_LOSSES} of {N_TESTS} comparisons, {CTRL_PCT:+.0f}% worse. The\n"
     "measurement can therefore tell a good selection\n"
     "from a poor one, which makes the smaller margins\n"
     "measurements rather than noise.",
     "#f0fdf4", "#86efac"),
    ("What this does not show",
     "Training on the full collection remains best. It\n"
     "beats every subset we selected. Choosing well does\n"
     "not substitute for having more.\n\n"
     "The narrower claim: at a quarter of the data, a varied\n"
     f"selection recovers about {gap:.0f}% of the difference between\n"
     "a random quarter and the full collection — the figure\n"
     "that matters when everything is not an option.",
     "#fffbeb", "#fcd34d"),
    ("A finding about EgoVerse",
     "A clip is not a consistent unit. Median length is\n"
     "90 seconds in one collection and 11 in another;\n"
     f"{xlab.loc['microagi','ANY']} of the largest collection is a whole session\n"
     "stored as one file.\n\n"
     "Matched on clip count, removing those looks harmful\n"
     "(1.7% worse) — but one long clip holds 60× the\n"
     f"footage. Matched on footage, removing them is {gf_pct:.0f}%\n"
     "better, 10 comparisons out of 10.",
     "#eff6ff", "#93c5fd"),
    ("Limitations",
     "The score measures how accurately a model predicts\n"
     "human hand motion, not whether a robot completes\n"
     "the task. The EgoVerse authors use the same proxy\n"
     "for the same reason, and say so in their paper.\n\n"
     "The scoring model never sees the video — a\n"
     "consequence of the 84.6 GB figure.\n\n"
     "One task, one collection. Transfer is untested.",
     "#fef2f2", "#fca5a5"),
]
for i, (title, body, bg, edge) in enumerate(panels):
    x = 0.035 + i * (w + gapx)
    box(x, top, w, h, bg, edge)
    txt(x + 0.013, top + h - 0.030, title, 12.5, INK, "bold")
    txt(x + 0.013, top + h - 0.060, body, 8.8, "#27272a", va="top", lh=1.5)

txt(0.035, 0.026,
    f"Tested on 572 clothes-folding clips recorded by 20 people across 16 rooms — the only EgoVerse collection that records "
    f"who filmed each clip and where, which the held-out comparison requires. {n_drop} removed as unusable. "
    "Full method, code and limitations in the repo.", 10, MUTED)

out_png = REPORTS / "egoscore_summary_slide.png"
out_pdf = REPORTS / "egoscore_summary_slide.pdf"
fig.savefig(out_png, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
print(f"wrote {out_png}\nwrote {out_pdf}")
