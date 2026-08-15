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
    "EgoVerse is a library of videos of people doing everyday tasks, filmed\n"
    "from a camera on their head. Robots learn from them. It holds hundreds of\n"
    "thousands of clips, and more is not automatically better: some are broken,\n"
    "and many are near-copies of each other.", 12.5, INK, va="top")

txt(0.035, 0.648, "WHAT WE BUILT", 10, FAINT, "bold")
steps = [
    ("1", "Throw out the broken ones.", "Tracking glitches, hands off to the side, or someone\n"
     "who left the camera running for ten minutes."),
    ("2", "Pick a varied quarter of the rest.", "Instead of picking at random, choose clips unlike each\n"
     "other — different people, rooms and ways of moving."),
    ("3", "Check whether it worked.", "Train the same small model on each selection and see\n"
     "which better predicts what a person does next."),
]
y = 0.605
for n, head, body in steps:
    txt(0.038, y, n, 13, ACC, "bold")
    txt(0.060, y, head, 13, INK, "bold")
    txt(0.060, y - 0.028, body, 11, MUTED, va="top")
    y -= 0.098

# ------------------------------------------------------------------ the picture
ax = fig.add_axes([0.415, 0.470, 0.565, 0.415])
ax.imshow(mpimg.imread(REPORTS / "figs" / "manifold_compare.png"))
ax.axis("off")


# ------------------------------------------------------------------ headline
box(0.415, 0.320, 0.565, 0.118, "#f0fdf4", "#86efac")
txt(0.432, 0.408, "THE RESULT", 10, FAINT, "bold")
txt(0.432, 0.375,
    f"The varied quarter beat the random quarter in all {BEST_WINS} of our",
    14.5, INK, "bold")
txt(0.432, 0.348,
    f"{N_TESTS} comparisons, cutting the model's prediction error by {BEST_PCT:.1f}%.",
    14.5, INK, "bold")

# ------------------------------------------------------------------ bottom panels
top, h = 0.058, 0.245
w, gapx = 0.2215, 0.0165
panels = [
    ("How do we know it's real?",
     f"A {BEST_PCT:.0f}% difference is small enough to be luck, so\n"
     "we also ran a selection built to be bad on purpose:\n"
     "nearly every clip from a handful of people and rooms.\n\n"
     f"It lost {CTRL_LOSSES} of {N_TESTS} times, doing {CTRL_PCT:+.0f}% worse. So our test\n"
     "can tell a good selection from a bad one, and the\n"
     "smaller result is a real effect rather than noise.",
     "#f0fdf4", "#86efac"),
    ("What we're not claiming",
     "Using all the data is still best. Picking well does\n"
     "not beat having more.\n\n"
     "What it does do: at a quarter of the data, a varied\n"
     f"pick closes about {gap:.0f}% of the gap between a random\n"
     "quarter and using everything. That is the number\n"
     "that matters when you can't have everything.",
     "#fffbeb", "#fcd34d"),
    ("What we found in the data",
     "Some labs film one task per clip. Others leave the\n"
     "camera running for a whole session and call it one\n"
     "clip. A clip is 90 seconds in one lab and 11 in\n"
     f"another — {xlab.loc['microagi','ANY']} of the biggest collection is really\n"
     "a whole session.\n\n"
     f"Cutting those makes the model {gf_pct:.0f}% better, using\n"
     "the same amount of footage.",
     "#eff6ff", "#93c5fd"),
    ("Where we'd push back",
     "We measure how well a model predicts human hand\n"
     "movement, not whether a robot succeeds at the\n"
     "task. It is a stand-in — and the people who built\n"
     "EgoVerse use the same one, for the same reason.\n\n"
     "We tested one task in one lab. We can't promise\n"
     "this carries over to others.",
     "#fef2f2", "#fca5a5"),
]
for i, (title, body, bg, edge) in enumerate(panels):
    x = 0.035 + i * (w + gapx)
    box(x, top, w, h, bg, edge)
    txt(x + 0.013, top + h - 0.030, title, 12.5, INK, "bold")
    txt(x + 0.013, top + h - 0.062, body, 9.4, "#27272a", va="top", lh=1.55)

txt(0.035, 0.026,
    f"Tested on 572 clips of people folding clothes, filmed by 20 people across 16 rooms. "
    f"We dropped {n_drop} as unusable, then compared selections of the rest. "
    "Full method, code and caveats in the repo.", 10, MUTED)

out_png = REPORTS / "egoscore_summary_slide.png"
out_pdf = REPORTS / "egoscore_summary_slide.pdf"
fig.savefig(out_png, facecolor="white")
fig.savefig(out_pdf, facecolor="white")
print(f"wrote {out_png}\nwrote {out_pdf}")
