"""Contact sheets of what the quality gate keeps and what it drops.

Rendering these frames is how we caught a real bug. An earlier gate had a rule that
projected hand keypoints into the image and flagged "hands out of frame"; the frames of
the episodes it dropped showed hands in plain view. Two different projection models both
failed (scripts/22_projection_check.py), so we removed the rule rather than guess again.
Looking at the pictures is what surfaced it. No amount of staring at the histogram would
have.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "data" / "strips"
OUT.mkdir(parents=True, exist_ok=True)

N_FRAMES = 5
FRAME_W = 300

# Which lab to illustrate. rl2 is almost perfectly clean, so its gate figure is boring by
# construction; microagi is where the gate actually fires.
LAB = sys.argv[1] if len(sys.argv) > 1 else "microagi"

if LAB == "handcheck":
    # Verification pass: the episodes MediaPipe says have no visible hands at all.
    hv = pd.read_csv(REPORTS / "hand_visibility_rl2.csv")
    kd = pd.read_csv(REPORTS / "keep_drop.csv").merge(hv, on="episode_hash")
    kd["keep"] = kd["hand_visible_frac"] > 0.0
    kd["drop_reason"] = np.where(kd["keep"], "", "mediapipe_no_hands")
    kd = kd.set_index("episode_hash")
elif LAB == "rl2":
    kd = pd.read_csv(REPORTS / "keep_drop.csv").set_index("episode_hash")
else:
    xl = pd.read_csv(REPORTS / "cross_lab_quality.csv")
    kd = xl[xl["lab"] == LAB].copy()
    kd["keep"] = ~kd["drop_rig_independent"]
    rule_cols = [c for c in kd.columns if c.startswith("rule_")]
    kd["drop_reason"] = [
        "|".join(c[5:] for c in rule_cols if r[c]) for _, r in kd.iterrows()
    ]
    kd = kd.set_index("episode_hash")

meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")

# Four rows keeps the composite landscape, which is what a 16:9 slide can actually show.
med = kd["duration_s"].median()
if LAB == "handcheck":
    kept = kd[kd["hand_visible_frac"] >= 1.0].head(1).index.tolist()
    dropped = kd[~kd["keep"]].head(3).index.tolist()
else:
    kept = (kd[kd["keep"]]
            .assign(_d=lambda d: (d.duration_s - med).abs())
            .nsmallest(2, "_d").index.tolist())
    dropped = kd[~kd["keep"]].nlargest(2, "duration_s").index.tolist()
    if len(dropped) < 2:
        dropped += kd[~kd["keep"]].nsmallest(2 - len(dropped), "duration_s").index.tolist()

ROWS = ([(h, "KEPT", "#1baf7a") for h in kept]
        + [(h, "TOO LONG", "#e34948") for h in dropped])
print(f"rendering {len(ROWS)} episodes")

creds = load_creds()
fs = r2_fs(creds)


def strip(ep: str):
    dest = OUT / f"row_{ep}.png"
    if dest.exists():
        return
    path = meta.loc[ep, "zarr_processed_path"]
    g = zarr.open_group(zarr.storage.FsspecStore(fs, path=path.replace("s3://", "")), mode="r")
    arr = g["images.front_1"]
    n = int(g.attrs.get("total_frames", arr.shape[0]))
    idx = np.linspace(int(0.08 * n), int(0.92 * n), N_FRAMES).astype(int)
    tiles = []
    for i in idx:
        blob = arr[int(i):int(i) + 1][0]
        img = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        tile = cv2.resize(img, (FRAME_W, int(h * FRAME_W / w)), interpolation=cv2.INTER_AREA)
        # Stamp the timestamp on every frame. Without it there is no way to see that the
        # five frames of a dropped episode span thirteen minutes rather than eleven seconds,
        # which is the entire reason it was dropped.
        label = f"t = {i / 30.0:.0f}s"
        cv2.rectangle(tile, (0, 0), (94, 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    gap = 6
    H = max(t.shape[0] for t in tiles)
    W = sum(t.shape[1] for t in tiles) + gap * (len(tiles) - 1)
    canvas = np.full((H, W, 3), 255, np.uint8)
    x = 0
    for t in tiles:
        canvas[: t.shape[0], x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    cv2.imwrite(str(dest), canvas)


t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(strip, [h for h, _, _ in ROWS]))
print(f"frames in {time.time()-t0:.0f}s")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def caption(ep, verdict):
    r = kd.loc[ep]
    if LAB == "handcheck":
        return (f"MediaPipe detected a hand in {100*r.hand_visible_frac:.0f}% of 8 sampled frames "
                f"\u00b7 {r.duration_s:.0f}s")
    if verdict == "KEPT":
        return (f"One garment, start to finish. Frames span {r.duration_s:.0f}s.")
    reason = str(r.drop_reason)
    if "runaway_length" in reason:
        return (f"Frames span {r.duration_s/60:.0f} minutes and the garment changes between "
                f"each one. This is a whole folding session labelled as one demonstration.")
    if "suspiciously_short" in reason:
        return (f"{r.duration_s:.0f}s — under 15% of the median of {med:.0f}s. A fragment · dropped")
    return f"{reason} · {r.duration_s:.0f}s · dropped"


# Layout: a duration bar beside every strip. The drop reason is *length*, and without
# seeing the lengths side by side a reader looks at the frames, sees hands, and reasonably
# concludes the label is wrong. The bar makes the actual reason the visual subject.
from matplotlib.gridspec import GridSpec

durs = [float(kd.loc[ep, "duration_s"]) for ep, _, _ in ROWS]
dmax = max(durs) * 1.45   # headroom so the value label sits clear of the longest bar
thresh = 3.0 * med

fig = plt.figure(figsize=(15.4, 2.05 * len(ROWS)), facecolor="white")
gs = GridSpec(len(ROWS), 2, figure=fig, width_ratios=[1, 3.6], wspace=0.06, hspace=0.42)

for r, (ep, verdict, color) in enumerate(ROWS):
    dur = float(kd.loc[ep, "duration_s"])

    axb = fig.add_subplot(gs[r, 0])
    axb.barh([0], [dur], height=0.40, color=color, zorder=3)
    axb.axvline(thresh, color="#e34948", ls="--", lw=1.4, zorder=4)
    axb.set_xlim(0, dmax)
    axb.set_ylim(-0.6, 0.75)
    axb.set_yticks([])
    axb.tick_params(labelsize=8.5, colors="#7c7b76")
    # Ticks only on the bottom row; the axis is shared conceptually and repeating it four
    # times just crowds the short bars.
    if r == len(ROWS) - 1:
        axb.set_xticks([thresh, round(dmax / 100) * 100])
        axb.set_xticklabels([f"{thresh:.0f}s cut-off", f"{round(dmax/100)*100:.0f}s"], fontsize=8.5)
    else:
        axb.set_xticks([])
    for sp in axb.spines.values():
        sp.set_visible(False)
    # Value pinned to the right edge so short and long bars label in the same place.
    axb.text(dur + dmax * 0.03, 0, f"{dur:.0f}s", va="center", ha="left",
             fontsize=14, fontweight="bold", color=color)
    axb.text(0, 0.92, verdict, transform=axb.transAxes, fontsize=11.5,
             fontweight="bold", color=color, va="top")

    axi = fig.add_subplot(gs[r, 1])
    pth = OUT / f"row_{ep}.png"
    if pth.exists():
        axi.imshow(cv2.cvtColor(cv2.imread(str(pth)), cv2.COLOR_BGR2RGB))
    axi.set_xticks([]); axi.set_yticks([])
    for sp in axi.spines.values():
        sp.set_color(color); sp.set_linewidth(2.2)
    axi.set_title(caption(ep, verdict), fontsize=10.3, color="#52514e", loc="left", pad=5)

fig.suptitle(f"The gate drops on length, not on what any single frame shows  \u2014  lab {LAB}",
             fontsize=16, fontweight="bold", color="#14161a", y=0.995)
fig.text(0.5, 0.005,
         "Left: episode length against the 3x-median cut-off. Right: five frames spread evenly across "
         "the episode, with timestamps. Every frame here has hands in it \u2014 that is not what the gate "
         "is judging.",
         ha="center", fontsize=10, color="#7c7b76")
fig.tight_layout(rect=[0.005, 0.022, 1, 0.972])
outfig = REPORTS / "figs" / "gate_examples.png"
fig.savefig(outfig, dpi=140, facecolor="white")
print(f"wrote {outfig}  ({outfig.stat().st_size/1024:.0f} KB)")
