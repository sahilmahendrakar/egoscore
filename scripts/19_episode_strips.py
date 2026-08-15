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

if LAB == "rl2":
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
# Two typical keepers (closest to the median length), plus whatever the gate actually caught.
kept = (kd[kd["keep"]]
        .assign(_d=lambda d: (d.duration_s - med).abs())
        .nsmallest(2, "_d").index.tolist())
dropped = kd[~kd["keep"]].nlargest(2, "duration_s").index.tolist()
if len(dropped) < 2:
    dropped += kd[~kd["keep"]].nsmallest(2 - len(dropped), "duration_s").index.tolist()

ROWS = ([(h, "KEPT", "#1baf7a") for h in kept]
        + [(h, "DROPPED", "#e34948") for h in dropped])
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
    if verdict == "KEPT":
        return (f"KEPT — one garment, start to finish in {r.duration_s:.0f}s "
                f"(the lab median is {med:.0f}s). No rule fired.")
    reason = str(r.drop_reason)
    if "runaway_length" in reason:
        return (f"DROPPED for length — {r.duration_s:.0f}s, {r.duration_s/med:.0f}x the "
                f"{med:.0f}s median. Watch the garment change between frames: this is a whole "
                f"folding session labelled as one demonstration.")
    if "suspiciously_short" in reason:
        return (f"{r.duration_s:.0f}s — under 15% of the median of {med:.0f}s. A fragment · dropped")
    return f"{reason} · {r.duration_s:.0f}s · dropped"


fig, axes = plt.subplots(len(ROWS), 1, figsize=(15.4, 2.08 * len(ROWS)), facecolor="white")
for ax, (ep, verdict, color) in zip(axes, ROWS):
    p = OUT / f"row_{ep}.png"
    if p.exists():
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(2.2)
    ax.text(-0.011, 0.5, verdict, transform=ax.transAxes, rotation=90, ha="right", va="center",
            fontsize=10.5, fontweight="bold", color=color)
    ax.set_title(caption(ep, verdict), fontsize=10.5, color="#52514e", loc="left", pad=5)

fig.suptitle(f"What the quality gate is actually looking at  —  lab {LAB}",
             fontsize=16.5, fontweight="bold", color="#14161a", y=0.998)
fig.text(0.5, 0.004,
         "Five frames sampled evenly across each episode. Note the timestamps: the kept episodes span seconds, "
         "the dropped ones span minutes and cover many different garments. Nothing here is about hands.",
         ha="center", fontsize=10, color="#7c7b76")
fig.tight_layout(rect=[0.012, 0.021, 1, 0.979])
outfig = REPORTS / "figs" / "gate_examples.png"
fig.savefig(outfig, dpi=140, facecolor="white")
print(f"wrote {outfig}  ({outfig.stat().st_size/1024:.0f} KB)")
