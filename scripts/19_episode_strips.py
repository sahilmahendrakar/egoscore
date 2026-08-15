"""Contact sheets of what the quality gate keeps, what it drops, and what it used to
drop by mistake.

Rendering these frames is how we caught a real bug: the stored intrinsics are a 3x4 K
matrix, which invites a pinhole projection, but the Aria camera is a fisheye. Under
pinhole, a hand 60 deg off-axis projects to r = f*tan(60) = 462 px and reads as "outside
a 640x480 image" while the fisheye frame shows it perfectly well at r = f*theta = 279 px.
The gate was flagging 23 episodes with clearly visible hands. Looking at the pictures is
what surfaced it -- no amount of staring at the histogram would have.
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

# Episodes the old pinhole model wrongly flagged; kept correctly once the projection was
# fixed. Recorded explicitly so the figure documents the correction.
PINHOLE_FALSE_POSITIVES = ["2025-10-29-22-05-17-839000", "2025-10-13-03-58-47-763000"]

kd = pd.read_csv(REPORTS / "keep_drop.csv").set_index("episode_hash")
meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")

# Four rows keeps the composite landscape, which is what a 16:9 slide can actually show.
kept = kd[kd["keep"]].nsmallest(2, "oof_max").index.tolist()
oof_drop = kd[kd["rule_hands_out_of_frame"]].index.tolist()[:1]
fp = [h for h in PINHOLE_FALSE_POSITIVES if h in kd.index][:1]

ROWS = (
    [(h, "KEPT", "#1baf7a") for h in kept]
    + [(h, "DROPPED", "#e34948") for h in oof_drop]
    + [(h, "RESCUED", "#eda100") for h in fp]
)
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
        tiles.append(cv2.resize(img, (FRAME_W, int(h * FRAME_W / w)), interpolation=cv2.INTER_AREA))
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
    oof = 100 * r.oof_max
    if verdict == "KEPT":
        return f"hands tracked throughout · {oof:.1f}% of frames off-frame · {r.duration_s:.0f}s"
    if verdict == "RESCUED":
        return (f"the old pinhole model called this 100% off-frame — the fisheye model says "
                f"{oof:.1f}% · kept · {r.duration_s:.0f}s")
    if r.rule_hands_out_of_frame:
        return f"a hand is outside the image in {oof:.0f}% of frames · dropped · {r.duration_s:.0f}s"
    return (f"{r.duration_s:.0f}s — above the 99th percentile, an un-segmented recording "
            f"rather than one demonstration · dropped")


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

fig.suptitle("What the quality gate is actually looking at",
             fontsize=16.5, fontweight="bold", color="#14161a", y=0.998)
fig.text(0.5, 0.004,
         "Five frames sampled evenly across each episode, chosen by ranking on the signal rather than by eye. "
         "The bottom row is a bug an earlier version of the gate made, which looking at the frames revealed.",
         ha="center", fontsize=10, color="#7c7b76")
fig.tight_layout(rect=[0.012, 0.021, 1, 0.979])
outfig = REPORTS / "figs" / "gate_examples.png"
fig.savefig(outfig, dpi=140, facecolor="white")
print(f"wrote {outfig}  ({outfig.stat().st_size/1024:.0f} KB)")
