"""Measure hand visibility from the pixels, using a real hand detector.

This replaces the rule we deleted. The deleted version tried to project the stored 3D hand
keypoints into the image and could not be made correct -- we never resolved the coordinate
convention, and two different projection models both put every keypoint outside the frame
on images where the hands are obviously there.

Detecting hands in the image sidesteps the whole problem. There is no convention to guess:
either MediaPipe finds a hand in the frame or it does not, and you can check the answer by
looking at the picture.

Writes reports/hand_visibility_<lab>.csv with, per episode, the fraction of sampled frames
in which at least one hand was detected.

RESULT: this does not work either, and we are not using it. MediaPipe detects a hand in only
59% of rl2 frames on average, and the 36 episodes it scores at 0% plainly show hands folding
garments when you render them (scripts/19_episode_strips.py handcheck). The egocentric fisheye
view -- extreme angle, hands entering at the frame edge, motion blur -- is well outside what
the detector handles.

So: two independent approaches to "are the hands visible", 3D keypoint projection and a 2D
hand detector, both fail on this footage. That is the evidence for not restoring the original
rule. Where it happened to agree with a human looking at the frames, that was coincidence.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

LAB = sys.argv[1] if len(sys.argv) > 1 else "rl2"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 8

poses = ROOT / "data" / ("poses" if LAB == "rl2" else f"poses_{LAB}")
have = sorted(p.stem for p in poses.glob("*.npz"))
meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")
eps = [h for h in have if h in meta.index]
print(f"{LAB}: {len(eps)} episodes x {N_FRAMES} frames")

creds = load_creds()
fs = r2_fs(creds)


def fetch(ep: str):
    """Return (episode, [BGR frames]) or (episode, None)."""
    try:
        g = zarr.open_group(
            zarr.storage.FsspecStore(fs, path=meta.loc[ep, "zarr_processed_path"].replace("s3://", "")),
            mode="r")
        arr = g["images.front_1"]
        n = int(g.attrs.get("total_frames", arr.shape[0]))
        idx = np.linspace(int(0.08 * n), int(0.92 * n), N_FRAMES).astype(int)
        out = []
        for i in idx:
            blob = arr[int(i):int(i) + 1][0]
            img = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                out.append(img)
        return ep, out
    except Exception as e:
        return ep, None


import mediapipe as mp

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                 min_detection_confidence=0.3)

q: Queue = Queue(maxsize=64)


def producer():
    with ThreadPoolExecutor(max_workers=16) as pool:
        for ep, frames in pool.map(fetch, eps):
            q.put((ep, frames))
    q.put(None)


import threading

threading.Thread(target=producer, daemon=True).start()

rows = []
t0 = time.time()
done = 0
while True:
    item = q.get()
    if item is None:
        break
    ep, frames = item
    done += 1
    if not frames:
        rows.append({"episode_hash": ep, "lab": LAB, "n_frames": 0, "hand_visible_frac": np.nan})
        continue
    seen = 0
    n_hands = []
    for img in frames:
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        k = len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0
        n_hands.append(k)
        seen += 1 if k > 0 else 0
    rows.append({
        "episode_hash": ep, "lab": LAB, "n_frames": len(frames),
        "hand_visible_frac": seen / len(frames),
        "mean_hands_per_frame": float(np.mean(n_hands)),
        "frames_with_two_hands": int(sum(1 for k in n_hands if k >= 2)),
    })
    if done % 100 == 0:
        print(f"  {done}/{len(eps)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
out = REPORTS / f"hand_visibility_{LAB}.csv"
df.to_csv(out, index=False)
print(f"\nwrote {out}  ({len(df)} episodes) in {time.time()-t0:.0f}s")
print(df["hand_visible_frac"].describe(percentiles=[0.01, 0.05, 0.5, 0.95]).to_string())
for thr in (0.0, 0.25, 0.5):
    n = int((df["hand_visible_frac"] <= thr).sum())
    print(f"  hands visible in <= {thr:.0%} of sampled frames: {n} episodes ({100*n/len(df):.1f}%)")
