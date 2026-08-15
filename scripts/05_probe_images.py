"""Can we read a handful of frames without pulling the whole 168MB image shard?

If zarr issues a ranged GET against the shard index, sampling ~8 frames/episode is
cheap and the visual feature block is affordable. If it materialises the whole shard,
visual features cost 84.6 GB for this slice and are off the table.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "reports" / "slice_fold_clothes.csv")
rl2 = df[(df["lab"] == "rl2") & (df["embodiment"] == "human_bimanual")]
rl2 = rl2[rl2["zarr_processed_path"].fillna("").str.strip() != ""]

creds = load_creds()
fs = r2_fs(creds)
prefix = rl2.iloc[0]["zarr_processed_path"].replace("s3://", "")

g = zarr.open_group(zarr.storage.FsspecStore(fs, path=prefix), mode="r")
arr = g["images.front_1"]
print(f"images.front_1: shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks} shards={arr.shards}")

n = arr.shape[0]
idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]

t0 = time.time()
frames = [bytes(arr[i].item() if hasattr(arr[i], "item") else arr[i]) for i in idx]
dt = time.time() - t0
tot = sum(len(f) for f in frames)
print(f"\nread {len(idx)} frames in {dt:.1f}s, {tot/1e3:.0f} KB of JPEG")

# Decode one to confirm it is a real image.
import numpy as np
import cv2

img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
print(f"decoded frame 0: {img.shape if img is not None else 'FAILED'}")

n_eps = len(rl2)
print(f"\n=> {len(idx)} frames/episode over {n_eps} episodes, 16-way parallel: ~{dt*n_eps/16/60:.1f} min")
print("   (compare: pulling all images for this slice = 84.6 GB)")
