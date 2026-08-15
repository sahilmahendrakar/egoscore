"""Probe rl2 episodes: array inventory, object sizes, and the cost of a pose-only read."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from egoscore.access import load_creds, r2_client, r2_fs

REPORTS = Path(__file__).resolve().parent.parent / "reports"
df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
rl2 = df[(df["lab"] == "rl2") & (df["embodiment"] == "human_bimanual")]
rl2 = rl2[rl2["zarr_processed_path"].fillna("").str.strip() != ""]
print(f"rl2 human episodes with a zarr path: {len(rl2)}")

creds = load_creds()
c = r2_client(creds)

# --- object sizes across a few episodes, to price the download ---
print("\n=== object sizes (3 episodes) ===")
img_mb, pose_mb = [], []
for _, row in rl2.head(3).iterrows():
    prefix = row["zarr_processed_path"].replace("s3://rldb/", "")
    paginator = c.get_paginator("list_objects_v2")
    sizes = {}
    for page in paginator.paginate(Bucket="rldb", Prefix=prefix):
        for o in page.get("Contents", []):
            arr = o["Key"][len(prefix):].strip("/").split("/")[0]
            sizes[arr] = sizes.get(arr, 0) + o["Size"]
    tot = sum(sizes.values())
    imgs = sum(v for k, v in sizes.items() if k.startswith("images"))
    print(f"\n  {prefix.split('/')[-1]}  frames={row['num_frames']:.0f}  total={tot/1e6:.1f}MB")
    for k, v in sorted(sizes.items(), key=lambda x: -x[1]):
        print(f"      {k:35s} {v/1e6:8.2f} MB")
    img_mb.append(imgs / 1e6)
    pose_mb.append((tot - imgs) / 1e6)

n = len(rl2)
print(f"\n=== projected for all {n} rl2 episodes ===")
print(f"  images:    {sum(img_mb)/len(img_mb)*n/1000:.1f} GB   <- too slow to pull wholesale")
print(f"  non-image: {sum(pose_mb)/len(pose_mb)*n/1000:.2f} GB  <- what we actually need")

# --- what the pose-only read costs ---
fs = r2_fs(creds)
import zarr

prefix = rl2.iloc[0]["zarr_processed_path"].replace("s3://", "")
t0 = time.time()
g = zarr.open_group(zarr.storage.FsspecStore(fs, path=prefix), mode="r")
print(f"\nopened group in {time.time()-t0:.1f}s")

print("\n=== arrays ===")
for name, arr in sorted(g.arrays()):
    print(f"  {name:42s} shape={str(arr.shape):20s} dtype={arr.dtype}")

print("\n=== attrs ===")
for k, v in g.attrs.items():
    s = str(v)
    print(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

t0 = time.time()
keys = [k for k in ["left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose",
                    "left.obs_keypoints", "right.obs_keypoints"] if k in g]
data = {k: g[k][:] for k in keys}
dt = time.time() - t0
print(f"\npose-only read of {len(keys)} arrays in {dt:.1f}s")
for k, v in data.items():
    print(f"  {k:24s} {str(v.shape):18s} nan_frac={float((v != v).mean()):.4f}")
print(f"\n=> projected pose-only pull for {n} episodes, 16-way parallel: ~{dt*n/16/60:.1f} min")
