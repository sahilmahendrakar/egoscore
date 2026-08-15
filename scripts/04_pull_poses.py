"""Pull pose/keypoint arrays for the rl2 fold_clothes slice into local .npz files.

Images are deliberately not pulled: they are 84.6 GB for this slice versus 2.3 GB for
everything else. A curation engine that costs more to run than the training it is
supposed to save is not a curation engine, so every signal we use is derived from
arrays that are ~4 MB/episode rather than ~170 MB/episode.

Two data traps handled here:
  1. Arrays are zero-padded up to a chunk boundary. The real length is ``total_frames``
     in the group attrs. Un-truncated padding looks exactly like a frozen tracker.
  2. SQL ``num_frames`` disagrees with the zarr ``total_frames``. We trust the zarr.
"""

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
OUT = ROOT / "data" / "poses"
OUT.mkdir(parents=True, exist_ok=True)

WANTED = [
    "left.obs_ee_pose",
    "right.obs_ee_pose",
    "left.obs_wrist_pose",
    "right.obs_wrist_pose",
    "obs_head_pose",
    "left.obs_keypoints",
    "right.obs_keypoints",
    "obs_eye_gaze",
    "obs_rgb_timestamps_ns",
]

LAB = sys.argv[1] if len(sys.argv) > 1 else "rl2"

df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
sel = df[(df["lab"] == LAB) & (df["embodiment"] == "human_bimanual")]
sel = sel[sel["zarr_processed_path"].fillna("").str.strip() != ""].reset_index(drop=True)
print(f"{LAB}: {len(sel)} episodes to pull")

creds = load_creds()
fs = r2_fs(creds)


def pull(row) -> dict:
    ep = row["episode_hash"]
    dest = OUT / f"{ep}.npz"
    if dest.exists():
        return {"episode_hash": ep, "status": "cached"}
    try:
        prefix = row["zarr_processed_path"].replace("s3://", "")
        g = zarr.open_group(zarr.storage.FsspecStore(fs, path=prefix), mode="r")
        attrs = dict(g.attrs)
        n = int(attrs.get("total_frames", 0)) or None

        arrays = {}
        for k in WANTED:
            if k in g:
                a = g[k][:]
                # Truncate the chunk-boundary padding; see module docstring.
                arrays[k.replace(".", "__")] = a[:n] if n else a

        n_ann = 0
        if "annotations" in g:
            try:
                ann = g["annotations"][:]
                n_ann = int(len(ann))
                if n_ann:
                    arrays["annotations"] = np.asarray(ann, dtype=object)
            except Exception:
                pass

        padded_len = int(g[WANTED[0]].shape[0]) if WANTED[0] in g else -1
        np.savez_compressed(dest, **arrays)
        return {
            "episode_hash": ep,
            "status": "ok",
            "total_frames": n,
            "padded_len": padded_len,
            "sql_num_frames": row["num_frames"],
            "n_annotations": n_ann,
            "fps": attrs.get("fps"),
            "n_arrays": len(arrays),
        }
    except Exception as e:
        return {"episode_hash": ep, "status": "error", "error": f"{type(e).__name__}: {e}"[:200]}


t0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=24) as pool:
    futs = {pool.submit(pull, r): r["episode_hash"] for _, r in sel.iterrows()}
    for i, f in enumerate(as_completed(futs), 1):
        results.append(f.result())
        if i % 25 == 0 or i == len(futs):
            ok = sum(r["status"] in ("ok", "cached") for r in results)
            print(f"  {i}/{len(futs)}  ok={ok}  elapsed={time.time()-t0:.0f}s", flush=True)

res = pd.DataFrame(results)
res.to_csv(REPORTS / f"pull_{LAB}.csv", index=False)
print(f"\ndone in {time.time()-t0:.0f}s")
print(res["status"].value_counts().to_string())
errs = res[res["status"] == "error"]
if len(errs):
    print("\nerrors (first 5):")
    for e in errs["error"].head(5):
        print("  ", e)
ok = res[res["status"] == "ok"]
if len(ok) and "total_frames" in ok:
    print("\nzarr total_frames vs SQL num_frames disagreement:")
    d = ok.dropna(subset=["total_frames", "sql_num_frames"])
    print(f"  disagree on {(d['total_frames'] != d['sql_num_frames']).sum()} / {len(d)} episodes")
    print(f"  episodes with annotations: {(ok['n_annotations'] > 0).sum()} / {len(ok)}")
