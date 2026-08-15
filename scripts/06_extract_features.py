"""Compute quality signals + embedding features for every pulled episode.

Writes reports/features.csv (one row per episode, joined to SQL metadata).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore.features import embedding_features, quality_signals

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
POSES = ROOT / "data" / "poses"

meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
meta = meta.set_index("episode_hash")

files = sorted(POSES.glob("*.npz"))
print(f"extracting from {len(files)} episodes")

rows = []
t0 = time.time()
for i, f in enumerate(files, 1):
    ep_hash = f.stem
    try:
        with np.load(f, allow_pickle=True) as z:
            ep = {k: z[k] for k in z.files}
        row = {"episode_hash": ep_hash}
        row.update(quality_signals(ep))
        row.update(embedding_features(ep))
        if ep_hash in meta.index:
            m = meta.loc[ep_hash]
            for col in ["lab", "operator", "scene", "num_frames", "task_description", "objects"]:
                if col in m:
                    row[col] = m[col]
        rows.append(row)
    except Exception as e:
        print(f"  ERROR {ep_hash}: {type(e).__name__}: {e}")
    if i % 100 == 0:
        print(f"  {i}/{len(files)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(REPORTS / "features.csv", index=False)
print(f"\nwrote {REPORTS/'features.csv'}  shape={df.shape}  in {time.time()-t0:.0f}s")

qcols = [c for c in df.columns if c.startswith(("nan_", "frozen_", "oof_", "motion", "path_len", "duration"))]
print("\n=== quality signal distributions ===")
print(df[qcols].describe(percentiles=[0.05, 0.5, 0.95]).T.to_string())

print("\n=== structure ===")
for c in ["lab", "operator", "scene"]:
    if c in df:
        print(f"  {c}: {df[c].nunique()} distinct")
