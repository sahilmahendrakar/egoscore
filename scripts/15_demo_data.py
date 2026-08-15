"""Build the data payload for the interactive demo page.

Projects the 41-dim episode embedding to 2D, records what each selector picks at the
demo budget, and emits a single JSON blob (thumbnails inlined as data URIs).
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from egoscore.gate import apply_gate
from egoscore.select import SELECTORS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
THUMBS = ROOT / "data" / "thumbs"

SEED = 0
K_FRAC = 0.25
CONDITIONS = ["random", "curated", "dpp", "kcenter", "degenerate"]

feat = pd.read_csv(REPORTS / "features.csv")
gated = apply_gate(feat)
fcols = [c for c in gated.columns if c.startswith("f_")]

Z_all = np.nan_to_num(gated[fcols].to_numpy(float))
Z_all = (Z_all - Z_all.mean(0)) / (Z_all.std(0) + 1e-9)

# 2D projection of every episode (kept and dropped), so the gate is visible too.
print("projecting ...")
p50 = PCA(n_components=min(20, Z_all.shape[1]), random_state=SEED).fit_transform(Z_all)
xy = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca").fit_transform(p50)
xy = (xy - xy.mean(0)) / (xy.max(0) - xy.min(0) + 1e-9)  # normalise to roughly [-0.5, 0.5]

# Selection runs on the gated pool only, exactly as in the experiment.
pool_mask = gated["keep"].to_numpy()
pool_idx = np.flatnonzero(pool_mask)
Zp = Z_all[pool_idx]
groups = (gated["operator"].astype(str) + "|" + gated["scene"].astype(str)).to_numpy()[pool_idx]
k = max(5, int(K_FRAC * len(pool_idx)))
print(f"pool={len(pool_idx)}  K={k}")

selections = {}
for cond in CONDITIONS:
    sel_local = SELECTORS[cond](Zp, k, seed=SEED, groups=groups)
    sel_global = pool_idx[sel_local]
    selections[cond] = {
        "idx": sorted(int(i) for i in sel_global),
        "n_groups": int(len(np.unique(groups[sel_local]))),
    }
    print(f"  {cond:11s} {len(sel_global)} episodes, {selections[cond]['n_groups']} groups")

# Measured Avg-MSE per condition, averaged over seeds, at this budget.
res = pd.read_csv(REPORTS / "results.csv")
mse = {}
for cond in CONDITIONS:
    g = res[(res.condition == cond) & (res.k_frac == K_FRAC)]
    if len(g):
        mse[cond] = {"op": float(g["mse_unseen_operator"].mean()),
                     "scene": float(g["mse_unseen_scene"].mean())}
rand_op = mse.get("random", {}).get("op")
for cond in mse:
    mse[cond]["pct_op"] = 100.0 * (mse[cond]["op"] - rand_op) / rand_op

# Per-episode payload.
ops = sorted(gated["operator"].astype(str).unique())
scenes = sorted(gated["scene"].astype(str).unique())
op_i = {o: i for i, o in enumerate(ops)}
sc_i = {s: i for i, s in enumerate(scenes)}

episodes = []
n_thumb = 0
for i, row in gated.iterrows():
    h = row["episode_hash"]
    t = THUMBS / f"{h}.jpg"
    thumb = ""
    if t.exists():
        thumb = "data:image/jpeg;base64," + base64.b64encode(t.read_bytes()).decode()
        n_thumb += 1
    episodes.append({
        "x": round(float(xy[i, 0]), 4),
        "y": round(float(xy[i, 1]), 4),
        "op": op_i[str(row["operator"])],
        "sc": sc_i[str(row["scene"])],
        "dur": round(float(row["duration_s"]), 1),
        "oof": round(float(row["oof_max"]), 3),
        "keep": bool(row["keep"]),
        "why": str(row["drop_reason"]),
        "t": thumb,
    })

payload = {
    "episodes": episodes,
    "operators": ops,
    "scenes": scenes,
    "selections": selections,
    "mse": mse,
    "k": k,
    "k_frac": K_FRAC,
    "n_pool": int(len(pool_idx)),
    "n_total": int(len(gated)),
    "n_dropped": int((~pool_mask).sum()),
}

out = REPORTS / "demo_data.json"
out.write_text(json.dumps(payload, separators=(",", ":")))
print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB, {n_thumb} thumbnails)")
