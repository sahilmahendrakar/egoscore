"""Build the data payload for the interactive demo page.

Two datasets, because they answer different questions:

  rl2      572 episodes with a full operator x scene grid. This is the only slice where the
           selection experiment can run, so it carries the selectors and the Avg-MSE numbers.
           It is also almost perfectly clean, so its quality gate is boring.

  microagi 1,200 episodes sampled from the largest fold_clothes slice. No operator or scene
           labels, so no selection experiment is possible — but the gate drops ~14% of it,
           so this is where you can actually see filtering do something.
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

from egoscore.features import embedding_features, quality_signals
from egoscore.gate import apply_gate
from egoscore.select import SELECTORS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

SEED = 0
K_FRAC = 0.25
CONDITIONS = ["random", "curated", "dpp", "kcenter", "degenerate"]

meta_all = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")


def load_features(lab: str) -> pd.DataFrame:
    """rl2 reuses the cached features.csv; other labs are computed here."""
    if lab == "rl2":
        return pd.read_csv(REPORTS / "features.csv")
    poses = ROOT / "data" / f"poses_{lab}"
    rows = []
    for f in sorted(poses.glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            ep = {k: z[k] for k in z.files}
        row = {"episode_hash": f.stem}
        row.update(quality_signals(ep))
        row.update(embedding_features(ep))
        if f.stem in meta_all.index:
            m = meta_all.loc[f.stem]
            for c in ["lab", "operator", "scene"]:
                row[c] = m[c] if c in m else ""
        rows.append(row)
    return pd.DataFrame(rows)


def _lab(row, col: str) -> str:
    v = row.get(col, "")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v)


def project(Z: np.ndarray) -> np.ndarray:
    p = PCA(n_components=min(20, Z.shape[1]), random_state=SEED).fit_transform(Z)
    xy = TSNE(n_components=2, random_state=SEED, perplexity=30, init="pca").fit_transform(p)
    return (xy - xy.mean(0)) / (xy.max(0) - xy.min(0) + 1e-9)


def build(lab: str, thumbs_dir: Path, with_selection: bool) -> dict:
    feat = load_features(lab)
    gated = apply_gate(feat)
    fcols = [c for c in gated.columns if c.startswith("f_")]

    Z = np.nan_to_num(gated[fcols].to_numpy(float))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    print(f"[{lab}] projecting {len(Z)} episodes ...")
    xy = project(Z)

    selections, mse = {}, {}
    if with_selection:
        pool_idx = np.flatnonzero(gated["keep"].to_numpy())
        Zp = Z[pool_idx]
        groups = (gated["operator"].astype(str) + "|" + gated["scene"].astype(str)).to_numpy()[pool_idx]
        k = max(5, int(K_FRAC * len(pool_idx)))
        for cond in CONDITIONS:
            loc = SELECTORS[cond](Zp, k, seed=SEED, groups=groups)
            sel = pool_idx[loc]
            selections[cond] = {"idx": sorted(int(i) for i in sel),
                                "n_groups": int(len(np.unique(groups[loc])))}
        res = pd.read_csv(REPORTS / "results.csv")
        for cond in CONDITIONS:
            g = res[(res.condition == cond) & (res.k_frac == K_FRAC)]
            if len(g):
                mse[cond] = {"op": float(g["mse_unseen_operator"].mean()),
                             "scene": float(g["mse_unseen_scene"].mean())}
        rand_op = mse.get("random", {}).get("op")
        for c in mse:
            mse[c]["pct_op"] = 100.0 * (mse[c]["op"] - rand_op) / rand_op

    def labels(col: str) -> list[str]:
        """Distinct string labels, NaN-safe. microagi has no operator/scene at all."""
        if col not in gated.columns:
            return [""]
        v = gated[col].fillna("").astype(str).replace({"nan": ""})
        return sorted(v.unique().tolist())

    ops, scenes = labels("operator"), labels("scene")
    op_i = {o: i for i, o in enumerate(ops)}
    sc_i = {s: i for i, s in enumerate(scenes)}

    episodes, n_thumb = [], 0
    for i, row in gated.reset_index(drop=True).iterrows():
        h = row["episode_hash"]
        t = thumbs_dir / f"{h}.jpg"
        thumb = ""
        if t.exists():
            thumb = "data:image/jpeg;base64," + base64.b64encode(t.read_bytes()).decode()
            n_thumb += 1
        episodes.append({
            "x": round(float(xy[i, 0]), 4),
            "y": round(float(xy[i, 1]), 4),
            "op": op_i.get(_lab(row, "operator"), 0),
            "sc": sc_i.get(_lab(row, "scene"), 0),
            "dur": round(float(row["duration_s"]), 1),
            "motion": round(float(row["motion_energy"]), 3),
            "keep": bool(row["keep"]),
            "why": str(row["drop_reason"]),
            "t": thumb,
        })

    med = float(gated["duration_s"].median())
    out = {
        "episodes": episodes, "operators": ops, "scenes": scenes,
        "selections": selections, "mse": mse,
        "k": int(max(5, K_FRAC * int(gated["keep"].sum()))) if with_selection else 0,
        "k_frac": K_FRAC,
        "n_pool": int(gated["keep"].sum()),
        "n_total": int(len(gated)),
        "n_dropped": int((~gated["keep"]).sum()),
        "median_dur": round(med, 1),
        "with_selection": with_selection,
    }
    print(f"[{lab}] {out['n_total']} episodes, {out['n_dropped']} dropped "
          f"({100*out['n_dropped']/out['n_total']:.1f}%), {n_thumb} thumbnails")
    return out


payload = {
    "datasets": {
        "rl2": build("rl2", ROOT / "data" / "thumbs", with_selection=True),
        "microagi": build("microagi", ROOT / "data" / "thumbs_microagi", with_selection=False),
    }
}

out = REPORTS / "demo_data.json"
# allow_nan=False so a stray NaN can never ship as invalid JSON again -- it did once, and
# the page failed silently with an empty canvas.
out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
print(f"\nwrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
