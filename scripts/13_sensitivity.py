"""Is the result an artifact of the proxy model's hyperparameters?

The obvious attack on this project is that we picked a ridge configuration that happened
to make our selector look good. So we re-run the core comparison across a grid of proxy
configurations -- regularisation strength, feature-map width, history length, and
prediction horizon -- and check that the *ranking* survives.

We deliberately do not re-tune anything per condition. If the conclusion only holds at
one setting, it is not a conclusion.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore import proxy
from egoscore.gate import apply_gate
from egoscore.proxy import RidgeChunkPolicy, avg_mse
from egoscore.select import SELECTORS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
POSES = ROOT / "data" / "poses"

SEEDS = [0, 1, 2, 3, 4]
K_FRAC = 0.25
CONDITIONS = ["random", "curated", "dpp", "kcenter", "degenerate"]

# (alpha, n_features, history, horizon)
GRID = [
    (1.0, 512, 10, 30),    # the configuration used for the headline result
    (0.1, 512, 10, 30),
    (10.0, 512, 10, 30),
    (1.0, 256, 10, 30),
    (1.0, 1024, 10, 30),
    (1.0, 512, 5, 30),
    (1.0, 512, 20, 30),
    (1.0, 512, 10, 15),
    (1.0, 512, 10, 60),
]

feat = pd.read_csv(REPORTS / "features.csv")
gated = apply_gate(feat)
fcols = [c for c in gated.columns if c.startswith("f_")]

# Raw episode arrays, loaded once; windowing is redone per config since HISTORY/HORIZON change.
raw: dict[str, dict] = {}
for h in feat["episode_hash"]:
    f = POSES / f"{h}.npz"
    if f.exists():
        with np.load(f, allow_pickle=True) as z:
            raw[h] = {k: z[k] for k in z.files}
print(f"loaded {len(raw)} episodes")

rows = []
t0 = time.time()
for alpha, nfeat, hist, horiz in GRID:
    proxy.HISTORY, proxy.HORIZON = hist, horiz
    cache = {}
    for h, ep in raw.items():
        X, Y = proxy.episode_windows(ep)
        if len(X):
            cache[h] = (X.astype(np.float32), Y.astype(np.float32))

    def stack(hs):
        xs = [cache[h][0] for h in hs if h in cache]
        ys = [cache[h][1] for h in hs if h in cache]
        if not xs:
            return np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32)
        return np.vstack(xs), np.vstack(ys)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ops = gated["operator"].dropna().unique()
        scenes = gated["scene"].dropna().unique()
        ho_ops = set(rng.choice(ops, max(1, int(0.25 * len(ops))), replace=False))
        ho_sc = set(rng.choice(scenes, max(1, int(0.25 * len(scenes))), replace=False))
        in_op, in_sc = gated["operator"].isin(ho_ops), gated["scene"].isin(ho_sc)

        Xe_op, Ye_op = stack(gated[in_op & ~in_sc]["episode_hash"])
        Xe_sc, Ye_sc = stack(gated[~in_op & in_sc]["episode_hash"])
        if len(Xe_op) == 0 or len(Xe_sc) == 0:
            continue

        pool = gated[~in_op & ~in_sc & gated["keep"]]
        Z = np.nan_to_num(pool[fcols].to_numpy(float))
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        groups = (pool["operator"].astype(str) + "|" + pool["scene"].astype(str)).to_numpy()
        hashes = pool["episode_hash"].to_numpy()
        k = max(5, int(K_FRAC * len(pool)))

        for cond in CONDITIONS:
            idx = SELECTORS[cond](Z, k, seed=seed, groups=groups)
            Xtr, Ytr = stack(hashes[idx])
            if len(Xtr) < 50:
                continue
            m = RidgeChunkPolicy(alpha=alpha, n_features=nfeat, seed=seed).fit(Xtr, Ytr)
            rows.append({
                "alpha": alpha, "n_features": nfeat, "history": hist, "horizon": horiz,
                "seed": seed, "condition": cond,
                "mse_unseen_operator": avg_mse(m, Xe_op, Ye_op),
                "mse_unseen_scene": avg_mse(m, Xe_sc, Ye_sc),
            })
    print(f"  alpha={alpha} nfeat={nfeat} hist={hist} horiz={horiz}  "
          f"({time.time()-t0:.0f}s)", flush=True)

# Restore module defaults so importing this module has no side effects elsewhere.
proxy.HISTORY, proxy.HORIZON = 10, 30

df = pd.DataFrame(rows)
df.to_csv(REPORTS / "sensitivity.csv", index=False)

METRICS = ["mse_unseen_operator", "mse_unseen_scene"]
cfg_cols = ["alpha", "n_features", "history", "horizon"]
base = df[df.condition == "random"].set_index(cfg_cols + ["seed"])[METRICS]
j = df[df.condition != "random"].join(
    base.rename(columns={m: f"b_{m}" for m in METRICS}), on=cfg_cols + ["seed"])

print("\n=== does the ranking survive? (% vs random at same config+seed) ===")
out = []
for (cond, *cfg), g in j.groupby(["condition"] + cfg_cols):
    r = {"condition": cond, "alpha": cfg[0], "n_features": cfg[1],
         "history": cfg[2], "horizon": cfg[3]}
    for m in METRICS:
        r[m] = 100.0 * ((g[m] - g[f"b_{m}"]) / g[f"b_{m}"]).mean()
    out.append(r)
o = pd.DataFrame(out)
piv = o.pivot_table(index=cfg_cols, columns="condition",
                    values="mse_unseen_operator").round(2)
print("\nunseen operator, % change vs random:")
print(piv.to_string())
o.to_csv(REPORTS / "sensitivity_summary.csv", index=False)

print("\n=== consistency across all configurations ===")
for cond in ["dpp", "kcenter", "curated", "degenerate"]:
    g = o[o.condition == cond]
    n_better = int((g["mse_unseen_operator"] < 0).sum())
    print(f"  {cond:11s} beats random in {n_better}/{len(g)} configurations   "
          f"(mean {g['mse_unseen_operator'].mean():+.2f}%, "
          f"worst {g['mse_unseen_operator'].max():+.2f}%)")
