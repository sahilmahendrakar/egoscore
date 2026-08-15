"""The experiment: does quality-gated, coverage-maximizing selection beat random?

For each seed we draw one held-out split (unseen operators, unseen scenes), then run
every selection condition against that identical split. Varying the split across seeds
means the error bars cover split variance as well as selection variance, rather than
flattering us by holding the evaluation set fixed.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore.gate import apply_gate, describe_thresholds
from egoscore.proxy import RidgeChunkPolicy, avg_mse, episode_windows
from egoscore.select import SELECTORS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
POSES = ROOT / "data" / "poses"

SEEDS = [0, 1, 2]
K_FRACS = [0.25, 0.50]
CONDITIONS = ["random", "curated", "dpp", "kcenter", "degenerate"]

# ---------------------------------------------------------------- features + gate
feat = pd.read_csv(REPORTS / "features.csv")
print(f"features: {feat.shape}")

gated = apply_gate(feat)
gated.to_csv(REPORTS / "keep_drop.csv", index=False)
audit = describe_thresholds(gated)
audit.to_csv(REPORTS / "gate_audit.csv", index=False)
print("\n=== quality gate audit ===")
print(audit.to_string(index=False))

# ---------------------------------------------------------------- window cache
print("\ncaching windows ...")
t0 = time.time()
cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
for i, h in enumerate(feat["episode_hash"], 1):
    f = POSES / f"{h}.npz"
    if not f.exists():
        continue
    with np.load(f, allow_pickle=True) as z:
        ep = {k: z[k] for k in z.files}
    X, Y = episode_windows(ep)
    if len(X):
        cache[h] = (X.astype(np.float32), Y.astype(np.float32))
    if i % 150 == 0:
        print(f"  {i}/{len(feat)}  {time.time()-t0:.0f}s", flush=True)
print(f"cached {len(cache)} episodes with windows in {time.time()-t0:.0f}s")
tot_w = sum(len(v[0]) for v in cache.values())
print(f"total windows: {tot_w}")


def stack(hashes) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for h in hashes:
        if h in cache:
            xs.append(cache[h][0])
            ys.append(cache[h][1])
    if not xs:
        return np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32)
    return np.vstack(xs), np.vstack(ys)


# ---------------------------------------------------------------- embedding space
fcols = [c for c in gated.columns if c.startswith("f_")]
print(f"\nembedding dims: {len(fcols)}")

results = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)

    ops = gated["operator"].dropna().unique()
    scenes = gated["scene"].dropna().unique()
    ho_ops = set(rng.choice(ops, max(1, int(0.25 * len(ops))), replace=False))
    ho_scenes = set(rng.choice(scenes, max(1, int(0.25 * len(scenes))), replace=False))

    in_ho_op = gated["operator"].isin(ho_ops)
    in_ho_sc = gated["scene"].isin(ho_scenes)

    eval_op = gated[in_ho_op & ~in_ho_sc]
    eval_sc = gated[~in_ho_op & in_ho_sc]
    pool_all = gated[~in_ho_op & ~in_ho_sc]           # before the gate
    pool = pool_all[pool_all["keep"]]                  # after the gate

    Xe_op, Ye_op = stack(eval_op["episode_hash"])
    Xe_sc, Ye_sc = stack(eval_sc["episode_hash"])
    print(
        f"\n--- seed {seed}: pool_gated={len(pool)} (of {len(pool_all)}), "
        f"eval_op={len(eval_op)} eps/{len(Xe_op)} win, eval_scene={len(eval_sc)} eps/{len(Xe_sc)} win"
    )
    if len(Xe_op) == 0 or len(Xe_sc) == 0:
        print("  !! empty eval split, skipping seed")
        continue

    Zraw = pool[fcols].to_numpy(dtype=np.float64)
    Zraw = np.nan_to_num(Zraw)
    Z = (Zraw - Zraw.mean(0)) / (Zraw.std(0) + 1e-9)   # whiten so no feature dominates
    groups = (pool["operator"].astype(str) + "|" + pool["scene"].astype(str)).to_numpy()
    hashes = pool["episode_hash"].to_numpy()

    for kf in K_FRACS:
        k = max(5, int(kf * len(pool)))

        for cond in CONDITIONS:
            t1 = time.time()
            idx = SELECTORS[cond](Z, k, seed=seed, groups=groups)
            sub = hashes[idx]
            Xtr, Ytr = stack(sub)
            if len(Xtr) < 50:
                continue
            m = RidgeChunkPolicy(alpha=1.0, seed=seed).fit(Xtr, Ytr)
            r = {
                "seed": seed, "k_frac": kf, "k": k, "condition": cond,
                "n_train_eps": len(sub), "n_train_win": len(Xtr),
                "mse_unseen_operator": avg_mse(m, Xe_op, Ye_op),
                "mse_unseen_scene": avg_mse(m, Xe_sc, Ye_sc),
                "n_groups": int(len(np.unique(groups[idx]))),
                "fit_s": time.time() - t1,
            }
            results.append(r)
            print(
                f"  k={kf:.0%} {cond:11s} eps={len(sub):4d} grp={r['n_groups']:3d} "
                f"op={r['mse_unseen_operator']:.5f} scene={r['mse_unseen_scene']:.5f} "
                f"({r['fit_s']:.1f}s)", flush=True
            )

        # Reference conditions, computed once per (seed, k).
        for name, hs in [
            ("random_nogate", pool_all["episode_hash"].to_numpy()),
        ]:
            r_idx = np.random.default_rng(seed).choice(len(hs), min(k, len(hs)), replace=False)
            Xtr, Ytr = stack(hs[r_idx])
            if len(Xtr) < 50:
                continue
            m = RidgeChunkPolicy(alpha=1.0, seed=seed).fit(Xtr, Ytr)
            results.append({
                "seed": seed, "k_frac": kf, "k": k, "condition": name,
                "n_train_eps": len(r_idx), "n_train_win": len(Xtr),
                "mse_unseen_operator": avg_mse(m, Xe_op, Ye_op),
                "mse_unseen_scene": avg_mse(m, Xe_sc, Ye_sc),
                "n_groups": -1, "fit_s": 0.0,
            })
            print(f"  k={kf:.0%} {name:11s} eps={len(r_idx):4d} "
                  f"op={results[-1]['mse_unseen_operator']:.5f} "
                  f"scene={results[-1]['mse_unseen_scene']:.5f}", flush=True)

    # Ceiling: everything that passes the gate.
    Xtr, Ytr = stack(hashes)
    m = RidgeChunkPolicy(alpha=1.0, seed=seed).fit(Xtr, Ytr)
    results.append({
        "seed": seed, "k_frac": 1.0, "k": len(hashes), "condition": "all_gated",
        "n_train_eps": len(hashes), "n_train_win": len(Xtr),
        "mse_unseen_operator": avg_mse(m, Xe_op, Ye_op),
        "mse_unseen_scene": avg_mse(m, Xe_sc, Ye_sc),
        "n_groups": int(len(np.unique(groups))), "fit_s": 0.0,
    })
    print(f"  k=100% all_gated   eps={len(hashes):4d} "
          f"op={results[-1]['mse_unseen_operator']:.5f} scene={results[-1]['mse_unseen_scene']:.5f}")

res = pd.DataFrame(results)
res.to_csv(REPORTS / "results.csv", index=False)
print(f"\nwrote {REPORTS/'results.csv'}  ({len(res)} rows)")

print("\n=== mean over seeds ===")
agg = res.groupby(["k_frac", "condition"])[["mse_unseen_operator", "mse_unseen_scene"]].agg(["mean", "std"])
print(agg.to_string())
