"""Does dropping the runaway episodes actually help? Measure it on microagi.

We could not run the *selection* experiment on microagi, because it has no operator or
scene labels and so no way to hold out an unseen operator or an unseen scene. But that is
not required to price the *gate*. The gate's question is narrower:

    at a fixed episode budget, is a subset drawn from gated data better than one drawn
    from ungated data?

That only needs a held-out set, not a held-out *axis*. So we hold out a random 25% of
episodes and compare like for like at equal budget.

What this measures, and what it does not: the held-out episodes come from the same pool as
the training ones, so this is an in-distribution comparison, not a generalization test. It
answers "do the runaway episodes hurt?" It does not answer "does this transfer to a new
operator?" -- for that you need rl2. We report it on those terms.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore.features import embedding_features, quality_signals
from egoscore.gate import apply_gate
from egoscore.proxy import RidgeChunkPolicy, avg_mse, episode_windows

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

LAB = sys.argv[1] if len(sys.argv) > 1 else "microagi"
SEEDS = list(range(10))
POSES = ROOT / "data" / ("poses" if LAB == "rl2" else f"poses_{LAB}")

meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")

print(f"loading {LAB} ...")
rows, cache = [], {}
for f in sorted(POSES.glob("*.npz")):
    with np.load(f, allow_pickle=True) as z:
        ep = {k: z[k] for k in z.files}
    r = {"episode_hash": f.stem}
    r.update(quality_signals(ep))
    r.update(embedding_features(ep))
    rows.append(r)
    X, Y = episode_windows(ep)
    if len(X):
        cache[f.stem] = (X.astype(np.float32), Y.astype(np.float32))

feat = pd.DataFrame(rows)
gated = apply_gate(feat)
print(f"{len(gated)} episodes, {int((~gated['keep']).sum())} dropped "
      f"({100*(~gated['keep']).mean():.1f}%), {len(cache)} with windows")


def stack(hs):
    xs = [cache[h][0] for h in hs if h in cache]
    ys = [cache[h][1] for h in hs if h in cache]
    if not xs:
        return np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32)
    return np.vstack(xs), np.vstack(ys)


hashes = gated["episode_hash"].to_numpy()
keep = gated["keep"].to_numpy()

results = []
t0 = time.time()
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(hashes))
    n_eval = int(0.25 * len(perm))
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]

    # Evaluate only on episodes that pass the gate: a runaway episode in the *eval* set
    # would make the comparison about predicting junk, not about training on it.
    eval_h = hashes[eval_idx][keep[eval_idx]]
    Xe, Ye = stack(eval_h)
    if len(Xe) == 0:
        continue

    pool_all = hashes[train_idx]
    pool_gated = hashes[train_idx][keep[train_idx]]

    # Two budgets, because they answer different questions.
    #
    # equal-episodes: the naive comparison. It is confounded -- a runaway episode carries
    #   60x the windows of a normal one, so the ungated arm silently trains on far more
    #   frames. Any win it shows may just be "more data".
    # equal-frames:   the fair one. Both arms get the same number of training windows, so
    #   the only difference is *which* frames they come from.
    budgets = {"episodes": len(pool_gated)}
    win_budget = sum(len(cache[h][0]) for h in pool_gated if h in cache)

    def take_by_windows(pool, target, r):
        order = r.permutation(len(pool))
        chosen, tot = [], 0
        for i in order:
            h = pool[i]
            if h not in cache:
                continue
            chosen.append(h)
            tot += len(cache[h][0])
            if tot >= target:
                break
        return chosen

    arms = []
    for cond, pool in [("gated", pool_gated), ("ungated", pool_all)]:
        sel = rng.choice(pool, size=min(budgets["episodes"], len(pool)), replace=False)
        arms.append((f"{cond}_equal_episodes", sel))
        arms.append((f"{cond}_equal_frames",
                     take_by_windows(pool, win_budget, np.random.default_rng(seed + 991))))

    for cond, sel in arms:
        Xtr, Ytr = stack(sel)
        if len(Xtr) < 50:
            continue
        m = RidgeChunkPolicy(alpha=1.0, seed=seed).fit(Xtr, Ytr)
        results.append({
            "seed": seed, "condition": cond, "n_train_eps": len(sel),
            "n_train_win": len(Xtr), "n_eval_eps": len(eval_h), "n_eval_win": len(Xe),
            "avg_mse": avg_mse(m, Xe, Ye),
        })
    a = {r["condition"]: r for r in results if r["seed"] == seed}
    line = f"  seed {seed}:"
    for mode in ("equal_episodes", "equal_frames"):
        g, u = a.get(f"gated_{mode}"), a.get(f"ungated_{mode}")
        if g and u:
            d = 100 * (g["avg_mse"] - u["avg_mse"]) / u["avg_mse"]
            line += f"  [{mode}] gate {d:+.2f}%"
    print(line, flush=True)

res = pd.DataFrame(results)
res.to_csv(REPORTS / f"gate_value_{LAB}.csv", index=False)

from scipy.stats import binomtest, wilcoxon

piv = res.pivot(index="seed", columns="condition", values="avg_mse")
print(f"\n=== {LAB}: does dropping the runaway episodes help? ===")
for mode, note in [("equal_episodes", "same episode count (confounded: ungated gets more frames)"),
                   ("equal_frames", "same number of training windows -- the fair comparison")]:
    g, u = f"gated_{mode}", f"ungated_{mode}"
    if g not in piv or u not in piv:
        continue
    d = (piv[g] - piv[u]).dropna()
    pct = 100.0 * d / piv[u]
    wins = int((d < 0).sum())
    ge = res[res.condition == g]
    print(f"\n  [{mode}] {note}")
    print(f"    training episodes/arm: gated {ge.n_train_eps.mean():.0f}, "
          f"windows {ge.n_train_win.mean():.0f} vs ungated "
          f"{res[res.condition==u].n_train_win.mean():.0f}")
    print(f"    gated {piv[g].mean():.5f}   ungated {piv[u].mean():.5f}")
    print(f"    gate wins {wins}/{len(d)} seeds, mean {pct.mean():+.2f}%  "
          f"(sign p={binomtest(wins, len(d), 0.5).pvalue:.3f}, "
          f"wilcoxon p={wilcoxon(d).pvalue:.4f})")
print(f"\n({time.time()-t0:.0f}s)")
