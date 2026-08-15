"""Are there real defects in rl2 that the current six rules miss?

The gate drops 1 of 572 rl2 episodes. Either the data is genuinely that clean, or we are
not testing for the things that are wrong with it. This measures candidate signals so the
answer comes from the distribution rather than from wanting a bigger number.

Nothing here is thresholded yet. We are looking for signals with a real tail.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FPS = 30.0

LABS = {"rl2": ROOT / "data" / "poses", "microagi": ROOT / "data" / "poses_microagi"}


def signals(ep: dict) -> dict:
    out = {}
    lee = ep.get("left__obs_ee_pose")
    ree = ep.get("right__obs_ee_pose")
    head = ep.get("obs_head_pose")
    if lee is None or ree is None:
        return out
    T = len(lee)
    out["duration_s"] = T / FPS

    # --- teleports: physically impossible frame-to-frame hand displacement -----
    jumps = []
    for a in (lee, ree):
        d = np.linalg.norm(np.diff(np.nan_to_num(a[:, :3]), axis=0), axis=-1) * FPS
        jumps.append(d)
    j = np.concatenate(jumps)
    out["max_hand_speed"] = float(j.max()) if len(j) else 0.0
    out["frac_over_5ms"] = float((j > 5.0).mean()) if len(j) else 0.0

    # --- head teleports (SLAM relocalisation glitches) -------------------------
    if head is not None and len(head) > 1:
        hd = np.linalg.norm(np.diff(np.nan_to_num(head[:, :3]), axis=0), axis=-1) * FPS
        out["max_head_speed"] = float(hd.max())
        # A quaternion that flips sign frame-to-frame is a tracker discontinuity.
        q = np.nan_to_num(head[:, 3:7])
        dq = np.linalg.norm(np.diff(q, axis=0), axis=-1)
        out["max_quat_step"] = float(dq.max())

    # --- dead air at the start / end -------------------------------------------
    sp = np.maximum(jumps[0], jumps[1][: len(jumps[0])]) if len(jumps[0]) else np.zeros(1)
    moving = sp > 0.05
    if moving.any():
        first, last = int(np.argmax(moving)), int(len(moving) - np.argmax(moving[::-1]))
        out["lead_in_s"] = first / FPS
        out["lead_out_s"] = (len(moving) - last) / FPS
    else:
        out["lead_in_s"] = out["lead_out_s"] = T / FPS
    out["dead_air_frac"] = (out["lead_in_s"] + out["lead_out_s"]) / max(T / FPS, 1e-6)

    # --- how much work per second (a long episode doing nothing is suspect) -----
    path = sum(float(np.linalg.norm(np.diff(np.nan_to_num(a[:, :3]), axis=0), axis=-1).sum())
               for a in (lee, ree))
    out["path_per_s"] = path / max(T / FPS, 1e-6)
    out["idle_frac"] = float((sp < 0.02).mean())
    return out


rows = []
for lab, d in LABS.items():
    if not d.exists():
        continue
    for f in sorted(d.glob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            s = signals({k: z[k] for k in z.files})
        if s:
            s["lab"] = lab
            s["episode_hash"] = f.stem
            rows.append(s)

df = pd.DataFrame(rows)
df.to_csv(REPORTS / "candidate_signals.csv", index=False)

cols = ["max_hand_speed", "frac_over_5ms", "max_head_speed", "max_quat_step",
        "lead_in_s", "lead_out_s", "dead_air_frac", "path_per_s", "idle_frac"]
for lab, g in df.groupby("lab"):
    print(f"\n=== {lab}  (n={len(g)}) ===")
    print(g[cols].describe(percentiles=[0.5, 0.9, 0.99]).T[
        ["50%", "90%", "99%", "max"]].round(3).to_string())

print("\n=== how many rl2 episodes would each candidate flag? ===")
r = df[df.lab == "rl2"]
for name, expr in [
    ("hand teleport >5 m/s", r.max_hand_speed > 5.0),
    ("hand teleport >10 m/s", r.max_hand_speed > 10.0),
    ("head teleport >3 m/s", r.get("max_head_speed", pd.Series(dtype=float)) > 3.0),
    ("dead air >30% of episode", r.dead_air_frac > 0.30),
    ("lead-in >10s", r.lead_in_s > 10.0),
    ("idle >50% of frames", r.idle_frac > 0.50),
    ("path_per_s < 0.3 m/s", r.path_per_s < 0.30),
]:
    try:
        n = int(expr.sum())
        print(f"  {name:28s} {n:4d}  ({100*n/len(r):.1f}%)")
    except Exception as e:
        print(f"  {name:28s} n/a ({e})")
