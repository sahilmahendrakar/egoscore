"""Cross-lab prevalence audit: does the quality gate actually discriminate between labs?

A gate that fires at the same rate everywhere is not measuring quality, it is measuring
its own thresholds. Running it against a second lab is the cheapest available check.

Important caveat handled here: the hands-out-of-frame signal depends on camera
intrinsics, which we only verified for the Aria rig used by rl2. For any non-Aria lab we
report the intrinsics-independent signals only, and say so, rather than quietly
reporting a number we cannot stand behind.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore.features import quality_signals

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

# Every rule the gate now uses. None depends on camera geometry, so all of them are
# directly comparable across rigs -- which is the whole point of running this.
ABS_RULES = [
    ("tracking_dropout", "nan_max", 0.05, "gt"),
    ("frozen_tracker", "frozen_run_max_s", 2.0, "gt"),
    ("too_short", "duration_s", 3.0, "lt"),
    ("no_motion", "path_len_total", 0.10, "lt"),
]
# Median-relative rules are computed per lab, since "typical episode" differs by 14x
# between labs and a shared threshold would be meaningless.
REL_RULES = [
    ("runaway_length", "duration_s", 3.0, "gt"),
    ("suspiciously_short", "duration_s", 0.15, "lt"),
]
RIG_INDEPENDENT = ABS_RULES

LABS = [("rl2", ROOT / "data" / "poses", "aria_gen1"),
        ("mecka", ROOT / "data" / "poses_mecka", "mecka"),
        ("microagi", ROOT / "data" / "poses_microagi", "microagi")]

meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")

rows = []
for lab, d, rig in LABS:
    if not d.exists():
        print(f"skip {lab}: {d} missing")
        continue
    files = sorted(d.glob("*.npz"))
    print(f"{lab}: {len(files)} episodes")
    for f in files:
        try:
            with np.load(f, allow_pickle=True) as z:
                ep = {k: z[k] for k in z.files}
            q = quality_signals(ep)
            q["episode_hash"] = f.stem
            q["lab"] = lab
            q["rig"] = rig
            q["n_arrays"] = len(ep)
            rows.append(q)
        except Exception as e:
            print(f"  ERROR {f.stem}: {type(e).__name__}: {e}")

df = pd.DataFrame(rows)
if df.empty:
    sys.exit("no episodes")

# Absolute rules, identical everywhere.
for name, col, thresh, op in ABS_RULES:
    v = df[col].astype(float)
    df[f"rule_{name}"] = (v > thresh) if op == "gt" else (v < thresh)

# Median-relative rules, recomputed within each lab.
for name, col, mult, op in REL_RULES:
    df[f"rule_{name}"] = False
    for lab, g in df.groupby("lab"):
        t = float(g[col].median()) * mult
        hit = (g[col] > t) if op == "gt" else (g[col] < t)
        df.loc[g.index, f"rule_{name}"] = hit.values

rule_cols = [f"rule_{n}" for n, _, _, _ in ABS_RULES + REL_RULES]
df["drop_rig_independent"] = df[rule_cols].any(axis=1)

df.to_csv(REPORTS / "cross_lab_quality.csv", index=False)

print("\n=== cross-lab prevalence ===")
summary = []
for lab, g in df.groupby("lab"):
    row = {"lab": lab, "n_episodes": len(g)}
    for name, _, _, _ in ABS_RULES + REL_RULES:
        row[name] = f"{100.0*g[f'rule_{name}'].mean():.1f}%"
    row["ANY"] = f"{100.0*g['drop_rig_independent'].mean():.1f}%"
    row["median_dur_s"] = f"{g['duration_s'].median():.0f}"
    row["median_motion"] = f"{g['motion_energy'].median():.2f}"
    summary.append(row)
s = pd.DataFrame(summary)
print(s.to_string(index=False))
s.to_csv(REPORTS / "cross_lab_summary.csv", index=False)

print("\n=== distribution comparison ===")
for col in ["duration_s", "motion_energy", "path_len_total", "frozen_run_max_s", "nan_max"]:
    line = f"  {col:20s}"
    for lab, g in df.groupby("lab"):
        line += f"  {lab}: p50={g[col].median():8.2f} p95={g[col].quantile(0.95):8.2f}"
    print(line)

print("\nAll rules above are camera-geometry-free, so the columns are directly comparable.")
print("Median-relative rules use each lab's own median episode length.")
