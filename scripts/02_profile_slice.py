"""Profile the fold_clothes slice to choose a working set with usable held-out splits."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

REPORTS = Path(__file__).resolve().parent.parent / "reports"
df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")

print(f"fold_clothes rows: {len(df)}")
print(f"columns: {list(df.columns)}\n")

human = df[df["embodiment"] == "human_bimanual"].copy()
print(f"human_bimanual: {len(human)}")

# num_frames sanity: -1/-2 are unset sentinels, and absurd lengths are un-segmented recordings.
bad = human[human["num_frames"] < 30]
print(f"  num_frames < 30 (sentinel/degenerate): {len(bad)}  values={sorted(bad['num_frames'].unique())[:10]}")
print(f"  num_frames > 3000 (likely un-segmented): {(human['num_frames'] > 3000).sum()}")

print("\n=== per-lab structure (human only) ===")
rows = []
for lab, g in human.groupby("lab"):
    rows.append(
        {
            "lab": lab,
            "episodes": len(g),
            "operators": g["operator"].nunique(),
            "scenes": g["scene"].nunique(),
            "median_frames": int(g["num_frames"].median()),
            "has_zarr": (g["zarr_processed_path"].fillna("").str.strip() != "").sum(),
        }
    )
print(pd.DataFrame(rows).sort_values("episodes", ascending=False).to_string(index=False))

print("\n=== operator x scene cross-tab for the largest labs ===")
for lab in ["rl2", "mecka", "microagi"]:
    g = human[human["lab"] == lab]
    if not len(g):
        continue
    print(f"\n--- {lab}: {len(g)} eps, {g['operator'].nunique()} operators, {g['scene'].nunique()} scenes")
    ct = pd.crosstab(g["operator"], g["scene"])
    print(f"    operator x scene cells populated: {(ct > 0).sum().sum()} / {ct.size}")
    print(f"    episodes per operator: {g['operator'].value_counts().describe()[['min','50%','max']].to_dict()}")
    print(f"    scenes: {sorted(g['scene'].fillna('').unique())[:15]}")

print("\n=== zarr path examples ===")
for p in human["zarr_processed_path"].dropna().unique()[:5]:
    print("  ", p)
