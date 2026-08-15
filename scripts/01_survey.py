"""Survey the EgoVerse episode table and pick our working slice.

Writes reports/episodes_all.csv and reports/slice_fold_clothes.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from egoscore.access import episode_table

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

REPORTS = Path(__file__).resolve().parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)

print("querying episode table ...")
df = episode_table()
print(f"rows: {len(df)}  cols: {list(df.columns)}")

df.to_csv(REPORTS / "episodes_all.csv", index=False)

live = df[~df["is_deleted"].fillna(False)] if "is_deleted" in df else df
print(f"\nlive (not deleted): {len(live)}")

for col in ["embodiment", "lab", "rig_name", "task"]:
    if col in live.columns:
        vc = live[col].value_counts()
        print(f"\n--- {col} ({vc.size} distinct) ---")
        print(vc.head(20).to_string())

if "task" in live.columns:
    fc = live[live["task"] == "fold_clothes"]
    print(f"\n=== fold_clothes: {len(fc)} episodes ===")
    for col in ["lab", "embodiment", "scene", "operator"]:
        if col in fc.columns:
            print(f"  {col}: {fc[col].nunique()} distinct -> {fc[col].value_counts().head(10).to_dict()}")
    if "num_frames" in fc.columns:
        print("\n  num_frames describe:")
        print(fc["num_frames"].describe().to_string())
    fc.to_csv(REPORTS / "slice_fold_clothes.csv", index=False)
    print(f"\nwrote {REPORTS/'slice_fold_clothes.csv'}")
