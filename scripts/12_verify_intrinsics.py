"""Verify that camera intrinsics really are constant across the rl2 slice.

features.py hardcodes a single Aria intrinsics matrix for the hands-out-of-frame signal.
That is only legitimate if the intrinsics do not vary across episodes, so we check it
rather than assume it.
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from egoscore.access import load_creds, r2_client
from egoscore.features import ARIA_K

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

N_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 60

df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
rl2 = df[(df["lab"] == "rl2") & (df["embodiment"] == "human_bimanual")]
rl2 = rl2[rl2["zarr_processed_path"].fillna("").str.strip() != ""]
sample = rl2.sample(n=min(N_SAMPLE, len(rl2)), random_state=0)
print(f"checking intrinsics on {len(sample)} of {len(rl2)} rl2 episodes")

c = r2_client(load_creds())


def fetch(path: str):
    key = path.replace("s3://rldb/", "").rstrip("/") + "/zarr.json"
    try:
        body = c.get_object(Bucket="rldb", Key=key)["Body"].read()
        attrs = json.loads(body).get("attributes", {})
        K = attrs.get("intrinsics", {})
        return json.dumps(K, sort_keys=True)
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


with ThreadPoolExecutor(max_workers=24) as pool:
    vals = list(pool.map(fetch, sample["zarr_processed_path"]))

counts = Counter(vals)
print(f"\ndistinct intrinsics values: {len(counts)}")
for v, n in counts.most_common(5):
    print(f"  n={n}: {v[:200]}")

ok = len(counts) == 1 and not next(iter(counts)).startswith("ERROR")
if ok:
    K = json.loads(next(iter(counts)))
    front = np.array(K["front_1"])
    print(f"\nfront_1 K =\n{front}")
    match = np.allclose(front[:3, :3], ARIA_K, atol=1e-6)
    print(f"\nmatches ARIA_K hardcoded in features.py: {match}")
    print("VERIFIED" if match else "MISMATCH — update ARIA_K")
else:
    print("\nNOT CONSTANT — features.py must read intrinsics per episode instead.")
