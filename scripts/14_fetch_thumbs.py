"""Fetch one representative frame per rl2 episode and downscale it to a thumbnail.

Used only by the demo page. Ranged reads against the zarr shard index make this cheap:
one frame is ~70 KB on the wire, versus 168 MB for a whole episode's image array.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
LAB = sys.argv[1] if len(sys.argv) > 1 else "rl2"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

OUT = ROOT / "data" / ("thumbs" if LAB == "rl2" else f"thumbs_{LAB}")
OUT.mkdir(parents=True, exist_ok=True)

THUMB_W = 128
JPEG_Q = 62

df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
rl2 = df[(df["lab"] == LAB) & (df["embodiment"] == "human_bimanual")]
rl2 = rl2[rl2["zarr_processed_path"].fillna("").str.strip() != ""].reset_index(drop=True)
if LIMIT and len(rl2) > LIMIT:
    rl2 = rl2.sample(n=LIMIT, random_state=0).reset_index(drop=True)
# Only fetch thumbs for episodes we actually pulled poses for.
have = {p.stem for p in (ROOT / "data" / ("poses" if LAB == "rl2" else f"poses_{LAB}")).glob("*.npz")}
rl2 = rl2[rl2["episode_hash"].isin(have)].reset_index(drop=True)
print(f"fetching thumbnails for {len(rl2)} {LAB} episodes")

creds = load_creds()
fs = r2_fs(creds)


def grab(row) -> dict:
    ep = row["episode_hash"]
    dest = OUT / f"{ep}.jpg"
    if dest.exists():
        return {"episode_hash": ep, "status": "cached"}
    try:
        g = zarr.open_group(
            zarr.storage.FsspecStore(fs, path=row["zarr_processed_path"].replace("s3://", "")),
            mode="r",
        )
        arr = g["images.front_1"]
        n = int(g.attrs.get("total_frames", arr.shape[0]))
        mid = max(0, min(n - 1, n // 2))
        # Slice rather than index: a scalar index returns nested 0-d object arrays.
        blob = arr[mid:mid + 1][0]
        img = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"episode_hash": ep, "status": "decode_failed"}
        h, w = img.shape[:2]
        small = cv2.resize(img, (THUMB_W, max(1, int(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        return {"episode_hash": ep, "status": "ok", "bytes": dest.stat().st_size}
    except Exception as e:
        return {"episode_hash": ep, "status": "error", "error": f"{type(e).__name__}: {e}"[:160]}


t0 = time.time()
res = []
with ThreadPoolExecutor(max_workers=24) as pool:
    futs = [pool.submit(grab, r) for _, r in rl2.iterrows()]
    for i, f in enumerate(as_completed(futs), 1):
        res.append(f.result())
        if i % 100 == 0 or i == len(futs):
            ok = sum(r["status"] in ("ok", "cached") for r in res)
            print(f"  {i}/{len(futs)}  ok={ok}  {time.time()-t0:.0f}s", flush=True)

r = pd.DataFrame(res)
print("\n" + r["status"].value_counts().to_string())
tot = sum(f.stat().st_size for f in OUT.glob("*.jpg"))
print(f"\n{len(list(OUT.glob('*.jpg')))} thumbs, {tot/1e6:.1f} MB total")
if (r["status"] == "error").any():
    print("\nerrors:")
    for e in r[r["status"] == "error"]["error"].head(3):
        print("  ", e)
