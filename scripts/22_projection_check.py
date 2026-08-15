"""Settle the keypoint frame question by drawing the projection on real frames.

Two candidate models:
  A) keypoints are already in the camera frame  (what we assumed)
  B) keypoints are in a world frame and must be transformed by the head pose first

Whichever puts the dots on the hands is the right one. This is the check we should have
run before ever shipping an "out of frame" number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import zarr

from egoscore.access import load_creds, r2_fs

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"

EP = sys.argv[1] if len(sys.argv) > 1 else "2025-10-29-22-04-05-569000"
K = np.array([[266.50860444, 0.0, 320.0], [0.0, 266.50860444, 240.0], [0.0, 0.0, 1.0]])


def quat_to_R(q):
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project_pinhole(P):
    """(N,3) camera-frame points -> (N,2) pixels, plus a validity mask (z forward)."""
    z = P[:, 2]
    ok = z > 0.01
    uv = np.full((len(P), 2), -1e6)
    uv[ok, 0] = K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]
    return uv, ok


meta = pd.read_csv(REPORTS / "slice_fold_clothes.csv").set_index("episode_hash")
g = zarr.open_group(
    zarr.storage.FsspecStore(r2_fs(load_creds()),
                             path=meta.loc[EP, "zarr_processed_path"].replace("s3://", "")),
    mode="r",
)
n = int(g.attrs.get("total_frames", 0))
lkp = g["left.obs_keypoints"][:n].reshape(n, 21, 3)
rkp = g["right.obs_keypoints"][:n].reshape(n, 21, 3)
head = g["obs_head_pose"][:n]

idx = [int(n * f) for f in (0.2, 0.4, 0.6, 0.8)]
panels = []
for i in idx:
    blob = g["images.front_1"][i:i + 1][0]
    img = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)

    Rh = quat_to_R(head[i, 3:7])
    th = head[i, :3]

    for label, tint in (("A_raw", (0, 0, 255)), ("B_headframe", (0, 255, 0))):
        panel = img.copy()
        inside_count = 0
        for kp, _hand in ((lkp[i], "L"), (rkp[i], "R")):
            P = kp if label == "A_raw" else (kp - th) @ Rh
            uv, ok = project_pinhole(P)
            for j in range(21):
                if not ok[j]:
                    continue
                u, v = uv[j]
                if 0 <= u < 640 and 0 <= v < 480:
                    inside_count += 1
                    cv2.circle(panel, (int(u), int(v)), 4, tint, -1)
                    cv2.circle(panel, (int(u), int(v)), 4, (255, 255, 255), 1)
        cv2.putText(panel, f"{label}: {inside_count}/42 in frame", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
        cv2.putText(panel, f"{label}: {inside_count}/42 in frame", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tint, 1)
        panels.append((label, i, panel))

rows = {}
for label, i, panel in panels:
    rows.setdefault(label, []).append(panel)

strips = []
for label in ("A_raw", "B_headframe"):
    strip = np.concatenate(rows[label], axis=1)
    strips.append(strip)
out = np.concatenate(strips, axis=0)
dest = FIGS / "projection_check.png"
cv2.imwrite(str(dest), out)
print(f"wrote {dest}")

# Numeric summary over the whole episode, both models.
for label in ("A_raw", "B_headframe"):
    fr = []
    for kp in (lkp, rkp):
        if label == "A_raw":
            P = kp.reshape(-1, 3)
        else:
            P = np.einsum("tij,tkj->tki", np.stack([quat_to_R(q) for q in head[:, 3:7]]).transpose(0, 2, 1),
                          kp - head[:, None, :3]).reshape(-1, 3)
        uv, ok = project_pinhole(P)
        inside = ok & (uv[:, 0] >= 0) & (uv[:, 0] < 640) & (uv[:, 1] >= 0) & (uv[:, 1] < 480)
        fr.append(1.0 - inside.reshape(len(kp), 21).mean(axis=1).mean())
    print(f"  {label:14s} mean fraction of keypoints OUT of frame: L={fr[0]:.3f} R={fr[1]:.3f}")
