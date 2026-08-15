"""Per-episode quality signals and embedding features, computed from pose arrays alone.

Everything here is deterministic and inspectable. No learned components, no LLM judge.
That is a deliberate constraint: a curation engine has to be cheap enough to run over the
whole dataset, and it has to be auditable when it tells you to throw an episode away.

Array conventions (EgoVerse human_bimanual):
  {left,right}.obs_ee_pose      (T, 7)   xyz + quaternion
  {left,right}.obs_wrist_pose   (T, 7)   xyz + quaternion
  obs_head_pose                 (T, 7)   xyz + quaternion, from visual-inertial SLAM
  {left,right}.obs_keypoints    (T, 63)  21 MANO keypoints x 3 (frame convention unresolved)
  obs_eye_gaze                  (T, 3)
"""

from __future__ import annotations

import numpy as np

FPS = 30.0

# NOTE: we deliberately compute nothing from camera geometry. An earlier version projected
# hand keypoints into the image to measure "hands out of frame"; two different projection
# models both put zero keypoints inside the frame on images where the hands are obviously
# visible (scripts/22_projection_check.py), so the signal was removed rather than guessed at
# again. Everything below uses durations, distances and motion only.


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a).all(axis=tuple(range(1, a.ndim)))] if a.ndim > 1 else a[np.isfinite(a)]


def _speed(xyz: np.ndarray) -> np.ndarray:
    """Per-frame speed (m/s) of an (T,3) position track."""
    if len(xyz) < 2:
        return np.zeros(0)
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
    return d * FPS


def _frozen_stats(a: np.ndarray) -> tuple[float, int]:
    """Fraction of frame-to-frame steps that are exactly identical, and the longest such run.

    Exact equality (not a tolerance) is the right test: a tracker that has dropped out
    republishes the previous value bit-for-bit, whereas genuinely slow motion still
    dithers in the low bits.
    """
    if len(a) < 2:
        return 0.0, 0
    same = np.all(np.isclose(np.diff(a, axis=0), 0.0, atol=0.0, rtol=0.0), axis=tuple(range(1, a.ndim)))
    frac = float(same.mean())
    longest = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    return frac, int(longest)


def _kp(arr: np.ndarray) -> np.ndarray:
    """(T,63) -> (T,21,3)"""
    return arr.reshape(len(arr), 21, 3)


def quality_signals(ep: dict[str, np.ndarray]) -> dict:
    """Signals that answer: is this episode usable as a training target at all?"""
    out: dict[str, float] = {}

    lee = ep.get("left__obs_ee_pose")
    ree = ep.get("right__obs_ee_pose")
    head = ep.get("obs_head_pose")
    lkp = ep.get("left__obs_keypoints")
    rkp = ep.get("right__obs_keypoints")

    T = len(lee) if lee is not None else 0
    out["n_frames"] = float(T)
    out["duration_s"] = T / FPS

    # --- tracking dropout -------------------------------------------------
    nan_fracs = []
    for name, a in [("left_ee", lee), ("right_ee", ree), ("head", head),
                    ("left_kp", lkp), ("right_kp", rkp)]:
        if a is None:
            out[f"nan_{name}"] = 1.0
            nan_fracs.append(1.0)
            continue
        f = float((~np.isfinite(a)).mean())
        out[f"nan_{name}"] = f
        nan_fracs.append(f)
    out["nan_max"] = float(max(nan_fracs))

    # --- frozen tracker ---------------------------------------------------
    frozen_fracs, frozen_runs = [], []
    for name, a in [("left_ee", lee), ("right_ee", ree), ("head", head)]:
        if a is None or len(a) < 2:
            continue
        fr, run = _frozen_stats(np.nan_to_num(a))
        out[f"frozen_frac_{name}"] = fr
        out[f"frozen_run_{name}"] = float(run)
        frozen_fracs.append(fr)
        frozen_runs.append(run)
    out["frozen_frac_max"] = float(max(frozen_fracs)) if frozen_fracs else 1.0
    out["frozen_run_max_s"] = float(max(frozen_runs)) / FPS if frozen_runs else 0.0

    # --- motion -----------------------------------------------------------
    speeds = []
    for name, a in [("left", lee), ("right", ree)]:
        if a is None or len(a) < 2:
            out[f"path_len_{name}"] = 0.0
            out[f"speed_mean_{name}"] = 0.0
            continue
        xyz = np.nan_to_num(a[:, :3])
        s = _speed(xyz)
        speeds.append(s)
        out[f"path_len_{name}"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum())
        out[f"speed_mean_{name}"] = float(s.mean())
        out[f"net_disp_{name}"] = float(np.linalg.norm(xyz[-1] - xyz[0]))
    out["motion_energy"] = float(np.concatenate(speeds).mean()) if speeds else 0.0
    out["path_len_total"] = out.get("path_len_left", 0.0) + out.get("path_len_right", 0.0)

    # --- annotations ------------------------------------------------------
    out["has_annotations"] = float("annotations" in ep and len(ep["annotations"]) > 0)

    return out


def embedding_features(ep: dict[str, np.ndarray]) -> dict:
    """Features describing *what kind of episode this is* — the space we cover over.

    Distinct from quality signals: these are meant to spread episodes apart so that
    coverage maximisation has a meaningful geometry to work in.
    """
    out: dict[str, float] = {}

    lee = ep.get("left__obs_ee_pose")
    ree = ep.get("right__obs_ee_pose")
    head = ep.get("obs_head_pose")
    lkp = ep.get("left__obs_keypoints")
    rkp = ep.get("right__obs_keypoints")

    T = len(lee) if lee is not None else 0
    out["f_duration_s"] = T / FPS
    out["f_log_duration"] = float(np.log1p(T / FPS))

    for name, a in [("left", lee), ("right", ree)]:
        if a is None or len(a) < 3:
            continue
        xyz = np.nan_to_num(a[:, :3])
        s = _speed(xyz)
        acc = np.diff(s) * FPS if len(s) > 1 else np.zeros(1)

        out[f"f_{name}_path"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum())
        out[f"f_{name}_speed_mean"] = float(s.mean())
        out[f"f_{name}_speed_std"] = float(s.std())
        out[f"f_{name}_speed_p90"] = float(np.percentile(s, 90))
        out[f"f_{name}_acc_std"] = float(acc.std())
        # Workspace: how much volume this hand swept.
        ext = xyz.max(axis=0) - xyz.min(axis=0)
        out[f"f_{name}_ext_x"], out[f"f_{name}_ext_y"], out[f"f_{name}_ext_z"] = map(float, ext)
        out[f"f_{name}_bbox_vol"] = float(np.prod(np.maximum(ext, 1e-6)))
        out[f"f_{name}_centroid_x"], out[f"f_{name}_centroid_y"], out[f"f_{name}_centroid_z"] = map(
            float, xyz.mean(axis=0)
        )
        # Fraction of the episode this hand was actually moving.
        out[f"f_{name}_active_frac"] = float((s > 0.02).mean())

    # --- bimanual coordination -------------------------------------------
    if lee is not None and ree is not None and len(lee) > 2 and len(ree) > 2:
        ls, rs = _speed(np.nan_to_num(lee[:, :3])), _speed(np.nan_to_num(ree[:, :3]))
        m = min(len(ls), len(rs))
        if m > 2 and ls[:m].std() > 1e-9 and rs[:m].std() > 1e-9:
            out["f_bimanual_speed_corr"] = float(np.corrcoef(ls[:m], rs[:m])[0, 1])
        else:
            out["f_bimanual_speed_corr"] = 0.0
        d = np.linalg.norm(np.nan_to_num(lee[:, :3]) - np.nan_to_num(ree[:, :3]), axis=-1)
        out["f_hand_dist_mean"] = float(d.mean())
        out["f_hand_dist_std"] = float(d.std())
        out["f_hand_dist_min"] = float(d.min())

    # --- hand shape (grasp signature) ------------------------------------
    for name, a in [("left", lkp), ("right", rkp)]:
        if a is None or len(a) == 0:
            continue
        k = _kp(np.nan_to_num(a))
        # Aperture: spread of fingertips about the wrist.
        spread = np.linalg.norm(k - k[:, :1, :], axis=-1).mean(axis=1)
        out[f"f_{name}_aperture_mean"] = float(spread.mean())
        out[f"f_{name}_aperture_std"] = float(spread.std())
        out[f"f_{name}_aperture_range"] = float(spread.max() - spread.min())

    # --- head / viewpoint -------------------------------------------------
    if head is not None and len(head) > 2:
        hxyz = np.nan_to_num(head[:, :3])
        out["f_head_path"] = float(np.linalg.norm(np.diff(hxyz, axis=0), axis=-1).sum())
        ext = hxyz.max(axis=0) - hxyz.min(axis=0)
        out["f_head_ext"] = float(np.linalg.norm(ext))
        hq = np.nan_to_num(head[:, 3:7])
        dq = np.linalg.norm(np.diff(hq, axis=0), axis=-1)
        out["f_head_rot_energy"] = float(dq.mean() * FPS)

    return out
