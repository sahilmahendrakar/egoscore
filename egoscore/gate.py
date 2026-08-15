"""The quality gate: which episodes are unusable as training targets, and why.

Every rule is deterministic and carries a human-readable reason, so a drop decision can
always be audited back to the signal that caused it. Thresholds are set from the
observed distribution of the slice rather than asserted a priori — see
``describe_thresholds`` for what each one actually rejects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Absolute rules: these describe broken data, not unusual data.
#
# A NOTE ON THE OFF-AXIS RULE. This started life as "hands out of frame" and that claim was
# wrong: we could never establish the keypoint-to-pixel mapping, and three separate attempts
# (pinhole, fisheye, MediaPipe on the pixels) all disagree with what the frames plainly show.
#
# The underlying measurement is sound, so it is kept under an accurate name. The angle
# between the hand and the camera's optical axis needs no lens model, and it separates the
# slice cleanly: most rl2 episodes sit around 19 deg, a tail sits around 57 deg. Those
# episodes really do show the demonstrator working out at the edge of the view.
#
# The claim is "the hands are far off-axis", which is checkable. It is not "the hands are
# invisible", which we cannot support.
HARD_RULES = [
    ("tracking_dropout", "nan_max", 0.05, "gt", "non-finite pose/keypoint values in >5% of frames"),
    ("frozen_tracker", "frozen_run_max_s", 2.0, "gt", "pose held bit-identical for >2s (tracker dropout)"),
    ("too_short", "duration_s", 3.0, "lt", "shorter than 3s — cannot contain a fold"),
    ("no_motion", "path_len_total", 0.10, "lt", "hands travelled <10cm in total"),
    # Named for what it measures, not for what we first assumed it meant. See the note in
    # features.py: the angle is solid, the claim "therefore out of frame" is not.
    ("hands_far_off_axis", "offaxis_max", 0.50, "gt",
     "a hand is held >45 deg off the camera axis for most of the episode"),
]

# Relative rules, scaled to the slice's own typical episode.
#
# We used to use the 99th percentile here, which is circular: a percentile rule drops
# exactly 1% of any dataset you point it at, however clean or dirty. A multiple of the
# median can fire at 0% on tidy data and at 30% on a badly segmented dump, which is what
# a prevalence audit needs in order to say anything.
MEDIAN_MULTIPLE_RULES = [
    ("runaway_length", "duration_s", 3.0, "gt",
     "more than 3x the median episode length — an un-segmented recording, not one demo"),
    ("suspiciously_short", "duration_s", 0.15, "lt",
     "under 15% of the median episode length — a fragment, not a demo"),
]


def apply_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with `keep`, `drop_reason`, and one boolean column per rule."""
    out = df.copy()
    reasons: list[list[str]] = [[] for _ in range(len(out))]

    for name, col, thresh, op, _desc in HARD_RULES:
        if col not in out.columns:
            out[f"rule_{name}"] = False
            continue
        v = out[col].astype(float)
        hit = v > thresh if op == "gt" else v < thresh
        hit = hit.fillna(True)  # a missing signal is itself disqualifying
        out[f"rule_{name}"] = hit
        for i in np.flatnonzero(hit.to_numpy()):
            reasons[i].append(name)

    for name, col, mult, op, _desc in MEDIAN_MULTIPLE_RULES:
        if col not in out.columns:
            out[f"rule_{name}"] = False
            continue
        v = out[col].astype(float)
        thresh = float(v.median()) * mult
        hit = (v > thresh) if op == "gt" else (v < thresh)
        hit = hit.fillna(True)
        out[f"rule_{name}"] = hit
        out.attrs[f"thresh_{name}"] = thresh
        for i in np.flatnonzero(hit.to_numpy()):
            reasons[i].append(name)

    out["drop_reason"] = ["|".join(r) for r in reasons]
    out["keep"] = out["drop_reason"] == ""
    return out


def describe_thresholds(gated: pd.DataFrame) -> pd.DataFrame:
    """Per-rule prevalence, for the audit table in the report."""
    rows = []
    n = len(gated)
    for name, col, thresh, op, desc in HARD_RULES:
        c = f"rule_{name}"
        if c not in gated:
            continue
        rows.append({
            "rule": name, "signal": col, "test": f"{op} {thresh}",
            "n_flagged": int(gated[c].sum()),
            "pct": 100.0 * gated[c].mean(),
            "meaning": desc,
        })
    for name, col, mult, op, desc in MEDIAN_MULTIPLE_RULES:
        c = f"rule_{name}"
        if c not in gated:
            continue
        t = gated.attrs.get(f"thresh_{name}", float("nan"))
        rows.append({
            "rule": name, "signal": col, "test": f"{op} {mult}x median ({t:.0f}s)",
            "n_flagged": int(gated[c].sum()),
            "pct": 100.0 * gated[c].mean(),
            "meaning": desc,
        })
    rows.append({
        "rule": "ANY (dropped)", "signal": "-", "test": "-",
        "n_flagged": int((~gated["keep"]).sum()),
        "pct": 100.0 * (~gated["keep"]).mean(),
        "meaning": f"union of all rules over {n} episodes",
    })
    return pd.DataFrame(rows)
