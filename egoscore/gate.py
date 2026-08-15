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
HARD_RULES = [
    ("tracking_dropout", "nan_max", 0.05, "gt", "non-finite pose/keypoint values in >5% of frames"),
    ("frozen_tracker", "frozen_run_max_s", 2.0, "gt", "pose held bit-identical for >2s (tracker dropout)"),
    ("hands_out_of_frame", "oof_max", 0.50, "gt", "a hand projects outside the image in >50% of frames"),
    ("too_short", "duration_s", 3.0, "lt", "shorter than 3s — cannot contain a fold"),
    ("no_motion", "path_len_total", 0.10, "lt", "hands travelled <10cm in total"),
]

# Relative rules: set from the slice's own distribution.
QUANTILE_RULES = [
    ("truncated_or_runaway_hi", "duration_s", 0.99, "gt", "duration above the 99th percentile — likely un-segmented recording"),
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

    for name, col, q, op, _desc in QUANTILE_RULES:
        if col not in out.columns:
            out[f"rule_{name}"] = False
            continue
        v = out[col].astype(float)
        thresh = v.quantile(q)
        hit = (v > thresh) if op == "gt" else (v < thresh)
        hit = hit.fillna(True)
        out[f"rule_{name}"] = hit
        out.attrs[f"thresh_{name}"] = float(thresh)
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
    for name, col, q, op, desc in QUANTILE_RULES:
        c = f"rule_{name}"
        if c not in gated:
            continue
        t = gated.attrs.get(f"thresh_{name}", float("nan"))
        rows.append({
            "rule": name, "signal": col, "test": f"{op} q{q} ({t:.1f})",
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
