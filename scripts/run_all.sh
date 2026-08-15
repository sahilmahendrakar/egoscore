#!/usr/bin/env bash
# Reproduce the whole EgoScore result from a clean checkout.
#
# Prerequisite: EgoVerse's public read-only AWS keys on disk. They are published in the
# EgoVerse README; write them to ~/.egoverse_aws_credentials in aws-credentials format:
#
#   [default]
#   aws_access_key_id = <key from EgoVerse README>
#   aws_secret_access_key = <secret from EgoVerse README>
#   region = us-east-2
#
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}

if [ ! -x "$PY" ]; then
  echo "==> creating venv"
  uv venv .venv --python 3.11
  uv pip install --python .venv/bin/python -r requirements.txt
fi

echo "==> 01: survey the episode table"
$PY scripts/01_survey.py

echo "==> 02: profile the fold_clothes slice"
$PY scripts/02_profile_slice.py

echo "==> 04: pull pose arrays for rl2 (~10 min, 1.2 GB)"
$PY scripts/04_pull_poses.py rl2

echo "==> 12: verify camera intrinsics are constant (assumption check)"
$PY scripts/12_verify_intrinsics.py 80

echo "==> 06: extract quality signals + embedding features"
$PY scripts/06_extract_features.py

echo "==> 07: run the experiment (gate, selection, Avg-MSE)"
$PY scripts/07_experiment.py

echo "==> 08: paired analysis + figures"
$PY scripts/08_analyze.py

echo "==> 04b: pull mecka + microagi samples for the cross-lab audit (~20 min)"
$PY scripts/04_pull_poses.py mecka 400
$PY scripts/04_pull_poses.py microagi 1200

echo "==> 10: cross-lab quality audit"
$PY scripts/10_cross_lab_audit.py

echo "==> 09: generate the validation report"
$PY scripts/09_report.py

echo "==> 11: build the summary slide"
$PY scripts/11_slide.py

echo "==> 13: proxy-hyperparameter sensitivity sweep"
$PY scripts/13_sensitivity.py

echo "==> 25: price the quality gate on microagi (equal-episode vs equal-frame budget)"
$PY scripts/25_gate_value.py microagi

echo "==> 14-18: demo thumbnails, projection, demo page and slide deck"
$PY scripts/14_fetch_thumbs.py rl2
$PY scripts/14_fetch_thumbs.py microagi 1200
$PY scripts/15_demo_data.py
$PY scripts/16_build_demo.py
$PY scripts/17_manifold_fig.py
$PY scripts/19_episode_strips.py microagi
$PY scripts/20_selector_explainer.py
$PY scripts/21_mse_explainer.py
$PY scripts/18_build_deck.py

echo
echo "Done."
echo "  report : reports/validation.md"
echo "  slide  : reports/egoscore_summary_slide.png"
echo "  demo   : demo/egoscore_demo.html"
echo "  deck   : demo/egoscore_deck.html"
