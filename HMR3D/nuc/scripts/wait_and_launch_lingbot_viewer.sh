#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/nyu/Codespace/CVPR"
PRED_REL="${1:?missing predictions npz relative path}"
SUMMARY_REL="${2:?missing summary json relative path}"
PORT="${3:-8080}"

PRED_PATH="$ROOT/$PRED_REL"
SUMMARY_PATH="$ROOT/$SUMMARY_REL"

while [[ ! -f "$PRED_PATH" || ! -f "$SUMMARY_PATH" ]]; do
  sleep 15
done

pkill -f "view_lingbot_predictions_viser.py --predictions-npz $PRED_REL" || true
pkill -f "view_lingbot_predictions_viser.py --port $PORT" || true

cd "$ROOT"
exec cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/view_lingbot_predictions_viser.py \
  --predictions-npz "$PRED_REL" \
  --summary-json "$SUMMARY_REL" \
  --port "$PORT" \
  --downsample-factor 1 \
  --point-size 0.0015 \
  --init-conf-threshold 1.0
