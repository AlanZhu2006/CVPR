#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHONPATH_ROOT="$ROOT/HMR3D/nuc/src"

SEQ="${SEQ:-$ROOT/cuVSLAM/examples/kitti/dataset/sequences/06}"
TRAJ="${TRAJ:-$SEQ/trajectory_tum.txt}"
CONFIG="${CONFIG:-$ROOT/HMR3D/nuc/configs/kitti06_v9_structured_init.yaml}"

JOB_NAME="${JOB_NAME:-kitti06_win16_step5}"
JOB_DIR="${JOB_DIR:-$ROOT/nuc_output/lingbot_dense_jobs/$JOB_NAME}"
JOB_OUTPUT_DIR="${JOB_OUTPUT_DIR:-$JOB_DIR/lingbot_output}"
DENSE_DIR="${DENSE_DIR:-$JOB_DIR/dense_geometry}"
GAUSSIAN_OUTPUT_DIR="${GAUSSIAN_OUTPUT_DIR:-$ROOT/nuc_output/${JOB_NAME}_dense_gaussian}"

WINDOW_KEYFRAMES="${WINDOW_KEYFRAMES:-16}"
FRAME_STEP="${FRAME_STEP:-5}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DENSE_STRIDE="${DENSE_STRIDE:-4}"
MIN_CONF="${MIN_CONF:-1.0}"
SUBMAP_ID="${SUBMAP_ID:-9165}"

REMOTE="${REMOTE:-gpu-worker}"
REMOTE_ROOT="${REMOTE_ROOT:-/media/chatsign/data-002/lingbot_dense_jobs}"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/media/chatsign/data-002/CVPR-remote-gpu}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/usr/bin/python3.10}"
LINGBOT_MAP_ROOT="${LINGBOT_MAP_ROOT:-/home/chatsign/work/lingbot-map}"
MODEL_PATH="${MODEL_PATH:-/media/chatsign/data-002/models/lingbot-map/lingbot-map-long.pt}"

if [[ ! -f "$TRAJ" ]]; then
  echo "Missing trajectory file: $TRAJ" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config file: $CONFIG" >&2
  exit 1
fi

echo "[1/4] Preparing KITTI06 LingBot dense job: $JOB_NAME"
PYTHONPATH="$PYTHONPATH_ROOT" \
python "$ROOT/HMR3D/nuc/scripts/prepare_lingbot_dense_job.py" \
  --sequence-path "$SEQ" \
  --trajectory-path "$TRAJ" \
  --config "$CONFIG" \
  --output-dir "$JOB_DIR" \
  --frame-step "$FRAME_STEP" \
  --max-frames "$MAX_FRAMES" \
  --window-keyframes "$WINDOW_KEYFRAMES"

echo "[2/4] Running remote LingBot dense inference on $REMOTE"
PYTHONPATH="$PYTHONPATH_ROOT" \
python "$ROOT/HMR3D/nuc/scripts/run_lingbot_dense_remote.py" \
  --remote "$REMOTE" \
  --job-dir "$JOB_DIR" \
  --remote-root "$REMOTE_ROOT" \
  --remote-repo-root "$REMOTE_REPO_ROOT" \
  --model-path "$MODEL_PATH" \
  --remote-python "$REMOTE_PYTHON" \
  --lingbot-map-root "$LINGBOT_MAP_ROOT"

echo "[3/4] Exporting normalized dense geometry"
PYTHONPATH="$PYTHONPATH_ROOT" \
python "$ROOT/HMR3D/nuc/scripts/export_lingbot_dense_geometry.py" \
  --predictions-npz "$JOB_OUTPUT_DIR/lingbot_predictions.npz" \
  --summary-json "$JOB_OUTPUT_DIR/lingbot_summary.json" \
  --image-root "$JOB_DIR" \
  --output-dir "$DENSE_DIR" \
  --stride "$DENSE_STRIDE" \
  --min-conf "$MIN_CONF"

echo "[4/4] Initializing Gaussian handle from dense geometry"
PYTHONPATH="$PYTHONPATH_ROOT" \
python "$ROOT/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py" \
  --dense-geometry-npz "$DENSE_DIR/lingbot_dense_geometry.npz" \
  --output-dir "$GAUSSIAN_OUTPUT_DIR" \
  --submap-id "$SUBMAP_ID" \
  --config "$CONFIG"

echo
echo "Done."
echo "Job dir: $JOB_DIR"
echo "Remote output: $JOB_OUTPUT_DIR"
echo "Dense geometry: $DENSE_DIR/lingbot_dense_geometry.npz"
echo "Gaussian output: $GAUSSIAN_OUTPUT_DIR"
