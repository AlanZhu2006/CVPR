#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
IMAGE_DIR="${IMAGE_DIR:-$LINGBOT_ROOT/example/church}"
SHIM_DIR="${SHIM_DIR:-$ROOT_DIR/nuc_output/lingbot_church_cuvslam_input}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/nuc_output/lingbot_student_distill/kitti0020_cached_teacher_depth8_300step/lingbot_depth_student.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/nuc_output/lingbot_church_cuvslam_student/church_student_$(date +%Y%m%d_%H%M%S)}"
MAX_FRAMES="${MAX_FRAMES:-80}"
PORT="${PORT:-19116}"
VISER_PORT="${VISER_PORT:-19117}"

if [[ ! -f "$MODEL_PATH" ]]; then
  cat >&2 <<EOF
Missing depth student checkpoint:
  $MODEL_PATH

This demo is the low-memory cuVSLAM-pose + LingBot-depth-student path.
Use HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh for the full LingBot
teacher smoke test, or set MODEL_PATH=/path/to/lingbot_depth_student.pt.
EOF
  exit 1
fi

LINGBOT_ROOT="$(cd "$LINGBOT_ROOT" && pwd)"
IMAGE_DIR="$(cd "$IMAGE_DIR" && pwd)"
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
SHIM_DIR="$(mkdir -p "$SHIM_DIR" && cd "$SHIM_DIR" && pwd)"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

"$PYTHON_BIN" HMR3D/nuc/scripts/prepare_lingbot_church_cuvslam_input.py \
  --image-dir "$IMAGE_DIR" \
  --output-dir "$SHIM_DIR" \
  --max-frames "$MAX_FRAMES"

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_MAP_ROOT="$LINGBOT_ROOT"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU="${LINGBOT_LOAD_CHECKPOINT_ON_CPU:-1}"
export LINGBOT_CHECKPOINT_MMAP="${LINGBOT_CHECKPOINT_MMAP:-0}"
export LINGBOT_MODEL_DTYPE="${LINGBOT_MODEL_DTYPE:-fp16}"
export LINGBOT_CPU_CAST_BEFORE_CUDA="${LINGBOT_CPU_CAST_BEFORE_CUDA:-1}"
export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-aggregator}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py \
  --sequence-dir "$SHIM_DIR" \
  --trajectory-path "$SHIM_DIR/unused_tum.txt" \
  --tracking-backend cuvslam_mono_rgb \
  --rgb-image-dir "$IMAGE_DIR" \
  --color-image-dir "$IMAGE_DIR" \
  --color-image-template "{frame_idx:06d}.png" \
  --model-path "$MODEL_PATH" \
  --lingbot-map-root "$LINGBOT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --image-size 224 \
  --model-image-size 224 \
  --model-patch-embed conv \
  --model-embed-dim 384 \
  --model-depth 8 \
  --model-num-heads 6 \
  --model-mlp-ratio 3.0 \
  --frame-step 1 \
  --max-frames "$MAX_FRAMES" \
  --window-size 2 \
  --stride 1 \
  --dense-frame-interval 1 \
  --max-queue 2 \
  --depth-scale 8.0 \
  --min-depth 0.05 \
  --max-depth 35.0 \
  --min-conf 0.2 \
  --sample-stride 2 \
  --max-points-per-frame 12000 \
  --max-active-frames 120 \
  --fusion-mode voxel \
  --voxel-size 0.04 \
  --fusion-max-points 900000 \
  --fusion-min-observations 1 \
  --adaptive-sampling \
  --near-depth-m 16.0 \
  --near-sample-stride 1 \
  --edge-sample-stride 1 \
  --edge-percentile 82 \
  --publish-every-windows 4 \
  --publish-every-frames 8 \
  --no-compress-output \
  --serve \
  --port "$PORT"

echo
echo "Starting Viser viewer for the fused map..."
echo "Open: http://127.0.0.1:$VISER_PORT"
"$PYTHON_BIN" HMR3D/nuc/scripts/launch_lingbot_live_viser.py \
  --map-dir "$OUTPUT_DIR" \
  --host 0.0.0.0 \
  --port "$VISER_PORT" \
  --point-size 0.026 \
  --max-points 450000 \
  --quantile-clip 0.0005 \
  --color-mode original \
  --initial-mode reveal \
  --fps 6
