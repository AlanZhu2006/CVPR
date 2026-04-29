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
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map.pt}"
PORT="${PORT:-19115}"
FIRST_K="${FIRST_K:-40}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
KEYFRAME_INTERVAL="${KEYFRAME_INTERVAL:-2}"
CAMERA_ITERS="${CAMERA_ITERS:-1}"
CONF_THRESHOLD="${CONF_THRESHOLD:-1.5}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-10}"
POINT_SIZE="${POINT_SIZE:-0.00001}"
MASK_SKY="${MASK_SKY:-1}"

if [[ ! -d "$LINGBOT_ROOT" ]]; then
  echo "Missing LingBot repo: $LINGBOT_ROOT" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Missing image dir: $IMAGE_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  cat >&2 <<EOF
Missing full LingBot checkpoint:
  $MODEL_PATH

Set MODEL_PATH=/path/to/lingbot-map.pt, or place the model at:
  third_party_research/lingbot_cache/lingbot-map.pt
EOF
  exit 1
fi

LINGBOT_ROOT="$(cd "$LINGBOT_ROOT" && pwd)"
IMAGE_DIR="$(cd "$IMAGE_DIR" && pwd)"
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"

export PYTHONPATH="$LINGBOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU="${LINGBOT_LOAD_CHECKPOINT_ON_CPU:-1}"
export LINGBOT_CHECKPOINT_MMAP="${LINGBOT_CHECKPOINT_MMAP:-1}"
export LINGBOT_MODEL_DTYPE="${LINGBOT_MODEL_DTYPE:-fp16}"
export LINGBOT_CPU_CAST_BEFORE_CUDA="${LINGBOT_CPU_CAST_BEFORE_CUDA:-1}"
export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-model}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

extra_args=()
if [[ "$MASK_SKY" == "1" ]]; then
  extra_args+=(--mask_sky)
fi

echo "LingBot church 16GB smoke"
echo "  python:       $PYTHON_BIN"
echo "  lingbot root: $LINGBOT_ROOT"
echo "  images:       $IMAGE_DIR"
echo "  model:        $MODEL_PATH"
echo "  first_k:      $FIRST_K"
echo "  image_size:   $IMAGE_SIZE"
echo "  keyframes:    $KEYFRAME_INTERVAL"
echo "  port:         $PORT"
echo
echo "Open after startup:"
echo "  http://127.0.0.1:$PORT"

cd "$LINGBOT_ROOT"
"$PYTHON_BIN" demo.py \
  --model_path "$MODEL_PATH" \
  --image_folder "$IMAGE_DIR" \
  --first_k "$FIRST_K" \
  --image_size "$IMAGE_SIZE" \
  --mode streaming \
  --keyframe_interval "$KEYFRAME_INTERVAL" \
  --camera_num_iterations "$CAMERA_ITERS" \
  --use_sdpa \
  --offload_to_cpu \
  --port "$PORT" \
  --conf_threshold "$CONF_THRESHOLD" \
  --downsample_factor "$DOWNSAMPLE_FACTOR" \
  --point_size "$POINT_SIZE" \
  "${extra_args[@]}"
