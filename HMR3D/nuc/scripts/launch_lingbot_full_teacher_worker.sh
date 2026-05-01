#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map.pt}"
IMAGE_DIR="${IMAGE_DIR:-$LINGBOT_ROOT/example/church}"
IMAGE_GLOB="${IMAGE_GLOB:-*.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/nuc_output/lingbot_full_teacher_worker/orin_$(date +%Y%m%d_%H%M%S)}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
MAX_FRAMES="${MAX_FRAMES:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
STRIDE="${STRIDE:-1}"
NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-2}"
KEYFRAME_INTERVAL="${KEYFRAME_INTERVAL:-2}"
MAX_QUEUE="${MAX_QUEUE:-2}"
FRAME_SLEEP_SEC="${FRAME_SLEEP_SEC:-0}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-0}"
ENABLE_POINT="${ENABLE_POINT:-0}"
ENABLE_3D_ROPE="${ENABLE_3D_ROPE:-1}"
COMPRESS_OUTPUTS="${COMPRESS_OUTPUTS:-0}"
SUBMIT_BLOCKING="${SUBMIT_BLOCKING:-1}"
PRELOAD_MODEL="${PRELOAD_MODEL:-1}"
WARMUP_FIRST_WINDOW="${WARMUP_FIRST_WINDOW:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing full LingBot checkpoint: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "Missing image dir: $IMAGE_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_MAP_ROOT="$LINGBOT_ROOT"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU="${LINGBOT_LOAD_CHECKPOINT_ON_CPU:-1}"
export LINGBOT_CHECKPOINT_MMAP="${LINGBOT_CHECKPOINT_MMAP:-1}"
export LINGBOT_MODEL_DTYPE="${LINGBOT_MODEL_DTYPE:-bf16}"
export LINGBOT_CPU_CAST_BEFORE_CUDA="${LINGBOT_CPU_CAST_BEFORE_CUDA:-1}"
export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-model}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "LingBot full teacher persistent worker"
echo "  output:          $OUTPUT_DIR"
echo "  model:           $MODEL_PATH"
echo "  image dir:       $IMAGE_DIR"
echo "  image size:      $IMAGE_SIZE"
echo "  model size:      $MODEL_IMAGE_SIZE"
echo "  window/stride:   $WINDOW_SIZE/$STRIDE"
echo "  scale/keyframes: $NUM_SCALE_FRAMES/$KEYFRAME_INTERVAL"
echo "  offload cpu:     $OFFLOAD_TO_CPU"
echo "  enable point:    $ENABLE_POINT"
echo "  preload model:   $PRELOAD_MODEL"
echo "  warmup window:   $WARMUP_FIRST_WINDOW"
echo "  max queue:       $MAX_QUEUE"

extra_args=(--enable-camera)
if [[ "$ENABLE_POINT" == "1" ]]; then
  extra_args+=(--enable-point)
fi
if [[ "$ENABLE_3D_ROPE" == "1" ]]; then
  extra_args+=(--enable-3d-rope)
fi
if [[ "$OFFLOAD_TO_CPU" == "0" ]]; then
  extra_args+=(--no-offload-to-cpu)
fi
if [[ "$COMPRESS_OUTPUTS" == "0" ]]; then
  extra_args+=(--no-compress-output)
fi
if [[ "$SUBMIT_BLOCKING" == "1" ]]; then
  extra_args+=(--submit-blocking)
fi
if [[ "$PRELOAD_MODEL" == "1" ]]; then
  extra_args+=(--preload-model)
fi
if [[ "$WARMUP_FIRST_WINDOW" == "1" ]]; then
  extra_args+=(--warmup-first-window)
fi
if [[ "$MAX_FRAMES" != "0" ]]; then
  extra_args+=(--max-frames "$MAX_FRAMES")
fi

"$PYTHON_BIN" HMR3D/nuc/scripts/run_lingbot_depth_worker.py \
  --image-dir "$IMAGE_DIR" \
  --glob "$IMAGE_GLOB" \
  --model-path "$MODEL_PATH" \
  --lingbot-map-root "$LINGBOT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --model-image-size "$MODEL_IMAGE_SIZE" \
  --window-size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --num-scale-frames "$NUM_SCALE_FRAMES" \
  --keyframe-interval "$KEYFRAME_INTERVAL" \
  --max-queue "$MAX_QUEUE" \
  --frame-sleep-sec "$FRAME_SLEEP_SEC" \
  "${extra_args[@]}"
