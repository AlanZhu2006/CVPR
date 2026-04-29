#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map.pt}"
IMAGE_DIR="${IMAGE_DIR:-$LINGBOT_ROOT/example/church}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
MAX_FRAMES="${MAX_FRAMES:-8}"
MAX_WINDOWS="${MAX_WINDOWS:-7}"
WARMUP_WINDOWS="${WARMUP_WINDOWS:-1}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-2}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/nuc_output/lingbot_realtime_bench}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/orin_depth_only_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ ! -f "$SOURCE_MODEL_PATH" ]]; then
    echo "Missing source checkpoint: $SOURCE_MODEL_PATH" >&2
    exit 1
  fi
  echo "Creating depth-only fp16 checkpoint: $MODEL_PATH"
  "$PYTHON_BIN" HMR3D/nuc/scripts/create_lingbot_fp16_checkpoint.py \
    --input "$SOURCE_MODEL_PATH" \
    --output "$MODEL_PATH" \
    --drop-prefix camera_head \
    --drop-prefix point_head \
    --drop-prefix local_point_head \
    --fp16
fi

mkdir -p "$RUN_DIR"
echo "Run dir:          $RUN_DIR"
echo "Image dir:        $IMAGE_DIR"
echo "Image size:       $IMAGE_SIZE"
echo "Model image size: $MODEL_IMAGE_SIZE"
echo "Model:            $MODEL_PATH"

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=0
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=aggregator
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" HMR3D/nuc/scripts/benchmark_lingbot_realtime.py \
  --image-dir "$IMAGE_DIR" \
  --glob '*.png' \
  --model-path "$MODEL_PATH" \
  --lingbot-map-root "$LINGBOT_ROOT" \
  --output-json "$RUN_DIR/result.json" \
  --image-size "$IMAGE_SIZE" \
  --model-image-size "$MODEL_IMAGE_SIZE" \
  --window-size "$WINDOW_SIZE" \
  --stride 1 \
  --max-frames "$MAX_FRAMES" \
  --max-windows "$MAX_WINDOWS" \
  --warmup-windows "$WARMUP_WINDOWS" \
  --num-scale-frames "$NUM_SCALE_FRAMES" \
  --keyframe-interval 1 \
  --camera-num-iterations 1 \
  --depth-only \
  --disable-3d-rope \
  --print-each-window \
  > "$RUN_DIR/stdout.log" \
  2> "$RUN_DIR/stderr.log"

cat "$RUN_DIR/result.json"
