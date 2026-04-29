#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt}"
SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map.pt}"
IMAGE_DIR="${IMAGE_DIR:-$LINGBOT_ROOT/example/church}"
IMAGE_GLOB="${IMAGE_GLOB:-*.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/nuc_output/lingbot_depth_worker/orin_$(date +%Y%m%d_%H%M%S)}"
IMAGE_SIZE="${IMAGE_SIZE:-336}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
MAX_FRAMES="${MAX_FRAMES:-0}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
STRIDE="${STRIDE:-1}"
NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-2}"
MAX_QUEUE="${MAX_QUEUE:-2}"
FRAME_SLEEP_SEC="${FRAME_SLEEP_SEC:-0}"
DEPTH_HEAD_TRT_ENGINE="${DEPTH_HEAD_TRT_ENGINE:-}"
MODEL_PATCH_EMBED="${MODEL_PATCH_EMBED:-}"
MODEL_EMBED_DIM="${MODEL_EMBED_DIM:-0}"
MODEL_DEPTH="${MODEL_DEPTH:-0}"
MODEL_NUM_HEADS="${MODEL_NUM_HEADS:-0}"
MODEL_MLP_RATIO="${MODEL_MLP_RATIO:-0}"
COMPRESS_OUTPUTS="${COMPRESS_OUTPUTS:-1}"

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

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=0
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=aggregator
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "LingBot depth worker"
echo "  output:          $OUTPUT_DIR"
echo "  model:           $MODEL_PATH"
echo "  image dir:       $IMAGE_DIR"
echo "  image size:      $IMAGE_SIZE"
echo "  window/stride:   $WINDOW_SIZE/$STRIDE"
echo "  max queue:       $MAX_QUEUE"
if [[ -n "$MODEL_PATCH_EMBED" || "$MODEL_EMBED_DIM" != "0" || "$MODEL_DEPTH" != "0" ]]; then
  echo "  direct model:    patch=$MODEL_PATCH_EMBED embed=$MODEL_EMBED_DIM depth=$MODEL_DEPTH heads=$MODEL_NUM_HEADS mlp=$MODEL_MLP_RATIO"
fi

extra_args=()
if [[ "$MAX_FRAMES" != "0" ]]; then
  extra_args+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$DEPTH_HEAD_TRT_ENGINE" ]]; then
  extra_args+=(--depth-head-trt-engine "$DEPTH_HEAD_TRT_ENGINE")
fi
if [[ -n "$MODEL_PATCH_EMBED" ]]; then
  extra_args+=(--model-patch-embed "$MODEL_PATCH_EMBED")
fi
if [[ "$MODEL_EMBED_DIM" != "0" ]]; then
  extra_args+=(--model-embed-dim "$MODEL_EMBED_DIM")
fi
if [[ "$MODEL_DEPTH" != "0" ]]; then
  extra_args+=(--model-depth "$MODEL_DEPTH")
fi
if [[ "$MODEL_NUM_HEADS" != "0" ]]; then
  extra_args+=(--model-num-heads "$MODEL_NUM_HEADS")
fi
if [[ "$MODEL_MLP_RATIO" != "0" ]]; then
  extra_args+=(--model-mlp-ratio "$MODEL_MLP_RATIO")
fi
if [[ "$COMPRESS_OUTPUTS" == "0" ]]; then
  extra_args+=(--no-compress-output)
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
  --max-queue "$MAX_QUEUE" \
  --frame-sleep-sec "$FRAME_SLEEP_SEC" \
  "${extra_args[@]}" \
  --submit-blocking
