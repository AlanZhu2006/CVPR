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
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
KEYFRAME_INTERVAL="${KEYFRAME_INTERVAL:-2}"
CAMERA_ITERS="${CAMERA_ITERS:-1}"
CONF_THRESHOLD="${CONF_THRESHOLD:-1.5}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-10}"
POINT_SIZE="${POINT_SIZE:-0.00001}"
MASK_SKY="${MASK_SKY:-1}"
SMOKE_MODE="${SMOKE_MODE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/nuc_output/lingbot_16gb_smoke/$(date +%Y%m%d_%H%M%S)}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-1}"
DISABLE_CAMERA="${DISABLE_CAMERA:-0}"
DISABLE_POINT="${DISABLE_POINT:-0}"

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
echo "  model_size:   $MODEL_IMAGE_SIZE"
echo "  keyframes:    $KEYFRAME_INTERVAL"
echo "  offload_cpu:  $OFFLOAD_TO_CPU"
echo "  disable_cam:  $DISABLE_CAMERA"
echo "  disable_point:$DISABLE_POINT"
echo "  port:         $PORT"
echo
echo "Open after startup:"
echo "  http://127.0.0.1:$PORT"

run_mode="$SMOKE_MODE"
if [[ "$run_mode" == "auto" ]]; then
  if [[ "$IMAGE_SIZE" == "$MODEL_IMAGE_SIZE" ]]; then
    run_mode="upstream"
  else
    run_mode="local_compat"
  fi
fi

if [[ "$run_mode" == "upstream" ]]; then
  upstream_args=()
  if [[ "$OFFLOAD_TO_CPU" == "1" ]]; then
    upstream_args+=(--offload_to_cpu)
  fi
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
    --port "$PORT" \
    --conf_threshold "$CONF_THRESHOLD" \
    --downsample_factor "$DOWNSAMPLE_FACTOR" \
    --point_size "$POINT_SIZE" \
    "${upstream_args[@]}" \
    "${extra_args[@]}"
  exit 0
fi

if [[ "$run_mode" != "local_compat" ]]; then
  echo "Unknown SMOKE_MODE: $run_mode" >&2
  echo "Use one of: auto, upstream, local_compat" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo
echo "Using HMR3D local compatibility path"
echo "  output:       $OUTPUT_DIR"

viewer_args=()
if [[ "$MASK_SKY" == "1" ]]; then
  viewer_args+=(--mask-sky)
fi
export_args=()
if [[ "$OFFLOAD_TO_CPU" == "0" ]]; then
  export_args+=(--no-offload-to-cpu)
fi
if [[ "$DISABLE_CAMERA" == "1" ]]; then
  export_args+=(--disable-camera)
fi
if [[ "$DISABLE_POINT" == "1" ]]; then
  export_args+=(--disable-point)
fi

"$PYTHON_BIN" HMR3D/nuc/scripts/run_lingbot_export.py \
  --model-path "$MODEL_PATH" \
  --lingbot-map-root "$LINGBOT_ROOT" \
  --image-folder "$IMAGE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --first-k "$FIRST_K" \
  --image-size "$IMAGE_SIZE" \
  --model-image-size "$MODEL_IMAGE_SIZE" \
  --mode streaming \
  --num-scale-frames 2 \
  --keyframe-interval "$KEYFRAME_INTERVAL" \
  --camera-num-iterations "$CAMERA_ITERS" \
  "${export_args[@]}"

"$PYTHON_BIN" HMR3D/nuc/scripts/view_lingbot_predictions_viser.py \
  --predictions-npz "$OUTPUT_DIR/lingbot_predictions.npz" \
  --summary-json "$OUTPUT_DIR/lingbot_summary.json" \
  --port "$PORT" \
  --image-size "$IMAGE_SIZE" \
  --downsample-factor "$DOWNSAMPLE_FACTOR" \
  --point-size "$POINT_SIZE" \
  --init-conf-threshold "$CONF_THRESHOLD" \
  "${viewer_args[@]}"
