#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/nyu/Codespace/CVPR"
VENV_SCRIPT="$ROOT/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh"
LINGBOT_DIR="$ROOT/third_party_research/lingbot-map"
CACHE_DIR="$ROOT/third_party_research/lingbot_cache"
MODEL_PATH="$CACHE_DIR/lingbot-map.pt"
IMAGE_DIR="$LINGBOT_DIR/example/church"

FIRST_K="${FIRST_K:-2}"
KEYFRAME_INTERVAL="${KEYFRAME_INTERVAL:-2}"
CAMERA_ITERS="${CAMERA_ITERS:-1}"
LINGBOT_FORCE_CPU="${LINGBOT_FORCE_CPU:-1}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model checkpoint: $MODEL_PATH" >&2
  exit 1
fi

source "$VENV_SCRIPT"
cd "$LINGBOT_DIR"

ARGS=(
  demo.py
  --model_path "$MODEL_PATH"
  --image_folder "$IMAGE_DIR"
  --first_k "$FIRST_K"
  --mode streaming
  --keyframe_interval "$KEYFRAME_INTERVAL"
  --camera_num_iterations "$CAMERA_ITERS"
  --use_sdpa
  --offload_to_cpu
)

if [[ "$LINGBOT_FORCE_CPU" == "1" ]]; then
  CUDA_VISIBLE_DEVICES='' python "${ARGS[@]}"
else
  python "${ARGS[@]}"
fi
