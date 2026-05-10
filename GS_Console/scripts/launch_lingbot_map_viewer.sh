#!/usr/bin/env bash
set -euo pipefail

LINGBOT_MAP_REPO="${LINGBOT_MAP_REPO:-/home/chatsign/work/lingbot-map}"
DEFAULT_MODEL_PATH="/media/chatsign/data-002/models/lingbot-map/lingbot-map-long.pt"
MODEL_PATH="${LINGBOT_MODEL_PATH:-$DEFAULT_MODEL_PATH}"
PYTHON_BIN="${LINGBOT_PYTHON_BIN:-}"
IMAGE_FOLDER="${LINGBOT_IMAGE_FOLDER:-$LINGBOT_MAP_REPO/example/church}"
VIDEO_PATH="${LINGBOT_VIDEO_PATH:-}"
PORT="${LINGBOT_PORT:-8080}"
MODE="${LINGBOT_MODE:-streaming}"
FPS="${LINGBOT_FPS:-10}"
FIRST_K="${LINGBOT_FIRST_K:-}"
STRIDE="${LINGBOT_STRIDE:-1}"
KEYFRAME_INTERVAL="${LINGBOT_KEYFRAME_INTERVAL:-}"
CAMERA_NUM_ITERATIONS="${LINGBOT_CAMERA_NUM_ITERATIONS:-4}"
CONF_THRESHOLD="${LINGBOT_CONF_THRESHOLD:-1.5}"
DOWNSAMPLE_FACTOR="${LINGBOT_DOWNSAMPLE_FACTOR:-10}"
POINT_SIZE="${LINGBOT_POINT_SIZE:-0.00001}"
MASK_SKY="${LINGBOT_MASK_SKY:-1}"
USE_SDPA="${LINGBOT_USE_SDPA:-1}"
EXTRA_ARGS="${LINGBOT_EXTRA_ARGS:-}"

if [[ ! -f "$LINGBOT_MAP_REPO/demo.py" ]]; then
  echo "LingBot-Map repo not found: $LINGBOT_MAP_REPO" >&2
  echo "Clone it with: git clone https://github.com/Robbyant/lingbot-map.git $LINGBOT_MAP_REPO" >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "LINGBOT_MODEL_PATH is required, for example /path/to/lingbot-map-long.pt" >&2
  echo "Model repository: https://huggingface.co/robbyant/lingbot-map" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "LingBot-Map checkpoint does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "No Python interpreter found. Set LINGBOT_PYTHON_BIN=/path/to/python." >&2
    exit 1
  fi
fi

args=(
  demo.py
  --model_path "$MODEL_PATH"
  --mode "$MODE"
  --fps "$FPS"
  --stride "$STRIDE"
  --camera_num_iterations "$CAMERA_NUM_ITERATIONS"
  --port "$PORT"
  --conf_threshold "$CONF_THRESHOLD"
  --downsample_factor "$DOWNSAMPLE_FACTOR"
  --point_size "$POINT_SIZE"
)

if [[ -n "$VIDEO_PATH" ]]; then
  if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "LINGBOT_VIDEO_PATH does not exist: $VIDEO_PATH" >&2
    exit 1
  fi
  args+=(--video_path "$VIDEO_PATH")
else
  if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "LINGBOT_IMAGE_FOLDER does not exist: $IMAGE_FOLDER" >&2
    exit 1
  fi
  args+=(--image_folder "$IMAGE_FOLDER")
fi

if [[ -n "$FIRST_K" ]]; then
  args+=(--first_k "$FIRST_K")
fi

if [[ -n "$KEYFRAME_INTERVAL" ]]; then
  args+=(--keyframe_interval "$KEYFRAME_INTERVAL")
fi

if [[ "$MASK_SKY" == "1" ]]; then
  args+=(--mask_sky)
fi

if [[ "$USE_SDPA" == "1" ]]; then
  args+=(--use_sdpa)
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  extra_args_array=($EXTRA_ARGS)
  args+=("${extra_args_array[@]}")
fi

echo "Starting LingBot-Map viser viewer on http://localhost:$PORT"
echo "Open GS-SDF frontend scene: http://localhost:5173/?scene=/scenes/lingbot-map-viewer/manifest.json&mode=gs"

cd "$LINGBOT_MAP_REPO"
exec "$PYTHON_BIN" "${args[@]}"
