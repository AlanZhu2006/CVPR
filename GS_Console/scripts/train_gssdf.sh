#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
INPUT_PATH="${1:-}"
CONFIG_PATH="${2:-$ROOT_DIR/config/fastlivo_campus_host.yaml}"
RUN_NAME="${3:-gs-sdf-train-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: $0 <bag-or-color_poses.txt> [config.yaml] [run-name]" >&2
  exit 1
fi

if [[ ! -e "$INPUT_PATH" ]]; then
  echo "Input path does not exist: $INPUT_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

INPUT_ABS="$(realpath "$INPUT_PATH")"
CONFIG_ABS="$(realpath "$CONFIG_PATH")"
INPUT_DIR="$(dirname "$INPUT_ABS")"
OUTPUT_DIR="$ROOT_DIR/runtime/output"
LOG_DIR="$ROOT_DIR/runtime/logs"
EXTRA_MOUNT_ARGS=()
if [[ -d "/media/chatsign/data-002/datasets" ]]; then
  EXTRA_MOUNT_ARGS+=(-v "/media/chatsign/data-002/datasets:/media/chatsign/data-002/datasets:ro")
fi
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "Image:      $IMAGE"
echo "Input:      $INPUT_ABS"
echo "Config:     $CONFIG_ABS"
echo "Output dir: $OUTPUT_DIR"
echo "Log file:   $LOG_FILE"
echo "Run name:   $RUN_NAME"

if [[ -f "$ROOT_DIR/scripts/prepare_gssdf_colmap_compat.py" ]]; then
  python3 "$ROOT_DIR/scripts/prepare_gssdf_colmap_compat.py" --dataset "$INPUT_ABS"
fi

docker run --rm --name "$RUN_NAME" \
  --gpus all \
  --shm-size=16g \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$INPUT_DIR:$INPUT_DIR" \
  "${EXTRA_MOUNT_ARGS[@]}" \
  -v "$OUTPUT_DIR:/root/gs_sdf_ws/src/GS-SDF/output" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    source /root/gs_sdf_ws/devel/setup.bash; \
    roscore >/tmp/roscore.log 2>&1 & \
    ROSCORE_PID=\$!; \
    trap 'kill \$ROSCORE_PID >/dev/null 2>&1 || true' EXIT; \
    sleep 3; \
    /root/gs_sdf_ws/devel/lib/neural_mapping/neural_mapping_node train \"$CONFIG_ABS\" \"$INPUT_ABS\"" \
  2>&1 | tee "$LOG_FILE"
