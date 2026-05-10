#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
BAG_PATH="${1:-/home/chatsign/fast_livo2_compressed.bag}"
CONTAINER_NAME="${2:-fastlivo-playback}"
RATE="${RATE:-1.0}"
START_OFFSET="${START_OFFSET:-0}"
LOOP="${LOOP:-0}"
START_PAUSED="${START_PAUSED:-0}"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "Bag file does not exist: $BAG_PATH" >&2
  exit 1
fi

BAG_ABS="$(realpath "$BAG_PATH")"
BAG_PARENT="$(dirname "$BAG_ABS")"

PLAY_ARGS=(--clock -r "$RATE")
if [[ "$START_OFFSET" != "0" ]]; then
  PLAY_ARGS+=(--start "$START_OFFSET")
fi
if [[ "$LOOP" == "1" ]]; then
  PLAY_ARGS+=(--loop)
fi
if [[ "$START_PAUSED" == "1" ]]; then
  PLAY_ARGS+=(--pause)
fi

PLAY_ARGS_STRING=""
for arg in "${PLAY_ARGS[@]}"; do
  PLAY_ARGS_STRING+=" $(printf '%q' "$arg")"
done

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Image:       $IMAGE"
echo "Bag:         $BAG_ABS"
echo "Container:   $CONTAINER_NAME"
echo "Rate:        $RATE"
echo "Start:       $START_OFFSET s"
echo "Loop:        $LOOP"
echo "Paused:      $START_PAUSED"

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  -v "$BAG_PARENT:$BAG_PARENT" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    roscore >/tmp/roscore.log 2>&1 & \
    ROSCORE_PID=\$!; \
    trap 'kill \$ROSCORE_PID >/dev/null 2>&1 || true' EXIT; \
    sleep 3; \
    rosparam set use_sim_time true; \
    rosbag play$PLAY_ARGS_STRING $(printf '%q' "$BAG_ABS")"

echo "Started $CONTAINER_NAME"
echo "Next:"
echo "  bash $ROOT_DIR/scripts/launch_rosbridge_sidecar.sh $CONTAINER_NAME 9090"
echo "  bash $ROOT_DIR/scripts/launch_web_ui_dev.sh 0.0.0.0 5173"
echo "  Open: http://localhost:5173/?mode=playback"
