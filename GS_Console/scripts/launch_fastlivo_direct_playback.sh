#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
BAG_PATH="${1:-/home/chatsign/fast_livo2_compressed.bag}"
CONTAINER_NAME="${2:-fastlivo-direct-playback}"
CONTROL_PORT="${CONTROL_PORT:-8765}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/fastlivo_cbd_host.yaml}"
RATE="${RATE:-1.0}"
START_OFFSET="${START_OFFSET:-0}"
LOOP="${LOOP:-0}"
START_PAUSED="${START_PAUSED:-0}"
WEB_MAX_POINTS="${WEB_MAX_POINTS:-2200}"
WEB_VOXEL_SIZE="${WEB_VOXEL_SIZE:-0.16}"
WEB_MIN_RANGE="${WEB_MIN_RANGE:-1.0}"
WEB_MAX_RANGE="${WEB_MAX_RANGE:-24.0}"
WEB_MAX_ABS_Z="${WEB_MAX_ABS_Z:-8.0}"
IMAGE_SYNC_TOLERANCE="${IMAGE_SYNC_TOLERANCE:-0.2}"
POSE_SYNC_TOLERANCE="${POSE_SYNC_TOLERANCE:-0.12}"
SCAN_MAX_POINTS="${SCAN_MAX_POINTS:-3600}"
SCAN_VOXEL_SIZE="${SCAN_VOXEL_SIZE:-0.12}"
MAP_MAX_POINTS="${MAP_MAX_POINTS:-90000}"
MAP_VOXEL_SIZE="${MAP_VOXEL_SIZE:-0.20}"
MAP_PUBLISH_EVERY="${MAP_PUBLISH_EVERY:-4}"
MJPEG_MAX_FPS="${MJPEG_MAX_FPS:-60}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-60}"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "Bag file does not exist: $BAG_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

abspath_preserve() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

BAG_ABS="$(abspath_preserve "$BAG_PATH")"
BAG_PARENT="$(dirname "$BAG_ABS")"
CONFIG_ABS="$(abspath_preserve "$CONFIG_PATH")"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Image:         $IMAGE"
echo "Bag:           $BAG_ABS"
echo "Config:        $CONFIG_ABS"
echo "Container:     $CONTAINER_NAME"
echo "Control port:  $CONTROL_PORT"
echo "Rate:          $RATE"
echo "Start:         $START_OFFSET s"
echo "Loop:          $LOOP"
echo "Paused:        $START_PAUSED"
echo "Web max pts:   $WEB_MAX_POINTS"
echo "Web voxel:     $WEB_VOXEL_SIZE"
echo "Web range:     $WEB_MIN_RANGE - $WEB_MAX_RANGE"
echo "Scan max pts:  $SCAN_MAX_POINTS"
echo "Scan voxel:    $SCAN_VOXEL_SIZE"
echo "Map max pts:   $MAP_MAX_POINTS"
echo "Map voxel:     $MAP_VOXEL_SIZE"
echo "MJPEG max fps: $MJPEG_MAX_FPS"
echo "Startup wait:  ${STARTUP_TIMEOUT_SEC}s"

docker run -d --rm \
  --name "$CONTAINER_NAME" \
  -p "$CONTROL_PORT:$CONTROL_PORT" \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$BAG_PARENT:$BAG_PARENT" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    roscore >/tmp/roscore.log 2>&1 & \
    ROSCORE_PID=\$!; \
    trap 'kill \$ROSCORE_PID >/dev/null 2>&1 || true' EXIT; \
    sleep 3; \
    rosparam set use_sim_time true; \
    python3 '$ROOT_DIR/scripts/fastlivo_direct_playback_server.py' \
      --bag-path '$BAG_ABS' \
      --config '$CONFIG_ABS' \
      --port '$CONTROL_PORT' \
      --default-rate '$RATE' \
      --start-offset '$START_OFFSET' \
      --container-name '$CONTAINER_NAME' \
      --image-name '$IMAGE' \
      --web-max-points '$WEB_MAX_POINTS' \
      --web-voxel-size '$WEB_VOXEL_SIZE' \
      --web-min-range '$WEB_MIN_RANGE' \
      --web-max-range '$WEB_MAX_RANGE' \
      --web-max-abs-z '$WEB_MAX_ABS_Z' \
      --image-sync-tolerance '$IMAGE_SYNC_TOLERANCE' \
      --pose-sync-tolerance '$POSE_SYNC_TOLERANCE' \
      --scan-max-points '$SCAN_MAX_POINTS' \
      --scan-voxel-size '$SCAN_VOXEL_SIZE' \
      --map-max-points '$MAP_MAX_POINTS' \
      --map-voxel-size '$MAP_VOXEL_SIZE' \
      --map-publish-every '$MAP_PUBLISH_EVERY' \
      --mjpeg-max-fps '$MJPEG_MAX_FPS' \
      $([[ "$LOOP" == "1" ]] && printf '%s' '--loop') \
      $([[ "$START_PAUSED" == "1" ]] && printf '%s' '--paused')"

deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
until curl -fs "http://127.0.0.1:$CONTROL_PORT/status" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    break
  fi
  if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container $CONTAINER_NAME exited before control API became ready." >&2
    docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 1
done

if ! curl -fs "http://127.0.0.1:$CONTROL_PORT/status" >/dev/null 2>&1; then
  echo "direct playback control API did not become ready on port $CONTROL_PORT" >&2
  echo "Container status:" >&2
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' >&2 || true
  echo "Recent logs:" >&2
  docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
  exit 1
fi

echo "Started $CONTAINER_NAME"
echo "Control API: http://localhost:$CONTROL_PORT"
echo "Next:"
echo "  bash $ROOT_DIR/scripts/launch_rosbridge_sidecar.sh $CONTAINER_NAME 9090"
echo "  bash $ROOT_DIR/scripts/launch_web_ui_dev.sh 0.0.0.0 5173"
echo "  Open: http://localhost:5173/?mode=playback"
