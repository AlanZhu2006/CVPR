#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
CACHE_DIR="${1:-$ROOT_DIR/runtime/playback-cache/fast_livo2_compressed}"
CONTAINER_NAME="${2:-fastlivo-cached-playback}"
CONTROL_PORT="${CONTROL_PORT:-8765}"
RATE="${RATE:-1.0}"
START_OFFSET="${START_OFFSET:-0}"
LOOP="${LOOP:-0}"
START_PAUSED="${START_PAUSED:-0}"
MJPEG_MAX_FPS="${MJPEG_MAX_FPS:-60}"
MAP_PUBLISH_EVERY="${MAP_PUBLISH_EVERY:-4}"
WEB_MAP_MAX_POINTS="${WEB_MAP_MAX_POINTS:-72000}"
WEB_SCAN_MAX_POINTS="${WEB_SCAN_MAX_POINTS:-12000}"
MAP_FUSE_MAX_POINTS="${MAP_FUSE_MAX_POINTS:-8000}"
SCAN_HISTORY_FRAMES="${SCAN_HISTORY_FRAMES:-10}"
RESTORE_REPLAY_MAX_FRAMES="${RESTORE_REPLAY_MAX_FRAMES:-8}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-60}"
PLAYBACK_VIDEO_PATH="${PLAYBACK_VIDEO_PATH:-${PLAYBACK_VIDEO:-}}"
PLAYBACK_VIDEO_FPS="${PLAYBACK_VIDEO_FPS:-12}"
PLAYBACK_VIDEO_SPEED="${PLAYBACK_VIDEO_SPEED:-1}"
PLAYBACK_VIDEO_START_OFFSET="${PLAYBACK_VIDEO_START_OFFSET:-0}"
PLAYBACK_VIDEO_END_OFFSET="${PLAYBACK_VIDEO_END_OFFSET:--1}"
PLAYBACK_VIDEO_LABEL="${PLAYBACK_VIDEO_LABEL:-}"

if [[ ! -f "$CACHE_DIR/meta.json" ]]; then
  echo "Playback cache metadata does not exist: $CACHE_DIR/meta.json" >&2
  exit 1
fi

abspath_preserve() {
  python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

CACHE_ABS="$(abspath_preserve "$CACHE_DIR")"
CACHE_PARENT="$(dirname "$CACHE_ABS")"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Image:         $IMAGE"
echo "Cache dir:     $CACHE_ABS"
echo "Container:     $CONTAINER_NAME"
echo "Control port:  $CONTROL_PORT"
echo "Rate:          $RATE"
echo "Start:         $START_OFFSET s"
echo "Loop:          $LOOP"
echo "Paused:        $START_PAUSED"
echo "MJPEG max fps: $MJPEG_MAX_FPS"
echo "Map publish:   every ${MAP_PUBLISH_EVERY} frame(s)"
echo "Web map pts:   $WEB_MAP_MAX_POINTS"
echo "Web scan pts:  $WEB_SCAN_MAX_POINTS"
echo "Map fuse pts:  $MAP_FUSE_MAX_POINTS (per frame, 0=no cap)"
echo "Scan history:  $SCAN_HISTORY_FRAMES frame(s)"
echo "Seek replay:   $RESTORE_REPLAY_MAX_FRAMES frame(s)"
echo "Startup wait:  ${STARTUP_TIMEOUT_SEC}s"
if [[ -n "$PLAYBACK_VIDEO_PATH" ]]; then
  echo "Video path:    $PLAYBACK_VIDEO_PATH"
fi

KEEP_CONTAINER_ON_EXIT="${KEEP_CONTAINER_ON_EXIT:-0}"
DOCKER_RM_FLAG="--rm"
if [[ "$KEEP_CONTAINER_ON_EXIT" == "1" ]]; then
  DOCKER_RM_FLAG=""
fi

docker run -d $DOCKER_RM_FLAG \
  --name "$CONTAINER_NAME" \
  --hostname "$CONTAINER_NAME" \
  -p "$CONTROL_PORT:$CONTROL_PORT" \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$CACHE_PARENT:$CACHE_PARENT" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    roscore >/tmp/roscore.log 2>&1 & \
    ROSCORE_PID=\$!; \
    trap 'kill \$ROSCORE_PID >/dev/null 2>&1 || true' EXIT; \
    for _ in \$(seq 1 60); do \
      if rosparam list >/dev/null 2>&1; then \
        break; \
      fi; \
      sleep 0.5; \
    done; \
    rosparam list >/dev/null 2>&1; \
    rosparam set use_sim_time true; \
    python3 '$ROOT_DIR/scripts/fastlivo_cached_playback_server.py' \
      --cache-dir '$CACHE_ABS' \
      --port '$CONTROL_PORT' \
      --default-rate '$RATE' \
      --start-offset '$START_OFFSET' \
      --container-name '$CONTAINER_NAME' \
      --image-name '$IMAGE' \
      --mjpeg-max-fps '$MJPEG_MAX_FPS' \
      --map-publish-every '$MAP_PUBLISH_EVERY' \
      --map-fuse-max-points '$MAP_FUSE_MAX_POINTS' \
      --scan-history-frames '$SCAN_HISTORY_FRAMES' \
      --restore-replay-max-frames '$RESTORE_REPLAY_MAX_FRAMES' \
      --web-map-max-points '$WEB_MAP_MAX_POINTS' \
      --web-scan-max-points '$WEB_SCAN_MAX_POINTS' \
      $([[ -n "$PLAYBACK_VIDEO_PATH" ]] && printf -- "--video-path '%s' --video-fps '%s' --video-speed '%s' --video-start-offset '%s' --video-end-offset '%s'" "$PLAYBACK_VIDEO_PATH" "$PLAYBACK_VIDEO_FPS" "$PLAYBACK_VIDEO_SPEED" "$PLAYBACK_VIDEO_START_OFFSET" "$PLAYBACK_VIDEO_END_OFFSET") \
      $([[ -n "$PLAYBACK_VIDEO_LABEL" ]] && printf -- " --video-label '%s'" "$PLAYBACK_VIDEO_LABEL") \
      $([[ "$LOOP" == "1" ]] && printf '%s' '--loop') \
      $([[ "$START_PAUSED" == "1" ]] && printf '%s' '--paused')"

deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
until curl -fs "http://127.0.0.1:$CONTROL_PORT/status" >/dev/null 2>&1; do
  if (( SECONDS >= deadline )); then
    break
  fi
  if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container $CONTAINER_NAME exited before control API became ready." >&2
    if docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
    else
      echo "Container was auto-removed before logs could be collected. Re-run with KEEP_CONTAINER_ON_EXIT=1 for debugging." >&2
    fi
    exit 1
  fi
  sleep 1
done

if ! curl -fs "http://127.0.0.1:$CONTROL_PORT/status" >/dev/null 2>&1; then
  echo "cached playback control API did not become ready on port $CONTROL_PORT" >&2
  echo "Container status:" >&2
  docker ps -a --filter "name=^${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' >&2 || true
  echo "Recent logs:" >&2
  if docker ps -a --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker logs --tail 200 "$CONTAINER_NAME" >&2 || true
  else
    echo "Container was auto-removed before logs could be collected. Re-run with KEEP_CONTAINER_ON_EXIT=1 for debugging." >&2
  fi
  exit 1
fi

echo "Started $CONTAINER_NAME"
echo "Control API: http://localhost:$CONTROL_PORT"
echo "Next:"
echo "  bash $ROOT_DIR/scripts/launch_rosbridge_sidecar.sh $CONTAINER_NAME 9090"
echo "  bash $ROOT_DIR/scripts/launch_web_ui_dev.sh 0.0.0.0 5173"
echo "  Open: http://localhost:5173/?mode=playback"
