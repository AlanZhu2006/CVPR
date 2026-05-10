#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml}"
VIEW_CONTAINER="${2:-gssdf-view-quality}"
BRIDGE_PORT="${3:-8876}"
CAPTURE_PORT="${4:-8877}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:88}"
XVFB_GEOMETRY="${XVFB_GEOMETRY:-1920x1080x24}"
WAIT_SECONDS="${WAIT_SECONDS:-90}"
XVFB_PID_FILE="${XVFB_PID_FILE:-/tmp/gssdf-native-window-xvfb-${DISPLAY_VALUE#:}.pid}"
XVFB_LOG_FILE="${XVFB_LOG_FILE:-/tmp/gssdf-native-window-xvfb-${DISPLAY_VALUE#:}.log}"
X11_SOCKET_PATH="/tmp/.X11-unix/X${DISPLAY_VALUE#:}"
X11_LOCK_PATH="/tmp/.X${DISPLAY_VALUE#:}-lock"

if ! xdpyinfo -display "$DISPLAY_VALUE" >/dev/null 2>&1; then
  rm -f "$X11_SOCKET_PATH" "$X11_LOCK_PATH"
  echo "Starting Xvfb on $DISPLAY_VALUE with geometry $XVFB_GEOMETRY"
  setsid -f bash -lc "exec Xvfb '$DISPLAY_VALUE' -screen 0 '$XVFB_GEOMETRY' -ac +extension GLX +render -noreset" \
    >"$XVFB_LOG_FILE" 2>&1 < /dev/null
  for ((i=0; i<15; i++)); do
    if xdpyinfo -display "$DISPLAY_VALUE" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  XVFB_PID="$(pgrep -n -f "Xvfb ${DISPLAY_VALUE} -screen 0 ${XVFB_GEOMETRY}" || true)"
  if [[ -n "$XVFB_PID" ]]; then
    echo "$XVFB_PID" > "$XVFB_PID_FILE"
  fi
fi

if ! xdpyinfo -display "$DISPLAY_VALUE" >/dev/null 2>&1; then
  echo "Xvfb on $DISPLAY_VALUE did not become ready." >&2
  tail -n 80 "$XVFB_LOG_FILE" >&2 || true
  exit 1
fi

bash "$ROOT_DIR/scripts/launch_gssdf_view_x11.sh" "$OUTPUT_DIR" "$VIEW_CONTAINER"

READY=0
for ((i=0; i<WAIT_SECONDS; i++)); do
  if docker exec "$VIEW_CONTAINER" bash -lc "source /opt/ros/noetic/setup.bash && rostopic list | grep -q '^/neural_mapping/path$'" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "Neural mapping topics in $VIEW_CONTAINER did not become ready within ${WAIT_SECONDS}s" >&2
  exit 1
fi

bash "$ROOT_DIR/scripts/launch_gssdf_hifi_bridge.sh" "$VIEW_CONTAINER" "$BRIDGE_PORT" "gssdf-hifi-bridge"

CAPTURE_SIZE="${XVFB_GEOMETRY%%x24}"
CAPTURE_WIDTH="${CAPTURE_SIZE%%x*}"
CAPTURE_HEIGHT="${CAPTURE_SIZE#*x}"
CAPTURE_WIDTH="$CAPTURE_WIDTH" CAPTURE_HEIGHT="$CAPTURE_HEIGHT" \
  bash "$ROOT_DIR/scripts/launch_gssdf_native_window_capture.sh" "$DISPLAY_VALUE" "$CAPTURE_PORT"

echo "Native window HiFi stack ready."
echo "Pose bridge:   http://localhost:${BRIDGE_PORT}"
echo "Window stream: http://localhost:${CAPTURE_PORT}"
