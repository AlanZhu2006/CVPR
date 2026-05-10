#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_VIDEO="${1:-$ROOT_DIR/runtime/videos/cbd_gs_hifi_orbit.mp4}"
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8876}"
CAPTURE_URL="${CAPTURE_URL:-$BRIDGE_URL/frame.mjpeg}"
CAPTURE_FPS="${CAPTURE_FPS:-16}"
DURATION_SEC="${DURATION_SEC:-14}"
PRE_ROLL_SEC="${PRE_ROLL_SEC:-1.5}"

mkdir -p "$(dirname "$OUTPUT_VIDEO")"

echo "Output video:  $OUTPUT_VIDEO"
echo "Bridge URL:    $BRIDGE_URL"
echo "Capture URL:   $CAPTURE_URL"
echo "Duration:      $DURATION_SEC s"
echo "Capture FPS:   $CAPTURE_FPS"

python3 "$ROOT_DIR/scripts/render_gssdf_hifi_camera_path.py" \
  --bridge-url "$BRIDGE_URL" \
  --duration "$DURATION_SEC" \
  --fps "$CAPTURE_FPS" \
  >/tmp/gssdf_hifi_path.log 2>&1 &
PATH_PID=$!

sleep "$PRE_ROLL_SEC"

ffmpeg -y \
  -fflags nobuffer \
  -flags low_delay \
  -use_wallclock_as_timestamps 1 \
  -i "$CAPTURE_URL" \
  -t "$DURATION_SEC" \
  -an \
  -c:v libx264 \
  -preset veryfast \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$OUTPUT_VIDEO"

wait "$PATH_PID" || true
echo "Saved $OUTPUT_VIDEO"
