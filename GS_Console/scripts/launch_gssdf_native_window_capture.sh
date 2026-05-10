#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISPLAY_VALUE="${1:-:88}"
PORT="${2:-8877}"
PID_FILE="${PID_FILE:-/tmp/gssdf-native-window-capture-${PORT}.pid}"
FPS="${FPS:-12}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
LOG_FILE="${LOG_FILE:-/tmp/gssdf-native-window-capture-${PORT}.log}"
CAPTURE_WIDTH="${CAPTURE_WIDTH:-1920}"
CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-1080}"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    kill "$OLD_PID" >/dev/null 2>&1 || true
    sleep 1
  fi
  rm -f "$PID_FILE"
fi

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -tiTCP:"$PORT" -sTCP:LISTEN | xargs -r kill >/dev/null 2>&1 || true
  sleep 1
fi

setsid -f bash -lc "exec python3 \"$ROOT_DIR/scripts/gssdf_native_window_capture_server.py\" \
  --display \"$DISPLAY_VALUE\" \
  --port \"$PORT\" \
  --fps \"$FPS\" \
  --width \"$CAPTURE_WIDTH\" \
  --height \"$CAPTURE_HEIGHT\" \
  --jpeg-quality \"$JPEG_QUALITY\" \
  --label \"GS-SDF Native Window\"" \
  >"$LOG_FILE" 2>&1 < /dev/null

sleep 1
CAPTURE_PID="$(pgrep -n -f "gssdf_native_window_capture_server.py --display ${DISPLAY_VALUE} --port ${PORT}" || true)"
if [[ -n "$CAPTURE_PID" ]]; then
  echo "$CAPTURE_PID" > "$PID_FILE"
fi

READY=0
for ((i=0; i<30; i++)); do
  if python3 - <<PY >/dev/null 2>&1
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:${PORT}/status", timeout=3) as response:
    payload = json.load(response)

if not payload.get("ready"):
    raise SystemExit(1)
PY
  then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "Native window capture server did not capture a first frame on port ${PORT}." >&2
  tail -n 40 "$LOG_FILE" >&2 || true
  exit 1
fi

echo "Native window capture ready on http://localhost:${PORT}"
echo "PID: ${CAPTURE_PID:-unknown}"
