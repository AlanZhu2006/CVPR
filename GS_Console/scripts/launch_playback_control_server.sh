#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_PATH="${1:-/home/chatsign/fast_livo2_compressed.bag}"
CONTAINER_NAME="${2:-fastlivo-playback}"
PORT="${3:-8765}"
PID_FILE="/tmp/gs_sdf_playback_control_${CONTAINER_NAME}.pid"
LOG_FILE="/tmp/gs_sdf_playback_control_${CONTAINER_NAME}.log"
LOOP_FLAG="${LOOP:-1}"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" >/dev/null 2>&1; then
    kill "$OLD_PID" >/dev/null 2>&1 || true
    sleep 0.2
  fi
  rm -f "$PID_FILE"
fi

setsid bash -lc "exec python3 -u '$ROOT_DIR/scripts/playback_control_server.py' \
  --bag-path '$BAG_PATH' \
  --container-name '$CONTAINER_NAME' \
  --port '$PORT' \
  $([[ \"$LOOP_FLAG\" == \"1\" ]] && printf '%s' '--loop') \
  >'$LOG_FILE' 2>&1" >/dev/null 2>&1 < /dev/null &

sleep 0.6
for _ in {1..10}; do
  if curl -fs "http://127.0.0.1:$PORT/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

SERVER_PID="$(pgrep -f 'playback_control_server.py' | tail -n 1 || true)"
echo "playback-control ready on http://localhost:$PORT"
echo "log: $LOG_FILE"
if [[ -n "$SERVER_PID" ]]; then
  echo "$SERVER_PID" > "$PID_FILE"
fi
if ! curl -fs "http://127.0.0.1:$PORT/status" >/dev/null 2>&1; then
  echo "playback-control is still warming up; retry in a second if the UI slider is disabled." >&2
fi
