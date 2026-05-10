#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd -P)"

VIDEO_PATH="${VIDEO_PATH:-}"
VIDEO_DIR="${VIDEO_DIR:-$WORKSPACE_ROOT/videos}"
VIDEO_GLOB="${VIDEO_GLOB:-*.mp4}"
FRAMES_DIR="${FRAMES_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_ROOT/nuc_output/video_real2sim_playback/live}"
REAL2SIM_DIR="${REAL2SIM_DIR:-$WORKSPACE_ROOT/nuc_output/video_real2sim_playback/real2sim}"
PLAYBACK_PY="${PLAYBACK_PY:-$WORKSPACE_ROOT/scripts/real2sim/run_video_real2sim_playback_webui.py}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-5173}"
WEB_CONTRACT="${WEB_CONTRACT:-/contracts/lingbot-map-video-playback.live-contract.json}"
BINARY_CLOUD_WS_PORT="${BINARY_CLOUD_WS_PORT:-19093}"
GLOBAL_BINARY_CLOUD_WS_PORT="${GLOBAL_BINARY_CLOUD_WS_PORT:-19094}"
PLAYBACK_CONTROL_PORT="${PLAYBACK_CONTROL_PORT:-${CONTROL_PORT:-8765}}"

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p "$OUTPUT_DIR" "$REAL2SIM_DIR"

for port in "$BINARY_CLOUD_WS_PORT" "$GLOBAL_BINARY_CLOUD_WS_PORT" "$PLAYBACK_CONTROL_PORT" "$WEB_PORT"; do
  pids_on_port="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids_on_port" ]]; then
    echo "Stopping existing listener on port $port: $pids_on_port"
    kill $pids_on_port >/dev/null 2>&1 || true
    sleep 0.5
  fi
done

playback_args=(
  "$PLAYBACK_PY"
  --output-dir "$OUTPUT_DIR"
  --real2sim-dir "$REAL2SIM_DIR"
  --extract-fps "${EXTRACT_FPS:-2}"
  --playback-fps "${PLAYBACK_FPS:-2}"
  --max-frames "${MAX_FRAMES:-80}"
  --points-per-frame "${POINTS_PER_FRAME:-8000}"
  --max-global-points "${MAX_GLOBAL_POINTS:-180000}"
  --global-publish-every "${GLOBAL_PUBLISH_EVERY:-1}"
  --global-voxel-size "${LINGBOT_GLOBAL_VOXEL_SIZE:-0.025}"
  --binary-cloud-ws-port "$BINARY_CLOUD_WS_PORT"
  --global-binary-cloud-ws-port "$GLOBAL_BINARY_CLOUD_WS_PORT"
  --binary-cloud-ws-host "${BINARY_CLOUD_WS_HOST:-0.0.0.0}"
  --control-port "$PLAYBACK_CONTROL_PORT"
  --write-ply-every "${WRITE_PLY_EVERY:-0}"
)
if [[ "${PLAYBACK_LOOP:-1}" == "0" ]]; then
  playback_args+=(--no-loop)
else
  playback_args+=(--loop)
fi

if [[ -n "$FRAMES_DIR" ]]; then
  playback_args+=(--frames-dir "$FRAMES_DIR")
elif [[ -n "$VIDEO_PATH" ]]; then
  playback_args+=(--video "$VIDEO_PATH")
else
  playback_args+=(--video-dir "$VIDEO_DIR" --video-glob "$VIDEO_GLOB")
fi
if [[ -n "${RGB_PREVIEW_FRAMES_DIR:-}" ]]; then
  playback_args+=(--rgb-preview-frames-dir "$RGB_PREVIEW_FRAMES_DIR")
fi
if [[ -n "${LINGBOT_PREDICTIONS_NPZ:-}" ]]; then
  playback_args+=(--lingbot-predictions-npz "$LINGBOT_PREDICTIONS_NPZ")
  if [[ -n "${LINGBOT_SUMMARY_JSON:-}" ]]; then
    playback_args+=(--lingbot-summary-json "$LINGBOT_SUMMARY_JSON")
  fi
  playback_args+=(--lingbot-geometry-source "${LINGBOT_GEOMETRY_SOURCE:-depth}")
  if [[ "${PRECOMPUTE_LINGBOT_CLOUDS:-1}" == "0" ]]; then
    playback_args+=(--no-precompute-lingbot-clouds)
  else
    playback_args+=(--precompute-lingbot-clouds)
  fi
  playback_args+=(--lingbot-conf-percentile "${LINGBOT_CONF_PERCENTILE:-45}")
  if [[ "${NORMALIZE_LINGBOT_WORLD:-1}" == "0" ]]; then
    playback_args+=(--no-normalize-lingbot-world)
  else
    playback_args+=(--normalize-lingbot-world)
  fi
fi

if [[ "${PREWARM_REAL2SIM_ASSETS:-1}" == "1" ]]; then
  PREWARM_GAUSSIAN_MANIFEST="${PREWARM_GAUSSIAN_MANIFEST:-$ROOT_DIR/examples/web-ui/public/scenes/lingbot-live/manifest.json}"
  echo "Prewarming default real2sim Gaussian assets into OS cache"
  python3 - "$PREWARM_GAUSSIAN_MANIFEST" "$REAL2SIM_DIR" <<'PY' \
    | xargs -r cat >/dev/null
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
real2sim_dir = Path(sys.argv[2])
try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    manifest = {}

gaussian = manifest.get("gaussian") or {}
for chunk in gaussian.get("chunks") or []:
    url = chunk.get("url") or ""
    marker = "/real2sim/"
    if marker not in url:
        continue
    rel = url.split(marker, 1)[1]
    path = real2sim_dir / rel
    if path.is_file():
        print(path)
PY
fi

echo "Starting video real-to-sim playback sidecar"
python3 "${playback_args[@]}" &
pids+=("$!")

echo "Starting GS Console WebUI"
(
  export WEB_ADAPTER=lingbot-live
  export WEB_MODE=live
  export WEB_LIVE_CONTRACT="$WEB_CONTRACT"
  export WEB_LINGBOT_LIVE_ASSET_ROOT="$OUTPUT_DIR"
  export WEB_LINGBOT_REAL2SIM_ASSET_ROOT="$REAL2SIM_DIR"
  export WEB_ROSBRIDGE_PORT="${WEB_ROSBRIDGE_PORT:-9090}"
  export WEB_PLAYBACK_CONTROL_PORT="$PLAYBACK_CONTROL_PORT"
  export WEB_WORLD_NAV_PORT="${WEB_WORLD_NAV_PORT:-8892}"
  exec "$ROOT_DIR/scripts/launch_web_ui_dev.sh" "$WEB_HOST" "$WEB_PORT"
) &
pids+=("$!")

echo "Video real-to-sim playback stack is up."
echo "Open: http://localhost:$WEB_PORT/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=$WEB_CONTRACT"
echo "LAN:  http://$(hostname -I | awk '{print $1}'):$WEB_PORT/?scene=/scenes/lingbot-live/manifest.json\\&mode=live\\&liveContract=$WEB_CONTRACT"
echo "live root: $OUTPUT_DIR"
echo "real2sim:  $REAL2SIM_DIR"

wait
