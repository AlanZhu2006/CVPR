#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CACHE_DIR="${1:-$ROOT_DIR/runtime/playback-cache/fast_livo2_compressed}"
PLAYBACK_CONTAINER="${2:-fastlivo-cached-playback}"
ROSBRIDGE_PORT="${3:-9090}"
WEB_PORT="${4:-5173}"
CONTROL_PORT="${5:-8765}"
HIFI_PORT="${HIFI_PORT:-8876}"
HIFI_VIEW_CONTAINER="${HIFI_VIEW_CONTAINER:-gssdf-view-quality}"
HIFI_BRIDGE_CONTAINER="${HIFI_BRIDGE_CONTAINER:-gssdf-hifi-bridge}"
HIFI_SCENE_MANIFEST="${HIFI_SCENE_MANIFEST:-$ROOT_DIR/examples/web-ui/public/scenes/fast-livo2-compressed-live/manifest.json}"
HIFI_OUTPUT_DIR="${HIFI_OUTPUT_DIR:-}"

resolve_hifi_output_dir() {
  if [[ -n "$HIFI_OUTPUT_DIR" ]]; then
    printf '%s\n' "$HIFI_OUTPUT_DIR"
    return 0
  fi
  if [[ ! -f "$HIFI_SCENE_MANIFEST" ]]; then
    return 1
  fi
  python3 - "$HIFI_SCENE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
try:
    manifest = json.loads(manifest_path.read_text())
except Exception:
    sys.exit(1)

output_dir = manifest.get("training", {}).get("outputDir")
if not output_dir:
    sys.exit(1)
print(output_dir)
PY
}

CONTROL_PORT="$CONTROL_PORT" bash "$ROOT_DIR/scripts/launch_fastlivo_cached_playback.sh" "$CACHE_DIR" "$PLAYBACK_CONTAINER"
bash "$ROOT_DIR/scripts/launch_rosbridge_sidecar.sh" "$PLAYBACK_CONTAINER" "$ROSBRIDGE_PORT"

if [[ "${DISABLE_HIFI:-0}" != "1" ]]; then
  if RESOLVED_HIFI_OUTPUT_DIR="$(resolve_hifi_output_dir)"; then
    if [[ -d "$RESOLVED_HIFI_OUTPUT_DIR" ]]; then
      VIEW_WIDTH="${VIEW_WIDTH:-1920}" VIEW_HEIGHT="${VIEW_HEIGHT:-1536}" MJPEG_MAX_FPS="${MJPEG_MAX_FPS:-45}" \
        bash "$ROOT_DIR/scripts/launch_gssdf_hifi_stack.sh" \
          "$RESOLVED_HIFI_OUTPUT_DIR" \
          "$HIFI_VIEW_CONTAINER" \
          "$HIFI_PORT" \
          "$HIFI_BRIDGE_CONTAINER"
    else
      echo "Skipping HiFi stack: output dir does not exist: $RESOLVED_HIFI_OUTPUT_DIR" >&2
    fi
  else
    echo "Skipping HiFi stack: could not resolve output dir from $HIFI_SCENE_MANIFEST" >&2
  fi
fi

WEB_ADAPTER="${WEB_ADAPTER:-fastlivo}" \
WEB_ROSBRIDGE_PORT="$ROSBRIDGE_PORT" \
WEB_PLAYBACK_CONTROL_PORT="$CONTROL_PORT" \
WEB_HIFI_BRIDGE_PORT="$HIFI_PORT" \
bash "$ROOT_DIR/scripts/launch_web_ui_dev.sh" 0.0.0.0 "$WEB_PORT"
