#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BAG_PATH="${1:-/home/chatsign/fast_livo2_compressed.bag}"
PLAYBACK_CONTAINER="${2:-fastlivo-direct-playback}"
ROSBRIDGE_PORT="${3:-9090}"
WEB_PORT="${4:-5173}"
CONTROL_PORT="${5:-8765}"

CONTROL_PORT="$CONTROL_PORT" bash "$ROOT_DIR/scripts/launch_fastlivo_direct_playback.sh" "$BAG_PATH" "$PLAYBACK_CONTAINER"
bash "$ROOT_DIR/scripts/launch_rosbridge_sidecar.sh" "$PLAYBACK_CONTAINER" "$ROSBRIDGE_PORT"
bash "$ROOT_DIR/scripts/launch_web_ui_dev.sh" 0.0.0.0 "$WEB_PORT"
