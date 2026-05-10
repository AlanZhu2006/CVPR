#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
CACHE_DIR="${1:-$ROOT_DIR/runtime/playback-cache/lingbot_church}"
PLAYBACK_CONTAINER="${2:-lingbot-map-playback}"
ROSBRIDGE_PORT="${3:-9090}"
WEB_PORT="${4:-5173}"
CONTROL_PORT="${5:-8765}"

DISABLE_HIFI=1 WEB_ADAPTER=lingbot-playback \
  bash "$ROOT_DIR/scripts/launch_fastlivo_cached_playback_stack.sh" \
    "$CACHE_DIR" \
    "$PLAYBACK_CONTAINER" \
    "$ROSBRIDGE_PORT" \
    "$WEB_PORT" \
    "$CONTROL_PORT"
