#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WEB_UI_DIR="$ROOT_DIR/examples/web-ui"
HOST="${1:-${WEB_HOST:-0.0.0.0}}"
PORT="${2:-${WEB_PORT:-5173}}"
WEB_ADAPTER="${WEB_ADAPTER:-${ADAPTER:-generic}}"
WEB_SCENE="${WEB_SCENE:-}"
WEB_MODE="${WEB_MODE:-}"
WEB_LIVE_CONTRACT="${WEB_LIVE_CONTRACT:-}"
WEB_ROSBRIDGE_PORT="${WEB_ROSBRIDGE_PORT:-${ROSBRIDGE_PORT:-9090}}"
WEB_PLAYBACK_CONTROL_PORT="${WEB_PLAYBACK_CONTROL_PORT:-${CONTROL_PORT:-8765}}"
WEB_HIFI_BRIDGE_PORT="${WEB_HIFI_BRIDGE_PORT:-${HIFI_PORT:-8876}}"
WEB_HIFI_WINDOW_PORT="${WEB_HIFI_WINDOW_PORT:-${HIFI_WINDOW_PORT:-8877}}"
WEB_ISAAC_GAUSSIAN_ONLINE_PORT="${WEB_ISAAC_GAUSSIAN_ONLINE_PORT:-${ISAAC_GAUSSIAN_ONLINE_PORT:-8890}}"
WEB_ISAAC_GAUSSIAN_MAPPER_PORT="${WEB_ISAAC_GAUSSIAN_MAPPER_PORT:-${ISAAC_GAUSSIAN_MAPPER_PORT:-8891}}"
WEB_WORLD_NAV_PORT="${WEB_WORLD_NAV_PORT:-${WORLD_NAV_PORT:-8892}}"
WEB_AUTO_INSTALL_DEPS="${WEB_AUTO_INSTALL_DEPS:-1}"
WEB_LINGBOT_LIVE_ASSET_ROOT="${WEB_LINGBOT_LIVE_ASSET_ROOT:-${LINGBOT_LIVE_ASSET_ROOT:-}}"
WEB_LINGBOT_REAL2SIM_ASSET_ROOT="${WEB_LINGBOT_REAL2SIM_ASSET_ROOT:-${LINGBOT_REAL2SIM_ASSET_ROOT:-}}"

case "$WEB_ADAPTER" in
  lingbot-map|lingbot_map)
    WEB_SCENE="${WEB_SCENE:-/scenes/lingbot-map-viewer/manifest.json}"
    WEB_MODE="${WEB_MODE:-gs}"
    ;;
  lingbot-live|lingbot_map_live)
    WEB_SCENE="${WEB_SCENE:-/scenes/lingbot-live/manifest.json}"
    WEB_MODE="${WEB_MODE:-live}"
    WEB_LIVE_CONTRACT="${WEB_LIVE_CONTRACT:-/contracts/lingbot-map-ros2-live.live-contract.json}"
    ;;
  sgsslam|sgs-slam)
    WEB_SCENE="${WEB_SCENE:-/scenes/sgs-fastlivo-rectified-photoreal/manifest.json}"
    WEB_MODE="${WEB_MODE:-gs}"
    ;;
  citygaussian|citygaussianv2)
    WEB_SCENE="${WEB_SCENE:-/scenes/citygaussian-sciart/manifest.json}"
    WEB_MODE="${WEB_MODE:-gs}"
    ;;
  fastlivo|playback)
    WEB_MODE="${WEB_MODE:-playback}"
    ;;
  lingbot-playback|lingbot_map_playback)
    WEB_MODE="${WEB_MODE:-playback}"
    ;;
  gssdf|gs-sdf)
    WEB_SCENE="${WEB_SCENE:-/scenes/fast-livo2-compressed-live/manifest.json}"
    WEB_MODE="${WEB_MODE:-gs}"
    ;;
esac

EXISTING_PORT_PIDS="$(lsof -tiTCP:$PORT -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$EXISTING_PORT_PIDS" ]]; then
  echo "Stopping existing server on port $PORT: $EXISTING_PORT_PIDS"
  kill $EXISTING_PORT_PIDS >/dev/null 2>&1 || true
  sleep 1
fi

OPEN_URL="http://localhost:$PORT/"
query=()
if [[ -n "$WEB_SCENE" ]]; then
  query+=("scene=$WEB_SCENE")
fi
if [[ -n "$WEB_MODE" ]]; then
  query+=("mode=$WEB_MODE")
fi
if [[ -n "$WEB_LIVE_CONTRACT" ]]; then
  query+=("liveContract=$WEB_LIVE_CONTRACT")
fi
if [[ "${#query[@]}" -gt 0 ]]; then
  query_string="$(IFS='&'; echo "${query[*]}")"
  OPEN_URL="$OPEN_URL?$query_string"
fi

echo "Web adapter preset: $WEB_ADAPTER"
echo "rosbridge proxy: ws://127.0.0.1:$WEB_ROSBRIDGE_PORT"
echo "playback proxy:  http://127.0.0.1:$WEB_PLAYBACK_CONTROL_PORT"
echo "world-nav proxy: http://127.0.0.1:$WEB_WORLD_NAV_PORT"
if [[ -n "$WEB_LINGBOT_LIVE_ASSET_ROOT" || -n "$WEB_LINGBOT_REAL2SIM_ASSET_ROOT" ]]; then
  echo "lingbot assets: live=$WEB_LINGBOT_LIVE_ASSET_ROOT real2sim=$WEB_LINGBOT_REAL2SIM_ASSET_ROOT"
fi
echo "Open: $OPEN_URL"

export WEB_ROSBRIDGE_PORT
export WEB_PLAYBACK_CONTROL_PORT
export WEB_HIFI_BRIDGE_PORT
export WEB_HIFI_WINDOW_PORT
export WEB_ISAAC_GAUSSIAN_ONLINE_PORT
export WEB_ISAAC_GAUSSIAN_MAPPER_PORT
export WEB_WORLD_NAV_PORT
export VITE_ROSBRIDGE_TARGET="${VITE_ROSBRIDGE_TARGET:-ws://127.0.0.1:$WEB_ROSBRIDGE_PORT}"
export VITE_PLAYBACK_CONTROL_TARGET="${VITE_PLAYBACK_CONTROL_TARGET:-http://127.0.0.1:$WEB_PLAYBACK_CONTROL_PORT}"
export VITE_HIFI_BRIDGE_TARGET="${VITE_HIFI_BRIDGE_TARGET:-http://127.0.0.1:$WEB_HIFI_BRIDGE_PORT}"
export VITE_HIFI_WINDOW_TARGET="${VITE_HIFI_WINDOW_TARGET:-http://127.0.0.1:$WEB_HIFI_WINDOW_PORT}"
export VITE_ISAAC_GAUSSIAN_ONLINE_TARGET="${VITE_ISAAC_GAUSSIAN_ONLINE_TARGET:-http://127.0.0.1:$WEB_ISAAC_GAUSSIAN_ONLINE_PORT}"
export VITE_ISAAC_GAUSSIAN_MAPPER_TARGET="${VITE_ISAAC_GAUSSIAN_MAPPER_TARGET:-http://127.0.0.1:$WEB_ISAAC_GAUSSIAN_MAPPER_PORT}"
export VITE_WORLD_NAV_TARGET="${VITE_WORLD_NAV_TARGET:-http://127.0.0.1:$WEB_WORLD_NAV_PORT}"
export VITE_LINGBOT_LIVE_ASSET_ROOT="${VITE_LINGBOT_LIVE_ASSET_ROOT:-$WEB_LINGBOT_LIVE_ASSET_ROOT}"
export VITE_LINGBOT_REAL2SIM_ASSET_ROOT="${VITE_LINGBOT_REAL2SIM_ASSET_ROOT:-$WEB_LINGBOT_REAL2SIM_ASSET_ROOT}"

cd "$WEB_UI_DIR"
if [[ ! -x node_modules/.bin/vite || ! -x node_modules/.bin/tsc ]]; then
  if [[ "$WEB_AUTO_INSTALL_DEPS" == "1" ]]; then
    echo "Web UI dependencies are missing; running npm ci in $WEB_UI_DIR"
    npm ci
  else
    echo "Web UI dependencies are missing. Run: cd $WEB_UI_DIR && npm ci" >&2
    exit 1
  fi
fi
npm run dev -- --host "$HOST" --port "$PORT"
