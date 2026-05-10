#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PORT="${1:-${WORLD_NAV_PORT:-8892}}"
ISAAC_ENV="${ISAAC_NAV_MVP_ACTIVATE:-/media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh}"
VALIDATION_REPORT="${WORLD_NAV_VALIDATION_REPORT:-/media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json}"
OUTPUT_DIR="${WORLD_NAV_OUTPUT_DIR:-/media/chatsign/data-002/isaac/nav-mvp/reports}"
BRIDGE_URL="${WORLD_NAV_BRIDGE_URL:-http://127.0.0.1:8890}"
DEVICE="${WORLD_NAV_DEVICE:-cuda:0}"
ROBOT_MODEL="${WORLD_NAV_ROBOT_MODEL:-unitree-go2}"
VIEWER="${WORLD_NAV_VIEWER:-0}"
STEP_SLEEP="${WORLD_NAV_STEP_SLEEP:-0.02}"
EPISODE_HOLD_SEC="${WORLD_NAV_EPISODE_HOLD_SEC:-1.0}"
HOLD_OPEN_SEC="${WORLD_NAV_HOLD_OPEN_SEC:-8}"
TIMEOUT_SEC="${WORLD_NAV_TIMEOUT_SEC:-45}"

echo "World nav module: http://localhost:${PORT}"
echo "Validation report: ${VALIDATION_REPORT}"
echo "Bridge URL: ${BRIDGE_URL}"
echo "Viewer mode: ${VIEWER}"

extra_args=()
if [[ "$VIEWER" == "1" || "$VIEWER" == "true" || "$VIEWER" == "yes" ]]; then
  export DISPLAY="${DISPLAY:-:0}"
  export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"
  extra_args+=(--viewer --step-sleep "$STEP_SLEEP" --episode-hold-sec "$EPISODE_HOLD_SEC" --hold-open-sec "$HOLD_OPEN_SEC")
fi

exec python3 "$ROOT_DIR/scripts/world_nav_module_server.py" \
  --port "$PORT" \
  --isaac-env "$ISAAC_ENV" \
  --validation-report "$VALIDATION_REPORT" \
  --output-dir "$OUTPUT_DIR" \
  --bridge-url "$BRIDGE_URL" \
  --device "$DEVICE" \
  --robot-model "$ROBOT_MODEL" \
  --timeout-sec "$TIMEOUT_SEC" \
  "${extra_args[@]}"
