#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8892}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
VALIDATION_REPORT="${NAV2_VALIDATION_REPORT:-${WORLD_NAV_VALIDATION_REPORT:-/media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json}}"
OUTPUT_DIR="${NAV2_OUTPUT_DIR:-${WORLD_NAV_OUTPUT_DIR:-/media/chatsign/data-002/isaac/nav-mvp/reports}}"
BRIDGE_URL="${NAV2_BRIDGE_URL:-${WORLD_NAV_BRIDGE_URL:-http://127.0.0.1:8890}}"
ROBOT_RADIUS="${NAV2_ROBOT_RADIUS:-0.28}"
COLLISION_RADIUS="${NAV2_COLLISION_RADIUS:-$ROBOT_RADIUS}"
START_POLICY="${NAV2_START_POLICY:-validation}"
START_CLEARANCE_MARGIN="${NAV2_START_CLEARANCE_MARGIN:-0.08}"
MOTION_MODE="${NAV2_MOTION_MODE:-cmd_vel}"
BRIDGE_POSE_MODE="${NAV2_BRIDGE_POSE_MODE:-pose-and-cloud}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/nav2_baseline_logs}"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup not found: $ROS_SETUP" >&2
  exit 1
fi

set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ -n "${NAV2_MAP_YAML:-}" ]]; then
  MAP_YAML="$NAV2_MAP_YAML"
else
  MAP_YAML="$(python3 - "$VALIDATION_REPORT" <<'PY'
import json
import sys
from pathlib import Path

report = Path(sys.argv[1])
if not report.is_file():
    raise SystemExit("")
payload = json.loads(report.read_text())
print(payload.get("inputs", {}).get("map_yaml", ""))
PY
)"
fi

if [[ -z "$MAP_YAML" || ! -f "$MAP_YAML" ]]; then
  echo "Nav2 map YAML is missing. Set NAV2_MAP_YAML or WORLD_NAV_VALIDATION_REPORT. Got: $MAP_YAML" >&2
  exit 1
fi

PARAMS_FILE="$LOG_DIR/nav2_params.yaml"
python3 "$SCRIPT_DIR/write_nav2_baseline_params.py" \
  --map-yaml "$MAP_YAML" \
  --output "$PARAMS_FILE" \
  --robot-radius "$ROBOT_RADIUS" >/dev/null

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

echo "Starting Nav2 stack with map: $MAP_YAML"
python3 "$SCRIPT_DIR/nav2_baseline_launch.py" \
  --params-file "$PARAMS_FILE" \
  >"$LOG_DIR/nav2-stack.log" 2>&1 &
pids+=("$!")

echo "Starting Nav2 PGM robot bridge"
python3 "$SCRIPT_DIR/nav2_pgm_robot_bridge.py" \
  --map-yaml "$MAP_YAML" \
  --validation-report "$VALIDATION_REPORT" \
  --bridge-url "$BRIDGE_URL" \
  --robot-radius "$COLLISION_RADIUS" \
  --start-clearance-margin "$START_CLEARANCE_MARGIN" \
  --start-policy "$START_POLICY" \
  --motion-mode "$MOTION_MODE" \
  --bridge-pose-mode "$BRIDGE_POSE_MODE" \
  >"$LOG_DIR/nav2-robot-bridge.log" 2>&1 &
pids+=("$!")

echo "Starting Nav2 HTTP control server on :$PORT"
python3 "$SCRIPT_DIR/nav2_baseline_control_server.py" \
  --port "$PORT" \
  --validation-report "$VALIDATION_REPORT" \
  --map-yaml "$MAP_YAML" \
  --start-policy "$START_POLICY" \
  --robot-radius "$COLLISION_RADIUS" \
  --start-clearance-margin "$START_CLEARANCE_MARGIN" \
  --bridge-url "$BRIDGE_URL" \
  >"$LOG_DIR/nav2-control.log" 2>&1 &
control_pid="$!"
pids+=("$control_pid")

wait "$control_pid"
