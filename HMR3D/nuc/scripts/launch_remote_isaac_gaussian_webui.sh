#!/usr/bin/env bash
set -euo pipefail

REMOTE="gpu-worker"
REMOTE_ROOT="/home/chatsign/gs-sdf"
WEB_PORT="55173"
BRIDGE_PORT="8890"
MAPPER_PORT="8891"
WORLD_NAV_PORT="8892"
HIFI_PORT="8876"
WEB_SCENE="/scenes/isaac-gaussian-online-offline-gs/manifest.json"
WEB_MODE="hifi"
START_HIFI="1"
START_ISAAC="1"
START_WORLD_NAV="1"
START_TUNNEL="1"
RESTART="0"
STOP="0"
ISAAC_SCENE="full-warehouse"
SCENE_USD="/media/chatsign/data-002/isaac/nav-mvp/assets/nurec_galileo_gssdf_mesh_collision.usd"
ISAAC_DEVICE="cuda:0"
ISAAC_BASE_HEIGHT=""
ISAAC_PATH_RADIUS="4.5"
ISAAC_PATH_SPEED="1.0"
ISAAC_HOLD_OPEN_SEC="0"
ISAAC_ENV="/media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh"
WORLD_NAV_VALIDATION_REPORT=""
WORLD_NAV_ROBOT_MODEL="unitree-go2"
DISPLAY_VALUE=":0"
XAUTHORITY_VALUE="/var/run/lightdm/root/:0"
RENDERER_URL=""
HIFI_OUTPUT_DIR="/media/chatsign/data-002/gs-sdf/runtime/output/2026-04-02-23-01-07_fast_livo2_compressed.bag_fastlivo_cbd_host.yaml"
HIFI_VIEW_CONTAINER="gssdf-view-quality"
HIFI_BRIDGE_CONTAINER="gssdf-hifi-bridge"
WEB_LIVE_GAUSSIAN_PATCH_MODE="fallback"
LOG_DIR=""

usage() {
  cat <<'EOF'
Launch the remote Isaac Gaussian WebUI stack from this machine.

Default stack:
  gpu-worker mapper  :8891
  gpu-worker bridge  :8890
  gpu-worker WebUI   :55173
  gpu-worker GS-SDF HiFi renderer :8876
  gpu-worker world-nav API :8892
  optional Isaac Lab publisher -> bridge
  local SSH tunnel for browser access

Usage:
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh [options]

Common options:
  --remote HOST              SSH target. Default: gpu-worker
  --scene-usd PATH           Isaac USD scene to load.
                             Default: converted NuRec/GS-SDF collision USD.
  --use-built-in-scene       Ignore --scene-usd/default USD and use --isaac-scene.
  --isaac-scene NAME         Built-in Isaac scene if --scene-usd is omitted.
                             Choices are handled remotely; common values:
                             full-warehouse, warehouse, office, hospital, empty.
  --no-isaac                 Start only mapper/bridge/world-nav/WebUI.
  --no-hifi                  Do not start the offline GS-SDF HiFi renderer.
  --no-world-nav             Skip world-nav API.
  --no-tunnel                Do not open local SSH tunnel after starting remote services.
  --restart                  Kill existing listeners on the managed remote ports first.
  --stop                     Stop the managed remote stack and exit.
  --renderer-url URL         GS-SDF HiFi renderer bridge.
                             Default with --hifi: http://127.0.0.1:8876.
  --hifi-output-dir PATH     Offline trained GS-SDF output dir for HiFi.
  --hifi-port PORT           Offline GS-SDF HiFi renderer bridge port. Default: 8876.
  --live-gaussian-patch-mode MODE
                             WebUI live mapper patch policy for the generated scene.
                             Default: fallback, which keeps offline trained GS visible.
                             Useful: fallback, disabled, replace.
  --web-mode MODE            WebUI mode. Default: hifi. Useful: hifi, live, gs.
  --web-port PORT            Remote/local WebUI port. Default: 55173.
  --world-nav-validation-report PATH
                             Validation report used by world-nav PointGoal.
                             If omitted, remote world-nav uses its script default.
  --world-nav-robot-model MODEL
                             unitree-go2 or capsule. Default: unitree-go2.
  --help                     Show this help.

Examples:
  # One-click online demo using our converted NuRec/GS-SDF scene.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh

  # One-click with an explicit converted collision USD.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh \
    --scene-usd /media/chatsign/data-002/isaac/nav-mvp/assets/nurec_galileo_gssdf_mesh_collision.usd

  # Fall back to a built-in Isaac scene.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --use-built-in-scene

  # Stop the remote stack.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --stop

  # Only bring up the browser stack, no Isaac publisher.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --no-isaac

  # Browser WebGL view of the offline trained Gaussian chunks.
  HMR3D/nuc/scripts/launch_remote_isaac_gaussian_webui.sh --web-mode gs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --web-port)
      WEB_PORT="$2"
      shift 2
      ;;
    --bridge-port)
      BRIDGE_PORT="$2"
      shift 2
      ;;
    --mapper-port)
      MAPPER_PORT="$2"
      shift 2
      ;;
    --world-nav-port)
      WORLD_NAV_PORT="$2"
      shift 2
      ;;
    --hifi-port)
      HIFI_PORT="$2"
      shift 2
      ;;
    --web-scene)
      WEB_SCENE="$2"
      shift 2
      ;;
    --web-mode)
      WEB_MODE="$2"
      shift 2
      ;;
    --scene-usd)
      SCENE_USD="$2"
      shift 2
      ;;
    --use-built-in-scene)
      SCENE_USD=""
      shift
      ;;
    --isaac-scene)
      ISAAC_SCENE="$2"
      shift 2
      ;;
    --isaac-device)
      ISAAC_DEVICE="$2"
      shift 2
      ;;
    --base-height)
      ISAAC_BASE_HEIGHT="$2"
      shift 2
      ;;
    --path-radius)
      ISAAC_PATH_RADIUS="$2"
      shift 2
      ;;
    --path-speed)
      ISAAC_PATH_SPEED="$2"
      shift 2
      ;;
    --hold-open-sec)
      ISAAC_HOLD_OPEN_SEC="$2"
      shift 2
      ;;
    --isaac-env)
      ISAAC_ENV="$2"
      shift 2
      ;;
    --world-nav-validation-report)
      WORLD_NAV_VALIDATION_REPORT="$2"
      shift 2
      ;;
    --world-nav-robot-model)
      WORLD_NAV_ROBOT_MODEL="$2"
      shift 2
      ;;
    --display)
      DISPLAY_VALUE="$2"
      shift 2
      ;;
    --xauthority)
      XAUTHORITY_VALUE="$2"
      shift 2
      ;;
    --renderer-url)
      RENDERER_URL="$2"
      shift 2
      ;;
    --hifi-output-dir)
      HIFI_OUTPUT_DIR="$2"
      shift 2
      ;;
    --hifi-view-container)
      HIFI_VIEW_CONTAINER="$2"
      shift 2
      ;;
    --hifi-bridge-container)
      HIFI_BRIDGE_CONTAINER="$2"
      shift 2
      ;;
    --live-gaussian-patch-mode)
      WEB_LIVE_GAUSSIAN_PATCH_MODE="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --no-hifi)
      START_HIFI="0"
      shift
      ;;
    --no-isaac)
      START_ISAAC="0"
      shift
      ;;
    --no-world-nav)
      START_WORLD_NAV="0"
      shift
      ;;
    --no-tunnel)
      START_TUNNEL="0"
      shift
      ;;
    --restart)
      RESTART="1"
      shift
      ;;
    --stop)
      STOP="1"
      START_TUNNEL="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Remote target: $REMOTE"
echo "Remote root:   $REMOTE_ROOT"
echo "WebUI:         http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
if [[ "$START_HIFI" == "1" ]]; then
  echo "HiFi GS:       ${RENDERER_URL:-http://127.0.0.1:${HIFI_PORT}}"
  echo "HiFi output:   $HIFI_OUTPUT_DIR"
fi
if [[ -n "$SCENE_USD" ]]; then
  echo "Isaac scene:   $SCENE_USD"
else
  echo "Isaac scene:   built-in ${ISAAC_SCENE}"
fi

ssh "$REMOTE" env \
  REMOTE_ROOT="$REMOTE_ROOT" \
  WEB_PORT="$WEB_PORT" \
  BRIDGE_PORT="$BRIDGE_PORT" \
  MAPPER_PORT="$MAPPER_PORT" \
  WORLD_NAV_PORT="$WORLD_NAV_PORT" \
  HIFI_PORT="$HIFI_PORT" \
  WEB_SCENE="$WEB_SCENE" \
  WEB_MODE="$WEB_MODE" \
  START_HIFI="$START_HIFI" \
  START_ISAAC="$START_ISAAC" \
  START_WORLD_NAV="$START_WORLD_NAV" \
  RESTART="$RESTART" \
  STOP="$STOP" \
  ISAAC_SCENE="$ISAAC_SCENE" \
  SCENE_USD="$SCENE_USD" \
  ISAAC_DEVICE="$ISAAC_DEVICE" \
  ISAAC_BASE_HEIGHT="$ISAAC_BASE_HEIGHT" \
  ISAAC_PATH_RADIUS="$ISAAC_PATH_RADIUS" \
  ISAAC_PATH_SPEED="$ISAAC_PATH_SPEED" \
  ISAAC_HOLD_OPEN_SEC="$ISAAC_HOLD_OPEN_SEC" \
  ISAAC_ENV="$ISAAC_ENV" \
  WORLD_NAV_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT" \
  WORLD_NAV_ROBOT_MODEL="$WORLD_NAV_ROBOT_MODEL" \
  DISPLAY_VALUE="$DISPLAY_VALUE" \
  XAUTHORITY_VALUE="$XAUTHORITY_VALUE" \
  RENDERER_URL="$RENDERER_URL" \
  HIFI_OUTPUT_DIR="$HIFI_OUTPUT_DIR" \
  HIFI_VIEW_CONTAINER="$HIFI_VIEW_CONTAINER" \
  HIFI_BRIDGE_CONTAINER="$HIFI_BRIDGE_CONTAINER" \
  WEB_LIVE_GAUSSIAN_PATCH_MODE="$WEB_LIVE_GAUSSIAN_PATCH_MODE" \
  LOG_DIR="$LOG_DIR" \
  bash -s <<'REMOTE_SCRIPT'
set -euo pipefail

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$REMOTE_ROOT/runtime/logs/one-click-isaac-webui/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LOG_DIR"

quote_cmd() {
  printf '%q ' "$@"
}

port_pids() {
  local port="$1"
  lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
}

port_is_open() {
  local port="$1"
  [[ -n "$(port_pids "$port")" ]]
}

stop_port() {
  local port="$1"
  local pids
  pids="$(port_pids "$port")"
  if [[ -n "$pids" ]]; then
    echo "Stopping remote listener(s) on port $port: $pids"
    kill $pids >/dev/null 2>&1 || true
    sleep 1
  fi
}

write_offline_gs_scene_manifest() {
  local target_dir="$REMOTE_ROOT/examples/web-ui/public/scenes/isaac-gaussian-online-offline-gs"
  local source_manifest="$REMOTE_ROOT/examples/web-ui/public/scenes/isaac-gaussian-online/manifest.json"
  local target_manifest="$target_dir/manifest.json"

  if [[ ! "$WEB_SCENE" =~ ^/scenes/isaac-gaussian-online-offline-gs/manifest\.json$ ]]; then
    return 0
  fi

  mkdir -p "$target_dir"
  python3 - "$source_manifest" "$target_manifest" "$WEB_LIVE_GAUSSIAN_PATCH_MODE" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
patch_mode = sys.argv[3]

manifest = json.loads(source_path.read_text())
manifest["sceneId"] = "isaac-gaussian-online-offline-gs"
source = dict(manifest.get("source") or {})
source["liveGaussianPatchMode"] = patch_mode
manifest["source"] = source
notes = manifest.get("externalViewer", {}).get("notes")
if isinstance(notes, str):
    manifest["externalViewer"]["notes"] = (
        notes
        + " This generated launch scene keeps the offline trained Gaussian visible "
        + f"with liveGaussianPatchMode={patch_mode!r}."
    )
target_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  echo "Generated WebUI scene manifest: $target_manifest"
}

start_hifi_renderer() {
  if [[ "$START_HIFI" != "1" ]]; then
    return 0
  fi

  if [[ -z "$RENDERER_URL" ]]; then
    RENDERER_URL="http://127.0.0.1:${HIFI_PORT}"
  fi

  if [[ "$RESTART" == "1" ]]; then
    docker rm -f "$HIFI_BRIDGE_CONTAINER" "$HIFI_VIEW_CONTAINER" >/dev/null 2>&1 || true
    stop_port "$HIFI_PORT"
  fi

  if port_is_open "$HIFI_PORT"; then
    echo "hifi-renderer already listening on :$HIFI_PORT; using $RENDERER_URL."
    return 0
  fi

  if [[ ! -d "$HIFI_OUTPUT_DIR" ]]; then
    echo "Offline GS-SDF output dir does not exist: $HIFI_OUTPUT_DIR" >&2
    exit 1
  fi

  local hifi_cmd
  hifi_cmd="$(quote_cmd env \
    VIEW_WIDTH=1920 \
    VIEW_HEIGHT=1536 \
    WAIT_SECONDS=120 \
    bash "$REMOTE_ROOT/scripts/launch_gssdf_hifi_stack.sh" \
      "$HIFI_OUTPUT_DIR" \
      "$HIFI_VIEW_CONTAINER" \
      "$HIFI_PORT" \
      "$HIFI_BRIDGE_CONTAINER")"
  start_service "hifi-renderer" "$HIFI_PORT" "$hifi_cmd"
  wait_http "hifi-renderer" "$RENDERER_URL/status" 1 180
}

if [[ "$STOP" == "1" ]]; then
  echo "Stopping remote one-click Isaac Gaussian WebUI stack."
  pkill -f "run_isaac_gaussian_online_demo.py.*--bridge-url http://127.0.0.1:${BRIDGE_PORT}" >/dev/null 2>&1 || true
  docker rm -f "$HIFI_BRIDGE_CONTAINER" "$HIFI_VIEW_CONTAINER" >/dev/null 2>&1 || true
  stop_port "$HIFI_PORT"
  stop_port "$WEB_PORT"
  stop_port "$WORLD_NAV_PORT"
  stop_port "$BRIDGE_PORT"
  stop_port "$MAPPER_PORT"
  echo "Stopped managed remote stack."
  exit 0
fi

start_service() {
  local name="$1"
  local port="$2"
  local command="$3"
  local log_path="$LOG_DIR/${name}.log"
  local pid_path="$LOG_DIR/${name}.pid"

  if [[ "$RESTART" == "1" ]]; then
    stop_port "$port"
  fi

  if port_is_open "$port"; then
    echo "$name already listening on :$port; leaving it running."
    return 0
  fi

  echo "Starting $name on :$port"
  echo "$command" > "$LOG_DIR/${name}.cmd"
  nohup bash -lc "$command" > "$log_path" 2>&1 < /dev/null &
  echo "$!" > "$pid_path"
}

wait_http() {
  local name="$1"
  local url="$2"
  local required="${3:-1}"
  local timeout="${4:-90}"

  for _ in $(seq 1 "$timeout"); do
    if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
      echo "$name ready: $url"
      return 0
    fi
    sleep 1
  done

  if [[ "$required" == "1" ]]; then
    echo "$name did not become ready: $url" >&2
    echo "Logs: $LOG_DIR/${name}.log" >&2
    tail -80 "$LOG_DIR/${name}.log" 2>/dev/null || true
    exit 1
  fi
  echo "$name not ready yet: $url"
}

wait_bridge_display_frame() {
  local timeout="${1:-180}"
  local expected_source="${2:-}"

  for _ in $(seq 1 "$timeout"); do
    if python3 - "$BRIDGE_PORT" "$expected_source" <<'PY'
import json
import sys
import urllib.request

port = sys.argv[1]
expected = sys.argv[2]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as response:
        status = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

if not status.get("ready"):
    sys.exit(1)
if expected and status.get("displaySource") != expected:
    sys.exit(1)
sys.exit(0)
PY
    then
      curl -fsS -m 2 "http://127.0.0.1:${BRIDGE_PORT}/frame.jpg" >/dev/null
      echo "bridge display frame ready: http://127.0.0.1:${BRIDGE_PORT}/frame.jpg"
      return 0
    fi
    sleep 1
  done

  echo "bridge did not produce the expected display frame." >&2
  echo "Bridge status:" >&2
  curl -fsS -m 2 "http://127.0.0.1:${BRIDGE_PORT}/status" >&2 || true
  echo >&2
  echo "Logs: $LOG_DIR/bridge.log and $LOG_DIR/isaac-publisher.log" >&2
  tail -80 "$LOG_DIR/bridge.log" 2>/dev/null || true
  tail -120 "$LOG_DIR/isaac-publisher.log" 2>/dev/null || true
  exit 1
}

write_offline_gs_scene_manifest
start_hifi_renderer

mapper_cmd="$(quote_cmd bash "$REMOTE_ROOT/scripts/launch_isaac_online_gaussian_mapper.sh" "$MAPPER_PORT")"
start_service "mapper" "$MAPPER_PORT" "$mapper_cmd"
wait_http "mapper" "http://127.0.0.1:${MAPPER_PORT}/status" 1 30

bridge_env=(MAPPER_URL="http://127.0.0.1:${MAPPER_PORT}")
if [[ -n "$RENDERER_URL" ]]; then
  bridge_env+=(RENDERER_URL="$RENDERER_URL")
  if port_is_open "$BRIDGE_PORT" && ! python3 - "$BRIDGE_PORT" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/status", timeout=2) as response:
        status = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

sys.exit(0 if status.get("rendererConfigured") else 1)
PY
  then
    echo "bridge already listening on :$BRIDGE_PORT without renderer; restarting it for $RENDERER_URL."
    stop_port "$BRIDGE_PORT"
  fi
fi
bridge_cmd="$(quote_cmd env "${bridge_env[@]}" bash "$REMOTE_ROOT/scripts/launch_isaac_gaussian_online_bridge.sh" "$BRIDGE_PORT")"
start_service "bridge" "$BRIDGE_PORT" "$bridge_cmd"
wait_http "bridge" "http://127.0.0.1:${BRIDGE_PORT}/status" 1 30

if [[ "$START_WORLD_NAV" == "1" ]]; then
  world_nav_cmd="$(quote_cmd env \
    DISPLAY="$DISPLAY_VALUE" \
    QT_X11_NO_MITSHM=1 \
    WORLD_NAV_VIEWER=1 \
    WORLD_NAV_HOLD_OPEN_SEC=60 \
    WORLD_NAV_EPISODE_HOLD_SEC=2 \
    WORLD_NAV_STEP_SLEEP=0.02 \
    WORLD_NAV_BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}" \
    WORLD_NAV_DEVICE="$ISAAC_DEVICE" \
    WORLD_NAV_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT" \
    WORLD_NAV_ROBOT_MODEL="$WORLD_NAV_ROBOT_MODEL" \
    bash "$REMOTE_ROOT/scripts/launch_world_nav_module.sh" "$WORLD_NAV_PORT")"
  start_service "world-nav" "$WORLD_NAV_PORT" "$world_nav_cmd"
  wait_http "world-nav" "http://127.0.0.1:${WORLD_NAV_PORT}/status" 1 30
fi

web_cmd="$(quote_cmd env \
  WEB_PORT="$WEB_PORT" \
  WEB_SCENE="$WEB_SCENE" \
  WEB_MODE="$WEB_MODE" \
  WEB_ISAAC_GAUSSIAN_ONLINE_PORT="$BRIDGE_PORT" \
  WEB_ISAAC_GAUSSIAN_MAPPER_PORT="$MAPPER_PORT" \
  WEB_WORLD_NAV_PORT="$WORLD_NAV_PORT" \
  bash "$REMOTE_ROOT/scripts/launch_web_ui_dev.sh")"
start_service "webui" "$WEB_PORT" "$web_cmd"
wait_http "webui" "http://127.0.0.1:${WEB_PORT}/" 1 60

if [[ "$START_ISAAC" == "1" ]]; then
  isaac_pattern="run_isaac_gaussian_online_demo.py.*--bridge-url http://127.0.0.1:${BRIDGE_PORT}"
  if [[ "$RESTART" == "1" ]]; then
    pkill -f "$isaac_pattern" >/dev/null 2>&1 || true
    sleep 1
  elif pgrep -f "$isaac_pattern" >/dev/null 2>&1; then
    echo "isaac-publisher already running for bridge :$BRIDGE_PORT; leaving it running."
    START_ISAAC="0"
  fi
fi

if [[ "$START_ISAAC" == "1" ]]; then
  isaac_args=(
    "$REMOTE_ROOT/scripts/run_isaac_gaussian_online_demo.py"
    --device "$ISAAC_DEVICE"
    --bridge-url "http://127.0.0.1:${BRIDGE_PORT}"
    --path-radius "$ISAAC_PATH_RADIUS"
    --path-speed "$ISAAC_PATH_SPEED"
    --hold-open-sec "$ISAAC_HOLD_OPEN_SEC"
  )
  if [[ -n "$SCENE_USD" ]]; then
    isaac_args+=(--scene-usd "$SCENE_USD" --scene empty)
  else
    isaac_args+=(--scene "$ISAAC_SCENE")
  fi
  if [[ -n "$ISAAC_BASE_HEIGHT" ]]; then
    isaac_args+=(--base-height "$ISAAC_BASE_HEIGHT")
  fi

  isaac_cmd="source $(printf '%q' "$ISAAC_ENV") && export OMNI_KIT_ACCEPT_EULA=YES DISPLAY=$(printf '%q' "$DISPLAY_VALUE") XAUTHORITY=$(printf '%q' "$XAUTHORITY_VALUE") QT_X11_NO_MITSHM=1 && \"\$ISAACSIM_PYTHON_EXE\" $(quote_cmd "${isaac_args[@]}")"
  echo "Starting isaac-publisher"
  echo "$isaac_cmd" > "$LOG_DIR/isaac-publisher.cmd"
  nohup bash -lc "$isaac_cmd" > "$LOG_DIR/isaac-publisher.log" 2>&1 < /dev/null &
  echo "$!" > "$LOG_DIR/isaac-publisher.pid"
  echo "Isaac publisher log: $LOG_DIR/isaac-publisher.log"
fi

if [[ "$START_HIFI" == "1" && "$START_ISAAC" == "1" ]]; then
  wait_bridge_display_frame 180 "renderer"
elif [[ "$START_ISAAC" == "1" ]]; then
  wait_bridge_display_frame 180
fi

cat > "$LOG_DIR/README.txt" <<EOF
Remote one-click Isaac Gaussian WebUI stack

WebUI:   http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}
Bridge:  http://127.0.0.1:${BRIDGE_PORT}/status
Mapper:  http://127.0.0.1:${MAPPER_PORT}/status
HiFi:    ${RENDERER_URL:-disabled}
World:   http://127.0.0.1:${WORLD_NAV_PORT}/status
Logs:    $LOG_DIR
EOF

echo "Remote stack launched."
echo "Logs: $LOG_DIR"
echo "Open through local tunnel:"
echo "  http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
REMOTE_SCRIPT

if [[ "$START_TUNNEL" == "1" ]]; then
  echo
  echo "Opening SSH tunnel. Keep this terminal open; press Ctrl-C to close the tunnel."
  echo "Browser URL:"
  echo "  http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
  exec ssh -N \
    -L "${WEB_PORT}:127.0.0.1:${WEB_PORT}" \
    -L "${BRIDGE_PORT}:127.0.0.1:${BRIDGE_PORT}" \
    -L "${MAPPER_PORT}:127.0.0.1:${MAPPER_PORT}" \
    -L "${WORLD_NAV_PORT}:127.0.0.1:${WORLD_NAV_PORT}" \
    -L "${HIFI_PORT}:127.0.0.1:${HIFI_PORT}" \
    "$REMOTE"
fi

echo "Browser URL:"
echo "  http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
