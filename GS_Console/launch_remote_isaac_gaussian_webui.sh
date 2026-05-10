#!/usr/bin/env bash
set -euo pipefail

REMOTE="local"
REMOTE_ROOT="/home/chatsign/gs-sdf"
WEB_PORT="55173"
BRIDGE_PORT="8890"
MAPPER_PORT="8891"
WORLD_NAV_PORT="8892"
HIFI_PORT="8876"
WEB_SCENE="/scenes/isaac-gaussian-online-offline-gs/manifest.json"
WEB_VISUAL_SCENE="/scenes/nurec-galileo-auto3dgrt/manifest.json"
WEB_MODE="hifi"
WEB_MODE_SET="0"
NAV_MODE="0"
START_HIFI="1"
START_ISAAC="1"
START_WORLD_NAV="1"
START_TUNNEL="1"
RESTART="0"
STOP="0"
NO_ISAAC_SET="0"
ISAAC_SCENE="full-warehouse"
SCENE_USD=""
SCENE_USD_AUTO="1"
ISAAC_DEVICE="cuda:0"
ISAAC_CONTROL_SOURCE="scripted"
ISAAC_BASE_HEIGHT=""
ISAAC_PATH_RADIUS="4.5"
ISAAC_PATH_SPEED="1.0"
ISAAC_HOLD_OPEN_SEC="0"
ISAAC_TRAINING_TRAJECTORY="1"
ISAAC_TRAINING_TRAJECTORY_SPLIT="train"
ISAAC_TRAINING_RGB_SOURCE="trajectory"
ISAAC_ENV="/media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh"
DISPLAY_VALUE=":0"
XAUTHORITY_VALUE="/var/run/lightdm/root/:0"
RENDERER_URL=""
HIFI_OUTPUT_DIR="/media/chatsign/data-002/gs-sdf/runtime/output/2026-04-28-10-57-18_nurec_galileo_frontleft_auto3dgrt_hq_stride10_surface40k_z3_gssdf_colmap_gssdf_colmap.yaml"
HIFI_VIEW_CONTAINER="gssdf-view-quality"
HIFI_BRIDGE_CONTAINER="gssdf-hifi-bridge"
HIFI_BASE_CONFIG="/root/gs_sdf_ws/src/GS-SDF/config/base.yaml"
HIFI_VIEW_WIDTH=""
HIFI_VIEW_HEIGHT=""
HIFI_RENDER_POSE_MAX_DISTANCE="50"
WEB_LIVE_GAUSSIAN_PATCH_MODE="fallback"
WEB_GAUSSIAN_PREPROCESS="${WEB_GAUSSIAN_PREPROCESS:-chunks}"
WEB_GAUSSIAN_CHUNK_GRID="${WEB_GAUSSIAN_CHUNK_GRID:-6,6}"
WEB_GAUSSIAN_CHUNK_MAX_SH="${WEB_GAUSSIAN_CHUNK_MAX_SH:-1}"
WEB_GAUSSIAN_CHUNK_OPACITY_THRESHOLD="${WEB_GAUSSIAN_CHUNK_OPACITY_THRESHOLD:-0.005}"
WORLD_NAV_VALIDATION_REPORT=""
WORLD_NAV_OUTPUT_DIR="/media/chatsign/data-002/isaac/nav-mvp/reports"
WORLD_NAV_ROBOT_MODEL="unitree-go2"
WORLD_NAV_ROBOT_MODEL_SET="0"
WORLD_NAV_BACKEND="world-nav"
NAV_MAP_INFLATE_RADIUS="${NAV_MAP_INFLATE_RADIUS:-}"
WORLD_NAV_VIEWER="1"
WORLD_NAV_HOLD_OPEN_SEC="8"
WORLD_NAV_EPISODE_HOLD_SEC="2"
WORLD_NAV_STEP_SLEEP="0.02"
WORLD_NAV_TIMEOUT_SEC="45"
LOG_DIR=""

usage() {
  cat <<'EOF'
Launch the Isaac Gaussian WebUI stack.

Default stack:
  local mapper  :8891
  local bridge  :8890
  local WebUI   :55173
  local GS-SDF HiFi renderer :8876
  local world-nav API :8892
  optional Isaac Lab publisher -> bridge
  optional SSH tunnel for browser access when --remote is used

Usage:
  ./launch_remote_isaac_gaussian_webui.sh [options]

Common options:
  --local                    Run directly on this machine. Default.
  --remote HOST              SSH target. Use this only when launching from another machine.
  --scene-usd PATH           Isaac USD scene to load.
                             Default: auto-generate one from the active offline
                             GS-SDF mesh under runtime/isaac_assets.
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
  --hifi-base-config PATH    GS-SDF base config path inside the Docker image.
                             Default: /root/gs_sdf_ws/src/GS-SDF/config/base.yaml.
  --hifi-view-size WxH       Optional HiFi view size override. By default the
                             trained scene config resolution is preserved.
  --visual-scene URL         Offline WebUI visual scene manifest to link into the
                             generated Isaac scene. Default:
                             /scenes/nurec-galileo-auto3dgrt/manifest.json
  --live-gaussian-patch-mode MODE
                             WebUI live mapper patch policy for the generated scene.
                             Default: fallback, which keeps offline trained GS visible.
                             Useful: fallback, disabled, replace.
  --gaussian-chunks          Preprocess the offline gs.ply into browser-streamable
                             SPZ chunks before syncing the WebUI scene. Default.
  --no-gaussian-chunks       Keep the raw gs.ply browser asset.
  --gaussian-chunk-grid X,Y  Chunk grid for --gaussian-chunks. Default: 6,6.
  --web-mode MODE            WebUI mode. Default: hifi. Useful: hifi, live, gs.
  --training-trajectory      Drive Isaac/bridge poses from the offline training video trajectory. Default.
  --scripted-isaac-path      Use the old scripted Isaac circle path instead of the training trajectory.
  --nav-mode                 Start the physical PointGoal baseline mode.
                             This disables the continuous Isaac training-video
                             publisher so world-nav is the only bridge pose
                             publisher. Defaults world-nav to capsule.
  --nav2-mode                Start the ROS2/Nav2 baseline backend on the same
                             WebUI world-nav API and drive the Isaac robot from
                             Nav2 /cmd_vel. Requires Nav2 Humble.
  --nav-backend BACKEND      Navigation backend for --nav-mode.
                             Choices: world-nav, nav2. Default: world-nav.
  --world-nav-robot-model MODEL
                             Robot proxy used by world-nav. Choices:
                             capsule, unitree-go2. Default: unitree-go2,
                             or capsule with --nav-mode.
  --web-port PORT            Remote/local WebUI port. Default: 55173.
  --help                     Show this help.

Examples:
  # One-click online demo using our converted NuRec/GS-SDF scene.
  ./launch_remote_isaac_gaussian_webui.sh

  # One-click with an explicit converted collision USD.
  ./launch_remote_isaac_gaussian_webui.sh \
    --scene-usd /media/chatsign/data-002/isaac/nav-mvp/assets/nurec_galileo_gssdf_mesh_collision.usd

  # Fall back to a built-in Isaac scene.
  ./launch_remote_isaac_gaussian_webui.sh --use-built-in-scene

  # Stop the managed stack.
  ./launch_remote_isaac_gaussian_webui.sh --stop

  # Only bring up the browser stack, no Isaac publisher.
  ./launch_remote_isaac_gaussian_webui.sh --no-isaac

  # Browser WebGL view of the offline trained Gaussian chunks.
  ./launch_remote_isaac_gaussian_webui.sh --web-mode gs

  # Launch from another machine over SSH.
  ./launch_remote_isaac_gaussian_webui.sh --remote gpu-worker --restart
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      REMOTE="local"
      shift
      ;;
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
    --visual-scene)
      WEB_VISUAL_SCENE="$2"
      shift 2
      ;;
    --web-mode)
      WEB_MODE="$2"
      WEB_MODE_SET="1"
      shift 2
      ;;
    --training-trajectory)
      ISAAC_TRAINING_TRAJECTORY="1"
      shift
      ;;
    --scripted-isaac-path)
      ISAAC_TRAINING_TRAJECTORY="0"
      shift
      ;;
    --nav-mode|--navigation-mode)
      NAV_MODE="1"
      START_ISAAC="0"
      START_WORLD_NAV="1"
      ISAAC_TRAINING_TRAJECTORY="0"
      shift
      ;;
    --nav2-mode)
      NAV_MODE="1"
      START_ISAAC="1"
      START_WORLD_NAV="1"
      ISAAC_CONTROL_SOURCE="ros2-cmd_vel"
      ISAAC_TRAINING_TRAJECTORY="0"
      WORLD_NAV_BACKEND="nav2"
      shift
      ;;
    --nav-backend)
      case "$2" in
        world-nav|nav2)
          WORLD_NAV_BACKEND="$2"
          ;;
        *)
          echo "--nav-backend expects world-nav or nav2, got: $2" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --world-nav-robot-model)
      case "$2" in
        capsule|unitree-go2)
          WORLD_NAV_ROBOT_MODEL="$2"
          WORLD_NAV_ROBOT_MODEL_SET="1"
          ;;
        *)
          echo "--world-nav-robot-model expects capsule or unitree-go2, got: $2" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --world-nav-headless)
      WORLD_NAV_VIEWER="0"
      shift
      ;;
    --world-nav-viewer)
      WORLD_NAV_VIEWER="1"
      shift
      ;;
    --scene-usd)
      SCENE_USD="$2"
      SCENE_USD_AUTO="0"
      shift 2
      ;;
    --use-built-in-scene)
      SCENE_USD=""
      SCENE_USD_AUTO="0"
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
    --hifi-base-config)
      HIFI_BASE_CONFIG="$2"
      shift 2
      ;;
    --hifi-view-size)
      if [[ "$2" != *x* ]]; then
        echo "--hifi-view-size expects WxH, got: $2" >&2
        exit 2
      fi
      HIFI_VIEW_WIDTH="${2%x*}"
      HIFI_VIEW_HEIGHT="${2#*x}"
      shift 2
      ;;
    --live-gaussian-patch-mode)
      WEB_LIVE_GAUSSIAN_PATCH_MODE="$2"
      shift 2
      ;;
    --gaussian-chunks)
      WEB_GAUSSIAN_PREPROCESS="chunks"
      shift
      ;;
    --no-gaussian-chunks)
      WEB_GAUSSIAN_PREPROCESS="disabled"
      shift
      ;;
    --gaussian-chunk-grid)
      WEB_GAUSSIAN_CHUNK_GRID="$2"
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
      NO_ISAAC_SET="1"
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

if [[ "$NAV_MODE" == "1" ]]; then
  START_WORLD_NAV="1"
  ISAAC_TRAINING_TRAJECTORY="0"
  if [[ "$WORLD_NAV_BACKEND" == "nav2" && "$NO_ISAAC_SET" != "1" ]]; then
    START_ISAAC="1"
    ISAAC_CONTROL_SOURCE="ros2-cmd_vel"
  else
    START_ISAAC="0"
  fi
  if [[ "$WEB_MODE_SET" != "1" ]]; then
    WEB_MODE="gs"
  fi
  if [[ "$WORLD_NAV_ROBOT_MODEL_SET" != "1" ]]; then
    if [[ "$WORLD_NAV_BACKEND" == "nav2" ]]; then
      WORLD_NAV_ROBOT_MODEL="unitree-go2"
    else
      WORLD_NAV_ROBOT_MODEL="capsule"
    fi
  fi
fi

is_local_target() {
  [[ "$REMOTE" == "local" || "$REMOTE" == "localhost" || "$REMOTE" == "127.0.0.1" || "$REMOTE" == "." ]]
}

if is_local_target; then
  echo "Launch target: local"
else
  echo "Launch target: remote SSH $REMOTE"
fi
echo "Stack root:    $REMOTE_ROOT"
if [[ "$NAV_MODE" == "1" ]]; then
  echo "Mode:          nav baseline"
  echo "Nav backend:   $WORLD_NAV_BACKEND"
  echo "Nav robot:     $WORLD_NAV_ROBOT_MODEL"
  echo "Isaac control: $ISAAC_CONTROL_SOURCE"
fi
echo "WebUI:         http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
if [[ "$START_HIFI" == "1" ]]; then
  echo "HiFi GS:       ${RENDERER_URL:-http://127.0.0.1:${HIFI_PORT}}"
  echo "HiFi output:   $HIFI_OUTPUT_DIR"
fi
if [[ -n "$SCENE_USD" ]]; then
  echo "Isaac scene:   $SCENE_USD"
elif [[ "$SCENE_USD_AUTO" == "1" ]]; then
  echo "Isaac scene:   auto-generated from offline mesh"
else
  echo "Isaac scene:   built-in ${ISAAC_SCENE}"
fi

if is_local_target; then
  REMOTE_RUNNER=(env)
else
  REMOTE_RUNNER=(ssh "$REMOTE" env)
fi

"${REMOTE_RUNNER[@]}" \
  REMOTE_ROOT="$REMOTE_ROOT" \
  WEB_PORT="$WEB_PORT" \
  BRIDGE_PORT="$BRIDGE_PORT" \
  MAPPER_PORT="$MAPPER_PORT" \
  WORLD_NAV_PORT="$WORLD_NAV_PORT" \
  HIFI_PORT="$HIFI_PORT" \
  WEB_SCENE="$WEB_SCENE" \
  WEB_VISUAL_SCENE="$WEB_VISUAL_SCENE" \
  WEB_MODE="$WEB_MODE" \
  START_HIFI="$START_HIFI" \
  START_ISAAC="$START_ISAAC" \
  START_WORLD_NAV="$START_WORLD_NAV" \
  NAV_MODE="$NAV_MODE" \
  WORLD_NAV_BACKEND="$WORLD_NAV_BACKEND" \
  RESTART="$RESTART" \
  STOP="$STOP" \
  ISAAC_SCENE="$ISAAC_SCENE" \
  SCENE_USD="$SCENE_USD" \
  SCENE_USD_AUTO="$SCENE_USD_AUTO" \
  ISAAC_DEVICE="$ISAAC_DEVICE" \
  ISAAC_CONTROL_SOURCE="$ISAAC_CONTROL_SOURCE" \
  ISAAC_BASE_HEIGHT="$ISAAC_BASE_HEIGHT" \
  ISAAC_PATH_RADIUS="$ISAAC_PATH_RADIUS" \
  ISAAC_PATH_SPEED="$ISAAC_PATH_SPEED" \
  ISAAC_HOLD_OPEN_SEC="$ISAAC_HOLD_OPEN_SEC" \
  ISAAC_TRAINING_TRAJECTORY="$ISAAC_TRAINING_TRAJECTORY" \
  ISAAC_TRAINING_TRAJECTORY_SPLIT="$ISAAC_TRAINING_TRAJECTORY_SPLIT" \
  ISAAC_TRAINING_RGB_SOURCE="$ISAAC_TRAINING_RGB_SOURCE" \
  ISAAC_ENV="$ISAAC_ENV" \
  DISPLAY_VALUE="$DISPLAY_VALUE" \
  XAUTHORITY_VALUE="$XAUTHORITY_VALUE" \
  RENDERER_URL="$RENDERER_URL" \
  HIFI_OUTPUT_DIR="$HIFI_OUTPUT_DIR" \
  HIFI_VIEW_CONTAINER="$HIFI_VIEW_CONTAINER" \
  HIFI_BRIDGE_CONTAINER="$HIFI_BRIDGE_CONTAINER" \
  HIFI_BASE_CONFIG="$HIFI_BASE_CONFIG" \
  HIFI_VIEW_WIDTH="$HIFI_VIEW_WIDTH" \
  HIFI_VIEW_HEIGHT="$HIFI_VIEW_HEIGHT" \
  HIFI_RENDER_POSE_MAX_DISTANCE="$HIFI_RENDER_POSE_MAX_DISTANCE" \
  WEB_LIVE_GAUSSIAN_PATCH_MODE="$WEB_LIVE_GAUSSIAN_PATCH_MODE" \
  WEB_GAUSSIAN_PREPROCESS="$WEB_GAUSSIAN_PREPROCESS" \
  WEB_GAUSSIAN_CHUNK_GRID="$WEB_GAUSSIAN_CHUNK_GRID" \
  WEB_GAUSSIAN_CHUNK_MAX_SH="$WEB_GAUSSIAN_CHUNK_MAX_SH" \
  WEB_GAUSSIAN_CHUNK_OPACITY_THRESHOLD="$WEB_GAUSSIAN_CHUNK_OPACITY_THRESHOLD" \
  WORLD_NAV_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT" \
  WORLD_NAV_OUTPUT_DIR="$WORLD_NAV_OUTPUT_DIR" \
  WORLD_NAV_ROBOT_MODEL="$WORLD_NAV_ROBOT_MODEL" \
  NAV_MAP_INFLATE_RADIUS="$NAV_MAP_INFLATE_RADIUS" \
  WORLD_NAV_VIEWER="$WORLD_NAV_VIEWER" \
  WORLD_NAV_HOLD_OPEN_SEC="$WORLD_NAV_HOLD_OPEN_SEC" \
  WORLD_NAV_EPISODE_HOLD_SEC="$WORLD_NAV_EPISODE_HOLD_SEC" \
  WORLD_NAV_STEP_SLEEP="$WORLD_NAV_STEP_SLEEP" \
  WORLD_NAV_TIMEOUT_SEC="$WORLD_NAV_TIMEOUT_SEC" \
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
    pids="$(port_pids "$port")"
    if [[ -n "$pids" ]]; then
      echo "Force stopping remote listener(s) on port $port: $pids"
      kill -9 $pids >/dev/null 2>&1 || true
      sleep 1
    fi
  fi
}

stop_isaac_publisher_for_bridge() {
  local pattern="run_isaac_gaussian_online_demo.py.*--bridge-url http://127.0.0.1:${BRIDGE_PORT}"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "Stopping existing Isaac bridge publisher for :$BRIDGE_PORT"
    pkill -f "$pattern" >/dev/null 2>&1 || true
    sleep 1
  fi
}

stop_nav2_baseline_stack() {
  local patterns=(
    "launch_nav2_baseline.sh ${WORLD_NAV_PORT}"
    "nav2_baseline_control_server.py --port ${WORLD_NAV_PORT}"
    "nav2_pgm_robot_bridge.py"
    "nav2_baseline_launch.py --params-file .*one-click-isaac-webui.*/nav2_params.yaml"
    "/opt/ros/humble/lib/nav2_.*--params-file .*one-click-isaac-webui.*/nav2_params.yaml"
    "lifecycle_manager_nav2_baseline"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "Stopping Nav2 baseline process(es): $pattern"
      pkill -f "$pattern" >/dev/null 2>&1 || true
    fi
  done
  sleep 1
}

ensure_visual_gaussian_chunks() {
  if [[ "$WEB_GAUSSIAN_PREPROCESS" != "chunks" ]]; then
    return 0
  fi
  if [[ -z "$WEB_VISUAL_SCENE" ]]; then
    return 0
  fi

  local gs_ply="$HIFI_OUTPUT_DIR/model/gs.ply"
  if [[ ! -f "$gs_ply" ]]; then
    echo "Skipping Gaussian chunk preprocessing; gs.ply not found: $gs_ply" >&2
    return 0
  fi

  local run_name processed_root chunks_meta
  run_name="$(basename "$HIFI_OUTPUT_DIR")"
  processed_root="$REMOTE_ROOT/runtime/processed/$run_name/model"
  chunks_meta="$processed_root/gs_chunks.json"
  if [[ -f "$chunks_meta" && "$chunks_meta" -nt "$gs_ply" && "$chunks_meta" -nt "$REMOTE_ROOT/scripts/preprocess_gaussian_stream.mjs" ]]; then
    echo "Gaussian chunks already prepared: $chunks_meta"
    return 0
  fi

  if ! command -v node >/dev/null 2>&1; then
    echo "Node.js is unavailable; WebUI will fall back to raw gs.ply." >&2
    return 0
  fi
  if [[ ! -d "$REMOTE_ROOT/examples/web-ui/node_modules/@sparkjsdev/spark" ]]; then
    echo "Spark node module is unavailable; WebUI will fall back to raw gs.ply." >&2
    return 0
  fi

  echo "Preprocessing Gaussian for chunked browser streaming: grid=$WEB_GAUSSIAN_CHUNK_GRID maxSh=$WEB_GAUSSIAN_CHUNK_MAX_SH"
  node "$REMOTE_ROOT/scripts/preprocess_gaussian_stream.mjs" \
    --output-dir "$HIFI_OUTPUT_DIR" \
    --grid "$WEB_GAUSSIAN_CHUNK_GRID" \
    --max-sh "$WEB_GAUSSIAN_CHUNK_MAX_SH" \
    --opacity-threshold "$WEB_GAUSSIAN_CHUNK_OPACITY_THRESHOLD" \
    --force
}

write_offline_gs_scene_manifest() {
  local target_dir="$REMOTE_ROOT/examples/web-ui/public/scenes/isaac-gaussian-online-offline-gs"
  local source_manifest="$REMOTE_ROOT/examples/web-ui/public/scenes/isaac-gaussian-online/manifest.json"
  local target_manifest="$target_dir/manifest.json"

  if [[ ! "$WEB_SCENE" =~ ^/scenes/isaac-gaussian-online-offline-gs/manifest\.json$ ]]; then
    return 0
  fi

  mkdir -p "$target_dir"
  python3 - \
    "$source_manifest" \
    "$target_manifest" \
    "$WEB_LIVE_GAUSSIAN_PATCH_MODE" \
    "$WEB_VISUAL_SCENE" \
    "$HIFI_OUTPUT_DIR" \
    "$REMOTE_ROOT/examples/web-ui/public" \
    "${WORLD_NAV_VALIDATION_REPORT:-}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
patch_mode = sys.argv[3]
visual_scene = sys.argv[4]
hifi_output_dir = sys.argv[5]
public_root = Path(sys.argv[6])
world_nav_report = Path(sys.argv[7]) if sys.argv[7] else None

manifest = json.loads(source_path.read_text())
manifest["sceneId"] = "isaac-gaussian-online-offline-gs"
source = dict(manifest.get("source") or {})
source["liveGaussianPatchMode"] = patch_mode
if visual_scene:
    source["visualManifestUrl"] = visual_scene
manifest["source"] = source
training = dict(manifest.get("training") or {})
training["visualOutputDir"] = hifi_output_dir
manifest["training"] = training
if visual_scene.startswith("/"):
    visual_manifest_path = public_root / visual_scene.lstrip("/")
else:
    visual_manifest_path = None
if visual_manifest_path and visual_manifest_path.is_file():
    visual_manifest = json.loads(visual_manifest_path.read_text())
    for key in (
        "camera",
        "initialView",
        "gaussian",
        "gaussianVariants",
        "mesh",
        "rawPointCloud",
        "occupancy",
        "robot",
        "meta",
    ):
        if key not in manifest and key in visual_manifest:
            manifest[key] = visual_manifest[key]
    manifest["assets"] = {
        **(visual_manifest.get("assets") or {}),
        **(manifest.get("assets") or {}),
    }


def read_map_yaml(path):
    data = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    origin = [float(v.strip()) for v in data["origin"].strip("[]").split(",")[:3]]
    image = data.get("image", path.with_suffix(".pgm").name).strip('"')
    return {
        "image": image,
        "resolution": float(data["resolution"]),
        "origin": origin,
    }


def read_pgm_size(path):
    with path.open("rb") as handle:
        if handle.readline().strip() not in (b"P5", b"P2"):
            return None
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = [int(v) for v in line.split()[:2]]
    return width, height


if world_nav_report and world_nav_report.is_file():
    report = json.loads(world_nav_report.read_text())
    map_yaml = Path((report.get("inputs") or {}).get("map_yaml") or "")
    if map_yaml.is_file():
        map_info = read_map_yaml(map_yaml)
        map_pgm = (map_yaml.parent / map_info["image"]).resolve()
        if map_pgm.is_file():
            target_dir = target_path.parent
            target_pgm = target_dir / "nav_map.pgm"
            target_yaml = target_dir / "nav_map.yaml"
            shutil.copy2(map_pgm, target_pgm)
            target_yaml.write_text(
                "\n".join(
                    [
                        "image: nav_map.pgm",
                        "mode: trinary",
                        f"resolution: {map_info['resolution']:.6f}",
                        "origin: "
                        + json.dumps(
                            [
                                map_info["origin"][0],
                                map_info["origin"][1],
                                map_info["origin"][2],
                            ]
                        ),
                        "negate: 0",
                        "occupied_thresh: 0.65",
                        "free_thresh: 0.196",
                        "",
                    ]
                )
            )
            scene_base = "/scenes/isaac-gaussian-online-offline-gs"
            manifest["occupancy"] = {
                "source": "prebuilt_grid",
                "url": f"{scene_base}/nav_map.pgm",
                "resolution": map_info["resolution"],
                "origin": {
                    "x": map_info["origin"][0],
                    "y": map_info["origin"][1],
                },
            }
            meta = dict(manifest.get("meta") or {})
            meta["leafSize"] = map_info["resolution"]
            meta["mapOrigin"] = {
                "x": map_info["origin"][0],
                "y": map_info["origin"][1],
                "z": map_info["origin"][2],
            }
            pgm_size = read_pgm_size(target_pgm)
            if pgm_size:
                meta["mapSize"] = max(pgm_size) * map_info["resolution"]
            manifest["meta"] = meta
            assets = dict(manifest.get("assets") or {})
            assets["occupancy"] = "ready"
            manifest["assets"] = assets
            robot = dict(manifest.get("robot") or {})
            robot["radius"] = max(float(robot.get("radius") or 0.0), 0.32)
            robot.setdefault("height", 0.45)
            manifest["robot"] = robot
            source = dict(manifest.get("source") or {})
            source["navMapUrl"] = f"{scene_base}/nav_map.yaml"
            manifest["source"] = source
notes = manifest.get("externalViewer", {}).get("notes")
if isinstance(notes, str):
    manifest["externalViewer"]["notes"] = (
        notes
        + " This generated launch scene keeps the offline trained Gaussian visible "
        + f"with liveGaussianPatchMode={patch_mode!r} and visualManifestUrl={visual_scene!r}."
    )
target_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY
  echo "Generated WebUI scene manifest: $target_manifest"
}

ensure_visual_scene_pack() {
  if [[ -z "$WEB_VISUAL_SCENE" ]]; then
    return 0
  fi

  if [[ ! "$WEB_VISUAL_SCENE" =~ ^/scenes/([^/]+)/manifest\.json$ ]]; then
    echo "Skipping visual scene pack sync for non-standard visual scene URL: $WEB_VISUAL_SCENE"
    return 0
  fi

  local scene_id="${BASH_REMATCH[1]}"
  local manifest_path="$REMOTE_ROOT/examples/web-ui/public/scenes/${scene_id}/manifest.json"
  if [[ -f "$manifest_path" ]]; then
    if python3 - "$manifest_path" "$HIFI_OUTPUT_DIR" "$REMOTE_ROOT" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
expected_output = Path(sys.argv[2]).resolve()
repo_root = Path(sys.argv[3]).resolve()
try:
    manifest = json.loads(manifest_path.read_text())
    current_output = Path(manifest.get("training", {}).get("outputDir") or "").resolve()
except Exception:
    raise SystemExit(1)

if current_output != expected_output:
    raise SystemExit(1)

processed_root = repo_root / "runtime" / "processed" / expected_output.name / "model"
chunks_meta = processed_root / "gs_chunks.json"
gaussian = manifest.get("gaussian") or {}
if chunks_meta.is_file():
    if gaussian.get("format") != "gs-chunks":
        raise SystemExit(1)
    if chunks_meta.stat().st_mtime > manifest_path.stat().st_mtime:
        raise SystemExit(1)

raise SystemExit(0)
PY
    then
      echo "Visual scene pack already exists: $manifest_path"
      return 0
    fi
    echo "Visual scene pack is stale for current HiFi output; rebuilding: $manifest_path"
  fi

  if [[ ! -d "$HIFI_OUTPUT_DIR" ]]; then
    echo "Cannot sync visual scene pack; output dir does not exist: $HIFI_OUTPUT_DIR" >&2
    exit 1
  fi

  echo "Syncing offline visual scene pack: $scene_id"
  python3 "$REMOTE_ROOT/scripts/sync_scene_pack.py" \
    --output-dir "$HIFI_OUTPUT_DIR" \
    --scene-root "$REMOTE_ROOT/examples/web-ui/public/scenes" \
    --scene-id "$scene_id" \
    --status completed
}

fix_hifi_scene_config() {
  local config_path="$HIFI_OUTPUT_DIR/model/config/scene/config.yaml"
  if [[ ! -f "$config_path" ]]; then
    echo "HiFi scene config not found: $config_path" >&2
    exit 1
  fi

  python3 - "$config_path" "$HIFI_BASE_CONFIG" <<'PY'
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
base_config = sys.argv[2]
text = config_path.read_text()
original_text = text
match = re.search(r'^(base_config:\s*")([^"]+)(")', text, flags=re.MULTILINE)
if not match:
    print(f"base_config not found in {config_path}", file=sys.stderr)
    sys.exit(1)

current = match.group(2)
messages = []
if current != base_config:
    text = text[: match.start(2)] + base_config + text[match.end(2) :]
    messages.append(f"Updated HiFi scene base_config: {current} -> {base_config}")
else:
    messages.append(f"HiFi scene base_config already OK: {base_config}")

if "map_origin:" not in text:
    map_match = re.search(r"^map:\s*$", text, flags=re.MULTILINE)
    if not map_match:
        print(f"map section not found in {config_path}", file=sys.stderr)
        sys.exit(1)
    insert = (
        "\n"
        "   map_origin: !!opencv-matrix\n"
        "      rows: 1\n"
        "      cols: 3\n"
        "      dt: d\n"
        "      data: [ 0, 0, 0 ]\n"
    )
    text = text[: map_match.end()] + insert + text[map_match.end() :]
    messages.append("Inserted missing map.map_origin: [0, 0, 0]")

if text != original_text:
    backup_path = config_path.with_suffix(config_path.suffix + ".bak_launch_fix")
    if not backup_path.exists():
        backup_path.write_text(original_text)
    config_path.write_text(text)
    messages.append(f"Backup: {backup_path}")

for message in messages:
    print(message)
PY
}

ensure_hifi_dataset_layout() {
  local config_path="$HIFI_OUTPUT_DIR/model/config/scene/config.yaml"
  local data_path
  data_path="$(python3 - "$config_path" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r'^data_path:\s*"([^"]+)"', text, flags=re.MULTILINE)
print(match.group(1) if match else "")
PY
)"

  if [[ -z "$data_path" || ! -d "$data_path" ]]; then
    echo "HiFi data_path is missing or not a directory: ${data_path:-<unset>}" >&2
    exit 1
  fi

  mkdir -p "$data_path/colmap"
  for entry in images depths sparse; do
    if [[ -e "$data_path/$entry" && ! -e "$data_path/colmap/$entry" ]]; then
      ln -s "../$entry" "$data_path/colmap/$entry"
      echo "Created GS-SDF view compatibility link: $data_path/colmap/$entry -> ../$entry"
    fi
  done

  mkdir -p "$data_path/colmap/postrior_lidar"
  for file in cameras.txt images.txt depths.txt; do
    if [[ "$file" != "images.txt" && "$file" != "depths.txt" && -f "$data_path/sparse/0/$file" && ! -e "$data_path/colmap/postrior_lidar/$file" ]]; then
      ln -s "../../sparse/0/$file" "$data_path/colmap/postrior_lidar/$file"
      echo "Created GS-SDF view compatibility link: $data_path/colmap/postrior_lidar/$file -> ../../sparse/0/$file"
    fi
  done

  if [[ -f "$data_path/sparse/0/images.txt" ]]; then
    python3 - \
      "$data_path/sparse/0/images.txt" \
      "$data_path/colmap/postrior_lidar/images.txt" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
rows = []
for line in source.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    parts = stripped.split()
    if len(parts) >= 10:
        rows.append(" ".join(parts))

content = "\n\n".join(rows) + ("\n\n" if rows else "")
if target.is_symlink() or target.exists():
    previous = target.read_text() if target.is_file() and not target.is_symlink() else ""
    if target.is_symlink() or previous != content:
        target.unlink()
target.write_text(content)
print(f"Wrote GS-SDF view color pose file: {target} ({len(rows)} poses)")
PY
  fi

  if [[ -f "$data_path/sparse/0/depths.txt" ]]; then
    python3 - \
      "$data_path/sparse/0/depths.txt" \
      "$data_path/depths/lidar_pose.txt" \
      "$data_path/colmap/postrior_lidar/depths.txt" <<'PY'
import math
import sys
from pathlib import Path

source = Path(sys.argv[1])
targets = [Path(arg) for arg in sys.argv[2:]]
rows = []

def quat_wxyz_to_rot(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("invalid zero quaternion")
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)),
        (2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)),
        (2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )

def rot_to_quat_wxyz(rot):
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2][1] - rot[1][2]) / scale
        qy = (rot[0][2] - rot[2][0]) / scale
        qz = (rot[1][0] - rot[0][1]) / scale
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
        qw = (rot[2][1] - rot[1][2]) / scale
        qx = 0.25 * scale
        qy = (rot[0][1] + rot[1][0]) / scale
        qz = (rot[0][2] + rot[2][0]) / scale
    elif rot[1][1] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
        qw = (rot[0][2] - rot[2][0]) / scale
        qx = (rot[0][1] + rot[1][0]) / scale
        qy = 0.25 * scale
        qz = (rot[1][2] + rot[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
        qw = (rot[1][0] - rot[0][1]) / scale
        qx = (rot[0][2] + rot[2][0]) / scale
        qy = (rot[1][2] + rot[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm

def invert_colmap_w2c_to_c2w(parts):
    item_id = int(parts[0])
    qw, qx, qy, qz = (float(value) for value in parts[1:5])
    tx, ty, tz = (float(value) for value in parts[5:8])
    rot_w2c = quat_wxyz_to_rot(qw, qx, qy, qz)
    rot_c2w = tuple(tuple(rot_w2c[col][row] for col in range(3)) for row in range(3))
    c2w_t = tuple(
        -sum(rot_c2w[row][col] * value for col, value in enumerate((tx, ty, tz)))
        for row in range(3)
    )
    values = (item_id, *rot_to_quat_wxyz(rot_c2w), *c2w_t)
    return " ".join(f"{value:.17g}" if isinstance(value, float) else str(value) for value in values)

for line in source.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    parts = stripped.split()
    if len(parts) >= 8:
        # GS-SDF Colmap depth poses use pose_type=5 and are not inverted by the
        # parser. Convert COLMAP world-to-camera rows to camera-to-world.
        rows.append(invert_colmap_w2c_to_c2w(parts))

content = "\n".join(rows) + ("\n" if rows else "")
for target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        previous = target.read_text() if target.is_file() and not target.is_symlink() else ""
        if target.is_symlink() or previous != content:
            target.unlink()
    target.write_text(content)
    print(f"Wrote GS-SDF view depth pose file: {target} ({len(rows)} poses)")
PY
  fi

  if ! compgen -G "$data_path/depths/*.pcd" >/dev/null && compgen -G "$data_path/depths/*.ply" >/dev/null; then
    echo "Converting NuRec depth PLY files to PCD for GS-SDF view mode..."
    python3 - "$data_path/depths" <<'PY'
import sys
from pathlib import Path

depth_dir = Path(sys.argv[1])

def read_ascii_ply_xyz(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        vertex_count = None
        for line in f:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")
        points = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return points

def write_ascii_pcd(path: Path, points):
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")
        for x, y, z in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")

converted = 0
for ply_path in sorted(depth_dir.glob("*.ply")):
    pcd_path = ply_path.with_suffix(".pcd")
    if pcd_path.exists():
        continue
    points = read_ascii_ply_xyz(ply_path)
    write_ascii_pcd(pcd_path, points)
    converted += 1
print(f"Converted {converted} depth PLY files to PCD in {depth_dir}")
PY
  fi
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

  fix_hifi_scene_config
  ensure_hifi_dataset_layout

  local hifi_cmd
  hifi_cmd="$(quote_cmd env \
    VIEW_WIDTH="$HIFI_VIEW_WIDTH" \
    VIEW_HEIGHT="$HIFI_VIEW_HEIGHT" \
    KEEP_CONTAINER_ON_EXIT=1 \
    WAIT_SECONDS=120 \
    bash "$REMOTE_ROOT/scripts/launch_gssdf_hifi_stack.sh" \
      "$HIFI_OUTPUT_DIR" \
      "$HIFI_VIEW_CONTAINER" \
      "$HIFI_PORT" \
      "$HIFI_BRIDGE_CONTAINER")"
  start_service "hifi-renderer" "$HIFI_PORT" "$hifi_cmd"
  wait_http "hifi-renderer" "$RENDERER_URL/status" 1 180
}

prime_hifi_training_pose() {
  if [[ "$START_HIFI" != "1" || -z "$RENDERER_URL" ]]; then
    return 0
  fi

  local pose_log="$LOG_DIR/hifi-training-camera-pose.json"
  echo "Priming HiFi renderer with GS-SDF training camera[0]..."
  local attempt
  for attempt in $(seq 1 10); do
    if python3 "$REMOTE_ROOT/scripts/gssdf_training_camera_pose.py" \
        --output-dir "$HIFI_OUTPUT_DIR" \
        --split train \
        --frame-index 0 \
        --post-url "http://127.0.0.1:${BRIDGE_PORT}" \
        --repeat 2 \
        >"${pose_log}.tmp"; then
      mv "${pose_log}.tmp" "$pose_log"
      if python3 - "$pose_log" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
status = payload.get("postStatus") or {}
ok = bool(status.get("rendererReady")) and status.get("displaySource") == "renderer"
raise SystemExit(0 if ok else 1)
PY
      then
        python3 - "$pose_log" <<'PY' || true
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
pose = payload.get("pose", {})
position = pose.get("position", {})
print(
    "HiFi training initial pose: "
    f"image={payload.get('sourceImage')} "
    f"position=({position.get('x'):.3f}, {position.get('y'):.3f}, {position.get('z'):.3f})"
)
PY
        return 0
      fi
    fi
    sleep 1
  done
  rm -f "${pose_log}.tmp"
  echo "Warning: could not prime HiFi training pose with a fresh renderer frame. See $pose_log" >&2
}

ensure_auto_isaac_scene_usd() {
  if [[ "$SCENE_USD_AUTO" != "1" || -n "$SCENE_USD" ]]; then
    return 0
  fi
  if [[ "$START_ISAAC" != "1" && "$START_WORLD_NAV" != "1" ]]; then
    return 0
  fi

  if [[ ! -d "$HIFI_OUTPUT_DIR" ]]; then
    echo "Auto Isaac USD skipped because HiFi output dir is missing: $HIFI_OUTPUT_DIR"
    return 0
  fi

  local mesh_path
  mesh_path="$(python3 - "$HIFI_OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
patterns = ("mesh*.ply", "model/mesh*.ply", "model/gs_*.ply")
for pattern in patterns:
    matches = sorted(root.glob(pattern))
    if matches:
        print(matches[0].resolve())
        break
PY
)"
  if [[ -z "$mesh_path" || ! -f "$mesh_path" ]]; then
    echo "Auto Isaac USD skipped because no offline mesh was found under $HIFI_OUTPUT_DIR"
    return 0
  fi

  local gaussian_ply="$HIFI_OUTPUT_DIR/model/gs.ply"

  local scene_slug
  scene_slug="$(python3 - "$HIFI_OUTPUT_DIR" <<'PY'
import re
import sys
from pathlib import Path

name = Path(sys.argv[1]).name
slug = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "scene"
print(slug[:180])
PY
)"
  local out_dir="$REMOTE_ROOT/runtime/isaac_assets"
  local out_usd="$out_dir/${scene_slug}_mesh_visual_collision.usd"
  mkdir -p "$out_dir"

  if [[ -f "$out_usd" && "$out_usd" -nt "$mesh_path" && "$out_usd" -nt "$REMOTE_ROOT/scripts/isaac_nav_ply_to_usd_collision.py" && ( ! -f "$gaussian_ply" || "$out_usd" -nt "$gaussian_ply" ) ]]; then
    SCENE_USD="$out_usd"
    echo "Using cached auto-generated Isaac USD: $SCENE_USD"
    return 0
  fi

  local build_cmd
  local converter_args=("$REMOTE_ROOT/scripts/isaac_nav_ply_to_usd_collision.py" --input "$mesh_path" --output "$out_usd" --collision-approximation meshSimplification)
  if [[ -f "$gaussian_ply" ]]; then
    converter_args+=(--gaussian-ply "$gaussian_ply")
  fi
  build_cmd="source $(printf '%q' "$ISAAC_ENV") && export OMNI_KIT_ACCEPT_EULA=YES DISPLAY=$(printf '%q' "$DISPLAY_VALUE") XAUTHORITY=$(printf '%q' "$XAUTHORITY_VALUE") QT_X11_NO_MITSHM=1 && \"\$ISAACSIM_PYTHON_EXE\" $(quote_cmd "${converter_args[@]}")"
  echo "Generating Isaac visual+collision USD from $mesh_path"
  bash -lc "$build_cmd"
  SCENE_USD="$out_usd"
  echo "Generated auto Isaac USD: $SCENE_USD"
}

ensure_world_nav_report() {
  if [[ "$START_WORLD_NAV" != "1" ]]; then
    return 0
  fi
  if [[ -n "$WORLD_NAV_VALIDATION_REPORT" && -f "$WORLD_NAV_VALIDATION_REPORT" ]]; then
    return 0
  fi
  if [[ -z "$SCENE_USD" || ! -f "$SCENE_USD" ]]; then
    echo "Skipping current-scene world-nav report; Isaac USD is not available: $SCENE_USD" >&2
    return 0
  fi

  local mesh_path="$HIFI_OUTPUT_DIR/mesh_gs_.ply"
  if [[ ! -f "$mesh_path" ]]; then
    echo "Skipping current-scene world-nav report; mesh is not available: $mesh_path" >&2
    return 0
  fi

  local slug
  slug="$(basename "$HIFI_OUTPUT_DIR" | tr -c 'A-Za-z0-9_-' '_' | sed 's/_*$//')"
  local map_dir="/media/chatsign/data-002/isaac/nav-mvp/maps"
  local map_prefix="$map_dir/${slug}_auto_nav"
  local report="$WORLD_NAV_OUTPUT_DIR/${slug}_minimal_agent_validation.json"
  local nav_map_inflate_radius="$NAV_MAP_INFLATE_RADIUS"
  if [[ -z "$nav_map_inflate_radius" ]]; then
    if [[ "$WORLD_NAV_BACKEND" == "nav2" ]]; then
      nav_map_inflate_radius="0.0"
    else
      nav_map_inflate_radius="0.28"
    fi
  fi
  mkdir -p "$map_dir" "$WORLD_NAV_OUTPUT_DIR"

  local nurec_map_yaml
  nurec_map_yaml="$(python3 - "$HIFI_OUTPUT_DIR/model/config/scene/config.yaml" <<'PY'
import json
import re
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit("")
match = re.search(r'^data_path:\s*"([^"]+)"', config_path.read_text(), flags=re.MULTILINE)
if not match:
    raise SystemExit("")
data_path = Path(match.group(1))
meta_path = data_path / "nurec_to_gssdf_meta.json"
candidates = []
if meta_path.is_file():
    try:
        input_root = Path(json.loads(meta_path.read_text()).get("inputRoot", ""))
        candidates.extend([input_root / "occupancy_map.yaml", input_root / "3dgrt" / "omap" / "occupancy_map.yaml"])
    except Exception:
        pass
candidates.extend([data_path / "occupancy_map.yaml", data_path / "3dgrt" / "omap" / "occupancy_map.yaml"])
for candidate in candidates:
    if candidate.is_file():
        print(candidate)
        break
PY
)"
  if [[ -n "$nurec_map_yaml" && -f "$nurec_map_yaml" ]]; then
    echo "Preparing NuRec nav map from source occupancy: $nurec_map_yaml"
    python3 "$REMOTE_ROOT/scripts/prepare_nurec_nav_map.py" \
      --source-yaml "$nurec_map_yaml" \
      --output-prefix "$map_prefix" \
      --target-resolution 0.05 \
      --inflate-radius "$nav_map_inflate_radius" >/dev/null
  else
    echo "NuRec source occupancy not found; falling back to mesh-projected nav map." >&2
    python3 "$REMOTE_ROOT/scripts/build_isaac_nav_occupancy_map.py" \
      --input "$mesh_path" \
      --output-prefix "$map_prefix" \
      --resolution 0.15 \
      --z-min 0.05 \
      --z-max 1.8 \
      --inflate-radius "$nav_map_inflate_radius" >/dev/null
  fi

  python3 - \
    "$map_prefix.yaml" \
    "$map_prefix.pgm" \
    "$SCENE_USD" \
    "$mesh_path" \
    "$report" \
    "$HIFI_OUTPUT_DIR/model/config/scene/config.yaml" <<'PY'
import collections
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

map_yaml = Path(sys.argv[1])
map_pgm = Path(sys.argv[2])
collision_usd = Path(sys.argv[3])
mesh_ply = Path(sys.argv[4])
report_path = Path(sys.argv[5])
scene_config = Path(sys.argv[6])


def read_map_yaml(path):
    data = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    origin = [float(v) for v in data["origin"].strip("[]").split(",")[:3]]
    return float(data["resolution"]), origin


def read_pgm(path):
    with path.open("rb") as handle:
        if handle.readline().strip() != b"P5":
            raise RuntimeError(f"Unsupported PGM: {path}")
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = [int(v) for v in line.split()]
        max_value = int(handle.readline())
        image = np.frombuffer(handle.read(width * height), dtype=np.uint8).reshape((height, width))
    if max_value != 255:
        image = (image.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    return image


def quat_wxyz_to_rot(q):
    qw, qx, qy, qz = q
    norm = math.sqrt(sum(v * v for v in q))
    qw, qx, qy, qz = (qw / norm, qx / norm, qy / norm, qz / norm)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def first_training_xy(config_path):
    if not config_path.is_file():
        return None
    match = re.search(r'^data_path:\s*"([^"]+)"', config_path.read_text(), flags=re.MULTILINE)
    if not match:
        return None
    data_path = Path(match.group(1))
    images_txt = data_path / "colmap" / "postrior_lidar" / "images.txt"
    if not images_txt.is_file():
        images_txt = data_path / "sparse" / "0" / "images.txt"
    if not images_txt.is_file():
        return None
    for line in images_txt.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            continue
        q = [float(v) for v in parts[1:5]]
        t = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
        rot = quat_wxyz_to_rot(q)
        center = -rot.T @ t
        return float(center[0]), float(center[1])
    return None


resolution, origin = read_map_yaml(map_yaml)
image = read_pgm(map_pgm)
height, width = image.shape
free = image > 200
free_count = int(free.sum())
if free_count == 0:
    raise RuntimeError(f"No free cells in map: {map_pgm}")


def cell_to_world(cell):
    x, y = cell
    return (
        origin[0] + (x + 0.5) * resolution,
        origin[1] + ((height - 1 - y) + 0.5) * resolution,
    )


def world_to_cell(x, y):
    cell_x = int(math.floor((x - origin[0]) / resolution))
    world_row = int(math.floor((y - origin[1]) / resolution))
    cell_y = (height - 1) - world_row
    return cell_x, cell_y


def nearest_free_cell(xy):
    if xy is not None:
        x, y = world_to_cell(*xy)
        if 0 <= x < width and 0 <= y < height and free[y, x]:
            return x, y
        ys, xs = np.nonzero(free)
        wx = origin[0] + (xs + 0.5) * resolution
        wy = origin[1] + ((height - 1 - ys) + 0.5) * resolution
        idx = int(np.argmin((wx - xy[0]) ** 2 + (wy - xy[1]) ** 2))
        return int(xs[idx]), int(ys[idx])
    ys, xs = np.nonzero(free)
    idx = len(xs) // 2
    return int(xs[idx]), int(ys[idx])


start_cell = nearest_free_cell(first_training_xy(scene_config))
queue = collections.deque([start_cell])
seen = {start_cell}
while queue:
    x, y = queue.popleft()
    for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
        if 0 <= nx < width and 0 <= ny < height and free[ny, nx] and (nx, ny) not in seen:
            seen.add((nx, ny))
            queue.append((nx, ny))

sx, sy = cell_to_world(start_cell)
component = sorted(
    seen,
    key=lambda cell: (cell_to_world(cell)[0] - sx) ** 2 + (cell_to_world(cell)[1] - sy) ** 2,
)
goals = []
for cell in component:
    gx, gy = cell_to_world(cell)
    if math.hypot(gx - sx, gy - sy) >= 0.6:
        goals.append([gx, gy, 0.0])
    if len(goals) >= 8:
        break
if not goals:
    goals.append([sx, sy, 0.0])

report = {
    "inputs": {
        "collision_usd": str(collision_usd),
        "mesh_ply": str(mesh_ply),
        "map_yaml": str(map_yaml),
        "dt": 1.0 / 120.0,
        "seed": 0,
    },
    "map": {
        "resolution": resolution,
        "origin": origin,
        "size": [width, height],
        "free_cell_count": free_count,
        "start_cell": [int(start_cell[0]), int(start_cell[1])],
        "start_xy": [sx, sy],
        "goals": goals,
    },
    "floor_estimate": {"floor_z": 0.0},
    "physics_validation": {"final": {"root_pos_w": [sx, sy, 0.0], "root_quat_w": [1.0, 0.0, 0.0, 0.0]}},
}
report_path.write_text(json.dumps(report, indent=2) + "\n")
print(report_path)
PY

  WORLD_NAV_VALIDATION_REPORT="$report"
  echo "Prepared current-scene world-nav report: $WORLD_NAV_VALIDATION_REPORT"
}

if [[ "$STOP" == "1" ]]; then
  echo "Stopping remote one-click Isaac Gaussian WebUI stack."
  stop_isaac_publisher_for_bridge
  stop_nav2_baseline_stack
  docker rm -f "$HIFI_BRIDGE_CONTAINER" "$HIFI_VIEW_CONTAINER" >/dev/null 2>&1 || true
  stop_port "$HIFI_PORT"
  stop_port "$WEB_PORT"
  stop_port "$WORLD_NAV_PORT"
  stop_port "$BRIDGE_PORT"
  stop_port "$MAPPER_PORT"
  echo "Stopped managed remote stack."
  exit 0
fi

if [[ "$RESTART" == "1" || "$NAV_MODE" == "1" || "$START_ISAAC" != "1" ]]; then
  stop_isaac_publisher_for_bridge
fi
if [[ "$RESTART" == "1" || "$WORLD_NAV_BACKEND" == "nav2" ]]; then
  stop_nav2_baseline_stack
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
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "$command" > "$log_path" 2>&1 < /dev/null &
  else
    nohup bash -lc "$command" > "$log_path" 2>&1 < /dev/null &
  fi
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

ensure_visual_gaussian_chunks
ensure_visual_scene_pack
write_offline_gs_scene_manifest
start_hifi_renderer

mapper_cmd="$(quote_cmd bash "$REMOTE_ROOT/scripts/launch_isaac_online_gaussian_mapper.sh" "$MAPPER_PORT")"
start_service "mapper" "$MAPPER_PORT" "$mapper_cmd"
wait_http "mapper" "http://127.0.0.1:${MAPPER_PORT}/status" 1 30

bridge_env=(MAPPER_URL="http://127.0.0.1:${MAPPER_PORT}")
if [[ -n "$RENDERER_URL" ]]; then
  bridge_env+=(RENDERER_URL="$RENDERER_URL")
  if [[ "$ISAAC_TRAINING_TRAJECTORY" == "1" ]]; then
    bridge_env+=(LIVE_RENDER_POSE_MODE="always")
    bridge_env+=(MANUAL_RENDER_POSE_HOLD_SEC="0")
  else
    bridge_env+=(LIVE_RENDER_POSE_MODE="never")
    bridge_env+=(MANUAL_RENDER_POSE_HOLD_SEC="3600")
  fi
  bridge_env+=(MAX_RENDER_POSE_DISTANCE_FROM_ORIGIN="$HIFI_RENDER_POSE_MAX_DISTANCE")
  if port_is_open "$BRIDGE_PORT" && ! python3 - "$BRIDGE_PORT" "$HIFI_RENDER_POSE_MAX_DISTANCE" "$ISAAC_TRAINING_TRAJECTORY" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/status", timeout=2) as response:
        status = json.loads(response.read().decode("utf-8"))
except Exception:
    sys.exit(1)

try:
    expected_max_distance = float(sys.argv[2])
    current_max_distance = float(status.get("maxRenderPoseDistanceFromOrigin") or 0.0)
except Exception:
    expected_max_distance = 0.0
    current_max_distance = -1.0
expected_live_mode = "always" if sys.argv[3] == "1" else "never"

ok = (
    status.get("rendererConfigured")
    and status.get("liveRenderPoseMode") == expected_live_mode
    and abs(current_max_distance - expected_max_distance) < 1e-6
)
sys.exit(0 if ok else 1)
PY
  then
    echo "bridge already listening on :$BRIDGE_PORT with stale renderer-pose settings; restarting it for $RENDERER_URL."
    stop_port "$BRIDGE_PORT"
  fi
fi
bridge_cmd="$(quote_cmd env "${bridge_env[@]}" bash "$REMOTE_ROOT/scripts/launch_isaac_gaussian_online_bridge.sh" "$BRIDGE_PORT")"
start_service "bridge" "$BRIDGE_PORT" "$bridge_cmd"
wait_http "bridge" "http://127.0.0.1:${BRIDGE_PORT}/status" 1 30
prime_hifi_training_pose
ensure_auto_isaac_scene_usd
ensure_world_nav_report
write_offline_gs_scene_manifest

if [[ "$START_WORLD_NAV" == "1" ]]; then
  if [[ "$WORLD_NAV_BACKEND" == "nav2" ]]; then
    world_nav_env=(
      WORLD_NAV_BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}" \
      WORLD_NAV_OUTPUT_DIR="$WORLD_NAV_OUTPUT_DIR" \
      NAV2_BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}" \
      NAV2_OUTPUT_DIR="$WORLD_NAV_OUTPUT_DIR" \
      NAV2_ROBOT_RADIUS="0.32" \
      NAV2_COLLISION_RADIUS="0.34" \
      NAV2_START_CLEARANCE_MARGIN="0.10" \
      NAV2_START_POLICY="rightmost" \
      NAV2_MOTION_MODE="cmd_vel" \
      NAV2_BRIDGE_POSE_MODE="$([[ "$START_ISAAC" == "1" ]] && echo cloud-only || echo pose-and-cloud)" \
      LOG_DIR="$LOG_DIR"
    )
    if [[ -n "$WORLD_NAV_VALIDATION_REPORT" ]]; then
      world_nav_env+=(WORLD_NAV_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT" NAV2_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT")
    fi
    world_nav_cmd="$(quote_cmd env "${world_nav_env[@]}" bash "$REMOTE_ROOT/scripts/launch_nav2_baseline.sh" "$WORLD_NAV_PORT")"
  else
    world_nav_env=(
      DISPLAY="$DISPLAY_VALUE" \
      QT_X11_NO_MITSHM=1 \
      WORLD_NAV_VIEWER="$WORLD_NAV_VIEWER" \
      WORLD_NAV_HOLD_OPEN_SEC="$WORLD_NAV_HOLD_OPEN_SEC" \
      WORLD_NAV_EPISODE_HOLD_SEC="$WORLD_NAV_EPISODE_HOLD_SEC" \
      WORLD_NAV_STEP_SLEEP="$WORLD_NAV_STEP_SLEEP" \
      WORLD_NAV_TIMEOUT_SEC="$WORLD_NAV_TIMEOUT_SEC" \
      WORLD_NAV_BRIDGE_URL="http://127.0.0.1:${BRIDGE_PORT}" \
      WORLD_NAV_DEVICE="$ISAAC_DEVICE" \
      WORLD_NAV_ROBOT_MODEL="$WORLD_NAV_ROBOT_MODEL" \
      WORLD_NAV_OUTPUT_DIR="$WORLD_NAV_OUTPUT_DIR"
    )
    if [[ -n "$WORLD_NAV_VALIDATION_REPORT" ]]; then
      world_nav_env+=(WORLD_NAV_VALIDATION_REPORT="$WORLD_NAV_VALIDATION_REPORT")
    fi
    world_nav_cmd="$(quote_cmd env "${world_nav_env[@]}" bash "$REMOTE_ROOT/scripts/launch_world_nav_module.sh" "$WORLD_NAV_PORT")"
  fi
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
  ensure_auto_isaac_scene_usd
  isaac_pattern="run_isaac_gaussian_online_demo.py.*--bridge-url http://127.0.0.1:${BRIDGE_PORT}"
  if [[ "$RESTART" == "1" ]]; then
    stop_isaac_publisher_for_bridge
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
    --control-source "$ISAAC_CONTROL_SOURCE"
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
  if [[ "$ISAAC_TRAINING_TRAJECTORY" == "1" ]]; then
    isaac_args+=(
      --training-trajectory-output-dir "$HIFI_OUTPUT_DIR"
      --training-trajectory-split "$ISAAC_TRAINING_TRAJECTORY_SPLIT"
      --training-rgb-source "$ISAAC_TRAINING_RGB_SOURCE"
    )
  fi

  isaac_ros_setup=""
  if [[ "$ISAAC_CONTROL_SOURCE" == "ros2-cmd_vel" ]]; then
    isaac_args+=(--ros-follow-odom)
    isaac_ros_setup="source /opt/ros/humble/setup.bash && "
  fi
  isaac_cmd="${isaac_ros_setup}source $(printf '%q' "$ISAAC_ENV") && export OMNI_KIT_ACCEPT_EULA=YES DISPLAY=$(printf '%q' "$DISPLAY_VALUE") XAUTHORITY=$(printf '%q' "$XAUTHORITY_VALUE") QT_X11_NO_MITSHM=1 && \"\$ISAACSIM_PYTHON_EXE\" $(quote_cmd "${isaac_args[@]}")"
  echo "Starting isaac-publisher"
  echo "$isaac_cmd" > "$LOG_DIR/isaac-publisher.cmd"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash -lc "$isaac_cmd" > "$LOG_DIR/isaac-publisher.log" 2>&1 < /dev/null &
  else
    nohup bash -lc "$isaac_cmd" > "$LOG_DIR/isaac-publisher.log" 2>&1 < /dev/null &
  fi
  echo "$!" > "$LOG_DIR/isaac-publisher.pid"
  echo "Isaac publisher log: $LOG_DIR/isaac-publisher.log"
fi

if [[ "$START_HIFI" == "1" && "$START_ISAAC" == "1" ]]; then
  wait_bridge_display_frame 180 "renderer"
elif [[ "$START_ISAAC" == "1" ]]; then
  wait_bridge_display_frame 180
fi

STACK_MODE_LABEL="review"
if [[ "$NAV_MODE" == "1" ]]; then
  STACK_MODE_LABEL="nav baseline"
fi

cat > "$LOG_DIR/README.txt" <<EOF
Remote one-click Isaac Gaussian WebUI stack

Mode:    ${STACK_MODE_LABEL}
WebUI:   http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}
Bridge:  http://127.0.0.1:${BRIDGE_PORT}/status
Mapper:  http://127.0.0.1:${MAPPER_PORT}/status
HiFi:    ${RENDERER_URL:-disabled}
World:   http://127.0.0.1:${WORLD_NAV_PORT}/status
Robot:   ${WORLD_NAV_ROBOT_MODEL}
Nav:     ${WORLD_NAV_BACKEND}
Isaac:   ${START_ISAAC} (${ISAAC_CONTROL_SOURCE})
Logs:    $LOG_DIR
EOF

echo "Stack launched."
echo "Logs: $LOG_DIR"
echo "Browser URL:"
echo "  http://127.0.0.1:${WEB_PORT}/?scene=${WEB_SCENE}&mode=${WEB_MODE}"
REMOTE_SCRIPT

if [[ "$START_TUNNEL" == "1" && ! is_local_target ]]; then
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
