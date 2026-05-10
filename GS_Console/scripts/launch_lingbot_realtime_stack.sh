#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd -P)"
CVPR_ROOT="${CVPR_ROOT:-$WORKSPACE_ROOT/CVPR}"
NUC_ROOT="${NUC_ROOT:-$CVPR_ROOT/HMR3D/nuc}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-5173}"
ROSBRIDGE_PORT="${ROSBRIDGE_PORT:-9090}"
WORLD_NAV_PORT="${WORLD_NAV_PORT:-8892}"

START_LINGBOT="${START_LINGBOT:-1}"
START_ROSBRIDGE="${START_ROSBRIDGE:-1}"
START_WEBUI="${START_WEBUI:-1}"
START_MONITOR_SYNC="${START_MONITOR_SYNC:-1}"
START_NAV2_STYLE="${START_NAV2_STYLE:-1}"
START_GAUSSIAN_MONITOR_RENDER="${START_GAUSSIAN_MONITOR_RENDER:-0}"
START_FEEDFORWARD_GS_JOBS="${START_FEEDFORWARD_GS_JOBS:-0}"
TRACKING_BACKEND_EFFECTIVE="${TRACKING_BACKEND:-hikrobot_mono_rgb}"
if [[ "$TRACKING_BACKEND_EFFECTIVE" == "realsense_mono_rgb" ]]; then
  START_REALSENSE_CAMERA="${START_REALSENSE_CAMERA:-1}"
else
  START_REALSENSE_CAMERA="${START_REALSENSE_CAMERA:-0}"
fi

LINGBOT_OUTPUT_DIR="${LINGBOT_OUTPUT_DIR:-$CVPR_ROOT/nuc_output/hikrobot_lingbot_ros2_current_cloud_live}"
GS_MONITOR_DIR="${GS_MONITOR_DIR:-$CVPR_ROOT/nuc_output/real2sim_hikrobot_lingbot_live_baseline}"
LINGBOT_LAUNCH="${LINGBOT_LAUNCH:-$NUC_ROOT/scripts/launch_cuvslam_lingbot_live_reconstruction.sh}"
LIVE_CONTRACT="${LIVE_CONTRACT:-/contracts/lingbot-map-ros2-live.live-contract.json}"
DEFAULT_CUVSLAM_SITE_PACKAGES="$CVPR_ROOT/cuVSLAM/.venv-jetson/lib/python3.10/site-packages"
DEFAULT_CUVSLAM_OVERLAY_DIR="$CVPR_ROOT/nuc_output/cuvslam_python_overlay"
DEFAULT_CUVSLAM_PYTHON="$CVPR_ROOT/cuVSLAM/.venv-jetson/bin/python"

if [[ "$START_LINGBOT" == "1" && ! -x "$LINGBOT_LAUNCH" ]]; then
  echo "LingBot launch script is missing or not executable: $LINGBOT_LAUNCH" >&2
  exit 1
fi

if [[ ("$START_LINGBOT" == "1" || "$START_ROSBRIDGE" == "1" || "$START_NAV2_STYLE" == "1") && ! -f "$ROS_SETUP" ]]; then
  echo "ROS setup not found: $ROS_SETUP" >&2
  echo "Set ROS_SETUP=/path/to/setup.bash or disable ROS pieces with START_ROSBRIDGE=0 START_NAV2_STYLE=0." >&2
  exit 1
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  if [[ "${START_REALSENSE_CAMERA:-0}" == "1" ]]; then
    pkill -f "/opt/ros/humble/lib/realsense2_camera/realsense2_camera_node" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p "$LINGBOT_OUTPUT_DIR" "$GS_MONITOR_DIR"
python3 "$CVPR_ROOT/HMR3D/nuc/scripts/generate_gs_console_monitor_viewer.py" \
  --baseline-dir "$GS_MONITOR_DIR" >/dev/null 2>&1 || true

if [[ "$START_ROSBRIDGE" == "1" ]]; then
  if bash -lc "source '$ROS_SETUP' && ros2 pkg prefix rosbridge_server >/dev/null 2>&1"; then
    echo "Starting ROS2 rosbridge on ws://localhost:$ROSBRIDGE_PORT"
    bash -lc "source '$ROS_SETUP' && ros2 launch rosbridge_server rosbridge_websocket_launch.xml address:=0.0.0.0 port:=$ROSBRIDGE_PORT" &
    pids+=("$!")
  elif [[ -n "${ROSBRIDGE_COMMAND:-}" ]]; then
    echo "Starting custom rosbridge command on ws://localhost:$ROSBRIDGE_PORT"
    bash -lc "$ROSBRIDGE_COMMAND" &
    pids+=("$!")
  else
    cat >&2 <<EOF
rosbridge_server is not installed in this ROS environment: $ROS_SETUP
Install it with:
  sudo apt update
  sudo apt install ros-humble-rosbridge-server

Then rerun this script, or set START_ROSBRIDGE=0 if another rosbridge is already
running on ws://localhost:$ROSBRIDGE_PORT.
EOF
    START_ROSBRIDGE=0
  fi
fi

if [[ "$START_REALSENSE_CAMERA" == "1" ]]; then
  if bash -lc "source '$ROS_SETUP' && ros2 pkg prefix realsense2_camera >/dev/null 2>&1"; then
    REALSENSE_WIDTH="${REALSENSE_WIDTH:-640}"
    REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-480}"
    REALSENSE_FPS="${REALSENSE_FPS:-30}"
    echo "Starting RealSense ROS2 camera ${REALSENSE_WIDTH}x${REALSENSE_HEIGHT}@${REALSENSE_FPS}"
    bash -lc "source '$ROS_SETUP' && ros2 launch realsense2_camera rs_launch.py \
      camera_namespace:=camera camera_name:=camera \
      enable_color:=true enable_depth:=true align_depth.enable:=true \
      rgb_camera.color_profile:=${REALSENSE_WIDTH}x${REALSENSE_HEIGHT}x${REALSENSE_FPS} \
      depth_module.depth_profile:=${REALSENSE_WIDTH}x${REALSENSE_HEIGHT}x${REALSENSE_FPS}" &
    pids+=("$!")
    sleep "${REALSENSE_STARTUP_SLEEP_SEC:-3}"
  else
    echo "realsense2_camera is not installed in this ROS environment: $ROS_SETUP" >&2
    echo "Set START_REALSENSE_CAMERA=0 if another RealSense ROS node is already running." >&2
  fi
fi

if [[ "$START_LINGBOT" == "1" ]]; then
  echo "Starting LingBot live reconstruction with ROS2 publish enabled"
  (
    cd "$CVPR_ROOT"
    export ROS2_PUBLISH=1
    export OUTPUT_DIR="$LINGBOT_OUTPUT_DIR"
    export TRACKING_BACKEND="${TRACKING_BACKEND:-hikrobot_mono_rgb}"
    if [[ "$TRACKING_BACKEND" == "realsense_mono_rgb" ]]; then
      # RealSense RGB already exists as a native ROS2 topic from realsense2_camera.
      # Keep our bridge from republishing it; the WebUI contract subscribes directly.
      export ROS2_IMAGE_TOPIC="${ROS2_IMAGE_TOPIC:-}"
      export ROS2_CAMERA_INFO_TOPIC="${ROS2_CAMERA_INFO_TOPIC:-}"
      export ROS2_CAMERA_FRAME_ID="${ROS2_CAMERA_FRAME_ID:-realsense_color_optical_frame}"
    else
      export ROS2_IMAGE_TOPIC="${ROS2_IMAGE_TOPIC:-/hikrobot/image_raw}"
      export ROS2_CAMERA_INFO_TOPIC="${ROS2_CAMERA_INFO_TOPIC:-/neural_mapping/camera_info}"
      export ROS2_CAMERA_FRAME_ID="${ROS2_CAMERA_FRAME_ID:-hikrobot_camera}"
    fi
    export ROS2_POSE_TOPIC="${ROS2_POSE_TOPIC:-/neural_mapping/pose}"
    export ROS2_PATH_TOPIC="${ROS2_PATH_TOPIC:-/lingbot_nav/trajectory}"
    export ROS2_CLOUD_TOPIC="${ROS2_CLOUD_TOPIC:-/lingbot/cloud_rgb}"
    export ROS2_CLOUD_MIN_INTERVAL_SEC="${ROS2_CLOUD_MIN_INTERVAL_SEC:-0.25}"
    export ROS2_IMAGE_MAX_WIDTH="${ROS2_IMAGE_MAX_WIDTH:-424}"
    export ROS2_IMAGE_MAX_HEIGHT="${ROS2_IMAGE_MAX_HEIGHT:-340}"
    export ROS2_MAX_CLOUD_POINTS="${ROS2_MAX_CLOUD_POINTS:-300000}"
    export ROS2_MAX_CURRENT_CLOUD_POINTS="${ROS2_MAX_CURRENT_CLOUD_POINTS:-30000}"
    export ROS2_CURRENT_CLOUD_REPUBLISH_INTERVAL_SEC="${ROS2_CURRENT_CLOUD_REPUBLISH_INTERVAL_SEC:-1.0}"
    export ASYNC_ARTIFACT_WRITER="${ASYNC_ARTIFACT_WRITER:-1}"
    export ARTIFACT_WRITER_MAX_JOBS="${ARTIFACT_WRITER_MAX_JOBS:-4}"
    export BINARY_CLOUD_WS_PORT="${BINARY_CLOUD_WS_PORT:-19093}"
    export BINARY_CLOUD_WS_HOST="${BINARY_CLOUD_WS_HOST:-0.0.0.0}"
    export BINARY_CLOUD_MAX_POINTS="${BINARY_CLOUD_MAX_POINTS:-60000}"
    export GLOBAL_MAP="${GLOBAL_MAP:-1}"
    export GLOBAL_MAP_VOXEL_SIZE="${GLOBAL_MAP_VOXEL_SIZE:-0.08}"
    export GLOBAL_MAP_RADIUS_M="${GLOBAL_MAP_RADIUS_M:-0.14}"
    export GLOBAL_MAP_MIN_NEIGHBORS="${GLOBAL_MAP_MIN_NEIGHBORS:-1}"
    export GLOBAL_MAP_MAX_POINTS="${GLOBAL_MAP_MAX_POINTS:-300000}"
    export GLOBAL_BINARY_CLOUD_WS_PORT="${GLOBAL_BINARY_CLOUD_WS_PORT:-19094}"
    export GLOBAL_BINARY_CLOUD_MAX_POINTS="${GLOBAL_BINARY_CLOUD_MAX_POINTS:-120000}"
    export PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_CUVSLAM_PYTHON}"
    if [[ ! -x "$PYTHON_BIN" ]]; then
      export PYTHON_BIN="$(command -v python3)"
    fi
    if [[ -z "${CUVSLAM_PYTHONPATH:-}" && -d "$DEFAULT_CUVSLAM_SITE_PACKAGES/cuvslam" ]]; then
      export CUVSLAM_OVERLAY_DIR="${CUVSLAM_OVERLAY_DIR:-$DEFAULT_CUVSLAM_OVERLAY_DIR}"
      mkdir -p "$CUVSLAM_OVERLAY_DIR"
      if [[ -e "$CUVSLAM_OVERLAY_DIR/cuvslam" && ! -L "$CUVSLAM_OVERLAY_DIR/cuvslam" ]]; then
        echo "cuVSLAM overlay path exists and is not a symlink: $CUVSLAM_OVERLAY_DIR/cuvslam" >&2
        echo "Set CUVSLAM_PYTHONPATH explicitly, or remove that path." >&2
      else
        ln -sfn "$DEFAULT_CUVSLAM_SITE_PACKAGES/cuvslam" "$CUVSLAM_OVERLAY_DIR/cuvslam"
        export CUVSLAM_PYTHONPATH="$CUVSLAM_OVERLAY_DIR"
      fi
    fi
    export MODEL_PATH="${MODEL_PATH:-$CVPR_ROOT/third_party_research/lingbot_cache/lingbot-map.pt}"
    export LINGBOT_ROOT="${LINGBOT_ROOT:-$CVPR_ROOT/third_party_research/lingbot-map}"
    export CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
    export HIKROBOT_FPS="${HIKROBOT_FPS:-5}"
    export HIKROBOT_WIDTH="${HIKROBOT_WIDTH:-640}"
    export HIKROBOT_HEIGHT="${HIKROBOT_HEIGHT:-512}"
    export HIKROBOT_EXPOSURE_US="${HIKROBOT_EXPOSURE_US:-15000}"
    export HIKROBOT_GAIN="${HIKROBOT_GAIN:-12}"
    export HIKROBOT_DISABLE_CUVSLAM="${HIKROBOT_DISABLE_CUVSLAM:-0}"
    export HIKROBOT_ASYNC_TRACKING="${HIKROBOT_ASYNC_TRACKING:-0}"
    export HIKROBOT_TRACKING_QUEUE_SIZE="${HIKROBOT_TRACKING_QUEUE_SIZE:-2}"
    export HIKROBOT_TRACKING_IDLE_FPS="${HIKROBOT_TRACKING_IDLE_FPS:-5.0}"
    export HIKROBOT_TRACKING_DENSE_FPS="${HIKROBOT_TRACKING_DENSE_FPS:-1.0}"
    export REALSENSE_INDEX="${REALSENSE_INDEX:-0}"
    export REALSENSE_INPUT_MODE="${REALSENSE_INPUT_MODE:-ros2}"
    export REALSENSE_SERIAL="${REALSENSE_SERIAL:-}"
    export REALSENSE_IMAGE_TOPIC="${REALSENSE_IMAGE_TOPIC:-/camera/camera/color/image_raw}"
    export REALSENSE_CAMERA_INFO_TOPIC="${REALSENSE_CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
    export REALSENSE_TIMEOUT_MS="${REALSENSE_TIMEOUT_MS:-2000}"
    export REALSENSE_FPS="${REALSENSE_FPS:-30}"
    export REALSENSE_WIDTH="${REALSENSE_WIDTH:-640}"
    export REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-480}"
    export REALSENSE_DISABLE_CUVSLAM="${REALSENSE_DISABLE_CUVSLAM:-0}"
    export REALSENSE_ASYNC_TRACKING="${REALSENSE_ASYNC_TRACKING:-${HIKROBOT_ASYNC_TRACKING:-0}}"
    export REALSENSE_TRACKING_QUEUE_SIZE="${REALSENSE_TRACKING_QUEUE_SIZE:-${HIKROBOT_TRACKING_QUEUE_SIZE:-2}}"
    export REALSENSE_TRACKING_IDLE_FPS="${REALSENSE_TRACKING_IDLE_FPS:-${HIKROBOT_TRACKING_IDLE_FPS:-5.0}}"
    export REALSENSE_TRACKING_DENSE_FPS="${REALSENSE_TRACKING_DENSE_FPS:-${HIKROBOT_TRACKING_DENSE_FPS:-1.0}}"
    export CAMERA_FX="${CAMERA_FX:-${HIKROBOT_CAMERA_FX:-0}}"
    export CAMERA_FY="${CAMERA_FY:-${HIKROBOT_CAMERA_FY:-0}}"
    export CAMERA_CX="${CAMERA_CX:-${HIKROBOT_CAMERA_CX:-0}}"
    export CAMERA_CY="${CAMERA_CY:-${HIKROBOT_CAMERA_CY:-0}}"
    export CAMERA_DISTORTION_COEFFS="${CAMERA_DISTORTION_COEFFS:-${HIKROBOT_DISTORTION_COEFFS:-}}"
    if [[ -n "$CAMERA_DISTORTION_COEFFS" ]]; then
      export CAMERA_UNDISTORT="${CAMERA_UNDISTORT:-1}"
    else
      export CAMERA_UNDISTORT="${CAMERA_UNDISTORT:-0}"
    fi
    export IMAGE_SIZE="${IMAGE_SIZE:-224}"
    export MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
    export WINDOW_SIZE="${WINDOW_SIZE:-2}"
    export WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
    export FRAME_STEP="${FRAME_STEP:-1}"
    export DENSE_FRAME_INTERVAL="${DENSE_FRAME_INTERVAL:-15}"
    export DENSE_SCHEDULER="${DENSE_SCHEDULER:-interval}"
    export DENSE_MIN_FRAME_GAP="${DENSE_MIN_FRAME_GAP:-0}"
    export DENSE_TRANSLATION_THRESH_M="${DENSE_TRANSLATION_THRESH_M:-0.25}"
    export DENSE_ROTATION_THRESH_DEG="${DENSE_ROTATION_THRESH_DEG:-12.0}"
    export DENSE_PIXEL_MOTION_THRESH="${DENSE_PIXEL_MOTION_THRESH:-18.0}"
    export DENSE_SUBMIT_WHEN_WORKER_IDLE="${DENSE_SUBMIT_WHEN_WORKER_IDLE:-1}"
    export MAX_QUEUE="${MAX_QUEUE:-1}"
    export MAX_FRAMES="${MAX_FRAMES:-0}"
    export SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
    export SAMPLING_PATTERN="${SAMPLING_PATTERN:-random}"
    export MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-15000}"
    export MAX_ACTIVE_FRAMES="${MAX_ACTIVE_FRAMES:-16}"
    export ROLLING_MAP="${ROLLING_MAP:-1}"
    export ROLLING_MAP_VOXEL_SIZE="${ROLLING_MAP_VOXEL_SIZE:-0.06}"
    export ROLLING_MAP_RADIUS_M="${ROLLING_MAP_RADIUS_M:-0.12}"
    export ROLLING_MAP_MIN_NEIGHBORS="${ROLLING_MAP_MIN_NEIGHBORS:-2}"
    export ROLLING_MAP_MAX_WINDOWS="${ROLLING_MAP_MAX_WINDOWS:-8}"
    export ROLLING_MAP_MAX_AGE_SEC="${ROLLING_MAP_MAX_AGE_SEC:-30.0}"
    export ROLLING_MAP_MAX_POINTS="${ROLLING_MAP_MAX_POINTS:-180000}"
    export KEYFRAME_TRANSLATION_THRESH_M="${KEYFRAME_TRANSLATION_THRESH_M:-0.2}"
    export KEYFRAME_ROTATION_THRESH_DEG="${KEYFRAME_ROTATION_THRESH_DEG:-10.0}"
    export KEYFRAME_TIME_THRESH_SEC="${KEYFRAME_TIME_THRESH_SEC:-2.0}"
    export KEYFRAME_MAX_COUNT="${KEYFRAME_MAX_COUNT:-200}"
    export DROP_WHEN_BUSY="${DROP_WHEN_BUSY:-1}"
    export PAUSE_TRACKING_WHILE_DENSE="${PAUSE_TRACKING_WHILE_DENSE:-1}"
    export DENSE_BUSY_TRACKING_POLICY="${DENSE_BUSY_TRACKING_POLICY:-none}"
    export DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC="${DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC:-1.0}"
    export OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-1}"
    export PRELOAD_LINGBOT_MODEL="${PRELOAD_LINGBOT_MODEL:-1}"
    export WARMUP_FIRST_WINDOW="${WARMUP_FIRST_WINDOW:-0}"
    export LINGBOT_PERSISTENT_STREAMING="${LINGBOT_PERSISTENT_STREAMING:-0}"
    export LINGBOT_ENABLE_CAMERA="${LINGBOT_ENABLE_CAMERA:-0}"
    export PREFER_LINGBOT_POSE="${PREFER_LINGBOT_POSE:-0}"
    export LINGBOT_POSE_TRANSLATION_SCALE="${LINGBOT_POSE_TRANSLATION_SCALE:-0}"
    export LINGBOT_EXTRINSIC_MODE="${LINGBOT_EXTRINSIC_MODE:-inverse}"
    export COMPILE_LINGBOT_MODEL="${COMPILE_LINGBOT_MODEL:-0}"
    export COMPILE_WARMUP_PASSES="${COMPILE_WARMUP_PASSES:-3}"
    export COMPILE_WARMUP_STREAM_FRAMES="${COMPILE_WARMUP_STREAM_FRAMES:-1}"
    if [[ -z "${USE_SDPA:-}" ]]; then
      if "$PYTHON_BIN" -c "import flashinfer" >/dev/null 2>&1; then
        export USE_SDPA=0
      else
        export USE_SDPA=1
      fi
    else
      export USE_SDPA
    fi
    export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.7}"
    export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-model}"
    set +u
    source "$ROS_SETUP"
    source "$CVPR_ROOT/HMR3D/nuc/configs/hikrobot_mvs_env.sh" 2>/dev/null || true
    source "$CVPR_ROOT/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh" 2>/dev/null || true
    set -u
    exec "$LINGBOT_LAUNCH"
  ) &
  pids+=("$!")
fi

if [[ "$START_MONITOR_SYNC" == "1" ]]; then
  echo "Syncing LingBot live assets into GS monitor directory"
  python3 "$CVPR_ROOT/HMR3D/nuc/scripts/sync_gs_console_monitor_assets.py" \
    --live-dir "$LINGBOT_OUTPUT_DIR" \
    --output-dir "$GS_MONITOR_DIR" \
    --interval-sec "${MONITOR_SYNC_INTERVAL_SEC:-0.75}" \
    --max-points "${MONITOR_MAX_POINTS:-28000}" &
  pids+=("$!")
fi

if [[ "$START_NAV2_STYLE" == "1" ]]; then
  echo "Starting lightweight Nav2-style map bridge from GS monitor JSON"
  bash -lc "source '$ROS_SETUP' && python3 '$CVPR_ROOT/HMR3D/nuc/scripts/publish_gs_console_nav2_style_ros2.py' --monitor-json '$GS_MONITOR_DIR/monitor/live_monitor.json' --interval-sec '${NAV2_STYLE_INTERVAL_SEC:-0.75}' --resolution '${NAV2_STYLE_RESOLUTION:-0.12}'" &
  pids+=("$!")
fi

if [[ "$START_FEEDFORWARD_GS_JOBS" == "1" ]]; then
  echo "Preparing feed-forward GS jobs for GenWildSplat/AnySplat sidecar"
  python3 "$CVPR_ROOT/HMR3D/nuc/scripts/prepare_feedforward_gs_live_jobs.py" \
    --worker-dir "$LINGBOT_OUTPUT_DIR/worker" \
    --output-dir "$GS_MONITOR_DIR" \
    --interval-sec "${FEEDFORWARD_GS_INTERVAL_SEC:-3.0}" \
    --context-views "${FEEDFORWARD_GS_CONTEXT_VIEWS:-6}" \
    --window-results "${FEEDFORWARD_GS_WINDOW_RESULTS:-12}" \
    --image-size "${FEEDFORWARD_GS_IMAGE_SIZE:-448}" \
    --min-new-windows "${FEEDFORWARD_GS_MIN_NEW_WINDOWS:-2}" \
    --sidecar-url "${FEEDFORWARD_GS_SIDECAR_URL:-}" &
  pids+=("$!")
fi

if [[ "$START_GAUSSIAN_MONITOR_RENDER" == "1" ]]; then
  echo "Starting GS monitor renderer at ${GAUSSIAN_RENDER_INTERVAL_SEC:-3.0}s cadence"
  python3 "$CVPR_ROOT/HMR3D/nuc/scripts/render_gs_console_monitor_frame.py" \
    --baseline-dir "$GS_MONITOR_DIR" \
    --worker-dir "$LINGBOT_OUTPUT_DIR/worker" \
    --source "${GAUSSIAN_RENDER_SOURCE:-baseline_seed}" \
    --live-map "${GAUSSIAN_RENDER_LIVE_MAP:-$LINGBOT_OUTPUT_DIR/live_map.npz}" \
    --interval-sec "${GAUSSIAN_RENDER_INTERVAL_SEC:-3.0}" \
    --width "${GAUSSIAN_RENDER_WIDTH:-480}" \
    --render-view-budget-points "${GAUSSIAN_RENDER_BUDGET_POINTS:-24000}" \
    --live-map-max-points "${GAUSSIAN_RENDER_LIVE_MAX_POINTS:-24000}" \
    --live-gaussian-scale "${GAUSSIAN_RENDER_LIVE_SCALE:-0.055}" \
    --live-gaussian-opacity "${GAUSSIAN_RENDER_LIVE_OPACITY:-0.72}" &
  pids+=("$!")
fi

if [[ "$START_WEBUI" == "1" ]]; then
echo "Starting GS Console WebUI on http://localhost:$WEB_PORT/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=$LIVE_CONTRACT"
  (
    export WEB_ADAPTER=lingbot-live
    export WEB_MODE=live
    export WEB_LIVE_CONTRACT="$LIVE_CONTRACT"
    export WEB_LINGBOT_LIVE_ASSET_ROOT="$LINGBOT_OUTPUT_DIR"
    export WEB_LINGBOT_REAL2SIM_ASSET_ROOT="$GS_MONITOR_DIR"
    export ROSBRIDGE_PORT="$ROSBRIDGE_PORT"
    export WORLD_NAV_PORT="$WORLD_NAV_PORT"
    exec "$ROOT_DIR/scripts/launch_web_ui_dev.sh" "$WEB_HOST" "$WEB_PORT"
  ) &
  pids+=("$!")
fi

echo "LingBot realtime stack is up."
echo "WebUI:      http://localhost:$WEB_PORT/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=$LIVE_CONTRACT"
echo "rosbridge:  ws://localhost:$ROSBRIDGE_PORT"
echo "live dir:   $LINGBOT_OUTPUT_DIR"
echo "monitor:    $GS_MONITOR_DIR/real2sim_gs_console_viewer.html"

wait
