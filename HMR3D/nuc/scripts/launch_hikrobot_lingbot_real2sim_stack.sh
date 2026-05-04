#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

ACTION="start"
WITH_RVIZ="${WITH_RVIZ:-0}"
RESTART="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|restart|status)
      ACTION="$1"
      shift
      ;;
    --with-rviz)
      WITH_RVIZ="1"
      shift
      ;;
    --no-rviz)
      WITH_RVIZ="0"
      shift
      ;;
    --restart)
      ACTION="restart"
      RESTART="1"
      shift
      ;;
    --help|-h)
      cat <<EOF
Usage:
  HMR3D/nuc/scripts/launch_hikrobot_lingbot_real2sim_stack.sh [start|stop|restart|status] [--with-rviz]

Starts the live full stack:
  HikRobot RGB -> cuVSLAM/LingBot -> ROS2/RViz topics
  LingBot worker outputs -> TSDF + marching-cubes mesh + Gaussian seed
  Local GS Console-style WebGL viewer on VIEWER_PORT

Important env overrides:
  PYTHON_BIN, LIVE_OUTPUT_DIR, REAL2SIM_OUTPUT_DIR, STACK_LOG_DIR
  LIVE_PORT, VIEWER_PORT, HIKROBOT_FPS, HIKROBOT_EXPOSURE_US, HIKROBOT_GAIN
  IMAGE_SIZE, MODEL_IMAGE_SIZE, MODEL_PATH, LINGBOT_ROOT
  REAL2SIM_MAX_FRAMES, REAL2SIM_VOXEL_SIZE, REAL2SIM_TSDF_MAX_DIM
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "$ACTION" == "restart" ]]; then
  RESTART="1"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map.pt}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
LIVE_OUTPUT_DIR="${LIVE_OUTPUT_DIR:-$ROOT_DIR/nuc_output/hikrobot_lingbot_ros2_current_cloud_live}"
REAL2SIM_OUTPUT_DIR="${REAL2SIM_OUTPUT_DIR:-$ROOT_DIR/nuc_output/real2sim_hikrobot_lingbot_live_baseline}"
STACK_LOG_DIR="${STACK_LOG_DIR:-$ROOT_DIR/nuc_output/hikrobot_lingbot_real2sim_stack/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/nuc_output/hikrobot_lingbot_real2sim_stack/pids}"

LIVE_PORT="${LIVE_PORT:-19102}"
VIEWER_PORT="${VIEWER_PORT:-19103}"

IMAGE_SIZE="${IMAGE_SIZE:-224}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
MAX_FRAMES="${MAX_FRAMES:-0}"
FRAME_STEP="${FRAME_STEP:-1}"
HIKROBOT_WIDTH="${HIKROBOT_WIDTH:-640}"
HIKROBOT_HEIGHT="${HIKROBOT_HEIGHT:-512}"
HIKROBOT_FPS="${HIKROBOT_FPS:-5}"
HIKROBOT_EXPOSURE_US="${HIKROBOT_EXPOSURE_US:-15000}"
HIKROBOT_GAIN="${HIKROBOT_GAIN:-12}"
ROS2_IMAGE_MAX_WIDTH="${ROS2_IMAGE_MAX_WIDTH:-640}"
ROS2_MAX_CLOUD_POINTS="${ROS2_MAX_CLOUD_POINTS:-300000}"
ROS2_MAX_CURRENT_CLOUD_POINTS="${ROS2_MAX_CURRENT_CLOUD_POINTS:-60000}"
LIVE_MAX_POINTS_PER_FRAME="${LIVE_MAX_POINTS_PER_FRAME:-15000}"
LIVE_MAX_ACTIVE_FRAMES="${LIVE_MAX_ACTIVE_FRAMES:-16}"

REAL2SIM_INTERVAL_SEC="${REAL2SIM_INTERVAL_SEC:-180}"
REAL2SIM_MIN_NEW_WINDOWS="${REAL2SIM_MIN_NEW_WINDOWS:-6}"
REAL2SIM_MIN_READY_WINDOWS="${REAL2SIM_MIN_READY_WINDOWS:-2}"
REAL2SIM_KEEP_LAST="${REAL2SIM_KEEP_LAST:-5}"
REAL2SIM_MAX_FRAMES="${REAL2SIM_MAX_FRAMES:-48}"
REAL2SIM_VOXEL_SIZE="${REAL2SIM_VOXEL_SIZE:-0.12}"
REAL2SIM_TSDF_MAX_DIM="${REAL2SIM_TSDF_MAX_DIM:-140}"
REAL2SIM_TSDF_CHUNK_VOXELS="${REAL2SIM_TSDF_CHUNK_VOXELS:-120000}"
REAL2SIM_GAUSSIAN_MAX_POINTS="${REAL2SIM_GAUSSIAN_MAX_POINTS:-80000}"
REAL2SIM_GAUSSIAN_SCALE="${REAL2SIM_GAUSSIAN_SCALE:-0.07}"

LIVE_PATTERN="run_cuvslam_lingbot_live_reconstruction.py --tracking-backend hikrobot_mono_rgb"
REAL2SIM_PATTERN="run_live_real2sim_baseline.py"
VIEWER_PATTERN="http.server ${VIEWER_PORT}"
RVIZ_PATTERN="rviz2 -d HMR3D/nuc/configs/lingbot_dual_cloud.rviz"

mkdir -p "$STACK_LOG_DIR" "$PID_DIR" "$REAL2SIM_OUTPUT_DIR"

pid_file() {
  echo "$PID_DIR/$1.pid"
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

find_pattern_pid() {
  local pattern="$1"
  pgrep -f "$pattern" | head -n 1 || true
}

component_pid() {
  local name="$1"
  local pattern="$2"
  local file
  file="$(pid_file "$name")"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if pid_alive "$pid"; then
      echo "$pid"
      return
    fi
  fi
  find_pattern_pid "$pattern"
}

stop_component() {
  local name="$1"
  local pattern="$2"
  local pid
  pid="$(component_pid "$name" "$pattern")"
  if [[ -z "$pid" ]]; then
    rm -f "$(pid_file "$name")"
    echo "[$name] not running"
    return
  fi
  echo "[$name] stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! pid_alive "$pid"; then
      break
    fi
    sleep 0.5
  done
  if pid_alive "$pid"; then
    echo "[$name] still running after SIGTERM; leaving it alive" >&2
  else
    rm -f "$(pid_file "$name")"
  fi
}

print_component_status() {
  local name="$1"
  local pattern="$2"
  local pid
  pid="$(component_pid "$name" "$pattern")"
  if [[ -n "$pid" ]]; then
    echo "[$name] running pid=$pid"
  else
    echo "[$name] stopped"
  fi
}

preflight() {
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found: $PYTHON_BIN" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_PATH" ]]; then
    echo "Missing model checkpoint: $MODEL_PATH" >&2
    exit 1
  fi
  if [[ ! -d "$LINGBOT_ROOT" ]]; then
    echo "Missing LingBot root: $LINGBOT_ROOT" >&2
    exit 1
  fi
}

start_live() {
  local existing
  existing="$(component_pid live "$LIVE_PATTERN")"
  if [[ -n "$existing" ]]; then
    echo "[live] already running pid=$existing"
    echo "$existing" > "$(pid_file live)"
    return
  fi
  local log="$STACK_LOG_DIR/live_lingbot.log"
  echo "[live] starting, log=$log"
  setsid env \
    ROOT_DIR="$ROOT_DIR" \
    PYTHON_BIN="$PYTHON_BIN" \
    MODEL_PATH="$MODEL_PATH" \
    LINGBOT_ROOT="$LINGBOT_ROOT" \
    LIVE_OUTPUT_DIR="$LIVE_OUTPUT_DIR" \
    IMAGE_SIZE="$IMAGE_SIZE" \
    MODEL_IMAGE_SIZE="$MODEL_IMAGE_SIZE" \
    WINDOW_SIZE="$WINDOW_SIZE" \
    WINDOW_STRIDE="$WINDOW_STRIDE" \
    MAX_FRAMES="$MAX_FRAMES" \
    FRAME_STEP="$FRAME_STEP" \
    HIKROBOT_WIDTH="$HIKROBOT_WIDTH" \
    HIKROBOT_HEIGHT="$HIKROBOT_HEIGHT" \
    HIKROBOT_FPS="$HIKROBOT_FPS" \
    HIKROBOT_EXPOSURE_US="$HIKROBOT_EXPOSURE_US" \
    HIKROBOT_GAIN="$HIKROBOT_GAIN" \
    ROS2_IMAGE_MAX_WIDTH="$ROS2_IMAGE_MAX_WIDTH" \
    ROS2_MAX_CLOUD_POINTS="$ROS2_MAX_CLOUD_POINTS" \
    ROS2_MAX_CURRENT_CLOUD_POINTS="$ROS2_MAX_CURRENT_CLOUD_POINTS" \
    LIVE_MAX_POINTS_PER_FRAME="$LIVE_MAX_POINTS_PER_FRAME" \
    LIVE_MAX_ACTIVE_FRAMES="$LIVE_MAX_ACTIVE_FRAMES" \
    LIVE_PORT="$LIVE_PORT" \
    PYTHONUNBUFFERED=1 \
    bash -lc '
      set -euo pipefail
      cd "$ROOT_DIR"
      set +u
      source /opt/ros/humble/setup.bash
      source HMR3D/nuc/configs/hikrobot_mvs_env.sh 2>/dev/null || true
      source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
      set -u
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.7}"
      export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
      export LINGBOT_CHECKPOINT_MMAP=1
      export LINGBOT_MODEL_DTYPE=fp16
      export LINGBOT_CPU_CAST_BEFORE_CUDA=1
      export LINGBOT_CPU_CAST_SCOPE=model
      exec "$PYTHON_BIN" -X faulthandler HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py \
        --tracking-backend hikrobot_mono_rgb \
        --model-path "$MODEL_PATH" \
        --lingbot-map-root "$LINGBOT_ROOT" \
        --output-dir "$LIVE_OUTPUT_DIR" \
        --image-size "$IMAGE_SIZE" \
        --model-image-size "$MODEL_IMAGE_SIZE" \
        --window-size "$WINDOW_SIZE" \
        --stride "$WINDOW_STRIDE" \
        --max-frames "$MAX_FRAMES" \
        --frame-step "$FRAME_STEP" \
        --hikrobot-width "$HIKROBOT_WIDTH" \
        --hikrobot-height "$HIKROBOT_HEIGHT" \
        --hikrobot-fps "$HIKROBOT_FPS" \
        --hikrobot-exposure-us "$HIKROBOT_EXPOSURE_US" \
        --hikrobot-gain "$HIKROBOT_GAIN" \
        --hikrobot-threaded-capture \
        --hikrobot-capture-queue-size 6 \
        --hikrobot-max-read-errors 100 \
        --ros2-publish \
        --ros2-image-max-width "$ROS2_IMAGE_MAX_WIDTH" \
        --ros2-max-cloud-points "$ROS2_MAX_CLOUD_POINTS" \
        --ros2-max-current-cloud-points "$ROS2_MAX_CURRENT_CLOUD_POINTS" \
        --ros2-republish-current-cloud-on-image \
        --ros2-cloud-min-interval-sec 1.0 \
        --publish-every-windows 1 \
        --publish-every-frames 0 \
        --sample-stride 1 \
        --sampling-pattern random \
        --max-points-per-frame "$LIVE_MAX_POINTS_PER_FRAME" \
        --max-active-frames "$LIVE_MAX_ACTIVE_FRAMES" \
        --drop-when-busy \
        --serve \
        --port "$LIVE_PORT"
    ' > "$log" 2>&1 < /dev/null &
  echo "$!" > "$(pid_file live)"
}

start_real2sim() {
  local existing
  existing="$(component_pid real2sim "$REAL2SIM_PATTERN")"
  if [[ -n "$existing" ]]; then
    echo "[real2sim] already running pid=$existing"
    echo "$existing" > "$(pid_file real2sim)"
    return
  fi
  local log="$STACK_LOG_DIR/real2sim_runner.log"
  echo "[real2sim] starting, log=$log"
  setsid env PYTHONUNBUFFERED=1 "$PYTHON_BIN" HMR3D/nuc/scripts/run_live_real2sim_baseline.py \
    --worker-dir "$LIVE_OUTPUT_DIR/worker" \
    --output-dir "$REAL2SIM_OUTPUT_DIR" \
    --sequence-prefix live_baseline \
    --latest-name latest \
    --interval-sec "$REAL2SIM_INTERVAL_SEC" \
    --min-new-windows "$REAL2SIM_MIN_NEW_WINDOWS" \
    --min-ready-windows "$REAL2SIM_MIN_READY_WINDOWS" \
    --keep-last "$REAL2SIM_KEEP_LAST" \
    --max-frames "$REAL2SIM_MAX_FRAMES" \
    --voxel-size "$REAL2SIM_VOXEL_SIZE" \
    --mesh-backend tsdf \
    --tsdf-trunc-multiplier 4 \
    --tsdf-max-dim "$REAL2SIM_TSDF_MAX_DIM" \
    --tsdf-chunk-voxels "$REAL2SIM_TSDF_CHUNK_VOXELS" \
    --tsdf-min-weight 1.0 \
    --tsdf-bounds-percentile 99 \
    --gaussian-max-points "$REAL2SIM_GAUSSIAN_MAX_POINTS" \
    --gaussian-scale "$REAL2SIM_GAUSSIAN_SCALE" \
    > "$log" 2>&1 < /dev/null &
  echo "$!" > "$(pid_file real2sim)"
}

start_viewer() {
  "$PYTHON_BIN" HMR3D/nuc/scripts/generate_real2sim_gs_console_viewer.py \
    --baseline-dir "$REAL2SIM_OUTPUT_DIR" >/dev/null
  local existing
  existing="$(component_pid viewer "$VIEWER_PATTERN")"
  if [[ -n "$existing" ]]; then
    echo "[viewer] already running pid=$existing"
    echo "$existing" > "$(pid_file viewer)"
    return
  fi
  local log="$STACK_LOG_DIR/gs_console_viewer_server.log"
  echo "[viewer] starting, log=$log"
  setsid env PYTHONUNBUFFERED=1 bash -lc '
    set -euo pipefail
    cd "$0"
    exec "$1" -m http.server "$2" --bind 0.0.0.0
  ' "$REAL2SIM_OUTPUT_DIR" "$PYTHON_BIN" "$VIEWER_PORT" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$(pid_file viewer)"
}

start_rviz() {
  local existing
  existing="$(component_pid rviz "$RVIZ_PATTERN")"
  if [[ -n "$existing" ]]; then
    echo "[rviz] already running pid=$existing"
    echo "$existing" > "$(pid_file rviz)"
    return
  fi
  local log="$STACK_LOG_DIR/rviz.log"
  local rviz_display="${DISPLAY:-:0}"
  local rviz_xauthority="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
  if [[ ! -S "/tmp/.X11-unix/X${rviz_display#:}" ]]; then
    rviz_display="${DISPLAY:-}"
  fi
  if [[ ! -f "$rviz_xauthority" ]]; then
    rviz_xauthority="${XAUTHORITY:-}"
  fi
  echo "[rviz] starting, log=$log"
  setsid env \
    PYTHONUNBUFFERED=1 \
    DISPLAY="$rviz_display" \
    XAUTHORITY="$rviz_xauthority" \
    QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}" \
    bash -lc '
    set -euo pipefail
    cd "$0"
    set +u
    source /opt/ros/humble/setup.bash
    set -u
    exec rviz2 -d HMR3D/nuc/configs/lingbot_dual_cloud.rviz
  ' "$ROOT_DIR" > "$log" 2>&1 < /dev/null &
  echo "$!" > "$(pid_file rviz)"
}

print_urls() {
  local ip
  ip="$(hostname -I 2>/dev/null | awk "{print \$1}")"
  ip="${ip:-127.0.0.1}"
  echo
  echo "Full stack URLs:"
  echo "  Live LingBot viewer:        http://${ip}:${LIVE_PORT}/live_viewer.html"
  echo "  Real-to-sim GS preview:     http://${ip}:${VIEWER_PORT}/real2sim_gs_console_viewer.html"
  echo "  latest_manifest.json:       http://${ip}:${VIEWER_PORT}/latest_manifest.json"
  echo
  echo "Logs:"
  echo "  $STACK_LOG_DIR"
}

case "$ACTION" in
  status)
    print_component_status live "$LIVE_PATTERN"
    print_component_status real2sim "$REAL2SIM_PATTERN"
    print_component_status viewer "$VIEWER_PATTERN"
    print_component_status rviz "$RVIZ_PATTERN"
    print_urls
    ;;
  stop)
    stop_component rviz "$RVIZ_PATTERN"
    stop_component viewer "$VIEWER_PATTERN"
    stop_component real2sim "$REAL2SIM_PATTERN"
    stop_component live "$LIVE_PATTERN"
    ;;
  restart)
    stop_component rviz "$RVIZ_PATTERN"
    stop_component viewer "$VIEWER_PATTERN"
    stop_component real2sim "$REAL2SIM_PATTERN"
    stop_component live "$LIVE_PATTERN"
    preflight
    start_live
    start_real2sim
    start_viewer
    if [[ "$WITH_RVIZ" == "1" ]]; then
      start_rviz
    fi
    print_urls
    ;;
  start)
    preflight
    start_live
    start_real2sim
    start_viewer
    if [[ "$WITH_RVIZ" == "1" ]]; then
      start_rviz
    fi
    print_urls
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 1
    ;;
esac
