#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CUVSLAM_PYTHON="$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_CUVSLAM_PYTHON}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
DEFAULT_CUVSLAM_SITE_PACKAGES="$ROOT_DIR/cuVSLAM/.venv-jetson/lib/python3.10/site-packages"
DEFAULT_CUVSLAM_OVERLAY_DIR="$ROOT_DIR/nuc_output/cuvslam_python_overlay"
SEQUENCE_DIR="${SEQUENCE_DIR:-$ROOT_DIR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-$ROOT_DIR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_tum.txt}"
TRACKING_BACKEND="${TRACKING_BACKEND:-pose_file}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"
RGB_IMAGE_DIR="${RGB_IMAGE_DIR:-}"
INTRINSIC_CAMERA_INDEX="${INTRINSIC_CAMERA_INDEX:-0}"
MONO_FIXED_STEP_SCALE="${MONO_FIXED_STEP_SCALE:-0.5}"
MONO_SCALE_SOURCE="${MONO_SCALE_SOURCE:-fixed}"
OXTS_DIR="${OXTS_DIR:-}"
HIKROBOT_INDEX="${HIKROBOT_INDEX:-0}"
HIKROBOT_TIMEOUT_MS="${HIKROBOT_TIMEOUT_MS:-2000}"
HIKROBOT_EXPOSURE_US="${HIKROBOT_EXPOSURE_US:-15000}"
HIKROBOT_GAIN="${HIKROBOT_GAIN:-12}"
HIKROBOT_FPS="${HIKROBOT_FPS:-5}"
HIKROBOT_WIDTH="${HIKROBOT_WIDTH:-640}"
HIKROBOT_HEIGHT="${HIKROBOT_HEIGHT:-512}"
HIKROBOT_THREADED_CAPTURE="${HIKROBOT_THREADED_CAPTURE:-1}"
HIKROBOT_CAPTURE_QUEUE_SIZE="${HIKROBOT_CAPTURE_QUEUE_SIZE:-6}"
HIKROBOT_MAX_READ_ERRORS="${HIKROBOT_MAX_READ_ERRORS:-100}"
HIKROBOT_DISABLE_CUVSLAM="${HIKROBOT_DISABLE_CUVSLAM:-0}"
HIKROBOT_ASYNC_TRACKING="${HIKROBOT_ASYNC_TRACKING:-0}"
HIKROBOT_TRACKING_QUEUE_SIZE="${HIKROBOT_TRACKING_QUEUE_SIZE:-2}"
HIKROBOT_TRACKING_IDLE_FPS="${HIKROBOT_TRACKING_IDLE_FPS:-5.0}"
HIKROBOT_TRACKING_DENSE_FPS="${HIKROBOT_TRACKING_DENSE_FPS:-1.0}"
REALSENSE_INDEX="${REALSENSE_INDEX:-0}"
REALSENSE_INPUT_MODE="${REALSENSE_INPUT_MODE:-ros2}"
REALSENSE_SERIAL="${REALSENSE_SERIAL:-}"
REALSENSE_IMAGE_TOPIC="${REALSENSE_IMAGE_TOPIC:-/camera/camera/color/image_raw}"
REALSENSE_CAMERA_INFO_TOPIC="${REALSENSE_CAMERA_INFO_TOPIC:-/camera/camera/color/camera_info}"
REALSENSE_TIMEOUT_MS="${REALSENSE_TIMEOUT_MS:-2000}"
REALSENSE_FPS="${REALSENSE_FPS:-30}"
REALSENSE_WIDTH="${REALSENSE_WIDTH:-640}"
REALSENSE_HEIGHT="${REALSENSE_HEIGHT:-480}"
REALSENSE_THREADED_CAPTURE="${REALSENSE_THREADED_CAPTURE:-1}"
REALSENSE_CAPTURE_QUEUE_SIZE="${REALSENSE_CAPTURE_QUEUE_SIZE:-6}"
REALSENSE_MAX_READ_ERRORS="${REALSENSE_MAX_READ_ERRORS:-100}"
REALSENSE_DISABLE_CUVSLAM="${REALSENSE_DISABLE_CUVSLAM:-0}"
REALSENSE_ASYNC_TRACKING="${REALSENSE_ASYNC_TRACKING:-0}"
REALSENSE_TRACKING_QUEUE_SIZE="${REALSENSE_TRACKING_QUEUE_SIZE:-2}"
REALSENSE_TRACKING_IDLE_FPS="${REALSENSE_TRACKING_IDLE_FPS:-5.0}"
REALSENSE_TRACKING_DENSE_FPS="${REALSENSE_TRACKING_DENSE_FPS:-1.0}"
CAMERA_FX="${CAMERA_FX:-${HIKROBOT_CAMERA_FX:-0}}"
CAMERA_FY="${CAMERA_FY:-${HIKROBOT_CAMERA_FY:-0}}"
CAMERA_CX="${CAMERA_CX:-${HIKROBOT_CAMERA_CX:-0}}"
CAMERA_CY="${CAMERA_CY:-${HIKROBOT_CAMERA_CY:-0}}"
CAMERA_DISTORTION_COEFFS="${CAMERA_DISTORTION_COEFFS:-${HIKROBOT_DISTORTION_COEFFS:-}}"
CAMERA_UNDISTORT="${CAMERA_UNDISTORT:-0}"
RGB_OUTPUT_DIR="${RGB_OUTPUT_DIR:-}"
COLOR_IMAGE_DIR="${COLOR_IMAGE_DIR:-}"
COLOR_IMAGE_TEMPLATE="${COLOR_IMAGE_TEMPLATE:-{frame_idx:010d}.png}"
LINGBOT_ROOT="${LINGBOT_ROOT:-$ROOT_DIR/third_party_research/lingbot-map}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/nuc_output/lingbot_live_reconstruction/kitti0020_live_$(date +%Y%m%d_%H%M%S)}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
MODEL_IMAGE_SIZE="${MODEL_IMAGE_SIZE:-518}"
FRAME_STEP="${FRAME_STEP:-4}"
MAX_FRAMES="${MAX_FRAMES:-32}"
WINDOW_SIZE="${WINDOW_SIZE:-2}"
WINDOW_STRIDE="${WINDOW_STRIDE:-1}"
DENSE_FRAME_INTERVAL="${DENSE_FRAME_INTERVAL:-1}"
DENSE_SCHEDULER="${DENSE_SCHEDULER:-interval}"
DENSE_MIN_FRAME_GAP="${DENSE_MIN_FRAME_GAP:-0}"
DENSE_TRANSLATION_THRESH_M="${DENSE_TRANSLATION_THRESH_M:-0.25}"
DENSE_ROTATION_THRESH_DEG="${DENSE_ROTATION_THRESH_DEG:-12.0}"
DENSE_PIXEL_MOTION_THRESH="${DENSE_PIXEL_MOTION_THRESH:-18.0}"
DENSE_SUBMIT_WHEN_WORKER_IDLE="${DENSE_SUBMIT_WHEN_WORKER_IDLE:-0}"
PAUSE_TRACKING_WHILE_DENSE="${PAUSE_TRACKING_WHILE_DENSE:-0}"
DENSE_BUSY_TRACKING_POLICY="${DENSE_BUSY_TRACKING_POLICY:-none}"
DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC="${DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC:-1.0}"
KEYFRAMES_ONLY="${KEYFRAMES_ONLY:-0}"
MAX_QUEUE="${MAX_QUEUE:-2}"
DROP_WHEN_BUSY="${DROP_WHEN_BUSY:-0}"
USE_SDPA="${USE_SDPA:-}"
OFFLOAD_TO_CPU="${OFFLOAD_TO_CPU:-1}"
PRELOAD_LINGBOT_MODEL="${PRELOAD_LINGBOT_MODEL:-0}"
WARMUP_FIRST_WINDOW="${WARMUP_FIRST_WINDOW:-0}"
COMPILE_LINGBOT_MODEL="${COMPILE_LINGBOT_MODEL:-0}"
COMPILE_WARMUP_PASSES="${COMPILE_WARMUP_PASSES:-3}"
COMPILE_WARMUP_STREAM_FRAMES="${COMPILE_WARMUP_STREAM_FRAMES:-10}"
LINGBOT_PERSISTENT_STREAMING="${LINGBOT_PERSISTENT_STREAMING:-0}"
FRAME_SLEEP_SEC="${FRAME_SLEEP_SEC:-0}"
DEPTH_SCALE="${DEPTH_SCALE:-20.0}"
LINGBOT_POSE_TRANSLATION_SCALE="${LINGBOT_POSE_TRANSLATION_SCALE:-0}"
LINGBOT_EXTRINSIC_MODE="${LINGBOT_EXTRINSIC_MODE:-inverse}"
LINGBOT_ENABLE_CAMERA="${LINGBOT_ENABLE_CAMERA:-0}"
PREFER_LINGBOT_POSE="${PREFER_LINGBOT_POSE:-0}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-8}"
SAMPLING_PATTERN="${SAMPLING_PATTERN:-grid}"
MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-2500}"
MAX_ACTIVE_FRAMES="${MAX_ACTIVE_FRAMES:-16}"
FUSION_MODE="${FUSION_MODE:-raw}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
FUSION_MAX_POINTS="${FUSION_MAX_POINTS:-500000}"
FUSION_MIN_OBSERVATIONS="${FUSION_MIN_OBSERVATIONS:-1}"
ROLLING_MAP="${ROLLING_MAP:-0}"
ROLLING_MAP_VOXEL_SIZE="${ROLLING_MAP_VOXEL_SIZE:-0.06}"
ROLLING_MAP_RADIUS_M="${ROLLING_MAP_RADIUS_M:-0.12}"
ROLLING_MAP_MIN_NEIGHBORS="${ROLLING_MAP_MIN_NEIGHBORS:-2}"
ROLLING_MAP_MAX_WINDOWS="${ROLLING_MAP_MAX_WINDOWS:-8}"
ROLLING_MAP_MAX_AGE_SEC="${ROLLING_MAP_MAX_AGE_SEC:-30.0}"
ROLLING_MAP_MAX_POINTS="${ROLLING_MAP_MAX_POINTS:-180000}"
GLOBAL_MAP="${GLOBAL_MAP:-0}"
GLOBAL_MAP_VOXEL_SIZE="${GLOBAL_MAP_VOXEL_SIZE:-0.08}"
GLOBAL_MAP_RADIUS_M="${GLOBAL_MAP_RADIUS_M:-0.14}"
GLOBAL_MAP_MIN_NEIGHBORS="${GLOBAL_MAP_MIN_NEIGHBORS:-1}"
GLOBAL_MAP_MAX_POINTS="${GLOBAL_MAP_MAX_POINTS:-300000}"
KEYFRAME_TRANSLATION_THRESH_M="${KEYFRAME_TRANSLATION_THRESH_M:-0.2}"
KEYFRAME_ROTATION_THRESH_DEG="${KEYFRAME_ROTATION_THRESH_DEG:-10.0}"
KEYFRAME_TIME_THRESH_SEC="${KEYFRAME_TIME_THRESH_SEC:-2.0}"
KEYFRAME_MAX_COUNT="${KEYFRAME_MAX_COUNT:-200}"
ADAPTIVE_SAMPLING="${ADAPTIVE_SAMPLING:-0}"
NEAR_DEPTH_M="${NEAR_DEPTH_M:-18.0}"
NEAR_SAMPLE_STRIDE="${NEAR_SAMPLE_STRIDE:-1}"
EDGE_SAMPLE_STRIDE="${EDGE_SAMPLE_STRIDE:-2}"
EDGE_PERCENTILE="${EDGE_PERCENTILE:-88.0}"
SEMANTIC_SAMPLE_STRIDE="${SEMANTIC_SAMPLE_STRIDE:-1}"
YOLO_MODEL="${YOLO_MODEL:-}"
YOLO_CONF="${YOLO_CONF:-0.25}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
SEMANTIC_COLOR_OUTPUT="${SEMANTIC_COLOR_OUTPUT:-0}"
DEPTH_HEAD_TRT_ENGINE_WAS_SET="${DEPTH_HEAD_TRT_ENGINE+x}"
DEFAULT_DEPTH_HEAD_TRT_ENGINE="$ROOT_DIR/nuc_output/lingbot_trt/depth_head_224/lingbot_depth_head_224_fp16.engine"
DEPTH_HEAD_TRT_ENGINE="${DEPTH_HEAD_TRT_ENGINE:-$DEFAULT_DEPTH_HEAD_TRT_ENGINE}"
MODEL_PATCH_EMBED="${MODEL_PATCH_EMBED:-}"
MODEL_EMBED_DIM="${MODEL_EMBED_DIM:-0}"
MODEL_DEPTH="${MODEL_DEPTH:-0}"
MODEL_NUM_HEADS="${MODEL_NUM_HEADS:-0}"
MODEL_MLP_RATIO="${MODEL_MLP_RATIO:-0}"
COMPRESS_OUTPUTS="${COMPRESS_OUTPUTS:-0}"
PUBLISH_EVERY_WINDOWS="${PUBLISH_EVERY_WINDOWS:-1}"
PUBLISH_EVERY_FRAMES="${PUBLISH_EVERY_FRAMES:-1}"
ASYNC_ARTIFACT_WRITER="${ASYNC_ARTIFACT_WRITER:-1}"
ARTIFACT_WRITER_MAX_JOBS="${ARTIFACT_WRITER_MAX_JOBS:-4}"
BINARY_CLOUD_WS_PORT="${BINARY_CLOUD_WS_PORT:-0}"
BINARY_CLOUD_WS_HOST="${BINARY_CLOUD_WS_HOST:-0.0.0.0}"
BINARY_CLOUD_MAX_POINTS="${BINARY_CLOUD_MAX_POINTS:-60000}"
GLOBAL_BINARY_CLOUD_WS_PORT="${GLOBAL_BINARY_CLOUD_WS_PORT:-0}"
GLOBAL_BINARY_CLOUD_MAX_POINTS="${GLOBAL_BINARY_CLOUD_MAX_POINTS:-120000}"
ROS2_PUBLISH="${ROS2_PUBLISH:-0}"
ROS2_IMAGE_TOPIC="${ROS2_IMAGE_TOPIC:-/neural_mapping/rgb}"
ROS2_CAMERA_INFO_TOPIC="${ROS2_CAMERA_INFO_TOPIC:-/neural_mapping/camera_info}"
ROS2_POSE_TOPIC="${ROS2_POSE_TOPIC:-/neural_mapping/pose}"
ROS2_PATH_TOPIC="${ROS2_PATH_TOPIC:-/neural_mapping/path}"
ROS2_CLOUD_TOPIC="${ROS2_CLOUD_TOPIC:-/neural_mapping/pointcloud}"
ROS2_PLAIN_CLOUD_TOPIC="${ROS2_PLAIN_CLOUD_TOPIC:-/lingbot/cloud_plain}"
ROS2_CURRENT_CLOUD_TOPIC="${ROS2_CURRENT_CLOUD_TOPIC:-/lingbot/current_cloud_rgb}"
ROS2_CURRENT_PLAIN_CLOUD_TOPIC="${ROS2_CURRENT_PLAIN_CLOUD_TOPIC:-/lingbot/current_cloud_plain}"
ROS2_CAMERA_FRAME_ID="${ROS2_CAMERA_FRAME_ID:-hikrobot_camera}"
ROS2_CLOUD_FRAME_ID="${ROS2_CLOUD_FRAME_ID:-map}"
ROS2_MAX_CLOUD_POINTS="${ROS2_MAX_CLOUD_POINTS:-120000}"
ROS2_MAX_CURRENT_CLOUD_POINTS="${ROS2_MAX_CURRENT_CLOUD_POINTS:-60000}"
ROS2_IMAGE_MAX_WIDTH="${ROS2_IMAGE_MAX_WIDTH:-960}"
ROS2_IMAGE_MAX_HEIGHT="${ROS2_IMAGE_MAX_HEIGHT:-540}"
ROS2_CLOUD_MIN_INTERVAL_SEC="${ROS2_CLOUD_MIN_INTERVAL_SEC:-0.25}"
ROS2_REPUBLISH_CURRENT_CLOUD_ON_IMAGE="${ROS2_REPUBLISH_CURRENT_CLOUD_ON_IMAGE:-1}"
ROS2_CURRENT_CLOUD_REPUBLISH_INTERVAL_SEC="${ROS2_CURRENT_CLOUD_REPUBLISH_INTERVAL_SEC:-1.0}"
ROS2_PATH_MAX_POSES="${ROS2_PATH_MAX_POSES:-1200}"
PORT="${PORT:-19092}"

if [[ -z "$DEPTH_HEAD_TRT_ENGINE_WAS_SET" && ( -n "$MODEL_PATCH_EMBED" || "$MODEL_EMBED_DIM" != "0" || "$MODEL_DEPTH" != "0" ) ]]; then
  # The default TensorRT head was exported from the full LingBot feature shape.
  # Do not attach it automatically to tiny/student backbones.
  DEPTH_HEAD_TRT_ENGINE=""
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing LingBot depth checkpoint: $MODEL_PATH" >&2
  exit 1
fi
if [[ "$TRACKING_BACKEND" != "hikrobot_mono_rgb" && "$TRACKING_BACKEND" != "realsense_mono_rgb" && ! -d "$SEQUENCE_DIR" ]]; then
  echo "Missing sequence dir: $SEQUENCE_DIR" >&2
  exit 1
fi
if [[ "$TRACKING_BACKEND" == "pose_file" && ! -f "$TRAJECTORY_PATH" ]]; then
  echo "Missing trajectory: $TRAJECTORY_PATH" >&2
  exit 1
fi
if [[ "$TRACKING_BACKEND" != "pose_file" && "$TRACKING_BACKEND" != "hikrobot_mono_rgb" && "$TRACKING_BACKEND" != "realsense_mono_rgb" && -z "$RGB_IMAGE_DIR" && -z "$COLOR_IMAGE_DIR" ]]; then
  echo "Tracking backend $TRACKING_BACKEND requires RGB_IMAGE_DIR or COLOR_IMAGE_DIR" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_MAP_ROOT="$LINGBOT_ROOT"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU="${LINGBOT_LOAD_CHECKPOINT_ON_CPU:-1}"
export LINGBOT_CHECKPOINT_MMAP="${LINGBOT_CHECKPOINT_MMAP:-1}"
export LINGBOT_MODEL_DTYPE="${LINGBOT_MODEL_DTYPE:-fp16}"
export LINGBOT_CPU_CAST_BEFORE_CUDA="${LINGBOT_CPU_CAST_BEFORE_CUDA:-1}"
export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-model}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.7}"
if [[ "$COMPILE_LINGBOT_MODEL" == "1" ]]; then
  # PyTorch CUDA graphs used by torch.compile/reduce-overhead are not stable
  # with expandable_segments on this Jetson stack.
  unset PYTORCH_CUDA_ALLOC_CONF
fi
if [[ -z "$USE_SDPA" ]]; then
  if "$PYTHON_BIN" -c "import flashinfer" >/dev/null 2>&1; then
    USE_SDPA=0
  else
    USE_SDPA=1
  fi
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

echo "cuVSLAM + LingBot live reconstruction"
echo "  sequence:   $SEQUENCE_DIR"
if [[ "$TRACKING_BACKEND" == "pose_file" ]]; then
  echo "  trajectory: $TRAJECTORY_PATH"
fi
echo "  tracking:   $TRACKING_BACKEND"
if [[ -n "$RGB_IMAGE_DIR" ]]; then
  echo "  rgb stream: $RGB_IMAGE_DIR"
fi
if [[ -n "$COLOR_IMAGE_DIR" ]]; then
  echo "  color RGB:  $COLOR_IMAGE_DIR ($COLOR_IMAGE_TEMPLATE)"
fi
if [[ "$TRACKING_BACKEND" == "hikrobot_mono_rgb" ]]; then
  echo "  hikrobot:   ${HIKROBOT_WIDTH}x${HIKROBOT_HEIGHT}@${HIKROBOT_FPS}fps exposure=${HIKROBOT_EXPOSURE_US} gain=${HIKROBOT_GAIN}"
fi
if [[ "$TRACKING_BACKEND" == "realsense_mono_rgb" ]]; then
  echo "  realsense:  ${REALSENSE_WIDTH}x${REALSENSE_HEIGHT}@${REALSENSE_FPS}fps serial=${REALSENSE_SERIAL:-auto}"
  echo "              input=${REALSENSE_INPUT_MODE} image=${REALSENSE_IMAGE_TOPIC} info=${REALSENSE_CAMERA_INFO_TOPIC}"
fi
echo "  output:     $OUTPUT_DIR"
echo "  python:     $PYTHON_BIN"
echo "  image size: $IMAGE_SIZE"
echo "  frame step: $FRAME_STEP"
echo "  max frames: $MAX_FRAMES"
echo "  dense every: $DENSE_FRAME_INTERVAL tracked frame(s)"
echo "  window stride: $WINDOW_STRIDE"
echo "  fusion:     $FUSION_MODE voxel=$VOXEL_SIZE max=$FUSION_MAX_POINTS"
echo "  max queue:  $MAX_QUEUE"
echo "  port:       $PORT"

extra_args=()
if [[ -n "$DEPTH_HEAD_TRT_ENGINE" && -f "$DEPTH_HEAD_TRT_ENGINE" ]]; then
  extra_args+=(--depth-head-trt-engine "$DEPTH_HEAD_TRT_ENGINE")
  echo "  TRT head:   $DEPTH_HEAD_TRT_ENGINE"
fi
if [[ -n "$MODEL_PATCH_EMBED" ]]; then
  extra_args+=(--model-patch-embed "$MODEL_PATCH_EMBED")
fi
if [[ "$MODEL_EMBED_DIM" != "0" ]]; then
  extra_args+=(--model-embed-dim "$MODEL_EMBED_DIM")
fi
if [[ "$MODEL_DEPTH" != "0" ]]; then
  extra_args+=(--model-depth "$MODEL_DEPTH")
fi
if [[ "$MODEL_NUM_HEADS" != "0" ]]; then
  extra_args+=(--model-num-heads "$MODEL_NUM_HEADS")
fi
if [[ "$MODEL_MLP_RATIO" != "0" ]]; then
  extra_args+=(--model-mlp-ratio "$MODEL_MLP_RATIO")
fi
if [[ -n "$MODEL_PATCH_EMBED" || "$MODEL_EMBED_DIM" != "0" || "$MODEL_DEPTH" != "0" ]]; then
  echo "  direct model: patch=$MODEL_PATCH_EMBED embed=$MODEL_EMBED_DIM depth=$MODEL_DEPTH heads=$MODEL_NUM_HEADS mlp=$MODEL_MLP_RATIO"
fi
if [[ "$COMPRESS_OUTPUTS" == "0" ]]; then
  extra_args+=(--no-compress-output)
  echo "  output npz: uncompressed"
fi
if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  extra_args+=(--clean-output)
  echo "  output: cleaning stale worker/live_map artifacts"
fi
if [[ "$KEYFRAMES_ONLY" == "1" ]]; then
  extra_args+=(--keyframes-only)
  echo "  dense policy: cuVSLAM keyframes only"
fi
if [[ "$DROP_WHEN_BUSY" == "1" ]]; then
  extra_args+=(--drop-when-busy)
  echo "  dense policy: drop when worker queue is full"
fi
if [[ "$USE_SDPA" == "1" ]]; then
  extra_args+=(--use-sdpa)
  echo "  attention:  PyTorch SDPA fallback"
fi
if [[ "$OFFLOAD_TO_CPU" == "0" ]]; then
  extra_args+=(--no-offload-to-cpu)
  echo "  inference:  keep predictions on GPU until postprocess"
else
  extra_args+=(--offload-to-cpu)
fi
if [[ "$PRELOAD_LINGBOT_MODEL" == "1" ]]; then
  extra_args+=(--preload-lingbot-model)
  echo "  model:      preload before dense queue consumption"
fi
if [[ "$WARMUP_FIRST_WINDOW" == "1" ]]; then
  extra_args+=(--warmup-first-window)
  echo "  model:      warm up first dense window"
fi
if [[ "$COMPILE_LINGBOT_MODEL" == "1" ]]; then
  extra_args+=(
    --compile-lingbot-model
    --compile-warmup-passes "$COMPILE_WARMUP_PASSES"
    --compile-warmup-stream-frames "$COMPILE_WARMUP_STREAM_FRAMES"
  )
  echo "  model:      torch.compile warmup passes=$COMPILE_WARMUP_PASSES stream_frames=$COMPILE_WARMUP_STREAM_FRAMES"
fi
if [[ "$LINGBOT_PERSISTENT_STREAMING" == "1" ]]; then
  extra_args+=(--persistent-lingbot-streaming)
  echo "  model:      persistent streaming KV cache across dense windows"
fi
if [[ "$DENSE_SUBMIT_WHEN_WORKER_IDLE" == "1" ]]; then
  extra_args+=(--dense-submit-when-worker-idle)
  echo "  dense policy: submit only when worker is idle"
fi
if [[ "$PAUSE_TRACKING_WHILE_DENSE" == "1" ]]; then
  extra_args+=(--pause-tracking-while-dense)
  echo "  tracking:   pause cuVSLAM while dense worker is active"
fi
if [[ "$DENSE_BUSY_TRACKING_POLICY" != "none" ]]; then
  extra_args+=(
    --dense-busy-tracking-policy "$DENSE_BUSY_TRACKING_POLICY"
    --dense-busy-tracking-min-interval-sec "$DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC"
  )
  echo "  tracking:   dense-busy policy=$DENSE_BUSY_TRACKING_POLICY min_interval=${DENSE_BUSY_TRACKING_MIN_INTERVAL_SEC}s"
fi
if [[ "$ADAPTIVE_SAMPLING" == "1" ]]; then
  extra_args+=(--adaptive-sampling)
  echo "  sampling: adaptive near=$NEAR_SAMPLE_STRIDE edge=$EDGE_SAMPLE_STRIDE semantic=$SEMANTIC_SAMPLE_STRIDE"
fi
if [[ "$ROLLING_MAP" == "1" ]]; then
  extra_args+=(--rolling-map)
  echo "  map:        rolling local map voxel=$ROLLING_MAP_VOXEL_SIZE max_windows=$ROLLING_MAP_MAX_WINDOWS"
fi
if [[ "$GLOBAL_MAP" == "1" ]]; then
  extra_args+=(
    --global-map
    --global-map-voxel-size "$GLOBAL_MAP_VOXEL_SIZE"
    --global-map-radius-m "$GLOBAL_MAP_RADIUS_M"
    --global-map-min-neighbors "$GLOBAL_MAP_MIN_NEIGHBORS"
    --global-map-max-points "$GLOBAL_MAP_MAX_POINTS"
  )
  echo "  map:        persistent global map voxel=$GLOBAL_MAP_VOXEL_SIZE max_points=$GLOBAL_MAP_MAX_POINTS"
fi
if [[ "$HIKROBOT_THREADED_CAPTURE" == "1" ]]; then
  extra_args+=(--hikrobot-threaded-capture)
fi
if [[ "$HIKROBOT_ASYNC_TRACKING" == "1" ]]; then
  extra_args+=(--hikrobot-async-tracking)
  echo "  tracking:   async HikRobot tracking idle=${HIKROBOT_TRACKING_IDLE_FPS}fps dense=${HIKROBOT_TRACKING_DENSE_FPS}fps"
fi
if [[ "$HIKROBOT_DISABLE_CUVSLAM" == "1" ]]; then
  extra_args+=(--hikrobot-disable-cuvslam)
  echo "  tracking:   HikRobot cuVSLAM disabled; using lightweight OpenCV pose source"
fi
if [[ "$REALSENSE_THREADED_CAPTURE" == "1" ]]; then
  extra_args+=(--realsense-threaded-capture)
fi
if [[ "$REALSENSE_ASYNC_TRACKING" == "1" ]]; then
  extra_args+=(--realsense-async-tracking)
  echo "  tracking:   async RealSense tracking idle=${REALSENSE_TRACKING_IDLE_FPS}fps dense=${REALSENSE_TRACKING_DENSE_FPS}fps"
fi
if [[ "$REALSENSE_DISABLE_CUVSLAM" == "1" ]]; then
  extra_args+=(--realsense-disable-cuvslam)
  echo "  tracking:   RealSense cuVSLAM disabled; using lightweight OpenCV pose source"
fi
if [[ -n "$CAMERA_DISTORTION_COEFFS" ]]; then
  extra_args+=(--camera-distortion-coeffs "$CAMERA_DISTORTION_COEFFS")
  echo "  camera:     calibrated distortion coeffs provided"
fi
if [[ "$CAMERA_UNDISTORT" == "1" ]]; then
  extra_args+=(--camera-undistort)
  echo "  camera:     undistort frames before tracking/dense"
fi
if [[ "$LINGBOT_ENABLE_CAMERA" == "1" ]]; then
  extra_args+=(--lingbot-enable-camera)
  echo "  model:      LingBot camera/pose head enabled"
fi
if [[ "$PREFER_LINGBOT_POSE" == "1" ]]; then
  extra_args+=(--prefer-lingbot-pose)
  echo "  fusion:     prefer LingBot predicted pose for dense geometry"
fi
if [[ -n "$YOLO_MODEL" ]]; then
  extra_args+=(--yolo-model "$YOLO_MODEL" --yolo-conf "$YOLO_CONF" --yolo-imgsz "$YOLO_IMGSZ")
  echo "  YOLO:       $YOLO_MODEL conf=$YOLO_CONF imgsz=$YOLO_IMGSZ"
fi
if [[ "$SEMANTIC_COLOR_OUTPUT" == "1" ]]; then
  extra_args+=(--semantic-color-output)
  echo "  color:      semantic labels override RGB when available"
fi
if [[ "$ROS2_PUBLISH" == "1" ]]; then
  extra_args+=(
    --ros2-publish
    --ros2-image-topic "$ROS2_IMAGE_TOPIC"
    --ros2-camera-info-topic "$ROS2_CAMERA_INFO_TOPIC"
    --ros2-pose-topic "$ROS2_POSE_TOPIC"
    --ros2-path-topic "$ROS2_PATH_TOPIC"
    --ros2-cloud-topic "$ROS2_CLOUD_TOPIC"
    --ros2-plain-cloud-topic "$ROS2_PLAIN_CLOUD_TOPIC"
    --ros2-current-cloud-topic "$ROS2_CURRENT_CLOUD_TOPIC"
    --ros2-current-plain-cloud-topic "$ROS2_CURRENT_PLAIN_CLOUD_TOPIC"
    --ros2-camera-frame-id "$ROS2_CAMERA_FRAME_ID"
    --ros2-cloud-frame-id "$ROS2_CLOUD_FRAME_ID"
    --ros2-max-cloud-points "$ROS2_MAX_CLOUD_POINTS"
    --ros2-max-current-cloud-points "$ROS2_MAX_CURRENT_CLOUD_POINTS"
    --ros2-image-max-width "$ROS2_IMAGE_MAX_WIDTH"
    --ros2-image-max-height "$ROS2_IMAGE_MAX_HEIGHT"
    --ros2-cloud-min-interval-sec "$ROS2_CLOUD_MIN_INTERVAL_SEC"
    --ros2-current-cloud-republish-interval-sec "$ROS2_CURRENT_CLOUD_REPUBLISH_INTERVAL_SEC"
    --ros2-path-max-poses "$ROS2_PATH_MAX_POSES"
  )
  if [[ "$ROS2_REPUBLISH_CURRENT_CLOUD_ON_IMAGE" == "1" ]]; then
    extra_args+=(--ros2-republish-current-cloud-on-image)
  fi
  echo "  ros2:       publish rgb/pose/path/cloud for GS Console"
fi
if [[ "$ASYNC_ARTIFACT_WRITER" == "1" ]]; then
  extra_args+=(--async-artifact-writer --artifact-writer-max-jobs "$ARTIFACT_WRITER_MAX_JOBS")
else
  extra_args+=(--no-async-artifact-writer)
fi
if [[ "$BINARY_CLOUD_WS_PORT" != "0" ]]; then
  extra_args+=(
    --binary-cloud-ws-port "$BINARY_CLOUD_WS_PORT"
    --binary-cloud-ws-host "$BINARY_CLOUD_WS_HOST"
    --binary-cloud-max-points "$BINARY_CLOUD_MAX_POINTS"
  )
  echo "  binary ws:  ws://0.0.0.0:$BINARY_CLOUD_WS_PORT/cloud max_points=$BINARY_CLOUD_MAX_POINTS"
fi
if [[ "$GLOBAL_BINARY_CLOUD_WS_PORT" != "0" ]]; then
  extra_args+=(
    --global-binary-cloud-ws-port "$GLOBAL_BINARY_CLOUD_WS_PORT"
    --global-binary-cloud-max-points "$GLOBAL_BINARY_CLOUD_MAX_POINTS"
  )
  echo "  global ws:  ws://0.0.0.0:$GLOBAL_BINARY_CLOUD_WS_PORT/cloud max_points=$GLOBAL_BINARY_CLOUD_MAX_POINTS"
fi

"$PYTHON_BIN" HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py \
  --sequence-dir "$SEQUENCE_DIR" \
  --trajectory-path "$TRAJECTORY_PATH" \
  --tracking-backend "$TRACKING_BACKEND" \
  --rgb-image-dir "$RGB_IMAGE_DIR" \
  --intrinsic-camera-index "$INTRINSIC_CAMERA_INDEX" \
  --mono-fixed-step-scale "$MONO_FIXED_STEP_SCALE" \
  --mono-scale-source "$MONO_SCALE_SOURCE" \
  --oxts-dir "$OXTS_DIR" \
  --hikrobot-index "$HIKROBOT_INDEX" \
  --hikrobot-timeout-ms "$HIKROBOT_TIMEOUT_MS" \
  --hikrobot-exposure-us "$HIKROBOT_EXPOSURE_US" \
  --hikrobot-gain "$HIKROBOT_GAIN" \
  --hikrobot-fps "$HIKROBOT_FPS" \
  --hikrobot-width "$HIKROBOT_WIDTH" \
  --hikrobot-height "$HIKROBOT_HEIGHT" \
  --hikrobot-capture-queue-size "$HIKROBOT_CAPTURE_QUEUE_SIZE" \
  --hikrobot-tracking-queue-size "$HIKROBOT_TRACKING_QUEUE_SIZE" \
  --hikrobot-tracking-idle-fps "$HIKROBOT_TRACKING_IDLE_FPS" \
  --hikrobot-tracking-dense-fps "$HIKROBOT_TRACKING_DENSE_FPS" \
  --hikrobot-max-read-errors "$HIKROBOT_MAX_READ_ERRORS" \
  --realsense-index "$REALSENSE_INDEX" \
  --realsense-input-mode "$REALSENSE_INPUT_MODE" \
  --realsense-serial "$REALSENSE_SERIAL" \
  --realsense-image-topic "$REALSENSE_IMAGE_TOPIC" \
  --realsense-camera-info-topic "$REALSENSE_CAMERA_INFO_TOPIC" \
  --realsense-timeout-ms "$REALSENSE_TIMEOUT_MS" \
  --realsense-fps "$REALSENSE_FPS" \
  --realsense-width "$REALSENSE_WIDTH" \
  --realsense-height "$REALSENSE_HEIGHT" \
  --realsense-capture-queue-size "$REALSENSE_CAPTURE_QUEUE_SIZE" \
  --realsense-tracking-queue-size "$REALSENSE_TRACKING_QUEUE_SIZE" \
  --realsense-tracking-idle-fps "$REALSENSE_TRACKING_IDLE_FPS" \
  --realsense-tracking-dense-fps "$REALSENSE_TRACKING_DENSE_FPS" \
  --realsense-max-read-errors "$REALSENSE_MAX_READ_ERRORS" \
  --camera-fx "$CAMERA_FX" \
  --camera-fy "$CAMERA_FY" \
  --camera-cx "$CAMERA_CX" \
  --camera-cy "$CAMERA_CY" \
  --rgb-output-dir "$RGB_OUTPUT_DIR" \
  --color-image-dir "$COLOR_IMAGE_DIR" \
  --color-image-template "$COLOR_IMAGE_TEMPLATE" \
  --model-path "$MODEL_PATH" \
  --lingbot-map-root "$LINGBOT_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --model-image-size "$MODEL_IMAGE_SIZE" \
  --frame-step "$FRAME_STEP" \
  --max-frames "$MAX_FRAMES" \
  --window-size "$WINDOW_SIZE" \
  --stride "$WINDOW_STRIDE" \
  --dense-frame-interval "$DENSE_FRAME_INTERVAL" \
  --dense-scheduler "$DENSE_SCHEDULER" \
  --dense-min-frame-gap "$DENSE_MIN_FRAME_GAP" \
  --dense-translation-thresh-m "$DENSE_TRANSLATION_THRESH_M" \
  --dense-rotation-thresh-deg "$DENSE_ROTATION_THRESH_DEG" \
  --dense-pixel-motion-thresh "$DENSE_PIXEL_MOTION_THRESH" \
  --max-queue "$MAX_QUEUE" \
  --frame-sleep-sec "$FRAME_SLEEP_SEC" \
  --depth-scale "$DEPTH_SCALE" \
  --lingbot-pose-translation-scale "$LINGBOT_POSE_TRANSLATION_SCALE" \
  --lingbot-extrinsic-mode "$LINGBOT_EXTRINSIC_MODE" \
  --sample-stride "$SAMPLE_STRIDE" \
  --sampling-pattern "$SAMPLING_PATTERN" \
  --max-points-per-frame "$MAX_POINTS_PER_FRAME" \
  --max-active-frames "$MAX_ACTIVE_FRAMES" \
  --fusion-mode "$FUSION_MODE" \
  --voxel-size "$VOXEL_SIZE" \
  --fusion-max-points "$FUSION_MAX_POINTS" \
  --fusion-min-observations "$FUSION_MIN_OBSERVATIONS" \
  --rolling-map-voxel-size "$ROLLING_MAP_VOXEL_SIZE" \
  --rolling-map-radius-m "$ROLLING_MAP_RADIUS_M" \
  --rolling-map-min-neighbors "$ROLLING_MAP_MIN_NEIGHBORS" \
  --rolling-map-max-windows "$ROLLING_MAP_MAX_WINDOWS" \
  --rolling-map-max-age-sec "$ROLLING_MAP_MAX_AGE_SEC" \
  --rolling-map-max-points "$ROLLING_MAP_MAX_POINTS" \
  --keyframe-translation-thresh-m "$KEYFRAME_TRANSLATION_THRESH_M" \
  --keyframe-rotation-thresh-deg "$KEYFRAME_ROTATION_THRESH_DEG" \
  --keyframe-time-thresh-sec "$KEYFRAME_TIME_THRESH_SEC" \
  --keyframe-max-count "$KEYFRAME_MAX_COUNT" \
  --near-depth-m "$NEAR_DEPTH_M" \
  --near-sample-stride "$NEAR_SAMPLE_STRIDE" \
  --edge-sample-stride "$EDGE_SAMPLE_STRIDE" \
  --edge-percentile "$EDGE_PERCENTILE" \
  --semantic-sample-stride "$SEMANTIC_SAMPLE_STRIDE" \
  --publish-every-windows "$PUBLISH_EVERY_WINDOWS" \
  --publish-every-frames "$PUBLISH_EVERY_FRAMES" \
  "${extra_args[@]}" \
  --serve \
  --port "$PORT"
