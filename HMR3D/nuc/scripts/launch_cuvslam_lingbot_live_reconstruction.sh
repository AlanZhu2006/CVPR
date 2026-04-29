#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/cuVSLAM/.venv-jetson/bin/python}"
SEQUENCE_DIR="${SEQUENCE_DIR:-$ROOT_DIR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-$ROOT_DIR/nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_tum.txt}"
TRACKING_BACKEND="${TRACKING_BACKEND:-pose_file}"
RGB_IMAGE_DIR="${RGB_IMAGE_DIR:-}"
INTRINSIC_CAMERA_INDEX="${INTRINSIC_CAMERA_INDEX:-0}"
MONO_FIXED_STEP_SCALE="${MONO_FIXED_STEP_SCALE:-0.5}"
MONO_SCALE_SOURCE="${MONO_SCALE_SOURCE:-fixed}"
OXTS_DIR="${OXTS_DIR:-}"
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
KEYFRAMES_ONLY="${KEYFRAMES_ONLY:-0}"
MAX_QUEUE="${MAX_QUEUE:-2}"
DROP_WHEN_BUSY="${DROP_WHEN_BUSY:-0}"
FRAME_SLEEP_SEC="${FRAME_SLEEP_SEC:-0}"
DEPTH_SCALE="${DEPTH_SCALE:-20.0}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-8}"
MAX_POINTS_PER_FRAME="${MAX_POINTS_PER_FRAME:-2500}"
MAX_ACTIVE_FRAMES="${MAX_ACTIVE_FRAMES:-16}"
FUSION_MODE="${FUSION_MODE:-raw}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
FUSION_MAX_POINTS="${FUSION_MAX_POINTS:-500000}"
FUSION_MIN_OBSERVATIONS="${FUSION_MIN_OBSERVATIONS:-1}"
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
if [[ ! -d "$SEQUENCE_DIR" ]]; then
  echo "Missing sequence dir: $SEQUENCE_DIR" >&2
  exit 1
fi
if [[ "$TRACKING_BACKEND" == "pose_file" && ! -f "$TRAJECTORY_PATH" ]]; then
  echo "Missing trajectory: $TRAJECTORY_PATH" >&2
  exit 1
fi
if [[ "$TRACKING_BACKEND" != "pose_file" && -z "$RGB_IMAGE_DIR" && -z "$COLOR_IMAGE_DIR" ]]; then
  echo "Tracking backend $TRACKING_BACKEND requires RGB_IMAGE_DIR or COLOR_IMAGE_DIR" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/HMR3D/nuc/src${PYTHONPATH:+:$PYTHONPATH}"
export LINGBOT_MAP_ROOT="$LINGBOT_ROOT"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=0
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=aggregator
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
echo "  output:     $OUTPUT_DIR"
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
if [[ "$KEYFRAMES_ONLY" == "1" ]]; then
  extra_args+=(--keyframes-only)
  echo "  dense policy: cuVSLAM keyframes only"
fi
if [[ "$DROP_WHEN_BUSY" == "1" ]]; then
  extra_args+=(--drop-when-busy)
  echo "  dense policy: drop when worker queue is full"
fi
if [[ "$ADAPTIVE_SAMPLING" == "1" ]]; then
  extra_args+=(--adaptive-sampling)
  echo "  sampling: adaptive near=$NEAR_SAMPLE_STRIDE edge=$EDGE_SAMPLE_STRIDE semantic=$SEMANTIC_SAMPLE_STRIDE"
fi
if [[ -n "$YOLO_MODEL" ]]; then
  extra_args+=(--yolo-model "$YOLO_MODEL" --yolo-conf "$YOLO_CONF" --yolo-imgsz "$YOLO_IMGSZ")
  echo "  YOLO:       $YOLO_MODEL conf=$YOLO_CONF imgsz=$YOLO_IMGSZ"
fi
if [[ "$SEMANTIC_COLOR_OUTPUT" == "1" ]]; then
  extra_args+=(--semantic-color-output)
  echo "  color:      semantic labels override RGB when available"
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
  --max-queue "$MAX_QUEUE" \
  --frame-sleep-sec "$FRAME_SLEEP_SEC" \
  --depth-scale "$DEPTH_SCALE" \
  --sample-stride "$SAMPLE_STRIDE" \
  --max-points-per-frame "$MAX_POINTS_PER_FRAME" \
  --max-active-frames "$MAX_ACTIVE_FRAMES" \
  --fusion-mode "$FUSION_MODE" \
  --voxel-size "$VOXEL_SIZE" \
  --fusion-max-points "$FUSION_MAX_POINTS" \
  --fusion-min-observations "$FUSION_MIN_OBSERVATIONS" \
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
