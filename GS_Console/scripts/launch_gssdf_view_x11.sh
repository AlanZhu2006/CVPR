#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
OUTPUT_DIR="${1:-$ROOT_DIR/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml}"
CONTAINER_NAME="${2:-gssdf-view-quality}"
DISPLAY_VALUE="${DISPLAY_VALUE:-:88}"

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "Output dir does not exist: $OUTPUT_DIR" >&2
  exit 1
fi

if [[ ! -f "$OUTPUT_DIR/model/local_map_checkpoint.pt" ]]; then
  echo "Checkpoint missing under $OUTPUT_DIR/model/local_map_checkpoint.pt" >&2
  exit 1
fi

if ! xdpyinfo -display "$DISPLAY_VALUE" >/dev/null 2>&1; then
  echo "DISPLAY=$DISPLAY_VALUE is not reachable from the host." >&2
  echo "Start Xvfb or a real X server first." >&2
  exit 1
fi

OUTPUT_ABS="$(realpath "$OUTPUT_DIR")"
OUTPUT_PARENT="$(dirname "$OUTPUT_ABS")"
CONFIG_PATH="$OUTPUT_ABS/model/config/scene/config.yaml"
DATA_PATH=""
DATA_MOUNT_ARGS=()
MODEL_DIR="$OUTPUT_ABS/model"
COMPAT_MODEL_DIR="$MODEL_DIR/model"
VIEW_WIDTH="${VIEW_WIDTH:-}"
VIEW_HEIGHT="${VIEW_HEIGHT:-}"
VIEW_SCALE="${VIEW_SCALE:-}"
KEEP_CONTAINER_ON_EXIT="${KEEP_CONTAINER_ON_EXIT:-0}"

if [[ -f "$CONFIG_PATH" ]]; then
  DATA_PATH="$(sed -n 's/^data_path: "\(.*\)"/\1/p' "$CONFIG_PATH" | head -n 1)"
fi

if [[ -n "$DATA_PATH" ]]; then
  if [[ ! -e "$DATA_PATH" ]]; then
    echo "Saved view config refers to missing data_path: $DATA_PATH" >&2
    exit 1
  fi
  DATA_PARENT="$(dirname "$(realpath "$DATA_PATH")")"
  DATA_MOUNT_ARGS=(-v "$DATA_PARENT:$DATA_PARENT")
fi

if [[ -n "$VIEW_WIDTH" || -n "$VIEW_HEIGHT" || -n "$VIEW_SCALE" ]]; then
  RESOLUTION_ARGS=(--scene-config "$CONFIG_PATH")
  if [[ -n "$VIEW_WIDTH" ]]; then
    RESOLUTION_ARGS+=(--width "$VIEW_WIDTH")
  fi
  if [[ -n "$VIEW_HEIGHT" ]]; then
    RESOLUTION_ARGS+=(--height "$VIEW_HEIGHT")
  fi
  if [[ -n "$VIEW_SCALE" ]]; then
    RESOLUTION_ARGS+=(--scale "$VIEW_SCALE")
  fi

  echo "Applying GS-SDF view resolution override..."
  docker run --rm \
    -v "$ROOT_DIR:$ROOT_DIR" \
    -v "$OUTPUT_PARENT:$OUTPUT_PARENT" \
    "$IMAGE" \
    python3 "$ROOT_DIR/scripts/override_gssdf_view_resolution.py" "${RESOLUTION_ARGS[@]}"
fi

docker run --rm \
  -v "$OUTPUT_PARENT:$OUTPUT_PARENT" \
  "$IMAGE" \
  bash -lc "set -euo pipefail; \
    mkdir -p \"$COMPAT_MODEL_DIR\"; \
    for asset in as_occ_prior.ply gs.ply local_map_checkpoint.pt; do \
      if [[ -f \"$MODEL_DIR/\$asset\" && ! -e \"$COMPAT_MODEL_DIR/\$asset\" ]]; then \
        ln -s \"../\$asset\" \"$COMPAT_MODEL_DIR/\$asset\"; \
      fi; \
    done"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "Image:      $IMAGE"
echo "Output dir: $OUTPUT_ABS"
echo "Container:  $CONTAINER_NAME"
echo "Display:    $DISPLAY_VALUE"

DOCKER_RM_FLAG="--rm"
if [[ "$KEEP_CONTAINER_ON_EXIT" == "1" ]]; then
  DOCKER_RM_FLAG=""
fi

docker run -d $DOCKER_RM_FLAG \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=16g \
  -e DISPLAY="$DISPLAY_VALUE" \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e MESA_LOADER_DRIVER_OVERRIDE=swrast \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$OUTPUT_PARENT:$OUTPUT_PARENT" \
  "${DATA_MOUNT_ARGS[@]}" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    source /root/gs_sdf_ws/devel/setup.bash; \
    roscore >/tmp/roscore.log 2>&1 & \
    ROSCORE_PID=\$!; \
    trap 'kill \$ROSCORE_PID >/dev/null 2>&1 || true' EXIT; \
    for _ in \$(seq 1 60); do \
      if rosparam list >/dev/null 2>&1; then \
        break; \
      fi; \
      sleep 0.5; \
    done; \
    rosparam list >/dev/null 2>&1; \
    /root/gs_sdf_ws/devel/lib/neural_mapping/neural_mapping_node view \"$OUTPUT_ABS\""

echo "Started $CONTAINER_NAME"
echo "Tail logs with: docker logs -f $CONTAINER_NAME"
