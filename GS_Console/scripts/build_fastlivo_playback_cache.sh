#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${GS_SDF_IMAGE:-gs_sdf_img:latest}"
BAG_PATH="${1:-/home/chatsign/fast_livo2_compressed.bag}"
CACHE_DIR="${2:-$ROOT_DIR/runtime/playback-cache/fast_livo2_compressed}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/fastlivo_cbd_host.yaml}"
SCAN_MAX_POINTS="${SCAN_MAX_POINTS:-3600}"
SCAN_VOXEL_SIZE="${SCAN_VOXEL_SIZE:-0.12}"
MAP_MAX_POINTS="${MAP_MAX_POINTS:-90000}"
MAP_VOXEL_SIZE="${MAP_VOXEL_SIZE:-0.20}"
WEB_MIN_RANGE="${WEB_MIN_RANGE:-1.0}"
WEB_MAX_RANGE="${WEB_MAX_RANGE:-24.0}"
WEB_MAX_ABS_Z="${WEB_MAX_ABS_Z:-8.0}"
IMAGE_SYNC_TOLERANCE="${IMAGE_SYNC_TOLERANCE:-0.2}"
POSE_SYNC_TOLERANCE="${POSE_SYNC_TOLERANCE:-0.12}"
KEYFRAME_EVERY="${KEYFRAME_EVERY:-64}"
JPEG_QUALITY="${JPEG_QUALITY:-82}"
MAX_FRAMES="${MAX_FRAMES:-0}"

if [[ ! -f "$BAG_PATH" ]]; then
  echo "Bag file does not exist: $BAG_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file does not exist: $CONFIG_PATH" >&2
  exit 1
fi

abspath_preserve() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

BAG_ABS="$(abspath_preserve "$BAG_PATH")"
BAG_PARENT="$(dirname "$BAG_ABS")"
CONFIG_ABS="$(abspath_preserve "$CONFIG_PATH")"
CACHE_ABS="$(abspath_preserve "$CACHE_DIR")"
CACHE_PARENT="$(dirname "$CACHE_ABS")"
mkdir -p "$CACHE_PARENT"

echo "Image:         $IMAGE"
echo "Bag:           $BAG_ABS"
echo "Config:        $CONFIG_ABS"
echo "Cache dir:     $CACHE_ABS"
echo "Scan max pts:  $SCAN_MAX_POINTS"
echo "Map max pts:   $MAP_MAX_POINTS"
echo "Keyframe every:$KEYFRAME_EVERY"
echo "Max frames:    $MAX_FRAMES"

docker run --rm \
  -v "$ROOT_DIR:$ROOT_DIR" \
  -v "$BAG_PARENT:$BAG_PARENT" \
  -v "$CACHE_PARENT:$CACHE_PARENT" \
  "$IMAGE" \
  bash -lc "set -eo pipefail; \
    source /opt/ros/noetic/setup.bash; \
    python3 '$ROOT_DIR/scripts/build_fastlivo_playback_cache.py' \
      --bag-path '$BAG_ABS' \
      --config '$CONFIG_ABS' \
      --output-dir '$CACHE_ABS' \
      --scan-max-points '$SCAN_MAX_POINTS' \
      --scan-voxel-size '$SCAN_VOXEL_SIZE' \
      --map-max-points '$MAP_MAX_POINTS' \
      --map-voxel-size '$MAP_VOXEL_SIZE' \
      --web-min-range '$WEB_MIN_RANGE' \
      --web-max-range '$WEB_MAX_RANGE' \
      --web-max-abs-z '$WEB_MAX_ABS_Z' \
      --image-sync-tolerance '$IMAGE_SYNC_TOLERANCE' \
      --pose-sync-tolerance '$POSE_SYNC_TOLERANCE' \
      --jpeg-quality '$JPEG_QUALITY' \
      --keyframe-every '$KEYFRAME_EVERY' \
      --max-frames '$MAX_FRAMES' \
      --overwrite"
