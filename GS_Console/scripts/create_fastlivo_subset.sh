#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ROS_BAG_IMAGE:-osrf/ros:noetic-desktop-full}"
INPUT_BAG="${1:-}"
START_OFFSET_SEC="${2:-}"
DURATION_SEC="${3:-}"
OUTPUT_BAG="${4:-}"

if [[ -z "$INPUT_BAG" || -z "$START_OFFSET_SEC" || -z "$DURATION_SEC" ]]; then
  echo "Usage: $0 <input.bag> <start-offset-sec> <duration-sec> [output.bag]" >&2
  echo "Example: $0 /home/chatsign/fast_livo2_compressed.bag 30 45" >&2
  exit 1
fi

if [[ ! -f "$INPUT_BAG" ]]; then
  echo "Input bag does not exist: $INPUT_BAG" >&2
  exit 1
fi

if ! [[ "$START_OFFSET_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "start-offset-sec must be a positive number" >&2
  exit 1
fi

if ! [[ "$DURATION_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "duration-sec must be a positive number" >&2
  exit 1
fi

INPUT_ABS="$(realpath "$INPUT_BAG")"
INPUT_DIR="$(dirname "$INPUT_ABS")"
INPUT_NAME="$(basename "$INPUT_ABS" .bag)"

if [[ -z "$OUTPUT_BAG" ]]; then
  OUTPUT_DIR="$ROOT_DIR/runtime/bags"
  mkdir -p "$OUTPUT_DIR"
  START_TAG="${START_OFFSET_SEC//./p}"
  DURATION_TAG="${DURATION_SEC//./p}"
  OUTPUT_BAG="$OUTPUT_DIR/${INPUT_NAME}_from_${START_TAG}s_len_${DURATION_TAG}s.bag"
fi

OUTPUT_ABS="$(realpath -m "$OUTPUT_BAG")"
OUTPUT_DIR="$(dirname "$OUTPUT_ABS")"
mkdir -p "$OUTPUT_DIR"

echo "Image:        $IMAGE"
echo "Input bag:    $INPUT_ABS"
echo "Start offset: ${START_OFFSET_SEC}s"
echo "Duration:     ${DURATION_SEC}s"
echo "Output bag:   $OUTPUT_ABS"

docker run --rm \
  -v "$INPUT_DIR:$INPUT_DIR" \
  -v "$OUTPUT_DIR:$OUTPUT_DIR" \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    source /opt/ros/noetic/setup.bash

    START_TS=\$(rosbag info --yaml '$INPUT_ABS' | python3 -c '
import sys, yaml
data = yaml.safe_load(sys.stdin.read())
print(data[\"start\"])
')
    END_TS=\$(python3 -c \"start=float('$START_OFFSET_SEC'); duration=float('$DURATION_SEC'); base=float('\$START_TS'); print(base + start + duration)\")
    SUBSET_START_TS=\$(python3 -c \"start=float('$START_OFFSET_SEC'); base=float('\$START_TS'); print(base + start)\")

    echo \"Bag start:    \$START_TS\"
    echo \"Subset start: \$SUBSET_START_TS\"
    echo \"Subset end:   \$END_TS\"

    rm -f '$OUTPUT_ABS'
    rosbag filter '$INPUT_ABS' '$OUTPUT_ABS' \"t.to_sec() >= float(\$SUBSET_START_TS) and t.to_sec() <= float(\$END_TS)\"
    echo
    rosbag info '$OUTPUT_ABS'
  "
