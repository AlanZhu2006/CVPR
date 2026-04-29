#!/usr/bin/env bash
set -euo pipefail

SRC_BAG="${1:-/home/nyu/Codespace/CVPR/datasets/ntu_viral/spms_02/spms_02/spms_02.bag}"
DST_BAG="${2:-/home/nyu/Codespace/CVPR/datasets/ntu_viral/spms_02_fastlivo2_ros2}"

mkdir -p "$(dirname "$DST_BAG")"

echo "[convert] src: $SRC_BAG"
echo "[convert] dst: $DST_BAG"

/home/nyu/.local/bin/rosbags-convert \
  --src "$SRC_BAG" \
  --dst "$DST_BAG" \
  --src-typestore ros1_noetic \
  --dst-typestore ros2_humble \
  --dst-storage sqlite3 \
  --include-topic /left/image_raw /imu/imu /os1_cloud_node1/points

echo "[convert] done"
