#!/usr/bin/env bash
set -eo pipefail

WS="/home/nyu/Codespace/CVPR/third_party_research/fastlivo2_ros2_ws"
REPO="/home/nyu/Codespace/CVPR/third_party_research/FASTLIVO2_ROS2"
BAG="/home/nyu/Codespace/CVPR/datasets/ntu_viral/spms_02_fastlivo2_ros2"
OUTDIR="/home/nyu/Codespace/CVPR/nuc_output/ntuviral_spms02_fastlivo2"
POSE_SRC="$REPO/Log/result/spms_02_ntu_viral.txt"
POSE_DST="$OUTDIR/fastlivo2_tum.txt"

mkdir -p "$REPO/Log/result" "$OUTDIR"
rm -f "$POSE_SRC" "$POSE_DST" "$OUTDIR/run_launch.log" "$OUTDIR/run_bagplay.log"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ros2 launch fast_livo mapping_ntu_viral.launch.py use_rviz:=False use_respawn:=False \
  > "$OUTDIR/run_launch.log" 2>&1 &
LAUNCH_PID=$!

cleanup() {
  kill "$LAUNCH_PID" >/dev/null 2>&1 || true
  wait "$LAUNCH_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 5

ros2 bag play "$BAG" --rate 1.0 > "$OUTDIR/run_bagplay.log" 2>&1

sleep 5

if [ ! -f "$POSE_SRC" ]; then
  echo "FAST-LIVO2 pose file not found: $POSE_SRC" >&2
  exit 1
fi

cp "$POSE_SRC" "$POSE_DST"
wc -l "$POSE_DST"
echo "Saved trajectory to $POSE_DST"
