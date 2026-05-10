#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/runtime/output/2026-04-02-23-01-07_fast_livo2_compressed.bag_fastlivo_cbd_host.yaml}"
VIEW_CONTAINER="${2:-gssdf-view}"
WAIT_SECONDS="${WAIT_SECONDS:-90}"
export RVIZ_CONFIG="${RVIZ_CONFIG:-$ROOT_DIR/rviz/neural_mapping_saved_view.rviz}"

bash "$ROOT_DIR/scripts/launch_gssdf_view.sh" "$OUTPUT_DIR" "$VIEW_CONTAINER"

READY=0
for ((i=0; i<WAIT_SECONDS; i++)); do
  if docker exec "$VIEW_CONTAINER" bash -lc "source /opt/ros/noetic/setup.bash && rostopic list | grep -q '^/neural_mapping/path$'" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "$READY" -ne 1 ]]; then
  echo "Neural mapping topics in $VIEW_CONTAINER did not become ready within ${WAIT_SECONDS}s" >&2
  exit 1
fi

if [[ "${LAUNCH_CAMERA_POSE_SIDECAR:-1}" == "1" ]]; then
  bash "$ROOT_DIR/scripts/launch_saved_view_camera_pose_sidecar.sh" "$VIEW_CONTAINER"
fi

bash "$ROOT_DIR/scripts/launch_rviz_sidecar.sh" "$VIEW_CONTAINER"
