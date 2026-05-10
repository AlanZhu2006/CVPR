#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
OUTPUT_DIR="${1:-$ROOT_DIR/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml}"
VIEW_CONTAINER="${2:-gssdf-view-quality}"
BRIDGE_PORT="${3:-8876}"
BRIDGE_CONTAINER="${4:-gssdf-hifi-bridge}"
WAIT_SECONDS="${WAIT_SECONDS:-90}"

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

bash "$ROOT_DIR/scripts/launch_gssdf_hifi_bridge.sh" "$VIEW_CONTAINER" "$BRIDGE_PORT" "$BRIDGE_CONTAINER"

echo "HiFi stack ready."
echo "Native render bridge: http://localhost:${BRIDGE_PORT}"
echo "Open the web UI and switch to the High Fidelity mode."
