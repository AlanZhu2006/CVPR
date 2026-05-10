#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PORT="${1:-${ISAAC_GAUSSIAN_MAPPER_PORT:-8891}}"
OUTPUT_DIR="${ISAAC_GAUSSIAN_MAPPER_OUTPUT_DIR:-/home/chatsign/gs-sdf/runtime/online-gaussian/isaac-online}"

echo "Isaac online gaussian mapper: http://localhost:$PORT"
echo "Output dir: $OUTPUT_DIR"

exec python3 "$ROOT_DIR/scripts/isaac_online_gaussian_mapper_server.py" \
  --port "$PORT" \
  --output-dir "$OUTPUT_DIR"
