#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/runtime/output/2026-04-02-21-33-37_fast_livo2_compressed.bag_fastlivo_cbd_host.yaml}"
SCENE_CONFIG="${2:-$ROOT_DIR/config/fastlivo_cbd_host.yaml}"
STATUS="${3:-running}"
INTERVAL="${4:-20}"

while true; do
  python3 "$ROOT_DIR/scripts/sync_scene_pack.py" \
    --output-dir "$OUTPUT_DIR" \
    --scene-config "$SCENE_CONFIG" \
    --status "$STATUS" \
    --scene-id "fast-livo2-compressed-live"
  sleep "$INTERVAL"
done
