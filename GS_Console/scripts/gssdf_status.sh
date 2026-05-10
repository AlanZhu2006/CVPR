#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-fast-livo2-compressed-cbd}"
OUTPUT_DIR="${2:-$ROOT_DIR/runtime/output/2026-04-02-21-33-37_fast_livo2_compressed.bag_fastlivo_cbd_host.yaml}"
LOG_FILE="${3:-$ROOT_DIR/runtime/logs/fast-livo2-compressed-cbd.log}"

echo "Container:"
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' | awk 'NR==1 || $1=="'"$RUN_NAME"'"'
echo

echo "Output Dir:"
echo "$OUTPUT_DIR"
if [[ -d "$OUTPUT_DIR" ]]; then
  find "$OUTPUT_DIR" -maxdepth 3 -type f | sort | sed -n '1,80p'
else
  echo "missing"
fi
echo

echo "Key Assets:"
for candidate in \
  "$OUTPUT_DIR/model/gs.ply" \
  "$OUTPUT_DIR/model/as_occ_prior.ply" \
  "$OUTPUT_DIR/model/train_points.ply"; do
  if [[ -f "$candidate" ]]; then
    ls -lh "$candidate"
  else
    echo "missing $candidate"
  fi
done
find "$OUTPUT_DIR" -maxdepth 2 -type f -name 'mesh*.ply' -o -name 'gs_*.ply' | sort | sed -n '1,20p'
echo

echo "Log Tail:"
if [[ -f "$LOG_FILE" ]]; then
  tail -n 40 "$LOG_FILE"
else
  echo "missing $LOG_FILE"
fi
