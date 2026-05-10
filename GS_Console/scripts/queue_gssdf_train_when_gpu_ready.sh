#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_PATH="${1:-}"
CONFIG_PATH="${2:-$ROOT_DIR/config/fastlivo_cbd_quality_v3.yaml}"
RUN_NAME="${3:-gs-sdf-train-quality-v3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
POLL_SEC="${POLL_SEC:-30}"
QUEUE_LOG="${QUEUE_LOG:-$ROOT_DIR/runtime/logs/${RUN_NAME}.queue.log}"

if [[ -z "$INPUT_PATH" ]]; then
  echo "Usage: $0 <bag-or-color_poses.txt> [config.yaml] [run-name]" >&2
  exit 1
fi

mkdir -p "$(dirname "$QUEUE_LOG")"
touch "$QUEUE_LOG"

echo "Queueing GS-SDF training." | tee -a "$QUEUE_LOG"
echo "Input:        $INPUT_PATH" | tee -a "$QUEUE_LOG"
echo "Config:       $CONFIG_PATH" | tee -a "$QUEUE_LOG"
echo "Run name:     $RUN_NAME" | tee -a "$QUEUE_LOG"
echo "Min free VRAM ${MIN_FREE_MIB} MiB, polling every ${POLL_SEC}s" | tee -a "$QUEUE_LOG"

while true; do
  FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  if [[ -n "$FREE_MIB" && "$FREE_MIB" =~ ^[0-9]+$ && "$FREE_MIB" -ge "$MIN_FREE_MIB" ]]; then
    echo "[$(date --iso-8601=seconds)] GPU free memory ${FREE_MIB} MiB >= ${MIN_FREE_MIB} MiB, starting training." | tee -a "$QUEUE_LOG"
    exec bash "$ROOT_DIR/scripts/train_gssdf.sh" "$INPUT_PATH" "$CONFIG_PATH" "$RUN_NAME" 2>&1 | tee -a "$QUEUE_LOG"
  fi
  echo "[$(date --iso-8601=seconds)] Waiting for GPU, free ${FREE_MIB:-unknown} MiB." >> "$QUEUE_LOG"
  sleep "$POLL_SEC"
done
