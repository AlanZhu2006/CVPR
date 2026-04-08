#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-nuc"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="${1:-$ROOT_DIR/nuc/configs/default_replay.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ $# -lt 2 ]; then
  echo "Usage: bash nuc/scripts/run_nuc_replay.sh [config.yaml] <input_path> [output_dir] [extra args...]"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtual environment at $VENV_DIR"
  echo "Run: bash nuc/scripts/setup_nuc.sh"
  exit 1
fi

INPUT_PATH="$2"
OUTPUT_DIR="${3:-$ROOT_DIR/nuc_output/run_$TIMESTAMP}"

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$ROOT_DIR/nuc/src:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [ $# -gt 3 ]; then
  EXTRA_ARGS=("${@:4}")
fi

"$PYTHON_BIN" "$ROOT_DIR/nuc/tools/run_nuc_replay.py" \
  --config "$CONFIG_PATH" \
  --input "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
