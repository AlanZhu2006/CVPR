#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv-nuc"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/nuc/requirements.txt"
bash "$ROOT_DIR/nuc/scripts/check_nuc_env.sh"

echo
echo "NUC prototype environment is ready."
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
