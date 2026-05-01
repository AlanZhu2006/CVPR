#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${VENV:-$ROOT_DIR/cuVSLAM/.venv-jetson}"
CUDSS_ROOT="${CUDSS_ROOT:-$VENV/opt/cudss-0.6.0}"
CUDSS_LIB="$CUDSS_ROOT/lib"

if [[ ! -d "$VENV" ]]; then
  echo "Virtualenv not found: $VENV" >&2
  echo "Create it with: HMR3D/nuc/scripts/install_jetson_gpu_backend.sh" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Python executable not found in venv: $VENV/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "$CUDSS_LIB" ]]; then
  echo "cuDSS library directory not found: $CUDSS_LIB" >&2
  echo "Reinstall with: HMR3D/nuc/scripts/install_jetson_gpu_backend.sh" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$CUDSS_LIB:${LD_LIBRARY_PATH:-}"

echo "Activated Jetson GPU backend environment"
echo "ROOT_DIR=$ROOT_DIR"
echo "VENV=$VIRTUAL_ENV"
echo "LD_LIBRARY_PATH prefixed with $CUDSS_LIB"
