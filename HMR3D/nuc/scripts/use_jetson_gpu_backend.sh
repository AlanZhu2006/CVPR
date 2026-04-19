#!/usr/bin/env bash

VENV="/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson"
CUDSS_LIB="$VENV/opt/cudss-0.6.0/lib"

if [ ! -d "$VENV" ]; then
  echo "Virtualenv not found: $VENV" >&2
  return 1 2>/dev/null || exit 1
fi

if [ ! -d "$CUDSS_LIB" ]; then
  echo "cuDSS library directory not found: $CUDSS_LIB" >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$CUDSS_LIB:${LD_LIBRARY_PATH:-}"

echo "Activated Jetson GPU backend environment"
echo "VENV=$VIRTUAL_ENV"
echo "LD_LIBRARY_PATH prefixed with $CUDSS_LIB"
