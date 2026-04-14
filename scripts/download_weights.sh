#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/checkpoints"

if ! python3 -m gdown --version >/dev/null 2>&1; then
  python3 -m pip install --user gdown
fi

TARGET="${ROOT_DIR}/checkpoints/cut3r_512_dpt_4_64.pth"
MIN_BYTES=3000000000
if [[ -f "${TARGET}" ]]; then
  CURRENT_BYTES="$(stat -c%s "${TARGET}")"
  if (( CURRENT_BYTES >= MIN_BYTES )); then
    echo "Checkpoint already exists at ${TARGET}"
    exit 0
  fi
  echo "Resuming partial checkpoint download at ${TARGET}"
fi

# Use `python3 -m gdown` so we pick up the pip module (system `gdown` may be an old stub without --fuzzy).
python3 -m gdown --continue "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD" -O "${TARGET}"
echo "Downloaded checkpoint to ${TARGET}"
