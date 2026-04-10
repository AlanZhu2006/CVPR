#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/checkpoints"

if ! command -v gdown >/dev/null 2>&1; then
  pip install gdown
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

gdown --continue --fuzzy "https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link" -O "${TARGET}"
echo "Downloaded checkpoint to ${TARGET}"
