#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -d "$ROOT_DIR/.venv-nuc" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv-nuc/bin/activate"
fi

python - <<'PY'
from importlib.metadata import version
import platform
import sys

import cv2
import numpy
import yaml

print("python:", sys.version.split()[0])
print("platform:", platform.platform())
print("opencv:", cv2.__version__)
print("numpy:", numpy.__version__)
print("pyyaml:", yaml.__version__)
print("rosbags:", version("rosbags"))
PY
