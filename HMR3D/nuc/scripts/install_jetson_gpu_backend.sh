#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/cuVSLAM/.venv-jetson}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
UV_BIN="${UV_BIN:-$(command -v uv)}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126}"
TORCH_URL="${TORCH_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl}"
TORCHVISION_URL="${TORCHVISION_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl}"
TORCHAUDIO_URL="${TORCHAUDIO_URL:-https://pypi.jetson-ai-lab.io/jp6/cu126/+f/81a/775c8af36ac85/torchaudio-2.8.0-cp310-cp310-linux_aarch64.whl}"
CUDSS_URL="${CUDSS_URL:-https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-aarch64/libcudss-linux-aarch64-0.6.0.5_cuda12-archive.tar.xz}"
CUDSS_ARCHIVE="${CUDSS_ARCHIVE:-$ROOT_DIR/cuVSLAM/.cache/libcudss-linux-aarch64-0.6.0.5_cuda12-archive.tar.xz}"
CUDSS_ROOT="${CUDSS_ROOT:-$VENV_DIR/opt/cudss-0.6.0}"

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "uv not found; please install uv first." >&2
  exit 1
fi

mkdir -p "$(dirname "$CUDSS_ARCHIVE")"
mkdir -p "$ROOT_DIR/cuVSLAM"

echo "Installing Jetson GPU backend"
echo "  root:       $ROOT_DIR"
echo "  venv:       $VENV_DIR"
echo "  python:     $PYTHON_BIN"
echo "  torch:      $TORCH_URL"
echo "  torchvision:$TORCHVISION_URL"
echo "  torchaudio: $TORCHAUDIO_URL"
echo "  cudss:      $CUDSS_URL"

"$UV_BIN" venv --seed --python "$PYTHON_BIN" "$VENV_DIR"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install "numpy==1.26.4"
"$VENV_DIR/bin/python" -m pip install "$TORCH_URL" "$TORCHVISION_URL" "$TORCHAUDIO_URL"
"$VENV_DIR/bin/python" -m pip install \
  Pillow \
  huggingface_hub \
  einops \
  safetensors \
  matplotlib \
  "opencv-python<4.12" \
  tqdm \
  scipy \
  viser \
  requests
"$VENV_DIR/bin/python" -m pip install --upgrade --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.15.3"

if [[ ! -f "$CUDSS_ARCHIVE" ]]; then
  wget -O "$CUDSS_ARCHIVE" "$CUDSS_URL"
fi

rm -rf "$CUDSS_ROOT"
mkdir -p "$(dirname "$CUDSS_ROOT")"
tar -xJf "$CUDSS_ARCHIVE" -C "$(dirname "$CUDSS_ROOT")"
if [[ ! -d "$CUDSS_ROOT" ]]; then
  extracted_dir="$(find "$(dirname "$CUDSS_ROOT")" -maxdepth 1 -type d -name 'libcudss-linux-aarch64-*' | head -n1 || true)"
  if [[ -n "$extracted_dir" ]]; then
    mv "$extracted_dir" "$CUDSS_ROOT"
  fi
fi

cat <<EOF

Jetson GPU backend installed.

Activate with:
  source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh

Quick check:
  $VENV_DIR/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
EOF
