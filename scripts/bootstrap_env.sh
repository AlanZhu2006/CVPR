#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${HMR3D_ENV_NAME:-hmr3d-ttt3r}"
PYTHON_VERSION="${HMR3D_PYTHON_VERSION:-3.11}"
CUDA_PKG="${HMR3D_PYTORCH_CUDA:-12.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}" cmake=3.14.0
fi

conda activate "${ENV_NAME}"
conda install -y pytorch torchvision pytorch-cuda="${CUDA_PKG}" -c pytorch -c nvidia
conda install -y cuda-nvcc=12.1.105 -c nvidia
conda install -y cuda-cudart-dev=12.1.105 cuda-cccl=12.1.109 -c nvidia
conda install -y mkl=2023.1.0 intel-openmp=2023.1.0
conda install -y ninja
pip install -r "${ROOT_DIR}/third_party/TTT3R/requirements.txt"
pip install evo open3d gdown
conda install -y 'llvm-openmp<16'

pushd "${ROOT_DIR}/third_party/TTT3R/src/croco/models/curope" >/dev/null
export CUDA_HOME="${CONDA_PREFIX}"
export CPATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/targets/x86_64-linux/include/cccl${CPATH:+:${CPATH}}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/targets/x86_64-linux/include:${CONDA_PREFIX}/targets/x86_64-linux/include/cccl${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
export LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python setup.py build_ext --inplace
popd >/dev/null

echo "Environment '${ENV_NAME}' is ready."
