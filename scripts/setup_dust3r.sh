#!/usr/bin/env bash
# Pull TTT3R (ships dust3r under third_party/TTT3R/src/dust3r) and install its Python deps.
# Eval/smoke entrypoints call bootstrap_ttt3r_imports() so dust3r resolves without pip-installing dust3r as a package.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ ! -f third_party/TTT3R/requirements.txt ]]; then
  echo "third_party/TTT3R missing or empty — initializing git submodule..."
  git submodule update --init --recursive third_party/TTT3R
fi

echo "Installing Python deps from third_party/TTT3R/requirements.txt ..."
python3 -m pip install -r third_party/TTT3R/requirements.txt

echo ""
echo "OK: dust3r import path is third_party/TTT3R/src/dust3r (added to sys.path by hmr3d_memory.ttt3r_env.bootstrap_ttt3r_imports)."
echo "Optional GPU speedup: compile croco curope (needs working CUDA toolchain; see scripts/bootstrap_env.sh)."
echo "  cd third_party/TTT3R/src/croco/models/curope && python3 setup.py build_ext --inplace"
echo "Apply HMR3D runtime compat patches to TTT3R (PyTorch 2.6 load, RoPE CPU, evo): bash scripts/apply_ttt3r_runtime_compat_patches.sh"
