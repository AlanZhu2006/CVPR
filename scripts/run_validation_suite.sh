#!/usr/bin/env bash
# Full local validation: pytest, TUM smoke fixture + relpose sweep, smoke including hmr_v2_full.
#
# GPU: use --device cuda only after the NVIDIA driver matches your PyTorch wheel (see https://pytorch.org).
# Optional: compile third_party/TTT3R/src/croco/models/curope for fast RoPE (see scripts/bootstrap_env.sh).
#
# Re-eval deps: pip install -r requirements-test.txt  (includes evo for ATE/RPE)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! python3 -c "import evo" 2>/dev/null; then
  echo "Installing evo (ATE/RPE) ..."
  python3 -m pip install --user "evo>=1.11.0"
fi

export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
echo "== pytest =="
python3 -m pytest tests/ -q

echo "== TUM relpose smoke fixture (data/long_tum_s1/smoke_seq) =="
python3 scripts/prepare_tum_relpose_smoke_fixture.py --num-frames 8 --stride 15

echo "== TTT3R runtime compat patches (for eval/smoke) =="
bash scripts/apply_ttt3r_runtime_compat_patches.sh

echo "== relpose sweep (CPU, tiny fixture) =="
python3 scripts/run_relpose_memory_sweep.py --config configs/tum_relpose_smoke_fixture.json

echo "== smoke test: cut3r, ttt3r, hmr_full, hmr_v2_full (CPU) =="
python3 scripts/run_smoke_test.py \
  --weights checkpoints/cut3r_512_dpt_4_64.pth \
  --modes cut3r ttt3r hmr_full hmr_v2_full \
  --num-frames 8 \
  --stride 5 \
  --size 224 \
  --device cpu \
  --output reports/generated/smoke_test_with_v2.json

echo "Done. See reports/generated/tum_relpose_smoke_fixture/leaderboard.json and smoke_test_with_v2.json"
