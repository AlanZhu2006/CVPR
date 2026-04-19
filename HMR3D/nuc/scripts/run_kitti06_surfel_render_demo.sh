#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/nyu/Codespace/CVPR"
SEQ="$ROOT/cuVSLAM/examples/kitti/dataset/sequences/06"
TRAJ="$SEQ/trajectory_tum.txt"
CONFIG="$ROOT/HMR3D/nuc/configs/kitti06_v4_render_benchmark.yaml"
OUTPUT="$ROOT/HMR3D/nuc_output/kitti06_render_benchmark_surfel"

python "$ROOT/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py" \
  --config "$CONFIG" \
  --sequence-path "$SEQ" \
  --trajectory-path "$TRAJ" \
  --max-frames 200 \
  --output-dir "$OUTPUT" \
  --save-images
