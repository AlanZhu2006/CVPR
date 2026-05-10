#!/usr/bin/env bash
set -euo pipefail

# Live edge capture template for HikRobot + cuVSLAM + LingBot.
# This runs the lower-level live reconstruction script. If GS_Console is
# available in your workspace, see docs/RUNBOOK_LIVE_EDGE_CAPTURE.md for the
# launcher-based WebUI command.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

source /opt/ros/humble/setup.bash
source HMR3D/nuc/configs/hikrobot_mvs_env.sh 2>/dev/null || true
source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-8.7}"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU="${LINGBOT_LOAD_CHECKPOINT_ON_CPU:-1}"
export LINGBOT_CHECKPOINT_MMAP="${LINGBOT_CHECKPOINT_MMAP:-1}"
export LINGBOT_MODEL_DTYPE="${LINGBOT_MODEL_DTYPE:-fp16}"
export LINGBOT_CPU_CAST_BEFORE_CUDA="${LINGBOT_CPU_CAST_BEFORE_CUDA:-1}"
export LINGBOT_CPU_CAST_SCOPE="${LINGBOT_CPU_CAST_SCOPE:-model}"

python -X faulthandler HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py \
  --tracking-backend hikrobot_mono_rgb \
  --model-path third_party_research/lingbot_cache/lingbot-map.pt \
  --lingbot-map-root third_party_research/lingbot-map \
  --output-dir nuc_output/hikrobot_lingbot_ros2_current_cloud_live \
  --image-size 224 \
  --model-image-size 518 \
  --window-size 2 \
  --stride 1 \
  --max-frames 0 \
  --frame-step 1 \
  --hikrobot-width 640 \
  --hikrobot-height 512 \
  --hikrobot-fps 5 \
  --hikrobot-exposure-us "${HIKROBOT_EXPOSURE_US:-3000}" \
  --hikrobot-gain "${HIKROBOT_GAIN:-0}" \
  --hikrobot-threaded-capture \
  --hikrobot-capture-queue-size 6 \
  --hikrobot-max-read-errors 100 \
  --ros2-publish \
  --ros2-image-max-width 640 \
  --ros2-max-cloud-points 300000 \
  --ros2-max-current-cloud-points 60000 \
  --ros2-republish-current-cloud-on-image \
  --ros2-cloud-min-interval-sec 1.0 \
  --publish-every-windows 3 \
  --publish-every-frames 0 \
  --sample-stride 1 \
  --sampling-pattern random \
  --max-points-per-frame 15000 \
  --max-active-frames 16 \
  --drop-when-busy \
  --serve \
  --port 19102
