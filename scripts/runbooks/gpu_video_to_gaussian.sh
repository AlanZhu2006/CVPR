#!/usr/bin/env bash
set -euo pipefail

# Backend GPU video-to-Gaussian template for the final-report Mono2Sim-GS path.
# Edit the paths below before running.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FRAME_DIR="${FRAME_DIR:-/path/to/video_frames}"
LINGBOT_OUT="${LINGBOT_OUT:-$REPO_ROOT/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full}"
SCENE_DIR="${SCENE_DIR:-$REPO_ROOT/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_12ctx/scene}"
GWS_OUT="${GWS_OUT:-$REPO_ROOT/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_depth_12ctx_wide_stride1/inference}"

cd "$REPO_ROOT"

echo "[1/4] Running LingBot export"
python -u HMR3D/nuc/scripts/run_lingbot_export.py \
  --model-path third_party_research/lingbot_cache/lingbot-map.pt \
  --lingbot-map-root third_party_research/lingbot-map \
  --image-folder "$FRAME_DIR" \
  --output-dir "$LINGBOT_OUT" \
  --first-k 0 \
  --image-size 518 \
  --model-image-size 518 \
  --mode streaming \
  --keyframe-interval 1 \
  --camera-num-iterations 4 \
  --use-sdpa

echo "[2/4] Preparing GenWildSplat sparse context scene"
python scripts/real2sim/prepare_genwildsplat_scene.py \
  --summary-json "$LINGBOT_OUT/lingbot_summary.json" \
  --fallback-frame-dir "$FRAME_DIR" \
  --scene-dir "$SCENE_DIR" \
  --keyframes 12 \
  --jpeg-quality 98

echo "[3/4] Run GenWildSplat LG-GS command manually"
echo "Open docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md and run Step 3 from third_party_research/GenWildSplat."

echo "[4/4] After Step 3, display-scale the Gaussian PLY with:"
cat <<EOF
python scripts/real2sim/inflate_gaussian_ply.py \\
  --input-ply "$GWS_OUT/gaussians.ply" \\
  --output-ply "$GWS_OUT/gaussians_display_s2.ply" \\
  --scale-multiplier 2.0 \\
  --min-linear-scale 0.003 \\
  --opacity-gamma 0.85
EOF
