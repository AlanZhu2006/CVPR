#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORKSPACE_ROOT="$(cd "$ROOT_DIR/.." && pwd -P)"
CVPR_ROOT="${CVPR_ROOT:-$WORKSPACE_ROOT/CVPR}"
VIDEOS_ROOT="${VIDEOS_ROOT:-$WORKSPACE_ROOT/videos}"

# Optional: A6000_SSH / A6000_ROOT / INCLUDE_* (see .a6000.env.example at workspace root)
_A6000_ENV="${A6000_ENV_FILE:-$WORKSPACE_ROOT/.a6000.env}"
if [[ -f "$_A6000_ENV" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "$_A6000_ENV"
  set -u
fi

BUNDLE_DIR="${BUNDLE_DIR:-$WORKSPACE_ROOT/a6000_migration_bundle}"
A6000_SSH="${A6000_SSH:-}"
A6000_ROOT="${A6000_ROOT:-~/lingbot-map}"
INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-1}"
INCLUDE_ALL_NUC_OUTPUT="${INCLUDE_ALL_NUC_OUTPUT:-0}"

RSYNC_COMMON=(
  -aH
  --mkpath
  --info=progress2
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='.vscode/'
  --exclude='.idea/'
  --exclude='examples/web-ui/node_modules/'
  --exclude='examples/web-ui/dist/'
  --exclude='examples/web-ui/*.tsbuildinfo'
)

copy_path() {
  local src="$1"
  local rel="$2"
  local dst
  if [[ -n "$A6000_SSH" ]]; then
    dst="$A6000_SSH:$A6000_ROOT/$rel"
    rsync "${RSYNC_COMMON[@]}" "$src" "$dst"
  else
    dst="$BUNDLE_DIR/$rel"
    mkdir -p "$(dirname "$dst")"
    rsync "${RSYNC_COMMON[@]}" "$src" "$dst"
  fi
}

echo "Preparing A6000 migration bundle"
echo "  workspace: $WORKSPACE_ROOT"
if [[ -f "$_A6000_ENV" ]]; then
  echo "  env file:   $_A6000_ENV"
fi
if [[ -n "$A6000_SSH" ]]; then
  echo "  target:    $A6000_SSH:$A6000_ROOT"
  # Ensure remote root exists (rsync --mkpath still needs a valid base path).
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$A6000_SSH" "mkdir -p \"$A6000_ROOT\""
else
  echo "  bundle:    $BUNDLE_DIR"
  mkdir -p "$BUNDLE_DIR"
fi

copy_path "$ROOT_DIR/" "GS_Console/"

if [[ -d "$WORKSPACE_ROOT/scripts/real2sim" ]]; then
  copy_path "$WORKSPACE_ROOT/scripts/" "scripts/"
fi

copy_path "$CVPR_ROOT/HMR3D/" "CVPR/HMR3D/"
copy_path "$CVPR_ROOT/README.md" "CVPR/README.md"
copy_path "$CVPR_ROOT/command.txt" "CVPR/command.txt"
copy_path "$CVPR_ROOT/.gitignore" "CVPR/.gitignore"

if [[ -d "$CVPR_ROOT/third_party_research/lingbot-map" ]]; then
  copy_path "$CVPR_ROOT/third_party_research/lingbot-map/" "CVPR/third_party_research/lingbot-map/"
fi
if [[ -d "$CVPR_ROOT/third_party_research/SplaTAM" ]]; then
  copy_path "$CVPR_ROOT/third_party_research/SplaTAM/" "CVPR/third_party_research/SplaTAM/"
fi
if [[ -f "$CVPR_ROOT/third_party_research/lingbot_cache/lingbot-map.pt" ]]; then
  copy_path "$CVPR_ROOT/third_party_research/lingbot_cache/lingbot-map.pt" \
    "CVPR/third_party_research/lingbot_cache/lingbot-map.pt"
fi
if [[ -d "$VIDEOS_ROOT" ]]; then
  copy_path "$VIDEOS_ROOT/" "videos/"
fi

if [[ "$INCLUDE_OUTPUTS" == "1" ]]; then
  if [[ "$INCLUDE_ALL_NUC_OUTPUT" == "1" ]]; then
    copy_path "$CVPR_ROOT/nuc_output/" "CVPR/nuc_output/"
    if [[ -d "$WORKSPACE_ROOT/nuc_output" ]]; then
      copy_path "$WORKSPACE_ROOT/nuc_output/" "nuc_output/"
    fi
  else
    for rel in \
      "video_real2sim_playback" \
      "real2sim_hikrobot_lingbot_live_baseline" \
      "hikrobot_lingbot_ros2_current_cloud_live"; do
      if [[ -d "$WORKSPACE_ROOT/nuc_output/$rel" ]]; then
        copy_path "$WORKSPACE_ROOT/nuc_output/$rel/" "nuc_output/$rel/"
      fi
      if [[ -d "$CVPR_ROOT/nuc_output/$rel" ]]; then
        copy_path "$CVPR_ROOT/nuc_output/$rel/" "CVPR/nuc_output/$rel/"
      fi
    done
  fi
fi

cat <<EOF

Done.

On A6000, first smoke command:
  cd $A6000_ROOT/CVPR
  python HMR3D/nuc/scripts/run_lingbot_export.py \\
    --model-path third_party_research/lingbot_cache/lingbot-map.pt \\
    --lingbot-map-root third_party_research/lingbot-map \\
    --image-folder ../videos/vid_frames \\
    --output-dir $A6000_ROOT/nuc_output/video_real2sim_playback/lingbot_vid20 \\
    --first-k 20 \\
    --image-size 518 \\
    --model-image-size 518 \\
    --mode streaming \\
    --keyframe-interval 2 \\
    --camera-num-iterations 1

Then show in WebUI:
  cd $A6000_ROOT
  LINGBOT_PREDICTIONS_NPZ=$A6000_ROOT/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_predictions.npz \\
  LINGBOT_SUMMARY_JSON=$A6000_ROOT/nuc_output/video_real2sim_playback/lingbot_vid20/lingbot_summary.json \\
  bash GS_Console/scripts/launch_video_real2sim_playback_stack.sh
EOF
