# Runbook: Backend GPU Video to Gaussian

This is the main path for the final report. It should be run on a backend GPU
machine when possible.

## Goal

```text
RGB video frames
  -> LingBot pose/depth/confidence/point cloud
  -> sparse context frames for GenWildSplat
  -> LG-GS pose-guided Gaussian
  -> WebUI-ready chunks
```

## Expected Inputs

```text
/path/to/video_frames/
  000001.jpg
  000002.jpg
  ...
third_party_research/lingbot_cache/lingbot-map.pt
third_party_research/GenWildSplat/checkpoint/model.safetensors
```

Large inputs and generated outputs should stay local under `nuc_output/`.

## Step 1: LingBot Full-Video Export

```bash
cd /path/to/Mono2Sim-GS

source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CUDA_ARCH_LIST="8.7"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=1
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=model

python -u HMR3D/nuc/scripts/run_lingbot_export.py \
  --model-path third_party_research/lingbot_cache/lingbot-map.pt \
  --lingbot-map-root third_party_research/lingbot-map \
  --image-folder /path/to/video_frames \
  --output-dir nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full \
  --first-k 0 \
  --image-size 518 \
  --model-image-size 518 \
  --mode streaming \
  --keyframe-interval 1 \
  --camera-num-iterations 4 \
  --use-sdpa
```

Expected outputs:

```text
nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/
  lingbot_predictions.npz
  lingbot_summary.json
```

## Step 2: Prepare Sparse Context Scene

The report's active route uses 12 keyframes over the full 301-frame video.

```bash
python scripts/real2sim/prepare_genwildsplat_scene.py \
  --summary-json nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_summary.json \
  --fallback-frame-dir /path/to/video_frames \
  --scene-dir nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_12ctx/scene \
  --keyframes 12 \
  --jpeg-quality 98
```

## Step 3: Run LG-GS / Route B in GenWildSplat

Run from the GenWildSplat checkout:

```bash
cd /path/to/Mono2Sim-GS/third_party_research/GenWildSplat

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HOME=/path/to/Mono2Sim-GS/cache/genwildsplat/huggingface \
XDG_CACHE_HOME=/path/to/Mono2Sim-GS/cache/genwildsplat/xdg \
python3 src/eval_nvs_video.py \
  --data_dir /path/to/Mono2Sim-GS/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_12ctx/scene \
  --output_path /path/to/Mono2Sim-GS/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_depth_12ctx_wide_stride1/inference \
  --ckpt_path checkpoint/model.safetensors \
  --no_refine \
  --export_ply \
  --export_safetensors \
  --export_poses \
  --image_preprocess lingbot_crop \
  --lingbot_image_width 518 \
  --lingbot_image_height 294 \
  --max_context_frames 12 \
  --skip_video_export \
  --disable_voxelize \
  --external_lingbot_sample_stride 1 \
  --external_lingbot_predictions_npz /path/to/Mono2Sim-GS/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_predictions.npz \
  --external_lingbot_summary_json /path/to/Mono2Sim-GS/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_summary.json
```

Expected outputs:

```text
gaussians.ply
gaussians.safetensors
camera_poses.json
```

## Step 4: Display-Scale the Gaussian PLY

This is display calibration, not reconstruction.

```bash
cd /path/to/Mono2Sim-GS

python scripts/real2sim/inflate_gaussian_ply.py \
  --input-ply nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_depth_12ctx_wide_stride1/inference/gaussians.ply \
  --output-ply nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_depth_12ctx_wide_stride1/inference/gaussians_display_s2.ply \
  --scale-multiplier 2.0 \
  --min-linear-scale 0.003 \
  --opacity-gamma 0.85
```

## Step 5: Convert to WebUI Chunks

The original working stack used `GS_Console/scripts/preprocess_gaussian_stream.mjs`.
If this repository is checked out without GS Console, run this from the adjacent
GS Console workspace or restore that script before using this step.

```bash
node GS_Console/scripts/preprocess_gaussian_stream.mjs \
  --input nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_depth_12ctx_wide_stride1/inference/gaussians_display_s2.ply \
  --variant genwildsplat_routeb_depth_12ctx_wide_stride1_s2 \
  --grid 8,4 \
  --max-sh 0 \
  --force
```

## Active Variant From Report

```text
genwildsplat_routeb_depth_12ctx_wide_stride1_s2
12 keyframes
wide 518x294 preprocessing
about 1.83M splats
LingBot depth + intrinsics + extrinsics initialize Gaussian means
```

## Common Failure Modes

| Symptom | Likely cause |
| --- | --- |
| RGB and Gaussian view disagree | wrong C2W/W2C convention or using raw GenWildSplat gauge |
| right wall / hallway missing | square crop removed wide FOV |
| grid-like wall gaps | low density, tiny scales, or low-opacity splats |
| black speckles | low-opacity/dark splats in black-background renderer |
| global scene blurry | too many weak-overlap keyframes in one feed-forward pass |

## Report Interpretation

This path supports the report's main claim: GenWildSplat is useful as a
sparse-view visual prior, but long-video consistency improves when LingBot
provides the geometric scaffold.
