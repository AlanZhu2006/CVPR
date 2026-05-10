# Mono2Sim-GS

**Calibration-light RGB real-to-sim reconstruction with LingBot-guided Gaussian Splatting.**

This repository is organized to match the final report in
[`reports/main.pdf`](reports/main.pdf). The project focus is not a finished
robot simulator. It is the reconstruction layer before full simulator export:
RGB video or robot RGB streams are converted into pose-aligned geometry and a
Gaussian visual asset that can be inspected in the WebUI.

## Main Idea

Mono2Sim-GS has two maintained lines:

1. **Backend GPU video-to-Gaussian path**
   - input: recorded monocular RGB video or extracted frames
   - LingBot-Map produces pose, depth, confidence, intrinsics, and colored
     point-cloud geometry
   - GenWildSplat produces a feed-forward Gaussian visual prior
   - LG-GS injects LingBot geometry into the GenWildSplat export path so the
     Gaussian means live in the LingBot world frame

2. **Live edge capture path**
   - input: HikRobot RGB camera on a Jetson-class edge device
   - cuVSLAM keeps the live tracking path responsive
   - LingBot dense geometry runs asynchronously
   - the WebUI shows RGB, pose, point cloud, trajectory, and debug map overlays

The long-term target is a hybrid real-to-sim scene bundle:

```text
RGB video / live robot RGB
  -> fast tracking and learned geometry
  -> pose, depth, confidence, colored point cloud
  -> pose-guided Gaussian visual asset
  -> future TSDF / mesh / occupancy / simulator export
```

## Current Scope

Implemented and evaluated in the report:

- HikRobot / ROS / cuVSLAM / LingBot live edge pipeline
- offline 301-frame RGB video processing with LingBot
- GenWildSplat local chunks, overlap chunks, and global keyframe variants
- **LingBot-Guided Gaussian Splatting (LG-GS / Route B)**
- WebUI inspection with RGB-vs-Gaussian view synchronization
- transport improvements for RGB, point clouds, and Gaussian chunks

Not claimed as complete in the final report:

- watertight TSDF or mesh export as the final result
- full Isaac Sim scene export
- full Nav2 deployment
- globally optimized 3DGS for the complete long video

## Repository Layout

```text
.
├── README.md
├── command.txt
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── FINAL_REPORT_ALIGNMENT.md
│   ├── RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md
│   ├── RUNBOOK_LIVE_EDGE_CAPTURE.md
│   └── EXPERIMENT_SUMMARY.md
├── reports/
│   ├── main.pdf
│   ├── legacy_8page_report.pdf
│   └── final_report_latex/
├── scripts/
│   ├── real2sim/
│   └── runbooks/
├── GS_Console/
│   ├── examples/web-ui/
│   ├── scripts/
│   ├── contracts/
│   └── docs/
├── HMR3D/nuc/
│   ├── scripts/
│   └── src/nuc_runtime/
├── cuVSLAM/
└── third_party_research/
```

Important paths:

- [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md): what each folder is
  for after cleanup
- [`docs/FINAL_REPORT_ALIGNMENT.md`](docs/FINAL_REPORT_ALIGNMENT.md): how the
  repo maps to the report sections
- [`docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md`](docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md):
  main backend GPU path
- [`docs/RUNBOOK_LIVE_EDGE_CAPTURE.md`](docs/RUNBOOK_LIVE_EDGE_CAPTURE.md):
  live HikRobot/cuVSLAM/LingBot path
- [`docs/EXPERIMENT_SUMMARY.md`](docs/EXPERIMENT_SUMMARY.md): the ablation
  results to cite in slides/report
- [`GS_Console/README.md`](GS_Console/README.md): WebUI, contracts, playback,
  and live inspection console

## Quick Start: Backend GPU Video-to-Gaussian

This is the main path for the final report.

```bash
cd /path/to/Mono2Sim-GS

# 1. Run LingBot on extracted video frames.
python HMR3D/nuc/scripts/run_lingbot_export.py \
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

# 2. Prepare 12 sparse context frames for GenWildSplat / LG-GS.
python scripts/real2sim/prepare_genwildsplat_scene.py \
  --summary-json nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_summary.json \
  --fallback-frame-dir /path/to/video_frames \
  --scene-dir nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_routeb_12ctx/scene \
  --keyframes 12 \
  --jpeg-quality 98
```

Continue with the full commands in
[`docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md`](docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md).

## Quick Start: Live Edge Capture

This path is for the live HikRobot + Jetson demo.

```bash
cd /path/to/Mono2Sim-GS

START_MONITOR_SYNC=0 \
START_NAV2_STYLE=0 \
ASYNC_ARTIFACT_WRITER=1 \
BINARY_CLOUD_WS_PORT=19093 \
BINARY_CLOUD_MAX_POINTS=60000 \
GLOBAL_MAP=1 \
GLOBAL_BINARY_CLOUD_WS_PORT=19094 \
GLOBAL_BINARY_CLOUD_MAX_POINTS=120000 \
HIKROBOT_ASYNC_TRACKING=1 \
PAUSE_TRACKING_WHILE_DENSE=0 \
DENSE_BUSY_TRACKING_POLICY=none \
HIKROBOT_TRACKING_IDLE_FPS=5.0 \
HIKROBOT_TRACKING_DENSE_FPS=0.5 \
HIKROBOT_TRACKING_QUEUE_SIZE=4 \
PUBLISH_EVERY_WINDOWS=3 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

See [`docs/RUNBOOK_LIVE_EDGE_CAPTURE.md`](docs/RUNBOOK_LIVE_EDGE_CAPTURE.md) for
the full setup and caveats.

## Method Name

The report uses **LG-GS** for the main method:

```text
Original GenWildSplat:
  images -> predicted pose + predicted depth + Gaussian

LG-GS / Route B:
  images + LingBot pose/depth/intrinsics -> pose-guided Gaussian visual asset
```

In implementation logs, this was originally called `Route B`. In final report
and slides, use **LingBot-Guided Gaussian Splatting (LG-GS)**.

## Notes on Large Files

This repo intentionally does not track generated outputs, checkpoints, or local
datasets. Expected local-only paths include:

- `nuc_output/`
- `datasets/`
- `third_party_research/lingbot_cache/`
- generated Gaussian PLY/SPZ/safetensors files

The final report PDFs are kept in `reports/` for presentation and grading.
