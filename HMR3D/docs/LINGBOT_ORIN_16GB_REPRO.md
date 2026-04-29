# LingBot / cuVSLAM 16GB Orin Reproduction Guide

Date: 2026-04-29

## What This Captures

This workspace now has two runnable LingBot paths:

1. **Full LingBot teacher smoke**, intended for a 16GB Orin or larger GPU.
   This uses LingBot's own `demo.py` and keeps the camera/depth/world-point
   reasoning path that makes the official `church` demo recognizable.
2. **Low-memory cuVSLAM + LingBot student path**, intended for 8GB Orin-class
   devices. This keeps cuVSLAM as the real-time pose source and runs a distilled
   depth-only LingBot mapper in the background.

The important lesson from the 8GB church test is that the depth-only student can
run locally, but it does not reproduce the official LingBot demo quality because
it has no LingBot camera/world-point heads. If cuVSLAM monocular pose drifts,
the fused global point cloud drifts too.

## Current Local Result

The 8GB local smoke used:

```text
church RGB
  -> cuVSLAM OdometryMode.Mono pose
  -> LingBot depth8 student
  -> RGB voxel fusion
  -> Viser playback
```

Output:

```text
nuc_output/lingbot_church_cuvslam_student/church80_student_depth8_voxel4cm
```

Observed:

- tracked frames: `80 / 80`
- LingBot windows: `79 / 79`
- fused point count: `863,932`
- worker failures: `0`
- mean worker window latency: about `1.4s`

Why it is not visually as clean as the official LingBot church demo:

- cuVSLAM monocular returns poses, but the church image sequence has no metric
  calibration/stereo/IMU. Scale jumps and repeated poses were observed.
- The low-memory student is depth-only; it does not include LingBot's full
  camera head or world-point head.
- More frames do not necessarily help if the pose is drifting; they can make the
  fused map look more like a cloud.

Useful local viewers from that run:

```bash
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python \
  HMR3D/nuc/scripts/launch_lingbot_live_viser.py \
  --map-dir nuc_output/lingbot_church_cuvslam_student/church80_student_depth8_voxel4cm \
  --host 0.0.0.0 \
  --port 19118 \
  --point-size 0.026 \
  --max-points 450000 \
  --quantile-clip 0.0005 \
  --color-mode original \
  --initial-mode reveal \
  --fps 6
```

Use `Mode=reveal` for growing playback and `Mode=current` for per-frame RGB-D
debugging. `Mode=all` is a static final map and will not visibly animate.

## First Test On A 16GB Machine

After pulling this repository on the 16GB machine, first test the full LingBot
teacher on the bundled `church` images:

```bash
cd /home/nyu/Codespace/CVPR

MODEL_PATH=/path/to/lingbot-map.pt \
FIRST_K=20 \
IMAGE_SIZE=336 \
KEYFRAME_INTERVAL=2 \
PORT=19115 \
HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
```

Open:

```text
http://127.0.0.1:19115
```

For a remote browser:

```bash
ssh -L 19115:127.0.0.1:19115 nyu@<orin-host>
```

If `FIRST_K=20` is stable, increase in this order:

```bash
FIRST_K=40  IMAGE_SIZE=336 HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
FIRST_K=80  IMAGE_SIZE=336 HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
FIRST_K=160 IMAGE_SIZE=336 HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
```

Only try the original larger setting after the 336 smoke passes:

```bash
FIRST_K=40 IMAGE_SIZE=518 KEYFRAME_INTERVAL=2 \
HMR3D/nuc/scripts/run_lingbot_church_16gb_smoke.sh
```

### Expected Behavior

The full teacher path should look closer to LingBot's official church demo than
the student path because it keeps the learned camera/world-point reasoning.

If it OOMs or the desktop becomes unstable:

- lower `FIRST_K` to `20`
- keep `IMAGE_SIZE=336`
- raise `KEYFRAME_INTERVAL` to `4`
- keep `--offload_to_cpu`, which is already enabled by the script

The script also sets the low-peak loading environment:

```bash
LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
LINGBOT_CHECKPOINT_MMAP=1
LINGBOT_MODEL_DTYPE=fp16
LINGBOT_CPU_CAST_BEFORE_CUDA=1
LINGBOT_CPU_CAST_SCOPE=model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Optional Low-Memory cuVSLAM + Student Demo

If the 16GB machine also has the distilled student checkpoint, run:

```bash
MODEL_PATH=/path/to/lingbot_depth_student.pt \
MAX_FRAMES=80 \
PORT=19116 \
VISER_PORT=19117 \
HMR3D/nuc/scripts/run_lingbot_church_cuvslam_student_demo.sh
```

This path is useful for testing our robotics-style architecture:

```text
cuVSLAM = realtime tracking
LingBot student = delayed dense mapper
Voxel fusion = local colored map
```

But it is not expected to match the official LingBot visual quality on ordinary
monocular church images. It needs either better pose (stereo/RGB-D/IMU/COLMAP)
or a student distilled with camera/world-point supervision, not depth only.

## Utility: Prepare cuVSLAM Mono Input

For any RGB folder:

```bash
python3 HMR3D/nuc/scripts/prepare_lingbot_church_cuvslam_input.py \
  --image-dir third_party_research/lingbot-map/example/church \
  --output-dir nuc_output/lingbot_church_cuvslam_input \
  --fps 10
```

It writes:

```text
calib.txt
times.txt
README.txt
```

This is only a pinhole-intrinsic shim. For accurate reconstruction, real camera
calibration is still better.

## What To Validate

On the 16GB machine, record:

- whether full LingBot `FIRST_K=20/40/80` starts successfully
- peak memory during loading and during streaming
- whether the official viewer shows a recognizable church facade
- whether increasing `FIRST_K` improves continuity or introduces drift

Quick memory watch:

```bash
watch -n 1 'free -h; echo; nvidia-smi || true'
```

## Interpretation

If full LingBot teacher works at `FIRST_K=40+`, then 16GB is enough for the next
stage: distill a smaller student that keeps more than depth, especially camera
or world-point supervision.

If full LingBot teacher still cannot run, use the current depth-only student as
the Orin-local mapper and treat full LingBot as an offline/remote teacher.
