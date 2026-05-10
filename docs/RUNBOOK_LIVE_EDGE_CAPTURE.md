# Runbook: Live Edge Capture

This is the secondary project path from the final report. It validates the
edge-cloud systems design:

```text
HikRobot RGB camera
  -> cuVSLAM low-latency tracking
  -> asynchronous LingBot dense geometry
  -> RGB / trajectory / point cloud WebUI
```

## Goal

Keep the robot-side loop responsive while dense learned geometry arrives at a
slower rate.

The live path should not require:

```text
RGB fps == dense geometry fps == Gaussian update fps
```

Instead, it uses:

```text
RGB preview and pose: fast
LingBot dense geometry: asynchronous
Gaussian monitor/render: optional and slower
```

## Hardware Used In Report

- HikRobot RGB industrial camera
- Jetson-class edge device
- ROS2 Humble
- cuVSLAM tracking
- LingBot-Map checkpoint
- optional backend / WebUI host

## Environment

```bash
cd /path/to/Mono2Sim-GS
source /opt/ros/humble/setup.bash
source HMR3D/nuc/configs/hikrobot_mvs_env.sh 2>/dev/null || true
source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CUDA_ARCH_LIST="8.7"
export LINGBOT_LOAD_CHECKPOINT_ON_CPU=1
export LINGBOT_CHECKPOINT_MMAP=1
export LINGBOT_MODEL_DTYPE=fp16
export LINGBOT_CPU_CAST_BEFORE_CUDA=1
export LINGBOT_CPU_CAST_SCOPE=model
```

## Camera Smoke Test

```bash
python3 HMR3D/nuc/scripts/hikrobot_mvs_smoke.py
```

If this fails, fix camera SDK/device access before starting the full stack.

## Live Reconstruction Command

```bash
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
  --hikrobot-exposure-us 3000 \
  --hikrobot-gain 0 \
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
```

## Full Stack Launcher

If the GS Console launch scripts are available in the workspace, use the
launcher form:

```bash
START_MONITOR_SYNC=0 \
START_NAV2_STYLE=0 \
ASYNC_ARTIFACT_WRITER=1 \
BINARY_CLOUD_WS_PORT=19093 \
BINARY_CLOUD_MAX_POINTS=60000 \
GLOBAL_MAP=1 \
GLOBAL_BINARY_CLOUD_WS_PORT=19094 \
GLOBAL_BINARY_CLOUD_MAX_POINTS=120000 \
HIKROBOT_EXPOSURE_US=3000 \
HIKROBOT_GAIN=0 \
HIKROBOT_ASYNC_TRACKING=1 \
PAUSE_TRACKING_WHILE_DENSE=0 \
DENSE_BUSY_TRACKING_POLICY=none \
HIKROBOT_TRACKING_IDLE_FPS=5.0 \
HIKROBOT_TRACKING_DENSE_FPS=0.5 \
HIKROBOT_TRACKING_QUEUE_SIZE=4 \
PUBLISH_EVERY_WINDOWS=3 \
bash GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

Open the WebUI:

```text
http://<host>:5173/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=/contracts/lingbot-map-ros2-live.live-contract.json
```

## Report Measurements

The final report cites this live-edge behavior:

| Metric | Observed value |
| --- | --- |
| HikRobot RGB rate | 4-5 fps |
| ROS RGB topic rate | 4-5 fps |
| Rolling map point count | 49,753 |
| Dense updates | 7 |
| Processed dense windows | 7 |
| Trajectory poses | 103 |
| Saved RGB keyframes | 73 |
| Worker failures | 0 |
| Queue drops | 0 |
| Worker end-to-end mean / median | 5.14 s / 4.09 s |
| Geometry age mean / median | 5.59 s / 4.47 s |
| Model forward mean / median | 4.11 s / 3.11 s |

## Interpretation

The live path demonstrates multi-rate operation. It is not a full-rate dense
Gaussian mapper. It is a practical edge-cloud backbone where current RGB and
pose remain available while learned dense geometry updates asynchronously.
