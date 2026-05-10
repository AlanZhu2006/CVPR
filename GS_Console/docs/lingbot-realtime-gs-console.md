# LingBot Realtime GS Console

## Target

This stack keeps the realtime signals synchronized through ROS2 timestamps and one live contract:

- RGB preview: `/neural_mapping/rgb`
- Pose and trajectory: `/neural_mapping/pose`, `/neural_mapping/path`
- Colored global point cloud: `/neural_mapping/pointcloud`
- Current local dense cloud: `/lingbot/current_cloud_rgb`
- Nav2-style goal input: `/goal_pose`

The WebUI consumes these through `rosbridge_suite` and
`/contracts/lingbot-map-ros2-live.live-contract.json`.

## Recommended Launch

From the repository root:

```bash
export ROS_SETUP=/opt/ros/humble/setup.bash
export CVPR_ROOT=/home/nvidia/twork/lingbot-map/CVPR
export MODEL_PATH=/home/nvidia/twork/lingbot-map/CVPR/third_party_research/lingbot_cache/lingbot-map.pt
export LINGBOT_ROOT=/home/nvidia/twork/lingbot-map/CVPR/third_party_research/lingbot-map

bash /home/nvidia/twork/lingbot-map/GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

Open:

```text
http://localhost:5173/?scene=/scenes/lingbot-live/manifest.json&mode=live&liveContract=/contracts/lingbot-map-ros2-live.live-contract.json
```

## 3s Gaussian Render Cadence

For the screenshot-style monitor tile, enable the lightweight renderer loop:

```bash
export START_GAUSSIAN_MONITOR_RENDER=1
export GAUSSIAN_RENDER_INTERVAL_SEC=3.0
export GAUSSIAN_RENDER_BUDGET_POINTS=24000
```

This follows the WildSplat-style idea of keeping a small active Gaussian set hot
and rendering only the latest camera pose. It is meant as a live monitor, not as
full retraining every 3 seconds. The slower TSDF / Gaussian seed refresh can run
in the background and the renderer will pick up the latest seed when available.

## Nav2 Notes

The included lightweight bridge publishes a derived `/map`, `/lingbot_nav/trajectory`,
and `/lingbot_nav/plan` from the live monitor JSON, and listens to `/goal_pose`.
Use it for WebUI/RViz validation first.

For full Nav2 planning, keep the same WebUI goal topic and replace the lightweight
bridge with a real Nav2 stack consuming an exported map YAML. The existing helper is:

```bash
NAV2_MAP_YAML=/path/to/map.yaml \
bash /home/nvidia/twork/lingbot-map/GS_Console/scripts/launch_nav2_baseline.sh
```

## Dependency Fixes

If the WebUI says `vite: not found`, install the frontend dependencies once:

```bash
cd /home/nvidia/twork/lingbot-map/GS_Console/examples/web-ui
npm ci
```

The launch script now does this automatically when `node_modules/.bin/vite`
or `node_modules/.bin/tsc` is missing.

If ROS says `Package 'rosbridge_server' not found`, install the ROS2 bridge:

```bash
sudo apt update
sudo apt install ros-humble-rosbridge-server
```

If another rosbridge is already running, keep it and launch with:

```bash
START_ROSBRIDGE=0 ROSBRIDGE_PORT=9090 \
bash /home/nvidia/twork/lingbot-map/GS_Console/scripts/launch_lingbot_realtime_stack.sh
```

## Performance Defaults

The launch script caps web-facing image and cloud throughput:

- RGB preview: 960x540 max
- Global cloud publish interval: 0.25s or slower
- Global cloud cap: 120k points
- Current cloud cap: 60k points
- Path cap: 1200 poses

These limits keep rosbridge responsive while LingBot and the renderer share the
GPU. Increase them only after RGB and pose latency are stable.
