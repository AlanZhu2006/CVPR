# World Navigation Module MVP

This MVP wires a pluggable navigation control layer around the existing Isaac
real-scene validation assets.

## What Runs

```text
WebUI
  -> world-nav module HTTP API :8892
  -> Isaac minimal nav episode runner
  -> Isaac Gaussian online bridge :8890
  -> WebUI robot / trajectory / nav status overlay
```

The visual layer is no longer the noisy online point-cloud splat by default.
`/scenes/isaac-gaussian-online/manifest.json` imports the existing trained
GS-SDF visual manifest:

```text
/scenes/fast-livo2-compressed-live/manifest.json
```

That gives the browser GS viewer the trained chunked Gaussian scene. The live
mapper remains available as a fallback/debug channel, but it does not replace
the trained Gaussian layer for this scene.

## Current Interfaces

World-nav module:

```text
GET  http://127.0.0.1:8892/status
GET  http://127.0.0.1:8892/semantic_goals.json
POST http://127.0.0.1:8892/control
```

Bridge:

```text
GET http://127.0.0.1:8890/status
GET http://127.0.0.1:8890/nav/status
GET http://127.0.0.1:8890/trajectory.json
GET http://127.0.0.1:8890/robot_pose
```

## Launch

Current NuRec physical PointGoal baseline:

```bash
./launch_remote_isaac_gaussian_webui.sh --restart --nav-mode
```

This mode intentionally disables the continuous Isaac training-video publisher.
That keeps `world-nav` as the only process publishing robot pose / trajectory /
nav status to the bridge while a PointGoal episode is running. The default robot
proxy in this mode is `capsule`, which is the stable collision-first baseline.

To test the open-loop Go2 articulation instead:

```bash
./launch_remote_isaac_gaussian_webui.sh --restart --nav-mode --world-nav-robot-model unitree-go2
```

Use `unitree-go2` only after the capsule baseline can plan and collide
correctly; the current Go2 path follower is not a full locomotion controller.

```bash
RENDERER_URL=http://127.0.0.1:8876 \
MAPPER_URL=http://127.0.0.1:8891 \
bash /home/chatsign/gs-sdf/scripts/launch_isaac_gaussian_online_bridge.sh 8890
```

```bash
WORLD_NAV_BRIDGE_URL=http://127.0.0.1:8890 \
WORLD_NAV_DEVICE=cuda:0 \
bash /home/chatsign/gs-sdf/scripts/launch_world_nav_module.sh 8892
```

```bash
WEB_PORT=55173 \
WEB_SCENE=/scenes/isaac-gaussian-online/manifest.json \
WEB_MODE=gs \
WEB_ISAAC_GAUSSIAN_ONLINE_PORT=8890 \
WEB_ISAAC_GAUSSIAN_MAPPER_PORT=8891 \
WEB_WORLD_NAV_PORT=8892 \
bash /home/chatsign/gs-sdf/scripts/launch_web_ui_dev.sh
```

Open:

```text
http://localhost:55173/?scene=/scenes/isaac-gaussian-online/manifest.json&mode=gs
```

## Control Examples

Run the default point-goal:

```bash
curl -fsS -X POST http://127.0.0.1:8892/control \
  -H 'Content-Type: application/json' \
  -d '{"command":"start_pointgoal"}'
```

Run a named semantic goal:

```bash
curl -fsS -X POST http://127.0.0.1:8892/control \
  -H 'Content-Type: application/json' \
  -d '{"command":"start_semantic","semanticGoal":"near_start"}'
```

Stop the current task:

```bash
curl -fsS -X POST http://127.0.0.1:8892/control \
  -H 'Content-Type: application/json' \
  -d '{"command":"stop"}'
```

## Current Result

Verified on the real reconstructed Fast-LIVO2 scene:

```text
near_start          -> success / goal_tolerance
inspection_point_a  -> collision / tilt_limit
```

The second result is useful: it proves the module can classify physical
failure, but it also shows the next engineering task is to tune the collision
proxy / Go2 height / local controller before claiming robust navigation.

## Current Limitations

- "Semantic goals" are named POIs derived from existing successful nav goals.
  They are not yet produced by a semantic detector or semantic map.
- The high-quality Gaussian is a trained/static visual world. The online mapper
  is still a point-cloud-to-splat proxy and should not be treated as a trained
  3DGS optimizer.
- The runner is an Isaac regression harness, not a full ROS2/Nav2 runtime yet.
  The stable API shape is in place so ROS2/Nav2 adapters can be added next.
- ROS2 Humble is available through `/opt/ros/humble/setup.bash`, but Nav2 is not
  installed on this machine yet. The expected packages are
  `ros-humble-navigation2` and `ros-humble-nav2-bringup`; installing them
  requires sudo privileges.
