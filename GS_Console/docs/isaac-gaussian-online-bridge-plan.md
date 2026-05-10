# Isaac Gaussian Online Bridge Plan

This document scopes a practical first implementation for an online loop:

```text
Isaac Lab / Isaac Sim
-> robot pose + RGB-D + semantics + optional lidar
-> bridge / contract
-> Gaussian reconstruction or renderer service
-> rendered Gaussian view + semantic/map products
-> Isaac / robot control stack
```

The goal is not to force everything into one process. The goal is to get a
stable real-time integration that can later be upgraded to native Omniverse
Gaussian rendering if the local stack supports it.

## Recommendation Summary

For the first polished demo, use a photoreal indoor mesh scene with existing
semantics, not Oxford RobotCar.

Recommended order:

1. `Replica` or `ReplicaCAD` for the first end-to-end loop.
2. `HM3D-Semantics` if you need larger indoor spaces and denser semantics.
3. Isaac native `office.usd` or `full_warehouse.usd` if you want the fastest
   Isaac-side bring-up with minimal import friction.
4. Oxford RobotCar later, as an outdoor reconstruction target rather than the
   first "pretty mesh" simulation environment.

Why:

- `Replica` already provides dense geometry, textures, semantic labels, and
  Habitat-ready assets such as semantic meshes and navmeshes.
- `ReplicaCAD` is designed for interactive embodied simulation and has both an
  interactive version and a baked-lighting version for nicer visualization.
- `HM3D-Semantics` is larger and richer semantically, but usually means more
  dataset and conversion work.
- Isaac native environment USDs are the easiest way to get a clean robot demo
  running quickly inside Isaac.
- Oxford RobotCar is excellent as a large-scale autonomy dataset, but it is a
  route-scale sensor dataset, not an out-of-the-box pretty mesh world for the
  first renderer-control integration.

## Direct Gaussian Rendering in This Workspace

Two official Omniverse rendering paths now exist:

1. `NuRec` rendering via USDZ conversion.
2. `Particle Fields` via OpenUSD Gaussian splat assets.

However, that does not mean the current local Isaac Sim workspace can use them
directly today.

Local check in this workspace on `2026-04-24`:

- Isaac Sim environment: `/media/chatsign/data-002/isaac/env_isaacsim45_lab21`
- Isaac Sim package version documented locally: `4.5.0.0`
- Local filesystem scan found no obvious `nurec`, `particle field`, or
  Gaussian-splat extension payloads inside the installed Isaac/Isaac Lab tree.

This is a negative signal, not absolute proof, but it is strong enough that the
first integration should not assume native direct Gaussian rendering inside the
current Isaac app.

There is a second hard constraint from the official Omniverse docs:

- NuRec primitives are not part of synthetic data passes.
- Proxy meshes must provide synthetic data such as semantics and depth-facing
  structure for synthetic AOVs.

So even with native NuRec support, you still want a proxy mesh / collision
layer for semantics, physics, and navigation.

Practical conclusion:

```text
First implementation = asynchronous Gaussian service + proxy mesh in Isaac.
```

If later you upgrade to an Omniverse / Isaac stack with confirmed Particle Field
or NuRec support, the Gaussian service can be collapsed back into the renderer
process. The contract should stay the same.

## Environment Choice

### Option A: Replica / ReplicaCAD

Use this if the goal is:

- pretty mesh
- photoreal visuals
- indoor mobile robot demo
- semantic loop

Why it is a strong first target:

- `Replica` provides high-quality reconstructions, dense geometry, HDR textures,
  and semantic class and instance segmentation.
- `ReplicaCAD` is specifically packaged for embodied simulation and provides an
  interactive simulation-oriented dataset plus a baked-lighting dataset for
  nicer visuals.

Tradeoff:

- Not native Isaac USD content, so you will need a conversion/import pass or a
  scene ingestion pipeline.

### Option B: HM3D-Semantics

Use this if the goal is:

- larger indoor scenes
- stronger semantic scale
- semantic navigation experiments

Tradeoff:

- More data handling and usually more conversion work than Replica.

### Option C: Isaac Native Office / Warehouse

Use this if the goal is:

- fastest possible bring-up inside Isaac
- fewer asset conversion steps
- stable ROS 2 sensor publication first

Tradeoff:

- Cleaner engineering path, but less interesting if you specifically want a
  "reconstructed world" look.

### Option D: Oxford RobotCar

Use this if the goal is:

- outdoor route-scale autonomy
- long trajectories
- real robot sensor realism

Do not use it as the first pretty mesh demo unless you are also ready to build:

- the mesh reconstruction layer
- the collision proxy layer
- the semantic backfill layer

Oxford is better as:

```text
dataset source -> reconstruction / Gaussian training target
```

than as:

```text
ready-made Isaac mesh scene
```

## System Architecture

Keep the system modular and process-separated.

### 1. Simulation Unit

Runs in Isaac Lab / Isaac Sim.

Responsibilities:

- load robot
- load collision mesh / nav mesh / semantic mesh
- simulate physics
- publish robot state and sensor outputs
- receive Gaussian-side semantic/map feedback

Outputs:

- `pose6d`
- `rgb`
- `depth`
- `camera_info`
- `semantic_labels`
- optional `lidar` or `pointcloud`
- optional `tf`

### 2. Gaussian Service

Runs as an independent process or machine.

Responsibilities:

- consume pose and optional sensor streams
- update Gaussian model or load a fixed Gaussian scene
- render requested viewpoints from pose
- return rendered RGB, depth-like proxy if available, semantic overlays, or map
  products

Modes:

- `offline fixed scene renderer`
- `online incremental Gaussian mapper`
- `hybrid`, where mapping is slower and rendering is faster from the latest
  checkpointed state

### 3. Bridge / Contract Layer

This should be the only layer both sides depend on.

The current repo already has the right shape for this:

- `docs/slam-adapter-contract.md`
- `contracts/live-contract.default.json`
- `scripts/gssdf_hifi_bridge_server.py`

That means you do not need a brand-new protocol design. You can extend the
existing channel contract and swap transports underneath it.

### 4. Control / Consumption Layer

Consumes:

- semantics returned by the Gaussian side
- map updates
- rendered views for operator UI
- optional relocalization confidence or loop-closure hints

This layer can live:

- inside Isaac for closed-loop experiments
- in ROS 2 navigation / autonomy nodes
- in the Web UI

## Transport Recommendation

Use different transports for different jobs.

### Real-Time Robotics Data

Use `ROS 2 topics`.

Why:

- native Isaac support already exists for RGB, depth, point cloud, odometry,
  semantic labels, and lidar helpers
- easy to inspect with existing robotics tools
- easy to record with `rosbag2`
- matches the control-system side of the stack

Recommended ROS 2 direction:

- Isaac publishes robot pose, RGB-D, semantic labels, point cloud
- controller publishes navigation goals and optional control commands
- Gaussian bridge subscribes to pose and selected sensor topics

### Renderer RPC / High-Rate View Streaming

Do not force all renderer traffic through `rosbag`.

Use one of:

- `ROS 2` for simple all-robotics deployments
- `gRPC` or `ZeroMQ` for a cleaner independent renderer service
- `HTTP + MJPEG` or `WebSocket` for operator preview and Web UI

Recommendation:

- keep robot-facing telemetry in `ROS 2`
- keep renderer request/response in a dedicated service interface

That split is cleaner if the Gaussian module later moves to another workstation
connected over Ethernet.

### Recording and Replay

Use `rosbag2`.

`rosbag2` is for:

- debugging
- benchmarking
- reproducibility
- offline training

It should not be the primary real-time transport between Isaac and the Gaussian
service.

## Proposed Live Channels

The current contract already contains useful canonical keys such as:

- `robot_pose`
- `rgb_frame`
- `depth_frame`
- `trajectory`
- `camera_pose`

Extend it with a Gaussian profile:

| key | direction | purpose |
| --- | --- | --- |
| `robot_pose` | Isaac -> bridge | 6-DoF body pose |
| `camera_pose` | Isaac -> bridge | 6-DoF sensor pose |
| `rgb_frame` | Isaac -> bridge | RGB frame |
| `depth_frame` | Isaac -> bridge | depth frame |
| `semantic_frame` | Isaac -> bridge | semantic image or labels |
| `lidar_frame` | Isaac -> bridge | optional lidar cloud |
| `gaussian_render_rgb` | Gaussian -> consumers | rendered Gaussian view |
| `gaussian_render_depth` | Gaussian -> consumers | optional rendered depth proxy |
| `gaussian_semantics` | Gaussian -> consumers | semantic map or overlay |
| `gaussian_map_state` | Gaussian -> consumers | map version, confidence, bounds |
| `nav_goal` | controller -> Isaac | navigation goals |
| `initial_pose` | controller -> Isaac | initialization or reset |

The important rule is:

```text
keep semantic keys stable, change transports underneath them
```

## First Executable Milestone

The first version should avoid online training complexity.

### Milestone 1

1. Pick one static scene:
   - `ReplicaCAD baked lighting` if you want a pretty demo
   - `office.usd` if you want the fastest Isaac-first path
2. Load a robot in Isaac Lab and publish:
   - pose
   - RGB
   - depth
   - semantic labels
3. Run a standalone Gaussian renderer service that accepts:
   - camera pose
   - optional scene id
4. Return rendered frames to:
   - Web UI preview
   - optional ROS 2 image topic
5. Return semantic or map products to Isaac / controller.
6. Record the whole session with `rosbag2`.

This gives you the full communication loop without blocking on online Gaussian
optimization.

### Milestone 2

Swap the static Gaussian renderer for an online mapper that updates the scene
every `N` frames or `T` seconds.

### Milestone 3

Move the Gaussian service to a second machine and keep the exact same bridge
contract.

## Repo Fit

The current repository already contains most of the skeleton:

- Isaac-side navigation MVP:
  `docs/isaac-gaussian-navigation-mvp.md`
- existing live contract:
  `docs/slam-adapter-contract.md`
- default live-channel manifest:
  `contracts/live-contract.default.json`
- HTTP bridge pattern for native GS-SDF render frames:
  `scripts/gssdf_hifi_bridge_server.py`
- existing scene importers:
  `scripts/import_citygaussian_scene.py`
  `scripts/import_sgsslam_scene.py`

That means the next implementation step should be:

```text
add a new live profile for isaac-gaussian-online
```

instead of building a parallel one-off integration.

## Suggested Next Step

Build the first version in this order:

1. Isaac-native scene first or ReplicaCAD first.
2. ROS 2 publication of pose + RGB-D + semantics.
3. Standalone Gaussian render service with pose-driven rendering only.
4. Bridge rendered frames back into the Web UI and optional ROS 2 image topics.
5. Add semantic/map return messages.

If the goal is fastest visible progress, choose:

```text
Isaac office/warehouse -> ROS 2 -> Gaussian renderer service -> Web UI
```

If the goal is the nicest first visual scene, choose:

```text
ReplicaCAD baked lighting -> Isaac import/proxy mesh -> ROS 2 -> Gaussian renderer service
```

## External References

- Isaac Sim environment assets:
  https://docs.isaacsim.omniverse.nvidia.com/4.5.0/assets/usd_assets_environments.html
- Isaac Sim ROS 2 bridge:
  https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.ros2.bridge/docs/index.html
- Isaac Sim ROS 2 camera publishing:
  https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/tutorial_ros2_camera.html
- Omniverse neural rendering:
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/neural-rendering.html
- Omniverse particle fields / Gaussian splats:
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html
- Replica dataset:
  https://github.com/facebookresearch/Replica-Dataset
- ReplicaCAD:
  https://aihabitat.org/datasets/replica_cad/
- HM3D-Semantics:
  https://aihabitat.org/datasets/hm3d-semantics/
- Oxford RobotCar:
  https://robotcar-dataset.robots.ox.ac.uk/
