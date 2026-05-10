# Isaac Gaussian Navigation MVP

This document is the working README for the first Isaac Sim / Isaac Lab
navigation prototype.

The goal is not to prove that a Gaussian scene can be opened in a viewer. The
goal is to build a minimal physically grounded navigation evaluation loop:

```text
real data / rosbag
-> Gaussian visual layer
-> mesh or proxy collision layer
-> 2D navigation map
-> Isaac Sim scene
-> robot navigation goals
-> failure logs
```

## Current Workspace Status

Checked on `2026-04-22` in this workspace:

- Machine: `x86_64`, Ubuntu 22.04, RTX A6000 48 GB.
- Existing Docker images include `gs_sdf_img:latest` and `gs-sdf-ros-tools:latest`.
- Isaac MVP Python environment:
  `/media/chatsign/data-002/isaac/env_isaacsim45_lab21`
- Isaac Sim installed in that environment:
  `isaacsim==4.5.0.0`
- Isaac Lab source checkout:
  `/media/chatsign/data-002/isaac/IsaacLab` at tag `v2.1.1`
- Isaac Lab editable packages installed:
  `isaaclab`, `isaaclab_assets`, `isaaclab_tasks`, `isaaclab_rl`
- PyTorch pinned to `torch==2.5.1+cu121` for the current 550 driver.
- Existing playback cache:
  `runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16`
- Existing scene pack:
  `examples/web-ui/public/scenes/fast-livo2-compressed-live/manifest.json`
- Existing GS-SDF output:
  `runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml`
- Isaac Sim headless first-run has completed after explicit NVIDIA EULA
  acceptance by the user. A one-frame smoke test started and shut down
  successfully.
- Generated Isaac collision USD:
  `/media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd`
- Generated 2D navigation map:
  `/media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml`
- The root filesystem is effectively full, so Isaac and its caches should live
  under `/media/chatsign/data-002`, not under `/`.

The useful existing assets for the MVP are:

| Layer | Current source |
| --- | --- |
| Playback / trajectory | `runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16` |
| Gaussian visual layer | `runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/model/gs.ply` |
| Mesh / proxy candidate | `runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/mesh_gs_.ply` |
| Occupancy / map candidate | `runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/model/as_occ_prior.ply` |
| Browser validation scene | `examples/web-ui/public/scenes/fast-livo2-compressed-live/manifest.json` |

Generated Isaac-side assets:

| Layer | Generated asset | Status |
| --- | --- | --- |
| Collision smoke test | `/media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_smoke.usd` | 10k-face format/schema validation |
| Collision full mesh | `/media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd` | 5,046,326 vertices, 2,614,220 faces, `CollisionAPI=True`, `MeshCollisionAPI=meshSimplification` |
| 2D navigation map | `/media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml` | 0.2 m/pixel, 546 x 969, z slice 0.2-2.0 m, 0.4 m inflation |

## MVP Scope

First version:

1. Reuse the existing FAST-LIVO / GS-SDF scene.
2. Verify the playback trajectory, RGB, and map in the current web UI.
3. Convert or import the mesh/proxy layer into an Isaac-compatible USD asset.
4. Generate a simple 2D navigation map from the existing occupancy or mesh.
5. Spawn one robot in Isaac Sim.
6. Run 20 fixed or random navigation goals.
7. Save a small failure report: success, collision, timeout, stuck, map issue.

Out of scope for the first version:

- Online Gaussian updates.
- Dynamic humans.
- Four-legged locomotion policy training.
- One million seeds.
- Full photoreal NuRec pipeline.
- Automatic semantic object decomposition.

Those are follow-up milestones after the static scene loop works.

## Readiness Check

Run:

```bash
bash /media/chatsign/data-002/gs-sdf/scripts/check_isaac_nav_mvp_host.sh
```

Expected current result:

- GPU, RAM, data disk, and Docker should pass.
- Root disk should be the only expected warning because `/` is nearly full.
- Isaac Sim / Isaac Lab should be detected from
  `/media/chatsign/data-002/isaac/env_isaacsim45_lab21` and
  `/media/chatsign/data-002/isaac/IsaacLab`.
- Existing GS-SDF cache and scene assets should be detected.

## Phase 0: Keep All New Isaac Files on the Data Disk

Use these locations unless there is a strong reason to change them:

```text
/media/chatsign/data-002/isaac/isaac-sim
/media/chatsign/data-002/isaac/IsaacLab
/media/chatsign/data-002/isaac/cache
/media/chatsign/data-002/isaac/nav-mvp
```

Recommended environment variables:

```bash
export ISAAC_WORK_ROOT=/media/chatsign/data-002/isaac
export ISAAC_NAV_MVP_ENV=$ISAAC_WORK_ROOT/env_isaacsim45_lab21
export ISAACSIM_PYTHON_EXE=$ISAAC_NAV_MVP_ENV/bin/python
export ISAACLAB_PATH=$ISAAC_WORK_ROOT/IsaacLab
export PIP_CACHE_DIR=$ISAAC_WORK_ROOT/cache/pip
export XDG_CACHE_HOME=$ISAAC_WORK_ROOT/cache/xdg
export TMPDIR=$ISAAC_WORK_ROOT/tmp
```

Do not install Isaac under `/home` or `/opt` on this machine unless the root
filesystem has been cleaned first.

Activate the environment with:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
```

Do not run `/media/chatsign/data-002/isaac/IsaacLab/isaaclab.sh -i` on this
checkout without editing it first. In this tag, that helper forcibly reinstalls
`torch==2.7.0+cu128`, which is not the compatibility target for this machine.
The current environment was installed manually with editable package installs
to preserve `torch==2.5.1+cu121`.

## Phase 1: Validate the Existing Scene

Start the existing playback stack:

```bash
VIEW_WIDTH=1920 VIEW_HEIGHT=1536 MJPEG_MAX_FPS=90 START_PAUSED=1 \
bash /media/chatsign/data-002/gs-sdf/scripts/launch_fastlivo_cached_playback_stack.sh \
  /media/chatsign/data-002/gs-sdf/runtime/playback-cache/fast_livo2_compressed_rgbdense_kf16 \
  fastlivo-cached-playback \
  9090 \
  5173 \
  8765
```

Open:

```text
http://localhost:5173/?mode=playback
```

Verify:

- RGB playback is visible.
- 3D point cloud and trajectory move correctly.
- 2D map appears.
- Start/goal editing in the 2D map is usable.

This proves the current data is good enough to become the first Isaac scene.

## Phase 2: Build Isaac Scene Layers

The Isaac scene should be split into layers:

| Layer | Purpose | Current input | Target |
| --- | --- | --- | --- |
| Visual | Photoreal / Gaussian visual reference | `model/gs.ply` | NuRec USDZ or visual-only USD |
| Collision | Robot contacts and walls | `mesh_gs_.ply` | `/media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd` |
| Navigation | Planner map | `as_occ_prior.ply` | `/media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml` |
| Robot | Agent under test | Isaac asset or URDF/MJCF | USD / URDF / MJCF |
| Evaluation | Goal list and results | generated | CSV/JSON logs |

Important rule:

```text
Gaussian != physics.
```

The Gaussian layer can make the scene look real. The collision layer decides
where the robot can stand, collide, and navigate.

The current collision USD was generated with:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES "$ISAACSIM_PYTHON_EXE" \
  /media/chatsign/data-002/gs-sdf/scripts/isaac_nav_ply_to_usd_collision.py \
  --input /media/chatsign/data-002/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/mesh_gs_.ply \
  --output /media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd \
  --collision-approximation meshSimplification
```

The current 2D navigation map was generated with:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
python /media/chatsign/data-002/gs-sdf/scripts/build_isaac_nav_occupancy_map.py \
  --input /media/chatsign/data-002/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/model/as_occ_prior.ply \
  --output-prefix /media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2 \
  --resolution 0.2 \
  --z-min 0.2 \
  --z-max 2.0 \
  --inflate-radius 0.4
```

This map is a first navigation proxy, not a final semantic floor plan. It marks
points in the selected height slice as obstacles and treats the remaining cells
as free, so the next validation step is to spawn a robot and check whether the
map and collision mesh agree.

## Phase 2.5: Minimal Physics Validation

Use the standalone validator to answer the first hard question:

```text
Can a simple agent spawn in this scene, touch the collision mesh, and settle
without tunneling through it or exploding?
```

Run:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
export OMNI_KIT_ACCEPT_EULA=YES
"$ISAACSIM_PYTHON_EXE" /media/chatsign/data-002/gs-sdf/scripts/validate_isaac_nav_scene.py \
  --headless \
  --collision-usd /media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd \
  --mesh-ply /media/chatsign/data-002/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/mesh_gs_.ply \
  --map-yaml /media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml \
  --output /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json
```

What the validator does:

- loads the existing collision USD into Isaac
- projects the source mesh into the existing 2D map frame
- picks a start cell that is both free in the map and locally supported by a flat-enough mesh patch
- spawns one capsule agent above that start pose
- steps physics and records contact, stability, min/max z, and final pose
- samples 20 navigation-goal positions from the same free-space map for the next stage

Important finding from the first run on `2026-04-22`:

- using the map center as a start pose was not sufficient; the agent fell through because a generic free cell does not guarantee a floor directly underneath
- after switching to a start cell with local flat mesh support, the same collision USD passed the minimal contact test

Current report path:

```text
/media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json
```

Current interpretation:

- existing `mesh_gs_.ply` plus the generated collision USD are good enough for the first physics check
- existing occupancy map is good enough to sample candidate starts and 20 goals
- existing `gs.ply` is still not part of the Isaac scene; it remains a visual-only asset until it is converted into an Isaac-loadable visual layer

## Phase 3: First Navigation Evaluation

The first evaluation is still intentionally small:

- one static scene
- one robot
- 20 goals
- fixed random seed list
- no moving humans
- no manipulation objects

The current runner is:

```text
/media/chatsign/data-002/gs-sdf/scripts/run_minimal_nav_episodes.py
```

Use the already validated safe start cell and then run the thin episode loop:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
export OMNI_KIT_ACCEPT_EULA=YES
"$ISAACSIM_PYTHON_EXE" /media/chatsign/data-002/gs-sdf/scripts/validate_isaac_nav_scene.py \
  --headless \
  --start-cell 266,488 \
  --collision-usd /media/chatsign/data-002/isaac/nav-mvp/assets/fast_livo2_collision_full.usd \
  --mesh-ply /media/chatsign/data-002/gs-sdf/runtime/output/2026-04-03-17-16-16_fast_livo2_compressed.bag_fastlivo_cbd_quality_v2.yaml/mesh_gs_.ply \
  --map-yaml /media/chatsign/data-002/isaac/nav-mvp/maps/fast_livo2_nav_z0p2_2p0_r0p2.yaml \
  --output /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json

"$ISAACSIM_PYTHON_EXE" /media/chatsign/data-002/gs-sdf/scripts/run_minimal_nav_episodes.py \
  --headless \
  --device cpu \
  --goal-tolerance 0.1 \
  --validation-report /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json \
  --output /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_nav_episodes.json \
  --timeout-sec 8
```

For viewer mode in Isaac Sim, the same runner now supports:

- visible per-goal pillar markers
- planned path polyline rendering
- a closer follow camera
- velocity-based motion instead of direct pose teleportation
- local occupancy-based obstacle-avoidance steering on top of the global A* path
- optional scene recentering so the validated start becomes `(0, 0)`

Example viewer command:

```bash
source /media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY=:0

"$ISAACSIM_PYTHON_EXE" /media/chatsign/data-002/gs-sdf/scripts/run_minimal_nav_episodes.py \
  --rendering_mode performance \
  --device cpu \
  --goal-tolerance 0.1 \
  --validation-report /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json \
  --output /media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_nav_episodes.json \
  --timeout-sec 8 \
  --episode-limit 5 \
  --step-sleep 0.02 \
  --episode-hold-sec 1.0 \
  --hold-open-sec 30 \
  --show-goals \
  --show-path \
  --follow-camera \
  --floating-agent \
  --path-lookahead-distance 0.75 \
  --avoidance-radius 0.7 \
  --avoidance-gain 1.1 \
  --recenter-start-to-origin
```

Current report path:

```text
/media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_nav_episodes.json
```

Current result on `2026-04-22`:

- `success`
- `collision`
- `timeout`
- `stuck`

Observed counts in the current report:

- `success = 15`
- `collision = 5`
- `timeout = 0`
- `stuck = 0`

Current interpretation:

- the minimal navigation agent can now run 20 episodes end to end and emit machine-readable results
- the current episode set is sourced from `start_component_resampled`, not from the original 20 map-wide random goals, because the physically validated start cell sits on a very small supported connected component
- the 5 collision outcomes are all `occupied_or_oob`, which means the local support/component proxy and the occupancy map still disagree near some neighboring cells
- this phase is currently CPU-only; the equivalent GPU kinematic reset loop triggered PhysX tensor/fabric CUDA illegal-memory-access failures in this environment
- this is a thin evaluation loop, not yet a robot policy benchmark; the next step is to replace the resampled local-goal set with connected, nontrivial goals from a larger physically valid region

## Definition of Done for the MVP

The first MVP is done when all of these are true:

1. The existing GS-SDF scene has an Isaac-compatible environment asset.
2. The environment has a collision layer independent of the Gaussian visual.
3. A 2D navigation map exists for the same coordinate frame.
4. A robot can spawn without falling through or exploding.
5. 20 navigation goals run without manual intervention.
6. Results are written to a machine-readable report.

After that, the next meaningful upgrades are:

- dynamic obstacles / humans
- quadruped robot instead of a wheeled robot
- automatic map/proxy generation from the reconstruction
- hundreds or thousands of seeds
- online updates from a new rosbag or live sensor stream
