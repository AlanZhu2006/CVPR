# Project Structure

This repository is cleaned around the final-report story: **backend GPU video to
Gaussian is the main line**, and **live edge capture is the secondary systems
line**.

## Top-Level Folders

| Path | Role |
| --- | --- |
| `reports/` | Final report PDF, legacy short report, and LaTeX source. |
| `docs/` | Human-readable project map and runbooks. Start here before using old scripts. |
| `scripts/real2sim/` | Backend GPU / video real-to-sim utilities used by the LG-GS experiments. |
| `scripts/runbooks/` | Short executable command templates for the two maintained paths. |
| `HMR3D/nuc/scripts/` | Jetson/live capture, LingBot export, cuVSLAM, viewer, and conversion scripts. |
| `HMR3D/nuc/src/nuc_runtime/` | Runtime support code for live tracking, dense workers, fusion, Gaussian helpers, and policies. |
| `third_party_research/` | External research systems: LingBot-Map, GenWildSplat, SplaTAM. |
| `cuVSLAM/` | Local cuVSLAM source/checkout used for the edge tracking path. |

## Maintained Project Lines

### 1. Backend GPU Video-to-Gaussian

This is the main report path.

```text
video frames
  -> LingBot export
  -> GenWildSplat scene preparation
  -> LG-GS external LingBot geometry injection
  -> Gaussian PLY / safetensors
  -> inflated display PLY
  -> WebUI chunks
```

Primary files:

- `HMR3D/nuc/scripts/run_lingbot_export.py`
- `scripts/real2sim/prepare_genwildsplat_scene.py`
- `scripts/real2sim/inflate_gaussian_ply.py`
- `scripts/real2sim/postopt_genwildsplat_gaussian.py`
- `scripts/real2sim/register_genwildsplat_gaussian.py`
- `scripts/real2sim/register_genwild_chunk_atlas.py`
- `scripts/runbooks/gpu_video_to_gaussian.sh`

### 2. Live Edge Capture

This is the secondary systems path.

```text
HikRobot RGB
  -> cuVSLAM pose
  -> asynchronous LingBot dense worker
  -> RGB / trajectory / point cloud WebUI
```

Primary files:

- `HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py`
- `HMR3D/nuc/scripts/hikrobot_mvs_ros2_publisher.py`
- `HMR3D/nuc/scripts/launch_hikrobot_lingbot_real2sim_stack.sh`
- `HMR3D/nuc/scripts/sync_gs_console_monitor_assets.py`
- `HMR3D/nuc/scripts/render_gs_console_monitor_frame.py`
- `scripts/runbooks/live_edge_capture.sh`

## Historical / Lower-Priority Lines

The following are kept because they document useful experiments, but they should
not be presented as the current main project line:

- KITTI/cuVSLAM benchmark scripts
- SplaTAM smoke and conversion experiments
- Isaac/Nav2 prototypes
- semantic navigation UI prototype
- old global GenWildSplat variants that are not the active LG-GS path

## Naming

Use these names consistently in report and slides:

- **Mono2Sim-GS**: the whole project
- **LG-GS**: LingBot-Guided Gaussian Splatting, the main method
- **Route B**: old implementation name for LG-GS
- **backend GPU path**: offline/semi-offline video-to-Gaussian path
- **live edge path**: HikRobot/cuVSLAM/LingBot capture path

## What Not To Claim

The final report explicitly does not claim:

- completed simulator export
- robust global Gaussian optimization for the full long video
- complete TSDF/mesh/navigation result
- full online Gaussian SLAM

Those are future work built on top of this reconstruction layer.
