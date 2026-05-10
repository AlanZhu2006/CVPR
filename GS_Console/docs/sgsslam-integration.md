# SGS-SLAM To Web UI

## Goal

Use an `SGS-SLAM` experiment directory directly in this web UI without forcing it into the `GS-SDF` runtime layout.

The current import path is:

1. Detect `params.ply` as the main gaussian asset
2. If only `params.npz` exists, export a browser-compatible gaussian PLY automatically
3. Export a lighter `params_rgb.ply` fallback from `params.npz` for `TopDownMap` and static point-cloud overlay
4. Reuse `gpu_input_root/<sequence>/pose` and `meta/camera_intrinsics.json` when available, so the browser starts from a reasonable trained view
5. Write `/home/chatsign/gs-sdf/examples/web-ui/public/scenes/<scene-id>/manifest.json`

## Current Local Repo

The repository was cloned to:

`/home/chatsign/sgs-slam`

Current checked-out commit:

`b99fbcc`

## Import Script

Use:

`/home/chatsign/gs-sdf/scripts/import_sgsslam_scene.py`

Basic example:

```bash
python3 /home/chatsign/gs-sdf/scripts/import_sgsslam_scene.py \
  --sgsslam-dir /home/chatsign/sgs-slam/experiments/FASTLIVO/fastlivo_rectified_1 \
  --scene-id sgs-fastlivo-rectified
```

Then open:

```text
http://localhost:5173/?scene=/scenes/sgs-fastlivo-rectified/manifest.json&mode=gs
```

This scene pack has already been generated locally at:

`/home/chatsign/gs-sdf/examples/web-ui/public/scenes/sgs-fastlivo-rectified/manifest.json`

## Recommended Import

For larger SGS-SLAM scenes, preprocess the gaussian asset into browser chunks:

```bash
python3 /home/chatsign/gs-sdf/scripts/import_sgsslam_scene.py \
  --sgsslam-dir /home/chatsign/sgs-slam/experiments/FASTLIVO/fastlivo_rectified_1 \
  --scene-id sgs-fastlivo-rectified \
  --preprocess-gaussian \
  --variant quality \
  --grid 6,6 \
  --max-sh 0 \
  --opacity-threshold 0.01
```

This path reuses:

- `/home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs`
- `/home/chatsign/gs-sdf/runtime/processed/<scene-id>/model`

## Default Asset Mapping

Current mapping strategy:

- gaussian:
  `params.ply`
- mesh:
  optional manual override only
- initial camera:
  infer from `gpu_input_root/<sequence>/pose/*.txt` when that input sequence can be resolved
- camera intrinsics:
  `gpu_input_root/<sequence>/meta/camera_intrinsics.json`
- occupancy:
  exported `params_rgb.ply` fallback
- raw point cloud:
  same exported `params_rgb.ply` fallback

Important implementation detail:

- the importer exports `params_rgb.ply` from `params.npz` with opacity filtering
- default export keeps only the highest-opacity `300000` points
- this keeps the 2D map and static overlay light enough for the browser

## Useful Flags

```text
--gaussian-path         override detected gaussian asset
--mesh-path             override mesh asset
--occupancy-path        force a specific PLY for TopDownMap
--raw-pointcloud-path   force a static point cloud asset
--gpu-input-sequence    explicitly choose gpu_input_root/<sequence>
--gpu-input-dir         explicitly choose a GPU input directory
--raw-opacity-threshold opacity threshold for exported params_rgb.ply
--raw-max-points        cap exported params_rgb.ply point count
--allow-gaussian-fallback
                        reuse gaussian PLY as occupancy/raw fallback when no RGB point cloud exists
--dry-run               print the manifest plan without writing files
```

## What Matches GS-SDF And What Does Not

What is close to current `GS-SDF` browser mode:

- browser `GS` view with gaussian rendering
- static point-cloud overlay
- 2D map fallback from projected point cloud
- initial scene framing from recovered camera poses

What is not the same as `GS-SDF`:

- no native `GS-SDF HiFi` renderer
- no `as_occ_prior.ply` style occupancy prior
- no `SDF mesh` unless you provide a mesh override manually
- semantic gaussians are not yet surfaced in this UI; the importer currently exposes RGB gaussian + RGB fallback point cloud
