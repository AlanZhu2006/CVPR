# CityGaussian To Web UI

## Goal

Use a cloned `CityGaussian` output directory directly in this web UI without forcing it into the `GS-SDF` runtime layout.

The minimal import path is:

1. Detect the latest gaussian asset from `point_cloud/iteration_*/point_cloud.ply`
2. Detect `fuse_post.ply` as the optional mesh
3. Reuse a PLY fallback for the 2D occupancy view when no dedicated occupancy prior exists
4. Write `/home/chatsign/gs-sdf/examples/web-ui/public/scenes/<scene-id>/manifest.json`

## Current Local Clone

The repository was cloned to:

`/home/chatsign/work/CityGaussian`

Current checked-out commit:

`e84c7c8`

## Import Script

Use:

`/home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py`

Basic example:

```bash
python3 /home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py \
  --citygaussian-dir /home/chatsign/work/CityGaussian/outputs/your_scene \
  --scene-id citygs-your-scene
```

Then open:

```text
http://localhost:5173/?scene=/scenes/citygs-your-scene/manifest.json&mode=gs
```

## Recommended Import

For large scenes, chunk the gaussian asset before opening it in the browser:

```bash
python3 /home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py \
  --citygaussian-dir /home/chatsign/work/CityGaussian/outputs/your_scene \
  --scene-id citygs-your-scene \
  --preprocess-gaussian \
  --variant quality \
  --grid 6,6 \
  --max-sh 2 \
  --opacity-threshold 0.005
```

That path reuses:

- `/home/chatsign/gs-sdf/scripts/preprocess_gaussian_stream.mjs`
- `/home/chatsign/gs-sdf/runtime/processed/<scene-id>/model`

## If Only A Checkpoint Exists

If the CityGaussian output directory only has `checkpoints/*.ckpt`, let the importer export a PLY first:

```bash
python3 /home/chatsign/gs-sdf/scripts/import_citygaussian_scene.py \
  --citygaussian-dir /home/chatsign/work/CityGaussian/outputs/your_scene \
  --scene-id citygs-your-scene \
  --auto-export-ckpt
```

This calls CityGaussian's own exporter:

`/home/chatsign/work/CityGaussian/utils/ckpt2ply.py`

## Asset Mapping

Current mapping strategy:

- gaussian:
  latest `point_cloud/iteration_*/point_cloud.ply`
- mesh:
  `fuse_post.ply` when present
- initial camera:
  infer from `cameras.json` when present, so the Web UI starts from a training view instead of framing the whole scene bounds
- occupancy:
  fallback to mesh PLY or another explicit PLY
- raw point cloud:
  fallback to occupancy or mesh

Important limitation:

- CityGaussian does not naturally produce `GS-SDF` style `as_occ_prior.ply`
- the 2D map therefore uses a projected PLY fallback, which is useful for inspection but not semantically identical to `GS-SDF` occupancy prior
- by default, the importer does not reuse the gaussian PLY itself as occupancy/raw fallback, because that can force the browser to load the same huge asset twice

## Useful Flags

```text
--gaussian-path         override the detected gaussian asset
--mesh-path             override the detected mesh
--occupancy-path        force a specific PLY for TopDownMap
--raw-pointcloud-path   force a static point cloud asset
--allow-gaussian-fallback
                        reuse gaussian PLY as occupancy/raw fallback
--leaf-size             set map resolution
--map-size              override inferred square map size
--map-origin-z          override ground Z
--dry-run               print the manifest plan without writing files
```
