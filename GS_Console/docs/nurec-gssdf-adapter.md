# NuRec To GS-SDF Adapter

This adapter is for the real-data navigation path:

```text
NuRec / real robot images + pose + depth or mesh
-> GS-SDF colmap-style dataset
-> GS-SDF reconstruction
-> WebUI / Isaac navigation benchmark
```

It does not make the navigation module consume the mesh directly. The mesh is
only used as a geometry prior when per-frame depth or point clouds are not
available.

## Input

Use a NuRec scene or similar real-capture dataset with:

- RGB images
- a TUM trajectory: `timestamp tx ty tz qx qy qz qw`
- camera intrinsics
- either per-frame local depth PLYs or a world-frame mesh/point-cloud PLY

For NuRec, prefer scenes that include mesh/occupancy assets:

- `nova_carter-galileo`
- `nova_carter-cafe`
- `nova_carter-wormhole`
- `hand_hold_endeavor-*`

## Convert

Example with images, TUM trajectory, and mesh PLY:

```bash
python3 /home/chatsign/gs-sdf/scripts/nurec_to_gssdf_colmap.py \
  --input-root /media/chatsign/data-002/datasets/nurec/nova_carter-galileo \
  --output-root /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_gssdf_colmap \
  --image-dir images/front_stereo_camera_left \
  --tum-path training_trajectory_poses.tum \
  --mesh-ply nvblox_mesh/nvblox_mesh.ply \
  --fx 588.0 \
  --fy 588.0 \
  --cx 320.0 \
  --cy 240.0 \
  --frame-stride 4 \
  --max-frames 800 \
  --points-per-depth 20000 \
  --force
```

For `nova_carter-galileo`, prefer the front-left camera calibration inside
`3dgrt/last.usdz`. The converter reads `rig_trajectories.json` from that USDZ,
selects `front_stereo_camera_left`, and treats `T_sensor_rig` as
`camera-to-ego` with `camera-frame-axis=as-is`. This fixes the old failure mode
where the `stereo.edex` path placed the front-left camera below the mesh floor
and produced depth points mostly in the upper image.

NuRec image filenames are nanosecond timestamps, while the trajectory is in
seconds:

```bash
python3 /home/chatsign/gs-sdf/scripts/nurec_to_gssdf_colmap.py \
  --input-root /media/chatsign/data-002/datasets/nurec/nova_carter-galileo \
  --output-root /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_frontleft_auto3dgrt_surface20k_z4_114f_gssdf_colmap \
  --image-dir raw_images_front_left/rosbag_mapping_data/front_stereo_camera_left \
  --tum-path training_trajectory_poses.tum \
  --calibration-source auto \
  --camera-name front_stereo_camera_left \
  --mesh-ply extracted_assets/stage_volume/mesh.ply \
  --mesh-depth-mode surface \
  --mesh-surface-samples 2000000 \
  --zbuffer-pixel-stride 4 \
  --match-mode timestamp \
  --image-timestamp-scale auto \
  --timestamp-tolerance-sec 0.08 \
  --frame-stride 20 \
  --max-frames 120 \
  --points-per-depth 20000 \
  --ds-pt-num 20000 \
  --sdf-iter-step 800 \
  --gs-iter-step 1500 \
  --force
```

If `3dgrt/last.usdz` is missing, `--calibration-source auto` can fall back to
`--camera-extrinsics-json`, but that path should be considered lower priority
for this scene.

If the dataset already has per-frame local PLY depth clouds, use:

```bash
python3 /home/chatsign/gs-sdf/scripts/nurec_to_gssdf_colmap.py \
  --input-root /path/to/scene \
  --output-root /path/to/converted_gssdf_colmap \
  --image-dir images/front_stereo_camera_left \
  --tum-path training_trajectory_poses.tum \
  --depth-ply-dir depths_ply \
  --fx FX \
  --fy FY \
  --cx CX \
  --cy CY \
  --force
```

## Output

The converter writes:

```text
converted_gssdf_colmap/
  images/
  depths/
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/depths.txt
  gssdf_colmap.yaml
  nurec_to_gssdf_meta.json
```

The pose files are written in COLMAP world-to-camera convention. The input TUM
trajectory is interpreted as camera-to-world.

## Train

Use the existing containerized GS-SDF runner:

```bash
bash /home/chatsign/gs-sdf/scripts/train_gssdf.sh \
  /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_gssdf_colmap \
  /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_gssdf_colmap/gssdf_colmap.yaml \
  nurec-galileo-gssdf
```

For the native GS-SDF binary, use a host-side config whose `base_config` points
to the host repo and keep zero distortion fields absent:

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH
/media/chatsign/data-002/gs-sdf/runtime/builds/GS-SDF-writable-cuda124-torch27/neural_mapping_node \
  train \
  /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_frontleft_optical_surface_smoke_gssdf_colmap/gssdf_colmap.native_nodist_gs2500.yaml \
  /media/chatsign/data-002/gs-sdf/runtime/datasets/nurec_galileo_frontleft_optical_surface_smoke_gssdf_colmap
```

## Caveats

- Mesh-generated depth PLYs should use triangle surface sampling plus z-buffer
  filtering. Raw vertices alone often only cover sparse edges or top/bottom
  bands in the image.
- Do not emit `d0..d4: 0.0` for already-rectified or pinhole images. In
  GS-SDF, merely having these fields triggers OpenCV undistortion and silently
  changes the camera matrix, which can smear color supervision.
- For NuRec scenes with `3dgrt/last.usdz`, prefer `--calibration-source auto`
  or `--calibration-source 3dgrt`. Do not force the old `stereo.edex` path
  unless you are intentionally comparing calibrations.
- If using an older `stereo.edex`-only scene, the camera extrinsic may be
  body-axis-like. In that fallback case, `--camera-frame-axis
  vehicle-forward-to-opencv` may still be needed.
- A true sensor replay is better when NuRec provides raw stereo/depth data.
- NuRec USDZ/3DGUT assets are intended for Isaac Sim 5.x. The mesh/occupancy
  pieces are safer to adapt into the current Isaac Sim 4.5 workflow.
