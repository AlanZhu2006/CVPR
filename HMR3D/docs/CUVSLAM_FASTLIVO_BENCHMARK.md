# cuVSLAM vs FAST-LIVO Benchmark

This document defines a fair benchmark pipeline for comparing:

- `cuVSLAM` as a visual-only frontend baseline
- `FAST-LIVO` as a LiDAR-IMU-Visual baseline

on the **same dataset**, with the **same ground-truth trajectory**, and with a **shared trajectory evaluation script**.

## Recommended Dataset

Use **NTU VIRAL** first.

Why:

- `FAST-LIVO` upstream explicitly provides an `NTU_VIRAL.yaml` config and `mapping_avia_ntu.launch`
- `NTU VIRAL` is a standard multi-sensor benchmark with camera, IMU, LiDAR, and GT
- it is a much more defensible basis for a fair `cuVSLAM vs FAST-LIVO` comparison than mixing unrelated datasets

## What Is Already Prepared

### 1. Shared trajectory benchmark

Use:

- [benchmark_cuvslam_vs_fastlivo.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/benchmark_cuvslam_vs_fastlivo.py:1)

It compares:

- `GT trajectory (TUM)`
- `cuVSLAM trajectory (TUM)`
- `FAST-LIVO trajectory (TUM)`

Outputs:

- `summary.json`
- `per_frame_translation_error.csv`
- `trajectory_topdown.png` when plotting works in the current environment

### 2. FAST-LIVO pose-output config generator

Use:

- [prepare_fastlivo_ntuviral_pose_config.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_fastlivo_ntuviral_pose_config.py:1)

This creates a copy of `NTU_VIRAL.yaml` with:

- `pose_output_en: true`

FAST-LIVO then writes:

- `camera_pose.txt`

which is already in TUM-like format:

`timestamp tx ty tz qx qy qz qw`

### 3. ROS bag extraction for cuVSLAM

Use:

- [extract_ntuviral_rosbag_for_cuvslam.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/extract_ntuviral_rosbag_for_cuvslam.py:1)

This extracts:

- `/left/image_raw`
- optional `/right/image_raw`

from a **ROS1 bag** into:

```text
output_dir/
  image_0/
  image_1/        # optional
  times.txt
  timestamps.csv
```

This script requires a ROS1 Python environment with:

- `rosbag`
- `cv_bridge`

### 4. cuVSLAM runner for NTU VIRAL

Use:

- [run_cuvslam_ntuviral.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_ntuviral.py:1)

This runs a **minimal monocular cuVSLAM baseline** on extracted `image_0` frames, using:

- [camera_NTU_VIRAL.yaml](/home/nyu/Codespace/CVPR/third_party_research/FAST-LIVO/config/camera_NTU_VIRAL.yaml:1)

It outputs:

- a TUM trajectory
- `cuvslam_ntuviral_summary.json`

## End-to-End Procedure

### A. Prepare FAST-LIVO

Generate a benchmark config with pose output:

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_fastlivo_ntuviral_pose_config.py \
  --output-yaml /home/nyu/Codespace/CVPR/nuc_output/ntuviral_fastlivo_pose/NTU_VIRAL_pose.yaml
```

Then use that config when launching FAST-LIVO.

Note:

- the upstream launch file loads `config/NTU_VIRAL.yaml`
- for a clean benchmark, either patch the launch file temporarily or copy the generated yaml over the original config in a benchmark branch/worktree

Expected FAST-LIVO output:

- `camera_pose.txt`

### B. Extract camera stream for cuVSLAM

Inside a ROS1 environment:

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/extract_ntuviral_rosbag_for_cuvslam.py \
  --bag /path/to/ntu_viral_sequence.bag \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam_input \
  --left-topic /left/image_raw
```

If the bag also contains a synchronized right camera topic and you want to experiment with stereo extraction:

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/extract_ntuviral_rosbag_for_cuvslam.py \
  --bag /path/to/ntu_viral_sequence.bag \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam_input \
  --left-topic /left/image_raw \
  --right-topic /right/image_raw
```

### C. Run cuVSLAM

Inside the cuVSLAM Python environment:

```bash
source /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_ntuviral.py \
  --sequence-dir /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam_input \
  --output-trajectory /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam/cuvslam_tum.txt
```

### D. Run the shared benchmark

Once you have:

- `gt_tum.txt`
- `cuvslam_tum.txt`
- `fastlivo_tum.txt`

run:

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/benchmark_cuvslam_vs_fastlivo.py \
  --gt /path/to/gt_tum.txt \
  --cuvslam /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam/cuvslam_tum.txt \
  --fastlivo /path/to/fastlivo_tum.txt \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/ntuviral_cuvslam_vs_fastlivo
```

## Important Caveat

This benchmark is only fair if:

- both systems run on the **same NTU VIRAL sequence**
- the **same GT** is used
- time association is consistent

Comparing `cuVSLAM on KITTI` against `FAST-LIVO on rosbag` is **not** a meaningful benchmark.

## Current Status

At the moment:

- the benchmark scripts are ready
- `FAST-LIVO` repo is cloned locally
- but we do **not** yet have a local NTU VIRAL sequence plus FAST-LIVO trajectory output in this workspace

So the benchmark pipeline is prepared, but the final numerical comparison still needs:

1. a downloaded NTU VIRAL sequence
2. a generated FAST-LIVO `camera_pose.txt`
3. the extracted camera stream for cuVSLAM
