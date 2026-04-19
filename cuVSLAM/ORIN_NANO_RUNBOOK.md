# cuVSLAM on Jetson Orin Nano

This runbook captures the setup and execution path that matches this machine:

- Jetson Orin Nano Developer Kit
- Ubuntu 22.04
- CUDA 12.6
- Python 3.10
- `cuvslam` wheel `15.0.0+cu12`

## Environment

Create the virtual environment:

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM
python3 -m venv .venv-jetson
.venv-jetson/bin/pip install --upgrade pip setuptools wheel
.venv-jetson/bin/pip install ./cuvslam-15.0.0+cu12-cp310-cp310-manylinux_2_35_aarch64.whl
.venv-jetson/bin/pip install -r examples/requirements.txt
.venv-jetson/bin/pip install pyrealsense2==2.57.7.10387
```

## Verified Offline Full Flow

This machine already passed the built-in map test:

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM/python/test
PYTHONPATH=/home/nyu/Codespace/CVPR/cuVSLAM/python/test \
  /home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python -m unittest -v test_map.TestMap.test_map
```

Run a full synthetic visualization demo that performs:

1. stereo tracking
2. SLAM mapping
3. map save
4. map load/localization
5. continued tracking after localization

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM
.venv-jetson/bin/python examples/orin_nano/run_synthetic_slam_demo.py
```

Headless Jetson systems default to `CUVSLAM_RERUN_MODE=save`, which writes:

- `outputs/orin_nano_demo/synthetic_slam.rrd`
- `outputs/orin_nano_demo/synthetic_map/`

To serve the viewer in a browser instead:

```bash
CUVSLAM_RERUN_MODE=web .venv-jetson/bin/python examples/orin_nano/run_synthetic_slam_demo.py
```

## RealSense Live Tracking

Check whether the camera is visible:

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM
.venv-jetson/bin/python examples/realsense/list_devices.py
```

Run single-stereo live tracking:

```bash
CUVSLAM_RERUN_MODE=save .venv-jetson/bin/python examples/realsense/run_stereo.py
```

The default output file is:

- `examples/realsense/recordings/realsense_tracking.rrd`

## RealSense Multi-Camera on Orin Nano

The multi-camera example now auto-selects `examples/realsense/frame_nano_rig.yaml` on Orin Nano.
You can also force it explicitly:

```bash
CUVSLAM_RERUN_MODE=save \
  .venv-jetson/bin/python examples/realsense/run_multicamera.py --rig-config frame_nano_rig.yaml
```

Before running, update the YAML with your actual camera serial numbers and extrinsics.
Use `examples/realsense/list_devices.py` to get the serials.

## KITTI Offline SLAM Demo

Download sequence `06` only:

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM/examples/kitti
/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson/bin/python download_kitti_sequence.py --sequence 06
```

Run the official KITTI SLAM example in headless mode:

```bash
cd /home/nyu/Codespace/CVPR/cuVSLAM
CUVSLAM_RERUN_MODE=save \
  .venv-jetson/bin/python examples/kitti/track_kitti_slam.py --force-remap
```

Useful options:

- `--max-frames 300` for a faster smoke test
- `--async-slam` to switch back to background-thread SLAM; the default is synchronous for offline Jetson stability
- `--rerun-mode web` to open the viewer from a browser on another machine

Default outputs:

- `outputs/kitti/track_kitti_slam.rrd`
- `examples/kitti/dataset/sequences/06/map/data.mdb`
- `examples/kitti/dataset/sequences/06/trajectory_tum.txt`

## Issue Notes Relevant to Orin Nano

- `cuVSLAM` release `v15.0.0` ships an `aarch64 + cu12 + cp310` wheel, so native Orin Nano setup does not require a source build.
- `cuVSLAM` issue `#36` states that VGA-class RealSense live tracking is expected to be feasible on Orin Nano; synchronization and frame delivery are the main practical bottlenecks.
- `cuVSLAM` issue `#39` and PR `#38` show that native Jetson builds may benefit from `--cuda_arch=87` when building from source on Orin Nano, but the current local setup uses the official wheel so this is not required.
- `isaac_ros_visual_slam` issue `#189` links to an NVIDIA forum resolution that JetPack 6.0 was unsupported for that Isaac ROS path. This machine is on L4T `36.4.7`, so the unsupported JetPack 6.0 case does not apply here.
