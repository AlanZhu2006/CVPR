# KITTI Raw Download Checklist

目标序列：

- date: `2011_09_30`
- drive: `2011_09_30_drive_0020_sync`

建议下载的 archive：

1. `2011_09_30_calib.zip`
2. `2011_09_30_drive_0020_sync.zip`

下载后至少需要保留这些内容：

- `2011_09_30_drive_0020_sync/image_00/data`
- `2011_09_30_drive_0020_sync/image_01/data`
- `2011_09_30_drive_0020_sync/image_00/timestamps.txt`
- `2011_09_30_drive_0020_sync/image_01/timestamps.txt`
- `2011_09_30_drive_0020_sync/velodyne_points/data`
- `2011_09_30_drive_0020_sync/velodyne_points/timestamps.txt`
- `2011_09_30_drive_0020_sync/oxts/data`
- `2011_09_30_drive_0020_sync/oxts/timestamps.txt`
- `2011_09_30/calib_cam_to_cam.txt`
- `2011_09_30/calib_imu_to_velo.txt`
- `2011_09_30/calib_velo_to_cam.txt`

本工作区默认的 benchmark 目标是把这些内容整理成：

- `cuVSLAM` stereo 输入
- `FAST-LIVO2` 多传感器输入
- `OXTS -> TUM GT`

拿到数据后，可直接运行：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_kitti_raw_benchmark.py \
  --raw-root /path/to/KITTI/raw \
  --date 2011_09_30 \
  --drive 0020 \
  --frame-start 0 \
  --frame-end 1100 \
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_2011_09_30_0020_benchmark
```
