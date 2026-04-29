from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a human-readable KITTI raw download checklist for a target drive."
    )
    parser.add_argument("--date", default="2011_09_30", help="KITTI raw date")
    parser.add_argument("--drive", default="0020", help="KITTI raw drive id without prefix/suffix")
    parser.add_argument("--output", required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    drive_name = f"{args.date}_drive_{args.drive}_sync"

    text = f"""# KITTI Raw Download Checklist

目标序列：

- date: `{args.date}`
- drive: `{drive_name}`

建议下载的 archive：

1. `{args.date}_calib.zip`
2. `{drive_name}.zip`

下载后至少需要保留这些内容：

- `{drive_name}/image_00/data`
- `{drive_name}/image_01/data`
- `{drive_name}/image_00/timestamps.txt`
- `{drive_name}/image_01/timestamps.txt`
- `{drive_name}/velodyne_points/data`
- `{drive_name}/velodyne_points/timestamps.txt`
- `{drive_name}/oxts/data`
- `{drive_name}/oxts/timestamps.txt`
- `{args.date}/calib_cam_to_cam.txt`
- `{args.date}/calib_imu_to_velo.txt`
- `{args.date}/calib_velo_to_cam.txt`

本工作区默认的 benchmark 目标是把这些内容整理成：

- `cuVSLAM` stereo 输入
- `FAST-LIVO2` 多传感器输入
- `OXTS -> TUM GT`

拿到数据后，可直接运行：

```bash
python /home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_kitti_raw_benchmark.py \\
  --raw-root /path/to/KITTI/raw \\
  --date {args.date} \\
  --drive {args.drive} \\
  --frame-start 0 \\
  --frame-end 1100 \\
  --output-dir /home/nyu/Codespace/CVPR/nuc_output/kitti_raw_{args.date}_{args.drive}_benchmark
```
"""
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
