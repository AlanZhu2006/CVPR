from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


EARTH_RADIUS_M = 6378137.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a KITTI raw drive for cuVSLAM / FAST-LIVO2 benchmarking by "
            "checking required files, slicing a frame range, creating a benchmark "
            "workspace, and exporting OXTS ground truth to TUM."
        )
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Root directory that contains KITTI raw date folders, e.g. /data/KITTI/raw",
    )
    parser.add_argument("--date", default="2011_09_30", help="KITTI raw date folder")
    parser.add_argument("--drive", default="0020", help="KITTI raw drive id without prefix/suffix")
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="Inclusive start frame index for the benchmark slice.",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=1100,
        help="Inclusive end frame index for the benchmark slice.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output benchmark workspace directory.",
    )
    return parser.parse_args()


def parse_timestamp_line(line: str) -> float:
    line = line.strip()
    if not line:
        raise ValueError("Empty timestamp line")
    # KITTI raw timestamps look like "2011-09-30 13:57:47.123456789"
    if "." in line:
        base, frac = line.split(".", 1)
        frac = (frac + "000000")[:6]
        dt = datetime.strptime(f"{base}.{frac}", "%Y-%m-%d %H:%M:%S.%f")
    else:
        dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp()


def load_timestamps(path: Path) -> list[float]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [parse_timestamp_line(line) for line in lines]


def parse_kitti_calib_file(path: Path) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, values_str = line.split(":", 1)
        try:
            values = [float(x) for x in values_str.strip().split()]
        except ValueError:
            # KITTI calibration files include metadata like `calib_time`.
            continue
        data[key] = np.array(values, dtype=np.float64)
    return data


def rotx(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def quat_from_rot(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def load_oxts_packet(path: Path) -> dict[str, float]:
    values = [float(v) for v in path.read_text(encoding="utf-8").strip().split()]
    if len(values) < 6:
        raise ValueError(f"Unexpected OXTS packet format in {path}")
    return {
        "lat": values[0],
        "lon": values[1],
        "alt": values[2],
        "roll": values[3],
        "pitch": values[4],
        "yaw": values[5],
    }


def oxts_to_pose(packet: dict[str, float], scale: float) -> np.ndarray:
    tx = scale * packet["lon"] * math.pi * EARTH_RADIUS_M / 180.0
    ty = scale * EARTH_RADIUS_M * math.log(math.tan((90.0 + packet["lat"]) * math.pi / 360.0))
    tz = packet["alt"]
    R = rotz(packet["yaw"]) @ roty(packet["pitch"]) @ rotx(packet["roll"])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target_path)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    drive_name = f"{args.date}_drive_{args.drive}_sync"
    drive_dir = raw_root / args.date / drive_name
    if not drive_dir.exists():
        raise SystemExit(f"Missing KITTI raw drive directory: {drive_dir}")

    image_00_dir = drive_dir / "image_00" / "data"
    image_01_dir = drive_dir / "image_01" / "data"
    velo_dir = drive_dir / "velodyne_points" / "data"
    oxts_dir = drive_dir / "oxts" / "data"
    image_00_ts_path = drive_dir / "image_00" / "timestamps.txt"
    image_01_ts_path = drive_dir / "image_01" / "timestamps.txt"
    velo_ts_path = drive_dir / "velodyne_points" / "timestamps.txt"
    oxts_ts_path = drive_dir / "oxts" / "timestamps.txt"
    calib_cam_to_cam = raw_root / args.date / "calib_cam_to_cam.txt"
    calib_imu_to_velo = raw_root / args.date / "calib_imu_to_velo.txt"
    calib_velo_to_cam = raw_root / args.date / "calib_velo_to_cam.txt"

    required_paths = [
        image_00_dir,
        image_01_dir,
        velo_dir,
        oxts_dir,
        image_00_ts_path,
        image_01_ts_path,
        velo_ts_path,
        oxts_ts_path,
        calib_cam_to_cam,
        calib_imu_to_velo,
        calib_velo_to_cam,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required KITTI raw files:\n- " + "\n- ".join(missing))

    image_00_paths = sorted(image_00_dir.glob("*.png"))
    image_01_paths = sorted(image_01_dir.glob("*.png"))
    velo_paths = sorted(velo_dir.glob("*.bin"))
    oxts_paths = sorted(oxts_dir.glob("*.txt"))
    frame_count = min(len(image_00_paths), len(image_01_paths), len(velo_paths), len(oxts_paths))
    if frame_count == 0:
        raise SystemExit(f"No synchronized frames found under {drive_dir}")

    frame_start = max(args.frame_start, 0)
    frame_end = min(args.frame_end, frame_count - 1)
    if frame_end < frame_start:
        raise SystemExit(f"Invalid frame range: start={frame_start}, end={frame_end}, frame_count={frame_count}")

    selected = slice(frame_start, frame_end + 1)
    image_00_paths = image_00_paths[selected]
    image_01_paths = image_01_paths[selected]
    velo_paths = velo_paths[selected]
    oxts_paths = oxts_paths[selected]

    image_times_abs = load_timestamps(image_00_ts_path)[selected]
    image_times_rel = [t - image_times_abs[0] for t in image_times_abs]
    velo_times_abs = load_timestamps(velo_ts_path)[selected]
    oxts_times_abs = load_timestamps(oxts_ts_path)[selected]

    oxts_packets = [load_oxts_packet(path) for path in oxts_paths]
    scale = math.cos(oxts_packets[0]["lat"] * math.pi / 180.0)
    first_pose_inv = np.linalg.inv(oxts_to_pose(oxts_packets[0], scale))

    gt_rows_rel: list[str] = []
    gt_rows_abs: list[str] = []
    for timestamp_rel, timestamp_abs, packet in zip(image_times_rel, image_times_abs, oxts_packets):
        T = first_pose_inv @ oxts_to_pose(packet, scale)
        q = quat_from_rot(T[:3, :3])
        t = T[:3, 3]
        row_rel = (
            f"{timestamp_rel:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
        row_abs = (
            f"{timestamp_abs:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
            f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}"
        )
        gt_rows_rel.append(row_rel)
        gt_rows_abs.append(row_abs)

    cuvslam_input_dir = output_dir / "cuvslam_input"
    fastlivo2_input_dir = output_dir / "fastlivo2_input"
    gt_dir = output_dir / "gt"
    manifest_dir = output_dir / "manifests"
    for directory in (cuvslam_input_dir, fastlivo2_input_dir, gt_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cuvslam_image_0 = cuvslam_input_dir / "image_0"
    cuvslam_image_1 = cuvslam_input_dir / "image_1"
    ensure_clean_dir(cuvslam_image_0)
    ensure_clean_dir(cuvslam_image_1)
    for out_idx, (left_path, right_path) in enumerate(zip(image_00_paths, image_01_paths)):
        ensure_symlink(cuvslam_image_0 / f"{out_idx:06d}.png", left_path)
        ensure_symlink(cuvslam_image_1 / f"{out_idx:06d}.png", right_path)
    (cuvslam_input_dir / "times.txt").write_text(
        "\n".join(f"{t:.6f}" for t in image_times_rel) + "\n",
        encoding="utf-8",
    )
    (cuvslam_input_dir / "timestamps_abs.txt").write_text(
        "\n".join(f"{t:.9f}" for t in image_times_abs) + "\n",
        encoding="utf-8",
    )
    (cuvslam_input_dir / "frame_ids.txt").write_text(
        "\n".join(f"{idx:06d}" for idx in range(frame_start, frame_end + 1)) + "\n",
        encoding="utf-8",
    )
    calib_data = parse_kitti_calib_file(calib_cam_to_cam)
    calib_keys = [f"P_rect_{i:02d}" for i in range(4)]
    missing_calib_keys = [key for key in calib_keys if key not in calib_data]
    if missing_calib_keys:
        raise SystemExit(
            "Missing required P_rect entries in calib_cam_to_cam.txt:\n- " + "\n- ".join(missing_calib_keys)
        )
    calib_rows = []
    for idx, key in enumerate(calib_keys):
        values = calib_data[key]
        if values.size != 12:
            raise SystemExit(f"Expected 12 values for {key}, got {values.size}")
        calib_rows.append(
            f"P{idx}: " + " ".join(f"{float(v):.12e}" for v in values.tolist())
        )
    (cuvslam_input_dir / "calib.txt").write_text("\n".join(calib_rows) + "\n", encoding="utf-8")

    ensure_symlink(fastlivo2_input_dir / "image_00", image_00_dir)
    ensure_symlink(fastlivo2_input_dir / "image_01", image_01_dir)
    ensure_symlink(fastlivo2_input_dir / "velodyne_points", velo_dir)
    ensure_symlink(fastlivo2_input_dir / "oxts_data", oxts_dir)
    ensure_symlink(fastlivo2_input_dir / "calib_cam_to_cam.txt", calib_cam_to_cam)
    ensure_symlink(fastlivo2_input_dir / "calib_imu_to_velo.txt", calib_imu_to_velo)
    ensure_symlink(fastlivo2_input_dir / "calib_velo_to_cam.txt", calib_velo_to_cam)
    (fastlivo2_input_dir / "image_00_timestamps.txt").write_text(
        "\n".join(f"{t:.9f}" for t in image_times_abs) + "\n",
        encoding="utf-8",
    )
    (fastlivo2_input_dir / "velodyne_timestamps.txt").write_text(
        "\n".join(f"{t:.9f}" for t in velo_times_abs) + "\n",
        encoding="utf-8",
    )
    (fastlivo2_input_dir / "oxts_timestamps.txt").write_text(
        "\n".join(f"{t:.9f}" for t in oxts_times_abs) + "\n",
        encoding="utf-8",
    )

    (gt_dir / "gt_tum_relative.txt").write_text("\n".join(gt_rows_rel) + "\n", encoding="utf-8")
    (gt_dir / "gt_tum_absolute.txt").write_text("\n".join(gt_rows_abs) + "\n", encoding="utf-8")

    manifest = {
        "raw_root": str(raw_root),
        "date": args.date,
        "drive": args.drive,
        "drive_name": drive_name,
        "drive_dir": str(drive_dir),
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count_total": frame_count,
        "frame_count_selected": len(image_00_paths),
        "cuvslam_sequence_dir": str(cuvslam_input_dir),
        "fastlivo2_sequence_dir": str(fastlivo2_input_dir),
        "gt_relative_tum": str(gt_dir / "gt_tum_relative.txt"),
        "gt_absolute_tum": str(gt_dir / "gt_tum_absolute.txt"),
        "notes": [
            "cuvslam_input uses symlinks to the full image_00/image_01 folders; use frame_ids.txt to slice the odometry window.",
            "fastlivo2_input contains symlinks and timestamp text files but not a ROS2 bag. A separate adapter or converter is still required.",
            "GT is derived from OXTS using the standard KITTI Mercator projection and normalized to the first selected pose.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    download_manifest = {
        "recommended_archives": [
            f"{args.date}_calib.zip",
            f"{drive_name}.zip",
        ],
        "required_inside_drive_zip": [
            "image_00/data",
            "image_01/data",
            "image_00/timestamps.txt",
            "image_01/timestamps.txt",
            "velodyne_points/data",
            "velodyne_points/timestamps.txt",
            "oxts/data",
            "oxts/timestamps.txt",
        ],
        "required_calibration_files": [
            "calib_cam_to_cam.txt",
            "calib_imu_to_velo.txt",
            "calib_velo_to_cam.txt",
        ],
    }
    (manifest_dir / "kitti_raw_download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Prepared KITTI raw benchmark workspace at {output_dir}")
    print(f"- cuVSLAM input: {cuvslam_input_dir}")
    print(f"- FAST-LIVO2 input: {fastlivo2_input_dir}")
    print(f"- GT (relative): {gt_dir / 'gt_tum_relative.txt'}")
    print(f"- GT (absolute): {gt_dir / 'gt_tum_absolute.txt'}")


if __name__ == "__main__":
    main()
