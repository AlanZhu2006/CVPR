from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cuVSLAM stereo odometry on a prepared KITTI raw benchmark workspace."
    )
    parser.add_argument(
        "--sequence-dir",
        required=True,
        help="Prepared sequence directory containing image_0, image_1, calib.txt, times.txt.",
    )
    parser.add_argument("--output-trajectory", required=True, help="Output TUM trajectory path.")
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use timestamps_abs.txt if present for TUM timestamps; otherwise use times.txt relative timestamps.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit.")
    return parser.parse_args()


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    return q


def load_sequence(sequence_dir: Path) -> tuple[np.ndarray, tuple[int, int], list[float], list[float] | None]:
    intrinsics = np.loadtxt(sequence_dir / "calib.txt", usecols=range(1, 13))[:4].reshape(4, 3, 4)
    size = Image.open(sequence_dir / "image_0" / "000000.png").size
    times_rel = [float(line.strip()) for line in (sequence_dir / "times.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    abs_path = sequence_dir / "timestamps_abs.txt"
    times_abs = [float(line.strip()) for line in abs_path.read_text(encoding="utf-8").splitlines() if line.strip()] if abs_path.exists() else None
    return intrinsics, size, times_rel, times_abs


def main() -> None:
    args = parse_args()
    import cuvslam

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    output_traj = Path(args.output_trajectory).expanduser().resolve()
    output_traj.parent.mkdir(parents=True, exist_ok=True)

    intrinsics, size, times_rel, times_abs = load_sequence(sequence_dir)
    frame_count = len(times_rel)
    if args.max_frames > 0:
        frame_count = min(frame_count, args.max_frames)

    cameras = [cuvslam.Camera(), cuvslam.Camera()]
    for i in [0, 1]:
        cameras[i].size = size
        cameras[i].principal = [float(intrinsics[i][0][2]), float(intrinsics[i][1][2])]
        cameras[i].focal = [float(intrinsics[i].diagonal()[0]), float(intrinsics[i].diagonal()[1])]
    cameras[1].rig_from_camera.translation[0] = float(-intrinsics[1][0][3] / intrinsics[1][0][0])

    cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_final_landmarks_export=True,
        rectified_stereo_camera=True,
    )
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg)

    rows: list[str] = []
    tracked = 0
    failed = 0
    for frame in range(frame_count):
        images = [
            np.asarray(Image.open(sequence_dir / f"image_{cam}" / f"{frame:06d}.png"))
            for cam in [0, 1]
        ]
        timestamp_ns = int(times_rel[frame] * 1e9)
        odom_pose_estimate, _ = tracker.track(timestamp_ns, images)
        if odom_pose_estimate.world_from_rig is None:
            failed += 1
            continue
        pose = odom_pose_estimate.world_from_rig.pose
        t = pose.translation
        q = pose.rotation
        ts_out = times_abs[frame] if (args.absolute_time and times_abs is not None) else times_rel[frame]
        rows.append(
            f"{ts_out:.9f} {float(t[0]):.9f} {float(t[1]):.9f} {float(t[2]):.9f} "
            f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}"
        )
        tracked += 1

    output_traj.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "sequence_dir": str(sequence_dir),
        "trajectory_path": str(output_traj),
        "image_count": frame_count,
        "tracked_frames": tracked,
        "failed_frames": failed,
        "absolute_time": bool(args.absolute_time and times_abs is not None),
    }
    (output_traj.parent / "cuvslam_kitti_raw_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
