from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal cuVSLAM monocular benchmark on NTU VIRAL extracted images."
    )
    parser.add_argument("--sequence-dir", required=True, help="Directory with image_0 and times.txt.")
    parser.add_argument(
        "--camera-yaml",
        default="/home/nyu/Codespace/CVPR/datasets/ntu_viral/spms_02/spms_02/camera_left.yaml",
        help="Left-camera YAML path.",
    )
    parser.add_argument(
        "--right-camera-yaml",
        default="",
        help="Optional right-camera YAML path. If provided and image_1 exists, run stereo/multicamera odometry.",
    )
    parser.add_argument("--output-trajectory", required=True, help="Output TUM trajectory path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit.")
    parser.add_argument(
        "--timestamps-csv",
        default="",
        help="Optional CSV with absolute timestamps exported during bag extraction. If set, TUM timestamps use left_stamp instead of times.txt relative time.",
    )
    return parser.parse_args()


def parse_scalar(text: str, key: str) -> float:
    pattern = rf"{re.escape(key)}:\s*([-+eE0-9\.]+)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find scalar {key} in YAML")
    return float(match.group(1))


def parse_opencv_matrix(text: str, key: str) -> np.ndarray:
    pattern = rf"{re.escape(key)}:\s*!!opencv-matrix\s*rows:\s*(\d+)\s*cols:\s*(\d+)\s*dt:\s*\w+\s*data:\s*\[([^\]]+)\]"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {key} in YAML")
    rows = int(match.group(1))
    cols = int(match.group(2))
    values = [float(v.strip()) for v in match.group(3).replace("\n", " ").split(",") if v.strip()]
    return np.array(values, dtype=np.float64).reshape(rows, cols)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    m = R
    trace = np.trace(m)
    if trace > 0:
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
    return q / max(np.linalg.norm(q), 1e-12)


def invert_pose_matrix(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def load_camera(camera_yaml: Path):
    import yaml
    import cuvslam

    text = camera_yaml.read_text(encoding="utf-8")
    config = yaml.safe_load(text.replace("%YAML:1.0", "").replace("!!opencv-matrix", ""))
    if "cam_width" not in config:
        cam, _ = load_ntu_viral_camera(camera_yaml)
        return cam
    cam = cuvslam.Camera()
    cam.size = [int(config["cam_width"]), int(config["cam_height"])]
    cam.focal = [float(config["cam_fx"]), float(config["cam_fy"])]
    cam.principal = [float(config["cam_cx"]), float(config["cam_cy"])]
    cam.distortion = cuvslam.Distortion(
        cuvslam.Distortion.Model.Brown,
        [
            float(config["cam_d0"]),
            float(config["cam_d1"]),
            0.0,
            float(config["cam_d2"]),
            float(config["cam_d3"]),
        ],
    )
    cam.border_top = 10
    cam.border_bottom = 10
    cam.border_left = 10
    cam.border_right = 10
    return cam


def load_ntu_viral_camera(camera_yaml: Path):
    import cuvslam

    text = camera_yaml.read_text(encoding="utf-8")
    cam = cuvslam.Camera()
    cam.size = [int(parse_scalar(text, "image_width")), int(parse_scalar(text, "image_height"))]
    cam.focal = [parse_scalar(text, "fx"), parse_scalar(text, "fy")]
    cam.principal = [parse_scalar(text, "cx"), parse_scalar(text, "cy")]
    cam.distortion = cuvslam.Distortion(
        cuvslam.Distortion.Model.Brown,
        [
            parse_scalar(text, "k1"),
            parse_scalar(text, "k2"),
            parse_scalar(text, "p1"),
            parse_scalar(text, "p2"),
            0.0,
        ],
    )
    cam.border_top = 10
    cam.border_bottom = 10
    cam.border_left = 10
    cam.border_right = 10
    return cam, parse_opencv_matrix(text, "T_Body2Cam")


def main() -> None:
    args = parse_args()
    import cuvslam

    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    camera_yaml = Path(args.camera_yaml).expanduser().resolve()
    output_traj = Path(args.output_trajectory).expanduser().resolve()
    output_traj.parent.mkdir(parents=True, exist_ok=True)

    left_dir = sequence_dir / "image_0"
    right_dir = sequence_dir / "image_1"
    left_paths = sorted(left_dir.glob("*.png"))
    if not left_paths:
        raise SystemExit(f"No png frames found in {left_dir}")

    times = [float(line.strip()) for line in (sequence_dir / "times.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(times) < len(left_paths):
        raise SystemExit("times.txt has fewer timestamps than images")

    absolute_times: list[float] | None = None
    right_names: list[str] | None = None
    if args.timestamps_csv:
        import csv

        absolute_times = []
        right_names = []
        with Path(args.timestamps_csv).expanduser().resolve().open("r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                left_stamp = row.get("left_stamp", "").strip()
                if left_stamp:
                    absolute_times.append(float(left_stamp))
                    right_names.append(row.get("right_image", "").strip())
        if len(absolute_times) < len(left_paths):
            raise SystemExit("timestamps.csv has fewer absolute timestamps than images")

    stereo_enabled = bool(args.right_camera_yaml) and right_dir.exists()
    if stereo_enabled:
        left_camera, T_body_to_left = load_ntu_viral_camera(camera_yaml)
        right_camera, T_body_to_right = load_ntu_viral_camera(Path(args.right_camera_yaml).expanduser().resolve())
        T_left_to_right = T_body_to_left @ invert_pose_matrix(T_body_to_right)
        right_camera.rig_from_camera.translation = T_left_to_right[:3, 3].astype(np.float64)
        right_camera.rig_from_camera.rotation = rot_to_quat(T_left_to_right[:3, :3]).astype(np.float64)
        rig = cuvslam.Rig([left_camera, right_camera])
    else:
        camera = load_camera(camera_yaml)
        rig = cuvslam.Rig([camera])
    cfg = cuvslam.Tracker.OdometryConfig()
    if stereo_enabled and hasattr(cuvslam.Tracker.OdometryMode, "Multicamera"):
        cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Multicamera
    elif hasattr(cuvslam.Tracker.OdometryMode, "Mono"):
        cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
    tracker = cuvslam.Tracker(rig, cfg)

    frame_count = len(left_paths)
    if stereo_enabled and right_names:
        valid_pairs = [(lp, right_dir / rn) for lp, rn in zip(left_paths, right_names) if rn]
        if not valid_pairs:
            raise SystemExit("No valid right-image pairs found in timestamps.csv")
        if args.max_frames > 0:
            valid_pairs = valid_pairs[: args.max_frames]
        left_paths = [pair[0] for pair in valid_pairs]
        right_paths = [pair[1] for pair in valid_pairs]
        frame_count = len(left_paths)
    else:
        if args.max_frames > 0:
            left_paths = left_paths[: args.max_frames]
        right_paths = []
        frame_count = len(left_paths)

    rows: list[str] = []
    tracked = 0
    failed = 0
    for idx, left_path in enumerate(left_paths):
        image_left = np.asarray(Image.open(left_path).convert("L"))
        timestamp_ns = int(times[idx] * 1e9)
        if stereo_enabled:
            image_right = np.asarray(Image.open(right_paths[idx]).convert("L"))
            images = [image_left, image_right]
        else:
            images = [image_left]
        pose_estimate, _ = tracker.track(timestamp_ns, images)
        if pose_estimate.world_from_rig is None:
            failed += 1
            continue
        pose = pose_estimate.world_from_rig.pose
        t = pose.translation
        q = pose.rotation
        timestamp_out = absolute_times[idx] if absolute_times is not None else times[idx]
        rows.append(
            f"{timestamp_out:.9f} {float(t[0]):.9f} {float(t[1]):.9f} {float(t[2]):.9f} "
            f"{float(q[0]):.9f} {float(q[1]):.9f} {float(q[2]):.9f} {float(q[3]):.9f}"
        )
        tracked += 1

    output_traj.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    summary = {
        "sequence_dir": str(sequence_dir),
        "camera_yaml": str(camera_yaml),
        "right_camera_yaml": str(Path(args.right_camera_yaml).expanduser().resolve()) if stereo_enabled else None,
        "stereo_enabled": stereo_enabled,
        "image_count": frame_count,
        "tracked_frames": tracked,
        "failed_frames": failed,
        "trajectory_path": str(output_traj),
    }
    (output_traj.parent / "cuvslam_ntuviral_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
