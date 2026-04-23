from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a prepared KITTI raw benchmark workspace to a FAST-LIVO2 ROS2 bag "
            "and generate FAST-LIVO2 parameter files from KITTI calibration."
        )
    )
    parser.add_argument("--workspace", required=True, help="Output workspace from prepare_kitti_raw_benchmark.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for bag, configs, and run helper.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick smoke tests. 0 means all frames.")
    parser.add_argument("--image-topic", default="/left/image_raw")
    parser.add_argument("--lidar-topic", default="/os1_cloud_node1/points")
    parser.add_argument("--imu-topic", default="/imu/imu")
    parser.add_argument("--bag-name", default="kitti_raw_fastlivo2_ros2")
    return parser.parse_args()


def read_floats_calib(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            out[key] = [float(x) for x in raw.split()]
        except ValueError:
            continue
    return out


def read_timestamps(path: Path) -> list[float]:
    return [float(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stamp_msg(time_cls, ts: float):
    sec = int(math.floor(ts))
    nanosec = int(round((ts - sec) * 1_000_000_000))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return time_cls(sec=sec, nanosec=nanosec)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def infer_hdl64_ring(points_xyz: np.ndarray) -> np.ndarray:
    horizontal_range = np.linalg.norm(points_xyz[:, :2], axis=1)
    vertical_angle = np.degrees(np.arctan2(points_xyz[:, 2], np.maximum(horizontal_range, 1e-6)))
    # KITTI HDL-64E vertical FOV is roughly -24.9 to +2.0 degrees.
    ring = np.rint((vertical_angle + 24.9) / (2.0 + 24.9) * 63.0)
    return np.clip(ring, 0, 63).astype(np.uint16)


def make_header(types: dict, ts: float, frame_id: str):
    return types["std_msgs/msg/Header"](stamp=stamp_msg(types["builtin_interfaces/msg/Time"], ts), frame_id=frame_id)


def make_image_msg(types: dict, image_path: Path, ts: float):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image.shape[:2]
    return types["sensor_msgs/msg/Image"](
        header=make_header(types, ts, "camera_gray_left"),
        height=height,
        width=width,
        encoding="mono8",
        is_bigendian=0,
        step=width,
        data=np.ascontiguousarray(image.reshape(-1), dtype=np.uint8),
    )


def make_cloud_msg(types: dict, cloud_path: Path, ts: float):
    raw = np.fromfile(cloud_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"Unexpected KITTI Velodyne binary shape: {cloud_path}")
    pts = raw.reshape(-1, 4)
    rings = infer_hdl64_ring(pts[:, :3])
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("time", "<f4"),
            ("ring", "<u2"),
            ("_pad", "u1", (2,)),
        ]
    )
    cloud = np.zeros(len(pts), dtype=dtype)
    cloud["x"] = pts[:, 0]
    cloud["y"] = pts[:, 1]
    cloud["z"] = pts[:, 2]
    cloud["intensity"] = pts[:, 3]
    cloud["ring"] = rings

    point_field = types["sensor_msgs/msg/PointField"]
    fields = [
        point_field(name="x", offset=0, datatype=point_field.FLOAT32, count=1),
        point_field(name="y", offset=4, datatype=point_field.FLOAT32, count=1),
        point_field(name="z", offset=8, datatype=point_field.FLOAT32, count=1),
        point_field(name="intensity", offset=12, datatype=point_field.FLOAT32, count=1),
        point_field(name="time", offset=16, datatype=point_field.FLOAT32, count=1),
        point_field(name="ring", offset=20, datatype=point_field.UINT16, count=1),
    ]
    return types["sensor_msgs/msg/PointCloud2"](
        header=make_header(types, ts, "velodyne"),
        height=1,
        width=len(pts),
        fields=fields,
        is_bigendian=False,
        point_step=24,
        row_step=24 * len(pts),
        data=np.frombuffer(cloud.tobytes(), dtype=np.uint8).copy(),
        is_dense=True,
    )


def make_imu_msg(types: dict, oxts_path: Path, ts: float):
    vals = [float(x) for x in oxts_path.read_text(encoding="utf-8").split()]
    if len(vals) < 23:
        raise ValueError(f"Unexpected OXTS packet: {oxts_path}")
    roll, pitch, yaw = vals[3], vals[4], vals[5]
    ax, ay, az = vals[11], vals[12], vals[13]
    wx, wy, wz = vals[17], vals[18], vals[19]
    qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)
    return types["sensor_msgs/msg/Imu"](
        header=make_header(types, ts, "imu"),
        orientation=types["geometry_msgs/msg/Quaternion"](x=qx, y=qy, z=qz, w=qw),
        orientation_covariance=np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01], dtype=np.float64),
        angular_velocity=types["geometry_msgs/msg/Vector3"](x=wx, y=wy, z=wz),
        angular_velocity_covariance=np.array([0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01], dtype=np.float64),
        linear_acceleration=types["geometry_msgs/msg/Vector3"](x=ax, y=ay, z=az),
        linear_acceleration_covariance=np.array([0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1], dtype=np.float64),
    )


def yaml_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{x:.12g}" for x in values) + "]"


def write_fastlivo2_configs(fast_input: Path, out_dir: Path, image_topic: str, lidar_topic: str, imu_topic: str) -> tuple[Path, Path]:
    cam_calib = read_floats_calib(fast_input / "calib_cam_to_cam.txt")
    velo_calib = read_floats_calib(fast_input / "calib_velo_to_cam.txt")
    imu_calib = read_floats_calib(fast_input / "calib_imu_to_velo.txt")

    P0 = cam_calib["P_rect_00"]
    Rcl = velo_calib["R"]
    Pcl = velo_calib["T"]
    Ril = imu_calib["R"]
    Pil = imu_calib["T"]

    sample_image = next((fast_input / "image_00").glob("*.png"))
    image = cv2.imread(str(sample_image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sample_image)
    height, width = image.shape[:2]

    camera_config = out_dir / "camera_KITTI_RAW.yaml"
    camera_config.write_text(
        f"""/**:
  ros__parameters:
    camera:
      model: Pinhole
      width: {width}
      height: {height}
      scale: 1.0
      fx: {P0[0]:.12g}
      fy: {P0[5]:.12g}
      cx: {P0[2]:.12g}
      cy: {P0[6]:.12g}
      d0: 0.0
      d1: 0.0
      d2: 0.0
      d3: 0.0
""",
        encoding="utf-8",
    )

    fast_config = out_dir / "KITTI_RAW.yaml"
    fast_config.write_text(
        f"""/**:
  ros__parameters:
    common:
      img_topic: "{image_topic}"
      lid_topic: "{lidar_topic}"
      imu_topic: "{imu_topic}"
      img_en: 1
      lidar_en: 1
      ros_driver_bug_fix: false

    extrin_calib:
      extrinsic_T: {yaml_list(Pil)}
      extrinsic_R: {yaml_list(Ril)}
      Rcl: {yaml_list(Rcl)}
      Pcl: {yaml_list(Pcl)}

    time_offset:
      imu_time_offset: 0.0
      img_time_offset: 0.0
      exposure_time_init: 0.0

    preprocess:
      point_filter_num: 1
      filter_size_surf: 0.1
      lidar_type: 2
      scan_line: 64
      scan_rate: 10
      blind: 1.0

    vio:
      max_iterations: 5
      outlier_threshold: 1000
      img_point_cov: 1000
      patch_size: 8
      patch_pyrimid_level: 4
      normal_en: true
      raycast_en: false
      inverse_composition_en: false
      exposure_estimate_en: true
      inv_expo_cov: 0.1

    imu:
      imu_en: true
      imu_int_frame: 30
      gravity_est_en: true
      ba_bg_est_en: true
      acc_cov: 2.0
      gyr_cov: 0.1
      b_acc_cov: 0.0001
      b_gyr_cov: 0.0001

    lio:
      max_iterations: 5
      dept_err: 0.02
      beam_err: 0.05
      min_eigen_value: 0.005
      voxel_size: 2.0
      max_layer: 2
      max_points_num: 50
      layer_init_num: [5, 5, 5, 5, 5]

    local_map:
      map_sliding_en: false
      half_map_size: 100
      sliding_thresh: 8.0

    uav:
      imu_rate_odom: false
      gravity_align_en: false

    publish:
      dense_map_en: true
      pub_effect_point_en: false
      pub_plane_en: false
      pub_scan_num: 1
      blind_rgb_points: 0.0

    evo:
      seq_name: "kitti_raw_2011_09_30_0020"
      pose_output_en: true

    pcd_save:
      pcd_save_en: false
      colmap_output_en: false
      filter_size_pcd: 0.15
      interval: -1
""",
        encoding="utf-8",
    )
    return fast_config, camera_config


def write_run_script(out_dir: Path, bag_dir: Path, fast_config: Path, camera_config: Path) -> Path:
    script = out_dir / "run_fastlivo2_kitti_raw.sh"
    script.write_text(
        f"""#!/usr/bin/env bash
set -eo pipefail

WS="/home/nyu/Codespace/CVPR/third_party_research/fastlivo2_ros2_ws"
REPO="/home/nyu/Codespace/CVPR/third_party_research/FASTLIVO2_ROS2"
OUTDIR="{out_dir}"
POSE_SRC="$REPO/Log/result/kitti_raw_2011_09_30_0020.txt"
POSE_DST="$OUTDIR/fastlivo2_tum.txt"

mkdir -p "$REPO/Log/result" "$OUTDIR"
rm -f "$POSE_SRC" "$POSE_DST" "$OUTDIR/run_launch.log" "$OUTDIR/run_bagplay.log"

source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

ros2 launch fast_livo mapping_ntu_viral.launch.py use_rviz:=False use_respawn:=False \\
  ntu_params_file:="{fast_config}" \\
  camera_params_file:="{camera_config}" \\
  > "$OUTDIR/run_launch.log" 2>&1 &
LAUNCH_PID=$!

cleanup() {{
  kill "$LAUNCH_PID" >/dev/null 2>&1 || true
  wait "$LAUNCH_PID" >/dev/null 2>&1 || true
}}
trap cleanup EXIT

sleep 5
ros2 bag play "{bag_dir}" --rate 1.0 > "$OUTDIR/run_bagplay.log" 2>&1
sleep 5

if [ ! -f "$POSE_SRC" ]; then
  echo "FAST-LIVO2 pose file not found: $POSE_SRC" >&2
  exit 1
fi

cp "$POSE_SRC" "$POSE_DST"
wc -l "$POSE_DST"
echo "Saved trajectory to $POSE_DST"
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def patch_ros2_metadata_for_humble(bag_dir: Path) -> None:
    metadata = bag_dir / "metadata.yaml"
    if not metadata.exists():
        return
    text = metadata.read_text(encoding="utf-8")
    # ROS2 Humble's yaml-cpp expects offered_qos_profiles to be a string in
    # rosbag2 metadata. rosbags may emit an empty list, which ros2 bag play
    # rejects with "bad conversion".
    text = text.replace("offered_qos_profiles: []", 'offered_qos_profiles: ""')
    metadata.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((workspace / "summary.json").read_text(encoding="utf-8"))
    fast_input = Path(summary["fastlivo2_sequence_dir"])
    bag_dir = output_dir / args.bag_name
    if bag_dir.exists():
        shutil.rmtree(bag_dir)

    image_ts = read_timestamps(fast_input / "image_00_timestamps.txt")
    lidar_ts = read_timestamps(fast_input / "velodyne_timestamps.txt")
    imu_ts = read_timestamps(fast_input / "oxts_timestamps.txt")
    count = min(len(image_ts), len(lidar_ts), len(imu_ts))
    if args.max_frames > 0:
        count = min(count, args.max_frames)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    types = typestore.types
    with Writer(bag_dir, version=9) as writer:
        image_conn = writer.add_connection(args.image_topic, "sensor_msgs/msg/Image", typestore=typestore)
        lidar_conn = writer.add_connection(args.lidar_topic, "sensor_msgs/msg/PointCloud2", typestore=typestore)
        imu_conn = writer.add_connection(args.imu_topic, "sensor_msgs/msg/Imu", typestore=typestore)

        for idx in range(count):
            stem = f"{idx:010d}"
            image_msg = make_image_msg(types, fast_input / "image_00" / f"{stem}.png", image_ts[idx])
            cloud_msg = make_cloud_msg(types, fast_input / "velodyne_points" / f"{stem}.bin", lidar_ts[idx])
            imu_msg = make_imu_msg(types, fast_input / "oxts_data" / f"{stem}.txt", imu_ts[idx])
            writer.write(image_conn, int(image_ts[idx] * 1_000_000_000), typestore.serialize_cdr(image_msg, "sensor_msgs/msg/Image"))
            writer.write(lidar_conn, int(lidar_ts[idx] * 1_000_000_000), typestore.serialize_cdr(cloud_msg, "sensor_msgs/msg/PointCloud2"))
            writer.write(imu_conn, int(imu_ts[idx] * 1_000_000_000), typestore.serialize_cdr(imu_msg, "sensor_msgs/msg/Imu"))
            if idx % 100 == 0:
                print(f"[bag] wrote frame {idx}/{count}")
    patch_ros2_metadata_for_humble(bag_dir)

    fast_config, camera_config = write_fastlivo2_configs(fast_input, output_dir, args.image_topic, args.lidar_topic, args.imu_topic)
    run_script = write_run_script(output_dir, bag_dir, fast_config, camera_config)

    export_summary = {
        "workspace": str(workspace),
        "bag_dir": str(bag_dir),
        "fastlivo2_config": str(fast_config),
        "camera_config": str(camera_config),
        "run_script": str(run_script),
        "frames_exported": count,
        "topics": {
            "image": args.image_topic,
            "lidar": args.lidar_topic,
            "imu": args.imu_topic,
        },
        "notes": [
            "KITTI Velodyne ring is approximated from vertical angle because raw .bin files do not store ring/time.",
            "This adapter targets FAST-LIVO2 ROS2's Velodyne PointCloud2 path with lidar_type=2 and scan_line=64.",
        ],
    }
    (output_dir / "fastlivo2_kitti_raw_export_summary.json").write_text(
        json.dumps(export_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(export_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
