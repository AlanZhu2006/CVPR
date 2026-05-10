#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import math
import struct
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rosbag
import rospy
import sensor_msgs.point_cloud2 as pc2
import yaml
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header


TOPICS = ("/aft_mapped_to_init", "/cloud_registered_body", "/origin_img")
WEB_POINT_TOPIC = "/cloud_registered_body_web"
CURRENT_SCAN_WORLD_TOPIC = "/fastlivo/current_scan_world"
GLOBAL_MAP_TOPIC = "/fastlivo/global_map"
RGB_NEUTRAL_FALLBACK = (164, 169, 178)


@dataclass
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    t_cl: np.ndarray
    t_bl: np.ndarray


@dataclass
class LatestImage:
    stamp_sec: float
    width: int
    height: int
    rgba: np.ndarray


@dataclass
class MapVoxel:
    x: float
    y: float
    z: float
    position_samples: int
    color: np.ndarray | None
    color_samples: int
    last_seen_stamp_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct FAST-LIVO2 bag playback server with web cloud topic.")
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--default-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--container-name", default="fastlivo-direct-playback")
    parser.add_argument("--image-name", default="gs_sdf_img:latest")
    parser.add_argument("--web-max-points", type=int, default=2200)
    parser.add_argument("--web-voxel-size", type=float, default=0.16)
    parser.add_argument("--web-min-range", type=float, default=1.0)
    parser.add_argument("--web-max-range", type=float, default=24.0)
    parser.add_argument("--web-max-abs-z", type=float, default=8.0)
    parser.add_argument("--image-sync-tolerance", type=float, default=0.2)
    parser.add_argument("--pose-sync-tolerance", type=float, default=0.12)
    parser.add_argument("--scan-max-points", type=int, default=3600)
    parser.add_argument("--scan-voxel-size", type=float, default=0.12)
    parser.add_argument("--map-max-points", type=int, default=90_000)
    parser.add_argument("--map-voxel-size", type=float, default=0.2)
    parser.add_argument("--map-publish-every", type=int, default=4)
    parser.add_argument("--mjpeg-max-fps", type=float, default=60.0)
    return parser.parse_args()


def ros_time_to_sec(stamp: rospy.Time) -> float:
    return float(stamp.secs) + float(stamp.nsecs) / 1_000_000_000.0


def load_calibration(config_path: Path) -> Calibration:
    raw = config_path.read_text()
    raw = raw.replace("%YAML:1.0", "")
    raw = raw.replace("!!opencv-matrix", "")
    data = yaml.safe_load(raw)
    camera = data["camera"]
    t_cl = np.array(data["extrinsic"]["T_C_L"]["data"], dtype=np.float64).reshape(4, 4)
    t_bl = np.array(data["extrinsic"]["T_B_L"]["data"], dtype=np.float64).reshape(4, 4)
    return Calibration(
        width=int(camera["width"]),
        height=int(camera["height"]),
        fx=float(camera["fx"]),
        fy=float(camera["fy"]),
        cx=float(camera["cx"]),
        cy=float(camera["cy"]),
        distortion=np.array(
            [
                float(camera.get("d0", 0.0)),
                float(camera.get("d1", 0.0)),
                float(camera.get("d2", 0.0)),
                float(camera.get("d3", 0.0)),
                float(camera.get("d4", 0.0)),
            ],
            dtype=np.float64,
        ),
        t_cl=t_cl,
        t_bl=t_bl,
    )


def decode_image(message: Image) -> LatestImage | None:
    if message.width <= 0 or message.height <= 0:
        return None

    raw = np.frombuffer(message.data, dtype=np.uint8)
    channels = int(message.step / message.width) if message.width else 0
    if channels < 3:
        return None

    frame = raw.reshape((message.height, message.width, channels))
    encoding = (message.encoding or "").lower()
    if encoding == "bgr8":
        rgba = np.empty((message.height, message.width, 4), dtype=np.uint8)
        rgba[..., 0] = frame[..., 2]
        rgba[..., 1] = frame[..., 1]
        rgba[..., 2] = frame[..., 0]
        rgba[..., 3] = 255
    elif encoding == "rgb8":
        rgba = np.empty((message.height, message.width, 4), dtype=np.uint8)
        rgba[..., 0:3] = frame[..., 0:3]
        rgba[..., 3] = 255
    elif encoding == "rgba8":
        rgba = frame[..., 0:4].copy()
    elif encoding == "bgra8":
        rgba = np.empty((message.height, message.width, 4), dtype=np.uint8)
        rgba[..., 0] = frame[..., 2]
        rgba[..., 1] = frame[..., 1]
        rgba[..., 2] = frame[..., 0]
        rgba[..., 3] = frame[..., 3]
    else:
        return None

    return LatestImage(
        stamp_sec=ros_time_to_sec(message.header.stamp),
        width=message.width,
        height=message.height,
        rgba=rgba,
    )


def intensity_color(intensity: float) -> tuple[int, int, int]:
    value = max(0.0, min(float(intensity), 1.0))
    if value < 0.33:
        t = value / 0.33
        return (0, int(120 + 135 * t), int(255 - 80 * t))
    if value < 0.66:
        t = (value - 0.33) / 0.33
        return (int(255 * t), 255, int(175 * (1.0 - t)))
    t = (value - 0.66) / 0.34
    return (255, int(255 - 135 * t), 0)


def pack_rgb(r: int, g: int, b: int) -> int:
    return ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)


def project_rgb(point_xyz: tuple[float, float, float], image: LatestImage, calibration: Calibration) -> tuple[int, int, int] | None:
    point = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    camera = calibration.t_cl @ point
    z = float(camera[2])
    if not math.isfinite(z) or z <= 0.05:
        return None

    x = float(camera[0]) / z
    y = float(camera[1]) / z
    k1, k2, p1, p2, k3 = calibration.distortion.tolist()
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = calibration.fx * x_distorted + calibration.cx
    v = calibration.fy * y_distorted + calibration.cy
    x = int(round(u))
    y = int(round(v))
    if x < 0 or x >= image.width or y < 0 or y >= image.height:
        return None
    pixel = image.rgba[y, x]
    return (int(pixel[0]), int(pixel[1]), int(pixel[2]))


def extract_scan_points(
    message: PointCloud2,
    latest_image: LatestImage | None,
    calibration: Calibration,
    max_points: int,
    voxel_size: float,
    min_range: float,
    max_range: float,
    max_abs_z: float,
    sync_tolerance: float,
    neutral_when_image_unavailable: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    source_count = max(int(message.width) * max(int(message.height), 1), 1)
    voxel_size = max(float(voxel_size), 1e-3)
    stride = max(1, int(math.ceil(source_count / max(max_points * 4, 1))))
    image_ok = (
        latest_image is not None
        and abs(latest_image.stamp_sec - ros_time_to_sec(message.header.stamp)) <= sync_tolerance
    )
    min_range_sq = max(0.0, min_range) ** 2
    max_range_sq = max(min_range, max_range) ** 2
    voxel_seen: set[tuple[int, int, int]] = set()

    selected_fields = ("x", "y", "z", "intensity")
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[int, int, int]] = []
    for index, point in enumerate(pc2.read_points(message, field_names=selected_fields, skip_nans=True)):
        if index % stride != 0:
            continue

        x = float(point[0])
        y = float(point[1])
        z = float(point[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue

        range_sq = x * x + y * y + z * z
        if range_sq < min_range_sq or range_sq > max_range_sq:
            continue
        if abs(z) > max_abs_z:
            continue

        voxel_key = (
            int(round(x / voxel_size)),
            int(round(y / voxel_size)),
            int(round(z / voxel_size)),
        )
        if voxel_key in voxel_seen:
            continue
        voxel_seen.add(voxel_key)

        intensity = float(point[3]) if len(point) > 3 else 0.5

        if image_ok:
            rgb = project_rgb((x, y, z), latest_image, calibration)
            if rgb is None:
                rgb = RGB_NEUTRAL_FALLBACK
        else:
            if neutral_when_image_unavailable:
                rgb = RGB_NEUTRAL_FALLBACK
            else:
                rgb = intensity_color(intensity if math.isfinite(intensity) else 0.5)

        positions.append((x, y, z))
        colors.append(rgb)
        if len(positions) >= max_points:
            break

    if not positions:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
    )


def create_rgb_cloud(header: Header, positions: np.ndarray, colors: np.ndarray) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    points = [
        (
            float(position[0]),
            float(position[1]),
            float(position[2]),
            pack_rgb(int(color[0]), int(color[1]), int(color[2])),
        )
        for position, color in zip(positions, colors)
    ]
    web_cloud = pc2.create_cloud(header, fields, points)
    web_cloud.is_dense = False
    return web_cloud


def create_web_cloud(
    message: PointCloud2,
    latest_image: LatestImage | None,
    calibration: Calibration,
    max_points: int,
    voxel_size: float,
    min_range: float,
    max_range: float,
    max_abs_z: float,
    sync_tolerance: float,
) -> PointCloud2:
    positions, colors = extract_scan_points(
        message,
        latest_image,
        calibration,
        max_points=max_points,
        voxel_size=voxel_size,
        min_range=min_range,
        max_range=max_range,
        max_abs_z=max_abs_z,
        sync_tolerance=sync_tolerance,
    )
    header = Header(frame_id=message.header.frame_id, stamp=message.header.stamp)
    return create_rgb_cloud(header, positions, colors)


def odom_to_matrix(message: Odometry) -> np.ndarray:
    orientation = message.pose.pose.orientation
    position = message.pose.pose.position
    x = float(orientation.x)
    y = float(orientation.y)
    z = float(orientation.z)
    w = float(orientation.w)
    tx = float(position.x)
    ty = float(position.y)
    tz = float(position.z)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(matrix: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if positions.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    homogeneous = np.concatenate(
        [positions.astype(np.float64, copy=False), np.ones((positions.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].astype(np.float32, copy=False)


def trim_map_voxels(voxels: dict[tuple[int, int, int], MapVoxel], max_points: int) -> None:
    overflow = len(voxels) - max_points
    if overflow <= 0:
        return

    victims = sorted(
        voxels.items(),
        key=lambda item: (item[1].position_samples, item[1].last_seen_stamp_sec),
    )[:overflow]
    for key, _ in victims:
        voxels.pop(key, None)


def encode_jpeg(image: LatestImage, quality: int = 82) -> bytes | None:
    rgba = image.rgba
    if rgba.size == 0:
        return None

    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(40, min(int(quality), 95))],
    )
    if not ok:
        return None
    return encoded.tobytes()


class DirectPlaybackController:
    def __init__(self, bag_path: Path, calibration: Calibration, args: argparse.Namespace):
        self.bag_path = bag_path
        self.calibration = calibration
        self.args = args
        self.bag = rosbag.Bag(str(bag_path))
        info = self.bag.get_type_and_topic_info()[1]
        self.topics = {
            topic: {"type": info[topic].msg_type, "count": int(info[topic].message_count)}
            for topic in TOPICS
            if topic in info
        }
        if "/cloud_registered_body" in self.topics:
            self.topics[WEB_POINT_TOPIC] = {
                "type": "sensor_msgs/PointCloud2",
                "count": int(self.topics["/cloud_registered_body"]["count"]),
            }
            self.topics[CURRENT_SCAN_WORLD_TOPIC] = {
                "type": "sensor_msgs/PointCloud2",
                "count": int(self.topics["/cloud_registered_body"]["count"]),
            }
            self.topics[GLOBAL_MAP_TOPIC] = {
                "type": "sensor_msgs/PointCloud2",
                "count": None,
            }
        self.start_sec = float(self.bag.get_start_time())
        self.end_sec = float(self.bag.get_end_time())
        self.duration_sec = max(0.0, self.end_sec - self.start_sec)

        self.odom_pub = rospy.Publisher("/aft_mapped_to_init", Odometry, queue_size=8)
        self.cloud_pub = rospy.Publisher("/cloud_registered_body", PointCloud2, queue_size=2)
        self.web_cloud_pub = rospy.Publisher(WEB_POINT_TOPIC, PointCloud2, queue_size=2)
        self.current_scan_world_pub = rospy.Publisher(
            CURRENT_SCAN_WORLD_TOPIC, PointCloud2, queue_size=2
        )
        self.global_map_pub = rospy.Publisher(
            GLOBAL_MAP_TOPIC, PointCloud2, queue_size=1, latch=True
        )
        self.image_pub = rospy.Publisher("/origin_img", Image, queue_size=2)
        self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=8)

        self.lock = threading.RLock()
        self.rate = float(args.default_rate)
        self.paused = bool(args.paused)
        self.loop = bool(args.loop)
        self.container_name = str(args.container_name)
        self.image_name = str(args.image_name)
        self.requested_offset_sec = max(0.0, min(float(args.start_offset), self.duration_sec))
        self.current_offset_sec = self.requested_offset_sec
        self.playback_alive = True
        self.control_version = 0
        self.session_start_wall = time.monotonic()
        self.session_start_offset = self.requested_offset_sec
        self.latest_image: LatestImage | None = None
        self.latest_image_jpeg: bytes | None = None
        self.latest_image_stamp_sec: float | None = None
        self.latest_pose_matrix: np.ndarray | None = None
        self.latest_pose_stamp_sec: float | None = None
        self.pose_history: collections.deque[tuple[float, np.ndarray]] = collections.deque(maxlen=256)
        self.global_map_voxels: dict[tuple[int, int, int], MapVoxel] = {}
        self.global_map_point_count = 0
        self.current_scan_point_count = 0
        self.map_publish_tick = 0
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def estimate_current_offset_sec(self) -> float:
        with self.lock:
            offset = self.current_offset_sec
            duration_sec = self.duration_sec
            if self.paused or not self.playback_alive:
                return max(0.0, min(offset, duration_sec))

            elapsed = max(0.0, time.monotonic() - self.session_start_wall)
            wall_estimated = self.session_start_offset + elapsed * max(self.rate, 0.1)
            if self.loop and duration_sec > 0:
                wall_estimated %= duration_sec
            else:
                wall_estimated = min(wall_estimated, duration_sec)
            return min(wall_estimated, offset)

    def status(self) -> dict[str, Any]:
        with self.lock:
            rate = self.rate
            paused = self.paused
            loop = self.loop
            requested_offset_sec = self.requested_offset_sec
            duration_sec = self.duration_sec
            bag_start_sec = self.start_sec
            bag_end_sec = self.end_sec
            playback_alive = self.playback_alive
            topics = dict(self.topics)
            bag_path = str(self.bag_path)
            container_name = self.container_name
            image_name = self.image_name
            latest_image_stamp_sec = self.latest_image_stamp_sec
            latest_pose_stamp_sec = self.latest_pose_stamp_sec
            global_map_point_count = self.global_map_point_count
            current_scan_point_count = self.current_scan_point_count
            current_offset_sec = self.current_offset_sec
            estimated_offset_sec = self.estimate_current_offset_sec()

        return {
            "bagPath": bag_path,
            "containerName": container_name,
            "image": image_name,
            "rate": rate,
            "paused": paused,
            "loop": loop,
            "requestedOffsetSec": requested_offset_sec,
            "currentOffsetSec": current_offset_sec,
            "estimatedOffsetSec": estimated_offset_sec,
            "durationSec": duration_sec,
            "bagStartSec": bag_start_sec,
            "bagEndSec": bag_end_sec,
            "playbackAlive": playback_alive,
            "latestImageStampSec": latest_image_stamp_sec,
            "latestPoseStampSec": latest_pose_stamp_sec,
            "globalMapPoints": global_map_point_count,
            "currentScanPoints": current_scan_point_count,
            "topics": topics,
            "mode": "direct-bag",
        }

    def apply_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if "rate" in payload and payload["rate"] is not None:
                self.rate = max(0.1, float(payload["rate"]))
            if "paused" in payload and payload["paused"] is not None:
                self.paused = bool(payload["paused"])
            if "offsetSec" in payload and payload["offsetSec"] is not None:
                self.requested_offset_sec = max(0.0, min(float(payload["offsetSec"]), self.duration_sec))
                self.current_offset_sec = self.requested_offset_sec
                self.reset_mapping_state()
            else:
                self.requested_offset_sec = self.current_offset_sec
            self.control_version += 1
            self.session_start_offset = self.requested_offset_sec
            self.session_start_wall = time.monotonic()
            return self.status()

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)
        self.bag.close()

    def get_latest_jpeg_frame(self) -> tuple[float | None, bytes | None]:
        with self.lock:
            return self.latest_image_stamp_sec, self.latest_image_jpeg

    def reset_mapping_state(self) -> None:
        self.latest_pose_matrix = None
        self.latest_pose_stamp_sec = None
        self.pose_history = collections.deque(maxlen=256)
        self.global_map_voxels = {}
        self.global_map_point_count = 0
        self.current_scan_point_count = 0
        self.map_publish_tick = 0

    def get_pose_matrix_for_stamp(self, stamp_sec: float) -> tuple[np.ndarray | None, float | None]:
        with self.lock:
            if not self.pose_history:
                return self.latest_pose_matrix, self.latest_pose_stamp_sec

            tolerance = float(self.args.pose_sync_tolerance)
            best_matrix: np.ndarray | None = None
            best_stamp: float | None = None
            best_distance = float("inf")

            for pose_stamp_sec, pose_matrix in self.pose_history:
                distance = abs(pose_stamp_sec - stamp_sec)
                if distance < best_distance:
                    best_distance = distance
                    best_stamp = pose_stamp_sec
                    best_matrix = pose_matrix

            if best_matrix is not None and best_distance <= tolerance:
                return best_matrix, best_stamp

            return self.latest_pose_matrix, self.latest_pose_stamp_sec

    def update_global_map(
        self,
        world_positions: np.ndarray,
        colors: np.ndarray,
        stamp_sec: float,
    ) -> None:
        voxel_size = max(float(self.args.map_voxel_size), 1e-3)
        with self.lock:
            voxels = self.global_map_voxels
            for index in range(world_positions.shape[0]):
                x = float(world_positions[index, 0])
                y = float(world_positions[index, 1])
                z = float(world_positions[index, 2])
                key = (
                    int(round(x / voxel_size)),
                    int(round(y / voxel_size)),
                    int(round(z / voxel_size)),
                )
                existing = voxels.get(key)
                if existing is None:
                    voxels[key] = MapVoxel(
                        x=x,
                        y=y,
                        z=z,
                        position_samples=1,
                        color=colors[index].astype(np.float64),
                        color_samples=1,
                        last_seen_stamp_sec=stamp_sec,
                    )
                    continue

                existing.position_samples += 1
                mix = 1.0 / float(existing.position_samples)
                existing.x += (x - existing.x) * mix
                existing.y += (y - existing.y) * mix
                existing.z += (z - existing.z) * mix
                existing.last_seen_stamp_sec = stamp_sec
                if existing.color is None:
                    existing.color = colors[index].astype(np.float64)
                    existing.color_samples = 1
                else:
                    existing.color_samples += 1
                    color_mix = 1.0 / float(existing.color_samples)
                    existing.color += (colors[index].astype(np.float64) - existing.color) * color_mix

            trim_map_voxels(voxels, int(self.args.map_max_points))
            self.global_map_point_count = len(voxels)
            self.map_publish_tick += 1

    def publish_global_map(self, stamp: rospy.Time) -> None:
        if self.global_map_pub.get_num_connections() <= 0 and self.global_map_point_count == 0:
            return

        with self.lock:
            voxels = list(self.global_map_voxels.values())

        if not voxels:
            return

        positions = np.empty((len(voxels), 3), dtype=np.float32)
        colors = np.empty((len(voxels), 3), dtype=np.uint8)
        for index, voxel in enumerate(voxels):
            positions[index, 0] = voxel.x
            positions[index, 1] = voxel.y
            positions[index, 2] = voxel.z
            if voxel.color is not None:
                colors[index] = np.clip(np.round(voxel.color), 0, 255).astype(np.uint8)
            else:
                colors[index] = np.array([216, 236, 255], dtype=np.uint8)

        header = Header(frame_id="world", stamp=stamp)
        self.global_map_pub.publish(create_rgb_cloud(header, positions, colors))

    def run(self) -> None:
        current_version = -1
        session_start_wall = 0.0
        session_start_offset = 0.0
        iterator = None

        while not self.stop_event.is_set() and not rospy.is_shutdown():
            with self.lock:
                paused = self.paused
                version = self.control_version
                requested_offset = self.requested_offset_sec
                rate = self.rate
                loop = self.loop

            if paused:
                time.sleep(0.05)
                continue

            if iterator is None or version != current_version:
                current_version = version
                session_start_offset = requested_offset
                session_start_wall = time.monotonic()
                with self.lock:
                    self.session_start_offset = session_start_offset
                    self.session_start_wall = session_start_wall
                    self.playback_alive = True
                iterator = self.bag.read_messages(
                    topics=list(TOPICS),
                    start_time=rospy.Time.from_sec(self.start_sec + session_start_offset),
                )

            try:
                topic, msg, stamp = next(iterator)
            except StopIteration:
                iterator = None
                with self.lock:
                    if loop:
                        self.reset_mapping_state()
                        self.requested_offset_sec = 0.0
                        self.current_offset_sec = 0.0
                        self.session_start_offset = 0.0
                        self.session_start_wall = time.monotonic()
                        self.control_version += 1
                    else:
                        self.paused = True
                        self.playback_alive = False
                continue

            offset_sec = max(0.0, ros_time_to_sec(stamp) - self.start_sec)
            target_wall = session_start_wall + max(0.0, (offset_sec - session_start_offset) / max(rate, 0.1))
            while time.monotonic() < target_wall:
                if self.stop_event.wait(0.002):
                    return
                with self.lock:
                    if self.control_version != current_version or self.paused:
                        iterator = None
                        break
            if iterator is None:
                continue

            self.clock_pub.publish(Clock(clock=stamp))
            if topic == "/origin_img":
                self.image_pub.publish(msg)
                latest_image = decode_image(msg)
                latest_jpeg = encode_jpeg(latest_image) if latest_image is not None else None
                with self.lock:
                    self.latest_image = latest_image
                    self.latest_image_jpeg = latest_jpeg
                    self.latest_image_stamp_sec = latest_image.stamp_sec if latest_image else None
            elif topic == "/aft_mapped_to_init":
                self.odom_pub.publish(msg)
                pose_stamp_sec = ros_time_to_sec(msg.header.stamp)
                pose_matrix = odom_to_matrix(msg)
                with self.lock:
                    self.latest_pose_matrix = pose_matrix
                    self.latest_pose_stamp_sec = pose_stamp_sec
                    self.pose_history.append((pose_stamp_sec, pose_matrix))
            elif topic == "/cloud_registered_body":
                self.cloud_pub.publish(msg)
                with self.lock:
                    latest_image = self.latest_image

                scan_positions_lidar, scan_colors = extract_scan_points(
                    msg,
                    latest_image,
                    self.calibration,
                    max_points=max(
                        int(self.args.scan_max_points),
                        int(self.args.web_max_points),
                    ),
                    voxel_size=min(
                        max(float(self.args.scan_voxel_size), 1e-3),
                        max(float(self.args.web_voxel_size), 1e-3),
                    ),
                    min_range=self.args.web_min_range,
                    max_range=self.args.web_max_range,
                    max_abs_z=self.args.web_max_abs_z,
                    sync_tolerance=self.args.image_sync_tolerance,
                )

                web_count = min(scan_positions_lidar.shape[0], int(self.args.web_max_points))
                web_positions = scan_positions_lidar[:web_count]
                web_colors = scan_colors[:web_count]
                web_header = Header(frame_id=msg.header.frame_id, stamp=msg.header.stamp)
                web_cloud = create_rgb_cloud(web_header, web_positions, web_colors)
                self.web_cloud_pub.publish(web_cloud)

                cloud_stamp_sec = ros_time_to_sec(msg.header.stamp)
                latest_pose_matrix, latest_pose_stamp_sec = self.get_pose_matrix_for_stamp(cloud_stamp_sec)
                pose_is_synced = (
                    latest_pose_matrix is not None
                    and latest_pose_stamp_sec is not None
                )
                if pose_is_synced:
                    scan_count = min(scan_positions_lidar.shape[0], int(self.args.scan_max_points))
                    scan_positions_world = transform_points(
                        latest_pose_matrix @ self.calibration.t_bl,
                        scan_positions_lidar[:scan_count],
                    )
                    scan_colors_world = scan_colors[:scan_count]
                    current_scan_header = Header(frame_id="world", stamp=msg.header.stamp)
                    self.current_scan_world_pub.publish(
                        create_rgb_cloud(current_scan_header, scan_positions_world, scan_colors_world)
                    )
                    with self.lock:
                        self.current_scan_point_count = scan_positions_world.shape[0]

                    self.update_global_map(scan_positions_world, scan_colors_world, cloud_stamp_sec)
                    if (
                        self.map_publish_tick <= 1
                        or self.map_publish_tick % max(int(self.args.map_publish_every), 1) == 0
                    ):
                        self.publish_global_map(msg.header.stamp)
                else:
                    with self.lock:
                        self.current_scan_point_count = 0

            with self.lock:
                self.current_offset_sec = offset_sec
                self.requested_offset_sec = offset_sec
                self.playback_alive = True


class Handler(BaseHTTPRequestHandler):
    controller: DirectPlaybackController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/status":
            self.respond_json(self.controller.status())
            return
        if self.path == "/topics":
            self.respond_json({"topics": self.controller.topics})
            return
        if self.path.startswith("/frame.jpg"):
            self.respond_jpeg_frame()
            return
        if self.path.startswith("/frame.mjpeg"):
            self.respond_mjpeg_stream()
            return
        self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/control":
            self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        self.respond_json(self.controller.apply_control(payload))

    def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_jpeg_frame(self) -> None:
        stamp_sec, frame = self.controller.get_latest_jpeg_frame()
        if frame is None:
            self.respond_json({"error": "No image frame available yet."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return

        self.send_response(HTTPStatus.OK)
        self.send_common_headers()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Frame-Stamp", str(stamp_sec or 0.0))
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def respond_mjpeg_stream(self) -> None:
        boundary = "frame"
        self.send_response(HTTPStatus.OK)
        self.send_common_headers()
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        last_stamp_sec: float | None = None
        last_frame: bytes | None = None

        try:
            while True:
                stamp_sec, frame = self.controller.get_latest_jpeg_frame()
                if frame is None:
                    if self.controller.stop_event.wait(0.05):
                        return
                    continue

                if last_frame is None or stamp_sec != last_stamp_sec:
                    last_stamp_sec = stamp_sec
                    last_frame = frame

                self.wfile.write(f"--{boundary}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(last_frame)}\r\n".encode("ascii"))
                self.wfile.write(f"X-Frame-Stamp: {last_stamp_sec or 0.0}\r\n\r\n".encode("ascii"))
                self.wfile.write(last_frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()

                mjpeg_max_fps = max(0.0, float(getattr(self.controller.args, "mjpeg_max_fps", 60.0)))
                if mjpeg_max_fps > 0.0:
                    if self.controller.stop_event.wait(1.0 / mjpeg_max_fps):
                        return
                elif self.controller.stop_event.wait(0.005):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    calibration = load_calibration(Path(args.config).resolve())
    rospy.init_node("fastlivo_direct_playback", anonymous=False, disable_signals=True)
    controller = DirectPlaybackController(Path(args.bag_path).resolve(), calibration, args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        rospy.loginfo("direct playback ready on http://0.0.0.0:%s", args.port)
        server.serve_forever()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
