#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import (
    CUVSLAMMonocularRGBAdapter,
    CUVSLAMOfflineKITTIAdapter,
    LingBotDepthWorker,
    LingBotDepthWorkerConfig,
    RGBMonocularVOAdapter,
    VoxelFusionMap,
    empty_point_batch,
    load_runtime_config,
)
from nuc_runtime.cuvslam_adapter import _quaternion_to_matrix
from nuc_runtime.descriptors import compute_global_descriptor
from nuc_runtime.models import TrackingOutput


def _matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to ROS xyzw quaternion order."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0 or not math.isfinite(norm):
        return 0.0, 0.0, 0.0, 1.0
    return float(qx / norm), float(qy / norm), float(qz / norm), float(qw / norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate live cuVSLAM tracking + LingBot background dense-depth mapping "
            "and continuously publish a browser-viewable local 3D reconstruction."
        )
    )
    parser.add_argument("--sequence-dir", default="nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input")
    parser.add_argument("--trajectory-path", default="nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_tum.txt")
    parser.add_argument(
        "--tracking-backend",
        choices=("pose_file", "opencv_mono_rgb", "cuvslam_mono_rgb", "hikrobot_mono_rgb"),
        default="pose_file",
        help=(
            "pose_file replays an existing TUM trajectory; opencv_mono_rgb is a small OpenCV baseline; "
            "cuvslam_mono_rgb runs PyCuVSLAM OdometryMode.Mono on the RGB stream; "
            "hikrobot_mono_rgb captures live RGB frames from a HikRobot MVS camera."
        ),
    )
    parser.add_argument(
        "--rgb-image-dir",
        default="",
        help="RGB image stream for monocular tracking and LingBot input. If empty, uses --color-image-dir for opencv_mono_rgb.",
    )
    parser.add_argument("--intrinsic-camera-index", type=int, default=0)
    parser.add_argument("--mono-fixed-step-scale", type=float, default=0.5)
    parser.add_argument("--mono-scale-source", choices=("fixed", "oxts"), default="fixed")
    parser.add_argument("--oxts-dir", default="")
    parser.add_argument("--hikrobot-index", type=int, default=0)
    parser.add_argument("--hikrobot-timeout-ms", type=int, default=2000)
    parser.add_argument("--hikrobot-exposure-us", type=float, default=None)
    parser.add_argument("--hikrobot-gain", type=float, default=None)
    parser.add_argument("--hikrobot-fps", type=float, default=15.0)
    parser.add_argument("--hikrobot-width", type=int, default=0)
    parser.add_argument("--hikrobot-height", type=int, default=0)
    parser.add_argument("--hikrobot-max-read-errors", type=int, default=30)
    parser.add_argument("--hikrobot-read-error-sleep-sec", type=float, default=0.05)
    parser.add_argument(
        "--hikrobot-threaded-capture",
        action="store_true",
        help="Read HikRobot frames and publish /hikrobot/image_raw from a background producer thread.",
    )
    parser.add_argument("--hikrobot-capture-queue-size", type=int, default=4)
    parser.add_argument("--camera-fx", type=float, default=0.0)
    parser.add_argument("--camera-fy", type=float, default=0.0)
    parser.add_argument("--camera-cx", type=float, default=0.0)
    parser.add_argument("--camera-cy", type=float, default=0.0)
    parser.add_argument(
        "--rgb-output-dir",
        default="",
        help="Directory for live HikRobot frames. Defaults to <output-dir>/rgb_stream.",
    )
    parser.add_argument(
        "--color-image-dir",
        default="",
        help="Optional RGB image directory used only for coloring fused points. Tracking/LingBot still use sequence-dir images.",
    )
    parser.add_argument(
        "--color-image-template",
        default="{frame_idx:010d}.png",
        help="Filename template under --color-image-dir. Use {frame_idx:06d}.png for cuVSLAM-style image folders.",
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--model-path", default="third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt")
    parser.add_argument("--lingbot-map-root", default="third_party_research/lingbot-map")
    parser.add_argument("--output-dir", default="nuc_output/lingbot_live_reconstruction/kitti0020_live")
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--dense-frame-interval",
        type=int,
        default=1,
        help="Submit only every Nth tracked frame to the LingBot dense worker while keeping every cuVSLAM pose.",
    )
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--max-queue", type=int, default=2)
    parser.add_argument("--depth-scale", type=float, default=20.0)
    parser.add_argument(
        "--lingbot-pose-translation-scale",
        type=float,
        default=0.0,
        help="Scale LingBot-predicted camera translations before fusing depth. 0 follows --depth-scale.",
    )
    parser.add_argument(
        "--lingbot-extrinsic-mode",
        choices=("inverse", "direct"),
        default="inverse",
        help=(
            "How to interpret LingBot predictions.extrinsic when no external pose is available. "
            "inverse matches the official depth unprojection path where extrinsic is world-to-camera."
        ),
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--sampling-pattern", choices=("grid", "jittered", "random"), default="grid")
    parser.add_argument("--max-points-per-frame", type=int, default=2500)
    parser.add_argument("--max-active-frames", type=int, default=16)
    parser.add_argument("--fusion-mode", choices=("raw", "voxel"), default="raw")
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--fusion-max-points", type=int, default=500000)
    parser.add_argument("--fusion-min-observations", type=int, default=1)
    parser.add_argument("--adaptive-sampling", action="store_true")
    parser.add_argument("--near-depth-m", type=float, default=18.0)
    parser.add_argument("--near-sample-stride", type=int, default=1)
    parser.add_argument("--edge-sample-stride", type=int, default=2)
    parser.add_argument("--edge-percentile", type=float, default=88.0)
    parser.add_argument("--semantic-sample-stride", type=int, default=1)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--semantic-color-output", action="store_true")
    parser.add_argument("--depth-head-trt-engine", default="")
    parser.add_argument(
        "--use-sdpa",
        action="store_true",
        help="Use PyTorch SDPA instead of LingBot/FlashInfer attention. Default keeps the faster LingBot demo path.",
    )
    parser.add_argument("--model-patch-embed", default="")
    parser.add_argument("--model-embed-dim", type=int, default=0)
    parser.add_argument("--model-depth", type=int, default=0)
    parser.add_argument("--model-num-heads", type=int, default=0)
    parser.add_argument("--model-mlp-ratio", type=float, default=0.0)
    parser.add_argument("--no-compress-output", action="store_true")
    parser.add_argument("--frame-sleep-sec", type=float, default=0.0)
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--publish-every-windows", type=int, default=1)
    parser.add_argument("--publish-every-frames", type=int, default=1)
    parser.add_argument("--ros2-publish", action="store_true", help="Publish the live HikRobot image and fused cloud to ROS2.")
    parser.add_argument("--ros2-image-topic", default="/hikrobot/image_raw")
    parser.add_argument("--ros2-camera-info-topic", default="/hikrobot/camera_info")
    parser.add_argument("--ros2-pose-topic", default="/neural_mapping/pose")
    parser.add_argument("--ros2-path-topic", default="/neural_mapping/path")
    parser.add_argument("--ros2-cloud-topic", default="/lingbot/cloud_rgb")
    parser.add_argument("--ros2-plain-cloud-topic", default="/lingbot/cloud_plain")
    parser.add_argument("--ros2-current-cloud-topic", default="/lingbot/current_cloud_rgb")
    parser.add_argument("--ros2-current-plain-cloud-topic", default="/lingbot/current_cloud_plain")
    parser.add_argument("--ros2-camera-frame-id", default="hikrobot_camera")
    parser.add_argument("--ros2-cloud-frame-id", default="map")
    parser.add_argument("--ros2-max-cloud-points", type=int, default=120000)
    parser.add_argument("--ros2-max-current-cloud-points", type=int, default=60000)
    parser.add_argument(
        "--ros2-republish-current-cloud-on-image",
        action="store_true",
        help="Republish the most recent current cloud after each image publish so RViz gets image-rate stamps.",
    )
    parser.add_argument(
        "--ros2-image-max-width",
        type=int,
        default=0,
        help="Resize only the ROS2 preview image to this width. 0 publishes the full camera frame.",
    )
    parser.add_argument(
        "--ros2-image-max-height",
        type=int,
        default=0,
        help="Resize only the ROS2 preview image to this height. 0 publishes the full camera frame.",
    )
    parser.add_argument(
        "--ros2-cloud-min-interval-sec",
        type=float,
        default=0.0,
        help="Minimum time between PointCloud2 publishes. Image publishing is unaffected.",
    )
    parser.add_argument(
        "--ros2-path-max-poses",
        type=int,
        default=1200,
        help="Maximum trajectory poses to include in each nav_msgs/Path publish. 0 publishes the full path.",
    )
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=19092)
    parser.add_argument("--no-blocking-submit", action="store_true")
    parser.add_argument(
        "--drop-when-busy",
        action="store_true",
        help="Alias for --no-blocking-submit: keep tracking realtime and drop dense windows if the worker queue is full.",
    )
    return parser.parse_args()


class ROS2LiveBridge:
    """Publish the single HikRobot/LingBot live stream to RViz-friendly ROS2 topics."""

    def __init__(
        self,
        *,
        image_topic: str,
        camera_info_topic: str,
        pose_topic: str,
        path_topic: str,
        cloud_topic: str,
        plain_cloud_topic: str,
        current_cloud_topic: str,
        current_plain_cloud_topic: str,
        camera_frame_id: str,
        cloud_frame_id: str,
        max_cloud_points: int,
        max_current_cloud_points: int,
        image_max_width: int,
        image_max_height: int,
        republish_current_cloud_on_image: bool,
        path_max_poses: int,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path as NavPath
        from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

        rclpy.init(args=None)
        self.rclpy = rclpy
        self.node = Node("lingbot_live_ros2_bridge")
        self.Image = Image
        self.CameraInfo = CameraInfo
        self.PoseStamped = PoseStamped
        self.NavPath = NavPath
        self.PointCloud2 = PointCloud2
        self.PointField = PointField
        self.image_pub = self.node.create_publisher(Image, image_topic, 4) if image_topic else None
        self.info_pub = self.node.create_publisher(CameraInfo, camera_info_topic, 4) if camera_info_topic else None
        self.pose_pub = self.node.create_publisher(PoseStamped, pose_topic, 10) if pose_topic else None
        self.path_pub = self.node.create_publisher(NavPath, path_topic, 4) if path_topic else None
        self.cloud_pub = self.node.create_publisher(PointCloud2, cloud_topic, 2) if cloud_topic else None
        self.plain_cloud_pub = self.node.create_publisher(PointCloud2, plain_cloud_topic, 2) if plain_cloud_topic else None
        self.current_cloud_pub = self.node.create_publisher(PointCloud2, current_cloud_topic, 2) if current_cloud_topic else None
        self.current_plain_cloud_pub = (
            self.node.create_publisher(PointCloud2, current_plain_cloud_topic, 2)
            if current_plain_cloud_topic
            else None
        )
        self.camera_frame_id = str(camera_frame_id)
        self.cloud_frame_id = str(cloud_frame_id)
        self.max_cloud_points = int(max_cloud_points)
        self.max_current_cloud_points = int(max_current_cloud_points)
        self.image_max_width = int(image_max_width)
        self.image_max_height = int(image_max_height)
        self.republish_current_cloud_on_image = bool(republish_current_cloud_on_image)
        self.path_max_poses = int(path_max_poses)
        self._last_current_xyz: np.ndarray | None = None
        self._last_current_rgb: np.ndarray | None = None
        self._lock = threading.Lock()
        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        print(
            "ROS2 live bridge: "
            f"image={image_topic or '(disabled)'} camera_info={camera_info_topic or '(disabled)'} "
            f"pose={pose_topic or '(disabled)'} path={path_topic or '(disabled)'} "
            f"cloud={cloud_topic or '(disabled)'} plain_cloud={plain_cloud_topic or '(disabled)'} "
            f"current_cloud={current_cloud_topic or '(disabled)'}",
            flush=True,
        )

    def publish_pose_path(self, pose: np.ndarray, trajectory: list[dict[str, Any]]) -> None:
        if self.pose_pub is None and self.path_pub is None:
            return
        pose = np.asarray(pose, dtype=np.float32)
        if pose.shape != (4, 4):
            return
        with self._lock:
            stamp = self.node.get_clock().now().to_msg()
            if self.pose_pub is not None:
                self.pose_pub.publish(self._pose_stamped_from_matrix(pose, stamp))
            if self.path_pub is not None:
                msg = self.NavPath()
                msg.header.stamp = stamp
                msg.header.frame_id = self.cloud_frame_id
                raw_items = trajectory[-self.path_max_poses :] if self.path_max_poses > 0 else trajectory
                poses = []
                for item in raw_items:
                    item_pose = np.asarray(item.get("pose"), dtype=np.float32)
                    if item_pose.shape == (4, 4):
                        poses.append(self._pose_stamped_from_matrix(item_pose, stamp))
                msg.poses = poses
                self.path_pub.publish(msg)
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_image(self, rgb: np.ndarray, K: np.ndarray, timestamp_sec: float | None = None) -> None:
        if self.image_pub is None:
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        K_msg = np.asarray(K, dtype=np.float32).copy()
        height0, width0 = rgb.shape[:2]
        scale = 1.0
        if self.image_max_width > 0 and width0 > self.image_max_width:
            scale = min(scale, float(self.image_max_width) / float(width0))
        if self.image_max_height > 0 and height0 > self.image_max_height:
            scale = min(scale, float(self.image_max_height) / float(height0))
        if scale < 1.0:
            width1 = max(1, int(round(width0 * scale)))
            height1 = max(1, int(round(height0 * scale)))
            rgb = cv2.resize(rgb, (width1, height1), interpolation=cv2.INTER_AREA)
            K_msg[0, :] *= scale
            K_msg[1, :] *= scale
        height, width = rgb.shape[:2]
        with self._lock:
            stamp = self.node.get_clock().now().to_msg()
            image = self.Image()
            image.header.stamp = stamp
            image.header.frame_id = self.camera_frame_id
            image.height = int(height)
            image.width = int(width)
            image.encoding = "rgb8"
            image.is_bigendian = False
            image.step = int(width) * 3
            image.data = rgb.tobytes()
            self.image_pub.publish(image)

            if self.info_pub is not None:
                info = self.CameraInfo()
                info.header = image.header
                info.height = int(height)
                info.width = int(width)
                fx = float(K_msg[0, 0])
                fy = float(K_msg[1, 1])
                cx = float(K_msg[0, 2])
                cy = float(K_msg[1, 2])
                info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.info_pub.publish(info)
            if (
                self.republish_current_cloud_on_image
                and self._last_current_xyz is not None
                and self._last_current_rgb is not None
            ):
                self._publish_cloud_locked(
                    self.current_cloud_pub,
                    self.current_plain_cloud_pub,
                    self._last_current_xyz,
                    self._last_current_rgb,
                    max_points=self.max_current_cloud_points,
                )
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_cloud(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if self.cloud_pub is None:
            return
        with self._lock:
            self._publish_cloud_locked(
                self.cloud_pub,
                self.plain_cloud_pub,
                xyz,
                rgb,
                max_points=self.max_cloud_points,
            )
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_current_cloud(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if self.current_cloud_pub is None:
            return
        self._last_current_xyz = np.asarray(xyz, dtype=np.float32).copy()
        self._last_current_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        with self._lock:
            self._publish_cloud_locked(
                self.current_cloud_pub,
                self.current_plain_cloud_pub,
                self._last_current_xyz,
                self._last_current_rgb,
                max_points=self.max_current_cloud_points,
            )
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def _publish_cloud_locked(
        self,
        cloud_pub: Any,
        plain_cloud_pub: Any,
        xyz: np.ndarray,
        rgb: np.ndarray,
        *,
        max_points: int,
    ) -> None:
        if cloud_pub is None:
            return
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header

        xyz = np.asarray(xyz, dtype=np.float32)
        rgb = np.asarray(rgb, dtype=np.uint8)
        if xyz.shape[0] != rgb.shape[0]:
            count = min(int(xyz.shape[0]), int(rgb.shape[0]))
            xyz = xyz[:count]
            rgb = rgb[:count]
        if max_points > 0 and xyz.shape[0] > max_points:
            idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
            xyz = xyz[idx]
            rgb = rgb[idx]
        cloud = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        if xyz.shape[0] > 0:
            cloud[:, :3] = xyz
            cloud[:, 3] = self._pack_rgb(rgb)
        header = Header()
        header.stamp = self.node.get_clock().now().to_msg()
        header.frame_id = self.cloud_frame_id
        msg = point_cloud2.create_cloud(header=header, fields=self.fields, points=cloud)
        cloud_pub.publish(msg)
        if plain_cloud_pub is not None:
            plain = cloud.copy()
            if plain.shape[0] > 0:
                plain_rgb = np.full((plain.shape[0], 3), 220, dtype=np.uint8)
                plain[:, 3] = self._pack_rgb(plain_rgb)
            plain_cloud_pub.publish(point_cloud2.create_cloud(header=header, fields=self.fields, points=plain))

    def _pose_stamped_from_matrix(self, pose: np.ndarray, stamp: Any) -> Any:
        msg = self.PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.cloud_frame_id
        msg.pose.position.x = float(pose[0, 3])
        msg.pose.position.y = float(pose[1, 3])
        msg.pose.position.z = float(pose[2, 3])
        qx, qy, qz, qw = _matrix_to_quaternion(pose[:3, :3])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg

    def close(self) -> None:
        with self._lock:
            self.node.destroy_node()
            try:
                self.rclpy.shutdown()
            except Exception:
                pass

    @staticmethod
    def _pack_rgb(rgb: np.ndarray) -> np.ndarray:
        packed = (
            (rgb[:, 0].astype(np.uint32) << 16)
            | (rgb[:, 1].astype(np.uint32) << 8)
            | rgb[:, 2].astype(np.uint32)
        )
        return packed.view(np.float32)


class HikRobotMonocularRGBAdapter:
    """Live HikRobot RGB source with cuVSLAM Mono poses and on-disk frames for LingBot."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        camera_index: int,
        timeout_ms: int,
        exposure_us: float | None,
        gain: float | None,
        fps: float | None,
        capture_width: int,
        capture_height: int,
        max_read_errors: int,
        read_error_sleep_sec: float,
        threaded_capture: bool,
        capture_queue_size: int,
        intrinsic_args: tuple[float, float, float, float],
        config: Any,
        fixed_step_scale: float = 0.08,
        frame_step: int = 1,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, np.ndarray, float], None] | None = None,
    ) -> None:
        from hikrobot_mvs_ros2_publisher import HikRobotCamera

        try:
            import cuvslam  # type: ignore
        except Exception:
            cuvslam = None

        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.times_path = self.output_dir / "times.txt"
        self.calib_path = self.output_dir / "calib.txt"
        self.frame_step = max(1, int(frame_step))
        self.max_frames = int(max_frames)
        self.frame_callback = frame_callback
        self.max_read_errors = int(max_read_errors)
        self.read_error_sleep_sec = float(read_error_sleep_sec)
        self.threaded_capture = bool(threaded_capture)
        self.capture_queue_size = max(1, int(capture_queue_size))
        self._capture_queue: queue.Queue[tuple[int, float, np.ndarray]] = queue.Queue(maxsize=self.capture_queue_size)
        self._capture_stop = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self.config = config
        self.fixed_step_scale = float(fixed_step_scale)
        self.orb = cv2.ORB_create(nfeatures=config.max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._prev_keypoints: list | None = None
        self._prev_descriptors: np.ndarray | None = None
        self._prev_gray: np.ndarray | None = None
        self._prev_pose: np.ndarray | None = None
        self._last_keyframe_idx = -10**9
        self._last_pose = np.eye(4, dtype=np.float32)
        self._start = time.perf_counter()
        self._odometry_mode = "OpenCV Essential"

        print(
            f"Opening HikRobot camera index={camera_index} timeout_ms={timeout_ms} "
            f"exposure_us={exposure_us} gain={gain} fps={fps}",
            flush=True,
        )
        self.camera = HikRobotCamera(camera_index, timeout_ms)
        self.camera.open(
            exposure_us,
            gain,
            fps,
            capture_width or None,
            capture_height or None,
        )
        first_rgb_bytes, width, height = self.camera.read_rgb()
        self._pending_rgb = np.frombuffer(first_rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
        self.height = int(height)
        self.width = int(width)
        fx_arg, fy_arg, cx_arg, cy_arg = intrinsic_args
        fx = float(fx_arg or width)
        fy = float(fy_arg or width)
        cx = float(cx_arg or width / 2.0)
        cy = float(cy_arg or height / 2.0)
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        self._write_calib_and_times_header()

        self.tracker = None
        if cuvslam is not None:
            camera = cuvslam.Camera()
            camera.size = [self.width, self.height]
            camera.focal = [float(self.K[0, 0]), float(self.K[1, 1])]
            camera.principal = [float(self.K[0, 2]), float(self.K[1, 2])]
            camera.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole, [])
            camera.border_top = 10
            camera.border_bottom = 10
            camera.border_left = 10
            camera.border_right = 10
            rig = cuvslam.Rig([camera])
            cfg = cuvslam.Tracker.OdometryConfig()
            if hasattr(cuvslam.Tracker.OdometryMode, "Mono"):
                cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
                self.tracker = cuvslam.Tracker(rig, cfg)
                self._odometry_mode = "cuVSLAM Mono"
        print(
            f"HikRobot stream opened: {self.width}x{self.height}, "
            f"fx={self.K[0, 0]:.2f}, fy={self.K[1, 1]:.2f}, odometry={self._odometry_mode}",
            flush=True,
        )

    def close(self) -> None:
        self._capture_stop.set()
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self.camera.close()

    def __iter__(self):
        if self.threaded_capture:
            self._start_capture_thread()
            yield from self._iter_threaded()
            return
        frame_idx = 0
        consecutive_read_errors = 0
        try:
            while self.max_frames <= 0 or frame_idx < self.max_frames:
                if frame_idx == 0:
                    rgb = self._pending_rgb
                else:
                    try:
                        rgb_bytes, width, height = self.camera.read_rgb()
                    except RuntimeError as exc:
                        consecutive_read_errors += 1
                        print(
                            f"Warning: HikRobot read failed ({consecutive_read_errors}/"
                            f"{self.max_read_errors}): {exc}",
                            flush=True,
                        )
                        if self.max_read_errors > 0 and consecutive_read_errors >= self.max_read_errors:
                            raise
                        time.sleep(max(0.0, self.read_error_sleep_sec))
                        continue
                    consecutive_read_errors = 0
                    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
                timestamp_sec = time.perf_counter() - self._start
                if self.frame_callback is not None:
                    self.frame_callback(rgb, self.K, timestamp_sec)
                self._append_time(timestamp_sec)
                if frame_idx % self.frame_step == 0:
                    yield self._build_output(frame_idx, timestamp_sec, rgb)
                frame_idx += 1
        finally:
            self.close()

    def _start_capture_thread(self) -> None:
        if self._capture_thread is not None:
            return
        self._capture_thread = threading.Thread(target=self._capture_loop, name="hikrobot-capture", daemon=True)
        self._capture_thread.start()
        print(
            f"HikRobot threaded capture enabled: queue_size={self.capture_queue_size}",
            flush=True,
        )

    def _capture_loop(self) -> None:
        frame_idx = 0
        consecutive_read_errors = 0
        while not self._capture_stop.is_set():
            try:
                if frame_idx == 0:
                    rgb = self._pending_rgb
                else:
                    rgb_bytes, width, height = self.camera.read_rgb()
                    rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
            except RuntimeError as exc:
                consecutive_read_errors += 1
                print(
                    f"Warning: HikRobot capture read failed ({consecutive_read_errors}/"
                    f"{self.max_read_errors}): {exc}",
                    flush=True,
                )
                if self.max_read_errors > 0 and consecutive_read_errors >= self.max_read_errors:
                    self._capture_stop.set()
                    break
                time.sleep(max(0.0, self.read_error_sleep_sec))
                continue
            consecutive_read_errors = 0
            timestamp_sec = time.perf_counter() - self._start
            if self.frame_callback is not None:
                self.frame_callback(rgb, self.K, timestamp_sec)
            self._put_capture_frame(frame_idx, timestamp_sec, rgb)
            frame_idx += 1

    def _put_capture_frame(self, frame_idx: int, timestamp_sec: float, rgb: np.ndarray) -> None:
        item = (int(frame_idx), float(timestamp_sec), rgb)
        while not self._capture_stop.is_set():
            try:
                self._capture_queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass

    def _iter_threaded(self):
        processed = 0
        try:
            while self.max_frames <= 0 or processed < self.max_frames:
                try:
                    frame_idx, timestamp_sec, rgb = self._capture_queue.get(timeout=1.0)
                except queue.Empty:
                    if self._capture_stop.is_set():
                        break
                    continue
                if frame_idx % self.frame_step != 0:
                    continue
                self._append_time(timestamp_sec)
                processed += 1
                yield self._build_output(frame_idx, timestamp_sec, rgb)
        finally:
            self.close()

    def _write_calib_and_times_header(self) -> None:
        p0 = np.array(
            [
                [self.K[0, 0], 0.0, self.K[0, 2], 0.0],
                [0.0, self.K[1, 1], self.K[1, 2], 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        rows = ["P0: " + " ".join(f"{v:.8f}" for v in p0.reshape(-1))]
        for idx in range(1, 4):
            rows.append(f"P{idx}: " + " ".join(f"{v:.8f}" for v in p0.reshape(-1)))
        self.calib_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.times_path.write_text("", encoding="utf-8")

    def _append_time(self, timestamp_sec: float) -> None:
        with self.times_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp_sec:.9f}\n")

    def _build_output(self, frame_idx: int, timestamp_sec: float, rgb: np.ndarray) -> TrackingOutput:
        image_path = self.output_dir / f"{frame_idx:06d}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = np.ascontiguousarray(gray)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        keypoints = keypoints or []
        descriptor = compute_global_descriptor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), descriptors)

        match_count, inlier_count, pixel_motion = self._match_stats(keypoints, descriptors)
        if self.tracker is not None:
            timestamp_ns = int(float(timestamp_sec) * 1e9)
            pose_estimate, _ = self.tracker.track(timestamp_ns, [gray])
            track_ok = bool(getattr(pose_estimate, "world_from_rig", None) is not None)
            if track_ok:
                pose = self._pose_estimate_to_matrix(pose_estimate)
                self._last_pose = pose
            else:
                pose = self._last_pose.copy()
        else:
            pose, track_ok = self._estimate_opencv_pose(keypoints, descriptors)

        if self._prev_pose is None:
            is_keyframe = True
        elif not track_ok:
            is_keyframe = False
        else:
            is_keyframe = self._is_keyframe(frame_idx, pose, match_count, pixel_motion)
        if is_keyframe:
            self._last_keyframe_idx = frame_idx

        self._prev_keypoints = keypoints
        self._prev_descriptors = descriptors
        self._prev_gray = gray
        self._prev_pose = pose
        notes = {
            "source": "hikrobot_mono_rgb",
            "odometry_mode": self._odometry_mode,
            "rgb_output_dir": str(self.output_dir),
        }
        if not track_ok:
            notes["warning"] = "cuvslam_world_from_rig_missing_reusing_last_pose"
        return TrackingOutput(
            frame_idx=frame_idx,
            timestamp_sec=float(timestamp_sec),
            pose=pose,
            is_keyframe=is_keyframe,
            descriptor=descriptor,
            orb_descriptors=descriptors,
            keypoints_xy=self._keypoints_to_array(keypoints),
            keypoint_count=len(keypoints),
            match_count=match_count,
            inlier_count=inlier_count,
            pixel_motion=pixel_motion,
            track_ok=track_ok,
            frame_shape=gray.shape[:2],
            image_path=str(image_path),
            right_image_path=None,
            notes=notes,
        )

    def _estimate_opencv_pose(
        self,
        keypoints: list,
        descriptors: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        if self._prev_keypoints is None or self._prev_descriptors is None or descriptors is None:
            return self._last_pose.copy(), True
        raw_matches = self.matcher.knnMatch(self._prev_descriptors, descriptors, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good.append(first)
        if len(good) < 12:
            return self._last_pose.copy(), False
        prev_pts = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good])
        curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good])
        essential, mask = cv2.findEssentialMat(
            prev_pts,
            curr_pts,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if essential is None:
            return self._last_pose.copy(), False
        try:
            _, rotation, translation, _ = cv2.recoverPose(essential, prev_pts, curr_pts, self.K, mask=mask)
        except cv2.error:
            return self._last_pose.copy(), False
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = rotation.astype(np.float32)
        t = translation.reshape(3).astype(np.float32)
        norm = float(np.linalg.norm(t))
        if norm > 1e-6:
            t = t / norm
        delta[:3, 3] = t * self.fixed_step_scale
        pose = (self._last_pose @ np.linalg.inv(delta).astype(np.float32)).astype(np.float32)
        self._last_pose = pose
        return pose, True

    def _pose_estimate_to_matrix(self, pose_estimate: Any) -> np.ndarray:
        pose_raw = pose_estimate.world_from_rig.pose
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = _quaternion_to_matrix(np.asarray(pose_raw.rotation, dtype=np.float32))
        matrix[:3, 3] = np.asarray(pose_raw.translation, dtype=np.float32)
        return matrix

    def _match_stats(self, keypoints: list, descriptors: np.ndarray | None) -> tuple[int, int, float]:
        if self._prev_keypoints is None or self._prev_descriptors is None or descriptors is None:
            return 0, 0, 0.0
        raw_matches = self.matcher.knnMatch(self._prev_descriptors, descriptors, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.config.ratio_test * second.distance:
                good_matches.append(first)
        match_count = len(good_matches)
        if match_count == 0:
            return 0, 0, 0.0
        prev_points = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good_matches])
        curr_points = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
        displacements = np.linalg.norm(curr_points - prev_points, axis=1)
        pixel_motion = float(np.median(displacements))
        inlier_count = match_count
        if match_count >= 8:
            homography, mask = cv2.findHomography(prev_points, curr_points, cv2.RANSAC, 3.0)
            if homography is not None and mask is not None:
                inlier_count = int(mask.ravel().sum())
        return match_count, inlier_count, pixel_motion

    def _is_keyframe(self, frame_idx: int, pose: np.ndarray, match_count: int, pixel_motion: float) -> bool:
        frames_since_kf = frame_idx - self._last_keyframe_idx
        if frames_since_kf >= self.config.max_keyframe_gap:
            return True
        if frames_since_kf < self.config.min_keyframe_gap:
            return False
        if pixel_motion >= self.config.keyframe_motion_threshold:
            return True
        if match_count < self.config.low_match_keyframe_threshold:
            return True
        if self._prev_pose is None:
            return True
        translation_step = float(np.linalg.norm(pose[:3, 3] - self._prev_pose[:3, 3]))
        return translation_step >= self.config.min_translation_step

    def _keypoints_to_array(self, keypoints: list) -> np.ndarray:
        if not keypoints:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def _load_kitti_intrinsic(sequence_dir: Path, camera_index: int = 0) -> np.ndarray:
    calib = np.loadtxt(sequence_dir / "calib.txt", usecols=range(1, 13))[:4].reshape(4, 3, 4)
    idx = min(max(int(camera_index), 0), calib.shape[0] - 1)
    return calib[idx, :3, :3].astype(np.float32)


def _scaled_intrinsic(K: np.ndarray, original_shape_hw: tuple[int, int], depth_shape_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = original_shape_hw
    dst_h, dst_w = depth_shape_hw
    scaled = K.copy().astype(np.float32)
    scaled[0, :] *= float(dst_w) / max(float(src_w), 1.0)
    scaled[1, :] *= float(dst_h) / max(float(src_h), 1.0)
    return scaled


def _load_rgb(path: str) -> np.ndarray | None:
    frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    return frame_bgr[:, :, ::-1].astype(np.uint8)


SEMANTIC_PALETTE = np.array(
    [
        [220, 20, 60],
        [0, 168, 132],
        [255, 170, 0],
        [86, 156, 214],
        [178, 102, 255],
        [45, 212, 191],
        [250, 204, 21],
        [244, 114, 182],
        [132, 204, 22],
        [251, 146, 60],
        [96, 165, 250],
        [167, 139, 250],
    ],
    dtype=np.uint8,
)


class YOLOSemanticProjector:
    def __init__(self, model_path: str, *, conf: float, imgsz: int) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.enabled = False
        self.error = ""
        self._cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, np.ndarray]] = {}
        if not self.model_path:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.model_path)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on optional package.
            self.model = None
            self.error = f"YOLO disabled: {type(exc).__name__}: {exc}"

    def project(
        self,
        image_path: str,
        rgb: np.ndarray | None,
        depth_shape_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros(depth_shape_hw, dtype=np.int32) - 1
        scores = np.zeros(depth_shape_hw, dtype=np.float32)
        if not self.enabled or rgb is None:
            return labels, scores
        key = (str(image_path), (int(depth_shape_hw[0]), int(depth_shape_hw[1])))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - optional runtime path.
            self.error = f"YOLO inference failed: {type(exc).__name__}: {exc}"
            return labels, scores
        if not results:
            self._cache[key] = (labels, scores)
            return labels, scores

        image_h, image_w = rgb.shape[:2]
        depth_h, depth_w = depth_shape_hw
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            self._cache[key] = (labels, scores)
            return labels, scores

        xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else np.zeros((0, 4))
        cls = boxes.cls.detach().cpu().numpy().astype(np.int32) if getattr(boxes, "cls", None) is not None else np.zeros((0,), dtype=np.int32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32) if getattr(boxes, "conf", None) is not None else np.zeros((0,), dtype=np.float32)
        order = np.argsort(conf)
        for det_idx in order:
            x0, y0, x1, y1 = xyxy[det_idx]
            dx0 = int(np.clip(np.floor(x0 * depth_w / max(image_w, 1)), 0, depth_w - 1))
            dx1 = int(np.clip(np.ceil(x1 * depth_w / max(image_w, 1)), 0, depth_w))
            dy0 = int(np.clip(np.floor(y0 * depth_h / max(image_h, 1)), 0, depth_h - 1))
            dy1 = int(np.clip(np.ceil(y1 * depth_h / max(image_h, 1)), 0, depth_h))
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            score = float(conf[det_idx])
            update = scores[dy0:dy1, dx0:dx1] <= score
            patch_labels = labels[dy0:dy1, dx0:dx1]
            patch_scores = scores[dy0:dy1, dx0:dx1]
            patch_labels[update] = int(cls[det_idx])
            patch_scores[update] = score
        self._cache[key] = (labels, scores)
        return labels, scores


def _semantic_colors(labels: np.ndarray, fallback_rgb: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    out = np.asarray(fallback_rgb, dtype=np.uint8).copy()
    valid = labels >= 0
    if valid.any():
        out[valid] = SEMANTIC_PALETTE[labels[valid] % len(SEMANTIC_PALETTE)]
    return out


def _resolve_color_image_path(args: argparse.Namespace, frame_idx: int) -> str:
    if not args.color_image_dir:
        return ""
    color_dir = Path(args.color_image_dir).expanduser()
    try:
        name = args.color_image_template.format(frame_idx=int(frame_idx))
    except Exception:
        name = f"{int(frame_idx):010d}.png"
    candidate = color_dir / name
    if candidate.exists():
        return str(candidate)
    for fallback_name in (
        f"{int(frame_idx):06d}.png",
        f"{int(frame_idx):010d}.png",
        f"{int(frame_idx)}.png",
    ):
        fallback = color_dir / fallback_name
        if fallback.exists():
            return str(fallback)
    return str(candidate)


def _sample_rgb(rgb: np.ndarray | None, xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb is None:
        return np.full((xs.shape[0], 3), 180, dtype=np.uint8)
    sample_x = np.clip((xs.astype(np.float32) + 0.5) * rgb.shape[1] / max(width, 1), 0, rgb.shape[1] - 1)
    sample_y = np.clip((ys.astype(np.float32) + 0.5) * rgb.shape[0] / max(height, 1), 0, rgb.shape[0] - 1)
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, rgb.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, rgb.shape[0] - 1)
    wx = (sample_x - x0).reshape(-1, 1)
    wy = (sample_y - y0).reshape(-1, 1)
    top = rgb[y0, x0].astype(np.float32) * (1.0 - wx) + rgb[y0, x1].astype(np.float32) * wx
    bottom = rgb[y1, x0].astype(np.float32) * (1.0 - wx) + rgb[y1, x1].astype(np.float32) * wx
    colors = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def _backproject(K: np.ndarray, pose: np.ndarray, xs: np.ndarray, ys: np.ndarray, depth: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth.astype(np.float32)
    x_cam = (xs.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y_cam = (ys.astype(np.float32) - cy) * z / max(fy, 1e-6)
    camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    return (camera @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def _lingbot_extrinsic_to_pose(
    extrinsic: np.ndarray,
    translation_scale: float = 1.0,
    mode: str = "inverse",
) -> np.ndarray | None:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (3, 4):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :4] = extrinsic
    elif extrinsic.shape == (4, 4):
        matrix = extrinsic.astype(np.float32).copy()
    else:
        return None
    if mode == "inverse":
        pose = np.linalg.inv(matrix).astype(np.float32)
    elif mode == "direct":
        pose = matrix.astype(np.float32).copy()
    else:
        raise ValueError(f"Unknown LingBot extrinsic mode: {mode}")
    pose[:3, 3] *= float(translation_scale)
    return pose


def _sample_pixels(height: int, width: int, stride: int, pattern: str = "grid", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    pattern = str(pattern)
    if pattern == "random":
        count_y = int(math.ceil(height / stride))
        count_x = int(math.ceil(width / stride))
        count = min(int(height * width), max(1, count_y * count_x))
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        linear = rng.choice(int(height * width), size=count, replace=False)
        ys = (linear // int(width)).astype(np.int32)
        xs = (linear % int(width)).astype(np.int32)
        return xs, ys

    ys_grid, xs_grid = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    if pattern == "jittered" and stride > 1 and xs_grid.size:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        xs_grid = np.clip(xs_grid + rng.integers(-(stride // 2), stride // 2 + 1, size=xs_grid.shape), 0, width - 1)
        ys_grid = np.clip(ys_grid + rng.integers(-(stride // 2), stride // 2 + 1, size=ys_grid.shape), 0, height - 1)
    return xs_grid.reshape(-1).astype(np.int32), ys_grid.reshape(-1).astype(np.int32)


def _grid_pixels(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    return _sample_pixels(height, width, stride, "grid", 0)


def _edge_score(rgb: np.ndarray | None, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    if rgb is None:
        return np.zeros(depth_shape_hw, dtype=np.float32)
    gray = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (int(depth_shape_hw[1]), int(depth_shape_hw[0])), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth


def _sample_frame_points(
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    rgb: np.ndarray | None,
    pose: np.ndarray,
    K: np.ndarray,
    frame_idx: int,
    args: argparse.Namespace,
    semantic_projector: YOLOSemanticProjector | None = None,
    image_path: str = "",
) -> dict[str, np.ndarray]:
    height, width = depth.shape
    stride = max(1, int(args.sample_stride))
    sampling_pattern = str(getattr(args, "sampling_pattern", "grid"))
    semantic_labels = np.zeros(depth.shape, dtype=np.int32) - 1
    semantic_scores = np.zeros(depth.shape, dtype=np.float32)
    if semantic_projector is not None:
        semantic_labels, semantic_scores = semantic_projector.project(image_path, rgb, depth.shape)

    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    xs_base, ys_base = _sample_pixels(height, width, stride, sampling_pattern, seed=int(frame_idx) * 1009 + 17)
    xs_parts.append(xs_base)
    ys_parts.append(ys_base)
    edge = np.zeros(depth.shape, dtype=np.float32)
    if getattr(args, "adaptive_sampling", False):
        metric_depth = depth.astype(np.float32) * float(args.depth_scale)
        near_stride = max(1, int(getattr(args, "near_sample_stride", stride)))
        if near_stride < stride:
            xs_near, ys_near = _sample_pixels(height, width, near_stride, sampling_pattern, seed=int(frame_idx) * 1009 + 29)
            near = metric_depth[ys_near, xs_near] <= float(getattr(args, "near_depth_m", 18.0))
            xs_parts.append(xs_near[near])
            ys_parts.append(ys_near[near])

        edge_stride = max(1, int(getattr(args, "edge_sample_stride", stride)))
        if edge_stride < stride:
            edge = _edge_score(rgb, depth.shape)
            threshold = np.percentile(edge[np.isfinite(edge)], float(getattr(args, "edge_percentile", 88.0))) if edge.size else np.inf
            xs_edge, ys_edge = _sample_pixels(height, width, edge_stride, sampling_pattern, seed=int(frame_idx) * 1009 + 43)
            keep_edge = edge[ys_edge, xs_edge] >= threshold
            xs_parts.append(xs_edge[keep_edge])
            ys_parts.append(ys_edge[keep_edge])

        semantic_stride = max(1, int(getattr(args, "semantic_sample_stride", stride)))
        if semantic_stride < stride and np.any(semantic_labels >= 0):
            xs_sem, ys_sem = _sample_pixels(height, width, semantic_stride, sampling_pattern, seed=int(frame_idx) * 1009 + 61)
            keep_sem = semantic_labels[ys_sem, xs_sem] >= 0
            xs_parts.append(xs_sem[keep_sem])
            ys_parts.append(ys_sem[keep_sem])

    xs = np.concatenate(xs_parts, axis=0) if xs_parts else np.zeros((0,), dtype=np.int32)
    ys = np.concatenate(ys_parts, axis=0) if ys_parts else np.zeros((0,), dtype=np.int32)
    if xs.size:
        linear = ys.astype(np.int64) * int(width) + xs.astype(np.int64)
        _, unique_idx = np.unique(linear, return_index=True)
        xs = xs[unique_idx]
        ys = ys[unique_idx]
    values = depth[ys, xs].astype(np.float32) * float(args.depth_scale)
    scores = conf[ys, xs].astype(np.float32)
    sem_scores = semantic_scores[ys, xs].astype(np.float32)
    sem_labels = semantic_labels[ys, xs].astype(np.int32)
    if getattr(args, "adaptive_sampling", False):
        if not edge.any():
            edge = _edge_score(rgb, depth.shape)
        local_edge = edge[ys, xs].astype(np.float32)
        if local_edge.size and np.nanmax(local_edge) > 0:
            local_edge = local_edge / max(float(np.nanmax(local_edge)), 1e-6)
        near_bonus = (values <= float(getattr(args, "near_depth_m", 18.0))).astype(np.float32) * 0.15
        scores = scores + 0.35 * sem_scores + 0.20 * local_edge + near_bonus
    valid = np.isfinite(values) & np.isfinite(scores)
    valid &= values >= float(args.min_depth)
    valid &= values <= float(args.max_depth)
    valid &= scores >= float(args.min_conf)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size > args.max_points_per_frame > 0:
        if sampling_pattern == "random":
            rng = np.random.default_rng((int(frame_idx) * 9176 + int(valid_idx.size) * 37) & 0xFFFFFFFF)
            valid_idx = rng.choice(valid_idx, size=int(args.max_points_per_frame), replace=False)
        else:
            keep_local = np.argpartition(scores[valid_idx], -args.max_points_per_frame)[-args.max_points_per_frame :]
            valid_idx = valid_idx[keep_local]
    if valid_idx.size == 0:
        return empty_point_batch()
    xs_keep = xs[valid_idx]
    ys_keep = ys[valid_idx]
    depth_keep = values[valid_idx]
    xyz = _backproject(K, pose, xs_keep, ys_keep, depth_keep)
    colors = _sample_rgb(rgb, xs_keep, ys_keep, width, height)
    labels_keep = sem_labels[valid_idx]
    sem_conf_keep = sem_scores[valid_idx]
    if getattr(args, "semantic_color_output", False):
        colors = _semantic_colors(labels_keep, colors)
    return {
        "xyz": xyz,
        "rgb": colors,
        "frame": np.full((xyz.shape[0],), int(frame_idx), dtype=np.int32),
        "semantic_label": labels_keep.astype(np.int32),
        "semantic_conf": sem_conf_keep.astype(np.float32),
        "observations": np.ones((xyz.shape[0],), dtype=np.int32),
    }


def _write_ascii_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    semantic_label: np.ndarray | None = None,
    semantic_conf: np.ndarray | None = None,
    observations: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_label = semantic_label if semantic_label is not None else np.zeros((xyz.shape[0],), dtype=np.int32) - 1
    semantic_conf = semantic_conf if semantic_conf is not None else np.zeros((xyz.shape[0],), dtype=np.float32)
    observations = observations if observations is not None else np.ones((xyz.shape[0],), dtype=np.int32)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int semantic_label\nproperty float semantic_conf\nproperty int observations\n")
        handle.write("end_header\n")
        for point, color, label, sem_conf, obs in zip(xyz, rgb, semantic_label, semantic_conf, observations):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(label)} {float(sem_conf):.4f} {int(obs)}\n"
            )


def _collect_live_arrays(
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if frame_points:
        xyz = np.concatenate([item["xyz"] for item in frame_points.values()], axis=0)
        rgb = np.concatenate([item["rgb"] for item in frame_points.values()], axis=0)
        frames = np.concatenate([item["frame"] for item in frame_points.values()], axis=0)
        semantic_label = np.concatenate(
            [item.get("semantic_label", np.full((item["xyz"].shape[0],), -1, dtype=np.int32)) for item in frame_points.values()],
            axis=0,
        )
        semantic_conf = np.concatenate(
            [item.get("semantic_conf", np.zeros((item["xyz"].shape[0],), dtype=np.float32)) for item in frame_points.values()],
            axis=0,
        )
        observations = np.concatenate(
            [item.get("observations", np.ones((item["xyz"].shape[0],), dtype=np.int32)) for item in frame_points.values()],
            axis=0,
        )
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        frames = np.zeros((0,), dtype=np.int32)
        semantic_label = np.zeros((0,), dtype=np.int32) - 1
        semantic_conf = np.zeros((0,), dtype=np.float32)
        observations = np.zeros((0,), dtype=np.int32)
    return xyz, rgb, frames, semantic_label, semantic_conf, observations


def _write_live_json(
    output_dir: Path,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    trajectory: list[dict[str, Any]],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    xyz, rgb, frames, semantic_label, semantic_conf, observations = _collect_live_arrays(frame_points)
    if xyz.shape[0] > 0:
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
    else:
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.ones(3, dtype=np.float32)

    max_json_points = 60000
    if xyz.shape[0] > max_json_points:
        idx = np.linspace(0, xyz.shape[0] - 1, max_json_points).astype(np.int64)
        xyz_json, rgb_json, frames_json = xyz[idx], rgb[idx], frames[idx]
        semantic_json, semantic_conf_json = semantic_label[idx], semantic_conf[idx]
    else:
        xyz_json, rgb_json, frames_json = xyz, rgb, frames
        semantic_json, semantic_conf_json = semantic_label, semantic_conf
    points = [
        [
            round(float(p[0]), 4),
            round(float(p[1]), 4),
            round(float(p[2]), 4),
            int(c[0]),
            int(c[1]),
            int(c[2]),
            int(f),
            int(label),
            round(float(sem_conf), 4),
        ]
        for p, c, f, label, sem_conf in zip(xyz_json, rgb_json, frames_json, semantic_json, semantic_conf_json)
    ]
    payload = {
        "schema": "lingbot_live_reconstruction.v1",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_count": int(xyz.shape[0]),
        "shown_point_count": int(xyz_json.shape[0]),
        "active_frames": list(frame_points.keys()),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "trajectory": trajectory,
        "events": events[-80:],
        "settings": {
            "tracking_backend": str(args.tracking_backend),
            "rgb_image_dir": str(args.rgb_image_dir),
            "intrinsic_camera_index": int(args.intrinsic_camera_index),
            "image_size": int(args.image_size),
            "frame_step": int(args.frame_step),
            "color_image_dir": str(args.color_image_dir),
            "color_image_template": str(args.color_image_template),
            "depth_scale": float(args.depth_scale),
            "lingbot_pose_translation_scale": float(
                getattr(args, "lingbot_pose_translation_scale", 0.0) or getattr(args, "depth_scale", 1.0)
            ),
            "lingbot_extrinsic_mode": str(getattr(args, "lingbot_extrinsic_mode", "inverse")),
            "sample_stride": int(args.sample_stride),
            "sampling_pattern": str(getattr(args, "sampling_pattern", "grid")),
            "max_points_per_frame": int(args.max_points_per_frame),
            "fusion_mode": str(getattr(args, "fusion_mode", "raw")),
            "voxel_size": float(getattr(args, "voxel_size", 0.0)),
            "fusion_max_points": int(getattr(args, "fusion_max_points", 0)),
            "adaptive_sampling": bool(getattr(args, "adaptive_sampling", False)),
            "yolo_model": str(getattr(args, "yolo_model", "")),
            "semantic_color_output": bool(getattr(args, "semantic_color_output", False)),
        },
        "points": points,
    }
    json_tmp = output_dir / "live_map.json.tmp"
    json_tmp.write_text(json.dumps(payload), encoding="utf-8")
    json_tmp.replace(output_dir / "live_map.json")
    npz_tmp = output_dir / "live_map.npz.tmp"
    with npz_tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            xyz=xyz,
            rgb=rgb,
            frame=frames,
            semantic_label=semantic_label,
            semantic_conf=semantic_conf,
            observations=observations,
        )
    npz_tmp.replace(output_dir / "live_map.npz")
    _write_ascii_ply(output_dir / "live_map.ply", xyz, rgb, semantic_label, semantic_conf, observations)
    return payload


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0.0}
    ordered = sorted(finite)
    return {
        "count": float(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(ordered[len(ordered) // 2]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p90": float(ordered[min(len(ordered) - 1, int(math.ceil(0.90 * len(ordered))) - 1)]),
    }


def _write_viewer_html(output_dir: Path) -> None:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LingBot Live Reconstruction</title>
  <style>
    :root { --bg:#07090b; --panel:#111820; --text:#e9f1e7; --muted:#9aa89c; --accent:#c88b45; }
    html, body { margin:0; height:100%; overflow:hidden; background:radial-gradient(circle at 20% 10%, #18251f 0, #07090b 48%, #030405 100%); color:var(--text); font-family: ui-sans-serif, system-ui, sans-serif; }
    #hud { position:fixed; left:18px; top:18px; z-index:10; background:rgba(9,13,16,.76); border:1px solid rgba(255,255,255,.12); border-radius:18px; padding:14px 16px; backdrop-filter: blur(12px); min-width:300px; box-shadow:0 20px 60px rgba(0,0,0,.3); }
    h1 { margin:0 0 6px; font-size:16px; letter-spacing:.08em; color:#f2c58b; text-transform:uppercase; }
    .row { display:flex; justify-content:space-between; gap:16px; font-size:13px; color:var(--muted); line-height:1.7; }
    .row b { color:var(--text); font-weight:650; }
    #hint { position:fixed; right:18px; bottom:18px; color:var(--muted); background:rgba(9,13,16,.66); padding:10px 12px; border-radius:14px; border:1px solid rgba(255,255,255,.1); font-size:12px; }
    canvas { display:block; width:100vw; height:100vh; cursor:grab; }
    canvas:active { cursor:grabbing; }
  </style>
</head>
<body>
  <div id="hud">
    <h1>LingBot Live Reconstruction</h1>
    <div class="row"><span>Status</span><b id="status">loading</b></div>
    <div class="row"><span>Points</span><b id="points">0</b></div>
    <div class="row"><span>Active frames</span><b id="frames">0</b></div>
    <div class="row"><span>Last event</span><b id="event">none</b></div>
    <div class="row"><span>Updated</span><b id="updated">-</b></div>
  </div>
  <canvas id="view"></canvas>
  <div id="hint">drag rotate · wheel zoom · auto-refreshes live_map.json</div>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
let data = {points:[], trajectory:[], bbox_min:[0,0,0], bbox_max:[1,1,1], events:[]};
let yaw = -0.72, pitch = -0.58, zoom = 1.0, dragging = false, lastX = 0, lastY = 0;
function resize(){ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; }
addEventListener('resize', resize); resize();
canvas.addEventListener('mousedown', e => { dragging = true; lastX=e.clientX; lastY=e.clientY; });
addEventListener('mouseup', () => dragging = false);
addEventListener('mousemove', e => { if(!dragging) return; yaw += (e.clientX-lastX)*0.006; pitch += (e.clientY-lastY)*0.006; pitch=Math.max(-1.45, Math.min(1.45, pitch)); lastX=e.clientX; lastY=e.clientY; draw(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); zoom=Math.max(.15, Math.min(8, zoom)); draw(); }, {passive:false});
async function load(){
  try {
    const res = await fetch('live_map.json?t=' + Date.now(), {cache:'no-store'});
    data = await res.json();
    document.getElementById('status').textContent = 'live';
    document.getElementById('points').textContent = `${data.shown_point_count}/${data.point_count}`;
    document.getElementById('frames').textContent = (data.active_frames||[]).join(', ');
    const ev = (data.events||[]).slice(-1)[0];
    document.getElementById('event').textContent = ev ? `${ev.type} ${ev.frame_idx ?? ev.window ?? ''}` : 'none';
    document.getElementById('updated').textContent = data.updated_at || '-';
    draw();
  } catch(e) { document.getElementById('status').textContent = 'waiting'; }
}
function rot(p){
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  let x=p[0], y=p[1], z=p[2];
  let x1=cy*x + sy*z, z1=-sy*x + cy*z;
  let y1=cp*y - sp*z1, z2=sp*y + cp*z1;
  return [x1,y1,z2];
}
function project(p, center, scale){
  const q=rot([(p[0]-center[0])*scale, (p[1]-center[1])*scale, (p[2]-center[2])*scale]);
  const s = Math.min(canvas.width, canvas.height) * 0.42 * zoom;
  return [canvas.width*0.5 + q[0]*s, canvas.height*0.54 - q[1]*s, q[2]];
}
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const mn=data.bbox_min||[0,0,0], mx=data.bbox_max||[1,1,1];
  const center=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  const span=Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1e-3);
  const scale=1/span;
  ctx.fillStyle='#071016'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const traj=(data.trajectory||[]).filter(t=>t.position);
  if(traj.length>1){
    ctx.lineWidth=2*devicePixelRatio; ctx.strokeStyle='rgba(244,169,83,.88)'; ctx.beginPath();
    traj.forEach((t,i)=>{ const q=project(t.position,center,scale); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); });
    ctx.stroke();
  }
  const pts=(data.points||[]).map(p=>{ const q=project(p,center,scale); return {p,q}; }).sort((a,b)=>a.q[2]-b.q[2]);
  const r=Math.max(1.0, 1.7*devicePixelRatio);
  for(const item of pts){
    const p=item.p, q=item.q;
    const shade=Math.max(.25, Math.min(1.25, .75 + q[2]*.25));
    ctx.fillStyle=`rgb(${Math.min(255,p[3]*shade)},${Math.min(255,p[4]*shade)},${Math.min(255,p[5]*shade)})`;
    ctx.fillRect(q[0],q[1],r,r);
  }
}
setInterval(load, 1000); load();
</script>
</body>
</html>
"""
    (output_dir / "live_viewer.html").write_text(html, encoding="utf-8")


def _find_free_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


def _start_http_server(output_dir: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0"],
        cwd=str(output_dir),
        stdout=(output_dir / "http_server.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _process_worker_result(
    result_path: Path,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    trajectory_by_frame: dict[int, dict[str, Any]],
    K_base: np.ndarray,
    args: argparse.Namespace,
    semantic_projector: YOLOSemanticProjector | None = None,
    fusion_map: VoxelFusionMap | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    process_started = time.perf_counter()
    pred_path = Path(result["predictions_npz"])
    summary_path = Path(result["summary_json"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pred = np.load(pred_path)
    depth = _squeeze_depth(pred["depth"])
    conf = _squeeze_depth(pred["depth_conf"]) if "depth_conf" in pred else np.ones_like(depth, dtype=np.float32)
    lingbot_extrinsic = np.asarray(pred["extrinsic"], dtype=np.float32) if "extrinsic" in pred else None
    lingbot_intrinsic = np.asarray(pred["intrinsic"], dtype=np.float32) if "intrinsic" in pred else None
    lingbot_translation_scale = float(
        getattr(args, "lingbot_pose_translation_scale", 0.0) or getattr(args, "depth_scale", 1.0)
    )
    frames = summary.get("metadata", {}).get("frames", [])
    updated = 0
    total_points = 0
    current_batches: list[dict[str, np.ndarray]] = []
    for local_idx, frame in enumerate(frames[: depth.shape[0]]):
        meta = frame.get("metadata") or {}
        frame_idx = int(frame.get("frame_idx", local_idx))
        pose = None
        if "pose" in meta:
            pose = np.asarray(meta["pose"], dtype=np.float32)
        elif frame_idx in trajectory_by_frame and "pose" in trajectory_by_frame[frame_idx]:
            pose = np.asarray(trajectory_by_frame[frame_idx]["pose"], dtype=np.float32)
        elif lingbot_extrinsic is not None and local_idx < lingbot_extrinsic.shape[0]:
            pose = _lingbot_extrinsic_to_pose(
                lingbot_extrinsic[local_idx],
                translation_scale=lingbot_translation_scale,
                mode=getattr(args, "lingbot_extrinsic_mode", "inverse"),
            )
        if pose is None:
            continue
        image_path = str(frame.get("image_path", ""))
        color_image_path = _resolve_color_image_path(args, frame_idx)
        sample_image_path = color_image_path or image_path
        original_shape = tuple(meta.get("frame_shape", [0, 0]))
        if len(original_shape) != 2 or original_shape[0] <= 0:
            rgb_probe = _load_rgb(sample_image_path)
            original_shape = rgb_probe.shape[:2] if rgb_probe is not None else depth[local_idx].shape
            rgb = rgb_probe
        else:
            rgb = _load_rgb(sample_image_path)
        if lingbot_intrinsic is not None and local_idx < lingbot_intrinsic.shape[0]:
            K = np.asarray(lingbot_intrinsic[local_idx], dtype=np.float32)
        else:
            K = _scaled_intrinsic(K_base, (int(original_shape[0]), int(original_shape[1])), depth[local_idx].shape)
        sampled = _sample_frame_points(
            depth=depth[local_idx],
            conf=conf[local_idx],
            rgb=rgb,
            pose=pose,
            K=K,
            frame_idx=frame_idx,
            args=args,
            semantic_projector=semantic_projector,
            image_path=sample_image_path,
        )
        if fusion_map is not None:
            fusion_map.update(sampled)
        else:
            frame_points[frame_idx] = sampled
            frame_points.move_to_end(frame_idx)
            while len(frame_points) > max(1, args.max_active_frames):
                frame_points.popitem(last=False)
        total_points += int(sampled["xyz"].shape[0])
        if sampled["xyz"].shape[0] > 0:
            current_batches.append(sampled)
        updated += 1
    process_elapsed = time.perf_counter() - process_started
    event = {
        "type": "window",
        "window": int(result["index"]),
        "updated_frames": updated,
        "points_added_or_replaced": total_points,
        "elapsed_sec": float(result["elapsed_sec"]),
        "queue_wait_sec": float(result.get("queue_wait_sec", 0.0)),
        "worker_end_to_end_sec": float(result.get("end_to_end_sec", result["elapsed_sec"])),
        "process_result_sec": process_elapsed,
    }
    if current_batches:
        event["_current_xyz"] = np.concatenate([item["xyz"] for item in current_batches], axis=0)
        event["_current_rgb"] = np.concatenate([item["rgb"] for item in current_batches], axis=0)
    return event


def _refresh_fusion_snapshot(
    fusion_map: VoxelFusionMap | None,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    fallback_frame_idx: int,
) -> int:
    if fusion_map is None:
        return int(sum(item["xyz"].shape[0] for item in frame_points.values()))
    snapshot = fusion_map.snapshot()
    frame_points.clear()
    if snapshot["xyz"].shape[0] > 0:
        frame_idx = int(snapshot["frame"].max())
    else:
        frame_idx = int(fallback_frame_idx)
    frame_points[frame_idx] = snapshot
    return int(snapshot["xyz"].shape[0])


def main() -> int:
    args = parse_args()
    if args.drop_when_busy:
        args.no_blocking_submit = True
    root = REPO_ROOT.parents[1] if False else Path.cwd()
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    trajectory_path = Path(args.trajectory_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lingbot_map_root:
        os.environ["LINGBOT_MAP_ROOT"] = str(Path(args.lingbot_map_root).expanduser().resolve())

    _write_viewer_html(output_dir)
    config = load_runtime_config(args.config or None)
    semantic_projector = YOLOSemanticProjector(args.yolo_model, conf=args.yolo_conf, imgsz=args.yolo_imgsz)
    ros_bridge: ROS2LiveBridge | None = None
    if args.ros2_publish:
        ros_bridge = ROS2LiveBridge(
            image_topic=args.ros2_image_topic,
            camera_info_topic=args.ros2_camera_info_topic,
            pose_topic=args.ros2_pose_topic,
            path_topic=args.ros2_path_topic,
            cloud_topic=args.ros2_cloud_topic,
            plain_cloud_topic=args.ros2_plain_cloud_topic,
            current_cloud_topic=args.ros2_current_cloud_topic,
            current_plain_cloud_topic=args.ros2_current_plain_cloud_topic,
            camera_frame_id=args.ros2_camera_frame_id,
            cloud_frame_id=args.ros2_cloud_frame_id,
            max_cloud_points=args.ros2_max_cloud_points,
            max_current_cloud_points=args.ros2_max_current_cloud_points,
            image_max_width=args.ros2_image_max_width,
            image_max_height=args.ros2_image_max_height,
            republish_current_cloud_on_image=args.ros2_republish_current_cloud_on_image,
            path_max_poses=args.ros2_path_max_poses,
        )
    fusion_map = (
        VoxelFusionMap(
            voxel_size=args.voxel_size,
            max_voxels=args.fusion_max_points,
            min_observations=args.fusion_min_observations,
        )
        if args.fusion_mode == "voxel"
        else None
    )
    if args.tracking_backend == "hikrobot_mono_rgb":
        rgb_output_dir = args.rgb_output_dir or str(output_dir / "rgb_stream")
        source = HikRobotMonocularRGBAdapter(
            output_dir=rgb_output_dir,
            camera_index=args.hikrobot_index,
            timeout_ms=args.hikrobot_timeout_ms,
            exposure_us=args.hikrobot_exposure_us,
            gain=args.hikrobot_gain,
            fps=args.hikrobot_fps,
            capture_width=args.hikrobot_width,
            capture_height=args.hikrobot_height,
            max_read_errors=args.hikrobot_max_read_errors,
            read_error_sleep_sec=args.hikrobot_read_error_sleep_sec,
            threaded_capture=args.hikrobot_threaded_capture,
            capture_queue_size=args.hikrobot_capture_queue_size,
            intrinsic_args=(args.camera_fx, args.camera_fy, args.camera_cx, args.camera_cy),
            config=config.tracking,
            fixed_step_scale=args.mono_fixed_step_scale,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
            frame_callback=ros_bridge.publish_image if ros_bridge is not None else None,
        )
        K_base = source.K.astype(np.float32)
        sequence_dir = Path(rgb_output_dir).expanduser().resolve()
        if not args.color_image_dir:
            args.color_image_dir = str(sequence_dir)
        if not args.rgb_image_dir:
            args.rgb_image_dir = str(sequence_dir)
        args.color_image_template = "{frame_idx:06d}.png"
    else:
        K_base = _load_kitti_intrinsic(sequence_dir, args.intrinsic_camera_index)

    if args.tracking_backend in {"opencv_mono_rgb", "cuvslam_mono_rgb"}:
        rgb_image_dir = args.rgb_image_dir or args.color_image_dir
        if not rgb_image_dir:
            raise SystemExit(f"--tracking-backend {args.tracking_backend} requires --rgb-image-dir or --color-image-dir")
        if not args.color_image_dir:
            args.color_image_dir = rgb_image_dir
        if args.tracking_backend == "cuvslam_mono_rgb":
            source = CUVSLAMMonocularRGBAdapter(
                image_dir=rgb_image_dir,
                timestamps_path=sequence_dir / "times.txt",
                intrinsic=K_base,
                config=config.tracking,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
            )
        else:
            source = RGBMonocularVOAdapter(
                image_dir=rgb_image_dir,
                timestamps_path=sequence_dir / "times.txt",
                intrinsic=K_base,
                config=config.tracking,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                fixed_step_scale=args.mono_fixed_step_scale,
                scale_source=args.mono_scale_source,
                oxts_dir=args.oxts_dir,
            )
    else:
        if args.tracking_backend != "hikrobot_mono_rgb":
            source = CUVSLAMOfflineKITTIAdapter(
                sequence_path=sequence_dir,
                trajectory_path=trajectory_path,
                config=config.tracking,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
            )
    worker = LingBotDepthWorker(
        LingBotDepthWorkerConfig(
            model_path=str(Path(args.model_path).expanduser().resolve()),
            output_dir=str(output_dir / "worker"),
            image_size=args.image_size,
            model_image_size=args.model_image_size,
            window_size=args.window_size,
            stride=args.stride,
            num_scale_frames=args.num_scale_frames,
            max_queue=args.max_queue,
            force_cpu=False,
            offload_to_cpu=True,
            use_sdpa=args.use_sdpa,
            enable_3d_rope=False,
            depth_head_trt_engine=args.depth_head_trt_engine,
            model_patch_embed=args.model_patch_embed,
            model_embed_dim=args.model_embed_dim,
            model_depth=args.model_depth,
            model_num_heads=args.model_num_heads,
            model_mlp_ratio=args.model_mlp_ratio,
            compress_outputs=not args.no_compress_output,
        )
    )
    frame_points: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
    trajectory: list[dict[str, Any]] = []
    trajectory_by_frame: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "track_next_sec": [],
        "submit_sec": [],
        "process_result_sec": [],
        "live_write_sec": [],
        "worker_elapsed_sec": [],
        "worker_queue_wait_sec": [],
        "worker_end_to_end_sec": [],
        "points_added_or_replaced": [],
        "tracking": [],
        "windows": [],
        "dense_submit": [],
    }
    processed_results: set[Path] = set()
    server = None
    if args.serve:
        port = _find_free_port(args.port)
        args.port = port
        server = _start_http_server(output_dir, port)
        events.append({"type": "server", "url": f"http://0.0.0.0:{port}/live_viewer.html"})
    if args.yolo_model and not semantic_projector.enabled:
        events.append({"type": "warning", "message": semantic_projector.error or "YOLO disabled"})
        print(semantic_projector.error or "YOLO disabled", flush=True)

    _write_live_json(output_dir, frame_points, trajectory, events, args)
    worker.start()
    start = time.perf_counter()
    last_cloud_publish_sec = 0.0

    def publish_ros_cloud() -> None:
        nonlocal last_cloud_publish_sec
        if ros_bridge is None:
            return
        now = time.perf_counter()
        min_interval = max(0.0, float(getattr(args, "ros2_cloud_min_interval_sec", 0.0)))
        if min_interval > 0.0 and now - last_cloud_publish_sec < min_interval:
            return
        xyz, rgb, *_ = _collect_live_arrays(frame_points)
        ros_bridge.publish_cloud(xyz, rgb)
        last_cloud_publish_sec = time.perf_counter()

    def process_new_results() -> None:
        for result_path in sorted((output_dir / "worker").glob("window_*/worker_result.json")):
            if result_path in processed_results:
                continue
            try:
                event = _process_worker_result(
                    result_path,
                    frame_points,
                    trajectory_by_frame,
                    K_base,
                    args,
                    semantic_projector=semantic_projector,
                    fusion_map=fusion_map,
                )
            except JSONDecodeError:
                continue
            processed_results.add(result_path)
            current_xyz = event.pop("_current_xyz", None)
            current_rgb = event.pop("_current_rgb", None)
            if ros_bridge is not None and current_xyz is not None and current_rgb is not None:
                ros_bridge.publish_current_cloud(current_xyz, current_rgb)
            events.append(event)
            should_publish = (
                args.publish_every_windows > 0
                and len(processed_results) % args.publish_every_windows == 0
            )
            live_write_sec = 0.0
            point_count = int(sum(item["xyz"].shape[0] for item in frame_points.values()))
            if should_publish:
                point_count = _refresh_fusion_snapshot(fusion_map, frame_points, int(event.get("window", 0)))
                write_started = time.perf_counter()
                payload = _write_live_json(output_dir, frame_points, trajectory, events, args)
                publish_ros_cloud()
                live_write_sec = time.perf_counter() - write_started
                point_count = int(payload["point_count"])
            event["live_write_sec"] = live_write_sec
            event["published"] = bool(should_publish)
            metrics["process_result_sec"].append(float(event.get("process_result_sec", 0.0)))
            if should_publish:
                metrics["live_write_sec"].append(live_write_sec)
            metrics["worker_elapsed_sec"].append(float(event.get("elapsed_sec", 0.0)))
            metrics["worker_queue_wait_sec"].append(float(event.get("queue_wait_sec", 0.0)))
            metrics["worker_end_to_end_sec"].append(float(event.get("worker_end_to_end_sec", event.get("elapsed_sec", 0.0))))
            metrics["points_added_or_replaced"].append(int(event.get("points_added_or_replaced", 0)))
            metrics["windows"].append(event)
            print(json.dumps({"live_update": event, "point_count": point_count}, ensure_ascii=False), flush=True)

    submitted = 0
    dense_candidates = 0
    dense_submitted_frames = 0
    dense_interval_skips = 0
    dense_non_keyframe_skips = 0
    dense_window_schedules = 0
    dense_queue_full_drops_last = 0
    skipped = 0
    try:
        source_iter = iter(source)
        while True:
            track_started = time.perf_counter()
            try:
                item = next(source_iter)
            except StopIteration:
                break
            track_next_sec = time.perf_counter() - track_started
            metrics["track_next_sec"].append(track_next_sec)
            pose = item.pose.astype(np.float32)
            traj_item = {
                "frame_idx": int(item.frame_idx),
                "timestamp_sec": float(item.timestamp_sec),
                "position": pose[:3, 3].astype(float).tolist(),
                "pose": pose.astype(float).tolist(),
                "is_keyframe": bool(item.is_keyframe),
                "track_ok": bool(item.track_ok),
            }
            trajectory.append(traj_item)
            trajectory_by_frame[int(item.frame_idx)] = traj_item
            if ros_bridge is not None:
                ros_bridge.publish_pose_path(pose, trajectory)
            events.append({"type": "track", "frame_idx": int(item.frame_idx), "is_keyframe": bool(item.is_keyframe)})
            if item.image_path is None:
                continue
            should_submit_dense = True
            dense_skip_reason = ""
            if args.keyframes_only and not item.is_keyframe:
                should_submit_dense = False
                dense_skip_reason = "non_keyframe"
                dense_non_keyframe_skips += 1
            dense_candidates += 1
            if should_submit_dense and args.dense_frame_interval > 1:
                if (dense_candidates - 1) % int(args.dense_frame_interval) != 0:
                    should_submit_dense = False
                    dense_skip_reason = "dense_frame_interval"
                    dense_interval_skips += 1
            if not should_submit_dense:
                skipped += 1
                metrics["tracking"].append(
                    {
                        "frame_idx": int(item.frame_idx),
                        "track_next_sec": track_next_sec,
                        "submit_sec": 0.0,
                        "dense_submitted": False,
                        "dense_skip_reason": dense_skip_reason,
                        "is_keyframe": bool(item.is_keyframe),
                        "track_ok": bool(item.track_ok),
                        "keypoint_count": int(item.keypoint_count),
                        "match_count": int(item.match_count),
                        "inlier_count": int(item.inlier_count),
                        "pixel_motion": float(item.pixel_motion),
                    }
                )
                process_new_results()
                if args.frame_sleep_sec > 0:
                    time.sleep(args.frame_sleep_sec)
                continue
            submit_started = time.perf_counter()
            scheduled_window = worker.submit(
                item.image_path,
                frame_idx=int(item.frame_idx),
                timestamp_sec=float(item.timestamp_sec),
                metadata={
                    "pose": pose.tolist(),
                    "frame_shape": [int(item.frame_shape[0]), int(item.frame_shape[1])],
                    "is_keyframe": bool(item.is_keyframe),
                    "track_ok": bool(item.track_ok),
                    "keypoint_count": int(item.keypoint_count),
                    "match_count": int(item.match_count),
                    "inlier_count": int(item.inlier_count),
                    "pixel_motion": float(item.pixel_motion),
                },
                block=not args.no_blocking_submit,
            )
            submit_sec = time.perf_counter() - submit_started
            dense_submitted_frames += 1
            worker_status = worker.status()
            queue_full_drops = int(worker_status.get("queue_full_drops", 0))
            queue_full_drop_delta = max(0, queue_full_drops - dense_queue_full_drops_last)
            dense_queue_full_drops_last = queue_full_drops
            if scheduled_window:
                dense_window_schedules += 1
            metrics["submit_sec"].append(submit_sec)
            metrics["dense_submit"].append(
                {
                    "frame_idx": int(item.frame_idx),
                    "submit_sec": submit_sec,
                    "scheduled_window": bool(scheduled_window),
                    "queue_full_drop_delta": int(queue_full_drop_delta),
                    "worker_queue_size": int(worker_status.get("queue_size", 0)),
                    "worker_scheduled_windows": int(worker_status.get("scheduled_windows", 0)),
                    "worker_completed_windows": int(worker_status.get("completed_windows", 0)),
                }
            )
            metrics["tracking"].append(
                {
                    "frame_idx": int(item.frame_idx),
                    "track_next_sec": track_next_sec,
                    "submit_sec": submit_sec,
                    "dense_submitted": True,
                    "dense_scheduled_window": bool(scheduled_window),
                    "dense_queue_full_drop_delta": int(queue_full_drop_delta),
                    "is_keyframe": bool(item.is_keyframe),
                    "track_ok": bool(item.track_ok),
                    "keypoint_count": int(item.keypoint_count),
                    "match_count": int(item.match_count),
                    "inlier_count": int(item.inlier_count),
                    "pixel_motion": float(item.pixel_motion),
                }
            )
            submitted += 1
            process_new_results()
            if args.publish_every_frames > 0 and submitted % args.publish_every_frames == 0:
                write_started = time.perf_counter()
                _write_live_json(output_dir, frame_points, trajectory, events, args)
                publish_ros_cloud()
                metrics["live_write_sec"].append(time.perf_counter() - write_started)
            if args.frame_sleep_sec > 0:
                time.sleep(args.frame_sleep_sec)

        worker.stop(drain=True)
        process_new_results()
    finally:
        if worker.status().get("stop_requested") is False:
            worker.stop(drain=False)

    _refresh_fusion_snapshot(fusion_map, frame_points, int(processed_results and len(processed_results) or 0))
    payload = _write_live_json(output_dir, frame_points, trajectory, events, args)
    publish_ros_cloud()
    final_worker_status = worker.status()
    summary = {
        "output_dir": str(output_dir),
        "tracking_backend": str(args.tracking_backend),
        "rgb_image_dir": str(args.rgb_image_dir),
        "viewer_html": str(output_dir / "live_viewer.html"),
        "live_map_json": str(output_dir / "live_map.json"),
        "live_map_ply": str(output_dir / "live_map.ply"),
        "tracked_frames": len(metrics["tracking"]),
        "submitted_frames": submitted,
        "skipped_frames": skipped,
        "dense_policy": {
            "keyframes_only": bool(args.keyframes_only),
            "dense_frame_interval": int(args.dense_frame_interval),
            "window_stride": int(args.stride),
            "max_queue": int(args.max_queue),
            "drop_when_busy": bool(args.no_blocking_submit),
        },
        "reconstruction": {
            "fusion_mode": str(args.fusion_mode),
            "voxel_size": float(args.voxel_size),
            "fusion_max_points": int(args.fusion_max_points),
            "fusion_min_observations": int(args.fusion_min_observations),
            "adaptive_sampling": bool(args.adaptive_sampling),
            "lingbot_extrinsic_mode": str(args.lingbot_extrinsic_mode),
            "sample_stride": int(args.sample_stride),
            "sampling_pattern": str(args.sampling_pattern),
            "near_sample_stride": int(args.near_sample_stride),
            "edge_sample_stride": int(args.edge_sample_stride),
            "semantic_sample_stride": int(args.semantic_sample_stride),
            "yolo_model": str(args.yolo_model),
            "yolo_enabled": bool(semantic_projector.enabled),
            "semantic_color_output": bool(args.semantic_color_output),
        },
        "dense_candidates": dense_candidates,
        "dense_submitted_frames": dense_submitted_frames,
        "dense_interval_skips": dense_interval_skips,
        "dense_non_keyframe_skips": dense_non_keyframe_skips,
        "dense_window_schedules": dense_window_schedules,
        "dense_queue_full_drops": int(final_worker_status.get("queue_full_drops", 0)),
        "worker_status": final_worker_status,
        "processed_windows": len(processed_results),
        "point_count": int(payload["point_count"]),
        "elapsed_sec": time.perf_counter() - start,
        "server_url": f"http://0.0.0.0:{args.port}/live_viewer.html" if args.serve else "",
        "server_pid": int(server.pid) if server is not None else 0,
    }
    metrics_summary = {
        "track_next_sec": _stats(metrics["track_next_sec"]),
        "submit_sec": _stats(metrics["submit_sec"]),
        "worker_elapsed_sec": _stats(metrics["worker_elapsed_sec"]),
        "worker_queue_wait_sec": _stats(metrics["worker_queue_wait_sec"]),
        "worker_end_to_end_sec": _stats(metrics["worker_end_to_end_sec"]),
        "process_result_sec": _stats(metrics["process_result_sec"]),
        "live_write_sec": _stats(metrics["live_write_sec"]),
        "points_added_or_replaced": _stats([float(v) for v in metrics["points_added_or_replaced"]]),
        "dense_submit_sec": _stats([float(item["submit_sec"]) for item in metrics["dense_submit"]]),
    }
    metrics_payload = {
        "summary": summary,
        "latency": metrics_summary,
        "tracking": metrics["tracking"],
        "dense_submit": metrics["dense_submit"],
        "windows": metrics["windows"],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "full_stack_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if ros_bridge is not None:
        ros_bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
