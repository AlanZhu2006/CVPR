#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a scripted Isaac Lab scene and publish pose/RGB-D/semantics to the online Gaussian bridge."
    )
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8890", help="Bridge base URL.")
    parser.add_argument(
        "--scene",
        default="full-warehouse",
        choices=["full-warehouse", "warehouse", "office", "hospital", "empty"],
        help="Environment preset.",
    )
    parser.add_argument(
        "--scene-usd",
        default="",
        help="Optional USD scene to load instead of the built-in scene preset.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=None,
        help="Override robot base height in world coordinates. Useful for reconstructed scenes with nonzero floor Z.",
    )
    parser.add_argument("--publish-rate-hz", type=float, default=6.0, help="Bridge publish rate.")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="Simulation time-step.")
    parser.add_argument("--path-radius", type=float, default=4.5, help="Scripted circle radius in meters.")
    parser.add_argument("--path-speed", type=float, default=1.0, help="Scripted path speed in meters/sec.")
    parser.add_argument("--path-center-x", type=float, default=0.0)
    parser.add_argument("--path-center-y", type=float, default=0.0)
    parser.add_argument(
        "--control-source",
        choices=("scripted", "ros2-cmd_vel"),
        default="scripted",
        help="Use the old scripted path or follow ROS2 /cmd_vel from the Nav2 baseline.",
    )
    parser.add_argument("--ros-cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--ros-cmd-timeout-sec", type=float, default=0.8)
    parser.add_argument("--ros-odom-topic", default="/odom")
    parser.add_argument(
        "--ros-follow-odom",
        action="store_true",
        help="In ros2-cmd_vel mode, lock the Isaac visual robot root to Nav2 odom so it follows the collision-checked navigation pose.",
    )
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=360)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    parser.add_argument("--semantic-jpeg-quality", type=int, default=84)
    parser.add_argument("--depth-max-m", type=float, default=15.0)
    parser.add_argument("--cloud-frame-max-points", type=int, default=4500)
    parser.add_argument("--cloud-accumulated-max-points", type=int, default=18000)
    parser.add_argument("--cloud-voxel-size", type=float, default=0.16)
    parser.add_argument("--cloud-min-depth-m", type=float, default=0.25)
    parser.add_argument("--cloud-max-depth-m", type=float, default=12.0)
    parser.add_argument("--cloud-world-z-min", type=float, default=-0.25)
    parser.add_argument("--cloud-world-z-max", type=float, default=2.5)
    parser.add_argument("--hold-open-sec", type=float, default=0.0)
    parser.add_argument(
        "--training-trajectory-output-dir",
        default="",
        help=(
            "Optional GS-SDF output dir. When set, robotPose/renderPose are "
            "published from the training camera trajectory instead of the scripted Isaac circle."
        ),
    )
    parser.add_argument(
        "--training-trajectory-split",
        choices=("train", "raw"),
        default="train",
        help="Camera split to replay when --training-trajectory-output-dir is set.",
    )
    parser.add_argument(
        "--training-rgb-source",
        choices=("none", "trajectory"),
        default="none",
        help="Use original training JPEG frames as bridge RGB so RGB and GS compare on the same video trajectory.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS_CLI = build_argparser().parse_args()
if hasattr(ARGS_CLI, "enable_cameras"):
    ARGS_CLI.enable_cameras = True
APP_LAUNCHER = AppLauncher(ARGS_CLI)
simulation_app = APP_LAUNCHER.app


import cv2
import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from gssdf_training_camera_pose import build_pose as build_training_camera_pose
from gssdf_training_camera_pose import training_rows


SCENE_PRESETS = {
    "full-warehouse": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
        "base_height": 0.40,
        "viewer_eye": (12.0, 8.0, 6.0),
        "viewer_target": (0.0, 0.0, 0.8),
    },
    "warehouse": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        "base_height": 0.40,
        "viewer_eye": (9.0, 6.0, 5.0),
        "viewer_target": (0.0, 0.0, 0.8),
    },
    "office": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
        "base_height": 0.42,
        "viewer_eye": (8.0, 5.0, 4.0),
        "viewer_target": (0.0, 0.0, 0.8),
    },
    "hospital": {
        "usd_path": f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
        "base_height": 0.42,
        "viewer_eye": (8.0, 5.0, 4.0),
        "viewer_target": (0.0, 0.0, 0.8),
    },
    "empty": {
        "usd_path": None,
        "base_height": 0.40,
        "viewer_eye": (6.0, 4.0, 4.0),
        "viewer_target": (0.0, 0.0, 0.8),
    },
}


@dataclass
class AgentControlState:
    default_joint_pos: torch.Tensor
    default_joint_vel: torch.Tensor
    leg_joint_ids: dict[str, int]
    gait_phase: float = 0.0


@dataclass(frozen=True)
class TrainingTrajectoryFrame:
    pose: dict[str, Any]
    source_image: str
    rgb_path: Path | None


@dataclass
class TrainingTrajectory:
    frames: list[TrainingTrajectoryFrame]

    def get(self, frame_index: int) -> TrainingTrajectoryFrame:
        return self.frames[frame_index % len(self.frames)]


class LivePointCloudAccumulator:
    def __init__(self, voxel_size: float, max_points: int):
        self.voxel_size = max(float(voxel_size), 1.0e-3)
        self.max_points = max(int(max_points), 1)
        self._voxels: OrderedDict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
        self._source_point_count = 0

    def ingest(self, positions_world: np.ndarray, colors_rgb_u8: np.ndarray) -> None:
        if positions_world.size == 0:
            return

        self._source_point_count += int(positions_world.shape[0])
        inv_voxel = 1.0 / self.voxel_size

        for point, color in zip(positions_world, colors_rgb_u8, strict=False):
            key = tuple(np.floor(point * inv_voxel).astype(np.int32).tolist())
            previous = self._voxels.pop(key, None)
            if previous is None:
                mean_point = point.astype(np.float32, copy=True)
                mean_color = color.astype(np.float32, copy=True)
                sample_count = 1
            else:
                previous_point, previous_color, previous_count = previous
                sample_count = previous_count + 1
                blend = 1.0 / float(sample_count)
                mean_point = previous_point + (point.astype(np.float32) - previous_point) * blend
                mean_color = previous_color + (color.astype(np.float32) - previous_color) * blend

            self._voxels[key] = (mean_point, mean_color, sample_count)
            if len(self._voxels) > self.max_points:
                self._voxels.popitem(last=False)

    def build_payload(self, stamp_ms: int, frame_id: str = "world") -> dict[str, Any] | None:
        if not self._voxels:
            return None

        point_count = len(self._voxels)
        positions = np.empty(point_count * 3, dtype=np.float32)
        colors = np.empty(point_count * 3, dtype=np.int32)

        for index, (position, color, _sample_count) in enumerate(self._voxels.values()):
            offset = index * 3
            positions[offset : offset + 3] = position
            colors[offset : offset + 3] = np.clip(np.round(color), 0, 255).astype(np.int32)

        return {
            "frameId": frame_id,
            "stampMs": int(stamp_ms),
            "sourcePointCount": int(max(self._source_point_count, point_count)),
            "renderedPointCount": int(point_count),
            "positions": np.round(positions, 4).astype(float).tolist(),
            "colors": colors.astype(int).tolist(),
        }


class BridgePublisher:
    def __init__(self, bridge_url: str, timeout_sec: float = 1.5):
        self.bridge_url = bridge_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._last_error_wall_sec = 0.0

    def publish(self, payload: dict[str, Any]) -> None:
        request = Request(
            f"{self.bridge_url}/ingest/state",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_sec) as response:
                response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            now = time.time()
            if now - self._last_error_wall_sec > 2.0:
                print(f"[WARN] Failed to publish bridge state: {exc}")
                self._last_error_wall_sec = now


class Ros2CmdVelFollower(Node):
    def __init__(self, topic_name: str, odom_topic_name: str | None = None):
        super().__init__("isaac_nav2_cmd_vel_follower")
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.last_command_wall_sec = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.last_odom_wall_sec = 0.0
        self.create_subscription(Twist, topic_name, self._on_cmd_vel, 10)
        if odom_topic_name:
            self.create_subscription(Odometry, odom_topic_name, self._on_odom, 10)

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.linear_x = float(msg.linear.x)
        self.angular_z = float(msg.angular.z)
        self.last_command_wall_sec = time.time()

    def _on_odom(self, msg: Odometry) -> None:
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        orientation = msg.pose.pose.orientation
        quat_wxyz = np.asarray(
            [float(orientation.w), float(orientation.x), float(orientation.y), float(orientation.z)],
            dtype=np.float32,
        )
        self.odom_yaw = quat_wxyz_to_yaw(quat_wxyz)
        self.last_odom_wall_sec = time.time()

    def command(self, timeout_sec: float) -> tuple[float, float]:
        if time.time() - self.last_command_wall_sec > max(float(timeout_sec), 0.05):
            return 0.0, 0.0
        return self.linear_x, self.angular_z

    def odom_pose(self, timeout_sec: float) -> tuple[float, float, float] | None:
        if time.time() - self.last_odom_wall_sec > max(float(timeout_sec), 0.05):
            return None
        return self.odom_x, self.odom_y, self.odom_yaw


@configclass
class IsaacGaussianOnlineSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Environment")
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.0,
        height=360,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e4),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.32, 0.0, 0.08), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        update_latest_camera_pose=True,
        semantic_filter=["class"],
        colorize_semantic_segmentation=True,
    )


def build_scene_cfg(args: argparse.Namespace) -> tuple[IsaacGaussianOnlineSceneCfg, dict[str, Any]]:
    preset = dict(SCENE_PRESETS[args.scene])
    scene_cfg = IsaacGaussianOnlineSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False, lazy_sensor_update=False)

    scene_usd_path = str(args.scene_usd).strip() or preset["usd_path"]
    if scene_usd_path and check_file_path(scene_usd_path) != 0:
        scene_cfg.environment = scene_cfg.environment.replace(
            prim_path="/World/Environment",
            spawn=sim_utils.UsdFileCfg(usd_path=scene_usd_path, semantic_tags=[("class", "environment")]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        preset["resolved_usd_path"] = scene_usd_path
        if str(args.scene_usd).strip():
            preset["viewer_eye"] = (
                float(args.path_center_x) + 6.0,
                float(args.path_center_y) + 6.0,
                float(args.base_height if args.base_height is not None else preset["base_height"]) + 4.0,
            )
            preset["viewer_target"] = (
                float(args.path_center_x),
                float(args.path_center_y),
                float(args.base_height if args.base_height is not None else preset["base_height"]),
            )
    else:
        if scene_usd_path:
            print(f"[WARN] Isaac environment asset not available: {scene_usd_path}. Falling back to ground plane.")
        scene_cfg.environment = scene_cfg.environment.replace(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(color=(0.28, 0.28, 0.28), semantic_tags=[("class", "ground")]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )
        preset["resolved_usd_path"] = None
    if args.base_height is not None:
        preset["base_height"] = float(args.base_height)

    robot_cfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_cfg = robot_cfg.replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(preset["base_height"])),
            joint_pos=UNITREE_GO2_CFG.init_state.joint_pos,
            joint_vel=UNITREE_GO2_CFG.init_state.joint_vel,
        ),
        spawn=robot_cfg.spawn.replace(semantic_tags=[("class", "robot")]),
    )
    scene_cfg.robot = robot_cfg
    scene_cfg.camera = scene_cfg.camera.replace(height=int(args.camera_height), width=int(args.camera_width))
    return scene_cfg, preset


def initialize_go2_control_state(robot) -> AgentControlState:
    leg_joint_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    joint_ids, joint_names = robot.find_joints(leg_joint_names, preserve_order=True)
    if len(joint_ids) != len(leg_joint_names):
        raise RuntimeError(f"Could not resolve all Go2 leg joints. Found: {joint_names}")
    return AgentControlState(
        default_joint_pos=robot.data.default_joint_pos.clone(),
        default_joint_vel=robot.data.default_joint_vel.clone(),
        leg_joint_ids={joint_name: int(joint_id) for joint_name, joint_id in zip(joint_names, joint_ids)},
    )


def build_go2_joint_targets(
    control_state: AgentControlState,
    *,
    commanded_speed: float,
    max_speed: float,
    yaw_rate: float,
    yaw_rate_limit: float,
    dt: float,
) -> torch.Tensor:
    target = control_state.default_joint_pos.clone()
    motion_ratio = max(
        min(float(commanded_speed) / max(float(max_speed), 1e-6), 1.0),
        min(abs(float(yaw_rate)) / max(float(yaw_rate_limit), 1e-6), 1.0) * 0.65,
    )
    if motion_ratio < 0.05:
        control_state.gait_phase = 0.0
        return target

    cycle_hz = 1.8 * max(0.4, motion_ratio)
    control_state.gait_phase = float((control_state.gait_phase + dt * cycle_hz * 2.0 * math.pi) % (2.0 * math.pi))
    stride_scale = 0.36 * max(0.35, motion_ratio)
    turn_ratio = float(np.clip(yaw_rate / max(float(yaw_rate_limit), 1e-6), -1.0, 1.0))
    leg_phase_offsets = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}

    for leg_name, phase_offset in leg_phase_offsets.items():
        phase = control_state.gait_phase + phase_offset
        swing = math.sin(phase)
        lift = max(0.0, swing)
        lateral_sign = 1.0 if leg_name.endswith("L") else -1.0
        hip_delta = lateral_sign * (0.03 * swing + 0.08 * turn_ratio)
        thigh_delta = stride_scale * (0.26 * swing - 0.10 * lift)
        calf_delta = stride_scale * (-0.48 * swing + 0.22 * lift)

        hip_id = control_state.leg_joint_ids[f"{leg_name}_hip_joint"]
        thigh_id = control_state.leg_joint_ids[f"{leg_name}_thigh_joint"]
        calf_id = control_state.leg_joint_ids[f"{leg_name}_calf_joint"]
        target[0, hip_id] = control_state.default_joint_pos[0, hip_id] + hip_delta
        target[0, thigh_id] = control_state.default_joint_pos[0, thigh_id] + thigh_delta
        target[0, calf_id] = control_state.default_joint_pos[0, calf_id] + calf_delta

    return target


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def compute_scripted_root_pose(
    sim_time: float,
    *,
    radius: float,
    speed: float,
    center_x: float,
    center_y: float,
    base_height: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    radius = max(float(radius), 0.2)
    speed = max(float(speed), 0.05)
    omega = speed / radius
    theta = omega * sim_time
    x = center_x + radius * math.cos(theta)
    y = center_y + radius * math.sin(theta)
    vx = -radius * omega * math.sin(theta)
    vy = radius * omega * math.cos(theta)
    yaw = math.atan2(vy, vx)
    quat = np.asarray(yaw_to_quat_wxyz(yaw), dtype=np.float32)
    pose = np.asarray([x, y, base_height, quat[0], quat[1], quat[2], quat[3]], dtype=np.float32)
    velocity = np.asarray([vx, vy, 0.0, 0.0, 0.0, omega], dtype=np.float32)
    return pose, velocity, yaw


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = [float(value) for value in quat_wxyz[:4]]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def fetch_bridge_robot_pose(bridge_url: str, timeout_sec: float = 1.0) -> tuple[float, float, float] | None:
    try:
        with urlopen(f"{bridge_url.rstrip('/')}/status", timeout=timeout_sec) as response:
            status = json.loads(response.read().decode("utf-8", errors="ignore") or "{}")
    except Exception:
        return None
    nav_pose = status.get("navStatus", {}).get("pose") if isinstance(status.get("navStatus"), dict) else None
    if isinstance(nav_pose, dict):
        try:
            return float(nav_pose["x"]), float(nav_pose["y"]), float(nav_pose.get("yaw", 0.0))
        except Exception:
            pass
    pose = status.get("robotPose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    try:
        quat_xyzw = np.asarray(
            [
                float(orientation.get("x", 0.0)),
                float(orientation.get("y", 0.0)),
                float(orientation.get("z", 0.0)),
                float(orientation.get("w", 1.0)),
            ],
            dtype=np.float32,
        )
        quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        return float(position["x"]), float(position["y"]), quat_wxyz_to_yaw(quat_wxyz)
    except Exception:
        return None


def _torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().numpy()
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def encode_rgb_jpeg(image: torch.Tensor, jpeg_quality: int) -> str | None:
    array = _torch_image_to_numpy(image)
    if array.ndim != 3:
        return None
    if array.shape[-1] == 4:
        bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(array[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


def encode_semantic_jpeg(image: torch.Tensor, jpeg_quality: int) -> str | None:
    return encode_rgb_jpeg(image, jpeg_quality)


def encode_depth_preview_jpeg(depth: torch.Tensor, jpeg_quality: int, depth_max_m: float) -> str | None:
    array = _torch_image_to_numpy(depth).astype(np.float32)
    if array.ndim != 2:
        return None
    valid = np.isfinite(array)
    if not np.any(valid):
        return None
    depth_max_m = max(float(depth_max_m), 0.5)
    clipped = np.clip(np.where(valid, array, depth_max_m), 0.0, depth_max_m)
    normalized = 1.0 - clipped / depth_max_m
    image_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    ok, encoded = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    return base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None


def pose_payload_from_wxyz(position: np.ndarray, quat_wxyz: np.ndarray, *, frame_id: str, stamp_ms: int) -> dict[str, Any]:
    return {
        "frameId": frame_id,
        "position": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "orientation": {
            "x": float(quat_wxyz[1]),
            "y": float(quat_wxyz[2]),
            "z": float(quat_wxyz[3]),
            "w": float(quat_wxyz[0]),
        },
        "stampMs": int(stamp_ms),
    }


def pose_payload_from_xyzw(pose: dict[str, Any], *, stamp_ms: int) -> dict[str, Any]:
    return {
        "frameId": str(pose.get("frameId") or "world"),
        "position": {
            "x": float(pose["position"]["x"]),
            "y": float(pose["position"]["y"]),
            "z": float(pose["position"]["z"]),
        },
        "orientation": {
            "x": float(pose["orientation"]["x"]),
            "y": float(pose["orientation"]["y"]),
            "z": float(pose["orientation"]["z"]),
            "w": float(pose["orientation"]["w"]),
        },
        "stampMs": int(stamp_ms),
    }


def load_training_trajectory(args: argparse.Namespace) -> TrainingTrajectory | None:
    output_dir = str(args.training_trajectory_output_dir or "").strip()
    if not output_dir:
        return None

    config_path = Path(output_dir) / "model/config/scene/config.yaml"
    if not config_path.is_file():
        raise RuntimeError(f"Training trajectory scene config not found: {config_path}")

    text = config_path.read_text()
    data_path = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("data_path:"):
            data_path = Path(stripped.split(":", 1)[1].strip().strip('"'))
            break
    if data_path is None or not data_path.is_dir():
        raise RuntimeError(f"Training trajectory data_path not found in {config_path}: {data_path}")

    images_path, rows = training_rows(data_path, str(args.training_trajectory_split))
    frames: list[TrainingTrajectoryFrame] = []
    for row in rows:
        built = build_training_camera_pose(row, frame_id="world", target_distance=4.0)
        rgb_path = data_path / "images" / str(built["sourceImage"])
        frames.append(
            TrainingTrajectoryFrame(
                pose=built["pose"],
                source_image=str(built["sourceImage"]),
                rgb_path=rgb_path if rgb_path.is_file() else None,
            )
        )

    if not frames:
        raise RuntimeError(f"No training trajectory frames found in {images_path}")
    print(f"[INFO] Loaded {len(frames)} {args.training_trajectory_split} training camera poses from {images_path}")
    return TrainingTrajectory(frames)


def encode_file_jpeg_base64(path: Path) -> str | None:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def build_live_point_cloud_payload(
    accumulator: LivePointCloudAccumulator,
    camera,
    rgb_image: torch.Tensor,
    depth_image: torch.Tensor,
    args: argparse.Namespace,
    stamp_ms: int,
) -> dict[str, Any] | None:
    if int(args.cloud_frame_max_points) <= 0:
        return None

    intrinsics = camera.data.intrinsic_matrices[0]
    depth_plane = depth_image
    if depth_plane.ndim == 3 and depth_plane.shape[-1] == 1:
        depth_plane = depth_plane[..., 0]

    points_camera = math_utils.unproject_depth(depth_plane, intrinsics, is_ortho=True)
    points_camera = points_camera.reshape(-1, 3)

    rgb_view = rgb_image[..., :3] if rgb_image.ndim == 3 else rgb_image
    colors_rgb = rgb_view.reshape(-1, 3)

    valid = torch.isfinite(points_camera).all(dim=-1)
    valid &= torch.isfinite(points_camera[:, 2])
    valid &= points_camera[:, 2] >= float(args.cloud_min_depth_m)
    valid &= points_camera[:, 2] <= float(args.cloud_max_depth_m)

    if not torch.any(valid):
        return accumulator.build_payload(stamp_ms)

    points_camera = points_camera[valid]
    colors_rgb = colors_rgb[valid]

    frame_limit = max(int(args.cloud_frame_max_points), 1)
    if points_camera.shape[0] > frame_limit:
        stride = max(1, int(math.ceil(float(points_camera.shape[0]) / float(frame_limit))))
        points_camera = points_camera[::stride]
        colors_rgb = colors_rgb[::stride]

    camera_quat_ros = camera.data.quat_w_ros[0].to(device=points_camera.device, dtype=points_camera.dtype)
    camera_pos_world = camera.data.pos_w[0].to(device=points_camera.device, dtype=points_camera.dtype)
    world_points = math_utils.quat_apply(
        camera_quat_ros.unsqueeze(0).expand(points_camera.shape[0], -1),
        points_camera,
    )
    world_points = world_points + camera_pos_world.unsqueeze(0)

    valid_world = torch.isfinite(world_points).all(dim=-1)
    valid_world &= world_points[:, 2] >= float(args.cloud_world_z_min)
    valid_world &= world_points[:, 2] <= float(args.cloud_world_z_max)

    if not torch.any(valid_world):
        return accumulator.build_payload(stamp_ms)

    world_points = world_points[valid_world].detach().cpu().to(dtype=torch.float32).numpy()
    colors_rgb = colors_rgb[valid_world].detach().cpu().to(dtype=torch.uint8).numpy()

    accumulator.ingest(world_points, colors_rgb)
    return accumulator.build_payload(stamp_ms)


def publish_scene_step(
    publisher: BridgePublisher,
    cloud_accumulator: LivePointCloudAccumulator,
    robot,
    camera,
    args: argparse.Namespace,
    training_frame: TrainingTrajectoryFrame | None = None,
    reset_trajectory: bool = False,
) -> None:
    rgb = camera.data.output.get("rgb")
    depth = camera.data.output.get("distance_to_image_plane")
    semantic = camera.data.output.get("semantic_segmentation")
    if rgb is None or depth is None:
        return

    stamp_ms = int(time.time() * 1000.0)
    robot_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
    robot_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
    camera_pos = camera.data.pos_w[0].detach().cpu().numpy()
    camera_quat = camera.data.quat_w_world[0].detach().cpu().numpy()
    robot_pose_payload = pose_payload_from_wxyz(robot_pos, robot_quat, frame_id="world", stamp_ms=stamp_ms)
    render_pose_payload = pose_payload_from_wxyz(camera_pos, camera_quat, frame_id="world", stamp_ms=stamp_ms)
    rgb_jpeg_base64 = encode_rgb_jpeg(rgb[0], int(args.jpeg_quality))
    frame_width = int(args.camera_width)
    frame_height = int(args.camera_height)

    if training_frame is not None:
        training_pose_payload = pose_payload_from_xyzw(training_frame.pose, stamp_ms=stamp_ms)
        robot_pose_payload = training_pose_payload
        render_pose_payload = training_pose_payload
        if str(args.training_rgb_source) == "trajectory" and training_frame.rgb_path is not None:
            rgb_jpeg_base64 = encode_file_jpeg_base64(training_frame.rgb_path) or rgb_jpeg_base64
            image = cv2.imread(str(training_frame.rgb_path), cv2.IMREAD_COLOR)
            if image is not None:
                frame_height, frame_width = image.shape[:2]

    payload = {
        "resetTrajectory": bool(reset_trajectory),
        "pose": robot_pose_payload,
        "renderPose": render_pose_payload,
        "frameWidth": frame_width,
        "frameHeight": frame_height,
        "rgbJpegBase64": rgb_jpeg_base64,
        "depthJpegBase64": encode_depth_preview_jpeg(depth[0], int(args.jpeg_quality), float(args.depth_max_m)),
        "semanticJpegBase64": encode_semantic_jpeg(semantic[0], int(args.semantic_jpeg_quality)) if semantic is not None else None,
        "sourceImage": training_frame.source_image if training_frame is not None else None,
        "livePointCloud": build_live_point_cloud_payload(
            cloud_accumulator,
            camera,
            rgb[0],
            depth[0],
            args,
            stamp_ms,
        ),
    }
    publisher.publish(payload)


def main() -> None:
    scene_cfg, preset = build_scene_cfg(ARGS_CLI)
    publish_interval_steps = max(1, int(round((1.0 / max(float(ARGS_CLI.publish_rate_hz), 1e-3)) / float(ARGS_CLI.dt))))
    publisher = BridgePublisher(str(ARGS_CLI.bridge_url))
    training_trajectory = load_training_trajectory(ARGS_CLI)
    ros_follower: Ros2CmdVelFollower | None = None
    if str(ARGS_CLI.control_source) == "ros2-cmd_vel":
        if not rclpy.ok():
            rclpy.init(args=None)
        ros_follower = Ros2CmdVelFollower(
            str(ARGS_CLI.ros_cmd_vel_topic),
            str(ARGS_CLI.ros_odom_topic) if bool(ARGS_CLI.ros_follow_odom) else None,
        )
        print(f"[INFO] Isaac robot will follow ROS2 Twist commands from {ARGS_CLI.ros_cmd_vel_topic}.")
        if bool(ARGS_CLI.ros_follow_odom):
            print(f"[INFO] Isaac robot root will be locked to ROS2 odom from {ARGS_CLI.ros_odom_topic}.")
    cloud_accumulator = LivePointCloudAccumulator(
        voxel_size=float(ARGS_CLI.cloud_voxel_size),
        max_points=int(ARGS_CLI.cloud_accumulated_max_points),
    )

    try:
        sim_cfg = sim_utils.SimulationCfg(dt=float(ARGS_CLI.dt), device=ARGS_CLI.device)
        with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False, add_lighting=False) as sim:
            sim._app_control_on_stop_handle = None
            sim.set_camera_view(eye=list(preset["viewer_eye"]), target=list(preset["viewer_target"]))
            scene = InteractiveScene(scene_cfg)
            sim.reset()
            scene.reset()

            robot = scene["robot"]
            camera = scene["camera"]
            control_state = initialize_go2_control_state(robot)
            robot.write_joint_state_to_sim(control_state.default_joint_pos, control_state.default_joint_vel)
            robot.set_joint_position_target(control_state.default_joint_pos)
            scene.reset()

            if ros_follower is not None:
                start_pose = fetch_bridge_robot_pose(str(ARGS_CLI.bridge_url))
                start_x, start_y, start_yaw = start_pose or (
                    float(ARGS_CLI.path_center_x),
                    float(ARGS_CLI.path_center_y),
                    0.0,
                )
                start_root = np.asarray(
                    [
                        start_x,
                        start_y,
                        float(preset["base_height"]),
                        math.cos(start_yaw * 0.5),
                        0.0,
                        0.0,
                        math.sin(start_yaw * 0.5),
                    ],
                    dtype=np.float32,
                )
                robot.write_root_pose_to_sim(torch.tensor(start_root.reshape(1, 7), dtype=torch.float32, device=scene.device))
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=scene.device))
                sim.set_camera_view(
                    eye=[start_x + 4.0, start_y + 4.0, float(preset["base_height"]) + 3.0],
                    target=[start_x, start_y, float(preset["base_height"])],
                )
                print(f"[INFO] Nav2 Isaac initial pose: x={start_x:.3f}, y={start_y:.3f}, yaw={start_yaw:.3f}")

            sim_time = 0.0
            step_count = 0
            publish_count = 0
            print("[INFO] Isaac Gaussian online demo ready.")
            print(f"[INFO] Scene: {ARGS_CLI.scene} ({preset['resolved_usd_path'] or 'ground plane fallback'})")
            print(f"[INFO] Bridge: {ARGS_CLI.bridge_url}")
            if training_trajectory is not None:
                print("[INFO] Publishing bridge poses from the GS-SDF training camera trajectory.")

            while simulation_app.is_running():
                if ros_follower is not None:
                    rclpy.spin_once(ros_follower, timeout_sec=0.0)
                    linear_x, angular_z = ros_follower.command(float(ARGS_CLI.ros_cmd_timeout_sec))
                    odom_pose = (
                        ros_follower.odom_pose(float(ARGS_CLI.ros_cmd_timeout_sec))
                        if bool(ARGS_CLI.ros_follow_odom)
                        else None
                    )
                    if odom_pose is not None:
                        odom_x, odom_y, odom_yaw = odom_pose
                        odom_root = np.asarray(
                            [
                                odom_x,
                                odom_y,
                                float(preset["base_height"]),
                                math.cos(odom_yaw * 0.5),
                                0.0,
                                0.0,
                                math.sin(odom_yaw * 0.5),
                            ],
                            dtype=np.float32,
                        )
                        robot.write_root_pose_to_sim(
                            torch.tensor(odom_root.reshape(1, 7), dtype=torch.float32, device=scene.device)
                        )
                    root_quat = robot.data.root_quat_w[0].detach().cpu().numpy()
                    yaw = quat_wxyz_to_yaw(root_quat)
                    velocity = np.asarray(
                        [
                            linear_x * math.cos(yaw),
                            linear_x * math.sin(yaw),
                            0.0,
                            0.0,
                            0.0,
                            angular_z,
                        ],
                        dtype=np.float32,
                    )
                    robot.write_root_velocity_to_sim(torch.tensor(velocity.reshape(1, 6), dtype=torch.float32, device=scene.device))
                    joint_target = build_go2_joint_targets(
                        control_state,
                        commanded_speed=abs(float(linear_x)),
                        max_speed=max(float(ARGS_CLI.path_speed), 1e-3),
                        yaw_rate=float(angular_z),
                        yaw_rate_limit=max(abs(float(angular_z)), 0.75),
                        dt=float(ARGS_CLI.dt),
                    )
                else:
                    pose, velocity, _ = compute_scripted_root_pose(
                        sim_time,
                        radius=float(ARGS_CLI.path_radius),
                        speed=float(ARGS_CLI.path_speed),
                        center_x=float(ARGS_CLI.path_center_x),
                        center_y=float(ARGS_CLI.path_center_y),
                        base_height=float(preset["base_height"]),
                    )
                    pose_tensor = torch.tensor(pose.reshape(1, 7), dtype=torch.float32, device=scene.device)
                    velocity_tensor = torch.tensor(velocity.reshape(1, 6), dtype=torch.float32, device=scene.device)
                    robot.write_root_pose_to_sim(pose_tensor)
                    robot.write_root_velocity_to_sim(velocity_tensor)
                    joint_target = build_go2_joint_targets(
                        control_state,
                        commanded_speed=float(np.linalg.norm(velocity[:2])),
                        max_speed=max(float(ARGS_CLI.path_speed), 1e-3),
                        yaw_rate=float(velocity[5]),
                        yaw_rate_limit=max(abs(float(velocity[5])), 0.75),
                        dt=float(ARGS_CLI.dt),
                    )
                robot.set_joint_position_target(joint_target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(float(ARGS_CLI.dt))

                if step_count % publish_interval_steps == 0:
                    training_frame = training_trajectory.get(publish_count) if training_trajectory is not None else None
                    publish_scene_step(
                        publisher,
                        cloud_accumulator,
                        robot,
                        camera,
                        ARGS_CLI,
                        training_frame,
                        reset_trajectory=publish_count == 0,
                    )
                    publish_count += 1

                sim_time += float(ARGS_CLI.dt)
                step_count += 1

            if float(ARGS_CLI.hold_open_sec) > 0.0:
                hold_until = time.time() + float(ARGS_CLI.hold_open_sec)
                while time.time() < hold_until and simulation_app.is_running():
                    sim.step()
                    scene.update(float(ARGS_CLI.dt))
    finally:
        if ros_follower is not None:
            ros_follower.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
    simulation_app.close()
