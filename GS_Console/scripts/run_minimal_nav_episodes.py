#!/usr/bin/env python3
"""Run a minimal navigation episode loop for the validated Isaac scene.

This script intentionally stays thin:

1. Reuse the collision USD and settled start pose from the validation report.
2. Reuse the 20 sampled goals from the same report.
3. Spawn either a simple capsule proxy or a quadruped asset.
4. Drive it with a simple goal-seeking planar controller.
5. Emit one result per episode with categories:
   success / collision / timeout / stuck
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", required=True, help="Validation JSON report produced by validate_isaac_nav_scene.py.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--episode-limit", type=int, default=None, help="Optional cap on the number of goals to run.")
    parser.add_argument(
        "--goal-cell",
        default=None,
        help="Optional single point-goal map cell as 'x,y'. Overrides sampled validation goals.",
    )
    parser.add_argument(
        "--goal-world",
        default=None,
        help="Optional single point-goal world/map XY as 'x,y'. Overrides sampled validation goals.",
    )
    parser.add_argument(
        "--goal-label",
        default=None,
        help="Optional human-readable label for the selected goal.",
    )
    parser.add_argument(
        "--semantic-goal",
        default=None,
        help="Optional semantic goal label recorded in the report/status. The caller resolves it to a concrete goal.",
    )
    parser.add_argument("--timeout-sec", type=float, default=12.0, help="Per-episode timeout in seconds.")
    parser.add_argument(
        "--robot-model",
        choices=("unitree-go2", "capsule"),
        default="unitree-go2",
        help="Which Isaac asset to use for the navigation demo.",
    )
    parser.add_argument("--max-speed", type=float, default=1.25, help="Maximum commanded planar speed in m/s.")
    parser.add_argument("--goal-tolerance", type=float, default=0.6, help="XY goal tolerance in meters.")
    parser.add_argument(
        "--slowdown-radius",
        type=float,
        default=2.0,
        help="Distance to goal where the commanded speed begins to ramp down.",
    )
    parser.add_argument(
        "--stuck-window-sec",
        type=float,
        default=2.0,
        help="Sliding time window used to detect lack of progress.",
    )
    parser.add_argument(
        "--stuck-displacement",
        type=float,
        default=0.2,
        help="Minimum XY displacement over the stuck window before the agent is considered moving.",
    )
    parser.add_argument(
        "--command-threshold",
        type=float,
        default=0.2,
        help="Minimum commanded speed for stuck/collision checks.",
    )
    parser.add_argument(
        "--collision-lookahead",
        type=float,
        default=0.8,
        help="Distance in meters to probe forward in the occupancy map when classifying collisions.",
    )
    parser.add_argument(
        "--collision-tilt-rad",
        type=float,
        default=0.65,
        help="Roll or pitch threshold above which the episode is classified as collision.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=24,
        help="Number of zero-command settle steps after each episode reset.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.0,
        help="Optional wall-clock sleep in seconds after each simulation step for viewer mode.",
    )
    parser.add_argument(
        "--episode-hold-sec",
        type=float,
        default=0.0,
        help="Optional extra viewer hold time after each episode finishes.",
    )
    parser.add_argument(
        "--hold-open-sec",
        type=float,
        default=0.0,
        help="Keep the viewer open for this many seconds after the last episode.",
    )
    parser.add_argument(
        "--show-goals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show goal markers in viewer mode.",
    )
    parser.add_argument(
        "--show-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the planned A* path in viewer mode.",
    )
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the viewer camera close to the capsule and pointed along the active goal direction.",
    )
    parser.add_argument(
        "--camera-distance",
        type=float,
        default=2.2,
        help="Distance behind the capsule for the follow camera, in meters.",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.8,
        help="Height above the capsule for the follow camera, in meters.",
    )
    parser.add_argument(
        "--camera-lookahead",
        type=float,
        default=1.6,
        help="How far ahead of the capsule the camera aims, in meters.",
    )
    parser.add_argument(
        "--camera-side-offset",
        type=float,
        default=0.7,
        help="Lateral offset for the follow camera, in meters.",
    )
    parser.add_argument(
        "--camera-target-height",
        type=float,
        default=0.35,
        help="Vertical offset above the capsule center for the follow-camera target, in meters.",
    )
    parser.add_argument(
        "--goal-marker-z-offset",
        type=float,
        default=0.05,
        help="Extra height above the floor for goal markers, in meters.",
    )
    parser.add_argument(
        "--path-line-z-offset",
        type=float,
        default=0.08,
        help="Extra height above the floor for the path polyline, in meters.",
    )
    parser.add_argument(
        "--path-lookahead-distance",
        type=float,
        default=0.75,
        help="How far ahead along the A* path the local controller aims, in meters.",
    )
    parser.add_argument(
        "--waypoint-reach-distance",
        type=float,
        default=0.15,
        help="Distance threshold used to advance to the next path waypoint, in meters.",
    )
    parser.add_argument(
        "--avoidance-radius",
        type=float,
        default=0.7,
        help="Occupancy-map obstacle influence radius for local steering, in meters.",
    )
    parser.add_argument(
        "--avoidance-gain",
        type=float,
        default=1.1,
        help="Weight applied to the local obstacle-repulsion vector.",
    )
    parser.add_argument(
        "--avoidance-speed-scale",
        type=float,
        default=0.35,
        help="How much obstacle pressure reduces forward speed.",
    )
    parser.add_argument(
        "--floating-agent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a dynamic no-gravity capsule so it can interact with scene colliders in a 2D navigation plane.",
    )
    parser.add_argument(
        "--quadruped-base-height",
        type=float,
        default=0.40,
        help="Base height above the estimated floor used when spawning the Go2 articulation.",
    )
    parser.add_argument(
        "--yaw-rate-gain",
        type=float,
        default=2.5,
        help="Proportional gain used to align the agent heading with the active path target.",
    )
    parser.add_argument(
        "--yaw-rate-limit",
        type=float,
        default=1.8,
        help="Clamp on the commanded root yaw rate, in rad/s.",
    )
    parser.add_argument(
        "--gait-cycle-hz",
        type=float,
        default=1.8,
        help="Base open-loop gait frequency for the Go2 articulation.",
    )
    parser.add_argument(
        "--gait-stride-scale",
        type=float,
        default=0.55,
        help="Stride amplitude scale for the open-loop Go2 gait animation.",
    )
    parser.add_argument(
        "--recenter-start-to-origin",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Translate the imported environment so the validated start XY becomes (0, 0).",
    )
    parser.add_argument(
        "--bridge-url",
        default="",
        help="Optional Isaac Gaussian online bridge URL. When set, publishes robot pose, trajectory, and nav status.",
    )
    parser.add_argument(
        "--bridge-publish-rate-hz",
        type=float,
        default=8.0,
        help="Pose/status publish rate for --bridge-url.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS_CLI = build_argparser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS_CLI)
simulation_app = APP_LAUNCHER.app


import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass

try:
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
except ModuleNotFoundError:
    UNITREE_GO2_CFG = None


@dataclass
class OccupancyMap:
    image_path: Path
    yaml_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    image: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def free_mask(self) -> np.ndarray:
        return self.image == 254

    @property
    def occupied_mask(self) -> np.ndarray:
        return self.image == 0


@dataclass
class SupportGrid:
    floor_z: np.ndarray
    point_count: np.ndarray
    z_spread: np.ndarray

    def is_supported(self, cell: tuple[int, int]) -> bool:
        return bool(np.isfinite(self.floor_z[cell[1], cell[0]]))


@dataclass(frozen=True)
class ViewerState:
    scene_translation_xy: np.ndarray
    goal_positions_sim: list[np.ndarray] | None = None
    goal_cells: list[tuple[int, int]] | None = None
    goal_markers: VisualizationMarkers | None = None
    path_draw: object | None = None


@dataclass
class AgentControlState:
    model: str
    default_joint_pos: torch.Tensor | None = None
    default_joint_vel: torch.Tensor | None = None
    leg_joint_ids: dict[str, int] | None = None
    gait_phase: float = 0.0


class BridgePublisher:
    def __init__(self, bridge_url: str, timeout_sec: float = 0.75):
        self.bridge_url = bridge_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self._last_error_wall_sec = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.bridge_url)

    def publish(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
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
                print(f"[WARN] Failed to publish nav bridge state: {exc}", flush=True)
                self._last_error_wall_sec = now


@configclass
class NavEpisodeSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Environment")
    agent = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        spawn=sim_utils.CapsuleCfg(
            radius=0.25,
            height=0.50,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.4,
                max_linear_velocity=50.0,
                max_angular_velocity=3600.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        track_pose=True,
        track_air_time=True,
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )


GOAL_MARKER_UPCOMING = 0
GOAL_MARKER_CURRENT = 1
GOAL_MARKER_COMPLETED = 2
GOAL_MARKER_HEIGHTS = np.asarray([0.60, 1.00, 0.40], dtype=np.float32)


def build_goal_marker_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/NavGoals",
        markers={
            "upcoming": sim_utils.CylinderCfg(
                radius=0.10,
                height=float(GOAL_MARKER_HEIGHTS[GOAL_MARKER_UPCOMING]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.35, 0.1)),
            ),
            "current": sim_utils.CylinderCfg(
                radius=0.14,
                height=float(GOAL_MARKER_HEIGHTS[GOAL_MARKER_CURRENT]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 1.0, 0.2)),
            ),
            "completed": sim_utils.CylinderCfg(
                radius=0.08,
                height=float(GOAL_MARKER_HEIGHTS[GOAL_MARKER_COMPLETED]),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.55, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic {magic!r}; expected P5.")

        def next_token() -> bytes:
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF while reading PGM header: {path}")
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                return line

        dims = next_token().split()
        if len(dims) != 2:
            raise ValueError(f"Invalid PGM size header in {path}")
        width, height = int(dims[0]), int(dims[1])
        max_value = int(next_token())
        if max_value > 255:
            raise ValueError(f"Unsupported PGM max value {max_value}; expected <= 255.")
        data = np.frombuffer(handle.read(width * height), dtype=np.uint8)
        if data.size != width * height:
            raise ValueError(f"PGM payload size mismatch in {path}")
        return data.reshape((height, width))


def load_occupancy_map(yaml_path: Path) -> OccupancyMap:
    text = yaml_path.read_text(encoding="utf-8")
    image_match = re.search(r"^image:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    resolution_match = re.search(r"^resolution:\s*([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
    origin_match = re.search(r"^origin:\s*\[([^\]]+)\]\s*$", text, flags=re.MULTILINE)
    if image_match is None or resolution_match is None or origin_match is None:
        raise ValueError(f"Could not parse occupancy-map metadata from {yaml_path}")

    image_path = (yaml_path.parent / image_match.group(1)).resolve()
    origin_values = [float(piece.strip()) for piece in origin_match.group(1).split(",")]
    if len(origin_values) < 2:
        raise ValueError(f"Invalid origin in {yaml_path}")

    return OccupancyMap(
        image_path=image_path,
        yaml_path=yaml_path,
        resolution=float(resolution_match.group(1)),
        origin_x=origin_values[0],
        origin_y=origin_values[1],
        image=read_pgm(image_path),
    )


def cell_to_world(map_info: OccupancyMap, x: int, y: int) -> tuple[float, float]:
    world_x = map_info.origin_x + (x + 0.5) * map_info.resolution
    world_y = map_info.origin_y + ((map_info.height - 1 - y) + 0.5) * map_info.resolution
    return world_x, world_y


def world_to_cell(map_info: OccupancyMap, x: float, y: float) -> tuple[int, int]:
    cell_x = int(math.floor((x - map_info.origin_x) / map_info.resolution))
    world_row = int(math.floor((y - map_info.origin_y) / map_info.resolution))
    cell_y = (map_info.height - 1) - world_row
    return cell_x, cell_y


def in_bounds(map_info: OccupancyMap, cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < map_info.width and 0 <= cell[1] < map_info.height


def is_free_cell(map_info: OccupancyMap, cell: tuple[int, int]) -> bool:
    return in_bounds(map_info, cell) and bool(map_info.free_mask[cell[1], cell[0]])


def is_navigable_cell(map_info: OccupancyMap, support_grid: SupportGrid, cell: tuple[int, int]) -> bool:
    return in_bounds(map_info, cell) and is_free_cell(map_info, cell) and support_grid.is_supported(cell)


def build_support_grid(
    map_info: OccupancyMap,
    vertices: np.ndarray,
    min_points: int,
    max_spread: float,
    floor_percentile: float,
) -> SupportGrid:
    width = map_info.width
    height = map_info.height
    floor_z = np.full((height, width), np.nan, dtype=np.float32)
    point_count = np.zeros((height, width), dtype=np.int32)
    z_spread = np.full((height, width), np.inf, dtype=np.float32)

    cells_x = np.floor((vertices[:, 0] - map_info.origin_x) / map_info.resolution).astype(np.int32)
    world_rows = np.floor((vertices[:, 1] - map_info.origin_y) / map_info.resolution).astype(np.int32)
    cells_y = (height - 1) - world_rows

    valid = (cells_x >= 0) & (cells_x < width) & (cells_y >= 0) & (cells_y < height)
    if not np.any(valid):
        return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)

    cells_x = cells_x[valid]
    cells_y = cells_y[valid]
    verts_z = vertices[valid, 2]

    free_mask = map_info.free_mask[cells_y, cells_x]
    cells_x = cells_x[free_mask]
    cells_y = cells_y[free_mask]
    verts_z = verts_z[free_mask]
    if verts_z.size == 0:
        return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)

    linear = cells_y.astype(np.int64) * width + cells_x.astype(np.int64)
    order = np.argsort(linear)
    linear = linear[order]
    verts_z = verts_z[order]

    unique_linear, start_indices, counts = np.unique(linear, return_index=True, return_counts=True)
    for index, count in enumerate(counts):
        row = int(unique_linear[index] // width)
        col = int(unique_linear[index] % width)
        point_count[row, col] = int(count)
        start = start_indices[index]
        end = start + count
        cell_z = verts_z[start:end]
        spread = float(np.max(cell_z) - np.min(cell_z))
        z_spread[row, col] = spread
        if int(count) >= min_points and spread <= max_spread:
            floor_z[row, col] = float(np.percentile(cell_z, floor_percentile))

    return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def quat_wxyz_to_roll_pitch_yaw(quat: list[float]) -> tuple[float, float, float]:
    w, x, y, z = quat
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quat_wxyz(yaw: float) -> list[float]:
    half_yaw = 0.5 * float(yaw)
    return [float(math.cos(half_yaw)), 0.0, 0.0, float(math.sin(half_yaw))]


def parse_xy_pair(value: str, label: str) -> tuple[float, float]:
    pieces = [piece.strip() for piece in str(value).split(",")]
    if len(pieces) != 2:
        raise ValueError(f"{label} must be formatted as 'x,y'. Got: {value!r}")
    return float(pieces[0]), float(pieces[1])


def parse_cell_pair(value: str, label: str) -> tuple[int, int]:
    x, y = parse_xy_pair(value, label)
    return int(round(x)), int(round(y))


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


def build_nav_status_payload(
    *,
    task_type: str,
    goal_label: str | None,
    semantic_goal: str | None,
    episode_index: int,
    episode_count: int,
    goal_xy_map: np.ndarray,
    goal_cell: tuple[int, int],
    result: str,
    reason: str,
    distance_to_goal: float | None,
    current_cell: tuple[int, int] | None,
    path_cell_count: int | None,
    running: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mode": "world-model-navigation",
        "taskType": task_type,
        "goalLabel": goal_label,
        "semanticGoal": semantic_goal,
        "episodeIndex": int(episode_index),
        "episodeCount": int(episode_count),
        "goal": {"x": float(goal_xy_map[0]), "y": float(goal_xy_map[1])},
        "goalCell": [int(goal_cell[0]), int(goal_cell[1])],
        "result": result,
        "reason": reason,
        "running": bool(running),
        "stampMs": int(time.time() * 1000.0),
    }
    if distance_to_goal is not None:
        payload["distanceToGoal"] = float(distance_to_goal)
    if current_cell is not None:
        payload["currentCell"] = [int(current_cell[0]), int(current_cell[1])]
    if path_cell_count is not None:
        payload["pathCellCount"] = int(path_cell_count)
    return payload


def publish_agent_bridge_state(
    bridge_publisher: BridgePublisher | None,
    agent,
    *,
    nav_status: dict[str, Any],
    reset_trajectory: bool = False,
) -> None:
    if bridge_publisher is None or not bridge_publisher.enabled:
        return
    stamp_ms = int(time.time() * 1000.0)
    root_pos = np.asarray(agent.data.root_pos_w[0].detach().cpu().tolist(), dtype=np.float32)
    root_quat = np.asarray(agent.data.root_quat_w[0].detach().cpu().tolist(), dtype=np.float32)
    bridge_publisher.publish(
        {
            "pose": pose_payload_from_wxyz(root_pos, root_quat, frame_id="world", stamp_ms=stamp_ms),
            "resetTrajectory": bool(reset_trajectory),
            "navStatus": nav_status,
        }
    )


def build_episode_start_pose(
    settled_pose: np.ndarray,
    start_xy_map: np.ndarray,
    target_xy_map: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(settled_pose, dtype=np.float32).copy()
    direction = np.asarray(target_xy_map, dtype=np.float32) - np.asarray(start_xy_map, dtype=np.float32)
    if float(np.linalg.norm(direction)) > 1e-5:
        pose[3:] = np.asarray(yaw_to_quat_wxyz(math.atan2(float(direction[1]), float(direction[0]))), dtype=np.float32)
    return pose


def viewer_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "headless", False))


def perform_sim_step(sim, scene, dt: float, *, render: bool, step_sleep: float = 0.0) -> None:
    scene.write_data_to_sim()
    sim.step(render=render)
    scene.update(dt=dt)
    if step_sleep > 0.0:
        time.sleep(step_sleep)


def set_default_camera_view(sim, target_xy: tuple[float, float], target_z: float) -> None:
    eye = [float(target_xy[0] - 4.0), float(target_xy[1] - 4.0), float(target_z + 3.0)]
    target = [float(target_xy[0]), float(target_xy[1]), float(target_z + 0.5)]
    sim.set_camera_view(eye=eye, target=target)


def build_scene_translation(start_xy: np.ndarray, recenter_start_to_origin: bool) -> np.ndarray:
    translation = np.zeros(2, dtype=np.float32)
    if recenter_start_to_origin:
        translation = -np.asarray(start_xy[:2], dtype=np.float32)
    return translation


def map_xy_to_sim_xy(xy_map: np.ndarray, scene_translation_xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy_map, dtype=np.float32) + np.asarray(scene_translation_xy, dtype=np.float32)


def sim_xy_to_map_xy(xy_sim: np.ndarray, scene_translation_xy: np.ndarray) -> np.ndarray:
    return np.asarray(xy_sim, dtype=np.float32) - np.asarray(scene_translation_xy, dtype=np.float32)


def map_pose_to_sim_pose(pose_map: np.ndarray, scene_translation_xy: np.ndarray) -> np.ndarray:
    pose_sim = np.asarray(pose_map, dtype=np.float32).copy()
    pose_sim[:2] += np.asarray(scene_translation_xy, dtype=np.float32)
    return pose_sim


def update_follow_camera(
    sim,
    current_xy_sim: np.ndarray,
    focus_xy_sim: np.ndarray,
    current_z: float,
    args: argparse.Namespace,
) -> None:
    direction = np.asarray(focus_xy_sim, dtype=np.float32) - np.asarray(current_xy_sim, dtype=np.float32)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-5:
        direction = np.asarray([1.0, 0.0], dtype=np.float32)
    else:
        direction /= norm
    lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    eye_xy = (
        np.asarray(current_xy_sim, dtype=np.float32)
        - direction * float(args.camera_distance)
        + lateral * float(args.camera_side_offset)
    )
    target_xy = np.asarray(current_xy_sim, dtype=np.float32) + direction * float(args.camera_lookahead)
    sim.set_camera_view(
        eye=[float(eye_xy[0]), float(eye_xy[1]), float(current_z + args.camera_height)],
        target=[float(target_xy[0]), float(target_xy[1]), float(current_z + args.camera_target_height)],
    )


def update_goal_markers(
    goal_markers: VisualizationMarkers | None,
    goal_positions_sim: list[np.ndarray],
    goal_cells: list[tuple[int, int]],
    support_grid: SupportGrid,
    active_goal_index: int | None,
    completed_goal_count: int,
    args: argparse.Namespace,
) -> None:
    if goal_markers is None or not goal_positions_sim:
        return

    marker_indices = np.full(len(goal_positions_sim), GOAL_MARKER_UPCOMING, dtype=np.int32)
    if completed_goal_count > 0:
        marker_indices[: min(completed_goal_count, len(marker_indices))] = GOAL_MARKER_COMPLETED
    if active_goal_index is not None and 0 <= active_goal_index < len(marker_indices):
        marker_indices[active_goal_index] = GOAL_MARKER_CURRENT

    translations = np.zeros((len(goal_positions_sim), 3), dtype=np.float32)
    for index, (goal_xy_sim, goal_cell) in enumerate(zip(goal_positions_sim, goal_cells)):
        marker_kind = int(marker_indices[index])
        floor_z = float(support_grid.floor_z[goal_cell[1], goal_cell[0]])
        translations[index, 0] = float(goal_xy_sim[0])
        translations[index, 1] = float(goal_xy_sim[1])
        translations[index, 2] = floor_z + float(args.goal_marker_z_offset) + float(GOAL_MARKER_HEIGHTS[marker_kind] * 0.5)

    goal_markers.visualize(translations=translations, marker_indices=marker_indices)


def clear_path_draw(path_draw: object | None) -> None:
    if path_draw is not None:
        path_draw.clear_lines()


def acquire_path_draw_interface() -> object | None:
    try:
        import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    except ModuleNotFoundError:
        try:
            import omni.debugdraw._debugDraw as omni_debug_draw
        except ModuleNotFoundError:
            print("[WARN]: Debug-draw module unavailable; disabling path visualization.", flush=True)
            return None
    return omni_debug_draw.acquire_debug_draw_interface()


def draw_path_lines(
    path_draw: object | None,
    path_cells: list[tuple[int, int]],
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    scene_translation_xy: np.ndarray,
    path_line_z_offset: float,
) -> None:
    if path_draw is None:
        return
    if len(path_cells) < 2:
        clear_path_draw(path_draw)
        return

    start_points: list[list[float]] = []
    end_points: list[list[float]] = []
    colors: list[list[float]] = []
    thicknesses: list[float] = []
    for start_cell, end_cell in zip(path_cells[:-1], path_cells[1:]):
        start_xy_map = np.asarray(cell_to_world(map_info, start_cell[0], start_cell[1]), dtype=np.float32)
        end_xy_map = np.asarray(cell_to_world(map_info, end_cell[0], end_cell[1]), dtype=np.float32)
        start_xy_sim = map_xy_to_sim_xy(start_xy_map, scene_translation_xy)
        end_xy_sim = map_xy_to_sim_xy(end_xy_map, scene_translation_xy)
        start_floor_z = float(support_grid.floor_z[start_cell[1], start_cell[0]])
        end_floor_z = float(support_grid.floor_z[end_cell[1], end_cell[0]])
        start_points.append([float(start_xy_sim[0]), float(start_xy_sim[1]), start_floor_z + float(path_line_z_offset)])
        end_points.append([float(end_xy_sim[0]), float(end_xy_sim[1]), end_floor_z + float(path_line_z_offset)])
        colors.append([1.0, 0.9, 0.1, 1.0])
        thicknesses.append(4.0)

    path_draw.clear_lines()
    path_draw.draw_lines(start_points, end_points, colors, thicknesses)


def obstacle_ahead(
    map_info: OccupancyMap,
    position_xy: np.ndarray,
    goal_xy: np.ndarray,
    lookahead: float,
) -> bool:
    direction = goal_xy - position_xy
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        return False

    direction /= distance
    sample_count = max(2, int(math.ceil(lookahead / max(map_info.resolution * 0.5, 1e-3))))
    for i in range(1, sample_count + 1):
        point_xy = position_xy + direction * min(lookahead, lookahead * i / sample_count)
        cell = world_to_cell(map_info, float(point_xy[0]), float(point_xy[1]))
        if not in_bounds(map_info, cell):
            return True
        if map_info.occupied_mask[cell[1], cell[0]]:
            return True
    return False


def build_velocity_command(
    current_xy: np.ndarray,
    current_z_vel: float,
    goal_xy: np.ndarray,
    max_speed: float,
    slowdown_radius: float,
) -> tuple[np.ndarray, float]:
    delta = goal_xy - current_xy
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return np.zeros(6, dtype=np.float32), distance

    direction = delta / distance
    speed = max_speed
    if distance < slowdown_radius:
        speed = max_speed * max(distance / slowdown_radius, 0.15)

    command = np.zeros(6, dtype=np.float32)
    command[:2] = direction * speed
    command[2] = current_z_vel
    return command, distance


def select_path_lookahead_waypoint(
    current_xy: np.ndarray,
    path_waypoints: list[np.ndarray],
    waypoint_index: int,
    lookahead_distance: float,
) -> tuple[int, np.ndarray]:
    if not path_waypoints:
        raise ValueError("path_waypoints must not be empty.")

    if waypoint_index >= len(path_waypoints):
        waypoint_index = len(path_waypoints) - 1

    accumulated = 0.0
    previous_xy = np.asarray(current_xy, dtype=np.float32)
    lookahead_index = waypoint_index
    for candidate_index in range(waypoint_index, len(path_waypoints)):
        candidate_xy = np.asarray(path_waypoints[candidate_index], dtype=np.float32)
        accumulated += float(np.linalg.norm(candidate_xy - previous_xy))
        lookahead_index = candidate_index
        previous_xy = candidate_xy
        if accumulated >= lookahead_distance:
            break
    return lookahead_index, np.asarray(path_waypoints[lookahead_index], dtype=np.float32)


def build_obstacle_avoidance_vector(
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    current_xy: np.ndarray,
    influence_radius: float,
) -> np.ndarray:
    if influence_radius <= 0.0:
        return np.zeros(2, dtype=np.float32)

    current_cell = world_to_cell(map_info, float(current_xy[0]), float(current_xy[1]))
    radius_cells = max(1, int(math.ceil(influence_radius / max(map_info.resolution, 1e-6))))
    repulsion = np.zeros(2, dtype=np.float32)

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            cell = (current_cell[0] + dx, current_cell[1] + dy)
            if not in_bounds(map_info, cell):
                continue
            if is_navigable_cell(map_info, support_grid, cell):
                continue

            obstacle_xy = np.asarray(cell_to_world(map_info, cell[0], cell[1]), dtype=np.float32)
            offset = np.asarray(current_xy, dtype=np.float32) - obstacle_xy
            distance = float(np.linalg.norm(offset))
            if distance <= 1e-4 or distance >= influence_radius:
                continue

            strength = ((influence_radius - distance) / influence_radius) ** 2
            repulsion += (offset / distance) * strength

    return repulsion


def build_steering_velocity_command(
    current_xy: np.ndarray,
    target_xy: np.ndarray,
    max_speed: float,
    slowdown_radius: float,
    avoidance_vec: np.ndarray,
    avoidance_gain: float,
    avoidance_speed_scale: float,
) -> tuple[np.ndarray, float, float]:
    delta = np.asarray(target_xy, dtype=np.float32) - np.asarray(current_xy, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return np.zeros(2, dtype=np.float32), distance, 0.0

    path_dir = delta / distance
    steering = path_dir + np.asarray(avoidance_vec, dtype=np.float32) * float(avoidance_gain)
    steering_norm = float(np.linalg.norm(steering))
    if steering_norm < 1e-6:
        steering = path_dir
        steering_norm = 1.0
    steering /= steering_norm

    speed = max_speed
    if distance < slowdown_radius:
        speed = max_speed * max(distance / slowdown_radius, 0.15)

    avoidance_norm = float(np.linalg.norm(avoidance_vec))
    if avoidance_norm > 1e-6:
        speed *= max(0.25, 1.0 - float(avoidance_speed_scale) * min(avoidance_norm, 1.0))

    return steering * speed, distance, avoidance_norm


def astar_path(
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if not is_free_cell(map_info, start_cell) or not support_grid.is_supported(start_cell):
        return None
    if not is_free_cell(map_info, goal_cell) or not support_grid.is_supported(goal_cell):
        return None

    neighbor_offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
    ]

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1])

    heapq.heappush(open_heap, (heuristic(start_cell), 0.0, start_cell))
    visited: set[tuple[int, int]] = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy, cost in neighbor_offsets:
            neighbor = (current[0] + dx, current[1] + dy)
            if not in_bounds(map_info, neighbor):
                continue
            if not is_free_cell(map_info, neighbor):
                continue
            if not support_grid.is_supported(neighbor):
                continue

            tentative_g = current_g + cost
            if tentative_g >= g_score.get(neighbor, math.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), tentative_g, neighbor))

    return None


def build_component_goal_list(
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    start_cell: tuple[int, int],
    count: int,
    seed: int,
) -> list[tuple[np.ndarray, tuple[int, int]]]:
    component_cells = []
    if is_free_cell(map_info, start_cell) and support_grid.is_supported(start_cell):
        queue = deque([start_cell])
        visited = {start_cell}
        neighbor_offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        while queue:
            current = queue.popleft()
            component_cells.append(current)
            for dx, dy in neighbor_offsets:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in visited:
                    continue
                if not in_bounds(map_info, neighbor):
                    continue
                if not is_free_cell(map_info, neighbor):
                    continue
                if not support_grid.is_supported(neighbor):
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

    candidate_cells = [cell for cell in component_cells if cell != start_cell]
    if not candidate_cells:
        return []

    start_xy = np.asarray(cell_to_world(map_info, start_cell[0], start_cell[1]), dtype=np.float32)
    candidate_cells.sort(
        key=lambda cell: (
            -float(np.linalg.norm(np.asarray(cell_to_world(map_info, cell[0], cell[1]), dtype=np.float32) - start_xy)),
            cell[1],
            cell[0],
        )
    )
    rng = np.random.default_rng(seed)
    goals: list[tuple[np.ndarray, tuple[int, int]]] = []
    while len(goals) < count:
        for index in rng.permutation(len(candidate_cells)):
            cell = candidate_cells[int(index)]
            world_xy = np.asarray(cell_to_world(map_info, cell[0], cell[1]), dtype=np.float32)
            goals.append((world_xy, cell))
            if len(goals) >= count:
                break
    return goals


def build_agent_cfg(args: argparse.Namespace):
    print(f"[INFO] Minimal nav build_agent_cfg robot_model={args.robot_model}.", flush=True)
    if args.robot_model == "unitree-go2":
        if UNITREE_GO2_CFG is None:
            raise ModuleNotFoundError("isaaclab_assets.robots.unitree is unavailable; cannot use --robot-model unitree-go2.")
        agent_cfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Agent")
        agent_cfg = agent_cfg.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, float(args.quadruped_base_height)),
                joint_pos=UNITREE_GO2_CFG.init_state.joint_pos,
                joint_vel=UNITREE_GO2_CFG.init_state.joint_vel,
            ),
            spawn=agent_cfg.spawn.replace(
                rigid_props=agent_cfg.spawn.rigid_props.replace(
                    linear_damping=0.6,
                    angular_damping=1.2,
                    max_depenetration_velocity=3.0,
                ),
                articulation_props=agent_cfg.spawn.articulation_props.replace(
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
        )
        return agent_cfg, "{ENV_REGEX_NS}/Agent/.*"

    print("[INFO] Minimal nav building capsule config.", flush=True)
    capsule_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        spawn=sim_utils.CapsuleCfg(
            radius=0.25,
            height=0.50,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=bool(args.floating_agent),
                linear_damping=2.5,
                angular_damping=5.0,
                max_linear_velocity=50.0,
                max_angular_velocity=3600.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    print("[INFO] Minimal nav capsule config ready.", flush=True)
    return capsule_cfg, capsule_cfg.prim_path


def initialize_agent_control_state(agent, args: argparse.Namespace) -> AgentControlState:
    state = AgentControlState(model=args.robot_model)
    if args.robot_model != "unitree-go2":
        return state

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
    joint_ids, joint_names = agent.find_joints(leg_joint_names, preserve_order=True)
    if len(joint_ids) != len(leg_joint_names):
        raise RuntimeError(f"Could not resolve all Go2 leg joints. Found: {joint_names}")

    state.default_joint_pos = agent.data.default_joint_pos.clone()
    state.default_joint_vel = agent.data.default_joint_vel.clone()
    state.leg_joint_ids = {joint_name: int(joint_id) for joint_name, joint_id in zip(joint_names, joint_ids)}
    return state


def build_go2_joint_targets(
    control_state: AgentControlState,
    args: argparse.Namespace,
    *,
    commanded_speed: float,
    yaw_rate: float,
    dt: float,
) -> torch.Tensor:
    if control_state.default_joint_pos is None or control_state.leg_joint_ids is None:
        raise RuntimeError("Go2 control state is not initialized.")

    target = control_state.default_joint_pos.clone()
    motion_ratio = max(
        min(float(commanded_speed) / max(float(args.max_speed), 1e-6), 1.0),
        min(abs(float(yaw_rate)) / max(float(args.yaw_rate_limit), 1e-6), 1.0) * 0.65,
    )
    if motion_ratio < 0.05:
        control_state.gait_phase = 0.0
        return target

    cycle_hz = float(args.gait_cycle_hz) * max(0.4, motion_ratio)
    control_state.gait_phase = float((control_state.gait_phase + dt * cycle_hz * 2.0 * math.pi) % (2.0 * math.pi))

    stride_scale = float(args.gait_stride_scale) * max(0.35, motion_ratio)
    turn_ratio = float(np.clip(yaw_rate / max(float(args.yaw_rate_limit), 1e-6), -1.0, 1.0))
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


def apply_agent_command(
    agent,
    scene,
    control_state: AgentControlState,
    command_xy: np.ndarray,
    *,
    commanded_speed: float,
    yaw_rate: float,
    root_lin_vel_z: float,
    args: argparse.Namespace,
    dt: float,
) -> None:
    velocity_command = np.zeros((1, 6), dtype=np.float32)
    velocity_command[0, 0] = float(command_xy[0])
    velocity_command[0, 1] = float(command_xy[1])
    velocity_command[0, 2] = 0.0 if args.robot_model == "capsule" and args.floating_agent else float(root_lin_vel_z)
    velocity_command[0, 5] = 0.0 if args.robot_model == "capsule" else float(np.clip(yaw_rate, -args.yaw_rate_limit, args.yaw_rate_limit))
    velocity_tensor = torch.tensor(velocity_command, dtype=torch.float32, device=scene.device)
    agent.write_root_velocity_to_sim(velocity_tensor)

    if args.robot_model == "unitree-go2":
        joint_target = build_go2_joint_targets(
            control_state,
            args,
            commanded_speed=commanded_speed,
            yaw_rate=yaw_rate,
            dt=dt,
        )
        agent.set_joint_position_target(joint_target)


def stop_agent(agent, scene, control_state: AgentControlState) -> None:
    stop_velocity = torch.zeros((1, 6), dtype=torch.float32, device=scene.device)
    agent.write_root_velocity_to_sim(stop_velocity)
    if control_state.model == "unitree-go2" and control_state.default_joint_pos is not None:
        agent.set_joint_position_target(control_state.default_joint_pos)


def reset_agent(
    agent,
    settled_pose: np.ndarray,
    settle_steps: int,
    sim,
    scene,
    dt: float,
    *,
    render: bool,
    step_sleep: float,
    control_state: AgentControlState,
) -> None:
    pose_tensor = torch.tensor(settled_pose.reshape(1, 7), dtype=torch.float32, device=scene.device)
    velocity_tensor = torch.zeros((1, 6), dtype=torch.float32, device=scene.device)
    agent.write_root_pose_to_sim(pose_tensor)
    agent.write_root_velocity_to_sim(velocity_tensor)
    if control_state.model == "unitree-go2":
        if control_state.default_joint_pos is None or control_state.default_joint_vel is None:
            raise RuntimeError("Go2 control state is not initialized.")
        agent.write_joint_state_to_sim(control_state.default_joint_pos, control_state.default_joint_vel)
        agent.set_joint_position_target(control_state.default_joint_pos)
        control_state.gait_phase = 0.0
    scene.reset()
    for _ in range(settle_steps):
        agent.write_root_velocity_to_sim(velocity_tensor)
        if control_state.model == "unitree-go2" and control_state.default_joint_pos is not None:
            agent.set_joint_position_target(control_state.default_joint_pos)
        perform_sim_step(sim, scene, dt, render=render, step_sleep=step_sleep)


def run_episode(
    episode_index: int,
    episode_count: int,
    goal_xy_map: np.ndarray,
    goal_cell: tuple[int, int],
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    agent,
    sensor,
    sim,
    scene,
    settled_pose: np.ndarray,
    control_state: AgentControlState,
    viewer_state: ViewerState,
    args: argparse.Namespace,
    bridge_publisher: BridgePublisher | None,
) -> dict:
    dt = float(args.dt)
    max_steps = int(math.ceil(args.timeout_sec / dt))
    render = viewer_enabled(args)

    progress_window = deque(maxlen=max(2, int(math.ceil(args.stuck_window_sec / dt))))
    commanded_speed = 0.0

    result = "timeout"
    reason = "timeout"
    final_snapshot = {}
    max_contact_force = 0.0
    task_type = "semantic_goal" if args.semantic_goal else "point_goal"
    goal_label = str(args.goal_label or args.semantic_goal or f"goal-{episode_index}")
    bridge_interval_steps = max(
        1,
        int(round((1.0 / max(float(args.bridge_publish_rate_hz), 1e-3)) / max(dt, 1e-6))),
    )

    scene_translation_xy = viewer_state.scene_translation_xy
    goal_xy_sim = map_xy_to_sim_xy(goal_xy_map, scene_translation_xy)
    start_xy = sim_xy_to_map_xy(settled_pose[:2], scene_translation_xy)
    start_cell = world_to_cell(map_info, float(start_xy[0]), float(start_xy[1]))
    if not in_bounds(map_info, goal_cell) or not is_free_cell(map_info, goal_cell):
        return {
            "episode": episode_index,
            "goal_xy": [float(goal_xy_map[0]), float(goal_xy_map[1]), 0.0],
            "goal_xy_w": [float(goal_xy_sim[0]), float(goal_xy_sim[1]), 0.0],
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            "result": "stuck",
            "reason": "goal_not_free",
            "duration_sec": 0.0,
            "max_contact_force": 0.0,
            "final": {
                "position_w": settled_pose[:3].tolist(),
                "position_map_xy": [float(start_xy[0]), float(start_xy[1])],
                "quat_w": settled_pose[3:].tolist(),
                "start_cell": [int(start_cell[0]), int(start_cell[1])],
            },
        }
    if not support_grid.is_supported(start_cell):
        return {
            "episode": episode_index,
            "goal_xy": [float(goal_xy_map[0]), float(goal_xy_map[1]), 0.0],
            "goal_xy_w": [float(goal_xy_sim[0]), float(goal_xy_sim[1]), 0.0],
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            "result": "stuck",
            "reason": "start_not_supported",
            "duration_sec": 0.0,
            "max_contact_force": 0.0,
            "final": {
                "position_w": settled_pose[:3].tolist(),
                "position_map_xy": [float(start_xy[0]), float(start_xy[1])],
                "quat_w": settled_pose[3:].tolist(),
                "start_cell": [int(start_cell[0]), int(start_cell[1])],
            },
        }
    if not support_grid.is_supported(goal_cell):
        return {
            "episode": episode_index,
            "goal_xy": [float(goal_xy_map[0]), float(goal_xy_map[1]), 0.0],
            "goal_xy_w": [float(goal_xy_sim[0]), float(goal_xy_sim[1]), 0.0],
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            "result": "stuck",
            "reason": "goal_not_supported",
            "duration_sec": 0.0,
            "max_contact_force": 0.0,
            "final": {
                "position_w": settled_pose[:3].tolist(),
                "position_map_xy": [float(start_xy[0]), float(start_xy[1])],
                "quat_w": settled_pose[3:].tolist(),
                "start_cell": [int(start_cell[0]), int(start_cell[1])],
            },
        }

    path_cells = astar_path(map_info, support_grid, start_cell, goal_cell)
    if path_cells is None:
        return {
            "episode": episode_index,
            "goal_xy": [float(goal_xy_map[0]), float(goal_xy_map[1]), 0.0],
            "goal_xy_w": [float(goal_xy_sim[0]), float(goal_xy_sim[1]), 0.0],
            "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            "result": "stuck",
            "reason": "planner_no_path",
            "duration_sec": 0.0,
            "max_contact_force": 0.0,
            "final": {
                "position_w": settled_pose[:3].tolist(),
                "position_map_xy": [float(start_xy[0]), float(start_xy[1])],
                "quat_w": settled_pose[3:].tolist(),
                "start_cell": [int(start_cell[0]), int(start_cell[1])],
                "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
            },
        }

    path_waypoints = [np.asarray(cell_to_world(map_info, cell[0], cell[1]), dtype=np.float32) for cell in path_cells]
    path_waypoints_sim = [map_xy_to_sim_xy(waypoint_xy, scene_translation_xy) for waypoint_xy in path_waypoints]
    waypoint_index = 1 if len(path_waypoints) > 1 else 0
    episode_start_pose = (
        build_episode_start_pose(settled_pose, start_xy, path_waypoints[waypoint_index])
        if args.robot_model == "unitree-go2"
        else np.asarray(settled_pose, dtype=np.float32).copy()
    )

    reset_agent(
        agent,
        episode_start_pose,
        args.settle_steps,
        sim,
        scene,
        dt,
        render=render,
        step_sleep=args.step_sleep,
        control_state=control_state,
    )

    publish_agent_bridge_state(
        bridge_publisher,
        agent,
        nav_status=build_nav_status_payload(
            task_type=task_type,
            goal_label=goal_label,
            semantic_goal=args.semantic_goal,
            episode_index=episode_index,
            episode_count=episode_count,
            goal_xy_map=goal_xy_map,
            goal_cell=goal_cell,
            result="running",
            reason="episode_started",
            distance_to_goal=float(np.linalg.norm(goal_xy_map - start_xy)),
            current_cell=start_cell,
            path_cell_count=len(path_cells),
            running=True,
        ),
        reset_trajectory=episode_index == 0,
    )

    if render:
        if args.show_goals and viewer_state.goal_markers is not None and viewer_state.goal_positions_sim is not None and viewer_state.goal_cells is not None:
            update_goal_markers(
                goal_markers=viewer_state.goal_markers,
                goal_positions_sim=viewer_state.goal_positions_sim,
                goal_cells=viewer_state.goal_cells,
                support_grid=support_grid,
                active_goal_index=episode_index,
                completed_goal_count=episode_index,
                args=args,
            )
        if args.show_path and viewer_state.path_draw is not None:
            draw_path_lines(
                path_draw=viewer_state.path_draw,
                path_cells=path_cells,
                map_info=map_info,
                support_grid=support_grid,
                scene_translation_xy=scene_translation_xy,
                path_line_z_offset=float(args.path_line_z_offset),
            )
        if args.follow_camera:
            update_follow_camera(
                sim,
                episode_start_pose[:2],
                path_waypoints_sim[waypoint_index],
                float(episode_start_pose[2]),
                args,
            )

    for step in range(max_steps):
        root_pos = agent.data.root_pos_w[0]
        root_quat = agent.data.root_quat_w[0]
        root_lin_vel = agent.data.root_lin_vel_w[0]
        current_xy_sim = np.asarray(root_pos[:2].detach().cpu().tolist(), dtype=np.float32)
        current_xy = sim_xy_to_map_xy(current_xy_sim, scene_translation_xy)
        while waypoint_index < len(path_waypoints) - 1 and np.linalg.norm(path_waypoints[waypoint_index] - current_xy) <= args.waypoint_reach_distance:
            waypoint_index += 1
        lookahead_index, control_target_xy = select_path_lookahead_waypoint(
            current_xy=current_xy,
            path_waypoints=path_waypoints,
            waypoint_index=waypoint_index,
            lookahead_distance=args.path_lookahead_distance,
        )
        waypoint_xy = path_waypoints[waypoint_index]
        waypoint_xy_sim = path_waypoints_sim[lookahead_index]

        avoidance_vec = build_obstacle_avoidance_vector(
            map_info=map_info,
            support_grid=support_grid,
            current_xy=current_xy,
            influence_radius=args.avoidance_radius,
        )
        command_xy, _, avoidance_norm = build_steering_velocity_command(
            current_xy=current_xy,
            target_xy=control_target_xy,
            max_speed=args.max_speed,
            slowdown_radius=args.slowdown_radius,
            avoidance_vec=avoidance_vec,
            avoidance_gain=args.avoidance_gain,
            avoidance_speed_scale=args.avoidance_speed_scale,
        )
        commanded_speed = float(np.linalg.norm(command_xy))
        _, _, current_yaw = quat_wxyz_to_roll_pitch_yaw(tensor_to_list(root_quat))
        target_heading = math.atan2(float(control_target_xy[1] - current_xy[1]), float(control_target_xy[0] - current_xy[0]))
        yaw_error = normalize_angle(target_heading - current_yaw)
        yaw_rate = float(np.clip(yaw_error * args.yaw_rate_gain, -args.yaw_rate_limit, args.yaw_rate_limit))
        apply_agent_command(
            agent,
            scene,
            control_state,
            command_xy,
            commanded_speed=commanded_speed,
            yaw_rate=yaw_rate,
            root_lin_vel_z=float(root_lin_vel[2].item()),
            args=args,
            dt=dt,
        )

        perform_sim_step(sim, scene, dt, render=render, step_sleep=args.step_sleep)

        root_pos = agent.data.root_pos_w[0]
        root_quat = agent.data.root_quat_w[0]
        root_lin_vel = agent.data.root_lin_vel_w[0]
        root_ang_vel = agent.data.root_ang_vel_w[0]
        current_xy_sim = np.asarray(root_pos[:2].detach().cpu().tolist(), dtype=np.float32)
        current_xy = sim_xy_to_map_xy(current_xy_sim, scene_translation_xy)
        current_cell = world_to_cell(map_info, float(current_xy[0]), float(current_xy[1]))
        progress_window.append(current_xy.copy())
        contact_force = 0.0
        if sensor is not None:
            contact_force = float(torch.linalg.norm(sensor.data.net_forces_w.reshape(-1, 3), dim=-1).max().item())
            max_contact_force = max(max_contact_force, contact_force)

        quat_list = tensor_to_list(root_quat)
        roll, pitch, yaw = quat_wxyz_to_roll_pitch_yaw(quat_list)

        final_snapshot = {
            "step": step,
            "position_w": tensor_to_list(root_pos),
            "position_map_xy": [float(current_xy[0]), float(current_xy[1])],
            "quat_w": quat_list,
            "lin_vel_w": tensor_to_list(root_lin_vel),
            "ang_vel_w": tensor_to_list(root_ang_vel),
            "goal_distance": float(np.linalg.norm(goal_xy_map - current_xy)),
            "current_cell": [int(current_cell[0]), int(current_cell[1])],
            "waypoint_index": int(waypoint_index),
            "lookahead_index": int(lookahead_index),
            "path_cell_count": len(path_cells),
            "commanded_speed": commanded_speed,
            "yaw_rate_cmd": yaw_rate,
            "avoidance_norm": avoidance_norm,
            "contact_force_norm_max": contact_force,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

        if step % bridge_interval_steps == 0:
            publish_agent_bridge_state(
                bridge_publisher,
                agent,
                nav_status=build_nav_status_payload(
                    task_type=task_type,
                    goal_label=goal_label,
                    semantic_goal=args.semantic_goal,
                    episode_index=episode_index,
                    episode_count=episode_count,
                    goal_xy_map=goal_xy_map,
                    goal_cell=goal_cell,
                    result="running",
                    reason="tracking_path",
                    distance_to_goal=float(final_snapshot["goal_distance"]),
                    current_cell=current_cell,
                    path_cell_count=len(path_cells),
                    running=True,
                ),
            )

        if not in_bounds(map_info, current_cell) or not is_free_cell(map_info, current_cell):
            result = "collision"
            reason = "occupied_or_oob"
            break

        if not support_grid.is_supported(current_cell):
            result = "collision"
            reason = "unsupported_floor"
            break

        if abs(roll) > args.collision_tilt_rad or abs(pitch) > args.collision_tilt_rad:
            result = "collision"
            reason = "tilt_limit"
            break

        if final_snapshot["goal_distance"] <= args.goal_tolerance:
            result = "success"
            reason = "goal_tolerance"
            break

        if len(progress_window) == progress_window.maxlen and commanded_speed >= args.command_threshold:
            displacement = float(np.linalg.norm(progress_window[-1] - progress_window[0]))
            blocked = obstacle_ahead(map_info, current_xy, goal_xy_map, args.collision_lookahead)
            if displacement < args.stuck_displacement:
                if blocked or not is_free_cell(map_info, current_cell):
                    result = "collision"
                    reason = "map_blocked"
                else:
                    result = "stuck"
                    reason = "low_progress"
                final_snapshot["progress_window_displacement"] = displacement
                break

        if render and args.follow_camera:
            update_follow_camera(sim, current_xy_sim, waypoint_xy_sim, float(root_pos[2].item()), args)
        if render and args.show_path and viewer_state.path_draw is not None:
            draw_path_lines(
                path_draw=viewer_state.path_draw,
                path_cells=path_cells,
                map_info=map_info,
                support_grid=support_grid,
                scene_translation_xy=scene_translation_xy,
                path_line_z_offset=float(args.path_line_z_offset),
            )

    episode = {
        "episode": episode_index,
        "goal_xy": [float(goal_xy_map[0]), float(goal_xy_map[1]), 0.0],
        "goal_xy_w": [float(goal_xy_sim[0]), float(goal_xy_sim[1]), 0.0],
        "goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
        "result": result,
        "reason": reason,
        "duration_sec": float((final_snapshot.get("step", max_steps - 1) + 1) * dt),
        "max_contact_force": max_contact_force,
        "final": final_snapshot,
    }
    stop_agent(agent, scene, control_state)
    publish_agent_bridge_state(
        bridge_publisher,
        agent,
        nav_status=build_nav_status_payload(
            task_type=task_type,
            goal_label=goal_label,
            semantic_goal=args.semantic_goal,
            episode_index=episode_index,
            episode_count=episode_count,
            goal_xy_map=goal_xy_map,
            goal_cell=goal_cell,
            result=result,
            reason=reason,
            distance_to_goal=float(final_snapshot.get("goal_distance")) if "goal_distance" in final_snapshot else None,
            current_cell=tuple(final_snapshot["current_cell"]) if "current_cell" in final_snapshot else None,
            path_cell_count=len(path_cells),
            running=False,
        ),
    )
    if render and args.show_goals and viewer_state.goal_markers is not None and viewer_state.goal_positions_sim is not None and viewer_state.goal_cells is not None:
        update_goal_markers(
            goal_markers=viewer_state.goal_markers,
            goal_positions_sim=viewer_state.goal_positions_sim,
            goal_cells=viewer_state.goal_cells,
            support_grid=support_grid,
            active_goal_index=None,
            completed_goal_count=episode_index + 1,
            args=args,
        )
    if render and args.episode_hold_sec > 0.0:
        hold_steps = max(1, int(math.ceil(args.episode_hold_sec / max(dt, 1e-3))))
        hold_sleep = args.step_sleep if args.step_sleep > 0.0 else min(dt, 1.0 / 30.0)
        for _ in range(hold_steps):
            perform_sim_step(sim, scene, dt, render=True, step_sleep=hold_sleep)
    return episode


def run_nav_loop(args: argparse.Namespace) -> dict:
    print("[INFO] Minimal nav episode runner entering run_nav_loop.", flush=True)
    validation_path = Path(args.validation_report).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation report not found: {validation_path}")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    collision_usd = Path(validation["inputs"]["collision_usd"]).expanduser().resolve()
    map_yaml = Path(validation["inputs"]["map_yaml"]).expanduser().resolve()
    mesh_ply = Path(validation["inputs"]["mesh_ply"]).expanduser().resolve()
    if not collision_usd.is_file():
        raise FileNotFoundError(f"Collision USD not found from validation report: {collision_usd}")
    if not map_yaml.is_file():
        raise FileNotFoundError(f"Map YAML not found from validation report: {map_yaml}")
    if not mesh_ply.is_file():
        raise FileNotFoundError(f"Mesh PLY not found from validation report: {mesh_ply}")

    map_info = load_occupancy_map(map_yaml)
    mesh = trimesh.load(mesh_ply, force="mesh", process=False)
    support_grid = build_support_grid(
        map_info=map_info,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        min_points=1,
        max_spread=1.5,
        floor_percentile=10.0,
    )
    args.dt = float(validation["inputs"]["dt"])
    validation_seed = int(validation["inputs"].get("seed", 0))
    start_cell = tuple(int(v) for v in validation["map"]["start_cell"])
    start_xy = np.asarray(validation["map"]["start_xy"], dtype=np.float32)
    start_floor_z = float(support_grid.floor_z[start_cell[1], start_cell[0]])
    if not np.isfinite(start_floor_z):
        raise RuntimeError(f"Validation start cell is not supported in the support grid: {start_cell}")
    settled_pose_map = np.zeros(7, dtype=np.float32)
    settled_pose_map[0] = float(start_xy[0])
    settled_pose_map[1] = float(start_xy[1])
    if args.robot_model == "unitree-go2":
        settled_pose_map[2] = float(start_floor_z + args.quadruped_base_height)
        settled_pose_map[3:] = np.asarray(yaw_to_quat_wxyz(0.0), dtype=np.float32)
    else:
        z_offset = float(
            np.asarray(validation["physics_validation"]["final"]["root_pos_w"], dtype=np.float32)[2]
            - float(validation["floor_estimate"]["floor_z"])
        )
        settled_pose_map[2] = float(start_floor_z + z_offset)
        settled_pose_map[3:] = np.asarray(validation["physics_validation"]["final"]["root_quat_w"], dtype=np.float32)
    scene_translation_xy = build_scene_translation(start_xy, args.recenter_start_to_origin)
    settled_pose = map_pose_to_sim_pose(settled_pose_map, scene_translation_xy)

    goal_override: tuple[np.ndarray, tuple[int, int]] | None = None
    if args.goal_world and args.goal_cell:
        raise ValueError("Use either --goal-world or --goal-cell, not both.")
    if args.goal_world:
        goal_x, goal_y = parse_xy_pair(args.goal_world, "--goal-world")
        goal_xy = np.asarray([goal_x, goal_y], dtype=np.float32)
        goal_override = (goal_xy, world_to_cell(map_info, float(goal_xy[0]), float(goal_xy[1])))
    elif args.goal_cell:
        goal_cell_override = parse_cell_pair(args.goal_cell, "--goal-cell")
        goal_xy = np.asarray(cell_to_world(map_info, goal_cell_override[0], goal_cell_override[1]), dtype=np.float32)
        goal_override = (goal_xy, goal_cell_override)

    requested_episode_count = 1 if goal_override is not None else (
        args.episode_limit if args.episode_limit is not None else max(20, len(validation["map"]["goals"]))
    )
    validation_goals: list[tuple[np.ndarray, tuple[int, int]]] = []
    for goal in validation["map"]["goals"]:
        goal_xy = np.asarray(goal[:2], dtype=np.float32)
        goal_cell = world_to_cell(map_info, float(goal_xy[0]), float(goal_xy[1]))
        if goal_cell == start_cell:
            continue
        if not in_bounds(map_info, goal_cell):
            continue
        if not is_free_cell(map_info, goal_cell):
            continue
        if not support_grid.is_supported(goal_cell):
            continue
        if astar_path(map_info, support_grid, start_cell, goal_cell) is None:
            continue
        validation_goals.append((goal_xy, goal_cell))

    if goal_override is not None:
        selected_goals = [goal_override]
        goal_source = "explicit_goal"
    elif len(validation_goals) >= requested_episode_count:
        selected_goals = validation_goals[:requested_episode_count]
        goal_source = "validation_report"
    else:
        selected_goals = build_component_goal_list(
            map_info=map_info,
            support_grid=support_grid,
            start_cell=start_cell,
            count=requested_episode_count,
            seed=validation_seed,
        )
        goal_source = "start_component_resampled"

    if not selected_goals:
        raise RuntimeError(f"No runnable goals found from start cell {start_cell}.")

    goals = [goal_xy for goal_xy, _ in selected_goals]
    goal_cells = [goal_cell for _, goal_cell in selected_goals]
    goal_positions_sim = [map_xy_to_sim_xy(goal_xy, scene_translation_xy) for goal_xy in goals]
    print(
        f"[INFO] Minimal nav selected {len(goals)} goal(s) from {goal_source}; "
        f"start_cell={start_cell} first_goal_cell={goal_cells[0] if goal_cells else None}.",
        flush=True,
    )

    scene_cfg = NavEpisodeSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False, lazy_sensor_update=False)
    print("[INFO] Minimal nav scene config allocated.", flush=True)
    scene_cfg.environment = scene_cfg.environment.replace(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=str(collision_usd)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(float(scene_translation_xy[0]), float(scene_translation_xy[1]), 0.0),
        ),
    )
    print("[INFO] Minimal nav environment USD configured.", flush=True)
    agent_cfg, contact_sensor_prim = build_agent_cfg(args)
    print("[INFO] Minimal nav agent config built.", flush=True)
    scene_cfg.agent = agent_cfg
    scene_cfg.contact_sensor = scene_cfg.contact_sensor.replace(prim_path=contact_sensor_prim)

    sim_cfg = sim_utils.SimulationCfg(dt=validation["inputs"]["dt"], device=args.device)
    print("[INFO] Minimal nav sim config allocated.", flush=True)
    sim_cfg.physx.enable_ccd = True
    sim_cfg.physx.enable_enhanced_determinism = True

    episodes: list[dict] = []
    counts = {"success": 0, "collision": 0, "timeout": 0, "stuck": 0}
    bridge_publisher = BridgePublisher(str(args.bridge_url)) if str(args.bridge_url or "").strip() else None

    print("[INFO] Minimal nav creating Isaac simulation context.", flush=True)
    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False, add_lighting=False) as sim:
        print("[INFO] Minimal nav Isaac simulation context ready.", flush=True)
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()
        render = viewer_enabled(args)
        goal_markers = None
        path_draw = None
        if render and args.show_goals:
            goal_markers = build_goal_marker_visualizer()
        if render and args.show_path:
            path_draw = acquire_path_draw_interface()
        viewer_state = ViewerState(
            scene_translation_xy=scene_translation_xy,
            goal_positions_sim=goal_positions_sim,
            goal_cells=goal_cells,
            goal_markers=goal_markers,
            path_draw=path_draw,
        )
        if render:
            if args.follow_camera and goal_positions_sim:
                update_follow_camera(sim, settled_pose[:2], goal_positions_sim[0], float(settled_pose[2]), args)
            else:
                set_default_camera_view(sim, (float(settled_pose[0]), float(settled_pose[1])), float(settled_pose[2]))

        agent = scene["agent"]
        sensor = scene["contact_sensor"]
        control_state = initialize_agent_control_state(agent, args)

        for episode_index, (goal_xy, goal_cell) in enumerate(zip(goals, goal_cells)):
            episode = run_episode(
                episode_index=episode_index,
                episode_count=len(goals),
                goal_xy_map=goal_xy,
                goal_cell=goal_cell,
                map_info=map_info,
                support_grid=support_grid,
                agent=agent,
                sensor=sensor,
                sim=sim,
                scene=scene,
                settled_pose=settled_pose,
                control_state=control_state,
                viewer_state=viewer_state,
                args=args,
                bridge_publisher=bridge_publisher,
            )
            counts[episode["result"]] += 1
            episodes.append(episode)
            print(
                f"episode={episode_index} result={episode['result']} reason={episode['reason']} "
                f"goal_cell={tuple(goal_cell)} duration_sec={episode['duration_sec']:.3f}",
                flush=True,
            )

        if render and args.hold_open_sec > 0.0:
            hold_steps = max(1, int(math.ceil(args.hold_open_sec / max(args.dt, 1e-3))))
            hold_sleep = args.step_sleep if args.step_sleep > 0.0 else min(args.dt, 1.0 / 30.0)
            for _ in range(hold_steps):
                perform_sim_step(sim, scene, args.dt, render=True, step_sleep=hold_sleep)
        clear_path_draw(path_draw)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Minimal nav writing report to {output_path}.", flush=True)
    report = {
        "validation_report": str(validation_path),
        "episode_count": len(episodes),
        "inputs": {
            "collision_usd": str(collision_usd),
            "mesh_ply": str(mesh_ply),
            "map_yaml": str(map_yaml),
            "device": args.device,
            "robot_model": args.robot_model,
            "goal_cell": args.goal_cell,
            "goal_world": args.goal_world,
            "goal_label": args.goal_label,
            "semantic_goal": args.semantic_goal,
            "timeout_sec": args.timeout_sec,
            "max_speed": args.max_speed,
            "goal_tolerance": args.goal_tolerance,
            "slowdown_radius": args.slowdown_radius,
            "stuck_window_sec": args.stuck_window_sec,
            "stuck_displacement": args.stuck_displacement,
            "collision_lookahead": args.collision_lookahead,
            "collision_tilt_rad": args.collision_tilt_rad,
            "settle_steps": args.settle_steps,
            "path_lookahead_distance": args.path_lookahead_distance,
            "waypoint_reach_distance": args.waypoint_reach_distance,
            "avoidance_radius": args.avoidance_radius,
            "avoidance_gain": args.avoidance_gain,
            "avoidance_speed_scale": args.avoidance_speed_scale,
            "floating_agent": args.floating_agent,
            "quadruped_base_height": args.quadruped_base_height,
            "yaw_rate_gain": args.yaw_rate_gain,
            "yaw_rate_limit": args.yaw_rate_limit,
            "gait_cycle_hz": args.gait_cycle_hz,
            "gait_stride_scale": args.gait_stride_scale,
            "show_goals": args.show_goals,
            "show_path": args.show_path,
            "follow_camera": args.follow_camera,
            "recenter_start_to_origin": args.recenter_start_to_origin,
            "bridge_url": args.bridge_url,
            "bridge_publish_rate_hz": args.bridge_publish_rate_hz,
        },
        "start_pose_w": settled_pose.tolist(),
        "start_pose_map": settled_pose_map.tolist(),
        "start_cell": [int(start_cell[0]), int(start_cell[1])],
        "scene_translation_w": [float(scene_translation_xy[0]), float(scene_translation_xy[1]), 0.0],
        "goal_source": goal_source,
        "requested_episode_count": int(requested_episode_count),
        "support_grid": {
            "supported_cell_count": int(np.isfinite(support_grid.floor_z).sum()),
        },
        "counts": counts,
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    print("[INFO] Minimal nav episode runner main started.", flush=True)
    try:
        run_nav_loop(ARGS_CLI)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
