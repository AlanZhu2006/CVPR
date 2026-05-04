#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path as NavPath
from std_msgs.msg import Header


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _pose_stamped(frame_id: str, stamp: Any, position: list[float], yaw: float = 0.0) -> PoseStamped:
    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.pose.position.x = float(position[0])
    msg.pose.position.y = float(position[1])
    msg.pose.position.z = float(position[2])
    qx, qy, qz, qw = _yaw_to_quat(yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    return msg


def _load_monitor(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class Nav2StylePublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("gs_console_nav2_style_bridge")
        self.args = args
        self.monitor_json = Path(args.monitor_json).expanduser().resolve()
        self.frame_id = args.frame_id
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 1)
        self.traj_pub = self.create_publisher(NavPath, "/lingbot_nav/trajectory", 1)
        self.plan_pub = self.create_publisher(NavPath, "/lingbot_nav/plan", 1)
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, 10)
        self.last_goal: list[float] | None = None
        self.last_plan: list[list[float]] = []
        self.timer = self.create_timer(float(args.interval_sec), self._tick)

    def _on_goal(self, msg: PoseStamped) -> None:
        # Web/Nav2 goals are 2D in the map x/y plane. LingBot live geometry uses x/z
        # as the ground plane, so convert map y back into LingBot z.
        goal_z = float(msg.pose.position.z)
        if abs(goal_z) < 1.0e-6:
            goal_z = float(msg.pose.position.y)
        self.last_goal = [float(msg.pose.position.x), 0.0, goal_z]
        self.get_logger().info(f"received /goal_pose -> {self.last_goal}")

    def _tick(self) -> None:
        payload = _load_monitor(self.monitor_json)
        if not payload:
            return
        stamp = self.get_clock().now().to_msg()
        points = np.asarray(payload.get("points", []), dtype=np.float32)
        trajectory = payload.get("trajectory", [])
        grid, origin, resolution = self._make_grid(points)
        self.map_pub.publish(self._occupancy_msg(grid, origin, resolution, stamp))
        self.traj_pub.publish(self._path_msg([item["position"] for item in trajectory if item.get("position")], stamp))
        if self.last_goal is not None and trajectory:
            start = trajectory[-1]["position"]
            self.last_plan = self._plan(points, start, self.last_goal)
        if self.last_plan:
            self.plan_pub.publish(self._path_msg(self.last_plan, stamp))

    def _make_grid(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        resolution = float(self.args.resolution)
        if points.size == 0:
            return np.full((64, 64), -1, dtype=np.int8), np.array([-5.0, -5.0], dtype=np.float32), resolution
        xyz = points[:, :3]
        xz = xyz[:, [0, 2]]
        mn = np.percentile(xz, 1, axis=0) - 0.8
        mx = np.percentile(xz, 99, axis=0) + 0.8
        size = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 16)
        size = np.minimum(size, int(self.args.max_cells))
        resolution = max(float((mx - mn).max() / max(size.max() - 1, 1)), resolution)
        occ_hits = np.zeros((int(size[0]), int(size[1])), dtype=np.int16)
        free_hits = np.zeros_like(occ_hits)
        cells = np.floor((xz - mn) / resolution).astype(np.int32)
        valid = (cells[:, 0] >= 0) & (cells[:, 1] >= 0) & (cells[:, 0] < size[0]) & (cells[:, 1] < size[1])
        cells = cells[valid]
        y = xyz[valid, 1]
        floor_y = float(np.percentile(xyz[:, 1], 8))
        high = y > floor_y + float(self.args.obstacle_height)
        for c in cells[~high]:
            free_hits[c[0], c[1]] += 1
        for c in cells[high]:
            occ_hits[c[0], c[1]] += 1
        grid = np.full_like(occ_hits, -1, dtype=np.int8)
        grid[free_hits >= 1] = 0
        grid[occ_hits >= int(self.args.occupied_hits)] = 100
        return grid, mn.astype(np.float32), resolution

    def _occupancy_msg(self, grid: np.ndarray, origin: np.ndarray, resolution: float, stamp: Any) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = Header(frame_id=self.frame_id, stamp=stamp)
        msg.info.resolution = float(resolution)
        msg.info.width = int(grid.shape[0])
        msg.info.height = int(grid.shape[1])
        msg.info.origin.position.x = float(origin[0])
        msg.info.origin.position.y = float(origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.T.reshape(-1).astype(np.int8).tolist()
        return msg

    def _path_msg(self, positions: list[list[float]], stamp: Any) -> NavPath:
        msg = NavPath()
        msg.header = Header(frame_id=self.frame_id, stamp=stamp)
        poses = []
        for idx, pos in enumerate(positions):
            yaw = 0.0
            if idx + 1 < len(positions):
                nxt = positions[idx + 1]
                yaw = math.atan2(float(nxt[2]) - float(pos[2]), float(nxt[0]) - float(pos[0]))
            poses.append(_pose_stamped(self.frame_id, stamp, pos, yaw))
        msg.poses = poses
        return msg

    def _plan(self, points: np.ndarray, start: list[float], goal: list[float]) -> list[list[float]]:
        # Keep the baseline deterministic and cheap; the true Nav2 planner can replace this topic later.
        steps = max(2, int(np.linalg.norm(np.asarray(goal) - np.asarray(start)) / max(0.1, float(self.args.resolution))))
        return [
            (np.asarray(start, dtype=np.float32) * (1.0 - t) + np.asarray(goal, dtype=np.float32) * t).astype(float).tolist()
            for t in np.linspace(0.0, 1.0, min(steps, 80))
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish GS Console monitor map as Nav2-style ROS2 topics.")
    parser.add_argument("--monitor-json", default="nuc_output/real2sim_hikrobot_lingbot_live_baseline/monitor/live_monitor.json")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=0.12)
    parser.add_argument("--max-cells", type=int, default=300)
    parser.add_argument("--obstacle-height", type=float, default=0.35)
    parser.add_argument("--occupied-hits", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = Nav2StylePublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
