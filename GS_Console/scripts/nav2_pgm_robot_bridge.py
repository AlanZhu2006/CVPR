#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformBroadcaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Differential-drive ROS2 robot shim for a Nav2 PGM map.")
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--validation-report", default="")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8890")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--scan-rate-hz", type=float, default=10.0)
    parser.add_argument("--bridge-rate-hz", type=float, default=8.0)
    parser.add_argument("--map-cloud-rate-hz", type=float, default=1.0)
    parser.add_argument("--map-cloud-max-points", type=int, default=18000)
    parser.add_argument("--bridge-pose-mode", choices=["pose-and-cloud", "cloud-only"], default="pose-and-cloud")
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument(
        "--start-clearance-margin",
        type=float,
        default=0.08,
        help="Extra obstacle clearance, in meters, required when selecting a clearance/rightmost start pose.",
    )
    parser.add_argument("--start-policy", choices=["validation", "clearance", "rightmost"], default="validation")
    parser.add_argument("--motion-mode", choices=["cmd_vel", "plan"], default="cmd_vel")
    parser.add_argument("--max-speed", type=float, default=0.55)
    parser.add_argument("--max-yaw-rate", type=float, default=1.6)
    return parser.parse_args()


def read_map_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    origin = [float(v.strip()) for v in data["origin"].strip("[]").split(",")[:3]]
    return {
        "image": data.get("image", path.with_suffix(".pgm").name),
        "resolution": float(data["resolution"]),
        "origin": origin,
    }


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise RuntimeError(f"Unsupported PGM: {path}")
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = [int(v) for v in line.split()[:2]]
        max_value = int(handle.readline())
        if magic == b"P5":
            image = np.frombuffer(handle.read(width * height), dtype=np.uint8).reshape((height, width))
        else:
            values = [int(v) for v in handle.read().split()]
            image = np.asarray(values, dtype=np.uint16).reshape((height, width))
    if max_value != 255:
        image = (image.astype(np.float32) * (255.0 / max_value)).astype(np.uint8)
    return image.astype(np.uint8)


class OccupancyMap:
    def __init__(self, yaml_path: Path):
        info = read_map_yaml(yaml_path)
        image_path = (yaml_path.parent / info["image"]).resolve()
        self.image = read_pgm(image_path)
        self.height, self.width = self.image.shape
        self.resolution = float(info["resolution"])
        self.origin_x = float(info["origin"][0])
        self.origin_y = float(info["origin"][1])
        self.occupied = self.image < 128

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        cx = int(math.floor((x - self.origin_x) / self.resolution))
        cy_from_bottom = int(math.floor((y - self.origin_y) / self.resolution))
        row = self.height - 1 - cy_from_bottom
        if cx < 0 or row < 0 or cx >= self.width or row >= self.height:
            return None
        return cx, row

    def is_occupied_world(self, x: float, y: float) -> bool:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return True
        return bool(self.occupied[cell[1], cell[0]])

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        col, row = cell
        if col < 0 or row < 0 or col >= self.width or row >= self.height:
            return False
        return not bool(self.occupied[row, col])

    def collides_disc(self, x: float, y: float, radius: float) -> bool:
        samples = 16
        if self.is_occupied_world(x, y):
            return True
        for index in range(samples):
            angle = 2.0 * math.pi * index / samples
            if self.is_occupied_world(x + radius * math.cos(angle), y + radius * math.sin(angle)):
                return True
        return False

    def raycast(self, x: float, y: float, yaw: float, max_range: float) -> float:
        step = max(self.resolution * 0.5, 0.03)
        distance = 0.0
        while distance < max_range:
            distance += step
            px = x + distance * math.cos(yaw)
            py = y + distance * math.sin(yaw)
            if self.is_occupied_world(px, py):
                return min(distance, max_range)
        return max_range

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        col, row = cell
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + ((self.height - 1 - row) + 0.5) * self.resolution,
        )

    def build_point_cloud_payload(self, max_points: int) -> dict[str, Any]:
        max_points = max(100, int(max_points))
        free_rows, free_cols = np.nonzero(~self.occupied)
        occupied_rows, occupied_cols = np.nonzero(self.occupied)

        positions: list[float] = []
        colors: list[float] = []

        def add_points(
            rows: np.ndarray,
            cols: np.ndarray,
            *,
            z_values: tuple[float, ...],
            color: tuple[float, float, float],
            budget: int,
        ) -> None:
            if budget <= 0 or len(rows) == 0:
                return
            candidates = len(rows) * len(z_values)
            stride = max(1, int(math.ceil(candidates / max(budget, 1))))
            emitted = 0
            for index, (row, col) in enumerate(zip(rows.tolist(), cols.tolist())):
                for z in z_values:
                    if (index * len(z_values)) % stride != 0:
                        continue
                    if emitted >= budget or len(positions) // 3 >= max_points:
                        return
                    x, y = self.cell_to_world((int(col), int(row)))
                    positions.extend([x, y, z])
                    colors.extend(color)
                    emitted += 1

        occupied_budget = min(max_points // 2, len(occupied_rows) * 3)
        free_budget = max_points - occupied_budget
        add_points(free_rows, free_cols, z_values=(0.015,), color=(0.12, 0.74, 0.42), budget=free_budget)
        add_points(
            occupied_rows,
            occupied_cols,
            z_values=(0.08, 0.32, 0.56),
            color=(0.08, 0.22, 0.86),
            budget=occupied_budget,
        )

        count = len(positions) // 3
        return {
            "frameId": "world",
            "stampMs": int(time.time() * 1000.0),
            "sourcePointCount": int(self.width * self.height),
            "renderedPointCount": count,
            "positions": positions,
            "colors": colors,
        }

    def best_clearance_world(self, x: float, y: float) -> tuple[float, float] | None:
        component = self._free_component_near(x, y)
        if not component:
            return None
        distance = self._obstacle_distance_cells()
        best = max(component, key=lambda cell: int(distance[cell[1], cell[0]]))
        return self.cell_to_world(best)

    def rightmost_world(self, x: float, y: float, min_clearance_m: float | None = None) -> tuple[float, float] | None:
        component = self._free_component_near(x, y)
        if not component:
            return None
        distance = self._obstacle_distance_cells()
        min_clearance = 0.25 if min_clearance_m is None else max(float(min_clearance_m), 0.0)
        min_clearance_cells = max(2, int(math.ceil(min_clearance / max(self.resolution, 1.0e-6))))
        candidates = [cell for cell in component if int(distance[cell[1], cell[0]]) >= min_clearance_cells]
        if not candidates:
            candidates = list(component)
        best = max(candidates, key=lambda cell: (cell[0], int(distance[cell[1], cell[0]])))
        return self.cell_to_world(best)

    def best_free_yaw(self, x: float, y: float) -> float:
        samples = 32
        best_yaw = 0.0
        best_distance = -1.0
        for index in range(samples):
            yaw = -math.pi + 2.0 * math.pi * index / samples
            distance = self.raycast(x, y, yaw, 8.0)
            if distance > best_distance:
                best_distance = distance
                best_yaw = yaw
        return best_yaw

    def _free_component_near(self, x: float, y: float) -> set[tuple[int, int]]:
        start = self.world_to_cell(x, y)
        if start is None or bool(self.occupied[start[1], start[0]]):
            start = self._nearest_free_cell(x, y)
        if start is None:
            return set()
        offsets8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        queue = [start]
        component = {start}
        for col, row in queue:
            for dx, dy in offsets8:
                next_cell = (col + dx, row + dy)
                next_col, next_row = next_cell
                if (
                    next_cell in component
                    or next_col < 0
                    or next_row < 0
                    or next_col >= self.width
                    or next_row >= self.height
                    or bool(self.occupied[next_row, next_col])
                ):
                    continue
                component.add(next_cell)
                queue.append(next_cell)
        return component

    def _nearest_free_cell(self, x: float, y: float) -> tuple[int, int] | None:
        raw = self.world_to_cell(x, y)
        if raw is None:
            col = int(math.floor((x - self.origin_x) / self.resolution))
            row = self.height - 1 - int(math.floor((y - self.origin_y) / self.resolution))
            raw = (min(max(col, 0), self.width - 1), min(max(row, 0), self.height - 1))
        raw_col, raw_row = raw
        for radius in range(0, max(self.width, self.height) + 1):
            for row in range(raw_row - radius, raw_row + radius + 1):
                if row < 0 or row >= self.height:
                    continue
                for col in range(raw_col - radius, raw_col + radius + 1):
                    if col < 0 or col >= self.width:
                        continue
                    if max(abs(col - raw_col), abs(row - raw_row)) != radius:
                        continue
                    if not bool(self.occupied[row, col]):
                        return col, row
        return None

    def _obstacle_distance_cells(self) -> np.ndarray:
        distance = np.full((self.height, self.width), 1_000_000, dtype=np.int32)
        obstacle_rows, obstacle_cols = np.nonzero(self.occupied)
        frontier = list(zip(obstacle_cols.tolist(), obstacle_rows.tolist()))
        for col, row in frontier:
            distance[row, col] = 0
        offsets4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        head = 0
        while head < len(frontier):
            col, row = frontier[head]
            head += 1
            next_distance = int(distance[row, col]) + 1
            for dx, dy in offsets4:
                next_col = col + dx
                next_row = row + dy
                if (
                    next_col < 0
                    or next_row < 0
                    or next_col >= self.width
                    or next_row >= self.height
                    or distance[next_row, next_col] <= next_distance
                ):
                    continue
                distance[next_row, next_col] = next_distance
                frontier.append((next_col, next_row))
        return distance


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def initial_pose_from_report(
    report_path: str,
    occ_map: OccupancyMap,
    start_policy: str,
    *,
    robot_radius: float,
    start_clearance_margin: float,
) -> tuple[float, float, float]:
    if report_path:
        path = Path(report_path)
        if path.is_file():
            try:
                report = json.loads(path.read_text())
                start = report.get("map", {}).get("start_xy")
                if isinstance(start, list) and len(start) >= 2:
                    x, y = float(start[0]), float(start[1])
                    if start_policy == "rightmost":
                        required_clearance = max(0.25, float(robot_radius) + float(start_clearance_margin))
                        rightmost = occ_map.rightmost_world(x, y, min_clearance_m=required_clearance)
                        if rightmost is not None:
                            return rightmost[0], rightmost[1], occ_map.best_free_yaw(rightmost[0], rightmost[1])
                    if start_policy == "clearance":
                        clear = occ_map.best_clearance_world(x, y)
                        if clear is not None:
                            return clear[0], clear[1], 0.0
                    return x, y, 0.0
            except Exception:
                pass
    for row in range(occ_map.height):
        for col in range(occ_map.width):
            if not occ_map.occupied[row, col]:
                x = occ_map.origin_x + (col + 0.5) * occ_map.resolution
                y = occ_map.origin_y + (occ_map.height - 1 - row + 0.5) * occ_map.resolution
                return x, y, 0.0
    return occ_map.origin_x, occ_map.origin_y, 0.0


class Nav2PgmRobotBridge(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("nav2_pgm_robot_bridge")
        self.args = args
        self.map = OccupancyMap(Path(args.map_yaml))
        self.x, self.y, self.yaw = initial_pose_from_report(
            args.validation_report,
            self.map,
            str(args.start_policy),
            robot_radius=float(args.robot_radius),
            start_clearance_margin=float(args.start_clearance_margin),
        )
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.last_cmd_wall = 0.0
        self.last_step_wall = time.monotonic()
        self.last_bridge_wall = 0.0
        self.last_map_cloud_wall = 0.0
        self.map_cloud_payload = self.map.build_point_cloud_payload(int(args.map_cloud_max_points))
        self.trajectory_reset_sent = False
        self.plan: list[tuple[float, float]] = []
        self.plan_cursor = 0
        self.last_plan_wall = 0.0

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 10)
        if str(args.motion_mode) == "plan":
            self.create_subscription(NavPath, "/plan", self._on_plan, 10)
        self.create_timer(1.0 / max(float(args.rate_hz), 1.0), self._step)
        self.create_timer(1.0 / max(float(args.scan_rate_hz), 1.0), self._publish_scan)
        self.get_logger().info(f"Nav2 PGM robot bridge ready at ({self.x:.3f}, {self.y:.3f}, {self.yaw:.3f}).")

    def _on_cmd_vel(self, msg: Twist) -> None:
        self.cmd_v = max(-0.2, min(float(msg.linear.x), float(self.args.max_speed)))
        self.cmd_w = max(-float(self.args.max_yaw_rate), min(float(msg.angular.z), float(self.args.max_yaw_rate)))
        self.last_cmd_wall = time.monotonic()

    def _on_plan(self, msg: NavPath) -> None:
        points = [(float(pose.pose.position.x), float(pose.pose.position.y)) for pose in msg.poses]
        if len(points) >= 2:
            points = self._expand_corner_cutting_diagonals(points)
            if len(points) < 2:
                return
            self.plan = points
            current_cell = self.map.world_to_cell(self.x, self.y)
            current_cell_indices = [
                index for index, point in enumerate(points) if self.map.world_to_cell(point[0], point[1]) == current_cell
            ]
            if current_cell_indices:
                self.plan_cursor = min(max(current_cell_indices) + 1, len(points) - 1)
            else:
                nearest = min(
                    range(len(points)),
                    key=lambda index: (points[index][0] - self.x) ** 2 + (points[index][1] - self.y) ** 2,
                )
                self.plan_cursor = min(nearest + 1, len(points) - 1)
            self.last_plan_wall = time.monotonic()

    def _expand_corner_cutting_diagonals(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        expanded: list[tuple[float, float]] = []
        last_cell: tuple[int, int] | None = None
        for point in points:
            cell = self.map.world_to_cell(point[0], point[1])
            if cell is None:
                expanded.append(point)
                last_cell = None
                continue
            if last_cell is not None:
                dx = cell[0] - last_cell[0]
                dy = cell[1] - last_cell[1]
                if abs(dx) == 1 and abs(dy) == 1:
                    candidates = [(last_cell[0], cell[1]), (cell[0], last_cell[1])]
                    bridge_cell = next((candidate for candidate in candidates if self.map.is_free_cell(candidate)), None)
                    if bridge_cell is not None:
                        expanded.append(self.map.cell_to_world(bridge_cell))
            expanded.append(point)
            last_cell = cell
        return expanded

    def _step(self) -> None:
        now = time.monotonic()
        dt = min(max(now - self.last_step_wall, 0.0), 0.15)
        self.last_step_wall = now
        if str(self.args.motion_mode) == "plan":
            v, w, next_x, next_y, next_yaw = self._step_plan_follow(now, dt)
        else:
            if now - self.last_cmd_wall > 0.8:
                v = 0.0
                w = 0.0
            else:
                v = self.cmd_v
                w = self.cmd_w
            next_yaw = self.yaw + w * dt
            next_x = self.x + v * math.cos(next_yaw) * dt
            next_y = self.y + v * math.sin(next_yaw) * dt
        if not self.map.collides_disc(next_x, next_y, float(self.args.robot_radius)):
            self.x, self.y = next_x, next_y
        self.yaw = math.atan2(math.sin(next_yaw), math.cos(next_yaw))
        self._publish_tf_and_odom(v, w)
        if now - self.last_bridge_wall >= 1.0 / max(float(self.args.bridge_rate_hz), 1.0):
            self.last_bridge_wall = now
            self._publish_bridge_pose()

    def _step_plan_follow(self, now: float, dt: float) -> tuple[float, float, float, float, float]:
        if len(self.plan) < 2 or now - self.last_plan_wall > 120.0:
            return 0.0, 0.0, self.x, self.y, self.yaw
        final_x, final_y = self.plan[-1]
        if math.hypot(final_x - self.x, final_y - self.y) < 0.18:
            return 0.0, 0.0, self.x, self.y, self.yaw

        self.plan_cursor = min(self.plan_cursor, len(self.plan) - 1)
        current_cell = self.map.world_to_cell(self.x, self.y)
        while self.plan_cursor < len(self.plan) - 1:
            target = self.plan[self.plan_cursor]
            target_cell = self.map.world_to_cell(target[0], target[1])
            if target_cell != current_cell or math.hypot(target[0] - self.x, target[1] - self.y) >= 0.16:
                break
            self.plan_cursor += 1

        target_x, target_y = self.plan[self.plan_cursor]
        desired_yaw = math.atan2(target_y - self.y, target_x - self.x)
        yaw_error = math.atan2(math.sin(desired_yaw - self.yaw), math.cos(desired_yaw - self.yaw))
        max_turn = float(self.args.max_yaw_rate) * dt
        next_yaw = self.yaw + max(-max_turn, min(max_turn, yaw_error))
        speed = min(float(self.args.max_speed), max(0.18, self.cmd_v if self.cmd_v > 0.01 else 0.32))
        if abs(yaw_error) > 0.65:
            speed = 0.0
        next_x = self.x + speed * math.cos(next_yaw) * dt
        next_y = self.y + speed * math.sin(next_yaw) * dt
        return speed, yaw_error / max(dt, 1e-3), next_x, next_y, next_yaw

    def _publish_tf_and_odom(self, v: float, w: float) -> None:
        stamp = self.get_clock().now().to_msg()
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = stamp
        map_to_odom.header.frame_id = "map"
        map_to_odom.child_frame_id = "odom"
        map_to_odom.transform.rotation.w = 1.0

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = stamp
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_link"
        odom_to_base.transform.translation.x = self.x
        odom_to_base.transform.translation.y = self.y
        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        odom_to_base.transform.rotation.x = qx
        odom_to_base.transform.rotation.y = qy
        odom_to_base.transform.rotation.z = qz
        odom_to_base.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform([map_to_odom, odom_to_base])

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

    def _publish_scan(self) -> None:
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "base_link"
        beam_count = 181
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (beam_count - 1)
        scan.time_increment = 0.0
        scan.scan_time = 1.0 / max(float(self.args.scan_rate_hz), 1.0)
        scan.range_min = 0.05
        scan.range_max = 8.0
        scan.ranges = [
            self.map.raycast(self.x, self.y, self.yaw + scan.angle_min + i * scan.angle_increment, scan.range_max)
            for i in range(beam_count)
        ]
        self.scan_pub.publish(scan)

    def _publish_bridge_pose(self) -> None:
        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        stamp_ms = int(time.time() * 1000.0)
        payload: dict[str, Any] = {
            "navStatus": {
                "backend": "nav2",
                "motionMode": str(self.args.motion_mode),
                "pose": {"x": self.x, "y": self.y, "yaw": self.yaw},
            },
        }
        if str(self.args.bridge_pose_mode) == "pose-and-cloud":
            payload["resetTrajectory"] = not self.trajectory_reset_sent
            payload["pose"] = {
                "frameId": "world",
                "position": {"x": self.x, "y": self.y, "z": 0.0},
                "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                "stampMs": stamp_ms,
            }
            self.trajectory_reset_sent = True
        now = time.monotonic()
        map_cloud_period = 1.0 / max(float(self.args.map_cloud_rate_hz), 0.1)
        if now - self.last_map_cloud_wall >= map_cloud_period:
            self.last_map_cloud_wall = now
            live_point_cloud = dict(self.map_cloud_payload)
            live_point_cloud["stampMs"] = stamp_ms
            payload["livePointCloud"] = live_point_cloud
        bridge_url = str(self.args.bridge_url or "").rstrip("/")
        if not bridge_url:
            return
        try:
            req = Request(
                f"{bridge_url}/ingest/state",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=0.25):
                pass
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Nav2PgmRobotBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
