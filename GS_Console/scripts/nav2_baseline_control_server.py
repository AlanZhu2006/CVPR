#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP control API for the Nav2 GS-SDF baseline.")
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--map-yaml", default="")
    parser.add_argument("--start-policy", choices=["validation", "clearance", "rightmost"], default="validation")
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument("--start-clearance-margin", type=float, default=0.08)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8890")
    parser.add_argument("--action-timeout-sec", type=float, default=90.0)
    return parser.parse_args()


def now_ms() -> int:
    return int(time.time() * 1000.0)


def yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, Any],
    status: HTTPStatus = HTTPStatus.OK,
    *,
    head_only: bool = False,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    if not head_only:
        handler.wfile.write(body)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def normalize_goal_world(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, list | tuple) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def semantic_goal_label(index: int) -> str:
    labels = [
        "near_start",
        "inspection_point_a",
        "inspection_point_b",
        "corridor_left",
        "corridor_forward",
        "return_check",
    ]
    return labels[index] if index < len(labels) else f"inspection_point_{index + 1:02d}"


def load_semantic_goals(validation_report: Path) -> list[dict[str, Any]]:
    if not validation_report.is_file():
        return []
    try:
        report = json.loads(validation_report.read_text())
    except Exception:
        return []
    goals: list[dict[str, Any]] = []
    for goal in report.get("map", {}).get("goals", []):
        xy = normalize_goal_world(goal[:2] if isinstance(goal, list) else None)
        if xy is None:
            continue
        goals.append(
            {
                "label": semantic_goal_label(len(goals)),
                "cell": None,
                "world": {"x": xy[0], "y": xy[1]},
                "source": validation_report.name,
                "lastResult": None,
                "lastReason": None,
            }
        )
        if len(goals) >= 8:
            break
    return goals


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


def read_pgm(path: Path) -> tuple[int, int, list[int]]:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise RuntimeError(f"Unsupported PGM map: {path}")
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = [int(v) for v in line.split()[:2]]
        max_value = int(handle.readline())
        if magic == b"P5":
            raw_values = list(handle.read(width * height))
        else:
            raw_values = [int(v) for v in handle.read().split()]
    if max_value != 255:
        raw_values = [int(round(value * 255.0 / max_value)) for value in raw_values]
    return width, height, raw_values


class OccupancyMap:
    def __init__(self, yaml_path: Path):
        info = read_map_yaml(yaml_path)
        image_path = (yaml_path.parent / info["image"]).resolve()
        self.width, self.height, self.pixels = read_pgm(image_path)
        self.resolution = float(info["resolution"])
        self.origin_x = float(info["origin"][0])
        self.origin_y = float(info["origin"][1])
        self._obstacle_distance_cache: dict[tuple[int, int], int] | None = None

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        cx = int(math.floor((x - self.origin_x) / self.resolution))
        cy = int(math.floor((y - self.origin_y) / self.resolution))
        if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
            return None
        return cx, cy

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (
            self.origin_x + (cell[0] + 0.5) * self.resolution,
            self.origin_y + (cell[1] + 0.5) * self.resolution,
        )

    def is_free_cell(self, cell: tuple[int, int]) -> bool:
        cx, cy = cell
        if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
            return False
        image_row = self.height - 1 - cy
        return self.pixels[image_row * self.width + cx] >= 128

    def nearest_free_cell(
        self,
        x: float,
        y: float,
        reachable: set[tuple[int, int]] | None = None,
        max_radius: int | None = None,
        min_clearance_m: float = 0.0,
    ) -> tuple[int, int] | None:
        raw = self.world_to_cell(x, y)
        if raw is None:
            raw = (
                min(max(int(math.floor((x - self.origin_x) / self.resolution)), 0), self.width - 1),
                min(max(int(math.floor((y - self.origin_y) / self.resolution)), 0), self.height - 1),
            )
        distance = self.obstacle_distance_cells() if min_clearance_m > 0.0 else None
        min_clearance_cells = int(math.ceil(min_clearance_m / max(self.resolution, 1.0e-6)))
        limit = max_radius or max(self.width, self.height)
        for radius in range(0, limit + 1):
            for cy in range(raw[1] - radius, raw[1] + radius + 1):
                if cy < 0 or cy >= self.height:
                    continue
                for cx in range(raw[0] - radius, raw[0] + radius + 1):
                    if cx < 0 or cx >= self.width:
                        continue
                    if max(abs(cx - raw[0]), abs(cy - raw[1])) != radius:
                        continue
                    cell = (cx, cy)
                    has_clearance = distance is None or distance.get(cell, 0) >= min_clearance_cells
                    if self.is_free_cell(cell) and has_clearance and (reachable is None or cell in reachable):
                        return cell
        return None

    def reachable_from(self, start_xy: tuple[float, float] | None) -> set[tuple[int, int]]:
        start = self.nearest_free_cell(start_xy[0], start_xy[1]) if start_xy else None
        if start is None:
            return set()
        queue: deque[tuple[int, int]] = deque([start])
        visited: set[tuple[int, int]] = {start}
        offsets = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in offsets:
                cell = (cx + dx, cy + dy)
                if cell in visited or not self.is_free_cell(cell):
                    continue
                visited.add(cell)
                queue.append(cell)
        return visited

    def best_clearance_world(self, start_xy: tuple[float, float] | None) -> tuple[float, float] | None:
        component = self.reachable_from(start_xy)
        if not component:
            return None
        distance = self.obstacle_distance_cells()
        best = max(component, key=lambda cell: distance.get(cell, 0))
        return self.cell_to_world(best)

    def rightmost_world(
        self,
        start_xy: tuple[float, float] | None,
        min_clearance_m: float | None = None,
    ) -> tuple[float, float] | None:
        component = self.reachable_from(start_xy)
        if not component:
            return None
        distance = self.obstacle_distance_cells()
        min_clearance = 0.25 if min_clearance_m is None else max(float(min_clearance_m), 0.0)
        min_clearance_cells = max(2, int(math.ceil(min_clearance / max(self.resolution, 1.0e-6))))
        candidates = [cell for cell in component if distance.get(cell, 0) >= min_clearance_cells]
        if not candidates:
            candidates = list(component)
        best = max(candidates, key=lambda cell: (cell[0], distance.get(cell, 0)))
        return self.cell_to_world(best)

    def obstacle_distance_cells(self) -> dict[tuple[int, int], int]:
        if self._obstacle_distance_cache is not None:
            return self._obstacle_distance_cache
        distance: dict[tuple[int, int], int] = {}
        queue: deque[tuple[int, int]] = deque()
        for cy in range(self.height):
            for cx in range(self.width):
                cell = (cx, cy)
                if self.is_free_cell(cell):
                    continue
                distance[cell] = 0
                queue.append(cell)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            cx, cy = queue.popleft()
            next_distance = distance[(cx, cy)] + 1
            for dx, dy in offsets:
                cell = (cx + dx, cy + dy)
                if cell[0] < 0 or cell[1] < 0 or cell[0] >= self.width or cell[1] >= self.height:
                    continue
                if cell in distance and distance[cell] <= next_distance:
                    continue
                distance[cell] = next_distance
                queue.append(cell)
        self._obstacle_distance_cache = distance
        return distance


def load_start_xy(validation_report: Path) -> tuple[float, float] | None:
    if not validation_report.is_file():
        return None
    try:
        report = json.loads(validation_report.read_text())
        start = report.get("map", {}).get("start_xy")
        if isinstance(start, list) and len(start) >= 2:
            return float(start[0]), float(start[1])
    except Exception:
        return None
    return None


class Nav2BaselineController(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("nav2_baseline_control_server")
        self.args = args
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.lock = threading.RLock()
        self.goal_handle = None
        self.result_future: Future | None = None
        self.goal_pending = False
        self.current_task: dict[str, Any] | None = None
        self.started_at_ms: int | None = None
        self.last_error: str | None = None
        self.last_report: dict[str, Any] | None = None
        self.feedback: dict[str, Any] | None = None
        validation_report = Path(args.validation_report)
        map_yaml = Path(args.map_yaml) if args.map_yaml else None
        self.occupancy_map = OccupancyMap(map_yaml) if map_yaml and map_yaml.is_file() else None
        start_xy = load_start_xy(validation_report)
        if self.occupancy_map and args.start_policy == "rightmost":
            required_clearance = max(0.25, float(args.robot_radius) + float(args.start_clearance_margin))
            start_xy = self.occupancy_map.rightmost_world(start_xy, required_clearance) or start_xy
        elif self.occupancy_map and args.start_policy == "clearance":
            start_xy = self.occupancy_map.best_clearance_world(start_xy) or start_xy
        self.reachable_cells = self.occupancy_map.reachable_from(start_xy) if self.occupancy_map else set()
        self.semantic_goals = self._snap_semantic_goals(load_semantic_goals(validation_report))

    def status(self) -> dict[str, Any]:
        with self.lock:
            running = self.goal_pending or (self.result_future is not None and not self.result_future.done())
            return {
                "mode": "nav2-baseline",
                "ready": self.action_client.server_is_ready(),
                "running": running,
                "returnCode": None if running else 0,
                "currentTask": self.current_task,
                "startedAtMs": self.started_at_ms,
                "lastReportPath": None,
                "lastLogPath": None,
                "lastError": self.last_error,
                "semanticGoals": self.semantic_goals,
                "lastReport": self.last_report,
                "feedback": self.feedback,
            }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.result_future is not None and not self.result_future.done():
                raise RuntimeError("A Nav2 goal is already running. Stop it first.")

        goal_xy = normalize_goal_world(payload.get("goalWorld"))
        goal_label = str(payload.get("goalLabel") or payload.get("semanticGoal") or "")
        if goal_xy is None:
            semantic_goal = str(payload.get("semanticGoal") or "")
            match = next((item for item in self.semantic_goals if item["label"] == semantic_goal), None)
            if match is None and self.semantic_goals:
                match = self.semantic_goals[0]
            if match and match.get("world"):
                goal_xy = (float(match["world"]["x"]), float(match["world"]["y"]))
                goal_label = str(match["label"])
        if goal_xy is None:
            raise RuntimeError("No goalWorld or semantic goal is available for Nav2.")

        if not self.action_client.wait_for_server(timeout_sec=8.0):
            raise RuntimeError("Nav2 navigate_to_pose action server is not ready.")

        nav_goal = self._snap_goal(goal_xy)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = nav_goal["world"]["x"]
        goal_msg.pose.pose.position.y = nav_goal["world"]["y"]
        qx, qy, qz, qw = yaw_to_quat(float(payload.get("yaw") or 0.0))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        task = {
            "command": str(payload.get("command") or "start_pointgoal"),
            "goalLabel": goal_label or f"nav2_goal_{goal_xy[0]:.2f}_{goal_xy[1]:.2f}",
            "semanticGoal": payload.get("semanticGoal"),
            "goalCell": nav_goal["cell"],
            "goalWorld": nav_goal["world"],
            "requestedGoalWorld": {"x": goal_xy[0], "y": goal_xy[1]},
            "snapped": nav_goal["snapped"],
            "reportPath": None,
            "logPath": None,
        }
        with self.lock:
            self.current_task = task
            self.started_at_ms = now_ms()
            self.last_error = None
            self.feedback = None
            self.last_report = None
            self.goal_pending = True
        self._publish_bridge_nav_status("running", "accepted")

        send_future = self.action_client.send_goal_async(goal_msg, feedback_callback=self._feedback_cb)
        send_future.add_done_callback(self._goal_response_cb)
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self.lock:
            goal_handle = self.goal_handle
        if goal_handle is not None:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                cancel_future.add_done_callback(lambda _future: self._publish_bridge_nav_status("canceled", "stop_requested"))
            except Exception as exc:
                with self.lock:
                    self.last_error = str(exc)
        return self.status()

    def _goal_response_cb(self, future: Future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            with self.lock:
                self.goal_pending = False
                self.last_error = str(exc)
                self.last_report = self._report("failed", str(exc))
            self._publish_bridge_nav_status("failed", str(exc))
            return
        if not goal_handle.accepted:
            with self.lock:
                self.goal_pending = False
                self.last_error = "Nav2 rejected the goal."
                self.last_report = self._report("failed", "goal_rejected")
            self._publish_bridge_nav_status("failed", "goal_rejected")
            return
        with self.lock:
            self.goal_handle = goal_handle
            self.goal_pending = False
            self.result_future = goal_handle.get_result_async()
            self.result_future.add_done_callback(self._result_cb)

    def _feedback_cb(self, feedback_msg: Any) -> None:
        feedback = feedback_msg.feedback
        with self.lock:
            self.feedback = {
                "distanceRemaining": float(getattr(feedback, "distance_remaining", 0.0)),
                "navigationTimeSec": float(getattr(getattr(feedback, "navigation_time", None), "sec", 0)),
                "estimatedTimeRemainingSec": float(getattr(getattr(feedback, "estimated_time_remaining", None), "sec", 0)),
            }
        self._publish_bridge_nav_status("running", "feedback")

    def _result_cb(self, future: Future) -> None:
        try:
            wrapped = future.result()
            status_code = int(getattr(wrapped, "status", -1))
            result = "success" if status_code == 4 else "failed"
            reason = f"nav2_status_{status_code}"
        except Exception as exc:
            result = "failed"
            reason = str(exc)
        with self.lock:
            self.goal_handle = None
            self.result_future = None
            self.goal_pending = False
            self.last_report = self._report(result, reason)
            self.last_error = None if result == "success" else reason
        self._publish_bridge_nav_status(result, reason)

    def _report(self, result: str, reason: str) -> dict[str, Any]:
        task = self.current_task or {}
        episode = {
            "episode": 0,
            "goal_xy": [
                float((task.get("goalWorld") or {}).get("x", 0.0)),
                float((task.get("goalWorld") or {}).get("y", 0.0)),
                0.0,
            ],
            "goal_cell": task.get("goalCell"),
            "result": result,
            "reason": reason,
            "duration_sec": max(0.0, (now_ms() - (self.started_at_ms or now_ms())) / 1000.0),
        }
        return {
            "counts": {"success": 1 if result == "success" else 0, "failed": 0 if result == "success" else 1},
            "episodeCount": 1,
            "goalSource": "explicit_goal",
            "episodes": [episode],
        }

    def _snap_semantic_goals(self, goals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapped_goals: list[dict[str, Any]] = []
        for goal in goals:
            world = goal.get("world")
            if not world:
                snapped_goals.append(goal)
                continue
            try:
                nav_goal = self._snap_goal((float(world["x"]), float(world["y"])))
                snapped_goals.append(
                    {
                        **goal,
                        "world": nav_goal["world"],
                        "cell": nav_goal["cell"],
                        "requestedWorld": world if nav_goal["snapped"] else None,
                        "snapped": nav_goal["snapped"],
                    }
                )
            except Exception:
                snapped_goals.append(goal)
        return snapped_goals

    def _snap_goal(self, goal_xy: tuple[float, float]) -> dict[str, Any]:
        if not self.occupancy_map:
            return {
                "world": {"x": goal_xy[0], "y": goal_xy[1]},
                "cell": None,
                "snapped": False,
            }
        reachable = self.reachable_cells if self.reachable_cells else None
        required_clearance = max(0.25, float(self.args.robot_radius) + float(self.args.start_clearance_margin))
        cell = self.occupancy_map.nearest_free_cell(
            goal_xy[0],
            goal_xy[1],
            reachable,
            min_clearance_m=required_clearance,
        )
        if cell is None:
            raise RuntimeError("No reachable collision-safe cell was found near the requested Nav2 goal.")
        snapped_xy = self.occupancy_map.cell_to_world(cell)
        snapped = math.hypot(snapped_xy[0] - goal_xy[0], snapped_xy[1] - goal_xy[1]) > (
            self.occupancy_map.resolution * 0.25
        )
        return {
            "world": {"x": snapped_xy[0], "y": snapped_xy[1]},
            "cell": [cell[0], cell[1]],
            "snapped": snapped,
        }

    def _publish_bridge_nav_status(self, result: str, reason: str) -> None:
        task = self.current_task or {}
        nav_status = {
            "mode": "nav2-baseline",
            "taskType": "point_goal",
            "goalLabel": task.get("goalLabel"),
            "semanticGoal": task.get("semanticGoal"),
            "episodeIndex": 0,
            "episodeCount": 1,
            "goal": task.get("goalWorld"),
            "goalCell": task.get("goalCell"),
            "result": result,
            "reason": reason,
            "running": result == "running",
            "stampMs": now_ms(),
        }
        feedback = self.feedback
        if feedback:
            nav_status["distanceToGoal"] = feedback.get("distanceRemaining")
        bridge_url = str(self.args.bridge_url or "").rstrip("/")
        if not bridge_url:
            return
        try:
            req = Request(
                f"{bridge_url}/ingest/state",
                data=json.dumps({"navStatus": nav_status}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=0.5):
                pass
        except Exception:
            pass


def make_handler(controller: Nav2BaselineController):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_OPTIONS(self) -> None:
            json_response(self, {}, head_only=True)

        def do_HEAD(self) -> None:
            if self.path == "/status":
                json_response(self, controller.status(), head_only=True)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_GET(self) -> None:
            if self.path == "/status":
                json_response(self, controller.status())
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = parse_json_body(self)
                command = str(payload.get("command") or "start_pointgoal")
                if command == "stop":
                    status = controller.stop()
                else:
                    status = controller.start(payload)
                json_response(self, status)
            except Exception as exc:
                with controller.lock:
                    controller.last_error = str(exc)
                json_response(self, {"error": str(exc), "status": controller.status()}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> None:
    args = parse_args()
    rclpy.init()
    controller = Nav2BaselineController(args)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), make_handler(controller))
    print(f"nav2-baseline ready: http://127.0.0.1:{args.port}/status")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        executor.shutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
