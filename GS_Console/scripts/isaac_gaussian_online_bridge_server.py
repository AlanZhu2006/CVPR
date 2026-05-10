#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone bridge for Isaac live state and an optional external Gaussian renderer."
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument(
        "--renderer-url",
        default="",
        help="Optional external renderer bridge base URL, for example http://127.0.0.1:8876",
    )
    parser.add_argument(
        "--mapper-url",
        default="",
        help="Optional online gaussian mapper base URL, for example http://127.0.0.1:8891",
    )
    parser.add_argument("--request-timeout-sec", type=float, default=2.0)
    parser.add_argument("--mjpeg-max-fps", type=float, default=24.0)
    parser.add_argument("--max-trajectory-poses", type=int, default=2048)
    parser.add_argument("--min-trajectory-distance", type=float, default=0.03)
    parser.add_argument("--min-trajectory-dt-ms", type=float, default=80.0)
    parser.add_argument(
        "--manual-render-pose-hold-sec",
        type=float,
        default=2.0,
        help=(
            "After /camera_pose is called by the WebUI, ignore live ingest renderPose "
            "updates for this many seconds so HiFi orbit control does not flicker."
        ),
    )
    parser.add_argument(
        "--live-render-pose-mode",
        choices=("always", "manual-hold", "never"),
        default="manual-hold",
        help=(
            "Controls whether live Isaac ingest renderPose updates are forwarded to the "
            "external renderer. Use 'never' for offline GS-SDF review so Isaac does not "
            "fight the training/video camera path."
        ),
    )
    parser.add_argument(
        "--max-render-pose-distance-from-origin",
        type=float,
        default=0.0,
        help=(
            "Reject manually requested renderer camera poses whose position norm exceeds "
            "this distance. 0 disables the guard."
        ),
    )
    return parser.parse_args()


def _now_wall_sec() -> float:
    return time.time()


def _json_response(
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


def _parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def _decode_base64_jpeg(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("data:"):
        _, _, value = value.partition(",")
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return None


def _normalize_pose_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    position = payload.get("position")
    orientation = payload.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    return {
        "frameId": str(payload.get("frameId") or "world"),
        "position": {
            "x": float(position.get("x", 0.0)),
            "y": float(position.get("y", 0.0)),
            "z": float(position.get("z", 0.0)),
        },
        "orientation": {
            "x": float(orientation.get("x", 0.0)),
            "y": float(orientation.get("y", 0.0)),
            "z": float(orientation.get("z", 0.0)),
            "w": float(orientation.get("w", 1.0)),
        },
        "stampMs": int(payload.get("stampMs")) if payload.get("stampMs") is not None else None,
    }


def _normalize_live_point_cloud_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    positions = payload.get("positions")
    if not isinstance(positions, list) or len(positions) < 3:
        return None
    colors = payload.get("colors")
    if colors is not None and (not isinstance(colors, list) or len(colors) != len(positions)):
        colors = None
    rendered_point_count = payload.get("renderedPointCount")
    if not isinstance(rendered_point_count, int) or rendered_point_count <= 0:
        rendered_point_count = max(len(positions) // 3, 0)
    source_point_count = payload.get("sourcePointCount")
    if not isinstance(source_point_count, int) or source_point_count < rendered_point_count:
        source_point_count = rendered_point_count
    stamp_ms = payload.get("stampMs")
    return {
        "frameId": str(payload.get("frameId") or "world"),
        "stampMs": int(stamp_ms) if isinstance(stamp_ms, (int, float)) else None,
        "renderedPointCount": int(rendered_point_count),
        "sourcePointCount": int(source_point_count),
        "positions": positions,
        "colors": colors,
    }


def _pose_distance(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    if not a or not b:
        return math.inf
    dx = float(a["position"]["x"]) - float(b["position"]["x"])
    dy = float(a["position"]["y"]) - float(b["position"]["y"])
    dz = float(a["position"]["z"]) - float(b["position"]["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _should_append_pose(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    min_distance: float,
    min_dt_ms: float,
) -> bool:
    if previous is None:
        return True
    if _pose_distance(previous, current) >= min_distance:
        return True
    previous_stamp = previous.get("stampMs")
    current_stamp = current.get("stampMs")
    if isinstance(previous_stamp, int) and isinstance(current_stamp, int):
        return current_stamp - previous_stamp >= min_dt_ms
    return False


def _http_json_request(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any] | None:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="ignore")
        if not body:
            return {}
        return json.loads(body)


def _http_get_bytes(url: str, timeout_sec: float) -> bytes:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout_sec) as response:
        return response.read()


def _http_get_json(url: str, timeout_sec: float) -> dict[str, Any]:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8", errors="ignore") or "{}")


def _status_number(status: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(status, dict):
        return None
    value = status.get(key)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class BridgeController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        self.robot_pose: dict[str, Any] | None = None
        self.render_pose: dict[str, Any] | None = None
        self.requested_camera_pose: dict[str, Any] | None = None
        self.requested_camera_pose_wall_sec: float | None = None
        self.trajectory: list[dict[str, Any]] = []
        self.live_point_cloud: dict[str, Any] | None = None
        self.nav_status: dict[str, Any] | None = None
        self.source_image: str | None = None

        self.latest_rgb_jpeg: bytes | None = None
        self.latest_depth_jpeg: bytes | None = None
        self.latest_semantic_jpeg: bytes | None = None
        self.latest_render_jpeg: bytes | None = None
        self.latest_render_preview_jpeg: bytes | None = None
        self.render_source_image: str | None = None
        self.render_stamp_ms: int | None = None

        self.frame_width = 0
        self.frame_height = 0
        self.preview_frame_width = 0
        self.preview_frame_height = 0

        self.ingest_count = 0
        self.frames_received = 0
        self.pose_messages_sent = 0
        self.last_ingest_wall_sec: float | None = None
        self.last_pose_wall_sec: float | None = None
        self.last_renderer_frame_wall_sec: float | None = None
        self.last_renderer_error: str | None = None
        self.last_rejected_camera_pose: dict[str, Any] | None = None
        self.last_rejected_camera_pose_reason: str | None = None
        self.last_rejected_camera_pose_wall_sec: float | None = None
        self.renderer_ready = False
        self.mapper_ready = False
        self.last_mapper_error: str | None = None
        self.last_mapper_update_wall_sec: float | None = None
        self.mapper_point_count = 0

    def ingest_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        pose = _normalize_pose_payload(payload.get("pose") or payload.get("robotPose"))
        render_pose = _normalize_pose_payload(payload.get("renderPose") or payload.get("cameraPose"))
        live_point_cloud = _normalize_live_point_cloud_payload(payload.get("livePointCloud"))
        nav_status = payload.get("navStatus")
        reset_trajectory = bool(payload.get("resetTrajectory"))

        rgb_jpeg = _decode_base64_jpeg(payload.get("rgbJpegBase64"))
        depth_jpeg = _decode_base64_jpeg(payload.get("depthJpegBase64"))
        semantic_jpeg = _decode_base64_jpeg(payload.get("semanticJpegBase64"))

        with self.lock:
            if reset_trajectory:
                self.trajectory = []
            if pose is not None:
                self.robot_pose = pose
                if _should_append_pose(
                    self.trajectory[-1] if self.trajectory else None,
                    pose,
                    min_distance=float(self.args.min_trajectory_distance),
                    min_dt_ms=float(self.args.min_trajectory_dt_ms),
                ):
                    self.trajectory.append(pose)
                    if len(self.trajectory) > int(self.args.max_trajectory_poses):
                        self.trajectory = self.trajectory[-int(self.args.max_trajectory_poses) :]
            if render_pose is not None:
                self.render_pose = render_pose
            if live_point_cloud is not None:
                self.live_point_cloud = live_point_cloud
            if isinstance(nav_status, dict):
                self.nav_status = nav_status
            source_image = payload.get("sourceImage")
            if isinstance(source_image, str) and source_image:
                self.source_image = source_image
            if rgb_jpeg is not None:
                self.latest_rgb_jpeg = rgb_jpeg
            if depth_jpeg is not None:
                self.latest_depth_jpeg = depth_jpeg
            if semantic_jpeg is not None:
                self.latest_semantic_jpeg = semantic_jpeg
            width = payload.get("frameWidth")
            height = payload.get("frameHeight")
            if isinstance(width, int) and width > 0:
                self.frame_width = width
            if isinstance(height, int) and height > 0:
                self.frame_height = height
            self.frames_received += 1 if rgb_jpeg is not None else 0
            self.ingest_count += 1
            self.last_ingest_wall_sec = _now_wall_sec()

        if render_pose is not None and self._should_forward_live_render_pose():
            self._forward_pose_to_renderer(
                render_pose,
                source_image=source_image if isinstance(source_image, str) and source_image else None,
                stamp_ms=render_pose.get("stampMs") if isinstance(render_pose, dict) else None,
            )
        self._forward_state_to_mapper(payload)

        return self.status()

    def publish_camera_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        pose = _normalize_pose_payload(payload)
        if pose is None:
            raise ValueError("camera_pose payload must contain position and orientation.")
        rejection_reason = self._renderer_pose_rejection_reason(pose)
        if rejection_reason is not None:
            with self.lock:
                self.last_rejected_camera_pose = pose
                self.last_rejected_camera_pose_reason = rejection_reason
                self.last_rejected_camera_pose_wall_sec = _now_wall_sec()
            return self.status()
        with self.lock:
            self.requested_camera_pose = pose
            self.requested_camera_pose_wall_sec = _now_wall_sec()
        self._forward_pose_to_renderer(pose)
        return self.status()

    def _should_forward_live_render_pose(self) -> bool:
        mode = str(self.args.live_render_pose_mode)
        if mode == "never":
            return False
        if mode == "always":
            return True
        return not self._manual_render_pose_active()

    def _manual_render_pose_active(self) -> bool:
        hold_sec = max(0.0, float(self.args.manual_render_pose_hold_sec))
        if hold_sec <= 0.0:
            return False
        with self.lock:
            requested_at = self.requested_camera_pose_wall_sec
        return requested_at is not None and (_now_wall_sec() - requested_at) <= hold_sec

    def _renderer_pose_rejection_reason(self, pose: dict[str, Any]) -> str | None:
        position = pose.get("position") if isinstance(pose, dict) else None
        orientation = pose.get("orientation") if isinstance(pose, dict) else None
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            return "pose is missing position or orientation"
        values = [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        ]
        if not all(math.isfinite(value) for value in values):
            return "pose contains a non-finite value"
        max_distance = max(0.0, float(self.args.max_render_pose_distance_from_origin))
        if max_distance <= 0.0:
            return None
        distance = math.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])
        if distance > max_distance:
            return f"position norm {distance:.3f} exceeds {max_distance:.3f}"
        return None

    def _renderer_base_url(self) -> str | None:
        renderer_url = str(self.args.renderer_url or "").strip()
        if not renderer_url:
            return None
        return renderer_url.rstrip("/")

    def _mapper_base_url(self) -> str | None:
        mapper_url = str(self.args.mapper_url or "").strip()
        if not mapper_url:
            return None
        return mapper_url.rstrip("/")

    def _wait_for_renderer_frame_after_pose(
        self,
        renderer_url: str,
        *,
        prior_status: dict[str, Any] | None,
        post_status: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        prior_frames = _status_number(prior_status, "framesReceived")
        pose_wall_sec = _status_number(post_status, "lastPoseWallSec")
        latest_status = post_status
        timeout_sec = max(0.05, min(float(self.args.request_timeout_sec), 2.0))
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            try:
                latest_status = _http_get_json(
                    f"{renderer_url}/status",
                    timeout_sec=min(0.2, timeout_sec),
                )
            except Exception:
                time.sleep(0.02)
                continue

            frames = _status_number(latest_status, "framesReceived")
            frame_wall_sec = _status_number(latest_status, "lastFrameWallSec")
            # The native GS-SDF view consumes /rviz/current_camera_pose
            # asynchronously. Several images after publishing a pose can still
            # be from the previous camera, especially at trajectory wraparound,
            # so prefer alignment over maximum live FPS.
            frame_advanced = prior_frames is None or (frames is not None and frames >= prior_frames + 4.0)
            pose_rendered = pose_wall_sec is None or (
                frame_wall_sec is not None and frame_wall_sec >= pose_wall_sec
            )
            if frame_advanced and pose_rendered:
                return latest_status
            time.sleep(0.02)

        return latest_status

    def _forward_pose_to_renderer(
        self,
        pose: dict[str, Any],
        *,
        source_image: str | None = None,
        stamp_ms: int | None = None,
    ) -> None:
        renderer_url = self._renderer_base_url()
        if renderer_url is None:
            return
        try:
            try:
                prior_status = _http_get_json(
                    f"{renderer_url}/status",
                    timeout_sec=float(self.args.request_timeout_sec),
                )
            except Exception:
                prior_status = None

            post_status = _http_json_request(
                f"{renderer_url}/camera_pose",
                {"position": pose["position"], "orientation": pose["orientation"]},
                timeout_sec=float(self.args.request_timeout_sec),
            )
            status_payload = self._wait_for_renderer_frame_after_pose(
                renderer_url,
                prior_status=prior_status,
                post_status=post_status,
            )
            frame = _http_get_bytes(f"{renderer_url}/frame.jpg", timeout_sec=float(self.args.request_timeout_sec))
            preview: bytes | None = None
            try:
                preview = _http_get_bytes(
                    f"{renderer_url}/frame.preview.jpg",
                    timeout_sec=float(self.args.request_timeout_sec),
                )
            except Exception:
                preview = frame

            with self.lock:
                self.latest_render_jpeg = frame
                self.latest_render_preview_jpeg = preview or frame
                if source_image:
                    self.render_source_image = source_image
                if isinstance(stamp_ms, int):
                    self.render_stamp_ms = stamp_ms
                self.pose_messages_sent += 1
                self.last_pose_wall_sec = _now_wall_sec()
                self.last_renderer_frame_wall_sec = (
                    _status_number(status_payload, "lastFrameWallSec") or self.last_pose_wall_sec
                )
                self.renderer_ready = True
                self.last_renderer_error = None
                if isinstance(status_payload, dict):
                    frame_width = status_payload.get("frameWidth")
                    frame_height = status_payload.get("frameHeight")
                    preview_width = status_payload.get("previewFrameWidth")
                    preview_height = status_payload.get("previewFrameHeight")
                    if isinstance(frame_width, int) and frame_width > 0:
                        self.frame_width = frame_width
                    if isinstance(frame_height, int) and frame_height > 0:
                        self.frame_height = frame_height
                    if isinstance(preview_width, int) and preview_width > 0:
                        self.preview_frame_width = preview_width
                    if isinstance(preview_height, int) and preview_height > 0:
                        self.preview_frame_height = preview_height
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            with self.lock:
                self.renderer_ready = False
                self.last_renderer_error = str(exc)

    def _forward_state_to_mapper(self, payload: dict[str, Any]) -> None:
        mapper_url = self._mapper_base_url()
        if mapper_url is None:
            return
        try:
            status_payload = _http_json_request(
                f"{mapper_url}/ingest/state",
                payload,
                timeout_sec=float(self.args.request_timeout_sec),
            )
            with self.lock:
                self.last_mapper_update_wall_sec = _now_wall_sec()
                self.mapper_ready = bool(status_payload.get("ready")) if isinstance(status_payload, dict) else True
                self.last_mapper_error = None
                if isinstance(status_payload, dict):
                    point_count = status_payload.get("pointCount")
                    if isinstance(point_count, int) and point_count >= 0:
                        self.mapper_point_count = point_count
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            with self.lock:
                self.mapper_ready = False
                self.last_mapper_error = str(exc)

    def get_display_frame(self, preview: bool = False) -> tuple[bytes | None, str]:
        with self.lock:
            if preview and self.latest_render_preview_jpeg is not None:
                return self.latest_render_preview_jpeg, "renderer-preview"
            if self.latest_render_jpeg is not None:
                return self.latest_render_jpeg, "renderer"
            if self.latest_rgb_jpeg is not None:
                return self.latest_rgb_jpeg, "isaac-rgb"
        return None, "none"

    def get_named_frame(self, name: str) -> bytes | None:
        with self.lock:
            if name == "rgb":
                return self.latest_rgb_jpeg
            if name == "depth":
                return self.latest_depth_jpeg
            if name == "semantic":
                return self.latest_semantic_jpeg
        return None

    def status(self) -> dict[str, Any]:
        frame, display_source = self.get_display_frame(preview=False)
        with self.lock:
            stamp_ms = self.robot_pose.get("stampMs") if self.robot_pose else None
            display_source_image = (
                self.render_source_image
                if display_source.startswith("renderer") and self.render_source_image
                else self.source_image
            )
            display_stamp_ms = (
                self.render_stamp_ms
                if display_source.startswith("renderer") and isinstance(self.render_stamp_ms, int)
                else stamp_ms
            )
            return {
                "mode": "isaac-gaussian-online",
                "ready": frame is not None,
                "rendererConfigured": self._renderer_base_url() is not None,
                "rendererReady": self.renderer_ready,
                "rendererError": self.last_renderer_error,
                "mapperConfigured": self._mapper_base_url() is not None,
                "mapperReady": self.mapper_ready,
                "mapperError": self.last_mapper_error,
                "mapperPointCount": self.mapper_point_count,
                "displaySource": display_source,
                "bridgePort": int(self.args.port),
                "imageTopic": "http:/ingest/state",
                "poseTopic": "http:/robot_pose",
                "sourceContainer": "isaac-gaussian-online",
                "image": self._renderer_base_url() or "isaac-rgb",
                "latestFrameStampSec": (float(display_stamp_ms) / 1000.0)
                if isinstance(display_stamp_ms, (int, float))
                else None,
                "frameWidth": self.frame_width,
                "frameHeight": self.frame_height,
                "previewFrameWidth": self.preview_frame_width or self.frame_width,
                "previewFrameHeight": self.preview_frame_height or self.frame_height,
                "framesReceived": self.frames_received,
                "poseMessagesSent": self.pose_messages_sent,
                "lastPoseWallSec": self.last_pose_wall_sec,
                "lastIngestWallSec": self.last_ingest_wall_sec,
                "lastRendererFrameWallSec": self.last_renderer_frame_wall_sec,
                "lastMapperUpdateWallSec": self.last_mapper_update_wall_sec,
                "liveRenderPoseMode": str(self.args.live_render_pose_mode),
                "maxRenderPoseDistanceFromOrigin": float(self.args.max_render_pose_distance_from_origin),
                "trajectoryLength": len(self.trajectory),
                "robotPose": self.robot_pose,
                "requestedCameraPose": self.requested_camera_pose,
                "sourceImage": display_source_image,
                "liveSourceImage": self.source_image,
                "renderSourceImage": self.render_source_image,
                "manualRenderPoseActive": self._manual_render_pose_active(),
                "lastRejectedCameraPose": self.last_rejected_camera_pose,
                "lastRejectedCameraPoseReason": self.last_rejected_camera_pose_reason,
                "lastRejectedCameraPoseWallSec": self.last_rejected_camera_pose_wall_sec,
                "navStatus": self.nav_status,
                "livePointCloudReady": self.live_point_cloud is not None,
                "livePointCloudPointCount": int(self.live_point_cloud.get("renderedPointCount", 0))
                if isinstance(self.live_point_cloud, dict)
                else 0,
                "rgbReady": self.latest_rgb_jpeg is not None,
                "depthReady": self.latest_depth_jpeg is not None,
                "semanticReady": self.latest_semantic_jpeg is not None,
            }


class Handler(BaseHTTPRequestHandler):
    controller: BridgeController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            _json_response(self, self.controller.status(), head_only=True)
            return
        if parsed.path == "/robot_pose":
            _json_response(self, {"pose": self.controller.robot_pose}, head_only=True)
            return
        if parsed.path == "/trajectory.json":
            _json_response(self, {"poses": self.controller.trajectory}, head_only=True)
            return
        if parsed.path == "/live_point_cloud.json":
            _json_response(self, {"pointCloud": self.controller.live_point_cloud}, head_only=True)
            return
        if parsed.path == "/nav/status":
            _json_response(self, {"navStatus": self.controller.nav_status}, head_only=True)
            return
        if parsed.path == "/frame.jpg":
            self._respond_jpeg(preview=False, head_only=True)
            return
        if parsed.path == "/frame.preview.jpg":
            self._respond_jpeg(preview=True, head_only=True)
            return
        if parsed.path == "/frame/rgb.jpg":
            self._respond_named_frame("rgb", head_only=True)
            return
        if parsed.path == "/frame/depth.jpg":
            self._respond_named_frame("depth", head_only=True)
            return
        if parsed.path == "/frame/semantic.jpg":
            self._respond_named_frame("semantic", head_only=True)
            return
        _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND, head_only=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            _json_response(self, self.controller.status())
            return
        if parsed.path == "/robot_pose":
            _json_response(self, {"pose": self.controller.robot_pose})
            return
        if parsed.path == "/trajectory.json":
            _json_response(self, {"poses": self.controller.trajectory})
            return
        if parsed.path == "/live_point_cloud.json":
            _json_response(self, {"pointCloud": self.controller.live_point_cloud})
            return
        if parsed.path == "/nav/status":
            _json_response(self, {"navStatus": self.controller.nav_status})
            return
        if parsed.path == "/frame.jpg":
            self._respond_jpeg(preview=False)
            return
        if parsed.path == "/frame.preview.jpg":
            self._respond_jpeg(preview=True)
            return
        if parsed.path == "/frame/rgb.jpg":
            self._respond_named_frame("rgb")
            return
        if parsed.path == "/frame/depth.jpg":
            self._respond_named_frame("depth")
            return
        if parsed.path == "/frame/semantic.jpg":
            self._respond_named_frame("semantic")
            return
        if parsed.path == "/frame.mjpeg":
            self._respond_mjpeg(preview=False)
            return
        if parsed.path == "/frame.preview.mjpeg":
            self._respond_mjpeg(preview=True)
            return
        _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = _parse_json_body(self)
            if parsed.path == "/ingest/state":
                _json_response(self, self.controller.ingest_state(payload))
                return
            if parsed.path == "/camera_pose":
                _json_response(self, self.controller.publish_camera_pose(payload))
                return
            _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError as exc:
            _json_response(self, {"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)

    def _respond_named_frame(self, name: str, *, head_only: bool = False) -> None:
        frame = self.controller.get_named_frame(name)
        if frame is None:
            _json_response(self, {"error": f"{name} frame unavailable"}, HTTPStatus.NOT_FOUND, head_only=head_only)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        if not head_only:
            self.wfile.write(frame)

    def _respond_jpeg(self, preview: bool, *, head_only: bool = False) -> None:
        frame, _ = self.controller.get_display_frame(preview=preview)
        if frame is None:
            _json_response(self, {"error": "frame unavailable"}, HTTPStatus.NOT_FOUND, head_only=head_only)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        if not head_only:
            self.wfile.write(frame)

    def _respond_mjpeg(self, preview: bool) -> None:
        boundary = "frame"
        max_fps = max(float(self.controller.args.mjpeg_max_fps), 1.0)
        min_interval = 1.0 / max_fps
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.end_headers()

        last_sent_wall_sec = 0.0
        while not self.controller.stop_event.is_set():
            frame, _ = self.controller.get_display_frame(preview=preview)
            if frame is not None:
                now = _now_wall_sec()
                sleep_needed = min_interval - (now - last_sent_wall_sec)
                if sleep_needed > 0:
                    if self.controller.stop_event.wait(sleep_needed):
                        return
                try:
                    self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                last_sent_wall_sec = _now_wall_sec()
            if self.controller.stop_event.wait(0.05):
                return

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    args = parse_args()
    controller = BridgeController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", int(args.port)), Handler)
    try:
        print(f"isaac_gaussian_online_bridge listening on http://0.0.0.0:{args.port}")
        if args.renderer_url:
            print(f"Forwarding render poses to {args.renderer_url}")
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
