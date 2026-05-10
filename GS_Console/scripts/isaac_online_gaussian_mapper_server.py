#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


C0 = 0.28209479177387814
GAUSSIAN_VERTEX_STRUCT = struct.Struct("<17f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap online Gaussian mapper for Isaac HTTP bridge payloads."
    )
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument(
        "--output-dir",
        default="/home/chatsign/gs-sdf/runtime/online-gaussian/isaac-online",
        help="Directory for generated live gaussian artifacts.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=24000,
        help="Maximum number of live cloud points converted into the gaussian map.",
    )
    parser.add_argument(
        "--gaussian-scale",
        type=float,
        default=0.045,
        help="Meters; isotropic scale assigned to each live gaussian.",
    )
    parser.add_argument(
        "--gaussian-alpha",
        type=float,
        default=0.92,
        help="Alpha converted into gaussian opacity logit.",
    )
    parser.add_argument(
        "--min-update-interval-sec",
        type=float,
        default=0.5,
        help="Minimum wall time between live gaussian rewrites.",
    )
    parser.add_argument(
        "--stale-after-sec",
        type=float,
        default=8.0,
        help="Status readiness timeout for generated gaussian artifacts.",
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
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
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
    if not isinstance(positions, list) or len(positions) < 3 or len(positions) % 3 != 0:
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


def _logit(alpha: float) -> float:
    clamped = min(max(alpha, 1.0e-4), 1.0 - 1.0e-4)
    return math.log(clamped / (1.0 - clamped))


class MapperController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()

        self.output_dir = os.path.abspath(str(args.output_dir))
        self.gaussian_dir = os.path.join(self.output_dir, "gaussian")
        self.gaussian_path = os.path.join(self.gaussian_dir, "live.gs.ply")

        self.frames_received = 0
        self.last_ingest_wall_sec: float | None = None
        self.last_write_wall_sec: float | None = None
        self.last_error: str | None = None
        self.latest_frame_stamp_sec: float | None = None
        self.latest_point_count = 0
        self.latest_source_point_count = 0
        self.latest_robot_pose: dict[str, Any] | None = None
        self.latest_render_pose: dict[str, Any] | None = None
        self.latest_rgb_jpeg: bytes | None = None
        self.latest_depth_jpeg: bytes | None = None
        self.latest_semantic_jpeg: bytes | None = None
        self.gaussian_revision = 0
        self.gaussian_source: dict[str, Any] | None = None

    def ingest_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        live_point_cloud = _normalize_live_point_cloud_payload(payload.get("livePointCloud"))
        robot_pose = _normalize_pose_payload(payload.get("pose") or payload.get("robotPose"))
        render_pose = _normalize_pose_payload(payload.get("renderPose") or payload.get("cameraPose"))
        rgb_jpeg = _decode_base64_jpeg(payload.get("rgbJpegBase64"))
        depth_jpeg = _decode_base64_jpeg(payload.get("depthJpegBase64"))
        semantic_jpeg = _decode_base64_jpeg(payload.get("semanticJpegBase64"))
        now = _now_wall_sec()

        with self.lock:
            self.frames_received += 1
            self.last_ingest_wall_sec = now
            self.latest_robot_pose = robot_pose or self.latest_robot_pose
            self.latest_render_pose = render_pose or self.latest_render_pose
            self.latest_rgb_jpeg = rgb_jpeg or self.latest_rgb_jpeg
            self.latest_depth_jpeg = depth_jpeg or self.latest_depth_jpeg
            self.latest_semantic_jpeg = semantic_jpeg or self.latest_semantic_jpeg
            if live_point_cloud is not None and live_point_cloud.get("stampMs") is not None:
                self.latest_frame_stamp_sec = float(live_point_cloud["stampMs"]) / 1000.0

        if live_point_cloud is not None:
            should_write = False
            with self.lock:
                if self.last_write_wall_sec is None:
                    should_write = True
                elif now - self.last_write_wall_sec >= float(self.args.min_update_interval_sec):
                    should_write = True
            if should_write:
                try:
                    self._write_live_gaussian(live_point_cloud)
                except Exception as exc:
                    with self.lock:
                        self.last_error = str(exc)

        return self.status()

    def _write_live_gaussian(self, live_point_cloud: dict[str, Any]) -> None:
        positions = live_point_cloud["positions"]
        colors = live_point_cloud.get("colors")
        total_points = len(positions) // 3
        if total_points <= 0:
            raise ValueError("livePointCloud contains no points.")

        stride = max(1, int(math.ceil(float(total_points) / float(max(int(self.args.max_points), 1)))))
        gaussian_scale_log = math.log(max(float(self.args.gaussian_scale), 1.0e-4))
        gaussian_opacity = _logit(float(self.args.gaussian_alpha))

        os.makedirs(self.gaussian_dir, exist_ok=True)

        fd, temp_path = tempfile.mkstemp(prefix="live_gaussian_", suffix=".ply", dir=self.gaussian_dir)
        selected_points = 0
        try:
            with os.fdopen(fd, "wb") as handle:
                estimated_points = (total_points + stride - 1) // stride
                header_lines = [
                    "ply",
                    "format binary_little_endian 1.0",
                    f"element vertex {estimated_points}",
                    "property float x",
                    "property float y",
                    "property float z",
                    "property float nx",
                    "property float ny",
                    "property float nz",
                    "property float f_dc_0",
                    "property float f_dc_1",
                    "property float f_dc_2",
                    "property float opacity",
                    "property float scale_0",
                    "property float scale_1",
                    "property float scale_2",
                    "property float rot_0",
                    "property float rot_1",
                    "property float rot_2",
                    "property float rot_3",
                    "end_header",
                ]
                handle.write(("\n".join(header_lines) + "\n").encode("ascii"))

                for point_index in range(0, total_points, stride):
                    offset = point_index * 3
                    x = float(positions[offset])
                    y = float(positions[offset + 1])
                    z = float(positions[offset + 2])

                    if colors is None:
                        red = green = blue = 0.75
                    else:
                        red = min(max(float(colors[offset]) / 255.0, 0.0), 1.0)
                        green = min(max(float(colors[offset + 1]) / 255.0, 0.0), 1.0)
                        blue = min(max(float(colors[offset + 2]) / 255.0, 0.0), 1.0)

                    f_dc_0 = (red - 0.5) / C0
                    f_dc_1 = (green - 0.5) / C0
                    f_dc_2 = (blue - 0.5) / C0

                    handle.write(
                        GAUSSIAN_VERTEX_STRUCT.pack(
                            x,
                            y,
                            z,
                            0.0,
                            0.0,
                            0.0,
                            f_dc_0,
                            f_dc_1,
                            f_dc_2,
                            gaussian_opacity,
                            gaussian_scale_log,
                            gaussian_scale_log,
                            gaussian_scale_log,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                        )
                    )
                    selected_points += 1

            if selected_points != estimated_points:
                raise RuntimeError(
                    f"Gaussian point count mismatch while writing live map: expected {estimated_points}, got {selected_points}"
                )

            os.replace(temp_path, self.gaussian_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

        revision = int(_now_wall_sec() * 1000.0)
        gaussian_source = {
            "format": "gs-ply",
            "url": "/__isaac_gaussian_mapper/gaussian/live.gs.ply",
            "shDegree": 0,
            "variant": "live",
            "label": "Live Mapper",
        }
        with self.lock:
            self.gaussian_revision = revision
            self.gaussian_source = gaussian_source
            self.last_write_wall_sec = _now_wall_sec()
            self.latest_point_count = selected_points
            self.latest_source_point_count = int(live_point_cloud.get("sourcePointCount", total_points))
            self.last_error = None

    def build_gaussian_manifest_patch(self) -> dict[str, Any]:
        with self.lock:
            gaussian_source = self.gaussian_source
            ready = gaussian_source is not None and os.path.exists(self.gaussian_path)
            assets = {"gaussian": "ready" if ready else "processing"}
            if not ready:
                return {"assets": assets}
            return {
                "assets": assets,
                "gaussian": gaussian_source,
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            ready = (
                self.gaussian_source is not None
                and self.last_write_wall_sec is not None
                and (_now_wall_sec() - self.last_write_wall_sec) <= float(self.args.stale_after_sec)
            )
            return {
                "mode": "isaac-online-gaussian-mapper",
                "ready": ready,
                "gaussianReady": self.gaussian_source is not None,
                "gaussianRevision": self.gaussian_revision or None,
                "gaussianUrl": self.gaussian_source["url"] if self.gaussian_source else None,
                "framesReceived": self.frames_received,
                "pointCount": self.latest_point_count,
                "sourcePointCount": self.latest_source_point_count,
                "latestFrameStampSec": self.latest_frame_stamp_sec,
                "lastIngestWallSec": self.last_ingest_wall_sec,
                "lastWriteWallSec": self.last_write_wall_sec,
                "lastError": self.last_error,
                "rgbReady": self.latest_rgb_jpeg is not None,
                "depthReady": self.latest_depth_jpeg is not None,
                "semanticReady": self.latest_semantic_jpeg is not None,
                "robotPose": self.latest_robot_pose,
                "renderPose": self.latest_render_pose,
            }

    def get_latest_jpeg(self, name: str) -> bytes | None:
        with self.lock:
            if name == "rgb":
                return self.latest_rgb_jpeg
            if name == "depth":
                return self.latest_depth_jpeg
            if name == "semantic":
                return self.latest_semantic_jpeg
        return None


class Handler(BaseHTTPRequestHandler):
    controller: MapperController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_GET(self) -> None:
        self._dispatch(head_only=False)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = _parse_json_body(self)
            if parsed.path in {"/ingest/state", "/ingest/frame"}:
                _json_response(self, self.controller.ingest_state(payload))
                return
            _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            _json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError as exc:
            _json_response(self, {"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)

    def _dispatch(self, *, head_only: bool) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            _json_response(self, self.controller.status(), head_only=head_only)
            return
        if parsed.path == "/gaussian_manifest.json":
            _json_response(self, self.controller.build_gaussian_manifest_patch(), head_only=head_only)
            return
        if parsed.path == "/gaussian/live.gs.ply":
            self._respond_file(self.controller.gaussian_path, "application/octet-stream", head_only=head_only)
            return
        if parsed.path == "/frame/rgb.jpg":
            self._respond_jpeg("rgb", head_only=head_only)
            return
        if parsed.path == "/frame/depth.jpg":
            self._respond_jpeg("depth", head_only=head_only)
            return
        if parsed.path == "/frame/semantic.jpg":
            self._respond_jpeg("semantic", head_only=head_only)
            return
        _json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND, head_only=head_only)

    def _respond_file(self, path: str, content_type: str, *, head_only: bool) -> None:
        if not os.path.exists(path):
            _json_response(self, {"error": "file unavailable"}, HTTPStatus.NOT_FOUND, head_only=head_only)
            return
        body = open(path, "rb").read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _respond_jpeg(self, name: str, *, head_only: bool) -> None:
        frame = self.controller.get_latest_jpeg(name)
        if frame is None:
            _json_response(self, {"error": f"{name} frame unavailable"}, HTTPStatus.NOT_FOUND, head_only=head_only)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        if not head_only:
            self.wfile.write(frame)


def main() -> None:
    args = parse_args()
    controller = MapperController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", int(args.port)), Handler)
    server.daemon_threads = True
    print(f"isaac_online_gaussian_mapper listening on http://0.0.0.0:{args.port}")
    print(f"Output dir: {controller.output_dir}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()
