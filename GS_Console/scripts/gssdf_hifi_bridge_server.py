#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Any

import cv2
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image

from fastlivo_direct_playback_server import LatestImage, decode_image, encode_jpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose native GS-SDF render frames over HTTP and accept camera poses."
    )
    parser.add_argument("--port", type=int, default=8876)
    parser.add_argument("--image-topic", default="/neural_mapping/rgb")
    parser.add_argument("--pose-topic", default="/rviz/current_camera_pose")
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--preview-jpeg-quality", type=int, default=84)
    parser.add_argument("--preview-scale", type=float, default=0.5)
    parser.add_argument("--mjpeg-max-fps", type=float, default=45.0)
    parser.add_argument("--source-container", default="gssdf-view")
    parser.add_argument("--image-name", default="gs_sdf_img:latest")
    return parser.parse_args()


def build_preview_image(image: LatestImage, scale: float) -> LatestImage:
    if scale >= 0.999:
        return image

    preview_width = max(1, int(round(image.width * scale)))
    preview_height = max(1, int(round(image.height * scale)))
    resized = cv2.resize(image.rgba, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    return LatestImage(
        stamp_sec=image.stamp_sec,
        width=preview_width,
        height=preview_height,
        rgba=resized,
    )


class HiFiBridgeController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.pose_pub = rospy.Publisher(args.pose_topic, Pose, queue_size=1)
        self.latest_frame_jpeg: bytes | None = None
        self.latest_preview_frame_jpeg: bytes | None = None
        self.latest_frame_stamp_sec: float | None = None
        self.latest_frame_width = 0
        self.latest_frame_height = 0
        self.latest_preview_frame_width = 0
        self.latest_preview_frame_height = 0
        self.frames_received = 0
        self.pose_messages_sent = 0
        self.last_pose_wall_sec: float | None = None
        self.last_frame_wall_sec: float | None = None
        self.stop_event = threading.Event()
        self.image_sub = rospy.Subscriber(
            args.image_topic,
            Image,
            self.handle_image,
            queue_size=1,
            buff_size=2**26,
        )

    def handle_image(self, message: Image) -> None:
        try:
            decoded = decode_image(message)
            if decoded is None:
                rospy.logwarn_throttle(
                    5.0,
                    "gssdf_hifi_bridge could not decode %s frame: encoding=%s width=%s height=%s step=%s",
                    self.args.image_topic,
                    message.encoding,
                    message.width,
                    message.height,
                    message.step,
                )
                return

            encoded = encode_jpeg(decoded, quality=self.args.jpeg_quality)
            if encoded is None:
                rospy.logwarn_throttle(5.0, "gssdf_hifi_bridge failed to encode JPEG for %s", self.args.image_topic)
                return
            preview = build_preview_image(decoded, max(0.1, min(float(self.args.preview_scale), 1.0)))
            preview_encoded = encode_jpeg(preview, quality=self.args.preview_jpeg_quality)
            if preview_encoded is None:
                preview_encoded = encoded
                preview = decoded

            with self.lock:
                self.latest_frame_jpeg = encoded
                self.latest_preview_frame_jpeg = preview_encoded
                self.latest_frame_stamp_sec = decoded.stamp_sec
                self.latest_frame_width = decoded.width
                self.latest_frame_height = decoded.height
                self.latest_preview_frame_width = preview.width
                self.latest_preview_frame_height = preview.height
                self.last_frame_wall_sec = time.time()
                self.frames_received += 1
        except Exception as exc:  # pragma: no cover - runtime ROS callback safety
            rospy.logerr_throttle(2.0, "gssdf_hifi_bridge image callback failed: %s", exc)

    def publish_camera_pose(self, payload: dict[str, Any]) -> dict[str, Any]:
        position = payload.get("position") or {}
        orientation = payload.get("orientation") or {}

        message = Pose()
        message.position.x = float(position.get("x", 0.0))
        message.position.y = float(position.get("y", 0.0))
        message.position.z = float(position.get("z", 0.0))
        message.orientation.x = float(orientation.get("x", 0.0))
        message.orientation.y = float(orientation.get("y", 0.0))
        message.orientation.z = float(orientation.get("z", 0.0))
        message.orientation.w = float(orientation.get("w", 1.0))
        self.pose_pub.publish(message)

        with self.lock:
            self.pose_messages_sent += 1
            self.last_pose_wall_sec = time.time()
            return self.status()

    def get_latest_frame(
        self,
        preview: bool = False,
        min_frames_received: int | None = None,
        timeout_sec: float = 0.0,
    ) -> tuple[float | None, bytes | None, int]:
        deadline = time.time() + max(0.0, timeout_sec)
        while True:
            with self.lock:
                ready = (
                    min_frames_received is None
                    or self.frames_received >= min_frames_received
                )
                if ready or timeout_sec <= 0.0:
                    return (
                        self.latest_frame_stamp_sec,
                        self.latest_preview_frame_jpeg if preview else self.latest_frame_jpeg,
                        self.frames_received,
                    )
            if time.time() >= deadline:
                with self.lock:
                    return (
                        self.latest_frame_stamp_sec,
                        self.latest_preview_frame_jpeg if preview else self.latest_frame_jpeg,
                        self.frames_received,
                    )
            if self.stop_event.wait(0.01):
                with self.lock:
                    return (
                        self.latest_frame_stamp_sec,
                        self.latest_preview_frame_jpeg if preview else self.latest_frame_jpeg,
                        self.frames_received,
                    )

    def status(self) -> dict[str, Any]:
        with self.lock:
            stale_after_sec = 2.0
            frame_is_fresh = self.last_frame_wall_sec is not None and (time.time() - self.last_frame_wall_sec) <= stale_after_sec
            return {
                "mode": "native-render",
                "ready": self.latest_frame_jpeg is not None and frame_is_fresh,
                "frameIsFresh": frame_is_fresh,
                "imageTopic": self.args.image_topic,
                "poseTopic": self.args.pose_topic,
                "sourceContainer": self.args.source_container,
                "image": self.args.image_name,
                "latestFrameStampSec": self.latest_frame_stamp_sec,
                "lastFrameWallSec": self.last_frame_wall_sec,
                "frameWidth": self.latest_frame_width,
                "frameHeight": self.latest_frame_height,
                "previewFrameWidth": self.latest_preview_frame_width,
                "previewFrameHeight": self.latest_preview_frame_height,
                "framesReceived": self.frames_received,
                "poseMessagesSent": self.pose_messages_sent,
                "lastPoseWallSec": self.last_pose_wall_sec,
            }

    def close(self) -> None:
        self.stop_event.set()
        self.image_sub.unregister()


class Handler(BaseHTTPRequestHandler):
    controller: HiFiBridgeController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self.respond_json(self.controller.status())
            return
        if parsed.path == "/frame.preview.jpg":
            self.respond_jpeg_frame(parsed.query, preview=True)
            return
        if parsed.path == "/frame.jpg":
            self.respond_jpeg_frame(parsed.query)
            return
        if parsed.path == "/frame.preview.mjpeg":
            self.respond_mjpeg_stream(preview=True)
            return
        if parsed.path == "/frame.mjpeg":
            self.respond_mjpeg_stream()
            return
        self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/camera_pose":
            self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        self.respond_json(self.controller.publish_camera_pose(payload))

    def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_jpeg_frame(self, query_string: str, preview: bool = False) -> None:
        query = parse_qs(query_string or "")
        min_frames_received = None
        timeout_sec = 0.0
        try:
            if "minFramesReceived" in query:
                min_frames_received = int(query["minFramesReceived"][0])
        except (TypeError, ValueError):
            min_frames_received = None
        try:
            if "timeoutMs" in query:
                timeout_sec = max(0.0, float(query["timeoutMs"][0]) / 1000.0)
        except (TypeError, ValueError):
            timeout_sec = 0.0

        stamp_sec, frame, frames_received = self.controller.get_latest_frame(
            preview=preview,
            min_frames_received=min_frames_received,
            timeout_sec=timeout_sec,
        )
        if frame is None:
            self.respond_json({"error": "No native render frame available yet."}, HTTPStatus.SERVICE_UNAVAILABLE)
            return

        self.send_response(HTTPStatus.OK)
        self.send_common_headers()
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Frame-Stamp", str(stamp_sec or 0.0))
        self.send_header("X-Frames-Received", str(frames_received))
        self.send_header("Content-Length", str(len(frame)))
        self.end_headers()
        self.wfile.write(frame)

    def respond_mjpeg_stream(self, preview: bool = False) -> None:
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
                stamp_sec, frame, _ = self.controller.get_latest_frame(preview=preview)
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

                mjpeg_max_fps = max(0.0, float(getattr(self.controller.args, "mjpeg_max_fps", 45.0)))
                if mjpeg_max_fps > 0.0:
                    if self.controller.stop_event.wait(1.0 / mjpeg_max_fps):
                        return
                elif self.controller.stop_event.wait(0.01):
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
    rospy.init_node("gssdf_hifi_bridge", anonymous=False, disable_signals=True)
    controller = HiFiBridgeController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        rospy.loginfo("gssdf hifi bridge ready on http://0.0.0.0:%s", args.port)
        server.serve_forever()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
