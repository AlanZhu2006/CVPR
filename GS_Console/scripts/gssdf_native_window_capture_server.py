#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a dedicated X11 display and expose it as JPEG/MJPEG for the web UI."
    )
    parser.add_argument("--display", default=":88")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--label", default="gssdf-native-window")
    return parser.parse_args()


class NativeWindowCaptureController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.latest_frame_jpeg: bytes | None = None
        self.latest_frame_wall_sec: float | None = None
        self.frames_captured = 0
        self.last_error: str | None = None
        self.last_window_probe_monotonic = 0.0
        self.last_window_available = False
        self.worker = threading.Thread(target=self.capture_loop, daemon=True)
        self.worker.start()

    def has_mapped_window(self) -> bool:
        now = time.monotonic()
        if now - self.last_window_probe_monotonic < 0.5:
            return self.last_window_available

        result = subprocess.run(
            ["xwininfo", "-display", self.args.display, "-root", "-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        available = result.returncode == 0 and "0 children." not in output
        self.last_window_probe_monotonic = now
        self.last_window_available = available
        return available

    def capture_frame(self) -> bytes | None:
        if not self.has_mapped_window():
            raise RuntimeError(f"No mapped X11 windows on {self.args.display}")

        command = [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "x11grab",
            "-video_size",
            f"{self.args.width}x{self.args.height}",
            "-i",
            self.args.display,
            "-frames:v",
            "1",
            "-q:v",
            str(max(2, min(int(round((100 - self.args.jpeg_quality) / 10)) + 2, 8))),
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(stderr or f"ffmpeg x11grab exited with code {result.returncode}")
        return result.stdout or None

    def capture_loop(self) -> None:
        interval = 1.0 / max(self.args.fps, 1.0)
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                frame = self.capture_frame()
                with self.lock:
                    self.latest_frame_jpeg = frame
                    self.latest_frame_wall_sec = time.time()
                    self.frames_captured += 1
                    self.last_error = None
            except Exception as error:
                with self.lock:
                    self.last_error = str(error)
            elapsed = time.monotonic() - started
            wait_time = max(0.01, interval - elapsed)
            if self.stop_event.wait(wait_time):
                return

    def get_latest_frame(self) -> tuple[float | None, bytes | None]:
        with self.lock:
            return self.latest_frame_wall_sec, self.latest_frame_jpeg

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": "native-window-capture",
                "ready": self.latest_frame_jpeg is not None,
                "display": self.args.display,
                "width": self.args.width,
                "height": self.args.height,
                "framesCaptured": self.frames_captured,
                "latestFrameWallSec": self.latest_frame_wall_sec,
                "lastError": self.last_error,
                "label": self.args.label,
            }

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)


class Handler(BaseHTTPRequestHandler):
    controller: NativeWindowCaptureController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/status":
            self.respond_json(self.controller.status())
            return
        if self.path.startswith("/frame.jpg"):
            self.respond_jpeg_frame()
            return
        if self.path.startswith("/frame.mjpeg"):
            self.respond_mjpeg_stream()
            return
        self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_jpeg_frame(self) -> None:
        stamp_sec, frame = self.controller.get_latest_frame()
        if frame is None:
            self.respond_json(self.controller.status(), HTTPStatus.SERVICE_UNAVAILABLE)
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
                stamp_sec, frame = self.controller.get_latest_frame()
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

                if self.controller.stop_event.wait(1.0 / max(self.controller.args.fps, 1.0)):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    args = parse_args()
    controller = NativeWindowCaptureController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
