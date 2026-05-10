#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP control bridge for FAST-LIVO rosbag playback.")
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--container-name", default="fastlivo-playback")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image", default=os.environ.get("GS_SDF_IMAGE", "gs_sdf_img:latest"))
    parser.add_argument("--default-rate", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    return parser.parse_args()


def shell_json(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def inspect_bag(image: str, bag_path: str) -> dict:
    bag_abs = str(Path(bag_path).resolve())
    bag_parent = str(Path(bag_abs).parent)
    output = shell_json(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{bag_parent}:{bag_parent}",
            image,
            "bash",
            "-lc",
            f"source /opt/ros/noetic/setup.bash && rosbag info --yaml {json.dumps(bag_abs)}",
        ]
    )
    meta: dict[str, float | str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"path", "version", "compression"}:
            meta[key] = value.strip()
        elif key in {"duration", "start", "end", "size", "messages", "compressed", "uncompressed"}:
            try:
                meta[key] = float(value.strip())
            except ValueError:
                pass
    return meta


class PlaybackController:
    def __init__(
        self,
        bag_path: str,
        container_name: str,
        image: str,
        default_rate: float,
        loop: bool,
        rosbridge_port: int,
    ) -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        self.launch_script = self.root_dir / "scripts" / "launch_fastlivo_playback.sh"
        self.rosbridge_script = self.root_dir / "scripts" / "launch_rosbridge_sidecar.sh"
        self.bag_path = str(Path(bag_path).resolve())
        self.container_name = container_name
        self.image = image
        self.loop = loop
        self.rosbridge_port = rosbridge_port
        self.warmup_sec = 3.0
        self.lock = threading.RLock()
        self.meta = inspect_bag(image, self.bag_path)
        self.state = {
            "bagPath": self.bag_path,
            "containerName": self.container_name,
            "image": self.image,
            "rate": float(default_rate),
            "paused": False,
            "loop": bool(loop),
            "requestedOffsetSec": 0.0,
            "launchedAtMono": time.monotonic(),
            "durationSec": float(self.meta.get("duration", 0.0)),
            "bagStartSec": float(self.meta.get("start", 0.0)),
            "bagEndSec": float(self.meta.get("end", 0.0)),
        }

    def current_offset_sec(self) -> float:
        with self.lock:
            offset = float(self.state["requestedOffsetSec"])
            if self.state["paused"]:
                return max(0.0, min(offset, float(self.state["durationSec"])))
            elapsed = max(0.0, time.monotonic() - float(self.state["launchedAtMono"]) - self.warmup_sec)
            current = offset + elapsed * float(self.state["rate"])
            duration = float(self.state["durationSec"])
            if self.loop and duration > 0:
                current %= duration
            return max(0.0, min(current, duration))

    def status(self) -> dict:
        with self.lock:
            return {
                "bagPath": self.state["bagPath"],
                "containerName": self.state["containerName"],
                "image": self.state["image"],
                "rate": self.state["rate"],
                "paused": self.state["paused"],
                "loop": self.state["loop"],
                "requestedOffsetSec": self.state["requestedOffsetSec"],
                "currentOffsetSec": self.current_offset_sec(),
                "durationSec": self.state["durationSec"],
                "bagStartSec": self.state["bagStartSec"],
                "bagEndSec": self.state["bagEndSec"],
                "playbackAlive": self.is_container_alive(self.container_name),
            }

    def restart(self, rate: float | None = None, paused: bool | None = None, offset_sec: float | None = None) -> dict:
        with self.lock:
            next_rate = float(rate if rate is not None else self.state["rate"])
            next_paused = bool(paused if paused is not None else self.state["paused"])
            next_offset = float(offset_sec if offset_sec is not None else self.current_offset_sec())
            next_offset = max(0.0, min(next_offset, float(self.state["durationSec"])))

        env = os.environ.copy()
        env["GS_SDF_IMAGE"] = self.image
        env["RATE"] = str(next_rate)
        env["START_OFFSET"] = str(next_offset)
        env["START_PAUSED"] = "1" if next_paused else "0"
        env["LOOP"] = "1" if self.loop else "0"

        subprocess.run(
            ["bash", str(self.launch_script), self.bag_path, self.container_name],
            cwd=str(self.root_dir),
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.restart_rosbridge()

        with self.lock:
            self.state["rate"] = next_rate
            self.state["paused"] = next_paused
            self.state["requestedOffsetSec"] = next_offset
            self.state["launchedAtMono"] = time.monotonic()
            return self.status()

    @staticmethod
    def is_container_alive(name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def restart_rosbridge(self) -> None:
        subprocess.run(
            [
                "bash",
                str(self.rosbridge_script),
                self.container_name,
                str(self.rosbridge_port),
            ],
            cwd=str(self.root_dir),
            env=os.environ.copy(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class Handler(BaseHTTPRequestHandler):
    controller: PlaybackController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/status":
            self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        self.respond_json(self.controller.status())

    def do_POST(self) -> None:
        if self.path != "/control":
            self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        status = self.controller.restart(
            rate=payload.get("rate"),
            paused=payload.get("paused"),
            offset_sec=payload.get("offsetSec"),
        )
        self.respond_json(status)

    def respond_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    args = parse_args()
    controller = PlaybackController(
        bag_path=args.bag_path,
        container_name=args.container_name,
        image=args.image,
        default_rate=args.default_rate,
        loop=args.loop,
        rosbridge_port=args.rosbridge_port,
    )
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"playback-control ready on http://localhost:{args.port} for {args.container_name}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
