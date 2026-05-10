#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a side-by-side RGB playback vs GS-SDF HiFi video along the cached playback trajectory."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8876")
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier relative to the bag timeline.",
    )
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--end-offset", type=float, default=-1.0)
    parser.add_argument("--frame-timeout", type=float, default=1.5)
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=3,
        help="After publishing a camera pose, wait for this many new HiFi frames before capturing.",
    )
    parser.add_argument("--capture-url", default="")
    parser.add_argument("--font-file", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_opencv_matrix(text: str, key: str) -> np.ndarray:
    marker = f"{key}: !!opencv-matrix"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"Could not find matrix {key}")
    data_marker = "data: ["
    data_start = text.find(data_marker, start)
    if data_start < 0:
        raise ValueError(f"Could not find data for {key}")
    data_start += len(data_marker)
    data_end = text.find("]", data_start)
    if data_end < 0:
        raise ValueError(f"Could not parse matrix data for {key}")
    values = [
        float(token.strip())
        for token in text[data_start:data_end].replace("\n", " ").split(",")
        if token.strip()
    ]
    if len(values) != 16:
        raise ValueError(f"Expected 16 values for {key}, got {len(values)}")
    return np.array(values, dtype=np.float64).reshape(4, 4)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> dict[str, float]:
    trace = float(rotation[0, 0] + rotation[1, 1] + rotation[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return {"x": x, "y": y, "z": z, "w": w}


def pose_to_camera_pose(frame: dict, t_b_c: np.ndarray) -> dict[str, dict[str, float]]:
    position = frame["pose"]["position"]
    orientation = frame["pose"]["orientation"]
    t_w_b = np.eye(4, dtype=np.float64)
    t_w_b[:3, :3] = quaternion_to_matrix(
        float(orientation[0]),
        float(orientation[1]),
        float(orientation[2]),
        float(orientation[3]),
    )
    t_w_b[:3, 3] = np.array(position, dtype=np.float64)
    t_w_c_optical = t_w_b @ t_b_c
    # ROS optical frame (+x right, +y down, +z forward) differs from the
    # viewer camera convention (+x right, +y up, -z forward). Convert before
    # publishing to /rviz/current_camera_pose.
    optical_to_view = np.eye(4, dtype=np.float64)
    optical_to_view[:3, :3] = np.diag([1.0, -1.0, -1.0])
    t_w_c = t_w_c_optical @ optical_to_view
    quat = matrix_to_quaternion(t_w_c[:3, :3])
    return {
        "position": {
            "x": float(t_w_c[0, 3]),
            "y": float(t_w_c[1, 3]),
            "z": float(t_w_c[2, 3]),
        },
        "orientation": quat,
    }


def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read()


def fetch_bytes_with_headers(url: str) -> tuple[bytes, dict[str, str]]:
    with urllib.request.urlopen(url, timeout=20) as response:
        return response.read(), {k: v for k, v in response.headers.items()}


def wait_for_new_frame(bridge_url: str, last_stamp: float | None, timeout: float) -> tuple[float, int]:
    deadline = time.monotonic() + timeout
    newest = last_stamp
    newest_frames_received = 0
    while time.monotonic() < deadline:
        status = http_get_json(f"{bridge_url.rstrip('/')}/status")
        stamp = status.get("latestFrameStampSec")
        frames_received = int(status.get("framesReceived", 0) or 0)
        if stamp is not None and stamp != newest:
            return float(stamp), frames_received
        time.sleep(0.03)
    return float(newest or 0.0), newest_frames_received


def append_query_params(url: str, params: dict[str, str]) -> str:
    parsed = urllib.parse.urlsplit(url)
    query_items = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_map = dict(query_items)
    query_map.update(params)
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(query_map),
            parsed.fragment,
        )
    )


def build_selected_frames(meta: dict, fps: float, speed: float, start_offset: float, end_offset: float) -> list[dict]:
    frames = list(meta.get("frames", []))
    offsets = [float(frame["offsetSec"]) for frame in frames]
    duration = float(meta.get("durationSec", 0.0))
    if end_offset <= 0.0 or end_offset > duration:
        end_offset = duration
    start_offset = max(0.0, min(start_offset, end_offset))
    if end_offset <= start_offset:
        return []
    step = max(speed / max(fps, 1e-3), 1e-3)
    selected: list[dict] = []
    target = start_offset
    while target <= end_offset + 1e-6:
        index = bisect.bisect_left(offsets, target)
        if index >= len(frames):
            index = len(frames) - 1
        elif index > 0:
            left = offsets[index - 1]
            right = offsets[index]
            if abs(target - left) <= abs(right - target):
                index -= 1
        # Keep duplicate source frames when exporting below the cache cadence.
        # Dropping them shortens the encoded video and breaks sidecar timing for 1x playback.
        selected.append(frames[index])
        target += step
    if selected and selected[-1]["index"] != frames[-1]["index"] and end_offset >= duration - 1e-3:
        selected.append(frames[-1])
    return selected


def ensure_bridge_ready(bridge_url: str) -> None:
    status = http_get_json(f"{bridge_url.rstrip('/')}/status")
    if not status.get("ready"):
        raise RuntimeError(f"HiFi bridge is not ready: {status}")


def export_frames(args: argparse.Namespace, selected_frames: list[dict], t_b_c: np.ndarray, temp_dir: Path) -> tuple[Path, Path]:
    left_dir = temp_dir / "left"
    right_dir = temp_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    bridge_url = args.bridge_url.rstrip("/")
    capture_url = args.capture_url.strip() or f"{bridge_url}/frame.jpg"
    initial_status = http_get_json(f"{bridge_url}/status")
    last_stamp = initial_status.get("latestFrameStampSec")
    last_frames_received = int(initial_status.get("framesReceived", 0) or 0)

    for output_index, frame in enumerate(selected_frames):
        source_image = cache_dir / frame["imageFile"]
        left_target = left_dir / f"frame_{output_index:05d}.jpg"
        shutil.copy2(source_image, left_target)

        camera_pose = pose_to_camera_pose(frame, t_b_c)
        http_post_json(f"{bridge_url}/camera_pose", camera_pose)
        # The native GS-SDF renderer can emit one or two transient frames right after a pose
        # update. Wait for several new frames before sampling the JPEG so the export records
        # the settled render instead of the intermediate flash.
        settle_frames = max(int(args.settle_frames), 1)
        _, observed_frames = wait_for_new_frame(bridge_url, last_stamp, args.frame_timeout)
        if observed_frames > 0:
            last_frames_received = observed_frames
        target_frames_received = last_frames_received + max(settle_frames - 1, 0)
        capture_request_url = append_query_params(
            capture_url,
            {
                "frame": str(frame["index"]),
                "ts": str(time.time_ns()),
                "minFramesReceived": str(target_frames_received),
                "timeoutMs": str(int(max(args.frame_timeout, 0.1) * 1000.0)),
            },
        )
        right_bytes, headers = fetch_bytes_with_headers(capture_request_url)
        try:
            last_frames_received = int(headers.get("X-Frames-Received", str(last_frames_received)))
        except ValueError:
            pass
        try:
            last_stamp = float(headers.get("X-Frame-Stamp", str(last_stamp or 0.0)))
        except ValueError:
            pass
        (right_dir / f"frame_{output_index:05d}.jpg").write_bytes(right_bytes)

    return left_dir, right_dir


def build_video(args: argparse.Namespace, left_dir: Path, right_dir: Path, frame_count: int) -> None:
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filter_graph = (
        "[0:v]scale=-2:1024,pad=1280:1024:(ow-iw)/2:(oh-ih)/2,setsar=1[left];"
        "[1:v]scale=-2:1024,pad=1280:1024:(ow-iw)/2:(oh-ih)/2,setsar=1[right];"
    )
    font_file = Path(args.font_file)
    if font_file.exists():
        escaped = str(font_file).replace(":", "\\:")
        filter_graph += (
            f"[left]drawtext=fontfile='{escaped}':text='Playback RGB':fontcolor=white:fontsize=36:x=40:y=40:"
            "box=1:boxcolor=black@0.45:boxborderw=12[left_l];"
            f"[right]drawtext=fontfile='{escaped}':text='GS-SDF Gaussian':fontcolor=white:fontsize=36:x=40:y=40:"
            "box=1:boxcolor=black@0.45:boxborderw=12[right_l];"
            "[left_l][right_l]hstack=inputs=2[v]"
        )
    else:
        filter_graph += "[left][right]hstack=inputs=2[v]"

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{args.fps}",
        "-i",
        str(left_dir / "frame_%05d.jpg"),
        "-framerate",
        f"{args.fps}",
        "-i",
        str(right_dir / "frame_%05d.jpg"),
        "-frames:v",
        str(frame_count),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def write_playback_video_sidecar(args: argparse.Namespace) -> None:
    payload = {
        "path": str(Path(args.output_video).resolve()),
        "fps": float(args.fps),
        "speed": float(args.speed),
        "startOffsetSec": float(args.start_offset),
        "endOffsetSec": float(args.end_offset),
        "label": "Playback RGB · GS-SDF",
        "generatedAtSec": time.time(),
    }

    cache_dir = Path(args.cache_dir).resolve()
    sidecar_paths = [cache_dir / "playback_video.json"]
    repo_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha1(str(cache_dir).encode("utf-8")).hexdigest()[:12]
    sidecar_paths.append(repo_root / "runtime" / "playback-video-index" / f"{cache_dir.name}-{digest}.json")

    encoded = json.dumps(payload, indent=2) + "\n"
    wrote_any = False
    for sidecar_path in sidecar_paths:
        try:
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            sidecar_path.write_text(encoded)
            print(f"Wrote playback video sidecar: {sidecar_path}")
            wrote_any = True
        except OSError as error:
            print(f"Warning: failed to write playback video sidecar {sidecar_path}: {error}")

    if not wrote_any:
        raise RuntimeError("Failed to write playback video sidecar metadata.")


def main() -> None:
    args = parse_args()
    ensure_bridge_ready(args.bridge_url)

    meta = load_json(Path(args.cache_dir) / "meta.json")
    config_text = Path(args.scene_config).read_text()
    t_c_l = parse_opencv_matrix(config_text, "T_C_L")
    t_b_l = parse_opencv_matrix(config_text, "T_B_L")
    t_b_c = t_b_l @ np.linalg.inv(t_c_l)

    selected_frames = build_selected_frames(
        meta,
        fps=args.fps,
        speed=args.speed,
        start_offset=args.start_offset,
        end_offset=args.end_offset,
    )
    if not selected_frames:
        raise RuntimeError("No playback frames selected for export.")

    if args.keep_frames:
        temp_root = Path(args.output_video).with_suffix("")
        temp_root.mkdir(parents=True, exist_ok=True)
        left_dir, right_dir = export_frames(args, selected_frames, t_b_c, temp_root)
    else:
        with tempfile.TemporaryDirectory(prefix="gssdf_playback_compare_") as tmp_dir:
            temp_root = Path(tmp_dir)
            left_dir, right_dir = export_frames(args, selected_frames, t_b_c, temp_root)
            build_video(args, left_dir, right_dir, len(selected_frames))
            write_playback_video_sidecar(args)
            return

    build_video(args, left_dir, right_dir, len(selected_frames))
    write_playback_video_sidecar(args)


if __name__ == "__main__":
    main()
