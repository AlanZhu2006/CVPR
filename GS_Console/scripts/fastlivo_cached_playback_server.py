#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import cv2
import functools
import hashlib
import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from fastlivo_direct_playback_server import (
    CURRENT_SCAN_WORLD_TOPIC,
    GLOBAL_MAP_TOPIC,
    MapVoxel,
    create_rgb_cloud,
    load_calibration,
    trim_map_voxels,
)

WEB_GLOBAL_MAP_TOPIC = "/fastlivo/global_map_web"
WEB_CURRENT_SCAN_WORLD_TOPIC = "/fastlivo/current_scan_world_web"
WEB_MAP_NEUTRAL_FALLBACK = np.array([160, 165, 174], dtype=np.uint8)
WEB_SCAN_NEUTRAL_FALLBACK = np.array([150, 156, 165], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached FAST-LIVO playback server.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--default-rate", type=float, default=1.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--container-name", default="fastlivo-cached-playback")
    parser.add_argument("--image-name", default="gs_sdf_img:latest")
    parser.add_argument("--mjpeg-max-fps", type=float, default=60.0)
    parser.add_argument("--map-publish-every", type=int, default=1)
    parser.add_argument(
        "--map-fuse-max-points",
        type=int,
        default=6_000,
        help="Cap scan points fused into the global map per frame (full scan still published on ROS). 0 = no cap.",
    )
    parser.add_argument(
        "--restore-replay-max-frames",
        type=int,
        default=8,
        help="When seeking/restoring, replay at most this many scans after the nearest keyframe to rebuild the map.",
    )
    parser.add_argument("--web-map-max-points", type=int, default=48000)
    parser.add_argument("--web-scan-max-points", type=int, default=8000)
    parser.add_argument(
        "--scan-history-frames",
        type=int,
        default=6,
        help="Blend this many recent scans for the web/showcase current-scan stream.",
    )
    parser.add_argument("--video-path", default="")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--video-speed", type=float, default=1.0)
    parser.add_argument("--video-start-offset", type=float, default=0.0)
    parser.add_argument("--video-end-offset", type=float, default=-1.0)
    parser.add_argument("--video-label", default="")
    return parser.parse_args()


@functools.lru_cache(maxsize=192)
def load_cloud_npz(path_str: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path_str) as data:
        positions = data["positions"].astype(np.float32, copy=False)
        colors = data["colors"].astype(np.uint8, copy=False)
    return positions, colors


@functools.lru_cache(maxsize=384)
def load_clean_cloud_npz(
    path_str: str,
    neutral_r: int,
    neutral_g: int,
    neutral_b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions, colors = load_cloud_npz(path_str)
    neutral_color = np.array([neutral_r, neutral_g, neutral_b], dtype=np.uint8)
    cleaned_colors, legacy_mask = neutralize_legacy_false_colors(colors, neutral_color)
    valid_mask = ~legacy_mask
    return (
        np.asarray(positions, dtype=np.float32, order="C"),
        np.asarray(cleaned_colors, dtype=np.uint8, order="C"),
        np.asarray(valid_mask, dtype=bool),
    )


@functools.lru_cache(maxsize=192)
def load_binary_file(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@functools.lru_cache(maxsize=192)
def load_rgb_image_file(path_str: str) -> np.ndarray | None:
    encoded = np.frombuffer(load_binary_file(path_str), dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return None
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def downsample_cloud(
    positions: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(positions.shape[0])
    if count <= max_points or max_points <= 0:
        return positions, colors
    indices = np.linspace(0, count - 1, num=max_points, dtype=np.int32)
    return positions[indices], colors[indices]


def sample_cloud_prioritizing_valid(
    positions: np.ndarray,
    colors: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(positions.shape[0])
    if count <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0,), dtype=bool),
        )

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if max_points <= 0 or count <= max_points:
        return (
            np.asarray(positions, dtype=np.float32, order="C"),
            np.asarray(colors, dtype=np.uint8, order="C"),
            valid_mask,
        )

    valid_indices = np.flatnonzero(valid_mask)
    invalid_indices = np.flatnonzero(~valid_mask)
    if valid_indices.size >= max_points:
        selected_indices = valid_indices[
            np.linspace(0, valid_indices.size - 1, num=max_points, dtype=np.int32)
        ]
    else:
        remaining = max_points - int(valid_indices.size)
        if invalid_indices.size > remaining:
            invalid_selected = invalid_indices[
                np.linspace(0, invalid_indices.size - 1, num=remaining, dtype=np.int32)
            ]
        else:
            invalid_selected = invalid_indices
        selected_indices = np.concatenate([valid_indices, invalid_selected]).astype(np.int32, copy=False)
        selected_indices.sort()

    return (
        np.asarray(positions[selected_indices], dtype=np.float32, order="C"),
        np.asarray(colors[selected_indices], dtype=np.uint8, order="C"),
        np.asarray(valid_mask[selected_indices], dtype=bool),
    )


def copy_or_repeat_fallback_colors(fallback_colors: np.ndarray, count: int) -> np.ndarray:
    fallback = np.asarray(fallback_colors, dtype=np.uint8)
    if fallback.ndim == 1:
        return np.repeat(fallback.reshape(1, 3), count, axis=0)
    if fallback.shape[0] == count:
        return np.array(fallback, dtype=np.uint8, copy=True)
    if fallback.shape[0] <= 0:
        return np.repeat(WEB_MAP_NEUTRAL_FALLBACK.reshape(1, 3), count, axis=0)
    indices = np.linspace(0, fallback.shape[0] - 1, num=count, dtype=np.int32)
    return np.asarray(fallback[indices], dtype=np.uint8, order="C")


def blend_scan_history_colors(colors: np.ndarray, age: int) -> np.ndarray:
    if age <= 0 or colors.size == 0:
        return colors
    blend_target = np.array([232.0, 234.0, 238.0], dtype=np.float32)
    blend = max(0.86, 0.97 - 0.03 * float(age))
    base = colors.astype(np.float32, copy=False)
    mixed = blend_target + (base - blend_target) * blend
    return np.clip(np.round(mixed), 0, 255).astype(np.uint8)


def _legacy_intensity_palette_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(float(value), 1.0))
    if value < 0.33:
        t = value / 0.33
        return (0, int(120 + 135 * t), int(255 - 80 * t))
    if value < 0.66:
        t = (value - 0.33) / 0.33
        return (int(255 * t), 255, int(175 * (1.0 - t)))
    t = (value - 0.66) / 0.34
    return (255, int(255 - 135 * t), 0)


@functools.lru_cache(maxsize=1)
def build_legacy_false_color_palette() -> np.ndarray:
    palette = np.array(
        [_legacy_intensity_palette_color(index / 2047.0) for index in range(2048)],
        dtype=np.uint8,
    )
    return np.unique(palette, axis=0)


def legacy_false_color_mask(colors: np.ndarray) -> np.ndarray:
    colors_u8 = np.asarray(colors, dtype=np.uint8)
    if colors_u8.size == 0:
        return np.empty((0,), dtype=bool)

    colors_f32 = colors_u8.astype(np.float32, copy=False)
    palette = build_legacy_false_color_palette().astype(np.float32, copy=False)
    threshold_sq = float(16.0 * 16.0)
    mask = np.zeros((colors_u8.shape[0],), dtype=bool)

    batch_size = 4096
    for start in range(0, colors_u8.shape[0], batch_size):
        end = min(start + batch_size, colors_u8.shape[0])
        batch = colors_f32[start:end]
        distances_sq = ((batch[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        mask[start:end] = np.min(distances_sq, axis=1) <= threshold_sq

    r = colors_f32[:, 0]
    g = colors_f32[:, 1]
    b = colors_f32[:, 2]
    max_c = np.maximum.reduce([r, g, b])
    min_c = np.minimum.reduce([r, g, b])
    saturation = np.divide(
        max_c - min_c,
        np.maximum(max_c, 1.0),
        out=np.zeros_like(max_c),
        where=max_c > 0,
    )

    warm_band = (r >= 185.0) & (g >= 90.0) & (g <= 175.0) & (b <= 90.0) & (saturation >= 0.48)
    cool_band = (r <= 90.0) & (g >= 80.0) & (g <= 190.0) & (b >= 165.0) & (saturation >= 0.44)
    green_band = (g >= 175.0) & (b <= 190.0) & (saturation >= 0.5) & ((r <= 210.0) | (b <= 120.0))
    return mask | warm_band | cool_band | green_band


def neutralize_legacy_false_colors(
    colors: np.ndarray,
    neutral_color: np.ndarray,
    active_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    colors_u8 = np.asarray(colors, dtype=np.uint8, order="C")
    if colors_u8.size == 0:
        return colors_u8, np.empty((0,), dtype=bool)

    legacy_mask = legacy_false_color_mask(colors_u8)
    if active_mask is not None:
        legacy_mask &= np.asarray(active_mask, dtype=bool)
    if np.any(legacy_mask):
        colors_u8 = np.array(colors_u8, copy=True)
        colors_u8[legacy_mask] = np.asarray(neutral_color, dtype=np.uint8)
    return colors_u8, legacy_mask


def pose_record_to_matrix(pose: dict[str, Any]) -> np.ndarray:
    position = pose.get("position") or [0.0, 0.0, 0.0]
    orientation = pose.get("orientation") or [0.0, 0.0, 0.0, 1.0]
    x = float(orientation[0])
    y = float(orientation[1])
    z = float(orientation[2])
    w = float(orientation[3])
    tx = float(position[0])
    ty = float(position[1])
    tz = float(position[2])

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_video_sync_map(
    frame_offsets: list[float],
    duration_sec: float,
    fps: float,
    speed: float,
    start_offset_sec: float,
    end_offset_sec: float,
) -> dict[str, Any]:
    if not frame_offsets:
        return {
            "fps": fps,
            "speed": speed,
            "videoFrameCount": 0,
            "playbackFrameCount": 0,
            "frameIndices": [],
            "offsetsSec": [],
            "videoDurationSec": 0.0,
            "playbackDurationSec": duration_sec,
        }

    fps = max(float(fps), 1e-3)
    speed = max(float(speed), 1e-3)
    if end_offset_sec <= 0.0 or end_offset_sec > duration_sec:
        end_offset_sec = duration_sec
    start_offset_sec = max(0.0, min(float(start_offset_sec), end_offset_sec))
    if end_offset_sec <= start_offset_sec:
        return {
            "fps": fps,
            "speed": speed,
            "videoFrameCount": 0,
            "playbackFrameCount": len(frame_offsets),
            "frameIndices": [],
            "offsetsSec": [],
            "videoDurationSec": 0.0,
            "playbackDurationSec": duration_sec,
        }

    step = speed / fps
    selected_indices: list[int] = []
    selected_offsets: list[float] = []
    target = start_offset_sec
    while target <= end_offset_sec + 1e-6:
        index = bisect.bisect_left(frame_offsets, target)
        if index >= len(frame_offsets):
            index = len(frame_offsets) - 1
        elif index > 0:
            left = frame_offsets[index - 1]
            right = frame_offsets[index]
            if abs(target - left) <= abs(right - target):
                index -= 1
        # Mirror export_playback_rgb_vs_gssdf_video.py exactly: keep duplicate source
        # frames when exporting or synchronizing below the cache cadence. Deduping here
        # shortens the sync table and makes video time drift ahead of bag time.
        selected_indices.append(index)
        selected_offsets.append(float(frame_offsets[index]))
        target += step

    if selected_indices and selected_indices[-1] != len(frame_offsets) - 1 and end_offset_sec >= duration_sec - 1e-3:
        selected_indices.append(len(frame_offsets) - 1)
        selected_offsets.append(float(frame_offsets[-1]))

    video_frame_count = len(selected_indices)
    return {
        "fps": fps,
        "speed": speed,
        "videoFrameCount": video_frame_count,
        "playbackFrameCount": len(frame_offsets),
        "frameIndices": selected_indices,
        "offsetsSec": selected_offsets,
        "videoDurationSec": float(video_frame_count / fps) if video_frame_count > 0 else 0.0,
        "playbackDurationSec": duration_sec,
    }


def resolve_optional_path(path_value: str | None, base_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path if path.exists() else None


def playback_video_registry_path(cache_dir: Path) -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha1(str(cache_dir.resolve()).encode("utf-8")).hexdigest()[:12]
    return repo_root / "runtime" / "playback-video-index" / f"{cache_dir.name}-{digest}.json"


def load_playback_video_descriptor(cache_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    video_path = str(getattr(args, "video_path", "") or "").strip()
    if video_path:
        resolved = resolve_optional_path(video_path, cache_dir)
        if resolved is None:
            print(f"[cached-playback] Ignoring missing playback video: {video_path}", flush=True)
            return None
        return {
            "path": resolved,
            "fps": max(float(getattr(args, "video_fps", 12.0)), 1e-3),
            "speed": max(float(getattr(args, "video_speed", 1.0)), 1e-3),
            "startOffsetSec": float(getattr(args, "video_start_offset", 0.0)),
            "endOffsetSec": float(getattr(args, "video_end_offset", -1.0)),
            "label": str(getattr(args, "video_label", "") or "").strip() or "Offline playback video",
        }

    for sidecar_path in (cache_dir / "playback_video.json", playback_video_registry_path(cache_dir)):
        if not sidecar_path.exists():
            continue

        try:
            payload = json.loads(sidecar_path.read_text())
        except Exception as error:
            print(f"[cached-playback] Failed to parse playback video sidecar {sidecar_path}: {error}", flush=True)
            continue

        resolved = resolve_optional_path(str(payload.get("path") or payload.get("url") or ""), sidecar_path.parent)
        if resolved is None:
            print(f"[cached-playback] Playback video sidecar points to a missing file: {sidecar_path}", flush=True)
            continue

        return {
            "path": resolved,
            "fps": max(float(payload.get("fps", 12.0)), 1e-3),
            "speed": max(float(payload.get("speed", 1.0)), 1e-3),
            "startOffsetSec": float(payload.get("startOffsetSec", 0.0)),
            "endOffsetSec": float(payload.get("endOffsetSec", -1.0)),
            "label": str(payload.get("label") or "Offline playback video"),
        }

    return None


class CachedPlaybackController:
    def __init__(self, cache_dir: Path, args: argparse.Namespace):
        self.cache_dir = cache_dir
        self.args = args
        self.meta = json.loads((cache_dir / "meta.json").read_text())
        self.config_path = Path(str(self.meta.get("configPath", ""))).resolve() if self.meta.get("configPath") else None
        self.rectify_calibration = None
        self.rectify_camera_matrix: np.ndarray | None = None
        self.rectify_new_camera_matrix: np.ndarray | None = None
        self.rectify_distortion: np.ndarray | None = None
        if self.config_path and self.config_path.exists():
            try:
                self.rectify_calibration = load_calibration(self.config_path)
                calibration = self.rectify_calibration
                self.rectify_camera_matrix = np.array(
                    [
                        [calibration.fx, 0.0, calibration.cx],
                        [0.0, calibration.fy, calibration.cy],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                self.rectify_new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    self.rectify_camera_matrix,
                    calibration.distortion,
                    (calibration.width, calibration.height),
                    0.0,
                    (calibration.width, calibration.height),
                )
                self.rectify_distortion = calibration.distortion
            except Exception:
                self.rectify_calibration = None
                self.rectify_camera_matrix = None
                self.rectify_new_camera_matrix = None
                self.rectify_distortion = None
        self.frames: list[dict[str, Any]] = list(self.meta.get("frames", []))
        self.keyframes: list[dict[str, Any]] = list(self.meta.get("keyframes", []))
        self.frame_offsets = [float(frame["offsetSec"]) for frame in self.frames]
        self.keyframe_indices = [int(item["frameIndex"]) for item in self.keyframes]
        self.keyframe_files = [str((cache_dir / item["file"]).resolve()) for item in self.keyframes]
        self.pose_track_payload: dict[str, Any] | None = None
        self.duration_sec = float(self.meta.get("durationSec", 0.0))
        self.start_sec = float(self.meta.get("startSec", 0.0))
        self.end_sec = float(self.meta.get("endSec", self.start_sec + self.duration_sec))
        self.playback_video = load_playback_video_descriptor(cache_dir, args)
        self.topics = {
            "/aft_mapped_to_init": {
                "type": "nav_msgs/Odometry",
                "count": len(self.frames),
            },
            CURRENT_SCAN_WORLD_TOPIC: {
                "type": "sensor_msgs/PointCloud2",
                "count": len(self.frames),
            },
            WEB_CURRENT_SCAN_WORLD_TOPIC: {
                "type": "sensor_msgs/PointCloud2",
                "count": len(self.frames),
            },
            GLOBAL_MAP_TOPIC: {
                "type": "sensor_msgs/PointCloud2",
                "count": len(self.keyframes) if self.keyframes else len(self.frames),
            },
            WEB_GLOBAL_MAP_TOPIC: {
                "type": "sensor_msgs/PointCloud2",
                "count": len(self.keyframes) if self.keyframes else len(self.frames),
            },
        }

        self.odom_pub = rospy.Publisher("/aft_mapped_to_init", Odometry, queue_size=1)
        self.current_scan_world_pub = rospy.Publisher(
            CURRENT_SCAN_WORLD_TOPIC, PointCloud2, queue_size=1
        )
        self.current_scan_world_web_pub = rospy.Publisher(
            WEB_CURRENT_SCAN_WORLD_TOPIC, PointCloud2, queue_size=1
        )
        self.global_map_pub = rospy.Publisher(
            GLOBAL_MAP_TOPIC, PointCloud2, queue_size=1, latch=True
        )
        self.global_map_web_pub = rospy.Publisher(
            WEB_GLOBAL_MAP_TOPIC, PointCloud2, queue_size=1, latch=True
        )
        self.clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)

        self.lock = threading.RLock()
        self.rate = float(args.default_rate)
        self.paused = bool(args.paused)
        self.loop = bool(args.loop)
        self.container_name = str(args.container_name)
        self.image_name = str(args.image_name)
        self.requested_offset_sec = max(0.0, min(float(args.start_offset), self.duration_sec))
        self.current_offset_sec = self.requested_offset_sec
        self.playback_alive = True
        self.control_version = 0
        self.session_start_wall = time.monotonic()
        self.session_start_offset = self.requested_offset_sec
        self.latest_image_jpeg: bytes | None = None
        self.latest_image_stamp_sec: float | None = None
        self.latest_pose_stamp_sec: float | None = None
        self.global_map_voxels: dict[tuple[int, int, int], MapVoxel] = {}
        self.global_map_point_count = 0
        self.current_scan_point_count = 0
        self.current_frame_index = -1
        self._overlay_map_cache_key: tuple[int, int] | None = None
        self._overlay_map_cache_payload: dict[str, Any] | None = None
        self._overlay_live_map_payloads: dict[int, dict[str, Any]] = {}
        self._overlay_map_refresh_guard = threading.Lock()
        self._overlay_map_refresh_busy = False
        self._overlay_map_refresh_pending_frame_index: int | None = None
        self.stop_event = threading.Event()
        self._map_publish_guard = threading.Lock()
        self._map_publish_busy = False
        self._map_publish_pending_frame: dict[str, Any] | None = None
        self.debug_worker_state = "init"
        self.debug_desired_offset_sec = self.current_offset_sec
        self.debug_target_index = self.current_frame_index
        self.debug_last_advance_wall_sec = time.monotonic()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def estimate_current_offset_sec(self) -> float:
        with self.lock:
            offset = self.current_offset_sec
            duration_sec = self.duration_sec
            if self.paused or not self.playback_alive:
                return max(0.0, min(offset, duration_sec))

            elapsed = max(0.0, time.monotonic() - self.session_start_wall)
            wall_estimated = self.session_start_offset + elapsed * max(self.rate, 0.1)
            if self.loop and duration_sec > 0:
                wall_estimated %= duration_sec
            else:
                wall_estimated = min(wall_estimated, duration_sec)
            return min(wall_estimated, offset)

    def status(self) -> dict[str, Any]:
        if not self.lock.acquire(timeout=0.05):
            current_frame_index = self.current_frame_index
            current_offset_sec = self.current_offset_sec
            requested_offset_sec = self.requested_offset_sec
            rate = self.rate
            paused = self.paused
            loop = self.loop
            playback_alive = self.playback_alive
            latest_image_stamp_sec = self.latest_image_stamp_sec
            latest_pose_stamp_sec = self.latest_pose_stamp_sec
            global_map_point_count = self.global_map_point_count
            current_scan_point_count = self.current_scan_point_count
        else:
            try:
                current_frame_index = self.current_frame_index
                current_offset_sec = self.current_offset_sec
                requested_offset_sec = self.requested_offset_sec
                rate = self.rate
                paused = self.paused
                loop = self.loop
                playback_alive = self.playback_alive
                latest_image_stamp_sec = self.latest_image_stamp_sec
                latest_pose_stamp_sec = self.latest_pose_stamp_sec
                global_map_point_count = self.global_map_point_count
                current_scan_point_count = self.current_scan_point_count
            finally:
                self.lock.release()

        current_pose = None
        if 0 <= current_frame_index < len(self.frames):
            current_frame = self.frames[current_frame_index]
            pose = current_frame.get("pose") or {}
            position = pose.get("position") or [0.0, 0.0, 0.0]
            orientation = pose.get("orientation") or [0.0, 0.0, 0.0, 1.0]
            stamp_sec = float(pose.get("stampSec") or current_frame.get("stampSec") or 0.0)
            current_pose = {
                "frameId": str(pose.get("frameId") or "world"),
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "orientation": {
                    "x": float(orientation[0]),
                    "y": float(orientation[1]),
                    "z": float(orientation[2]),
                    "w": float(orientation[3]),
                },
                "stampMs": int(round(stamp_sec * 1000.0)),
            }

        if paused or not playback_alive:
            estimated_offset_sec = max(0.0, min(current_offset_sec, self.duration_sec))
        else:
            elapsed = max(0.0, time.monotonic() - self.session_start_wall)
            wall_estimated = self.session_start_offset + elapsed * max(rate, 0.1)
            if loop and self.duration_sec > 0:
                wall_estimated %= self.duration_sec
            else:
                wall_estimated = min(wall_estimated, self.duration_sec)
            # Never run ahead of the last published frame (keeps JSON timeline in sync with RGB).
            estimated_offset_sec = min(wall_estimated, current_offset_sec)

        return {
            "bagPath": str(self.meta.get("bagPath", self.cache_dir)),
            "containerName": self.container_name,
            "image": self.image_name,
            "rate": rate,
            "paused": paused,
            "loop": loop,
            "requestedOffsetSec": requested_offset_sec,
            "currentOffsetSec": current_offset_sec,
            "estimatedOffsetSec": estimated_offset_sec,
            "durationSec": self.duration_sec,
            "bagStartSec": self.start_sec,
            "bagEndSec": self.end_sec,
            "playbackAlive": playback_alive,
            "latestImageStampSec": latest_image_stamp_sec,
            "latestPoseStampSec": latest_pose_stamp_sec,
            "globalMapPoints": global_map_point_count,
            "currentScanPoints": current_scan_point_count,
            "topics": dict(self.topics),
            "mode": "cache",
            "cacheDir": str(self.cache_dir),
            "frameIndex": current_frame_index,
            "frameCount": len(self.frames),
            "keyframeCount": len(self.keyframes),
            "currentPose": current_pose,
            "video": self.playback_video_status(),
            "debug": {
                "workerState": self.debug_worker_state,
                "desiredOffsetSec": self.debug_desired_offset_sec,
                "targetIndex": self.debug_target_index,
                "lastAdvanceWallAgeSec": max(0.0, time.monotonic() - self.debug_last_advance_wall_sec),
            },
        }

    def apply_control(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            previous_rate = self.rate
            previous_paused = self.paused
            now_wall = time.monotonic()
            if "rate" in payload and payload["rate"] is not None:
                self.rate = max(0.1, float(payload["rate"]))
            if "paused" in payload and payload["paused"] is not None:
                self.paused = bool(payload["paused"])
            seek_requested = "offsetSec" in payload and payload["offsetSec"] is not None
            if seek_requested:
                self.requested_offset_sec = max(0.0, min(float(payload["offsetSec"]), self.duration_sec))
                target_index = self.offset_to_frame_index(self.requested_offset_sec)
                if 0 <= target_index < len(self.frames):
                    self.current_frame_index = target_index
                    self.current_offset_sec = float(self.frames[target_index]["offsetSec"])
                else:
                    self.current_offset_sec = self.requested_offset_sec
                self.control_version += 1
                self.session_start_offset = self.requested_offset_sec
                self.session_start_wall = now_wall
            else:
                self.requested_offset_sec = self.current_offset_sec
                if abs(self.rate - previous_rate) > 1e-6 or self.paused != previous_paused:
                    self.session_start_offset = self.current_offset_sec
                    self.session_start_wall = now_wall
            return self.status()

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=2.0)

    def get_latest_jpeg_frame(self) -> tuple[float | None, bytes | None]:
        with self.lock:
            return self.latest_image_stamp_sec, self.latest_image_jpeg

    def get_jpeg_frame_for_index(self, frame_index: int) -> tuple[float | None, bytes | None]:
        if not self.frames:
            return None, None
        clamped_index = max(0, min(int(frame_index), len(self.frames) - 1))
        frame = self.frames[clamped_index]
        image_rel = frame.get("imageFile")
        if not image_rel:
            return None, None
        stamp_sec = (
            float(frame.get("imageStampSec")) if frame.get("imageStampSec") is not None else float(frame["stampSec"])
        )
        return stamp_sec, load_binary_file(str((self.cache_dir / str(image_rel)).resolve()))

    def get_rectified_jpeg_frame_for_index(self, frame_index: int) -> tuple[float | None, bytes | None]:
        stamp_sec, frame = self.get_jpeg_frame_for_index(frame_index)
        if frame is None:
            return stamp_sec, None
        if (
            self.rectify_camera_matrix is None
            or self.rectify_new_camera_matrix is None
            or self.rectify_distortion is None
        ):
            return stamp_sec, frame

        encoded = np.frombuffer(frame, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            return stamp_sec, frame

        rectified = cv2.undistort(
            decoded,
            self.rectify_camera_matrix,
            self.rectify_distortion,
            None,
            self.rectify_new_camera_matrix,
        )
        ok, rectified_encoded = cv2.imencode(
            ".jpg",
            rectified,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        if not ok:
            return stamp_sec, frame
        return stamp_sec, rectified_encoded.tobytes()

    def offset_to_frame_index(self, offset_sec: float) -> int:
        if not self.frame_offsets:
            return -1
        clamped = max(0.0, min(offset_sec, self.duration_sec))
        index = bisect.bisect_right(self.frame_offsets, clamped) - 1
        return max(0, min(index, len(self.frame_offsets) - 1))

    def video_sync_map(
        self,
        fps: float,
        speed: float,
        start_offset_sec: float,
        end_offset_sec: float,
    ) -> dict[str, Any]:
        payload = build_video_sync_map(
            self.frame_offsets,
            self.duration_sec,
            fps,
            speed,
            start_offset_sec,
            end_offset_sec,
        )
        payload["mode"] = "cache-video-sync"
        payload["cacheDir"] = str(self.cache_dir)
        return payload

    def pose_track_state(self) -> dict[str, Any]:
        if self.pose_track_payload is None:
            self.pose_track_payload = {
                "mode": "cache-pose-track",
                "cacheDir": str(self.cache_dir),
                "frameCount": len(self.frames),
                "poses": [self.build_pose_payload(frame) for frame in self.frames],
            }
        return self.pose_track_payload

    def playback_video_status(self) -> dict[str, Any] | None:
        if not self.playback_video:
            return None
        return {
            "url": "video.mp4",
            "fps": float(self.playback_video["fps"]),
            "speed": float(self.playback_video["speed"]),
            "startOffsetSec": float(self.playback_video["startOffsetSec"]),
            "endOffsetSec": float(self.playback_video["endOffsetSec"]),
            "label": str(self.playback_video["label"]),
        }

    def get_playback_video_path(self) -> Path | None:
        if not self.playback_video:
            return None
        return Path(self.playback_video["path"])

    def build_pose_payload(self, frame: dict[str, Any]) -> dict[str, Any]:
        pose = frame.get("pose") or {}
        position = pose.get("position") or [0.0, 0.0, 0.0]
        orientation = pose.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        stamp_sec = float(pose.get("stampSec") or frame.get("stampSec") or 0.0)
        return {
            "frameId": str(pose.get("frameId") or "world"),
            "position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            },
            "orientation": {
                "x": float(orientation[0]),
                "y": float(orientation[1]),
                "z": float(orientation[2]),
                "w": float(orientation[3]),
            },
            "stampMs": int(round(stamp_sec * 1000.0)),
        }

    def reproject_colors_for_frame(
        self,
        frame: dict[str, Any],
        positions: np.ndarray,
        fallback_colors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = int(positions.shape[0])
        if count <= 0:
            return (
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=bool),
            )

        colors = copy_or_repeat_fallback_colors(fallback_colors, count)
        if self.rectify_calibration is None:
            return colors, np.zeros((count,), dtype=bool)

        image_rel = frame.get("imageFile")
        pose = frame.get("pose") or {}
        if not image_rel:
            return colors, np.zeros((count,), dtype=bool)

        image = load_rgb_image_file(str((self.cache_dir / str(image_rel)).resolve()))
        if image is None:
            return colors, np.zeros((count,), dtype=bool)

        try:
            world_to_lidar = np.linalg.inv(pose_record_to_matrix(pose) @ self.rectify_calibration.t_bl)
        except np.linalg.LinAlgError:
            return colors, np.zeros((count,), dtype=bool)

        calibration = self.rectify_calibration
        lidar_rotation = world_to_lidar[:3, :3]
        lidar_translation = world_to_lidar[:3, 3]
        lidar_points = positions.astype(np.float64, copy=False) @ lidar_rotation.T + lidar_translation

        camera_rotation = calibration.t_cl[:3, :3]
        camera_translation = calibration.t_cl[:3, 3]
        camera_points = lidar_points @ camera_rotation.T + camera_translation

        z = camera_points[:, 2]
        valid = np.isfinite(z) & (z > 0.05)
        if not np.any(valid):
            return colors, valid

        x = np.zeros((count,), dtype=np.float64)
        y = np.zeros((count,), dtype=np.float64)
        x[valid] = camera_points[valid, 0] / z[valid]
        y[valid] = camera_points[valid, 1] / z[valid]

        k1, k2, p1, p2, k3 = calibration.distortion.tolist()
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

        u = np.rint(calibration.fx * x_distorted + calibration.cx).astype(np.int32)
        v = np.rint(calibration.fy * y_distorted + calibration.cy).astype(np.int32)
        valid &= (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
        if not np.any(valid):
            return colors, valid

        colors[valid] = image[v[valid], u[valid]]
        return colors, valid

    def resolve_frame_index(self, frame_index: int | None) -> int:
        if not self.frames:
            return -1
        if frame_index is None:
            with self.lock:
                active_index = self.current_frame_index
            if active_index < 0:
                active_index = self.offset_to_frame_index(self.requested_offset_sec)
            return max(0, min(active_index, len(self.frames) - 1))
        return max(0, min(int(frame_index), len(self.frames) - 1))

    def overlay_state(
        self,
        frame_index: int | None = None,
        trajectory_max_points: int = 2048,
        scan_max_points: int = 1600,
        trajectory_tail_sec: float = 0.0,
        scan_history_frames: int | None = None,
        map_max_points: int = 0,
    ) -> dict[str, Any]:
        resolved_index = self.resolve_frame_index(frame_index)
        if resolved_index < 0 or resolved_index >= len(self.frames):
            return {
                "frameIndex": -1,
                "trajectory": [],
                "scan": None,
                "map": None,
            }

        current_frame = self.frames[resolved_index]
        current_stamp_sec = float(current_frame["stampSec"])
        trajectory_start_index = 0
        if trajectory_tail_sec > 0.0:
            cutoff_stamp_sec = current_stamp_sec - max(trajectory_tail_sec, 0.0)
            trajectory_start_index = resolved_index
            while trajectory_start_index > 0:
                previous_stamp_sec = float(self.frames[trajectory_start_index - 1]["stampSec"])
                if previous_stamp_sec < cutoff_stamp_sec:
                    break
                trajectory_start_index -= 1

        trajectory: list[dict[str, Any]] = []
        if int(trajectory_max_points) > 0:
            trajectory_limit = max(int(trajectory_max_points), 1)
            trajectory_count = resolved_index - trajectory_start_index + 1
            trajectory_stride = max(1, int(math.ceil(float(trajectory_count) / float(trajectory_limit))))
            for index in range(trajectory_start_index, resolved_index + 1, trajectory_stride):
                trajectory.append(self.build_pose_payload(self.frames[index]))
            if trajectory and trajectory[-1]["stampMs"] != self.build_pose_payload(current_frame)["stampMs"]:
                trajectory.append(self.build_pose_payload(current_frame))

        scan_stamp_ms = int(round(float(current_frame["stampSec"]) * 1000.0))
        scan_payload: dict[str, Any] | None = None
        if int(scan_max_points) > 0:
            scan_limit = max(int(scan_max_points), 1)
            history_frames = max(
                int(
                    scan_history_frames
                    if scan_history_frames is not None
                    else (getattr(self.args, "scan_history_frames", 4) or 1)
                ),
                1,
            )
            sampled_positions, sampled_colors = self.compose_recent_scan_cloud(
                resolved_index,
                scan_limit,
                history_frames,
            )
            scan_payload = {
                "frameId": "world",
                "stampMs": scan_stamp_ms,
                "renderedPointCount": int(sampled_positions.shape[0]),
                "sourcePointCount": int(sampled_positions.shape[0]),
                "positions": np.round(sampled_positions.reshape(-1), 4).astype(float).tolist(),
                "colors": sampled_colors.reshape(-1).astype(int).tolist(),
            }
        return {
            "frameIndex": resolved_index,
            "trajectory": trajectory,
            "scan": scan_payload,
            "map": self.build_overlay_map_payload(
                resolved_index,
                max_points=max(int(map_max_points), 0),
            ),
        }

    def restore_map_from_keyframe_into_voxels(
        self,
        frame_index: int,
        voxels: dict[tuple[int, int, int], MapVoxel],
    ) -> int:
        voxels.clear()
        if not self.keyframe_indices:
            return -1

        keyframe_slot = bisect.bisect_right(self.keyframe_indices, frame_index) - 1
        if keyframe_slot < 0:
            return -1

        keyframe_frame_index = self.keyframe_indices[keyframe_slot]
        positions, restored_colors, restored_valid_mask = load_clean_cloud_npz(
            self.keyframe_files[keyframe_slot],
            int(WEB_MAP_NEUTRAL_FALLBACK[0]),
            int(WEB_MAP_NEUTRAL_FALLBACK[1]),
            int(WEB_MAP_NEUTRAL_FALLBACK[2]),
        )
        keyframe_frame = self.frames[keyframe_frame_index]
        stamp_sec = float(keyframe_frame["stampSec"])
        voxel_size = max(float(self.meta["mapVoxelSize"]), 1e-3)

        for index in range(positions.shape[0]):
            x = float(positions[index, 0])
            y = float(positions[index, 1])
            z = float(positions[index, 2])
            key = (
                int(round(x / voxel_size)),
                int(round(y / voxel_size)),
                int(round(z / voxel_size)),
            )
            voxels[key] = MapVoxel(
                x=x,
                y=y,
                z=z,
                position_samples=1,
                color=restored_colors[index].astype(np.float64),
                color_samples=1 if bool(restored_valid_mask[index]) else 0,
                last_seen_stamp_sec=stamp_sec,
            )

        return keyframe_frame_index

    def update_map_voxels_from_scan(
        self,
        voxels: dict[tuple[int, int, int], MapVoxel],
        positions: np.ndarray,
        colors: np.ndarray,
        stamp_sec: float,
        color_valid_mask: np.ndarray | None = None,
    ) -> None:
        voxel_size = max(float(self.meta.get("mapVoxelSize", 0.2)), 1e-3)
        max_points = int(self.meta.get("mapMaxPoints", 90_000))
        for index in range(positions.shape[0]):
            x = float(positions[index, 0])
            y = float(positions[index, 1])
            z = float(positions[index, 2])
            key = (
                int(round(x / voxel_size)),
                int(round(y / voxel_size)),
                int(round(z / voxel_size)),
            )
            existing = voxels.get(key)
            incoming_color = colors[index].astype(np.float64)
            color_is_valid = True if color_valid_mask is None else bool(color_valid_mask[index])
            if existing is None:
                voxels[key] = MapVoxel(
                    x=x,
                    y=y,
                    z=z,
                    position_samples=1,
                    color=incoming_color,
                    color_samples=1 if color_is_valid else 0,
                    last_seen_stamp_sec=stamp_sec,
                )
                continue

            existing.position_samples += 1
            mix = 1.0 / float(existing.position_samples)
            existing.x += (x - existing.x) * mix
            existing.y += (y - existing.y) * mix
            existing.z += (z - existing.z) * mix
            stale_gap_sec = max(0.0, stamp_sec - existing.last_seen_stamp_sec)
            if not color_is_valid:
                existing.last_seen_stamp_sec = stamp_sec
                continue
            if existing.color is None or existing.color_samples <= 0:
                existing.color = incoming_color
                existing.color_samples = 1
            else:
                existing.color_samples += 1
                if stale_gap_sec >= 0.45:
                    color_mix = 0.58
                elif existing.color_samples <= 3:
                    color_mix = 0.46
                elif existing.color_samples <= 8:
                    color_mix = 0.31
                elif existing.color_samples <= 20:
                    color_mix = 0.19
                else:
                    color_mix = 0.11
                existing.color += (incoming_color - existing.color) * color_mix
            existing.last_seen_stamp_sec = stamp_sec

        trim_map_voxels(voxels, max_points)

    def build_map_payload_from_voxels(
        self,
        voxels: dict[tuple[int, int, int], MapVoxel],
        snapshot_frame_index: int,
        max_points: int,
    ) -> dict[str, Any] | None:
        if max_points <= 0 or not voxels:
            return None

        voxel_values = list(voxels.values())
        return self.build_map_payload_from_voxel_values(
            voxel_values,
            snapshot_frame_index,
            max_points,
        )

    def build_map_payload_from_voxel_values(
        self,
        voxel_values: list[MapVoxel],
        snapshot_frame_index: int,
        max_points: int,
    ) -> dict[str, Any] | None:
        if max_points <= 0 or not voxel_values:
            return None

        n = len(voxel_values)
        positions = np.empty((n, 3), dtype=np.float32)
        colors = np.empty((n, 3), dtype=np.uint8)
        color_valid_mask = np.zeros((n,), dtype=bool)
        default_rgb = np.array([216, 236, 255], dtype=np.uint8)
        for index, voxel in enumerate(voxel_values):
            positions[index, 0] = voxel.x
            positions[index, 1] = voxel.y
            positions[index, 2] = voxel.z
            c = voxel.color
            if c is None:
                colors[index] = default_rgb
            else:
                colors[index] = np.clip(np.round(c), 0, 255).astype(np.uint8)
            color_valid_mask[index] = bool((voxel.color is not None) and (voxel.color_samples > 0))

        return self.build_map_payload_from_arrays(
            positions,
            colors,
            color_valid_mask,
            snapshot_frame_index,
            max_points,
        )

    def build_map_payload_from_arrays(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        color_valid_mask: np.ndarray,
        snapshot_frame_index: int,
        max_points: int,
    ) -> dict[str, Any] | None:
        if max_points <= 0 or int(positions.shape[0]) <= 0:
            return None

        sampled_positions, sampled_colors, sampled_valid_mask = sample_cloud_prioritizing_valid(
            positions,
            colors,
            color_valid_mask,
            max_points,
        )
        sampled_colors, _ = neutralize_legacy_false_colors(
            sampled_colors,
            WEB_MAP_NEUTRAL_FALLBACK,
            ~sampled_valid_mask,
        )
        snapshot_frame_index = max(0, min(snapshot_frame_index, len(self.frames) - 1))
        snapshot_frame = self.frames[snapshot_frame_index]
        return {
            "frameIndex": snapshot_frame_index,
            "frameId": "world",
            "stampMs": int(round(float(snapshot_frame["stampSec"]) * 1000.0)),
            "renderedPointCount": int(sampled_positions.shape[0]),
            "sourcePointCount": int(positions.shape[0]),
            "positions": np.round(sampled_positions.reshape(-1), 4).astype(float).tolist(),
            "colors": sampled_colors.reshape(-1).astype(int).tolist(),
        }

    def refresh_overlay_live_map_payloads(self, frame_index: int) -> None:
        payloads: dict[int, dict[str, Any]] = {}
        for max_points in (3_000, 6_000):
            payload = self.build_map_payload_from_voxels(
                self.global_map_voxels,
                frame_index,
                max_points,
            )
            if payload is not None:
                payloads[max_points] = payload
        with self.lock:
            self._overlay_live_map_payloads = payloads
            self._overlay_map_cache_key = None
            self._overlay_map_cache_payload = None

    def schedule_refresh_overlay_live_map_payloads(self, frame_index: int) -> None:
        with self._overlay_map_refresh_guard:
            if self._overlay_map_refresh_busy:
                self._overlay_map_refresh_pending_frame_index = int(frame_index)
                return
            self._overlay_map_refresh_busy = True
            self._overlay_map_refresh_pending_frame_index = None

        def run(next_frame_index: int) -> None:
            try:
                while True:
                    voxel_values: list[MapVoxel] | None = None
                    for _ in range(3):
                        try:
                            voxel_values = list(self.global_map_voxels.values())
                            break
                        except RuntimeError:
                            time.sleep(0.002)
                    if voxel_values is not None:
                        payloads: dict[int, dict[str, Any]] = {}
                        for max_points in (3_000, 6_000):
                            payload = self.build_map_payload_from_voxel_values(
                                voxel_values,
                                next_frame_index,
                                max_points,
                            )
                            if payload is not None:
                                payloads[max_points] = payload
                        with self.lock:
                            self._overlay_live_map_payloads = payloads
                            self._overlay_map_cache_key = None
                            self._overlay_map_cache_payload = None

                    with self._overlay_map_refresh_guard:
                        pending = self._overlay_map_refresh_pending_frame_index
                        if pending is None:
                            self._overlay_map_refresh_busy = False
                            return
                        self._overlay_map_refresh_pending_frame_index = None
                        next_frame_index = int(pending)
            except Exception:
                with self._overlay_map_refresh_guard:
                    self._overlay_map_refresh_busy = False
                raise

        threading.Thread(
            target=run,
            args=(int(frame_index),),
            daemon=True,
            name="cached-overlay-map-refresh",
        ).start()

    def build_overlay_map_payload(
        self,
        requested_frame_index: int,
        max_points: int,
    ) -> dict[str, Any] | None:
        resolved_index = self.resolve_frame_index(requested_frame_index)
        if max_points <= 0 or resolved_index < 0 or resolved_index >= len(self.frames):
            return None
        with self.lock:
            if self._overlay_map_cache_key == (resolved_index, max_points) and self._overlay_map_cache_payload is not None:
                return dict(self._overlay_map_cache_payload)

        keyframe_slot = bisect.bisect_right(self.keyframe_indices, resolved_index) - 1
        if keyframe_slot < 0:
            return None

        restored_keyframe_index = self.keyframe_indices[keyframe_slot]
        keyframe_positions, keyframe_colors, keyframe_valid_mask = load_clean_cloud_npz(
            self.keyframe_files[keyframe_slot],
            int(WEB_MAP_NEUTRAL_FALLBACK[0]),
            int(WEB_MAP_NEUTRAL_FALLBACK[1]),
            int(WEB_MAP_NEUTRAL_FALLBACK[2]),
        )
        frames_after_keyframe = max(0, resolved_index - restored_keyframe_index)
        # Keep the accumulated map globally dense. The older version spent too much
        # of the point budget on very recent scans, which made the current robot
        # neighborhood look dense while already-visited areas became sparse.
        recent_fraction = min(0.22, 0.012 * float(frames_after_keyframe))
        recent_budget = int(round(max_points * recent_fraction)) if frames_after_keyframe > 0 else 0
        keyframe_budget = max(max_points - recent_budget, 1)
        sampled_keyframe_positions, sampled_keyframe_colors, sampled_keyframe_valid_mask = (
            sample_cloud_prioritizing_valid(
                keyframe_positions,
                keyframe_colors,
                keyframe_valid_mask,
                keyframe_budget,
            )
        )

        merged_positions: list[np.ndarray] = [sampled_keyframe_positions]
        merged_colors: list[np.ndarray] = [sampled_keyframe_colors]
        merged_valid_masks: list[np.ndarray] = [sampled_keyframe_valid_mask]

        fuse_cap = max(int(getattr(self.args, "map_fuse_max_points", 0) or 0), 1)
        replay_indices = list(range(restored_keyframe_index + 1, resolved_index + 1))
        if replay_indices and recent_budget > 0:
            weights = np.array(
                [0.28 + 0.72 * (0.62 ** (resolved_index - index)) for index in replay_indices],
                dtype=np.float64,
            )
            weights /= max(float(weights.sum()), 1e-6)
            allocations = np.maximum(1, np.floor(weights * recent_budget).astype(np.int32))
            while int(allocations.sum()) < recent_budget:
                allocations[np.argmax(weights)] += 1
            while int(allocations.sum()) > recent_budget:
                victim = int(np.argmax(allocations))
                if allocations[victim] <= 1:
                    break
                allocations[victim] -= 1
        else:
            allocations = np.zeros((len(replay_indices),), dtype=np.int32)

        for index, allocation in zip(replay_indices, allocations.tolist()):
            if allocation <= 0:
                continue
            frame = self.frames[index]
            positions, scan_colors, scan_valid_mask = load_clean_cloud_npz(
                str((self.cache_dir / frame["scanFile"]).resolve()),
                int(WEB_MAP_NEUTRAL_FALLBACK[0]),
                int(WEB_MAP_NEUTRAL_FALLBACK[1]),
                int(WEB_MAP_NEUTRAL_FALLBACK[2]),
            )
            positions_fuse, colors_fuse, colors_fuse_valid_mask = sample_cloud_prioritizing_valid(
                positions,
                scan_colors,
                scan_valid_mask,
                min(fuse_cap, allocation),
            )
            if positions_fuse.shape[0] <= 0:
                continue
            merged_positions.append(np.asarray(positions_fuse, dtype=np.float32, order="C"))
            merged_colors.append(np.asarray(colors_fuse, dtype=np.uint8, order="C"))
            merged_valid_masks.append(np.asarray(colors_fuse_valid_mask, dtype=bool))

        payload = self.build_map_payload_from_arrays(
            np.concatenate(merged_positions, axis=0),
            np.concatenate(merged_colors, axis=0),
            np.concatenate(merged_valid_masks, axis=0),
            resolved_index,
            max_points,
        )
        if payload is None:
            return None

        with self.lock:
            self._overlay_map_cache_key = (resolved_index, max_points)
            self._overlay_map_cache_payload = dict(payload)
            return dict(payload)

    def restore_map_from_keyframe(self, frame_index: int) -> int:
        self.global_map_voxels = {}
        keyframe_frame_index = self.restore_map_from_keyframe_into_voxels(
            frame_index,
            self.global_map_voxels,
        )
        self.global_map_point_count = len(self.global_map_voxels)
        return keyframe_frame_index

    def update_global_map_from_scan(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        stamp_sec: float,
        color_valid_mask: np.ndarray | None = None,
    ) -> None:
        self.update_map_voxels_from_scan(
            self.global_map_voxels,
            positions,
            colors,
            stamp_sec,
            color_valid_mask,
        )
        self.global_map_point_count = len(self.global_map_voxels)

    def compose_recent_scan_cloud(
        self,
        frame_index: int,
        total_max_points: int,
        history_frames: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        resolved_index = self.resolve_frame_index(frame_index)
        if resolved_index < 0 or resolved_index >= len(self.frames):
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        history = max(int(history_frames), 1)
        start_index = max(0, resolved_index - history + 1)
        indices = list(range(start_index, resolved_index + 1))
        if not indices:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )

        weights = np.array([0.28 + 0.72 * (0.62 ** (resolved_index - idx)) for idx in indices], dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-6)
        point_budget = max(int(total_max_points), 1)
        allocations = np.maximum(1, np.floor(weights * point_budget).astype(np.int32))
        while int(allocations.sum()) < point_budget:
            allocations[np.argmax(weights)] += 1
        while int(allocations.sum()) > point_budget:
            victim = int(np.argmax(allocations))
            if allocations[victim] <= 1:
                break
            allocations[victim] -= 1

        merged_positions: list[np.ndarray] = []
        merged_colors: list[np.ndarray] = []
        for index, allocation in zip(indices, allocations.tolist()):
            frame = self.frames[index]
            positions, reprojected_colors, valid_mask = load_clean_cloud_npz(
                str((self.cache_dir / frame["scanFile"]).resolve()),
                int(WEB_SCAN_NEUTRAL_FALLBACK[0]),
                int(WEB_SCAN_NEUTRAL_FALLBACK[1]),
                int(WEB_SCAN_NEUTRAL_FALLBACK[2]),
            )
            sampled_positions, sampled_colors, sampled_valid_mask = sample_cloud_prioritizing_valid(
                positions,
                reprojected_colors,
                valid_mask,
                allocation,
            )
            if sampled_positions.shape[0] <= 0:
                continue
            age = resolved_index - index
            sampled_colors = blend_scan_history_colors(sampled_colors, age)
            merged_positions.append(np.asarray(sampled_positions, dtype=np.float32, order="C"))
            merged_colors.append(np.asarray(sampled_colors, dtype=np.uint8, order="C"))

        return (
            np.concatenate(merged_positions, axis=0) if merged_positions else np.empty((0, 3), dtype=np.float32),
            np.concatenate(merged_colors, axis=0) if merged_colors else np.empty((0, 3), dtype=np.uint8),
        )

    def publish_global_map(self, frame: dict[str, Any]) -> None:
        voxels = list(self.global_map_voxels.values())
        if not voxels:
            return

        n = len(voxels)
        positions = np.empty((n, 3), dtype=np.float32)
        colors = np.empty((n, 3), dtype=np.uint8)
        color_valid_mask = np.zeros((n,), dtype=bool)
        default_rgb = np.array([216, 236, 255], dtype=np.uint8)
        for index, voxel in enumerate(voxels):
            positions[index, 0] = voxel.x
            positions[index, 1] = voxel.y
            positions[index, 2] = voxel.z
            c = voxel.color
            if c is None:
                colors[index] = default_rgb
            else:
                colors[index] = np.clip(np.round(c), 0, 255).astype(np.uint8)
            color_valid_mask[index] = bool((voxel.color is not None) and (voxel.color_samples > 0))

        stamp_sec = float(frame["stampSec"])
        header = Header(frame_id="world", stamp=rospy.Time.from_sec(stamp_sec))
        self.global_map_pub.publish(create_rgb_cloud(header, positions, colors))
        web_positions, web_colors, web_valid_mask = sample_cloud_prioritizing_valid(
            positions,
            colors,
            color_valid_mask,
            max(int(getattr(self.args, "web_map_max_points", 32000)), 1),
        )
        web_colors, _ = neutralize_legacy_false_colors(
            web_colors,
            WEB_MAP_NEUTRAL_FALLBACK,
            ~web_valid_mask,
        )
        self.global_map_web_pub.publish(create_rgb_cloud(header, web_positions, web_colors))

    def schedule_publish_global_map(self, frame: dict[str, Any]) -> None:
        """Publish map clouds off the playback thread; coalesce to the latest requested frame."""

        with self._map_publish_guard:
            if self._map_publish_busy:
                self._map_publish_pending_frame = dict(frame)
                return
            self._map_publish_busy = True
            self._map_publish_pending_frame = None

        def run(frame_payload: dict[str, Any]) -> None:
            try:
                with self._map_publish_guard:
                    next_frame_payload = dict(frame_payload)

                while True:
                    try:
                        self.publish_global_map(next_frame_payload)
                    except Exception as exc:
                        rospy.logwarn("publish_global_map failed: %s", exc)

                    with self._map_publish_guard:
                        pending = self._map_publish_pending_frame
                        if pending is None:
                            self._map_publish_busy = False
                            return
                        self._map_publish_pending_frame = None
                        next_frame_payload = dict(pending)
            except Exception:
                with self._map_publish_guard:
                    self._map_publish_busy = False
                raise

        threading.Thread(
            target=run, args=(dict(frame),), daemon=True, name="cached-map-publish"
        ).start()

    def publish_pose(self, frame: dict[str, Any]) -> None:
        pose = frame["pose"]
        stamp_sec = float(frame["stampSec"])
        message = Odometry()
        message.header.stamp = rospy.Time.from_sec(stamp_sec)
        message.header.frame_id = str(pose.get("frameId") or "world")
        message.child_frame_id = str(pose.get("childFrameId") or "base_link")
        position = pose.get("position") or [0.0, 0.0, 0.0]
        orientation = pose.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        message.pose.pose.position.x = float(position[0])
        message.pose.pose.position.y = float(position[1])
        message.pose.pose.position.z = float(position[2])
        message.pose.pose.orientation.x = float(orientation[0])
        message.pose.pose.orientation.y = float(orientation[1])
        message.pose.pose.orientation.z = float(orientation[2])
        message.pose.pose.orientation.w = float(orientation[3])
        self.odom_pub.publish(message)
        self.latest_pose_stamp_sec = float(pose.get("stampSec") or stamp_sec)

    def publish_scan(self, frame: dict[str, Any], positions: np.ndarray, colors: np.ndarray) -> None:
        stamp_sec = float(frame["stampSec"])
        header = Header(frame_id="world", stamp=rospy.Time.from_sec(stamp_sec))
        self.current_scan_world_pub.publish(create_rgb_cloud(header, positions, colors))
        web_positions, web_colors = self.compose_recent_scan_cloud(
            int(frame.get("index", self.current_frame_index if self.current_frame_index >= 0 else 0)),
            max(int(getattr(self.args, "web_scan_max_points", 5000)), 1),
            max(int(getattr(self.args, "scan_history_frames", 4) or 1), 1),
        )
        self.current_scan_world_web_pub.publish(create_rgb_cloud(header, web_positions, web_colors))
        self.current_scan_point_count = int(positions.shape[0])

    def set_latest_image(self, frame: dict[str, Any]) -> None:
        image_rel = frame.get("imageFile")
        if image_rel:
            self.latest_image_jpeg = load_binary_file(str((self.cache_dir / str(image_rel)).resolve()))
            self.latest_image_stamp_sec = (
                float(frame.get("imageStampSec")) if frame.get("imageStampSec") is not None else None
            )
        else:
            self.latest_image_jpeg = None
            self.latest_image_stamp_sec = None

    def restore_to_frame(self, frame_index: int) -> None:
        if not self.frames:
            return

        frame_index = max(0, min(frame_index, len(self.frames) - 1))
        restored_keyframe_index = self.restore_map_from_keyframe(frame_index)
        replay_max_frames = max(int(getattr(self.args, "restore_replay_max_frames", 8) or 0), 0)
        start_index = restored_keyframe_index + 1
        if replay_max_frames > 0:
            start_index = max(start_index, frame_index - replay_max_frames + 1)
        current_positions: np.ndarray | None = None
        current_colors: np.ndarray | None = None
        current_frame = self.frames[frame_index]
        fuse_cap = int(getattr(self.args, "map_fuse_max_points", 0) or 0)

        for index in range(start_index, frame_index + 1):
            frame = self.frames[index]
            positions, colors_reprojected, colors_valid_mask = load_clean_cloud_npz(
                str((self.cache_dir / frame["scanFile"]).resolve()),
                int(WEB_MAP_NEUTRAL_FALLBACK[0]),
                int(WEB_MAP_NEUTRAL_FALLBACK[1]),
                int(WEB_MAP_NEUTRAL_FALLBACK[2]),
            )
            positions_fuse, colors_fuse, colors_fuse_valid_mask = sample_cloud_prioritizing_valid(
                positions,
                colors_reprojected,
                colors_valid_mask,
                fuse_cap,
            )
            self.update_global_map_from_scan(
                positions_fuse,
                colors_fuse,
                float(frame["stampSec"]),
                colors_fuse_valid_mask,
            )
            if index == frame_index:
                current_positions = positions
                current_colors = colors

        if current_positions is None or current_colors is None:
            current_positions, current_colors = load_cloud_npz(
                str((self.cache_dir / current_frame["scanFile"]).resolve())
            )

        stamp_sec = float(current_frame["stampSec"])
        self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(stamp_sec)))
        self.publish_pose(current_frame)
        self.publish_scan(current_frame, current_positions, current_colors)
        self.set_latest_image(current_frame)
        self.schedule_publish_global_map(current_frame)
        self.current_frame_index = frame_index
        self.current_offset_sec = float(current_frame["offsetSec"])
        self.playback_alive = True

    def run(self) -> None:
        if not self.frames:
            return

        current_version = -1
        needs_restore = True

        while not self.stop_event.is_set() and not rospy.is_shutdown():
            with self.lock:
                paused = self.paused
                version = self.control_version
                requested_offset = self.requested_offset_sec
                rate = self.rate
                loop = self.loop
                session_start_offset = self.session_start_offset
                session_start_wall = self.session_start_wall

            if needs_restore or version != current_version:
                self.debug_worker_state = "restore"
                target_index = self.offset_to_frame_index(requested_offset)
                self.restore_to_frame(target_index)
                with self.lock:
                    current_version = version
                    self.session_start_offset = self.current_offset_sec
                    self.session_start_wall = time.monotonic()
                needs_restore = False
                continue

            if paused:
                self.debug_worker_state = "paused"
                time.sleep(0.05)
                continue

            desired_offset = session_start_offset + max(0.0, time.monotonic() - session_start_wall) * max(rate, 0.1)
            if loop and self.duration_sec > 0:
                desired_offset %= self.duration_sec
            else:
                desired_offset = min(desired_offset, self.duration_sec)
            self.debug_desired_offset_sec = desired_offset

            target_index = self.offset_to_frame_index(desired_offset)
            self.debug_target_index = target_index
            if target_index < 0:
                self.debug_worker_state = "invalid-target"
                time.sleep(0.002)
                continue

            if target_index <= self.current_frame_index:
                self.debug_worker_state = "waiting-for-next-frame"
                if desired_offset >= self.duration_sec and self.current_frame_index >= len(self.frames) - 1:
                    with self.lock:
                        if loop:
                            self.requested_offset_sec = 0.0
                            self.current_offset_sec = 0.0
                            self.control_version += 1
                        else:
                            self.paused = True
                            self.playback_alive = False
                    needs_restore = True
                else:
                    time.sleep(0.002)
                continue

            next_index = min(target_index, len(self.frames) - 1)
            if next_index >= len(self.frames):
                self.debug_worker_state = "past-end"
                with self.lock:
                    if loop:
                        self.requested_offset_sec = 0.0
                        self.current_offset_sec = 0.0
                        self.control_version += 1
                    else:
                        self.paused = True
                        self.playback_alive = False
                needs_restore = True
                continue

            next_frame = self.frames[next_index]
            positions, colors = load_cloud_npz(str((self.cache_dir / next_frame["scanFile"]).resolve()))
            fuse_cap = int(getattr(self.args, "map_fuse_max_points", 0) or 0)
            _, colors_reprojected, colors_valid_mask = load_clean_cloud_npz(
                str((self.cache_dir / next_frame["scanFile"]).resolve()),
                int(WEB_MAP_NEUTRAL_FALLBACK[0]),
                int(WEB_MAP_NEUTRAL_FALLBACK[1]),
                int(WEB_MAP_NEUTRAL_FALLBACK[2]),
            )
            positions_fuse, colors_fuse, colors_fuse_valid_mask = sample_cloud_prioritizing_valid(
                positions,
                colors_reprojected,
                colors_valid_mask,
                fuse_cap,
            )
            stamp_sec = float(next_frame["stampSec"])
            with self.lock:
                if self.control_version != current_version or self.paused:
                    needs_restore = True
            if needs_restore:
                self.debug_worker_state = "needs-restore-before-publish"
                continue
            self.debug_worker_state = "publishing"
            # Refresh JPEG before heavy map work so /frame.mjpeg and /frame.jpg stay in step with playback
            # (global_map publish can block for a long time and previously made RGB look "frozen").
            self.set_latest_image(next_frame)
            self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(stamp_sec)))
            self.publish_pose(next_frame)
            self.publish_scan(next_frame, positions, colors)
            self.update_global_map_from_scan(
                positions_fuse,
                colors_fuse,
                stamp_sec,
                colors_fuse_valid_mask,
            )
            map_publish_every = max(
                int(
                    getattr(self.args, "map_publish_every", 0)
                    or int(self.meta.get("keyframeEvery", 64) // 8)
                ),
                1,
            )
            if next_index <= 1 or next_index % map_publish_every == 0:
                self.schedule_publish_global_map(next_frame)
            with self.lock:
                if self.control_version != current_version or self.paused:
                    needs_restore = True
                else:
                    self.current_frame_index = next_index
                    self.current_offset_sec = float(next_frame["offsetSec"])
                    self.playback_alive = True
                    self.debug_worker_state = "advanced"
                    self.debug_last_advance_wall_sec = time.monotonic()


class Handler(BaseHTTPRequestHandler):
    controller: CachedPlaybackController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_common_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/video.mp4":
            self.respond_playback_video(head_only=True)
            return
        if parsed.path == "/status":
            body = json.dumps(self.controller.status()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_common_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/status":
            self.respond_json(self.controller.status())
            return
        if parsed.path == "/topics":
            self.respond_json({"topics": self.controller.topics})
            return
        if parsed.path == "/overlay.json":
            query = parse_qs(parsed.query)
            frame_index_raw = query.get("frameIndex", [None])[0]
            trajectory_max_points_raw = query.get("trajectoryMaxPoints", ["2048"])[0] or "2048"
            trajectory_tail_sec_raw = query.get("trajectoryTailSec", ["0"])[0] or "0"
            scan_max_points_raw = query.get("scanMaxPoints", ["1600"])[0] or "1600"
            scan_history_frames_raw = query.get("scanHistoryFrames", ["0"])[0] or "0"
            map_max_points_raw = query.get("mapMaxPoints", ["0"])[0] or "0"
            try:
                frame_index = int(frame_index_raw) if frame_index_raw is not None else None
                trajectory_max_points = int(trajectory_max_points_raw)
                trajectory_tail_sec = float(trajectory_tail_sec_raw)
                scan_max_points = int(scan_max_points_raw)
                scan_history_frames = int(scan_history_frames_raw)
                map_max_points = int(map_max_points_raw)
            except ValueError:
                self.respond_json({"error": "Invalid overlay query."}, HTTPStatus.BAD_REQUEST)
                return
            self.respond_json(
                self.controller.overlay_state(
                    frame_index=frame_index,
                    trajectory_max_points=trajectory_max_points,
                    trajectory_tail_sec=trajectory_tail_sec,
                    scan_max_points=scan_max_points,
                    scan_history_frames=scan_history_frames if scan_history_frames > 0 else None,
                    map_max_points=map_max_points,
                )
            )
            return
        if parsed.path == "/video_sync":
            query = parse_qs(parsed.query)
            fps = float(query.get("fps", ["12"])[0] or "12")
            speed = float(query.get("speed", ["1"])[0] or "1")
            start_offset_sec = float(query.get("startOffsetSec", ["0"])[0] or "0")
            end_offset_sec = float(query.get("endOffsetSec", ["-1"])[0] or "-1")
            self.respond_json(
                self.controller.video_sync_map(
                    fps=fps,
                    speed=speed,
                    start_offset_sec=start_offset_sec,
                    end_offset_sec=end_offset_sec,
                )
            )
            return
        if parsed.path == "/pose_track.json":
            self.respond_json(self.controller.pose_track_state())
            return
        if parsed.path == "/video.mp4":
            self.respond_playback_video()
            return
        if parsed.path.startswith("/frame.jpg"):
            self.respond_jpeg_frame(parsed)
            return
        if parsed.path.startswith("/frame.mjpeg"):
            self.respond_mjpeg_stream()
            return
        self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/control":
            self.respond_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        self.respond_json(self.controller.apply_control(payload))

    def respond_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_jpeg_frame(self, parsed) -> None:
        query = parse_qs(parsed.query)
        frame_index_raw = query.get("frameIndex", [None])[0]
        rectified = query.get("rectified", ["0"])[0] in {"1", "true", "yes"}
        if frame_index_raw is not None:
            try:
                if rectified:
                    stamp_sec, frame = self.controller.get_rectified_jpeg_frame_for_index(int(frame_index_raw))
                else:
                    stamp_sec, frame = self.controller.get_jpeg_frame_for_index(int(frame_index_raw))
            except ValueError:
                self.respond_json({"error": f"Invalid frameIndex: {frame_index_raw}"}, HTTPStatus.BAD_REQUEST)
                return
        else:
            stamp_sec, frame = self.controller.get_latest_jpeg_frame()
        if frame is None:
            self.respond_json({"error": "No image frame available yet."}, HTTPStatus.SERVICE_UNAVAILABLE)
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

    def respond_playback_video(self, head_only: bool = False) -> None:
        path = self.controller.get_playback_video_path()
        if path is None or not path.exists():
            self.respond_json({"error": "Playback video is not configured."}, HTTPStatus.NOT_FOUND)
            return
        self.respond_file(path, "video/mp4", head_only=head_only)

    def respond_file(self, path: Path, content_type: str, head_only: bool = False) -> None:
        file_size = path.stat().st_size
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        range_header = self.headers.get("Range", "").strip()
        if range_header.startswith("bytes="):
            byte_range = range_header[6:].split(",", 1)[0].strip()
            start_text, _, end_text = byte_range.partition("-")
            try:
                if start_text:
                    start = int(start_text)
                    end = int(end_text) if end_text else end
                elif end_text:
                    suffix_length = int(end_text)
                    start = max(file_size - suffix_length, 0)
                else:
                    raise ValueError("empty range")
            except ValueError:
                self.respond_json({"error": f"Invalid Range header: {range_header}"}, HTTPStatus.BAD_REQUEST)
                return

            end = min(end, file_size - 1)
            if start < 0 or start > end or start >= file_size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_common_headers()
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = max(end - start + 1, 0)
        self.send_response(status)
        self.send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if head_only:
            return

        with path.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(remaining, 64 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

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
                stamp_sec, frame = self.controller.get_latest_jpeg_frame()
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

                mjpeg_max_fps = max(0.0, float(getattr(self.controller.args, "mjpeg_max_fps", 60.0)))
                if mjpeg_max_fps > 0.0:
                    if self.controller.stop_event.wait(1.0 / mjpeg_max_fps):
                        return
                elif self.controller.stop_event.wait(0.005):
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
    cache_dir = Path(args.cache_dir).resolve()
    if not (cache_dir / "meta.json").exists():
        raise SystemExit(f"Cache metadata missing: {cache_dir / 'meta.json'}")

    rospy.init_node("fastlivo_cached_playback", anonymous=False, disable_signals=True)
    controller = CachedPlaybackController(cache_dir, args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        rospy.loginfo("cached playback ready on http://0.0.0.0:%s", args.port)
        server.serve_forever()
    finally:
        controller.close()


if __name__ == "__main__":
    main()
