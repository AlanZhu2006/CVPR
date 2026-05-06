#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import threading
import time
from typing import Any

import numpy as np
from PIL import Image


class BinaryPointCloudWebSocketServer:
    """Small binary WebSocket publisher compatible with GS Console live mode."""

    MAGIC = b"LBPC1"
    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, *, host: str, port: int, max_points: int, frame_id: str = "map") -> None:
        self.host = str(host)
        self.port = int(port)
        self.max_points = int(max_points)
        self.frame_id = str(frame_id)
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self.connected_clients = 0
        self.published_messages = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._serve, name=f"binary-cloud-ws-{port}", daemon=True)
        self._thread.start()

    def publish_cloud(self, xyz: np.ndarray, rgb: np.ndarray, *, stamp_ms: int | None = None) -> None:
        xyz_arr = np.asarray(xyz, dtype=np.float32)
        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        if xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3 or xyz_arr.shape[0] == 0:
            return
        if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3 or rgb_arr.shape[0] != xyz_arr.shape[0]:
            rgb_arr = np.full((xyz_arr.shape[0], 3), 255, dtype=np.uint8)

        source_count = int(xyz_arr.shape[0])
        if self.max_points > 0 and source_count > self.max_points:
            keep = np.linspace(0, source_count - 1, self.max_points).astype(np.int64)
            xyz_arr = xyz_arr[keep]
            rgb_arr = rgb_arr[keep]

        xyz_bytes = np.ascontiguousarray(xyz_arr.astype("<f4", copy=False)).tobytes()
        rgb_bytes = np.ascontiguousarray(rgb_arr.astype(np.uint8, copy=False)).tobytes()
        header = {
            "schema": "lingbot.binary_point_cloud.v1",
            "frameId": self.frame_id,
            "stampMs": int(stamp_ms if stamp_ms is not None else time.time() * 1000),
            "sourcePointCount": source_count,
            "renderedPointCount": int(xyz_arr.shape[0]),
            "xyzBytes": len(xyz_bytes),
            "rgbBytes": len(rgb_bytes),
        }
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        payload = self.MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + xyz_bytes + rgb_bytes
        frame = self._websocket_binary_frame(payload)

        stale: list[socket.socket] = []
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.sendall(frame)
            except OSError:
                stale.append(client)
        if stale:
            with self._lock:
                for client in stale:
                    if client in self._clients:
                        self._clients.remove(client)
                    try:
                        client.close()
                    except OSError:
                        pass
                self.connected_clients = len(self._clients)
        self.published_messages += 1

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        with self._lock:
            clients = list(self._clients)
            self._clients.clear()
            self.connected_clients = 0
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        self._thread.join(timeout=1.0)

    def status(self) -> dict[str, Any]:
        with self._lock:
            clients = len(self._clients)
        return {
            "host": self.host,
            "port": self.port,
            "connected_clients": clients,
            "published_messages": int(self.published_messages),
            "last_error": self.last_error,
        }

    def _serve(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server = server
        try:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(8)
            server.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    client, _addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                server.close()
            except OSError:
                pass

    def _handle_client(self, client: socket.socket) -> None:
        try:
            client.settimeout(2.0)
            request = b""
            while b"\r\n\r\n" not in request and len(request) < 8192:
                chunk = client.recv(1024)
                if not chunk:
                    client.close()
                    return
                request += chunk
            headers = self._parse_http_headers(request.decode("latin1", errors="ignore"))
            key = headers.get("sec-websocket-key")
            if not key:
                client.close()
                return
            accept = base64.b64encode(hashlib.sha1((key + self.GUID).encode("ascii")).digest()).decode("ascii")
            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n"
                "\r\n"
            )
            client.sendall(response.encode("ascii"))
            client.settimeout(None)
            with self._lock:
                self._clients.append(client)
                self.connected_clients = len(self._clients)
        except OSError:
            try:
                client.close()
            except OSError:
                pass

    @staticmethod
    def _parse_http_headers(raw: str) -> dict[str, str]:
        headers: dict[str, str] = {}
        for line in raw.split("\r\n")[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return headers

    @staticmethod
    def _websocket_binary_frame(payload: bytes) -> bytes:
        length = len(payload)
        if length < 126:
            return struct.pack("!BB", 0x82, length) + payload
        if length <= 0xFFFF:
            return struct.pack("!BBH", 0x82, 126, length) + payload
        return struct.pack("!BBQ", 0x82, 127, length) + payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay RGB videos into the GS Console live WebUI as an offline real-to-sim preview."
    )
    parser.add_argument("--video", default="", help="Input video. Used if --frames-dir is empty.")
    parser.add_argument("--video-dir", default="", help="Directory containing videos for playlist playback.")
    parser.add_argument("--video-glob", default="*.mp4", help="Glob used with --video-dir.")
    parser.add_argument("--frames-dir", default="", help="Existing RGB frame directory.")
    parser.add_argument("--output-dir", default="CVPR/nuc_output/video_real2sim_playback/live")
    parser.add_argument("--real2sim-dir", default="CVPR/nuc_output/video_real2sim_playback/real2sim")
    parser.add_argument("--extract-fps", type=float, default=2.0)
    parser.add_argument("--playback-fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--points-per-frame", type=int, default=8000)
    parser.add_argument("--max-global-points", type=int, default=180000)
    parser.add_argument("--lingbot-predictions-npz", default="", help="Optional LingBot predictions npz with world_points/depth.")
    parser.add_argument("--lingbot-summary-json", default="", help="Optional LingBot summary json for image paths.")
    parser.add_argument("--lingbot-conf-percentile", type=float, default=45.0)
    parser.add_argument("--normalize-lingbot-world", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary-cloud-ws-port", type=int, default=19093)
    parser.add_argument("--global-binary-cloud-ws-port", type=int, default=19094)
    parser.add_argument("--binary-cloud-ws-host", default="0.0.0.0")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-ply-every", type=int, default=8)
    parser.add_argument(
        "--synthetic-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a deterministic RGB-derived depth scaffold until LingBot/Mem3R predictions are available.",
    )
    return parser.parse_args()


class LingBotPointTimeline:
    def __init__(
        self,
        *,
        predictions_npz: Path,
        summary_json: Path | None,
        points_per_frame: int,
        conf_percentile: float,
        normalize_world: bool,
    ) -> None:
        self.predictions_npz = predictions_npz
        self.summary_json = summary_json
        self.points_per_frame = max(512, int(points_per_frame))
        self.conf_percentile = float(conf_percentile)
        self.normalize_world = bool(normalize_world)
        self.predictions = np.load(predictions_npz)
        self.summary = json.loads(summary_json.read_text(encoding="utf-8")) if summary_json and summary_json.exists() else {}
        self.image_paths = _resolve_lingbot_image_paths(self.summary, predictions_npz.parent)
        self.frame_count = self._infer_frame_count()
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0
        if self.normalize_world:
            self.center, self.scale = self._estimate_normalization()

    def _infer_frame_count(self) -> int:
        for key in ("world_points", "depth", "pose_enc"):
            if key in self.predictions and self.predictions[key].ndim >= 1:
                return int(self.predictions[key].shape[0])
        return len(self.image_paths)

    def _estimate_normalization(self) -> tuple[np.ndarray, float]:
        samples: list[np.ndarray] = []
        frame_indices = np.linspace(0, max(0, self.frame_count - 1), min(self.frame_count, 8)).astype(np.int64)
        for idx in frame_indices:
            try:
                xyz, _rgb = self.frame_cloud(int(idx))
            except Exception:
                continue
            if xyz.shape[0] > 0:
                samples.append(xyz[:: max(1, xyz.shape[0] // 2000)])
        if not samples:
            return np.zeros(3, dtype=np.float32), 1.0
        merged = np.concatenate(samples, axis=0)
        finite = np.isfinite(merged).all(axis=1)
        merged = merged[finite]
        if merged.shape[0] == 0:
            return np.zeros(3, dtype=np.float32), 1.0
        center = np.median(merged, axis=0).astype(np.float32)
        radius = np.percentile(np.linalg.norm(merged - center[None, :], axis=1), 90)
        scale = 2.8 / max(float(radius), 1e-4)
        return center, float(scale)

    def frame_path(self, frame_index: int) -> Path | None:
        if not self.image_paths:
            return None
        return Path(self.image_paths[min(max(frame_index, 0), len(self.image_paths) - 1)])

    def frame_cloud(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        frame_index = min(max(0, int(frame_index)), max(0, self.frame_count - 1))
        if "world_points" in self.predictions:
            points = np.asarray(self.predictions["world_points"][frame_index], dtype=np.float32)
            conf = np.asarray(
                self.predictions["world_points_conf"][frame_index]
                if "world_points_conf" in self.predictions
                else self.predictions["depth_conf"][frame_index]
                if "depth_conf" in self.predictions
                else np.ones(points.shape[:2], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            points, conf = self._points_from_depth(frame_index)
        rgb = self._rgb_for_frame(frame_index, points.shape[:2])
        xyz, colors = _sample_point_map(points, conf, rgb, self.points_per_frame, self.conf_percentile)
        if self.normalize_world:
            xyz = (xyz - self.center[None, :]) * self.scale
        return xyz.astype(np.float32), colors.astype(np.uint8)

    def _points_from_depth(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        if "depth" not in self.predictions or "intrinsic" not in self.predictions or "extrinsic" not in self.predictions:
            raise RuntimeError("LingBot predictions need world_points or depth+intrinsic+extrinsic.")
        depth = np.asarray(self.predictions["depth"][frame_index], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        h, w = depth.shape[:2]
        conf = np.asarray(
            self.predictions["depth_conf"][frame_index]
            if "depth_conf" in self.predictions
            else np.ones((h, w), dtype=np.float32),
            dtype=np.float32,
        )
        K = np.asarray(self.predictions["intrinsic"][frame_index], dtype=np.float32)
        E = np.asarray(self.predictions["extrinsic"][frame_index], dtype=np.float32)
        if E.shape == (3, 4):
            T = np.eye(4, dtype=np.float32)
            T[:3, :4] = E
        elif E.shape == (4, 4):
            T = E.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported extrinsic shape: {E.shape}")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        fx = float(K[0, 0]) if K.shape[0] >= 3 else float(w)
        fy = float(K[1, 1]) if K.shape[0] >= 3 else float(w)
        cx = float(K[0, 2]) if K.shape[0] >= 3 else float(w) * 0.5
        cy = float(K[1, 2]) if K.shape[0] >= 3 else float(h) * 0.5
        x = (xx - cx) / max(abs(fx), 1e-6) * depth
        y = (yy - cy) / max(abs(fy), 1e-6) * depth
        cam = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
        world = cam @ T.T
        return world[..., :3].astype(np.float32), conf

    def _rgb_for_frame(self, frame_index: int, shape_hw: tuple[int, int]) -> np.ndarray:
        image_path = self.frame_path(frame_index)
        h, w = shape_hw
        if image_path is None or not image_path.exists():
            return np.full((h, w, 3), 220, dtype=np.uint8)
        img = Image.open(image_path).convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)


def _resolve_lingbot_image_paths(summary: dict[str, Any], base_dir: Path) -> list[str]:
    raw = list(summary.get("image_paths", []))
    metadata = summary.get("metadata") if isinstance(summary.get("metadata"), dict) else {}
    if not raw:
        raw = list(metadata.get("original_image_paths", []))
    out: list[str] = []
    for item in raw:
        path = Path(str(item)).expanduser()
        candidates = [path] if path.is_absolute() else [base_dir / path, base_dir / path.name]
        resolved = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        out.append(str(resolved))
    return out


def _sample_point_map(
    points: np.ndarray,
    conf: np.ndarray,
    rgb: np.ndarray,
    max_points: int,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    conf_flat = np.asarray(conf, dtype=np.float32).reshape(-1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(conf_flat)
    if finite.any():
        threshold = np.percentile(conf_flat[finite], np.clip(float(conf_percentile), 0.0, 99.0))
        finite &= conf_flat >= threshold
    xyz = xyz[finite]
    colors = colors[finite]
    if xyz.shape[0] == 0:
        return xyz, colors
    keep_count = min(int(max_points), xyz.shape[0])
    if xyz.shape[0] > keep_count:
        keep = np.linspace(0, xyz.shape[0] - 1, keep_count).astype(np.int64)
        xyz = xyz[keep]
        colors = colors[keep]
    return xyz, colors


def _extract_frames(video: Path, output_dir: Path, fps: float, max_frames: int) -> list[Path]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("OpenCV is required to extract frames from video. Use --frames-dir instead.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in output_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if existing:
        return existing[:max_frames] if max_frames > 0 else existing

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(source_fps / max(0.1, fps))))
    saved: list[Path] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            out = output_dir / f"{len(saved):06d}.jpg"
            cv2.imwrite(str(out), frame)
            saved.append(out)
            if max_frames > 0 and len(saved) >= max_frames:
                break
        index += 1
    cap.release()
    return saved


def _load_frame_paths(args: argparse.Namespace, output_dir: Path) -> list[Path]:
    if args.frames_dir:
        frames_dir = Path(args.frames_dir).expanduser().resolve()
        paths = sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    else:
        videos: list[Path] = []
        if args.video:
            videos = [Path(args.video).expanduser().resolve()]
        elif args.video_dir:
            videos = sorted(Path(args.video_dir).expanduser().resolve().glob(str(args.video_glob)))
        if not videos:
            raise FileNotFoundError("No input video found. Set --video, --video-dir, or --frames-dir.")
        paths = []
        remaining = int(args.max_frames)
        for video in videos:
            if not video.exists():
                raise FileNotFoundError(video)
            per_video_max = remaining if remaining > 0 else 0
            extracted = _extract_frames(
                video,
                output_dir / "source_frames" / video.stem,
                float(args.extract_fps),
                per_video_max,
            )
            paths.extend(extracted)
            if remaining > 0:
                remaining = max(0, int(args.max_frames) - len(paths))
                if remaining <= 0:
                    break
    if args.max_frames > 0:
        paths = paths[: int(args.max_frames)]
    if not paths:
        raise RuntimeError("No frames found for playback.")
    return paths


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_arr = np.asarray(xyz, dtype=np.float32)
    rgb_arr = np.asarray(rgb, dtype=np.uint8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {int(xyz_arr.shape[0])}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for p, c in zip(xyz_arr, rgb_arr):
            handle.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )
    tmp.replace(path)


def _write_gaussian_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, scale_value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = float(scale_value)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {int(xyz.shape[0])}\n")
        for prop in (
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float scale",
            "property float opacity",
            "property float axis_u_x",
            "property float axis_u_y",
            "property float axis_u_z",
            "property float axis_v_x",
            "property float axis_v_y",
            "property float axis_v_z",
        ):
            handle.write(prop + "\n")
        handle.write("end_header\n")
        for p, c in zip(xyz, rgb):
            handle.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {scale:.6f} 0.720000 "
                f"{scale:.6f} 0.000000 0.000000 0.000000 {scale:.6f} 0.000000\n"
            )
    tmp.replace(path)


def _write_gaussian_seed(real2sim_dir: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    seed_dir = real2sim_dir / "latest" / "gaussian_seed"
    seed_dir.mkdir(parents=True, exist_ok=True)
    if xyz.shape[0] == 0:
        return
    scale = np.full((xyz.shape[0],), 0.055, dtype=np.float32)
    opacity = np.full((xyz.shape[0],), 0.72, dtype=np.float32)
    axis_u = np.tile(np.array([[0.055, 0.0, 0.0]], dtype=np.float32), (xyz.shape[0], 1))
    axis_v = np.tile(np.array([[0.0, 0.055, 0.0]], dtype=np.float32), (xyz.shape[0], 1))
    gsplat_scales = np.tile(np.array([[0.055, 0.055, 0.020]], dtype=np.float32), (xyz.shape[0], 1))
    gsplat_quats = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    gsplat_quats[:, 0] = 1.0
    tmp_npz = seed_dir / "gaussians_seed.npz.tmp"
    np.savez_compressed(
        tmp_npz,
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.uint8),
        scale=scale,
        opacity=opacity,
        axis_u=axis_u,
        axis_v=axis_v,
        gsplat_scales=gsplat_scales,
        gsplat_quats=gsplat_quats,
        confidence=np.ones((xyz.shape[0],), dtype=np.float32),
    )
    final_npz = seed_dir / "gaussians_seed.npz"
    generated_npz = seed_dir / "gaussians_seed.npz.tmp.npz"
    (generated_npz if generated_npz.exists() else tmp_npz).replace(final_npz)
    _write_gaussian_ply(seed_dir / "gaussians_seed.ply", xyz, rgb, 0.055)
    _write_json(
        seed_dir / "manifest.json",
        {
            "schema": "lingbot_gaussian_seed.v1",
            "source": "video_playback_scaffold",
            "point_count": int(xyz.shape[0]),
            "npz": str(final_npz),
            "ply": str(seed_dir / "gaussians_seed.ply"),
        },
    )


def _write_real2sim_manifest(real2sim_dir: Path, point_count: int) -> None:
    latest = real2sim_dir / "latest"
    (latest / "geometry").mkdir(parents=True, exist_ok=True)
    mesh_path = latest / "geometry" / "scene_mesh.ply"
    if not mesh_path.exists():
        mesh_path.write_text(
            "ply\nformat ascii 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face 0\nproperty list uchar int vertex_indices\nend_header\n",
            encoding="utf-8",
        )
    _write_json(
        latest / "manifest.json",
        {
            "schema": "lingbot_real2sim_export.v1",
            "sequence": "video_playback_latest",
            "source": "video_playback_scaffold",
            "geometry": {
                "points_ply": "geometry/scene_points.ply",
                "mesh_ply": "geometry/scene_mesh.ply",
                "point_count": int(point_count),
            },
            "gaussian_seed": {
                "ply": "gaussian_seed/gaussians_seed.ply",
                "npz": "gaussian_seed/gaussians_seed.npz",
                "point_count": int(point_count),
            },
        },
    )
    _write_json(real2sim_dir / "latest_manifest.json", {"latest": "latest", "point_count": int(point_count)})


def _sample_frame_cloud(image_path: Path, frame_index: int, frame_count: int, points_per_frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    long_side = max(width, height)
    scale = min(1.0, 480.0 / max(1, long_side))
    if scale < 1.0:
        img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    target = max(512, int(points_per_frame))
    stride = max(1, int(math.sqrt((h * w) / target)))
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    pix = arr[yy, xx].reshape(-1, 3)
    x_norm = (xx.reshape(-1).astype(np.float32) - (w - 1) * 0.5) / max(1.0, w)
    y_norm = ((h - 1) * 0.5 - yy.reshape(-1).astype(np.float32)) / max(1.0, w)
    luma = (0.2126 * pix[:, 0] + 0.7152 * pix[:, 1] + 0.0722 * pix[:, 2]).astype(np.float32) / 255.0
    radial = np.sqrt(x_norm * x_norm + y_norm * y_norm)
    depth = 2.0 + 0.65 * (1.0 - luma) + 0.35 * radial
    local = np.stack([x_norm * depth * 2.0, y_norm * depth * 2.0, depth], axis=1).astype(np.float32)

    # A deterministic orbit path gives the WebUI a growing world map while the
    # real LingBot/Mem3R adapter is being wired in.
    denom = max(1, frame_count - 1)
    phase = (frame_index / denom - 0.5) * 1.10
    yaw = phase * 0.65
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    trans = np.array([phase * 1.35, 0.0, -frame_index * 0.018], dtype=np.float32)
    world = local @ rot.T + trans
    world[:, 1] += 0.05 * np.sin(frame_index * 0.3)
    return local, world.astype(np.float32), pix.astype(np.uint8)


def _pose_for_frame(frame_index: int, frame_count: int) -> dict[str, Any]:
    denom = max(1, frame_count - 1)
    phase = (frame_index / denom - 0.5) * 1.10
    yaw = phase * 0.65
    return {
        "frameId": "map",
        "stampMs": int(time.time() * 1000),
        "position": {"x": float(phase * 1.35), "y": 0.0, "z": float(-frame_index * 0.018)},
        "orientation": {"x": 0.0, "y": float(math.sin(yaw * 0.5)), "z": 0.0, "w": float(math.cos(yaw * 0.5))},
    }


def _downsample_global(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, rgb
    keep = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
    return xyz[keep], rgb[keep]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    real2sim_dir = Path(args.real2sim_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rgb_preview").mkdir(parents=True, exist_ok=True)
    lingbot_timeline: LingBotPointTimeline | None = None
    if args.lingbot_predictions_npz:
        pred_path = Path(args.lingbot_predictions_npz).expanduser().resolve()
        summary_path = Path(args.lingbot_summary_json).expanduser().resolve() if args.lingbot_summary_json else pred_path.with_name("lingbot_summary.json")
        lingbot_timeline = LingBotPointTimeline(
            predictions_npz=pred_path,
            summary_json=summary_path if summary_path.exists() else None,
            points_per_frame=int(args.points_per_frame),
            conf_percentile=float(args.lingbot_conf_percentile),
            normalize_world=bool(args.normalize_lingbot_world),
        )
        frames = [lingbot_timeline.frame_path(idx) for idx in range(lingbot_timeline.frame_count)]
        frames = [path for path in frames if path is not None and path.exists()]
        if int(args.max_frames) > 0:
            frames = frames[: int(args.max_frames)]
        if not frames:
            frames = _load_frame_paths(args, output_dir)
    else:
        frames = _load_frame_paths(args, output_dir)

    current_ws = BinaryPointCloudWebSocketServer(
        host=args.binary_cloud_ws_host,
        port=args.binary_cloud_ws_port,
        max_points=int(args.points_per_frame),
    )
    global_ws = BinaryPointCloudWebSocketServer(
        host=args.binary_cloud_ws_host,
        port=args.global_binary_cloud_ws_port,
        max_points=int(args.max_global_points),
    )

    trajectory: list[dict[str, Any]] = []
    global_xyz_parts: list[np.ndarray] = []
    global_rgb_parts: list[np.ndarray] = []
    playback_started = time.time()
    frame_delay = 1.0 / max(0.1, float(args.playback_fps))
    frame_index = 0

    _write_json(
        output_dir / "playback_source.json",
        {
            "schema": "video_real2sim_playback_source.v1",
            "video": str(Path(args.video).expanduser().resolve()) if args.video else "",
            "frames_dir": str(Path(args.frames_dir).expanduser().resolve()) if args.frames_dir else str(output_dir / "source_frames"),
            "frame_count": len(frames),
            "synthetic_depth": bool(args.synthetic_depth) and lingbot_timeline is None,
            "lingbot_predictions_npz": str(Path(args.lingbot_predictions_npz).expanduser().resolve()) if args.lingbot_predictions_npz else "",
            "lingbot_summary_json": str(Path(args.lingbot_summary_json).expanduser().resolve()) if args.lingbot_summary_json else "",
        },
    )

    print(f"Video playback sidecar serving {len(frames)} frames")
    print(f"  live root:   {output_dir}")
    print(f"  real2sim:    {real2sim_dir}")
    print(f"  current ws:  ws://{args.binary_cloud_ws_host}:{args.binary_cloud_ws_port}/cloud")
    print(f"  global ws:   ws://{args.binary_cloud_ws_host}:{args.global_binary_cloud_ws_port}/cloud")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            path = frames[frame_index]
            stamp_ms = int(time.time() * 1000)
            _atomic_copy(path, output_dir / "rgb_preview" / "latest.jpg")
            if lingbot_timeline is not None:
                world_xyz, rgb = lingbot_timeline.frame_cloud(frame_index % max(1, lingbot_timeline.frame_count))
                local_xyz = world_xyz
            else:
                local_xyz, world_xyz, rgb = _sample_frame_cloud(path, frame_index, len(frames), int(args.points_per_frame))
            global_xyz_parts.append(world_xyz)
            global_rgb_parts.append(rgb)
            global_xyz, global_rgb = _downsample_global(
                np.concatenate(global_xyz_parts, axis=0),
                np.concatenate(global_rgb_parts, axis=0),
                int(args.max_global_points),
            )
            global_xyz_parts = [global_xyz]
            global_rgb_parts = [global_rgb]

            current_ws.publish_cloud(local_xyz, rgb, stamp_ms=stamp_ms)
            global_ws.publish_cloud(global_xyz, global_rgb, stamp_ms=stamp_ms)

            pose = _pose_for_frame(frame_index, len(frames))
            trajectory.append(pose)
            if len(trajectory) > len(frames):
                trajectory = trajectory[-len(frames):]
            _write_json(output_dir / "pose.json", {"pose": pose})
            _write_json(output_dir / "trajectory.json", {"poses": trajectory})
            _write_json(
                output_dir / "metrics.json",
                {
                    "schema": "video_real2sim_playback_metrics.v1",
                    "frame_index": int(frame_index),
                    "frame_count": len(frames),
                    "elapsed_sec": round(time.time() - playback_started, 3),
                    "current_point_count": int(local_xyz.shape[0]),
                    "global_point_count": int(global_xyz.shape[0]),
                    "dense_update_count": int(frame_index + 1),
                    "geometry_age_sec": 0.0,
                    "current_ws": current_ws.status(),
                    "global_ws": global_ws.status(),
                },
            )
            if frame_index % max(1, int(args.write_ply_every)) == 0:
                _write_ascii_ply(output_dir / "live_map.ply", global_xyz, global_rgb)
                _write_ascii_ply(real2sim_dir / "latest" / "geometry" / "scene_points.ply", global_xyz, global_rgb)
                _write_gaussian_seed(real2sim_dir, global_xyz, global_rgb)
                _write_real2sim_manifest(real2sim_dir, int(global_xyz.shape[0]))

            frame_index += 1
            if frame_index >= len(frames):
                if not args.loop:
                    break
                frame_index = 0
                global_xyz_parts.clear()
                global_rgb_parts.clear()
                trajectory.clear()
            time.sleep(frame_delay)
    except KeyboardInterrupt:
        pass
    finally:
        current_ws.close()
        global_ws.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
