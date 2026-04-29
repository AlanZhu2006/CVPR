#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import viser


def _turbo_colormap(value: np.ndarray) -> np.ndarray:
    """Small Turbo-like colormap for visualization-only point colors."""
    x = np.clip(value.astype(np.float32), 0.0, 1.0)
    r = 0.13572138 + 4.61539260 * x - 42.66032258 * x**2 + 132.13108234 * x**3 - 152.94239396 * x**4 + 59.28637943 * x**5
    g = 0.09140261 + 2.19418839 * x + 4.84296658 * x**2 - 14.18503333 * x**3 + 4.27729857 * x**4 + 2.82956604 * x**5
    b = 0.10667330 + 12.64194608 * x - 60.58204836 * x**2 + 110.36276771 * x**3 - 89.90310912 * x**4 + 27.34824973 * x**5
    return (np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0) * 255.0).astype(np.uint8)


SEMANTIC_PALETTE = np.array(
    [
        [220, 20, 60],
        [0, 168, 132],
        [255, 170, 0],
        [86, 156, 214],
        [178, 102, 255],
        [45, 212, 191],
        [250, 204, 21],
        [244, 114, 182],
        [132, 204, 22],
        [251, 146, 60],
        [96, 165, 250],
        [167, 139, 250],
    ],
    dtype=np.uint8,
)


def _load_live_map(map_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz_path = map_dir / "live_map.npz"
    json_path = map_dir / "live_map.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing live_map.npz: {npz_path}")
    data = np.load(npz_path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    frame = np.asarray(data["frame"], dtype=np.int32)
    semantic = np.asarray(data["semantic_label"], dtype=np.int32) if "semantic_label" in data else np.full((xyz.shape[0],), -1, dtype=np.int32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    frame = frame[finite]
    semantic = semantic[finite]

    trajectory = np.zeros((0, 3), dtype=np.float32)
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        poses = [item.get("position", []) for item in payload.get("trajectory", [])]
        poses = [pose for pose in poses if len(pose) == 3 and np.all(np.isfinite(pose))]
        if poses:
            trajectory = np.asarray(poses, dtype=np.float32)
    return xyz, rgb, frame, semantic, trajectory


def _filter_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    frame: np.ndarray,
    semantic: np.ndarray,
    *,
    max_points: int,
    quantile_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if xyz.size == 0:
        return xyz, rgb, frame, semantic
    if 0.0 < quantile_clip < 0.5 and xyz.shape[0] > 32:
        lo = np.quantile(xyz, quantile_clip, axis=0)
        hi = np.quantile(xyz, 1.0 - quantile_clip, axis=0)
        keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        if keep.any():
            xyz = xyz[keep]
            rgb = rgb[keep]
            frame = frame[keep]
            semantic = semantic[keep]
    if max_points > 0 and xyz.shape[0] > max_points:
        indices = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
        xyz = xyz[indices]
        rgb = rgb[indices]
        frame = frame[indices]
        semantic = semantic[indices]
    return (
        xyz.astype(np.float32, copy=False),
        rgb.astype(np.uint8, copy=False),
        frame.astype(np.int32, copy=False),
        semantic.astype(np.int32, copy=False),
    )


def _apply_color_mode(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray, semantic: np.ndarray, mode: str) -> np.ndarray:
    if mode == "original":
        return rgb
    if xyz.shape[0] == 0:
        return rgb
    if mode == "semantic":
        out = rgb.copy()
        valid = semantic >= 0
        if valid.any():
            out[valid] = SEMANTIC_PALETTE[semantic[valid] % len(SEMANTIC_PALETTE)]
        return out
    if mode == "height":
        value = xyz[:, 1]
    elif mode == "depth":
        value = xyz[:, 2]
    elif mode == "frame":
        value = frame.astype(np.float32)
    else:
        return rgb
    lo, hi = np.nanpercentile(value, [1, 99])
    normalized = (value - lo) / max(float(hi - lo), 1e-6)
    return _turbo_colormap(normalized)


def _trajectory_segments(trajectory: np.ndarray) -> np.ndarray:
    if trajectory.shape[0] < 2:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.stack([trajectory[:-1], trajectory[1:]], axis=1).astype(np.float32)


class PlaybackViewer:
    def __init__(
        self,
        server: viser.ViserServer,
        xyz: np.ndarray,
        rgb: np.ndarray,
        frame: np.ndarray,
        trajectory: np.ndarray,
        *,
        point_size: float,
        fps: float,
        frame_nodes: bool,
        initial_mode: str,
    ) -> None:
        self.server = server
        self.xyz = xyz
        self.rgb = rgb
        self.frame = frame
        self.trajectory = trajectory
        self.point_size = point_size
        self.frame_ids = np.unique(frame)
        self.frame_ids.sort()
        if self.frame_ids.size == 0:
            self.frame_ids = np.array([0], dtype=np.int32)
        self.point_handle = None
        self.current_pose_handle = None
        self.frame_point_handles = []
        self._lock = threading.Lock()
        self._last_render_key: tuple[int, str] | None = None
        self.frame_nodes = frame_nodes

        server.gui.add_text("Dataset", f"{xyz.shape[0]} pts / {self.frame_ids.size} frames", disabled=True)
        with server.gui.add_folder("Playback", expand_by_default=True):
            self.gui_frame = server.gui.add_slider(
                "Frame",
                min=0,
                max=int(self.frame_ids.size - 1),
                step=1,
                initial_value=0,
            )
            self.gui_prev = server.gui.add_button("Prev")
            self.gui_next = server.gui.add_button("Next")
            self.gui_playing = server.gui.add_checkbox("Playing", initial_value=False)
            self.gui_fps = server.gui.add_slider("FPS", min=1, max=30, step=1, initial_value=fps)
            self.gui_mode = server.gui.add_dropdown(
                "Mode",
                options=("current", "reveal", "all"),
                initial_value=initial_mode,
            )
            self.gui_point_size = server.gui.add_slider(
                "Point size",
                min=0.005,
                max=0.12,
                step=0.005,
                initial_value=point_size,
            )

        @self.gui_next.on_click
        def _(_) -> None:
            self.gui_frame.value = (int(self.gui_frame.value) + 1) % int(self.frame_ids.size)

        @self.gui_prev.on_click
        def _(_) -> None:
            self.gui_frame.value = (int(self.gui_frame.value) - 1) % int(self.frame_ids.size)

        @self.gui_frame.on_update
        def _(_) -> None:
            self.render()

        @self.gui_mode.on_update
        def _(_) -> None:
            self.render(force=True)

        @self.gui_point_size.on_update
        def _(_) -> None:
            self.point_size = float(self.gui_point_size.value)
            self.render(force=True)

        if self.frame_nodes:
            self._build_frame_nodes()
        self.render(force=True)

    def tick(self) -> None:
        if not bool(self.gui_playing.value):
            return
        self.gui_frame.value = (int(self.gui_frame.value) + 1) % int(self.frame_ids.size)

    def _point_mask(self, frame_index: int, mode: str) -> np.ndarray:
        frame_id = int(self.frame_ids[frame_index])
        if mode == "all":
            return np.ones((self.frame.shape[0],), dtype=bool)
        if mode == "current":
            return self.frame == frame_id
        return self.frame <= frame_id

    def _build_frame_nodes(self) -> None:
        with self.server.atomic():
            parent = self.server.scene.add_frame("/lingbot/frame_points", show_axes=False)
            parent.visible = True
            for index, frame_id in enumerate(self.frame_ids):
                mask = self.frame == int(frame_id)
                handle = self.server.scene.add_point_cloud(
                    f"/lingbot/frame_points/{index:04d}",
                    points=self.xyz[mask],
                    colors=self.rgb[mask],
                    point_size=self.point_size,
                    point_shape="circle",
                    visible=False,
                )
                self.frame_point_handles.append(handle)
        self.server.flush()

    def _render_frame_nodes(self, frame_index: int, mode: str, *, force: bool = False) -> None:
        key = (frame_index, mode)
        if not force and key == self._last_render_key:
            return
        with self._lock:
            with self.server.atomic():
                for index, handle in enumerate(self.frame_point_handles):
                    if mode == "all":
                        handle.visible = True
                    elif mode == "current":
                        handle.visible = index == frame_index
                    else:
                        handle.visible = index <= frame_index
                if self.current_pose_handle is not None:
                    self.current_pose_handle.remove()
                if self.trajectory.shape[0] > 0:
                    pose_index = min(frame_index, self.trajectory.shape[0] - 1)
                    self.current_pose_handle = self.server.scene.add_frame(
                        "/lingbot/current_frame",
                        show_axes=True,
                        axes_length=0.8,
                        axes_radius=0.02,
                        position=self.trajectory[pose_index],
                    )
            self.server.flush()
        self._last_render_key = key

    def render(self, *, force: bool = False) -> None:
        frame_index = int(self.gui_frame.value)
        mode = str(self.gui_mode.value)
        if self.frame_nodes and self.frame_point_handles:
            self._render_frame_nodes(frame_index, mode, force=force)
            return
        key = (frame_index, mode)
        if not force and key == self._last_render_key:
            return
        mask = self._point_mask(frame_index, mode)
        points = self.xyz[mask]
        colors = self.rgb[mask]
        with self._lock:
            with self.server.atomic():
                if self.point_handle is not None:
                    self.point_handle.remove()
                self.point_handle = self.server.scene.add_point_cloud(
                    "/lingbot/playback_points",
                    points=points,
                    colors=colors,
                    point_size=self.point_size,
                    point_shape="circle",
                )
                if self.current_pose_handle is not None:
                    self.current_pose_handle.remove()
                if self.trajectory.shape[0] > 0:
                    pose_index = min(frame_index, self.trajectory.shape[0] - 1)
                    self.current_pose_handle = self.server.scene.add_frame(
                        "/lingbot/current_frame",
                        show_axes=True,
                        axes_length=0.8,
                        axes_radius=0.02,
                        position=self.trajectory[pose_index],
                    )
            self.server.flush()
        self._last_render_key = key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a LingBot-style Viser viewer for a live_map directory.")
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--point-size", type=float, default=0.035)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--quantile-clip", type=float, default=0.002)
    parser.add_argument("--color-mode", choices=("original", "height", "depth", "frame", "semantic"), default="frame")
    parser.add_argument("--single-cloud", action="store_true", help="Use one re-uploaded point cloud instead of per-frame Viser nodes.")
    parser.add_argument("--initial-mode", choices=("current", "reveal", "all"), default="reveal")
    parser.add_argument("--poll-sec", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    map_dir = args.map_dir.expanduser().resolve()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/grid", width=120.0, height=120.0, cell_size=5.0)
    server.scene.add_frame("/world", axes_length=1.0, axes_radius=0.02)
    traj_handle = None

    print(f"LingBot live viser map: {map_dir}", flush=True)
    print(f"Open: http://127.0.0.1:{args.port}", flush=True)
    xyz, rgb, frame, semantic, trajectory = _load_live_map(map_dir)
    xyz, rgb, frame, semantic = _filter_points(
        xyz,
        rgb,
        frame,
        semantic,
        max_points=args.max_points,
        quantile_clip=args.quantile_clip,
    )
    rgb = _apply_color_mode(xyz, rgb, frame, semantic, args.color_mode)
    if trajectory.shape[0] > 1:
        traj_handle = server.scene.add_line_segments(
            "/lingbot/trajectory",
            points=_trajectory_segments(trajectory),
            colors=np.array([255, 178, 64], dtype=np.uint8),
            line_width=3.0,
        )
    for index, position in enumerate(trajectory[:: max(1, len(trajectory) // 20)]):
        server.scene.add_frame(
            f"/lingbot/cameras/{index:04d}",
            show_axes=True,
            axes_length=0.25,
            axes_radius=0.01,
            position=position,
        )
    viewer = PlaybackViewer(
        server,
        xyz,
        rgb,
        frame,
        trajectory,
        point_size=args.point_size,
        fps=args.fps,
        frame_nodes=not args.single_cloud,
        initial_mode=args.initial_mode,
    )
    print(f"Updated viser: points={xyz.shape[0]} frames={viewer.frame_ids.size} trajectory={trajectory.shape[0]}", flush=True)
    while True:
        viewer.tick()
        if args.once:
            break
        time.sleep(max(0.01, 1.0 / max(1.0, float(viewer.gui_fps.value))))
    if args.once:
        server.sleep_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
