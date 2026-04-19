from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RERUN_SITE_PATHS = [
    REPO_ROOT.parent / "cuVSLAM" / ".venv-jetson" / "lib" / "python3.10" / "site-packages",
    REPO_ROOT.parent / "cuVSLAM" / ".venv-jetson" / "lib" / "python3.10" / "site-packages" / "rerun_sdk",
    REPO_ROOT.parent / "cuVSLAM" / ".venv-jetson" / "local" / "lib" / "python3.10" / "dist-packages",
]
for site_path in RERUN_SITE_PATHS:
    if site_path.exists() and str(site_path) not in sys.path:
        sys.path.insert(0, str(site_path))

import rerun as rr
import rerun.blueprint as rrb

from nuc_runtime import CUVSLAMOfflineKITTIAdapter, MemoryRouter
from nuc_runtime.config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay KITTI with live incremental Gaussian visualization in Rerun."
    )
    parser.add_argument("--sequence-path", required=True, help="KITTI sequence directory.")
    parser.add_argument("--trajectory-path", required=True, help="trajectory_tum.txt produced by cuVSLAM.")
    parser.add_argument("--config", default="", help="Optional runtime YAML config.")
    parser.add_argument("--frame-step", type=int, default=0, help="Replay every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit replay length.")
    parser.add_argument(
        "--rerun-mode",
        default=os.environ.get("HMR3D_RERUN_MODE", ""),
        choices=["", "spawn", "web", "save"],
        help="Rerun output mode. Defaults to spawn with DISPLAY or save on headless systems.",
    )
    parser.add_argument(
        "--rerun-file",
        default=os.environ.get("HMR3D_RERUN_FILE", ""),
        help="Output .rrd file path when rerun mode is save.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for gaussian exports. Defaults to config output_dir.",
    )
    return parser.parse_args()


def init_rerun(mode: str, rerun_file: str, output_dir: Path) -> None:
    rr.init("HMR3D Gaussian Replay", strict=True, spawn=False)
    if not mode:
        mode = "spawn" if os.environ.get("DISPLAY") else "save"

    default_rrd = output_dir / "gaussian_live.rrd"
    if mode == "spawn":
        rr.spawn()
    elif mode == "web":
        web_port = int(os.environ.get("HMR3D_RERUN_WEB_PORT", "9091"))
        ws_port = int(os.environ.get("HMR3D_RERUN_WS_PORT", "9878"))
        rr.serve_web(open_browser=False, web_port=web_port, ws_port=ws_port)
    elif mode == "save":
        output_file = Path(rerun_file).expanduser().resolve() if rerun_file else default_rrd
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rr.save(output_file)
    else:
        raise ValueError("Unsupported rerun mode.")

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(
                column_shares=[0.65, 0.35],
                contents=[
                    rrb.Spatial3DView(origin="world"),
                    rrb.Spatial2DView(origin="world/camera/image"),
                ],
            ),
        ),
        make_active=True,
    )
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log(
        "world/axes",
        rr.Arrows3D(
            vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        ),
        static=True,
    )


def pose_quaternion_xyzw(pose: np.ndarray) -> np.ndarray:
    rotation = pose[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    quat /= np.linalg.norm(quat) + 1e-8
    return quat


class GaussianRerunLogger:
    def __init__(self, router: MemoryRouter):
        self.router = router
        self.trajectory: list[np.ndarray] = []
        self.archived_ids: set[int] = set()
        self.archived_xyz: list[np.ndarray] = []
        self.archived_radii: list[np.ndarray] = []

    def log_frame(self, output, events: list) -> None:
        rr.set_time_sequence("frame", output.frame_idx)
        rr.set_time_seconds("time", output.timestamp_sec)

        self._log_camera(output)
        self._log_image(output.image_path)
        self._consume_archive_events(events)
        self._log_archived()
        self._log_active()
        self._log_warmstart()

    def _log_camera(self, output) -> None:
        translation = output.pose[:3, 3]
        quat = pose_quaternion_xyzw(output.pose)
        self.trajectory.append(translation.copy())

        rr.log(
            "world/camera/current",
            rr.Transform3D(translation=translation, quaternion=quat),
            rr.Arrows3D(vectors=np.eye(3) * 0.4, colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]),
        )
        rr.log(
            "world/camera/path",
            rr.LineStrips3D([np.asarray(self.trajectory, dtype=np.float32)], colors=[[255, 255, 255]]),
        )

    def _log_image(self, image_path: str | None) -> None:
        if not image_path:
            return
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return
        rr.log("world/camera/image", rr.Image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))

    def _consume_archive_events(self, events: list) -> None:
        if self.router.gaussian_builder is None:
            return
        for event in events:
            if event.event_type != "archived":
                continue
            archived_id = event.payload.get("archived_id")
            if archived_id in self.archived_ids:
                continue
            archived = next((item for item in self.router.bank if item.submap_id == archived_id), None)
            if archived is None or archived.gaussian_handle is None:
                continue
            points = self.router.gaussian_builder.handle_points(archived.gaussian_handle)
            if points["xyz"].shape[0] == 0:
                continue
            self.archived_ids.add(archived_id)
            self.archived_xyz.append(points["xyz"])
            self.archived_radii.append(np.clip(points["scale"], 0.01, 0.2))

    def _log_archived(self) -> None:
        if not self.archived_xyz:
            rr.log("world/gaussians/archived", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
            return
        xyz = np.vstack(self.archived_xyz).astype(np.float32)
        radii = np.concatenate(self.archived_radii).astype(np.float32)
        colors = np.tile(np.array([[70, 130, 255]], dtype=np.uint8), (xyz.shape[0], 1))
        rr.log("world/gaussians/archived", rr.Points3D(xyz, colors=colors, radii=radii))

    def _log_active(self) -> None:
        if self.router.active is None or self.router.gaussian_builder is None:
            return
        points = self.router.gaussian_builder.active_points(self.router.active.submap_id)
        colors = np.tile(np.array([[80, 255, 140]], dtype=np.uint8), (points["xyz"].shape[0], 1))
        rr.log(
            "world/gaussians/active",
            rr.Points3D(points["xyz"], colors=colors, radii=np.clip(points["scale"], 0.01, 0.2)),
        )

    def _log_warmstart(self) -> None:
        if self.router.active is None or self.router.gaussian_builder is None:
            return
        handles = self.router.active.recovered_gaussian_handles
        if not handles:
            rr.log("world/gaussians/warmstart", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
            return
        bundles = [self.router.gaussian_builder.handle_points(handle) for handle in handles]
        bundles = [bundle for bundle in bundles if bundle["xyz"].shape[0] > 0]
        if not bundles:
            rr.log("world/gaussians/warmstart", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))
            return
        xyz = np.vstack([bundle["xyz"] for bundle in bundles]).astype(np.float32)
        radii = np.concatenate([bundle["scale"] for bundle in bundles]).astype(np.float32)
        colors = np.tile(np.array([[255, 210, 60]], dtype=np.uint8), (xyz.shape[0], 1))
        rr.log("world/gaussians/warmstart", rr.Points3D(xyz, colors=colors, radii=np.clip(radii, 0.01, 0.2)))


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    config.memory.enable_incremental_gaussian = True
    output_dir = Path(args.output_dir or config.output.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_step = args.frame_step or config.input.frame_step
    max_frames = args.max_frames or config.input.max_frames

    init_rerun(args.rerun_mode, args.rerun_file, output_dir)

    adapter = CUVSLAMOfflineKITTIAdapter(
        sequence_path=args.sequence_path,
        trajectory_path=args.trajectory_path,
        config=config.tracking,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    router = MemoryRouter(config.memory, output_dir=output_dir)
    visualizer = GaussianRerunLogger(router)

    for output in adapter:
        events, _snapshot = router.process(output)
        visualizer.log_frame(output, events)

    summary = router.finalize()
    rr.log("world/run_summary", rr.TextDocument(str(summary)))
    print(
        {
            "output_dir": str(output_dir),
            "archives": summary["stats"].get("archives", 0),
            "recoveries": summary["stats"].get("recoveries", 0),
            "gaussian_archives": summary["stats"].get("gaussian_archives", 0),
            "gaussian_warmstart_requests": summary["stats"].get("gaussian_warmstart_requests", 0),
        }
    )


if __name__ == "__main__":
    main()
