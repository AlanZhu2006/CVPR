#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import GaussianSplatRenderer
from nuc_runtime.config import load_runtime_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_sequence(baseline_dir: Path) -> Path:
    latest_manifest = baseline_dir / "latest_manifest.json"
    if latest_manifest.exists():
        payload = _read_json(latest_manifest)
        manifest = Path(str(payload.get("manifest", "")))
        if manifest.exists():
            return manifest.parent
    return baseline_dir / "latest"


def _default_live_map_path(baseline_dir: Path) -> Path:
    return baseline_dir.parent / "hikrobot_lingbot_ros2_current_cloud_live" / "live_map.npz"


def _sample_rows(count: int, max_points: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points).astype(np.int64)


def _load_live_map_bundle(path: Path, max_points: int, scale: float, opacity: float) -> dict[str, np.ndarray]:
    data = np.load(path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    if xyz.shape[0] > 0:
        rows = _sample_rows(int(xyz.shape[0]), int(max_points))
        xyz = xyz[rows]
        rgb = rgb[rows]
    count = int(xyz.shape[0])
    scale_arr = np.full((count,), float(scale), dtype=np.float32)
    opacity_arr = np.full((count,), float(opacity), dtype=np.float32)
    axis_u = np.tile(np.array([[float(scale), 0.0, 0.0]], dtype=np.float32), (count, 1))
    axis_v = np.tile(np.array([[0.0, float(scale), 0.0]], dtype=np.float32), (count, 1))
    gsplat_scales = np.stack(
        [
            scale_arr,
            scale_arr,
            np.maximum(0.35 * scale_arr, 1e-4),
        ],
        axis=1,
    ).astype(np.float32)
    gsplat_quats = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1))
    return {
        "xyz": xyz,
        "rgb": rgb,
        "scale": scale_arr,
        "opacity": opacity_arr,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "unstable": np.zeros((count,), dtype=np.float32),
        "recentness": np.ones((count,), dtype=np.float32),
        "source": np.full((count,), 3, dtype=np.int8),
        "provenance": np.full((count,), 3, dtype=np.int8),
        "gsplat_scales": gsplat_scales,
        "gsplat_quats": gsplat_quats,
    }


def _load_seed_bundle(path: Path) -> dict[str, np.ndarray]:
    seed = np.load(path)
    return {key: seed[key] for key in seed.files}


def _resolve_gaussian_bundle(args: argparse.Namespace, baseline_dir: Path, seq_dir: Path) -> tuple[dict[str, np.ndarray], Path, str, float]:
    live_map = Path(args.live_map).expanduser().resolve() if args.live_map else _default_live_map_path(baseline_dir)
    seed_npz = seq_dir / "gaussian_seed" / "gaussians_seed.npz"
    source = str(args.source)
    if source == "auto":
        source = "live_map" if live_map.exists() else "baseline_seed"

    if source == "live_map":
        if not live_map.exists():
            raise FileNotFoundError(live_map)
        return (
            _load_live_map_bundle(
                live_map,
                max_points=int(args.live_map_max_points),
                scale=float(args.live_gaussian_scale),
                opacity=float(args.live_gaussian_opacity),
            ),
            live_map,
            "live_map",
            live_map.stat().st_mtime,
        )

    if not seed_npz.exists():
        raise FileNotFoundError(seed_npz)
    return _load_seed_bundle(seed_npz), seed_npz, "baseline_seed", seed_npz.stat().st_mtime


def _latest_frame(worker_dir: Path, limit_windows: int) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for result_path in sorted(worker_dir.glob("window_*/worker_result.json"))[-limit_windows:]:
        try:
            result = _read_json(result_path)
            summary = _read_json(Path(result["summary_json"]))
        except Exception:
            continue
        for frame in summary.get("metadata", {}).get("frames", []):
            meta = frame.get("metadata") or {}
            pose = np.asarray(meta.get("pose"), dtype=np.float32)
            if pose.shape == (3, 4):
                pose4 = np.eye(4, dtype=np.float32)
                pose4[:3, :4] = pose
                pose = pose4
            if pose.shape != (4, 4):
                continue
            item = {
                "frame_idx": int(frame.get("frame_idx", -1)),
                "timestamp_sec": float(frame.get("timestamp_sec", 0.0)),
                "image_path": str(frame.get("image_path", "")),
                "pose": pose.astype(np.float32),
            }
            if best is None or item["frame_idx"] >= int(best["frame_idx"]):
                best = item
    return best


def _write_intrinsics(path: Path, K: np.ndarray) -> Path:
    sidecar = path.with_suffix(path.suffix + ".intrinsics.json")
    sidecar.write_text(json.dumps({"K": K.astype(float).tolist()}), encoding="utf-8")
    return sidecar


def render_once(args: argparse.Namespace, renderer: GaussianSplatRenderer | None = None) -> dict[str, Any]:
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    worker_dir = Path(args.worker_dir).expanduser().resolve()
    out_dir = baseline_dir / "monitor"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = _latest_sequence(baseline_dir)
    frame = _latest_frame(worker_dir, args.worker_windows)
    if frame is None:
        raise RuntimeError("No worker pose frame available")
    bgr = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(frame["image_path"])
    src_h, src_w = bgr.shape[:2]
    width = int(args.width)
    scale = float(width) / max(1.0, float(src_w))
    height = max(1, int(round(src_h * scale)))
    K = np.array(
        [
            [float(args.fx or src_w) * scale, 0.0, float(args.cx or src_w / 2.0) * scale],
            [0.0, float(args.fy or src_w) * scale, float(args.cy or src_h / 2.0) * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if renderer is None:
        config = load_runtime_config(args.config or None)
        config.output.render_backend = args.backend
        config.output.render_view_budget_points = int(args.render_view_budget_points)
        config.output.render_depth_window_m = float(args.render_depth_window_m)
        config.output.render_internal_scale = 1.0
        renderer = GaussianSplatRenderer(config.output)
    bundle, gaussian_source_path, gaussian_source_kind, gaussian_source_mtime = _resolve_gaussian_bundle(args, baseline_dir, seq_dir)
    image_path = out_dir / "latest_gaussian.png"
    sidecar = _write_intrinsics(image_path, K)
    start = time.perf_counter()
    result = renderer.render_bundle_view(
        bundle=bundle,
        pose=np.asarray(frame["pose"], dtype=np.float32),
        image_shape=(height, width),
        image_path=str(sidecar),
    )
    render_ms = (time.perf_counter() - start) * 1000.0
    tmp = out_dir / "latest_gaussian.png.tmp.png"
    cv2.imwrite(str(tmp), cv2.cvtColor(result.image_rgb, cv2.COLOR_RGB2BGR))
    tmp.replace(image_path)
    summary = {
        "schema": "gs_console_monitor_gs_render.v1",
        "backend": args.backend,
        "gaussian_source": gaussian_source_kind,
        "frame_idx": int(frame["frame_idx"]),
        "source_image": str(frame["image_path"]),
        "gaussian_source_path": str(gaussian_source_path),
        "gaussian_source_mtime": round(float(gaussian_source_mtime), 3),
        "seed_npz": str(gaussian_source_path) if gaussian_source_kind == "baseline_seed" else "",
        "width": width,
        "height": height,
        "render_ms": round(render_ms, 3),
        "point_count": int(result.point_count),
        "projected_points": int(result.projected_points),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp_json = out_dir / "latest_gaussian.json.tmp"
    tmp_json.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    tmp_json.replace(out_dir / "latest_gaussian.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a real gsplat frame for the GS Console monitor tile.")
    parser.add_argument("--baseline-dir", default="nuc_output/real2sim_hikrobot_lingbot_live_baseline")
    parser.add_argument("--worker-dir", default="nuc_output/hikrobot_lingbot_ros2_current_cloud_live/worker")
    parser.add_argument("--config", default="")
    parser.add_argument("--backend", choices=("gsplat", "cpu"), default="gsplat")
    parser.add_argument("--source", choices=("auto", "baseline_seed", "live_map"), default="auto")
    parser.add_argument("--live-map", default="")
    parser.add_argument("--live-map-max-points", type=int, default=90000)
    parser.add_argument("--live-gaussian-scale", type=float, default=0.055)
    parser.add_argument("--live-gaussian-opacity", type=float, default=0.72)
    parser.add_argument("--interval-sec", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--worker-windows", type=int, default=900)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--render-view-budget-points", type=int, default=24000)
    parser.add_argument("--render-depth-window-m", type=float, default=55.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    config.output.render_backend = args.backend
    config.output.render_view_budget_points = int(args.render_view_budget_points)
    config.output.render_depth_window_m = float(args.render_depth_window_m)
    config.output.render_internal_scale = 1.0
    renderer = GaussianSplatRenderer(config.output)
    while True:
        try:
            summary = render_once(args, renderer)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}", "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}), flush=True)
        if args.once:
            break
        time.sleep(max(1.0, float(args.interval_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
