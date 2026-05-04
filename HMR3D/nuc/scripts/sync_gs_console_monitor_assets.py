#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np


def _latest_rgb(rgb_dir: Path) -> Path | None:
    try:
        return max(rgb_dir.glob("*.png"), key=lambda p: p.name)
    except ValueError:
        return None


def _load_worker_trajectory(worker_dir: Path, limit_windows: int) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for result_path in sorted(worker_dir.glob("window_*/worker_result.json"))[-limit_windows:]:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
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
            idx = int(frame.get("frame_idx", len(records)))
            records[idx] = {
                "frame_idx": idx,
                "timestamp_sec": float(frame.get("timestamp_sec", 0.0)),
                "position": pose[:3, 3].astype(float).tolist(),
                "track_ok": bool(meta.get("track_ok", True)),
                "image": Path(str(frame.get("image_path", ""))).name,
            }
    return [records[k] for k in sorted(records)]


def _compact_points(live_dir: Path, max_points: int) -> dict[str, Any]:
    live_json_path = live_dir / "live_map.json"
    updated_at = ""
    active_frames: list[int] = []
    if live_json_path.exists():
        try:
            live_json = json.loads(live_json_path.read_text(encoding="utf-8"))
            updated_at = str(live_json.get("updated_at", ""))
            active_frames = [int(x) for x in live_json.get("active_frames", [])]
        except Exception:
            pass
    npz = live_dir / "live_map.npz"
    if not npz.exists():
        return {
            "points": [],
            "raw_point_count": 0,
            "shown_point_count": 0,
            "bbox_min": [-1, -1, -1],
            "bbox_max": [1, 1, 1],
            "updated_at": updated_at,
            "active_frames": active_frames,
        }
    data = np.load(npz)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    raw_count = int(xyz.shape[0])
    if raw_count > max_points:
        idx = np.linspace(0, raw_count - 1, max_points).astype(np.int64)
        xyz = xyz[idx]
        rgb = rgb[idx]
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    if xyz.shape[0]:
        bbox_min = xyz.min(axis=0).astype(float).tolist()
        bbox_max = xyz.max(axis=0).astype(float).tolist()
    else:
        bbox_min = [-1.0, -1.0, -1.0]
        bbox_max = [1.0, 1.0, 1.0]
    points = np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1).round(4).tolist()
    return {
        "points": points,
        "raw_point_count": raw_count,
        "shown_point_count": int(xyz.shape[0]),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "updated_at": updated_at,
        "active_frames": active_frames,
    }


def sync_once(args: argparse.Namespace) -> None:
    live_dir = Path(args.live_dir).expanduser().resolve()
    worker_dir = Path(args.worker_dir).expanduser().resolve() if args.worker_dir else live_dir / "worker"
    out_dir = Path(args.output_dir).expanduser().resolve()
    monitor_dir = out_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    latest = _latest_rgb(live_dir / "rgb_stream")
    if latest is not None:
        tmp = monitor_dir / f"latest_rgb.png.{os.getpid()}.{time.monotonic_ns()}.tmp"
        shutil.copyfile(latest, tmp)
        tmp.replace(monitor_dir / "latest_rgb.png")

    payload = _compact_points(live_dir, args.max_points)
    payload["trajectory"] = _load_worker_trajectory(worker_dir, args.worker_windows)
    payload["latest_rgb"] = latest.name if latest is not None else ""
    payload["synced_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp_json = monitor_dir / f"live_monitor.json.{os.getpid()}.{time.monotonic_ns()}.tmp"
    tmp_json.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    tmp_json.replace(monitor_dir / "live_monitor.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync live RGB/map assets into the GS Console static directory.")
    parser.add_argument("--live-dir", default="nuc_output/hikrobot_lingbot_ros2_current_cloud_live")
    parser.add_argument("--worker-dir", default="")
    parser.add_argument("--output-dir", default="nuc_output/real2sim_hikrobot_lingbot_live_baseline")
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--max-points", type=int, default=28000)
    parser.add_argument("--worker-windows", type=int, default=900)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    while True:
        sync_once(args)
        if args.once:
            break
        time.sleep(max(0.2, float(args.interval_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
