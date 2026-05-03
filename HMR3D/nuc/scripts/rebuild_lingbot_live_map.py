#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import VoxelFusionMap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild live_map.* from saved LingBot worker depth windows.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, default=Path("nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input"))
    parser.add_argument("--intrinsic-camera-index", type=int, default=2)
    parser.add_argument("--color-image-dir", default="datasets/kitti_raw/2011_09_30/2011_09_30_drive_0020_sync/image_02/data")
    parser.add_argument("--color-image-template", default="{frame_idx:010d}.png")
    parser.add_argument("--tracking-backend", default="rebuild")
    parser.add_argument(
        "--lingbot-pose-mode",
        choices=("chain_relative", "window_local"),
        default="chain_relative",
        help=(
            "chain_relative stitches LingBot sliding-window poses through overlapping frames; "
            "window_local preserves the old behavior and treats every window pose as global."
        ),
    )
    parser.add_argument(
        "--lingbot-pose-translation-scale",
        type=float,
        default=0.0,
        help="Scale LingBot-predicted camera translations before fusing depth. 0 follows --depth-scale.",
    )
    parser.add_argument(
        "--lingbot-extrinsic-mode",
        choices=("inverse", "direct"),
        default="inverse",
        help=(
            "How to interpret LingBot predictions.extrinsic. inverse matches the official "
            "depth unprojection path where extrinsic is world-to-camera."
        ),
    )
    parser.add_argument("--rgb-image-dir", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=20.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument("--max-active-frames", type=int, default=200)
    parser.add_argument("--fusion-mode", choices=("raw", "voxel"), default="voxel")
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--fusion-max-points", type=int, default=500000)
    parser.add_argument("--fusion-min-observations", type=int, default=1)
    parser.add_argument("--adaptive-sampling", action="store_true")
    parser.add_argument("--near-depth-m", type=float, default=18.0)
    parser.add_argument("--near-sample-stride", type=int, default=1)
    parser.add_argument("--edge-sample-stride", type=int, default=2)
    parser.add_argument("--edge-percentile", type=float, default=88.0)
    parser.add_argument("--semantic-sample-stride", type=int, default=1)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--semantic-color-output", action="store_true")
    return parser.parse_args()


def _load_live_module():
    script = REPO_ROOT / "nuc" / "scripts" / "run_cuvslam_lingbot_live_reconstruction.py"
    spec = importlib.util.spec_from_file_location("live_reconstruction", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trajectory_item(frame_idx: int, timestamp_sec: float, pose: np.ndarray, source: str) -> dict[str, Any]:
    return {
        "frame_idx": int(frame_idx),
        "timestamp_sec": float(timestamp_sec),
        "position": pose[:3, 3].astype(float).tolist(),
        "pose": pose.astype(float).tolist(),
        "is_keyframe": True,
        "track_ok": True,
        "source": source,
    }


def _build_chained_lingbot_trajectory(
    result_paths: list[Path],
    *,
    live_module: Any,
    translation_scale: float,
    extrinsic_mode: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    trajectory_by_frame: dict[int, dict[str, Any]] = {}
    disconnected_windows = 0
    pose_windows = 0
    initialized = False
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary_path = Path(result["summary_json"])
        pred_path = Path(result["predictions_npz"])
        if not summary_path.exists() or not pred_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        with np.load(pred_path) as pred:
            if "extrinsic" not in pred:
                continue
            extrinsics = np.asarray(pred["extrinsic"], dtype=np.float32)
        frames = summary.get("metadata", {}).get("frames", [])
        local_poses: list[tuple[int, float, np.ndarray]] = []
        for local_idx, frame in enumerate(frames[: extrinsics.shape[0]]):
            pose = live_module._lingbot_extrinsic_to_pose(
                extrinsics[local_idx],
                translation_scale=translation_scale,
                mode=extrinsic_mode,
            )
            if pose is None:
                continue
            frame_idx = int(frame.get("frame_idx", local_idx))
            timestamp_sec = float(frame.get("timestamp_sec", frame_idx))
            local_poses.append((frame_idx, timestamp_sec, pose.astype(np.float32)))
        if not local_poses:
            continue

        anchor_idx = None
        for idx, (frame_idx, _, _) in enumerate(local_poses):
            if frame_idx in trajectory_by_frame:
                anchor_idx = idx
                break
        if anchor_idx is None:
            if initialized:
                disconnected_windows += 1
            anchor_idx = 0
            frame_idx, timestamp_sec, _ = local_poses[anchor_idx]
            trajectory_by_frame.setdefault(
                frame_idx,
                _trajectory_item(frame_idx, timestamp_sec, np.eye(4, dtype=np.float32), "lingbot_chain_seed"),
            )
            initialized = True

        anchor_frame_idx, _, anchor_local_pose = local_poses[anchor_idx]
        anchor_global_pose = np.asarray(trajectory_by_frame[anchor_frame_idx]["pose"], dtype=np.float32)
        local_to_global = anchor_global_pose @ np.linalg.inv(anchor_local_pose)
        for frame_idx, timestamp_sec, local_pose in local_poses:
            if frame_idx in trajectory_by_frame:
                continue
            global_pose = (local_to_global @ local_pose).astype(np.float32)
            trajectory_by_frame[frame_idx] = _trajectory_item(
                frame_idx,
                timestamp_sec,
                global_pose,
                "lingbot_chain_relative",
            )
        pose_windows += 1

    trajectory = [trajectory_by_frame[idx] for idx in sorted(trajectory_by_frame)]
    stats = {
        "pose_windows": int(pose_windows),
        "trajectory_frames": int(len(trajectory)),
        "disconnected_windows": int(disconnected_windows),
        "translation_scale": float(translation_scale),
        "extrinsic_mode": str(extrinsic_mode),
    }
    return trajectory, trajectory_by_frame, stats


def main() -> int:
    args = parse_args()
    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    live = _load_live_module()

    sequence_dir = args.sequence_dir.expanduser().resolve()
    if (sequence_dir / "calib.txt").exists():
        K = live._load_kitti_intrinsic(sequence_dir, args.intrinsic_camera_index)
    else:
        K = np.eye(3, dtype=np.float32)
    trajectory = []
    trajectory_by_frame: dict[int, dict[str, Any]] = {}
    old_json = source_run / "live_map.json"
    if old_json.exists():
        trajectory = json.loads(old_json.read_text(encoding="utf-8")).get("trajectory", [])
        trajectory_by_frame = {
            int(item["frame_idx"]): item
            for item in trajectory
            if isinstance(item, dict) and "frame_idx" in item
        }

    semantic_projector = live.YOLOSemanticProjector(args.yolo_model, conf=args.yolo_conf, imgsz=args.yolo_imgsz)
    if args.yolo_model and not semantic_projector.enabled:
        print(semantic_projector.error or "YOLO disabled", flush=True)
    fusion_map = (
        VoxelFusionMap(
            voxel_size=args.voxel_size,
            max_voxels=args.fusion_max_points,
            min_observations=args.fusion_min_observations,
        )
        if args.fusion_mode == "voxel"
        else None
    )

    live._write_viewer_html(output_dir)
    frame_points: OrderedDict[int, dict] = OrderedDict()
    events = []
    start = time.perf_counter()
    worker_dir = source_run / "worker"
    result_root = worker_dir if worker_dir.exists() else source_run
    result_paths = sorted(result_root.glob("window_*/worker_result.json"))
    lingbot_translation_scale = float(args.lingbot_pose_translation_scale or args.depth_scale)
    args.lingbot_pose_translation_scale = lingbot_translation_scale
    pose_stats: dict[str, Any] = {
        "pose_windows": 0,
        "trajectory_frames": len(trajectory_by_frame),
        "disconnected_windows": 0,
        "translation_scale": lingbot_translation_scale,
        "extrinsic_mode": str(args.lingbot_extrinsic_mode),
    }
    if args.lingbot_pose_mode == "chain_relative":
        trajectory, trajectory_by_frame, pose_stats = _build_chained_lingbot_trajectory(
            result_paths,
            live_module=live,
            translation_scale=lingbot_translation_scale,
            extrinsic_mode=args.lingbot_extrinsic_mode,
        )
        if args.tracking_backend == "rebuild":
            args.tracking_backend = "lingbot_chain_relative"
    elif args.tracking_backend == "rebuild":
        args.tracking_backend = "lingbot_window_local"

    for result_path in result_paths:
        event = live._process_worker_result(
            result_path,
            frame_points,
            trajectory_by_frame,
            K,
            args,
            semantic_projector=semantic_projector,
            fusion_map=fusion_map,
        )
        events.append(event)
    live._refresh_fusion_snapshot(fusion_map, frame_points, len(events))
    payload = live._write_live_json(output_dir, frame_points, trajectory, events, args)
    summary = {
        "source_run": str(source_run),
        "output_dir": str(output_dir),
        "elapsed_sec": time.perf_counter() - start,
        "processed_windows": len(events),
        "point_count": int(payload["point_count"]),
        "shown_point_count": int(payload["shown_point_count"]),
        "fusion_mode": str(args.fusion_mode),
        "voxel_size": float(args.voxel_size),
        "sample_stride": int(args.sample_stride),
        "lingbot_pose_mode": str(args.lingbot_pose_mode),
        "lingbot_pose_stats": pose_stats,
        "adaptive_sampling": bool(args.adaptive_sampling),
        "yolo_model": str(args.yolo_model),
        "yolo_enabled": bool(semantic_projector.enabled),
    }
    (output_dir / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
