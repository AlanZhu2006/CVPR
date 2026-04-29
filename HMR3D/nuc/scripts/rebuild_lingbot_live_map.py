#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path


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


def main() -> int:
    args = parse_args()
    source_run = args.source_run.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    live = _load_live_module()

    K = live._load_kitti_intrinsic(args.sequence_dir.expanduser().resolve(), args.intrinsic_camera_index)
    trajectory = []
    old_json = source_run / "live_map.json"
    if old_json.exists():
        trajectory = json.loads(old_json.read_text(encoding="utf-8")).get("trajectory", [])

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
    for result_path in sorted((source_run / "worker").glob("window_*/worker_result.json")):
        event = live._process_worker_result(
            result_path,
            frame_points,
            {},
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
        "adaptive_sampling": bool(args.adaptive_sampling),
        "yolo_model": str(args.yolo_model),
        "yolo_enabled": bool(semantic_projector.enabled),
    }
    (output_dir / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
