#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0.0}
    ordered = sorted(finite)
    return {
        "count": float(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(ordered[len(ordered) // 2]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p90": float(ordered[min(len(ordered) - 1, int(math.ceil(0.90 * len(ordered))) - 1)]),
        "p95": float(ordered[min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)]),
    }


def _size_mb(path: Path) -> float:
    return float(path.stat().st_size) / (1024.0 * 1024.0) if path.exists() else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _format_sec(value: float) -> str:
    return f"{value:.4f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a LingBot full-stack reconstruction run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary = _read_json(run_dir / "run_summary.json")
    metrics = _read_json(run_dir / "full_stack_metrics.json")
    live_json = _read_json(run_dir / "live_map.json")
    worker_results = [_read_json(path) for path in sorted((run_dir / "worker").glob("window_*/worker_result.json"))]
    worker_summaries = [_read_json(path) for path in sorted((run_dir / "worker").glob("window_*/lingbot_summary.json"))]

    point_count = int(summary.get("point_count") or live_json.get("point_count") or 0)
    frame_ids = []
    xyz_stats: dict[str, Any] = {}
    live_npz = run_dir / "live_map.npz"
    if live_npz.exists():
        npz = np.load(live_npz)
        xyz = np.asarray(npz["xyz"], dtype=np.float32)
        frame = np.asarray(npz["frame"], dtype=np.int32)
        frame_ids = sorted(int(value) for value in np.unique(frame).tolist())
        if xyz.size:
            xyz_stats = {
                "bbox_min": np.nanmin(xyz, axis=0).astype(float).tolist(),
                "bbox_max": np.nanmax(xyz, axis=0).astype(float).tolist(),
                "bbox_p01": np.nanpercentile(xyz, 1, axis=0).astype(float).tolist(),
                "bbox_p99": np.nanpercentile(xyz, 99, axis=0).astype(float).tolist(),
            }

    worker_elapsed = [float(item.get("elapsed_sec", 0.0)) for item in worker_results]
    worker_queue = [float(item.get("queue_wait_sec", 0.0)) for item in worker_results]
    worker_e2e = [float(item.get("end_to_end_sec", item.get("elapsed_sec", 0.0))) for item in worker_results]
    files = {
        "live_map_json_mb": _size_mb(run_dir / "live_map.json"),
        "live_map_npz_mb": _size_mb(run_dir / "live_map.npz"),
        "live_map_ply_mb": _size_mb(run_dir / "live_map.ply"),
        "worker_dir_mb": 0.0,
    }
    worker_dir = run_dir / "worker"
    if worker_dir.exists():
        files["worker_dir_mb"] = sum(path.stat().st_size for path in worker_dir.rglob("*") if path.is_file()) / (1024.0 * 1024.0)

    first_summary = worker_summaries[0] if worker_summaries else {}
    tracking_rows = metrics.get("tracking", []) if isinstance(metrics.get("tracking"), list) else []
    keypoints = [float(row.get("keypoint_count", 0)) for row in tracking_rows]
    matches = [float(row.get("match_count", 0)) for row in tracking_rows]
    inliers = [float(row.get("inlier_count", 0)) for row in tracking_rows]

    elapsed = float(summary.get("elapsed_sec", 0.0))
    submitted = int(summary.get("submitted_frames", 0))
    windows = int(summary.get("processed_windows", len(worker_results)))
    report = {
        "run_dir": str(run_dir),
        "summary": summary,
        "effective_rates": {
            "submitted_frames_per_sec": submitted / elapsed if elapsed > 0 else 0.0,
            "processed_windows_per_sec": windows / elapsed if elapsed > 0 else 0.0,
            "points_per_sec": point_count / elapsed if elapsed > 0 else 0.0,
        },
        "latency": {
            "track_next_sec": metrics.get("latency", {}).get("track_next_sec", {}),
            "submit_sec": metrics.get("latency", {}).get("submit_sec", {}),
            "worker_elapsed_sec_from_results": _stats(worker_elapsed),
            "worker_queue_wait_sec_from_results": _stats(worker_queue),
            "worker_end_to_end_sec_from_results": _stats(worker_e2e),
            "process_result_sec": metrics.get("latency", {}).get("process_result_sec", {}),
            "live_write_sec": metrics.get("latency", {}).get("live_write_sec", {}),
        },
        "tracking_quality": {
            "keypoint_count": _stats(keypoints),
            "match_count": _stats(matches),
            "inlier_count": _stats(inliers),
        },
        "data": {
            "point_count": point_count,
            "frame_count_with_points": len(frame_ids),
            "frame_id_min": min(frame_ids) if frame_ids else None,
            "frame_id_max": max(frame_ids) if frame_ids else None,
            "depth_shape": first_summary.get("depth_shape", []),
            "prediction_keys": first_summary.get("prediction_keys", []),
            "model": {
                "depth_head_backend": first_summary.get("depth_head_backend", ""),
                "model_patch_embed": first_summary.get("model_patch_embed", ""),
                "model_embed_dim": first_summary.get("model_embed_dim", 0),
                "model_depth": first_summary.get("model_depth", 0),
                "model_num_heads": first_summary.get("model_num_heads", 0),
                "model_mlp_ratio": first_summary.get("model_mlp_ratio", 0.0),
            },
            "xyz_stats": xyz_stats,
            "files": files,
        },
    }

    output_json = args.output_json or (run_dir / "full_stack_report.json")
    output_md = args.output_md or (run_dir / "full_stack_report.md")
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    latency = report["latency"]
    rates = report["effective_rates"]
    data = report["data"]
    lines = [
        "# LingBot Full-Stack Run Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Submitted frames: {submitted}",
        f"- Processed windows: {windows}",
        f"- Total elapsed: {_format_sec(elapsed)}",
        f"- Effective frame rate: {rates['submitted_frames_per_sec']:.3f} frames/s",
        f"- Effective window rate: {rates['processed_windows_per_sec']:.3f} windows/s",
        f"- Point count: {point_count}",
        "",
        "## Latency",
        "",
        "| Stage | Mean | Median | P90 | Max | Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in [
        ("cuVSLAM next/frame", "track_next_sec"),
        ("submit to worker", "submit_sec"),
        ("worker queue wait", "worker_queue_wait_sec_from_results"),
        ("worker depth bundle", "worker_elapsed_sec_from_results"),
        ("worker end-to-end", "worker_end_to_end_sec_from_results"),
        ("backproject/process result", "process_result_sec"),
        ("publish live map", "live_write_sec"),
    ]:
        stat = latency.get(key, {})
        lines.append(
            f"| {label} | {stat.get('mean', 0):.4f}s | {stat.get('median', 0):.4f}s | "
            f"{stat.get('p90', 0):.4f}s | {stat.get('max', 0):.4f}s | {int(stat.get('count', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Data",
            "",
            f"- Frames with points: {data['frame_count_with_points']} ({data['frame_id_min']}..{data['frame_id_max']})",
            f"- Depth shape: `{data['depth_shape']}`",
            f"- Prediction keys: `{data['prediction_keys']}`",
            f"- Model: `{data['model']}`",
            f"- live_map.json: {files['live_map_json_mb']:.2f} MB",
            f"- live_map.npz: {files['live_map_npz_mb']:.2f} MB",
            f"- live_map.ply: {files['live_map_ply_mb']:.2f} MB",
            f"- worker artifacts: {files['worker_dir_mb']:.2f} MB",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
