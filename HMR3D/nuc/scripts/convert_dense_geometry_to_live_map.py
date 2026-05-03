#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert export_lingbot_dense_geometry.py output into live_map.* viewer assets."
    )
    parser.add_argument("--dense-geometry-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-json-points", type=int, default=60000)
    parser.add_argument("--max-ply-points", type=int, default=0, help="0 writes all points to PLY.")
    return parser.parse_args()


def _select_even(count: int, max_count: int) -> np.ndarray:
    if max_count <= 0 or count <= max_count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_count).astype(np.int64)


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main() -> None:
    args = parse_args()
    dense_path = Path(args.dense_geometry_npz).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dense_path)
    xyz = np.asarray(data["points_world"], dtype=np.float32)
    rgb = np.asarray(data["point_rgb"], dtype=np.uint8)
    frame = np.asarray(
        data["point_frame_idx"] if "point_frame_idx" in data else data.get("point_frame_local", np.zeros((xyz.shape[0],), dtype=np.int32)),
        dtype=np.int32,
    )
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    frame = frame[finite]
    semantic_label = np.full((xyz.shape[0],), -1, dtype=np.int32)
    semantic_conf = np.zeros((xyz.shape[0],), dtype=np.float32)
    observations = np.ones((xyz.shape[0],), dtype=np.int32)

    np.savez_compressed(
        output_dir / "live_map.npz",
        xyz=xyz,
        rgb=rgb,
        frame=frame,
        semantic_label=semantic_label,
        semantic_conf=semantic_conf,
        observations=observations,
    )

    json_idx = _select_even(xyz.shape[0], int(args.max_json_points))
    points = [
        [
            round(float(p[0]), 4),
            round(float(p[1]), 4),
            round(float(p[2]), 4),
            int(c[0]),
            int(c[1]),
            int(c[2]),
            int(f),
            -1,
            0.0,
        ]
        for p, c, f in zip(xyz[json_idx], rgb[json_idx], frame[json_idx])
    ]
    bbox_min = xyz.min(axis=0).astype(float).tolist() if xyz.shape[0] else [0.0, 0.0, 0.0]
    bbox_max = xyz.max(axis=0).astype(float).tolist() if xyz.shape[0] else [1.0, 1.0, 1.0]
    payload = {
        "schema": "lingbot_live_reconstruction.v1",
        "source": "lingbot_dense_geometry",
        "dense_geometry_npz": str(dense_path),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_count": int(xyz.shape[0]),
        "shown_point_count": int(len(points)),
        "active_frames": [int(v) for v in np.unique(frame).tolist()],
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "trajectory": [],
        "events": [{"type": "convert_dense_geometry", "point_count": int(xyz.shape[0])}],
        "settings": {
            "max_json_points": int(args.max_json_points),
            "max_ply_points": int(args.max_ply_points),
        },
        "points": points,
    }
    (output_dir / "live_map.json").write_text(json.dumps(payload), encoding="utf-8")

    ply_idx = _select_even(xyz.shape[0], int(args.max_ply_points))
    _write_ascii_ply(output_dir / "live_map.ply", xyz[ply_idx], rgb[ply_idx])

    summary = {
        "dense_geometry_npz": str(dense_path),
        "output_dir": str(output_dir),
        "point_count": int(xyz.shape[0]),
        "shown_point_count": int(len(points)),
        "ply_point_count": int(ply_idx.shape[0]),
        "frame_count": int(np.unique(frame).shape[0]) if frame.shape[0] else 0,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }
    (output_dir / "convert_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
