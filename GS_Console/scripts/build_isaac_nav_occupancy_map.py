#!/usr/bin/env python3
"""Project a PLY point cloud into a ROS-style 2D occupancy map."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PGM/YAML 2D occupancy map from a PLY point cloud.")
    parser.add_argument("--input", required=True, help="Input PLY point cloud.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix without extension.")
    parser.add_argument("--resolution", type=float, default=0.2, help="Map resolution in meters per pixel.")
    parser.add_argument("--z-min", type=float, default=0.2, help="Minimum z height included as obstacle.")
    parser.add_argument("--z-max", type=float, default=2.0, help="Maximum z height included as obstacle.")
    parser.add_argument("--padding", type=float, default=1.0, help="XY padding around filtered points, in meters.")
    parser.add_argument("--inflate-radius", type=float, default=0.4, help="Obstacle inflation radius in meters.")
    return parser.parse_args()


def write_pgm(path: Path, image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("PGM image must be a 2D uint8 array")
    with path.open("wb") as f:
        f.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        f.write(image.tobytes(order="C"))


def write_yaml(path: Path, image_name: str, resolution: float, origin: tuple[float, float, float]) -> None:
    text = "\n".join(
        [
            f"image: {image_name}",
            "mode: trinary",
            f"resolution: {resolution:.6f}",
            f"origin: [{origin[0]:.6f}, {origin[1]:.6f}, {origin[2]:.6f}]",
            "negate: 0",
            "occupied_thresh: 0.65",
            "free_thresh: 0.196",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def inflate_grid(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return occupied
    inflated = occupied.copy()
    ys, xs = np.nonzero(occupied)
    height, width = occupied.shape
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.append((dy, dx))
    for dy, dx in offsets:
        yy = np.clip(ys + dy, 0, height - 1)
        xx = np.clip(xs + dx, 0, width - 1)
        inflated[yy, xx] = True
    return inflated


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cloud = trimesh.load(input_path, process=False)
    points = np.asarray(cloud.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point cloud, got {points.shape}")

    z_mask = (points[:, 2] >= args.z_min) & (points[:, 2] <= args.z_max)
    filtered = points[z_mask]
    if len(filtered) == 0:
        raise ValueError(f"No points remain after z filter [{args.z_min}, {args.z_max}]")

    min_xy = filtered[:, :2].min(axis=0) - args.padding
    max_xy = filtered[:, :2].max(axis=0) + args.padding
    width = int(math.ceil((float(max_xy[0] - min_xy[0])) / args.resolution)) + 1
    height = int(math.ceil((float(max_xy[1] - min_xy[1])) / args.resolution)) + 1

    cells = np.floor((filtered[:, :2] - min_xy) / args.resolution).astype(np.int64)
    cells[:, 0] = np.clip(cells[:, 0], 0, width - 1)
    cells[:, 1] = np.clip(cells[:, 1], 0, height - 1)

    occupied = np.zeros((height, width), dtype=bool)
    occupied[cells[:, 1], cells[:, 0]] = True

    radius_cells = int(math.ceil(args.inflate_radius / args.resolution))
    inflated = inflate_grid(occupied, radius_cells)

    image = np.full((height, width), 254, dtype=np.uint8)
    image[np.flipud(inflated)] = 0

    pgm_path = output_prefix.with_suffix(".pgm")
    yaml_path = output_prefix.with_suffix(".yaml")
    manifest_path = output_prefix.with_suffix(".manifest.json")

    write_pgm(pgm_path, image)
    write_yaml(yaml_path, pgm_path.name, args.resolution, (float(min_xy[0]), float(min_xy[1]), 0.0))

    manifest = {
        "input": str(input_path),
        "pgm": str(pgm_path),
        "yaml": str(yaml_path),
        "resolution": args.resolution,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "padding": args.padding,
        "inflate_radius": args.inflate_radius,
        "source_point_count": int(points.shape[0]),
        "filtered_point_count": int(filtered.shape[0]),
        "width": width,
        "height": height,
        "origin": [float(min_xy[0]), float(min_xy[1]), 0.0],
        "occupied_cells_raw": int(occupied.sum()),
        "occupied_cells_inflated": int(inflated.sum()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
