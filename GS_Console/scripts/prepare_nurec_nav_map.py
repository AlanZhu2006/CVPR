#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a conservative ROS PGM map from a NuRec occupancy map.")
    parser.add_argument("--source-yaml", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--target-resolution", type=float, default=0.05)
    parser.add_argument("--free-pixel-threshold", type=float, default=250.0)
    parser.add_argument("--inflate-radius", type=float, default=0.28)
    return parser.parse_args()


def read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def parse_origin(value: str) -> list[float]:
    return [float(v.strip()) for v in value.strip("[]").split(",")[:3]]


def load_source_map(source_yaml: Path) -> tuple[np.ndarray, float, list[float], Path]:
    info = read_simple_yaml(source_yaml)
    image_path = (source_yaml.parent / info["image"]).resolve()
    image = Image.open(image_path).convert("L")
    pixels = np.asarray(image, dtype=np.uint8)
    return pixels, float(info["resolution"]), parse_origin(info["origin"]), image_path


def downsample_free_mask(source_free: np.ndarray, source_resolution: float, target_resolution: float) -> np.ndarray:
    if target_resolution <= source_resolution * 1.001:
        return source_free.copy()

    height, width = source_free.shape
    target_width = int(math.ceil(width * source_resolution / target_resolution))
    target_height = int(math.ceil(height * source_resolution / target_resolution))
    target_free = np.zeros((target_height, target_width), dtype=bool)

    scale = target_resolution / source_resolution
    for row in range(target_height):
        row0 = int(math.floor(row * scale))
        row1 = int(math.ceil((row + 1) * scale))
        row0 = min(max(row0, 0), height - 1)
        row1 = min(max(row1, row0 + 1), height)
        for col in range(target_width):
            col0 = int(math.floor(col * scale))
            col1 = int(math.ceil((col + 1) * scale))
            col0 = min(max(col0, 0), width - 1)
            col1 = min(max(col1, col0 + 1), width)
            target_free[row, col] = bool(np.all(source_free[row0:row1, col0:col1]))
    return target_free


def inflate_obstacles(free: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return free
    obstacles = ~free
    inflated = obstacles.copy()
    rows, cols = np.nonzero(obstacles)
    height, width = obstacles.shape
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= radius_cells * radius_cells:
                offsets.append((dy, dx))
    for dy, dx in offsets:
        rr = np.clip(rows + dy, 0, height - 1)
        cc = np.clip(cols + dx, 0, width - 1)
        inflated[rr, cc] = True
    return ~inflated


def write_pgm(path: Path, free: np.ndarray) -> None:
    image = np.where(free, 254, 0).astype(np.uint8)
    with path.open("wb") as handle:
        handle.write(f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
        handle.write(image.tobytes(order="C"))


def write_yaml(path: Path, image_name: str, resolution: float, origin: list[float]) -> None:
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


def main() -> None:
    args = parse_args()
    source_yaml = Path(args.source_yaml).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pixels, source_resolution, origin, source_image = load_source_map(source_yaml)
    target_resolution = max(float(args.target_resolution), source_resolution)

    # NuRec maps use white as known free, black as occupied, gray as unknown.
    # For navigation, unknown must not become traversable free space.
    source_free = pixels >= float(args.free_pixel_threshold)
    free = downsample_free_mask(source_free, source_resolution, target_resolution)
    free = inflate_obstacles(free, int(math.ceil(float(args.inflate_radius) / target_resolution)))

    pgm_path = output_prefix.with_suffix(".pgm")
    yaml_path = output_prefix.with_suffix(".yaml")
    manifest_path = output_prefix.with_suffix(".manifest.json")
    write_pgm(pgm_path, free)
    write_yaml(yaml_path, pgm_path.name, target_resolution, origin)

    manifest: dict[str, Any] = {
        "source_yaml": str(source_yaml),
        "source_image": str(source_image),
        "pgm": str(pgm_path),
        "yaml": str(yaml_path),
        "source_resolution": source_resolution,
        "resolution": target_resolution,
        "origin": origin,
        "width": int(free.shape[1]),
        "height": int(free.shape[0]),
        "free_cell_count": int(free.sum()),
        "occupied_cell_count": int((~free).sum()),
        "free_pixel_threshold": float(args.free_pixel_threshold),
        "inflate_radius": float(args.inflate_radius),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
