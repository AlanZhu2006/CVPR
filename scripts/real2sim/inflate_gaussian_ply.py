#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inflate Gaussian PLY log-scale fields for denser WebUI display.")
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--scale-multiplier", type=float, default=3.0)
    parser.add_argument("--min-linear-scale", type=float, default=0.0)
    parser.add_argument("--opacity-gamma", type=float, default=1.0)
    parser.add_argument("--drop-opacity-below", type=float, default=0.0)
    parser.add_argument("--drop-dark-luma-below", type=float, default=0.0)
    parser.add_argument("--drop-dark-opacity-below", type=float, default=0.0)
    return parser.parse_args()


def rgb_luma(vertex: np.ndarray) -> np.ndarray | None:
    required = ("f_dc_0", "f_dc_1", "f_dc_2")
    if not all(name in vertex.dtype.names for name in required):
        return None
    colors = np.stack([vertex[name] for name in required], axis=1).astype(np.float32)
    colors = np.clip(colors * 0.28209479177387814 + 0.5, 0.0, 1.0)
    return 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]


def main() -> None:
    args = parse_args()
    if args.scale_multiplier <= 0:
        raise ValueError("--scale-multiplier must be positive")
    if args.opacity_gamma <= 0:
        raise ValueError("--opacity-gamma must be positive")

    ply = PlyData.read(args.input_ply)
    vertex = ply["vertex"].data.copy()
    source_count = int(vertex.shape[0])

    keep = np.ones((vertex.shape[0],), dtype=bool)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32) if "opacity" in vertex.dtype.names else None
    if args.drop_opacity_below > 0 and opacity is not None:
        keep &= opacity >= float(args.drop_opacity_below)

    luma = rgb_luma(vertex)
    if args.drop_dark_luma_below > 0 and args.drop_dark_opacity_below > 0 and luma is not None and opacity is not None:
        dark_low_opacity = (luma < float(args.drop_dark_luma_below)) & (opacity < float(args.drop_dark_opacity_below))
        keep &= ~dark_low_opacity

    if not np.all(keep):
        vertex = vertex[keep].copy()
        if vertex.shape[0] == 0:
            raise ValueError("All splats were filtered out.")

    scale_offset = math.log(float(args.scale_multiplier))
    for name in ("scale_0", "scale_1", "scale_2"):
        if name not in vertex.dtype.names:
            continue
        values = np.asarray(vertex[name], dtype=np.float32) + scale_offset
        if args.min_linear_scale > 0:
            values = np.maximum(values, math.log(float(args.min_linear_scale)))
        vertex[name] = values.astype(vertex[name].dtype, copy=False)

    if args.opacity_gamma != 1.0 and "opacity" in vertex.dtype.names:
        opacity = np.asarray(vertex["opacity"], dtype=np.float32)
        opacity = np.clip(opacity, 0.0, 1.0)
        opacity = np.power(opacity, float(args.opacity_gamma))
        vertex["opacity"] = opacity.astype(vertex["opacity"].dtype, copy=False)

    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply.text, byte_order=ply.byte_order).write(args.output_ply)
    print(
        {
            "input": str(args.input_ply),
            "output": str(args.output_ply),
            "scale_multiplier": args.scale_multiplier,
            "min_linear_scale": args.min_linear_scale,
            "opacity_gamma": args.opacity_gamma,
            "drop_opacity_below": args.drop_opacity_below,
            "drop_dark_luma_below": args.drop_dark_luma_below,
            "drop_dark_opacity_below": args.drop_dark_opacity_below,
            "source_vertices": source_count,
            "vertices": int(vertex.shape[0]),
            "dropped_vertices": source_count - int(vertex.shape[0]),
        }
    )


if __name__ == "__main__":
    main()
