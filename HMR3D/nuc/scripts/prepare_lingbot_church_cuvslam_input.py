#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the small calib.txt/times.txt shim needed to run cuVSLAM Mono "
            "on an arbitrary RGB image folder such as LingBot's church demo."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("third_party_research/lingbot-map/example/church"),
        help="RGB image directory used by cuVSLAM/LingBot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nuc_output/lingbot_church_cuvslam_input"),
        help="Directory where calib.txt and times.txt will be written.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Synthetic timestamp FPS.")
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=0.95,
        help="Used as focal=max(width,height)*scale when --fx/--fy are omitted.",
    )
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=-1.0)
    parser.add_argument("--cy", type=float, default=-1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    images = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if args.max_frames > 0:
        images = images[: args.max_frames]
    if not images:
        raise SystemExit(f"No images found under {image_dir}")

    with Image.open(images[0]) as image:
        width, height = image.size

    focal = float(args.focal_scale) * float(max(width, height))
    fx = float(args.fx) if args.fx > 0 else focal
    fy = float(args.fy) if args.fy > 0 else focal
    cx = float(args.cx) if args.cx >= 0 else (float(width) - 1.0) * 0.5
    cy = float(args.cy) if args.cy >= 0 else (float(height) - 1.0) * 0.5

    output_dir.mkdir(parents=True, exist_ok=True)
    row = (
        f"{fx:.6f} 0.000000 {cx:.6f} 0.000000 "
        f"0.000000 {fy:.6f} {cy:.6f} 0.000000 "
        "0.000000 0.000000 1.000000 0.000000"
    )
    (output_dir / "calib.txt").write_text(
        "\n".join(f"P{idx}: {row}" for idx in range(4)) + "\n",
        encoding="utf-8",
    )
    fps = max(float(args.fps), 1e-6)
    (output_dir / "times.txt").write_text(
        "".join(f"{idx / fps:.6f}\n" for idx in range(len(images))),
        encoding="utf-8",
    )
    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Synthetic cuVSLAM Mono shim.",
                f"image_dir: {image_dir}",
                f"images: {len(images)}",
                f"size: {width}x{height}",
                f"fx/fy/cx/cy: {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}",
                f"fps: {fps:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote cuVSLAM Mono shim: {output_dir}")
    print(f"Images: {len(images)} size={width}x{height} fx={fx:.3f} fy={fy:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
