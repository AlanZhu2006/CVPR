#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing GS-SDF gt image folders by matching rendered image "
            "filenames back to the original RGB frames."
        )
    )
    parser.add_argument("--output-root", required=True, help="GS-SDF output directory containing gs_log/.")
    parser.add_argument(
        "--source-image-dir",
        required=True,
        help="Directory containing the original RGB frames, e.g. NuRec front_stereo_camera_left.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="GS-SDF split folders under gs_log to repair.",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to materialize gt images.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing gt files.")
    return parser.parse_args()


def extract_timestamp_name(render_name: str) -> str | None:
    match = re.search(r"(\d{12,})(\.[A-Za-z0-9]+)$", render_name)
    if match:
        return match.group(1) + match.group(2).lower()
    return None


def find_source_image(source_dir: Path, render_name: str) -> Path | None:
    timestamp_name = extract_timestamp_name(render_name)
    if timestamp_name:
        candidate = source_dir / timestamp_name
        if candidate.exists():
            return candidate
        stem = Path(timestamp_name).stem
        for suffix in IMAGE_EXTENSIONS:
            candidate = source_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    render_stem = Path(render_name).stem
    for suffix in IMAGE_EXTENSIONS:
        candidate = source_dir / f"{render_stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def materialize(src: Path, dst: Path, mode: str, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def repair_split(output_root: Path, source_dir: Path, split: str, mode: str, force: bool) -> dict[str, int]:
    color_root = output_root / "gs_log" / split / "color"
    renders_dir = color_root / "renders"
    gt_dir = color_root / "gt"
    if not renders_dir.exists():
        return {"renders": 0, "matched": 0, "missing": 0, "existing": 0}

    render_files = sorted(path for path in renders_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    matched = 0
    missing = 0
    existing = 0
    for render_path in render_files:
        dst = gt_dir / render_path.name
        if (dst.exists() or dst.is_symlink()) and not force:
            existing += 1
            continue
        source = find_source_image(source_dir, render_path.name)
        if source is None:
            missing += 1
            continue
        materialize(source, dst, mode, force)
        matched += 1
    return {"renders": len(render_files), "matched": matched, "missing": missing, "existing": existing}


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    source_dir = Path(args.source_image_dir).expanduser().resolve()
    if not output_root.exists():
        raise SystemExit(f"Output root does not exist: {output_root}")
    if not source_dir.exists():
        raise SystemExit(f"Source image directory does not exist: {source_dir}")

    for split in args.splits:
        stats = repair_split(output_root, source_dir, split, args.mode, args.force)
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()
