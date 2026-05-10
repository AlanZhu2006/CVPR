#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a smaller parsed FAST-LIVO2 dataset subset for GS-SDF."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Directory that contains images/, depths/, color_poses.txt, depth_poses.txt",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory to create the subset dataset in",
    )
    parser.add_argument("--color-start", type=int, help="Start index in sorted color images")
    parser.add_argument("--color-count", type=int, help="Number of color frames to keep")
    parser.add_argument("--depth-start", type=int, help="Start index in sorted depth clouds")
    parser.add_argument("--depth-count", type=int, help="Number of depth frames to keep")
    parser.add_argument(
        "--fraction-start",
        type=float,
        default=0.4,
        help="Fallback normalized start position when explicit starts are omitted (default: 0.4)",
    )
    parser.add_argument(
        "--fraction-length",
        type=float,
        default=0.16,
        help="Fallback normalized length when explicit counts are omitted (default: 0.16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    return parser.parse_args()


def read_pose_blocks(path: Path) -> list[list[str]]:
    lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 4 != 0:
        raise ValueError(f"{path} does not contain a multiple of 4 non-empty lines.")
    return [lines[index : index + 4] for index in range(0, len(lines), 4)]


def write_pose_blocks(path: Path, blocks: list[list[str]]) -> None:
    lines: list[str] = []
    for block in blocks:
        lines.extend(block)
    path.write_text("\n".join(lines) + "\n")


def compute_slice(
    total: int,
    start: int | None,
    count: int | None,
    fraction_start: float,
    fraction_length: float,
) -> tuple[int, int]:
    if total <= 0:
        raise ValueError("Total frame count must be positive.")
    if start is None:
        start = max(0, min(total - 1, int(math.floor(total * fraction_start))))
    if count is None:
        count = max(32, int(math.floor(total * fraction_length)))
    start = max(0, min(start, total - 1))
    count = max(1, min(count, total - start))
    return start, count


def materialize_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    images_dir = dataset_root / "images"
    depths_dir = dataset_root / "depths"
    color_poses_path = dataset_root / "color_poses.txt"
    depth_poses_path = dataset_root / "depth_poses.txt"

    required_paths = [images_dir, depths_dir, color_poses_path, depth_poses_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Dataset root is incomplete. Missing: {missing}")

    color_files = sorted(images_dir.glob("*.png"), key=lambda path: float(path.stem))
    depth_files = sorted(depths_dir.glob("*.ply"), key=lambda path: float(path.stem))
    color_pose_blocks = read_pose_blocks(color_poses_path)
    depth_pose_blocks = read_pose_blocks(depth_poses_path)

    if len(color_files) != len(color_pose_blocks):
        raise SystemExit(
            f"Color frame count mismatch: {len(color_files)} images vs {len(color_pose_blocks)} pose blocks"
        )
    if len(depth_files) != len(depth_pose_blocks):
        raise SystemExit(
            f"Depth frame count mismatch: {len(depth_files)} clouds vs {len(depth_pose_blocks)} pose blocks"
        )

    color_start, color_count = compute_slice(
        len(color_files),
        args.color_start,
        args.color_count,
        args.fraction_start,
        args.fraction_length,
    )
    depth_start, depth_count = compute_slice(
        len(depth_files),
        args.depth_start,
        args.depth_count,
        args.fraction_start,
        args.fraction_length,
    )

    if output_root.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_root}. Re-run with --force.")
        shutil.rmtree(output_root)

    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "depths").mkdir(parents=True, exist_ok=True)

    selected_color_files = color_files[color_start : color_start + color_count]
    selected_depth_files = depth_files[depth_start : depth_start + depth_count]

    for index, src in enumerate(selected_color_files):
        dst = output_root / "images" / f"{index:06d}.png"
        materialize_file(src, dst)

    for index, src in enumerate(selected_depth_files):
        dst = output_root / "depths" / f"{index:06d}.ply"
        materialize_file(src, dst)

    write_pose_blocks(
        output_root / "color_poses.txt",
        color_pose_blocks[color_start : color_start + color_count],
    )
    write_pose_blocks(
        output_root / "depth_poses.txt",
        depth_pose_blocks[depth_start : depth_start + depth_count],
    )

    metadata = {
        "sourceRoot": str(dataset_root),
        "outputRoot": str(output_root),
        "color": {
            "start": color_start,
            "count": color_count,
            "sourceFirst": selected_color_files[0].name if selected_color_files else None,
            "sourceLast": selected_color_files[-1].name if selected_color_files else None,
        },
        "depth": {
            "start": depth_start,
            "count": depth_count,
            "sourceFirst": selected_depth_files[0].name if selected_depth_files else None,
            "sourceLast": selected_depth_files[-1].name if selected_depth_files else None,
        },
    }
    (output_root / "subset_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
