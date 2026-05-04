#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hikrobot_mvs_ros2_publisher import HikRobotCamera  # noqa: E402


def _write_ppm(path: Path, rgb: bytes, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        fh.write(rgb)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    default_output_dir = repo_root / "nuc_output" / "hikrobot_first_ten_frames"

    parser = argparse.ArgumentParser(
        description="Read the first N frames from a HikRobot MVS camera and save them as PPM images."
    )
    parser.add_argument("--index", type=int, default=0, help="Camera index returned by the MVS SDK.")
    parser.add_argument("--count", type=int, default=10, help="Number of frames to capture.")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="Per-frame timeout in milliseconds.")
    parser.add_argument("--exposure-us", type=float, default=None, help="Optional exposure time in microseconds.")
    parser.add_argument("--gain", type=float, default=None, help="Optional analog gain.")
    parser.add_argument("--fps", type=float, default=None, help="Optional acquisition frame rate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory to write frames into. Default: {default_output_dir}",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="File prefix for saved images. Files are written as <prefix>_000.ppm, ...",
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")

    camera = HikRobotCamera(args.index, args.timeout_ms)
    camera.open(args.exposure_us, args.gain, args.fps)
    print(f"Saving {args.count} frame(s) to {args.output_dir}", flush=True)
    try:
        for frame_idx in range(args.count):
            rgb, width, height = camera.read_rgb()
            output_path = args.output_dir / f"{args.prefix}_{frame_idx:03d}.ppm"
            _write_ppm(output_path, rgb, width, height)
            print(
                f"[{frame_idx + 1}/{args.count}] saved {output_path.name} "
                f"({width}x{height}, {len(rgb)} bytes rgb8)",
                flush=True,
            )
    finally:
        camera.close()

    print("Capture complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
