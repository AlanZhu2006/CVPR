#!/usr/bin/env python3
"""Create a minimal TUM-style tree under data/long_tum_s1/ for relpose smoke evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="smoke_seq", help="Sequence folder name under data/long_tum_s1/")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--video",
        type=Path,
        default=REPO_ROOT / "third_party" / "TTT3R" / "examples" / "taylor.mp4",
    )
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()

    import cv2

    rgb_dir = REPO_ROOT / "data" / "long_tum_s1" / args.seq / "rgb_1000"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gt_path = REPO_ROOT / "data" / "long_tum_s1" / args.seq / "groundtruth_1000.txt"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    saved = 0
    frame_idx = 0
    lines: list[str] = []
    dt = 1.0 / 30.0
    while saved < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            t = saved * dt
            out = rgb_dir / f"{t:.6f}.png"
            cv2.imwrite(str(out), frame)
            # Small per-frame translation so evo Umeyama alignment is non-degenerate (constant GT breaks align=True).
            x = saved * 0.02
            lines.append(f"{t:.6f} {x:.6f} 0 0 0 0 0 1\n")
            saved += 1
        frame_idx += 1
    cap.release()

    if saved < 2:
        raise SystemExit("Need at least 2 frames; increase video length or lower --stride.")

    gt_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {saved} frames under {rgb_dir}")
    print(f"Wrote {gt_path}")


if __name__ == "__main__":
    main()
