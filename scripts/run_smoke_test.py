#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr3d_memory.adapter import run_sequence_with_mode
from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.ttt3r_env import bootstrap_ttt3r_imports
from hmr3d_memory.ttt3r_io import prepare_input


def extract_frames(video_path: Path, output_dir: Path, num_frames: int, stride: int) -> list[str]:
    capture = cv2.VideoCapture(str(video_path))
    saved = []
    frame_idx = 0
    saved_idx = 0
    while capture.isOpened() and len(saved) < num_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_path = output_dir / f"frame_{saved_idx:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            saved.append(str(frame_path))
            saved_idx += 1
        frame_idx += 1
    capture.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", default="third_party/TTT3R/examples/taylor.mp4")
    parser.add_argument("--modes", nargs="+", default=["cut3r", "ttt3r", "hmr_full"])
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="reports/generated/smoke_test.json")
    args = parser.parse_args()

    bootstrap_ttt3r_imports(args.weights)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(args.weights).to(args.device)
    model.eval()
    memory_config = MemoryConfig()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hmr3d-smoke-") as temp_dir:
        frame_paths = extract_frames(Path(args.video), Path(temp_dir), args.num_frames, args.stride)
        views = prepare_input(frame_paths, size=args.size, revisit=1, update=True, crop=True)
        summaries = []
        for mode in args.modes:
            outputs, memory_stats = run_sequence_with_mode(
                views=views,
                model=model,
                device=args.device,
                mode=mode,
                memory_config=memory_config,
            )
            memory_events = memory_stats.pop("events", None)
            summaries.append(
                {
                    "mode": mode,
                    "num_frames": len(outputs["pred"]),
                    "memory": memory_stats,
                    "events": memory_events,
                }
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"video": args.video, "summaries": summaries}, handle, indent=2)


if __name__ == "__main__":
    main()
