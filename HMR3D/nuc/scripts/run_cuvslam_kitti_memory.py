from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import CUVSLAMOfflineKITTIAdapter, MemoryRouter
from nuc_runtime.config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved cuVSLAM KITTI outputs into HMR3D MemoryRouter."
    )
    parser.add_argument(
        "--sequence-path",
        required=True,
        help="KITTI sequence directory containing image_0 and times.txt.",
    )
    parser.add_argument(
        "--trajectory-path",
        required=True,
        help="trajectory_tum.txt produced by cuVSLAM.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional runtime YAML config.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=0,
        help="Replay every N-th frame. Defaults to config input.frame_step.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit replay length (0 means use config input.max_frames).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to config output_dir.",
    )
    parser.add_argument(
        "--enable-v2",
        action="store_true",
        help="Enable the minimal v2 write-policy and recover-stability features.",
    )
    parser.add_argument(
        "--enable-v3",
        action="store_true",
        help="Enable v2 plus hierarchical bank, multi-candidate merge, and local adaptation.",
    )
    parser.add_argument(
        "--enable-gaussian",
        action="store_true",
        help="Enable incremental Gaussian submap building and warm-start recovery.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    if args.enable_v2:
        config.memory.enable_v2_write_policy = True
        config.memory.enable_pose_anchor_gate = True
        config.memory.enable_shadow_recover = True
    if args.enable_v3:
        config.memory.enable_v2_write_policy = True
        config.memory.enable_pose_anchor_gate = True
        config.memory.enable_shadow_recover = True
        config.memory.enable_hierarchical_bank = True
        config.memory.enable_multi_candidate_merge = True
        config.memory.enable_local_adapt = True
    if args.enable_gaussian:
        config.memory.enable_incremental_gaussian = True
    output_dir = Path(args.output_dir or config.output.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_step = args.frame_step or config.input.frame_step
    max_frames = args.max_frames or config.input.max_frames

    adapter = CUVSLAMOfflineKITTIAdapter(
        sequence_path=args.sequence_path,
        trajectory_path=args.trajectory_path,
        config=config.tracking,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    router = MemoryRouter(config.memory, output_dir=output_dir)

    events_log = []
    keyframe_count = 0
    for output in adapter:
        if output.is_keyframe:
            keyframe_count += 1
        events, snapshot = router.process(output)
        if events:
            events_log.extend(
                {
                    "frame_idx": event.frame_idx,
                    "timestamp_sec": event.timestamp_sec,
                    "event_type": event.event_type,
                    "payload": event.payload,
                    "snapshot": snapshot,
                }
                for event in events
            )

    summary = router.finalize()
    summary["config"] = config.to_dict()
    summary["sequence_path"] = str(Path(args.sequence_path).expanduser().resolve())
    summary["trajectory_path"] = str(Path(args.trajectory_path).expanduser().resolve())
    summary["effective_input"] = {"frame_step": frame_step, "max_frames": max_frames}
    summary["keyframe_count"] = keyframe_count
    summary["event_count"] = len(events_log)

    (output_dir / "events.json").write_text(
        json.dumps(events_log, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps({
        "output_dir": str(output_dir),
        "keyframe_count": keyframe_count,
        "event_count": len(events_log),
        "archives": summary["stats"].get("archives", 0),
        "recoveries": summary["stats"].get("recoveries", 0),
        "bank_size": len(summary["bank"]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
