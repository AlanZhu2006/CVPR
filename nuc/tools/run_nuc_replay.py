from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CPU-friendly NUC memory lifecycle prototype.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--input", type=str, required=True, help="Input video path or image directory.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from the config file.",
    )
    parser.add_argument(
        "--disable-recover",
        action="store_true",
        help="Run the same pipeline but skip descriptor injection on retrieve hits.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from nuc_runtime.config import load_runtime_config
    from nuc_runtime.io import iter_frames
    from nuc_runtime.memory_router import MemoryRouter
    from nuc_runtime.output import ArtifactWriter
    from nuc_runtime.tracking import ORBTrackingFrontend

    config = load_runtime_config(args.config)
    if args.output_dir:
        config.output.output_dir = args.output_dir
    if args.disable_recover:
        config.memory.enable_recover = False

    output_root = Path(config.output.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tracker = ORBTrackingFrontend(config.tracking)
    router = MemoryRouter(config.memory)
    writer = ArtifactWriter(output_root, config.output)

    frame_iter = iter_frames(
        input_path=args.input,
        frame_step=config.input.frame_step,
        max_frames=config.input.max_frames,
        default_fps=config.input.default_fps,
        resize_width=config.input.resize_width,
    )

    frame_count = 0
    keyframe_count = 0
    last_event_type = None

    try:
        for packet in tqdm(frame_iter, desc="nuc-replay"):
            tracking = tracker.process(packet)
            if tracking.is_keyframe:
                tracking.image_path = writer.save_keyframe_image(packet.frame_idx, packet.frame_bgr)
                keyframe_count += 1

            events, snapshot = router.process(tracking)
            if events:
                last_event_type = events[-1].event_type
            writer.append_events(events)
            writer.append_pose(tracking, snapshot)
            writer.maybe_write_debug_frame(packet.frame_bgr, tracking, snapshot, last_event_type)
            frame_count += 1
    finally:
        router_dump = router.finalize()
        summary = {
            "input": args.input,
            "config": config.to_dict(),
            "runtime": {
                "frames": frame_count,
                "keyframes": keyframe_count,
                "recover_enabled": config.memory.enable_recover,
            },
            "stats": router_dump["stats"],
        }
        writer.write_summary(summary, router_dump)
        writer.close()


if __name__ == "__main__":
    main()
