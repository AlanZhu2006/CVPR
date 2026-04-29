from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import (
    CUVSLAMLingBotReconAdapter,
    build_lingbot_window_descriptor,
    load_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scheme-B baseline: cuVSLAM keyframe window + LingBot reconstruction export."
    )
    parser.add_argument("--sequence-path", required=True)
    parser.add_argument("--trajectory-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-keyframes", type=int, default=2)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    adapter = CUVSLAMLingBotReconAdapter(
        sequence_path=args.sequence_path,
        trajectory_path=args.trajectory_path,
        tracking_config=config.tracking,
        model_path=args.model_path,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        lingbot_window_keyframes=args.window_keyframes,
        lingbot_force_cpu=args.force_cpu,
    )
    bundle = adapter.export_first_window(args.output_dir)
    descriptor = build_lingbot_window_descriptor(bundle)
    result = {
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "predictions_npz": str(bundle.predictions_npz),
        "summary_json": str(bundle.summary_json),
        "frame_count": len(bundle.image_paths),
        "prediction_keys": bundle.summary.get("prediction_keys", []),
        "frame_indices": bundle.summary.get("metadata", {}).get("frame_indices", []),
        "descriptor_dim": int(descriptor.shape[0]),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
