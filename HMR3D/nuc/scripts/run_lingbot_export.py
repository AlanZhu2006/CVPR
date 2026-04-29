from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import LingBotReconstructor, build_lingbot_window_descriptor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LingBot reconstruction on an image folder and export predictions."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-k", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--force-cpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        str(path)
        for path in Path(args.image_folder).expanduser().resolve().iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if args.first_k > 0:
        image_paths = image_paths[: args.first_k]
    if args.stride > 1:
        image_paths = image_paths[:: args.stride]

    reconstructor = LingBotReconstructor(
        model_path=args.model_path,
        force_cpu=args.force_cpu,
        camera_num_iterations=1,
        keyframe_interval=1,
    )
    bundle = reconstructor.export_bundle(
        image_paths=image_paths,
        output_dir=args.output_dir,
        metadata={"source": "standalone_lingbot_export"},
    )
    descriptor = build_lingbot_window_descriptor(bundle)
    result = {
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "frame_count": len(bundle.image_paths),
        "predictions_npz": str(bundle.predictions_npz),
        "summary_json": str(bundle.summary_json),
        "descriptor_dim": int(descriptor.shape[0]),
        "prediction_keys": bundle.summary.get("prediction_keys", []),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
