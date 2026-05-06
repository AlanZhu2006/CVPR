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
    parser.add_argument("--lingbot-map-root", default="")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--first-k", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--model-image-size", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--no-offload-to-cpu", action="store_true")
    parser.add_argument("--disable-camera", action="store_true")
    parser.add_argument("--disable-point", action="store_true")
    parser.add_argument("--force-cpu", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lingbot_map_root:
        import os

        os.environ["LINGBOT_MAP_ROOT"] = str(Path(args.lingbot_map_root).expanduser().resolve())
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
        image_size=args.image_size,
        model_image_size=args.model_image_size or None,
        patch_size=args.patch_size,
        mode=args.mode,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        camera_num_iterations=args.camera_num_iterations,
        use_sdpa=args.use_sdpa,
        offload_to_cpu=not args.no_offload_to_cpu,
        enable_camera=not args.disable_camera,
        enable_point=not args.disable_point,
        force_cpu=args.force_cpu,
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
