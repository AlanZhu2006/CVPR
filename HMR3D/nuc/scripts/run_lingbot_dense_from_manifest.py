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
        description="Run LingBot dense reconstruction from a prepared job manifest."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--force-cpu", action="store_true", default=False)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--no-offload-to-cpu", action="store_true")
    return parser.parse_args()


def _resolve_image_path(path_str: str, manifest_dir: Path, image_root: Path | None) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())

    candidates = []
    if image_root is not None:
        candidates.append(image_root / path)
        candidates.append(image_root / path.name)
    candidates.append(manifest_dir / path)
    candidates.append(manifest_dir / path.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Failed to resolve image path {path_str}. Searched: {searched}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_json).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent
    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else manifest_dir
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest_dir / "lingbot_output"
    )

    image_paths = [
        _resolve_image_path(path_str, manifest_dir=manifest_dir, image_root=image_root)
        for path_str in manifest.get("image_paths", [])
    ]
    if not image_paths:
        raise RuntimeError("Manifest does not contain image_paths")

    metadata = dict(manifest.get("metadata", {}))
    metadata.update(
        {
            "source": metadata.get("source", "cuvslam_plus_lingbot_dense_job"),
            "dense_job_manifest": str(manifest_path),
            "original_image_paths": manifest.get("original_image_paths", []),
            "job_schema_version": manifest.get("schema_version", 0),
        }
    )

    reconstructor = LingBotReconstructor(
        model_path=args.model_path,
        image_size=args.image_size,
        patch_size=args.patch_size,
        mode=args.mode,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        camera_num_iterations=args.camera_num_iterations,
        offload_to_cpu=not args.no_offload_to_cpu,
        force_cpu=args.force_cpu,
    )
    bundle = reconstructor.export_bundle(
        image_paths=image_paths,
        output_dir=output_dir,
        metadata=metadata,
    )
    descriptor = build_lingbot_window_descriptor(bundle)
    result = {
        "output_dir": str(output_dir),
        "predictions_npz": str(bundle.predictions_npz),
        "summary_json": str(bundle.summary_json),
        "frame_count": len(bundle.image_paths),
        "frame_indices": metadata.get("frame_indices", []),
        "descriptor_dim": int(descriptor.shape[0]),
        "device": bundle.summary.get("device", ""),
        "prediction_keys": bundle.summary.get("prediction_keys", []),
    }
    run_summary_path = output_dir / "lingbot_remote_run_summary.json"
    run_summary_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
