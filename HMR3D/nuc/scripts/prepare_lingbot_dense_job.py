from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import CUVSLAMOfflineKITTIAdapter, load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a self-contained cuVSLAM keyframe window for remote "
            "LingBot dense-geometry inference."
        )
    )
    parser.add_argument("--sequence-path", required=True, help="KITTI sequence directory.")
    parser.add_argument("--trajectory-path", required=True, help="cuVSLAM TUM trajectory.")
    parser.add_argument("--output-dir", required=True, help="Directory for the job bundle.")
    parser.add_argument("--config", default="", help="Runtime config; tracking section is used.")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-keyframes", type=int, default=8)
    parser.add_argument("--manifest-name", default="lingbot_job_manifest.json")
    parser.add_argument("--image-subdir", default="images")
    parser.add_argument("--no-copy-images", action="store_true")
    return parser.parse_args()


def _jsonable_array(array: np.ndarray | None) -> list[Any]:
    if array is None:
        return []
    return array.tolist()


def _copy_or_reference_image(
    image_path: str,
    output_dir: Path,
    image_subdir: str,
    frame_idx: int,
    copy_images: bool,
) -> tuple[str, str]:
    source = Path(image_path).expanduser().resolve()
    if not copy_images:
        return str(source), str(source)

    image_dir = output_dir / image_subdir
    image_dir.mkdir(parents=True, exist_ok=True)
    target = image_dir / f"{frame_idx:06d}{source.suffix.lower() or '.png'}"
    shutil.copy2(source, target)
    return str(Path(image_subdir) / target.name), str(source)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_runtime_config(args.config or None)
    source = CUVSLAMOfflineKITTIAdapter(
        sequence_path=args.sequence_path,
        trajectory_path=args.trajectory_path,
        config=config.tracking,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
    )

    keyframes = []
    for item in source:
        if item.is_keyframe:
            keyframes.append(item)
        if len(keyframes) >= max(1, args.window_keyframes):
            break
    if not keyframes:
        raise RuntimeError("No cuVSLAM keyframes were collected")

    image_paths = []
    original_image_paths = []
    for item in keyframes:
        if not item.image_path:
            raise RuntimeError(f"Keyframe {item.frame_idx} has no image path")
        job_path, original_path = _copy_or_reference_image(
            image_path=item.image_path,
            output_dir=output_dir,
            image_subdir=args.image_subdir,
            frame_idx=item.frame_idx,
            copy_images=not args.no_copy_images,
        )
        image_paths.append(job_path)
        original_image_paths.append(original_path)

    descriptor_stack = np.vstack([item.descriptor for item in keyframes]).astype(np.float32)
    manifest = {
        "schema_version": 1,
        "source": "cuvslam_lingbot_dense_job",
        "sequence_path": str(Path(args.sequence_path).expanduser().resolve()),
        "trajectory_path": str(Path(args.trajectory_path).expanduser().resolve()),
        "config_path": str(Path(args.config).expanduser().resolve()) if args.config else "",
        "frame_step": int(args.frame_step),
        "max_frames": int(args.max_frames),
        "window_keyframes": int(args.window_keyframes),
        "image_paths": image_paths,
        "original_image_paths": original_image_paths,
        "metadata": {
            "source": "cuvslam_plus_lingbot_dense_job",
            "frame_indices": [int(item.frame_idx) for item in keyframes],
            "timestamps_sec": [float(item.timestamp_sec) for item in keyframes],
            "cuvslam_poses": [item.pose.astype(np.float32).tolist() for item in keyframes],
            "cuvslam_descriptors": [item.descriptor.astype(np.float32).tolist() for item in keyframes],
            "descriptors_mean": descriptor_stack.mean(axis=0).tolist(),
            "frame_shapes": [list(item.frame_shape) for item in keyframes],
            "keypoint_counts": [int(item.keypoint_count) for item in keyframes],
            "match_counts": [int(item.match_count) for item in keyframes],
            "inlier_counts": [int(item.inlier_count) for item in keyframes],
            "pixel_motion": [float(item.pixel_motion) for item in keyframes],
            "right_image_paths": [item.right_image_path or "" for item in keyframes],
            "keypoints_xy": [_jsonable_array(item.keypoints_xy) for item in keyframes],
        },
    }

    manifest_path = output_dir / args.manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = {
        "job_dir": str(output_dir),
        "manifest_json": str(manifest_path),
        "frame_count": len(keyframes),
        "frame_indices": manifest["metadata"]["frame_indices"],
        "copied_images": not args.no_copy_images,
        "image_count": len(image_paths),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
