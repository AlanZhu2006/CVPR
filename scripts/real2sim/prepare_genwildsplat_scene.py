#!/usr/bin/env python3
"""Prepare a GenWildSplat sparse-view scene from LingBot video outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_summary() -> Path:
    return _repo_root() / "CVPR/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_summary.json"


def _default_scene_dir() -> Path:
    return _repo_root() / "CVPR/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_full_6ctx/scene"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolve_frame_path(path_text: str, fallback_frame_dir: Path | None) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    if fallback_frame_dir is not None:
        candidate = fallback_frame_dir / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Frame path does not exist: {path_text}")


def _select_uniform_indices(frame_count: int, keyframes: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if keyframes <= 0:
        raise ValueError("keyframes must be positive")
    if keyframes > frame_count:
        raise ValueError(f"Requested {keyframes} keyframes from only {frame_count} frames")
    return np.rint(np.linspace(0, frame_count - 1, keyframes)).astype(int).tolist()


def _parse_frame_indices(value: str) -> list[int]:
    indices: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def _select_indices(args: argparse.Namespace, frame_count: int) -> tuple[list[int], str]:
    if args.frame_indices:
        selected = _parse_frame_indices(args.frame_indices)
        policy = "explicit_frame_indices"
    else:
        start = max(0, int(args.start_frame))
        end = int(args.end_frame) if int(args.end_frame) >= 0 else frame_count - 1
        end = min(frame_count - 1, max(start, end))
        local_count = end - start + 1
        local_selected = _select_uniform_indices(local_count, args.keyframes)
        selected = [start + index for index in local_selected]
        policy = "uniform_window" if start > 0 or end < frame_count - 1 else "uniform_full_sequence"

    if not selected:
        raise ValueError("No GenWildSplat context frames selected.")
    if len(selected) > args.keyframes:
        raise ValueError(f"Selected {len(selected)} frames but --keyframes is {args.keyframes}.")
    for index in selected:
        if index < 0 or index >= frame_count:
            raise ValueError(f"Selected frame {index} outside valid range [0, {frame_count - 1}]")
    return selected, policy


def _copy_image(src: Path, dst: Path, jpeg_quality: int) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    if dst.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(dst, quality=jpeg_quality, subsampling=0, optimize=True)
    else:
        image.save(dst)
    return image.size


def _write_empty_mask(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (0, 0, 0)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=_default_summary(), help="LingBot summary JSON containing image_paths.")
    parser.add_argument("--scene-dir", type=Path, default=_default_scene_dir(), help="Output GenWildSplat scene directory.")
    parser.add_argument("--fallback-frame-dir", type=Path, default=None, help="Fallback directory used if image_paths were moved.")
    parser.add_argument("--keyframes", type=int, default=6, help="Number of GenWildSplat context frames.")
    parser.add_argument("--frame-indices", default="", help="Comma-separated explicit LingBot frame indices for the context set.")
    parser.add_argument("--start-frame", type=int, default=0, help="First LingBot frame for uniform local-window context selection.")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last LingBot frame for uniform local-window context selection. -1 means final frame.")
    parser.add_argument("--jpeg-quality", type=int, default=98, help="JPEG quality for exported context frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _read_json(args.summary_json)
    image_paths = summary.get("image_paths") or []
    if not image_paths:
        raise ValueError(f"No image_paths found in {args.summary_json}")

    frame_count = int(summary.get("frame_count") or len(image_paths))
    if frame_count != len(image_paths):
        frame_count = len(image_paths)
    selected, keyframe_policy = _select_indices(args, frame_count)

    images_dir = args.scene_dir / "images"
    masks_dir = args.scene_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for order, frame_index in enumerate(selected):
        src = _resolve_frame_path(image_paths[frame_index], args.fallback_frame_dir)
        stem = f"_ctx_{order:02d}_frame_{frame_index:06d}"
        image_name = f"{stem}.jpg"
        mask_name = f"{stem}.png"
        size = _copy_image(src, images_dir / image_name, args.jpeg_quality)
        _write_empty_mask(masks_dir / mask_name, size)
        entries.append(
            {
                "context_order": order,
                "frame_index": int(frame_index),
                "source_image": str(src),
                "image": f"images/{image_name}",
                "mask": f"masks/{mask_name}",
                "width": int(size[0]),
                "height": int(size[1]),
                "mask_policy": "empty_no_transient_objects",
            }
        )

    output_root = args.scene_dir.parent
    inference_dir = output_root / "inference"
    repo = _repo_root()
    genwildsplat_root = repo / "CVPR/third_party_research/GenWildSplat"
    cache_dir = repo / "CVPR/cache/genwildsplat"
    command = (
        f"cd {genwildsplat_root} && "
        f"HF_HOME={cache_dir / 'huggingface'} XDG_CACHE_HOME={cache_dir / 'xdg'} "
        f"python3 src/eval_nvs_video.py "
        f"--data_dir {args.scene_dir.resolve()} "
        f"--output_path {inference_dir.resolve()} "
        f"--ckpt_path checkpoint/model.safetensors "
        f"--no_refine --export_ply --export_safetensors --export_poses "
        f"--max_context_frames {len(selected)}"
    )
    (output_root / "run_inference.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n", encoding="utf-8")

    manifest = {
        "schema": "lingbot_genwildsplat_scene.v1",
        "source_summary_json": str(args.summary_json.resolve()),
        "source_frame_count": int(frame_count),
        "keyframe_policy": keyframe_policy,
        "keyframe_indices": [int(i) for i in selected],
        "scene_dir": str(args.scene_dir.resolve()),
        "images_dir": str(images_dir.resolve()),
        "masks_dir": str(masks_dir.resolve()),
        "mask_policy": "empty masks because YOLOv8 transient mask weights are not present locally",
        "entries": entries,
        "genwildsplat": {
            "repo": str(genwildsplat_root.resolve()),
            "inference_dir": str(inference_dir.resolve()),
            "command": command,
            "requires": [
                "checkpoint/model.safetensors",
                "checkpoint/config.json",
            ],
        },
    }
    _write_json(output_root / "manifest.json", manifest)
    _write_json(args.scene_dir / "manifest.json", manifest)
    print(json.dumps({"scene_dir": str(args.scene_dir), "keyframe_indices": selected, "command": command}, indent=2))


if __name__ == "__main__":
    main()
