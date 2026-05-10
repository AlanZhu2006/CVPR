#!/usr/bin/env python3
"""Create GenWildSplat transient-object masks with YOLO segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


TARGET_CLASSES = {0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 56, 67, 73}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_scene_dir() -> Path:
    return _repo_root() / "CVPR/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_full_6ctx/scene"


def _default_weights() -> Path:
    return _repo_root() / "CVPR/third_party_research/GenWildSplat/checkpoint/yolov8x-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, default=_default_scene_dir(), help="GenWildSplat scene directory containing images/ and masks/.")
    parser.add_argument("--weights", type=Path, default=_default_weights(), help="YOLO segmentation checkpoint.")
    parser.add_argument("--conf", type=float, default=0.01, help="YOLO confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install ultralytics to generate YOLO masks: python3 -m pip install --user ultralytics") from exc

    images_dir = args.scene_dir / "images"
    masks_dir = args.scene_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    summary = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        result = model(str(image_path), conf=args.conf, verbose=False)[0]
        h, w = result.orig_img.shape[:2]
        binary = np.zeros((h, w), dtype=np.uint8)
        if result.masks is not None:
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                if int(cls) in TARGET_CLASSES:
                    pts = mask.reshape(-1, 1, 2).astype(int)
                    cv2.fillPoly(binary, [pts], 255)
        output_path = masks_dir / image_path.with_suffix(".png").name
        cv2.imwrite(str(output_path), binary)
        masked_pixels = int((binary > 0).sum())
        summary.append(
            {
                "image": image_path.name,
                "mask": output_path.name,
                "masked_pixels": masked_pixels,
                "masked_percent": masked_pixels / float(max(1, h * w)) * 100.0,
            }
        )

    summary_path = args.scene_dir / "mask_summary.json"
    summary_path.write_text(json.dumps({"schema": "lingbot_genwildsplat_masks.v1", "masks": summary}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene_dir": str(args.scene_dir), "masks": summary}, indent=2))


if __name__ == "__main__":
    main()
