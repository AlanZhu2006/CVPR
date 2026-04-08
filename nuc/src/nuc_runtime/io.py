from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from nuc_runtime.models import FramePacket


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _resize_if_needed(frame, resize_width: int):
    if resize_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame
    scale = resize_width / float(width)
    new_height = int(height * scale)
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def iter_frames(
    input_path: str | Path,
    frame_step: int = 1,
    max_frames: int = 0,
    default_fps: float = 30.0,
    resize_width: int = 0,
) -> Iterator[FramePacket]:
    path = Path(input_path)
    frame_step = max(frame_step, 1)

    if path.is_dir():
        image_paths = sorted(
            item for item in path.iterdir() if item.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_frames > 0:
            image_paths = image_paths[: max_frames * frame_step]

        emitted = 0
        for idx, image_path in enumerate(image_paths[::frame_step]):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame = _resize_if_needed(frame, resize_width)
            yield FramePacket(
                frame_idx=idx,
                timestamp_sec=idx / float(default_fps),
                frame_bgr=frame,
                source_name=image_path.name,
            )
            emitted += 1
            if max_frames > 0 and emitted >= max_frames:
                break
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open input path: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = default_fps

    frame_idx = 0
    emitted = 0
    source_name = path.name
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        frame = _resize_if_needed(frame, resize_width)
        yield FramePacket(
            frame_idx=emitted,
            timestamp_sec=frame_idx / float(fps),
            frame_bgr=frame,
            source_name=f"{source_name}:{frame_idx}",
        )
        emitted += 1
        frame_idx += 1
        if max_frames > 0 and emitted >= max_frames:
            break

    cap.release()
