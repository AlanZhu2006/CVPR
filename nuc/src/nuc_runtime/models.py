from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def identity_pose() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def pose_translation(pose: np.ndarray) -> np.ndarray:
    return pose[:3, 3]


@dataclass
class FramePacket:
    frame_idx: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    source_name: str


@dataclass
class TrackingOutput:
    frame_idx: int
    timestamp_sec: float
    pose: np.ndarray
    is_keyframe: bool
    descriptor: np.ndarray
    orb_descriptors: np.ndarray | None
    keypoint_count: int
    match_count: int
    inlier_count: int
    pixel_motion: float
    track_ok: bool
    frame_shape: tuple[int, int]
    image_path: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyframeRecord:
    frame_idx: int
    timestamp_sec: float
    pose: np.ndarray
    descriptor: np.ndarray
    orb_descriptors: np.ndarray | None
    image_path: str | None
    keypoint_count: int
    match_count: int
    inlier_count: int
    pixel_motion: float


@dataclass
class ActiveSubmap:
    submap_id: int
    created_frame_idx: int
    keyframes: list[KeyframeRecord] = field(default_factory=list)
    recovered_from: list[int] = field(default_factory=list)
    injected_descriptors: list[np.ndarray] = field(default_factory=list)
    last_recover_frame_idx: int = -1

    def keyframe_count(self) -> int:
        return len(self.keyframes)

    def descriptor(self) -> np.ndarray:
        vectors = [item.descriptor for item in self.keyframes]
        vectors.extend(self.injected_descriptors)
        if not vectors:
            return np.zeros(1, dtype=np.float32)
        stacked = np.vstack(vectors).astype(np.float32)
        mean_vec = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm > 0:
            mean_vec /= norm
        return mean_vec

    def centroid(self) -> np.ndarray:
        if not self.keyframes:
            return np.zeros(3, dtype=np.float32)
        translations = np.vstack([pose_translation(item.pose) for item in self.keyframes])
        return translations.mean(axis=0)


@dataclass
class ArchivedSubmap:
    submap_id: int
    frame_indices: list[int]
    descriptor: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    anchor_image_path: str | None
    anchor_orb_descriptors: np.ndarray | None
    anchor_frame_idx: int
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventRecord:
    frame_idx: int
    timestamp_sec: float
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
