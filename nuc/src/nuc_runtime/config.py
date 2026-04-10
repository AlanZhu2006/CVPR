from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    frame_step: int = 1
    max_frames: int = 0
    resize_width: int = 0
    default_fps: float = 30.0
    rosbag_left_topic: str | None = None
    rosbag_right_topic: str | None = None
    rosbag_sync_tolerance_sec: float = 0.01


@dataclass
class TrackingConfig:
    max_features: int = 2000
    min_matches: int = 40
    min_pose_inliers: int = 16
    ratio_test: float = 0.75
    min_keyframe_gap: int = 3
    max_keyframe_gap: int = 15
    keyframe_motion_threshold: float = 18.0
    low_match_keyframe_threshold: int = 80
    focal_length_scale: float = 0.9
    min_translation_step: float = 0.05
    max_translation_step: float = 0.5
    stereo_baseline_m: float = 0.11
    stereo_ratio_test: float = 0.75
    min_stereo_disparity: float = 1.0
    max_stereo_vertical_diff: float = 2.0
    min_stereo_points: int = 24
    max_stereo_depth_m: float = 40.0


@dataclass
class MemoryConfig:
    short_window_size: int = 20
    active_max_keyframes: int = 12
    active_max_age: int = 120
    active_max_distance: float = 2.5
    active_similarity_floor: float = 0.72
    retrieve_topk: int = 3
    retrieve_similarity_threshold: float = 0.82
    retrieve_cooldown_frames: int = 20
    geo_verify_min_matches: int = 25
    enable_recover: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "nuc_output/default_run"
    save_debug_video: bool = True
    save_keyframe_images: bool = True
    debug_video_fps: float = 10.0
    debug_video_name: str = "debug.mp4"


@dataclass
class RuntimeConfig:
    input: InputConfig = field(default_factory=InputConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_updates(section: Any, updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if hasattr(section, key):
            setattr(section, key, value)


def load_runtime_config(path: str | Path | None) -> RuntimeConfig:
    config = RuntimeConfig()
    if path is None:
        return config

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section_name in ("input", "tracking", "memory", "output"):
        updates = data.get(section_name)
        if isinstance(updates, dict):
            _apply_updates(getattr(config, section_name), updates)
    return config
