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
    min_translation_step: float = 0.05
    max_translation_step: float = 0.5


@dataclass
class MemoryConfig:
    short_window_size: int = 20
    active_max_keyframes: int = 12
    active_max_age: int = 120
    active_max_distance: float = 2.5
    active_similarity_floor: float = 0.72
    archive_min_keyframes: int = 3
    archive_min_mean_match_count: float = 100.0
    archive_min_mean_descriptor_score: float = 0.78
    anchor_topk: int = 3
    retrieve_topk: int = 3
    retrieve_similarity_threshold: float = 0.82
    retrieve_cooldown_frames: int = 20
    geo_verify_min_matches: int = 25
    enable_recover: bool = True
    enable_v2_write_policy: bool = False
    enable_pose_anchor_gate: bool = False
    pose_anchor_translation_threshold: float = 20.0
    pose_anchor_rotation_threshold_deg: float = 15.0
    enable_shadow_recover: bool = False
    shadow_promote_min_matches: int = 30
    shadow_similarity_threshold: float = 0.88
    enable_hierarchical_bank: bool = False
    scene_summary_distance_threshold: float = 35.0
    scene_summary_max_entries: int = 4
    scene_topk: int = 2
    enable_multi_candidate_merge: bool = False
    merge_topk: int = 3
    merge_min_candidates: int = 2
    merge_support_translation_ratio: float = 1.35
    merge_support_rotation_ratio: float = 1.35
    merge_support_similarity_floor: float = 0.9
    enable_local_adapt: bool = False
    local_adapt_descriptor_gain: float = 0.35
    enable_incremental_gaussian: bool = False
    enable_local_surface_volume: bool = True
    enable_semantic_like_region_filter: bool = True
    gaussian_min_pair_matches: int = 24
    gaussian_max_points_per_pair: int = 96
    gaussian_reproj_error_px: float = 3.0
    gaussian_default_scale: float = 0.05
    gaussian_warmstart_max_points: int = 192
    enable_gaussian_stereo_seed: bool = True
    gaussian_stereo_grid_stride: int = 12
    gaussian_stereo_num_disparities: int = 64
    gaussian_stereo_block_size: int = 5
    gaussian_stereo_min_disparity_px: float = 1.0
    gaussian_stereo_max_depth_m: float = 60.0
    gaussian_stereo_max_points_per_frame: int = 512
    gaussian_stereo_stride_scale: float = 0.9
    gaussian_pair_stride_scale: float = 0.65
    gaussian_optimize_steps: int = 3
    gaussian_optimize_window: int = 4
    gaussian_optimize_lr_position: float = 0.12
    gaussian_optimize_lr_scale: float = 0.08
    gaussian_optimize_lr_opacity: float = 0.12
    gaussian_optimize_lr_color: float = 0.28
    gaussian_stereo_feature_points: int = 768
    gaussian_stereo_consistency_threshold_px: float = 1.5
    gaussian_stereo_texture_threshold: float = 6.0
    gaussian_stereo_fusion_voxel_size: float = 0.10
    gaussian_stereo_quadtree_min_block: int = 28
    gaussian_stereo_quadtree_var_threshold: float = 0.45
    gaussian_stereo_depth_edge_threshold: float = 1.4
    gaussian_local_volume_voxel_size: float = 0.08
    gaussian_local_volume_extract_points: int = 1800
    gaussian_local_volume_fuse_stride: int = 6
    gaussian_local_volume_confidence_threshold: float = 0.18
    gaussian_local_volume_min_observations: int = 2
    gaussian_local_volume_visible_depth_margin_m: float = 0.35
    gaussian_local_volume_visible_max_depth_m: float = 35.0
    gaussian_local_volume_normal_view_cosine: float = -0.35
    gaussian_local_volume_block_bad_ratio: float = 0.55
    gaussian_local_volume_thin_cell_px: int = 10
    gaussian_local_volume_thin_max_layers: int = 1
    gaussian_local_volume_thin_depth_gap_m: float = 0.12
    gaussian_region_sky_top_ratio: float = 0.46
    gaussian_region_far_depth_m: float = 28.0
    gaussian_region_low_confidence_threshold: float = 0.22
    gaussian_region_vegetation_texture_scale: float = 2.4
    gaussian_region_dynamic_min_depth_m: float = 2.0
    gaussian_region_dynamic_max_depth_m: float = 18.0
    gaussian_optimize_topk: int = 6000
    gaussian_optimize_error_threshold: float = 20.0
    gaussian_optimize_unstable_decay: float = 0.88
    gaussian_optimize_new_point_boost: float = 0.45
    gaussian_optimize_recover_boost: float = 0.65
    gaussian_optimize_depth_weight: float = 1.15
    gaussian_optimize_depth_score_weight: float = 0.85
    gaussian_optimize_depth_gate_m: float = 2.5
    gaussian_coarse_voxel_size: float = 0.45
    gaussian_coarse_max_points: int = 1800
    gaussian_full_recent_archives: int = 2


@dataclass
class OutputConfig:
    output_dir: str = "nuc_output/default_run"
    render_save_images: bool = True
    render_compare_stride: int = 10
    render_max_archived_submaps: int = 8
    render_min_radius_px: float = 1.5
    render_max_radius_px: float = 20.0
    render_background_gray: int = 0
    render_enable_hole_fill: bool = True
    render_hole_fill_kernel: int = 7
    render_hole_fill_max_area: int = 4096
    render_depth_soften: float = 0.92
    render_internal_scale: float = 0.5
    render_tile_size: int = 64
    render_max_active_points: int = 48000
    render_depth_window_m: float = 55.0
    render_max_archived_points: int = 12000
    render_max_warmstart_points: int = 4096
    render_view_budget_points: int = 14000
    render_surface_depth_sigma: float = 0.35
    render_surface_opacity_gain: float = 1.15


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
