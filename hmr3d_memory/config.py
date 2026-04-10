from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class MemoryConfig:
    archive_interval: int = 120
    max_archives: int = 64
    retrieval_topk: int = 3
    sequence_consistency_window: int = 0
    sequence_similarity_thresh: float = 0.0
    retrieval_similarity_thresh: float = 0.9
    verification_similarity_thresh: float = 0.82
    max_state_similarity_for_recover: float = 0.985
    query_state_gap_thresh: float = 0.0
    recovery_alpha: float = 0.5
    enable_geometric_verification: bool = True
    verification_geo_ratio_thresh: float = 0.99
    verification_conf_tolerance: float = 0.05
    verification_min_geo_gain: float = 0.0
    verification_stale_min_geo_gain: float = 0.0
    verification_ambiguity_gap_thresh: float = 0.01
    verification_ambiguity_rank_threshold: int = 2
    min_frames_before_retrieve: int = 30
    retrieval_cooldown: int = 30
    retrieval_attempt_cooldown: int = 0
    state_gate_hidden_dim: int = 384
    state_gate_init_bias: float = 10.0
    fast_weight_decay: float = 0.995
    fast_weight_decay_scale: float = 0.01
    fast_weight_lr: float = 0.001
    base_update_type: str = "ttt3r"
    enable_archive: bool = True
    enable_retrieval: bool = True
    reset_on_archive: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "MemoryConfig":
        if payload is None:
            return cls()
        return cls(**payload)

    def for_mode(self, mode: str) -> "MemoryConfig":
        cfg = MemoryConfig(**asdict(self))
        if mode == "hmr_archive_only":
            cfg.enable_archive = True
            cfg.enable_retrieval = False
        elif mode == "hmr_full":
            cfg.enable_archive = True
            cfg.enable_retrieval = True
        elif mode == "mem3r_pose_probe":
            cfg.enable_archive = False
            cfg.enable_retrieval = False
            cfg.base_update_type = "ttt3r"
        elif mode == "mem3r_like_runtime":
            cfg.enable_archive = False
            cfg.enable_retrieval = False
            cfg.base_update_type = "ttt3r"
        elif mode in {"cut3r", "ttt3r"}:
            cfg.enable_archive = False
            cfg.enable_retrieval = False
            cfg.base_update_type = mode
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
