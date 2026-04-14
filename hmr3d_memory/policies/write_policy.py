from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from ..config import MemoryConfig


@dataclass
class WriteDecision:
    accept: bool
    delay_frames: int = 0
    reason: str = ""
    quality_score: float = 1.0


class WritePolicy:
    """Rule-based archive write gate (HMR3D v2). Disabled when config.enable_v2_write_gate is False."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def decide(
        self,
        *,
        frame_idx: int,
        segment_descriptors: List[torch.Tensor],
        last_archived_segment_desc: Optional[torch.Tensor],
        optional_state_conf: Optional[float],
    ) -> WriteDecision:
        if not self.config.enable_v2_write_gate:
            return WriteDecision(True, 0, "v2_write_gate_disabled", 1.0)

        if not segment_descriptors:
            return WriteDecision(False, 0, "no_segment_descriptors", 0.0)

        current = F.normalize(
            torch.stack([d.flatten() for d in segment_descriptors], dim=0).mean(dim=0),
            dim=-1,
        )

        novelty = 1.0
        if last_archived_segment_desc is not None:
            la = last_archived_segment_desc.to(current.device).flatten()
            la = F.normalize(la, dim=-1)
            sim = float(F.cosine_similarity(current.unsqueeze(0), la.unsqueeze(0), dim=-1).mean())
            novelty = max(0.0, min(1.0, 1.0 - sim))
            thresh = 1.0 - float(self.config.write_min_segment_novelty)
            if sim > thresh and self.config.write_min_segment_novelty > 0.0:
                return WriteDecision(False, 0, "novelty_too_low", quality_score=novelty)

        if optional_state_conf is not None and self.config.write_min_state_confidence > 0.0:
            if optional_state_conf < self.config.write_min_state_confidence:
                delay = int(self.config.write_delay_frames_on_low_conf)
                if delay > 0:
                    return WriteDecision(False, delay, "state_conf_low_deferred", novelty)
                return WriteDecision(False, 0, "state_conf_low", novelty)

        quality = float(novelty)
        if optional_state_conf is not None:
            quality = 0.5 * novelty + 0.5 * max(0.0, min(1.0, (optional_state_conf + 10.0) / 10.0))

        if self.config.archive_quality_score_thresh > 0.0 and quality < self.config.archive_quality_score_thresh:
            return WriteDecision(False, 0, "quality_below_thresh", quality)

        return WriteDecision(True, 0, "accepted", quality_score=quality)
