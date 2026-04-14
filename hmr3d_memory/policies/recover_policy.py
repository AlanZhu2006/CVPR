from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import MemoryConfig


@dataclass
class RecoverDecision:
    allow: bool
    effective_alpha: float
    reason: str


class RecoverPolicy:
    """Rule-based modulation of recovery injection after geometry verification (HMR3D v2)."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config

    def decide_after_verify(
        self,
        proposal: Any,
        *,
        geo_gain: float,
        conf_delta: float,
        baseline_anchor_score: float | None = None,
        candidate_anchor_score: float | None = None,
    ) -> RecoverDecision:
        if not self.config.enable_v2_recover_gate:
            return RecoverDecision(True, proposal.recovery_alpha, "v2_recover_gate_disabled")

        alpha = float(proposal.recovery_alpha)
        alpha = min(alpha, float(self.config.recover_max_injection_alpha))

        if self.config.recover_min_pose_agreement > 0.0 and baseline_anchor_score is not None and candidate_anchor_score is not None:
            agree = 1.0 - abs(baseline_anchor_score - candidate_anchor_score) / max(
                abs(baseline_anchor_score), 1e-6
            )
            if agree < self.config.recover_min_pose_agreement:
                return RecoverDecision(False, alpha, "pose_agreement_low")

        if geo_gain < float(self.config.recover_geo_gain_soft_thresh):
            alpha *= float(self.config.recover_alpha_scale_on_low_geo_gain)

        if conf_delta < float(self.config.recover_conf_delta_soft_thresh):
            alpha *= float(self.config.recover_alpha_scale_on_low_conf_delta)

        alpha = max(0.0, min(alpha, float(self.config.recover_max_injection_alpha)))
        if alpha < float(self.config.recover_min_effective_alpha):
            return RecoverDecision(False, alpha, "effective_alpha_below_min")

        if self.config.recover_blend_with_identity and alpha < proposal.recovery_alpha:
            return RecoverDecision(True, alpha, "alpha_reduced_for_stability")

        return RecoverDecision(True, alpha, "accepted")

