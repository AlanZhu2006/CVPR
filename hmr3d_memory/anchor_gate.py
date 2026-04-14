"""Anchor-pose verification helpers (kept separate from adapter to avoid heavy deps in unit tests)."""

from __future__ import annotations

from typing import Dict

from .config import MemoryConfig
from .router import RecoveryProposal


def should_accept_anchor_pose_candidate(
    *,
    proposal: RecoveryProposal,
    baseline_anchor_quality: Dict[str, float] | None,
    candidate_anchor_quality: Dict[str, float] | None,
    cfg: MemoryConfig,
) -> bool:
    if not cfg.enable_anchor_pose_verification:
        return True
    if cfg.anchor_pose_only_for_ambiguous:
        is_ambiguous = (
            proposal.candidate_rank >= cfg.verification_ambiguity_rank_threshold
            or abs(proposal.query_state_gap) < cfg.verification_ambiguity_gap_thresh
            or not proposal.is_latest_archive
        )
        if not is_ambiguous:
            return True
    if baseline_anchor_quality is None or candidate_anchor_quality is None:
        return False
    baseline_score = baseline_anchor_quality["anchor_score"]
    candidate_score = candidate_anchor_quality["anchor_score"]
    ratio_ok = candidate_score <= baseline_score * cfg.anchor_pose_score_ratio_thresh
    gain_ok = (baseline_score - candidate_score) >= cfg.anchor_pose_min_score_gain
    return ratio_ok and gain_ok


# Back-compat for code/tests that still expect the private name
_should_accept_anchor_pose_candidate = should_accept_anchor_pose_candidate
