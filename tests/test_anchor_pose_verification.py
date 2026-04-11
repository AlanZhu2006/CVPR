from __future__ import annotations

import math
import unittest

import torch

from hmr3d_memory.adapter import _should_accept_anchor_pose_candidate
from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.router import RecoveryProposal
from hmr3d_memory.ttt3r_io import compute_anchor_pose_score_from_matrices


def _dummy_proposal(*, candidate_rank: int, query_state_gap: float, is_latest_archive: bool) -> RecoveryProposal:
    zero = torch.zeros(1, 1, 1)
    return RecoveryProposal(
        archive_id=0,
        archive_frame_idx=0,
        candidate_rank=candidate_rank,
        query_similarity=0.95,
        state_similarity=0.90,
        sequence_similarity=0.0,
        query_state_gap=query_state_gap,
        recovery_alpha=0.35,
        is_latest_archive=is_latest_archive,
        archive_camera_pose=None,
        state_args=(zero, zero, zero, zero, zero),
    )


class AnchorPoseVerificationTest(unittest.TestCase):
    def test_anchor_pose_score_from_matrices_reflects_translation_and_rotation(self) -> None:
        archive_pose = torch.eye(4).unsqueeze(0)
        pred_pose = torch.eye(4).unsqueeze(0)
        pred_pose[:, 0, 3] = 2.0
        angle = math.pi / 2
        pred_pose[:, :3, :3] = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        quality = compute_anchor_pose_score_from_matrices(
            pred_pose,
            archive_pose,
            rotation_weight=0.5,
        )

        self.assertAlmostEqual(quality["anchor_translation_error"], 2.0, places=5)
        self.assertAlmostEqual(quality["anchor_rotation_error"], math.pi / 2, places=5)
        self.assertAlmostEqual(quality["anchor_score"], 2.0 + 0.5 * (math.pi / 2), places=5)

    def test_unambiguous_latest_candidate_skips_anchor_gate(self) -> None:
        cfg = MemoryConfig(
            enable_anchor_pose_verification=True,
            anchor_pose_only_for_ambiguous=True,
            verification_ambiguity_rank_threshold=2,
            verification_ambiguity_gap_thresh=0.01,
        )
        proposal = _dummy_proposal(candidate_rank=1, query_state_gap=0.2, is_latest_archive=True)

        accepted = _should_accept_anchor_pose_candidate(
            proposal=proposal,
            baseline_anchor_quality=None,
            candidate_anchor_quality=None,
            cfg=cfg,
        )

        self.assertTrue(accepted)

    def test_ambiguous_candidate_requires_anchor_improvement(self) -> None:
        cfg = MemoryConfig(
            enable_anchor_pose_verification=True,
            anchor_pose_only_for_ambiguous=True,
            anchor_pose_score_ratio_thresh=1.0,
            anchor_pose_min_score_gain=0.05,
            verification_ambiguity_rank_threshold=2,
            verification_ambiguity_gap_thresh=0.01,
        )
        proposal = _dummy_proposal(candidate_rank=2, query_state_gap=0.0, is_latest_archive=False)

        accepted = _should_accept_anchor_pose_candidate(
            proposal=proposal,
            baseline_anchor_quality={"anchor_score": 1.0},
            candidate_anchor_quality={"anchor_score": 0.9},
            cfg=cfg,
        )
        rejected = _should_accept_anchor_pose_candidate(
            proposal=proposal,
            baseline_anchor_quality={"anchor_score": 1.0},
            candidate_anchor_quality={"anchor_score": 0.98},
            cfg=cfg,
        )

        self.assertTrue(accepted)
        self.assertFalse(rejected)


if __name__ == "__main__":
    unittest.main()
