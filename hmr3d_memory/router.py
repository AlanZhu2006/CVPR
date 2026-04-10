from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .config import MemoryConfig


StateTuple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _flatten_descriptor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() == 3:
        value = value.mean(dim=1)
    elif value.dim() > 3:
        value = value.reshape(value.shape[0], -1)
    value = value.reshape(value.shape[0], -1)
    value = F.normalize(value, dim=-1)
    return value.detach().cpu()


@dataclass
class ArchiveEntry:
    archive_id: int
    frame_idx: int
    segment_start: int
    segment_length: int
    descriptor: torch.Tensor
    tail_descriptors: torch.Tensor
    state_descriptor: torch.Tensor
    state_feat: torch.Tensor
    state_pos: torch.Tensor
    init_state_feat: torch.Tensor
    mem: torch.Tensor
    init_mem: torch.Tensor


@dataclass
class RecoveryProposal:
    archive_id: int
    archive_frame_idx: int
    candidate_rank: int
    query_similarity: float
    state_similarity: float
    sequence_similarity: float
    query_state_gap: float
    recovery_alpha: float
    is_latest_archive: bool
    state_args: StateTuple


class MemoryRouter:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.archive_bank: List[ArchiveEntry] = []
        self.segment_descriptors: List[torch.Tensor] = []
        self.recent_query_descriptors: List[torch.Tensor] = []
        self.segment_start = 0
        self.last_archive_frame = -1
        self.last_recovery_frame = -1
        self.last_attempt_frame = -1
        self.next_archive_id = 0
        self.events: List[Dict[str, float | int | bool]] = []
        self.stats: Dict[str, float | int] = {
            "archive_count": 0,
            "retrieval_attempts": 0,
            "retrieval_successes": 0,
            "retrieval_redundant_skips": 0,
            "retrieval_threshold_rejects": 0,
            "retrieval_gap_rejects": 0,
            "retrieval_sequence_rejects": 0,
            "evicted_archives": 0,
            "best_similarity_sum": 0.0,
            "best_gap_sum": 0.0,
            "best_sequence_similarity_sum": 0.0,
            "verified_similarity_sum": 0.0,
            "verified_gap_sum": 0.0,
            "verified_sequence_similarity_sum": 0.0,
            "geometry_verification_rollouts": 0,
            "geometry_verification_accepts": 0,
            "geometry_verification_rejects": 0,
            "geometry_verification_geo_gain_sum": 0.0,
            "geometry_verification_conf_delta_sum": 0.0,
        }

    @staticmethod
    def describe_global_feat(global_img_feat: torch.Tensor) -> torch.Tensor:
        return _flatten_descriptor(global_img_feat)

    @staticmethod
    def describe_state(state_feat: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        state_desc = _flatten_descriptor(state_feat)
        mem_desc = _flatten_descriptor(mem)
        combo = torch.cat([state_desc, mem_desc], dim=-1)
        return F.normalize(combo, dim=-1)

    def observe(self, descriptor: torch.Tensor) -> None:
        self.segment_descriptors.append(descriptor)
        if self.config.sequence_consistency_window > 0:
            self.recent_query_descriptors.append(descriptor)
            if len(self.recent_query_descriptors) > self.config.sequence_consistency_window:
                self.recent_query_descriptors = self.recent_query_descriptors[
                    -self.config.sequence_consistency_window :
                ]

    def should_archive(self, frame_idx: int) -> bool:
        if not self.config.enable_archive:
            return False
        active_length = frame_idx - self.segment_start + 1
        return active_length >= self.config.archive_interval

    def _segment_descriptor(self) -> torch.Tensor:
        if not self.segment_descriptors:
            raise RuntimeError("Cannot archive without observed descriptors.")
        return F.normalize(torch.stack(self.segment_descriptors, dim=0).mean(dim=0), dim=-1)

    def _segment_tail_descriptors(self) -> torch.Tensor:
        window = self.config.sequence_consistency_window
        if window <= 0 or not self.segment_descriptors:
            return torch.empty(0)
        tail = self.segment_descriptors[-window:]
        return torch.stack(tail, dim=0)

    def _sequence_similarity(self, tail_descriptors: torch.Tensor) -> float:
        window = self.config.sequence_consistency_window
        if window <= 1 or tail_descriptors.numel() == 0 or not self.recent_query_descriptors:
            return 0.0
        query_tail = self.recent_query_descriptors[-window:]
        length = min(len(query_tail), int(tail_descriptors.shape[0]), window)
        if length <= 1:
            return 0.0
        query_stack = torch.stack(query_tail[-length:], dim=0)
        archive_stack = tail_descriptors[-length:]
        similarities = F.cosine_similarity(query_stack, archive_stack, dim=-1)
        return similarities.mean().item()

    def propose_recovery(
        self,
        frame_idx: int,
        state_args: StateTuple,
        query_descriptor: torch.Tensor,
    ) -> List[RecoveryProposal]:
        if not self.config.enable_retrieval or not self.archive_bank:
            return []
        if frame_idx - self.last_recovery_frame < self.config.retrieval_cooldown:
            return []
        if frame_idx - self.last_attempt_frame < self.config.retrieval_attempt_cooldown:
            return []

        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        current_state_desc = self.describe_state(state_feat, mem)

        candidates = []
        latest_archive_id = self.archive_bank[-1].archive_id if self.archive_bank else -1
        for entry in self.archive_bank:
            if frame_idx - entry.frame_idx < self.config.min_frames_before_retrieve:
                continue
            query_sim = F.cosine_similarity(query_descriptor, entry.descriptor).mean().item()
            state_sim = F.cosine_similarity(current_state_desc, entry.state_descriptor).mean().item()
            sequence_sim = self._sequence_similarity(entry.tail_descriptors)
            candidates.append((query_sim, state_sim, sequence_sim, entry))

        if not candidates:
            return []

        self.stats["retrieval_attempts"] += 1
        self.last_attempt_frame = frame_idx
        candidates.sort(key=lambda item: (item[2], item[0]), reverse=True)
        top_candidates = candidates[: max(self.config.retrieval_topk, 1)]
        top_query_sim, top_state_sim, top_sequence_sim, best_entry = top_candidates[0]
        top_gap = top_query_sim - top_state_sim
        self.stats["best_similarity_sum"] += top_query_sim
        self.stats["best_gap_sum"] += top_gap
        self.stats["best_sequence_similarity_sum"] += top_sequence_sim

        event = {
            "frame_idx": frame_idx,
            "archive_id": best_entry.archive_id,
            "query_similarity": top_query_sim,
            "state_similarity": top_state_sim,
            "sequence_similarity": top_sequence_sim,
            "query_state_gap": top_gap,
            "recovered": False,
            "reason": "below_query_threshold",
        }

        selected_candidates: List[RecoveryProposal] = []
        for rank, (query_sim, state_sim, sequence_sim, entry) in enumerate(top_candidates, start=1):
            gap = query_sim - state_sim
            if query_sim < self.config.retrieval_similarity_thresh:
                self.stats["retrieval_threshold_rejects"] += 1
                event.update(
                    {
                        "archive_id": entry.archive_id,
                        "query_similarity": query_sim,
                        "state_similarity": state_sim,
                        "sequence_similarity": sequence_sim,
                        "query_state_gap": gap,
                        "candidate_rank": rank,
                        "reason": "below_query_threshold",
                    }
                )
                break
            if state_sim < self.config.verification_similarity_thresh:
                self.stats["retrieval_threshold_rejects"] += 1
                event.update(
                    {
                        "archive_id": entry.archive_id,
                        "query_similarity": query_sim,
                        "state_similarity": state_sim,
                        "sequence_similarity": sequence_sim,
                        "query_state_gap": gap,
                        "candidate_rank": rank,
                        "reason": "below_state_threshold",
                    }
                )
                continue
            if state_sim > self.config.max_state_similarity_for_recover:
                self.stats["retrieval_redundant_skips"] += 1
                event.update(
                    {
                        "archive_id": entry.archive_id,
                        "query_similarity": query_sim,
                        "state_similarity": state_sim,
                        "sequence_similarity": sequence_sim,
                        "query_state_gap": gap,
                        "candidate_rank": rank,
                        "reason": "redundant_state_match",
                    }
                )
                continue
            if sequence_sim < self.config.sequence_similarity_thresh:
                self.stats["retrieval_sequence_rejects"] += 1
                event.update(
                    {
                        "archive_id": entry.archive_id,
                        "query_similarity": query_sim,
                        "state_similarity": state_sim,
                        "sequence_similarity": sequence_sim,
                        "query_state_gap": gap,
                        "candidate_rank": rank,
                        "reason": "below_sequence_threshold",
                    }
                )
                continue
            if gap < self.config.query_state_gap_thresh:
                self.stats["retrieval_gap_rejects"] += 1
                event.update(
                    {
                        "archive_id": entry.archive_id,
                        "query_similarity": query_sim,
                        "state_similarity": state_sim,
                        "sequence_similarity": sequence_sim,
                        "query_state_gap": gap,
                        "candidate_rank": rank,
                        "reason": "below_gap_threshold",
                    }
                )
                continue
            alpha = self.config.recovery_alpha
            is_latest_archive = entry.archive_id == latest_archive_id
            recovered_state = alpha * entry.state_feat.to(state_feat.device) + (1.0 - alpha) * state_feat
            recovered_mem = alpha * entry.mem.to(mem.device) + (1.0 - alpha) * mem
            selected_candidates.append(
                RecoveryProposal(
                    archive_id=entry.archive_id,
                    archive_frame_idx=entry.frame_idx,
                    candidate_rank=rank,
                    query_similarity=query_sim,
                    state_similarity=state_sim,
                    sequence_similarity=sequence_sim,
                    query_state_gap=gap,
                    recovery_alpha=alpha,
                    is_latest_archive=is_latest_archive,
                    state_args=(recovered_state, state_pos, init_state_feat, recovered_mem, init_mem),
                )
            )

        if not selected_candidates:
            self.events.append(event)
            return []

        selected_candidates.sort(
            key=lambda proposal: (
                proposal.sequence_similarity,
                proposal.query_state_gap,
                proposal.query_similarity,
            ),
            reverse=True,
        )
        return selected_candidates

    def record_geometry_verification(
        self,
        *,
        baseline_geo_rmse: float,
        baseline_conf: float,
        candidate_geo_rmse: float,
        candidate_conf: float,
        accepted: bool,
    ) -> None:
        self.stats["geometry_verification_rollouts"] += 1
        self.stats["geometry_verification_geo_gain_sum"] += baseline_geo_rmse - candidate_geo_rmse
        self.stats["geometry_verification_conf_delta_sum"] += candidate_conf - baseline_conf
        if accepted:
            self.stats["geometry_verification_accepts"] += 1
        else:
            self.stats["geometry_verification_rejects"] += 1

    def accept_recovery(
        self,
        frame_idx: int,
        proposal: RecoveryProposal,
        *,
        baseline_geo_rmse: float,
        baseline_conf: float,
        candidate_geo_rmse: float,
        candidate_conf: float,
    ) -> None:
        self.last_recovery_frame = frame_idx
        self.stats["retrieval_successes"] += 1
        self.stats["verified_similarity_sum"] += proposal.state_similarity
        self.stats["verified_gap_sum"] += proposal.query_state_gap
        self.stats["verified_sequence_similarity_sum"] += proposal.sequence_similarity
        self.record_geometry_verification(
            baseline_geo_rmse=baseline_geo_rmse,
            baseline_conf=baseline_conf,
            candidate_geo_rmse=candidate_geo_rmse,
            candidate_conf=candidate_conf,
            accepted=True,
        )
        self.events.append(
            {
                "frame_idx": frame_idx,
                "archive_id": proposal.archive_id,
                "archive_frame_idx": proposal.archive_frame_idx,
                "candidate_rank": proposal.candidate_rank,
                "query_similarity": proposal.query_similarity,
                "state_similarity": proposal.state_similarity,
                "sequence_similarity": proposal.sequence_similarity,
                "query_state_gap": proposal.query_state_gap,
                "recovery_alpha": proposal.recovery_alpha,
                "is_latest_archive": proposal.is_latest_archive,
                "baseline_geo_rmse": baseline_geo_rmse,
                "candidate_geo_rmse": candidate_geo_rmse,
                "baseline_conf": baseline_conf,
                "candidate_conf": candidate_conf,
                "recovered": True,
                "reason": "recovered_after_geometry_verification",
            }
        )

    def accept_recovery_without_geometry(
        self,
        frame_idx: int,
        proposal: RecoveryProposal,
    ) -> None:
        self.last_recovery_frame = frame_idx
        self.stats["retrieval_successes"] += 1
        self.stats["verified_similarity_sum"] += proposal.state_similarity
        self.stats["verified_gap_sum"] += proposal.query_state_gap
        self.stats["verified_sequence_similarity_sum"] += proposal.sequence_similarity
        self.events.append(
            {
                "frame_idx": frame_idx,
                "archive_id": proposal.archive_id,
                "archive_frame_idx": proposal.archive_frame_idx,
                "candidate_rank": proposal.candidate_rank,
                "query_similarity": proposal.query_similarity,
                "state_similarity": proposal.state_similarity,
                "sequence_similarity": proposal.sequence_similarity,
                "query_state_gap": proposal.query_state_gap,
                "recovery_alpha": proposal.recovery_alpha,
                "is_latest_archive": proposal.is_latest_archive,
                "recovered": True,
                "reason": "recovered_without_geometry_verification",
            }
        )

    def reject_recovery(
        self,
        frame_idx: int,
        proposal: RecoveryProposal | None,
        *,
        baseline_geo_rmse: float,
        baseline_conf: float,
        candidate_geo_rmse: float | None = None,
        candidate_conf: float | None = None,
        reason: str = "rejected_by_geometry_verification",
    ) -> None:
        if proposal is not None and candidate_geo_rmse is not None and candidate_conf is not None:
            self.record_geometry_verification(
                baseline_geo_rmse=baseline_geo_rmse,
                baseline_conf=baseline_conf,
                candidate_geo_rmse=candidate_geo_rmse,
                candidate_conf=candidate_conf,
                accepted=False,
            )
        event = {
            "frame_idx": frame_idx,
            "baseline_geo_rmse": baseline_geo_rmse,
            "baseline_conf": baseline_conf,
            "recovered": False,
            "reason": reason,
        }
        if proposal is not None:
            event.update(
                {
                    "archive_id": proposal.archive_id,
                    "archive_frame_idx": proposal.archive_frame_idx,
                    "candidate_rank": proposal.candidate_rank,
                    "query_similarity": proposal.query_similarity,
                    "state_similarity": proposal.state_similarity,
                    "sequence_similarity": proposal.sequence_similarity,
                    "query_state_gap": proposal.query_state_gap,
                    "recovery_alpha": proposal.recovery_alpha,
                    "is_latest_archive": proposal.is_latest_archive,
                }
            )
        if candidate_geo_rmse is not None:
            event["candidate_geo_rmse"] = candidate_geo_rmse
        if candidate_conf is not None:
            event["candidate_conf"] = candidate_conf
        self.events.append(event)

    def archive(self, frame_idx: int, state_args: StateTuple) -> None:
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        entry = ArchiveEntry(
            archive_id=self.next_archive_id,
            frame_idx=frame_idx,
            segment_start=self.segment_start,
            segment_length=frame_idx - self.segment_start + 1,
            descriptor=self._segment_descriptor(),
            tail_descriptors=self._segment_tail_descriptors(),
            state_descriptor=self.describe_state(state_feat, mem),
            state_feat=state_feat.detach().cpu(),
            state_pos=state_pos.detach().cpu(),
            init_state_feat=init_state_feat.detach().cpu(),
            mem=mem.detach().cpu(),
            init_mem=init_mem.detach().cpu(),
        )
        self.next_archive_id += 1
        self.archive_bank.append(entry)
        self.stats["archive_count"] += 1
        self.last_archive_frame = frame_idx
        self.events.append(
            {
                "frame_idx": frame_idx,
                "archive_id": entry.archive_id,
                "segment_length": entry.segment_length,
                "archived": True,
            }
        )

        while len(self.archive_bank) > self.config.max_archives:
            self.archive_bank.pop(0)
            self.stats["evicted_archives"] += 1

        self.segment_start = frame_idx + 1
        self.segment_descriptors = []

    def finalize(self) -> Dict[str, float | int]:
        attempts = int(self.stats["retrieval_attempts"])
        successes = int(self.stats["retrieval_successes"])
        summary = dict(self.stats)
        summary["archive_bank_size"] = len(self.archive_bank)
        summary["avg_best_similarity"] = (
            float(self.stats["best_similarity_sum"]) / attempts if attempts else 0.0
        )
        summary["avg_best_gap"] = float(self.stats["best_gap_sum"]) / attempts if attempts else 0.0
        summary["avg_best_sequence_similarity"] = (
            float(self.stats["best_sequence_similarity_sum"]) / attempts if attempts else 0.0
        )
        summary["avg_verified_similarity"] = (
            float(self.stats["verified_similarity_sum"]) / successes if successes else 0.0
        )
        summary["avg_verified_gap"] = (
            float(self.stats["verified_gap_sum"]) / successes if successes else 0.0
        )
        summary["avg_verified_sequence_similarity"] = (
            float(self.stats["verified_sequence_similarity_sum"]) / successes if successes else 0.0
        )
        verification_rollouts = int(self.stats["geometry_verification_rollouts"])
        summary["avg_geometry_verification_geo_gain"] = (
            float(self.stats["geometry_verification_geo_gain_sum"]) / verification_rollouts
            if verification_rollouts
            else 0.0
        )
        summary["avg_geometry_verification_conf_delta"] = (
            float(self.stats["geometry_verification_conf_delta_sum"]) / verification_rollouts
            if verification_rollouts
            else 0.0
        )
        return summary
