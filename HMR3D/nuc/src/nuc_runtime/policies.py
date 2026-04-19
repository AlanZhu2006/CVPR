from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from nuc_runtime.config import MemoryConfig
from nuc_runtime.descriptors import cosine_similarity, normalize_vector
from nuc_runtime.models import ArchivedSubmap, SceneSummary, TrackingOutput, pose_translation


def rotation_angle_deg(left_pose: np.ndarray, right_pose: np.ndarray) -> float:
    relative = left_pose[:3, :3].T @ right_pose[:3, :3]
    trace = float(np.trace(relative))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


@dataclass
class CandidateScore:
    archived: ArchivedSubmap
    similarity: float
    match_count: int
    translation_delta: float
    rotation_delta_deg: float
    pose_gate_passed: bool
    support_merge_passed: bool

    @property
    def fusion_weight(self) -> float:
        match_term = min(1.0, self.match_count / 200.0)
        pose_term = 1.0 / (1.0 + 0.05 * self.translation_delta + 0.02 * self.rotation_delta_deg)
        return max(1e-6, self.similarity * (0.5 + 0.5 * match_term) * pose_term)


class WritePolicy:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def passes(self, active) -> bool:
        if not self.config.enable_v2_write_policy:
            return True
        if active.keyframe_count() < self.config.archive_min_keyframes:
            return False
        mean_match_count = float(np.mean([item.match_count for item in active.keyframes]))
        descriptor_score = float(np.mean([
            cosine_similarity(item.descriptor, active.descriptor()) for item in active.keyframes
        ]))
        return (
            mean_match_count >= self.config.archive_min_mean_match_count
            and descriptor_score >= self.config.archive_min_mean_descriptor_score
        )

    def select_anchor(self, active):
        if not self.config.enable_v2_write_policy or active.keyframe_count() <= 1:
            return active.keyframes[-1]
        active_descriptor = active.descriptor()
        scored = []
        for item in active.keyframes:
            descriptor_score = cosine_similarity(item.descriptor, active_descriptor)
            score = item.match_count + 50.0 * descriptor_score
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[: max(1, self.config.anchor_topk)][0][1]


class RetrievalPolicy:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def build_scene_summaries(self, bank: list[ArchivedSubmap]) -> list[SceneSummary]:
        if not bank:
            return []
        if not self.config.enable_hierarchical_bank:
            return [
                self._make_scene_summary(scene_id=i, members=[item])
                for i, item in enumerate(bank)
            ]

        groups: list[list[ArchivedSubmap]] = []
        for item in bank:
            if not groups:
                groups.append([item])
                continue
            last_group = groups[-1]
            last_centroid = np.mean([member.centroid for member in last_group], axis=0)
            distance = float(np.linalg.norm(item.centroid - last_centroid))
            if (
                len(last_group) < self.config.scene_summary_max_entries
                and distance <= self.config.scene_summary_distance_threshold
            ):
                last_group.append(item)
            else:
                groups.append([item])

        return [
            self._make_scene_summary(scene_id=idx, members=members)
            for idx, members in enumerate(groups)
        ]

    def route(self, output: TrackingOutput, bank: list[ArchivedSubmap], scene_summaries: list[SceneSummary]) -> list[ArchivedSubmap]:
        if not bank:
            return []
        if not self.config.enable_hierarchical_bank or not scene_summaries:
            ranked = sorted(
                bank,
                key=lambda archived: cosine_similarity(output.descriptor, archived.descriptor),
                reverse=True,
            )
            return ranked[: self.config.merge_topk]

        ranked_summaries = sorted(
            scene_summaries,
            key=lambda scene: cosine_similarity(output.descriptor, scene.descriptor),
            reverse=True,
        )[: self.config.scene_topk]

        allowed_ids = {member_id for scene in ranked_summaries for member_id in scene.member_submap_ids}
        candidates = [item for item in bank if item.submap_id in allowed_ids]
        candidates.sort(key=lambda archived: cosine_similarity(output.descriptor, archived.descriptor), reverse=True)
        return candidates[: self.config.merge_topk]

    def _make_scene_summary(self, scene_id: int, members: list[ArchivedSubmap]) -> SceneSummary:
        descriptors = np.vstack([member.descriptor for member in members]).astype(np.float32)
        descriptor = normalize_vector(descriptors.mean(axis=0))
        centroids = np.vstack([member.centroid for member in members]).astype(np.float32)
        bbox_min = np.min(np.vstack([member.bbox_min for member in members]), axis=0)
        bbox_max = np.max(np.vstack([member.bbox_max for member in members]), axis=0)
        return SceneSummary(
            scene_id=scene_id,
            member_submap_ids=[member.submap_id for member in members],
            descriptor=descriptor,
            centroid=centroids.mean(axis=0),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            summary={"member_count": len(members)},
        )


class VerifyPolicy:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def score_candidates(self, output: TrackingOutput, candidates: list[ArchivedSubmap]) -> list[CandidateScore]:
        scored: list[CandidateScore] = []
        for archived in candidates:
            similarity = cosine_similarity(output.descriptor, archived.descriptor)
            if similarity < self.config.retrieve_similarity_threshold:
                continue
            match_count = self.good_match_count(output.orb_descriptors, archived.anchor_orb_descriptors)
            if match_count < self.config.geo_verify_min_matches:
                continue
            translation_delta = float(np.linalg.norm(
                pose_translation(output.pose) - pose_translation(archived.anchor_pose)
            )) if archived.anchor_pose is not None else 0.0
            rotation_delta_deg = rotation_angle_deg(output.pose, archived.anchor_pose) if archived.anchor_pose is not None else 0.0
            pose_gate_passed = True
            if self.config.enable_pose_anchor_gate:
                pose_gate_passed = (
                    translation_delta <= self.config.pose_anchor_translation_threshold
                    and rotation_delta_deg <= self.config.pose_anchor_rotation_threshold_deg
                )
            support_merge_passed = (
                similarity >= self.config.merge_support_similarity_floor
                and translation_delta <= (
                    self.config.pose_anchor_translation_threshold
                    * self.config.merge_support_translation_ratio
                )
                and rotation_delta_deg <= (
                    self.config.pose_anchor_rotation_threshold_deg
                    * self.config.merge_support_rotation_ratio
                )
            )
            scored.append(
                CandidateScore(
                    archived=archived,
                    similarity=similarity,
                    match_count=match_count,
                    translation_delta=translation_delta,
                    rotation_delta_deg=rotation_delta_deg,
                    pose_gate_passed=pose_gate_passed,
                    support_merge_passed=support_merge_passed,
                )
            )
        scored.sort(key=lambda item: (item.pose_gate_passed, item.fusion_weight), reverse=True)
        return scored[: self.config.merge_topk]

    def good_match_count(self, left_desc: np.ndarray | None, right_desc: np.ndarray | None) -> int:
        if left_desc is None or right_desc is None:
            return 0
        raw_matches = self.matcher.knnMatch(left_desc, right_desc, k=2)
        good = 0
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good += 1
        return good


class RecoverPolicy:
    def __init__(self, config: MemoryConfig):
        self.config = config

    def merge_candidates(self, scored: list[CandidateScore]) -> tuple[list[CandidateScore], np.ndarray | None]:
        accepted = [item for item in scored if item.pose_gate_passed]
        if not accepted:
            return [], None
        primary = accepted[0]
        if not self.config.enable_multi_candidate_merge:
            descriptor = primary.archived.descriptor.copy()
            return [primary], descriptor

        merge_pool = [primary]
        for item in scored:
            if item.archived.submap_id == primary.archived.submap_id:
                continue
            if item.pose_gate_passed or item.support_merge_passed:
                merge_pool.append(item)

        if len(merge_pool) < self.config.merge_min_candidates:
            descriptor = primary.archived.descriptor.copy()
            return [primary], descriptor

        weights = np.array([item.fusion_weight for item in merge_pool], dtype=np.float32)
        weights /= np.sum(weights)
        merged = np.zeros_like(primary.archived.descriptor, dtype=np.float32)
        for weight, item in zip(weights, merge_pool):
            merged += weight * item.archived.descriptor
        return merge_pool, normalize_vector(merged)

    def apply_local_adapt(self, active, merged_descriptor: np.ndarray) -> np.ndarray:
        if not self.config.enable_local_adapt:
            return merged_descriptor
        base = active.descriptor()
        gain = self.config.local_adapt_descriptor_gain
        active.local_adapt_steps += 1
        return normalize_vector((1.0 - gain) * base + gain * merged_descriptor)
