from __future__ import annotations

from collections import Counter, deque
from pathlib import Path

import numpy as np

from nuc_runtime.config import MemoryConfig
from nuc_runtime.descriptors import cosine_similarity
from nuc_runtime.gaussian_builder import IncrementalGaussianBuilder
from nuc_runtime.models import (
    ActiveSubmap,
    ArchivedSubmap,
    EventRecord,
    KeyframeRecord,
    SceneSummary,
    TrackingOutput,
    pose_translation,
)
from nuc_runtime.policies import RecoverPolicy, RetrievalPolicy, VerifyPolicy, WritePolicy


class MemoryRouter:
    def __init__(self, config: MemoryConfig, output_dir: str | Path | None = None):
        self.config = config
        self.short_window: deque[KeyframeRecord] = deque(maxlen=config.short_window_size)
        self.active: ActiveSubmap | None = None
        self.bank: list[ArchivedSubmap] = []
        self.scene_summaries: list[SceneSummary] = []
        self._next_submap_id = 0
        self.stats = Counter()
        self.write_policy = WritePolicy(config)
        self.retrieval_policy = RetrievalPolicy(config)
        self.verify_policy = VerifyPolicy(config)
        self.recover_policy = RecoverPolicy(config)
        self.gaussian_builder = None
        if config.enable_incremental_gaussian and output_dir is not None:
            self.gaussian_builder = IncrementalGaussianBuilder(Path(output_dir) / "gaussian_bank", config)

    def process(self, output: TrackingOutput) -> tuple[list[EventRecord], dict]:
        events: list[EventRecord] = []
        if not output.is_keyframe:
            return events, self.snapshot()

        keyframe = self._to_keyframe(output)
        self.short_window.append(keyframe)
        self.stats["keyframes"] += 1

        if self.active is None:
            self.active = ActiveSubmap(
                submap_id=self._allocate_submap_id(),
                created_frame_idx=output.frame_idx,
                keyframes=[keyframe],
            )
            self._update_active_gaussian(keyframe)
            events.append(
                EventRecord(
                    frame_idx=output.frame_idx,
                    timestamp_sec=output.timestamp_sec,
                    event_type="active_started",
                    payload={"active_id": self.active.submap_id, "reason": "bootstrap"},
                )
            )
        else:
            reason = self._archive_reason(output)
            if reason is not None:
                archived = self._archive_active(reason)
                events.append(
                    EventRecord(
                        frame_idx=output.frame_idx,
                        timestamp_sec=output.timestamp_sec,
                        event_type="archived",
                        payload={
                            "archived_id": archived.submap_id,
                            "reason": reason,
                            "bank_size": len(self.bank),
                        },
                    )
                )
                self.active = ActiveSubmap(
                    submap_id=self._allocate_submap_id(),
                    created_frame_idx=output.frame_idx,
                    keyframes=[keyframe],
                )
                self._update_active_gaussian(keyframe)
                events.append(
                    EventRecord(
                        frame_idx=output.frame_idx,
                        timestamp_sec=output.timestamp_sec,
                        event_type="active_started",
                        payload={"active_id": self.active.submap_id, "reason": "post_archive"},
                    )
                )
            else:
                self.active.keyframes.append(keyframe)
                self._update_active_gaussian(keyframe)
                events.append(
                    EventRecord(
                        frame_idx=output.frame_idx,
                        timestamp_sec=output.timestamp_sec,
                        event_type="promoted",
                        payload={
                            "active_id": self.active.submap_id,
                            "active_keyframes": self.active.keyframe_count(),
                        },
                    )
                )

        retrieve_event, recover_event = self._retrieve_and_recover(output)
        if retrieve_event is not None:
            events.append(retrieve_event)
        if recover_event is not None:
            events.append(recover_event)
        optimize_event = self._optimize_active_gaussian()
        if optimize_event is not None:
            events.append(
                EventRecord(
                    frame_idx=output.frame_idx,
                    timestamp_sec=output.timestamp_sec,
                    event_type="gaussian_optimized",
                    payload=optimize_event,
                )
            )

        return events, self.snapshot()

    def finalize(self) -> dict:
        return {
            "stats": dict(self.stats),
            "active": self._active_to_dict(),
            "bank": [self._archived_to_dict(item) for item in self.bank],
            "scene_summaries": [self._scene_to_dict(item) for item in self.scene_summaries],
        }

    def snapshot(self) -> dict:
        return {
            "short_size": len(self.short_window),
            "active_id": None if self.active is None else self.active.submap_id,
            "active_keyframes": 0 if self.active is None else self.active.keyframe_count(),
            "bank_size": len(self.bank),
            "recoveries": int(self.stats.get("recoveries", 0)),
        }

    def _archive_reason(self, output: TrackingOutput) -> str | None:
        assert self.active is not None
        if self.active.keyframe_count() >= self.config.active_max_keyframes:
            if self.write_policy.passes(self.active):
                return "max_keyframes"
            self.stats["archive_write_rejects"] += 1
            self.stats["archive_deferred_max_keyframes"] += 1
            return None

        age = output.frame_idx - self.active.created_frame_idx
        if age >= self.config.active_max_age:
            if self.write_policy.passes(self.active):
                return "max_age"
            self.stats["archive_write_rejects"] += 1
            self.stats["archive_deferred_max_age"] += 1
            return None

        centroid = self.active.centroid()
        current_t = pose_translation(output.pose)
        if float(np.linalg.norm(current_t - centroid)) >= self.config.active_max_distance:
            if self.write_policy.passes(self.active):
                return "pose_distance"
            self.stats["archive_write_rejects"] += 1
            self.stats["archive_deferred_pose_distance"] += 1
            return None

        similarity = cosine_similarity(output.descriptor, self.active.descriptor())
        if similarity < self.config.active_similarity_floor and self.active.keyframe_count() >= 3:
            if self.write_policy.passes(self.active):
                return "active_similarity_drop"
            self.stats["archive_write_rejects"] += 1
            self.stats["archive_deferred_similarity_drop"] += 1
            return None

        return None

    def _archive_active(self, reason: str) -> ArchivedSubmap:
        assert self.active is not None
        archived = self._build_archived_submap(self.active, reason)
        self.bank.append(archived)
        self.scene_summaries = self.retrieval_policy.build_scene_summaries(self.bank)
        self.stats["archives"] += 1
        return archived

    def _retrieve_and_recover(
        self,
        output: TrackingOutput,
    ) -> tuple[EventRecord | None, EventRecord | None]:
        if self.active is None or not self.bank:
            return None, None

        if self.active.last_recover_frame_idx >= 0:
            cooldown = output.frame_idx - self.active.last_recover_frame_idx
            if cooldown < self.config.retrieve_cooldown_frames:
                return None, None

        routed = self.retrieval_policy.route(output, self.bank, self.scene_summaries)
        if not routed:
            return None, None
        self.stats["retrieve_routed_candidates"] += len(routed)
        if self.config.enable_hierarchical_bank:
            self.stats["scene_routing_hits"] += 1

        scored = self.verify_policy.score_candidates(output, routed)
        if not scored:
            self.stats["retrieve_geo_rejects"] += 1
            return None, None

        best = scored[0]

        retrieve_event = EventRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            event_type="retrieved",
            payload={
                "query_active_id": self.active.submap_id,
                "candidate_id": best.archived.submap_id,
                "similarity": round(best.similarity, 4),
                "verified": True,
                "verified_match_count": best.match_count,
                "routed_candidates": len(routed),
                "verified_candidates": len(scored),
            },
        )
        self.stats["retrieve_hits"] += 1

        if not self.config.enable_recover:
            self.stats["recover_skipped"] += 1
            return retrieve_event, None

        accepted, merged_descriptor = self.recover_policy.merge_candidates(scored)
        if not accepted:
            self.stats["recover_pose_anchor_rejects"] += len(scored)
            gated_event = EventRecord(
                frame_idx=output.frame_idx,
                timestamp_sec=output.timestamp_sec,
                event_type="recover_rejected",
                payload={
                    "active_id": self.active.submap_id,
                    "candidate_ids": [item.archived.submap_id for item in scored],
                    "reason": "pose_anchor_gate",
                },
            )
            return retrieve_event, gated_event

        accepted_ids = [item.archived.submap_id for item in accepted]
        if any(candidate_id in self.active.recovered_from for candidate_id in accepted_ids):
            return retrieve_event, None

        if self.config.enable_shadow_recover:
            lead = accepted[0]
            if not self._promote_shadow_candidate(output, lead.archived, lead.similarity, lead.match_count):
                shadow_event = EventRecord(
                    frame_idx=output.frame_idx,
                    timestamp_sec=output.timestamp_sec,
                    event_type="shadow_buffered",
                    payload={
                        "active_id": self.active.submap_id,
                        "candidate_id": lead.archived.submap_id,
                        "similarity": round(lead.similarity, 4),
                        "verified_match_count": lead.match_count,
                    },
                )
                return retrieve_event, shadow_event

        for candidate_id in accepted_ids:
            self.active.recovered_from.append(candidate_id)
        if merged_descriptor is not None:
            adapted_descriptor = self.recover_policy.apply_local_adapt(self.active, merged_descriptor)
            self.active.injected_descriptors.append(adapted_descriptor)
            if self.config.enable_local_adapt:
                self.stats["local_adapt_applied"] += 1
        recovered_handles = [
            item.archived.gaussian_handle for item in accepted
            if item.archived.gaussian_handle is not None and item.archived.gaussian_handle.get("point_count", 0) > 0
        ]
        if recovered_handles:
            self.active.recovered_gaussian_handles.extend(recovered_handles)
            self.stats["gaussian_warmstart_requests"] += len(recovered_handles)
            if self.gaussian_builder is not None:
                summary = self.gaussian_builder.warm_start_submap(self.active.submap_id, recovered_handles)
                self.active.gaussian_handle = summary
                self.stats["gaussian_warmstart_points"] += int(summary.get("last_seed_points_added", 0))
        self.active.last_recover_frame_idx = output.frame_idx
        self.active.shadow_candidate_id = -1
        self.active.shadow_candidate_similarity = 0.0
        self.stats["recoveries"] += 1
        if len(accepted) >= 2:
            self.stats["merge_events"] += 1
            self.stats["merged_candidates_total"] += len(accepted)

        recover_event = EventRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            event_type="recovered",
            payload={
                "active_id": self.active.submap_id,
                "from_submap_ids": accepted_ids,
                "verified_match_count": max(item.match_count for item in accepted),
                "merge_count": len(accepted),
                "gaussian_handles": recovered_handles,
            },
        )
        return retrieve_event, recover_event

    def _promote_shadow_candidate(
        self,
        output: TrackingOutput,
        archived: ArchivedSubmap,
        similarity: float,
        verified_matches: int,
    ) -> bool:
        assert self.active is not None
        if self.active.shadow_candidate_id != archived.submap_id:
            self.active.shadow_candidate_id = archived.submap_id
            self.active.shadow_candidate_similarity = similarity
            self.stats["shadow_buffered"] += 1
            return False
        if verified_matches < self.config.shadow_promote_min_matches:
            self.stats["shadow_rejects"] += 1
            return False
        if similarity < self.config.shadow_similarity_threshold:
            self.stats["shadow_rejects"] += 1
            return False
        self.stats["shadow_promotions"] += 1
        return True

    def _build_archived_submap(self, active: ActiveSubmap, reason: str) -> ArchivedSubmap:
        descriptors = np.vstack([item.descriptor for item in active.keyframes]).astype(np.float32)
        descriptor = descriptors.mean(axis=0)
        descriptor /= np.linalg.norm(descriptor) + 1e-8

        translations = np.vstack([pose_translation(item.pose) for item in active.keyframes])
        bbox_min = translations.min(axis=0)
        bbox_max = translations.max(axis=0)
        centroid = translations.mean(axis=0)
        anchor = self.write_policy.select_anchor(active)
        anchor_descriptor_score = float(cosine_similarity(anchor.descriptor, descriptor))
        gaussian_handle = self._finalize_active_gaussian(active, reason)

        return ArchivedSubmap(
            submap_id=active.submap_id,
            frame_indices=[item.frame_idx for item in active.keyframes],
            descriptor=descriptor.astype(np.float32),
            centroid=centroid.astype(np.float32),
            bbox_min=bbox_min.astype(np.float32),
            bbox_max=bbox_max.astype(np.float32),
            anchor_image_path=anchor.image_path,
            anchor_orb_descriptors=anchor.orb_descriptors,
            anchor_frame_idx=anchor.frame_idx,
            anchor_pose=anchor.pose.copy(),
            anchor_match_count=anchor.match_count,
            anchor_descriptor_score=anchor_descriptor_score,
            gaussian_handle=gaussian_handle,
            summary={
                "reason": reason,
                "keyframe_count": len(active.keyframes),
                "recovered_from": list(active.recovered_from),
                "mean_match_count": float(np.mean([item.match_count for item in active.keyframes])),
                "anchor_descriptor_score": anchor_descriptor_score,
                "gaussian_point_count": 0 if gaussian_handle is None else int(gaussian_handle.get("point_count", 0)),
            },
        )

    def _to_keyframe(self, output: TrackingOutput) -> KeyframeRecord:
        return KeyframeRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            pose=output.pose.copy(),
            descriptor=output.descriptor.copy(),
            orb_descriptors=None if output.orb_descriptors is None else output.orb_descriptors.copy(),
            keypoints_xy=None if output.keypoints_xy is None else output.keypoints_xy.copy(),
            image_path=output.image_path,
            right_image_path=output.right_image_path,
            keypoint_count=output.keypoint_count,
            match_count=output.match_count,
            inlier_count=output.inlier_count,
            pixel_motion=output.pixel_motion,
        )

    def _allocate_submap_id(self) -> int:
        submap_id = self._next_submap_id
        self._next_submap_id += 1
        return submap_id

    def _active_to_dict(self) -> dict | None:
        if self.active is None:
            return None
        return {
            "submap_id": self.active.submap_id,
            "created_frame_idx": self.active.created_frame_idx,
            "keyframe_count": self.active.keyframe_count(),
            "recovered_from": list(self.active.recovered_from),
            "frame_indices": [item.frame_idx for item in self.active.keyframes],
            "local_adapt_steps": self.active.local_adapt_steps,
            "gaussian_handle": None if self.active.gaussian_handle is None else dict(self.active.gaussian_handle),
            "recovered_gaussian_handles": [dict(item) for item in self.active.recovered_gaussian_handles],
        }

    def _archived_to_dict(self, item: ArchivedSubmap) -> dict:
        return {
            "submap_id": item.submap_id,
            "frame_indices": list(item.frame_indices),
            "centroid": item.centroid.tolist(),
            "bbox_min": item.bbox_min.tolist(),
            "bbox_max": item.bbox_max.tolist(),
            "anchor_image_path": item.anchor_image_path,
            "anchor_frame_idx": item.anchor_frame_idx,
            "anchor_match_count": item.anchor_match_count,
            "anchor_descriptor_score": item.anchor_descriptor_score,
            "gaussian_handle": None if item.gaussian_handle is None else dict(item.gaussian_handle),
            "summary": dict(item.summary),
        }

    def _scene_to_dict(self, item: SceneSummary) -> dict:
        return {
            "scene_id": item.scene_id,
            "member_submap_ids": list(item.member_submap_ids),
            "centroid": item.centroid.tolist(),
            "bbox_min": item.bbox_min.tolist(),
            "bbox_max": item.bbox_max.tolist(),
            "summary": dict(item.summary),
        }

    def _update_active_gaussian(self, keyframe: KeyframeRecord) -> None:
        if self.active is None or self.gaussian_builder is None:
            return
        summary = self.gaussian_builder.ingest_keyframe(self.active.submap_id, keyframe)
        self.active.gaussian_handle = summary
        self.stats["gaussian_updates"] += 1
        self.stats["gaussian_points_active"] = int(summary.get("point_count", 0))

    def _optimize_active_gaussian(self) -> dict | None:
        if self.active is None or self.gaussian_builder is None or not self.active.keyframes:
            return None
        summary = self.gaussian_builder.optimize_active_window(self.active.submap_id, self.active.keyframes)
        optimized_points = int(summary.get("optimized_points", 0))
        if optimized_points <= 0:
            return None
        self.active.gaussian_handle = summary
        self.stats["gaussian_optimize_calls"] += 1
        self.stats["gaussian_optimized_points_total"] += optimized_points
        self.stats["gaussian_optimize_steps_total"] += int(summary.get("optimize_steps", 0))
        self.stats["gaussian_points_active"] = int(summary.get("point_count", 0))
        return {
            "active_id": self.active.submap_id,
            "optimized_points": optimized_points,
            "optimize_steps": int(summary.get("optimize_steps", 0)),
        }

    def _finalize_active_gaussian(self, active: ActiveSubmap, reason: str) -> dict | None:
        if self.gaussian_builder is None:
            return None
        handle = self.gaussian_builder.finalize_submap(active.submap_id, reason)
        if handle is not None and handle.get("point_count", 0) > 0:
            self.stats["gaussian_archives"] += 1
            self.stats["gaussian_archived_points_total"] += int(handle["point_count"])
        return handle
