from __future__ import annotations

from collections import Counter, deque

import cv2
import numpy as np

from nuc_runtime.config import MemoryConfig
from nuc_runtime.descriptors import cosine_similarity
from nuc_runtime.models import (
    ActiveSubmap,
    ArchivedSubmap,
    EventRecord,
    KeyframeRecord,
    TrackingOutput,
    pose_translation,
)


class MemoryRouter:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.short_window: deque[KeyframeRecord] = deque(maxlen=config.short_window_size)
        self.active: ActiveSubmap | None = None
        self.bank: list[ArchivedSubmap] = []
        self._next_submap_id = 0
        self.stats = Counter()

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

        return events, self.snapshot()

    def finalize(self) -> dict:
        return {
            "stats": dict(self.stats),
            "active": self._active_to_dict(),
            "bank": [self._archived_to_dict(item) for item in self.bank],
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
            return "max_keyframes"

        age = output.frame_idx - self.active.created_frame_idx
        if age >= self.config.active_max_age:
            return "max_age"

        centroid = self.active.centroid()
        current_t = pose_translation(output.pose)
        if float(np.linalg.norm(current_t - centroid)) >= self.config.active_max_distance:
            return "pose_distance"

        similarity = cosine_similarity(output.descriptor, self.active.descriptor())
        if similarity < self.config.active_similarity_floor and self.active.keyframe_count() >= 3:
            return "active_similarity_drop"

        return None

    def _archive_active(self, reason: str) -> ArchivedSubmap:
        assert self.active is not None
        archived = self._build_archived_submap(self.active, reason)
        self.bank.append(archived)
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

        ranked = sorted(
            (
                (
                    cosine_similarity(output.descriptor, archived.descriptor),
                    archived,
                )
                for archived in self.bank
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        top_ranked = ranked[: self.config.retrieve_topk]
        if not top_ranked:
            return None, None

        best_similarity, best_submap = top_ranked[0]
        if best_similarity < self.config.retrieve_similarity_threshold:
            return None, None

        verified_submap = None
        verified_matches = 0
        for similarity, candidate in top_ranked:
            match_count = self._good_match_count(output.orb_descriptors, candidate.anchor_orb_descriptors)
            if match_count >= self.config.geo_verify_min_matches:
                verified_submap = candidate
                verified_matches = match_count
                best_similarity = similarity
                break

        retrieve_event = EventRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            event_type="retrieved",
            payload={
                "query_active_id": self.active.submap_id,
                "candidate_id": best_submap.submap_id,
                "similarity": round(best_similarity, 4),
                "verified": verified_submap is not None,
                "verified_match_count": verified_matches,
            },
        )
        self.stats["retrieve_hits"] += 1

        if verified_submap is None:
            self.stats["retrieve_geo_rejects"] += 1
            return retrieve_event, None

        if not self.config.enable_recover:
            self.stats["recover_skipped"] += 1
            return retrieve_event, None

        if verified_submap.submap_id in self.active.recovered_from:
            return retrieve_event, None

        self.active.recovered_from.append(verified_submap.submap_id)
        self.active.injected_descriptors.append(verified_submap.descriptor)
        self.active.last_recover_frame_idx = output.frame_idx
        self.stats["recoveries"] += 1

        recover_event = EventRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            event_type="recovered",
            payload={
                "active_id": self.active.submap_id,
                "from_submap_id": verified_submap.submap_id,
                "verified_match_count": verified_matches,
            },
        )
        return retrieve_event, recover_event

    def _build_archived_submap(self, active: ActiveSubmap, reason: str) -> ArchivedSubmap:
        descriptors = np.vstack([item.descriptor for item in active.keyframes]).astype(np.float32)
        descriptor = descriptors.mean(axis=0)
        descriptor /= np.linalg.norm(descriptor) + 1e-8

        translations = np.vstack([pose_translation(item.pose) for item in active.keyframes])
        bbox_min = translations.min(axis=0)
        bbox_max = translations.max(axis=0)
        centroid = translations.mean(axis=0)
        anchor = active.keyframes[-1]

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
            summary={
                "reason": reason,
                "keyframe_count": len(active.keyframes),
                "recovered_from": list(active.recovered_from),
            },
        )

    def _to_keyframe(self, output: TrackingOutput) -> KeyframeRecord:
        return KeyframeRecord(
            frame_idx=output.frame_idx,
            timestamp_sec=output.timestamp_sec,
            pose=output.pose.copy(),
            descriptor=output.descriptor.copy(),
            orb_descriptors=None if output.orb_descriptors is None else output.orb_descriptors.copy(),
            image_path=output.image_path,
            keypoint_count=output.keypoint_count,
            match_count=output.match_count,
            inlier_count=output.inlier_count,
            pixel_motion=output.pixel_motion,
        )

    def _allocate_submap_id(self) -> int:
        submap_id = self._next_submap_id
        self._next_submap_id += 1
        return submap_id

    def _good_match_count(
        self,
        left_desc: np.ndarray | None,
        right_desc: np.ndarray | None,
    ) -> int:
        if left_desc is None or right_desc is None:
            return 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_matches = matcher.knnMatch(left_desc, right_desc, k=2)
        good = 0
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good += 1
        return good

    def _active_to_dict(self) -> dict | None:
        if self.active is None:
            return None
        return {
            "submap_id": self.active.submap_id,
            "created_frame_idx": self.active.created_frame_idx,
            "keyframe_count": self.active.keyframe_count(),
            "recovered_from": list(self.active.recovered_from),
            "frame_indices": [item.frame_idx for item in self.active.keyframes],
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
            "summary": dict(item.summary),
        }
