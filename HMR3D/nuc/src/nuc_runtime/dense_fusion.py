from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SampledPointBatch:
    xyz: np.ndarray
    rgb: np.ndarray
    frame: np.ndarray
    semantic_label: np.ndarray
    semantic_conf: np.ndarray


def empty_point_batch() -> dict[str, np.ndarray]:
    return {
        "xyz": np.zeros((0, 3), dtype=np.float32),
        "rgb": np.zeros((0, 3), dtype=np.uint8),
        "frame": np.zeros((0,), dtype=np.int32),
        "semantic_label": np.zeros((0,), dtype=np.int32) - 1,
        "semantic_conf": np.zeros((0,), dtype=np.float32),
        "observations": np.zeros((0,), dtype=np.int32),
    }


class VoxelFusionMap:
    """Lightweight CPU voxel fusion for live RGB-D style point accumulation.

    This is intentionally simple and predictable on Jetson: each voxel stores a
    running mean of xyz/rgb plus a semantic label chosen by accumulated score.
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.08,
        max_voxels: int = 500_000,
        min_observations: int = 1,
    ) -> None:
        self.voxel_size = max(1e-4, float(voxel_size))
        self.max_voxels = max(1, int(max_voxels))
        self.min_observations = max(1, int(min_observations))
        self._voxels: OrderedDict[tuple[int, int, int], dict[str, Any]] = OrderedDict()

    def update(self, points: dict[str, np.ndarray]) -> None:
        xyz = np.asarray(points.get("xyz", np.zeros((0, 3))), dtype=np.float32)
        if xyz.size == 0:
            return
        rgb = np.asarray(points.get("rgb", np.zeros((xyz.shape[0], 3))), dtype=np.float32)
        frame = np.asarray(points.get("frame", np.zeros((xyz.shape[0],))), dtype=np.int32)
        labels = np.asarray(points.get("semantic_label", np.full((xyz.shape[0],), -1)), dtype=np.int32)
        conf = np.asarray(points.get("semantic_conf", np.zeros((xyz.shape[0],))), dtype=np.float32)
        keys = np.floor(xyz / self.voxel_size).astype(np.int32)
        finite = np.isfinite(xyz).all(axis=1)

        for idx in np.flatnonzero(finite):
            key = (int(keys[idx, 0]), int(keys[idx, 1]), int(keys[idx, 2]))
            item = self._voxels.get(key)
            if item is None:
                item = {
                    "xyz_sum": np.zeros(3, dtype=np.float64),
                    "rgb_sum": np.zeros(3, dtype=np.float64),
                    "count": 0,
                    "last_frame": int(frame[idx]),
                    "semantic_scores": {},
                }
                self._voxels[key] = item
            item["xyz_sum"] += xyz[idx].astype(np.float64)
            item["rgb_sum"] += rgb[idx].astype(np.float64)
            item["count"] += 1
            item["last_frame"] = max(int(item["last_frame"]), int(frame[idx]))
            label = int(labels[idx]) if idx < labels.shape[0] else -1
            score = float(conf[idx]) if idx < conf.shape[0] else 0.0
            if label >= 0 and score > 0.0:
                scores = item["semantic_scores"]
                scores[label] = float(scores.get(label, 0.0)) + score
            self._voxels.move_to_end(key)
        self._prune()

    def snapshot(self) -> dict[str, np.ndarray]:
        kept = [item for item in self._voxels.values() if int(item["count"]) >= self.min_observations]
        if not kept:
            return empty_point_batch()
        xyz = np.zeros((len(kept), 3), dtype=np.float32)
        rgb = np.zeros((len(kept), 3), dtype=np.uint8)
        frame = np.zeros((len(kept),), dtype=np.int32)
        observations = np.zeros((len(kept),), dtype=np.int32)
        semantic_label = np.zeros((len(kept),), dtype=np.int32) - 1
        semantic_conf = np.zeros((len(kept),), dtype=np.float32)
        for idx, item in enumerate(kept):
            count = max(1, int(item["count"]))
            xyz[idx] = (item["xyz_sum"] / count).astype(np.float32)
            rgb[idx] = np.clip(item["rgb_sum"] / count, 0, 255).astype(np.uint8)
            frame[idx] = int(item["last_frame"])
            observations[idx] = count
            scores = item["semantic_scores"]
            if scores:
                label, score = max(scores.items(), key=lambda kv: kv[1])
                semantic_label[idx] = int(label)
                semantic_conf[idx] = float(score / max(1, count))
        return {
            "xyz": xyz,
            "rgb": rgb,
            "frame": frame,
            "semantic_label": semantic_label,
            "semantic_conf": semantic_conf,
            "observations": observations,
        }

    def _prune(self) -> None:
        overflow = len(self._voxels) - self.max_voxels
        if overflow <= 0:
            return
        # OrderedDict is insertion/update ordered, so this removes old voxels
        # first. That keeps the active map bounded for long live runs.
        for _ in range(overflow):
            if not self._voxels:
                break
            self._voxels.popitem(last=False)
