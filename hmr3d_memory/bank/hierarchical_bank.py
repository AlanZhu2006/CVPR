from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

import torch
import torch.nn.functional as F


@dataclass
class SceneNode:
    scene_id: int
    sum_vec: torch.Tensor
    count: int
    member_archive_ids: List[int] = field(default_factory=list)

    @property
    def centroid(self) -> torch.Tensor:
        if self.count <= 0:
            return F.normalize(self.sum_vec, dim=-1)
        return F.normalize(self.sum_vec / float(self.count), dim=-1)


class HierarchicalMemoryBank:
    """Level-2 scene centroids for coarse-to-fine retrieval (HMR3D v2, rule-based)."""

    def __init__(self, max_scenes: int) -> None:
        self.max_scenes = max_scenes
        self.scenes: List[SceneNode] = []
        self._next_scene_id = 0
        self.entry_to_scene: Dict[int, int] = {}

    def register_entry(self, archive_id: int, descriptor: torch.Tensor) -> None:
        desc = descriptor.detach().float().cpu().flatten()
        best_idx = -1
        best_sim = -1.0
        for i, scene in enumerate(self.scenes):
            c = scene.centroid
            sim = float(F.cosine_similarity(desc.unsqueeze(0), c.unsqueeze(0), dim=-1).mean())
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_idx < 0 or not self.scenes:
            self._add_new_scene(archive_id, desc)
            return
        if len(self.scenes) >= self.max_scenes and best_sim < 0.25:
            self._merge_smallest_two()
            self.register_entry(archive_id, descriptor)
            return
        scene = self.scenes[best_idx]
        scene.sum_vec = scene.sum_vec + desc
        scene.count += 1
        scene.member_archive_ids.append(archive_id)
        self.entry_to_scene[archive_id] = scene.scene_id

    def _add_new_scene(self, archive_id: int, desc: torch.Tensor) -> None:
        if len(self.scenes) >= self.max_scenes:
            self._merge_smallest_two()
        sid = self._next_scene_id
        self._next_scene_id += 1
        self.scenes.append(
            SceneNode(scene_id=sid, sum_vec=desc.clone(), count=1, member_archive_ids=[archive_id])
        )
        self.entry_to_scene[archive_id] = sid

    def _merge_smallest_two(self) -> None:
        if len(self.scenes) < 2:
            return
        self.scenes.sort(key=lambda s: len(s.member_archive_ids))
        a, b = self.scenes[0], self.scenes[1]
        merged = SceneNode(
            scene_id=a.scene_id,
            sum_vec=a.sum_vec + b.sum_vec,
            count=a.count + b.count,
            member_archive_ids=a.member_archive_ids + b.member_archive_ids,
        )
        self.scenes = [merged] + self.scenes[2:]
        for aid in merged.member_archive_ids:
            self.entry_to_scene[aid] = merged.scene_id

    def coarse_archive_ids(self, query: torch.Tensor, top_scenes: int) -> Set[int]:
        q = query.detach().float().cpu().flatten()
        q = F.normalize(q, dim=-1)
        if not self.scenes:
            return set()
        scored: List[tuple[float, SceneNode]] = []
        for scene in self.scenes:
            sim = float(F.cosine_similarity(q.unsqueeze(0), scene.centroid.unsqueeze(0), dim=-1).mean())
            scored.append((sim, scene))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: Set[int] = set()
        for _, scene in scored[: max(1, top_scenes)]:
            out.update(scene.member_archive_ids)
        return out
