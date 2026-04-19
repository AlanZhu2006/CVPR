from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nuc_runtime.config import MemoryConfig


@dataclass
class _VoxelRecord:
    position: np.ndarray
    rgb: np.ndarray
    normal: np.ndarray
    confidence: float
    observations: int


class LocalSurfaceVolume:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._voxels: dict[tuple[int, int, int], _VoxelRecord] = {}

    def fuse_frame(
        self,
        pose: np.ndarray,
        depth: np.ndarray,
        confidence: np.ndarray,
        rgb: np.ndarray,
        K: np.ndarray,
        stable_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if depth.size == 0:
            return {"fused_points": 0, "voxel_count": len(self._voxels)}

        xyz_cam, colors, normals, conf = self._sample_frame_surface(
            depth=depth,
            confidence=confidence,
            rgb=rgb,
            K=K,
            stable_mask=stable_mask,
        )
        if xyz_cam.size == 0:
            return {"fused_points": 0, "voxel_count": len(self._voxels)}

        xyz_world = self._camera_to_world(pose, xyz_cam)
        rot = pose[:3, :3].astype(np.float32)
        normals_world = normals @ rot.T

        voxel_size = max(1e-3, float(self.config.gaussian_local_volume_voxel_size))
        fused = 0
        for idx, xyz in enumerate(xyz_world):
            key = tuple(np.floor(xyz / voxel_size).astype(np.int32))
            weight = float(max(0.05, conf[idx]))
            record = self._voxels.get(key)
            if record is None:
                self._voxels[key] = _VoxelRecord(
                    position=xyz.astype(np.float32),
                    rgb=colors[idx].astype(np.float32),
                    normal=self._normalize(normals_world[idx]),
                    confidence=weight,
                    observations=1,
                )
                fused += 1
                continue

            total = max(1e-6, record.confidence + weight)
            alpha = weight / total
            record.position = (1.0 - alpha) * record.position + alpha * xyz.astype(np.float32)
            record.rgb = (1.0 - alpha) * record.rgb + alpha * colors[idx].astype(np.float32)
            record.normal = self._normalize((1.0 - alpha) * record.normal + alpha * normals_world[idx])
            record.confidence = min(12.0, record.confidence + 0.55 * weight)
            record.observations += 1
            fused += 1

        return {"fused_points": fused, "voxel_count": len(self._voxels)}

    def extract_gaussians(
        self,
        max_points: int,
        default_scale: float,
        pose: np.ndarray | None = None,
        depth_map: np.ndarray | None = None,
        K: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if not self._voxels:
            return self._empty_bundle()

        records = list(self._voxels.values())
        if pose is not None and depth_map is not None and K is not None:
            records = self._filter_near_visible(records, pose=pose, depth_map=depth_map, K=K)
            records = self._extract_thin_visible_surface(records, pose=pose, K=K)
        records = [record for record in records if record.observations >= max(1, self.config.gaussian_local_volume_min_observations)]
        if not records:
            return self._empty_bundle()
        score = np.array(
            [record.confidence + 0.15 * min(8, record.observations) for record in records],
            dtype=np.float32,
        )
        keep = min(len(records), max(1, int(max_points)))
        if len(records) > keep:
            order = np.argpartition(score, -keep)[-keep:]
            records = [records[idx] for idx in order[np.argsort(score[order])[::-1]]]

        xyz = np.stack([record.position for record in records], axis=0).astype(np.float32)
        rgb = np.clip(np.stack([record.rgb for record in records], axis=0), 0.0, 255.0).astype(np.uint8)
        normals = np.stack([self._normalize(record.normal) for record in records], axis=0).astype(np.float32)
        tangent_u, tangent_v = self._tangent_axes(normals)
        conf = np.array([record.confidence for record in records], dtype=np.float32)
        obs = np.array([record.observations for record in records], dtype=np.float32)
        base_scale = max(0.02, float(self.config.gaussian_local_volume_voxel_size) * 0.75)
        scale = np.clip(base_scale * (0.9 + 0.12 / np.sqrt(np.clip(conf, 0.15, None))), 0.25 * default_scale, 2.5 * default_scale).astype(np.float32)
        opacity = np.clip(0.55 + 0.05 * np.minimum(obs, 6.0) + 0.06 * np.log1p(conf), 0.18, 0.96).astype(np.float32)
        axis_u = (tangent_u * scale[:, None]).astype(np.float32)
        axis_v = (tangent_v * scale[:, None]).astype(np.float32)
        unstable = np.clip(0.85 / np.sqrt(np.clip(conf, 0.2, None)), 0.12, 0.78).astype(np.float32)
        recentness = np.clip(0.45 + 0.06 * np.minimum(obs, 6.0), 0.35, 0.92).astype(np.float32)
        source = np.full((xyz.shape[0],), 1, dtype=np.int8)
        return {
            "xyz": xyz,
            "rgb": rgb,
            "scale": scale,
            "opacity": opacity,
            "axis_u": axis_u,
            "axis_v": axis_v,
            "unstable": unstable,
            "recentness": recentness,
            "source": source,
        }

    def summary(self) -> dict[str, Any]:
        return {"voxel_count": len(self._voxels)}

    def _sample_frame_surface(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        rgb: np.ndarray,
        K: np.ndarray,
        stable_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stride = max(2, int(self.config.gaussian_local_volume_fuse_stride))
        h, w = depth.shape
        ys = np.arange(stride // 2, h - 1, stride, dtype=np.int32)
        xs = np.arange(stride // 2, w - 1, stride, dtype=np.int32)
        if ys.size == 0 or xs.size == 0:
            return self._empty_frame_samples()
        xx, yy = np.meshgrid(xs, ys)
        sample_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        if stable_mask is not None:
            valid_mask = stable_mask[sample_xy[:, 1], sample_xy[:, 0]]
        else:
            valid_mask = np.ones((sample_xy.shape[0],), dtype=bool)

        bad_blocks = self._bad_depth_blocks(depth, confidence, stable_mask)
        if bad_blocks is not None:
            valid_mask &= bad_blocks[sample_xy[:, 1], sample_xy[:, 0]]

        sampled_depth = depth[sample_xy[:, 1], sample_xy[:, 0]]
        sampled_conf = confidence[sample_xy[:, 1], sample_xy[:, 0]]
        valid = valid_mask.astype(bool)
        valid &= np.isfinite(sampled_depth)
        valid &= sampled_depth > 0.05
        valid &= sampled_conf >= self.config.gaussian_local_volume_confidence_threshold
        if not np.any(valid):
            return self._empty_frame_samples()

        sample_xy = sample_xy[valid]
        sampled_depth = sampled_depth[valid].astype(np.float32)
        sampled_conf = sampled_conf[valid].astype(np.float32)
        colors = rgb[sample_xy[:, 1], sample_xy[:, 0]].astype(np.uint8)

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (sample_xy[:, 0].astype(np.float32) - cx) * sampled_depth / fx
        y = (sample_xy[:, 1].astype(np.float32) - cy) * sampled_depth / fy
        xyz_cam = np.stack([x, y, sampled_depth], axis=1).astype(np.float32)

        nx = np.clip(sample_xy[:, 0] + stride, 0, w - 1)
        ny = np.clip(sample_xy[:, 1] + stride, 0, h - 1)
        depth_x = depth[sample_xy[:, 1], nx].astype(np.float32)
        depth_y = depth[ny, sample_xy[:, 0]].astype(np.float32)
        depth_x = np.where(np.isfinite(depth_x) & (depth_x > 0.05), depth_x, sampled_depth)
        depth_y = np.where(np.isfinite(depth_y) & (depth_y > 0.05), depth_y, sampled_depth)
        xyz_x = np.stack(
            [
                (nx.astype(np.float32) - cx) * depth_x / fx,
                (sample_xy[:, 1].astype(np.float32) - cy) * depth_x / fy,
                depth_x,
            ],
            axis=1,
        ).astype(np.float32)
        xyz_y = np.stack(
            [
                (sample_xy[:, 0].astype(np.float32) - cx) * depth_y / fx,
                (ny.astype(np.float32) - cy) * depth_y / fy,
                depth_y,
            ],
            axis=1,
        ).astype(np.float32)
        normals = np.cross(xyz_x - xyz_cam, xyz_y - xyz_cam)
        normals = np.stack([self._normalize(normal) for normal in normals], axis=0).astype(np.float32)
        return xyz_cam, colors, normals, sampled_conf

    def _bad_depth_blocks(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        stable_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        h, w = depth.shape
        stride = max(4, int(self.config.gaussian_local_volume_fuse_stride) * 2)
        valid = np.isfinite(depth) & (depth > 0.05) & (confidence >= self.config.gaussian_local_volume_confidence_threshold)
        if stable_mask is not None:
            valid &= stable_mask.astype(bool)
        keep = np.ones((h, w), dtype=bool)
        for y0 in range(0, h, stride):
            y1 = min(h, y0 + stride)
            for x0 in range(0, w, stride):
                x1 = min(w, x0 + stride)
                block_valid = valid[y0:y1, x0:x1]
                if block_valid.size == 0:
                    continue
                bad_ratio = 1.0 - float(np.mean(block_valid))
                if bad_ratio >= self.config.gaussian_local_volume_block_bad_ratio:
                    keep[y0:y1, x0:x1] = False
        return keep

    def _filter_near_visible(
        self,
        records: list[_VoxelRecord],
        pose: np.ndarray,
        depth_map: np.ndarray,
        K: np.ndarray,
    ) -> list[_VoxelRecord]:
        if not records:
            return records
        xyz_world = np.stack([record.position for record in records], axis=0).astype(np.float32)
        xyz_cam = self._world_to_camera(pose, xyz_world)
        depth = xyz_cam[:, 2]
        valid = np.isfinite(xyz_cam).all(axis=1)
        valid &= depth > 0.05
        valid &= depth <= self.config.gaussian_local_volume_visible_max_depth_m
        if not np.any(valid):
            return []
        projected = xyz_cam @ K.T
        uv = projected[:, :2] / np.clip(projected[:, 2:], 1e-8, None)
        h, w = depth_map.shape
        valid &= uv[:, 0] >= 1
        valid &= uv[:, 0] < w - 1
        valid &= uv[:, 1] >= 1
        valid &= uv[:, 1] < h - 1
        if not np.any(valid):
            return []
        idx = np.flatnonzero(valid)
        uv_valid = uv[idx]
        observed = self._sample_bilinear(depth_map.astype(np.float32), uv_valid)
        visible = np.isfinite(observed) & (observed > 0.05)
        visible &= depth[idx] <= observed + self.config.gaussian_local_volume_visible_depth_margin_m
        if not np.any(visible):
            return []
        front_idx = idx[visible]
        cam_dirs = xyz_cam[front_idx]
        cam_dirs = cam_dirs / np.clip(np.linalg.norm(cam_dirs, axis=1, keepdims=True), 1e-6, None)
        keep_records: list[_VoxelRecord] = []
        for record_idx, dir_vec in zip(front_idx, cam_dirs):
            normal = self._normalize(records[record_idx].normal)
            view_cos = float(np.dot(normal, -dir_vec.astype(np.float32)))
            if view_cos >= self.config.gaussian_local_volume_normal_view_cosine:
                keep_records.append(records[record_idx])
        return keep_records

    def _extract_thin_visible_surface(
        self,
        records: list[_VoxelRecord],
        pose: np.ndarray,
        K: np.ndarray,
    ) -> list[_VoxelRecord]:
        if not records:
            return records
        xyz_world = np.stack([record.position for record in records], axis=0).astype(np.float32)
        xyz_cam = self._world_to_camera(pose, xyz_world)
        depth = xyz_cam[:, 2]
        valid = np.isfinite(xyz_cam).all(axis=1) & (depth > 0.05)
        if not np.any(valid):
            return []
        projected = xyz_cam @ K.T
        uv = projected[:, :2] / np.clip(projected[:, 2:], 1e-8, None)
        idx = np.flatnonzero(valid)
        uv_valid = uv[idx]
        depth_valid = depth[idx]
        cell_px = max(2, int(self.config.gaussian_local_volume_thin_cell_px))
        max_layers = max(1, int(self.config.gaussian_local_volume_thin_max_layers))
        gap_m = max(1e-3, float(self.config.gaussian_local_volume_thin_depth_gap_m))

        buckets: dict[tuple[int, int], list[tuple[float, float, int]]] = {}
        for local_i, record_idx in enumerate(idx):
            u, v = uv_valid[local_i]
            cell = (int(np.floor(u / cell_px)), int(np.floor(v / cell_px)))
            record = records[record_idx]
            score = float(record.confidence + 0.2 * min(6, record.observations))
            buckets.setdefault(cell, []).append((float(depth_valid[local_i]), score, int(record_idx)))

        keep_indices: list[int] = []
        for items in buckets.values():
            items.sort(key=lambda item: (item[0], -item[1]))
            picked_depths: list[float] = []
            picked = 0
            for dep, _, record_idx in items:
                if picked >= max_layers:
                    break
                if picked_depths and abs(dep - picked_depths[-1]) < gap_m:
                    continue
                keep_indices.append(record_idx)
                picked_depths.append(dep)
                picked += 1

        if not keep_indices:
            return []
        keep_indices = sorted(set(keep_indices))
        return [records[idx] for idx in keep_indices]

    def _tangent_axes(self, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ref = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (normals.shape[0], 1))
        parallel = np.abs(np.sum(ref * normals, axis=1)) > 0.9
        if np.any(parallel):
            ref[parallel] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        tangent_u = np.cross(normals, ref)
        tangent_u = np.stack([self._normalize(v) for v in tangent_u], axis=0).astype(np.float32)
        tangent_v = np.cross(normals, tangent_u)
        tangent_v = np.stack([self._normalize(v) for v in tangent_v], axis=0).astype(np.float32)
        return tangent_u, tangent_v

    def _camera_to_world(self, pose: np.ndarray, xyz_camera: np.ndarray) -> np.ndarray:
        xyz_h = np.concatenate([xyz_camera, np.ones((xyz_camera.shape[0], 1), dtype=np.float32)], axis=1)
        return (xyz_h @ pose.T)[:, :3]

    def _world_to_camera(self, pose: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(pose)
        xyz_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
        return (xyz_h @ w2c.T)[:, :3]

    def _sample_bilinear(self, image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x = np.clip(points_xy[:, 0], 0.0, w - 2.001)
        y = np.clip(points_xy[:, 1], 0.0, h - 2.001)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)
        top = (1.0 - wx) * image[y0, x0] + wx * image[y0, x1]
        bottom = (1.0 - wx) * image[y1, x0] + wx * image[y1, x1]
        return ((1.0 - wy) * top + wy * bottom).astype(np.float32)

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def _empty_bundle(self) -> dict[str, np.ndarray]:
        return {
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
            "scale": np.zeros((0,), dtype=np.float32),
            "opacity": np.zeros((0,), dtype=np.float32),
            "axis_u": np.zeros((0, 3), dtype=np.float32),
            "axis_v": np.zeros((0, 3), dtype=np.float32),
            "unstable": np.zeros((0,), dtype=np.float32),
            "recentness": np.zeros((0,), dtype=np.float32),
            "source": np.zeros((0,), dtype=np.int8),
        }

    def _empty_frame_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
