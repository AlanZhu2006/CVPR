from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nuc_runtime.config import MemoryConfig
from nuc_runtime.local_tsdf import LocalSurfaceVolume
from nuc_runtime.models import KeyframeRecord


@dataclass
class IncrementalGaussianState:
    submap_id: int
    keyframe_count: int = 0
    pair_updates: int = 0
    stereo_updates: int = 0
    point_count: int = 0
    recovered_seed_points: int = 0
    optimize_steps: int = 0
    last_keyframe: KeyframeRecord | None = None
    xyz: list[np.ndarray] = field(default_factory=list)
    rgb: list[np.ndarray] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    opacity: list[float] = field(default_factory=list)
    axis_u: list[np.ndarray] = field(default_factory=list)
    axis_v: list[np.ndarray] = field(default_factory=list)
    unstable: list[np.ndarray] = field(default_factory=list)
    recentness: list[np.ndarray] = field(default_factory=list)
    source: list[np.ndarray] = field(default_factory=list)

    def append_points(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        scale: np.ndarray,
        opacity: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        unstable: np.ndarray,
        recentness: np.ndarray,
        source: np.ndarray,
    ) -> None:
        if xyz.size == 0:
            return
        self.xyz.append(xyz.astype(np.float32))
        self.rgb.append(rgb.astype(np.uint8))
        self.scale.append(scale.astype(np.float32))
        self.opacity.append(opacity.astype(np.float32))
        self.axis_u.append(axis_u.astype(np.float32))
        self.axis_v.append(axis_v.astype(np.float32))
        self.unstable.append(unstable.astype(np.float32))
        self.recentness.append(recentness.astype(np.float32))
        self.source.append(source.astype(np.int8))
        self.point_count += int(xyz.shape[0])

    def replace_points(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        scale: np.ndarray,
        opacity: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        unstable: np.ndarray,
        recentness: np.ndarray,
        source: np.ndarray,
    ) -> None:
        self.xyz = [xyz.astype(np.float32)] if xyz.size > 0 else []
        self.rgb = [rgb.astype(np.uint8)] if rgb.size > 0 else []
        self.scale = [scale.astype(np.float32)] if scale.size > 0 else []
        self.opacity = [opacity.astype(np.float32)] if opacity.size > 0 else []
        self.axis_u = [axis_u.astype(np.float32)] if axis_u.size > 0 else []
        self.axis_v = [axis_v.astype(np.float32)] if axis_v.size > 0 else []
        self.unstable = [unstable.astype(np.float32)] if unstable.size > 0 else []
        self.recentness = [recentness.astype(np.float32)] if recentness.size > 0 else []
        self.source = [source.astype(np.int8)] if source.size > 0 else []
        self.point_count = int(xyz.shape[0])


class IncrementalGaussianBuilder:
    def __init__(self, output_dir: str | Path, config: MemoryConfig):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.states: dict[int, IncrementalGaussianState] = {}
        self._intrinsics_cache: dict[Path, np.ndarray] = {}
        self._stereo_cache: dict[Path, tuple[np.ndarray, float]] = {}
        self._handle_cache: dict[str, dict[str, np.ndarray]] = {}
        self._local_volumes: dict[int, LocalSurfaceVolume] = {}

    def start_submap(self, submap_id: int) -> IncrementalGaussianState:
        state = IncrementalGaussianState(submap_id=submap_id)
        self.states[submap_id] = state
        if self.config.enable_local_surface_volume:
            self._local_volumes[submap_id] = LocalSurfaceVolume(self.config)
        return state

    def ingest_keyframe(self, submap_id: int, keyframe: KeyframeRecord) -> dict[str, Any]:
        state = self.states.get(submap_id)
        if state is None:
            state = self.start_submap(submap_id)

        state.keyframe_count += 1
        stereo_info = self._get_stereo_depth_info(keyframe)
        self._update_local_surface_volume(submap_id, keyframe, stereo_info)
        if state.last_keyframe is None:
            stereo_xyz, stereo_rgb, stereo_scale, stereo_opacity, stereo_axis_u, stereo_axis_v = self._triangulate_stereo_keyframe(keyframe)
            if stereo_xyz.size > 0:
                self._fuse_points_into_state(
                    state,
                    stereo_xyz,
                    stereo_rgb,
                    stereo_scale,
                    stereo_opacity,
                    stereo_axis_u,
                    stereo_axis_v,
                    np.full((stereo_xyz.shape[0],), 0.75, dtype=np.float32),
                    np.ones((stereo_xyz.shape[0],), dtype=np.float32),
                    np.full((stereo_xyz.shape[0],), 1, dtype=np.int8),
                )
                state.stereo_updates += 1
            state.last_keyframe = keyframe
            return self._state_summary(state)

        new_xyz, new_rgb, new_scale, new_opacity, new_axis_u, new_axis_v = self._triangulate_pair(state.last_keyframe, keyframe)
        if new_xyz.size > 0:
            state.append_points(
                new_xyz,
                new_rgb,
                new_scale,
                new_opacity,
                new_axis_u,
                new_axis_v,
                unstable=np.full((new_xyz.shape[0],), 0.62, dtype=np.float32),
                recentness=np.ones((new_xyz.shape[0],), dtype=np.float32),
                source=np.full((new_xyz.shape[0],), 2, dtype=np.int8),
            )
            state.pair_updates += 1
        stereo_xyz, stereo_rgb, stereo_scale, stereo_opacity, stereo_axis_u, stereo_axis_v = self._triangulate_stereo_keyframe(keyframe)
        if stereo_xyz.size > 0:
            self._fuse_points_into_state(
                state,
                stereo_xyz,
                stereo_rgb,
                stereo_scale,
                stereo_opacity,
                stereo_axis_u,
                stereo_axis_v,
                np.full((stereo_xyz.shape[0],), 0.72, dtype=np.float32),
                np.ones((stereo_xyz.shape[0],), dtype=np.float32),
                np.full((stereo_xyz.shape[0],), 1, dtype=np.int8),
            )
            state.stereo_updates += 1
        state.last_keyframe = keyframe
        return self._state_summary(state)

    def _fuse_points_into_state(
        self,
        state: IncrementalGaussianState,
        xyz: np.ndarray,
        rgb: np.ndarray,
        scale: np.ndarray,
        opacity: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        unstable: np.ndarray,
        recentness: np.ndarray,
        source: np.ndarray,
    ) -> None:
        if xyz.size == 0:
            return
        if state.point_count <= 0:
            state.append_points(xyz, rgb, scale, opacity, axis_u, axis_v, unstable, recentness, source)
            return

        bundle = self.active_points(state.submap_id)
        cur_xyz = bundle["xyz"]
        cur_rgb = bundle["rgb"].astype(np.float32)
        cur_scale = bundle["scale"]
        cur_opacity = bundle["opacity"]
        cur_axis_u = bundle["axis_u"]
        cur_axis_v = bundle["axis_v"]
        cur_unstable = bundle["unstable"]
        cur_recentness = bundle["recentness"]
        cur_source = bundle["source"]
        voxel = max(1e-3, float(self.config.gaussian_stereo_fusion_voxel_size))
        voxel_map = {
            tuple(cell): idx
            for idx, cell in enumerate(np.floor(cur_xyz / voxel).astype(np.int32))
        }

        for idx, cell in enumerate(np.floor(xyz / voxel).astype(np.int32)):
            key = tuple(cell)
            existing = voxel_map.get(key)
            if existing is None:
                voxel_map[key] = cur_xyz.shape[0]
                cur_xyz = np.vstack([cur_xyz, xyz[idx : idx + 1]])
                cur_rgb = np.vstack([cur_rgb, rgb[idx : idx + 1].astype(np.float32)])
                cur_scale = np.concatenate([cur_scale, scale[idx : idx + 1]])
                cur_opacity = np.concatenate([cur_opacity, opacity[idx : idx + 1]])
                cur_axis_u = np.vstack([cur_axis_u, axis_u[idx : idx + 1]])
                cur_axis_v = np.vstack([cur_axis_v, axis_v[idx : idx + 1]])
                cur_unstable = np.concatenate([cur_unstable, unstable[idx : idx + 1]])
                cur_recentness = np.concatenate([cur_recentness, recentness[idx : idx + 1]])
                cur_source = np.concatenate([cur_source, source[idx : idx + 1]])
                continue

            old_w = max(0.15, float(cur_opacity[existing]))
            new_w = max(0.15, float(opacity[idx]))
            mix = new_w / (old_w + new_w)
            cur_xyz[existing] = (1.0 - mix) * cur_xyz[existing] + mix * xyz[idx]
            cur_rgb[existing] = (1.0 - mix) * cur_rgb[existing] + mix * rgb[idx].astype(np.float32)
            cur_scale[existing] = np.float32((1.0 - mix) * cur_scale[existing] + mix * scale[idx])
            cur_opacity[existing] = np.float32(np.clip(0.6 * cur_opacity[existing] + 0.4 * opacity[idx], 0.08, 0.98))
            cur_axis_u[existing] = (1.0 - mix) * cur_axis_u[existing] + mix * axis_u[idx]
            cur_axis_v[existing] = (1.0 - mix) * cur_axis_v[existing] + mix * axis_v[idx]
            cur_unstable[existing] = np.float32(max(cur_unstable[existing], unstable[idx]))
            cur_recentness[existing] = np.float32(max(cur_recentness[existing], recentness[idx]))
            if cur_source[existing] != 3:
                cur_source[existing] = source[idx]

        self.states[state.submap_id].replace_points(
            cur_xyz.astype(np.float32),
            np.clip(cur_rgb, 0.0, 255.0).astype(np.uint8),
            cur_scale.astype(np.float32),
            cur_opacity.astype(np.float32),
            cur_axis_u.astype(np.float32),
            cur_axis_v.astype(np.float32),
            cur_unstable.astype(np.float32),
            cur_recentness.astype(np.float32),
            cur_source.astype(np.int8),
        )

    def warm_start_submap(self, submap_id: int, handles: list[dict[str, Any]]) -> dict[str, Any]:
        state = self.states.get(submap_id)
        if state is None:
            state = self.start_submap(submap_id)

        total_added = 0
        for handle in handles:
            npz_path = handle.get("npz_path")
            if not npz_path:
                continue
            npz_file = Path(npz_path)
            if not npz_file.exists():
                continue
            bundle = np.load(npz_file)
            xyz = bundle["xyz"]
            rgb = bundle["rgb"]
            scale = bundle["scale"]
            opacity = bundle["opacity"]
            limit = min(self.config.gaussian_warmstart_max_points, xyz.shape[0])
            if limit <= 0:
                continue
            xyz = xyz[:limit]
            rgb = rgb[:limit]
            scale = 0.8 * scale[:limit]
            opacity = 0.7 * opacity[:limit]
            axis_u = bundle["axis_u"][:limit] if "axis_u" in bundle else self._default_axes_from_scale(scale, axis="u")
            axis_v = bundle["axis_v"][:limit] if "axis_v" in bundle else self._default_axes_from_scale(scale, axis="v")
            unstable = bundle["unstable"][:limit] if "unstable" in bundle else np.full((limit,), 0.7, dtype=np.float32)
            recentness = np.full((limit,), 1.0, dtype=np.float32)
            source = np.full((limit,), 3, dtype=np.int8)
            state.append_points(xyz, rgb, scale, opacity, axis_u, axis_v, unstable, recentness, source)
            total_added += int(limit)
        state.recovered_seed_points += total_added
        summary = self._state_summary(state)
        summary["last_seed_points_added"] = total_added
        return summary

    def finalize_submap(self, submap_id: int, reason: str) -> dict[str, Any] | None:
        state = self.states.get(submap_id)
        if state is None:
            return None
        submap_dir = self.output_dir / f"submap_{submap_id:04d}"
        submap_dir.mkdir(parents=True, exist_ok=True)
        if state.point_count <= 0:
            handle = {
                "submap_id": submap_id,
                "reason": reason,
                "point_count": 0,
                "pair_updates": state.pair_updates,
                "keyframe_count": state.keyframe_count,
                "ply_path": "",
                "npz_path": "",
                "recovered_seed_points": state.recovered_seed_points,
            }
            self.states.pop(submap_id, None)
            return handle

        xyz = np.vstack(state.xyz).astype(np.float32)
        rgb = np.vstack(state.rgb).astype(np.uint8)
        scale = np.concatenate(state.scale).astype(np.float32)
        opacity = np.concatenate(state.opacity).astype(np.float32)
        axis_u = np.vstack(state.axis_u).astype(np.float32)
        axis_v = np.vstack(state.axis_v).astype(np.float32)
        unstable = np.concatenate(state.unstable).astype(np.float32)
        recentness = np.concatenate(state.recentness).astype(np.float32)
        source = np.concatenate(state.source).astype(np.int8)

        npz_path = submap_dir / "gaussians.npz"
        ply_path = submap_dir / "gaussians.ply"
        np.savez_compressed(
            npz_path,
            xyz=xyz,
            rgb=rgb,
            scale=scale,
            opacity=opacity,
            axis_u=axis_u,
            axis_v=axis_v,
            unstable=unstable,
            recentness=recentness,
            source=source,
        )
        self._write_ply(ply_path, xyz, rgb, scale, opacity, axis_u, axis_v)
        coarse = self._build_coarse_bundle(
            xyz=xyz,
            rgb=rgb,
            scale=scale,
            opacity=opacity,
            axis_u=axis_u,
            axis_v=axis_v,
            unstable=unstable,
            recentness=recentness,
            source=source,
        )
        coarse_npz_path = submap_dir / "gaussians_coarse.npz"
        coarse_ply_path = submap_dir / "gaussians_coarse.ply"
        np.savez_compressed(coarse_npz_path, **coarse)
        self._write_ply(
            coarse_ply_path,
            coarse["xyz"],
            coarse["rgb"],
            coarse["scale"],
            coarse["opacity"],
            coarse["axis_u"],
            coarse["axis_v"],
        )

        handle = {
            "submap_id": submap_id,
            "reason": reason,
            "point_count": int(xyz.shape[0]),
            "pair_updates": state.pair_updates,
            "keyframe_count": state.keyframe_count,
            "ply_path": str(ply_path),
            "npz_path": str(npz_path),
            "coarse_ply_path": str(coarse_ply_path),
            "coarse_npz_path": str(coarse_npz_path),
            "coarse_point_count": int(coarse["xyz"].shape[0]),
            "recovered_seed_points": state.recovered_seed_points,
        }
        self.states.pop(submap_id, None)
        self._local_volumes.pop(submap_id, None)
        return handle

    def _update_local_surface_volume(
        self,
        submap_id: int,
        keyframe: KeyframeRecord,
        stereo_info: dict[str, Any] | None,
    ) -> None:
        if not self.config.enable_local_surface_volume or stereo_info is None or keyframe.image_path is None:
            return
        volume = self._local_volumes.get(submap_id)
        if volume is None:
            volume = LocalSurfaceVolume(self.config)
            self._local_volumes[submap_id] = volume
        image_bgr = cv2.imread(keyframe.image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return
        volume.fuse_frame(
            pose=keyframe.pose,
            depth=stereo_info["depth"],
            confidence=stereo_info["confidence"],
            rgb=image_bgr[:, :, ::-1],
            K=stereo_info["K"],
            stable_mask=stereo_info["stable_mask"] > 0,
        )
        fused = volume.extract_gaussians(
            max_points=self.config.gaussian_local_volume_extract_points,
            default_scale=self.config.gaussian_default_scale,
            pose=keyframe.pose,
            depth_map=stereo_info["depth"],
            K=stereo_info["K"],
        )
        if fused["xyz"].size == 0:
            return
        state = self.states.get(submap_id)
        if state is None:
            return
        self._fuse_points_into_state(
            state,
            fused["xyz"],
            fused["rgb"],
            fused["scale"],
            fused["opacity"],
            fused["axis_u"],
            fused["axis_v"],
            fused["unstable"],
            fused["recentness"],
            fused["source"],
        )

    def _build_coarse_bundle(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        scale: np.ndarray,
        opacity: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        unstable: np.ndarray,
        recentness: np.ndarray,
        source: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if xyz.shape[0] <= self.config.gaussian_coarse_max_points:
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
        voxel = max(1e-3, float(self.config.gaussian_coarse_voxel_size))
        voxel_ids = np.floor(xyz / voxel).astype(np.int32)
        _, unique_idx = np.unique(voxel_ids, axis=0, return_index=True)
        if unique_idx.shape[0] > self.config.gaussian_coarse_max_points:
            coarse_score = unstable[unique_idx] + 0.35 * recentness[unique_idx] + 0.15 * opacity[unique_idx]
            keep = unique_idx[np.argpartition(coarse_score, -self.config.gaussian_coarse_max_points)[-self.config.gaussian_coarse_max_points :]]
            unique_idx = np.sort(keep)
        return {
            "xyz": xyz[unique_idx].astype(np.float32),
            "rgb": rgb[unique_idx].astype(np.uint8),
            "scale": scale[unique_idx].astype(np.float32),
            "opacity": opacity[unique_idx].astype(np.float32),
            "axis_u": axis_u[unique_idx].astype(np.float32),
            "axis_v": axis_v[unique_idx].astype(np.float32),
            "unstable": unstable[unique_idx].astype(np.float32),
            "recentness": recentness[unique_idx].astype(np.float32),
            "source": source[unique_idx].astype(np.int8),
        }

    def active_points(self, submap_id: int) -> dict[str, np.ndarray]:
        state = self.states.get(submap_id)
        if state is None or state.point_count <= 0:
            return self._empty_bundle()
        return {
            "xyz": np.vstack(state.xyz).astype(np.float32),
            "rgb": np.vstack(state.rgb).astype(np.uint8),
            "scale": np.concatenate(state.scale).astype(np.float32),
            "opacity": np.concatenate(state.opacity).astype(np.float32),
            "axis_u": np.vstack(state.axis_u).astype(np.float32),
            "axis_v": np.vstack(state.axis_v).astype(np.float32),
            "unstable": np.concatenate(state.unstable).astype(np.float32),
            "recentness": np.concatenate(state.recentness).astype(np.float32),
            "source": np.concatenate(state.source).astype(np.int8),
        }

    def optimize_active_window(
        self,
        submap_id: int,
        keyframes: list[KeyframeRecord],
    ) -> dict[str, Any]:
        state = self.states.get(submap_id)
        if state is None or state.point_count <= 0 or not keyframes:
            return {"optimized_points": 0, "optimize_steps": 0}

        bundle = self.active_points(submap_id)
        xyz = bundle["xyz"]
        rgb = bundle["rgb"].astype(np.float32)
        scale = bundle["scale"]
        opacity = bundle["opacity"]
        axis_u = bundle["axis_u"]
        axis_v = bundle["axis_v"]
        unstable = bundle["unstable"]
        recentness = bundle["recentness"]
        source = bundle["source"]
        recent = keyframes[-max(1, self.config.gaussian_optimize_window) :]
        optimized_points = 0
        unstable = np.clip(
            unstable * self.config.gaussian_optimize_unstable_decay + recentness * self.config.gaussian_optimize_new_point_boost,
            0.0,
            3.0,
        ).astype(np.float32)
        recentness = np.clip(recentness * 0.78, 0.0, 1.0).astype(np.float32)

        for _ in range(max(1, self.config.gaussian_optimize_steps)):
            for keyframe in recent:
                if keyframe.image_path is None:
                    continue
                image_bgr = cv2.imread(keyframe.image_path, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    continue
                image_rgb = image_bgr[:, :, ::-1].astype(np.float32)
                stereo_info = self._get_stereo_depth_info(keyframe)
                K = self._load_intrinsics(Path(keyframe.image_path))
                xyz_camera = self._world_to_camera(keyframe.pose, xyz)
                valid = np.isfinite(xyz_camera).all(axis=1)
                valid &= xyz_camera[:, 2] > 0.1
                if not np.any(valid):
                    continue
                uv = (xyz_camera @ K.T)
                uv = uv[:, :2] / np.clip(uv[:, 2:], 1e-8, None)
                valid &= uv[:, 0] >= 2
                valid &= uv[:, 0] < image_rgb.shape[1] - 2
                valid &= uv[:, 1] >= 2
                valid &= uv[:, 1] < image_rgb.shape[0] - 2
                if not np.any(valid):
                    continue

                valid_idx = np.flatnonzero(valid)
                uv_valid = uv[valid]
                sampled_rgb = self._sample_bilinear_rgb(image_rgb, uv_valid)
                current_rgb = rgb[valid].copy()
                residual = sampled_rgb - current_rgb
                residual_mag = np.linalg.norm(residual, axis=1).astype(np.float32)
                unstable[valid] = np.maximum(
                    unstable[valid],
                    np.clip(residual_mag / max(1e-6, self.config.gaussian_optimize_error_threshold), 0.0, 2.5),
                )
                if np.any(source[valid] == 3):
                    unstable[valid] = np.where(
                        source[valid] == 3,
                        np.minimum(3.0, unstable[valid] + self.config.gaussian_optimize_recover_boost),
                        unstable[valid],
                    )

                select_score = unstable[valid] + 0.55 * recentness[valid]
                if stereo_info is not None:
                    observed_depth_valid = self._sample_bilinear_gray(stereo_info["depth"], uv_valid)
                    observed_conf_valid = self._sample_bilinear_gray(stereo_info["confidence"], uv_valid)
                    depth_residual_valid = np.abs(observed_depth_valid - xyz_camera[valid, 2]).astype(np.float32)
                    depth_score = (
                        np.isfinite(observed_depth_valid).astype(np.float32)
                        * (observed_conf_valid >= 0.1).astype(np.float32)
                        * np.clip(depth_residual_valid / max(1e-6, self.config.gaussian_optimize_depth_gate_m), 0.0, 1.2)
                    )
                    select_score = select_score + self.config.gaussian_optimize_depth_score_weight * depth_score
                error_boost = residual_mag >= self.config.gaussian_optimize_error_threshold
                if np.any(error_boost):
                    select_score = np.where(error_boost, select_score + 0.8, select_score)
                topk = min(self.config.gaussian_optimize_topk, valid_idx.shape[0])
                if topk <= 0:
                    continue
                if topk < valid_idx.shape[0]:
                    selected_local = np.argpartition(select_score, -topk)[-topk:]
                else:
                    selected_local = np.arange(valid_idx.shape[0], dtype=np.int32)
                selected_idx = valid_idx[selected_local]
                uv_selected = uv[selected_idx]
                sampled_rgb = sampled_rgb[selected_local]
                current_rgb = current_rgb[selected_local]
                residual = residual[selected_local]
                residual_mag = residual_mag[selected_local]
                predicted_depth = xyz_camera[selected_idx, 2].astype(np.float32)
                depth_residual = np.zeros_like(predicted_depth)
                depth_valid = np.zeros_like(predicted_depth, dtype=bool)
                if stereo_info is not None:
                    observed_depth = self._sample_bilinear_gray(stereo_info["depth"], uv_selected)
                    observed_conf = self._sample_bilinear_gray(stereo_info["confidence"], uv_selected)
                    depth_valid = np.isfinite(observed_depth)
                    depth_valid &= observed_depth > 0.05
                    depth_valid &= observed_conf >= 0.1
                    depth_residual = (observed_depth - predicted_depth).astype(np.float32)
                    gate = np.clip(
                        np.abs(depth_residual) / max(1e-6, self.config.gaussian_optimize_depth_gate_m),
                        0.0,
                        2.0,
                    )
                    unstable[selected_idx] = np.clip(
                        unstable[selected_idx] + depth_valid.astype(np.float32) * gate * self.config.gaussian_optimize_depth_score_weight,
                        0.0,
                        3.0,
                    )

                rgb[selected_idx] = np.clip(
                    current_rgb + self.config.gaussian_optimize_lr_color * residual,
                    0.0,
                    255.0,
                )
                gray_target = 0.299 * sampled_rgb[:, 0] + 0.587 * sampled_rgb[:, 1] + 0.114 * sampled_rgb[:, 2]
                gray_current = 0.299 * current_rgb[:, 0] + 0.587 * current_rgb[:, 1] + 0.114 * current_rgb[:, 2]
                gray_residual = gray_target - gray_current
                grad_x, grad_y = self._sample_image_gradient(image_rgb, uv_selected)
                flow_x = np.clip(self.config.gaussian_optimize_lr_position * grad_x * gray_residual / 255.0, -1.0, 1.0)
                flow_y = np.clip(self.config.gaussian_optimize_lr_position * grad_y * gray_residual / 255.0, -1.0, 1.0)
                depth = xyz_camera[selected_idx, 2]
                fx = float(K[0, 0])
                fy = float(K[1, 1])
                delta_cam = np.stack(
                    [
                        flow_x * depth / fx,
                        flow_y * depth / fy,
                        np.clip(-0.002 * gray_residual, -0.03, 0.03),
                    ],
                    axis=1,
                ).astype(np.float32)
                xyz_cam_updated = xyz_camera[selected_idx] + delta_cam
                if np.any(depth_valid):
                    target_depth = np.clip(
                        predicted_depth + self.config.gaussian_optimize_depth_weight * depth_residual,
                        0.05,
                        self.config.gaussian_stereo_max_depth_m,
                    )
                    target_depth = np.where(depth_valid, target_depth, xyz_cam_updated[:, 2]).astype(np.float32)
                    scale_ratio = np.clip(target_depth / np.clip(xyz_cam_updated[:, 2], 1e-3, None), 0.7, 1.35)
                    xyz_cam_updated[:, 0] *= scale_ratio
                    xyz_cam_updated[:, 1] *= scale_ratio
                    xyz_cam_updated[:, 2] = target_depth
                xyz[selected_idx] = self._camera_to_world(keyframe.pose, xyz_cam_updated)
                texture = np.sqrt(grad_x * grad_x + grad_y * grad_y)
                scale_adjust = 1.0 + self.config.gaussian_optimize_lr_scale * np.clip(texture / 24.0 - 0.35, -0.45, 0.45)
                if np.any(depth_valid):
                    depth_texture = np.clip(np.abs(depth_residual) / max(1e-6, self.config.gaussian_optimize_depth_gate_m), 0.0, 0.8)
                    scale_adjust = np.where(depth_valid, scale_adjust * (1.0 - 0.35 * depth_texture), scale_adjust)
                scale[selected_idx] = np.clip(
                    scale[selected_idx] * scale_adjust.astype(np.float32),
                    0.25 * self.config.gaussian_default_scale,
                    6.0 * self.config.gaussian_default_scale,
                )
                opacity_delta = self.config.gaussian_optimize_lr_opacity * np.clip(1.0 - np.abs(gray_residual) / 80.0, -0.5, 0.5)
                opacity[selected_idx] = np.clip(opacity[selected_idx] + opacity_delta.astype(np.float32), 0.08, 0.98)
                axis_u[selected_idx], axis_v[selected_idx] = self._camera_axes_to_world(
                    pose=keyframe.pose,
                    xyz_camera=xyz_cam_updated,
                    scale_u=np.maximum(np.linalg.norm(axis_u[selected_idx], axis=1), scale[selected_idx]).astype(np.float32),
                    scale_v=np.maximum(np.linalg.norm(axis_v[selected_idx], axis=1), scale[selected_idx]).astype(np.float32),
                )
                unstable[selected_idx] = np.clip(
                    0.55 * unstable[selected_idx]
                    + np.clip(residual_mag / max(1e-6, self.config.gaussian_optimize_error_threshold), 0.0, 1.0),
                    0.0,
                    3.0,
                )
                if np.any(depth_valid):
                    unstable[selected_idx] = np.clip(
                        unstable[selected_idx]
                        + depth_valid.astype(np.float32)
                        * np.clip(np.abs(depth_residual) / max(1e-6, self.config.gaussian_optimize_depth_gate_m), 0.0, 1.0),
                        0.0,
                        3.0,
                    )
                optimized_points += int(selected_idx.shape[0])

        state.replace_points(
            xyz,
            rgb.astype(np.uint8),
            scale,
            opacity,
            axis_u,
            axis_v,
            unstable,
            recentness,
            source,
        )
        state.optimize_steps += max(1, self.config.gaussian_optimize_steps)
        summary = self._state_summary(state)
        summary["optimized_points"] = optimized_points
        summary["optimize_steps"] = max(1, self.config.gaussian_optimize_steps)
        return summary

    def handle_points(self, handle: dict[str, Any] | None, use_coarse: bool = False) -> dict[str, np.ndarray]:
        if not handle:
            return self._empty_bundle()
        npz_path = handle.get("coarse_npz_path", "") if use_coarse else handle.get("npz_path", "")
        if not npz_path:
            return self._empty_bundle()
        cached = self._handle_cache.get(npz_path)
        if cached is not None:
            return cached
        npz_file = Path(npz_path)
        if not npz_file.exists():
            return self._empty_bundle()
        bundle = np.load(npz_file)
        cached = {
            "xyz": bundle["xyz"].astype(np.float32),
            "rgb": bundle["rgb"].astype(np.uint8),
            "scale": bundle["scale"].astype(np.float32),
            "opacity": bundle["opacity"].astype(np.float32),
            "axis_u": bundle["axis_u"].astype(np.float32) if "axis_u" in bundle else self._default_axes_from_scale(bundle["scale"].astype(np.float32), axis="u"),
            "axis_v": bundle["axis_v"].astype(np.float32) if "axis_v" in bundle else self._default_axes_from_scale(bundle["scale"].astype(np.float32), axis="v"),
            "unstable": bundle["unstable"].astype(np.float32) if "unstable" in bundle else np.full((bundle["xyz"].shape[0],), 0.35, dtype=np.float32),
            "recentness": bundle["recentness"].astype(np.float32) if "recentness" in bundle else np.zeros((bundle["xyz"].shape[0],), dtype=np.float32),
            "source": bundle["source"].astype(np.int8) if "source" in bundle else np.zeros((bundle["xyz"].shape[0],), dtype=np.int8),
        }
        self._handle_cache[npz_path] = cached
        return cached

    def _triangulate_pair(
        self,
        left_keyframe: KeyframeRecord,
        right_keyframe: KeyframeRecord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if left_keyframe.orb_descriptors is None or right_keyframe.orb_descriptors is None:
            return self._empty_points()
        if left_keyframe.keypoints_xy is None or right_keyframe.keypoints_xy is None:
            return self._empty_points()
        if left_keyframe.image_path is None or right_keyframe.image_path is None:
            return self._empty_points()

        raw_matches = self.matcher.knnMatch(left_keyframe.orb_descriptors, right_keyframe.orb_descriptors, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good.append(first)
        if len(good) < self.config.gaussian_min_pair_matches:
            return self._empty_points()

        good.sort(key=lambda item: item.distance)
        good = good[: self.config.gaussian_max_points_per_pair]

        left_points = np.float32([left_keyframe.keypoints_xy[m.queryIdx] for m in good])
        right_points = np.float32([right_keyframe.keypoints_xy[m.trainIdx] for m in good])
        K = self._load_intrinsics(Path(right_keyframe.image_path))
        left_proj = K @ np.linalg.inv(left_keyframe.pose)[:3, :]
        right_proj = K @ np.linalg.inv(right_keyframe.pose)[:3, :]
        hom = cv2.triangulatePoints(left_proj, right_proj, left_points.T, right_points.T)
        xyz_world = (hom[:3] / np.clip(hom[3:], 1e-8, None)).T

        left_cam = self._world_to_camera(left_keyframe.pose, xyz_world)
        right_cam = self._world_to_camera(right_keyframe.pose, xyz_world)
        valid = np.isfinite(xyz_world).all(axis=1)
        valid &= left_cam[:, 2] > 0.05
        valid &= right_cam[:, 2] > 0.05

        left_err = self._reprojection_error(K, left_cam, left_points)
        right_err = self._reprojection_error(K, right_cam, right_points)
        valid &= left_err <= self.config.gaussian_reproj_error_px
        valid &= right_err <= self.config.gaussian_reproj_error_px
        if not np.any(valid):
            return self._empty_points()

        xyz_world = xyz_world[valid]
        right_points = right_points[valid]
        right_cam = right_cam[valid]
        rgb = self._sample_rgb(Path(right_keyframe.image_path), right_points)
        scale = np.clip(
            0.01 * right_cam[:, 2],
            0.5 * self.config.gaussian_default_scale,
            4.0 * self.config.gaussian_default_scale,
        ).astype(np.float32)
        opacity = np.full((xyz_world.shape[0],), 0.85, dtype=np.float32)
        axis_u, axis_v = self._camera_axes_to_world(
            pose=right_keyframe.pose,
            xyz_camera=right_cam,
            scale_u=self.config.gaussian_pair_stride_scale * right_cam[:, 2] / float(K[0, 0]),
            scale_v=self.config.gaussian_pair_stride_scale * right_cam[:, 2] / float(K[1, 1]),
        )
        return xyz_world.astype(np.float32), rgb, scale, opacity, axis_u, axis_v

    def _load_intrinsics(self, image_path: Path) -> np.ndarray:
        sequence_path = image_path.parents[1]
        cached = self._intrinsics_cache.get(sequence_path)
        if cached is not None:
            return cached
        calib_path = sequence_path / "calib.txt"
        if not calib_path.exists():
            K = np.array([[700.0, 0.0, 600.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            self._intrinsics_cache[sequence_path] = K
            return K
        first_line = calib_path.read_text(encoding="utf-8").splitlines()[0]
        values = np.array([float(item) for item in first_line.split()[1:]], dtype=np.float32).reshape(3, 4)
        K = values[:, :3]
        self._intrinsics_cache[sequence_path] = K
        return K

    def _sample_rgb(self, image_path: Path, points_xy: np.ndarray) -> np.ndarray:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((points_xy.shape[0], 3), dtype=np.uint8)
        h, w = image.shape[:2]
        coords = np.round(points_xy).astype(np.int32)
        coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
        bgr = image[coords[:, 1], coords[:, 0]]
        return bgr[:, ::-1].astype(np.uint8)

    def _sample_bilinear_rgb(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        x = np.clip(points_xy[:, 0], 0.0, w - 2.001)
        y = np.clip(points_xy[:, 1], 0.0, h - 2.001)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)
        top = (1.0 - wx)[:, None] * image_rgb[y0, x0] + wx[:, None] * image_rgb[y0, x1]
        bottom = (1.0 - wx)[:, None] * image_rgb[y1, x0] + wx[:, None] * image_rgb[y1, x1]
        return ((1.0 - wy)[:, None] * top + wy[:, None] * bottom).astype(np.float32)

    def _sample_image_gradient(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gx = self._sample_bilinear_gray(grad_x, points_xy)
        gy = self._sample_bilinear_gray(grad_y, points_xy)
        return gx, gy

    def _sample_bilinear_gray(self, image_gray: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        h, w = image_gray.shape[:2]
        x = np.clip(points_xy[:, 0], 0.0, w - 2.001)
        y = np.clip(points_xy[:, 1], 0.0, h - 2.001)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = (x - x0).astype(np.float32)
        wy = (y - y0).astype(np.float32)
        top = (1.0 - wx) * image_gray[y0, x0] + wx * image_gray[y0, x1]
        bottom = (1.0 - wx) * image_gray[y1, x0] + wx * image_gray[y1, x1]
        return ((1.0 - wy) * top + wy * bottom).astype(np.float32)

    def _world_to_camera(self, camera_pose: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(camera_pose)
        xyz_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
        return (xyz_h @ w2c.T)[:, :3]

    def _reprojection_error(self, K: np.ndarray, xyz_camera: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        projected = (xyz_camera @ K.T)
        projected = projected[:, :2] / np.clip(projected[:, 2:], 1e-8, None)
        return np.linalg.norm(projected - points_xy, axis=1)

    def _write_ply(
        self,
        ply_path: Path,
        xyz: np.ndarray,
        rgb: np.ndarray,
        scale: np.ndarray,
        opacity: np.ndarray,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
    ) -> None:
        header = [
            "ply",
            "format ascii 1.0",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float scale",
            "property float opacity",
            "property float axis_u_x",
            "property float axis_u_y",
            "property float axis_u_z",
            "property float axis_v_x",
            "property float axis_v_y",
            "property float axis_v_z",
            "end_header",
        ]
        lines = [" ".join([
            f"{point[0]:.6f}",
            f"{point[1]:.6f}",
            f"{point[2]:.6f}",
            str(int(color[0])),
            str(int(color[1])),
            str(int(color[2])),
            f"{point_scale:.6f}",
            f"{point_opacity:.6f}",
            f"{point_axis_u[0]:.6f}",
            f"{point_axis_u[1]:.6f}",
            f"{point_axis_u[2]:.6f}",
            f"{point_axis_v[0]:.6f}",
            f"{point_axis_v[1]:.6f}",
            f"{point_axis_v[2]:.6f}",
        ]) for point, color, point_scale, point_opacity, point_axis_u, point_axis_v in zip(xyz, rgb, scale, opacity, axis_u, axis_v)]
        ply_path.write_text("\n".join(header + lines) + "\n", encoding="utf-8")

    def _state_summary(self, state: IncrementalGaussianState) -> dict[str, Any]:
        unstable_count = 0
        if state.unstable:
            unstable_values = np.concatenate(state.unstable).astype(np.float32)
            unstable_count = int(np.sum(unstable_values >= 0.75))
        local_voxel_count = 0
        volume = self._local_volumes.get(state.submap_id)
        if volume is not None:
            local_voxel_count = int(volume.summary().get("voxel_count", 0))
        return {
            "submap_id": state.submap_id,
            "keyframe_count": state.keyframe_count,
            "pair_updates": state.pair_updates,
            "stereo_updates": state.stereo_updates,
            "point_count": state.point_count,
            "recovered_seed_points": state.recovered_seed_points,
            "unstable_points": unstable_count,
            "local_volume_voxels": local_voxel_count,
        }

    def _empty_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    def _empty_bundle(self) -> dict[str, np.ndarray]:
        xyz, rgb, scale, opacity, axis_u, axis_v = self._empty_points()
        return {
            "xyz": xyz,
            "rgb": rgb,
            "scale": scale,
            "opacity": opacity,
            "axis_u": axis_u,
            "axis_v": axis_v,
            "unstable": np.zeros((0,), dtype=np.float32),
            "recentness": np.zeros((0,), dtype=np.float32),
            "source": np.zeros((0,), dtype=np.int8),
        }

    def _triangulate_stereo_keyframe(
        self,
        keyframe: KeyframeRecord,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.config.enable_gaussian_stereo_seed:
            return self._empty_points()
        if keyframe.image_path is None or keyframe.right_image_path is None:
            return self._empty_points()
        left_bgr = cv2.imread(keyframe.image_path, cv2.IMREAD_COLOR)
        stereo_info = self._get_stereo_depth_info(keyframe)
        if left_bgr is None or stereo_info is None:
            return self._empty_points()

        disparity = stereo_info["disparity"]
        depth_map = stereo_info["depth"]
        texture = stereo_info["texture"]
        stable_mask = stereo_info["stable_mask"]
        left_gray = stereo_info["left_gray"]
        K = stereo_info["K"]
        baseline = float(stereo_info["baseline"])
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        corners = cv2.goodFeaturesToTrack(
            left_gray,
            maxCorners=int(self.config.gaussian_stereo_feature_points),
            qualityLevel=0.01,
            minDistance=max(4, self.config.gaussian_stereo_grid_stride // 2),
            mask=stable_mask,
            blockSize=5,
            useHarrisDetector=False,
        )

        stride = max(2, self.config.gaussian_stereo_grid_stride)
        sample_xy = self._quadtree_sample_pixels(
            depth_map=depth_map,
            texture=texture,
            stable_mask=stable_mask > 0,
            min_block=max(8, int(self.config.gaussian_stereo_quadtree_min_block)),
            var_threshold=float(self.config.gaussian_stereo_quadtree_var_threshold),
        )
        if sample_xy.size == 0:
            ys = np.arange(stride // 2, left_gray.shape[0], stride, dtype=np.int32)
            xs = np.arange(stride // 2, left_gray.shape[1], stride, dtype=np.int32)
            xx, yy = np.meshgrid(xs, ys)
            sample_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
        if corners is not None and corners.size > 0:
            corners_xy = np.round(corners.reshape(-1, 2)).astype(np.int32)
            corners_xy[:, 0] = np.clip(corners_xy[:, 0], 0, left_gray.shape[1] - 1)
            corners_xy[:, 1] = np.clip(corners_xy[:, 1], 0, left_gray.shape[0] - 1)
            sample_xy = np.vstack([corners_xy, sample_xy])
            sample_xy = np.unique(sample_xy, axis=0)

        disp = disparity[sample_xy[:, 1], sample_xy[:, 0]]
        local_consistency = stereo_info["consistency"][sample_xy[:, 1], sample_xy[:, 0]]
        local_texture = texture[sample_xy[:, 1], sample_xy[:, 0]]
        valid = np.isfinite(disp)
        valid &= disp >= self.config.gaussian_stereo_min_disparity_px
        valid &= local_consistency <= self.config.gaussian_stereo_consistency_threshold_px
        valid &= local_texture >= self.config.gaussian_stereo_texture_threshold
        if not np.any(valid):
            return self._empty_points()

        sample_xy = sample_xy[valid]
        depth = depth_map[sample_xy[:, 1], sample_xy[:, 0]]
        valid_depth = np.isfinite(depth)
        valid_depth &= depth > 0.05
        valid_depth &= depth <= self.config.gaussian_stereo_max_depth_m
        if not np.any(valid_depth):
            return self._empty_points()

        sample_xy = sample_xy[valid_depth]
        depth = depth[valid_depth]
        x_cam = (sample_xy[:, 0].astype(np.float32) - cx) * depth / fx
        y_cam = (sample_xy[:, 1].astype(np.float32) - cy) * depth / fy
        xyz_camera = np.stack([x_cam, y_cam, depth.astype(np.float32)], axis=1)
        xyz_world = self._camera_to_world(keyframe.pose, xyz_camera)
        rgb = left_bgr[sample_xy[:, 1], sample_xy[:, 0]][:, ::-1].astype(np.uint8)
        scale = np.clip(
            0.006 * xyz_camera[:, 2],
            0.35 * self.config.gaussian_default_scale,
            3.0 * self.config.gaussian_default_scale,
        ).astype(np.float32)
        opacity = np.full((xyz_world.shape[0],), 0.9, dtype=np.float32)
        axis_u, axis_v = self._estimate_stereo_surfel_axes(
            pose=keyframe.pose,
            xyz_camera=xyz_camera,
            sample_xy=sample_xy,
            disparity=disparity,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            baseline=baseline,
            stride=stride,
        )

        if xyz_world.shape[0] > self.config.gaussian_stereo_max_points_per_frame:
            order = np.linspace(
                0,
                xyz_world.shape[0] - 1,
                self.config.gaussian_stereo_max_points_per_frame,
                dtype=np.int32,
            )
            xyz_world = xyz_world[order]
            rgb = rgb[order]
            scale = scale[order]
            opacity = opacity[order]
            axis_u = axis_u[order]
            axis_v = axis_v[order]
        return xyz_world.astype(np.float32), rgb, scale, opacity, axis_u, axis_v

    def _get_stereo_depth_info(self, keyframe: KeyframeRecord) -> dict[str, Any] | None:
        if keyframe.image_path is None or keyframe.right_image_path is None:
            return None
        cache_key = f"{keyframe.image_path}|{keyframe.right_image_path}"
        cached = self._handle_cache.get(cache_key)
        if cached is not None:
            return cached

        left_bgr = cv2.imread(keyframe.image_path, cv2.IMREAD_COLOR)
        left_gray = cv2.imread(keyframe.image_path, cv2.IMREAD_GRAYSCALE)
        right_gray = cv2.imread(keyframe.right_image_path, cv2.IMREAD_GRAYSCALE)
        if left_bgr is None or left_gray is None or right_gray is None:
            return None
        left_gray = cv2.bilateralFilter(left_gray, 5, 18, 7)
        right_gray = cv2.bilateralFilter(right_gray, 5, 18, 7)
        K, baseline = self._load_stereo_params(Path(keyframe.image_path))
        if baseline <= 1e-6:
            return None

        num_disparities = int(np.ceil(self.config.gaussian_stereo_num_disparities / 16.0) * 16)
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max(16, num_disparities),
            blockSize=max(3, int(self.config.gaussian_stereo_block_size) | 1),
            P1=8 * 3 * self.config.gaussian_stereo_block_size ** 2,
            P2=32 * 3 * self.config.gaussian_stereo_block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=8,
            speckleWindowSize=50,
            speckleRange=1,
            preFilterCap=31,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        valid_disp = np.isfinite(disparity).astype(np.uint8)
        valid_disp[disparity < self.config.gaussian_stereo_min_disparity_px] = 0
        median_disp = cv2.medianBlur(np.where(valid_disp > 0, disparity, 0).astype(np.float32), 5)
        consistency = np.abs(disparity - median_disp).astype(np.float32)
        grad_x = cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = cv2.magnitude(grad_x, grad_y).astype(np.float32)
        fx = float(K[0, 0])
        depth = (fx * baseline / np.clip(disparity, 1e-6, None)).astype(np.float32)
        depth[(valid_disp == 0) | (~np.isfinite(depth))] = 0.0
        stable_mask = (
            (valid_disp > 0)
            & (consistency <= self.config.gaussian_stereo_consistency_threshold_px)
            & (texture >= self.config.gaussian_stereo_texture_threshold)
        )
        depth_edge = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
        stable_mask &= np.abs(depth_edge) <= self.config.gaussian_stereo_depth_edge_threshold
        confidence = np.clip(
            (texture / max(1.0, self.config.gaussian_stereo_texture_threshold))
            * np.exp(-consistency / max(1e-6, self.config.gaussian_stereo_consistency_threshold_px)),
            0.0,
            2.0,
        ).astype(np.float32)
        region_keep_mask = self._build_semantic_like_region_keep_mask(
            left_bgr=left_bgr,
            left_gray=left_gray,
            depth=depth,
            consistency=consistency,
            texture=texture,
            depth_edge=depth_edge,
            confidence=confidence,
        )
        stable_mask &= region_keep_mask
        confidence *= stable_mask.astype(np.float32)
        info = {
            "left_gray": left_gray,
            "left_rgb": left_bgr[:, :, ::-1].astype(np.uint8),
            "K": K.astype(np.float32),
            "baseline": float(baseline),
            "disparity": disparity.astype(np.float32),
            "depth": depth.astype(np.float32),
            "consistency": consistency.astype(np.float32),
            "texture": texture.astype(np.float32),
            "stable_mask": stable_mask.astype(np.uint8) * 255,
            "region_keep_mask": region_keep_mask.astype(np.uint8) * 255,
            "confidence": confidence.astype(np.float32),
        }
        self._handle_cache[cache_key] = info
        return info

    def _build_semantic_like_region_keep_mask(
        self,
        left_bgr: np.ndarray,
        left_gray: np.ndarray,
        depth: np.ndarray,
        consistency: np.ndarray,
        texture: np.ndarray,
        depth_edge: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        if not self.config.enable_semantic_like_region_filter:
            return np.ones(depth.shape, dtype=bool)
        h, w = depth.shape
        yy = np.arange(h, dtype=np.float32)[:, None] / max(1.0, float(h - 1))
        brightness = left_gray.astype(np.float32)
        far_depth = max(self.config.gaussian_region_far_depth_m, 0.6 * self.config.gaussian_stereo_max_depth_m)
        low_conf = confidence < self.config.gaussian_region_low_confidence_threshold
        invalid_depth = depth <= 0.05

        sky_mask = (
            (yy <= self.config.gaussian_region_sky_top_ratio)
            & (invalid_depth | (depth >= far_depth))
            & (texture <= 1.35 * self.config.gaussian_stereo_texture_threshold)
            & (brightness >= 96.0)
        )
        far_low_value = (
            (depth >= far_depth)
            & low_conf
        )
        vegetation_like = (
            (texture >= self.config.gaussian_region_vegetation_texture_scale * self.config.gaussian_stereo_texture_threshold)
            & (consistency >= 0.6 * self.config.gaussian_stereo_consistency_threshold_px)
            & low_conf
        )
        unstable_edges = (
            (np.abs(depth_edge) >= 0.8 * self.config.gaussian_stereo_depth_edge_threshold)
            & low_conf
        )
        dynamic_like = (
            (depth >= self.config.gaussian_region_dynamic_min_depth_m)
            & (depth <= self.config.gaussian_region_dynamic_max_depth_m)
            & low_conf
            & (consistency >= 0.7 * self.config.gaussian_stereo_consistency_threshold_px)
            & (texture >= 1.2 * self.config.gaussian_stereo_texture_threshold)
        )

        drop_mask = sky_mask | far_low_value | vegetation_like | unstable_edges | dynamic_like
        drop_u8 = (drop_mask.astype(np.uint8) * 255)
        drop_u8 = cv2.morphologyEx(drop_u8, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        drop_u8 = cv2.morphologyEx(drop_u8, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        keep = drop_u8 == 0
        return keep.astype(bool)

    def _quadtree_sample_pixels(
        self,
        depth_map: np.ndarray,
        texture: np.ndarray,
        stable_mask: np.ndarray,
        min_block: int,
        var_threshold: float,
    ) -> np.ndarray:
        h, w = depth_map.shape
        samples: list[tuple[int, int]] = []

        def recurse(x0: int, y0: int, x1: int, y1: int) -> None:
            region_mask = stable_mask[y0:y1, x0:x1]
            if region_mask.size == 0 or not np.any(region_mask):
                return
            local_depth = depth_map[y0:y1, x0:x1][region_mask]
            local_texture = texture[y0:y1, x0:x1][region_mask]
            split = False
            if (x1 - x0) > min_block and (y1 - y0) > min_block:
                if local_depth.size > 16:
                    depth_var = float(np.std(local_depth))
                    tex_var = float(np.std(local_texture))
                    split = depth_var > var_threshold or tex_var > (0.75 * self.config.gaussian_stereo_texture_threshold)
            if split:
                xm = (x0 + x1) // 2
                ym = (y0 + y1) // 2
                recurse(x0, y0, xm, ym)
                recurse(xm, y0, x1, ym)
                recurse(x0, ym, xm, y1)
                recurse(xm, ym, x1, y1)
                return

            ys, xs = np.where(region_mask)
            if xs.size == 0:
                return
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            coords = np.stack([xs + x0, ys + y0], axis=1).astype(np.float32)
            depth_values = depth_map[coords[:, 1].astype(np.int32), coords[:, 0].astype(np.int32)]
            valid = depth_values > 0.05
            if not np.any(valid):
                return
            coords = coords[valid]
            dists = np.linalg.norm(coords - center[None, :], axis=1)
            pick = coords[int(np.argmin(dists))]
            samples.append((int(pick[0]), int(pick[1])))

        recurse(0, 0, w, h)
        if not samples:
            return np.zeros((0, 2), dtype=np.int32)
        return np.unique(np.array(samples, dtype=np.int32), axis=0)

    def _load_stereo_params(self, image_path: Path) -> tuple[np.ndarray, float]:
        sequence_path = image_path.parents[1]
        cached = self._stereo_cache.get(sequence_path)
        if cached is not None:
            return cached
        calib_path = sequence_path / "calib.txt"
        if not calib_path.exists():
            K = np.array([[700.0, 0.0, 600.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            params = (K, 0.54)
            self._stereo_cache[sequence_path] = params
            return params
        lines = calib_path.read_text(encoding="utf-8").splitlines()
        p0 = np.array([float(item) for item in lines[0].split()[1:]], dtype=np.float32).reshape(3, 4)
        p1 = np.array([float(item) for item in lines[1].split()[1:]], dtype=np.float32).reshape(3, 4)
        K = p0[:, :3]
        baseline = abs(float(p1[0, 3] / p1[0, 0]))
        params = (K, baseline)
        self._stereo_cache[sequence_path] = params
        return params

    def _camera_to_world(self, camera_pose: np.ndarray, xyz_camera: np.ndarray) -> np.ndarray:
        xyz_h = np.concatenate([xyz_camera, np.ones((xyz_camera.shape[0], 1), dtype=np.float32)], axis=1)
        return (xyz_h @ camera_pose.T)[:, :3]

    def _estimate_stereo_surfel_axes(
        self,
        pose: np.ndarray,
        xyz_camera: np.ndarray,
        sample_xy: np.ndarray,
        disparity: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        baseline: float,
        stride: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        offsets_u = np.array([stride, 0], dtype=np.int32)
        offsets_v = np.array([0, stride], dtype=np.int32)

        def neighbor_points(offset: np.ndarray) -> np.ndarray:
            neighbor_xy = sample_xy + offset[None, :]
            neighbor_xy[:, 0] = np.clip(neighbor_xy[:, 0], 0, disparity.shape[1] - 1)
            neighbor_xy[:, 1] = np.clip(neighbor_xy[:, 1], 0, disparity.shape[0] - 1)
            disp_n = disparity[neighbor_xy[:, 1], neighbor_xy[:, 0]]
            depth_n = fx * baseline / np.clip(disp_n, 1e-6, None)
            x_n = (neighbor_xy[:, 0].astype(np.float32) - cx) * depth_n / fx
            y_n = (neighbor_xy[:, 1].astype(np.float32) - cy) * depth_n / fy
            return np.stack([x_n, y_n, depth_n.astype(np.float32)], axis=1)

        neighbor_u = neighbor_points(offsets_u)
        neighbor_v = neighbor_points(offsets_v)
        delta_u = neighbor_u - xyz_camera
        delta_v = neighbor_v - xyz_camera
        norm_u = np.linalg.norm(delta_u, axis=1)
        norm_v = np.linalg.norm(delta_v, axis=1)
        fallback_u = self.config.gaussian_stereo_stride_scale * stride * xyz_camera[:, 2] / fx
        fallback_v = self.config.gaussian_stereo_stride_scale * stride * xyz_camera[:, 2] / fy
        delta_u = np.where((norm_u > 1e-5)[:, None], delta_u, np.stack([fallback_u, np.zeros_like(fallback_u), np.zeros_like(fallback_u)], axis=1))
        delta_v = np.where((norm_v > 1e-5)[:, None], delta_v, np.stack([np.zeros_like(fallback_v), fallback_v, np.zeros_like(fallback_v)], axis=1))
        axis_u = delta_u @ pose[:3, :3].astype(np.float32).T
        axis_v = delta_v @ pose[:3, :3].astype(np.float32).T
        return axis_u.astype(np.float32), axis_v.astype(np.float32)

    def _camera_axes_to_world(
        self,
        pose: np.ndarray,
        xyz_camera: np.ndarray,
        scale_u: np.ndarray,
        scale_v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = pose[:3, :3].astype(np.float32)
        cam_u = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (xyz_camera.shape[0], 1))
        cam_v = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (xyz_camera.shape[0], 1))
        axis_u = (cam_u * scale_u[:, None]) @ rotation.T
        axis_v = (cam_v * scale_v[:, None]) @ rotation.T
        return axis_u.astype(np.float32), axis_v.astype(np.float32)

    def _default_axes_from_scale(self, scale: np.ndarray, axis: str) -> np.ndarray:
        basis = np.array([1.0, 0.0, 0.0] if axis == "u" else [0.0, 1.0, 0.0], dtype=np.float32)
        return (scale[:, None].astype(np.float32) * basis[None, :]).astype(np.float32)
