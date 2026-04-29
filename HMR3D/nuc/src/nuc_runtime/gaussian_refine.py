from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nuc_runtime.gaussian_builder import IncrementalGaussianBuilder
from nuc_runtime.local_tsdf import LocalSurfaceVolume
from nuc_runtime.gaussian_renderer import GaussianSplatRenderer, save_render_triplet


@dataclass
class GaussianRefineConfig:
    refine_steps: int = 400
    batch_size: int = 3
    image_scale: float = 1.0
    lr_xyz: float = 1e-3
    lr_scale: float = 5e-4
    lr_rotation: float = 5e-4
    lr_opacity: float = 5e-3
    lr_rgb: float = 1e-2
    lambda_depth: float = 0.2
    lambda_opacity: float = 1e-4
    lambda_scale: float = 1e-4
    lambda_rotation: float = 1e-4
    lambda_isotropic: float = 0.0
    lambda_tangent_max: float = 0.0
    lambda_normal_max: float = 0.0
    lambda_anisotropy_ratio: float = 0.0
    lambda_render_normal: float = 0.0
    lambda_surface_xyz: float = 0.0
    lambda_surface_scale: float = 0.0
    lambda_surface_rotation: float = 0.0
    lambda_anchor: float = 1e-4
    lambda_rgb_anchor: float = 0.0
    lambda_opacity_anchor: float = 0.0
    depth_conf_threshold: float = 1.0
    depth_alpha_threshold: float = 0.01
    prune_every: int = 50
    min_opacity: float = 0.02
    max_points: int = 4800
    preview_every: int = 100
    preview_frames: int = 3
    rgb_ssim_weight: float = 0.2
    background_gray: float = 0.0
    near_plane: float = 0.01
    far_plane: float = 60.0
    radius_clip: float = 0.0
    tile_size: int = 16
    rasterize_mode: str = "classic"
    init_max_tangent_scale: float = 2.0
    init_max_normal_scale: float = 0.15
    train_max_tangent_scale: float = 2.0
    train_max_normal_scale: float = 0.15
    max_anisotropy_ratio: float = 6.0
    normal_alpha_threshold: float = 0.05
    split_large_count: int = 0
    split_scale_threshold: float = 0.0
    split_anisotropy_threshold: float = 0.0
    split_offset_ratio: float = 0.35
    split_scale_shrink: float = 1.75
    enable_surface_prior: bool = False
    enable_surface_bootstrap: bool = False
    surface_prior_match_radius: float = 0.35
    surface_local_radius: float = 1.0
    surface_prior_blend: float = 0.35
    surface_prior_max_points: int = 1800
    surface_optimize_local_only: bool = False
    surface_unmatched_opacity_scale: float = 1.0
    surface_unmatched_isotropic_scale: float = 0.0
    random_seed: int = 13
    device: str = "cuda"


@dataclass
class RefineFrame:
    frame_idx: int
    image_path: str
    pose: np.ndarray
    viewmat: torch.Tensor
    K: torch.Tensor
    width: int
    height: int
    image_rgb: torch.Tensor
    depth: torch.Tensor
    depth_mask: torch.Tensor
    K_np: np.ndarray
    image_rgb_u8: np.ndarray
    depth_np: np.ndarray
    depth_conf_np: np.ndarray
    depth_mask_np: np.ndarray


def load_gaussian_refine_config(path: str | Path | None) -> GaussianRefineConfig:
    config = GaussianRefineConfig()
    if path is None:
        return config
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = data.get("refine", {})
    if not isinstance(section, dict):
        return config
    valid_keys = {field.name for field in fields(GaussianRefineConfig)}
    for key, value in section.items():
        if key in valid_keys:
            setattr(config, key, value)
    return config


def load_refine_frames(
    summary_json: str | Path,
    dense_geometry_npz: str | Path,
    image_root: str | Path | None,
    image_scale: float,
    depth_conf_threshold: float,
    device: str,
) -> list[RefineFrame]:
    summary_json = Path(summary_json).expanduser().resolve()
    dense_geometry_npz = Path(dense_geometry_npz).expanduser().resolve()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    dense = np.load(dense_geometry_npz)

    image_paths_raw = summary.get("image_paths", [])
    metadata = summary.get("metadata", {})
    frame_indices = list(metadata.get("frame_indices", []))
    poses = [np.asarray(item, dtype=np.float32) for item in metadata.get("cuvslam_poses", [])]
    depth = dense["depth"].astype(np.float32)
    depth_conf = dense["depth_conf"].astype(np.float32) if "depth_conf" in dense else np.ones_like(depth, dtype=np.float32)
    intrinsics = dense["intrinsic"].astype(np.float32) if "intrinsic" in dense else None

    count = min(len(image_paths_raw), len(poses), int(depth.shape[0]))
    if intrinsics is not None:
        count = min(count, int(intrinsics.shape[0]))
    if not frame_indices:
        frame_indices = list(range(count))
    else:
        count = min(count, len(frame_indices))

    image_root_path = Path(image_root).expanduser().resolve() if image_root else None
    frames: list[RefineFrame] = []
    scale = float(image_scale)
    scale = max(0.1, min(scale, 1.0))
    for local_idx in range(count):
        frame_idx = int(frame_indices[local_idx])
        image_path = _resolve_image_path(image_paths_raw[local_idx], frame_idx, image_root_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_map = depth[local_idx]
        conf_map = depth_conf[local_idx]
        base_h, base_w = depth_map.shape[:2]
        target_w = max(16, int(round(base_w * scale)))
        target_h = max(16, int(round(base_h * scale)))
        rgb_small = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        depth_small = cv2.resize(depth_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        conf_small = cv2.resize(conf_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if intrinsics is not None:
            K = intrinsics[local_idx].copy()
        else:
            fx = 700.0 * (target_w / max(1, rgb.shape[1]))
            fy = 700.0 * (target_h / max(1, rgb.shape[0]))
            K = np.array([[fx, 0.0, 0.5 * target_w], [0.0, fy, 0.5 * target_h], [0.0, 0.0, 1.0]], dtype=np.float32)
        K[0, :] *= target_w / max(1.0, float(base_w))
        K[1, :] *= target_h / max(1.0, float(base_h))

        pose = poses[local_idx].astype(np.float32)
        viewmat = np.linalg.inv(pose).astype(np.float32)
        depth_valid = np.isfinite(depth_small) & (depth_small > 0.05)
        depth_valid &= np.isfinite(conf_small) & (conf_small >= float(depth_conf_threshold))
        frames.append(
            RefineFrame(
                frame_idx=frame_idx,
                image_path=str(image_path),
                pose=pose,
                viewmat=torch.from_numpy(viewmat).to(device=device, dtype=torch.float32),
                K=torch.from_numpy(K).to(device=device, dtype=torch.float32),
                width=int(target_w),
                height=int(target_h),
                image_rgb=torch.from_numpy(rgb_small.astype(np.float32) / 255.0).to(device=device, dtype=torch.float32),
                depth=torch.from_numpy(depth_small).to(device=device, dtype=torch.float32),
                depth_mask=torch.from_numpy(depth_valid).to(device=device, dtype=torch.bool),
                K_np=K.astype(np.float32),
                image_rgb_u8=rgb_small.astype(np.uint8),
                depth_np=depth_small.astype(np.float32),
                depth_conf_np=conf_small.astype(np.float32),
                depth_mask_np=depth_valid.astype(bool),
            )
        )
    return frames


def build_surface_prior_bundle(
    frames: list[RefineFrame],
    runtime_memory_config: Any,
    max_points: int,
) -> dict[str, np.ndarray]:
    if not frames:
        return _empty_bundle()
    extract_points = max(1, min(int(max_points), int(runtime_memory_config.gaussian_local_volume_extract_points)))
    min_good = max(64, min(256, extract_points // 6))
    candidates = [
        runtime_memory_config,
        replace(
            runtime_memory_config,
            gaussian_local_volume_voxel_size=max(float(runtime_memory_config.gaussian_local_volume_voxel_size), 0.18),
            gaussian_structured_anchor_voxel_size=max(float(runtime_memory_config.gaussian_structured_anchor_voxel_size), 0.32),
            gaussian_local_volume_min_observations=1,
        ),
        replace(
            runtime_memory_config,
            gaussian_local_volume_voxel_size=max(float(runtime_memory_config.gaussian_local_volume_voxel_size), 0.3),
            gaussian_structured_anchor_voxel_size=max(float(runtime_memory_config.gaussian_structured_anchor_voxel_size), 0.45),
            gaussian_local_volume_min_observations=1,
        ),
    ]

    surface_candidates: list[dict[str, np.ndarray]] = []
    for config_variant in candidates:
        volume = LocalSurfaceVolume(config_variant)
        for frame in frames:
            volume.fuse_frame(
                pose=frame.pose.astype(np.float32),
                depth=frame.depth_np.astype(np.float32),
                confidence=frame.depth_conf_np.astype(np.float32),
                rgb=frame.image_rgb_u8.astype(np.uint8),
                K=frame.K_np.astype(np.float32),
                stable_mask=frame.depth_mask_np.astype(bool),
            )
        for visible_only in (True, False):
            kwargs = {}
            if visible_only:
                kwargs = {
                    "pose": frames[-1].pose.astype(np.float32),
                    "depth_map": frames[-1].depth_np.astype(np.float32),
                    "K": frames[-1].K_np.astype(np.float32),
                }
            surface = volume.extract_gaussians(
                max_points=extract_points,
                default_scale=float(config_variant.gaussian_default_scale),
                **kwargs,
            )
            count = int(surface["xyz"].shape[0])
            if count > 0:
                surface["provenance"] = np.full((count,), 5, dtype=np.int8)
                surface["confidence"] = (1.0 / (1.0 + np.clip(surface["unstable"], 0.0, None))).astype(np.float32)
                surface_candidates.append(surface)
            if count >= min_good:
                break
        if any(int(item["xyz"].shape[0]) >= min_good for item in surface_candidates):
            break

    if not surface_candidates:
        return _empty_bundle()
    return _merge_surface_candidates(
        surface_candidates,
        max_points=extract_points,
        voxel_size=max(float(runtime_memory_config.gaussian_structured_anchor_voxel_size), 0.18),
    )


class LocalGaussianRefiner:
    def __init__(
        self,
        bundle: dict[str, np.ndarray],
        runtime_output_config: Any,
        refine_config: GaussianRefineConfig,
        surface_bundle: dict[str, np.ndarray] | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Gaussian refine requires CUDA")
        try:
            import gsplat  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("gsplat is required for Gaussian refine") from exc

        self.gsplat = gsplat
        self.refine_config = refine_config
        self.device = torch.device(refine_config.device)
        self.renderer = GaussianSplatRenderer(runtime_output_config)
        if surface_bundle is not None and bool(refine_config.enable_surface_prior) and bool(refine_config.enable_surface_bootstrap):
            bundle = _merge_surface_bootstrap(
                renderer=self.renderer,
                seed_bundle=bundle,
                surface_bundle=surface_bundle,
                max_points=int(refine_config.max_points),
            )
        normalized = self.renderer._normalize_external_bundle(bundle)
        scales0, quats0 = self.renderer._axes_to_gsplat_params(
            axis_u=normalized["axis_u"],
            axis_v=normalized["axis_v"],
            scale_base=normalized["scale"],
        )
        rgb0 = np.clip(normalized["rgb"].astype(np.float32) / 255.0, 1e-4, 1.0 - 1e-4)
        opacity0 = np.clip(normalized["opacity"].astype(np.float32), 1e-4, 1.0 - 1e-4)
        scales0 = np.clip(scales0.astype(np.float32), 1e-4, None)
        scales0[:, :2] = np.minimum(scales0[:, :2], float(refine_config.init_max_tangent_scale))
        scales0[:, 2] = np.minimum(scales0[:, 2], float(refine_config.init_max_normal_scale))

        self.source = normalized["source"].astype(np.int8)
        self.recentness = normalized["recentness"].astype(np.float32)
        self.unstable = normalized["unstable"].astype(np.float32)
        self.confidence = (1.0 / (1.0 + np.clip(self.unstable, 0.0, None))).astype(np.float32)
        self.provenance = normalized["provenance"].astype(np.int8)

        self.xyz_anchor = torch.from_numpy(normalized["xyz"].astype(np.float32)).to(device=self.device, dtype=torch.float32)
        self.scale_anchor = torch.from_numpy(scales0).to(device=self.device, dtype=torch.float32)
        self.quat_anchor = torch.from_numpy(quats0.astype(np.float32)).to(device=self.device, dtype=torch.float32)
        self.rgb_anchor = torch.from_numpy(rgb0).to(device=self.device, dtype=torch.float32)
        self.opacity_anchor = torch.from_numpy(opacity0).to(device=self.device, dtype=torch.float32)

        self.xyz = torch.nn.Parameter(self.xyz_anchor.clone())
        self.log_scales = torch.nn.Parameter(torch.log(self.scale_anchor.clone()))
        self.raw_quats = torch.nn.Parameter(self.quat_anchor.clone())
        self.opacity_logits = torch.nn.Parameter(_inverse_sigmoid(self.opacity_anchor))
        self.rgb_logits = torch.nn.Parameter(_inverse_sigmoid(self.rgb_anchor))

        self.surface_match_mask = torch.zeros((self.xyz_anchor.shape[0],), device=self.device, dtype=torch.bool)
        self.surface_local_mask = torch.zeros((self.xyz_anchor.shape[0],), device=self.device, dtype=torch.bool)
        self.surface_xyz_anchor = self.xyz_anchor.clone()
        self.surface_scale_anchor = self.scale_anchor.clone()
        self.surface_quat_anchor = self.quat_anchor.clone()
        self.surface_prior_stats = {
            "enabled": bool(refine_config.enable_surface_prior),
            "surface_point_count": 0,
            "matched_count": 0,
            "match_ratio": 0.0,
            "match_radius": float(refine_config.surface_prior_match_radius),
            "local_count": 0,
            "local_ratio": 0.0,
            "local_radius": float(refine_config.surface_local_radius),
        }
        if surface_bundle is not None and bool(refine_config.enable_surface_prior):
            self._attach_surface_prior(surface_bundle)
        self._stabilize_unmatched_points()

        self._rng = np.random.default_rng(int(refine_config.random_seed))
        self._build_optimizer()

    def _build_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.xyz], "lr": float(self.refine_config.lr_xyz)},
                {"params": [self.log_scales], "lr": float(self.refine_config.lr_scale)},
                {"params": [self.raw_quats], "lr": float(self.refine_config.lr_rotation)},
                {"params": [self.opacity_logits], "lr": float(self.refine_config.lr_opacity)},
                {"params": [self.rgb_logits], "lr": float(self.refine_config.lr_rgb)},
            ]
        )

    def _attach_surface_prior(self, surface_bundle: dict[str, np.ndarray]) -> None:
        if surface_bundle.get("xyz", np.zeros((0, 3), dtype=np.float32)).shape[0] <= 0:
            return
        normalized = self.renderer._normalize_external_bundle(surface_bundle)
        surface_scales, surface_quats = self.renderer._axes_to_gsplat_params(
            axis_u=normalized["axis_u"],
            axis_v=normalized["axis_v"],
            scale_base=normalized["scale"],
        )
        surface_xyz = torch.from_numpy(normalized["xyz"].astype(np.float32)).to(device=self.device, dtype=torch.float32)
        surface_scale_anchor = torch.from_numpy(np.clip(surface_scales.astype(np.float32), 1e-4, None)).to(
            device=self.device, dtype=torch.float32
        )
        surface_quat_anchor = torch.from_numpy(surface_quats.astype(np.float32)).to(device=self.device, dtype=torch.float32)
        dist = torch.cdist(self.xyz_anchor, surface_xyz)
        nearest_dist, nearest_idx = torch.min(dist, dim=1)
        match_mask = nearest_dist <= float(self.refine_config.surface_prior_match_radius)
        local_mask = nearest_dist <= float(self.refine_config.surface_local_radius)
        if not torch.any(match_mask):
            self.surface_prior_stats = {
                "enabled": True,
                "surface_point_count": int(surface_xyz.shape[0]),
                "matched_count": 0,
                "match_ratio": 0.0,
                "match_radius": float(self.refine_config.surface_prior_match_radius),
                "local_count": int(local_mask.sum().item()),
                "local_ratio": float(local_mask.float().mean().item()),
                "local_radius": float(self.refine_config.surface_local_radius),
            }
            self.surface_local_mask = local_mask
            return
        self.surface_match_mask = match_mask
        self.surface_local_mask = local_mask
        matched_idx = nearest_idx[match_mask]
        blend = float(np.clip(self.refine_config.surface_prior_blend, 0.0, 1.0))
        self.surface_xyz_anchor = self.xyz_anchor.clone()
        self.surface_xyz_anchor[match_mask] = (
            (1.0 - blend) * self.xyz_anchor[match_mask] + blend * surface_xyz[matched_idx]
        )
        self.surface_scale_anchor = self.scale_anchor.clone()
        self.surface_scale_anchor[match_mask] = (
            (1.0 - blend) * self.scale_anchor[match_mask] + blend * surface_scale_anchor[matched_idx]
        )
        self.surface_quat_anchor = self.quat_anchor.clone()
        self.surface_quat_anchor[match_mask] = surface_quat_anchor[matched_idx]
        self.surface_prior_stats = {
            "enabled": True,
            "surface_point_count": int(surface_xyz.shape[0]),
            "matched_count": int(match_mask.sum().item()),
            "match_ratio": float(match_mask.float().mean().item()),
            "match_radius": float(self.refine_config.surface_prior_match_radius),
            "local_count": int(local_mask.sum().item()),
            "local_ratio": float(local_mask.float().mean().item()),
            "local_radius": float(self.refine_config.surface_local_radius),
        }

    def _stabilize_unmatched_points(self) -> None:
        if not bool(self.refine_config.enable_surface_prior):
            return
        if self.surface_local_mask.numel() <= 0:
            return
        unmatched_mask = ~self.surface_local_mask
        if not torch.any(unmatched_mask):
            return
        opacity_scale = float(np.clip(self.refine_config.surface_unmatched_opacity_scale, 0.0, 1.0))
        isotropic_scale = float(self.refine_config.surface_unmatched_isotropic_scale)
        with torch.no_grad():
            if opacity_scale < 1.0:
                opacity = torch.sigmoid(self.opacity_logits.detach())
                opacity[unmatched_mask] = torch.clamp(opacity[unmatched_mask] * opacity_scale, 1e-4, 1.0 - 1e-4)
                self.opacity_logits.data.copy_(_inverse_sigmoid(opacity))
                self.opacity_anchor[unmatched_mask] = opacity[unmatched_mask]
            if isotropic_scale > 0.0:
                scales = torch.exp(self.log_scales.detach())
                clipped = torch.clamp(
                    torch.full_like(scales[unmatched_mask], isotropic_scale),
                    min=1e-4,
                )
                clipped[:, 2] = torch.clamp(clipped[:, 2] * 0.5, min=1e-4)
                scales[unmatched_mask] = clipped
                self.log_scales.data.copy_(torch.log(scales))
                self.scale_anchor[unmatched_mask] = scales[unmatched_mask]

    def _mask_nonlocal_gradients(self) -> None:
        if not bool(self.refine_config.surface_optimize_local_only):
            return
        if self.surface_local_mask.numel() <= 0:
            return
        train_mask = self.surface_local_mask
        freeze_mask = ~train_mask
        if not torch.any(freeze_mask):
            return
        for param in (self.xyz, self.log_scales, self.raw_quats, self.opacity_logits, self.rgb_logits):
            if param.grad is not None:
                param.grad[freeze_mask] = 0.0

    def run(
        self,
        frames: list[RefineFrame],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        if not frames:
            raise RuntimeError("No refine frames available")
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_dir = output_dir / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)

        preview_indices = _preview_indices(len(frames), int(self.refine_config.preview_frames))
        metrics: list[dict[str, Any]] = []
        self._save_preview(preview_dir / "step_0000", frames, preview_indices)

        for step in range(1, max(1, int(self.refine_config.refine_steps)) + 1):
            batch_indices = self._sample_batch_indices(len(frames))
            total_loss = torch.zeros((), device=self.device, dtype=torch.float32)
            batch_terms = {
                "rgb_l1": 0.0,
                "rgb_dssim": 0.0,
                "depth": 0.0,
                "normal": 0.0,
                "surface_xyz": 0.0,
                "surface_scale": 0.0,
                "surface_rotation": 0.0,
            }
            self.optimizer.zero_grad(set_to_none=True)
            for frame_idx in batch_indices:
                frame = frames[int(frame_idx)]
                render_rgb, render_depth, render_alpha, render_normal = self._render_frame(frame)
                rgb_l1 = F.l1_loss(render_rgb, frame.image_rgb)
                rgb_dssim = 1.0 - _ssim_rgb(render_rgb, frame.image_rgb)
                rgb_loss = (1.0 - float(self.refine_config.rgb_ssim_weight)) * rgb_l1
                rgb_loss = rgb_loss + float(self.refine_config.rgb_ssim_weight) * rgb_dssim

                depth_mask = frame.depth_mask
                depth_mask = depth_mask & (render_alpha > float(self.refine_config.depth_alpha_threshold))
                if torch.any(depth_mask):
                    depth_loss = F.smooth_l1_loss(render_depth[depth_mask], frame.depth[depth_mask], beta=0.5)
                else:
                    depth_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                if float(self.refine_config.lambda_render_normal) > 0.0:
                    depth_normal, normal_mask = _depth_to_normal_map(render_depth, frame.K)
                    normal_mask = normal_mask & (
                        render_alpha > float(self.refine_config.normal_alpha_threshold)
                    )
                    if torch.any(normal_mask):
                        normal_dot = torch.sum(depth_normal * render_normal, dim=-1).abs()
                        normal_loss = (
                            render_alpha[normal_mask] * (1.0 - torch.clamp(normal_dot[normal_mask], 0.0, 1.0))
                        ).mean()
                    else:
                        normal_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                else:
                    normal_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                loss = rgb_loss + float(self.refine_config.lambda_depth) * depth_loss
                loss = loss + float(self.refine_config.lambda_render_normal) * normal_loss
                total_loss = total_loss + loss
                batch_terms["rgb_l1"] += float(rgb_l1.detach().cpu())
                batch_terms["rgb_dssim"] += float(rgb_dssim.detach().cpu())
                batch_terms["depth"] += float(depth_loss.detach().cpu())
                batch_terms["normal"] += float(normal_loss.detach().cpu())

            rgb_current = torch.sigmoid(self.rgb_logits)
            opacity_current = torch.sigmoid(self.opacity_logits)
            anchor_loss = F.mse_loss(self.xyz, self.xyz_anchor)
            scale_loss = F.mse_loss(self.log_scales, torch.log(self.scale_anchor))
            rotation_loss = _quaternion_anchor_loss(self._current_quats(), self.quat_anchor)
            opacity_reg = opacity_current.mean()
            rgb_anchor_loss = F.mse_loss(rgb_current, self.rgb_anchor)
            opacity_anchor_loss = F.mse_loss(opacity_current, self.opacity_anchor)
            scales_current = self._current_scales()
            tangent_scales = scales_current[:, :2]
            normal_scale = scales_current[:, 2]
            isotropic_loss = (tangent_scales - tangent_scales.mean(dim=1, keepdim=True)).abs().mean()
            tangent_max_loss = F.relu(tangent_scales - float(self.refine_config.train_max_tangent_scale)).mean()
            normal_max_loss = F.relu(normal_scale - float(self.refine_config.train_max_normal_scale)).mean()
            anisotropy_ratio = tangent_scales.max(dim=1).values / torch.clamp(
                tangent_scales.min(dim=1).values,
                min=1e-4,
            )
            anisotropy_ratio_loss = F.relu(anisotropy_ratio - float(self.refine_config.max_anisotropy_ratio)).mean()
            if torch.any(self.surface_match_mask):
                surface_xyz_loss = F.smooth_l1_loss(
                    self.xyz[self.surface_match_mask],
                    self.surface_xyz_anchor[self.surface_match_mask],
                    beta=0.1,
                )
                surface_scale_loss = F.mse_loss(
                    self.log_scales[self.surface_match_mask],
                    torch.log(self.surface_scale_anchor[self.surface_match_mask]),
                )
                surface_rotation_loss = _quaternion_anchor_loss(
                    self._current_quats()[self.surface_match_mask],
                    self.surface_quat_anchor[self.surface_match_mask],
                )
            else:
                surface_xyz_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                surface_scale_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                surface_rotation_loss = torch.zeros((), device=self.device, dtype=torch.float32)

            total_loss = total_loss / max(1, len(batch_indices))
            total_loss = total_loss + float(self.refine_config.lambda_anchor) * anchor_loss
            total_loss = total_loss + float(self.refine_config.lambda_scale) * scale_loss
            total_loss = total_loss + float(self.refine_config.lambda_rotation) * rotation_loss
            total_loss = total_loss + float(self.refine_config.lambda_isotropic) * isotropic_loss
            total_loss = total_loss + float(self.refine_config.lambda_tangent_max) * tangent_max_loss
            total_loss = total_loss + float(self.refine_config.lambda_normal_max) * normal_max_loss
            total_loss = total_loss + float(self.refine_config.lambda_anisotropy_ratio) * anisotropy_ratio_loss
            total_loss = total_loss + float(self.refine_config.lambda_surface_xyz) * surface_xyz_loss
            total_loss = total_loss + float(self.refine_config.lambda_surface_scale) * surface_scale_loss
            total_loss = total_loss + float(self.refine_config.lambda_surface_rotation) * surface_rotation_loss
            total_loss = total_loss + float(self.refine_config.lambda_opacity) * opacity_reg
            total_loss = total_loss + float(self.refine_config.lambda_rgb_anchor) * rgb_anchor_loss
            total_loss = total_loss + float(self.refine_config.lambda_opacity_anchor) * opacity_anchor_loss
            total_loss.backward()
            self._mask_nonlocal_gradients()
            self.optimizer.step()
            self._clamp_scales_()

            batch_terms["surface_xyz"] = float(surface_xyz_loss.detach().cpu())
            batch_terms["surface_scale"] = float(surface_scale_loss.detach().cpu())
            batch_terms["surface_rotation"] = float(surface_rotation_loss.detach().cpu())

            metrics.append(
                {
                    "step": step,
                    "loss": round(float(total_loss.detach().cpu()), 6),
                    "rgb_l1": round(batch_terms["rgb_l1"] / max(1, len(batch_indices)), 6),
                    "rgb_dssim": round(batch_terms["rgb_dssim"] / max(1, len(batch_indices)), 6),
                    "depth": round(batch_terms["depth"] / max(1, len(batch_indices)), 6),
                    "normal": round(batch_terms["normal"] / max(1, len(batch_indices)), 6),
                    "surface_xyz": round(batch_terms["surface_xyz"], 6),
                    "surface_scale": round(batch_terms["surface_scale"], 6),
                    "surface_rotation": round(batch_terms["surface_rotation"], 6),
                    "point_count": int(self.xyz.shape[0]),
                    "mean_opacity": round(float(torch.sigmoid(self.opacity_logits).mean().detach().cpu()), 6),
                    "rotation_deg": round(float(_quaternion_delta_deg(self._current_quats().detach(), self.quat_anchor.detach()).mean().cpu()), 6),
                    "isotropic": round(float(isotropic_loss.detach().cpu()), 6),
                    "tangent_max": round(float(tangent_max_loss.detach().cpu()), 6),
                    "normal_max": round(float(normal_max_loss.detach().cpu()), 6),
                    "anisotropy_ratio": round(float(anisotropy_ratio.mean().detach().cpu()), 6),
                }
            )

            if self.refine_config.prune_every > 0 and step % int(self.refine_config.prune_every) == 0:
                self._prune()
            if self.refine_config.preview_every > 0 and step % int(self.refine_config.preview_every) == 0:
                self._save_preview(preview_dir / f"step_{step:04d}", frames, preview_indices)

        final_bundle = self.export_bundle()
        summary = {
            "metrics": metrics,
            "point_count": int(final_bundle["xyz"].shape[0]),
            "preview_dir": str(preview_dir),
            "surface_prior": dict(self.surface_prior_stats),
        }
        return {"bundle": final_bundle, "summary": summary}

    def export_bundle(self) -> dict[str, np.ndarray]:
        xyz = self.xyz.detach().cpu().numpy().astype(np.float32)
        scales_xyz = torch.exp(self.log_scales).detach()
        quats = self._current_quats().detach()
        axis_u, axis_v = _quats_scales_to_axes(quats, scales_xyz)
        scales_xyz_np = scales_xyz.cpu().numpy().astype(np.float32)
        scale = np.clip(0.5 * (scales_xyz_np[:, 0] + scales_xyz_np[:, 1]), 1e-4, None).astype(np.float32)
        rgb = np.clip(torch.sigmoid(self.rgb_logits).detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
        opacity = np.clip(torch.sigmoid(self.opacity_logits).detach().cpu().numpy(), 0.02, 0.999).astype(np.float32)
        bundle = {
            "xyz": xyz,
            "rgb": rgb,
            "scale": scale,
            "opacity": opacity,
            "axis_u": axis_u.detach().cpu().numpy().astype(np.float32),
            "axis_v": axis_v.detach().cpu().numpy().astype(np.float32),
            "rotation": quats.cpu().numpy().astype(np.float32),
            "scale_xyz": scales_xyz_np,
            "confidence": self.confidence.astype(np.float32),
            "unstable": np.clip(self.unstable * 0.8, 0.05, 1.5).astype(np.float32),
            "recentness": self.recentness.astype(np.float32),
            "source": self.source.astype(np.int8),
            "provenance": self.provenance.astype(np.int8),
        }
        return _split_large_gaussians(bundle, self.refine_config)

    def _render_frame(self, frame: RefineFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        colors, alphas, _ = self.gsplat.rasterization(
            means=self.xyz,
            quats=self._current_quats(),
            scales=self._current_scales(),
            opacities=torch.sigmoid(self.opacity_logits),
            colors=torch.sigmoid(self.rgb_logits),
            viewmats=frame.viewmat.unsqueeze(0),
            Ks=frame.K.unsqueeze(0),
            width=int(frame.width),
            height=int(frame.height),
            near_plane=float(self.refine_config.near_plane),
            far_plane=float(self.refine_config.far_plane),
            radius_clip=float(self.refine_config.radius_clip),
            packed=True,
            tile_size=int(self.refine_config.tile_size),
            render_mode="RGB+ED",
            rasterize_mode=str(self.refine_config.rasterize_mode),
        )
        rgb, depth = _split_rgb_depth(colors)
        alpha = _normalize_alpha(alphas)
        render_normal = self._render_normal_map(frame)
        return rgb, depth, alpha, render_normal

    def _render_normal_map(self, frame: RefineFrame) -> torch.Tensor:
        normals_world = _quaternion_to_rotation_matrix(self._current_quats())[:, :, 2]
        rot_wc = frame.viewmat[:3, :3]
        normals_camera = normals_world @ rot_wc.T
        normals_camera = normals_camera / torch.clamp(
            torch.linalg.norm(normals_camera, dim=1, keepdim=True),
            min=1e-6,
        )
        normal_colors = 0.5 * (normals_camera + 1.0)
        render_normal_rgb, _, _ = self.gsplat.rasterization(
            means=self.xyz,
            quats=self._current_quats(),
            scales=self._current_scales(),
            opacities=torch.sigmoid(self.opacity_logits),
            colors=normal_colors,
            viewmats=frame.viewmat.unsqueeze(0),
            Ks=frame.K.unsqueeze(0),
            width=int(frame.width),
            height=int(frame.height),
            near_plane=float(self.refine_config.near_plane),
            far_plane=float(self.refine_config.far_plane),
            radius_clip=float(self.refine_config.radius_clip),
            packed=True,
            tile_size=int(self.refine_config.tile_size),
            render_mode="RGB",
            rasterize_mode=str(self.refine_config.rasterize_mode),
        )
        if render_normal_rgb.ndim == 4:
            render_normal_rgb = render_normal_rgb[0]
        if render_normal_rgb.ndim == 3 and render_normal_rgb.shape[0] == 3:
            render_normal_rgb = torch.permute(render_normal_rgb, (1, 2, 0))
        render_normal = render_normal_rgb[..., :3] * 2.0 - 1.0
        render_normal = render_normal / torch.clamp(
            torch.linalg.norm(render_normal, dim=-1, keepdim=True),
            min=1e-6,
        )
        return render_normal

    def _current_scales(self) -> torch.Tensor:
        return torch.exp(self.log_scales)

    def _current_quats(self) -> torch.Tensor:
        return self.raw_quats / torch.clamp(torch.linalg.norm(self.raw_quats, dim=1, keepdim=True), min=1e-6)

    def _clamp_scales_(self) -> None:
        with torch.no_grad():
            scales = torch.exp(self.log_scales.detach())
            scales[:, :2].clamp_(1e-4, float(self.refine_config.train_max_tangent_scale))
            scales[:, 2].clamp_(1e-4, float(self.refine_config.train_max_normal_scale))
            self.log_scales.data.copy_(torch.log(scales))

    def _sample_batch_indices(self, frame_count: int) -> np.ndarray:
        batch_size = min(frame_count, max(1, int(self.refine_config.batch_size)))
        return self._rng.choice(frame_count, size=batch_size, replace=False)

    def _prune(self) -> None:
        opacity = torch.sigmoid(self.opacity_logits).detach()
        scales_xyz = torch.exp(self.log_scales).detach()
        keep = torch.isfinite(self.xyz.detach()).all(dim=1)
        keep &= torch.isfinite(scales_xyz).all(dim=1)
        keep &= opacity >= float(self.refine_config.min_opacity)
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if keep_idx.numel() == 0:
            return
        if keep_idx.numel() > int(self.refine_config.max_points):
            score = opacity[keep_idx]
            top_idx = torch.topk(score, k=int(self.refine_config.max_points), largest=True).indices
            keep_idx = keep_idx[top_idx]
        if keep_idx.numel() == self.xyz.shape[0]:
            return
        keep_idx_np = keep_idx.detach().cpu().numpy().astype(np.int64)

        self.xyz = torch.nn.Parameter(self.xyz.detach()[keep_idx].clone())
        self.log_scales = torch.nn.Parameter(self.log_scales.detach()[keep_idx].clone())
        self.raw_quats = torch.nn.Parameter(self.raw_quats.detach()[keep_idx].clone())
        self.opacity_logits = torch.nn.Parameter(self.opacity_logits.detach()[keep_idx].clone())
        self.rgb_logits = torch.nn.Parameter(self.rgb_logits.detach()[keep_idx].clone())
        self.xyz_anchor = self.xyz_anchor[keep_idx].clone()
        self.scale_anchor = self.scale_anchor[keep_idx].clone()
        self.quat_anchor = self.quat_anchor[keep_idx].clone()
        self.surface_match_mask = self.surface_match_mask[keep_idx].clone()
        self.surface_local_mask = self.surface_local_mask[keep_idx].clone()
        self.surface_xyz_anchor = self.surface_xyz_anchor[keep_idx].clone()
        self.surface_scale_anchor = self.surface_scale_anchor[keep_idx].clone()
        self.surface_quat_anchor = self.surface_quat_anchor[keep_idx].clone()
        self.rgb_anchor = self.rgb_anchor[keep_idx].clone()
        self.opacity_anchor = self.opacity_anchor[keep_idx].clone()
        self.source = self.source[keep_idx_np]
        self.confidence = self.confidence[keep_idx_np]
        self.provenance = self.provenance[keep_idx_np]
        self.recentness = self.recentness[keep_idx_np]
        self.unstable = self.unstable[keep_idx_np]
        self._build_optimizer()

    def _save_preview(
        self,
        output_dir: Path,
        frames: list[RefineFrame],
        preview_indices: list[int],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for idx in preview_indices:
                frame = frames[idx]
                render_rgb, _, _, _ = self._render_frame(frame)
                target_u8 = np.clip(frame.image_rgb.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
                render_u8 = np.clip(render_rgb.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(np.uint8)
                save_render_triplet(output_dir, frame.frame_idx, target_u8, render_u8)


def write_refined_outputs(
    output_dir: str | Path,
    bundle: dict[str, np.ndarray],
    runtime_memory_config: Any,
) -> dict[str, str]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "gaussians_refined.npz"
    ply_path = output_dir / "gaussians_refined.ply"
    coarse_npz_path = output_dir / "gaussians_refined_coarse.npz"
    coarse_ply_path = output_dir / "gaussians_refined_coarse.ply"
    np.savez_compressed(npz_path, **bundle)

    builder = IncrementalGaussianBuilder(output_dir=output_dir, config=runtime_memory_config)
    builder._write_ply(
        ply_path,
        bundle["xyz"],
        bundle["rgb"],
        bundle["scale"],
        bundle["opacity"],
        bundle["axis_u"],
        bundle["axis_v"],
    )
    coarse = builder._build_coarse_bundle(
        xyz=bundle["xyz"],
        rgb=bundle["rgb"],
        scale=bundle["scale"],
        opacity=bundle["opacity"],
        axis_u=bundle["axis_u"],
        axis_v=bundle["axis_v"],
        unstable=bundle["unstable"],
        recentness=bundle["recentness"],
        source=bundle["source"],
    )
    np.savez_compressed(coarse_npz_path, **coarse)
    builder._write_ply(
        coarse_ply_path,
        coarse["xyz"],
        coarse["rgb"],
        coarse["scale"],
        coarse["opacity"],
        coarse["axis_u"],
        coarse["axis_v"],
    )
    return {
        "npz_path": str(npz_path),
        "ply_path": str(ply_path),
        "coarse_npz_path": str(coarse_npz_path),
        "coarse_ply_path": str(coarse_ply_path),
    }


def _inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-4, 1.0 - 1e-4)
    return torch.log(x / (1.0 - x))


def _quaternion_anchor_loss(current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    dots = torch.sum(current * anchor, dim=1).abs()
    return torch.mean(1.0 - torch.clamp(dots, 0.0, 1.0))


def _quaternion_delta_deg(current: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    dots = torch.sum(current * anchor, dim=1).abs()
    dots = torch.clamp(dots, 0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dots))


def _quats_scales_to_axes(quats: torch.Tensor, scales_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rot = _quaternion_to_rotation_matrix(quats)
    axis_u = rot[:, :, 0] * scales_xyz[:, 0:1]
    axis_v = rot[:, :, 1] * scales_xyz[:, 1:2]
    return axis_u, axis_v


def _quaternion_to_rotation_matrix(quats: torch.Tensor) -> torch.Tensor:
    quats = quats / torch.clamp(torch.linalg.norm(quats, dim=1, keepdim=True), min=1e-6)
    w, x, y, z = torch.unbind(quats, dim=1)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    row0 = torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=1)
    row1 = torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=1)
    row2 = torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=1)
    return torch.stack([row0, row1, row2], dim=1)


def _resolve_image_path(path_str: str, frame_idx: int, image_root: Path | None) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if image_root is not None:
        fallback = image_root / "images" / f"{frame_idx:06d}.png"
        if fallback.exists():
            return fallback.resolve()
    raise FileNotFoundError(f"Missing image for frame {frame_idx}: {path_str}")


def _split_rgb_depth(render_colors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if render_colors.ndim == 4:
        render_colors = render_colors[0]
    if render_colors.ndim == 3 and render_colors.shape[0] in {3, 4}:
        render_colors = torch.permute(render_colors, (1, 2, 0))
    if render_colors.shape[-1] < 4:
        raise RuntimeError("Expected gsplat render_mode RGB+ED to return 4 channels")
    rgb = torch.clamp(render_colors[..., :3], 0.0, 1.0)
    depth = torch.clamp(render_colors[..., 3], min=0.0)
    return rgb, depth


def _normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim == 4:
        alpha = alpha[0]
    if alpha.ndim == 3 and alpha.shape[-1] == 1:
        alpha = alpha[..., 0]
    if alpha.ndim == 3 and alpha.shape[0] == 1:
        alpha = alpha[0]
    return torch.clamp(alpha, 0.0, 1.0)


def _depth_to_normal_map(depth: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = int(depth.shape[0]), int(depth.shape[1])
    device = depth.device
    dtype = depth.dtype
    normals = torch.zeros((height, width, 3), device=device, dtype=dtype)
    valid = torch.isfinite(depth) & (depth > 0.05)
    if height < 3 or width < 3:
        return normals, torch.zeros_like(valid)

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = torch.clamp(K[0, 0], min=1e-6)
    fy = torch.clamp(K[1, 1], min=1e-6)
    cx = K[0, 2]
    cy = K[1, 2]

    z = depth
    x = (xx - cx) / fx * z
    y = (yy - cy) / fy * z
    points = torch.stack([x, y, z], dim=-1)

    dx = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    dy = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    center_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
    )
    inner = torch.cross(dx, dy, dim=-1)
    inner = inner / torch.clamp(torch.linalg.norm(inner, dim=-1, keepdim=True), min=1e-6)
    normals[1:-1, 1:-1, :] = inner
    mask = torch.zeros_like(valid)
    mask[1:-1, 1:-1] = center_valid & torch.isfinite(inner).all(dim=-1)
    return normals, mask


def _split_large_gaussians(bundle: dict[str, np.ndarray], config: GaussianRefineConfig) -> dict[str, np.ndarray]:
    count = int(bundle["xyz"].shape[0])
    split_count = max(0, int(config.split_large_count))
    if count <= 0 or split_count <= 0:
        return bundle
    if "rotation" not in bundle or "scale_xyz" not in bundle:
        return bundle

    scales = bundle["scale_xyz"].astype(np.float32)
    tang = scales[:, :2]
    max_tangent = np.max(tang, axis=1)
    min_tangent = np.maximum(np.min(tang, axis=1), 1e-6)
    ratio = max_tangent / min_tangent
    eligible = max_tangent >= float(config.split_scale_threshold)
    eligible &= ratio >= float(config.split_anisotropy_threshold)
    eligible_idx = np.nonzero(eligible)[0]
    if eligible_idx.size == 0:
        return bundle

    confidence = bundle.get("confidence", np.ones((count,), dtype=np.float32)).astype(np.float32)
    opacity = bundle["opacity"].astype(np.float32)
    score = max_tangent * np.maximum(ratio, 1.0) * np.clip(opacity, 0.05, 1.0) * np.clip(confidence, 0.05, 2.0)
    ranked = eligible_idx[np.argsort(score[eligible_idx])[::-1]]
    selected = ranked[: min(split_count, ranked.size)]
    if selected.size == 0:
        return bundle

    keep_mask = np.ones((count,), dtype=bool)
    keep_mask[selected] = False

    base_xyz = bundle["xyz"][selected].astype(np.float32)
    base_quat = bundle["rotation"][selected].astype(np.float32)
    base_scale = scales[selected].astype(np.float32)
    base_rgb = bundle["rgb"][selected].astype(np.uint8)
    base_opacity = opacity[selected].astype(np.float32)
    rot = _quaternion_to_rotation_matrix(torch.from_numpy(base_quat)).cpu().numpy().astype(np.float32)
    dominant = np.argmax(base_scale[:, :2], axis=1)
    offset_ratio = float(config.split_offset_ratio)
    shrink = max(1.1, float(config.split_scale_shrink))

    child_xyz = []
    child_scale = []
    child_quat = []
    child_rgb = []
    child_opacity = []
    child_meta: dict[str, list[np.ndarray]] = {
        "confidence": [],
        "unstable": [],
        "recentness": [],
        "source": [],
        "provenance": [],
    }
    for local_i in range(selected.size):
        axis_id = int(dominant[local_i])
        axis_dir = rot[local_i, :, axis_id]
        step = axis_dir * (base_scale[local_i, axis_id] * offset_ratio)
        split_scale = base_scale[local_i].copy()
        split_scale[axis_id] = max(split_scale[axis_id] / shrink, 1e-4)
        other_axis = 1 - axis_id
        split_scale[other_axis] = max(split_scale[other_axis] / (0.85 * shrink), 1e-4)
        for sign in (-1.0, 1.0):
            child_xyz.append((base_xyz[local_i] + sign * step).astype(np.float32))
            child_scale.append(split_scale.copy())
            child_quat.append(base_quat[local_i].copy())
            child_rgb.append(base_rgb[local_i].copy())
            child_opacity.append(np.float32(np.clip(base_opacity[local_i] * 0.92, 0.02, 0.999)))
            for key in child_meta:
                child_meta[key].append(bundle[key][selected[local_i] : selected[local_i] + 1].copy())

    merged: dict[str, np.ndarray] = {}
    merged["xyz"] = np.concatenate([bundle["xyz"][keep_mask], np.stack(child_xyz, axis=0)], axis=0).astype(np.float32)
    merged["rgb"] = np.concatenate([bundle["rgb"][keep_mask], np.stack(child_rgb, axis=0)], axis=0).astype(np.uint8)
    merged_scales = np.concatenate([scales[keep_mask], np.stack(child_scale, axis=0)], axis=0).astype(np.float32)
    merged_quats = np.concatenate([bundle["rotation"][keep_mask], np.stack(child_quat, axis=0)], axis=0).astype(np.float32)
    merged["rotation"] = merged_quats
    merged["scale_xyz"] = merged_scales
    axis_u, axis_v = _quats_scales_to_axes(
        torch.from_numpy(merged_quats),
        torch.from_numpy(merged_scales),
    )
    merged["axis_u"] = axis_u.cpu().numpy().astype(np.float32)
    merged["axis_v"] = axis_v.cpu().numpy().astype(np.float32)
    merged["scale"] = np.clip(0.5 * (merged_scales[:, 0] + merged_scales[:, 1]), 1e-4, None).astype(np.float32)
    merged["opacity"] = np.concatenate(
        [bundle["opacity"][keep_mask], np.asarray(child_opacity, dtype=np.float32)],
        axis=0,
    ).astype(np.float32)

    for key in ["confidence", "unstable", "recentness", "source", "provenance"]:
        merged[key] = np.concatenate(
            [bundle[key][keep_mask], np.concatenate(child_meta[key], axis=0)],
            axis=0,
        ).astype(bundle[key].dtype)

    max_points = int(config.max_points)
    if merged["xyz"].shape[0] > max_points:
        keep_score = np.clip(merged["opacity"], 0.02, 1.0) * np.clip(merged["confidence"], 0.05, 2.0)
        order = np.argsort(keep_score)[::-1][:max_points]
        order = np.sort(order)
        for key, value in list(merged.items()):
            merged[key] = value[order]
    return merged


def _merge_surface_bootstrap(
    renderer: GaussianSplatRenderer,
    seed_bundle: dict[str, np.ndarray],
    surface_bundle: dict[str, np.ndarray],
    max_points: int,
) -> dict[str, np.ndarray]:
    seed = renderer._normalize_external_bundle(seed_bundle)
    surface = renderer._normalize_external_bundle(surface_bundle)
    surface_count = int(surface["xyz"].shape[0])
    if surface_count <= 0:
        return seed_bundle
    max_points = max(1, int(max_points))
    keep_surface = min(surface_count, max_points)
    keep_seed = max(0, max_points - keep_surface)

    if keep_surface < surface_count:
        surface_conf = surface_bundle.get("confidence", np.ones((surface_count,), dtype=np.float32)).astype(np.float32)
        order = np.argsort(surface_conf)[::-1][:keep_surface]
        order = np.sort(order)
        for key in list(surface.keys()):
            surface[key] = surface[key][order]
    else:
        keep_surface = surface_count

    seed_count = int(seed["xyz"].shape[0])
    if keep_seed < seed_count:
        seed_conf = (1.0 / (1.0 + np.clip(seed["unstable"], 0.0, None))).astype(np.float32)
        seed_score = 0.6 * seed_conf + 0.25 * np.clip(seed["opacity"], 0.0, 1.0) + 0.15 * np.clip(seed["recentness"], 0.0, 1.0)
        order = np.argsort(seed_score)[::-1][:keep_seed]
        order = np.sort(order)
        for key in list(seed.keys()):
            seed[key] = seed[key][order]

    merged = {}
    for key in ["xyz", "rgb", "scale", "opacity", "axis_u", "axis_v", "unstable", "recentness", "source", "provenance"]:
        merged[key] = np.concatenate([seed[key], surface[key]], axis=0)
    return merged


def _merge_surface_candidates(
    candidates: list[dict[str, np.ndarray]],
    max_points: int,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    if not candidates:
        return _empty_bundle()
    merged: dict[str, list[np.ndarray]] = {
        "xyz": [],
        "rgb": [],
        "scale": [],
        "opacity": [],
        "axis_u": [],
        "axis_v": [],
        "unstable": [],
        "recentness": [],
        "source": [],
        "provenance": [],
        "confidence": [],
    }
    for bundle in candidates:
        count = int(bundle["xyz"].shape[0])
        if count <= 0:
            continue
        for key in merged.keys():
            if key in bundle:
                merged[key].append(bundle[key])
            elif key == "confidence":
                merged[key].append(np.ones((count,), dtype=np.float32))
            elif key == "provenance":
                merged[key].append(np.full((count,), 5, dtype=np.int8))
            else:
                raise KeyError(f"Missing key '{key}' in surface candidate bundle")
    if not merged["xyz"]:
        return _empty_bundle()

    packed = {
        "xyz": np.concatenate(merged["xyz"], axis=0).astype(np.float32),
        "rgb": np.concatenate(merged["rgb"], axis=0).astype(np.uint8),
        "scale": np.concatenate(merged["scale"], axis=0).astype(np.float32),
        "opacity": np.concatenate(merged["opacity"], axis=0).astype(np.float32),
        "axis_u": np.concatenate(merged["axis_u"], axis=0).astype(np.float32),
        "axis_v": np.concatenate(merged["axis_v"], axis=0).astype(np.float32),
        "unstable": np.concatenate(merged["unstable"], axis=0).astype(np.float32),
        "recentness": np.concatenate(merged["recentness"], axis=0).astype(np.float32),
        "source": np.concatenate(merged["source"], axis=0).astype(np.int8),
        "provenance": np.concatenate(merged["provenance"], axis=0).astype(np.int8),
        "confidence": np.concatenate(merged["confidence"], axis=0).astype(np.float32),
    }
    if packed["xyz"].shape[0] <= 1:
        return packed

    score = (
        np.clip(packed["confidence"], 0.05, 2.0)
        * (0.5 + np.clip(packed["opacity"], 0.02, 1.0))
        * (1.25 - np.clip(packed["unstable"], 0.0, 1.0))
    ).astype(np.float32)
    voxel = max(float(voxel_size), 1e-3)
    voxel_key = np.floor(packed["xyz"] / voxel).astype(np.int32)
    best_idx: dict[tuple[int, int, int], int] = {}
    for idx, key in enumerate(map(tuple, voxel_key.tolist())):
        prev = best_idx.get(key)
        if prev is None or score[idx] > score[prev]:
            best_idx[key] = idx
    keep = np.array(sorted(best_idx.values()), dtype=np.int64)
    for key, value in list(packed.items()):
        packed[key] = value[keep]

    if packed["xyz"].shape[0] > int(max_points):
        order = np.argsort(score[keep])[::-1][: int(max_points)]
        order = np.sort(order)
        for key, value in list(packed.items()):
            packed[key] = value[order]
    return packed


def _empty_bundle() -> dict[str, np.ndarray]:
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
        "provenance": np.zeros((0,), dtype=np.int8),
        "confidence": np.zeros((0,), dtype=np.float32),
    }


def _preview_indices(frame_count: int, preview_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    preview_frames = max(1, preview_frames)
    raw = np.linspace(0, frame_count - 1, num=min(frame_count, preview_frames), dtype=np.int32)
    return [int(item) for item in np.unique(raw)]


def _ssim_rgb(pred_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
    pred = pred_rgb.permute(2, 0, 1).unsqueeze(0)
    target = target_rgb.permute(2, 0, 1).unsqueeze(0)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_x = F.avg_pool2d(pred, kernel_size=11, stride=1, padding=5)
    mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    sigma_x = F.avg_pool2d(pred * pred, kernel_size=11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-6
    )
    return torch.mean(score)
