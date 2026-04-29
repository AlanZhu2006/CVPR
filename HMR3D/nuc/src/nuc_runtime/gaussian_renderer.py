from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nuc_runtime.config import OutputConfig
from nuc_runtime.gaussian_builder import IncrementalGaussianBuilder
from nuc_runtime.memory_router import MemoryRouter


@dataclass
class RenderResult:
    image_rgb: np.ndarray
    point_count: int
    active_points: int
    archived_points: int
    warmstart_points: int
    projected_points: int


class GaussianSplatRenderer:
    def __init__(self, output_config: OutputConfig):
        self.output_config = output_config
        self._intrinsics_cache: dict[Path, np.ndarray] = {}

    def render_current_view(
        self,
        router: MemoryRouter,
        pose: np.ndarray,
        image_shape: tuple[int, int],
        image_path: str | None,
    ) -> RenderResult:
        builder = router.gaussian_builder
        if builder is None:
            h, w = image_shape
            blank = np.full((h, w, 3), self.output_config.render_background_gray, dtype=np.uint8)
            return RenderResult(blank, 0, 0, 0, 0, 0)

        active_bundle = self._collect_active_bundle(router, builder)
        archived_bundle = self._collect_archived_bundle(router, builder)
        warm_bundle = self._collect_warmstart_bundle(router, builder)
        merged = self._merge_bundles([archived_bundle, warm_bundle, active_bundle])
        render_rgb, projected = self._render_bundle(
            merged,
            pose=pose,
            image_shape=image_shape,
            image_path=image_path,
        )
        return RenderResult(
            image_rgb=render_rgb,
            point_count=int(merged["xyz"].shape[0]),
            active_points=int(active_bundle["xyz"].shape[0]),
            archived_points=int(archived_bundle["xyz"].shape[0]),
            warmstart_points=int(warm_bundle["xyz"].shape[0]),
            projected_points=projected,
        )

    def render_bundle_view(
        self,
        bundle: dict[str, np.ndarray],
        pose: np.ndarray,
        image_shape: tuple[int, int],
        image_path: str | None,
        *,
        active_points: int | None = None,
        archived_points: int | None = None,
        warmstart_points: int | None = None,
    ) -> RenderResult:
        normalized = self._normalize_external_bundle(bundle)
        render_rgb, projected = self._render_bundle(
            normalized,
            pose=pose,
            image_shape=image_shape,
            image_path=image_path,
        )
        count = int(normalized["xyz"].shape[0])
        return RenderResult(
            image_rgb=render_rgb,
            point_count=count,
            active_points=int(active_points if active_points is not None else count),
            archived_points=int(archived_points if archived_points is not None else 0),
            warmstart_points=int(warmstart_points if warmstart_points is not None else 0),
            projected_points=projected,
        )

    def _collect_active_bundle(
        self,
        router: MemoryRouter,
        builder: IncrementalGaussianBuilder,
    ) -> dict[str, np.ndarray]:
        if router.active is None:
            return self._empty_bundle()
        return self._tag_bundle(builder.active_points(router.active.submap_id), provenance=3)

    def _collect_archived_bundle(
        self,
        router: MemoryRouter,
        builder: IncrementalGaussianBuilder,
    ) -> dict[str, np.ndarray]:
        if not router.bank:
            return self._empty_bundle()
        selected = router.bank[-self.output_config.render_max_archived_submaps :]
        bundles = []
        recent_keep = max(0, int(getattr(builder.config, "gaussian_full_recent_archives", 2)))
        split_index = max(0, len(selected) - recent_keep)
        for idx, item in enumerate(selected):
            if not item.gaussian_handle:
                continue
            use_coarse = idx < split_index
            bundles.append(self._tag_bundle(builder.handle_points(item.gaussian_handle, use_coarse=use_coarse), provenance=1))
        merged = self._merge_bundles(bundles)
        return self._limit_bundle_points(merged, self.output_config.render_max_archived_points)

    def _collect_warmstart_bundle(
        self,
        router: MemoryRouter,
        builder: IncrementalGaussianBuilder,
    ) -> dict[str, np.ndarray]:
        if router.active is None or not router.active.recovered_gaussian_handles:
            return self._empty_bundle()
        bundles = [self._tag_bundle(builder.handle_points(handle), provenance=2) for handle in router.active.recovered_gaussian_handles]
        merged = self._merge_bundles(bundles)
        return self._limit_bundle_points(merged, self.output_config.render_max_warmstart_points)

    def _merge_bundles(self, bundles: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        bundles = [bundle for bundle in bundles if bundle["xyz"].shape[0] > 0]
        if not bundles:
            return self._empty_bundle()
        return {
            "xyz": np.vstack([bundle["xyz"] for bundle in bundles]).astype(np.float32),
            "rgb": np.vstack([bundle["rgb"] for bundle in bundles]).astype(np.uint8),
            "scale": np.concatenate([bundle["scale"] for bundle in bundles]).astype(np.float32),
            "opacity": np.concatenate([bundle["opacity"] for bundle in bundles]).astype(np.float32),
            "axis_u": np.vstack([bundle["axis_u"] for bundle in bundles]).astype(np.float32),
            "axis_v": np.vstack([bundle["axis_v"] for bundle in bundles]).astype(np.float32),
            "unstable": np.concatenate([bundle["unstable"] for bundle in bundles]).astype(np.float32),
            "recentness": np.concatenate([bundle["recentness"] for bundle in bundles]).astype(np.float32),
            "source": np.concatenate([bundle["source"] for bundle in bundles]).astype(np.int8),
            "provenance": np.concatenate([bundle["provenance"] for bundle in bundles]).astype(np.int8),
        }

    def _normalize_external_bundle(self, bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        count = int(bundle.get("xyz", np.zeros((0, 3), dtype=np.float32)).shape[0])
        normalized = {
            "xyz": bundle.get("xyz", np.zeros((0, 3), dtype=np.float32)).astype(np.float32),
            "rgb": bundle.get("rgb", np.zeros((0, 3), dtype=np.uint8)).astype(np.uint8),
            "scale": bundle.get("scale", np.zeros((count,), dtype=np.float32)).astype(np.float32),
            "opacity": bundle.get("opacity", np.zeros((count,), dtype=np.float32)).astype(np.float32),
            "axis_u": bundle.get("axis_u", np.zeros((count, 3), dtype=np.float32)).astype(np.float32),
            "axis_v": bundle.get("axis_v", np.zeros((count, 3), dtype=np.float32)).astype(np.float32),
            "unstable": bundle.get("unstable", np.zeros((count,), dtype=np.float32)).astype(np.float32),
            "recentness": bundle.get("recentness", np.zeros((count,), dtype=np.float32)).astype(np.float32),
            "source": bundle.get("source", np.zeros((count,), dtype=np.int8)).astype(np.int8),
            "provenance": bundle.get("provenance", np.full((count,), 3, dtype=np.int8)).astype(np.int8),
        }
        return normalized

    def _limit_bundle_points(self, bundle: dict[str, np.ndarray], max_points: int) -> dict[str, np.ndarray]:
        count = int(bundle["xyz"].shape[0])
        if count <= 0 or count <= max_points:
            return bundle
        order = np.linspace(0, count - 1, max_points, dtype=np.int32)
        return {
            "xyz": bundle["xyz"][order],
            "rgb": bundle["rgb"][order],
            "scale": bundle["scale"][order],
            "opacity": bundle["opacity"][order],
            "axis_u": bundle["axis_u"][order],
            "axis_v": bundle["axis_v"][order],
            "unstable": bundle["unstable"][order],
            "recentness": bundle["recentness"][order],
            "source": bundle["source"][order],
            "provenance": bundle["provenance"][order],
        }

    def _tag_bundle(self, bundle: dict[str, np.ndarray], provenance: int) -> dict[str, np.ndarray]:
        count = int(bundle["xyz"].shape[0])
        if count <= 0:
            tagged = dict(bundle)
            tagged["provenance"] = np.zeros((0,), dtype=np.int8)
            return tagged
        tagged = dict(bundle)
        tagged["provenance"] = np.full((count,), provenance, dtype=np.int8)
        return tagged

    def _render_bundle(
        self,
        bundle: dict[str, np.ndarray],
        pose: np.ndarray,
        image_shape: tuple[int, int],
        image_path: str | None,
    ) -> tuple[np.ndarray, int]:
        backend = (self.output_config.render_backend or "cpu").strip().lower()
        if backend == "gsplat":
            try:
                return self._render_bundle_gsplat(
                    bundle=bundle,
                    pose=pose,
                    image_shape=image_shape,
                    image_path=image_path,
                )
            except Exception:
                # Fall back to the existing CPU renderer if gsplat is unavailable or fails.
                return self._render_bundle_cpu(
                    bundle=bundle,
                    pose=pose,
                    image_shape=image_shape,
                    image_path=image_path,
                )
        return self._render_bundle_cpu(
            bundle=bundle,
            pose=pose,
            image_shape=image_shape,
            image_path=image_path,
        )

    def _render_bundle_cpu(
        self,
        bundle: dict[str, np.ndarray],
        pose: np.ndarray,
        image_shape: tuple[int, int],
        image_path: str | None,
    ) -> tuple[np.ndarray, int]:
        h, w = image_shape
        image = np.full((h, w, 3), self.output_config.render_background_gray, dtype=np.float32)
        if bundle["xyz"].shape[0] == 0:
            return image.astype(np.uint8), 0

        internal_scale = float(np.clip(self.output_config.render_internal_scale, 0.2, 1.0))
        render_h = max(32, int(round(h * internal_scale)))
        render_w = max(32, int(round(w * internal_scale)))
        image_small = np.full((render_h, render_w, 3), self.output_config.render_background_gray, dtype=np.float32)
        K = self._load_intrinsics(image_path, w=w, h=h).copy()
        K[0, :] *= render_w / max(1.0, float(w))
        K[1, :] *= render_h / max(1.0, float(h))
        xyz_camera = self._world_to_camera(pose, bundle["xyz"])
        axis_u_camera = self._world_vec_to_camera(pose, bundle["axis_u"])
        axis_v_camera = self._world_vec_to_camera(pose, bundle["axis_v"])
        valid = np.isfinite(xyz_camera).all(axis=1)
        valid &= xyz_camera[:, 2] > 0.05
        valid &= xyz_camera[:, 2] <= self.output_config.render_depth_window_m
        if not np.any(valid):
            return image.astype(np.uint8), 0

        xyz_camera = xyz_camera[valid]
        rgb = bundle["rgb"][valid]
        scale = bundle["scale"][valid]
        opacity = bundle["opacity"][valid]
        axis_u_camera = axis_u_camera[valid]
        axis_v_camera = axis_v_camera[valid]
        unstable = bundle["unstable"][valid]
        recentness = bundle["recentness"][valid]
        source = bundle["source"][valid]
        provenance = bundle["provenance"][valid]

        projected = xyz_camera @ K.T
        uv = projected[:, :2] / np.clip(projected[:, 2:], 1e-8, None)
        depth = xyz_camera[:, 2].astype(np.float32)
        radius_px = np.clip(K[0, 0] * scale / np.clip(depth, 1e-4, None), self.output_config.render_min_radius_px, self.output_config.render_max_radius_px)
        keep = self._view_aware_budget(
            uv=uv,
            canvas_shape=(render_h, render_w),
            depth=depth,
            radius_px=radius_px,
            unstable=unstable,
            recentness=recentness,
            source=source,
            provenance=provenance,
        )
        xyz_camera = xyz_camera[keep]
        rgb = rgb[keep]
        scale = scale[keep]
        opacity = opacity[keep]
        axis_u_camera = axis_u_camera[keep]
        axis_v_camera = axis_v_camera[keep]
        uv = uv[keep]
        depth = depth[keep]
        radius_px = radius_px[keep]

        projected_points, rgb_acc, alpha = self._render_tiled(
            uv=uv,
            xyz_camera=xyz_camera,
            axis_u_camera=axis_u_camera,
            axis_v_camera=axis_v_camera,
            rgb=rgb,
            opacity=opacity,
            depth=depth,
            radius_px=radius_px,
            K=K,
            canvas_shape=(render_h, render_w),
        )

        composed = rgb_acc + (1.0 - alpha[..., None]) * image_small
        composed_u8 = np.clip(composed, 0, 255).astype(np.uint8)
        if self.output_config.render_enable_hole_fill:
            composed_u8 = self._fill_small_holes(composed_u8, alpha)
        if render_h != h or render_w != w:
            composed_u8 = cv2.resize(composed_u8, (w, h), interpolation=cv2.INTER_CUBIC)
        return composed_u8, projected_points

    def _render_bundle_gsplat(
        self,
        bundle: dict[str, np.ndarray],
        pose: np.ndarray,
        image_shape: tuple[int, int],
        image_path: str | None,
    ) -> tuple[np.ndarray, int]:
        import torch
        import gsplat

        h, w = image_shape
        if bundle["xyz"].shape[0] == 0:
            blank = np.full((h, w, 3), self.output_config.render_background_gray, dtype=np.uint8)
            return blank, 0
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat backend requested but CUDA is unavailable")

        K = self._load_intrinsics(image_path, w=w, h=h).astype(np.float32)
        xyz_camera = self._world_to_camera(pose, bundle["xyz"])
        valid = np.isfinite(xyz_camera).all(axis=1)
        valid &= xyz_camera[:, 2] > 0.05
        valid &= xyz_camera[:, 2] <= self.output_config.render_depth_window_m
        if not np.any(valid):
            blank = np.full((h, w, 3), self.output_config.render_background_gray, dtype=np.uint8)
            return blank, 0

        xyz = bundle["xyz"][valid].astype(np.float32)
        rgb = bundle["rgb"][valid].astype(np.float32) / 255.0
        opacity = np.clip(bundle["opacity"][valid].astype(np.float32), 0.0, 0.999)
        axis_u = bundle["axis_u"][valid].astype(np.float32)
        axis_v = bundle["axis_v"][valid].astype(np.float32)
        unstable = bundle["unstable"][valid].astype(np.float32)
        recentness = bundle["recentness"][valid].astype(np.float32)
        source = bundle["source"][valid].astype(np.int8)
        provenance = bundle["provenance"][valid].astype(np.int8)

        xyz_camera = xyz_camera[valid]
        radius_px = np.clip(
            K[0, 0] * bundle["scale"][valid].astype(np.float32) / np.clip(xyz_camera[:, 2], 1e-4, None),
            self.output_config.render_min_radius_px,
            self.output_config.render_max_radius_px,
        )
        keep = self._view_aware_budget(
            uv=(xyz_camera[:, :2] @ K[:2, :2].T) / np.clip(xyz_camera[:, 2:], 1e-8, None)
            + K[:2, 2][None, :],
            canvas_shape=(h, w),
            depth=xyz_camera[:, 2].astype(np.float32),
            radius_px=radius_px,
            unstable=unstable,
            recentness=recentness,
            source=source,
            provenance=provenance,
        )
        xyz = xyz[keep]
        rgb = rgb[keep]
        opacity = opacity[keep]
        axis_u = axis_u[keep]
        axis_v = axis_v[keep]
        scale_base = bundle["scale"][valid].astype(np.float32)[keep]
        projected_points = int(xyz.shape[0])
        if projected_points <= 0:
            blank = np.full((h, w, 3), self.output_config.render_background_gray, dtype=np.uint8)
            return blank, 0

        means = torch.from_numpy(xyz).to(device="cuda", dtype=torch.float32)
        colors = torch.from_numpy(rgb).to(device="cuda", dtype=torch.float32)
        opacities = torch.from_numpy(opacity).to(device="cuda", dtype=torch.float32)
        scales, quats = self._axes_to_gsplat_params(axis_u=axis_u, axis_v=axis_v, scale_base=scale_base)
        scales_t = torch.from_numpy(scales).to(device="cuda", dtype=torch.float32)
        quats_t = torch.from_numpy(quats).to(device="cuda", dtype=torch.float32)
        viewmat = torch.from_numpy(np.linalg.inv(pose).astype(np.float32)).to(device="cuda").unsqueeze(0)
        Ks = torch.from_numpy(K).to(device="cuda", dtype=torch.float32).unsqueeze(0)
        background = torch.full(
            (1, 3),
            float(self.output_config.render_background_gray) / 255.0,
            device="cuda",
            dtype=torch.float32,
        )

        render_colors, _, _ = gsplat.rasterization(
            means=means,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=Ks,
            width=int(w),
            height=int(h),
            near_plane=0.01,
            far_plane=float(self.output_config.render_depth_window_m),
            radius_clip=float(self.output_config.render_gsplat_radius_clip),
            packed=True,
            tile_size=int(self.output_config.render_gsplat_tile_size),
            backgrounds=background,
            render_mode="RGB",
            rasterize_mode=str(self.output_config.render_gsplat_rasterize_mode),
        )
        render_np = render_colors.detach().float().cpu().numpy()
        if render_np.ndim == 4:
            render_np = render_np[0]
        if render_np.ndim == 3 and render_np.shape[0] == 3:
            render_np = np.transpose(render_np, (1, 2, 0))
        render_np = np.clip(render_np, 0.0, 1.0)
        render_u8 = (255.0 * render_np).astype(np.uint8)
        return render_u8, projected_points

    def _view_aware_budget(
        self,
        uv: np.ndarray | None,
        canvas_shape: tuple[int, int] | None,
        depth: np.ndarray,
        radius_px: np.ndarray,
        unstable: np.ndarray,
        recentness: np.ndarray,
        source: np.ndarray,
        provenance: np.ndarray,
    ) -> np.ndarray:
        count = int(depth.shape[0])
        budget = min(count, int(self.output_config.render_view_budget_points))
        if budget <= 0 or count <= budget:
            return np.arange(count, dtype=np.int32)
        near_term = 1.0 / (1.0 + 0.08 * depth)
        radius_term = np.clip(radius_px / max(1e-6, self.output_config.render_max_radius_px), 0.0, 1.0)
        provenance_weight = np.choose(np.clip(provenance, 0, 3), [0.4, 0.7, 1.0, 1.2]).astype(np.float32)
        source_weight = np.choose(np.clip(source, 0, 3), [0.4, 0.75, 0.65, 1.0]).astype(np.float32)
        stability_term = 1.0 / (1.0 + np.clip(unstable, 0.0, 3.0))
        score = (
            1.35 * stability_term
            + 0.75 * recentness
            + 0.9 * near_term
            + 0.6 * radius_term
            + provenance_weight
            + 0.5 * source_weight
        ).astype(np.float32)
        if uv is not None and canvas_shape is not None and uv.shape[0] == count:
            covered_keep = self._coverage_aware_keep(
                uv=uv,
                canvas_shape=canvas_shape,
                radius_px=radius_px,
                depth=depth,
                score=score,
                budget=budget,
            )
            if covered_keep.shape[0] > 0:
                return covered_keep
        keep = np.argpartition(score, -budget)[-budget:]
        return keep[np.argsort(depth[keep])]

    def _coverage_aware_keep(
        self,
        uv: np.ndarray,
        canvas_shape: tuple[int, int],
        radius_px: np.ndarray,
        depth: np.ndarray,
        score: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        h, w = canvas_shape
        if budget <= 0 or h <= 0 or w <= 0:
            return np.zeros((0,), dtype=np.int32)
        finite = np.isfinite(uv).all(axis=1)
        margin = np.clip(radius_px, self.output_config.render_min_radius_px, self.output_config.render_max_radius_px)
        visible = finite
        visible &= uv[:, 0] >= -margin
        visible &= uv[:, 0] < float(w) + margin
        visible &= uv[:, 1] >= -margin
        visible &= uv[:, 1] < float(h) + margin
        visible_idx = np.flatnonzero(visible)
        if visible_idx.size == 0:
            return np.zeros((0,), dtype=np.int32)

        aspect = max(1e-6, float(w) / max(1.0, float(h)))
        grid_cols = int(np.clip(np.sqrt(max(1, budget) * aspect), 8, 32))
        grid_rows = int(np.clip(np.ceil(grid_cols / aspect), 4, 18))
        quota = max(1, int(np.ceil(budget / max(1, grid_cols * grid_rows))))

        uv_visible = uv[visible_idx]
        cell_x = np.clip((uv_visible[:, 0] * grid_cols / max(1.0, float(w))).astype(np.int32), 0, grid_cols - 1)
        cell_y = np.clip((uv_visible[:, 1] * grid_rows / max(1.0, float(h))).astype(np.int32), 0, grid_rows - 1)
        cells = cell_y * grid_cols + cell_x
        order = np.argsort(-score[visible_idx])
        selected: list[int] = []
        selected_mask = np.zeros((score.shape[0],), dtype=bool)
        cell_counts: dict[int, int] = {}

        for local_idx in order:
            global_idx = int(visible_idx[local_idx])
            cell = int(cells[local_idx])
            if cell_counts.get(cell, 0) >= quota:
                continue
            selected.append(global_idx)
            selected_mask[global_idx] = True
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            if len(selected) >= budget:
                break

        if len(selected) < budget:
            for local_idx in order:
                global_idx = int(visible_idx[local_idx])
                if selected_mask[global_idx]:
                    continue
                selected.append(global_idx)
                selected_mask[global_idx] = True
                if len(selected) >= budget:
                    break

        keep = np.array(selected, dtype=np.int32)
        if keep.size == 0:
            return keep
        return keep[np.argsort(depth[keep])]

    def _render_tiled(
        self,
        uv: np.ndarray,
        xyz_camera: np.ndarray,
        axis_u_camera: np.ndarray,
        axis_v_camera: np.ndarray,
        rgb: np.ndarray,
        opacity: np.ndarray,
        depth: np.ndarray,
        radius_px: np.ndarray,
        K: np.ndarray,
        canvas_shape: tuple[int, int],
    ) -> tuple[int, np.ndarray, np.ndarray]:
        h, w = canvas_shape
        tile_size = max(24, int(self.output_config.render_tile_size))
        num_tiles_x = int(np.ceil(w / tile_size))
        num_tiles_y = int(np.ceil(h / tile_size))
        tile_bins: dict[tuple[int, int], list[int]] = {}
        projected_points = 0
        tile_data: list[tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray, float, float]] = []

        for idx in np.argsort(depth)[::-1]:
            axis_u_px, axis_v_px = self._project_screen_axes(K, xyz_camera[idx], axis_u_camera[idx], axis_v_camera[idx])
            if axis_u_px is None or axis_v_px is None:
                continue
            u, v = float(uv[idx, 0]), float(uv[idx, 1])
            bound_radius = max(radius_px[idx], np.linalg.norm(axis_u_px), np.linalg.norm(axis_v_px), self.output_config.render_min_radius_px)
            if u < -bound_radius or v < -bound_radius or u >= w + bound_radius or v >= h + bound_radius:
                continue
            projected_points += 1
            tile_data.append((idx, u, v, axis_u_px, axis_v_px, rgb[idx].astype(np.float32), float(opacity[idx]), float(depth[idx])))
            tx0 = max(0, int((u - bound_radius) // tile_size))
            tx1 = min(num_tiles_x - 1, int((u + bound_radius) // tile_size))
            ty0 = max(0, int((v - bound_radius) // tile_size))
            ty1 = min(num_tiles_y - 1, int((v + bound_radius) // tile_size))
            point_ref = len(tile_data) - 1
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tile_bins.setdefault((tx, ty), []).append(point_ref)

        rgb_acc = np.zeros((h, w, 3), dtype=np.float32)
        alpha = np.zeros((h, w), dtype=np.float32)
        depth_acc = np.full((h, w), np.inf, dtype=np.float32)
        for (tx, ty), point_refs in tile_bins.items():
            x0 = tx * tile_size
            y0 = ty * tile_size
            x1 = min(w, x0 + tile_size)
            y1 = min(h, y0 + tile_size)
            rgb_tile = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
            alpha_tile = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            depth_tile = np.full((y1 - y0, x1 - x0), np.inf, dtype=np.float32)
            for ref in point_refs:
                _, u, v, axis_u_px, axis_v_px, color, point_opacity, point_depth = tile_data[ref]
                self._splat_one(
                    rgb_tile,
                    alpha_tile,
                    depth_tile,
                    center=(u - x0, v - y0),
                    axis_u_px=axis_u_px,
                    axis_v_px=axis_v_px,
                    color=color,
                    opacity=point_opacity,
                    depth=point_depth,
                )
            rgb_acc[y0:y1, x0:x1] = rgb_acc[y0:y1, x0:x1] + (1.0 - alpha[y0:y1, x0:x1])[..., None] * rgb_tile
            alpha[y0:y1, x0:x1] = alpha[y0:y1, x0:x1] + (1.0 - alpha[y0:y1, x0:x1]) * alpha_tile
            depth_acc[y0:y1, x0:x1] = np.minimum(depth_acc[y0:y1, x0:x1], depth_tile)
        return projected_points, rgb_acc, alpha

    def _splat_one(
        self,
        rgb_acc: np.ndarray,
        alpha_acc: np.ndarray,
        depth_acc: np.ndarray,
        center: tuple[float, float],
        axis_u_px: np.ndarray,
        axis_v_px: np.ndarray,
        color: np.ndarray,
        opacity: float,
        depth: float,
    ) -> None:
        h, w = alpha_acc.shape
        extent = max(
            self.output_config.render_min_radius_px,
            np.linalg.norm(axis_u_px) * 2.5,
            np.linalg.norm(axis_v_px) * 2.5,
        )
        radius_int = max(2, int(np.ceil(extent)))
        x0 = max(0, int(np.floor(center[0])) - radius_int)
        y0 = max(0, int(np.floor(center[1])) - radius_int)
        x1 = min(w, int(np.floor(center[0])) + radius_int + 1)
        y1 = min(h, int(np.floor(center[1])) + radius_int + 1)
        if x0 >= x1 or y0 >= y1:
            return

        inv_cov = self._ellipse_inv_cov(axis_u_px, axis_v_px)
        if inv_cov is None:
            return
        xs = np.arange(x0, x1, dtype=np.float32) - center[0]
        ys = np.arange(y0, y1, dtype=np.float32) - center[1]
        xx, yy = np.meshgrid(xs, ys)
        quad = (
            inv_cov[0, 0] * xx * xx
            + 2.0 * inv_cov[0, 1] * xx * yy
            + inv_cov[1, 1] * yy * yy
        )
        patch = np.exp(-0.5 * quad).astype(np.float32) * np.clip(opacity, 0.0, 1.0)
        patch[quad > 9.0] = 0.0
        if patch.size == 0:
            return

        dst_alpha = alpha_acc[y0:y1, x0:x1]
        dst_rgb = rgb_acc[y0:y1, x0:x1]
        dst_depth = depth_acc[y0:y1, x0:x1]
        sigma = max(1e-3, float(self.output_config.render_surface_depth_sigma))
        close_weight = np.where(
            np.isfinite(dst_depth),
            np.exp(-0.5 * np.square((dst_depth - depth) / sigma)),
            1.0,
        ).astype(np.float32)
        front_boost = np.where(
            np.isfinite(dst_depth),
            np.where(depth <= dst_depth + sigma, self.output_config.render_surface_opacity_gain, 0.55),
            self.output_config.render_surface_opacity_gain,
        ).astype(np.float32)
        one_minus = 1.0 - dst_alpha
        effective = patch * one_minus * close_weight * front_boost
        dst_rgb += effective[..., None] * color[None, None, :]
        dst_alpha += effective
        dst_depth[:] = np.minimum(dst_depth, np.where(patch > 0.02, depth, np.inf))

    def _world_to_camera(self, camera_pose: np.ndarray, xyz_world: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(camera_pose)
        xyz_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
        return (xyz_h @ w2c.T)[:, :3]

    def _world_vec_to_camera(self, camera_pose: np.ndarray, xyz_world_vec: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(camera_pose)[:3, :3].astype(np.float32)
        return xyz_world_vec @ w2c.T

    def _axes_to_gsplat_params(
        self,
        axis_u: np.ndarray,
        axis_v: np.ndarray,
        scale_base: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        count = int(axis_u.shape[0])
        scales = np.zeros((count, 3), dtype=np.float32)
        quats = np.zeros((count, 4), dtype=np.float32)
        for idx in range(count):
            u = axis_u[idx]
            v = axis_v[idx]
            su = max(1e-4, float(np.linalg.norm(u)))
            sv = max(1e-4, float(np.linalg.norm(v)))
            u_dir = u / su
            v_proj = v - np.dot(v, u_dir) * u_dir
            sv_proj = max(1e-4, float(np.linalg.norm(v_proj)))
            v_dir = v_proj / sv_proj
            n_dir = np.cross(u_dir, v_dir).astype(np.float32)
            n_norm = float(np.linalg.norm(n_dir))
            if not np.isfinite(n_norm) or n_norm <= 1e-6:
                n_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                n_dir = n_dir / n_norm
            # Re-orthogonalize v against u/n to keep a stable rotation frame.
            v_dir = np.cross(n_dir, u_dir).astype(np.float32)
            v_dir /= max(1e-6, float(np.linalg.norm(v_dir)))
            rot = np.stack([u_dir, v_dir, n_dir], axis=1).astype(np.float32)
            quats[idx] = self._rotation_matrix_to_quaternion(rot)
            scales[idx] = np.array(
                [
                    np.clip(su, 1e-4, self.output_config.render_max_radius_px),
                    np.clip(max(sv, sv_proj), 1e-4, self.output_config.render_max_radius_px),
                    np.clip(max(0.35 * float(scale_base[idx]), 1e-4), 1e-4, self.output_config.render_max_radius_px),
                ],
                dtype=np.float32,
            )
        return scales, quats

    def _rotation_matrix_to_quaternion(self, rot: np.ndarray) -> np.ndarray:
        m = rot.astype(np.float32)
        trace = float(np.trace(m))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([w, x, y, z], dtype=np.float32)
        quat /= max(1e-6, float(np.linalg.norm(quat)))
        return quat

    def _project_screen_axes(
        self,
        K: np.ndarray,
        center_cam: np.ndarray,
        axis_u_cam: np.ndarray,
        axis_v_cam: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        z = float(center_cam[2])
        if z <= 1e-4:
            return None, None
        center_uv = (center_cam @ K.T)[:2] / z

        def project_offset(offset_cam: np.ndarray) -> np.ndarray | None:
            endpoint = center_cam + offset_cam
            if endpoint[2] <= 1e-4:
                return None
            endpoint_uv = (endpoint @ K.T)[:2] / endpoint[2]
            return (endpoint_uv - center_uv).astype(np.float32)

        du = project_offset(axis_u_cam)
        dv = project_offset(axis_v_cam)
        if du is None or dv is None:
            return None, None
        norm_u = max(float(np.linalg.norm(du)), self.output_config.render_min_radius_px * 0.6)
        norm_v = max(float(np.linalg.norm(dv)), self.output_config.render_min_radius_px * 0.6)
        du = du * (np.clip(norm_u, self.output_config.render_min_radius_px * 0.6, self.output_config.render_max_radius_px) / max(norm_u, 1e-6))
        dv = dv * (np.clip(norm_v, self.output_config.render_min_radius_px * 0.6, self.output_config.render_max_radius_px) / max(norm_v, 1e-6))
        return du, dv

    def _ellipse_inv_cov(self, axis_u_px: np.ndarray, axis_v_px: np.ndarray) -> np.ndarray | None:
        basis = np.stack([axis_u_px, axis_v_px], axis=1).astype(np.float32)
        cov = basis @ basis.T
        cov += np.eye(2, dtype=np.float32) * 0.25
        det = float(np.linalg.det(cov))
        if not np.isfinite(det) or det <= 1e-6:
            return None
        return np.linalg.inv(cov).astype(np.float32)

    def _fill_small_holes(self, image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if alpha.size == 0:
            return image_rgb
        kernel_size = max(3, int(self.output_config.render_hole_fill_kernel) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        solid = (alpha > 0.10).astype(np.uint8) * 255
        dilated = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, kernel)
        holes = cv2.bitwise_and(dilated, cv2.bitwise_not(solid))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        refined = np.zeros_like(holes)
        for idx in range(1, num_labels):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area <= self.output_config.render_hole_fill_max_area:
                refined[labels == idx] = 255
        if int(refined.sum()) == 0:
            return image_rgb
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, refined, 3, cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)

    def _load_intrinsics(self, image_path: str | None, w: int, h: int) -> np.ndarray:
        if image_path is None:
            return np.array([[700.0, 0.0, w * 0.5], [0.0, 700.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        sequence_path = Path(image_path).expanduser().resolve().parents[1]
        cached = self._intrinsics_cache.get(sequence_path)
        if cached is not None:
            return cached
        calib_path = sequence_path / "calib.txt"
        if not calib_path.exists():
            K = np.array([[700.0, 0.0, w * 0.5], [0.0, 700.0, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
            self._intrinsics_cache[sequence_path] = K
            return K
        first_line = calib_path.read_text(encoding="utf-8").splitlines()[0]
        values = np.array([float(item) for item in first_line.split()[1:]], dtype=np.float32).reshape(3, 4)
        K = values[:, :3]
        self._intrinsics_cache[sequence_path] = K
        return K

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
            "provenance": np.zeros((0,), dtype=np.int8),
        }


def psnr(pred_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    pred = pred_rgb.astype(np.float32)
    target = target_rgb.astype(np.float32)
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 1e-8:
        return 99.0
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def ssim_rgb(pred_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    pred = pred_rgb.astype(np.float32)
    target = target_rgb.astype(np.float32)
    scores = []
    for channel in range(3):
        x = pred[..., channel]
        y = target[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        score_map = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        scores.append(float(np.mean(score_map)))
    return float(np.mean(scores))


def save_render_triplet(
    output_dir: str | Path,
    frame_idx: int,
    target_rgb: np.ndarray,
    render_rgb: np.ndarray,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    diff = cv2.absdiff(target_rgb, render_rgb)
    triplet = np.concatenate([target_rgb, render_rgb, diff], axis=1)

    gt_path = output_dir / f"{frame_idx:06d}_gt.png"
    render_path = output_dir / f"{frame_idx:06d}_render.png"
    triplet_path = output_dir / f"{frame_idx:06d}_triplet.png"
    cv2.imwrite(str(gt_path), cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(triplet_path), cv2.cvtColor(triplet, cv2.COLOR_RGB2BGR))
    return {
        "gt_path": str(gt_path),
        "render_path": str(render_path),
        "triplet_path": str(triplet_path),
    }
