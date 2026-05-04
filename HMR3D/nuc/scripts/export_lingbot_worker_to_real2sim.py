#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np


DEPTH_PNG_SCALE = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LingBot live worker outputs as RGB-D keyframes plus a "
            "projective TSDF volume, marching-cubes mesh, and Gaussian seed "
            "for real-to-sim geometry."
        )
    )
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", default="hikrobot_lingbot")
    parser.add_argument("--depth-scale", type=float, default=20.0)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=40.0)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-points-per-frame", type=int, default=25000)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--mesh-backend", choices=("tsdf", "voxel"), default="tsdf")
    parser.add_argument("--tsdf-trunc-multiplier", type=float, default=4.0)
    parser.add_argument("--tsdf-max-dim", type=int, default=180)
    parser.add_argument("--tsdf-chunk-voxels", type=int, default=150000)
    parser.add_argument("--tsdf-min-weight", type=float, default=1.0)
    parser.add_argument("--tsdf-weight-cap", type=float, default=32.0)
    parser.add_argument("--tsdf-bounds-percentile", type=float, default=99.0)
    parser.add_argument("--tsdf-max-frames", type=int, default=0)
    parser.add_argument("--mesh-max-voxels", type=int, default=120000)
    parser.add_argument("--gaussian-max-points", type=int, default=120000)
    parser.add_argument("--gaussian-scale", type=float, default=0.06)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (F,H,W) or (F,H,W,1), got {depth.shape}")
    return depth


def _iter_worker_results(worker_dir: Path) -> list[Path]:
    return sorted(worker_dir.glob("window_*/worker_result.json"))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _scale_intrinsic(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    return np.array(
        [
            [float(fx) * sx, 0.0, float(cx) * sx],
            [0.0, float(fy) * sy, float(cy) * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _unproject_points(
    depth: np.ndarray,
    conf: np.ndarray,
    rgb: np.ndarray,
    pose_c2w: np.ndarray,
    K: np.ndarray,
    *,
    min_depth: float,
    max_depth: float,
    min_conf: float,
    sample_stride: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = depth.shape
    stride = max(1, int(sample_stride))
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    xx, yy = np.meshgrid(xs, ys)
    x = xx.ravel()
    y = yy.ravel()
    z = depth[y, x].astype(np.float32)
    c = conf[y, x].astype(np.float32)
    valid = np.isfinite(z)
    valid &= z >= float(min_depth)
    valid &= z <= float(max_depth)
    valid &= c >= float(min_conf)
    if not np.any(valid):
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_rgb = np.zeros((0, 3), dtype=np.uint8)
        return empty_xyz, empty_rgb, np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    x = x[valid]
    y = y[valid]
    z = z[valid]
    c = c[valid]
    if max_points > 0 and z.shape[0] > max_points:
        # Deterministic even sampling keeps exports reproducible.
        keep = np.linspace(0, z.shape[0] - 1, int(max_points)).astype(np.int64)
        x, y, z, c = x[keep], y[keep], z[keep], c[keep]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    xyz_cam = np.stack(
        [
            (x.astype(np.float32) - cx) * z / max(fx, 1e-6),
            (y.astype(np.float32) - cy) * z / max(fy, 1e-6),
            z,
        ],
        axis=1,
    ).astype(np.float32)
    xyz_h = np.concatenate([xyz_cam, np.ones((xyz_cam.shape[0], 1), dtype=np.float32)], axis=1)
    xyz_world = (pose_c2w.astype(np.float32) @ xyz_h.T).T[:, :3].astype(np.float32)
    colors = rgb[y, x].astype(np.uint8)
    pixels = np.stack([x, y], axis=1).astype(np.int32)
    return xyz_world, colors, c.astype(np.float32), pixels


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {int(xyz.shape[0])}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(xyz, rgb):
            handle.write(
                f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _build_voxel_surface(
    xyz: np.ndarray,
    rgb: np.ndarray,
    conf: np.ndarray,
    voxel_size: float,
) -> dict[str, np.ndarray]:
    if xyz.shape[0] == 0:
        return {
            "voxel_indices": np.zeros((0, 3), dtype=np.int32),
            "centers": np.zeros((0, 3), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
            "confidence": np.zeros((0,), dtype=np.float32),
            "observations": np.zeros((0,), dtype=np.int32),
            "origin": np.zeros((3,), dtype=np.float32),
        }
    voxel_size = max(1e-4, float(voxel_size))
    origin = np.floor(np.min(xyz, axis=0) / voxel_size).astype(np.float32) * voxel_size
    indices = np.floor((xyz - origin[None, :]) / voxel_size).astype(np.int32)
    uniq, inverse = np.unique(indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    sums_xyz = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    sums_rgb = np.zeros((uniq.shape[0], 3), dtype=np.float64)
    sums_conf = np.zeros((uniq.shape[0],), dtype=np.float64)
    np.add.at(sums_xyz, inverse, xyz.astype(np.float64))
    np.add.at(sums_rgb, inverse, rgb.astype(np.float64))
    np.add.at(sums_conf, inverse, conf.astype(np.float64))
    centers = (sums_xyz / np.clip(counts[:, None], 1.0, None)).astype(np.float32)
    colors = np.clip(sums_rgb / np.clip(counts[:, None], 1.0, None), 0, 255).astype(np.uint8)
    confidence = (sums_conf / np.clip(counts, 1.0, None)).astype(np.float32)
    return {
        "voxel_indices": uniq.astype(np.int32),
        "centers": centers,
        "rgb": colors,
        "confidence": confidence,
        "observations": counts.astype(np.int32),
        "origin": origin.astype(np.float32),
    }


def _write_mesh_from_voxels(path: Path, centers: np.ndarray, rgb: np.ndarray, voxel_size: float, max_voxels: int) -> dict[str, Any]:
    if centers.shape[0] == 0:
        return {"path": "", "vertex_count": 0, "face_count": 0, "voxel_count": 0}
    try:
        import trimesh
        from trimesh.voxel import ops as voxel_ops
    except Exception as exc:
        return {"path": "", "vertex_count": 0, "face_count": 0, "voxel_count": int(centers.shape[0]), "error": str(exc)}

    keep = np.arange(centers.shape[0])
    if max_voxels > 0 and centers.shape[0] > max_voxels:
        keep = np.linspace(0, centers.shape[0] - 1, int(max_voxels)).astype(np.int64)
    mesh = voxel_ops.multibox(
        centers=centers[keep].astype(np.float64),
        pitch=float(voxel_size),
        colors=rgb[keep],
        remove_internal_faces=True,
    )
    if isinstance(mesh, tuple):
        vertices, faces = mesh[:2]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return {
        "path": str(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "voxel_count": int(len(keep)),
        "backend": "voxel_multibox",
    }


def _robust_bounds(xyz: np.ndarray, percentile: float) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] == 0:
        return np.zeros((3,), dtype=np.float32), np.zeros((3,), dtype=np.float32)
    percentile = float(np.clip(percentile, 50.0, 100.0))
    if percentile >= 99.999:
        lo = np.min(xyz, axis=0)
        hi = np.max(xyz, axis=0)
    else:
        tail = (100.0 - percentile) * 0.5
        lo = np.percentile(xyz, tail, axis=0)
        hi = np.percentile(xyz, 100.0 - tail, axis=0)
    return lo.astype(np.float32), hi.astype(np.float32)


def _tsdf_volume_spec(
    xyz: np.ndarray,
    requested_voxel_size: float,
    max_dim: int,
    trunc_multiplier: float,
    bounds_percentile: float,
) -> dict[str, Any]:
    if xyz.shape[0] == 0:
        raise ValueError("Cannot build TSDF volume without points")
    lo, hi = _robust_bounds(xyz, bounds_percentile)
    requested_voxel_size = max(1e-4, float(requested_voxel_size))
    max_dim = max(16, int(max_dim))
    trunc_multiplier = max(1.0, float(trunc_multiplier))

    voxel_size = requested_voxel_size
    for _ in range(3):
        trunc = voxel_size * trunc_multiplier
        origin = lo - trunc
        upper = hi + trunc
        extent = np.maximum(upper - origin, voxel_size)
        dims = np.ceil(extent / voxel_size).astype(np.int32) + 1
        largest = int(np.max(dims))
        if largest <= max_dim:
            break
        voxel_size *= float(largest) / float(max_dim)

    trunc = voxel_size * trunc_multiplier
    origin = lo - trunc
    upper = hi + trunc
    extent = np.maximum(upper - origin, voxel_size)
    dims = np.ceil(extent / voxel_size).astype(np.int32) + 1
    dims = np.maximum(dims, 2)
    return {
        "origin": origin.astype(np.float32),
        "upper": upper.astype(np.float32),
        "dims": tuple(int(v) for v in dims.tolist()),
        "voxel_size": float(voxel_size),
        "requested_voxel_size": float(requested_voxel_size),
        "truncation": float(trunc),
        "bounds_percentile": float(bounds_percentile),
    }


def _integrate_tsdf(
    frames: list[dict[str, np.ndarray]],
    xyz_bounds_source: np.ndarray,
    *,
    requested_voxel_size: float,
    max_dim: int,
    trunc_multiplier: float,
    bounds_percentile: float,
    chunk_voxels: int,
    min_depth: float,
    max_depth: float,
    min_conf: float,
    weight_cap: float,
) -> dict[str, Any]:
    spec = _tsdf_volume_spec(
        xyz_bounds_source,
        requested_voxel_size=requested_voxel_size,
        max_dim=max_dim,
        trunc_multiplier=trunc_multiplier,
        bounds_percentile=bounds_percentile,
    )
    dims = spec["dims"]
    origin = spec["origin"].astype(np.float32)
    voxel_size = float(spec["voxel_size"])
    trunc = float(spec["truncation"])
    total_voxels = int(np.prod(dims))
    tsdf = np.ones(dims, dtype=np.float32)
    weights = np.zeros(dims, dtype=np.float32)
    color = np.zeros((*dims, 3), dtype=np.float32)
    tsdf_flat = tsdf.reshape(-1)
    weights_flat = weights.reshape(-1)
    color_flat = color.reshape(-1, 3)
    chunk_voxels = max(4096, int(chunk_voxels))
    weight_cap = max(1.0, float(weight_cap))
    nx, ny, nz = dims

    for frame in frames:
        depth = frame["depth"].astype(np.float32, copy=False)
        conf = frame["conf"].astype(np.float32, copy=False)
        rgb = frame["rgb"].astype(np.float32, copy=False)
        K = frame["K"].astype(np.float32, copy=False)
        pose = frame["pose"].astype(np.float32, copy=False)
        try:
            world_to_cam = np.linalg.inv(pose).astype(np.float32)
        except np.linalg.LinAlgError:
            continue
        h, w = depth.shape
        for start in range(0, total_voxels, chunk_voxels):
            stop = min(start + chunk_voxels, total_voxels)
            flat = np.arange(start, stop, dtype=np.int64)
            iz = flat % nz
            iy = (flat // nz) % ny
            ix = flat // (ny * nz)
            xyz_world = np.stack(
                [
                    origin[0] + (ix.astype(np.float32) + 0.5) * voxel_size,
                    origin[1] + (iy.astype(np.float32) + 0.5) * voxel_size,
                    origin[2] + (iz.astype(np.float32) + 0.5) * voxel_size,
                ],
                axis=1,
            )
            xyz_cam = xyz_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3][None, :]
            z = xyz_cam[:, 2]
            valid = z > max(float(min_depth), 1e-4)
            valid &= z < float(max_depth) + trunc
            if not np.any(valid):
                continue
            x = xyz_cam[:, 0]
            y = xyz_cam[:, 1]
            u = np.rint(K[0, 0] * x / np.maximum(z, 1e-6) + K[0, 2]).astype(np.int32)
            v = np.rint(K[1, 1] * y / np.maximum(z, 1e-6) + K[1, 2]).astype(np.int32)
            valid &= u >= 0
            valid &= u < w
            valid &= v >= 0
            valid &= v < h
            if not np.any(valid):
                continue
            valid_idx = np.flatnonzero(valid)
            obs_depth = depth[v[valid_idx], u[valid_idx]]
            obs_conf = conf[v[valid_idx], u[valid_idx]]
            obs_valid = np.isfinite(obs_depth)
            obs_valid &= obs_depth >= float(min_depth)
            obs_valid &= obs_depth <= float(max_depth)
            obs_valid &= obs_conf >= float(min_conf)
            if not np.any(obs_valid):
                continue
            valid_idx = valid_idx[obs_valid]
            obs_depth = obs_depth[obs_valid]
            obs_conf = obs_conf[obs_valid]
            sdf = obs_depth - z[valid_idx]
            sdf_valid = sdf >= -trunc
            if not np.any(sdf_valid):
                continue
            valid_idx = valid_idx[sdf_valid]
            sdf = sdf[sdf_valid]
            obs_conf = obs_conf[sdf_valid]
            ids = flat[valid_idx]
            tsdf_obs = np.clip(sdf / trunc, -1.0, 1.0).astype(np.float32)
            obs_weight = np.clip(obs_conf.astype(np.float32), 0.25, 4.0)
            old_weight = weights_flat[ids]
            new_weight = np.minimum(old_weight + obs_weight, weight_cap)
            denom = np.maximum(old_weight + obs_weight, 1e-6)
            tsdf_flat[ids] = (old_weight * tsdf_flat[ids] + obs_weight * tsdf_obs) / denom
            obs_rgb = rgb[v[valid_idx], u[valid_idx]]
            color_flat[ids] = (old_weight[:, None] * color_flat[ids] + obs_weight[:, None] * obs_rgb) / denom[:, None]
            weights_flat[ids] = new_weight

    color_u8 = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    spec.update(
        {
            "tsdf": tsdf,
            "weights": weights,
            "rgb": color_u8,
            "integrated_frame_count": int(len(frames)),
            "observed_voxel_count": int(np.count_nonzero(weights > 0)),
        }
    )
    return spec


def _surface_points_from_tsdf(tsdf_data: dict[str, Any], min_weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tsdf = tsdf_data["tsdf"]
    weights = tsdf_data["weights"]
    rgb = tsdf_data["rgb"]
    origin = tsdf_data["origin"].astype(np.float32)
    voxel_size = float(tsdf_data["voxel_size"])
    mask = (weights >= float(min_weight)) & (np.abs(tsdf) <= 0.35)
    if not np.any(mask):
        mask = weights >= float(min_weight)
    indices = np.argwhere(mask)
    if indices.shape[0] == 0:
        empty_xyz = np.zeros((0, 3), dtype=np.float32)
        empty_rgb = np.zeros((0, 3), dtype=np.uint8)
        return empty_xyz, empty_rgb, np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    centers = origin[None, :] + (indices.astype(np.float32) + 0.5) * voxel_size
    colors = rgb[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.uint8)
    confidence = weights[indices[:, 0], indices[:, 1], indices[:, 2]].astype(np.float32)
    observations = np.rint(confidence).astype(np.int32)
    return centers.astype(np.float32), colors, confidence, observations


def _write_mesh_from_tsdf(path: Path, tsdf_data: dict[str, Any], min_weight: float) -> dict[str, Any]:
    try:
        from trimesh.voxel import ops as voxel_ops
    except Exception as exc:
        return {"path": "", "vertex_count": 0, "face_count": 0, "backend": "tsdf_marching_cubes", "error": str(exc)}

    tsdf = tsdf_data["tsdf"].astype(np.float32, copy=True)
    weights = tsdf_data["weights"]
    rgb = tsdf_data["rgb"]
    origin = tsdf_data["origin"].astype(np.float32)
    voxel_size = float(tsdf_data["voxel_size"])
    tsdf[weights < float(min_weight)] = 1.0
    if not (np.any(tsdf < 0.0) and np.any(tsdf > 0.0)):
        return {
            "path": "",
            "vertex_count": 0,
            "face_count": 0,
            "backend": "tsdf_marching_cubes",
            "error": "TSDF volume has no zero crossing",
        }
    try:
        mesh = voxel_ops.matrix_to_marching_cubes(tsdf, pitch=voxel_size, threshold=0.0)
    except Exception as exc:
        return {"path": "", "vertex_count": 0, "face_count": 0, "backend": "tsdf_marching_cubes", "error": str(exc)}
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return {
            "path": "",
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "backend": "tsdf_marching_cubes",
            "error": "marching cubes returned an empty mesh",
        }
    mesh.vertices = mesh.vertices + origin[None, :]
    local = np.rint((np.asarray(mesh.vertices) - origin[None, :]) / max(voxel_size, 1e-6)).astype(np.int32)
    local[:, 0] = np.clip(local[:, 0], 0, rgb.shape[0] - 1)
    local[:, 1] = np.clip(local[:, 1], 0, rgb.shape[1] - 1)
    local[:, 2] = np.clip(local[:, 2], 0, rgb.shape[2] - 1)
    vertex_rgb = rgb[local[:, 0], local[:, 1], local[:, 2]]
    alpha = np.full((vertex_rgb.shape[0], 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.concatenate([vertex_rgb.astype(np.uint8), alpha], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return {
        "path": str(path),
        "vertex_count": int(len(mesh.vertices)),
        "face_count": int(len(mesh.faces)),
        "backend": "tsdf_marching_cubes",
        "voxel_size": voxel_size,
        "truncation": float(tsdf_data["truncation"]),
        "observed_voxel_count": int(tsdf_data["observed_voxel_count"]),
        "integrated_frame_count": int(tsdf_data["integrated_frame_count"]),
    }


def _write_gaussian_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    scale: np.ndarray,
    opacity: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {int(xyz.shape[0])}\n")
        for prop in (
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
        ):
            handle.write(prop + "\n")
        handle.write("end_header\n")
        for p, c, s, o, u, v in zip(xyz, rgb, scale, opacity, axis_u, axis_v):
            handle.write(
                f"{float(p[0]):.6f} {float(p[1]):.6f} {float(p[2]):.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} "
                f"{float(s):.6f} {float(o):.6f} "
                f"{float(u[0]):.6f} {float(u[1]):.6f} {float(u[2]):.6f} "
                f"{float(v[0]):.6f} {float(v[1]):.6f} {float(v[2]):.6f}\n"
            )


def _write_gaussian_seed(
    output_dir: Path,
    centers: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    observations: np.ndarray,
    *,
    max_points: int,
    base_scale: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if centers.shape[0] == 0:
        return {"point_count": 0, "npz": "", "ply": ""}
    score = confidence.astype(np.float32) + 0.1 * np.minimum(observations.astype(np.float32), 8.0)
    keep_count = centers.shape[0] if max_points <= 0 else min(int(max_points), centers.shape[0])
    if centers.shape[0] > keep_count:
        keep = np.argpartition(score, -keep_count)[-keep_count:]
        keep = keep[np.argsort(score[keep])[::-1]]
    else:
        keep = np.arange(centers.shape[0])
    xyz = centers[keep].astype(np.float32)
    colors = rgb[keep].astype(np.uint8)
    conf = confidence[keep].astype(np.float32)
    obs = observations[keep].astype(np.float32)
    scale = np.full((xyz.shape[0],), float(base_scale), dtype=np.float32)
    opacity = np.clip(0.45 + 0.08 * np.minimum(obs, 6.0) + 0.04 * np.log1p(np.clip(conf, 0, None)), 0.18, 0.96).astype(np.float32)
    axis_u = np.tile(np.array([[float(base_scale), 0.0, 0.0]], dtype=np.float32), (xyz.shape[0], 1))
    axis_v = np.tile(np.array([[0.0, float(base_scale), 0.0]], dtype=np.float32), (xyz.shape[0], 1))
    gsplat_scales = np.stack(
        [
            scale,
            scale,
            np.maximum(0.35 * scale, 1e-4),
        ],
        axis=1,
    ).astype(np.float32)
    gsplat_quats = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    gsplat_quats[:, 0] = 1.0
    unstable = np.clip(0.8 / np.sqrt(np.clip(conf, 0.2, None)), 0.1, 0.8).astype(np.float32)
    recentness = np.clip(0.5 + 0.05 * np.minimum(obs, 6.0), 0.35, 0.95).astype(np.float32)
    source = np.full((xyz.shape[0],), 6, dtype=np.int8)
    npz_path = output_dir / "gaussians_seed.npz"
    ply_path = output_dir / "gaussians_seed.ply"
    np.savez_compressed(
        npz_path,
        xyz=xyz,
        rgb=colors,
        scale=scale,
        opacity=opacity,
        axis_u=axis_u,
        axis_v=axis_v,
        gsplat_scales=gsplat_scales,
        gsplat_quats=gsplat_quats,
        unstable=unstable,
        recentness=recentness,
        source=source,
        confidence=conf,
    )
    _write_gaussian_ply(ply_path, xyz, colors, scale, opacity, axis_u, axis_v)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "lingbot_gaussian_seed.v1",
                "npz": str(npz_path),
                "ply": str(ply_path),
                "point_count": int(xyz.shape[0]),
                "base_scale": float(base_scale),
                "source": "real2sim_tsdf_surface",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "point_count": int(xyz.shape[0]),
        "npz": str(npz_path),
        "ply": str(ply_path),
        "manifest": str(manifest_path),
    }


def main() -> int:
    args = parse_args()
    worker_dir = Path(args.worker_dir).expanduser().resolve()
    out_root = Path(args.output_dir).expanduser().resolve()
    seq_dir = out_root / args.sequence
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    pose_dir = seq_dir / "pose"
    conf_dir = seq_dir / "confidence"
    if seq_dir.exists() and args.overwrite:
        shutil.rmtree(seq_dir)
    for directory in (rgb_dir, depth_dir, pose_dir, conf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exported_frames: list[dict[str, Any]] = []
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    tsdf_frames: list[dict[str, np.ndarray]] = []
    seen: set[int] = set()

    for result_path in _iter_worker_results(worker_dir):
        if args.max_frames > 0 and len(exported_frames) >= args.max_frames:
            break
        result = _read_json(result_path)
        if result is None:
            continue
        summary = _read_json(Path(result["summary_json"]))
        if summary is None:
            continue
        pred = np.load(result["predictions_npz"])
        depth = _squeeze_depth(pred["depth"]) * float(args.depth_scale)
        conf = _squeeze_depth(pred["depth_conf"]) if "depth_conf" in pred else np.ones_like(depth, dtype=np.float32)
        frames = summary.get("metadata", {}).get("frames", [])
        for local_idx, frame in enumerate(frames[: depth.shape[0]]):
            if args.max_frames > 0 and len(exported_frames) >= args.max_frames:
                break
            frame_idx = int(frame.get("frame_idx", local_idx))
            if frame_idx in seen:
                continue
            meta = frame.get("metadata") or {}
            if "pose" not in meta:
                continue
            image_path = Path(frame.get("image_path", "")).expanduser()
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            src_shape = tuple(meta.get("frame_shape", rgb_full.shape[:2]))
            if len(src_shape) != 2 or int(src_shape[0]) <= 0:
                src_shape = rgb_full.shape[:2]
            d = depth[local_idx].astype(np.float32)
            c = conf[local_idx].astype(np.float32)
            h, w = d.shape
            rgb = cv2.resize(rgb_full, (w, h), interpolation=cv2.INTER_AREA)
            pose = np.asarray(meta["pose"], dtype=np.float32)
            if pose.shape == (3, 4):
                pose4 = np.eye(4, dtype=np.float32)
                pose4[:3, :4] = pose
                pose = pose4
            fx = float(args.fx or src_shape[1])
            fy = float(args.fy or src_shape[1])
            cx = float(args.cx or src_shape[1] / 2.0)
            cy = float(args.cy or src_shape[0] / 2.0)
            K = _scale_intrinsic(fx, fy, cx, cy, (int(src_shape[0]), int(src_shape[1])), (h, w))

            export_idx = len(exported_frames)
            name = f"{export_idx:06d}"
            cv2.imwrite(str(rgb_dir / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            np.save(depth_dir / f"{name}.npy", d)
            depth_png = np.clip(np.rint(d * DEPTH_PNG_SCALE), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            cv2.imwrite(str(depth_dir / f"{name}.png"), depth_png)
            np.save(conf_dir / f"{name}.npy", c)
            np.savetxt(pose_dir / f"{name}.txt", pose, fmt="%.9f")

            xyz, colors, point_conf, _pixels = _unproject_points(
                d,
                c,
                rgb,
                pose,
                K,
                min_depth=args.min_depth,
                max_depth=args.max_depth,
                min_conf=args.min_conf,
                sample_stride=args.sample_stride,
                max_points=args.max_points_per_frame,
            )
            if xyz.shape[0] > 0:
                all_xyz.append(xyz)
                all_rgb.append(colors)
                all_conf.append(point_conf)
            if args.tsdf_max_frames <= 0 or len(tsdf_frames) < int(args.tsdf_max_frames):
                tsdf_frames.append(
                    {
                        "depth": d,
                        "conf": c,
                        "rgb": rgb,
                        "pose": pose,
                        "K": K,
                    }
                )

            exported_frames.append(
                {
                    "index": export_idx,
                    "source_frame_idx": frame_idx,
                    "source_window": int(result.get("index", -1)),
                    "rgb": f"rgb/{name}.png",
                    "depth_npy": f"depth/{name}.npy",
                    "depth_png": f"depth/{name}.png",
                    "confidence_npy": f"confidence/{name}.npy",
                    "pose_txt": f"pose/{name}.txt",
                    "source_image_path": str(image_path),
                    "intrinsics": K.astype(float).tolist(),
                    "depth_scale": float(args.depth_scale),
                    "point_count": int(xyz.shape[0]),
                }
            )
            seen.add(frame_idx)

    if not exported_frames:
        raise SystemExit(f"No RGB-D keyframes exported from {worker_dir}")

    xyz_all = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3), dtype=np.float32)
    rgb_all = np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), dtype=np.uint8)
    conf_all = np.concatenate(all_conf, axis=0) if all_conf else np.zeros((0,), dtype=np.float32)

    geom_dir = seq_dir / "geometry"
    geom_dir.mkdir(parents=True, exist_ok=True)
    _write_ascii_ply(geom_dir / "scene_points.ply", xyz_all, rgb_all)

    voxel = _build_voxel_surface(xyz_all, rgb_all, conf_all, float(args.voxel_size))
    _write_ascii_ply(geom_dir / "scene_voxel_points.ply", voxel["centers"], voxel["rgb"])
    tsdf_data: dict[str, Any] | None = None
    mesh_summary: dict[str, Any]
    gaussian_xyz = voxel["centers"]
    gaussian_rgb = voxel["rgb"]
    gaussian_conf = voxel["confidence"]
    gaussian_obs = voxel["observations"]
    if args.mesh_backend == "tsdf":
        tsdf_data = _integrate_tsdf(
            tsdf_frames,
            xyz_all,
            requested_voxel_size=float(args.voxel_size),
            max_dim=int(args.tsdf_max_dim),
            trunc_multiplier=float(args.tsdf_trunc_multiplier),
            bounds_percentile=float(args.tsdf_bounds_percentile),
            chunk_voxels=int(args.tsdf_chunk_voxels),
            min_depth=float(args.min_depth),
            max_depth=float(args.max_depth),
            min_conf=float(args.min_conf),
            weight_cap=float(args.tsdf_weight_cap),
        )
        np.savez_compressed(
            geom_dir / "scene_tsdf.npz",
            tsdf=tsdf_data["tsdf"],
            weights=tsdf_data["weights"],
            rgb=tsdf_data["rgb"],
            origin=tsdf_data["origin"],
            upper=tsdf_data["upper"],
            dims=np.array(tsdf_data["dims"], dtype=np.int32),
            voxel_size=np.array([float(tsdf_data["voxel_size"])], dtype=np.float32),
            requested_voxel_size=np.array([float(tsdf_data["requested_voxel_size"])], dtype=np.float32),
            truncation=np.array([float(tsdf_data["truncation"])], dtype=np.float32),
            bounds_percentile=np.array([float(tsdf_data["bounds_percentile"])], dtype=np.float32),
            integrated_frame_count=np.array([int(tsdf_data["integrated_frame_count"])], dtype=np.int32),
            observed_voxel_count=np.array([int(tsdf_data["observed_voxel_count"])], dtype=np.int32),
        )
        mesh_summary = _write_mesh_from_tsdf(geom_dir / "scene_mesh.ply", tsdf_data, float(args.tsdf_min_weight))
        surface_xyz, surface_rgb, surface_conf, surface_obs = _surface_points_from_tsdf(tsdf_data, float(args.tsdf_min_weight))
        if surface_xyz.shape[0] > 0:
            _write_ascii_ply(geom_dir / "scene_tsdf_surface_points.ply", surface_xyz, surface_rgb)
            gaussian_xyz = surface_xyz
            gaussian_rgb = surface_rgb
            gaussian_conf = surface_conf
            gaussian_obs = surface_obs
    else:
        np.savez_compressed(
            geom_dir / "scene_voxel_surface.npz",
            xyz=xyz_all,
            rgb=rgb_all,
            confidence=conf_all,
            voxel_indices=voxel["voxel_indices"],
            voxel_centers=voxel["centers"],
            voxel_rgb=voxel["rgb"],
            voxel_confidence=voxel["confidence"],
            voxel_observations=voxel["observations"],
            voxel_origin=voxel["origin"],
            voxel_size=np.array([float(args.voxel_size)], dtype=np.float32),
        )
        mesh_summary = _write_mesh_from_voxels(
            geom_dir / "scene_mesh.ply",
            voxel["centers"],
            voxel["rgb"],
            float(args.voxel_size),
            int(args.mesh_max_voxels),
        )
    gaussian_summary = _write_gaussian_seed(
        seq_dir / "gaussian_seed",
        gaussian_xyz,
        gaussian_rgb,
        gaussian_conf,
        gaussian_obs,
        max_points=int(args.gaussian_max_points),
        base_scale=float(args.gaussian_scale),
    )

    manifest = {
        "schema": "lingbot_real2sim_archive.v1",
        "sequence": args.sequence,
        "worker_dir": str(worker_dir),
        "output_dir": str(seq_dir),
        "frame_count": len(exported_frames),
        "depth_png_scale": DEPTH_PNG_SCALE,
        "depth_scale": float(args.depth_scale),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "min_conf": float(args.min_conf),
        "sample_stride": int(args.sample_stride),
        "voxel_size": float(args.voxel_size),
        "mesh_backend": str(args.mesh_backend),
        "tsdf": (
            {
                "voxel_size": float(tsdf_data["voxel_size"]),
                "requested_voxel_size": float(tsdf_data["requested_voxel_size"]),
                "truncation": float(tsdf_data["truncation"]),
                "dims": list(tsdf_data["dims"]),
                "max_dim": int(args.tsdf_max_dim),
                "bounds_percentile": float(tsdf_data["bounds_percentile"]),
                "integrated_frame_count": int(tsdf_data["integrated_frame_count"]),
                "observed_voxel_count": int(tsdf_data["observed_voxel_count"]),
            }
            if tsdf_data is not None
            else None
        ),
        "raw_point_count": int(xyz_all.shape[0]),
        "voxel_count": int(voxel["centers"].shape[0]),
        "mesh": mesh_summary,
        "gaussian_seed": gaussian_summary,
        "frames": exported_frames,
        "geometry": {
            "tsdf_npz": "geometry/scene_tsdf.npz" if tsdf_data is not None else "",
            "points_ply": "geometry/scene_points.ply",
            "voxel_points_ply": "geometry/scene_voxel_points.ply",
            "tsdf_surface_points_ply": "geometry/scene_tsdf_surface_points.ply" if tsdf_data is not None else "",
            "mesh_ply": "geometry/scene_mesh.ply" if mesh_summary.get("path") else "",
        },
    }
    (seq_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(seq_dir),
        "frame_count": len(exported_frames),
        "raw_point_count": int(xyz_all.shape[0]),
        "voxel_count": int(voxel["centers"].shape[0]),
        "mesh_backend": str(args.mesh_backend),
        "mesh": mesh_summary,
        "gaussian_seed": gaussian_summary,
        "manifest": str(seq_dir / "manifest.json"),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
