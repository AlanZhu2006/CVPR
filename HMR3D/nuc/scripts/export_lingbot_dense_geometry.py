from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize LingBot predictions into a dense-geometry schema aligned "
            "to cuVSLAM poses when available."
        )
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--max-points-per-frame", type=int, default=25000)
    parser.add_argument("--output-name", default="lingbot_dense_geometry.npz")
    parser.add_argument(
        "--geometry-source",
        default="auto",
        choices=["auto", "cuvslam", "lingbot_depth", "world_points"],
        help=(
            "Source used to build world points. auto preserves the old priority "
            "order when cuVSLAM poses are present, then falls back to LingBot "
            "world_points, then official LingBot depth+extrinsic unprojection."
        ),
    )
    parser.add_argument(
        "--pose-align",
        default="auto",
        choices=["auto", "off"],
        help=(
            "When cuVSLAM poses and LingBot extrinsics are both available, "
            "optionally fit a window-level Sim(3) alignment and backproject "
            "using aligned LingBot poses."
        ),
    )
    parser.add_argument(
        "--depth-scale",
        default="auto",
        help=(
            "Scale factor applied to LingBot depth before backprojection. "
            "Use a positive float to force a fixed scale, or 'auto' to fit "
            "a window-level scale from LingBot extrinsics to cuVSLAM poses."
        ),
    )
    return parser.parse_args()


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (F,H,W) or (F,H,W,1), got {depth.shape}")
    return depth


def _pose_array(raw_poses: Any, frame_count: int) -> np.ndarray:
    if raw_poses is None:
        return np.zeros((0, 4, 4), dtype=np.float32)
    try:
        if len(raw_poses) == 0:
            return np.zeros((0, 4, 4), dtype=np.float32)
    except TypeError:
        pass
    if isinstance(raw_poses, str) and not raw_poses.strip():
        return np.zeros((0, 4, 4), dtype=np.float32)
    poses = np.asarray(raw_poses, dtype=np.float32)
    if poses.ndim != 3:
        return np.zeros((0, 4, 4), dtype=np.float32)
    if poses.shape[1:] == (3, 4):
        full = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], poses.shape[0], axis=0)
        full[:, :3, :4] = poses
        poses = full
    if poses.shape[1:] != (4, 4):
        return np.zeros((0, 4, 4), dtype=np.float32)
    return poses[:frame_count].astype(np.float32)


def _parse_depth_scale_arg(raw_value: str) -> float | None:
    value = str(raw_value or "").strip().lower()
    if value in {"", "auto"}:
        return None
    scale = float(raw_value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"depth_scale must be positive, got {raw_value!r}")
    return float(scale)


def _camera_centers_from_poses(poses: np.ndarray, mode: str) -> np.ndarray:
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return np.zeros((0, 3), dtype=np.float32)
    rotation = poses[:, :3, :3].astype(np.float32)
    translation = poses[:, :3, 3].astype(np.float32)
    if mode == "direct_translation":
        return translation
    if mode == "inverse_center":
        return -(np.transpose(rotation, (0, 2, 1)) @ translation[:, :, None])[:, :, 0]
    raise ValueError(f"Unknown camera-center mode: {mode}")


def _pose_candidates(lingbot_extrinsic: np.ndarray | None) -> dict[str, np.ndarray]:
    if lingbot_extrinsic is None or lingbot_extrinsic.ndim != 3 or lingbot_extrinsic.shape[1:] != (4, 4):
        return {}
    direct = lingbot_extrinsic.astype(np.float32)
    inverse = np.linalg.inv(direct).astype(np.float32)
    return {
        "direct_pose": direct,
        "inverse_pose": inverse,
    }


def _rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a.T @ rotation_b
    trace_term = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace_term)))


def _umeyama_similarity(source_xyz: np.ndarray, target_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source_xyz.shape != target_xyz.shape or source_xyz.ndim != 2 or source_xyz.shape[1] != 3:
        raise ValueError("source_xyz and target_xyz must both have shape (N,3)")
    count = int(source_xyz.shape[0])
    if count < 3:
        raise ValueError("Need at least 3 points for similarity fit")

    src_mean = np.mean(source_xyz, axis=0)
    dst_mean = np.mean(target_xyz, axis=0)
    src_centered = source_xyz - src_mean
    dst_centered = target_xyz - dst_mean
    covariance = (dst_centered.T @ src_centered) / max(count, 1)
    left_u, singular_values, right_vt = np.linalg.svd(covariance)
    reflect = np.eye(3, dtype=np.float64)
    if np.linalg.det(left_u) * np.linalg.det(right_vt) < 0.0:
        reflect[-1, -1] = -1.0
    rotation = left_u @ reflect @ right_vt
    src_var = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    scale = float(np.trace(np.diag(singular_values) @ reflect) / max(src_var, 1e-12))
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _apply_similarity_to_poses(
    poses: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], poses.shape[0], axis=0)
    aligned[:, :3, :3] = np.einsum("ij,njk->nik", rotation.astype(np.float32), poses[:, :3, :3].astype(np.float32))
    aligned[:, :3, 3] = (float(scale) * (rotation.astype(np.float32) @ poses[:, :3, 3].astype(np.float32).T)).T + translation.astype(np.float32)
    return aligned.astype(np.float32)


def _estimate_pose_alignment(
    cuvslam_poses: np.ndarray,
    lingbot_candidates: dict[str, np.ndarray],
) -> dict[str, Any]:
    fallback = {
        "enabled": False,
        "source": "disabled",
        "valid_pairs": 0,
    }
    if cuvslam_poses.shape[0] < 3 or not lingbot_candidates:
        return fallback

    target_centers = cuvslam_poses[:, :3, 3].astype(np.float64)
    target_rotations = cuvslam_poses[:, :3, :3].astype(np.float32)
    best_fit: dict[str, Any] | None = None
    for mode_name, source_poses in lingbot_candidates.items():
        if source_poses.shape[0] < 3:
            continue
        source_centers = source_poses[:, :3, 3].astype(np.float64)
        valid = np.isfinite(source_centers).all(axis=1) & np.isfinite(target_centers).all(axis=1)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 3:
            continue

        scale, sim_rotation, sim_translation = _umeyama_similarity(
            source_centers[valid_indices],
            target_centers[valid_indices],
        )
        aligned_centers = (scale * (sim_rotation.astype(np.float64) @ source_centers[valid_indices].T)).T + sim_translation.astype(np.float64)
        center_error = np.linalg.norm(aligned_centers - target_centers[valid_indices], axis=1)

        rotation_errors = []
        for local_idx, frame_idx in enumerate(valid_indices.tolist()):
            aligned_rotation = sim_rotation @ source_poses[frame_idx, :3, :3].astype(np.float32)
            target_rotation = target_rotations[frame_idx]
            rotation_errors.append(_rotation_error_deg(aligned_rotation, target_rotation))
        rotation_errors_arr = np.asarray(rotation_errors, dtype=np.float32)

        fit = {
            "enabled": True,
            "source": f"auto:{mode_name}",
            "lingbot_pose_mode": mode_name,
            "scale": float(scale),
            "valid_pairs": int(valid_indices.size),
            "center_median_error": float(np.median(center_error)),
            "center_mean_error": float(np.mean(center_error)),
            "center_max_error": float(np.max(center_error)),
            "rotation_median_error_deg": float(np.median(rotation_errors_arr)),
            "rotation_mean_error_deg": float(np.mean(rotation_errors_arr)),
            "rotation_max_error_deg": float(np.max(rotation_errors_arr)),
            "similarity_rotation": sim_rotation.astype(np.float32).tolist(),
            "similarity_translation": sim_translation.astype(np.float32).tolist(),
        }
        if best_fit is None:
            best_fit = fit
            continue
        if fit["center_median_error"] < best_fit["center_median_error"] - 1e-6:
            best_fit = fit
            continue
        if abs(fit["center_median_error"] - best_fit["center_median_error"]) <= 1e-6 and fit["rotation_median_error_deg"] < best_fit["rotation_median_error_deg"]:
            best_fit = fit

    return best_fit or fallback


def _estimate_depth_scale(
    cuvslam_poses: np.ndarray,
    lingbot_extrinsic: np.ndarray | None,
) -> dict[str, Any]:
    fallback = {
        "enabled": False,
        "scale": 1.0,
        "source": "identity",
        "valid_pairs": 0,
    }
    if lingbot_extrinsic is None or lingbot_extrinsic.shape[0] < 2 or cuvslam_poses.shape[0] < 2:
        return fallback

    cuv_centers = _camera_centers_from_poses(cuvslam_poses, mode="direct_translation")
    cuv_steps = np.linalg.norm(np.diff(cuv_centers, axis=0), axis=1)
    if cuv_steps.size == 0:
        return fallback

    best_fit: dict[str, Any] | None = None
    for mode in ("direct_translation", "inverse_center"):
        lingbot_centers = _camera_centers_from_poses(lingbot_extrinsic, mode=mode)
        if lingbot_centers.shape[0] < 2:
            continue
        lingbot_steps = np.linalg.norm(np.diff(lingbot_centers, axis=0), axis=1)
        valid = np.isfinite(cuv_steps) & np.isfinite(lingbot_steps)
        valid &= cuv_steps > 1e-4
        valid &= lingbot_steps > 1e-6
        if int(np.count_nonzero(valid)) <= 0:
            continue

        ratios = cuv_steps[valid] / lingbot_steps[valid]
        scale = float(np.median(ratios))
        if not np.isfinite(scale) or scale <= 0.0:
            continue

        rel_error = np.abs(cuv_steps[valid] - (lingbot_steps[valid] * scale)) / np.clip(cuv_steps[valid], 1e-6, None)
        fit = {
            "enabled": True,
            "scale": scale,
            "source": f"auto:{mode}",
            "mode": mode,
            "valid_pairs": int(np.count_nonzero(valid)),
            "median_rel_error": float(np.median(rel_error)),
            "mean_rel_error": float(np.mean(rel_error)),
            "target_step_mean": float(np.mean(cuv_steps[valid])),
            "source_step_mean": float(np.mean(lingbot_steps[valid])),
        }
        if best_fit is None:
            best_fit = fit
            continue
        if fit["valid_pairs"] > best_fit["valid_pairs"]:
            best_fit = fit
            continue
        if fit["valid_pairs"] == best_fit["valid_pairs"] and fit["median_rel_error"] < best_fit["median_rel_error"]:
            best_fit = fit

    return best_fit or fallback


def _resolve_one_image(path_str: str, summary_dir: Path, image_root: Path | None) -> str:
    path = Path(path_str).expanduser()
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    if image_root is not None:
        candidates.append(image_root / path)
        candidates.append(image_root / path.name)
        candidates.append(image_root / "images" / path.name)
    candidates.append(summary_dir / path)
    candidates.append(summary_dir / path.name)
    candidates.append(summary_dir.parent / "images" / path.name)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _resolve_image_paths(
    summary: dict[str, Any],
    summary_json: Path,
    image_root_arg: str,
) -> list[str]:
    image_root = Path(image_root_arg).expanduser().resolve() if image_root_arg else None
    summary_dir = summary_json.parent
    metadata = summary.get("metadata", {})
    primary = list(summary.get("image_paths", []))
    fallback = list(metadata.get("original_image_paths", []))
    resolved = []
    for idx in range(max(len(primary), len(fallback))):
        candidates = []
        if idx < len(primary):
            candidates.append(primary[idx])
        if idx < len(fallback):
            candidates.append(fallback[idx])
        resolved_path = ""
        for candidate in candidates:
            resolved_path = _resolve_one_image(candidate, summary_dir, image_root)
            if resolved_path:
                break
        resolved.append(resolved_path)
    return resolved


def _load_rgb(path_str: str) -> np.ndarray | None:
    if not path_str:
        return None
    frame_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    return frame_bgr[:, :, ::-1].astype(np.uint8)


def _sample_rgb(rgb: np.ndarray | None, xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb is None:
        return np.full((xs.shape[0], 3), 127, dtype=np.uint8)
    sample_x = np.clip((xs.astype(np.float32) + 0.5) * rgb.shape[1] / max(width, 1), 0, rgb.shape[1] - 1)
    sample_y = np.clip((ys.astype(np.float32) + 0.5) * rgb.shape[0] / max(height, 1), 0, rgb.shape[0] - 1)
    return rgb[sample_y.astype(np.int32), sample_x.astype(np.int32)].astype(np.uint8)


def _backproject_pixels(
    K: np.ndarray,
    pose: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = depth.astype(np.float32)
    x_cam = (xs.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y_cam = (ys.astype(np.float32) - cy) * z / max(fy, 1e-6)
    points_camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    points_world = (points_camera @ pose[:3, :3].T) + pose[:3, 3]
    return points_camera.astype(np.float32), points_world.astype(np.float32)


def _estimate_normals(
    centers: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    normals = np.cross(axis_u - centers, axis_v - centers).astype(np.float32)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    normals[valid] /= norm[valid]
    normals[~valid] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return normals.astype(np.float32)


def _sample_depth_frame(
    frame_idx: int,
    depth: np.ndarray,
    conf: np.ndarray,
    K: np.ndarray | None,
    pose: np.ndarray | None,
    world_points: np.ndarray | None,
    rgb: np.ndarray | None,
    args: argparse.Namespace,
    actual_frame_idx: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stride = max(1, int(args.stride))
    height, width = depth.shape[:2]
    ys_grid, xs_grid = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    xs_flat = xs_grid.reshape(-1).astype(np.int32)
    ys_flat = ys_grid.reshape(-1).astype(np.int32)
    depth_flat = depth[ys_flat, xs_flat].astype(np.float32)
    conf_flat = conf[ys_flat, xs_flat].astype(np.float32)
    valid = np.isfinite(depth_flat)
    valid &= np.isfinite(conf_flat)
    valid &= depth_flat >= float(args.min_depth)
    if float(args.max_depth) > 0:
        valid &= depth_flat <= float(args.max_depth)
    valid &= conf_flat >= float(args.min_conf)
    if world_points is not None:
        sampled_world = world_points[ys_flat, xs_flat].astype(np.float32)
        valid &= np.isfinite(sampled_world).all(axis=1)
    valid_idx = np.flatnonzero(valid)

    if args.max_points_per_frame > 0 and valid_idx.shape[0] > args.max_points_per_frame:
        scores = conf_flat[valid_idx]
        keep_local = np.argpartition(scores, -args.max_points_per_frame)[-args.max_points_per_frame :]
        valid_idx = valid_idx[keep_local]

    if valid_idx.size == 0:
        empty = {
            "points_world": np.zeros((0, 3), dtype=np.float32),
            "points_camera": np.zeros((0, 3), dtype=np.float32),
            "point_rgb": np.zeros((0, 3), dtype=np.uint8),
            "point_conf": np.zeros((0,), dtype=np.float32),
            "point_depth": np.zeros((0,), dtype=np.float32),
            "point_uv": np.zeros((0, 2), dtype=np.float32),
            "point_normal": np.zeros((0, 3), dtype=np.float32),
            "point_axis_u": np.zeros((0, 3), dtype=np.float32),
            "point_axis_v": np.zeros((0, 3), dtype=np.float32),
            "point_frame_local": np.zeros((0,), dtype=np.int32),
            "point_frame_idx": np.zeros((0,), dtype=np.int32),
        }
        return empty, {"frame_idx": actual_frame_idx, "count": 0, "valid_ratio": 0.0}

    xs = xs_flat[valid_idx]
    ys = ys_flat[valid_idx]
    point_depth = depth_flat[valid_idx]
    point_conf = conf_flat[valid_idx]
    point_rgb = _sample_rgb(rgb, xs, ys, width=width, height=height)
    point_uv = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    if K is not None and pose is not None:
        points_camera, points_world = _backproject_pixels(K, pose, xs, ys, point_depth)
        xs_u = np.clip(xs + 1, 0, width - 1)
        ys_v = np.clip(ys + 1, 0, height - 1)
        _, points_u = _backproject_pixels(K, pose, xs_u, ys, depth[ys, xs_u].astype(np.float32))
        _, points_v = _backproject_pixels(K, pose, xs, ys_v, depth[ys_v, xs].astype(np.float32))
    elif world_points is not None:
        points_world = world_points[ys, xs].astype(np.float32)
        points_camera = np.zeros_like(points_world, dtype=np.float32)
        points_u = world_points[ys, np.clip(xs + 1, 0, width - 1)].astype(np.float32)
        points_v = world_points[np.clip(ys + 1, 0, height - 1), xs].astype(np.float32)
    else:
        raise ValueError("Need either cuVSLAM pose+intrinsic or LingBot world_points")

    point_normal = _estimate_normals(points_world, points_u, points_v)
    point_axis_u = (points_u - points_world).astype(np.float32)
    point_axis_v = (points_v - points_world).astype(np.float32)
    arrays = {
        "points_world": points_world.astype(np.float32),
        "points_camera": points_camera.astype(np.float32),
        "point_rgb": point_rgb.astype(np.uint8),
        "point_conf": point_conf.astype(np.float32),
        "point_depth": point_depth.astype(np.float32),
        "point_uv": point_uv.astype(np.float32),
        "point_normal": point_normal.astype(np.float32),
        "point_axis_u": point_axis_u.astype(np.float32),
        "point_axis_v": point_axis_v.astype(np.float32),
        "point_frame_local": np.full((valid_idx.size,), frame_idx, dtype=np.int32),
        "point_frame_idx": np.full((valid_idx.size,), actual_frame_idx, dtype=np.int32),
    }
    stats = {
        "frame_idx": int(actual_frame_idx),
        "local_frame": int(frame_idx),
        "count": int(valid_idx.size),
        "valid_ratio": float(valid_idx.size / max(1, xs_flat.size)),
        "mean_depth": float(np.mean(point_depth)),
        "mean_conf": float(np.mean(point_conf)),
    }
    return arrays, stats


def main() -> None:
    args = parse_args()
    total_start = time.perf_counter()
    profile: dict[str, float] = {}
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = np.load(predictions_path)
    profile["load_inputs_sec"] = time.perf_counter() - load_start

    if "depth" not in predictions:
        raise KeyError(f"{predictions_path} does not contain depth")
    depth = _squeeze_depth(predictions["depth"])
    frame_count = int(depth.shape[0])
    depth_conf = (
        _squeeze_depth(predictions["depth_conf"])
        if "depth_conf" in predictions
        else np.ones_like(depth, dtype=np.float32)
    )
    if depth_conf.shape[:3] != depth.shape[:3]:
        raise ValueError(f"depth_conf shape {depth_conf.shape} does not match depth {depth.shape}")

    metadata = summary.get("metadata", {})
    frame_indices = np.asarray(
        metadata.get("frame_indices", list(range(frame_count)))[:frame_count],
        dtype=np.int32,
    )
    timestamps_sec = np.asarray(
        metadata.get("timestamps_sec", [0.0] * frame_count)[:frame_count],
        dtype=np.float64,
    )
    cuvslam_poses = _pose_array(metadata.get("cuvslam_poses", []), frame_count)
    intrinsics = predictions["intrinsic"].astype(np.float32)[:frame_count] if "intrinsic" in predictions else None
    world_points = predictions["world_points"].astype(np.float32)[:frame_count] if "world_points" in predictions else None
    lingbot_extrinsic = _pose_array(predictions["extrinsic"], frame_count) if "extrinsic" in predictions else None
    lingbot_pose_candidates = _pose_candidates(lingbot_extrinsic)

    has_cuvslam = intrinsics is not None and cuvslam_poses.shape[0] >= frame_count
    has_world_points = world_points is not None and world_points.shape[0] >= frame_count
    has_lingbot_depth_pose = (
        intrinsics is not None
        and lingbot_extrinsic is not None
        and lingbot_extrinsic.shape[0] >= frame_count
    )

    requested_geometry_source = str(args.geometry_source)
    if requested_geometry_source == "auto":
        use_cuvslam = has_cuvslam
        use_world_points = (not use_cuvslam) and has_world_points
        use_lingbot_depth = (not use_cuvslam) and (not use_world_points) and has_lingbot_depth_pose
    else:
        use_cuvslam = requested_geometry_source == "cuvslam"
        use_world_points = requested_geometry_source == "world_points"
        use_lingbot_depth = requested_geometry_source == "lingbot_depth"

    missing_reason = ""
    if use_cuvslam and not has_cuvslam:
        missing_reason = "cuvslam requires predictions.intrinsic and summary.metadata.cuvslam_poses"
    if use_world_points and not has_world_points:
        missing_reason = "world_points requires predictions.world_points"
    if use_lingbot_depth and not has_lingbot_depth_pose:
        missing_reason = "lingbot_depth requires predictions.depth, predictions.intrinsic, and predictions.extrinsic"
    if missing_reason:
        raise RuntimeError(f"--geometry-source {requested_geometry_source} unavailable: {missing_reason}")
    if not use_cuvslam and not use_world_points and not use_lingbot_depth:
        raise RuntimeError(
            "Cannot build metric points: need intrinsic+cuvslam_poses, LingBot world_points, "
            "or LingBot intrinsic+extrinsic for official depth unprojection"
        )
    lingbot_depth_poses = (
        np.linalg.inv(lingbot_extrinsic).astype(np.float32)
        if use_lingbot_depth and lingbot_extrinsic is not None
        else None
    )

    requested_depth_scale = _parse_depth_scale_arg(args.depth_scale)
    scale_fit = {"enabled": False, "scale": 1.0, "source": "identity", "valid_pairs": 0}
    pose_alignment = {"enabled": False, "source": "disabled", "valid_pairs": 0}
    aligned_lingbot_poses = None
    if requested_depth_scale is None:
        if use_cuvslam and args.pose_align == "auto":
            pose_alignment = _estimate_pose_alignment(cuvslam_poses, lingbot_pose_candidates)
            if pose_alignment.get("enabled"):
                sim_rotation = np.asarray(pose_alignment["similarity_rotation"], dtype=np.float32)
                sim_translation = np.asarray(pose_alignment["similarity_translation"], dtype=np.float32)
                selected_mode = str(pose_alignment["lingbot_pose_mode"])
                aligned_lingbot_poses = _apply_similarity_to_poses(
                    lingbot_pose_candidates[selected_mode],
                    scale=float(pose_alignment["scale"]),
                    rotation=sim_rotation,
                    translation=sim_translation,
                )
                scale_fit = {
                    "enabled": True,
                    "scale": float(pose_alignment["scale"]),
                    "source": f"pose_alignment:{pose_alignment['source']}",
                    "valid_pairs": int(pose_alignment["valid_pairs"]),
                }
        if not scale_fit.get("enabled") and use_cuvslam:
            scale_fit = _estimate_depth_scale(cuvslam_poses, lingbot_extrinsic)
        depth_scale = float(scale_fit["scale"])
        depth_scale_source = str(scale_fit.get("source", "identity"))
    else:
        depth_scale = float(requested_depth_scale)
        scale_fit = {
            "enabled": True,
            "scale": depth_scale,
            "source": "manual",
            "valid_pairs": 0,
        }
        depth_scale_source = "manual"
    depth = (depth.astype(np.float32) * depth_scale).astype(np.float32)

    image_paths = _resolve_image_paths(summary, summary_path, args.image_root)
    frame_arrays = []
    frame_stats = []
    frame_loop_start = time.perf_counter()
    for local_idx in range(frame_count):
        rgb = _load_rgb(image_paths[local_idx] if local_idx < len(image_paths) else "")
        K = intrinsics[local_idx] if (use_cuvslam or use_lingbot_depth) else None
        if use_cuvslam and aligned_lingbot_poses is not None and local_idx < aligned_lingbot_poses.shape[0]:
            pose = aligned_lingbot_poses[local_idx]
        elif use_lingbot_depth and lingbot_depth_poses is not None and local_idx < lingbot_depth_poses.shape[0]:
            pose = lingbot_depth_poses[local_idx]
        else:
            pose = cuvslam_poses[local_idx] if use_cuvslam else None
        pointmap = world_points[local_idx] if use_world_points else None
        arrays, stats = _sample_depth_frame(
            frame_idx=local_idx,
            depth=depth[local_idx],
            conf=depth_conf[local_idx],
            K=K,
            pose=pose,
            world_points=pointmap,
            rgb=rgb,
            args=args,
            actual_frame_idx=int(frame_indices[local_idx]) if local_idx < frame_indices.size else local_idx,
        )
        frame_arrays.append(arrays)
        stats["image_path"] = image_paths[local_idx] if local_idx < len(image_paths) else ""
        frame_stats.append(stats)
    profile["frame_sampling_sec"] = time.perf_counter() - frame_loop_start

    def concat(key: str) -> np.ndarray:
        values = [item[key] for item in frame_arrays]
        if not values:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(values, axis=0)

    dense_npz = output_dir / args.output_name
    save_payload = {
        "schema_version": np.asarray([1], dtype=np.int32),
        "depth": depth.astype(np.float32),
        "depth_conf": depth_conf.astype(np.float32),
        "depth_scale": np.asarray([depth_scale], dtype=np.float32),
        "frame_indices": frame_indices,
        "timestamps_sec": timestamps_sec,
        "points_world": concat("points_world").astype(np.float32),
        "points_camera": concat("points_camera").astype(np.float32),
        "point_rgb": concat("point_rgb").astype(np.uint8),
        "point_conf": concat("point_conf").astype(np.float32),
        "point_depth": concat("point_depth").astype(np.float32),
        "point_uv": concat("point_uv").astype(np.float32),
        "point_normal": concat("point_normal").astype(np.float32),
        "point_axis_u": concat("point_axis_u").astype(np.float32),
        "point_axis_v": concat("point_axis_v").astype(np.float32),
        "point_frame_local": concat("point_frame_local").astype(np.int32),
        "point_frame_idx": concat("point_frame_idx").astype(np.int32),
    }
    if intrinsics is not None:
        save_payload["intrinsic"] = intrinsics.astype(np.float32)
    if cuvslam_poses.shape[0] > 0:
        save_payload["cuvslam_poses"] = cuvslam_poses.astype(np.float32)
    if aligned_lingbot_poses is not None and aligned_lingbot_poses.shape[0] > 0:
        save_payload["aligned_lingbot_poses"] = aligned_lingbot_poses.astype(np.float32)
    save_start = time.perf_counter()
    np.savez_compressed(dense_npz, **save_payload)
    profile["save_npz_sec"] = time.perf_counter() - save_start

    pose_source = "predictions.world_points"
    coordinate_frame = "lingbot_world_points"
    if use_cuvslam:
        coordinate_frame = "cuvslam_world"
        if aligned_lingbot_poses is not None:
            pose_source = f"predictions.extrinsic[{pose_alignment['lingbot_pose_mode']}] aligned_to_cuvslam"
        else:
            pose_source = "summary.metadata.cuvslam_poses"
    elif use_lingbot_depth:
        coordinate_frame = "lingbot_depth_world"
        pose_source = "inverse(predictions.extrinsic), matching official depth unprojection"

    dense_summary = {
        "schema_version": 1,
        "source": "lingbot_dense_geometry",
        "predictions_npz": str(predictions_path),
        "summary_json": str(summary_path),
        "dense_npz": str(dense_npz),
        "frame_count": frame_count,
        "depth_shape": list(depth.shape),
        "point_count": int(save_payload["points_world"].shape[0]),
        "depth_scale": float(depth_scale),
        "depth_scale_source": depth_scale_source,
        "depth_scale_fit": scale_fit,
        "pose_alignment": pose_alignment,
        "stride": int(args.stride),
        "geometry_source_requested": requested_geometry_source,
        "geometry_source_resolved": (
            "cuvslam" if use_cuvslam else "lingbot_depth" if use_lingbot_depth else "world_points"
        ),
        "min_conf": float(args.min_conf),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "max_points_per_frame": int(args.max_points_per_frame),
        "coordinate_frame": coordinate_frame,
        "pose_source": pose_source,
        "frame_stats": frame_stats,
        "npz_keys": sorted(save_payload.keys()),
        "profile_sec": profile,
    }
    dense_summary["profile_sec"]["total_sec"] = time.perf_counter() - total_start
    dense_summary_path = output_dir / "lingbot_dense_geometry_summary.json"
    dense_summary_path.write_text(
        json.dumps(dense_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(dense_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
