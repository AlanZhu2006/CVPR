from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
    return parser.parse_args()


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (F,H,W) or (F,H,W,1), got {depth.shape}")
    return depth


def _pose_array(raw_poses: Any, frame_count: int) -> np.ndarray:
    if not raw_poses:
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
    predictions_path = Path(args.predictions_npz).expanduser().resolve()
    summary_path = Path(args.summary_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = np.load(predictions_path)

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

    use_cuvslam = intrinsics is not None and cuvslam_poses.shape[0] >= frame_count
    use_world_points = not use_cuvslam and world_points is not None
    if not use_cuvslam and not use_world_points:
        raise RuntimeError(
            "Cannot build metric points: need intrinsic+cuvslam_poses or LingBot world_points"
        )

    image_paths = _resolve_image_paths(summary, summary_path, args.image_root)
    frame_arrays = []
    frame_stats = []
    for local_idx in range(frame_count):
        rgb = _load_rgb(image_paths[local_idx] if local_idx < len(image_paths) else "")
        K = intrinsics[local_idx] if use_cuvslam else None
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
    np.savez_compressed(dense_npz, **save_payload)

    dense_summary = {
        "schema_version": 1,
        "source": "lingbot_dense_geometry",
        "predictions_npz": str(predictions_path),
        "summary_json": str(summary_path),
        "dense_npz": str(dense_npz),
        "frame_count": frame_count,
        "depth_shape": list(depth.shape),
        "point_count": int(save_payload["points_world"].shape[0]),
        "stride": int(args.stride),
        "min_conf": float(args.min_conf),
        "min_depth": float(args.min_depth),
        "max_depth": float(args.max_depth),
        "max_points_per_frame": int(args.max_points_per_frame),
        "coordinate_frame": "cuvslam_world" if use_cuvslam else "lingbot_world_points",
        "pose_source": "summary.metadata.cuvslam_poses" if use_cuvslam else "predictions.world_points",
        "frame_stats": frame_stats,
        "npz_keys": sorted(save_payload.keys()),
    }
    dense_summary_path = output_dir / "lingbot_dense_geometry_summary.json"
    dense_summary_path.write_text(
        json.dumps(dense_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(dense_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
