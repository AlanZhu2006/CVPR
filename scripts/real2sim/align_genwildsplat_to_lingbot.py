#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import itertools
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align GenWildSplat Gaussian coordinates to LingBot world.")
    parser.add_argument("--genwild-manifest", type=Path, required=True)
    parser.add_argument("--genwild-camera-poses", type=Path, required=True)
    parser.add_argument("--genwild-ply", type=Path, required=True)
    parser.add_argument("--lingbot-predictions-npz", type=Path, required=True)
    parser.add_argument("--lingbot-summary-json", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--output-alignment-json", type=Path, required=True)
    parser.add_argument("--geometry-source", default="depth")
    parser.add_argument("--conf-percentile", type=float, default=65.0)
    parser.add_argument("--points-per-frame", type=int, default=8000)
    parser.add_argument("--axis-weight", type=float, default=0.35)
    parser.add_argument("--inlier-threshold", type=float, default=0.8)
    parser.add_argument("--icp-refine", action="store_true")
    parser.add_argument("--icp-max-frame", type=int, default=100)
    parser.add_argument("--icp-frame-step", type=int, default=3)
    parser.add_argument("--icp-source-samples", type=int, default=20000)
    parser.add_argument("--icp-target-samples", type=int, default=30000)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-percentile", type=float, default=60.0)
    return parser.parse_args()


def load_lingbot_timeline(args: argparse.Namespace) -> Any:
    script_path = Path(__file__).with_name("run_video_real2sim_playback_webui.py")
    spec = importlib.util.spec_from_file_location("lingbot_playback_webui", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LingBotPointTimeline(
        predictions_npz=args.lingbot_predictions_npz,
        summary_json=args.lingbot_summary_json,
        points_per_frame=args.points_per_frame,
        geometry_source=args.geometry_source,
        conf_percentile=args.conf_percentile,
        normalize_world=True,
        precompute_clouds=False,
    )


def normalized_lingbot_c2w(timeline: Any, frame_index: int) -> np.ndarray:
    c2w = timeline.camera_c2w(frame_index).astype(np.float64)
    c2w[:3, 3] = (c2w[:3, 3] - timeline.center.astype(np.float64)) * float(timeline.scale)
    return c2w


def load_genwild_c2w(camera_poses_path: Path) -> np.ndarray:
    payload = json.loads(camera_poses_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    poses = []
    for entry in entries:
        pose = np.asarray(entry["extrinsic_c2w"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise RuntimeError(f"Expected 4x4 GenWild pose, got {pose.shape}")
        poses.append(pose)
    if not poses:
        raise RuntimeError(f"No poses found in {camera_poses_path}")
    return np.stack(poses, axis=0)


def augmented_pose_points(poses: np.ndarray, axis_weight: float) -> np.ndarray:
    centers = poses[:, :3, 3]
    span = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
    axis_length = max(float(span) * float(axis_weight), 0.05)
    parts = [centers]
    for axis in range(3):
        parts.append(centers + poses[:, :3, axis] * axis_length)
    return np.concatenate(parts, axis=0)


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected Nx3 source/target, got {source.shape} and {target.shape}")
    n = source.shape[0]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / float(n)
    u, singular_values, vh = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vh) < 0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vh
    source_variance = np.sum(source_centered * source_centered) / float(n)
    scale = float(np.sum(singular_values * sign) / max(source_variance, 1e-12))
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def fit_pose_alignment(
    source_poses: np.ndarray,
    target_poses: np.ndarray,
    axis_weight: float,
    inlier_threshold: float,
) -> tuple[float, np.ndarray, np.ndarray, list[int], np.ndarray]:
    count = source_poses.shape[0]
    source_centers = source_poses[:, :3, 3]
    target_centers = target_poses[:, :3, 3]
    best: tuple[int, float, tuple[int, ...], float, np.ndarray, np.ndarray, np.ndarray] | None = None

    min_subset = min(3, count)
    candidate_subsets: list[tuple[int, ...]] = []
    if count <= 12:
        for subset_size in range(min_subset, count + 1):
            candidate_subsets.extend(tuple(int(index) for index in subset) for subset in itertools.combinations(range(count), subset_size))
    else:
        candidate_subsets.append(tuple(range(count)))
        candidate_subsets.extend(tuple(range(start, start + min_subset)) for start in range(0, count - min_subset + 1))
        thirds = np.linspace(0, count - 1, min_subset).round().astype(int)
        candidate_subsets.append(tuple(int(index) for index in thirds))
        rng = np.random.default_rng(0)
        seen = set(candidate_subsets)
        for _ in range(384):
            subset = tuple(sorted(int(index) for index in rng.choice(count, size=min_subset, replace=False)))
            if subset not in seen:
                seen.add(subset)
                candidate_subsets.append(subset)

    for subset in candidate_subsets:
        source_points = augmented_pose_points(source_poses[list(subset)], axis_weight)
        target_points = augmented_pose_points(target_poses[list(subset)], axis_weight)
        scale, rotation, translation = umeyama(source_points, target_points)
        aligned_centers = scale * (source_centers @ rotation.T) + translation[None, :]
        errors = np.linalg.norm(aligned_centers - target_centers, axis=1)
        inliers = np.flatnonzero(errors <= inlier_threshold)
        if inliers.size < min_subset:
            continue
        inlier_mean = float(errors[inliers].mean())
        score = (int(inliers.size), -inlier_mean, len(subset))
        if best is None or score > (best[0], -best[1], len(best[2])):
            best = (int(inliers.size), inlier_mean, tuple(inliers.tolist()), scale, rotation, translation, errors)

    if best is None:
        source_points = augmented_pose_points(source_poses, axis_weight)
        target_points = augmented_pose_points(target_poses, axis_weight)
        scale, rotation, translation = umeyama(source_points, target_points)
        errors = np.linalg.norm(scale * (source_centers @ rotation.T) + translation[None, :] - target_centers, axis=1)
        return scale, rotation, translation, list(range(count)), errors

    inliers = list(best[2])
    source_points = augmented_pose_points(source_poses[inliers], axis_weight)
    target_points = augmented_pose_points(target_poses[inliers], axis_weight)
    scale, rotation, translation = umeyama(source_points, target_points)
    errors = np.linalg.norm(scale * (source_centers @ rotation.T) + translation[None, :] - target_centers, axis=1)
    return scale, rotation, translation, inliers, errors


def transform_ply(input_ply: Path, output_ply: Path, scale: float, rotation: np.ndarray, translation: np.ndarray) -> int:
    ply = PlyData.read(input_ply)
    vertex = ply["vertex"]
    data = vertex.data.copy()
    xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    xyz_out = scale * (xyz @ rotation.T) + translation[None, :]
    data["x"] = xyz_out[:, 0].astype(np.float32)
    data["y"] = xyz_out[:, 1].astype(np.float32)
    data["z"] = xyz_out[:, 2].astype(np.float32)

    if all(name in data.dtype.names for name in ("nx", "ny", "nz")):
        normals = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(np.float64)
        normals_out = normals @ rotation.T
        data["nx"] = normals_out[:, 0].astype(np.float32)
        data["ny"] = normals_out[:, 1].astype(np.float32)
        data["nz"] = normals_out[:, 2].astype(np.float32)

    if all(name in data.dtype.names for name in ("rot_0", "rot_1", "rot_2", "rot_3")):
        q_wxyz = np.column_stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]]).astype(np.float64)
        q_xyzw = np.column_stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]])
        aligned_rot = Rotation.from_matrix(rotation)
        q_out_xyzw = (aligned_rot * Rotation.from_quat(q_xyzw)).as_quat()
        data["rot_0"] = q_out_xyzw[:, 3].astype(np.float32)
        data["rot_1"] = q_out_xyzw[:, 0].astype(np.float32)
        data["rot_2"] = q_out_xyzw[:, 1].astype(np.float32)
        data["rot_3"] = q_out_xyzw[:, 2].astype(np.float32)

    log_scale_delta = float(np.log(max(scale, 1e-12)))
    for name in ("scale_0", "scale_1", "scale_2"):
        if name in data.dtype.names:
            data[name] = (data[name].astype(np.float64) + log_scale_delta).astype(np.float32)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False, byte_order="<").write(output_ply)
    return int(data.shape[0])


def load_ply_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"]
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)


def deterministic_sample(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] <= count:
        return points
    indices = np.linspace(0, points.shape[0] - 1, int(count)).astype(np.int64)
    return points[indices]


def load_lingbot_target_points(timeline: Any, max_frame: int, frame_step: int, target_samples: int) -> np.ndarray:
    parts = []
    stop = min(int(max_frame), int(timeline.frame_count))
    for frame_index in range(0, max(1, stop), max(1, int(frame_step))):
        xyz, _rgb = timeline.frame_cloud(frame_index)
        if xyz.size:
            parts.append(xyz.astype(np.float64))
    if not parts:
        raise RuntimeError("No LingBot target points available for ICP refinement.")
    points = np.concatenate(parts, axis=0)
    finite = np.isfinite(points).all(axis=1)
    return deterministic_sample(points[finite], target_samples)


def geometry_initial_similarity(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    source_radius = np.percentile(np.linalg.norm(source - source_center[None, :], axis=1), 90)
    target_radius = np.percentile(np.linalg.norm(target - target_center[None, :], axis=1), 90)
    scale = float(target_radius / max(source_radius, 1e-12))
    rotation = np.eye(3, dtype=np.float64)
    translation = target_center - scale * (rotation @ source_center)
    return scale, rotation, translation


def refine_similarity_icp(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int,
    trim_percentile: float,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    scale, rotation, translation = geometry_initial_similarity(source, target)
    tree = cKDTree(target)
    stats: dict[str, float] = {}
    for _iteration in range(max(1, int(iterations))):
        transformed = scale * (source @ rotation.T) + translation[None, :]
        distances, indices = tree.query(transformed, k=1, workers=-1)
        threshold = np.percentile(distances, np.clip(float(trim_percentile), 5.0, 95.0))
        keep = np.isfinite(distances) & (distances <= threshold)
        if keep.sum() < 32:
            break
        scale, rotation, translation = umeyama(source[keep], target[indices[keep]])
        stats = {
            "trimmed_mean_distance": float(distances[keep].mean()),
            "trimmed_max_distance": float(distances[keep].max()),
            "all_p90_distance": float(np.percentile(distances, 90)),
            "kept_fraction": float(keep.mean()),
        }
    return scale, rotation, translation, stats


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.genwild_manifest.read_text(encoding="utf-8"))
    keyframe_indices = [int(index) for index in manifest["keyframe_indices"]]
    genwild_poses = load_genwild_c2w(args.genwild_camera_poses)
    if len(keyframe_indices) != genwild_poses.shape[0]:
        raise RuntimeError(f"{len(keyframe_indices)} LingBot keyframes but {genwild_poses.shape[0]} GenWild poses")

    timeline = load_lingbot_timeline(args)
    lingbot_poses = np.stack([normalized_lingbot_c2w(timeline, index) for index in keyframe_indices], axis=0)
    scale, rotation, translation, inliers, center_error = fit_pose_alignment(
        genwild_poses,
        lingbot_poses,
        axis_weight=args.axis_weight,
        inlier_threshold=args.inlier_threshold,
    )
    icp_stats: dict[str, float] | None = None
    if args.icp_refine:
        source_sample = deterministic_sample(load_ply_xyz(args.genwild_ply), args.icp_source_samples)
        target_sample = load_lingbot_target_points(
            timeline,
            max_frame=args.icp_max_frame,
            frame_step=args.icp_frame_step,
            target_samples=args.icp_target_samples,
        )
        scale, rotation, translation, icp_stats = refine_similarity_icp(
            source_sample,
            target_sample,
            iterations=args.icp_iterations,
            trim_percentile=args.icp_trim_percentile,
        )

    aligned_count = transform_ply(args.genwild_ply, args.output_ply, scale, rotation, translation)

    source_centers = genwild_poses[:, :3, 3]
    target_centers = lingbot_poses[:, :3, 3]
    aligned_centers = scale * (source_centers @ rotation.T) + translation[None, :]
    output = {
        "schema": "genwildsplat_lingbot_alignment.v1",
        "source": "genwildsplat",
        "target": "lingbot_normalized_world",
        "keyframe_indices": keyframe_indices,
        "inlier_context_orders": inliers,
        "outlier_context_orders": [index for index in range(len(keyframe_indices)) if index not in set(inliers)],
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "source_to_target_4x4_row_major": [
            [float(scale * rotation[0, 0]), float(scale * rotation[0, 1]), float(scale * rotation[0, 2]), float(translation[0])],
            [float(scale * rotation[1, 0]), float(scale * rotation[1, 1]), float(scale * rotation[1, 2]), float(translation[1])],
            [float(scale * rotation[2, 0]), float(scale * rotation[2, 1]), float(scale * rotation[2, 2]), float(translation[2])],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "center_error_mean": float(center_error.mean()),
        "center_error_max": float(center_error.max()),
        "center_error_by_context_order": center_error.astype(float).tolist(),
        "inlier_center_error_mean": float(center_error[inliers].mean()) if inliers else float(center_error.mean()),
        "inlier_center_error_max": float(center_error[inliers].max()) if inliers else float(center_error.max()),
        "icp_refine": bool(args.icp_refine),
        "icp_stats": icp_stats,
        "aligned_vertex_count": aligned_count,
        "output_ply": str(args.output_ply),
        "lingbot_normalization": {
            "center": np.asarray(timeline.center, dtype=float).tolist(),
            "scale": float(timeline.scale),
        },
    }
    args.output_alignment_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_alignment_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
