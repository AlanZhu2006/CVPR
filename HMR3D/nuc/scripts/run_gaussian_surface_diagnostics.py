from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import GaussianSplatRenderer, load_runtime_config, psnr, save_render_triplet, ssim_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run surface-focused diagnostics on a refined Gaussian bundle."
    )
    parser.add_argument("--gaussians-npz", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--surface-prior-npz", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--pose-mode", default="cuvslam", choices=["auto", "lingbot_direct", "lingbot_inverse", "cuvslam"])
    parser.add_argument("--match-radius", type=float, default=0.35)
    parser.add_argument("--local-radius", type=float, default=1.0)
    parser.add_argument("--isotropic-scale", type=float, default=0.08)
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def _to_pose(matrix_3x4: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = matrix_3x4.astype(np.float32)
    return pose


def _candidate_poses(summary: dict, predictions: dict[str, np.ndarray], mode: str) -> dict[str, list[np.ndarray]]:
    metadata = summary.get("metadata", {})
    candidates: dict[str, list[np.ndarray]] = {}
    if mode in {"auto", "lingbot_direct"} and "extrinsic" in predictions:
        candidates["lingbot_direct"] = [_to_pose(item) for item in predictions["extrinsic"]]
    if mode in {"auto", "lingbot_inverse"} and "extrinsic" in predictions:
        candidates["lingbot_inverse"] = [np.linalg.inv(_to_pose(item)).astype(np.float32) for item in predictions["extrinsic"]]
    if mode in {"auto", "cuvslam"} and metadata.get("cuvslam_poses"):
        candidates["cuvslam"] = [np.asarray(item, dtype=np.float32) for item in metadata["cuvslam_poses"]]
    return candidates


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


def _resolve_image_paths(summary: dict, summary_json: Path, image_root_arg: str) -> list[str]:
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


def _evaluate_variant(
    renderer: GaussianSplatRenderer,
    bundle: dict[str, np.ndarray],
    image_paths: list[str],
    frame_indices: list[int],
    poses: list[np.ndarray],
    output_dir: Path,
    save_images: bool,
) -> tuple[dict, list[dict]]:
    image_dir = output_dir / "renders"
    per_frame: list[dict] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    total_render_time = 0.0
    measured_frames = 0

    for local_idx, (frame_idx, image_path, pose) in enumerate(zip(frame_indices, image_paths, poses)):
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        target_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        render_start = time.perf_counter()
        render = renderer.render_bundle_view(
            bundle=bundle,
            pose=pose,
            image_shape=target_rgb.shape[:2],
            image_path=image_path,
        )
        render_dt = time.perf_counter() - render_start
        total_render_time += render_dt
        measured_frames += 1

        frame_psnr = psnr(render.image_rgb, target_rgb)
        frame_ssim = ssim_rgb(render.image_rgb, target_rgb)
        psnr_values.append(frame_psnr)
        ssim_values.append(frame_ssim)

        saved = {}
        if save_images:
            saved = save_render_triplet(image_dir, frame_idx, target_rgb, render.image_rgb)

        per_frame.append(
            {
                "frame_idx": frame_idx,
                "local_idx": local_idx,
                "image_path": image_path,
                "render_ms": round(render_dt * 1000.0, 3),
                "psnr": round(frame_psnr, 4),
                "ssim": round(frame_ssim, 5),
                "point_count": render.point_count,
                "projected_points": render.projected_points,
                "active_points": render.active_points,
                "archived_points": render.archived_points,
                "warmstart_points": render.warmstart_points,
                **saved,
            }
        )

    summary = {
        "measured_frames": measured_frames,
        "render_metrics": {
            "mean_psnr": round(float(np.mean(psnr_values)) if psnr_values else 0.0, 4),
            "mean_ssim": round(float(np.mean(ssim_values)) if ssim_values else 0.0, 5),
            "mean_update_ms": 0.0,
            "mean_render_ms": round(1000.0 * total_render_time / max(1, measured_frames), 3),
            "approx_fps": round(measured_frames / max(1e-8, total_render_time), 3),
            "max_point_count": max((item["point_count"] for item in per_frame), default=0),
            "max_projected_points": max((item["projected_points"] for item in per_frame), default=0),
        },
    }
    return summary, per_frame


def _slice_bundle(bundle: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = mask.astype(bool)
    out: dict[str, np.ndarray] = {}
    count = int(mask.shape[0])
    for key, value in bundle.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.shape[0] == count:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _normalize_vecs(vecs: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norm, 1e-6, None)


def _isotropize_subset(bundle: dict[str, np.ndarray], mask: np.ndarray, isotropic_scale: float) -> dict[str, np.ndarray]:
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in bundle.items()}
    if out["xyz"].shape[0] == 0:
        return out
    mask = mask.astype(bool)
    scale = np.full((int(mask.sum()),), float(isotropic_scale), dtype=np.float32)
    u_dir = _normalize_vecs(out["axis_u"][mask].astype(np.float32))
    v_dir = _normalize_vecs(out["axis_v"][mask].astype(np.float32))
    out["axis_u"][mask] = u_dir * scale[:, None]
    out["axis_v"][mask] = v_dir * scale[:, None]
    out["scale"][mask] = scale
    if "scale_xyz" in out:
        out["scale_xyz"][mask, 0] = scale
        out["scale_xyz"][mask, 1] = scale
        out["scale_xyz"][mask, 2] = np.minimum(out["scale_xyz"][mask, 2], 0.5 * scale)
    return out


def _nearest_surface_stats(bundle: dict[str, np.ndarray], surface_bundle: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    xyz = torch.from_numpy(bundle["xyz"].astype(np.float32))
    surface_xyz = torch.from_numpy(surface_bundle["xyz"].astype(np.float32))
    dist = torch.cdist(xyz, surface_xyz)
    nearest_dist, nearest_idx = torch.min(dist, dim=1)
    return nearest_dist.cpu().numpy().astype(np.float32), nearest_idx.cpu().numpy().astype(np.int64)


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles_dir = output_dir / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    renderer = GaussianSplatRenderer(config.output)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    bundle_npz = np.load(args.gaussians_npz)
    surface_npz = np.load(args.surface_prior_npz)
    bundle = {key: bundle_npz[key] for key in bundle_npz.files}
    surface_bundle = {key: surface_npz[key] for key in surface_npz.files}

    if int(bundle["xyz"].shape[0]) == 0:
        raise SystemExit("Empty gaussian bundle.")
    if int(surface_bundle["xyz"].shape[0]) == 0:
        raise SystemExit("Empty surface prior bundle.")

    nearest_dist, _ = _nearest_surface_stats(bundle, surface_bundle)
    matched_mask = nearest_dist <= float(args.match_radius)
    local_mask = nearest_dist <= float(args.local_radius)
    unmatched_mask = ~matched_mask

    variants: dict[str, dict[str, np.ndarray]] = {
        "all": bundle,
        "prior_only": surface_bundle,
        "matched_only": _slice_bundle(bundle, matched_mask),
        "unmatched_only": _slice_bundle(bundle, unmatched_mask),
        "isotropic_unmatched": _isotropize_subset(bundle, unmatched_mask, float(args.isotropic_scale)),
        "local_window_only": _slice_bundle(bundle, local_mask),
    }

    diagnostics = {
        "match_radius": float(args.match_radius),
        "local_radius": float(args.local_radius),
        "isotropic_scale": float(args.isotropic_scale),
        "base_point_count": int(bundle["xyz"].shape[0]),
        "surface_point_count": int(surface_bundle["xyz"].shape[0]),
        "matched_count": int(matched_mask.sum()),
        "unmatched_count": int(unmatched_mask.sum()),
        "local_count": int(local_mask.sum()),
        "match_ratio": float(np.mean(matched_mask)),
        "local_ratio": float(np.mean(local_mask)),
        "nearest_surface_distance": {
            "p50": round(float(np.median(nearest_dist)), 4),
            "p95": round(float(np.quantile(nearest_dist, 0.95)), 4),
            "max": round(float(np.max(nearest_dist)), 4),
        },
        "local_window_definition": "gaussians within local_radius of surface_prior centers; the current bundle is already single-window scoped.",
    }

    for name, variant in variants.items():
        np.savez_compressed(bundles_dir / f"{name}.npz", **variant)

    summary_json = Path(args.summary_json).expanduser().resolve()
    image_paths = _resolve_image_paths(summary, summary_json, args.image_root)
    frame_indices = list(summary.get("metadata", {}).get("frame_indices", list(range(len(image_paths)))))
    raw_predictions = np.load(summary_json.with_name("lingbot_predictions.npz"))
    pose_variants = _candidate_poses(summary, raw_predictions, args.pose_mode)
    if not pose_variants:
        raise SystemExit("No pose candidates available for the requested pose mode.")

    variant_summaries: dict[str, dict] = {}
    for name, variant in variants.items():
        best_name = None
        best_score = -1.0
        best_summary = None
        best_frames = None
        for pose_name, poses in pose_variants.items():
            variant_dir = output_dir / name / f"variant_{pose_name}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            summary_metrics, per_frame = _evaluate_variant(
                renderer=renderer,
                bundle=variant,
                image_paths=image_paths,
                frame_indices=frame_indices,
                poses=poses,
                output_dir=variant_dir,
                save_images=bool(args.save_images),
            )
            score = float(summary_metrics["render_metrics"]["mean_psnr"])
            if score > best_score:
                best_score = score
                best_name = pose_name
                best_summary = summary_metrics
                best_frames = per_frame
        assert best_name is not None and best_summary is not None and best_frames is not None
        variant_payload = {
            "selected_pose_variant": best_name,
            **best_summary,
            "point_count": int(variant["xyz"].shape[0]),
        }
        (output_dir / name / "render_benchmark_summary.json").write_text(
            json.dumps(variant_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / name / "render_benchmark_frames.json").write_text(
            json.dumps(best_frames, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        variant_summaries[name] = variant_payload

    final_summary = {
        "config": config.to_dict(),
        "gaussians_npz": str(Path(args.gaussians_npz).expanduser().resolve()),
        "surface_prior_npz": str(Path(args.surface_prior_npz).expanduser().resolve()),
        "summary_json": str(summary_json),
        "image_paths": image_paths,
        "diagnostics": diagnostics,
        "variants": variant_summaries,
    }
    (output_dir / "surface_diagnostics_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **diagnostics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
