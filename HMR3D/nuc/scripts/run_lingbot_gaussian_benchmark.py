from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import GaussianSplatRenderer, load_runtime_config, psnr, save_render_triplet, ssim_rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LingBot-derived Gaussian handles against ground-truth images."
    )
    parser.add_argument("--gaussians-npz", required=True, help="Gaussian handle npz exported from LingBot predictions.")
    parser.add_argument("--summary-json", required=True, help="LingBot summary json with image paths and metadata.")
    parser.add_argument("--config", default="", help="Optional runtime config YAML.")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory.")
    parser.add_argument(
        "--pose-mode",
        default="auto",
        choices=["auto", "lingbot_direct", "lingbot_inverse", "cuvslam"],
        help="Pose source to use for rendering. auto benchmarks all variants and keeps the best by mean PSNR.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save GT/render/triplet images for every evaluated frame.",
    )
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


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = GaussianSplatRenderer(config.output)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    predictions = np.load(args.gaussians_npz)
    bundle = {key: predictions[key] for key in predictions.files}
    image_paths = [str(Path(path).expanduser().resolve()) for path in summary["image_paths"]]
    frame_indices = list(summary.get("metadata", {}).get("frame_indices", list(range(len(image_paths)))))

    raw_predictions = np.load(Path(args.summary_json).with_name("lingbot_predictions.npz"))
    variants = _candidate_poses(summary, raw_predictions, args.pose_mode)
    if not variants:
        raise SystemExit("No pose candidates available for the requested pose mode.")

    variant_results: dict[str, dict] = {}
    best_name = None
    best_score = -1.0
    best_frames: list[dict] = []
    best_summary: dict | None = None
    for name, poses in variants.items():
        variant_dir = output_dir / f"variant_{name}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_summary, per_frame = _evaluate_variant(
            renderer=renderer,
            bundle=bundle,
            image_paths=image_paths,
            frame_indices=frame_indices,
            poses=poses,
            output_dir=variant_dir,
            save_images=args.save_images or True,
        )
        variant_results[name] = variant_summary["render_metrics"]
        score = float(variant_summary["render_metrics"]["mean_psnr"])
        if score > best_score:
            best_score = score
            best_name = name
            best_frames = per_frame
            best_summary = variant_summary

    assert best_name is not None and best_summary is not None
    chosen_variant_dir = output_dir / f"variant_{best_name}" / "renders"
    final_renders = output_dir / "renders"
    final_renders.mkdir(parents=True, exist_ok=True)
    for item in best_frames:
        for key in ("gt_path", "render_path", "triplet_path"):
            src = item.get(key)
            if not src:
                continue
            src_path = Path(src)
            dst_path = final_renders / src_path.name
            if src_path.exists() and src_path.resolve() != dst_path.resolve():
                dst_path.write_bytes(src_path.read_bytes())
            item[key] = str(dst_path)

    benchmark = {
        "config": config.to_dict(),
        "lingbot_summary_path": str(Path(args.summary_json).expanduser().resolve()),
        "gaussians_npz": str(Path(args.gaussians_npz).expanduser().resolve()),
        "image_paths": image_paths,
        "frame_indices": frame_indices,
        "pose_mode_requested": args.pose_mode,
        "pose_variant_results": variant_results,
        "selected_pose_variant": best_name,
        **best_summary,
    }
    (output_dir / "render_benchmark_summary.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "render_benchmark_frames.json").write_text(
        json.dumps(best_frames, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "selected_pose_variant": best_name, **best_summary["render_metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
