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

from nuc_runtime import (
    GaussianSplatRenderer,
    IncrementalGaussianBuilder,
    load_runtime_config,
    psnr,
    save_render_triplet,
    ssim_rgb,
)
from nuc_runtime.models import KeyframeRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LingBot-aligned Gaussian initialization with v11/v12-style fast refine."
    )
    parser.add_argument("--gaussians-npz", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _build_keyframes(summary: dict) -> list[KeyframeRecord]:
    metadata = summary.get("metadata", {})
    frame_indices = metadata.get("frame_indices", [])
    timestamps = metadata.get("timestamps_sec", [])
    poses = [np.asarray(item, dtype=np.float32) for item in metadata.get("cuvslam_poses", [])]
    descriptors = [np.asarray(item, dtype=np.float32) for item in metadata.get("cuvslam_descriptors", [])]
    image_paths = [str(Path(path).expanduser().resolve()) for path in summary.get("image_paths", [])]
    keyframes: list[KeyframeRecord] = []
    for idx, image_path, ts, pose, desc in zip(frame_indices, image_paths, timestamps, poses, descriptors):
        right_path = str(Path(image_path).parent.parent / "image_1" / Path(image_path).name)
        keyframes.append(
            KeyframeRecord(
                frame_idx=int(idx),
                timestamp_sec=float(ts),
                pose=pose,
                descriptor=desc,
                orb_descriptors=None,
                keypoints_xy=None,
                image_path=image_path,
                right_image_path=right_path,
                keypoint_count=0,
                match_count=0,
                inlier_count=0,
                pixel_motion=0.0,
            )
        )
    return keyframes


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "renders"
    image_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
    keyframes = _build_keyframes(summary)
    if not keyframes:
        raise SystemExit("No keyframes available in LingBot summary metadata.")

    data = np.load(args.gaussians_npz)
    bundle = {key: data[key] for key in data.files}
    count = int(bundle["xyz"].shape[0])
    bundle["source"] = np.full((count,), 4, dtype=np.int8)
    bundle["recentness"] = np.full((count,), 1.0, dtype=np.float32)
    bundle["unstable"] = np.clip(bundle.get("unstable", np.full((count,), 0.45, dtype=np.float32)).astype(np.float32), 0.15, 1.2)

    builder = IncrementalGaussianBuilder(output_dir=output_dir, config=config.memory)
    state = builder.start_submap(submap_id=13001)
    state.replace_points(
        bundle["xyz"],
        bundle["rgb"],
        bundle["scale"],
        bundle["opacity"],
        bundle["axis_u"],
        bundle["axis_v"],
        bundle["unstable"],
        bundle["recentness"],
        bundle["source"],
    )
    state.keyframe_count = len(keyframes)
    state.last_keyframe = keyframes[-1]
    state.recover_refine_frames_left = int(config.memory.gaussian_optimize_recover_refine_frames)

    renderer = GaussianSplatRenderer(config.output)
    per_frame: list[dict] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    total_update = 0.0
    total_render = 0.0

    window: list[KeyframeRecord] = []
    for local_idx, keyframe in enumerate(keyframes):
        window.append(keyframe)
        update_start = time.perf_counter()
        optimize_summary = builder.optimize_active_window(state.submap_id, window)
        update_dt = time.perf_counter() - update_start
        total_update += update_dt

        target_bgr = cv2.imread(keyframe.image_path, cv2.IMREAD_COLOR)
        if target_bgr is None:
            continue
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        render_start = time.perf_counter()
        render = renderer.render_bundle_view(
            builder.active_points(state.submap_id),
            pose=keyframe.pose,
            image_shape=target_rgb.shape[:2],
            image_path=keyframe.image_path,
        )
        render_dt = time.perf_counter() - render_start
        total_render += render_dt
        frame_psnr = psnr(render.image_rgb, target_rgb)
        frame_ssim = ssim_rgb(render.image_rgb, target_rgb)
        psnr_values.append(frame_psnr)
        ssim_values.append(frame_ssim)
        saved = save_render_triplet(image_dir, keyframe.frame_idx, target_rgb, render.image_rgb)
        per_frame.append(
            {
                "frame_idx": keyframe.frame_idx,
                "local_idx": local_idx,
                "image_path": keyframe.image_path,
                "update_ms": round(update_dt * 1000.0, 3),
                "render_ms": round(render_dt * 1000.0, 3),
                "psnr": round(frame_psnr, 4),
                "ssim": round(frame_ssim, 5),
                "point_count": render.point_count,
                "projected_points": render.projected_points,
                "active_points": render.active_points,
                "archived_points": render.archived_points,
                "warmstart_points": render.warmstart_points,
                "optimize_summary": optimize_summary,
                **saved,
            }
        )

    benchmark = {
        "config": config.to_dict(),
        "summary_json": str(Path(args.summary_json).expanduser().resolve()),
        "gaussians_npz": str(Path(args.gaussians_npz).expanduser().resolve()),
        "measured_frames": len(per_frame),
        "render_metrics": {
            "mean_psnr": round(float(np.mean(psnr_values)) if psnr_values else 0.0, 4),
            "mean_ssim": round(float(np.mean(ssim_values)) if ssim_values else 0.0, 5),
            "mean_update_ms": round(1000.0 * total_update / max(1, len(per_frame)), 3),
            "mean_render_ms": round(1000.0 * total_render / max(1, len(per_frame)), 3),
            "approx_fps": round(len(per_frame) / max(1e-8, total_update + total_render), 3),
            "max_point_count": max((item["point_count"] for item in per_frame), default=0),
            "max_projected_points": max((item["projected_points"] for item in per_frame), default=0),
        },
    }
    (output_dir / "render_benchmark_summary.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "render_benchmark_frames.json").write_text(
        json.dumps(per_frame, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **benchmark["render_metrics"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
