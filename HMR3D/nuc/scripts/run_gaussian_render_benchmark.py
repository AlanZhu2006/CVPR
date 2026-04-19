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
    CUVSLAMOfflineKITTIAdapter,
    GaussianSplatRenderer,
    MemoryRouter,
    psnr,
    save_render_triplet,
    ssim_rgb,
)
from nuc_runtime.config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay KITTI and evaluate a minimal gsplat-style Gaussian renderer."
    )
    parser.add_argument("--sequence-path", required=True, help="KITTI sequence directory.")
    parser.add_argument("--trajectory-path", required=True, help="trajectory_tum.txt produced by cuVSLAM.")
    parser.add_argument("--config", default="", help="Optional runtime YAML config.")
    parser.add_argument("--frame-step", type=int, default=0, help="Replay every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit replay length.")
    parser.add_argument("--output-dir", default="", help="Optional output directory.")
    parser.add_argument("--save-images", action="store_true", help="Save render/gt/diff image triplets.")
    parser.add_argument(
        "--compare-stride",
        type=int,
        default=0,
        help="Save one image triplet every N frames. Defaults to config output.render_compare_stride.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_runtime_config(args.config or None)
    config.memory.enable_v2_write_policy = True
    config.memory.enable_pose_anchor_gate = True
    config.memory.enable_shadow_recover = True
    config.memory.enable_hierarchical_bank = True
    config.memory.enable_multi_candidate_merge = True
    config.memory.enable_local_adapt = True
    config.memory.enable_incremental_gaussian = True

    output_dir = Path(args.output_dir or config.output.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "renders"
    frame_step = args.frame_step or config.input.frame_step
    max_frames = args.max_frames or config.input.max_frames
    compare_stride = args.compare_stride or config.output.render_compare_stride
    save_images = args.save_images or config.output.render_save_images

    adapter = CUVSLAMOfflineKITTIAdapter(
        sequence_path=args.sequence_path,
        trajectory_path=args.trajectory_path,
        config=config.tracking,
        frame_step=frame_step,
        max_frames=max_frames,
    )
    router = MemoryRouter(config.memory, output_dir=output_dir)
    renderer = GaussianSplatRenderer(config.output)

    per_frame: list[dict] = []
    keyframe_count = 0
    total_render_time = 0.0
    total_update_time = 0.0
    measured_frames = 0
    psnr_values: list[float] = []
    ssim_values: list[float] = []

    for output in adapter:
        update_start = time.perf_counter()
        events, snapshot = router.process(output)
        update_dt = time.perf_counter() - update_start
        total_update_time += update_dt
        if output.is_keyframe:
            keyframe_count += 1

        target_bgr = cv2.imread(output.image_path, cv2.IMREAD_COLOR) if output.image_path else None
        if target_bgr is None:
            continue
        target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)

        render_start = time.perf_counter()
        render = renderer.render_current_view(
            router=router,
            pose=output.pose,
            image_shape=target_rgb.shape[:2],
            image_path=output.image_path,
        )
        render_dt = time.perf_counter() - render_start
        total_render_time += render_dt
        measured_frames += 1

        frame_psnr = psnr(render.image_rgb, target_rgb)
        frame_ssim = ssim_rgb(render.image_rgb, target_rgb)
        psnr_values.append(frame_psnr)
        ssim_values.append(frame_ssim)

        saved = {}
        if save_images and compare_stride > 0 and (measured_frames - 1) % compare_stride == 0:
            saved = save_render_triplet(image_dir, output.frame_idx, target_rgb, render.image_rgb)

        per_frame.append(
            {
                "frame_idx": output.frame_idx,
                "timestamp_sec": output.timestamp_sec,
                "is_keyframe": output.is_keyframe,
                "update_ms": round(update_dt * 1000.0, 3),
                "render_ms": round(render_dt * 1000.0, 3),
                "psnr": round(frame_psnr, 4),
                "ssim": round(frame_ssim, 5),
                "point_count": render.point_count,
                "projected_points": render.projected_points,
                "active_points": render.active_points,
                "archived_points": render.archived_points,
                "warmstart_points": render.warmstart_points,
                "events": [event.event_type for event in events],
                "snapshot": snapshot,
                **saved,
            }
        )

    summary = router.finalize()
    benchmark = {
        "config": config.to_dict(),
        "sequence_path": str(Path(args.sequence_path).expanduser().resolve()),
        "trajectory_path": str(Path(args.trajectory_path).expanduser().resolve()),
        "effective_input": {"frame_step": frame_step, "max_frames": max_frames},
        "keyframe_count": keyframe_count,
        "measured_frames": measured_frames,
        "memory_summary": summary,
        "render_metrics": {
            "mean_psnr": round(float(np.mean(psnr_values)) if psnr_values else 0.0, 4),
            "mean_ssim": round(float(np.mean(ssim_values)) if ssim_values else 0.0, 5),
            "mean_update_ms": round(1000.0 * total_update_time / max(1, measured_frames), 3),
            "mean_render_ms": round(1000.0 * total_render_time / max(1, measured_frames), 3),
            "approx_fps": round(measured_frames / max(1e-8, total_update_time + total_render_time), 3),
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

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "measured_frames": measured_frames,
                "mean_psnr": benchmark["render_metrics"]["mean_psnr"],
                "mean_ssim": benchmark["render_metrics"]["mean_ssim"],
                "mean_update_ms": benchmark["render_metrics"]["mean_update_ms"],
                "mean_render_ms": benchmark["render_metrics"]["mean_render_ms"],
                "approx_fps": benchmark["render_metrics"]["approx_fps"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
