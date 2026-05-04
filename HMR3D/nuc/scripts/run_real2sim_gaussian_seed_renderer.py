#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import GaussianSplatRenderer, psnr, save_render_triplet, ssim_rgb
from nuc_runtime.config import load_runtime_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the latest real-to-sim Gaussian seed against exported RGB keyframes."
    )
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--backend", choices=("cpu", "gsplat"), default="cpu")
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--render-view-budget-points", type=int, default=32000)
    parser.add_argument("--render-depth-window-m", type=float, default=55.0)
    parser.add_argument("--render-internal-scale", type=float, default=0.75)
    parser.add_argument("--render-min-radius-px", type=float, default=1.0)
    parser.add_argument("--render-max-radius-px", type=float, default=28.0)
    parser.add_argument("--compare-viewer", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_latest_sequence(baseline_dir: Path) -> tuple[Path, dict[str, Any]]:
    latest_manifest = baseline_dir / "latest_manifest.json"
    if latest_manifest.exists():
        latest = _read_json(latest_manifest)
        manifest_path = Path(str(latest.get("manifest", "")))
        if manifest_path.exists():
            return manifest_path.parent, _read_json(manifest_path)
    latest_dir = baseline_dir / "latest"
    manifest_path = latest_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing latest manifest under {baseline_dir}")
    return latest_dir, _read_json(manifest_path)


def _relative_path(seq_dir: Path, rel: str) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else seq_dir / path


def _scale_pose_image(rgb: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or rgb.shape[1] == width:
        return rgb
    scale = float(width) / max(1.0, float(rgb.shape[1]))
    height = max(1, int(round(rgb.shape[0] * scale)))
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def _scale_intrinsics(K: np.ndarray, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    out = K.astype(np.float32).copy()
    out[0, :] *= float(dst_w) / max(1.0, float(src_w))
    out[1, :] *= float(dst_h) / max(1.0, float(src_h))
    return out


def _write_intrinsics_sidecar(image_path: Path, K: np.ndarray) -> Path:
    sidecar = image_path.with_suffix(image_path.suffix + ".intrinsics.json")
    payload = {"K": K.astype(float).tolist()}
    sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return sidecar


def main() -> int:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).expanduser().resolve()
    seq_dir, manifest = _load_latest_sequence(baseline_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else seq_dir / "gaussian_seed_renders"
    render_dir = output_dir / "renders"
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    config = load_runtime_config(args.config or None)
    config.output.render_backend = args.backend
    config.output.render_view_budget_points = int(args.render_view_budget_points)
    config.output.render_depth_window_m = float(args.render_depth_window_m)
    config.output.render_internal_scale = float(args.render_internal_scale)
    config.output.render_min_radius_px = float(args.render_min_radius_px)
    config.output.render_max_radius_px = float(args.render_max_radius_px)
    renderer = GaussianSplatRenderer(config.output)

    seed_npz = seq_dir / "gaussian_seed" / "gaussians_seed.npz"
    if not seed_npz.exists():
        raise FileNotFoundError(f"Missing Gaussian seed: {seed_npz}")
    seed = np.load(seed_npz)
    bundle = {key: seed[key] for key in seed.files}
    if (
        args.backend == "gsplat"
        and bundle["xyz"].shape[0] > 0
        and ("gsplat_scales" not in bundle or "gsplat_quats" not in bundle)
    ):
        start_precompute = time.perf_counter()
        scales, quats = renderer._axes_to_gsplat_params(
            axis_u=bundle["axis_u"].astype(np.float32),
            axis_v=bundle["axis_v"].astype(np.float32),
            scale_base=bundle["scale"].astype(np.float32),
        )
        bundle["gsplat_scales"] = scales
        bundle["gsplat_quats"] = quats
        gsplat_precompute_ms = (time.perf_counter() - start_precompute) * 1000.0
    else:
        gsplat_precompute_ms = 0.0

    frames = manifest.get("frames", [])
    if args.frame_stride > 1:
        frames = frames[:: int(args.frame_stride)]
    if args.max_frames > 0:
        frames = frames[: int(args.max_frames)]
    if not frames:
        raise RuntimeError("No frames selected for rendering")

    per_frame: list[dict[str, Any]] = []
    total_render = 0.0
    for item in frames:
        rgb_path = _relative_path(seq_dir, str(item["rgb"]))
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        target_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src_shape = target_rgb.shape[:2]
        target_rgb = _scale_pose_image(target_rgb, int(args.width))
        K = np.asarray(item["intrinsics"], dtype=np.float32)
        K = _scale_intrinsics(K, src_shape, target_rgb.shape[:2])
        pose = np.loadtxt(_relative_path(seq_dir, str(item["pose_txt"]))).astype(np.float32)
        image_sidecar = _write_intrinsics_sidecar(render_dir / f"{int(item['index']):06d}.png", K)

        start = time.perf_counter()
        rendered = renderer.render_bundle_view(
            bundle=bundle,
            pose=pose,
            image_shape=target_rgb.shape[:2],
            image_path=str(image_sidecar),
        )
        render_ms = (time.perf_counter() - start) * 1000.0
        total_render += render_ms
        frame_psnr = psnr(rendered.image_rgb, target_rgb)
        frame_ssim = ssim_rgb(rendered.image_rgb, target_rgb)
        saved = save_render_triplet(render_dir, int(item["index"]), target_rgb, rendered.image_rgb)
        per_frame.append(
            {
                "frame_idx": int(item["index"]),
                "source_frame_idx": int(item.get("source_frame_idx", item["index"])),
                "render_ms": round(render_ms, 3),
                "psnr": round(frame_psnr, 4),
                "ssim": round(frame_ssim, 5),
                "point_count": rendered.point_count,
                "projected_points": rendered.projected_points,
                **saved,
            }
        )

    summary = {
        "schema": "real2sim_gaussian_seed_render.v1",
        "baseline_dir": str(baseline_dir),
        "sequence_dir": str(seq_dir),
        "output_dir": str(output_dir),
        "backend_requested": args.backend,
        "seed_npz": str(seed_npz),
        "seed_point_count": int(bundle["xyz"].shape[0]),
        "gsplat_precompute_ms": round(gsplat_precompute_ms, 3),
        "rendered_frames": len(per_frame),
        "mean_render_ms": round(total_render / max(1, len(per_frame)), 3),
        "mean_psnr": round(float(np.mean([f["psnr"] for f in per_frame])) if per_frame else 0.0, 4),
        "mean_ssim": round(float(np.mean([f["ssim"] for f in per_frame])) if per_frame else 0.0, 5),
        "frames": per_frame,
    }
    (output_dir / "render_benchmark_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "render_benchmark_frames.json").write_text(json.dumps(per_frame, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.compare_viewer:
        viewer_script = Path(__file__).resolve().with_name("generate_gsconsole_style_compare_viewer.py")
        import subprocess

        subprocess.run(
            [sys.executable, str(viewer_script), "--run-dir", str(output_dir)],
            check=False,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
