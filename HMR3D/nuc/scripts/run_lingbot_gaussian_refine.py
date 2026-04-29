from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import load_runtime_config
from nuc_runtime.gaussian_refine import (
    LocalGaussianRefiner,
    build_surface_prior_bundle,
    load_gaussian_refine_config,
    load_refine_frames,
    write_refined_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short-window photometric Gaussian refine on a LingBot-seeded Gaussian handle."
    )
    parser.add_argument("--gaussians-npz", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--dense-geometry-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="", help="Runtime config YAML. If it has a top-level refine section, it will also be used.")
    parser.add_argument("--image-root", default="", help="Optional job root for remapping local image paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_config = load_runtime_config(args.config or None)
    refine_config = load_gaussian_refine_config(args.config or None)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussians_npz = Path(args.gaussians_npz).expanduser().resolve()
    bundle_npz = np_load(gaussians_npz)
    bundle = {key: bundle_npz[key] for key in bundle_npz.files}

    frames = load_refine_frames(
        summary_json=args.summary_json,
        dense_geometry_npz=args.dense_geometry_npz,
        image_root=args.image_root or None,
        image_scale=float(refine_config.image_scale),
        depth_conf_threshold=float(refine_config.depth_conf_threshold),
        device=str(refine_config.device),
    )
    surface_bundle = build_surface_prior_bundle(
        frames=frames,
        runtime_memory_config=runtime_config.memory,
        max_points=int(refine_config.surface_prior_max_points),
    )
    surface_paths = {}
    if surface_bundle["xyz"].size > 0:
        surface_dir = output_dir / "surface_prior"
        surface_dir.mkdir(parents=True, exist_ok=True)
        surface_paths = write_refined_outputs(
            output_dir=surface_dir,
            bundle=surface_bundle,
            runtime_memory_config=runtime_config.memory,
        )
    start = time.perf_counter()
    refiner = LocalGaussianRefiner(
        bundle=bundle,
        runtime_output_config=runtime_config.output,
        refine_config=refine_config,
        surface_bundle=surface_bundle,
    )
    result = refiner.run(frames=frames, output_dir=output_dir)
    elapsed = time.perf_counter() - start

    paths = write_refined_outputs(
        output_dir=output_dir,
        bundle=result["bundle"],
        runtime_memory_config=runtime_config.memory,
    )
    metrics_path = output_dir / "refine_metrics.json"
    metrics_path.write_text(
        json.dumps(result["summary"]["metrics"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_payload = {}
    if args.config:
        config_payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    summary = {
        "gaussians_npz": str(gaussians_npz),
        "summary_json": str(Path(args.summary_json).expanduser().resolve()),
        "dense_geometry_npz": str(Path(args.dense_geometry_npz).expanduser().resolve()),
        "image_root": str(Path(args.image_root).expanduser().resolve()) if args.image_root else "",
        "frame_count": len(frames),
        "elapsed_sec": round(float(elapsed), 3),
        "runtime_config": runtime_config.to_dict(),
        "config_yaml": config_payload,
        "refine_config": refine_config.__dict__,
        "surface_prior_summary": result["summary"].get("surface_prior", {}),
        "surface_prior_paths": surface_paths,
        "metrics_path": str(metrics_path),
        **paths,
        "point_count": int(result["bundle"]["xyz"].shape[0]),
        "preview_dir": result["summary"]["preview_dir"],
    }
    summary_path = output_dir / "refine_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **paths, "point_count": summary["point_count"], "elapsed_sec": summary["elapsed_sec"]}, indent=2, ensure_ascii=False))


def np_load(path: Path):
    import numpy as np

    return np.load(path)


if __name__ == "__main__":
    main()
