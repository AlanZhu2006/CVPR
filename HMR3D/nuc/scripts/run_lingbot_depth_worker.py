#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import LingBotDepthWorker, LingBotDepthWorkerConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _resolve_path(path_str: str, *, base: Path | None) -> str:
    path = Path(path_str).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    if base is not None:
        candidates.append(base / path)
        candidates.append(base / path.name)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Failed to resolve image path {path_str}. Searched: {searched}")


def _load_manifest_images(manifest_json: Path, image_root: Path | None) -> list[str]:
    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    base = image_root or manifest_json.parent
    return [_resolve_path(path, base=base) for path in manifest.get("image_paths", [])]


def _load_image_list(image_list: Path, image_root: Path | None) -> list[str]:
    base = image_root or image_list.parent
    image_paths: list[str] = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        image_paths.append(_resolve_path(stripped, base=base))
    return image_paths


def _load_image_dir(image_dir: Path, pattern: str) -> list[str]:
    return [
        str(path.resolve())
        for path in sorted(image_dir.glob(pattern))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a persistent LingBot depth-only background worker on image streams."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-json", help="LingBot dense job manifest.")
    source.add_argument("--image-dir", help="Directory containing input images.")
    source.add_argument("--image-list", help="Text file with one image path per line.")
    parser.add_argument("--image-root", default="", help="Root used to resolve relative image paths.")
    parser.add_argument("--glob", default="*", help="Glob under --image-dir.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lingbot-map-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-sleep-sec", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--max-queue", type=int, default=4)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-offload-to-cpu", action="store_true")
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--disable-depth", action="store_true")
    parser.add_argument("--enable-point", action="store_true")
    parser.add_argument("--enable-3d-rope", action="store_true")
    parser.add_argument("--depth-head-trt-engine", default="", help="Optional fixed-shape TensorRT DPT depth-head engine.")
    parser.add_argument("--model-patch-embed", default="", help="Direct GCTStream patch embed override, e.g. conv for tiny students.")
    parser.add_argument("--model-embed-dim", type=int, default=0)
    parser.add_argument("--model-depth", type=int, default=0)
    parser.add_argument("--model-num-heads", type=int, default=0)
    parser.add_argument("--model-mlp-ratio", type=float, default=0.0)
    parser.add_argument("--no-compress-output", action="store_true")
    parser.add_argument("--preload-model", action="store_true")
    parser.add_argument("--warmup-first-window", action="store_true")
    parser.add_argument("--submit-blocking", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lingbot_map_root:
        os.environ["LINGBOT_MAP_ROOT"] = str(Path(args.lingbot_map_root).expanduser().resolve())

    image_root = Path(args.image_root).expanduser().resolve() if args.image_root else None
    if args.manifest_json:
        image_paths = _load_manifest_images(Path(args.manifest_json).expanduser().resolve(), image_root)
    elif args.image_list:
        image_paths = _load_image_list(Path(args.image_list).expanduser().resolve(), image_root)
    else:
        image_paths = _load_image_dir(Path(args.image_dir).expanduser().resolve(), args.glob)
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise RuntimeError("No input images found")

    config = LingBotDepthWorkerConfig(
        model_path=str(Path(args.model_path).expanduser().resolve()),
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        image_size=args.image_size,
        model_image_size=args.model_image_size,
        window_size=args.window_size,
        stride=args.stride,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        max_queue=args.max_queue,
        force_cpu=args.force_cpu,
        offload_to_cpu=not args.no_offload_to_cpu,
        enable_camera=args.enable_camera,
        enable_depth=not args.disable_depth,
        enable_point=args.enable_point,
        enable_3d_rope=args.enable_3d_rope,
        depth_head_trt_engine=args.depth_head_trt_engine,
        model_patch_embed=args.model_patch_embed,
        model_embed_dim=args.model_embed_dim,
        model_depth=args.model_depth,
        model_num_heads=args.model_num_heads,
        model_mlp_ratio=args.model_mlp_ratio,
        compress_outputs=not args.no_compress_output,
        preload_model=args.preload_model,
        warmup_first_window=args.warmup_first_window,
    )
    worker = LingBotDepthWorker(config)
    started = time.perf_counter()
    worker.start()
    submitted_windows = 0
    dropped_windows = 0
    try:
        for frame_idx, image_path in enumerate(image_paths):
            submitted = worker.submit(
                image_path,
                frame_idx=frame_idx,
                timestamp_sec=float(frame_idx),
                block=args.submit_blocking,
            )
            submitted_windows += int(submitted)
            dropped_windows += int(not submitted and frame_idx + 1 >= args.window_size)
            if args.frame_sleep_sec > 0:
                time.sleep(args.frame_sleep_sec)
        worker.stop(drain=True)
    except KeyboardInterrupt:
        worker.stop(drain=False)
        raise

    summary = worker.status()
    summary.update(
        {
            "input_image_count": len(image_paths),
            "submitted_window_events": submitted_windows,
            "dropped_window_events": dropped_windows,
            "elapsed_sec": time.perf_counter() - started,
        }
    )
    summary_path = Path(args.output_dir).expanduser().resolve() / "worker_run_summary.json"
    summary_text = json.dumps(summary, indent=2)
    summary_path.write_text(summary_text, encoding="utf-8")
    print(summary_text)
    return 0 if summary.get("failed_windows", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
