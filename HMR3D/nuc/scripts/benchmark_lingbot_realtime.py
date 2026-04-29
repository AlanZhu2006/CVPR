from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import LingBotReconstructor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LingBot as a streaming/keyframe mapper by repeatedly "
            "running sliding-window inference with one model instance."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-json", help="LingBot dense job manifest.")
    source.add_argument("--image-dir", help="Directory containing input images.")
    source.add_argument("--image-list", help="Text file with one image path per line.")
    parser.add_argument("--image-root", default="", help="Root used to resolve manifest/list relative paths.")
    parser.add_argument("--glob", default="*", help="Glob under --image-dir. Default: *")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lingbot-map-root", default="", help="Optional lingbot-map checkout.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--target-hz", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument(
        "--model-image-size",
        type=int,
        default=0,
        help="LingBot model construction size. Use 518 when loading official 518 checkpoints with smaller inputs.",
    )
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--mode", choices=("streaming", "windowed"), default="streaming")
    parser.add_argument("--num-scale-frames", type=int, default=4)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--depth-only", action="store_true", help="Disable LingBot camera/point heads.")
    parser.add_argument("--disable-camera", action="store_true", help="Disable LingBot camera head.")
    parser.add_argument("--disable-point", action="store_true", help="Disable LingBot point head.")
    parser.add_argument("--disable-3d-rope", action="store_true", help="Disable LingBot temporal 3D RoPE.")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-offload-to-cpu", action="store_true")
    parser.add_argument("--print-each-window", action="store_true")
    return parser.parse_args()


def _resolve_path(path_str: str, *, base: Path | None) -> str:
    path = Path(path_str).expanduser()
    candidates = []
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
    image_paths = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        image_paths.append(_resolve_path(stripped, base=base))
    return image_paths


def _load_image_dir(image_dir: Path, pattern: str) -> list[str]:
    paths = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return [str(path.resolve()) for path in sorted(paths)]


def _windows(image_paths: list[str], window_size: int, stride: int) -> list[list[str]]:
    if window_size < 1:
        raise ValueError("--window-size must be >= 1")
    if stride < 1:
        raise ValueError("--stride must be >= 1")
    if len(image_paths) < window_size:
        return []
    return [
        image_paths[start : start + window_size]
        for start in range(0, len(image_paths) - window_size + 1, stride)
    ]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(values)
    p50 = sorted_values[len(sorted_values) // 2]
    p90 = sorted_values[min(len(sorted_values) - 1, int(round(0.9 * (len(sorted_values) - 1))))]
    return {
        "count": float(len(values)),
        "mean_sec": float(statistics.fmean(values)),
        "median_sec": float(p50),
        "p90_sec": float(p90),
        "min_sec": float(min(values)),
        "max_sec": float(max(values)),
    }


def _shape_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("depth_shape", "world_points_shape", "extrinsic_shape", "prediction_keys")
    return {key: summary[key] for key in keys if key in summary}


def main() -> None:
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
    windows = _windows(image_paths, window_size=args.window_size, stride=args.stride)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    if not windows:
        raise RuntimeError(
            f"Not enough images for window_size={args.window_size}; image_count={len(image_paths)}"
        )

    reconstructor = LingBotReconstructor(
        model_path=args.model_path,
        image_size=args.image_size,
        model_image_size=args.model_image_size or None,
        patch_size=args.patch_size,
        mode=args.mode,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        camera_num_iterations=args.camera_num_iterations,
        offload_to_cpu=not args.no_offload_to_cpu,
        force_cpu=args.force_cpu,
        enable_camera=not (args.depth_only or args.disable_camera),
        enable_point=not (args.depth_only or args.disable_point),
        enable_depth=True,
        enable_3d_rope=not args.disable_3d_rope,
    )

    if reconstructor.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(reconstructor.device)

    measured: list[dict[str, Any]] = []
    warmup_count = min(max(0, args.warmup_windows), len(windows))
    for index, window in enumerate(windows):
        start = time.perf_counter()
        predictions_np, summary = reconstructor.run_on_image_paths(window)
        elapsed = time.perf_counter() - start
        is_warmup = index < warmup_count
        row = {
            "index": index,
            "warmup": is_warmup,
            "elapsed_sec": float(elapsed),
            "effective_hz": float(1.0 / elapsed) if elapsed > 0 else 0.0,
            "frame_start": Path(window[0]).name,
            "frame_end": Path(window[-1]).name,
            "prediction_keys": sorted(predictions_np.keys()),
            **_shape_summary(summary),
        }
        measured.append(row)
        if args.print_each_window:
            print(json.dumps(row, ensure_ascii=False))

    eval_rows = [row for row in measured if not row["warmup"]]
    elapsed_values = [float(row["elapsed_sec"]) for row in eval_rows]
    stats = _stats(elapsed_values)
    mean_hz = float(1.0 / stats["mean_sec"]) if stats else 0.0
    result = {
        "image_count": len(image_paths),
        "window_count": len(windows),
        "measured_window_count": len(eval_rows),
        "warmup_windows": warmup_count,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "image_size": int(args.image_size),
        "model_image_size": int(args.model_image_size or args.image_size),
        "patch_size": int(args.patch_size),
        "mode": args.mode,
        "num_scale_frames": int(args.num_scale_frames),
        "keyframe_interval": int(args.keyframe_interval),
        "camera_num_iterations": int(args.camera_num_iterations),
        "depth_only": bool(args.depth_only),
        "enable_camera": not (args.depth_only or args.disable_camera),
        "enable_point": not (args.depth_only or args.disable_point),
        "force_cpu": bool(args.force_cpu),
        "offload_to_cpu": not bool(args.no_offload_to_cpu),
        "device": str(reconstructor.device),
        "target_hz": float(args.target_hz),
        "mean_hz": mean_hz,
        "meets_target": bool(mean_hz >= args.target_hz) if stats else False,
        "latency": stats,
        "windows": measured,
    }
    if reconstructor.device.type == "cuda":
        result["cuda_memory"] = {
            "max_allocated_gb": float(torch.cuda.max_memory_allocated(reconstructor.device) / (1024**3)),
            "max_reserved_gb": float(torch.cuda.max_memory_reserved(reconstructor.device) / (1024**3)),
        }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
