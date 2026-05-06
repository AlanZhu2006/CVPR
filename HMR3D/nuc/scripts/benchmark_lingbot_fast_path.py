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

from nuc_runtime.lingbot_adapter import LingBotReconstructor  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LingBot model/preprocess/postprocess without ROS, WebUI, "
            "PointCloud2 serialization, or live-map fusion."
        )
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing RGB frames.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lingbot-map-root", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--camera-num-iterations", type=int, default=1)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--use-sdpa", action="store_true")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--compile-lingbot-model", action="store_true")
    parser.add_argument("--compile-warmup-passes", type=int, default=3)
    parser.add_argument("--compile-warmup-stream-frames", type=int, default=10)
    parser.add_argument("--depth-head-trt-engine", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def collect_images(image_dir: Path) -> list[str]:
    paths = [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return [str(path.resolve()) for path in paths]


def stats(values: list[float]) -> dict[str, float]:
    finite = sorted(float(value) for value in values if value == value)
    if not finite:
        return {"count": 0.0}
    return {
        "count": float(len(finite)),
        "mean": float(sum(finite) / len(finite)),
        "median": float(finite[len(finite) // 2]),
        "min": float(finite[0]),
        "max": float(finite[-1]),
        "p90": float(finite[min(len(finite) - 1, int(0.9 * len(finite)))]),
    }


def main() -> int:
    args = parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    images = collect_images(image_dir)
    if len(images) < args.window_size:
        raise SystemExit(f"Need at least {args.window_size} images in {image_dir}; found {len(images)}")
    if args.lingbot_map_root:
        os.environ["LINGBOT_MAP_ROOT"] = str(Path(args.lingbot_map_root).expanduser().resolve())
    if args.dtype != "auto":
        os.environ["LINGBOT_MODEL_DTYPE"] = args.dtype

    reconstructor = LingBotReconstructor(
        model_path=args.model_path,
        image_size=args.image_size,
        model_image_size=args.model_image_size,
        mode="streaming",
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=1,
        camera_num_iterations=args.camera_num_iterations,
        offload_to_cpu=args.offload_to_cpu,
        use_sdpa=args.use_sdpa,
        force_cpu=args.force_cpu,
        enable_camera=False,
        enable_depth=True,
        enable_point=False,
        enable_3d_rope=False,
        depth_head_trt_engine=args.depth_head_trt_engine or None,
        compile_model=args.compile_lingbot_model,
        compile_warmup_passes=args.compile_warmup_passes,
        compile_warmup_stream_frames=args.compile_warmup_stream_frames,
    )

    windows: list[list[str]] = []
    max_start = max(0, len(images) - args.window_size)
    for iteration in range(max(1, args.warmup + args.iterations)):
        start = min(max_start, iteration * max(1, args.stride))
        windows.append(images[start : start + args.window_size])

    records: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration, window in enumerate(windows):
        _, summary = reconstructor.run_on_image_paths(window)
        profile = summary.get("profile_sec", {})
        record = {key: float(value) for key, value in profile.items() if isinstance(value, (int, float))}
        record["iteration"] = float(iteration)
        record["warmup"] = float(iteration < args.warmup)
        record["frame_count"] = float(len(window))
        records.append(record)
        print(json.dumps({"iteration": iteration, "profile_sec": record}, ensure_ascii=False), flush=True)

    measured = records[args.warmup :]
    forward = [item.get("model_forward_sec", 0.0) for item in measured]
    total = [item.get("total_sec", 0.0) for item in measured]
    frame_count = max(1, args.window_size)
    summary_payload = {
        "image_dir": str(image_dir),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "image_size": int(args.image_size),
        "model_image_size": int(args.model_image_size),
        "window_size": int(args.window_size),
        "iterations": int(args.iterations),
        "warmup": int(args.warmup),
        "dtype_env": os.environ.get("LINGBOT_MODEL_DTYPE", "auto"),
        "use_sdpa": bool(args.use_sdpa),
        "offload_to_cpu": bool(args.offload_to_cpu),
        "force_cpu": bool(args.force_cpu),
        "compile_lingbot_model": bool(args.compile_lingbot_model),
        "compile_warmup_passes": int(args.compile_warmup_passes),
        "compile_warmup_stream_frames": int(args.compile_warmup_stream_frames),
        "wall_sec": time.perf_counter() - started,
        "pure_model_forward_sec_per_window": stats(forward),
        "pure_model_forward_ms_per_frame": stats([(value / frame_count) * 1000.0 for value in forward]),
        "pure_model_forward_fps": stats([frame_count / value for value in forward if value > 0]),
        "total_sec_per_window": stats(total),
        "total_fps_including_prepost": stats([frame_count / value for value in total if value > 0]),
        "profiles": measured,
    }
    if records:
        summary_payload["last_profile"] = records[-1]
    output = json.dumps(summary_payload, indent=2, ensure_ascii=False)
    print(output)
    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
