#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_paths(lingbot_map_root: Path | None = None) -> None:
    root = _repo_root()
    if lingbot_map_root is not None:
        lingbot_root = lingbot_map_root.expanduser()
    elif os.environ.get("LINGBOT_MAP_ROOT"):
        lingbot_root = Path(os.environ["LINGBOT_MAP_ROOT"]).expanduser()
    else:
        lingbot_root = root / "third_party_research" / "lingbot-map"
    if str(lingbot_root) not in sys.path:
        sys.path.insert(0, str(lingbot_root))


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def _load_image_paths(args: argparse.Namespace) -> list[str]:
    if args.image_list:
        base = Path(args.image_root).expanduser().resolve() if args.image_root else Path(args.image_list).parent
        paths = []
        for line in Path(args.image_list).expanduser().read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped).expanduser()
            if not path.is_absolute():
                path = base / path
            paths.append(str(path.resolve()))
        return paths

    image_dir = Path(args.image_dir).expanduser().resolve()
    return [
        str(path.resolve())
        for path in sorted(image_dir.glob(args.glob))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def _windows(image_paths: list[str], window_size: int, stride: int) -> list[list[str]]:
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
    return {
        "count": float(len(values)),
        "mean_sec": float(statistics.fmean(values)),
        "median_sec": float(sorted_values[len(sorted_values) // 2]),
        "min_sec": float(min(values)),
        "max_sec": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a LingBot depth student checkpoint.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-dir")
    source.add_argument("--image-list")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--glob", default="*")
    parser.add_argument("--lingbot-map-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-image-size", type=int, default=224)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--patch-embed", default="conv")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--aggregator-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--print-each-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_paths(args.lingbot_map_root)
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.aggregator_dtype]

    image_paths = _load_image_paths(args)
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    windows = _windows(image_paths, args.window_size, args.stride)
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
    if not windows:
        raise RuntimeError("Not enough images for benchmark windows")

    model = GCTStream(
        img_size=args.model_image_size,
        patch_size=14,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        patch_embed=args.patch_embed,
        enable_camera=False,
        enable_point=False,
        enable_local_point=False,
        enable_depth=True,
        enable_3d_rope=False,
        use_sdpa=True,
        camera_num_iterations=1,
        num_frame_for_scale=args.num_scale_frames,
        kv_cache_sliding_window=max(1, args.window_size),
        kv_cache_scale_frames=max(1, args.num_scale_frames),
        use_gradient_checkpoint=False,
    )
    state = _load_state(args.checkpoint.expanduser().resolve())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)
    model = model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rows: list[dict[str, Any]] = []
    warmup = min(args.warmup_windows, len(windows))
    for index, window in enumerate(windows):
        start = time.perf_counter()
        images = load_and_preprocess_images(
            window,
            mode="crop",
            image_size=args.image_size,
            patch_size=14,
        ).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                out = model(
                    images,
                    num_frame_for_scale=args.num_scale_frames,
                    num_frame_per_block=args.num_scale_frames,
                    causal_inference=True,
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = {
            "index": index,
            "warmup": index < warmup,
            "elapsed_sec": elapsed,
            "hz": 1.0 / elapsed if elapsed > 0 else 0.0,
            "depth_shape": list(out["depth"].shape),
            "frame_start": Path(window[0]).name,
            "frame_end": Path(window[-1]).name,
        }
        rows.append(row)
        if args.print_each_window:
            print(json.dumps(row), flush=True)

    eval_rows = [row for row in rows if not row["warmup"]]
    latencies = [float(row["elapsed_sec"]) for row in eval_rows]
    stats = _stats(latencies)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "image_count": len(image_paths),
        "window_count": len(windows),
        "measured_window_count": len(eval_rows),
        "image_size": args.image_size,
        "model_image_size": args.model_image_size,
        "patch_embed": args.patch_embed,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "latency": stats,
        "mean_hz": 1.0 / stats["mean_sec"] if stats else 0.0,
        "windows": rows,
    }
    if device.type == "cuda":
        result["cuda_memory"] = {
            "max_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
            "max_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024**3)),
        }
    text = json.dumps(result, indent=2)
    if args.output_json:
        args.output_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output_json.expanduser().resolve().write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
