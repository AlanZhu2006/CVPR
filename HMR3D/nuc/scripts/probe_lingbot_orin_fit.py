#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_lingbot_path() -> None:
    root = _repo_root() / "third_party_research" / "lingbot-map"
    sys.path.insert(0, str(root))


def _bytes_of_module(module: torch.nn.Module) -> int:
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return total


def _fmt_gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _cuda_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "allocated_gib": _fmt_gib(torch.cuda.memory_allocated()),
        "reserved_gib": _fmt_gib(torch.cuda.memory_reserved()),
        "max_allocated_gib": _fmt_gib(torch.cuda.max_memory_allocated()),
        "max_reserved_gib": _fmt_gib(torch.cuda.max_memory_reserved()),
    }


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")
    return state


def _build_model(args: argparse.Namespace):
    _add_lingbot_path()
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=14,
        embed_dim=1024,
        patch_embed="dinov2_vitl14_reg",
        enable_camera=False,
        enable_point=False,
        enable_local_point=False,
        enable_depth=True,
        use_sdpa=True,
        enable_3d_rope=args.enable_3d_rope,
        camera_num_iterations=1,
        num_frame_for_scale=args.num_scale_frames,
        kv_cache_sliding_window=max(1, args.window_size),
        kv_cache_scale_frames=max(1, args.num_scale_frames),
        use_gradient_checkpoint=False,
    )
    model.eval()
    return model


def _select_target(model: torch.nn.Module, target: str) -> torch.nn.Module:
    if target == "full":
        return model
    if target == "aggregator":
        return model.aggregator
    if target == "depth_head":
        return model.depth_head
    if target == "first_frame_block":
        return model.aggregator.frame_blocks[0]
    if target == "first_global_block":
        return model.aggregator.global_blocks[0]
    raise ValueError(f"Unknown target: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe which LingBot depth-only components fit on Jetson CUDA."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_repo_root() / "third_party_research" / "lingbot_cache" / "lingbot-map-depth-fp16.pt",
    )
    parser.add_argument(
        "--target",
        choices=("full", "aggregator", "depth_head", "first_frame_block", "first_global_block"),
        default="aggregator",
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--enable-3d-rope", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "target": args.target,
        "dtype": args.dtype,
        "image_size": args.image_size,
        "window_size": args.window_size,
        "num_scale_frames": args.num_scale_frames,
        "ok": False,
        "steps": [],
    }

    start = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        model = _build_model(args)
        result["steps"].append({"step": "build_cpu_model", "cuda": _cuda_snapshot()})

        state = _load_state(args.checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        result["missing_keys"] = len(missing)
        result["unexpected_keys"] = len(unexpected)
        result["sample_missing_keys"] = list(missing[:10])
        result["sample_unexpected_keys"] = list(unexpected[:10])
        del state
        gc.collect()
        result["steps"].append({"step": "load_checkpoint_cpu", "cuda": _cuda_snapshot()})

        target_module = _select_target(model, args.target)
        result["target_cpu_param_gib"] = _fmt_gib(_bytes_of_module(target_module))

        target_module.to(dtype=dtype)
        gc.collect()
        result["steps"].append({"step": "cast_target_cpu", "cuda": _cuda_snapshot()})

        target_module.to(device="cuda")
        torch.cuda.synchronize()
        result["steps"].append({"step": "move_target_cuda", "cuda": _cuda_snapshot()})
        result["ok"] = True
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        if torch.cuda.is_available():
            result["cuda_after_error"] = _cuda_snapshot()
    finally:
        result["elapsed_sec"] = time.perf_counter() - start
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))

    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
