#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_paths(lingbot_map_root: Path | None = None) -> None:
    root = _repo_root()
    if lingbot_map_root is not None:
        lingbot_root = lingbot_map_root
    elif os.environ.get("LINGBOT_MAP_ROOT"):
        lingbot_root = Path(os.environ["LINGBOT_MAP_ROOT"]).expanduser()
    else:
        lingbot_root = root / "third_party_research" / "lingbot-map"
    if str(lingbot_root) not in sys.path:
        sys.path.insert(0, str(lingbot_root))
    # The Jetson CUDA venv intentionally does not include onnx. Import torch first,
    # then append user-site so a CPU-only user-site torch cannot shadow CUDA torch.
    user_site = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
    if user_site.exists() and str(user_site) not in sys.path:
        sys.path.append(str(user_site))


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


class DepthOnlyWrapper(torch.nn.Module):
    def __init__(self, model, num_scale_frames: int, dtype: torch.dtype):
        super().__init__()
        self.model = model
        self.num_scale_frames = int(num_scale_frames)
        self.dtype = dtype

    def forward(self, images: torch.Tensor):
        autocast_enabled = images.is_cuda and self.dtype != torch.float32
        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=autocast_enabled):
            out = self.model(
                images,
                num_frame_for_scale=self.num_scale_frames,
                num_frame_per_block=self.num_scale_frames,
                causal_inference=True,
            )
        return out["depth"], out["depth_conf"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export fixed-shape LingBot depth-only ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_repo_root() / "third_party_research" / "lingbot_cache" / "lingbot-map-depth-fp16.pt",
    )
    parser.add_argument(
        "--lingbot-map-root",
        type=Path,
        default=None,
        help="Path to the lingbot-map checkout. Defaults to LINGBOT_MAP_ROOT or third_party_research/lingbot-map.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--input-width", type=int, default=336)
    parser.add_argument("--input-height", type=int, default=196)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used by PyTorch during ONNX tracing. CPU avoids Jetson NvMap export failures.",
    )
    parser.add_argument(
        "--keep-antialias",
        action="store_true",
        help="Keep bicubic antialiasing in DINO positional interpolation. Disabled by default for ONNX compatibility.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    _add_paths(args.lingbot_map_root.expanduser().resolve() if args.lingbot_map_root else None)
    from lingbot_map.models.gct_stream import GCTStream

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    result = {
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "model_image_size": args.model_image_size,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "num_scale_frames": args.num_scale_frames,
        "dtype": args.dtype,
        "device": args.device,
        "ok": False,
    }
    start = time.perf_counter()
    try:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this export smoke test")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        model = GCTStream(
            img_size=args.model_image_size,
            patch_size=14,
            embed_dim=1024,
            patch_embed="dinov2_vitl14_reg",
            enable_camera=False,
            enable_point=False,
            enable_local_point=False,
            enable_depth=True,
            enable_3d_rope=False,
            use_sdpa=True,
            camera_num_iterations=1,
            num_frame_for_scale=args.num_scale_frames,
            kv_cache_sliding_window=max(1, args.num_scale_frames),
            kv_cache_scale_frames=max(1, args.num_scale_frames),
            use_gradient_checkpoint=False,
        )
        if not args.keep_antialias and hasattr(model.aggregator.patch_embed, "interpolate_antialias"):
            model.aggregator.patch_embed.interpolate_antialias = False
        state = _load_state(args.checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        result["missing_keys"] = len(missing)
        result["unexpected_keys"] = len(unexpected)
        del state
        gc.collect()

        # Keep DPT depth head fp32. LingBot upcasts features before DPT heads.
        if dtype != torch.float32:
            model.aggregator = model.aggregator.to(dtype=dtype)
        model = model.to(args.device).eval()
        wrapper = DepthOnlyWrapper(model, args.num_scale_frames, dtype).eval()
        dummy = torch.zeros(
            1,
            args.num_scale_frames,
            3,
            args.input_height,
            args.input_width,
            device=args.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            depth, depth_conf = wrapper(dummy)
        if args.device == "cuda":
            torch.cuda.synchronize()
        result["smoke_depth_shape"] = list(depth.shape)
        result["smoke_depth_conf_shape"] = list(depth_conf.shape)
        if args.device == "cuda":
            result["smoke_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            result["smoke_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 1e9

        if not args.smoke_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.onnx.export(
                wrapper,
                (dummy,),
                str(args.output),
                input_names=["images"],
                output_names=["depth", "depth_conf"],
                opset_version=18,
                dynamo=False,
                external_data=True,
                do_constant_folding=False,
            )
            result["onnx_size_bytes"] = args.output.stat().st_size if args.output.exists() else 0
        result["ok"] = True
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = time.perf_counter() - start
        print(json.dumps(result, indent=2))

    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
