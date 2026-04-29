#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


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


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False, mmap=False)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)!r}")


def _load_teacher_windows(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for summary_path in sorted(root.expanduser().resolve().glob("window_*/lingbot_summary.json")):
        pred_path = summary_path.with_name("lingbot_predictions.npz")
        if not pred_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        image_paths = [str(Path(path).expanduser().resolve()) for path in summary.get("image_paths", [])]
        if not image_paths:
            frames = summary.get("metadata", {}).get("frames", [])
            image_paths = [str(Path(frame.get("image_path", "")).expanduser().resolve()) for frame in frames]
        if not image_paths or not all(Path(path).exists() for path in image_paths):
            continue
        npz = np.load(pred_path)
        if "depth" not in npz:
            continue
        samples.append(
            {
                "summary_json": str(summary_path),
                "predictions_npz": str(pred_path),
                "image_paths": image_paths,
                "depth": np.asarray(npz["depth"], dtype=np.float32),
                "images": np.asarray(npz["images"], dtype=np.float32) if "images" in npz else None,
            }
        )
    if not samples:
        raise RuntimeError(f"No teacher windows found under {root}")
    return samples


def _build_student(args: argparse.Namespace):
    from lingbot_map.models.gct_stream import GCTStream

    return GCTStream(
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


def _predict_depth(model, images: torch.Tensor, num_scale_frames: int) -> torch.Tensor:
    out = model(
        images,
        num_frame_for_scale=num_scale_frames,
        num_frame_per_block=num_scale_frames,
        causal_inference=True,
    )
    depth = out["depth"]
    if depth.ndim == 5 and depth.shape[2] == 1:
        depth = depth.permute(0, 1, 3, 4, 2)
    return depth


def _resize_teacher_to_pred(teacher: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if pred.shape == teacher.shape:
        return teacher
    teacher_nchw = teacher[..., 0].reshape(-1, teacher.shape[-3], teacher.shape[-2]).unsqueeze(1)
    pred_nchw = pred[..., 0].reshape(-1, pred.shape[-3], pred.shape[-2]).unsqueeze(1)
    teacher_nchw = F.interpolate(
        teacher_nchw,
        size=pred_nchw.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return teacher_nchw.squeeze(1).reshape(pred.shape[:-1]).unsqueeze(-1)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    return {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(values[len(values) // 2]),
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def _colorize(value: np.ndarray, *, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 98))
    norm = (arr - vmin) / max(vmax - vmin, 1e-6)
    gray = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    try:
        import cv2

        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    except Exception:
        return np.repeat(gray[..., None], 3, axis=-1)


def _save_preview(
    out_path: Path,
    rgb_chw: np.ndarray | None,
    teacher: np.ndarray,
    pred: np.ndarray,
) -> None:
    from PIL import Image

    teacher_hw = teacher[..., 0]
    pred_hw = pred[..., 0]
    error_hw = np.abs(np.log(np.clip(pred_hw, 1e-3, None)) - np.log(np.clip(teacher_hw, 1e-3, None)))
    if rgb_chw is None:
        rgb = np.zeros((*teacher_hw.shape, 3), dtype=np.uint8)
    else:
        rgb = np.clip(np.moveaxis(rgb_chw, 0, -1) * 255.0, 0, 255).astype(np.uint8)
    depth_min = float(np.nanpercentile(teacher_hw, 2))
    depth_max = float(np.nanpercentile(teacher_hw, 98))
    tiles = [
        rgb,
        _colorize(teacher_hw, vmin=depth_min, vmax=depth_max),
        _colorize(pred_hw, vmin=depth_min, vmax=depth_max),
        _colorize(error_hw, vmin=0.0, vmax=max(0.25, float(np.nanpercentile(error_hw, 98)))),
    ]
    canvas = np.concatenate(tiles, axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def _write_preview_index(output_dir: Path, preview_count: int) -> None:
    previews = sorted((output_dir / "previews").glob("window_*.png"))[:preview_count]
    cards = "\n".join(
        f'<figure><img src="previews/{path.name}" /><figcaption>{path.stem}: RGB | teacher | student | log-error</figcaption></figure>'
        for path in previews
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LingBot student depth eval</title>
  <style>
    body {{ margin: 24px; background: #101214; color: #e9ecef; font: 15px/1.5 sans-serif; }}
    h1 {{ font-size: 22px; }}
    figure {{ margin: 0 0 24px; padding: 16px; background: #171a1d; border: 1px solid #2b3035; border-radius: 12px; }}
    img {{ width: min(100%, 1200px); image-rendering: auto; border-radius: 8px; }}
    figcaption {{ margin-top: 8px; color: #aeb6bf; }}
    code {{ color: #ffd27d; }}
  </style>
</head>
<body>
  <h1>LingBot Student Depth Eval</h1>
  <p>Each strip is <code>RGB | teacher depth | student depth | log-depth error</code>.</p>
  {cards}
</body>
</html>
"""
    (output_dir / "preview_index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LingBot depth student against cached teacher windows.")
    parser.add_argument("--teacher-windows-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lingbot-map-root", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-image-size", type=int, default=224)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--patch-embed", default="conv")
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=3.0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--aggregator-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_paths(args.lingbot_map_root)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.aggregator_dtype]

    samples = _load_teacher_windows(args.teacher_windows_dir)
    if args.max_windows > 0:
        samples = samples[: args.max_windows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = _build_student(args)
    state = _load_state(args.checkpoint.expanduser().resolve())
    missing, unexpected = model.load_state_dict(state, strict=False)
    if dtype != torch.float32:
        model.aggregator = model.aggregator.to(dtype=dtype)
    model = model.to(device).eval()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, sample in enumerate(samples):
            images = load_and_preprocess_images(
                sample["image_paths"],
                mode="crop",
                image_size=args.image_size,
                patch_size=14,
            ).to(device)
            if hasattr(model, "clean_kv_cache"):
                model.clean_kv_cache()
            with torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32):
                pred = _predict_depth(model, images, args.num_scale_frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if hasattr(model, "clean_kv_cache"):
                model.clean_kv_cache()

            teacher = torch.from_numpy(sample["depth"]).float()
            if teacher.ndim == 4:
                teacher = teacher.unsqueeze(0)
            teacher = _resize_teacher_to_pred(teacher.to(device), pred)
            pred_f = pred.float()
            teacher_f = teacher.float()
            valid = torch.isfinite(pred_f) & torch.isfinite(teacher_f) & (teacher_f > 1e-4) & (pred_f > 1e-4)
            log_diff = torch.abs(torch.log(torch.clamp(pred_f[valid], min=1e-3)) - torch.log(torch.clamp(teacher_f[valid], min=1e-3)))
            abs_diff = torch.abs(pred_f[valid] - teacher_f[valid])
            abs_rel = abs_diff / torch.clamp(torch.abs(teacher_f[valid]), min=1e-3)
            ratio = pred_f[valid] / torch.clamp(teacher_f[valid], min=1e-3)
            row = {
                "index": index,
                "summary_json": sample["summary_json"],
                "frame_start": Path(sample["image_paths"][0]).name,
                "frame_end": Path(sample["image_paths"][-1]).name,
                "log_mae": float(log_diff.mean().detach().cpu()),
                "log_rmse": float(torch.sqrt((log_diff * log_diff).mean()).detach().cpu()),
                "abs_rel": float(abs_rel.mean().detach().cpu()),
                "median_scale_ratio": float(torch.median(ratio).detach().cpu()),
                "bad_10pct": float((log_diff > np.log(1.10)).float().mean().detach().cpu()),
                "valid_pixels": int(valid.sum().detach().cpu()),
            }
            rows.append(row)
            if index < args.preview_count:
                teacher_np = teacher_f[0, 0].detach().cpu().numpy()
                pred_np = pred_f[0, 0].detach().cpu().numpy()
                rgb = sample["images"][0] if sample.get("images") is not None else None
                _save_preview(args.output_dir / "previews" / f"window_{index:06d}.png", rgb, teacher_np, pred_np)
            print(json.dumps(row), flush=True)

    report = {
        "checkpoint": str(args.checkpoint),
        "teacher_windows_dir": str(args.teacher_windows_dir),
        "window_count": len(rows),
        "image_size": args.image_size,
        "model_image_size": args.model_image_size,
        "patch_embed": args.patch_embed,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "metrics": {
            "log_mae": _stats([row["log_mae"] for row in rows]),
            "log_rmse": _stats([row["log_rmse"] for row in rows]),
            "abs_rel": _stats([row["abs_rel"] for row in rows]),
            "median_scale_ratio": _stats([row["median_scale_ratio"] for row in rows]),
            "bad_10pct": _stats([row["bad_10pct"] for row in rows]),
        },
        "rows": rows,
    }
    report_path = args.output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_preview_index(args.output_dir, args.preview_count)
    print(json.dumps(report["metrics"], indent=2))
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
