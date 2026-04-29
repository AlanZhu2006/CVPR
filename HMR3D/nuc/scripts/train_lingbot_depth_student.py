#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

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


def _load_teacher_window_samples(root: Path) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
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
        import numpy as np

        npz = np.load(pred_path)
        if "depth" not in npz:
            continue
        depth = torch.from_numpy(np.asarray(npz["depth"])).float()
        samples.append(
            {
                "summary_json": str(summary_path),
                "predictions_npz": str(pred_path),
                "image_paths": image_paths,
                "depth": depth,
            }
        )
    if not samples:
        raise RuntimeError(f"No teacher windows found under {root}")
    return samples


def _build_model(args: argparse.Namespace, *, student: bool):
    from lingbot_map.models.gct_stream import GCTStream

    if student:
        embed_dim = args.student_embed_dim
        patch_embed = args.student_patch_embed
        img_size = args.student_model_image_size or args.model_image_size
        depth = args.student_depth
        num_heads = args.student_num_heads
        mlp_ratio = args.student_mlp_ratio
    else:
        embed_dim = 1024
        patch_embed = args.teacher_patch_embed
        img_size = args.model_image_size
        depth = 24
        num_heads = 16
        mlp_ratio = 4.0

    return GCTStream(
        img_size=img_size,
        patch_size=14,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        patch_embed=patch_embed,
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


def _window(paths: list[str], start: int, size: int) -> list[str]:
    if len(paths) < size:
        raise RuntimeError(f"Need at least {size} images, got {len(paths)}")
    start = start % (len(paths) - size + 1)
    return paths[start : start + size]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a smaller LingBot depth model from teacher or cached teacher depth."
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--image-dir")
    source.add_argument("--image-list")
    parser.add_argument("--image-root", default="")
    parser.add_argument("--glob", default="*")
    parser.add_argument("--lingbot-map-root", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--teacher-predictions-npz", type=Path, default=None)
    parser.add_argument(
        "--teacher-windows-dir",
        type=Path,
        default=None,
        help="Directory containing window_*/lingbot_predictions.npz and summaries from the live worker.",
    )
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--teacher-patch-embed", default="dinov2_vitl14_reg")
    parser.add_argument("--student-patch-embed", default="dinov2_vits14_reg")
    parser.add_argument("--student-embed-dim", type=int, default=384)
    parser.add_argument("--student-depth", type=int, default=8)
    parser.add_argument("--student-num-heads", type=int, default=6)
    parser.add_argument("--student-mlp-ratio", type=float, default=3.0)
    parser.add_argument("--student-model-image-size", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--teacher-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--student-dtype", choices=("fp16", "bf16", "fp32"), default="fp32")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _add_paths(args.lingbot_map_root)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    teacher_dtype = dtype_map[args.teacher_dtype]
    student_dtype = dtype_map[args.student_dtype]

    teacher_samples = _load_teacher_window_samples(args.teacher_windows_dir) if args.teacher_windows_dir else []
    image_paths = _load_image_paths(args) if (args.image_dir or args.image_list) else []
    if teacher_samples:
        unique_images = []
        seen = set()
        for sample in teacher_samples:
            for path in sample["image_paths"]:
                if path not in seen:
                    seen.add(path)
                    unique_images.append(path)
        image_paths = unique_images
    if not image_paths:
        raise RuntimeError("No images found; provide --image-dir/--image-list or --teacher-windows-dir")
    if args.teacher_checkpoint is None and args.teacher_predictions_npz is None and not teacher_samples:
        raise RuntimeError("Provide --teacher-checkpoint, --teacher-predictions-npz, or --teacher-windows-dir")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "image_count": len(image_paths),
        "window_size": args.window_size,
        "student_patch_embed": args.student_patch_embed,
        "student_embed_dim": args.student_embed_dim,
        "student_depth": args.student_depth,
        "student_num_heads": args.student_num_heads,
        "student_mlp_ratio": args.student_mlp_ratio,
        "steps": args.steps,
        "losses": [],
    }

    teacher = None
    cached_depth = None
    if teacher_samples:
        report["teacher_source"] = str(args.teacher_windows_dir)
        report["teacher_window_count"] = len(teacher_samples)
    elif args.teacher_predictions_npz is not None:
        cached = dict(torch.load(args.teacher_predictions_npz, map_location="cpu")) if args.teacher_predictions_npz.suffix == ".pt" else None
        if cached is None:
            import numpy as np

            npz = np.load(args.teacher_predictions_npz)
            cached_depth = torch.from_numpy(np.asarray(npz["depth"])).float()
        else:
            cached_depth = cached["depth"].float()
        report["teacher_source"] = str(args.teacher_predictions_npz)
    else:
        teacher = _build_model(args, student=False)
        state = _load_state(args.teacher_checkpoint.expanduser().resolve())
        missing, unexpected = teacher.load_state_dict(state, strict=False)
        report["teacher_missing_keys"] = len(missing)
        report["teacher_unexpected_keys"] = len(unexpected)
        if teacher_dtype != torch.float32:
            teacher.aggregator = teacher.aggregator.to(dtype=teacher_dtype)
        teacher = teacher.to(device).eval()
        report["teacher_source"] = str(args.teacher_checkpoint)

    student = _build_model(args, student=True)
    if args.student_checkpoint is not None:
        state = _load_state(args.student_checkpoint.expanduser().resolve())
        missing, unexpected = student.load_state_dict(state, strict=False)
        report["student_missing_keys"] = len(missing)
        report["student_unexpected_keys"] = len(unexpected)
    if student_dtype != torch.float32:
        student.aggregator = student.aggregator.to(dtype=student_dtype)
    student = student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    if args.dry_run:
        report["dry_run"] = True
        (args.output_dir / "distill_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 0

    start_time = time.perf_counter()
    for step in range(args.steps):
        if teacher_samples:
            sample = teacher_samples[step % len(teacher_samples)]
            paths = list(sample["image_paths"])
            target = sample["depth"]
            if target.ndim == 4:
                target = target.unsqueeze(0)
            target = target.to(device)
        else:
            window_start = step % (len(image_paths) - args.window_size + 1)
            paths = _window(image_paths, window_start, args.window_size)
        images = load_and_preprocess_images(
            paths,
            mode="crop",
            image_size=args.image_size,
            patch_size=14,
        ).to(device)

        if teacher_samples:
            pass
        elif cached_depth is not None:
            target = cached_depth[window_start : window_start + args.window_size]
            if target.shape[0] != args.window_size:
                target = cached_depth[: args.window_size]
            if target.ndim == 4:
                target = target.unsqueeze(0)
            target = target.to(device)
        else:
            with torch.no_grad():
                if hasattr(teacher, "clean_kv_cache"):
                    teacher.clean_kv_cache()
                with torch.amp.autocast("cuda", dtype=teacher_dtype, enabled=device.type == "cuda" and teacher_dtype != torch.float32):
                    target = _predict_depth(teacher, images, args.num_scale_frames).detach()
                if hasattr(teacher, "clean_kv_cache"):
                    teacher.clean_kv_cache()

        optimizer.zero_grad(set_to_none=True)
        if hasattr(student, "clean_kv_cache"):
            student.clean_kv_cache()
        with torch.amp.autocast("cuda", dtype=student_dtype, enabled=device.type == "cuda" and student_dtype != torch.float32):
            pred = _predict_depth(student, images, args.num_scale_frames)
            target_resized = target
            if pred.shape != target.shape:
                pred_nchw = pred[..., 0].reshape(-1, pred.shape[-3], pred.shape[-2]).unsqueeze(1)
                target_nchw = target[..., 0].reshape(-1, target.shape[-3], target.shape[-2]).unsqueeze(1)
                target_nchw = F.interpolate(target_nchw, size=pred_nchw.shape[-2:], mode="bilinear", align_corners=False)
                target_resized = target_nchw.squeeze(1).reshape(pred.shape[:-1]).unsqueeze(-1)
            pred_log = torch.log(torch.clamp(pred.float(), min=1e-3))
            target_log = torch.log(torch.clamp(target_resized.float(), min=1e-3))
            loss = F.smooth_l1_loss(pred_log, target_log)
        loss.backward()
        optimizer.step()
        if hasattr(student, "clean_kv_cache"):
            student.clean_kv_cache()

        loss_value = float(loss.detach().cpu())
        report["losses"].append(loss_value)
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(json.dumps({"step": step, "loss": loss_value}, ensure_ascii=False), flush=True)

    report["elapsed_sec"] = time.perf_counter() - start_time
    checkpoint_path = args.output_dir / "lingbot_depth_student.pt"
    torch.save({"model": student.state_dict(), "report": report}, checkpoint_path)
    report["checkpoint"] = str(checkpoint_path)
    (args.output_dir / "distill_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
