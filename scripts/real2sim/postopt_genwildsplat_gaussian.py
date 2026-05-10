#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from plyfile import PlyData, PlyElement

from gsplat import rasterization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-optimize an aligned GenWildSplat Gaussian with LingBot RGB frames and poses."
    )
    parser.add_argument("--input-ply", type=Path, required=True)
    parser.add_argument("--output-ply", type=Path, required=True)
    parser.add_argument("--predictions-npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-frame-count", type=int, default=64)
    parser.add_argument("--render-scale", type=float, default=1.0)
    parser.add_argument("--max-splats", type=int, default=0, help="Debug option; 0 keeps all splats.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr-sh", type=float, default=5e-3)
    parser.add_argument("--lr-opacity", type=float, default=2e-3)
    parser.add_argument("--lr-scale", type=float, default=5e-4)
    parser.add_argument("--lr-mean", type=float, default=0.0)
    parser.add_argument("--opacity-reg", type=float, default=1e-4)
    parser.add_argument("--scale-reg", type=float, default=1e-4)
    parser.add_argument("--packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def import_playback_module() -> Any:
    script_path = Path(__file__).with_name("run_video_real2sim_playback_webui.py")
    spec = importlib.util.spec_from_file_location("lingbot_playback_webui", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_timeline(args: argparse.Namespace) -> Any:
    module = import_playback_module()
    return module.LingBotPointTimeline(
        predictions_npz=args.predictions_npz,
        summary_json=args.summary_json,
        points_per_frame=1024,
        geometry_source="depth",
        conf_percentile=65.0,
        normalize_world=True,
        precompute_clouds=False,
    )


def normalized_c2w(timeline: Any, frame_index: int) -> np.ndarray:
    c2w = timeline.camera_c2w(frame_index).astype(np.float32)
    c2w[:3, 3] = (c2w[:3, 3] - timeline.center.astype(np.float32)) * float(timeline.scale)
    return c2w


def load_ply(path: Path, max_splats: int, seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    vertex = PlyData.read(path)["vertex"].data
    count = len(vertex)
    if max_splats > 0 and count > max_splats:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(count, size=max_splats, replace=False))
    else:
        keep = np.arange(count)
    fields = {name: np.asarray(vertex[name])[keep].copy() for name in vertex.dtype.names}
    return keep, fields


def resize_rgb(rgb: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    new_size = (max(2, int(round(w * scale))), max(2, int(round(h * scale))))
    return np.asarray(Image.fromarray(rgb).resize(new_size, Image.BILINEAR), dtype=np.uint8)


def frame_tensors(timeline: Any, frame_indices: np.ndarray, render_scale: float, device: str) -> list[dict[str, torch.Tensor]]:
    frames: list[dict[str, torch.Tensor]] = []
    for index in frame_indices:
        depth_shape = timeline.depth_shape(int(index))
        if depth_shape is None:
            raise RuntimeError("LingBot predictions need depth shape for post-opt RGB preprocessing.")
        rgb = timeline._rgb_for_frame(int(index), depth_shape)
        rgb = resize_rgb(rgb, render_scale)
        image = torch.from_numpy(rgb.astype(np.float32) / 255.0).to(device)
        K = timeline.intrinsic(int(index))
        if K is None:
            raise RuntimeError("LingBot predictions need intrinsics for post-opt.")
        K = K.astype(np.float32).copy()
        K[0, :] *= render_scale
        K[1, :] *= render_scale
        c2w = normalized_c2w(timeline, int(index))
        frames.append(
            {
                "index": torch.tensor(int(index), device=device),
                "image": image,
                "K": torch.from_numpy(K).to(device),
                "c2w": torch.from_numpy(c2w).to(device),
            }
        )
    return frames


def export_ply(path: Path, template: dict[str, np.ndarray], params: dict[str, torch.Tensor], keep: np.ndarray, total_count: int) -> None:
    if len(keep) != total_count:
        # Debug/subset output intentionally contains only optimized splats.
        names = list(template.keys())
        output = {name: template[name].copy() for name in names}
    else:
        names = list(template.keys())
        output = {name: template[name].copy() for name in names}

    output["x"] = params["means"][:, 0].detach().cpu().numpy().astype(np.float32)
    output["y"] = params["means"][:, 1].detach().cpu().numpy().astype(np.float32)
    output["z"] = params["means"][:, 2].detach().cpu().numpy().astype(np.float32)
    sh0 = params["sh0"].detach().cpu().numpy().astype(np.float32)
    output["f_dc_0"] = sh0[:, 0]
    output["f_dc_1"] = sh0[:, 1]
    output["f_dc_2"] = sh0[:, 2]
    output["opacity"] = torch.sigmoid(params["opacity"]).detach().cpu().numpy().astype(np.float32)
    scales = params["scales"].detach().cpu().numpy().astype(np.float32)
    output["scale_0"] = scales[:, 0]
    output["scale_1"] = scales[:, 1]
    output["scale_2"] = scales[:, 2]
    quats = F.normalize(params["quats"], dim=-1).detach().cpu().numpy().astype(np.float32)
    output["rot_0"] = quats[:, 0]
    output["rot_1"] = quats[:, 1]
    output["rot_2"] = quats[:, 2]
    output["rot_3"] = quats[:, 3]

    dtype = [(name, "f4") for name in names]
    vertex = np.empty(len(output["x"]), dtype=dtype)
    for name in names:
        vertex[name] = output[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Post-opt requires CUDA for gsplat rasterization.")

    keep, fields = load_ply(args.input_ply, args.max_splats, args.seed)
    total_input_count = len(PlyData.read(args.input_ply)["vertex"].data)
    means = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float32)
    scales = np.stack([fields["scale_0"], fields["scale_1"], fields["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([fields["rot_0"], fields["rot_1"], fields["rot_2"], fields["rot_3"]], axis=1).astype(np.float32)
    sh0 = np.stack([fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]], axis=1).astype(np.float32)
    opacity = np.clip(fields["opacity"].astype(np.float32), 1e-4, 1.0 - 1e-4)

    params = {
        "means": torch.nn.Parameter(torch.from_numpy(means).to(device), requires_grad=args.lr_mean > 0),
        "scales": torch.nn.Parameter(torch.from_numpy(scales).to(device), requires_grad=args.lr_scale > 0),
        "quats": torch.nn.Parameter(torch.from_numpy(quats).to(device), requires_grad=False),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh0).to(device), requires_grad=args.lr_sh > 0),
        "opacity": torch.nn.Parameter(torch.logit(torch.from_numpy(opacity).to(device)), requires_grad=args.lr_opacity > 0),
    }
    optim_params = []
    if args.lr_mean > 0:
        optim_params.append({"params": [params["means"]], "lr": args.lr_mean})
    if args.lr_scale > 0:
        optim_params.append({"params": [params["scales"]], "lr": args.lr_scale})
    if args.lr_sh > 0:
        optim_params.append({"params": [params["sh0"]], "lr": args.lr_sh})
    if args.lr_opacity > 0:
        optim_params.append({"params": [params["opacity"]], "lr": args.lr_opacity})
    optimizer = torch.optim.Adam(optim_params, eps=1e-15)

    timeline = load_timeline(args)
    frame_count = min(timeline.frame_count, len(timeline.image_paths) if timeline.image_paths else timeline.frame_count)
    train_count = min(max(1, int(args.train_frame_count)), frame_count)
    train_indices = np.linspace(0, frame_count - 1, train_count).round().astype(np.int64)
    frames = frame_tensors(timeline, train_indices, float(args.render_scale), device)
    rng = np.random.default_rng(args.seed)
    print(
        json.dumps(
            {
                "input_ply": str(args.input_ply),
                "output_ply": str(args.output_ply),
                "input_splats": int(total_input_count),
                "optimized_splats": int(len(keep)),
                "train_frames": train_indices.tolist(),
                "steps": int(args.steps),
                "render_scale": float(args.render_scale),
            },
            indent=2,
        ),
        flush=True,
    )

    for step in range(int(args.steps)):
        sample = frames[int(rng.integers(0, len(frames)))]
        gt = sample["image"].unsqueeze(0)
        height, width = gt.shape[1:3]
        render, _alpha, _info = rasterization(
            means=params["means"],
            quats=F.normalize(params["quats"], dim=-1),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacity"]),
            colors=params["sh0"].unsqueeze(1),
            viewmats=torch.linalg.inv(sample["c2w"]).unsqueeze(0),
            Ks=sample["K"].unsqueeze(0),
            width=int(width),
            height=int(height),
            packed=bool(args.packed),
            sh_degree=0,
            render_mode="RGB",
            radius_clip=float(args.radius_clip),
        )
        loss_l1 = F.l1_loss(render[..., :3], gt)
        loss_mse = F.mse_loss(render[..., :3], gt)
        loss = 0.8 * loss_l1 + 0.2 * loss_mse
        if args.opacity_reg > 0:
            loss = loss + float(args.opacity_reg) * torch.sigmoid(params["opacity"]).mean()
        if args.scale_reg > 0:
            loss = loss + float(args.scale_reg) * torch.exp(params["scales"]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            print(
                f"step {step + 1}/{args.steps} frame {int(sample['index'].item())} "
                f"loss={loss.item():.5f} l1={loss_l1.item():.5f}",
                flush=True,
            )

    export_ply(args.output_ply, fields, params, keep, total_input_count if args.max_splats <= 0 else len(keep))
    print(f"wrote {args.output_ply}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
