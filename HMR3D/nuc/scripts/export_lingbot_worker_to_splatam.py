#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np


DEPTH_PNG_SCALE = 6553.5


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth


def _opencv_c2w_to_nerfcapture(c2w: np.ndarray) -> np.ndarray:
    # SplaTAM's NeRFCapture loader applies P @ c2w @ P.T internally. Store the
    # inverse conversion so the pose used by SplaTAM matches our OpenCV c2w pose.
    p = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return (p @ c2w @ p.T).astype(np.float32)


def _scale_intrinsic(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    sx = float(dst_w) / max(float(src_w), 1.0)
    sy = float(dst_h) / max(float(src_h), 1.0)
    return fx * sx, fy * sy, cx * sx, cy * sy


def _iter_worker_results(worker_dir: Path) -> list[Path]:
    return sorted(worker_dir.glob("window_*/worker_result.json"))


def _write_splatam_config(
    path: Path,
    *,
    basedir: Path,
    sequence: str,
    width: int,
    height: int,
    num_frames: int,
    mapping_iters: int,
) -> None:
    config = f'''primary_device = "cuda:0"
seed = 0

config = dict(
    workdir="{basedir / sequence}",
    run_name="SplaTAM_LingBot",
    seed=seed,
    primary_device=primary_device,
    map_every=1,
    keyframe_every=max(1, {num_frames} // 5),
    mapping_window_size=min(24, {num_frames}),
    report_global_progress_every=5,
    eval_every=1,
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    gaussian_distribution="isotropic",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=5,
    use_wandb=False,
    data=dict(
        dataset_name="nerfcapture",
        basedir="{basedir}",
        sequence="{sequence}",
        desired_image_height={height},
        desired_image_width={width},
        desired_image_height_init={height},
        desired_image_width_init={width},
        start=0,
        end=-1,
        stride=1,
        num_frames={num_frames},
        eval_stride=1,
        eval_num_frames=-1,
    ),
    train=dict(
        num_iters_mapping={mapping_iters},
        sil_thres=0.5,
        use_sil_for_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs_mapping=dict(
            means3D=0.00016,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.0025,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),
        lrs_mapping_means3D_final=0.0000016,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=3000,
        ),
    ),
    viz=dict(
        render_mode="color",
        offset_first_viz_cam=True,
        show_sil=False,
        visualize_cams=True,
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=2,
        viz_fps=5,
        enter_interactive_post_online=False,
    ),
)
'''
    path.write_text(config, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert LingBot depth-worker window outputs to SplaTAM NeRFCapture RGB-D format."
    )
    parser.add_argument("--worker-dir", required=True, help="Directory containing window_*/worker_result.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", default="lingbot_live")
    parser.add_argument("--depth-scale", type=float, default=20.0, help="Scale LingBot relative depth to metric-ish meters.")
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--pose-convention", choices=("opencv_c2w", "nerfcapture"), default="opencv_c2w")
    parser.add_argument("--mapping-iters", type=int, default=300)
    parser.add_argument(
        "--allow-identity-poses",
        action="store_true",
        help="Smoke-test fallback when worker metadata has no external poses. Do not use for real reconstruction.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    worker_dir = Path(args.worker_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    sequence_dir = output_root / args.sequence
    rgb_dir = sequence_dir / "rgb"
    depth_dir = sequence_dir / "depth"
    if sequence_dir.exists() and args.overwrite:
        shutil.rmtree(sequence_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[int, dict[str, Any]] = {}
    dst_shape: tuple[int, int] | None = None
    src_shape: tuple[int, int] | None = None

    for result_path in _iter_worker_results(worker_dir):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
        pred = np.load(result["predictions_npz"])
        depth = _squeeze_depth(pred["depth"])
        frames = summary.get("metadata", {}).get("frames", [])
        for local_idx, frame in enumerate(frames[: depth.shape[0]]):
            frame_idx = int(frame.get("frame_idx", local_idx))
            if frame_idx in exported:
                continue
            meta = frame.get("metadata") or {}
            if "pose" not in meta and not args.allow_identity_poses:
                continue
            image_path = Path(frame.get("image_path", "")).expanduser()
            rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if rgb is None:
                continue
            d = depth[local_idx].astype(np.float32) * float(args.depth_scale)
            h, w = d.shape
            dst_shape = (h, w)
            src_shape = rgb.shape[:2]
            rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)

            out_name = f"{len(exported):06d}.png"
            cv2.imwrite(str(rgb_dir / out_name), rgb_resized)
            d_png = np.clip(np.rint(d * DEPTH_PNG_SCALE), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            cv2.imwrite(str(depth_dir / out_name), d_png)

            pose = np.asarray(meta.get("pose", np.eye(4, dtype=np.float32)), dtype=np.float32)
            if pose.shape == (3, 4):
                pose4 = np.eye(4, dtype=np.float32)
                pose4[:3, :4] = pose
                pose = pose4
            if args.pose_convention == "opencv_c2w":
                pose = _opencv_c2w_to_nerfcapture(pose)

            exported[frame_idx] = {
                "file_path": f"rgb/{out_name}",
                "transform_matrix": pose.astype(float).tolist(),
                "source_frame_idx": frame_idx,
                "source_image_path": str(image_path),
            }

    if not exported or dst_shape is None or src_shape is None:
        raise SystemExit("No frames with depth + RGB + pose were exported.")

    h, w = dst_shape
    fx = float(args.fx or src_shape[1])
    fy = float(args.fy or src_shape[1])
    cx = float(args.cx or src_shape[1] / 2.0)
    cy = float(args.cy or src_shape[0] / 2.0)
    fx, fy, cx, cy = _scale_intrinsic(fx, fy, cx, cy, src_shape, dst_shape)
    frames = [exported[idx] for idx in sorted(exported)]
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "frames": frames,
        "depth_png_scale": DEPTH_PNG_SCALE,
        "lingbot_depth_scale": float(args.depth_scale),
    }
    (sequence_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    config_path = output_root / f"{args.sequence}_splatam_gaussian_splatting.py"
    _write_splatam_config(
        config_path,
        basedir=output_root,
        sequence=args.sequence,
        width=w,
        height=h,
        num_frames=len(frames),
        mapping_iters=args.mapping_iters,
    )
    print(f"Exported {len(frames)} frames to {sequence_dir}")
    print(f"SplaTAM config: {config_path}")
    print("Run:")
    print(f"  cd third_party_research/SplaTAM && python scripts/gaussian_splatting.py {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
