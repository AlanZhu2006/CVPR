#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "nuc" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nuc_runtime import (
    CUVSLAMMonocularRGBAdapter,
    CUVSLAMOfflineKITTIAdapter,
    LingBotDepthWorker,
    LingBotDepthWorkerConfig,
    RGBMonocularVOAdapter,
    VoxelFusionMap,
    empty_point_batch,
    load_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate live cuVSLAM tracking + LingBot background dense-depth mapping "
            "and continuously publish a browser-viewable local 3D reconstruction."
        )
    )
    parser.add_argument("--sequence-dir", default="nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_input")
    parser.add_argument("--trajectory-path", default="nuc_output/kitti_raw_2011_09_30_0020_benchmark/cuvslam_tum.txt")
    parser.add_argument(
        "--tracking-backend",
        choices=("pose_file", "opencv_mono_rgb", "cuvslam_mono_rgb"),
        default="pose_file",
        help=(
            "pose_file replays an existing TUM trajectory; opencv_mono_rgb is a small OpenCV baseline; "
            "cuvslam_mono_rgb runs PyCuVSLAM OdometryMode.Mono on the RGB stream."
        ),
    )
    parser.add_argument(
        "--rgb-image-dir",
        default="",
        help="RGB image stream for monocular tracking and LingBot input. If empty, uses --color-image-dir for opencv_mono_rgb.",
    )
    parser.add_argument("--intrinsic-camera-index", type=int, default=0)
    parser.add_argument("--mono-fixed-step-scale", type=float, default=0.5)
    parser.add_argument("--mono-scale-source", choices=("fixed", "oxts"), default="fixed")
    parser.add_argument("--oxts-dir", default="")
    parser.add_argument(
        "--color-image-dir",
        default="",
        help="Optional RGB image directory used only for coloring fused points. Tracking/LingBot still use sequence-dir images.",
    )
    parser.add_argument(
        "--color-image-template",
        default="{frame_idx:010d}.png",
        help="Filename template under --color-image-dir. Use {frame_idx:06d}.png for cuVSLAM-style image folders.",
    )
    parser.add_argument("--config", default="")
    parser.add_argument("--model-path", default="third_party_research/lingbot_cache/lingbot-map-depth-fp16.pt")
    parser.add_argument("--lingbot-map-root", default="third_party_research/lingbot-map")
    parser.add_argument("--output-dir", default="nuc_output/lingbot_live_reconstruction/kitti0020_live")
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--keyframes-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-image-size", type=int, default=518)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--dense-frame-interval",
        type=int,
        default=1,
        help="Submit only every Nth tracked frame to the LingBot dense worker while keeping every cuVSLAM pose.",
    )
    parser.add_argument("--num-scale-frames", type=int, default=2)
    parser.add_argument("--max-queue", type=int, default=2)
    parser.add_argument("--depth-scale", type=float, default=20.0)
    parser.add_argument(
        "--lingbot-pose-translation-scale",
        type=float,
        default=0.0,
        help="Scale LingBot-predicted camera translations before fusing depth. 0 follows --depth-scale.",
    )
    parser.add_argument(
        "--lingbot-extrinsic-mode",
        choices=("inverse", "direct"),
        default="inverse",
        help=(
            "How to interpret LingBot predictions.extrinsic when no external pose is available. "
            "inverse matches the official depth unprojection path where extrinsic is world-to-camera."
        ),
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--max-points-per-frame", type=int, default=2500)
    parser.add_argument("--max-active-frames", type=int, default=16)
    parser.add_argument("--fusion-mode", choices=("raw", "voxel"), default="raw")
    parser.add_argument("--voxel-size", type=float, default=0.08)
    parser.add_argument("--fusion-max-points", type=int, default=500000)
    parser.add_argument("--fusion-min-observations", type=int, default=1)
    parser.add_argument("--adaptive-sampling", action="store_true")
    parser.add_argument("--near-depth-m", type=float, default=18.0)
    parser.add_argument("--near-sample-stride", type=int, default=1)
    parser.add_argument("--edge-sample-stride", type=int, default=2)
    parser.add_argument("--edge-percentile", type=float, default=88.0)
    parser.add_argument("--semantic-sample-stride", type=int, default=1)
    parser.add_argument("--yolo-model", default="")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--semantic-color-output", action="store_true")
    parser.add_argument("--depth-head-trt-engine", default="")
    parser.add_argument("--model-patch-embed", default="")
    parser.add_argument("--model-embed-dim", type=int, default=0)
    parser.add_argument("--model-depth", type=int, default=0)
    parser.add_argument("--model-num-heads", type=int, default=0)
    parser.add_argument("--model-mlp-ratio", type=float, default=0.0)
    parser.add_argument("--no-compress-output", action="store_true")
    parser.add_argument("--frame-sleep-sec", type=float, default=0.0)
    parser.add_argument("--poll-sec", type=float, default=0.1)
    parser.add_argument("--publish-every-windows", type=int, default=1)
    parser.add_argument("--publish-every-frames", type=int, default=1)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=19092)
    parser.add_argument("--no-blocking-submit", action="store_true")
    parser.add_argument(
        "--drop-when-busy",
        action="store_true",
        help="Alias for --no-blocking-submit: keep tracking realtime and drop dense windows if the worker queue is full.",
    )
    return parser.parse_args()


def _load_kitti_intrinsic(sequence_dir: Path, camera_index: int = 0) -> np.ndarray:
    calib = np.loadtxt(sequence_dir / "calib.txt", usecols=range(1, 13))[:4].reshape(4, 3, 4)
    idx = min(max(int(camera_index), 0), calib.shape[0] - 1)
    return calib[idx, :3, :3].astype(np.float32)


def _scaled_intrinsic(K: np.ndarray, original_shape_hw: tuple[int, int], depth_shape_hw: tuple[int, int]) -> np.ndarray:
    src_h, src_w = original_shape_hw
    dst_h, dst_w = depth_shape_hw
    scaled = K.copy().astype(np.float32)
    scaled[0, :] *= float(dst_w) / max(float(src_w), 1.0)
    scaled[1, :] *= float(dst_h) / max(float(src_h), 1.0)
    return scaled


def _load_rgb(path: str) -> np.ndarray | None:
    frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None
    return frame_bgr[:, :, ::-1].astype(np.uint8)


SEMANTIC_PALETTE = np.array(
    [
        [220, 20, 60],
        [0, 168, 132],
        [255, 170, 0],
        [86, 156, 214],
        [178, 102, 255],
        [45, 212, 191],
        [250, 204, 21],
        [244, 114, 182],
        [132, 204, 22],
        [251, 146, 60],
        [96, 165, 250],
        [167, 139, 250],
    ],
    dtype=np.uint8,
)


class YOLOSemanticProjector:
    def __init__(self, model_path: str, *, conf: float, imgsz: int) -> None:
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.enabled = False
        self.error = ""
        self._cache: dict[tuple[str, tuple[int, int]], tuple[np.ndarray, np.ndarray]] = {}
        if not self.model_path:
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(self.model_path)
            self.enabled = True
        except Exception as exc:  # pragma: no cover - depends on optional package.
            self.model = None
            self.error = f"YOLO disabled: {type(exc).__name__}: {exc}"

    def project(
        self,
        image_path: str,
        rgb: np.ndarray | None,
        depth_shape_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros(depth_shape_hw, dtype=np.int32) - 1
        scores = np.zeros(depth_shape_hw, dtype=np.float32)
        if not self.enabled or rgb is None:
            return labels, scores
        key = (str(image_path), (int(depth_shape_hw[0]), int(depth_shape_hw[1])))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        try:
            results = self.model.predict(
                source=image_path,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - optional runtime path.
            self.error = f"YOLO inference failed: {type(exc).__name__}: {exc}"
            return labels, scores
        if not results:
            self._cache[key] = (labels, scores)
            return labels, scores

        image_h, image_w = rgb.shape[:2]
        depth_h, depth_w = depth_shape_hw
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            self._cache[key] = (labels, scores)
            return labels, scores

        xyxy = boxes.xyxy.detach().cpu().numpy() if getattr(boxes, "xyxy", None) is not None else np.zeros((0, 4))
        cls = boxes.cls.detach().cpu().numpy().astype(np.int32) if getattr(boxes, "cls", None) is not None else np.zeros((0,), dtype=np.int32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32) if getattr(boxes, "conf", None) is not None else np.zeros((0,), dtype=np.float32)
        order = np.argsort(conf)
        for det_idx in order:
            x0, y0, x1, y1 = xyxy[det_idx]
            dx0 = int(np.clip(np.floor(x0 * depth_w / max(image_w, 1)), 0, depth_w - 1))
            dx1 = int(np.clip(np.ceil(x1 * depth_w / max(image_w, 1)), 0, depth_w))
            dy0 = int(np.clip(np.floor(y0 * depth_h / max(image_h, 1)), 0, depth_h - 1))
            dy1 = int(np.clip(np.ceil(y1 * depth_h / max(image_h, 1)), 0, depth_h))
            if dx1 <= dx0 or dy1 <= dy0:
                continue
            score = float(conf[det_idx])
            update = scores[dy0:dy1, dx0:dx1] <= score
            patch_labels = labels[dy0:dy1, dx0:dx1]
            patch_scores = scores[dy0:dy1, dx0:dx1]
            patch_labels[update] = int(cls[det_idx])
            patch_scores[update] = score
        self._cache[key] = (labels, scores)
        return labels, scores


def _semantic_colors(labels: np.ndarray, fallback_rgb: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    out = np.asarray(fallback_rgb, dtype=np.uint8).copy()
    valid = labels >= 0
    if valid.any():
        out[valid] = SEMANTIC_PALETTE[labels[valid] % len(SEMANTIC_PALETTE)]
    return out


def _resolve_color_image_path(args: argparse.Namespace, frame_idx: int) -> str:
    if not args.color_image_dir:
        return ""
    color_dir = Path(args.color_image_dir).expanduser()
    try:
        name = args.color_image_template.format(frame_idx=int(frame_idx))
    except Exception:
        name = f"{int(frame_idx):010d}.png"
    candidate = color_dir / name
    if candidate.exists():
        return str(candidate)
    for fallback_name in (
        f"{int(frame_idx):06d}.png",
        f"{int(frame_idx):010d}.png",
        f"{int(frame_idx)}.png",
    ):
        fallback = color_dir / fallback_name
        if fallback.exists():
            return str(fallback)
    return str(candidate)


def _sample_rgb(rgb: np.ndarray | None, xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb is None:
        return np.full((xs.shape[0], 3), 180, dtype=np.uint8)
    sample_x = np.clip((xs.astype(np.float32) + 0.5) * rgb.shape[1] / max(width, 1), 0, rgb.shape[1] - 1)
    sample_y = np.clip((ys.astype(np.float32) + 0.5) * rgb.shape[0] / max(height, 1), 0, rgb.shape[0] - 1)
    return rgb[sample_y.astype(np.int32), sample_x.astype(np.int32)].astype(np.uint8)


def _backproject(K: np.ndarray, pose: np.ndarray, xs: np.ndarray, ys: np.ndarray, depth: np.ndarray) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    z = depth.astype(np.float32)
    x_cam = (xs.astype(np.float32) - cx) * z / max(fx, 1e-6)
    y_cam = (ys.astype(np.float32) - cy) * z / max(fy, 1e-6)
    camera = np.stack([x_cam, y_cam, z], axis=1).astype(np.float32)
    return (camera @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def _lingbot_extrinsic_to_pose(
    extrinsic: np.ndarray,
    translation_scale: float = 1.0,
    mode: str = "inverse",
) -> np.ndarray | None:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (3, 4):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :4] = extrinsic
    elif extrinsic.shape == (4, 4):
        matrix = extrinsic.astype(np.float32).copy()
    else:
        return None
    if mode == "inverse":
        pose = np.linalg.inv(matrix).astype(np.float32)
    elif mode == "direct":
        pose = matrix.astype(np.float32).copy()
    else:
        raise ValueError(f"Unknown LingBot extrinsic mode: {mode}")
    pose[:3, 3] *= float(translation_scale)
    return pose


def _grid_pixels(height: int, width: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    ys_grid, xs_grid = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    return xs_grid.reshape(-1).astype(np.int32), ys_grid.reshape(-1).astype(np.int32)


def _edge_score(rgb: np.ndarray | None, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    if rgb is None:
        return np.zeros(depth_shape_hw, dtype=np.float32)
    gray = cv2.cvtColor(rgb[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (int(depth_shape_hw[1]), int(depth_shape_hw[0])), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy).astype(np.float32)


def _squeeze_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    return depth


def _sample_frame_points(
    *,
    depth: np.ndarray,
    conf: np.ndarray,
    rgb: np.ndarray | None,
    pose: np.ndarray,
    K: np.ndarray,
    frame_idx: int,
    args: argparse.Namespace,
    semantic_projector: YOLOSemanticProjector | None = None,
    image_path: str = "",
) -> dict[str, np.ndarray]:
    height, width = depth.shape
    stride = max(1, int(args.sample_stride))
    semantic_labels = np.zeros(depth.shape, dtype=np.int32) - 1
    semantic_scores = np.zeros(depth.shape, dtype=np.float32)
    if semantic_projector is not None:
        semantic_labels, semantic_scores = semantic_projector.project(image_path, rgb, depth.shape)

    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    xs_base, ys_base = _grid_pixels(height, width, stride)
    xs_parts.append(xs_base)
    ys_parts.append(ys_base)
    edge = np.zeros(depth.shape, dtype=np.float32)
    if getattr(args, "adaptive_sampling", False):
        metric_depth = depth.astype(np.float32) * float(args.depth_scale)
        near_stride = max(1, int(getattr(args, "near_sample_stride", stride)))
        if near_stride < stride:
            xs_near, ys_near = _grid_pixels(height, width, near_stride)
            near = metric_depth[ys_near, xs_near] <= float(getattr(args, "near_depth_m", 18.0))
            xs_parts.append(xs_near[near])
            ys_parts.append(ys_near[near])

        edge_stride = max(1, int(getattr(args, "edge_sample_stride", stride)))
        if edge_stride < stride:
            edge = _edge_score(rgb, depth.shape)
            threshold = np.percentile(edge[np.isfinite(edge)], float(getattr(args, "edge_percentile", 88.0))) if edge.size else np.inf
            xs_edge, ys_edge = _grid_pixels(height, width, edge_stride)
            keep_edge = edge[ys_edge, xs_edge] >= threshold
            xs_parts.append(xs_edge[keep_edge])
            ys_parts.append(ys_edge[keep_edge])

        semantic_stride = max(1, int(getattr(args, "semantic_sample_stride", stride)))
        if semantic_stride < stride and np.any(semantic_labels >= 0):
            xs_sem, ys_sem = _grid_pixels(height, width, semantic_stride)
            keep_sem = semantic_labels[ys_sem, xs_sem] >= 0
            xs_parts.append(xs_sem[keep_sem])
            ys_parts.append(ys_sem[keep_sem])

    xs = np.concatenate(xs_parts, axis=0) if xs_parts else np.zeros((0,), dtype=np.int32)
    ys = np.concatenate(ys_parts, axis=0) if ys_parts else np.zeros((0,), dtype=np.int32)
    if xs.size:
        linear = ys.astype(np.int64) * int(width) + xs.astype(np.int64)
        _, unique_idx = np.unique(linear, return_index=True)
        xs = xs[unique_idx]
        ys = ys[unique_idx]
    values = depth[ys, xs].astype(np.float32) * float(args.depth_scale)
    scores = conf[ys, xs].astype(np.float32)
    sem_scores = semantic_scores[ys, xs].astype(np.float32)
    sem_labels = semantic_labels[ys, xs].astype(np.int32)
    if getattr(args, "adaptive_sampling", False):
        if not edge.any():
            edge = _edge_score(rgb, depth.shape)
        local_edge = edge[ys, xs].astype(np.float32)
        if local_edge.size and np.nanmax(local_edge) > 0:
            local_edge = local_edge / max(float(np.nanmax(local_edge)), 1e-6)
        near_bonus = (values <= float(getattr(args, "near_depth_m", 18.0))).astype(np.float32) * 0.15
        scores = scores + 0.35 * sem_scores + 0.20 * local_edge + near_bonus
    valid = np.isfinite(values) & np.isfinite(scores)
    valid &= values >= float(args.min_depth)
    valid &= values <= float(args.max_depth)
    valid &= scores >= float(args.min_conf)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size > args.max_points_per_frame > 0:
        keep_local = np.argpartition(scores[valid_idx], -args.max_points_per_frame)[-args.max_points_per_frame :]
        valid_idx = valid_idx[keep_local]
    if valid_idx.size == 0:
        return empty_point_batch()
    xs_keep = xs[valid_idx]
    ys_keep = ys[valid_idx]
    depth_keep = values[valid_idx]
    xyz = _backproject(K, pose, xs_keep, ys_keep, depth_keep)
    colors = _sample_rgb(rgb, xs_keep, ys_keep, width, height)
    labels_keep = sem_labels[valid_idx]
    sem_conf_keep = sem_scores[valid_idx]
    if getattr(args, "semantic_color_output", False):
        colors = _semantic_colors(labels_keep, colors)
    return {
        "xyz": xyz,
        "rgb": colors,
        "frame": np.full((xyz.shape[0],), int(frame_idx), dtype=np.int32),
        "semantic_label": labels_keep.astype(np.int32),
        "semantic_conf": sem_conf_keep.astype(np.float32),
        "observations": np.ones((xyz.shape[0],), dtype=np.int32),
    }


def _write_ascii_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    semantic_label: np.ndarray | None = None,
    semantic_conf: np.ndarray | None = None,
    observations: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    semantic_label = semantic_label if semantic_label is not None else np.zeros((xyz.shape[0],), dtype=np.int32) - 1
    semantic_conf = semantic_conf if semantic_conf is not None else np.zeros((xyz.shape[0],), dtype=np.float32)
    observations = observations if observations is not None else np.ones((xyz.shape[0],), dtype=np.int32)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {xyz.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("property int semantic_label\nproperty float semantic_conf\nproperty int observations\n")
        handle.write("end_header\n")
        for point, color, label, sem_conf, obs in zip(xyz, rgb, semantic_label, semantic_conf, observations):
            handle.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(label)} {float(sem_conf):.4f} {int(obs)}\n"
            )


def _write_live_json(
    output_dir: Path,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    trajectory: list[dict[str, Any]],
    events: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if frame_points:
        xyz = np.concatenate([item["xyz"] for item in frame_points.values()], axis=0)
        rgb = np.concatenate([item["rgb"] for item in frame_points.values()], axis=0)
        frames = np.concatenate([item["frame"] for item in frame_points.values()], axis=0)
        semantic_label = np.concatenate(
            [item.get("semantic_label", np.full((item["xyz"].shape[0],), -1, dtype=np.int32)) for item in frame_points.values()],
            axis=0,
        )
        semantic_conf = np.concatenate(
            [item.get("semantic_conf", np.zeros((item["xyz"].shape[0],), dtype=np.float32)) for item in frame_points.values()],
            axis=0,
        )
        observations = np.concatenate(
            [item.get("observations", np.ones((item["xyz"].shape[0],), dtype=np.int32)) for item in frame_points.values()],
            axis=0,
        )
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        frames = np.zeros((0,), dtype=np.int32)
        semantic_label = np.zeros((0,), dtype=np.int32) - 1
        semantic_conf = np.zeros((0,), dtype=np.float32)
        observations = np.zeros((0,), dtype=np.int32)
    if xyz.shape[0] > 0:
        bbox_min = xyz.min(axis=0)
        bbox_max = xyz.max(axis=0)
    else:
        bbox_min = np.zeros(3, dtype=np.float32)
        bbox_max = np.ones(3, dtype=np.float32)

    max_json_points = 60000
    if xyz.shape[0] > max_json_points:
        idx = np.linspace(0, xyz.shape[0] - 1, max_json_points).astype(np.int64)
        xyz_json, rgb_json, frames_json = xyz[idx], rgb[idx], frames[idx]
        semantic_json, semantic_conf_json = semantic_label[idx], semantic_conf[idx]
    else:
        xyz_json, rgb_json, frames_json = xyz, rgb, frames
        semantic_json, semantic_conf_json = semantic_label, semantic_conf
    points = [
        [
            round(float(p[0]), 4),
            round(float(p[1]), 4),
            round(float(p[2]), 4),
            int(c[0]),
            int(c[1]),
            int(c[2]),
            int(f),
            int(label),
            round(float(sem_conf), 4),
        ]
        for p, c, f, label, sem_conf in zip(xyz_json, rgb_json, frames_json, semantic_json, semantic_conf_json)
    ]
    payload = {
        "schema": "lingbot_live_reconstruction.v1",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "point_count": int(xyz.shape[0]),
        "shown_point_count": int(xyz_json.shape[0]),
        "active_frames": list(frame_points.keys()),
        "bbox_min": bbox_min.astype(float).tolist(),
        "bbox_max": bbox_max.astype(float).tolist(),
        "trajectory": trajectory,
        "events": events[-80:],
        "settings": {
            "tracking_backend": str(args.tracking_backend),
            "rgb_image_dir": str(args.rgb_image_dir),
            "intrinsic_camera_index": int(args.intrinsic_camera_index),
            "image_size": int(args.image_size),
            "frame_step": int(args.frame_step),
            "color_image_dir": str(args.color_image_dir),
            "color_image_template": str(args.color_image_template),
            "depth_scale": float(args.depth_scale),
            "lingbot_pose_translation_scale": float(
                getattr(args, "lingbot_pose_translation_scale", 0.0) or getattr(args, "depth_scale", 1.0)
            ),
            "lingbot_extrinsic_mode": str(getattr(args, "lingbot_extrinsic_mode", "inverse")),
            "sample_stride": int(args.sample_stride),
            "max_points_per_frame": int(args.max_points_per_frame),
            "fusion_mode": str(getattr(args, "fusion_mode", "raw")),
            "voxel_size": float(getattr(args, "voxel_size", 0.0)),
            "fusion_max_points": int(getattr(args, "fusion_max_points", 0)),
            "adaptive_sampling": bool(getattr(args, "adaptive_sampling", False)),
            "yolo_model": str(getattr(args, "yolo_model", "")),
            "semantic_color_output": bool(getattr(args, "semantic_color_output", False)),
        },
        "points": points,
    }
    (output_dir / "live_map.json").write_text(json.dumps(payload), encoding="utf-8")
    np.savez_compressed(
        output_dir / "live_map.npz",
        xyz=xyz,
        rgb=rgb,
        frame=frames,
        semantic_label=semantic_label,
        semantic_conf=semantic_conf,
        observations=observations,
    )
    _write_ascii_ply(output_dir / "live_map.ply", xyz, rgb, semantic_label, semantic_conf, observations)
    return payload


def _stats(values: list[float]) -> dict[str, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0.0}
    ordered = sorted(finite)
    return {
        "count": float(len(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(ordered[len(ordered) // 2]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p90": float(ordered[min(len(ordered) - 1, int(math.ceil(0.90 * len(ordered))) - 1)]),
    }


def _write_viewer_html(output_dir: Path) -> None:
    html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LingBot Live Reconstruction</title>
  <style>
    :root { --bg:#07090b; --panel:#111820; --text:#e9f1e7; --muted:#9aa89c; --accent:#c88b45; }
    html, body { margin:0; height:100%; overflow:hidden; background:radial-gradient(circle at 20% 10%, #18251f 0, #07090b 48%, #030405 100%); color:var(--text); font-family: ui-sans-serif, system-ui, sans-serif; }
    #hud { position:fixed; left:18px; top:18px; z-index:10; background:rgba(9,13,16,.76); border:1px solid rgba(255,255,255,.12); border-radius:18px; padding:14px 16px; backdrop-filter: blur(12px); min-width:300px; box-shadow:0 20px 60px rgba(0,0,0,.3); }
    h1 { margin:0 0 6px; font-size:16px; letter-spacing:.08em; color:#f2c58b; text-transform:uppercase; }
    .row { display:flex; justify-content:space-between; gap:16px; font-size:13px; color:var(--muted); line-height:1.7; }
    .row b { color:var(--text); font-weight:650; }
    #hint { position:fixed; right:18px; bottom:18px; color:var(--muted); background:rgba(9,13,16,.66); padding:10px 12px; border-radius:14px; border:1px solid rgba(255,255,255,.1); font-size:12px; }
    canvas { display:block; width:100vw; height:100vh; cursor:grab; }
    canvas:active { cursor:grabbing; }
  </style>
</head>
<body>
  <div id="hud">
    <h1>LingBot Live Reconstruction</h1>
    <div class="row"><span>Status</span><b id="status">loading</b></div>
    <div class="row"><span>Points</span><b id="points">0</b></div>
    <div class="row"><span>Active frames</span><b id="frames">0</b></div>
    <div class="row"><span>Last event</span><b id="event">none</b></div>
    <div class="row"><span>Updated</span><b id="updated">-</b></div>
  </div>
  <canvas id="view"></canvas>
  <div id="hint">drag rotate · wheel zoom · auto-refreshes live_map.json</div>
<script>
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
let data = {points:[], trajectory:[], bbox_min:[0,0,0], bbox_max:[1,1,1], events:[]};
let yaw = -0.72, pitch = -0.58, zoom = 1.0, dragging = false, lastX = 0, lastY = 0;
function resize(){ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; }
addEventListener('resize', resize); resize();
canvas.addEventListener('mousedown', e => { dragging = true; lastX=e.clientX; lastY=e.clientY; });
addEventListener('mouseup', () => dragging = false);
addEventListener('mousemove', e => { if(!dragging) return; yaw += (e.clientX-lastX)*0.006; pitch += (e.clientY-lastY)*0.006; pitch=Math.max(-1.45, Math.min(1.45, pitch)); lastX=e.clientX; lastY=e.clientY; draw(); });
canvas.addEventListener('wheel', e => { e.preventDefault(); zoom *= Math.exp(-e.deltaY*0.001); zoom=Math.max(.15, Math.min(8, zoom)); draw(); }, {passive:false});
async function load(){
  try {
    const res = await fetch('live_map.json?t=' + Date.now(), {cache:'no-store'});
    data = await res.json();
    document.getElementById('status').textContent = 'live';
    document.getElementById('points').textContent = `${data.shown_point_count}/${data.point_count}`;
    document.getElementById('frames').textContent = (data.active_frames||[]).join(', ');
    const ev = (data.events||[]).slice(-1)[0];
    document.getElementById('event').textContent = ev ? `${ev.type} ${ev.frame_idx ?? ev.window ?? ''}` : 'none';
    document.getElementById('updated').textContent = data.updated_at || '-';
    draw();
  } catch(e) { document.getElementById('status').textContent = 'waiting'; }
}
function rot(p){
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  let x=p[0], y=p[1], z=p[2];
  let x1=cy*x + sy*z, z1=-sy*x + cy*z;
  let y1=cp*y - sp*z1, z2=sp*y + cp*z1;
  return [x1,y1,z2];
}
function project(p, center, scale){
  const q=rot([(p[0]-center[0])*scale, (p[1]-center[1])*scale, (p[2]-center[2])*scale]);
  const s = Math.min(canvas.width, canvas.height) * 0.42 * zoom;
  return [canvas.width*0.5 + q[0]*s, canvas.height*0.54 - q[1]*s, q[2]];
}
function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const mn=data.bbox_min||[0,0,0], mx=data.bbox_max||[1,1,1];
  const center=[(mn[0]+mx[0])/2,(mn[1]+mx[1])/2,(mn[2]+mx[2])/2];
  const span=Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2], 1e-3);
  const scale=1/span;
  ctx.fillStyle='#071016'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const traj=(data.trajectory||[]).filter(t=>t.position);
  if(traj.length>1){
    ctx.lineWidth=2*devicePixelRatio; ctx.strokeStyle='rgba(244,169,83,.88)'; ctx.beginPath();
    traj.forEach((t,i)=>{ const q=project(t.position,center,scale); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); });
    ctx.stroke();
  }
  const pts=(data.points||[]).map(p=>{ const q=project(p,center,scale); return {p,q}; }).sort((a,b)=>a.q[2]-b.q[2]);
  const r=Math.max(1.0, 1.7*devicePixelRatio);
  for(const item of pts){
    const p=item.p, q=item.q;
    const shade=Math.max(.25, Math.min(1.25, .75 + q[2]*.25));
    ctx.fillStyle=`rgb(${Math.min(255,p[3]*shade)},${Math.min(255,p[4]*shade)},${Math.min(255,p[5]*shade)})`;
    ctx.fillRect(q[0],q[1],r,r);
  }
}
setInterval(load, 1000); load();
</script>
</body>
</html>
"""
    (output_dir / "live_viewer.html").write_text(html, encoding="utf-8")


def _find_free_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _start_http_server(output_dir: Path, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=str(output_dir),
        stdout=(output_dir / "http_server.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _process_worker_result(
    result_path: Path,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    trajectory_by_frame: dict[int, dict[str, Any]],
    K_base: np.ndarray,
    args: argparse.Namespace,
    semantic_projector: YOLOSemanticProjector | None = None,
    fusion_map: VoxelFusionMap | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    process_started = time.perf_counter()
    pred_path = Path(result["predictions_npz"])
    summary_path = Path(result["summary_json"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pred = np.load(pred_path)
    depth = _squeeze_depth(pred["depth"])
    conf = _squeeze_depth(pred["depth_conf"]) if "depth_conf" in pred else np.ones_like(depth, dtype=np.float32)
    lingbot_extrinsic = np.asarray(pred["extrinsic"], dtype=np.float32) if "extrinsic" in pred else None
    lingbot_intrinsic = np.asarray(pred["intrinsic"], dtype=np.float32) if "intrinsic" in pred else None
    lingbot_translation_scale = float(
        getattr(args, "lingbot_pose_translation_scale", 0.0) or getattr(args, "depth_scale", 1.0)
    )
    frames = summary.get("metadata", {}).get("frames", [])
    updated = 0
    total_points = 0
    for local_idx, frame in enumerate(frames[: depth.shape[0]]):
        meta = frame.get("metadata") or {}
        frame_idx = int(frame.get("frame_idx", local_idx))
        pose = None
        if "pose" in meta:
            pose = np.asarray(meta["pose"], dtype=np.float32)
        elif frame_idx in trajectory_by_frame and "pose" in trajectory_by_frame[frame_idx]:
            pose = np.asarray(trajectory_by_frame[frame_idx]["pose"], dtype=np.float32)
        elif lingbot_extrinsic is not None and local_idx < lingbot_extrinsic.shape[0]:
            pose = _lingbot_extrinsic_to_pose(
                lingbot_extrinsic[local_idx],
                translation_scale=lingbot_translation_scale,
                mode=getattr(args, "lingbot_extrinsic_mode", "inverse"),
            )
        if pose is None:
            continue
        image_path = str(frame.get("image_path", ""))
        color_image_path = _resolve_color_image_path(args, frame_idx)
        sample_image_path = color_image_path or image_path
        original_shape = tuple(meta.get("frame_shape", [0, 0]))
        if len(original_shape) != 2 or original_shape[0] <= 0:
            rgb_probe = _load_rgb(sample_image_path)
            original_shape = rgb_probe.shape[:2] if rgb_probe is not None else depth[local_idx].shape
            rgb = rgb_probe
        else:
            rgb = _load_rgb(sample_image_path)
        if lingbot_intrinsic is not None and local_idx < lingbot_intrinsic.shape[0]:
            K = np.asarray(lingbot_intrinsic[local_idx], dtype=np.float32)
        else:
            K = _scaled_intrinsic(K_base, (int(original_shape[0]), int(original_shape[1])), depth[local_idx].shape)
        sampled = _sample_frame_points(
            depth=depth[local_idx],
            conf=conf[local_idx],
            rgb=rgb,
            pose=pose,
            K=K,
            frame_idx=frame_idx,
            args=args,
            semantic_projector=semantic_projector,
            image_path=sample_image_path,
        )
        if fusion_map is not None:
            fusion_map.update(sampled)
        else:
            frame_points[frame_idx] = sampled
            frame_points.move_to_end(frame_idx)
            while len(frame_points) > max(1, args.max_active_frames):
                frame_points.popitem(last=False)
        total_points += int(sampled["xyz"].shape[0])
        updated += 1
    process_elapsed = time.perf_counter() - process_started
    return {
        "type": "window",
        "window": int(result["index"]),
        "updated_frames": updated,
        "points_added_or_replaced": total_points,
        "elapsed_sec": float(result["elapsed_sec"]),
        "queue_wait_sec": float(result.get("queue_wait_sec", 0.0)),
        "worker_end_to_end_sec": float(result.get("end_to_end_sec", result["elapsed_sec"])),
        "process_result_sec": process_elapsed,
    }


def _refresh_fusion_snapshot(
    fusion_map: VoxelFusionMap | None,
    frame_points: OrderedDict[int, dict[str, np.ndarray]],
    fallback_frame_idx: int,
) -> int:
    if fusion_map is None:
        return int(sum(item["xyz"].shape[0] for item in frame_points.values()))
    snapshot = fusion_map.snapshot()
    frame_points.clear()
    if snapshot["xyz"].shape[0] > 0:
        frame_idx = int(snapshot["frame"].max())
    else:
        frame_idx = int(fallback_frame_idx)
    frame_points[frame_idx] = snapshot
    return int(snapshot["xyz"].shape[0])


def main() -> int:
    args = parse_args()
    if args.drop_when_busy:
        args.no_blocking_submit = True
    root = REPO_ROOT.parents[1] if False else Path.cwd()
    sequence_dir = Path(args.sequence_dir).expanduser().resolve()
    trajectory_path = Path(args.trajectory_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lingbot_map_root:
        os.environ["LINGBOT_MAP_ROOT"] = str(Path(args.lingbot_map_root).expanduser().resolve())

    _write_viewer_html(output_dir)
    K_base = _load_kitti_intrinsic(sequence_dir, args.intrinsic_camera_index)
    config = load_runtime_config(args.config or None)
    semantic_projector = YOLOSemanticProjector(args.yolo_model, conf=args.yolo_conf, imgsz=args.yolo_imgsz)
    fusion_map = (
        VoxelFusionMap(
            voxel_size=args.voxel_size,
            max_voxels=args.fusion_max_points,
            min_observations=args.fusion_min_observations,
        )
        if args.fusion_mode == "voxel"
        else None
    )
    if args.tracking_backend in {"opencv_mono_rgb", "cuvslam_mono_rgb"}:
        rgb_image_dir = args.rgb_image_dir or args.color_image_dir
        if not rgb_image_dir:
            raise SystemExit(f"--tracking-backend {args.tracking_backend} requires --rgb-image-dir or --color-image-dir")
        if not args.color_image_dir:
            args.color_image_dir = rgb_image_dir
        if args.tracking_backend == "cuvslam_mono_rgb":
            source = CUVSLAMMonocularRGBAdapter(
                image_dir=rgb_image_dir,
                timestamps_path=sequence_dir / "times.txt",
                intrinsic=K_base,
                config=config.tracking,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
            )
        else:
            source = RGBMonocularVOAdapter(
                image_dir=rgb_image_dir,
                timestamps_path=sequence_dir / "times.txt",
                intrinsic=K_base,
                config=config.tracking,
                frame_step=args.frame_step,
                max_frames=args.max_frames,
                fixed_step_scale=args.mono_fixed_step_scale,
                scale_source=args.mono_scale_source,
                oxts_dir=args.oxts_dir,
            )
    else:
        source = CUVSLAMOfflineKITTIAdapter(
            sequence_path=sequence_dir,
            trajectory_path=trajectory_path,
            config=config.tracking,
            frame_step=args.frame_step,
            max_frames=args.max_frames,
        )
    worker = LingBotDepthWorker(
        LingBotDepthWorkerConfig(
            model_path=str(Path(args.model_path).expanduser().resolve()),
            output_dir=str(output_dir / "worker"),
            image_size=args.image_size,
            model_image_size=args.model_image_size,
            window_size=args.window_size,
            stride=args.stride,
            num_scale_frames=args.num_scale_frames,
            max_queue=args.max_queue,
            force_cpu=False,
            offload_to_cpu=True,
            enable_3d_rope=False,
            depth_head_trt_engine=args.depth_head_trt_engine,
            model_patch_embed=args.model_patch_embed,
            model_embed_dim=args.model_embed_dim,
            model_depth=args.model_depth,
            model_num_heads=args.model_num_heads,
            model_mlp_ratio=args.model_mlp_ratio,
            compress_outputs=not args.no_compress_output,
        )
    )
    frame_points: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
    trajectory: list[dict[str, Any]] = []
    trajectory_by_frame: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "track_next_sec": [],
        "submit_sec": [],
        "process_result_sec": [],
        "live_write_sec": [],
        "worker_elapsed_sec": [],
        "worker_queue_wait_sec": [],
        "worker_end_to_end_sec": [],
        "points_added_or_replaced": [],
        "tracking": [],
        "windows": [],
        "dense_submit": [],
    }
    processed_results: set[Path] = set()
    server = None
    if args.serve:
        port = _find_free_port(args.port)
        args.port = port
        server = _start_http_server(output_dir, port)
        events.append({"type": "server", "url": f"http://127.0.0.1:{port}/live_viewer.html"})
    if args.yolo_model and not semantic_projector.enabled:
        events.append({"type": "warning", "message": semantic_projector.error or "YOLO disabled"})
        print(semantic_projector.error or "YOLO disabled", flush=True)

    _write_live_json(output_dir, frame_points, trajectory, events, args)
    worker.start()
    start = time.perf_counter()

    def process_new_results() -> None:
        for result_path in sorted((output_dir / "worker").glob("window_*/worker_result.json")):
            if result_path in processed_results:
                continue
            event = _process_worker_result(
                result_path,
                frame_points,
                trajectory_by_frame,
                K_base,
                args,
                semantic_projector=semantic_projector,
                fusion_map=fusion_map,
            )
            processed_results.add(result_path)
            events.append(event)
            should_publish = (
                args.publish_every_windows > 0
                and len(processed_results) % args.publish_every_windows == 0
            )
            live_write_sec = 0.0
            point_count = int(sum(item["xyz"].shape[0] for item in frame_points.values()))
            if should_publish:
                point_count = _refresh_fusion_snapshot(fusion_map, frame_points, int(event.get("window", 0)))
                write_started = time.perf_counter()
                payload = _write_live_json(output_dir, frame_points, trajectory, events, args)
                live_write_sec = time.perf_counter() - write_started
                point_count = int(payload["point_count"])
            event["live_write_sec"] = live_write_sec
            event["published"] = bool(should_publish)
            metrics["process_result_sec"].append(float(event.get("process_result_sec", 0.0)))
            if should_publish:
                metrics["live_write_sec"].append(live_write_sec)
            metrics["worker_elapsed_sec"].append(float(event.get("elapsed_sec", 0.0)))
            metrics["worker_queue_wait_sec"].append(float(event.get("queue_wait_sec", 0.0)))
            metrics["worker_end_to_end_sec"].append(float(event.get("worker_end_to_end_sec", event.get("elapsed_sec", 0.0))))
            metrics["points_added_or_replaced"].append(int(event.get("points_added_or_replaced", 0)))
            metrics["windows"].append(event)
            print(json.dumps({"live_update": event, "point_count": point_count}, ensure_ascii=False), flush=True)

    submitted = 0
    dense_candidates = 0
    dense_submitted_frames = 0
    dense_interval_skips = 0
    dense_non_keyframe_skips = 0
    dense_window_schedules = 0
    dense_queue_full_drops_last = 0
    skipped = 0
    try:
        source_iter = iter(source)
        while True:
            track_started = time.perf_counter()
            try:
                item = next(source_iter)
            except StopIteration:
                break
            track_next_sec = time.perf_counter() - track_started
            metrics["track_next_sec"].append(track_next_sec)
            pose = item.pose.astype(np.float32)
            traj_item = {
                "frame_idx": int(item.frame_idx),
                "timestamp_sec": float(item.timestamp_sec),
                "position": pose[:3, 3].astype(float).tolist(),
                "pose": pose.astype(float).tolist(),
                "is_keyframe": bool(item.is_keyframe),
                "track_ok": bool(item.track_ok),
            }
            trajectory.append(traj_item)
            trajectory_by_frame[int(item.frame_idx)] = traj_item
            events.append({"type": "track", "frame_idx": int(item.frame_idx), "is_keyframe": bool(item.is_keyframe)})
            if item.image_path is None:
                continue
            should_submit_dense = True
            dense_skip_reason = ""
            if args.keyframes_only and not item.is_keyframe:
                should_submit_dense = False
                dense_skip_reason = "non_keyframe"
                dense_non_keyframe_skips += 1
            dense_candidates += 1
            if should_submit_dense and args.dense_frame_interval > 1:
                if (dense_candidates - 1) % int(args.dense_frame_interval) != 0:
                    should_submit_dense = False
                    dense_skip_reason = "dense_frame_interval"
                    dense_interval_skips += 1
            if not should_submit_dense:
                skipped += 1
                metrics["tracking"].append(
                    {
                        "frame_idx": int(item.frame_idx),
                        "track_next_sec": track_next_sec,
                        "submit_sec": 0.0,
                        "dense_submitted": False,
                        "dense_skip_reason": dense_skip_reason,
                        "is_keyframe": bool(item.is_keyframe),
                        "track_ok": bool(item.track_ok),
                        "keypoint_count": int(item.keypoint_count),
                        "match_count": int(item.match_count),
                        "inlier_count": int(item.inlier_count),
                        "pixel_motion": float(item.pixel_motion),
                    }
                )
                process_new_results()
                if args.frame_sleep_sec > 0:
                    time.sleep(args.frame_sleep_sec)
                continue
            submit_started = time.perf_counter()
            scheduled_window = worker.submit(
                item.image_path,
                frame_idx=int(item.frame_idx),
                timestamp_sec=float(item.timestamp_sec),
                metadata={
                    "pose": pose.tolist(),
                    "frame_shape": [int(item.frame_shape[0]), int(item.frame_shape[1])],
                    "is_keyframe": bool(item.is_keyframe),
                    "track_ok": bool(item.track_ok),
                    "keypoint_count": int(item.keypoint_count),
                    "match_count": int(item.match_count),
                    "inlier_count": int(item.inlier_count),
                    "pixel_motion": float(item.pixel_motion),
                },
                block=not args.no_blocking_submit,
            )
            submit_sec = time.perf_counter() - submit_started
            dense_submitted_frames += 1
            worker_status = worker.status()
            queue_full_drops = int(worker_status.get("queue_full_drops", 0))
            queue_full_drop_delta = max(0, queue_full_drops - dense_queue_full_drops_last)
            dense_queue_full_drops_last = queue_full_drops
            if scheduled_window:
                dense_window_schedules += 1
            metrics["submit_sec"].append(submit_sec)
            metrics["dense_submit"].append(
                {
                    "frame_idx": int(item.frame_idx),
                    "submit_sec": submit_sec,
                    "scheduled_window": bool(scheduled_window),
                    "queue_full_drop_delta": int(queue_full_drop_delta),
                    "worker_queue_size": int(worker_status.get("queue_size", 0)),
                    "worker_scheduled_windows": int(worker_status.get("scheduled_windows", 0)),
                    "worker_completed_windows": int(worker_status.get("completed_windows", 0)),
                }
            )
            metrics["tracking"].append(
                {
                    "frame_idx": int(item.frame_idx),
                    "track_next_sec": track_next_sec,
                    "submit_sec": submit_sec,
                    "dense_submitted": True,
                    "dense_scheduled_window": bool(scheduled_window),
                    "dense_queue_full_drop_delta": int(queue_full_drop_delta),
                    "is_keyframe": bool(item.is_keyframe),
                    "track_ok": bool(item.track_ok),
                    "keypoint_count": int(item.keypoint_count),
                    "match_count": int(item.match_count),
                    "inlier_count": int(item.inlier_count),
                    "pixel_motion": float(item.pixel_motion),
                }
            )
            submitted += 1
            process_new_results()
            if args.publish_every_frames > 0 and submitted % args.publish_every_frames == 0:
                write_started = time.perf_counter()
                _write_live_json(output_dir, frame_points, trajectory, events, args)
                metrics["live_write_sec"].append(time.perf_counter() - write_started)
            if args.frame_sleep_sec > 0:
                time.sleep(args.frame_sleep_sec)

        worker.stop(drain=True)
        process_new_results()
    finally:
        if worker.status().get("stop_requested") is False:
            worker.stop(drain=False)

    _refresh_fusion_snapshot(fusion_map, frame_points, int(processed_results and len(processed_results) or 0))
    payload = _write_live_json(output_dir, frame_points, trajectory, events, args)
    final_worker_status = worker.status()
    summary = {
        "output_dir": str(output_dir),
        "tracking_backend": str(args.tracking_backend),
        "rgb_image_dir": str(args.rgb_image_dir),
        "viewer_html": str(output_dir / "live_viewer.html"),
        "live_map_json": str(output_dir / "live_map.json"),
        "live_map_ply": str(output_dir / "live_map.ply"),
        "tracked_frames": len(metrics["tracking"]),
        "submitted_frames": submitted,
        "skipped_frames": skipped,
        "dense_policy": {
            "keyframes_only": bool(args.keyframes_only),
            "dense_frame_interval": int(args.dense_frame_interval),
            "window_stride": int(args.stride),
            "max_queue": int(args.max_queue),
            "drop_when_busy": bool(args.no_blocking_submit),
        },
        "reconstruction": {
            "fusion_mode": str(args.fusion_mode),
            "voxel_size": float(args.voxel_size),
            "fusion_max_points": int(args.fusion_max_points),
            "fusion_min_observations": int(args.fusion_min_observations),
            "adaptive_sampling": bool(args.adaptive_sampling),
            "lingbot_extrinsic_mode": str(args.lingbot_extrinsic_mode),
            "sample_stride": int(args.sample_stride),
            "near_sample_stride": int(args.near_sample_stride),
            "edge_sample_stride": int(args.edge_sample_stride),
            "semantic_sample_stride": int(args.semantic_sample_stride),
            "yolo_model": str(args.yolo_model),
            "yolo_enabled": bool(semantic_projector.enabled),
            "semantic_color_output": bool(args.semantic_color_output),
        },
        "dense_candidates": dense_candidates,
        "dense_submitted_frames": dense_submitted_frames,
        "dense_interval_skips": dense_interval_skips,
        "dense_non_keyframe_skips": dense_non_keyframe_skips,
        "dense_window_schedules": dense_window_schedules,
        "dense_queue_full_drops": int(final_worker_status.get("queue_full_drops", 0)),
        "worker_status": final_worker_status,
        "processed_windows": len(processed_results),
        "point_count": int(payload["point_count"]),
        "elapsed_sec": time.perf_counter() - start,
        "server_url": f"http://127.0.0.1:{args.port}/live_viewer.html" if args.serve else "",
        "server_pid": int(server.pid) if server is not None else 0,
    }
    metrics_summary = {
        "track_next_sec": _stats(metrics["track_next_sec"]),
        "submit_sec": _stats(metrics["submit_sec"]),
        "worker_elapsed_sec": _stats(metrics["worker_elapsed_sec"]),
        "worker_queue_wait_sec": _stats(metrics["worker_queue_wait_sec"]),
        "worker_end_to_end_sec": _stats(metrics["worker_end_to_end_sec"]),
        "process_result_sec": _stats(metrics["process_result_sec"]),
        "live_write_sec": _stats(metrics["live_write_sec"]),
        "points_added_or_replaced": _stats([float(v) for v in metrics["points_added_or_replaced"]]),
        "dense_submit_sec": _stats([float(item["submit_sec"]) for item in metrics["dense_submit"]]),
    }
    metrics_payload = {
        "summary": summary,
        "latency": metrics_summary,
        "tracking": metrics["tracking"],
        "dense_submit": metrics["dense_submit"],
        "windows": metrics["windows"],
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "full_stack_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
