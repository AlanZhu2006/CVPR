#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


DEFAULT_LINGBOT_REPO = Path("/home/chatsign/work/lingbot-map")
DEFAULT_MODEL_PATH = Path("/media/chatsign/data-002/models/lingbot-map/lingbot-map-long.pt")


@dataclass
class MapVoxel:
    x: float
    y: float
    z: float
    position_samples: int
    color: np.ndarray
    color_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a playback cache from LingBot-Map predictions.")
    parser.add_argument("--lingbot-repo", default=str(DEFAULT_LINGBOT_REPO))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--image-folder", default="")
    parser.add_argument("--video-path", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--first-k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--max-frame-num", type=int, default=1024)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--keyframe-interval", type=int, default=None)
    parser.add_argument("--kv-cache-sliding-window", type=int, default=64)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--overlap-size", type=int, default=16)
    parser.add_argument("--use-sdpa", action="store_true", default=True)
    parser.add_argument("--no-use-sdpa", action="store_false", dest="use_sdpa")
    parser.add_argument("--offload-to-cpu", action="store_true", default=False)
    parser.add_argument("--conf-threshold", type=float, default=1.5)
    parser.add_argument("--scan-max-points", type=int, default=12000)
    parser.add_argument("--map-max-points", type=int, default=240000)
    parser.add_argument("--map-voxel-size", type=float, default=0.002)
    parser.add_argument("--keyframe-every", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_demo_module(repo_path: Path) -> ModuleType:
    demo_path = repo_path / "demo.py"
    if not demo_path.exists():
      raise SystemExit(f"LingBot-Map demo.py not found: {demo_path}")

    sys.path.insert(0, str(repo_path))
    spec = importlib.util.spec_from_file_location("lingbot_map_demo", demo_path)
    if spec is None or spec.loader is None:
      raise SystemExit(f"Failed to import LingBot-Map demo from {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def squeeze_single_batch(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def predictions_to_numpy(predictions: dict[str, Any]) -> dict[str, Any]:
    return {key: squeeze_single_batch(value) for key, value in predictions.items()}


def images_to_numpy(images: Any) -> np.ndarray:
    images_np = squeeze_single_batch(images)
    if isinstance(images_np, torch.Tensor):
        images_np = images_np.detach().cpu().numpy()
    if not isinstance(images_np, np.ndarray):
        raise TypeError("LingBot images must be a tensor or numpy array.")
    if images_np.ndim == 5 and images_np.shape[0] == 1:
        images_np = images_np[0]
    if images_np.ndim != 4:
        raise ValueError(f"Expected images shaped Sx3xHxW, got {images_np.shape}.")
    return images_np


def downsample_by_confidence(
    positions: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray | None,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = int(positions.shape[0])
    if max_points <= 0 or count <= max_points:
        return positions, colors

    if confidence is not None and confidence.shape[0] == count:
        selected = np.argpartition(confidence, -max_points)[-max_points:]
        selected.sort()
    else:
        selected = np.linspace(0, count - 1, num=max_points, dtype=np.int32)
    return positions[selected], colors[selected]


def sample_frame_cloud(
    world_points: np.ndarray,
    frame_colors: np.ndarray,
    confidence: np.ndarray | None,
    conf_threshold: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = world_points.reshape(-1, 3).astype(np.float32, copy=False)
    colors = frame_colors.reshape(-1, 3).astype(np.uint8, copy=False)
    conf_flat = confidence.reshape(-1).astype(np.float32, copy=False) if confidence is not None else None

    finite_mask = np.isfinite(positions).all(axis=1)
    if conf_flat is not None:
        finite_mask &= np.isfinite(conf_flat) & (conf_flat >= float(conf_threshold))

    positions = positions[finite_mask]
    colors = colors[finite_mask]
    selected_conf = conf_flat[finite_mask] if conf_flat is not None else None
    return downsample_by_confidence(positions, colors, selected_conf, max_points=max_points)


def write_scan(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    np.savez_compressed(path, positions=positions.astype(np.float32), colors=colors.astype(np.uint8))


def update_global_map(
    voxels: dict[tuple[int, int, int], MapVoxel],
    world_positions: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    max_points: int,
) -> None:
    voxel_size = max(float(voxel_size), 1e-4)
    for index in range(world_positions.shape[0]):
        x = float(world_positions[index, 0])
        y = float(world_positions[index, 1])
        z = float(world_positions[index, 2])
        key = (
            int(round(x / voxel_size)),
            int(round(y / voxel_size)),
            int(round(z / voxel_size)),
        )
        color = colors[index].astype(np.float64)
        existing = voxels.get(key)
        if existing is None:
            voxels[key] = MapVoxel(
                x=x,
                y=y,
                z=z,
                position_samples=1,
                color=color,
                color_samples=1,
            )
            continue

        existing.position_samples += 1
        position_mix = 1.0 / float(existing.position_samples)
        existing.x += (x - existing.x) * position_mix
        existing.y += (y - existing.y) * position_mix
        existing.z += (z - existing.z) * position_mix
        existing.color_samples += 1
        color_mix = 1.0 / float(existing.color_samples)
        existing.color += (color - existing.color) * color_mix

    trim_voxels(voxels, max_points=max_points)


def trim_voxels(voxels: dict[tuple[int, int, int], MapVoxel], max_points: int) -> None:
    if max_points <= 0 or len(voxels) <= max_points:
        return
    keys = list(voxels.keys())
    for key in keys[: len(voxels) - max_points]:
        voxels.pop(key, None)


def write_keyframe(path: Path, voxels: dict[tuple[int, int, int], MapVoxel]) -> int:
    positions = np.empty((len(voxels), 3), dtype=np.float32)
    colors = np.empty((len(voxels), 3), dtype=np.uint8)
    for index, voxel in enumerate(voxels.values()):
        positions[index] = (voxel.x, voxel.y, voxel.z)
        colors[index] = np.clip(np.round(voxel.color), 0, 255).astype(np.uint8)
    np.savez_compressed(path, positions=positions, colors=colors)
    return int(positions.shape[0])


def rotation_to_quaternion(rotation_matrix: np.ndarray) -> list[float]:
    try:
        quat = Rotation.from_matrix(rotation_matrix).as_quat()
    except ValueError:
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def encode_rgb_jpeg(image_rgb: np.ndarray, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB frame as JPEG.")
    return encoded.tobytes()


def main() -> None:
    args = parse_args()
    lingbot_repo = Path(args.lingbot_repo).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_path.exists():
        raise SystemExit(f"LingBot-Map checkpoint not found: {model_path}")
    if not args.image_folder and not args.video_path:
        raise SystemExit("Provide --image-folder or --video-path.")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    scans_dir = output_dir / "scans"
    images_dir = output_dir / "images"
    keyframes_dir = output_dir / "keyframes"
    scans_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    demo = load_demo_module(lingbot_repo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, paths, resolved_image_folder = demo.load_images(
        image_folder=args.image_folder or None,
        video_path=args.video_path or None,
        fps=int(args.fps),
        first_k=args.first_k,
        stride=args.stride,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )

    model_args = argparse.Namespace(
        model_path=str(model_path),
        image_size=args.image_size,
        patch_size=args.patch_size,
        mode=args.mode,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        num_scale_frames=args.num_scale_frames,
        keyframe_interval=args.keyframe_interval,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        camera_num_iterations=args.camera_num_iterations,
        use_sdpa=args.use_sdpa,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        offload_to_cpu=args.offload_to_cpu,
    )
    model = demo.load_model(model_args, device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = int(images.shape[0])
    if model_args.keyframe_interval is None:
        model_args.keyframe_interval = (num_frames + 319) // 320 if args.mode == "streaming" and num_frames > 320 else 1

    output_device = torch.device("cpu") if args.offload_to_cpu else None
    start_wall = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=model_args.keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                output_device=output_device,
            )

    images_for_post = predictions["images"] if args.offload_to_cpu else images
    predictions, images_cpu = demo.postprocess(predictions, images_for_post)
    pred_np = predictions_to_numpy(predictions)
    images_np = images_to_numpy(images_cpu)

    from lingbot_map.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map

    if "world_points" in pred_np:
        world_points = np.asarray(pred_np["world_points"], dtype=np.float32)
        confidence = pred_np.get("world_points_conf", pred_np.get("depth_conf"))
    else:
        world_points = unproject_depth_map_to_point_map(
            np.asarray(pred_np["depth"]),
            np.asarray(pred_np["extrinsic"]),
            np.asarray(pred_np["intrinsic"]),
        ).astype(np.float32)
        confidence = pred_np.get("depth_conf")
    confidence = np.asarray(confidence, dtype=np.float32) if confidence is not None else None

    image_rgb_u8 = np.clip(np.round(images_np.transpose(0, 2, 3, 1) * 255.0), 0, 255).astype(np.uint8)
    extrinsics = np.asarray(pred_np["extrinsic"], dtype=np.float32)
    cam_to_world = closed_form_inverse_se3(extrinsics)

    frame_count = int(world_points.shape[0])
    frame_dt = 1.0 / max(float(args.fps), 1e-6)
    start_sec = 0.0
    end_sec = max(0.0, (frame_count - 1) * frame_dt)
    frames: list[dict[str, Any]] = []
    keyframes: list[dict[str, Any]] = []
    voxels: dict[tuple[int, int, int], MapVoxel] = {}

    for frame_index in range(frame_count):
        frame_conf = confidence[frame_index] if confidence is not None else None
        scan_positions, scan_colors = sample_frame_cloud(
            world_points[frame_index],
            image_rgb_u8[frame_index],
            frame_conf,
            conf_threshold=float(args.conf_threshold),
            max_points=int(args.scan_max_points),
        )

        update_global_map(
            voxels,
            scan_positions,
            scan_colors,
            voxel_size=float(args.map_voxel_size),
            max_points=int(args.map_max_points),
        )

        scan_path = scans_dir / f"scan_{frame_index:06d}.npz"
        write_scan(scan_path, scan_positions, scan_colors)

        image_path = images_dir / f"frame_{frame_index:06d}.jpg"
        image_path.write_bytes(encode_rgb_jpeg(image_rgb_u8[frame_index], jpeg_quality=int(args.jpeg_quality)))

        stamp_sec = frame_index * frame_dt
        pose_matrix = cam_to_world[frame_index]
        pose_record = {
            "stampSec": stamp_sec,
            "frameId": "world",
            "childFrameId": "lingbot_camera",
            "position": [
                float(pose_matrix[0, 3]),
                float(pose_matrix[1, 3]),
                float(pose_matrix[2, 3]),
            ],
            "orientation": rotation_to_quaternion(np.asarray(pose_matrix[:3, :3], dtype=np.float64)),
        }
        frame_meta = {
            "index": frame_index,
            "offsetSec": stamp_sec,
            "stampSec": stamp_sec,
            "scanFile": scan_path.relative_to(output_dir).as_posix(),
            "scanPointCount": int(scan_positions.shape[0]),
            "pose": pose_record,
            "imageFile": image_path.relative_to(output_dir).as_posix(),
            "imageStampSec": stamp_sec,
        }
        frames.append(frame_meta)

        if frame_index == 0 or frame_index % max(int(args.keyframe_every), 1) == 0:
            keyframe_path = keyframes_dir / f"map_{frame_index:06d}.npz"
            point_count = write_keyframe(keyframe_path, voxels)
            keyframes.append(
                {
                    "frameIndex": frame_index,
                    "offsetSec": stamp_sec,
                    "file": keyframe_path.relative_to(output_dir).as_posix(),
                    "pointCount": point_count,
                }
            )

        if frame_index % 25 == 0:
            print(
                json.dumps(
                    {
                        "frames": frame_index + 1,
                        "scanPoints": int(scan_positions.shape[0]),
                        "mapPoints": len(voxels),
                    }
                ),
                flush=True,
            )

    meta = {
        "version": 1,
        "mode": "lingbot-map-playback-cache",
        "bagPath": args.video_path or args.image_folder or str(resolved_image_folder),
        "configPath": str(lingbot_repo / "pyproject.toml"),
        "builtAtSec": time.time(),
        "startSec": start_sec,
        "endSec": end_sec,
        "durationSec": end_sec,
        "frameCount": len(frames),
        "topics": {
            "/lingbot_map/camera_pose": {"type": "geometry_msgs/PoseStamped", "count": len(frames)},
            "/lingbot_map/rgb": {"type": "sensor_msgs/Image", "count": len(frames)},
            "/lingbot_map/world_points": {"type": "sensor_msgs/PointCloud2", "count": len(frames)},
        },
        "scanMaxPoints": int(args.scan_max_points),
        "scanVoxelSize": 0.0,
        "mapMaxPoints": int(args.map_max_points),
        "mapVoxelSize": float(args.map_voxel_size),
        "keyframeEvery": int(args.keyframe_every),
        "imageSyncTolerance": 0.0,
        "poseSyncTolerance": 0.0,
        "source": {
            "kind": "lingbot_map",
            "repoPath": str(lingbot_repo),
            "modelPath": str(model_path),
            "imageFolder": args.image_folder,
            "videoPath": args.video_path,
            "fps": float(args.fps),
            "confThreshold": float(args.conf_threshold),
        },
        "frames": frames,
        "keyframes": keyframes,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "cacheDir": str(output_dir),
                "frameCount": len(frames),
                "keyframes": len(keyframes),
                "durationSec": end_sec,
                "elapsedSec": round(time.time() - start_wall, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
