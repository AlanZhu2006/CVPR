#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rosbag
from nav_msgs.msg import Odometry

from fastlivo_direct_playback_server import (
    TOPICS,
    Calibration,
    LatestImage,
    MapVoxel,
    decode_image,
    encode_jpeg,
    extract_scan_points,
    load_calibration,
    odom_to_matrix,
    ros_time_to_sec,
    transform_points,
    trim_map_voxels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cached FAST-LIVO playback pack from a rosbag.")
    parser.add_argument("--bag-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scan-max-points", type=int, default=6000)
    parser.add_argument("--scan-voxel-size", type=float, default=0.10)
    parser.add_argument("--map-max-points", type=int, default=160_000)
    parser.add_argument("--map-voxel-size", type=float, default=0.16)
    parser.add_argument("--web-min-range", type=float, default=1.0)
    parser.add_argument("--web-max-range", type=float, default=24.0)
    parser.add_argument("--web-max-abs-z", type=float, default=8.0)
    parser.add_argument("--image-sync-tolerance", type=float, default=0.2)
    parser.add_argument("--pose-sync-tolerance", type=float, default=0.12)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--keyframe-every", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_pose_record_for_stamp(
    pose_history: collections.deque[tuple[float, np.ndarray, dict[str, object]]],
    latest_pose_matrix: np.ndarray | None,
    latest_pose_record: dict[str, object] | None,
    stamp_sec: float,
    tolerance: float,
) -> tuple[np.ndarray | None, dict[str, object] | None]:
    if not pose_history:
        return latest_pose_matrix, latest_pose_record

    best_matrix: np.ndarray | None = None
    best_record: dict[str, object] | None = None
    best_distance = float("inf")

    for pose_stamp_sec, pose_matrix, pose_record in pose_history:
        distance = abs(pose_stamp_sec - stamp_sec)
        if distance < best_distance:
            best_distance = distance
            best_matrix = pose_matrix
            best_record = pose_record

    if best_matrix is not None and best_distance <= tolerance:
        return best_matrix, best_record

    return latest_pose_matrix, latest_pose_record


def pose_record_from_msg(message: Odometry) -> dict[str, object]:
    pose = message.pose.pose
    return {
        "stampSec": ros_time_to_sec(message.header.stamp),
        "frameId": message.header.frame_id or "world",
        "childFrameId": message.child_frame_id or "base_link",
        "position": [
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        ],
        "orientation": [
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        ],
    }


def update_global_map(
    voxels: dict[tuple[int, int, int], MapVoxel],
    world_positions: np.ndarray,
    colors: np.ndarray,
    stamp_sec: float,
    voxel_size: float,
    max_points: int,
) -> None:
    voxel_size = max(float(voxel_size), 1e-3)
    for index in range(world_positions.shape[0]):
        x = float(world_positions[index, 0])
        y = float(world_positions[index, 1])
        z = float(world_positions[index, 2])
        key = (
            int(round(x / voxel_size)),
            int(round(y / voxel_size)),
            int(round(z / voxel_size)),
        )
        existing = voxels.get(key)
        if existing is None:
            voxels[key] = MapVoxel(
                x=x,
                y=y,
                z=z,
                position_samples=1,
                color=colors[index].astype(np.float64),
                color_samples=1,
                last_seen_stamp_sec=stamp_sec,
            )
            continue

        existing.position_samples += 1
        mix = 1.0 / float(existing.position_samples)
        existing.x += (x - existing.x) * mix
        existing.y += (y - existing.y) * mix
        existing.z += (z - existing.z) * mix
        existing.last_seen_stamp_sec = stamp_sec
        if existing.color is None:
            existing.color = colors[index].astype(np.float64)
            existing.color_samples = 1
        else:
            existing.color_samples += 1
            color_mix = 1.0 / float(existing.color_samples)
            existing.color += (colors[index].astype(np.float64) - existing.color) * color_mix

    trim_map_voxels(voxels, max_points)


def write_scan(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    np.savez_compressed(path, positions=positions.astype(np.float32), colors=colors.astype(np.uint8))


def write_keyframe(path: Path, voxels: dict[tuple[int, int, int], MapVoxel]) -> int:
    positions = np.empty((len(voxels), 3), dtype=np.float32)
    colors = np.empty((len(voxels), 3), dtype=np.uint8)

    for index, voxel in enumerate(voxels.values()):
        positions[index, 0] = voxel.x
        positions[index, 1] = voxel.y
        positions[index, 2] = voxel.z
        if voxel.color is None:
            colors[index] = np.array([216, 236, 255], dtype=np.uint8)
        else:
            colors[index] = np.clip(np.round(voxel.color), 0, 255).astype(np.uint8)

    np.savez_compressed(path, positions=positions, colors=colors)
    return int(positions.shape[0])


def main() -> None:
    args = parse_args()
    bag_path = Path(args.bag_path).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    calibration: Calibration = load_calibration(config_path)

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

    bag = rosbag.Bag(str(bag_path))
    info = bag.get_type_and_topic_info()[1]
    start_sec = float(bag.get_start_time())
    end_sec = float(bag.get_end_time())
    duration_sec = max(0.0, end_sec - start_sec)

    latest_image: LatestImage | None = None
    latest_image_jpeg: bytes | None = None
    latest_pose_matrix: np.ndarray | None = None
    latest_pose_record: dict[str, object] | None = None
    pose_history: collections.deque[tuple[float, np.ndarray, dict[str, object]]] = collections.deque(maxlen=256)
    global_map_voxels: dict[tuple[int, int, int], MapVoxel] = {}
    frames: list[dict[str, object]] = []
    keyframes: list[dict[str, object]] = []

    cloud_frame_index = 0
    start_wall = time.time()

    for topic, msg, stamp in bag.read_messages(topics=list(TOPICS)):
        if topic == "/origin_img":
            latest_image = decode_image(msg)
            latest_image_jpeg = encode_jpeg(latest_image, quality=args.jpeg_quality) if latest_image else None
            continue

        if topic == "/aft_mapped_to_init":
            latest_pose_matrix = odom_to_matrix(msg)
            latest_pose_record = pose_record_from_msg(msg)
            pose_history.append((ros_time_to_sec(msg.header.stamp), latest_pose_matrix, latest_pose_record))
            continue

        if topic != "/cloud_registered_body":
            continue

        cloud_stamp_sec = ros_time_to_sec(msg.header.stamp)
        pose_matrix, pose_record = get_pose_record_for_stamp(
            pose_history,
            latest_pose_matrix,
            latest_pose_record,
            cloud_stamp_sec,
            float(args.pose_sync_tolerance),
        )
        if pose_matrix is None or pose_record is None:
            continue

        scan_positions_lidar, scan_colors = extract_scan_points(
            msg,
            latest_image,
            calibration,
            max_points=int(args.scan_max_points),
            voxel_size=float(args.scan_voxel_size),
            min_range=float(args.web_min_range),
            max_range=float(args.web_max_range),
            max_abs_z=float(args.web_max_abs_z),
            sync_tolerance=float(args.image_sync_tolerance),
            neutral_when_image_unavailable=True,
        )
        if scan_positions_lidar.size == 0:
            continue

        scan_positions_world = transform_points(pose_matrix @ calibration.t_bl, scan_positions_lidar)
        update_global_map(
            global_map_voxels,
            scan_positions_world,
            scan_colors,
            cloud_stamp_sec,
            voxel_size=float(args.map_voxel_size),
            max_points=int(args.map_max_points),
        )

        scan_file = scans_dir / f"scan_{cloud_frame_index:06d}.npz"
        write_scan(scan_file, scan_positions_world, scan_colors)

        image_rel: str | None = None
        image_stamp_sec: float | None = latest_image.stamp_sec if latest_image is not None else None
        if latest_image_jpeg is not None:
            image_path = images_dir / f"frame_{cloud_frame_index:06d}.jpg"
            image_path.write_bytes(latest_image_jpeg)
            image_rel = image_path.relative_to(output_dir).as_posix()

        frame_meta: dict[str, object] = {
            "index": cloud_frame_index,
            "offsetSec": max(0.0, cloud_stamp_sec - start_sec),
            "stampSec": cloud_stamp_sec,
            "scanFile": scan_file.relative_to(output_dir).as_posix(),
            "scanPointCount": int(scan_positions_world.shape[0]),
            "pose": pose_record,
            "imageFile": image_rel,
            "imageStampSec": image_stamp_sec,
        }
        frames.append(frame_meta)

        if cloud_frame_index == 0 or cloud_frame_index % max(int(args.keyframe_every), 1) == 0:
            keyframe_file = keyframes_dir / f"map_{cloud_frame_index:06d}.npz"
            point_count = write_keyframe(keyframe_file, global_map_voxels)
            keyframes.append(
                {
                    "frameIndex": cloud_frame_index,
                    "offsetSec": frame_meta["offsetSec"],
                    "file": keyframe_file.relative_to(output_dir).as_posix(),
                    "pointCount": point_count,
                }
            )

        cloud_frame_index += 1
        if cloud_frame_index % 100 == 0:
            elapsed = max(time.time() - start_wall, 1e-5)
            print(
                json.dumps(
                    {
                        "frames": cloud_frame_index,
                        "elapsedSec": round(elapsed, 2),
                        "fps": round(cloud_frame_index / elapsed, 2),
                        "mapPoints": len(global_map_voxels),
                    }
                ),
                flush=True,
            )

        if int(args.max_frames) > 0 and cloud_frame_index >= int(args.max_frames):
            break

    bag.close()

    meta = {
        "version": 1,
        "mode": "fastlivo-playback-cache",
        "bagPath": str(bag_path),
        "configPath": str(config_path),
        "builtAtSec": time.time(),
        "startSec": start_sec,
        "endSec": end_sec,
        "durationSec": duration_sec,
        "frameCount": len(frames),
        "topics": {
            topic: {"type": info[topic].msg_type, "count": int(info[topic].message_count)}
            for topic in TOPICS
            if topic in info
        },
        "scanMaxPoints": int(args.scan_max_points),
        "scanVoxelSize": float(args.scan_voxel_size),
        "mapMaxPoints": int(args.map_max_points),
        "mapVoxelSize": float(args.map_voxel_size),
        "keyframeEvery": int(args.keyframe_every),
        "imageSyncTolerance": float(args.image_sync_tolerance),
        "poseSyncTolerance": float(args.pose_sync_tolerance),
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
                "durationSec": duration_sec,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
