#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a LingBot live_map.npz as ROS2 PointCloud2 for RViz2.")
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/lingbot/cloud")
    parser.add_argument("--plain-topic", default="", help="Optional second topic with the same xyz and neutral gray RGB.")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--mode", choices=("all", "current", "reveal"), default="reveal")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Reload live_map.npz when it changes on disk.")
    parser.add_argument("--semantic-color", action="store_true")
    return parser.parse_args()


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


def load_cloud(map_dir: Path, semantic_color: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(map_dir.expanduser().resolve() / "live_map.npz")
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    frame = np.asarray(data["frame"], dtype=np.int32)
    semantic = np.asarray(data["semantic_label"], dtype=np.int32) if "semantic_label" in data else np.full((xyz.shape[0],), -1, dtype=np.int32)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb, frame, semantic = xyz[finite], rgb[finite], frame[finite], semantic[finite]
    if semantic_color:
        valid = semantic >= 0
        rgb = rgb.copy()
        rgb[valid] = SEMANTIC_PALETTE[semantic[valid] % len(SEMANTIC_PALETTE)]
    return xyz, rgb, frame


def select_points(xyz: np.ndarray, rgb: np.ndarray, frame: np.ndarray, frame_id: int, mode: str, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if mode == "all":
        mask = np.ones((frame.shape[0],), dtype=bool)
    elif mode == "current":
        mask = frame == frame_id
    else:
        mask = frame <= frame_id
    xyz_sel = xyz[mask]
    rgb_sel = rgb[mask]
    if max_points > 0 and xyz_sel.shape[0] > max_points:
        idx = np.linspace(0, xyz_sel.shape[0] - 1, max_points).astype(np.int64)
        xyz_sel = xyz_sel[idx]
        rgb_sel = rgb_sel[idx]
    return xyz_sel, rgb_sel


def pack_rgb(rgb: np.ndarray) -> np.ndarray:
    packed = (
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32)
    )
    return packed.view(np.float32)


def main() -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    args = parse_args()
    map_path = args.map_dir.expanduser().resolve() / "live_map.npz"
    last_mtime = 0.0
    xyz, rgb, frame = load_cloud(args.map_dir, args.semantic_color)
    last_mtime = map_path.stat().st_mtime if map_path.exists() else 0.0
    frame_ids = np.unique(frame)
    frame_ids.sort()
    if frame_ids.size == 0:
        raise SystemExit("No points to publish")

    rclpy.init()
    node = Node("lingbot_live_map_publisher")
    pub = node.create_publisher(PointCloud2, args.topic, 2)
    plain_pub = node.create_publisher(PointCloud2, args.plain_topic, 2) if args.plain_topic else None
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    print(
        f"Publishing {xyz.shape[0]} points from {args.map_dir} to {args.topic} "
        f"mode={args.mode} fps={args.fps}"
        + (f" plain_topic={args.plain_topic}" if args.plain_topic else ""),
        flush=True,
    )
    period = 1.0 / max(0.1, float(args.fps))
    index = 0
    try:
        while rclpy.ok():
            if args.watch and map_path.exists():
                mtime = map_path.stat().st_mtime
                if mtime > last_mtime:
                    try:
                        xyz, rgb, frame = load_cloud(args.map_dir, args.semantic_color)
                        frame_ids = np.unique(frame)
                        frame_ids.sort()
                        index = min(index, max(0, frame_ids.size - 1))
                        last_mtime = mtime
                        print(f"reloaded {xyz.shape[0]} points from {map_path}", flush=True)
                    except Exception as exc:
                        print(f"Warning: failed to reload {map_path}: {type(exc).__name__}: {exc}", flush=True)
            if frame_ids.size == 0:
                time.sleep(period)
                continue
            frame_id = int(frame_ids[index])
            xyz_sel, rgb_sel = select_points(xyz, rgb, frame, frame_id, args.mode, args.max_points)
            cloud = np.zeros((xyz_sel.shape[0], 4), dtype=np.float32)
            cloud[:, :3] = xyz_sel
            cloud[:, 3] = pack_rgb(rgb_sel)
            header = Header()
            header.stamp = node.get_clock().now().to_msg()
            header.frame_id = args.frame_id
            msg = point_cloud2.create_cloud(
                header=header,
                fields=fields,
                points=cloud,
            )
            pub.publish(msg)
            if plain_pub is not None:
                plain_cloud = cloud.copy()
                plain_rgb = np.full((xyz_sel.shape[0], 3), 220, dtype=np.uint8)
                plain_cloud[:, 3] = pack_rgb(plain_rgb)
                plain_pub.publish(
                    point_cloud2.create_cloud(
                        header=header,
                        fields=fields,
                        points=plain_cloud,
                    )
                )
            print(f"frame={frame_id} published_points={xyz_sel.shape[0]}", flush=True)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
            index += 1
            if index >= frame_ids.size:
                if not args.loop:
                    break
                index = 0
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
