from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract NTU VIRAL stereo images from a ROS1 bag into a cuVSLAM-friendly folder layout."
    )
    parser.add_argument("--bag", required=True, help="Path to NTU VIRAL ROS1 bag.")
    parser.add_argument("--output-dir", required=True, help="Output sequence directory.")
    parser.add_argument("--left-topic", default="/left/image_raw", help="ROS topic for the left image.")
    parser.add_argument("--right-topic", default="", help="ROS topic for the right image. Leave empty for mono.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit. 0 means all frames.",
    )
    parser.add_argument(
        "--pair-tolerance-sec",
        type=float,
        default=0.02,
        help="Maximum absolute time delta allowed when pairing right images to left images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2  # type: ignore

    bag_path = Path(args.bag).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    left_dir = output_dir / "image_0"
    right_dir = output_dir / "image_1"
    left_dir.mkdir(parents=True, exist_ok=True)
    if args.right_topic:
        right_dir.mkdir(parents=True, exist_ok=True)

    left_entries_raw: list[tuple[float, np.ndarray]] = []
    right_entries_raw: list[tuple[float, np.ndarray]] = []
    left_entries: list[tuple[float, str]] = []
    right_entries: list[tuple[float, str]] = []

    used_backend = ""
    try:
        import rosbag  # type: ignore
        from cv_bridge import CvBridge  # type: ignore

        bridge = CvBridge()
        used_backend = "rosbag"
        with rosbag.Bag(str(bag_path), "r") as bag:
            topics = [args.left_topic] + ([args.right_topic] if args.right_topic else [])
            for topic, msg, t in bag.read_messages(topics=topics):
                stamp = msg.header.stamp.to_sec() if getattr(msg, "header", None) else t.to_sec()
                if topic == args.left_topic:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                    left_entries_raw.append((stamp, image))
                elif args.right_topic and topic == args.right_topic:
                    image = bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                    right_entries_raw.append((stamp, image))
    except Exception:
        from rosbags.highlevel import AnyReader

        used_backend = "rosbags"
        with AnyReader([bag_path]) as reader:
            topics = {args.left_topic}
            if args.right_topic:
                topics.add(args.right_topic)
            connections = [conn for conn in reader.connections if conn.topic in topics]
            for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, conn.msgtype)
                stamp = float(timestamp_ns) * 1e-9
                height = int(msg.height)
                width = int(msg.width)
                encoding = str(msg.encoding)
                if encoding != "mono8":
                    raise SystemExit(
                        f"Only mono8 images are currently supported in rosbags fallback, got {encoding!r}"
                    )
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width)
                if conn.topic == args.left_topic:
                    left_entries_raw.append((stamp, image))
                elif args.right_topic and conn.topic == args.right_topic:
                    right_entries_raw.append((stamp, image))

    if not left_entries_raw:
        raise SystemExit(f"No frames extracted from {args.left_topic}")

    left_count = 0
    right_count = 0
    if args.right_topic:
        if not right_entries_raw:
            raise SystemExit(f"Requested right topic {args.right_topic} but no frames were extracted.")
        right_stamps = np.array([stamp for stamp, _ in right_entries_raw], dtype=np.float64)
        paired_left: list[tuple[float, np.ndarray]] = []
        paired_right: list[tuple[float, np.ndarray]] = []
        for left_stamp, left_image in left_entries_raw:
            nearest_idx = int(np.argmin(np.abs(right_stamps - left_stamp)))
            right_stamp, right_image = right_entries_raw[nearest_idx]
            if abs(right_stamp - left_stamp) > args.pair_tolerance_sec:
                continue
            paired_left.append((left_stamp, left_image))
            paired_right.append((right_stamp, right_image))
            if args.max_frames > 0 and len(paired_left) >= args.max_frames:
                break
        left_entries_raw = paired_left
        right_entries_raw = paired_right
        if not left_entries_raw:
            raise SystemExit(
                f"No stereo pairs found within tolerance {args.pair_tolerance_sec:.3f}s for {args.left_topic} / {args.right_topic}"
            )
    elif args.max_frames > 0:
        left_entries_raw = left_entries_raw[: args.max_frames]

    for stamp, image in left_entries_raw:
        name = f"{left_count:06d}.png"
        cv2.imwrite(str(left_dir / name), image)
        left_entries.append((stamp, name))
        left_count += 1
    if args.right_topic:
        for stamp, image in right_entries_raw:
            name = f"{right_count:06d}.png"
            cv2.imwrite(str(right_dir / name), image)
            right_entries.append((stamp, name))
            right_count += 1

    times_path = output_dir / "times.txt"
    timestamps_csv = output_dir / "timestamps.csv"
    with times_path.open("w", encoding="utf-8") as file:
        for stamp, _ in left_entries:
            file.write(f"{stamp - left_entries[0][0]:.9f}\n")

    with timestamps_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_idx", "left_stamp", "left_image", "right_stamp", "right_image"])
        for idx, (left_stamp, left_name) in enumerate(left_entries):
            if args.right_topic and idx < len(right_entries):
                right_stamp, right_name = right_entries[idx]
            else:
                right_stamp, right_name = "", ""
            writer.writerow([idx, f"{left_stamp:.9f}", left_name, right_stamp, right_name])

    summary = output_dir / "summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"bag={bag_path}",
                f"left_topic={args.left_topic}",
                f"right_topic={args.right_topic or '(mono)'}",
                f"left_frames={left_count}",
                f"right_frames={right_count}",
                f"backend={used_backend}",
                f"times={times_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
