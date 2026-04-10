from __future__ import annotations

import argparse
from pathlib import Path

from rosbags.highlevel import AnyReader


STEREO_TOPIC_CANDIDATES = (
    ("/cam0/image_raw", "/cam1/image_raw"),
    ("/cam0/image_raw/compressed", "/cam1/image_raw/compressed"),
    ("/camera/left/image_raw", "/camera/right/image_raw"),
    ("/camera/left/image_raw/compressed", "/camera/right/image_raw/compressed"),
    ("/stereo/left/image_raw", "/stereo/right/image_raw"),
    ("/stereo/left/image_raw/compressed", "/stereo/right/image_raw/compressed"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect rosbag topics and suggest stereo image pairs.")
    parser.add_argument("input", type=Path, help="Path to rosbag1 (*.bag) or rosbag2 directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with AnyReader([args.input]) as reader:
        print(f"input: {args.input}")
        print(f"messages: {reader.message_count}")
        print(f"start_time_ns: {reader.start_time}")
        print(f"end_time_ns: {reader.end_time}")
        print(f"duration_ns: {reader.duration}")
        print()
        print("topics:")
        for conn in sorted(reader.connections, key=lambda item: (item.topic, item.msgtype)):
            print(f"  {conn.topic} [{conn.msgtype}]")

        topic_names = {conn.topic for conn in reader.connections}
        suggestions = [pair for pair in STEREO_TOPIC_CANDIDATES if pair[0] in topic_names and pair[1] in topic_names]

        print()
        if suggestions:
            print("suggested_stereo_pairs:")
            for left_topic, right_topic in suggestions:
                print(f"  left={left_topic} right={right_topic}")
        else:
            print("suggested_stereo_pairs: none")


if __name__ == "__main__":
    main()
