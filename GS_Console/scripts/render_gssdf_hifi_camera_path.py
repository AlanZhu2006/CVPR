#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Vec3:
    x: float
    y: float
    z: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a scripted camera path to the GS-SDF HiFi bridge."
    )
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8876")
    parser.add_argument("--duration", type=float, default=14.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--mode", choices=["orbit"], default="orbit")
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)
    parser.add_argument("--center-x", type=float, default=10.886779744651108)
    parser.add_argument("--center-y", type=float, default=8.276177854283173)
    parser.add_argument("--center-z", type=float, default=17.233192011804636)
    parser.add_argument("--radius", type=float, default=4.0)
    parser.add_argument("--height", type=float, default=8.0)
    parser.add_argument("--start-yaw-deg", type=float, default=130.0)
    parser.add_argument("--sweep-deg", type=float, default=220.0)
    parser.add_argument("--settle-seconds", type=float, default=1.2)
    return parser.parse_args()


def normalize(vec: Vec3) -> Vec3:
    length = math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)
    if length < 1e-8:
        return Vec3(0.0, 0.0, -1.0)
    return Vec3(vec.x / length, vec.y / length, vec.z / length)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )


def look_at_quaternion(eye: Vec3, target: Vec3) -> dict[str, float]:
    forward = normalize(Vec3(target.x - eye.x, target.y - eye.y, target.z - eye.z))
    up = Vec3(0.0, 0.0, 1.0)
    right = normalize(cross(forward, up))
    corrected_up = cross(right, forward)

    # Camera basis: x=right, y=up, z=-forward
    m00, m01, m02 = right.x, corrected_up.x, -forward.x
    m10, m11, m12 = right.y, corrected_up.y, -forward.y
    m20, m21, m22 = right.z, corrected_up.z, -forward.z

    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return {"x": x, "y": y, "z": z, "w": w}


def publish_pose(bridge_url: str, position: Vec3, target: Vec3) -> None:
    payload = json.dumps(
        {
            "position": {"x": position.x, "y": position.y, "z": position.z},
            "orientation": look_at_quaternion(position, target),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{bridge_url.rstrip('/')}/camera_pose",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


def ease_in_out(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def main() -> None:
    args = parse_args()
    target = Vec3(args.target_x, args.target_y, args.target_z)
    center = Vec3(args.center_x, args.center_y, args.center_z)
    frame_count = max(2, int(round(args.duration * args.fps)))
    start_angle = math.radians(args.start_yaw_deg)
    sweep = math.radians(args.sweep_deg)
    frame_interval = 1.0 / max(args.fps, 1.0)

    for _ in range(max(1, int(round(args.settle_seconds * args.fps)))):
        publish_pose(args.bridge_url, center, target)
        time.sleep(frame_interval)

    start_wall = time.perf_counter()
    for frame_index in range(frame_count):
        t = frame_index / max(frame_count - 1, 1)
        eased = ease_in_out(t)
        angle = start_angle + sweep * eased
        position = Vec3(
            target.x + math.cos(angle) * args.radius,
            target.y + math.sin(angle) * args.radius,
            target.z + args.height,
        )
        publish_pose(args.bridge_url, position, target)
        next_wall = start_wall + (frame_index + 1) * frame_interval
        delay = next_wall - time.perf_counter()
        if delay > 0:
            time.sleep(delay)

    for _ in range(max(1, int(round(args.settle_seconds * args.fps)))):
        publish_pose(args.bridge_url, position, target)
        time.sleep(frame_interval)


if __name__ == "__main__":
    main()
