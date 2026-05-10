#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


OPENCV_TO_OPENGL_CAMERA_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or publish GS-SDF renderer poses from the original training camera path."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="GS-SDF dataset root containing colmap/postrior_lidar/images.txt.")
    source.add_argument("--scene-config", help="GS-SDF scene config yaml containing data_path.")
    source.add_argument("--output-dir", help="GS-SDF output dir containing model/config/scene/config.yaml.")
    parser.add_argument("--frame-index", type=int, default=0, help="Zero-based frame index in the selected split.")
    parser.add_argument("--split", choices=("train", "raw"), default="train")
    parser.add_argument("--target-distance", type=float, default=4.0)
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--post-url", default="", help="Optional bridge/renderer base URL. Posts to /camera_pose.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--repeat-interval-sec", type=float, default=0.08)
    parser.add_argument("--print-max-distance", action="store_true")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset:
        return Path(args.dataset)
    if args.output_dir:
        config_path = Path(args.output_dir) / "model/config/scene/config.yaml"
    else:
        config_path = Path(args.scene_config)
    text = config_path.read_text()
    match = re.search(r'^data_path:\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"data_path not found in {config_path}")
    return Path(match.group(1))


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ),
        (
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qw * qx),
        ),
        (
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def mat3_transpose_vec3(matrix: tuple[tuple[float, float, float], ...], vector: tuple[float, float, float]):
    return (
        matrix[0][0] * vector[0] + matrix[1][0] * vector[1] + matrix[2][0] * vector[2],
        matrix[0][1] * vector[0] + matrix[1][1] * vector[1] + matrix[2][1] * vector[2],
        matrix[0][2] * vector[0] + matrix[1][2] * vector[1] + matrix[2][2] * vector[2],
    )


def mat3_multiply(left: tuple[tuple[float, float, float], ...], right: tuple[tuple[float, float, float], ...]):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def rotation_matrix_to_quaternion_xyzw(matrix: tuple[tuple[float, float, float], ...]) -> dict[str, float]:
    m = matrix
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m[2][1] - m[1][2]) / scale
        qy = (m[0][2] - m[2][0]) / scale
        qz = (m[1][0] - m[0][1]) / scale
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        scale = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / scale
        qx = 0.25 * scale
        qy = (m[0][1] + m[1][0]) / scale
        qz = (m[0][2] + m[2][0]) / scale
    elif m[1][1] > m[2][2]:
        scale = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / scale
        qx = (m[0][1] + m[1][0]) / scale
        qy = 0.25 * scale
        qz = (m[1][2] + m[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / scale
        qx = (m[0][2] + m[2][0]) / scale
        qy = (m[1][2] + m[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return {"x": qx / norm, "y": qy / norm, "z": qz / norm, "w": qw / norm}


def parse_colmap_image_rows(images_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in images_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = (float(parts[index]) for index in range(1, 5))
            tx, ty, tz = (float(parts[index]) for index in range(5, 8))
            camera_id = int(parts[8])
        except ValueError:
            continue
        rows.append(
            {
                "imageId": image_id,
                "cameraId": camera_id,
                "q": (qw, qx, qy, qz),
                "t": (tx, ty, tz),
                "imageName": parts[9],
            }
        )
    return rows


def training_rows(dataset_path: Path, split: str) -> tuple[Path, list[dict[str, Any]]]:
    candidates = [
        dataset_path / "colmap/postrior_lidar/images.txt",
        dataset_path / "sparse/0/images.txt",
    ]
    images_path = next((path for path in candidates if path.is_file()), None)
    if images_path is None:
        raise RuntimeError(f"No COLMAP images.txt found under {dataset_path}")
    rows = parse_colmap_image_rows(images_path)
    if split == "train":
        rows = [row for one_based, row in enumerate(rows, start=1) if one_based % 8 != 0]
    if not rows:
        raise RuntimeError(f"No {split} camera rows found in {images_path}")
    return images_path, rows


def build_pose(row: dict[str, Any], *, frame_id: str, target_distance: float) -> dict[str, Any]:
    qw, qx, qy, qz = row["q"]
    tx, ty, tz = row["t"]
    world_to_camera = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    camera_center = mat3_transpose_vec3(world_to_camera, (-tx, -ty, -tz))
    camera_to_world = tuple(tuple(world_to_camera[col][row_index] for col in range(3)) for row_index in range(3))
    renderer_input_rotation = mat3_multiply(camera_to_world, OPENCV_TO_OPENGL_CAMERA_ROTATION)
    forward = (camera_to_world[0][2], camera_to_world[1][2], camera_to_world[2][2])
    pose = {
        "frameId": frame_id,
        "position": {
            "x": camera_center[0],
            "y": camera_center[1],
            "z": camera_center[2],
        },
        "orientation": rotation_matrix_to_quaternion_xyzw(renderer_input_rotation),
    }
    target = {
        "x": camera_center[0] + forward[0] * target_distance,
        "y": camera_center[1] + forward[1] * target_distance,
        "z": camera_center[2] + forward[2] * target_distance,
    }
    return {"pose": pose, "target": target, "sourceImage": row["imageName"]}


def post_camera_pose(base_url: str, pose: dict[str, Any], *, timeout_sec: float = 5.0) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/camera_pose"
    payload = json.dumps({"position": pose["position"], "orientation": pose["orientation"]}).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8", errors="ignore")
        return json.loads(body or "{}")


def main() -> int:
    args = parse_args()
    dataset_path = resolve_dataset_path(args)
    images_path, rows = training_rows(dataset_path, args.split)
    if args.print_max_distance:
        distances = []
        for row in rows:
            built = build_pose(row, frame_id=args.frame_id, target_distance=args.target_distance)
            position = built["pose"]["position"]
            distances.append(math.sqrt(position["x"] ** 2 + position["y"] ** 2 + position["z"] ** 2))
        print(max(distances) if distances else 0.0)
        return 0
    if args.frame_index < 0 or args.frame_index >= len(rows):
        raise RuntimeError(f"frame-index {args.frame_index} out of range for {len(rows)} {args.split} rows")
    built = build_pose(rows[args.frame_index], frame_id=args.frame_id, target_distance=args.target_distance)
    built.update(
        {
            "dataset": str(dataset_path),
            "imagesPath": str(images_path),
            "split": args.split,
            "frameIndex": args.frame_index,
        }
    )
    post_status = None
    if args.post_url:
        repeat = max(1, int(args.repeat))
        for index in range(repeat):
            post_status = post_camera_pose(args.post_url, built["pose"])
            if index + 1 < repeat:
                time.sleep(max(0.0, float(args.repeat_interval_sec)))
        built["postUrl"] = args.post_url.rstrip("/")
        built["postStatus"] = post_status
    print(json.dumps(built, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"gssdf_training_camera_pose: {exc}", file=sys.stderr)
        raise SystemExit(1)
