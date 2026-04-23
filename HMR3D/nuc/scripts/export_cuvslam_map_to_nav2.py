from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CUVSLAM_FROM_ROS = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project a cuVSLAM sparse map into a rough 2D Nav2 occupancy map."
    )
    parser.add_argument("--map-json", required=True, help="Path to map.json from map_extractor.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for map.pgm and map.yaml.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.10,
        help="Map resolution in meters/pixel.",
    )
    parser.add_argument(
        "--padding-m",
        type=float,
        default=5.0,
        help="Padding around the projected map in meters.",
    )
    parser.add_argument(
        "--height-min",
        type=float,
        default=-2.0,
        help="Min landmark height relative to median pose height to keep.",
    )
    parser.add_argument(
        "--height-max",
        type=float,
        default=2.0,
        help="Max landmark height relative to median pose height to keep.",
    )
    parser.add_argument(
        "--occupied-radius-cells",
        type=int,
        default=1,
        help="Dilation radius for occupied landmark cells.",
    )
    parser.add_argument(
        "--free-radius-cells",
        type=int,
        default=2,
        help="Radius used to carve free space around the camera path.",
    )
    parser.add_argument(
        "--free-line-step-m",
        type=float,
        default=0.20,
        help="Spacing used when rasterizing free-space along the pose path.",
    )
    return parser.parse_args()


def cuvslam_to_ros_xyz(points: np.ndarray) -> np.ndarray:
    return points @ CUVSLAM_FROM_ROS


def load_map(map_json: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(map_json.read_text(encoding="utf-8"))

    pose_xyz = np.array(
        [
            [
                pose["translation"]["x"],
                pose["translation"]["y"],
                pose["translation"]["z"],
            ]
            for pose in data.get("poses", [])
        ],
        dtype=np.float64,
    )
    landmark_xyz = np.array(
        [
            [
                landmark["pose"]["x"],
                landmark["pose"]["y"],
                landmark["pose"]["z"],
            ]
            for landmark in data.get("landmarks", [])
        ],
        dtype=np.float64,
    )

    if pose_xyz.size == 0 or landmark_xyz.size == 0:
        raise ValueError("map.json must contain both poses and landmarks")

    return cuvslam_to_ros_xyz(pose_xyz), cuvslam_to_ros_xyz(landmark_xyz)


def world_to_grid(
    xy: np.ndarray,
    origin_xy: np.ndarray,
    resolution: float,
    height: int,
) -> np.ndarray:
    gx = np.floor((xy[:, 0] - origin_xy[0]) / resolution).astype(np.int32)
    gy = np.floor((xy[:, 1] - origin_xy[1]) / resolution).astype(np.int32)
    return np.stack([gx, height - 1 - gy], axis=1)


def disk_offsets(radius: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                offsets.append((dy, dx))
    return offsets


def stamp_disk(image: np.ndarray, center_rc: np.ndarray, value: int, offsets: list[tuple[int, int]]) -> None:
    row, col = int(center_rc[1]), int(center_rc[0])
    h, w = image.shape
    for dy, dx in offsets:
        rr = row + dy
        cc = col + dx
        if 0 <= rr < h and 0 <= cc < w:
            image[rr, cc] = value


def draw_free_path(
    image: np.ndarray,
    pose_xy: np.ndarray,
    origin_xy: np.ndarray,
    resolution: float,
    free_radius: int,
    free_line_step_m: float,
) -> None:
    offsets = disk_offsets(free_radius)
    height = image.shape[0]
    for idx in range(len(pose_xy) - 1):
        start = pose_xy[idx]
        end = pose_xy[idx + 1]
        segment = end - start
        length = float(np.linalg.norm(segment))
        steps = max(1, int(math.ceil(length / free_line_step_m)))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            sample = start + alpha * segment
            grid = world_to_grid(sample[None, :], origin_xy, resolution, height)[0]
            stamp_disk(image, grid, 254, offsets)


def draw_occupied(
    image: np.ndarray,
    landmark_xy: np.ndarray,
    origin_xy: np.ndarray,
    resolution: float,
    occupied_radius: int,
) -> None:
    offsets = disk_offsets(occupied_radius)
    height = image.shape[0]
    grid_points = world_to_grid(landmark_xy, origin_xy, resolution, height)
    for grid in grid_points:
        stamp_disk(image, grid, 0, offsets)


def write_pgm(path: Path, image: np.ndarray) -> None:
    header = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        file.write(image.astype(np.uint8).tobytes())


def write_yaml(path: Path, image_name: str, resolution: float, origin_xy: np.ndarray) -> None:
    yaml_text = (
        f"image: {image_name}\n"
        f"resolution: {resolution:.6f}\n"
        f"origin: [{origin_xy[0]:.6f}, {origin_xy[1]:.6f}, 0.000000]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n"
        "mode: trinary\n"
    )
    path.write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    map_json = Path(args.map_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_xyz, landmark_xyz = load_map(map_json)

    pose_height_median = float(np.median(pose_xyz[:, 2]))
    min_height = pose_height_median + args.height_min
    max_height = pose_height_median + args.height_max

    landmark_mask = (landmark_xyz[:, 2] >= min_height) & (landmark_xyz[:, 2] <= max_height)
    filtered_landmarks = landmark_xyz[landmark_mask]
    if filtered_landmarks.size == 0:
        raise ValueError("All landmarks were filtered out. Try widening height range.")

    pose_xy = pose_xyz[:, :2]
    landmark_xy = filtered_landmarks[:, :2]

    all_xy = np.concatenate([pose_xy, landmark_xy], axis=0)
    min_xy = np.min(all_xy, axis=0) - args.padding_m
    max_xy = np.max(all_xy, axis=0) + args.padding_m

    width = int(math.ceil((max_xy[0] - min_xy[0]) / args.resolution)) + 1
    height = int(math.ceil((max_xy[1] - min_xy[1]) / args.resolution)) + 1

    image = np.full((height, width), 205, dtype=np.uint8)

    draw_free_path(
        image=image,
        pose_xy=pose_xy,
        origin_xy=min_xy,
        resolution=args.resolution,
        free_radius=args.free_radius_cells,
        free_line_step_m=args.free_line_step_m,
    )
    draw_occupied(
        image=image,
        landmark_xy=landmark_xy,
        origin_xy=min_xy,
        resolution=args.resolution,
        occupied_radius=args.occupied_radius_cells,
    )

    pgm_path = output_dir / "map.pgm"
    yaml_path = output_dir / "map.yaml"
    write_pgm(pgm_path, image)
    write_yaml(yaml_path, pgm_path.name, args.resolution, min_xy)

    summary = {
        "pose_count": int(pose_xyz.shape[0]),
        "landmark_count": int(landmark_xyz.shape[0]),
        "filtered_landmark_count": int(filtered_landmarks.shape[0]),
        "resolution": args.resolution,
        "size": [int(width), int(height)],
        "origin_xy": [float(min_xy[0]), float(min_xy[1])],
        "height_filter": [float(min_height), float(max_height)],
        "pgm_path": str(pgm_path),
        "yaml_path": str(yaml_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
