#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GS-SDF Colmap parser compatibility files for NuRec-style generated datasets."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Dataset root containing images/depths/sparse/0.")
    source.add_argument("--scene-config", help="Scene config yaml containing data_path.")
    return parser.parse_args()


def dataset_from_scene_config(path: Path) -> Path:
    text = path.read_text()
    match = re.search(r'^data_path:\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"data_path not found in {path}")
    return Path(match.group(1))


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        previous = path.read_text() if path.is_file() and not path.is_symlink() else ""
        if path.is_symlink() or previous != content:
            path.unlink()
    path.write_text(content)


def clean_colmap_pose_rows(source: Path) -> list[str]:
    rows: list[str] = []
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10:
            rows.append(" ".join(parts))
    return rows


def quat_wxyz_to_rot(qw: float, qx: float, qy: float, qz: float) -> tuple[tuple[float, float, float], ...]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        raise ValueError("invalid zero quaternion")
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


def rot_to_quat_wxyz(rot: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float, float]:
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2][1] - rot[1][2]) / scale
        qy = (rot[0][2] - rot[2][0]) / scale
        qz = (rot[1][0] - rot[0][1]) / scale
    elif rot[0][0] > rot[1][1] and rot[0][0] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[0][0] - rot[1][1] - rot[2][2]) * 2.0
        qw = (rot[2][1] - rot[1][2]) / scale
        qx = 0.25 * scale
        qy = (rot[0][1] + rot[1][0]) / scale
        qz = (rot[0][2] + rot[2][0]) / scale
    elif rot[1][1] > rot[2][2]:
        scale = math.sqrt(1.0 + rot[1][1] - rot[0][0] - rot[2][2]) * 2.0
        qw = (rot[0][2] - rot[2][0]) / scale
        qx = (rot[0][1] + rot[1][0]) / scale
        qy = 0.25 * scale
        qz = (rot[1][2] + rot[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rot[2][2] - rot[0][0] - rot[1][1]) * 2.0
        qw = (rot[1][0] - rot[0][1]) / scale
        qx = (rot[0][2] + rot[2][0]) / scale
        qy = (rot[1][2] + rot[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm


def invert_colmap_w2c_to_c2w_row(parts: list[str]) -> str:
    item_id = int(parts[0])
    qw, qx, qy, qz = (float(value) for value in parts[1:5])
    tx, ty, tz = (float(value) for value in parts[5:8])
    rot_w2c = quat_wxyz_to_rot(qw, qx, qy, qz)
    rot_c2w = tuple(tuple(rot_w2c[col][row] for col in range(3)) for row in range(3))
    c2w_t = tuple(
        -sum(rot_c2w[row][col] * value for col, value in enumerate((tx, ty, tz)))
        for row in range(3)
    )
    c2w_q = rot_to_quat_wxyz(rot_c2w)
    values = (item_id, *c2w_q, *c2w_t)
    return " ".join(f"{value:.17g}" if isinstance(value, float) else str(value) for value in values)


def clean_depth_pose_rows(source: Path) -> list[str]:
    rows: list[str] = []
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 8:
            # GS-SDF's Colmap parser loads depth pose_type=5 without inverse,
            # so this file must be camera-to-world even when the source is
            # COLMAP's world-to-camera images/depths format.
            rows.append(invert_colmap_w2c_to_c2w_row(parts))
    return rows


def read_ascii_ply_xyz(path: Path) -> list[tuple[float, float, float]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        vertex_count = None
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Missing vertex count in {path}")
        points: list[tuple[float, float, float]] = []
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return points


def write_ascii_pcd(path: Path, points: list[tuple[float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z\n")
        handle.write("SIZE 4 4 4\n")
        handle.write("TYPE F F F\n")
        handle.write("COUNT 1 1 1\n")
        handle.write(f"WIDTH {len(points)}\n")
        handle.write("HEIGHT 1\n")
        handle.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {len(points)}\n")
        handle.write("DATA ascii\n")
        for x, y, z in points:
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def ensure_symlink(link: Path, target: str) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    print(f"Created compatibility link: {link} -> {target}")


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset) if args.dataset else dataset_from_scene_config(Path(args.scene_config))
    if not dataset.is_dir():
        raise SystemExit(f"Dataset does not exist: {dataset}")

    ensure_symlink(dataset / "colmap/images", "../images")
    ensure_symlink(dataset / "colmap/sparse", "../sparse")
    ensure_symlink(dataset / "colmap/depths", "../depths")
    ensure_symlink(dataset / "colmap/postrior_lidar/cameras.txt", "../../sparse/0/cameras.txt")

    images_txt = dataset / "sparse/0/images.txt"
    depths_txt = dataset / "sparse/0/depths.txt"
    if images_txt.is_file():
        rows = clean_colmap_pose_rows(images_txt)
        # Pose type 4 uses skip_line=true, so preserve the two-line COLMAP cadence.
        atomic_write_text(dataset / "colmap/postrior_lidar/images.txt", "\n\n".join(rows) + ("\n\n" if rows else ""))
        print(f"Wrote color pose compatibility file with {len(rows)} poses.")
    if depths_txt.is_file():
        rows = clean_depth_pose_rows(depths_txt)
        content = "\n".join(rows) + ("\n" if rows else "")
        atomic_write_text(dataset / "depths/lidar_pose.txt", content)
        atomic_write_text(dataset / "colmap/postrior_lidar/depths.txt", content)
        print(f"Wrote depth pose compatibility files with {len(rows)} poses.")

    depth_dir = dataset / "depths"
    if depth_dir.is_dir() and not any(depth_dir.glob("*.pcd")):
        converted = 0
        for ply_path in sorted(depth_dir.glob("*.ply")):
            points = read_ascii_ply_xyz(ply_path)
            write_ascii_pcd(ply_path.with_suffix(".pcd"), points)
            converted += 1
        print(f"Converted {converted} depth PLY files to PCD.")


if __name__ == "__main__":
    main()
