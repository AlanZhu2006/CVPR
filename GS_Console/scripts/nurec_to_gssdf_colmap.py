#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class TumPose:
    timestamp: float
    c2w: np.ndarray


@dataclass(frozen=True)
class FrameMatch:
    image_path: Path
    pose: TumPose


@dataclass(frozen=True)
class CameraSpec:
    camera_id: int
    camera_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    transform: np.ndarray
    calibration_source: str
    calibration_path: str
    calibration_key: str


@dataclass(frozen=True)
class PlyMesh:
    vertices: np.ndarray
    faces: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert NuRec-style images + TUM trajectory + optional depth/mesh "
            "geometry into the GS-SDF colmap parser layout."
        )
    )
    parser.add_argument("--input-root", required=True, help="NuRec scene root or dataset root.")
    parser.add_argument("--output-root", required=True, help="Output GS-SDF colmap-style dataset root.")
    parser.add_argument(
        "--image-dir",
        required=True,
        help="Image directory, absolute or relative to --input-root.",
    )
    parser.add_argument(
        "--tum-path",
        required=True,
        help="TUM trajectory file, absolute or relative to --input-root. TUM is timestamp tx ty tz qx qy qz qw.",
    )
    parser.add_argument(
        "--depth-ply-dir",
        default="",
        help="Optional per-frame local point cloud PLY directory, absolute or relative to --input-root.",
    )
    parser.add_argument(
        "--mesh-ply",
        default="",
        help=(
            "Optional world-frame mesh/point-cloud PLY. If --depth-ply-dir is omitted, "
            "visible/cropped mesh vertices are transformed into local camera PLYs."
        ),
    )
    parser.add_argument(
        "--camera-extrinsics-json",
        default="",
        help=(
            "Optional NuRec stereo.edex-style JSON. Used as a fallback when "
            "--calibration-source is edex or auto cannot find 3dgrt/last.usdz."
        ),
    )
    parser.add_argument(
        "--calibration-source",
        choices=("auto", "3dgrt", "edex", "manual"),
        default="auto",
        help=(
            "Camera calibration source. auto prefers NuRec 3dgrt/last.usdz "
            "rig_trajectories.json, then falls back to --camera-extrinsics-json."
        ),
    )
    parser.add_argument(
        "--nurec-last-usdz",
        default="",
        help=(
            "Optional path to NuRec 3dgrt/last.usdz or extracted rig_trajectories.json. "
            "Defaults to --input-root/3dgrt/last.usdz when available."
        ),
    )
    parser.add_argument(
        "--camera-name",
        default="front_stereo_camera_left",
        help="Logical camera name for NuRec 3DGRT calibration.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=-1,
        help="Camera id fallback inside --camera-extrinsics-json or 3DGRT unique_sensor_idx.",
    )
    parser.add_argument(
        "--tum-pose-frame",
        choices=("camera", "ego"),
        default="camera",
        help="Whether TUM poses are already camera-to-world or ego/rig-to-world.",
    )
    parser.add_argument(
        "--camera-extrinsic-direction",
        choices=("camera-to-ego", "ego-to-camera"),
        default="camera-to-ego",
        help="Convention of the selected extrinsic matrix in --camera-extrinsics-json.",
    )
    parser.add_argument(
        "--camera-frame-axis",
        choices=("as-is", "vehicle-forward-to-opencv"),
        default="as-is",
        help=(
            "Axis convention of the selected camera extrinsic. Use "
            "vehicle-forward-to-opencv when the extrinsic rotation is body-like "
            "x-forward/y-left/z-up and image rays need OpenCV x-right/y-down/z-forward."
        ),
    )
    parser.add_argument("--width", type=int, default=0, help="Image width. Inferred from first image if omitted.")
    parser.add_argument("--height", type=int, default=0, help="Image height. Inferred from first image if omitted.")
    parser.add_argument("--fx", type=float, default=0.0, required=False)
    parser.add_argument("--fy", type=float, default=0.0, required=False)
    parser.add_argument("--cx", type=float, default=0.0, required=False)
    parser.add_argument("--cy", type=float, default=0.0, required=False)
    parser.add_argument(
        "--match-mode",
        choices=("auto", "index", "timestamp"),
        default="auto",
        help="How to match images to TUM rows. auto uses timestamp when image stems parse as floats.",
    )
    parser.add_argument(
        "--timestamp-tolerance-sec",
        type=float,
        default=0.05,
        help="Maximum image/TUM timestamp difference in timestamp mode.",
    )
    parser.add_argument(
        "--image-timestamp-scale",
        default="auto",
        help=(
            "Scale image filename timestamps into seconds. Use auto, 1, 1e-3, "
            "1e-6, or 1e-9. auto treats Unix nanosecond/microsecond/millisecond "
            "stems as needed."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth matched frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap after stride.")
    parser.add_argument(
        "--copy-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize images/depth PLYs in the output directory.",
    )
    parser.add_argument(
        "--image-output-ext",
        default="",
        help="Force output image suffix, for example .png. Default keeps source suffix.",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=40.0)
    parser.add_argument("--sdf-iter-step", type=int, default=5000)
    parser.add_argument("--gs-iter-step", type=int, default=30000)
    parser.add_argument("--map-size", type=int, default=300)
    parser.add_argument("--leaf-size", type=float, default=0.2)
    parser.add_argument("--ds-pt-num", type=int, default=20000)
    parser.add_argument("--background", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument(
        "--mesh-depth-mode",
        choices=("vertices", "surface"),
        default="vertices",
        help="Use raw mesh vertices or sampled mesh triangle surfaces to synthesize local depth PLYs.",
    )
    parser.add_argument(
        "--mesh-surface-samples",
        type=int,
        default=0,
        help="Number of world-frame mesh surface points to pre-sample in surface mode. 0 chooses an automatic value.",
    )
    parser.add_argument(
        "--zbuffer-pixel-stride",
        type=int,
        default=4,
        help="Quantized pixel stride for z-buffering synthetic mesh depth. 0 disables z-buffering.",
    )
    parser.add_argument(
        "--points-per-depth",
        type=int,
        default=20000,
        help="Maximum local points per generated depth PLY when using --mesh-ply.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for mesh point subsampling.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def list_images(image_dir: Path) -> list[Path]:
    images = [path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda path: natural_image_key(path))


def natural_image_key(path: Path) -> tuple[int, float | str, str]:
    stem = path.stem
    try:
        return (0, float(stem), path.name)
    except ValueError:
        return (1, stem, path.name)


def infer_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Pillow is required to infer image size. Install Pillow or pass --width/--height."
        ) from exc

    with Image.open(path) as image:
        return image.size


def read_tum(path: Path) -> list[TumPose]:
    poses: list[TumPose] = []
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 8:
            raise ValueError(f"{path}:{line_no}: expected 8 TUM values, got {len(values)}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, values)
        c2w = pose_matrix_from_t_q(np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
        poses.append(TumPose(timestamp=timestamp, c2w=c2w))
    if not poses:
        raise ValueError(f"No poses loaded from {path}")
    return poses


def load_camera_spec(path: Path, camera_id: int) -> CameraSpec:
    if camera_id < 0:
        raise ValueError("--camera-id is required when --camera-extrinsics-json is set")
    data = json.loads(path.read_text())
    groups = data if isinstance(data, list) else [data]
    for group in groups:
        if not isinstance(group, dict):
            continue
        for camera in group.get("cameras", []):
            if int(camera.get("id", -1)) != camera_id:
                continue
            intrinsics = camera["intrinsics"]
            focal = intrinsics["focal"]
            principal = intrinsics["principal"]
            size = intrinsics["size"]
            transform = normalize_transform(camera["transform"], f"{path}: camera id {camera_id}")
            return CameraSpec(
                camera_id=camera_id,
                camera_name=str(camera.get("name", "")),
                width=int(size[0]),
                height=int(size[1]),
                fx=float(focal[0]),
                fy=float(focal[1]),
                cx=float(principal[0]),
                cy=float(principal[1]),
                transform=transform,
                calibration_source="edex",
                calibration_path=str(path),
                calibration_key=f"camera_id={camera_id}",
            )
    raise ValueError(f"Camera id {camera_id} not found in {path}")


def normalize_transform(raw_transform: Any, context: str) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    raw = np.asarray(raw_transform, dtype=np.float64)
    if raw.shape == (3, 4):
        transform[:3, :] = raw
    elif raw.shape == (4, 4):
        transform = raw
    else:
        raise ValueError(f"Unsupported camera transform shape {raw.shape}: {context}")
    return transform


def load_3dgrt_camera_spec(path: Path, camera_name: str, camera_id: int) -> CameraSpec:
    data = load_3dgrt_rig_json(path)
    calibrations = data.get("camera_calibrations")
    if not isinstance(calibrations, dict) or not calibrations:
        raise ValueError(f"No camera_calibrations found in {path}")

    selected_key, selected = select_3dgrt_camera(calibrations, camera_name, camera_id)
    params = selected["camera_model"]["parameters"]
    resolution = params.get("resolution")
    focal = params.get("focal_length") or params.get("focal")
    principal = params.get("principal_point") or params.get("principal")
    if not resolution or not focal or not principal:
        raise ValueError(f"3DGRT camera calibration is missing intrinsics: {selected_key}")

    unique_sensor_idx = int(selected.get("unique_sensor_idx", camera_id))
    logical_name = str(selected.get("logical_sensor_name", camera_name or selected_key.split("@", 1)[0]))
    return CameraSpec(
        camera_id=unique_sensor_idx,
        camera_name=logical_name,
        width=int(resolution[0]),
        height=int(resolution[1]),
        fx=float(focal[0]),
        fy=float(focal[1]),
        cx=float(principal[0]),
        cy=float(principal[1]),
        transform=normalize_transform(selected["T_sensor_rig"], f"{path}: {selected_key}.T_sensor_rig"),
        calibration_source="3dgrt",
        calibration_path=str(path),
        calibration_key=selected_key,
    )


def load_3dgrt_rig_json(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix == ".usdz":
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist() if name.endswith("rig_trajectories.json")]
            if not candidates:
                raise ValueError(f"rig_trajectories.json not found inside {path}")
            return json.loads(archive.read(candidates[0]))
    raise ValueError(f"Unsupported 3DGRT calibration path: {path}")


def select_3dgrt_camera(
    calibrations: dict[str, Any],
    camera_name: str,
    camera_id: int,
) -> tuple[str, dict[str, Any]]:
    if camera_name:
        name_matches = []
        for key, camera in calibrations.items():
            logical_name = str(camera.get("logical_sensor_name", ""))
            if key == camera_name or key.startswith(f"{camera_name}@") or logical_name == camera_name:
                name_matches.append((key, camera))
        if name_matches:
            return sorted(name_matches, key=lambda item: item[0])[0]

    if camera_id >= 0:
        id_matches = []
        for key, camera in calibrations.items():
            try:
                unique_sensor_idx = int(camera.get("unique_sensor_idx", -1))
            except (TypeError, ValueError):
                continue
            if unique_sensor_idx == camera_id:
                id_matches.append((key, camera))
        if id_matches:
            return sorted(id_matches, key=lambda item: item[0])[0]

    if len(calibrations) == 1:
        key = next(iter(calibrations))
        return key, calibrations[key]

    available = ", ".join(sorted(calibrations))
    raise ValueError(
        f"Could not select 3DGRT camera name={camera_name!r} id={camera_id}. Available: {available}"
    )


def resolve_camera_calibration(input_root: Path, args: argparse.Namespace) -> tuple[CameraSpec | None, dict[str, Any]]:
    metadata: dict[str, Any] = {
        "requestedSource": args.calibration_source,
        "requestedCameraName": args.camera_name,
        "requestedCameraId": args.camera_id if args.camera_id >= 0 else None,
    }

    if args.calibration_source in ("auto", "3dgrt"):
        candidates = (
            [resolve_path(input_root, args.nurec_last_usdz)]
            if args.nurec_last_usdz
            else [
                input_root / "3dgrt" / "last.usdz",
                input_root / "last.usdz",
                input_root / "rig_trajectories.json",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                spec = load_3dgrt_camera_spec(candidate, args.camera_name, args.camera_id)
                metadata.update(
                    {
                        "source": "3dgrt",
                        "path": str(candidate),
                        "key": spec.calibration_key,
                        "cameraName": spec.camera_name,
                        "cameraId": spec.camera_id,
                        "poseConvention": {
                            "tumPoseFrame": "ego",
                            "cameraExtrinsicDirection": "camera-to-ego",
                            "cameraFrameAxis": "as-is",
                        },
                    }
                )
                return spec, metadata
        if args.calibration_source == "3dgrt":
            searched = ", ".join(str(path) for path in candidates)
            raise SystemExit(f"3DGRT calibration requested but not found. Searched: {searched}")

    if args.calibration_source in ("auto", "edex") and args.camera_extrinsics_json:
        path = resolve_path(input_root, args.camera_extrinsics_json)
        spec = load_camera_spec(path, args.camera_id)
        metadata.update(
            {
                "source": "edex",
                "path": str(path),
                "key": spec.calibration_key,
                "cameraName": spec.camera_name,
                "cameraId": spec.camera_id,
            }
        )
        return spec, metadata

    if args.calibration_source == "edex":
        raise SystemExit("--calibration-source=edex requires --camera-extrinsics-json and --camera-id")

    metadata["source"] = "manual"
    return None, metadata


def effective_pose_convention(args: argparse.Namespace, camera_spec: CameraSpec | None) -> tuple[str, str, str]:
    if camera_spec is not None and camera_spec.calibration_source == "3dgrt":
        return "ego", "camera-to-ego", "as-is"
    return args.tum_pose_frame, args.camera_extrinsic_direction, args.camera_frame_axis


def pose_matrix_from_t_q(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    q = q_xyzw.astype(np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError("Invalid zero quaternion")
    x, y, z, w = q / norm
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = t.astype(np.float64)
    return mat


def matrix_to_colmap_qt(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    rot = w2c[:3, :3]
    trans = w2c[:3, 3]
    q_wxyz = rotation_matrix_to_quaternion_wxyz(rot)
    return q_wxyz, trans


def rotation_matrix_to_quaternion_wxyz(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    return q / np.linalg.norm(q)


def camera_c2w_from_pose(
    pose: TumPose,
    camera_spec: CameraSpec | None,
    tum_pose_frame: str,
    extrinsic_direction: str,
    camera_frame_axis: str,
) -> np.ndarray:
    if tum_pose_frame == "camera":
        return pose.c2w
    if camera_spec is None:
        raise ValueError("--camera-extrinsics-json is required when --tum-pose-frame=ego")
    transform = camera_spec.transform
    if camera_frame_axis == "vehicle-forward-to-opencv":
        transform = transform @ vehicle_forward_t_opencv()
    if extrinsic_direction == "camera-to-ego":
        ego_t_camera = transform
    else:
        ego_t_camera = np.linalg.inv(transform)
    return pose.c2w @ ego_t_camera


def vehicle_forward_t_opencv() -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return mat


def image_timestamp_to_seconds(value: float, scale: str) -> float:
    if scale != "auto":
        return value * float(scale)
    abs_value = abs(value)
    if abs_value > 1.0e17:
        return value * 1.0e-9
    if abs_value > 1.0e14:
        return value * 1.0e-6
    if abs_value > 1.0e11:
        return value * 1.0e-3
    return value


def match_frames(
    images: list[Path],
    poses: list[TumPose],
    mode: str,
    tolerance: float,
    timestamp_scale: str,
) -> list[FrameMatch]:
    if mode == "auto":
        numeric_count = sum(1 for path in images if parse_float_or_none(path.stem) is not None)
        mode = "timestamp" if numeric_count == len(images) else "index"

    if mode == "index":
        count = min(len(images), len(poses))
        return [FrameMatch(images[index], poses[index]) for index in range(count)]

    pose_times = np.array([pose.timestamp for pose in poses], dtype=np.float64)
    matches: list[FrameMatch] = []
    for image in images:
        image_time_raw = parse_float_or_none(image.stem)
        image_time = (
            image_timestamp_to_seconds(image_time_raw, timestamp_scale)
            if image_time_raw is not None
            else None
        )
        if image_time is None:
            raise ValueError(f"Image stem is not a timestamp: {image.name}")
        nearest = int(np.argmin(np.abs(pose_times - image_time)))
        diff = abs(float(pose_times[nearest]) - image_time)
        if diff <= tolerance:
            matches.append(FrameMatch(image, poses[nearest]))
    return matches


def parse_float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported copy mode: {mode}")


def write_cameras_txt(path: Path, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Camera list with one line of data per camera:",
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
                "# Number of cameras: 1",
                f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}",
                "",
            ]
        )
    )


def write_pose_txt(path: Path, rows: Iterable[tuple[int, np.ndarray, np.ndarray, str]], label: str) -> None:
    lines = [
        f"# {label} list with two lines per entry:",
        "#   ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] or empty line",
    ]
    count = 0
    for item_id, quat, trans, filename in rows:
        count += 1
        values = [
            str(item_id),
            *(f"{value:.17g}" for value in quat),
            *(f"{value:.17g}" for value in trans),
            "1",
            filename,
        ]
        lines.append(" ".join(values))
        lines.append("")
    lines.insert(2, f"# Number of {label.lower()}s: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_gssdf_config(
    path: Path,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color_type: str,
    args: argparse.Namespace,
) -> None:
    path.write_text(
        f"""%YAML:1.0

base_config: "/root/gs_sdf_ws/src/GS-SDF/config/base.yaml"

sdf_iter_step: {args.sdf_iter_step}
gs_iter_step: {args.gs_iter_step}

preload: 1
llff: 1

# dataset_type:
#   Colmap = 6
dataset_type: 6
camera_path: "sparse/0/cameras.txt"
color_path: "images"
color_type: "{color_type}"
color_pose_path: "sparse/0/images.txt"
color_pose_type: 4
color_pose_w2c: 1
depth_path: "depths"
depth_pose_path: "sparse/0/depths.txt"
depth_pose_type: 4
depth_type: 1

sh_degree: 3
bck_color: {args.background}

frame_rate: 10
min_range: {args.min_depth}
max_range: {args.max_depth}
ds_pt_num: {args.ds_pt_num}
max_pt_num: 1000000

outlier_remove: 1
outlier_dist: 0.05
outlier_removal_interval: 2000

leaf_sizes: {args.leaf_size}
bce_sigma: 1.0

vis_resolution: 0.1
export_resolution: 0.04
fps: 10

camera:
   scale: 1.0
   model: 0
   width: {width}
   height: {height}
   fx: {fx:.12g}
   fy: {fy:.12g}
   cx: {cx:.12g}
   cy: {cy:.12g}
   # Keep d0..d4 absent when distortion is zero. GS-SDF treats the presence
   # of distortion fields as a request to undistort and recompute K.

map:
   map_origin: !!opencv-matrix
      rows: 1
      cols: 3
      dt: d
      data: [ 0, 0, 0 ]

   map_size: {args.map_size}
"""
    )


def read_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(decoded)
            if decoded.strip() == "end_header":
                break
        header_size = handle.tell()

    if not header_lines or header_lines[0].strip() != "ply":
        raise ValueError(f"Not a PLY file: {path}")

    fmt = ""
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
        elif parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        elif in_vertex and parts[0] == "property" and len(parts) >= 3 and parts[1] != "list":
            vertex_properties.append((parts[2], parts[1]))

    if vertex_count <= 0:
        raise ValueError(f"No vertices in PLY: {path}")

    names = [name for name, _ in vertex_properties]
    try:
        xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]
    except ValueError as exc:
        raise ValueError(f"PLY does not contain x/y/z vertex properties: {path}") from exc

    if fmt == "ascii":
        data = np.loadtxt(path, comments=None, skiprows=len(header_lines), max_rows=vertex_count)
        if data.ndim == 1:
            data = data[None, :]
        return data[:, xyz_indices].astype(np.float32)

    endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else ""
    if not endian:
        raise ValueError(f"Unsupported PLY format {fmt}: {path}")

    dtype_fields = []
    for name, prop_type in vertex_properties:
        dtype_fields.append((name, endian + ply_type_to_dtype(prop_type)))
    dtype = np.dtype(dtype_fields)
    with path.open("rb") as handle:
        handle.seek(header_size)
        data = np.fromfile(handle, dtype=dtype, count=vertex_count)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)


def read_ply_mesh(path: Path) -> PlyMesh:
    with path.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            decoded = line.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(decoded)
            if decoded.strip() == "end_header":
                break
        header_size = handle.tell()

    fmt = ""
    vertex_count = 0
    face_count = 0
    vertex_properties: list[tuple[str, str]] = []
    face_list_property: tuple[str, str] | None = None
    current_element = ""
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
            elif current_element == "face":
                face_count = int(parts[2])
        elif parts[0] == "property" and current_element == "vertex" and len(parts) >= 3 and parts[1] != "list":
            vertex_properties.append((parts[2], parts[1]))
        elif (
            parts[0] == "property"
            and current_element == "face"
            and len(parts) >= 5
            and parts[1] == "list"
            and parts[4] == "vertex_indices"
        ):
            face_list_property = (parts[2], parts[3])

    vertices = read_ply_vertices(path)
    if face_count <= 0 or face_list_property is None:
        return PlyMesh(vertices=vertices, faces=None)

    if fmt == "ascii":
        faces: list[list[int]] = []
        with path.open("r") as handle:
            for _ in range(len(header_lines) + vertex_count):
                handle.readline()
            for _ in range(face_count):
                parts = handle.readline().split()
                if not parts:
                    continue
                count = int(parts[0])
                indices = [int(value) for value in parts[1 : 1 + count]]
                faces.extend(triangulate_face(indices))
        return PlyMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int32))

    endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else ""
    if not endian:
        raise ValueError(f"Unsupported PLY format {fmt}: {path}")

    vertex_dtype_fields = []
    for name, prop_type in vertex_properties:
        vertex_dtype_fields.append((name, endian + ply_type_to_dtype(prop_type)))
    vertex_dtype = np.dtype(vertex_dtype_fields)
    face_count_type, face_index_type = face_list_property
    count_dtype = np.dtype(endian + ply_type_to_dtype(face_count_type))
    index_dtype = np.dtype(endian + ply_type_to_dtype(face_index_type))
    vertex_bytes = vertex_count * vertex_dtype.itemsize

    with path.open("rb") as handle:
        handle.seek(header_size + vertex_bytes)
        blob = handle.read()

    fixed_triangle_bytes = face_count * (count_dtype.itemsize + 3 * index_dtype.itemsize)
    if len(blob) >= fixed_triangle_bytes:
        dtype = np.dtype([("count", count_dtype), ("idx", index_dtype, (3,))])
        records = np.frombuffer(blob, dtype=dtype, count=face_count)
        if np.all(records["count"] == 3):
            return PlyMesh(vertices=vertices, faces=records["idx"].astype(np.int32, copy=False))

    faces = []
    offset = 0
    for _ in range(face_count):
        if offset + count_dtype.itemsize > len(blob):
            break
        count = int(np.frombuffer(blob, dtype=count_dtype, count=1, offset=offset)[0])
        offset += count_dtype.itemsize
        byte_count = count * index_dtype.itemsize
        if offset + byte_count > len(blob):
            break
        indices = np.frombuffer(blob, dtype=index_dtype, count=count, offset=offset).astype(np.int64)
        offset += byte_count
        faces.extend(triangulate_face(indices.tolist()))
    return PlyMesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int32) if faces else None)


def triangulate_face(indices: list[int]) -> list[list[int]]:
    if len(indices) < 3:
        return []
    return [[indices[0], indices[index], indices[index + 1]] for index in range(1, len(indices) - 1)]


def ply_type_to_dtype(prop_type: str) -> str:
    mapping = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "i2",
        "ushort": "u2",
        "int16": "i2",
        "uint16": "u2",
        "int": "i4",
        "uint": "u4",
        "int32": "i4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }
    if prop_type not in mapping:
        raise ValueError(f"Unsupported PLY property type: {prop_type}")
    return mapping[prop_type]


def write_ascii_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for x, y, z in points:
            handle.write(f"{x:.7g} {y:.7g} {z:.7g}\n")


def choose_mesh_points(mesh: PlyMesh, mode: str, sample_count: int, rng: np.random.Generator) -> np.ndarray:
    if mode == "vertices" or mesh.faces is None or len(mesh.faces) == 0:
        return mesh.vertices

    faces = mesh.faces
    if sample_count <= 0:
        sample_count = int(min(2_000_000, max(250_000, len(mesh.vertices) * 2)))
    v0 = mesh.vertices[faces[:, 0]].astype(np.float32, copy=False)
    v1 = mesh.vertices[faces[:, 1]].astype(np.float32, copy=False)
    v2 = mesh.vertices[faces[:, 2]].astype(np.float32, copy=False)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    valid = np.isfinite(areas) & (areas > 1.0e-12)
    if not np.any(valid):
        return mesh.vertices
    valid_indices = np.flatnonzero(valid)
    valid_areas = areas[valid]
    probabilities = valid_areas / valid_areas.sum()
    picked = valid_indices[rng.choice(len(valid_indices), size=sample_count, replace=True, p=probabilities)]
    a = mesh.vertices[faces[picked, 0]]
    b = mesh.vertices[faces[picked, 1]]
    c = mesh.vertices[faces[picked, 2]]
    r1 = np.sqrt(rng.random(sample_count, dtype=np.float32))[:, None]
    r2 = rng.random(sample_count, dtype=np.float32)[:, None]
    return ((1.0 - r1) * a + r1 * (1.0 - r2) * b + r1 * r2 * c).astype(np.float32)


def generate_local_depth_from_mesh(
    mesh_points_world: np.ndarray,
    c2w: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float,
    max_depth: float,
    max_points: int,
    zbuffer_pixel_stride: int,
    rng: np.random.Generator,
) -> np.ndarray:
    w2c = np.linalg.inv(c2w)
    points_h = np.column_stack([mesh_points_world, np.ones((len(mesh_points_world), 1), dtype=np.float32)])
    points_cam = (w2c @ points_h.T).T[:, :3]

    z = points_cam[:, 2]
    mask = (z > min_depth) & (z < max_depth)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)
    points_cam = points_cam[mask]
    z = points_cam[:, 2]
    u = fx * (points_cam[:, 0] / z) + cx
    v = fy * (points_cam[:, 1] / z) + cy
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    points_cam = points_cam[mask]
    z = z[mask]
    u = u[mask]
    v = v[mask]
    if len(points_cam) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if zbuffer_pixel_stride > 0:
        stride = max(1, zbuffer_pixel_stride)
        bucket_width = int(math.ceil(width / stride))
        u_bucket = np.floor(u / stride).astype(np.int64)
        v_bucket = np.floor(v / stride).astype(np.int64)
        keys = v_bucket * bucket_width + u_bucket
        order = np.lexsort((z, keys))
        sorted_keys = keys[order]
        keep = np.ones(len(order), dtype=bool)
        keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
        points_cam = points_cam[order[keep]]
    if len(points_cam) > max_points:
        indices = rng.choice(len(points_cam), size=max_points, replace=False)
        points_cam = points_cam[indices]
    return points_cam.astype(np.float32)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    image_dir = resolve_path(input_root, args.image_dir)
    tum_path = resolve_path(input_root, args.tum_path)
    depth_ply_dir = resolve_path(input_root, args.depth_ply_dir) if args.depth_ply_dir else None
    mesh_ply = resolve_path(input_root, args.mesh_ply) if args.mesh_ply else None
    camera_spec, calibration_metadata = resolve_camera_calibration(input_root, args)
    effective_tum_pose_frame, effective_extrinsic_direction, effective_camera_frame_axis = (
        effective_pose_convention(args, camera_spec)
    )

    if output_root.exists():
        if not args.force:
            raise SystemExit(f"Output directory already exists: {output_root}. Re-run with --force.")
        shutil.rmtree(output_root)

    images = list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found under {image_dir}")
    poses = read_tum(tum_path)
    matches = match_frames(
        images,
        poses,
        args.match_mode,
        args.timestamp_tolerance_sec,
        args.image_timestamp_scale,
    )
    if args.frame_stride < 1:
        raise SystemExit("--frame-stride must be >= 1")
    matches = matches[:: args.frame_stride]
    if args.max_frames > 0:
        matches = matches[: args.max_frames]
    if not matches:
        raise SystemExit("No image/pose matches produced.")

    width = args.width
    height = args.height
    fx = args.fx
    fy = args.fy
    cx = args.cx
    cy = args.cy
    if camera_spec is not None:
        width = width or camera_spec.width
        height = height or camera_spec.height
        fx = fx or camera_spec.fx
        fy = fy or camera_spec.fy
        cx = cx or camera_spec.cx
        cy = cy or camera_spec.cy
    if width <= 0 or height <= 0:
        width, height = infer_image_size(matches[0].image_path)
    if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
        raise SystemExit(
            "Camera intrinsics are required: pass --fx --fy --cx --cy or --camera-extrinsics-json/--camera-id."
        )

    output_images = output_root / "images"
    output_depths = output_root / "depths"
    sparse = output_root / "sparse" / "0"
    output_images.mkdir(parents=True, exist_ok=True)
    output_depths.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    write_cameras_txt(sparse / "cameras.txt", width, height, fx, fy, cx, cy)

    image_rows = []
    depth_rows = []
    depth_point_counts: list[int] = []
    depth_sources = sorted(depth_ply_dir.glob("*.ply"), key=natural_image_key) if depth_ply_dir else []
    rng = np.random.default_rng(args.seed)
    mesh = read_ply_mesh(mesh_ply) if mesh_ply else None
    mesh_points = (
        choose_mesh_points(mesh, args.mesh_depth_mode, args.mesh_surface_samples, rng) if mesh is not None else None
    )

    if depth_ply_dir and not depth_sources:
        raise SystemExit(f"--depth-ply-dir has no .ply files: {depth_ply_dir}")
    if not depth_ply_dir and mesh_points is None:
        raise SystemExit("Pass either --depth-ply-dir or --mesh-ply so GS-SDF has geometry observations.")

    for index, match in enumerate(matches, start=1):
        suffix = args.image_output_ext or match.image_path.suffix.lower()
        if not suffix.startswith("."):
            suffix = "." + suffix
        image_name = f"{index:06d}{suffix}"
        materialize(match.image_path, output_images / image_name, args.copy_mode)

        c2w = camera_c2w_from_pose(
            match.pose,
            camera_spec,
            effective_tum_pose_frame,
            effective_extrinsic_direction,
            effective_camera_frame_axis,
        )
        quat, trans = matrix_to_colmap_qt(c2w)
        image_rows.append((index, quat, trans, image_name))

        depth_name = f"{index:06d}.ply"
        if depth_ply_dir:
            src_depth = depth_sources[min(index - 1, len(depth_sources) - 1)]
            materialize(src_depth, output_depths / depth_name, args.copy_mode)
        else:
            assert mesh_points is not None
            local_points = generate_local_depth_from_mesh(
                mesh_points,
                c2w,
                width,
                height,
                fx,
                fy,
                cx,
                cy,
                args.min_depth,
                args.max_depth,
                args.points_per_depth,
                args.zbuffer_pixel_stride,
                rng,
            )
            write_ascii_ply(output_depths / depth_name, local_points)
            depth_point_counts.append(int(len(local_points)))
        depth_rows.append((index, quat, trans, depth_name))

    write_pose_txt(sparse / "images.txt", image_rows, "Image")
    write_pose_txt(sparse / "depths.txt", depth_rows, "Depth")
    first_color_type = args.image_output_ext or matches[0].image_path.suffix.lower()
    if not first_color_type.startswith("."):
        first_color_type = "." + first_color_type
    write_gssdf_config(
        output_root / "gssdf_colmap.yaml",
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        first_color_type,
        args,
    )

    metadata = {
        "inputRoot": str(input_root),
        "outputRoot": str(output_root),
        "imageDir": str(image_dir),
        "tumPath": str(tum_path),
        "depthPlyDir": str(depth_ply_dir) if depth_ply_dir else None,
        "meshPly": str(mesh_ply) if mesh_ply else None,
        "frames": len(matches),
        "camera": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "cameraId": camera_spec.camera_id if camera_spec is not None else None,
            "cameraName": camera_spec.camera_name if camera_spec is not None else args.camera_name,
            "calibration": calibration_metadata,
            "tumPoseFrame": effective_tum_pose_frame,
            "cameraExtrinsicDirection": effective_extrinsic_direction,
            "cameraFrameAxis": effective_camera_frame_axis,
            "requestedTumPoseFrame": args.tum_pose_frame,
            "requestedCameraExtrinsicDirection": args.camera_extrinsic_direction,
            "requestedCameraFrameAxis": args.camera_frame_axis,
        },
        "mesh": {
            "mode": args.mesh_depth_mode if mesh is not None else None,
            "sourceVertexCount": int(len(mesh.vertices)) if mesh is not None else 0,
            "sourceFaceCount": int(len(mesh.faces)) if mesh is not None and mesh.faces is not None else 0,
            "sampledPointCount": int(len(mesh_points)) if mesh_points is not None else 0,
            "zbufferPixelStride": args.zbuffer_pixel_stride,
        },
        "depthPointCounts": {
            "min": min(depth_point_counts) if depth_point_counts else None,
            "max": max(depth_point_counts) if depth_point_counts else None,
            "mean": float(np.mean(depth_point_counts)) if depth_point_counts else None,
        },
        "layout": {
            "images": "images",
            "depths": "depths",
            "cameras": "sparse/0/cameras.txt",
            "imagePoses": "sparse/0/images.txt",
            "depthPoses": "sparse/0/depths.txt",
            "gssdfConfig": "gssdf_colmap.yaml",
        },
        "notes": [
            f"TUM poses are interpreted as {effective_tum_pose_frame}-to-world.",
            "COLMAP pose files are written as world-to-camera.",
            "Mesh-generated depth PLYs are synthetic local camera point clouds; they are a geometry prior, not a replacement for raw depth.",
        ],
    }
    (output_root / "nurec_to_gssdf_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
