#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None


SCENE_ROOT_DEFAULT = "/home/chatsign/gs-sdf/examples/web-ui/public/scenes"
PLY_TYPE_INFO = {
    "char": ("b", 1, "i1"),
    "uchar": ("B", 1, "u1"),
    "short": ("h", 2, "i2"),
    "ushort": ("H", 2, "u2"),
    "int": ("i", 4, "i4"),
    "uint": ("I", 4, "u4"),
    "float": ("f", 4, "f4"),
    "float32": ("f", 4, "f4"),
    "double": ("d", 8, "f8"),
    "float64": ("d", 8, "f8"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a CityGaussian output directory into the GS-SDF web scene-pack layout."
    )
    parser.add_argument(
        "--citygaussian-dir",
        required=True,
        help="CityGaussian output directory, usually outputs/<scene-name>",
    )
    parser.add_argument("--scene-id", help="Scene id written under public/scenes/<scene-id>")
    parser.add_argument("--scene-root", default=SCENE_ROOT_DEFAULT, help="Web scene root directory")
    parser.add_argument("--url-prefix", default="/scenes", help="Public URL prefix for the scene root")
    parser.add_argument("--frame-id", default="world", help="Frame id to expose in manifest.json")
    parser.add_argument("--status", default="completed", help="Training status label")
    parser.add_argument("--gaussian-path", help="Override gaussian asset (.ply, .splat, .spz, or .ckpt)")
    parser.add_argument("--mesh-path", help="Override mesh asset (.ply or .glb)")
    parser.add_argument(
        "--occupancy-path",
        help="Optional PLY used by TopDownMap. Defaults to mesh or another PLY fallback.",
    )
    parser.add_argument(
        "--raw-pointcloud-path",
        help="Optional static point cloud for the 3D overlay. Defaults to occupancy or mesh fallback.",
    )
    parser.add_argument(
        "--allow-gaussian-fallback",
        action="store_true",
        help="Allow using the gaussian PLY as occupancy/raw point-cloud fallback when no mesh or explicit point cloud exists.",
    )
    parser.add_argument("--leaf-size", type=float, default=0.2, help="2D map resolution in meters")
    parser.add_argument("--map-padding", type=float, default=2.0, help="Extra XY padding around inferred bounds")
    parser.add_argument(
        "--map-origin-z",
        type=float,
        help="Override map ground Z. By default it is inferred from the chosen PLY bounds source.",
    )
    parser.add_argument(
        "--map-size",
        type=float,
        help="Override square map size in meters. By default it is inferred from the chosen PLY bounds source.",
    )
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument("--robot-height", type=float, default=0.45)
    parser.add_argument(
        "--citygaussian-repo",
        default="/home/chatsign/work/CityGaussian",
        help="Path to the cloned CityGaussian repository used for ckpt export",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used when calling CityGaussian export helpers",
    )
    parser.add_argument(
        "--auto-export-ckpt",
        action="store_true",
        help="If only a checkpoint is found, run CityGaussian utils/ckpt2ply.py automatically",
    )
    parser.add_argument(
        "--preprocess-gaussian",
        action="store_true",
        help="Run preprocess_gaussian_stream.mjs and expose chunked browser assets instead of raw PLY",
    )
    parser.add_argument(
        "--variant",
        default="quality",
        help="Variant name used when chunking the gaussian asset. Use balanced to keep the default file names.",
    )
    parser.add_argument("--grid", default="6,6", help="Chunk grid passed to preprocess_gaussian_stream.mjs")
    parser.add_argument("--max-sh", type=int, default=2, help="Max SH degree kept during chunk preprocessing")
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.005,
        help="Opacity threshold passed to preprocess_gaussian_stream.mjs",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        help="Override manifest gaussian.shDegree. By default it is inferred from the PLY header.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing scene-pack links and chunk outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned manifest without writing files")
    return parser.parse_args()


def ensure_file(path_value: Optional[str], label: str) -> Optional[Path]:
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def sanitize_scene_id(raw: str) -> str:
    scene_id = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip()).strip("-._")
    return scene_id or "citygaussian-scene"


def safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def find_latest_iteration_file(paths: Iterable[Path]) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_iteration = -1
    for path in paths:
        match = re.search(r"iteration_(\d+)", str(path))
        iteration = int(match.group(1)) if match else -1
        if iteration > best_iteration:
            best_iteration = iteration
            best_path = path
    return best_path


def find_latest_checkpoint(root: Path) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_step = -1
    for path in root.glob("checkpoints/*.ckpt"):
        match = re.search(r"step=(\d+)", path.name)
        step = int(match.group(1)) if match else -1
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def detect_gaussian_asset(root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override

    point_cloud_ply = find_latest_iteration_file(root.glob("point_cloud/iteration_*/point_cloud.ply"))
    if point_cloud_ply is not None:
        return point_cloud_ply

    point_cloud_splat = find_latest_iteration_file(root.glob("point_cloud/iteration_*/point_cloud.splat"))
    if point_cloud_splat is not None:
        return point_cloud_splat

    for relative in ("scene_point_cloud.ply", "point_cloud.ply", "scene_point_cloud.splat", "point_cloud.splat"):
        candidate = root / relative
        if candidate.is_file():
            return candidate

    return find_latest_checkpoint(root)


def detect_mesh_asset(root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override

    patterns = (
        "fuse_post.ply",
        "mesh/fuse_post.ply",
        "mesh/fuse.ply",
        "mesh.ply",
        "mesh/*.ply",
        "mesh/*.glb",
        "*.glb",
    )
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def infer_sh_degree_from_ply_header(path: Path) -> int:
    with path.open("rb") as handle:
        rest_count = 0
        while True:
            line = handle.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("property float f_rest_"):
                rest_count += 1
            if text == "end_header":
                break
    if rest_count >= 45:
        return 3
    if rest_count >= 24:
        return 2
    if rest_count >= 9:
        return 1
    return 0


def gaussian_format_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".spz":
        return "spz"
    if suffix == ".splat":
        return "splat"
    return "gs-ply"


def parse_grid_arg(value: str) -> Tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise SystemExit("--grid expects x,y")
    try:
        grid = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise SystemExit("--grid expects integer x,y") from exc
    if grid[0] < 1 or grid[1] < 1:
        raise SystemExit("--grid expects positive integers")
    return grid


def export_checkpoint_to_ply(args: argparse.Namespace, ckpt_path: Path, repo_root: Path, scene_id: str) -> Path:
    citygaussian_repo = Path(args.citygaussian_repo).expanduser().resolve()
    exporter = citygaussian_repo / "utils" / "ckpt2ply.py"
    if not exporter.is_file():
        raise SystemExit(f"CityGaussian ckpt exporter not found: {exporter}")

    export_root = repo_root / "runtime" / "imports" / "citygaussian" / scene_id / "exported"
    export_root.mkdir(parents=True, exist_ok=True)
    output_path = export_root / "point_cloud.ply"
    if output_path.exists() and not args.force:
        return output_path
    if output_path.exists():
        output_path.unlink()

    command = [
        args.python,
        str(exporter),
        str(ckpt_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, cwd=str(citygaussian_repo))
    if not output_path.is_file():
        raise SystemExit(f"CityGaussian export did not create: {output_path}")
    return output_path


def read_ply_header(path: Path) -> Tuple[str, int, List[Tuple[str, str]], int, int]:
    with path.open("rb") as handle:
        header_lines: List[str] = []
        vertex_props: List[Tuple[str, str]] = []
        format_name: Optional[str] = None
        vertex_count: Optional[int] = None
        in_vertex = False

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise ValueError(f"Invalid PLY header in {path}")
            line = raw_line.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break
            if not line:
                continue
            tokens = line.split()
            if tokens[0] == "format" and len(tokens) >= 2:
                format_name = tokens[1]
            elif tokens[0] == "element" and len(tokens) >= 3:
                in_vertex = tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                    vertex_props = []
            elif tokens[0] == "property" and in_vertex:
                if len(tokens) < 3 or tokens[1] == "list":
                    raise ValueError(f"Unsupported PLY vertex property in {path}: {line}")
                vertex_props.append((tokens[2], tokens[1]))

        if format_name is None or vertex_count is None:
            raise ValueError(f"Could not find vertex header in {path}")
        return format_name, vertex_count, vertex_props, handle.tell(), len(header_lines)


def compute_ply_bounds(path: Path) -> Optional[Dict[str, float]]:
    format_name, vertex_count, props, header_offset, header_lines = read_ply_header(path)
    prop_names = [name for name, _ in props]
    if not {"x", "y", "z"}.issubset(set(prop_names)):
        raise ValueError(f"PLY does not expose x/y/z vertex properties: {path}")

    if np is not None:
        bounds = compute_ply_bounds_numpy(path, format_name, vertex_count, props, header_offset, header_lines)
        if bounds is not None:
            return bounds

    if format_name == "ascii":
        return compute_ply_bounds_ascii(path, vertex_count, prop_names, header_lines)
    if format_name in {"binary_little_endian", "binary_big_endian"}:
        return compute_ply_bounds_binary(path, format_name, vertex_count, props, header_offset)
    raise ValueError(f"Unsupported PLY format in {path}: {format_name}")


def compute_ply_bounds_numpy(
    path: Path,
    format_name: str,
    vertex_count: int,
    props: List[Tuple[str, str]],
    header_offset: int,
    header_lines: int,
) -> Optional[Dict[str, float]]:
    try:
        if format_name == "ascii":
            usecols = [index for index, (name, _) in enumerate(props) if name in {"x", "y", "z"}]
            data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count, usecols=usecols)
            if data.size == 0:
                return None
            if data.ndim == 1:
                data = data.reshape(1, -1)
            x_values = data[:, 0]
            y_values = data[:, 1]
            z_values = data[:, 2]
        else:
            endian = "<" if format_name == "binary_little_endian" else ">"
            dtype_fields = []
            for name, type_name in props:
                info = PLY_TYPE_INFO.get(type_name)
                if info is None:
                    return None
                dtype_fields.append((name, endian + info[2]))
            dtype = np.dtype(dtype_fields)
            with path.open("rb") as handle:
                handle.seek(header_offset)
                vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
            if vertices.size == 0:
                return None
            x_values = vertices["x"]
            y_values = vertices["y"]
            z_values = vertices["z"]
        return {
            "minX": float(np.min(x_values)),
            "maxX": float(np.max(x_values)),
            "minY": float(np.min(y_values)),
            "maxY": float(np.max(y_values)),
            "minZ": float(np.min(z_values)),
            "maxZ": float(np.max(z_values)),
        }
    except Exception:
        return None


def compute_ply_bounds_ascii(
    path: Path,
    vertex_count: int,
    prop_names: List[str],
    header_lines: int,
) -> Optional[Dict[str, float]]:
    x_index = prop_names.index("x")
    y_index = prop_names.index("y")
    z_index = prop_names.index("z")
    bounds = init_bounds()

    with path.open("r", encoding="ascii", errors="replace") as handle:
        for _ in range(header_lines):
            next(handle)
        for _ in range(vertex_count):
            line = handle.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) <= max(x_index, y_index, z_index):
                continue
            expand_bounds(bounds, float(parts[x_index]), float(parts[y_index]), float(parts[z_index]))

    return bounds if is_bounds_valid(bounds) else None


def compute_ply_bounds_binary(
    path: Path,
    format_name: str,
    vertex_count: int,
    props: List[Tuple[str, str]],
    header_offset: int,
) -> Optional[Dict[str, float]]:
    endian = "<" if format_name == "binary_little_endian" else ">"
    offsets: Dict[str, int] = {}
    formats: List[str] = []
    stride = 0
    for name, type_name in props:
        info = PLY_TYPE_INFO.get(type_name)
        if info is None:
            raise ValueError(f"Unsupported PLY property type in {path}: {type_name}")
        formats.append(info[0])
        offsets[name] = stride
        stride += info[1]

    struct_format = struct.Struct(endian + "".join(formats))
    x_slot = prop_slot(props, "x")
    y_slot = prop_slot(props, "y")
    z_slot = prop_slot(props, "z")
    bounds = init_bounds()

    with path.open("rb") as handle:
        handle.seek(header_offset)
        for _ in range(vertex_count):
            record = handle.read(stride)
            if len(record) < stride:
                break
            values = struct_format.unpack(record)
            expand_bounds(bounds, float(values[x_slot]), float(values[y_slot]), float(values[z_slot]))

    return bounds if is_bounds_valid(bounds) else None


def prop_slot(props: List[Tuple[str, str]], target: str) -> int:
    for index, (name, _) in enumerate(props):
        if name == target:
            return index
    raise ValueError(f"Missing property {target}")


def init_bounds() -> Dict[str, float]:
    return {
        "minX": float("inf"),
        "maxX": float("-inf"),
        "minY": float("inf"),
        "maxY": float("-inf"),
        "minZ": float("inf"),
        "maxZ": float("-inf"),
    }


def expand_bounds(bounds: Dict[str, float], x: float, y: float, z: float) -> None:
    bounds["minX"] = min(bounds["minX"], x)
    bounds["maxX"] = max(bounds["maxX"], x)
    bounds["minY"] = min(bounds["minY"], y)
    bounds["maxY"] = max(bounds["maxY"], y)
    bounds["minZ"] = min(bounds["minZ"], z)
    bounds["maxZ"] = max(bounds["maxZ"], z)


def is_bounds_valid(bounds: Dict[str, float]) -> bool:
    return (
        bounds["minX"] < float("inf")
        and bounds["maxX"] > float("-inf")
        and bounds["minY"] < float("inf")
        and bounds["maxY"] > float("-inf")
    )


def detect_citygaussian_camera_path(root: Path) -> Optional[Path]:
    for relative in ("cameras.json", "sparse/cameras.json"):
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def load_citygaussian_cameras(root: Path) -> List[dict]:
    camera_path = detect_citygaussian_camera_path(root)
    if camera_path is None:
        return []

    try:
        payload = json.loads(camera_path.read_text())
    except Exception:
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for key in ("cameras", "frames"):
            cameras = payload.get(key)
            if isinstance(cameras, list):
                return [item for item in cameras if isinstance(item, dict)]

    return []


def parse_vec3(value: object) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result: List[float] = []
    for entry in value:
        try:
            result.append(float(entry))
        except Exception:
            return None
    return result


def parse_mat3(value: object) -> Optional[List[List[float]]]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    rows: List[List[float]] = []
    for row in value:
        parsed = parse_vec3(row)
        if parsed is None:
            return None
        rows.append(parsed)
    return rows


def normalize_vec3(values: List[float]) -> Optional[List[float]]:
    if np is not None:
        vector = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return None
        vector = vector / norm
        return [float(entry) for entry in vector.tolist()]

    norm = sum(entry * entry for entry in values) ** 0.5
    if norm < 1e-9:
        return None
    return [entry / norm for entry in values]


def camera_forward_vector(camera: dict) -> Optional[List[float]]:
    rotation = parse_mat3(camera.get("rotation"))
    if rotation is None:
        return None
    return normalize_vec3([rotation[0][2], rotation[1][2], rotation[2][2]])


def maybe_int(value: object) -> Optional[int]:
    try:
        number = int(round(float(value)))
    except Exception:
        return None
    return number if number > 0 else None


def maybe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def build_manifest_camera_entry(camera: dict) -> Optional[dict]:
    width = maybe_int(camera.get("width"))
    height = maybe_int(camera.get("height"))
    fx = maybe_float(camera.get("fx"))
    fy = maybe_float(camera.get("fy"))
    if width is None and height is None and fx is None and fy is None:
        return None
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width * 0.5 if width is not None else None,
        "cy": height * 0.5 if height is not None else None,
    }


def estimate_camera_focus_point(cameras: List[dict]) -> Optional[List[float]]:
    if np is None:
        return None

    positions = []
    directions = []
    for camera in cameras:
        position = parse_vec3(camera.get("position"))
        direction = camera_forward_vector(camera)
        if position is None or direction is None:
            continue
        positions.append(np.asarray(position, dtype=float))
        directions.append(np.asarray(direction, dtype=float))

    if len(positions) < 2:
        return None

    eye = np.eye(3, dtype=float)
    matrix = np.zeros((3, 3), dtype=float)
    vector = np.zeros(3, dtype=float)
    for position, direction in zip(positions, directions):
        projector = eye - np.outer(direction, direction)
        matrix += projector
        vector += projector @ position

    try:
        focus = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(focus)):
        return None
    return [float(entry) for entry in focus.tolist()]


def offset_target(position: List[float], direction: List[float], distance: float = 30.0) -> List[float]:
    return [
        position[0] + direction[0] * distance,
        position[1] + direction[1] * distance,
        position[2] + direction[2] * distance,
    ]


def build_initial_view(cameras: List[dict], frame_id: str) -> Tuple[Optional[dict], Optional[dict]]:
    parsed = []
    for camera in cameras:
        position = parse_vec3(camera.get("position"))
        direction = camera_forward_vector(camera)
        if position is None or direction is None:
            continue
        parsed.append(
            {
                "camera": camera,
                "position": position,
                "direction": direction,
            }
        )

    camera_entry = None
    for camera in cameras:
        camera_entry = build_manifest_camera_entry(camera)
        if camera_entry is not None:
            break

    if not parsed:
        return None, camera_entry

    selected = parsed[len(parsed) // 2]
    focus = estimate_camera_focus_point([item["camera"] for item in parsed])

    if np is not None and focus is not None:
        positions = np.asarray([item["position"] for item in parsed], dtype=float)
        median_position = np.median(positions, axis=0)
        focus_vector = np.asarray(focus, dtype=float)
        best_score = float("inf")

        for item in parsed:
            position = np.asarray(item["position"], dtype=float)
            direction = np.asarray(item["direction"], dtype=float)
            delta = focus_vector - position
            distance = float(np.linalg.norm(delta))
            if distance > 1e-6:
                facing = float(np.dot(direction, delta / distance))
            else:
                facing = 1.0
            projection = float(np.dot(delta, direction))
            closest = position + direction * projection
            ray_distance = float(np.linalg.norm(focus_vector - closest))
            centrality = float(np.linalg.norm(position - median_position))
            score = ray_distance + 0.15 * centrality - 8.0 * facing
            if projection < 2.0:
                score += 10.0
            if score < best_score:
                best_score = score
                selected = item

    target = focus if focus is not None else offset_target(selected["position"], selected["direction"])
    if np is not None and focus is not None:
        distance_to_target = float(
            np.linalg.norm(np.asarray(target, dtype=float) - np.asarray(selected["position"], dtype=float))
        )
        if distance_to_target < 3.0:
            target = offset_target(selected["position"], selected["direction"])

    if camera_entry is None:
        camera_entry = build_manifest_camera_entry(selected["camera"])

    return (
        {
            "pose": {
                "frameId": frame_id,
                "position": {
                    "x": selected["position"][0],
                    "y": selected["position"][1],
                    "z": selected["position"][2],
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
            "target": {
                "x": target[0],
                "y": target[1],
                "z": target[2],
            },
        },
        camera_entry,
    )


def format_variant_label(variant: str) -> str:
    return variant.replace("_", " ").replace("-", " ").title()


def chunk_meta_name(variant: str) -> str:
    return "gs_chunks.json" if variant == "balanced" else f"gs_chunks_{variant}.json"


def chunk_dir_name(variant: str) -> str:
    return "gs_chunks" if variant == "balanced" else f"gs_chunks_{variant}"


def ensure_preprocessed_chunks(
    args: argparse.Namespace,
    repo_root: Path,
    source_ply: Path,
    scene_id: str,
) -> Path:
    processed_meta = repo_root / "runtime" / "processed" / scene_id / "model" / chunk_meta_name(args.variant)
    if processed_meta.is_file() and not args.force:
        return processed_meta

    if source_ply.suffix.lower() != ".ply":
        raise SystemExit("--preprocess-gaussian only supports .ply gaussian assets")

    stage_root = repo_root / "runtime" / "imports" / "citygaussian" / scene_id
    stage_model_root = stage_root / "model"
    stage_model_root.mkdir(parents=True, exist_ok=True)
    safe_link(source_ply, stage_model_root / "gs.ply")

    grid_x, grid_y = parse_grid_arg(args.grid)
    command = [
        "node",
        str(repo_root / "scripts" / "preprocess_gaussian_stream.mjs"),
        "--output-dir",
        str(stage_root),
        "--grid",
        f"{grid_x},{grid_y}",
        "--max-sh",
        str(args.max_sh),
        "--opacity-threshold",
        str(args.opacity_threshold),
    ]
    if args.variant != "balanced":
        command.extend(["--variant", args.variant])
    if args.force:
        command.append("--force")

    subprocess.run(command, check=True, cwd=str(repo_root))
    if not processed_meta.is_file():
        raise SystemExit(f"Chunk preprocessing did not create: {processed_meta}")
    return processed_meta


def build_chunk_source(
    chunks_manifest: dict,
    model_root: Path,
    scene_id: str,
    url_prefix: str,
    variant: str,
) -> dict:
    chunk_root = model_root / chunk_dir_name(variant)
    if chunk_root.exists() or chunk_root.is_symlink():
        if chunk_root.is_dir() and not chunk_root.is_symlink():
            shutil.rmtree(chunk_root)
        else:
            chunk_root.unlink()
    chunk_root.mkdir(parents=True, exist_ok=True)

    chunks = []
    for chunk in chunks_manifest.get("chunks", []):
        src = Path(chunk["path"]).resolve()
        dst = chunk_root / chunk["file"]
        safe_link(src, dst)
        chunks.append(
            {
                "id": chunk.get("id"),
                "url": f"{url_prefix}/{scene_id}/model/{chunk_dir_name(variant)}/{chunk['file']}",
                "splats": chunk.get("splats"),
                "bytes": chunk.get("bytes", src.stat().st_size),
                "center": chunk.get("center"),
                "bounds": chunk.get("bounds"),
            }
        )

    return {
        "format": "gs-chunks",
        "shDegree": int(chunks_manifest.get("output", {}).get("shDegree", 1)),
        "variant": variant,
        "label": format_variant_label(variant),
        "chunks": chunks,
    }


class SceneAssetLinker:
    def __init__(self, model_root: Path, scene_id: str, url_prefix: str):
        self.model_root = model_root
        self.scene_id = scene_id
        self.url_prefix = url_prefix.rstrip("/")
        self._cache: Dict[Path, dict] = {}

    def link(self, src: Path, preferred_name: Optional[str] = None) -> dict:
        src = src.resolve()
        cached = self._cache.get(src)
        if cached is not None:
            return cached

        base_name = preferred_name or src.name
        candidate = self.model_root / base_name
        if candidate.exists() and candidate.resolve() != src.resolve():
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            index = 2
            while True:
                candidate = self.model_root / f"{stem}_{index}{suffix}"
                if not candidate.exists():
                    break
                index += 1

        safe_link(src, candidate)
        linked = {
            "path": candidate,
            "url": f"{self.url_prefix}/{self.scene_id}/model/{candidate.name}",
        }
        self._cache[src] = linked
        return linked


class PlannedSceneAssetLinker(SceneAssetLinker):
    def __init__(self, model_root: Path, scene_id: str, url_prefix: str):
        super().__init__(model_root, scene_id, url_prefix)

    def link(self, src: Path, preferred_name: Optional[str] = None) -> dict:
        src = src.resolve()
        cached = self._cache.get(src)
        if cached is not None:
            return cached
        name = preferred_name or src.name
        linked = {
            "path": self.model_root / name,
            "url": f"{self.url_prefix}/{self.scene_id}/model/{name}",
        }
        self._cache[src] = linked
        return linked


def citygaussian_commit(path: Path) -> Optional[str]:
    git_dir = path / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def choose_bounds_source(
    occupancy_path: Optional[Path],
    raw_pointcloud_path: Optional[Path],
    mesh_path: Optional[Path],
    gaussian_path: Path,
) -> Optional[Path]:
    for candidate in (occupancy_path, raw_pointcloud_path, mesh_path, gaussian_path):
        if candidate is not None and candidate.suffix.lower() == ".ply":
            return candidate
    return None


def build_manifest(
    args: argparse.Namespace,
    scene_id: str,
    city_root: Path,
    repo_root: Path,
    scene_root: Path,
    model_root: Path,
    gaussian_path: Path,
    mesh_path: Optional[Path],
    occupancy_path: Optional[Path],
    raw_pointcloud_path: Optional[Path],
    occupancy_state: str,
    write_assets: bool,
) -> dict:
    if write_assets:
        scene_root.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)
        linker: SceneAssetLinker = SceneAssetLinker(model_root, scene_id, args.url_prefix)
    else:
        linker = PlannedSceneAssetLinker(model_root, scene_id, args.url_prefix)

    initial_view, camera_entry = build_initial_view(load_citygaussian_cameras(city_root), args.frame_id)

    gaussian_manifest: Optional[dict] = None
    gaussian_variants: Optional[dict] = None
    if args.preprocess_gaussian:
        if write_assets:
            chunk_meta_path = ensure_preprocessed_chunks(args, repo_root, gaussian_path, scene_id)
            chunk_manifest = load_json(chunk_meta_path)
            if not chunk_manifest or not chunk_manifest.get("chunks"):
                raise SystemExit(f"Chunk manifest is empty or invalid: {chunk_meta_path}")
            gaussian_manifest = build_chunk_source(
                chunk_manifest,
                model_root=model_root,
                scene_id=scene_id,
                url_prefix=args.url_prefix,
                variant=args.variant,
            )
        else:
            gaussian_manifest = {
                "format": "gs-chunks",
                "shDegree": args.max_sh,
                "variant": args.variant,
                "label": format_variant_label(args.variant),
                "chunks": [],
            }
        gaussian_variants = {args.variant: gaussian_manifest}
    else:
        linked_gaussian = linker.link(gaussian_path)
        sh_degree = args.sh_degree
        if sh_degree is None:
            if gaussian_path.suffix.lower() == ".ply":
                sh_degree = infer_sh_degree_from_ply_header(gaussian_path)
            elif gaussian_path.suffix.lower() == ".spz":
                sh_degree = 1
            else:
                sh_degree = 0
        gaussian_manifest = {
            "format": gaussian_format_for_path(gaussian_path),
            "url": linked_gaussian["url"],
            "shDegree": sh_degree,
            "variant": "raw",
            "label": "Raw",
        }

    bounds_source = choose_bounds_source(occupancy_path, raw_pointcloud_path, mesh_path, gaussian_path)
    bounds = compute_ply_bounds(bounds_source) if bounds_source is not None else None
    if occupancy_path is not None and bounds is None and args.map_size is None:
        raise SystemExit(
            "Could not infer map bounds from PLY assets. Provide --map-size, or point --occupancy-path/--mesh-path to a PLY file."
        )

    map_origin_z = args.map_origin_z if args.map_origin_z is not None else (bounds["minZ"] if bounds else 0.0)
    if args.map_size is not None:
        map_size = args.map_size
        map_origin_x = bounds["minX"] - args.map_padding if bounds else 0.0
        map_origin_y = bounds["minY"] - args.map_padding if bounds else 0.0
    elif bounds is not None:
        span_x = bounds["maxX"] - bounds["minX"]
        span_y = bounds["maxY"] - bounds["minY"]
        span = max(span_x, span_y)
        map_size = max(args.leaf_size * 8, span + args.map_padding * 2)
        map_origin_x = bounds["minX"] - args.map_padding
        map_origin_y = bounds["minY"] - args.map_padding
    else:
        map_size = None
        map_origin_x = 0.0
        map_origin_y = 0.0

    manifest = {
        "schemaVersion": "slam-adapter.scene-manifest/v1alpha1",
        "sceneId": scene_id,
        "frameId": args.frame_id,
        "gaussian": gaussian_manifest,
        "training": {
            "status": args.status,
            "outputDir": str(city_root),
            "configPath": None,
        },
        "robot": {
            "radius": args.robot_radius,
            "height": args.robot_height,
        },
        "meta": {
            "leafSize": args.leaf_size,
            "mapOrigin": {
                "x": map_origin_x,
                "y": map_origin_y,
                "z": map_origin_z,
            },
            "mapSize": map_size,
        },
        "assets": {
            "gaussian": "ready",
            "mesh": "missing",
            "occupancy": "missing",
            "rawPointCloud": "missing",
        },
        "source": {
            "kind": "citygaussian",
            "outputDir": str(city_root),
            "repoPath": str(Path(args.citygaussian_repo).expanduser().resolve()),
            "repoCommit": citygaussian_commit(Path(args.citygaussian_repo).expanduser().resolve()),
            "liveContractUrl": "/contracts/live-contract.default.json",
            "capabilityMatrixUrl": "/contracts/adapter-capability-matrix.default.json",
        },
    }
    if camera_entry is not None:
        manifest["camera"] = camera_entry
    if initial_view is not None:
        manifest["initialView"] = initial_view
    if gaussian_variants is not None:
        manifest["gaussianVariants"] = gaussian_variants

    if mesh_path is not None:
        linked_mesh = linker.link(mesh_path)
        manifest["mesh"] = {
            "format": "glb" if mesh_path.suffix.lower() == ".glb" else "ply",
            "url": linked_mesh["url"],
        }
        manifest["assets"]["mesh"] = "ready"

    if occupancy_path is not None:
        linked_occupancy = linker.link(occupancy_path)
        manifest["occupancy"] = {
            "source": "mesh_projection" if occupancy_state == "fallback" and mesh_path and occupancy_path.resolve() == mesh_path.resolve() else "prebuilt_grid",
            "url": linked_occupancy["url"],
            "resolution": args.leaf_size,
            "origin": {
                "x": map_origin_x,
                "y": map_origin_y,
            },
        }
        manifest["assets"]["occupancy"] = occupancy_state

    if raw_pointcloud_path is not None:
        linked_raw = linker.link(raw_pointcloud_path)
        manifest["rawPointCloud"] = {
            "format": "pcd" if raw_pointcloud_path.suffix.lower() == ".pcd" else "ply",
            "url": linked_raw["url"],
        }
        manifest["assets"]["rawPointCloud"] = "fallback" if occupancy_state == "fallback" else "ready"

    return manifest


def select_occupancy_and_raw_assets(
    gaussian_path: Path,
    mesh_path: Optional[Path],
    occupancy_override: Optional[Path],
    raw_pointcloud_override: Optional[Path],
    allow_gaussian_fallback: bool,
) -> Tuple[Optional[Path], Optional[Path], str]:
    occupancy_path = occupancy_override
    occupancy_state = "ready" if occupancy_override else "missing"

    if occupancy_path is None:
        if mesh_path is not None and mesh_path.suffix.lower() == ".ply":
            occupancy_path = mesh_path
            occupancy_state = "fallback"
        elif raw_pointcloud_override is not None and raw_pointcloud_override.suffix.lower() == ".ply":
            occupancy_path = raw_pointcloud_override
            occupancy_state = "fallback"
        elif allow_gaussian_fallback and gaussian_path.suffix.lower() == ".ply":
            occupancy_path = gaussian_path
            occupancy_state = "fallback"

    raw_pointcloud_path = raw_pointcloud_override
    if raw_pointcloud_path is None:
        if occupancy_override is not None:
            raw_pointcloud_path = occupancy_override
        elif mesh_path is not None and mesh_path.suffix.lower() in {".ply", ".pcd"}:
            raw_pointcloud_path = mesh_path
        elif occupancy_path is not None and occupancy_path.suffix.lower() in {".ply", ".pcd"}:
            raw_pointcloud_path = occupancy_path

    return occupancy_path, raw_pointcloud_path, occupancy_state


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    city_root = Path(args.citygaussian_dir).expanduser().resolve()
    if not city_root.is_dir():
        raise SystemExit(f"CityGaussian output dir not found: {city_root}")

    scene_id = sanitize_scene_id(args.scene_id or city_root.name)
    gaussian_override = ensure_file(args.gaussian_path, "Gaussian asset")
    mesh_override = ensure_file(args.mesh_path, "Mesh asset")
    occupancy_override = ensure_file(args.occupancy_path, "Occupancy asset")
    raw_override = ensure_file(args.raw_pointcloud_path, "Raw point cloud asset")

    gaussian_path = detect_gaussian_asset(city_root, gaussian_override)
    if gaussian_path is None:
        raise SystemExit(
            "No CityGaussian gaussian asset found. Expected point_cloud/iteration_*/point_cloud.ply or a checkpoint under checkpoints/."
        )

    if gaussian_path.suffix.lower() == ".ckpt":
        if not args.auto_export_ckpt:
            exporter = Path(args.citygaussian_repo).expanduser().resolve() / "utils" / "ckpt2ply.py"
            raise SystemExit(
                "Only a CityGaussian checkpoint was found. Re-run with --auto-export-ckpt, or export manually with:\n"
                f"{args.python} {exporter} {gaussian_path}"
            )
        gaussian_path = export_checkpoint_to_ply(args, gaussian_path, repo_root, scene_id)

    mesh_path = detect_mesh_asset(city_root, mesh_override)
    occupancy_path, raw_pointcloud_path, occupancy_state = select_occupancy_and_raw_assets(
        gaussian_path=gaussian_path,
        mesh_path=mesh_path,
        occupancy_override=occupancy_override,
        raw_pointcloud_override=raw_override,
        allow_gaussian_fallback=args.allow_gaussian_fallback,
    )

    scene_root = Path(args.scene_root).expanduser().resolve() / scene_id
    model_root = scene_root / "model"
    manifest = build_manifest(
        args=args,
        scene_id=scene_id,
        city_root=city_root,
        repo_root=repo_root,
        scene_root=scene_root,
        model_root=model_root,
        gaussian_path=gaussian_path,
        mesh_path=mesh_path,
        occupancy_path=occupancy_path,
        raw_pointcloud_path=raw_pointcloud_path,
        occupancy_state=occupancy_state,
        write_assets=not args.dry_run,
    )

    manifest_path = scene_root / "manifest.json"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sceneId": scene_id,
                    "manifestPath": str(manifest_path),
                    "gaussianPath": str(gaussian_path),
                    "meshPath": str(mesh_path) if mesh_path else None,
                    "occupancyPath": str(occupancy_path) if occupancy_path else None,
                    "rawPointCloudPath": str(raw_pointcloud_path) if raw_pointcloud_path else None,
                    "manifest": manifest,
                },
                indent=2,
            )
        )
        return

    scene_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "sceneId": scene_id,
                "manifest": str(manifest_path),
                "gaussian": manifest["assets"]["gaussian"],
                "mesh": manifest["assets"]["mesh"],
                "occupancy": manifest["assets"]["occupancy"],
                "rawPointCloud": manifest["assets"]["rawPointCloud"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
