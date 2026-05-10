#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import struct
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sync GS-SDF output into a web scene pack.")
    parser.add_argument("--output-dir", required=True, help="GS-SDF output directory")
    parser.add_argument("--scene-config", help="Host-side scene config yaml")
    parser.add_argument(
        "--scene-root",
        default="/home/chatsign/gs-sdf/examples/web-ui/public/scenes",
        help="Web scene root directory",
    )
    parser.add_argument("--scene-id", help="Override scene id")
    parser.add_argument("--url-prefix", default="/scenes", help="Public URL prefix")
    parser.add_argument("--status", default="running", help="training status label")
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument("--robot-height", type=float, default=0.45)
    return parser.parse_args()


def find_text(path: Path, pattern: str):
    if not path.is_file():
        return None
    text = path.read_text()
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1) if match else None


def parse_data_path(config_path: Path):
    value = find_text(config_path, r'^data_path:\s*"([^"]+)"')
    if not value:
        return None
    return Path(value).expanduser().resolve()


def parse_scene_meta(config_path: Path):
    meta = {
        "leaf_size": None,
        "map_origin": {"x": 0.0, "y": 0.0, "z": 0.0},
        "map_size": None,
    }
    if not config_path or not config_path.is_file():
        return meta

    text = config_path.read_text()

    leaf_match = re.search(r"leaf_sizes:\s*([0-9.+-eE]+)", text)
    if leaf_match:
        meta["leaf_size"] = float(leaf_match.group(1))

    map_size_match = re.search(r"map_size:\s*([0-9.+-eE]+)", text)
    if map_size_match:
        meta["map_size"] = float(map_size_match.group(1))

    origin_match = re.search(
        r"map_origin:.*?data:\s*\[\s*([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)\s*,\s*([0-9.+-eE]+)\s*\]",
        text,
        re.DOTALL,
    )
    if origin_match:
        meta["map_origin"] = {
            "x": float(origin_match.group(1)),
            "y": float(origin_match.group(2)),
            "z": float(origin_match.group(3)),
        }

    return meta


def parse_camera_config(config_path: Path):
    camera = {
        "width": None,
        "height": None,
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "tCL": None,
        "tBL": None,
    }
    if not config_path or not config_path.is_file():
        return camera

    text = config_path.read_text()
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        match = re.search(rf"{key}:\s*([0-9.+-eE]+)", text)
        if match:
            camera[key] = float(match.group(1))

    def parse_matrix(name: str):
        match = re.search(
            rf"{name}:\s*!!opencv-matrix.*?data:\s*\[(.*?)\]",
            text,
            re.DOTALL,
        )
        if not match:
            return None
        values = [float(token.strip()) for token in match.group(1).replace("\n", " ").split(",") if token.strip()]
        if len(values) != 16:
            return None
        return values

    camera["tCL"] = parse_matrix("T_C_L")
    camera["tBL"] = parse_matrix("T_B_L")
    return camera


def safe_link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src)


def first_match(root: Path, patterns):
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def quaternion_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 1e-12:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
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


def mat3_transpose_vec3(matrix, vector):
    return (
        matrix[0][0] * vector[0] + matrix[1][0] * vector[1] + matrix[2][0] * vector[2],
        matrix[0][1] * vector[0] + matrix[1][1] * vector[1] + matrix[2][1] * vector[2],
        matrix[0][2] * vector[0] + matrix[1][2] * vector[1] + matrix[2][2] * vector[2],
    )


def mat3_multiply(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def rotation_matrix_to_quaternion_xyzw(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2][1] - matrix[1][2]) / scale
        qy = (matrix[0][2] - matrix[2][0]) / scale
        qz = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        qw = (matrix[2][1] - matrix[1][2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0][1] + matrix[1][0]) / scale
        qz = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        qw = (matrix[0][2] - matrix[2][0]) / scale
        qx = (matrix[0][1] + matrix[1][0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        qw = (matrix[1][0] - matrix[0][1]) / scale
        qx = (matrix[0][2] + matrix[2][0]) / scale
        qy = (matrix[1][2] + matrix[2][1]) / scale
        qz = 0.25 * scale
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return {
        "x": qx / norm,
        "y": qy / norm,
        "z": qz / norm,
        "w": qw / norm,
    }


def normalize_vec3(vector):
    length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if length <= 1e-12:
        return None
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def build_initial_view_from_colmap_images(images_path: Path, frame_id: str):
    if not images_path.is_file():
        return None
    for raw_line in images_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            qw, qx, qy, qz = (float(parts[index]) for index in range(1, 5))
            tx, ty, tz = (float(parts[index]) for index in range(5, 8))
        except ValueError:
            continue

        rotation = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        camera_center = mat3_transpose_vec3(rotation, (-tx, -ty, -tz))
        camera_to_world = tuple(tuple(rotation[col][row] for col in range(3)) for row in range(3))
        renderer_input_rotation = mat3_multiply(
            camera_to_world,
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
        )
        forward = normalize_vec3(mat3_transpose_vec3(rotation, (0.0, 0.0, 1.0)))
        if forward is None:
            continue
        target_distance = 4.0
        return {
            "pose": {
                "frameId": frame_id,
                "position": {
                    "x": camera_center[0],
                    "y": camera_center[1],
                    "z": camera_center[2],
                },
                "orientation": rotation_matrix_to_quaternion_xyzw(renderer_input_rotation),
            },
            "target": {
                "x": camera_center[0] + forward[0] * target_distance,
                "y": camera_center[1] + forward[1] * target_distance,
                "z": camera_center[2] + forward[2] * target_distance,
            },
            "source": "colmap_train_start",
            "sourceImage": parts[9] if len(parts) >= 10 else None,
            "rendererPoseConvention": "gssdf_rviz_opengl_camera",
        }
    return None


PLY_STRUCT_CODES = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def parse_ply_vertex_layout(path: Path):
    fmt = None
    vertex_count = None
    vertex_props = []
    header_size = 0
    in_vertex = False
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            header_size += len(line)
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("format "):
                parts = text.split()
                if len(parts) >= 2:
                    fmt = parts[1]
            elif text.startswith("element "):
                parts = text.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    vertex_props = []
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) == 3:
                    vertex_props.append((parts[2], parts[1]))
            elif text == "end_header":
                break
    if fmt is None or vertex_count is None or not vertex_props:
        return None
    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_props": vertex_props,
        "header_size": header_size,
    }


def sample_ply_positions(path: Path, max_samples: int = 200000):
    layout = parse_ply_vertex_layout(path)
    if layout is None:
        return []
    names = [name for name, _ in layout["vertex_props"]]
    try:
        x_index = names.index("x")
        y_index = names.index("y")
        z_index = names.index("z")
    except ValueError:
        return []

    vertex_count = int(layout["vertex_count"])
    sample_stride = max(1, math.ceil(vertex_count / max_samples))
    positions = []

    if layout["format"] == "ascii":
        with path.open("rb") as f:
            f.seek(layout["header_size"])
            for index in range(vertex_count):
                line = f.readline()
                if not line:
                    break
                if index % sample_stride != 0:
                    continue
                parts = line.decode("utf-8", errors="replace").split()
                if len(parts) <= max(x_index, y_index, z_index):
                    continue
                try:
                    positions.append(
                        (
                            float(parts[x_index]),
                            float(parts[y_index]),
                            float(parts[z_index]),
                        )
                    )
                except ValueError:
                    continue
        return positions

    if layout["format"] != "binary_little_endian":
        return []

    try:
        vertex_struct = struct.Struct(
            "<" + "".join(PLY_STRUCT_CODES[prop_type] for _, prop_type in layout["vertex_props"])
        )
    except KeyError:
        return []

    with path.open("rb") as f:
        f.seek(layout["header_size"])
        for index in range(vertex_count):
            chunk = f.read(vertex_struct.size)
            if len(chunk) != vertex_struct.size:
                break
            if index % sample_stride != 0:
                continue
            values = vertex_struct.unpack(chunk)
            positions.append(
                (
                    float(values[x_index]),
                    float(values[y_index]),
                    float(values[z_index]),
                )
            )
    return positions


def build_initial_view_from_bounds(frame_id: str, positions):
    if len(positions) < 8:
        return None

    xs = sorted(position[0] for position in positions)
    ys = sorted(position[1] for position in positions)
    zs = sorted(position[2] for position in positions)
    count = len(xs)
    trim = min(max(int(count * 0.01), 0), max(count // 4, 0))
    low = trim
    high = count - trim - 1
    if high <= low:
        low = 0
        high = count - 1

    min_x, max_x = xs[low], xs[high]
    min_y, max_y = ys[low], ys[high]
    min_z, max_z = zs[low], zs[high]
    span_x = max_x - min_x
    span_y = max_y - min_y
    span_z = max_z - min_z
    span_xy = max(span_x, span_y, 1.0)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    center_z = (min_z + max_z) * 0.5
    target_z = min_z + span_z * 0.35 if span_z > 1e-4 else center_z
    camera_distance = span_xy * 1.15
    position_x = center_x + camera_distance * 0.92
    position_y = center_y - camera_distance * 0.78
    position_z = max(max_z + max(span_z, span_xy * 0.3), center_z + camera_distance * 0.58)
    return {
        "pose": {
            "frameId": frame_id,
            "position": {
                "x": position_x,
                "y": position_y,
                "z": position_z,
            },
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "w": 1.0,
            },
        },
        "target": {
            "x": center_x,
            "y": center_y,
            "z": target_z,
        },
        "source": "geometry_bounds",
        "sourceImage": None,
    }


def build_initial_view(output_dir: Path, config_path: Path | None, frame_id: str, mesh_ply: Path | None):
    data_path = parse_data_path(config_path) if config_path else None
    if data_path and data_path.is_dir():
        colmap_candidates = [
            data_path / "colmap" / "postrior_lidar" / "images.txt",
            data_path / "sparse" / "0" / "images.txt",
        ]
        for candidate in colmap_candidates:
            initial_view = build_initial_view_from_colmap_images(candidate, frame_id)
            if initial_view is not None:
                return initial_view

    for candidate in (
        mesh_ply,
        output_dir / "model" / "as_occ_prior.ply",
        output_dir / "train_points.ply",
        output_dir / "model" / "gs.ply",
    ):
        if not candidate or not candidate.is_file():
            continue
        positions = sample_ply_positions(candidate)
        initial_view = build_initial_view_from_bounds(frame_id, positions)
        if initial_view is not None:
            return initial_view

    return None


def derive_map_meta_from_geometry(meta: dict, candidates: list[Path | None], padding: float = 1.0) -> dict:
    best_positions = []
    for candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        positions = sample_ply_positions(candidate, max_samples=200000)
        if len(positions) >= 8:
            best_positions = positions
            break

    if len(best_positions) < 8:
        return meta

    xs = sorted(position[0] for position in best_positions)
    ys = sorted(position[1] for position in best_positions)
    count = len(xs)
    trim = min(max(int(count * 0.01), 0), max(count // 4, 0))
    low = trim
    high = count - trim - 1
    if high <= low:
        low = 0
        high = count - 1

    min_x, max_x = xs[low], xs[high]
    min_y, max_y = ys[low], ys[high]
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 1e-6 or span_y <= 1e-6:
        return meta

    derived_size = max(span_x, span_y) + 2.0 * padding
    current_size = meta.get("map_size")
    should_override = (
        current_size is None
        or current_size <= 0
        or current_size > derived_size * 2.0
        or current_size < derived_size * 0.25
    )
    if not should_override:
        return meta

    updated = dict(meta)
    origin = dict(meta.get("map_origin") or {})
    origin["x"] = min_x - padding
    origin["y"] = min_y - padding
    origin.setdefault("z", 0.0)
    updated["map_origin"] = origin
    updated["map_size"] = derived_size
    return updated


def format_variant_label(variant: str):
    return variant.replace("_", " ").replace("-", " ").title()


def build_chunk_source(chunks_manifest, args, scene_id: str, model_root: Path, variant: str):
    chunk_dir_name = "gs_chunks" if variant == "balanced" else f"gs_chunks_{variant}"
    chunk_scene_root = model_root / chunk_dir_name
    if chunk_scene_root.exists() or chunk_scene_root.is_symlink():
        if chunk_scene_root.is_dir() and not chunk_scene_root.is_symlink():
            shutil.rmtree(chunk_scene_root)
        else:
            chunk_scene_root.unlink()
    chunk_scene_root.mkdir(parents=True, exist_ok=True)

    chunk_entries = []
    for chunk in chunks_manifest["chunks"]:
        src = Path(chunk["path"]).resolve()
        dst = chunk_scene_root / chunk["file"]
        safe_link(src, dst)
        chunk_entries.append(
            {
                "id": chunk.get("id"),
                "url": f"{args.url_prefix}/{scene_id}/model/{chunk_dir_name}/{chunk['file']}",
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
        "chunks": chunk_entries,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"Output dir not found: {output_dir}")

    scene_id = args.scene_id or output_dir.name
    scene_root = Path(args.scene_root).resolve() / scene_id
    model_root = scene_root / "model"

    config_path = Path(args.scene_config).resolve() if args.scene_config else None
    if not config_path or not config_path.is_file():
      fallback = output_dir / "model" / "config" / "scene" / "config.yaml"
      if fallback.is_file():
          config_path = fallback

    meta = parse_scene_meta(config_path) if config_path else parse_scene_meta(Path())
    camera = parse_camera_config(config_path) if config_path else parse_camera_config(Path())

    repo_root = Path(__file__).resolve().parent.parent
    processed_model_root = repo_root / "runtime" / "processed" / output_dir.name / "model"
    gs_chunks_meta = processed_model_root / "gs_chunks.json"
    gs_runtime_spz = output_dir / "model" / "gs_runtime.spz"
    gs_runtime_meta = output_dir / "model" / "gs_runtime.json"
    if not gs_runtime_spz.is_file():
        processed_spz = processed_model_root / "gs_runtime.spz"
        processed_meta = processed_model_root / "gs_runtime.json"
        if processed_spz.is_file():
            gs_runtime_spz = processed_spz
        if processed_meta.is_file():
            gs_runtime_meta = processed_meta
    gs_ply = output_dir / "model" / "gs.ply"
    occ_ply = output_dir / "model" / "as_occ_prior.ply"
    train_points = output_dir / "train_points.ply"
    mesh_ply = first_match(output_dir, ["mesh*.ply", "model/mesh*.ply", "model/gs_*.ply"])
    meta = derive_map_meta_from_geometry(meta, [mesh_ply, occ_ply, train_points, gs_ply])

    scene_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    runtime_sh_degree = 1
    if gs_runtime_meta.is_file():
        try:
            runtime_meta = json.loads(gs_runtime_meta.read_text())
            runtime_sh_degree = int(runtime_meta.get("output", {}).get("shDegree", runtime_sh_degree))
        except Exception:
            runtime_sh_degree = 1

    initial_view = build_initial_view(output_dir, config_path, "world", mesh_ply)

    manifest = {
        "schemaVersion": "slam-adapter.scene-manifest/v1alpha1",
        "sceneId": scene_id,
        "frameId": "world",
        "training": {
            "status": args.status,
            "outputDir": str(output_dir),
            "configPath": str(config_path) if config_path else None,
        },
        "robot": {
            "radius": args.robot_radius,
            "height": args.robot_height,
        },
        "meta": {
            "leafSize": meta["leaf_size"],
            "mapOrigin": meta["map_origin"],
            "mapSize": meta["map_size"],
        },
        "camera": camera,
        "assets": {},
        "source": {
            "kind": "gssdf",
            "outputDir": str(output_dir),
            "repoPath": str(repo_root),
            "repoCommit": None,
            "liveContractUrl": "/contracts/live-contract.default.json",
            "capabilityMatrixUrl": "/contracts/adapter-capability-matrix.default.json",
        },
    }
    if initial_view is not None:
        manifest["initialView"] = {
            "pose": initial_view["pose"],
            "target": initial_view["target"],
        }

    gaussian_variants = {}
    if gs_chunks_meta.is_file():
        chunks_manifest = load_json(gs_chunks_meta)
        if chunks_manifest and chunks_manifest.get("chunks"):
            gaussian_variants["balanced"] = build_chunk_source(
                chunks_manifest, args, scene_id, model_root, "balanced"
            )

    for extra_meta in sorted(processed_model_root.glob("gs_chunks_*.json")):
        variant = extra_meta.stem.removeprefix("gs_chunks_")
        if not variant:
            continue
        chunks_manifest = load_json(extra_meta)
        if not chunks_manifest or not chunks_manifest.get("chunks"):
            continue
        gaussian_variants[variant] = build_chunk_source(
            chunks_manifest, args, scene_id, model_root, variant
        )

    if gaussian_variants:
        preferred_variants = ["quality", "balanced", "ultra", "fast"]
        default_variant = next(
            (variant for variant in preferred_variants if variant in gaussian_variants),
            sorted(gaussian_variants.keys())[0],
        )
        manifest["gaussian"] = gaussian_variants[default_variant]
        manifest["gaussianVariants"] = gaussian_variants
        manifest["assets"]["gaussian"] = "ready"
    elif gs_runtime_spz.is_file():
        dst = model_root / "gs_runtime.spz"
        safe_link(gs_runtime_spz, dst)
        manifest["gaussian"] = {
            "format": "spz",
            "url": f"{args.url_prefix}/{scene_id}/model/gs_runtime.spz",
            "shDegree": runtime_sh_degree,
            "variant": "balanced",
            "label": "Balanced",
        }
        manifest["assets"]["gaussian"] = "ready"
    elif gs_ply.is_file():
        dst = model_root / "gs.ply"
        safe_link(gs_ply, dst)
        manifest["gaussian"] = {
            "format": "gs-ply",
            "url": f"{args.url_prefix}/{scene_id}/model/gs.ply",
            "shDegree": 3,
            "variant": "raw",
            "label": "Raw",
        }
        manifest["assets"]["gaussian"] = "ready"
    else:
        manifest["assets"]["gaussian"] = "missing"

    if occ_ply.is_file():
        dst = model_root / "as_occ_prior.ply"
        safe_link(occ_ply, dst)
        manifest["occupancy"] = {
            "source": "as_occ_prior",
            "url": f"{args.url_prefix}/{scene_id}/model/as_occ_prior.ply",
            "resolution": meta["leaf_size"],
            "origin": {
                "x": meta["map_origin"]["x"],
                "y": meta["map_origin"]["y"],
            },
        }
        manifest["assets"]["occupancy"] = "ready"
    else:
        manifest["assets"]["occupancy"] = "missing"

    if mesh_ply and mesh_ply.is_file():
        dst = model_root / mesh_ply.name
        safe_link(mesh_ply, dst)
        manifest["mesh"] = {
            "format": "ply",
            "url": f"{args.url_prefix}/{scene_id}/model/{mesh_ply.name}",
        }
        manifest["assets"]["mesh"] = "ready"
    else:
        manifest["assets"]["mesh"] = "missing"

    if train_points.is_file():
        dst = model_root / "train_points.ply"
        safe_link(train_points, dst)
        manifest["rawPointCloud"] = {
            "format": "ply",
            "url": f"{args.url_prefix}/{scene_id}/model/train_points.ply",
        }
        manifest["assets"]["rawPointCloud"] = "ready"
    elif occ_ply.is_file():
        dst = model_root / "as_occ_prior.ply"
        safe_link(occ_ply, dst)
        manifest["rawPointCloud"] = {
            "format": "ply",
            "url": f"{args.url_prefix}/{scene_id}/model/as_occ_prior.ply",
        }
        manifest["assets"]["rawPointCloud"] = "fallback"
    else:
        manifest["assets"]["rawPointCloud"] = "missing"

    manifest_path = scene_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"sceneRoot": str(scene_root), "manifest": str(manifest_path), "assets": manifest["assets"]}, indent=2))


if __name__ == "__main__":
    main()
