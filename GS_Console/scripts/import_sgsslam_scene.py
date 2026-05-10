#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

from import_citygaussian_scene import (
    SCENE_ROOT_DEFAULT,
    PlannedSceneAssetLinker,
    SceneAssetLinker,
    build_chunk_source,
    build_initial_view,
    compute_ply_bounds,
    ensure_file,
    format_variant_label,
    gaussian_format_for_path,
    infer_sh_degree_from_ply_header,
    load_json,
    parse_grid_arg,
    safe_link,
    sanitize_scene_id,
)


C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import an SGS-SLAM experiment directory into the GS-SDF web scene-pack layout."
    )
    parser.add_argument(
        "--sgsslam-dir",
        required=True,
        help="SGS-SLAM experiment directory, usually experiments/<group>/<run-name>",
    )
    parser.add_argument("--scene-id", help="Scene id written under public/scenes/<scene-id>")
    parser.add_argument("--scene-root", default=SCENE_ROOT_DEFAULT, help="Web scene root directory")
    parser.add_argument("--url-prefix", default="/scenes", help="Public URL prefix for the scene root")
    parser.add_argument("--frame-id", default="world", help="Frame id to expose in manifest.json")
    parser.add_argument("--status", default="completed", help="Training status label")
    parser.add_argument(
        "--gaussian-path",
        help="Override gaussian asset (.ply, .splat, .spz, or .npz). Defaults to params.ply or params.npz.",
    )
    parser.add_argument("--mesh-path", help="Optional mesh asset (.ply or .glb)")
    parser.add_argument(
        "--occupancy-path",
        help="Optional PLY used by TopDownMap. Defaults to exported params_rgb.ply or mesh fallback.",
    )
    parser.add_argument(
        "--raw-pointcloud-path",
        help="Optional static point cloud for the 3D overlay. Defaults to exported params_rgb.ply.",
    )
    parser.add_argument(
        "--allow-gaussian-fallback",
        action="store_true",
        help="Allow using gaussian PLY as occupancy/raw fallback when no lighter RGB point cloud exists.",
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
        "--sgsslam-repo",
        default="/home/chatsign/sgs-slam",
        help="Path to the cloned SGS-SLAM repository",
    )
    parser.add_argument(
        "--gpu-input-sequence",
        help="Explicit SGS input sequence name under <sgsslam-repo>/gpu_input_root/",
    )
    parser.add_argument(
        "--gpu-input-dir",
        help="Explicit GPU input directory. Overrides --gpu-input-sequence auto detection.",
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
    parser.add_argument("--max-sh", type=int, default=0, help="Max SH degree kept during chunk preprocessing")
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.01,
        help="Opacity threshold passed to preprocess_gaussian_stream.mjs",
    )
    parser.add_argument(
        "--raw-opacity-threshold",
        type=float,
        default=0.05,
        help="Opacity threshold used when exporting params_rgb.ply from params.npz",
    )
    parser.add_argument(
        "--raw-max-points",
        type=int,
        default=300000,
        help="Keep at most this many highest-opacity points in exported params_rgb.ply",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        help="Override manifest gaussian.shDegree. By default it is inferred from the PLY header.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing scene-pack links and exported assets")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned manifest without writing files")
    return parser.parse_args()


def require_numpy() -> None:
    if np is None:
        raise SystemExit("numpy is required for SGS-SLAM import helpers")


def sigmoid(values):
    require_numpy()
    return 1.0 / (1.0 + np.exp(-values))


def maybe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def maybe_int(value: object) -> Optional[int]:
    try:
        parsed = int(round(float(value)))
    except Exception:
        return None
    return parsed if parsed > 0 else None


def git_commit(path: Path) -> Optional[str]:
    git_head = path / ".git"
    if not git_head.exists():
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def find_latest_params_npz(root: Path) -> Optional[Path]:
    final_npz = root / "params.npz"
    if final_npz.is_file():
        return final_npz

    best_path: Optional[Path] = None
    best_index = -1
    for candidate in root.glob("params*.npz"):
        match = re.fullmatch(r"params(\d+)\.npz", candidate.name)
        if not match:
            continue
        index = int(match.group(1))
        if index > best_index:
            best_index = index
            best_path = candidate
    return best_path


def planned_export_root(repo_root: Path, scene_id: str) -> Path:
    return repo_root / "runtime" / "imports" / "sgsslam" / scene_id / "exported"


def detect_gaussian_asset(root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override

    for relative in ("params.ply", "scene_point_cloud.ply", "scene_point_cloud.splat", "scene_point_cloud.spz"):
        candidate = root / relative
        if candidate.is_file():
            return candidate

    return find_latest_params_npz(root)


def detect_mesh_asset(root: Path, override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override

    patterns = (
        "mesh/fuse_post.ply",
        "mesh/fuse.ply",
        "mesh/*.ply",
        "mesh/*.glb",
        "eval/mesh*.ply",
        "*.glb",
        "*.ply",
    )
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            if candidate.name in {"params.ply", "params_rgb.ply", "seg_params.ply"}:
                continue
            return candidate
    return None


def gaussian_dtype() -> List[Tuple[str, str]]:
    return [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("nx", "<f4"),
        ("ny", "<f4"),
        ("nz", "<f4"),
        ("f_dc_0", "<f4"),
        ("f_dc_1", "<f4"),
        ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"),
        ("scale_1", "<f4"),
        ("scale_2", "<f4"),
        ("rot_0", "<f4"),
        ("rot_1", "<f4"),
        ("rot_2", "<f4"),
        ("rot_3", "<f4"),
    ]


def write_binary_ply(path: Path, vertices, properties: List[Tuple[str, str]]) -> None:
    property_types = {
        "<f4": "float",
        "<u1": "uchar",
        "u1": "uchar",
    }
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
    ]
    for name, dtype_name in properties:
        ply_type = property_types.get(dtype_name, property_types.get(str(dtype_name)))
        if ply_type is None:
            raise ValueError(f"Unsupported PLY dtype {dtype_name} for property {name}")
        header_lines.append(f"property {ply_type} {name}")
    header_lines.append("end_header")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertices.tofile(handle)


def export_gaussian_ply(npz_path: Path, output_path: Path, force: bool) -> Path:
    require_numpy()
    if output_path.exists() and not force:
        return output_path

    params = np.load(npz_path)
    xyz = np.asarray(params["means3D"], dtype=np.float32)
    rgb = np.asarray(params["rgb_colors"], dtype=np.float32)
    f_dc = (rgb - 0.5) / C0
    normals = np.zeros_like(xyz, dtype=np.float32)
    opacity = np.asarray(params["logit_opacities"], dtype=np.float32).reshape(-1, 1)
    scale = np.asarray(params["log_scales"], dtype=np.float32)
    if scale.ndim == 1:
        scale = scale.reshape(-1, 1)
    if scale.shape[1] == 1:
        scale = np.repeat(scale, 3, axis=1)
    elif scale.shape[1] != 3:
        raise ValueError(f"Unsupported log_scales shape in {npz_path}: {scale.shape}")
    rotation = np.asarray(params["unnorm_rotations"], dtype=np.float32)
    if rotation.ndim != 2 or rotation.shape[1] != 4:
        raise ValueError(f"Unsupported unnorm_rotations shape in {npz_path}: {rotation.shape}")

    vertices = np.empty(xyz.shape[0], dtype=gaussian_dtype())
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    vertices["f_dc_0"] = f_dc[:, 0]
    vertices["f_dc_1"] = f_dc[:, 1]
    vertices["f_dc_2"] = f_dc[:, 2]
    vertices["opacity"] = opacity[:, 0]
    vertices["scale_0"] = scale[:, 0]
    vertices["scale_1"] = scale[:, 1]
    vertices["scale_2"] = scale[:, 2]
    vertices["rot_0"] = rotation[:, 0]
    vertices["rot_1"] = rotation[:, 1]
    vertices["rot_2"] = rotation[:, 2]
    vertices["rot_3"] = rotation[:, 3]

    write_binary_ply(output_path, vertices, gaussian_dtype())
    return output_path


def rgb_point_dtype() -> List[Tuple[str, str]]:
    return [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]


def export_rgb_pointcloud(
    npz_path: Path,
    output_path: Path,
    opacity_threshold: float,
    max_points: Optional[int],
    force: bool,
) -> Tuple[Path, int]:
    require_numpy()
    if output_path.exists() and not force:
        with output_path.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return output_path, int(line.split()[2])
                if line == "end_header":
                    break
        return output_path, 0

    params = np.load(npz_path)
    xyz = np.asarray(params["means3D"], dtype=np.float32)
    rgb = np.asarray(params["rgb_colors"], dtype=np.float32)
    opacity = sigmoid(np.asarray(params["logit_opacities"], dtype=np.float32).reshape(-1))

    keep_mask = opacity >= opacity_threshold
    if not np.any(keep_mask):
        keep_mask = opacity > 0
    xyz = xyz[keep_mask]
    rgb = rgb[keep_mask]
    opacity = opacity[keep_mask]

    if max_points is not None and max_points > 0 and xyz.shape[0] > max_points:
        keep_idx = np.argpartition(opacity, -max_points)[-max_points:]
        keep_idx.sort()
        xyz = xyz[keep_idx]
        rgb = rgb[keep_idx]

    rgb_u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    vertices = np.empty(xyz.shape[0], dtype=rgb_point_dtype())
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"] = rgb_u8[:, 0]
    vertices["green"] = rgb_u8[:, 1]
    vertices["blue"] = rgb_u8[:, 2]

    write_binary_ply(output_path, vertices, rgb_point_dtype())
    return output_path, int(xyz.shape[0])


def infer_gpu_input_dir(args: argparse.Namespace, experiment_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    if args.gpu_input_dir:
        gpu_input_dir = Path(args.gpu_input_dir).expanduser().resolve()
        return (gpu_input_dir if gpu_input_dir.is_dir() else None), gpu_input_dir.name

    repo_root = Path(args.sgsslam_repo).expanduser().resolve()
    base_root = repo_root / "gpu_input_root"
    if not base_root.is_dir():
        return None, None

    candidates: List[str] = []
    if args.gpu_input_sequence:
        candidates.append(args.gpu_input_sequence)
    else:
        run_name = experiment_dir.name
        stripped = re.sub(r"_\d+$", "", run_name)
        if stripped != run_name:
            candidates.append(stripped)
        candidates.append(run_name)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        gpu_input_dir = base_root / candidate
        if gpu_input_dir.is_dir():
            return gpu_input_dir, candidate
    return None, None


def load_camera_intrinsics(gpu_input_dir: Optional[Path]) -> Optional[dict]:
    if gpu_input_dir is None:
        return None

    meta_json = gpu_input_dir / "meta" / "camera_intrinsics.json"
    if meta_json.is_file():
        try:
            payload = json.loads(meta_json.read_text())
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    intrinsic_txt = gpu_input_dir / "intrinsic" / "intrinsic_color.txt"
    if not intrinsic_txt.is_file():
        return None
    try:
        rows = []
        for line in intrinsic_txt.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(part) for part in stripped.split()])
        if len(rows) < 3 or len(rows[0]) < 3:
            return None
        fx = rows[0][0]
        fy = rows[1][1]
        cx = rows[0][2]
        cy = rows[1][2]
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }
    except Exception:
        return None


def build_camera_entry(intrinsics: Optional[dict]) -> Optional[dict]:
    if not intrinsics:
        return None
    width = maybe_int(intrinsics.get("image_width") or intrinsics.get("width"))
    height = maybe_int(intrinsics.get("image_height") or intrinsics.get("height"))
    fx = maybe_float(intrinsics.get("fx"))
    fy = maybe_float(intrinsics.get("fy"))
    cx = maybe_float(intrinsics.get("cx"))
    cy = maybe_float(intrinsics.get("cy"))
    if width is None and height is None and fx is None and fy is None and cx is None and cy is None:
        return None
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def parse_pose_matrix(path: Path) -> Optional[List[List[float]]]:
    rows: List[List[float]] = []
    try:
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append([float(part) for part in stripped.split()])
    except Exception:
        return None
    if len(rows) < 3 or any(len(row) < 4 for row in rows[:3]):
        return None
    return [row[:4] for row in rows[:4]]


def load_pose_cameras(gpu_input_dir: Optional[Path], max_samples: int = 120) -> List[dict]:
    if gpu_input_dir is None:
        return []
    pose_dir = gpu_input_dir / "pose"
    if not pose_dir.is_dir():
        return []

    pose_files = sorted(pose_dir.glob("*.txt"))
    if not pose_files:
        return []

    step = max(len(pose_files) // max(max_samples, 1), 1)
    sampled = pose_files[::step]
    if sampled[-1] != pose_files[-1]:
        sampled.append(pose_files[-1])

    cameras: List[dict] = []
    for path in sampled:
        matrix = parse_pose_matrix(path)
        if matrix is None:
            continue
        cameras.append(
            {
                "position": [matrix[0][3], matrix[1][3], matrix[2][3]],
                "rotation": [
                    [matrix[0][0], matrix[0][1], matrix[0][2]],
                    [matrix[1][0], matrix[1][1], matrix[1][2]],
                    [matrix[2][0], matrix[2][1], matrix[2][2]],
                ],
            }
        )
    return cameras


def chunk_dir_name(variant: str) -> str:
    return "gs_chunks" if variant == "balanced" else f"gs_chunks_{variant}"


def ensure_preprocessed_chunks(
    args: argparse.Namespace,
    repo_root: Path,
    source_ply: Path,
    scene_id: str,
) -> Path:
    processed_meta = repo_root / "runtime" / "processed" / scene_id / "model" / (
        "gs_chunks.json" if args.variant == "balanced" else f"gs_chunks_{args.variant}.json"
    )
    if processed_meta.is_file() and not args.force:
        return processed_meta

    if source_ply.suffix.lower() != ".ply":
        raise SystemExit("--preprocess-gaussian only supports .ply gaussian assets")

    stage_root = repo_root / "runtime" / "imports" / "sgsslam" / scene_id
    stage_model_root = stage_root / "model"
    stage_model_root.mkdir(parents=True, exist_ok=True)
    safe_link(source_ply, stage_model_root / "gs.ply")

    grid_x, grid_y = parse_grid_arg(args.grid)
    import subprocess

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


def choose_bounds_source_existing(candidates: List[Optional[Path]]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".ply":
            return candidate
    return None


def select_occupancy_and_raw_assets(
    gaussian_path: Path,
    mesh_path: Optional[Path],
    occupancy_override: Optional[Path],
    raw_pointcloud_override: Optional[Path],
    exported_rgb_pointcloud: Optional[Path],
    allow_gaussian_fallback: bool,
) -> Tuple[Optional[Path], Optional[Path], str, str]:
    occupancy_path = occupancy_override
    occupancy_state = "ready" if occupancy_override else "missing"

    if occupancy_path is None:
        if exported_rgb_pointcloud is not None:
            occupancy_path = exported_rgb_pointcloud
            occupancy_state = "fallback"
        elif mesh_path is not None and mesh_path.suffix.lower() == ".ply":
            occupancy_path = mesh_path
            occupancy_state = "fallback"
        elif raw_pointcloud_override is not None and raw_pointcloud_override.suffix.lower() == ".ply":
            occupancy_path = raw_pointcloud_override
            occupancy_state = "fallback"
        elif allow_gaussian_fallback and gaussian_path.suffix.lower() == ".ply":
            occupancy_path = gaussian_path
            occupancy_state = "fallback"

    raw_pointcloud_path = raw_pointcloud_override
    raw_state = "ready" if raw_pointcloud_override else "missing"
    if raw_pointcloud_path is None:
        if exported_rgb_pointcloud is not None:
            raw_pointcloud_path = exported_rgb_pointcloud
            raw_state = "fallback"
        elif occupancy_override is not None:
            raw_pointcloud_path = occupancy_override
            raw_state = "fallback"
        elif mesh_path is not None and mesh_path.suffix.lower() in {".ply", ".pcd"}:
            raw_pointcloud_path = mesh_path
            raw_state = "fallback"
        elif occupancy_path is not None and occupancy_path.suffix.lower() in {".ply", ".pcd"}:
            raw_pointcloud_path = occupancy_path
            raw_state = "fallback"
        elif allow_gaussian_fallback and gaussian_path.suffix.lower() == ".ply":
            raw_pointcloud_path = gaussian_path
            raw_state = "fallback"

    return occupancy_path, raw_pointcloud_path, occupancy_state, raw_state


def build_manifest(
    args: argparse.Namespace,
    scene_id: str,
    experiment_dir: Path,
    repo_root: Path,
    scene_root: Path,
    model_root: Path,
    gaussian_path: Path,
    mesh_path: Optional[Path],
    occupancy_path: Optional[Path],
    raw_pointcloud_path: Optional[Path],
    occupancy_state: str,
    raw_state: str,
    gpu_input_dir: Optional[Path],
    gpu_input_sequence: Optional[str],
    write_assets: bool,
) -> dict:
    if write_assets:
        scene_root.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)
        linker: SceneAssetLinker = SceneAssetLinker(model_root, scene_id, args.url_prefix)
    else:
        linker = PlannedSceneAssetLinker(model_root, scene_id, args.url_prefix)

    camera_entry = build_camera_entry(load_camera_intrinsics(gpu_input_dir))
    initial_view, _ = build_initial_view(load_pose_cameras(gpu_input_dir), args.frame_id)

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

    bounds_source = choose_bounds_source_existing(
        [occupancy_path, raw_pointcloud_path, mesh_path, gaussian_path]
    )
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

    config_path = experiment_dir / "config.py"
    manifest = {
        "schemaVersion": "slam-adapter.scene-manifest/v1alpha1",
        "sceneId": scene_id,
        "frameId": args.frame_id,
        "gaussian": gaussian_manifest,
        "training": {
            "status": args.status,
            "outputDir": str(experiment_dir),
            "configPath": str(config_path) if config_path.is_file() else None,
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
            "kind": "sgsslam",
            "outputDir": str(experiment_dir),
            "repoPath": str(Path(args.sgsslam_repo).expanduser().resolve()),
            "repoCommit": git_commit(Path(args.sgsslam_repo).expanduser().resolve()),
            "liveContractUrl": "/contracts/live-contract.default.json",
            "capabilityMatrixUrl": "/contracts/adapter-capability-matrix.default.json",
            "gpuInputDir": str(gpu_input_dir) if gpu_input_dir is not None else None,
            "gpuInputSequence": gpu_input_sequence,
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
            "source": "mesh_projection"
            if occupancy_state == "fallback" and mesh_path and occupancy_path.resolve() == mesh_path.resolve()
            else "prebuilt_grid",
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
        manifest["assets"]["rawPointCloud"] = raw_state

    return manifest


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    experiment_dir = Path(args.sgsslam_dir).expanduser().resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"SGS-SLAM experiment dir not found: {experiment_dir}")

    scene_id = sanitize_scene_id(args.scene_id or experiment_dir.name)
    gaussian_override = ensure_file(args.gaussian_path, "Gaussian asset")
    mesh_override = ensure_file(args.mesh_path, "Mesh asset")
    occupancy_override = ensure_file(args.occupancy_path, "Occupancy asset")
    raw_override = ensure_file(args.raw_pointcloud_path, "Raw point cloud asset")

    gaussian_source = detect_gaussian_asset(experiment_dir, gaussian_override)
    if gaussian_source is None:
        raise SystemExit("No SGS-SLAM gaussian asset found. Expected params.ply or params.npz.")

    source_npz: Optional[Path] = None
    if gaussian_source.suffix.lower() == ".npz":
        source_npz = gaussian_source
        gaussian_path = planned_export_root(repo_root, scene_id) / "params.ply"
        if not args.dry_run:
            gaussian_path = export_gaussian_ply(source_npz, gaussian_path, force=args.force)
    else:
        gaussian_path = gaussian_source
        default_npz = experiment_dir / "params.npz"
        source_npz = default_npz if default_npz.is_file() else find_latest_params_npz(experiment_dir)

    exported_rgb_pointcloud: Optional[Path] = None
    if raw_override is None and occupancy_override is None and source_npz is not None:
        exported_rgb_pointcloud = planned_export_root(repo_root, scene_id) / "params_rgb.ply"
        if not args.dry_run:
            exported_rgb_pointcloud, _ = export_rgb_pointcloud(
                source_npz,
                exported_rgb_pointcloud,
                opacity_threshold=args.raw_opacity_threshold,
                max_points=args.raw_max_points,
                force=args.force,
            )

    mesh_path = detect_mesh_asset(experiment_dir, mesh_override)
    gpu_input_dir, gpu_input_sequence = infer_gpu_input_dir(args, experiment_dir)
    occupancy_path, raw_pointcloud_path, occupancy_state, raw_state = select_occupancy_and_raw_assets(
        gaussian_path=gaussian_path,
        mesh_path=mesh_path,
        occupancy_override=occupancy_override,
        raw_pointcloud_override=raw_override,
        exported_rgb_pointcloud=exported_rgb_pointcloud,
        allow_gaussian_fallback=args.allow_gaussian_fallback,
    )

    scene_root = Path(args.scene_root).expanduser().resolve() / scene_id
    model_root = scene_root / "model"
    manifest = build_manifest(
        args=args,
        scene_id=scene_id,
        experiment_dir=experiment_dir,
        repo_root=repo_root,
        scene_root=scene_root,
        model_root=model_root,
        gaussian_path=gaussian_path,
        mesh_path=mesh_path,
        occupancy_path=occupancy_path,
        raw_pointcloud_path=raw_pointcloud_path,
        occupancy_state=occupancy_state,
        raw_state=raw_state,
        gpu_input_dir=gpu_input_dir,
        gpu_input_sequence=gpu_input_sequence,
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
                    "sourceNpz": str(source_npz) if source_npz else None,
                    "meshPath": str(mesh_path) if mesh_path else None,
                    "occupancyPath": str(occupancy_path) if occupancy_path else None,
                    "rawPointCloudPath": str(raw_pointcloud_path) if raw_pointcloud_path else None,
                    "gpuInputDir": str(gpu_input_dir) if gpu_input_dir else None,
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
                "gpuInputDir": str(gpu_input_dir) if gpu_input_dir else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
