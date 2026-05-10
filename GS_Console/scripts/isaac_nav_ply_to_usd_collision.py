#!/usr/bin/env python3
"""Convert a triangular PLY mesh into a static Isaac USD collision asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a triangular PLY mesh into a USD mesh with Isaac/PhysX collision schema."
    )
    parser.add_argument("--input", required=True, help="Input triangular PLY mesh.")
    parser.add_argument("--output", required=True, help="Output USD path.")
    parser.add_argument(
        "--prim-path",
        default="/World/Environment/CollisionMesh",
        help="USD prim path for the collision mesh.",
    )
    parser.add_argument(
        "--collision-approximation",
        default="meshSimplification",
        choices=["convexDecomposition", "convexHull", "boundingCube", "boundingSphere", "meshSimplification", "none"],
        help="UsdPhysics MeshCollisionAPI approximation token.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Optional deterministic face cap for smoke-test assets. Omit for full mesh.",
    )
    parser.add_argument("--meters-per-unit", type=float, default=1.0, help="USD meters-per-unit metadata.")
    parser.add_argument("--width", type=int, default=64, help="Headless Isaac render width.")
    parser.add_argument("--height", type=int, default=64, help="Headless Isaac render height.")
    parser.add_argument(
        "--gaussian-ply",
        default="",
        help=(
            "Optional GS-SDF/3DGS PLY. If the mesh has uniform vertex colors, "
            "transfer nearest-neighbor SH DC colors from this Gaussian cloud."
        ),
    )
    parser.add_argument(
        "--gaussian-color-transfer-max-distance",
        type=float,
        default=0.0,
        help="Optional nearest Gaussian distance cutoff for color transfer. 0 uses all nearest matches.",
    )
    return parser.parse_args()


def _vt_int_array(values):
    from pxr import Vt

    try:
        return Vt.IntArray.FromNumpy(values)
    except Exception:
        return Vt.IntArray(values.tolist())


def _vt_vec3f_array(values):
    from pxr import Gf, Vt

    try:
        return Vt.Vec3fArray.FromNumpy(values)
    except Exception:
        return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in values])


def _vt_float_array(values):
    from pxr import Vt

    try:
        return Vt.FloatArray.FromNumpy(values)
    except Exception:
        return Vt.FloatArray([float(value) for value in values])


def _ply_vertex_dtype(properties: list[tuple[str, str]]):
    import numpy as np

    type_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    return np.dtype([(name, type_map[typ]) for name, typ in properties])


def _load_binary_little_endian_ply_vertices(path: Path):
    import numpy as np

    vertex_count = None
    vertex_properties: list[tuple[str, str]] = []
    header_size = 0
    in_vertex = False
    ply_format = None
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            header_size += len(line)
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("format "):
                parts = text.split()
                if len(parts) >= 2:
                    ply_format = parts[1]
            elif text.startswith("element "):
                parts = text.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    vertex_properties = []
            elif in_vertex and text.startswith("property "):
                parts = text.split()
                if len(parts) == 3:
                    vertex_properties.append((parts[2], parts[1]))
            elif text == "end_header":
                break

        if ply_format != "binary_little_endian" or vertex_count is None:
            return None
        dtype = _ply_vertex_dtype(vertex_properties)
        stream.seek(header_size)
        return np.fromfile(stream, dtype=dtype, count=vertex_count)


def _load_gaussian_dc_colors(path: Path):
    import numpy as np

    vertices = _load_binary_little_endian_ply_vertices(path)
    if vertices is None:
        return None
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    if not required.issubset(vertices.dtype.names or ()):
        return None
    positions = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)
    dc = np.stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(dc * 0.28209479177387814 + 0.5, 0.0, 1.0).astype(np.float32)
    return positions, colors


def _vertex_colors_are_uniform(vertex_colors) -> bool:
    import numpy as np

    if vertex_colors is None or vertex_colors.size == 0:
        return True
    color_channels = vertex_colors[:, :3]
    return bool(float(np.ptp(color_channels, axis=0).max()) < 1e-3)


def _transfer_gaussian_colors(vertices, original_colors, gaussian_ply: Path, max_distance: float):
    import numpy as np
    from scipy.spatial import cKDTree

    loaded = _load_gaussian_dc_colors(gaussian_ply)
    if loaded is None:
        return original_colors, None
    gaussian_positions, gaussian_colors = loaded
    distances, indices = cKDTree(gaussian_positions).query(vertices, k=1, workers=-1)
    transferred = gaussian_colors[indices].astype(np.float32, copy=True)
    if max_distance > 0.0 and original_colors is not None:
        far_mask = distances > max_distance
        transferred[far_mask] = original_colors[far_mask]
    return transferred, {
        "gaussian_ply": str(gaussian_ply),
        "nearest_distance_mean": float(np.mean(distances)),
        "nearest_distance_max": float(np.max(distances)),
    }


def _write_usd(args: argparse.Namespace) -> dict:
    import numpy as np
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(input_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if len(mesh.vertices) else np.zeros((0, 3), dtype=np.float32)
    visual = getattr(mesh, "visual", None)
    vertex_colors = None
    if visual is not None:
        raw_vertex_colors = getattr(visual, "vertex_colors", None)
        if raw_vertex_colors is not None:
            raw_vertex_colors = np.asarray(raw_vertex_colors)
            if raw_vertex_colors.ndim == 2 and raw_vertex_colors.shape[0] == vertices.shape[0] and raw_vertex_colors.shape[1] >= 3:
                vertex_colors = raw_vertex_colors[:, :3].astype(np.float32) / 255.0
                if raw_vertex_colors.shape[1] >= 4:
                    alpha = raw_vertex_colors[:, 3].astype(np.float32) / 255.0
                else:
                    alpha = None
            else:
                alpha = None
        else:
            alpha = None
    else:
        alpha = None
    original_vertex_colors_uniform = _vertex_colors_are_uniform(vertex_colors)
    color_transfer = None
    if args.gaussian_ply and original_vertex_colors_uniform:
        gaussian_ply = Path(args.gaussian_ply).expanduser().resolve()
        if gaussian_ply.is_file():
            vertex_colors, color_transfer = _transfer_gaussian_colors(
                vertices,
                vertex_colors,
                gaussian_ply,
                max(0.0, float(args.gaussian_color_transfer_max_distance)),
            )
            if color_transfer is not None:
                alpha = None
    original_vertex_count = int(vertices.shape[0])
    original_face_count = int(faces.shape[0])

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces, got face array shape {faces.shape}")

    if args.max_faces is not None and original_face_count > args.max_faces:
        faces = faces[: args.max_faces]
        used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used_vertices]
        if vertex_normals.shape[0] == original_vertex_count:
            vertex_normals = vertex_normals[used_vertices]
        if vertex_colors is not None and vertex_colors.shape[0] == original_vertex_count:
            vertex_colors = vertex_colors[used_vertices]
        if alpha is not None and alpha.shape[0] == original_vertex_count:
            alpha = alpha[used_vertices]
        faces = inverse.reshape((-1, 3)).astype(np.int32, copy=False)

    face_vertex_counts = np.full((faces.shape[0],), 3, dtype=np.int32)
    face_vertex_indices = faces.reshape(-1).astype(np.int32, copy=False)
    bounds_min = vertices.min(axis=0).astype(float).tolist()
    bounds_max = vertices.max(axis=0).astype(float).tolist()

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, args.meters_per_unit)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Environment")

    usd_mesh = UsdGeom.Mesh.Define(stage, args.prim_path)
    usd_mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    usd_mesh.CreateDoubleSidedAttr().Set(True)
    usd_mesh.CreatePointsAttr().Set(_vt_vec3f_array(vertices))
    usd_mesh.CreateFaceVertexCountsAttr().Set(_vt_int_array(face_vertex_counts))
    usd_mesh.CreateFaceVertexIndicesAttr().Set(_vt_int_array(face_vertex_indices))
    if vertex_normals.shape == vertices.shape and vertex_normals.size:
        usd_mesh.CreateNormalsAttr().Set(_vt_vec3f_array(vertex_normals))
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if vertex_colors is not None and vertex_colors.shape == vertices.shape:
        usd_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(_vt_vec3f_array(vertex_colors))
        if alpha is not None and alpha.shape[0] == vertices.shape[0] and float(np.min(alpha)) < 0.999:
            usd_mesh.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex).Set(_vt_float_array(alpha))
    else:
        usd_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.52, 0.56, 0.60)])

    prim = usd_mesh.GetPrim()
    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr().Set(args.collision_approximation != "none")
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr().Set(args.collision_approximation)

    stage.GetRootLayer().Save()

    manifest = {
        "input": str(input_path),
        "output": str(output_path),
        "prim_path": args.prim_path,
        "collision_approximation": args.collision_approximation,
        "meters_per_unit": args.meters_per_unit,
        "original_vertex_count": original_vertex_count,
        "original_face_count": original_face_count,
        "written_vertex_count": int(vertices.shape[0]),
        "written_face_count": int(faces.shape[0]),
        "has_vertex_colors": bool(vertex_colors is not None and vertex_colors.shape == vertices.shape),
        "has_vertex_alpha": bool(alpha is not None and alpha.shape[0] == vertices.shape[0]),
        "has_vertex_normals": bool(vertex_normals.shape == vertices.shape and vertex_normals.size),
        "original_vertex_colors_uniform": original_vertex_colors_uniform,
        "color_transfer": color_transfer,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": args.width,
            "height": args.height,
            "multi_gpu": False,
            "fast_shutdown": True,
        }
    )

    manifest = _write_usd(args)
    print(json.dumps(manifest, indent=2), flush=True)
    app.update()
    app.close(wait_for_replicator=False)


if __name__ == "__main__":
    main()
