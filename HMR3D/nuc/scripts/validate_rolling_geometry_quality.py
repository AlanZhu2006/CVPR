#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _read_ascii_ply(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file")
        vertex_count = 0
        properties: list[str] = []
        in_vertex = False
        header_lines = 1
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has no end_header")
            header_lines += 1
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                properties.append(line.split()[-1])
            if line == "end_header":
                break
    if vertex_count <= 0:
        return {
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "rgb": np.zeros((0, 3), dtype=np.uint8),
        }
    data = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    index = {name: idx for idx, name in enumerate(properties)}
    xyz = data[:, [index["x"], index["y"], index["z"]]].astype(np.float32)
    if all(name in index for name in ("red", "green", "blue")):
        rgb = np.clip(data[:, [index["red"], index["green"], index["blue"]]], 0, 255).astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 180, dtype=np.uint8)
    return {"xyz": xyz, "rgb": rgb}


def _read_trajectory(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split()
        if len(parts) >= 4:
            rows.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)


def _path_length(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def _sample(xyz: np.ndarray, max_points: int) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    idx = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
    return xyz[idx]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0.0}
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0.0}
    return {
        "count": float(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def _cloud_stats(xyz: np.ndarray, rgb: np.ndarray, trajectory: np.ndarray | None = None) -> dict[str, Any]:
    finite = np.isfinite(xyz).all(axis=1) if xyz.size else np.zeros((0,), dtype=bool)
    xyz_f = xyz[finite]
    stats: dict[str, Any] = {
        "point_count": int(xyz.shape[0]),
        "finite_ratio": float(np.mean(finite)) if finite.size else 1.0,
    }
    if xyz_f.shape[0] == 0:
        return stats
    bbox_min = xyz_f.min(axis=0)
    bbox_max = xyz_f.max(axis=0)
    span = bbox_max - bbox_min
    centroid = xyz_f.mean(axis=0)
    radius = np.linalg.norm(xyz_f - centroid, axis=1)
    stats.update(
        {
            "bbox_min": bbox_min.astype(float).tolist(),
            "bbox_max": bbox_max.astype(float).tolist(),
            "bbox_span": span.astype(float).tolist(),
            "bbox_volume": float(np.prod(np.maximum(span, 1e-6))),
            "centroid": centroid.astype(float).tolist(),
            "radius": _percentiles(radius),
            "z": _percentiles(xyz_f[:, 2]),
            "rgb_std": np.std(rgb.astype(np.float32), axis=0).astype(float).tolist() if rgb.size else [0.0, 0.0, 0.0],
        }
    )
    if trajectory is not None and trajectory.shape[0] > 0:
        sample = _sample(xyz_f, 12000)
        distances, _ = cKDTree(trajectory).query(sample, k=1)
        stats["distance_to_trajectory"] = _percentiles(distances.astype(np.float32))
        traj_distances, _ = cKDTree(sample).query(trajectory, k=1)
        stats["trajectory_to_cloud_distance"] = _percentiles(traj_distances.astype(np.float32))
    return stats


def _bbox_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    if "bbox_min" not in left or "bbox_min" not in right:
        return 0.0
    lmin = np.asarray(left["bbox_min"], dtype=np.float32)
    lmax = np.asarray(left["bbox_max"], dtype=np.float32)
    rmin = np.asarray(right["bbox_min"], dtype=np.float32)
    rmax = np.asarray(right["bbox_max"], dtype=np.float32)
    inter = np.maximum(0.0, np.minimum(lmax, rmax) - np.maximum(lmin, rmin))
    inter_vol = float(np.prod(np.maximum(inter, 0.0)))
    min_vol = max(1e-6, min(float(left.get("bbox_volume", 0.0)), float(right.get("bbox_volume", 0.0))))
    return float(inter_vol / min_vol)


def _pair_metrics(left_xyz: np.ndarray, right_xyz: np.ndarray) -> dict[str, Any]:
    if left_xyz.shape[0] == 0 or right_xyz.shape[0] == 0:
        return {"nearest_distance": {"count": 0.0}}
    left_sample = _sample(left_xyz[np.isfinite(left_xyz).all(axis=1)], 7000)
    right_sample = _sample(right_xyz[np.isfinite(right_xyz).all(axis=1)], 7000)
    distances, _ = cKDTree(left_sample).query(right_sample, k=1)
    return {"nearest_distance": _percentiles(distances.astype(np.float32))}


def _window_index(path: Path) -> int:
    match = re.search(r"window_(\d+)_", path.name)
    return int(match.group(1)) if match else -1


def _quality_flags(report: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    rolling = report.get("rolling_map", {}).get("stats", {})
    if rolling.get("point_count", 0) <= 0:
        flags.append("rolling_map_empty")
    rolling_span = rolling.get("bbox_span", [0.0, 0.0, 0.0])
    if max(rolling_span or [0.0]) > 80.0:
        flags.append("rolling_map_span_too_large")
    rolling_dist = rolling.get("distance_to_trajectory", {})
    if rolling_dist.get("median", 0.0) > 10.0:
        flags.append("rolling_map_far_from_trajectory")
    for item in report.get("windows", []):
        local = item.get("local_stats", {})
        world = item.get("world_stats", {})
        if local.get("point_count", 0) <= 0:
            flags.append(f"window_{item['index']:06d}_local_empty")
        if world.get("point_count", 0) <= 0:
            flags.append(f"window_{item['index']:06d}_world_empty")
        if local.get("z", {}).get("median", 1.0) <= 0.0:
            flags.append(f"window_{item['index']:06d}_local_z_not_forward")
        if max(world.get("bbox_span", [0.0, 0.0, 0.0]) or [0.0]) > 80.0:
            flags.append(f"window_{item['index']:06d}_world_span_too_large")
        if world.get("distance_to_trajectory", {}).get("median", 0.0) > 10.0:
            flags.append(f"window_{item['index']:06d}_world_far_from_trajectory")
        path_length = float(report.get("trajectory", {}).get("path_length_m", 0.0))
        local_z_median = float(local.get("z", {}).get("median", 0.0))
        if path_length > 0.1 and local_z_median / path_length > 10.0:
            flags.append(f"window_{item['index']:06d}_depth_pose_scale_mismatch")
    for item in report.get("consecutive_world_alignment", []):
        if item.get("nearest_distance", {}).get("median", 0.0) > 2.5:
            flags.append(f"window_{item['left']:06d}_{item['right']:06d}_weak_overlap")
    return flags


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    flags = report.get("quality_flags", [])
    lines = [
        "# Geometry Quality Report",
        "",
        f"Output dir: `{report['output_dir']}`",
        f"Windows: `{len(report.get('windows', []))}`",
        f"Trajectory poses: `{report.get('trajectory', {}).get('pose_count', 0)}`",
        f"Trajectory path length: `{report.get('trajectory', {}).get('path_length_m', 0.0):.3f} m`",
        f"Quality flags: `{', '.join(flags) if flags else 'none'}`",
        "",
        "## Rolling Map",
    ]
    rolling = report.get("rolling_map", {}).get("stats", {})
    lines += [
        f"- points: `{rolling.get('point_count', 0)}`",
        f"- bbox span: `{rolling.get('bbox_span', [])}`",
        f"- distance to trajectory median: `{rolling.get('distance_to_trajectory', {}).get('median', 'n/a')}`",
        "",
        "## Windows",
        "",
        "| window | local points | local z median | world points | world span | world-traj median |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for item in report.get("windows", []):
        local = item.get("local_stats", {})
        world = item.get("world_stats", {})
        lines.append(
            f"| {item['index']} | {local.get('point_count', 0)} | "
            f"{local.get('z', {}).get('median', float('nan')):.3f} | "
            f"{world.get('point_count', 0)} | {world.get('bbox_span', [])} | "
            f"{world.get('distance_to_trajectory', {}).get('median', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Consecutive World Alignment",
        "",
        "| pair | bbox overlap ratio | nearest median | nearest p90 |",
        "|---|---:|---:|---:|",
    ]
    for item in report.get("consecutive_world_alignment", []):
        nearest = item.get("nearest_distance", {})
        lines.append(
            f"| {item['left']} -> {item['right']} | {item.get('bbox_overlap_ratio', 0.0):.3f} | "
            f"{nearest.get('median', float('nan')):.3f} | {nearest.get('p90', float('nan')):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _viewer_points(xyz: np.ndarray, rgb: np.ndarray, max_points: int) -> list[list[float]]:
    if xyz.shape[0] == 0:
        return []
    idx = np.linspace(0, xyz.shape[0] - 1, min(max_points, xyz.shape[0])).astype(np.int64)
    xyz_s = xyz[idx]
    rgb_s = rgb[idx] if rgb.shape[0] == xyz.shape[0] else np.full((idx.size, 3), 180, dtype=np.uint8)
    return [
        [float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])]
        for p, c in zip(xyz_s, rgb_s)
    ]


def _write_viewer(path: Path, report: dict[str, Any], world_clouds: list[dict[str, Any]], trajectory: np.ndarray) -> None:
    colors = ["#ef476f", "#06d6a0", "#118ab2", "#ffd166", "#a78bfa", "#f97316", "#22c55e", "#e879f9"]
    payload = {
        "trajectory": trajectory.astype(float).tolist(),
        "clouds": [
            {
                "name": f"window {item['index']}",
                "color": colors[i % len(colors)],
                "points": _viewer_points(item["xyz"], item["rgb"], 2500),
            }
            for i, item in enumerate(world_clouds)
        ],
        "flags": report.get("quality_flags", []),
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Geometry Quality Viewer</title>
  <style>
    body {{ margin: 0; background: #090c10; color: #f5f5f5; font: 14px system-ui, sans-serif; }}
    #bar {{ position: fixed; top: 0; left: 0; right: 0; padding: 10px 14px; background: rgba(0,0,0,.65); z-index: 2; }}
    canvas {{ display: block; width: 100vw; height: 100vh; }}
    .flag {{ color: #ffcc66; margin-left: 12px; }}
  </style>
</head>
<body>
<div id="bar">Top-down XY diagnostic <span class="flag">flags: {((', '.join(payload['flags'])) or 'none')}</span></div>
<canvas id="c"></canvas>
<script>
const data = {json.dumps(payload)};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
function resize() {{ canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio; draw(); }}
function bounds() {{
  const pts = [];
  for (const p of data.trajectory) pts.push(p);
  for (const cloud of data.clouds) for (const p of cloud.points) pts.push(p);
  if (!pts.length) return [[-1,-1],[1,1]];
  let mn=[Infinity,Infinity], mx=[-Infinity,-Infinity];
  for (const p of pts) {{ mn[0]=Math.min(mn[0],p[0]); mn[1]=Math.min(mn[1],p[1]); mx[0]=Math.max(mx[0],p[0]); mx[1]=Math.max(mx[1],p[1]); }}
  return [mn,mx];
}}
function map(p, mn, mx) {{
  const pad = 70 * devicePixelRatio;
  const span = Math.max(mx[0]-mn[0], mx[1]-mn[1], 1e-3);
  const scale = Math.min((canvas.width-pad*2)/span, (canvas.height-pad*2)/span);
  return [pad + (p[0]-mn[0])*scale, canvas.height - pad - (p[1]-mn[1])*scale];
}}
function draw() {{
  ctx.fillStyle='#090c10'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const [mn,mx]=bounds();
  for (const cloud of data.clouds) {{
    ctx.fillStyle = cloud.color;
    for (const p of cloud.points) {{ const q=map(p,mn,mx); ctx.fillRect(q[0],q[1],2*devicePixelRatio,2*devicePixelRatio); }}
  }}
  if (data.trajectory.length > 1) {{
    ctx.strokeStyle='#ffffff'; ctx.lineWidth=2*devicePixelRatio; ctx.beginPath();
    data.trajectory.forEach((p,i)=>{{ const q=map(p,mn,mx); if(i===0) ctx.moveTo(q[0],q[1]); else ctx.lineTo(q[0],q[1]); }});
    ctx.stroke();
  }}
}}
addEventListener('resize', resize); resize();
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate rolling LingBot/cuvSLAM geometry quality artifacts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CVPR/nuc_output/hikrobot_lingbot_ros2_current_cloud_live"),
    )
    parser.add_argument("--max-windows", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    dense_dir = output_dir / "dense_windows"
    trajectory = _read_trajectory(output_dir / "trajectory.txt")
    report: dict[str, Any] = {
        "output_dir": str(output_dir),
        "trajectory": {
            "path": str(output_dir / "trajectory.txt"),
            "pose_count": int(trajectory.shape[0]),
            "path_length_m": _path_length(trajectory),
            "stats": _cloud_stats(trajectory, np.zeros((trajectory.shape[0], 3), dtype=np.uint8)),
        },
        "windows": [],
        "consecutive_world_alignment": [],
    }

    rolling = _read_ascii_ply(output_dir / "rolling_map.ply")
    report["rolling_map"] = {
        "path": str(output_dir / "rolling_map.ply"),
        "stats": _cloud_stats(rolling["xyz"], rolling["rgb"], trajectory),
    }

    local_paths = { _window_index(path): path for path in dense_dir.glob("window_*_local.ply") }
    world_paths = { _window_index(path): path for path in dense_dir.glob("window_*_world.ply") }
    indices = sorted(set(local_paths) | set(world_paths))
    if args.max_windows > 0:
        indices = indices[: args.max_windows]

    world_clouds: list[dict[str, Any]] = []
    previous_world: dict[str, Any] | None = None
    previous_world_stats: dict[str, Any] | None = None
    for idx in indices:
        local = _read_ascii_ply(local_paths[idx]) if idx in local_paths else {"xyz": np.zeros((0, 3)), "rgb": np.zeros((0, 3), dtype=np.uint8)}
        world = _read_ascii_ply(world_paths[idx]) if idx in world_paths else {"xyz": np.zeros((0, 3)), "rgb": np.zeros((0, 3), dtype=np.uint8)}
        local_stats = _cloud_stats(local["xyz"], local["rgb"])
        world_stats = _cloud_stats(world["xyz"], world["rgb"], trajectory)
        local_radius_p95 = float(local_stats.get("radius", {}).get("p95", 0.0))
        world_radius_p95 = float(world_stats.get("radius", {}).get("p95", 0.0))
        item = {
            "index": int(idx),
            "local_path": str(local_paths.get(idx, "")),
            "world_path": str(world_paths.get(idx, "")),
            "local_stats": local_stats,
            "world_stats": world_stats,
            "local_world_radius_p95_abs_diff": abs(local_radius_p95 - world_radius_p95),
        }
        report["windows"].append(item)
        world_entry = {"index": int(idx), "xyz": world["xyz"], "rgb": world["rgb"]}
        world_clouds.append(world_entry)
        if previous_world is not None and previous_world_stats is not None:
            pair = _pair_metrics(previous_world["xyz"], world["xyz"])
            pair.update(
                {
                    "left": int(previous_world["index"]),
                    "right": int(idx),
                    "bbox_overlap_ratio": _bbox_overlap(previous_world_stats, world_stats),
                }
            )
            report["consecutive_world_alignment"].append(pair)
        previous_world = world_entry
        previous_world_stats = world_stats

    report["quality_flags"] = _quality_flags(report)

    json_path = output_dir / "geometry_quality_report.json"
    md_path = output_dir / "geometry_quality_report.md"
    viewer_path = output_dir / "geometry_quality_viewer.html"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report)
    _write_viewer(viewer_path, report, world_clouds, trajectory)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "viewer": str(viewer_path), "flags": report["quality_flags"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
