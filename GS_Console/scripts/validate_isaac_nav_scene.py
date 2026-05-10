#!/usr/bin/env python3
"""Validate a minimal Isaac agent against an existing collision USD scene.

This script is intentionally narrow:

1. Load an existing collision USD scene.
2. Pick a start pose from the existing 2D occupancy map.
3. Estimate the local floor height from the source mesh.
4. Spawn a simple capsule agent above that pose.
5. Step physics and report whether the agent settles, makes contact, and avoids obvious instability.
6. Sample a fixed list of 2D navigation goals from the same occupancy map for the next stage.

The current Gaussian PLY is not used for physics. It remains a visual-layer asset outside this script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collision-usd", required=True, help="Collision USD scene used for physics.")
    parser.add_argument("--mesh-ply", required=True, help="Source mesh PLY used to estimate local floor height.")
    parser.add_argument("--map-yaml", required=True, help="ROS-style occupancy-map YAML file.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--start-cell", default=None, help="Optional map cell override as 'x,y'.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for goal sampling.")
    parser.add_argument("--num-goals", type=int, default=20, help="Number of 2D navigation goals to sample.")
    parser.add_argument(
        "--goal-min-distance",
        type=float,
        default=2.0,
        help="Minimum XY distance in meters between accepted goals.",
    )
    parser.add_argument(
        "--start-clearance",
        type=float,
        default=0.4,
        help="Minimum distance in meters from the start to sampled goals.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 120.0, help="Physics timestep in seconds.")
    parser.add_argument("--sim-steps", type=int, default=720, help="Number of physics steps to run.")
    parser.add_argument(
        "--drop-height",
        type=float,
        default=1.2,
        help="Height above the estimated floor where the agent is spawned.",
    )
    parser.add_argument("--agent-radius", type=float, default=0.25, help="Capsule radius in meters.")
    parser.add_argument("--agent-height", type=float, default=0.50, help="Capsule cylinder height in meters.")
    parser.add_argument(
        "--floor-search-percentile",
        type=float,
        default=10.0,
        help="Percentile of local mesh z values used as the floor estimate.",
    )
    parser.add_argument(
        "--floor-search-radii",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated XY radii in meters used to search for nearby mesh vertices.",
    )
    parser.add_argument(
        "--min-floor-points",
        type=int,
        default=64,
        help="Minimum number of nearby mesh vertices required for a floor estimate.",
    )
    parser.add_argument(
        "--support-cell-min-points",
        type=int,
        default=8,
        help="Minimum number of mesh vertices in a free map cell for start-cell support selection.",
    )
    parser.add_argument(
        "--support-cell-max-spread",
        type=float,
        default=0.25,
        help="Maximum z spread in a free map cell for start-cell support selection.",
    )
    parser.add_argument(
        "--goal-support-min-points",
        type=int,
        default=1,
        help="Minimum number of mesh vertices in a free map cell for goal support sampling.",
    )
    parser.add_argument(
        "--goal-support-max-spread",
        type=float,
        default=1.5,
        help="Maximum z spread in a free map cell for goal support sampling.",
    )
    parser.add_argument(
        "--min-start-component-size",
        type=int,
        default=12,
        help="Minimum loose-support connected-component size to consider during automatic start selection.",
    )
    parser.add_argument(
        "--max-start-candidates",
        type=int,
        default=8,
        help="Maximum number of ranked start cells to evaluate with physics before picking one.",
    )
    parser.add_argument(
        "--max-start-cells-per-component",
        type=int,
        default=2,
        help="Maximum number of strict-supported start cells to try from each loose-support component.",
    )
    parser.add_argument(
        "--settle-speed-threshold",
        type=float,
        default=0.25,
        help="Maximum linear speed at the end of rollout to count as settled.",
    )
    parser.add_argument(
        "--explosion-speed-threshold",
        type=float,
        default=25.0,
        help="Linear-speed threshold above which the rollout is marked unstable.",
    )
    parser.add_argument(
        "--penetration-tolerance",
        type=float,
        default=0.25,
        help="Allowed depth below the estimated floor before flagging penetration.",
    )
    parser.add_argument(
        "--step-sleep",
        type=float,
        default=0.0,
        help="Optional wall-clock sleep in seconds after each simulation step for viewer mode.",
    )
    parser.add_argument(
        "--hold-open-sec",
        type=float,
        default=0.0,
        help="Keep the viewer open for this many seconds after validation completes.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


ARGS_CLI = build_argparser().parse_args()
APP_LAUNCHER = AppLauncher(ARGS_CLI)
simulation_app = APP_LAUNCHER.app


import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils import configclass


@dataclass
class OccupancyMap:
    image_path: Path
    yaml_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    image: np.ndarray

    @property
    def free_mask(self) -> np.ndarray:
        return self.image == 254


@dataclass
class SupportGrid:
    floor_z: np.ndarray
    point_count: np.ndarray
    z_spread: np.ndarray

    def is_supported(self, cell: tuple[int, int]) -> bool:
        return bool(np.isfinite(self.floor_z[cell[1], cell[0]]))


@configclass
class ValidationSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Environment")
    agent = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        spawn=sim_utils.CapsuleCfg(
            radius=0.25,
            height=0.50,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.4,
                max_linear_velocity=50.0,
                max_angular_velocity=3600.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Agent",
        track_pose=True,
        track_air_time=True,
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )


def parse_float_list(raw: str) -> list[float]:
    values = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_int_pair(raw: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in raw.split(",")]
    if len(pieces) != 2:
        raise ValueError(f"Expected 'x,y' pair, got: {raw!r}")
    return int(pieces[0]), int(pieces[1])


def read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic {magic!r}; expected P5.")

        def next_token() -> bytes:
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"Unexpected EOF while reading PGM header: {path}")
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                return line

        dims = next_token().split()
        if len(dims) != 2:
            raise ValueError(f"Invalid PGM size header in {path}")
        width, height = int(dims[0]), int(dims[1])
        max_value = int(next_token())
        if max_value > 255:
            raise ValueError(f"Unsupported PGM max value {max_value}; expected <= 255.")
        data = np.frombuffer(handle.read(width * height), dtype=np.uint8)
        if data.size != width * height:
            raise ValueError(f"PGM payload size mismatch in {path}")
        return data.reshape((height, width))


def load_occupancy_map(yaml_path: Path) -> OccupancyMap:
    text = yaml_path.read_text(encoding="utf-8")
    image_match = re.search(r"^image:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    resolution_match = re.search(r"^resolution:\s*([-+0-9.eE]+)\s*$", text, flags=re.MULTILINE)
    origin_match = re.search(r"^origin:\s*\[([^\]]+)\]\s*$", text, flags=re.MULTILINE)
    if image_match is None or resolution_match is None or origin_match is None:
        raise ValueError(f"Could not parse occupancy-map metadata from {yaml_path}")

    image_path = (yaml_path.parent / image_match.group(1)).resolve()
    origin_values = [float(piece.strip()) for piece in origin_match.group(1).split(",")]
    if len(origin_values) < 2:
        raise ValueError(f"Invalid origin in {yaml_path}")

    return OccupancyMap(
        image_path=image_path,
        yaml_path=yaml_path,
        resolution=float(resolution_match.group(1)),
        origin_x=origin_values[0],
        origin_y=origin_values[1],
        image=read_pgm(image_path),
    )


def cell_to_world(map_info: OccupancyMap, x: int, y: int) -> tuple[float, float]:
    world_x = map_info.origin_x + (x + 0.5) * map_info.resolution
    world_y = map_info.origin_y + ((map_info.image.shape[0] - 1 - y) + 0.5) * map_info.resolution
    return world_x, world_y


def in_bounds(map_info: OccupancyMap, cell: tuple[int, int]) -> bool:
    return 0 <= cell[0] < map_info.image.shape[1] and 0 <= cell[1] < map_info.image.shape[0]


def is_free_cell(map_info: OccupancyMap, cell: tuple[int, int]) -> bool:
    return in_bounds(map_info, cell) and bool(map_info.free_mask[cell[1], cell[0]])


def pick_center_free_cell(map_info: OccupancyMap) -> tuple[int, int]:
    ys, xs = np.nonzero(map_info.free_mask)
    if len(xs) == 0:
        raise ValueError(f"No free cells found in {map_info.image_path}")
    cy = map_info.image.shape[0] // 2
    cx = map_info.image.shape[1] // 2
    index = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    return int(xs[index]), int(ys[index])


def pick_supported_start_cell(
    map_info: OccupancyMap,
    vertices: np.ndarray,
    min_points: int,
    max_spread: float,
) -> dict | None:
    width = map_info.image.shape[1]
    height = map_info.image.shape[0]

    cells_x = np.floor((vertices[:, 0] - map_info.origin_x) / map_info.resolution).astype(np.int32)
    world_rows = np.floor((vertices[:, 1] - map_info.origin_y) / map_info.resolution).astype(np.int32)
    cells_y = (height - 1) - world_rows

    valid = (cells_x >= 0) & (cells_x < width) & (cells_y >= 0) & (cells_y < height)
    if not np.any(valid):
        return None

    cells_x = cells_x[valid]
    cells_y = cells_y[valid]
    z_values = vertices[valid, 2]

    free_mask = map_info.free_mask[cells_y, cells_x]
    if not np.any(free_mask):
        return None

    cells_x = cells_x[free_mask]
    cells_y = cells_y[free_mask]
    z_values = z_values[free_mask]

    linear = cells_y.astype(np.int64) * width + cells_x.astype(np.int64)
    order = np.argsort(linear)
    linear = linear[order]
    z_values = z_values[order]

    unique_linear, start_indices, counts = np.unique(linear, return_index=True, return_counts=True)
    cy = height // 2
    cx = width // 2

    best_candidate = None
    best_score = None
    for index, count in enumerate(counts):
        if int(count) < min_points:
            continue
        start = start_indices[index]
        end = start + count
        cell_z = z_values[start:end]
        z_min = float(np.min(cell_z))
        z_max = float(np.max(cell_z))
        z_spread = z_max - z_min
        if z_spread > max_spread:
            continue

        row = int(unique_linear[index] // width)
        col = int(unique_linear[index] % width)
        distance2 = float((row - cy) ** 2 + (col - cx) ** 2)
        score = distance2 - 4.0 * float(count)

        if best_score is None or score < best_score:
            world_x, world_y = cell_to_world(map_info, col, row)
            best_score = score
            best_candidate = {
                "cell": [col, row],
                "xy": [float(world_x), float(world_y)],
                "point_count": int(count),
                "z_min": z_min,
                "z_max": z_max,
                "z_spread": z_spread,
                "score": score,
            }

    return best_candidate


def build_support_grid(
    map_info: OccupancyMap,
    vertices: np.ndarray,
    min_points: int,
    max_spread: float,
    floor_percentile: float,
) -> SupportGrid:
    width = map_info.image.shape[1]
    height = map_info.image.shape[0]
    floor_z = np.full((height, width), np.nan, dtype=np.float32)
    point_count = np.zeros((height, width), dtype=np.int32)
    z_spread = np.full((height, width), np.inf, dtype=np.float32)

    cells_x = np.floor((vertices[:, 0] - map_info.origin_x) / map_info.resolution).astype(np.int32)
    world_rows = np.floor((vertices[:, 1] - map_info.origin_y) / map_info.resolution).astype(np.int32)
    cells_y = (height - 1) - world_rows

    valid = (cells_x >= 0) & (cells_x < width) & (cells_y >= 0) & (cells_y < height)
    if not np.any(valid):
        return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)

    cells_x = cells_x[valid]
    cells_y = cells_y[valid]
    z_values = vertices[valid, 2]

    free_mask = map_info.free_mask[cells_y, cells_x]
    cells_x = cells_x[free_mask]
    cells_y = cells_y[free_mask]
    z_values = z_values[free_mask]
    if z_values.size == 0:
        return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)

    linear = cells_y.astype(np.int64) * width + cells_x.astype(np.int64)
    order = np.argsort(linear)
    linear = linear[order]
    z_values = z_values[order]

    unique_linear, start_indices, counts = np.unique(linear, return_index=True, return_counts=True)
    for index, count in enumerate(counts):
        row = int(unique_linear[index] // width)
        col = int(unique_linear[index] % width)
        point_count[row, col] = int(count)
        start = start_indices[index]
        end = start + count
        cell_z = z_values[start:end]
        spread = float(np.max(cell_z) - np.min(cell_z))
        z_spread[row, col] = spread
        if int(count) >= min_points and spread <= max_spread:
            floor_z[row, col] = float(np.percentile(cell_z, floor_percentile))

    return SupportGrid(floor_z=floor_z, point_count=point_count, z_spread=z_spread)


def collect_supported_component_cells(
    map_info: OccupancyMap,
    support_grid: SupportGrid,
    start_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    if not is_free_cell(map_info, start_cell) or not support_grid.is_supported(start_cell):
        return []

    queue = deque([start_cell])
    visited = {start_cell}
    cells: list[tuple[int, int]] = []
    neighbor_offsets = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while queue:
        current = queue.popleft()
        cells.append(current)
        for dx, dy in neighbor_offsets:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited:
                continue
            if not is_free_cell(map_info, neighbor):
                continue
            if not support_grid.is_supported(neighbor):
                continue
            visited.add(neighbor)
            queue.append(neighbor)

    return cells


def collect_supported_components(
    map_info: OccupancyMap,
    support_grid: SupportGrid,
) -> list[list[tuple[int, int]]]:
    width = map_info.image.shape[1]
    height = map_info.image.shape[0]
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            cell = (col, row)
            if visited[row, col]:
                continue
            visited[row, col] = True
            if not is_free_cell(map_info, cell) or not support_grid.is_supported(cell):
                continue
            components.append(collect_supported_component_cells(map_info, support_grid, cell))
            for component_col, component_row in components[-1]:
                visited[component_row, component_col] = True

    return components


def pick_start_cell_from_components(
    map_info: OccupancyMap,
    strict_support_grid: SupportGrid,
    components: list[list[tuple[int, int]]],
    min_component_size: int,
) -> tuple[tuple[int, int], dict] | None:
    cy = map_info.image.shape[0] // 2
    cx = map_info.image.shape[1] // 2
    best = None
    best_score = None

    for component in components:
        component_size = len(component)
        if component_size < min_component_size:
            continue

        strict_cells = [cell for cell in component if strict_support_grid.is_supported(cell)]
        if not strict_cells:
            continue

        for col, row in strict_cells:
            distance2 = float((row - cy) ** 2 + (col - cx) ** 2)
            score = (distance2, -component_size)
            if best_score is None or score < best_score:
                world_x, world_y = cell_to_world(map_info, col, row)
                best_score = score
                best = (
                    (col, row),
                    {
                        "cell": [int(col), int(row)],
                        "xy": [float(world_x), float(world_y)],
                        "component_size": int(component_size),
                        "distance2_to_center": distance2,
                    },
                )

    return best


def build_start_candidate_info(
    map_info: OccupancyMap,
    strict_support_grid: SupportGrid,
    support_grid: SupportGrid,
    start_cell: tuple[int, int],
    *,
    source: str,
    component_rank: int | None = None,
) -> dict:
    world_x, world_y = cell_to_world(map_info, *start_cell)
    component_cells = collect_supported_component_cells(map_info, support_grid, start_cell)
    point_count = 0
    z_spread = math.inf
    if in_bounds(map_info, start_cell):
        point_count = int(strict_support_grid.point_count[start_cell[1], start_cell[0]])
        z_spread = float(strict_support_grid.z_spread[start_cell[1], start_cell[0]])
    cy = map_info.image.shape[0] // 2
    cx = map_info.image.shape[1] // 2
    return {
        "source": source,
        "cell": [int(start_cell[0]), int(start_cell[1])],
        "xy": [float(world_x), float(world_y)],
        "component_size": int(len(component_cells)),
        "strict_supported": bool(strict_support_grid.is_supported(start_cell)),
        "strict_point_count": point_count,
        "strict_z_spread": z_spread,
        "distance2_to_center": float((start_cell[1] - cy) ** 2 + (start_cell[0] - cx) ** 2),
        "component_rank": None if component_rank is None else int(component_rank),
    }


def rank_start_candidates(
    map_info: OccupancyMap,
    strict_support_grid: SupportGrid,
    support_grid: SupportGrid,
    min_component_size: int,
    max_candidates: int,
    max_cells_per_component: int,
) -> list[dict]:
    if max_candidates <= 0 or max_cells_per_component <= 0:
        return []

    cy = map_info.image.shape[0] // 2
    cx = map_info.image.shape[1] // 2
    components = collect_supported_components(map_info, support_grid)
    ranked_components = sorted(
        components,
        key=lambda component: (
            -len(component),
            min(float((row - cy) ** 2 + (col - cx) ** 2) for col, row in component),
        ),
    )

    candidates: list[dict] = []
    for component_rank, component in enumerate(ranked_components):
        component_size = len(component)
        if component_size < min_component_size:
            continue

        strict_cells = [cell for cell in component if strict_support_grid.is_supported(cell)]
        if not strict_cells:
            continue

        strict_cells.sort(
            key=lambda cell: (
                float((cell[1] - cy) ** 2 + (cell[0] - cx) ** 2),
                -int(strict_support_grid.point_count[cell[1], cell[0]]),
                float(strict_support_grid.z_spread[cell[1], cell[0]]),
                cell[1],
                cell[0],
            )
        )

        for start_cell in strict_cells[:max_cells_per_component]:
            candidate = build_start_candidate_info(
                map_info=map_info,
                strict_support_grid=strict_support_grid,
                support_grid=support_grid,
                start_cell=start_cell,
                source="ranked_component",
                component_rank=component_rank,
            )
            candidate["component_size"] = int(component_size)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def sample_goals(
    map_info: OccupancyMap,
    start_xy: tuple[float, float],
    count: int,
    seed: int,
    min_distance: float,
    start_clearance: float,
    candidate_cells: list[tuple[int, int]] | None = None,
) -> list[list[float]]:
    if candidate_cells is None:
        ys, xs = np.nonzero(map_info.free_mask)
        if len(xs) == 0:
            raise ValueError(f"No free cells found in {map_info.image_path}")
        candidate_count = len(xs)
    else:
        if not candidate_cells:
            raise ValueError("No candidate goal cells available.")
        xs = np.asarray([cell[0] for cell in candidate_cells], dtype=np.int32)
        ys = np.asarray([cell[1] for cell in candidate_cells], dtype=np.int32)
        candidate_count = len(candidate_cells)

    rng = np.random.default_rng(seed)
    order = rng.permutation(candidate_count)
    accepted: list[list[float]] = []
    start_xy_np = np.asarray(start_xy, dtype=np.float32)

    for index in order:
        world_xy = np.asarray(cell_to_world(map_info, int(xs[index]), int(ys[index])), dtype=np.float32)
        if np.linalg.norm(world_xy - start_xy_np) < start_clearance:
            continue
        if any(np.linalg.norm(world_xy - np.asarray(goal[:2], dtype=np.float32)) < min_distance for goal in accepted):
            continue
        accepted.append([float(world_xy[0]), float(world_xy[1]), 0.0])
        if len(accepted) >= count:
            break

    return accepted


def estimate_floor_height(
    vertices: np.ndarray,
    xy: tuple[float, float],
    radii: list[float],
    percentile: float,
    min_points: int,
) -> dict:
    xy_np = np.asarray(xy, dtype=np.float32)
    best_points = None
    best_radius = None

    for radius in radii:
        delta = vertices[:, :2] - xy_np
        mask = np.sum(delta * delta, axis=1) <= radius * radius
        count = int(mask.sum())
        if count < 1:
            continue
        points = vertices[mask]
        best_points = points
        best_radius = radius
        if count >= min_points:
            break

    if best_points is None or best_radius is None:
        raise ValueError(f"Could not find any mesh points near spawn xy={xy}")

    z_values = best_points[:, 2]
    floor_z = float(np.percentile(z_values, percentile))
    return {
        "floor_z": floor_z,
        "search_radius": float(best_radius),
        "point_count": int(best_points.shape[0]),
        "z_min": float(np.min(z_values)),
        "z_max": float(np.max(z_values)),
        "z_median": float(np.median(z_values)),
    }


def tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def viewer_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "headless", False))


def perform_sim_step(sim, scene, dt: float, *, render: bool, step_sleep: float = 0.0) -> None:
    scene.write_data_to_sim()
    sim.step(render=render)
    scene.update(dt=dt)
    if step_sleep > 0.0:
        time.sleep(step_sleep)


def set_default_camera_view(sim, target_xy: tuple[float, float], floor_z: float) -> None:
    eye = [float(target_xy[0] - 4.0), float(target_xy[1] - 4.0), float(floor_z + 3.0)]
    target = [float(target_xy[0]), float(target_xy[1]), float(floor_z + 0.5)]
    sim.set_camera_view(eye=eye, target=target)


def run_candidate_physics_validation(
    args: argparse.Namespace,
    collision_usd: Path,
    start_xy: tuple[float, float],
    floor_info: dict,
    *,
    render: bool,
) -> dict:
    spawn_pose = (
        float(start_xy[0]),
        float(start_xy[1]),
        float(floor_info["floor_z"] + args.agent_radius + 0.5 * args.agent_height + args.drop_height),
    )

    scene_cfg = ValidationSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=False, lazy_sensor_update=False)
    scene_cfg.environment = scene_cfg.environment.replace(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(usd_path=str(collision_usd)),
    )
    scene_cfg.agent = scene_cfg.agent.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pose),
        spawn=scene_cfg.agent.spawn.replace(radius=args.agent_radius, height=args.agent_height),
    )
    scene_cfg.contact_sensor = scene_cfg.contact_sensor.replace(prim_path=scene_cfg.agent.prim_path)

    sim_cfg = sim_utils.SimulationCfg(dt=args.dt, device=args.device)
    sim_cfg.physx.enable_ccd = True
    sim_cfg.physx.enable_enhanced_determinism = True

    max_linear_speed = 0.0
    max_angular_speed = 0.0
    min_z = math.inf
    max_z = -math.inf
    max_contact_force = 0.0
    unstable_detected = False
    first_contact_step = None
    final_snapshot = {}

    with build_simulation_context(create_new_stage=True, sim_cfg=sim_cfg, add_ground_plane=False, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()
        if render:
            set_default_camera_view(sim, start_xy, floor_info["floor_z"])

        agent = scene["agent"]
        sensor = scene["contact_sensor"]

        for step in range(args.sim_steps):
            perform_sim_step(sim, scene, args.dt, render=render, step_sleep=args.step_sleep)

            root_pos = agent.data.root_pos_w[0]
            root_lin_vel = agent.data.root_lin_vel_w[0]
            root_ang_vel = agent.data.root_ang_vel_w[0]

            z_value = float(root_pos[2].item())
            linear_speed = float(torch.linalg.norm(root_lin_vel).item())
            angular_speed = float(torch.linalg.norm(root_ang_vel).item())

            if torch.any(torch.isnan(agent.data.root_state_w)):
                unstable_detected = True
            if linear_speed > args.explosion_speed_threshold:
                unstable_detected = True

            min_z = min(min_z, z_value)
            max_z = max(max_z, z_value)
            max_linear_speed = max(max_linear_speed, linear_speed)
            max_angular_speed = max(max_angular_speed, angular_speed)

            contact_force = float(torch.linalg.norm(sensor.data.net_forces_w.reshape(-1, 3), dim=-1).max().item())
            max_contact_force = max(max_contact_force, contact_force)
            if first_contact_step is None and (
                contact_force > 1e-3
                or float(torch.max(sensor.data.current_contact_time).item()) > 0.0
                or float(torch.max(sensor.data.last_contact_time).item()) > 0.0
            ):
                first_contact_step = step

            final_snapshot = {
                "step": step,
                "root_pos_w": tensor_to_list(root_pos),
                "root_quat_w": tensor_to_list(agent.data.root_quat_w[0]),
                "root_lin_vel_w": tensor_to_list(root_lin_vel),
                "root_ang_vel_w": tensor_to_list(root_ang_vel),
                "contact_force_norm_max": contact_force,
                "current_contact_time": float(torch.max(sensor.data.current_contact_time).item()),
                "last_contact_time": float(torch.max(sensor.data.last_contact_time).item()),
                "current_air_time": float(torch.max(sensor.data.current_air_time).item()),
                "last_air_time": float(torch.max(sensor.data.last_air_time).item()),
            }

        if render and args.hold_open_sec > 0.0:
            hold_steps = max(1, int(math.ceil(args.hold_open_sec / max(args.dt, 1e-3))))
            hold_sleep = args.step_sleep if args.step_sleep > 0.0 else min(args.dt, 1.0 / 30.0)
            for _ in range(hold_steps):
                perform_sim_step(sim, scene, args.dt, render=True, step_sleep=hold_sleep)

    penetration_suspected = bool(min_z < floor_info["floor_z"] - args.penetration_tolerance)
    contact_observed = bool(
        first_contact_step is not None
        or max_contact_force > 1e-3
        or final_snapshot.get("current_contact_time", 0.0) > 0.0
        or final_snapshot.get("last_contact_time", 0.0) > 0.0
    )
    settled = bool(final_snapshot and np.linalg.norm(final_snapshot["root_lin_vel_w"]) <= args.settle_speed_threshold)
    passed = bool(contact_observed and settled and not penetration_suspected and not unstable_detected)

    return {
        "spawn_pose_w": list(spawn_pose),
        "physics_validation": {
            "passed": passed,
            "contact_observed": contact_observed,
            "settled": settled,
            "penetration_suspected": penetration_suspected,
            "unstable_detected": unstable_detected,
            "first_contact_step": first_contact_step,
            "max_linear_speed": max_linear_speed,
            "max_angular_speed": max_angular_speed,
            "max_contact_force": max_contact_force,
            "min_z": min_z,
            "max_z": max_z,
            "final": final_snapshot,
        },
    }


def run_validation(args: argparse.Namespace) -> dict:
    collision_usd = Path(args.collision_usd).expanduser().resolve()
    mesh_ply = Path(args.mesh_ply).expanduser().resolve()
    map_yaml = Path(args.map_yaml).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not collision_usd.is_file():
        raise FileNotFoundError(f"Collision USD not found: {collision_usd}")
    if not mesh_ply.is_file():
        raise FileNotFoundError(f"Mesh PLY not found: {mesh_ply}")
    if not map_yaml.is_file():
        raise FileNotFoundError(f"Map YAML not found: {map_yaml}")

    mesh = trimesh.load(mesh_ply, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    map_info = load_occupancy_map(map_yaml)
    strict_support_grid = build_support_grid(
        map_info=map_info,
        vertices=vertices,
        min_points=args.support_cell_min_points,
        max_spread=args.support_cell_max_spread,
        floor_percentile=args.floor_search_percentile,
    )
    support_grid = build_support_grid(
        map_info=map_info,
        vertices=vertices,
        min_points=args.goal_support_min_points,
        max_spread=args.goal_support_max_spread,
        floor_percentile=args.floor_search_percentile,
    )
    support_candidate = pick_supported_start_cell(
        map_info=map_info,
        vertices=vertices,
        min_points=args.support_cell_min_points,
        max_spread=args.support_cell_max_spread,
    )
    selection_mode = "auto_ranked_component"
    start_candidates: list[dict] = []
    if args.start_cell is not None:
        start_cell_override = parse_int_pair(args.start_cell)
        if not is_free_cell(map_info, start_cell_override):
            raise ValueError(f"Requested start cell is not free: {start_cell_override}")
        start_candidates = [
            build_start_candidate_info(
                map_info=map_info,
                strict_support_grid=strict_support_grid,
                support_grid=support_grid,
                start_cell=start_cell_override,
                source="explicit_start_cell",
            )
        ]
        selection_mode = "explicit_start_cell"
    else:
        start_candidates = rank_start_candidates(
            map_info=map_info,
            strict_support_grid=strict_support_grid,
            support_grid=support_grid,
            min_component_size=args.min_start_component_size,
            max_candidates=args.max_start_candidates,
            max_cells_per_component=args.max_start_cells_per_component,
        )
        if support_candidate is not None:
            legacy_candidate = build_start_candidate_info(
                map_info=map_info,
                strict_support_grid=strict_support_grid,
                support_grid=support_grid,
                start_cell=tuple(int(v) for v in support_candidate["cell"]),
                source="legacy_support_candidate",
            )
            if not any(candidate["cell"] == legacy_candidate["cell"] for candidate in start_candidates):
                start_candidates.append(legacy_candidate)
        if not start_candidates:
            selection_mode = "legacy_support_candidate"
            start_candidates = [
                build_start_candidate_info(
                    map_info=map_info,
                    strict_support_grid=strict_support_grid,
                    support_grid=support_grid,
                    start_cell=tuple(int(v) for v in support_candidate["cell"]),
                    source="legacy_support_candidate",
                )
            ] if support_candidate is not None else []
        if not start_candidates:
            fallback_cell = pick_center_free_cell(map_info)
            start_candidates = [
                build_start_candidate_info(
                    map_info=map_info,
                    strict_support_grid=strict_support_grid,
                    support_grid=support_grid,
                    start_cell=fallback_cell,
                    source="center_free_fallback",
                )
            ]
            selection_mode = "center_free_fallback"

    floor_search_radii = parse_float_list(args.floor_search_radii)
    selection_attempts: list[dict] = []
    selected_candidate = None
    selected_floor_info = None
    selected_rollout = None
    for attempt_index, candidate in enumerate(start_candidates):
        candidate_xy = tuple(float(v) for v in candidate["xy"])
        floor_info = estimate_floor_height(
            vertices=vertices,
            xy=candidate_xy,
            radii=floor_search_radii,
            percentile=args.floor_search_percentile,
            min_points=args.min_floor_points,
        )
        rollout = run_candidate_physics_validation(
            args=args,
            collision_usd=collision_usd,
            start_xy=candidate_xy,
            floor_info=floor_info,
            render=False,
        )
        attempt_record = dict(candidate)
        attempt_record["attempt_index"] = int(attempt_index)
        attempt_record["floor_estimate"] = floor_info
        attempt_record["spawn_pose_w"] = rollout["spawn_pose_w"]
        attempt_record["physics_validation"] = rollout["physics_validation"]
        selection_attempts.append(attempt_record)
        if rollout["physics_validation"]["passed"]:
            selected_candidate = candidate
            selected_floor_info = floor_info
            selected_rollout = rollout
            break

    if selected_candidate is None:
        selected_candidate = start_candidates[0]
        selected_floor_info = selection_attempts[0]["floor_estimate"]
        selected_rollout = {
            "spawn_pose_w": selection_attempts[0]["spawn_pose_w"],
            "physics_validation": selection_attempts[0]["physics_validation"],
        }

    start_cell = tuple(int(v) for v in selected_candidate["cell"])
    start_xy = tuple(float(v) for v in selected_candidate["xy"])
    start_component_info = dict(selected_candidate)
    if viewer_enabled(args):
        selected_rollout = run_candidate_physics_validation(
            args=args,
            collision_usd=collision_usd,
            start_xy=start_xy,
            floor_info=selected_floor_info,
            render=True,
        )

    goal_candidate_cells = collect_supported_component_cells(map_info, support_grid, start_cell)
    goals = sample_goals(
        map_info=map_info,
        start_xy=start_xy,
        count=args.num_goals,
        seed=args.seed,
        min_distance=args.goal_min_distance,
        start_clearance=args.start_clearance,
        candidate_cells=goal_candidate_cells if goal_candidate_cells else None,
    )

    report = {
        "inputs": {
            "collision_usd": str(collision_usd),
            "mesh_ply": str(mesh_ply),
            "map_yaml": str(map_yaml),
            "device": args.device,
            "seed": args.seed,
            "dt": args.dt,
            "sim_steps": args.sim_steps,
            "drop_height": args.drop_height,
            "agent_radius": args.agent_radius,
            "agent_height": args.agent_height,
        },
        "map": {
            "resolution": map_info.resolution,
            "origin": [map_info.origin_x, map_info.origin_y, 0.0],
            "size": [int(map_info.image.shape[1]), int(map_info.image.shape[0])],
            "free_cell_count": int(map_info.free_mask.sum()),
            "start_cell": [int(start_cell[0]), int(start_cell[1])],
            "start_xy": [float(start_xy[0]), float(start_xy[1])],
            "start_support_candidate": support_candidate,
            "start_component_pick": start_component_info,
            "start_selection": {
                "mode": selection_mode,
                "candidate_count": int(len(start_candidates)),
                "attempted_count": int(len(selection_attempts)),
                "selected_attempt_index": next(
                    (
                        int(attempt["attempt_index"])
                        for attempt in selection_attempts
                        if attempt["cell"] == start_component_info["cell"]
                    ),
                    None,
                ),
                "selected_candidate": start_component_info,
                "attempts": selection_attempts,
            },
            "goal_sampling": {
                "mode": "supported_connected_component" if goal_candidate_cells else "free_space",
                "candidate_cell_count": int(len(goal_candidate_cells)) if goal_candidate_cells else int(map_info.free_mask.sum()),
                "support_cell_count": int(np.isfinite(support_grid.floor_z).sum()),
            },
            "goals": goals,
        },
        "floor_estimate": selected_floor_info,
        "spawn_pose_w": selected_rollout["spawn_pose_w"],
        "physics_validation": selected_rollout["physics_validation"],
        "notes": {
            "gaussian_visual_layer_loaded_in_isaac": False,
            "gaussian_comment": (
                "The existing gs.ply is not used here. Physics validation is running on the collision USD "
                "generated from mesh_gs_.ply. The Gaussian asset remains a visual-only layer until it is "
                "converted into an Isaac-loadable visual asset."
            ),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    try:
        run_validation(ARGS_CLI)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
