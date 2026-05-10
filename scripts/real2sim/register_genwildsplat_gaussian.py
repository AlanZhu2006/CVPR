#!/usr/bin/env python3
"""Register a GenWildSplat Gaussian output under real2sim/latest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _repo_root() / "CVPR/nuc_output/video_real2sim_playback/genwildsplat/lingbot_vid30fps_518_full_6ctx/inference"


def _default_real2sim_dir() -> Path:
    return _repo_root() / "CVPR/nuc_output/video_real2sim_playback/real2sim"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _count_ply_vertices(path: Path) -> int | None:
    with path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genwildsplat-output-dir", type=Path, default=_default_output_dir(), help="Directory containing GenWildSplat gaussians.ply.")
    parser.add_argument("--real2sim-dir", type=Path, default=_default_real2sim_dir(), help="real2sim root containing latest/.")
    parser.add_argument("--ply-name", default="gaussians.ply", help="GenWildSplat PLY filename inside output dir.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing registered Gaussian asset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src_ply = args.genwildsplat_output_dir / args.ply_name
    if not src_ply.exists():
        raise FileNotFoundError(f"GenWildSplat Gaussian PLY not found: {src_ply}")

    latest = args.real2sim_dir / "latest"
    dst_dir = latest / "gaussian_genwildsplat"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_ply = dst_dir / "gaussians_genwildsplat.ply"
    if dst_ply.exists() and not args.force:
        raise FileExistsError(f"Destination exists; pass --force to overwrite: {dst_ply}")
    shutil.copy2(src_ply, dst_ply)

    src_tensors = args.genwildsplat_output_dir / "gaussians.safetensors"
    dst_tensors = dst_dir / "gaussians_genwildsplat.safetensors"
    tensors_rel = None
    if src_tensors.exists():
        if dst_tensors.exists() and not args.force:
            raise FileExistsError(f"Destination exists; pass --force to overwrite: {dst_tensors}")
        shutil.copy2(src_tensors, dst_tensors)
        tensors_rel = "gaussian_genwildsplat/gaussians_genwildsplat.safetensors"

    point_count = _count_ply_vertices(dst_ply)
    gaussian_manifest = {
        "schema": "lingbot_gaussian_visual.v1",
        "source": "genwildsplat",
        "point_count": point_count,
        "ply": str(dst_ply),
        "source_ply": str(src_ply.resolve()),
        "source_output_dir": str(args.genwildsplat_output_dir.resolve()),
    }
    if tensors_rel is not None:
        gaussian_manifest["safetensors"] = str(dst_tensors)
    _write_json(dst_dir / "manifest.json", gaussian_manifest)

    latest_manifest_path = latest / "manifest.json"
    latest_manifest = _read_json(latest_manifest_path)
    latest_manifest.setdefault("schema", "lingbot_real2sim_export.v1")
    latest_manifest["gaussian_genwildsplat"] = {
        "ply": "gaussian_genwildsplat/gaussians_genwildsplat.ply",
        "manifest": "gaussian_genwildsplat/manifest.json",
        "point_count": point_count,
    }
    if tensors_rel is not None:
        latest_manifest["gaussian_genwildsplat"]["safetensors"] = tensors_rel
    _write_json(latest_manifest_path, latest_manifest)

    print(json.dumps({"registered": str(dst_ply), "point_count": point_count}, indent=2))


if __name__ == "__main__":
    main()
