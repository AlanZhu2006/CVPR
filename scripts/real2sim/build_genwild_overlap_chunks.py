#!/usr/bin/env python3
"""Build overlapping GenWildSplat chunks and register them as the live atlas."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "CVPR/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_summary.json"
DEFAULT_PREDICTIONS = REPO_ROOT / "CVPR/nuc_output/video_real2sim_playback/lingbot_vid30fps_518_full/lingbot_predictions.npz"
DEFAULT_FRAME_DIR = REPO_ROOT / "videos/vid_frames_full30"
DEFAULT_GENWILD_ROOT = REPO_ROOT / "CVPR/nuc_output/video_real2sim_playback/genwildsplat"
DEFAULT_GAUSSIAN_DIR = REPO_ROOT / "nuc_output/video_real2sim_playback/real2sim/latest/gaussian_genwildsplat"
DEFAULT_CHUNKS = [(0, 120), (60, 180), (120, 240), (180, 300)]


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd or REPO_ROOT), check=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def chunk_specs(value: str) -> list[tuple[int, int]]:
    if not value:
        return DEFAULT_CHUNKS
    chunks: list[tuple[int, int]] = []
    for raw in value.split(","):
        start_text, end_text = raw.split("-", 1)
        chunks.append((int(start_text), int(end_text)))
    return chunks


def prepare_scene(args: argparse.Namespace, start: int, end: int, chunk_root: Path) -> None:
    manifest = chunk_root / "manifest.json"
    if args.skip_existing and manifest.exists():
        return
    run(
        [
            "python3",
            "scripts/real2sim/prepare_genwildsplat_scene.py",
            "--summary-json",
            str(args.summary_json),
            "--fallback-frame-dir",
            str(args.fallback_frame_dir),
            "--scene-dir",
            str(chunk_root / "scene"),
            "--start-frame",
            str(start),
            "--end-frame",
            str(end),
            "--keyframes",
            str(args.keyframes),
            "--jpeg-quality",
            "98",
        ]
    )


def run_inference(args: argparse.Namespace, chunk_root: Path) -> None:
    output_ply = chunk_root / "inference/gaussians.ply"
    camera_poses = chunk_root / "inference/camera_poses.json"
    if args.skip_existing and output_ply.exists() and camera_poses.exists():
        return
    run(["bash", str(chunk_root / "run_inference.sh")])


def align_chunk(args: argparse.Namespace, start: int, end: int, chunk_root: Path, variant: str) -> Path:
    output_ply = args.gaussian_dir / f"gaussians_{variant}_lingbot_aligned.ply"
    output_alignment = args.gaussian_dir / f"alignment_chunk{start:03d}_{end:03d}_lingbot.json"
    if args.skip_existing and output_ply.exists() and output_alignment.exists():
        return output_ply
    run(
        [
            "python3",
            "scripts/real2sim/align_genwildsplat_to_lingbot.py",
            "--genwild-manifest",
            str(chunk_root / "manifest.json"),
            "--genwild-camera-poses",
            str(chunk_root / "inference/camera_poses.json"),
            "--genwild-ply",
            str(chunk_root / "inference/gaussians.ply"),
            "--lingbot-predictions-npz",
            str(args.predictions_npz),
            "--lingbot-summary-json",
            str(args.summary_json),
            "--output-ply",
            str(output_ply),
            "--output-alignment-json",
            str(output_alignment),
            "--geometry-source",
            "depth",
            "--conf-percentile",
            str(args.conf_percentile),
            "--points-per-frame",
            str(args.points_per_frame),
            "--icp-refine",
            "--icp-max-frame",
            str(end),
            "--icp-frame-step",
            "3",
        ]
    )
    return output_ply


def preprocess_for_web(args: argparse.Namespace, aligned_ply: Path, variant: str) -> dict[str, Any]:
    manifest_entry = args.gaussian_dir / f"{variant}_manifest_entry.json"
    if args.skip_existing and manifest_entry.exists():
        return load_json(manifest_entry)

    run(
        [
            "node",
            "GS_Console/scripts/preprocess_gaussian_stream.mjs",
            "--input",
            str(aligned_ply),
            "--variant",
            variant,
            "--grid",
            args.grid,
            "--max-sh",
            "0",
            "--force",
        ]
    )

    processed_meta = (
        REPO_ROOT
        / "GS_Console/runtime/processed"
        / aligned_ply.stem
        / "model"
        / f"gs_chunks_{variant}.json"
    )
    metadata = load_json(processed_meta)
    chunks = []
    for chunk in metadata.get("chunks", []):
        src = Path(chunk["path"])
        dst = args.gaussian_dir / f"{variant}_{chunk['file']}"
        shutil.copy2(src, dst)
        chunks.append(
            {
                "id": chunk.get("id"),
                "url": f"/lingbot-live-assets/real2sim/latest/gaussian_genwildsplat/{dst.name}",
                "bytes": chunk.get("bytes", dst.stat().st_size),
                "splats": chunk.get("splats"),
                "center": chunk.get("center"),
                "bounds": chunk.get("bounds"),
            }
        )
    entry = {
        "format": "gs-chunks",
        "variant": variant,
        "label": f"GenWildSplat overlap chunk {variant.removeprefix('genwildsplat_chunk')} full quality",
        "shDegree": int(metadata.get("output", {}).get("shDegree", 0)),
        "chunks": chunks,
    }
    write_json(manifest_entry, entry)
    return entry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--predictions-npz", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--fallback-frame-dir", type=Path, default=DEFAULT_FRAME_DIR)
    parser.add_argument("--genwild-root", type=Path, default=DEFAULT_GENWILD_ROOT)
    parser.add_argument("--gaussian-dir", type=Path, default=DEFAULT_GAUSSIAN_DIR)
    parser.add_argument("--chunks", default="", help="Comma-separated ranges, e.g. 0-120,60-180")
    parser.add_argument("--keyframes", type=int, default=8)
    parser.add_argument("--grid", default="8,4")
    parser.add_argument("--conf-percentile", type=float, default=65.0)
    parser.add_argument("--points-per-frame", type=int, default=8000)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.gaussian_dir.mkdir(parents=True, exist_ok=True)
    for start, end in chunk_specs(args.chunks):
        variant = f"genwildsplat_chunk{start:03d}_{end:03d}"
        chunk_root = args.genwild_root / f"lingbot_vid30fps_518_overlap_chunk{start:03d}_{end:03d}_{args.keyframes}ctx"
        prepare_scene(args, start, end, chunk_root)
        run_inference(args, chunk_root)
        aligned_ply = align_chunk(args, start, end, chunk_root, variant)
        preprocess_for_web(args, aligned_ply, variant)

    run(["python3", "scripts/real2sim/register_genwild_chunk_atlas.py"])


if __name__ == "__main__":
    main()
