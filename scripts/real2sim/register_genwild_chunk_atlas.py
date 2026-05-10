#!/usr/bin/env python3
"""Register aligned GenWildSplat chunks as the live Gaussian atlas.

This does not build new Gaussians. It promotes already aligned chunk PLYs in the
WebUI manifest so live playback selects a local Gaussian by frame index instead
of a single long-sequence global asset.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAUSSIAN_DIR = REPO_ROOT / "nuc_output/video_real2sim_playback/real2sim/latest/gaussian_genwildsplat"
DEFAULT_GENWILD_ROOT = REPO_ROOT / "CVPR/nuc_output/video_real2sim_playback/genwildsplat"
DEFAULT_MANIFESTS = [
    REPO_ROOT / "GS_Console/examples/web-ui/public/scenes/lingbot-live/manifest.json",
    REPO_ROOT / "GS_Console/examples/web-ui/public/scenes/lingbot-real2sim-genwild/manifest.json",
]
PREFERRED_OVERLAP_RANGES = [(0, 120), (60, 180), (120, 240), (180, 300)]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compact_alignment(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": alignment.get("schema"),
        "target": alignment.get("target"),
        "scale": alignment.get("scale"),
        "center_error_mean": alignment.get("center_error_mean"),
        "center_error_max": alignment.get("center_error_max"),
        "inlier_center_error_mean": alignment.get("inlier_center_error_mean"),
        "inlier_center_error_max": alignment.get("inlier_center_error_max"),
        "icp_refine": alignment.get("icp_refine"),
        "icp_stats": alignment.get("icp_stats"),
        "aligned_vertex_count": alignment.get("aligned_vertex_count"),
        "source_to_target_4x4_row_major": alignment.get("source_to_target_4x4_row_major"),
    }


def find_genwild_manifest(genwild_root: Path, start: int, end: int) -> Path | None:
    candidates = sorted(genwild_root.glob(f"*chunk{start:03d}_{end:03d}/manifest.json"))
    return candidates[0] if candidates else None


def discover_atlas_chunks(gaussian_dir: Path, genwild_root: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for alignment_path in sorted(gaussian_dir.glob("alignment_chunk*_lingbot.json")):
        match = re.search(r"alignment_chunk(\d+)_(\d+)_lingbot\.json$", alignment_path.name)
        if not match:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        alignment = load_json(alignment_path)
        genwild_manifest_path = find_genwild_manifest(genwild_root, start, end)
        genwild_manifest: dict[str, Any] = {}
        if genwild_manifest_path and genwild_manifest_path.exists():
            genwild_manifest = load_json(genwild_manifest_path)
        keyframes = alignment.get("keyframe_indices") or genwild_manifest.get("keyframe_indices") or []
        chunks.append(
            {
                "variant": f"genwildsplat_chunk{start:03d}_{end:03d}",
                "frameRange": {"start": start, "end": end},
                "keyframeIndices": keyframes,
                "alignmentPath": str(alignment_path.relative_to(REPO_ROOT)),
                "alignment": compact_alignment(alignment),
                "genwildManifestPath": (
                    str(genwild_manifest_path.relative_to(REPO_ROOT)) if genwild_manifest_path else None
                ),
            }
        )
    return chunks


def choose_active_atlas_chunks(chunks: list[dict[str, Any]], prefer_overlap: bool) -> list[dict[str, Any]]:
    by_range = {
        (int(entry["frameRange"]["start"]), int(entry["frameRange"]["end"])): entry
        for entry in chunks
    }
    if prefer_overlap and all(frame_range in by_range for frame_range in PREFERRED_OVERLAP_RANGES):
        return [by_range[frame_range] for frame_range in PREFERRED_OVERLAP_RANGES]
    return sorted(chunks, key=lambda item: (int(item["frameRange"]["start"]), int(item["frameRange"]["end"])))


def load_variant_source(gaussian_dir: Path, variant: str) -> dict[str, Any] | None:
    manifest_entry = gaussian_dir / f"{variant}_manifest_entry.json"
    if manifest_entry.exists():
        return load_json(manifest_entry)
    return None


def update_manifest(path: Path, gaussian_dir: Path, atlas_chunks: list[dict[str, Any]], active_variants: set[str]) -> None:
    manifest = load_json(path)
    variants = manifest.setdefault("gaussianVariants", {})

    for entry in atlas_chunks:
        variant_name = entry["variant"]
        if variant_name not in variants:
            source = load_variant_source(gaussian_dir, variant_name)
            if source is None:
                raise RuntimeError(f"{path} is missing Gaussian variant {variant_name} and no manifest entry exists")
            variants[variant_name] = source
        variant = variants[variant_name]
        variant["variant"] = entry["variant"]
        variant["atlasRole"] = "overlap_chunk" if variant_name in active_variants else "local_chunk_fallback"
        variant["frameRange"] = entry["frameRange"]
        variant["keyframeIndices"] = entry["keyframeIndices"]
        variant["alignment"] = entry["alignment"]
        variant["description"] = (
            f"Aligned GenWildSplat local chunk for frames "
            f"{entry['frameRange']['start']}-{entry['frameRange']['end']}."
        )

    source = manifest.setdefault("source", {})
    source["liveGaussianPolicy"] = "chunk_atlas"
    source["chunkCrossFadeFrames"] = 18
    source.setdefault("globalGaussianPolicy", "available_as_manual_fallback")
    source["chunkAtlasManifestUrl"] = "/lingbot-live-assets/real2sim/latest/gaussian_genwildsplat/chunk_atlas.json"

    first_chunk = next(entry["variant"] for entry in atlas_chunks if entry["variant"] in active_variants)
    manifest["gaussian"] = variants[first_chunk]
    manifest.setdefault("assets", {})["gaussian"] = "chunk-atlas-ready"
    write_json(path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-dir", type=Path, default=DEFAULT_GAUSSIAN_DIR)
    parser.add_argument("--genwild-root", type=Path, default=DEFAULT_GENWILD_ROOT)
    parser.add_argument("--atlas-json", type=Path, default=DEFAULT_GAUSSIAN_DIR / "chunk_atlas.json")
    parser.add_argument("--manifest", action="append", type=Path, dest="manifests")
    parser.add_argument("--no-prefer-overlap", action="store_true", help="Do not prefer 0-120/60-180/120-240/180-300 chunks when they exist.")
    args = parser.parse_args()

    manifests = args.manifests or DEFAULT_MANIFESTS
    all_chunks = discover_atlas_chunks(args.gaussian_dir, args.genwild_root)
    if not all_chunks:
        raise RuntimeError(f"No aligned chunk alignment JSON files found under {args.gaussian_dir}")
    atlas_chunks = choose_active_atlas_chunks(all_chunks, prefer_overlap=not args.no_prefer_overlap)
    active_variants = {entry["variant"] for entry in atlas_chunks}

    atlas_payload = {
        "schema": "genwildsplat_chunk_atlas.v1",
        "policy": "chunk_atlas",
        "activeVariants": sorted(active_variants),
        "notes": (
            "Live playback selects one aligned local GenWildSplat Gaussian chunk "
            "by frame index. Global variants remain available for manual inspection."
        ),
        "chunks": atlas_chunks,
    }
    write_json(args.atlas_json, atlas_payload)

    for manifest_path in manifests:
        update_manifest(manifest_path, args.gaussian_dir, all_chunks, active_variants)

    print(f"Registered {len(atlas_chunks)} GenWildSplat chunks")
    print(f"Wrote atlas: {args.atlas_json}")
    for manifest_path in manifests:
        print(f"Updated manifest: {manifest_path}")


if __name__ == "__main__":
    main()
