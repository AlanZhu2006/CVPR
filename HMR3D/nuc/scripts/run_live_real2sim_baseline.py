#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch LingBot live worker outputs and periodically export a "
            "real-to-sim baseline with RGB-D keyframes, TSDF, mesh, and Gaussian seed."
        )
    )
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence-prefix", default="live_real2sim")
    parser.add_argument("--latest-name", default="latest")
    parser.add_argument("--interval-sec", type=float, default=180.0)
    parser.add_argument("--min-new-windows", type=int, default=6)
    parser.add_argument("--min-ready-windows", type=int, default=2)
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--depth-scale", type=float, default=20.0)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=35.0)
    parser.add_argument("--min-conf", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-points-per-frame", type=int, default=12000)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--voxel-size", type=float, default=0.12)
    parser.add_argument("--mesh-backend", choices=("tsdf", "voxel"), default="tsdf")
    parser.add_argument("--tsdf-trunc-multiplier", type=float, default=4.0)
    parser.add_argument("--tsdf-max-dim", type=int, default=140)
    parser.add_argument("--tsdf-chunk-voxels", type=int, default=120000)
    parser.add_argument("--tsdf-min-weight", type=float, default=1.0)
    parser.add_argument("--tsdf-weight-cap", type=float, default=32.0)
    parser.add_argument("--tsdf-bounds-percentile", type=float, default=99.0)
    parser.add_argument("--tsdf-max-frames", type=int, default=0)
    parser.add_argument("--mesh-max-voxels", type=int, default=120000)
    parser.add_argument("--gaussian-max-points", type=int, default=80000)
    parser.add_argument("--gaussian-scale", type=float, default=0.07)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ready_results(worker_dir: Path) -> list[Path]:
    ready: list[Path] = []
    for path in sorted(worker_dir.glob("window_*/worker_result.json")):
        result = _read_json(path)
        if not result:
            continue
        pred = Path(str(result.get("predictions_npz", "")))
        summary = Path(str(result.get("summary_json", "")))
        if pred.exists() and summary.exists():
            ready.append(path)
    return ready


def _sequence_name(prefix: str, ready_count: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{prefix}_{stamp}_w{ready_count:04d}"


def _script_path() -> Path:
    return Path(__file__).resolve().with_name("export_lingbot_worker_to_real2sim.py")


def _export_once(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(_script_path()),
        "--worker-dir",
        str(Path(args.worker_dir).expanduser().resolve()),
        "--output-dir",
        str(Path(args.output_dir).expanduser().resolve()),
        "--sequence",
        sequence,
        "--depth-scale",
        str(args.depth_scale),
        "--fx",
        str(args.fx),
        "--fy",
        str(args.fy),
        "--cx",
        str(args.cx),
        "--cy",
        str(args.cy),
        "--min-depth",
        str(args.min_depth),
        "--max-depth",
        str(args.max_depth),
        "--min-conf",
        str(args.min_conf),
        "--sample-stride",
        str(args.sample_stride),
        "--max-points-per-frame",
        str(args.max_points_per_frame),
        "--max-frames",
        str(args.max_frames),
        "--voxel-size",
        str(args.voxel_size),
        "--mesh-backend",
        str(args.mesh_backend),
        "--tsdf-trunc-multiplier",
        str(args.tsdf_trunc_multiplier),
        "--tsdf-max-dim",
        str(args.tsdf_max_dim),
        "--tsdf-chunk-voxels",
        str(args.tsdf_chunk_voxels),
        "--tsdf-min-weight",
        str(args.tsdf_min_weight),
        "--tsdf-weight-cap",
        str(args.tsdf_weight_cap),
        "--tsdf-bounds-percentile",
        str(args.tsdf_bounds_percentile),
        "--tsdf-max-frames",
        str(args.tsdf_max_frames),
        "--mesh-max-voxels",
        str(args.mesh_max_voxels),
        "--gaussian-max-points",
        str(args.gaussian_max_points),
        "--gaussian-scale",
        str(args.gaussian_scale),
        "--overwrite",
    ]
    print(f"[real2sim] exporting sequence={sequence}", flush=True)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if proc.stdout:
        print(proc.stdout.rstrip(), flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"export failed with code {proc.returncode}")
    manifest = Path(args.output_dir).expanduser().resolve() / sequence / "manifest.json"
    data = _read_json(manifest)
    if data is None:
        raise RuntimeError(f"export did not produce a valid manifest: {manifest}")
    return data


def _update_latest(output_dir: Path, latest_name: str, sequence_dir: Path, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = output_dir / latest_name
    temp = output_dir / f".{latest_name}.tmp"
    if temp.exists() or temp.is_symlink():
        if temp.is_dir() and not temp.is_symlink():
            shutil.rmtree(temp)
        else:
            temp.unlink()
    try:
        os.symlink(sequence_dir.name, temp, target_is_directory=True)
        os.replace(temp, latest)
        latest_kind = "symlink"
    except OSError as exc:
        if temp.exists() or temp.is_symlink():
            temp.unlink()
        latest_kind = f"manifest_only:{exc}"
    latest_manifest = output_dir / f"{latest_name}_manifest.json"
    latest_manifest_tmp = output_dir / f".{latest_name}_manifest.json.tmp"
    payload = {
        "latest_kind": latest_kind,
        "sequence": sequence_dir.name,
        "sequence_dir": str(sequence_dir),
        "manifest": str(sequence_dir / "manifest.json"),
        "updated_at": time.time(),
        "export": manifest,
    }
    latest_manifest_tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(latest_manifest_tmp, latest_manifest)


def _cleanup_old(output_dir: Path, prefix: str, keep_last: int, latest_sequence: str) -> None:
    keep_last = int(keep_last)
    if keep_last <= 0:
        return
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(prefix + "_")]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates[keep_last:]:
        if path.name == latest_sequence:
            continue
        shutil.rmtree(path, ignore_errors=True)


def _write_state(output_dir: Path, state: dict[str, Any]) -> None:
    path = output_dir / "live_real2sim_runner_state.json"
    tmp = output_dir / ".live_real2sim_runner_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    worker_dir = Path(args.worker_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_export_ready_count = -1
    last_sequence = ""

    while True:
        ready = _ready_results(worker_dir)
        ready_count = len(ready)
        should_export = bool(args.force and last_export_ready_count < 0)
        should_export |= ready_count >= int(args.min_ready_windows) and last_export_ready_count < 0
        should_export |= ready_count >= last_export_ready_count + max(1, int(args.min_new_windows))
        if should_export and ready_count > 0:
            sequence = _sequence_name(str(args.sequence_prefix), ready_count)
            try:
                manifest = _export_once(args, sequence)
                sequence_dir = output_dir / sequence
                _update_latest(output_dir, str(args.latest_name), sequence_dir, manifest)
                _cleanup_old(output_dir, str(args.sequence_prefix), int(args.keep_last), sequence)
                last_export_ready_count = ready_count
                last_sequence = sequence
                _write_state(
                    output_dir,
                    {
                        "status": "ok",
                        "last_sequence": last_sequence,
                        "last_export_ready_count": last_export_ready_count,
                        "ready_count": ready_count,
                        "updated_at": time.time(),
                    },
                )
            except Exception as exc:
                _write_state(
                    output_dir,
                    {
                        "status": "error",
                        "error": str(exc),
                        "last_sequence": last_sequence,
                        "last_export_ready_count": last_export_ready_count,
                        "ready_count": ready_count,
                        "updated_at": time.time(),
                    },
                )
                print(f"[real2sim] export error: {exc}", file=sys.stderr, flush=True)
        else:
            print(
                f"[real2sim] waiting ready={ready_count} "
                f"last_export={last_export_ready_count} sequence={last_sequence or '-'}",
                flush=True,
            )
        if args.once:
            break
        time.sleep(max(1.0, float(args.interval_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
