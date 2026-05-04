#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
from urllib import request as urllib_request


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ready_results(worker_dir: Path) -> list[Path]:
    ready: list[Path] = []
    for path in sorted(worker_dir.glob("window_*/worker_result.json")):
        payload = _read_json(path)
        if not payload:
            continue
        summary = Path(str(payload.get("summary_json", "")))
        predictions = Path(str(payload.get("predictions_npz", "")))
        if summary.exists() and predictions.exists():
            ready.append(path)
    return ready


def _frame_records(result_paths: list[Path]) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for result_path in result_paths:
        result = _read_json(result_path)
        if not result:
            continue
        summary_path = Path(str(result.get("summary_json", "")))
        summary = _read_json(summary_path)
        if not summary:
            continue
        result_images = [str(path) for path in result.get("image_paths", [])]
        for local_index, frame in enumerate(summary.get("metadata", {}).get("frames", [])):
            image_path = Path(str(frame.get("image_path", "")))
            if not image_path.exists() and local_index < len(result_images):
                image_path = Path(result_images[local_index])
            if not image_path.exists():
                continue
            idx = int(frame.get("frame_idx", result.get("index", len(records))))
            meta = frame.get("metadata") or {}
            records[idx] = {
                "frame_idx": idx,
                "timestamp_sec": float(frame.get("timestamp_sec", 0.0)),
                "image_path": str(image_path),
                "worker_result": str(result_path),
                "track_ok": bool(meta.get("track_ok", True)),
                "pose": meta.get("pose"),
            }
    return [records[key] for key in sorted(records)]


def _select_context(records: list[dict[str, Any]], views: int) -> list[dict[str, Any]]:
    if not records:
        return []
    views = max(2, int(views))
    if len(records) <= views:
        return records
    indices = [round(i * (len(records) - 1) / max(1, views - 1)) for i in range(views)]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            continue
        seen.add(index)
        selected.append(records[index])
    return selected


def _copy_context_images(job_dir: Path, records: list[dict[str, Any]], image_size: int) -> list[dict[str, Any]]:
    images_dir = job_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for out_index, record in enumerate(records):
        src = Path(str(record["image_path"]))
        dst = images_dir / f"{out_index:06d}{src.suffix.lower() or '.png'}"
        shutil.copyfile(src, dst)
        copied.append(
            {
                **record,
                "context_index": out_index,
                "job_image": str(dst),
                "job_image_relative": str(dst.relative_to(job_dir)),
                "target_image_size": int(image_size),
            }
        )
    return copied


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _post_sidecar(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=float(timeout_sec)) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw.strip() else {"ok": True}


def prepare_once(args: argparse.Namespace, last_ready_count: int) -> tuple[int, dict[str, Any] | None]:
    worker_dir = Path(args.worker_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    jobs_dir = output_dir / "monitor" / "feedforward_jobs"
    ready = _ready_results(worker_dir)
    ready_count = len(ready)
    if ready_count < int(args.min_ready_windows):
        return ready_count, None
    if not args.force and ready_count - last_ready_count < int(args.min_new_windows):
        return ready_count, None

    window_count = max(int(args.context_views), int(args.window_results))
    records = _frame_records(ready[-window_count:])
    context = _select_context(records, int(args.context_views))
    if len(context) < 2:
        return ready_count, None

    job_id = f"ffgs_{time.strftime('%Y%m%d_%H%M%S')}_w{ready_count:04d}_f{context[0]['frame_idx']:06d}_{context[-1]['frame_idx']:06d}"
    job_dir = jobs_dir / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    copied = _copy_context_images(job_dir, context, int(args.image_size))

    request_payload = {
        "schema": "lingbot_feedforward_gs_job.v1",
        "job_id": job_id,
        "kind": "genwildsplat_like_context",
        "created_at": time.time(),
        "created_at_text": time.strftime("%Y-%m-%d %H:%M:%S"),
        "worker_dir": str(worker_dir),
        "ready_window_count": ready_count,
        "context_views": len(copied),
        "image_size": int(args.image_size),
        "job_dir": str(job_dir),
        "images_dir": str(job_dir / "images"),
        "frames": copied,
        "expected_outputs": {
            "gaussian_ply": str(job_dir / "outputs" / "gaussians.ply"),
            "preview_png": str(job_dir / "outputs" / "preview.png"),
            "summary_json": str(job_dir / "outputs" / "summary.json"),
        },
        "notes": (
            "Feed-forward GS sidecar input. GenWildSplat/AnySplat can ignore poses, "
            "but LingBot poses are included for optional alignment and temporal smoothing."
        ),
    }
    _atomic_write_json(job_dir / "request.json", request_payload)
    _atomic_write_json(output_dir / "monitor" / "latest_feedforward_request.json", request_payload)

    sidecar_response: dict[str, Any] | None = None
    if args.sidecar_url:
        sidecar_response = _post_sidecar(
            args.sidecar_url,
            {"job_dir": str(job_dir), "request": request_payload},
            float(args.sidecar_timeout_sec),
        )
        _atomic_write_json(job_dir / "sidecar_response.json", sidecar_response)

    status = {
        "schema": "lingbot_feedforward_gs_status.v1",
        "job_id": job_id,
        "job_dir": str(job_dir),
        "ready_window_count": ready_count,
        "context_views": len(copied),
        "sidecar_url": args.sidecar_url,
        "sidecar_response": sidecar_response,
        "updated_at": time.time(),
    }
    _atomic_write_json(output_dir / "monitor" / "latest_feedforward_status.json", status)
    return ready_count, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare GenWildSplat/AnySplat-style feed-forward live GS jobs from LingBot keyframes."
    )
    parser.add_argument("--worker-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-views", type=int, default=6)
    parser.add_argument("--window-results", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--interval-sec", type=float, default=3.0)
    parser.add_argument("--min-ready-windows", type=int, default=2)
    parser.add_argument("--min-new-windows", type=int, default=2)
    parser.add_argument("--sidecar-url", default="")
    parser.add_argument("--sidecar-timeout-sec", type=float, default=2.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    last_ready_count = -10**9 if args.force else 0
    while True:
        try:
            ready_count, status = prepare_once(args, last_ready_count)
            if status:
                last_ready_count = ready_count
                print(json.dumps(status, ensure_ascii=False), flush=True)
            elif args.once:
                print(json.dumps({"status": "skipped", "ready_window_count": ready_count}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), flush=True)
        if args.once:
            break
        time.sleep(max(0.5, float(args.interval_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
