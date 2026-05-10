#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

from gssdf_training_camera_pose import build_pose, resolve_dataset_path, training_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and compare one GS-SDF training camera frame across RGB, saved render, and live HiFi."
    )
    parser.add_argument("--output-dir", required=True, help="GS-SDF output directory.")
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8890")
    parser.add_argument("--hifi-url", default="http://127.0.0.1:8876")
    parser.add_argument("--split", choices=("train", "raw"), default="train")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--settle-frames", type=int, default=3)
    parser.add_argument("--output-report-dir", default="")
    return parser.parse_args()


def http_get_json(url: str) -> dict:
    with urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8") or "{}")


def fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=20) as response:
        return response.read()


def wait_for_frame_advance(base_url: str, start_count: int, settle_frames: int, *, counter_key: str) -> dict:
    target = start_count + max(1, int(settle_frames))
    deadline = time.monotonic() + 8.0
    latest: dict = {}
    while time.monotonic() < deadline:
        latest = http_get_json(f"{base_url.rstrip('/')}/status")
        if int(latest.get(counter_key, 0) or 0) >= target:
            return latest
        time.sleep(0.05)
    return latest


def image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 1.0e-12 else float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))
    return {"mse": mse, "mae": mae, "psnr": psnr}


def find_saved_render(output_dir: Path, split: str, rgb_path: Path) -> Path | None:
    target_stem = rgb_path.resolve().stem
    render_dir = output_dir / "gs_log" / split / "color" / "renders"
    if not render_dir.is_dir():
        return None
    for candidate in sorted(render_dir.iterdir()):
        if candidate.is_file() and target_stem in candidate.stem:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_path = resolve_dataset_path(argparse.Namespace(dataset=None, output_dir=str(output_dir), scene_config=None))
    images_path, rows = training_rows(dataset_path, args.split)
    if args.frame_index < 0 or args.frame_index >= len(rows):
        raise SystemExit(f"frame-index {args.frame_index} out of range for {len(rows)} {args.split} rows")

    built = build_pose(rows[args.frame_index], frame_id="world", target_distance=4.0)
    source_image = str(built["sourceImage"])
    rgb_path = dataset_path / "images" / source_image
    if not rgb_path.is_file():
        raise SystemExit(f"Source RGB image not found: {rgb_path}")

    hifi_url = args.hifi_url.rstrip("/")
    hifi_status_before = http_get_json(f"{hifi_url}/status")
    http_post_json(f"{hifi_url}/camera_pose", {"position": built["pose"]["position"], "orientation": built["pose"]["orientation"]})
    hifi_status_after = wait_for_frame_advance(
        hifi_url,
        int(hifi_status_before.get("poseMessagesSent", 0) or 0),
        int(args.settle_frames),
        counter_key="poseMessagesSent",
    )

    live_gs_bytes = fetch_bytes(f"{hifi_url}/frame.jpg")

    report_dir = (
        Path(args.output_report_dir).expanduser().resolve()
        if args.output_report_dir
        else output_dir / "quality_reports" / f"{args.split}_{args.frame_index:04d}_{Path(source_image).stem}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "live_hifi_gs.jpg").write_bytes(live_gs_bytes)

    original_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    live_gs = cv2.imread(str(report_dir / "live_hifi_gs.jpg"), cv2.IMREAD_COLOR)
    if original_rgb is None or live_gs is None:
        raise SystemExit("Failed to decode one or more comparison images.")
    cv2.imwrite(str(report_dir / "original_rgb.jpg"), original_rgb)

    saved_render_path = find_saved_render(output_dir, args.split, rgb_path)
    saved_render_metrics = None
    if saved_render_path is not None:
        saved_render = cv2.imread(str(saved_render_path), cv2.IMREAD_COLOR)
        if saved_render is not None:
            saved_render_metrics = image_metrics(original_rgb, saved_render)
            cv2.imwrite(str(report_dir / "saved_gs_render.jpg"), saved_render)

    report = {
        "outputDir": str(output_dir),
        "datasetPath": str(dataset_path),
        "imagesPath": str(images_path),
        "split": args.split,
        "frameIndex": args.frame_index,
        "sourceImage": source_image,
        "sourceRgbPath": str(rgb_path),
        "savedRenderPath": str(saved_render_path) if saved_render_path else None,
        "hifiStatus": {
            key: hifi_status_after.get(key)
            for key in (
                "ready",
                "frameIsFresh",
                "frameWidth",
                "frameHeight",
                "poseMessagesSent",
            )
        },
        "metrics": {
            "liveHifiGsVsOriginal": image_metrics(original_rgb, live_gs),
            "savedRenderVsOriginal": saved_render_metrics,
        },
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
