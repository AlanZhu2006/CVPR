#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "queue_wait_sec",
    "lingbot_elapsed_sec",
    "worker_end_to_end_sec",
    "geometry_age_sec",
    "new_geometry_fps",
    "dense_queue_size",
    "dropped_window_count",
    "processed_window_count",
    "preprocess_sec",
    "image_to_device_sec",
    "model_forward_sec",
    "postprocess_sec",
    "tensor_to_numpy_sec",
    "predictions_npz_load_sec",
    "pointcloud_build_sec",
    "process_result_sec",
    "ros_current_publish_sec",
    "ros_publish_sec",
    "live_write_sec",
    "points_added_or_replaced",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LingBot live_update JSON metrics from a log file.")
    parser.add_argument("--log", default="/tmp/lingbot_live.log")
    parser.add_argument("--last", type=int, default=80, help="Use only the last N live_update events. 0 uses all.")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0.0}
    return {
        "count": float(len(finite)),
        "mean": float(sum(finite) / len(finite)),
        "median": float(finite[len(finite) // 2]),
        "min": float(finite[0]),
        "max": float(finite[-1]),
        "p90": float(finite[min(len(finite) - 1, int(math.ceil(0.90 * len(finite))) - 1)]),
    }


def iter_live_updates(path: Path) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = payload.get("live_update")
        if isinstance(event, dict):
            event = dict(event)
            if "point_count" in payload:
                event["point_count"] = payload["point_count"]
            updates.append(event)
    return updates


def summarize(updates: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    summary = {
        "event_count": len(updates),
        "first_window": updates[0].get("window") if updates else None,
        "last_window": updates[-1].get("window") if updates else None,
        "metrics": {},
        "latest": updates[-1] if updates else {},
    }
    for name in metric_names:
        values: list[float] = []
        for event in updates:
            value = event.get(name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        summary["metrics"][name] = stats(values)
    return summary


def print_table(summary: dict[str, Any], metric_names: list[str]) -> None:
    print(f"events: {summary['event_count']}  windows: {summary['first_window']} -> {summary['last_window']}")
    print(f"{'metric':32s} {'mean':>10s} {'median':>10s} {'p90':>10s} {'min':>10s} {'max':>10s}")
    print("-" * 88)
    for name in metric_names:
        item = summary["metrics"].get(name, {})
        if not item or item.get("count", 0.0) <= 0:
            continue
        print(
            f"{name:32s} "
            f"{item.get('mean', 0.0):10.4f} "
            f"{item.get('median', 0.0):10.4f} "
            f"{item.get('p90', 0.0):10.4f} "
            f"{item.get('min', 0.0):10.4f} "
            f"{item.get('max', 0.0):10.4f}"
        )
    latest = summary.get("latest") or {}
    if latest:
        print("\nlatest:")
        keys = [
            "window",
            "queue_wait_sec",
            "worker_end_to_end_sec",
            "geometry_age_sec",
            "new_geometry_fps",
            "dense_queue_size",
            "dropped_window_count",
            "model_forward_sec",
            "pointcloud_build_sec",
            "ros_current_publish_sec",
        ]
        for key in keys:
            if key in latest:
                print(f"  {key}: {latest[key]}")


def main() -> int:
    args = parse_args()
    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"Missing log: {log_path}")
    updates = iter_live_updates(log_path)
    if args.last > 0:
        updates = updates[-args.last :]
    summary = summarize(updates, args.metrics)
    print_table(summary, args.metrics)
    if args.output_json:
        Path(args.output_json).expanduser().resolve().write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
