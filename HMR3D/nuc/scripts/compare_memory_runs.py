from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((run_dir / "events.json").read_text(encoding="utf-8"))
    return summary, events


def compact(summary: dict, events: list[dict]) -> dict:
    counts = Counter(event["event_type"] for event in events)
    stats = summary.get("stats", {})
    return {
        "keyframes": summary.get("keyframe_count", 0),
        "event_count": summary.get("event_count", 0),
        "bank_size": len(summary.get("bank", [])),
        "scene_summaries": len(summary.get("scene_summaries", [])),
        "archives": stats.get("archives", 0),
        "recoveries": stats.get("recoveries", 0),
        "retrieve_hits": stats.get("retrieve_hits", 0),
        "archive_write_rejects": stats.get("archive_write_rejects", 0),
        "archive_deferred_max_keyframes": stats.get("archive_deferred_max_keyframes", 0),
        "archive_deferred_max_age": stats.get("archive_deferred_max_age", 0),
        "archive_deferred_pose_distance": stats.get("archive_deferred_pose_distance", 0),
        "archive_deferred_similarity_drop": stats.get("archive_deferred_similarity_drop", 0),
        "recover_pose_anchor_rejects": stats.get("recover_pose_anchor_rejects", 0),
        "shadow_buffered": stats.get("shadow_buffered", 0),
        "shadow_promotions": stats.get("shadow_promotions", 0),
        "retrieve_routed_candidates": stats.get("retrieve_routed_candidates", 0),
        "scene_routing_hits": stats.get("scene_routing_hits", 0),
        "merge_events": stats.get("merge_events", 0),
        "merged_candidates_total": stats.get("merged_candidates_total", 0),
        "local_adapt_applied": stats.get("local_adapt_applied", 0),
        "gaussian_archives": stats.get("gaussian_archives", 0),
        "gaussian_archived_points_total": stats.get("gaussian_archived_points_total", 0),
        "gaussian_warmstart_requests": stats.get("gaussian_warmstart_requests", 0),
        "gaussian_warmstart_points": stats.get("gaussian_warmstart_points", 0),
        "event_types": dict(counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two HMR3D memory replay outputs.")
    parser.add_argument("--left", required=True, help="Path to first output dir.")
    parser.add_argument("--right", required=True, help="Path to second output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_dir = Path(args.left).expanduser().resolve()
    right_dir = Path(args.right).expanduser().resolve()

    left_summary, left_events = load_run(left_dir)
    right_summary, right_events = load_run(right_dir)

    report = {
        "left": {
            "run_dir": str(left_dir),
            "summary": compact(left_summary, left_events),
        },
        "right": {
            "run_dir": str(right_dir),
            "summary": compact(right_summary, right_events),
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
