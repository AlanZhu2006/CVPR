from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_summary(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare_run_group(summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not summaries:
        raise ValueError("No summaries provided.")

    task = summaries[0]["task"]
    dataset = summaries[0]["dataset"]
    baseline = None
    for summary in summaries:
        if summary["mode"] == "ttt3r":
            baseline = summary
            break

    rows = []
    for summary in summaries:
        row = {
            "mode": summary["mode"],
            "avg_fps": summary.get("avg_fps"),
            "archive_count": summary.get("archive_count"),
            "retrieval_attempts": summary.get("retrieval_attempts"),
            "retrieval_successes": summary.get("retrieval_successes"),
            "retrieval_redundant_skips": summary.get("retrieval_redundant_skips"),
            "retrieval_threshold_rejects": summary.get("retrieval_threshold_rejects"),
            "retrieval_gap_rejects": summary.get("retrieval_gap_rejects"),
            "retrieval_sequence_rejects": summary.get("retrieval_sequence_rejects"),
            "geometry_verification_rollouts": summary.get("geometry_verification_rollouts"),
            "geometry_verification_accepts": summary.get("geometry_verification_accepts"),
            "geometry_verification_rejects": summary.get("geometry_verification_rejects"),
            "avg_geometry_verification_geo_gain": summary.get("avg_geometry_verification_geo_gain"),
            "recovery_success_rate": summary.get("recovery_success_rate"),
        }
        if task == "relpose":
            row["avg_ate"] = summary.get("avg_ate")
            row["avg_rpe_trans"] = summary.get("avg_rpe_trans")
            row["avg_rpe_rot"] = summary.get("avg_rpe_rot")
            if baseline is not None and summary["mode"] != baseline["mode"]:
                row["delta_ate_vs_ttt3r"] = summary["avg_ate"] - baseline["avg_ate"]
                row["delta_rpe_trans_vs_ttt3r"] = summary["avg_rpe_trans"] - baseline["avg_rpe_trans"]
        else:
            scale_shift = summary.get("metrics", {}).get("scale&shift", {})
            row["rmse_scale_shift"] = scale_shift.get("rmse")
            row["abs_rel_scale_shift"] = scale_shift.get("abs_rel")
            if baseline is not None and summary["mode"] != baseline["mode"]:
                baseline_metric = baseline.get("metrics", {}).get("scale&shift", {})
                if "rmse" in scale_shift and "rmse" in baseline_metric:
                    row["delta_rmse_vs_ttt3r"] = scale_shift["rmse"] - baseline_metric["rmse"]
        rows.append(row)

    return {"task": task, "dataset": dataset, "rows": rows}
