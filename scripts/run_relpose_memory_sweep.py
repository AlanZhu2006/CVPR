#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _merge_memory(base: Dict[str, Any] | None, override: Dict[str, Any] | None) -> MemoryConfig:
    from hmr3d_memory.config import MemoryConfig

    payload: Dict[str, Any] = {}
    if base:
        payload.update(base)
    if override:
        payload.update(override)
    return MemoryConfig.from_dict(payload)


def _build_row(
    summary: Dict[str, Any],
    *,
    trial_name: str,
    memory_override: Dict[str, Any],
    baseline: Dict[str, Any] | None,
) -> Dict[str, Any]:
    row = {
        "trial_name": trial_name,
        "mode": summary["mode"],
        "avg_ate": summary["avg_ate"],
        "avg_rpe_trans": summary["avg_rpe_trans"],
        "avg_rpe_rot": summary["avg_rpe_rot"],
        "avg_fps": summary["avg_fps"],
        "archive_count": summary["archive_count"],
        "retrieval_attempts": summary["retrieval_attempts"],
        "retrieval_successes": summary["retrieval_successes"],
        "retrieval_redundant_skips": summary.get("retrieval_redundant_skips", 0),
        "retrieval_threshold_rejects": summary.get("retrieval_threshold_rejects", 0),
        "retrieval_gap_rejects": summary.get("retrieval_gap_rejects", 0),
        "retrieval_sequence_rejects": summary.get("retrieval_sequence_rejects", 0),
        "geometry_verification_rollouts": summary.get("geometry_verification_rollouts", 0),
        "geometry_verification_accepts": summary.get("geometry_verification_accepts", 0),
        "geometry_verification_rejects": summary.get("geometry_verification_rejects", 0),
        "avg_geometry_verification_geo_gain": summary.get("avg_geometry_verification_geo_gain", 0.0),
        "anchor_pose_verification_accepts": summary.get("anchor_pose_verification_accepts", 0),
        "anchor_pose_verification_rejects": summary.get("anchor_pose_verification_rejects", 0),
        "avg_anchor_pose_score_gain": summary.get("avg_anchor_pose_score_gain", 0.0),
        "shadow_recovery_starts": summary.get("shadow_recovery_starts", 0),
        "shadow_recovery_commits": summary.get("shadow_recovery_commits", 0),
        "shadow_recovery_rejects": summary.get("shadow_recovery_rejects", 0),
        "shadow_recovery_frames": summary.get("shadow_recovery_frames", 0),
        "avg_shadow_recovery_geo_gain": summary.get("avg_shadow_recovery_geo_gain", 0.0),
        "avg_shadow_recovery_conf_delta": summary.get("avg_shadow_recovery_conf_delta", 0.0),
        "avg_shadow_recovery_frames": summary.get("avg_shadow_recovery_frames", 0.0),
        "recovery_success_rate": summary.get("recovery_success_rate", 0.0),
        "avg_best_similarity": summary.get("avg_best_similarity", 0.0),
        "avg_best_gap": summary.get("avg_best_gap", 0.0),
        "avg_best_sequence_similarity": summary.get("avg_best_sequence_similarity", 0.0),
        "avg_verified_similarity": summary.get("avg_verified_similarity", 0.0),
        "avg_verified_gap": summary.get("avg_verified_gap", 0.0),
        "avg_verified_sequence_similarity": summary.get("avg_verified_sequence_similarity", 0.0),
        "memory_override": memory_override,
    }
    if baseline is not None and trial_name != baseline["trial_name"]:
        row["delta_ate_vs_baseline"] = summary["avg_ate"] - baseline["avg_ate"]
        row["delta_rpe_trans_vs_baseline"] = summary["avg_rpe_trans"] - baseline["avg_rpe_trans"]
        row["delta_rpe_rot_vs_baseline"] = summary["avg_rpe_rot"] - baseline["avg_rpe_rot"]
        row["delta_fps_vs_baseline"] = summary["avg_fps"] - baseline["avg_fps"]
    return row


def _release_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tum_relpose_sweep_224.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from hmr3d_memory.eval_relpose import evaluate_relpose

    config = _load_json(args.config)
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    common_memory = config.get("memory", {})
    common_eval = {
        "weights": config["weights"],
        "dataset_name": config["dataset"],
        "size": int(config.get("size", 512)),
        "revisit": int(config.get("revisit", 1)),
        "freeze_state": bool(config.get("freeze_state", False)),
        "solve_pose": bool(config.get("solve_pose", False)),
        "seq_list": config.get("seq_list"),
        "max_seqs": config.get("max_seqs"),
        "device": config.get("device", "cuda"),
        "seed": int(config.get("seed", 0)),
        "deterministic": bool(config.get("deterministic", False)),
    }

    trial_summaries: List[Dict[str, Any]] = []
    for trial in config["trials"]:
        trial_name = trial["name"]
        mode = trial["mode"]
        trial_dir = output_root / trial_name
        summary_path = trial_dir / "summary.json"

        if args.resume and summary_path.exists():
            summary = _load_json(str(summary_path))
        else:
            summary = evaluate_relpose(
                output_dir=str(trial_dir),
                mode=mode,
                memory_config=_merge_memory(common_memory, trial.get("memory")),
                **common_eval,
            )
            _release_cuda_cache()
        summary["trial_name"] = trial_name
        summary["trial_memory_override"] = trial.get("memory", {})
        trial_summaries.append(summary)

    baseline_name = config.get("baseline_trial")
    baseline_summary = None
    if baseline_name is not None:
        for summary in trial_summaries:
            if summary["trial_name"] == baseline_name:
                baseline_summary = summary
                break

    rows = [
        _build_row(
            summary,
            trial_name=summary["trial_name"],
            memory_override=summary.get("trial_memory_override", {}),
            baseline=baseline_summary,
        )
        for summary in trial_summaries
    ]

    baseline_rows = [row for row in rows if row["trial_name"] == baseline_name]
    candidate_rows = [row for row in rows if row["trial_name"] != baseline_name]
    candidate_rows.sort(key=lambda row: (row["avg_ate"], row["avg_rpe_trans"], -row["avg_fps"]))
    leaderboard = {
        "dataset": config["dataset"],
        "weights": config["weights"],
        "size": common_eval["size"],
        "device": common_eval["device"],
        "seq_list": common_eval["seq_list"],
        "seed": common_eval["seed"],
        "deterministic": common_eval["deterministic"],
        "baseline_trial": baseline_name,
        "rows": baseline_rows + candidate_rows,
    }
    with open(output_root / "leaderboard.json", "w", encoding="utf-8") as handle:
        json.dump(leaderboard, handle, indent=2)


if __name__ == "__main__":
    main()
