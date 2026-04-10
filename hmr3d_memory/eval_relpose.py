from __future__ import annotations

import json
import os
import time
import gc
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import torch

from .adapter import run_sequence_with_mode
from .config import MemoryConfig
from .runtime import configure_reproducibility
from .ttt3r_env import bootstrap_ttt3r_imports
from .ttt3r_io import prepare_input, prepare_relpose_output


def _ensure_headless_plot_backend() -> None:
    os.environ["MPLBACKEND"] = "Agg"
    settings_path = Path.home() / ".evo" / "settings.json"
    if not settings_path.exists():
        return
    with open(settings_path, "r", encoding="utf-8") as handle:
        settings = json.load(handle)
    if settings.get("plot_backend") == "Agg":
        return
    settings["plot_backend"] = "Agg"
    with open(settings_path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=4, sort_keys=True)


def _resolve_seq_list(metadata: Dict[str, object], seq_list: List[str] | None) -> List[str]:
    if seq_list is not None:
        return sorted(seq_list)
    if metadata.get("full_seq", False):
        img_path = metadata["img_path"]
        return sorted([name for name in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, name))])
    return sorted(metadata.get("seq_list", []))


def evaluate_relpose(
    *,
    weights: str,
    dataset_name: str,
    output_dir: str,
    mode: str,
    memory_config: MemoryConfig,
    size: int = 512,
    revisit: int = 1,
    freeze_state: bool = False,
    solve_pose: bool = False,
    seq_list: List[str] | None = None,
    max_seqs: int | None = None,
    device: str = "cuda",
    seed: int = 0,
    deterministic: bool = False,
) -> Dict[str, object]:
    bootstrap_ttt3r_imports(weights)
    _ensure_headless_plot_backend()
    configure_reproducibility(seed=seed, deterministic=deterministic)
    from dust3r.model import ARCroco3DStereo
    from eval.relpose.metadata import dataset_metadata
    from eval.relpose.utils import (
        calculate_averages,
        eval_metrics,
        get_tum_poses,
        load_traj,
        plot_trajectory,
        process_directory,
        save_focals,
        save_intrinsics,
        save_tum_poses,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = dataset_metadata[dataset_name]
    model = ARCroco3DStereo.from_pretrained(weights).to(device)
    model.eval()

    resolved_seq_list = _resolve_seq_list(metadata, seq_list)
    if max_seqs is not None:
        resolved_seq_list = resolved_seq_list[:max_seqs]

    per_sequence = []
    for seq in resolved_seq_list:
        dir_path = metadata["dir_path_func"](metadata["img_path"], seq)
        filelist = sorted(os.path.join(dir_path, name) for name in os.listdir(dir_path))
        views = prepare_input(
            filelist,
            size=size,
            revisit=revisit,
            update=not freeze_state,
            crop=True,
        )

        seq_dir = output_root / seq
        seq_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        outputs, memory_stats = run_sequence_with_mode(
            views=views,
            model=model,
            device=device,
            mode=mode,
            memory_config=memory_config,
        )
        fps = len(filelist) / max(time.time() - start, 1e-6)
        (
            _colors,
            _pts3ds_self,
            _pts3ds_other,
            _conf_self,
            _conf_other,
            cam_dict,
            pr_poses,
        ) = prepare_relpose_output(outputs, revisit=revisit, solve_pose=solve_pose)

        pred_traj = get_tum_poses(pr_poses)
        save_tum_poses(pr_poses, str(seq_dir / "pred_traj.txt"))
        save_focals(cam_dict, str(seq_dir / "pred_focal.txt"))
        save_intrinsics(cam_dict, str(seq_dir / "pred_intrinsics.txt"))

        gt_traj_file = metadata["gt_traj_func"](metadata["img_path"], metadata.get("anno_path", None), seq)
        traj_format = metadata.get("traj_format", None)
        if dataset_name == "sintel":
            gt_traj = load_traj(gt_traj_file=gt_traj_file, stride=1)
        elif traj_format is not None:
            gt_traj = load_traj(gt_traj_file=gt_traj_file, traj_format=traj_format, stride=1)
        else:
            gt_traj = None

        if gt_traj is not None:
            ate, rpe_trans, rpe_rot = eval_metrics(
                pred_traj,
                gt_traj,
                seq=seq,
                filename=str(output_root / f"{seq}_eval_metric.txt"),
            )
            plot_trajectory(pred_traj, gt_traj, title=seq, filename=str(output_root / f"{seq}.png"))
        else:
            ate, rpe_trans, rpe_rot = 0.0, 0.0, 0.0

        memory_stats["fps"] = fps
        memory_events = memory_stats.pop("events", None)
        with open(seq_dir / "memory_stats.json", "w", encoding="utf-8") as handle:
            json.dump(memory_stats, handle, indent=2)
        if memory_events is not None:
            with open(seq_dir / "memory_events.json", "w", encoding="utf-8") as handle:
                json.dump(memory_events, handle, indent=2)

        per_sequence.append(
            {
                "sequence": seq,
                "ate": ate,
                "rpe_trans": rpe_trans,
                "rpe_rot": rpe_rot,
                "fps": fps,
                "memory": memory_stats,
            }
        )
        del views
        del outputs
        del memory_stats
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = process_directory(str(output_root))
    avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)
    retrieval_successes = sum(int(item["memory"]["retrieval_successes"]) for item in per_sequence)
    retrieval_attempts = sum(int(item["memory"]["retrieval_attempts"]) for item in per_sequence)
    retrieval_redundant_skips = sum(
        int(item["memory"].get("retrieval_redundant_skips", 0)) for item in per_sequence
    )
    retrieval_threshold_rejects = sum(
        int(item["memory"].get("retrieval_threshold_rejects", 0)) for item in per_sequence
    )
    retrieval_gap_rejects = sum(int(item["memory"].get("retrieval_gap_rejects", 0)) for item in per_sequence)
    retrieval_sequence_rejects = sum(
        int(item["memory"].get("retrieval_sequence_rejects", 0)) for item in per_sequence
    )
    geometry_verification_rollouts = sum(
        int(item["memory"].get("geometry_verification_rollouts", 0)) for item in per_sequence
    )
    geometry_verification_accepts = sum(
        int(item["memory"].get("geometry_verification_accepts", 0)) for item in per_sequence
    )
    geometry_verification_rejects = sum(
        int(item["memory"].get("geometry_verification_rejects", 0)) for item in per_sequence
    )
    archive_count = sum(int(item["memory"]["archive_count"]) for item in per_sequence)
    best_similarity_weighted_sum = sum(
        float(item["memory"].get("avg_best_similarity", 0.0)) * int(item["memory"]["retrieval_attempts"])
        for item in per_sequence
    )
    best_gap_weighted_sum = sum(
        float(item["memory"].get("avg_best_gap", 0.0)) * int(item["memory"]["retrieval_attempts"])
        for item in per_sequence
    )
    best_sequence_similarity_weighted_sum = sum(
        float(item["memory"].get("avg_best_sequence_similarity", 0.0))
        * int(item["memory"]["retrieval_attempts"])
        for item in per_sequence
    )
    verified_similarity_weighted_sum = sum(
        float(item["memory"].get("avg_verified_similarity", 0.0))
        * int(item["memory"]["retrieval_successes"])
        for item in per_sequence
    )
    verified_gap_weighted_sum = sum(
        float(item["memory"].get("avg_verified_gap", 0.0))
        * int(item["memory"]["retrieval_successes"])
        for item in per_sequence
    )
    verified_sequence_similarity_weighted_sum = sum(
        float(item["memory"].get("avg_verified_sequence_similarity", 0.0))
        * int(item["memory"]["retrieval_successes"])
        for item in per_sequence
    )
    geometry_geo_gain_weighted_sum = sum(
        float(item["memory"].get("avg_geometry_verification_geo_gain", 0.0))
        * int(item["memory"].get("geometry_verification_rollouts", 0))
        for item in per_sequence
    )
    geometry_conf_delta_weighted_sum = sum(
        float(item["memory"].get("avg_geometry_verification_conf_delta", 0.0))
        * int(item["memory"].get("geometry_verification_rollouts", 0))
        for item in per_sequence
    )
    avg_fps = sum(float(item["fps"]) for item in per_sequence) / max(len(per_sequence), 1)

    summary = {
        "task": "relpose",
        "dataset": dataset_name,
        "mode": mode,
        "weights": weights,
        "output_dir": str(output_root),
        "avg_ate": avg_ate,
        "avg_rpe_trans": avg_rpe_trans,
        "avg_rpe_rot": avg_rpe_rot,
        "avg_fps": avg_fps,
        "archive_count": archive_count,
        "retrieval_attempts": retrieval_attempts,
        "retrieval_successes": retrieval_successes,
        "retrieval_redundant_skips": retrieval_redundant_skips,
        "retrieval_threshold_rejects": retrieval_threshold_rejects,
        "retrieval_gap_rejects": retrieval_gap_rejects,
        "retrieval_sequence_rejects": retrieval_sequence_rejects,
        "geometry_verification_rollouts": geometry_verification_rollouts,
        "geometry_verification_accepts": geometry_verification_accepts,
        "geometry_verification_rejects": geometry_verification_rejects,
        "recovery_success_rate": retrieval_successes / retrieval_attempts if retrieval_attempts else 0.0,
        "avg_best_similarity": best_similarity_weighted_sum / retrieval_attempts if retrieval_attempts else 0.0,
        "avg_best_gap": best_gap_weighted_sum / retrieval_attempts if retrieval_attempts else 0.0,
        "avg_best_sequence_similarity": (
            best_sequence_similarity_weighted_sum / retrieval_attempts if retrieval_attempts else 0.0
        ),
        "avg_verified_similarity": (
            verified_similarity_weighted_sum / retrieval_successes if retrieval_successes else 0.0
        ),
        "avg_verified_gap": verified_gap_weighted_sum / retrieval_successes if retrieval_successes else 0.0,
        "avg_verified_sequence_similarity": (
            verified_sequence_similarity_weighted_sum / retrieval_successes if retrieval_successes else 0.0
        ),
        "avg_geometry_verification_geo_gain": (
            geometry_geo_gain_weighted_sum / geometry_verification_rollouts
            if geometry_verification_rollouts
            else 0.0
        ),
        "avg_geometry_verification_conf_delta": (
            geometry_conf_delta_weighted_sum / geometry_verification_rollouts
            if geometry_verification_rollouts
            else 0.0
        ),
        "sequences": per_sequence,
        "memory_config": memory_config.for_mode(mode).to_dict(),
        "seed": seed,
        "deterministic": deterministic,
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    del model
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
