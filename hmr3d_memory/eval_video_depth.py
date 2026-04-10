from __future__ import annotations

import json
import os
import time
import gc
from pathlib import Path
from typing import Dict, List

import torch

from .adapter import run_sequence_with_mode
from .config import MemoryConfig
from .runtime import configure_reproducibility
from .ttt3r_env import bootstrap_ttt3r_imports
from .ttt3r_io import prepare_input, prepare_video_depth_output


def _resolve_seq_list(metadata: Dict[str, object], seq_list: List[str] | None) -> List[str]:
    if seq_list is not None:
        return sorted(seq_list)
    if metadata.get("full_seq", False):
        img_path = metadata["img_path"]
        return sorted([name for name in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, name))])
    return sorted(metadata.get("seq_list", []))


def evaluate_video_depth(
    *,
    weights: str,
    dataset_name: str,
    output_dir: str,
    mode: str,
    memory_config: MemoryConfig,
    size: int = 512,
    revisit: int = 1,
    seq_list: List[str] | None = None,
    max_seqs: int | None = None,
    device: str = "cuda",
    seed: int = 0,
    deterministic: bool = False,
) -> Dict[str, object]:
    bootstrap_ttt3r_imports(weights)
    configure_reproducibility(seed=seed, deterministic=deterministic)
    from dust3r.model import ARCroco3DStereo
    from eval.video_depth.metadata import dataset_metadata
    from eval.video_depth.eval_depth import get_args_parser, main as depth_eval_main
    from eval.video_depth.utils import save_depth_maps

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
        views = prepare_input(filelist, size=size, revisit=revisit, update=True, crop=True)
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
            pts3ds_self,
            _pts3ds_other,
            conf_self,
            _conf_other,
            _cam_dict,
            _pr_poses,
        ) = prepare_video_depth_output(outputs, revisit=revisit)
        save_depth_maps(pts3ds_self, str(seq_dir), conf_self=conf_self)

        memory_stats["fps"] = fps
        memory_events = memory_stats.pop("events", None)
        with open(seq_dir / "memory_stats.json", "w", encoding="utf-8") as handle:
            json.dump(memory_stats, handle, indent=2)
        if memory_events is not None:
            with open(seq_dir / "memory_events.json", "w", encoding="utf-8") as handle:
                json.dump(memory_events, handle, indent=2)

        per_sequence.append({"sequence": seq, "fps": fps, "memory": memory_stats})
        del views
        del outputs
        del memory_stats
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_results: Dict[str, Dict[str, float]] = {}
    parser = get_args_parser()
    for align in ("metric", "scale", "scale&shift"):
        args = parser.parse_args([])
        args.output_dir = str(output_root)
        args.eval_dataset = dataset_name
        args.align = align
        depth_eval_main(args)
        result_path = output_root / f"result_{align}.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as handle:
                metric_results[align] = json.load(handle)

    summary = {
        "task": "video_depth",
        "dataset": dataset_name,
        "mode": mode,
        "weights": weights,
        "output_dir": str(output_root),
        "avg_fps": sum(float(item["fps"]) for item in per_sequence) / max(len(per_sequence), 1),
        "archive_count": sum(int(item["memory"]["archive_count"]) for item in per_sequence),
        "retrieval_attempts": sum(int(item["memory"]["retrieval_attempts"]) for item in per_sequence),
        "retrieval_successes": sum(int(item["memory"]["retrieval_successes"]) for item in per_sequence),
        "retrieval_redundant_skips": sum(
            int(item["memory"].get("retrieval_redundant_skips", 0)) for item in per_sequence
        ),
        "retrieval_threshold_rejects": sum(
            int(item["memory"].get("retrieval_threshold_rejects", 0)) for item in per_sequence
        ),
        "retrieval_gap_rejects": sum(
            int(item["memory"].get("retrieval_gap_rejects", 0)) for item in per_sequence
        ),
        "retrieval_sequence_rejects": sum(
            int(item["memory"].get("retrieval_sequence_rejects", 0)) for item in per_sequence
        ),
        "geometry_verification_rollouts": sum(
            int(item["memory"].get("geometry_verification_rollouts", 0)) for item in per_sequence
        ),
        "geometry_verification_accepts": sum(
            int(item["memory"].get("geometry_verification_accepts", 0)) for item in per_sequence
        ),
        "geometry_verification_rejects": sum(
            int(item["memory"].get("geometry_verification_rejects", 0)) for item in per_sequence
        ),
        "metrics": metric_results,
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
