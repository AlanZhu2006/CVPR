#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.eval_relpose import evaluate_relpose
from hmr3d_memory.eval_video_depth import evaluate_video_depth
from hmr3d_memory.summarize import compare_run_group


def _load_config(config_path: str | None) -> Dict[str, object]:
    if config_path is None:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/longseq_eval.json")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--output-root", default="reports/generated/longseq_eval")
    parser.add_argument("--modes", nargs="+", default=None)
    parser.add_argument("--relpose-datasets", nargs="+", default=None)
    parser.add_argument("--video-depth-datasets", nargs="+", default=None)
    parser.add_argument("--task", choices=["all", "relpose", "video_depth"], default="all")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--revisit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    weights = args.weights or config.get("weights")
    if not weights:
        raise ValueError("Model weights were not provided.")

    modes = args.modes or config.get("modes", ["cut3r", "ttt3r", "hmr_archive_only", "hmr_full"])
    relpose_datasets = args.relpose_datasets or config.get("relpose_datasets", [])
    video_depth_datasets = args.video_depth_datasets or config.get("video_depth_datasets", [])
    memory_config = MemoryConfig.from_dict(config.get("memory"))
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    deterministic = args.deterministic or bool(config.get("deterministic", False))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    groups: Dict[tuple[str, str], List[Dict[str, object]]] = {}

    if args.task in {"all", "relpose"}:
        for dataset in relpose_datasets:
            for mode in modes:
                summary = evaluate_relpose(
                    weights=weights,
                    dataset_name=dataset,
                    output_dir=str(output_root / "relpose" / dataset / mode),
                    mode=mode,
                    memory_config=memory_config,
                    size=args.size,
                    revisit=args.revisit,
                    max_seqs=args.max_seqs,
                    device=args.device,
                    seed=seed,
                    deterministic=deterministic,
                )
                groups.setdefault(("relpose", dataset), []).append(summary)

    if args.task in {"all", "video_depth"}:
        for dataset in video_depth_datasets:
            for mode in modes:
                summary = evaluate_video_depth(
                    weights=weights,
                    dataset_name=dataset,
                    output_dir=str(output_root / "video_depth" / dataset / mode),
                    mode=mode,
                    memory_config=memory_config,
                    size=args.size,
                    revisit=args.revisit,
                    max_seqs=args.max_seqs,
                    device=args.device,
                    seed=seed,
                    deterministic=deterministic,
                )
                groups.setdefault(("video_depth", dataset), []).append(summary)

    comparison_rows = []
    for (_task, _dataset), summaries in groups.items():
        comparison_rows.append(compare_run_group(summaries))

    report = {
        "weights": weights,
        "task": args.task,
        "comparisons": comparison_rows,
        "memory_config": memory_config.to_dict(),
        "seed": seed,
        "deterministic": deterministic,
    }
    with open(output_root / "comparison.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
