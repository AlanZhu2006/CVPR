#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hmr3d_memory.config import MemoryConfig
from hmr3d_memory.mem3r_probe import Mem3RLikeRuntime
from hmr3d_memory.runtime import configure_reproducibility
from hmr3d_memory.ttt3r_env import bootstrap_ttt3r_imports


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, partial):
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="checkpoints/mem3r_scaffold_init.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_reproducibility(seed=args.seed, deterministic=True)
    bootstrap_ttt3r_imports(args.weights)
    from dust3r.model import ARCroco3DStereo

    model = ARCroco3DStereo.from_pretrained(args.weights).cpu()
    cfg = MemoryConfig()
    runtime = Mem3RLikeRuntime.from_model(model, cfg, enable_state_gate=True)
    base_model_config = _json_safe(model.config.to_dict())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_weights": args.weights,
        "seed": args.seed,
        "base_model_config": base_model_config,
        "memory_config": cfg.to_dict(),
        "pose_memory": runtime.pose_memory.state_dict(),
        "state_gate": runtime.state_gate.state_dict() if runtime.state_gate is not None else None,
        "initial_pose_token": runtime.initial_pose_token.cpu(),
        "notes": {
            "purpose": "Initialization scaffold for local Mem3R-like runtime reproduction before training.",
            "status": "Adapter/runtime scaffold only. Not an official Mem3R checkpoint.",
        },
    }
    torch.save(payload, output_path)

    sidecar_path = output_path.with_suffix(".json")
    with open(sidecar_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_weights": args.weights,
                "seed": args.seed,
                "base_model_config": base_model_config,
                "memory_config": cfg.to_dict(),
                "output": str(output_path),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
