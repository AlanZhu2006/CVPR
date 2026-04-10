# Mem3R HPC Handoff

This repo now contains a local `Mem3R` reproduction scaffold on top of `TTT3R`.
It stops before training.

## Status

- Paper: `Mem3R: Streaming 3D Reconstruction with Hybrid Memory via Test-Time Training`
- Source paper: https://arxiv.org/pdf/2604.07279
- Project page: https://lck666666.github.io/Mem3R/
- Official repo: https://github.com/lck666666/Mem3R
- As of `2026-04-10`, the official repo does not provide the training code or checkpoints needed for direct reproduction.

## What Is Implemented

- `mem3r_pose_probe`
  - A fast-weight implicit pose-memory path that replaces `TTT3R` local pose retrieval during recurrent inference.
- `mem3r_like_runtime`
  - The same fast-weight pose memory plus a channel-wise explicit state gate.
- Deterministic relpose evaluation for local sanity checks.
- Initialization export for moving the scaffold to HPC before training.

Core files:

- [mem3r_probe.py](/home/pengyue/Codespace/HMR3D/hmr3d_memory/mem3r_probe.py)
- [adapter.py](/home/pengyue/Codespace/HMR3D/hmr3d_memory/adapter.py)
- [config.py](/home/pengyue/Codespace/HMR3D/hmr3d_memory/config.py)
- [export_mem3r_scaffold.py](/home/pengyue/Codespace/HMR3D/scripts/export_mem3r_scaffold.py)

## Artifacts To Move To HPC

- Base weights: [cut3r_512_dpt_4_64.pth](/home/pengyue/Codespace/HMR3D/checkpoints/cut3r_512_dpt_4_64.pth)
- Mem3R scaffold init: [mem3r_scaffold_init.pt](/home/pengyue/Codespace/HMR3D/checkpoints/mem3r_scaffold_init.pt)
- Mem3R scaffold manifest: [mem3r_scaffold_init.json](/home/pengyue/Codespace/HMR3D/checkpoints/mem3r_scaffold_init.json)
- Static eval config: [tum_relpose_mem3r_runtime_static.json](/home/pengyue/Codespace/HMR3D/configs/tum_relpose_mem3r_runtime_static.json)
- Reproduction notes: [mem3r_repro_notes.md](/home/pengyue/Codespace/HMR3D/reports/generated/mem3r_repro_notes.md)

## Local Sanity Results

Deterministic single-sequence `TUM sitting_static` at `224`:

- Baseline result: [leaderboard.json](/home/pengyue/Codespace/HMR3D/reports/generated/tum_relpose_mem3r_runtime_static/leaderboard.json)
- `ttt3r_baseline`: `ATE 0.030135`, `RPE_trans 0.011343`, `RPE_rot 2.118125`
- `mem3r_pose_probe`: `ATE 0.032776`, `RPE_trans 0.010761`, `RPE_rot 0.629563`
- `mem3r_like_runtime`: `ATE 0.031929`, `RPE_trans 0.011505`, `RPE_rot 0.639112`

Interpretation:

- The scaffold is runnable and integrated into the recurrent inference path.
- It is not yet a performance reproduction.
- The missing step is end-to-end training on HPC.

## Suggested HPC Next Step

1. Start from [mem3r_scaffold_init.pt](/home/pengyue/Codespace/HMR3D/checkpoints/mem3r_scaffold_init.pt) and [cut3r_512_dpt_4_64.pth](/home/pengyue/Codespace/HMR3D/checkpoints/cut3r_512_dpt_4_64.pth).
2. Keep the current local eval config as the first regression gate: [tum_relpose_mem3r_runtime_static.json](/home/pengyue/Codespace/HMR3D/configs/tum_relpose_mem3r_runtime_static.json).
3. Train the fast-weight pose memory and channel-wise state gate on HPC.
4. Re-run the deterministic relpose eval before expanding to larger benchmarks.

## Not Implemented Yet

- Official Mem3R training loop
- Official Mem3R loss recipe
- Full backbone-level training integration into upstream `TTT3R`
- Multi-dataset reproduction beyond the current local sanity checks
