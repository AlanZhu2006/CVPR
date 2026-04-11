# HMR3D

HMR3D is the orchestration repo for building and evaluating a memory-lifecycle extension on top of [TTT3R](https://github.com/Inception3D/TTT3R).

Current scope:

- pin `TTT3R` under `third_party/TTT3R`
- implement a state-level `archive / retrieve / recover` memory system without modifying the upstream submodule
- provide a local `Mem3R` reproduction scaffold before HPC training
- provide benchmark preparation scripts for the long-sequence datasets used by the TTT3R evaluation protocol
- provide a single evaluation entrypoint that compares `cut3r`, `ttt3r`, `hmr_archive_only`, and `hmr_full`

## Layout

- `hmr3d_memory/`: local package for the memory router, TTT3R adapter, and evaluation helpers
- `scripts/`: environment bootstrap, checkpoint download, benchmark preparation, and evaluation entrypoints
- `configs/`: default experiment manifests
- `third_party/TTT3R/`: pinned upstream codebase

## Quick Start

```bash
bash scripts/bootstrap_env.sh
bash scripts/download_weights.sh
python scripts/prepare_longseq_benchmarks.py --help
python scripts/run_longseq_eval.py --help
python scripts/run_relpose_memory_sweep.py --help
```

## Mem3R Scaffold

The repo also contains a local `Mem3R` reproduction scaffold:

- `mem3r_pose_probe`: fast-weight implicit pose memory
- `mem3r_like_runtime`: fast-weight pose memory + channel-wise explicit state gate

Useful entrypoints:

```bash
python scripts/run_relpose_memory_sweep.py --config configs/tum_relpose_mem3r_runtime_static.json
python scripts/export_mem3r_scaffold.py --weights checkpoints/cut3r_512_dpt_4_64.pth
```

HPC handoff notes are in [docs/mem3r_hpc_handoff.md](/home/pengyue/Codespace/HMR3D/docs/mem3r_hpc_handoff.md).

## Notes

- The first implementation works at the recurrent state level. It proves whether explicit memory lifecycle improves long-sequence robustness before introducing Gaussian active-map code.
- The current default `configs/longseq_eval.json` is tuned to the conservative preset that performed best across the first TUM relpose sweeps: `reset_on_archive=false`, stricter retrieval thresholds, and a redundant-state guard.
- `scripts/run_longseq_eval.py` and the sweep runner now support `seed` and `deterministic` mode. The default manifest enables deterministic evaluation because no-op memory trials were otherwise showing metric drift that looked like false gains/regressions.
- The new `retrieval_attempt_cooldown` only throttles repeated rejected retrievals. It reduces redundant retrieval checks without changing trajectory metrics in deterministic no-op trials.
- On this workstation (`RTX 2060 6GB`), long-sequence evaluation at `512` resolution still runs out of memory. Use `python scripts/run_longseq_eval.py --size 224` for local sweeps unless you move to a larger GPU.
- `ScanNet` still requires manual credentialed download. The prep script supports it, but does not bypass the dataset license flow.
- The new no-train `HMR3D` optimization direction is geometric recovery verification: retrieve candidates by descriptor, then verify them with a counterfactual rollout on the current frame. The literature note is in [docs/hmr3d_retrieval_review.md](/home/pengyue/Codespace/HMR3D/docs/hmr3d_retrieval_review.md), and the current experimental long-sequence manifest is [configs/longseq_eval_geometry_v6.json](/home/pengyue/Codespace/HMR3D/configs/longseq_eval_geometry_v6.json).
- The next experimental step is `shadow recover`: after a candidate passes current-frame verification, keep it on a short shadow branch for 2 future frames and only commit it if the local evidence stays favorable. The current manifests are [configs/tum_relpose_xyz_triple_v10_shadow_recover.json](/home/pengyue/Codespace/HMR3D/configs/tum_relpose_xyz_triple_v10_shadow_recover.json) and [configs/tum_relpose_sweep_224_v10_shadow_full7.json](/home/pengyue/Codespace/HMR3D/configs/tum_relpose_sweep_224_v10_shadow_full7.json). This path is still experimental and is not the default preset yet.
- The current follow-up experiment is `pose-anchor verification`: keep current-frame geometry verification, but add an explicit anchor-pose check for ambiguous or stale recoveries. The targeted manifest is [configs/tum_relpose_xyz_triple_v11_pose_anchor.json](/home/pengyue/Codespace/HMR3D/configs/tum_relpose_xyz_triple_v11_pose_anchor.json).
