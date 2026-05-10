# Final Report Alignment

Use [`reports/main.pdf`](../reports/main.pdf) as the source of truth. The older
8-page PDF is kept as [`reports/legacy_8page_report.pdf`](../reports/legacy_8page_report.pdf)
only for reference.

## Report Thesis

Mono2Sim-GS asks:

```text
Can a calibration-light monocular RGB video or robot RGB stream be turned into
a pose-aligned 3D visual scene layer that is useful as the first stage of a
real-to-sim pipeline?
```

The answer implemented here is an **edge-cloud reconstruction layer**:

- live edge side: fast RGB preview and cuVSLAM pose
- learned geometry side: LingBot pose/depth/confidence/point cloud
- backend visual side: GenWildSplat / LG-GS Gaussian asset
- inspection side: GS Console-style RGB, point cloud, trajectory, and
  RGB-vs-Gaussian comparison

## How Code Maps To Report Sections

| Report section | Repository location |
| --- | --- |
| Abstract / Introduction | `README.md`, this file |
| System positioning | `docs/PROJECT_STRUCTURE.md` |
| Related work support | `reports/final_report_latex/references.bib` |
| Data: offline 301-frame video | `HMR3D/nuc/scripts/run_lingbot_export.py`, `scripts/real2sim/*` |
| Data: live RGB stream | `HMR3D/nuc/scripts/run_cuvslam_lingbot_live_reconstruction.py` |
| System overview | `docs/RUNBOOK_GPU_VIDEO_TO_GAUSSIAN.md`, `docs/RUNBOOK_LIVE_EDGE_CAPTURE.md` |
| LingBot geometry frontend | `HMR3D/nuc/src/nuc_runtime/lingbot_adapter.py`, `HMR3D/nuc/scripts/run_lingbot_export.py` |
| Raw GenWildSplat route | `scripts/real2sim/prepare_genwildsplat_scene.py`, `scripts/real2sim/register_genwildsplat_gaussian.py` |
| Sim(3) alignment | `scripts/real2sim/align_genwildsplat_to_lingbot.py` |
| Chunk atlas | `scripts/real2sim/build_genwild_overlap_chunks.py`, `scripts/real2sim/register_genwild_chunk_atlas.py` |
| LG-GS / Route B | modified GenWildSplat checkout plus `scripts/real2sim/prepare_genwildsplat_scene.py` |
| WebUI inspection | live/playback launchers and WebUI-related scripts under `HMR3D/nuc/scripts/` |
| Experiments | `docs/EXPERIMENT_SUMMARY.md` |

## Final-Report Claims

The report claims five concrete contributions:

1. A working calibration-light edge-cloud reconstruction backbone.
2. A hybrid real-to-sim representation design.
3. LingBot-Guided Gaussian Splatting.
4. A multi-rate systems design.
5. An ablation-style experimental study.

When presenting the repo, keep these as the primary claims. Do not let older
Isaac/Nav2/TSDF experiments become the main story unless they are clearly marked
as future work or prototype infrastructure.

## Current Main Method: LG-GS

Original GenWildSplat:

```text
I_1:k -> predicted poses + predicted depth + Gaussian
```

LG-GS:

```text
I_1:k + LingBot K_i + LingBot T_i + LingBot D_i -> pose-guided Gaussian
```

Responsibility split:

```text
LingBot:      pose, depth, scale, geometric frame
GenWildSplat: color, opacity, scale/rotation, visual Gaussian attributes
```

## Report Caveats To Preserve

The final report is careful about scope. Preserve this wording:

- The project is a reconstruction and inspection layer, not a complete
  simulator.
- Global 3DGS over the whole long video is not solved.
- Display cleanup is not the same as photometric optimization.
- TSDF/mesh/navigation export is the future physical branch, not the core
  completed result.
