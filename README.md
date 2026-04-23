# CVPR Workspace: cuVSLAM + HMR3D + Streaming Gaussian

这个工作区的主线是：

> 用 `cuVSLAM` 做低延迟 tracking，用 `HMR3D` 管长期记忆生命周期，用 Gaussian submap 做局部可渲染表示。

它现在已经是一个能跑通的系统原型，但还不是 StreamGS / GS-SLAM / GS-SDF 那种高画质 Gaussian 重建器。当前最重要的判断是：

> 现有 Gaussian 画质差，不是 WebUI 问题，而是几何和 Gaussian 参数生成后端还太弱。`cuVSLAM` 能告诉我们相机在哪里，但它的 visual landmarks 不是 dense renderable geometry。

更详细的阶段性备忘在：

- [PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md](/home/nyu/Codespace/CVPR/PROJECT_STATUS_AND_REALTIME_GAUSSIAN_PLAN.md)

## Current Status

### 已经跑通

- `cuVSLAM -> HMR3D` adapter
- KITTI 06 offline replay
- HMR3D memory lifecycle:
  - active submap
  - archive
  - retrieve
  - recover
  - merge
  - hierarchical bank
- Incremental Gaussian submap:
  - active Gaussian update
  - archived Gaussian handle
  - recovered Gaussian warm start
  - PLY / NPZ export
- CPU surfel-style renderer
- experimental `gsplat` backend adapter
- LingBot-Map export path:
  - `LingBot predictions -> depth / pointmap / confidence / intrinsic / extrinsic`
  - `LingBot predictions -> Gaussian handle`
- Web timeline demos with draggable playback
- SSH tunnel based browser viewing through port `19090`

### 当前最真实的问题

1. Gaussian render 画质很差。

   `realtime_budget` 版能接近 1 FPS，但图像非常糊。它是低预算预览，不是高质量 Gaussian rendering。

2. 质量版很慢。

   `surfel_denseviewer` 能比 realtime 版好一些，但只有约 `0.2 FPS`。

3. `cuVSLAM` 地图不是 renderable dense map。

   `cuVSLAM` 的 landmark 是为了 tracking / localization，不是为了覆盖每个可见表面。它缺少 dense depth、normal、surface continuity、opacity、SH color 等渲染所需信息。

4. LingBot-Map 在当前小设备上容易 OOM。

   本地实测：
   - 2 frame 可以用于实验
   - 6 frame 已经会接近或触发内存危险
   - 120 frame 会被 guard 停掉

5. LingBot-Map 不是最终 Gaussian renderer。

   它更适合作为 dense geometry front-end / teacher，而不是直接替代 HMR3D 或直接成为实时主循环。

6. Mem3R / LingBot / StreamGS 这类 foundation model 不能假设能在 Orin Nano 上原版实时跑。

   它们论文里的 real-time 通常依赖更强 GPU、更优 CUDA kernel、更低 overhead 的推理环境。机器人本机更现实的角色是跑 tracking；dense geometry / Gaussian 后端应先做异步或远端 GPU。

## Repository Layout

- [cuVSLAM](/home/nyu/Codespace/CVPR/cuVSLAM)
  - NVIDIA cuVSLAM experiment tree
  - KITTI example data and trajectory outputs
  - Jetson Python env: [cuVSLAM/.venv-jetson](/home/nyu/Codespace/CVPR/cuVSLAM/.venv-jetson)

- [HMR3D](/home/nyu/Codespace/CVPR/HMR3D)
  - current main runtime and experiments
  - core code: [HMR3D/nuc/src/nuc_runtime](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime)
  - scripts: [HMR3D/nuc/scripts](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts)
  - configs: [HMR3D/nuc/configs](/home/nyu/Codespace/CVPR/HMR3D/nuc/configs)
  - historical outputs: [HMR3D/nuc_output](/home/nyu/Codespace/CVPR/HMR3D/nuc_output)

- [nuc_output](/home/nyu/Codespace/CVPR/nuc_output)
  - newer local benchmark outputs and generated WebUI pages

- [third_party_research](/home/nyu/Codespace/CVPR/third_party_research)
  - LingBot-Map
  - Gaussian-SLAM
  - GSFusion
  - FAST-LIVO / FAST-LIVO2 references

## Core Code

Tracking / adapter:

- [cuvslam_adapter.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/cuvslam_adapter.py)

Memory lifecycle:

- [memory_router.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/memory_router.py)
- [policies.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/policies.py)
- [models.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/models.py)

Gaussian:

- [gaussian_builder.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_builder.py)
- [gaussian_renderer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/gaussian_renderer.py)
- [local_tsdf.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/src/nuc_runtime/local_tsdf.py)

Key scripts:

- [run_gaussian_render_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_gaussian_render_benchmark.py)
- [run_cuvslam_kitti_memory.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_kitti_memory.py)
- [run_cuvslam_lingbot_recon.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_cuvslam_lingbot_recon.py)
- [prepare_lingbot_dense_job.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_lingbot_dense_job.py)
- [run_lingbot_dense_from_manifest.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_from_manifest.py)
- [run_lingbot_dense_remote.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_remote.py)
- [export_lingbot_dense_geometry.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/export_lingbot_dense_geometry.py)
- [run_lingbot_gaussian_init.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py)
- [run_lingbot_gaussian_benchmark.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_benchmark.py)
- [generate_gaussian_timeline_demo.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/generate_gaussian_timeline_demo.py)
- [generate_quad_run_compare_viewer.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/generate_quad_run_compare_viewer.py)
- [open_gaussian_web.sh](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh)

## Architecture

Current intended split:

```text
camera / KITTI frames
  -> cuVSLAM
      pose, tracking, keyframe rhythm
  -> HMR3D MemoryRouter
      active/archive/retrieve/recover/merge
  -> Gaussian Builder
      active Gaussian submap, archived Gaussian handles
  -> Renderer / WebUI
      current-view visualization and debugging
```

Future intended split:

```text
camera / KITTI frames
  -> local cuVSLAM
      low-latency tracking, pose backbone
  -> async dense geometry front-end
      LingBot / Mem3R / TTT3R / DUSt3R-style pointmaps
  -> HMR3D memory-native submaps
      active/archive/recover lifecycle
  -> Gaussian backend
      standard 3DGS params + gsplat refinement
  -> robot UI / viewer
      delayed high-quality map render
```

The important distinction:

- `cuVSLAM` is the pose/tracking backbone.
- `dense geometry` should come from a reconstruction front-end.
- `Gaussian` should become a learned or optimized renderable representation.
- `HMR3D` should keep the memory lifecycle identity of the project.

## Current Metrics

Representative runs:

| Run | Purpose | PSNR | SSIM | FPS | Max Points | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `realtime_budget` | playable WebUI preview | 13.1584 | 0.45105 | 0.986 | 7200 | looks very blurry; near-1-FPS budget |
| `surfel_denseviewer` | current quality-ish ceiling | 14.6421 | 0.45309 | 0.198 | 150144 | better coverage, far too slow |
| `v15_fast_stable` | fast/stable experimental branch | 11.3769 | 0.38815 | 1.023 | 3057 | stable but visually weak |
| `LingBot balanced` | 2-frame LingBot Gaussian init test | 10.3089 | 0.22699 | 1.768 | 1100 | balanced frame budget, still weak |

Interpretation:

- The best current image quality is still far below StreamGS-style output.
- Adding points helps but makes rendering too slow.
- Optimizing only point budgets/radius/opacity is not enough.
- The next quality jump needs better dense geometry and real Gaussian optimization.

## Demos

The current web server usually serves [nuc_output](/home/nyu/Codespace/CVPR/nuc_output) on port `19090`.

Tunnel from your local machine:

```bash
ssh -L 19090:127.0.0.1:19090 nyu@nuc-6913-frp
```

Open:

- realtime timeline:
  `http://127.0.0.1:19090/gaussian_timeline_demo_realtime_budget.html?v=2`
- quality-ish timeline:
  `http://127.0.0.1:19090/gaussian_timeline_demo_surfel_denseviewer.html?v=1`
- LingBot old vs balanced compare:
  `http://127.0.0.1:19090/gaussian_quad_compare_lingbot_balanced_v9_v12_nocache.html`

If the server is not running:

```bash
/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/open_gaussian_web.sh \
  /home/nyu/Codespace/CVPR/nuc_output/gaussian_timeline_demo_realtime_budget.html \
  19090
```

## LingBot-Map Status

LingBot-Map is useful, but the role should be clear.

What works:

- `LingBot predictions -> npz/json export`
- `LingBot depth + intrinsic + cuVSLAM poses -> Gaussian handle`
- 2-frame benchmark and compare viewer
- balanced Gaussian anchor selection across LingBot frames

Important outputs:

- [kitti06_cuvslam_lingbot_cmp](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_cmp)
- [kitti06_cuvslam_lingbot_gaussian_cmp_balanced](/home/nyu/Codespace/CVPR/nuc_output/kitti06_cuvslam_lingbot_gaussian_cmp_balanced)
- [kitti06_render_benchmark_lingbot_gaussian_cmp_balanced](/home/nyu/Codespace/CVPR/nuc_output/kitti06_render_benchmark_lingbot_gaussian_cmp_balanced)

Known problems:

- On-device LingBot inference is memory-heavy.
- 6-frame local run already approaches OOM.
- 120-frame local run had to be stopped by memory guard.
- LingBot world coordinates need careful alignment to cuVSLAM coordinates.
- The first aligned version looked good on frame 0 but failed on frame 25 because most Gaussian budget was consumed by frame 0.
- Balanced anchor selection fixed the 830/270 frame imbalance to roughly 550/550, improving frame 25 but lowering frame 0.

Current judgment:

> LingBot-Map is a promising dense geometry teacher/front-end, not a drop-in realtime loop on this hardware.

## Mem3R / TTT3R / StreamGS Judgment

These methods are relevant, but not all in the same role.

- `LingBot-Map`
  - best short-term candidate because we already have an adapter
  - useful as dense geometry front-end
  - currently too heavy locally

- `Mem3R`
  - conceptually very close to our desired memory-native geometry front-end
  - hybrid memory, streaming reconstruction, fixed-size state
  - likely still heavy for Orin Nano unless optimized/quantized

- `TTT3R`
  - useful as test-time adaptation / streaming geometry improvement
  - likely more of an algorithmic component than a full replacement for HMR3D

- `StreamGS`
  - closest to the render quality target
  - learned Gaussian generation and merging, not just geometry
  - should be treated as high-quality teacher/baseline first
  - original-style real-time does not automatically mean real-time on our small local device

Current engineering assumption:

> Local robot should run `cuVSLAM` and memory bookkeeping. Dense geometry / learned Gaussian generation should initially run asynchronously on a remote GPU or a larger onboard GPU.

## Why cuVSLAM Alone Is Not Enough

`cuVSLAM` is very useful, but for a different objective.

It provides:

- low-latency pose
- tracking stability
- keyframe rhythm
- visual landmarks for localization

Gaussian rendering needs:

- dense depth or pointmaps
- surface normals
- visibility
- scale/rotation/opacity
- color or SH coefficients
- multi-view photometric consistency

Visual landmarks are not enough because they are sparse and selected for tracking, not for covering visible surfaces. They are good anchors for localization, but poor raw material for high-quality rendering.

## Next Plan

### Phase 1: Stop Over-optimizing the Current Heuristic Renderer

Current CPU/surfel renderer is useful for debugging memory lifecycle, but not enough for high-quality output.

Do:

- keep it as a diagnostic viewer
- keep timeline demos for presentations and debugging
- avoid spending too much time tuning radius/opacity/budget only

Success condition:

- viewer remains usable
- metrics remain reproducible
- no claim that this is StreamGS-quality rendering

### Phase 2: Remote GPU Dense Geometry Front-End

Build an async dense geometry service.

Input:

- selected keyframe window
- image paths or compressed frames
- optional cuVSLAM poses

Output:

- depth map
- point map
- confidence
- intrinsic/extrinsic
- optional normals
- metadata for alignment

Initial candidate:

- LingBot-Map, because adapter exists.

Current first-pass implementation:

- [prepare_lingbot_dense_job.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/prepare_lingbot_dense_job.py) selects cuVSLAM keyframes and writes a self-contained job manifest.
- [run_lingbot_dense_remote.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_remote.py) uploads the job to a remote GPU host, runs LingBot, and downloads predictions.
- [run_lingbot_dense_from_manifest.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_dense_from_manifest.py) is the remote-side runner.
- [export_lingbot_dense_geometry.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/export_lingbot_dense_geometry.py) converts LingBot `depth/depth_conf/intrinsic` plus cuVSLAM poses into a normalized dense geometry `.npz`.
- [run_lingbot_gaussian_init.py](/home/nyu/Codespace/CVPR/HMR3D/nuc/scripts/run_lingbot_gaussian_init.py) can now initialize Gaussian handles directly from the dense geometry `.npz`.

Next candidate:

- Mem3R, because it is a better conceptual match for long streaming memory.

Success condition:

- can process more than 2 frames without local OOM
- outputs align to cuVSLAM coordinates
- can inject dense geometry into active HMR3D submap

### Phase 3: Standard Gaussian Representation

Move from current surfel-like state to more standard 3DGS state:

- `xyz`
- `rotation`
- `scale`
- `opacity`
- `SH/color`
- source frame / confidence / memory provenance

Use dense geometry to initialize `xyz`, normals, and scale.

Success condition:

- Gaussian data can be consumed by `gsplat`
- output quality is meaningfully above current surfel renderer
- data can still be archived/recovered by HMR3D

### Phase 4: Differentiable Local Refinement

Use `gsplat` or a similar backend for local refinement.

Optimize:

- color / SH
- opacity
- scale
- rotation
- limited xyz correction

Use losses:

- RGB
- SSIM or perceptual proxy
- depth consistency
- opacity sparsity
- temporal consistency

Success condition:

- active submap improves after background optimization
- frontend tracking is not blocked
- map quality improves without losing HMR3D lifecycle behavior

### Phase 5: Memory-Native Gaussian System

Once the Gaussian backend is stronger, make HMR3D matter again at representation level.

Add:

- active Gaussian lifecycle policy
- archive compression
- coarse/full Gaussian bank
- recover alignment
- cross-submap merge
- confidence-aware memory routing

Success condition:

- revisiting an old region recovers useful Gaussian state
- recovered Gaussian improves render or update speed
- memory lifecycle gives measurable benefit beyond a single active model

## Immediate TODO

Most useful next tasks:

1. Use the new remote LingBot dense job path on a real GPU host.
2. Run LingBot on a longer KITTI 06 window without local OOM.
3. Validate the dense geometry schema on more than 2 keyframes.
4. Improve robust alignment and scale between LingBot depth and cuVSLAM coordinates.
5. Replace current heuristic Gaussian init with dense-geometry-derived Gaussian init.
6. Add a real `gsplat` local optimization loop.
7. Benchmark against StreamGS-style output as a teacher/baseline.
8. Keep HMR3D memory lifecycle as the research identity, not the renderer itself.

## Useful Commands

Run current realtime-ish benchmark:

```bash
source HMR3D/nuc/scripts/use_jetson_gpu_backend.sh
cuVSLAM/.venv-jetson/bin/python HMR3D/nuc/scripts/run_gaussian_render_benchmark.py \
  --sequence-path cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --config HMR3D/nuc/configs/kitti06_v4_realtime_budget.yaml \
  --save-images
```

Generate a timeline demo:

```bash
python HMR3D/nuc/scripts/generate_gaussian_timeline_demo.py \
  --run-dir nuc_output/hmr3d_nuc_output/kitti06_render_benchmark_realtime_budget \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --output-html nuc_output/gaussian_timeline_demo_realtime_budget.html \
  --title "KITTI 06 Gaussian Walkthrough" \
  --label "realtime budget" \
  --root-relative
```

Open web viewer:

```bash
HMR3D/nuc/scripts/open_gaussian_web.sh \
  nuc_output/gaussian_timeline_demo_realtime_budget.html \
  19090
```

Prepare a LingBot dense job locally:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/prepare_lingbot_dense_job.py \
  --sequence-path cuVSLAM/examples/kitti/dataset/sequences/06 \
  --trajectory-path cuVSLAM/examples/kitti/dataset/sequences/06/trajectory_tum.txt \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml \
  --output-dir nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --window-keyframes 8
```

Run that job on a remote GPU:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_dense_remote.py \
  --remote USER@GPU_HOST \
  --job-dir nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --remote-root /tmp/lingbot_dense_jobs \
  --remote-repo-root /path/to/CVPR \
  --model-path /path/to/lingbot-map.pt \
  --remote-python /path/to/python \
  --lingbot-map-root /path/to/lingbot-map
```

Export normalized dense geometry:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/export_lingbot_dense_geometry.py \
  --predictions-npz nuc_output/lingbot_dense_jobs/kitti06_win8/lingbot_output/lingbot_predictions.npz \
  --summary-json nuc_output/lingbot_dense_jobs/kitti06_win8/lingbot_output/lingbot_summary.json \
  --image-root nuc_output/lingbot_dense_jobs/kitti06_win8 \
  --output-dir nuc_output/lingbot_dense_jobs/kitti06_win8/dense_geometry \
  --stride 4 \
  --min-conf 1.0
```

Run LingBot Gaussian init:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_gaussian_init.py \
  --predictions-npz nuc_output/kitti06_cuvslam_lingbot_cmp/lingbot_predictions.npz \
  --summary-json nuc_output/kitti06_cuvslam_lingbot_cmp/lingbot_summary.json \
  --output-dir nuc_output/kitti06_cuvslam_lingbot_gaussian_cmp_balanced \
  --submap-id 9025 \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml
```

Run Gaussian init from normalized dense geometry:

```bash
PYTHONPATH=/home/nyu/Codespace/CVPR/HMR3D/nuc/src \
python HMR3D/nuc/scripts/run_lingbot_gaussian_init.py \
  --dense-geometry-npz nuc_output/lingbot_dense_jobs/kitti06_win8/dense_geometry/lingbot_dense_geometry.npz \
  --output-dir nuc_output/kitti06_lingbot_dense_gaussian \
  --submap-id 9100 \
  --config HMR3D/nuc/configs/kitti06_v9_structured_init.yaml
```

## Working Conclusion

The project is not blocked because the WebUI looks bad. The WebUI is showing the truth.

Current system has proven:

- cuVSLAM can feed HMR3D.
- HMR3D can manage Gaussian submap lifecycle.
- Gaussian handles can be archived, recovered, and visualized.
- LingBot-style dense geometry can be exported and injected.

Current system has not yet solved:

- high-quality dense geometry
- standard 3DGS parameter generation
- real differentiable Gaussian refinement
- embedded real-time foundation-model inference

The next real step is therefore:

> Keep cuVSLAM + HMR3D as the system backbone, and replace the weak heuristic Gaussian front-end with an async dense-geometry / learned-Gaussian backend.
